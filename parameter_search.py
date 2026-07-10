'''
Functions for searching for the best hyperparameters for the DiffPool and MinCut models on LFR graphs and real networks. 
The search is performed by running multiple experiments with different hyperparameter combinations and evaluating the 
performance of the models using metrics such as ECS and NMI and ratio of found clusters vs true labels. 
The results are saved to a CSV file for further analysis.
'''

from email import parser
import itertools
import csv
import time
import random
from xml.sax.handler import feature_external_ges
import numpy as np
import torch
import networkx as nx
import argparse
import os

from lfr_generator import generate_featurized_lfr_graph
from load_real_network import load_featurized_graph,  load_real_graph
import attacks
from sklearn.metrics import normalized_mutual_info_score
from torch_geometric.utils import from_networkx


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    data,
    true_labels,
    model_name,
    num_features,
    num_layers=1,
    hidden=32,
    gcn_skip=False,
    dropout=0.0,
    epochs=200,
    lr=0.005,
    ortho_regularization=1.0,
    link_pred_regularization=1.0,
    entropy_regularization=1.0,
    train_seed=None,
    verbose=False,
):
    import dmon

    if train_seed is not None:
        set_seed(train_seed)

    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    device = get_device()

    num_clusters = len(set(true_labels))

    if model_name == "mincut":
        model = dmon.MinCut(
            in_channels=num_features,
            num_clusters=num_clusters,
            hidden_channels=hidden,
            num_layers=num_layers,
            gcn_skip=gcn_skip,
            dropout=dropout,
            ortho_regularization=ortho_regularization,
        ).to(device)

    elif model_name == "diffpool":
        model = dmon.DiffPool(
            in_channels=num_features,
            num_clusters=num_clusters,
            hidden_channels=hidden,
            num_layers=num_layers,
            gcn_skip=gcn_skip,
            dropout=dropout,
            link_pred_regularization=link_pred_regularization,
            entropy_regularization=entropy_regularization,
        ).to(device)

    else:
        raise ValueError(f"Unknown model_name={model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    data = data.to(device)

    last_loss = None

    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()

        ca, loss = model(data.x, data.edge_index)

        loss.backward()
        optimizer.step()

        last_loss = loss.item()

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

    model.eval()

    with torch.no_grad():
        ca, _ = model(data.x, data.edge_index)

    pred_labels = ca.argmax(dim=1).cpu().numpy()

    return pred_labels, last_loss


def evaluate_predictions(true_labels, pred_labels):
    ecs = attacks.compute_ECS(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    n_true_clusters = len(set(true_labels))
    n_pred_clusters = len(set(pred_labels))

    cluster_ratio = n_pred_clusters / n_true_clusters

    return {
        "ECS": ecs,
        "NMI": nmi,
        "n_true_clusters": n_true_clusters,
        "n_pred_clusters": n_pred_clusters,
        "cluster_ratio": cluster_ratio,
    }


def make_param_grid():
    configs = []

    # MinCut search grid
    mincut_grid = {
        "model_name": ["mincut"],
        "num_layers": [1, 2],
        "hidden": [64],
        "gcn_skip": [False, True],
        "dropout": [0.0, 0.2, 0.5],
        "epochs": [200, 500],
        "lr": [0.001, 0.005, 0.01],
        "ortho_regularization": [0.5, 1.0],
        "link_pred_regularization": [None],
        "entropy_regularization": [None],
    }

    # DiffPool search grid
    diffpool_grid = {
        "model_name": ["diffpool"],
        "num_layers": [1, 2],
        "hidden": [64],
        "gcn_skip": [False, True],
        "dropout": [0.0, 0.2, 0.5],
        "epochs": [200, 500],
        "lr": [0.001, 0.005, 0.01],
        "ortho_regularization": [None],
        "link_pred_regularization": [1.0],
        "entropy_regularization": [0.5, 1.0],
    }

    for grid in [mincut_grid, diffpool_grid]:
        keys = list(grid.keys())
        values = [grid[k] for k in keys]

        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            configs.append(config)

    return configs

def load_dataset(args):
    if args.dataset_type == "lfr":
        print("Generating clean LFR graph...")

        G, data, true_labels = generate_featurized_lfr_graph(
            mu=args.mu,
            n=args.n,
            min_community=args.min_community,
            feature_mode="gaussian",
            sigma_c=args.sigma_c,
            seed=args.graph_seed,
        )

        dataset_label = (
            f"lfr_mu{args.mu}_sigma{args.sigma_c}_"
            f"mincomm{args.min_community}_seed{args.graph_seed}"
        )

        return G, data, np.asarray(true_labels).astype(int), dataset_label

    elif args.dataset_type == "real":
        network_name = args.network_name

        if network_name is None:
            raise ValueError("--network_name must be provided when --dataset_type real")

        print(f"Loading real graph: {network_name}")

        if args.featurized:
            folder_path = args.featurized_folder
            base_path = "featurized_" + network_name

            G, data, true_labels = load_featurized_graph(
                folder_path,
                base_path,
                args.featurized_sigma_c,
            )

            dataset_label = (
                f"real_{network_name}_featurized_"
                f"sigma{args.featurized_sigma_c}"
            )

            print(f"Loaded featurized graph with sigma_c={args.featurized_sigma_c}")

        else:
            G, data, true_labels = load_real_graph(name=network_name)

            dataset_label = f"real_{network_name}_original_features"
            folder = args.real_graph_folder
            labels_path = os.path.join(folder, f"{network_name}_consensus_louvain.membership")
            true_labels = np.loadtxt(labels_path, dtype=int)

            dataset_label = f"real_{network_name}_original_features_consensus"

            print("Loaded consensus ",len(true_labels)," labels and ", len(np.unique(true_labels))," communities.")

        true_labels = np.asarray(true_labels).astype(int)

        return G, data, true_labels, dataset_label

    else:
        raise ValueError(f"Unknown dataset_type={args.dataset_type}")

def main(args):
    # Number of repeated trainings per configuration.
    training_repeats = args.training_repeats
    G, data, true_labels, dataset_label = load_dataset(args)
    print(f"G loaded/generated with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"Number of communities: {len(set(true_labels))}")
    print(f"Feature dimension: {data.x.shape[1]}")
    print(f"Dataset label: {dataset_label}")
    outfile = args.outfile_csv 

    configs = make_param_grid()
    print(f"Total configurations: {len(configs)}")

    fieldnames = [
    "config_id",
    "dataset_type",
    "dataset_label",
    "network_name",
    "featurized_sigma_c",
    "model_name",
    "num_layers",
    "hidden",
    "gcn_skip",
    "dropout",
    "epochs",
    "lr",
    "ortho_regularization",
    "link_pred_regularization",
    "entropy_regularization",
    "mu",
    "sigma_c",
    "min_community",
    "graph_seed",
    "num_nodes",
    "num_edges",
    "feature_dim",
    "training_repeat",
    "ECS",
    "NMI",
    "n_true_clusters",
    "n_pred_clusters",
    "cluster_ratio",
    "last_loss",
    "elapsed_time_sec",
]

    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for config_id, cfg in enumerate(configs, start=1):
            print("\n" + "=" * 80)
            print(f"Config {config_id}/{len(configs)}")
            print(cfg)

            for repeat in range(training_repeats):
                train_seed = 100000 * config_id + repeat

                start = time.time()

                pred_labels, last_loss = train_model(
                    data=data,
                    true_labels=true_labels,
                    model_name=cfg["model_name"],
                    num_features=data.x.shape[1],
                    num_layers=cfg["num_layers"],
                    hidden=cfg["hidden"],
                    gcn_skip=cfg["gcn_skip"],
                    dropout=cfg["dropout"],
                    epochs=cfg["epochs"],
                    lr=cfg["lr"],
                    ortho_regularization=cfg["ortho_regularization"] if cfg["ortho_regularization"] is not None else 1.0,
                    link_pred_regularization=cfg["link_pred_regularization"] if cfg["link_pred_regularization"] is not None else 1.0,
                    entropy_regularization=cfg["entropy_regularization"] if cfg["entropy_regularization"] is not None else 1.0,
                    train_seed=train_seed,
                    verbose=False,
                )

                elapsed = time.time() - start

                metrics = evaluate_predictions(true_labels, pred_labels)

                row = {
                    "config_id": config_id,
                    "dataset_type": args.dataset_type,
                    "dataset_label": dataset_label,
                    "network_name": args.network_name,
                    "featurized_sigma_c": args.featurized_sigma_c if args.dataset_type == "real" and args.featurized else None,
                    **cfg,
                    "mu": args.mu if args.dataset_type == "lfr" else None,
                    "sigma_c": args.sigma_c if args.dataset_type == "lfr" else None,
                    "min_community": args.min_community if args.dataset_type == "lfr" else None,
                    "graph_seed": args.graph_seed if args.dataset_type == "lfr" else None,
                    "num_nodes": G.number_of_nodes(),
                    "num_edges": G.number_of_edges(),
                    "feature_dim": data.x.shape[1],
                    "training_repeat": repeat + 1,
                    **metrics,
                    "last_loss": last_loss,
                    "elapsed_time_sec": elapsed,
                }

                writer.writerow(row)
                f.flush()

                print(
                    f"repeat={repeat + 1}/{training_repeats} "
                    f"ECS={metrics['ECS']:.4f} "
                    f"NMI={metrics['NMI']:.4f} "
                    f"clusters={metrics['n_pred_clusters']}/{metrics['n_true_clusters']} "
                    f"loss={last_loss:.4f} "
                    f"time={elapsed:.1f}s"
                )

    print(f"\nSaved raw tuning results to: {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search for DiffPool and MinCut on LFR graphs")
    parser.add_argument("--training_repeats", type=int, default=3, help="Number of repeated trainings per configuration")
    parser.add_argument("--dataset_type", type=str, choices=["lfr", "real"], default="lfr", help="Type of dataset to use: 'lfr' for synthetic LFR graphs or 'real' for real-world networks")
    # LFR graph parameters
    parser.add_argument("--mu", type=float, default=0.01, help="Mixing parameter for LFR graph")
    parser.add_argument("--sigma_c", type=float, default=5.0, help="Standard deviation for Gaussian features in LFR graph")
    parser.add_argument("--min_community", type=int, default=10, help="Minimum community size for LFR graph")
    parser.add_argument("--graph_seed", type=int, default=42, help="Random seed for LFR graph generation")
    # Real network parameters
    parser.add_argument("--network_name", type=str, default=None, help="Name of the real network to load (required if dataset_type is 'real')")
    parser.add_argument("--featurized", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--featurized_sigma_c", type=float, default=5.0, help="Standard deviation for Gaussian features in featurized real network")
    parser.add_argument("--featurized_folder", type=str, default="featurized_graphs")
    parser.add_argument("--real_graph_folder", type=str, default="real_graphs")

    parser.add_argument("--outfile_csv", type=str, default=None)
    args = parser.parse_args()
    if args.outfile_csv is None:
        if args.dataset_type == "lfr":
            # Set the output CSV file name based on the provided argument
            mu_str = f"{args.mu}".replace(".", "")
            sigma_str = f"{args.sigma_c:.1g}".replace(".", "")
            min_size_str = f"{args.min_community}"
            args.outfile_csv = f"local_mincut_diffpool_hparam_search_mu{mu_str}_sigma{sigma_str}_minsize{min_size_str}.csv"
        else:
            if args.featurized:
                args.outfile_csv = f"local_mincut_diffpool_hparam_search_real_{args.network_name}_featurized_sigma{args.featurized_sigma_c:.1g}.csv"
            else:
                args.outfile_csv = f"local_mincut_diffpool_hparam_search_real_{args.network_name}_original_features.csv"
    main(args)