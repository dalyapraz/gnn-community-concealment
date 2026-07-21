'''
This script runs experiments on real networks using the DMoN model and evaluates the effects of DICE attacks.
It loads real network graphs, trains the DMoN model, applies DICE attacks, and computes metrics like ECS, M1, and M2.
It saves the results in a CSV file for further analysis.

Author: Dalya Manatova
'''

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import random
import torch
import argparse
import os
import itertools
import csv
import networkx as nx
from lfr_generator import precompute_allpairs_neg_sqeuclidean, precompute_node_comm_neg_sqeuclidean
import attacks
from load_real_network import load_featurized_graph,  load_real_graph
from torch_geometric.utils import from_networkx
import time


print("torch.cuda.is_available():", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")


def train_model(data, true_labels, model_name, num_features, num_layers=2, hidden=64, epochs=200, lr=0.001, dropout=0.5, regularization=1.0):
    import dmon
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_clusters = len(set(true_labels))
    # for Wiki dataset use these hyperparameters
    if model_name == "dmon":
        model = dmon.DMoN(in_channels=num_features, num_clusters=num_clusters, hidden_channels=hidden,num_layers=num_layers, gcn_skip=True, dropout=dropout, collapse_regularization=regularization).to(device)
    elif model_name == "mincut":
        model = dmon.MinCut(in_channels=num_features, num_clusters=num_clusters, hidden_channels=hidden,num_layers=num_layers,gcn_skip=False, dropout=dropout, ortho_regularization=regularization).to(device)
    elif model_name == "diffpool":
        model = dmon.DiffPool(in_channels=num_features, num_clusters=num_clusters, hidden_channels=hidden,num_layers=num_layers, gcn_skip=False, dropout=dropout, entropy_regularization=regularization).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    # model = dmon.DMoN(
    #     in_channels=num_features, 
    #     num_clusters=num_clusters, 
    #     hidden_channels=hidden,
    #     num_layers=num_layers, # default is 1 for shallow model or 2 for typical GCN
    #     dropout=dropout,  
    #     gcn_skip=True,
    #     collapse_regularization=regularization  
    # ).to(device)
    # model = dmon.DMoN(in_channels=num_features, num_clusters=num_clusters, gcn_skip=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    data = data.to(device)  # send features and edges to GPU
    for epoch in range(epochs+1):
        model.train()
        optimizer.zero_grad()
        ca, loss = model(data.x, data.edge_index)
        loss.backward()
        optimizer.step()
        # if epoch % 20 == 0:
        #     print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
    model.eval()
    data = data.to(device)  # ensure data is on same device for eval
    ca, _ = model(data.x, data.edge_index)
    pred_labels = ca.argmax(dim=1).cpu().numpy()  # bring prediction back to CPU
    return pred_labels

def main(args):
    
    results_dir = "real_graphs"
    os.makedirs(results_dir, exist_ok=True)

    realizations = args.realizations
    network_name = args.network_name
    b_percentages = args.b_percentages
    FComDICE = args.FComDICE
    model_name = args.model_name
    epochs = args.epochs
    hidden = args.hidden
    num_layers = args.num_layers
    lr = args.lr
    dropout = args.dropout
    regularization = args.regularization
    if FComDICE:
        print("Using Feature + Community DICE attack")
    else:
        print("Using Community DICE attack only")
    
    results_rows = []

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    print(f"Loading real graph")
    if args.featurized: # if use featurized graphs -> true_labels are based on consensus louvain
        folder_path = "featurized_graphs"
        base_path = "featurized_" + network_name
        G, data, true_labels = load_featurized_graph(folder_path, base_path, args.featurized_sigma_c)
        print(f"Loaded featurized graph with sigma_c={args.featurized_sigma_c}")
    else:
        G, data, true_labels = load_real_graph(name=network_name)
        if args.consensus: # use consensus labels from multiple DMoN runs before attack if features are original
            folder = "real_graphs"
            if args.consensus_labels_file is not None:
                if args.consensus_labels_file.endswith(".npy"):
                    true_labels = np.load(os.path.join(folder, args.consensus_labels_file))
                elif args.consensus_labels_file.endswith(".membership"):
                    true_labels = np.loadtxt(os.path.join(folder, args.consensus_labels_file), dtype=int)
            else:
                labels_path = os.path.join(folder, f"{network_name}_consensus_louvain.membership")
                true_labels = np.loadtxt(labels_path, dtype=int)
                #true_labels = np.load(os.path.join(folder, f"final_labels_{network_name}_consensus.npy"))
            true_labels = true_labels.astype(int)
            print("Loaded consensus", len(true_labels), "labels and", len(np.unique(true_labels)), "communities.")
    dim_features = data.x.shape[1]
    print(f"Number of nodes: {data.num_nodes}, Number of edges: {data.num_edges}, Feature dimension: {dim_features}")
    # Precompute pairwise similarities for feature-based attacks
    # _, S_nc = precompute_node_comm_neg_sqeuclidean(G, true_labels)
    if FComDICE:
        _, S_nc = precompute_node_comm_neg_sqeuclidean(G, true_labels)
    # === Save graph and membership ===
    # base_name = f"graph_{network_name}"
    # graph_file = os.path.join(results_dir, base_name + ".edgelist")
    # membership_file = os.path.join(results_dir, base_name + ".membership")
    # Save the graph as an edge list
    # nx.write_edgelist(G, graph_file, delimiter=' ', data=False)
    # Save the membership as a 1D array (each line: community ID)
    # np.savetxt(membership_file, true_labels, delimiter=' ', fmt='%d')
    # features_file = os.path.join(results_dir, base_name + ".features")
    # with open(features_file, "w", newline="") as f:
    #     writer = csv.writer(f, delimiter=' ')
    #     for node, node_data in G.nodes(data=True):
    #         # Ensure feature is NumPy array of floats
    #         x = node_data['x']
    #         if isinstance(x, torch.Tensor):
    #             x = x.detach().cpu().numpy()
    #         row = [node] + list(map(float, x))
    #         writer.writerow(row)

    # print(f"Generated and saved graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    number_of_communities = len(set(true_labels))
    print(f"Number of communities: {number_of_communities}")
    for target_label in set(true_labels):
        target_community = [nn for nn, label in enumerate(true_labels) if label == target_label]
        target_size = len(target_community)
        if target_size == 1:
            print(f"Skipping target label {target_label} because it has only one node.")
            continue
        # Compute initial ECS, M1, M2 before attack r times for averaging
        for realization in range(realizations):
            # import psutil
            # print("RSS GB:", psutil.Process(os.getpid()).memory_info().rss / 1e9)
            # if torch.cuda.is_available():
            #     print("GPU GB:", torch.cuda.memory_allocated() / 1e9)
            pred_labels = train_model(data, true_labels, model_name=model_name, num_features=dim_features, num_layers=num_layers, hidden=hidden, epochs=epochs, lr=lr, dropout=dropout, regularization=regularization)
            # print("RSS GB:", psutil.Process(os.getpid()).memory_info().rss / 1e9)
            # if torch.cuda.is_available():
            #     print("GPU GB:", torch.cuda.memory_allocated() / 1e9)
            ecs_initial = attacks.compute_ECS(true_labels, pred_labels)
            M1 = attacks.compute_M1(target_list=target_community, labels=pred_labels)
            M2 = attacks.compute_M2(target_list=target_community, labels=pred_labels)
        
            results_rows.append([
                        network_name, target_label, target_size, 0, 0, None, realization+1, ecs_initial, M1, M2, None
                    ])
        # Budget as % of intra-community edges
        G_target = G.subgraph(target_community)
        intra_edges = G_target.number_of_edges()
        for p in b_percentages:
            bb = int(np.round(p * intra_edges))
            for p_val in args.p_values:
                for realization in range(realizations):
                    # print(f"Realization {realization+1}/{realizations} | mu={mu} sigma_c={sigma_c} label={target_label} b={bb}")
                    # print(f"Realization {realization+1}/{realizations} | label={target_label} b={bb}")
                    start = time.time()
                    if args.FComDICE:
                        # print("Using Feature + Community DICE attack")
                        G_attacked = attacks.dice_cfeature_comm_attack(G.copy(), target_community, true_labels, S_nc,  bb, p=p_val, feature_mode= args.attack_feature_mode)
                    else:
                        G_attacked = attacks.dice_community_attack(G.copy(), target_community, bb, p=p_val)
                        # print(f"Performed DICE community attack with budget {bb} and p={p_val}")
                    data_attacked = from_networkx(G_attacked)
                    # print("Graph converted to PyG Data")
                    data_attacked.x = torch.stack([G_attacked.nodes[i]['x'] for i in range(len(G_attacked))])
                    # print("Memory after attack before training")
                    # print("RSS GB:", psutil.Process(os.getpid()).memory_info().rss / 1e9)
                    # if torch.cuda.is_available():
                    #     print("GPU GB:", torch.cuda.memory_allocated() / 1e9)
                    pred_labels_attacked = train_model(data_attacked, true_labels, model_name=model_name, num_features=dim_features, num_layers=num_layers, hidden=hidden, epochs=epochs, lr=lr, dropout=dropout, regularization=regularization)
                    # print("Memory after attack after training")
                    # print("RSS GB:", psutil.Process(os.getpid()).memory_info().rss / 1e9)
                    # if torch.cuda.is_available():
                    #     print("GPU GB:", torch.cuda.memory_allocated() / 1e9)
                    ecs = attacks.compute_ECS(true_labels, pred_labels_attacked)
                    M1 = attacks.compute_M1(target_list=target_community, labels=pred_labels_attacked)
                    M2 = attacks.compute_M2(target_list=target_community, labels=pred_labels_attacked)
                    elapsed_time = time.time() - start
                    results_rows.append([
                        network_name, target_label, target_size, bb, p, p_val, realization+1, ecs, M1, M2, elapsed_time
                    ])
                    # print(f"The results: {realization+1}, {ecs}, {M1}, {M2}, {elapsed_time}")
                print(f"Completed p = {p_val} and budget {p} for target label {target_label}")
        # Save results to CSV
    with open(args.outfile_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["network_name", "target_label", "target_size", "b", 'b_percentage', "p", "realization", "ECS", "M1", "M2", "elapsed_time_sec"])
        writer.writerows(results_rows)
        print(f"Results saved to {args.outfile_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DMoN + DICE real network experiments")
    parser.add_argument("--network_name", type=str, default="Wiki")
    parser.add_argument("--featurized", action=argparse.BooleanOptionalAction, default=False, help="Use featurized graph")
    parser.add_argument("--featurized_sigma_c", type=float, default=1.0, help="Sigma_c value for featurized graph")
    parser.add_argument("--consensus", action=argparse.BooleanOptionalAction, default=True, help="Use consensus labels from multiple DMoN runs before attack")
    parser.add_argument("--consensus_labels_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="dmon", choices=["dmon", "mincut", "diffpool"], help="Which model to use for training and evaluation")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--realizations", type=int, default=50)
    # parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outfile_csv", type=str, default=None)
    parser.add_argument("--b_percentages", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,  0.9, 0.95, 1])
    # parser.add_argument("--b_percentages", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    parser.add_argument("--p_values", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--FComDICE", action=argparse.BooleanOptionalAction, default=False, help="Use feature + community DICE attack")
    parser.add_argument("--attack_feature_mode", type=str, default="average_community", choices=[None, "connecting_node", "average_community"])

    args = parser.parse_args()

        # Dynamically set outfile_csv if not provided
    if args.outfile_csv is None:
        p_str = "_".join(str(p) for p in args.p_values)
        sigma_str = f"sigma{args.featurized_sigma_c}" if args.featurized else "original"
        attack_str = "fcomdice" if args.FComDICE else "dice"
        args.outfile_csv = f"{args.model_name}_{attack_str}_sigma{sigma_str}_{args.network_name}.csv"


    print(f"Results will be saved to: {args.outfile_csv}")
    main(args)
