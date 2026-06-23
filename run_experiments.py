'''
This script runs experiments on LFR graphs using the DMoN model and evaluates the effects of DICE attacks.
It generates LFR graphs, trains the DMoN model, applies DICE attacks, and computes metrics like ECS, M1, and M2.
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
from lfr_generator import generate_featurized_lfr_graph, precompute_allpairs_neg_sqeuclidean, precompute_node_comm_neg_sqeuclidean
import attacks
from torch_geometric.utils import from_networkx
import time


print("torch.cuda.is_available():", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA")


def train_model(data, true_labels, model_name,  num_features=32, num_layers=1, hidden=32, epochs=200, lr=0.005):
    import dmon
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_clusters = len(set(true_labels))
    if model_name == "dmon":
        model = dmon.DMoN(in_channels=num_features, num_clusters=num_clusters, hidden_channels=hidden,num_layers=num_layers, gcn_skip=True).to(device)
    elif model_name == "mincut":
        model = dmon.MinCut(in_channels=num_features, num_clusters=num_clusters, hidden_channels=hidden,num_layers=num_layers,gcn_skip=False).to(device)
    elif model_name == "diffpool":
        model = dmon.DiffPool(in_channels=num_features, num_clusters=num_clusters, hidden_channels=hidden,num_layers=num_layers, gcn_skip=False).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
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
    
    results_dir = "fgraphs_dice_mincut"
    os.makedirs(results_dir, exist_ok=True)

    sigma_c_values = args.sigma_c_values
    mu_values = args.mu_values
    realizations = args.realizations
    n = args.n
    b_percentages = args.b_percentages
    min_community = args.min_community
    seed = args.seed
    FComDICE = args.FComDICE
    model_name = args.model_name
    epochs = args.epochs
    hidden = args.hidden
    num_layers = args.num_layers
    lr = args.lr
    if FComDICE:
        print("Using Feature + Community DICE attack")
    else:
        print("Using Community DICE attack only")
    results_rows = []

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    for mu in mu_values:
        for sigma_c in sigma_c_values:
            print(f"Generating graphs with mu={mu}, sigma_c={sigma_c}")
            G, data, true_labels = generate_featurized_lfr_graph(
                mu=mu, n=n, min_community=min_community,
                feature_mode='gaussian', sigma_c=sigma_c, seed=seed)
            # Precompute pairwise similarities for feature-based attacks
            # _, S = precompute_allpairs_neg_sqeuclidean(G)
            if FComDICE:
                _, S_nc = precompute_node_comm_neg_sqeuclidean(G, true_labels)
            # === Save graph and membership ===
            base_name = f"graph_n{n}_mu{mu}_sigma{sigma_c}_min_comm{min_community}"
            graph_file = os.path.join(results_dir, base_name + ".edgelist")
            membership_file = os.path.join(results_dir, base_name + ".membership")
            # Save the graph as an edge list
            nx.write_edgelist(G, graph_file, delimiter=' ', data=False)
            # Save the membership as a 1D array (each line: community ID)
            np.savetxt(membership_file, true_labels, delimiter=' ', fmt='%d')
            features_file = os.path.join(results_dir, base_name + ".features")
            with open(features_file, "w", newline="") as f:
                writer = csv.writer(f, delimiter=' ')
                for node, node_data in G.nodes(data=True):
                    # Ensure feature is NumPy array of floats
                    x = node_data['x']
                    if isinstance(x, torch.Tensor):
                        x = x.detach().cpu().numpy()
                    row = [node] + list(map(float, x))
                    writer.writerow(row)

            print(f"Generated and saved graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

            number_of_communities = len(set(true_labels))
            print(f"Number of communities: {number_of_communities}")

            for target_label in set(true_labels):
                target_community = [nn for nn, label in enumerate(true_labels) if label == target_label]
                target_size = len(target_community)

                # Baseline on the clean graph for each realization
                for realization in range(realizations):
                    pred_labels = train_model(data, true_labels, model_name, epochs=epochs, hidden=hidden, num_layers=num_layers, lr=lr)
                    ecs_initial = attacks.compute_ECS(true_labels, pred_labels)
                    M1 = attacks.compute_M1(target_list=target_community, labels=pred_labels)
                    M2 = attacks.compute_M2(target_list=target_community, labels=pred_labels)

                    results_rows.append([
                                mu, sigma_c, target_label, target_size, 0, 0, None, realization + 1, ecs_initial, M1, M2, None
                            ])
            # old -----
            # pred_labels = train_model(data, true_labels, model_name, epochs=epochs, hidden=hidden, num_layers=num_layers, lr=lr)
            # ecs_initial = attacks.compute_ECS(true_labels, pred_labels)
            # for target_label in set(true_labels):
            #     target_community = [nn for nn, label in enumerate(true_labels) if label == target_label]
            #     target_size = len(target_community)
            #     M1 = attacks.compute_M1(target_list=target_community, labels=pred_labels)
            #     M2 = attacks.compute_M2(target_list=target_community, labels=pred_labels)
                
            #     results_rows.append([
            #                     mu, sigma_c, target_label, target_size, 0, 0, None, 0, ecs_initial, M1, M2, None
            #                 ])
                
            # -------
                # Budget as % of intra-community edges
                G_target = G.subgraph(target_community)
                intra_edges = G_target.number_of_edges()
                for p in b_percentages:
                    bb = int(np.round(p * intra_edges))
                    for p_val in args.p_values:
                        for realization in range(realizations):
                            # print(f"Realization {realization+1}/{realizations} | mu={mu} sigma_c={sigma_c} label={target_label} b={bb}")
                            start = time.time()
                            # G_attacked = attacks.dice_community_attack(G.copy(), target_community, bb)
                            # G_attacked = attacks.dice_community_attack(G.copy(), target_community, bb, p=p_val)
                            # G_attacked = attacks.dicehd_community_attack(G.copy(), target_community, bb)
                            # G_attacked = attacks.dicehdcd_community_attack(G.copy(), target_community, true_labels, bb)
                            # G_attacked = attacks.dicecdhd_community_attack(G.copy(), target_community, true_labels, bb)
                            # G_attacked = attacks.dicecdhc_community_attack(G.copy(), target_community, true_labels, bb)
                            # G_attacked = attacks.dice_cfeature_node_attack(G.copy(), target_community, true_labels, S,  bb, p=p_val, feature_mode= args.attack_feature_mode)
                            if FComDICE:
                                G_attacked = attacks.dice_cfeature_comm_attack(G.copy(), target_community, true_labels, S_nc,  bb, p=p_val, feature_mode= args.attack_feature_mode)
                            else:
                                G_attacked = attacks.dice_community_attack(G.copy(), target_community, bb, p=p_val)
                            data_attacked = from_networkx(G_attacked)
                            data_attacked.x = torch.stack([G_attacked.nodes[i]['x'] for i in range(len(G_attacked))])
                            pred_labels_attacked = train_model(data_attacked, true_labels, model_name, epochs=epochs, hidden=hidden, num_layers=num_layers, lr=lr)
                            ecs = attacks.compute_ECS(true_labels, pred_labels_attacked)
                            M1 = attacks.compute_M1(target_list=target_community, labels=pred_labels_attacked)
                            M2 = attacks.compute_M2(target_list=target_community, labels=pred_labels_attacked)
                            elapsed_time = time.time() - start
                            results_rows.append([
                                mu, sigma_c, target_label, target_size, bb, p, p_val, realization+1, ecs, M1, M2, elapsed_time
                            ])
                        print(f"Completed p = {p_val} and budget {p} for target label {target_label}")
        # Save results to CSV
    with open(args.outfile_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mu", "sigma_c", "target_label", "target_size", "b", 'b_percentage', "p", "realization", "ECS", "M1", "M2", "elapsed_time_sec"])
        writer.writerows(results_rows)
        print(f"Results saved to {args.outfile_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DMoN + DICE LFR graph experiments")
    parser.add_argument("--sigma_c_values", nargs="+", type=float, default=[0.01, 0.1, 0.5, 1, 2, 5])
    parser.add_argument("--mu_values", nargs="+", type=float, default=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--realizations", type=int, default=50)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--min_community", type=int, default=60)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outfile_csv", type=str, default=None)
    parser.add_argument("--b_percentages", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,  0.9, 0.95, 1])
    # parser.add_argument("--b_percentages", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    parser.add_argument("--p_values", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--model_name", type=str, default="dmon", choices=["dmon", "mincut", "diffpool"], help="Which model to use for training and evaluation")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--FComDICE", action=argparse.BooleanOptionalAction, default=False, help="Use feature + community DICE attack")
    parser.add_argument("--attack_feature_mode", type=str, default="average_community", choices=[None, "connecting_node", "average_community"])

    args = parser.parse_args()

        # Dynamically set outfile_csv if not provided
    if args.outfile_csv is None:
        # Join mu values as a string for filename
        mu_str = "_".join(str(mu) for mu in args.mu_values)
        sigma_str = "_".join(str(sigma) for sigma in args.sigma_c_values)
        p_str = "_".join(str(p) for p in args.p_values)
        attack_str = "fcomdice" if args.FComDICE else "dice"
        # args.outfile_csv = f"dmon_dice_{mu_str}_sigma{sigma_str}_p_{p_str}_mincomm_{args.min_community}.csv"
        args.outfile_csv = (
            f"{args.model_name}_{attack_str}_{mu_str}_"
            f"sigma{sigma_str}_p_{p_str}_mincomm_{args.min_community}.csv"
        )

    print(f"Results will be saved to: {args.outfile_csv}")
    main(args)
