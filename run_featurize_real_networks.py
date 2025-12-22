#!/usr/bin/env python3
"""
Run Louvain consensus clustering and featurize real networks
 - graph G: featurized_<name>.edgelist
 - consensus labels: featurized_<name>.membership
 - features: featurized_<name>_<sigma_c>.features

Usage (example):
  python run_featurize_real_networks.py --name Wiki --R 50 --tau 0.3 --sigma_c 1.0 --num_features 64 --dir featurized_graphs/

Author: Dalya Manatova with assistance from Copilot (GPT-5)
"""

from pyexpat import features
import argparse, time, os, numpy as np, torch, networkx as nx
from load_real_network import load_real_graph
from lfr_generator import generate_features_from_communities
from torch_geometric.utils import from_networkx


# ----------------------------------------
# Run consensus louvain R times → labels
# ----------------------------------------
def consensus_clustering_louvain(G, R=50, tau=0.3):
    from consensus_clustering import consensus_matrix_dense, consensus_labels
    D = consensus_matrix_dense("louvain", R=R, G=G)
    CC_labels = consensus_labels(D, tau=tau, reps=100)
    return CC_labels

# -------------------------------------
# Featurize G based on CC_labels → X
# -------------------------------------
def featurize_from_consensus_labels(G, CC_labels, num_features=64, sigma_c=1.0):
    features = generate_features_from_communities(CC_labels, mode = 'gaussian', num_features=num_features, sigma_c=sigma_c)
    # Add features to NetworkX nodes
    for i, feat in enumerate(features):
        G.nodes[i]['x'] = torch.tensor(feat, dtype=torch.float)
    # Convert to PyG Data object
    data = from_networkx(G)
    data.x = torch.stack([data.x[i] for i in range(data.num_nodes)])
    return G, data


def main():
    p = argparse.ArgumentParser(description="Featurize real networks based on consensus clustering labels")
    p.add_argument('--name', type=str, required=True, help='Name of the dataset')
    p.add_argument('--R', type=int, default=50, help='Number of consensus runs')
    p.add_argument('--tau', type=float, default=0.3, help='Threshold for consensus labels')
    p.add_argument('--sigma_c', type=float, default=1.0, help='Sigma parameter for Gaussian features')
    p.add_argument('--num_features', type=int, default=64, help='Number of features to generate')
    p.add_argument('--dir', type=str, required=True, help='Output directory')
    args = p.parse_args()

    G, data, true_labels = load_real_graph(name=args.name)
    CC_labels = consensus_clustering_louvain(G, R=args.R, tau=args.tau)
    G, data = featurize_from_consensus_labels(G, CC_labels, num_features=args.num_features, sigma_c=args.sigma_c)

    # ---- Save graph and membership -----
    base_name = f"featurized_{args.name}"
    graph_file = os.path.join(args.dir, base_name + ".edgelist")
    membership_file = os.path.join(args.dir, base_name + ".membership")
    features_file = os.path.join(args.dir, base_name + f"_{args.sigma_c}.features")

   # Save edge list (structure)
    nx.write_edgelist(G, graph_file, delimiter=' ', data=False)

    # Save consensus labels
    np.savetxt(membership_file, CC_labels, fmt='%d')

    # Save the features as torch tensors
    torch.save(data.x, features_file)
    print(f"Saved graph data to {graph_file, membership_file, features_file}")

if __name__ == "__main__":
    main()