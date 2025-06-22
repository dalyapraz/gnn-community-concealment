'''
This module implements the DICE attack on a networkx graph to obscure a specific community.
It includes functions to perform the attack, compute the M1 & M2 metrics for target nodes.

Author: Dalya Manatova
'''

  

import random
import networkx as nx
import numpy as np
from clusim.clustering import Clustering
from clusim.sim import element_sim # Element-centric similarity from A.J. Gates and YY Ahn

def dice_community_attack(G, target_comm, b, p=0.5, seed=None):
    """
    Apply DICE attack to obscure a specific community.

    Parameters:
    - G: networkx.Graph, undirected
    - target_comm: list of node IDs belonging to the target community       
    - b: budget of modifications (edges to remove and add)
    - p: fraction of budget to allocate to edge removal in the target community
    - seed: random seed
    Returns:
    - G_attacked: a modified networkx.Graph with edges perturbed
    """
    random.seed(seed)
    np.random.seed(seed)
    G = G.copy()

    target_set = set(target_comm)
    non_target_nodes = list(set(G.nodes()) - target_set)

    # Step 1: Remove b*p edges from the target community
    n_remove = np.floor(b * p).astype(int)
    intra_edges = [(u, v) for u, v in G.edges() if u in target_set and v in target_set]
    remove_edges = random.sample(intra_edges, min(n_remove, len(intra_edges)))
    G.remove_edges_from(remove_edges)

    # Step 2: Add b - n_remove inter-community edges
    add_edges = set()
    while len(add_edges) < (b - n_remove) and len(non_target_nodes) > 0:
        # Randomly select a node from the target community and a node from the non-target community
        u = random.choice(target_comm)
        v = random.choice(non_target_nodes)
        if not G.has_edge(u, v):
            add_edges.add((u, v))
    G.add_edges_from(add_edges)

    return G


def compute_M1(target_list, labels):
    """
    Compute M1 metric for the target nodes in the graph.
    
    Parameters:
    - G: networkx.Graph, undirected
    - target_list: list of node IDs for which to compute M1
    - labels: np.array of shape [n_nodes] with community labels 
    
    Returns:
    - M1: float, the M1 score
    """
    unique_comms = np.unique(labels)
    # print("Unique communities:", unique_comms)
    target_comms = [labels[node] for node in target_list] # list communities in target nodes
    # print("Target communities:", target_comms)
    # max overlap between target nodes and any community
    max_overlap = np.max(np.bincount(target_comms))
    # print("Max overlap:", max_overlap)
    # Normalize denominator (avoid divide by zero)
    denominator = max(len(unique_comms) - 1, 1) * max_overlap
    # print("Denominator:", denominator)
    # print("Target communities set:", set(target_comms))
    M1 = (len(set(target_comms)) - 1) / denominator

    return M1

def compute_M2(target_list, labels):
    """
    Compute M2 metric for the target nodes in the graph.
    
    Parameters:
    - G: networkx.Graph, undirected
    - target_list: list of node IDs for which to compute M2
    - labels: np.array of shape [n_nodes] with community labels 
    
    Returns:
    - M2: float, the M2 score
    """
    comm_blending_sizes = []
    unique_comms = np.unique(labels)
    # print("Unique communities:", unique_comms)
    target_comms = [labels[node] for node in target_list]  # list communities in target nodes
    # print("Target communities:", target_comms)
    # Count how many target nodes belong to each community
    comm_counts = np.bincount(labels, minlength=len(unique_comms))
    # print("Community counts:", comm_counts)
    target_comm_counts = np.bincount(target_comms)
    # print("Target community counts:", target_comm_counts)
    for comm, count in enumerate(target_comm_counts):
        if count > 0:
            # print(f"Community {comm} has {count} target nodes out of {comm_counts[comm]} total nodes.")
            # Calculate |C_i \ V|
            comm_blending_sizes.append(comm_counts[comm] - count)
    # print("Community blending sizes:", comm_blending_sizes)
    # Calculate M2
    total_n = len(labels)  # Total number of nodes in the graph
    const_denominator = max(total_n - len(target_list), 1)  # Avoid division by zero
    # print("Total number of nodes:", total_n)
    # print("Constant denominator:", const_denominator)
    M2 = np.sum(comm_blending_sizes) / const_denominator

    return M2

def compute_ECS(true_labels, pred_labels):
    """
    Compute Element-Centric Similarity (ECS) between true and predicted labels.

    Parameters:
    - true_labels: np.array of shape [n_nodes] with ground truth community labels
    - pred_labels: np.array of shape [n_nodes] with predicted community labels

    Returns:
    - ecs: float, the ECS score
    """
    true_clustering = Clustering().from_membership_list(true_labels)
    pred_clustering = Clustering().from_membership_list(pred_labels)
    ecs = element_sim(true_clustering, pred_clustering)
    return ecs