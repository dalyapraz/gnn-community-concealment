import networkx as nx
import numpy as np
import pandas as pd
import glob

def read_membership_file(membership_file):
    with open(membership_file, 'r') as f:
        return {i: int(line.strip()) for i, line in enumerate(f)}

def load_graph_with_membership(edgelist_file, membership_file):
    G = nx.read_edgelist(edgelist_file, nodetype=int)
    membership = read_membership_file(membership_file)
    nx.set_node_attributes(G, membership, 'community')
    return G, membership

def build_supergraph(G, membership):
    communities = set(membership.values())
    superG = nx.Graph()
    superG.add_nodes_from(communities)
    edge_weights = {}
    for u, v in G.edges():
        cu = membership[u]
        cv = membership[v]
        if cu != cv:
            key = tuple(sorted((cu, cv)))
            edge_weights[key] = edge_weights.get(key, 0) + 1
    for (c1, c2), weight in edge_weights.items():
        superG.add_edge(c1, c2, weight=weight)
    return superG

def extract_community_features(G, membership):
    comm_labels = set(membership.values())
    superG = build_supergraph(G, membership)
    comm_degree_centrality = nx.degree_centrality(superG)
    comm_betweenness = nx.betweenness_centrality(superG, weight='weight')
    comm_closeness = nx.closeness_centrality(superG, distance='weight')

    # Calculate centralities for all nodes
    deg_centrality = nx.degree_centrality(G)
    betw_centrality = nx.betweenness_centrality(G)
    close_centrality = nx.closeness_centrality(G)
    eig_centrality = nx.eigenvector_centrality(G, max_iter=500)

    data = []
    for comm in comm_labels:
        comm_members = [n for n, c in membership.items() if c == comm]
        G_sub = G.subgraph(comm_members)
        size = len(comm_members)
        intra_edges = G_sub.number_of_edges()
        inter_edges = 0
        for node in comm_members:
            for neighbor in G.neighbors(node):
                if membership[neighbor] != comm:
                    inter_edges += 1

        inter_intra_ratio = (inter_edges / intra_edges) if intra_edges != 0 else np.inf

        node_degree_centrality = [deg_centrality[n] for n in comm_members]
        betweenness = [betw_centrality[n] for n in comm_members]
        closeness = [close_centrality[n] for n in comm_members]
        eigenvector = [eig_centrality[n] for n in comm_members]

        data.append({
            "target_label": comm,
            "community_size": size,
            "intra_edges": intra_edges,
            "inter_edges": inter_edges,
            "inter_intra_ratio": inter_intra_ratio,
            "comm_degree_centrality": comm_degree_centrality[comm],
            "comm_betweenness": comm_betweenness[comm],
            "comm_closeness": comm_closeness[comm],
            'degree_centrality_mean': np.mean(node_degree_centrality),
            'degree_centrality_std': np.std(node_degree_centrality),
            'degree_centrality_median': np.median(node_degree_centrality),
            'betweenness_mean': np.mean(betweenness),
            'betweenness_std': np.std(betweenness),
            'betweenness_median': np.median(betweenness),
            'closeness_mean': np.mean(closeness),
            'closeness_std': np.std(closeness),
            'closeness_median': np.median(closeness),
            'eigenvector_mean': np.mean(eigenvector),
            'eigenvector_std': np.std(eigenvector),
            'eigenvector_median': np.median(eigenvector)
        })
    return pd.DataFrame(data)




