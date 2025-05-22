
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt

def generate_features_from_communities(true_labels, mode='random', num_features=32, noise_scale=0.1, seed=42):
    """
    Generate node features based on community structure or randomly.

    Parameters:
    - true_labels: np.array of shape [n_nodes]
    - mode: 'random', 'matched', 'grouped', or 'nested'
    - num_features: number of features per node
    - noise_scale: noise standard deviation
    - seed: random seed

    Returns:
    - features: np.array of shape [n_nodes, num_features]
    """
    np.random.seed(seed)
    n_nodes = len(true_labels)
    n_communities = len(set(true_labels))

    if mode == 'random':
        features = np.random.randn(n_nodes, num_features)
        # print("Features shape:", features.shape, "Features:", features)
        return features

    elif mode == 'matched':
        # Generate features that are matched to the community structure with gaussian pattern
        centroids = np.random.randn(n_communities, num_features)
        # print("Centroids shape:", centroids.shape, "Centroids:", centroids)

        features = np.vstack([np.random.normal(loc=centroids[label], scale=noise_scale)
                                for label in true_labels])
        
        # print("Features shape:", features.shape, "Features:", features)

    else:
        raise ValueError("mode must be one of: 'random', 'matched'")

    features += np.random.normal(scale=noise_scale, size=features.shape) # Add random noise to the data.
    return features



def generate_lfr_graph(
    n=1000,
    tau1 = 2,  # Power-law exponent for the degree distribution
    tau2 = 1.1,  # Power-law exponent for the community size distribution
    mu = 0.1,
    avg_degree = 25,    # Average Degree
    max_degree_ratio = 0.1,
    min_community = 60,  # Min Community Size
    num_features = 32,  # Number of features for generation
    seed = 17,
    feature_mode = 'random',  # 'random', 'matched', 'grouped', or 'nested'
    noise_scale = 0.1  # Gaussian noise to add to features
):
    max_degree = int(max_degree_ratio * n)  # Max Degree
    max_community = int(max_degree_ratio * n)  # Max Community Size

    # Generate LFR benchmark graph
    G = nx.generators.community.LFR_benchmark_graph(
        n, tau1, tau2, mu,
        average_degree=avg_degree,
        max_degree=max_degree,
        min_community=min_community,
        max_community=max_community,
        seed=seed
    )

    # Clean up multi-edges and self-loops
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Map each unique community set to a unique ID
    community_sets = list({frozenset(G.nodes[n]['community']) for n in G.nodes})
    community_id_map = {com: idx for idx, com in enumerate(community_sets)}
    com_labels = {n: community_id_map[frozenset(G.nodes[n]['community'])] for n in G.nodes}
    true_labels = np.array([com_labels[i] for i in range(n)])

    # Generate structured or random features
    features = generate_features_from_communities(
        true_labels, mode=feature_mode, num_features=num_features,
        noise_scale=noise_scale, seed=seed
    )

    # Add features to NetworkX nodes
    for i, feat in enumerate(features):
        G.nodes[i]['x'] = torch.tensor(feat, dtype=torch.float)

    # Convert to PyG Data object
    data = from_networkx(G)
    data.x = torch.stack([data.x[i] for i in range(data.num_nodes)])

    return G, data, true_labels





def visualize_lfr_graph(G, com_labels, layout_seed=42, figsize=(10, 10), title=None):
    """
    Visualize an LFR graph with communities.

    Parameters:
    - G: NetworkX graph
    - com_labels: dict {node: community_id}
    - layout_seed: random seed for layout reproducibility
    - figsize: tuple for plot size
    - title: optional title string
    """
    unique_coms = list(set(com_labels.values()))
    color_palette = plt.cm.tab20.colors
    color_map = {com: color_palette[i % len(color_palette)] for i, com in enumerate(unique_coms)}
    node_colors = [color_map[com_labels[n]] for n in G.nodes]

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=layout_seed)
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color=node_colors, alpha=0.6)
    nx.draw_networkx_edges(G, pos, width=0.1, alpha=0.3)
    
    plt.title(title or f"LFR Graph: {G.number_of_nodes()} nodes, {len(unique_coms)} communities")
    plt.axis('off')
    plt.show()
