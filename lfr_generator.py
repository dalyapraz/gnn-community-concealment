
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt

def get_nesting_map(larger_k, smaller_k):
    """
    Maps indices in the larger set to the smaller one as evenly as possible.
    
    Returns:
    - A dict mapping each item in the larger set to a group in the smaller set
    """
    assignment = {}
    base = larger_k // smaller_k
    remainder = larger_k % smaller_k

    index = 0
    for group in range(smaller_k):
        # print(f"Group {group}:")
        # Calculate how many items to assign to this group
        # If the group index is less than the remainder, it gets one extra item
        # Otherwise, it gets the base number of items
        count = base + (1 if group < remainder else 0)
        # print(f"  Assigning {count} items to group {group}")
        for _ in range(count):
            assignment[index] = group
            index += 1
            # print(f"  Assigned item {index-1} to group {group}")
    # print(f"Mapping from {larger_k} to {smaller_k} groups: {assignment}")
    return assignment


def generate_features_from_communities(
        true_labels,
        mode='random',
        num_features=32,
        sigma_c=3.0,
        sigma=1.0,
        n_clusters_f=None,
        seed=7):
    """
    Generate node features based on community structure or randomly.

    Parameters:
    - true_labels: np.array of shape [n_nodes]
    - mode: 'random' or 'gaussian'
    - num_features: number of features per node
    - sigma_c: std for generating feature centroids
    - sigma: std for generating points around each centroid
    - n_clusters_f: number of feature clusters (None = set equal to n_communities)
    - seed: random seed

    Returns:
    - features: np.array of shape [n_nodes, num_features]
    """
    np.random.seed(seed)
    n_nodes = len(true_labels)
    true_labels = np.array(true_labels)
    unique_comms = np.unique(true_labels)
    n_communities = len(unique_comms)

    if mode == 'random':
        features = np.random.randn(n_nodes, num_features) # drawn from ~N(0, 1)
        return features

    elif mode == 'gaussian':
        if n_clusters_f is None:
            n_clusters_f = n_communities    # by default, set number of feature clusters equal to number of communities
        # centroids are drawn from ~N(0, sigma_c*I)
        centroids = np.random.multivariate_normal(np.zeros(num_features), 
                                                np.identity(num_features) * sigma_c, n_clusters_f)
        # print("Centroids shape:", centroids.shape, "Centroids:", centroids)

        if n_clusters_f == n_communities:   # Generate MATCHED features to the community structure
            comm_to_f_cluster = {c: i for i, c in enumerate(unique_comms)}   # Map community to feature cluster
            # features are drawn from ~N(centroid_label, sigma)
            features = np.vstack([
                np.random.normal(loc=centroids[comm_to_f_cluster[label]], scale=sigma)
                for label in true_labels])
        
        elif n_clusters_f < n_communities:  # Generate GROUPED features: several communities in the same feature cluster
            # Randomly assign communities to feature clusters
            # group_assignments = np.random.randint(0, n_clusters_f, size=n_communities)
            # Evenly distribute communities across feature clusters
            group_assignments = get_nesting_map(n_communities, n_clusters_f)
            comm_to_f_cluster = {c: group_assignments[c] for c in unique_comms} 
            # features are drawn from ~N(centroid_label, sigma)
            features = np.vstack([
                np.random.normal(loc=centroids[comm_to_f_cluster[label]], scale=sigma)
                for label in true_labels])
        elif n_clusters_f > n_communities:  # Generate NESTED features: several feature clusters in the same community
            # Randomly assign feature clusters to communities
            # group_assignments = np.random.randint(0, n_clusters_f, size=n_communities)
            # Evenly distribute feature clusters across communities
            group_assignments = get_nesting_map(n_clusters_f, n_communities)
            # then reverse it to map community → list of feature clusters
            comm_to_f_cluster = {c: [] for c in unique_comms}
            for f, c in group_assignments.items():
                comm_to_f_cluster[c].append(f)
            # features are drawn from ~N(centroid_label, sigma), but random cluster_f for each community
            # (i.e. each community has several feature clusters)
            # print("Community to feature cluster mapping:", comm_to_f_cluster)
            features = np.vstack([
                np.random.normal(loc=centroids[np.random.choice(comm_to_f_cluster[label])], scale=sigma)
                for label in true_labels])
        # print("Features shape:", features.shape, "Features:", features)
        
    else:
        raise ValueError("mode must be one of: 'random', 'gaussian'")

    # features += np.random.normal(scale=noise_scale, size=features.shape) # Add random noise to the data.
    return features


def generate_featurized_lfr_graph(
    n=1000,
    tau1 = 2,  # Power-law exponent for the degree distribution
    tau2 = 1.1,  # Power-law exponent for the community size distribution
    mu = 0.5,
    avg_degree = 25,    # Average Degree
    max_degree_ratio = 0.1,
    min_community = 60,  # Min Community Size
    num_features = 32,  # Number of features for generation
    feature_mode = 'random', # 'random' or 'gaussian'
    sigma_c = 3.0,  # Standard deviation for community feature centroids
    sigma = 1.0,  # Standard deviation for feature generation
    n_clusters_f=None,  # Number of feature clusters (None = set equal to n_communities)
    seed = None
):
    """
    Generate a LFR benchmark graph.
    Parameters:
    - n: number of nodes
    - tau1: power-law exponent for the degree distribution
    - tau2: power-law exponent for the community size distribution
    - mu: mixing parameter (fraction of edges that are inter-community)
    - avg_degree: average degree of the graph
    - max_degree_ratio: maximum degree as a fraction of n (e.g., 0.1 means max degree is 10% of n)
    - min_community: minimum size of a community
    - num_features: number of features per node
    - feature_mode: 'random' or 'gaussian' for feature generation
    - sigma_c: standard deviation for community feature centroids (for 'gaussian' mode)
    - sigma: standard deviation for feature generation (for 'gaussian' mode)
    - n_clusters_f: number of feature clusters (None = set equal to n_communities)
    - seed: random seed for reproducibility
    Returns:
    - G: NetworkX graph object
    - data: PyG Data object with node features
    - true_labels: np.array of shape [n_nodes] with community labels    
    """
    max_degree = int(max_degree_ratio * n)  # Max Degree
    max_community = int(max_degree_ratio * n)  # Max Community Size

    # Generate LFR benchmark graph
    G = nx.generators.community.LFR_benchmark_graph(
        n, tau1, tau2, mu,
        average_degree=avg_degree,
        max_degree=max_degree,
        min_community=min_community,
        max_community=max_community,
        seed=seed,
        # max_iters=2000,  # Increase max iterations for convergence
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
        sigma_c=sigma_c, sigma=sigma, n_clusters_f=n_clusters_f,
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



@torch.inference_mode()
def precompute_allpairs_neg_sqeuclidean(G, feature_key='x', device=None, dtype=torch.float32):
    """
    Returns:
      nodes : list of node ids in a fixed order
      idx_of: dict mapping node_id -> row/col index
      F     : (N,D) feature tensor on `device`
      S     : (N,N) NEG squared Euclidean similarity (bigger = closer), diag = -inf
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    nodes  = list(G.nodes())
    idx_of = {n: i for i, n in enumerate(nodes)}

    feats = []
    for n in nodes:
        x = G.nodes[n][feature_key]                 # grab this node's feature vector
        if isinstance(x, torch.Tensor):             # if it's a torch tensor,
            x = x.detach().cpu().numpy()            # detach and move to CPU NumPy
        feats.append(np.asarray(x, dtype=np.float32))

    # stack to (N,D) and move to the chosen device (CPU or GPU)
    F = torch.as_tensor(np.stack(feats), dtype=dtype, device=device).contiguous()

    # ||x_i||^2 for every node i, shape (N,1)
    X2 = (F * F).sum(dim=1, keepdim=True)

    # pairwise negative squared distances:
    #   -||x_i - x_j||^2 = -(||x_i||^2 + ||x_j||^2 - 2 * x_i·x_j)
    S = (X2 + X2.T - 2.0 * (F @ F.T)).clamp_min_(0).neg_()

    # avodid self-loops in similarity graph
    S.fill_diagonal_(-float('inf'))

    return nodes, idx_of, F, S

