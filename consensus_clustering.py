

from tty import CC
import numpy as np
import cdlib.algorithms
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from node2vec import Node2Vec
from scipy.stats import rankdata


def consensus_matrix_dense(method, R, G):
    """
    Build LF-style agreement (consensus) matrix D \in [0,1]^{n x n}
    from R stochastic runs of a CDLIB method on graph G.
    method: method to use from cdlib.algorithms str from ['louvain', 'greedy_modularity', 'label_propagation', 'infomap']
    R: number of runs
    G: networkx graph
    Returns: D (n x n) numpy float array in [0,1]
    """
    node_order = list(G.nodes())
    n = len(node_order)
    partitions = {}
    for r in range(R):
        if method == 'louvain':
            partition = cdlib.algorithms.louvain(G)
            partition = [list(c) for c in partition.communities]
        elif method == 'greedy_modularity':
            partition = cdlib.algorithms.greedy_modularity(G)
            partition = [list(c) for c in partition.communities]
        elif method == 'label_propagation':
            partition = cdlib.algorithms.label_propagation(G)
            partition = [list(c) for c in partition.communities]
        elif method == 'infomap':
            partition = cdlib.algorithms.infomap(G)
            partition = [list(c) for c in partition.communities]
        else:
            raise ValueError(f"Unknown method: {method}")
        partitions[r] = partition
    method_labels = {}
    for r, partition in partitions.items():
        # conver partition to labels and save as method_labels
        labels = {}
        for i, community in enumerate(partition):
            for node in community:
                labels[node] = i
        method_labels[r] = [labels[node] for node in G.nodes()]

    labels_matrix = np.vstack([np.asarray(method_labels[r], dtype=int) for r in range(R)])  # [R, n]
    Rruns, n = labels_matrix.shape
    D = np.zeros((n, n), dtype=np.float32)

    for r in range(Rruns):
        lab = labels_matrix[r]
        # If have noise labels like -1, skip them
        for c in np.unique(lab):    
            if c == -1:
                continue
            idx = np.where(lab == c)[0]
            D[np.ix_(idx, idx)] += 1.0

    np.fill_diagonal(D, Rruns)   # always co-occur with themselves
    D /= Rruns
    return D

def consensus_labels(D, tau=0.3, reps=100):
    # pip install git+https://github.com/fiuneuro/brainconn.git
    from brainconn.clustering import consensus_und
    # D is the agreement matrix you build from R partitions
    S = consensus_und(D, tau=tau, reps=reps)
    return S

def calculate_modularity(G, labels):
    """
    G : networkx.Graph
        The input graph
    labels : array-like
        Community labels for each node (same order as G.nodes())
    """
    communities = {}
    for node, label in zip(G.nodes(), labels):
        if label not in communities:
            communities[label] = set()
        communities[label].add(node)
    
    # Convert to list of sets for networkx modularity function
    communities_list = list(communities.values())
    
    # Calculate modularity
    mod = nx.community.modularity(G, communities_list)
    return mod


# ---------------------------------------------------------------------
# Investigating Feature, True Labels and Structure Relationships
# ---------------------------------------------------------------------

from sklearn.metrics import normalized_mutual_info_score
from load_real_network import load_real_graph
from attacks import compute_ECS
from networkx.convert_matrix import to_scipy_sparse_array
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csgraph
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import torch
import csv
import os

# -------------------------------------------------------
# Logistic regression F1: features predicting true labels
# -------------------------------------------------------
def compute_feature_predictive_score(X, y):
    # test if features predict ground truth labels 
    # X = data.x.detach().cpu().numpy() if torch.is_tensor(data.x) else np.asarray(data.x)
    # y = true_labels
    pipe = make_pipeline(StandardScaler(with_mean=False),  # sparse-friendly
                         LogisticRegression(max_iter=2000, n_jobs=-1))
    f1_score = cross_val_score(pipe, X, y, cv=5, scoring='f1_macro').mean()
    return f1_score

# -------------------------------------------------------
# Feature-only clustering vs to labels or consensus clusters
# -------------------------------------------------------
def compute_diff_feature_clust_vs_labels(X, y):
    # Features-only clustering vs. Labels
    k = len(np.unique(y))
    # X_red = PCA(n_components=16, random_state=0).fit_transform(X)
    # X_red = StandardScaler(with_mean=True).fit_transform(X_red)
    # X = X_red
    C_feat = KMeans(n_clusters=k, n_init='auto', random_state=0).fit_predict(X) # K-means clustering
    # C_feat = GaussianMixture(n_components=k, random_state=0).fit_predict(X) # Gaussian Mixture Model clustering
    # C_feat = HDBSCAN(min_cluster_size=k).fit_predict(X) # HDBSCAN clustering
    nmi_feat_labels = normalized_mutual_info_score(y, C_feat)
    ecs_feat_labels = compute_ECS(y, C_feat)
    return nmi_feat_labels, ecs_feat_labels, C_feat


# -------------------------------------------------------
# Mantel test: correlation of structure embedding vs feature embedding
# -------------------------------------------------------

# def mantel_embedding_correlation(G, X, d=32, nperm=999, seed=0):
#     # --- structural embedding: node2vec ---
#     n2v = Node2Vec(
#         G,
#         dimensions=d,
#         walk_length=40,
#         num_walks=80,
#         seed=seed
#     )
#     model = n2v.fit(window=10, min_count=1, batch_words=4)
#     Z_struct = np.vstack([model.wv[str(n)] for n in G.nodes()])

#     # --- feature embedding: PCA ---
#     X_mat = X.toarray() if hasattr(X, "toarray") else X
#     Z_feat = PCA(n_components=d, random_state=seed).fit_transform(X_mat)

#     # --- distance matrices ---
#     D_struct = squareform(pdist(Z_struct))
#     D_feat   = squareform(pdist(Z_feat))

#     # --- Mantel test (Spearman, permutation-based) ---
#     n = D_struct.shape[0]
#     upper = np.triu_indices(n, 1)

#     v1 = D_struct[upper]
#     v2 = D_feat[upper]

#     r_obs = spearmanr(v1, v2)[0]

#     rng = np.random.default_rng(seed)
#     count = 0
#     for _ in range(nperm):
#         perm = rng.permutation(n)
#         v2p = D_feat[perm][:, perm][upper]
#         r_perm = spearmanr(v1, v2p)[0]
#         if abs(r_perm) >= abs(r_obs):
#             count += 1

#     pval = (count + 1) / (nperm + 1)
#     return r_obs, pval


def mantel_embedding_correlation(G, X, d=32, nperm=499, seed=0):
    # --- structural embedding: node2vec ---
    n2v = Node2Vec(
        G,
        dimensions=d,
        walk_length=40,
        num_walks=80,
        seed=seed,
        workers=1
    )
    model = n2v.fit(window=10, min_count=1, batch_words=4)
    Z_struct = np.vstack([model.wv[str(n)] for n in G.nodes()])

    # --- features reduced with PCA ---
    X_mat = X.toarray() if hasattr(X, "toarray") else X
    Z_feat = PCA(n_components=d, random_state=seed).fit_transform(X_mat)

    # --- pairwise distances (condensed form!) ---
    v1 = pdist(Z_struct)   # length = n(n-1)/2
    v2 = pdist(Z_feat)

    # --- rank once (Spearman) ---
    r1 = rankdata(v1)
    r2 = rankdata(v2)

    # observed correlation
    r_obs = np.corrcoef(r1, r2)[0, 1]

    # --- permutation test ---
    rng = np.random.default_rng(seed)
    n = Z_feat.shape[0]
    count = 0

    for _ in range(nperm):
        perm = rng.permutation(n)
        v2p = pdist(Z_feat[perm])
        r2p = rankdata(v2p)
        r_perm = np.corrcoef(r1, r2p)[0, 1]

        if abs(r_perm) >= abs(r_obs):
            count += 1

    pval = (count + 1) / (nperm + 1)
    return r_obs, pval



# -------------------------------------------------------
# Compute EVERYTHING for one dataset
# -------------------------------------------------------
def compute_dataset_metrics(dataset_name, csv_file="dataset_metrics.csv"):
    print(f"\n=== Processing dataset: {dataset_name} ===")

    # Load graph
    G, data, true_labels = load_real_graph(name=dataset_name)
    X = data.x.detach().cpu().numpy() if torch.is_tensor(data.x) else np.asarray(data.x)
    y = true_labels
    # Basic stats
    n = G.number_of_nodes()
    m = G.number_of_edges()
    modularity = calculate_modularity(G, y)
    d_features = X.shape[1]
    density = nx.density(G)
    num_labels = len(np.unique(y))

    # Assortativity
    for node, label in zip(G.nodes(), y):
        G.nodes[node]["label"] = int(label)
    assort = nx.attribute_assortativity_coefficient(G, "label")

    # Consensus clustering via Louvain
    print("Computing consensus partition...")
    D = consensus_matrix_dense("louvain", R=50, G=G)
    CC = consensus_labels(D, reps=100)
    # simple one louvain
    # print("Computing Louvain partition...")
    # partition = cdlib.algorithms.louvain(G)
    # partition = [list(c) for c in partition.communities]
    # labels = {}
    # for i, community in enumerate(partition):
    #     for node in community:
    #         labels[node] = i 
    # CC = [labels[node] for node in G.nodes()]
    k_consensus = len(set(CC))

    # Structure vs True labels
    nmi_struct_true = normalized_mutual_info_score(y, CC)
    ecs_struct_true = compute_ECS(y, CC)

    # F1: Features predictive of labels
    f1_score = compute_feature_predictive_score(X, y)

    # Feature clustering vs labels
    nmi_feat_true, ecs_feat_true, C_feat = compute_diff_feature_clust_vs_labels(X, y)

    # Feature clustering vs consensus
    nmi_feat_cons, ecs_feat_cons, _ = compute_diff_feature_clust_vs_labels(X, CC)

    # Mantel feature-structure correlation
    print("Computing Mantel test for feature-structure correlation...")
    mantel_r, mantel_p = mantel_embedding_correlation(G, X)

    # Save row
    header = [
        "dataset", "nodes", "edges", "modularity", 
        "num_features", "density", "num_labels",
        "assortativity",
        "consensus_louvain_k",
        "NMI_true_vs_consensus_louvain", "ECS_true_vs_consensus_louvain",
        "F1_feature_predict_true_labels",
        "NMI_features_vs_true_labels", "ECS_features_vs_true_labels",
        "NMI_features_vs_consensus_louvain", "ECS_features_vs_consensus_louvain",
        "Mantel_r", "Mantel_p", "links to dataset"
    ]

    row = [
        dataset_name, n, m, modularity, d_features, density, num_labels, assort,
        k_consensus,
        nmi_struct_true, ecs_struct_true,
        f1_score,
        nmi_feat_true, ecs_feat_true,
        nmi_feat_cons, ecs_feat_cons,
        mantel_r, mantel_p, ''
    ]

    write_header = not os.path.exists(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(f"Saved results to {csv_file}")
    print(row)
    return row


from tty import CC
import numpy as np
import cdlib.algorithms
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from node2vec import Node2Vec
from scipy.stats import rankdata


def consensus_matrix_dense(method, R, G):
    """
    Build LF-style agreement (consensus) matrix D \in [0,1]^{n x n}
    from R stochastic runs of a CDLIB method on graph G.
    method: method to use from cdlib.algorithms str from ['louvain', 'greedy_modularity', 'label_propagation', 'infomap']
    R: number of runs
    G: networkx graph
    Returns: D (n x n) numpy float array in [0,1]
    """
    node_order = list(G.nodes())
    n = len(node_order)
    partitions = {}
    for r in range(R):
        if method == 'louvain':
            partition = cdlib.algorithms.louvain(G)
            partition = [list(c) for c in partition.communities]
        elif method == 'greedy_modularity':
            partition = cdlib.algorithms.greedy_modularity(G)
            partition = [list(c) for c in partition.communities]
        elif method == 'label_propagation':
            partition = cdlib.algorithms.label_propagation(G)
            partition = [list(c) for c in partition.communities]
        elif method == 'infomap':
            partition = cdlib.algorithms.infomap(G)
            partition = [list(c) for c in partition.communities]
        else:
            raise ValueError(f"Unknown method: {method}")
        partitions[r] = partition
    method_labels = {}
    for r, partition in partitions.items():
        # conver partition to labels and save as method_labels
        labels = {}
        for i, community in enumerate(partition):
            for node in community:
                labels[node] = i
        method_labels[r] = [labels[node] for node in G.nodes()]

    labels_matrix = np.vstack([np.asarray(method_labels[r], dtype=int) for r in range(R)])  # [R, n]
    Rruns, n = labels_matrix.shape
    D = np.zeros((n, n), dtype=np.float32)

    for r in range(Rruns):
        lab = labels_matrix[r]
        # If have noise labels like -1, skip them
        for c in np.unique(lab):    
            if c == -1:
                continue
            idx = np.where(lab == c)[0]
            D[np.ix_(idx, idx)] += 1.0

    np.fill_diagonal(D, Rruns)   # always co-occur with themselves
    D /= Rruns
    return D

def consensus_labels(D, tau=0.3, reps=100):
    # pip install git+https://github.com/fiuneuro/brainconn.git
    from brainconn.clustering import consensus_und
    # D is the agreement matrix you build from your R partitions
    S = consensus_und(D, tau=tau, reps=reps)
    return S

def calculate_modularity(G, labels):
    """
    G : networkx.Graph
        The input graph
    labels : array-like
        Community labels for each node (same order as G.nodes())
    """
    communities = {}
    for node, label in zip(G.nodes(), labels):
        if label not in communities:
            communities[label] = set()
        communities[label].add(node)
    
    # Convert to list of sets for networkx modularity function
    communities_list = list(communities.values())
    
    # Calculate modularity
    mod = nx.community.modularity(G, communities_list)
    return mod


# ---------------------------------------------------------------------
# Investigating Feature, True Labels and Structure Relationships
# ---------------------------------------------------------------------

from sklearn.metrics import normalized_mutual_info_score
from load_real_network import load_real_graph
from attacks import compute_ECS
from networkx.convert_matrix import to_scipy_sparse_array
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csgraph
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import torch
import csv
import os

# -------------------------------------------------------
# Logistic regression F1: features predicting true labels
# -------------------------------------------------------
def compute_feature_predictive_score(X, y):
    # test if features predict ground truth labels 
    # X = data.x.detach().cpu().numpy() if torch.is_tensor(data.x) else np.asarray(data.x)
    # y = true_labels
    pipe = make_pipeline(StandardScaler(with_mean=False),  # sparse-friendly
                         LogisticRegression(max_iter=2000, n_jobs=-1))
    f1_score = cross_val_score(pipe, X, y, cv=5, scoring='f1_macro').mean()
    return f1_score

# -------------------------------------------------------
# Feature-only clustering vs to labels or consensus clusters
# -------------------------------------------------------
def compute_diff_feature_clust_vs_labels(X, y):
    # Features-only clustering vs. Labels
    k = len(np.unique(y))
    # X_red = PCA(n_components=16, random_state=0).fit_transform(X)
    # X_red = StandardScaler(with_mean=True).fit_transform(X_red)
    # X = X_red
    C_feat = KMeans(n_clusters=k, n_init='auto', random_state=0).fit_predict(X) # K-means clustering
    # C_feat = GaussianMixture(n_components=k, random_state=0).fit_predict(X) # Gaussian Mixture Model clustering
    # C_feat = HDBSCAN(min_cluster_size=k).fit_predict(X) # HDBSCAN clustering
    nmi_feat_labels = normalized_mutual_info_score(y, C_feat)
    ecs_feat_labels = compute_ECS(y, C_feat)
    return nmi_feat_labels, ecs_feat_labels, C_feat


# -------------------------------------------------------
# Mantel test: correlation of structure embedding vs feature embedding
# -------------------------------------------------------

# def mantel_embedding_correlation(G, X, d=32, nperm=999, seed=0):
#     # --- structural embedding: node2vec ---
#     n2v = Node2Vec(
#         G,
#         dimensions=d,
#         walk_length=40,
#         num_walks=80,
#         seed=seed
#     )
#     model = n2v.fit(window=10, min_count=1, batch_words=4)
#     Z_struct = np.vstack([model.wv[str(n)] for n in G.nodes()])

#     # --- feature embedding: PCA ---
#     X_mat = X.toarray() if hasattr(X, "toarray") else X
#     Z_feat = PCA(n_components=d, random_state=seed).fit_transform(X_mat)

#     # --- distance matrices ---
#     D_struct = squareform(pdist(Z_struct))
#     D_feat   = squareform(pdist(Z_feat))

#     # --- Mantel test (Spearman, permutation-based) ---
#     n = D_struct.shape[0]
#     upper = np.triu_indices(n, 1)

#     v1 = D_struct[upper]
#     v2 = D_feat[upper]

#     r_obs = spearmanr(v1, v2)[0]

#     rng = np.random.default_rng(seed)
#     count = 0
#     for _ in range(nperm):
#         perm = rng.permutation(n)
#         v2p = D_feat[perm][:, perm][upper]
#         r_perm = spearmanr(v1, v2p)[0]
#         if abs(r_perm) >= abs(r_obs):
#             count += 1

#     pval = (count + 1) / (nperm + 1)
#     return r_obs, pval


def mantel_embedding_correlation(G, X, d=32, nperm=499, seed=0):
    # --- structural embedding: node2vec ---
    n2v = Node2Vec(
        G,
        dimensions=d,
        walk_length=40,
        num_walks=80,
        seed=seed,
        workers=1
    )
    model = n2v.fit(window=10, min_count=1, batch_words=4)
    Z_struct = np.vstack([model.wv[str(n)] for n in G.nodes()])

    # --- features reduced with PCA ---
    X_mat = X.toarray() if hasattr(X, "toarray") else X
    Z_feat = PCA(n_components=d, random_state=seed).fit_transform(X_mat)

    # --- pairwise distances (condensed form!) ---
    v1 = pdist(Z_struct)   # length = n(n-1)/2
    v2 = pdist(Z_feat)

    # --- rank once (Spearman) ---
    r1 = rankdata(v1)
    r2 = rankdata(v2)

    # observed correlation
    r_obs = np.corrcoef(r1, r2)[0, 1]

    # --- permutation test ---
    rng = np.random.default_rng(seed)
    n = Z_feat.shape[0]
    count = 0

    for _ in range(nperm):
        perm = rng.permutation(n)
        v2p = pdist(Z_feat[perm])
        r2p = rankdata(v2p)
        r_perm = np.corrcoef(r1, r2p)[0, 1]

        if abs(r_perm) >= abs(r_obs):
            count += 1

    pval = (count + 1) / (nperm + 1)
    return r_obs, pval



# -------------------------------------------------------
# Compute EVERYTHING for one dataset
# -------------------------------------------------------
def compute_dataset_metrics(dataset_name, csv_file="dataset_metrics.csv"):
    print(f"\n=== Processing dataset: {dataset_name} ===")

    # Load graph
    G, data, true_labels = load_real_graph(name=dataset_name)
    X = data.x.detach().cpu().numpy() if torch.is_tensor(data.x) else np.asarray(data.x)
    y = true_labels
    # Basic stats
    n = G.number_of_nodes()
    m = G.number_of_edges()
    modularity = calculate_modularity(G, y)
    d_features = X.shape[1]
    density = nx.density(G)
    num_labels = len(np.unique(y))

    # Assortativity
    for node, label in zip(G.nodes(), y):
        G.nodes[node]["label"] = int(label)
    assort = nx.attribute_assortativity_coefficient(G, "label")

    # Consensus clustering via Louvain
    print("Computing consensus partition...")
    D = consensus_matrix_dense("louvain", R=50, G=G)
    CC = consensus_labels(D, reps=100)
    # simple one louvain
    # print("Computing Louvain partition...")
    # partition = cdlib.algorithms.louvain(G)
    # partition = [list(c) for c in partition.communities]
    # labels = {}
    # for i, community in enumerate(partition):
    #     for node in community:
    #         labels[node] = i 
    # CC = [labels[node] for node in G.nodes()]
    k_consensus = len(set(CC))

    # Structure vs True labels
    nmi_struct_true = normalized_mutual_info_score(y, CC)
    ecs_struct_true = compute_ECS(y, CC)

    # F1: Features predictive of labels
    f1_score = compute_feature_predictive_score(X, y)

    # Feature clustering vs labels
    nmi_feat_true, ecs_feat_true, C_feat = compute_diff_feature_clust_vs_labels(X, y)

    # Feature clustering vs consensus
    nmi_feat_cons, ecs_feat_cons, _ = compute_diff_feature_clust_vs_labels(X, CC)

    # Mantel feature-structure correlation
    print("Computing Mantel test for feature-structure correlation...")
    mantel_r, mantel_p = mantel_embedding_correlation(G, X)

    # Save row
    header = [
        "dataset", "nodes", "edges", "modularity", 
        "num_features", "density", "num_labels",
        "assortativity",
        "consensus_louvain_k",
        "NMI_true_vs_consensus_louvain", "ECS_true_vs_consensus_louvain",
        "F1_feature_predict_true_labels",
        "NMI_features_vs_true_labels", "ECS_features_vs_true_labels",
        "NMI_features_vs_consensus_louvain", "ECS_features_vs_consensus_louvain",
        "Mantel_r", "Mantel_p", "links to dataset"
    ]

    row = [
        dataset_name, n, m, modularity, d_features, density, num_labels, assort,
        k_consensus,
        nmi_struct_true, ecs_struct_true,
        f1_score,
        nmi_feat_true, ecs_feat_true,
        nmi_feat_cons, ecs_feat_cons,
        mantel_r, mantel_p, ''
    ]

    write_header = not os.path.exists(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(f"Saved results to {csv_file}")
    print(row)
    return row