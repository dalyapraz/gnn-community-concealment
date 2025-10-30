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


def read_features_file_plain(features_file, nodetype=int):
    """
    Lines: <node_id> <f1> <f2> ...
    Returns dict[node_id] -> np.ndarray(float32, D)
    """
    fmap = {}
    with open(features_file, "r") as f:
        for ln, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            nid = nodetype(parts[0])
            feats = np.asarray([float(v) for v in parts[1:]], dtype=np.float32)
            fmap[nid] = feats
    return fmap


def community_centroids_and_scatter_from_map(membership, feature_map):
    """
    membership: dict[node_id] -> comm_id
    feature_map: dict[node_id] -> np.ndarray(D,)
    Returns:
      centroids: dict[comm] -> np.ndarray(D,) (community mean)
      trace_cov: dict[comm] -> float   (sum of per-dim variances)
      sizes    : dict[comm] -> int (number of nodes in comm)
    """
    comm_to_feats = {}
    for n, c in membership.items():
        x = feature_map.get(n)
        if x is None:
            raise KeyError(f"Node {n} missing in features file.")
        comm_to_feats.setdefault(c, []).append(x)

    centroids, trace_cov, sizes = {}, {}, {}
    for c, rows in comm_to_feats.items():
        X = np.vstack(rows)                 # (Nc, D)
        mu = X.mean(axis=0)
        var = X.var(axis=0, ddof=0)         # population variance
        centroids[c]  = mu.astype(np.float32)
        trace_cov[c]  = float(var.sum()) 
        sizes[c]      = int(X.shape[0]) 
    return centroids, trace_cov, sizes

def pairwise_comm_sqdist(centroids, trace_cov):
    """
    Returns two DataFrames indexed by community id:
      D2_centroid[c1,c2] = ||mu_c1 - mu_c2||^2
      D2_expected[c1,c2] = ||mu_c1 - mu_c2||^2 + trΣ_c1 + trΣ_c2
    """
    comms = sorted(centroids.keys())
    MU = np.vstack([centroids[c] for c in comms])  # (C, D)
    MU2 = (MU * MU).sum(axis=1, keepdims=True)     # (C, 1)
    D2c = (MU2 + MU2.T - 2.0 * MU @ MU.T).clip(min=0.0)

    tr = np.array([trace_cov[c] for c in comms], dtype=np.float64)
    D2e = D2c + tr[:, None] + tr[None, :]

    idx = pd.Index(comms, name="community")
    return pd.DataFrame(D2c, index=idx, columns=idx), pd.DataFrame(D2e, index=idx, columns=idx)


def comm_centroid_sqdist_matrix_and_avgs(
    membership: dict,
    feature_map: dict,
    weighted_by_other_size: bool = False,
):
    """
    Compute centroid-based squared distances between communities and
    the per-community average distance to all other communities.

    Parameters
    ----------
    membership : dict[node_id] -> comm_id
    feature_map: dict[node_id] -> np.ndarray(D,)
    weighted_by_other_size : bool
        If True, the average distance from community c to others is
        weighted by the sizes of the *other* communities.

    Returns
    -------
    D2_df : pd.DataFrame (C x C)
        Squared distances between community centroids:
        D2_df[c1, c2] = ||mu_c1 - mu_c2||^2
        Diagonal is 0.
    avg_dist_to_others : pd.Series (length C)
        For each community c, the average of D2_df[c, others].
        If weighted_by_other_size=True, weights are |others|.
    sizes : dict[comm_id] -> int
        Community sizes (number of nodes per community).
    """
    # ---- gather features per community ----
    comm_to_feats = {}
    for n, c in membership.items():
        x = feature_map.get(n)
        if x is None:
            raise KeyError(f"Node {n} missing in features file.")
        comm_to_feats.setdefault(c, []).append(x)

    # ---- centroids and sizes ----
    centroids, sizes = {}, {}
    for c, rows in comm_to_feats.items():
        X = np.vstack(rows)             # (Nc, D)
        centroids[c] = X.mean(axis=0).astype(np.float32)
        sizes[c] = int(X.shape[0])

    # ---- squared distances between centroids ----
    comms = sorted(centroids.keys())
    MU = np.vstack([centroids[c] for c in comms])    # (C, D)
    MU2 = (MU * MU).sum(axis=1, keepdims=True)       # (C, 1)
    D2 = (MU2 + MU2.T - 2.0 * (MU @ MU.T)).clip(min=0.0)  # (C, C)
    np.fill_diagonal(D2, 0.0)

    D2_df = pd.DataFrame(D2, index=pd.Index(comms, name="community"),
                              columns=pd.Index(comms, name="community"))

    # ---- per-community average to others ----
    avg_vals = []
    size_vec = np.array([sizes[c] for c in comms], dtype=np.float64)

    for i, c in enumerate(comms):
        # mask to exclude self
        mask = np.ones(len(comms), dtype=bool)
        mask[i] = False
        row = D2[i, mask]  # distances from c to others

        if row.size == 0:
            avg_vals.append(0.0)
            continue

        if weighted_by_other_size:
            w = size_vec[mask]
            w = w / w.sum()
            avg_vals.append(float(np.dot(row, w)))
        else:
            avg_vals.append(float(row.mean()))

    avg_dist_to_others = pd.Series(avg_vals, index=D2_df.index, name="avg_centroid_sqdist_to_others")

    return D2_df, avg_dist_to_others, sizes


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


def extract_community_features(G, membership, feature_map, weighted_by_other_size: bool = False, return_distance_matrix: bool = False):
    """
    Extends your extractor with centroid-based community-to-community distances.

    New columns:
      - avg_centroid_sqdist_to_others
      - min_centroid_sqdist_to_others
      - max_centroid_sqdist_to_others
      - nearest_comm_by_centroid
    """
    comm_labels = set(membership.values())

    # Supergraph-level centralities
    superG = build_supergraph(G, membership)
    comm_degree_centrality = nx.degree_centrality(superG)
    comm_betweenness = nx.betweenness_centrality(superG, weight='weight')
    comm_closeness = nx.closeness_centrality(superG, distance='weight')

    # Node-level centralities
    deg_centrality = nx.degree_centrality(G)
    betw_centrality = nx.betweenness_centrality(G)
    close_centrality = nx.closeness_centrality(G)
    eig_centrality = nx.eigenvector_centrality(G, max_iter=500)

    # NEW: centroid distance matrix & per-community averages
    D2_df, avg_to_others, sizes = comm_centroid_sqdist_matrix_and_avgs(
        membership, feature_map, weighted_by_other_size=weighted_by_other_size
    )

    data = []
    for comm in comm_labels:
        comm_members = [n for n, c in membership.items() if c == comm]
        G_sub = G.subgraph(comm_members)
        size = len(comm_members)

        intra_edges = G_sub.number_of_edges()
        inter_edges = sum(1 for node in comm_members for neighbor in G.neighbors(node) if membership[neighbor] != comm)
        inter_intra_ratio = (inter_edges / intra_edges) if intra_edges != 0 else np.inf

        node_degree_centrality = [deg_centrality[n] for n in comm_members]
        betweenness = [betw_centrality[n] for n in comm_members]
        closeness = [close_centrality[n] for n in comm_members]
        eigenvector = [eig_centrality[n] for n in comm_members]

        # Distances to other communities (exclude self)
        others = [c for c in D2_df.columns if c != comm]
        if others:
            row = D2_df.loc[comm, others]
            avg_centroid_sqdist_to_others = float(avg_to_others.loc[comm])
            min_centroid_sqdist_to_others = float(row.min())
            max_centroid_sqdist_to_others = float(row.max())
            nearest_comm_by_centroid = int(row.idxmin())
        else:
            avg_centroid_sqdist_to_others = 0.0
            min_centroid_sqdist_to_others = 0.0
            max_centroid_sqdist_to_others = 0.0
            nearest_comm_by_centroid = comm

        data.append({
            "target_label": comm,
            "community_size": size,
            "intra_edges": intra_edges,
            "inter_edges": inter_edges,
            "inter_intra_ratio": inter_intra_ratio,
            "comm_degree_centrality": comm_degree_centrality.get(comm, 0.0),
            "comm_betweenness": comm_betweenness.get(comm, 0.0),
            "comm_closeness": comm_closeness.get(comm, 0.0),
            "degree_centrality_mean": np.mean(node_degree_centrality),
            "degree_centrality_std": np.std(node_degree_centrality),
            "degree_centrality_median": np.median(node_degree_centrality),
            "betweenness_mean": np.mean(betweenness),
            "betweenness_std": np.std(betweenness),
            "betweenness_median": np.median(betweenness),
            "closeness_mean": np.mean(closeness),
            "closeness_std": np.std(closeness),
            "closeness_median": np.median(closeness),
            "eigenvector_mean": np.mean(eigenvector),
            "eigenvector_std": np.std(eigenvector),
            "eigenvector_median": np.median(eigenvector),

            # NEW feature-based community distance stats
            "avg_centroid_sqdist_to_others": avg_centroid_sqdist_to_others,
            "min_centroid_sqdist_to_others": min_centroid_sqdist_to_others,
            "max_centroid_sqdist_to_others": max_centroid_sqdist_to_others,
            "nearest_comm_by_centroid": nearest_comm_by_centroid,
        })

    df = pd.DataFrame(data)
    if return_distance_matrix:
        return df, D2_df  # also return the full CxC matrix for optional analysis
    return df
