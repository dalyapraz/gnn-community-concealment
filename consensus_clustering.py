import numpy as np
import cdlib.algorithms
import networkx as nx


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

def consensus_labels(D, tau=0.1, reps=100):
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