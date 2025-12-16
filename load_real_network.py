from typing import Tuple, Optional

import os
from more_itertools import last
import numpy as np
import torch
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import AttributedGraphDataset, CitationFull, EllipticBitcoinDataset
from torch_geometric.datasets import Actor, Reddit2

import json
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
import networkx as nx

def load_snap_graph(folder):
    """
    Load a SNAP dataset (e.g., Deezer Europe, LastFM Asia)
    with files:
        edges.csv
        features.json
        target.csv
    
    Returns:
        G  (networkx graph)
        data (PyG Data object)
        y (np.ndarray labels)
    """

    # --- Load edges ---
    edges = pd.read_csv(f"datasets/{folder}/{folder}_edges.csv")
    src = edges.iloc[:,0].values
    dst = edges.iloc[:,1].values

    n_nodes = max(src.max(), dst.max()) + 1

    # --- Load features ---
    with open(f"datasets/{folder}/{folder}_features.json", "r") as f:
        feat_raw = json.load(f)

    # Find total number of feature dimensions
    max_feat = 0
    for v in feat_raw.values():
        if v:
            max_feat = max(max_feat, max(v))
    num_features = max_feat + 1

    # Create sparse feature matrix
    X = sp.lil_matrix((n_nodes, num_features), dtype=np.float32)
    for node_str, feats in feat_raw.items():
        node = int(node_str)
        X[node, feats] = 1.0

    X = X.tocsr()

    # --- Load labels ---
    labels_df = pd.read_csv(f"datasets/{folder}/{folder}_target.csv")
    labels_df = labels_df.sort_values("id")  # ensure correct ordering
    y = labels_df["target"].values.astype(int)

    # --- Build scipy adjacency ---
    A = sp.coo_matrix(
        (np.ones(len(src)), (src, dst)),
        shape=(n_nodes, n_nodes),
        dtype=np.int32
    )

    # Make undirected
    A = A + A.T
    A[A > 1] = 1

    # --- Convert to PyG ---
    edge_index, _ = from_scipy_sparse_matrix(A)
    x = torch.tensor(X.toarray(), dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y_torch)

    # --- Build networkx graph ---
    G = nx.from_scipy_sparse_array(A)

    return G, data, y


def load_real_graph(
    name: str = "Cora",
    root = None
    ):
    """
    Load a real graph dataset from PyG and return (G, data, true_labels)
    in the SAME format as `generate_featurized_lfr_graph` returns.

    Parameters
    - name: dataset name (e.g., 'Cora', 'Cora_ML', etc.)
    - root: optional root folder for dataset cache; defaults to ~/pyg_data/<source>

    Returns
    - G: networkx.Graph with node attribute 'x' as a torch.Tensor per node
    - data: torch_geometric.data.Data with fields edge_index, x, y, ...
    - true_labels: np.ndarray of shape [num_nodes]
    """
    name_clean = name.lower().strip()
    transform = T.Compose([T.ToUndirected(), T.RemoveSelfLoops(), T.RemoveDuplicatedEdges()])
    # --- Choose dataset root ---
    if root is None:
        root = os.path.expanduser(os.path.join("~", "pyg_data", name_clean))
    if name_clean in ["cora", "citeseer", "pubmed", "wiki", "facebook", "blogcatalog", "flickr"]:
        dataset = AttributedGraphDataset(root=root, name=name, transform=transform)
    elif name_clean in ['bitcointransactions']:
        dataset = EllipticBitcoinDataset(root=root, transform=transform)
    elif name_clean in ["lastfmasia"]:
        return load_snap_graph("lasftm_asia")
    elif name_clean in ["deezer_europe", "lastfm_asia"]:
        return load_snap_graph(name_clean)
    
    # elif name_clean in ["github"]: 
    #     dataset = GitHub(root=root, transform=transform) # doesn't exist (404 error)
    # elif name_clean in ["lastfmasia"]:
    #     dataset = LastFMAsia(root=root, transform=transform)
    elif name_clean in ["actor"]:
        dataset = Actor(root=root, transform=transform)
    elif name_clean in ["reddit2"]:
        dataset = Reddit2(root=root, transform=transform)
    # elif name_clean in ["twitch"]:
    #     dataset = Twitch(root=root, name = "EN", transform=transform)
    # elif name_clean in ["gemsecdeezer"]:
    #     dataset = GemsecDeezer(root=root, name = "RO", transform=transform)
    else:
        raise ValueError(f"Unsupported dataset name '{name}'. Add it to the loader.")

    # --- Load dataset ---
    # transform_list =[]
    # # if make_undirected:
    # transform_list.append(T.ToUndirected())
    # transform_list.append(T.RemoveSelfLoops())
    # transform = T.Compose([T.ToUndirected(), T.RemoveSelfLoops(), T.RemoveDuplicatedEdges()])
    # transform = T.Compose([T.ToUndirected()])

    # source = (source or "").strip()
    # if source == "AttributedGraphDataset":
    #     dataset = AttributedGraphDataset(root=root, name=name, transform=transform)
    # elif source == "CitationFull":
    #     dataset = CitationFull(root=root, name=name, transform=transform)
    # else:
    #     raise ValueError(f"Unsupported source '{source}'. Use 'AttributedGraphDataset' or 'CitationFull'.")

    if len(dataset) == 0:
        raise RuntimeError(f"Loaded empty dataset for source={source}, name={name}")
    data = dataset[0]
    # make sure data is undirected and has no self-loops
    data = transform(data)
    # make sure the edges are unique

    # Build NetworkX graph with 'x' per node (torch.Tensor) 
    # Include the node attribute 'x' when converting
    G = to_networkx(data, to_undirected=True, node_attrs=["x"])  # type: ignore
    # only use largest connected component 
    # largest_cc = max(nx.connected_components(G), key=len)
    # G = G.subgraph(largest_cc).copy()
    # # keep only nodes in largest connected component in data as well
    # node_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(largest_cc))}
    # data = T.Subgraph(list(largest_cc))(data)
    # # re-map node indices to be consecutive starting from 0
    # data.edge_index = torch.tensor([
    #     [node_id_map[int(src)] for src in data.edge_index[0].tolist()],
    #     [node_id_map[int(dst)] for dst in data.edge_index[1].tolist()]
    # ], dtype=torch.long)
    # Remove self-loops at the NX level too (to mirror LFR function exactly)
    # G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Make sure every node has 'x' as a torch.Tensor
    for n in G.nodes:
        x = G.nodes[n].get("x")
        if x is None:
            # align with Data.x row if missing
            G.nodes[n]["x"] = data.x[n].detach() if torch.is_tensor(data.x) else torch.tensor(data.x[n])
        elif not torch.is_tensor(x):
            G.nodes[n]["x"] = torch.as_tensor(x, dtype=torch.float32)

    # --- Labels ---
    if getattr(data, "y", None) is not None:
        # Convert labels to a 1-D array of integer class ids
        if torch.is_tensor(data.y):
            y = data.y.detach().cpu()
            # If labels are one-hot or 2D, reduce to class indices via argmax
            if y.ndim > 1:
                true_labels = y.argmax(dim=1).numpy()
            else:
                true_labels = y.numpy()
        else:
            y = np.asarray(data.y)
            if y.ndim > 1:
                # Assume one-hot rows -> take argmax across classes
                true_labels = y.argmax(axis=1)
            else:
                true_labels = y
        # Ensure plain 1-D int array
        true_labels = np.asarray(true_labels).reshape(-1).astype(int)
    
    if name_clean == "bitcointransactions":
        # special case: EllipticBitcoinDataset only use LCC
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        data = data.subgraph(torch.tensor(sorted(largest_cc), dtype=torch.long))
        true_labels = true_labels[list(largest_cc)]

    return G, data, true_labels


# import os
# import numpy as np
# import torch
# import networkx as nx
# from torch_geometric.utils import to_networkx
# import torch_geometric.transforms as T

# # PyG datasets
# from torch_geometric.datasets import (
#     AttributedGraphDataset,
#     CitationFull,
#     GitHub,
#     HeterophilousGraphDataset,  # includes LastFMAsia, Actor
# )

# def load_real_graph(name: str, root=None):
#     """
#     Unified loader for multiple PyG graph datasets.
#     Returns (G, data, true_labels):
#         G = networkx.Graph with node feature 'x' (torch.Tensor)
#         data = PyG Data object
#         true_labels = np.ndarray of shape [num_nodes]
#     """

#     name_clean = name.lower().strip()

#     # -----------------------------
#     # Decide dataset root
#     # -----------------------------
#     if root is None:
#         root = os.path.expanduser(os.path.join("~", "pyg_data", name_clean))

#     # -----------------------------
#     # Standard transforms
#     # -----------------------------
#     transform = T.Compose([
#         T.ToUndirected(),
#         T.RemoveSelfLoops(),
#         T.RemoveDuplicatedEdges()
#     ])

#     # -----------------------------
#     # Dataset dispatch
#     # -----------------------------
#     if name_clean in ["cora", "citeseer", "pubmed", "wiki", "facebook"]:
#         # your old behavior
#         dataset = AttributedGraphDataset(root=root, name=name, transform=transform)

#     elif name_clean in ["cora_full"]:
#         dataset = CitationFull(root=root, name="Cora", transform=transform)

#     elif name_clean in ["github"]:
#         dataset = GitHub(root=root, transform=transform)

#     elif name_clean in ["lastfmasia", "lastfm", "actor", "minesweeper", "roman-empire",
#                         "questions", "amazon-ratings", "tolokers"]:
#         # All are inside HeterophilousGraphDataset
#         dataset = HeterophilousGraphDataset(root=root, name=name_clean, transform=transform)

#     else:
#         raise ValueError(f"Unsupported dataset name '{name}'. Add it to the loader.")

#     # -----------------------------
#     # Validate dataset
#     # -----------------------------
#     if len(dataset) == 0:
#         raise RuntimeError(f"Loaded empty dataset: name={name}, root={root}")

#     # PyG Data object
#     data = dataset[0]
#     data = transform(data)   # ensure safety

#     # -----------------------------
#     # Convert to NetworkX graph
#     # -----------------------------
#     # include node features "x" if present
#     node_attrs = ["x"] if hasattr(data, "x") else []
#     G = to_networkx(data, to_undirected=True, node_attrs=node_attrs)

#     # remove remaining self-loops
#     G.remove_edges_from(nx.selfloop_edges(G))

#     # -----------------------------
#     # Ensure each node has a tensor feature
#     # -----------------------------
#     if hasattr(data, "x") and data.x is not None:
#         for n in G.nodes():
#             x = G.nodes[n].get("x")
#             if x is None:
#                 G.nodes[n]["x"] = data.x[n].detach() if torch.is_tensor(data.x) else torch.tensor(data.x[n])
#             elif not torch.is_tensor(x):
#                 G.nodes[n]["x"] = torch.tensor(x, dtype=torch.float32)
#     else:
#         # Some heterophilous datasets (like Actor) have no x → create dummy features
#         print(f"[Warning] Dataset '{name}' has no node features. Creating identity features.")
#         N = G.number_of_nodes()
#         X = torch.eye(N)
#         data.x = X
#         for n in G.nodes():
#             G.nodes[n]["x"] = X[n]

#     # -----------------------------
#     # Extract labels (data.y)
#     # -----------------------------
#     if hasattr(data, "y") and data.y is not None:
#         y = data.y
#         if torch.is_tensor(y):
#             y = y.cpu().numpy()
#         y = np.asarray(y).reshape(-1).astype(int)
#         true_labels = y
#     else:
#         print(f"[Warning] Dataset '{name}' has no labels (data.y). Creating dummy zero labels.")
#         true_labels = np.zeros(data.num_nodes, dtype=int)

#     return G, data, true_labels
