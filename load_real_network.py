from typing import Tuple, Optional

import os
import numpy as np
import torch
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import AttributedGraphDataset, CitationFull


def load_real_graph(
    name: str = "Cora",
    source: str = "AttributedGraphDataset",
    root = None
    ):
    """
    Load a real graph dataset from PyG and return (G, data, true_labels)
    in the SAME format as `generate_featurized_lfr_graph` returns.

    Parameters
    - name: dataset name (e.g., 'Cora', 'Cora_ML', etc.)
    - source: which PyG dataset to use: 'AttributedGraphDataset' or 'CitationFull'
    - root: optional root folder for dataset cache; defaults to ~/pyg_data/<source>
    - make_undirected: convert the graph to undirected and remove self-loops to match LFR format

    Returns
    - G: networkx.Graph with node attribute 'x' as a torch.Tensor per node
    - data: torch_geometric.data.Data with fields edge_index, x, y, ...
    - true_labels: np.ndarray of shape [num_nodes]
    """

    # --- Choose dataset root ---
    if root is None:
        root = os.path.expanduser(os.path.join("~", "pyg_data", source))

    # --- Load dataset ---
    # transform_list =[]
    # # if make_undirected:
    # transform_list.append(T.ToUndirected())
    # transform_list.append(T.RemoveSelfLoops())
    transform = T.Compose([T.ToUndirected(), T.RemoveSelfLoops(), T.RemoveDuplicatedEdges()])
    # transform = T.Compose([T.ToUndirected()])

    source = (source or "").strip()
    if source == "AttributedGraphDataset":
        dataset = AttributedGraphDataset(root=root, name=name, transform=transform)
    elif source == "CitationFull":
        dataset = CitationFull(root=root, name=name, transform=transform)
    else:
        raise ValueError(f"Unsupported source '{source}'. Use 'AttributedGraphDataset' or 'CitationFull'.")

    if len(dataset) == 0:
        raise RuntimeError(f"Loaded empty dataset for source={source}, name={name}")
    data = dataset[0]

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

    return G, data, true_labels
