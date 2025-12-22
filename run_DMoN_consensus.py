#!/usr/bin/env python3
"""
Run DMoN R times on a real graph loaded via load_real_graph() and save:
  - agreement matrix D:  D_<name>_k{k}_R{R}.npy
  - labels matrix (R x n): D_<name>_k{k}_R{R}_labels.npy

Usage (example):
  python run_DMoN_consensus.py --name Wiki --k 53 --R 50 --epochs 500 --out D_Wiki_k53.npy

Author: Dalya Manatova with assistance from Copilot (GPT-5)
"""

import argparse, time, os, numpy as np, torch
from load_real_network import load_real_graph 


# -----------------------------
# One DMoN run → labels
# -----------------------------
def run_dmon_once(data, k, lr=1e-3, weight_decay=5e-4,
                  hidden=64, num_layers=2, dropout=0.5, gcn_skip=True,
                  collapse_regularization=1.0, epochs=500, device=None, seed=None):
    import dmon  

    if seed is not None:
        import random
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    in_channels = x.size(1)

    model = dmon.DMoN(
        in_channels=in_channels,
        num_clusters=k,
        hidden_channels=hidden,
        num_layers=num_layers,
        dropout=dropout,
        gcn_skip=gcn_skip,
        collapse_regularization=collapse_regularization,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        ca, loss = model(x, edge_index)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        ca, _ = model(x, edge_index)
        labels = ca.argmax(dim=1).cpu().numpy().astype(int)
    return labels


# -----------------------------------
# consensus matrix
# -----------------------------------
def agreement_matrix_from_labels(labels_matrix: np.ndarray) -> np.ndarray:
    """
    labels_matrix: shape [R, n], ints
    returns D: shape [n, n], float in [0,1]
    """
    R, n = labels_matrix.shape
    D = np.zeros((n, n), dtype=np.float32)
    for r in range(R):
        lab = labels_matrix[r]
        for c in np.unique(lab):
            if c < 0:  # skip noise labels if any
                continue
            idx = np.where(lab == c)[0]
            D[np.ix_(idx, idx)] += 1.0
    np.fill_diagonal(D, R)
    D /= float(R)
    return D


# -----------------------------------
# Consensus clustering from agreement matrix D
# -----------------------------------
def consensus_labels(D, tau=0.1, reps=100):
    try:
        from brainconn.clustering import consensus_und
    except ImportError as e:
        raise RuntimeError(
            "brainconn not installed in GPU environment.\n"
            "Run:\n"
            "  module load python/gpu/3.10.10\n"
            "  pip install --user git+https://github.com/fiuneuro/brainconn.git"
        ) from e

    return consensus_und(D, tau=tau, reps=reps)

# -----------------------------------------------------
# Run DMoN R times at fixed k → labels_matrix & D
# -----------------------------------------------------
def dmon_consensus_agreement(data, k, R=50, seeds=None, tau=0.3, reps=100, **train_kwargs):
    n = data.num_nodes
    labels_runs = np.zeros((R, n), dtype=int)
    
    if seeds is None:
        seeds = list(range(R))
    for i in range(R):
        t0 = time.time()
        labels_runs[i] = run_dmon_once(data, k, seed=seeds[i], **train_kwargs)
        print(f"[{i+1}/{R}] elapsed {time.time()-t0:.1f}s", flush=True)
    D = agreement_matrix_from_labels(labels_runs)
    labels_consensus = consensus_labels(D, tau=tau, reps=reps)
    return labels_runs, D, labels_consensus


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, required=True, help="Dataset name for load_real_graph (e.g., Wiki)")
    p.add_argument("--k", type=int, required=True, help="Number of clusters for DMoN")
    p.add_argument("--R", type=int, default=50, help="Number of independent DMoN runs")
    p.add_argument("--tau", type=float, default=0.3, help="Tau parameter for consensus clustering")
    p.add_argument("--reps", type=int, default=100, help="Number of repetitions for consensus clustering")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--out", type=str, default=None, help="Output filename for D (npy). If omitted, auto-named.")
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | None(auto)")
    args = p.parse_args()

    # Load real graph (we ignore true_labels here)
    G, data, _ = load_real_graph(name=args.name)
    print(f"Loaded graph '{args.name}': n={data.num_nodes}, m={data.edge_index.size(1)}")

    device = torch.device(args.device) if args.device else None

    labels_matrix, D, labels_consensus = dmon_consensus_agreement(
        data,
        k=args.k,
        R=args.R,
        tau=args.tau,
        reps=args.reps,
        epochs=args.epochs,
        device=device,
    )
    print("Total number of consenus clusters:", len(set(labels_consensus)))
    # Save outputs
    out_D = args.out or f"D_{args.name}_k{args.k}_R{args.R}.npy"
    out_labels = out_D.replace(".npy", "_labels.npy")
    out_consensus = out_D.replace(f"R{args.R}.npy", f"tau{args.tau}_consensus.npy")
    np.save(out_D, D)
    np.save(out_labels, labels_matrix)
    np.save(out_consensus, labels_consensus)
    print(f"Saved D to {out_D}")
    print(f"Saved labels_matrix to {out_labels}")
    print(f"Saved labels_consensus to {out_consensus}")

if __name__ == "__main__":
    main()
