import sys
import argparse
from consensus_clustering import compute_dataset_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', help='Name of the real network dataset')
    args = parser.parse_args()
    compute_dataset_metrics(args.dataset, csv_file="all_dataset_metrics_upd.csv")
