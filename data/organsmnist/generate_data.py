"""Adapted from: https://github.com/omarfoq/knn-per/tree/main"""

import os
import argparse
import pickle
import random
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

import sys
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(os.path.dirname(current))
sys.path.append(root)

from data.organsmnist.utils_data import *
from utils.utils import load_embeddings, seed_everything
 
PATH = os.path.join(root, "data/organsmnist/all_clients_data/") 
N_CLASSES = 11


def save_data(data, path_):
    with open(path_, 'wb') as f:
        pickle.dump(data, f)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Split OrganSMNIST embeddings among n_clients (IID/non-IID). '
                    'Can be used with `pathological_split` or `by_labels_split` for different methods of non-IID splits.'
    )
    parser.add_argument(
        '--n_clients',
        help='number of clients',
        type=int,
        required=True
    )
    parser.add_argument(
            "--split_method",
            help='method to be used to split data among n_clients;'
                 ' possible are "iid", "by_labels_split" and "pathological_split";'
                 ' 1) "by_labels_split": the dataset will be split as follow:'
                 '  a) classes are grouped into `n_clusters`'
                 '  b) for each cluster `c`, samples are partitioned across clients using Dirichlet distribution'
                 ' Inspired by "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440);'
                 ' 2) "pathological_split": the dataset will be split as follow:'
                 '  a) sort the data by label'
                 '  b) divide it into `n_clients * n_classes_per_client` shards, of equal size.'
                 '  c) assign each of the `n_clients` with `n_classes_per_client` shards'
                 ' Similar to "Communication-Efficient Learning of Deep Networks from Decentralized Data"'
                 ' __(https://arxiv.org/abs/1602.05629);'
                 'default is "iid"',
            type=str,
            default="iid"
    )
    parser.add_argument(
        '--n_shards',
        help='number of shards given to each client/task; ignored if `pathological_split` is not used;',
        type=int,
        default=2
    )
    parser.add_argument(
        '--n_components',
        help='number of components/clusters; ignored if `by_labels_split` is not used',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--alpha',
        help='Dirichlet parameter controlling clients\' dissimilarity, the smaller alpha is the more clients\' data distributions are dissimilar;'
             'ignored if `by_labels_split` is not used',
        type=float,
        default=0.5
    )
    parser.add_argument(
        '--s_frac',
        help='fraction of the dataset to be used',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--test_clients_frac',
        help='fraction of clients held out as test clients',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--test_data_frac',
        help='fraction of each client\'s data for test',
        type=float,
        default=0.2
    )
    parser.add_argument(
        '--seed',
        help='seed for reproducibility',
        type=int,
        default=12345
    )
    return parser.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)

    embeddings, labels = load_embeddings("organsmnist")
    dataset = list(zip(embeddings, labels)) 

    if args.split_method == "pathological_split":
        all_clients_indices = pathological_non_iid_split(
            dataset=dataset,
            n_classes=N_CLASSES,
            n_clients=args.n_clients,
            n_classes_per_client=args.n_shards,
            frac=args.s_frac,
            seed=args.seed
        )
    elif args.split_method == "by_labels_split":
        all_clients_indices = by_labels_non_iid_split(
            dataset=dataset,
            n_classes=N_CLASSES,
            n_clients=args.n_clients,
            n_clusters=args.n_components,
            alpha=args.alpha,
            frac=args.s_frac,
            seed=args.seed
        )
    elif args.split_method == "iid":
        all_clients_indices = iid_split(
            dataset=dataset,
            n_clients=args.n_clients,
            frac=args.s_frac,
            seed=args.seed
        )
    else:
        raise ValueError(f"Invalid split method: {args.split_method}")

    if args.test_clients_frac > 0:
        train_clients_indices, test_clients_indices = train_test_split(
            all_clients_indices, test_size=args.test_clients_frac, random_state=args.seed
        )
    else:
        train_clients_indices, test_clients_indices = all_clients_indices, []

    os.makedirs(os.path.join(PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(PATH, "test"), exist_ok=True)

    for mode, clients_indices in [('train', train_clients_indices), ('test', test_clients_indices)]:
        for client_id, indices in enumerate(clients_indices):
            indices = np.array(indices)

            # Perform stratified train-test split
            rng = random.Random(args.seed)  
            np.random.seed(args.seed)
            rng.shuffle(indices)

            # Group indices by label
            label_to_indices = defaultdict(list)
            for idx in indices:
                label = labels[idx].item()
                label_to_indices[label].append(idx)

            # Split indices per label and aggregate train-test splits
            train_indices = []
            test_indices = []
 
            for label, label_indices in label_to_indices.items():
                rng.shuffle(label_indices)
                if len(label_indices) == 1:
                    train_indices.extend(label_indices)  # Only one sample with this label, put it in training
                else:
                    split_point = int((1 - args.test_data_frac) * len(label_indices))
                    train_indices.extend(label_indices[:split_point])
                    test_indices.extend(label_indices[split_point:])
            
            if not train_indices or not test_indices:
                continue
        
            # Save train and test indices for each client
            client_path = os.path.join(PATH, mode, "client_{}".format(client_id))
            os.makedirs(client_path, exist_ok=True)
            save_data(train_indices, os.path.join(client_path, "train.pkl"))
            save_data(test_indices, os.path.join(client_path, "test.pkl"))

            print(f"[{mode} mode] client_{client_id}: train/val/test splits saved.")
            
if __name__ == "__main__":
    main()