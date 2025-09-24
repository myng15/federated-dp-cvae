import os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from utils.utils import get_client, get_learner, get_loaders
from utils.constants import EMBEDDING_DIM


def init_clients(args_, data_dir, chkpts_dir, features_dimension=None, for_ours=False):
    """
    Initialize clients from data folders

    :param args_: Parsed arguments
    :param data_dir: Path to data directory
    :param chkpts_dir: Directory to save checkpoints
    :param features_dimension: Optional, used for eval_ours.py
    :param for_ours: If True, use eval_ours.py logic
    :return: List[Client]
    """
    os.makedirs(chkpts_dir, exist_ok=True)
    clients_ = []

    # FOR DATASETS WITH NATURAL DOMAINS
    if args_.experiment in ["camelyon17"]:
        train_loaders, val_loaders, test_loaders = get_loaders(
            experiment_=args_.experiment,
            data_dir=data_dir,
            batch_size=args_.bz,
            is_validation=(not for_ours and args_.validation),
        )
        num_clients = len(train_loaders)
        
        # Initialize a Client object for every `train loader` and `test loader`
        for client_id, (train_loader, val_loader, test_loader) in enumerate(tqdm(
            zip(train_loaders, val_loaders, test_loaders), 
            total=num_clients
        )):
            if train_loader is None or test_loader is None:
                continue

            if args_.verbose > 0:
                print(f"[Client ID: {client_id}] N_Train: {len(train_loader.dataset)} | N_Val: {len(val_loader.dataset)} | N_Test: {len(test_loader.dataset)}")

            if not for_ours:
                learner = get_learner(
                    name=args_.experiment,
                    device=args_.device,
                    optimizer_name=args_.optimizer,
                    scheduler_name=args_.lr_scheduler,
                    initial_lr=args_.fedavg_lr,
                    n_rounds=args_.n_rounds,
                    algorithm=args_.algorithm,
                    input_dimension=EMBEDDING_DIM[args_.backbone], 
                )
                client = get_client(
                    client_type=args_.client_type,
                    learner=learner,
                    train_iterator=train_loader,
                    val_iterator=val_loader,
                    test_iterator=test_loader,
                    local_steps=args_.local_steps,
                    client_id=client_id,
                    save_path=os.path.join(chkpts_dir, f"client_{client_id}.pt")
                )

            else:
                client = get_client(
                    client_type=args_.client_type, 
                    learner=None,
                    train_iterator=train_loader,
                    val_iterator=val_loader,
                    test_iterator=test_loader,
                    client_id=client_id,
                    args=args_,
                    features_dimension=features_dimension
                )
                if client.n_train_samples == 0 or client.n_test_samples == 0:
                    continue
                client.load_all_features_and_labels()

            clients_.append(client)

    # FOR DATASETS WITH ARTIFICIAL CLIENT SPLITS
    else: 
        if not for_ours:
            train_loaders, _, test_loaders = get_loaders(
                experiment_=args_.experiment,
                data_dir=data_dir,
                batch_size=args_.bz,
                is_validation=args_.validation,
            )
        
        else:
            _, train_loaders, test_loaders = get_loaders(
                experiment_=args_.experiment, 
                data_dir=data_dir, 
                batch_size=args_.bz,
                is_validation=False,
            )

        num_clients = len(train_loaders)

        # Initialize a Client object for every `train loader` and `test loader`
        for client_id, (train_loader, test_loader) in enumerate(tqdm(
            zip(train_loaders, test_loaders), 
            total=num_clients
        )):
            if train_loader is None or test_loader is None:
                continue
            dataset = train_loader.dataset

            # Group indices by label
            label_to_indices = defaultdict(list)
            for idx in range(len(dataset)):
                _, label, _ = dataset[idx]
                label_to_indices[label.item()].append(idx)

            train_indices = []
            val_indices = []

            # Ensure deterministic and stratified split per label
            for label, indices in label_to_indices.items():
                np.random.seed(args_.seed + label)  # Unique seed per label for deterministic split
                np.random.shuffle(indices)

                if len(indices) == 1:
                    train_indices.extend(indices)  # If only one sample with this label, put it in training
                else:
                    split_point = int((1 - args_.val_frac) * len(indices))
                    train_indices.extend(indices[:split_point])
                    val_indices.extend(indices[split_point:])

            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_loader.batch_size)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_loader.batch_size)

            if args_.verbose > 0:
                print(f"[Client ID: {client_id}] N_Train: {len(train_loader.dataset)} | N_Val: {len(val_loader.dataset)} | N_Test: {len(test_loader.dataset)}")
            
            if not for_ours:
                learner = get_learner(
                    name=args_.experiment,
                    device=args_.device,
                    optimizer_name=args_.optimizer,
                    scheduler_name=args_.lr_scheduler,
                    initial_lr=args_.fedavg_lr,
                    n_rounds=args_.n_rounds,
                    algorithm=args_.algorithm,
                    input_dimension=EMBEDDING_DIM[args_.backbone], 
                )
                client = get_client(
                    client_type=args_.client_type,
                    learner=learner,
                    train_iterator=train_loader,
                    val_iterator=val_loader,
                    test_iterator=test_loader,
                    local_steps=args_.local_steps,
                    client_id=client_id,
                    save_path=os.path.join(chkpts_dir, f"client_{client_id}.pt")
                )
            
            else:
                client = get_client(
                    client_type=args_.client_type, 
                    learner=None,
                    train_iterator=train_loader,
                    val_iterator=val_loader,
                    test_iterator=test_loader,
                    client_id=client_id,
                    args=args_,
                    features_dimension=features_dimension, 
                )
                if client.n_train_samples == 0 or client.n_test_samples == 0:
                    continue
                client.load_all_features_and_labels()

            clients_.append(client)

    return clients_

            



        