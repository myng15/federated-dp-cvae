"""
Adapted from: https://github.com/omarfoq/knn-per/tree/main
"""

import os
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aggregator import CentralizedAggregator, NoCommunicationAggregator
from client import Client, PersonalizedClient
from learner import Learner
from models.linear import LinearLayer
from datasets import SubCAMELYON17, SubMedMNIST
from .constants import LOADER_TYPE, EXTENSIONS, N_CLASSES, NUM_WORKERS
from .optim import get_optimizer, get_lr_scheduler
from .metrics import accuracy

def seed_everything(seed=42):
    """
    Ensure reproducibility

    :param seed: (int) seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_embeddings(experiment_):
    """
    Load embeddings and labels (and metadata) from the precomputed .npz files

    :param experiment_: name of the experiment
    :return:
        - embeddings, labels: np.ndarray, np.ndarray: Or:
        - embeddings, labels, metadatas: np.ndarray, np.ndarray, np.ndarray (for Camelyon17)
    """
    raw_data_path = "data/" + experiment_ + "/database/"

    train_data = np.load(os.path.join(raw_data_path, 'train.npz'))
    test_data = np.load(os.path.join(raw_data_path, 'test.npz'))
    train_embeddings = train_data['embeddings']
    test_embeddings = test_data['embeddings']

    type_ = LOADER_TYPE[experiment_]
    if type_ == "medmnist":
        train_labels = train_data['labels'].squeeze().astype(int)
        test_labels = test_data['labels'].squeeze().astype(int)
    else:
        train_labels = train_data['labels']
        test_labels = test_data['labels']
    
    embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)

    if type_ == "camelyon17":
        train_metadata = train_data['metadata']
        test_metadata = test_data['metadata']
        metadatas = np.concatenate([train_metadata, test_metadata], axis=0)
        return embeddings, labels, metadatas

    return embeddings, labels

def get_data_dir(experiment_name):
    """
    Get path to the per-client data split directories for an experiment
    """
    return os.path.join("data", experiment_name, "all_clients_data")

def get_loader(type_, path, batch_size, train, inputs=None, targets=None):
    """
    Construct a DataLoader from the given path

    :param type_: loader type of the dataset, possible are `medmnist` and `camelyon17`
    :param path: path to the data file
                - camelyon17: .npz file with data arrays {embeddings, labels}
                - medmnist (e.g. OrganSMNIST): .pkl file with data indices
    :param batch_size: batch size for the DataLoader
    :param train: flag indicating if train loader or test loader
    :param inputs: tensor storing the input data; default is None
    :param targets: tensor storing the labels; default is None
    :return: torch.utils.DataLoader
    """
    if type_ == "camelyon17": 
        dataset = SubCAMELYON17(path) 
    elif type_ == "medmnist":
        dataset = SubMedMNIST(path, mnist_data=inputs, mnist_targets=targets)
    else:
        raise NotImplementedError(f"{type_} not recognized type; possible are: {list(LOADER_TYPE.values)}")

    if len(dataset) == 0:
        return

    # drop last batch
    drop_last = (len(dataset) > batch_size) and train

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last, num_workers=NUM_WORKERS) 

def get_loaders(experiment_, data_dir, batch_size, is_validation):
    """
    Construct lists of DataLoaders (train, val, test) for each client in `data_dir`,
     corresponding to `train_iterator`, `val_iterator` and `test_iterator`;
     `val_iterator` iterates on the same dataset as `train_iterator`, the difference is only in drop_last

    :param experiment_: name of the experiment
    :param data_dir: directory of the data folder
    :param batch_size: batch size for the DataLoaders
    :param is_validation: (bool) if `True` validation set is used as test set
    :return:
        - train_iterators, val_iterators, test_iterators: 
        List[torch.utils.DataLoader], List[torch.utils.DataLoader], List[torch.utils.DataLoader]
    """
    type_ = LOADER_TYPE[experiment_]
    if type_ == "medmnist": 
        inputs, targets = load_embeddings(experiment_) # For MedMNIST datasets like OrganSMNIST
    else:
        inputs, targets = None, None # For Camelyon17, which loads data from .npz files directly

    train_iterators, val_iterators, test_iterators = [], [], []

    for client_dir in tqdm(os.listdir(data_dir)):
        client_data_path = os.path.join(data_dir, client_dir)

        train_iterator = get_loader(
            type_=type_,
            path=os.path.join(client_data_path, "train.npz") if type_ == "camelyon17" 
                    else os.path.join(client_data_path, f"train{EXTENSIONS[type_]}"),
            batch_size=batch_size,
            train=True,
            inputs=inputs,
            targets=targets
        )

        val_iterator = get_loader(
            type_=type_, 
            path=os.path.join(client_data_path, "val.npz") if type_ == "camelyon17" 
                    else os.path.join(client_data_path, f"train{EXTENSIONS[type_]}"),
            batch_size=batch_size,
            train=False,
            inputs=inputs,
            targets=targets
        )

        test_set = "val" if is_validation else "test"

        test_iterator = get_loader(
            type_=type_,
            path=os.path.join(client_data_path, "test.npz") if type_ == "camelyon17" 
                    else os.path.join(client_data_path, f"{test_set}{EXTENSIONS[type_]}"),
            batch_size=batch_size,
            train=False,
            inputs=inputs,
            targets=targets
        )

        train_iterators.append(train_iterator)
        val_iterators.append(val_iterator)
        test_iterators.append(test_iterator)

    return train_iterators, val_iterators, test_iterators

def get_model(name, device, input_dimension=None, chkpt_path=None):
    """
    Create a classifier model and optionally load weights from a checkpoint

    :param name: experiment's name
    :param device: either cpu or cuda
    :param input_dimension: input dimension of the model
    :param chkpt_path: path to checkpoint; if specified the weights of the model are initialized from checkpoint,
                        otherwise the weights are initialized randomly; default is None.
    :return: model
    """
    model = LinearLayer(
        input_dimension=input_dimension,
        num_classes=N_CLASSES[name]
    ).to(device)

    if chkpt_path is not None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        chkpt = torch.load(chkpt_path, map_location=map_location, weights_only=True)
        try:
            model.load_state_dict(chkpt['model_state_dict'])
        except KeyError:
            try:
                model.load_state_dict(chkpt['net'])
            except KeyError:
                model.load_state_dict(chkpt)
    
    return model

def get_client(
    client_type,
    learner,
    train_iterator,
    val_iterator,
    test_iterator,
    local_steps=None,
    client_id=None,
    save_path=None,
    args=None,
    features_dimension=None,
):
    """
    Instantiate a client of the specified type
    """
    if client_type == "personalized":
        return PersonalizedClient(
            learner=None,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            id_=client_id,
            args_=args,
            features_dimension=features_dimension,
            num_classes=N_CLASSES[args.experiment],
            device=args.device,
            seed=args.seed
        )
    else:
        return Client(
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            local_steps=local_steps,
            id_=client_id,
            save_path=save_path
        )

def get_learner(
    name,
    device,
    optimizer_name,
    scheduler_name,
    initial_lr,
    n_rounds,
    algorithm=None,
    input_dimension=None,
    chkpt_path=None,
):
    """
    Construct the learner corresponding to an experiment for a given seed

    :param name: name of the experiment; 
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param n_rounds: number of training rounds
    :param algorithm: name of the federated learning algorithm (e.g. "fedprox"); default is None
    :param input_dimension: input dimension of the model
    :param chkpt_path: path to chkpts; if specified the weights of the model are initialized from chkpts,
            otherwise the weights are initialized randomly; default is None.
    :return: Learner
    """
    criterion = nn.CrossEntropyLoss(reduction="none").to(device)
    metric = accuracy

    model = get_model(
        name=name,
        device=device,
        chkpt_path=chkpt_path,
        input_dimension=input_dimension
    )

    optimizer = get_optimizer(
        optimizer_name=optimizer_name,
        model=model,
        lr_initial=initial_lr
    )

    lr_scheduler = get_lr_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        n_rounds=n_rounds
    )

    return Learner(
        model=model,
        criterion=criterion,
        metric=metric,
        device=device,
        optimizer=optimizer,
        algorithm=algorithm, 
        lr_scheduler=lr_scheduler
    )

def get_aggregator(
    aggregator_type,
    clients,
    algorithm=None,
    global_learner=None,
    sampling_rate=None,
    log_freq=None,
    test_clients=None,
    verbose=None,
    seed=None
):
    """
    Instantiate an aggregator of the specified type, used by eval_fedavg.py
    """
    if aggregator_type == "local":
        return NoCommunicationAggregator(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "centralized": 
        return CentralizedAggregator(
            clients=clients,
            algorithm=algorithm,
            global_learner=global_learner,
            log_freq=log_freq,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    else:
        raise NotImplementedError(
            f"{aggregator_type} is not available; possible are: `local` and `centralized`."
        )


