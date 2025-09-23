"""Adapted from: https://github.com/omarfoq/knn-per/tree/main"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class SubMedMNIST(Dataset):
    """
    Construct a subset of a MedMNIST dataset (e.g. OrganSMNIST) from a pickle file (which stores a list of indices)
    """
    def __init__(self, path, mnist_data=None, mnist_targets=None):
        """
        :param path: path to .pkl file; expected to store a list of indices
        :param mnist_data: (np.ndarray) MedMNIST dataset inputs 
        :param mnist_targets: (np.ndarray) MedMNIST dataset labels
        """
        with open(path, "rb") as f:
            indices = pickle.load(f)

        if mnist_data is not None and mnist_targets is not None:
            self.data = torch.tensor(mnist_data[indices], dtype=torch.float32)  
            self.targets = torch.tensor(mnist_targets[indices], dtype=torch.int64)  
        else:
            raise NotImplementedError("Loading from file path is not implemented in this example.")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index], self.targets[index], index


class SubCAMELYON17(Dataset):
    """
    Construct a subset of the Camelyon17 dataset from a precomputed client split (stored in a .npz file)
    """
    def __init__(self, path):
        """
        :param path: path to .pkl file; expected to store a list of indices
        """
        split_data = np.load(path)
        self.data = torch.tensor(split_data['embeddings'], dtype=torch.float32)  
        self.targets = torch.tensor(split_data['labels'], dtype=torch.int64)  

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index], self.targets[index], index
        