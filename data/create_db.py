"""
Adapted from: https://github.com/francescodisalvo05/cvae-anonymization/blob/main/create_db.py 
"""

import os
import pickle
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import timm

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from wilds import get_dataset
import medmnist
from medmnist import INFO

import sys
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)

from utils.utils import *


class CustomCamelyon(Dataset):
    def __init__(self, root, transform):
        self.dataset = get_dataset(dataset="camelyon17", root_dir=root, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label, metadata = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img) 
        return img, label, metadata

def _compute_hospital_indices(dataset, save_path):
    """
    Compute and save sample indices per hospital (metadata[0] is hospital_id 0..4)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    hospital_indices = {f"client_{i}": [] for i in range(5)}
    for idx in range(len(dataset)):
        _, _, metadata = dataset[idx]
        hospital_id = int(metadata[0])
        hospital_indices[f"client_{hospital_id}"].append(idx)

    with open(save_path, "wb") as f:
        pickle.dump(hospital_indices, f)
        print(f"Hospital indices successfully saved at {save_path}")

    return hospital_indices

def _load_or_build_hospital_indices(dataset, indices_dir):
    save_path = os.path.join(indices_dir, "hospital_indices.pkl")

    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            return pickle.load(f)

    return _compute_hospital_indices(dataset, save_path)

def get_dataloaders_by_hospitals(args, dataset):
    """
    Build a Dataloader per hospital (client)

    :return dataloaders: a dict containing dataloaders for the clients
    """
    indices_dir = os.path.abspath(args.indices_root)
    hospital_indices = _load_or_build_hospital_indices(dataset, indices_dir)

    # Create DataLoader for each hospital
    dataloaders = {}

    for hospital_id, indices in hospital_indices.items():
        subset = Subset(dataset, indices)
        dataloaders[hospital_id] = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    return dataloaders

def get_dataloaders(args, transforms):
    """
    Get the dataloaders for the selected dataset and backbone
    """
    data_path = args.dataset_root
    os.makedirs(data_path, exist_ok=True)

    if args.dataset == "camelyon17":
        dataset = CustomCamelyon(root=data_path, transform=transforms)
        return get_dataloaders_by_hospitals(args=args, dataset=dataset)

    if args.dataset in ["organsmnist"]: # For MedMNIST datasets 
        info = INFO[args.dataset] 
        DataClass = getattr(medmnist, info['python_class']) 
        trainset = DataClass(split='train', transform=transforms, download=False, as_rgb=True, size=224, root=data_path, mmap_mode='r')
        valset = DataClass(split='val', transform=transforms, download=False, as_rgb=True, size=224, root=data_path, mmap_mode='r')
        testset = DataClass(split='test', transform=transforms, download=False, as_rgb=True, size=224, root=data_path, mmap_mode='r')
        trainset = ConcatDataset([trainset, valset])
    else:
        raise ValueError(f"{args.dataset} not available")
    
    dataloaders = {}
    trainloader = DataLoader(trainset, batch_size = args.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    testloader = DataLoader(testset, batch_size = args.batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    dataloaders['train'] = trainloader
    dataloaders['test'] = testloader

    return dataloaders

def extract_embeddings(model, device, dataset, dataloader):
    """
    Run feature extractor using the chosen backbone and collect embeddings, labels and metadata (if available)

    :param model: delected backbone
    :param device: device to use (cpu|cuda)
    :param dataset: name of the dataset
    :param dataloader: current dataloader (train|test)
    :return data: dictionary containing the extracted data
    """

    embeddings_db, labels_db, metadata_db = [], [], []

    for batch in tqdm(dataloader):
        if dataset == "camelyon17":
            images, labels, metadata = batch
        else:
            images, labels = batch
        
        images = images.to(device)
        output = model.forward_features(images)
        output = model.forward_head(output, pre_logits=True)

        embeddings_db.extend(output.detach().cpu().numpy())
        labels_db.extend(labels)
        if dataset == "camelyon17":
            metadata_db.extend(metadata)
    
    data = {
        'embeddings': embeddings_db,
        'labels': labels_db
    }
    if dataset == "camelyon17":
        data['metadata'] = metadata_db

    return data

def main(args):
    seed_everything(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get model from timm
    model = timm.create_model(args.backbone, pretrained=True, num_classes=0).to(device)
    model.requires_grad_(False)
    model = model.eval()

    # Get the required transform function for the given feature extractor
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Get dataloaders
    dataloaders = get_dataloaders(args, transforms)
    os.makedirs(os.path.join(args.database_root), exist_ok=True)

    if args.dataset == "camelyon17":
        splits = ['client_0','client_1','client_2','client_3','client_4']
    else:
        splits = ['train','test']

    for split in splits:
        db = extract_embeddings(
            model = model, 
            device = device,
            dataset = args.dataset,
            dataloader = dataloaders[split]
        )
        
        # store database: e.g. database_root / train|test.npz
        np.savez(os.path.join(args.database_root,f'{split}.npz'), **db)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--dataset_root', type=str, required=True, help='path to raw dataset root') # e.g. dataset_root="data/organsmnist/dataset/"
    parser.add_argument('--database_root', type=str, required=True, help='output directory for the embedding database') # e.g. database_root="data/organsmnist/database/"
    parser.add_argument('--dataset', type=str, required=True, help='dataset name') # e.g. dataset="organsmnist"
    parser.add_argument('--indices_root', type=str, default='assets/data', help='path to `hospital_indices.pkl`') # for Camelyon17 
    parser.add_argument('--backbone', type=str, default='vit_base_patch14_dinov2.lvd142m', help='backbone for feature extraction') 
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for feature extraction')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    main(args)