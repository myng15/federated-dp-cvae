"""
Adapted from: https://github.com/francescodisalvo05/cvae-anonymization/blob/main/src/anonymizer/cvae.py 
"""

import os
from collections import Counter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from opacus import GradSampleModule

from models.cvae import CVAE, CVAETrainer
from utils.constants import EMBEDDING_DIM, N_CLASSES


class CVAEAnonymizer:
    def __init__(self, args, g):
        """
        :param args: Namespace containing parameters and hyperparameters of the current run
        :param g: torch.Generator for reproducibility
        """
        self.args = args
        self.g = g

        self.device = self.args.device 
        self.input_dim = EMBEDDING_DIM[self.args.backbone] 
        self.hidden_dim = 256
        self.latent_dim = 100 
        self.num_classes = N_CLASSES[self.args.experiment] 
        self.lr = self.args.lr
        self.beta = self.args.cvae_beta
        self.var = self.args.cvae_var

    def apply(self, client):
        """
        Anonymize the data of a client by generating synthetic samples using a trained CVAE
        
        :param client: (Client object) client whose data is to be anonymized
        :return gen_samples, gen_labels: generated samples and their corresponding labels
        """
        data, labels = client.train_features, client.train_labels
        val_data, val_labels = client.val_features, client.val_labels
        train_loader, val_loader = self.prepare_data(data, labels, val_data, val_labels)

        base_dir = os.path.join(self.args.chkpts_dir , "cvae_fedavg") 
        os.makedirs(base_dir, exist_ok=True)
        best_chkpt = os.path.join(base_dir, f'client_{client.id}.pt') 
        
        cvae_model = CVAE(
                input_dim=self.input_dim, 
                hidden_dim = self.hidden_dim,
                latent_dim = self.latent_dim,
                condition_dim = self.num_classes,
                device = self.device
        )
        trainer = CVAETrainer(
                model=cvae_model, 
                args=self.args, 
                device=self.device, 
                chkpt_path=best_chkpt, 
                n_classes=self.num_classes, 
                lr=self.lr,
                beta=self.beta
        )

        if not os.path.isfile(best_chkpt): 
            trainer.fit(train_loader, val_loader, num_epochs=100, lr_scheduler="multi_step")
        
        trainer.load_checkpoint() # load best checkpoint
        
        if isinstance(trainer.model, GradSampleModule):
            model = trainer.model._module
        else:
            model = trainer.model

        gen_samples, gen_labels = self.generate_samples(
            model=model, 
            real_labels=labels, 
            train_min=self.train_min, 
            train_max=self.train_max, 
            var=self.var, 
            generated_factor=self.args.generated_factor
        )

        return gen_samples, gen_labels

    def prepare_data(self, data, labels, val_data, val_labels):
        """
        Prepare data loaders for anonymizer training and validation
        """
        data, labels = torch.Tensor(data).to(self.device), torch.Tensor(labels).to(self.device).to(int)
        val_data, val_labels = torch.Tensor(val_data).to(self.device), torch.Tensor(val_labels).to(self.device).to(int)

        data, train_min, train_max = self._normalize(data)
        val_data, _, _ = self._normalize(val_data, train_min, train_max)
        self.train_min = train_min
        self.train_max = train_max

        train_dataset = TensorDataset(data, labels)
        val_dataset = TensorDataset(val_data, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, generator=self.g) 
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, generator=self.g) 
        
        return train_loader, val_loader

    def _get_samples_per_class(self, labels, factor=1.0):
        label_counts = Counter(labels)
        samples_per_class = torch.zeros(self.num_classes, dtype=torch.long)

        for label, count in label_counts.items():
            samples_per_class[label] = int(count * factor)

        return samples_per_class

    def generate_samples(self, model, real_labels, train_min, train_max, var=1.0, generated_factor=1.0):
        """
        Generate synthetic samples based on the size and label distribution of the real data

        :param model: trained CVAE model
        :param real_labels: labels of the real data
        :param train_min: minimum value for un-normalizing
        :param train_max: maximum value for un-normalizing
        :param var: for controlling noise variance
        """
        gen_samples, gen_labels = [], []
        samples_per_class = self._get_samples_per_class(real_labels, generated_factor)

        model.eval()  
        model.decoder.to(self.device) 

        with torch.no_grad():
            all_labels = []
            all_latents = []

            for label in range(self.num_classes):
                n_samples = samples_per_class[label].item()
                if n_samples <= 0:
                    continue
                all_labels.append(torch.full((n_samples,), label, dtype=torch.long, device=self.device))
                all_latents.append(var * torch.randn((n_samples, self.latent_dim), device=self.device))

            all_labels = torch.cat(all_labels, dim=0)  
            all_latents = torch.cat(all_latents, dim=0)  
            one_hot_labels = F.one_hot(all_labels, num_classes=self.num_classes).to(self.device)

            x_hat = model.decode(all_latents, one_hot_labels).to(self.device)
            x_hat = self._un_normalize(x_hat, train_min, train_max)

            gen_samples = x_hat.cpu().numpy()
            gen_labels = all_labels.cpu().numpy().tolist()

        return gen_samples, gen_labels

    def _normalize(self, tensor, _min = None, _max = None):
        """
        Normalize tensor

        :param tensor: embedding to normalize
        :param _min: minimum value for normalization
        :param _max: maximum value for normalization
        :return normalized tensor
        """
        if _min == None and _max == None:
            _min = tensor.min(dim=0).values
            _max = tensor.max(dim=0).values
        return (tensor - _min) / (_max - _min), _min, _max

    def _un_normalize(self, tensor, _min, _max):
        """
        Un-normalize tensor

        :param tensor: embedding to normalize
        :param _min: minimum value for un-normalizing
        :param _max: maximum value for un-normalizing
        :return un-normalized tensor
        """
        return (tensor) * (_max - _min) + _min

    def get_trainer(self, client, global_model=None, is_trained=True, num_epochs=None):
        """
        Get a CVAETrainer for either a client (local) or the global model - necessary for the FedAvg training of DP-CVAE
        """
        base_dir = os.path.join(self.args.chkpts_dir, "cvae_fedavg")
        os.makedirs(base_dir, exist_ok=True)
        best_chkpt = os.path.join(base_dir, f"client_{client.id}.pt") if client else os.path.join(base_dir, "global.pt")
        
        if not is_trained:
            cvae_model = CVAE(
                input_dim=self.input_dim, 
                hidden_dim = self.hidden_dim,
                latent_dim = self.latent_dim,
                condition_dim = self.num_classes,
                device = self.device
            )
            trainer = CVAETrainer(
                model=cvae_model, 
                args=self.args, 
                device=self.device, 
                chkpt_path=best_chkpt, 
                n_classes=self.num_classes, 
                num_epochs=num_epochs,
                lr=self.lr,
                beta=self.beta
            )

            return trainer

        train_loader, val_loader = self.prepare_data(client.train_features, client.train_labels, client.val_features, client.val_labels)
        client.trainer.fit(train_loader, val_loader, global_model=global_model, num_epochs=num_epochs, lr_scheduler=None) 

        return client.trainer
    