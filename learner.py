"""Adapted from: https://github.com/omarfoq/knn-per/tree/main"""

import torch

from utils.metrics import safe_balanced_accuracy


class Learner:
    """
    Responsible of training and evaluating a (deep-)learning model

    :param model: (nn.Module) the model trained by the learner
    :param criterion: (torch.nn.modules.loss) loss function used to train the `model`, should have reduction="none"
    :param metric: (fn) function to compute the metric, should accept as input two vectors and return a scalar
    :param device: (str or torch.device)
    :param optimizer: (torch.optim.Optimizer)
    :param algorithm: (str or None) if "fedprox", add a proximal term to the loss function
    :param global_model: (nn.Module or None) if algorithm == "fedprox", the global model to compute the proximal term
    :param lr_scheduler: (torch.optim.lr_scheduler or None)
    :param mu: (float) FedProx regularization weight
    """
    def __init__(
            self,
            model,
            criterion,
            metric,
            device,
            optimizer,
            algorithm=None, 
            global_model=None, 
            lr_scheduler=None,
            mu=0.05 
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer
        self.algorithm = algorithm 
        self.lr_scheduler = lr_scheduler
        self.global_model = global_model 
        self.mu = mu  

        self.is_ready = True

    def fit_epoch(self, iterator, weights=None, frozen_modules=None):
        """
        Perform one epoch on all batches drawn from `iterator`.
        Add a proximal term if algorithm == "fedprox".

        :param iterator: (DataLoader) iterator providing data batches
        :param weights: (torch.tensor or None) tensor with the learners' weights of each sample or None
        :param frozen_modules: list of frozen models; default is None
        :return: global_loss and global_metric over the epoch
        """
        if frozen_modules is None:
            frozen_modules = []

        self.model.train()

        global_loss = 0.0
        global_metric = 0.0
        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            n_samples += y.size(0)

            self.optimizer.zero_grad()

            y_pred = self.model(x)
            loss_vec = self.criterion(y_pred, y)
            
            if weights is not None:
                weights = weights.to(self.device)
                loss = (loss_vec.T @ weights[indices]) / loss_vec.size(0)
            else:
                loss = loss_vec.mean()

            if self.algorithm == "fedprox":
                # Add FedProx regularization term
                self.global_model = self.global_model.to(self.device)
                prox_term = sum(
                    (param - global_param).norm(2) ** 2 
                    for param, global_param in zip(self.model.parameters(), self.global_model.parameters())
                )
                loss += (self.mu) * prox_term 

            loss.backward()

            for frozen_module in frozen_modules:
                frozen_module.zero_grad()

            self.optimizer.step()

            global_loss += loss.item() * loss_vec.size(0)
            global_metric += self.metric(y_pred, y).item()

        return global_loss / n_samples, global_metric / n_samples

    def fit_epochs(self, iterator, n_epochs, weights=None, frozen_modules=None):
        """
        Perform multiple training epochs

        :param iterator:(DataLoader) iterator providing data batches
        :param n_epochs: (int) number of epochs
        :param weights: (torch.tensor or None) tensor with the learners_weights of each sample or None
        :param frozen_modules: list of frozen models; default is None
        """
        for step in range(n_epochs):
            self.fit_epoch(iterator, weights, frozen_modules=frozen_modules)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def evaluate_iterator(self, iterator):
        """
        Evaluate learner on `iterator`

        :param iterator: (DataLoader) iterator providing data batches
        :return: global_loss, global_metric and global_balanced_accuracy over the iterator
        """
        self.model.eval()

        global_loss = 0.0
        global_metric = 0.0
        n_samples = 0

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for x, y, _ in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)

                y_pred = self.model(x)

                global_loss += self.criterion(y_pred, y).sum().item()
                global_metric += self.metric(y_pred, y).item()

                _, predicted = torch.max(y_pred, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

                n_samples += y.size(0)

        global_balanced_accuracy = safe_balanced_accuracy(all_labels, all_preds)
        
        return global_loss / n_samples, global_metric / n_samples, global_balanced_accuracy

    def free_memory(self):
        """
        Free the memory allocated by the model weights
        """
        if not self.is_ready:
            return

        self.optimizer.zero_grad(set_to_none=True)
        del self.lr_scheduler
        del self.optimizer
        del self.model
        self.is_ready = False

    def save_checkpoint(self, path):
        """
        Save the model, the optimizer and the learning rate scheduler

        :param path: path to a `.pt` file
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }

        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """
        Load the model, the optimizer and the learning rate scheduler

        :param path: path to a `.pt` file storing the checkpoint
        """
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if "scheduler_state_dict" in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])