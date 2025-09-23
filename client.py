import os
import random
import numpy as np
import time
import warnings
from scipy.special import softmax
import torch
import torch.nn as nn

from anonymizer.cvae import CVAEAnonymizer
from anonymizer.cgan import CGANAnonymizer
from models.linear import *
from utils.metrics import compute_classification_metrics
from utils.optim import get_optimizer


class Client(object):
    r"""
    Base class for Client
    Adapted from: https://github.com/omarfoq/knn-per/tree/main

    :param learner: (Learner object) responsible of training and evaluating a (deep-)learning model
    :param train_iterator: (DataLoader) training data iterator
    :param val_iterator: (DataLoader) validation data iterator
    :param test_iterator: (DataLoader) test data iterator
    :param local_steps: number of local training steps in each FedAvg round
    :param save_path: path to save the model
    :param id_: client ID
    """
    def __init__(
            self,
            learner,
            train_iterator,
            val_iterator,
            test_iterator,
            local_steps,
            save_path=None,
            id_=None
    ):
        self.learner = learner

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator

        self.n_train_samples = len(self.train_iterator.dataset)
        self.n_val_samples = len(self.val_iterator.dataset)
        self.n_test_samples = len(self.test_iterator.dataset)

        self.local_steps = local_steps
        self.save_path = save_path
        self.id = id_ if id_ is not None else -1 
        self.counter = 0

    def is_ready(self):
        return self.learner.is_ready

    def step(self):
        self.counter += 1
        self.learner.fit_epochs(
            iterator=self.train_iterator,
            n_epochs=self.local_steps,
        )

    def compute_metrics(self):
        train_loss, train_acc, train_balanced_acc = self.learner.evaluate_iterator(self.val_iterator)
        test_loss, test_acc, test_balanced_acc = self.learner.evaluate_iterator(self.test_iterator)
        return train_loss, train_acc, train_balanced_acc, test_loss, test_acc, test_balanced_acc

    def save_state(self, path=None):
        """
        :param path: expected to be a `.pt` file
        """
        if path is None:
            if self.save_path is None:
                warnings.warn("Client state was not saved", RuntimeWarning)
                return
            else:
                self.learner.save_checkpoint(self.save_path)
                return

        self.learner.save_checkpoint(path)

    def load_state(self, path=None):
        if path is None:
            if self.save_path is None:
                warnings.warn("Client state was not loaded", RuntimeWarning)
                return
            else:
                self.learner.load_checkpoint(self.save_path)
                return

        self.learner.load_checkpoint(path)

    def free_memory(self):
        self.learner.free_memory()


class PersonalizedClient(Client):
    def __init__(
            self, 
            learner,
            train_iterator, 
            val_iterator,
            test_iterator, 
            id_,
            args_,
            features_dimension, 
            num_classes,
            device,
            seed=1234,
            *args, 
            **kwargs
    ):
        super(PersonalizedClient, self).__init__(
            learner=learner,
            train_iterator=train_iterator,
            val_iterator=val_iterator,  
            test_iterator=test_iterator,
            id_=id_,
            local_steps=None,  
            *args, 
            **kwargs
        )

        self.args_ = args_
        self.device = device 
        self.seed = seed

        self.features_dimension = int(features_dimension)
        self.num_classes = int(num_classes)

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator

        self.n_train_samples = len(self.train_iterator.dataset)
        self.n_val_samples = len(self.val_iterator.dataset)
        self.n_test_samples = len(self.test_iterator.dataset)

        # Initialize local data
        self.train_features = np.zeros(shape=(self.n_train_samples, self.features_dimension), dtype=np.float32)
        self.train_labels = np.zeros(shape=self.n_train_samples, dtype=np.int64)
        self.val_features = np.zeros(shape=(self.n_val_samples, self.features_dimension), dtype=np.float32)
        self.val_labels = np.zeros(shape=self.n_val_samples, dtype=np.int64)
        self.test_features = np.zeros(shape=(self.n_test_samples, self.features_dimension), dtype=np.float32)
        self.test_labels = np.zeros(shape=self.n_test_samples, dtype=np.int64)

        # Initialize global data
        self.global_features = np.array([], dtype=np.float32)
        self.global_labels = np.array([], dtype=np.int64)

        # Initialize classifiers and outputs
        self.local_classifier = None
        self.global_classifier = None
        self.local_outputs = None 
        self.glocal_knn_outputs = None 

        self.trainer = None # For DP-CVAE and DP-CGAN

    def _seed_everything_client(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_all_features_and_labels(self):
        # Load train data
        train_features_list = []
        train_labels_list = []
        for batch_features, batch_labels, _ in self.train_iterator:
            train_features_list.append(batch_features.numpy())
            train_labels_list.append(batch_labels.numpy())
        self.train_features = np.concatenate(train_features_list, axis=0)
        self.train_labels = np.concatenate(train_labels_list, axis=0)

        # Load validation data
        val_features_list = []
        val_labels_list = []
        for batch_features, batch_labels, _ in self.val_iterator:
            val_features_list.append(batch_features.numpy())
            val_labels_list.append(batch_labels.numpy())
        self.val_features = np.concatenate(val_features_list, axis=0) 
        self.val_labels = np.concatenate(val_labels_list, axis=0)

        # Load test data
        test_features_list = []
        test_labels_list = []
        for batch_features, batch_labels, _ in self.test_iterator:
            test_features_list.append(batch_features.numpy())
            test_labels_list.append(batch_labels.numpy())
        self.test_features = np.concatenate(test_features_list, axis=0)
        self.test_labels = np.concatenate(test_labels_list, axis=0)

    def anonymize_data(self, train_features=None, train_labels=None, val_features=None, val_labels=None):
        self._seed_everything_client()

        print(f"Starting anonymization...")
        start = time.time()
        
        train_features = train_features if train_features is not None else self.train_features
        train_labels = train_labels if train_labels is not None else self.train_labels
        val_features = val_features if val_features is not None else self.val_features
        val_labels = val_labels if val_labels is not None else self.val_labels
        
        g = torch.Generator()
        g.manual_seed(self.seed)

        if self.args_.anonymizer == "cvae_fedavg": 
            anonymizer = CVAEAnonymizer(args=self.args_, g=g)
            anonymized_features, anonymized_labels = anonymizer.apply(client=self)
        elif self.args_.anonymizer == "cgan_fedavg": 
            anonymizer = CGANAnonymizer(args=self.args_, g=g)
            anonymized_features, anonymized_labels = anonymizer.apply(self)
        else:
            raise ValueError("Unsupported anonymizer. Expected 'cvae_fedavg' or 'cgan_fedavg'.")
        
        print(f"\tElapsed time for anonymizing data = {(time.time() - start):.2f}s")
        return anonymized_features, anonymized_labels

    def integrate_global_data(self, global_features, global_labels): 
        """
        Receive global anonymized data (embeddings + labels) from the server
        """
        if global_features is None or global_labels is None or len(global_features) == 0:
            self.global_features = np.array([], dtype=np.float32)
            self.global_labels = np.array([], dtype=np.int64)
            return
        
        self.global_features = np.asarray(global_features, dtype=np.float32)
        self.global_labels = np.asarray(global_labels, dtype=np.int64)

    def train_linear_classifier(self, scope, model, chkpt_path, epochs=100): 
        """
        Train a linear classifier on the client's local or global data

        :param scope: "local" or "global"
        :param model: the model to be trained
        :param chkpt_path: path to save the trained model
        :param epochs: number of training epochs
        :return: trained model
        """
        if scope == "local":
            x, y = self.train_features, self.train_labels
        elif scope == "global":
            x, y = self.global_features, self.global_labels
        else:
            raise ValueError("Scope must be 'local' or 'global'.")
        
        if x is None or y is None or len(x) == 0:
            warnings.warn(f"No training data available for scope={scope}", RuntimeWarning)
            return model

        device = self.args_.device
        model.to(device)
        model.train()

        optimizer = get_optimizer(
            optimizer_name=self.args_.classifier_optimizer, 
            model=model, 
            lr_initial=self.args_.lr,
            weight_decay=5e-4
        ) 
        criterion = nn.CrossEntropyLoss()
        x = torch.tensor(x, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
        torch.save(model.state_dict(), chkpt_path)
        
        return model
    
    def compute_linear_outputs(self, x, scope="local"):
        """
        Compute logits from the local or global linear classifier
        """
        device = self.args_.device
        x = torch.tensor(x, dtype=torch.float32, device=device)

        if scope == "local":
            if self.local_classifier is None:
                return None
            self.local_classifier.to(device) 
            self.local_classifier.eval()
            with torch.no_grad():
                outputs = self.local_classifier(x).cpu().numpy()
            return outputs

        elif scope == "global":
            if self.global_classifier is None:
                return None
            self.global_classifier.to(device) 
            self.global_classifier.eval()
            with torch.no_grad():
                outputs = self.global_classifier(x).cpu().numpy()
            return outputs
        
        else:
            raise ValueError("Scope must be 'local' or 'global'.")

    def evaluate(self, weight, val_mode):
        """
        Evaluate the client for a given weight (λ) between local and global models

        :param weight: float in [0, 1]
        :param val_mode: if True, evaluate on validation set; otherwise, on test set
        :return: acc, balanced_acc, f1_score_macro, f1_score_weighted
        """
        features = self.val_features if val_mode else self.test_features
        labels = self.val_labels if val_mode else self.test_labels 

        if self.args_.classifier == 'linear':
            chkpt_path = os.path.join(self.args_.chkpts_dir, 'client_classifiers')
            os.makedirs(chkpt_path, exist_ok=True)
            local_classifier_chkpt = os.path.join(chkpt_path, f'client_{self.id}.pt')
            global_classifier_chkpt = os.path.join(chkpt_path, f'client_{self.id}_global.pt')
            
            self.local_classifier = LinearLayer(self.features_dimension, self.num_classes) 
            self.global_classifier = LinearLayer(self.features_dimension, self.num_classes)
            if not os.path.isfile(local_classifier_chkpt) or not os.path.isfile(global_classifier_chkpt): 
                self.local_classifier = self.train_linear_classifier(scope="local", model=self.local_classifier, chkpt_path=local_classifier_chkpt, epochs=self.args_.local_epochs)
                self.global_classifier = self.train_linear_classifier(scope="global", model=self.global_classifier, chkpt_path=global_classifier_chkpt, epochs=self.args_.local_epochs)

            self.local_classifier.load_state_dict(torch.load(local_classifier_chkpt, weights_only=True))
            self.global_classifier.load_state_dict(torch.load(global_classifier_chkpt, weights_only=True))
            print(f"Local and global model loaded successfully from {local_classifier_chkpt}")
        
        elif self.args_.classifier == 'linear_fedavg': 
            chkpt_path = os.path.join(self.args_.chkpts_dir, 'client_classifiers')
            os.makedirs(chkpt_path, exist_ok=True)
            local_classifier_chkpt = os.path.join(chkpt_path, f'client_{self.id}.pt')
            global_classifier_chkpt = os.path.join(self.args_.fedavg_chkpts_dir, f'global_{self.args_.n_fedavg_rounds}.pt')

            self.local_classifier = LinearLayer(self.features_dimension, self.num_classes) 
            self.global_classifier = LinearLayer(self.features_dimension, self.num_classes) 
            if not os.path.isfile(local_classifier_chkpt): 
                self.local_classifier = self.train_linear_classifier(scope="local", model=self.local_classifier, chkpt_path=local_classifier_chkpt)

            self.local_classifier.load_state_dict(torch.load(local_classifier_chkpt, weights_only=True))
            self.global_classifier.load_state_dict(torch.load(global_classifier_chkpt, weights_only=True))
            print(f"Local model loaded successfully from {local_classifier_chkpt}")
            print(f"Global model loaded successfully from {global_classifier_chkpt}")

        else:
            raise ValueError("Unsupported classifier. Expected 'linear' or 'linear_fedavg'.")

        # Compute and interpolate outputs from local and global classifiers for final prediction        
        self.local_outputs = self.compute_linear_outputs(features, scope="local")
        self.global_outputs = self.compute_linear_outputs(features, scope="global")
        
        if self.local_outputs is not None and self.global_outputs is not None:
            self.local_outputs = softmax(self.local_outputs, axis=1)
            self.global_outputs = softmax(self.global_outputs, axis=1)
            outputs = weight * self.local_outputs + (1 - weight) * self.global_outputs
        elif self.local_outputs is None and self.global_outputs is not None:
            warnings.warn("Evaluation is done only with global outputs", RuntimeWarning)
            outputs = softmax(self.global_outputs, axis=1)
        elif self.local_outputs is not None and self.global_outputs is None:
            warnings.warn("Evaluation is done only with local outputs", RuntimeWarning)
            outputs = softmax(self.local_outputs, axis=1)
        else:
            raise ValueError("Both local and global outputs are None. Cannot evaluate.")
        
        # Get final predictions and metrics
        predictions = np.argmax(outputs, axis=1) 
        acc, balanced_acc, f1_score_macro, f1_score_weighted = compute_classification_metrics(
            labels, predictions, num_classes=self.num_classes
        )

        return acc, balanced_acc, f1_score_macro, f1_score_weighted
