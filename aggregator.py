import os
import pickle
import random
import numpy as np
import torch
from copy import deepcopy
from abc import ABC, abstractmethod
from opacus import GradSampleModule

from utils.torch_utils import *


class Aggregator(ABC):
    """ Base class for Aggregator. `Aggregator` dictates communications between clients
    Adapted from: https://github.com/omarfoq/knn-per/tree/main
    
    :param clients: (list of Client objects) list of clients for training   
    :param global_learner: (Learner object) global learner to be shared among clients
    :param log_freq: (int) frequency of logging
    :param algorithm: (str) algorithm for optimizing FedAvg, e.g. "fedprox"
    :param sampling_rate: (float) fraction of clients to be sampled at each round
    :param sample_with_replacement: (bool) whether to sample clients with replacement
    :param test_clients: (list of Client objects) list of clients for testing
    :param verbose: (int) verbosity level; choices are 0 (quiet), 1 (show global logs), 2 (show also clients' logs)
    :param seed: (int) random seed for reproducibility
    """
    def __init__(
            self,
            clients,
            global_learner,
            log_freq,
            algorithm=None,
            sampling_rate=1.0,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None
    ):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.global_learner = global_learner
        self.device = self.global_learner.device if self.global_learner else None

        self.clients = clients
        self.test_clients = test_clients or []
        self.n_clients = len(clients)
        self.n_test_clients = len(test_clients) if test_clients is not None else 0

        # Weights used for FedAvg (proportional to client train sizes)
        self.clients_weights =\
            torch.tensor(
                [client.n_train_samples for client in self.clients],
                dtype=torch.float32,
                device=self.device
            )
        self.clients_weights = self.clients_weights / self.clients_weights.sum()

        self.sampling_rate = sampling_rate
        self.sample_with_replacement = sample_with_replacement
        self.n_clients_per_round = max(1, int(self.sampling_rate * self.n_clients))
        self.sampled_clients_ids = list()
        self.sampled_clients = list()

        self.algorithm = algorithm
        self.log_freq = log_freq
        self.verbose = verbose
        self.c_round = 0

    @abstractmethod
    def mix(self):
        """
        Perform one communication round:
        - Select clients
        - Perform client local steps
        - Aggregate into global
        - Broadcast back to clients (if applicable)
        """
        pass

    @abstractmethod
    def toggle_client(self, client_id, mode):
        """
        Toggle client at index `client_id`, if `mode=="train"`, `client_id` is selected in `self.clients`,
        otherwise it is selected in `self.test_clients`
        """
        pass

    def toggle_clients(self):
        for client_id in range(self.n_clients):
            self.toggle_client(client_id, mode="train")

    def toggle_sampled_clients(self):
        for client_id in self.sampled_clients_ids:
            self.toggle_client(client_id, mode="train")

    def toggle_test_clients(self):
        for client_id in range(self.n_test_clients):
            self.toggle_client(client_id, mode="test")
    
    def sample_clients(self):
        """
        Sample a list of clients for a round
        """
        if self.sample_with_replacement:
            self.sampled_clients_ids = \
                self.rng.choices(
                    population=range(self.n_clients),
                    weights=self.clients_weights,
                    k=self.n_clients_per_round,
                )
        else:
            self.sampled_clients_ids = self.rng.sample(range(self.n_clients), k=self.n_clients_per_round)

        self.sampled_clients = [self.clients[idx] for idx in self.sampled_clients_ids]

    def aggregate_metrics(self, epoch=None, max_epochs=None, save_path=None):
        """
        Aggregate metrics from all clients and print global metrics

        :param epoch: (int) current epoch
        :param max_epochs: (int) maximum (total) number of epochs
        :param save_path: (str) path to save the metrics
        """
        # Prepare test clients (if any)
        self.toggle_test_clients()

        for clients, mode in [(self.clients, "train"),(self.test_clients, "test")]:
            if len(clients) == 0:
                continue

            global_train_loss = 0.0
            global_train_acc = 0.0
            global_train_balanced_acc = 0.0
            global_test_loss = 0.0
            global_test_acc = 0.0
            global_test_balanced_acc = 0.0

            total_n_train_samples = 0
            total_n_test_samples = 0

            train_accuracies = []
            test_accuracies = []
            train_balanced_accuracies = []
            test_balanced_accuracies = []

            all_client_ids = []

            client_results_dict = {}

            for client in clients:
                train_loss, train_acc, train_balanced_acc, test_loss, test_acc, test_balanced_acc = client.compute_metrics()

                if self.verbose > 1: 
                    print("*" * 30)
                    print(f"Client {client.id}..")
                    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train Balanced Acc: {train_balanced_acc * 100:.2f}%")
                    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}% | Test Balanced Acc: {test_balanced_acc * 100:.2f}%")

                if save_path and epoch == max_epochs - 1:
                    results_dict = {
                        'client_id': client.id,
                        'epoch': epoch+1,
                        'train_loss': train_loss, 
                        'train_acc': train_acc * 100, 
                        'train_balanced_acc': train_balanced_acc * 100, 
                        'test_loss': test_loss, 
                        'test_acc': test_acc * 100, 
                        'test_balanced_acc': test_balanced_acc * 100,
                        'n_train_samples': client.n_train_samples, 
                        'n_val_samples': client.n_val_samples, 
                        'n_test_samples': client.n_test_samples, 
                    } 
                    client_results_dict[client.id] = results_dict

                global_train_loss += train_loss * client.n_train_samples
                global_train_acc += train_acc * client.n_train_samples
                global_train_balanced_acc += train_balanced_acc * client.n_train_samples
                global_test_loss += test_loss * client.n_test_samples
                global_test_acc += test_acc * client.n_test_samples
                global_test_balanced_acc += test_balanced_acc * client.n_test_samples

                total_n_train_samples += client.n_train_samples
                total_n_test_samples += client.n_test_samples

                all_client_ids.append(client.id)
                train_accuracies.append(train_acc)
                test_accuracies.append(test_acc)
                train_balanced_accuracies.append(train_balanced_acc)
                test_balanced_accuracies.append(test_balanced_acc)

            global_train_loss /= total_n_train_samples
            global_test_loss /= total_n_test_samples
            global_train_acc /= total_n_train_samples
            global_train_balanced_acc /= total_n_train_samples
            global_test_acc /= total_n_test_samples
            global_test_balanced_acc /= total_n_test_samples

            # Calculate mean and standard deviations for train and test accuracies
            train_acc_mean, train_acc_std = np.mean(train_accuracies), np.std(train_accuracies)
            test_acc_mean, test_acc_std = np.mean(test_accuracies), np.std(test_accuracies)
            train_balanced_acc_mean, train_balanced_acc_std = np.mean(train_balanced_accuracies), np.std(train_balanced_accuracies)
            test_balanced_acc_mean, test_balanced_acc_std = np.mean(test_balanced_accuracies), np.std(test_balanced_accuracies)
            
            if self.verbose > 0:
                print("Global..")
                print(f"Train Loss: {global_train_loss:.3f} | Train Acc: {global_train_acc * 100:.2f}% | Train Balanced Acc: {global_train_balanced_acc * 100:.2f}%")
                print(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_acc * 100:.2f}% | Test Balanced Acc: {global_test_balanced_acc * 100:.2f}%")
                print(f"Train Acc Mean/Std: {train_acc_mean * 100:.2f}%/{train_acc_std * 100:.2f}% | Test Acc Mean/Std: {test_acc_mean * 100:.2f}%/{test_acc_std * 100:.2f}%")
                print(f"Train Balanced Acc Mean/Std: {train_balanced_acc_mean * 100:.2f}%/{train_balanced_acc_std * 100:.2f}% | Test Balanced Acc Mean/Std: {test_balanced_acc_mean * 100:.2f}%/{test_balanced_acc_std * 100:.2f}%")

            if save_path and epoch == max_epochs - 1:
                global_results_dict = {
                    'all_client_ids': all_client_ids,
                    'epoch': epoch+1,
                    'global_train_loss': global_train_loss, 
                    'global_train_acc': global_train_acc * 100, 
                    'global_train_balanced_acc': global_train_balanced_acc * 100, 
                    'global_test_loss': global_test_loss, 
                    'global_test_acc': global_test_acc * 100, 
                    'global_test_balanced_acc': global_test_balanced_acc * 100, 
                    'total_n_train_samples': total_n_train_samples, 
                    'total_n_test_samples': total_n_test_samples, 
                }

                save_dir = os.path.join(save_path, mode)
                os.makedirs(save_dir, exist_ok=True)

                with open(os.path.join(save_dir, "global_results_dict.pkl"), 'wb') as f:
                    pickle.dump(global_results_dict, f)
                
                with open(os.path.join(save_dir, "client_results_dict.pkl"), 'wb') as f:
                    pickle.dump(client_results_dict, f)

        if self.verbose > 0:
            print("#" * 80)

    def save_state(self, path):
        """
        Save the state of the aggregator, i.e., the state dictionary of  `global_learner` as `.pt` file,
         and the state of each client in `self.clients`

        :param path: (str) directory path to save the state
        """
        save_path = os.path.join(path, "global.pt")
        torch.save(self.global_learner.model.state_dict(), save_path)

        for client_id, client in enumerate(self.clients):
            self.toggle_client(client_id, mode="train")
            client.save_state()
            if not isinstance(self, NoCommunicationAggregator):
                client.free_memory()

    def load_state(self, path):
        """
        Load the state of the aggregator

        :param path: (str) directory path to load the state from
        """
        chkpt_path = os.path.join(path, f"global.pt")
        self.global_learner.model.load_state_dict(torch.load(chkpt_path))

        for client_id, client in self.clients:
            self.toggle_client(client_id, mode="train")
            client.load_state()
            if not isinstance(self, NoCommunicationAggregator):
                client.free_memory()


class CentralizedAggregator(Aggregator):
    """Aggregator for standard FedAvg
    All clients get fully synchronized with the average client.
    
    Adapted from: https://github.com/omarfoq/knn-per/tree/main
    """
    def mix(self):
        self.sample_clients()
        self.toggle_sampled_clients()

        # Local updates
        for client in self.sampled_clients:
            client.learner.algorithm = self.algorithm
            client.learner.global_model = deepcopy(self.global_learner.model)
            client.step()

        learners = [client.learner for client in self.sampled_clients]
            
        average_learners(
                learners=learners,
                target_learner=self.global_learner,
                weights=self.clients_weights[self.sampled_clients_ids] / self.sampling_rate,
                average_params=True,
                average_gradients=False
            )

        for client in self.clients:
            copy_model(client.learner.model, self.global_learner.model)

        self.c_round += 1

    def toggle_client(self, client_id, mode):
        if mode == "train":
            client = self.clients[client_id]
        else:
            client = self.test_clients[client_id]

        if client.is_ready():
            copy_model(client.learner.model, self.global_learner.model)
        else:
            client.learner = deepcopy(self.global_learner)

        if callable(getattr(client.learner.optimizer, "set_initial_params", None)):
            client.learner.optimizer.set_initial_params(
                self.global_learner.model.parameters()
            )

    def save_state(self, path):
        """
        Save the state of the aggregator, i.e., the state dictionary of  `global_learner` as `.pt` file,
         and the state of each client in `self.clients`

        :param path: (str) directory path to save the state
        """
        save_path = os.path.join(path, f"global_{self.c_round}.pt")
        torch.save(self.global_learner.model.state_dict(), save_path)

    def load_state(self, path):
        """
        Load the state of the aggregator

        :param path: (str) directory path to load the state from
        """
        chkpt_path = os.path.join(path, f"global_{self.c_round}.pt")
        self.global_learner.model.load_state_dict(torch.load(chkpt_path))
 

class CVAEAggregator(Aggregator):
    """Aggregator for FedAvg over CVAE models
    - Average decoder weights across clients into a global trainer
    - Broadcast the global decoder back to clients
    """
    def __init__(
        self,
        clients,
        global_trainer,
        anonymizer,
        global_learner=None,
        log_freq=10,
        verbose=1, 
        seed=None,
        *args, 
        **kwargs
    ):
        super().__init__(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            sampling_rate=1.0,
            sample_with_replacement=False,
            test_clients=None,
            verbose=verbose,
            seed=seed,
            *args, 
            **kwargs
        )

        self.global_trainer = global_trainer
        self.device = self.global_trainer.device
        self.anonymizer = anonymizer

    def mix(self):
        self.sample_clients()
        self.toggle_sampled_clients()

        for client in self.sampled_clients:
            client.trainer = self.anonymizer.get_trainer(
                client=client, 
                global_model=self.global_trainer.model, 
                is_trained=True,
                num_epochs=5) 

        trainers = [client.trainer for client in self.sampled_clients]

        average_trainers(
            learners=trainers,
            target_learner=self.global_trainer,
            weights=self.clients_weights[self.sampled_clients_ids] / self.sampling_rate,
            average_params=True,
            average_gradients=False
        )

        for client in self.clients:
            copy_decoder_only(client.trainer.model, self.global_trainer.model)

        self.c_round += 1

    def toggle_client(self, client_id, mode):
        if mode == "train":
            client = self.clients[client_id]
        else:
            client = self.test_clients[client_id]

        if client.trainer.is_ready:
            copy_decoder_only(client.trainer.model, self.global_trainer.model)

        else:
            client.trainer = deepcopy(self.global_trainer)

        if callable(getattr(client.trainer.optimizer, "set_initial_params", None)):
            client.trainer.optimizer.set_initial_params(
                self.global_trainer.model.parameters()
            )

    def aggregate_metrics(self, anonymizer):
        global_val_loss = 0.
        total_n_val_samples = 0

        for client in self.clients:
            _, val_loader = anonymizer.prepare_data(client.train_features, client.train_labels, client.val_features, client.val_labels)
            val_loss = client.trainer.evaluate(val_loader)

            if self.verbose > 0: 
                print("*" * 30)
                print(f"Client {client.id}..")
                print(f"Val Loss: {val_loss:.3f}")

            global_val_loss += val_loss * client.n_val_samples
            total_n_val_samples += client.n_val_samples

        global_val_loss /= total_n_val_samples

        if self.verbose > 0:
            print("Global..")
            print(f"Val Loss: {global_val_loss:.3f}")


    def save_state(self, path):
        """
        Save the state of the aggregator, i.e., the state dictionary of  `global_learner` as `.pt` file,
         and the state of each client in `self.clients`

        :param path: (str) directory path to save the state
        """
        os.makedirs(path, exist_ok=True)
        global_chkpt_path = os.path.join(path, "global.pt")

        checkpoint = {
            'model_state_dict': self.global_trainer.model._module.state_dict() if isinstance(self.global_trainer.model, GradSampleModule) else self.global_trainer.model.state_dict(), 
            'optimizer_state_dict': self.global_trainer.optimizer.state_dict()
        }

        if self.global_trainer.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.global_trainer.lr_scheduler.state_dict()

        torch.save(checkpoint, global_chkpt_path)
        print(f"Global model checkpoint saved successfully at {global_chkpt_path}.")

        for client in self.clients:
            self.toggle_client(client.id, mode="train")
            client_chkpt_path = os.path.join(path, f"client_{client.id}.pt")
            checkpoint = {
                'model_state_dict': client.trainer.model._module.state_dict() if isinstance(client.trainer.model, GradSampleModule) else client.trainer.model.state_dict(), 
                'optimizer_state_dict': client.trainer.optimizer.state_dict()
            }

            if client.trainer.lr_scheduler is not None:
                checkpoint['scheduler_state_dict'] = client.trainer.lr_scheduler.state_dict()

            torch.save(checkpoint, client_chkpt_path)
            print(f"Client_{client.id}'s model checkpoint saved successfully at {client_chkpt_path}.")

    def load_state(self, path):
        """
        Load the state of the aggregator

        :param path: (str) directory path to load the state from
        """
        global_chkpt_path = os.path.join(path, f"global.pt")
        checkpoint = torch.load(global_chkpt_path, map_location=self.device, weights_only=True)
        print(f"Global model checkpoint loaded successfully from {global_chkpt_path}.")

        for key in checkpoint["model_state_dict"]:
            if "decoder" in key:
                self.global_trainer.model.state_dict()[key].copy_(checkpoint["model_state_dict"][key])

        for client in self.clients:
            self.toggle_client(client.id, mode="train")
            client_chkpt_path = os.path.join(path, f"client_{client.id}.pt")
            checkpoint = torch.load(client_chkpt_path, map_location=self.device, weights_only=True)

            if isinstance(client.trainer.model, GradSampleModule):
                client.trainer.model._module.load_state_dict(checkpoint["model_state_dict"])
            else:
                client.trainer.model.load_state_dict(checkpoint["model_state_dict"])

            client.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint:
                client.trainer.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            print(f"Client_{client.id}'s model checkpoint loaded successfully from {client_chkpt_path}.")


class CGANAggregator(Aggregator):
    """Aggregator for FedAvg over CGAN models
    - Average generator weights across clients into a global trainer
    - Broadcast the global generator back to clients
    """
    def __init__(
        self,
        clients,
        global_trainer,
        anonymizer,
        global_learner=None,
        log_freq=10,
        verbose=1, 
        seed=None,
        *args, 
        **kwargs
    ):
        super().__init__(
            clients=clients,
            global_learner=global_learner,
            log_freq=log_freq,
            sampling_rate=1.0,
            sample_with_replacement=False,
            test_clients=None,
            verbose=verbose,
            seed=seed,
            *args, 
            **kwargs
        )

        self.global_trainer = global_trainer
        self.device = self.global_trainer.device
        self.anonymizer = anonymizer

    def mix(self):
        self.sample_clients()
        self.toggle_sampled_clients()

        for client in self.sampled_clients:
            client.trainer = self.anonymizer.get_trainer(
                client=client, 
                global_model=self.global_trainer.generator, 
                is_trained=True,
                num_epochs=5) 

        trainers = [client.trainer for client in self.sampled_clients]

        average_trainers_cgan(
            learners=trainers,
            target_learner=self.global_trainer,
            weights=self.clients_weights[self.sampled_clients_ids] / self.sampling_rate,
            average_params=True,
            average_gradients=False
        )
        
        for client in self.clients:
            copy_model(client.trainer.generator, self.global_trainer.generator)

        self.c_round += 1

    def toggle_client(self, client_id, mode):
        if mode == "train":
            client = self.clients[client_id]
        else:
            client = self.test_clients[client_id]

        if client.trainer.is_ready:
            copy_model(client.trainer.generator, self.global_trainer.generator)

        else:
            client.trainer = deepcopy(self.global_trainer)

        if callable(getattr(client.trainer.g_optimizer, "set_initial_params", None)):
            client.trainer.g_optimizer.set_initial_params(
                self.global_trainer.generator.parameters()
            )

    def aggregate_metrics(self, anonymizer):
        global_val_g_loss = 0.
        global_val_d_accuracy = 0.
        total_n_val_samples = 0

        for client_id, client in enumerate(self.clients):
            _, val_loader = anonymizer.prepare_data(client.train_features, client.train_labels, client.val_features, client.val_labels)
            val_g_loss, val_d_accuracy = client.trainer.evaluate(val_loader)

            if self.verbose > 0: 
                print("*" * 30)
                print(f"Client {client.id}..")
                print(f"Val G_Loss: {val_g_loss:.3f} | Val D_Accuracy: {val_d_accuracy * 100:.3f}%")

            global_val_g_loss += val_g_loss * client.n_val_samples
            global_val_d_accuracy += val_d_accuracy * client.n_val_samples
            total_n_val_samples += client.n_val_samples

        global_val_g_loss /= total_n_val_samples
        global_val_d_accuracy /= total_n_val_samples

        if self.verbose > 0:
            print("Global..")
            print(f"Val G_Loss: {global_val_g_loss:.3f} | Val D_Accuracy: {global_val_d_accuracy * 100:.3f}%")

    def save_state(self, path):
        """
        Save the state of the aggregator, i.e., the state dictionary of  `global_learner` as `.pt` file,
         and the state of each client in `self.clients`

        :param path: (str) directory path to save the state
        """
        os.makedirs(path, exist_ok=True)
        global_chkpt_path = os.path.join(path, "global.pt")

        checkpoint = {
            'generator_state_dict': self.global_trainer.generator.state_dict(), 
            'discriminator_state_dict': self.global_trainer.discriminator._module.state_dict() if isinstance(self.global_trainer.discriminator, GradSampleModule) else self.global_trainer.discriminator.state_dict(), 
            'g_optimizer_state_dict': self.global_trainer.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.global_trainer.d_optimizer.state_dict()
        }

        if self.global_trainer.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.global_trainer.lr_scheduler.state_dict()

        torch.save(checkpoint, global_chkpt_path)
        print(f"Global model checkpoint saved successfully at {global_chkpt_path}.")

        for client in self.clients:
            self.toggle_client(client.id, mode="train")
            client_chkpt_path = os.path.join(path, f"client_{client.id}.pt")
            checkpoint = {
                'generator_state_dict': client.trainer.generator.state_dict(), 
                'discriminator_state_dict': client.trainer.discriminator._module.state_dict() if isinstance(client.trainer.discriminator, GradSampleModule) else client.trainer.discriminator.state_dict(), 
                'g_optimizer_state_dict': client.trainer.g_optimizer.state_dict(),
                'd_optimizer_state_dict': client.trainer.d_optimizer.state_dict()
            }

            if client.trainer.lr_scheduler is not None:
                checkpoint['scheduler_state_dict'] = client.trainer.lr_scheduler.state_dict()

            torch.save(checkpoint, client_chkpt_path)
            print(f"Client_{client.id}'s model checkpoint saved successfully at {client_chkpt_path}.")

    def load_state(self, path):
        """
        Load the state of the aggregator

        :param path: (str) directory path to load the state from
        """
        global_chkpt_path = os.path.join(path, f"global.pt")
        checkpoint = torch.load(global_chkpt_path, map_location=self.device, weights_only=True)
        self.global_trainer.generator.load_state_dict(checkpoint["generator_state_dict"])
        print(f"Global model checkpoint loaded successfully from {global_chkpt_path}.")

        if isinstance(self.global_trainer.discriminator, GradSampleModule):
            self.global_trainer.discriminator._module.load_state_dict(checkpoint['discriminator_state_dict'])
        else:
            self.global_trainer.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        for client in self.clients:
            self.toggle_client(client.id, mode="train")
            client_chkpt_path = os.path.join(path, f"client_{client.id}.pt")
            checkpoint = torch.load(client_chkpt_path, map_location=self.device, weights_only=True)

            client.trainer.generator.load_state_dict(checkpoint["generator_state_dict"])

            if isinstance(client.trainer.discriminator, GradSampleModule):
                client.trainer.discriminator._module.load_state_dict(checkpoint['discriminator_state_dict'])
            else:
                client.trainer.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

            client.trainer.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint:
                client.trainer.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            print(f"Client_{client.id}'s model checkpoint loaded successfully from {client_chkpt_path}.")
            

class NoCommunicationAggregator(Aggregator):
    """Clients do not communicate. Each client works locally.
    Adapted from: https://github.com/omarfoq/knn-per/tree/main
    """
    def mix(self):
        self.sample_clients()

        for client in self.sampled_clients:
            client.step()

        self.c_round += 1

    def toggle_client(self, client_id, mode):
        pass

