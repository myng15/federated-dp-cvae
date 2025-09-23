import torch
import warnings
import argparse
from abc import ABC

from utils.constants import EMBEDDING_DIM


class ArgumentsManager(ABC):
    """Base class for ArgumentsManager
    Define arguments used for both standard FedAvg pipeline (eval_fedavg.py) and Ours pipeline (eval_ours.py)
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=__doc__)
        self.args = None
        self.initialized = False

        # Datasets and clients
        self.parser.add_argument(
            'experiment',
            help='name of experiment',
            type=str,
            choices=['camelyon17', 'organsmnist']
        )
        self.parser.add_argument(
            '--backbone',
            help='the backbone used to extract feature embeddings, must match utils.constants.EMBEDDING_DIM keys;'
                 'default=base_patch14_dinov2',
            type=str,
            default="base_patch14_dinov2",
            choices=list(EMBEDDING_DIM.keys())
        )
        self.parser.add_argument(
            '--client_type',
            help='client type',
            type=str,
            default="normal",
            choices=["normal", "personalized"]
        )
        self.parser.add_argument(
            '--val_frac',
            help='fraction of each client\'s train dataset used for validation',
            type=float,
            default=0.1
        )

        # Runtime
        self.parser.add_argument(
            '--bz',
            help='batch_size',
            type=int,
            default=1
        )
        self.parser.add_argument(
            '--device',
            help='device to use ("cpu" or "cuda")',
            type=str,
            default="cpu"
        )
        self.parser.add_argument(
            "--seed",
            help='random seed for reproducibility',
            type=int,
            default=42
        )
        self.parser.add_argument(
            '--chkpts_dir',
            help='directory to save checkpoints;',
            type=str,
            default="chkpts"
        )
        self.parser.add_argument(
            "--results_dir",
            help='directory to save results/logs',
            default="results"
        )
        self.parser.add_argument(
            "--verbose",
            help='verbosity level: `0` (quiet), `1` (show global logs) and `2` (show both global and local logs)',
            type=int,
            default=0
        )
        
    def parse_arguments(self, args_list=None):
        args = self.parser.parse_args(args_list) if args_list else self.parser.parse_args()
    
        if args.device == "cuda" and not torch.cuda.is_available():
            args.device = "cpu"
            warnings.warn("CUDA is not available, device is automatically set to \"CPU\"!", RuntimeWarning)

        self.args = args
        self.initialized = True


class FedAvgArgumentsManager(ArgumentsManager):
    """
    Define arguments specific to eval_fedavg.py
    """
    def __init__(self):
        super(FedAvgArgumentsManager, self).__init__()

        # FedAvg
        self.parser.add_argument(
            '--aggregator_type',
            help='aggregator type', choices=["local", "centralized"],
            type=str,
            default="centralized",
        )
        self.parser.add_argument(
            '--algorithm',
            help='algorithm for optimizing FedAvg', choices=["normal", "fedprox"],
            type=str,
            default="normal"
        )
        self.parser.add_argument(
            '--sampling_rate',
            help='fraction of clients sampled at each round',
            type=float,
            default=1.0
        )
        self.parser.add_argument(
            '--n_rounds',
            help='number of FedAvg communication rounds',
            type=int,
            default=1
        )
        self.parser.add_argument(
            '--local_steps',
            help='number of local training steps before FedAvg communication',
            type=int,
            default=5 
        )
        self.parser.add_argument(
            '--log_freq',
            help='frequency of writing logs',
            type=int,
            default=1 # log every round
        )
        self.parser.add_argument(
            '--validation',
            help='if chosen, use the validation set instead of test set;'
                 ' make sure to use `val_frac > 0` in `generate_data.py`;',
            action='store_true',
            default=False
        )

        # Optimizer and LR scheduler
        self.parser.add_argument(
            '--optimizer',
            help='optimizer to be used for training', choices=["sgd", "adam"],
            type=str,
            default="sgd"
        )
        self.parser.add_argument(
            "--fedavg_lr",
            type=float,
            help='learning rate',
            default=1e-3
        )
        self.parser.add_argument(
            "--lr_scheduler",
            help='learning rate scheduler to be used;',
            choices=["sqrt", "linear", "constant", "cosine_annealing", "multi_step", "warmup"],
            type=str,
            default="constant",
        )


class OursArgumentsManager(ArgumentsManager):
    """
    Define arguments specific to eval_ours.py
    """
    def __init__(self):
        super(OursArgumentsManager, self).__init__()

        # FedAvg
        self.parser.add_argument(
            '--fedavg_chkpts_dir',
            help='directory containing checkpoint of a global model trained via FedAvg; used in FedLambda method',
            type=str,
            default=None
        )
        self.parser.add_argument(
            '--n_fedavg_rounds',
            help='number of FedAvg rounds for anonymizer training',
            type=int,
            default=1
        )

        # Linear classifier and model interpolation
        self.parser.add_argument(
            '--classifier', 
            help='the client\'s classifier head', choices=['linear', 'linear_fedavg'], # 'linear_fedavg' for FedLambda
            type=str, 
            default='linear'
        )
        self.parser.add_argument(
            '--classifier_optimizer', 
            help='optimizer for training the client\'s classifier head', choices=['sgd', 'adam'],
            type=str, 
            default='sgd'
        )
        self.parser.add_argument(
            "--local_epochs",
            help='number of training epochs for the client\'s linear classifiers',
            type=int,
            default=100
        )
        self.parser.add_argument(
            '--weights_grid_resolution',
            help='resolution of the local model weights grid (0.0 ... 1.0), the smaller it is the higher the resolution;'
                 'higher value of resolution requires more computation time.',
            type=float,
            default=0.1
        )

        # Anonymizer and data generation
        self.parser.add_argument(
            '--anonymizer', 
            help='the generative model used for data anonymization (None for FedLambda)', choices=['cvae_fedavg', 'cgan_fedavg', None],
            type=str, 
            default=None
        )
        self.parser.add_argument(
            "--lr",
            type=float,
            help='learning rate for anonymizer and classifier training',
            default=1e-3
        )
        self.parser.add_argument(
            "--cvae_beta",
            help='beta for CVAE training',
            type=float,
            default=0.1
        )
        self.parser.add_argument(
            "--cvae_var",
            help='latent variance scaling for CVAE generation',
            type=float,
            default=1.0
        )
        self.parser.add_argument(
            '--generated_factor', 
            help='factor to scale the number of generated samples vs. the original sample size',
            type=float, 
            default=1.0
        )

        # Differential Privacy (DP)
        self.parser.add_argument(
            "--enable_dp",
            help="enable differentially private training for anonymizer",
            action="store_true",
            default=False,
        )
        self.parser.add_argument(
            "--max_grad_norm",
            help="max gradient norm for DP clipping",
            type=float,
            default=1.0,
        )
        self.parser.add_argument(
            "--epsilon",
            help="target epsilon (privacy budget) for DP accounting",
            type=float,
            default=1.0,
        )
        self.parser.add_argument(
            "--delta",
            help="target delta for DP accounting",
            type=float,
            default=1e-5,
        )
