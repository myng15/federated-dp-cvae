"""Adapted from: https://github.com/omarfoq/knn-per/tree/main"""

import torch.optim as optim
import numpy as np


def get_optimizer(optimizer_name, model, lr_initial, weight_decay=5e-4):
    """
    Get torch.optim.Optimizer given an optimizer name, a model and learning rate
    
    :param optimizer_name: (str) possible are adam and sgd
    :param model: (nn.Module) model to be optimized
    :param lr_initial: (float) initial learning used to build the optimizer
    :return: torch.optim.Optimizer
    """
    if optimizer_name == "adam":
        return optim.Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            weight_decay=weight_decay 
        )
    elif optimizer_name == "sgd":
        return optim.SGD(
            [param for param in model.parameters() if param.requires_grad],
            lr=lr_initial,
            momentum=0.5, 
            weight_decay=weight_decay 
        )
    else:
        raise NotImplementedError(f"{optimizer_name} is not supported; possible are `adam` and `sgd`")

def get_lr_scheduler(optimizer, scheduler_name, n_rounds=None, warmup_epochs=5): 
    """
    Get torch.optim.lr_scheduler given an lr_scheduler name and an optimizer
    :param optimizer: (torch.optim.Optimizer) optimizer to be scheduled
    :param scheduler_name: (str) possible are `sqrt`, `linear`, `constant`, `cosine_annealing`, `multi_step`, `warmup`
    :param warmup_epochs: (int) number of epochs for warmup, only used if `scheduler_name == warmup`
    :param n_rounds: (int) number of training rounds, only used if `scheduler_name == multi_step`
    :return: torch.optim.lr_scheduler
    """
    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    elif scheduler_name == "multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        milestones = [n_rounds//2, 3*(n_rounds//4)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    elif scheduler_name == "warmup":
        assert n_rounds is not None and warmup_epochs is not None, "Warmup scheduler requires local_epoch and warmup_epochs!"
        return optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=lambda ep: ((n_rounds * ep) / warmup_epochs) if ((n_rounds * ep) < warmup_epochs) else 1
        )
    else:
        raise NotImplementedError(f"{scheduler_name} is not supported; "
                                  "possible are `sqrt`, `linear`, `constant`, `cosine_annealing`, `multi_step` and `warmup`"
                                  )
