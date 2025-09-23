import warnings
from opacus import GradSampleModule
import torch


def average_learners(
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False
):
    """
    Compute the weighted average of client models and store it into target_learner's model (FedAvg)
    Adapted from: https://github.com/omarfoq/knn-per/tree/main
    
    :param learners: (list of Learner objects) list of learners to be averaged
    :param target_learner: (Learner object) the learner to store the averaged model
    :param weights: (torch.Tensor) 1D tensor of the same length as len(learners), having values between 0 and 1, and summing to 1,
                    if None, uniform weights are used
    :param average_params: (bool) if set to true the parameters are averaged; default is True
    :param average_gradients: (bool) if set to true the gradient are also averaged; default is False
    """
    if not average_params and not average_gradients:
        return

    if weights is None:
        n_learners = len(learners)
        weights = (1.0 / n_learners) * torch.ones(n_learners, device=learners[0].device)
    else:
        weights = weights.to(learners[0].device)

    target_state_dict = target_learner.model.state_dict(keep_vars=True)

    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            if average_params:
                target_state_dict[key].data.fill_(0.)

            if average_gradients:
                target_state_dict[key].grad = target_state_dict[key].data.clone()
                target_state_dict[key].grad.data.fill_(0.)

            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)
                if average_params:
                    target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()
                if average_gradients:
                    if state_dict[key].grad is not None:
                        target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "Trying to average gradients before back propagation, you should set `average_gradients=False`."
                        )

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()

def average_trainers(
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False
):
    """
    Compute the weighted average of local CVAE models and store it into target_learner's model
    """
    if not average_params and not average_gradients:
        return
    
    if weights is None:
        n_learners = len(learners)
        weights = (1.0 / n_learners) * torch.ones(n_learners, device=learners[0].device)
    else:
        weights = weights.to(learners[0].device)

    target_state_dict = target_learner.model._module.state_dict(keep_vars=True) if isinstance(target_learner.model, GradSampleModule) else target_learner.model.state_dict(keep_vars=True)

    decoder_keys = [key for key in target_state_dict if "decoder" in key]

    for key in decoder_keys:
        if target_state_dict[key].data.dtype == torch.float32:
            if average_params:
                target_state_dict[key].data.fill_(0.)

            if average_gradients:
                target_state_dict[key].grad = target_state_dict[key].data.clone()
                target_state_dict[key].grad.data.fill_(0.)

            for learner_id, learner in enumerate(learners):
                state_dict = learner.model._module.state_dict(keep_vars=True) if isinstance(learner.model, GradSampleModule) else learner.model.state_dict(keep_vars=True)
                if average_params:
                    target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()
                if average_gradients:
                    if state_dict[key].grad is not None:
                        target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "Trying to average_gradients before back propagation, you should set `average_gradients=False`."
                        )

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model._module.state_dict() if isinstance(learner.model, GradSampleModule) else learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()


def average_trainers_cgan(
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False
):
    """
    Compute the weighted average of local CGAN models and store it into target_learner's model.
    """
    if not average_params and not average_gradients:
        return
    
    if weights is None:
        n_learners = len(learners)
        weights = (1.0 / n_learners) * torch.ones(n_learners, device=learners[0].device)
    else:
        weights = weights.to(learners[0].device)

    target_state_dict = target_learner.generator.state_dict(keep_vars=True) 

    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:
            if average_params:
                target_state_dict[key].data.fill_(0.)

            if average_gradients:
                target_state_dict[key].grad = target_state_dict[key].data.clone()
                target_state_dict[key].grad.data.fill_(0.)

            for learner_id, learner in enumerate(learners):
                state_dict = learner.generator.state_dict(keep_vars=True)
                if average_params:
                    target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()
                if average_gradients:
                    if state_dict[key].grad is not None:
                        target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "Trying to average_gradients before back propagation, you should set `average_gradients=False`."
                        )

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.generator.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()
    
def copy_model(target, source):
    """
    Copy weights from source to target
    """
    target.load_state_dict(source.state_dict())

def copy_decoder_only(target, source):
    """
    Copy only decoder weights from source to target (for CVAE)
    """
    target_state_dict = target._module.state_dict() if isinstance(target, GradSampleModule) else target.state_dict()
    source_state_dict = source._module.state_dict() if isinstance(source, GradSampleModule) else source.state_dict()
    
    for key in source_state_dict:
        if "decoder" in key:  
            target_state_dict[key].copy_(source_state_dict[key])

