import numpy as np
from typing import Optional
from sklearn.metrics import recall_score, f1_score
import torch


def accuracy(y_pred, y):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y).float()
    acc = correct.sum()
    return acc

def safe_balanced_accuracy(y_true, y_pred, labels: Optional[np.ndarray] = None, num_classes: Optional[int] = None) -> float:
    """
    Safely compute balanced accuracy as mean of per-class recalls.
    - If labels or num_classes is given, evaluate on that fixed label space.
    - Else, use the union of observed labels as the evaluation label space to avoid undefined classes.
    - Average only over classes present in y_true to avoid bias when some classes are unseen (which is common for non-IID splits).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is not None:
        eval_labels = np.asarray(labels)
    elif num_classes is not None:
        eval_labels = np.arange(num_classes, dtype=int)
    else:
        eval_labels = np.unique(np.concatenate([y_true, y_pred]))  # union

    recalls = recall_score(
        y_true, y_pred,
        labels=eval_labels,
        average=None,
        zero_division=0,
    )

    present_mask = np.isin(eval_labels, np.unique(y_true))
    if present_mask.any():
        return float(np.mean(recalls[present_mask]))
    return 0.0

def compute_classification_metrics(y_true, y_pred, num_classes: Optional[int] = None):
    """
    Compute accuracy, balanced accuracy, macro F1-score and weighted F1-score
    Uses safe_balanced_accuracy and safe F1 with fixed label space when num_classes is provided
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float((y_true == y_pred).sum()) / float(len(y_true)) if len(y_true) > 0 else 0.0

    bal_acc = safe_balanced_accuracy(y_true, y_pred, num_classes=num_classes)

    labels = np.arange(num_classes, dtype=int) if num_classes is not None else None
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    return acc, float(bal_acc), float(f1_macro), float(f1_weighted)

