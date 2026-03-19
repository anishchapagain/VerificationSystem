"""
Model Evaluation Utilities
============================
Module  : ml/evaluate.py
Purpose : Metrics for signature verification: EER, ROC curve, accuracy at threshold.

Author  : Signature Verifier Team
Version : 1.0.0
"""

from typing import List, Tuple
import numpy as np


def compute_eer(labels: List[float], scores: List[float]) -> float:
    """
    Compute Equal Error Rate (EER) — the threshold where FAR == FRR.

    Lower EER = better model. A perfect model has EER = 0.0.
    State-of-the-art Siamese networks achieve EER ~2–5% on CEDAR.

    Args:
        labels : Ground truth. 1.0 = genuine pair, 0.0 = impostor pair.
        scores : Cosine similarity scores in [-1, 1].

    Returns:
        float: EER value in [0.0, 1.0].
    """
    labels_arr = np.array(labels)
    scores_arr = np.array(scores)

    thresholds = np.linspace(scores_arr.min(), scores_arr.max(), 500)
    far_list, frr_list = [], []

    for thresh in thresholds:
        predicted = (scores_arr >= thresh).astype(int)
        # FAR: impostors accepted (predicted=1 when label=0)
        impostor_mask = labels_arr == 0
        far = predicted[impostor_mask].mean() if impostor_mask.any() else 0.0
        # FRR: genuines rejected (predicted=0 when label=1)
        genuine_mask = labels_arr == 1
        frr = (1 - predicted[genuine_mask]).mean() if genuine_mask.any() else 0.0
        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    idx = np.argmin(np.abs(far_arr - frr_arr))
    eer = (far_arr[idx] + frr_arr[idx]) / 2.0
    return float(eer)


def accuracy_at_threshold(
    labels: List[float], scores: List[float], threshold: float
) -> dict:
    """
    Compute classification metrics at a fixed cosine similarity threshold.

    Returns dict with: accuracy, precision, recall, f1, far, frr.
    """
    labels_arr = np.array(labels)
    preds = (np.array(scores) >= threshold).astype(int)

    tp = int(((preds == 1) & (labels_arr == 1)).sum())
    tn = int(((preds == 0) & (labels_arr == 0)).sum())
    fp = int(((preds == 1) & (labels_arr == 0)).sum())
    fn = int(((preds == 0) & (labels_arr == 1)).sum())

    accuracy  = (tp + tn) / len(labels_arr)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    far       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "threshold": threshold, "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "far": far, "frr": frr,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
