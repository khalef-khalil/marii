from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def multilabel_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """y_* shape (N, K), binary {0,1}."""
    return {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def logits_to_preds(logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (probs >= threshold).astype(np.int32)
