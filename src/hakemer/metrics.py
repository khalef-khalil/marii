from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def logits_to_preds(logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    probs = logits_to_probs(logits)
    return (probs >= threshold).astype(np.int32)


def exact_match_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.all(y_true == y_pred, axis=1)))


def multilabel_map(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean AP over labels (macro), matching the chapter 4 test protocol."""
    per_label: list[float] = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if col.sum() == 0:
            continue
        per_label.append(
            float(average_precision_score(col, y_score[:, j]))
        )
    if not per_label:
        return 0.0
    return float(np.mean(per_label))


def multilabel_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    """y_* shape (N, K), binary {0,1}; y_score in [0,1] for mAP."""
    out = {
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match": exact_match_ratio(y_true, y_pred),
    }
    if y_score is not None:
        out["map"] = multilabel_map(y_true, y_score)
    return out
