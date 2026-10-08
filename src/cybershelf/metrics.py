"""Macro ROC-AUC without invented scores for single-class targets."""

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_auc(y, scores, names):
    y, scores = np.asarray(y), np.asarray(scores)
    if y.shape != scores.shape or y.shape[1] != len(names):
        raise ValueError("Target and prediction schemas differ")
    if not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Require finite predictions and binary targets")
    per_target = {
        name: float(roc_auc_score(y[:, i], scores[:, i]))
        if len(np.unique(y[:, i])) == 2 else None
        for i, name in enumerate(names)
    }
    defined = [s for s in per_target.values() if s is not None]
    return {
        "macro_auc": float(np.mean(defined)) if len(defined) == len(names) else None,
        "defined_target_mean_auc": float(np.mean(defined)) if defined else None,
        "undefined_targets": [n for n, s in per_target.items() if s is None],
        "per_target_auc": per_target,
    }
