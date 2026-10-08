"""Explicit customer and target contracts for Parquet tables."""

from pathlib import Path

import numpy as np
import pandas as pd

ID = "customer_id"


def validate_ids(frame):
    if ID not in frame or frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError("Each table requires unique, non-null customer_id values")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names")


def align(reference_ids, frame):
    """Require exactly the same ID set and restore the reference order."""
    validate_ids(frame)
    if set(reference_ids) != set(frame[ID]):
        raise ValueError("Customer ID sets differ")
    return frame.set_index(ID).loc[list(reference_ids)].reset_index()


def load_features(main_path, extra_path=None):
    frame = pd.read_parquet(main_path).reset_index(drop=True)
    validate_ids(frame)
    if extra_path:
        extra = align(frame[ID], pd.read_parquet(extra_path))
        overlap = set(frame.columns).intersection(extra.columns) - {ID}
        if overlap:
            raise ValueError(f"Overlapping feature names: {sorted(overlap)}")
        frame = pd.concat([frame, extra.drop(columns=ID)], axis=1)
    allowed = [c for c in frame if c.startswith(("cat_feature_", "num_feature_"))]
    if set(frame.columns) != {ID, *allowed} or not allowed:
        raise ValueError("Only customer_id, cat_feature_* and num_feature_* are accepted")
    for c in allowed:
        frame[c] = pd.to_numeric(frame[c], errors="raise")
        if np.isinf(frame[c].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite feature values in {c}")
        if c.startswith("num_feature_"):
            frame[c] = frame[c].astype("float32")
    return frame


def load_targets(path, ids, expected_count=41):
    frame = align(ids, pd.read_parquet(path))
    names = [c for c in frame if c.startswith("target_")]
    if set(frame.columns) != {ID, *names} or len(names) != expected_count:
        raise ValueError(f"Require exactly {expected_count} target_* columns")
    values = frame[names].to_numpy()
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Targets must be binary and complete")
    return frame[names].astype("float32")


def submission(ids, scores, target_names):
    scores = np.asarray(scores)
    if scores.shape != (len(ids), len(target_names)) or not np.isfinite(scores).all():
        raise ValueError("Prediction shape or finite-value contract violated")
    names = [c.replace("target_", "predict_", 1) for c in target_names]
    result = pd.DataFrame(scores.astype("float64"), columns=names)
    result.insert(0, ID, np.asarray(ids))
    validate_ids(result)
    return result


def write_submission(path, ids, scores, targets):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission(ids, scores, targets).to_parquet(output, index=False)
