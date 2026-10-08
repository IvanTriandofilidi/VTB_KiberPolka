"""Outer holdout, cross-fitted base predictions, and a neural meta-model."""

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)

from .data import ID, submission
from .metrics import macro_auc
from .models import TreeBranch
from .stacking import StackingNet, predict_logits, train_stacker


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def require_two_classes(y, context):
    missing = y.columns[y.nunique() < 2].tolist()
    if missing:
        raise ValueError(f"{context}: single-class targets {missing}; revise data or split")


def train_pipeline(frame, targets, output, config, folds=5, epochs=100, batch_size=512):
    if folds < 2 or epochs < 1 or config.iterations < 1 or config.xgb_iterations < 1:
        raise ValueError("Require folds >= 2 and positive iterations/epochs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    names = targets.columns.tolist()
    split = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=config.seed
    )
    development, holdout = next(split.split(frame, targets))
    x = frame.iloc[development].reset_index(drop=True)
    y = targets.iloc[development].reset_index(drop=True)
    require_two_classes(y, "Development")
    require_two_classes(targets.iloc[holdout], "Outer holdout")
    split_rows = pd.DataFrame({ID: frame[ID], "partition": "development", "fold": -1})
    split_rows.loc[holdout, "partition"] = "holdout"
    oof = np.full((len(x), 2 * len(names)), np.nan, dtype="float32")
    seen = np.zeros(len(x), dtype=int)
    cv = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=config.seed)
    for fold, (fit, valid) in enumerate(cv.split(x, y)):
        print(f"Base fold {fold + 1}/{folds}")
        require_two_classes(y.iloc[fit], f"Fold {fold} training")
        for branch_id, name in enumerate(("catboost", "xgboost")):
            branch = TreeBranch(name, config).fit(x.iloc[fit], y.iloc[fit])
            start = branch_id * len(names)
            oof[valid, start:start + len(names)] = branch.predict(x.iloc[valid])
        seen[valid] += 1
        split_rows.loc[development[valid], "fold"] = fold
    if not (seen == 1).all() or not np.isfinite(oof).all():
        raise RuntimeError("Each development row must have exactly one finite OOF prediction")
    prediction_names = [f"{b}__{n}" for b in ("catboost", "xgboost") for n in names]
    oof_frame = pd.DataFrame(oof, columns=prediction_names)
    oof_frame.insert(0, ID, x[ID].to_numpy())
    oof_frame.to_parquet(output / "oof.parquet", index=False)
    model, quantile, history, _meta_train, meta_valid = train_stacker(
        oof, y.to_numpy(), names, epochs, config.seed, config.device, batch_size
    )
    final_branches = [TreeBranch(n, config).fit(x, y) for n in ("catboost", "xgboost")]
    torch.save({"state_dict": model.cpu().state_dict(), "input_dim": oof.shape[1],
                "output_dim": len(names)}, output / "stacker.pt")
    joblib.dump({"branches": final_branches, "quantile": quantile, "targets": names,
                 "feature_columns": [c for c in frame if c != ID],
                 "training_ids": x[ID].tolist()}, output / "pipeline.joblib")
    meta_roles = np.full(len(development), "meta_train", dtype=object)
    meta_roles[meta_valid] = "meta_validation"
    split_rows["meta_partition"] = "unused"
    split_rows.loc[development, "meta_partition"] = meta_roles
    split_rows.to_parquet(output / "split.parquet", index=False)
    write_json(output / "config.json", {"trees": asdict(config), "folds": folds,
               "epochs": epochs, "batch_size": batch_size, "target_order": names})
    write_json(output / "history.json", history)
    scores = predict_pipeline(output, frame.iloc[holdout], device=config.device)
    submission(frame.iloc[holdout][ID], scores, names).to_parquet(
        output / "holdout_predictions.parquet", index=False
    )
    result = macro_auc(targets.iloc[holdout], scores, names)
    write_json(output / "holdout_metrics.json", result)
    print({"holdout_macro_auc": result["macro_auc"], "rows": len(holdout)})
    return result


def load_pipeline(path, device="cpu"):
    """Only load artifacts you trust: joblib uses executable Python serialization."""
    path = Path(path)
    bundle = joblib.load(path / "pipeline.joblib")
    state = torch.load(path / "stacker.pt", map_location="cpu", weights_only=True)
    model = StackingNet(state["input_dim"], state["output_dim"]).to(device)
    model.load_state_dict(state["state_dict"])
    return bundle, model


def predict_pipeline(path, frame, device="cpu", batch_size=1024, chunk_size=10000):
    bundle, model = load_pipeline(path, device)
    return predict_loaded(bundle, model, frame, device, batch_size, chunk_size)


def predict_loaded(bundle, model, frame, device="cpu", batch_size=1024, chunk_size=10000):
    """Predict in chunks with an already loaded model bundle."""
    outputs = []
    for start in range(0, len(frame), chunk_size):
        chunk = frame.iloc[start:start + chunk_size]
        probabilities = np.concatenate([b.predict(chunk) for b in bundle["branches"]], axis=1)
        features = bundle["quantile"].transform(probabilities).astype("float32")
        outputs.append(predict_logits(model, features, device, batch_size))
    return np.concatenate(outputs)
