"""Train and predict the two-branch neural stacking pipeline."""

import argparse
from pathlib import Path

import torch

from .data import ID, load_features, load_targets, write_submission
from .metrics import macro_auc
from .models import TreeConfig
from .pipeline import load_pipeline, predict_loaded, train_pipeline, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "predict", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--main", required=True, help="Main feature Parquet")
        p.add_argument("--extra", help="Optional extra features; exact ID set required")
        p.add_argument("--output", required=True)
        p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        p.add_argument("--threads", type=int, default=2)
        if name == "train":
            p.add_argument("--targets", required=True)
            p.add_argument("--folds", type=int, default=5)
            p.add_argument("--iterations", type=int, default=3500)
            p.add_argument("--xgb-iterations", type=int, default=1000)
            p.add_argument("--epochs", type=int, default=100)
            p.add_argument("--batch-size", type=int, default=512)
            p.add_argument("--seed", type=int, default=42)
        else:
            p.add_argument("--model", required=True, help="Trusted local artifact directory")
            if name == "evaluate":
                p.add_argument("--targets", required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    frame = load_features(args.main, args.extra)
    if frame.empty:
        parser.error("Input table must not be empty")
    if args.command == "train":
        targets = load_targets(args.targets, frame[ID])
        config = TreeConfig(args.iterations, args.xgb_iterations, args.seed,
                            args.device, args.threads)
        train_pipeline(frame, targets, args.output, config, args.folds,
                       args.epochs, args.batch_size)
    else:
        bundle, model = load_pipeline(args.model, args.device)
        scores = predict_loaded(bundle, model, frame, args.device)
        write_submission(args.output, frame[ID], scores, bundle["targets"])
        if args.command == "evaluate":
            targets = load_targets(args.targets, frame[ID])
            if set(targets.columns) != set(bundle["targets"]):
                raise ValueError("Evaluation target names differ from fitted targets")
            if targets.columns.tolist() != bundle["targets"]:
                targets = targets[bundle["targets"]]
            result = macro_auc(targets, scores, bundle["targets"])
            result["customers_seen_during_base_training"] = len(
                set(frame[ID]).intersection(bundle["training_ids"])
            )
            write_json(Path(args.output).with_suffix(".metrics.json"), result)
            print(result)
        print(f"Wrote {len(frame)} customers and {scores.shape[1]} product scores")


if __name__ == "__main__":
    main()
