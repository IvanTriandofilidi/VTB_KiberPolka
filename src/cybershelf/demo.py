"""Generate anonymous synthetic inputs for execution checks, never leaderboard claims."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(output, rows=1000, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(rows, 12))
    main = pd.DataFrame(z[:, :8], columns=[f"num_feature_{i}" for i in range(1, 9)])
    main.insert(0, "customer_id", np.arange(1000000, 1000000 + rows))
    for i in range(1, 4):
        main[f"cat_feature_{i}"] = rng.integers(0, 5, rows).astype(float)
    main.loc[rng.choice(rows, rows // 10, replace=False), "num_feature_1"] = np.nan
    main.loc[rng.choice(rows, rows // 15, replace=False), "cat_feature_1"] = np.nan
    extra = pd.DataFrame(z[:, 8:], columns=[f"num_feature_{i}" for i in range(9, 13)])
    extra.insert(0, "customer_id", main["customer_id"])
    logits = z @ rng.normal(size=(12, 41)) * 0.3 + rng.uniform(-1.8, -0.1, (1, 41))
    values = rng.random((rows, 41)) < 1 / (1 + np.exp(-logits))
    targets = pd.DataFrame(values.astype("float32"),
                           columns=[f"target_{i}_1" for i in range(1, 42)])
    targets.insert(0, "customer_id", main["customer_id"])
    for name, frame in (("main", main), ("extra", extra), ("targets", targets)):
        frame.sample(frac=1, random_state=seed + len(name)).to_parquet(
            output / f"{name}.parquet", index=False
        )
    # Demo test customers have a different ID set; their features are synthetic as well.
    test = main.iloc[:37].copy()
    test["customer_id"] += rows
    test.to_parquet(output / "test.parquet", index=False)
    test_extra = extra.iloc[:37].copy()
    test_extra["customer_id"] += rows
    test_extra.to_parquet(output / "test_extra.parquet", index=False)
    print(f"Generated {rows} synthetic training customers, 41 targets, and 37 test customers")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/demo-data")
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
