"""Model-specific categorical encoding and train-fitted row statistics."""

import numpy as np
import pandas as pd


class FeatureBuilder:
    def fit(self, frame):
        self.columns = [c for c in frame if c != "customer_id"]
        self.numeric = [c for c in self.columns if c.startswith("num_feature_")]
        self.categorical = [c for c in self.columns if c.startswith("cat_feature_")]
        if not self.numeric:
            raise ValueError("At least one numerical feature required")
        counts = frame[self.columns].isna().sum(axis=1)
        self.count_mean = float(counts.mean())
        self.count_std = float(counts.std(ddof=0)) or 1.0
        self.categories = {}
        for c in self.categorical:
            values = frame[c].dropna()
            if not np.equal(values, np.floor(values)).all():
                raise ValueError(f"Category codes must be integral: {c}")
            self.categories[c] = ["__MISSING__", *sorted(set(values.map(self.token)))]
        return self

    @staticmethod
    def token(value):
        if pd.isna(value):
            return "__MISSING__"
        if not np.isfinite(value) or value != np.floor(value):
            raise ValueError("Category codes must be finite integers")
        return str(int(value))

    def transform(self, frame, backend):
        if backend not in {"catboost", "xgboost", "lightgbm"}:
            raise ValueError("Unknown backend")
        if set(frame.columns) - {"customer_id"} != set(self.columns):
            raise ValueError("Feature schema differs from fitted schema")
        result = frame[self.columns].copy()
        for c in self.categorical:
            tokens = result[c].map(self.token)
            if backend == "catboost":
                result[c] = tokens
            else:
                result[c] = pd.Categorical(tokens, categories=self.categories[c])
        numeric = frame[self.numeric]
        result["num_feature_missing_count"] = (
            frame[self.columns].isna().sum(axis=1) - self.count_mean
        ) / self.count_std
        for name in ("mean", "std", "min", "max"):
            result[f"num_feature_row_{name}"] = getattr(numeric, name)(axis=1)
        for c in result.select_dtypes(include="number"):
            result[c] = result[c].astype("float32")
        return result
