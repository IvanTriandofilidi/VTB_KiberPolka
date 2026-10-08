"""CatBoost and XGBoost branches, with a serialized feature contract."""

from dataclasses import dataclass

import numpy as np
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from .features import FeatureBuilder


@dataclass
class TreeConfig:
    iterations: int = 3500
    xgb_iterations: int = 1000
    seed: int = 42
    device: str = "cpu"
    threads: int = 2


class TreeBranch:
    def __init__(self, name, config):
        self.name, self.config = name, config

    def fit(self, frame, y):
        if (y.nunique() < 2).any():
            raise ValueError("Every training target needs positive and negative examples")
        self.builder = FeatureBuilder().fit(frame)
        x = self.builder.transform(frame, self.name)
        cfg = self.config
        if self.name == "catboost":
            self.model = CatBoostClassifier(
                iterations=cfg.iterations, depth=7, l2_leaf_reg=12, learning_rate=0.03,
                loss_function="MultiLogloss", bootstrap_type="Bernoulli", subsample=0.66,
                random_seed=cfg.seed, one_hot_max_size=15,
                task_type="GPU" if cfg.device == "cuda" else "CPU",
                thread_count=cfg.threads, verbose=False, allow_writing_files=False,
            )
            self.model.fit(x, y, cat_features=self.builder.categorical)
        elif self.name == "xgboost":
            counts = y.sum(axis=1)
            frequencies = counts.value_counts()
            weights = counts.map(lambda n: np.log(len(y) / frequencies[n]) + 1).to_numpy()
            self.model = XGBClassifier(
                n_estimators=cfg.xgb_iterations, learning_rate=0.03, max_depth=3,
                tree_method="hist", device=cfg.device, max_bin=64,
                objective="binary:logistic", multi_strategy="one_output_per_tree",
                gamma=0.5, reg_lambda=15, subsample=0.7, colsample_bytree=0.4,
                enable_categorical=True, random_state=cfg.seed, n_jobs=cfg.threads,
            )
            self.model.fit(x, y, sample_weight=weights)
        else:
            raise ValueError("Expected catboost or xgboost")
        return self

    def predict(self, frame):
        x = self.builder.transform(frame, self.name)
        return np.asarray(self.model.predict_proba(x), dtype="float32")
