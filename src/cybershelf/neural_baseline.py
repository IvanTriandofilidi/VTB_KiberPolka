"""Standalone numerical-feature MLP from the supplementary banking experiment."""

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from .features import FeatureBuilder
from .stacking import batch_indices, predict_logits


class NumericNet(nn.Module):
    def __init__(self, input_dim, output_dim=41):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.model(x)


class NumericBaseline:
    """Fit only on supplied training rows; evaluate externally on untouched data."""

    def fit(self, frame, targets, epochs=20, batch_size=2048, seed=42, device="cpu"):
        if epochs < 1 or len(frame) != len(targets) or len(frame) < 2:
            raise ValueError("Require positive epochs and at least two aligned training rows")
        y = np.asarray(targets, dtype="float32")
        if y.ndim != 2 or not np.isin(y, [0, 1]).all():
            raise ValueError("Require a complete binary target matrix")
        torch.manual_seed(seed)
        self.builder = FeatureBuilder().fit(frame)
        features = self.builder.transform(frame, "catboost")
        self.columns = [c for c in features if c.startswith("num_feature_")]
        self.preprocess = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True),
            StandardScaler(),
        )
        x = self.preprocess.fit_transform(features[self.columns]).astype("float32")
        self.model = NumericNet(x.shape[1], y.shape[1]).to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(seed)
        self.history = []
        for epoch in range(1, epochs + 1):
            self.model.train()
            total = 0.0
            for indices in batch_indices(len(x), batch_size, rng):
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(torch.as_tensor(x[indices], device=device))
                loss = criterion(logits, torch.as_tensor(y[indices], device=device))
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite baseline loss")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            self.history.append({"epoch": epoch, "train_bce": total / len(x)})
        return self

    def predict(self, frame, device="cpu", batch_size=1024):
        features = self.builder.transform(frame, "catboost")
        x = self.preprocess.transform(features[self.columns]).astype("float32")
        self.model.to(device)
        return predict_logits(self.model, x, device, batch_size)
