"""Source-derived neural stacker with train-only quantile normalization."""

import copy

import numpy as np
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import QuantileTransformer
from torch import nn
from torch.nn import functional as F

from .metrics import macro_auc


class StackingNet(nn.Module):
    def __init__(self, input_dim=82, output_dim=41):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(256, output_dim)

    def forward(self, x):
        return self.classifier(self.layer2(self.layer1(x)))


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Notebook objective: alpha is a global scale, not class-specific weighting."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    return (alpha * (1 - pt).pow(gamma) * bce).mean()


def batch_indices(length, batch_size, rng):
    """Merge a singleton tail into the previous batch for BatchNorm; retain all rows."""
    if length < 2 or batch_size < 2:
        raise ValueError("BatchNorm training requires at least two rows and batch_size >= 2")
    chunks = list(np.array_split(rng.permutation(length), np.ceil(length / batch_size).astype(int)))
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        merged = np.concatenate([chunks[-2], chunks[-1]])
        chunks[-2:] = [merged]
    return chunks


@torch.inference_mode()
def predict_logits(model, x, device="cpu", batch_size=1024):
    model.eval()
    return np.concatenate([
        model(torch.as_tensor(x[i:i + batch_size], dtype=torch.float32, device=device))
        .cpu().numpy() for i in range(0, len(x), batch_size)
    ])


def train_stacker(oof, y, names, epochs=100, seed=42, device="cpu", batch_size=512):
    torch.manual_seed(seed)
    split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train, valid = next(split.split(oof, y))
    quantile = QuantileTransformer(
        output_distribution="normal", random_state=seed, n_quantiles=min(1000, len(train))
    ).fit(oof[train])
    x = quantile.transform(oof).astype("float32")
    model = StackingNet(oof.shape[1], y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best, best_state, history = -float("inf"), None, []
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in batch_indices(len(train), batch_size, rng):
            indices = train[batch]
            inputs = torch.as_tensor(x[indices], dtype=torch.float32, device=device)
            targets = torch.as_tensor(y[indices], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = focal_loss(model(inputs), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite stacking loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        scores = macro_auc(y[valid], predict_logits(model, x[valid], device), names)
        if scores["macro_auc"] is None:
            raise ValueError("Meta-validation has single-class targets; revise split/data")
        history.append({"epoch": epoch, "train_loss": total / len(train),
                        "meta_validation_auc": scores["macro_auc"]})
        print(history[-1])
        if scores["macro_auc"] > best:
            best = scores["macro_auc"]
            best_state = copy.deepcopy(model.state_dict())
        scheduler.step()
    model.load_state_dict(best_state)
    return model, quantile, history, train, valid
