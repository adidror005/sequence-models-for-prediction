"""Shared chronological training pipeline for the sequence-model examples.

Supply a causal feature matrix shaped [time, features] and a standardized target
shaped [time]. Forecast targets remain entirely inside their chronological split.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    patience: int = 5
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    gradient_clip: Optional[float] = 1.0


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ForecastDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        origins: np.ndarray,
        lookback: int,
        horizon: int,
    ) -> None:
        features = np.asarray(features, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        origins = np.asarray(origins, dtype=np.int64)

        if features.ndim != 2:
            raise ValueError("features must have shape [time, features]")
        if target.ndim != 1 or len(target) != len(features):
            raise ValueError("target must have shape [time]")
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise ValueError("features and target must be finite")

        self.features = features
        self.target = target
        self.origins = origins
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        origin = int(self.origins[index])
        x = self.features[origin - self.lookback : origin]
        y = self.target[origin : origin + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


def chronological_origins(
    n_rows: int,
    lookback: int,
    horizon: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    first_valid_origin: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create origins whose entire target horizon stays inside each split."""
    train_end = int(n_rows * train_fraction)
    validation_end = int(n_rows * (train_fraction + validation_fraction))
    first = max(lookback, first_valid_origin or lookback)

    train = np.arange(first, train_end - horizon + 1, dtype=np.int64)
    validation = np.arange(
        max(train_end, first),
        validation_end - horizon + 1,
        dtype=np.int64,
    )
    test = np.arange(
        max(validation_end, first),
        n_rows - horizon + 1,
        dtype=np.int64,
    )
    return train, validation, test


def make_loaders(
    features: np.ndarray,
    target: np.ndarray,
    origins: tuple[np.ndarray, np.ndarray, np.ndarray],
    lookback: int,
    horizon: int,
    batch_size: int = 256,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    datasets = [
        ForecastDataset(features, target, split, lookback, horizon)
        for split in origins
    ]
    return (
        DataLoader(datasets[0], batch_size=batch_size, shuffle=True),
        DataLoader(datasets[1], batch_size=batch_size, shuffle=False),
        DataLoader(datasets[2], batch_size=batch_size, shuffle=False),
    )


def evaluate_mse(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    loss_function = nn.MSELoss()
    total_loss = 0.0
    observations = 0

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            loss = loss_function(model(x), y)
            total_loss += loss.item() * len(x)
            observations += len(x)

    return total_loss / max(observations, 1)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainConfig = TrainConfig(),
    device: Optional[torch.device] = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    device = device or choose_device()
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.MSELoss()
    best_validation = float("inf")
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        observations = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x), y)

            if not torch.isfinite(loss):
                raise RuntimeError("training produced a non-finite loss")

            loss.backward()
            if config.gradient_clip is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip,
                )
            optimizer.step()
            total_loss += loss.item() * len(x)
            observations += len(x)

        train_mse = total_loss / max(observations, 1)
        validation_mse = evaluate_mse(model, validation_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_mse": train_mse,
                "validation_mse": validation_mse,
            }
        )

        if validation_mse < best_validation:
            best_validation = validation_mse
            best_state = deepcopy(
                {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            )
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a valid checkpoint")

    model.load_state_dict(best_state)
    return model.to(device), history


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray]:
    device = device or choose_device()
    model = model.to(device)
    model.eval()
    predictions = []
    targets = []

    with torch.inference_mode():
        for x, y in loader:
            predictions.append(model(x.to(device)).cpu().numpy())
            targets.append(y.numpy())

    return np.concatenate(predictions), np.concatenate(targets)


def metrics_on_original_scale(
    predictions_z: np.ndarray,
    targets_z: np.ndarray,
    training_mean: float,
    training_std: float,
) -> dict[str, float]:
    predictions = predictions_z * training_std + training_mean
    targets = targets_z * training_std + training_mean
    errors = predictions - targets
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def smoke_test() -> None:
    """Train a tiny direct forecaster on synthetic chronological data."""
    set_seed(42)
    rows, lookback, horizon = 240, 24, 3
    time = np.arange(rows, dtype=np.float32)
    target = np.sin(2 * np.pi * time / 24.0).astype(np.float32)
    features = target[:, None]
    origins = chronological_origins(
        rows,
        lookback,
        horizon,
        train_fraction=0.60,
        validation_fraction=0.20,
    )
    train_loader, validation_loader, test_loader = make_loaders(
        features,
        target,
        origins,
        lookback,
        horizon,
        batch_size=32,
    )
    model = nn.Sequential(nn.Flatten(), nn.Linear(lookback, horizon))
    model, history = fit_model(
        model,
        train_loader,
        validation_loader,
        TrainConfig(epochs=2, patience=2, learning_rate=1e-3),
        device=torch.device("cpu"),
    )
    predictions, targets = predict(model, test_loader, device=torch.device("cpu"))
    assert history
    assert predictions.shape == targets.shape
    assert predictions.shape[1] == horizon
    print("training pipeline:", predictions.shape)


if __name__ == "__main__":
    smoke_test()
