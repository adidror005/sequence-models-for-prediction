# A Leak-Free PyTorch Dataset, DataLoader, and Training Loop for Sequence Prediction

*The reusable plumbing behind every model in the series—from chronological forecast origins to early stopping and original-scale metrics*

**Series:** Sequence Models for Prediction, Part 2 of 15
**Suggested Medium tags:** PyTorch, Time Series, Deep Learning, Data Engineering, Machine Learning

An architecture definition is only part of a forecasting system.

The model may accept a tensor shaped `[batch, lookback, features]`, but somebody still has to create those windows, keep forecast targets inside the correct chronological split, batch them efficiently, train without looking at test data, restore the best validation checkpoint, and transform the final error back into meaningful units.

That plumbing is where many apparently impressive forecasting results go wrong.

This companion builds one shared PyTorch pipeline for every model in the series. The complete reusable source is also available in [`training_pipeline.py`](code/training_pipeline.py).

## The data contract

Assume preprocessing has produced two aligned arrays:

```text
features: [time, n_features]
target:   [time]
```

For an electricity example:

```text
lookback = 168 hours
horizon  = 24 hours
```

At forecast origin `t`, the dataset returns:

```text
x = features[t-168 : t]
y = target[t : t+24]
```

The forecast origin belongs to a split only if its **entire target horizon** belongs to that split. An origin near the end of training cannot be included if some of its target hours fall inside validation.

## Imports, configuration, and reproducibility

The shared configuration keeps optimization choices visible. Reproducibility still depends on hardware and the operations used, but seeding Python, NumPy, and PyTorch removes several avoidable sources of variation.

```python
import random
from copy import deepcopy
from dataclasses import dataclass

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
    gradient_clip: float = 1.0


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")
```

The configuration is passed to the trainer instead of hidden inside it. That makes learning rate, weight decay, epoch limit, patience, and clipping part of the experiment record.

## Construct chronological forecast origins

Splitting individual windows randomly would let nearly identical overlapping histories appear in both training and validation. Instead, define chronological boundaries first and generate valid forecast origins inside each period.

```python
def chronological_origins(
    n_rows,
    lookback,
    horizon,
    train_fraction=0.70,
    validation_fraction=0.15,
    first_valid_origin=None,
):
    train_end = int(n_rows * train_fraction)
    validation_end = int(
        n_rows * (
            train_fraction + validation_fraction
        )
    )

    first = max(
        lookback,
        first_valid_origin or lookback,
    )

    train = np.arange(
        first,
        train_end - horizon + 1,
        dtype=np.int64,
    )
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
```

Why subtract `horizon` from each upper boundary? Suppose training ends at position 100 and the model predicts 24 steps. Origin 90 would require targets through position 113, crossing into validation. The final valid training origin must finish before position 100.

`first_valid_origin` handles feature warm-up. A weekly lag or rolling statistic may be undefined near the beginning of the series. When comparing feature sets, calculate the first origin valid for the richest representation and use it for all variants. Otherwise the models receive different dates and sample counts.

## Build the PyTorch Dataset

The dataset stores the aligned arrays and a list of allowed origins. It creates each input and target window only when requested, avoiding one large precomputed three-dimensional array.

```python
class ForecastDataset(Dataset):
    def __init__(
        self,
        features,
        target,
        origins,
        lookback,
        horizon,
    ):
        features = np.asarray(
            features,
            dtype=np.float32,
        )
        target = np.asarray(
            target,
            dtype=np.float32,
        )
        origins = np.asarray(
            origins,
            dtype=np.int64,
        )

        if features.ndim != 2:
            raise ValueError(
                "features must be [time, features]"
            )

        if (
            target.ndim != 1
            or len(target) != len(features)
        ):
            raise ValueError(
                "target must have shape [time]"
            )

        if (
            not np.isfinite(features).all()
            or not np.isfinite(target).all()
        ):
            raise ValueError(
                "features and target must be finite"
            )

        self.features = features
        self.target = target
        self.origins = origins
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, index):
        origin = int(self.origins[index])

        x = self.features[
            origin - self.lookback : origin
        ]
        y = self.target[
            origin : origin + self.horizon
        ]

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
        )
```

For one input feature, one item has shapes:

```text
x: [168, 1]
y: [24]
```

The `DataLoader` adds the batch dimension.

## Build the DataLoaders

Training windows may be shuffled after their chronological membership has been fixed. Shuffling the training loader changes minibatch order; it does **not** move observations between time periods.

Validation and test loaders should remain deterministic.

```python
def make_loaders(
    features,
    target,
    origins,
    lookback,
    horizon,
    batch_size=256,
):
    datasets = [
        ForecastDataset(
            features,
            target,
            split_origins,
            lookback,
            horizon,
        )
        for split_origins in origins
    ]

    train_loader = DataLoader(
        datasets[0],
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        datasets[1],
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        datasets[2],
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )
```

For a very large dataset, `num_workers`, pinned memory, persistent workers, or an iterable dataset may improve throughput. Those settings affect data delivery rather than the forecast definition and can be tuned for the execution environment.

## Validation loss and the training loop

The trainer uses AdamW and mean squared error, clips unusually large gradients, evaluates validation loss after every epoch, and keeps a deep copy of the best checkpoint.

```python
def evaluate_mse(model, loader, device):
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
    model,
    train_loader,
    validation_loader,
    config=TrainConfig(),
    device=None,
):
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
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        observations = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = loss_function(prediction, y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "training produced a non-finite loss"
                )

            loss.backward()

            if config.gradient_clip is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip,
                )

            optimizer.step()
            total_loss += loss.item() * len(x)
            observations += len(x)

        train_mse = (
            total_loss / max(observations, 1)
        )
        validation_mse = evaluate_mse(
            model,
            validation_loader,
            device,
        )

        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "validation_mse": validation_mse,
        })

        print(
            f"epoch={epoch:03d} "
            f"train={train_mse:.6f} "
            f"validation={validation_mse:.6f}"
        )

        if validation_mse < best_validation:
            best_validation = validation_mse
            best_state = deepcopy({
                name: value.detach().cpu().clone()
                for name, value
                in model.state_dict().items()
            })
            stale_epochs = 0
        else:
            stale_epochs += 1

            if stale_epochs >= config.patience:
                print("early stopping")
                break

    if best_state is None:
        raise RuntimeError(
            "no valid checkpoint was produced"
        )

    model.load_state_dict(best_state)
    return model.to(device), history
```

The test loader never appears in `fit_model`. Test performance should be calculated once after model selection. Repeatedly consulting test RMSE while tuning turns the test period into another validation set.

## Prediction and original-scale metrics

If training uses a standardized target, standardized RMSE is useful for optimization but not for communication. Store the mean and standard deviation fitted on the training period and use them to transform predictions and targets back to the original unit.

```python
def predict(model, loader, device=None):
    device = device or choose_device()
    model = model.to(device)
    model.eval()
    predictions = []
    targets = []

    with torch.inference_mode():
        for x, y in loader:
            output = model(x.to(device))
            predictions.append(
                output.cpu().numpy()
            )
            targets.append(y.numpy())

    return (
        np.concatenate(predictions),
        np.concatenate(targets),
    )


def metrics_on_original_scale(
    predictions_z,
    targets_z,
    training_mean,
    training_std,
):
    predictions = (
        predictions_z * training_std
        + training_mean
    )
    targets = (
        targets_z * training_std
        + training_mean
    )
    errors = predictions - targets

    return {
        "mae": float(
            np.mean(np.abs(errors))
        ),
        "rmse": float(
            np.sqrt(np.mean(errors ** 2))
        ),
    }
```

For multi-horizon prediction, also calculate one metric for each output column. Aggregate RMSE can hide a model that is excellent one hour ahead and poor the following morning.

## Put the complete pipeline together

The final example assumes `raw_target` and a time-aligned causal feature matrix already exist. It demonstrates the crucial scaling boundary: the target mean and standard deviation are calculated from the training period only.

```python
from sequence_models import GRUForecaster


LOOKBACK = 168
HORIZON = 24
TRAIN_FRACTION = 0.70

train_end = int(
    len(raw_target) * TRAIN_FRACTION
)
training_mean = float(
    raw_target[:train_end].mean()
)
training_std = float(
    raw_target[:train_end].std()
)

target_z = (
    raw_target - training_mean
) / training_std

# Demand-only example. For multivariate input,
# stack causally prepared columns instead.
features = target_z[:, None].astype(np.float32)
target_z = target_z.astype(np.float32)

origins = chronological_origins(
    n_rows=len(target_z),
    lookback=LOOKBACK,
    horizon=HORIZON,
)

train_loader, validation_loader, test_loader = (
    make_loaders(
        features,
        target_z,
        origins,
        lookback=LOOKBACK,
        horizon=HORIZON,
        batch_size=256,
    )
)

set_seed(42)
model = GRUForecaster(
    n_features=features.shape[1],
    horizon=HORIZON,
)

model, history = fit_model(
    model,
    train_loader,
    validation_loader,
    TrainConfig(
        epochs=50,
        patience=5,
        learning_rate=5e-4,
    ),
)

predictions_z, targets_z = predict(
    model,
    test_loader,
)
metrics = metrics_on_original_scale(
    predictions_z,
    targets_z,
    training_mean,
    training_std,
)

print(metrics)
```

Swap `GRUForecaster` for any model in the [companion model module](code/sequence_models.py). The tensor contract remains the same:

```text
model input:  [batch, lookback, n_features]
model output: [batch, horizon]
```

## What this pipeline does—and does not—guarantee

It guarantees several useful structural properties:

- targets remain inside their chronological period;
- all model families can receive the same origins;
- validation, not test performance, selects the checkpoint;
- every model uses the same input/output tensor contract;
- reported metrics can be returned to their original scale.

It cannot automatically guarantee that the supplied feature matrix is causal. A centered rolling average, a scaler fitted on the full dataset, a revised economic release, or a forward-filled value unavailable at the real forecast deadline can leak future information before the dataset ever sees it.

The safest order is:

1. define what was knowable at the forecast origin;
2. construct only causal features;
3. fit learned preprocessing on training data;
4. identify origins valid for every representation;
5. build chronological datasets and loaders;
6. train with validation-only checkpoint selection;
7. evaluate the untouched test period.

The architecture is one component of that chain. A clean Dataset and training loop are not glamorous, but they determine whether the model comparison means anything.

---

**Series navigation:** [Series index](README.md) · [Previous: Sequence-model map](01-sequence-models-for-prediction.md) · [Next: Linear forecasting](02-linear-forecaster.md)
