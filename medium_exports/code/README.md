# Runnable companion code

The code in this folder mirrors the complete PyTorch implementations embedded in Parts 2–10.

## Files

- [`sequence_models.py`](sequence_models.py) contains all nine forecasting architectures and a shape smoke test.
- [`training_pipeline.py`](training_pipeline.py) contains chronological window construction, data loaders, training-only early stopping, prediction, inverse-scale metrics, and parameter counting.
- [`catboost_baseline.py`](catboost_baseline.py) contains the proposed causal tabular benchmark with one explicitly validated model per forecast horizon. It was not run in the supplied notebook.

Every neural model follows the same contract:

```text
input:  [batch, lookback, n_features]
output: [batch, horizon]
```

## Minimal usage

```python
import numpy as np

from sequence_models import GRUForecaster
from training_pipeline import (
    TrainConfig,
    chronological_origins,
    fit_model,
    make_loaders,
    metrics_on_original_scale,
    predict,
    set_seed,
)

LOOKBACK = 168
HORIZON = 24

# Replace these arrays with causally prepared data.
# Fit mean and standard deviation on the training period only.
features = np.load("features_standardized.npy")  # [time, features]
target = np.load("target_standardized.npy")      # [time]
training_mean = float(np.load("training_mean.npy"))
training_std = float(np.load("training_std.npy"))

origins = chronological_origins(
    n_rows=len(target),
    lookback=LOOKBACK,
    horizon=HORIZON,
)
train_loader, validation_loader, test_loader = make_loaders(
    features,
    target,
    origins,
    lookback=LOOKBACK,
    horizon=HORIZON,
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
    TrainConfig(learning_rate=5e-4),
)

predictions_z, targets_z = predict(model, test_loader)
metrics = metrics_on_original_scale(
    predictions_z,
    targets_z,
    training_mean,
    training_std,
)
print(metrics)
```

## Important preprocessing boundary

The companion pipeline deliberately does not fit scalers or construct lagged features. Those operations depend on the dataset and must happen causally:

- fit all learned preprocessing on the training period only;
- construct lags and rolling features without using future values;
- start all model variants at identical valid forecast origins;
- ensure each target horizon remains fully inside its assigned chronological split;
- record how far every engineered feature reaches into raw history.

These rules are part of the model comparison, not incidental data cleaning.
