# Linear Forecasting: The Baseline That Sees the Whole Window

*Why a single dense layer is often the most revealing model in a time-series experiment*

**Series:** Sequence Models for Prediction, Part 3 of 18
**Suggested Medium tags:** Linear Regression, Time Series, Forecasting, Machine Learning, Baselines

A linear forecaster is easy to underestimate.

It has no recurrent state, convolution, attention, or learned notion of memory. Yet after we flatten a time window, a linear model can connect every historical value directly to every future horizon.

That makes it more than a weak baseline. It is a diagnostic instrument.

If a large neural model cannot beat a well-constructed linear forecast, either the relationship is mostly linear, the dataset is too small for the complex model, or the experiment is rewarding the wrong behavior.

## The mental model: one weighted forecast per horizon

Suppose the input contains 168 hours and `F` features per hour. Flattening produces a vector with `168 × F` values.

For forecast horizon `h`, the linear model learns:

```text
prediction[h] = bias[h]
              + Σ weight[h, time, feature] × input[time, feature]
```

A direct 24-step implementation is only a few lines:

```python
class LinearForecaster(nn.Module):
    def __init__(self, n_features, horizon=24):
        super().__init__()
        self.fc = nn.Linear(168 * n_features, horizon)

    def forward(self, x):
        return self.fc(x.flatten(1))
```

Each output hour receives its own set of weights. The model can learn that hour one depends heavily on the most recent demand, while tomorrow evening depends more on the same time yesterday.

There is no requirement that adjacent timestamps receive similar weights. Time order matters only because each flattened position has a fixed column index.

## Why this counts as a sequence baseline

The architecture is not sequential, but the input is. Position 167 always means “most recent hour,” position 144 means “24 hours ago,” and position 0 means “168 hours ago.”

A direct path connects each position to every forecast horizon. The linear model does not have to transport the oldest observation through 168 recurrent updates or a stack of local convolutions.

Its memory is global but rigid.

That rigidity is both its strength and weakness:

- it can learn stable lag relationships efficiently;
- it cannot make the importance of a lag depend nonlinearly on the current regime;
- it assumes the positional relationship learned during training continues into validation and test;
- it cannot share a learned local pattern across different positions unless the data force similar weights.

## Input width and regularization

With one input channel, the input width is 168 and the layer contains 4,056 trainable parameters:

```text
168 inputs × 24 outputs + 24 biases
```

With six channels, the width becomes 1,008 and the model grows to 24,216 parameters. With 16 features, it reaches 64,536 parameters.

This is a crucial caveat. Adding features changes three things at once:

1. the values available to the model;
2. the number of coefficients it must estimate;
3. the amount of regularization needed to control those coefficients.

Time-series inputs are often strongly correlated: adjacent values resemble one another, and derived variables may overlap. Multicollinearity does not necessarily prevent prediction, but it can make coefficients unstable and optimization more sensitive.

A regularized linear model—ridge regression, for example—is often a stronger default than an unpenalized fit. L1 regularization can produce sparse weights, while elastic net combines sparsity with stability among correlated inputs.

## How the model learns

For point forecasting, the usual objective is mean squared error or mean absolute error across samples and forecast horizons.

Classical linear regression can be solved directly with linear algebra. In a neural-network pipeline, it is often trained with the same mini-batch optimizer as the larger models. That keeps the surrounding code consistent, although optimizer settings and regularization still need tuning.

There are two common multi-horizon designs:

- one model produces all future horizons jointly;
- a separate linear model is trained for each horizon.

A single 24-output layer is mathematically close to fitting 24 linear regressions that share an input matrix. It does not model dependencies among its output errors, but it is simple, fast, and easy to inspect.

## The hidden advantage: interpretability by horizon

Reshape the learned weight matrix from:

```text
[24, 168 × F]
```

into:

```text
[forecast horizon, historical time, feature]
```

Now we can visualize which past hours influence each future hour.

For seasonal demand, we might expect diagonal bands: a future evening hour placing weight near recent evening observations.

Coefficient plots are not automatically causal explanations. Correlated inputs can divide weight arbitrarily. But they are an excellent debugging tool. If a supposedly causal model places surprising weight on a padding artifact or a missingness pattern, we learn that before deploying it.

## Complete PyTorch implementation

This is the complete direct linear model used in the experiment. It accepts any feature count while keeping the lookback and horizon explicit. The final assertion is a useful shape test before training.

```python
import torch
from torch import nn


class LinearForecaster(nn.Module):
    def __init__(self, lookback, n_features, horizon):
        super().__init__()
        self.fc = nn.Linear(
            lookback * n_features,
            horizon,
        )

    def forward(self, x):
        # x: [batch, lookback, features]
        flat_history = x.flatten(start_dim=1)
        return self.fc(flat_history)


LOOKBACK = 168
N_FEATURES = 1
HORIZON = 24

model = LinearForecaster(
    lookback=LOOKBACK,
    n_features=N_FEATURES,
    horizon=HORIZON,
)

x = torch.randn(32, LOOKBACK, N_FEATURES)
y_hat = model(x)
assert y_hat.shape == (32, HORIZON)
```

The same verified implementation is available in the [companion model module](code/sequence_models.py). The complete Dataset, DataLoader, and fit/evaluate workflow is explained in the [shared PyTorch training companion](pytorch-data-pipeline-and-training-loop.md).

## Common failure modes

A linear forecaster fails when the relationship changes with context. It cannot express rules such as “use yesterday’s pattern on weekdays but a different pattern on weekends” unless interaction terms are supplied explicitly.

It can also look better than it is when the target is persistent. Always compare it with a domain-specific naive forecast, use chronological evaluation, and inspect each forecast horizon separately.

Finally, coefficients should not be mistaken for causal effects. Correlation among historical positions means several weight patterns can make nearly identical predictions.

## When to choose the linear forecaster

Use it when:

- you need a fast, transparent benchmark;
- the dataset is modest;
- stable seasonal lags dominate;
- you want a coefficient map by forecast horizon;
- you need to detect whether complexity is earning its cost.

Move beyond it when interactions and regimes are repeatable enough to justify nonlinear capacity.

The linear forecaster has no glamour, which is precisely why it is valuable. It gives the rest of the model zoo nowhere to hide.

Part 4 adds nonlinear interactions while preserving global access to the window: the multilayer perceptron.

---

**Series navigation:** [Series index](README.md) · [Previous: PyTorch data pipeline](pytorch-data-pipeline-and-training-loop.md) · [Next: MLP forecasting](03-mlp-forecaster.md)
