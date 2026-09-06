# 1D CNN Forecasting: Learning Local Shapes in Parallel

*How temporal filters detect motifs—and why pooling can erase the positions a forecast needs*

**Series:** Sequence Models for Prediction, Part 7 of 15
**Suggested Medium tags:** Convolutional Neural Networks, Time Series, Forecasting, Deep Learning, CNN

A one-dimensional convolution does not remember a sequence. It scans it.

A small filter moves across time and asks the same question at every position. One filter might activate on a sharp evening ramp. Another might detect a peak followed by a decline. A deeper layer combines those local motifs into more abstract shapes.

This weight sharing gives a temporal CNN an attractive property: a useful pattern can be recognized wherever it appears in the input window.

## The filter intuition

Take a kernel covering five hours. For one input channel, it contains five learned weights. At each position, the convolution computes a weighted combination of those neighboring values.

```text
historical sequence:  ─────────────────────────────
five-hour filter:             [ w1 w2 w3 w4 w5 ]
                                      ↓ slide
feature map:         ─────────────────────────────
```

With multiple input channels, each filter combines time and feature dimensions. A detector can respond jointly to demand, hour-of-day phase, and recent volatility.

The same weights are reused at every time position. This is the convolutional inductive bias: local patterns matter, and their detector should not have to be relearned separately for every hour in the window.

## A simple forecasting architecture

A compact model can apply two five-step convolutions with 64 channels:

```python
self.net = nn.Sequential(
    nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.Dropout(0.10),
    nn.Conv1d(64, 64, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.AdaptiveAvgPool1d(1),
)
```

PyTorch convolutions expect channels before time, so the input changes from:

```text
[batch, time, features]
```

to:

```text
[batch, features, time]
```

After the convolutional stack, adaptive average pooling reduces every 168-step feature map to one number. A linear layer maps the resulting 64 values to the 24-hour forecast.

## Receptive field: only nine hours per local activation

One five-step convolution sees five adjacent hours. Stacking a second stride-one five-step convolution expands the receptive field to nine:

```text
R = 1 + (5 - 1) + (5 - 1) = 9 hours
```

The global average pool aggregates activations from the full week, so the final vector reflects motifs found anywhere in the window. But each motif detector is built from only nine-hour neighborhoods.

This is very different from globally comparing the current hour with the same hour yesterday or last week.

## The pooling trade-off

Average pooling answers:

> How strongly did this pattern occur across the window?

It is less suited to:

> Did this pattern occur specifically during the most recent evening?

Exact position matters in forecasting. An evening spike six days ago and one occurring an hour ago should not necessarily influence the next hour equally.

Calendar channels can partly restore position because the local activation can include hour and weekday identity. A lag feature can bring a meaningful remote comparison into the local nine-hour neighborhood.

Alternative CNN heads could preserve more timing information:

- take the final activation instead of the global average;
- use attention pooling;
- pool separately over recent and older segments;
- flatten the final map, accepting a larger parameter count;
- add causal dilation to grow the receptive field.

The last option leads directly to the TCN in Part 8.

## Is symmetric padding a forecasting leak?

A symmetric five-step convolution can use `padding=2`, so each intermediate activation can inspect neighbors on both sides of a historical position.

That is not target leakage here. The complete 168-hour window is observed before the 24-hour forecast is issued. A feature at an earlier input position may use a later value inside that same historical window without seeing any target value.

For streaming, step-by-step representations, or sequence labeling, causal padding would matter. Forecasting causality depends on the boundary between observed context and future target, not on a blanket rule that every internal layer must be causal.

## The main design choices

Kernel size controls the width of each local pattern. Depth and dilation control how local patterns combine across a longer range. Channel count controls how many detectors the model can learn.

The forecast head is just as important as the convolutional body. Global average pooling emphasizes whether a motif occurred, final-position pooling emphasizes recent causal context, and attention pooling learns which positions to retrieve.

Normalization, residual connections, and dropout become increasingly important as the stack deepens. The receptive field should always be calculated explicitly rather than inferred from the input length.

## Complete PyTorch implementation

The complete implementation includes the required time-to-channel transpose, both convolutional layers, global average pooling, and the direct forecast head.

```python
import torch
from torch import nn


class CNN1DForecaster(nn.Module):
    def __init__(
        self,
        n_features,
        horizon,
        channels=64,
        dropout=0.10,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(
                n_features,
                channels,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(channels, horizon)

    def forward(self, x):
        # Conv1d expects [batch, channels, time].
        channel_first = x.transpose(1, 2)
        summary = self.features(
            channel_first
        ).squeeze(-1)
        return self.head(summary)


LOOKBACK = 168
N_FEATURES = 1
HORIZON = 24

model = CNN1DForecaster(
    n_features=N_FEATURES,
    horizon=HORIZON,
)

x = torch.randn(32, LOOKBACK, N_FEATURES)
y_hat = model(x)
assert y_hat.shape == (32, HORIZON)
```

The same verified implementation is available in the [companion model module](code/sequence_models.py). The complete Dataset, DataLoader, and fit/evaluate workflow is explained in the [shared PyTorch training companion](pytorch-data-pipeline-and-training-loop.md).

## Common failure modes

A shallow CNN can receive a long sequence while each activation sees only a tiny neighborhood. Aggressive global pooling may erase recency and exact position. Boundary padding can create artificial patterns near the beginning and end of the window.

Filters can also learn motifs that look meaningful but are unstable out of sample. Test different window alignments and chronological periods before treating a learned shape as reusable.

## When to choose a shallow 1D CNN

Choose it when local motifs are important, parallel training matters, and you want a compact model with shared temporal filters.

Be deliberate about three choices:

1. receptive field;
2. causal versus full-context convolution;
3. the pooling mechanism that turns a feature map into a forecast.

Those choices determine what the model can know far more than the label “CNN.”

Part 8 expands the receptive field with dilation and keeps the final representation tied to the end of the observed sequence: the temporal convolutional network.

---

**Series navigation:** [Series index](README.md) · [Previous: GRU forecasting](05-gru-forecaster.md) · [Next: TCN forecasting](07-tcn-forecaster.md)
