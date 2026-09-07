# TCN Forecasting: Long Memory Through Dilated Convolutions

*How causal filters create an efficient sequence model—and how to calculate what it can actually see*

**Series:** Sequence Models for Prediction, Part 8 of 18
**Suggested Medium tags:** TCN, Convolutional Neural Networks, Time Series, Forecasting, Deep Learning

A temporal convolutional network replaces recurrent memory with a hierarchy of causal filters.

The key idea is dilation. A standard convolution reads neighboring positions. A dilated convolution leaves gaps between the positions it samples. Stack layers with exponentially increasing dilation, and the network can connect recent outputs to a long history using a short computational path.

This makes TCNs parallelizable across time while preserving a clear notion of historical direction.

## Causal convolution

For a forecast representation at historical time `u`, a causal convolution may use `u` and earlier inputs but not later ones.

With kernel size three and dilation one, the filter reads:

```text
u, u-1, u-2
```

With dilation two:

```text
u, u-2, u-4
```

With dilation eight:

```text
u, u-8, u-16
```

A compact PyTorch layer can implement causality by padding and then cropping the extra values from the right:

```python
y = self.conv(x)
if self.pad:
    y = y[:, :, :-self.pad]
```

The output length matches the input length, and no activation at position `u` depends on a later input row.

## Dilation grows memory exponentially

Consider four kernel-three layers with dilations:

```text
1, 2, 4, 8
```

For stride one, the theoretical receptive field is:

```text
R = 1 + (kernel_size - 1) × Σ dilations
  = 1 + 2 × (1 + 2 + 4 + 8)
  = 31 hours
```

This calculation should be done for every TCN design.

The input tensor contains 168 hours. The model’s final activation—the one used for forecasting—can depend on only the last 31 input rows.

Feeding a long tensor does not guarantee long memory.

## A minimal architecture

Each causal layer produces 64 channels, followed by ReLU and, in the first three layers, dropout. The final time position goes into a 24-output linear head.

```text
[batch, features, 168]
        ↓ dilation 1
        ↓ dilation 2
        ↓ dilation 4
        ↓ dilation 8
[batch, 64, 168]
        ↓ take final position
[batch, 64]
        ↓
[batch, 24]
```

This implementation is intentionally compact. Many production TCNs also use residual blocks, normalization, repeated layers per dilation, and carefully tuned kernels.

Residual connections improve optimization and make deeper stacks practical.

## Residual blocks make depth usable

A typical TCN groups one or more dilated convolutions into a residual block:

```text
block_output = activation(convolution_path(x) + shortcut(x))
```

If input and output channels differ, the shortcut can use a one-by-one convolution to match their dimensions. The direct path lets information and gradients bypass a block, so adding depth does not force every layer to rewrite the representation.

Normalization and dropout are commonly placed inside the convolutional path. Exact ordering varies, and it should be treated as part of the architecture rather than an inconsequential implementation detail.

## Producing a forecast

For direct multi-horizon forecasting, the model can take the final causal activation and map it to all future steps. This mirrors a recurrent final-state head.

Other options preserve the full TCN output sequence and apply pooling or horizon-specific queries. For sequence-to-sequence tasks, a causal output can be produced at every input position.

The final-position design is efficient and naturally aligned with the forecast origin. Its risk is compression: one channel vector must carry everything needed for every future horizon.

## Designing for the required memory

To cover the entire raw 168-hour window with one kernel-three layer per dilation, extend the stack:

```text
dilations = [1, 2, 4, 8, 16, 32, 64]
R = 1 + 2 × 127 = 255 hours
```

The larger field covers the full 168-step input. It also adds depth and computation, so receptive field, channel width, and residual structure should be tuned together.

Theoretical receptive field means that a path exists. Effective receptive field describes how strongly the trained network actually uses distant positions. Gradient-based sensitivity or controlled input perturbations can reveal whether the oldest context matters in practice.

## TCN strengths

TCNs offer:

- parallel computation across time;
- a controllable receptive field;
- short gradient paths to distant inputs;
- translation-equivariant local detectors;
- natural causal streaming behavior;
- modest parameter growth with additional features.

They are often a strong default when local and multi-scale temporal patterns dominate.

## TCN limitations

The receptive field must be designed, not assumed. A theoretically large field also does not guarantee that every old position has equal effective influence.

Padding can create boundary effects. Very deep dilation schedules may skip useful local detail unless layers are combined carefully. A final-position head can bottleneck information, and a compact TCN may need residual connections to train reliably.

## Complete PyTorch implementation

The complete implementation below reproduces the notebook’s compact four-layer TCN. `CausalConv1d` pads both sides internally and then removes the right-side outputs, ensuring that position `u` cannot use a later input. The calculated receptive field is stored on the model for inspection.

```python
import torch
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
    ):
        super().__init__()
        self.left_padding = (
            kernel_size - 1
        ) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.left_padding,
        )

    def forward(self, x):
        y = self.conv(x)
        if self.left_padding:
            y = y[:, :, :-self.left_padding]
        return y


class TCNForecaster(nn.Module):
    def __init__(
        self,
        n_features,
        horizon,
        channels=64,
        kernel_size=3,
        dilations=(1, 2, 4, 8),
        dropout=0.10,
    ):
        super().__init__()
        layers = []
        in_channels = n_features

        for index, dilation in enumerate(dilations):
            layers.extend([
                CausalConv1d(
                    in_channels,
                    channels,
                    kernel_size,
                    dilation,
                ),
                nn.ReLU(),
            ])
            if index < len(dilations) - 1:
                layers.append(nn.Dropout(dropout))
            in_channels = channels

        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(channels, horizon)
        self.receptive_field = 1 + (
            kernel_size - 1
        ) * sum(dilations)

    def forward(self, x):
        channel_first = x.transpose(1, 2)
        states = self.features(channel_first)
        final_state = states[:, :, -1]
        return self.head(final_state)


LOOKBACK = 168
N_FEATURES = 1
HORIZON = 24

model = TCNForecaster(
    n_features=N_FEATURES,
    horizon=HORIZON,
)

x = torch.randn(32, LOOKBACK, N_FEATURES)
y_hat = model(x)
assert y_hat.shape == (32, HORIZON)
assert model.receptive_field == 31
```

This intentionally matches the 31-step experimental model. To cover the entire 168-step input, pass `dilations=(1, 2, 4, 8, 16, 32, 64)` and consider adding residual blocks. The same verified compact implementation is available in the [companion model module](code/sequence_models.py). The full Dataset, DataLoader, and fit/evaluate workflow is explained in the [shared PyTorch training companion](pytorch-data-pipeline-and-training-loop.md).

## Training and failure modes

Very shallow TCNs underuse long inputs. Very deep ones can be difficult to optimize without residual connections. Large dilation jumps may miss useful fine-scale interactions unless smaller dilations remain in the stack.

Padding produces synthetic boundary values, and the final-position head may overemphasize recent context. Inspect performance as lookback and receptive field change, not just as channel count changes.

Part 9 moves from fixed filters to content-dependent global connections: the Transformer.

## Further reading

- [Bai, Kolter, and Koltun, “An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling”](https://arxiv.org/abs/1803.01271)

---

**Series navigation:** [Series index](README.md) · [Previous: 1D CNN forecasting](06-cnn1d-forecaster.md) · [Next: Transformer forecasting](08-transformer-forecaster.md)
