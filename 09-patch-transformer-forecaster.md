# Patch Transformer Forecasting: Turning Time Steps into Temporal Tokens

*Why grouping nearby observations can make attention cheaper—and exact lags harder to recover*

**Series:** Sequence Models for Prediction, Part 10 of 16
**Suggested Medium tags:** PatchTST, Transformers, Time Series, Forecasting, Deep Learning

A language Transformer receives words or subwords. What should count as a token in a time series?

One answer is an individual timestamp. Another is a short contiguous segment—a patch.

Patching changes the representation before attention begins. Instead of asking the Transformer to relate 168 individual tokens, a model can group them into fourteen 12-step tokens. Each token contains a local segment shape.

The idea trades temporal resolution for shorter attention and richer local units.

## From points to patches

For one feature channel, a 12-step patch is a vector of 12 values. With 16 channels, it contains 192 values.

A concrete example uses:

```text
lookback:   168 hours
patch size: 12 hours
stride:     12 hours
patches:    14
```

The patches do not overlap. PyTorch creates them with `unfold`, rearranges their dimensions, and flattens each patch across time and features:

```python
patches = x.unfold(dimension=1, size=12, step=12)
patches = patches.permute(0, 1, 3, 2).flatten(2)
```

A learned linear projection maps every flattened patch to a 64-dimensional token.

```text
[batch, 168, F]
      ↓ patch
[batch, 14, 12 × F]
      ↓ project
[batch, 14, 64]
```

The Transformer encoder now attends over 14 tokens rather than 168.

## Why patches can help

Individual scalar timestamps have little semantic content. A local segment can express a shape: rising morning use, an evening peak, or a quiet overnight period.

Patches offer three practical advantages:

1. **Shorter attention.** Pairwise attention positions fall from `168² = 28,224` to `14² = 196` per head.
2. **Local pattern packaging.** The projection can learn a compact representation of each 12-hour shape.
3. **Longer feasible context.** Because token count is smaller, the same attention budget can cover more raw history.

The cost is compression. The model must preserve useful within-patch timing through a single projected token.

## Exact offsets become awkward

A daily lag is 24 hours—exactly two patches in this setup. A weekly lag spans all fourteen.

That sounds convenient, but a 7 p.m. observation is not necessarily aligned to the same coordinate in every 12-hour patch unless forecast origins and patch boundaries share a stable phase. The projection must preserve within-patch position before attention can compare corresponding hours.

Overlapping patches can reduce boundary sensitivity. So can patch lengths aligned with known cycles in the domain.

Patch size is not merely a performance knob. It encodes a belief about the temporal unit that deserves a token.

## A simple patch architecture

After patch projection, the model adds learned positional embeddings and applies two encoder layers with four attention heads. It normalizes and averages all patch tokens, then maps the pooled representation to 24 future values.

```python
z = self.patch_proj(patches)
z = z + self.pos[:, :z.size(1)]
z = self.encoder(z)
forecast = self.fc(self.norm(z.mean(dim=1)))
```

Mean pooling treats every patch symmetrically after positional encoding. It is efficient, but it may dilute recency. A learned forecast token or horizon-specific query could let the model retrieve different patches for different future hours.

## Why this is not a PatchTST benchmark

The influential PatchTST architecture combines patching with channel independence: each variable is treated as a separate univariate sequence while projection and Transformer weights are shared across channels.

A simpler implementation may flatten all feature channels together inside each patch. That is a reasonable educational patch Transformer, but it is structurally different.

This distinction matters. PatchTST can motivate the design without making every patched encoder a reproduction of the published architecture.

## Choosing patch length and stride

Short patches preserve detail but produce more tokens. Long patches reduce attention cost but demand more compression from the projection layer.

Stride controls overlap. A stride equal to patch length is cheap and simple. A smaller stride lets patterns cross patch boundaries and increases token count.

Treat patch length and stride as model hyperparameters. Test whether performance changes when forecast origins shift relative to patch boundaries; a good design should not succeed only under one lucky alignment.

## Complete PyTorch implementation

The complete implementation shows the dimension change that is easy to get wrong. PyTorch’s `unfold` places the feature dimension before the patch dimension, so the tensor is permuted before time and features are flattened into one patch vector.

```python
import torch
from torch import nn


class PatchTransformerForecaster(nn.Module):
    def __init__(
        self,
        lookback,
        n_features,
        horizon,
        patch_len=12,
        stride=12,
        d_model=64,
        nhead=4,
        layers=2,
        dropout=0.10,
    ):
        super().__init__()
        if lookback < patch_len:
            raise ValueError(
                "lookback must be at least patch_len"
            )
        if d_model % nhead != 0:
            raise ValueError(
                "d_model must be divisible by nhead"
            )

        self.patch_len = patch_len
        self.stride = stride
        n_patches = 1 + (
            lookback - patch_len
        ) // stride

        self.patch_projection = nn.Linear(
            patch_len * n_features,
            d_model,
        )
        self.position = nn.Parameter(
            torch.zeros(1, n_patches, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        patches = x.unfold(
            dimension=1,
            size=self.patch_len,
            step=self.stride,
        )
        patches = patches.permute(
            0, 1, 3, 2
        ).flatten(start_dim=2)

        if patches.size(1) > self.position.size(1):
            raise ValueError(
                "input creates too many patches"
            )

        z = self.patch_projection(patches)
        z = z + self.position[:, :z.size(1), :]
        z = self.encoder(z)
        pooled = self.norm(z.mean(dim=1))
        return self.head(pooled)


LOOKBACK = 168
N_FEATURES = 1
HORIZON = 24

model = PatchTransformerForecaster(
    lookback=LOOKBACK,
    n_features=N_FEATURES,
    horizon=HORIZON,
    patch_len=12,
    stride=12,
)

x = torch.randn(32, LOOKBACK, N_FEATURES)
y_hat = model(x)
assert y_hat.shape == (32, HORIZON)
```

This is the simplified multichannel patch Transformer used in the notebook, not a complete PatchTST reproduction. The same verified implementation is available in the [companion model module](code/sequence_models.py). The full Dataset, DataLoader, and fit/evaluate workflow is explained in the [shared PyTorch training companion](pytorch-data-pipeline-and-training-loop.md).

## Common failure modes

Patches can hide sharp events, blur exact offsets, and make recency harder to preserve under mean pooling. Wide multichannel patches can also make the projection layer large.

The model may appear efficient because attention is shorter while silently moving most parameters into the patch projection. Report both token count and total parameter count.

## When to choose patching

Patching is attractive when context is long, local segments form meaningful units, and full point-wise attention is expensive.

Always test:

- patch length;
- stride and overlap;
- boundary alignment;
- channel mixing;
- pooling or decoder strategy;
- performance by forecast horizon.

The best patch is a property of the data and forecasting interface, not a universal constant.

Part 11 turns away from attention and returns to dense layers, arranged as iterative backcast and forecast residual blocks: N-BEATS-style forecasting.

## Further reading

- [Nie et al., “A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers”](https://openreview.net/pdf?id=Jbdc0vTOcol)

---

**Series navigation:** [Series index](README.md) · [Previous: Transformer forecasting](08-transformer-forecaster.md) · [Next: N-BEATS-style forecasting](10-nbeats-style-forecaster.md)
