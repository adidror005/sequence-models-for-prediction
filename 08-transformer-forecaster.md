# Transformer Forecasting: Let Every Time Step Look at Every Other Step

*Self-attention intuition, positional information, and the limits of a generic encoder*

**Series:** Sequence Models for Prediction, Part 9 of 16
**Suggested Medium tags:** Transformers, Attention, Time Series, Forecasting, Deep Learning

A recurrent model carries the past forward. A convolution builds the past from local neighborhoods. A Transformer lets every input position interact directly with every other position.

That global connectivity is the attraction. A forecast representation can compare the most recent evening with yesterday evening or the same hour last week without passing information through 168 recurrent transitions.

The mechanism that creates those connections is self-attention.

## Query, key, and value intuition

For each historical token, the Transformer constructs three learned vectors:

- a **query** describing what this token is looking for;
- a **key** describing what this token offers;
- a **value** containing the information to retrieve.

Attention compares queries with keys, converts the scores into weights, and forms a weighted combination of values:

```text
Attention(Q, K, V) = softmax(QKᵀ / √d) V
```

For a seasonal series, one attention head might learn that the final historical token should retrieve similar positions from earlier cycles. Another might connect the latest ramp with older ramps, regardless of their exact position.

These interpretations are possibilities, not guaranteed meanings. Attention heads are learned numerical operations, and attention weights alone are not complete explanations.

## Why position must be added

Self-attention by itself does not know whether one token came before another. Permuting the tokens would permute the outputs in the same way.

A simple encoder can add a learned positional tensor with one vector for each of the 168 input locations:

```python
z = self.proj(x)
z = z + self.pos[:, :x.size(1)]
z = self.encoder(z)
```

This gives the model a stable identity for “oldest hour,” “24 hours ago,” and “most recent hour.”

Calendar features provide a different kind of position. Learned embeddings describe location relative to the input window; hour and weekday describe location in real-world cycles.

## A compact encoder architecture

The model projects each feature vector to dimension 64, then applies two Transformer encoder layers:

```text
input projection:     F → 64
attention heads:      4
encoder layers:       2
feed-forward width:   128
dropout:              0.10
```

After encoding all 168 tokens, the model keeps the final token, normalizes it, and maps it to 24 forecasts.

```python
return self.fc(self.norm(z[:, -1]))
```

The final token can attend to the entire input window. It still has to compress everything useful for all forecast horizons into 64 values.

Other designs use a forecast token, attention pooling, a decoder with one query per future horizon, or explicit future-known covariates. Final-token pooling is simply the smallest understandable version.

## Does it need a causal mask?

There is no causal attention mask inside the encoder.

That is acceptable for this setup. All 168 input tokens are observed before the forecast origin, and no target token enters the encoder. Letting an early historical token attend to a later historical token does not expose the future being predicted.

A causal mask would be required for autoregressive next-token training or any setup in which target-period values appear inside the sequence.

Causality is defined by the data boundary, not by copying a language-model mask into every time-series architecture.

## Global access is not free

Full attention compares every pair of tokens. Its time and memory cost scale roughly with the square of sequence length:

```text
168 tokens → 28,224 pairwise positions per head
336 tokens → 112,896 pairwise positions per head
```

Doubling context roughly quadruples the attention matrix. At one week of hourly data this is manageable; at years of minute bars it is not.

Patching, sparse attention, downsampling, or state-space approaches can reduce that burden.

## What happens inside an encoder block

Multi-head attention repeats the query-key-value operation in several learned subspaces. Different heads can specialize in different relationships, and their outputs are concatenated and projected back to the model dimension.

The attention result passes through a position-wise feed-forward network. Residual connections wrap both the attention and feed-forward sublayers, while layer normalization helps stabilize optimization.

Stacking blocks lets the model build higher-order relationships: the first layer can retrieve relevant observations, and the next can reason over the retrieved representations. More layers add capacity and cost but do not automatically improve forecasting.

## What this Transformer is—and is not

It is a generic encoder adapted to emit a direct 24-hour forecast.

It is not the Temporal Fusion Transformer, which distinguishes static variables, observed historical inputs, and known future covariates while adding gating and variable selection. It is not PatchTST, which tokenizes subseries and applies channel-independent processing. It is not an autoregressive language model for numbers.

“Transformer” names a family of attention-based building blocks, not one forecasting algorithm.

## Complete PyTorch implementation

This complete encoder matches the experiment. It projects each observation, adds a learned relative position, applies two pre-normalized encoder layers, and uses the final historical token for a direct 24-step forecast. No causal mask is needed because the tensor contains historical inputs only.

```python
import torch
from torch import nn


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        lookback,
        n_features,
        horizon,
        d_model=64,
        nhead=4,
        layers=2,
        dropout=0.10,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                "d_model must be divisible by nhead"
            )

        self.input_projection = nn.Linear(
            n_features,
            d_model,
        )
        self.position = nn.Parameter(
            torch.zeros(1, lookback, d_model)
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
        if x.size(1) > self.position.size(1):
            raise ValueError(
                "input exceeds configured lookback"
            )

        z = self.input_projection(x)
        z = z + self.position[:, :x.size(1), :]
        z = self.encoder(z)
        final_token = self.norm(z[:, -1, :])
        return self.head(final_token)


LOOKBACK = 168
N_FEATURES = 1
HORIZON = 24

model = TransformerForecaster(
    lookback=LOOKBACK,
    n_features=N_FEATURES,
    horizon=HORIZON,
)

x = torch.randn(32, LOOKBACK, N_FEATURES)
y_hat = model(x)
assert y_hat.shape == (32, HORIZON)
```

The same verified implementation is available in the [companion model module](code/sequence_models.py). The complete Dataset, DataLoader, and fit/evaluate workflow is explained in the [shared PyTorch training companion](pytorch-data-pipeline-and-training-loop.md).

## Training and failure modes

Transformers need enough data to learn both content relationships and useful positional structure. Small datasets can favor simpler models.

Watch for overfitting, unstable attention patterns, excessive memory use, and weak performance at particular horizons. Attention weights may be interesting to visualize, but they are not a substitute for perturbation tests and out-of-sample evaluation.

The input projection, model dimension, number of heads, feed-forward width, depth, dropout, positional scheme, and forecast head all affect behavior. “Using a Transformer” leaves many consequential choices unspecified.

## When to choose a Transformer

Choose it when global interactions are plausible, the dataset can support its capacity, and parallel training matters. Prefer a forecasting-specific design when known future covariates, static metadata, multiple variables, or probabilistic outputs are central.

Do not choose it because attention is fashionable. At this context length, linear, recurrent, and convolutional models already provide strong alternatives.

Part 10 reduces attention cost and changes the unit of representation from individual hours to patches.

## Further reading

- [Vaswani et al., “Attention Is All You Need”](https://papers.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
- [Lim et al., “Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting”](https://arxiv.org/abs/1912.09363)

---

**Series navigation:** [Series index](README.md) · [Previous: TCN forecasting](07-tcn-forecaster.md) · [Next: Patch Transformer forecasting](09-patch-transformer-forecaster.md)
