# LSTM Forecasting: Learning What to Remember

*An intuitive guide to gates, cell state, and direct multi-horizon prediction*

**Series:** Sequence Models for Prediction, Part 5 of 16
**Suggested Medium tags:** LSTM, Recurrent Neural Networks, Time Series, Forecasting, Deep Learning

An LSTM reads a time series the way we often describe ourselves reading a story: one step at a time, carrying forward a changing memory of what mattered.

The metaphor is useful, but incomplete. An LSTM does not store selected observations in a neat archive. It continually transforms a fixed-size state. Its gates learn how much of the previous state to retain, how much new information to write, and how much memory to expose as the current output.

That mechanism was designed to address the difficulty ordinary recurrent networks have learning long-range dependencies.

## The two-state intuition

At time `t`, an LSTM maintains:

- a **cell state** `c_t`, the longer-lived memory stream;
- a **hidden state** `h_t`, the exposed representation used by the next step and downstream layers.

Four learned calculations update them:

```text
forget gate:    f_t = sigmoid(...)
input gate:     i_t = sigmoid(...)
candidate:      g_t = tanh(...)
output gate:    o_t = sigmoid(...)

cell:           c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t
hidden:         h_t = o_t ⊙ tanh(c_t)
```

The sigmoid gates produce values between zero and one. Element by element, the forget gate scales old memory, the input gate controls new content, and the output gate controls what becomes visible.

This additive cell update gives gradients a more stable route through time than repeatedly multiplying through a plain recurrent transition.

## Compressing a window into a final state

A simple forecaster can use one LSTM layer with 64 hidden units:

```python
self.rnn = nn.LSTM(
    input_size=n_features,
    hidden_size=64,
    batch_first=True,
)
```

The LSTM emits a hidden state at every historical hour. The model keeps only the final one, applies dropout, and maps it to 24 predictions:

```python
out, _ = self.rnn(x)
forecast = self.fc(self.drop(out[:, -1]))
```

The entire week is therefore compressed into one 64-dimensional vector.

This is efficient, but it creates an information bottleneck. The network must preserve everything useful for all 24 future horizons in its final state: level, daily rhythm, weekday identity, recent direction, volatility, and unusual events.

Attention-based decoders or pooling over all recurrent states can reduce that bottleneck. The final-state design is simply the smallest useful version.

## Sequence length and practical memory

The LSTM has a theoretical path from the first input to the final state, but that does not mean it preserves every early value accurately.

At each step, the forget gate scales the existing cell state and new content competes to enter it. A useful signal can gradually decay or be overwritten. Longer sequences also make optimization slower because recurrent steps must be evaluated in order.

Hidden width determines how much state the network can carry. More units increase capacity and cost; additional stacked LSTM layers create a deeper transformation at every time step. Bidirectional LSTMs can read a fully observed historical window from both directions, but they are not appropriate when outputs at each step must remain strictly online.

## Forecast-head choices

The final state can feed:

- one next-step prediction;
- a vector containing every future horizon;
- a decoder that generates the future sequentially;
- separate horizon-specific attention queries over all hidden states.

Direct vector output is fast and avoids feeding predictions back into the model. An autoregressive decoder can model relationships among future steps more explicitly but accumulates error and complicates training.

## LSTM strengths

LSTMs remain useful because they combine several practical properties:

- parameter sharing across time;
- variable-length sequence support;
- a state that naturally supports online updates;
- better long-range gradient flow than a plain RNN;
- modest parameter growth as input features are added.

Input features affect the gate matrices, so parameter count grows with channel width, but much less dramatically than in a flattened MLP.

## LSTM limitations

Recurrent processing is sequential. Training cannot parallelize across time as freely as a convolution or Transformer.

Theoretical memory is not the same as reliable recall. A 64-dimensional state can forget precise old observations, especially when many signals compete for representation.

The final-state forecast head creates another bottleneck. It does not revisit particular historical hours for different future horizons.

And an LSTM does not solve nonstationarity. It can learn a stable historical dependency beautifully and fail when behavior changes.

## Complete PyTorch implementation

This complete version keeps the state sequence returned by PyTorch, selects the final historical state, applies dropout, and emits the whole forecast horizon directly.

```python
import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_features,
        horizon,
        hidden=64,
        dropout=0.10,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        # states: [batch, lookback, hidden]
        states, _ = self.rnn(x)
        final_state = states[:, -1, :]
        return self.head(
            self.dropout(final_state)
        )


LOOKBACK = 168
N_FEATURES = 1
HORIZON = 24

model = LSTMForecaster(
    n_features=N_FEATURES,
    horizon=HORIZON,
)

x = torch.randn(32, LOOKBACK, N_FEATURES)
y_hat = model(x)
assert y_hat.shape == (32, HORIZON)
```

The same verified implementation is available in the [companion model module](code/sequence_models.py). The complete Dataset, DataLoader, and fit/evaluate workflow is explained in the [shared PyTorch training companion](pytorch-data-pipeline-and-training-loop.md).

## Training considerations

Standardize inputs using training-only statistics. Shuffle training windows if each window is an independent supervised example, while preserving chronological validation and test periods.

Gradient clipping can stabilize recurrent training when gradients become unusually large. Validation curves should determine early stopping, and hidden width, layer count, dropout, learning rate, and lookback length should be treated as real hyperparameters.

Stateful online inference is possible, but it changes the training and serving contract. A model trained on independent windows should not silently be deployed with hidden state carried forever between requests.

## When to choose an LSTM

Choose an LSTM when ordered state evolution is a natural inductive bias, latency permits sequential processing, and the dataset is large enough to learn gate behavior.

Do not choose it merely because the data have timestamps. Always compare it with seasonal, linear, and simpler nonlinear baselines.

The LSTM’s core idea is controlled memory. Part 6 examines the GRU, which pursues a similar goal with fewer gates and a smaller state machine.

## Further reading

- [Hochreiter and Schmidhuber, “Long Short-Term Memory”](https://doi.org/10.1162/neco.1997.9.8.1735)

---

**Series navigation:** [Series index](README.md) · [Previous: MLP forecasting](03-mlp-forecaster.md) · [Next: GRU forecasting](05-gru-forecaster.md)
