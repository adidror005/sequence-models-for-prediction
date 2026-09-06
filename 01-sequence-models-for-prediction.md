# Sequence Models for Prediction: A Practical Map

*How inputs, outputs, memory, and model families fit together before we write any architecture code*

**Series:** Sequence Models for Prediction, Part 1 of 16
**Suggested Medium tags:** Time Series, Forecasting, Deep Learning, Neural Networks, Machine Learning

A sequence is an ordered collection in which position and context matter.

It might contain words, sensor measurements, transactions, medical events, audio samples, video frames, or market observations. Nearby elements may be related, patterns may repeat at several scales, and the meaning of one element can depend on what came before it.

Before comparing LSTMs, convolutional networks, Transformers, and N-BEATS, we need a shared language for sequence prediction. Otherwise architecture names conceal differences in data, targets, and evaluation.

This article builds that language. Part 2 builds the shared PyTorch data pipeline, Parts 3–11 explain the algorithms one at a time, and Part 12 compares them as design choices. Electricity forecasting and one-minute financial direction appear later as case studies rather than the premise of the series.

## The series roadmap

This article also serves as the evolving table of contents:

1. **Sequence Models for Prediction: A Practical Map** — the shared language developed here.
2. [A Leak-Free PyTorch Dataset, DataLoader, and Training Loop](pytorch-data-pipeline-and-training-loop.md) — the shared forecasting machinery.
3. [Linear Forecasting](02-linear-forecaster.md) — the transparent global baseline.
4. [MLP Forecasting](03-mlp-forecaster.md) — nonlinear prediction from a flattened window.
5. [LSTM Forecasting](04-lstm-forecaster.md) — gated recurrent memory.
6. [GRU Forecasting](05-gru-forecaster.md) — a leaner recurrent state machine.
7. [1D CNN Forecasting](06-cnn1d-forecaster.md) — reusable local temporal patterns.
8. [TCN Forecasting](07-tcn-forecaster.md) — causal convolution with designed long-range memory.
9. [Transformer Forecasting](08-transformer-forecaster.md) — content-dependent global access.
10. [Patch Transformer Forecasting](09-patch-transformer-forecaster.md) — local segments as attention tokens.
11. [N-BEATS-Style Forecasting](10-nbeats-style-forecaster.md) — backcast and forecast residual refinement.
12. [Which Time-Series Model Should You Use?](11-choosing-a-sequence-model.md) — a practical selection framework.
13. [Comparing Sequence Models on Electricity Prediction](12-electricity-results-and-interpretation.md) — raw-history results.
14. [Do Calendar and Lagged Features Help?](13-calendar-and-lagged-features.md) — the later representation experiment.
15. [How to Design an Experiment You Can Trust](14-designing-a-trustworthy-experiment.md) — evaluation discipline.
16. [When Sequence Models Meet Market Noise](15-finance-direction-case-study.md) — a one-minute META direction case study with six neural models and CatBoost.

As each installment goes live, replace its local draft link with the public Medium URL. Unpublished titles can remain plain text so readers never encounter a dead link.

Part 2 implements the shared forecasting machinery before choosing an architecture: [A Leak-Free PyTorch Dataset, DataLoader, and Training Loop](pytorch-data-pipeline-and-training-loop.md).

## The main sequence-prediction tasks

Different tasks ask a model to produce different output structures.

### Sequence to one

The complete input sequence produces one result: sentiment from a sentence, equipment-failure risk from sensor history, or fraud probability from a transaction trail.

```text
[x₁, x₂, ..., xₜ] → y
```

The output can be a class, probability, or continuous value.

### Sequence to sequence

The model produces an output sequence: a label for every video frame, a translated sentence, or a forecast path.

```text
[x₁, x₂, ..., xₜ] → [y₁, y₂, ..., yₖ]
```

Input and output lengths may be equal or different.

### Next-element prediction

The model predicts what comes next from everything observed so far. Language modeling and one-step forecasting share this structure even though their targets are very different.

### Multi-step prediction

The model predicts several future values from an observed prefix. This will be our main numerical example because it makes memory, causality, and evaluation especially concrete.

The same architectural ideas recur across all four tasks: what context can the model access, how is that context compressed, and how does the output retrieve it?

## Time-series forecasting begins at an origin

Imagine standing at time `t`. Everything before `t` is observed. Everything from `t` onward is unknown.

`t` is the **forecast origin**.

We choose a lookback length `L` and forecast horizon `H`:

```text
observed context                   future target

x[t-L] ... x[t-2] x[t-1]   │   y[t] y[t+1] ... y[t+H-1]
                            ↑
                      forecast origin
```

Sliding the origin through history creates supervised examples. With a 168-step lookback and 24-step horizon, one example has:

```text
X: [168, number_of_features]
y: [24]
```

A mini-batch adds the first dimension:

```text
X: [batch, 168, features]
y: [batch, 24]
```

Every algorithm in this series accepts that same conceptual contract. What changes is how it moves information from the context to the forecast.

## Four ways to produce the future

The forecast head matters as much as the historical encoder.

### One-step prediction

The model predicts only the next value. This is simple to train and evaluate, but many applications need an entire future path.

### Recursive prediction

Predict one step, append that prediction to the context, and repeat.

Recursive forecasting can use one shared next-step model for any horizon. Its weakness is error accumulation: later predictions are conditioned on earlier guesses rather than observations.

### Direct multi-horizon prediction

Emit all `H` values at once:

```text
historical representation → [ŷ_t, ŷ_(t+1), ..., ŷ_(t+H-1)]
```

This avoids recursive feedback and lets each horizon learn a different relationship with the past. The electricity example uses this design.

### Encoder-decoder prediction

An encoder represents the history, while a decoder generates or queries the future. Decoders can receive known future information and model dependencies among forecast steps, at the cost of more complexity.

None of these choices is universally best. They define different learning problems even when the historical model is identical.

## What counts as an input feature?

Forecasting systems usually combine several kinds of information.

### Observed historical variables

These are measurements known only through the forecast origin: demand, temperature readings, prices, volume, sensor values, or sales.

### Known future variables

These are available for the target period before it occurs: hour of day, holidays, scheduled promotions, tariffs, or planned capacity.

Known future variables should be passed through an explicit future path when possible. Attaching calendar fields only to historical rows is not the same thing.

### Static variables

These do not change within the sequence: location, customer type, device category, or asset identity.

### Derived historical variables

Lags, differences, rolling means, and rolling volatility summarize observed history. They can be useful, redundant, or misleading depending on how they are constructed. Part 14 treats that as its own question.

The algorithm articles do not assume a particular feature recipe. Each model can receive one channel or many.

## Why time order changes evaluation

Ordinary random train/test splitting is usually wrong for forecasting. It allows later periods to influence training while earlier periods appear in validation, reversing the real prediction direction.

Use chronological splits:

```text
training period │ validation period │ test period
```

The validation period selects hyperparameters and checkpoints. The final test period estimates future-like performance only after model choices are fixed.

Rolling-origin evaluation repeats this process across several dates, revealing whether a conclusion depends on one unusually easy or difficult period.

Scaling also respects time. Means, standard deviations, quantiles, and learned preprocessing parameters are fitted on training data, then applied unchanged to validation and test data.

## Overlapping windows are not independent observations

Hourly windows shifted by one hour share nearly all their contents. A dataset may contain tens of thousands of windows without containing tens of thousands of independent situations.

This has several consequences:

- random splitting causes near-duplicate leakage;
- confidence intervals based on independent samples become too optimistic;
- large networks can memorize period-specific detail;
- boundaries must ensure forecast targets remain inside their assigned split.

The number of generated windows is not the same as the amount of independent evidence.

## Always begin with naive baselines

A forecasting model should beat a rule that captures the domain’s obvious structure.

Examples include:

- repeat the most recent value;
- repeat the corresponding value from the previous season;
- predict zero change;
- predict the training-period mean;
- use a regularized linear model over the same window.

The correct baseline depends on the target. For a persistent price series, repeating the last price can be formidable. For strongly daily demand, repeating yesterday may be more meaningful.

Complexity earns its place only through out-of-sample improvement over these rules.

## The uncomfortable strong baseline: gradient boosting

Naive rules and linear models are not enough. For small and medium-sized forecasting datasets, a boosted-tree model such as CatBoost, LightGBM, or XGBoost is often competitive with—and sometimes better than—far more elaborate sequence networks.

The tree model does not discover temporal order in the same way. We usually construct a causal table containing lags, calendar variables, rolling statistics, differences, and other information available at the forecast origin. Boosting then excels at thresholds, nonlinear interactions, mixed feature scales, and limited-data optimization.

This is not an unfair comparison. The practitioner’s goal is accurate, reliable prediction, not proving that a preferred neural architecture can win. Every serious sequence-model experiment in this series should therefore include a boosted-tree benchmark using the same forecast origins and permitted information.

If CatBoost wins, the lesson is not that sequence models are useless. It is that this dataset rewards explicit tabular structure more than the tested networks’ learned representations—and that is exactly the kind of result a benchmark should reveal.

## The model families in this series

The next nine articles form a progression of inductive biases.

### Linear forecaster

Every historical coordinate connects directly to every future horizon through one weighted sum. It is transparent, global, and rigid.

### Multilayer perceptron

The complete window is flattened and passed through nonlinear dense layers. It gains flexible interactions but no built-in understanding of local time structure.

### LSTM

The sequence is processed step by step through a gated cell and hidden state. The architecture learns what to retain, update, and expose.

### GRU

A simpler gated recurrent unit carries one state with update and reset gates, often using fewer parameters than an LSTM.

### 1D CNN

Shared filters scan for local temporal motifs in parallel. Kernel size, depth, and pooling determine what information survives.

### Temporal convolutional network

Causal, dilated convolutions expand the receptive field while retaining parallel computation.

### Transformer

Self-attention lets each historical token interact directly with every other token, with positional information added separately.

### Patch Transformer

Short segments become tokens, reducing attention cost and learning representations of local shapes.

### N-BEATS

Fully connected residual blocks repeatedly explain the historical window and add contributions to the forecast.

The goal is not to rank them in the abstract. Each architecture makes a different assumption about how temporal information should be reused, compressed, and retrieved.

## Point forecasts are only one output

The examples begin with a single predicted value per horizon. Real decisions often require uncertainty.

A model can instead predict:

- several quantiles;
- parameters of a probability distribution;
- samples from possible future paths;
- lower and upper prediction intervals.

Uncertainty is not created by adding error bars after training. The output head, loss, and calibration procedure must support it.

## A checklist before choosing an algorithm

Write down:

1. What is the forecast origin?
2. Which variables are observed historically, known in the future, or static?
3. What are the lookback and horizon?
4. Is the output one-step, recursive, direct, or decoder based?
5. What naive rule must the model beat?
6. How will chronological generalization be measured?
7. Do we need a point forecast or a distribution?

Once those choices are explicit, architecture comparisons become meaningful.

Part 2 builds the Dataset, DataLoaders, and training loop shared by every model. Part 3 then begins with the least complicated learned architecture and one of the most important: the direct linear forecaster.

---

**Series navigation:** [Series index](README.md) · [Next: PyTorch data pipeline](pytorch-data-pipeline-and-training-loop.md)
