# Medium series: Sequence Models for Prediction

This folder contains a 15-part series that starts with the general problem of learning from ordered data, builds a leak-free PyTorch pipeline, explains nine prediction algorithms individually, and only then compares them on an electricity-forecasting task.

The public GitHub Pages edition is published at [Sequence Models for Prediction](https://adidror005.github.io/sequence-models-for-prediction/).

The editorial angle is **model-first, feature-second**:

1. What makes prediction from sequences different from ordinary tabular prediction?
2. How does each architecture move information from past observations to future outputs?
3. What happens when every architecture receives the same raw history?
4. When do calendar variables, lags, differences, and rolling summaries add value?

Across all four questions, CatBoost or another gradient-boosted tree model remains a mandatory reality check. Sophisticated sequence architectures often lose to boosted trees on carefully constructed tabular time-series features, especially when the amount of independent training data is modest.

Stock prediction is a future application of the same framework. It is deliberately not part of the numbered series until the experiment has been specified and executed.

## Publication structure

### Parts 1–11: foundations and standalone algorithm explainers

The opening article introduces sequence inputs, prediction targets, windowing, forecasting horizons, chronological validation, and recurrent versus convolutional, attention-based, and fully connected approaches. Part 2 then implements the shared Dataset, DataLoaders, and training loop used throughout the series.

Each following article explains one model independently. The posts focus on intuition, information flow, equations, tensor shapes, strengths, limitations, implementation choices, and common failure modes. Readers do not need the electricity experiment to understand them.

### Part 12: choosing an architecture

The synthesis article compares the models as design choices: local versus global access, fixed windows versus recurrent state, effective memory, data requirements, compute cost, and useful baselines.

### Parts 13–14: the electricity case study

Part 13 holds the representation fixed. All nine models receive the same 168 hours of raw demand and predict the same next 24 hours. This isolates the practical architecture comparison.

Part 14 then changes the inputs. It tests historical calendar variables and engineered lagged history, while explaining the hidden changes in information reach, parameter count, and optimization.

### Part 15: experimental discipline

The closing article turns the lessons into a reusable protocol for evaluating any sequence-prediction system without temporal leakage or misleading comparisons.

## Publication order

| Part | Working title | Central question |
|---:|---|---|
| 1 | Sequence Models for Prediction: A Practical Map | What is a sequence-prediction problem, and how should it be framed? |
| 2 | Linear Forecasting: The Baseline That Sees the Whole Window | How far can a direct linear map take us? |
| 3 | MLP Forecasting: A Nonlinear Map from Past Window to Future Path | What does a dense network add to a fixed historical window? |
| 4 | LSTM Forecasting: Learning What to Remember | How do gates create a persistent recurrent memory? |
| 5 | GRU Forecasting: A Leaner Gated Memory | What does the GRU simplify, preserve, and trade away? |
| 6 | 1D CNN Forecasting: Learning Local Shapes in Parallel | How do temporal filters recognize reusable local patterns? |
| 7 | TCN Forecasting: Long Memory Through Dilated Convolutions | How do causality, dilation, and receptive field create convolutional memory? |
| 8 | Transformer Forecasting: Let Every Time Step Look at Every Other Step | How does attention build direct relationships across a sequence? |
| 9 | Patch Transformer Forecasting: Turning Time Steps into Temporal Tokens | When does grouping neighboring observations make attention more effective? |
| 10 | N-BEATS-Style Forecasting: Explaining the Past to Build the Future | How do backcast and forecast residuals refine a prediction? |
| 11 | Which Time-Series Model Should You Use? | How should the architecture match the data, horizon, and operational constraints? |
| 12 | Comparing Sequence Models on Electricity Prediction | Which models perform best when all receive the same raw week? |
| 13 | Do Calendar and Lagged Features Help Sequence Models? | When do extra temporal inputs help, duplicate, or shortcut learned memory? |
| 14 | How to Design a Time-Series Experiment You Can Trust | How do we separate architecture, information, representation, and variance? |

## Release strategy

Publish two articles per week, three or four days apart, over roughly eight weeks. The complete cadence, promotion workflow, live-URL registry, flagship publication candidates, and preflight checklist are in the [publication plan](PUBLICATION_PLAN.md).

Each source draft includes local navigation. Before copying a post to Medium, replace those local links with the public article URLs recorded in the publication plan. Part 1 should become the continuously updated public table of contents.

## Drafts

- [Part 1 — Sequence-prediction foundations](01-sequence-models-for-prediction.md)
- [Part 2 — PyTorch Dataset, DataLoader, and training loop](pytorch-data-pipeline-and-training-loop.md)
- [Part 3 — Linear](02-linear-forecaster.md)
- [Part 4 — MLP](03-mlp-forecaster.md)
- [Part 5 — LSTM](04-lstm-forecaster.md)
- [Part 6 — GRU](05-gru-forecaster.md)
- [Part 7 — CNN1D](06-cnn1d-forecaster.md)
- [Part 8 — TCN](07-tcn-forecaster.md)
- [Part 9 — Transformer](08-transformer-forecaster.md)
- [Part 10 — Patch Transformer](09-patch-transformer-forecaster.md)
- [Part 11 — N-BEATS-style](10-nbeats-style-forecaster.md)
- [Part 12 — Choosing a model](11-choosing-a-sequence-model.md)
- [Part 13 — Electricity model comparison](12-electricity-results-and-interpretation.md)
- [Part 14 — Calendar and lagged features](13-calendar-and-lagged-features.md)
- [Part 15 — Trustworthy experiments](14-designing-a-trustworthy-experiment.md)

## Companion code

Parts 3–11 each contain a complete, standalone PyTorch model definition and an executable tensor-shape check. The reusable source is collected in:

- [practical PyTorch Dataset, DataLoader, and training-loop companion](pytorch-data-pipeline-and-training-loop.md);
- [all nine model implementations](code/sequence_models.py);
- [chronological training and evaluation pipeline](code/training_pipeline.py);
- [causal direct multi-horizon CatBoost benchmark](code/catboost_baseline.py);
- [companion-code usage guide](code/README.md).

Keeping the training pipeline shared prevents each Medium article from repeating the same dataset, early-stopping, and evaluation code while still making every architecture implementation complete.

## Future extension: stock prediction

The planned stock-price study is outlined in [Future stock-prediction roadmap](future-stock-prediction-roadmap.md). It should reuse the same model-first structure while changing the prediction target, baselines, features, and evaluation criteria to suit financial data.

It is intentionally labeled as a roadmap rather than a finished Medium article. No stock results should be published until the data contract, walk-forward evaluation, transaction assumptions, and experiments are complete.

## Figures

| Figure | Suggested use | Alt text |
|---|---|---|
| `three-weeks-electricity-demand.png` | Part 13 | Hourly household electricity demand over three weeks, showing repeating but irregular daily behavior. |
| `electricity-raw-history-model-comparison.png` | Part 13 | Horizontal bars comparing raw-history RMSE for nine models, with three naïve baselines for context. |
| `model-feature-rmse-editorial.png` | Part 14 | Dot plot comparing test RMSE for three feature representations across nine models. |
| `engineered-history-gain-editorial.png` | Part 14 | Diverging bars showing improvement or degradation after adding engineered history. |
| `transformer-rmse-by-horizon.png` | Part 14 | Transformer RMSE from one through 24 hours ahead under each feature representation. |
| `lstm-validation-curves.png` | Part 14 | LSTM validation loss by epoch for each representation. |
| `transformer-example-forecast.png` | Optional Part 14 figure | One actual 24-hour demand path and three Transformer forecasts. |
| `model-feature-rmse.png` | Source figure | Original grouped-bar output from the notebook. |

## Editorial safeguards

- Keep Parts 1–12 understandable without the electricity case study.
- Treat the electricity numbers as **preliminary single-seed results**.
- In Part 13, compare architectures using demand-only inputs before discussing feature engineering.
- In Part 14, compare feature representations primarily within each model family.
- Introduce gradient boosting once as a global reality check in Part 1, treat it fully in Part 12, and mention it in the applied articles only where it changes interpretation. Do not repeat the same warning in every algorithm explainer or imply the current notebook tested it.
- Call the patch model a **simplified patch Transformer**, not a full reproduction of PatchTST.
- Call the residual dense model **N-BEATS-style**, not a universal N-BEATS implementation.
- Do not claim that the engineered representation contains only the same 168 raw observations; `lag_168` extends its raw reach to 336 hours.
- Do not generalize one electricity run into a claim that an architecture always does or does not need engineered features.
- Do not turn the stock roadmap into a results article until the experiment has actually run.

## Source experiment

The electricity case study is derived from the companion feature-ablation notebook used to produce the reported results and figures.
