# Medium series: Sequence Models for Prediction

This folder contains an 18-part series that starts with the general problem of learning from ordered data, builds a leak-free PyTorch pipeline, explains eleven prediction algorithms individually, and then tests models in electricity and financial case studies.

The public GitHub Pages edition is published at [Sequence Models for Prediction](https://adidror005.github.io/sequence-models-for-prediction/).

The editorial angle is **model-first, feature-second**:

1. What makes prediction from sequences different from ordinary tabular prediction?
2. How does each architecture move information from past observations to future outputs?
3. What happens when every architecture receives the same raw history?
4. When do calendar variables, lags, differences, and rolling summaries add value?

Across all four questions, CatBoost or another gradient-boosted tree model remains a mandatory reality check. Sophisticated sequence architectures often lose to boosted trees on carefully constructed tabular time-series features, especially when the amount of independent training data is modest.

Part 18 applies the framework to a saved one-minute META direction experiment and, more importantly, develops the local-versus-global design for multiple symbols. It keeps weak statistical signal separate from any claim about trading profitability. A broader multi-asset, walk-forward stock study remains a future extension.

## Publication structure

### Parts 1–13: foundations and standalone algorithm explainers

The opening article introduces sequence inputs, prediction targets, windowing, forecasting horizons, chronological validation, and recurrent versus convolutional, attention-based, and fully connected approaches. Part 2 then implements the shared Dataset, DataLoaders, and training loop used throughout the series.

Each following article explains one model independently. The posts focus on intuition, information flow, equations, tensor shapes, strengths, limitations, implementation choices, and common failure modes. Readers do not need the electricity experiment to understand them.

Parts 12 and 13 add state-space models and Mamba. The first develops stable latent dynamics and the recurrence/convolution connection. The second makes memory selective and clearly separates an educational PyTorch implementation from the optimized official Mamba package.

### Part 14: choosing an architecture

The synthesis article compares the models as design choices: local versus global access, fixed windows versus recurrent state, effective memory, data requirements, compute cost, and useful baselines.

### Parts 15–16: the electricity case study

Part 15 holds the representation fixed. The original nine tested models receive the same 168 hours of raw demand and predict the same next 24 hours. This isolates the practical architecture comparison. State-space and Mamba rows are not invented; they remain explicit follow-up experiments.

Part 16 then changes the inputs. It tests historical calendar variables and engineered lagged history, while explaining the hidden changes in information reach, parameter count, and optimization.

### Part 17: experimental discipline

This article turns the lessons into a reusable protocol for evaluating any sequence-prediction system without temporal leakage or misleading comparisons.

### Part 18: local versus global financial models

The final article shows how to build per-symbol sequence datasets, pool them safely, and condition a shared neural model on a learned ticker embedding. The saved META run is retained as the single-symbol control, not presented as the article's main subject.

## Publication order

| Part | Working title | Central question |
|---:|---|---|
| 1 | Sequence Models for Prediction: A Practical Map | What is a sequence-prediction problem, and how should it be framed? |
| 2 | A Leak-Free PyTorch Dataset, DataLoader, and Training Loop | How do we create windows, split time, batch examples, train, and evaluate without leakage? |
| 3 | Linear Forecasting: The Baseline That Sees the Whole Window | How far can a direct linear map take us? |
| 4 | MLP Forecasting: A Nonlinear Map from Past Window to Future Path | What does a dense network add to a fixed historical window? |
| 5 | LSTM Forecasting: Learning What to Remember | How do gates create a persistent recurrent memory? |
| 6 | GRU Forecasting: A Leaner Gated Memory | What does the GRU simplify, preserve, and trade away? |
| 7 | 1D CNN Forecasting: Learning Local Shapes in Parallel | How do temporal filters recognize reusable local patterns? |
| 8 | TCN Forecasting: Long Memory Through Dilated Convolutions | How do causality, dilation, and receptive field create convolutional memory? |
| 9 | Transformer Forecasting: Let Every Time Step Look at Every Other Step | How does attention build direct relationships across a sequence? |
| 10 | Patch Transformer Forecasting: Turning Time Steps into Temporal Tokens | When does grouping neighboring observations make attention more effective? |
| 11 | N-BEATS-Style Forecasting: Explaining the Past to Build the Future | How do backcast and forecast residuals refine a prediction? |
| 12 | State-Space Models for Forecasting: A Learnable Dynamical System | How do stable latent dynamics create multiple memory scales? |
| 13 | Mamba for Time-Series Forecasting: Selective State-Space Memory | How can the current input choose what memory retains and exposes? |
| 14 | Which Time-Series Model Should You Use? | How should the architecture match the data, horizon, and operational constraints? |
| 15 | Comparing Sequence Models on Electricity Prediction | Which models perform best when all receive the same raw week? |
| 16 | Do Calendar and Lagged Features Help Sequence Models? | When do extra temporal inputs help, duplicate, or shortcut learned memory? |
| 17 | How to Design a Time-Series Experiment You Can Trust | How do we separate architecture, information, representation, and variance? |
| 18 | One Model Per Stock or One Model for the Market? | When should we train local models versus one pooled, symbol-aware sequence model? |

## Release strategy

Publish two articles per week, three or four days apart, over roughly nine weeks. The complete cadence, promotion workflow, live-URL registry, flagship publication candidates, and preflight checklist are in the [publication plan](PUBLICATION_PLAN.md).

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
- [Part 12 — State-space models](state-space-models-for-forecasting.md)
- [Part 13 — Mamba](mamba-for-time-series.md)
- [Part 14 — Choosing a model](11-choosing-a-sequence-model.md)
- [Part 15 — Electricity model comparison](12-electricity-results-and-interpretation.md)
- [Part 16 — Calendar and lagged features](13-calendar-and-lagged-features.md)
- [Part 17 — Trustworthy experiments](14-designing-a-trustworthy-experiment.md)
- [Part 18 — Local versus global stock models](15-finance-direction-case-study.md)

## Editing and publishing an article

Edit the article's Markdown source from the list above. For example, edit [`mamba-for-time-series.md`](mamba-for-time-series.md) to change the Mamba page. **Do not edit files inside `site/` or `medium_exports/` directly**; those are generated files and will be overwritten the next time the builders run.

From the repository root, rebuild both versions:

```bash
python tools/build_github_site.py
python tools/build_medium_exports.py
```

The first command rebuilds the public GitHub Pages files in `site/`. The second rebuilds the copy-friendly Medium package in `medium_exports/`. Before publishing, open the corresponding HTML file in `site/` to check the article, code blocks, images, and navigation.

When the preview looks right, commit and publish the changes:

```bash
git add -A
git commit -m "Update article"
git push origin main
```

GitHub Pages normally updates within a few minutes. If an article is added, removed, or reordered rather than merely edited, also update `SERIES` in [`tools/build_github_site.py`](tools/build_github_site.py) and `SOURCE_ORDER` in [`tools/build_medium_exports.py`](tools/build_medium_exports.py) before rebuilding.

## Companion code

Parts 3–13 each contain a complete, standalone PyTorch model definition and an executable tensor-shape check. The reusable source is collected in:

- [practical PyTorch Dataset, DataLoader, and training-loop companion](pytorch-data-pipeline-and-training-loop.md);
- [all eleven model implementations](code/sequence_models.py);
- [chronological training and evaluation pipeline](code/training_pipeline.py);
- [causal direct multi-horizon CatBoost benchmark](code/catboost_baseline.py);
- [companion-code usage guide](code/README.md).
- [local-versus-global stock-model notebook](notebooks/local_vs_global_stock_models.ipynb) and its [data/setup guide](notebooks/README.md).

Keeping the training pipeline shared prevents each Medium article from repeating the same dataset, early-stopping, and evaluation code while still making every architecture implementation complete.

## Finance case study and future extension

The completed first finance application is [Part 18 — One Model Per Stock or One Model for the Market?](15-finance-direction-case-study.md). It explains per-symbol datasets, symbol-safe pooling, symbol IDs, learned embeddings, and the experiment needed to compare local, symbol-blind pooled, and symbol-aware pooled models. A saved META run supplies the local control.

The broader [Future stock-prediction roadmap](future-stock-prediction-roadmap.md) remains intentionally separate. It specifies the multi-asset, walk-forward, cost-aware work required before making a financial-usefulness claim.

## Figures

| Figure | Suggested use | Alt text |
|---|---|---|
| `state-space-recurrence.png` | Part 12 | Input and previous latent state flowing through one stable state-space recurrence. |
| `mamba-selective-state-space.png` | Part 13 | Fixed state-space memory compared with Mamba-style content-dependent updates. |
| `three-weeks-electricity-demand.png` | Part 15 | Hourly household electricity demand over three weeks, showing repeating but irregular daily behavior. |
| `electricity-raw-history-model-comparison.png` | Part 15 | Horizontal bars comparing raw-history RMSE for nine models, with three naïve baselines for context. |
| `model-feature-rmse-editorial.png` | Part 16 | Dot plot comparing test RMSE for three feature representations across nine models. |
| `engineered-history-gain-editorial.png` | Part 16 | Diverging bars showing improvement or degradation after adding engineered history. |
| `transformer-rmse-by-horizon.png` | Part 16 | Transformer RMSE from one through 24 hours ahead under each feature representation. |
| `lstm-validation-curves.png` | Part 16 | LSTM validation loss by epoch for each representation. |
| `transformer-example-forecast.png` | Optional Part 16 figure | One actual 24-hour demand path and three Transformer forecasts. |
| `model-feature-rmse.png` | Source figure | Original grouped-bar output from the notebook. |
| `finance-model-test-auc.png` | Part 18 | Test ROC AUC for CatBoost and six neural models on META next-minute direction. |
| `finance-mlp-price-memory-ablation.png` | Part 18 | Validation and test AUC for the MLP with base versus added price-memory features. |
| `finance-catboost-feature-importance.png` | Part 18 | CatBoost's ten highest endpoint-feature importances. |
| `finance-local-vs-global-models.png` | Part 18 | Local per-stock models compared with a shared model trained from per-symbol datasets and a learned ticker embedding. |

## Editorial safeguards

- Keep Parts 1–14 understandable without the electricity case study.
- Treat the electricity numbers as **preliminary single-seed results**.
- In Part 15, compare architectures using demand-only inputs before discussing feature engineering.
- In Part 16, compare feature representations primarily within each model family.
- Introduce gradient boosting once as a global reality check in Part 1, treat it fully in Part 14, and mention it in the applied articles only where it changes interpretation. Do not repeat the same warning in every algorithm explainer or imply the current notebook tested it.
- Call the portable selective implementation **Mamba-style**, and use the official package for claims about Mamba speed or benchmark quality.
- Do not add unmeasured state-space or Mamba scores to the original nine-model electricity tables.
- Call the patch model a **simplified patch Transformer**, not a full reproduction of PatchTST.
- Call the residual dense model **N-BEATS-style**, not a universal N-BEATS implementation.
- Do not claim that the engineered representation contains only the same 168 raw observations; `lag_168` extends its raw reach to 336 hours.
- Do not generalize one electricity run into a claim that an architecture always does or does not need engineered features.
- Describe the saved META output as the single-symbol control; the notebook contains multi-symbol scaffolding, but no pooled multi-symbol result has yet been recorded.
- Do not interpret ROC AUC or balanced accuracy as evidence of trading profitability.
- Do not report the notebook's CatBoost-plus-technical-analysis variant because its saved cells contain no executed result.
- Keep the broader stock roadmap separate until its multi-asset, walk-forward, cost-aware experiment has actually run.

## Source experiment

The electricity case study is derived from the companion feature-ablation notebook used to produce its reported results and figures. The finance case study is derived from the saved `deep_learning_comparison` notebook output supplied for this project.
