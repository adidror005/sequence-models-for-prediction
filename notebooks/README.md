# Local versus global stock-model notebook

[`local_vs_global_stock_models.ipynb`](local_vs_global_stock_models.ipynb) is the public, code-complete companion to [One Model Per Stock or One Model for the Market?](https://adidror005.github.io/sequence-models-for-prediction/15-finance-direction-case-study.html).

It includes the saved outputs reported in the article:

- causal intraday and price-memory feature engineering;
- whole-session chronological train, validation, and test splits;
- a training-only return-magnitude cutoff;
- a lazy PyTorch `Dataset` and `DataLoader` for 78-bar windows;
- MLP, LSTM, GRU, CNN1D, TCN, and Transformer classifiers;
- optimizer and tiny-subset memorization diagnostics;
- an MLP base-versus-price-memory feature ablation;
- a CatBoost endpoint-feature baseline.

The notebook supports both a single-symbol control and pooled multi-symbol training. Its saved outputs contain only the META control, so no multi-symbol performance result is claimed. It omits the original notebook's failed inference example and unexecuted technical-analysis extension.

## Data expected

Market data is not included or redistributed. Supply regular-trading-hours one-minute Parquet bars that you are licensed to use in one directory per ticker:

```text
data/<SYMBOL>/1_min_TRADES_RTH/*.parquet

# examples
data/META/1_min_TRADES_RTH/*.parquet
data/MSFT/1_min_TRADES_RTH/*.parquet
data/NVDA/1_min_TRADES_RTH/*.parquet
```

Each row must contain:

| Column | Meaning |
|---|---|
| `date` | Unique bar timestamp |
| `open` | One-minute open |
| `high` | One-minute high |
| `low` | One-minute low |
| `close` | One-minute close |
| `volume` | One-minute volume |
| `average` | Optional average price; required to reproduce the saved 31-feature run exactly |

Timestamps are interpreted in `America/New_York`. Bars should already be restricted to the regular session, sorted by time, and adjusted consistently with your data vendor's methodology.

## Run it

Install the companion dependencies:

```bash
python -m pip install -r notebooks/requirements-finance.txt
```

Then open the notebook locally, or use the [Google Colab version](https://colab.research.google.com/github/adidror005/sequence-models-for-prediction/blob/main/notebooks/local_vs_global_stock_models.ipynb) and upload or mount the expected data directory.

Choose the condition in the configuration cell:

```python
# Saved single-symbol control
EXPERIMENT_MODE = "single"
USE_SYMBOL_EMBEDDING = True

# Pooled model without ticker identity
EXPERIMENT_MODE = "pooled"
USE_SYMBOL_EMBEDDING = False

# Pooled, symbol-aware model
EXPERIMENT_MODE = "pooled"
USE_SYMBOL_EMBEDDING = True
```

Set `POOLED_SYMBOLS` to the tickers available under `data/`. For the final comparison, also train each ticker in `single` mode and keep the same chronological boundaries, model settings, and seed set across conditions.

The complete experiment is computationally substantial: the saved run processed more than 1.3 million one-minute rows and trained six neural models plus CatBoost. Start with a shorter chronological period when checking the pipeline.

## Interpretation boundary

The saved AUC and balanced-accuracy values are single-seed classification results. The notebook does not model transaction costs, spread, slippage, latency, turnover, position sizing, or drawdown. It should not be interpreted as a profitable trading strategy.
