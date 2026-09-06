# Future Expansion: Sequence Models for Stock Prediction

*A pre-registered comparison designed to separate price persistence from genuine forecasting signal*

**Status:** Future experiment roadmap; not part of the current publication sequence
**Suggested Medium tags:** Stock Market, Time Series, Forecasting, Deep Learning, Quantitative Finance

Electricity demand is rhythmic. Stock prices are adversarial.

Household power consumption repeats daily and weekly behaviors that remain visible in a short chart. Markets adapt, regimes shift, and easy patterns attract capital until they weaken. A model that performs well on electricity therefore deserves a second test where predictability is smaller and evaluation traps are larger.

This article defines that test before looking at results.

The goal is not to ask which network predicts the highest stock price. It is to ask:

> Which architectural advantages survive after we remove the easy persistence of price level and evaluate across assets, time periods, and realistic baselines?

## Why raw price RMSE is dangerous

Suppose a stock closes at 100 today and 100.20 tomorrow. A model that predicts 100 has a small price error without forecasting the positive return.

Because adjacent prices are usually close, the rule:

```text
tomorrow's price = today's price
```

can look excellent under level RMSE.

That is the random-walk baseline. Any stock-price study should report it prominently.

We will evaluate two related tasks.

### Task A: normalized price path

Predict the next five trading-day prices relative to the final observed price:

```text
relative_price[h] = future_price[h] / last_context_price - 1
```

This removes nominal scale while retaining the path users recognize as a price forecast.

### Task B: future log returns

Predict the next five daily log returns:

```text
return[t] = log(adjusted_close[t] / adjusted_close[t-1])
```

This directly tests incremental changes. A zero vector is the natural baseline.

Publishing both prevents a low level error from masquerading as directional signal.

## Proposed dataset

The recommended first pass uses daily, corporate-action-adjusted prices for:

```text
SPY, QQQ, AAPL, MSFT, AMZN, META, JPM, XOM
```

This is a deliberately selected educational basket, not a survivorship-bias-free equity universe. Results must be described as case studies across liquid assets, not evidence for the whole market.

For a stronger follow-up, use historical index constituents and preserve point-in-time membership.

Each model should be trained and evaluated per symbol first. Report the macro-average across symbols so one volatile stock does not dominate the table. A later global-model experiment can share parameters across assets with an explicit symbol identifier.

## Forecasting contract

The stock version uses:

```text
lookback:         168 trading days
forecast horizon: 5 trading days
frequency:        daily
targets:          normalized price path and log return
```

Keeping a 168-step lookback preserves the architectural comparison from electricity while changing its real-world duration from one week to roughly eight trading months.

That distinction should be explicit. The sequence length is controlled; elapsed calendar time is not.

## Feature sets

The feature ablation mirrors the electricity experiment without copying its features mechanically.

### 1. Market history only

```text
log_return
```

For the price-path task, the historical normalized price path can be included as a separate raw channel. Do not feed the absolute dollar price if models are compared across symbols.

### 2. Market history plus known calendar

- day-of-week sine and cosine;
- month-of-year sine and cosine;
- month-end indicator.

Calendar effects in markets are often weak and unstable. That makes them an ablation, not an assumption.

### 3. Market history plus engineered history

- lagged returns at 1, 5, and 21 trading days;
- cumulative returns over 5, 21, and 63 days;
- rolling realized volatility over 5, 21, and 63 days;
- volume change and standardized volume, if point-in-time volume is available;
- price distance from a 21- and 63-day moving average.

All rolling and normalization operations must use information available before the forecast origin. Every scaler is fitted inside the training portion of each fold.

## Keep the raw information budget equal

Part 14 showed how a lag channel can secretly extend historical reach. The stock experiment will avoid that ambiguity.

Every derived feature must be computed from the same permitted 168 raw trading days. If a 63-day statistic is attached to every row, early rows would require data outside the window. We will use one of two clean designs:

1. provide 231 raw days to every representation so the earliest 63-day feature has an equal source history; or
2. calculate fixed-budget summaries at the forecast origin and pass them through a separate context vector.

The first design is simpler for the existing model zoo. Whichever design is selected must be shared across all architectures.

## Time splits

Random train/test splitting is not acceptable.

Use expanding rolling origins with date-based folds. For example:

```text
fold 1: train ───────── validate ─ test
fold 2: train ───────────────── validate ─ test
fold 3: train ───────────────────────── validate ─ test
```

Exact dates will be fixed after choosing the data source and available start date. They should include calm, crisis, recovery, inflationary, and rate-transition periods where possible.

Overlapping five-day labels require care. Forecast origins near a boundary must not place target days on both sides of that boundary. Hyperparameters are chosen without inspecting the final test fold.

## Baselines

The stock comparison is not credible without:

| Target | Required baseline |
|---|---|
| Price path | Last adjusted close repeated for five days |
| Returns | Zero return |
| Returns | Training-period mean return |
| Either | Regularized direct linear model |
| Either | CatBoost on causal lag, rolling, calendar, and market-context features |
| Volatility analysis | Recent realized volatility carried forward |

A neural network that beats “yesterday” on electricity but loses to zero return—or to a disciplined boosted-tree model—on stocks has taught us something important about domain structure.

## Models

Use the same nine families:

- linear;
- MLP;
- LSTM;
- GRU;
- CNN1D;
- short- and full-receptive-field TCN;
- Transformer;
- simplified patch Transformer;
- N-BEATS-style.

The TCN receives two variants because the electricity study revealed that receptive field changes the meaning of engineered features. The patch model should test five-day-aligned and overlapping patches.

## Metrics

No single metric is enough.

### Statistical metrics

- MAE and RMSE of normalized price path;
- MAE and RMSE of returns;
- directional accuracy with confidence intervals;
- correlation between predicted and realized returns;
- performance by forecast horizon;
- mean and standard deviation across seeds, folds, and symbols.

### Economic diagnostics

- turnover;
- gross and net return under a predeclared decision rule;
- transaction-cost sensitivity;
- maximum drawdown;
- exposure and volatility;
- performance by market regime.

The trading rule must be specified before viewing test results. Otherwise the strategy layer becomes another tuning surface.

## Results table to populate

Do not publish this section until the experiment has run.

| Model | Best feature set | Price-path RMSE vs. random walk | Return MAE vs. zero | Directional accuracy | Net result after costs |
|---|---|---:|---:|---:|---:|
| Linear | Pending | Pending | Pending | Pending | Pending |
| MLP | Pending | Pending | Pending | Pending | Pending |
| LSTM | Pending | Pending | Pending | Pending | Pending |
| GRU | Pending | Pending | Pending | Pending | Pending |
| CNN1D | Pending | Pending | Pending | Pending | Pending |
| TCN | Pending | Pending | Pending | Pending | Pending |
| Transformer | Pending | Pending | Pending | Pending | Pending |
| Patch Transformer | Pending | Pending | Pending | Pending | Pending |
| N-BEATS-style | Pending | Pending | Pending | Pending | Pending |

## The comparison we want to make

Once results exist, the electricity and stock articles should share one summary table:

| Question | Electricity | Stocks |
|---|---|---|
| Dominant predictable structure | Daily and weekly seasonality | To be measured; likely weak and regime-dependent |
| Strong naive baseline | Repeat yesterday | Random walk / zero return |
| Value of calendar | Helped 7 of 9 model families in one run | Pending |
| Value of engineered history | −2.82% to +2.05% by architecture | Pending |
| Best architecture | 31-hour TCN with engineered features, preliminary | Pending |
| Main evaluation risk | Unequal history and receptive field | Persistence, selection bias, and costs |

The most interesting outcome may be that rankings reverse. A model whose inductive bias matches stable electricity cycles may overfit market noise, while a disciplined linear or recurrent baseline may hold up better.

That is why the stock article is specified before the data are scored. We are not searching for a market story. We are testing whether the architectural lessons transfer.

This becomes a later article or follow-up series after the market-data experiment has been implemented and run.
