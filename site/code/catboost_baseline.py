"""Causal direct multi-horizon CatBoost benchmark.

This benchmark is proposed for the follow-up experiment. It was not executed in
the supplied electricity notebook and therefore has no result in the articles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def make_causal_feature_frame(
    target: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create features available immediately before each forecast origin."""
    series = pd.Series(np.asarray(target, dtype=np.float32), index=timestamps)
    previous = series.shift(1)
    hour = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()

    features = pd.DataFrame(index=timestamps)
    features["lag_1"] = previous
    features["lag_24"] = series.shift(24)
    features["lag_168"] = series.shift(168)
    features["change_1"] = previous - series.shift(2)
    features["change_24"] = previous - series.shift(25)
    features["mean_24"] = previous.rolling(24, min_periods=24).mean()
    features["std_24"] = previous.rolling(24, min_periods=24).std(ddof=0)
    features["mean_168"] = previous.rolling(168, min_periods=168).mean()
    features["std_168"] = previous.rolling(168, min_periods=168).std(ddof=0)
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    features["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    features["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    features["weekend"] = (day_of_week >= 5).astype(np.float32)
    return features.astype(np.float32)


def rows_at_origins(
    feature_frame: pd.DataFrame,
    target: np.ndarray,
    origins: np.ndarray,
    horizon: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    origins = np.asarray(origins, dtype=np.int64)
    x = feature_frame.iloc[origins].copy()
    y = np.column_stack(
        [np.asarray(target)[origins + step] for step in range(horizon)]
    ).astype(np.float32)

    valid = np.isfinite(x.to_numpy()).all(axis=1) & np.isfinite(y).all(axis=1)
    return x.loc[valid], y[valid]


class DirectCatBoostForecaster:
    """Fit one explicitly validated CatBoost regressor per forecast horizon."""

    def __init__(
        self,
        horizon: int,
        iterations: int = 2000,
        learning_rate: float = 0.03,
        depth: int = 8,
        random_seed: int = 42,
    ) -> None:
        self.horizon = horizon
        self.parameters = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "l2_leaf_reg": 5.0,
            "random_seed": random_seed,
            "verbose": False,
            "allow_writing_files": False,
        }
        self.models: list[CatBoostRegressor] = []

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: np.ndarray,
        x_validation: pd.DataFrame,
        y_validation: np.ndarray,
    ) -> "DirectCatBoostForecaster":
        if y_train.ndim != 2 or y_train.shape[1] != self.horizon:
            raise ValueError("y_train must have shape [samples, horizon]")
        if y_validation.ndim != 2 or y_validation.shape[1] != self.horizon:
            raise ValueError("y_validation must have shape [samples, horizon]")

        self.models = []
        for step in range(self.horizon):
            model = CatBoostRegressor(**self.parameters)
            model.fit(
                x_train,
                y_train[:, step],
                eval_set=(x_validation, y_validation[:, step]),
                use_best_model=True,
                early_stopping_rounds=100,
            )
            self.models.append(model)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if len(self.models) != self.horizon:
            raise RuntimeError("fit must be called before predict")
        return np.column_stack([model.predict(x) for model in self.models])


def smoke_test() -> None:
    rows, horizon = 500, 3
    timestamps = pd.date_range("2020-01-01", periods=rows, freq="h")
    time = np.arange(rows, dtype=np.float32)
    target = np.sin(2 * np.pi * time / 24.0).astype(np.float32)
    features = make_causal_feature_frame(target, timestamps)

    train_origins = np.arange(168, 350 - horizon + 1)
    validation_origins = np.arange(350, 425 - horizon + 1)
    test_origins = np.arange(425, rows - horizon + 1)
    x_train, y_train = rows_at_origins(
        features, target, train_origins, horizon
    )
    x_validation, y_validation = rows_at_origins(
        features, target, validation_origins, horizon
    )
    x_test, y_test = rows_at_origins(features, target, test_origins, horizon)

    model = DirectCatBoostForecaster(
        horizon=horizon,
        iterations=5,
        learning_rate=0.10,
        depth=3,
    ).fit(x_train, y_train, x_validation, y_validation)
    predictions = model.predict(x_test)
    assert predictions.shape == y_test.shape
    print("CatBoost baseline:", predictions.shape)


if __name__ == "__main__":
    smoke_test()
