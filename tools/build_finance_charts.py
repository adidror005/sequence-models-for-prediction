"""Build the editorial charts for the META direction-prediction case study."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

INK = "#14251f"
MUTED = "#5f6f68"
GREEN = "#256c56"
RUST = "#d96c42"
CREAM = "#f0ecdf"
PAPER = "#fbfaf5"


def finish(figure: plt.Figure, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        ASSETS / name,
        dpi=180,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def model_auc_chart() -> None:
    scores = {
        "CatBoost": 0.531812,
        "GRU": 0.530355,
        "Transformer": 0.530250,
        "TCN": 0.529784,
        "LSTM": 0.528563,
        "MLP": 0.513493,
        "CNN1D": 0.507748,
    }
    labels = list(scores)
    values = np.array(list(scores.values()))
    y = np.arange(len(labels))

    figure, axis = plt.subplots(figsize=(10.8, 6.4), facecolor=PAPER)
    axis.set_facecolor(PAPER)
    colors = [RUST if label == "CatBoost" else GREEN for label in labels]
    axis.barh(y, values - 0.5, left=0.5, color=colors, height=0.62)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0.5, 0.535)
    axis.set_xticks([0.50, 0.51, 0.52, 0.53])
    axis.axvline(0.5, color=INK, linewidth=1.2)
    axis.grid(axis="x", color="#d9ded7", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_xlabel("Test ROC AUC (0.50 = chance)", color=MUTED, labelpad=12)
    axis.set_title(
        "Weak signal, narrow separation",
        loc="left",
        color=INK,
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
    axis.text(
        0,
        1.02,
        "META next-minute direction · 78 bars × 31 features · single saved run",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=11,
    )
    for index, value in enumerate(values):
        axis.text(value + 0.00045, index, f"{value:.4f}", va="center", color=INK, fontsize=10)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#c9d0ca")
    axis.tick_params(colors=MUTED, length=0)
    figure.text(
        0.125,
        0.015,
        "Bars show only the amount above chance; the exact AUC is printed beside each model.",
        color=MUTED,
        fontsize=9,
    )
    finish(figure, "finance-model-test-auc.png")


def ablation_chart() -> None:
    labels = ["Validation", "Test"]
    base = np.array([0.519222, 0.519120])
    enhanced = np.array([0.514169, 0.513493])
    x = np.arange(len(labels))
    width = 0.32

    figure, axis = plt.subplots(figsize=(9.8, 6.2), facecolor=PAPER)
    axis.set_facecolor(PAPER)
    axis.bar(x - width / 2, base - 0.5, bottom=0.5, width=width, color=GREEN, label="25 base features")
    axis.bar(
        x + width / 2,
        enhanced - 0.5,
        bottom=0.5,
        width=width,
        color=RUST,
        label="Base + 6 price-memory features",
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0.5, 0.5225)
    axis.set_ylabel("ROC AUC (0.50 = chance)", color=MUTED, labelpad=10)
    axis.grid(axis="y", color="#d9ded7", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_title(
        "More price memory did not help this MLP",
        loc="left",
        color=INK,
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
    axis.text(
        0,
        1.02,
        "Same eligible rows, labels, windows, and training rules",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=11,
    )
    for positions, values in ((x - width / 2, base), (x + width / 2, enhanced)):
        for position, value in zip(positions, values, strict=True):
            axis.text(position, value + 0.0004, f"{value:.4f}", ha="center", color=INK, fontsize=10)
    axis.legend(frameon=False, loc="upper right")
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#c9d0ca")
    axis.tick_params(colors=MUTED, length=0)
    figure.text(
        0.125,
        0.015,
        "Test AUC fell by 0.0056. This is one model and one run—not a universal verdict on price memory.",
        color=MUTED,
        fontsize=9,
    )
    finish(figure, "finance-mlp-price-memory-ablation.png")


def importance_chart() -> None:
    scores = {
        "close location": 10.035295,
        "close vs. average": 4.835745,
        "2-minute return": 4.278921,
        "gap from prior close": 3.660246,
        "log volume": 3.604862,
        "time-of-day cosine": 3.590283,
        "close vs. 20-day MA": 3.578893,
        "bar range": 3.493063,
        "close vs. 30-minute MA": 3.427459,
        "5-minute relative volume": 3.345426,
    }
    labels = list(scores)[::-1]
    values = np.array(list(scores.values()))[::-1]
    y = np.arange(len(labels))

    figure, axis = plt.subplots(figsize=(10.8, 7.0), facecolor=PAPER)
    axis.set_facecolor(PAPER)
    axis.barh(y, values, color=GREEN, height=0.62)
    axis.set_yticks(y, labels)
    axis.set_xlabel("CatBoost feature importance", color=MUTED, labelpad=12)
    axis.grid(axis="x", color="#d9ded7", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.set_title(
        "Short-horizon geometry dominated the ranking",
        loc="left",
        color=INK,
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
    axis.text(
        0,
        1.02,
        "Top 10 of 31 endpoint features in the fitted CatBoost model",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=11,
    )
    for index, value in enumerate(values):
        axis.text(value + 0.12, index, f"{value:.2f}", va="center", color=INK, fontsize=10)
    axis.set_xlim(0, 11.4)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#c9d0ca")
    axis.tick_params(colors=MUTED, length=0)
    figure.text(
        0.125,
        0.015,
        "Importance is descriptive, not causal; correlated features can divide or redirect credit.",
        color=MUTED,
        fontsize=9,
    )
    finish(figure, "finance-catboost-feature-importance.png")


def main() -> None:
    model_auc_chart()
    ablation_chart()
    importance_chart()
    print("Built 3 finance case-study charts")


if __name__ == "__main__":
    main()
