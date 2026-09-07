"""Build explanatory diagrams for the state-space and Mamba articles."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

INK = "#14251f"
MUTED = "#5f6f68"
GREEN = "#256c56"
RUST = "#d96c42"
LIME = "#ddec72"
CREAM = "#f0ecdf"
PAPER = "#fbfaf5"


def box(axis, x, y, width, height, label, color, fontsize=12):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.028",
        linewidth=1.3,
        edgecolor=INK,
        facecolor=color,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
        fontweight="bold",
    )


def arrow(axis, start, end, color=MUTED, curve=0.0):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color=color,
            connectionstyle=f"arc3,rad={curve}",
        )
    )


def finish(figure, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        ASSETS / name,
        dpi=180,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def state_space_diagram() -> None:
    figure, axis = plt.subplots(figsize=(12.5, 6.3), facecolor=PAPER)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.04,
        0.93,
        "A state-space model carries a compact latent state through time",
        color=INK,
        fontsize=21,
        fontweight="bold",
    )
    axis.text(
        0.04,
        0.865,
        "The same transition is reused at every time step.",
        color=MUTED,
        fontsize=12,
    )

    box(axis, 0.06, 0.36, 0.17, 0.17, "input\n$u_t$", CREAM)
    box(axis, 0.39, 0.32, 0.25, 0.25, "latent state\n$h_t = \\bar A h_{t-1} + \\bar B u_t$", LIME)
    box(axis, 0.79, 0.36, 0.15, 0.17, "output\n$y_t$", "#f4c7b7")

    arrow(axis, (0.23, 0.445), (0.39, 0.445))
    arrow(axis, (0.64, 0.445), (0.79, 0.445))
    arrow(axis, (0.515, 0.70), (0.515, 0.57), color=GREEN)
    axis.text(0.515, 0.755, "previous state  $h_{t-1}$", ha="center", color=GREEN, fontsize=12, fontweight="bold")

    axis.text(0.515, 0.20, "$y_t = C h_t + D u_t$", ha="center", color=INK, fontsize=15)
    axis.text(
        0.04,
        0.07,
        "Stable decay modes determine how quickly old information fades; B writes to memory and C reads from it.",
        color=MUTED,
        fontsize=11,
    )
    finish(figure, "state-space-recurrence.png")


def mamba_diagram() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 6.5), facecolor=PAPER)
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")

    fixed, selective = axes
    fixed.set_title("FIXED STATE SPACE", loc="left", color=INK, fontsize=17, fontweight="bold", pad=16)
    fixed.text(0, 0.94, "One transition rule for every input", color=MUTED, fontsize=11)
    box(fixed, 0.03, 0.58, 0.22, 0.13, "quiet hour", CREAM, fontsize=10)
    box(fixed, 0.03, 0.26, 0.22, 0.13, "demand spike", CREAM, fontsize=10)
    box(fixed, 0.42, 0.41, 0.25, 0.18, "same\n$\\bar A, \\bar B, C$", LIME, fontsize=12)
    arrow(fixed, (0.25, 0.645), (0.42, 0.52))
    arrow(fixed, (0.25, 0.325), (0.42, 0.48))
    arrow(fixed, (0.67, 0.50), (0.91, 0.50))
    fixed.text(0.04, 0.09, "Memory does not change its policy\naccording to the current observation.", color=MUTED, fontsize=11)

    selective.set_title("MAMBA-STYLE SELECTIVE STATE SPACE", loc="left", color=INK, fontsize=17, fontweight="bold", pad=16)
    selective.text(0, 0.94, "The input helps choose what to retain", color=MUTED, fontsize=11)
    box(selective, 0.03, 0.58, 0.22, 0.13, "quiet hour", CREAM, fontsize=10)
    box(selective, 0.03, 0.26, 0.22, 0.13, "demand spike", "#f4c7b7", fontsize=10)
    box(selective, 0.40, 0.55, 0.31, 0.17, "small update\n$\\Delta_t, B_t, C_t$", "#e9efc1", fontsize=11)
    box(selective, 0.40, 0.23, 0.31, 0.17, "different update\n$\\Delta_t, B_t, C_t$", LIME, fontsize=11)
    arrow(selective, (0.25, 0.645), (0.40, 0.635))
    arrow(selective, (0.25, 0.325), (0.40, 0.315), color=RUST)
    arrow(selective, (0.71, 0.635), (0.94, 0.53))
    arrow(selective, (0.71, 0.315), (0.94, 0.47), color=RUST)
    selective.text(0.04, 0.09, "Selection makes the recurrence\ncontent-dependent.", color=MUTED, fontsize=11)

    figure.suptitle(
        "Mamba turns a fixed memory system into a selective one",
        x=0.06,
        y=1.02,
        ha="left",
        color=INK,
        fontsize=23,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.055, right=0.98, top=0.82, bottom=0.08, wspace=0.14)
    finish(figure, "mamba-selective-state-space.png")


if __name__ == "__main__":
    state_space_diagram()
    mamba_diagram()
