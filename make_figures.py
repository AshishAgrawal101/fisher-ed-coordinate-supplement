#!/usr/bin/env python3
"""Create the v13 manuscript figures from verified CSV outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent


def read_csv(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def paired_torus_figure() -> None:
    rows = read_csv("paired_torus_sensitivity.csv")
    m = np.array([float(row["integer_shear_m"]) for row in rows])
    ratio = np.array([float(row["effective_dimension_over_d"]) for row in rows])
    fig, axis = plt.subplots(figsize=(6.4, 3.5))
    axis.plot(m, ratio, marker="o", linewidth=2.0, color="#1167b1")
    axis.set_xlabel("integer shear magnitude $m$")
    axis.set_ylabel("normalized effective dimension $D_\\kappa/d$")
    axis.set_xticks(m)
    axis.set_xlim(-0.2, 10.2)
    axis.set_ylim(0.5, 1.03)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(HERE / "paired_torus_sensitivity.png", dpi=240)
    plt.close(fig)


def _band(axis, rows, model: str, color: str, marker: str) -> None:
    model_rows = [row for row in rows if row["model"] == model]
    lows = sorted(
        (row for row in model_rows if row["endpoint"] == "low"),
        key=lambda row: int(row["shear_m_or_matched_budget"]),
    )
    highs = sorted(
        (row for row in model_rows if row["endpoint"] == "high"),
        key=lambda row: int(row["shear_m_or_matched_budget"]),
    )
    x = np.array([float(row["condition_bound"]) for row in lows])
    low = np.array([float(row["estimate"]) for row in lows])
    high = np.array([float(row["estimate"]) for row in highs])
    axis.fill_between(x, low, high, color=color, alpha=0.08)
    axis.plot(x, low, color=color, marker=marker, linewidth=1.8)
    axis.plot(x, high, color=color, marker=marker, linewidth=1.2, linestyle="--")


def coordinate_band_figure() -> None:
    paired = read_csv("abbas_matched_paired_shear_bands.csv")
    rescaling = read_csv("abbas_classical_rescaling_bands.csv")
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.45), width_ratios=(1.55, 1.0))
    styles = {
        "QNN": ("#238b45", "o", "QNN"),
        "easy QNN": ("#2171b5", "s", "easy QNN"),
        "published classical archive": ("#7a5195", "D", "published classical ($d=28$)"),
        "rebuilt classical 2000x2000": ("#cb181d", "^", "rebuilt classical ($d=40$)"),
    }
    for model, (color, marker, _) in styles.items():
        _band(left, paired, model, color, marker)
    _band(right, rescaling, "rebuilt classical 2000x2000", "#cb181d", "^")

    ticks = np.array([1.0, 6.854, 33.971, 118.992, 321.997, 726.999])
    tick_labels = ["1", "6.9", "34", "119", "322", "727"]
    for axis in (left, right):
        axis.set_xscale("log")
        axis.set_xticks(ticks, tick_labels)
        axis.set_xlabel("distortion budget $\\mathrm{cond}(A^\\top A)$")
        axis.grid(alpha=0.22)
    left.set_ylabel("normalized effective dimension")
    left.set_ylim(0.36, 0.93)
    right.set_ylim(0.27, 0.71)
    right.tick_params(axis="x", labelsize=7, rotation=28)
    left.set_title("(a) Same paired-shear family", fontsize=9)
    right.set_title("(b) Classical rescaling orbit", fontsize=9)

    model_handles = [
        Line2D([0], [0], color=color, marker=marker, linewidth=1.6, label=label)
        for color, marker, label in styles.values()
    ]
    endpoint_handles = [
        Line2D([0], [0], color="0.25", linewidth=1.8, label="lower endpoint"),
        Line2D([0], [0], color="0.25", linewidth=1.2, linestyle="--", label="upper endpoint"),
    ]
    fig.legend(
        handles=model_handles + endpoint_handles,
        frameon=False,
        fontsize=7.1,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(HERE / "abbas_coordinate_bands.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    paired_torus_figure()
    coordinate_band_figure()
    print("figure generation: PASS")
