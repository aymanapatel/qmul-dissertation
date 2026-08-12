#!/usr/bin/env python3
"""Generate focused learning-v2 final-detection overview figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_STUDY = (
    REPOSITORY_ROOT
    / "3_Learning"
    / "learning_v2"
    / "artifacts_evidence-v3.0"
    / "phase_6_7_final"
    / "phase_7_detection_study.json"
)
DEFAULT_OUTPUT_DIR = HERE / "images"

BLUE = "#4E79A7"
GREEN = "#2E8B57"
ORANGE = "#D04A02"
RED = "#C51B1D"
PURPLE = "#8073AC"
GRID = "#D9E0E8"

METHODS = (
    ("mlp_specialist", "MLP"),
    ("graphsage_specialist", "GraphSAGE"),
    ("gat_specialist", "GAT"),
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_panel(ax: plt.Axes, *, percent: bool = False) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAB4C0")
    ax.spines["bottom"].set_color("#AAB4C0")
    ax.tick_params(labelsize=8)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))


def add_value_labels(
    ax: plt.Axes,
    bars,
    *,
    percent: bool = False,
    fontsize: float = 8,
    rotation: float = 0,
) -> None:
    labels = [
        f"{bar.get_height():.1%}" if percent else f"{bar.get_height():,.0f}"
        for bar in bars
    ]
    ax.bar_label(bars, labels=labels, padding=2, fontsize=fontsize, rotation=rotation)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_matching_outcomes(study: dict, output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    x = np.arange(len(METHODS))
    width = 0.23
    series = (
        ("True positives", "tp", GREEN),
        ("False positives", "fp", ORANGE),
        ("False negatives", "fn", RED),
    )
    for index, (label, field, color) in enumerate(series):
        values = [study["methods"][key][field] for key, _ in METHODS]
        bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
        add_value_labels(ax, bars, fontsize=5.5, rotation=90)
    ax.set_xticks(x, [name for _, name in METHODS])
    ax.set_ylabel("Site–criterion pairs", fontsize=7)
    ax.set_title("Final site–criterion matching outcomes", fontsize=8, pad=18)
    ax.set_ylim(0, 170)
    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        fontsize=5.5,
        handlelength=1.2,
        columnspacing=0.55,
    )
    style_panel(ax)
    ax.tick_params(axis="x", labelsize=7)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.76, bottom=0.18)
    save_figure(fig, output_dir, "learning_v2_matching_outcomes", dpi)


def plot_detection_quality(study: dict, output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.65))
    x = np.arange(len(METHODS))
    width = 0.23
    series = (
        ("Precision", "precision", GREEN),
        ("Recall", "recall", RED),
        ("F1", "f1", PURPLE),
    )
    for index, (label, field, color) in enumerate(series):
        values = [study["methods"][key][field] for key, _ in METHODS]
        bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
        add_value_labels(ax, bars, percent=True, fontsize=5.5, rotation=90)
    ax.set_xticks(x, [name for _, name in METHODS])
    ax.set_ylabel("Score", fontsize=7)
    ax.set_title("Final detection quality", fontsize=8, pad=18)
    ax.set_ylim(0, 1.0)
    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        fontsize=5.5,
        handlelength=1.2,
        columnspacing=0.9,
    )
    style_panel(ax, percent=True)
    ax.tick_params(axis="x", labelsize=7)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.76, bottom=0.18)
    save_figure(fig, output_dir, "learning_v2_detection_quality", dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    with args.study.open("r", encoding="utf-8") as handle:
        study = json.load(handle)
    if study.get("truth_source") != "independent_manual":
        raise ValueError("The overview requires the independent-manual final study.")

    configure_style()
    plot_matching_outcomes(study, args.output_dir, args.dpi)
    plot_detection_quality(study, args.output_dir, args.dpi)


if __name__ == "__main__":
    main()
