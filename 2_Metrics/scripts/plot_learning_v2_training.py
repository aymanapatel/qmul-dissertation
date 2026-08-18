#!/usr/bin/env python3
"""Plot the recorded training dynamics of the final learning_v2 specialist."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qmul-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_HISTORY = (
    REPO_ROOT
    / "3_Learning"
    / "learning_v2"
    / "artifacts_evidence-v3.0"
    / "phase_5_live_ax_final_v4_with_matched_validation_loss"
    / "a11y-tree"
    / "graphsage"
    / "history.json"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "images"

TEAL = "#2A9D8F"
ORANGE = "#E07A3F"
PURPLE = "#7A5195"
GREY = "#777777"
GRID = "#D8D8D8"


def load_history(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Training history not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)
    if not isinstance(history, list) or not history:
        raise ValueError(f"Expected a non-empty epoch list in {path}")
    return history


def extract_points(
    history: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract each recorded data point without smoothing or interpolation."""
    epochs = np.array([point["epoch"] for point in history], dtype=int)
    training_loss = np.array(
        [point["training"]["loss"] for point in history],
        dtype=float,
    )
    missing_matched_loss = [
        point["epoch"]
        for point in history
        if "validation" not in point or "sampled_loss" not in point["validation"]
    ]
    if missing_matched_loss:
        raise ValueError(
            "Training-matched validation loss is absent for epochs "
            f"{missing_matched_loss}. Rerun learning_v2 training with the updated trainer."
        )
    validation_sampled_loss = np.array(
        [point["validation"]["sampled_loss"] for point in history],
        dtype=float,
    )
    validation_full_loss = np.array(
        [point["validation"]["loss"] for point in history],
        dtype=float,
    )
    validation_ap = np.array(
        [point["validation_selection"]["value"] for point in history],
        dtype=float,
    )
    sampled_loss_types = {point["validation"].get("sampled_loss_type") for point in history}
    if sampled_loss_types != {"fixed_sample_training_matched_bce"}:
        raise ValueError(
            "Unexpected sampled validation loss types: "
            f"{sorted(str(item) for item in sampled_loss_types)}"
        )
    sampling_seeds = {point["validation"].get("sampling_seed") for point in history}
    if len(sampling_seeds) != 1:
        raise ValueError(f"Validation sampling seed changed between epochs: {sampling_seeds}")
    if not np.array_equal(epochs, np.arange(1, len(history) + 1)):
        raise ValueError("Epochs must be consecutive and start at 1")
    return epochs, training_loss, validation_sampled_loss, validation_full_loss, validation_ap


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_training(
    epochs: np.ndarray,
    training_loss: np.ndarray,
    validation_sampled_loss: np.ndarray,
    validation_full_loss: np.ndarray,
    validation_ap: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> None:
    best_index = int(np.argmax(validation_ap))
    best_epoch = int(epochs[best_index])
    fig, (loss_ax, metric_ax) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.3),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.10},
    )

    loss_line = loss_ax.plot(
        epochs,
        training_loss,
        color=TEAL,
        marker="o",
        markersize=3.8,
        linewidth=1.7,
        label="Training sampled BCE loss",
        zorder=3,
    )[0]
    validation_line = loss_ax.plot(
        epochs,
        validation_sampled_loss,
        color=ORANGE,
        marker="s",
        markersize=3.6,
        linewidth=1.7,
        label="Validation sampled BCE loss",
        zorder=3,
    )[0]

    loss_ax.set_xlim(0.5, epochs.max() + 0.5)
    loss_ax.set_ylim(bottom=0)
    loss_ax.set_ylabel("Sampled BCE loss")
    loss_ax.set_title("Learning v2 accessibility-tree GraphSAGE training dynamics")
    loss_ax.grid(axis="both")
    loss_ax.legend(
        [loss_line, validation_line],
        [loss_line.get_label(), validation_line.get_label()],
        loc="upper right",
    )

    metric_ax.plot(
        epochs,
        validation_ap,
        color=PURPLE,
        marker="D",
        markersize=3.5,
        linewidth=1.5,
        label="Validation rule macro AP",
        zorder=3,
    )
    metric_ax.scatter(
        [best_epoch],
        [validation_ap[best_index]],
        s=42,
        facecolor="white",
        edgecolor=PURPLE,
        linewidth=1.3,
        zorder=4,
    )
    metric_ax.annotate(
        f"Selected epoch {best_epoch}: {validation_ap[best_index]:.3f}",
        xy=(best_epoch, validation_ap[best_index]),
        xytext=(-115, -19),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": PURPLE, "linewidth": 0.8},
        fontsize=7.5,
    )
    metric_ax.set_ylim(0, 1.02)
    metric_ax.set_ylabel("Macro AP")
    metric_ax.set_xlabel("Epoch")
    metric_ax.set_xticks(np.arange(1, epochs.max() + 1, 2))
    metric_ax.grid(axis="both")
    metric_ax.legend(loc="lower right")

    for axis in (loss_ax, metric_ax):
        axis.axvline(best_epoch, color=GREY, linewidth=0.8, linestyle="--", alpha=0.7)

    fig.text(
        0.125,
        0.015,
        "Validation loss uses the same configured negative policy and a fixed subset each epoch; full-pair BCE is retained in the CSV.",
        fontsize=7.5,
    )
    fig.subplots_adjust(bottom=0.14)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "learning_v2_training_dynamics"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "training_sampled_bce",
                "validation_sampled_bce",
                "validation_full_bce",
                "validation_rule_macro_average_precision",
            ]
        )
        writer.writerows(
            zip(
                epochs.tolist(),
                training_loss.tolist(),
                validation_sampled_loss.tolist(),
                validation_full_loss.tolist(),
                validation_ap.tolist(),
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    points = extract_points(load_history(args.history))
    plot_training(*points, args.output_dir, args.dpi)
    print(
        f"Generated learning_v2 training PNG, PDF, and CSV from {len(points[0])} recorded "
        f"epoch data points in {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
