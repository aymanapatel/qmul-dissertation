#!/usr/bin/env python3
"""
Visualize training-log metrics for dissertation figures.

Example:
  python3 3_Learning/scripts/visualize_training_log.py \
    --log path/to/training.log
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


DEFAULT_OUTPUT_DIR = Path("3_Learning/reports/training_log_visualization")
DEFAULT_FORMATS = ("png", "pdf")

COLORS = {
    "ink": "#1f2933",
    "muted": "#65758a",
    "grid": "#d8dee6",
    "train": "#2f6f73",
    "val": "#c4663a",
    "node_p": "#517aa3",
    "node_r": "#9b4d48",
    "node_f1": "#2f855a",
    "best": "#b7791f",
    "rule_p": "#6f4e7c",
    "rule_r": "#2d7d8f",
    "rule_f1": "#c2410c",
}

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"

EPOCH_RE = re.compile(
    rf"Epoch\s+(?P<epoch>\d+)/(?P<total>\d+)\s+\|\s+"
    rf"Loss:\s+(?P<loss>{FLOAT})\s+\|\s+"
    rf"NodeAcc:\s+(?P<node_acc>{FLOAT})\s+\|\s+"
    rf"PosPred:\s+(?P<pos_pred>{FLOAT})/(?P<pos_target>{FLOAT})\s+\|\s+"
    rf"GraphAcc:\s+(?P<graph_acc>{FLOAT})\s+\|\s+"
    rf"HardNeg:\s+(?P<hard_neg>\d+)\s+\|\s+"
    rf"HardPos:\s+(?P<hard_pos>\d+)\s+\|\s+"
    rf"PosWeight:\s+(?P<pos_weight>{FLOAT})\s+\|\s+"
    rf"NodeW:\s+(?P<node_w>{FLOAT})\s+\|\s+"
    rf"CleanNodeLoss:\s+(?P<clean_node_loss>{FLOAT})\s+\|\s+"
    rf"PosPageNodeLoss:\s+(?P<pos_page_node_loss>{FLOAT})\s+\|\s+"
    rf"GraphNodeLoss:\s+(?P<graph_node_loss>{FLOAT})\s+\|\|\s+"
    rf"Val Loss:\s+(?P<val_loss>{FLOAT})\s+\|\s+"
    rf"Val Node P/R/F1:\s+(?P<val_node_p>{FLOAT})/(?P<val_node_r>{FLOAT})/(?P<val_node_f1>{FLOAT})\s+\|\s+"
    rf"Val Graph P/R/F1:\s+(?P<val_graph_p>{FLOAT})/(?P<val_graph_r>{FLOAT})/(?P<val_graph_f1>{FLOAT})\s+\|\s+"
    rf"Rule P/R/F1\(micro\):\s+(?P<rule_p>{FLOAT})/(?P<rule_r>{FLOAT})/(?P<rule_f1>{FLOAT})"
)

SAVE_RE = re.compile(r"Saved best model \(node_f1_pos=(?P<node_f1_pos>{})\)".format(FLOAT))
LOADED_RE = re.compile(r"Loaded best model from epoch (?P<epoch>\d+) \(node_f1_pos=(?P<node_f1_pos>{})\)".format(FLOAT))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def to_float(value: str) -> float:
    return float(value)


def to_int(value: str) -> int:
    return int(value)


def parse_log(path: Path) -> tuple[list[dict], list[dict], dict | None]:
    rows: list[dict] = []
    saves: list[dict] = []
    loaded_best: dict | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            raw = epoch_match.groupdict()
            row = {
                "epoch": to_int(raw["epoch"]),
                "total_epochs": to_int(raw["total"]),
                "loss": to_float(raw["loss"]),
                "node_acc": to_float(raw["node_acc"]),
                "pos_pred": to_float(raw["pos_pred"]),
                "pos_target": to_float(raw["pos_target"]),
                "graph_acc": to_float(raw["graph_acc"]),
                "hard_neg": to_int(raw["hard_neg"]),
                "hard_pos": to_int(raw["hard_pos"]),
                "pos_weight": to_float(raw["pos_weight"]),
                "node_w": to_float(raw["node_w"]),
                "clean_node_loss": to_float(raw["clean_node_loss"]),
                "pos_page_node_loss": to_float(raw["pos_page_node_loss"]),
                "graph_node_loss": to_float(raw["graph_node_loss"]),
                "val_loss": to_float(raw["val_loss"]),
                "val_node_p": to_float(raw["val_node_p"]),
                "val_node_r": to_float(raw["val_node_r"]),
                "val_node_f1": to_float(raw["val_node_f1"]),
                "val_graph_p": to_float(raw["val_graph_p"]),
                "val_graph_r": to_float(raw["val_graph_r"]),
                "val_graph_f1": to_float(raw["val_graph_f1"]),
                "rule_p": to_float(raw["rule_p"]),
                "rule_r": to_float(raw["rule_r"]),
                "rule_f1": to_float(raw["rule_f1"]),
            }
            rows.append(row)
            continue

        save_match = SAVE_RE.search(line)
        if save_match:
            saves.append(
                {
                    "after_epoch_line": rows[-1]["epoch"] if rows else None,
                    "node_f1_pos": to_float(save_match.group("node_f1_pos")),
                }
            )
            continue

        loaded_match = LOADED_RE.search(line)
        if loaded_match:
            loaded_best = {
                "epoch": to_int(loaded_match.group("epoch")),
                "node_f1_pos": to_float(loaded_match.group("node_f1_pos")),
            }

    if not rows:
        raise ValueError(f"No epoch lines were parsed from {path}")

    best = -1.0
    for row in rows:
        is_new_best = row["val_node_f1"] > best
        if is_new_best:
            best = row["val_node_f1"]
        row["best_node_f1_so_far"] = best
        row["checkpoint_saved"] = is_new_best

    for save in saves:
        match = next((row for row in rows if abs(row["val_node_f1"] - save["node_f1_pos"]) < 1e-4), None)
        save["matching_epoch"] = match["epoch"] if match else None

    return rows, saves, loaded_best


def pct(value, _pos=None):
    return f"{value:.0%}"


def save_figure(fig, output_dir: Path, stem: str, formats: tuple[str, ...], dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    plt.close(fig)
    return paths


def write_csv(rows: list[dict], saves: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "training_epoch_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    saves_path = output_dir / "checkpoint_save_events.csv"
    with saves_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["after_epoch_line", "node_f1_pos", "matching_epoch"])
        writer.writeheader()
        writer.writerows(saves)

    return metrics_path, saves_path


def plot_loss_and_node_f1(rows: list[dict], saves: list[dict], loaded_best: dict | None):
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.7))

    ax = axes[0]
    ax.plot(epochs, [row["loss"] for row in rows], marker="o", linewidth=2.0, color=COLORS["train"], label="Train loss")
    ax.plot(epochs, [row["val_loss"] for row in rows], marker="s", linewidth=1.8, color=COLORS["val"], label="Validation loss")
    ax.set_title("Loss over training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend()

    start_loss = rows[0]["loss"]
    final_loss = rows[-1]["loss"]
    ax.annotate(
        f"Train loss {start_loss:.2f} -> {final_loss:.2f}",
        xy=(epochs[-1], final_loss),
        xytext=(-95, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "lw": 1.0},
        fontsize=9,
        color=COLORS["ink"],
    )

    ax = axes[1]
    ax.plot(epochs, [row["val_node_p"] for row in rows], color=COLORS["node_p"], linewidth=1.8, label="Node precision")
    ax.plot(epochs, [row["val_node_r"] for row in rows], color=COLORS["node_r"], linewidth=1.8, label="Node recall")
    ax.plot(epochs, [row["val_node_f1"] for row in rows], color=COLORS["node_f1"], marker="o", linewidth=2.1, label="Node F1")
    ax.step(epochs, [row["best_node_f1_so_far"] for row in rows], where="post", color=COLORS["best"], linewidth=2.0, label="Best node F1 so far")

    checkpoint_rows = [row for row in rows if row["checkpoint_saved"] and row["val_node_f1"] > 0]
    if checkpoint_rows:
        ax.scatter(
            [row["epoch"] for row in checkpoint_rows],
            [row["val_node_f1"] for row in checkpoint_rows],
            marker="*",
            s=150,
            color=COLORS["best"],
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label="New best checkpoint",
        )

    best_row = max(rows, key=lambda row: row["val_node_f1"])
    ax.annotate(
        f"Best node F1 = {best_row['val_node_f1']:.3f}",
        xy=(best_row["epoch"], best_row["val_node_f1"]),
        xytext=(15, -30),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "lw": 1.0},
        fontsize=9,
        color=COLORS["ink"],
    )
    if loaded_best:
        ax.text(
            0.02,
            0.04,
            f"Log reports loaded best: epoch {loaded_best['epoch']} ({loaded_best['node_f1_pos']:.4f})",
            transform=ax.transAxes,
            fontsize=8,
            color=COLORS["muted"],
        )

    # ax.set_title("Validation node P/R/F1 and checkpoint saves")
    # ax.set_xlabel("Epoch")
    # ax.set_ylabel("Score")
    # ax.set_ylim(-0.03, 1.03)
    # ax.yaxis.set_major_formatter(FuncFormatter(pct))
    # ax.grid(True)
    # ax.set_axisbelow(True)
    # ax.legend(loc="upper left", fontsize=8)

    ax.set_title("Rule-level micro precision, recall, and F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Micro score")
    ax.set_ylim(-0.03, 0.55)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    fig.tight_layout()
    return fig


def plot_rule_trends(rows: list[dict]):
    epochs = [row["epoch"] for row in rows]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(epochs, [row["rule_p"] for row in rows], marker="o", linewidth=2.0, color=COLORS["rule_p"], label="Rule precision")
    ax.plot(epochs, [row["rule_r"] for row in rows], marker="o", linewidth=2.0, color=COLORS["rule_r"], label="Rule recall")
    ax.plot(epochs, [row["rule_f1"] for row in rows], marker="o", linewidth=2.4, color=COLORS["rule_f1"], label="Rule F1")

    best_rule_row = max(rows, key=lambda row: row["rule_f1"])
    ax.scatter([best_rule_row["epoch"]], [best_rule_row["rule_f1"]], marker="*", s=180, color=COLORS["best"], edgecolor="white", linewidth=0.8, zorder=5)
    ax.annotate(
        f"Best rule F1 = {best_rule_row['rule_f1']:.3f}",
        xy=(best_rule_row["epoch"], best_rule_row["rule_f1"]),
        xytext=(15, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "lw": 1.0},
        fontsize=9,
        color=COLORS["ink"],
    )

    best_recall_row = max(rows, key=lambda row: row["rule_r"])
    ax.annotate(
        f"Peak recall = {best_recall_row['rule_r']:.3f}",
        xy=(best_recall_row["epoch"], best_recall_row["rule_r"]),
        xytext=(10, 18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "lw": 1.0},
        fontsize=9,
        color=COLORS["ink"],
    )

    ax.set_title("Rule-level micro precision, recall, and F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Micro score")
    ax.set_ylim(-0.03, 0.55)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_loss_only(rows: list[dict]):
    epochs = [row["epoch"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.plot(epochs, [row["loss"] for row in rows], marker="o", linewidth=2.0, color=COLORS["train"], label="Train loss")
    ax.plot(epochs, [row["val_loss"] for row in rows], marker="s", linewidth=1.8, color=COLORS["val"], label="Validation loss")
    ax.set_title("Loss over training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_loss_and_rule_together(rows: list[dict]):
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.7))

    ax = axes[0]
    ax.plot(epochs, [row["loss"] for row in rows], marker="o", linewidth=2.0, color=COLORS["train"], label="Train loss")
    ax.plot(epochs, [row["val_loss"] for row in rows], marker="s", linewidth=1.8, color=COLORS["val"], label="Validation loss")
    ax.set_title("Loss over training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, [row["rule_p"] for row in rows], marker="o", linewidth=2.0, color=COLORS["rule_p"], label="Rule precision")
    ax.plot(epochs, [row["rule_r"] for row in rows], marker="o", linewidth=2.0, color=COLORS["rule_r"], label="Rule recall")
    ax.plot(epochs, [row["rule_f1"] for row in rows], marker="o", linewidth=2.4, color=COLORS["rule_f1"], label="Rule F1")

    best_rule_row = max(rows, key=lambda row: row["rule_f1"])
    ax.scatter([best_rule_row["epoch"]], [best_rule_row["rule_f1"]], marker="*", s=180, color=COLORS["best"], edgecolor="white", linewidth=0.8, zorder=5)
    ax.annotate(
        f"Best rule F1 = {best_rule_row['rule_f1']:.3f}",
        xy=(best_rule_row["epoch"], best_rule_row["rule_f1"]),
        xytext=(15, 20),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "lw": 1.0},
        fontsize=9,
        color=COLORS["ink"],
    )

    ax.set_title("Rule-level micro precision, recall, and F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Micro score")
    ax.set_ylim(-0.03, 0.55)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    fig.tight_layout()
    return fig


def plot_dashboard(rows: list[dict]):
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    fig.suptitle("Training dynamics summary", fontsize=15, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    ax.plot(epochs, [row["loss"] for row in rows], color=COLORS["train"], linewidth=2.2, label="Train")
    ax.plot(epochs, [row["val_loss"] for row in rows], color=COLORS["val"], linewidth=2.0, label="Validation")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, [row["val_node_f1"] for row in rows], color=COLORS["node_f1"], marker="o", linewidth=2.1, label="Node F1")
    ax.step(epochs, [row["best_node_f1_so_far"] for row in rows], where="post", color=COLORS["best"], linewidth=2.0, label="Best so far")
    ax.scatter(
        [row["epoch"] for row in rows if row["checkpoint_saved"] and row["val_node_f1"] > 0],
        [row["val_node_f1"] for row in rows if row["checkpoint_saved"] and row["val_node_f1"] > 0],
        marker="*",
        s=130,
        color=COLORS["best"],
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    ax.set_title("Validation node F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.03, 0.62)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.grid(True)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(epochs, [row["rule_p"] for row in rows], color=COLORS["rule_p"], linewidth=2.0, label="Rule P")
    ax.plot(epochs, [row["rule_r"] for row in rows], color=COLORS["rule_r"], linewidth=2.0, label="Rule R")
    ax.plot(epochs, [row["rule_f1"] for row in rows], color=COLORS["rule_f1"], linewidth=2.2, label="Rule F1")
    ax.set_title("Rule micro metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.03, 0.55)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.grid(True)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(epochs, [row["pos_pred"] for row in rows], color=COLORS["node_p"], linewidth=2.0, marker="o", label="Positive predictions")
    ax.plot(epochs, [row["pos_target"] for row in rows], color=COLORS["node_r"], linewidth=2.0, marker="s", label="Positive target")
    ax.set_title("Positive node prediction pressure")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average nodes")
    ax.grid(True)
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def write_notes(rows: list[dict], loaded_best: dict | None, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_node = max(rows, key=lambda row: row["val_node_f1"])
    best_rule = max(rows, key=lambda row: row["rule_f1"])
    peak_rule_recall = max(rows, key=lambda row: row["rule_r"])
    loss_drop = rows[0]["loss"] - rows[-1]["loss"]
    lines = [
        "# Training Log Figure Notes",
        "",
        f"- Epochs parsed: {len(rows)}",
        f"- Training loss decreased from {rows[0]['loss']:.4f} to {rows[-1]['loss']:.4f} (drop {loss_drop:.4f}).",
        f"- Best validation node F1 was {best_node['val_node_f1']:.4f} at epoch {best_node['epoch']}.",
        f"- Best rule micro F1 was {best_rule['rule_f1']:.4f} at epoch {best_rule['epoch']}.",
        f"- Peak rule micro recall was {peak_rule_recall['rule_r']:.4f} at epoch {peak_rule_recall['epoch']}.",
    ]
    if loaded_best:
        lines.append(f"- Log reports loaded best checkpoint from epoch {loaded_best['epoch']} with node_f1_pos={loaded_best['node_f1_pos']:.4f}.")
    path = output_dir / "training_figure_notes.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize training loss, node F1 checkpoint saves, and rule micro metrics.")
    parser.add_argument("--log", type=Path, required=True, help="Path to the pasted training log.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for figures and CSV outputs.")
    parser.add_argument("--formats", nargs="+", default=list(DEFAULT_FORMATS), help="Figure formats, for example: png pdf svg")
    parser.add_argument("--dpi", type=int, default=300, help="Raster export DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()

    rows, saves, loaded_best = parse_log(args.log)
    formats = tuple(fmt.lower().lstrip(".") for fmt in args.formats)

    generated: list[Path] = []
    generated.extend(save_figure(plot_loss_and_node_f1(rows, saves, loaded_best), args.output_dir, "01_loss_and_best_node_f1", formats, args.dpi))
    generated.extend(save_figure(plot_rule_trends(rows), args.output_dir, "02_rule_micro_prf1_trends", formats, args.dpi))
    generated.extend(save_figure(plot_dashboard(rows), args.output_dir, "03_training_dashboard", formats, args.dpi))
    generated.extend(save_figure(plot_loss_and_rule_together(rows), args.output_dir, "04_loss_and_rule_micro_together", formats, args.dpi))
    generated.extend(save_figure(plot_loss_only(rows), args.output_dir, "05_loss_over_training", formats, args.dpi))
    generated.extend(save_figure(plot_rule_trends(rows), args.output_dir, "06_rule_micro_prf1_only", formats, args.dpi))
    metrics_path, saves_path = write_csv(rows, saves, args.output_dir)
    notes_path = write_notes(rows, loaded_best, args.output_dir)

    print(f"Parsed {len(rows)} epochs from {args.log}")
    print(f"Parsed {len(saves)} checkpoint save messages")
    print(f"Wrote {len(generated)} figure files to {args.output_dir}")
    for path in generated:
        print(f"  - {path}")
    print(f"Wrote metrics CSV: {metrics_path}")
    print(f"Wrote checkpoint CSV: {saves_path}")
    print(f"Wrote notes: {notes_path}")


if __name__ == "__main__":
    main()
