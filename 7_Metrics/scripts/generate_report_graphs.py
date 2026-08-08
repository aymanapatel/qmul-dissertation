#!/usr/bin/env python3
"""Generate publication-ready figures from the frozen v3.0 evidence artefacts.

The script intentionally reads the JSON evidence files on every run so that the
figures cannot silently drift away from the reported metrics.
"""

from __future__ import annotations

import argparse
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
EVIDENCE_ROOT = REPO_ROOT / "3_Learning" / "learning_v2" / "artifacts_evidence-v3.0"

FINAL_STUDY = EVIDENCE_ROOT / "phase_6_7_final" / "phase_7_detection_study.json"
MULTIVIEW = EVIDENCE_ROOT / "phase_5_multiview_final_v2" / "comparison.json"
VISUAL_ABLATION = EVIDENCE_ROOT / "visual_ablation_final_v2" / "visual_ablation.json"
RETRIEVAL_BENCHMARK = EVIDENCE_ROOT / "controlled_llm_repair_benchmark" / "benchmark_report.json"

BLUE = "#2F5DA8"
TEAL = "#2A9D8F"
ORANGE = "#E07A3F"
PURPLE = "#7A5195"
GOLD = "#D4A72C"
RED = "#C44E52"
GREY = "#777777"
GRID = "#D8D8D8"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required evidence file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, bars, digits: int = 3, padding: float = 2.5) -> None:
    labels = [f"{bar.get_height():.{digits}f}" for bar in bars]
    ax.bar_label(bars, labels=labels, padding=padding, fontsize=7.5, rotation=90)


def plot_final_f1_with_ci(study: dict, output_dir: Path, dpi: int) -> None:
    methods = study["methods"]
    order = [
        "axe_alone",
        "custom_deterministic",
        "axe_plus_custom",
        "mlp_specialist",
        "graphsage_specialist",
        "gat_specialist",
        "visual_specialist",
        "uncalibrated_union",
        "calibrated_routed_fusion",
    ]
    labels = [
        "axe-core",
        "Custom checks",
        "axe + custom",
        "MLP",
        "GraphSAGE",
        "GAT",
        "Visual*",
        "Union",
        "Routed fusion",
    ]
    colors = [GREY, GREY, "#999999", BLUE, TEAL, PURPLE, ORANGE, GOLD, RED]
    values = np.array([methods[key]["f1"] for key in order])
    intervals = np.array([methods[key]["bootstrap_95_ci"]["f1"] for key in order])
    errors = np.vstack((values - intervals[:, 0], intervals[:, 1] - values))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(order))
    bars = ax.barh(y, values, color=colors, edgecolor="white", height=0.68, zorder=3)
    ax.errorbar(values, y, xerr=errors, fmt="none", ecolor="#222222", capsize=3, lw=1, zorder=4)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1 score (95% site-bootstrap CI)")
    ax.set_title("Final detection performance against independent manual truth")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for bar, value, interval in zip(bars, values, intervals):
        ax.text(interval[1] + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    ax.text(
        0,
        -0.18,
        "101 sites, 404 site–criterion pairs; *visual specialist covers colour contrast only (coverage = 0.25).",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.22, bottom=0.22)
    save_figure(fig, output_dir, "final_detection_f1_ci", dpi)


def plot_precision_recall(study: dict, output_dir: Path, dpi: int) -> None:
    methods = study["methods"]
    order = [
        "axe_alone",
        "custom_deterministic",
        "axe_plus_custom",
        "mlp_specialist",
        "graphsage_specialist",
        "gat_specialist",
        "visual_specialist",
        "uncalibrated_union",
        "calibrated_routed_fusion",
    ]
    labels = {
        "axe_alone": "axe-core",
        "custom_deterministic": "Custom",
        "axe_plus_custom": "axe + custom",
        "mlp_specialist": "MLP",
        "graphsage_specialist": "GraphSAGE",
        "gat_specialist": "GAT",
        "visual_specialist": "Visual*",
        "uncalibrated_union": "Union",
        "calibrated_routed_fusion": "Fusion",
    }
    colors = {
        "axe_alone": GREY,
        "custom_deterministic": "#999999",
        "axe_plus_custom": "#666666",
        "mlp_specialist": BLUE,
        "graphsage_specialist": TEAL,
        "gat_specialist": PURPLE,
        "visual_specialist": ORANGE,
        "uncalibrated_union": GOLD,
        "calibrated_routed_fusion": RED,
    }
    offsets = {
        "axe_alone": (5, 5),
        "custom_deterministic": (5, -12),
        "axe_plus_custom": (5, 5),
        "mlp_specialist": (-30, 8),
        "graphsage_specialist": (-10, -15),
        "gat_specialist": (5, 5),
        "visual_specialist": (5, -12),
        "uncalibrated_union": (-32, 7),
        "calibrated_routed_fusion": (-35, 7),
    }

    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    for key in order:
        metric = methods[key]
        coverage = metric.get("coverage", 1.0)
        ax.scatter(
            metric["recall"],
            metric["precision"],
            s=65 + 95 * coverage,
            color=colors[key],
            edgecolor="#222222",
            linewidth=0.7,
            alpha=0.95,
            zorder=3,
        )
        ax.annotate(
            labels[key],
            (metric["recall"], metric["precision"]),
            xytext=offsets[key],
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall operating points on the final manual test")
    ax.text(
        0.02,
        0.02,
        "Marker area encodes criterion coverage; *visual coverage = 0.25.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    save_figure(fig, output_dir, "final_precision_recall_tradeoff", dpi)


def plot_per_criterion_f1(study: dict, output_dir: Path, dpi: int) -> None:
    methods = study["methods"]
    order = ["axe_alone", "mlp_specialist", "graphsage_specialist", "gat_specialist"]
    labels = ["axe-core", "MLP", "GraphSAGE", "GAT"]
    colors = [GREY, BLUE, TEAL, PURPLE]
    criteria = ["1.1.1", "1.4.3", "2.4.4", "4.1.2"]
    criterion_labels = ["1.1.1\nNon-text content", "1.4.3\nContrast", "2.4.4\nLink purpose", "4.1.2\nName, role, value"]

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(len(criteria))
    width = 0.19
    for index, (key, label, color) in enumerate(zip(order, labels, colors)):
        values = [methods[key]["per_criterion"][criterion]["f1"] for criterion in criteria]
        bars = ax.bar(x + (index - 1.5) * width, values, width, label=label, color=color, edgecolor="white")
        add_bar_labels(ax, bars, digits=2, padding=2)
    ax.set_xticks(x, criterion_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1 score")
    ax.set_title("Final per-criterion F1 against independent manual truth")
    ax.legend(ncol=4, loc="upper center")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "final_per_criterion_f1", dpi)


def plot_accessibility_tree_specialists(
    comparison: dict,
    output_dir: Path,
    dpi: int,
) -> None:
    records = [
        record
        for record in comparison["results"]
        if record["view"] == "a11y-tree"
    ]

    preferred_order = ["mlp", "graphsage", "gat"]
    records.sort(
        key=lambda record: preferred_order.index(record["architecture"])
    )

    labels = ["MLP", "GraphSAGE", "GAT"]

    series = [
        ("node_f1", "Node F1", BLUE),
        ("rule_f1", "Rule F1", TEAL),
        ("page_f1", "Page F1", PURPLE),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    x = np.arange(len(records))
    width = 0.23

    for index, (metric_key, metric_label, color) in enumerate(series):
        values = [record["test"][metric_key] for record in records]

        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=metric_label,
            color=color,
            edgecolor="white",
        )

        add_bar_labels(
            ax,
            bars,
            digits=3,
            padding=2,
        )

    ax.set_xticks(x, labels)
    ax.set_ylim(0.75, 1.025)
    ax.set_ylabel("F1 score")
    ax.set_title(
        "Accessibility-tree specialists on the development-label test"
    )

    # Put legend above the chart so colours remain clear.
    ax.legend(
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        columnspacing=1.4,
        handlelength=1.8,
    )

    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()

    save_figure(
        fig,
        output_dir,
        "accessibility_tree_specialist_f1",
        dpi,
    )
def plot_visual_ablation(ablation: dict, output_dir: Path, dpi: int) -> None:
    aggregate = ablation["aggregate"]
    variants = ["full", "without_visual_features", "without_spatial_edges", "structure_only"]
    labels = ["Full", "No visual\nfeatures", "No spatial\nedges", "Structure\nonly"]
    architectures = [("mlp", "MLP", BLUE), ("graphsage", "GraphSAGE", TEAL)]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = np.arange(len(variants))
    width = 0.34
    for index, (architecture, label, color) in enumerate(architectures):
        means = [aggregate[variant][architecture]["rule_f1"]["mean"] for variant in variants]
        stdevs = [aggregate[variant][architecture]["rule_f1"]["stdev"] for variant in variants]
        bars = ax.bar(
            x + (index - 0.5) * width,
            means,
            width,
            yerr=stdevs,
            capsize=3,
            label=label,
            color=color,
            edgecolor="white",
            error_kw={"elinewidth": 1, "ecolor": "#222222"},
        )
        add_bar_labels(ax, bars, digits=3, padding=5)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("Mean colour-contrast rule F1")
    ax.set_title("Rendered-visual ablation across seeds 41, 42, and 43")
    ax.legend(ncol=2, loc="upper right")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.text(0, -0.18, "Error bars show one standard deviation across three seeds.", transform=ax.transAxes, fontsize=8)
    fig.subplots_adjust(bottom=0.23)
    save_figure(fig, output_dir, "visual_ablation_f1", dpi)


def plot_retrieval_benchmark(benchmark: dict, output_dir: Path, dpi: int) -> None:
    metrics = benchmark["metrics"]
    conditions = ["no_rag", "flat_vector_rag", "graph_constrained_rag"]
    labels = ["No retrieval", "Flat vector", "Graph-constrained"]
    series = [
        ("mean_recall_at_k", "Recall@5", BLUE),
        ("source_correctness", "Source correctness", ORANGE),
        ("record_type_diversity", "Record diversity", TEAL),
    ]

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    x = np.arange(len(conditions))
    width = 0.23
    for index, (metric_key, metric_label, color) in enumerate(series):
        values = [metrics[condition][metric_key] for condition in conditions]
        bars = ax.bar(x + (index - 1) * width, values, width, label=metric_label, color=color, edgecolor="white")
        add_bar_labels(ax, bars, digits=3, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.24)
    ax.set_ylabel("Score")
    ax.set_title("Controlled retrieval benchmark under a matched evidence budget")
    ax.legend(ncol=3, loc="upper center")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.text(
        0,
        -0.18,
        "Six queries; top-k = 5; 5,000-character budget; leakage count = 0 for every condition.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.23)
    save_figure(fig, output_dir, "retrieval_benchmark", dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "images",
        help="Directory for generated PNG and PDF figures (default: scripts/images).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300 DPI).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()

    final_study = load_json(FINAL_STUDY)
    comparison = load_json(MULTIVIEW)
    visual_ablation = load_json(VISUAL_ABLATION)
    retrieval_benchmark = load_json(RETRIEVAL_BENCHMARK)

    plot_final_f1_with_ci(final_study, args.output_dir, args.dpi)
    plot_precision_recall(final_study, args.output_dir, args.dpi)
    plot_per_criterion_f1(final_study, args.output_dir, args.dpi)
    plot_accessibility_tree_specialists(comparison, args.output_dir, args.dpi)
    plot_visual_ablation(visual_ablation, args.output_dir, args.dpi)
    plot_retrieval_benchmark(retrieval_benchmark, args.output_dir, args.dpi)

    print(f"Generated 6 figures (PNG and PDF) in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
