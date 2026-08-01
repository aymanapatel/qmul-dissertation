#!/usr/bin/env python3
"""
Export models_multiview_v2 metrics as LaTeX tables.

The script reads JSON artifacts only, so it does not require torch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MODEL_DIR = Path("3_Learning/models_multiview_v2")
DEFAULT_OUTPUT = Path("3_Learning/reports/model_metrics/models_multiview_v2_metrics.tex")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value, ndigits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, (int, float)):
        if abs(value) < 0.01 and value != 0:
            return f"{value:.4g}"
        return f"{value:.{ndigits}f}"
    return str(value).replace("_", r"\_")


def tex_escape(value: str) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def view_label(view: str) -> str:
    return {
        "a11y-tree": "Accessibility tree",
        "dom": "DOM",
        "rendered-visual": "Rendered visual",
    }.get(view, view)


def table_split(split: dict) -> str:
    train = len(split.get("train", []))
    val = len(split.get("val", []))
    test = len(split.get("test", []))
    total = train + val + test
    args = split.get("args", {})
    rows = [
        ("Architecture", split.get("architecture", "--")),
        ("Training sites", train),
        ("Validation sites", val),
        ("Test sites", test),
        ("Total sites", total),
        ("Reused split", split.get("reused_split", False)),
        ("Epochs", args.get("epochs")),
        ("Batch size", args.get("batch_size")),
        ("Learning rate", args.get("lr")),
        ("Hidden dimension", args.get("hidden")),
        ("Layers", args.get("layers")),
        ("Attention heads", args.get("heads")),
        ("Dropout", args.get("dropout")),
        ("Selection metric", args.get("selection_metric")),
    ]
    body = "\n".join(f"{tex_escape(k)} & {fmt(v)} \\\\" for k, v in rows)
    return rf"""
\begin{{table}}[htbp]
\centering
\caption{{Dataset split and training configuration for \texttt{{models\_multiview\_v2}}.}}
\label{{tab:multiview-v2-split}}
\begin{{tabular}}{{lr}}
\toprule
Property & Value \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
""".strip()


def table_test_metrics(manifest: dict) -> str:
    lines = []
    for view, details in manifest.get("views", {}).items():
        metrics = details.get("test_metrics", {})
        if not metrics:
            lines.append(
                f"{tex_escape(view_label(view))} & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\\\"
            )
            continue
        lines.append(
            " & ".join(
                [
                    tex_escape(view_label(view)),
                    fmt(metrics.get("loss")),
                    fmt(metrics.get("node_precision")),
                    fmt(metrics.get("node_recall")),
                    fmt(metrics.get("node_f1_pos")),
                    fmt(metrics.get("rule_precision_micro")),
                    fmt(metrics.get("rule_recall_micro")),
                    fmt(metrics.get("rule_f1_micro")),
                    fmt(metrics.get("graph_precision")),
                    fmt(metrics.get("graph_recall")),
                    fmt(metrics.get("graph_f1")),
                ]
            )
            + r" \\"
        )
    body = "\n".join(lines)
    return rf"""
\begin{{table}}[htbp]
\centering
\caption{{Held-out test metrics for each specialist view in \texttt{{models\_multiview\_v2}}. The rendered-visual specialist has a checkpoint but no saved test metrics in the manifest.}}
\label{{tab:multiview-v2-test-metrics}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrrrrr}}
\toprule
View & Loss & Node P & Node R & Node F1 & Rule P$_\mu$ & Rule R$_\mu$ & Rule F1$_\mu$ & Graph P & Graph R & Graph F1 \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
""".strip()


def table_calibration(model_dir: Path, calibration: dict) -> str:
    lines = []
    for view, details in calibration.get("views", {}).items():
        rec = details.get("recommended", {})
        graph = details.get("best_graph", {})
        node = details.get("best_node_precision_floor", {})
        rule = details.get("best_rule_precision_floor", {})
        thresholds = details.get("rule_thresholds", {})
        disabled_rules = sum(1 for threshold in thresholds.values() if isinstance(threshold, (int, float)) and threshold > 1)
        lines.append(
            " & ".join(
                [
                    tex_escape(view_label(view)),
                    fmt(rec.get("graph_threshold")),
                    fmt(rec.get("node_threshold")),
                    fmt(rec.get("rule_threshold")),
                    fmt(graph.get("f1")),
                    fmt(node.get("precision")),
                    fmt(node.get("recall")),
                    fmt(node.get("f1")),
                    fmt(rule.get("precision")),
                    fmt(rule.get("recall")),
                    fmt(rule.get("f1")),
                    str(disabled_rules),
                ]
            )
            + r" \\"
        )
    body = "\n".join(lines)
    return rf"""
\begin{{table}}[htbp]
\centering
\caption{{Validation calibration summary for \texttt{{models\_multiview\_v2}} using a precision floor of {fmt(calibration.get("precision_floor"))}. Thresholds of 1.01 indicate disabled prediction heads because the precision floor could not be met.}}
\label{{tab:multiview-v2-calibration}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrrrrrr}}
\toprule
View & Graph $\tau$ & Node $\tau$ & Rule $\tau$ & Best Graph F1 & Node P & Node R & Node F1 & Rule P & Rule R & Rule F1 & Disabled Rules \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
""".strip()


def table_rules(manifest: dict, which: str) -> str:
    lines = []
    for view, details in manifest.get("views", {}).items():
        metrics = details.get("test_metrics", {})
        rules = metrics.get(which, [])
        if not rules:
            lines.append(f"{tex_escape(view_label(view))} & -- & -- & -- & -- & -- \\\\")
            continue
        names = [tex_escape(rule) for rule, _score in rules[:5]]
        scores = [fmt(score, 4) for _rule, score in rules[:5]]
        lines.append(
            f"{tex_escape(view_label(view))} & "
            + " & ".join(f"{name} ({score})" for name, score in zip(names, scores))
            + r" \\"
        )
    body = "\n".join(lines)
    caption_kind = "highest" if which == "top_rules" else "lowest"
    label_kind = "top" if which == "top_rules" else "worst"
    return rf"""
\begin{{table}}[htbp]
\centering
\caption{{Rules with the {caption_kind} saved F1 scores per specialist view.}}
\label{{tab:multiview-v2-{label_kind}-rules}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{llllll}}
\toprule
View & Rule 1 & Rule 2 & Rule 3 & Rule 4 & Rule 5 \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}
""".strip()


def hf_dataset_note() -> str:
    return r"""
\begin{table}[htbp]
\centering
\caption{Hugging Face Dataset Viewer metrics placeholder. A dataset repository identifier, for example \texttt{namespace/repo}, is required before Dataset Viewer metrics such as split sizes, parquet shards, and column statistics can be queried.}
\label{tab:hf-dataset-viewer-placeholder}
\begin{tabular}{ll}
\toprule
Required input & Dataset repository id (\texttt{namespace/repo}) \\
\midrule
Available read-only endpoints & \texttt{/splits}, \texttt{/size}, \texttt{/statistics}, \texttt{/parquet} \\
Current status & Not queried; no Hugging Face dataset id was provided \\
\bottomrule
\end{tabular}
\end{table}
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export models_multiview_v2 metrics as LaTeX.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model_dir = args.model_dir
    manifest = load_json(model_dir / "manifest.json")
    split = load_json(model_dir / "split.json")
    calibration = load_json(model_dir / "calibration_multiview.json")

    sections = [
        "% Requires \\usepackage{booktabs} and \\usepackage{graphicx} for \\resizebox.",
        table_split(split),
        table_test_metrics(manifest),
        table_calibration(model_dir, calibration),
        table_rules(manifest, "top_rules"),
        table_rules(manifest, "worst_rules"),
        hf_dataset_note(),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote LaTeX metrics to {args.output}")


if __name__ == "__main__":
    main()
