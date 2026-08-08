"""Same-session webpage capture and frozen specialist inference for the demo API.

This module is deliberately inference-only. Axe output is stored as audit
evidence and may populate labels in a cached graph, but the model receives only
the feature-contract tensors: ``x``, ``edge_index`` and ``tag_indices``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import torch
from bs4 import BeautifulSoup, Tag

from .build_same_session_ax_cache import SNAPSHOT_MARKER_ATTRIBUTE, build_site
from .contracts import DetectorObservation
from .data import sanitise_graph
from .fusion import FusionPolicy, fuse_observations
from .models import ModelConfig, build_model
from .rules import rule_metadata
from .schema import FeatureContract, inference_fingerprint


LEARNING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LEARNING_ROOT.parent
LEARNING_V1_SRC = LEARNING_ROOT / "learning_v1" / "src"
BROWSER_CAPTURE_SRC = REPO_ROOT / "2_Data" / "browser-use"
for dependency_path in (BROWSER_CAPTURE_SRC, LEARNING_V1_SRC):
    if str(dependency_path) not in sys.path:
        sys.path.insert(0, str(dependency_path))

from feature_extractor import FeatureExtractor  # noqa: E402
from graph_sources import GRAPH_SOURCE_RENDERED_VISUAL  # noqa: E402
from html_graph_builder import get_dom_path  # noqa: E402
from rendered_snapshot import (  # noqa: E402
    CAPTURE_SCRIPT,
    CLEANUP_SCRIPT,
    SNAPSHOT_VERSION,
    build_backend_marker_map,
)


DEFAULT_PHASE5 = LEARNING_ROOT / "learning_v2/artifacts_evidence-v3.0/phase_5_multiview_final_v2"
DEFAULT_FUSION_POLICY = LEARNING_ROOT / "learning_v2/artifacts_evidence-v3.0/phase_6_7_final/phase_6_fusion_policy.json"
DEFAULT_ARCHITECTURES = ("mlp", "graphsage", "gat")
MODEL_LOCK = threading.RLock()
ProgressCallback = Callable[[str, str, str, dict[str, Any]], None]


def _progress(
    callback: ProgressCallback | None,
    event_id: str,
    status: str,
    label: str,
    **details: Any,
) -> None:
    if callback is not None:
        callback(event_id, status, label, details)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_aligned_page(page, site_dir: Path, axe_report: dict[str, Any]) -> dict[str, Any]:
    """Capture marked HTML, visual features, AX tree and screenshot once."""

    site_dir.mkdir(parents=True, exist_ok=True)
    html_path = site_dir / "0.html"
    visual_path = site_dir / "0.visual.json"
    ax_path = site_dir / "0.ax.json"
    axe_path = site_dir / "page-0_home.json"
    screenshot_path = site_dir / "page.png"
    snapshot = None
    try:
        snapshot = page.evaluate(CAPTURE_SCRIPT)
        if isinstance(snapshot, str):
            snapshot = json.loads(snapshot)
        session = page.context.new_cdp_session(page)
        document = session.send("DOM.getDocument", {"depth": -1, "pierce": True})["root"]
        ax_nodes = session.send("Accessibility.getFullAXTree").get("nodes", [])
        backend_to_marker = build_backend_marker_map(document)
        page.screenshot(path=screenshot_path, full_page=True)
        mapped = sum(
            str(node.get("backendDOMNodeId")) in backend_to_marker
            for node in ax_nodes
        )
        ax_payload = {
            "version": SNAPSHOT_VERSION,
            "captured_at": datetime.now(UTC).isoformat(),
            "captured_url": snapshot["visual"].get("captured_url"),
            "viewport": snapshot["visual"].get("viewport"),
            "nodes": ax_nodes,
            "backend_dom_to_snapshot_node": backend_to_marker,
            "mapping_stats": {
                "dom_nodes_with_marker": len(backend_to_marker),
                "ax_nodes": len(ax_nodes),
                "ax_nodes_mapped_to_snapshot": mapped,
            },
        }
        html_path.write_text(snapshot["html"], encoding="utf-8")
        visual_path.write_text(json.dumps(snapshot["visual"], indent=2), encoding="utf-8")
        ax_path.write_text(json.dumps(ax_payload, indent=2), encoding="utf-8")
        axe_path.write_text(json.dumps(axe_report, indent=2), encoding="utf-8")
    finally:
        if snapshot is not None:
            page.evaluate(CLEANUP_SCRIPT)
    return {
        "site_dir": str(site_dir),
        "screenshot": str(screenshot_path),
        "html_sha256": _sha256(html_path),
        "visual_sha256": _sha256(visual_path),
        "ax_sha256": _sha256(ax_path),
        "axe_sha256": _sha256(axe_path),
    }


@lru_cache(maxsize=2)
def _extractor(device: str) -> FeatureExtractor:
    return FeatureExtractor(device=device)


@lru_cache(maxsize=8)
def _checkpoint_bundle(run_dir_text: str, device: str):
    run_dir = Path(run_dir_text)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    calibration = json.loads((run_dir / "calibration.json").read_text(encoding="utf-8"))
    config = ModelConfig.from_dict(checkpoint["model_config"])
    contract = FeatureContract.from_dict(checkpoint["feature_contract"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, calibration, contract


def _element_for_marker(soup: BeautifulSoup, marker: int) -> Tag | None:
    return soup.find(attrs={SNAPSHOT_MARKER_ATTRIBUTE: str(marker)}) if marker >= 0 else None


def _clean_html(element: Tag | None) -> str:
    if element is None:
        return ""
    copied = BeautifulSoup(str(element), "lxml").find()
    if isinstance(copied, Tag):
        copied.attrs.pop(SNAPSHOT_MARKER_ATTRIBUTE, None)
        return str(copied)[:3000]
    return ""


def _node_evidence(
    *, view: str, node_index: int, graph, node_map: dict[int, Any], soup: BeautifulSoup,
) -> dict[str, Any]:
    if view == "rendered-visual":
        node = node_map.get(node_index)
        if node is None:
            return {"selector": "html", "html": "", "text": ""}
        marker = str(getattr(node, "attrs", {}).get(SNAPSHOT_MARKER_ATTRIBUTE, ""))
        try:
            element = _element_for_marker(soup, int(marker))
        except (TypeError, ValueError):
            element = None
        visual = dict(getattr(node, "visual", {}))
        return {
            "selector": getattr(node, "dom_path", "") or "html",
            "html": _clean_html(element),
            "text": str(getattr(node, "text_content", ""))[:500],
            "tag": str(getattr(node, "tag", "")),
            "visual": {
                key: visual.get(key)
                for key in (
                    "foreground_rgb", "background_rgb", "contrast_ratio",
                    "required_contrast_ratio", "font_size", "font_weight",
                )
            },
        }
    dom_indices = getattr(graph, "dom_indices", None)
    marker = int(dom_indices[node_index]) if dom_indices is not None else -1
    element = _element_for_marker(soup, marker)
    return {
        "selector": get_dom_path(element) if element is not None else "html",
        "html": _clean_html(element),
        "text": element.get_text(" ", strip=True)[:500] if element is not None else "",
        "tag": str(element.name) if element is not None else "",
        "snapshot_node_id": marker,
    }


def _score_view(
    *, view: str, graph, node_map: dict[int, Any], site_dir: Path,
    phase5_dir: Path, architecture: str, device: str,
) -> dict[str, Any]:
    run_dir = phase5_dir / view / architecture
    model, checkpoint, calibration, contract = _checkpoint_bundle(str(run_dir.resolve()), device)
    graph = sanitise_graph(
        graph, graph_source=view, rule_indices=list(checkpoint["rule_indices"]),
        require_labels=False,
    )
    contract.validate(graph)
    graph = graph.to(device)
    with torch.no_grad():
        x, edge_index, tag_indices = inference_fingerprint(graph)
        probabilities = torch.sigmoid(model(x, edge_index, tag_indices)).cpu()
    graph = graph.cpu()
    valid = getattr(graph, "label_mask", torch.ones(graph.num_nodes, dtype=torch.bool)).bool()
    soup = BeautifulSoup((site_dir / "0.html").read_text(encoding="utf-8"), "lxml")
    rules = []
    findings = []
    for local_index, rule_id in enumerate(checkpoint["rule_ids"]):
        scores = probabilities[:, local_index].clone()
        scores[~valid] = -1.0
        node_index = int(scores.argmax()) if scores.numel() else 0
        probability = max(0.0, float(scores[node_index])) if scores.numel() else 0.0
        threshold = float(calibration["recommended"]["rule_thresholds"][rule_id])
        predicted_fail = probability >= threshold
        rule = {
            **rule_metadata(rule_id),
            "probability": round(probability, 8),
            "threshold": round(threshold, 8),
            "predicted_fail": predicted_fail,
            "node_index": node_index,
        }
        rules.append(rule)
        if predicted_fail:
            evidence = _node_evidence(
                view=view, node_index=node_index, graph=graph,
                node_map=node_map, soup=soup,
            )
            findings.append({
                **rule,
                "graph_view": view,
                "architecture": architecture,
                "detector_id": f"{view}:{architecture}:{rule_id}",
                "evidence": evidence,
            })
    return {
        "view": view,
        "architecture": architecture,
        "axe_used_for_prediction": False,
        "node_count": int(graph.num_nodes),
        "edge_count": int(graph.edge_index.shape[1]),
        "feature_contract": contract.to_dict(),
        "checkpoint_sha256": _sha256(run_dir / "best_model.pt"),
        "rules": rules,
        "findings": findings,
    }


def _fuse_predictions(predictions: list[dict[str, Any]], policy_path: Path) -> list[dict[str, Any]]:
    raw_policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy = FusionPolicy(
        source_thresholds=raw_policy["source_thresholds"],
        fail_threshold=float(raw_policy["fail_threshold"]),
        review_threshold=float(raw_policy["review_threshold"]),
        schema_version=int(raw_policy.get("schema_version", 1)),
    )
    output = []
    for prediction in predictions:
        selector = str(prediction["evidence"].get("selector") or "page")
        for criterion_id in prediction["wcag_ids"]:
            raw_id = f"live|{criterion_id}|{selector}|{prediction['detector_id']}"
            observation = DetectorObservation(
                observation_id=hashlib.sha256(raw_id.encode()).hexdigest()[:20],
                site_id="live-input",
                criterion_id=str(criterion_id),
                detector_id=prediction["detector_id"],
                status="fail",
                confidence=float(prediction["probability"]),
                rule_id=prediction["rule_id"],
                target_id=selector,
                evidence={
                    "probability": prediction["probability"],
                    "threshold": prediction["threshold"],
                    "graph_view": prediction["graph_view"],
                    "architecture": prediction["architecture"],
                    "axe_used_for_prediction": False,
                },
            )
            fused = fuse_observations([observation], policy)[0]
            output.append({
                **prediction,
                "criterion_id": str(criterion_id),
                "routing_status": fused.status,
                "routing_confidence": fused.confidence,
                "human_review_required": fused.human_review_required,
                "contributing_observations": list(fused.contributing_observations),
            })
    return output


def run_live_specialists(
    site_dir: Path,
    output_dir: Path,
    *,
    phase5_dir: Path = DEFAULT_PHASE5,
    fusion_policy_path: Path = DEFAULT_FUSION_POLICY,
    architectures: tuple[str, ...] = DEFAULT_ARCHITECTURES,
    device: str = "cpu",
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Build both graph views once and run every requested frozen specialist."""

    required = [site_dir / name for name in ("0.html", "0.visual.json", "0.ax.json", "page-0_home.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Live graph evidence is incomplete: {missing}")
    unsupported = sorted(set(architectures) - set(DEFAULT_ARCHITECTURES))
    if unsupported:
        raise ValueError(f"Unsupported live architectures: {unsupported}")
    missing_artifacts = []
    for architecture in architectures:
        for view in ("a11y-tree", "rendered-visual"):
            run_dir = phase5_dir / view / architecture
            for name in ("best_model.pt", "calibration.json", "manifest.json"):
                if not (run_dir / name).is_file():
                    missing_artifacts.append(str(run_dir / name))
    if missing_artifacts:
        raise FileNotFoundError(f"Frozen specialist artifacts are incomplete: {missing_artifacts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with MODEL_LOCK:
        extractor = _extractor(device)
        a11y_path = output_dir / "a11y-tree.pt"
        _progress(progress, "build_a11y_tree", "running", "Build accessibility-tree graph")
        build_site(site_dir, a11y_path, extractor.text_model)
        a11y_raw = torch.load(a11y_path, map_location="cpu", weights_only=False)["data"]
        _progress(
            progress, "build_a11y_tree", "completed", "Build accessibility-tree graph",
            node_count=int(a11y_raw.num_nodes), edge_count=int(a11y_raw.edge_index.shape[1]),
            artifact="learning_v2_graphs/a11y-tree.pt",
        )

        _progress(progress, "build_rendered_visual", "running", "Build rendered-visual graph")
        rendered_page = extractor.process_page(
            html_path=site_dir / "0.html",
            axe_report_path=None,
            extract_visual=True,
            graph_source=GRAPH_SOURCE_RENDERED_VISUAL,
        )
        rendered_path = output_dir / "rendered-visual.pt"
        rendered_page.save(rendered_path)
        _progress(
            progress, "build_rendered_visual", "completed", "Build rendered-visual graph",
            node_count=int(rendered_page.data.num_nodes),
            edge_count=int(rendered_page.data.edge_index.shape[1]),
            artifact="learning_v2_graphs/rendered-visual.pt",
        )

        views = (
            ("a11y-tree", a11y_raw, {}),
            ("rendered-visual", rendered_page.data, rendered_page.node_map),
        )
        runs = []
        for architecture in architectures:
            for view, graph, node_map in views:
                event_id = f"run_{architecture}_{view.replace('-', '_')}"
                label = f"Run {architecture.upper() if architecture == 'mlp' else architecture.title()} on {view}"
                _progress(
                    progress, event_id, "running", label,
                    architecture=architecture, graph_view=view,
                )
                run = _score_view(
                    view=view, graph=graph, node_map=node_map, site_dir=site_dir,
                    phase5_dir=phase5_dir, architecture=architecture, device=device,
                )
                runs.append(run)
                _progress(
                    progress, event_id, "completed", label,
                    architecture=architecture, graph_view=view,
                    node_count=run["node_count"], edge_count=run["edge_count"],
                    finding_count=len(run["findings"]), checkpoint_sha256=run["checkpoint_sha256"],
                )
    predictions = [finding for run in runs for finding in run["findings"]]
    _progress(
        progress, "route_findings", "running", "Apply frozen calibration and routing",
        candidate_count=len(predictions),
    )
    findings = _fuse_predictions(predictions, fusion_policy_path)
    status_order = {"fail": 0, "needs_review": 1, "pass": 2, "unsupported": 3}
    findings.sort(key=lambda item: (
        status_order.get(str(item["routing_status"]), 9),
        -float(item["routing_confidence"]), -float(item["probability"]),
        str(item["architecture"]), str(item["graph_view"]), str(item["rule_id"]),
    ))
    _progress(
        progress, "route_findings", "completed", "Apply frozen calibration and routing",
        candidate_count=len(predictions), routed_finding_count=len(findings),
    )
    return {
        "schema_version": 2,
        "architectures": list(architectures),
        "training_artifacts": str(phase5_dir),
        "fusion_policy": str(fusion_policy_path),
        "model_runs": runs,
        "findings": findings,
    }


__all__ = ["capture_aligned_page", "run_live_specialists"]
