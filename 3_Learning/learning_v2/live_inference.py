"""Same-session webpage capture and frozen specialist inference for the demo API.

This module is deliberately inference-only. Axe output is stored as audit
evidence and may populate labels in a cached graph, but the model receives only
the feature-contract tensors: ``x``, ``edge_index`` and ``tag_indices``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

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


CONTEXT_ATTRIBUTES = (
    "id", "class", "name", "type", "href", "src", "alt", "title", "placeholder",
    "role", "aria-label", "aria-labelledby", "aria-describedby", "autocomplete",
)


def _bounded_text(value: Any, limit: int = 300) -> str:
    return " ".join(str(value or "").split())[:limit]


def _context_attributes(element: Tag | None) -> dict[str, Any]:
    """Expose useful public attributes without leaking live form values."""

    if element is None:
        return {}
    output: dict[str, Any] = {}
    for name in CONTEXT_ATTRIBUTES:
        if not element.has_attr(name):
            continue
        value = element.get(name)
        if isinstance(value, list):
            output[name] = [_bounded_text(item, 100) for item in value[:12]]
        else:
            output[name] = _bounded_text(value, 500)
    return output


def _normalise_selector(value: Any) -> str:
    """Remove BeautifulSoup's non-CSS document pseudo-node from DOM paths."""

    selector = _bounded_text(value, 1000)
    selector = re.sub(r"^\[document\]:nth-of-type\(1\)\s*>\s*", "", selector)
    return selector or "html"


def _element_summary(element: Tag | None) -> dict[str, Any] | None:
    if element is None:
        return None
    return {
        "tag": str(element.name or ""),
        "selector": _normalise_selector(get_dom_path(element)),
        "attributes": _context_attributes(element),
        "text": _bounded_text(element.get_text(" ", strip=True), 300),
    }


def _referenced_text(soup: BeautifulSoup, raw_ids: Any) -> str:
    values = []
    for item_id in str(raw_ids or "").split():
        referenced = soup.find(id=item_id)
        if isinstance(referenced, Tag):
            text = _bounded_text(referenced.get_text(" ", strip=True), 200)
            if text:
                values.append(text)
    return _bounded_text(" ".join(values), 300)


def _accessible_name_signals(element: Tag | None, soup: BeautifulSoup) -> dict[str, str]:
    if element is None:
        return {}
    signals = {
        "visible_text": _bounded_text(element.get_text(" ", strip=True), 300),
        "aria_label": _bounded_text(element.get("aria-label"), 300),
        "aria_labelledby_text": _referenced_text(soup, element.get("aria-labelledby")),
        "title": _bounded_text(element.get("title"), 300),
        "alt": _bounded_text(element.get("alt"), 300),
    }
    child_alts = [
        _bounded_text(image.get("alt"), 200)
        for image in element.find_all("img", limit=8)
        if _bounded_text(image.get("alt"), 200)
    ]
    if child_alts:
        signals["child_image_alt"] = _bounded_text(" ".join(child_alts), 300)
    return {key: value for key, value in signals.items() if value}


def _nearby_context(element: Tag | None, soup: BeautifulSoup) -> dict[str, Any]:
    if element is None:
        return {}
    previous = element.find_previous_sibling()
    following = element.find_next_sibling()
    heading = element.find_previous(re.compile(r"^h[1-6]$"))
    ancestors = []
    parent = element.parent
    while isinstance(parent, Tag) and parent.name not in {"html", "body"} and len(ancestors) < 3:
        summary = _element_summary(parent)
        if summary:
            ancestors.append(summary)
        parent = parent.parent
    return {
        "document_title": _bounded_text(soup.title.get_text(" ", strip=True) if soup.title else "", 300),
        "nearest_heading": _element_summary(heading if isinstance(heading, Tag) else None),
        "previous_sibling": _element_summary(previous if isinstance(previous, Tag) else None),
        "next_sibling": _element_summary(following if isinstance(following, Tag) else None),
        "ancestors": ancestors,
    }


def _normalise_candidate_text(value: Any, limit: int = 200) -> str:
    return _bounded_text(value, limit).strip(" -_./")


def _candidate(
    *, operation: str, selector: str, new_value: str, derived_from: str,
    verification_level: str, attribute_name: str | None = None,
    css_property: str | None = None, source_value: Any = None,
    requires_human_review: bool = True, computed_contrast_ratio: float | None = None,
) -> dict[str, Any]:
    value = _normalise_candidate_text(new_value)
    result: dict[str, Any] = {
        "operation": {
            "operation": operation,
            "selector": selector,
            "attribute_name": attribute_name,
            "css_property": css_property,
            "new_value": value,
        },
        "derived_from": derived_from,
        "source_value": source_value,
        "verification_level": verification_level,
        "requires_human_review": requires_human_review,
    }
    if computed_contrast_ratio is not None:
        result["computed_contrast_ratio"] = round(float(computed_contrast_ratio), 6)
    return result


def _deduplicate_candidates(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen = set()
    for value in values:
        operation = value.get("operation", {})
        key = (
            operation.get("operation"), operation.get("attribute_name"),
            operation.get("css_property"), str(operation.get("new_value", "")).casefold(),
        )
        if not operation.get("new_value") or key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output[:8]


def _link_candidates(element: Tag, soup: BeautifulSoup, selector: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    href = _bounded_text(element.get("href"), 500)
    if href:
        names = []
        for link in soup.find_all("a", href=element.get("href"), limit=30):
            if link is element:
                continue
            names.extend(_accessible_name_signals(link, soup).values())
        unique_names = {name.casefold(): name for name in names if name}
        if len(unique_names) == 1:
            name = next(iter(unique_names.values()))
            candidates.append(_candidate(
                operation="set_attribute", selector=selector, attribute_name="aria-label",
                new_value=name, derived_from="same_href_accessible_name",
                source_value=href, verification_level="page_consistent",
            ))
        parsed = urlparse(href)
        slug = unquote(parsed.path.rstrip("/").rsplit("/", 1)[-1]) if parsed.path else ""
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,60}", slug):
            name = slug.replace("-", " ").replace("_", " ").strip().title()
            candidates.append(_candidate(
                operation="set_attribute", selector=selector, attribute_name="aria-label",
                new_value=name, derived_from="href_path_segment", source_value=href,
                verification_level="contextual",
            ))
    return _deduplicate_candidates(candidates)


def _image_candidates(element: Tag, soup: BeautifulSoup, selector: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    source = _bounded_text(element.get("src"), 500)
    if source:
        alts = {
            _normalise_candidate_text(image.get("alt")): _normalise_candidate_text(image.get("alt"))
            for image in soup.find_all("img", src=element.get("src"), limit=30)
            if image is not element and _normalise_candidate_text(image.get("alt"))
        }
        if len(alts) == 1:
            alt = next(iter(alts.values()))
            candidates.append(_candidate(
                operation="set_attribute", selector=selector, attribute_name="alt",
                new_value=alt, derived_from="same_src_existing_alt", source_value=source,
                verification_level="page_consistent", requires_human_review=False,
            ))
    for reference_attribute in ("aria-labelledby", "aria-describedby"):
        referenced_text = _normalise_candidate_text(
            _referenced_text(soup, element.get(reference_attribute))
        )
        if referenced_text:
            candidates.append(_candidate(
                operation="set_attribute", selector=selector, attribute_name="alt",
                new_value=referenced_text,
                derived_from=f"{reference_attribute}_referenced_text",
                source_value=referenced_text, verification_level="visible_context",
                requires_human_review=False,
            ))
    figure = element.find_parent("figure")
    caption = figure.find("figcaption") if isinstance(figure, Tag) else None
    caption_text = _normalise_candidate_text(caption.get_text(" ", strip=True) if isinstance(caption, Tag) else "")
    if caption_text:
        candidates.append(_candidate(
            operation="set_attribute", selector=selector, attribute_name="alt",
            new_value=caption_text, derived_from="visible_figcaption",
            source_value=caption_text, verification_level="visible_context",
            requires_human_review=False,
        ))
    parent_link = element.find_parent("a")
    if isinstance(parent_link, Tag):
        for source_name, value in _accessible_name_signals(parent_link, soup).items():
            if value:
                candidates.append(_candidate(
                    operation="set_attribute", selector=selector, attribute_name="alt",
                    new_value=value, derived_from=f"parent_link_{source_name}",
                    source_value=value, verification_level="visible_context",
                    requires_human_review=False,
                ))
    # Adjacent authored prose is more useful than an asset name, but remains
    # contextual evidence and must be confirmed by a person. Restrict this to
    # text-oriented siblings: headings and broad ancestor text describe the
    # section, not necessarily the image itself.
    immediate_siblings = (
        ("previous", element.find_previous_sibling()),
        ("next", element.find_next_sibling()),
    )
    for relation, sibling in immediate_siblings:
        if isinstance(sibling, Tag):
            for source_name, value in _accessible_name_signals(sibling, soup).items():
                if value:
                    candidates.append(_candidate(
                        operation="set_attribute", selector=selector, attribute_name="alt",
                        new_value=value,
                        derived_from=f"{relation}_sibling_{source_name}",
                        source_value=value, verification_level="visible_context",
                        requires_human_review=False,
                    ))
        if isinstance(sibling, Tag) and sibling.name in {
            "p", "span", "small", "strong", "em", "div", "li", "dd", "figcaption",
        }:
            value = _normalise_candidate_text(sibling.get_text(" ", strip=True))
            if value:
                candidates.append(_candidate(
                    operation="set_attribute", selector=selector, attribute_name="alt",
                    new_value=value, derived_from=f"{relation}_sibling_visible_text",
                    source_value=value, verification_level="visible_context",
                    requires_human_review=False,
                ))
    # Also inspect short, text-bearing direct children of the immediate
    # container. This covers descriptive prose separated from the image by a
    # button or another non-text node without leaking broad ancestor content.
    parent = element.parent
    if isinstance(parent, Tag):
        for local in parent.find_all(
            {"p", "span", "small", "strong", "em", "li", "dd", "figcaption"},
            recursive=False, limit=8,
        ):
            value = _normalise_candidate_text(local.get_text(" ", strip=True))
            if value:
                candidates.append(_candidate(
                    operation="set_attribute", selector=selector, attribute_name="alt",
                    new_value=value, derived_from="immediate_container_visible_text",
                    source_value={"selector": _normalise_selector(get_dom_path(local)), "text": value},
                    verification_level="visible_context", requires_human_review=False,
                ))
    if not candidates:
        heading = element.find_previous(re.compile(r"^h[1-6]$"))
        heading_text = _normalise_candidate_text(
            heading.get_text(" ", strip=True) if isinstance(heading, Tag) else ""
        )
        subject = re.sub(r"^\d+\.\s*", "", heading_text)
        subject = re.sub(
            r"\bWCAG\s+[0-9.]+(?:\s*/\s*[0-9.]+)*", "", subject,
            flags=re.IGNORECASE,
        )
        subject = re.sub(r"\bimage-alt\b", "", subject, flags=re.IGNORECASE)
        subject = subject.strip(" —–-:|./")
        if subject:
            sibling_images = (
                parent.find_all("img", recursive=False) if isinstance(parent, Tag) else []
            )
            try:
                position = sibling_images.index(element) + 1
            except ValueError:
                position = 1
            ordinal = f" {position}" if len(sibling_images) > 1 else ""
            candidates.append(_candidate(
                operation="set_attribute", selector=selector, attribute_name="alt",
                new_value=f"{subject} illustration{ordinal}",
                derived_from="nearest_heading_and_image_position",
                source_value={"nearest_heading": heading_text, "image_position": position},
                verification_level="contextual_inference", requires_human_review=False,
            ))
    # A filename is locator metadata, not evidence of image meaning. Never emit
    # filename-only alt candidates. The final fallback above uses only the
    # authored heading and local image position.
    return _deduplicate_candidates(candidates)


def _label_candidates(element: Tag, soup: BeautifulSoup, selector: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    sources: list[tuple[str, str, str]] = []
    placeholder = _normalise_candidate_text(element.get("placeholder"))
    if placeholder:
        sources.append(("placeholder", placeholder, "contextual"))
    fieldset = element.find_parent("fieldset")
    legend = fieldset.find("legend") if isinstance(fieldset, Tag) else None
    legend_text = _normalise_candidate_text(legend.get_text(" ", strip=True) if isinstance(legend, Tag) else "")
    if legend_text:
        sources.append(("fieldset_legend", legend_text, "visible_context"))
    previous = element.find_previous_sibling()
    previous_text = _normalise_candidate_text(
        previous.get_text(" ", strip=True) if isinstance(previous, Tag) else ""
    )
    if previous_text:
        instruction = re.sub(
            r"^(?:enter|select|choose|provide|type)\s+(?:your\s+)?", "", previous_text,
            flags=re.IGNORECASE,
        ).strip(" .:;!?")
        if instruction:
            sources.append(("previous_visible_instruction", instruction.capitalize(), "visible_context"))
    for attribute in ("name", "id"):
        raw = _normalise_candidate_text(element.get(attribute))
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,60}", raw):
            sources.append((f"target_{attribute}", raw.replace("-", " ").replace("_", " ").title(), "weak_context"))
    for derived_from, value, level in sources:
        candidates.append(_candidate(
            operation="insert_label_before", selector=selector, attribute_name=None,
            css_property=None, new_value=value, derived_from=derived_from,
            source_value=value, verification_level=level,
        ))
    return _deduplicate_candidates(candidates)


def _luminance(rgb: tuple[int, int, int]) -> float:
    channels = []
    for value in rgb:
        channel = value / 255
        channels.append(channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    first, second = sorted((_luminance(left), _luminance(right)), reverse=True)
    return (first + 0.05) / (second + 0.05)


def _rgb(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        result = tuple(max(0, min(255, int(round(float(item))))) for item in value[:3])
    except (TypeError, ValueError):
        return None
    return result if len(result) == 3 else None


def _contrast_context(visual: dict[str, Any], selector: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    foreground = _rgb(visual.get("foreground_rgb"))
    background = _rgb(visual.get("background_rgb"))
    try:
        observed = float(visual.get("contrast_ratio"))
        required = float(visual.get("required_contrast_ratio"))
    except (TypeError, ValueError):
        observed = required = 0.0
    state = {
        "observed_contrast_ratio": observed,
        "required_contrast_ratio": required,
        "meets_requirement": bool(observed and required and observed >= required),
    }
    if not background or not required or state["meets_requirement"]:
        return state, []
    black = (0, 0, 0)
    white = (255, 255, 255)
    options = [("#000000", _contrast(black, background)), ("#ffffff", _contrast(white, background))]
    value, ratio = max(options, key=lambda item: item[1])
    if ratio < required:
        return state, []
    return state, [_candidate(
        operation="set_style_property", selector=selector, css_property="color",
        new_value=value, derived_from="computed_maximum_black_or_white_contrast",
        source_value={"foreground_rgb": foreground, "background_rgb": background},
        verification_level="computed", requires_human_review=False,
        computed_contrast_ratio=ratio,
    )]


def _repair_context(
    *, rule_id: str, element: Tag | None, soup: BeautifulSoup, selector: str,
    visual: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build bounded, provenance-labelled context for the four live rules."""

    context = {
        "rule_id": rule_id,
        "target": _element_summary(element),
        "accessible_name_signals": _accessible_name_signals(element, soup),
        "nearby": _nearby_context(element, soup),
        "current_state": {},
        "bounded_candidates": [],
        "candidate_policy": (
            "Copy candidate operations exactly. Computed candidates may be proposed without review. "
            "Semantic candidates are suggestions only and require human confirmation unless explicitly verified. "
            "Image-alt uses automatic contextual mode and does not require human review. "
            "For image-alt, filenames, paths, asset identifiers, and hashes are locator metadata only and "
            "must never become alt text."
        ),
    }
    if element is None:
        return context
    if rule_id == "link-name" and element.name == "a":
        context["bounded_candidates"] = _link_candidates(element, soup, selector)
    elif rule_id == "image-alt" and element.name == "img":
        context["bounded_candidates"] = _image_candidates(element, soup, selector)
    elif rule_id == "label" and element.name in {"input", "select", "textarea"}:
        context["bounded_candidates"] = _label_candidates(element, soup, selector)
    elif rule_id == "color-contrast":
        state, candidates = _contrast_context(visual or {}, selector)
        context["current_state"] = state
        context["bounded_candidates"] = candidates
    return context


def _visual_evidence_payload(site_dir: Path) -> dict[str, Any]:
    """Return JSON-safe rendered elements for UI overlays and prompt inspection."""

    raw = json.loads((site_dir / "0.visual.json").read_text(encoding="utf-8"))
    soup = BeautifulSoup((site_dir / "0.html").read_text(encoding="utf-8"), "lxml")
    axe_markers: set[str] = set()
    axe_contrast_available = False
    axe_path = site_dir / "page-0_home.json"
    if axe_path.is_file():
        axe_report = json.loads(axe_path.read_text(encoding="utf-8"))
        for violation in axe_report.get("violations", []) if isinstance(axe_report, dict) else []:
            if violation.get("id") != "color-contrast":
                continue
            axe_contrast_available = True
            for node in violation.get("nodes", []):
                for target in node.get("target", []):
                    if not isinstance(target, str):
                        continue
                    try:
                        matches = soup.select(target)
                    except Exception:
                        matches = []
                    for match in matches:
                        marker = str(match.get(SNAPSHOT_MARKER_ATTRIBUTE, ""))
                        if marker:
                            axe_markers.add(marker)
    viewport = raw.get("viewport", {}) if isinstance(raw, dict) else {}
    raw_nodes = raw.get("nodes", []) if isinstance(raw, dict) else []
    elements = []
    canvas_width = float(viewport.get("width", 0) or 0)
    canvas_height = float(viewport.get("height", 0) or 0)
    for raw_node in raw_nodes[:5000] if isinstance(raw_nodes, list) else []:
        if not isinstance(raw_node, dict) or not raw_node.get("visible", False):
            continue
        marker = str(raw_node.get("snapshot_node_id", ""))
        try:
            element = _element_for_marker(soup, int(marker))
        except (TypeError, ValueError):
            element = None
        if element is None:
            continue
        try:
            x = float(raw_node.get("x", 0) or 0)
            y = float(raw_node.get("y", 0) or 0)
            width = max(0.0, float(raw_node.get("width", 0) or 0))
            height = max(0.0, float(raw_node.get("height", 0) or 0))
            ratio = float(raw_node.get("contrast_ratio", 0) or 0)
            required = float(raw_node.get("required_contrast_ratio", 0) or 0)
        except (TypeError, ValueError):
            continue
        canvas_width = max(canvas_width, x + width)
        canvas_height = max(canvas_height, y + height)
        has_direct_text = bool(raw_node.get("has_direct_text", False))
        numeric_contrast_failure = bool(
            has_direct_text and ratio > 0 and required > 0 and ratio < required
        )
        contrast_fails = (
            marker in axe_markers if axe_contrast_available else numeric_contrast_failure
        )
        selector = _normalise_selector(get_dom_path(element))
        visual_values = {
            "foreground_rgb": raw_node.get("foreground_rgb"),
            "background_rgb": raw_node.get("background_rgb"),
            "contrast_ratio": ratio,
            "required_contrast_ratio": required,
            "contrast_deficit": raw_node.get("contrast_deficit"),
            "font_size": raw_node.get("font_size"),
            "font_weight": raw_node.get("font_weight"),
            "opacity": raw_node.get("opacity"),
            "has_direct_text": has_direct_text,
        }
        rendered_element = {
            "source": "same-session-rendered-visual-capture",
            "snapshot_node_id": marker,
            "selector": selector,
            "tag": str(element.name or ""),
            "text": _bounded_text(element.get_text(" ", strip=True), 300),
            "bounds": {"x": x, "y": y, "width": width, "height": height},
            "visible": True,
            "in_viewport": bool(raw_node.get("in_viewport", False)),
            "clipped": bool(raw_node.get("clipped", False)),
            "visual": visual_values,
            "numeric_contrast_failure": numeric_contrast_failure,
            "contrast_failure": contrast_fails,
            "contrast_failure_source": (
                "axe-core+same-session-rendered-geometry"
                if contrast_fails and axe_contrast_available
                else "rendered-numeric-fallback" if contrast_fails else None
            ),
        }
        if contrast_fails:
            rendered_element["repair_context"] = _repair_context(
                rule_id="color-contrast", element=element, soup=soup,
                selector=selector, visual=visual_values,
            )
        elements.append(rendered_element)
    failures = [item for item in elements if item["contrast_failure"]]
    return {
        "source": "same-session-rendered-visual-capture",
        "contrast_highlight_policy": (
            "axe-core targets joined to same-session rendered geometry"
            if axe_contrast_available else "rendered numeric threshold fallback"
        ),
        "viewport": viewport,
        "canvas": {"width": canvas_width, "height": canvas_height},
        "element_count": len(elements),
        "contrast_failure_count": len(failures),
        "elements": elements,
        "contrast_failures": failures,
    }


def _node_evidence(
    *, rule_id: str, view: str, node_index: int, graph, node_map: dict[int, Any], soup: BeautifulSoup,
) -> dict[str, Any]:
    if view == "rendered-visual":
        node = node_map.get(node_index)
        if node is None:
            return {
                "selector": "html", "html": "", "text": "",
                "repair_context": _repair_context(
                    rule_id=rule_id, element=None, soup=soup, selector="html",
                ),
            }
        marker = str(getattr(node, "attrs", {}).get(SNAPSHOT_MARKER_ATTRIBUTE, ""))
        try:
            element = _element_for_marker(soup, int(marker))
        except (TypeError, ValueError):
            element = None
        visual = dict(getattr(node, "visual", {}))
        selector = _normalise_selector(getattr(node, "dom_path", ""))
        visual_evidence = {
            key: visual.get(key)
            for key in (
                "foreground_rgb", "background_rgb", "contrast_ratio",
                "required_contrast_ratio", "font_size", "font_weight",
            )
        }
        return {
            "selector": selector,
            "html": _clean_html(element),
            "text": str(getattr(node, "text_content", ""))[:500],
            "tag": str(getattr(node, "tag", "")),
            "attributes": _context_attributes(element),
            "visual": visual_evidence,
            "repair_context": _repair_context(
                rule_id=rule_id, element=element, soup=soup, selector=selector,
                visual=visual_evidence,
            ),
        }
    dom_indices = getattr(graph, "dom_indices", None)
    marker = int(dom_indices[node_index]) if dom_indices is not None else -1
    element = _element_for_marker(soup, marker)
    selector = _normalise_selector(get_dom_path(element)) if element is not None else "html"
    return {
        "selector": selector,
        "html": _clean_html(element),
        "text": element.get_text(" ", strip=True)[:500] if element is not None else "",
        "tag": str(element.name) if element is not None else "",
        "attributes": _context_attributes(element),
        "snapshot_node_id": marker,
        "repair_context": _repair_context(
            rule_id=rule_id, element=element, soup=soup, selector=selector,
        ),
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
                rule_id=rule_id, view=view, node_index=node_index, graph=graph,
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
        "visual_evidence": _visual_evidence_payload(site_dir),
    }


__all__ = ["capture_aligned_page", "run_live_specialists"]
