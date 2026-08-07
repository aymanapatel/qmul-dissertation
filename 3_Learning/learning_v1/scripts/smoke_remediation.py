#!/usr/bin/env python3
"""
Smoke-test contextual remediation from a GNN prediction report.

This script treats the GNN as a classifier, 

Input: HTML page and the prediction from GNN
Output: Suggestions as JSON. Suggestion and not final fix

Example:
    ./.venv/bin/python scripts/smoke_remediation.py \
      --html ../../2_Data/browser-use/outputs/axe-core/www.furaffinity.net/0.html \
      --prediction ../reports/gnn_batch_100/predictions/www.furaffinity.net.json \
      --output ../reports/remediation_smoke_furaffinity.json \
      --max-candidates 10
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Tag


SUPPORTED_RULES = {
    "image-alt",
    "svg-img-alt",
    "role-img-alt",
    "input-image-alt",
    "link-name",
    "button-name",
    "input-button-name",
    "aria-command-name",
    "label",
    "select-name",
}

DECORATIVE_HINTS = {
    "icon",
    "sprite",
    "decorative",
    "divider",
    "spacer",
    "bg",
    "background",
    "chevron",
    "arrow",
}


def clean_text(value: str | None, limit: int = 220) -> str:
    if not value:
        return ""
    text = re.sub(r"\s+", " ", value).strip()
    return text[:limit].strip()


def attr_text(tag: Tag | None, *names: str) -> str:
    if not tag:
        return ""
    for name in names:
        value = tag.get(name)
        if isinstance(value, list):
            value = " ".join(str(item) for item in value)
        value = clean_text(str(value)) if value else ""
        if value:
            return value
    return ""


def compact_html(tag: Tag | None, limit: int = 500) -> str:
    if not tag:
        return ""
    html = clean_text(str(tag), limit=limit)
    return html


def class_tokens(tag: Tag | None) -> set[str]:
    if not tag:
        return set()
    classes = tag.get("class", [])
    if isinstance(classes, str):
        classes = classes.split()
    return {str(item).lower() for item in classes}


def filename_label(src: str) -> str:
    if not src:
        return ""
    path = urlparse(src).path
    stem = Path(path).stem
    stem = re.sub(r"[_-]+", " ", stem)
    stem = re.sub(r"\b(icon|img|image|photo|pic|logo|final|copy)\b", "", stem, flags=re.I)
    stem = clean_text(stem)
    return stem


def page_label(context: dict[str, Any]) -> str:
    label = context.get("page_h1") or context.get("page_title") or ""
    if "--" in label:
        label = label.split("--")[-1]
    return clean_text(label)


def accessible_text(tag: Tag | None) -> str:
    if not tag:
        return ""
    explicit = attr_text(tag, "aria-label", "title", "alt", "value", "placeholder")
    if explicit:
        return explicit
    labelledby = attr_text(tag, "aria-labelledby")
    if labelledby:
        root = tag
        while root.parent and isinstance(root.parent, Tag):
            root = root.parent
        parts = []
        for ident in labelledby.split():
            labelled = root.find(id=ident)
            if isinstance(labelled, Tag):
                parts.append(clean_text(labelled.get_text(" ", strip=True)))
        if parts:
            return clean_text(" ".join(parts))
    return clean_text(tag.get_text(" ", strip=True))


def nearest(tag: Tag | None, names: set[str]) -> Tag | None:
    current = tag
    while current and isinstance(current, Tag):
        if current.name in names:
            return current
        current = current.parent if isinstance(current.parent, Tag) else None
    return None


def nearest_heading(tag: Tag | None) -> str:
    current = tag
    while current and isinstance(current, Tag):
        previous = current.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
        if previous:
            text = clean_text(previous.get_text(" ", strip=True))
            if text:
                return text
        current = current.parent if isinstance(current.parent, Tag) else None
    return ""


def nearby_text(tag: Tag | None, limit_items: int = 6) -> list[str]:
    if not tag:
        return []
    values: list[str] = []
    for source in [
        tag.parent if isinstance(tag.parent, Tag) else None,
        tag.find_previous(["h1", "h2", "h3", "h4", "h5", "h6", "p", "span", "a", "button"]),
        tag.find_next(["p", "span", "a", "button"]),
    ]:
        if isinstance(source, Tag):
            text = clean_text(source.get_text(" ", strip=True))
            if text and text not in values:
                values.append(text)
        if len(values) >= limit_items:
            break
    return values[:limit_items]


def find_by_prediction(soup: BeautifulSoup, prediction: dict[str, Any]) -> Tag | None:
    dom_path = prediction.get("dom_path") or ""
    if dom_path:
        try:
            found = soup.select_one(dom_path)
            if isinstance(found, Tag):
                return found
        except Exception:
            pass

    attrs = prediction.get("attributes") or {}
    tag_name = prediction.get("tag")
    candidates = soup.find_all(tag_name) if tag_name else soup.find_all(True)

    best: tuple[int, Tag] | None = None
    for tag in candidates:
        score = 0
        for key, value in attrs.items():
            if key not in tag.attrs:
                continue
            tag_value = tag.get(key)
            if isinstance(value, list):
                value = " ".join(str(item) for item in value)
            if isinstance(tag_value, list):
                tag_value = " ".join(str(item) for item in tag_value)
            if str(tag_value) == str(value):
                score += 4
            elif str(value) and str(value) in str(tag_value):
                score += 2
        if score and (best is None or score > best[0]):
            best = (score, tag)
    return best[1] if best else None


def page_context(soup: BeautifulSoup) -> dict[str, Any]:
    title = clean_text(soup.title.get_text(" ", strip=True) if soup.title else "")
    description = ""
    meta = soup.find("meta", attrs={"name": "description"})
    if isinstance(meta, Tag):
        description = attr_text(meta, "content")
    h1 = soup.find("h1")
    return {
        "title": title,
        "description": description,
        "h1": clean_text(h1.get_text(" ", strip=True)) if isinstance(h1, Tag) else "",
    }


def classify_image_purpose(tag: Tag | None, link: Tag | None, context: dict[str, Any]) -> str:
    if link or context.get("parent_button_tag") or context.get("parent_label_tag"):
        return "functional"
    if not tag:
        return "unknown"
    tokens = class_tokens(tag)
    src = attr_text(tag, "src")
    if tokens & DECORATIVE_HINTS or any(hint in src.lower() for hint in DECORATIVE_HINTS):
        if context.get("nearby_text") or context.get("section_heading"):
            return "possibly_decorative"
        return "decorative"
    return "informative"


def propose_alt(tag: Tag | None, context: dict[str, Any]) -> tuple[str, str, str]:
    link = context.get("parent_link_tag")
    purpose = classify_image_purpose(tag, link, context)

    if purpose == "functional":
        control_type = "link" if link else "button" if context.get("parent_button_tag") else "label/control"
        label = (
            accessible_text(link)
            or accessible_text(context.get("parent_button_tag"))
            or accessible_text(context.get("parent_label_tag"))
        )
        href = attr_text(link, "href")
        src = attr_text(tag, "src")
        tokens = class_tokens(tag)
        if not label:
            filename = filename_label(src)
            if "logo" in tokens or "logo" in src.lower():
                label = page_label(context) or filename
            elif "menu" in tokens or "burger" in src.lower() or "menu" in filename.lower():
                label = "Menu"
            elif href in {"/", "#", ""}:
                label = page_label(context) or filename
            else:
                label = context.get("section_heading") or filename or context.get("page_title")
        label = clean_text(label or "Link destination")
        destination = f" ({href})" if href else ""
        return label, "functional", f"Image is used as a functional {control_type}; alt should describe the action or destination{destination}."

    if purpose == "decorative":
        return "", "decorative", "Image appears decorative from filename/class hints and has no strong nearby content dependency."

    explicit = attr_text(tag, "title", "aria-label")
    if explicit:
        return explicit, purpose, "Existing title/ARIA text is the strongest available contextual label."

    nearby = context.get("nearby_text") or []
    heading = context.get("section_heading") or context.get("page_h1") or context.get("page_title")
    filename = filename_label(attr_text(tag, "src"))

    parts = []
    if filename:
        parts.append(filename)
    if heading and heading.lower() not in " ".join(parts).lower():
        parts.append(heading)
    if nearby:
        text = nearby[0]
        if text.lower() not in " ".join(parts).lower():
            parts.append(text)

    suggestion = clean_text(" - ".join(parts), limit=140)
    if not suggestion:
        suggestion = "Describe the image's purpose in this section"
    return suggestion, purpose, "Alt text is derived from filename, nearest heading, and surrounding text; human review is required."


def build_context(soup: BeautifulSoup, tag: Tag | None, page: dict[str, Any]) -> dict[str, Any]:
    link = nearest(tag, {"a"})
    button = nearest(tag, {"button"})
    label = nearest(tag, {"label"})
    context = {
        "page_title": page.get("title", ""),
        "page_description": page.get("description", ""),
        "page_h1": page.get("h1", ""),
        "section_heading": nearest_heading(tag),
        "nearby_text": nearby_text(tag),
        "parent_link_href": attr_text(link, "href"),
        "parent_link_text": accessible_text(link),
        "parent_button_text": accessible_text(button),
        "parent_label_text": accessible_text(label),
        "current_accessible_name": accessible_text(tag),
        "node_html": compact_html(tag),
        "_parent_link_tag": link,
    }
    context["parent_link_tag"] = link
    context["parent_button_tag"] = button
    context["parent_label_tag"] = label
    return context


def patch_for_rule(rule_id: str, tag: Tag | None, context: dict[str, Any]) -> dict[str, Any]:
    if rule_id in {"image-alt", "svg-img-alt", "role-img-alt", "input-image-alt"}:
        alt, purpose, rationale = propose_alt(tag, context)
        attr = "alt" if (tag and tag.name in {"img", "input"}) else "aria-label"
        return {
            "kind": "contextual_alt",
            "image_purpose": purpose,
            "attribute": attr,
            "suggested_value": alt,
            "suggested_patch": f'Set {attr}="{alt}"' if alt else f'Set {attr}=""',
            "rationale": rationale,
            "requires_human_review": True,
        }

    if rule_id in {"link-name", "aria-command-name"}:
        label = (
            context.get("parent_link_text")
            or context.get("current_accessible_name")
            or context.get("section_heading")
            or context.get("page_title")
            or "Describe link destination"
        )
        return {
            "kind": "accessible_name",
            "attribute": "aria-label",
            "suggested_value": clean_text(label, 120),
            "suggested_patch": f'Add aria-label="{clean_text(label, 120)}" or visible text to the link/control.',
            "rationale": "Name is grounded in existing link text, section heading, and page title.",
            "requires_human_review": True,
        }

    if rule_id in {"button-name", "input-button-name"}:
        label = (
            context.get("parent_button_text")
            or context.get("current_accessible_name")
            or context.get("section_heading")
            or "Describe button action"
        )
        return {
            "kind": "accessible_name",
            "attribute": "aria-label",
            "suggested_value": clean_text(label, 120),
            "suggested_patch": f'Add aria-label="{clean_text(label, 120)}" or visible text to the button.',
            "rationale": "Button name should describe the action in this page section.",
            "requires_human_review": True,
        }

    if rule_id in {"label", "select-name"}:
        label = (
            context.get("parent_label_text")
            or context.get("section_heading")
            or context.get("nearby_text", [""])[0]
            or "Describe this field"
        )
        return {
            "kind": "form_label",
            "attribute": "label/aria-label",
            "suggested_value": clean_text(label, 120),
            "suggested_patch": f'Associate a visible <label> or add aria-label="{clean_text(label, 120)}".',
            "rationale": "Field label is inferred from nearby form/section context.",
            "requires_human_review": True,
        }

    return {
        "kind": "unsupported",
        "suggested_patch": "No smoke remediation rule implemented.",
        "rationale": f"Rule {rule_id} is outside this smoke script's scoped rules.",
        "requires_human_review": True,
    }


def candidate_rule(prediction: dict[str, Any]) -> str:
    if prediction.get("axe_rule_id"):
        return str(prediction["axe_rule_id"])
    rules = prediction.get("predicted_rules") or []
    if rules:
        return str(rules[0].get("rule_id", ""))
    return ""


def load_candidates(prediction_report: dict[str, Any], include_candidates: bool) -> list[dict[str, Any]]:
    candidates = list(prediction_report.get("predicted_violations") or [])
    if include_candidates:
        candidates.extend(prediction_report.get("candidate_warnings") or [])
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test contextual remediation from GNN predictions.")
    parser.add_argument("--html", required=True, type=Path, help="Saved rendered HTML file.")
    parser.add_argument("--prediction", required=True, type=Path, help="GNN prediction JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Output remediation JSON.")
    parser.add_argument("--max-candidates", type=int, default=20, help="Maximum candidates to include.")
    parser.add_argument("--include-candidate-warnings", action="store_true", help="Also process non-confirmed candidate warnings.")
    parser.add_argument("--rules", nargs="*", default=sorted(SUPPORTED_RULES), help="Rules to remediate.")
    args = parser.parse_args()

    soup = BeautifulSoup(args.html.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    report = json.loads(args.prediction.read_text(encoding="utf-8"))
    page = page_context(soup)
    allowed_rules = set(args.rules)

    remediations = []
    for prediction in load_candidates(report, args.include_candidate_warnings):
        rule_id = candidate_rule(prediction)
        if rule_id not in allowed_rules:
            continue
        tag = find_by_prediction(soup, prediction)
        context = build_context(soup, tag, page)
        parent_link_tag = context.pop("parent_link_tag", None)
        parent_button_tag = context.pop("parent_button_tag", None)
        parent_label_tag = context.pop("parent_label_tag", None)
        context["_parent_link_tag"] = None
        context["_parent_button_tag"] = None
        context["_parent_label_tag"] = None
        patch_context = {
            **context,
            "parent_link_tag": parent_link_tag,
            "parent_button_tag": parent_button_tag,
            "parent_label_tag": parent_label_tag,
        }
        remediation = patch_for_rule(rule_id, tag, patch_context)
        remediations.append(
            {
                "rule_id": rule_id,
                "source_view": prediction.get("source_view"),
                "node_id": prediction.get("node_id"),
                "probability": prediction.get("probability"),
                "rule_probability": prediction.get("rule_probability"),
                "dom_path": prediction.get("dom_path"),
                "matched_node": tag is not None,
                "current_html": compact_html(tag),
                "context": context,
                "remediation": remediation,
            }
        )
        if len(remediations) >= args.max_candidates:
            break

    output = {
        "html_path": str(args.html),
        "prediction_path": str(args.prediction),
        "page": page,
        "summary": {
            "input_predicted_violations": len(report.get("predicted_violations") or []),
            "input_candidate_warnings": len(report.get("candidate_warnings") or []),
            "remediation_count": len(remediations),
            "rules": sorted(allowed_rules),
            "mode": "predicted_plus_candidates" if args.include_candidate_warnings else "predicted_only",
        },
        "remediations": remediations,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {len(remediations)} remediation suggestions to {args.output}")


if __name__ == "__main__":
    main()
