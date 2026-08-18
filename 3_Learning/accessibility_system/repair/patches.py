"""Atomic application of allow-listed, typed HTML repair operations."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from bs4 import BeautifulSoup, Tag

from .contracts import RepairProposal


ALLOWED_ATTRIBUTES = frozenset({
    "alt", "aria-describedby", "aria-expanded", "aria-label", "aria-labelledby",
    "aria-modal", "aria-pressed", "aria-required", "for", "id", "lang", "role",
    "tabindex", "title",
})
ALLOWED_CSS_PROPERTIES = frozenset({
    "background-color", "color", "min-height", "min-width", "outline", "outline-offset",
})
FORBIDDEN_TARGET_TAGS = frozenset({"embed", "iframe", "object", "script", "style", "template"})
SAFE_ATTRIBUTE_NAME = re.compile(r"^[a-z][a-z0-9:-]*$")
SAFE_IDREFS = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*(?:\s+[A-Za-z][A-Za-z0-9_.:-]*)*$")


class PatchApplicationError(ValueError):
    pass


def _one(soup: BeautifulSoup, selector: str) -> Tag:
    try:
        matches = soup.select(selector)
    except Exception as exc:
        raise PatchApplicationError(f"Invalid selector {selector!r}: {exc}") from exc
    if len(matches) != 1:
        raise PatchApplicationError(f"Selector {selector!r} resolved {len(matches)} nodes; exactly one is required")
    if matches[0].name in FORBIDDEN_TARGET_TAGS:
        raise PatchApplicationError(f"Target element {matches[0].name!r} cannot be modified")
    return matches[0]


def _safe_text(value: str, *, field: str) -> str:
    if "<" in value or ">" in value or "javascript:" in value.lower():
        raise PatchApplicationError(f"Unsafe markup or script-like content in {field}")
    return value


def _style_map(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for declaration in value.split(";"):
        if not declaration.strip() or ":" not in declaration:
            continue
        name, item = declaration.split(":", 1)
        result[name.strip().lower()] = item.strip()
    return result


def _serialize_style(values: dict[str, str]) -> str:
    return "; ".join(f"{key}: {values[key]}" for key in sorted(values))


def apply_typed_patch(source_html: str, proposal: RepairProposal) -> tuple[str, dict[str, Any]]:
    """Apply all operations in memory or fail atomically; source_html is never modified."""
    if proposal.decision == "leave_unchanged":
        return source_html, {"applied": False, "operation_count": 0, "operations": []}
    soup = BeautifulSoup(source_html, "lxml")
    evidence: list[dict[str, Any]] = []
    for index, operation in enumerate(proposal.operations):
        target = _one(soup, operation.selector)
        before = str(target)
        name = operation.operation
        if name in {"set_attribute", "remove_attribute"}:
            attribute = str(operation.attribute_name).lower()
            if not SAFE_ATTRIBUTE_NAME.fullmatch(attribute) or attribute not in ALLOWED_ATTRIBUTES:
                raise PatchApplicationError(f"Attribute {attribute!r} is not allow-listed")
            if name == "remove_attribute":
                target.attrs.pop(attribute, None)
            else:
                value = _safe_text(str(operation.new_value), field=attribute).strip()
                if attribute in {"aria-describedby", "aria-labelledby", "for", "id"} and value and not SAFE_IDREFS.fullmatch(value):
                    raise PatchApplicationError(f"Invalid ID/IDREF value for {attribute}")
                target[attribute] = value
        elif name == "replace_text":
            target.clear()
            target.append(_safe_text(str(operation.new_value), field="text"))
        elif name == "insert_label_before":
            text = _safe_text(str(operation.new_value), field="label text").strip()
            if target.name not in {"input", "select", "textarea"}:
                raise PatchApplicationError("insert_label_before requires a form control")
            control_id = str(target.get("id", "")).strip()
            if not control_id:
                control_id = "phase9-" + hashlib.sha256(f"{proposal.proposal_id}|{index}".encode()).hexdigest()[:12]
                target["id"] = control_id
            label = soup.new_tag("label", attrs={"for": control_id})
            label.string = text
            target.insert_before(label)
        elif name == "set_style_property":
            prop = str(operation.css_property).lower()
            value = _safe_text(str(operation.new_value), field=prop).strip()
            if prop not in ALLOWED_CSS_PROPERTIES:
                raise PatchApplicationError(f"CSS property {prop!r} is not allow-listed")
            if any(token in value.lower() for token in ("url(", "expression(", "javascript:")):
                raise PatchApplicationError(f"Unsafe CSS value for {prop}")
            styles = _style_map(str(target.get("style", "")))
            styles[prop] = value
            target["style"] = _serialize_style(styles)
        elif name == "remove_meta_viewport_restriction":
            if target.name != "meta" or str(target.get("name", "")).lower() != "viewport":
                raise PatchApplicationError("Viewport repair must target meta[name=viewport]")
            parts = [part.strip() for part in str(target.get("content", "")).split(",")]
            parts = [part for part in parts if not re.match(r"^(?:user-scalable\s*=\s*no|maximum-scale\s*=\s*(?:0|1)(?:\.0+)?)$", part, re.I)]
            target["content"] = ", ".join(part for part in parts if part)
        else:  # pragma: no cover - schema prevents this
            raise PatchApplicationError(f"Unsupported operation: {name}")
        evidence.append({
            "index": index,
            "operation": name,
            "selector": operation.selector,
            "before": before[:3000],
            "after": str(target)[:3000],
        })
    return str(soup), {"applied": bool(evidence), "operation_count": len(evidence), "operations": evidence}
