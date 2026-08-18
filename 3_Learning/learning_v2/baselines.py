"""Axe adapter and deterministic HTML baselines using the common Finding schema."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup, Tag

from learning_v1.src.wcag_rules import RULE_BY_ID

from .contracts import Finding, NodeIdentity
from .evidence import _css_path, complete_site_dirs


LANGUAGE_TAG = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
FORM_CONTROL_TYPES_WITHOUT_LABEL = {"hidden", "submit", "reset", "button", "image"}


def _criteria(rule_id: str) -> list[str]:
    spec = RULE_BY_ID.get(rule_id)
    return list(spec.wcag_ids) if spec else []


def _identity(tag: Tag | None, node_id: int = -1) -> NodeIdentity | None:
    return NodeIdentity(node_id, _css_path(tag), tag.name) if tag else None


def _finding(site_id: str, rule_id: str, detector: str, tag: Tag | None, evidence: dict, index: int) -> Finding:
    raw = f"{site_id}|{rule_id}|{detector}|{evidence}|{index}"
    return Finding(
        finding_id=hashlib.sha256(raw.encode()).hexdigest()[:20], site_id=site_id,
        rule_id=rule_id, criterion_ids=_criteria(rule_id), detector=detector,
        node=_identity(tag, index), evidence=evidence,
    )


def axe_findings(site_dir: Path) -> list[Finding]:
    report = json.loads((site_dir / "page-0_home.json").read_text(encoding="utf-8"))
    findings = []
    for violation in report.get("violations", []):
        rule_id = str(violation.get("id", "unknown"))
        for index, node in enumerate(violation.get("nodes", [])):
            target = node.get("target", [])
            selector = target[0] if target and isinstance(target[0], str) else ""
            identity = NodeIdentity(index, selector, "unknown") if selector else None
            findings.append(Finding(
                finding_id=hashlib.sha256(f"{site_dir.name}|axe|{rule_id}|{target}|{index}".encode()).hexdigest()[:20],
                site_id=site_dir.name, rule_id=rule_id, criterion_ids=_criteria(rule_id), detector="axe-core",
                node=identity, impact=violation.get("impact"),
                evidence={"target": target, "html": node.get("html", ""), "failure_summary": node.get("failureSummary", "")},
            ))
    return findings


def _has_name(tag: Tag) -> bool:
    return bool(tag.get_text(" ", strip=True) or tag.get("aria-label") or tag.get("aria-labelledby") or tag.get("title"))


def _labelled(control: Tag, soup: BeautifulSoup) -> bool:
    if control.get("aria-label") or control.get("aria-labelledby") or control.find_parent("label"):
        return True
    control_id = control.get("id")
    return bool(control_id and soup.find("label", attrs={"for": control_id}))


def deterministic_findings(html_path: Path, *, site_id: str | None = None) -> list[Finding]:
    site_id = site_id or html_path.parent.name
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="replace"), "lxml")
    findings: list[Finding] = []; index = 0

    def add(rule_id: str, tag: Tag | None, **evidence):
        nonlocal index
        findings.append(_finding(site_id, rule_id, "deterministic-html", tag, evidence, index)); index += 1

    html = soup.find("html")
    if not soup.title or not soup.title.get_text(strip=True): add("document-title", soup.head if soup.head else html, reason="missing_or_empty_title")
    language = str(html.get("lang", "")).strip() if html else ""
    if not language: add("html-has-lang", html, reason="missing_lang")
    elif not LANGUAGE_TAG.match(language): add("html-lang-valid", html, reason="invalid_language_tag", value=language)
    for tag in soup.find_all("img"):
        if not tag.has_attr("alt"): add("image-alt", tag, reason="missing_alt")
    for tag in soup.find_all("button"):
        if not _has_name(tag): add("button-name", tag, reason="missing_accessible_name")
    for tag in soup.find_all("a", href=True):
        if not _has_name(tag): add("link-name", tag, reason="missing_accessible_name")
    for tag in soup.find_all(["input", "textarea", "select"]):
        if tag.name == "input" and str(tag.get("type", "text")).lower() in FORM_CONTROL_TYPES_WITHOUT_LABEL: continue
        if not _labelled(tag, soup): add("select-name" if tag.name == "select" else "label", tag, reason="missing_programmatic_label")
    viewport = soup.find("meta", attrs={"name": re.compile(r"^viewport$", re.I)})
    if viewport:
        content = str(viewport.get("content", "")).lower().replace(" ", "")
        if "user-scalable=no" in content or re.search(r"maximum-scale=(?:0|1)(?:\.0+)?(?:,|$)", content):
            add("meta-viewport", viewport, reason="zoom_restricted", content=content)
    interactive = "a[href],button,input:not([type=hidden]),select,textarea,summary,[tabindex],[role=button],[role=link]"
    for parent in soup.select(interactive):
        descendant = parent.select_one(interactive)
        if descendant is not None: add("nested-interactive", parent, reason="interactive_descendant", descendant=_css_path(descendant))
    return findings


def compare_site_rule_sets(predicted: Iterable[Finding], truth: Iterable[Finding]) -> dict:
    predicted_pairs = {(finding.site_id, finding.rule_id) for finding in predicted}
    truth_pairs = {(finding.site_id, finding.rule_id) for finding in truth}
    tp = len(predicted_pairs & truth_pairs); fp = len(predicted_pairs - truth_pairs); fn = len(truth_pairs - predicted_pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "unit": "site_rule_pair", "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "predicted_pairs": len(predicted_pairs), "actual_pairs": len(truth_pairs),
    }


def run_corpus_baseline(corpus_dir: Path, site_ids: set[str] | None = None) -> dict:
    predicted: list[Finding] = []; truth: list[Finding] = []; failures = []
    for site_dir in complete_site_dirs(corpus_dir):
        if site_ids is not None and site_dir.name not in site_ids: continue
        try:
            predicted.extend(deterministic_findings(site_dir / "0.html", site_id=site_dir.name))
            truth.extend(axe_findings(site_dir))
        except Exception as exc:
            failures.append({"site_id": site_dir.name, "error": f"{type(exc).__name__}: {exc}"})
    metrics = compare_site_rule_sets(predicted, truth)
    return {
        "schema_version": 1, "site_count": len({finding.site_id for finding in truth}),
        "metrics": metrics,
        "predicted_rule_counts": dict(sorted(Counter(f.rule_id for f in predicted).items())),
        "axe_rule_counts": dict(sorted(Counter(f.rule_id for f in truth).items())),
        "failure_count": len(failures), "failures": failures,
    }
