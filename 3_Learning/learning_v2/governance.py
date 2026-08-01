"""Corpus inventory, duplicate grouping, governed splits, and support diagnostics."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .data import split_hash, validate_split
from .evidence import complete_site_dirs, sha256_file


def _axe_counts(path: Path) -> tuple[dict[str, int], str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(violation.get("id", "unknown")): len(violation.get("nodes", []))
        for violation in report.get("violations", [])
    }, str(report.get("url", ""))


def inventory_corpus(corpus_dir: Path) -> dict[str, Any]:
    sites = []
    html_groups: dict[str, list[str]] = defaultdict(list)
    rule_site_support = Counter(); rule_node_support = Counter()
    for directory in complete_site_dirs(corpus_dir):
        html_hash = sha256_file(directory / "0.html")
        axe_hash = sha256_file(directory / "page-0_home.json")
        counts, url = _axe_counts(directory / "page-0_home.json")
        html_groups[html_hash].append(directory.name)
        for rule_id, count in counts.items():
            rule_site_support[rule_id] += 1; rule_node_support[rule_id] += count
        sites.append({
            "site_id": directory.name, "url": url, "html_sha256": html_hash, "axe_sha256": axe_hash,
            "html_bytes": (directory / "0.html").stat().st_size,
            "rule_node_counts": counts, "rule_ids": sorted(counts),
        })
    return {
        "schema_version": 1, "corpus_dir": str(corpus_dir), "complete_site_count": len(sites),
        "site_count_with_any_violation": sum(bool(site["rule_ids"]) for site in sites),
        "sites": sites,
        "duplicate_html_groups": [sorted(group) for group in html_groups.values() if len(group) > 1],
        "unique_html_count": len(html_groups),
        "rule_site_support": dict(sorted(rule_site_support.items())),
        "rule_node_support": dict(sorted(rule_node_support.items())),
    }


def create_grouped_multilabel_split(inventory: dict[str, Any], *, seed: int = 42) -> dict[str, Any]:
    site_by_id = {site["site_id"]: site for site in inventory["sites"]}
    group_by_hash: dict[str, list[str]] = defaultdict(list)
    for site in inventory["sites"]:
        group_by_hash[site["html_sha256"]].append(site["site_id"])
    groups = []
    frequency = Counter()
    for html_hash, site_ids in group_by_hash.items():
        labels = set().union(*(site_by_id[site_id]["rule_ids"] for site_id in site_ids))
        groups.append({"hash": html_hash, "sites": sorted(site_ids), "labels": labels})
        frequency.update(labels)
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda group: sum(1 / max(1, frequency[label]) for label in group["labels"]), reverse=True)
    names = ("train", "val", "test"); ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    target_size = {name: len(site_by_id) * ratios[name] for name in names}
    assigned = {name: [] for name in names}; label_counts = {name: Counter() for name in names}
    for group in groups:
        def score(name: str) -> tuple[float, float, str]:
            size_pressure = (len(assigned[name]) + len(group["sites"])) / max(1.0, target_size[name])
            label_pressure = sum(label_counts[name][label] / max(1.0, frequency[label] * ratios[name]) for label in group["labels"])
            return size_pressure + label_pressure / max(1, len(group["labels"])), size_pressure, name
        chosen = min(names, key=score)
        assigned[chosen].extend(group["sites"]); label_counts[chosen].update(group["labels"])
    split = {"seed": seed, **{name: sorted(assigned[name]) for name in names}}
    validate_split(split, available_sites=set(site_by_id))
    split["split_hash"] = split_hash(split)
    split["support"] = {
        name: {
            "site_count": len(split[name]),
            "sites_with_any_violation": sum(bool(site_by_id[site]["rule_ids"]) for site in split[name]),
            "rule_site_support": dict(sorted(Counter(rule for site in split[name] for rule in site_by_id[site]["rule_ids"]).items())),
        } for name in names
    }
    return split


def write_governance_artifacts(corpus_dir: Path, output_dir: Path, *, seed: int = 42) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = inventory_corpus(corpus_dir); split = create_grouped_multilabel_split(inventory, seed=seed)
    inventory_path = output_dir / "corpus_inventory.json"; split_path = output_dir / "governed_split.json"
    inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return inventory_path, split_path

