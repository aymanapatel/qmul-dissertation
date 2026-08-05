"""Freeze the independently annotated final-evaluation universe.

The final test partition is derived only from the adjudicated annotation file:
every retained site must have one adjudicated record for every annotated
criterion. Sites in the governed test partition without complete independent
truth are excluded rather than assigned an imputed pass/fail label. The source
truth file is read-only and its digest is recorded in the resulting artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data import split_hash, validate_split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_final_evaluation_split(
    governed: dict,
    truth: dict,
    *,
    governed_path: Path,
    truth_path: Path,
    exclusion_reason_code: str,
    exclusion_reason: str,
) -> dict:
    validate_split(governed)
    rows = truth.get("pairs", [])
    if not rows:
        raise ValueError("Independent truth contains no site/criterion pairs")

    records: dict[tuple[str, str], dict] = {}
    criteria = set()
    sites = set()
    for row in rows:
        pair = (str(row.get("site_id", "")), str(row.get("criterion_id", "")))
        if not all(pair) or pair in records:
            raise ValueError(f"Independent truth needs unique non-empty pairs: {pair}")
        if row.get("status") not in {"pass", "fail"} or not row.get("adjudicated"):
            raise ValueError(f"Independent truth pair is not finalized: {pair}")
        if len(set(row.get("annotator_ids", []))) < 2:
            raise ValueError(f"Independent truth pair lacks two annotators: {pair}")
        records[pair] = row
        sites.add(pair[0])
        criteria.add(pair[1])

    expected = {(site, criterion) for site in sites for criterion in criteria}
    if set(records) != expected:
        missing = sorted(expected - set(records))
        raise ValueError(f"Independent truth is not a complete site/criterion matrix: {missing[:3]}")

    governed_test = set(governed["test"])
    outside = sites - governed_test
    if outside:
        raise ValueError(f"Annotated sites are outside the governed test partition: {sorted(outside)[:3]}")
    if not exclusion_reason_code.strip() or not exclusion_reason.strip():
        raise ValueError("A documented reason code and explanation are required for exclusions")

    excluded = sorted(governed_test - sites)
    artifact_root = truth_path.parent.parent

    def exclusion_record(site: str) -> dict:
        evidence_paths = [
            f"detection_annotation_packet/evidence/{site}/rendered.png",
            f"detection_annotation_packet/evidence/{site}/resource_audit.json",
        ]
        evidence_sha256 = {
            relative: _sha256(artifact_root / relative)
            for relative in evidence_paths if (artifact_root / relative).is_file()
        }
        return {
            "site_id": site,
            "partition": "test",
            "reason_code": exclusion_reason_code,
            "reason": exclusion_reason,
            "label_imputed": False,
            "evidence_paths": evidence_paths,
            "evidence_sha256": evidence_sha256,
        }

    payload = {
        "schema_version": 1,
        "status": "frozen_independent_manual_evaluation_split",
        "seed": governed.get("seed", 42),
        "train": sorted(governed["train"]),
        "val": sorted(governed["val"]),
        "test": sorted(sites),
        "criteria": sorted(criteria),
        "expected_truth_pair_count": len(expected),
        "source_test_site_count": len(governed_test),
        "retained_test_site_count": len(sites),
        "exclusions": [exclusion_record(site) for site in excluded],
        "provenance": {
            "governed_split": str(governed_path.resolve()),
            "governed_split_sha256": _sha256(governed_path),
            "independent_truth": str(truth_path.resolve()),
            "independent_truth_sha256": _sha256(truth_path),
            "derivation": "Retain governed test sites having a complete Cartesian matrix of dual-annotated, adjudicated manual truth; do not impute excluded labels.",
        },
    }
    validate_split(payload)
    payload["split_hash"] = split_hash(payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--governed-split", type=Path, required=True)
    parser.add_argument("--truth-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclusion-reason-code",
        default="annotation_evidence_unavailable_bot_or_capture_block",
    )
    parser.add_argument(
        "--exclusion-reason",
        default=(
            "Excluded before final evaluation because the independently reviewed evidence was a bot-protection, "
            "challenge, or unusable capture and no defensible pass/fail annotation was retained."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    governed = json.loads(args.governed_split.read_text(encoding="utf-8"))
    truth = json.loads(args.truth_file.read_text(encoding="utf-8"))
    payload = build_final_evaluation_split(
        governed,
        truth,
        governed_path=args.governed_split,
        truth_path=args.truth_file,
        exclusion_reason_code=args.exclusion_reason_code,
        exclusion_reason=args.exclusion_reason,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "train": len(payload["train"]),
        "val": len(payload["val"]),
        "test": len(payload["test"]),
        "criteria": payload["criteria"],
        "truth_pairs": payload["expected_truth_pair_count"],
        "documented_exclusions": len(payload["exclusions"]),
        "split_hash": payload["split_hash"],
    }, indent=2))


if __name__ == "__main__":
    main()
