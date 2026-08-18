"""Create and finalize blinded human-rating packets for matched repair runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any


RATING_FIELDS = ("contextual_correctness", "safety", "helpfulness")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected CONDITION=REPORT, received: {value}")
        condition, path = value.split("=", 1)
        if not condition or condition in result:
            raise ValueError("Repair conditions must be non-empty and unique")
        result[condition] = Path(path)
    return result


def _candidate_id(condition: str, query_id: str, seed: int) -> str:
    return "repair-" + hashlib.sha256(f"{seed}|{condition}|{query_id}".encode()).hexdigest()[:20]


def _copy_if_present(source_value: str | None, destination: Path) -> str | None:
    if not source_value:
        return None
    source = Path(source_value)
    if not source.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def create_packet(args: argparse.Namespace) -> dict[str, Any]:
    run_paths = _runs(args.run)
    inputs = json.loads(args.generator_inputs.read_text(encoding="utf-8"))
    input_map = {(str(item["condition"]), str(item["query_id"])): item for item in inputs}
    inputs_by_query: dict[str, list[dict[str, Any]]] = {}
    for item in inputs:
        inputs_by_query.setdefault(str(item["query_id"]), []).append(item)
    identities = []
    rater_cases = []
    for condition, report_path in sorted(run_paths.items()):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for attempt in report.get("attempts", []):
            query_id = str(attempt["query_id"])
            generator_input = input_map.get((condition, query_id))
            # The deterministic control reuses the same frozen finding universe
            # but is intentionally absent from LLM generator inputs.  Restore
            # its finding by query ID without exposing condition to raters.
            if generator_input is None and condition == "deterministic_template":
                candidates = inputs_by_query.get(query_id, [])
                no_rag = [item for item in candidates if item.get("condition") == "no_rag"]
                generator_input = no_rag[0] if len(no_rag) == 1 else None
            if generator_input is None:
                raise ValueError(f"No generator input for {condition}/{query_id}")
            candidate_id = _candidate_id(condition, query_id, args.seed)
            generation = attempt.get("generation", {})
            proposal_path = Path(generation["proposal_path"]) if generation.get("proposal_path") else None
            proposal = json.loads(proposal_path.read_text(encoding="utf-8")) if proposal_path and proposal_path.is_file() else None
            if proposal:
                proposal = {key: value for key, value in proposal.items() if key not in {
                    "proposal_id", "query_id", "finding_id", "cited_record_ids", "confidence",
                }}
            evidence_dir = args.output_dir / "evidence" / candidate_id
            validation = attempt.get("validation", {})
            artifact_paths = validation.get("artifact_paths", {})
            copied = {
                "before_html": _copy_if_present(artifact_paths.get("before_html"), evidence_dir / "before.html"),
                "after_html": _copy_if_present(artifact_paths.get("after_html"), evidence_dir / "after.html"),
                "before_screenshot": _copy_if_present(artifact_paths.get("before_screenshot"), evidence_dir / "before.png"),
                "after_screenshot": _copy_if_present(artifact_paths.get("after_screenshot"), evidence_dir / "after.png"),
            }
            copied = {key: str(Path(value).relative_to(args.output_dir)) if value else None for key, value in copied.items()}
            identities.append({"candidate_id": candidate_id, "condition": condition, "query_id": query_id})
            finding = generator_input.get("original_finding", {})
            rater_cases.append({
                "candidate_id": candidate_id,
                "criterion_id": finding.get("criterion_id"),
                "rule_id": finding.get("rule_id"),
                "original_finding": {
                    "status": finding.get("status"),
                    "evidence": finding.get("evidence", {}),
                },
                "repair_generated": proposal is not None,
                "proposed_repair": proposal,
                "evidence_paths": copied,
                "contextual_correctness": None,
                "safety": None,
                "helpfulness": None,
                "acceptable": None,
                "notes": "",
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coordinator = args.output_dir / "coordinator"; raters = args.output_dir / "rater_packets"
    coordinator.mkdir(exist_ok=True); raters.mkdir(exist_ok=True)
    (coordinator / "identity_map.json").write_text(json.dumps({"schema_version": 1, "identities": identities}, indent=2), encoding="utf-8")
    for index in (1, 2):
        rows = [dict(item) for item in rater_cases]
        random.Random(args.seed + index).shuffle(rows)
        (raters / f"rater_{index}.json").write_text(json.dumps({
            "schema_version": 1, "rater_id": f"rater-{index}",
            "blinded_to": ["retrieval condition", "query identity", "automatic validation outcome", "oracle result"],
            "scale": "1=poor/unsafe/unhelpful, 5=excellent/safe/helpful",
            "ratings": rows,
        }, indent=2), encoding="utf-8")
    (raters / "adjudicator.json").write_text(json.dumps({
        "schema_version": 1, "rater_id": "adjudicator",
        "instructions": "Complete only candidates where rater-1 and rater-2 disagree on acceptable.",
        "ratings": [{**item, "notes": "Complete only if requested after independent ratings are frozen."} for item in rater_cases],
    }, indent=2), encoding="utf-8")
    instructions = args.output_dir / "RATING_INSTRUCTIONS.md"
    instructions.write_text(
        "# Blinded repair rating instructions\n\n"
        "Raters work independently and must not inspect `coordinator/`, run reports, citations, or each other's sheets. Inspect the original finding, typed proposal, and before/after evidence. Score contextual correctness, safety, and helpfulness from 1 to 5 and set `acceptable` to true or false. A syntactic detector pass is not sufficient for acceptability. Freeze both sheets before adjudication. Run the `finalize` command; complete adjudicator entries only for acceptable disagreements, then rerun it.\n",
        encoding="utf-8",
    )
    artifacts = [path for path in args.output_dir.rglob("*") if path.is_file() and path.name != "packet_manifest.json"]
    manifest = {
        "schema_version": 1, "status": "blinded_repair_rating_packet",
        "condition_count": len(run_paths), "candidate_count": len(identities),
        "generator_inputs": str(args.generator_inputs.resolve()), "generator_inputs_sha256": _sha256(args.generator_inputs),
        "run_reports": {condition: {"path": str(path.resolve()), "sha256": _sha256(path)} for condition, path in run_paths.items()},
        "blinding": {"condition_hidden_from_raters": True, "validation_outcome_hidden_from_raters": True},
        "artifacts": {str(path.relative_to(args.output_dir)): _sha256(path) for path in sorted(artifacts)},
    }
    (args.output_dir / "packet_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _completed(row: dict[str, Any]) -> bool:
    return (
        all(isinstance(row.get(field), (int, float)) and 1 <= row[field] <= 5 for field in RATING_FIELDS)
        and isinstance(row.get("acceptable"), bool)
    )


def finalize_packet(args: argparse.Namespace) -> dict[str, Any]:
    packet = args.packet_dir
    identities = {
        item["candidate_id"]: item
        for item in json.loads((packet / "coordinator/identity_map.json").read_text(encoding="utf-8"))["identities"]
    }
    rater_rows = {}
    output = []
    for index in (1, 2):
        payload = json.loads((packet / f"rater_packets/rater_{index}.json").read_text(encoding="utf-8"))
        rows = {item["candidate_id"]: item for item in payload["ratings"]}
        if set(rows) != set(identities) or not all(_completed(item) for item in rows.values()):
            raise ValueError(f"rater-{index} sheet is incomplete or does not match the frozen candidate universe")
        rater_rows[index] = rows
        for candidate_id, row in rows.items():
            identity = identities[candidate_id]
            output.append({
                "condition": identity["condition"], "query_id": identity["query_id"], "rater_id": f"rater-{index}",
                **{field: row[field] for field in RATING_FIELDS}, "acceptable": row["acceptable"], "notes": row.get("notes", ""),
            })
    disagreements = [candidate_id for candidate_id in identities if rater_rows[1][candidate_id]["acceptable"] != rater_rows[2][candidate_id]["acceptable"]]
    if disagreements:
        adjudicator = {
            item["candidate_id"]: item
            for item in json.loads((packet / "rater_packets/adjudicator.json").read_text(encoding="utf-8"))["ratings"]
        }
        missing = [candidate_id for candidate_id in disagreements if candidate_id not in adjudicator or not _completed(adjudicator[candidate_id])]
        if missing:
            raise ValueError(f"Adjudicator sheet is incomplete for {len(missing)} acceptable disagreements: {missing[:3]}")
        for candidate_id in disagreements:
            row = adjudicator[candidate_id]; identity = identities[candidate_id]
            output.append({
                "condition": identity["condition"], "query_id": identity["query_id"], "rater_id": "adjudicator",
                **{field: row[field] for field in RATING_FIELDS}, "acceptable": row["acceptable"], "notes": row.get("notes", ""),
            })
    result = {
        "schema_version": 1, "blinded": True, "adjudicated": True,
        "candidate_count": len(identities), "acceptable_disagreement_count": len(disagreements), "ratings": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--run", action="append", required=True, help="CONDITION=phase_9_report.json")
    create.add_argument("--generator-inputs", type=Path, required=True)
    create.add_argument("--output-dir", type=Path, required=True)
    create.add_argument("--seed", type=int, default=42)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--packet-dir", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = create_packet(args) if args.command == "create" else finalize_packet(args)
    keys = ("status", "condition_count", "candidate_count") if args.command == "create" else ("candidate_count", "acceptable_disagreement_count")
    print(json.dumps({key: result[key] for key in keys}, indent=2))


if __name__ == "__main__":
    main()
