"""Assemble separately collected specialist views into one final-study bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from .data import split_hash, validate_split


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected VIEW=PATH, received: {value}")
        view, path = value.split("=", 1)
        if not view or view in result:
            raise ValueError(f"View names must be non-empty and unique: {view}")
        result[view] = Path(path)
    return result


def _materialise(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and _sha256(destination) == _sha256(source):
            return "reused"
        raise FileExistsError(f"Bundle destination differs from source: {destination}")
    try:
        os.link(source, destination)
        return "hard_linked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied"


def assemble(args: argparse.Namespace) -> dict:
    split = json.loads(args.split.read_text(encoding="utf-8"))
    validate_split(split)
    sites = [site for partition in ("train", "val", "test") for site in split[partition]]
    caches = _mapping(args.view_cache)
    models = _mapping(args.view_model)
    if set(caches) != set(models):
        raise ValueError("--view-cache and --view-model must define the same views")
    # Complete every split/model/cache compatibility check before writing any
    # bundle output, so a failed assembly cannot leave a convincing partial
    # final-study directory behind.
    cache_sources = []
    for site in sites:
        for view, cache_root in caches.items():
            source = cache_root / site / f"{view}.pt"
            if not source.is_file():
                raise FileNotFoundError(f"Missing {view} cache for frozen site {site}: {source}")
            cache_sources.append((site, view, source, args.output_cache / site / f"{view}.pt"))
    merged_results = []
    source_split_records = {}
    split_modes = set()
    model_specs = []
    for view, model_root in models.items():
        source_view = model_root / view
        if not source_view.is_dir():
            raise FileNotFoundError(f"Missing trained model view: {source_view}")
        comparison = json.loads((model_root / "comparison.json").read_text(encoding="utf-8"))
        source_split_path = model_root / comparison.get("split", "pilot_split.json")
        source_split = json.loads(source_split_path.read_text(encoding="utf-8"))
        mismatched = [
            partition for partition in ("train", "val", "test")
            if list(source_split.get(partition, [])) != list(split.get(partition, []))
        ]
        if mismatched:
            raise ValueError(
                f"{view} model partitions {mismatched} do not match the requested bundle split. "
                "Retrain this view on the exact final evaluation split before assembly."
            )
        split_mode = comparison.get("split_mode", "pilot" if comparison.get("pilot", True) else "governed")
        split_modes.add(split_mode)
        if split_mode != "governed" and not getattr(args, "allow_pilot_models", False):
            raise ValueError(
                f"{view} model source is a {split_mode} run. Pilot checkpoints cannot be attached "
                "to a final multiview split; pass --allow-pilot-models only for historical reproduction."
            )
        source_split_records[view] = {
            "path": str(source_split_path.resolve()),
            "sha256": _sha256(source_split_path),
            "split_hash": source_split.get("split_hash"),
            "split_mode": split_mode,
        }
        model_specs.append((view, source_view, args.output_phase5 / view, comparison))
        merged_results.extend(item for item in comparison["results"] if item["view"] == view)

    records = []
    for site, view, source, destination in cache_sources:
        records.append({
            "site_id": site, "view": view, "source": str(source.resolve()),
            "destination": str(destination), "status": _materialise(source, destination),
            "sha256": _sha256(source),
        })
    args.output_phase5.mkdir(parents=True, exist_ok=True)
    for _, source_view, destination_view, _ in model_specs:
        shutil.copytree(source_view, destination_view, dirs_exist_ok=True)
    bundled_split = dict(split)
    bundled_split["split_hash"] = split_hash(bundled_split)
    governed_bundle = split_modes == {"governed"}
    split_filename = "governed_split.json" if governed_bundle else "pilot_split.json"
    (args.output_phase5 / split_filename).write_text(json.dumps(bundled_split, indent=2), encoding="utf-8")
    (args.output_phase5 / "comparison.json").write_text(json.dumps({
        "schema_version": 2, "pilot": not governed_bundle,
        "split_mode": "governed" if governed_bundle else "pilot", "views": sorted(caches),
        "split": split_filename, "results": merged_results,
    }, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "aligned_multiview_study_bundle",
        "split": str(args.split.resolve()), "split_sha256": _sha256(args.split),
        "split_hash": bundled_split["split_hash"], "site_count": len(sites), "views": sorted(caches),
        "split_mode": "governed" if governed_bundle else "pilot",
        "model_source_splits": source_split_records,
        "cache_records": records,
        "model_sources": {view: str(path.resolve()) for view, path in models.items()},
        "model_output_hashes": {
            str(path.relative_to(args.output_phase5)): _sha256(path)
            for path in sorted(args.output_phase5.rglob("*")) if path.is_file() and path.name != "bundle_manifest.json"
        },
    }
    (args.output_phase5 / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--view-cache", action="append", required=True, help="VIEW=cache-root; repeat for each view")
    parser.add_argument("--view-model", action="append", required=True, help="VIEW=phase5-root; repeat for each view")
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--output-phase5", type=Path, required=True)
    parser.add_argument("--allow-pilot-models", action="store_true", help="Historical reproduction only")
    return parser.parse_args()


def main() -> None:
    report = assemble(parse_args())
    print(json.dumps({key: report[key] for key in ("status", "site_count", "views", "split_hash")}, indent=2))


if __name__ == "__main__":
    main()
