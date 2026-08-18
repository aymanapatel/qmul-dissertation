"""Materialise static plus local-browser evidence for one saved corpus site."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evidence import capture_local_browser_evidence, collect_static_evidence, write_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    artifact = collect_static_evidence(args.site_dir)
    try:
        artifact.browser_evidence = capture_local_browser_evidence(args.site_dir / "0.html", args.screenshot)
        static_paths = {node.identity.css_path for node in artifact.nodes}
        browser_paths = {node["css_path"] for node in artifact.browser_evidence.get("layout", [])}
        common = static_paths & browser_paths
        artifact.browser_evidence["alignment"] = {
            "static_nodes": len(static_paths), "browser_nodes": len(browser_paths), "matched_paths": len(common),
            "static_coverage": len(common) / max(1, len(static_paths)),
            "browser_coverage": len(common) / max(1, len(browser_paths)),
        }
    except Exception as exc:
        artifact.collection_failures.append(f"browser_capture: {type(exc).__name__}: {exc}")
    write_evidence(artifact, args.output)
    print(
        f"saved={args.output} nodes={len(artifact.nodes)} "
        f"layout={len(artifact.browser_evidence.get('layout', []))} "
        f"ax={len(artifact.browser_evidence.get('accessibility_tree', []))} "
        f"failures={len(artifact.collection_failures)}"
    )


if __name__ == "__main__":
    main()
