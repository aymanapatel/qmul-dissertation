"""Finalize completed blinded detection ratings into Phase 7 truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .annotation_packet import finalize_packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = finalize_packet(args.packet_dir, args.output)
    print(json.dumps(report["agreement"], indent=2))


if __name__ == "__main__":
    main()
