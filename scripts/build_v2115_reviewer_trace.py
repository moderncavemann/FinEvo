#!/usr/bin/env python3
"""Build the zero-provider V2.11.5 reviewer closed-loop trace package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verified_memory.reviewer_closed_loop_trace import (  # noqa: E402
    build_trace_package,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic descriptive trace from the sealed V2.11.5 "
            "raw tree. This command makes zero new provider calls."
        )
    )
    parser.add_argument(
        "--source-repo-root",
        type=Path,
        required=True,
        help="Detached, tracked-clean checkout at annotated tag pilot-v2.11.5-science.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory; an existing path is never overwritten.",
    )
    parser.add_argument(
        "--publisher-repo-root",
        type=Path,
        default=ROOT,
        help="Tracked-clean publication-consumer checkout (default: this repository).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trace_path = build_trace_package(
        source_repo_root=args.source_repo_root,
        output_dir=args.output_dir,
        publisher_repo_root=args.publisher_repo_root,
    )
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "artifact": str(trace_path),
                "artifact_content_sha256": payload["integrity"]["content_sha256"],
                "publication_provider_calls": 0,
                "source_provider_call_scope": (
                    "historical-observations-read-from-sealed-logs"
                ),
                "stage_go": payload.get("provenance", {}).get("stage_go"),
                "stage_status": payload.get("provenance", {}).get("stage_status"),
                "status": payload["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
