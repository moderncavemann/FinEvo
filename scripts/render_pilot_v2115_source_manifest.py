#!/usr/bin/env python3
"""Replay and render the sealed V2.11.5 source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verified_memory.pilot_v2115_parent_import import (
    build_v2115_source_manifest,
)


TRACKED_SOURCE_MANIFEST_PATH = (
    ROOT / "experiments" / "pilot_v2_11_5_source_manifest.json"
)


def render_source_manifest(
    *,
    source_repo_root: Path,
    lineage_repo_root: Path,
    evidence_repo_root: Path,
) -> tuple[dict, bytes]:
    """Return the deterministic manifest and its canonical tracked bytes."""

    manifest = build_v2115_source_manifest(
        source_repo_root=source_repo_root,
        lineage_repo_root=lineage_repo_root,
        evidence_repo_root=evidence_repo_root,
    )
    payload = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return manifest, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo-root", type=Path, required=True)
    parser.add_argument("--lineage-repo-root", type=Path, required=True)
    parser.add_argument("--evidence-repo-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/pilot_v2_11_5_source_manifest.json"),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify output bytes instead of writing them",
    )
    args = parser.parse_args(argv)
    manifest, payload = render_source_manifest(
        source_repo_root=args.source_repo_root,
        lineage_repo_root=args.lineage_repo_root,
        evidence_repo_root=args.evidence_repo_root,
    )
    output = args.output.resolve(strict=False)
    if args.check:
        if not output.is_file() or output.read_bytes() != payload:
            raise SystemExit("tracked V2.11.5 source manifest drifted")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
    print(
        json.dumps(
            {
                "content_sha256": manifest["integrity"]["content_sha256"],
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "byte_size": len(payload),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
