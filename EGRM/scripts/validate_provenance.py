#!/usr/bin/env python3
"""Validate extracted module hashes declared by SOURCE_PROVENANCE.json."""

from __future__ import annotations

import argparse
import json

from egrm.provenance import validate_extraction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-repo-root")
    parser.add_argument(
        "--destination-only",
        action="store_true",
        help="skip Git source-object checks; never sufficient for a release gate",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate_extraction(
                args.manifest,
                source_repo_root=args.source_repo_root,
                verify_source=not args.destination_only,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
