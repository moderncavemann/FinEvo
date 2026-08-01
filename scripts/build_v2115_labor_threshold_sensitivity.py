#!/usr/bin/env python3
"""Build or validate the provider-free V2.11.5 labor diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verified_memory.v2115_labor_threshold_sensitivity import (  # noqa: E402
    OUTPUT_RELATIVE,
    build_v2115_labor_threshold_sensitivity,
    validate_v2115_labor_threshold_package,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a retrospective, descriptive V2.11.5 executed-labor "
            "threshold diagnostic without provider calls or credential reads."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="exact clean FinEvo Git worktree root",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=None,
        help=(
            "parent directory for the new package; defaults to "
            "<repo-root>/evidence/current_v2"
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate an existing package instead of building it",
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=None,
        help="existing package path for --validate-only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.absolute()
    if args.validate_only:
        package_dir = (
            args.package_dir.absolute()
            if args.package_dir is not None
            else repo_root / OUTPUT_RELATIVE
        )
        result = validate_v2115_labor_threshold_package(
            package_dir=package_dir,
            repo_root=repo_root,
        )
    else:
        if args.package_dir is not None:
            raise SystemExit("--package-dir is only valid with --validate-only")
        package = build_v2115_labor_threshold_sensitivity(
            repo_root=repo_root,
            build_root=args.build_root,
        )
        result = validate_v2115_labor_threshold_package(
            package_dir=package.package_dir,
            repo_root=repo_root,
        )
        result["package_dir"] = package.package_dir.name
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
