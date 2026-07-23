#!/usr/bin/env python3
"""Unique execution entry for the frozen FinEvo mechanism micro-pilot.

Examples:
    python run_pilot.py --contract experiments/pilot_v1.yaml \
        --stage capability-preflight --resume
    python run_pilot.py --contract experiments/pilot_v1.yaml \
        --stage development-a-d --development-fake --resume

Real stages fail closed unless the worktree is clean and HEAD is exactly the
peeled commit of the annotated ``pilot-v1`` tag.  The development stage never
uses a network provider and never emits scientific evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from verified_memory.pilot_evidence import build_pilot_evidence_package
from verified_memory.pilot_orchestrator import (
    DEFAULT_RAW_ROOT,
    PilotOrchestrationError,
    execute_stage,
    run_development_fake_matrix,
)


ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "experiments" / "pilot_v1.yaml",
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume untouched cells; unresolved reservations are never redispatched",
    )
    parser.add_argument(
        "--development-fake",
        action="store_true",
        help="allow only the no-network development-a-d diagnostic matrix",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="raw ignored pilot root (primarily useful for isolated validation)",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=ROOT / "evidence",
        help=(
            "reviewer-package root used only by the zero-provider "
            "publish-evidence stage"
        ),
    )
    return parser


def execute(args: argparse.Namespace) -> dict:
    if args.development_fake:
        if args.stage != "development-a-d":
            raise PilotOrchestrationError(
                "--development-fake requires --stage development-a-d"
            )
        return run_development_fake_matrix(
            contract_path=args.contract,
            resume=args.resume,
            raw_root=args.raw_root,
        )
    if args.stage == "development-a-d":
        raise PilotOrchestrationError(
            "development-a-d requires the explicit --development-fake flag"
        )
    if args.stage == "publish-evidence":
        package = build_pilot_evidence_package(
            contract_path=args.contract,
            run_ledger_path=args.raw_root / "run_ledger.json",
            raw_root=args.raw_root,
            build_root=args.evidence_root,
        )
        return {
            "status": (
                "complete"
                if package.scientific_complete
                else "complete-with-no-go"
            ),
            "provider_calls": 0,
            "package_dir": str(package.package_dir),
            "manifest_path": str(package.manifest_path),
            "checksums_path": str(package.checksums_path),
            "contract_sha256": package.contract_hash,
            "scientific_complete": package.scientific_complete,
            "claim_gates": package.claim_gates,
        }
    return execute_stage(
        contract_path=args.contract,
        stage_id=args.stage,
        resume=args.resume,
        raw_root=args.raw_root,
        repo_root=ROOT,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"pass", "complete", "complete-with-no-go"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
