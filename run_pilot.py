#!/usr/bin/env python3
"""Unique execution entry for a frozen FinEvo mechanism micro-pilot.

Examples:
    python run_pilot.py --contract experiments/pilot_v2.yaml \
        --stage capability-gate --resume
    python run_pilot.py --contract experiments/pilot_v2.yaml \
        --stage development-a-d --development-fake --resume

Real stages fail closed unless the worktree is clean and HEAD is exactly the
peeled commit of the annotated tag named by the selected contract.  The
development stage never uses a network provider and never emits scientific
evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_evidence import build_pilot_evidence_package
from verified_memory.pilot_orchestrator import (
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
        default=ROOT / "experiments" / "pilot_v2.yaml",
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
        default=None,
        help=(
            "raw ignored pilot root; when omitted it is derived from the "
            "selected contract ID"
        ),
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


def _raw_root_for_contract(contract_path: Path) -> Path:
    contract = load_pilot_contract(contract_path)
    namespace = contract.contract_id
    if namespace.startswith("finevo-"):
        namespace = namespace[len("finevo-") :]
    if not namespace or any(part in namespace for part in ("/", "\\", "..")):
        raise PilotOrchestrationError(
            "contract_id cannot be mapped to a safe experiment-results namespace"
        )
    return ROOT / "experiment_results" / namespace / "raw"


def execute(args: argparse.Namespace) -> dict:
    raw_root = (
        args.raw_root
        if args.raw_root is not None
        else _raw_root_for_contract(args.contract)
    )
    if args.development_fake:
        if args.stage != "development-a-d":
            raise PilotOrchestrationError(
                "--development-fake requires --stage development-a-d"
            )
        return run_development_fake_matrix(
            contract_path=args.contract,
            resume=args.resume,
            raw_root=raw_root,
        )
    if args.stage == "development-a-d":
        raise PilotOrchestrationError(
            "development-a-d requires the explicit --development-fake flag"
        )
    if args.stage == "publish-evidence":
        package = build_pilot_evidence_package(
            contract_path=args.contract,
            run_ledger_path=raw_root / "run_ledger.json",
            raw_root=raw_root,
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
        raw_root=raw_root,
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
