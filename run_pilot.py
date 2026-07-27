#!/usr/bin/env python3
"""Unique execution entry for a frozen FinEvo mechanism micro-pilot.

Examples:
    python run_pilot.py --contract experiments/pilot_v2_4.yaml \
        --stage parent-import \
        --parent-repo-root ../finevo-pilot-v2-3-release --resume
    python run_pilot.py --contract experiments/pilot_v2_5.yaml \
        --stage parent-import \
        --parent-repo-root ../finevo-pilot-v2-3-release --resume
    python run_pilot.py --contract experiments/pilot_v2_6.yaml \
        --stage parent-import \
        --parent-repo-root ../finevo-pilot-v2-5-science --resume
    python run_pilot.py --contract experiments/pilot_v2_7.yaml \
        --stage parent-import \
        --parent-repo-root ../finevo-pilot-v2-6-science --resume
    python run_pilot.py --contract experiments/pilot_v2_8.yaml \
        --stage parent-import \
        --parent-repo-root ../finevo-pilot-v2-7-science --resume
    python run_pilot.py --contract experiments/pilot_v2_9.yaml \
        --stage publish-evidence --resume
    python run_pilot.py --contract experiments/pilot_v2_3.yaml \
        --stage capability-gate --resume
    python run_pilot.py --contract experiments/pilot_v2_3.yaml \
        --stage development-a-d --development-fake --resume

Real stages fail closed unless the worktree is clean and HEAD is exactly the
peeled commit of the annotated tag named by the selected contract.  The
development stage never uses a network provider and never emits scientific
evidence.  The original pilot-v2 contract remains readable as an immutable
failed-attempt record; pilot-v2.1 is its single operational retry amendment.
Pilot-v2.2 is the evaluator-only correction that imports both immutable
capability attempts without provider redispatch. Pilot-v2.3 preserves that
denominator and adds the contract-bound capability-usage bootstrap needed to
measure the closed-loop preflight p95 before normal scientific dispatch.
Pilot-v2.4 is a prospective local-first matrix amendment.  Its zero-call
parent-import revalidates V2.3 without reopening any terminal V2.3 cell.
Pilot-v2.5 is the outcome-blind operational retry of that one failed import;
it preserves the terminal V2.4 no-go and uses a fresh raw namespace.
Pilot-v2.6 preserves the terminal V2.5 Stage-0 interface no-go and adds the
missing versioned observed-p95 reader adapter under another fresh namespace.
Pilot-v2.7 preserves V2.6 as an immutable no-go, imports only its sixteen
completed parent/q-ref/Stage-0 cells without provider redispatch, and applies
the dedicated baseline-only Stage-0 evaluator before any new A--D dispatch.
Pilot-v2.8 preserves V2.7 as an immutable no-go, regenerates q-ref through the
fresh local scripted path, and imports exactly the fourteen nested V2.6
Stage-0 cells without provider construction or redispatch.
Pilot-v2.9 preserves V2.8 as an immutable no-go and retries only q-ref's
deterministic run-summary equivalence under an identity-bound, allowlisted
projection; only fresh V2.9 A--D cells may enter treatment-effect gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_evidence import build_pilot_evidence_package
from verified_memory.pilot_v24_evidence import (
    PILOT_V24_CONTRACT_ID,
    PILOT_V29_CONTRACT_ID,
    build_pilot_v24_evidence_package,
)
from verified_memory.pilot_v25_parent_import import V25_CONTRACT_ID
from verified_memory.pilot_v26_parent_import import V26_CONTRACT_ID
from verified_memory.pilot_v27_stage0_import import V27_CONTRACT_ID
from verified_memory.pilot_v28_stage0_import import V28_CONTRACT_ID
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
        default=ROOT / "experiments" / "pilot_v2_3.yaml",
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
    parser.add_argument(
        "--parent-repo-root",
        type=Path,
        default=None,
        help=(
            "read-only parent source/raw checkout required only by the "
            "V2.4/V2.5/V2.6/V2.7/V2.8/V2.9 zero-provider parent-import stage"
        ),
    )
    parser.add_argument(
        "--source-repo-root",
        type=Path,
        default=None,
        help=(
            "read-only repository containing an immutable raw tree; accepted "
            "only by publish-evidence when publication code is newer than the "
            "source science tag"
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
    parent_repo_root = getattr(args, "parent_repo_root", None)
    if parent_repo_root is not None and args.stage != "parent-import":
        raise PilotOrchestrationError(
            "--parent-repo-root is accepted only for a parent-import stage"
        )
    source_repo_root = getattr(args, "source_repo_root", None)
    if source_repo_root is not None and args.stage != "publish-evidence":
        raise PilotOrchestrationError(
            "--source-repo-root is accepted only for publish-evidence"
        )
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
        contract = load_pilot_contract(args.contract)
        builder = (
            build_pilot_v24_evidence_package
            if contract.contract_id
            in {
                PILOT_V24_CONTRACT_ID,
                V25_CONTRACT_ID,
                V26_CONTRACT_ID,
                V27_CONTRACT_ID,
                V28_CONTRACT_ID,
                PILOT_V29_CONTRACT_ID,
            }
            else build_pilot_evidence_package
        )
        build_kwargs = {
            "contract_path": args.contract,
            "run_ledger_path": raw_root / "run_ledger.json",
            "raw_root": raw_root,
            "build_root": args.evidence_root,
        }
        if source_repo_root is not None:
            build_kwargs["source_repo_root"] = source_repo_root
        package = builder(**build_kwargs)
        return {
            "status": (
                "complete" if package.scientific_complete else "complete-with-no-go"
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
        parent_repo_root=parent_repo_root,
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
    return (
        0 if result.get("status") in {"pass", "complete", "complete-with-no-go"} else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
