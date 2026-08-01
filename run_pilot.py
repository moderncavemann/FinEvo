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
    python run_pilot.py --contract experiments/pilot_v2_10.yaml \
        --stage parent-import \
        --parent-repo-root ../finevo-pilot-v2-9-science --resume
    python run_pilot.py --contract experiments/pilot_v2_10_1.yaml \
        --stage parent-import \
        --parent-repo-root ../finevo-pilot-v2-10-science --resume
    python run_pilot.py --contract experiments/pilot_v2_10_2.yaml \
        --stage parent-import \
        --parent-repo-root ../finevo-pilot-v2-10-1-science --resume
    python run_pilot.py --contract experiments/pilot_v2_11.yaml \
        --stage parent-import \
        --parent-repo-root ../finevo-pilot-v2-10-2-science --resume
    python run_pilot.py --contract experiments/pilot_v2_11_1.yaml \
        --stage parent-import \
        --parent-repo-root ../finevo-pilot-v2-11-science --resume
    python run_pilot.py --contract experiments/pilot_v2_11_2.yaml \
        --stage parent-import \
        --parent-repo-root ../finevo-pilot-v2-11-1-science --resume
    python run_pilot.py --contract experiments/pilot_v2_11_3.yaml \
        --stage parent-import \
        --parent-repo-root ../finevo-pilot-v2-11-2-science --resume
    python run_pilot.py --contract experiments/pilot_v2_11_4.yaml \
        --stage parent-import \
        --parent-repo-root ../finevo-pilot-v2-11-3-science \
        --authority-repo-root ../finevo-pilot-v2-11-2-science --resume
    python run_pilot.py --contract experiments/pilot_v2_11_5.yaml \
        --stage parent-import \
        --parent-repo-root ../finevo-pilot-v2-11-4-science \
        --authority-repo-root ../finevo-pilot-v2-11-2-science --resume
    python run_pilot.py --contract experiments/pilot_v2_11_6.yaml \
        --stage parent-import \
        --parent-repo-root ../finevo-pilot-v2-11-5-science --resume
    python run_pilot.py --contract experiments/pilot_v2_11_8.yaml \
        --stage parent-import \
        --failed-repo-root ../finevo-pilot-v2-11-7-science \
        --authority-repo-root ../finevo-pilot-v2-11-5-science --resume
    python run_pilot.py --contract experiments/pilot_v2_11_10.yaml \
        --stage parent-import \
        --failed-repo-root ../finevo-pilot-v2-11-9-science \
        --authority-repo-root ../finevo-pilot-v2-11-5-science --resume
    python run_pilot.py --contract experiments/pilot_v2_11_10.yaml \
        --stage publish-evidence \
        --source-repo-root ../finevo-pilot-v2-11-10-science \
        --failed-repo-root ../finevo-pilot-v2-11-9-science \
        --authority-repo-root ../finevo-pilot-v2-11-5-science --resume
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
Pilot-v2.10 preserves the terminal V2.9 implementation-interface no-go,
imports exactly its sixteen completed prerequisites without provider
construction, reseals the observed-p95 authority to the current release, and
dispatches every one of its 195 A--D cells under a new denominator.
Pilot-v2.10.1 preserves V2.10's terminal q-ref interface no-go, verifies its
published lineage, then imports the exact nested V2.9 prerequisites and
reseals them under a fresh current-release authority before any provider work.
Pilot-v2.10.2 preserves V2.10.1's terminal consumer-adapter no-go, imports
only the same sixteen nested V2.9 prerequisites, and reruns all 195 A--D cells
under the repaired current-release observed-p95 consumer boundary.
Pilot-v2.11 starts a fresh 136-cell hosted-model denominator. Its zero-provider
parent import consumes only V2.10.2 q-ref, nu-0.5 utility calibration,
absolute-flow threshold, and cumulative budget debit; capability, long-context
preflight, observed p95, A--D, and cross-model evidence must all be fresh.
Pilot-v2.11.1 freezes V2.11's zero-dispatch preflight no-go, imports its two
passed capability cells without new provider calls, and retries only the exact
2x12 long-context preflight under a conservative contract-envelope bootstrap.
All later science still requires the newly sealed observed-p95 authority.
Pilot-v2.11.2 freezes V2.11.1's paid preflight no-go and its complete ITT
denominator, imports only calibration and capability wrappers, repairs the
active-rule hysteresis validator, and requires a wholly fresh 2x12 preflight.
The old failed journals remain budget/failure audit evidence only.
Pilot-v2.11.3 freezes V2.11.2's terminal consumer-adapter no-go, registers a
fresh prospective 136-cell denominator, and re-verifies then reseals only its
valid capability and preflight dispatch authority with zero provider calls.
No V2.11.2 treatment cell is resumed, retried, or reclassified.
Pilot-v2.11.4 freezes V2.11.3's zero-call authority-comparison no-go, registers
a fresh 136-cell denominator, and validates the four reseal-only receipt
provenance fields separately from the exact source reservation core.
Pilot-v2.11.5 freezes V2.11.4's pre-dispatch acceptance no-go, preserves its
complete failure/budget denominator, and compares independently verified
source/current authority generations through a closed field allowlist before
any fresh scientific provider dispatch.
Pilot-v2.11.6 preserves V2.11.5's immutable 50-cell terminal prefix, imports
only its non-effect authority with zero provider calls, and registers a fresh
87-cell continuation containing one import plus the untouched D/B/cross rows.
Pilot-v2.11.7 preserves V2.11.6's immutable 87-cell integrity no-go, imports
the last valid V2.11.5 authority through an explicit second source root, and
registers a fresh 87-cell recovery denominator without provider calls.
Pilot-v2.11.8 preserves V2.11.7's immutable 87-cell terminal no-go, imports
the last valid V2.11.5 authority through the same explicit second source root,
and registers a fresh 87-cell context-recovery denominator without provider
calls.
Pilot-v2.11.9 preserves V2.11.8's immutable 87-cell terminal no-go, repairs
only the historical release-binding reconstruction, and registers another
fresh 87-cell continuation without provider calls during import.
Pilot-v2.11.10 preserves V2.11.9's immutable zero-completion no-go, imports
the independently valid V2.11.5 authority through a separate source root,
and registers fresh continuation identities under the repaired 13-to-17-to-13
authority roundtrip.  Its zero-provider evidence publisher consumes the
separate V2.11.10 science, immutable V2.11.9 no-go, and V2.11.5 authority
checkouts without importing V2.11.9 rows into the scientific denominator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_11,
    PILOT_CONTRACT_ID_V2_11_1,
    PILOT_CONTRACT_ID_V2_11_2,
    PILOT_CONTRACT_ID_V2_11_3,
    PILOT_CONTRACT_ID_V2_11_4,
    PILOT_CONTRACT_ID_V2_11_5,
    PILOT_CONTRACT_ID_V2_11_6,
    PILOT_CONTRACT_ID_V2_11_7,
    PILOT_CONTRACT_ID_V2_11_8,
    PILOT_CONTRACT_ID_V2_11_9,
    PILOT_CONTRACT_ID_V2_11_10,
    load_pilot_contract,
)
from verified_memory.pilot_evidence import build_pilot_evidence_package
from verified_memory.pilot_v24_evidence import (
    PILOT_V24_CONTRACT_ID,
    PILOT_V29_CONTRACT_ID,
    PILOT_V210_CONTRACT_ID,
    PILOT_V2101_CONTRACT_ID,
    build_pilot_v24_evidence_package,
)
from verified_memory.pilot_v25_parent_import import V25_CONTRACT_ID
from verified_memory.pilot_v26_parent_import import V26_CONTRACT_ID
from verified_memory.pilot_v27_stage0_import import V27_CONTRACT_ID
from verified_memory.pilot_v28_stage0_import import V28_CONTRACT_ID
from verified_memory.pilot_v2102_parent_import import V2102_CONTRACT_ID
from verified_memory.pilot_v2112_evidence import (
    build_pilot_v2112_evidence_package,
)
from verified_memory.pilot_v2115_evidence import (
    build_pilot_v2115_evidence_package,
)
from verified_memory.pilot_v2119_evidence import (
    build_pilot_v2119_evidence_package,
)
from verified_memory.pilot_v21110_evidence import (
    build_pilot_v21110_evidence_package,
)
from verified_memory.pilot_v2113_acceptance import (
    accept_v2113_scientific_dispatch,
)
from verified_memory.pilot_v2114_acceptance import (
    accept_v2114_scientific_dispatch,
)
from verified_memory.pilot_v2115_acceptance import (
    accept_v2115_scientific_dispatch,
)
from verified_memory.pilot_v2116_continuation import (
    accept_v2116_scientific_dispatch,
)
from verified_memory.pilot_v2117_continuation import (
    accept_v2117_scientific_dispatch,
)
from verified_memory.pilot_v2118_continuation import (
    accept_v2118_scientific_dispatch,
)
from verified_memory.pilot_v2119_continuation import (
    accept_v2119_scientific_dispatch,
)
from verified_memory.pilot_v21110_continuation import (
    accept_v21110_scientific_dispatch,
)
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
    execution_mode = parser.add_mutually_exclusive_group(required=True)
    execution_mode.add_argument("--stage")
    execution_mode.add_argument(
        "--accept-scientific-dispatch",
        action="store_true",
        help=(
            "run the V2.11.3/V2.11.4/V2.11.5/V2.11.6/V2.11.7/V2.11.8/V2.11.9/"
            "V2.11.10 "
            "zero-provider dispatch "
            "acceptance gate "
            "before loading provider credentials"
        ),
    )
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
            "V2.4/V2.5/V2.6/V2.7/V2.8/V2.9/V2.10/V2.10.1/V2.10.2/V2.11 "
            "V2.11.1, V2.11.2, V2.11.3, V2.11.4, V2.11.5, or V2.11.6 zero-provider "
            "parent-import stage; "
            "for V2.11 this "
            "is the immutable V2.10.2 science checkout, and for V2.11.1 "
            "this is the immutable V2.11 science checkout (the evidence "
            "checkout remains source-manifest bound); for V2.11.2 this is "
            "the immutable V2.11.1 science checkout; for V2.11.3 this is "
            "the immutable V2.11.2 science checkout; for V2.11.4 this is "
            "the immutable V2.11.3 terminal no-go checkout; for V2.11.5 this "
            "is the immutable V2.11.4 pre-dispatch acceptance no-go checkout; "
            "for V2.11.6 this is the immutable V2.11.5 science checkout"
        ),
    )
    parser.add_argument(
        "--failed-repo-root",
        type=Path,
        default=None,
        help=(
            "read-only immutable terminal no-go checkout required only by the "
            "V2.11.7, V2.11.8, V2.11.9, or V2.11.10 zero-provider parent-import "
            "stage, or by V2.11.10 publish-evidence; "
            "V2.11.7 requires V2.11.6, V2.11.8 requires V2.11.7, and V2.11.9 "
            "requires V2.11.8; V2.11.10 requires the immutable V2.11.9 no-go "
            "for both parent-import and publication"
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
    parser.add_argument(
        "--authority-repo-root",
        type=Path,
        default=None,
        help=(
            "read-only authority checkout required only by the V2.11.4, "
            "V2.11.5, V2.11.7, V2.11.8, V2.11.9, or V2.11.10 parent-import; "
            "V2.11.7 through V2.11.10 require the immutable V2.11.5 authority "
            "checkout while V2.11.9 and V2.11.10 publish-evidence also require "
            "that same immutable authority checkout; "
            "--failed-repo-root separately identifies their immutable terminal "
            "no-go checkout"
        ),
    )
    parser.add_argument(
        "--acceptance-output",
        type=Path,
        default=None,
        help=(
            "immutable V2.11.3/V2.11.4/V2.11.5/V2.11.6/V2.11.7/V2.11.8/V2.11.9/"
            "V2.11.10 "
            "scientific-dispatch acceptance "
            "receipt; "
            "accepted only with --accept-scientific-dispatch"
        ),
    )
    parser.add_argument(
        "--scientific-launch-input",
        type=Path,
        default=None,
        help=(
            "verified scientific launch input; defaults to "
            "<raw-root>/scientific_launch_input.json and is accepted only "
            "with --accept-scientific-dispatch"
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
    stage = getattr(args, "stage", None)
    acceptance_mode = bool(getattr(args, "accept_scientific_dispatch", False))
    resume = bool(getattr(args, "resume", False))
    development_fake = bool(getattr(args, "development_fake", False))
    parent_repo_root = getattr(args, "parent_repo_root", None)
    failed_repo_root = getattr(args, "failed_repo_root", None)
    authority_repo_root = getattr(args, "authority_repo_root", None)
    if parent_repo_root is not None and stage != "parent-import":
        raise PilotOrchestrationError(
            "--parent-repo-root is accepted only for a parent-import stage"
        )
    if authority_repo_root is not None and stage not in {
        "parent-import",
        "publish-evidence",
    }:
        raise PilotOrchestrationError(
            "--authority-repo-root is accepted only for a parent-import stage "
            "or V2.11.9/V2.11.10 publish-evidence"
        )
    if failed_repo_root is not None and stage not in {
        "parent-import",
        "publish-evidence",
    }:
        raise PilotOrchestrationError(
            "--failed-repo-root is accepted only for a parent-import stage "
            "or V2.11.10 publish-evidence"
        )
    selected_contract = None
    if stage == "parent-import":
        selected_contract = load_pilot_contract(args.contract)
        if (
            selected_contract.contract_id
            in {
                PILOT_CONTRACT_ID_V2_11,
                PILOT_CONTRACT_ID_V2_11_1,
                PILOT_CONTRACT_ID_V2_11_2,
                PILOT_CONTRACT_ID_V2_11_3,
                PILOT_CONTRACT_ID_V2_11_4,
                PILOT_CONTRACT_ID_V2_11_5,
                PILOT_CONTRACT_ID_V2_11_6,
            }
            and parent_repo_root is None
        ):
            contract_label = {
                PILOT_CONTRACT_ID_V2_11: "V2.11",
                PILOT_CONTRACT_ID_V2_11_1: "V2.11.1",
                PILOT_CONTRACT_ID_V2_11_2: "V2.11.2",
                PILOT_CONTRACT_ID_V2_11_3: "V2.11.3",
                PILOT_CONTRACT_ID_V2_11_4: "V2.11.4",
                PILOT_CONTRACT_ID_V2_11_5: "V2.11.5",
                PILOT_CONTRACT_ID_V2_11_6: "V2.11.6",
            }[selected_contract.contract_id]
            raise PilotOrchestrationError(
                f"{contract_label} parent-import requires "
                "--parent-repo-root pointing to its immutable parent science "
                "checkout"
            )
        if selected_contract.contract_id in {
            PILOT_CONTRACT_ID_V2_11_7,
            PILOT_CONTRACT_ID_V2_11_8,
            PILOT_CONTRACT_ID_V2_11_9,
            PILOT_CONTRACT_ID_V2_11_10,
        }:
            contract_label = {
                PILOT_CONTRACT_ID_V2_11_7: "V2.11.7",
                PILOT_CONTRACT_ID_V2_11_8: "V2.11.8",
                PILOT_CONTRACT_ID_V2_11_9: "V2.11.9",
                PILOT_CONTRACT_ID_V2_11_10: "V2.11.10",
            }[selected_contract.contract_id]
            failed_label = {
                PILOT_CONTRACT_ID_V2_11_7: "V2.11.6",
                PILOT_CONTRACT_ID_V2_11_8: "V2.11.7",
                PILOT_CONTRACT_ID_V2_11_9: "V2.11.8",
                PILOT_CONTRACT_ID_V2_11_10: "V2.11.9",
            }[selected_contract.contract_id]
            if parent_repo_root is not None:
                raise PilotOrchestrationError(
                    f"{contract_label} parent-import uses --failed-repo-root for "
                    f"the immutable {failed_label} no-go checkout; "
                    "--parent-repo-root is not accepted"
                )
            if failed_repo_root is None:
                raise PilotOrchestrationError(
                    f"{contract_label} parent-import requires --failed-repo-root "
                    f"pointing to the immutable {failed_label} no-go checkout"
                )
        elif failed_repo_root is not None:
            raise PilotOrchestrationError(
                "--failed-repo-root is accepted only by the V2.11.7/V2.11.8/"
                "V2.11.9/V2.11.10 parent-import stage"
            )
        if (
            selected_contract.contract_id
            in {
                PILOT_CONTRACT_ID_V2_11_4,
                PILOT_CONTRACT_ID_V2_11_5,
                PILOT_CONTRACT_ID_V2_11_7,
                PILOT_CONTRACT_ID_V2_11_8,
                PILOT_CONTRACT_ID_V2_11_9,
                PILOT_CONTRACT_ID_V2_11_10,
            }
            and authority_repo_root is None
        ):
            contract_label = {
                PILOT_CONTRACT_ID_V2_11_4: "V2.11.4",
                PILOT_CONTRACT_ID_V2_11_5: "V2.11.5",
                PILOT_CONTRACT_ID_V2_11_7: "V2.11.7",
                PILOT_CONTRACT_ID_V2_11_8: "V2.11.8",
                PILOT_CONTRACT_ID_V2_11_9: "V2.11.9",
                PILOT_CONTRACT_ID_V2_11_10: "V2.11.10",
            }[selected_contract.contract_id]
            authority_label = (
                "V2.11.5"
                if selected_contract.contract_id
                in {
                    PILOT_CONTRACT_ID_V2_11_7,
                    PILOT_CONTRACT_ID_V2_11_8,
                    PILOT_CONTRACT_ID_V2_11_9,
                    PILOT_CONTRACT_ID_V2_11_10,
                }
                else "V2.11.2"
            )
            raise PilotOrchestrationError(
                f"{contract_label} "
                "parent-import requires --authority-repo-root pointing "
                f"to the immutable {authority_label} authority checkout"
            )
        if (
            selected_contract.contract_id
            not in {
                PILOT_CONTRACT_ID_V2_11_4,
                PILOT_CONTRACT_ID_V2_11_5,
                PILOT_CONTRACT_ID_V2_11_7,
                PILOT_CONTRACT_ID_V2_11_8,
                PILOT_CONTRACT_ID_V2_11_9,
                PILOT_CONTRACT_ID_V2_11_10,
            }
            and authority_repo_root is not None
        ):
            raise PilotOrchestrationError(
                "--authority-repo-root is accepted only by the "
                "V2.11.4/V2.11.5/V2.11.7/V2.11.8/V2.11.9/V2.11.10 "
                "parent-import stage"
            )
    source_repo_root = getattr(args, "source_repo_root", None)
    if source_repo_root is not None and stage != "publish-evidence":
        raise PilotOrchestrationError(
            "--source-repo-root is accepted only for publish-evidence"
        )
    if stage == "development-a-d" and not development_fake:
        raise PilotOrchestrationError(
            "development-a-d requires the explicit --development-fake flag"
        )
    if not development_fake or acceptance_mode:
        if selected_contract is None:
            selected_contract = load_pilot_contract(args.contract)
        if (
            selected_contract.contract_id
            in {
                PILOT_CONTRACT_ID_V2_11_2,
                PILOT_CONTRACT_ID_V2_11_3,
                PILOT_CONTRACT_ID_V2_11_4,
                PILOT_CONTRACT_ID_V2_11_5,
                PILOT_CONTRACT_ID_V2_11_6,
                PILOT_CONTRACT_ID_V2_11_7,
                PILOT_CONTRACT_ID_V2_11_8,
                PILOT_CONTRACT_ID_V2_11_9,
                PILOT_CONTRACT_ID_V2_11_10,
            }
            and selected_contract.status != "frozen"
        ):
            contract_label = {
                PILOT_CONTRACT_ID_V2_11_2: "V2.11.2",
                PILOT_CONTRACT_ID_V2_11_3: "V2.11.3",
                PILOT_CONTRACT_ID_V2_11_4: "V2.11.4",
                PILOT_CONTRACT_ID_V2_11_5: "V2.11.5",
                PILOT_CONTRACT_ID_V2_11_6: "V2.11.6",
                PILOT_CONTRACT_ID_V2_11_7: "V2.11.7",
                PILOT_CONTRACT_ID_V2_11_8: "V2.11.8",
                PILOT_CONTRACT_ID_V2_11_9: "V2.11.9",
                PILOT_CONTRACT_ID_V2_11_10: "V2.11.10",
            }[selected_contract.contract_id]
            raise PilotOrchestrationError(
                f"{contract_label} real stages require a frozen contract; the draft "
                "contract permits only development-a-d --development-fake"
            )
    requested_raw_root = getattr(args, "raw_root", None)
    raw_root = (
        requested_raw_root
        if requested_raw_root is not None
        else _raw_root_for_contract(args.contract)
    )
    acceptance_output = getattr(args, "acceptance_output", None)
    scientific_launch_input = getattr(args, "scientific_launch_input", None)
    if acceptance_mode:
        assert selected_contract is not None
        incompatible = []
        if resume:
            incompatible.append("--resume")
        if development_fake or bool(getattr(args, "development_fake_provider", False)):
            incompatible.append("--development-fake")
        if parent_repo_root is not None:
            incompatible.append("--parent-repo-root")
        if failed_repo_root is not None:
            incompatible.append("--failed-repo-root")
        if source_repo_root is not None:
            incompatible.append("--source-repo-root")
        if authority_repo_root is not None:
            incompatible.append("--authority-repo-root")
        if incompatible:
            raise PilotOrchestrationError(
                "--accept-scientific-dispatch is incompatible with "
                + ", ".join(incompatible)
            )
        if (
            selected_contract.contract_id
            not in {
                PILOT_CONTRACT_ID_V2_11_3,
                PILOT_CONTRACT_ID_V2_11_4,
                PILOT_CONTRACT_ID_V2_11_5,
                PILOT_CONTRACT_ID_V2_11_6,
                PILOT_CONTRACT_ID_V2_11_7,
                PILOT_CONTRACT_ID_V2_11_8,
                PILOT_CONTRACT_ID_V2_11_9,
                PILOT_CONTRACT_ID_V2_11_10,
            }
            or selected_contract.status != "frozen"
        ):
            raise PilotOrchestrationError(
                "--accept-scientific-dispatch requires the frozen production "
                "V2.11.3, V2.11.4, V2.11.5, V2.11.6, V2.11.7, V2.11.8, "
                "V2.11.9, or V2.11.10 "
                "contract"
            )
        acceptance_dispatch = {
            PILOT_CONTRACT_ID_V2_11_3: accept_v2113_scientific_dispatch,
            PILOT_CONTRACT_ID_V2_11_4: accept_v2114_scientific_dispatch,
            PILOT_CONTRACT_ID_V2_11_5: accept_v2115_scientific_dispatch,
            PILOT_CONTRACT_ID_V2_11_6: accept_v2116_scientific_dispatch,
            PILOT_CONTRACT_ID_V2_11_7: accept_v2117_scientific_dispatch,
            PILOT_CONTRACT_ID_V2_11_8: accept_v2118_scientific_dispatch,
            PILOT_CONTRACT_ID_V2_11_9: accept_v2119_scientific_dispatch,
            PILOT_CONTRACT_ID_V2_11_10: accept_v21110_scientific_dispatch,
        }[selected_contract.contract_id]
        return acceptance_dispatch(
            contract_path=args.contract,
            repo_root=ROOT,
            raw_root=raw_root,
            scientific_launch_input_path=(
                scientific_launch_input
                if scientific_launch_input is not None
                else raw_root / "scientific_launch_input.json"
            ),
            receipt_path=acceptance_output,
        )
    if acceptance_output is not None or scientific_launch_input is not None:
        raise PilotOrchestrationError(
            "--acceptance-output and --scientific-launch-input require "
            "--accept-scientific-dispatch"
        )
    if development_fake:
        if stage != "development-a-d":
            raise PilotOrchestrationError(
                "--development-fake requires --stage development-a-d"
            )
        return run_development_fake_matrix(
            contract_path=args.contract,
            resume=resume,
            raw_root=raw_root,
        )
    if stage == "publish-evidence":
        contract = load_pilot_contract(args.contract)
        if contract.contract_id == PILOT_CONTRACT_ID_V2_11_10:
            if source_repo_root is None:
                raise PilotOrchestrationError(
                    "V2.11.10 publish-evidence requires --source-repo-root"
                )
            if failed_repo_root is None:
                raise PilotOrchestrationError(
                    "V2.11.10 publish-evidence requires --failed-repo-root"
                )
            if authority_repo_root is None:
                raise PilotOrchestrationError(
                    "V2.11.10 publish-evidence requires --authority-repo-root"
                )
        elif contract.contract_id == PILOT_CONTRACT_ID_V2_11_9:
            if source_repo_root is None:
                raise PilotOrchestrationError(
                    "V2.11.9 publish-evidence requires --source-repo-root"
                )
            if authority_repo_root is None:
                raise PilotOrchestrationError(
                    "V2.11.9 publish-evidence requires --authority-repo-root"
                )
        elif authority_repo_root is not None:
            raise PilotOrchestrationError(
                "--authority-repo-root is accepted by publish-evidence only "
                "for V2.11.9 or V2.11.10"
            )
        if (
            failed_repo_root is not None
            and contract.contract_id != PILOT_CONTRACT_ID_V2_11_10
        ):
            raise PilotOrchestrationError(
                "--failed-repo-root is accepted by publish-evidence only "
                "for V2.11.10"
            )
        builder = (
            build_pilot_v21110_evidence_package
            if contract.contract_id == PILOT_CONTRACT_ID_V2_11_10
            else (
                build_pilot_v2119_evidence_package
                if contract.contract_id == PILOT_CONTRACT_ID_V2_11_9
                else (
                    build_pilot_v2115_evidence_package
                    if contract.contract_id == PILOT_CONTRACT_ID_V2_11_5
                    else (
                        build_pilot_v2112_evidence_package
                        if contract.contract_id == PILOT_CONTRACT_ID_V2_11_2
                        else (
                            build_pilot_v24_evidence_package
                            if contract.contract_id
                            in {
                                PILOT_V24_CONTRACT_ID,
                                V25_CONTRACT_ID,
                                V26_CONTRACT_ID,
                                V27_CONTRACT_ID,
                                V28_CONTRACT_ID,
                                PILOT_V29_CONTRACT_ID,
                                PILOT_V210_CONTRACT_ID,
                                PILOT_V2101_CONTRACT_ID,
                                V2102_CONTRACT_ID,
                            }
                            else build_pilot_evidence_package
                        )
                    )
                )
            )
        )
        build_kwargs = {
            "contract_path": args.contract,
            "run_ledger_path": raw_root / "run_ledger.json",
            "raw_root": raw_root,
            "build_root": args.evidence_root,
        }
        if source_repo_root is not None:
            build_kwargs["source_repo_root"] = source_repo_root
        if contract.contract_id == PILOT_CONTRACT_ID_V2_11_10:
            build_kwargs["failed_repo_root"] = failed_repo_root
            build_kwargs["authority_repo_root"] = authority_repo_root
        elif contract.contract_id == PILOT_CONTRACT_ID_V2_11_9:
            build_kwargs["authority_repo_root"] = authority_repo_root
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
    stage_kwargs = {
        "contract_path": args.contract,
        "stage_id": stage,
        "resume": resume,
        "raw_root": raw_root,
        "repo_root": ROOT,
        "parent_repo_root": (
            failed_repo_root
            if selected_contract.contract_id
            in {
                PILOT_CONTRACT_ID_V2_11_7,
                PILOT_CONTRACT_ID_V2_11_8,
                PILOT_CONTRACT_ID_V2_11_9,
                PILOT_CONTRACT_ID_V2_11_10,
            }
            else parent_repo_root
        ),
    }
    if selected_contract.contract_id in {
        PILOT_CONTRACT_ID_V2_11_4,
        PILOT_CONTRACT_ID_V2_11_5,
        PILOT_CONTRACT_ID_V2_11_7,
        PILOT_CONTRACT_ID_V2_11_8,
        PILOT_CONTRACT_ID_V2_11_9,
        PILOT_CONTRACT_ID_V2_11_10,
    }:
        stage_kwargs["authority_repo_root"] = authority_repo_root
    return execute_stage(**stage_kwargs)


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
        0
        if result.get("status") in {"go", "pass", "complete", "complete-with-no-go"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
