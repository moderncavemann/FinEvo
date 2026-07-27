"""Lane-separated evidence publication for the FinEvo V2.4--V2.10 pilot.

V2.4 is deliberately not a continuation of the terminal V2.3 denominator.
Its scientific matrix contains two independently interpreted lanes:

* a complete local Llama-3.3 mechanism lane; and
* a bounded GPT-5.2 confirmatory lane.

This adapter never pools seed directions or treatment effects across those
lanes.  Every C -> A -> D -> B stage receives its own 4-of-5 complete-pair
gate, while all failed, stopped, nonterminal, and missing ITT cells remain in
the package denominator.  Narrative intervention is explicitly deferred and
is not silently treated as either completed or failed evidence.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from statistics import median
import sys
import tempfile
from typing import Any, Mapping, Sequence

from .pilot_contract import PilotContract, load_pilot_contract
from .pilot_evidence import (
    HISTORICAL_SCOPE,
    PILOT_CHECKSUM_SCHEMA_VERSION,
    PILOT_EXPERIMENT_C_SENSITIVITY_SCHEMA_VERSION,
    PILOT_FAILURE_LEDGER_SCHEMA_VERSION,
    PilotEvidenceError,
    PilotEvidencePackage,
    _aggregate_csv,
    _atomic_bytes,
    _evidence_namespace,
    _experiment_a_gate,
    _experiment_b_summary,
    _experiment_c_gate,
    _experiment_d_gate,
    _json_copy,
    _method_scaffold,
    _normalize_ledger,
    _pretty_bytes,
    _sha256_file,
    _strict_json_load,
    _validated_release_controls,
    source_repository_context,
)


PILOT_V24_EVIDENCE_SCHEMA_VERSION = "finevo-pilot-v2.4-evidence-package-v1"
PILOT_V24_CONTRACT_ID = "finevo-pilot-v2.4"
PILOT_V25_EVIDENCE_SCHEMA_VERSION = "finevo-pilot-v2.5-evidence-package-v1"
PILOT_V25_CONTRACT_ID = "finevo-pilot-v2.5"
PILOT_V26_EVIDENCE_SCHEMA_VERSION = "finevo-pilot-v2.6-evidence-package-v1"
PILOT_V26_CONTRACT_ID = "finevo-pilot-v2.6"
PILOT_V27_EVIDENCE_SCHEMA_VERSION = "finevo-pilot-v2.7-evidence-package-v1"
PILOT_V27_CONTRACT_ID = "finevo-pilot-v2.7"
PILOT_V28_EVIDENCE_SCHEMA_VERSION = "finevo-pilot-v2.8-evidence-package-v1"
PILOT_V28_CONTRACT_ID = "finevo-pilot-v2.8"
PILOT_V29_EVIDENCE_SCHEMA_VERSION = "finevo-pilot-v2.9-evidence-package-v1"
PILOT_V29_CONTRACT_ID = "finevo-pilot-v2.9"
PILOT_V210_EVIDENCE_SCHEMA_VERSION = "finevo-pilot-v2.10-evidence-package-v1"
PILOT_V210_CONTRACT_ID = "finevo-pilot-v2.10"
PILOT_V29_IMPLEMENTATION_FAILURE_SCHEMA_VERSION = (
    "finevo-pilot-v2.9-implementation-failure-summary-v1"
)
_PILOT_V29_RELEASE_COMMIT = "2349ccd41560383965da8880744cf4df366c9ee5"
_PILOT_V29_EVIDENCE_PUBLICATION_COMMIT = (
    "51525614e138e5b7ac498d15b409048d5110b753"
)
_PILOT_V29_EVIDENCE_MERGE_COMMIT = (
    "08fcbc0dd9319fcc86c3f4e812c3db504a0c5a17"
)
_PILOT_V29_EVIDENCE_PACKAGE_MANIFEST_SHA256 = (
    "6d006ba59c5af6a1e0dd3931466b90d4599edc0ded47e2de3ea4f8ecd6c4831a"
)
_PILOT_V29_EVIDENCE_CHECKSUMS_SHA256 = (
    "b0de7185c710b69736ddfe1d331b7f6308165a9f03bb0c616f14ec1fd7a515db"
)
_PILOT_V29_RECEIPT_PATH_FAILURE_SHA256 = (
    "d4b516ad7a51dc7a09dcad56e2abe7f7f5236cc49523014fc7b8b8c3fdf2870e"
)
_PILOT_V29_FAILURE_STAGE_COUNTS = {
    "experiment-a": 20,
    "experiment-b": 15,
    "experiment-c": 20,
    "experiment-d": 30,
    "local-experiment-a": 20,
    "local-experiment-b": 25,
    "local-experiment-c": 20,
    "local-experiment-d": 35,
}
_PILOT_V29_SOURCE_AUDIT = {
    "producer": {
        "path": "verified_memory/pilot_v29_stage0_import.py",
        "git_blob_oid": "82acc1e9fadbd2a632f732eceeaaec7812d73192",
        "file_sha256": (
            "e15eca803b3a978591d2df6895548d6527e29d5035361063eabff2410cd6622c"
        ),
        "function": "verify_v29_imported_v28_observed_p95",
        "returned_binding": (
            "authority.path/file_sha256/content_sha256 plus source_git_commit"
        ),
    },
    "consumer": {
        "path": "verified_memory/pilot_orchestrator.py",
        "git_blob_oid": "e0b0de6a83c3a59a081954ba4fa5e22b77d68e92",
        "file_sha256": (
            "062dbcf664a1488e191f8c5ecd6eb7f7ea5bfe41caa4faf260fdac2864ae4f1b"
        ),
        "function": "_runner_p95_reservations",
        "expected_binding": (
            "receipt_path/receipt_file_sha256/" "receipt_content_sha256/git_commit"
        ),
    },
}
_PILOT_V27_EVIDENCE_CHECKSUMS_FILE_SHA256 = (
    "b28889b0fc590ec884c69fdf43f88b01ce8f384491168a031ac5fdb2a6b3caad"
)
PILOT_V24_STAGE_ORDER = (
    "experiment-c",
    "experiment-a",
    "experiment-d",
    "experiment-b",
)
PILOT_V24_MIN_PAIRED_SEEDS = 4
PILOT_V24_TOTAL_PAIRED_SEEDS = 5

_V24_STAGE_IDS = (
    "parent-import",
    "q-ref-resolution",
    "stage0-calibration",
    "local-experiment-c",
    "local-experiment-a",
    "local-experiment-d",
    "local-experiment-b",
    "experiment-c",
    "experiment-a",
    "experiment-d",
    "experiment-b",
)
_V27_IMPORTED_PREREQUISITE_COUNTS = {
    "parent-import": 1,
    "q-ref-resolution": 1,
    "stage0-calibration": 14,
}
_V28_PREREQUISITE_COUNTS = {
    "parent-import": 1,
    "q-ref-resolution": 1,
    "stage0-calibration": 14,
}
_V29_PREREQUISITE_COUNTS = {
    "parent-import": 1,
    "q-ref-resolution": 1,
    "stage0-calibration": 14,
}
_V210_PREREQUISITE_COUNTS = {
    "parent-import": 1,
    "q-ref-resolution": 1,
    "stage0-calibration": 14,
}
_V24_LANES: Mapping[str, Mapping[str, Any]] = {
    "local": {
        "model_id": "llama33_local_controlled",
        "stage_ids": {
            "experiment-c": "local-experiment-c",
            "experiment-a": "local-experiment-a",
            "experiment-d": "local-experiment-d",
            "experiment-b": "local-experiment-b",
        },
        "arms": {
            "experiment-c": (
                "full",
                "unverified-dual",
                "verified-error-candidate",
                "verified-error-forced",
                "unverified-error-forced",
            ),
            "experiment-a": (
                "no-context",
                "prompt-only",
                "retrieval-only",
                "full",
            ),
            "experiment-d": (
                "matched-a",
                "matched-b",
                "no-memory",
                "shuffled-episodic",
                "wrong-context",
                "error-verified",
                "error-unverified",
            ),
            "experiment-b": (
                "no-memory",
                "episodic-only",
                "semantic-only",
                "unverified-dual",
                "full",
            ),
        },
    },
    "gpt52": {
        "model_id": "gpt52_main",
        "stage_ids": {
            "experiment-c": "experiment-c",
            "experiment-a": "experiment-a",
            "experiment-d": "experiment-d",
            "experiment-b": "experiment-b",
        },
        "arms": {
            "experiment-c": (
                "full",
                "unverified-dual",
                "verified-error-candidate",
                "verified-error-forced",
                "unverified-error-forced",
            ),
            "experiment-a": (
                "no-context",
                "prompt-only",
                "retrieval-only",
                "full",
            ),
            "experiment-d": (
                "matched-a",
                "matched-b",
                "no-memory",
                "wrong-context",
                "error-verified",
                "error-unverified",
            ),
            "experiment-b": (
                "full",
                "episodic-only",
                "no-memory",
            ),
        },
    },
}
_V210_C_SENSITIVITY_FILES = {
    "local": "local_experiment_c_rule_sensitivity.json",
    "gpt52": "experiment_c_rule_sensitivity.json",
}


def _contract_id_version_label(contract_id: Any) -> str:
    if contract_id == PILOT_V24_CONTRACT_ID:
        return "V2.4"
    if contract_id == PILOT_V25_CONTRACT_ID:
        return "V2.5"
    if contract_id == PILOT_V26_CONTRACT_ID:
        return "V2.6"
    if contract_id == PILOT_V27_CONTRACT_ID:
        return "V2.7"
    if contract_id == PILOT_V28_CONTRACT_ID:
        return "V2.8"
    if contract_id == PILOT_V29_CONTRACT_ID:
        return "V2.9"
    if contract_id == PILOT_V210_CONTRACT_ID:
        return "V2.10"
    raise PilotEvidenceError(
        "lane-separated evidence adapter received another contract"
    )


def _contract_version_label(contract: PilotContract) -> str:
    return _contract_id_version_label(contract.contract_id)


def _evidence_schema_version(contract: PilotContract) -> str:
    if contract.contract_id == PILOT_V24_CONTRACT_ID:
        return PILOT_V24_EVIDENCE_SCHEMA_VERSION
    if contract.contract_id == PILOT_V25_CONTRACT_ID:
        return PILOT_V25_EVIDENCE_SCHEMA_VERSION
    if contract.contract_id == PILOT_V26_CONTRACT_ID:
        return PILOT_V26_EVIDENCE_SCHEMA_VERSION
    if contract.contract_id == PILOT_V27_CONTRACT_ID:
        return PILOT_V27_EVIDENCE_SCHEMA_VERSION
    if contract.contract_id == PILOT_V28_CONTRACT_ID:
        return PILOT_V28_EVIDENCE_SCHEMA_VERSION
    if contract.contract_id == PILOT_V29_CONTRACT_ID:
        return PILOT_V29_EVIDENCE_SCHEMA_VERSION
    if contract.contract_id == PILOT_V210_CONTRACT_ID:
        return PILOT_V210_EVIDENCE_SCHEMA_VERSION
    raise PilotEvidenceError(
        "lane-separated evidence adapter received another contract"
    )


def _atomic_install_directory_no_replace(source: Path, target: Path) -> None:
    """Atomically install one directory while refusing any destination entry."""

    source = source.absolute()
    target = target.absolute()
    if source.parent != target.parent:
        raise PilotEvidenceError(
            "V2.4 evidence source and destination must share one parent"
        )
    if not source.is_dir() or source.is_symlink():
        raise PilotEvidenceError(
            "V2.4 evidence install source must be a real directory"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    source_raw = os.fsencode(source)
    target_raw = os.fsencode(target)
    result: int
    if sys.platform == "darwin":
        rename_exclusive = 0x00000004
        renamex = getattr(libc, "renamex_np", None)
        if renamex is None:
            raise PilotEvidenceError(
                "Darwin atomic no-replace rename primitive is unavailable"
            )
        renamex.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renamex.restype = ctypes.c_int
        result = int(renamex(source_raw, target_raw, rename_exclusive))
    elif sys.platform.startswith("linux"):
        rename_noreplace = 0x00000001
        at_fdcwd = -100
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise PilotEvidenceError(
                "Linux atomic no-replace rename primitive is unavailable"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = int(
            renameat2(
                at_fdcwd,
                source_raw,
                at_fdcwd,
                target_raw,
                rename_noreplace,
            )
        )
    else:
        raise PilotEvidenceError(
            f"atomic no-replace publication is unsupported on {sys.platform!r}"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise PilotEvidenceError(
            f"refusing to overwrite V2.4 evidence package: {target}"
        )
    raise PilotEvidenceError(
        "atomic V2.4 evidence installation failed: " f"{os.strerror(error_number)}"
    )


def _validate_v24_contract_matrix(contract: PilotContract) -> None:
    version_label = _contract_version_label(contract)
    if tuple(contract.stage_ids) != _V24_STAGE_IDS:
        raise PilotEvidenceError(
            f"{version_label} evidence stages differ from the fixed "
            "local-first matrix"
        )
    expected_seeds = tuple(int(seed) for seed in contract.seeds["sets"]["main"])
    if (
        len(expected_seeds) != PILOT_V24_TOTAL_PAIRED_SEEDS
        or len(set(expected_seeds)) != PILOT_V24_TOTAL_PAIRED_SEEDS
    ):
        raise PilotEvidenceError(
            f"{version_label} evidence requires five unique main seeds"
        )

    for lane_id, lane in _V24_LANES.items():
        model_id = str(lane["model_id"])
        stage_ids = lane["stage_ids"]
        arms_by_stage = lane["arms"]
        observed_order = tuple(
            canonical
            for canonical in PILOT_V24_STAGE_ORDER
            if str(stage_ids[canonical]) in contract.stage_ids
        )
        if observed_order != PILOT_V24_STAGE_ORDER:
            raise PilotEvidenceError(
                f"{version_label} {lane_id} lane does not preserve " "C -> A -> D -> B"
            )
        for canonical in PILOT_V24_STAGE_ORDER:
            stage_id = str(stage_ids[canonical])
            expected_arms = tuple(str(arm) for arm in arms_by_stage[canonical])
            specs = contract.expand(stage=stage_id)
            if (
                {spec.model_id for spec in specs} != {model_id}
                or {spec.environment_seed for spec in specs} != set(expected_seeds)
                or {spec.arm_id for spec in specs} != set(expected_arms)
                or len(specs) != len(expected_arms) * len(expected_seeds)
            ):
                raise PilotEvidenceError(
                    f"{version_label} {lane_id}/{canonical} registered "
                    "matrix drifted"
                )
    if any(
        spec.arm_id == "narrative-content" or spec.narrative_id != "none"
        for spec in contract.expand()
    ):
        raise PilotEvidenceError(
            f"{version_label} narrative intervention must remain deferred "
            "and unregistered"
        )


def _v27_imported_prerequisite_summary(
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Bind imported V2.7 prerequisites without admitting them to A--D gates."""

    if contract.contract_id != PILOT_V27_CONTRACT_ID:
        return None
    expected_specs = {
        spec.run_id: spec
        for stage_id in _V27_IMPORTED_PREREQUISITE_COUNTS
        for spec in contract.expand(stage=stage_id)
    }
    expected_count = sum(_V27_IMPORTED_PREREQUISITE_COUNTS.values())
    if len(expected_specs) != expected_count:
        raise PilotEvidenceError("V2.7 imported prerequisite contract matrix drifted")
    observed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        stage_id = str(row.get("stage_id"))
        if stage_id not in _V27_IMPORTED_PREREQUISITE_COUNTS:
            continue
        run_id = str(row.get("run_id"))
        if run_id not in expected_specs or run_id in observed:
            raise PilotEvidenceError(
                "V2.7 imported prerequisite identity or multiplicity drifted"
            )
        spec = expected_specs[run_id]
        expected_scientific_eligible = (
            stage_id == "stage0-calibration" and row.get("status") == "complete"
        )
        if (
            stage_id != spec.stage_id
            or row.get("contract_id") != contract.contract_id
            or row.get("model_id") != spec.model_id
            or row.get("arm_id") != spec.arm_id
            or row.get("environment_seed") != spec.environment_seed
            or row.get("scientific_eligible") is not expected_scientific_eligible
        ):
            raise PilotEvidenceError(
                "V2.7 imported prerequisite row differs from its "
                "registered eligibility boundary"
            )
        observed[run_id] = row
    if set(observed) != set(expected_specs):
        raise PilotEvidenceError("V2.7 imported prerequisite denominator is incomplete")

    by_stage: dict[str, Any] = {}
    for stage_id, stage_count in _V27_IMPORTED_PREREQUISITE_COUNTS.items():
        stage_rows = [
            row for row in observed.values() if row.get("stage_id") == stage_id
        ]
        statuses: dict[str, int] = {}
        for row in stage_rows:
            status = str(row.get("status"))
            statuses[status] = statuses.get(status, 0) + 1
        by_stage[stage_id] = {
            "registered_cells": stage_count,
            "observed_cells": len(stage_rows),
            "status_counts": dict(sorted(statuses.items())),
            "all_complete": statuses == {"complete": stage_count},
            "scientific_eligible_cells": (
                sum(row.get("scientific_eligible") is True for row in stage_rows)
            ),
            "evidence_scope": (
                "stage0-baseline-calibration"
                if stage_id == "stage0-calibration"
                else "operational-prerequisite"
            ),
            "used_in_a_d_effect_gates": False,
        }
    return {
        "source_contract_id": PILOT_V26_CONTRACT_ID,
        "imported_registered_cells": expected_count,
        "imported_observed_cells": len(observed),
        "all_imported_complete": all(
            stage["all_complete"] for stage in by_stage.values()
        ),
        "stages": by_stage,
        "a_d_treatment_effect_evidence": False,
        "used_in_a_d_effect_gates": False,
        "claim_boundary": (
            "parent authority and q-ref are operational prerequisites; "
            "Stage-0 is eligible calibration evidence only; none are A-D "
            "treatment-effect evidence"
        ),
    }


def _v27_inherited_budget_boundary(
    contract: PilotContract,
    *,
    denominator: Mapping[str, Any],
    release_controls: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Require the exact cumulative V2.6 debit before V2.7 publication."""

    if contract.contract_id != PILOT_V27_CONTRACT_ID:
        return None
    amendment = getattr(contract, "stage0_evaluator_retry_amendment", None)
    if not isinstance(amendment, Mapping):
        raise PilotEvidenceError(
            "V2.7 evidence contract lacks its Stage-0 evaluator amendment"
        )
    carry = amendment.get("budget_carry_forward")
    if not isinstance(carry, Mapping):
        raise PilotEvidenceError(
            "V2.7 evidence contract lacks its inherited budget boundary"
        )
    expected = carry.get("cumulative_prior")
    if not isinstance(expected, Mapping):
        raise PilotEvidenceError("V2.7 inherited cumulative debit is malformed")
    budget = release_controls.get("budget_ledger")
    if not isinstance(budget, Mapping):
        raise PilotEvidenceError("V2.7 release controls lack a budget ledger")
    checks = budget.get("checks")
    totals = budget.get("actual_totals")
    stage_cost = budget.get("actual_stage_cost_usd")
    if (
        not isinstance(checks, Mapping)
        or not isinstance(totals, Mapping)
        or not isinstance(stage_cost, Mapping)
    ):
        raise PilotEvidenceError("V2.7 budget evidence lacks exact debit accounting")

    def number(value: Any, name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise PilotEvidenceError(f"V2.7 budget {name} is invalid")
        return float(value)

    expected_cost = number(expected.get("cost_usd"), "prior cost")
    expected_completions = number(
        expected.get("hosted_completions"),
        "prior completions",
    )
    expected_storage = number(expected.get("storage_bytes"), "prior storage")
    observed_cost = number(totals.get("cost_usd"), "actual cost")
    observed_completions = number(
        totals.get("completions"),
        "actual completions",
    )
    observed_storage = number(totals.get("storage_bytes"), "actual storage")
    parent_bucket = str(expected.get("stage_bucket"))
    inherited_stage_cost = number(
        stage_cost.get(parent_bucket),
        "inherited stage cost",
    )
    binding_checks = {
        "denominator_exact": (
            denominator.get("expected_count") == 211
            and denominator.get("observed_ledger_count") == 211
        ),
        "parent_debit_exact": checks.get("parent_debit_exact") is True,
        "parent_stage_cost_exact": math.isclose(
            inherited_stage_cost,
            expected_cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "cumulative_cost_not_reset": observed_cost >= expected_cost,
        "cumulative_completions_not_reset": (
            observed_completions >= expected_completions
        ),
        "cumulative_storage_not_reset": observed_storage >= expected_storage,
    }
    if not all(binding_checks.values()):
        raise PilotEvidenceError(
            "V2.7 evidence does not preserve its inherited debit/denominator"
        )
    return {
        "source_contract_id": PILOT_V26_CONTRACT_ID,
        "expected_cumulative_prior": _json_copy(dict(expected)),
        "observed_cumulative_totals": {
            "cost_usd": observed_cost,
            "hosted_completions": observed_completions,
            "storage_bytes": observed_storage,
        },
        "checks": binding_checks,
        "pass": True,
    }


def _v28_amendment(contract: PilotContract) -> Mapping[str, Any]:
    """Return the sealed V2.8 identity-amendment boundary."""

    amendment = getattr(contract, "qref_identity_retry_amendment", None)
    if not isinstance(amendment, Mapping):
        raise PilotEvidenceError(
            "V2.8 evidence contract lacks its q-ref identity retry amendment"
        )
    return amendment


def _v28_parent_evidence_lineage(
    contract: PilotContract,
) -> dict[str, Any] | None:
    """Expose the immutable V2.7 no-go lineage without importing its rows."""

    if contract.contract_id != PILOT_V28_CONTRACT_ID:
        return None
    amendment = _v28_amendment(contract)
    lineage = amendment.get("evidence_lineage")
    failure = amendment.get("failure_classification")
    retry = amendment.get("retry_policy")
    if (
        not isinstance(lineage, Mapping)
        or not isinstance(failure, Mapping)
        or not isinstance(retry, Mapping)
    ):
        raise PilotEvidenceError("V2.8 parent evidence lineage is malformed")
    checks = {
        "parent_contract_id": failure.get("parent_contract_id")
        == PILOT_V27_CONTRACT_ID,
        "parent_status_preserved": (
            failure.get("terminal_status") == "complete-with-no-go"
            and lineage.get("parent_evidence_status") == "complete-with-no-go"
        ),
        "parent_denominator_preserved": (
            failure.get("registered_cells") == 211
            and failure.get("status_counts")
            == {"complete": 1, "integrity-stopped": 210}
            and retry.get("preserve_parent_denominator") is True
            and retry.get("v2_7_status_counts_rewrite") == "forbidden"
            and retry.get("v2_7_terminal_cell_reclassification") == "forbidden"
        ),
        "parent_package_rewrite_forbidden": (
            lineage.get("parent_evidence_rewrite") == "forbidden"
            and lineage.get("parent_claim_reclassification") == "forbidden"
        ),
        "a_d_effects_are_fresh_only": (
            lineage.get("v2_8_effect_aggregation_uses_only_v2_8_a_d_cells") is True
            and failure.get("a_d_treatment_effect_outcomes_generated") is False
            and failure.get("a_d_treatment_effect_outcomes_inspected") is False
        ),
    }
    if not all(checks.values()):
        raise PilotEvidenceError(
            "V2.8 evidence does not preserve the immutable V2.7 no-go lineage"
        )
    return {
        "source_contract_id": PILOT_V27_CONTRACT_ID,
        "source_contract_sha256": failure.get("parent_contract_sha256"),
        "source_release_tag": failure.get("parent_release_tag"),
        "source_release_commit": failure.get("parent_release_commit"),
        "source_evidence_commit": lineage.get("parent_evidence_commit"),
        "source_evidence_merge_commit": lineage.get("parent_evidence_merge_commit"),
        "source_evidence_namespace": lineage.get("parent_evidence_namespace"),
        "source_evidence_status": lineage.get("parent_evidence_status"),
        "package_manifest_file_sha256": failure.get(
            "evidence_package_manifest_file_sha256"
        ),
        "root_cause": {
            "code": failure.get("root_cause_code"),
            "message": failure.get("root_cause_message"),
            "failed_stage_id": failure.get("failed_stage_id"),
        },
        "parent_registered_cells": failure.get("registered_cells"),
        "parent_status_counts": _json_copy(dict(failure["status_counts"])),
        "parent_rows_imported_into_v2_8_effect_aggregate": 0,
        "parent_package_rewritten": False,
        "checks": checks,
        "pass": True,
    }


def _v28_prerequisite_summary(
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Classify V2.8 authority, q-ref, and Stage-0 cells as non-effect inputs."""

    if contract.contract_id != PILOT_V28_CONTRACT_ID:
        return None
    amendment = _v28_amendment(contract)
    qref = amendment.get("q_ref_regeneration")
    stage0 = amendment.get("stage0_import")
    if not isinstance(qref, Mapping) or not isinstance(stage0, Mapping):
        raise PilotEvidenceError("V2.8 prerequisite policy is malformed")
    if (
        qref.get("fresh_zero_hosted_provider_regeneration") is not True
        or qref.get("scripted_diagnostic_provider_required") is not True
        or qref.get("hosted_provider_construction_during_regeneration") is not False
        or qref.get("hosted_provider_calls") != 0
        or qref.get("hosted_cost_usd") != 0.0
        or qref.get("scripted_diagnostic_calls") != 48
        or stage0.get("imported_complete_cells") != 14
        or stage0.get("imported_cell_breakdown") != {"stage0-calibration": 14}
        or stage0.get("provider_construction_during_import") is not False
        or stage0.get("provider_redispatch_for_imported_cells") != "forbidden"
    ):
        raise PilotEvidenceError("V2.8 prerequisite execution boundary drifted")

    expected_specs = {
        spec.run_id: spec
        for stage_id in _V28_PREREQUISITE_COUNTS
        for spec in contract.expand(stage=stage_id)
    }
    expected_count = sum(_V28_PREREQUISITE_COUNTS.values())
    if len(expected_specs) != expected_count:
        raise PilotEvidenceError("V2.8 prerequisite contract matrix drifted")
    observed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        stage_id = str(row.get("stage_id"))
        if stage_id not in _V28_PREREQUISITE_COUNTS:
            continue
        run_id = str(row.get("run_id"))
        if run_id not in expected_specs or run_id in observed:
            raise PilotEvidenceError(
                "V2.8 prerequisite identity or multiplicity drifted"
            )
        spec = expected_specs[run_id]
        expected_scientific_eligible = (
            stage_id == "stage0-calibration" and row.get("status") == "complete"
        )
        if (
            stage_id != spec.stage_id
            or row.get("contract_id") != contract.contract_id
            or row.get("model_id") != spec.model_id
            or row.get("arm_id") != spec.arm_id
            or row.get("environment_seed") != spec.environment_seed
            or row.get("scientific_eligible") is not expected_scientific_eligible
        ):
            raise PilotEvidenceError(
                "V2.8 prerequisite row differs from its registered "
                "eligibility boundary"
            )
        observed[run_id] = row
    if set(observed) != set(expected_specs):
        raise PilotEvidenceError("V2.8 prerequisite denominator is incomplete")

    classifications = {
        "parent-import": {
            "origin": "immutable-v2.7-parent-authority",
            "execution": "imported-prerequisite",
            "evidence_scope": "operational-prerequisite",
        },
        "q-ref-resolution": {
            "origin": "fresh-v2.8-scripted-diagnostic",
            "execution": "fresh-scripted-zero-hosted",
            "evidence_scope": "q-ref-calibration-prerequisite",
        },
        "stage0-calibration": {
            "origin": "v2.6-stage0-via-v2.7-nested-snapshot",
            "execution": "hash-verified-import-no-provider-dispatch",
            "evidence_scope": "stage0-baseline-calibration",
        },
    }
    by_stage: dict[str, Any] = {}
    for stage_id, stage_count in _V28_PREREQUISITE_COUNTS.items():
        stage_rows = [
            row for row in observed.values() if row.get("stage_id") == stage_id
        ]
        statuses: dict[str, int] = {}
        for row in stage_rows:
            status = str(row.get("status"))
            statuses[status] = statuses.get(status, 0) + 1
        by_stage[stage_id] = {
            **classifications[stage_id],
            "registered_cells": stage_count,
            "observed_cells": len(stage_rows),
            "status_counts": dict(sorted(statuses.items())),
            "all_complete": statuses == {"complete": stage_count},
            "scientific_eligible_cells": sum(
                row.get("scientific_eligible") is True for row in stage_rows
            ),
            "used_in_a_d_effect_gates": False,
            "treatment_effect_evidence": False,
        }
    return {
        "registered_cells": expected_count,
        "observed_cells": len(observed),
        "all_prerequisites_complete": all(
            stage["all_complete"] for stage in by_stage.values()
        ),
        "stages": by_stage,
        "q_ref_provider_accounting": {
            "hosted_provider_calls": 0,
            "hosted_cost_usd": 0.0,
            "scripted_diagnostic_calls": 48,
        },
        "stage0_imported_cells": 14,
        "a_d_treatment_effect_evidence": False,
        "used_in_a_d_effect_gates": False,
        "claim_boundary": (
            "parent authority, the fresh scripted q-ref, and the 14 imported "
            "Stage-0 cells are prerequisites/non-effect evidence; only fresh "
            "V2.8 A-D cells may support treatment-effect claims"
        ),
    }


def _v28_itt_row_preservation(
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
    *,
    denominator: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Require exactly one retained row for every V2.8 registered identity."""

    if contract.contract_id != PILOT_V28_CONTRACT_ID:
        return None
    expected = {spec.run_id: spec.to_dict() for spec in contract.expand()}
    if len(expected) != 211:
        raise PilotEvidenceError("V2.8 registered ITT matrix is not 211 cells")
    observed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        run_id = str(row.get("run_id"))
        spec = expected.get(run_id)
        if spec is None or run_id in observed:
            raise PilotEvidenceError(
                "V2.8 evidence rows contain an unknown or duplicate ITT identity"
            )
        if any(row.get(field) != value for field, value in spec.items()):
            raise PilotEvidenceError(
                "V2.8 evidence row differs from its registered ITT identity"
            )
        observed[run_id] = row
    if set(observed) != set(expected):
        raise PilotEvidenceError("V2.8 evidence does not retain all 211 ITT rows")
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    status_counts = dict(sorted(status_counts.items()))
    if (
        denominator.get("expected_count") != 211
        or denominator.get("status_counts") != status_counts
    ):
        raise PilotEvidenceError(
            "V2.8 denominator does not match the retained 211 ITT rows"
        )
    return {
        "registered_rows": 211,
        "retained_rows": len(rows),
        "failed_or_stopped_rows": sum(row.get("status") != "complete" for row in rows),
        "status_counts": status_counts,
        "all_registered_rows_retained": True,
        "failures_retained": True,
    }


def _v28_inherited_budget_boundary(
    contract: PilotContract,
    *,
    denominator: Mapping[str, Any],
    release_controls: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Require V2.8 to debit the V2.7 cumulative spend under the $500 cap."""

    if contract.contract_id != PILOT_V28_CONTRACT_ID:
        return None
    amendment = _v28_amendment(contract)
    carry = amendment.get("budget_carry_forward")
    qref = amendment.get("q_ref_regeneration")
    if not isinstance(carry, Mapping) or not isinstance(qref, Mapping):
        raise PilotEvidenceError("V2.8 cumulative budget boundary is malformed")
    expected = carry.get("cumulative_prior")
    budget = release_controls.get("budget_ledger")
    if not isinstance(expected, Mapping) or not isinstance(budget, Mapping):
        raise PilotEvidenceError("V2.8 budget evidence lacks cumulative accounting")
    checks = budget.get("checks")
    totals = budget.get("actual_totals")
    stage_cost = budget.get("actual_stage_cost_usd")
    if (
        not isinstance(checks, Mapping)
        or not isinstance(totals, Mapping)
        or not isinstance(stage_cost, Mapping)
    ):
        raise PilotEvidenceError("V2.8 budget evidence lacks exact debit accounting")

    def number(value: Any, name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise PilotEvidenceError(f"V2.8 budget {name} is invalid")
        return float(value)

    expected_cost = number(expected.get("cost_usd"), "prior cost")
    expected_completions = number(
        expected.get("hosted_completions"),
        "prior completions",
    )
    expected_storage = number(expected.get("storage_bytes"), "prior storage")
    observed_cost = number(totals.get("cost_usd"), "actual cost")
    observed_completions = number(totals.get("completions"), "actual completions")
    observed_storage = number(totals.get("storage_bytes"), "actual storage")
    total_cap = number(carry.get("total_cap_usd"), "total cap")
    contract_cap = number(contract.budgets.get("total_usd"), "contract total cap")
    parent_bucket = str(expected.get("stage_bucket"))
    inherited_stage_cost = number(
        stage_cost.get(parent_bucket),
        "inherited stage cost",
    )
    binding_checks = {
        "denominator_exact": (
            denominator.get("expected_count") == 211
            and denominator.get("observed_ledger_count") == 211
        ),
        "parent_debit_exact": checks.get("parent_debit_exact") is True,
        "parent_stage_cost_exact": math.isclose(
            inherited_stage_cost,
            expected_cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "cumulative_cost_not_reset": observed_cost >= expected_cost,
        "cumulative_completions_not_reset": (
            observed_completions >= expected_completions
        ),
        "cumulative_storage_not_reset": observed_storage >= expected_storage,
        "total_cap_is_500": math.isclose(
            total_cap,
            500.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            contract_cap,
            500.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "q_ref_zero_hosted": (
            qref.get("hosted_provider_calls") == 0
            and qref.get("hosted_cost_usd") == 0.0
            and qref.get("hosted_provider_construction_during_regeneration") is False
        ),
        "q_ref_scripted_calls_exact": qref.get("scripted_diagnostic_calls") == 48,
    }
    if not all(binding_checks.values()):
        raise PilotEvidenceError(
            "V2.8 evidence does not preserve its inherited debit/denominator"
        )
    return {
        "source_contract_id": PILOT_V27_CONTRACT_ID,
        "total_cap_usd": total_cap,
        "expected_cumulative_prior": _json_copy(dict(expected)),
        "observed_cumulative_totals": {
            "cost_usd": observed_cost,
            "hosted_completions": observed_completions,
            "storage_bytes": observed_storage,
        },
        "q_ref_incremental": {
            "hosted_provider_calls": 0,
            "hosted_cost_usd": 0.0,
            "scripted_diagnostic_calls": 48,
        },
        "checks": binding_checks,
        "pass": True,
    }


def _v29_amendment(contract: PilotContract) -> Mapping[str, Any]:
    """Return the sealed V2.9 deterministic-summary amendment boundary."""

    amendment = getattr(
        contract,
        "qref_summary_equivalence_amendment",
        None,
    )
    if not isinstance(amendment, Mapping):
        raise PilotEvidenceError(
            "V2.9 evidence contract lacks its q-ref summary-equivalence " "amendment"
        )
    return amendment


def _v29_parent_evidence_lineage(
    contract: PilotContract,
) -> dict[str, Any] | None:
    """Expose immutable V2.8 no-go lineage without importing parent rows."""

    if contract.contract_id != PILOT_V29_CONTRACT_ID:
        return None
    amendment = _v29_amendment(contract)
    lineage = amendment.get("evidence_lineage")
    failure = amendment.get("failure_classification")
    retry = amendment.get("retry_policy")
    if (
        not isinstance(lineage, Mapping)
        or not isinstance(failure, Mapping)
        or not isinstance(retry, Mapping)
    ):
        raise PilotEvidenceError("V2.9 parent evidence lineage is malformed")
    expected_statuses = {
        "complete": 1,
        "failed": 1,
        "integrity-stopped": 209,
    }
    checks = {
        "parent_contract_id": failure.get("parent_contract_id")
        == PILOT_V28_CONTRACT_ID,
        "parent_status_preserved": (
            failure.get("terminal_status") == "complete-with-no-go"
            and lineage.get("parent_evidence_status") == "complete-with-no-go"
        ),
        "parent_denominator_preserved": (
            failure.get("registered_cells") == 211
            and failure.get("status_counts") == expected_statuses
            and retry.get("preserve_parent_denominator") is True
            and retry.get("v2_8_status_counts_rewrite") == "forbidden"
            and retry.get("v2_8_terminal_cell_reclassification") == "forbidden"
        ),
        "parent_package_rewrite_forbidden": (
            lineage.get("parent_evidence_rewrite") == "forbidden"
            and lineage.get("parent_claim_reclassification") == "forbidden"
        ),
        "a_d_effects_are_fresh_only": (
            lineage.get("v2_9_effect_aggregation_uses_only_v2_9_a_d_cells") is True
            and failure.get("a_d_treatment_effect_outcomes_generated") is False
            and failure.get("a_d_treatment_effect_outcomes_inspected") is False
        ),
    }
    if not all(checks.values()):
        raise PilotEvidenceError(
            "V2.9 evidence does not preserve the immutable V2.8 no-go lineage"
        )
    return {
        "source_contract_id": PILOT_V28_CONTRACT_ID,
        "source_contract_sha256": failure.get("parent_contract_sha256"),
        "source_release_tag": failure.get("parent_release_tag"),
        "source_release_commit": failure.get("parent_release_commit"),
        "source_evidence_commit": lineage.get("parent_evidence_commit"),
        "source_evidence_merge_commit": lineage.get("parent_evidence_merge_commit"),
        "source_evidence_namespace": lineage.get("parent_evidence_namespace"),
        "source_evidence_status": lineage.get("parent_evidence_status"),
        "package_manifest_file_sha256": failure.get(
            "evidence_package_manifest_file_sha256"
        ),
        "checksums_file_sha256": failure.get("evidence_checksums_file_sha256"),
        "root_cause": {
            "code": failure.get("root_cause_code"),
            "message": failure.get("root_cause_message"),
            "failed_stage_id": failure.get("failed_stage_id"),
        },
        "parent_registered_cells": failure.get("registered_cells"),
        "parent_status_counts": _json_copy(dict(failure["status_counts"])),
        "parent_rows_imported_into_v2_9_effect_aggregate": 0,
        "parent_package_rewritten": False,
        "checks": checks,
        "pass": True,
    }


def _v29_prerequisite_summary(
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Classify V2.9 authority, q-ref, and Stage-0 cells as non-effect inputs."""

    if contract.contract_id != PILOT_V29_CONTRACT_ID:
        return None
    amendment = _v29_amendment(contract)
    qref = amendment.get("q_ref_regeneration")
    stage0 = amendment.get("stage0_import")
    if not isinstance(qref, Mapping) or not isinstance(stage0, Mapping):
        raise PilotEvidenceError("V2.9 prerequisite policy is malformed")
    if (
        qref.get("fresh_zero_hosted_provider_regeneration") is not True
        or qref.get("scripted_diagnostic_provider_required") is not True
        or qref.get("hosted_provider_construction_during_regeneration") is not False
        or qref.get("hosted_provider_calls") != 0
        or qref.get("hosted_cost_usd") != 0.0
        or qref.get("scripted_diagnostic_calls") != 48
        or stage0.get("imported_complete_cells") != 14
        or stage0.get("imported_cell_breakdown") != {"stage0-calibration": 14}
        or stage0.get("provider_construction_during_import") is not False
        or stage0.get("provider_redispatch_for_imported_cells") != "forbidden"
    ):
        raise PilotEvidenceError("V2.9 prerequisite execution boundary drifted")

    expected_specs = {
        spec.run_id: spec
        for stage_id in _V29_PREREQUISITE_COUNTS
        for spec in contract.expand(stage=stage_id)
    }
    expected_count = sum(_V29_PREREQUISITE_COUNTS.values())
    if len(expected_specs) != expected_count:
        raise PilotEvidenceError("V2.9 prerequisite contract matrix drifted")
    observed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        stage_id = str(row.get("stage_id"))
        if stage_id not in _V29_PREREQUISITE_COUNTS:
            continue
        run_id = str(row.get("run_id"))
        if run_id not in expected_specs or run_id in observed:
            raise PilotEvidenceError(
                "V2.9 prerequisite identity or multiplicity drifted"
            )
        spec = expected_specs[run_id]
        expected_scientific_eligible = (
            stage_id == "stage0-calibration" and row.get("status") == "complete"
        )
        if (
            stage_id != spec.stage_id
            or row.get("contract_id") != contract.contract_id
            or row.get("model_id") != spec.model_id
            or row.get("arm_id") != spec.arm_id
            or row.get("environment_seed") != spec.environment_seed
            or row.get("scientific_eligible") is not expected_scientific_eligible
        ):
            raise PilotEvidenceError(
                "V2.9 prerequisite row differs from its registered "
                "eligibility boundary"
            )
        observed[run_id] = row
    if set(observed) != set(expected_specs):
        raise PilotEvidenceError("V2.9 prerequisite denominator is incomplete")

    classifications = {
        "parent-import": {
            "origin": "immutable-v2.8-parent-authority",
            "execution": "imported-prerequisite",
            "evidence_scope": "operational-prerequisite",
        },
        "q-ref-resolution": {
            "origin": "fresh-v2.9-scripted-diagnostic",
            "execution": "fresh-scripted-zero-hosted",
            "evidence_scope": "q-ref-calibration-prerequisite",
        },
        "stage0-calibration": {
            "origin": "v2.6-stage0-via-v2.8-nested-snapshot",
            "execution": "hash-verified-import-no-provider-dispatch",
            "evidence_scope": "stage0-baseline-calibration",
        },
    }
    by_stage: dict[str, Any] = {}
    for stage_id, stage_count in _V29_PREREQUISITE_COUNTS.items():
        stage_rows = [
            row for row in observed.values() if row.get("stage_id") == stage_id
        ]
        statuses: dict[str, int] = {}
        for row in stage_rows:
            status = str(row.get("status"))
            statuses[status] = statuses.get(status, 0) + 1
        by_stage[stage_id] = {
            **classifications[stage_id],
            "registered_cells": stage_count,
            "observed_cells": len(stage_rows),
            "status_counts": dict(sorted(statuses.items())),
            "all_complete": statuses == {"complete": stage_count},
            "scientific_eligible_cells": sum(
                row.get("scientific_eligible") is True for row in stage_rows
            ),
            "used_in_a_d_effect_gates": False,
            "treatment_effect_evidence": False,
        }
    return {
        "registered_cells": expected_count,
        "observed_cells": len(observed),
        "all_prerequisites_complete": all(
            stage["all_complete"] for stage in by_stage.values()
        ),
        "stages": by_stage,
        "q_ref_provider_accounting": {
            "hosted_provider_calls": 0,
            "hosted_cost_usd": 0.0,
            "scripted_diagnostic_calls": 48,
        },
        "stage0_imported_cells": 14,
        "a_d_treatment_effect_evidence": False,
        "used_in_a_d_effect_gates": False,
        "claim_boundary": (
            "parent authority, the fresh scripted q-ref, and the 14 imported "
            "Stage-0 cells are prerequisites/non-effect evidence; only fresh "
            "V2.9 A-D cells may support treatment-effect claims"
        ),
    }


def _v29_itt_row_preservation(
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
    *,
    denominator: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Require exactly one retained row for every V2.9 registered identity."""

    if contract.contract_id != PILOT_V29_CONTRACT_ID:
        return None
    expected = {spec.run_id: spec.to_dict() for spec in contract.expand()}
    if len(expected) != 211:
        raise PilotEvidenceError("V2.9 registered ITT matrix is not 211 cells")
    observed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        run_id = str(row.get("run_id"))
        spec = expected.get(run_id)
        if spec is None or run_id in observed:
            raise PilotEvidenceError(
                "V2.9 evidence rows contain an unknown or duplicate ITT identity"
            )
        if any(row.get(field) != value for field, value in spec.items()):
            raise PilotEvidenceError(
                "V2.9 evidence row differs from its registered ITT identity"
            )
        observed[run_id] = row
    if set(observed) != set(expected):
        raise PilotEvidenceError("V2.9 evidence does not retain all 211 ITT rows")
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    status_counts = dict(sorted(status_counts.items()))
    if (
        denominator.get("expected_count") != 211
        or denominator.get("status_counts") != status_counts
    ):
        raise PilotEvidenceError(
            "V2.9 denominator does not match the retained 211 ITT rows"
        )
    return {
        "registered_rows": 211,
        "retained_rows": len(rows),
        "failed_or_stopped_rows": sum(row.get("status") != "complete" for row in rows),
        "status_counts": status_counts,
        "all_registered_rows_retained": True,
        "failures_retained": True,
    }


def _v29_inherited_budget_boundary(
    contract: PilotContract,
    *,
    denominator: Mapping[str, Any],
    release_controls: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Require V2.9 to debit cumulative V2.8 spend under the $500 cap."""

    if contract.contract_id != PILOT_V29_CONTRACT_ID:
        return None
    amendment = _v29_amendment(contract)
    carry = amendment.get("budget_carry_forward")
    qref = amendment.get("q_ref_regeneration")
    if not isinstance(carry, Mapping) or not isinstance(qref, Mapping):
        raise PilotEvidenceError("V2.9 cumulative budget boundary is malformed")
    expected = carry.get("cumulative_prior")
    v28_incremental = carry.get("v2_8_incremental")
    budget = release_controls.get("budget_ledger")
    if (
        not isinstance(expected, Mapping)
        or not isinstance(v28_incremental, Mapping)
        or not isinstance(budget, Mapping)
    ):
        raise PilotEvidenceError("V2.9 budget evidence lacks cumulative accounting")
    checks = budget.get("checks")
    totals = budget.get("actual_totals")
    stage_cost = budget.get("actual_stage_cost_usd")
    if (
        not isinstance(checks, Mapping)
        or not isinstance(totals, Mapping)
        or not isinstance(stage_cost, Mapping)
    ):
        raise PilotEvidenceError("V2.9 budget evidence lacks exact debit accounting")

    def number(value: Any, name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise PilotEvidenceError(f"V2.9 budget {name} is invalid")
        return float(value)

    expected_cost = number(expected.get("cost_usd"), "prior cost")
    expected_completions = number(
        expected.get("hosted_completions"),
        "prior completions",
    )
    expected_storage = number(expected.get("storage_bytes"), "prior storage")
    observed_cost = number(totals.get("cost_usd"), "actual cost")
    observed_completions = number(
        totals.get("completions"),
        "actual completions",
    )
    observed_storage = number(totals.get("storage_bytes"), "actual storage")
    total_cap = number(carry.get("total_cap_usd"), "total cap")
    contract_cap = number(contract.budgets.get("total_usd"), "contract total cap")
    parent_bucket = str(expected.get("stage_bucket"))
    inherited_stage_cost = number(
        stage_cost.get(parent_bucket),
        "inherited stage cost",
    )
    binding_checks = {
        "denominator_exact": (
            denominator.get("expected_count") == 211
            and denominator.get("observed_ledger_count") == 211
        ),
        "parent_debit_exact": checks.get("parent_debit_exact") is True,
        "parent_stage_cost_exact": math.isclose(
            inherited_stage_cost,
            expected_cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "cumulative_cost_not_reset": observed_cost >= expected_cost,
        "cumulative_completions_not_reset": (
            observed_completions >= expected_completions
        ),
        "cumulative_storage_not_reset": observed_storage >= expected_storage,
        "total_cap_is_500": math.isclose(
            total_cap,
            500.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            contract_cap,
            500.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "v2_8_incremental_zero_hosted": (
            v28_incremental.get("cost_usd") == 0.0
            and v28_incremental.get("hosted_completions") == 0
            and v28_incremental.get("scripted_diagnostic_calls") == 48
        ),
        "q_ref_zero_hosted": (
            qref.get("hosted_provider_calls") == 0
            and qref.get("hosted_cost_usd") == 0.0
            and qref.get("hosted_provider_construction_during_regeneration") is False
        ),
        "q_ref_scripted_calls_exact": qref.get("scripted_diagnostic_calls") == 48,
    }
    if not all(binding_checks.values()):
        raise PilotEvidenceError(
            "V2.9 evidence does not preserve its inherited debit/denominator"
        )
    return {
        "source_contract_id": PILOT_V28_CONTRACT_ID,
        "total_cap_usd": total_cap,
        "expected_cumulative_prior": _json_copy(dict(expected)),
        "observed_cumulative_totals": {
            "cost_usd": observed_cost,
            "hosted_completions": observed_completions,
            "storage_bytes": observed_storage,
        },
        "v2_8_incremental": _json_copy(dict(v28_incremental)),
        "q_ref_incremental": {
            "hosted_provider_calls": 0,
            "hosted_cost_usd": 0.0,
            "scripted_diagnostic_calls": 48,
        },
        "checks": binding_checks,
        "pass": True,
    }


def _v210_amendment(contract: PilotContract) -> Mapping[str, Any]:
    """Return the sealed V2.10 current-release runner-binding amendment."""

    amendment = getattr(contract, "p95_runner_binding_retry_amendment", None)
    if not isinstance(amendment, Mapping):
        raise PilotEvidenceError(
            "V2.10 evidence contract lacks its p95 runner-binding amendment"
        )
    return amendment


def _v210_parent_evidence_lineage(
    contract: PilotContract,
) -> dict[str, Any] | None:
    """Expose V2.9's immutable implementation no-go without importing effects."""

    if contract.contract_id != PILOT_V210_CONTRACT_ID:
        return None
    amendment = _v210_amendment(contract)
    lineage = amendment.get("evidence_lineage")
    failure = amendment.get("failure_classification")
    retry = amendment.get("retry_policy")
    observation = amendment.get("observation_boundary")
    fresh = amendment.get("fresh_science_dispatch")
    if not all(
        isinstance(value, Mapping)
        for value in (lineage, failure, retry, observation, fresh)
    ):
        raise PilotEvidenceError("V2.10 parent evidence lineage is malformed")
    expected_statuses = {"complete": 26, "failed": 185}
    expected_failed_stages = dict(_PILOT_V29_FAILURE_STAGE_COUNTS)
    source_evidence_commit = (
        lineage.get("parent_evidence_commit")
        or failure.get("parent_evidence_commit")
    )
    source_evidence_merge_commit = (
        lineage.get("parent_evidence_merge_commit")
        or failure.get("parent_evidence_merge_commit")
    )
    manifest_sha = failure.get("evidence_package_manifest_file_sha256")
    checksums_sha = failure.get("evidence_checksums_file_sha256")
    if contract.status == "draft":
        source_evidence_commit = (
            source_evidence_commit or _PILOT_V29_EVIDENCE_PUBLICATION_COMMIT
        )
        source_evidence_merge_commit = (
            source_evidence_merge_commit or _PILOT_V29_EVIDENCE_MERGE_COMMIT
        )
        manifest_sha = manifest_sha or _PILOT_V29_EVIDENCE_PACKAGE_MANIFEST_SHA256
        checksums_sha = checksums_sha or _PILOT_V29_EVIDENCE_CHECKSUMS_SHA256
    checks = {
        "parent_contract_id": (
            failure.get("parent_contract_id") == PILOT_V29_CONTRACT_ID
        ),
        "parent_status_preserved": (
            failure.get("terminal_status") == "complete-with-no-go"
            and lineage.get("parent_evidence_status") == "complete-with-no-go"
        ),
        "parent_denominator_preserved": (
            failure.get("registered_cells") == 211
            and failure.get("status_counts") == expected_statuses
            and failure.get("completed_cell_breakdown")
            == {
                "experiment-c-offline-candidate-admission": 5,
                "local-experiment-c-offline-candidate-admission": 5,
                "parent-import": 1,
                "q-ref-resolution": 1,
                "stage0-calibration": 14,
            }
            and failure.get("failed_actor_cell_count") == 185
            and failure.get("failed_actor_stage_counts") == expected_failed_stages
            and retry.get("preserve_parent_denominator") is True
            and retry.get("v2_9_status_counts_rewrite") == "forbidden"
            and retry.get("v2_9_terminal_cell_reclassification") == "forbidden"
        ),
        "implementation_failure_preserved": (
            failure.get("root_cause_code")
            == "imported-p95-runner-binding-shape-mismatch"
            and failure.get("failure_phase")
            == "before-provider-construction-and-dispatch"
            and failure.get("incremental_cost_usd") == 0.0
            and failure.get("incremental_hosted_completions") == 0
            and failure.get("actor_action_utility_rule_exposure_outcomes_generated")
            is False
        ),
        "offline_candidate_outcomes_disclosed": (
            failure.get("offline_candidate_admission_cells_generated") == 10
            and observation.get("offline_candidate_admission_outcomes_generated")
            == 10
            and observation.get("offline_candidate_admission_outcomes_observed")
            is True
            and observation.get("all_a_d_outcomes_unobserved_claim_forbidden")
            is True
        ),
        "parent_package_rewrite_forbidden": (
            lineage.get("parent_evidence_rewrite") == "forbidden"
            and lineage.get("parent_claim_reclassification") == "forbidden"
        ),
        "fresh_v2_10_effects_only": (
            lineage.get(
                "v2_10_effect_aggregation_uses_only_fresh_v2_10_a_d_cells"
            )
            is True
            and lineage.get(
                "v2_9_offline_candidate_outcomes_are_not_v2_10_effect_evidence"
            )
            is True
            and fresh.get("a_d_cells") == 195
            and fresh.get("imported_a_d_completions") == 0
            and fresh.get("v2_9_offline_candidate_admission_reuse")
            == "forbidden"
        ),
        "immutable_parent_evidence_bound": (
            source_evidence_commit == _PILOT_V29_EVIDENCE_PUBLICATION_COMMIT
            and source_evidence_merge_commit == _PILOT_V29_EVIDENCE_MERGE_COMMIT
            and manifest_sha == _PILOT_V29_EVIDENCE_PACKAGE_MANIFEST_SHA256
            and checksums_sha == _PILOT_V29_EVIDENCE_CHECKSUMS_SHA256
        ),
    }
    if not all(checks.values()):
        raise PilotEvidenceError(
            "V2.10 evidence does not preserve the immutable V2.9 "
            "implementation no-go lineage"
        )
    return {
        "source_contract_id": PILOT_V29_CONTRACT_ID,
        "source_contract_sha256": failure.get("parent_contract_sha256"),
        "source_release_tag": failure.get("parent_release_tag"),
        "source_release_commit": failure.get("parent_release_commit"),
        "source_evidence_commit": source_evidence_commit,
        "source_evidence_merge_commit": source_evidence_merge_commit,
        "source_evidence_namespace": lineage.get("parent_evidence_namespace"),
        "source_evidence_status": lineage.get("parent_evidence_status"),
        "package_manifest_file_sha256": manifest_sha,
        "checksums_file_sha256": checksums_sha,
        "root_cause": {
            "code": failure.get("root_cause_code"),
            "message": failure.get("root_cause_message"),
            "failure_phase": failure.get("failure_phase"),
        },
        "parent_registered_cells": failure.get("registered_cells"),
        "parent_status_counts": _json_copy(dict(failure["status_counts"])),
        "parent_completed_cell_breakdown": _json_copy(
            dict(failure["completed_cell_breakdown"])
        ),
        "parent_failed_actor_cells": failure.get("failed_actor_cell_count"),
        "parent_offline_candidate_admission_cells_generated": 10,
        "parent_actor_treatment_effect_outcomes_generated": False,
        "parent_rows_imported_into_v2_10_effect_aggregate": 0,
        "parent_offline_candidates_imported_as_v2_10_effects": 0,
        "parent_package_rewritten": False,
        "checks": checks,
        "pass": True,
    }


def _v210_prerequisite_summary(
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Classify exactly 16 imported V2.10 prerequisites as non-effect inputs."""

    if contract.contract_id != PILOT_V210_CONTRACT_ID:
        return None
    amendment = _v210_amendment(contract)
    imported = amendment.get("prerequisite_import")
    fresh = amendment.get("fresh_science_dispatch")
    design = amendment.get("science_design_invariance")
    if not all(isinstance(value, Mapping) for value in (imported, fresh, design)):
        raise PilotEvidenceError("V2.10 prerequisite policy is malformed")
    if (
        imported.get("source_contract_id") != PILOT_V29_CONTRACT_ID
        or imported.get("imported_complete_cells") != 16
        or imported.get("imported_cell_breakdown") != _V210_PREREQUISITE_COUNTS
        or imported.get("provider_construction_during_import") is not False
        or imported.get("provider_redispatch_for_imported_cells") != "forbidden"
        or imported.get("prerequisites_are_treatment_effect_evidence") is not False
        or fresh.get("a_d_cells") != 195
        or fresh.get("imported_a_d_completions") != 0
        or fresh.get("a_d_provider_dispatch") != "fresh-only"
        or design.get("prerequisite_cells") != 16
        or design.get("fresh_a_d_cells") != 195
        or design.get("registered_cells") != 211
    ):
        raise PilotEvidenceError("V2.10 prerequisite execution boundary drifted")

    expected_specs = {
        spec.run_id: spec
        for stage_id in _V210_PREREQUISITE_COUNTS
        for spec in contract.expand(stage=stage_id)
    }
    if len(expected_specs) != 16:
        raise PilotEvidenceError("V2.10 prerequisite contract matrix drifted")
    observed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        stage_id = str(row.get("stage_id"))
        if stage_id not in _V210_PREREQUISITE_COUNTS:
            continue
        run_id = str(row.get("run_id"))
        if run_id not in expected_specs or run_id in observed:
            raise PilotEvidenceError(
                "V2.10 prerequisite identity or multiplicity drifted"
            )
        spec = expected_specs[run_id]
        expected_scientific_eligible = (
            stage_id == "stage0-calibration" and row.get("status") == "complete"
        )
        if (
            stage_id != spec.stage_id
            or row.get("contract_id") != contract.contract_id
            or row.get("model_id") != spec.model_id
            or row.get("arm_id") != spec.arm_id
            or row.get("environment_seed") != spec.environment_seed
            or row.get("scientific_eligible") is not expected_scientific_eligible
        ):
            raise PilotEvidenceError(
                "V2.10 prerequisite row differs from its registered "
                "eligibility boundary"
            )
        observed[run_id] = row
    if set(observed) != set(expected_specs):
        raise PilotEvidenceError("V2.10 prerequisite denominator is incomplete")

    classifications = {
        "parent-import": {
            "origin": "immutable-v2.9-parent-authority",
            "evidence_scope": "operational-prerequisite",
        },
        "q-ref-resolution": {
            "origin": "hash-verified-v2.9-q-ref-import",
            "evidence_scope": "q-ref-calibration-prerequisite",
        },
        "stage0-calibration": {
            "origin": "hash-verified-v2.9-stage0-import",
            "evidence_scope": "stage0-baseline-calibration",
        },
    }
    by_stage: dict[str, Any] = {}
    for stage_id, stage_count in _V210_PREREQUISITE_COUNTS.items():
        stage_rows = [
            row for row in observed.values() if row.get("stage_id") == stage_id
        ]
        statuses: dict[str, int] = {}
        for row in stage_rows:
            status = str(row.get("status"))
            statuses[status] = statuses.get(status, 0) + 1
        by_stage[stage_id] = {
            **classifications[stage_id],
            "execution": "hash-verified-import-no-provider-dispatch",
            "registered_cells": stage_count,
            "observed_cells": len(stage_rows),
            "status_counts": dict(sorted(statuses.items())),
            "all_complete": statuses == {"complete": stage_count},
            "scientific_eligible_cells": sum(
                row.get("scientific_eligible") is True for row in stage_rows
            ),
            "used_in_a_d_effect_gates": False,
            "treatment_effect_evidence": False,
        }
    return {
        "source_contract_id": PILOT_V29_CONTRACT_ID,
        "registered_cells": 16,
        "observed_cells": len(observed),
        "all_prerequisites_complete": all(
            stage["all_complete"] for stage in by_stage.values()
        ),
        "stages": by_stage,
        "import_provider_accounting": {
            "provider_construction": False,
            "provider_calls": 0,
            "hosted_cost_usd": 0.0,
        },
        "stage0_imported_cells": 14,
        "fresh_a_d_cells_required": 195,
        "imported_a_d_effect_cells": 0,
        "a_d_treatment_effect_evidence": False,
        "used_in_a_d_effect_gates": False,
        "claim_boundary": (
            "the 16 V2.9-derived parent, q-ref, and Stage-0 cells are "
            "hash-verified V2.10 prerequisites only; all 195 V2.10 A-D "
            "cells, including candidate-admission cells, must be fresh"
        ),
    }


def _v210_itt_row_preservation(
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
    *,
    denominator: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Retain all 211 V2.10 identities and bind the 16/195 split."""

    if contract.contract_id != PILOT_V210_CONTRACT_ID:
        return None
    expected = {spec.run_id: spec.to_dict() for spec in contract.expand()}
    expected_effect = {
        run_id: spec
        for run_id, spec in expected.items()
        if spec["stage_id"] not in _V210_PREREQUISITE_COUNTS
    }
    if len(expected) != 211 or len(expected_effect) != 195:
        raise PilotEvidenceError("V2.10 registered ITT matrix is not 16 + 195 cells")
    observed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        run_id = str(row.get("run_id"))
        spec = expected.get(run_id)
        if spec is None or run_id in observed:
            raise PilotEvidenceError(
                "V2.10 evidence rows contain an unknown or duplicate ITT identity"
            )
        if any(row.get(field) != value for field, value in spec.items()):
            raise PilotEvidenceError(
                "V2.10 evidence row differs from its registered ITT identity"
            )
        observed[run_id] = row
    if set(observed) != set(expected):
        raise PilotEvidenceError("V2.10 evidence does not retain all 211 ITT rows")
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    status_counts = dict(sorted(status_counts.items()))
    if (
        denominator.get("expected_count") != 211
        or denominator.get("observed_ledger_count") != 211
        or denominator.get("status_counts") != status_counts
    ):
        raise PilotEvidenceError(
            "V2.10 denominator does not match the retained 211 ITT rows"
        )
    return {
        "registered_rows": 211,
        "retained_rows": len(rows),
        "prerequisite_rows": 16,
        "fresh_a_d_rows": len(expected_effect),
        "imported_a_d_rows": 0,
        "failed_or_stopped_rows": sum(row.get("status") != "complete" for row in rows),
        "status_counts": status_counts,
        "all_registered_rows_retained": True,
        "failures_retained": True,
    }


def _v210_inherited_budget_boundary(
    contract: PilotContract,
    *,
    denominator: Mapping[str, Any],
    release_controls: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Debit the complete V2.9 lineage before V2.10 dispatch under $500."""

    if contract.contract_id != PILOT_V210_CONTRACT_ID:
        return None
    amendment = _v210_amendment(contract)
    carry = amendment.get("budget_carry_forward")
    if not isinstance(carry, Mapping):
        raise PilotEvidenceError("V2.10 cumulative budget boundary is malformed")
    expected = carry.get("cumulative_prior")
    v29_incremental = carry.get("v2_9_incremental")
    budget = release_controls.get("budget_ledger")
    if not all(
        isinstance(value, Mapping)
        for value in (expected, v29_incremental, budget)
    ):
        raise PilotEvidenceError("V2.10 budget evidence lacks cumulative accounting")
    checks = budget.get("checks")
    totals = budget.get("actual_totals")
    stage_cost = budget.get("actual_stage_cost_usd")
    if not all(isinstance(value, Mapping) for value in (checks, totals, stage_cost)):
        raise PilotEvidenceError("V2.10 budget evidence lacks exact debit accounting")

    def number(value: Any, name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise PilotEvidenceError(f"V2.10 budget {name} is invalid")
        return float(value)

    expected_cost = number(expected.get("cost_usd"), "prior cost")
    expected_completions = number(
        expected.get("hosted_completions"), "prior completions"
    )
    expected_storage = number(expected.get("storage_bytes"), "prior storage")
    observed_cost = number(totals.get("cost_usd"), "actual cost")
    observed_completions = number(totals.get("completions"), "actual completions")
    observed_storage = number(totals.get("storage_bytes"), "actual storage")
    total_cap = number(carry.get("total_cap_usd"), "total cap")
    contract_cap = number(contract.budgets.get("total_usd"), "contract total cap")
    inherited_stage_cost = number(
        stage_cost.get(str(expected.get("stage_bucket"))),
        "inherited stage cost",
    )
    binding_checks = {
        "denominator_exact": (
            denominator.get("expected_count") == 211
            and denominator.get("observed_ledger_count") == 211
        ),
        "parent_debit_exact": checks.get("parent_debit_exact") is True,
        "parent_stage_cost_exact": math.isclose(
            inherited_stage_cost, expected_cost, rel_tol=0.0, abs_tol=1e-12
        ),
        "cumulative_prior_exact": (
            math.isclose(expected_cost, 3.212770875, rel_tol=0.0, abs_tol=1e-12)
            and expected_completions == 184
            and expected_storage == 50_425_235
        ),
        "cumulative_cost_not_reset": observed_cost >= expected_cost,
        "cumulative_completions_not_reset": (
            observed_completions >= expected_completions
        ),
        "cumulative_storage_not_reset": observed_storage >= expected_storage,
        "total_cap_is_500": (
            math.isclose(total_cap, 500.0, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(contract_cap, 500.0, rel_tol=0.0, abs_tol=1e-12)
        ),
        "v2_9_incremental_zero_hosted": (
            v29_incremental.get("cost_usd") == 0.0
            and v29_incremental.get("hosted_completions") == 0
            and v29_incremental.get("offline_candidate_admission_cells") == 10
            and v29_incremental.get("scripted_diagnostic_calls") == 48
        ),
        "reserve_not_automatic": (
            carry.get("manual_reserve_automatic_use") is False
        ),
    }
    if not all(binding_checks.values()):
        raise PilotEvidenceError(
            "V2.10 evidence does not preserve its inherited debit/denominator"
        )
    return {
        "source_contract_id": PILOT_V29_CONTRACT_ID,
        "total_cap_usd": total_cap,
        "expected_cumulative_prior": _json_copy(dict(expected)),
        "observed_cumulative_totals": {
            "cost_usd": observed_cost,
            "hosted_completions": observed_completions,
            "storage_bytes": observed_storage,
        },
        "v2_9_incremental": _json_copy(dict(v29_incremental)),
        "automatically_dispatchable_usd_before_v2_10": (
            total_cap - expected_cost - 1.0
        ),
        "manual_reserve_usd": 1.0,
        "checks": binding_checks,
        "pass": True,
    }


def _paired_stage_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage_id: str,
    model_id: str,
    arms: Sequence[str],
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    registered_arms = tuple(str(arm) for arm in arms)
    by_identity: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("stage_id") != stage_id or row.get("model_id") != model_id:
            continue
        identity = (str(row.get("arm_id")), int(row["environment_seed"]))
        by_identity.setdefault(identity, []).append(row)

    seed_rows: dict[str, Any] = {}
    complete_seeds: list[int] = []
    for seed in expected_seeds:
        arm_status: dict[str, Any] = {}
        for arm in registered_arms:
            candidates = by_identity.get((arm, int(seed)), [])
            unique = len(candidates) == 1
            row = candidates[0] if unique else None
            eligible = bool(
                row is not None
                and row.get("status") == "complete"
                and row.get("scientific_eligible") is True
            )
            arm_status[arm] = {
                "ledger_row_count": len(candidates),
                "status": None if row is None else row.get("status"),
                "scientific_eligible": (
                    False if row is None else row.get("scientific_eligible") is True
                ),
                "complete_and_eligible": eligible,
            }
        complete = all(item["complete_and_eligible"] for item in arm_status.values())
        if complete:
            complete_seeds.append(int(seed))
        seed_rows[str(seed)] = {
            "complete_pair": complete,
            "arms": arm_status,
        }
    return {
        "stage_id": stage_id,
        "model_id": model_id,
        "registered_arms": list(registered_arms),
        "expected_seeds": [int(seed) for seed in expected_seeds],
        "complete_paired_seeds": complete_seeds,
        "incomplete_or_failed_seeds": [
            int(seed) for seed in expected_seeds if int(seed) not in complete_seeds
        ],
        "complete_pair_count": len(complete_seeds),
        "required_complete_pair_count": PILOT_V24_MIN_PAIRED_SEEDS,
        "total_registered_pair_count": PILOT_V24_TOTAL_PAIRED_SEEDS,
        "pass": len(complete_seeds) >= PILOT_V24_MIN_PAIRED_SEEDS,
        "seed_rows": seed_rows,
    }


def _v210_sensitivity_lane_definition(lane_id: str) -> dict[str, str]:
    try:
        lane = _V24_LANES[lane_id]
        package_path = _V210_C_SENSITIVITY_FILES[lane_id]
    except KeyError as exc:
        raise PilotEvidenceError(
            f"unknown V2.10 Experiment C sensitivity lane: {lane_id!r}"
        ) from exc
    return {
        "lane_id": lane_id,
        "stage_id": str(lane["stage_ids"]["experiment-c"]),
        "model_id": str(lane["model_id"]),
        "package_path": package_path,
    }


def _validated_v210_sensitivity_controls(
    release_controls: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw = release_controls.get("experiment_c_rule_sensitivities")
    if not isinstance(raw, Mapping) or set(raw) != set(
        _V210_C_SENSITIVITY_FILES
    ):
        raise PilotEvidenceError(
            "V2.10 release controls require both lane-specific Experiment C "
            "rule sensitivities"
        )
    controls: dict[str, dict[str, Any]] = {}
    for lane_id in _V210_C_SENSITIVITY_FILES:
        definition = _v210_sensitivity_lane_definition(lane_id)
        value = raw.get(lane_id)
        if not isinstance(value, Mapping):
            raise PilotEvidenceError(
                f"V2.10 {lane_id} Experiment C sensitivity control is malformed"
            )
        control = dict(value)
        available = control.get("available")
        passed = control.get("pass")
        if (
            control.get("lane_id") != lane_id
            or control.get("stage_id") != definition["stage_id"]
            or control.get("model_id") != definition["model_id"]
            or control.get("package_path") != definition["package_path"]
            or not isinstance(available, bool)
            or not isinstance(passed, bool)
            or passed is not available
            or control.get("provider_calls") != 0
            or control.get("descriptive_only") is not True
            or control.get("effectiveness_gate") is not False
        ):
            raise PilotEvidenceError(
                f"V2.10 {lane_id} Experiment C sensitivity control drifted"
            )
        if passed:
            for field in (
                "path",
                "file_sha256",
                "content_sha256",
                "source_run_count",
                "grid_cell_count",
            ):
                if field not in control:
                    raise PilotEvidenceError(
                        f"V2.10 {lane_id} Experiment C sensitivity control "
                        f"lacks {field!r}"
                    )
            if (
                not isinstance(control["path"], str)
                or not control["path"]
                or not isinstance(control["file_sha256"], str)
                or len(control["file_sha256"]) != 64
                or not isinstance(control["content_sha256"], str)
                or len(control["content_sha256"]) != 64
                or control["source_run_count"] != 5
                or control["grid_cell_count"] != 9
            ):
                raise PilotEvidenceError(
                    f"V2.10 {lane_id} Experiment C sensitivity hashes/counts "
                    "are invalid"
                )
        elif not isinstance(control.get("reason"), str) or not control["reason"]:
            raise PilotEvidenceError(
                f"V2.10 {lane_id} unavailable sensitivity lacks a reason"
            )
        controls[lane_id] = _json_copy(control)
    return controls


def _validated_v210_experiment_c_sensitivities(
    contract: PilotContract,
    *,
    raw_root: Path,
    rows: Sequence[Mapping[str, Any]],
    common_commit: str | None,
    source_repo_root: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Recompute and bind both V2.10 lane-specific zero-API C controls."""

    if contract.contract_id != PILOT_V210_CONTRACT_ID:
        raise PilotEvidenceError(
            "V2.10 Experiment C sensitivity validation requested for another "
            "contract"
        )
    from .pilot_orchestrator import (  # pylint: disable=import-outside-toplevel
        _load_verified_experiment_c_sensitivity,
    )

    payloads: dict[str, dict[str, Any]] = {}
    controls: dict[str, dict[str, Any]] = {}
    sensitivity_contract = contract.stop_go["experiment_c"][
        "zero_api_sensitivity"
    ]
    expected_weights = list(sensitivity_contract["alternative_success_weights"])
    expected_outcomes = list(sensitivity_contract["outcome_definitions"])
    expected_grid = {
        (weight, outcome)
        for weight in expected_weights
        for outcome in expected_outcomes
    }

    with source_repository_context(source_repo_root, raw_root=raw_root):
        for lane_id in _V210_C_SENSITIVITY_FILES:
            definition = _v210_sensitivity_lane_definition(lane_id)
            stage_id = definition["stage_id"]
            model_id = definition["model_id"]
            stage_specs = tuple(
                contract.expand(stage=stage_id, model=model_id)
            )
            expected_stage_ids = {spec.run_id for spec in stage_specs}
            stage_rows = [
                row
                for row in rows
                if row.get("stage_id") == stage_id
                and row.get("model_id") == model_id
            ]
            stage_complete = bool(stage_specs) and bool(
                len(stage_rows) == len(stage_specs)
                and {str(row.get("run_id")) for row in stage_rows}
                == expected_stage_ids
                and all(
                    row.get("status") == "complete"
                    and row.get("scientific_eligible") is True
                    for row in stage_rows
                )
            )
            path = raw_root / stage_id / "rule_sensitivity.json"
            base_control = {
                **definition,
                "provider_calls": 0,
                "descriptive_only": True,
                "effectiveness_gate": False,
            }
            if not stage_complete:
                controls[lane_id] = {
                    **base_control,
                    "pass": False,
                    "available": False,
                    "reason": (
                        f"{stage_id} ITT cells are not all complete and "
                        "scientifically eligible"
                    ),
                }
                continue
            if path.is_symlink() or not path.is_file():
                raise PilotEvidenceError(
                    f"V2.10 {lane_id} Experiment C sensitivity is missing or "
                    f"unsafe: {path}"
                )
            try:
                value = _load_verified_experiment_c_sensitivity(
                    contract,
                    raw_root=raw_root,
                    paid=None,
                    stage_id=stage_id,
                    model_id=model_id,
                    authority_repo_root=source_repo_root,
                )
            except Exception as exc:
                raise PilotEvidenceError(
                    f"V2.10 {lane_id} Experiment C sensitivity failed "
                    f"revalidation: {exc}"
                ) from exc
            bindings = value.get("bindings")
            integrity = value.get("integrity")
            cells = value.get("aggregate_cells")
            observed_grid = (
                {
                    (
                        cell.get("alternative_success_weight"),
                        cell.get("outcome_definition"),
                    )
                    for cell in cells
                    if isinstance(cell, Mapping)
                }
                if isinstance(cells, Sequence)
                and not isinstance(cells, (str, bytes))
                else set()
            )
            source_specs = tuple(
                contract.expand(
                    stage=stage_id,
                    model=model_id,
                    arm="full",
                )
            )
            expected_source_ids = {spec.run_id for spec in source_specs}
            source_rows = {
                str(row.get("run_id")): row
                for row in stage_rows
                if row.get("arm_id") == "full"
            }
            expected_sources = {
                run_id: row.get("artifact_sha256")
                for run_id, row in source_rows.items()
                if row.get("artifact_kind") == "verified-run-manifest"
                and row.get("status") == "complete"
                and row.get("scientific_eligible") is True
            }
            source_manifests = (
                bindings.get("source_manifests")
                if isinstance(bindings, Mapping)
                else None
            )
            observed_sources = (
                {
                    str(source.get("run_id")): source.get("manifest_sha256")
                    for source in source_manifests
                    if isinstance(source, Mapping)
                }
                if isinstance(source_manifests, Sequence)
                and not isinstance(source_manifests, (str, bytes))
                else {}
            )
            if (
                common_commit is None
                or value.get("schema_version")
                != PILOT_EXPERIMENT_C_SENSITIVITY_SCHEMA_VERSION
                or value.get("status") != "pass"
                or value.get("terminal") is not True
                or value.get("control_kind")
                != "zero-api-offline-rule-sensitivity"
                or value.get("provider_calls") != 0
                or value.get("descriptive_only") is not True
                or value.get("effectiveness_gate") is not False
                or value.get("scientific_evidence") is not True
                or value.get("alternative_success_weights") != expected_weights
                or value.get("outcome_definitions") != expected_outcomes
                or not isinstance(cells, Sequence)
                or isinstance(cells, (str, bytes))
                or len(cells) != len(expected_grid)
                or observed_grid != expected_grid
                or not isinstance(bindings, Mapping)
                or bindings.get("contract_sha256") != contract.canonical_hash
                or bindings.get("git_tag")
                != contract.implementation["required_git_tag"]
                or bindings.get("git_commit") != common_commit
                or bindings.get("source_stage") != stage_id
                or bindings.get("source_arm") != "full"
                or len(source_specs) != 5
                or set(expected_sources) != expected_source_ids
                or observed_sources != expected_sources
                or not isinstance(integrity, Mapping)
                or not isinstance(integrity.get("content_sha256"), str)
                or len(str(integrity["content_sha256"])) != 64
            ):
                raise PilotEvidenceError(
                    f"V2.10 {lane_id} Experiment C sensitivity bindings, "
                    "grid, or source manifests drifted"
                )
            control = {
                **base_control,
                "pass": True,
                "available": True,
                "path": str(path),
                "file_sha256": _sha256_file(path),
                "content_sha256": integrity["content_sha256"],
                "source_run_count": value.get("source_run_count"),
                "grid_cell_count": len(cells),
            }
            controls[lane_id] = control
            payloads[lane_id] = {
                "payload": _json_copy(value),
                "control": _json_copy(control),
            }
    return payloads, _validated_v210_sensitivity_controls(
        {
            "experiment_c_rule_sensitivities": controls,
        }
    )


def _lane_aggregate(
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
    *,
    lane_id: str,
) -> dict[str, Any]:
    lane = _V24_LANES[lane_id]
    model_id = str(lane["model_id"])
    stage_ids = lane["stage_ids"]
    arms = lane["arms"]
    expected_seeds = tuple(int(seed) for seed in contract.seeds["sets"]["main"])
    paired_gates = {
        canonical: _paired_stage_gate(
            rows,
            stage_id=str(stage_ids[canonical]),
            model_id=model_id,
            arms=arms[canonical],
            expected_seeds=expected_seeds,
        )
        for canonical in PILOT_V24_STAGE_ORDER
    }
    detailed = {
        "experiment-c": _experiment_c_gate(
            contract,
            rows,
            stage_id=str(stage_ids["experiment-c"]),
            model_id=model_id,
        ),
        "experiment-a": _experiment_a_gate(
            contract,
            rows,
            stage_id=str(stage_ids["experiment-a"]),
            model_id=model_id,
        ),
        "experiment-d": _experiment_d_gate(
            contract,
            rows,
            stage_id=str(stage_ids["experiment-d"]),
            model_id=model_id,
            arms=arms["experiment-d"],
        ),
        "experiment-b": _experiment_b_summary(
            rows,
            stage_id=str(stage_ids["experiment-b"]),
            model_id=model_id,
            arms=arms["experiment-b"],
        ),
    }
    detailed["experiment-b"] = {
        **detailed["experiment-b"],
        "status": (
            "descriptive-complete" if paired_gates["experiment-b"]["pass"] else "no-go"
        ),
        "scientific_evidence_complete": paired_gates["experiment-b"]["pass"],
        "claim_action": (
            "report the registered architecture comparison descriptively; "
            "do not select a winner by wealth"
            if paired_gates["experiment-b"]["pass"]
            else "report the incomplete architecture denominator without a winner"
        ),
    }
    paired_matrix_pass = all(gate["pass"] for gate in paired_gates.values())
    effect_claims_supported = all(
        detailed[stage]["status"] == "supported"
        for stage in ("experiment-c", "experiment-a", "experiment-d")
    )
    return {
        "lane_id": lane_id,
        "model_id": model_id,
        "stage_order": list(PILOT_V24_STAGE_ORDER),
        "stage_ids": {
            canonical: str(stage_ids[canonical]) for canonical in PILOT_V24_STAGE_ORDER
        },
        "paired_seed_gates": paired_gates,
        "paired_matrix_complete": paired_matrix_pass,
        "effect_claims_supported": effect_claims_supported,
        "gates": detailed,
        "direction_count_scope": (
            f"{lane_id}-only; no direction count is pooled across backbones"
        ),
    }


def _finite_direction(values: Any) -> str | None:
    if not isinstance(values, Mapping) or not values:
        return None
    normalized = []
    for value in values.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return None
        normalized.append(float(value))
    center = median(normalized)
    return "positive" if center > 0 else "negative" if center < 0 else "zero"


def _mechanism_direction(
    stage: str,
    gate: Mapping[str, Any],
) -> Any:
    if gate.get("status") != "supported":
        return None
    if stage == "experiment-c":
        return "verified-lowers-registered-harm"
    if stage == "experiment-a":
        primary = gate.get("primary_contrast")
        return (
            _finite_direction(primary.get("raw_paired_deltas"))
            if isinstance(primary, Mapping)
            else None
        )
    if stage == "experiment-d":
        treatment_gates = gate.get("treatment_gates")
        supported = gate.get("supported_treatments")
        if (
            not isinstance(treatment_gates, Mapping)
            or not isinstance(supported, Sequence)
            or isinstance(supported, (str, bytes))
        ):
            return None
        directions: dict[str, str] = {}
        for treatment in supported:
            treatment_gate = treatment_gates.get(str(treatment))
            if not isinstance(treatment_gate, Mapping):
                continue
            utility_gate = treatment_gate.get("six_step_discounted_utility_gate")
            direction = (
                _finite_direction(utility_gate.get("treatment_deltas"))
                if isinstance(utility_gate, Mapping)
                else None
            )
            if direction is not None:
                directions[str(treatment)] = direction
        return directions or None
    return None


def _cross_lane_mechanism_comparison(
    lanes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for stage in PILOT_V24_STAGE_ORDER:
        local = lanes["local"]
        hosted = lanes["gpt52"]
        local_gate = local["gates"][stage]
        hosted_gate = hosted["gates"][stage]
        local_paired = local["paired_seed_gates"][stage]
        hosted_paired = hosted["paired_seed_gates"][stage]
        local_direction = _mechanism_direction(stage, local_gate)
        hosted_direction = _mechanism_direction(stage, hosted_gate)
        both_supported = bool(
            local_paired["pass"]
            and hosted_paired["pass"]
            and local_gate["status"] == "supported"
            and hosted_gate["status"] == "supported"
        )
        common_registered_treatments: list[str] = []
        local_only_registered_treatments: list[str] = []
        gpt52_only_registered_treatments: list[str] = []
        common_direction_qualified_treatments: list[str] = []
        if stage == "experiment-d":
            excluded_controls = {"matched-a", "matched-b"}
            local_registered = set(local_paired["registered_arms"]) - excluded_controls
            hosted_registered = (
                set(hosted_paired["registered_arms"]) - excluded_controls
            )
            common_registered_treatments = sorted(local_registered & hosted_registered)
            local_only_registered_treatments = sorted(
                local_registered - hosted_registered
            )
            gpt52_only_registered_treatments = sorted(
                hosted_registered - local_registered
            )
            local_directions = (
                dict(local_direction) if isinstance(local_direction, Mapping) else {}
            )
            hosted_directions = (
                dict(hosted_direction) if isinstance(hosted_direction, Mapping) else {}
            )
            common_direction_qualified_treatments = sorted(
                set(common_registered_treatments)
                & set(local_directions)
                & set(hosted_directions)
            )
            directions_known = bool(common_direction_qualified_treatments)
            interaction = bool(
                both_supported
                and any(
                    local_directions[treatment] != hosted_directions[treatment]
                    for treatment in common_direction_qualified_treatments
                )
            )
            same_direction = bool(
                both_supported and directions_known and not interaction
            )
        else:
            directions_known = (
                local_direction is not None and hosted_direction is not None
            )
            same_direction = bool(
                both_supported
                and directions_known
                and local_direction == hosted_direction
            )
            interaction = bool(
                both_supported
                and directions_known
                and local_direction != hosted_direction
            )
        classification = (
            "same-direction-in-two-backbone-micro-pilots"
            if same_direction
            else "backbone-interaction" if interaction else "inconclusive"
        )
        rows.append(
            {
                "stage": stage,
                "local_status": local_gate["status"],
                "gpt52_status": hosted_gate["status"],
                "local_4_of_5_pass": local_paired["pass"],
                "gpt52_4_of_5_pass": hosted_paired["pass"],
                "local_direction": _json_copy(local_direction),
                "gpt52_direction": _json_copy(hosted_direction),
                "common_registered_treatments": (common_registered_treatments),
                "local_only_registered_treatments": (local_only_registered_treatments),
                "gpt52_only_registered_treatments": (gpt52_only_registered_treatments),
                "common_direction_qualified_treatments": (
                    common_direction_qualified_treatments
                ),
                "direction_agreement": same_direction,
                "classification": classification,
                "claim_boundary": (
                    (
                        "the registered direction appeared separately in the "
                        "local and GPT-5.2 micro-pilots; this is not a "
                        "backbone-independent claim"
                    )
                    if same_direction
                    else (
                        (
                            "report a possible backbone interaction; do not pool "
                            "seed directions"
                        )
                        if interaction
                        else (
                            "cross-backbone mechanism direction is inconclusive; "
                            "do not pool seed directions"
                        )
                    )
                ),
            }
        )
    return {
        "aggregation_policy": (
            "compare lane-level registered directions only; never add or pool "
            "seed direction counts"
        ),
        "direction_counts_merged": False,
        "rows": rows,
    }


def _claim_rows(
    contract: PilotContract,
    lanes: Mapping[str, Mapping[str, Any]],
    *,
    denominator: Mapping[str, Any],
    cross_lane: Mapping[str, Any],
) -> list[dict[str, Any]]:
    version_label = _contract_version_label(contract)
    claims: list[dict[str, Any]] = []
    definitions = {
        "experiment-c": (
            "Evidence grounding improves erroneous-rule reliability",
            "false activation, harmful exposure, and cumulative utility-loss directions",
        ),
        "experiment-a": (
            "M1 retrieval contributes beyond regime prompting",
            "full minus prompt-only shock+recovery discounted utility",
        ),
        "experiment-d": (
            "A focal memory/error pulse changes the matched six-step continuation",
            "matched-null- and action-bin-qualified continuation deltas",
        ),
        "experiment-b": (
            "Registered memory architectures can be compared descriptively",
            "seed-level utility, action, retrieval, proposal, and lifecycle summaries",
        ),
    }
    for lane_id, lane in lanes.items():
        for stage in PILOT_V24_STAGE_ORDER:
            gate = lane["gates"][stage]
            claim, metric = definitions[stage]
            claims.append(
                {
                    "lane": lane_id,
                    "claim": claim,
                    "metric": metric,
                    "artifact": f"aggregate.json#/lanes/{lane_id}/gates/{stage}",
                    "status": gate["status"],
                    "boundary": gate["claim_action"],
                }
            )
    for comparison in cross_lane["rows"]:
        claims.append(
            {
                "lane": "cross-lane",
                "claim": (
                    f"{comparison['stage']} direction appears in two "
                    "backbone micro-pilots"
                ),
                "metric": (
                    "separate lane-level 4/5 gate, mechanism status, and "
                    "primary-direction agreement"
                ),
                "artifact": (
                    "aggregate.json#/cross_lane_mechanism_comparison/"
                    f"{comparison['stage']}"
                ),
                "status": comparison["classification"],
                "boundary": comparison["claim_boundary"],
            }
        )
    claims.extend(
        [
            {
                "lane": "not-applicable",
                "claim": "Narrative channel shows controlled semantic response",
                "metric": f"not registered in the {version_label} core matrix",
                "artifact": "aggregate.json#/narrative",
                "status": "deferred-unregistered",
                "boundary": (
                    f"make no {version_label} narrative or "
                    "real-news-understanding claim"
                ),
            },
            {
                "lane": "cross-lane",
                "claim": "Backbone-independent improvement",
                "metric": "prohibited pooled inference",
                "artifact": "aggregate.json#/cross_lane_policy",
                "status": "prohibited",
                "boundary": (
                    "report local and GPT-5.2 directions separately; never pool "
                    "direction counts or use backbone-independent wording"
                ),
            },
            {
                "lane": "all",
                "claim": f"Complete {version_label} preregistered ITT denominator",
                "metric": "one terminal ledger row for every expanded contract cell",
                "artifact": "failure_ledger.json",
                "status": "supported" if denominator.get("pass") else "no-go",
                "boundary": (
                    "retain every failed, stopped, nonterminal, and missing cell"
                ),
            },
        ]
    )
    return claims


def _claim_narrowing(
    lanes: Mapping[str, Mapping[str, Any]],
    *,
    denominator: Mapping[str, Any],
    release_controls: Mapping[str, Any],
    cross_lane: Mapping[str, Any],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for lane_id, lane in lanes.items():
        for stage in PILOT_V24_STAGE_ORDER:
            gate = lane["gates"][stage]
            paired = lane["paired_seed_gates"][stage]
            if not paired["pass"]:
                output.append(
                    {
                        "scope": f"{lane_id}/{stage}",
                        "reason": (
                            f"{paired['complete_pair_count']}/5 complete paired "
                            "seeds; the registered minimum is 4/5"
                        ),
                        "required_wording": (
                            "denominator/failure report only; no effectiveness claim"
                        ),
                    }
                )
            elif gate["status"] not in {"supported", "descriptive-complete"}:
                output.append(
                    {
                        "scope": f"{lane_id}/{stage}",
                        "reason": "the preregistered mechanism gate was not supported",
                        "required_wording": str(gate["claim_action"]),
                    }
                )
    if denominator.get("pass") is not True:
        output.append(
            {
                "scope": "full-denominator",
                "reason": "one or more registered ITT rows are missing or nonterminal",
                "required_wording": (
                    "incomplete; do not publish an immutable evidence package"
                ),
            }
        )
    if release_controls.get("pass") is not True:
        output.append(
            {
                "scope": "release-stage0-budget",
                "reason": "release, Stage-0, or budget controls did not all pass",
                "required_wording": (
                    "complete-with-no-go; do not report scientific effectiveness"
                ),
            }
        )
    for comparison in cross_lane["rows"]:
        if (
            comparison["classification"]
            != "same-direction-in-two-backbone-micro-pilots"
        ):
            output.append(
                {
                    "scope": f"cross-lane/{comparison['stage']}",
                    "reason": (
                        "the two independently gated lanes do not establish "
                        "the same registered primary direction"
                    ),
                    "required_wording": comparison["claim_boundary"],
                }
            )
    output.extend(
        [
            {
                "scope": "narrative",
                "reason": "narrative intervention is deferred and unregistered",
                "required_wording": ("no narrative or real-news-understanding claim"),
            },
            {
                "scope": "cross-lane",
                "reason": "the local and GPT lanes are separate replications",
                "required_wording": (
                    "report each lane's seed directions separately; never pool them"
                ),
            },
        ]
    )
    return output


def aggregate_v24_evidence(
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
    *,
    denominator: Mapping[str, Any],
    release_controls: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the pure lane-separated aggregate without artifact I/O."""

    _validate_v24_contract_matrix(contract)
    version_label = _contract_version_label(contract)
    schema_version = _evidence_schema_version(contract)
    v210_sensitivity_controls: dict[str, dict[str, Any]] | None = None
    if contract.contract_id == PILOT_V210_CONTRACT_ID:
        v210_sensitivity_controls = _validated_v210_sensitivity_controls(
            release_controls
        )
        effective_release_controls = _json_copy(release_controls)
        effective_release_controls["experiment_c_rule_sensitivities"] = (
            v210_sensitivity_controls
        )
        effective_release_controls["pass"] = bool(
            release_controls.get("pass") is True
            and all(
                control["pass"]
                for control in v210_sensitivity_controls.values()
            )
        )
        release_controls = effective_release_controls
    imported_prerequisites = _v27_imported_prerequisite_summary(
        contract,
        rows,
    )
    itt_row_preservation = _v28_itt_row_preservation(
        contract,
        rows,
        denominator=denominator,
    )
    if itt_row_preservation is None:
        itt_row_preservation = _v29_itt_row_preservation(
            contract,
            rows,
            denominator=denominator,
        )
    if itt_row_preservation is None:
        itt_row_preservation = _v210_itt_row_preservation(
            contract,
            rows,
            denominator=denominator,
        )
    prerequisites = _v28_prerequisite_summary(contract, rows)
    if prerequisites is None:
        prerequisites = _v29_prerequisite_summary(contract, rows)
    if prerequisites is None:
        prerequisites = _v210_prerequisite_summary(contract, rows)
    parent_evidence_lineage = _v28_parent_evidence_lineage(contract)
    if parent_evidence_lineage is None:
        parent_evidence_lineage = _v29_parent_evidence_lineage(contract)
    if parent_evidence_lineage is None:
        parent_evidence_lineage = _v210_parent_evidence_lineage(contract)
    inherited_budget_boundary = _v27_inherited_budget_boundary(
        contract,
        denominator=denominator,
        release_controls=release_controls,
    )
    if inherited_budget_boundary is None:
        inherited_budget_boundary = _v28_inherited_budget_boundary(
            contract,
            denominator=denominator,
            release_controls=release_controls,
        )
    if inherited_budget_boundary is None:
        inherited_budget_boundary = _v29_inherited_budget_boundary(
            contract,
            denominator=denominator,
            release_controls=release_controls,
        )
    if inherited_budget_boundary is None:
        inherited_budget_boundary = _v210_inherited_budget_boundary(
            contract,
            denominator=denominator,
            release_controls=release_controls,
        )
    prerequisite_stage_ids: set[str] = set()
    if contract.contract_id == PILOT_V27_CONTRACT_ID:
        prerequisite_stage_ids = set(_V27_IMPORTED_PREREQUISITE_COUNTS)
    elif contract.contract_id == PILOT_V28_CONTRACT_ID:
        prerequisite_stage_ids = set(_V28_PREREQUISITE_COUNTS)
    elif contract.contract_id == PILOT_V29_CONTRACT_ID:
        prerequisite_stage_ids = set(_V29_PREREQUISITE_COUNTS)
    elif contract.contract_id == PILOT_V210_CONTRACT_ID:
        prerequisite_stage_ids = set(_V210_PREREQUISITE_COUNTS)
    effect_rows = (
        [row for row in rows if row.get("stage_id") not in prerequisite_stage_ids]
        if prerequisite_stage_ids
        else list(rows)
    )
    lanes = {
        lane_id: _lane_aggregate(
            contract,
            effect_rows,
            lane_id=lane_id,
        )
        for lane_id in ("local", "gpt52")
    }
    cross_lane = _cross_lane_mechanism_comparison(lanes)
    claims = _claim_rows(
        contract,
        lanes,
        denominator=denominator,
        cross_lane=cross_lane,
    )
    if v210_sensitivity_controls is not None:
        for lane_id, control in v210_sensitivity_controls.items():
            claims.append(
                {
                    "lane": lane_id,
                    "claim": (
                        "registered zero-API Experiment C rule sensitivity is "
                        "available for this lane"
                    ),
                    "metric": (
                        "3 alternative-success weights x 3 outcome definitions "
                        "replayed from five full-control seeds"
                    ),
                    "artifact": control["package_path"],
                    "status": (
                        "complete-descriptive"
                        if control["pass"]
                        else "no-go"
                    ),
                    "boundary": (
                        "descriptive sensitivity over natural proposals only; "
                        "it cannot rescue a failed Experiment C effectiveness "
                        "contrast"
                    ),
                }
            )
    if imported_prerequisites is not None:
        claims.append(
            {
                "lane": "calibration-prerequisite",
                "claim": (
                    "V2.7 parent, q-ref, and Stage-0 imports are exact " "prerequisites"
                ),
                "metric": (
                    "16 registered imported cells with per-stage terminal "
                    "status and A-D exclusion"
                ),
                "artifact": "aggregate.json#/imported_prerequisites",
                "status": (
                    "complete"
                    if imported_prerequisites["all_imported_complete"]
                    else "no-go"
                ),
                "boundary": imported_prerequisites["claim_boundary"],
            }
        )
    if prerequisites is not None:
        parent_version = {
            PILOT_V28_CONTRACT_ID: "V2.7",
            PILOT_V29_CONTRACT_ID: "V2.8",
            PILOT_V210_CONTRACT_ID: "V2.9",
        }[contract.contract_id]
        prerequisite_claim = (
            f"{version_label} imports the exact parent, q-ref, and Stage-0 "
            "prerequisites without provider dispatch"
            if contract.contract_id == PILOT_V210_CONTRACT_ID
            else (
                f"{version_label} parent authority, fresh scripted q-ref, and "
                "imported Stage-0 inputs are complete prerequisites"
            )
        )
        prerequisite_metric = (
            "16 registered imported prerequisite cells; 0 provider calls "
            "during import; 195 fresh A-D cells required"
            if contract.contract_id == PILOT_V210_CONTRACT_ID
            else (
                "16 registered prerequisite cells; q-ref 0 hosted / "
                "48 scripted diagnostic calls; 14 imported Stage-0 cells"
            )
        )
        claims.extend(
            [
                {
                    "lane": "prerequisite-non-effect",
                    "claim": prerequisite_claim,
                    "metric": prerequisite_metric,
                    "artifact": "aggregate.json#/prerequisites",
                    "status": (
                        "complete"
                        if prerequisites["all_prerequisites_complete"]
                        else "no-go"
                    ),
                    "boundary": prerequisites["claim_boundary"],
                },
                {
                    "lane": "parent-lineage",
                    "claim": (
                        f"{parent_version} remains an immutable "
                        "complete-with-no-go package"
                    ),
                    "metric": (
                        "exact parent manifest hash, terminal denominator, "
                        "evidence commit, and merge commit"
                    ),
                    "artifact": "parent_evidence_reference.json",
                    "status": "preserved",
                    "boundary": (
                        "reference only; do not rewrite, reclassify, or import "
                        f"{parent_version} rows into {version_label} effects"
                    ),
                },
            ]
        )
    narrowing = _claim_narrowing(
        lanes,
        denominator=denominator,
        release_controls=release_controls,
        cross_lane=cross_lane,
    )
    scientific_matrix_complete = bool(
        denominator.get("pass") is True
        and release_controls.get("pass") is True
        and (
            prerequisites is None or prerequisites["all_prerequisites_complete"] is True
        )
        and all(lane["paired_matrix_complete"] for lane in lanes.values())
    )
    scientific_claim_gates_supported = all(
        lane["effect_claims_supported"] for lane in lanes.values()
    )
    scientific_complete = bool(
        scientific_matrix_complete and scientific_claim_gates_supported
    )
    denominator_terminal = denominator.get("pass") is True
    publication_status = (
        "incomplete"
        if not denominator_terminal
        else "complete" if scientific_complete else "complete-with-no-go"
    )
    aggregate = {
        "schema_version": schema_version,
        "evidence_namespace": _evidence_namespace(contract),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "pilot_tag": contract.implementation["required_git_tag"],
        "fixed_matrix_order": list(PILOT_V24_STAGE_ORDER),
        "denominator": _json_copy(denominator),
        "budget": _json_copy(release_controls.get("budget_ledger", {})),
        "release_controls": _json_copy(release_controls),
        "lanes": _json_copy(lanes),
        "cross_lane_mechanism_comparison": _json_copy(cross_lane),
        "cross_lane_policy": {
            "direction_counts_merged": False,
            "effect_estimates_pooled": False,
            "allowed_interpretation": (
                "separate local and GPT-5.2 mechanism-pilot directions only"
            ),
            "prohibited_wording": "backbone-independent",
        },
        "narrative": {
            "status": "deferred-unregistered",
            "registered_cells": 0,
            "claim_boundary": (
                f"no {version_label} narrative or real-news-understanding claim"
            ),
        },
        "claims": claims,
        "claim_narrowing": narrowing,
        "scientific_matrix_complete": scientific_matrix_complete,
        "scientific_claim_gates_supported": scientific_claim_gates_supported,
        "scientific_complete": scientific_complete,
        "publication_status": publication_status,
    }
    if imported_prerequisites is not None:
        aggregate["imported_prerequisites"] = imported_prerequisites
    if prerequisites is not None:
        aggregate["prerequisites"] = prerequisites
        aggregate["effect_aggregation_scope"] = {
            "included_stage_ids": sorted(
                {
                    str(row.get("stage_id"))
                    for row in effect_rows
                    if str(row.get("stage_id")).endswith(tuple(PILOT_V24_STAGE_ORDER))
                }
            ),
            "prerequisite_stage_ids_excluded": sorted(prerequisite_stage_ids),
            {
                PILOT_V28_CONTRACT_ID: "v2_8_a_d_cells_only",
                PILOT_V29_CONTRACT_ID: "v2_9_a_d_cells_only",
                PILOT_V210_CONTRACT_ID: "fresh_v2_10_a_d_cells_only",
            }[contract.contract_id]: True,
        }
    if itt_row_preservation is not None:
        aggregate["itt_row_preservation"] = itt_row_preservation
    if parent_evidence_lineage is not None:
        aggregate["parent_evidence_lineage"] = parent_evidence_lineage
    if inherited_budget_boundary is not None:
        aggregate["inherited_budget_boundary"] = inherited_budget_boundary
    if v210_sensitivity_controls is not None:
        aggregate["experiment_c_rule_sensitivities"] = (
            v210_sensitivity_controls
        )
    return aggregate


def _require_publishable_terminal_denominator(
    aggregate: Mapping[str, Any],
) -> None:
    if aggregate.get("publication_status") == "incomplete":
        version_label = _contract_id_version_label(aggregate.get("contract_id"))
        denominator = aggregate.get("denominator")
        status_counts = (
            denominator.get("status_counts")
            if isinstance(denominator, Mapping)
            else None
        )
        raise PilotEvidenceError(
            f"{version_label} immutable evidence publication requires all "
            "211 ITT cells "
            f"to be present and terminal; status_counts={status_counts!r}"
        )


def _sanitized_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fields = (
        "run_id",
        "contract_id",
        "stage_id",
        "model_id",
        "requested_model",
        "arm_id",
        "narrative_id",
        "environment_seed",
        "decoding_seed",
        "utility_profile_id",
        "shock_id",
        "budget_bucket",
        "num_agents",
        "episode_length",
        "execution_mode",
        "status",
        "failure",
        "artifact_kind",
        "artifact_sha256",
        "scientific_eligible",
        "metrics",
        "gate_evidence",
        "capability",
        "narrative",
    )
    return [{field: _json_copy(row.get(field)) for field in fields} for row in rows]


def _v29_implementation_failure_summary(
    aggregate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    resolved_git_commit: str | None,
) -> dict[str, Any] | None:
    """Describe the terminal V2.9 adapter failure from sealed rows and source."""

    if aggregate.get("contract_id") != PILOT_V29_CONTRACT_ID:
        return None
    failures = [row for row in rows if row.get("status") != "complete"]
    stage_counts: dict[str, int] = {}
    for row in failures:
        stage_id = row.get("stage_id")
        if not isinstance(stage_id, str):
            raise PilotEvidenceError("V2.9 failed row lacks a stage identity")
        stage_counts[stage_id] = stage_counts.get(stage_id, 0) + 1
    failure_signatures = {
        (
            failure.get("error_type"),
            failure.get("message"),
            failure.get("message_sha256"),
            failure.get("message_truncated"),
        )
        for row in failures
        for failure in [row.get("failure")]
        if isinstance(failure, Mapping)
    }
    offline_candidates = [
        row
        for row in rows
        if row.get("status") == "complete"
        and row.get("arm_id") == "verified-error-candidate"
        and row.get("stage_id") in {"local-experiment-c", "experiment-c"}
    ]
    budget = aggregate.get("budget")
    stage_costs = (
        budget.get("actual_stage_cost_usd") if isinstance(budget, Mapping) else None
    )
    expected_signature = {
        (
            "KeyError",
            "'receipt_path'",
            _PILOT_V29_RECEIPT_PATH_FAILURE_SHA256,
            False,
        )
    }
    if (
        resolved_git_commit != _PILOT_V29_RELEASE_COMMIT
        or len(failures) != 185
        or stage_counts != _PILOT_V29_FAILURE_STAGE_COUNTS
        or failure_signatures != expected_signature
        or len(offline_candidates) != 10
        or not isinstance(stage_costs, Mapping)
        or stage_costs.get("hosted_confirmatory") != 0.0
        or stage_costs.get("local") != 0.0
        or aggregate.get("scientific_matrix_complete") is not False
        or aggregate.get("scientific_claim_gates_supported") is not False
    ):
        raise PilotEvidenceError(
            "V2.9 implementation-failure summary differs from the sealed "
            "denominator, budget ledger, or release source"
        )
    return {
        "schema_version": PILOT_V29_IMPLEMENTATION_FAILURE_SCHEMA_VERSION,
        "classification": "implementation-interface-no-go",
        "root_cause_code": "imported-p95-runner-binding-shape-mismatch",
        "resolved_git_commit": resolved_git_commit,
        "observed_failure": {
            "error_type": "KeyError",
            "message": "'receipt_path'",
            "message_sha256": _PILOT_V29_RECEIPT_PATH_FAILURE_SHA256,
            "failed_cell_count": 185,
            "failed_stage_counts": dict(sorted(stage_counts.items())),
        },
        "provider_boundary": {
            "failure_phase": "before-provider-construction-and-dispatch",
            "v2_9_local_stage_cost_usd": 0.0,
            "v2_9_hosted_stage_cost_usd": 0.0,
            "v2_9_hosted_completions": 0,
            "partial_actor_streams_persisted": False,
        },
        "outcome_boundary": {
            "actor_action_utility_rule_exposure_outcomes_generated": False,
            "offline_candidate_admission_cells_generated": 10,
            "all_a_d_outcomes_unobserved": False,
            "claim": (
                "V2.9 produced no actor treatment-effect outcome; its ten "
                "offline candidate-admission outcomes remain reported and "
                "must not be described as unobserved."
            ),
        },
        "source_audit": {
            **_json_copy(_PILOT_V29_SOURCE_AUDIT),
            "diagnosis": (
                "The imported authority producer returned nested receipt "
                "identity fields while the runner consumer dereferenced the "
                "legacy flat names."
            ),
        },
        "evidence_use": (
            "terminal implementation failure and amendment provenance only; "
            "not model-capability or A-D treatment-effect evidence"
        ),
    }


def _implementation_failure_summary_for_contract(
    aggregate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    resolved_git_commit: str | None,
) -> dict[str, Any] | None:
    """Keep the terminal V2.9 implementation diagnosis scoped to V2.9."""

    if aggregate.get("contract_id") != PILOT_V29_CONTRACT_ID:
        return None
    return _v29_implementation_failure_summary(
        aggregate,
        rows,
        resolved_git_commit=resolved_git_commit,
    )


def _report_markdown(
    aggregate: Mapping[str, Any],
) -> str:
    version_label = _contract_id_version_label(aggregate.get("contract_id"))
    lines = [
        f"# FinEvo {version_label} local-first mechanism pilot evidence report",
        "",
        f"- Contract: `{aggregate['contract_id']}` / "
        f"`{aggregate['contract_sha256']}`",
        f"- Publication status: `{aggregate['publication_status']}`",
        f"- Registered denominator: "
        f"`{aggregate['denominator']['expected_count']}` cells",
        "- Matrix order: `C -> A -> D -> B` in each lane.",
        "- Local and GPT-5.2 directions are never pooled.",
        "- Narrative intervention: `deferred-unregistered`.",
        "",
        "## Claim -> metric -> artifact",
        "",
        "| Lane | Claim | Metric | Artifact | Status | Boundary |",
        "|---|---|---|---|---|---|",
    ]
    for claim in aggregate["claims"]:
        lines.append(
            "| "
            + " | ".join(
                str(claim[field]).replace("|", "\\|")
                for field in (
                    "lane",
                    "claim",
                    "metric",
                    "artifact",
                    "status",
                    "boundary",
                )
            )
            + " |"
        )
    if aggregate.get("contract_id") in {
        PILOT_V28_CONTRACT_ID,
        PILOT_V29_CONTRACT_ID,
        PILOT_V210_CONTRACT_ID,
    }:
        lineage = aggregate["parent_evidence_lineage"]
        prerequisites = aggregate["prerequisites"]
        inherited_budget = aggregate["inherited_budget_boundary"]
        version_label = _contract_id_version_label(aggregate["contract_id"])
        parent_version = {
            PILOT_V28_CONTRACT_ID: "V2.7",
            PILOT_V29_CONTRACT_ID: "V2.8",
            PILOT_V210_CONTRACT_ID: "V2.9",
        }[aggregate["contract_id"]]
        prerequisite_description = (
            "The 16 hash-verified V2.9 parent/q-ref/Stage-0 prerequisites "
            "used 0 provider calls during V2.10 import and are excluded from "
            "all A-D gates; every one of the 195 V2.10 A-D cells is fresh."
            if aggregate["contract_id"] == PILOT_V210_CONTRACT_ID
            else (
                "Parent authority, fresh scripted q-ref, and 14 imported "
                "Stage-0 cells are excluded from every A-D treatment-effect gate."
            )
        )
        lines.extend(
            [
                "",
                f"## {version_label} amendment lineage and prerequisite boundary",
                "",
                f"- {parent_version} remains an immutable "
                "`complete-with-no-go` package; "
                f"namespace `{lineage['source_evidence_namespace']}`, evidence "
                f"commit `{lineage['source_evidence_commit']}`, merge commit "
                f"`{lineage['source_evidence_merge_commit']}`.",
                f"- {parent_version} root cause: "
                f"`{lineage['root_cause']['code']}` — "
                f"{lineage['root_cause']['message']}.",
                "- Parent denominator preserved: "
                f"`{lineage['parent_registered_cells']}` cells / "
                f"`{json.dumps(lineage['parent_status_counts'], sort_keys=True)}`.",
                f"- Cumulative hosted budget before new {version_label} dispatch: "
                f"`${inherited_budget['expected_cumulative_prior']['cost_usd']}` / "
                f"`{inherited_budget['expected_cumulative_prior']['hosted_completions']}` "
                f"hosted completions, under the `${inherited_budget['total_cap_usd']}` "
                "hard cap.",
                (
                    "- Parent outcome boundary: V2.9 generated no actor "
                    "treatment-effect outcome; its 10 offline "
                    "candidate-admission outcomes are disclosed but are not "
                    "V2.10 effect evidence."
                    if aggregate["contract_id"] == PILOT_V210_CONTRACT_ID
                    else (
                        "- Fresh q-ref accounting: `0` hosted provider calls, "
                        "`$0` hosted cost, and `48` scripted diagnostic calls."
                    )
                ),
                f"- Prerequisite classification: {prerequisite_description}",
                f"- All prerequisites complete: "
                f"`{str(prerequisites['all_prerequisites_complete']).lower()}`.",
            ]
        )
    implementation_failure = aggregate.get("implementation_failure")
    if implementation_failure is not None:
        observed = implementation_failure["observed_failure"]
        boundary = implementation_failure["provider_boundary"]
        outcome = implementation_failure["outcome_boundary"]
        lines.extend(
            [
                "",
                "## Terminal implementation failure",
                "",
                f"- Classification: " f"`{implementation_failure['classification']}`.",
                f"- Root cause: " f"`{implementation_failure['root_cause_code']}`.",
                f"- All `{observed['failed_cell_count']}` failed A-D cells "
                f"recorded `{observed['error_type']}: "
                f"{observed['message']}`.",
                "- Source audit: the imported-p95 producer returned nested "
                "`authority.path/file_sha256/content_sha256` plus "
                "`source_git_commit`, while `_runner_p95_reservations` "
                "dereferenced the legacy flat receipt fields.",
                f"- Provider boundary: `{boundary['failure_phase']}`; V2.9 "
                "local and hosted stage cost were both `$0`, with `0` hosted "
                "completions.",
                "- Outcome boundary: no actor action, utility, or rule-exposure "
                "outcome was generated. The "
                f"`{outcome['offline_candidate_admission_cells_generated']}` "
                "offline candidate-admission outcomes were generated and "
                "remain in the denominator.",
                "- Evidence use: implementation/amendment provenance only; "
                "this is not a model-capability failure or a negative A-D "
                "effect result.",
            ]
        )
    for lane_id, lane in aggregate["lanes"].items():
        lines.extend(
            [
                "",
                f"## {lane_id} lane",
                "",
                f"- Model profile: `{lane['model_id']}`",
                f"- 4/5 paired matrix complete: "
                f"`{str(lane['paired_matrix_complete']).lower()}`",
            ]
        )
        for stage in PILOT_V24_STAGE_ORDER:
            paired = lane["paired_seed_gates"][stage]
            gate = lane["gates"][stage]
            lines.append(
                f"- `{stage}`: {paired['complete_pair_count']}/5 complete "
                f"paired seeds; gate `{gate['status']}`; "
                f"claim action: {gate['claim_action']}."
            )
    lines.extend(
        [
            "",
            "## Cross-lane mechanism comparison",
            "",
            "| Stage | Local status / 4-of-5 | GPT-5.2 status / 4-of-5 | "
            "Direction agreement | Classification | Boundary |",
            "|---|---|---|---|---|---|",
        ]
    )
    for comparison in aggregate["cross_lane_mechanism_comparison"]["rows"]:
        lines.append(
            f"| `{comparison['stage']}` | "
            f"`{comparison['local_status']}` / "
            f"`{str(comparison['local_4_of_5_pass']).lower()}` | "
            f"`{comparison['gpt52_status']}` / "
            f"`{str(comparison['gpt52_4_of_5_pass']).lower()}` | "
            f"`{str(comparison['direction_agreement']).lower()}` | "
            f"`{comparison['classification']}` | "
            f"{comparison['claim_boundary']} |"
        )
    lines.extend(
        [
            "",
            "## Denominator, failures, and budget",
            "",
            f"- ITT denominator pass: "
            f"`{str(aggregate['denominator']['pass']).lower()}`",
            f"- Status counts: "
            f"`{json.dumps(aggregate['denominator']['status_counts'], sort_keys=True)}`",
            f"- Budget control: "
            f"`{str(bool(aggregate['budget'].get('pass'))).lower()}`",
            "- Every failed, stopped, nonterminal, and missing cell remains in "
            "`failure_ledger.json` and the aggregate rows.",
            "",
            "## Explicit claim narrowing",
            "",
        ]
    )
    for item in aggregate["claim_narrowing"]:
        lines.append(
            f"- `{item['scope']}`: {item['reason']}; " f"{item['required_wording']}."
        )
    return "\n".join(lines) + "\n"


def _validated_v28_parent_evidence_reference(
    contract: PilotContract,
    *,
    contract_path: Path,
) -> dict[str, Any] | None:
    """Revalidate V2.7 in place and return a reference, never a copied package."""

    lineage = _v28_parent_evidence_lineage(contract)
    if lineage is None:
        return None
    repository_root = contract_path.resolve().parent.parent
    namespace = str(lineage["source_evidence_namespace"])
    if namespace != "evidence/current_v2/pilot-v2.7":
        raise PilotEvidenceError("V2.8 parent evidence namespace drifted")
    package_root = repository_root / namespace
    if (
        not package_root.is_dir()
        or package_root.is_symlink()
        or package_root.resolve()
        != (repository_root / "evidence/current_v2/pilot-v2.7").resolve()
    ):
        raise PilotEvidenceError(
            "V2.8 immutable V2.7 evidence package is missing or unsafe"
        )
    manifest_path = package_root / "package_manifest.json"
    checksums_path = package_root / "checksums.json"
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not checksums_path.is_file()
        or checksums_path.is_symlink()
    ):
        raise PilotEvidenceError(
            "V2.8 immutable V2.7 evidence manifest/checksums are missing"
        )
    expected_manifest_sha = str(lineage["package_manifest_file_sha256"])
    if _sha256_file(manifest_path) != expected_manifest_sha:
        raise PilotEvidenceError("V2.8 immutable V2.7 package manifest hash mismatch")
    if _sha256_file(checksums_path) != _PILOT_V27_EVIDENCE_CHECKSUMS_FILE_SHA256:
        raise PilotEvidenceError("V2.8 immutable V2.7 package checksums hash mismatch")
    manifest = _strict_json_load(manifest_path)
    checksums = _strict_json_load(checksums_path)
    if (
        manifest.get("schema_version") != PILOT_V27_EVIDENCE_SCHEMA_VERSION
        or manifest.get("contract_id") != PILOT_V27_CONTRACT_ID
        or manifest.get("evidence_namespace") != "current_v2/pilot-v2.7"
        or manifest.get("contract_sha256") != lineage["source_contract_sha256"]
        or manifest.get("pilot_tag") != lineage["source_release_tag"]
        or manifest.get("publication_status") != "complete-with-no-go"
        or manifest.get("scientific_complete") is not False
        or manifest.get("resolved_git_commit") != lineage["source_release_commit"]
        or checksums.get("schema_version") != PILOT_CHECKSUM_SCHEMA_VERSION
        or checksums.get("contract_sha256") != lineage["source_contract_sha256"]
    ):
        raise PilotEvidenceError(
            "V2.8 immutable V2.7 evidence semantic binding mismatch"
        )
    checksum_rows = checksums.get("files")
    if not isinstance(checksum_rows, Sequence) or isinstance(
        checksum_rows, (str, bytes)
    ):
        raise PilotEvidenceError("V2.7 parent evidence checksum rows are malformed")
    observed_paths: set[str] = set()
    for row in checksum_rows:
        if not isinstance(row, Mapping):
            raise PilotEvidenceError("V2.7 parent evidence checksum row is malformed")
        relative = str(row.get("path", ""))
        candidate_relative = Path(relative)
        if (
            not relative
            or candidate_relative.is_absolute()
            or ".." in candidate_relative.parts
            or relative in observed_paths
        ):
            raise PilotEvidenceError(
                "V2.7 parent evidence checksum path is unsafe or duplicated"
            )
        observed_paths.add(relative)
        candidate = package_root / candidate_relative
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or _sha256_file(candidate) != row.get("sha256")
            or candidate.stat().st_size != row.get("byte_size")
        ):
            raise PilotEvidenceError(
                "V2.7 parent evidence checksum verification failed"
            )
    actual_paths = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != observed_paths | {"checksums.json"}:
        raise PilotEvidenceError(
            "V2.7 parent evidence inventory differs from its checksum ledger"
        )
    published = manifest.get("published_files")
    if (
        not isinstance(published, Sequence)
        or isinstance(published, (str, bytes))
        or not set(map(str, published)).issubset(observed_paths)
        or "package_manifest.json" not in observed_paths
        or next(
            (
                row
                for row in checksum_rows
                if row.get("path") == "package_manifest.json"
            ),
            {},
        ).get("sha256")
        != expected_manifest_sha
    ):
        raise PilotEvidenceError(
            "V2.7 parent evidence manifest is not bound by its checksums"
        )
    aggregate = _strict_json_load(package_root / "aggregate.json")
    failure_ledger = _strict_json_load(package_root / "failure_ledger.json")
    parent_denominator = aggregate.get("denominator")
    failure_denominator = failure_ledger.get("denominator")
    expected_statuses = {"complete": 1, "integrity-stopped": 210}
    if (
        not isinstance(parent_denominator, Mapping)
        or parent_denominator.get("expected_count") != 211
        or parent_denominator.get("observed_ledger_count") != 211
        or parent_denominator.get("status_counts") != expected_statuses
        or failure_denominator != parent_denominator
    ):
        raise PilotEvidenceError(
            "V2.7 parent evidence denominator does not match its terminal no-go"
        )
    return {
        **lineage,
        "reference_kind": "immutable-external-package-reference",
        "source_package_path": namespace,
        "source_package_copied": False,
        "checksums_file_sha256": _sha256_file(checksums_path),
        "checksum_entry_count": len(checksum_rows),
        "inventory_verified": True,
        "semantic_binding_verified": True,
    }


def _validated_v29_parent_evidence_reference(
    contract: PilotContract,
    *,
    contract_path: Path,
) -> dict[str, Any] | None:
    """Revalidate V2.8 in place and return a reference, never copied rows."""

    lineage = _v29_parent_evidence_lineage(contract)
    if lineage is None:
        return None
    repository_root = contract_path.resolve().parent.parent
    namespace = str(lineage["source_evidence_namespace"])
    if namespace != "evidence/current_v2/pilot-v2.8":
        raise PilotEvidenceError("V2.9 parent evidence namespace drifted")
    package_root = repository_root / namespace
    expected_root = repository_root / "evidence/current_v2/pilot-v2.8"
    if (
        not package_root.is_dir()
        or package_root.is_symlink()
        or package_root.resolve() != expected_root.resolve()
    ):
        raise PilotEvidenceError(
            "V2.9 immutable V2.8 evidence package is missing or unsafe"
        )
    manifest_path = package_root / "package_manifest.json"
    checksums_path = package_root / "checksums.json"
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not checksums_path.is_file()
        or checksums_path.is_symlink()
    ):
        raise PilotEvidenceError(
            "V2.9 immutable V2.8 evidence manifest/checksums are missing"
        )

    observed_manifest_sha = _sha256_file(manifest_path)
    observed_checksums_sha = _sha256_file(checksums_path)
    expected_manifest_sha = lineage.get("package_manifest_file_sha256")
    expected_checksums_sha = lineage.get("checksums_file_sha256")
    if expected_manifest_sha is None:
        if contract.status != "draft":
            raise PilotEvidenceError(
                "V2.9 frozen parent package manifest hash is unbound"
            )
        expected_manifest_sha = observed_manifest_sha
    if expected_checksums_sha is None:
        if contract.status != "draft":
            raise PilotEvidenceError(
                "V2.9 frozen parent package checksums hash is unbound"
            )
        expected_checksums_sha = observed_checksums_sha
    if (
        not isinstance(expected_manifest_sha, str)
        or len(expected_manifest_sha) != 64
        or observed_manifest_sha != expected_manifest_sha
    ):
        raise PilotEvidenceError("V2.9 immutable V2.8 package manifest hash mismatch")
    if (
        not isinstance(expected_checksums_sha, str)
        or len(expected_checksums_sha) != 64
        or observed_checksums_sha != expected_checksums_sha
    ):
        raise PilotEvidenceError("V2.9 immutable V2.8 package checksums hash mismatch")

    manifest = _strict_json_load(manifest_path)
    checksums = _strict_json_load(checksums_path)
    if (
        manifest.get("schema_version") != PILOT_V28_EVIDENCE_SCHEMA_VERSION
        or manifest.get("contract_id") != PILOT_V28_CONTRACT_ID
        or manifest.get("evidence_namespace") != "current_v2/pilot-v2.8"
        or manifest.get("contract_sha256") != lineage["source_contract_sha256"]
        or manifest.get("pilot_tag") != lineage["source_release_tag"]
        or manifest.get("publication_status") != "complete-with-no-go"
        or manifest.get("scientific_complete") is not False
        or manifest.get("resolved_git_commit") != lineage["source_release_commit"]
        or checksums.get("schema_version") != PILOT_CHECKSUM_SCHEMA_VERSION
        or checksums.get("contract_sha256") != lineage["source_contract_sha256"]
    ):
        raise PilotEvidenceError(
            "V2.9 immutable V2.8 evidence semantic binding mismatch"
        )
    checksum_rows = checksums.get("files")
    if not isinstance(checksum_rows, Sequence) or isinstance(
        checksum_rows, (str, bytes)
    ):
        raise PilotEvidenceError("V2.8 parent evidence checksum rows are malformed")
    observed_paths: set[str] = set()
    for row in checksum_rows:
        if not isinstance(row, Mapping):
            raise PilotEvidenceError("V2.8 parent evidence checksum row is malformed")
        relative = str(row.get("path", ""))
        candidate_relative = Path(relative)
        if (
            not relative
            or candidate_relative.is_absolute()
            or ".." in candidate_relative.parts
            or relative in observed_paths
        ):
            raise PilotEvidenceError(
                "V2.8 parent evidence checksum path is unsafe or duplicated"
            )
        observed_paths.add(relative)
        candidate = package_root / candidate_relative
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or _sha256_file(candidate) != row.get("sha256")
            or candidate.stat().st_size != row.get("byte_size")
        ):
            raise PilotEvidenceError(
                "V2.8 parent evidence checksum verification failed"
            )
    actual_paths = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != observed_paths | {"checksums.json"}:
        raise PilotEvidenceError(
            "V2.8 parent evidence inventory differs from its checksum ledger"
        )
    published = manifest.get("published_files")
    if (
        not isinstance(published, Sequence)
        or isinstance(published, (str, bytes))
        or not set(map(str, published)).issubset(observed_paths)
        or "package_manifest.json" not in observed_paths
        or next(
            (
                row
                for row in checksum_rows
                if row.get("path") == "package_manifest.json"
            ),
            {},
        ).get("sha256")
        != expected_manifest_sha
    ):
        raise PilotEvidenceError(
            "V2.8 parent evidence manifest is not bound by its checksums"
        )
    aggregate = _strict_json_load(package_root / "aggregate.json")
    failure_ledger = _strict_json_load(package_root / "failure_ledger.json")
    parent_denominator = aggregate.get("denominator")
    failure_denominator = failure_ledger.get("denominator")
    expected_statuses = {
        "complete": 1,
        "failed": 1,
        "integrity-stopped": 209,
    }
    if (
        not isinstance(parent_denominator, Mapping)
        or parent_denominator.get("expected_count") != 211
        or parent_denominator.get("observed_ledger_count") != 211
        or parent_denominator.get("status_counts") != expected_statuses
        or failure_denominator != parent_denominator
    ):
        raise PilotEvidenceError(
            "V2.8 parent evidence denominator does not match its terminal no-go"
        )
    return {
        **lineage,
        "package_manifest_file_sha256": observed_manifest_sha,
        "checksums_file_sha256": observed_checksums_sha,
        "reference_kind": "immutable-external-package-reference",
        "source_package_path": namespace,
        "source_package_copied": False,
        "checksum_entry_count": len(checksum_rows),
        "inventory_verified": True,
        "semantic_binding_verified": True,
    }


def _validated_v210_parent_evidence_reference(
    contract: PilotContract,
    *,
    contract_path: Path,
) -> dict[str, Any] | None:
    """Revalidate V2.9's immutable implementation-no-go package in place."""

    lineage = _v210_parent_evidence_lineage(contract)
    if lineage is None:
        return None
    repository_root = contract_path.resolve().parent.parent
    namespace = str(lineage["source_evidence_namespace"])
    if namespace != "evidence/current_v2/pilot-v2.9":
        raise PilotEvidenceError("V2.10 parent evidence namespace drifted")
    package_root = repository_root / namespace
    expected_root = repository_root / "evidence/current_v2/pilot-v2.9"
    if (
        not package_root.is_dir()
        or package_root.is_symlink()
        or package_root.resolve() != expected_root.resolve()
    ):
        raise PilotEvidenceError(
            "V2.10 immutable V2.9 evidence package is missing or unsafe"
        )
    manifest_path = package_root / "package_manifest.json"
    checksums_path = package_root / "checksums.json"
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not checksums_path.is_file()
        or checksums_path.is_symlink()
    ):
        raise PilotEvidenceError(
            "V2.10 immutable V2.9 evidence manifest/checksums are missing"
        )
    observed_manifest_sha = _sha256_file(manifest_path)
    observed_checksums_sha = _sha256_file(checksums_path)
    if (
        observed_manifest_sha
        != lineage.get("package_manifest_file_sha256")
        or observed_manifest_sha
        != _PILOT_V29_EVIDENCE_PACKAGE_MANIFEST_SHA256
    ):
        raise PilotEvidenceError(
            "V2.10 immutable V2.9 package manifest hash mismatch"
        )
    if (
        observed_checksums_sha != lineage.get("checksums_file_sha256")
        or observed_checksums_sha != _PILOT_V29_EVIDENCE_CHECKSUMS_SHA256
    ):
        raise PilotEvidenceError(
            "V2.10 immutable V2.9 package checksums hash mismatch"
        )

    manifest = _strict_json_load(manifest_path)
    checksums = _strict_json_load(checksums_path)
    if (
        manifest.get("schema_version") != PILOT_V29_EVIDENCE_SCHEMA_VERSION
        or manifest.get("contract_id") != PILOT_V29_CONTRACT_ID
        or manifest.get("evidence_namespace") != "current_v2/pilot-v2.9"
        or manifest.get("contract_sha256")
        != lineage["source_contract_sha256"]
        or manifest.get("pilot_tag") != lineage["source_release_tag"]
        or manifest.get("publication_status") != "complete-with-no-go"
        or manifest.get("scientific_complete") is not False
        or manifest.get("resolved_git_commit") != _PILOT_V29_RELEASE_COMMIT
        or checksums.get("schema_version") != PILOT_CHECKSUM_SCHEMA_VERSION
        or checksums.get("contract_sha256")
        != lineage["source_contract_sha256"]
    ):
        raise PilotEvidenceError(
            "V2.10 immutable V2.9 evidence semantic binding mismatch"
        )
    checksum_rows = checksums.get("files")
    if (
        not isinstance(checksum_rows, Sequence)
        or isinstance(checksum_rows, (str, bytes))
        or len(checksum_rows) != 17
    ):
        raise PilotEvidenceError("V2.9 parent evidence checksum rows are malformed")
    observed_paths: set[str] = set()
    for row in checksum_rows:
        if not isinstance(row, Mapping):
            raise PilotEvidenceError(
                "V2.9 parent evidence checksum row is malformed"
            )
        relative = str(row.get("path", ""))
        candidate_relative = Path(relative)
        if (
            not relative
            or candidate_relative.is_absolute()
            or ".." in candidate_relative.parts
            or relative in observed_paths
        ):
            raise PilotEvidenceError(
                "V2.9 parent evidence checksum path is unsafe or duplicated"
            )
        observed_paths.add(relative)
        candidate = package_root / candidate_relative
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or _sha256_file(candidate) != row.get("sha256")
            or candidate.stat().st_size != row.get("byte_size")
        ):
            raise PilotEvidenceError(
                "V2.9 parent evidence checksum verification failed"
            )
    actual_paths = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
    }
    if actual_paths != observed_paths | {"checksums.json"}:
        raise PilotEvidenceError(
            "V2.9 parent evidence inventory differs from its checksum ledger"
        )
    published = manifest.get("published_files")
    if (
        not isinstance(published, Sequence)
        or isinstance(published, (str, bytes))
        or not set(map(str, published)).issubset(observed_paths)
        or "package_manifest.json" not in observed_paths
        or next(
            (
                row
                for row in checksum_rows
                if row.get("path") == "package_manifest.json"
            ),
            {},
        ).get("sha256")
        != observed_manifest_sha
    ):
        raise PilotEvidenceError(
            "V2.9 parent evidence manifest is not bound by its checksums"
        )

    aggregate = _strict_json_load(package_root / "aggregate.json")
    failure_ledger = _strict_json_load(package_root / "failure_ledger.json")
    parent_denominator = aggregate.get("denominator")
    failure_denominator = failure_ledger.get("denominator")
    implementation = aggregate.get("implementation_failure")
    expected_statuses = {"complete": 26, "failed": 185}
    if (
        not isinstance(parent_denominator, Mapping)
        or parent_denominator.get("expected_count") != 211
        or parent_denominator.get("observed_ledger_count") != 211
        or parent_denominator.get("status_counts") != expected_statuses
        or failure_denominator != parent_denominator
        or aggregate.get("contract_id") != PILOT_V29_CONTRACT_ID
        or aggregate.get("contract_sha256")
        != lineage["source_contract_sha256"]
        or aggregate.get("resolved_git_commit") != _PILOT_V29_RELEASE_COMMIT
        or aggregate.get("publication_status") != "complete-with-no-go"
        or aggregate.get("scientific_complete") is not False
        or aggregate.get("scientific_claim_gates_supported") is not False
        or not isinstance(implementation, Mapping)
        or implementation.get("classification") != "implementation-interface-no-go"
        or implementation.get("root_cause_code")
        != "imported-p95-runner-binding-shape-mismatch"
        or not isinstance(implementation.get("provider_boundary"), Mapping)
        or implementation["provider_boundary"].get("failure_phase")
        != "before-provider-construction-and-dispatch"
        or implementation["provider_boundary"].get("v2_9_hosted_completions") != 0
        or implementation["provider_boundary"].get("v2_9_hosted_stage_cost_usd")
        != 0.0
        or not isinstance(implementation.get("outcome_boundary"), Mapping)
        or implementation["outcome_boundary"].get(
            "actor_action_utility_rule_exposure_outcomes_generated"
        )
        is not False
        or implementation["outcome_boundary"].get(
            "offline_candidate_admission_cells_generated"
        )
        != 10
    ):
        raise PilotEvidenceError(
            "V2.9 parent evidence does not match its terminal "
            "implementation-interface no-go"
        )
    return {
        **lineage,
        "package_manifest_file_sha256": observed_manifest_sha,
        "checksums_file_sha256": observed_checksums_sha,
        "reference_kind": "immutable-external-package-reference",
        "source_package_path": namespace,
        "source_package_copied": False,
        "checksum_entry_count": len(checksum_rows),
        "inventory_verified": True,
        "semantic_binding_verified": True,
        "implementation_no_go_verified": True,
        "offline_candidate_disclosure_verified": True,
    }


def _v210_source_manifest_amendment_chain(
    contract: PilotContract,
) -> tuple[tuple[Mapping[str, Any] | None, str], ...]:
    """Return V2.10's complete newest-to-oldest source-manifest chain."""

    if contract.contract_id != PILOT_V210_CONTRACT_ID:
        raise PilotEvidenceError(
            "V2.10 source-manifest chain requested for another contract"
        )
    return (
        (
            contract.p95_runner_binding_retry_amendment,
            "pilot_v2_10_source_manifest.json",
        ),
        (
            contract.qref_summary_equivalence_amendment,
            "pilot_v2_9_source_manifest.json",
        ),
        (
            contract.qref_identity_retry_amendment,
            "pilot_v2_8_source_manifest.json",
        ),
        (
            contract.stage0_evaluator_retry_amendment,
            "pilot_v2_7_source_manifest.json",
        ),
        (
            contract.p95_authority_retry_amendment,
            "pilot_v2_6_source_manifest.json",
        ),
        (
            contract.parent_import_retry_amendment,
            "pilot_v2_5_source_manifest.json",
        ),
    )


def _write_v24_package(
    root: Path,
    *,
    contract_path: Path,
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    common_commit: str | None,
    experiment_c_sensitivities: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Path, Path]:
    version_label = _contract_version_label(contract)
    schema_version = _evidence_schema_version(contract)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise PilotEvidenceError(
            f"temporary {version_label} package directory is not empty: {root}"
        )
    contract_target = root / "contract" / contract_path.name
    contract_target.parent.mkdir(parents=True, exist_ok=True)
    if contract.matrix_amendment is None:
        raise PilotEvidenceError(f"{version_label} contract lacks its matrix amendment")
    parent_binding_raw = contract.matrix_amendment.get("parent_source_manifest")
    if not isinstance(parent_binding_raw, Mapping):
        raise PilotEvidenceError(
            f"{version_label} contract lacks its parent source manifest binding"
        )
    parent_binding = dict(parent_binding_raw)
    parent_manifest_name = "pilot_v2_4_parent_source_manifest.json"
    parent_manifest_path = str(parent_binding.get("path", ""))
    if (
        parent_manifest_path != f"experiments/{parent_manifest_name}"
        or Path(parent_manifest_path).name != parent_manifest_name
    ):
        raise PilotEvidenceError(
            f"{version_label} parent source manifest package path drifted"
        )
    parent_manifest_source = contract_path.with_name(parent_manifest_name)
    if not parent_manifest_source.is_file():
        raise PilotEvidenceError(
            f"{version_label} parent source manifest sibling is missing"
        )
    parent_manifest_target = contract_target.with_name(parent_manifest_name)
    shutil.copyfile(contract_path, contract_target)
    shutil.copyfile(parent_manifest_source, parent_manifest_target)
    if _sha256_file(parent_manifest_target) != parent_binding.get("file_sha256"):
        raise PilotEvidenceError(
            f"copied {version_label} parent source manifest failed " "hash revalidation"
        )

    retry_binding: dict[str, Any] | None = None
    retry_manifest_name: str | None = None
    inherited_retry_binding: dict[str, Any] | None = None
    inherited_retry_manifest_name: str | None = None
    ancestral_retry_binding: dict[str, Any] | None = None
    ancestral_retry_manifest_name: str | None = None
    great_ancestral_retry_binding: dict[str, Any] | None = None
    great_ancestral_retry_manifest_name: str | None = None
    deep_ancestral_retry_binding: dict[str, Any] | None = None
    deep_ancestral_retry_manifest_name: str | None = None
    deepest_ancestral_retry_binding: dict[str, Any] | None = None
    deepest_ancestral_retry_manifest_name: str | None = None
    base_binding: dict[str, Any] | None = None
    base_contract_name: str | None = None
    if contract.contract_id == PILOT_V25_CONTRACT_ID:
        retry_amendment = contract.parent_import_retry_amendment
        if not isinstance(retry_amendment, Mapping):
            raise PilotEvidenceError(
                "V2.5 contract lacks its parent-import retry amendment"
            )
        retry_binding_raw = retry_amendment.get("source_manifest")
        if not isinstance(retry_binding_raw, Mapping):
            raise PilotEvidenceError(
                "V2.5 contract lacks its retry source manifest binding"
            )
        retry_binding = dict(retry_binding_raw)
        retry_manifest_name = "pilot_v2_5_source_manifest.json"
        retry_manifest_path = str(retry_binding.get("path", ""))
        if (
            retry_manifest_path != f"experiments/{retry_manifest_name}"
            or Path(retry_manifest_path).name != retry_manifest_name
        ):
            raise PilotEvidenceError("V2.5 retry source manifest package path drifted")
        retry_manifest_source = contract_path.with_name(retry_manifest_name)
        if not retry_manifest_source.is_file():
            raise PilotEvidenceError("V2.5 retry source manifest sibling is missing")
        retry_manifest_target = contract_target.with_name(retry_manifest_name)
        shutil.copyfile(retry_manifest_source, retry_manifest_target)
        if _sha256_file(retry_manifest_target) != retry_binding.get("file_sha256"):
            raise PilotEvidenceError(
                "copied V2.5 retry source manifest failed hash revalidation"
            )

        contract_document = _strict_json_load(contract_path)
        base_binding_raw = contract_document.get("base_contract")
        if base_binding_raw is not None:
            if not isinstance(base_binding_raw, Mapping):
                raise PilotEvidenceError(
                    "V2.5 overlay base contract binding is malformed"
                )
            base_binding = dict(base_binding_raw)
            base_contract_name = "pilot_v2_4.yaml"
            if (
                base_binding.get("path") != base_contract_name
                or Path(str(base_binding.get("path", ""))).name != base_contract_name
            ):
                raise PilotEvidenceError(
                    "V2.5 overlay base contract package path drifted"
                )
            base_contract_source = contract_path.with_name(base_contract_name)
            if not base_contract_source.is_file():
                raise PilotEvidenceError(
                    "V2.5 overlay base contract sibling is missing"
                )
            shutil.copyfile(
                base_contract_source,
                contract_target.with_name(base_contract_name),
            )
    elif contract.contract_id in {
        PILOT_V28_CONTRACT_ID,
        PILOT_V29_CONTRACT_ID,
        PILOT_V210_CONTRACT_ID,
    }:
        is_v29 = contract.contract_id == PILOT_V29_CONTRACT_ID
        is_v210 = contract.contract_id == PILOT_V210_CONTRACT_ID
        if is_v210:
            lineage_bindings = _v210_source_manifest_amendment_chain(contract)
        else:
            lineage_bindings = (
                (
                    (
                        contract.qref_summary_equivalence_amendment
                        if is_v29
                        else contract.qref_identity_retry_amendment
                    ),
                    (
                        "pilot_v2_9_source_manifest.json"
                        if is_v29
                        else "pilot_v2_8_source_manifest.json"
                    ),
                ),
                *(
                    (
                        (
                            contract.qref_identity_retry_amendment,
                            "pilot_v2_8_source_manifest.json",
                        ),
                    )
                    if is_v29
                    else ()
                ),
                (
                    contract.stage0_evaluator_retry_amendment,
                    "pilot_v2_7_source_manifest.json",
                ),
                (
                    contract.p95_authority_retry_amendment,
                    "pilot_v2_6_source_manifest.json",
                ),
                (
                    contract.parent_import_retry_amendment,
                    "pilot_v2_5_source_manifest.json",
                ),
            )
        copied_bindings: list[dict[str, Any]] = []
        for amendment, manifest_name in lineage_bindings:
            if not isinstance(amendment, Mapping):
                raise PilotEvidenceError(
                    f"{version_label} contract lacks a required "
                    "source-manifest amendment"
                )
            binding_raw = amendment.get("source_manifest")
            if not isinstance(binding_raw, Mapping):
                raise PilotEvidenceError(
                    f"{version_label} contract lacks a required "
                    "source-manifest binding"
                )
            binding = dict(binding_raw)
            if (
                binding.get("path") != f"experiments/{manifest_name}"
                or not isinstance(binding.get("file_sha256"), str)
                or len(str(binding["file_sha256"])) != 64
                or not isinstance(binding.get("content_sha256"), str)
                or len(str(binding["content_sha256"])) != 64
            ):
                raise PilotEvidenceError(
                    f"{version_label} source-manifest binding is unsealed or " "drifted"
                )
            source = contract_path.with_name(manifest_name)
            target = contract_target.with_name(manifest_name)
            if not source.is_file() or source.is_symlink():
                raise PilotEvidenceError(
                    f"{version_label} source manifest is missing or unsafe: "
                    f"{manifest_name}"
                )
            shutil.copyfile(source, target)
            if _sha256_file(target) != binding["file_sha256"]:
                raise PilotEvidenceError(
                    f"copied {version_label} source manifest failed hash "
                    "revalidation: "
                    f"{manifest_name}"
                )
            copied_bindings.append(binding)
        (
            retry_binding,
            inherited_retry_binding,
            ancestral_retry_binding,
            great_ancestral_retry_binding,
            *deep_bindings,
        ) = copied_bindings
        (
            retry_manifest_name,
            inherited_retry_manifest_name,
            ancestral_retry_manifest_name,
            great_ancestral_retry_manifest_name,
            *deep_manifest_names,
        ) = tuple(name for _, name in lineage_bindings)
        if deep_bindings:
            deep_ancestral_retry_binding = deep_bindings[0]
            deep_ancestral_retry_manifest_name = deep_manifest_names[0]
        if len(deep_bindings) > 1:
            deepest_ancestral_retry_binding = deep_bindings[1]
            deepest_ancestral_retry_manifest_name = deep_manifest_names[1]

        failure = (
            _v210_amendment(contract)
            if is_v210
            else _v29_amendment(contract)
            if is_v29
            else _v28_amendment(contract)
        ).get("failure_classification")
        if not isinstance(failure, Mapping):
            raise PilotEvidenceError(
                f"{version_label} parent contract binding is malformed"
            )
        base_contract_name = (
            "pilot_v2_9.yaml"
            if is_v210
            else "pilot_v2_8.yaml"
            if is_v29
            else "pilot_v2_7.yaml"
        )
        parent_contract_id = (
            PILOT_V29_CONTRACT_ID
            if is_v210
            else PILOT_V28_CONTRACT_ID
            if is_v29
            else PILOT_V27_CONTRACT_ID
        )
        base_binding = {
            "path": base_contract_name,
            "schema_version": "finevo-pilot-contract-v2",
            "contract_id": parent_contract_id,
            "canonical_sha256": failure.get("parent_contract_sha256"),
        }
        base_contract_source = contract_path.with_name(base_contract_name)
        base_contract_target = contract_target.with_name(base_contract_name)
        if not base_contract_source.is_file() or base_contract_source.is_symlink():
            raise PilotEvidenceError(
                f"{version_label} base contract is missing or unsafe"
            )
        shutil.copyfile(base_contract_source, base_contract_target)
        copied_base = load_pilot_contract(base_contract_target)
        if (
            copied_base.contract_id != parent_contract_id
            or copied_base.canonical_hash != base_binding["canonical_sha256"]
        ):
            raise PilotEvidenceError(
                f"copied {version_label} base contract failed identity " "revalidation"
            )
    elif contract.contract_id in {
        PILOT_V26_CONTRACT_ID,
        PILOT_V27_CONTRACT_ID,
    }:
        is_v27 = contract.contract_id == PILOT_V27_CONTRACT_ID
        retry_amendment = getattr(
            contract,
            (
                "stage0_evaluator_retry_amendment"
                if is_v27
                else "p95_authority_retry_amendment"
            ),
            None,
        )
        if not isinstance(retry_amendment, Mapping):
            raise PilotEvidenceError(
                f"{version_label} contract lacks its retry amendment"
            )
        retry_binding_raw = retry_amendment.get("source_manifest")
        if not isinstance(retry_binding_raw, Mapping):
            raise PilotEvidenceError(
                f"{version_label} contract lacks its retry source " "manifest binding"
            )
        retry_binding = dict(retry_binding_raw)
        retry_manifest_name = (
            "pilot_v2_7_source_manifest.json"
            if is_v27
            else "pilot_v2_6_source_manifest.json"
        )
        retry_manifest_path = str(retry_binding.get("path", ""))
        if (
            retry_manifest_path != f"experiments/{retry_manifest_name}"
            or Path(retry_manifest_path).name != retry_manifest_name
        ):
            raise PilotEvidenceError(
                f"{version_label} retry source manifest package path drifted"
            )
        retry_manifest_source = contract_path.with_name(retry_manifest_name)
        if not retry_manifest_source.is_file():
            raise PilotEvidenceError(
                f"{version_label} retry source manifest sibling is missing"
            )
        retry_manifest_target = contract_target.with_name(retry_manifest_name)
        shutil.copyfile(retry_manifest_source, retry_manifest_target)
        if _sha256_file(retry_manifest_target) != retry_binding.get("file_sha256"):
            raise PilotEvidenceError(
                f"copied {version_label} retry source manifest failed "
                "hash revalidation"
            )

        inherited_retry = (
            contract.p95_authority_retry_amendment
            if is_v27
            else contract.parent_import_retry_amendment
        )
        if not isinstance(inherited_retry, Mapping):
            raise PilotEvidenceError(
                f"{version_label} contract lacks its inherited retry amendment"
            )
        inherited_retry_binding_raw = inherited_retry.get("source_manifest")
        if not isinstance(inherited_retry_binding_raw, Mapping):
            raise PilotEvidenceError(
                f"{version_label} contract lacks its inherited source binding"
            )
        inherited_retry_binding = dict(inherited_retry_binding_raw)
        inherited_retry_manifest_name = (
            "pilot_v2_6_source_manifest.json"
            if is_v27
            else "pilot_v2_5_source_manifest.json"
        )
        if (
            inherited_retry_binding.get("path")
            != f"experiments/{inherited_retry_manifest_name}"
        ):
            raise PilotEvidenceError(
                f"{version_label} inherited source-manifest path drifted"
            )
        inherited_retry_source = contract_path.with_name(inherited_retry_manifest_name)
        inherited_retry_target = contract_target.with_name(
            inherited_retry_manifest_name
        )
        if not inherited_retry_source.is_file():
            raise PilotEvidenceError(
                f"{version_label} inherited source manifest is missing"
            )
        shutil.copyfile(inherited_retry_source, inherited_retry_target)
        if _sha256_file(inherited_retry_target) != inherited_retry_binding.get(
            "file_sha256"
        ):
            raise PilotEvidenceError(
                f"copied {version_label} inherited source manifest failed "
                "hash revalidation"
            )

        if is_v27:
            ancestral_retry = contract.parent_import_retry_amendment
            if not isinstance(ancestral_retry, Mapping):
                raise PilotEvidenceError(
                    "V2.7 contract lacks its inherited V2.5 retry amendment"
                )
            ancestral_binding_raw = ancestral_retry.get("source_manifest")
            if not isinstance(ancestral_binding_raw, Mapping):
                raise PilotEvidenceError(
                    "V2.7 contract lacks its inherited V2.5 source binding"
                )
            ancestral_retry_binding = dict(ancestral_binding_raw)
            ancestral_retry_manifest_name = "pilot_v2_5_source_manifest.json"
            if (
                ancestral_retry_binding.get("path")
                != f"experiments/{ancestral_retry_manifest_name}"
            ):
                raise PilotEvidenceError(
                    "V2.7 inherited V2.5 source-manifest path drifted"
                )
            ancestral_source = contract_path.with_name(ancestral_retry_manifest_name)
            ancestral_target = contract_target.with_name(ancestral_retry_manifest_name)
            if not ancestral_source.is_file():
                raise PilotEvidenceError(
                    "V2.7 inherited V2.5 source manifest is missing"
                )
            shutil.copyfile(ancestral_source, ancestral_target)
            if _sha256_file(ancestral_target) != ancestral_retry_binding.get(
                "file_sha256"
            ):
                raise PilotEvidenceError(
                    "copied V2.7 inherited V2.5 source manifest failed "
                    "hash revalidation"
                )

        contract_document = _strict_json_load(contract_path)
        base_binding_raw = contract_document.get("base_contract")
        base_contract_name = "pilot_v2_6.yaml" if is_v27 else "pilot_v2_5.yaml"
        if base_binding_raw is None:
            parent_failure = (
                contract.stage0_evaluator_retry_amendment["failure_classification"]
                if is_v27
                else contract.p95_authority_retry_amendment["failure_classification"]
            )
            base_binding = {
                "path": base_contract_name,
                "schema_version": "finevo-pilot-contract-v2",
                "contract_id": (
                    PILOT_V26_CONTRACT_ID if is_v27 else PILOT_V25_CONTRACT_ID
                ),
                "canonical_sha256": parent_failure["parent_contract_sha256"],
            }
        elif isinstance(base_binding_raw, Mapping):
            base_binding = dict(base_binding_raw)
        else:
            raise PilotEvidenceError(
                f"{version_label} overlay base contract binding is malformed"
            )
        if base_binding.get("path") != base_contract_name:
            raise PilotEvidenceError(
                f"{version_label} base contract package path drifted"
            )
        base_contract_source = contract_path.with_name(base_contract_name)
        if not base_contract_source.is_file():
            raise PilotEvidenceError(
                f"{version_label} base contract sibling is missing"
            )
        shutil.copyfile(
            base_contract_source,
            contract_target.with_name(base_contract_name),
        )
        copied_base = load_pilot_contract(contract_target.with_name(base_contract_name))
        if copied_base.contract_id != (
            PILOT_V26_CONTRACT_ID if is_v27 else PILOT_V25_CONTRACT_ID
        ) or copied_base.canonical_hash != base_binding.get("canonical_sha256"):
            raise PilotEvidenceError(
                f"copied {version_label} base contract failed identity " "revalidation"
            )

    copied = load_pilot_contract(contract_target)
    if copied.canonical_hash != contract.canonical_hash:
        raise PilotEvidenceError(
            f"copied {version_label} contract failed hash revalidation"
        )

    parent_evidence_reference = _validated_v28_parent_evidence_reference(
        contract,
        contract_path=contract_path,
    )
    if parent_evidence_reference is None:
        parent_evidence_reference = _validated_v29_parent_evidence_reference(
            contract,
            contract_path=contract_path,
        )
    if parent_evidence_reference is None:
        parent_evidence_reference = _validated_v210_parent_evidence_reference(
            contract,
            contract_path=contract_path,
        )
    if parent_evidence_reference is not None:
        parent_reference_schema = {
            PILOT_V28_CONTRACT_ID: (
                "finevo-pilot-v2.8-parent-evidence-reference-v1"
            ),
            PILOT_V29_CONTRACT_ID: (
                "finevo-pilot-v2.9-parent-evidence-reference-v1"
            ),
            PILOT_V210_CONTRACT_ID: (
                "finevo-pilot-v2.10-parent-evidence-reference-v1"
            ),
        }[contract.contract_id]
        _atomic_bytes(
            root / "parent_evidence_reference.json",
            _pretty_bytes(
                {
                    "schema_version": parent_reference_schema,
                    "contract_id": contract.contract_id,
                    "contract_sha256": contract.canonical_hash,
                    **parent_evidence_reference,
                }
            ),
        )

    sensitivity_published_files: list[str] = []
    sensitivity_manifest: dict[str, dict[str, Any]] | None = None
    if contract.contract_id == PILOT_V210_CONTRACT_ID:
        controls_raw = aggregate.get("experiment_c_rule_sensitivities")
        if not isinstance(controls_raw, Mapping):
            raise PilotEvidenceError(
                "V2.10 package lacks lane-specific Experiment C sensitivity "
                "controls"
            )
        controls = _validated_v210_sensitivity_controls(
            {
                "experiment_c_rule_sensitivities": controls_raw,
            }
        )
        supplied = (
            {}
            if experiment_c_sensitivities is None
            else dict(experiment_c_sensitivities)
        )
        expected_available = {
            lane_id
            for lane_id, control in controls.items()
            if control["available"]
        }
        if set(supplied) != expected_available:
            raise PilotEvidenceError(
                "V2.10 published Experiment C sensitivities differ from the "
                "validated available lanes"
            )
        sensitivity_manifest = {}
        for lane_id, control in controls.items():
            manifest_entry = {
                **_json_copy(control),
                "published": bool(control["available"]),
            }
            if control["available"]:
                supplied_entry = supplied[lane_id]
                if not isinstance(supplied_entry, Mapping):
                    raise PilotEvidenceError(
                        f"V2.10 {lane_id} sensitivity payload is malformed"
                    )
                payload = supplied_entry.get("payload")
                supplied_control = supplied_entry.get("control")
                if (
                    not isinstance(payload, Mapping)
                    or supplied_control != control
                ):
                    raise PilotEvidenceError(
                        f"V2.10 {lane_id} sensitivity payload/control binding "
                        "drifted"
                    )
                source = Path(control["path"])
                if source.is_symlink() or not source.is_file():
                    raise PilotEvidenceError(
                        f"V2.10 {lane_id} sensitivity source became missing or "
                        "unsafe before publication"
                    )
                source_bytes = source.read_bytes()
                if (
                    hashlib.sha256(source_bytes).hexdigest()
                    != control["file_sha256"]
                ):
                    raise PilotEvidenceError(
                        f"V2.10 {lane_id} sensitivity source changed after "
                        "validation"
                    )
                try:
                    decoded = json.loads(source_bytes)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise PilotEvidenceError(
                        f"V2.10 {lane_id} sensitivity source is not canonical "
                        "JSON"
                    ) from exc
                if decoded != payload:
                    raise PilotEvidenceError(
                        f"V2.10 {lane_id} sensitivity source/payload mismatch"
                    )
                package_path = str(control["package_path"])
                target = root / package_path
                _atomic_bytes(target, source_bytes)
                if _sha256_file(target) != control["file_sha256"]:
                    raise PilotEvidenceError(
                        f"published V2.10 {lane_id} sensitivity failed hash "
                        "revalidation"
                    )
                sensitivity_published_files.append(package_path)
            sensitivity_manifest[lane_id] = manifest_entry

    sanitized = _sanitized_rows(rows)
    implementation_failure = _implementation_failure_summary_for_contract(
        aggregate,
        sanitized,
        resolved_git_commit=common_commit,
    )
    aggregate_payload = {
        **_json_copy(aggregate),
        "resolved_git_commit": common_commit,
        "rows": sanitized,
    }
    if implementation_failure is not None:
        aggregate_payload["implementation_failure"] = implementation_failure
    _atomic_bytes(root / "aggregate.json", _pretty_bytes(aggregate_payload))
    _atomic_bytes(root / "aggregate.csv", _aggregate_csv(sanitized))
    _atomic_bytes(
        root / "claim_metric_artifact.json",
        _pretty_bytes(
            {
                "schema_version": schema_version,
                "contract_sha256": contract.canonical_hash,
                "claims": aggregate["claims"],
                **(
                    {"implementation_failure": implementation_failure}
                    if implementation_failure is not None
                    else {}
                ),
            }
        ),
    )
    _atomic_bytes(
        root / "claim_narrowing.json",
        _pretty_bytes(
            {
                "schema_version": schema_version,
                "contract_sha256": contract.canonical_hash,
                "rows": aggregate["claim_narrowing"],
            }
        ),
    )
    failures = [
        {
            "run_id": row["run_id"],
            "stage_id": row["stage_id"],
            "model_id": row["model_id"],
            "arm_id": row["arm_id"],
            "environment_seed": row["environment_seed"],
            "status": row["status"],
            "failure": row["failure"],
        }
        for row in sanitized
        if row["status"] != "complete"
    ]
    _atomic_bytes(
        root / "failure_ledger.json",
        _pretty_bytes(
            {
                "schema_version": PILOT_FAILURE_LEDGER_SCHEMA_VERSION,
                "contract_sha256": contract.canonical_hash,
                "denominator": aggregate["denominator"],
                **(
                    {"implementation_failure": implementation_failure}
                    if implementation_failure is not None
                    else {}
                ),
                "rows": failures,
            }
        ),
    )
    _atomic_bytes(
        root / "method_differences_scaffold.json",
        _pretty_bytes(_method_scaffold(contract_path.name)),
    )
    _atomic_bytes(
        root / "reviewer_report.md",
        _report_markdown(aggregate_payload).encode("utf-8"),
    )
    published_files = sorted(
        [
            "aggregate.csv",
            "aggregate.json",
            "claim_metric_artifact.json",
            "claim_narrowing.json",
            f"contract/{contract_path.name}",
            f"contract/{parent_manifest_name}",
            "failure_ledger.json",
            "method_differences_scaffold.json",
            "reviewer_report.md",
        ]
        + sensitivity_published_files
        + (
            [f"contract/{retry_manifest_name}"]
            if retry_manifest_name is not None
            else []
        )
        + (
            [f"contract/{inherited_retry_manifest_name}"]
            if inherited_retry_manifest_name is not None
            else []
        )
        + (
            [f"contract/{ancestral_retry_manifest_name}"]
            if ancestral_retry_manifest_name is not None
            else []
        )
        + (
            [f"contract/{great_ancestral_retry_manifest_name}"]
            if great_ancestral_retry_manifest_name is not None
            else []
        )
        + (
            [f"contract/{deep_ancestral_retry_manifest_name}"]
            if deep_ancestral_retry_manifest_name is not None
            else []
        )
        + (
            [f"contract/{deepest_ancestral_retry_manifest_name}"]
            if deepest_ancestral_retry_manifest_name is not None
            else []
        )
        + ([f"contract/{base_contract_name}"] if base_contract_name is not None else [])
        + (
            ["parent_evidence_reference.json"]
            if parent_evidence_reference is not None
            else []
        )
    )
    manifest = {
        "schema_version": schema_version,
        "evidence_namespace": _evidence_namespace(contract),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "pilot_tag": contract.implementation["required_git_tag"],
        "resolved_git_commit": common_commit,
        "scientific_matrix_complete": aggregate["scientific_matrix_complete"],
        "scientific_claim_gates_supported": aggregate[
            "scientific_claim_gates_supported"
        ],
        "scientific_complete": aggregate["scientific_complete"],
        "publication_status": aggregate["publication_status"],
        "lane_separated": True,
        "direction_counts_merged": False,
        "narrative_status": "deferred-unregistered",
        "parent_source_manifest": {
            **parent_binding,
            "package_path": f"contract/{parent_manifest_name}",
        },
        "published_files": published_files,
        "excluded_sources": [
            HISTORICAL_SCOPE,
            "V2.3 scientific outcomes",
            "pooled local/GPT direction counts",
            "unregistered narrative intervention",
            "raw prompts and raw provider outputs",
        ]
        + (
            [
                (
                    "V2.9 offline candidate-admission outcomes as V2.10 "
                    "treatment-effect evidence"
                    if contract.contract_id == PILOT_V210_CONTRACT_ID
                    else "V2.8 treatment-effect outcomes"
                    if contract.contract_id == PILOT_V29_CONTRACT_ID
                    else "V2.7 treatment-effect outcomes"
                )
            ]
            if contract.contract_id == PILOT_V28_CONTRACT_ID
            or contract.contract_id == PILOT_V29_CONTRACT_ID
            or contract.contract_id == PILOT_V210_CONTRACT_ID
            else []
        ),
    }
    if sensitivity_manifest is not None:
        manifest["experiment_c_rule_sensitivities"] = sensitivity_manifest
    if retry_binding is not None and retry_manifest_name is not None:
        manifest["retry_source_manifest"] = {
            **retry_binding,
            "package_path": f"contract/{retry_manifest_name}",
        }
    if (
        inherited_retry_binding is not None
        and inherited_retry_manifest_name is not None
    ):
        manifest["inherited_retry_source_manifest"] = {
            **inherited_retry_binding,
            "package_path": f"contract/{inherited_retry_manifest_name}",
        }
    if (
        ancestral_retry_binding is not None
        and ancestral_retry_manifest_name is not None
    ):
        manifest["ancestral_retry_source_manifest"] = {
            **ancestral_retry_binding,
            "package_path": f"contract/{ancestral_retry_manifest_name}",
        }
    if (
        great_ancestral_retry_binding is not None
        and great_ancestral_retry_manifest_name is not None
    ):
        manifest["great_ancestral_retry_source_manifest"] = {
            **great_ancestral_retry_binding,
            "package_path": (f"contract/{great_ancestral_retry_manifest_name}"),
        }
    if (
        deep_ancestral_retry_binding is not None
        and deep_ancestral_retry_manifest_name is not None
    ):
        manifest["deep_ancestral_retry_source_manifest"] = {
            **deep_ancestral_retry_binding,
            "package_path": f"contract/{deep_ancestral_retry_manifest_name}",
        }
    if (
        deepest_ancestral_retry_binding is not None
        and deepest_ancestral_retry_manifest_name is not None
    ):
        manifest["deepest_ancestral_retry_source_manifest"] = {
            **deepest_ancestral_retry_binding,
            "package_path": f"contract/{deepest_ancestral_retry_manifest_name}",
        }
    if base_binding is not None and base_contract_name is not None:
        manifest["base_contract"] = {
            **base_binding,
            "package_path": f"contract/{base_contract_name}",
        }
    if parent_evidence_reference is not None:
        manifest["parent_evidence_reference"] = {
            **parent_evidence_reference,
            "package_path": "parent_evidence_reference.json",
        }
    manifest_path = root / "package_manifest.json"
    _atomic_bytes(manifest_path, _pretty_bytes(manifest))
    checksum_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.json"
    )
    checksums = {
        "schema_version": PILOT_CHECKSUM_SCHEMA_VERSION,
        "contract_sha256": contract.canonical_hash,
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "byte_size": path.stat().st_size,
            }
            for path in checksum_files
        ],
    }
    checksums_path = root / "checksums.json"
    _atomic_bytes(checksums_path, _pretty_bytes(checksums))
    for row in checksums["files"]:
        path = root / row["path"]
        if (
            _sha256_file(path) != row["sha256"]
            or path.stat().st_size != row["byte_size"]
        ):
            raise PilotEvidenceError(
                f"{version_label} package checksum self-verification failed"
            )
    raw_storage = aggregate["budget"].get("raw_root_storage_bytes")
    package_storage = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    if (
        not isinstance(raw_storage, (int, float))
        or isinstance(raw_storage, bool)
        or float(raw_storage) < 0
        or float(raw_storage) + package_storage
        > float(contract.budgets["max_storage_bytes"])
    ):
        raise PilotEvidenceError(
            f"{version_label} raw evidence plus reviewer package exceeds " "storage cap"
        )
    return manifest_path, checksums_path


def build_pilot_v24_evidence_package(
    *,
    contract_path: str | Path,
    run_ledger_path: str | Path,
    raw_root: str | Path,
    build_root: str | Path,
    source_repo_root: str | Path | None = None,
) -> PilotEvidencePackage:
    """Validate and publish a lane-separated package without provider calls."""

    contract_source = Path(contract_path).resolve()
    contract = load_pilot_contract(contract_source)
    _validate_v24_contract_matrix(contract)
    version_label = _contract_version_label(contract)
    if contract.status != "frozen":
        raise PilotEvidenceError(
            f"{version_label} evidence publication requires the frozen "
            "science contract"
        )
    raw = Path(raw_root).resolve()
    if not raw.is_dir():
        raise PilotEvidenceError(
            f"{version_label} pilot raw root does not exist: {raw}"
        )
    ledger = _strict_json_load(Path(run_ledger_path).resolve())
    with source_repository_context(
        source_repo_root,
        raw_root=raw,
    ) as source_root:
        rows, denominator, common_commit = _normalize_ledger(
            contract,
            ledger,
            raw_root=raw,
            source_repo_root=source_root,
        )
    release_controls = _validated_release_controls(
        contract,
        raw_root=raw,
        rows=rows,
        common_commit=common_commit,
        source_repo_root=source_root,
    )
    experiment_c_sensitivities: dict[str, dict[str, Any]] = {}
    if contract.contract_id == PILOT_V210_CONTRACT_ID:
        (
            experiment_c_sensitivities,
            sensitivity_controls,
        ) = _validated_v210_experiment_c_sensitivities(
            contract,
            raw_root=raw,
            rows=rows,
            common_commit=common_commit,
            source_repo_root=source_root,
        )
        release_controls["experiment_c_rule_sensitivities"] = (
            sensitivity_controls
        )
        release_controls["pass"] = bool(
            release_controls.get("pass") is True
            and all(control["pass"] for control in sensitivity_controls.values())
        )
    aggregate = aggregate_v24_evidence(
        contract,
        rows,
        denominator=denominator,
        release_controls=release_controls,
    )
    _require_publishable_terminal_denominator(aggregate)

    target = Path(build_root).resolve() / _evidence_namespace(contract)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}-build-",
            dir=target.parent,
        )
    )
    try:
        manifest, checksums = _write_v24_package(
            temporary,
            contract_path=contract_source,
            contract=contract,
            rows=rows,
            aggregate=aggregate,
            common_commit=common_commit,
            experiment_c_sensitivities=experiment_c_sensitivities,
        )
        _atomic_install_directory_no_replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return PilotEvidencePackage(
        package_dir=target,
        manifest_path=target / manifest.name,
        checksums_path=target / checksums.name,
        contract_hash=contract.canonical_hash,
        scientific_complete=bool(aggregate["scientific_complete"]),
        claim_gates={
            "lanes": _json_copy(aggregate["lanes"]),
            "cross_lane_mechanism_comparison": _json_copy(
                aggregate["cross_lane_mechanism_comparison"]
            ),
            "narrative": _json_copy(aggregate["narrative"]),
            "cross_lane_policy": _json_copy(aggregate["cross_lane_policy"]),
            **(
                {
                    "experiment_c_rule_sensitivities": _json_copy(
                        aggregate["experiment_c_rule_sensitivities"]
                    )
                }
                if contract.contract_id == PILOT_V210_CONTRACT_ID
                else {}
            ),
        },
    )


aggregate_lane_separated_evidence = aggregate_v24_evidence
build_lane_separated_evidence_package = build_pilot_v24_evidence_package


__all__ = [
    "PILOT_V24_CONTRACT_ID",
    "PILOT_V24_EVIDENCE_SCHEMA_VERSION",
    "PILOT_V24_MIN_PAIRED_SEEDS",
    "PILOT_V24_STAGE_ORDER",
    "PILOT_V24_TOTAL_PAIRED_SEEDS",
    "PILOT_V25_CONTRACT_ID",
    "PILOT_V25_EVIDENCE_SCHEMA_VERSION",
    "PILOT_V26_CONTRACT_ID",
    "PILOT_V26_EVIDENCE_SCHEMA_VERSION",
    "PILOT_V27_CONTRACT_ID",
    "PILOT_V27_EVIDENCE_SCHEMA_VERSION",
    "PILOT_V28_CONTRACT_ID",
    "PILOT_V28_EVIDENCE_SCHEMA_VERSION",
    "PILOT_V29_CONTRACT_ID",
    "PILOT_V29_EVIDENCE_SCHEMA_VERSION",
    "PILOT_V210_CONTRACT_ID",
    "PILOT_V210_EVIDENCE_SCHEMA_VERSION",
    "aggregate_lane_separated_evidence",
    "aggregate_v24_evidence",
    "build_lane_separated_evidence_package",
    "build_pilot_v24_evidence_package",
]
