"""Fail-closed controls for the one authorized pilot-v2.1 retry.

Pilot-v2 remains an immutable failed attempt.  The v2.1 overlay may dispatch
only the previously non-evaluable GPT-5.2 capability cell.  The valid local
Llama capability no-go is inherited into the new denominator without a
provider call, and the exact conservative parent-ledger debit is imported into
the new budget ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from .pilot_budget import ParentBudgetDebit
from .pilot_contract import (
    PILOT_CONTRACT_V2_SCIENCE_DESIGN_SHA256,
    PilotContract,
    PilotRunSpec,
    canonical_sha256,
    science_design_sha256,
)


PILOT_V21_CONTRACT_ID = "finevo-pilot-v2.1"
PILOT_V21_TAG = "pilot-v2.1-science"
PILOT_V2_FAILURE_RECEIPT_SCHEMA_VERSION = (
    "finevo-pilot-v2-operational-failure-receipt-v1"
)
PILOT_V2_FAILURE_RECEIPT_RELATIVE_PATH = Path(
    "experiments/pilot_v2_failure_receipt.json"
)
PILOT_V21_RAW_RECEIPT_FILENAME = "operational_amendment_receipt.json"

PARENT_CONTRACT_ID = "finevo-pilot-v2"
PARENT_CONTRACT_SHA256 = (
    "980deddf2f82a762db7d73baa6ee0428c5e653298f4f275c5b3a5b23a95865c5"
)
PARENT_RUN_LEDGER_SHA256 = (
    "9d54ac1f22a56bafbe59164c7074d87bf914d290bc1282c631d50a4529f41fff"
)
PARENT_BUDGET_LEDGER_SHA256 = (
    "d9ec2c1bdfcc407aeb555ba71ee9d5e274d924e9a98791c5839ec749f3b1a0f2"
)
PARENT_RUN_LEDGER_EVENT_HEAD = (
    "5e202f541a5c58f53fe2e79855bcadd3e954845f0788c017caf1986bbb9fe1cc"
)
PARENT_BUDGET_LEDGER_EVENT_HEAD = (
    "2b49f53463e0c7e5b6080906a2faae1799cc3489c9c863a51033429ac3ffb908"
)
PARENT_DEBIT_RECORD_SHA256 = (
    "b2bb046832de4aa324a641920ab6a2f7dcb48168706a88dbd0d4fae54d1a4e50"
)
SCIENCE_DESIGN_BASE_SHA256 = PILOT_CONTRACT_V2_SCIENCE_DESIGN_SHA256
EXECUTION_INVARIANTS_BASE_SHA256 = (
    "454c20476131406b8976e9a18e21daeca082987931942f9fc139d2c63d3d773b"
)

GPT52_PROFILE_ID = "gpt52_main"
LLAMA33_PROFILE_ID = "llama33_local_controlled"


class PilotAmendmentError(RuntimeError):
    """Raised before dispatch when the operational amendment is not exact."""


def _strict_json_bytes(raw: bytes, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise PilotAmendmentError(f"non-finite JSON constant is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PilotAmendmentError(
                    f"duplicate amendment receipt key is forbidden: {key!r}"
                )
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8", "strict")
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotAmendmentError(
            f"cannot read operational amendment receipt: {name}"
        ) from exc
    if not isinstance(value, dict):
        raise PilotAmendmentError("operational amendment receipt must be an object")
    return value


def _read_regular_bytes_once(
    path: Path,
    *,
    name: str,
) -> bytes:
    """Read one regular file through one no-follow descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    elif path.is_symlink():  # pragma: no cover - Linux/macOS expose O_NOFOLLOW
        raise PilotAmendmentError(f"{name} must not be a symlink")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PilotAmendmentError(f"{name} must be a regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            raw = stream.read()
            after = os.fstat(stream.fileno())
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise PilotAmendmentError(f"{name} changed while it was read")
        return raw
    except OSError as exc:
        raise PilotAmendmentError(f"cannot safely read {name}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotAmendmentError(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise PilotAmendmentError(
            f"{name} has wrong fields: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _finite_equal(actual: Any, expected: float, name: str) -> None:
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(float(actual))
        or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise PilotAmendmentError(f"{name} differs from the frozen parent value")


def _execution_invariants_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project design fields plus global limits, excluding amended stage caps."""

    fields = {
        key: value[key]
        for key in (
            "seeds",
            "provider_profiles",
            "parameter_dispatch_policy",
            "task_output_contracts",
            "model_roles",
            "arms",
            "narratives",
            "shocks",
            "utility",
            "stop_go",
            "stages",
        )
    }
    denominator = dict(
        _mapping(value["denominator_policy"], "denominator_policy")
    )
    denominator.pop("policy_id", None)
    budgets = dict(_mapping(value["budgets"], "budgets"))
    budgets.pop("stage_usd_caps", None)
    fields["denominator_policy_without_id"] = denominator
    fields["budgets_without_stage_caps"] = budgets
    return fields


def execution_invariants_sha256(contract_value: Mapping[str, Any]) -> str:
    """Hash the frozen design/runtime invariants outside stage-cap reallocation."""

    return canonical_sha256(_execution_invariants_projection(contract_value))


def _validate_receipt(
    value: Mapping[str, Any],
    *,
    contract: PilotContract,
) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "status",
            "operational_amendment",
            "parent_ledger_summary",
            "evidence_abort",
            "science_freeze",
            "integrity",
        },
        "operational amendment receipt",
    )
    if (
        value["schema_version"] != PILOT_V2_FAILURE_RECEIPT_SCHEMA_VERSION
        or value["status"] != "frozen"
    ):
        raise PilotAmendmentError("unsupported or unfrozen amendment receipt")

    unsigned = dict(value)
    integrity = _mapping(unsigned.pop("integrity"), "receipt integrity")
    _exact_keys(
        integrity,
        {"canonicalization", "content_sha256"},
        "receipt integrity",
    )
    if (
        integrity["canonicalization"] != "json-sort-keys-utf8-v1"
        or integrity["content_sha256"] != canonical_sha256(unsigned)
    ):
        raise PilotAmendmentError("operational amendment receipt hash mismatch")

    amendment = getattr(contract, "operational_amendment", None)
    if not isinstance(amendment, Mapping):
        raise PilotAmendmentError("pilot-v2.1 lacks operational_amendment metadata")
    if dict(value["operational_amendment"]) != contract.to_dict().get(
        "operational_amendment"
    ):
        raise PilotAmendmentError(
            "tracked receipt amendment differs from the loaded contract"
        )

    summary = _mapping(value["parent_ledger_summary"], "parent ledger summary")
    _exact_keys(
        summary,
        {
            "run_ledger_event_head",
            "registered_cell_count",
            "run_status_counts",
            "budget_ledger_event_head",
        },
        "parent ledger summary",
    )
    if summary != {
        "run_ledger_event_head": PARENT_RUN_LEDGER_EVENT_HEAD,
        "registered_cell_count": 174,
        "run_status_counts": {"capability-no-go": 149, "scheduled": 25},
        "budget_ledger_event_head": PARENT_BUDGET_LEDGER_EVENT_HEAD,
    }:
        raise PilotAmendmentError("parent ledger summary drifted")

    abort = _mapping(value["evidence_abort"], "evidence abort")
    _exact_keys(
        abort,
        {"error_type", "message", "offline_reproduced", "code"},
        "evidence abort",
    )
    if abort != {
        "error_type": "PilotEvidenceError",
        "message": "capability v3 row 'action-01' lacks served_model",
        "offline_reproduced": True,
        "code": "null-served-model-pre-response-validator-contradiction",
    }:
        raise PilotAmendmentError("evidence-abort diagnosis drifted")

    freeze = _mapping(value["science_freeze"], "science freeze")
    _exact_keys(
        freeze,
        {
            "science_design_projection",
            "base_science_design_sha256",
            "amended_science_design_sha256",
            "science_design_field_changes",
            "execution_invariants_projection",
            "base_execution_invariants_sha256",
            "amended_execution_invariants_sha256",
            "operational_budget_cap_changes",
            "scientific_effect_outcomes_inspected_for_retry",
            "capability_gate_outcomes_inspected",
            "inherited_result_scope",
        },
        "science freeze",
    )
    if freeze != {
        "science_design_projection": "finevo-pilot-v2-science-design-v1",
        "base_science_design_sha256": SCIENCE_DESIGN_BASE_SHA256,
        "amended_science_design_sha256": SCIENCE_DESIGN_BASE_SHA256,
        "science_design_field_changes": [],
        "execution_invariants_projection": (
            "finevo-pilot-v2-execution-invariants-v1"
        ),
        "base_execution_invariants_sha256": (
            EXECUTION_INVARIANTS_BASE_SHA256
        ),
        "amended_execution_invariants_sha256": (
            EXECUTION_INVARIANTS_BASE_SHA256
        ),
        "operational_budget_cap_changes": [
            {
                "path": "budgets.stage_usd_caps.capability",
                "parent_value": 2.0,
                "amended_value": 3.0701145,
            },
            {
                "path": "budgets.stage_usd_caps.cross_model",
                "parent_value": 6.0,
                "amended_value": 4.9298855,
            },
        ],
        "scientific_effect_outcomes_inspected_for_retry": False,
        "capability_gate_outcomes_inspected": True,
        "inherited_result_scope": "capability-only-no-go",
    }:
        raise PilotAmendmentError("science-freeze receipt drifted")

    if science_design_sha256(contract.to_dict()) != SCIENCE_DESIGN_BASE_SHA256:
        raise PilotAmendmentError(
            "pilot-v2.1 science-design fields differ from frozen pilot-v2"
        )
    if (
        execution_invariants_sha256(contract.to_dict())
        != EXECUTION_INVARIANTS_BASE_SHA256
    ):
        raise PilotAmendmentError(
            "pilot-v2.1 execution invariants differ from frozen pilot-v2"
        )


def load_operational_failure_receipt(
    *,
    repo_root: str | Path,
    contract: PilotContract,
) -> tuple[dict[str, Any], Path]:
    """Load the tracked secret-free receipt and prove its contract equality."""

    value, path, _ = _load_operational_failure_receipt_once(
        repo_root=repo_root,
        contract=contract,
    )
    return value, path


def _load_operational_failure_receipt_once(
    *,
    repo_root: str | Path,
    contract: PilotContract,
) -> tuple[dict[str, Any], Path, bytes]:
    """Read, parse, and validate the tracked receipt from one byte buffer."""

    if contract.contract_id != PILOT_V21_CONTRACT_ID:
        raise PilotAmendmentError("operational receipt is only valid for pilot-v2.1")
    if contract.implementation.get("required_git_tag") != PILOT_V21_TAG:
        raise PilotAmendmentError("pilot-v2.1 requires its own annotated tag")

    root = Path(repo_root).resolve()
    path = root / PILOT_V2_FAILURE_RECEIPT_RELATIVE_PATH
    try:
        parent = path.parent.resolve(strict=True)
        parent.relative_to(root)
    except (FileNotFoundError, ValueError) as exc:
        raise PilotAmendmentError(
            "tracked amendment receipt is missing or escapes the repository"
        ) from exc
    if parent != root / "experiments":
        raise PilotAmendmentError(
            "tracked amendment receipt must remain in experiments"
        )
    source_bytes = _read_regular_bytes_once(
        path,
        name="tracked amendment receipt",
    )
    value = _strict_json_bytes(
        source_bytes,
        name="tracked amendment receipt",
    )
    _validate_receipt(value, contract=contract)
    return value, path, source_bytes


def persist_operational_failure_receipt(
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    contract: PilotContract,
) -> tuple[dict[str, Any], Path]:
    """Copy the exact tracked bytes into raw evidence, without overwrite."""

    value, _, source_bytes = _load_operational_failure_receipt_once(
        repo_root=repo_root,
        contract=contract,
    )
    raw = Path(raw_root).resolve()
    raw.mkdir(parents=True, exist_ok=True)
    output = raw / PILOT_V21_RAW_RECEIPT_FILENAME
    descriptor: int | None = None
    try:
        descriptor = os.open(
            output,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(source_bytes)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        existing = _read_regular_bytes_once(
            output,
            name="persisted operational amendment receipt",
        )
        if existing != source_bytes:
            raise PilotAmendmentError(
                "persisted or concurrent amendment receipt differs from tracked bytes"
            )
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return value, output


def parent_budget_debit_for_contract(
    contract: PilotContract,
) -> ParentBudgetDebit | None:
    """Construct the inherited parent-ledger debit, or none for other contracts."""

    if contract.contract_id != PILOT_V21_CONTRACT_ID:
        return None
    amendment = _mapping(
        getattr(contract, "operational_amendment", None),
        "operational_amendment",
    )
    parent = _mapping(amendment["parent"], "operational_amendment.parent")
    carry = _mapping(
        amendment["budget_carry_forward"],
        "operational_amendment.budget_carry_forward",
    )
    debit = ParentBudgetDebit(
        parent_contract_sha256=str(parent["contract_sha256"]),
        parent_run_ledger_sha256=str(parent["run_ledger_internal_sha256"]),
        parent_budget_ledger_sha256=str(parent["budget_ledger_internal_sha256"]),
        stage_bucket=str(carry["source_stage_bucket"]),
        cost_usd=float(carry["cost_usd"]),
        hosted_completions=int(carry["hosted_completions"]),
        storage_bytes=int(carry["storage_bytes"]),
    )
    if (
        debit.parent_contract_sha256 != PARENT_CONTRACT_SHA256
        or debit.parent_run_ledger_sha256 != PARENT_RUN_LEDGER_SHA256
        or debit.parent_budget_ledger_sha256 != PARENT_BUDGET_LEDGER_SHA256
        or debit.record_sha256 != PARENT_DEBIT_RECORD_SHA256
    ):
        raise PilotAmendmentError("derived parent budget debit drifted")
    _finite_equal(debit.cost_usd, 1.0701145, "parent debit cost")
    if (
        debit.stage_bucket != "capability"
        or debit.hosted_completions != 30
        or debit.storage_bytes != 479_367
    ):
        raise PilotAmendmentError("derived parent budget debit totals drifted")
    return debit


def _inherited_failure(receipt: Mapping[str, Any]) -> dict[str, Any]:
    amendment = _mapping(
        receipt["operational_amendment"],
        "receipt operational amendment",
    )
    inherited_values = amendment["inherited_results"]
    if not isinstance(inherited_values, list) or len(inherited_values) != 1:
        raise PilotAmendmentError("receipt must inherit exactly one result")
    inherited = _mapping(inherited_values[0], "inherited result")
    integrity = _mapping(receipt["integrity"], "receipt integrity")
    return {
        "error_type": "InheritedCapabilityNoGo",
        "message": (
            "valid pilot-v2 local-Llama capability no-go is carried forward; "
            "redispatch is forbidden"
        ),
        "source_contract_id": PARENT_CONTRACT_ID,
        "source_contract_sha256": PARENT_CONTRACT_SHA256,
        "source_run_id": inherited["run_id"],
        "source_capability_sha256": inherited["capability_sha256"],
        "source_gate_sha256": inherited["gate_sha256"],
        "source_terminal_sha256": inherited["terminal_sha256"],
        "source_scores": dict(inherited["scores"]),
        "amendment_receipt": PILOT_V21_RAW_RECEIPT_FILENAME,
        "amendment_receipt_content_sha256": integrity["content_sha256"],
        "provider_calls_current_attempt": 0,
        "redispatch_forbidden": True,
    }


def apply_inherited_capability_no_go(
    *,
    contract: PilotContract,
    run_ledger: Any,
    receipt: Mapping[str, Any],
) -> PilotRunSpec:
    """Terminalize the inherited local capability cell, idempotently."""

    if contract.contract_id != PILOT_V21_CONTRACT_ID:
        raise PilotAmendmentError("capability inheritance is only valid for pilot-v2.1")
    specs = contract.expand(stage="capability-gate", model=LLAMA33_PROFILE_ID)
    if len(specs) != 1:
        raise PilotAmendmentError(
            "pilot-v2.1 must contain one exact local-Llama capability cell"
        )
    spec = specs[0]
    run_ledger.finalize(
        spec.run_id,
        status="capability-no-go",
        artifact=None,
        failure=_inherited_failure(receipt),
    )
    return spec


def assert_amended_capability_dispatch_scope(
    *,
    contract: PilotContract,
    run_ledger: Any,
) -> None:
    """Prove GPT-5.2 is the only cell that can dispatch in the primary gate."""

    if contract.contract_id != PILOT_V21_CONTRACT_ID:
        return
    specs = contract.expand(stage="capability-gate")
    by_model = {spec.model_id: spec for spec in specs}
    if set(by_model) != {GPT52_PROFILE_ID, LLAMA33_PROFILE_ID}:
        raise PilotAmendmentError("pilot-v2.1 primary capability matrix drifted")
    if run_ledger.status(by_model[LLAMA33_PROFILE_ID].run_id) != "capability-no-go":
        raise PilotAmendmentError(
            "inherited local-Llama capability cell is not terminal no-go"
        )
    nonterminal = [
        spec.model_id for spec in specs if not run_ledger.is_terminal(spec.run_id)
    ]
    if nonterminal not in ([], [GPT52_PROFILE_ID]):
        raise PilotAmendmentError(
            "only GPT-5.2 may remain dispatchable in the amended primary gate"
        )


def amendment_control_path(
    *,
    contract: PilotContract,
    raw_root: str | Path,
    stage_id: str,
) -> Path | None:
    """Return the persisted receipt that must bind the capability stage."""

    if (
        contract.contract_id == PILOT_V21_CONTRACT_ID
        and stage_id == "capability-gate"
    ):
        return Path(raw_root).resolve() / PILOT_V21_RAW_RECEIPT_FILENAME
    return None


def receipt_file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "GPT52_PROFILE_ID",
    "LLAMA33_PROFILE_ID",
    "PILOT_V21_CONTRACT_ID",
    "PILOT_V21_RAW_RECEIPT_FILENAME",
    "PILOT_V2_FAILURE_RECEIPT_RELATIVE_PATH",
    "PILOT_V2_FAILURE_RECEIPT_SCHEMA_VERSION",
    "EXECUTION_INVARIANTS_BASE_SHA256",
    "SCIENCE_DESIGN_BASE_SHA256",
    "PilotAmendmentError",
    "amendment_control_path",
    "apply_inherited_capability_no_go",
    "assert_amended_capability_dispatch_scope",
    "load_operational_failure_receipt",
    "parent_budget_debit_for_contract",
    "persist_operational_failure_receipt",
    "receipt_file_sha256",
    "execution_invariants_sha256",
    "science_design_sha256",
]
