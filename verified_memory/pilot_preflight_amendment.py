"""Fail-closed controls for the pilot-v2.3 preflight bootstrap amendment.

The closed-loop preflight is the measurement that produces the sealed
model-by-call-kind p95 used by later scientific runs.  Pilot-v2.2 attempted to
apply that later-run requirement to the measurement itself, creating a cycle:
the preflight could not dispatch until its own output already existed.

V2.3 does not weaken the runner-wide guard.  It derives a separate bootstrap
authority from the already observed, contract-bound 30-task capability rows,
normalizes the two output contracts to the runner's ``action``/``semantic``
call kinds, and allows that authority only for registered
``closed_loop_preflight`` cells.  A successful preflight then emits the normal
sealed observed-plus-25-percent projection used everywhere else.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .pilot_budget import ParentBudgetDebit, preflight_p95
from .pilot_capability import (
    CAPABILITY_SCHEMA_VERSION,
    CAPABILITY_TASKSET_SHA256,
    build_capability_tasks,
)
from .pilot_contract import PilotContract, PilotRunSpec, canonical_sha256
from .pilot_evaluation_amendment import CAPABILITY_IMPORT_SCHEMA_VERSION
from .pilot_evidence import PilotEvidenceError, _validate_capability_v4


PREFLIGHT_BOOTSTRAP_AMENDMENT_SCHEMA_VERSION = (
    "finevo-pilot-preflight-bootstrap-amendment-v1"
)
PREFLIGHT_BOOTSTRAP_AMENDMENT_ID = (
    "finevo-pilot-v2.3-closed-loop-preflight-bootstrap-1"
)
PREFLIGHT_BOOTSTRAP_CONTROL_SCHEMA_VERSION = (
    "finevo-pilot-preflight-bootstrap-control-v1"
)
PREFLIGHT_BOOTSTRAP_PROJECTION_SCHEMA_VERSION = (
    "finevo-capability-bootstrap-projection-v1"
)
PREFLIGHT_BOOTSTRAP_CONTROL_FILENAME = (
    "preflight_bootstrap_amendment_control.json"
)
PREFLIGHT_BOOTSTRAP_PROJECTION_FILENAME = "bootstrap_projection_p95.json"
V23_PARENT_DEBIT_RECORD_SHA256 = (
    "71a1349168861a0ff9dc1546a394e7e0da6ea783e7029e4b3b1056c23906ff59"
)


class PilotPreflightAmendmentError(RuntimeError):
    """Raised before provider construction when bootstrap authority is invalid."""


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _strict_json_object_from_bytes(
    value: bytes,
    *,
    name: str,
) -> dict[str, Any]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise PilotPreflightAmendmentError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            result[key] = item
        return result

    def reject_nonfinite(item: str) -> None:
        raise PilotPreflightAmendmentError(
            f"{name} contains non-finite JSON number {item}"
        )

    try:
        parsed = json.loads(
            value.decode("utf-8", "strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotPreflightAmendmentError(
            f"{name} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise PilotPreflightAmendmentError(f"{name} must contain a JSON object")
    return parsed


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotPreflightAmendmentError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PilotPreflightAmendmentError(f"{name} must be an array")
    return value


def _amendment(
    contract: PilotContract | Mapping[str, Any],
) -> Mapping[str, Any] | None:
    value = (
        contract
        if isinstance(contract, Mapping)
        else contract.to_dict()
    )
    amendment = value.get("preflight_bootstrap_amendment")
    if amendment is None:
        return None
    result = _mapping(amendment, "preflight_bootstrap_amendment")
    if (
        result.get("schema_version")
        != PREFLIGHT_BOOTSTRAP_AMENDMENT_SCHEMA_VERSION
        or result.get("amendment_id") != PREFLIGHT_BOOTSTRAP_AMENDMENT_ID
    ):
        raise PilotPreflightAmendmentError(
            "preflight bootstrap amendment identity drifted"
        )
    return result


def build_preflight_amendment_control(
    contract: PilotContract,
) -> dict[str, Any] | None:
    """Return a self-hashed raw control copy of the tracked amendment."""

    amendment = _amendment(contract)
    if amendment is None:
        return None
    payload: dict[str, Any] = {
        "schema_version": PREFLIGHT_BOOTSTRAP_CONTROL_SCHEMA_VERSION,
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "amendment": _json_copy(amendment),
        "scientific_evidence": False,
        "evidence_use": "preflight budget authority and parent failure audit only",
    }
    payload["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(payload),
    }
    return payload


def validate_preflight_amendment_control(
    value: Mapping[str, Any],
    contract: PilotContract,
) -> None:
    expected = build_preflight_amendment_control(contract)
    if expected is None:
        raise PilotPreflightAmendmentError(
            "contract has no preflight bootstrap amendment"
        )
    if dict(value) != expected:
        raise PilotPreflightAmendmentError(
            "persisted preflight amendment control differs from the contract"
        )


def preflight_amendment_control_path(*, raw_root: str | Path) -> Path:
    return Path(raw_root).resolve() / PREFLIGHT_BOOTSTRAP_CONTROL_FILENAME


def parent_budget_debit_for_preflight_amendment(
    contract: PilotContract | Mapping[str, Any],
) -> ParentBudgetDebit | None:
    amendment = _amendment(contract)
    if amendment is None:
        return None
    carry = _mapping(amendment.get("budget_carry_forward"), "budget carry")
    parent = _mapping(amendment.get("parent"), "amendment parent")
    try:
        debit = ParentBudgetDebit(
            parent_contract_sha256=str(carry["source_contract_sha256"]),
            parent_run_ledger_sha256=str(
                parent["run_ledger_internal_sha256"]
            ),
            parent_budget_ledger_sha256=str(
                parent["budget_ledger_internal_sha256"]
            ),
            stage_bucket=str(carry["source_stage_bucket"]),
            cost_usd=float(carry["cost_usd"]),
            hosted_completions=int(carry["hosted_completions"]),
            storage_bytes=int(carry["storage_bytes"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PilotPreflightAmendmentError(
            "cannot derive V2.3 parent budget debit"
        ) from exc
    if debit.record_sha256 != V23_PARENT_DEBIT_RECORD_SHA256:
        raise PilotPreflightAmendmentError(
            "V2.3 parent budget debit differs from the frozen cumulative debit"
        )
    return debit


def _runtime_model(contract: PilotContract, model_id: str) -> str:
    profile = contract.provider_profiles[model_id]
    provider = {
        "openai": "openai",
        "openrouter": "thirdparty",
        "ollama": "ollama",
    }.get(profile.transport)
    if provider is None:
        raise PilotPreflightAmendmentError(
            f"{model_id} has no dispatchable bootstrap transport"
        )
    return f"{provider}/{profile.requested_model}"


def _capability_rows(capability: Mapping[str, Any]) -> Sequence[Any]:
    schema = capability.get("schema_version")
    if schema == CAPABILITY_SCHEMA_VERSION:
        return _sequence(capability.get("rows"), "capability rows")
    if schema == CAPABILITY_IMPORT_SCHEMA_VERSION:
        return _sequence(
            capability.get("usage_projection_rows"),
            "capability usage projection rows",
        )
    raise PilotPreflightAmendmentError(
        f"unsupported bootstrap capability schema {schema!r}"
    )


def _validate_capability_source(
    contract: PilotContract,
    spec: PilotRunSpec,
    capability: Mapping[str, Any],
) -> Sequence[Any]:
    if (
        spec.execution_mode != "capability_probe"
        or capability.get("pass") is not True
        or not isinstance(capability.get("interface_gate"), Mapping)
        or capability["interface_gate"].get("pass") is not True
        or capability.get("contract_sha256") != contract.canonical_hash
    ):
        raise PilotPreflightAmendmentError(
            "bootstrap source is not a passing contract-bound capability probe"
        )
    schema = capability.get("schema_version")
    if schema == CAPABILITY_SCHEMA_VERSION:
        if (
            capability.get("run_spec") != spec.to_dict()
            or capability.get("taskset_sha256")
            != CAPABILITY_TASKSET_SHA256
        ):
            raise PilotPreflightAmendmentError(
                "fresh capability source contract/spec/taskset binding drifted"
            )
        try:
            _validate_capability_v4(capability)
        except PilotEvidenceError as exc:
            raise PilotPreflightAmendmentError(
                f"fresh capability source failed semantic validation: {exc}"
            ) from exc
    elif schema == CAPABILITY_IMPORT_SCHEMA_VERSION:
        unsigned = _json_copy(capability)
        integrity = _mapping(
            unsigned.pop("integrity", None),
            "capability import integrity",
        )
        if (
            set(integrity)
            != {"canonicalization", "content_sha256"}
            or integrity.get("canonicalization")
            != "json-sort-keys-utf8-v1"
            or integrity.get("content_sha256")
            != canonical_sha256(unsigned)
            or capability.get("contract_id") != contract.contract_id
            or capability.get("target_run_id") != spec.run_id
            or capability.get("target_execution_mode")
            != "capability_probe"
            or capability.get("model_id") != spec.model_id
            or capability.get("taskset_sha256")
            != CAPABILITY_TASKSET_SHA256
            or capability.get("provider_calls_current_attempt") != 0
        ):
            raise PilotPreflightAmendmentError(
                "imported capability source provenance or integrity drifted"
            )
    else:
        raise PilotPreflightAmendmentError(
            f"unsupported bootstrap capability schema {schema!r}"
        )
    rows = _capability_rows(capability)
    expected_tasks = build_capability_tasks()
    expected_ids = [task.task_id for task in expected_tasks]
    observed_ids = [
        str(_mapping(row, f"capability row {index}").get("task_id"))
        for index, row in enumerate(rows)
    ]
    if observed_ids != expected_ids or len(set(observed_ids)) != len(expected_ids):
        raise PilotPreflightAmendmentError(
            "capability bootstrap task denominator/order drifted"
        )
    return rows


def _normalized_capability_usage_rows(
    contract: PilotContract,
    spec: PilotRunSpec,
    capability: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    amendment = _amendment(contract)
    if amendment is None:
        raise PilotPreflightAmendmentError(
            "bootstrap projection requires the V2.3 amendment"
        )
    capability_rows = _validate_capability_source(
        contract,
        spec,
        capability,
    )
    policy = _mapping(amendment["bootstrap_policy"], "bootstrap policy")
    kind_map = _mapping(
        policy["source_output_contract_map"],
        "bootstrap source output-contract map",
    )
    required_counts = _mapping(
        policy["required_sample_counts"],
        "bootstrap required sample counts",
    )
    runtime_model = _runtime_model(contract, spec.model_id)
    frozen_served_model = contract.provider_profiles[spec.model_id].served_model
    normalized: list[dict[str, Any]] = []
    expected_contract_by_task = {
        task.task_id: task.output_contract_id
        for task in build_capability_tasks()
    }
    for index, value in enumerate(capability_rows):
        row = _mapping(value, f"capability row {index}")
        output_contract = row.get("output_contract_id")
        if (
            not isinstance(output_contract, str)
            or (
                row.get("call_kind") is not None
                and row.get("call_kind") != output_contract
            )
            or output_contract
            != expected_contract_by_task[str(row["task_id"])]
        ):
            raise PilotPreflightAmendmentError(
                f"capability row {index} output-contract binding drifted"
            )
        call_kind = kind_map.get(output_contract)
        if call_kind not in {"action", "semantic"}:
            raise PilotPreflightAmendmentError(
                f"capability row {index} has an unregistered output contract"
            )
        served_model = row.get("served_model")
        response_model = row.get("response_model")
        if (
            served_model is not None
            and response_model is not None
            and served_model != response_model
        ):
            raise PilotPreflightAmendmentError(
                f"capability row {index} served-model fields conflict"
            )
        observed_model = (
            served_model if served_model is not None else response_model
        )
        if observed_model != frozen_served_model:
            raise PilotPreflightAmendmentError(
                f"capability row {index} served-model binding drifted"
            )
        usage = _mapping(row.get("usage"), f"capability row {index} usage")
        if set(usage) != {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cost_usd",
        }:
            raise PilotPreflightAmendmentError(
                f"capability row {index} usage fields drifted"
            )
        numeric: dict[str, float | int] = {}
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cost_usd",
        ):
            item = usage.get(field)
            if (
                isinstance(item, bool)
                or (
                    field != "cost_usd"
                    and (not isinstance(item, int) or item < 0)
                )
                or (
                    field == "cost_usd"
                    and (
                        not isinstance(item, (int, float))
                        or not math.isfinite(float(item))
                        or float(item) < 0
                    )
                )
            ):
                raise PilotPreflightAmendmentError(
                    f"capability row {index} has invalid {field}"
                )
            numeric[field] = item
        if numeric["total_tokens"] != (
            numeric["prompt_tokens"] + numeric["completion_tokens"]
        ):
            raise PilotPreflightAmendmentError(
                f"capability row {index} token accounting is not additive"
            )
        normalized.append(
            {
                "response_model": runtime_model,
                "call_kind": call_kind,
                "usage": _json_copy(usage),
            }
        )
    observed_counts = {
        kind: sum(row["call_kind"] == kind for row in normalized)
        for kind in ("action", "semantic")
    }
    expected_counts = {
        kind: int(required_counts[kind])
        for kind in ("action", "semantic")
    }
    if observed_counts != expected_counts:
        raise PilotPreflightAmendmentError(
            "capability bootstrap sample denominator drifted"
        )
    return normalized, runtime_model


def build_capability_bootstrap_projection(
    contract: PilotContract,
    spec: PilotRunSpec,
    target_preflight_spec: PilotRunSpec,
    capability: Mapping[str, Any],
    *,
    source_capability_path: str | Path,
    source_capability_file_sha256: str,
    git_tag: str,
    git_commit: str,
    authorized_config_sha256: str,
) -> dict[str, Any]:
    """Derive the exact per-call authority for one later preflight."""

    registered_specs = contract.expand()
    if (
        not isinstance(spec, PilotRunSpec)
        or sum(candidate == spec for candidate in registered_specs) != 1
        or not isinstance(target_preflight_spec, PilotRunSpec)
        or sum(
            candidate == target_preflight_spec
            for candidate in registered_specs
        )
        != 1
    ):
        raise PilotPreflightAmendmentError(
            "bootstrap source and target must be exact registered contract cells"
        )
    source_input = Path(source_capability_path)
    try:
        if source_input.is_symlink() or not source_input.is_file():
            raise PilotPreflightAmendmentError(
                "bootstrap source capability must be a regular non-symlink file"
            )
        source_bytes = source_input.read_bytes()
        actual_source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    except OSError as exc:
        raise PilotPreflightAmendmentError(
            "bootstrap source capability cannot be read"
        ) from exc
    if actual_source_sha256 != source_capability_file_sha256:
        raise PilotPreflightAmendmentError(
            "bootstrap source capability file hash mismatch"
        )
    source_document = _strict_json_object_from_bytes(
        source_bytes,
        name="bootstrap source capability",
    )
    try:
        supplied_capability = _json_copy(capability)
    except (TypeError, ValueError) as exc:
        raise PilotPreflightAmendmentError(
            "bootstrap supplied capability is not canonical JSON"
        ) from exc
    if source_document != supplied_capability:
        raise PilotPreflightAmendmentError(
            "bootstrap source capability file/payload mismatch"
        )
    if (
        not isinstance(git_commit, str)
        or len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        raise PilotPreflightAmendmentError(
            "bootstrap git_commit must be a lowercase 40-hex commit"
        )

    rows, runtime_model = _normalized_capability_usage_rows(
        contract,
        spec,
        capability,
    )
    if (
        target_preflight_spec.execution_mode != "closed_loop_preflight"
        or target_preflight_spec.model_id != spec.model_id
        or target_preflight_spec.environment_seed
        != contract.seeds["preflight_seed"]
        or git_tag != contract.implementation["required_git_tag"]
    ):
        raise PilotPreflightAmendmentError(
            "bootstrap target preflight cell/model/seed/tag drifted"
        )
    for name, value in (
        ("source_capability_file_sha256", source_capability_file_sha256),
        ("authorized_config_sha256", authorized_config_sha256),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise PilotPreflightAmendmentError(
                f"{name} must be a lowercase SHA-256 digest"
            )
    amendment = _mapping(
        _amendment(contract),
        "preflight bootstrap amendment",
    )
    policy = _mapping(amendment["bootstrap_policy"], "bootstrap policy")
    projection = preflight_p95(
        rows,
        reserve_multiplier=float(policy["reserve_multiplier"]),
    )
    expected_keys = {
        f"{runtime_model}::action",
        f"{runtime_model}::semantic",
    }
    if set(projection) != expected_keys:
        raise PilotPreflightAmendmentError(
            "bootstrap projection lacks exact runner call kinds"
        )
    profile = contract.provider_profiles[spec.model_id]
    for key, value in projection.items():
        reserved = _mapping(value.get("reserved_p95"), f"{key} reservation")
        if (
            int(reserved.get("prompt_tokens", 0)) < 1
            or int(reserved.get("completion_tokens", 0)) < 1
        ):
            raise PilotPreflightAmendmentError(
                f"{key} bootstrap reservation must reserve positive tokens"
            )
        if profile.transport in {"openai", "openrouter"} and float(
            reserved.get("cost_usd", 0.0)
        ) <= 0:
            raise PilotPreflightAmendmentError(
                f"{key} hosted bootstrap reservation must reserve positive cost"
            )

    source = source_input.resolve()
    payload: dict[str, Any] = {
        "schema_version": PREFLIGHT_BOOTSTRAP_PROJECTION_SCHEMA_VERSION,
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "model_id": spec.model_id,
        "runtime_model": runtime_model,
        "source_run_id": spec.run_id,
        "bindings": {
            "amendment_id": PREFLIGHT_BOOTSTRAP_AMENDMENT_ID,
            "git_tag": git_tag,
            "git_commit": git_commit,
            "source_capability": str(source),
            "source_capability_file_sha256": source_capability_file_sha256,
            "target_run_id": target_preflight_spec.run_id,
            "target_run_spec_sha256": canonical_sha256(
                target_preflight_spec.to_dict()
            ),
            "authorized_runner_run_id": (
                f"{target_preflight_spec.run_id}--actor-preflight"
            ),
            "authorized_seed": target_preflight_spec.environment_seed,
            "authorized_config_sha256": authorized_config_sha256,
            "source_group_sha256": canonical_sha256(rows),
            "bootstrap_policy_sha256": canonical_sha256(policy),
        },
        "projection": projection,
        "scientific_evidence": False,
        "evidence_use": "closed-loop preflight bootstrap reservation only",
    }
    payload["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(payload),
    }
    return payload


def validate_capability_bootstrap_projection(
    value: Mapping[str, Any],
    contract: PilotContract,
    spec: PilotRunSpec,
    target_preflight_spec: PilotRunSpec,
    capability: Mapping[str, Any],
    *,
    source_capability_path: str | Path,
    source_capability_file_sha256: str,
    git_tag: str,
    git_commit: str,
    authorized_config_sha256: str,
) -> None:
    expected = build_capability_bootstrap_projection(
        contract,
        spec,
        target_preflight_spec,
        capability,
        source_capability_path=source_capability_path,
        source_capability_file_sha256=source_capability_file_sha256,
        git_tag=git_tag,
        git_commit=git_commit,
        authorized_config_sha256=authorized_config_sha256,
    )
    if dict(value) != expected:
        raise PilotPreflightAmendmentError(
            "bootstrap projection differs from its capability source"
        )


def runner_reservations_from_bootstrap_projection(
    value: Mapping[str, Any],
    *,
    contract: PilotContract,
    capability_spec: PilotRunSpec,
    target_preflight_spec: PilotRunSpec,
    capability: Mapping[str, Any],
    source_capability_path: str | Path,
    source_capability_file_sha256: str,
    git_tag: str,
    git_commit: str,
    authorized_config_sha256: str,
) -> dict[str, dict[str, Any]]:
    validate_capability_bootstrap_projection(
        value,
        contract,
        capability_spec,
        target_preflight_spec,
        capability,
        source_capability_path=source_capability_path,
        source_capability_file_sha256=source_capability_file_sha256,
        git_tag=git_tag,
        git_commit=git_commit,
        authorized_config_sha256=authorized_config_sha256,
    )
    runtime_model = value.get("runtime_model")
    projection = _mapping(value.get("projection"), "bootstrap projection")
    bindings = _mapping(value.get("bindings"), "bootstrap bindings")
    integrity = _mapping(value.get("integrity"), "bootstrap integrity")
    if not isinstance(runtime_model, str) or not runtime_model:
        raise PilotPreflightAmendmentError(
            "bootstrap runtime model is missing"
        )
    result: dict[str, Any] = {}
    for call_kind in ("action", "semantic"):
        key = f"{runtime_model}::{call_kind}"
        row = projection.get(key)
        if not isinstance(row, Mapping):
            raise PilotPreflightAmendmentError(
                f"bootstrap projection lacks {key}"
            )
        result[call_kind] = {
            "authority": {
                "authority_id": "finevo-capability-observed-bootstrap-v1",
                "pilot_contract_hash": contract.canonical_hash,
                "pilot_tag": git_tag,
                "authorized_run_id": bindings.get(
                    "authorized_runner_run_id"
                ),
                "authorized_seed": bindings.get("authorized_seed"),
                "authorized_config_sha256": bindings.get(
                    "authorized_config_sha256"
                ),
                "target_run_spec_sha256": bindings.get(
                    "target_run_spec_sha256"
                ),
                "source_run_id": capability_spec.run_id,
                "source_capability_file_sha256": bindings.get(
                    "source_capability_file_sha256"
                ),
                "source_group_sha256": bindings.get("source_group_sha256"),
                "policy_sha256": bindings.get("bootstrap_policy_sha256"),
                "source_projection_sha256": integrity.get("content_sha256"),
            },
            "reservation": _json_copy(row),
        }
    return {runtime_model: result}


__all__ = [
    "PREFLIGHT_BOOTSTRAP_AMENDMENT_ID",
    "PREFLIGHT_BOOTSTRAP_AMENDMENT_SCHEMA_VERSION",
    "PREFLIGHT_BOOTSTRAP_CONTROL_FILENAME",
    "PREFLIGHT_BOOTSTRAP_CONTROL_SCHEMA_VERSION",
    "PREFLIGHT_BOOTSTRAP_PROJECTION_FILENAME",
    "PREFLIGHT_BOOTSTRAP_PROJECTION_SCHEMA_VERSION",
    "PilotPreflightAmendmentError",
    "V23_PARENT_DEBIT_RECORD_SHA256",
    "build_capability_bootstrap_projection",
    "build_preflight_amendment_control",
    "parent_budget_debit_for_preflight_amendment",
    "preflight_amendment_control_path",
    "runner_reservations_from_bootstrap_projection",
    "validate_capability_bootstrap_projection",
    "validate_preflight_amendment_control",
]
