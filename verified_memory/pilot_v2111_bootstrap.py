"""Fail-closed V2.11.1 bootstrap for the 2x12 long-context preflight.

The frozen V2.11 capability cells are useful provenance for breaking the
preflight measurement cycle, but their prompt distribution is not a defensible
proxy for a twelve-month closed-loop context.  This module therefore preserves
the exact same-model capability p95-plus-25-percent calculation as an audit
layer while authorizing dispatch with a separate, strictly larger contract
envelope: 200,000 prompt tokens and 4,096 completion tokens per call, priced
from the frozen provider profile.

The resulting authority is operational only.  It cannot be parsed as a normal
observed-preflight authority and cannot authorize any post-preflight science.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .pilot_budget import preflight_p95
from .pilot_capability import CAPABILITY_SCHEMA_VERSION
from .pilot_contract import (
    PILOT_CONTRACT_V2_11_CANONICAL_SHA256,
    canonical_sha256,
)
from .pilot_v211_gate import PilotV211GateError, _capability_rows
from .runner import (
    V2111_CONTRACT_ENVELOPE_AUTHORITY_ID,
    V2111_CONTRACT_ID,
    V2111_PREFLIGHT_CAPABILITY_SAMPLE_COUNTS,
    V2111_PREFLIGHT_COMPLETION_ENVELOPE_TOKENS,
    V2111_PREFLIGHT_ENVELOPE_COST_USD,
    V2111_PREFLIGHT_PROMPT_ENVELOPE_TOKENS,
    V2111_PREFLIGHT_SEED,
    V2111_RELEASE_TAG,
    V2111_RUNTIME_MODEL_BY_MODEL_ID,
    V2111_SOURCE_CONTRACT_ID,
    V2111_SOURCE_RELEASE_TAG,
)


V2111_BOOTSTRAP_SCHEMA_VERSION = "finevo-pilot-v2.11.1-contract-envelope-bootstrap-v1"
V2111_BOOTSTRAP_POLICY_ID = "finevo-pilot-v2.11.1-long-context-contract-envelope-1"
V2111_BOOTSTRAP_PROJECTION_FILENAME = "v2111_contract_envelope_bootstrap.json"
V2111_SOURCE_GIT_COMMIT = "5d6c7920bd4a872b02931fdee8a47b9ac4e7b352"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_INTEGRITY_FIELDS = {"canonicalization", "content_sha256"}
_USAGE_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
}
_PRICE_SNAPSHOTS: Mapping[str, Mapping[str, Any]] = {
    "gpt52_main": {
        "captured_at": "2026-07-22",
        "catalog_cached_input": 0.175,
        "catalog_input": 1.75,
        "catalog_output": 14.0,
        "currency": "USD",
        "dispatch_basis": "endpoint",
        "endpoint_cached_input": 0.175,
        "endpoint_input": 1.75,
        "endpoint_output": 14.0,
        "source": "https://developers.openai.com/api/docs/models/gpt-5.2",
        "unit": "per_million_tokens",
    },
    "gpt56_diagnostic": {
        "captured_at": "2026-07-31",
        "catalog_cached_input": 0.5,
        "catalog_input": 5.0,
        "catalog_output": 30.0,
        "currency": "USD",
        "dispatch_basis": "endpoint",
        "endpoint_cached_input": 0.5,
        "endpoint_output": 30.0,
        "endpoint_input": 5.0,
        "source": "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
        "unit": "per_million_tokens",
    },
}
_REQUESTED_MODELS = {
    "gpt52_main": "gpt-5.2-2025-12-11",
    "gpt56_diagnostic": "gpt-5.6-sol",
}

V2111_BOOTSTRAP_POLICY: Mapping[str, Any] = {
    "policy_id": V2111_BOOTSTRAP_POLICY_ID,
    "allowed_execution_mode": "closed_loop_preflight",
    "target_shape": {
        "num_agents": 2,
        "episode_length": 12,
        "action_calls": 24,
        "semantic_calls": 8,
    },
    "source": {
        "contract_id": V2111_SOURCE_CONTRACT_ID,
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "same_model_required": True,
        "required_sample_counts": dict(V2111_PREFLIGHT_CAPABILITY_SAMPLE_COUNTS),
    },
    "capability_audit": {
        "p95_method": "nearest-rank-with-observed-maximum-floor",
        "reserve_multiplier": 1.25,
        "dispatch_reservation": False,
    },
    "effective_contract_envelope": {
        "prompt_tokens_per_call": (V2111_PREFLIGHT_PROMPT_ENVELOPE_TOKENS),
        "completion_tokens_per_call": (V2111_PREFLIGHT_COMPLETION_ENVELOPE_TOKENS),
        "cached_input_discount_assumed": False,
        "price_basis": "frozen-provider-profile-dispatch-endpoint",
    },
    "missing_or_malformed_source_policy": "stop-before-dispatch",
    "scientific_evidence": False,
    "normal_scientific_dispatch_reservation_source": (
        "sealed-long-context-preflight-observed-p95-only"
    ),
}


class PilotV2111BootstrapError(ValueError):
    """Raised when a V2.11.1 bootstrap source or binding is not exact."""


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise PilotV2111BootstrapError("bootstrap value is not canonical JSON") from exc


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotV2111BootstrapError(f"{name} must be an object")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PilotV2111BootstrapError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _commit(value: Any, name: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise PilotV2111BootstrapError(f"{name} must be a lowercase 40-hex commit")
    return value


def _to_dict(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _json_copy(value)
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, Mapping):
            return _json_copy(converted)
    raise PilotV2111BootstrapError(f"{name} must be a mapping or expose to_dict()")


def _strict_json_object_from_bytes(
    payload: bytes,
    *,
    name: str,
) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PilotV2111BootstrapError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise PilotV2111BootstrapError(f"{name} contains non-finite value {value}")

    try:
        decoded = payload.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotV2111BootstrapError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PilotV2111BootstrapError(f"{name} must contain one JSON object")
    return value


def _validated_source_file(
    path: str | Path,
    *,
    expected_file_sha256: str,
    capability: Mapping[str, Any],
) -> tuple[Path, str]:
    supplied_hash = _sha256(
        expected_file_sha256,
        "source_capability_file_sha256",
    )
    source = Path(path)
    try:
        if source.is_symlink() or not source.is_file():
            raise PilotV2111BootstrapError(
                "source capability must be a regular non-symlink file"
            )
        source_bytes = source.read_bytes()
    except OSError as exc:
        raise PilotV2111BootstrapError("source capability cannot be read") from exc
    actual_hash = hashlib.sha256(source_bytes).hexdigest()
    if actual_hash != supplied_hash:
        raise PilotV2111BootstrapError("source capability file hash mismatch")
    source_document = _strict_json_object_from_bytes(
        source_bytes,
        name="source capability",
    )
    if source_document != _json_copy(capability):
        raise PilotV2111BootstrapError("source capability file/payload mismatch")
    return source.resolve(), actual_hash


def _validate_specs(
    *,
    model_id: str,
    source_capability_spec: Any,
    target_preflight_spec: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _to_dict(source_capability_spec, "source_capability_spec")
    target = _to_dict(target_preflight_spec, "target_preflight_spec")
    requested_model = _REQUESTED_MODELS.get(model_id)
    if requested_model is None:
        raise PilotV2111BootstrapError(
            f"unsupported V2.11.1 bootstrap model {model_id!r}"
        )
    common = {
        "model_id": model_id,
        "requested_model": requested_model,
        "environment_seed": V2111_PREFLIGHT_SEED,
        "decoding_seed": None,
        "narrative_id": "none",
        "num_agents": 2,
    }
    expected_source = {
        **common,
        "contract_id": V2111_SOURCE_CONTRACT_ID,
        "stage_id": "capability-gate",
        "execution_mode": "capability_probe",
        "run_id": (
            f"{V2111_SOURCE_CONTRACT_ID}--capability-gate--{model_id}"
            "--capability-probe--none--provider-preflight-default"
            f"--s{V2111_PREFLIGHT_SEED}"
        ),
        "arm_id": "capability-probe",
        "budget_bucket": "hosted_v211",
        "episode_length": 1,
        "shock_id": "baseline-3pct",
        "utility_profile_id": "provider-preflight-default",
    }
    if source != expected_source:
        raise PilotV2111BootstrapError(
            "source capability run spec differs from the exact V2.11 cell"
        )
    expected_target = {
        **common,
        "contract_id": V2111_CONTRACT_ID,
        "stage_id": "long-context-preflight",
        "execution_mode": "closed_loop_preflight",
        "run_id": (
            f"{V2111_CONTRACT_ID}--long-context-preflight--{model_id}"
            "--closed-loop-preflight--none--stage0-selected"
            f"--s{V2111_PREFLIGHT_SEED}"
        ),
        "arm_id": "closed-loop-preflight",
        "budget_bucket": "hosted_v2111",
        "episode_length": 12,
        "shock_id": "registered-rate-shock",
        "utility_profile_id": "stage0-selected",
    }
    if target != expected_target:
        raise PilotV2111BootstrapError(
            "target preflight run spec differs from the exact V2.11.1 cell"
        )
    return source, target


def _validated_profile(
    value: Any,
    *,
    model_id: str,
) -> tuple[dict[str, Any], str, float]:
    profile = _to_dict(value, "provider_profile")
    expected_runtime_model = V2111_RUNTIME_MODEL_BY_MODEL_ID.get(model_id)
    expected_price = _PRICE_SNAPSHOTS.get(model_id)
    requested_model = _REQUESTED_MODELS.get(model_id)
    price = _mapping(profile.get("price_snapshot"), "provider price snapshot")
    if (
        expected_runtime_model is None
        or expected_price is None
        or requested_model is None
        or profile.get("profile_id") != model_id
        or profile.get("transport") != "openai"
        or profile.get("requested_model") != requested_model
        or profile.get("served_model") != requested_model
        or _json_copy(price) != _json_copy(expected_price)
    ):
        raise PilotV2111BootstrapError(
            "provider profile differs from the frozen V2.11.1 model/price"
        )
    dispatch_input = price.get("endpoint_input")
    dispatch_output = price.get("endpoint_output")
    if (
        isinstance(dispatch_input, bool)
        or not isinstance(dispatch_input, (int, float))
        or isinstance(dispatch_output, bool)
        or not isinstance(dispatch_output, (int, float))
        or float(dispatch_input) <= 0
        or float(dispatch_output) <= 0
    ):
        raise PilotV2111BootstrapError(
            "provider profile lacks positive endpoint input/output prices"
        )
    envelope_cost = (
        V2111_PREFLIGHT_PROMPT_ENVELOPE_TOKENS * float(dispatch_input)
        + V2111_PREFLIGHT_COMPLETION_ENVELOPE_TOKENS * float(dispatch_output)
    ) / 1_000_000.0
    expected_cost = V2111_PREFLIGHT_ENVELOPE_COST_USD[expected_runtime_model]
    if not math.isclose(
        envelope_cost,
        expected_cost,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise PilotV2111BootstrapError(
            "provider profile envelope cost differs from the frozen value"
        )
    return profile, expected_runtime_model, envelope_cost


def _normalized_capability_usage(
    capability: Mapping[str, Any],
    *,
    model_id: str,
    source_contract_sha256: str,
    source_spec: Mapping[str, Any],
    runtime_model: str,
) -> list[dict[str, Any]]:
    if (
        capability.get("schema_version") != CAPABILITY_SCHEMA_VERSION
        or capability.get("contract_sha256") != source_contract_sha256
        or capability.get("run_spec") != source_spec
        or capability.get("pass") is not True
        or not isinstance(capability.get("interface_gate"), Mapping)
        or capability["interface_gate"].get("pass") is not True
    ):
        raise PilotV2111BootstrapError(
            "bootstrap source is not a passing contract-bound V2.11 " "capability cell"
        )
    envelope = {
        "run_id": source_spec["run_id"],
        "artifact_sha256": canonical_sha256(capability),
        "payload": capability,
    }
    try:
        validated = _capability_rows(
            model_id=model_id,
            envelope=envelope,
            contract_sha256=source_contract_sha256,
        )
    except PilotV211GateError as exc:
        raise PilotV2111BootstrapError(
            f"source capability V5 audit failed: {exc}"
        ) from exc
    if (
        validated.get("run_id") != source_spec["run_id"]
        or validated.get("capability_pass") is not True
    ):
        raise PilotV2111BootstrapError(
            "source capability run/threshold binding drifted"
        )
    rows: list[dict[str, Any]] = []
    for raw in validated["usage_rows"]:
        row = _mapping(raw, "validated capability usage row")
        call_kind = row.get("call_kind")
        usage = _mapping(row.get("usage"), "capability usage")
        if call_kind not in {"action", "semantic"} or set(usage) != _USAGE_FIELDS:
            raise PilotV2111BootstrapError("validated capability usage schema drifted")
        rows.append(
            {
                "response_model": runtime_model,
                "call_kind": call_kind,
                "usage": _json_copy(usage),
            }
        )
    counts = {
        kind: sum(row["call_kind"] == kind for row in rows)
        for kind in ("action", "semantic")
    }
    if counts != dict(V2111_PREFLIGHT_CAPABILITY_SAMPLE_COUNTS):
        raise PilotV2111BootstrapError(
            "capability bootstrap denominator is not 24 action + 6 semantic"
        )
    return rows


def build_v2111_contract_envelope_bootstrap_projection(
    capability: Mapping[str, Any],
    *,
    model_id: str,
    source_contract_sha256: str,
    source_capability_spec: Any,
    target_contract_sha256: str,
    target_preflight_spec: Any,
    provider_profile: Any,
    source_capability_path: str | Path,
    source_capability_file_sha256: str,
    source_git_tag: str,
    source_git_commit: str,
    target_git_tag: str,
    target_git_commit: str,
    authorized_config_sha256: str,
) -> dict[str, Any]:
    """Build one exact same-model operational authority projection."""

    if not isinstance(capability, Mapping):
        raise PilotV2111BootstrapError("capability must be an object")
    source_contract_hash = _sha256(
        source_contract_sha256,
        "source_contract_sha256",
    )
    if (
        PILOT_CONTRACT_V2_11_CANONICAL_SHA256 is None
        or source_contract_hash
        != PILOT_CONTRACT_V2_11_CANONICAL_SHA256
    ):
        raise PilotV2111BootstrapError(
            "source contract hash differs from frozen V2.11"
        )
    target_contract_hash = _sha256(
        target_contract_sha256,
        "target_contract_sha256",
    )
    config_hash = _sha256(
        authorized_config_sha256,
        "authorized_config_sha256",
    )
    if (
        source_git_tag != V2111_SOURCE_RELEASE_TAG
        or source_git_commit != V2111_SOURCE_GIT_COMMIT
        or target_git_tag != V2111_RELEASE_TAG
    ):
        raise PilotV2111BootstrapError(
            "bootstrap source/target release lineage drifted"
        )
    source_commit = _commit(source_git_commit, "source_git_commit")
    target_commit = _commit(target_git_commit, "target_git_commit")
    source_spec, target_spec = _validate_specs(
        model_id=model_id,
        source_capability_spec=source_capability_spec,
        target_preflight_spec=target_preflight_spec,
    )
    profile, runtime_model, envelope_cost = _validated_profile(
        provider_profile,
        model_id=model_id,
    )
    source_path, source_file_hash = _validated_source_file(
        source_capability_path,
        expected_file_sha256=source_capability_file_sha256,
        capability=capability,
    )
    usage_rows = _normalized_capability_usage(
        capability,
        model_id=model_id,
        source_contract_sha256=source_contract_hash,
        source_spec=source_spec,
        runtime_model=runtime_model,
    )
    projected = preflight_p95(usage_rows, reserve_multiplier=1.25)
    expected_projection_keys = {
        f"{runtime_model}::action",
        f"{runtime_model}::semantic",
    }
    if set(projected) != expected_projection_keys:
        raise PilotV2111BootstrapError(
            "capability p95 projection lacks exact runner call kinds"
        )
    capability_projection = {
        call_kind: _json_copy(projected[f"{runtime_model}::{call_kind}"])
        for call_kind in ("action", "semantic")
    }
    for call_kind, row in capability_projection.items():
        if row.get("sample_count") != (
            V2111_PREFLIGHT_CAPABILITY_SAMPLE_COUNTS[call_kind]
        ):
            raise PilotV2111BootstrapError(
                "capability p95 projection sample count drifted"
            )
        reserved = _mapping(
            row.get("reserved_p95"),
            f"{call_kind} capability reservation",
        )
        if (
            int(reserved.get("prompt_tokens", 0))
            >= V2111_PREFLIGHT_PROMPT_ENVELOPE_TOKENS
            or int(reserved.get("completion_tokens", 0))
            > V2111_PREFLIGHT_COMPLETION_ENVELOPE_TOKENS
            or float(reserved.get("cost_usd", 0.0)) <= 0
        ):
            raise PilotV2111BootstrapError(
                "capability audit projection is invalid or exceeds its "
                "registered contract envelope"
            )
    contract_envelope = {
        call_kind: {
            "prompt_tokens": V2111_PREFLIGHT_PROMPT_ENVELOPE_TOKENS,
            "completion_tokens": (V2111_PREFLIGHT_COMPLETION_ENVELOPE_TOKENS),
            "total_tokens": (
                V2111_PREFLIGHT_PROMPT_ENVELOPE_TOKENS
                + V2111_PREFLIGHT_COMPLETION_ENVELOPE_TOKENS
            ),
            "cost_usd": envelope_cost,
        }
        for call_kind in ("action", "semantic")
    }
    policy = _json_copy(V2111_BOOTSTRAP_POLICY)
    target_run_id = target_spec["run_id"]
    payload: dict[str, Any] = {
        "schema_version": V2111_BOOTSTRAP_SCHEMA_VERSION,
        "target": {
            "contract_id": V2111_CONTRACT_ID,
            "contract_sha256": target_contract_hash,
            "git_tag": target_git_tag,
            "git_commit": target_commit,
            "run_spec": target_spec,
            "run_spec_sha256": canonical_sha256(target_spec),
            "authorized_runner_run_id": f"{target_run_id}--actor-preflight",
            "authorized_seed": target_spec["environment_seed"],
            "authorized_config_sha256": config_hash,
        },
        "source": {
            "contract_id": V2111_SOURCE_CONTRACT_ID,
            "contract_sha256": source_contract_hash,
            "git_tag": source_git_tag,
            "git_commit": source_commit,
            "run_spec": source_spec,
            "run_spec_sha256": canonical_sha256(source_spec),
            "capability_path": str(source_path),
            "capability_file_sha256": source_file_hash,
            "capability_payload_sha256": canonical_sha256(capability),
            "normalized_usage_group_sha256": canonical_sha256(usage_rows),
        },
        "model": {
            "model_id": model_id,
            "runtime_model": runtime_model,
            "provider_profile_sha256": canonical_sha256(profile),
            "price_snapshot": _json_copy(profile["price_snapshot"]),
            "price_snapshot_sha256": canonical_sha256(profile["price_snapshot"]),
        },
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "capability_projection": capability_projection,
        "capability_projection_sha256": canonical_sha256(capability_projection),
        "contract_envelope": contract_envelope,
        "scientific_evidence": False,
        "evidence_use": ("V2.11.1 closed-loop preflight operational bootstrap only"),
    }
    payload["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(payload),
    }
    return payload


def validate_v2111_contract_envelope_bootstrap_projection(
    value: Mapping[str, Any],
    capability: Mapping[str, Any],
    *,
    model_id: str,
    source_contract_sha256: str,
    source_capability_spec: Any,
    target_contract_sha256: str,
    target_preflight_spec: Any,
    provider_profile: Any,
    source_capability_path: str | Path,
    source_capability_file_sha256: str,
    source_git_tag: str,
    source_git_commit: str,
    target_git_tag: str,
    target_git_commit: str,
    authorized_config_sha256: str,
) -> None:
    """Rebuild and compare every source, policy, profile, and target binding."""

    if not isinstance(value, Mapping):
        raise PilotV2111BootstrapError("bootstrap projection must be an object")
    integrity = _mapping(value.get("integrity"), "bootstrap integrity")
    if set(integrity) != _INTEGRITY_FIELDS:
        raise PilotV2111BootstrapError("bootstrap integrity fields drifted")
    unsigned = _json_copy(value)
    unsigned.pop("integrity")
    if integrity.get("canonicalization") != "json-sort-keys-utf8-v1" or integrity.get(
        "content_sha256"
    ) != canonical_sha256(unsigned):
        raise PilotV2111BootstrapError("bootstrap projection self-hash mismatch")
    expected = build_v2111_contract_envelope_bootstrap_projection(
        capability,
        model_id=model_id,
        source_contract_sha256=source_contract_sha256,
        source_capability_spec=source_capability_spec,
        target_contract_sha256=target_contract_sha256,
        target_preflight_spec=target_preflight_spec,
        provider_profile=provider_profile,
        source_capability_path=source_capability_path,
        source_capability_file_sha256=source_capability_file_sha256,
        source_git_tag=source_git_tag,
        source_git_commit=source_git_commit,
        target_git_tag=target_git_tag,
        target_git_commit=target_git_commit,
        authorized_config_sha256=authorized_config_sha256,
    )
    if _json_copy(value) != expected:
        raise PilotV2111BootstrapError(
            "bootstrap projection differs from its exact reconstructed source"
        )


def runner_reservations_from_v2111_bootstrap_projection(
    value: Mapping[str, Any],
    capability: Mapping[str, Any],
    *,
    model_id: str,
    source_contract_sha256: str,
    source_capability_spec: Any,
    target_contract_sha256: str,
    target_preflight_spec: Any,
    provider_profile: Any,
    source_capability_path: str | Path,
    source_capability_file_sha256: str,
    source_git_tag: str,
    source_git_commit: str,
    target_git_tag: str,
    target_git_commit: str,
    authorized_config_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Return the exact runner mapping after full source reconstruction."""

    validate_v2111_contract_envelope_bootstrap_projection(
        value,
        capability,
        model_id=model_id,
        source_contract_sha256=source_contract_sha256,
        source_capability_spec=source_capability_spec,
        target_contract_sha256=target_contract_sha256,
        target_preflight_spec=target_preflight_spec,
        provider_profile=provider_profile,
        source_capability_path=source_capability_path,
        source_capability_file_sha256=source_capability_file_sha256,
        source_git_tag=source_git_tag,
        source_git_commit=source_git_commit,
        target_git_tag=target_git_tag,
        target_git_commit=target_git_commit,
        authorized_config_sha256=authorized_config_sha256,
    )
    target = _mapping(value["target"], "bootstrap target")
    source = _mapping(value["source"], "bootstrap source")
    model = _mapping(value["model"], "bootstrap model")
    integrity = _mapping(value["integrity"], "bootstrap integrity")
    capability_projection = _mapping(
        value["capability_projection"],
        "bootstrap capability_projection",
    )
    contract_envelope = _mapping(
        value["contract_envelope"],
        "bootstrap contract_envelope",
    )
    runtime_model = str(model["runtime_model"])
    authority = {
        "authority_id": V2111_CONTRACT_ENVELOPE_AUTHORITY_ID,
        "target_contract_id": target["contract_id"],
        "pilot_contract_hash": target["contract_sha256"],
        "pilot_tag": target["git_tag"],
        "pilot_commit": target["git_commit"],
        "model_id": model["model_id"],
        "authorized_run_id": target["authorized_runner_run_id"],
        "authorized_seed": target["authorized_seed"],
        "authorized_config_sha256": target["authorized_config_sha256"],
        "target_run_spec_sha256": target["run_spec_sha256"],
        "source_contract_id": source["contract_id"],
        "source_contract_hash": source["contract_sha256"],
        "source_tag": source["git_tag"],
        "source_commit": source["git_commit"],
        "source_run_id": source["run_spec"]["run_id"],
        "source_run_spec_sha256": source["run_spec_sha256"],
        "source_capability_file_sha256": source["capability_file_sha256"],
        "source_capability_payload_sha256": source["capability_payload_sha256"],
        "source_group_sha256": source["normalized_usage_group_sha256"],
        "capability_projection_sha256": value["capability_projection_sha256"],
        "policy_sha256": value["policy_sha256"],
        "provider_profile_sha256": model["provider_profile_sha256"],
        "price_snapshot_sha256": model["price_snapshot_sha256"],
        "source_projection_sha256": integrity["content_sha256"],
    }
    return {
        runtime_model: {
            call_kind: {
                "authority": _json_copy(authority),
                "capability_projection": _json_copy(capability_projection[call_kind]),
                "contract_envelope": _json_copy(contract_envelope[call_kind]),
            }
            for call_kind in ("action", "semantic")
        }
    }


__all__ = [
    "PilotV2111BootstrapError",
    "V2111_BOOTSTRAP_POLICY",
    "V2111_BOOTSTRAP_POLICY_ID",
    "V2111_BOOTSTRAP_PROJECTION_FILENAME",
    "V2111_BOOTSTRAP_SCHEMA_VERSION",
    "V2111_SOURCE_GIT_COMMIT",
    "build_v2111_contract_envelope_bootstrap_projection",
    "runner_reservations_from_v2111_bootstrap_projection",
    "validate_v2111_contract_envelope_bootstrap_projection",
]
