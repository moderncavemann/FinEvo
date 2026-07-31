"""Fail-closed V2.11.2 bootstrap for the 2x12 long-context preflight.

V2.11.2 may reuse only the immutable V2.11.1 capability-gate wrapper as
operational bootstrap evidence.  The wrapper and its V2.11.1 parent-import
receipt are both file-hash and self-hash bound.  The historical 24 action and
6 semantic samples remain a p95 audit only; dispatch reserves the separately
frozen 200,000 prompt plus 4,096 completion-token contract envelope.

This authority is release-specific and operational only.  A V2.11.1
bootstrap projection, a same-release V2.11.2 capability wrapper, or any
cross-release contract/tag/run binding is rejected before dispatch.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .pilot_budget import preflight_p95
from .pilot_contract import (
    PILOT_CONTRACT_V2_11_1_CANONICAL_SHA256,
    PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256,
    canonical_sha256,
)
from .pilot_v2111_parent_import import (
    PilotV2111ParentImportError,
    V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION,
    V2111_PARENT_IMPORT_SCHEMA_VERSION,
    capability_wrappers_from_v2111_receipt,
)
from .runner import (
    V2112_CONTRACT_ENVELOPE_AUTHORITY_ID,
    V2112_CONTRACT_ID,
    V2112_PREFLIGHT_CAPABILITY_SAMPLE_COUNTS,
    V2112_PREFLIGHT_COMPLETION_ENVELOPE_TOKENS,
    V2112_PREFLIGHT_ENVELOPE_COST_USD,
    V2112_PREFLIGHT_PROMPT_ENVELOPE_TOKENS,
    V2112_PREFLIGHT_SEED,
    V2112_RELEASE_TAG,
    V2112_RUNTIME_MODEL_BY_MODEL_ID,
    V2112_SOURCE_CONTRACT_ID,
    V2112_SOURCE_RELEASE_TAG,
)


V2112_BOOTSTRAP_SCHEMA_VERSION = "finevo-pilot-v2.11.2-contract-envelope-bootstrap-v1"
V2112_BOOTSTRAP_POLICY_ID = "finevo-pilot-v2.11.2-long-context-contract-envelope-1"
V2112_BOOTSTRAP_PROJECTION_FILENAME = "v2112_contract_envelope_bootstrap.json"
V2112_SOURCE_GIT_COMMIT = "e9871353ad307fdd134f3c74764d201efbc81081"
V2112_SOURCE_PARENT_RECEIPT_FILE_SHA256 = (
    "e5d2f79e9f5a5c960aa213e38a38b0f9be513f954703ddbb0515024149f04aa0"
)
V2112_SOURCE_PARENT_RECEIPT_CONTENT_SHA256 = (
    "6e574f20f32d589597dc14dadfdcff6343554ac92493c3bfa742a848eb34873a"
)
V2112_SOURCE_CAPABILITY_WRAPPER_BINDINGS: Mapping[str, Mapping[str, str]] = {
    "gpt52_main": {
        "file_sha256": (
            "d486da4eefb47d57d40ccb1ea86c7354b745b4d823e795c4110f16d6b786c508"
        ),
        "content_sha256": (
            "08454e2b881f4e199aaabd9b0ea22b2695e4ecaf6befeffe81844cbcdc85afd2"
        ),
    },
    "gpt56_diagnostic": {
        "file_sha256": (
            "351126acf5291324de19433549ccb6f90679f3474b2d4aa12f3823b7412e920c"
        ),
        "content_sha256": (
            "aa2f881f9ad6b1c38f33dacdff4c63813329ae9ec738d3652f0d27a7bf60ecb9"
        ),
    },
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_INTEGRITY_FIELDS = {"canonicalization", "content_sha256"}
_USAGE_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
}
_MODEL_IDS = frozenset(V2112_RUNTIME_MODEL_BY_MODEL_ID)
_ZERO_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "cost_usd": 0.0,
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
        "endpoint_input": 5.0,
        "endpoint_output": 30.0,
        "source": "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
        "unit": "per_million_tokens",
    },
}
_REQUESTED_MODELS = {
    "gpt52_main": "gpt-5.2-2025-12-11",
    "gpt56_diagnostic": "gpt-5.6-sol",
}

V2112_BOOTSTRAP_POLICY: Mapping[str, Any] = {
    "policy_id": V2112_BOOTSTRAP_POLICY_ID,
    "allowed_execution_mode": "closed_loop_preflight",
    "target_shape": {
        "num_agents": 2,
        "episode_length": 12,
        "action_calls": 24,
        "semantic_calls": 8,
    },
    "source": {
        "contract_id": V2112_SOURCE_CONTRACT_ID,
        "receipt_schema_version": V2111_PARENT_IMPORT_SCHEMA_VERSION,
        "wrapper_schema_version": V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION,
        "same_model_required": True,
        "required_sample_counts": dict(V2112_PREFLIGHT_CAPABILITY_SAMPLE_COUNTS),
    },
    "capability_audit": {
        "p95_method": "nearest-rank-with-observed-maximum-floor",
        "reserve_multiplier": 1.25,
        "dispatch_reservation": False,
    },
    "effective_contract_envelope": {
        "prompt_tokens_per_call": V2112_PREFLIGHT_PROMPT_ENVELOPE_TOKENS,
        "completion_tokens_per_call": (V2112_PREFLIGHT_COMPLETION_ENVELOPE_TOKENS),
        "cached_input_discount_assumed": False,
        "price_basis": "frozen-provider-profile-dispatch-endpoint",
    },
    "missing_or_malformed_source_policy": "stop-before-dispatch",
    "scientific_evidence": False,
    "normal_scientific_dispatch_reservation_source": (
        "sealed-v2.11.2-long-context-preflight-observed-p95-only"
    ),
}

_PROJECTION_FIELDS = {
    "schema_version",
    "target",
    "source",
    "model",
    "policy",
    "policy_sha256",
    "capability_projection",
    "capability_projection_sha256",
    "contract_envelope",
    "scientific_evidence",
    "evidence_use",
    "integrity",
}


class PilotV2112BootstrapError(ValueError):
    """Raised when a V2.11.2 bootstrap source or binding is not exact."""


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
        raise PilotV2112BootstrapError("bootstrap value is not canonical JSON") from exc


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotV2112BootstrapError(f"{name} must be an object")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PilotV2112BootstrapError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _commit(value: Any, name: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise PilotV2112BootstrapError(f"{name} must be a lowercase 40-hex commit")
    return value


def _to_dict(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _json_copy(value)
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, Mapping):
            return _json_copy(converted)
    raise PilotV2112BootstrapError(f"{name} must be a mapping or expose to_dict()")


def _strict_json_object_from_bytes(
    payload: bytes,
    *,
    name: str,
) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PilotV2112BootstrapError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise PilotV2112BootstrapError(f"{name} contains non-finite value {value}")

    try:
        decoded = payload.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotV2112BootstrapError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PilotV2112BootstrapError(f"{name} must contain one JSON object")
    return value


def _validated_json_file(
    path: str | Path,
    *,
    expected_file_sha256: str,
    payload: Mapping[str, Any],
    name: str,
) -> tuple[Path, str]:
    supplied_hash = _sha256(expected_file_sha256, f"{name}_file_sha256")
    source = Path(path)
    try:
        if source.is_symlink() or not source.is_file():
            raise PilotV2112BootstrapError(f"{name} must be a regular non-symlink file")
        source_bytes = source.read_bytes()
    except OSError as exc:
        raise PilotV2112BootstrapError(f"{name} cannot be read") from exc
    actual_hash = hashlib.sha256(source_bytes).hexdigest()
    if actual_hash != supplied_hash:
        raise PilotV2112BootstrapError(f"{name} file hash mismatch")
    document = _strict_json_object_from_bytes(source_bytes, name=name)
    if document != _json_copy(payload):
        raise PilotV2112BootstrapError(f"{name} file/payload mismatch")
    return source.resolve(), actual_hash


def _verify_seal(
    value: Mapping[str, Any],
    *,
    schema_version: str,
    name: str,
) -> None:
    if value.get("schema_version") != schema_version:
        raise PilotV2112BootstrapError(f"{name} schema drifted")
    integrity = _mapping(value.get("integrity"), f"{name} integrity")
    if set(integrity) != _INTEGRITY_FIELDS:
        raise PilotV2112BootstrapError(f"{name} integrity fields drifted")
    unsigned = _json_copy(value)
    unsigned_integrity = _mapping(
        unsigned.get("integrity"),
        f"{name} unsigned integrity",
    )
    unsigned_integrity.pop("content_sha256", None)
    if integrity.get("canonicalization") != "json-sort-keys-utf8-v1" or integrity.get(
        "content_sha256"
    ) != canonical_sha256(unsigned):
        raise PilotV2112BootstrapError(f"{name} self-hash mismatch")


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
        raise PilotV2112BootstrapError(
            f"unsupported V2.11.2 bootstrap model {model_id!r}"
        )
    common = {
        "model_id": model_id,
        "requested_model": requested_model,
        "environment_seed": V2112_PREFLIGHT_SEED,
        "decoding_seed": None,
        "narrative_id": "none",
        "num_agents": 2,
    }
    expected_source = {
        **common,
        "contract_id": V2112_SOURCE_CONTRACT_ID,
        "stage_id": "capability-gate",
        "execution_mode": "capability_authority_import",
        "run_id": (
            f"{V2112_SOURCE_CONTRACT_ID}--capability-gate--{model_id}"
            "--capability-probe--none--provider-preflight-default"
            f"--s{V2112_PREFLIGHT_SEED}"
        ),
        "arm_id": "capability-probe",
        "budget_bucket": "parent_v211",
        "episode_length": 1,
        "shock_id": "baseline-3pct",
        "utility_profile_id": "provider-preflight-default",
    }
    if source != expected_source:
        raise PilotV2112BootstrapError(
            "source capability run spec differs from the exact V2.11.1 cell"
        )
    expected_target = {
        **common,
        "contract_id": V2112_CONTRACT_ID,
        "stage_id": "long-context-preflight",
        "execution_mode": "closed_loop_preflight",
        "run_id": (
            f"{V2112_CONTRACT_ID}--long-context-preflight--{model_id}"
            "--closed-loop-preflight--none--stage0-selected"
            f"--s{V2112_PREFLIGHT_SEED}"
        ),
        "arm_id": "closed-loop-preflight",
        "budget_bucket": "hosted_v2112",
        "episode_length": 12,
        "shock_id": "registered-rate-shock",
        "utility_profile_id": "stage0-selected",
    }
    if target != expected_target:
        raise PilotV2112BootstrapError(
            "target preflight run spec differs from the exact V2.11.2 cell"
        )
    return source, target


def _validated_profile(
    value: Any,
    *,
    model_id: str,
) -> tuple[dict[str, Any], str, float]:
    profile = _to_dict(value, "provider_profile")
    expected_runtime_model = V2112_RUNTIME_MODEL_BY_MODEL_ID.get(model_id)
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
        raise PilotV2112BootstrapError(
            "provider profile differs from the frozen V2.11.2 model/price"
        )
    dispatch_input = price.get("endpoint_input")
    dispatch_output = price.get("endpoint_output")
    if (
        isinstance(dispatch_input, bool)
        or not isinstance(dispatch_input, (int, float))
        or isinstance(dispatch_output, bool)
        or not isinstance(dispatch_output, (int, float))
        or not math.isfinite(float(dispatch_input))
        or not math.isfinite(float(dispatch_output))
        or float(dispatch_input) <= 0
        or float(dispatch_output) <= 0
    ):
        raise PilotV2112BootstrapError(
            "provider profile lacks positive endpoint input/output prices"
        )
    envelope_cost = (
        V2112_PREFLIGHT_PROMPT_ENVELOPE_TOKENS * float(dispatch_input)
        + V2112_PREFLIGHT_COMPLETION_ENVELOPE_TOKENS * float(dispatch_output)
    ) / 1_000_000.0
    expected_cost = V2112_PREFLIGHT_ENVELOPE_COST_USD[expected_runtime_model]
    if not math.isclose(
        envelope_cost,
        expected_cost,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise PilotV2112BootstrapError(
            "provider profile envelope cost differs from the frozen value"
        )
    return profile, expected_runtime_model, envelope_cost


def _validated_v2111_capability_wrapper(
    wrapper: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    model_id: str,
    source_contract_sha256: str,
    source_git_tag: str,
    source_git_commit: str,
    runtime_model: str,
) -> list[dict[str, Any]]:
    receipt_value = _json_copy(receipt)
    wrapper_value = _json_copy(wrapper)
    _verify_seal(
        receipt_value,
        schema_version=V2111_PARENT_IMPORT_SCHEMA_VERSION,
        name="V2.11.1 parent import receipt",
    )
    _verify_seal(
        wrapper_value,
        schema_version=V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION,
        name="V2.11.1 capability wrapper",
    )
    expected_release = {
        "contract_id": V2112_SOURCE_CONTRACT_ID,
        "contract_sha256": source_contract_sha256,
        "git_tag": source_git_tag,
        "resolved_git_commit": source_git_commit,
    }
    if (
        receipt_value.get("child_release") != expected_release
        or wrapper_value.get("child_release") != expected_release
        or receipt_value.get("scientific_evidence") is not False
        or wrapper_value.get("scientific_evidence") is not False
    ):
        raise PilotV2112BootstrapError(
            "V2.11.1 receipt/wrapper release lineage drifted"
        )
    try:
        wrappers = capability_wrappers_from_v2111_receipt(receipt_value)
    except PilotV2111ParentImportError as exc:
        raise PilotV2112BootstrapError(
            f"V2.11.1 parent receipt validation failed: {exc}"
        ) from exc
    if wrappers.get(model_id) != wrapper_value:
        raise PilotV2112BootstrapError(
            "V2.11.1 parent receipt does not bind the exact capability wrapper"
        )
    if (
        wrapper_value.get("provider_construction_current_attempt") is not False
        or wrapper_value.get("provider_calls_current_attempt") != 0
        or wrapper_value.get("hosted_provider_calls_current_attempt") != 0
        or wrapper_value.get("current_attempt_usage") != _ZERO_USAGE
        or wrapper_value.get("imported_effect_cells") != 0
        or wrapper_value.get("imported_p95_authorities") != []
    ):
        raise PilotV2112BootstrapError(
            "V2.11.1 capability wrapper import boundary drifted"
        )
    capability = _mapping(
        wrapper_value.get("capability"),
        "V2.11.1 wrapped capability",
    )
    requested_model = _REQUESTED_MODELS[model_id]
    historical_run_id = (
        f"finevo-pilot-v2.11--capability-gate--{model_id}"
        "--capability-probe--none--provider-preflight-default"
        f"--s{V2112_PREFLIGHT_SEED}"
    )
    usage_rows = capability.get("usage_rows")
    samples = capability.get("samples")
    if (
        capability.get("model_id") != model_id
        or capability.get("run_id") != historical_run_id
        or capability.get("runtime_model") != runtime_model
        or capability.get("requested_model") != requested_model
        or capability.get("served_model") != requested_model
        or capability.get("historical_source_calls") != 30
        or capability.get("action_sample_count") != 24
        or capability.get("semantic_sample_count") != 6
        or capability.get("capability_pass") is not True
        or capability.get("interface_pass") is not True
        or not isinstance(capability.get("interface_gate"), Mapping)
        or capability["interface_gate"].get("pass") is not True
        or capability.get("provider_failure_count") != 0
        or capability.get("parse_failure_count") != 0
        or capability.get("recovered_parse_count") != 0
        or capability.get("strict_parse_count") != 30
        or capability.get("truncation_count") != 0
        or not isinstance(samples, Mapping)
        or not isinstance(samples.get("action"), list)
        or len(samples["action"]) != 24
        or not isinstance(samples.get("semantic"), list)
        or len(samples["semantic"]) != 6
        or not isinstance(usage_rows, list)
        or len(usage_rows) != 30
    ):
        raise PilotV2112BootstrapError("V2.11.1 wrapped capability semantics drifted")
    source_artifacts = _mapping(
        wrapper_value.get("source_artifacts"),
        "V2.11.1 wrapper source_artifacts",
    )
    if (
        source_artifacts.get("run_id") != historical_run_id
        or source_artifacts.get("runtime_model") != runtime_model
        or source_artifacts.get("historical_source_calls") != 30
        or source_artifacts.get("action_sample_count") != 24
        or source_artifacts.get("semantic_sample_count") != 6
        or source_artifacts.get("capability_pass") is not True
        or source_artifacts.get("interface_pass") is not True
        or source_artifacts.get("scientific_evidence") is not False
    ):
        raise PilotV2112BootstrapError(
            "V2.11.1 wrapper historical source binding drifted"
        )

    normalized: list[dict[str, Any]] = []
    summed = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }
    for raw in usage_rows:
        row = _mapping(raw, "V2.11.1 capability usage row")
        call_kind = row.get("call_kind")
        usage = _mapping(row.get("usage"), "V2.11.1 capability usage")
        if (
            set(row) != {"response_model", "call_kind", "usage"}
            or row.get("response_model") != requested_model
            or call_kind not in {"action", "semantic"}
            or set(usage) != _USAGE_FIELDS
        ):
            raise PilotV2112BootstrapError(
                "V2.11.1 capability usage schema/model drifted"
            )
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            item = usage[name]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise PilotV2112BootstrapError(
                    "V2.11.1 capability usage token value is invalid"
                )
        cost = usage["cost_usd"]
        if (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or float(cost) < 0
            or usage["total_tokens"]
            != usage["prompt_tokens"] + usage["completion_tokens"]
        ):
            raise PilotV2112BootstrapError(
                "V2.11.1 capability usage accounting drifted"
            )
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            summed[name] += int(usage[name])
        summed["cost_usd"] += float(cost)
        normalized.append(
            {
                "response_model": runtime_model,
                "call_kind": call_kind,
                "usage": _json_copy(usage),
            }
        )
    counts = {
        kind: sum(row["call_kind"] == kind for row in normalized)
        for kind in ("action", "semantic")
    }
    actual = _mapping(
        capability.get("actual_usage"),
        "V2.11.1 capability actual_usage",
    )
    if (
        counts != dict(V2112_PREFLIGHT_CAPABILITY_SAMPLE_COUNTS)
        or set(actual) != _USAGE_FIELDS
        or any(
            actual.get(name) != summed[name]
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        or not isinstance(actual.get("cost_usd"), (int, float))
        or isinstance(actual.get("cost_usd"), bool)
        or not math.isclose(
            float(actual["cost_usd"]),
            summed["cost_usd"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise PilotV2112BootstrapError(
            "V2.11.1 capability denominator/usage total drifted"
        )
    return normalized


def build_v2112_contract_envelope_bootstrap_projection(
    capability_wrapper: Mapping[str, Any],
    *,
    source_parent_receipt: Mapping[str, Any],
    model_id: str,
    source_contract_sha256: str,
    source_capability_spec: Any,
    target_contract_sha256: str,
    target_preflight_spec: Any,
    provider_profile: Any,
    source_parent_receipt_path: str | Path,
    source_parent_receipt_file_sha256: str,
    source_capability_path: str | Path,
    source_capability_file_sha256: str,
    source_git_tag: str,
    source_git_commit: str,
    target_git_tag: str,
    target_git_commit: str,
    authorized_config_sha256: str,
) -> dict[str, Any]:
    """Build one exact V2.11.1-to-V2.11.2 operational projection."""

    if not isinstance(capability_wrapper, Mapping):
        raise PilotV2112BootstrapError("capability_wrapper must be an object")
    if not isinstance(source_parent_receipt, Mapping):
        raise PilotV2112BootstrapError("source_parent_receipt must be an object")
    source_contract_hash = _sha256(
        source_contract_sha256,
        "source_contract_sha256",
    )
    if (
        PILOT_CONTRACT_V2_11_1_CANONICAL_SHA256 is None
        or source_contract_hash != PILOT_CONTRACT_V2_11_1_CANONICAL_SHA256
    ):
        raise PilotV2112BootstrapError(
            "source contract hash differs from frozen V2.11.1"
        )
    target_contract_hash = _sha256(
        target_contract_sha256,
        "target_contract_sha256",
    )
    if (
        PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256 is not None
        and target_contract_hash != PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256
    ):
        raise PilotV2112BootstrapError(
            "target contract hash differs from frozen V2.11.2"
        )
    config_hash = _sha256(
        authorized_config_sha256,
        "authorized_config_sha256",
    )
    if (
        source_git_tag != V2112_SOURCE_RELEASE_TAG
        or source_git_commit != V2112_SOURCE_GIT_COMMIT
        or target_git_tag != V2112_RELEASE_TAG
    ):
        raise PilotV2112BootstrapError(
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
    usage_rows = _validated_v2111_capability_wrapper(
        capability_wrapper,
        source_parent_receipt,
        model_id=model_id,
        source_contract_sha256=source_contract_hash,
        source_git_tag=source_git_tag,
        source_git_commit=source_commit,
        runtime_model=runtime_model,
    )
    receipt_path, receipt_file_hash = _validated_json_file(
        source_parent_receipt_path,
        expected_file_sha256=source_parent_receipt_file_sha256,
        payload=source_parent_receipt,
        name="source V2.11.1 parent receipt",
    )
    source_path, source_file_hash = _validated_json_file(
        source_capability_path,
        expected_file_sha256=source_capability_file_sha256,
        payload=capability_wrapper,
        name="source V2.11.1 capability wrapper",
    )
    wrapper_binding = V2112_SOURCE_CAPABILITY_WRAPPER_BINDINGS.get(model_id)
    if (
        receipt_file_hash != V2112_SOURCE_PARENT_RECEIPT_FILE_SHA256
        or source_parent_receipt.get("integrity", {}).get("content_sha256")
        != V2112_SOURCE_PARENT_RECEIPT_CONTENT_SHA256
        or wrapper_binding is None
        or source_file_hash != wrapper_binding["file_sha256"]
        or capability_wrapper.get("integrity", {}).get("content_sha256")
        != wrapper_binding["content_sha256"]
    ):
        raise PilotV2112BootstrapError(
            "immutable V2.11.1 receipt/wrapper byte binding drifted"
        )
    projected = preflight_p95(usage_rows, reserve_multiplier=1.25)
    expected_projection_keys = {
        f"{runtime_model}::action",
        f"{runtime_model}::semantic",
    }
    if set(projected) != expected_projection_keys:
        raise PilotV2112BootstrapError(
            "capability p95 projection lacks exact runner call kinds"
        )
    capability_projection = {
        call_kind: _json_copy(projected[f"{runtime_model}::{call_kind}"])
        for call_kind in ("action", "semantic")
    }
    for call_kind, row in capability_projection.items():
        if row.get("sample_count") != (
            V2112_PREFLIGHT_CAPABILITY_SAMPLE_COUNTS[call_kind]
        ):
            raise PilotV2112BootstrapError(
                "capability p95 projection sample count drifted"
            )
        reserved = _mapping(
            row.get("reserved_p95"),
            f"{call_kind} capability reservation",
        )
        if (
            int(reserved.get("prompt_tokens", 0))
            >= V2112_PREFLIGHT_PROMPT_ENVELOPE_TOKENS
            or int(reserved.get("completion_tokens", 0))
            > V2112_PREFLIGHT_COMPLETION_ENVELOPE_TOKENS
            or float(reserved.get("cost_usd", 0.0)) <= 0
        ):
            raise PilotV2112BootstrapError(
                "capability audit projection is invalid or exceeds its "
                "registered contract envelope"
            )
    contract_envelope = {
        call_kind: {
            "prompt_tokens": V2112_PREFLIGHT_PROMPT_ENVELOPE_TOKENS,
            "completion_tokens": V2112_PREFLIGHT_COMPLETION_ENVELOPE_TOKENS,
            "total_tokens": (
                V2112_PREFLIGHT_PROMPT_ENVELOPE_TOKENS
                + V2112_PREFLIGHT_COMPLETION_ENVELOPE_TOKENS
            ),
            "cost_usd": envelope_cost,
        }
        for call_kind in ("action", "semantic")
    }
    policy = _json_copy(V2112_BOOTSTRAP_POLICY)
    target_run_id = target_spec["run_id"]
    payload: dict[str, Any] = {
        "schema_version": V2112_BOOTSTRAP_SCHEMA_VERSION,
        "target": {
            "contract_id": V2112_CONTRACT_ID,
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
            "contract_id": V2112_SOURCE_CONTRACT_ID,
            "contract_sha256": source_contract_hash,
            "git_tag": source_git_tag,
            "git_commit": source_commit,
            "run_spec": source_spec,
            "run_spec_sha256": canonical_sha256(source_spec),
            "parent_receipt_path": str(receipt_path),
            "parent_receipt_file_sha256": receipt_file_hash,
            "parent_receipt_payload_sha256": canonical_sha256(source_parent_receipt),
            "capability_path": str(source_path),
            "capability_file_sha256": source_file_hash,
            "capability_payload_sha256": canonical_sha256(capability_wrapper),
            "capability_wrapper_content_sha256": capability_wrapper["integrity"][
                "content_sha256"
            ],
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
        "evidence_use": ("V2.11.2 closed-loop preflight operational bootstrap only"),
    }
    payload["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(payload),
    }
    return payload


def validate_v2112_contract_envelope_bootstrap_projection(
    value: Mapping[str, Any],
    capability_wrapper: Mapping[str, Any],
    *,
    source_parent_receipt: Mapping[str, Any],
    model_id: str,
    source_contract_sha256: str,
    source_capability_spec: Any,
    target_contract_sha256: str,
    target_preflight_spec: Any,
    provider_profile: Any,
    source_parent_receipt_path: str | Path,
    source_parent_receipt_file_sha256: str,
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
        raise PilotV2112BootstrapError("bootstrap projection must be an object")
    if value.get("schema_version") != V2112_BOOTSTRAP_SCHEMA_VERSION:
        raise PilotV2112BootstrapError("bootstrap projection is not the V2.11.2 schema")
    if set(value) != _PROJECTION_FIELDS:
        raise PilotV2112BootstrapError("bootstrap projection fields drifted")
    integrity = _mapping(value.get("integrity"), "bootstrap integrity")
    if set(integrity) != _INTEGRITY_FIELDS:
        raise PilotV2112BootstrapError("bootstrap integrity fields drifted")
    unsigned = _json_copy(value)
    unsigned.pop("integrity")
    if integrity.get("canonicalization") != "json-sort-keys-utf8-v1" or integrity.get(
        "content_sha256"
    ) != canonical_sha256(unsigned):
        raise PilotV2112BootstrapError("bootstrap projection self-hash mismatch")
    expected = build_v2112_contract_envelope_bootstrap_projection(
        capability_wrapper,
        source_parent_receipt=source_parent_receipt,
        model_id=model_id,
        source_contract_sha256=source_contract_sha256,
        source_capability_spec=source_capability_spec,
        target_contract_sha256=target_contract_sha256,
        target_preflight_spec=target_preflight_spec,
        provider_profile=provider_profile,
        source_parent_receipt_path=source_parent_receipt_path,
        source_parent_receipt_file_sha256=(source_parent_receipt_file_sha256),
        source_capability_path=source_capability_path,
        source_capability_file_sha256=source_capability_file_sha256,
        source_git_tag=source_git_tag,
        source_git_commit=source_git_commit,
        target_git_tag=target_git_tag,
        target_git_commit=target_git_commit,
        authorized_config_sha256=authorized_config_sha256,
    )
    if _json_copy(value) != expected:
        raise PilotV2112BootstrapError(
            "bootstrap projection differs from its exact reconstructed source"
        )


def runner_reservations_from_v2112_bootstrap_projection(
    value: Mapping[str, Any],
    capability_wrapper: Mapping[str, Any],
    *,
    source_parent_receipt: Mapping[str, Any],
    model_id: str,
    source_contract_sha256: str,
    source_capability_spec: Any,
    target_contract_sha256: str,
    target_preflight_spec: Any,
    provider_profile: Any,
    source_parent_receipt_path: str | Path,
    source_parent_receipt_file_sha256: str,
    source_capability_path: str | Path,
    source_capability_file_sha256: str,
    source_git_tag: str,
    source_git_commit: str,
    target_git_tag: str,
    target_git_commit: str,
    authorized_config_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Return the exact runner mapping after full source reconstruction."""

    validate_v2112_contract_envelope_bootstrap_projection(
        value,
        capability_wrapper,
        source_parent_receipt=source_parent_receipt,
        model_id=model_id,
        source_contract_sha256=source_contract_sha256,
        source_capability_spec=source_capability_spec,
        target_contract_sha256=target_contract_sha256,
        target_preflight_spec=target_preflight_spec,
        provider_profile=provider_profile,
        source_parent_receipt_path=source_parent_receipt_path,
        source_parent_receipt_file_sha256=(source_parent_receipt_file_sha256),
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
        "authority_id": V2112_CONTRACT_ENVELOPE_AUTHORITY_ID,
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
    "PilotV2112BootstrapError",
    "V2112_BOOTSTRAP_POLICY",
    "V2112_BOOTSTRAP_POLICY_ID",
    "V2112_BOOTSTRAP_PROJECTION_FILENAME",
    "V2112_BOOTSTRAP_SCHEMA_VERSION",
    "V2112_SOURCE_CAPABILITY_WRAPPER_BINDINGS",
    "V2112_SOURCE_GIT_COMMIT",
    "V2112_SOURCE_PARENT_RECEIPT_CONTENT_SHA256",
    "V2112_SOURCE_PARENT_RECEIPT_FILE_SHA256",
    "build_v2112_contract_envelope_bootstrap_projection",
    "runner_reservations_from_v2112_bootstrap_projection",
    "validate_v2112_contract_envelope_bootstrap_projection",
]
