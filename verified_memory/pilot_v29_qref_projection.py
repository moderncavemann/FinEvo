"""Deterministic V2.9 q-ref run-summary equivalence primitives.

The verified runner deliberately records run-local identities and monotonic
clock telemetry in ``result.summary``.  Those fields are useful raw evidence,
but they cannot be compared literally across two otherwise identical q-ref
executions.  This module implements a fail-closed, allowlist-first projection:

* run and budget identities are checked against caller-bound expectations
  before being replaced by stable role sentinels;
* clock fields are checked for finite, non-negative, internally consistent
  values before being omitted;
* every other registered field, including provider and token/cost accounting,
  remains in the deterministic projection and is compared exactly.

No function in this module constructs or calls a provider.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


QREF_RUN_SUMMARY_PROJECTION_SCHEMA_VERSION = (
    "finevo-pilot-v2.9-qref-run-summary-projection-v1"
)
QREF_RUN_SUMMARY_EQUIVALENCE_SCHEMA_VERSION = (
    "finevo-pilot-v2.9-qref-run-summary-equivalence-v1"
)
QREF_RUN_SUMMARY_CANONICALIZATION = "json-sort-keys-utf8-v1"

QREF_PROVIDER_MODEL = "diagnostic/scripted-v1"
QREF_COMPLETION_COUNT = 48
QREF_AGENT_COUNT = 4
QREF_PERIOD_COUNT = 12
QREF_EXPECTED_LEAF_PATH_COUNT = 1002
QREF_EXPECTED_NORMALIZED_LEAF_PATH_COUNT = 195

_RUN_ID_SENTINEL = "<QREF_RUN_ID>"
_BUDGET_ID_SENTINEL = "<QREF_BUDGET_ID>"
_FLOAT_ABS_TOLERANCE = 1e-12

_TOP_LEVEL_KEYS = frozenset(
    {
        "action_diagnostics",
        "api",
        "diagnostic_only",
        "episode_length",
        "final_metrics",
        "memory_diagnostics",
        "num_agents",
        "provider_model",
        "result_complete",
        "result_scope",
        "run_id",
        "schema_version",
        "scientific_evidence",
        "validation",
    }
)
_API_KEYS = frozenset(
    {
        "accounted_usage",
        "active_calls",
        "active_reservations",
        "budget_id",
        "completed_calls",
        "completions",
        "effective_usage",
        "elapsed_seconds",
        "limits",
        "reserved_usage",
        "rolled_back_calls",
        "stop_reasons",
        "stopped",
    }
)
_COMPLETION_KEYS = frozenset(
    {
        "budget_id",
        "elapsed_seconds",
        "estimated_usage",
        "finished_elapsed_seconds",
        "label",
        "model",
        "reservation_id",
        "started_elapsed_seconds",
        "tags",
        "usage",
    }
)
_USAGE_KEYS = frozenset(
    {
        "completion_tokens",
        "cost_usd",
        "prompt_tokens",
        "total_tokens",
    }
)
_TAG_KEYS = frozenset(
    {
        "agent_id",
        "batch_index",
        "call_kind",
        "decision_t",
    }
)
_LIMIT_KEYS = frozenset(
    {
        "max_calls",
        "max_completion_tokens",
        "max_cost_usd",
        "max_elapsed_seconds",
        "max_prompt_tokens",
        "max_total_tokens",
    }
)

EXACT_RETAINED_PATHS = (
    "$.schema_version",
    "$.result_complete",
    "$.result_scope",
    "$.scientific_evidence",
    "$.diagnostic_only",
    "$.provider_model",
    "$.num_agents",
    "$.episode_length",
    "$.final_metrics",
    "$.action_diagnostics",
    "$.memory_diagnostics",
    "$.validation",
    "$.api.limits",
    "$.api.accounted_usage",
    "$.api.reserved_usage",
    "$.api.effective_usage",
    "$.api.completed_calls",
    "$.api.active_calls",
    "$.api.rolled_back_calls",
    "$.api.stopped",
    "$.api.stop_reasons",
    "$.api.active_reservations",
    "$.api.completions[*].reservation_id",
    "$.api.completions[*].label",
    "$.api.completions[*].model",
    "$.api.completions[*].estimated_usage",
    "$.api.completions[*].usage",
    "$.api.completions[*].tags",
)
IDENTITY_NORMALIZED_PATHS = (
    "$.run_id",
    "$.api.budget_id",
    "$.api.completions[*].budget_id",
)
VALIDATED_VOLATILE_PATHS = (
    "$.api.elapsed_seconds",
    "$.api.completions[*].started_elapsed_seconds",
    "$.api.completions[*].finished_elapsed_seconds",
    "$.api.completions[*].elapsed_seconds",
)


class PilotV29QRefProjectionError(ValueError):
    """Raised when a q-ref summary cannot enter the deterministic projection."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PilotV29QRefProjectionError(
            "q-ref summary is not canonical finite JSON"
        ) from exc


def canonical_sha256(value: Any) -> str:
    """Return the repository-standard canonical JSON SHA-256."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _leaf_paths(value: Any, *, path: str = "$") -> tuple[str, ...]:
    rows: list[str] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            rows.extend(_leaf_paths(value[key], path=f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for index, item in enumerate(value):
            rows.extend(_leaf_paths(item, path=f"{path}[{index}]"))
    else:
        rows.append(path)
    return tuple(rows)


def _normalized_leaf_paths() -> tuple[str, ...]:
    rows = [
        "$.run_id",
        "$.api.budget_id",
        "$.api.elapsed_seconds",
    ]
    for index in range(QREF_COMPLETION_COUNT):
        prefix = f"$.api.completions[{index}]"
        rows.extend(
            (
                f"{prefix}.budget_id",
                f"{prefix}.elapsed_seconds",
                f"{prefix}.finished_elapsed_seconds",
                f"{prefix}.started_elapsed_seconds",
            )
        )
    return tuple(sorted(rows))


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotV29QRefProjectionError(f"{path} must be an object")
    return value


def _sequence(value: Any, *, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PilotV29QRefProjectionError(f"{path} must be an array")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    path: str,
) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise PilotV29QRefProjectionError(
            f"{path} key inventory drifted; missing={missing}, extra={extra}"
        )


def _non_negative_float(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PilotV29QRefProjectionError(
            f"{path} must be a finite non-negative number"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise PilotV29QRefProjectionError(
            f"{path} must be a finite non-negative number"
        )
    return result


def _non_negative_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PilotV29QRefProjectionError(f"{path} must be a non-negative integer")
    return value


def _usage(value: Any, *, path: str) -> dict[str, int | float]:
    usage = _mapping(value, path=path)
    _exact_keys(usage, _USAGE_KEYS, path=path)
    prompt = _non_negative_int(
        usage["prompt_tokens"],
        path=f"{path}.prompt_tokens",
    )
    completion = _non_negative_int(
        usage["completion_tokens"],
        path=f"{path}.completion_tokens",
    )
    total = _non_negative_int(
        usage["total_tokens"],
        path=f"{path}.total_tokens",
    )
    if total != prompt + completion:
        raise PilotV29QRefProjectionError(
            f"{path}.total_tokens differs from prompt+completion"
        )
    cost = _non_negative_float(usage["cost_usd"], path=f"{path}.cost_usd")
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cost_usd": cost,
    }


def _usage_sum(
    values: Sequence[Mapping[str, int | float]],
) -> dict[str, int | float]:
    return {
        "prompt_tokens": sum(int(value["prompt_tokens"]) for value in values),
        "completion_tokens": sum(int(value["completion_tokens"]) for value in values),
        "total_tokens": sum(int(value["total_tokens"]) for value in values),
        "cost_usd": sum(float(value["cost_usd"]) for value in values),
    }


def _usage_equal(
    left: Mapping[str, int | float],
    right: Mapping[str, int | float],
) -> bool:
    return (
        left["prompt_tokens"] == right["prompt_tokens"]
        and left["completion_tokens"] == right["completion_tokens"]
        and left["total_tokens"] == right["total_tokens"]
        and math.isclose(
            float(left["cost_usd"]),
            float(right["cost_usd"]),
            rel_tol=0.0,
            abs_tol=_FLOAT_ABS_TOLERANCE,
        )
    )


def _validate_identity(
    summary: Mapping[str, Any],
    api: Mapping[str, Any],
    completions: Sequence[Mapping[str, Any]],
    *,
    expected_run_id: str,
    expected_budget_id: str,
) -> None:
    if not isinstance(expected_run_id, str) or not expected_run_id:
        raise PilotV29QRefProjectionError("expected_run_id must be non-empty")
    if not isinstance(expected_budget_id, str) or not expected_budget_id:
        raise PilotV29QRefProjectionError("expected_budget_id must be non-empty")
    if summary["run_id"] != expected_run_id:
        raise PilotV29QRefProjectionError(
            "$.run_id differs from its caller-bound expected identity"
        )
    if api["budget_id"] != expected_budget_id:
        raise PilotV29QRefProjectionError(
            "$.api.budget_id differs from its caller-bound expected identity"
        )
    if any(row["budget_id"] != expected_budget_id for row in completions):
        raise PilotV29QRefProjectionError(
            "$.api.completions[*].budget_id differs from $.api.budget_id"
        )


def _validate_timing(
    api: Mapping[str, Any],
    completions: Sequence[Mapping[str, Any]],
) -> None:
    api_elapsed = _non_negative_float(
        api["elapsed_seconds"],
        path="$.api.elapsed_seconds",
    )
    latest_finish = 0.0
    for index, row in enumerate(completions):
        prefix = f"$.api.completions[{index}]"
        started = _non_negative_float(
            row["started_elapsed_seconds"],
            path=f"{prefix}.started_elapsed_seconds",
        )
        finished = _non_negative_float(
            row["finished_elapsed_seconds"],
            path=f"{prefix}.finished_elapsed_seconds",
        )
        elapsed = _non_negative_float(
            row["elapsed_seconds"],
            path=f"{prefix}.elapsed_seconds",
        )
        if started > finished:
            raise PilotV29QRefProjectionError(f"{prefix} starts after it finishes")
        expected_elapsed = finished - started
        if not math.isclose(
            elapsed,
            expected_elapsed,
            rel_tol=0.0,
            abs_tol=_FLOAT_ABS_TOLERANCE,
        ):
            raise PilotV29QRefProjectionError(
                f"{prefix}.elapsed_seconds differs from finished-started"
            )
        latest_finish = max(latest_finish, finished)
    if api_elapsed + _FLOAT_ABS_TOLERANCE < latest_finish:
        raise PilotV29QRefProjectionError(
            "$.api.elapsed_seconds precedes a completion finish"
        )


def _validate_frozen_qref_boundary(
    summary: Mapping[str, Any],
    api: Mapping[str, Any],
    completions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        summary["schema_version"] != "verified-simulation-runner-v3"
        or summary["result_complete"] is not True
        or summary["result_scope"] != "bounded_method_smoke"
        or summary["scientific_evidence"] is not False
        or summary["diagnostic_only"] is not True
        or summary["provider_model"] != QREF_PROVIDER_MODEL
        or summary["num_agents"] != QREF_AGENT_COUNT
        or summary["episode_length"] != QREF_PERIOD_COUNT
    ):
        raise PilotV29QRefProjectionError(
            "q-ref runner/result/provider boundary drifted"
        )
    validation = _mapping(summary["validation"], path="$.validation")
    if (
        validation.get("status") != "pass"
        or validation.get("diagnostic_only") is not True
        or validation.get("scientific_evidence") is not False
    ):
        raise PilotV29QRefProjectionError(
            "$.validation differs from the diagnostic q-ref boundary"
        )
    limits = _mapping(api["limits"], path="$.api.limits")
    _exact_keys(limits, _LIMIT_KEYS, path="$.api.limits")
    if limits["max_calls"] != QREF_COMPLETION_COUNT:
        raise PilotV29QRefProjectionError("$.api.limits.max_calls differs from 48")
    if (
        api["completed_calls"] != QREF_COMPLETION_COUNT
        or api["active_calls"] != 0
        or api["rolled_back_calls"] != 0
        or api["active_reservations"] != []
        or api["stopped"] is not True
        or api["stop_reasons"] != ["call_limit"]
        or len(completions) != QREF_COMPLETION_COUNT
    ):
        raise PilotV29QRefProjectionError(
            "q-ref call-count or terminal accounting boundary drifted"
        )

    completion_usage: list[dict[str, int | float]] = []
    model_counts: Counter[str] = Counter()
    call_kind_counts: Counter[str] = Counter()
    for index, row in enumerate(completions):
        prefix = f"$.api.completions[{index}]"
        expected_t = index // QREF_AGENT_COUNT
        expected_agent = index % QREF_AGENT_COUNT
        tags = _mapping(row["tags"], path=f"{prefix}.tags")
        _exact_keys(tags, _TAG_KEYS, path=f"{prefix}.tags")
        expected_tags = {
            "agent_id": str(expected_agent),
            "batch_index": str(expected_agent),
            "call_kind": "action",
            "decision_t": str(expected_t),
        }
        if (
            row["reservation_id"] != index + 1
            or row["label"] != f"action:t{expected_t}:a{expected_agent}"
            or row["model"] != QREF_PROVIDER_MODEL
            or dict(tags) != expected_tags
        ):
            raise PilotV29QRefProjectionError(
                f"{prefix} differs from the frozen q-ref identity grid"
            )
        _usage(row["estimated_usage"], path=f"{prefix}.estimated_usage")
        actual = _usage(row["usage"], path=f"{prefix}.usage")
        if actual["cost_usd"] != 0.0:
            raise PilotV29QRefProjectionError(
                f"{prefix}.usage.cost_usd must be zero for scripted q-ref"
            )
        completion_usage.append(actual)
        model_counts[str(row["model"])] += 1
        call_kind_counts[str(tags["call_kind"])] += 1

    accounted = _usage(api["accounted_usage"], path="$.api.accounted_usage")
    reserved = _usage(api["reserved_usage"], path="$.api.reserved_usage")
    effective = _usage(api["effective_usage"], path="$.api.effective_usage")
    completion_sum = _usage_sum(completion_usage)
    if not _usage_equal(accounted, completion_sum):
        raise PilotV29QRefProjectionError(
            "$.api.accounted_usage differs from completion usage sum"
        )
    if any(value != 0 for value in reserved.values()):
        raise PilotV29QRefProjectionError(
            "$.api.reserved_usage must be zero after q-ref completion"
        )
    if not _usage_equal(effective, accounted):
        raise PilotV29QRefProjectionError(
            "$.api.effective_usage differs from accounted usage"
        )
    if accounted["cost_usd"] != 0.0:
        raise PilotV29QRefProjectionError("$.api.accounted_usage.cost_usd must be zero")

    provider_boundary = {
        "provider_model": summary["provider_model"],
        "completion_models": dict(sorted(model_counts.items())),
        "call_kind_counts": dict(sorted(call_kind_counts.items())),
        "scripted_diagnostic_calls": QREF_COMPLETION_COUNT,
        "hosted_provider_calls": 0,
        "hosted_cost_usd": 0.0,
    }
    accounting = {
        "completed_calls": api["completed_calls"],
        "active_calls": api["active_calls"],
        "rolled_back_calls": api["rolled_back_calls"],
        "stopped": api["stopped"],
        "stop_reasons": _json_copy(api["stop_reasons"]),
        "accounted_usage": accounted,
        "reserved_usage": reserved,
        "effective_usage": effective,
        "completion_usage_sum": completion_sum,
    }
    return provider_boundary, accounting


def _analyze_summary(
    summary: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_budget_id: str,
) -> dict[str, Any]:
    source = _mapping(summary, path="$")
    _exact_keys(source, _TOP_LEVEL_KEYS, path="$")
    api = _mapping(source["api"], path="$.api")
    _exact_keys(api, _API_KEYS, path="$.api")
    completion_values = _sequence(
        api["completions"],
        path="$.api.completions",
    )
    completions: list[Mapping[str, Any]] = []
    for index, value in enumerate(completion_values):
        row = _mapping(value, path=f"$.api.completions[{index}]")
        _exact_keys(
            row,
            _COMPLETION_KEYS,
            path=f"$.api.completions[{index}]",
        )
        completions.append(row)

    _validate_identity(
        source,
        api,
        completions,
        expected_run_id=expected_run_id,
        expected_budget_id=expected_budget_id,
    )
    _validate_timing(api, completions)
    provider_boundary, accounting = _validate_frozen_qref_boundary(
        source,
        api,
        completions,
    )
    observed_leaf_paths = _leaf_paths(source)
    normalized_leaf_paths = _normalized_leaf_paths()
    if len(set(observed_leaf_paths)) != len(observed_leaf_paths):
        raise PilotV29QRefProjectionError(
            "q-ref leaf-path inventory contains duplicate paths"
        )
    if (
        len(normalized_leaf_paths)
        != QREF_EXPECTED_NORMALIZED_LEAF_PATH_COUNT
        or not set(normalized_leaf_paths).issubset(observed_leaf_paths)
    ):
        raise PilotV29QRefProjectionError(
            "q-ref normalized leaf-path inventory differs from the frozen "
            "195-path boundary"
        )

    projected_summary = {
        key: _json_copy(source[key])
        for key in sorted(_TOP_LEVEL_KEYS - {"api", "run_id"})
    }
    projected_summary["run_id"] = _RUN_ID_SENTINEL
    projected_api = {
        key: _json_copy(api[key])
        for key in sorted(
            _API_KEYS
            - {
                "budget_id",
                "completions",
                "elapsed_seconds",
            }
        )
    }
    projected_api["budget_id"] = _BUDGET_ID_SENTINEL
    projected_completions = []
    for row in completions:
        projected = {
            key: _json_copy(row[key])
            for key in sorted(
                _COMPLETION_KEYS
                - {
                    "budget_id",
                    "elapsed_seconds",
                    "finished_elapsed_seconds",
                    "started_elapsed_seconds",
                }
            )
        }
        projected["budget_id"] = _BUDGET_ID_SENTINEL
        projected_completions.append(projected)
    projected_api["completions"] = projected_completions
    projected_summary["api"] = projected_api
    projection = {
        "schema_version": QREF_RUN_SUMMARY_PROJECTION_SCHEMA_VERSION,
        "summary": projected_summary,
    }
    return {
        "run_id": source["run_id"],
        "budget_id": api["budget_id"],
        "raw_summary_sha256": canonical_sha256(source),
        "projection": projection,
        "projection_sha256": canonical_sha256(projection),
        "provider_boundary": provider_boundary,
        "accounting": accounting,
        "leaf_path_count": len(observed_leaf_paths),
        "normalized_leaf_path_count": len(normalized_leaf_paths),
    }


def project_qref_run_summary(
    summary: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_budget_id: str,
) -> dict[str, Any]:
    """Validate and project one q-ref summary into deterministic JSON."""

    return _json_copy(
        _analyze_summary(
            summary,
            expected_run_id=expected_run_id,
            expected_budget_id=expected_budget_id,
        )["projection"]
    )


def _seal_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _json_copy(value)
    payload.pop("integrity", None)
    payload["integrity"] = {
        "canonicalization": QREF_RUN_SUMMARY_CANONICALIZATION,
        "content_sha256": canonical_sha256(payload),
    }
    return payload


def build_qref_run_summary_equivalence_receipt(
    current_summary: Mapping[str, Any],
    historical_summary: Mapping[str, Any],
    *,
    expected_current_run_id: str,
    expected_current_budget_id: str,
    expected_historical_run_id: str,
    expected_historical_budget_id: str,
) -> dict[str, Any]:
    """Compare two q-ref summaries after validated deterministic projection."""

    if expected_current_run_id == expected_historical_run_id:
        raise PilotV29QRefProjectionError(
            "current and historical run identities must be distinct"
        )
    if expected_current_budget_id == expected_historical_budget_id:
        raise PilotV29QRefProjectionError(
            "current and historical budget identities must be distinct"
        )
    current = _analyze_summary(
        current_summary,
        expected_run_id=expected_current_run_id,
        expected_budget_id=expected_current_budget_id,
    )
    historical = _analyze_summary(
        historical_summary,
        expected_run_id=expected_historical_run_id,
        expected_budget_id=expected_historical_budget_id,
    )
    if current["projection"] != historical["projection"]:
        raise PilotV29QRefProjectionError(
            "q-ref deterministic run-summary projections differ"
        )
    if current["provider_boundary"] != historical["provider_boundary"]:
        raise PilotV29QRefProjectionError("q-ref provider boundaries differ")
    if current["accounting"] != historical["accounting"]:
        raise PilotV29QRefProjectionError("q-ref API accounting differs")

    common_projection_sha256 = current["projection_sha256"]
    return _seal_receipt(
        {
            "schema_version": QREF_RUN_SUMMARY_EQUIVALENCE_SCHEMA_VERSION,
            "status": "pass",
            "policy": {
                "projection_schema_version": (
                    QREF_RUN_SUMMARY_PROJECTION_SCHEMA_VERSION
                ),
                "mode": "allowlist-first-fail-closed",
                "exact_retained_paths": list(EXACT_RETAINED_PATHS),
                "identity_normalized_paths": list(IDENTITY_NORMALIZED_PATHS),
                "validated_volatile_paths": list(VALIDATED_VOLATILE_PATHS),
                "unknown_paths": "reject",
                "completion_order": "exact",
                "raw_summary_hash_basis": ("full-unprojected-summary-canonical-json"),
            },
            "comparison": {
                "identity_relations_validated_before_projection": True,
                "timing_values_validated_before_omission": True,
                "deterministic_projection_exact": True,
                "provider_boundary_exact": True,
                "api_accounting_exact": True,
                "leaf_path_count": current["leaf_path_count"],
                "normalized_leaf_path_count": (
                    current["normalized_leaf_path_count"]
                ),
            },
            "current": {
                "run_id": current["run_id"],
                "budget_id": current["budget_id"],
                "raw_summary_sha256": current["raw_summary_sha256"],
                "projection_sha256": current["projection_sha256"],
                "provider_boundary": current["provider_boundary"],
                "accounting": current["accounting"],
            },
            "historical_reference": {
                "run_id": historical["run_id"],
                "budget_id": historical["budget_id"],
                "raw_summary_sha256": historical["raw_summary_sha256"],
                "projection_sha256": historical["projection_sha256"],
                "provider_boundary": historical["provider_boundary"],
                "accounting": historical["accounting"],
            },
            "common_projection_sha256": common_projection_sha256,
            "raw_summaries_reused_as_projection": False,
        }
    )
