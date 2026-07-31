"""Fresh post-gate authority for the prospective FinEvo V2.11 pilot.

This module is deliberately provider-free.  It consumes the two terminal
capability payloads and the two terminal 2x12 preflight checkpoint/exactness
pairs, preserves their complete ITT denominators, and emits one self-hashed
zero-provider authority receipt.

Eligibility is model scoped.  Both models still pay and reconcile their exact
30 capability plus 32 preflight calls before eligibility is decided.  A
GPT-5.6 no-go therefore does not block GPT-5.2 A--D, and a GPT-5.2 no-go does
not delete the six registered GPT-5.6 cells.  Ineligible descendants remain
bound as no-go ledger cells; they are never removed from the 136-cell matrix.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence

from .pilot_budget import preflight_p95
from .pilot_capability import (
    CAPABILITY_SCHEMA_VERSION,
    CAPABILITY_TASKSET_SHA256,
    CAPABILITY_THRESHOLDS,
    build_capability_tasks,
)
from .pilot_checkpoint import (
    PILOT_CHECKPOINT_SCHEMA_VERSION_V4,
    V211_LONG_CONTEXT_PREFLIGHT_CHECKPOINT_PURPOSE,
)
from .pilot_interface_gate import (
    PilotInterfaceGateError,
    interface_sample_gate,
)
from .pilot_v211_parent_import import (
    V211_SOURCE_MANIFEST_CONTENT_SHA256,
    V211_SOURCE_MANIFEST_FILE_SHA256,
)
from .pilot_v211_projection import (
    PilotV211ProjectionError,
    V211_MODEL_GATE_REGISTERED_CALLS,
    V211_MODEL_SCIENCE_LEDGER_CELLS,
    V211_PREFLIGHT_ACTUAL_COMPLETIONS,
    V211_REMAINING_SCIENCE_COMPLETIONS,
    V211_TOTAL_LEDGER_CELLS,
    project_v211_full_matrix,
)
from .runner import (
    OBSERVED_P95_AUTHORITY_ID,
    OBSERVED_P95_PROJECTION_SCHEMA_VERSION,
    OBSERVED_P95_SOURCE_KIND,
    ObservedPreflightP95Reservation,
    PreflightP95Reservation,
)


V211_GATE_SCHEMA_VERSION = "finevo-pilot-v2.11-post-gate-authority-v1"
V211_PREFLIGHT_EXACTNESS_SCHEMA_VERSION = (
    "finevo-v2.11-long-context-preflight-exactness-receipt-v1"
)
V211_GATE_RELEASE_TAG = "pilot-v2.11-science"
V211_GATE_RESERVE_MULTIPLIER = 1.25
V211_GATE_WIRE_CAP_TOKENS = 4_096
V211_GATE_PROMPT_TIER_CEILING = 200_000
V211_GATE_ACTION_SAMPLES_PER_MODEL = 48
V211_GATE_SEMANTIC_SAMPLES_PER_MODEL = 14
V211_GATE_CAPABILITY_CALLS_PER_MODEL = 30
V211_GATE_PREFLIGHT_CALLS_PER_MODEL = 32
V211_PREFLIGHT_CHECKPOINT_RUN_SUFFIX = "--actor-preflight"
V211_GATE_SCIENCE_RUNS_PER_MODEL: Mapping[str, int] = {
    "gpt52_main": 125,
    "gpt56_diagnostic": 6,
}

_MODEL_PROFILES: Mapping[str, Mapping[str, Any]] = {
    "gpt52_main": {
        "runtime_model": "openai/gpt-5.2-2025-12-11",
        "requested_model": "gpt-5.2-2025-12-11",
        "served_model": "gpt-5.2-2025-12-11",
        "provider": "openai",
        "response_provider": "OpenAI-direct",
        "response_route": "direct",
        "provider_pin": ["OpenAI-direct"],
        "artifact_identity": {
            "route": "OpenAI-direct",
            "served_snapshot": "gpt-5.2-2025-12-11",
        },
        "price_source": (
            "https://developers.openai.com/api/docs/models/gpt-5.2"
        ),
        "price_captured_at": "2026-07-22",
    },
    "gpt56_diagnostic": {
        "runtime_model": "openai/gpt-5.6-sol",
        "requested_model": "gpt-5.6-sol",
        "served_model": "gpt-5.6-sol",
        "provider": "openai",
        "response_provider": "OpenAI-direct",
        "response_route": "direct",
        "provider_pin": ["OpenAI-direct"],
        "artifact_identity": {
            "route": "OpenAI-direct",
            "served_snapshot": "gpt-5.6-sol",
        },
        "price_source": (
            "https://developers.openai.com/api/docs/models/gpt-5.6-sol"
        ),
        "price_captured_at": "2026-07-31",
    },
}
_MODEL_IDS = frozenset(_MODEL_PROFILES)
_MODEL_TERMINAL_STATUSES = frozenset(
    {"eligible", "capability-no-go", "interface-no-go"}
)
_REQUEST_PARAMETERS = frozenset(
    {
        "max_completion_tokens",
        "messages",
        "model",
        "reasoning_effort",
        "response_format",
    }
)
_USAGE_FIELDS = frozenset(
    {"prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"}
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class PilotV211GateError(ValueError):
    """Raised when V2.11 gate evidence cannot authorize any dispatch."""


def canonical_sha256(value: Any) -> str:
    """Return the repository's canonical JSON SHA-256."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PilotV211GateError("gate value is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotV211GateError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PilotV211GateError(f"{name} must be an array")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise PilotV211GateError(f"{name} must be a lowercase SHA-256")
    return value


def _commit(value: Any, name: str) -> str:
    if not isinstance(value, str) or _COMMIT_PATTERN.fullmatch(value) is None:
        raise PilotV211GateError(f"{name} must be a lowercase 40-hex commit")
    return value


def _run_id(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise PilotV211GateError(f"{name} must be normalized non-empty text")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PilotV211GateError(f"{name} must be a nonnegative integer")
    return value


def _usage(
    value: Any,
    name: str,
    *,
    positive: bool = True,
) -> dict[str, int | float]:
    row = _mapping(value, name)
    if set(row) != _USAGE_FIELDS:
        raise PilotV211GateError(
            f"{name} must contain exactly prompt/completion/total/cost"
        )
    tokens: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = row[field]
        minimum = 1 if positive else 0
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < minimum
        ):
            raise PilotV211GateError(
                f"{name}.{field} must be an integer >= {minimum}"
            )
        tokens[field] = item
    if tokens["total_tokens"] != (
        tokens["prompt_tokens"] + tokens["completion_tokens"]
    ):
        raise PilotV211GateError(f"{name}.total_tokens is not additive")
    cost = row["cost_usd"]
    if (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < (0.0 if not positive else 0.0)
        or (positive and float(cost) <= 0.0)
    ):
        raise PilotV211GateError(
            f"{name}.cost_usd must be finite"
            + (" and positive" if positive else " and nonnegative")
        )
    return {**tokens, "cost_usd": float(cost)}


def _sum_usage(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    prompt = sum(int(row["prompt_tokens"]) for row in rows)
    completion = sum(int(row["completion_tokens"]) for row in rows)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cost_usd": math.fsum(float(row["cost_usd"]) for row in rows),
    }


def _usage_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        int(left["prompt_tokens"]) == int(right["prompt_tokens"])
        and int(left["completion_tokens"]) == int(right["completion_tokens"])
        and int(left["total_tokens"]) == int(right["total_tokens"])
        and math.isclose(
            float(left["cost_usd"]),
            float(right["cost_usd"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    )


def _artifact_envelope(
    value: Any,
    *,
    model_id: str,
    kind: str,
) -> tuple[str, str, Mapping[str, Any]]:
    envelope = _mapping(value, f"{model_id} {kind} artifact")
    if set(envelope) != {"run_id", "artifact_sha256", "payload"}:
        raise PilotV211GateError(
            f"{model_id} {kind} artifact envelope fields drifted"
        )
    run_id = _run_id(envelope["run_id"], f"{model_id} {kind} run_id")
    artifact_sha256 = _sha256(
        envelope["artifact_sha256"],
        f"{model_id} {kind} artifact_sha256",
    )
    payload = _mapping(envelope["payload"], f"{model_id} {kind} payload")
    if canonical_sha256(payload) != artifact_sha256:
        raise PilotV211GateError(
            f"{model_id} {kind} artifact canonical hash mismatch"
        )
    return run_id, artifact_sha256, payload


def _provider_row(
    row: Mapping[str, Any],
    *,
    model_id: str,
    name: str,
    requested_model_field: str,
    served_model_field: str,
    response_provider_field: str,
    response_route_field: str,
    require_request_id: bool = True,
) -> tuple[dict[str, int | float], int, int]:
    profile = _MODEL_PROFILES[model_id]
    expected = {
        requested_model_field: profile["requested_model"],
        served_model_field: profile["served_model"],
        "provider": profile["provider"],
        response_provider_field: profile["response_provider"],
        response_route_field: profile["response_route"],
        "request_profile_id": model_id,
        "request_provider_pin": profile["provider_pin"],
        "request_artifact_identity": profile["artifact_identity"],
        "request_price_snapshot_source": profile["price_source"],
        "request_price_snapshot_captured_at": profile["price_captured_at"],
        "attempts": 1,
        "temperature_dispatch": "omitted_unsupported",
    }
    for field, expected_value in expected.items():
        if row.get(field) != expected_value:
            raise PilotV211GateError(
                f"{name} frozen provider profile mismatch at {field}"
            )
    if set(_sequence(row.get("request_parameters"), f"{name}.request_parameters")) != (
        _REQUEST_PARAMETERS
    ):
        raise PilotV211GateError(
            f"{name} request parameters differ from the frozen direct route"
        )
    if (
        not isinstance(row.get("provider_sdk_name"), str)
        or not row["provider_sdk_name"]
        or not isinstance(row.get("provider_sdk_version"), str)
        or not row["provider_sdk_version"]
        or (
            require_request_id
            and (
                not isinstance(row.get("provider_request_id"), str)
                or not row["provider_request_id"]
            )
        )
    ):
        raise PilotV211GateError(
            f"{name} provider SDK/request provenance is incomplete"
        )
    usage = _usage(row.get("usage"), f"{name}.usage")
    cached = _nonnegative_integer(
        row.get("cached_prompt_tokens"),
        f"{name}.cached_prompt_tokens",
    )
    reasoning = _nonnegative_integer(
        row.get("reasoning_tokens"),
        f"{name}.reasoning_tokens",
    )
    visible = _nonnegative_integer(
        row.get("visible_completion_tokens"),
        f"{name}.visible_completion_tokens",
    )
    if cached > int(usage["prompt_tokens"]):
        raise PilotV211GateError(f"{name} cached usage exceeds prompt usage")
    if reasoning + visible != int(usage["completion_tokens"]):
        raise PilotV211GateError(
            f"{name} reasoning/visible usage is not additive"
        )
    return usage, reasoning, visible


def _capability_rows(
    *,
    model_id: str,
    envelope: Mapping[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    run_id, artifact_sha256, payload = _artifact_envelope(
        envelope,
        model_id=model_id,
        kind="capability",
    )
    if (
        payload.get("schema_version") != CAPABILITY_SCHEMA_VERSION
        or payload.get("taskset_sha256") != CAPABILITY_TASKSET_SHA256
        or payload.get("provider_model")
        != _MODEL_PROFILES[model_id]["runtime_model"]
    ):
        raise PilotV211GateError(
            f"{model_id} capability v5 identity drifted"
        )
    prompt_gate = _mapping(
        payload.get("prompt_tier_gate"),
        f"{model_id} capability prompt tier gate",
    )
    if (
        prompt_gate.get("upper_bound_method") != "utf8-bytes-plus-256-v1"
        or prompt_gate.get("ceiling_tokens")
        != V211_GATE_PROMPT_TIER_CEILING
        or prompt_gate.get("passed") is not True
        or isinstance(prompt_gate.get("maximum_upper_bound_tokens"), bool)
        or not isinstance(prompt_gate.get("maximum_upper_bound_tokens"), int)
        or not 0
        < int(prompt_gate["maximum_upper_bound_tokens"])
        < V211_GATE_PROMPT_TIER_CEILING
    ):
        raise PilotV211GateError(
            f"{model_id} capability prompt-tier binding is invalid"
        )
    rows = _sequence(payload.get("rows"), f"{model_id} capability rows")
    if len(rows) != V211_GATE_CAPABILITY_CALLS_PER_MODEL:
        raise PilotV211GateError(
            f"{model_id} capability must retain exactly 30 calls"
        )
    task_ids: set[str] = set()
    usage_rows: list[dict[str, Any]] = []
    samples: dict[str, list[dict[str, Any]]] = {
        "action": [],
        "semantic": [],
    }
    category_rows: dict[str, list[Mapping[str, Any]]] = {
        category: [] for category in CAPABILITY_THRESHOLDS
    }
    actual_rows: list[dict[str, int | float]] = []
    frozen_tasks = build_capability_tasks()
    if len(frozen_tasks) != V211_GATE_CAPABILITY_CALLS_PER_MODEL:
        raise PilotV211GateError("frozen capability task denominator drifted")
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"{model_id} capability row {index}")
        frozen_task = frozen_tasks[index]
        task_id = _run_id(
            row.get("task_id"),
            f"{model_id} capability row {index}.task_id",
        )
        if (
            task_id != frozen_task.task_id
            or row.get("task_kind") != frozen_task.task_kind
            or row.get("category") != frozen_task.category
            or row.get("output_contract_id")
            != frozen_task.output_contract_id
            or row.get("taskset_sha256") != CAPABILITY_TASKSET_SHA256
            or row.get("prompt_sha256")
            != hashlib.sha256(
                frozen_task.prompt.encode("utf-8")
            ).hexdigest()
        ):
            raise PilotV211GateError(
                f"{model_id} capability row {index} differs from the "
                "frozen task set"
            )
        if task_id in task_ids:
            raise PilotV211GateError(
                f"{model_id} capability contains a duplicate task"
            )
        task_ids.add(task_id)
        task_kind = row.get("task_kind")
        if task_kind in {"action_generation", "rule_application"}:
            call_kind = "action"
            expected_visible_cap = 512
        elif task_kind == "rule_proposal":
            call_kind = "semantic"
            expected_visible_cap = 4_096
        else:
            raise PilotV211GateError(
                f"{model_id} capability row {index} task kind is invalid"
            )
        expected_category = {
            "action_generation": "utility-ranking",
            "rule_application": "rule-application",
            "rule_proposal": "rule-proposal",
        }[str(task_kind)]
        if row.get("category") != expected_category:
            raise PilotV211GateError(
                f"{model_id} capability row {index} category drifted"
            )
        category_rows[expected_category].append(row)
        name = f"{model_id} capability row {index}"
        usage, reasoning, visible = _provider_row(
            row,
            model_id=model_id,
            name=name,
            requested_model_field="requested_model",
            served_model_field="served_model",
            response_provider_field="response_provider",
            response_route_field="response_route",
            require_request_id=False,
        )
        if (
            row.get("request_max_completion_tokens")
            != V211_GATE_WIRE_CAP_TOKENS
            or row.get("visible_json_max_bytes") != expected_visible_cap
            or isinstance(row.get("output_bytes"), bool)
            or not isinstance(row.get("output_bytes"), int)
            or row["output_bytes"] < 0
            or row["output_bytes"] > expected_visible_cap
        ):
            raise PilotV211GateError(
                f"{name} output contract differs from V2.11"
            )
        if row.get("provider_reported_usage_available") is not True:
            raise PilotV211GateError(
                f"{name} lacks provider-reported usage"
            )
        provider_reported = _usage(
            row.get("provider_reported_usage"),
            f"{name}.provider_reported_usage",
        )
        if (
            not _usage_equal(provider_reported, usage)
            or not _usage_equal(
                _usage(row.get("usage"), f"{name}.usage"),
                usage,
            )
        ):
            raise PilotV211GateError(
                f"{name} provider-reported usage aliases drifted"
            )
        accounted = _usage(
            row.get("budget_accounted_usage"),
            f"{name}.budget_accounted_usage",
        )
        actual_rows.append(accounted)
        if call_kind == "action":
            action = _mapping(row.get("action"), f"{name}.action")
            clipped = action.get("clipped")
            if not isinstance(clipped, bool):
                raise PilotV211GateError(
                    f"{name} action clipping disposition is absent"
                )
        else:
            clipped = False
        boolean_fields = (
            "correct",
            "legal",
            "interface_valid",
            "evaluable",
            "truncation",
            "finish_contract_valid",
            "within_visible_limit",
        )
        if any(not isinstance(row.get(field), bool) for field in boolean_fields):
            raise PilotV211GateError(
                f"{name} capability score/interface flags are incomplete"
            )
        strict_parse = (
            row.get("parse_mode") == "exact_json"
            and row.get("strict_parse") is True
            and row.get("accepted_parse_mode") is True
        )
        expected_interface_valid = (
            row.get("provider_error") is None
            and row.get("truncation") is False
            and row.get("finish_contract_valid") is True
            and row.get("within_visible_limit") is True
        )
        if (
            row.get("interface_valid") is not expected_interface_valid
            or row.get("evaluable") is not expected_interface_valid
            or row.get("correct")
            is not (
                expected_interface_valid
                and strict_parse
                and row.get("legal") is True
            )
        ):
            raise PilotV211GateError(
                f"{name} capability score/interface accounting drifted"
            )
        samples[call_kind].append(
            {
                "finish_reason": row.get("finish_reason"),
                "response_completed": row.get("response_completed"),
                "output_disposition": row.get("output_disposition"),
                "error_type": row.get("provider_error"),
                "parse_success": strict_parse,
                "clipped": clipped,
                "prompt_tokens": int(usage["prompt_tokens"]),
                "completion_tokens": int(usage["completion_tokens"]),
                "reasoning_tokens": reasoning,
                "visible_completion_tokens": visible,
            }
        )
        usage_rows.append(
            {
                "response_model": _MODEL_PROFILES[model_id]["served_model"],
                "call_kind": call_kind,
                "usage": usage,
            }
        )
    if (
        len(samples["action"]) != 24
        or len(samples["semantic"]) != 6
    ):
        raise PilotV211GateError(
            f"{model_id} capability denominator is not 24 action + 6 semantic"
        )
    expected_category_counts = {
        "utility-ranking": 12,
        "rule-application": 12,
        "rule-proposal": 6,
    }
    if {
        category: len(values) for category, values in category_rows.items()
    } != expected_category_counts:
        raise PilotV211GateError(
            f"{model_id} capability category denominator drifted"
        )
    category_totals: dict[str, dict[str, Any]] = {}
    for category, required in CAPABILITY_THRESHOLDS.items():
        values = category_rows[category]
        registered_correct = sum(
            row.get("correct") is True for row in values
        )
        evaluable_count = sum(
            row.get("evaluable") is True for row in values
        )
        conditional_correct = sum(
            row.get("correct") is True
            for row in values
            if row.get("evaluable") is True
        )
        category_totals[category] = {
            "correct": registered_correct,
            "denominator": len(values),
            "required": required,
            "registered_correct": registered_correct,
            "registered_total": len(values),
            "evaluable_count": evaluable_count,
            "conditional_correct": conditional_correct,
            "conditional_accuracy": (
                conditional_correct / evaluable_count
                if evaluable_count
                else None
            ),
            "interface_failure_count": len(values) - evaluable_count,
        }
    checks = {
        category: row["registered_correct"] >= row["required"]
        for category, row in category_totals.items()
    }
    conditional_checks = {
        category: (
            None
            if row["conditional_accuracy"] is None
            else row["conditional_accuracy"]
            >= row["required"] / row["registered_total"]
        )
        for category, row in category_totals.items()
    }
    interface_failure_count = sum(
        not row.get("evaluable") for values in category_rows.values()
        for row in values
    )
    threshold_pass = all(checks.values())
    capability_status = (
        "not_evaluable"
        if interface_failure_count
        else "pass" if threshold_pass else "fail"
    )
    assessment_pass: bool | None = (
        None if interface_failure_count else threshold_pass
    )
    if (
        payload.get("category_totals") != category_totals
        or payload.get("checks") != checks
        or payload.get("interface_gate")
        != {
            "pass": interface_failure_count == 0,
            "failure_count": interface_failure_count,
        }
        or payload.get("capability_assessment")
        != {
            "status": capability_status,
            "pass": assessment_pass,
            "checks": conditional_checks,
        }
        or payload.get("pass")
        is not (interface_failure_count == 0 and threshold_pass)
    ):
        raise PilotV211GateError(
            f"{model_id} capability threshold/interface receipt drifted"
        )
    budget = _mapping(
        payload.get("budget"),
        f"{model_id} capability budget",
    )
    if (
        budget.get("completed_calls")
        != V211_GATE_CAPABILITY_CALLS_PER_MODEL
        or budget.get("active_calls") != 0
    ):
        raise PilotV211GateError(
            f"{model_id} capability budget is not terminal"
        )
    actual = _sum_usage(actual_rows)
    accounted = _usage(
        budget.get("accounted_usage"),
        f"{model_id} capability budget.accounted_usage",
    )
    if not _usage_equal(actual, accounted):
        raise PilotV211GateError(
            f"{model_id} capability budget double-count/mismatch"
        )
    completions = _sequence(
        budget.get("completions"),
        f"{model_id} capability budget completions",
    )
    if len(completions) != V211_GATE_CAPABILITY_CALLS_PER_MODEL:
        raise PilotV211GateError(
            f"{model_id} capability budget completion denominator drifted"
        )
    completion_usage: dict[str, dict[str, int | float]] = {}
    for index, raw_completion in enumerate(completions):
        completion = _mapping(
            raw_completion,
            f"{model_id} capability budget completion {index}",
        )
        label = completion.get("label")
        if not isinstance(label, str) or not label.startswith("capability:"):
            raise PilotV211GateError(
                f"{model_id} capability completion label is invalid"
            )
        task_id = label.removeprefix("capability:")
        if task_id in completion_usage:
            raise PilotV211GateError(
                f"{model_id} capability completion is duplicated"
            )
        completion_usage[task_id] = _usage(
            completion.get("usage"),
            f"{model_id} capability completion {index}.usage",
        )
    if set(completion_usage) != task_ids:
        raise PilotV211GateError(
            f"{model_id} capability budget omits registered tasks"
        )
    for row in rows:
        if not _usage_equal(
            completion_usage[str(row["task_id"])],
            _usage(
                row["budget_accounted_usage"],
                f"{model_id} capability row budget usage",
            ),
        ):
            raise PilotV211GateError(
                f"{model_id} capability row/budget completion usage differs"
            )
    return {
        "run_id": run_id,
        "artifact_sha256": artifact_sha256,
        "schema_version": payload["schema_version"],
        # The registered category thresholds are independent of the strict
        # post-gate interface sample.  For example, one failed proposal parse
        # can still satisfy 5/6 while correctly producing an interface no-go.
        "capability_pass": threshold_pass,
        "samples": samples,
        "usage_rows": usage_rows,
        "actual_usage": actual,
        "contract_sha256": contract_sha256,
    }


def _preflight_envelope(
    value: Any,
    *,
    model_id: str,
    contract_sha256: str,
) -> dict[str, Any]:
    envelope = _mapping(value, f"{model_id} preflight artifact")
    expected_fields = {
        "ledger_run_id",
        "checkpoint_run_id",
        "run_spec_sha256",
        "checkpoint_artifact_sha256",
        "checkpoint",
        "exactness_artifact_sha256",
        "exactness",
    }
    if set(envelope) != expected_fields:
        raise PilotV211GateError(
            f"{model_id} preflight artifact envelope fields drifted"
        )
    ledger_run_id = _run_id(
        envelope["ledger_run_id"],
        f"{model_id} preflight ledger_run_id",
    )
    checkpoint_run_id = _run_id(
        envelope["checkpoint_run_id"],
        f"{model_id} preflight checkpoint_run_id",
    )
    if checkpoint_run_id != (
        ledger_run_id + V211_PREFLIGHT_CHECKPOINT_RUN_SUFFIX
    ):
        raise PilotV211GateError(
            f"{model_id} checkpoint run ID is not bound to its ledger cell"
        )
    run_spec_sha256 = _sha256(
        envelope["run_spec_sha256"],
        f"{model_id} preflight run_spec_sha256",
    )
    checkpoint = _mapping(
        envelope["checkpoint"],
        f"{model_id} preflight checkpoint",
    )
    checkpoint_artifact_sha256 = _sha256(
        envelope["checkpoint_artifact_sha256"],
        f"{model_id} checkpoint artifact SHA-256",
    )
    if canonical_sha256(checkpoint) != checkpoint_artifact_sha256:
        raise PilotV211GateError(
            f"{model_id} checkpoint artifact hash mismatch"
        )
    checkpoint_body = dict(checkpoint)
    checkpoint_hash = checkpoint_body.pop("checkpoint_hash", None)
    if (
        _SHA256_PATTERN.fullmatch(str(checkpoint_hash)) is None
        or canonical_sha256(checkpoint_body) != checkpoint_hash
    ):
        raise PilotV211GateError(
            f"{model_id} checkpoint self-hash mismatch"
        )
    exactness = _mapping(
        envelope["exactness"],
        f"{model_id} preflight exactness receipt",
    )
    exactness_artifact_sha256 = _sha256(
        envelope["exactness_artifact_sha256"],
        f"{model_id} exactness artifact SHA-256",
    )
    if canonical_sha256(exactness) != exactness_artifact_sha256:
        raise PilotV211GateError(
            f"{model_id} exactness artifact hash mismatch"
        )
    exactness_body = dict(exactness)
    exactness_hash = exactness_body.pop("receipt_hash", None)
    if (
        _SHA256_PATTERN.fullmatch(str(exactness_hash)) is None
        or canonical_sha256(exactness_body) != exactness_hash
        or exactness.get("schema_version")
        != V211_PREFLIGHT_EXACTNESS_SCHEMA_VERSION
    ):
        raise PilotV211GateError(
            f"{model_id} exactness receipt self-hash mismatch"
        )
    run_config = _mapping(
        checkpoint.get("run_config"),
        f"{model_id} preflight run config",
    )
    if (
        run_config.get("run_id") != checkpoint_run_id
        or run_config.get("pilot_contract_hash") != contract_sha256
        or run_config.get("num_agents") != 2
        or run_config.get("episode_length") != 12
        or run_config.get("action_max_tokens")
        != V211_GATE_WIRE_CAP_TOKENS
        or run_config.get("rule_max_tokens")
        != V211_GATE_WIRE_CAP_TOKENS
        or run_config.get("action_max_visible_json_bytes") != 1_024
        or run_config.get("rule_max_visible_json_bytes") != 4_096
        or run_config.get("prompt_tier_ceiling_tokens")
        != V211_GATE_PROMPT_TIER_CEILING
        or run_config.get("accepted_action_parse_modes")
        != ["exact_json"]
        or run_config.get("accepted_semantic_parse_modes")
        != ["exact_json"]
        or run_config.get("fail_on_clipped_action") is not True
    ):
        raise PilotV211GateError(
            f"{model_id} preflight is not the frozen 2x12 V2.11 config"
        )
    if (
        checkpoint.get("schema_version")
        != PILOT_CHECKPOINT_SCHEMA_VERSION_V4
        or checkpoint.get("checkpoint_purpose")
        != V211_LONG_CONTEXT_PREFLIGHT_CHECKPOINT_PURPOSE
    ):
        raise PilotV211GateError(
            f"{model_id} checkpoint purpose is not long-context preflight"
        )
    rows = _sequence(
        checkpoint.get("provider_calls"),
        f"{model_id} preflight provider calls",
    )
    if (
        len(rows) != V211_GATE_PREFLIGHT_CALLS_PER_MODEL
        or checkpoint.get("provider_calls_hash") != canonical_sha256(rows)
        or [row.get("call_index") for row in rows] != list(range(32))
    ):
        raise PilotV211GateError(
            f"{model_id} preflight must retain exactly 32 hashed calls"
        )
    samples: dict[str, list[dict[str, Any]]] = {
        "action": [],
        "semantic": [],
    }
    usage_rows: list[dict[str, Any]] = []
    provider_actual_rows: list[dict[str, int | float]] = []
    semantic_rows_by_index: dict[int, Mapping[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"{model_id} preflight row {index}")
        call_kind = row.get("call_kind")
        if call_kind not in {"action", "semantic"}:
            raise PilotV211GateError(
                f"{model_id} preflight row {index} call kind is invalid"
            )
        name = f"{model_id} preflight row {index}"
        usage, reasoning, visible = _provider_row(
            row,
            model_id=model_id,
            name=name,
            requested_model_field="model",
            served_model_field="served_model",
            response_provider_field="served_provider",
            response_route_field="served_route",
        )
        task_cap = _mapping(row.get("task_cap"), f"{name}.task_cap")
        expected_task_cap = {
            "max_visible_tokens": V211_GATE_WIRE_CAP_TOKENS,
            "max_visible_json_bytes": (
                1_024 if call_kind == "action" else 4_096
            ),
        }
        if task_cap != expected_task_cap:
            raise PilotV211GateError(
                f"{name} task cap differs from V2.11"
            )
        if (
            row.get("prompt_token_upper_bound_method")
            != "utf8-bytes-plus-256-v1"
            or row.get("prompt_tier_ceiling_tokens")
            != V211_GATE_PROMPT_TIER_CEILING
            or isinstance(row.get("prompt_token_upper_bound"), bool)
            or not isinstance(row.get("prompt_token_upper_bound"), int)
            or not 0
            < row["prompt_token_upper_bound"]
            < V211_GATE_PROMPT_TIER_CEILING
        ):
            raise PilotV211GateError(
                f"{name} short-tier prompt binding is invalid"
            )
        disposition = _mapping(
            row.get("parse_disposition"),
            f"{name}.parse_disposition",
        )
        clipped = disposition.get("clipped", False)
        if not isinstance(clipped, bool):
            raise PilotV211GateError(
                f"{name} clipping disposition is invalid"
            )
        parse_success = (
            disposition.get("parse_status") == "success"
            and disposition.get("parse_mode") == "exact_json"
            and disposition.get("accepted") is True
        )
        samples[call_kind].append(
            {
                "finish_reason": row.get("finish_reason"),
                "response_completed": row.get("response_completed"),
                "output_disposition": row.get("output_disposition"),
                "error_type": row.get("error_type"),
                "parse_success": parse_success,
                "clipped": clipped,
                "prompt_tokens": int(usage["prompt_tokens"]),
                "completion_tokens": int(usage["completion_tokens"]),
                "reasoning_tokens": reasoning,
                "visible_completion_tokens": visible,
            }
        )
        usage_rows.append(
            {
                "response_model": _MODEL_PROFILES[model_id]["served_model"],
                "call_kind": call_kind,
                "usage": usage,
            }
        )
        provider_actual_rows.append(usage)
        if call_kind == "semantic":
            semantic_rows_by_index[index] = row
    if (
        len(samples["action"]) != 24
        or len(samples["semantic"]) != 8
    ):
        raise PilotV211GateError(
            f"{model_id} preflight denominator is not 24 action + 8 semantic"
        )
    proposal_outcomes = _sequence(
        checkpoint.get("proposal_outcomes"),
        f"{model_id} preflight proposal outcomes",
    )
    if (
        len(proposal_outcomes) != 8
        or checkpoint.get("proposal_outcomes_hash")
        != canonical_sha256(proposal_outcomes)
        or {row.get("call_index") for row in proposal_outcomes}
        != set(semantic_rows_by_index)
    ):
        raise PilotV211GateError(
            f"{model_id} preflight must bind all 8 proposal outcomes"
        )
    semantic_candidate_parse_failures = 0
    for outcome_index, raw_outcome in enumerate(proposal_outcomes):
        outcome = _mapping(
            raw_outcome,
            f"{model_id} proposal outcome {outcome_index}",
        )
        call_index = outcome.get("call_index")
        source = semantic_rows_by_index.get(call_index)
        if source is None:
            raise PilotV211GateError(
                f"{model_id} proposal outcome source call is missing"
            )
        events = outcome.get("semantic_events")
        if (
            outcome.get("current_t") != source.get("decision_t")
            or outcome.get("agent_id") != source.get("agent_id")
            or outcome.get("prompt_hash") != source.get("prompt_hash")
            or outcome.get("raw_output_hash")
            != source.get("raw_output_hash")
            or not isinstance(events, list)
            or outcome.get("semantic_events_hash")
            != canonical_sha256(events)
        ):
            raise PilotV211GateError(
                f"{model_id} proposal outcome source binding is incomplete"
            )
        disposition = _mapping(
            source.get("parse_disposition"),
            f"{model_id} proposal source disposition",
        )
        status = outcome.get("candidate_parse_status")
        if status == "success":
            valid = (
                outcome.get("candidate_parse_mode") == "exact_json"
                and outcome.get("failure_reason") is None
                and disposition.get("parse_status") == "success"
                and disposition.get("parse_mode") == "exact_json"
                and disposition.get("accepted") is True
            )
        elif status == "failure":
            semantic_candidate_parse_failures += 1
            valid = (
                outcome.get("candidate_parse_mode")
                in {
                    "exact_json",
                    "fenced_recovery",
                    "substring_recovery",
                    "parse_failure",
                }
                and isinstance(outcome.get("failure_reason"), str)
                and bool(outcome["failure_reason"])
                and events == []
                and disposition.get("parse_status") == "failure"
                and disposition.get("accepted") is False
            )
        else:
            valid = False
        if not valid:
            raise PilotV211GateError(
                f"{model_id} proposal outcome is not fully accounted"
            )
    # A candidate parse failure is a registered record-and-skip outcome, not a
    # provider/interface failure.  The proposal outcome validation above keeps
    # it in the denominator and proves that it was not silently omitted.
    for sample in samples["semantic"]:
        sample["parse_success"] = True
    denominator = {
        "planned_calls": 32,
        "observed_calls": 32,
        "successful_terminal_calls": 32,
        "failed_calls": 0,
        "action_calls": 24,
        "semantic_calls": 8,
        "semantic_candidate_parse_failures": (
            semantic_candidate_parse_failures
        ),
    }
    if checkpoint.get("provider_denominator") != denominator:
        raise PilotV211GateError(
            f"{model_id} preflight denominator receipt drifted"
        )
    provider_actual = _sum_usage(provider_actual_rows)
    totals = _mapping(
        checkpoint.get("provider_totals"),
        f"{model_id} preflight provider totals",
    )
    if (
        totals.get("call_count") != 32
        or totals.get("action_call_count") != 24
        or totals.get("semantic_call_count") != 8
        or totals.get("prompt_tokens") != provider_actual["prompt_tokens"]
        or totals.get("completion_tokens")
        != provider_actual["completion_tokens"]
        or not math.isclose(
            float(totals.get("cost_usd", -1.0)),
            float(provider_actual["cost_usd"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or checkpoint.get("provider_totals_hash")
        != canonical_sha256(totals)
    ):
        raise PilotV211GateError(
            f"{model_id} preflight provider totals/hash mismatch"
        )
    budget = _mapping(
        checkpoint.get("budget_snapshot_at_checkpoint"),
        f"{model_id} preflight budget snapshot",
    )
    if (
        checkpoint.get("budget_snapshot_hash") != canonical_sha256(budget)
        or budget.get("completed_calls") != 32
        or budget.get("active_calls") != 0
    ):
        raise PilotV211GateError(
            f"{model_id} preflight budget is not terminal"
        )
    actual = _usage(
        budget.get("accounted_usage"),
        f"{model_id} preflight budget.accounted_usage",
    )
    if not _usage_equal(actual, provider_actual):
        raise PilotV211GateError(
            f"{model_id} preflight provider/budget actuals differ"
        )
    journal_binding = _mapping(
        checkpoint.get("provider_call_journal_binding"),
        f"{model_id} preflight journal binding",
    )
    journal_sha256 = _sha256(
        journal_binding.get("journal_sha256"),
        f"{model_id} preflight journal SHA-256",
    )
    if (
        checkpoint.get("provider_call_journal_binding_hash")
        != canonical_sha256(journal_binding)
        or journal_binding.get("enabled") is not True
        or journal_binding.get("run_id") != checkpoint_run_id
        or journal_binding.get("contract_hash") != contract_sha256
        or journal_binding.get("event_count") != 64
        or journal_binding.get("completion_event_count") != 32
        or journal_binding.get("parse_disposition_event_count") != 32
    ):
        raise PilotV211GateError(
            f"{model_id} preflight provider journal binding is invalid"
        )
    verified_components = _mapping(
        exactness.get("verified_components"),
        f"{model_id} exactness verified components",
    )
    if (
        not verified_components
        or not all(value is True for value in verified_components.values())
        or exactness.get("checkpoint_hash") != checkpoint_hash
        or exactness.get("num_agents") != 2
        or exactness.get("completed_months") != 12
        or exactness.get("provider_calls_during_verification") != 0
        or exactness.get("provider_calls_hash")
        != checkpoint.get("provider_calls_hash")
        or exactness.get("proposal_outcomes_hash")
        != checkpoint.get("proposal_outcomes_hash")
        or exactness.get("provider_totals_hash")
        != checkpoint.get("provider_totals_hash")
        or exactness.get("budget_snapshot_hash")
        != checkpoint.get("budget_snapshot_hash")
        or exactness.get("provider_call_journal_binding_hash")
        != checkpoint.get("provider_call_journal_binding_hash")
        or exactness.get("provider_denominator") != denominator
    ):
        raise PilotV211GateError(
            f"{model_id} preflight exactness receipt is incomplete"
        )
    return {
        "ledger_run_id": ledger_run_id,
        "checkpoint_run_id": checkpoint_run_id,
        "run_spec_sha256": run_spec_sha256,
        "checkpoint_artifact_sha256": checkpoint_artifact_sha256,
        "checkpoint_content_sha256": str(checkpoint_hash),
        "checkpoint_schema_version": checkpoint.get("schema_version"),
        "exactness_artifact_sha256": exactness_artifact_sha256,
        "exactness_content_sha256": str(exactness_hash),
        "exactness_schema_version": exactness.get("schema_version"),
        "provider_call_journal_sha256": journal_sha256,
        "recorded_semantic_candidate_parse_failures": (
            semantic_candidate_parse_failures
        ),
        "samples": samples,
        "usage_rows": usage_rows,
        "actual_usage": actual,
    }


def _science_run_ids(
    value: Any,
) -> tuple[dict[str, list[str]], list[str]]:
    by_model = _mapping(value, "science_run_ids_by_model")
    if set(by_model) != _MODEL_IDS:
        raise PilotV211GateError(
            "science_run_ids_by_model must contain both registered models"
        )
    normalized: dict[str, list[str]] = {}
    all_ids: list[str] = []
    for model_id in sorted(_MODEL_IDS):
        rows = _sequence(
            by_model[model_id],
            f"{model_id} science run IDs",
        )
        expected = V211_GATE_SCIENCE_RUNS_PER_MODEL[model_id]
        if len(rows) != expected:
            raise PilotV211GateError(
                f"{model_id} must bind exactly {expected} science run IDs"
            )
        model_ids = [
            _run_id(item, f"{model_id} science run ID")
            for item in rows
        ]
        if len(model_ids) != len(set(model_ids)):
            raise PilotV211GateError(
                f"{model_id} science run IDs contain duplicates"
            )
        normalized[model_id] = model_ids
        all_ids.extend(model_ids)
    if (
        len(all_ids) != 131
        or len(all_ids) != len(set(all_ids))
    ):
        raise PilotV211GateError(
            "science run IDs must bind 131 unique registered cells"
        )
    return normalized, all_ids


def _source_manifest_hashes(value: Any) -> dict[str, str]:
    row = _mapping(value, "source_manifest_hashes")
    if set(row) != {"file_sha256", "content_sha256"}:
        raise PilotV211GateError(
            "source_manifest_hashes must contain file/content SHA-256"
        )
    normalized = {
        name: _sha256(item, f"source_manifest_hashes.{name}")
        for name, item in row.items()
    }
    expected = {
        "file_sha256": V211_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": V211_SOURCE_MANIFEST_CONTENT_SHA256,
    }
    if normalized != expected:
        raise PilotV211GateError(
            "source_manifest_hashes differ from the frozen V2.11 manifest"
        )
    return normalized


def _expected_model_status(
    *,
    capability_pass: bool,
    interface_pass: bool,
) -> str:
    if not capability_pass:
        return "capability-no-go"
    if not interface_pass:
        return "interface-no-go"
    return "eligible"


def build_v211_post_gate_authority(
    *,
    contract_sha256: str,
    release_tag: str,
    release_commit: str,
    parent_import_run_id: str,
    capability_artifacts: Mapping[str, Mapping[str, Any]],
    preflight_artifacts: Mapping[str, Mapping[str, Any]],
    model_terminal_statuses: Mapping[str, str],
    pre_science_actual_storage_bytes: int,
    ledger_event_chain_head: str,
    science_run_ids_by_model: Mapping[str, Sequence[str]],
    source_manifest_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Build the zero-provider V2.11 post-gate authority receipt.

    Structural, provenance, profile, usage, or denominator corruption raises
    :class:`PilotV211GateError`.  Valid terminal capability/parse failures stay
    in their ITT samples and produce a model-scoped no-go decision.
    """

    contract_hash = _sha256(contract_sha256, "contract_sha256")
    if release_tag != V211_GATE_RELEASE_TAG:
        raise PilotV211GateError("release_tag differs from V2.11")
    commit = _commit(release_commit, "release_commit")
    parent_run_id = _run_id(parent_import_run_id, "parent_import_run_id")
    ledger_head = _sha256(
        ledger_event_chain_head,
        "ledger_event_chain_head",
    )
    manifest_hashes = _source_manifest_hashes(source_manifest_hashes)
    storage_bytes = _nonnegative_integer(
        pre_science_actual_storage_bytes,
        "pre_science_actual_storage_bytes",
    )
    if (
        not isinstance(capability_artifacts, Mapping)
        or set(capability_artifacts) != _MODEL_IDS
        or not isinstance(preflight_artifacts, Mapping)
        or set(preflight_artifacts) != _MODEL_IDS
        or not isinstance(model_terminal_statuses, Mapping)
        or set(model_terminal_statuses) != _MODEL_IDS
    ):
        raise PilotV211GateError(
            "gate inputs must contain exactly both registered models"
        )
    statuses: dict[str, str] = {}
    for model_id in sorted(_MODEL_IDS):
        status = model_terminal_statuses[model_id]
        if status not in _MODEL_TERMINAL_STATUSES:
            raise PilotV211GateError(
                f"{model_id} terminal status is invalid"
            )
        statuses[model_id] = str(status)
    science_by_model, all_science_ids = _science_run_ids(
        science_run_ids_by_model
    )

    capability: dict[str, dict[str, Any]] = {}
    preflight: dict[str, dict[str, Any]] = {}
    all_actual_rows: list[Mapping[str, Any]] = []
    gate_calls_by_model: dict[str, int] = {}
    model_decisions: dict[str, dict[str, Any]] = {}
    observed_reservations: dict[str, dict[str, Any]] = {}
    dispatch_reservations: dict[str, dict[str, Any]] = {}
    gate_artifact_bindings: dict[str, dict[str, Any]] = {}
    authority_sources: dict[str, dict[str, Any]] = {}
    eligible_model_ids: list[str] = []

    for model_id in sorted(_MODEL_IDS):
        capability[model_id] = _capability_rows(
            model_id=model_id,
            envelope=capability_artifacts[model_id],
            contract_sha256=contract_hash,
        )
        preflight[model_id] = _preflight_envelope(
            preflight_artifacts[model_id],
            model_id=model_id,
            contract_sha256=contract_hash,
        )
        combined_samples = {
            call_kind: [
                *capability[model_id]["samples"][call_kind],
                *preflight[model_id]["samples"][call_kind],
            ]
            for call_kind in ("action", "semantic")
        }
        combined_usage_rows = [
            *capability[model_id]["usage_rows"],
            *preflight[model_id]["usage_rows"],
        ]
        projection_rows = preflight_p95(
            combined_usage_rows,
            reserve_multiplier=V211_GATE_RESERVE_MULTIPLIER,
        )
        served_model = str(_MODEL_PROFILES[model_id]["served_model"])
        expected_projection_keys = {
            f"{served_model}::action",
            f"{served_model}::semantic",
        }
        if set(projection_rows) != expected_projection_keys:
            raise PilotV211GateError(
                f"{model_id} fresh P95 call-kind grouping drifted"
            )
        reservations_by_kind = {
            call_kind: projection_rows[f"{served_model}::{call_kind}"]
            for call_kind in ("action", "semantic")
        }
        expected_counts = {
            "action": V211_GATE_ACTION_SAMPLES_PER_MODEL,
            "semantic": V211_GATE_SEMANTIC_SAMPLES_PER_MODEL,
        }
        interface_gates: dict[str, dict[str, Any]] = {}
        for call_kind in ("action", "semantic"):
            if (
                len(combined_samples[call_kind])
                != expected_counts[call_kind]
                or reservations_by_kind[call_kind].get("sample_count")
                != expected_counts[call_kind]
            ):
                raise PilotV211GateError(
                    f"{model_id} must retain exactly "
                    f"{expected_counts[call_kind]} {call_kind} samples"
                )
            try:
                gate = interface_sample_gate(
                    call_kind=call_kind,
                    wire_cap_tokens=V211_GATE_WIRE_CAP_TOKENS,
                    reservation=reservations_by_kind[call_kind],
                    samples=combined_samples[call_kind],
                    expected_sample_count=expected_counts[call_kind],
                    prompt_tier_ceiling_tokens=(
                        V211_GATE_PROMPT_TIER_CEILING
                    ),
                    minimum_headroom_fraction=0.25,
                )
            except PilotInterfaceGateError as exc:
                raise PilotV211GateError(
                    f"{model_id} {call_kind} interface sample is malformed: "
                    f"{exc}"
                ) from exc
            interface_gates[call_kind] = gate.to_dict()
        interface_pass = all(
            bool(value["passed"]) for value in interface_gates.values()
        )
        expected_status = _expected_model_status(
            capability_pass=bool(
                capability[model_id]["capability_pass"]
            ),
            interface_pass=interface_pass,
        )
        if statuses[model_id] != expected_status:
            raise PilotV211GateError(
                f"{model_id} terminal status differs from recomputed gates"
            )
        eligible = expected_status == "eligible"
        if eligible:
            eligible_model_ids.append(model_id)
            dispatch_reservations[model_id] = reservations_by_kind
        observed_reservations[model_id] = reservations_by_kind
        gate_calls_by_model[model_id] = (
            V211_GATE_CAPABILITY_CALLS_PER_MODEL
            + V211_GATE_PREFLIGHT_CALLS_PER_MODEL
        )
        all_actual_rows.extend(
            (
                capability[model_id]["actual_usage"],
                preflight[model_id]["actual_usage"],
            )
        )
        sample_hashes = {
            call_kind: canonical_sha256(combined_samples[call_kind])
            for call_kind in ("action", "semantic")
        }
        model_decisions[model_id] = {
            "terminal_status": expected_status,
            "eligible_for_science_dispatch": eligible,
            "capability_pass": bool(
                capability[model_id]["capability_pass"]
            ),
            "interface_pass": interface_pass,
            "sample_counts": dict(expected_counts),
            "sample_hashes": sample_hashes,
            "interface_gates": interface_gates,
            "recorded_preflight_semantic_candidate_parse_failures": (
                preflight[model_id][
                    "recorded_semantic_candidate_parse_failures"
                ]
            ),
            "no_go_science_cell_count": (
                0
                if eligible
                else V211_MODEL_SCIENCE_LEDGER_CELLS[model_id]
            ),
        }
        gate_artifact_bindings[model_id] = {
            "capability": {
                "run_id": capability[model_id]["run_id"],
                "artifact_sha256": capability[model_id][
                    "artifact_sha256"
                ],
                "schema_version": capability[model_id]["schema_version"],
            },
            "preflight": {
                "ledger_run_id": preflight[model_id]["ledger_run_id"],
                "checkpoint_run_id": preflight[model_id][
                    "checkpoint_run_id"
                ],
                "run_spec_sha256": preflight[model_id][
                    "run_spec_sha256"
                ],
                "checkpoint_artifact_sha256": preflight[model_id][
                    "checkpoint_artifact_sha256"
                ],
                "checkpoint_content_sha256": preflight[model_id][
                    "checkpoint_content_sha256"
                ],
                "checkpoint_schema_version": preflight[model_id][
                    "checkpoint_schema_version"
                ],
                "exactness_artifact_sha256": preflight[model_id][
                    "exactness_artifact_sha256"
                ],
                "exactness_content_sha256": preflight[model_id][
                    "exactness_content_sha256"
                ],
                "exactness_schema_version": preflight[model_id][
                    "exactness_schema_version"
                ],
                "provider_call_journal_sha256": preflight[model_id][
                    "provider_call_journal_sha256"
                ],
            },
        }
        if eligible:
            authority_sources[model_id] = {
                "source_preflight_run_id": preflight[model_id][
                    "ledger_run_id"
                ],
                "source_preflight_run_spec_sha256": preflight[model_id][
                    "run_spec_sha256"
                ],
                "source_model_id": model_id,
                "source_served_model": served_model,
                "source_execution_artifact_sha256": preflight[model_id][
                    "checkpoint_artifact_sha256"
                ],
                "source_provider_call_journal_sha256": preflight[model_id][
                    "provider_call_journal_sha256"
                ],
            }

    actual_usage = _sum_usage(all_actual_rows)
    actual_calls = sum(gate_calls_by_model.values())
    if actual_calls != V211_PREFLIGHT_ACTUAL_COMPLETIONS:
        raise PilotV211GateError(
            "post-gate actual denominator must remain exactly 124 calls"
        )
    no_go_counts = {
        model_id: (
            0
            if model_id in eligible_model_ids
            else V211_MODEL_SCIENCE_LEDGER_CELLS[model_id]
        )
        for model_id in sorted(_MODEL_IDS)
    }
    try:
        projection_reservations = {
            model_id: {
                call_kind: dispatch_reservations[model_id][call_kind][
                    "reserved_p95"
                ]
                for call_kind in ("action", "semantic")
            }
            for model_id in eligible_model_ids
        }
        full_projection = project_v211_full_matrix(
            projection_reservations,
            pre_science_actual_usage=actual_usage,
            pre_science_actual_hosted_completions=actual_calls,
            pre_science_actual_storage_bytes=storage_bytes,
            eligible_model_ids=eligible_model_ids,
            gate_actual_calls_by_model=gate_calls_by_model,
            no_go_ledger_cells_by_model=no_go_counts,
        )
    except PilotV211ProjectionError as exc:
        raise PilotV211GateError(
            f"V2.11 full-matrix projection is invalid: {exc}"
        ) from exc

    gate_run_ids = [
        run_id
        for model_id in sorted(_MODEL_IDS)
        for run_id in (
            gate_artifact_bindings[model_id]["capability"]["run_id"],
            gate_artifact_bindings[model_id]["preflight"]["ledger_run_id"],
        )
    ]
    all_ledger_run_ids = [parent_run_id, *gate_run_ids, *all_science_ids]
    if (
        len(all_ledger_run_ids) != V211_TOTAL_LEDGER_CELLS
        or len(all_ledger_run_ids) != len(set(all_ledger_run_ids))
    ):
        raise PilotV211GateError(
            "parent/gate/science run IDs do not bind all 136 unique cells"
        )
    no_go_science_run_ids = {
        model_id: (
            []
            if model_id in eligible_model_ids
            else list(science_by_model[model_id])
        )
        for model_id in sorted(_MODEL_IDS)
    }
    global_reasons = list(full_projection.reasons)
    if not eligible_model_ids:
        global_reasons.append("no-dispatch-eligible-models")
    global_go = not global_reasons
    receipt: dict[str, Any] = {
        "schema_version": V211_GATE_SCHEMA_VERSION,
        "contract_sha256": contract_hash,
        "release": {
            "tag": release_tag,
            "commit": commit,
        },
        "bindings": {
            "parent_import_run_id": parent_run_id,
            "gate_artifacts": gate_artifact_bindings,
            "ledger_event_chain_head": ledger_head,
            "source_manifest_hashes": manifest_hashes,
        },
        "denominator": {
            "registered_ledger_cells": V211_TOTAL_LEDGER_CELLS,
            "registered_science_cells": len(all_science_ids),
            "registered_hosted_calls": 5_940,
            "gate_actual_calls": actual_calls,
            "registered_remaining_science_calls": (
                V211_REMAINING_SCIENCE_COMPLETIONS
            ),
            "science_run_ids_by_model": science_by_model,
            "science_run_ids_sha256": canonical_sha256(all_science_ids),
            "all_ledger_run_ids_sha256": canonical_sha256(
                all_ledger_run_ids
            ),
            "eligible_model_ids": eligible_model_ids,
            "no_go_ledger_cells_by_model": no_go_counts,
            "no_go_science_run_ids_by_model": no_go_science_run_ids,
        },
        "model_decisions": model_decisions,
        "actuals": {
            "usage": actual_usage,
            "hosted_completions": actual_calls,
            "storage_bytes": storage_bytes,
            "calls_by_model": gate_calls_by_model,
        },
        "observed_reservations": observed_reservations,
        "dispatch_reservations": dispatch_reservations,
        "authority_sources": authority_sources,
        "projection": full_projection.to_dict(),
        "provider_construction_during_authority": False,
        "provider_calls_during_authority": 0,
        "go": global_go,
        "reasons": global_reasons,
        "claim_boundary": (
            "This receipt is a zero-provider model-scoped interface and "
            "budget authority. It preserves all ITT cells and is not "
            "scientific effect evidence."
        ),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _verify_receipt_structure(
    receipt: Mapping[str, Any],
    *,
    expected_contract_sha256: str | None,
    expected_git_commit: str | None,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "contract_sha256",
        "release",
        "bindings",
        "denominator",
        "model_decisions",
        "actuals",
        "observed_reservations",
        "dispatch_reservations",
        "authority_sources",
        "projection",
        "provider_construction_during_authority",
        "provider_calls_during_authority",
        "go",
        "reasons",
        "claim_boundary",
        "receipt_sha256",
    }
    if set(receipt) != expected_fields:
        raise PilotV211GateError("V2.11 gate receipt fields drifted")
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256", None)
    if (
        _SHA256_PATTERN.fullmatch(str(claimed)) is None
        or canonical_sha256(unsigned) != claimed
    ):
        raise PilotV211GateError("V2.11 gate receipt self-hash mismatch")
    if receipt.get("schema_version") != V211_GATE_SCHEMA_VERSION:
        raise PilotV211GateError("V2.11 gate receipt schema drifted")
    contract_hash = _sha256(
        receipt.get("contract_sha256"),
        "receipt contract_sha256",
    )
    if (
        expected_contract_sha256 is not None
        and contract_hash
        != _sha256(
            expected_contract_sha256,
            "expected_contract_sha256",
        )
    ):
        raise PilotV211GateError("V2.11 gate contract binding mismatch")
    release = _mapping(receipt.get("release"), "receipt release")
    if release.get("tag") != V211_GATE_RELEASE_TAG:
        raise PilotV211GateError("V2.11 gate release tag mismatch")
    release_commit = _commit(
        release.get("commit"),
        "receipt release commit",
    )
    if (
        expected_git_commit is not None
        and release_commit
        != _commit(expected_git_commit, "expected_git_commit")
    ):
        raise PilotV211GateError("V2.11 gate release commit mismatch")
    if (
        receipt.get("provider_construction_during_authority") is not False
        or receipt.get("provider_calls_during_authority") != 0
    ):
        raise PilotV211GateError(
            "V2.11 gate authority is not zero-provider"
        )
    bindings = _mapping(receipt.get("bindings"), "receipt bindings")
    if set(bindings) != {
        "parent_import_run_id",
        "gate_artifacts",
        "ledger_event_chain_head",
        "source_manifest_hashes",
    }:
        raise PilotV211GateError("V2.11 gate receipt bindings drifted")
    _run_id(
        bindings.get("parent_import_run_id"),
        "receipt parent import run ID",
    )
    _sha256(
        bindings.get("ledger_event_chain_head"),
        "receipt ledger event-chain head",
    )
    _source_manifest_hashes(bindings.get("source_manifest_hashes"))
    gate_artifacts = _mapping(
        bindings.get("gate_artifacts"),
        "receipt gate artifact bindings",
    )
    if set(gate_artifacts) != _MODEL_IDS:
        raise PilotV211GateError(
            "V2.11 receipt gate artifact model set drifted"
        )
    gate_run_ids: list[str] = []
    for model_id in sorted(_MODEL_IDS):
        model_artifacts = _mapping(
            gate_artifacts[model_id],
            f"{model_id} receipt gate artifacts",
        )
        if set(model_artifacts) != {"capability", "preflight"}:
            raise PilotV211GateError(
                f"{model_id} receipt gate artifact kinds drifted"
            )
        capability_binding = _mapping(
            model_artifacts["capability"],
            f"{model_id} receipt capability binding",
        )
        if (
            set(capability_binding)
            != {"run_id", "artifact_sha256", "schema_version"}
            or capability_binding.get("schema_version")
            != CAPABILITY_SCHEMA_VERSION
        ):
            raise PilotV211GateError(
                f"{model_id} receipt capability binding drifted"
            )
        gate_run_ids.append(
            _run_id(
                capability_binding.get("run_id"),
                f"{model_id} receipt capability run ID",
            )
        )
        _sha256(
            capability_binding.get("artifact_sha256"),
            f"{model_id} receipt capability artifact SHA-256",
        )
        preflight_binding = _mapping(
            model_artifacts["preflight"],
            f"{model_id} receipt preflight binding",
        )
        if (
            set(preflight_binding)
            != {
                "ledger_run_id",
                "checkpoint_run_id",
                "run_spec_sha256",
                "checkpoint_artifact_sha256",
                "checkpoint_content_sha256",
                "checkpoint_schema_version",
                "exactness_artifact_sha256",
                "exactness_content_sha256",
                "exactness_schema_version",
                "provider_call_journal_sha256",
            }
            or preflight_binding.get("checkpoint_schema_version")
            != PILOT_CHECKPOINT_SCHEMA_VERSION_V4
            or preflight_binding.get("exactness_schema_version")
            != V211_PREFLIGHT_EXACTNESS_SCHEMA_VERSION
        ):
            raise PilotV211GateError(
                f"{model_id} receipt preflight binding drifted"
            )
        gate_run_ids.append(
            _run_id(
                preflight_binding.get("ledger_run_id"),
                f"{model_id} receipt preflight ledger run ID",
            )
        )
        checkpoint_run_id = _run_id(
            preflight_binding.get("checkpoint_run_id"),
            f"{model_id} receipt preflight checkpoint run ID",
        )
        if checkpoint_run_id != (
            preflight_binding["ledger_run_id"]
            + V211_PREFLIGHT_CHECKPOINT_RUN_SUFFIX
        ):
            raise PilotV211GateError(
                f"{model_id} receipt checkpoint/ledger run IDs drifted"
            )
        for name in (
            "run_spec_sha256",
            "checkpoint_artifact_sha256",
            "checkpoint_content_sha256",
            "exactness_artifact_sha256",
            "exactness_content_sha256",
            "provider_call_journal_sha256",
        ):
            _sha256(
                preflight_binding.get(name),
                f"{model_id} receipt preflight {name}",
            )
    denominator = _mapping(
        receipt.get("denominator"),
        "receipt denominator",
    )
    if (
        denominator.get("registered_ledger_cells")
        != V211_TOTAL_LEDGER_CELLS
        or denominator.get("registered_science_cells") != 131
        or denominator.get("registered_hosted_calls") != 5_940
        or denominator.get("gate_actual_calls")
        != V211_PREFLIGHT_ACTUAL_COMPLETIONS
        or denominator.get("registered_remaining_science_calls")
        != V211_REMAINING_SCIENCE_COMPLETIONS
    ):
        raise PilotV211GateError("V2.11 gate denominator drifted")
    science_by_model, all_science_ids = _science_run_ids(
        denominator.get("science_run_ids_by_model")
    )
    if denominator.get("science_run_ids_sha256") != canonical_sha256(
        all_science_ids
    ):
        raise PilotV211GateError(
            "V2.11 science run-ID hash mismatch"
        )
    all_ledger_ids = [
        str(bindings["parent_import_run_id"]),
        *gate_run_ids,
        *all_science_ids,
    ]
    if (
        len(all_ledger_ids) != V211_TOTAL_LEDGER_CELLS
        or len(all_ledger_ids) != len(set(all_ledger_ids))
        or denominator.get("all_ledger_run_ids_sha256")
        != canonical_sha256(all_ledger_ids)
    ):
        raise PilotV211GateError(
            "V2.11 all-ledger run-ID binding drifted"
        )
    eligible = denominator.get("eligible_model_ids")
    if (
        not isinstance(eligible, list)
        or eligible != sorted(eligible)
        or len(eligible) != len(set(eligible))
        or not set(eligible) <= _MODEL_IDS
    ):
        raise PilotV211GateError(
            "V2.11 eligible model IDs are invalid"
        )
    no_go_counts = _mapping(
        denominator.get("no_go_ledger_cells_by_model"),
        "receipt no-go cell counts",
    )
    expected_no_go_counts = {
        model_id: (
            0
            if model_id in eligible
            else V211_MODEL_SCIENCE_LEDGER_CELLS[model_id]
        )
        for model_id in sorted(_MODEL_IDS)
    }
    if no_go_counts != expected_no_go_counts:
        raise PilotV211GateError(
            "V2.11 no-go cell counts differ from model eligibility"
        )
    no_go_ids = _mapping(
        denominator.get("no_go_science_run_ids_by_model"),
        "receipt no-go science run IDs",
    )
    if no_go_ids != {
        model_id: (
            []
            if model_id in eligible
            else science_by_model[model_id]
        )
        for model_id in sorted(_MODEL_IDS)
    }:
        raise PilotV211GateError(
            "V2.11 no-go run IDs differ from model eligibility"
        )
    model_decisions = _mapping(
        receipt.get("model_decisions"),
        "receipt model decisions",
    )
    if set(model_decisions) != _MODEL_IDS:
        raise PilotV211GateError("V2.11 model decision set drifted")
    for model_id in sorted(_MODEL_IDS):
        decision = _mapping(
            model_decisions[model_id],
            f"{model_id} receipt decision",
        )
        model_eligible = model_id in eligible
        if (
            decision.get("eligible_for_science_dispatch")
            is not model_eligible
            or decision.get("no_go_science_cell_count")
            != expected_no_go_counts[model_id]
            or decision.get("sample_counts")
            != {
                "action": V211_GATE_ACTION_SAMPLES_PER_MODEL,
                "semantic": V211_GATE_SEMANTIC_SAMPLES_PER_MODEL,
            }
            or isinstance(
                decision.get(
                    "recorded_preflight_semantic_candidate_parse_failures"
                ),
                bool,
            )
            or not isinstance(
                decision.get(
                    "recorded_preflight_semantic_candidate_parse_failures"
                ),
                int,
            )
            or not 0
            <= decision[
                "recorded_preflight_semantic_candidate_parse_failures"
            ]
            <= 8
        ):
            raise PilotV211GateError(
                f"{model_id} receipt decision/denominator drifted"
            )
        expected_status = _expected_model_status(
            capability_pass=decision.get("capability_pass") is True,
            interface_pass=decision.get("interface_pass") is True,
        )
        if (
            decision.get("terminal_status") != expected_status
            or model_eligible != (expected_status == "eligible")
        ):
            raise PilotV211GateError(
                f"{model_id} receipt terminal status drifted"
            )
        gates = _mapping(
            decision.get("interface_gates"),
            f"{model_id} receipt interface gates",
        )
        if set(gates) != {"action", "semantic"}:
            raise PilotV211GateError(
                f"{model_id} receipt interface gate set drifted"
            )
        recomputed_interface_pass = all(
            _mapping(
                gates[call_kind],
                f"{model_id} {call_kind} receipt gate",
            ).get("passed")
            is True
            for call_kind in ("action", "semantic")
        )
        if recomputed_interface_pass != (
            decision.get("interface_pass") is True
        ):
            raise PilotV211GateError(
                f"{model_id} receipt interface pass drifted"
            )
    actuals = _mapping(receipt.get("actuals"), "receipt actuals")
    actual_usage = _usage(actuals.get("usage"), "receipt actual usage")
    if (
        actuals.get("hosted_completions")
        != V211_PREFLIGHT_ACTUAL_COMPLETIONS
        or actuals.get("calls_by_model")
        != dict(V211_MODEL_GATE_REGISTERED_CALLS)
    ):
        raise PilotV211GateError("V2.11 receipt actual call counts drifted")
    storage = _nonnegative_integer(
        actuals.get("storage_bytes"),
        "receipt actual storage",
    )
    observed = _mapping(
        receipt.get("observed_reservations"),
        "receipt observed reservations",
    )
    if set(observed) != _MODEL_IDS:
        raise PilotV211GateError(
            "V2.11 receipt observed reservation models drifted"
        )
    for model_id in sorted(_MODEL_IDS):
        by_kind = _mapping(
            observed[model_id],
            f"{model_id} observed reservations",
        )
        if set(by_kind) != {"action", "semantic"}:
            raise PilotV211GateError(
                f"{model_id} observed reservation kinds drifted"
            )
        for call_kind in ("action", "semantic"):
            try:
                reservation = PreflightP95Reservation.from_dict(
                    model=str(_MODEL_PROFILES[model_id]["runtime_model"]),
                    call_kind=call_kind,
                    value=by_kind[call_kind],
                )
            except (TypeError, ValueError) as exc:
                raise PilotV211GateError(
                    f"{model_id} {call_kind} reservation is invalid: {exc}"
                ) from exc
            expected_samples = (
                V211_GATE_ACTION_SAMPLES_PER_MODEL
                if call_kind == "action"
                else V211_GATE_SEMANTIC_SAMPLES_PER_MODEL
            )
            if reservation.sample_count != expected_samples:
                raise PilotV211GateError(
                    f"{model_id} {call_kind} reservation sample count drifted"
                )
    dispatch = _mapping(
        receipt.get("dispatch_reservations"),
        "receipt dispatch reservations",
    )
    if set(dispatch) != set(eligible):
        raise PilotV211GateError(
            "V2.11 dispatch reservations differ from model eligibility"
        )
    if any(dispatch[model_id] != observed[model_id] for model_id in eligible):
        raise PilotV211GateError(
            "V2.11 dispatch reservations differ from observed authority"
        )
    sources = _mapping(
        receipt.get("authority_sources"),
        "receipt authority sources",
    )
    if set(sources) != set(eligible):
        raise PilotV211GateError(
            "V2.11 authority sources differ from model eligibility"
        )
    for model_id in eligible:
        source = _mapping(
            sources[model_id],
            f"{model_id} authority source",
        )
        if set(source) != {
            "source_preflight_run_id",
            "source_preflight_run_spec_sha256",
            "source_model_id",
            "source_served_model",
            "source_execution_artifact_sha256",
            "source_provider_call_journal_sha256",
        }:
            raise PilotV211GateError(
                f"{model_id} authority source fields drifted"
            )
        if (
            source.get("source_model_id") != model_id
            or source.get("source_served_model")
            != _MODEL_PROFILES[model_id]["served_model"]
            or source.get("source_preflight_run_id")
            != gate_artifacts[model_id]["preflight"]["ledger_run_id"]
            or source.get("source_preflight_run_spec_sha256")
            != gate_artifacts[model_id]["preflight"]["run_spec_sha256"]
            or source.get("source_execution_artifact_sha256")
            != gate_artifacts[model_id]["preflight"][
                "checkpoint_artifact_sha256"
            ]
            or source.get("source_provider_call_journal_sha256")
            != gate_artifacts[model_id]["preflight"][
                "provider_call_journal_sha256"
            ]
        ):
            raise PilotV211GateError(
                f"{model_id} authority source binding drifted"
            )
        _run_id(
            source.get("source_preflight_run_id"),
            f"{model_id} source preflight run ID",
        )
        for name in (
            "source_preflight_run_spec_sha256",
            "source_execution_artifact_sha256",
            "source_provider_call_journal_sha256",
        ):
            _sha256(source.get(name), f"{model_id}.{name}")
    projection = _mapping(
        receipt.get("projection"),
        "receipt full-matrix projection",
    )
    try:
        projection_reservations = {
            model_id: {
                call_kind: dispatch[model_id][call_kind]["reserved_p95"]
                for call_kind in ("action", "semantic")
            }
            for model_id in eligible
        }
        recomputed_projection = project_v211_full_matrix(
            projection_reservations,
            pre_science_actual_usage=actual_usage,
            pre_science_actual_hosted_completions=(
                V211_PREFLIGHT_ACTUAL_COMPLETIONS
            ),
            pre_science_actual_storage_bytes=storage,
            eligible_model_ids=eligible,
            gate_actual_calls_by_model=dict(
                V211_MODEL_GATE_REGISTERED_CALLS
            ),
            no_go_ledger_cells_by_model=expected_no_go_counts,
        ).to_dict()
    except PilotV211ProjectionError as exc:
        raise PilotV211GateError(
            f"V2.11 receipt projection cannot be recomputed: {exc}"
        ) from exc
    if projection != recomputed_projection:
        raise PilotV211GateError(
            "V2.11 receipt full-matrix projection drifted"
        )
    reasons = receipt.get("reasons")
    if not isinstance(reasons, list):
        raise PilotV211GateError("V2.11 receipt reasons must be an array")
    expected_reasons = list(recomputed_projection["reasons"])
    if not eligible:
        expected_reasons.append("no-dispatch-eligible-models")
    expected_go = not expected_reasons
    if (
        reasons != expected_reasons
        or receipt.get("go") is not expected_go
    ):
        raise PilotV211GateError("V2.11 receipt go/no-go drifted")
    return dict(receipt)


def verify_v211_gate_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_contract_sha256: str | None = None,
    expected_git_commit: str | None = None,
) -> dict[str, Any]:
    """Verify a loaded V2.11 gate receipt without constructing a provider."""

    return _verify_receipt_structure(
        _mapping(receipt, "V2.11 gate receipt"),
        expected_contract_sha256=expected_contract_sha256,
        expected_git_commit=expected_git_commit,
    )


def _strict_json_object(data: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PilotV211GateError(f"{name} is not UTF-8") from exc

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise PilotV211GateError(f"{name} is not strict JSON") from exc
    return _mapping(value, name)


def _safe_relative_receipt_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str):
        raise PilotV211GateError("receipt_path must be text")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or path.as_posix() != value
        or len(path.parts) < 3
        or path.parts[0] != "experiment_results"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PilotV211GateError(
            "receipt_path must be normalized below experiment_results/"
        )
    return path


def _read_no_follow(
    *,
    repo_root: str | Path,
    receipt_path: PurePosixPath,
) -> tuple[Path, bytes]:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise PilotV211GateError(
            "guarded receipt reads require no-follow directory support"
        )
    root = Path(repo_root).absolute()
    try:
        root_mode = os.lstat(root).st_mode
    except OSError as exc:
        raise PilotV211GateError("repo_root cannot be inspected") from exc
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise PilotV211GateError(
            "repo_root must be a real non-symlink directory"
        )
    target = root.joinpath(*receipt_path.parts)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(root, directory_flags)
    except OSError as exc:
        raise PilotV211GateError(
            "repo_root cannot be opened for guarded receipt reads"
        ) from exc
    try:
        for index, part in enumerate(receipt_path.parts):
            final = index == len(receipt_path.parts) - 1
            flags = os.O_RDONLY | os.O_NOFOLLOW
            if not final:
                flags |= os.O_DIRECTORY
            try:
                next_descriptor = os.open(
                    part,
                    flags,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise PilotV211GateError(
                    "receipt path cannot traverse symlinks or unsafe nodes"
                ) from exc
            os.close(descriptor)
            descriptor = next_descriptor
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise PilotV211GateError("receipt must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
        ):
            raise PilotV211GateError(
                "receipt changed during guarded read"
            )
        return target, b"".join(chunks)
    finally:
        os.close(descriptor)


def _source_authority(
    *,
    receipt: Mapping[str, Any],
    model_id: str,
    receipt_file_sha256: str,
) -> dict[str, str]:
    source = _mapping(
        _mapping(
            receipt.get("authority_sources"),
            "receipt authority sources",
        ).get(model_id),
        f"{model_id} authority source",
    )
    return {
        "authority_id": OBSERVED_P95_AUTHORITY_ID,
        "source_kind": OBSERVED_P95_SOURCE_KIND,
        "pilot_contract_hash": str(receipt["contract_sha256"]),
        "pilot_tag": str(receipt["release"]["tag"]),
        "source_projection_schema_version": (
            OBSERVED_P95_PROJECTION_SCHEMA_VERSION
        ),
        "source_projection_file_sha256": receipt_file_sha256,
        "source_projection_content_sha256": str(
            receipt["receipt_sha256"]
        ),
        "source_preflight_run_id": str(
            source["source_preflight_run_id"]
        ),
        "source_preflight_run_spec_sha256": str(
            source["source_preflight_run_spec_sha256"]
        ),
        "source_model_id": str(source["source_model_id"]),
        "source_served_model": str(source["source_served_model"]),
        "source_execution_artifact_sha256": str(
            source["source_execution_artifact_sha256"]
        ),
        "source_provider_call_journal_sha256": str(
            source["source_provider_call_journal_sha256"]
        ),
    }


def verified_v211_gate_authority_binding(
    receipt_path: str,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
    expected_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify one immutable receipt file and return its flat source binding.

    The returned ``reservations`` use runtime model keys and contain source
    authority fields plus the numeric reservation.  The four receipt-level
    fields are intentionally flat, matching the existing source-backed runner
    verifier convention.
    """

    relative = _safe_relative_receipt_path(receipt_path)
    _, data = _read_no_follow(
        repo_root=repo_root,
        receipt_path=relative,
    )
    file_sha256 = hashlib.sha256(data).hexdigest()
    receipt = verify_v211_gate_receipt(
        _strict_json_object(data, name="V2.11 gate receipt file"),
        expected_contract_sha256=expected_contract_sha256,
        expected_git_commit=expected_git_commit,
    )
    if receipt.get("go") is not True:
        raise PilotV211GateError(
            "V2.11 gate receipt is global no-go"
        )
    reservations: dict[str, dict[str, Any]] = {}
    eligible = receipt["denominator"]["eligible_model_ids"]
    for model_id in eligible:
        runtime_model = str(_MODEL_PROFILES[model_id]["runtime_model"])
        source = _source_authority(
            receipt=receipt,
            model_id=model_id,
            receipt_file_sha256=file_sha256,
        )
        reservations[runtime_model] = {
            call_kind: {
                "authority": dict(source),
                "reservation": receipt["dispatch_reservations"][model_id][
                    call_kind
                ],
            }
            for call_kind in ("action", "semantic")
        }
    return {
        "receipt_path": relative.as_posix(),
        "receipt_file_sha256": file_sha256,
        "receipt_content_sha256": receipt["receipt_sha256"],
        "git_commit": expected_git_commit,
        "reservations": reservations,
    }


def runner_reservations_from_v211_gate_binding(
    binding: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Expand a verified flat binding to runner-compatible authority rows."""

    value = _mapping(binding, "V2.11 gate authority binding")
    expected = {
        "receipt_path",
        "receipt_file_sha256",
        "receipt_content_sha256",
        "git_commit",
        "reservations",
    }
    if set(value) != expected:
        raise PilotV211GateError(
            "V2.11 gate authority binding fields drifted"
        )
    receipt_path = _safe_relative_receipt_path(value["receipt_path"]).as_posix()
    file_sha256 = _sha256(
        value["receipt_file_sha256"],
        "binding receipt_file_sha256",
    )
    content_sha256 = _sha256(
        value["receipt_content_sha256"],
        "binding receipt_content_sha256",
    )
    git_commit = _commit(value["git_commit"], "binding git_commit")
    source = _mapping(value["reservations"], "binding reservations")
    expected_runtime_models = {
        str(profile["runtime_model"]) for profile in _MODEL_PROFILES.values()
    }
    if not set(source) <= expected_runtime_models or not source:
        raise PilotV211GateError(
            "binding reservations contain no registered runtime model"
        )
    result: dict[str, dict[str, Any]] = {}
    for runtime_model, raw_by_kind in source.items():
        by_kind = _mapping(
            raw_by_kind,
            f"{runtime_model} binding reservations",
        )
        if set(by_kind) != {"action", "semantic"}:
            raise PilotV211GateError(
                f"{runtime_model} binding call kinds drifted"
            )
        result[runtime_model] = {}
        for call_kind in ("action", "semantic"):
            entry = _mapping(
                by_kind[call_kind],
                f"{runtime_model}::{call_kind} binding",
            )
            if set(entry) != {"authority", "reservation"}:
                raise PilotV211GateError(
                    f"{runtime_model}::{call_kind} binding fields drifted"
                )
            authority = dict(
                _mapping(
                    entry["authority"],
                    f"{runtime_model}::{call_kind} authority",
                )
            )
            authority.update(
                {
                    "source_authority_receipt_path": receipt_path,
                    "source_authority_receipt_file_sha256": file_sha256,
                    "source_authority_receipt_content_sha256": content_sha256,
                    "source_release_commit": git_commit,
                }
            )
            candidate = {
                "authority": authority,
                "reservation": _mapping(
                    entry["reservation"],
                    f"{runtime_model}::{call_kind} reservation",
                ),
            }
            try:
                sealed = ObservedPreflightP95Reservation.from_dict(
                    model=runtime_model,
                    call_kind=call_kind,
                    value=candidate,
                ).to_dict()
            except (TypeError, ValueError) as exc:
                raise PilotV211GateError(
                    f"{runtime_model}::{call_kind} observed authority is invalid: "
                    f"{exc}"
                ) from exc
            result[runtime_model][call_kind] = sealed
    return result


def runner_reservations_from_v211_gate(
    receipt_path: str,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
    expected_contract_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Verify a receipt file and return runner-compatible reservation rows."""

    binding = verified_v211_gate_authority_binding(
        receipt_path,
        repo_root=repo_root,
        expected_git_commit=expected_git_commit,
        expected_contract_sha256=expected_contract_sha256,
    )
    return runner_reservations_from_v211_gate_binding(binding)


__all__ = [
    "PilotV211GateError",
    "V211_GATE_ACTION_SAMPLES_PER_MODEL",
    "V211_GATE_CAPABILITY_CALLS_PER_MODEL",
    "V211_GATE_PREFLIGHT_CALLS_PER_MODEL",
    "V211_GATE_PROMPT_TIER_CEILING",
    "V211_PREFLIGHT_CHECKPOINT_RUN_SUFFIX",
    "V211_GATE_RELEASE_TAG",
    "V211_GATE_RESERVE_MULTIPLIER",
    "V211_GATE_SCHEMA_VERSION",
    "V211_GATE_SEMANTIC_SAMPLES_PER_MODEL",
    "V211_GATE_SCIENCE_RUNS_PER_MODEL",
    "V211_GATE_WIRE_CAP_TOKENS",
    "build_v211_post_gate_authority",
    "canonical_sha256",
    "runner_reservations_from_v211_gate",
    "runner_reservations_from_v211_gate_binding",
    "verified_v211_gate_authority_binding",
    "verify_v211_gate_receipt",
]
