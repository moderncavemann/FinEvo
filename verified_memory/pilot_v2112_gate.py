"""Zero-provider post-gate authority for the FinEvo V2.11.2 pilot.

V2.11.2 inherits two replay-verified, child-bound capability wrappers from
the immutable V2.11.1 no-go release and adds two fresh 2x12 long-context
preflights.  The inherited 60 calls are evidence, but their spend and calls
already live in the cumulative parent debit.  Only the fresh 64 preflight
calls are charged to the V2.11.2 attempt.

The normal-science reservation is observed p95 plus 25 percent over only the
fresh, same-model V2.11.2 long-context samples.  Historical V2.11.1
capability samples remain capability-audit evidence and never contribute to
the dispatch reservation.  The operational contract-envelope bootstrap is
intentionally not accepted by this module.
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

from .pilot_budget import ParentBudgetDebit, preflight_p95
from .pilot_interface_gate import PilotInterfaceGateError, interface_sample_gate
from .pilot_v211_gate import (
    PilotV211GateError,
    V211_PREFLIGHT_CHECKPOINT_RUN_SUFFIX,
    _preflight_envelope,
)
from .pilot_v2112_parent_import import (
    V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION,
    V2112_CAPABILITY_WRAPPER_SCHEMA_VERSION,
    V2112_SOURCE_MANIFEST_CONTENT_SHA256,
    V2112_SOURCE_MANIFEST_FILE_SHA256,
    V2112_SOURCE_MANIFEST_PATH,
    V2111_BUDGET_LEDGER_SHA256 as V211_PARENT_BUDGET_LEDGER_SHA256,
    V2111_CONTRACT_ID as V211_PARENT_CONTRACT_ID,
    V2111_CONTRACT_SHA256 as V211_PARENT_CONTRACT_SHA256,
    V2111_CUMULATIVE_COMPLETIONS as V211_PARENT_CUMULATIVE_COMPLETIONS,
    V2111_CUMULATIVE_COST_USD as V211_PARENT_CUMULATIVE_COST_USD,
    V2111_CUMULATIVE_STORAGE_BYTES as V211_PARENT_CUMULATIVE_STORAGE_BYTES,
    V2111_PARENT_DEBIT_RECORD_SHA256 as V211_PARENT_DEBIT_RECORD_SHA256,
    V2111_RUN_LEDGER_SHA256 as V211_PARENT_RUN_LEDGER_SHA256,
    V2111_SCIENCE_COMMIT as V211_PARENT_SCIENCE_COMMIT,
    V2111_SCIENCE_TAG as V211_PARENT_SCIENCE_TAG,
    V2111_SCIENCE_TAG_OBJECT as V211_PARENT_SCIENCE_TAG_OBJECT,
)
from .runner import (
    PROVIDER_CALL_JOURNAL_SCHEMA_VERSION,
    OBSERVED_P95_AUTHORITY_ID,
    OBSERVED_P95_PROJECTION_SCHEMA_VERSION,
    OBSERVED_P95_SOURCE_KIND,
    ContractEnvelopeBootstrapReservation,
    ObservedPreflightP95Reservation,
    PreflightP95Reservation,
    V2112_CONTRACT_ENVELOPE_AUTHORITY_ID,
    V2112_SOURCE_CONTRACT_ID,
    V2112_SOURCE_RELEASE_TAG,
    VerifiedRunError,
    verify_provider_call_journal,
)


V2112_GATE_SCHEMA_VERSION = "finevo-pilot-v2.11.2-post-gate-authority-v1"
V2112_PREFLIGHT_EXACTNESS_SCHEMA_VERSION = (
    "finevo-v2.11.2-long-context-preflight-exactness-receipt-v1"
)
V2112_GATE_CONTRACT_ID = "finevo-pilot-v2.11.2"
V2112_GATE_RELEASE_TAG = "pilot-v2.11.2-science"
V2112_GATE_RESERVE_MULTIPLIER = 1.25
V2112_GATE_WIRE_CAP_TOKENS = 4_096
V2112_GATE_PROMPT_TIER_CEILING = 200_000

V2112_INHERITED_CAPABILITY_CALLS_PER_MODEL = 30
V2112_FRESH_PREFLIGHT_CALLS_PER_MODEL = 32
V2112_GATE_EVIDENCE_CALLS_PER_MODEL = 62
V2112_INHERITED_CAPABILITY_CALLS = 60
V2112_FRESH_PREFLIGHT_CALLS = 64
V2112_GATE_EVIDENCE_CALLS = 124
V2112_ACTION_SAMPLES_PER_MODEL = 24
V2112_SEMANTIC_SAMPLES_PER_MODEL = 8

V2112_REGISTERED_SCIENCE_CALLS = 5_816
V2112_REGISTERED_FRESH_CALLS = 5_880
V2112_PARENT_HOSTED_COMPLETIONS = 940
V2112_CUMULATIVE_FULL_CALLS = 6_820
V2112_REGISTERED_CALL_HEADROOM = 680
V2112_REGISTERED_SCIENCE_CELLS = 131
V2112_SCIENCE_STORAGE_RESERVATION_BYTES = 1_920_000_000

V2112_TOTAL_HARD_CAP_USD = 500.0
V2112_TOTAL_HOSTED_COMPLETION_CAP = 7_500
V2112_TOTAL_STORAGE_CAP_BYTES = 5_000_000_000

V2112_GATE_CLAIM_BOUNDARY = (
    "This zero-provider receipt authorizes normal scientific dispatch only "
    "from sealed V2.11.2 fresh long-context observed p95. Exactly 24 action "
    "and 8 semantic fresh samples per model contribute to dispatch. The "
    "direct V2.11.1 capability wrappers contribute exactly 24 action and 6 "
    "semantic samples per model to capability audit only, never to dispatch, "
    "and remain charged only through the parent debit. V2.11.1 failed "
    "preflight samples, bootstrap reservations, checkpoints, journals, and "
    "P95 authorities are forbidden. Any mixed terminal status or global cap "
    "failure is a matrix-wide no-go."
)
V2112_CAPABILITY_EVIDENCE_USE = (
    "Capability/interface gate and its original 30 historical capability "
    "samples only; no V2.11.1 failed-preflight sample, checkpoint, P95, or "
    "treatment effect."
)

V2112_SCIENCE_CALLS: Mapping[str, Mapping[str, int]] = {
    "gpt52_main": {"action": 4_560, "semantic": 920},
    "gpt56_diagnostic": {"action": 288, "semantic": 48},
}
V2112_SCIENCE_CELLS: Mapping[str, int] = {
    "gpt52_main": 125,
    "gpt56_diagnostic": 6,
}

_MODEL_PROFILES: Mapping[str, Mapping[str, Any]] = {
    "gpt52_main": {
        "runtime_model": "openai/gpt-5.2-2025-12-11",
        "requested_model": "gpt-5.2-2025-12-11",
        "served_model": "gpt-5.2-2025-12-11",
    },
    "gpt56_diagnostic": {
        "runtime_model": "openai/gpt-5.6-sol",
        "requested_model": "gpt-5.6-sol",
        "served_model": "gpt-5.6-sol",
    },
}
_MODEL_IDS = frozenset(_MODEL_PROFILES)
_MODEL_TERMINAL_STATUSES = frozenset(
    {"eligible", "capability-no-go", "interface-no-go"}
)
_ZERO_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "cost_usd": 0.0,
}
_USAGE_FIELDS = frozenset(_ZERO_USAGE)
_SAMPLE_FIELDS = frozenset(
    {
        "finish_reason",
        "response_completed",
        "output_disposition",
        "error_type",
        "parse_success",
        "clipped",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "visible_completion_tokens",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class PilotV2112GateError(ValueError):
    """Raised when V2.11.2 post-gate evidence cannot authorize dispatch."""


def canonical_sha256(value: Any) -> str:
    """Return the canonical JSON SHA-256 used by gate receipts."""

    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PilotV2112GateError("gate value is not canonical JSON") from exc
    return hashlib.sha256(raw).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotV2112GateError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PilotV2112GateError(f"{name} must be an array")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PilotV2112GateError(f"{name} must be a lowercase SHA-256")
    return value


def _commit(value: Any, name: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise PilotV2112GateError(f"{name} must be a lowercase 40-hex commit")
    return value


def _normalized_text(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise PilotV2112GateError(f"{name} must be normalized non-empty text")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PilotV2112GateError(f"{name} must be a nonnegative integer")
    return value


def _usage(
    value: Any,
    name: str,
    *,
    positive: bool = True,
) -> dict[str, int | float]:
    row = _mapping(value, name)
    if set(row) != _USAGE_FIELDS:
        raise PilotV2112GateError(f"{name} usage fields drifted")
    result: dict[str, int | float] = {}
    minimum = 1 if positive else 0
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = row[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
            raise PilotV2112GateError(f"{name}.{field} must be an integer >= {minimum}")
        result[field] = item
    if result["total_tokens"] != (
        result["prompt_tokens"] + result["completion_tokens"]
    ):
        raise PilotV2112GateError(f"{name}.total_tokens is not additive")
    cost = row["cost_usd"]
    if (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0.0
        or (positive and float(cost) <= 0.0)
    ):
        raise PilotV2112GateError(f"{name}.cost_usd is invalid")
    result["cost_usd"] = float(cost)
    return result


def _sum_usage(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
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


def _verify_self_seal(
    value: Mapping[str, Any],
    *,
    schema_version: str,
    name: str,
) -> str:
    candidate = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    integrity = candidate.get("integrity")
    if not isinstance(integrity, dict):
        raise PilotV2112GateError(f"{name} integrity is absent")
    claimed = integrity.pop("content_sha256", None)
    if (
        value.get("schema_version") != schema_version
        or set(value.get("integrity", {})) != {"canonicalization", "content_sha256"}
        or integrity.get("canonicalization") != "json-sort-keys-utf8-v1"
        or _SHA256_RE.fullmatch(str(claimed)) is None
        or canonical_sha256(candidate) != claimed
    ):
        raise PilotV2112GateError(f"{name} schema or self-hash drifted")
    return str(claimed)


def _expected_manifest_binding() -> dict[str, str]:
    expected = {
        "path": V2112_SOURCE_MANIFEST_PATH.as_posix(),
        "file_sha256": V2112_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": V2112_SOURCE_MANIFEST_CONTENT_SHA256,
    }
    _sha256(expected["file_sha256"], "frozen source manifest file SHA-256")
    _sha256(
        expected["content_sha256"],
        "frozen source manifest content SHA-256",
    )
    return expected


def _parent_debit(value: Any) -> dict[str, Any]:
    try:
        debit = ParentBudgetDebit.from_dict(_mapping(value, "parent debit"))
    except (TypeError, ValueError) as exc:
        raise PilotV2112GateError("parent debit is malformed") from exc
    expected = {
        "parent_contract_sha256": V211_PARENT_CONTRACT_SHA256,
        "parent_run_ledger_sha256": V211_PARENT_RUN_LEDGER_SHA256,
        "parent_budget_ledger_sha256": V211_PARENT_BUDGET_LEDGER_SHA256,
        "stage_bucket": "parent_v2111",
        "cost_usd": V211_PARENT_CUMULATIVE_COST_USD,
        "hosted_completions": V211_PARENT_CUMULATIVE_COMPLETIONS,
        "storage_bytes": V211_PARENT_CUMULATIVE_STORAGE_BYTES,
        "record_sha256": V211_PARENT_DEBIT_RECORD_SHA256,
    }
    observed = debit.to_dict()
    for field, expected_value in expected.items():
        value_at_field = observed.get(field)
        if isinstance(expected_value, float):
            match = math.isclose(
                float(value_at_field),
                expected_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        else:
            match = value_at_field == expected_value
        if not match:
            raise PilotV2112GateError(
                f"parent debit differs from frozen V2.11.1 at {field}"
            )
    return observed


def _parent_import_binding(value: Any) -> dict[str, str]:
    row = _mapping(value, "parent import receipt binding")
    expected_fields = {"path", "file_sha256", "content_sha256"}
    if set(row) != expected_fields:
        raise PilotV2112GateError("parent import receipt binding fields drifted")
    path = _safe_relative_receipt_path(row["path"]).as_posix()
    if not path.endswith("/parent-import/parent_import_receipt.json"):
        raise PilotV2112GateError("parent import receipt binding path drifted")
    return {
        "path": path,
        "file_sha256": _sha256(
            row["file_sha256"],
            "parent import receipt file SHA-256",
        ),
        "content_sha256": _sha256(
            row["content_sha256"],
            "parent import receipt content SHA-256",
        ),
    }


def _capability_binding(
    value: Any,
    *,
    model_id: str,
    contract_sha256: str,
    release_commit: str,
) -> dict[str, Any]:
    binding = _mapping(value, f"{model_id} inherited capability binding")
    if set(binding) != {
        "model_id",
        "wrapper_content_sha256",
        "payload",
        "provider_construction_during_verification",
        "provider_calls_during_verification",
    }:
        raise PilotV2112GateError(
            f"{model_id} inherited capability binding fields drifted"
        )
    if (
        binding.get("model_id") != model_id
        or binding.get("provider_construction_during_verification") is not False
        or binding.get("provider_calls_during_verification") != 0
    ):
        raise PilotV2112GateError(
            f"{model_id} inherited capability binding is not zero-provider"
        )
    wrapper = _mapping(binding["payload"], f"{model_id} capability wrapper")
    wrapper_hash = _verify_self_seal(
        wrapper,
        schema_version=V2112_CAPABILITY_WRAPPER_SCHEMA_VERSION,
        name=f"{model_id} capability wrapper",
    )
    expected_wrapper_fields = {
        "schema_version",
        "child_release",
        "parent_release",
        "source_manifest",
        "source_capability_wrapper",
        "capability",
        "provider_construction_current_attempt",
        "provider_calls_current_attempt",
        "hosted_provider_calls_current_attempt",
        "current_attempt_usage",
        "imported_effect_cells",
        "imported_preflight_samples",
        "imported_checkpoint_artifacts",
        "imported_p95_authorities",
        "scientific_evidence",
        "evidence_scope",
        "evidence_use",
        "integrity",
    }
    if (
        set(wrapper) != expected_wrapper_fields
        or binding.get("wrapper_content_sha256") != wrapper_hash
    ):
        raise PilotV2112GateError(
            f"{model_id} capability wrapper fields or binding hash drifted"
        )
    child = _mapping(
        wrapper.get("child_release"),
        f"{model_id} wrapper child release",
    )
    if child != {
        "contract_id": V2112_GATE_CONTRACT_ID,
        "contract_sha256": contract_sha256,
        "git_tag": V2112_GATE_RELEASE_TAG,
        "resolved_git_commit": release_commit,
    }:
        raise PilotV2112GateError(f"{model_id} capability wrapper is not child-bound")
    parent = _mapping(
        wrapper.get("parent_release"),
        f"{model_id} wrapper parent release",
    )
    if parent != {
        "contract_id": V211_PARENT_CONTRACT_ID,
        "contract_sha256": V211_PARENT_CONTRACT_SHA256,
        "git_tag": V211_PARENT_SCIENCE_TAG,
        "git_tag_object": V211_PARENT_SCIENCE_TAG_OBJECT,
        "resolved_git_commit": V211_PARENT_SCIENCE_COMMIT,
    }:
        raise PilotV2112GateError(
            f"{model_id} capability wrapper parent release drifted"
        )
    if wrapper.get("source_manifest") != _expected_manifest_binding():
        raise PilotV2112GateError(
            f"{model_id} capability wrapper source manifest drifted"
        )
    if (
        wrapper.get("provider_construction_current_attempt") is not False
        or wrapper.get("provider_calls_current_attempt") != 0
        or wrapper.get("hosted_provider_calls_current_attempt") != 0
        or wrapper.get("current_attempt_usage") != _ZERO_USAGE
        or wrapper.get("imported_effect_cells") != 0
        or wrapper.get("imported_preflight_samples") != 0
        or wrapper.get("imported_checkpoint_artifacts") != []
        or wrapper.get("imported_p95_authorities") != []
        or wrapper.get("scientific_evidence") is not False
        or wrapper.get("evidence_scope") != "preregistered_task_capability_gate"
        or wrapper.get("evidence_use") != V2112_CAPABILITY_EVIDENCE_USE
    ):
        raise PilotV2112GateError(f"{model_id} capability wrapper scope drifted")
    source = _mapping(
        wrapper.get("source_capability_wrapper"),
        f"{model_id} direct V2.11.1 capability wrapper source",
    )
    capability = _mapping(
        wrapper.get("capability"),
        f"{model_id} inherited capability",
    )
    profile = _MODEL_PROFILES[model_id]
    if (
        set(source)
        != {
            "path",
            "byte_size",
            "file_sha256",
            "content_sha256",
            "schema_version",
            "run_id",
            "run_spec",
            "historical_source_calls",
        }
        or source.get("schema_version") != V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION
        or source.get("historical_source_calls")
        != V2112_INHERITED_CAPABILITY_CALLS_PER_MODEL
        or not str(source.get("path", "")).startswith(
            "experiment_results/pilot-v2.11.1/raw/capability-gate/"
        )
        or not str(source.get("path", "")).endswith("/capability.json")
        or "long-context-preflight" in str(source.get("path", ""))
        or "bootstrap" in str(source.get("path", ""))
        or "journal" in str(source.get("path", ""))
        or source.get("run_id")
        != (
            "finevo-pilot-v2.11.1--capability-gate--"
            f"{model_id}--capability-probe--none--"
            "provider-preflight-default--s2010922376"
        )
        or not isinstance(source.get("run_spec"), Mapping)
        or source["run_spec"].get("contract_id") != V211_PARENT_CONTRACT_ID
        or source["run_spec"].get("run_id") != source.get("run_id")
        or capability.get("model_id") != model_id
        or capability.get("runtime_model") != profile["runtime_model"]
        or capability.get("requested_model") != profile["requested_model"]
        or capability.get("served_model") != profile["served_model"]
        # The V2.11.1 wrapper is the direct imported authority, while its
        # normalized capability payload deliberately retains the originating
        # V2.11 capability cell identity.  Bind both layers instead of
        # laundering the nested payload into the wrapper's newer run ID.
        or capability.get("run_id")
        != (
            "finevo-pilot-v2.11--capability-gate--"
            f"{model_id}--capability-probe--none--"
            "provider-preflight-default--s2010922376"
        )
        or capability.get("historical_source_calls")
        != V2112_INHERITED_CAPABILITY_CALLS_PER_MODEL
        or capability.get("action_sample_count") != 24
        or capability.get("semantic_sample_count") != 6
        or capability.get("capability_pass") is not True
        or capability.get("interface_pass") is not True
    ):
        raise PilotV2112GateError(
            f"{model_id} inherited capability denominator drifted"
        )
    samples_raw = _mapping(
        capability.get("samples"),
        f"{model_id} capability samples",
    )
    if set(samples_raw) != {"action", "semantic"}:
        raise PilotV2112GateError(f"{model_id} capability sample kinds drifted")
    samples: dict[str, list[dict[str, Any]]] = {}
    expected_counts = {"action": 24, "semantic": 6}
    for call_kind in ("action", "semantic"):
        rows = _sequence(
            samples_raw[call_kind],
            f"{model_id} capability {call_kind} samples",
        )
        if len(rows) != expected_counts[call_kind]:
            raise PilotV2112GateError(
                f"{model_id} capability {call_kind} count drifted"
            )
        normalized_rows: list[dict[str, Any]] = []
        for index, raw_row in enumerate(rows):
            row = _mapping(
                raw_row,
                f"{model_id} capability {call_kind} sample {index}",
            )
            if set(row) != _SAMPLE_FIELDS:
                raise PilotV2112GateError(
                    f"{model_id} capability sample fields drifted"
                )
            for field in (
                "parse_success",
                "clipped",
                "response_completed",
            ):
                if not isinstance(row[field], bool):
                    raise PilotV2112GateError(
                        f"{model_id} capability sample flag drifted"
                    )
            for field in (
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "visible_completion_tokens",
            ):
                _nonnegative_int(
                    row[field],
                    f"{model_id} capability sample {field}",
                )
            normalized_rows.append(dict(row))
        samples[call_kind] = normalized_rows
    usage_rows_raw = _sequence(
        capability.get("usage_rows"),
        f"{model_id} capability usage rows",
    )
    if len(usage_rows_raw) != 30:
        raise PilotV2112GateError(f"{model_id} capability usage denominator drifted")
    usage_rows: list[dict[str, Any]] = []
    grouped_usage: dict[str, list[dict[str, int | float]]] = {
        "action": [],
        "semantic": [],
    }
    for index, raw_row in enumerate(usage_rows_raw):
        row = _mapping(
            raw_row,
            f"{model_id} capability usage row {index}",
        )
        if set(row) != {"response_model", "call_kind", "usage"}:
            raise PilotV2112GateError(f"{model_id} capability usage row fields drifted")
        call_kind = row.get("call_kind")
        if (
            call_kind not in {"action", "semantic"}
            or row.get("response_model") != profile["served_model"]
        ):
            raise PilotV2112GateError(f"{model_id} capability usage route drifted")
        normalized_usage = _usage(
            row.get("usage"),
            f"{model_id} capability usage row {index}",
        )
        grouped_usage[str(call_kind)].append(normalized_usage)
        usage_rows.append(
            {
                "response_model": profile["served_model"],
                "call_kind": call_kind,
                "usage": normalized_usage,
            }
        )
    if {key: len(rows) for key, rows in grouped_usage.items()} != expected_counts:
        raise PilotV2112GateError(
            f"{model_id} capability usage call-kind counts drifted"
        )
    for call_kind in ("action", "semantic"):
        for sample, usage in zip(
            samples[call_kind],
            grouped_usage[call_kind],
            strict=True,
        ):
            if (
                sample["prompt_tokens"] != usage["prompt_tokens"]
                or sample["completion_tokens"] != usage["completion_tokens"]
            ):
                raise PilotV2112GateError(
                    f"{model_id} capability sample/usage tokens differ"
                )
    actual = _usage(
        capability.get("actual_usage"),
        f"{model_id} capability actual usage",
    )
    if not _usage_equal(
        actual,
        _sum_usage([row["usage"] for row in usage_rows]),
    ):
        raise PilotV2112GateError(f"{model_id} capability actual usage is not additive")
    for field in ("file_sha256", "content_sha256"):
        _sha256(source.get(field), f"{model_id} source wrapper {field}")
    _nonnegative_int(
        source.get("byte_size"),
        f"{model_id} source wrapper byte_size",
    )
    return {
        "wrapper_content_sha256": wrapper_hash,
        "run_id": str(source["run_id"]),
        "source_run_spec_sha256": canonical_sha256(source["run_spec"]),
        "source_wrapper_file_sha256": str(source["file_sha256"]),
        "source_wrapper_content_sha256": str(source["content_sha256"]),
        "samples": samples,
        "usage_rows": usage_rows,
        "actual_usage": actual,
        "capability_pass": True,
        "interface_pass": True,
    }


def _provider_call_key(value: Mapping[str, Any], name: str) -> tuple[Any, ...]:
    call_kind = value.get("call_kind")
    decision_t = value.get("decision_t")
    agent_id = value.get("agent_id")
    if (
        call_kind not in {"action", "semantic"}
        or isinstance(decision_t, bool)
        or not isinstance(decision_t, int)
        or isinstance(agent_id, bool)
        or not isinstance(agent_id, int)
    ):
        raise PilotV2112GateError(f"{name} call identity is malformed")
    return (
        decision_t,
        agent_id,
        call_kind,
        _sha256(value.get("prompt_hash"), f"{name}.prompt_hash"),
        _sha256(value.get("raw_output_hash"), f"{name}.raw_output_hash"),
    )


def _verify_fresh_journal(
    value: Any,
    *,
    artifact_sha256: Any,
    model_id: str,
    checkpoint_run_id: str,
    contract_sha256: str,
    provider_rows: Any,
) -> dict[str, str]:
    journal = _mapping(value, f"{model_id} fresh provider journal")
    if set(journal) != {
        "schema_version",
        "run_id",
        "contract_hash",
        "events",
        "journal_sha256",
    }:
        raise PilotV2112GateError(f"{model_id} fresh provider journal fields drifted")
    if (
        journal.get("schema_version") != PROVIDER_CALL_JOURNAL_SCHEMA_VERSION
        or journal.get("run_id") != checkpoint_run_id
        or journal.get("contract_hash") != contract_sha256
        or canonical_sha256(journal)
        != _sha256(
            artifact_sha256,
            f"{model_id} provider journal artifact SHA-256",
        )
    ):
        raise PilotV2112GateError(
            f"{model_id} provider journal is not fresh V2.11.2 evidence"
        )
    try:
        verified_journal = verify_provider_call_journal(
            journal,
            expected_run_id=checkpoint_run_id,
            expected_contract_hash=contract_sha256,
            require_terminal_dispositions=True,
        )
    except VerifiedRunError as exc:
        raise PilotV2112GateError(
            f"{model_id} provider journal failed terminal verification: {exc}"
        ) from exc
    if verified_journal != journal:
        raise PilotV2112GateError(
            f"{model_id} provider journal canonical roundtrip drifted"
        )
    journal_body = dict(journal)
    journal_sha256 = journal_body.pop("journal_sha256", None)
    if (
        _SHA256_RE.fullmatch(str(journal_sha256)) is None
        or canonical_sha256(journal_body) != journal_sha256
    ):
        raise PilotV2112GateError(f"{model_id} provider journal self-hash mismatch")
    events = _sequence(journal.get("events"), f"{model_id} journal events")
    if len(events) != 64:
        raise PilotV2112GateError(
            f"{model_id} provider journal must retain exactly 64 events"
        )
    previous = "0" * 64
    completion: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    disposition: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for index, raw_event in enumerate(events):
        event = _mapping(raw_event, f"{model_id} journal event {index}")
        if set(event) != {
            "event_index",
            "event_type",
            "previous_event_sha256",
            "payload",
            "event_sha256",
        }:
            raise PilotV2112GateError(f"{model_id} journal event fields drifted")
        event_body = dict(event)
        event_sha256 = event_body.pop("event_sha256", None)
        if (
            event.get("event_index") != index
            or event.get("previous_event_sha256") != previous
            or _SHA256_RE.fullmatch(str(event_sha256)) is None
            or canonical_sha256(event_body) != event_sha256
        ):
            raise PilotV2112GateError(f"{model_id} provider journal hash chain drifted")
        payload = _mapping(
            event.get("payload"),
            f"{model_id} journal event {index} payload",
        )
        key = _provider_call_key(
            payload,
            f"{model_id} journal event {index}",
        )
        event_type = event.get("event_type")
        target = (
            completion
            if event_type == "completion_received"
            else disposition if event_type == "parse_disposition" else None
        )
        if target is None or key in target:
            raise PilotV2112GateError(
                f"{model_id} provider journal event denominator drifted"
            )
        target[key] = payload
        previous = str(event_sha256)
    rows = _sequence(provider_rows, f"{model_id} checkpoint provider rows")
    if len(rows) != 32 or len(completion) != 32 or len(disposition) != 32:
        raise PilotV2112GateError(
            f"{model_id} journal/checkpoint call denominator drifted"
        )
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"{model_id} checkpoint provider row {index}")
        key = _provider_call_key(
            row,
            f"{model_id} checkpoint provider row {index}",
        )
        completion_payload = completion.get(key)
        disposition_payload = disposition.get(key)
        parse = _mapping(
            row.get("parse_disposition"),
            f"{model_id} checkpoint parse disposition {index}",
        )
        if (
            completion_payload is None
            or disposition_payload is None
            or any(
                row.get(field) != field_value
                for field, field_value in completion_payload.items()
            )
            or not _usage_equal(
                _usage(
                    completion_payload.get("usage"),
                    f"{model_id} journal completion usage {index}",
                ),
                _usage(
                    row.get("usage"),
                    f"{model_id} checkpoint provider usage {index}",
                ),
            )
            or disposition_payload.get("parse_status") != parse.get("parse_status")
            or disposition_payload.get("parse_mode") != parse.get("parse_mode")
            or disposition_payload.get("accepted") is not parse.get("accepted")
        ):
            raise PilotV2112GateError(
                f"{model_id} provider journal/checkpoint row binding drifted"
            )
    return {
        "artifact_sha256": str(artifact_sha256),
        "journal_sha256": str(journal_sha256),
    }


def _fresh_preflight(
    value: Any,
    *,
    model_id: str,
    contract_sha256: str,
    release_commit: str,
) -> dict[str, Any]:
    envelope = _mapping(value, f"{model_id} fresh preflight artifact")
    expected_fields = {
        "ledger_run_id",
        "checkpoint_run_id",
        "run_spec_sha256",
        "checkpoint_artifact_sha256",
        "checkpoint",
        "exactness_artifact_sha256",
        "exactness",
        "provider_call_journal_artifact_sha256",
        "provider_call_journal",
    }
    if set(envelope) != expected_fields:
        raise PilotV2112GateError(
            f"{model_id} fresh preflight artifact fields drifted; historical "
            "bootstrap, checkpoint, journal, and P95 injection is forbidden"
        )
    ledger_run_id = (
        "finevo-pilot-v2.11.2--long-context-preflight--"
        f"{model_id}--closed-loop-preflight--none--stage0-selected--"
        "s2010922376"
    )
    checkpoint_run_id = ledger_run_id + V211_PREFLIGHT_CHECKPOINT_RUN_SUFFIX
    if (
        envelope.get("ledger_run_id") != ledger_run_id
        or envelope.get("checkpoint_run_id") != checkpoint_run_id
    ):
        raise PilotV2112GateError(
            f"{model_id} fresh preflight run IDs are not exact V2.11.2 IDs"
        )
    checkpoint = _mapping(
        envelope.get("checkpoint"),
        f"{model_id} fresh checkpoint",
    )
    if canonical_sha256(checkpoint) != _sha256(
        envelope.get("checkpoint_artifact_sha256"),
        f"{model_id} checkpoint artifact SHA-256",
    ):
        raise PilotV2112GateError(f"{model_id} fresh checkpoint artifact hash mismatch")
    run_config = _mapping(
        checkpoint.get("run_config"),
        f"{model_id} fresh checkpoint run config",
    )
    if (
        run_config.get("run_id") != checkpoint_run_id
        or run_config.get("pilot_contract_hash") != contract_sha256
        or run_config.get("pilot_tag") != V2112_GATE_RELEASE_TAG
        or run_config.get("seed") != 2010922376
        or run_config.get("preflight_p95_reservations") not in (None, {})
    ):
        raise PilotV2112GateError(
            f"{model_id} checkpoint is not fresh V2.11.2 preflight state"
        )
    bootstrap = _mapping(
        run_config.get("contract_bootstrap_reservations"),
        f"{model_id} V2.11.2 bootstrap reservations",
    )
    runtime_model = str(_MODEL_PROFILES[model_id]["runtime_model"])
    by_kind = _mapping(
        bootstrap.get(runtime_model),
        f"{model_id} V2.11.2 bootstrap model binding",
    )
    if set(bootstrap) != {runtime_model} or set(by_kind) != {
        "action",
        "semantic",
    }:
        raise PilotV2112GateError(
            f"{model_id} bootstrap model/call-kind denominator drifted"
        )
    parsed_bootstrap: dict[str, ContractEnvelopeBootstrapReservation] = {}
    for call_kind in ("action", "semantic"):
        try:
            reservation = ContractEnvelopeBootstrapReservation.from_dict(
                model=runtime_model,
                call_kind=call_kind,
                value=by_kind[call_kind],
            )
        except (TypeError, ValueError) as exc:
            raise PilotV2112GateError(
                f"{model_id} {call_kind} V2.11.2 bootstrap is invalid: {exc}"
            ) from exc
        if (
            reservation.authority_id != V2112_CONTRACT_ENVELOPE_AUTHORITY_ID
            or reservation.target_contract_id != V2112_GATE_CONTRACT_ID
            or reservation.pilot_contract_hash != contract_sha256
            or reservation.pilot_tag != V2112_GATE_RELEASE_TAG
            or reservation.authorized_run_id != checkpoint_run_id
            or reservation.source_contract_id != V2112_SOURCE_CONTRACT_ID
            or reservation.source_tag != V2112_SOURCE_RELEASE_TAG
            or reservation.pilot_commit != release_commit
            or reservation.source_contract_hash != V211_PARENT_CONTRACT_SHA256
            or reservation.source_commit != V211_PARENT_SCIENCE_COMMIT
            or reservation.source_run_id
            != (
                "finevo-pilot-v2.11.1--capability-gate--"
                f"{model_id}--capability-probe--none--"
                "provider-preflight-default--s2010922376"
            )
            or reservation.target_run_spec_sha256 != envelope.get("run_spec_sha256")
        ):
            raise PilotV2112GateError(f"{model_id} bootstrap release lineage drifted")
        parsed_bootstrap[call_kind] = reservation
    if (
        parsed_bootstrap["action"].authority_binding
        != parsed_bootstrap["semantic"].authority_binding
    ):
        raise PilotV2112GateError(
            f"{model_id} bootstrap authority differs by call kind"
        )
    exactness = _mapping(
        envelope.get("exactness"),
        f"{model_id} fresh exactness receipt",
    )
    if exactness.get(
        "schema_version"
    ) != V2112_PREFLIGHT_EXACTNESS_SCHEMA_VERSION or canonical_sha256(
        exactness
    ) != _sha256(
        envelope.get("exactness_artifact_sha256"),
        f"{model_id} exactness artifact SHA-256",
    ):
        raise PilotV2112GateError(
            f"{model_id} exactness receipt is not fresh V2.11.2 evidence"
        )
    journal_binding = _mapping(
        checkpoint.get("provider_call_journal_binding"),
        f"{model_id} checkpoint journal binding",
    )
    journal = _verify_fresh_journal(
        envelope.get("provider_call_journal"),
        artifact_sha256=envelope.get("provider_call_journal_artifact_sha256"),
        model_id=model_id,
        checkpoint_run_id=checkpoint_run_id,
        contract_sha256=contract_sha256,
        provider_rows=checkpoint.get("provider_calls"),
    )
    if (
        journal_binding.get("journal_sha256") != journal["journal_sha256"]
        or journal_binding.get("run_id") != checkpoint_run_id
        or journal_binding.get("contract_hash") != contract_sha256
    ):
        raise PilotV2112GateError(f"{model_id} checkpoint/journal binding drifted")

    # The V2.11.2 exactness receipt deliberately changes only its release
    # schema.  Rebind a private copy to the already-audited V2.11 verifier so
    # every unchanged checkpoint component is still checked by the original
    # fail-closed implementation.
    compatible = json.loads(json.dumps(envelope, sort_keys=True))
    compatible.pop("provider_call_journal_artifact_sha256")
    compatible.pop("provider_call_journal")
    compatible_exactness = compatible["exactness"]
    compatible_exactness.pop("receipt_hash", None)
    compatible_exactness["schema_version"] = (
        "finevo-v2.11-long-context-preflight-exactness-receipt-v1"
    )
    compatible_exactness["receipt_hash"] = canonical_sha256(compatible_exactness)
    compatible["exactness_artifact_sha256"] = canonical_sha256(compatible_exactness)
    try:
        parsed = _preflight_envelope(
            compatible,
            model_id=model_id,
            contract_sha256=contract_sha256,
        )
    except PilotV211GateError as exc:
        raise PilotV2112GateError(
            f"{model_id} fresh preflight is invalid: {exc}"
        ) from exc
    if (
        len(parsed["samples"]["action"]) != 24
        or len(parsed["samples"]["semantic"]) != 8
    ):
        raise PilotV2112GateError(f"{model_id} fresh preflight denominator drifted")
    exactness_body = dict(exactness)
    exactness_content_sha256 = exactness_body.pop("receipt_hash", None)
    if (
        _SHA256_RE.fullmatch(str(exactness_content_sha256)) is None
        or canonical_sha256(exactness_body) != exactness_content_sha256
    ):
        raise PilotV2112GateError(f"{model_id} V2.11.2 exactness self-hash mismatch")
    parsed.update(
        {
            "exactness_artifact_sha256": str(envelope["exactness_artifact_sha256"]),
            "exactness_content_sha256": str(exactness_content_sha256),
            "exactness_schema_version": (V2112_PREFLIGHT_EXACTNESS_SCHEMA_VERSION),
            "provider_call_journal_sha256": journal["journal_sha256"],
            "provider_call_journal_artifact_sha256": journal["artifact_sha256"],
            "bootstrap_source_run_id": parsed_bootstrap["action"].source_run_id,
            "bootstrap_source_run_spec_sha256": parsed_bootstrap[
                "action"
            ].source_run_spec_sha256,
            "bootstrap_source_capability_file_sha256": parsed_bootstrap[
                "action"
            ].source_capability_file_sha256,
            "bootstrap_source_capability_payload_sha256": parsed_bootstrap[
                "action"
            ].source_capability_payload_sha256,
        }
    )
    return parsed


def _science_run_ids(
    value: Any,
) -> tuple[dict[str, list[str]], list[str]]:
    by_model = _mapping(value, "science run IDs by model")
    if set(by_model) != _MODEL_IDS:
        raise PilotV2112GateError("science run IDs must contain both registered models")
    normalized: dict[str, list[str]] = {}
    all_ids: list[str] = []
    for model_id in sorted(_MODEL_IDS):
        rows = _sequence(
            by_model[model_id],
            f"{model_id} science run IDs",
        )
        if len(rows) != V2112_SCIENCE_CELLS[model_id]:
            raise PilotV2112GateError(f"{model_id} science cell denominator drifted")
        model_rows = [
            _normalized_text(row, f"{model_id} science run ID") for row in rows
        ]
        if len(model_rows) != len(set(model_rows)):
            raise PilotV2112GateError(f"{model_id} science run IDs contain duplicates")
        normalized[model_id] = model_rows
        all_ids.extend(model_rows)
    if len(all_ids) != V2112_REGISTERED_SCIENCE_CELLS or len(all_ids) != len(
        set(all_ids)
    ):
        raise PilotV2112GateError("science run IDs must bind exactly 131 unique cells")
    return normalized, all_ids


def _model_status(*, capability_pass: bool, interface_pass: bool) -> str:
    if not capability_pass:
        return "capability-no-go"
    if not interface_pass:
        return "interface-no-go"
    return "eligible"


def _projection(
    *,
    parent_debit: Mapping[str, Any],
    fresh_preflight_usage: Mapping[str, Any],
    current_attempt_pre_science_storage_bytes: int,
    observed_reservations: Mapping[str, Mapping[str, Any]],
    eligible_model_ids: Sequence[str],
) -> dict[str, Any]:
    parent = _parent_debit(parent_debit)
    actual = _usage(
        fresh_preflight_usage,
        "fresh preflight actual usage",
    )
    storage = _nonnegative_int(
        current_attempt_pre_science_storage_bytes,
        "current-attempt pre-science storage bytes",
    )
    if set(observed_reservations) != _MODEL_IDS:
        raise PilotV2112GateError(
            "projection reservations must contain both registered models"
        )
    science_usage_rows: list[dict[str, int | float]] = []
    per_model: dict[str, dict[str, Any]] = {}
    for model_id in sorted(_MODEL_IDS):
        by_kind = _mapping(
            observed_reservations[model_id],
            f"{model_id} observed reservations",
        )
        if set(by_kind) != {"action", "semantic"}:
            raise PilotV2112GateError(f"{model_id} observed reservation kinds drifted")
        model_usage = dict(_ZERO_USAGE)
        for call_kind in ("action", "semantic"):
            try:
                reservation = PreflightP95Reservation.from_dict(
                    model=str(_MODEL_PROFILES[model_id]["runtime_model"]),
                    call_kind=call_kind,
                    value=by_kind[call_kind],
                )
            except (TypeError, ValueError) as exc:
                raise PilotV2112GateError(
                    f"{model_id} {call_kind} reservation is invalid"
                ) from exc
            expected_samples = (
                V2112_ACTION_SAMPLES_PER_MODEL
                if call_kind == "action"
                else V2112_SEMANTIC_SAMPLES_PER_MODEL
            )
            if reservation.sample_count != expected_samples:
                raise PilotV2112GateError(
                    f"{model_id} {call_kind} p95 sample count drifted"
                )
            calls = V2112_SCIENCE_CALLS[model_id][call_kind]
            row = {
                "prompt_tokens": (calls * reservation.reserved_usage.prompt_tokens),
                "completion_tokens": (
                    calls * reservation.reserved_usage.completion_tokens
                ),
                "total_tokens": (calls * reservation.reserved_usage.total_tokens),
                "cost_usd": (calls * reservation.reserved_usage.cost_usd),
            }
            science_usage_rows.append(row)
            model_usage = _sum_usage([model_usage, row])
        per_model[model_id] = {
            "registered_science_calls": sum(V2112_SCIENCE_CALLS[model_id].values()),
            "eligible_for_dispatch": model_id in eligible_model_ids,
            "reserved_science_usage": model_usage,
        }
    science = _sum_usage(science_usage_rows)
    fresh_full = _sum_usage([actual, science])
    cumulative = _sum_usage(
        [
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": float(parent["cost_usd"]),
            },
            fresh_full,
        ]
    )
    cumulative_storage = (
        int(parent["storage_bytes"]) + storage + V2112_SCIENCE_STORAGE_RESERVATION_BYTES
    )
    reasons: list[str] = []
    if float(cumulative["cost_usd"]) > V2112_TOTAL_HARD_CAP_USD + 1e-12:
        reasons.append("cumulative-cost-exceeds-hard-cap")
    if V2112_CUMULATIVE_FULL_CALLS > V2112_TOTAL_HOSTED_COMPLETION_CAP:
        reasons.append("cumulative-hosted-completions-exceed-hard-cap")
    if cumulative_storage > V2112_TOTAL_STORAGE_CAP_BYTES:
        reasons.append("cumulative-storage-exceeds-hard-cap")
    return {
        "schema_version": "finevo-pilot-v2.11.2-full-matrix-projection-v1",
        "parent_debit": parent,
        "fresh_preflight": {
            "hosted_completions": V2112_FRESH_PREFLIGHT_CALLS,
            "usage": actual,
        },
        "current_attempt_pre_science": {
            "storage_bytes": storage,
        },
        "remaining_science": {
            "registered_cells": V2112_REGISTERED_SCIENCE_CELLS,
            "registered_hosted_completions": (V2112_REGISTERED_SCIENCE_CALLS),
            "reserved_usage": science,
            "storage_reservation_bytes": (V2112_SCIENCE_STORAGE_RESERVATION_BYTES),
            "by_model": per_model,
        },
        "fresh_full_matrix": {
            "registered_hosted_completions": V2112_REGISTERED_FRESH_CALLS,
            "projected_usage": fresh_full,
            "projected_storage_bytes": (
                storage + V2112_SCIENCE_STORAGE_RESERVATION_BYTES
            ),
        },
        "cumulative_full_matrix": {
            "hosted_completions": V2112_CUMULATIVE_FULL_CALLS,
            "remaining_hosted_completion_headroom": (V2112_REGISTERED_CALL_HEADROOM),
            "projected_cost_usd": cumulative["cost_usd"],
            "projected_storage_bytes": cumulative_storage,
        },
        "caps": {
            "cost_usd": V2112_TOTAL_HARD_CAP_USD,
            "hosted_completions": V2112_TOTAL_HOSTED_COMPLETION_CAP,
            "storage_bytes": V2112_TOTAL_STORAGE_CAP_BYTES,
        },
        "go": not reasons,
        "reasons": reasons,
    }


def build_v2112_post_gate_authority(
    *,
    contract_sha256: str,
    release_tag: str,
    release_commit: str,
    parent_import_receipt_binding: Mapping[str, Any],
    parent_budget_debit: Mapping[str, Any],
    inherited_capability_bindings: Mapping[str, Mapping[str, Any]],
    fresh_preflight_artifacts: Mapping[str, Mapping[str, Any]],
    model_terminal_statuses: Mapping[str, str],
    current_attempt_pre_science_storage_bytes: int,
    ledger_event_chain_head: str,
    science_run_ids_by_model: Mapping[str, Sequence[str]],
    source_manifest_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Build one self-hashed, zero-provider V2.11.2 post-gate receipt."""

    contract_hash = _sha256(contract_sha256, "contract SHA-256")
    if release_tag != V2112_GATE_RELEASE_TAG:
        raise PilotV2112GateError("release tag differs from V2.11.2")
    commit = _commit(release_commit, "release commit")
    ledger_head = _sha256(
        ledger_event_chain_head,
        "ledger event-chain head",
    )
    parent_binding = _parent_import_binding(parent_import_receipt_binding)
    parent = _parent_debit(parent_budget_debit)
    expected_manifest = _expected_manifest_binding()
    supplied_manifest = _mapping(
        source_manifest_hashes,
        "source manifest hashes",
    )
    if supplied_manifest != {
        "file_sha256": expected_manifest["file_sha256"],
        "content_sha256": expected_manifest["content_sha256"],
    }:
        raise PilotV2112GateError(
            "source manifest hashes differ from frozen V2.11.2 manifest"
        )
    if (
        not isinstance(inherited_capability_bindings, Mapping)
        or set(inherited_capability_bindings) != _MODEL_IDS
        or not isinstance(fresh_preflight_artifacts, Mapping)
        or set(fresh_preflight_artifacts) != _MODEL_IDS
        or not isinstance(model_terminal_statuses, Mapping)
        or set(model_terminal_statuses) != _MODEL_IDS
    ):
        raise PilotV2112GateError(
            "gate inputs must contain exactly both registered models"
        )
    science_by_model, all_science_ids = _science_run_ids(science_run_ids_by_model)

    inherited: dict[str, dict[str, Any]] = {}
    fresh: dict[str, dict[str, Any]] = {}
    observed_reservations: dict[str, dict[str, Any]] = {}
    dispatch_reservations: dict[str, dict[str, Any]] = {}
    model_decisions: dict[str, dict[str, Any]] = {}
    authority_sources: dict[str, dict[str, str]] = {}
    artifact_bindings: dict[str, dict[str, Any]] = {}
    inherited_actual_rows: list[Mapping[str, Any]] = []
    fresh_actual_rows: list[Mapping[str, Any]] = []
    locally_eligible: list[str] = []

    for model_id in sorted(_MODEL_IDS):
        inherited[model_id] = _capability_binding(
            inherited_capability_bindings[model_id],
            model_id=model_id,
            contract_sha256=contract_hash,
            release_commit=commit,
        )
        fresh[model_id] = _fresh_preflight(
            fresh_preflight_artifacts[model_id],
            model_id=model_id,
            contract_sha256=contract_hash,
            release_commit=commit,
        )
        if (
            fresh[model_id]["bootstrap_source_run_id"] != inherited[model_id]["run_id"]
            or fresh[model_id]["bootstrap_source_run_spec_sha256"]
            != inherited[model_id]["source_run_spec_sha256"]
            or fresh[model_id]["bootstrap_source_capability_file_sha256"]
            != inherited[model_id]["source_wrapper_file_sha256"]
        ):
            raise PilotV2112GateError(
                f"{model_id} fresh bootstrap does not bind the inherited "
                "V2.11.1 capability source"
            )
        dispatch_samples = fresh[model_id]["samples"]
        projection_rows = preflight_p95(
            fresh[model_id]["usage_rows"],
            reserve_multiplier=V2112_GATE_RESERVE_MULTIPLIER,
        )
        served_model = str(_MODEL_PROFILES[model_id]["served_model"])
        if set(projection_rows) != {
            f"{served_model}::action",
            f"{served_model}::semantic",
        }:
            raise PilotV2112GateError(f"{model_id} observed p95 grouping drifted")
        by_kind = {
            call_kind: projection_rows[f"{served_model}::{call_kind}"]
            for call_kind in ("action", "semantic")
        }
        gates: dict[str, dict[str, Any]] = {}
        for call_kind, expected_count in (
            ("action", V2112_ACTION_SAMPLES_PER_MODEL),
            ("semantic", V2112_SEMANTIC_SAMPLES_PER_MODEL),
        ):
            if (
                len(dispatch_samples[call_kind]) != expected_count
                or by_kind[call_kind].get("sample_count") != expected_count
            ):
                raise PilotV2112GateError(
                    f"{model_id} {call_kind} evidence denominator drifted"
                )
            try:
                gate = interface_sample_gate(
                    call_kind=call_kind,
                    wire_cap_tokens=V2112_GATE_WIRE_CAP_TOKENS,
                    reservation=by_kind[call_kind],
                    samples=dispatch_samples[call_kind],
                    expected_sample_count=expected_count,
                    prompt_tier_ceiling_tokens=(V2112_GATE_PROMPT_TIER_CEILING),
                    minimum_headroom_fraction=0.25,
                )
            except PilotInterfaceGateError as exc:
                raise PilotV2112GateError(
                    f"{model_id} {call_kind} interface gate is malformed: " f"{exc}"
                ) from exc
            gates[call_kind] = gate.to_dict()
        interface_pass = all(row["passed"] is True for row in gates.values())
        status = _model_status(
            capability_pass=bool(inherited[model_id]["capability_pass"]),
            interface_pass=interface_pass,
        )
        supplied_status = model_terminal_statuses[model_id]
        if supplied_status not in _MODEL_TERMINAL_STATUSES or supplied_status != status:
            raise PilotV2112GateError(
                f"{model_id} terminal status differs from recomputed gates"
            )
        model_eligible = status == "eligible"
        if model_eligible:
            locally_eligible.append(model_id)
        observed_reservations[model_id] = by_kind
        inherited_actual_rows.append(inherited[model_id]["actual_usage"])
        fresh_actual_rows.append(fresh[model_id]["actual_usage"])
        model_decisions[model_id] = {
            "terminal_status": status,
            "locally_eligible_for_science_dispatch": model_eligible,
            "eligible_for_science_dispatch": False,
            "capability_pass": inherited[model_id]["capability_pass"],
            "interface_pass": interface_pass,
            "sample_counts": {
                "action": V2112_ACTION_SAMPLES_PER_MODEL,
                "semantic": V2112_SEMANTIC_SAMPLES_PER_MODEL,
            },
            "sample_hashes": {
                call_kind: canonical_sha256(dispatch_samples[call_kind])
                for call_kind in ("action", "semantic")
            },
            "interface_gates": gates,
            "no_go_science_cell_count": (V2112_SCIENCE_CELLS[model_id]),
        }
        artifact_bindings[model_id] = {
            "inherited_capability": {
                "run_id": inherited[model_id]["run_id"],
                "wrapper_content_sha256": inherited[model_id]["wrapper_content_sha256"],
                "schema_version": (V2112_CAPABILITY_WRAPPER_SCHEMA_VERSION),
                "historical_source_calls": (V2112_INHERITED_CAPABILITY_CALLS_PER_MODEL),
            },
            "fresh_preflight": {
                key: fresh[model_id][key]
                for key in (
                    "ledger_run_id",
                    "checkpoint_run_id",
                    "run_spec_sha256",
                    "checkpoint_artifact_sha256",
                    "checkpoint_content_sha256",
                    "checkpoint_schema_version",
                    "exactness_artifact_sha256",
                    "exactness_content_sha256",
                    "exactness_schema_version",
                    "provider_call_journal_artifact_sha256",
                    "provider_call_journal_sha256",
                )
            },
        }
        authority_sources[model_id] = {
            "source_preflight_run_id": fresh[model_id]["ledger_run_id"],
            "source_preflight_run_spec_sha256": fresh[model_id]["run_spec_sha256"],
            "source_model_id": model_id,
            "source_served_model": str(_MODEL_PROFILES[model_id]["served_model"]),
            "source_execution_artifact_sha256": fresh[model_id][
                "checkpoint_artifact_sha256"
            ],
            "source_provider_call_journal_sha256": fresh[model_id][
                "provider_call_journal_sha256"
            ],
            "source_capability_wrapper_content_sha256": inherited[model_id][
                "wrapper_content_sha256"
            ],
        }

    inherited_actual = _sum_usage(inherited_actual_rows)
    fresh_actual = _sum_usage(fresh_actual_rows)
    evidence_actual = _sum_usage([inherited_actual, fresh_actual])
    global_eligible = set(locally_eligible) == _MODEL_IDS
    eligible = sorted(_MODEL_IDS) if global_eligible else []
    if global_eligible:
        dispatch_reservations = {
            model_id: observed_reservations[model_id] for model_id in sorted(_MODEL_IDS)
        }
        for model_id in sorted(_MODEL_IDS):
            model_decisions[model_id]["eligible_for_science_dispatch"] = True
            model_decisions[model_id]["no_go_science_cell_count"] = 0
    projection = _projection(
        parent_debit=parent,
        fresh_preflight_usage=fresh_actual,
        current_attempt_pre_science_storage_bytes=(
            current_attempt_pre_science_storage_bytes
        ),
        observed_reservations=observed_reservations,
        eligible_model_ids=eligible,
    )
    reasons = list(projection["reasons"])
    if locally_eligible and not global_eligible:
        reasons.append("mixed-model-terminal-status-global-no-go")
    elif not locally_eligible:
        reasons.append("no-dispatch-eligible-models")
    if reasons:
        eligible = []
        dispatch_reservations = {}
        for model_id in sorted(_MODEL_IDS):
            model_decisions[model_id]["eligible_for_science_dispatch"] = False
            model_decisions[model_id]["no_go_science_cell_count"] = V2112_SCIENCE_CELLS[
                model_id
            ]
        projection = _projection(
            parent_debit=parent,
            fresh_preflight_usage=fresh_actual,
            current_attempt_pre_science_storage_bytes=(
                current_attempt_pre_science_storage_bytes
            ),
            observed_reservations=observed_reservations,
            eligible_model_ids=eligible,
        )
    receipt: dict[str, Any] = {
        "schema_version": V2112_GATE_SCHEMA_VERSION,
        "contract_id": V2112_GATE_CONTRACT_ID,
        "contract_sha256": contract_hash,
        "release": {
            "tag": V2112_GATE_RELEASE_TAG,
            "commit": commit,
        },
        "bindings": {
            "parent_import_receipt": parent_binding,
            "source_manifest": expected_manifest,
            "ledger_event_chain_head": ledger_head,
            "gate_artifacts": artifact_bindings,
        },
        "denominator": {
            "registered_science_cells": V2112_REGISTERED_SCIENCE_CELLS,
            "science_run_ids_by_model": science_by_model,
            "science_run_ids_sha256": canonical_sha256(all_science_ids),
            "inherited_capability_evidence_calls": (V2112_INHERITED_CAPABILITY_CALLS),
            "fresh_preflight_calls": V2112_FRESH_PREFLIGHT_CALLS,
            "gate_evidence_calls": V2112_GATE_EVIDENCE_CALLS,
            "registered_remaining_science_calls": (V2112_REGISTERED_SCIENCE_CALLS),
            "registered_fresh_full_matrix_calls": (V2112_REGISTERED_FRESH_CALLS),
            "parent_debit_calls": V2112_PARENT_HOSTED_COMPLETIONS,
            "cumulative_full_matrix_calls": V2112_CUMULATIVE_FULL_CALLS,
            "registered_call_headroom": V2112_REGISTERED_CALL_HEADROOM,
            "locally_eligible_model_ids": sorted(locally_eligible),
            "eligible_model_ids": sorted(eligible),
            "no_go_science_run_ids_by_model": {
                model_id: ([] if model_id in eligible else science_by_model[model_id])
                for model_id in sorted(_MODEL_IDS)
            },
        },
        "model_decisions": model_decisions,
        "evidence_actuals": {
            "inherited_capability": {
                "hosted_completions": (V2112_INHERITED_CAPABILITY_CALLS),
                "usage": inherited_actual,
                "budget_treatment": "already-in-parent-debit",
            },
            "fresh_preflight": {
                "hosted_completions": V2112_FRESH_PREFLIGHT_CALLS,
                "usage": fresh_actual,
                "budget_treatment": "current-attempt-actual",
            },
            "combined_gate_evidence": {
                "hosted_completions": V2112_GATE_EVIDENCE_CALLS,
                "usage": evidence_actual,
                "budget_treatment": ("evidence-only-never-added-as-one-budget-row"),
            },
        },
        "observed_reservations": observed_reservations,
        "dispatch_reservations": dispatch_reservations,
        "authority_sources": authority_sources,
        "projection": projection,
        "bootstrap_envelope_used_as_observed_p95": False,
        "provider_construction_during_authority": False,
        "provider_calls_during_authority": 0,
        "go": not reasons,
        "reasons": reasons,
        "scientific_evidence": False,
        "claim_boundary": V2112_GATE_CLAIM_BOUNDARY,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def verify_v2112_gate_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_contract_sha256: str | None = None,
    expected_git_commit: str | None = None,
) -> dict[str, Any]:
    """Strictly verify a loaded V2.11.2 post-gate authority receipt."""

    value = _mapping(receipt, "V2.11.2 gate receipt")
    expected_fields = {
        "schema_version",
        "contract_id",
        "contract_sha256",
        "release",
        "bindings",
        "denominator",
        "model_decisions",
        "evidence_actuals",
        "observed_reservations",
        "dispatch_reservations",
        "authority_sources",
        "projection",
        "bootstrap_envelope_used_as_observed_p95",
        "provider_construction_during_authority",
        "provider_calls_during_authority",
        "go",
        "reasons",
        "scientific_evidence",
        "claim_boundary",
        "receipt_sha256",
    }
    if set(value) != expected_fields:
        raise PilotV2112GateError("V2.11.2 gate receipt fields drifted")
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_sha256", None)
    if (
        _SHA256_RE.fullmatch(str(claimed)) is None
        or canonical_sha256(unsigned) != claimed
    ):
        raise PilotV2112GateError("V2.11.2 gate receipt self-hash mismatch")
    if (
        value.get("schema_version") != V2112_GATE_SCHEMA_VERSION
        or value.get("contract_id") != V2112_GATE_CONTRACT_ID
    ):
        raise PilotV2112GateError("V2.11.2 gate schema/contract drifted")
    contract_hash = _sha256(
        value.get("contract_sha256"),
        "gate contract SHA-256",
    )
    if expected_contract_sha256 is not None and contract_hash != _sha256(
        expected_contract_sha256,
        "expected contract SHA-256",
    ):
        raise PilotV2112GateError("gate contract binding mismatch")
    release = _mapping(value.get("release"), "gate release")
    if (
        set(release) != {"tag", "commit"}
        or release.get("tag") != V2112_GATE_RELEASE_TAG
    ):
        raise PilotV2112GateError("V2.11.2 gate release tag drifted")
    commit = _commit(release.get("commit"), "gate release commit")
    if expected_git_commit is not None and commit != _commit(
        expected_git_commit, "expected release commit"
    ):
        raise PilotV2112GateError("gate release commit binding mismatch")
    if (
        value.get("bootstrap_envelope_used_as_observed_p95") is not False
        or value.get("provider_construction_during_authority") is not False
        or value.get("provider_calls_during_authority") != 0
        or value.get("scientific_evidence") is not False
        or value.get("claim_boundary") != V2112_GATE_CLAIM_BOUNDARY
    ):
        raise PilotV2112GateError("V2.11.2 gate scope/zero-provider invariant drifted")
    bindings = _mapping(value.get("bindings"), "gate bindings")
    if set(bindings) != {
        "parent_import_receipt",
        "source_manifest",
        "ledger_event_chain_head",
        "gate_artifacts",
    }:
        raise PilotV2112GateError("gate binding fields drifted")
    _parent_import_binding(bindings["parent_import_receipt"])
    if bindings.get("source_manifest") != _expected_manifest_binding():
        raise PilotV2112GateError("gate source manifest binding drifted")
    _sha256(
        bindings.get("ledger_event_chain_head"),
        "gate ledger event-chain head",
    )
    artifacts = _mapping(bindings.get("gate_artifacts"), "gate artifacts")
    if set(artifacts) != _MODEL_IDS:
        raise PilotV2112GateError("gate artifact model denominator drifted")
    for model_id in sorted(_MODEL_IDS):
        row = _mapping(artifacts[model_id], f"{model_id} gate artifacts")
        if set(row) != {"inherited_capability", "fresh_preflight"}:
            raise PilotV2112GateError(f"{model_id} gate artifact kinds drifted")
        inherited = _mapping(
            row["inherited_capability"],
            f"{model_id} inherited capability binding",
        )
        if (
            set(inherited)
            != {
                "run_id",
                "wrapper_content_sha256",
                "schema_version",
                "historical_source_calls",
            }
            or inherited.get("schema_version")
            != V2112_CAPABILITY_WRAPPER_SCHEMA_VERSION
            or inherited.get("historical_source_calls")
            != V2112_INHERITED_CAPABILITY_CALLS_PER_MODEL
        ):
            raise PilotV2112GateError(f"{model_id} inherited artifact binding drifted")
        if inherited.get("run_id") != (
            "finevo-pilot-v2.11.1--capability-gate--"
            f"{model_id}--capability-probe--none--"
            "provider-preflight-default--s2010922376"
        ):
            raise PilotV2112GateError(
                f"{model_id} inherited capability source release drifted"
            )
        _sha256(
            inherited.get("wrapper_content_sha256"),
            "capability wrapper content SHA-256",
        )
        preflight = _mapping(
            row["fresh_preflight"],
            f"{model_id} fresh preflight binding",
        )
        expected_preflight_fields = {
            "ledger_run_id",
            "checkpoint_run_id",
            "run_spec_sha256",
            "checkpoint_artifact_sha256",
            "checkpoint_content_sha256",
            "checkpoint_schema_version",
            "exactness_artifact_sha256",
            "exactness_content_sha256",
            "exactness_schema_version",
            "provider_call_journal_artifact_sha256",
            "provider_call_journal_sha256",
        }
        if set(preflight) != expected_preflight_fields:
            raise PilotV2112GateError(
                f"{model_id} fresh preflight binding fields drifted"
            )
        ledger_run_id = _normalized_text(
            preflight.get("ledger_run_id"),
            f"{model_id} preflight ledger run ID",
        )
        if (
            preflight.get("checkpoint_run_id")
            != (ledger_run_id + V211_PREFLIGHT_CHECKPOINT_RUN_SUFFIX)
            or not ledger_run_id.startswith(
                "finevo-pilot-v2.11.2--long-context-preflight--"
            )
            or preflight.get("exactness_schema_version")
            != (V2112_PREFLIGHT_EXACTNESS_SCHEMA_VERSION)
        ):
            raise PilotV2112GateError(
                f"{model_id} fresh checkpoint/exactness release drifted"
            )
        for field in (
            "run_spec_sha256",
            "checkpoint_artifact_sha256",
            "checkpoint_content_sha256",
            "exactness_artifact_sha256",
            "exactness_content_sha256",
            "provider_call_journal_artifact_sha256",
            "provider_call_journal_sha256",
        ):
            _sha256(preflight.get(field), f"{model_id} {field}")
    denominator = _mapping(value.get("denominator"), "gate denominator")
    if set(denominator) != {
        "registered_science_cells",
        "science_run_ids_by_model",
        "science_run_ids_sha256",
        "inherited_capability_evidence_calls",
        "fresh_preflight_calls",
        "gate_evidence_calls",
        "registered_remaining_science_calls",
        "registered_fresh_full_matrix_calls",
        "parent_debit_calls",
        "cumulative_full_matrix_calls",
        "registered_call_headroom",
        "locally_eligible_model_ids",
        "eligible_model_ids",
        "no_go_science_run_ids_by_model",
    }:
        raise PilotV2112GateError("gate denominator fields drifted")
    expected_scalar_counts = {
        "registered_science_cells": V2112_REGISTERED_SCIENCE_CELLS,
        "inherited_capability_evidence_calls": (V2112_INHERITED_CAPABILITY_CALLS),
        "fresh_preflight_calls": V2112_FRESH_PREFLIGHT_CALLS,
        "gate_evidence_calls": V2112_GATE_EVIDENCE_CALLS,
        "registered_remaining_science_calls": (V2112_REGISTERED_SCIENCE_CALLS),
        "registered_fresh_full_matrix_calls": V2112_REGISTERED_FRESH_CALLS,
        "parent_debit_calls": V2112_PARENT_HOSTED_COMPLETIONS,
        "cumulative_full_matrix_calls": V2112_CUMULATIVE_FULL_CALLS,
        "registered_call_headroom": V2112_REGISTERED_CALL_HEADROOM,
    }
    for field, expected in expected_scalar_counts.items():
        if denominator.get(field) != expected:
            raise PilotV2112GateError(f"gate denominator drifted at {field}")
    science_by_model, all_science_ids = _science_run_ids(
        denominator.get("science_run_ids_by_model")
    )
    if denominator.get("science_run_ids_sha256") != canonical_sha256(all_science_ids):
        raise PilotV2112GateError("science run-ID hash mismatch")
    locally_eligible = denominator.get("locally_eligible_model_ids")
    eligible = denominator.get("eligible_model_ids")
    if (
        not isinstance(locally_eligible, list)
        or locally_eligible != sorted(locally_eligible)
        or len(locally_eligible) != len(set(locally_eligible))
        or not set(locally_eligible) <= _MODEL_IDS
        or not isinstance(eligible, list)
        or eligible != sorted(eligible)
        or len(eligible) != len(set(eligible))
        or frozenset(eligible) not in {frozenset(), _MODEL_IDS}
        or (eligible and set(locally_eligible) != _MODEL_IDS)
    ):
        raise PilotV2112GateError("eligible model IDs drifted")
    expected_no_go_ids = {
        model_id: ([] if model_id in eligible else science_by_model[model_id])
        for model_id in sorted(_MODEL_IDS)
    }
    if denominator.get("no_go_science_run_ids_by_model") != expected_no_go_ids:
        raise PilotV2112GateError("no-go science run IDs drifted")
    decisions = _mapping(value.get("model_decisions"), "model decisions")
    observed = _mapping(
        value.get("observed_reservations"),
        "observed reservations",
    )
    dispatch = _mapping(
        value.get("dispatch_reservations"),
        "dispatch reservations",
    )
    sources = _mapping(value.get("authority_sources"), "authority sources")
    if (
        set(decisions) != _MODEL_IDS
        or set(observed) != _MODEL_IDS
        or set(dispatch) != set(eligible)
        or set(sources) != _MODEL_IDS
    ):
        raise PilotV2112GateError(
            "model decision/reservation/source denominator drifted"
        )
    for model_id in sorted(_MODEL_IDS):
        decision = _mapping(
            decisions[model_id],
            f"{model_id} decision",
        )
        if set(decision) != {
            "terminal_status",
            "locally_eligible_for_science_dispatch",
            "eligible_for_science_dispatch",
            "capability_pass",
            "interface_pass",
            "sample_counts",
            "sample_hashes",
            "interface_gates",
            "no_go_science_cell_count",
        }:
            raise PilotV2112GateError(f"{model_id} decision fields drifted")
        model_eligible = model_id in eligible
        model_locally_eligible = model_id in locally_eligible
        expected_status = _model_status(
            capability_pass=decision.get("capability_pass") is True,
            interface_pass=decision.get("interface_pass") is True,
        )
        if (
            decision.get("terminal_status") != expected_status
            or decision.get("eligible_for_science_dispatch") is not model_eligible
            or decision.get("locally_eligible_for_science_dispatch")
            is not model_locally_eligible
            or model_locally_eligible != (expected_status == "eligible")
            or decision.get("sample_counts")
            != {
                "action": V2112_ACTION_SAMPLES_PER_MODEL,
                "semantic": V2112_SEMANTIC_SAMPLES_PER_MODEL,
            }
            or decision.get("no_go_science_cell_count")
            != (0 if model_eligible else V2112_SCIENCE_CELLS[model_id])
        ):
            raise PilotV2112GateError(f"{model_id} decision accounting drifted")
        sample_hashes = _mapping(
            decision.get("sample_hashes"),
            f"{model_id} sample hashes",
        )
        interface_gates = _mapping(
            decision.get("interface_gates"),
            f"{model_id} interface gates",
        )
        if set(sample_hashes) != {"action", "semantic"} or set(interface_gates) != {
            "action",
            "semantic",
        }:
            raise PilotV2112GateError(f"{model_id} interface gate decision drifted")
        for call_kind in ("action", "semantic"):
            _sha256(
                sample_hashes.get(call_kind),
                f"{model_id} {call_kind} sample hash",
            )
        if decision.get("interface_pass") is not all(
            _mapping(
                interface_gates[call_kind],
                f"{model_id} {call_kind} interface gate",
            ).get("passed")
            is True
            for call_kind in ("action", "semantic")
        ):
            raise PilotV2112GateError(f"{model_id} interface pass is not additive")
        by_kind = _mapping(
            observed[model_id],
            f"{model_id} observed reservations",
        )
        if set(by_kind) != {"action", "semantic"}:
            raise PilotV2112GateError(f"{model_id} observed reservation kinds drifted")
        for call_kind in ("action", "semantic"):
            try:
                parsed = PreflightP95Reservation.from_dict(
                    model=str(_MODEL_PROFILES[model_id]["runtime_model"]),
                    call_kind=call_kind,
                    value=by_kind[call_kind],
                )
            except (TypeError, ValueError) as exc:
                raise PilotV2112GateError(
                    f"{model_id} {call_kind} reservation is invalid"
                ) from exc
            expected_samples = (
                V2112_ACTION_SAMPLES_PER_MODEL
                if call_kind == "action"
                else V2112_SEMANTIC_SAMPLES_PER_MODEL
            )
            if parsed.sample_count != expected_samples:
                raise PilotV2112GateError(
                    f"{model_id} {call_kind} sample count drifted"
                )
        if model_eligible and dispatch[model_id] != by_kind:
            raise PilotV2112GateError(
                f"{model_id} dispatch reservation differs from observed p95"
            )
        source = _mapping(
            sources[model_id],
            f"{model_id} authority source",
        )
        artifact = artifacts[model_id]
        inherited_artifact = artifact["inherited_capability"]
        preflight_artifact = artifact["fresh_preflight"]
        if (
            set(source)
            != {
                "source_preflight_run_id",
                "source_preflight_run_spec_sha256",
                "source_model_id",
                "source_served_model",
                "source_execution_artifact_sha256",
                "source_provider_call_journal_sha256",
                "source_capability_wrapper_content_sha256",
            }
            or source.get("source_preflight_run_id")
            != preflight_artifact["ledger_run_id"]
            or source.get("source_preflight_run_spec_sha256")
            != preflight_artifact["run_spec_sha256"]
            or source.get("source_model_id") != model_id
            or source.get("source_served_model")
            != _MODEL_PROFILES[model_id]["served_model"]
            or source.get("source_execution_artifact_sha256")
            != preflight_artifact["checkpoint_artifact_sha256"]
            or source.get("source_provider_call_journal_sha256")
            != preflight_artifact["provider_call_journal_sha256"]
            or source.get("source_capability_wrapper_content_sha256")
            != inherited_artifact["wrapper_content_sha256"]
        ):
            raise PilotV2112GateError(f"{model_id} authority source binding drifted")
        for field in (
            "source_preflight_run_spec_sha256",
            "source_execution_artifact_sha256",
            "source_provider_call_journal_sha256",
            "source_capability_wrapper_content_sha256",
        ):
            _sha256(source.get(field), f"{model_id} source {field}")
    evidence = _mapping(value.get("evidence_actuals"), "evidence actuals")
    if set(evidence) != {
        "inherited_capability",
        "fresh_preflight",
        "combined_gate_evidence",
    }:
        raise PilotV2112GateError("evidence actual sections drifted")
    inherited_usage = _usage(
        _mapping(
            evidence["inherited_capability"],
            "inherited capability actuals",
        ).get("usage"),
        "inherited capability actual usage",
    )
    fresh_usage = _usage(
        _mapping(
            evidence["fresh_preflight"],
            "fresh preflight actuals",
        ).get("usage"),
        "fresh preflight actual usage",
    )
    combined_usage = _usage(
        _mapping(
            evidence["combined_gate_evidence"],
            "combined gate evidence actuals",
        ).get("usage"),
        "combined gate evidence usage",
    )
    expected_evidence_treatments = {
        "inherited_capability": "already-in-parent-debit",
        "fresh_preflight": "current-attempt-actual",
        "combined_gate_evidence": ("evidence-only-never-added-as-one-budget-row"),
    }
    for section, treatment in expected_evidence_treatments.items():
        row = _mapping(evidence[section], f"{section} actuals")
        if (
            set(row) != {"hosted_completions", "usage", "budget_treatment"}
            or row.get("budget_treatment") != treatment
        ):
            raise PilotV2112GateError(f"{section} budget treatment drifted")
    if (
        evidence["inherited_capability"].get("hosted_completions")
        != V2112_INHERITED_CAPABILITY_CALLS
        or evidence["fresh_preflight"].get("hosted_completions")
        != V2112_FRESH_PREFLIGHT_CALLS
        or evidence["combined_gate_evidence"].get("hosted_completions")
        != V2112_GATE_EVIDENCE_CALLS
        or not _usage_equal(
            combined_usage,
            _sum_usage([inherited_usage, fresh_usage]),
        )
    ):
        raise PilotV2112GateError("evidence actual accounting drifted")
    projection = _mapping(value.get("projection"), "full matrix projection")
    recomputed_projection = _projection(
        parent_debit=projection.get("parent_debit"),
        fresh_preflight_usage=fresh_usage,
        current_attempt_pre_science_storage_bytes=_mapping(
            projection.get("current_attempt_pre_science"),
            "projection current-attempt pre-science",
        ).get("storage_bytes"),
        observed_reservations=observed,
        eligible_model_ids=eligible,
    )
    if projection != recomputed_projection:
        raise PilotV2112GateError("full-matrix projection drifted")
    reasons = value.get("reasons")
    if not isinstance(reasons, list):
        raise PilotV2112GateError("gate reasons must be an array")
    expected_reasons = list(projection["reasons"])
    if locally_eligible and set(locally_eligible) != _MODEL_IDS:
        expected_reasons.append("mixed-model-terminal-status-global-no-go")
    elif not locally_eligible:
        expected_reasons.append("no-dispatch-eligible-models")
    expected_eligible = sorted(_MODEL_IDS) if not expected_reasons else []
    if (
        reasons != expected_reasons
        or eligible != expected_eligible
        or value.get("go") is not (not expected_reasons)
    ):
        raise PilotV2112GateError("gate go/no-go decision drifted")
    return dict(value)


def _strict_json_object(data: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PilotV2112GateError(f"{name} is not UTF-8") from exc

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PilotV2112GateError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise PilotV2112GateError(f"{name} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise PilotV2112GateError(f"{name} must contain one JSON object")
    return value


def _safe_relative_receipt_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str):
        raise PilotV2112GateError("receipt path must be text")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or len(path.parts) < 2
        or path.parts[0] != "experiment_results"
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise PilotV2112GateError(
            "receipt path must be normalized below experiment_results/"
        )
    return path


def _read_no_follow(
    *,
    repo_root: str | Path,
    receipt_path: PurePosixPath,
) -> bytes:
    root = Path(repo_root).expanduser().absolute()
    try:
        if root.is_symlink() or not root.is_dir():
            raise PilotV2112GateError(
                "repository root must be a regular non-symlink directory"
            )
        current = root
        for part in receipt_path.parts[:-1]:
            current = current / part
            info = os.lstat(current)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise PilotV2112GateError(
                    "receipt parent contains a symlink/non-directory"
                )
        target = current / receipt_path.parts[-1]
        info = os.lstat(target)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise PilotV2112GateError("receipt must be a regular non-symlink file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(target, flags)
        try:
            data = os.read(fd, info.st_size + 1)
        finally:
            os.close(fd)
    except FileNotFoundError as exc:
        raise PilotV2112GateError("gate receipt file is missing") from exc
    except OSError as exc:
        raise PilotV2112GateError("gate receipt file cannot be read") from exc
    if len(data) != info.st_size:
        raise PilotV2112GateError("gate receipt changed while reading")
    return data


def _source_authority(
    *,
    receipt: Mapping[str, Any],
    model_id: str,
    receipt_file_sha256: str,
) -> dict[str, str]:
    source = _mapping(
        receipt["authority_sources"][model_id],
        f"{model_id} authority source",
    )
    return {
        "authority_id": OBSERVED_P95_AUTHORITY_ID,
        "source_kind": OBSERVED_P95_SOURCE_KIND,
        "pilot_contract_hash": str(receipt["contract_sha256"]),
        "pilot_tag": str(receipt["release"]["tag"]),
        "source_projection_schema_version": (OBSERVED_P95_PROJECTION_SCHEMA_VERSION),
        "source_projection_file_sha256": receipt_file_sha256,
        "source_projection_content_sha256": str(receipt["receipt_sha256"]),
        "source_preflight_run_id": str(source["source_preflight_run_id"]),
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


def verified_v2112_gate_authority_binding(
    receipt_path: str,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
    expected_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify one immutable receipt file and return its source binding."""

    relative = _safe_relative_receipt_path(receipt_path)
    data = _read_no_follow(repo_root=repo_root, receipt_path=relative)
    file_sha256 = hashlib.sha256(data).hexdigest()
    receipt = verify_v2112_gate_receipt(
        _strict_json_object(data, name="V2.11.2 gate receipt file"),
        expected_contract_sha256=expected_contract_sha256,
        expected_git_commit=expected_git_commit,
    )
    if receipt.get("go") is not True:
        raise PilotV2112GateError("V2.11.2 gate receipt is global no-go")
    reservations: dict[str, dict[str, Any]] = {}
    for model_id in receipt["denominator"]["eligible_model_ids"]:
        runtime_model = str(_MODEL_PROFILES[model_id]["runtime_model"])
        source = _source_authority(
            receipt=receipt,
            model_id=model_id,
            receipt_file_sha256=file_sha256,
        )
        reservations[runtime_model] = {
            call_kind: {
                "authority": dict(source),
                "reservation": receipt["dispatch_reservations"][model_id][call_kind],
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


def runner_reservations_from_v2112_gate_binding(
    binding: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Expand a verified gate file binding to runner authority rows."""

    value = _mapping(binding, "V2.11.2 gate authority binding")
    if set(value) != {
        "receipt_path",
        "receipt_file_sha256",
        "receipt_content_sha256",
        "git_commit",
        "reservations",
    }:
        raise PilotV2112GateError("V2.11.2 gate authority binding fields drifted")
    receipt_path = _safe_relative_receipt_path(value["receipt_path"]).as_posix()
    file_sha256 = _sha256(
        value["receipt_file_sha256"],
        "gate receipt file SHA-256",
    )
    content_sha256 = _sha256(
        value["receipt_content_sha256"],
        "gate receipt content SHA-256",
    )
    git_commit = _commit(value["git_commit"], "gate release commit")
    source = _mapping(value["reservations"], "gate reservations")
    expected_runtime_models = {
        str(profile["runtime_model"]) for profile in _MODEL_PROFILES.values()
    }
    if set(source) != expected_runtime_models:
        raise PilotV2112GateError(
            "gate binding must contain both registered runtime reservations"
        )
    result: dict[str, dict[str, Any]] = {}
    for runtime_model, raw_by_kind in source.items():
        by_kind = _mapping(
            raw_by_kind,
            f"{runtime_model} gate reservations",
        )
        if set(by_kind) != {"action", "semantic"}:
            raise PilotV2112GateError(f"{runtime_model} reservation kinds drifted")
        result[runtime_model] = {}
        for call_kind in ("action", "semantic"):
            entry = _mapping(
                by_kind[call_kind],
                f"{runtime_model} {call_kind} gate reservation",
            )
            if set(entry) != {"authority", "reservation"}:
                raise PilotV2112GateError(
                    f"{runtime_model} {call_kind} binding fields drifted"
                )
            authority = dict(_mapping(entry["authority"], "observed p95 authority"))
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
                    "observed p95 reservation",
                ),
            }
            try:
                sealed = ObservedPreflightP95Reservation.from_dict(
                    model=runtime_model,
                    call_kind=call_kind,
                    value=candidate,
                ).to_dict()
            except (TypeError, ValueError) as exc:
                raise PilotV2112GateError(
                    f"{runtime_model} {call_kind} authority is invalid"
                ) from exc
            result[runtime_model][call_kind] = sealed
    return result


def runner_reservations_from_v2112_gate(
    receipt_path: str,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
    expected_contract_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Verify a receipt file and return runner-compatible reservations."""

    binding = verified_v2112_gate_authority_binding(
        receipt_path,
        repo_root=repo_root,
        expected_git_commit=expected_git_commit,
        expected_contract_sha256=expected_contract_sha256,
    )
    return runner_reservations_from_v2112_gate_binding(binding)


__all__ = [
    "PilotV2112GateError",
    "V2112_ACTION_SAMPLES_PER_MODEL",
    "V2112_CUMULATIVE_FULL_CALLS",
    "V2112_FRESH_PREFLIGHT_CALLS",
    "V2112_GATE_EVIDENCE_CALLS",
    "V2112_GATE_CLAIM_BOUNDARY",
    "V2112_GATE_RELEASE_TAG",
    "V2112_GATE_SCHEMA_VERSION",
    "V2112_INHERITED_CAPABILITY_CALLS",
    "V2112_REGISTERED_FRESH_CALLS",
    "V2112_REGISTERED_CALL_HEADROOM",
    "V2112_REGISTERED_SCIENCE_CALLS",
    "V2112_PREFLIGHT_EXACTNESS_SCHEMA_VERSION",
    "V2112_SEMANTIC_SAMPLES_PER_MODEL",
    "build_v2112_post_gate_authority",
    "canonical_sha256",
    "runner_reservations_from_v2112_gate",
    "runner_reservations_from_v2112_gate_binding",
    "verified_v2112_gate_authority_binding",
    "verify_v2112_gate_receipt",
]
