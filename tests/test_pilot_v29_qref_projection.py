from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import pytest

from verified_memory.pilot_v29_qref_projection import (
    PilotV29QRefProjectionError,
    QREF_RUN_SUMMARY_EQUIVALENCE_SCHEMA_VERSION,
    QREF_RUN_SUMMARY_PROJECTION_SCHEMA_VERSION,
    build_qref_run_summary_equivalence_receipt,
    canonical_sha256,
    project_qref_run_summary,
)


CURRENT_RUN_ID = (
    "finevo-pilot-v2.9--q-ref-resolution--qref_scripted--"
    "qref-scripted--none--provider-preflight-default--s2010922376"
)
HISTORICAL_RUN_ID = "q-ref-resolution-s2010922376"
CURRENT_BUDGET_ID = f"{CURRENT_RUN_ID}-budget"
HISTORICAL_SOURCE_RUN_ID = (
    "finevo-pilot-v2.6--q-ref-resolution--qref_scripted--"
    "qref-scripted--none--provider-preflight-default--s2010922376"
)
HISTORICAL_BUDGET_ID = f"{HISTORICAL_SOURCE_RUN_ID}-budget"

REAL_V28_SCIENCE_ROOT = Path(
    "/Users/guanghaowu/Develop/financial world/worktrees/" "finevo-pilot-v2-8-science"
)
REAL_V28_RUN_ID = CURRENT_RUN_ID.replace(
    "finevo-pilot-v2.9",
    "finevo-pilot-v2.8",
    1,
)
REAL_V28_BUDGET_ID = f"{REAL_V28_RUN_ID}-budget"
REAL_V28_SUMMARY_FILE_SHA256 = (
    "7d25fb5bcd9bb3553b66a7626aa97517214556fbd3703f14c07d605c89ee0499"
)
REAL_V28_SUMMARY_CANONICAL_SHA256 = (
    "11cbec2509ace1e25b0cfc3c3f06c404b7829836851401d39c3997a9cced8274"
)
REAL_V26_SUMMARY_CANONICAL_SHA256 = (
    "ef68345da71e8826e451ec30521d0eed48a92bdbba02d4c7f18c5f12b5d2762d"
)
REAL_PAIR_PROJECTION_SHA256 = (
    "7d909ba445fddb1ab13288271959165aa1215e703a668e3fa3a9e5171a332a42"
)


def _usage(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float = 0.0,
) -> dict[str, int | float]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost_usd,
    }


def _sum_usage(values: list[dict[str, int | float]]) -> dict[str, int | float]:
    return {
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in values),
        "completion_tokens": sum(int(row["completion_tokens"]) for row in values),
        "total_tokens": sum(int(row["total_tokens"]) for row in values),
        "cost_usd": sum(float(row["cost_usd"]) for row in values),
    }


def _summary(
    *,
    run_id: str,
    budget_id: str,
    timing_offset: float,
    timing_scale: float,
) -> dict[str, Any]:
    completions = []
    actual_usage = []
    for index in range(48):
        decision_t = index // 4
        agent_id = index % 4
        started = timing_offset + index * timing_scale
        finished = started + timing_scale / 10
        actual = _usage(
            prompt_tokens=300 + (index % 7),
            completion_tokens=26,
        )
        estimated = _usage(
            prompt_tokens=301 + (index % 7),
            completion_tokens=220,
        )
        actual_usage.append(actual)
        completions.append(
            {
                "budget_id": budget_id,
                "reservation_id": index + 1,
                "label": f"action:t{decision_t}:a{agent_id}",
                "model": "diagnostic/scripted-v1",
                "started_elapsed_seconds": started,
                "finished_elapsed_seconds": finished,
                "elapsed_seconds": finished - started,
                "estimated_usage": estimated,
                "usage": actual,
                "tags": {
                    "agent_id": str(agent_id),
                    "batch_index": str(agent_id),
                    "call_kind": "action",
                    "decision_t": str(decision_t),
                },
            }
        )
    accounted = _sum_usage(actual_usage)
    zero_usage = _usage(prompt_tokens=0, completion_tokens=0)
    return {
        "schema_version": "verified-simulation-runner-v3",
        "run_id": run_id,
        "result_complete": True,
        "result_scope": "bounded_method_smoke",
        "scientific_evidence": False,
        "diagnostic_only": True,
        "provider_model": "diagnostic/scripted-v1",
        "num_agents": 4,
        "episode_length": 12,
        "final_metrics": {
            "average_flow_utility": 3.5,
            "average_wealth": 100.0,
        },
        "action_diagnostics": {
            "clipped_action_count": 0,
            "intermediate_action_count": 48,
        },
        "memory_diagnostics": {
            "episodic_retrieval_count": 0,
            "semantic_activation_observed": False,
        },
        "validation": {
            "status": "pass",
            "diagnostic_only": True,
            "scientific_evidence": False,
            "checks": {"completed_all_periods": True},
        },
        "api": {
            "budget_id": budget_id,
            "limits": {
                "max_calls": 48,
                "max_prompt_tokens": 500_000,
                "max_completion_tokens": 100_000,
                "max_total_tokens": 600_000,
                "max_cost_usd": 1e-9,
                "max_elapsed_seconds": 3_600.0,
            },
            "accounted_usage": accounted,
            "reserved_usage": zero_usage,
            "effective_usage": deepcopy(accounted),
            "completed_calls": 48,
            "active_calls": 0,
            "rolled_back_calls": 0,
            "elapsed_seconds": (
                completions[-1]["finished_elapsed_seconds"] + timing_scale
            ),
            "stopped": True,
            "stop_reasons": ["call_limit"],
            "active_reservations": [],
            "completions": completions,
        },
    }


@pytest.fixture
def summary_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    current = _summary(
        run_id=CURRENT_RUN_ID,
        budget_id=CURRENT_BUDGET_ID,
        timing_offset=0.005,
        timing_scale=0.001,
    )
    historical = _summary(
        run_id=HISTORICAL_RUN_ID,
        budget_id=HISTORICAL_BUDGET_ID,
        timing_offset=0.010,
        timing_scale=0.002,
    )
    return current, historical


def _receipt(
    current: dict[str, Any],
    historical: dict[str, Any],
) -> dict[str, Any]:
    return build_qref_run_summary_equivalence_receipt(
        current,
        historical,
        expected_current_run_id=CURRENT_RUN_ID,
        expected_current_budget_id=CURRENT_BUDGET_ID,
        expected_historical_run_id=HISTORICAL_RUN_ID,
        expected_historical_budget_id=HISTORICAL_BUDGET_ID,
    )


def test_projection_normalizes_bound_ids_and_omits_only_validated_timing(
    summary_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    current, historical = summary_pair
    original = deepcopy(current)
    current_projection = project_qref_run_summary(
        current,
        expected_run_id=CURRENT_RUN_ID,
        expected_budget_id=CURRENT_BUDGET_ID,
    )
    historical_projection = project_qref_run_summary(
        historical,
        expected_run_id=HISTORICAL_RUN_ID,
        expected_budget_id=HISTORICAL_BUDGET_ID,
    )

    assert current == original
    assert current_projection == historical_projection
    assert current_projection["schema_version"] == (
        QREF_RUN_SUMMARY_PROJECTION_SCHEMA_VERSION
    )
    projected = current_projection["summary"]
    assert projected["run_id"] == "<QREF_RUN_ID>"
    assert projected["api"]["budget_id"] == "<QREF_BUDGET_ID>"
    assert {row["budget_id"] for row in projected["api"]["completions"]} == {
        "<QREF_BUDGET_ID>"
    }
    assert "elapsed_seconds" not in projected["api"]
    assert all(
        not {
            "started_elapsed_seconds",
            "finished_elapsed_seconds",
            "elapsed_seconds",
        }.intersection(row)
        for row in projected["api"]["completions"]
    )


def test_receipt_preserves_raw_hashes_provider_boundary_and_accounting(
    summary_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    current, historical = summary_pair
    receipt = _receipt(current, historical)

    assert receipt["schema_version"] == (QREF_RUN_SUMMARY_EQUIVALENCE_SCHEMA_VERSION)
    assert receipt["status"] == "pass"
    assert receipt["current"]["raw_summary_sha256"] == canonical_sha256(current)
    assert receipt["historical_reference"]["raw_summary_sha256"] == (
        canonical_sha256(historical)
    )
    assert receipt["current"]["raw_summary_sha256"] != (
        receipt["historical_reference"]["raw_summary_sha256"]
    )
    assert receipt["current"]["projection_sha256"] == (
        receipt["historical_reference"]["projection_sha256"]
    )
    assert receipt["common_projection_sha256"] == (
        receipt["current"]["projection_sha256"]
    )
    assert receipt["current"]["provider_boundary"] == {
        "provider_model": "diagnostic/scripted-v1",
        "completion_models": {"diagnostic/scripted-v1": 48},
        "call_kind_counts": {"action": 48},
        "scripted_diagnostic_calls": 48,
        "hosted_provider_calls": 0,
        "hosted_cost_usd": 0.0,
    }
    assert receipt["current"]["accounting"]["completed_calls"] == 48
    assert receipt["comparison"]["leaf_path_count"] > 195
    assert receipt["comparison"]["normalized_leaf_path_count"] == 195
    assert receipt["current"]["accounting"]["accounted_usage"] == (
        receipt["current"]["accounting"]["completion_usage_sum"]
    )
    unsealed = deepcopy(receipt)
    integrity = unsealed.pop("integrity")
    assert integrity == {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(unsealed),
    }


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda value: value.__setitem__("run_id", "wrong-run"),
            "run_id differs",
        ),
        (
            lambda value: value["api"].__setitem__(
                "budget_id",
                "wrong-budget",
            ),
            "api.budget_id differs",
        ),
        (
            lambda value: value["api"]["completions"][0].__setitem__(
                "budget_id",
                "wrong-budget",
            ),
            "completions",
        ),
    ],
)
def test_identity_relations_fail_before_normalization(
    summary_pair: tuple[dict[str, Any], dict[str, Any]],
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    current, historical = summary_pair
    tampered = deepcopy(current)
    mutate(tampered)
    with pytest.raises(PilotV29QRefProjectionError, match=match):
        _receipt(tampered, historical)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda value: value["api"].__setitem__(
                "elapsed_seconds",
                -0.1,
            ),
            "finite non-negative",
        ),
        (
            lambda value: value["api"]["completions"][0].__setitem__(
                "started_elapsed_seconds",
                math.nan,
            ),
            "finite non-negative",
        ),
        (
            lambda value: value["api"]["completions"][0].update(
                {
                    "started_elapsed_seconds": 0.2,
                    "finished_elapsed_seconds": 0.1,
                    "elapsed_seconds": 0.0,
                }
            ),
            "starts after",
        ),
        (
            lambda value: value["api"]["completions"][0].__setitem__(
                "elapsed_seconds",
                99.0,
            ),
            "finished-started",
        ),
    ],
)
def test_volatile_timing_is_validated_not_blindly_deleted(
    summary_pair: tuple[dict[str, Any], dict[str, Any]],
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    current, historical = summary_pair
    tampered = deepcopy(current)
    mutate(tampered)
    with pytest.raises(PilotV29QRefProjectionError, match=match):
        _receipt(tampered, historical)


def _tamper_accounted_tokens(value: dict[str, Any]) -> None:
    completion = value["api"]["completions"][0]["usage"]
    completion["prompt_tokens"] += 1
    completion["total_tokens"] += 1
    for field in ("accounted_usage", "effective_usage"):
        value["api"][field]["prompt_tokens"] += 1
        value["api"][field]["total_tokens"] += 1


def _tamper_estimated_tokens(value: dict[str, Any]) -> None:
    usage = value["api"]["completions"][0]["estimated_usage"]
    usage["prompt_tokens"] += 1
    usage["total_tokens"] += 1


def _tamper_zero_cost(value: dict[str, Any]) -> None:
    value["api"]["completions"][0]["usage"]["cost_usd"] = 0.25
    value["api"]["accounted_usage"]["cost_usd"] = 0.25
    value["api"]["effective_usage"]["cost_usd"] = 0.25


@pytest.mark.parametrize(
    "mutate",
    [
        _tamper_accounted_tokens,
        _tamper_estimated_tokens,
        lambda value: value["api"]["completions"][0].__setitem__(
            "label",
            "action:t0:a9",
        ),
        lambda value: value["api"]["completions"][0].__setitem__(
            "model",
            "hosted/model",
        ),
        _tamper_zero_cost,
        lambda value: value["api"]["completions"][0]["tags"].__setitem__(
            "agent_id",
            "9",
        ),
        lambda value: value.__setitem__("unexpected", True),
        lambda value: value["api"].__setitem__("unexpected", True),
        lambda value: value["api"]["completions"][0].__setitem__(
            "unexpected",
            True,
        ),
    ],
)
def test_retained_or_unknown_path_tamper_fails_closed(
    summary_pair: tuple[dict[str, Any], dict[str, Any]],
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    current, historical = summary_pair
    tampered = deepcopy(current)
    mutate(tampered)
    with pytest.raises(PilotV29QRefProjectionError):
        _receipt(tampered, historical)


def test_completion_order_is_part_of_deterministic_projection(
    summary_pair: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    current, historical = summary_pair
    tampered = deepcopy(current)
    tampered["api"]["completions"][0], tampered["api"]["completions"][1] = (
        tampered["api"]["completions"][1],
        tampered["api"]["completions"][0],
    )
    with pytest.raises(PilotV29QRefProjectionError):
        _receipt(tampered, historical)


def _real_summary_paths() -> tuple[Path, Path]:
    current = (
        REAL_V28_SCIENCE_ROOT
        / "experiment_results/pilot-v2.8/raw/q-ref-resolution/runs"
        / REAL_V28_RUN_ID
        / "streams/summary.jsonl"
    )
    historical = (
        REAL_V28_SCIENCE_ROOT / "experiment_results/pilot-v2.8/raw/parent-import/"
        "v2_7_raw_snapshot/parent-import/v2_6_raw_snapshot/"
        "q-ref-resolution/q_ref_resolution.json"
    )
    return current, historical


def test_real_v28_failure_and_nested_v26_reference_project_exactly() -> None:
    current_path, historical_path = _real_summary_paths()
    if not current_path.is_file() or not historical_path.is_file():
        pytest.skip("immutable local V2.8 q-ref failure fixture is unavailable")
    actual_file_sha256 = hashlib.sha256(current_path.read_bytes()).hexdigest()
    assert actual_file_sha256 == REAL_V28_SUMMARY_FILE_SHA256

    current = json.loads(current_path.read_text(encoding="utf-8"))
    historical_document = json.loads(historical_path.read_text(encoding="utf-8"))
    historical = historical_document["source"]["run_summary"]
    assert canonical_sha256(current) == REAL_V28_SUMMARY_CANONICAL_SHA256
    assert canonical_sha256(historical) == REAL_V26_SUMMARY_CANONICAL_SHA256
    assert historical_document["bindings"]["run_summary_hash"] == (
        REAL_V26_SUMMARY_CANONICAL_SHA256
    )

    receipt = build_qref_run_summary_equivalence_receipt(
        current,
        historical,
        expected_current_run_id=REAL_V28_RUN_ID,
        expected_current_budget_id=REAL_V28_BUDGET_ID,
        expected_historical_run_id=HISTORICAL_RUN_ID,
        expected_historical_budget_id=HISTORICAL_BUDGET_ID,
    )
    assert receipt["status"] == "pass"
    assert receipt["current"]["raw_summary_sha256"] == (
        REAL_V28_SUMMARY_CANONICAL_SHA256
    )
    assert receipt["historical_reference"]["raw_summary_sha256"] == (
        REAL_V26_SUMMARY_CANONICAL_SHA256
    )
    assert receipt["common_projection_sha256"] == (REAL_PAIR_PROJECTION_SHA256)
    assert receipt["current"]["provider_boundary"]["hosted_provider_calls"] == 0
    assert receipt["current"]["accounting"]["accounted_usage"] == {
        "prompt_tokens": 14657,
        "completion_tokens": 1248,
        "total_tokens": 15905,
        "cost_usd": 0.0,
    }
    assert receipt["comparison"]["leaf_path_count"] == 1002
    assert receipt["comparison"]["normalized_leaf_path_count"] == 195
