from __future__ import annotations

from pathlib import Path

import pytest

from verified_memory.pilot_budget import (
    PilotBudgetCaps,
    PilotBudgetError,
    PilotBudgetLedger,
    RunProjection,
    preflight_p95,
)


CONTRACT_HASH = "a" * 64


def _projection(run_id: str, cost: float = 0.5) -> RunProjection:
    return RunProjection(
        run_id=run_id,
        stage_bucket="capability_preflight",
        cost_usd=cost,
        completions=10,
        storage_bytes=100,
        basis={"source": "test"},
    )


def test_reservation_and_finalize_are_idempotent(tmp_path: Path) -> None:
    ledger = PilotBudgetLedger(tmp_path / "budget.json", contract_hash=CONTRACT_HASH)
    projection = _projection("run-a")
    ledger.reserve(projection)
    ledger.reserve(projection)
    ledger.finalize(
        "run-a",
        status="complete",
        cost_usd=0.25,
        completions=8,
        storage_bytes=90,
    )
    ledger.finalize(
        "run-a",
        status="complete",
        cost_usd=0.25,
        completions=8,
        storage_bytes=90,
    )
    snapshot = ledger.snapshot()
    assert snapshot["committed"]["cost_usd"] == pytest.approx(0.25)
    assert snapshot["committed_plus_reserved"]["cost_usd"] == pytest.approx(0.25)


def test_stage_cap_stops_before_dispatch(tmp_path: Path) -> None:
    ledger = PilotBudgetLedger(tmp_path / "budget.json", contract_hash=CONTRACT_HASH)
    ledger.reserve(_projection("run-a", cost=1.5))
    with pytest.raises(PilotBudgetError, match="capability_preflight USD"):
        ledger.reserve(_projection("run-b", cost=0.6))


def test_manual_reserve_cannot_be_dispatched(tmp_path: Path) -> None:
    ledger = PilotBudgetLedger(tmp_path / "budget.json", contract_hash=CONTRACT_HASH)
    with pytest.raises(PilotBudgetError, match="manual reserve"):
        ledger.reserve(
            RunProjection(
                run_id="forbidden",
                stage_bucket="manual_reserve",
                cost_usd=0.1,
                completions=1,
                storage_bytes=1,
                basis={},
            )
        )


def test_caps_require_exact_frozen_allocation() -> None:
    with pytest.raises(ValueError, match="sum"):
        PilotBudgetCaps(stage_usd_caps={"manual_reserve": 1.0})


def test_preflight_p95_groups_model_and_call_kind() -> None:
    rows = [
        {
            "response_model": "model-a",
            "call_kind": "action",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "cost_usd": 0.01,
            },
        },
        {
            "response_model": "model-a",
            "call_kind": "action",
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 20,
                "total_tokens": 220,
                "cost_usd": 0.02,
            },
        },
    ]
    result = preflight_p95(rows)
    assert result["model-a::action"]["sample_count"] == 2
    assert result["model-a::action"]["reserved_p95"]["cost_usd"] > 0.02


def test_preflight_p95_never_reserves_below_an_observed_outlier() -> None:
    rows = [
        {
            "response_model": "model-a",
            "call_kind": "action",
            "usage": {
                "prompt_tokens": value,
                "completion_tokens": value,
                "total_tokens": value * 2,
                "cost_usd": value / 1000,
            },
        }
        for value in ([10] * 11 + [100])
    ]

    result = preflight_p95(rows)["model-a::action"]

    assert result["sample_count"] == 12
    assert result["raw_p95"] == {
        "prompt_tokens": 100.0,
        "completion_tokens": 100.0,
        "total_tokens": 200.0,
        "cost_usd": 0.1,
    }
    assert result["reserved_p95"] == {
        "prompt_tokens": 125,
        "completion_tokens": 125,
        "total_tokens": 250,
        "cost_usd": 0.125,
    }
