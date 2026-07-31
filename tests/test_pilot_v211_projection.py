from __future__ import annotations

import pytest

from verified_memory.pilot_v211_projection import (
    PilotV211ProjectionError,
    V211_STAGE_LEDGER_CELLS,
    project_v211_full_matrix,
)


def _historical_reserved_p95() -> dict:
    return {
        "gpt52_main": {
            "action": {
                "prompt_tokens": 1138,
                "completion_tokens": 2499,
                "total_tokens": 3637,
                "cost_usd": 0.0363540625,
            },
            "semantic": {
                "prompt_tokens": 3018,
                "completion_tokens": 1379,
                "total_tokens": 4397,
                "cost_usd": 0.0224896875,
            },
        },
        "gpt56_diagnostic": {
            "action": {
                "prompt_tokens": 1138,
                "completion_tokens": 1539,
                "total_tokens": 2677,
                "cost_usd": 0.04819375,
            },
            "semantic": {
                "prompt_tokens": 3093,
                "completion_tokens": 577,
                "total_tokens": 3670,
                "cost_usd": 0.03275,
            },
        },
    }


def _pre_science_actual_fixture() -> dict:
    # Design-time fixture only: the real authority must read these four values
    # from the terminal capability/preflight budget ledger.
    return {
        "prompt_tokens": 194_802,
        "completion_tokens": 221_208,
        "total_tokens": 416_010,
        "cost_usd": 4.831650625,
    }


def _project(**overrides):
    arguments = {
        "pre_science_actual_usage": _pre_science_actual_fixture(),
        "pre_science_actual_hosted_completions": 124,
        "pre_science_actual_storage_bytes": 100_000_000,
    }
    arguments.update(overrides)
    return project_v211_full_matrix(
        _historical_reserved_p95(),
        **arguments,
    )


def test_full_scope_projection_preserves_all_registered_denominators() -> None:
    projection = _project()

    assert projection.stage_calls == {
        "capability-gate": 60,
        "long-context-preflight": 64,
        "experiment-a": 1280,
        "experiment-b": 1440,
        "experiment-c": 1280,
        "experiment-d": 1480,
        "cross-model": 336,
    }
    assert projection.model_calls == {
        "gpt52_main": 5542,
        "gpt56_diagnostic": 398,
    }
    assert V211_STAGE_LEDGER_CELLS == {
        "parent-import": 1,
        "capability-gate": 2,
        "long-context-preflight": 2,
        "experiment-a": 20,
        "experiment-b": 25,
        "experiment-c": 25,
        "experiment-d": 55,
        "cross-model": 6,
    }
    assert projection.new_hosted_completions == 5940
    assert projection.pre_science_actual_hosted_completions == 124
    assert projection.remaining_science_hosted_completions == 5816
    assert projection.cumulative_hosted_completions == 6756
    assert projection.new_cost_usd == pytest.approx(206.748488125)
    assert projection.cumulative_cost_usd == pytest.approx(
        222.79341093750002
    )
    assert projection.prompt_tokens == 8_636_850
    assert projection.completion_tokens == 13_356_256
    assert projection.total_tokens == 21_993_106
    assert projection.cumulative_storage_bytes == 2_237_010_835
    assert projection.go is True
    assert projection.reasons == ()
    assert "gate ledger actuals" in projection.to_dict()["claim_boundary"]


def test_projection_no_go_is_cumulative_and_does_not_shrink_matrix() -> None:
    reservations = _historical_reserved_p95()
    for by_kind in reservations.values():
        for row in by_kind.values():
            row["cost_usd"] = 0.10

    projection = project_v211_full_matrix(
        reservations,
        pre_science_actual_usage=_pre_science_actual_fixture(),
        pre_science_actual_hosted_completions=124,
        pre_science_actual_storage_bytes=100_000_000,
    )

    assert projection.new_hosted_completions == 5940
    assert projection.cumulative_hosted_completions == 6756
    assert projection.go is False
    assert projection.reasons == ("cumulative-cost-exceeds-hard-cap",)


def test_projection_fails_closed_on_missing_model_or_call_kind() -> None:
    reservations = _historical_reserved_p95()
    del reservations["gpt56_diagnostic"]["semantic"]

    with pytest.raises(PilotV211ProjectionError, match="exactly action"):
        project_v211_full_matrix(
            reservations,
            pre_science_actual_usage=_pre_science_actual_fixture(),
            pre_science_actual_hosted_completions=124,
            pre_science_actual_storage_bytes=100_000_000,
        )


def test_projection_fails_closed_on_nonadditive_token_row() -> None:
    reservations = _historical_reserved_p95()
    reservations["gpt52_main"]["action"]["total_tokens"] = 1

    with pytest.raises(PilotV211ProjectionError, match="not additive"):
        project_v211_full_matrix(
            reservations,
            pre_science_actual_usage=_pre_science_actual_fixture(),
            pre_science_actual_hosted_completions=124,
            pre_science_actual_storage_bytes=100_000_000,
        )


def test_projection_fails_closed_on_storage_cap() -> None:
    projection = _project(
        pre_science_actual_storage_bytes=3_000_000_000,
    )

    assert projection.go is False
    assert projection.reasons == ("cumulative-storage-exceeds-hard-cap",)


def test_projection_rejects_nonterminal_preflight_denominator() -> None:
    with pytest.raises(PilotV211ProjectionError, match="exactly 124"):
        _project(pre_science_actual_hosted_completions=123)
