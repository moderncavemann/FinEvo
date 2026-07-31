from __future__ import annotations

import pytest

from verified_memory.pilot_interface_gate import (
    PilotInterfaceGateError,
    completion_capacity_gate,
    interface_sample_gate,
)


def _reservation(
    *,
    raw_completion: float = 1999.0,
    reserved_completion: int = 2499,
    multiplier: float = 1.25,
    sample_count: int = 36,
) -> dict:
    return {
        "raw_p95": {
            "prompt_tokens": 910.0,
            "completion_tokens": raw_completion,
            "total_tokens": 2909.0,
            "cost_usd": 0.02908325,
        },
        "reserve_multiplier": multiplier,
        "reserved_p95": {
            "prompt_tokens": 1138,
            "completion_tokens": reserved_completion,
            "total_tokens": 3637,
            "cost_usd": 0.0363540625,
        },
        "sample_count": sample_count,
    }


def _sample(**overrides) -> dict:
    value = {
        "finish_reason": "stop",
        "response_completed": True,
        "output_disposition": "accepted",
        "error_type": None,
        "parse_success": True,
        "clipped": False,
        "prompt_tokens": 1_200,
        "completion_tokens": 1_500,
        "reasoning_tokens": 1_450,
        "visible_completion_tokens": 50,
    }
    value.update(overrides)
    return value


def test_v2102_observed_action_profile_fails_2048_wire_cap() -> None:
    result = completion_capacity_gate(
        call_kind="action",
        wire_cap_tokens=2048,
        reservation=_reservation(),
        minimum_sample_count=12,
    )

    assert result.passed is False
    assert result.reasons == (
        "wire-cap-below-reserved-p95",
        "raw-p95-too-close-to-wire-cap",
    )
    assert result.headroom_tokens == 49
    assert result.to_dict()["reserved_p95_completion_tokens"] == 2499


def test_same_outcome_blind_profile_passes_candidate_4096_cap() -> None:
    result = completion_capacity_gate(
        call_kind="action",
        wire_cap_tokens=4096,
        reservation=_reservation(),
        minimum_sample_count=12,
    )

    assert result.passed is True
    assert result.reasons == ()
    assert result.headroom_tokens == 2097
    assert result.headroom_fraction == pytest.approx(2097 / 4096)


@pytest.mark.parametrize(
    ("call_kind", "raw_completion", "reserved_completion", "samples"),
    [
        ("action", 1231.0, 1539, 36),
        ("semantic", 461.0, 577, 10),
    ],
)
def test_historical_gpt56_interface_profiles_pass_candidate_caps(
    call_kind: str,
    raw_completion: float,
    reserved_completion: int,
    samples: int,
) -> None:
    result = completion_capacity_gate(
        call_kind=call_kind,
        wire_cap_tokens=4096,
        reservation=_reservation(
            raw_completion=raw_completion,
            reserved_completion=reserved_completion,
            sample_count=samples,
        ),
        minimum_sample_count=samples,
    )

    assert result.passed is True
    assert result.reasons == ()


def test_capacity_gate_reports_insufficient_interface_samples() -> None:
    result = completion_capacity_gate(
        call_kind="action",
        wire_cap_tokens=4096,
        reservation=_reservation(sample_count=11),
        minimum_sample_count=12,
    )

    assert result.passed is False
    assert result.reasons == ("insufficient-samples",)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: row["reserved_p95"].update(
                {"completion_tokens": 2498}
            ),
            "does not equal",
        ),
        (
            lambda row: row.update({"reserve_multiplier": 1.0}),
            "greater than 1",
        ),
        (
            lambda row: row.update({"extra": True}),
            "must contain exactly",
        ),
    ],
)
def test_capacity_gate_fails_closed_on_malformed_reservation(
    mutation,
    message: str,
) -> None:
    reservation = _reservation()
    mutation(reservation)

    with pytest.raises(PilotInterfaceGateError, match=message):
        completion_capacity_gate(
            call_kind="action",
            wire_cap_tokens=4096,
            reservation=reservation,
            minimum_sample_count=12,
        )


def test_full_interface_gate_requires_every_registered_terminal_sample() -> None:
    result = interface_sample_gate(
        call_kind="action",
        wire_cap_tokens=4096,
        reservation=_reservation(sample_count=48),
        samples=[_sample() for _ in range(48)],
        expected_sample_count=48,
    )

    assert result.passed is True
    assert result.observed_sample_count == 48
    assert result.maximum_prompt_tokens == 1200
    assert result.failed_sample_indices == ()


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            {
                "finish_reason": "length",
                "response_completed": False,
                "output_disposition": "discarded_incomplete",
                "error_type": "IncompleteCompletionError",
                "reasoning_tokens": 1500,
                "visible_completion_tokens": 0,
            },
            "provider-error",
        ),
        ({"parse_success": False}, "parse-not-exact"),
        ({"prompt_tokens": 200_000}, "prompt-tier-ceiling-reached"),
        (
            {
                "completion_tokens": 4097,
                "reasoning_tokens": 4000,
                "visible_completion_tokens": 97,
            },
            "completion-exceeds-wire-cap",
        ),
    ],
)
def test_full_interface_gate_rejects_terminal_or_tier_failures(
    mutation: dict,
    reason: str,
) -> None:
    samples = [_sample() for _ in range(48)]
    samples[47] = _sample(**mutation)

    result = interface_sample_gate(
        call_kind="action",
        wire_cap_tokens=4096,
        reservation=_reservation(sample_count=48),
        samples=samples,
        expected_sample_count=48,
    )

    assert result.passed is False
    assert reason in result.reasons
    assert result.failed_sample_indices in {(), (47,)}


def test_full_interface_gate_rejects_sample_count_drift() -> None:
    result = interface_sample_gate(
        call_kind="semantic",
        wire_cap_tokens=4096,
        reservation=_reservation(sample_count=14),
        samples=[_sample() for _ in range(13)],
        expected_sample_count=14,
    )

    assert result.passed is False
    assert result.reasons == ("sample-count-mismatch",)


def test_full_interface_gate_rejects_nonadditive_usage() -> None:
    with pytest.raises(PilotInterfaceGateError, match="not additive"):
        interface_sample_gate(
            call_kind="action",
            wire_cap_tokens=4096,
            reservation=_reservation(sample_count=48),
            samples=[
                _sample(
                    completion_tokens=1500,
                    reasoning_tokens=1499,
                    visible_completion_tokens=0,
                )
                for _ in range(48)
            ],
            expected_sample_count=48,
        )
