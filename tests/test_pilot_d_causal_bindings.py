import hashlib

import pytest

from verified_memory.pilot_continuation import (
    MEMORY_PULSE_CONTRACT,
    MEMORY_PULSE_TREATMENTS,
    NARRATIVE_PULSE_CONTRACT,
)
from verified_memory.pilot_orchestrator import (
    PilotOrchestrationError,
    _d_continuation_causal_bindings,
    _d_narrative_causal_bindings,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _rng_binding() -> dict:
    return {
        "schema_version": "finevo-pilot-d-rng-schedule-v1",
        "derivation": "checkpoint-bound-domain-separated-sha256",
        "generated_before_provider_calls": True,
        "source_hash": _digest("source"),
        "schedule_hash": _digest("schedule"),
        "horizon": 6,
    }


def _branch(treatment: str = "matched-a") -> dict:
    hashes = [_digest(f"rng:{offset}") for offset in range(6)]
    pulse_only = treatment in MEMORY_PULSE_TREATMENTS
    intervention = {
        "kind": treatment,
        "pulse_only": pulse_only,
        "memory_pulse_binding": (
            {"kind": treatment} if pulse_only else None
        ),
    }
    if treatment.startswith("erroneous-"):
        intervention["forced_active_start_hash"] = _digest("common-start")
    return {
        "treatment": treatment,
        "shock_schedule_hash": _digest("shock"),
        "rng_pre_step_hashes": hashes,
        "api_usage": [{} for _ in range(24)],
        "provider_call_journal": {
            "enabled": True,
            "journal_sha256": _digest(f"journal:{treatment}"),
        },
        "proposal_counters_before": {
            "0": 2,
            "1": 2,
            "2": 2,
            "3": 2,
        },
        "proposal_counters_after": {
            "0": 2,
            "1": 2,
            "2": 2,
            "3": 2,
        },
        "freeze_proposals": True,
        "intervention": intervention,
        "narrative": {
            "narrative_id": "aligned",
            "text_hash": _digest("aligned text"),
            "pulse_only": True,
        },
    }


def _continuation() -> dict:
    hashes = [_digest(f"rng:{offset}") for offset in range(6)]
    return {
        "checkpoint_hash": _digest("checkpoint"),
        "prefix_hash": _digest("prefix"),
        "pre_generated_rng_hashes": hashes,
        "rng_schedule_binding": _rng_binding(),
        "result_hash": _digest("continuation"),
        "matched_replay_equal": True,
        "focal_agent_id": 0,
        "wrong_context_source_agent_id": 1,
        "memory_pulse_contract": dict(MEMORY_PULSE_CONTRACT),
        "action_grid": {
            "labor_step_hours": 8.0,
            "consumption_step": 0.02,
        },
        "erroneous_forced_active_common_start": {
            "equal": True,
            "forced_active_start_hash": _digest("common-start"),
        },
    }


def test_continuation_binding_carries_exact_shared_and_branch_receipts() -> None:
    continuation = _continuation()
    branch = _branch("erroneous-verified")

    causal = _d_continuation_causal_bindings(continuation, branch)

    assert causal["kind"] == "continuation"
    assert causal["checkpoint_hash"] == continuation["checkpoint_hash"]
    assert causal["shared_result_hash"] == continuation["result_hash"]
    assert causal["pre_generated_rng_hashes"] == branch[
        "rng_pre_step_hashes"
    ]
    assert causal["branch_action_completions"] == 24
    assert causal["branch_provider_call_journal"] == branch[
        "provider_call_journal"
    ]
    assert causal["memory_pulse_contract"] == MEMORY_PULSE_CONTRACT
    assert causal["proposals_frozen"] is True
    assert causal["proposal_counters_before"] == causal[
        "proposal_counters_after"
    ]
    assert (
        causal["branch_forced_active_start_hash"]
        == causal["error_common_start_hash"]
    )


def test_non_error_continuation_has_no_forced_start_hash() -> None:
    causal = _d_continuation_causal_bindings(
        _continuation(),
        _branch("wrong-context"),
    )
    assert causal["branch_forced_active_start_hash"] is None
    assert causal["wrong_context_source_agent_id"] == 1
    assert causal["branch_intervention_pulse_only"] is True
    assert causal["branch_memory_pulse_binding"] == {"kind": "wrong-context"}


def test_narrative_binding_carries_fixture_text_and_rng_receipts() -> None:
    branch = _branch("narrative-aligned")
    narratives = {
        "checkpoint_hash": _digest("checkpoint"),
        "prefix_hash": _digest("prefix"),
        "shock_schedule_hash": _digest("shock"),
        "pre_generated_rng_hashes": branch["rng_pre_step_hashes"],
        "rng_schedule_binding": _rng_binding(),
        "result_hash": _digest("narratives"),
        "fixture_hash": _digest("fixtures"),
        "focal_agent_id": 0,
        "narrative_pulse_contract": dict(NARRATIVE_PULSE_CONTRACT),
        "action_grid": {
            "labor_step_hours": 8.0,
            "consumption_step": 0.02,
        },
    }

    causal = _d_narrative_causal_bindings(narratives, branch)

    assert causal["kind"] == "narrative"
    assert causal["fixture_hash"] == narratives["fixture_hash"]
    assert causal["branch_narrative_id"] == "aligned"
    assert causal["branch_text_hash"] == _digest("aligned text")
    assert causal["branch_action_completions"] == 24
    assert causal["shock_schedule_hash"] == narratives["shock_schedule_hash"]
    assert causal["branch_provider_call_journal"] == branch[
        "provider_call_journal"
    ]
    assert causal["narrative_pulse_contract"] == NARRATIVE_PULSE_CONTRACT
    assert causal["branch_narrative_pulse_only"] is True
    assert causal["pre_generated_rng_hashes"] == causal[
        "branch_rng_pre_step_hashes"
    ]


@pytest.mark.parametrize(
    ("builder", "shared", "branch", "message"),
    [
        (
            _d_continuation_causal_bindings,
            _continuation(),
            {**_branch(), "api_usage": None},
            "completion ledger",
        ),
        (
            _d_continuation_causal_bindings,
            {
                **_continuation(),
                "erroneous_forced_active_common_start": None,
            },
            _branch(),
            "common erroneous-rule start",
        ),
        (
            _d_narrative_causal_bindings,
            {
                "checkpoint_hash": _digest("checkpoint"),
                "prefix_hash": _digest("prefix"),
            },
            {**_branch(), "narrative": None},
            "execution receipts",
        ),
    ],
)
def test_binding_builders_fail_closed_on_missing_execution_receipts(
    builder,
    shared,
    branch,
    message,
) -> None:
    with pytest.raises(PilotOrchestrationError, match=message):
        builder(shared, branch)
