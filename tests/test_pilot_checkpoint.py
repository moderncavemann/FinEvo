from copy import deepcopy
from pathlib import Path

import pytest

from llm_providers import MultiModelLLM
from verified_memory.budget import BudgetLimits, RunBudget
from verified_memory.pilot_checkpoint import (
    PILOT_CHECKPOINT_SCHEMA_VERSION,
    PilotCheckpoint,
    PilotCheckpointError,
    build_pilot_checkpoint,
    canonical_hash,
    capture_environment_state,
    restore_pilot_checkpoint,
)
from verified_memory.runner import ShockEvent, VerifiedRunConfig
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


ROOT = Path(__file__).resolve().parents[1]


def _shock_schedule() -> tuple[ShockEvent, ...]:
    return tuple(
        ShockEvent(
            decision_t=decision_t,
            phase=(
                "pre-shock"
                if decision_t <= 4
                else "shock"
                if decision_t <= 7
                else "recovery"
            ),
            interest_rate=0.08 if 5 <= decision_t <= 7 else 0.03,
        )
        for decision_t in range(12)
    )


def _build_checkpoint(run_id: str = "pilot-checkpoint-test") -> PilotCheckpoint:
    return build_pilot_checkpoint(
        VerifiedRunConfig(
            run_id=run_id,
            seed=17,
            num_agents=4,
            episode_length=12,
            max_rule_proposals_per_agent=4,
            freeze_new_proposals_after=6,
            shock_schedule=_shock_schedule(),
        ),
        llm=MultiModelLLM(
            ScriptedDiagnosticProvider(), num_workers=4
        ),
        budget=RunBudget(
            BudgetLimits(max_calls=40, max_cost_usd=0.01),
            budget_id=f"{run_id}-budget",
        ),
        env_config_source=ROOT / "config.yaml",
    )


def test_checkpoint_round_trip_replays_exact_rng_environment_memory_and_ledger() -> None:
    checkpoint = _build_checkpoint()

    assert (
        checkpoint.payload["schema_version"]
        == PILOT_CHECKPOINT_SCHEMA_VERSION
    )
    assert checkpoint.next_decision_t == 6
    assert [step["decision_t"] for step in checkpoint.payload["prefix_steps"]] == list(
        range(6)
    )
    assert len(checkpoint.payload["ledger_records"]) == 24
    assert checkpoint.payload["proposals_made"] == {
        "0": 2,
        "1": 2,
        "2": 2,
        "3": 2,
    }
    assert all(
        [
            event["timestamp"]
            for event in memory["semantic"]["events"]
            if event["event_type"]
            in {
                "candidate_verified",
                "candidate_rejected",
                "duplicate_semantic_candidate_ignored",
            }
        ]
        == [3, 6]
        for memory in checkpoint.payload["memories"].values()
    )
    assert all(
        len(step["step_seed_state"]["keys"]) == 624
        for step in checkpoint.payload["prefix_steps"]
    )
    assert all(
        len(step["python_step_seed_state"]["internal"]) == 625
        for step in checkpoint.payload["prefix_steps"]
    )
    assert [
        step["shock_event"]["interest_rate"]
        for step in checkpoint.payload["prefix_steps"]
    ] == [0.03, 0.03, 0.03, 0.03, 0.03, 0.08]
    assert all(
        step["shock_event_hash"] == canonical_hash(step["shock_event"])
        for step in checkpoint.payload["prefix_steps"]
    )

    loaded = PilotCheckpoint.from_dict(checkpoint.to_dict())
    restored_a = restore_pilot_checkpoint(loaded)
    restored_b = restore_pilot_checkpoint(loaded)

    assert capture_environment_state(restored_a.env) == checkpoint.payload[
        "previous_state"
    ]
    assert capture_environment_state(restored_b.env) == checkpoint.payload[
        "previous_state"
    ]
    assert restored_a.ledger.records() == checkpoint.payload["ledger_records"]
    assert restored_b.ledger.records() == checkpoint.payload["ledger_records"]
    assert {
        str(agent_id): memory.to_dict()
        for agent_id, memory in restored_a.memories.items()
    } == checkpoint.payload["memories"]
    assert restored_a.prefix_hash == restored_b.prefix_hash
    assert restored_a.last_decisions.keys() == {"0", "1", "2", "3"}
    assert restored_a.last_transitions.keys() == {"0", "1", "2", "3"}


def test_checkpoint_tampering_and_code_drift_fail_closed() -> None:
    checkpoint = _build_checkpoint("pilot-checkpoint-tamper")

    tampered_prefix = checkpoint.to_dict()
    tampered_prefix["prefix_steps"][0]["foundation_actions"]["0"][0] += 1
    with pytest.raises(PilotCheckpointError, match="checkpoint hash mismatch"):
        PilotCheckpoint.from_dict(tampered_prefix)

    tampered_proposals = checkpoint.to_dict()
    tampered_proposals["proposals_made"]["0"] = 1
    proposal_body = deepcopy(tampered_proposals)
    proposal_body.pop("checkpoint_hash")
    tampered_proposals["checkpoint_hash"] = canonical_hash(proposal_body)
    with pytest.raises(PilotCheckpointError, match="two proposal attempts"):
        PilotCheckpoint.from_dict(tampered_proposals)

    tampered_code = checkpoint.to_dict()
    source_hashes = tampered_code["code_binding"]["source_hashes"]
    source_hashes["verified_memory/actions.py"] = "0" * 64
    tampered_code["code_binding"]["binding_hash"] = canonical_hash(
        {"source_hashes": source_hashes}
    )
    body = deepcopy(tampered_code)
    body.pop("checkpoint_hash")
    tampered_code["checkpoint_hash"] = canonical_hash(body)
    rebound = PilotCheckpoint.from_dict(tampered_code)
    with pytest.raises(PilotCheckpointError, match="code binding"):
        restore_pilot_checkpoint(rebound)
