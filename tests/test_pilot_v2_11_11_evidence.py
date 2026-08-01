from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from verified_memory.pilot_contract import canonical_sha256
from verified_memory.pilot_evidence import (
    PilotEvidenceError,
    _v21111_experiment_d_binding_checks,
    _v21111_recomputed_branch_metrics,
    _validate_v21111_isolated_budget,
    _validate_v21111_trajectory,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _trajectory_fixture():
    previous = {
        str(agent_id): {"state": {"inventory": {"Coin": float(100 + agent_id)}}}
        for agent_id in range(4)
    }
    checkpoint = SimpleNamespace(
        payload={
            "previous_state": {"agents": previous},
            "run_config": {"utility": {"discount_factor": 0.99}},
        }
    )
    wealth = {str(agent_id): float(100 + agent_id) for agent_id in range(4)}
    schedule = []
    trajectory = []
    api_usage = []
    for offset, decision_t in enumerate(range(6, 12)):
        shock = {"decision_t": decision_t, "interest_rate": 0.08}
        schedule.append(shock)
        decisions = {}
        prompt_hashes = {}
        memory_texts = {}
        memory_hashes = {}
        ledger_rows = []
        for agent_id in range(4):
            key = str(agent_id)
            prompt_hash = _hash(f"prompt-{decision_t}-{agent_id}")
            output_hash = _hash(f"output-{decision_t}-{agent_id}")
            memory_text = f"memory-{agent_id}"
            decisions[key] = {
                "clipped": False,
                "raw_output_hash": output_hash,
                "labor_action_index": 1,
                "consumption_action_index": 5,
                "executed_labor_hours": 8.0,
                "executed_consumption_rate": 0.1,
            }
            prompt_hashes[key] = prompt_hash
            memory_texts[key] = memory_text
            memory_hashes[key] = _hash(memory_text)
            flow = float(agent_id + 1)
            next_wealth = wealth[key] + 1.0
            ledger_rows.append(
                {
                    "agent_id": agent_id,
                    "period": decision_t,
                    "budget_residual": 0.0,
                    "flow_utility": flow,
                    "discount_weight": 0.99**decision_t,
                    "discounted_flow_utility": flow * (0.99**decision_t),
                    "wealth_pre": wealth[key],
                    "wealth_post": next_wealth,
                    "cumulative_production_post": float(offset + 1),
                    "executed_labor_hours": 8.0,
                    "executed_consumption_rate": 0.1,
                }
            )
            wealth[key] = next_wealth
            api_usage.append(
                {
                    "decision_t": decision_t,
                    "agent_id": agent_id,
                    "prompt_hash": prompt_hash,
                    "raw_output_hash": output_hash,
                }
            )
        trajectory.append(
            {
                "decision_t": decision_t,
                "outcome_t": decision_t + 1,
                "decisions": decisions,
                "prompt_hashes": prompt_hashes,
                "memory_hashes": memory_hashes,
                "memory_texts": memory_texts,
                "memory_pulse_bindings": {},
                "ledger_rows": ledger_rows,
                "rng_pre_step_hash": _hash(f"rng-{decision_t}"),
                "environment_state_hash": _hash(f"env-{decision_t}"),
                "shock_event": shock,
                "shock_event_hash": canonical_sha256(shock),
                "low_labor_rate": 0.25,
            }
        )
    return checkpoint, schedule, trajectory, api_usage


def test_v21111_trajectory_and_metrics_are_recomputed_from_rows() -> None:
    checkpoint, schedule, trajectory, api_usage = _trajectory_fixture()
    rows, binding = _validate_v21111_trajectory(
        {"trajectory": trajectory},
        api_usage=api_usage,
        shock_schedule=schedule,
    )
    metrics = _v21111_recomputed_branch_metrics(checkpoint, rows)

    assert binding["trajectory_rows_sha256"] == canonical_sha256(trajectory)
    assert metrics["focal"]["final_wealth"] == 106.0
    assert metrics["focal"]["discounted_flow_utility_sum"] == pytest.approx(
        sum(0.99**period for period in range(6, 12))
    )
    assert metrics["population"]["average_final_wealth"] == 107.5
    assert metrics["population"]["flow_utility_sum"] == 60.0

    tampered = deepcopy(trajectory)
    tampered[1]["ledger_rows"][0]["wealth_pre"] += 1.0
    with pytest.raises(PilotEvidenceError, match="continuity drifted"):
        _v21111_recomputed_branch_metrics(checkpoint, tampered)


def test_v21111_isolated_budget_accepts_reordering_and_rejects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projection = {
        "cost_usd": 1.0,
        "basis": {
            "prompt_tokens": 240,
            "completion_tokens": 48,
            "total_tokens": 288,
        },
    }
    monkeypatch.setattr(
        "verified_memory.pilot_evidence._v21111_split_reservation",
        lambda *_args, **_kwargs: projection,
    )
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 1,
        "total_tokens": 11,
        "cost_usd": 0.001,
    }
    estimate = {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "cost_usd": 0.002,
    }
    budget_id = "run-1-budget"
    api_usage = []
    completions = []
    reservation_id = 1
    for decision_t in range(6, 12):
        for agent_id in range(4):
            api_usage.append(
                {
                    "decision_t": decision_t,
                    "agent_id": agent_id,
                    "treatment": "matched-a",
                    "usage": deepcopy(usage),
                }
            )
            completions.append(
                {
                    "budget_id": budget_id,
                    "reservation_id": reservation_id,
                    "label": f"pilot-D:matched-a:t{decision_t}:a{agent_id}",
                    "model": "gpt52_main",
                    "started_elapsed_seconds": 0.0,
                    "finished_elapsed_seconds": 0.1,
                    "elapsed_seconds": 0.1,
                    "estimated_usage": deepcopy(estimate),
                    "usage": deepcopy(usage),
                    "tags": {
                        "call_kind": "pilot_continuation_action",
                        "treatment": "matched-a",
                        "decision_t": str(decision_t),
                        "agent_id": str(agent_id),
                        "batch_index": str(agent_id),
                    },
                }
            )
            reservation_id += 1
    accounted = {
        "prompt_tokens": 240,
        "completion_tokens": 24,
        "total_tokens": 264,
        "cost_usd": sum(0.001 for _ in range(24)),
    }
    budget = {
        "budget_id": budget_id,
        "limits": {
            "max_calls": 24,
            "max_prompt_tokens": 240,
            "max_completion_tokens": 48,
            "max_total_tokens": 288,
            "max_cost_usd": 1.0,
            "max_elapsed_seconds": 3600.0,
        },
        "accounted_usage": accounted,
        "reserved_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        },
        "effective_usage": deepcopy(accounted),
        "completed_calls": 24,
        "active_calls": 0,
        "rolled_back_calls": 0,
        "elapsed_seconds": 1.0,
        "stopped": True,
        "stop_reasons": ["call_limit"],
        "active_reservations": [],
        "completions": list(reversed(completions)),
    }
    result = _validate_v21111_isolated_budget(
        budget,
        contract=object(),
        spec={"run_id": "run-1"},
        api_usage=api_usage,
        raw_root=tmp_path,
        expected_treatment="matched-a",
    )
    assert result["completed_calls"] == 24

    tampered = deepcopy(budget)
    tampered["completions"][0]["reservation_id"] = 1
    with pytest.raises(PilotEvidenceError, match="completions differ"):
        _validate_v21111_isolated_budget(
            tampered,
            contract=object(),
            spec={"run_id": "run-1"},
            api_usage=api_usage,
            raw_root=tmp_path,
            expected_treatment="matched-a",
        )


def _d_binding(arm: str, *, trajectory_hash: str) -> dict:
    treatment = {
        "error-verified": "erroneous-verified",
        "error-unverified": "erroneous-unverified",
    }.get(arm, arm)
    memories = {str(agent_id): _hash(f"memory-{agent_id}") for agent_id in range(4)}
    prompts = {str(agent_id): _hash(f"prompt-{agent_id}") for agent_id in range(4)}
    pulse = None
    if arm in {"no-memory", "shuffled-episodic", "wrong-context"}:
        treated = memories["1"] if arm == "wrong-context" else _hash(arm)
        pulse = {
            "original_memory_hash": memories["0"],
            "treated_memory_hash": treated,
        }
        memories["0"] = treated
        prompts["0"] = _hash(f"prompt-{arm}")
    error_start = (
        {"rule_id": "rule", "forced_active_start_hash": _hash("error")}
        if arm in {"error-verified", "error-unverified"}
        else None
    )
    return {
        "schema_version": "finevo-pilot-v2.11.11-isolated-d-binding-v1",
        "kind": "continuation",
        "treatment_id": treatment,
        "branch_id": treatment,
        "checkpoint_hash": _hash("checkpoint"),
        "checkpoint_file_sha256": _hash("checkpoint-file"),
        "prefix_hash": _hash("prefix"),
        "shock_schedule_hash": _hash("shock"),
        "branch_result_hash": _hash(f"result-{arm}"),
        "branch_source_file_sha256": _hash(f"source-{arm}"),
        "trajectory_rows_sha256": trajectory_hash,
        "proposals_frozen": True,
        "later_memory_pulses_empty": True,
        "pre_generated_rng_hashes": [_hash(f"rng-{i}") for i in range(6)],
        "rng_schedule_binding": {"kind": "bound"},
        "branch_rng_pre_step_hashes": [_hash(f"rng-{i}") for i in range(6)],
        "proposal_counters_before": {str(i): 2 for i in range(4)},
        "proposal_counters_after": {str(i): 2 for i in range(4)},
        "action_grid": {"labor_step_hours": 8.0, "consumption_step": 0.02},
        "prefix_budget_binding": {"budget_id": "prefix"},
        "first_step_memory_text_sha256": memories,
        "first_step_prompt_hashes": prompts,
        "memory_pulse_binding": pulse,
        "error_start": error_start,
    }


def test_v21111_d_binding_preserves_failed_sibling_denominator() -> None:
    arms = (
        "matched-a",
        "matched-b",
        "no-memory",
        "shuffled-episodic",
        "wrong-context",
        "error-verified",
        "error-unverified",
    )
    seeds = (11, 22)
    by_arm = {
        arm: {
            seed: {
                "gate_evidence": {
                    "isolated_branch_binding": _d_binding(
                        arm,
                        trajectory_hash=_hash(
                            f"trajectory-{arm if arm == 'matched-b' else 'a'}"
                        ),
                    )
                }
            }
            for seed in seeds
        }
        for arm in arms
    }
    del by_arm["wrong-context"][22]
    checks = _v21111_experiment_d_binding_checks(
        object(),
        by_arm,
        expected=seeds,
        arms=arms,
        action_grid={"labor_step_hours": 8.0, "consumption_step": 0.02},
    )

    assert checks["11"]["pass"] is True
    assert checks["11"]["matched_a_b_trajectory_equal"] is False
    assert checks["22"]["pass"] is False
    assert checks["22"]["complete_arm_count"] == 6
    assert checks["22"]["errors"] == [
        "wrong-context:missing or non-scientific terminal row"
    ]
