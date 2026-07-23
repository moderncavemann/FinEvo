from __future__ import annotations

import pytest

from verified_memory.pilot_analysis import (
    continuation_effect_gate,
    paired_delta_summary,
    retrieval_effect_gate,
    stage0_gate,
    summarize_run,
    topk_overlap,
    validate_itt_denominator,
)


def _records() -> dict[str, list[dict]]:
    actions = []
    utility = []
    episodes = []
    traces = []
    for period in range(12):
        phase_penalty = 0.4 if 5 <= period <= 7 else 0.0
        for agent in range(2):
            episode_id = f"ep-{period}-{agent}"
            labor = 80.0 if (period + agent) % 2 else 88.0
            consumption = 0.4 if (period + agent) % 2 else 0.5
            actions.append(
                {
                    "decision_t": period,
                    "agent_id": agent,
                    "decision": {
                        "executed_labor_hours": labor,
                        "executed_consumption_rate": consumption,
                        "clipped": False,
                    },
                    "retrieved_episode_ids": [],
                    "selected_rule_ids": [],
                }
            )
            utility.append(
                {
                    "period": period,
                    "agent_id": str(agent),
                    "flow_utility": 1.0 - phase_penalty,
                    "discounted_flow_utility": (0.99**period)
                    * (1.0 - phase_penalty),
                    "consumption_utility": 1.0,
                    "labor_disutility": 1.0,
                    "budget_residual": 0.0,
                }
            )
            episodes.append(
                {
                    "episode_id": episode_id,
                    "run_id": "analysis-fixture",
                    "seed": 1099057501,
                    "decision_t": period,
                    "agent_id": agent,
                }
            )
            traces.append(
                {
                    "decision_t": period,
                    "agent_id": agent,
                    "retrieved_episode_ids": (
                        [f"ep-{period - 1}-{agent}"] if period else []
                    ),
                }
            )
    return {
        "actions": actions,
        "utility_ledger": utility,
        "episodes": episodes,
        "context_trace": traces,
        "semantic_proposals": [],
        "semantic_rules": [],
        "semantic_rule_events": [],
        "errors": [],
    }


def test_run_summary_and_stage0_gate_are_frozen() -> None:
    summary = summarize_run(_records(), max_labor_hours=168.0)
    assert summary["utility"]["pre_shock_mean"] == pytest.approx(1.0)
    assert summary["utility"]["utility_deficit_auc"] == pytest.approx(1.2)
    assert summary["actions"]["interior_labor_rate"] == 1.0
    assert summary["actions"]["interior_consumption_rate"] == 1.0
    assert stage0_gate(summary)["pass"] is True


def test_paired_effect_gate_retains_all_seed_deltas() -> None:
    treatment = {1: 11.0, 2: 10.0, 3: 12.0, 4: 10.5, 5: 9.0}
    control = {1: 10.0, 2: 9.0, 3: 10.0, 4: 10.0, 5: 10.0}
    summary = paired_delta_summary(treatment, control)
    assert set(summary["raw_paired_deltas"]) == {"1", "2", "3", "4", "5"}
    assert summary["direction_count"] == 4
    assert retrieval_effect_gate(summary)["support_retrieval_effect"] is True


def test_topk_overlap_requires_matched_decision_index() -> None:
    left = _records()
    right = _records()
    overlap = topk_overlap(left, right)
    assert overlap["mean_jaccard"] == 1.0
    right["context_trace"].pop()
    with pytest.raises(ValueError, match="decision index"):
        topk_overlap(left, right)


def test_continuation_gate_uses_null_and_action_bin() -> None:
    result = continuation_effect_gate(
        {1: 9.0, 2: 8.0, 3: 10.0, 4: 7.0, 5: -1.0},
        matched_null_deltas={1: 0.2, 2: -0.1, 3: 0.0, 4: 0.1, 5: -0.2},
        action_bin_width=4.0,
    )
    assert result["passes"] is True


def test_itt_denominator_keeps_failures_and_capability_no_go() -> None:
    result = validate_itt_denominator(
        ["a", "b", "c", "d"],
        [
            {"run_id": "a", "status": "complete"},
            {"run_id": "b", "status": "failed"},
            {"run_id": "c", "status": "budget-stopped"},
            {"run_id": "d", "status": "capability-no-go"},
        ],
    )
    assert result["pass"] is True
    assert result["status_counts"] == {
        "budget-stopped": 1,
        "capability-no-go": 1,
        "complete": 1,
        "failed": 1,
    }


def _rule(
    *,
    agent_id: int,
    family_id: str,
    status: str,
    injected: bool,
) -> dict:
    return {
        "agent_id": agent_id,
        "rule_id": f"{family_id}:v1",
        "rule_family_id": family_id,
        "rule_version": 1,
        "status": status,
        "created_at": 5,
        "updated_at": 7,
        "injected": injected,
        "injection_provenance": (
            {"fixture": "fixed-error"} if injected else None
        ),
    }


def _event(
    *,
    agent_id: int,
    family_id: str,
    event_id: str,
    timestamp: int,
    event_type: str,
    from_status: str | None,
    to_status: str | None,
) -> dict:
    return {
        "agent_id": agent_id,
        "rule_id": f"{family_id}:v1",
        "event_id": event_id,
        "timestamp": timestamp,
        "event_type": event_type,
        "from_status": from_status,
        "to_status": to_status,
        "episode_ids": [f"ep-{timestamp}-{agent_id}"],
    }


def test_rule_reliability_retains_heterogeneous_agent_families() -> None:
    records = _records()
    records["semantic_rules"] = [
        _rule(
            agent_id=0,
            family_id="fixed-rejected",
            status="rejected",
            injected=False,
        ),
        _rule(
            agent_id=1,
            family_id="fixed-active",
            status="retired",
            injected=True,
        ),
    ]
    records["error_rule_injections"] = [
        {
            "agent_id": 0,
            "decision_t": 5,
            "rule_id": "fixed-rejected:v1",
            "mode": "candidate-admission",
            "semantic_policy": "evidence-grounded",
            "fixed_rule_hash": "0" * 64,
            "verifier_bypassed": False,
            "rule_status": "rejected",
        },
        {
            "agent_id": 1,
            "decision_t": 5,
            "rule_id": "fixed-active:v1",
            "mode": "forced-active",
            "semantic_policy": "evidence-grounded",
            "fixed_rule_hash": "0" * 64,
            "verifier_bypassed": True,
            "rule_status": "active",
        },
    ]
    records["semantic_rule_events"] = [
        _event(
            agent_id=0,
            family_id="fixed-rejected",
            event_id="e-rejected",
            timestamp=5,
            event_type="candidate_rejected",
            from_status=None,
            to_status="rejected",
        ),
        _event(
            agent_id=1,
            family_id="fixed-active",
            event_id="e-active",
            timestamp=5,
            event_type="experimental_rule_injected_active",
            from_status=None,
            to_status="active",
        ),
        _event(
            agent_id=1,
            family_id="fixed-active",
            event_id="e-retrieved",
            timestamp=5,
            event_type="active_rule_retrieved",
            from_status="active",
            to_status="active",
        ),
        _event(
            agent_id=1,
            family_id="fixed-active",
            event_id="e-harmful",
            timestamp=6,
            event_type="harmful_compliance_evidence_added",
            from_status="active",
            to_status="active",
        ),
        _event(
            agent_id=1,
            family_id="fixed-active",
            event_id="e-retired",
            timestamp=7,
            event_type="rule_retired",
            from_status="active",
            to_status="retired",
        ),
    ]
    for action in records["actions"]:
        if action["agent_id"] == 1 and action["decision_t"] in {5, 6}:
            action["selected_rule_ids"] = ["fixed-active:v1"]
            action["prompt_hash"] = f"prompt-{action['decision_t']}"
    for trace in records["context_trace"]:
        if trace["agent_id"] == 1 and trace["decision_t"] in {5, 6}:
            trace["context_hash"] = f"context-{trace['decision_t']}"
            trace["context_mode"] = "full"

    summary = summarize_run(records, max_labor_hours=168.0)
    rows = {
        (row["agent_id"], row["rule_family_id"]): row
        for row in summary["rule_reliability"]["by_agent_rule_family"]
    }
    rejected = rows[(0, "fixed-rejected")]
    active = rows[(1, "fixed-active")]

    # The aggregate retains its old "any injected rule" meaning, while the
    # analysis unit does not spread one agent's activation to another agent.
    assert summary["rule_reliability"]["false_rule_ever_active"] is True
    assert rejected["ever_active"] is False
    assert rejected["false_rule_ever_active"] is False
    assert active["ever_active"] is True
    assert active["false_rule_ever_active"] is True
    assert active["first_harmful_compliance_t"] == 6
    assert active["retirement_t"] == 7
    assert active["harmful_to_retirement_delay"] == 1
    assert active["terminal_status"] == "retired"
    assert active["source"] == "injected"
    assert active["run_id"] == "analysis-fixture"
    assert active["seed"] == 1099057501
    assert active["unit_id"].startswith(
        "analysis-fixture:s1099057501:a1:family:"
    )

    # Three lifecycle observations do not become three actor exposures. Only
    # the two action rows that actually selected the rule count.
    assert active["active_exposure_steps"] == 2
    assert [
        row["decision_t"] for row in active["actor_exposure_steps"]
    ] == [5, 6]
    assert active["actor_exposure_steps"][0]["context_hash"] == "context-5"
    assert active["actor_exposure_steps"][0]["flow_utility"] == pytest.approx(0.6)


def test_rule_activation_event_without_actor_selection_is_not_exposure() -> None:
    records = _records()
    records["semantic_rules"] = [
        _rule(
            agent_id=0,
            family_id="natural-active",
            status="active",
            injected=False,
        )
    ]
    records["semantic_rule_events"] = [
        _event(
            agent_id=0,
            family_id="natural-active",
            event_id="e-activated",
            timestamp=5,
            event_type="rule_activated",
            from_status="provisional",
            to_status="active",
        ),
        _event(
            agent_id=0,
            family_id="natural-active",
            event_id="e-retrieved",
            timestamp=6,
            event_type="active_rule_retrieved",
            from_status="active",
            to_status="active",
        ),
    ]

    row = summarize_run(records, max_labor_hours=168.0)[
        "rule_reliability"
    ]["by_agent_rule_family"][0]
    assert row["source"] == "natural"
    assert row["ever_active"] is True
    assert row["active_exposure_steps"] == 0
    assert row["actor_exposure_steps"] == []
