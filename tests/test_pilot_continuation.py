import random
from dataclasses import replace
from pathlib import Path

import numpy as np

from llm_providers import MultiModelLLM
from verified_memory.budget import BudgetLimits, RunBudget
from verified_memory.pilot_checkpoint import build_pilot_checkpoint
from verified_memory.pilot_continuation import (
    DEFAULT_NARRATIVES,
    DEFAULT_TREATMENTS,
    PILOT_CONTINUATION_SCHEMA_VERSION,
    PILOT_NARRATIVE_SCHEMA_VERSION,
    run_pilot_continuations,
    run_pilot_narratives,
)
from verified_memory.runner import ShockEvent, VerifiedRunConfig
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


ROOT = Path(__file__).resolve().parents[1]


class _RecordingScriptedProvider(ScriptedDiagnosticProvider):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def get_structured_completion(self, messages, **kwargs):
        self.prompts.append(self._prompt(messages))
        return replace(
            super().get_structured_completion(messages, **kwargs),
            request_price_snapshot_source="fixture-price-snapshot",
            request_price_snapshot_captured_at="2026-07-24T00:00:00Z",
        )


class _RngMutatingScriptedProvider(ScriptedDiagnosticProvider):
    """Synthetic provider that deliberately contaminates process-global RNG."""

    def get_structured_completion(self, messages, **kwargs):
        np.random.seed(987654321)
        np.random.random(128)
        random.seed(123456789)
        for _ in range(128):
            random.random()
        return super().get_structured_completion(messages, **kwargs)


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


def test_four_agent_experiment_d_runs_all_matched_branches_without_api() -> None:
    provider = _RecordingScriptedProvider()
    llm = MultiModelLLM(
        provider, num_workers=4
    )
    checkpoint = build_pilot_checkpoint(
        VerifiedRunConfig(
            run_id="pilot-continuation-test",
            seed=23,
            num_agents=4,
            episode_length=12,
            max_rule_proposals_per_agent=4,
            freeze_new_proposals_after=6,
            shock_schedule=_shock_schedule(),
        ),
        llm=llm,
        budget=RunBudget(
            BudgetLimits(max_calls=40, max_cost_usd=0.01),
            budget_id="pilot-continuation-prefix",
        ),
        env_config_source=ROOT / "config.yaml",
    )
    result = run_pilot_continuations(
        checkpoint,
        llm=llm,
        budget=RunBudget(
            BudgetLimits(max_calls=200, max_cost_usd=0.01),
            budget_id="pilot-continuation-branches",
        ),
    ).to_dict()

    assert (
        result["schema_version"]
        == PILOT_CONTINUATION_SCHEMA_VERSION
    )
    assert result["treatments"] == list(DEFAULT_TREATMENTS)
    assert result["branch_after_decision_t"] == 5
    assert result["first_continuation_decision_t"] == 6
    assert result["horizon"] == 6
    assert result["num_agents"] == 4
    assert result["focal_agent_id"] == 0
    assert result["wrong_context_source_agent_id"] == 1
    assert result["action_grid"] == {
        "labor_step_hours": 8.0,
        "consumption_step": 0.02,
    }
    assert result["freeze_proposals"] is True
    assert result["matched_replay_equal"] is True
    assert len(result["pre_generated_rng_hashes"]) == 6
    assert (
        result["rng_schedule_binding"]["generated_before_provider_calls"]
        is True
    )
    assert len(result["rng_schedule_binding"]["schedule_hash"]) == 64

    branches = result["branches"]
    expected_rng = result["pre_generated_rng_hashes"]
    matched_shocks = [
        (
            row["shock_event"],
            row["shock_event_hash"],
            row["shock_prompt_text"],
        )
        for row in branches["matched-a"]["trajectory"]
    ]
    for treatment in DEFAULT_TREATMENTS:
        branch = branches[treatment]
        assert branch["prefix_hash"] == result["prefix_hash"]
        assert branch["checkpoint_hash"] == result["checkpoint_hash"]
        assert branch["rng_pre_step_hashes"] == expected_rng
        assert [
            (
                row["shock_event"],
                row["shock_event_hash"],
                row["shock_prompt_text"],
            )
            for row in branch["trajectory"]
        ] == matched_shocks
        assert branch["proposal_counters_before"] == branch[
            "proposal_counters_after"
        ]
        assert set(branch["proposal_counters_before"].values()) == {2}
        assert len(branch["api_usage"]) == 24
        assert {
            (row["decision_t"], row["agent_id"])
            for row in branch["api_usage"]
        } == {
            (decision_t, agent_id)
            for decision_t in range(6, 12)
            for agent_id in range(4)
        }
        assert all(
            row["treatment"] == treatment
            and row["call_kind"] == "pilot_continuation_action"
            and row["usage"]["completion_tokens"] > 0
            and row["request_price_snapshot_source"]
            == "fixture-price-snapshot"
            and row["request_price_snapshot_captured_at"]
            == "2026-07-24T00:00:00Z"
            for row in branch["api_usage"]
        )
        assert [row["decision_t"] for row in branch["trajectory"]] == list(
            range(6, 12)
        )
        assert all(
            set(row["decisions"]) == {"0", "1", "2", "3"}
            for row in branch["trajectory"]
        )
        assert set(branch["metrics"]) == {"focal", "population"}
        assert branch["metrics"]["population"]["num_agents"] == 4
        assert set(branch["metrics"]["focal"]["first_step"]) == {
            "labor_hours",
            "consumption_rate",
            "immediate_flow_utility",
            "next_wealth",
            "next_cumulative_production",
        }
        assert set(branch["metrics"]["population"]["first_step"]) == {
            "average_next_wealth",
            "gini_next_wealth",
            "flow_utility_sum",
            "low_labor_rate",
        }
        assert {
            "focal_first_labor_hours",
            "focal_first_consumption_rate",
            "focal_immediate_flow_utility",
            "focal_next_wealth",
            "focal_discounted_flow_utility_sum",
            "population_next_average_wealth",
            "population_next_gini",
            "population_next_low_labor_rate",
            "population_first_step_flow_utility_sum",
            "population_final_gini",
            "population_mean_low_labor_rate",
        } <= set(branch["delta_vs_matched_a"])

    assert [
        row["shock_event"]["interest_rate"]
        for row in branches["matched-a"]["trajectory"]
    ] == [0.08, 0.08, 0.03, 0.03, 0.03, 0.03]
    assert [
        row["shock_event"]["phase"]
        for row in branches["matched-a"]["trajectory"]
    ] == ["shock", "shock", "recovery", "recovery", "recovery", "recovery"]
    for row in branches["matched-a"]["trajectory"][:2]:
        assert "recovery" not in row["shock_prompt_text"]
        assert "3.0000%" not in row["shock_prompt_text"]
    for row in branches["matched-a"]["trajectory"][2:]:
        assert "recovery" in row["shock_prompt_text"]
        assert "8.0000%" not in row["shock_prompt_text"]
    action_prompts = [
        prompt
        for prompt in provider.prompts
        if "monthly decision t=" in prompt
    ]
    for decision_t in (6, 7):
        current = [
            prompt
            for prompt in action_prompts
            if f"monthly decision t={decision_t}." in prompt
        ]
        assert current
        assert all("current phase is shock" in prompt for prompt in current)
        assert all("recovery" not in prompt for prompt in current)
        assert all("3.0000%" not in prompt for prompt in current)
    for decision_t in range(8, 12):
        current = [
            prompt
            for prompt in action_prompts
            if f"monthly decision t={decision_t}." in prompt
        ]
        assert current
        assert all("current phase is recovery" in prompt for prompt in current)
        assert all("8.0000%" not in prompt for prompt in current)

    assert (
        branches["matched-a"]["trajectory_hash"]
        == branches["matched-b"]["trajectory_hash"]
    )
    assert (
        branches["no-memory"]["trajectory"][0]["memory_texts"]["0"]
        == ""
    )
    assert (
        branches["wrong-context"]["trajectory"][0]["memory_texts"]["0"]
        == branches["wrong-context"]["trajectory"][0]["memory_texts"]["1"]
    )
    assert (
        branches["shuffled-episodic"]["trajectory"][0]["memory_hashes"]["0"]
        != branches["matched-a"]["trajectory"][0]["memory_hashes"]["0"]
    )
    assert (
        branches["erroneous-verified"]["intervention"]["rule_status"]
        == "active"
    )
    assert (
        branches["erroneous-verified"]["intervention"][
            "verifier_bypassed"
        ]
        is False
    )
    assert (
        branches["erroneous-unverified"]["intervention"]["rule_status"]
        == "active"
    )
    assert (
        branches["erroneous-unverified"]["intervention"][
            "verifier_bypassed"
        ]
        is True
    )
    assert (
        branches["erroneous-verified"]["intervention"]["rule_id"]
        == branches["erroneous-unverified"]["intervention"]["rule_id"]
    )
    assert all(
        branches[name]["intervention"]["forced_active_common_start"] is True
        for name in ("erroneous-verified", "erroneous-unverified")
    )
    verified = branches["erroneous-verified"]["intervention"]
    unverified = branches["erroneous-unverified"]["intervention"]
    assert result["erroneous_forced_active_common_start"]["equal"] is True
    for field in (
        "rule_id",
        "forced_active_rule_hash",
        "forced_active_memory_hash",
        "forced_active_start_hash",
    ):
        assert verified[field] == unverified[field]
        assert (
            result["erroneous_forced_active_common_start"][field]
            == verified[field]
        )
    assert (
        branches["erroneous-verified"]["trajectory"][0]["prompt_hashes"]["0"]
        == branches["erroneous-unverified"]["trajectory"][0][
            "prompt_hashes"
        ]["0"]
    )
    assert verified["final_rule_status"] == "retired"
    assert "rule_retired" in verified["lifecycle_event_types"]
    assert unverified["final_rule_status"] == "active"
    assert "rule_retired" not in unverified["lifecycle_event_types"]
    assert not any(
        event_type.endswith("_evidence_added")
        for event_type in unverified["lifecycle_event_types"]
    )

    narrative = run_pilot_narratives(
        checkpoint,
        llm=llm,
        budget=RunBudget(
            BudgetLimits(max_calls=120, max_cost_usd=0.01),
            budget_id="pilot-narrative-branches",
        ),
    ).to_dict()
    assert narrative["schema_version"] == PILOT_NARRATIVE_SCHEMA_VERSION
    assert narrative["fixtures"] == DEFAULT_NARRATIVES
    assert set(narrative["branches"]) == set(DEFAULT_NARRATIVES)
    assert narrative["semantic_equivalence_within_one_action_bin"]["pass"] is True
    assert narrative["action_grid"] == {
        "labor_step_hours": 8.0,
        "consumption_step": 0.02,
    }
    assert (
        narrative["branches"]["aligned"]["narrative"]["text"]
        == DEFAULT_NARRATIVES["aligned"]
    )
    assert (
        narrative["branches"]["opposite"]["narrative"]["text_hash"]
        != narrative["branches"]["aligned"]["narrative"]["text_hash"]
    )
    assert all(
        branch["rng_pre_step_hashes"]
        == narrative["pre_generated_rng_hashes"]
        for branch in narrative["branches"].values()
    )
    narrative_branches = narrative["branches"]
    assert len(
        {
            branch["trajectory"][0]["prompt_hashes"]["0"]
            for branch in narrative_branches.values()
        }
    ) == len(DEFAULT_NARRATIVES)
    assert len(
        {
            branch["trajectory"][0]["prompt_hashes"]["1"]
            for branch in narrative_branches.values()
        }
    ) == 1
    for offset in range(1, 6):
        assert len(
            {
                branch["trajectory"][offset]["prompt_hashes"]["0"]
                for branch in narrative_branches.values()
            }
        ) == 1
    assert all(
        [
            (
                row["shock_event"],
                row["shock_event_hash"],
                row["shock_prompt_text"],
            )
            for row in branch["trajectory"]
        ]
        == [
            (
                row["shock_event"],
                row["shock_event_hash"],
                row["shock_prompt_text"],
            )
            for row in narrative_branches["none"]["trajectory"]
        ]
        for branch in narrative_branches.values()
    )
    narrative_action_prompts = [
        prompt
        for prompt in provider.prompts
        if "Controlled narrative fixture" in prompt
    ]
    assert len(narrative_action_prompts) == 3
    assert all("monthly decision t=6." in prompt for prompt in narrative_action_prompts)
    for text in DEFAULT_NARRATIVES.values():
        if text:
            assert sum(
                text in prompt for prompt in narrative_action_prompts
            ) == 1


def test_branch_rng_schedule_is_provider_independent() -> None:
    prefix_llm = MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=4)
    checkpoint = build_pilot_checkpoint(
        VerifiedRunConfig(
            run_id="pilot-continuation-rng-independence",
            seed=29,
            num_agents=4,
            episode_length=12,
            max_rule_proposals_per_agent=4,
            freeze_new_proposals_after=6,
            shock_schedule=_shock_schedule(),
        ),
        llm=prefix_llm,
        budget=RunBudget(
            BudgetLimits(max_calls=40, max_cost_usd=0.01),
            budget_id="pilot-continuation-rng-prefix",
        ),
        env_config_source=ROOT / "config.yaml",
    )

    def run(provider, budget_id):
        return run_pilot_continuations(
            checkpoint,
            llm=MultiModelLLM(provider, num_workers=4),
            budget=RunBudget(
                BudgetLimits(max_calls=200, max_cost_usd=0.01),
                budget_id=budget_id,
            ),
        ).to_dict()

    clean = run(
        ScriptedDiagnosticProvider(), "pilot-continuation-rng-clean"
    )
    contaminated = run(
        _RngMutatingScriptedProvider(),
        "pilot-continuation-rng-contaminated",
    )

    assert (
        clean["rng_schedule_binding"]
        == contaminated["rng_schedule_binding"]
    )
    assert (
        clean["pre_generated_rng_hashes"]
        == contaminated["pre_generated_rng_hashes"]
    )
    for treatment in DEFAULT_TREATMENTS:
        clean_branch = clean["branches"][treatment]
        contaminated_branch = contaminated["branches"][treatment]
        assert (
            clean_branch["rng_pre_step_hashes"]
            == contaminated_branch["rng_pre_step_hashes"]
        )
        assert [
            row["environment_state_hash"]
            for row in clean_branch["trajectory"]
        ] == [
            row["environment_state_hash"]
            for row in contaminated_branch["trajectory"]
        ]
        assert (
            clean_branch["trajectory_hash"]
            == contaminated_branch["trajectory_hash"]
        )
