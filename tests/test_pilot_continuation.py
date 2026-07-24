import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from llm_providers import MultiModelLLM
from verified_memory.actions import ActionParseError
from verified_memory.budget import BudgetExceeded, BudgetLimits, RunBudget, UsageRecord
from verified_memory.pilot_checkpoint import build_pilot_checkpoint
import verified_memory.pilot_continuation as continuation_module
from verified_memory.pilot_continuation import (
    DEFAULT_NARRATIVES,
    DEFAULT_TREATMENTS,
    MEMORY_PULSE_CONTRACT,
    MEMORY_PULSE_TREATMENTS,
    PILOT_CONTINUATION_SCHEMA_VERSION,
    PILOT_NARRATIVE_SCHEMA_VERSION,
    SHUFFLE_ALGORITHM,
    PilotContinuationError,
    run_pilot_continuations,
    run_pilot_narratives,
)
from verified_memory.runner import (
    ShockEvent,
    VerifiedRunConfig,
    verify_provider_call_journal,
)
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


class _PositiveCostScriptedProvider(ScriptedDiagnosticProvider):
    def get_structured_completion(self, messages, **kwargs):
        result = super().get_structured_completion(messages, **kwargs)
        return replace(
            result,
            usage=UsageRecord(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                cost_usd=0.0001,
            ),
        )


def _journal_targets(
    root: Path,
    names,
    *,
    prefix: str,
) -> dict[str, dict[str, object]]:
    return {
        str(name): {
            "path": root / f"{prefix}-{name}.json",
            "run_id": f"{prefix}-{name}",
            "contract_hash": None,
        }
        for name in names
    }


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


def test_four_agent_experiment_d_runs_all_matched_branches_without_api(
    tmp_path: Path,
) -> None:
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
        provider_call_journals=_journal_targets(
            tmp_path,
            DEFAULT_TREATMENTS,
            prefix="continuation",
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
    assert result["memory_pulse_contract"] == MEMORY_PULSE_CONTRACT
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
        journal_binding = branch["provider_call_journal"]
        assert journal_binding["enabled"] is True
        assert journal_binding["event_count"] == 48
        assert journal_binding["completion_event_count"] == 24
        assert journal_binding["parse_disposition_event_count"] == 24
        journal = verify_provider_call_journal(
            journal_binding["path"],
            expected_run_id=journal_binding["run_id"],
            expected_contract_hash=None,
            require_terminal_dispositions=True,
        )
        assert len(journal["events"]) == 48
        assert "Synthetic diagnostic action" not in Path(
            journal_binding["path"]
        ).read_text(encoding="utf-8")
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
    for treatment in MEMORY_PULSE_TREATMENTS:
        intervention = branches[treatment]["intervention"]
        pulse = intervention["memory_pulse_binding"]
        assert intervention["pulse_only"] is True
        assert intervention["decision_t"] == 6
        assert intervention["continuation_horizon_steps"] == 6
        assert pulse["pulse_only"] is True
        assert pulse["duration_decisions"] == 1
        assert pulse["checkpoint_hash"] == result["checkpoint_hash"]
        assert (
            branches[treatment]["trajectory"][0]["memory_pulse_bindings"]["0"]
            == pulse
        )
        assert all(
            row["memory_pulse_bindings"] == {}
            for row in branches[treatment]["trajectory"][1:]
        )
    shuffle = branches["shuffled-episodic"]["intervention"][
        "memory_pulse_binding"
    ]["shuffle_binding"]
    assert shuffle["algorithm"] == SHUFFLE_ALGORITHM
    assert shuffle["non_identity"] is True
    assert shuffle["not_fixed_reversal"] is True
    assert sorted(shuffle["permutation"]) == list(
        range(len(shuffle["permutation"]))
    )
    assert shuffle["shuffled_episode_ids"] == [
        shuffle["original_episode_ids"][index]
        for index in shuffle["permutation"]
    ]
    assert branches["matched-a"]["intervention"]["pulse_only"] is False
    assert branches["erroneous-verified"]["intervention"]["pulse_only"] is False
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
        provider_call_journals=_journal_targets(
            tmp_path,
            DEFAULT_NARRATIVES,
            prefix="narrative",
        ),
    ).to_dict()
    assert narrative["schema_version"] == PILOT_NARRATIVE_SCHEMA_VERSION
    assert narrative["fixtures"] == DEFAULT_NARRATIVES
    assert set(narrative["branches"]) == set(DEFAULT_NARRATIVES)
    assert narrative["narrative_pulse_contract"]["decision_t"] == 6
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
    assert all(
        branch["provider_call_journal"]["event_count"] == 48
        and branch["provider_call_journal"][
            "terminal_dispositions_verified"
        ]
        is True
        for branch in narrative["branches"].values()
    )
    assert narrative["branches"]["none"]["narrative"]["pulse_only"] is False
    assert all(
        narrative["branches"][narrative_id]["narrative"]["pulse_only"]
        is True
        for narrative_id in ("aligned", "paraphrase", "opposite")
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


def test_branch_rng_schedule_is_provider_independent(tmp_path: Path) -> None:
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
            provider_call_journals=_journal_targets(
                tmp_path,
                DEFAULT_TREATMENTS,
                prefix=budget_id,
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
    assert (
        clean["branches"]["shuffled-episodic"]["intervention"][
            "memory_pulse_binding"
        ]["shuffle_binding"]
        == contaminated["branches"]["shuffled-episodic"]["intervention"][
            "memory_pulse_binding"
        ]["shuffle_binding"]
    )


def _continuation_checkpoint(run_id: str) -> object:
    llm = MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=4)
    return build_pilot_checkpoint(
        VerifiedRunConfig(
            run_id=run_id,
            seed=31,
            num_agents=4,
            episode_length=12,
            max_rule_proposals_per_agent=4,
            freeze_new_proposals_after=6,
            shock_schedule=_shock_schedule(),
        ),
        llm=llm,
        budget=RunBudget(
            BudgetLimits(max_calls=40, max_cost_usd=0.01),
            budget_id=f"{run_id}-prefix",
        ),
        env_config_source=ROOT / "config.yaml",
    )


def test_branch_parse_failure_terminalizes_every_dispatched_completion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = _continuation_checkpoint("pilot-continuation-parse-failure")
    journals = _journal_targets(
        tmp_path,
        DEFAULT_TREATMENTS,
        prefix="parse-failure",
    )

    def fail_parse(*args, **kwargs):
        raise ActionParseError("synthetic parse failure")

    monkeypatch.setattr(
        continuation_module,
        "parse_direct_action",
        fail_parse,
    )
    with pytest.raises(PilotContinuationError, match="parse failure"):
        run_pilot_continuations(
            checkpoint,
            llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=4),
            budget=RunBudget(
                BudgetLimits(max_calls=200, max_cost_usd=0.01),
                budget_id="parse-failure-branches",
            ),
            provider_call_journals=journals,
        )

    journal = verify_provider_call_journal(
        journals["matched-a"]["path"],
        expected_run_id=journals["matched-a"]["run_id"],
        expected_contract_hash=None,
        require_terminal_dispositions=True,
    )
    completions = [
        event for event in journal["events"]
        if event["event_type"] == "completion_received"
    ]
    dispositions = [
        event["payload"] for event in journal["events"]
        if event["event_type"] == "parse_disposition"
    ]
    assert len(completions) == len(dispositions) == 4
    assert dispositions[0]["parse_mode"] == "parse_failure"
    assert all(row["accepted"] is False for row in dispositions)


def test_branch_budget_overage_terminalizes_every_dispatched_completion(
    tmp_path: Path,
) -> None:
    checkpoint = _continuation_checkpoint("pilot-continuation-budget-overage")
    journals = _journal_targets(
        tmp_path,
        DEFAULT_TREATMENTS,
        prefix="budget-overage",
    )
    with pytest.raises(BudgetExceeded):
        run_pilot_continuations(
            checkpoint,
            llm=MultiModelLLM(_PositiveCostScriptedProvider(), num_workers=4),
            budget=RunBudget(
                BudgetLimits(max_calls=200, max_cost_usd=0.00015),
                budget_id="budget-overage-branches",
            ),
            provider_call_journals=journals,
        )

    journal = verify_provider_call_journal(
        journals["matched-a"]["path"],
        expected_run_id=journals["matched-a"]["run_id"],
        expected_contract_hash=None,
        require_terminal_dispositions=True,
    )
    completions = [
        event for event in journal["events"]
        if event["event_type"] == "completion_received"
    ]
    dispositions = [
        event["payload"] for event in journal["events"]
        if event["event_type"] == "parse_disposition"
    ]
    assert len(completions) == len(dispositions) == 4
    assert all(row["parse_mode"] == "budget_failure" for row in dispositions)
    assert all(row["accepted"] is False for row in dispositions)
