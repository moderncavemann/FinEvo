from pathlib import Path

import pytest

from llm_providers import MultiModelLLM, StructuredCompletion
from verified_memory.budget import BudgetLimits, RunBudget
from verified_memory.foundation_adapter import locate_component
from verified_memory.runner import (
    VerifiedRunConfig,
    VerifiedRunError,
    run_verified_experiment,
)
from verified_memory.runner_artifacts import (
    load_verified_run_artifacts,
    write_verified_run_artifacts,
)
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


ROOT = Path(__file__).resolve().parents[1]


class MalformedSemanticProvider(ScriptedDiagnosticProvider):
    @staticmethod
    def _proposal(prompt: str) -> str:
        return "malformed semantic candidate"


class FailingSemanticProvider(ScriptedDiagnosticProvider):
    def get_structured_completion(self, messages, **kwargs):
        prompt = self._prompt(messages)
        if "Propose one semantic decision rule" not in prompt:
            return super().get_structured_completion(messages, **kwargs)
        successful_shape = super().get_structured_completion(messages, **kwargs)
        return StructuredCompletion(
            text="Error",
            usage=successful_shape.usage,
            model=successful_shape.model,
            provider=successful_shape.provider,
            attempts=successful_shape.attempts,
            latency_seconds=successful_shape.latency_seconds,
            error_type="SyntheticTransportError",
            request_seed=successful_shape.request_seed,
            response_model=successful_shape.response_model,
        )


class FencedSemanticProvider(ScriptedDiagnosticProvider):
    @staticmethod
    def _proposal(prompt: str) -> str:
        payload = ScriptedDiagnosticProvider._proposal(prompt)
        return f"```json\n{payload}\n```"


class PrefixedSemanticProvider(ScriptedDiagnosticProvider):
    @staticmethod
    def _proposal(prompt: str) -> str:
        payload = ScriptedDiagnosticProvider._proposal(prompt)
        return f"Candidate follows: {payload}"


@pytest.mark.parametrize("consumption_step", [0.02, 0.05])
def test_two_agent_six_month_diagnostic_closes_full_verified_loop(
    consumption_step: float,
) -> None:
    config = VerifiedRunConfig(
        run_id="diagnostic-loop",
        seed=11,
        num_agents=2,
        episode_length=6,
        context_mode="retrieval-only",
        consumption_step=consumption_step,
    )
    budget = RunBudget(
        BudgetLimits(
            max_calls=20,
            max_prompt_tokens=100_000,
            max_completion_tokens=20_000,
            max_cost_usd=0.01,
        ),
        budget_id="diagnostic-loop-budget",
    )
    result = run_verified_experiment(
        config,
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=2),
        budget=budget,
        env_config_source=ROOT / "config.yaml",
    )

    assert result.summary["result_complete"] is True
    assert result.summary["diagnostic_only"] is True
    assert result.summary["scientific_evidence"] is False
    assert result.validation_status["status"] == "pass"
    assert len(result.stream("actions")) == 12
    assert len(result.stream("episodes")) == 12
    assert len(result.stream("utility_ledger")) == 12
    assert len(result.stream("semantic_proposals")) == 2
    assert all(row["outcome_t"] == row["decision_t"] + 1 for row in result.stream("episodes"))
    assert all(abs(row["budget_residual"]) <= 1e-8 for row in result.stream("utility_ledger"))
    assert result.summary["memory_diagnostics"]["semantic_rule_status_counts"]["active"] == 2
    assert result.summary["memory_diagnostics"]["active_rule_retrieval_count"] > 0
    assert result.summary["memory_diagnostics"]["semantic_candidate_parse"][
        "mode_counts"
    ]["exact_json"] == 2
    assert result.summary["action_diagnostics"]["unique_labor_hours"] == [40.0, 88.0, 128.0]
    assert result.summary["action_diagnostics"]["labor_hours_counts"] == {
        "40": 4,
        "88": 6,
        "128": 2,
    }
    assert result.summary["action_diagnostics"]["ceiling_labor_rate"] == 0.0
    assert result.config["foundation_env"]["n_agents"] == 2
    assert result.config["foundation_env"]["episode_length"] == 6
    _, consumption_config = locate_component(
        result.config["foundation_env"], "SimpleConsumption"
    )
    assert consumption_config["consumption_rate_step"] == pytest.approx(
        consumption_step
    )
    assert len(result.config["foundation_env_hash"]) == 64
    assert result.config["registered_outcome_criterion"] == {
        "metric": "utility_advantage",
        "operator": ">",
        "value": 0.0,
        "tolerance": 0.0,
    }
    assert (
        result.summary["memory_diagnostics"]["registered_outcome_criterion"]
        == result.config["registered_outcome_criterion"]
    )
    assert result.config["episodic_prompt_capacity"] == 24
    assert result.config["effective_semantic_verifier"][
        "registered_outcome_criterion"
    ] == result.config["registered_outcome_criterion"]
    assert result.config["effective_semantic_verifier"]["evidence_weights"] == {
        "support": 1.0,
        "harmful_compliance": 1.0,
        "alternative_success": 0.5,
        "alternative_failure": 0.0,
        "irrelevant": 0.0,
    }
    assert all(row["request_seed"] == 11 for row in result.stream("api_usage"))
    assert all(row["response_model"] == "scripted-v1" for row in result.stream("api_usage"))


def test_foundation_minimum_agent_contract_is_explicit() -> None:
    try:
        VerifiedRunConfig(run_id="invalid", num_agents=1)
    except ValueError as exc:
        assert "num_agents >= 2" in str(exc)
    else:
        raise AssertionError("single-agent Foundation run should be rejected")


def test_verified_runner_rejects_hidden_provider_retries() -> None:
    try:
        VerifiedRunConfig(run_id="invalid-retries", max_retries=2)
    except ValueError as exc:
        assert "hard-budget call" in str(exc)
    else:
        raise AssertionError("budgeted verified runner should require one attempt")


def test_verified_runner_rejects_consumption_grids_that_cannot_reach_one() -> None:
    with pytest.raises(ValueError, match="divide one"):
        VerifiedRunConfig(run_id="invalid-consumption-grid", consumption_step=0.03)


@pytest.mark.parametrize(
    ("context_mode", "context_to_prompt"),
    [
        ("no-context", False),
        ("prompt-only", True),
        ("retrieval-only", False),
        ("full", True),
    ],
)
def test_bootstrap_missingness_contract_is_shared_by_all_context_routes(
    context_mode: str, context_to_prompt: bool
) -> None:
    result = run_verified_experiment(
        VerifiedRunConfig(
            run_id=f"bootstrap-{context_mode}",
            episode_length=2,
            context_mode=context_mode,
            enable_semantic=False,
        ),
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=2),
        budget=RunBudget(BudgetLimits(max_calls=4, max_cost_usd=0.01)),
        env_config_source=ROOT / "config.yaml",
    )
    t0_trace = next(
        row
        for row in result.stream("context_trace")
        if row["decision_t"] == 0 and row["agent_id"] == 0
    )
    t0_snapshot = next(
        row
        for row in result.stream("decision_snapshots")
        if row["decision_t"] == 0 and row["agent_id"] == 0
    )
    t0_episode = next(
        row
        for row in result.stream("episodes")
        if row["decision_t"] == 0 and row["agent_id"] == 0
    )
    t1_episode = next(
        row
        for row in result.stream("episodes")
        if row["decision_t"] == 1 and row["agent_id"] == 0
    )

    summary = t0_trace["context_packet"]["prompt_summary"]
    assert "prior_low_labor_rate=unavailable" in summary
    assert "prior_low_labor_rate=0" not in summary
    assert "No completed prior month is available" in t0_snapshot["base_prompt"]
    assert "last completed month you worked 0" not in t0_snapshot["base_prompt"]
    assert bool(t0_snapshot["protected_context_text"]) is context_to_prompt
    assert "low_labor_rate" not in t0_episode["pre_state"]
    assert "unemployment_rate" not in t0_episode["pre_state"]
    assert t0_episode["pre_state"]["low_labor_rate_available"] == 0.0
    assert t1_episode["pre_state"]["low_labor_rate_available"] == 1.0
    assert "low_labor_rate" in t1_episode["pre_state"]
    assert "unemployment_rate" in t1_episode["pre_state"]
    assert (
        result.summary["memory_diagnostics"]["semantic_activation_observed"]
        is False
    )


def test_all_semantic_parse_failures_remain_in_denominator_and_run_seals(
    tmp_path: Path,
) -> None:
    budget = RunBudget(BudgetLimits(max_calls=8, max_cost_usd=0.01))
    result = run_verified_experiment(
        VerifiedRunConfig(run_id="all-parse-fail", episode_length=3),
        llm=MultiModelLLM(MalformedSemanticProvider(), num_workers=2),
        budget=budget,
        env_config_source=ROOT / "config.yaml",
    )

    parse = result.summary["memory_diagnostics"]["semantic_candidate_parse"]
    assert result.summary["result_complete"] is True
    assert result.validation_status["status"] == "pass"
    assert parse == {
        "attempt_count": 2,
        "success_count": 0,
        "failure_count": 2,
        "failure_rate": 1.0,
        "mode_counts": {
            "exact_json": 0,
            "fenced_recovery": 0,
            "substring_recovery": 0,
            "parse_failure": 2,
        },
    }
    assert len(result.stream("semantic_rules")) == 0
    assert len(result.stream("semantic_proposals")) == 2
    assert len(result.stream("errors")) == 2
    assert all(
        row["candidate_parse_status"] == "failure"
        for row in result.stream("semantic_proposals")
    )
    manifest = write_verified_run_artifacts(
        tmp_path / "all-parse-fail",
        result,
        provenance={"purpose": "parse contract test"},
        git_commit="test",
        git_dirty=True,
    )
    assert manifest.exists()
    loaded = load_verified_run_artifacts(tmp_path / "all-parse-fail")
    assert loaded.stream("semantic_rules") == ()
    assert len(loaded.stream("semantic_proposals")) == 2
    assert loaded.summary["validation"]["status"] == "pass"


@pytest.mark.parametrize(
    ("provider", "expected_mode"),
    [
        (FencedSemanticProvider(), "fenced_recovery"),
        (PrefixedSemanticProvider(), "substring_recovery"),
    ],
)
def test_semantic_parse_recovery_modes_are_counted(
    provider: ScriptedDiagnosticProvider, expected_mode: str
) -> None:
    result = run_verified_experiment(
        VerifiedRunConfig(run_id=f"parse-{expected_mode}", episode_length=3),
        llm=MultiModelLLM(provider, num_workers=2),
        budget=RunBudget(BudgetLimits(max_calls=8, max_cost_usd=0.01)),
        env_config_source=ROOT / "config.yaml",
    )
    parse = result.summary["memory_diagnostics"]["semantic_candidate_parse"]
    assert parse["success_count"] == 2
    assert parse["failure_count"] == 0
    assert parse["mode_counts"][expected_mode] == 2


def test_semantic_parse_fail_run_has_structured_failure_details() -> None:
    with pytest.raises(VerifiedRunError) as raised:
        run_verified_experiment(
            VerifiedRunConfig(
                run_id="parse-fail-run",
                episode_length=3,
                semantic_parse_failure_policy="fail-run",
            ),
            llm=MultiModelLLM(MalformedSemanticProvider(), num_workers=2),
            budget=RunBudget(BudgetLimits(max_calls=8, max_cost_usd=0.01)),
            env_config_source=ROOT / "config.yaml",
        )
    details = raised.value.failure.to_dict()
    assert details["error_stage"] == "semantic_candidate_parser"
    assert details["error_type"] == "CandidateParseError"
    assert details["attempts"] == 1
    assert len(details["raw_output_hash"]) == 64


def test_semantic_provider_failure_always_aborts_with_structured_receipt_data() -> None:
    with pytest.raises(VerifiedRunError) as raised:
        run_verified_experiment(
            VerifiedRunConfig(run_id="semantic-provider-fail", episode_length=3),
            llm=MultiModelLLM(FailingSemanticProvider(), num_workers=2),
            budget=RunBudget(BudgetLimits(max_calls=8, max_cost_usd=0.01)),
            env_config_source=ROOT / "config.yaml",
        )
    details = raised.value.failure.to_dict()
    assert details["error_stage"] == "semantic_provider"
    assert details["error_type"] == "SyntheticTransportError"
    assert len(details["prompt_hash"]) == 64
