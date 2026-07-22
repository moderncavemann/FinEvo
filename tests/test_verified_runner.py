from pathlib import Path

from llm_providers import MultiModelLLM
from verified_memory.budget import BudgetLimits, RunBudget
from verified_memory.runner import VerifiedRunConfig, run_verified_experiment
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


ROOT = Path(__file__).resolve().parents[1]


def test_two_agent_six_month_diagnostic_closes_full_verified_loop() -> None:
    config = VerifiedRunConfig(
        run_id="diagnostic-loop",
        seed=11,
        num_agents=2,
        episode_length=6,
        context_mode="retrieval-only",
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
    assert result.summary["action_diagnostics"]["unique_labor_hours"] == [40.0, 88.0, 128.0]


def test_foundation_minimum_agent_contract_is_explicit() -> None:
    try:
        VerifiedRunConfig(run_id="invalid", num_agents=1)
    except ValueError as exc:
        assert "num_agents >= 2" in str(exc)
    else:
        raise AssertionError("single-agent Foundation run should be rejected")
