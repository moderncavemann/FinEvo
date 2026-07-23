from pathlib import Path

import pytest

from llm_providers import MultiModelLLM, StructuredCompletion
from verified_memory.budget import BudgetLimits, RunBudget
from verified_memory.foundation_adapter import locate_component
from verified_memory.runner import (
    ERROR_RULE_INJECTION_SCHEMA_VERSION,
    SHOCK_EVENT_SCHEMA_VERSION,
    ShockEvent,
    VerifiedRunConfig,
    VerifiedRunError,
    build_sealed_run_config,
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
    assert result.config == build_sealed_run_config(
        config,
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


def test_runner_can_omit_a_provider_unsupported_decoding_seed() -> None:
    result = run_verified_experiment(
        VerifiedRunConfig(
            run_id="seed-unsupported",
            seed=19,
            num_agents=2,
            episode_length=2,
            enable_semantic=False,
            send_decoding_seed=False,
        ),
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=2),
        budget=RunBudget(BudgetLimits(max_calls=4, max_cost_usd=0.01)),
        env_config_source=ROOT / "config.yaml",
    )

    assert all(
        row["request_seed"] is None for row in result.stream("api_usage")
    )
    assert result.config["seed"] == 19
    assert result.config["send_decoding_seed"] is False


def test_shock_schedule_is_current_period_only_and_applied_to_prompt_and_step() -> None:
    schedule = (
        ShockEvent(decision_t=0, phase="shock", interest_rate=0.08),
        ShockEvent(decision_t=1, phase="recovery", interest_rate=0.03),
    )
    result = run_verified_experiment(
        VerifiedRunConfig(
            run_id="shock-hook",
            episode_length=2,
            enable_semantic=False,
            shock_schedule=schedule,
        ),
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=2),
        budget=RunBudget(BudgetLimits(max_calls=4, max_cost_usd=0.01)),
        env_config_source=ROOT / "config.yaml",
    )

    assert result.stream("shock_events") == tuple(
        event.to_dict() for event in schedule
    )
    assert all(
        set(row)
        == {
            "schema_version",
            "decision_t",
            "phase",
            "interest_rate",
            "applied_before_prompt",
            "applied_before_step",
        }
        and row["schema_version"] == SHOCK_EVENT_SCHEMA_VERSION
        for row in result.stream("shock_events")
    )
    for snapshot in result.stream("decision_snapshots"):
        current = schedule[snapshot["decision_t"]]
        other = schedule[1 - snapshot["decision_t"]]
        assert snapshot["shock_event"] == current.to_dict()
        assert current.phase in snapshot["base_prompt"]
        assert f"{current.interest_rate:.4%}" in snapshot["base_prompt"]
        assert other.phase not in snapshot["base_prompt"]
    assert {
        row["period"]: row["interest_rate"]
        for row in result.stream("utility_ledger")
        if row["agent_id"] == "0"
    } == {0: 0.08, 1: 0.03}
    assert result.validation_status["checks"]["shock_schedule_applied_exactly"]
    assert result.validation_status["checks"]["no_future_shock_in_prompt"]


def test_proposal_freeze_and_full_verifier_config_are_sealed() -> None:
    result = run_verified_experiment(
        VerifiedRunConfig(
            run_id="proposal-freeze",
            episode_length=6,
            max_rule_proposals_per_agent=4,
            freeze_new_proposals_after=3,
            min_candidate_support=2,
            activation_min_support=4,
            activation_min_margin=1.5,
            activation_confidence_threshold=0.65,
            proposal_confidence_floor=0.55,
            retirement_patience=3,
            retirement_confidence_threshold=0.40,
            evidence_weights={
                "support": 1.0,
                "harmful_compliance": 1.0,
                "alternative_success": 0.25,
                "alternative_failure": 0.0,
                "irrelevant": 0.0,
            },
        ),
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=2),
        budget=RunBudget(BudgetLimits(max_calls=16, max_cost_usd=0.01)),
        env_config_source=ROOT / "config.yaml",
    )

    assert len(result.stream("semantic_proposals")) == 2
    assert {row["current_t"] for row in result.stream("semantic_proposals")} == {3}
    verifier = result.config["effective_semantic_verifier"]
    assert verifier["activation_min_support"] == 4
    assert verifier["activation_min_margin"] == 1.5
    assert verifier["activation_confidence_threshold"] == 0.65
    assert verifier["proposal_confidence_floor"] == 0.55
    assert verifier["retirement_patience"] == 3
    assert verifier["retirement_confidence_threshold"] == 0.40
    assert verifier["evidence_weights"]["alternative_success"] == 0.25
    assert result.validation_status["checks"]["proposal_freeze_respected"]


def test_unverified_immediate_activates_valid_candidates_without_evidence_lifecycle() -> None:
    result = run_verified_experiment(
        VerifiedRunConfig(
            run_id="unverified-immediate",
            episode_length=4,
            semantic_policy="unverified-immediate",
        ),
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=2),
        budget=RunBudget(BudgetLimits(max_calls=10, max_cost_usd=0.01)),
        env_config_source=ROOT / "config.yaml",
    )

    assert len(result.stream("semantic_proposals")) == 2
    assert all(
        row["rule_status"] == "active"
        for row in result.stream("semantic_proposals")
    )
    unverified_rules = [
        row
        for row in result.stream("semantic_rules")
        if (row["injection_provenance"] or {}).get("semantic_policy")
        == "unverified-immediate"
    ]
    assert len(unverified_rules) == 2
    assert all(
        row["supporting_episode_ids"] == []
        and row["contradicting_episode_ids"] == []
        and row["status"] == "active"
        for row in unverified_rules
    )
    assert all(
        not row["event_type"].endswith("_evidence_added")
        and row["event_type"] != "rule_retired"
        for row in result.stream("semantic_rule_events")
        if row["rule_id"] in {rule["rule_id"] for rule in unverified_rules}
    )
    assert any(
        row["selected_rule_ids"]
        for row in result.stream("actions")
        if row["decision_t"] == 3
    )
    assert result.validation_status["checks"][
        "unverified_policy_has_no_evidence_or_retirement"
    ]


@pytest.mark.parametrize(
    ("mode", "expected_creation_status", "bypassed"),
    [
        ("candidate-admission", "rejected", False),
        ("forced-active", "active", True),
    ],
)
def test_fixed_erroneous_rule_has_two_auditable_injection_modes(
    mode: str, expected_creation_status: str, bypassed: bool
) -> None:
    result = run_verified_experiment(
        VerifiedRunConfig(
            run_id=f"fixed-error-{mode}",
            episode_length=4,
            error_rule_mode=mode,
            error_rule_injection_t=3,
            freeze_new_proposals_after=2,
        ),
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=2),
        budget=RunBudget(BudgetLimits(max_calls=8, max_cost_usd=0.01)),
        env_config_source=ROOT / "config.yaml",
    )

    rows = result.stream("error_rule_injections")
    assert len(rows) == 2
    assert all(
        row["schema_version"] == ERROR_RULE_INJECTION_SCHEMA_VERSION
        and row["mode"] == mode
        and row["rule_status"] == expected_creation_status
        and row["verifier_bypassed"] is bypassed
        and row["fixed_rule"]["condition"]["field"] == "interest_rate"
        and row["fixed_rule"]["condition"]["operator"] == ">="
        and row["fixed_rule"]["action_guidance"]["target"]
        == "consumption_fraction"
        and row["fixed_rule"]["action_guidance"]["direction"] == "at_most"
        and row["fixed_rule"]["action_guidance"]["threshold"] == 0.0
        for row in rows
    )
    fixed_ids = {row["rule_id"] for row in rows}
    selected = {
        rule_id
        for row in result.stream("actions")
        for rule_id in row["selected_rule_ids"]
    }
    if mode == "candidate-admission":
        assert fixed_ids.isdisjoint(selected)
    else:
        assert fixed_ids <= selected
    assert result.validation_status["checks"]["error_rule_injection_accounted"]


def test_forced_error_rule_starts_identically_but_unverified_arm_is_frozen() -> None:
    def run(policy: str):
        return run_verified_experiment(
            VerifiedRunConfig(
                run_id=f"forced-error-{policy}",
                episode_length=4,
                semantic_policy=policy,
                error_rule_mode="forced-active",
                error_rule_injection_t=3,
                freeze_new_proposals_after=2,
            ),
            llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=2),
            budget=RunBudget(BudgetLimits(max_calls=8, max_cost_usd=0.01)),
            env_config_source=ROOT / "config.yaml",
        )

    verified = run("evidence-grounded")
    unverified = run("unverified-immediate")
    verified_injections = verified.stream("error_rule_injections")
    unverified_injections = unverified.stream("error_rule_injections")
    assert [row["fixed_rule_hash"] for row in verified_injections] == [
        row["fixed_rule_hash"] for row in unverified_injections
    ]
    assert [row["fixed_rule"] for row in verified_injections] == [
        row["fixed_rule"] for row in unverified_injections
    ]
    assert all(row["rule_status"] == "active" for row in verified_injections)
    assert all(row["rule_status"] == "active" for row in unverified_injections)

    verified_ids = {row["rule_id"] for row in verified_injections}
    unverified_ids = {row["rule_id"] for row in unverified_injections}
    verified_creation = [
        row
        for row in verified.stream("semantic_rule_events")
        if row["rule_id"] in verified_ids
        and row["event_type"] == "experimental_rule_injected_active"
    ]
    unverified_creation = [
        row
        for row in unverified.stream("semantic_rule_events")
        if row["rule_id"] in unverified_ids
        and row["event_type"] == "experimental_rule_injected_active"
    ]
    assert all(
        row["provenance"]["evidence_admission"] is True
        and row["provenance"]["retirement_enabled"] is True
        for row in verified_creation
    )
    assert all(
        row["provenance"]["evidence_admission"] is False
        and row["provenance"]["retirement_enabled"] is False
        for row in unverified_creation
    )
    assert any(
        row["event_type"].endswith("_evidence_added")
        for row in verified.stream("semantic_rule_events")
        if row["rule_id"] in verified_ids
    )
    assert all(
        not row["event_type"].endswith("_evidence_added")
        and row["event_type"] != "rule_retired"
        for row in unverified.stream("semantic_rule_events")
        if row["rule_id"] in unverified_ids
    )


def test_scientific_scope_requires_explicit_contract_fields_and_stays_false_for_fixture() -> None:
    schedule = (
        ShockEvent(decision_t=0, phase="pre-shock", interest_rate=0.03),
        ShockEvent(decision_t=1, phase="shock", interest_rate=0.08),
    )
    with pytest.raises(ValueError, match="allow_scientific_scope"):
        VerifiedRunConfig(
            run_id="unauthorized-pilot",
            episode_length=2,
            scientific_scope="preregistered_mechanism_micro_pilot",
            shock_schedule=schedule,
        )
    with pytest.raises(ValueError, match="only valid"):
        VerifiedRunConfig(
            run_id="forged-smoke",
            allow_scientific_scope=True,
            pilot_contract_hash="a" * 64,
            pilot_tag="pilot-v1",
        )

    result = run_verified_experiment(
        VerifiedRunConfig(
            run_id="authorized-fixture",
            episode_length=2,
            enable_semantic=False,
            shock_schedule=schedule,
            scientific_scope="preregistered_mechanism_micro_pilot",
            pilot_contract_hash="a" * 64,
            pilot_tag="pilot-v1",
            allow_scientific_scope=True,
        ),
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=2),
        budget=RunBudget(BudgetLimits(max_calls=4, max_cost_usd=0.01)),
        env_config_source=ROOT / "config.yaml",
    )
    assert result.summary["result_scope"] == "preregistered_mechanism_micro_pilot"
    assert result.summary["diagnostic_only"] is True
    assert result.summary["scientific_evidence"] is False


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
