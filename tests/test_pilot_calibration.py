from __future__ import annotations

import hashlib
import math
from pathlib import Path
from statistics import median

import pytest

from llm_providers import MultiModelLLM
from verified_memory.budget import BudgetLimits, RunBudget
from verified_memory.pilot_calibration import (
    PilotCalibrationError,
    Q_REF_STANDALONE_RUN_ID,
    Q_REF_CONSUMPTION_FRACTION_CYCLE,
    Q_REF_EPISODE_LENGTH,
    Q_REF_EXPECTED_ROWS,
    Q_REF_NUM_AGENTS,
    Q_REF_SCHEMA_VERSION,
    Q_REF_SEED,
    Q_REF_WORK_FRACTION_CYCLE,
    STAGE0_OFAT_SCHEMA_VERSION,
    STAGE0_PROFILE_ORDER,
    STAGE0_SELECTION_SCHEMA_VERSION,
    build_q_ref_resolution,
    canonical_hash,
    expand_stage0_ofat,
    q_ref_run_config,
    resolve_q_ref,
    select_stage0_profile,
    stable_baseline_shock_schedule,
)
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.runner import run_verified_experiment
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v1.yaml"
ENVIRONMENT_CONFIG_PATH = ROOT / "config.yaml"


def _stage0_summary(
    *,
    interior_labor: float = 0.80,
    interior_consumption: float = 0.80,
    component_ratio: float = 1.0,
    clipping_count: int = 0,
    treatment_outcome: float = 0.0,
) -> dict[str, object]:
    return {
        "actions": {
            "interior_labor_rate": interior_labor,
            "interior_consumption_rate": interior_consumption,
            "ceiling_labor_rate": 0.0,
            "zero_labor_rate": 0.0,
            "clipping_count": clipping_count,
        },
        "guardrails": {
            "max_abs_budget_residual": 0.0,
            "median_labor_disutility_to_consumption_utility": component_ratio,
        },
        # This deliberately attractive result is outside the selection projection.
        "utility": {
            "treatment_outcome": treatment_outcome,
            "wealth": treatment_outcome,
        },
    }


def _summaries(
    **overrides: dict[str, object],
) -> dict[str, list[dict[str, object]]]:
    result = {
        profile_id: [_stage0_summary()]
        for profile_id in STAGE0_PROFILE_ORDER
    }
    for profile_id, summary in overrides.items():
        result[profile_id] = [summary]
    return result


def test_q_ref_config_freezes_no_context_no_memory_scripted_fixture() -> None:
    config = q_ref_run_config(run_id=Q_REF_STANDALONE_RUN_ID)
    assert config.run_id == Q_REF_STANDALONE_RUN_ID
    assert config.seed == Q_REF_SEED
    assert config.num_agents == Q_REF_NUM_AGENTS
    assert config.episode_length == Q_REF_EPISODE_LENGTH
    assert config.context_mode == "no-context"
    assert config.enable_episodic_retrieval is False
    assert config.enable_semantic is False
    assert config.retrieval_k == 0
    assert config.rule_budget == 0

    shocks = stable_baseline_shock_schedule()
    assert len(shocks) == Q_REF_EPISODE_LENGTH
    assert [event.decision_t for event in shocks] == list(
        range(Q_REF_EPISODE_LENGTH)
    )
    assert {event.phase for event in shocks} == {"baseline"}
    assert {event.interest_rate for event in shocks} == {0.03}
    assert all(event.applied_before_prompt for event in shocks)
    assert all(event.applied_before_step for event in shocks)


def test_q_ref_config_requires_an_explicit_valid_run_id() -> None:
    with pytest.raises(TypeError, match="run_id"):
        q_ref_run_config()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="run_id"):
        q_ref_run_config(run_id=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="run_id"):
        q_ref_run_config(run_id="   ")


def test_q_ref_long_run_id_preserves_frozen_deterministic_semantics() -> None:
    long_run_id = (
        "finevo-pilot-v2.8--q-ref-resolution--qref_scripted--"
        "qref-scripted--none--provider-preflight-default--s2010922376"
    )

    def execute(run_id: str):
        result = run_verified_experiment(
            q_ref_run_config(run_id=run_id),
            llm=MultiModelLLM(
                ScriptedDiagnosticProvider(),
                num_workers=Q_REF_NUM_AGENTS,
            ),
            budget=RunBudget(
                BudgetLimits(
                    max_calls=Q_REF_EXPECTED_ROWS,
                    max_prompt_tokens=500_000,
                    max_completion_tokens=100_000,
                    max_cost_usd=0.01,
                ),
                budget_id=f"{run_id}-budget",
            ),
            env_config_source=ENVIRONMENT_CONFIG_PATH,
        )
        return result, build_q_ref_resolution(
            result,
            contract_hash="a" * 64,
            environment_source_hash=hashlib.sha256(
                ENVIRONMENT_CONFIG_PATH.read_bytes()
            ).hexdigest(),
        )

    legacy_result, legacy_resolution = execute(Q_REF_STANDALONE_RUN_ID)
    long_result, long_resolution = execute(long_run_id)

    for stream_name in ("actions", "utility_ledger", "shock_events"):
        assert long_result.stream(stream_name) == legacy_result.stream(stream_name)
    assert long_resolution["q_ref"] == legacy_resolution["q_ref"]
    assert long_resolution["source"]["utility_ledger"] == (
        legacy_resolution["source"]["utility_ledger"]
    )
    assert long_resolution["run_contract"] == legacy_resolution["run_contract"]
    assert long_resolution["checks"] == legacy_resolution["checks"]

    legacy_config = dict(legacy_result.config)
    long_config = dict(long_result.config)
    assert legacy_config.pop("run_id") == Q_REF_STANDALONE_RUN_ID
    assert long_config.pop("run_id") == long_run_id
    assert long_config == legacy_config
    assert (
        long_resolution["bindings"]["source_config_hash"]
        != legacy_resolution["bindings"]["source_config_hash"]
    )
    assert (
        long_resolution["bindings"]["ledger_hash"]
        == legacy_resolution["bindings"]["ledger_hash"]
    )


def test_q_ref_resolution_is_exact_no_network_and_hash_bound() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    artifact = resolve_q_ref(
        contract_hash=contract.canonical_hash,
        env_config_source=ENVIRONMENT_CONFIG_PATH,
    )

    assert artifact["schema_version"] == Q_REF_SCHEMA_VERSION
    assert artifact["status"] == "pass"
    assert artifact["aggregation"] == "median"
    assert artifact["ledger_field"] == "realized_consumption_quantity"
    assert artifact["row_count"] == Q_REF_EXPECTED_ROWS
    assert math.isfinite(artifact["q_ref"])
    assert artifact["q_ref"] > 0
    assert all(artifact["checks"].values())

    run_contract = artifact["run_contract"]
    assert run_contract["seed"] == Q_REF_SEED
    assert run_contract["num_agents"] == Q_REF_NUM_AGENTS
    assert run_contract["episode_length"] == Q_REF_EPISODE_LENGTH
    assert run_contract["provider_model"] == "diagnostic/scripted-v1"
    assert run_contract["context_mode"] == "no-context"
    assert run_contract["episodic_enabled"] is False
    assert run_contract["semantic_enabled"] is False
    assert tuple(run_contract["work_fraction_cycle"]) == (
        Q_REF_WORK_FRACTION_CYCLE
    )
    assert tuple(run_contract["consumption_fraction_cycle"]) == (
        Q_REF_CONSUMPTION_FRACTION_CYCLE
    )
    assert len(run_contract["shock_schedule"]) == Q_REF_EPISODE_LENGTH
    assert {
        row["interest_rate"] for row in run_contract["shock_schedule"]
    } == {0.03}

    source = artifact["source"]
    assert source["config"]["run_id"] == Q_REF_STANDALONE_RUN_ID
    ledger = source["utility_ledger"]
    assert len(ledger) == Q_REF_EXPECTED_ROWS
    assert artifact["q_ref"] == median(
        row["realized_consumption_quantity"] for row in ledger
    )
    assert source["run_summary"]["diagnostic_only"] is True
    assert source["run_summary"]["scientific_evidence"] is False

    bindings = artifact["bindings"]
    assert bindings["contract_hash"] == contract.canonical_hash
    assert bindings["source_config_hash"] == canonical_hash(source["config"])
    assert bindings["run_summary_hash"] == canonical_hash(
        source["run_summary"]
    )
    assert bindings["ledger_hash"] == canonical_hash(ledger)
    assert bindings["environment_source_hash"] == hashlib.sha256(
        ENVIRONMENT_CONFIG_PATH.read_bytes()
    ).hexdigest()
    assert "not FinEvo treatment-effect" in artifact["evidence_boundary"]


def test_stage0_expansion_is_exact_seven_point_ofat() -> None:
    q_ref = 64.0
    artifact = expand_stage0_ofat(q_ref)
    assert artifact["schema_version"] == STAGE0_OFAT_SCHEMA_VERSION
    assert artifact["design"] == "seven-point-one-factor-at-a-time"
    assert tuple(artifact["profile_order"]) == STAGE0_PROFILE_ORDER
    assert artifact["expansion_hash"] == canonical_hash(
        {
            key: value
            for key, value in artifact.items()
            if key != "expansion_hash"
        }
    )

    profiles = artifact["profiles"]
    assert tuple(profile["profile_id"] for profile in profiles) == (
        STAGE0_PROFILE_ORDER
    )
    center = profiles[0]["utility"]
    assert center["rho"] == 1.0
    assert center["labor_weight"] == 2.0
    assert center["inverse_frisch"] == 1.0
    assert center["consumption_scale"] == q_ref
    assert center["discount_factor"] == 0.99

    expected = {
        "psi-1": ("labor_weight", 1.0),
        "psi-4": ("labor_weight", 4.0),
        "nu-0.5": ("inverse_frisch", 0.5),
        "nu-2": ("inverse_frisch", 2.0),
        "q0-0.5x": ("consumption_scale", 0.5 * q_ref),
        "q0-2x": ("consumption_scale", 2.0 * q_ref),
    }
    calibrated_fields = (
        "labor_weight",
        "inverse_frisch",
        "consumption_scale",
    )
    for profile in profiles:
        assert profile["profile_hash"] == canonical_hash(
            {
                "profile_id": profile["profile_id"],
                "changed_factor": profile["changed_factor"],
                "utility": profile["utility"],
            }
        )
        if profile["profile_id"] == "center":
            assert profile["changed_factor"] == "center"
            assert profile["center_distance"] == 0.0
            continue
        changed_field, changed_value = expected[profile["profile_id"]]
        assert profile["changed_factor"] == changed_field
        assert profile["utility"][changed_field] == changed_value
        assert [
            field
            for field in calibrated_fields
            if profile["utility"][field] != center[field]
        ] == [changed_field]
        assert profile["utility"]["rho"] == center["rho"]
        assert (
            profile["utility"]["discount_factor"]
            == center["discount_factor"]
        )


def test_stage0_selection_applies_guardrail_then_registered_tiebreaks() -> None:
    ofat = expand_stage0_ofat(64.0)
    summaries = _summaries(
        center=_stage0_summary(
            interior_labor=0.95,
            interior_consumption=0.95,
            component_ratio=1.0,
            clipping_count=1,
        ),
        **{
            "psi-1": _stage0_summary(
                interior_labor=0.90,
                interior_consumption=0.90,
                component_ratio=1.5,
            ),
            "psi-4": _stage0_summary(
                interior_labor=0.90,
                interior_consumption=0.90,
                component_ratio=1.0,
            ),
        },
    )

    artifact = select_stage0_profile(ofat, summaries)
    assert artifact["schema_version"] == STAGE0_SELECTION_SCHEMA_VERSION
    # Center has the best coverage but fails the zero-clipping gate.  psi-1 and
    # psi-4 tie on coverage, so the better-balanced psi-4 must win.
    assert artifact["selected_profile_id"] == "psi-4"
    assert artifact["outcome_fields_used"] == []
    by_id = {row["profile_id"]: row for row in artifact["ranking"]}
    assert by_id["center"]["all_seed_gates_pass"] is False
    assert by_id["center"]["rank"] is None
    assert by_id["psi-4"]["rank"] == 1


def test_stage0_selection_uses_center_distance_after_other_ties() -> None:
    ofat = expand_stage0_ofat(64.0)
    artifact = select_stage0_profile(
        ofat,
        _summaries(
            center=_stage0_summary(component_ratio=2.0),
            **{
                "psi-1": _stage0_summary(component_ratio=1.0),
                "psi-4": _stage0_summary(component_ratio=1.0),
            },
        ),
    )
    # psi-1 and psi-4 tie on interior coverage and component balance.  Their
    # normalized center distances are 0.5 and 1.0 respectively.
    assert artifact["selected_profile_id"] == "psi-1"


def test_stage0_selection_is_blind_to_treatment_outcomes() -> None:
    ofat = expand_stage0_ofat(64.0)
    low = _summaries()
    high = _summaries()
    for index, profile_id in enumerate(STAGE0_PROFILE_ORDER):
        low[profile_id][0]["utility"]["treatment_outcome"] = -10_000 + index
        low[profile_id][0]["utility"]["wealth"] = 10_000 - index
        high[profile_id][0]["utility"]["treatment_outcome"] = 10_000 - index
        high[profile_id][0]["utility"]["wealth"] = -10_000 + index

    low_artifact = select_stage0_profile(ofat, low)
    high_artifact = select_stage0_profile(ofat, high)
    assert low_artifact["selected_profile_id"] == "center"
    assert high_artifact["selected_profile_id"] == "center"
    assert (
        low_artifact["selection_input_hash"]
        == high_artifact["selection_input_hash"]
    )
    assert low_artifact["ranking"] == high_artifact["ranking"]


def test_stage0_selection_requires_an_eligible_profile() -> None:
    ofat = expand_stage0_ofat(64.0)
    summaries = {
        profile_id: [_stage0_summary(clipping_count=1)]
        for profile_id in STAGE0_PROFILE_ORDER
    }
    with pytest.raises(PilotCalibrationError, match="no Stage-0 utility profile"):
        select_stage0_profile(ofat, summaries)
