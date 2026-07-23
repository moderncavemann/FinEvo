"""Outcome-blind utility calibration for the FinEvo mechanism micro-pilot.

This module contains two deliberately separate calibration steps:

* resolve ``q_ref`` from one deterministic, no-network Foundation run; and
* expand and select among the seven preregistered one-factor-at-a-time utility
  profiles using action-distribution and utility-component guardrails only.

Neither path consumes a FinEvo treatment contrast, wealth comparison, or other
method-performance outcome when resolving a utility parameter.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from llm_providers import MultiModelLLM

from .budget import BudgetLimits, RunBudget
from .m0_utility import UtilityConfig
from .pilot_analysis import stage0_gate
from .runner import ShockEvent, VerifiedRunConfig, VerifiedRunResult, run_verified_experiment
from .scripted_provider import (
    DIAGNOSTIC_MODEL_NAME,
    DIAGNOSTIC_PROVIDER_NAME,
    ScriptedDiagnosticProvider,
)


Q_REF_SCHEMA_VERSION = "finevo-q-ref-resolution-v1"
STAGE0_OFAT_SCHEMA_VERSION = "finevo-stage0-ofat-v1"
STAGE0_SELECTION_SCHEMA_VERSION = "finevo-stage0-selection-v1"

Q_REF_SEED = 2010922376
Q_REF_NUM_AGENTS = 4
Q_REF_EPISODE_LENGTH = 12
Q_REF_EXPECTED_ROWS = Q_REF_NUM_AGENTS * Q_REF_EPISODE_LENGTH
Q_REF_WORK_FRACTION_CYCLE = (0.25, 0.50, 0.75, 0.50)
Q_REF_CONSUMPTION_FRACTION_CYCLE = (0.30, 0.35, 0.30, 0.25)
Q_REF_LEDGER_FIELD = "realized_consumption_quantity"
Q_REF_AGGREGATION = "median"

STAGE0_PROFILE_ORDER = (
    "center",
    "psi-1",
    "psi-4",
    "nu-0.5",
    "nu-2",
    "q0-0.5x",
    "q0-2x",
)


class PilotCalibrationError(ValueError):
    """Raised when a calibration contract or source artifact is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PilotCalibrationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PilotCalibrationError(f"{name} must be finite")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0:
        raise PilotCalibrationError(f"{name} must be strictly positive")
    return result


def _sha256_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PilotCalibrationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _environment_source_hash(
    source: Mapping[str, Any] | str | Path,
) -> str:
    if isinstance(source, Mapping):
        return canonical_hash(dict(source))
    path = Path(source)
    if not path.is_file():
        raise PilotCalibrationError(f"environment config does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_baseline_shock_schedule() -> tuple[ShockEvent, ...]:
    """Return the frozen 3% baseline event for all twelve q-ref periods."""

    return tuple(
        ShockEvent(
            decision_t=decision_t,
            phase="baseline",
            interest_rate=0.03,
        )
        for decision_t in range(Q_REF_EPISODE_LENGTH)
    )


def q_ref_run_config() -> VerifiedRunConfig:
    """Return the exact deterministic runner configuration for q-ref."""

    return VerifiedRunConfig(
        run_id=f"q-ref-resolution-s{Q_REF_SEED}",
        seed=Q_REF_SEED,
        num_agents=Q_REF_NUM_AGENTS,
        episode_length=Q_REF_EPISODE_LENGTH,
        context_mode="no-context",
        enable_episodic_retrieval=False,
        enable_semantic=False,
        retrieval_k=0,
        rule_budget=0,
        shock_schedule=stable_baseline_shock_schedule(),
        utility=UtilityConfig(
            rho=1.0,
            labor_weight=2.0,
            inverse_frisch=1.0,
            consumption_scale=1.0,
            max_labor_hours=168.0,
            discount_factor=0.99,
            budget_tolerance=1e-8,
        ),
    )


def _validate_q_ref_source(result: VerifiedRunResult) -> dict[str, bool]:
    if not isinstance(result, VerifiedRunResult):
        raise TypeError("result must be a VerifiedRunResult")
    actions = result.stream("actions")
    ledger = result.stream("utility_ledger")
    shocks = result.stream("shock_events")
    semantic_streams_empty = all(
        not result.stream(name)
        for name in (
            "semantic_proposals",
            "semantic_rules",
            "semantic_rule_events",
        )
    )
    expected_action_keys = {
        (decision_t, agent_id)
        for decision_t in range(Q_REF_EPISODE_LENGTH)
        for agent_id in range(Q_REF_NUM_AGENTS)
    }
    action_keys = {
        (row.get("decision_t"), row.get("agent_id")) for row in actions
    }
    ledger_keys = {
        (row.get("period"), int(row.get("agent_id", -1))) for row in ledger
    }
    action_schedule_exact = all(
        math.isclose(
            float(row["decision"]["proposed_work_fraction"]),
            Q_REF_WORK_FRACTION_CYCLE[
                int(row["decision_t"]) % len(Q_REF_WORK_FRACTION_CYCLE)
            ],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(row["decision"]["proposed_consumption_fraction"]),
            Q_REF_CONSUMPTION_FRACTION_CYCLE[
                int(row["decision_t"]) % len(Q_REF_CONSUMPTION_FRACTION_CYCLE)
            ],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in actions
    )
    shock_schedule_exact = tuple(shocks) == tuple(
        event.to_dict() for event in stable_baseline_shock_schedule()
    )
    config = result.config
    checks = {
        "run_complete": result.summary.get("result_complete") is True,
        "validation_pass": result.validation_status.get("status") == "pass",
        "diagnostic_only": result.summary.get("diagnostic_only") is True,
        "not_scientific_evidence": (
            result.summary.get("scientific_evidence") is False
        ),
        "provider_exact": result.summary.get("provider_model")
        == f"{DIAGNOSTIC_PROVIDER_NAME}/{DIAGNOSTIC_MODEL_NAME}",
        "seed_exact": config.get("seed") == Q_REF_SEED,
        "shape_exact": config.get("num_agents") == Q_REF_NUM_AGENTS
        and config.get("episode_length") == Q_REF_EPISODE_LENGTH,
        "no_context": config.get("context_mode") == "no-context",
        "memory_disabled": config.get("enable_episodic_retrieval") is False
        and config.get("enable_semantic") is False
        and config.get("retrieval_k") == 0
        and config.get("rule_budget") == 0,
        "row_count_exact": len(actions) == Q_REF_EXPECTED_ROWS
        and len(ledger) == Q_REF_EXPECTED_ROWS,
        "identity_grid_exact": action_keys == expected_action_keys
        and ledger_keys == expected_action_keys,
        "scripted_action_schedule_exact": action_schedule_exact,
        "shock_schedule_exact": shock_schedule_exact,
        "semantic_streams_empty": semantic_streams_empty,
    }
    return checks


def build_q_ref_resolution(
    result: VerifiedRunResult,
    *,
    contract_hash: str,
    environment_source_hash: str,
) -> dict[str, Any]:
    """Build and validate the hash-bound q-ref resolution artifact."""

    contract_hash = _sha256_digest(contract_hash, "contract_hash")
    environment_source_hash = _sha256_digest(
        environment_source_hash,
        "environment_source_hash",
    )
    checks = _validate_q_ref_source(result)
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise PilotCalibrationError(
            f"q-ref source failed the frozen contract: {failed}"
        )

    ledger = [_json_copy(row) for row in result.stream("utility_ledger")]
    quantities = [
        _finite(row.get(Q_REF_LEDGER_FIELD), Q_REF_LEDGER_FIELD) for row in ledger
    ]
    q_ref = median(quantities)
    checks = {
        **checks,
        "all_ledger_values_finite": len(quantities) == Q_REF_EXPECTED_ROWS,
        "q_ref_finite": math.isfinite(q_ref),
        "q_ref_strictly_positive": q_ref > 0,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise PilotCalibrationError(f"q-ref resolution failed: {failed}")

    source_config = _json_copy(result.config)
    run_summary = _json_copy(result.summary)
    source = {
        "config": source_config,
        "run_summary": run_summary,
        "utility_ledger": ledger,
    }
    return {
        "schema_version": Q_REF_SCHEMA_VERSION,
        "status": "pass",
        "q_ref": q_ref,
        "aggregation": Q_REF_AGGREGATION,
        "ledger_field": Q_REF_LEDGER_FIELD,
        "row_count": len(quantities),
        "run_contract": {
            "seed": Q_REF_SEED,
            "num_agents": Q_REF_NUM_AGENTS,
            "episode_length": Q_REF_EPISODE_LENGTH,
            "provider_model": (
                f"{DIAGNOSTIC_PROVIDER_NAME}/{DIAGNOSTIC_MODEL_NAME}"
            ),
            "context_mode": "no-context",
            "episodic_enabled": False,
            "semantic_enabled": False,
            "work_fraction_cycle": list(Q_REF_WORK_FRACTION_CYCLE),
            "consumption_fraction_cycle": list(
                Q_REF_CONSUMPTION_FRACTION_CYCLE
            ),
            "shock_schedule": [
                event.to_dict() for event in stable_baseline_shock_schedule()
            ],
        },
        "checks": checks,
        "bindings": {
            "contract_hash": contract_hash,
            "source_config_hash": canonical_hash(source_config),
            "run_summary_hash": canonical_hash(run_summary),
            "ledger_hash": canonical_hash(ledger),
            "environment_source_hash": environment_source_hash,
        },
        "source": source,
        "evidence_boundary": (
            "Deterministic no-network scale resolution only; this artifact is "
            "not FinEvo treatment-effect or model-performance evidence."
        ),
    }


def resolve_q_ref(
    *,
    contract_hash: str,
    env_config_source: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Run the frozen no-network q-ref fixture and return its sealed resolution."""

    config = q_ref_run_config()
    result = run_verified_experiment(
        config,
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
            budget_id=f"q-ref-resolution-s{Q_REF_SEED}-budget",
        ),
        env_config_source=env_config_source,
    )
    return build_q_ref_resolution(
        result,
        contract_hash=contract_hash,
        environment_source_hash=_environment_source_hash(env_config_source),
    )


def _profile(
    profile_id: str,
    *,
    changed_factor: str,
    labor_weight: float,
    inverse_frisch: float,
    consumption_scale: float,
) -> dict[str, Any]:
    utility = UtilityConfig(
        rho=1.0,
        labor_weight=labor_weight,
        inverse_frisch=inverse_frisch,
        consumption_scale=consumption_scale,
        max_labor_hours=168.0,
        discount_factor=0.99,
        budget_tolerance=1e-8,
    )
    return {
        "profile_id": profile_id,
        "changed_factor": changed_factor,
        "utility": utility.to_dict(),
    }


def _center_distance(utility: Mapping[str, Any], q_ref: float) -> float:
    labor_weight = _positive(utility.get("labor_weight"), "labor_weight")
    inverse_frisch = _positive(utility.get("inverse_frisch"), "inverse_frisch")
    consumption_scale = _positive(
        utility.get("consumption_scale"),
        "consumption_scale",
    )
    return math.sqrt(
        ((labor_weight - 2.0) / 2.0) ** 2
        + (inverse_frisch - 1.0) ** 2
        + math.log2(consumption_scale / q_ref) ** 2
    )


def expand_stage0_ofat(q_ref: float) -> dict[str, Any]:
    """Expand the exact seven-point, one-factor-at-a-time Stage-0 grid."""

    q_ref = _positive(q_ref, "q_ref")
    profiles = [
        _profile(
            "center",
            changed_factor="center",
            labor_weight=2.0,
            inverse_frisch=1.0,
            consumption_scale=q_ref,
        ),
        _profile(
            "psi-1",
            changed_factor="labor_weight",
            labor_weight=1.0,
            inverse_frisch=1.0,
            consumption_scale=q_ref,
        ),
        _profile(
            "psi-4",
            changed_factor="labor_weight",
            labor_weight=4.0,
            inverse_frisch=1.0,
            consumption_scale=q_ref,
        ),
        _profile(
            "nu-0.5",
            changed_factor="inverse_frisch",
            labor_weight=2.0,
            inverse_frisch=0.5,
            consumption_scale=q_ref,
        ),
        _profile(
            "nu-2",
            changed_factor="inverse_frisch",
            labor_weight=2.0,
            inverse_frisch=2.0,
            consumption_scale=q_ref,
        ),
        _profile(
            "q0-0.5x",
            changed_factor="consumption_scale",
            labor_weight=2.0,
            inverse_frisch=1.0,
            consumption_scale=0.5 * q_ref,
        ),
        _profile(
            "q0-2x",
            changed_factor="consumption_scale",
            labor_weight=2.0,
            inverse_frisch=1.0,
            consumption_scale=2.0 * q_ref,
        ),
    ]
    for profile in profiles:
        profile["center_distance"] = _center_distance(profile["utility"], q_ref)
        profile["profile_hash"] = canonical_hash(
            {
                "profile_id": profile["profile_id"],
                "changed_factor": profile["changed_factor"],
                "utility": profile["utility"],
            }
        )
    payload = {
        "schema_version": STAGE0_OFAT_SCHEMA_VERSION,
        "q_ref": q_ref,
        "design": "seven-point-one-factor-at-a-time",
        "profile_order": list(STAGE0_PROFILE_ORDER),
        "profiles": profiles,
    }
    payload["expansion_hash"] = canonical_hash(payload)
    return payload


def _selection_projection(summary: Mapping[str, Any]) -> dict[str, float]:
    actions = summary.get("actions")
    guardrails = summary.get("guardrails")
    if not isinstance(actions, Mapping) or not isinstance(guardrails, Mapping):
        raise PilotCalibrationError(
            "Stage-0 summary must contain actions and guardrails"
        )
    return {
        "interior_labor_rate": _finite(
            actions.get("interior_labor_rate"),
            "interior_labor_rate",
        ),
        "interior_consumption_rate": _finite(
            actions.get("interior_consumption_rate"),
            "interior_consumption_rate",
        ),
        "ceiling_labor_rate": _finite(
            actions.get("ceiling_labor_rate"),
            "ceiling_labor_rate",
        ),
        "zero_labor_rate": _finite(
            actions.get("zero_labor_rate"),
            "zero_labor_rate",
        ),
        "clipping_count": _finite(
            actions.get("clipping_count"),
            "clipping_count",
        ),
        "max_abs_budget_residual": _finite(
            guardrails.get("max_abs_budget_residual"),
            "max_abs_budget_residual",
        ),
        "component_ratio": _positive(
            guardrails.get(
                "median_labor_disutility_to_consumption_utility"
            ),
            "component_ratio",
        ),
    }


def select_stage0_profile(
    ofat: Mapping[str, Any],
    summaries_by_profile: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Select a Stage-0 utility profile without reading treatment outcomes.

    Profiles must pass every registered guardrail for every supplied calibration
    seed. Eligible profiles are ranked lexicographically by:

    1. higher mean interior action coverage;
    2. lower absolute log distance of the median component ratio from one;
    3. lower normalized distance from the center utility profile; and
    4. frozen OFAT declaration order.
    """

    if not isinstance(ofat, Mapping):
        raise TypeError("ofat must be a mapping")
    if ofat.get("schema_version") != STAGE0_OFAT_SCHEMA_VERSION:
        raise PilotCalibrationError("unsupported Stage-0 OFAT schema")
    q_ref = _positive(ofat.get("q_ref"), "ofat.q_ref")
    raw_profiles = ofat.get("profiles")
    if isinstance(raw_profiles, (str, bytes)) or not isinstance(
        raw_profiles, Sequence
    ):
        raise PilotCalibrationError("ofat.profiles must be a sequence")
    profiles = tuple(raw_profiles)
    profile_ids = tuple(
        str(profile.get("profile_id"))
        for profile in profiles
        if isinstance(profile, Mapping)
    )
    if profile_ids != STAGE0_PROFILE_ORDER or len(profiles) != len(profile_ids):
        raise PilotCalibrationError(
            "Stage-0 profiles do not match the frozen declaration order"
        )
    if set(summaries_by_profile) != set(profile_ids):
        raise PilotCalibrationError(
            "summaries_by_profile must cover exactly the seven OFAT profiles"
        )

    rows: list[dict[str, Any]] = []
    selection_projection: dict[str, Any] = {}
    for declaration_index, profile in enumerate(profiles):
        profile_id = str(profile["profile_id"])
        summaries = summaries_by_profile[profile_id]
        if isinstance(summaries, (str, bytes)) or not isinstance(
            summaries, Sequence
        ) or not summaries:
            raise PilotCalibrationError(
                f"profile {profile_id} requires at least one calibration summary"
            )
        projections = [
            _selection_projection(summary) for summary in summaries
        ]
        gates = [stage0_gate(summary) for summary in summaries]
        all_gates_pass = all(bool(gate["pass"]) for gate in gates)
        interior_coverage = mean(
            (
                projection["interior_labor_rate"]
                + projection["interior_consumption_rate"]
            )
            / 2.0
            for projection in projections
        )
        component_ratio = median(
            projection["component_ratio"] for projection in projections
        )
        component_balance_distance = abs(math.log(component_ratio))
        utility = profile.get("utility")
        if not isinstance(utility, Mapping):
            raise PilotCalibrationError(
                f"profile {profile_id} is missing its utility mapping"
            )
        center_distance = _center_distance(utility, q_ref)
        row = {
            "profile_id": profile_id,
            "declaration_index": declaration_index,
            "all_seed_gates_pass": all_gates_pass,
            "per_seed_gate_pass": [bool(gate["pass"]) for gate in gates],
            "interior_coverage": interior_coverage,
            "component_ratio": component_ratio,
            "component_balance_distance": component_balance_distance,
            "center_distance": center_distance,
        }
        rows.append(row)
        selection_projection[profile_id] = {
            "guardrail_inputs": projections,
            "gate_checks": [gate["checks"] for gate in gates],
        }

    eligible = [row for row in rows if row["all_seed_gates_pass"]]
    if not eligible:
        raise PilotCalibrationError(
            "no Stage-0 utility profile passed all guardrails for all seeds"
        )
    ranked = sorted(
        eligible,
        key=lambda row: (
            -row["interior_coverage"],
            row["component_balance_distance"],
            row["center_distance"],
            row["declaration_index"],
        ),
    )
    rank_by_profile = {
        row["profile_id"]: rank for rank, row in enumerate(ranked, start=1)
    }
    ranked_rows = [
        {**row, "rank": rank_by_profile.get(row["profile_id"])}
        for row in rows
    ]
    return {
        "schema_version": STAGE0_SELECTION_SCHEMA_VERSION,
        "selected_profile_id": ranked[0]["profile_id"],
        "selected_utility": _json_copy(
            next(
                profile["utility"]
                for profile in profiles
                if profile["profile_id"] == ranked[0]["profile_id"]
            )
        ),
        "ranking": ranked_rows,
        "selection_basis": [
            "all registered per-seed guardrails pass",
            "maximize mean interior action coverage",
            "minimize component-balance log distance from one",
            "minimize normalized center distance",
            "frozen OFAT declaration order",
        ],
        "outcome_fields_used": [],
        "selection_input_hash": canonical_hash(selection_projection),
        "evidence_boundary": (
            "Selection uses no FinEvo treatment contrast, wealth, GDP, Gini, "
            "unemployment, or method-performance outcome."
        ),
    }


__all__ = [
    "PilotCalibrationError",
    "Q_REF_AGGREGATION",
    "Q_REF_CONSUMPTION_FRACTION_CYCLE",
    "Q_REF_EPISODE_LENGTH",
    "Q_REF_EXPECTED_ROWS",
    "Q_REF_LEDGER_FIELD",
    "Q_REF_NUM_AGENTS",
    "Q_REF_SCHEMA_VERSION",
    "Q_REF_SEED",
    "Q_REF_WORK_FRACTION_CYCLE",
    "STAGE0_OFAT_SCHEMA_VERSION",
    "STAGE0_PROFILE_ORDER",
    "STAGE0_SELECTION_SCHEMA_VERSION",
    "build_q_ref_resolution",
    "canonical_hash",
    "expand_stage0_ofat",
    "q_ref_run_config",
    "resolve_q_ref",
    "select_stage0_profile",
    "stable_baseline_shock_schedule",
]
