from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_6,
    PILOT_CONTRACT_ID_V2_7,
    PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_7,
    PILOT_CONTRACT_TAG_V2_7,
    PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256,
    PILOT_CONTRACT_V2_6_CANONICAL_SHA256,
    PILOT_CONTRACT_V2_7_CANONICAL_SHA256,
    PilotContractError,
    canonical_contract_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
V26_PATH = EXPERIMENTS / "pilot_v2_6.yaml"
V24_SOURCE_PATH = EXPERIMENTS / "pilot_v2_4_parent_source_manifest.json"
V25_SOURCE_PATH = EXPERIMENTS / "pilot_v2_5_source_manifest.json"
V26_SOURCE_PATH = EXPERIMENTS / "pilot_v2_6_source_manifest.json"
OVERLAY_PATH = EXPERIMENTS / "pilot_v2_7_overlay.yaml"
FULL_PATH = EXPERIMENTS / "pilot_v2_7.yaml"


def _overlay_document() -> dict[str, Any]:
    return json.loads(OVERLAY_PATH.read_text(encoding="utf-8"))


def _normalized_specs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in load_pilot_contract(path).expand():
        row = spec.to_dict()
        row.pop("contract_id")
        row.pop("run_id")
        rows.append(row)
    return rows


def _write_resealed_overlay(
    tmp_path: Path,
    value: dict[str, Any],
) -> Path:
    value["integrity"]["declared_sha256"] = canonical_contract_sha256(value)
    for source in (
        V26_PATH,
        V24_SOURCE_PATH,
        V25_SOURCE_PATH,
        V26_SOURCE_PATH,
    ):
        (tmp_path / source.name).write_bytes(source.read_bytes())
    path = tmp_path / "pilot_v2_7_overlay.yaml"
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_v2_7_draft_overlay_and_expanded_contract_are_identical() -> None:
    source = _overlay_document()
    parent = load_pilot_contract(V26_PATH)
    overlay = load_pilot_contract(OVERLAY_PATH)
    full = load_pilot_contract(FULL_PATH)

    assert source["schema_version"] == (
        PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_7
    )
    assert source["integrity"]["declared_sha256"] == (
        canonical_contract_sha256(source)
    )
    assert parent.contract_id == PILOT_CONTRACT_ID_V2_6
    assert parent.canonical_hash == PILOT_CONTRACT_V2_6_CANONICAL_SHA256
    assert overlay.contract_id == full.contract_id == PILOT_CONTRACT_ID_V2_7
    assert overlay.status == full.status == "draft"
    assert overlay.to_dict() == full.to_dict()
    assert overlay.canonical_hash == full.canonical_hash
    assert overlay.declared_sha256 == overlay.canonical_hash
    assert overlay.canonical_hash == (
        "886c0b820199c092c408dd667acfea6c4f688715aeaf127bf66bd8ab47e2e4f0"
    )
    assert PILOT_CONTRACT_V2_7_CANONICAL_SHA256 is None
    assert overlay.implementation["required_git_tag"] == PILOT_CONTRACT_TAG_V2_7
    assert overlay.release_requirements is not None
    assert overlay.release_requirements.tag == PILOT_CONTRACT_TAG_V2_7
    assert set(overlay.release_requirements.expected_ci.values()) == {None}
    with pytest.raises(
        PilotContractError,
        match="paid provenance cannot be validated from a draft contract",
    ):
        overlay.validate_provenance("1" * 40, PILOT_CONTRACT_TAG_V2_7)


def test_v2_7_preserves_exact_v2_6_211_209_science_design() -> None:
    v26 = load_pilot_contract(V26_PATH)
    v27 = load_pilot_contract(OVERLAY_PATH)
    v26_specs = v26.expand()
    v27_specs = v27.expand()

    assert len(v26_specs) == len(v27_specs) == 211
    assert _normalized_specs(V26_PATH) == _normalized_specs(OVERLAY_PATH)
    assert (
        sum(
            spec.stage_id not in {"parent-import", "q-ref-resolution"}
            for spec in v27_specs
        )
        == 209
    )
    assert Counter(spec.stage_id for spec in v27_specs) == {
        "parent-import": 1,
        "q-ref-resolution": 1,
        "stage0-calibration": 14,
        "local-experiment-c": 25,
        "local-experiment-a": 20,
        "local-experiment-d": 35,
        "local-experiment-b": 25,
        "experiment-c": 25,
        "experiment-a": 20,
        "experiment-d": 30,
        "experiment-b": 15,
    }
    assert science_design_sha256(v26.to_dict()) == (
        PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256
    )
    assert science_design_sha256(v27.to_dict()) == (
        PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256
    )
    assert v27.seeds == v26.seeds
    assert v27.arms == v26.arms
    assert v27.model_ids == v26.model_ids
    assert v27.provider_profiles == v26.provider_profiles
    assert v27.shocks == v26.shocks
    assert v27.utility == v26.utility
    assert v27.stop_go == v26.stop_go
    assert v27.budgets == v26.budgets
    assert v27.p95_authority_retry_amendment == (
        v26.p95_authority_retry_amendment
    )


def test_v2_7_records_observation_boundary_and_exact_import_policy() -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    amendment = contract.stage0_evaluator_retry_amendment

    assert amendment is not None
    failure = amendment["failure_classification"]
    assert failure["parent_contract_id"] == PILOT_CONTRACT_ID_V2_6
    assert failure["parent_contract_sha256"] == (
        PILOT_CONTRACT_V2_6_CANONICAL_SHA256
    )
    assert failure["status_counts"] == {
        "complete": 16,
        "integrity-stopped": 195,
    }
    assert failure["completed_cell_breakdown"] == {
        "parent-import": 1,
        "q-ref-resolution": 1,
        "stage0-calibration": 14,
    }
    assert failure["local_model_calls"] == 672
    assert failure["hosted_provider_calls"] == 0
    assert (
        failure["stage0_calibration_selection_observed_before_amendment"]
        is True
    )
    assert failure["a_d_treatment_effect_outcomes_generated"] is False
    assert failure["a_d_treatment_effect_outcomes_inspected"] is False

    boundary = amendment["observation_boundary"]
    assert boundary["stage0_calibration_selection_observed_before_amendment"]
    assert boundary["amendment_is_outcome_blind_with_respect_to_a_d_effects"]
    assert boundary["calibration_thresholds_unchanged"]
    assert boundary["calibration_tiebreak_order_unchanged"]
    assert boundary["calibration_seed_set_unchanged"]
    assert boundary["calibration_candidate_profiles_unchanged"]

    imported = amendment["artifact_import"]
    assert imported["imported_complete_cells"] == 16
    assert imported["provider_construction_during_import"] is False
    assert imported["provider_redispatch_for_imported_cells"] == "forbidden"
    assert imported["source_raw_namespace"] == (
        "experiment_results/pilot-v2.6/raw"
    )
    assert imported["child_raw_namespace"] == (
        "experiment_results/pilot-v2.7/raw"
    )
    assert imported["shared_namespace"] is False


def test_v2_7_stage0_reader_is_baseline_only_and_phase_agnostic() -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    amendment = contract.stage0_evaluator_retry_amendment

    assert amendment is not None
    reader = amendment["stage0_reader_correction"]
    assert reader["reader_schema"] == "finevo-pilot-stage0-analysis-v1"
    assert reader["reader_scope"] == "stage0-baseline-calibration"
    assert reader["baseline_only_schedule"] is True
    assert reader["phase_agnostic"] is True
    assert reader["pre_shock_phase_required"] is False
    assert reader["shock_phase_required"] is False
    assert reader["recovery_phase_required"] is False
    assert reader["shock_recovery_effect_metrics_computed"] is False
    assert reader["future_treatment_information_allowed"] is False
    assert tuple(reader["inherited_calibration_seeds"]) == (
        1942013315,
        760687867,
    )
    assert tuple(reader["inherited_candidate_profiles"]) == (
        "center",
        "psi-1",
        "psi-4",
        "nu-0.5",
        "nu-2",
        "q0-0.5x",
        "q0-2x",
    )
    assert tuple(reader["allowed_selector_inputs"]) == (
        "max_abs_budget_residual",
        "clipping_count",
        "ceiling_labor_rate",
        "zero_labor_rate",
        "interior_labor_rate",
        "interior_consumption_rate",
        "median_labor_disutility_to_consumption_utility",
    )


def test_v2_7_retains_the_authorized_500_dollar_envelope() -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    amendment = contract.stage0_evaluator_retry_amendment

    assert amendment is not None
    carry = amendment["budget_carry_forward"]
    assert carry["total_cap_usd"] == 500.0
    assert carry["max_provider_completions"] == 7500
    assert carry["max_storage_bytes"] == 5_000_000_000
    assert carry["cumulative_prior"]["cost_usd"] == 3.212770875
    assert carry["cumulative_prior"]["hosted_completions"] == 184
    assert carry["v2_6_incremental"] == {
        "cost_usd": 0.0,
        "hosted_completions": 0,
        "local_model_calls": 672,
    }
    assert carry["budget_reset"] is False
    assert carry["debit_before_new_dispatch"] is True
    assert carry["manual_reserve_automatic_use"] is False
    assert dict(contract.budgets["stage_usd_caps"]) == {
        "parent_v23": 3.212770875,
        "local": 0.0,
        "hosted_confirmatory": 495.787229125,
        "manual_reserve": 1.0,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["stage0_evaluator_retry_amendment"][
                "failure_classification"
            ].update(
                {"stage0_calibration_selection_observed_before_amendment": False}
            ),
            "Stage-0 evaluator retry amendment drifted",
        ),
        (
            lambda value: value["stage0_evaluator_retry_amendment"][
                "observation_boundary"
            ].update({"calibration_thresholds_unchanged": False}),
            "Stage-0 evaluator retry amendment drifted",
        ),
        (
            lambda value: value["stage0_evaluator_retry_amendment"][
                "artifact_import"
            ].update({"provider_redispatch_for_imported_cells": "allowed"}),
            "Stage-0 evaluator retry amendment drifted",
        ),
        (
            lambda value: value["stage0_evaluator_retry_amendment"][
                "stage0_reader_correction"
            ].update({"phase_agnostic": False}),
            "Stage-0 evaluator retry amendment drifted",
        ),
        (
            lambda value: value["stage0_evaluator_retry_amendment"][
                "budget_carry_forward"
            ].update({"total_cap_usd": 501.0}),
            "Stage-0 evaluator retry amendment drifted",
        ),
        (
            lambda value: value["changes"]["denominator_policy"].update(
                {"policy_id": "finevo-pilot-v2.6-itt"}
            ),
            "denominator identifier drifted",
        ),
    ],
)
def test_v2_7_resealed_method_or_provenance_drift_fails(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    value = _overlay_document()
    mutate(value)
    path = _write_resealed_overlay(tmp_path, value)
    with pytest.raises(PilotContractError, match=message):
        load_pilot_contract(path)


def test_v2_7_cannot_be_frozen_before_source_and_ci_are_bound(
    tmp_path: Path,
) -> None:
    value = _overlay_document()
    value["status"] = "frozen"
    value["changes"]["release_requirements"]["expected_ci"] = {
        "test_count": 1,
        "test_collection_sha256": "1" * 64,
        "compiled_source_count": 1,
        "compiled_source_inventory_sha256": "2" * 64,
        "sealed_manifest_inventory_sha256": "3" * 64,
    }
    path = _write_resealed_overlay(tmp_path, value)
    with pytest.raises(
        PilotContractError,
        match="cannot be frozen before its canonical hash and CI inventory",
    ):
        load_pilot_contract(path)


def test_v2_6_remains_immutable_and_has_no_stage0_retry_amendment() -> None:
    contract = load_pilot_contract(V26_PATH)

    assert contract.contract_id == PILOT_CONTRACT_ID_V2_6
    assert contract.canonical_hash == PILOT_CONTRACT_V2_6_CANONICAL_SHA256
    assert len(contract.expand()) == 211
    assert contract.stage0_evaluator_retry_amendment is None
