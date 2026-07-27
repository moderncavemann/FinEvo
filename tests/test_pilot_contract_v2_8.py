from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_7,
    PILOT_CONTRACT_ID_V2_8,
    PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_8,
    PILOT_CONTRACT_TAG_V2_8,
    PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256,
    PILOT_CONTRACT_V2_7_CANONICAL_SHA256,
    PILOT_CONTRACT_V2_8_CANONICAL_SHA256,
    PILOT_V2_8_SOURCE_MANIFEST_CONTENT_SHA256,
    PILOT_V2_8_SOURCE_MANIFEST_FILE_SHA256,
    PilotContractError,
    canonical_contract_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
V27_PATH = EXPERIMENTS / "pilot_v2_7.yaml"
OVERLAY_PATH = EXPERIMENTS / "pilot_v2_8_overlay.yaml"
FULL_PATH = EXPERIMENTS / "pilot_v2_8.yaml"
DEPENDENCIES = (
    V27_PATH,
    EXPERIMENTS / "pilot_v2_4_parent_source_manifest.json",
    EXPERIMENTS / "pilot_v2_5_source_manifest.json",
    EXPERIMENTS / "pilot_v2_6_source_manifest.json",
    EXPERIMENTS / "pilot_v2_7_source_manifest.json",
    EXPERIMENTS / "pilot_v2_8_source_manifest.json",
)


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
    for source in DEPENDENCIES:
        (tmp_path / source.name).write_bytes(source.read_bytes())
    path = tmp_path / "pilot_v2_8_overlay.yaml"
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_v2_8_frozen_overlay_and_expanded_contract_are_identical() -> None:
    source = _overlay_document()
    parent = load_pilot_contract(V27_PATH)
    overlay = load_pilot_contract(OVERLAY_PATH)
    full = load_pilot_contract(FULL_PATH)

    assert source["schema_version"] == (
        PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_8
    )
    assert source["integrity"]["declared_sha256"] == (
        canonical_contract_sha256(source)
    )
    assert parent.contract_id == PILOT_CONTRACT_ID_V2_7
    assert parent.canonical_hash == PILOT_CONTRACT_V2_7_CANONICAL_SHA256
    assert overlay.contract_id == full.contract_id == PILOT_CONTRACT_ID_V2_8
    assert overlay.status == full.status == "frozen"
    assert overlay.to_dict() == full.to_dict()
    assert overlay.canonical_hash == full.canonical_hash
    assert overlay.declared_sha256 == overlay.canonical_hash
    assert overlay.implementation["required_git_tag"] == PILOT_CONTRACT_TAG_V2_8
    assert overlay.release_requirements is not None
    assert overlay.release_requirements.tag == PILOT_CONTRACT_TAG_V2_8
    assert dict(overlay.release_requirements.expected_ci) == {
        "test_count": 995,
        "test_collection_sha256": (
            "033b69bf6a2d38b926cd29d9ae9d568ead210d8eeab81a8110284ef3e764f388"
        ),
        "compiled_source_count": 179,
        "compiled_source_inventory_sha256": (
            "842fbb8d9d3217a3db82e389bafbce410686326af71f884f441915a4a216c133"
        ),
        "sealed_manifest_inventory_sha256": (
            "b5c5a817d09d10752c1f5f00ba556b417d16e06c64b5fcbb15671e49a1d81952"
        ),
    }
    assert overlay.canonical_hash == PILOT_CONTRACT_V2_8_CANONICAL_SHA256
    assert dict(overlay.qref_identity_retry_amendment["source_manifest"]) == {
        "path": "experiments/pilot_v2_8_source_manifest.json",
        "schema_version": "finevo-pilot-v2.8-source-manifest-v1",
        "file_sha256": PILOT_V2_8_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": PILOT_V2_8_SOURCE_MANIFEST_CONTENT_SHA256,
    }
    provenance = overlay.validate_provenance(
        "1" * 40,
        PILOT_CONTRACT_TAG_V2_8,
    )
    assert provenance["resolved_git_commit"] == "1" * 40


def test_v2_8_preserves_the_exact_v2_7_science_design_and_denominator() -> None:
    parent = load_pilot_contract(V27_PATH)
    contract = load_pilot_contract(OVERLAY_PATH)
    parent_specs = parent.expand()
    specs = contract.expand()

    assert len(parent_specs) == len(specs) == 211
    assert _normalized_specs(V27_PATH) == _normalized_specs(OVERLAY_PATH)
    assert Counter(spec.stage_id for spec in specs) == {
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
    assert science_design_sha256(parent.to_dict()) == (
        PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256
    )
    assert science_design_sha256(contract.to_dict()) == (
        PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256
    )
    assert contract.seeds == parent.seeds
    assert contract.arms == parent.arms
    assert contract.provider_profiles == parent.provider_profiles
    assert contract.shocks == parent.shocks
    assert contract.utility == parent.utility
    assert contract.stop_go == parent.stop_go
    assert contract.budgets == parent.budgets
    assert contract.p95_authority_retry_amendment == (
        parent.p95_authority_retry_amendment
    )
    assert contract.stage0_evaluator_retry_amendment == (
        parent.stage0_evaluator_retry_amendment
    )
    assert contract.denominator_policy is not None
    assert contract.denominator_policy.policy_id == "finevo-pilot-v2.8-itt"


def test_v2_8_binds_the_exact_v2_7_no_go_and_evidence_lineage() -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    amendment = contract.qref_identity_retry_amendment

    assert amendment is not None
    failure = amendment["failure_classification"]
    assert failure["parent_contract_id"] == PILOT_CONTRACT_ID_V2_7
    assert failure["parent_contract_sha256"] == (
        PILOT_CONTRACT_V2_7_CANONICAL_SHA256
    )
    assert failure["parent_release_commit"] == (
        "60566410f38f7842169e93ae9822f180235b60b6"
    )
    assert failure["parent_evidence_commit"] == (
        "f15a26418264b5de31f53dbe7c46c1949761fcb6"
    )
    assert failure["parent_evidence_merge_commit"] == (
        "e951aa865186a7c2e841316fc6bb08a716aeaf80"
    )
    assert failure["status_counts"] == {
        "complete": 1,
        "integrity-stopped": 210,
    }
    assert failure["completed_cell_breakdown"] == {
        "parent-import": 1,
        "q-ref-resolution": 0,
        "stage0-calibration": 0,
    }
    assert failure["root_cause_code"] == (
        "qref-contract-cell-id-conflated-with-runner-execution-id"
    )
    assert failure["hosted_provider_calls"] == 0
    assert failure["a_d_treatment_effect_outcomes_generated"] is False
    assert failure["a_d_treatment_effect_outcomes_inspected"] is False

    boundary = amendment["observation_boundary"]
    assert boundary["q_ref_identity_failure_observed_before_amendment"]
    assert boundary["stage0_calibration_selection_observed_before_amendment"]
    assert boundary["stage0_candidate_winner_may_have_been_observed"]
    assert boundary["a_d_treatment_effect_outcomes_observed"] is False
    assert boundary["amendment_is_outcome_blind_with_respect_to_a_d_effects"]

    evidence = amendment["evidence_lineage"]
    assert evidence["parent_evidence_status"] == "complete-with-no-go"
    assert evidence["parent_evidence_merge_commit"] == (
        "e951aa865186a7c2e841316fc6bb08a716aeaf80"
    )
    assert evidence["parent_evidence_rewrite"] == "forbidden"
    assert evidence["parent_claim_reclassification"] == "forbidden"


def test_v2_8_freezes_fresh_qref_imported_stage0_and_fresh_a_d_roles() -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    amendment = contract.qref_identity_retry_amendment

    assert amendment is not None
    q_ref_spec = contract.expand(stage="q-ref-resolution")[0]
    assert q_ref_spec.run_id == (
        "finevo-pilot-v2.8--q-ref-resolution--qref_scripted--"
        "qref-scripted--none--provider-preflight-default--s2010922376"
    )
    q_ref = amendment["q_ref_regeneration"]
    assert q_ref["source_result_reuse"] == "forbidden"
    assert q_ref["fresh_zero_hosted_provider_regeneration"] is True
    assert q_ref["hosted_provider_construction_during_regeneration"] is False
    assert q_ref["scripted_diagnostic_provider_required"] is True
    assert q_ref["hosted_provider_calls"] == 0
    assert q_ref["scripted_diagnostic_calls"] == 48
    assert q_ref["hosted_cost_usd"] == 0.0
    assert q_ref["config_run_id_policy"] == (
        "must-equal-current-contract-cell-run-id"
    )

    imported = amendment["stage0_import"]
    assert imported["source_via_v2_7_nested_snapshot"] is True
    assert imported["imported_complete_cells"] == 14
    assert imported["imported_cell_breakdown"] == {
        "stage0-calibration": 14,
    }
    assert imported["provider_construction_during_import"] is False
    assert imported["provider_redispatch_for_imported_cells"] == "forbidden"

    fresh = amendment["fresh_science_dispatch"]
    assert fresh["a_d_cells"] == 195
    assert fresh["a_d_provider_dispatch"] == "fresh-only"
    assert fresh["imported_a_d_completions"] == 0
    assert fresh["decoded_completion_reuse"] == "forbidden"


def test_v2_8_retains_the_authorized_cumulative_500_dollar_envelope() -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    amendment = contract.qref_identity_retry_amendment

    assert amendment is not None
    carry = amendment["budget_carry_forward"]
    assert carry["total_cap_usd"] == 500.0
    assert carry["max_provider_completions"] == 7500
    assert carry["max_storage_bytes"] == 5_000_000_000
    assert carry["cumulative_prior"]["cost_usd"] == 3.212770875
    assert carry["cumulative_prior"]["hosted_completions"] == 184
    assert carry["cumulative_prior"]["storage_bytes"] == 32_158_175
    assert carry["cumulative_prior"]["record_sha256"] == (
        "a5caad9515eb797a035c26d32d0a0cf7bfd7f0df210e7362bd3b93da18ff3ff7"
    )
    assert carry["v2_7_incremental"] == {
        "cost_usd": 0.0,
        "hosted_completions": 0,
        "local_model_calls": 0,
    }
    assert carry["budget_reset"] is False
    assert carry["debit_before_new_dispatch"] is True
    assert carry["manual_reserve_automatic_use"] is False
    assert carry["whole_remaining_matrix_projection_required"] is True
    assert carry["projection_reserve_multiplier"] == 1.25
    assert carry["unknown_price_policy"] == "stop-before-dispatch"
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
            lambda value: value["qref_identity_retry_amendment"][
                "failure_classification"
            ].update({"parent_evidence_merge_commit": "0" * 40}),
            "q-ref identity retry amendment drifted",
        ),
        (
            lambda value: value["qref_identity_retry_amendment"][
                "failure_classification"
            ].update({"status_counts": {"complete": 2, "integrity-stopped": 209}}),
            "q-ref identity retry amendment drifted",
        ),
        (
            lambda value: value["qref_identity_retry_amendment"][
                "observation_boundary"
            ].update({"a_d_treatment_effect_outcomes_observed": True}),
            "q-ref identity retry amendment drifted",
        ),
        (
            lambda value: value["qref_identity_retry_amendment"][
                "q_ref_regeneration"
            ].update({"scripted_diagnostic_calls": 0}),
            "q-ref identity retry amendment drifted",
        ),
        (
            lambda value: value["qref_identity_retry_amendment"][
                "q_ref_regeneration"
            ].update({"hosted_provider_calls": 1}),
            "q-ref identity retry amendment drifted",
        ),
        (
            lambda value: value["qref_identity_retry_amendment"][
                "stage0_import"
            ].update({"imported_complete_cells": 15}),
            "q-ref identity retry amendment drifted",
        ),
        (
            lambda value: value["qref_identity_retry_amendment"][
                "budget_carry_forward"
            ].update({"total_cap_usd": 501.0}),
            "q-ref identity retry amendment drifted",
        ),
        (
            lambda value: value["changes"]["denominator_policy"].update(
                {"policy_id": "finevo-pilot-v2.7-itt"}
            ),
            "denominator identifier drifted",
        ),
    ],
)
def test_v2_8_resealed_method_or_provenance_drift_fails(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    value = _overlay_document()
    mutate(value)
    path = _write_resealed_overlay(tmp_path, value)
    with pytest.raises(PilotContractError, match=message):
        load_pilot_contract(path)


def test_v2_8_frozen_contract_rejects_a_draft_ci_inventory(
    tmp_path: Path,
) -> None:
    value = _overlay_document()
    value["changes"]["release_requirements"]["expected_ci"] = {
        field: None
        for field in value["changes"]["release_requirements"]["expected_ci"]
    }
    path = _write_resealed_overlay(tmp_path, value)
    with pytest.raises(
        PilotContractError,
        match="frozen expected_ci must be exactly all-concrete",
    ):
        load_pilot_contract(path)


def test_v2_8_frozen_contract_requires_exact_tracked_sibling_manifest(
    tmp_path: Path,
) -> None:
    value = _overlay_document()
    path = _write_resealed_overlay(tmp_path, value)
    source = tmp_path / "pilot_v2_8_source_manifest.json"
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(
        PilotContractError,
        match="V2.8 source manifest file hash drifted",
    ):
        load_pilot_contract(path)


def test_v2_7_remains_immutable_and_has_no_qref_identity_amendment() -> None:
    parent = load_pilot_contract(V27_PATH)

    assert parent.contract_id == PILOT_CONTRACT_ID_V2_7
    assert parent.status == "frozen"
    assert parent.canonical_hash == PILOT_CONTRACT_V2_7_CANONICAL_SHA256
    assert parent.qref_identity_retry_amendment is None
