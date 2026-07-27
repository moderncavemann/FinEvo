from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_5,
    PILOT_CONTRACT_ID_V2_6,
    PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_6,
    PILOT_CONTRACT_TAG_V2_6,
    PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256,
    PILOT_CONTRACT_V2_5_CANONICAL_SHA256,
    PILOT_CONTRACT_V2_6_CANONICAL_SHA256,
    PilotContractError,
    canonical_contract_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
V25_PATH = EXPERIMENTS / "pilot_v2_5.yaml"
V24_SOURCE_PATH = EXPERIMENTS / "pilot_v2_4_parent_source_manifest.json"
V25_SOURCE_PATH = EXPERIMENTS / "pilot_v2_5_source_manifest.json"
V26_SOURCE_PATH = EXPERIMENTS / "pilot_v2_6_source_manifest.json"
OVERLAY_PATH = EXPERIMENTS / "pilot_v2_6_overlay.yaml"
FULL_PATH = EXPERIMENTS / "pilot_v2_6.yaml"


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
        V25_PATH,
        V24_SOURCE_PATH,
        V25_SOURCE_PATH,
        V26_SOURCE_PATH,
    ):
        (tmp_path / source.name).write_bytes(source.read_bytes())
    path = tmp_path / "pilot_v2_6_overlay.yaml"
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_v2_6_frozen_overlay_and_expanded_contract_are_identical() -> None:
    source = _overlay_document()
    parent = load_pilot_contract(V25_PATH)
    overlay = load_pilot_contract(OVERLAY_PATH)
    full = load_pilot_contract(FULL_PATH)

    assert source["schema_version"] == (PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_6)
    assert source["integrity"]["declared_sha256"] == (canonical_contract_sha256(source))
    assert parent.contract_id == PILOT_CONTRACT_ID_V2_5
    assert parent.canonical_hash == PILOT_CONTRACT_V2_5_CANONICAL_SHA256
    assert overlay.contract_id == full.contract_id == PILOT_CONTRACT_ID_V2_6
    assert overlay.status == full.status == "frozen"
    assert overlay.to_dict() == full.to_dict()
    assert overlay.canonical_hash == full.canonical_hash
    assert overlay.declared_sha256 == overlay.canonical_hash
    assert overlay.canonical_hash == (
        "bb6b12d71227c423e5a67452dc496f26843dec74e359b9b04bf096dc17d0c509"
    )
    assert overlay.canonical_hash == PILOT_CONTRACT_V2_6_CANONICAL_SHA256
    assert overlay.implementation["required_git_tag"] == PILOT_CONTRACT_TAG_V2_6
    assert overlay.release_requirements is not None
    assert overlay.release_requirements.tag == PILOT_CONTRACT_TAG_V2_6
    assert dict(overlay.release_requirements.expected_ci) == {
        "test_count": 883,
        "test_collection_sha256": (
            "72a639cefda0226dd3ae6493fe4e6bc6bf597e6cc6b9c067c55e8dde376c44c5"
        ),
        "compiled_source_count": 168,
        "compiled_source_inventory_sha256": (
            "d4d233d3f50fc9bf0ed3238b36e05d9933cbed5369a13fb83159f067f4b5b5ce"
        ),
        "sealed_manifest_inventory_sha256": (
            "b5c5a817d09d10752c1f5f00ba556b417d16e06c64b5fcbb15671e49a1d81952"
        ),
    }


def test_v2_6_preserves_exact_v2_5_211_209_science_design() -> None:
    v25 = load_pilot_contract(V25_PATH)
    v26 = load_pilot_contract(OVERLAY_PATH)
    v25_specs = v25.expand()
    v26_specs = v26.expand()

    assert len(v25_specs) == len(v26_specs) == 211
    assert _normalized_specs(V25_PATH) == _normalized_specs(OVERLAY_PATH)
    assert (
        sum(
            spec.stage_id not in {"parent-import", "q-ref-resolution"}
            for spec in v26_specs
        )
        == 209
    )
    assert Counter(spec.stage_id for spec in v26_specs) == {
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
    assert science_design_sha256(v25.to_dict()) == (
        PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256
    )
    assert science_design_sha256(v26.to_dict()) == (
        PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256
    )
    assert v26.seeds == v25.seeds
    assert v26.arms == v25.arms
    assert v26.model_ids == v25.model_ids
    assert v26.provider_profiles == v25.provider_profiles
    assert v26.shocks == v25.shocks
    assert v26.utility == v25.utility
    assert v26.stop_go == v25.stop_go
    assert v26.budgets == v25.budgets
    assert v26.parent_import_retry_amendment == (v25.parent_import_retry_amendment)


def test_v2_6_binds_terminal_v2_5_and_cumulative_budget() -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    amendment = contract.p95_authority_retry_amendment

    assert amendment is not None
    failure = amendment["failure_classification"]
    assert failure["parent_contract_id"] == PILOT_CONTRACT_ID_V2_5
    assert failure["parent_contract_sha256"] == (PILOT_CONTRACT_V2_5_CANONICAL_SHA256)
    assert failure["registered_cells"] == 211
    assert failure["scientific_cells"] == 209
    assert failure["status_counts"] == {
        "complete": 2,
        "failed": 14,
        "integrity-stopped": 195,
    }
    assert failure["provider_calls"] == 0
    assert failure["incremental_cost_usd"] == 0.0
    assert failure["incremental_hosted_completions"] == 0
    assert failure["scientific_effect_outcomes_available"] is False
    assert failure["scientific_effect_outcomes_inspected"] is False

    carry = amendment["budget_carry_forward"]
    assert carry["total_cap_usd"] == 500.0
    assert carry["max_provider_completions"] == 7500
    assert carry["max_storage_bytes"] == 5_000_000_000
    assert carry["cumulative_prior"] == {
        "stage_bucket": "parent_v23",
        "parent_contract_sha256": (
            "1f9809062684a1a2afb96b7342b88a06810e0e87ac883aa63a858a65a81d188d"
        ),
        "parent_run_ledger_sha256": (
            "7d223ddc2cc46b022f051217b9f6767bf9264fb66212b1a63a3498fb6447220f"
        ),
        "parent_budget_ledger_sha256": (
            "7b448a0ebc002b932150c68f2c4e552e940ce186ea5e58afed8673af627d9162"
        ),
        "cost_usd": 3.212770875,
        "hosted_completions": 184,
        "storage_bytes": 6_303_635,
        "record_sha256": (
            "4f445491738ea756280fca0b8c5c82823f4cefe7574cd368ed0c2c51c6a48802"
        ),
    }
    assert carry["v2_5_incremental"] == {
        "cost_usd": 0.0,
        "hosted_completions": 0,
        "raw_file_count": 61,
        "storage_bytes": 1_589_313,
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


def test_v2_6_frozen_source_hashes_and_paid_provenance_are_bound() -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    amendment = contract.p95_authority_retry_amendment

    assert amendment is not None
    assert amendment["source_manifest"] == {
        "path": "experiments/pilot_v2_6_source_manifest.json",
        "schema_version": "finevo-pilot-v2.6-source-manifest-v1",
        "file_sha256": (
            "f84778ed279b8ca98b9b61e26619669fade54b95d0c3e4f17874733acbc84efe"
        ),
        "content_sha256": (
            "78d42a49f16cbbee4fc5e76de17ff26c501a5dcb04a5eb1f79cbe080d2b1b669"
        ),
    }
    assert V26_SOURCE_PATH.is_file()
    provenance = contract.validate_provenance(
        "1" * 40,
        PILOT_CONTRACT_TAG_V2_6,
    )
    assert provenance["resolved_git_commit"] == "1" * 40


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["p95_authority_retry_amendment"][
                "failure_classification"
            ]["status_counts"].update({"failed": 13}),
            "p95-authority retry amendment drifted",
        ),
        (
            lambda value: value["p95_authority_retry_amendment"][
                "budget_carry_forward"
            ]["cumulative_prior"].update({"storage_bytes": 5_712_571}),
            "p95-authority retry amendment drifted",
        ),
        (
            lambda value: value["p95_authority_retry_amendment"][
                "budget_carry_forward"
            ]["cumulative_prior"].update({"stage_bucket": "parent_v25"}),
            "p95-authority retry amendment drifted",
        ),
        (
            lambda value: value["p95_authority_retry_amendment"][
                "correction_policy"
            ].update({"unknown_inherited_schema_fails_closed": False}),
            "p95-authority retry amendment drifted",
        ),
        (
            lambda value: value["p95_authority_retry_amendment"][
                "source_manifest"
            ].update({"file_sha256": "1" * 64}),
            "p95-authority retry amendment drifted",
        ),
        (
            lambda value: value["changes"]["denominator_policy"].update(
                {"policy_id": "finevo-pilot-v2.5-itt"}
            ),
            "denominator identifier drifted",
        ),
        (
            lambda value: value["changes"].update({"budgets": {"total_usd": 501.0}}),
            "invalid V2.6 changes keys",
        ),
    ],
)
def test_v2_6_resealed_operational_or_provenance_drift_fails(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    value = _overlay_document()
    mutate(value)
    path = _write_resealed_overlay(tmp_path, value)
    with pytest.raises(PilotContractError, match=message):
        load_pilot_contract(path)


def test_v2_6_frozen_contract_rejects_a_draft_ci_inventory(
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


def test_v2_5_contract_remains_immutable_and_has_no_v2_6_amendment() -> None:
    contract = load_pilot_contract(V25_PATH)

    assert contract.contract_id == PILOT_CONTRACT_ID_V2_5
    assert contract.canonical_hash == PILOT_CONTRACT_V2_5_CANONICAL_SHA256
    assert len(contract.expand()) == 211
    assert contract.p95_authority_retry_amendment is None
