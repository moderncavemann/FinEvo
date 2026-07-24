from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_1,
    PILOT_CONTRACT_ID_V2_2,
    PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_2,
    PILOT_CONTRACT_SCHEMA_VERSION_V2,
    PILOT_CONTRACT_TAG_V2_2,
    PILOT_CONTRACT_V2_1_CANONICAL_SHA256,
    PILOT_CONTRACT_V2_2_CANONICAL_SHA256,
    PILOT_CONTRACT_V2_SCIENCE_DESIGN_SHA256,
    PilotContract,
    PilotContractError,
    canonical_contract_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "experiments" / "pilot_v2_1.yaml"
OVERLAY_PATH = ROOT / "experiments" / "pilot_v2_2_overlay.yaml"
FULL_PATH = ROOT / "experiments" / "pilot_v2_2.yaml"

FROZEN_EXPECTED_CI = {
    "test_count": 748,
    "test_collection_sha256": (
        "17644ad90479f854a1ac472fea6aa281e8be8d541a3f1e3082a6df71dd7c6f1d"
    ),
    "compiled_source_count": 142,
    "compiled_source_inventory_sha256": (
        "83b7065cd13e315a660c639dd112912e42395d723184e938a18a9ead97830844"
    ),
    "sealed_manifest_inventory_sha256": (
        "b5c5a817d09d10752c1f5f00ba556b417d16e06c64b5fcbb15671e49a1d81952"
    ),
}

SCIENCE_DESIGN_FIELDS = (
    "seeds",
    "provider_profiles",
    "arms",
    "narratives",
    "shocks",
    "utility",
    "stop_go",
    "stages",
    "parameter_dispatch_policy",
    "task_output_contracts",
    "model_roles",
    "non_claims",
)


def _overlay_document() -> dict[str, Any]:
    return json.loads(OVERLAY_PATH.read_text(encoding="utf-8"))


def _write_resealed_overlay(
    tmp_path: Path,
    value: dict[str, Any],
) -> Path:
    value["integrity"]["declared_sha256"] = canonical_contract_sha256(value)
    (tmp_path / "pilot_v2_1.yaml").write_text(
        BASE_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    path = tmp_path / "pilot_v2_2_overlay.yaml"
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _concrete_expected_ci() -> dict[str, Any]:
    return {
        "test_count": 700,
        "test_collection_sha256": "1" * 64,
        "compiled_source_count": 140,
        "compiled_source_inventory_sha256": "2" * 64,
        "sealed_manifest_inventory_sha256": "3" * 64,
    }


def test_v2_2_overlay_expands_over_exact_v2_1_without_science_drift() -> None:
    source = _overlay_document()
    base = load_pilot_contract(BASE_PATH)
    amended = load_pilot_contract(OVERLAY_PATH)
    full = load_pilot_contract(FULL_PATH)

    assert source["schema_version"] == (
        PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_2
    )
    assert source["integrity"]["declared_sha256"] == (
        canonical_contract_sha256(source)
    )
    assert source["status"] == "frozen"
    assert base.contract_id == PILOT_CONTRACT_ID_V2_1
    assert base.canonical_hash == PILOT_CONTRACT_V2_1_CANONICAL_SHA256
    assert amended.schema_version == PILOT_CONTRACT_SCHEMA_VERSION_V2
    assert amended.contract_id == PILOT_CONTRACT_ID_V2_2
    assert amended.status == "frozen"
    assert amended.canonical_hash == PILOT_CONTRACT_V2_2_CANONICAL_SHA256
    assert full.canonical_hash == PILOT_CONTRACT_V2_2_CANONICAL_SHA256
    assert full.to_dict() == amended.to_dict()
    assert amended.implementation["required_git_tag"] == (
        PILOT_CONTRACT_TAG_V2_2
    )
    assert amended.release_requirements is not None
    assert amended.release_requirements.tag == PILOT_CONTRACT_TAG_V2_2
    assert dict(amended.release_requirements.expected_ci) == FROZEN_EXPECTED_CI
    assert amended.denominator_policy is not None
    assert amended.denominator_policy.policy_id == "finevo-pilot-v2.2-itt"
    assert amended.budgets == base.budgets
    assert amended.operational_amendment == base.operational_amendment
    assert amended.evaluator_amendment is not None
    assert science_design_sha256(amended.to_dict()) == (
        PILOT_CONTRACT_V2_SCIENCE_DESIGN_SHA256
    )
    assert len(amended.expand()) == len(base.expand()) == 174
    assert all(
        run.contract_id == PILOT_CONTRACT_ID_V2_2
        and run.run_id.startswith(f"{PILOT_CONTRACT_ID_V2_2}--")
        for run in amended.expand()
    )


def test_v2_2_only_changes_release_identity_and_evaluator_amendment() -> None:
    base = load_pilot_contract(BASE_PATH).to_dict()
    amended = load_pilot_contract(OVERLAY_PATH).to_dict()

    for field in SCIENCE_DESIGN_FIELDS:
        assert amended[field] == base[field]
    assert amended["budgets"] == base["budgets"]
    assert amended["operational_amendment"] == base["operational_amendment"]

    base_denominator = dict(base["denominator_policy"])
    amended_denominator = dict(amended["denominator_policy"])
    assert base_denominator.pop("policy_id") == "finevo-pilot-v2.1-itt"
    assert amended_denominator.pop("policy_id") == "finevo-pilot-v2.2-itt"
    assert amended_denominator == base_denominator

    base_implementation = dict(base["implementation"])
    amended_implementation = dict(amended["implementation"])
    assert base_implementation.pop("required_git_tag") == "pilot-v2.1-science"
    assert (
        amended_implementation.pop("required_git_tag")
        == PILOT_CONTRACT_TAG_V2_2
    )
    assert amended_implementation == base_implementation

    assert amended["evaluator_amendment"]["rescore_policy"][
        "provider_calls"
    ] == 0
    assert amended["evaluator_amendment"]["rescore_policy"][
        "apply_uniformly_to_all_source_models"
    ] is True
    assert amended["evaluator_amendment"]["defect"][
        "semantic_match_disposition"
    ] == "diagnostic-only"


def test_v2_2_binds_both_corrected_results_and_cumulative_parent_debit() -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    amendment = contract.evaluator_amendment
    assert amendment is not None

    source_attempts = {
        row["model_id"]: row for row in amendment["source_attempts"]
    }
    assert set(source_attempts) == {
        "gpt52_main",
        "llama33_local_controlled",
    }
    assert source_attempts["gpt52_main"]["old_scores"]["rule_proposal"] == {
        "correct": 0,
        "denominator": 6,
    }
    assert source_attempts["llama33_local_controlled"]["old_scores"][
        "rule_application"
    ] == {"correct": 10, "denominator": 12}

    corrected = {
        row["model_id"]: row for row in amendment["corrected_results"]
    }
    assert set(corrected) == set(source_attempts)
    assert all(row["status"] == "complete" for row in corrected.values())
    assert all(row["provider_calls"] == 0 for row in corrected.values())
    assert all(
        row["scores"]["rule_proposal"] == {
            "correct": 6,
            "denominator": 6,
        }
        for row in corrected.values()
    )
    assert amendment["budget_carry_forward"] == {
        "source_contract_id": "finevo-pilot-v2.1",
        "source_contract_sha256": PILOT_CONTRACT_V2_1_CANONICAL_SHA256,
        "source_stage_bucket": "capability",
        "cost_usd": 1.53775475,
        "hosted_completions": 60,
        "storage_bytes": 715860,
    }


def test_v2_2_unresealed_overlay_tamper_is_rejected(tmp_path: Path) -> None:
    value = _overlay_document()
    value["evaluator_amendment"]["rescore_policy"]["provider_calls"] = 1
    (tmp_path / "pilot_v2_1.yaml").write_text(
        BASE_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    path = tmp_path / "pilot_v2_2_overlay.yaml"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(PilotContractError, match="overlay hash mismatch"):
        load_pilot_contract(path)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["base_contract"].__setitem__(
                "canonical_sha256",
                "1" * 64,
            ),
            "base contract binding drifted",
        ),
        (
            lambda value: value["changes"].__setitem__("budgets", {}),
            "invalid V2.2 changes keys",
        ),
        (
            lambda value: value["evaluator_amendment"]["source_attempts"][0].__setitem__(
                "capability_sha256",
                "2" * 64,
            ),
            "evaluator amendment drifted",
        ),
        (
            lambda value: value["evaluator_amendment"][
                "rescore_policy"
            ].__setitem__("provider_redispatch", "allowed"),
            "evaluator amendment drifted",
        ),
        (
            lambda value: value["evaluator_amendment"][
                "corrected_results"
            ][1]["scores"]["rule_proposal"].__setitem__("correct", 5),
            "evaluator amendment drifted",
        ),
        (
            lambda value: value["evaluator_amendment"][
                "budget_carry_forward"
            ].__setitem__("cost_usd", 1.0),
            "evaluator amendment drifted",
        ),
    ],
)
def test_v2_2_resealed_forbidden_amendment_is_rejected(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    value = _overlay_document()
    mutator(value)
    path = _write_resealed_overlay(tmp_path, value)

    with pytest.raises(PilotContractError, match=message):
        load_pilot_contract(path)


@pytest.mark.parametrize("missing", ["operational_amendment", "evaluator_amendment"])
def test_expanded_v2_2_requires_complete_amendment_chain(missing: str) -> None:
    value = load_pilot_contract(OVERLAY_PATH).to_dict()
    value.pop(missing)
    value["integrity"]["declared_sha256"] = canonical_contract_sha256(value)

    with pytest.raises(PilotContractError, match=missing):
        PilotContract.from_dict(value)


def test_v2_2_frozen_overlay_requires_concrete_ci(tmp_path: Path) -> None:
    value = _overlay_document()
    value["status"] = "frozen"
    value["changes"]["release_requirements"]["expected_ci"] = (
        _concrete_expected_ci()
    )
    contract = load_pilot_contract(_write_resealed_overlay(tmp_path, value))

    assert contract.status == "frozen"
    assert contract.release_requirements is not None
    assert dict(contract.release_requirements.expected_ci) == (
        _concrete_expected_ci()
    )
    assert contract.validate_provenance(
        "1" * 40,
        PILOT_CONTRACT_TAG_V2_2,
    )["contract_id"] == PILOT_CONTRACT_ID_V2_2


def test_v2_2_draft_cannot_validate_paid_provenance(tmp_path: Path) -> None:
    value = _overlay_document()
    value["status"] = "draft"
    value["changes"]["release_requirements"]["expected_ci"] = {
        key: None for key in FROZEN_EXPECTED_CI
    }
    contract = load_pilot_contract(_write_resealed_overlay(tmp_path, value))

    with pytest.raises(PilotContractError, match="draft contract"):
        contract.validate_provenance("1" * 40, PILOT_CONTRACT_TAG_V2_2)


@pytest.mark.parametrize("status", ["draft", "frozen"])
def test_v2_2_expected_ci_rejects_mixed_state(
    tmp_path: Path,
    status: str,
) -> None:
    value = _overlay_document()
    value["status"] = status
    if status == "draft":
        value["changes"]["release_requirements"]["expected_ci"] = {
            key: None for key in FROZEN_EXPECTED_CI
        }
        value["changes"]["release_requirements"]["expected_ci"][
            "test_count"
        ] = 700
    else:
        value["changes"]["release_requirements"]["expected_ci"] = dict(
            FROZEN_EXPECTED_CI
        )
        value["changes"]["release_requirements"]["expected_ci"][
            "test_collection_sha256"
        ] = None
    path = _write_resealed_overlay(tmp_path, value)

    with pytest.raises(PilotContractError, match="exactly all-"):
        load_pilot_contract(path)
