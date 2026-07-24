from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2,
    PILOT_CONTRACT_ID_V2_1,
    PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_1,
    PILOT_CONTRACT_SCHEMA_VERSION_V2,
    PILOT_CONTRACT_TAG_V2_1,
    PILOT_CONTRACT_V2_CANONICAL_SHA256,
    PilotContract,
    PilotContractError,
    canonical_contract_sha256,
    load_pilot_contract,
)
from verified_memory.scientific_release_attestation import (
    ScientificReleaseAttestationError,
    ScientificReleaseRequirements,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "experiments" / "pilot_v2.yaml"
LAUNCH_PATH = ROOT / "experiments" / "pilot_v2_1.yaml"
OVERLAY_PATH = ROOT / "experiments" / "pilot_v2_1_overlay.yaml"

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
    (tmp_path / "pilot_v2.yaml").write_text(
        BASE_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    path = tmp_path / "pilot_v2_1.yaml"
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _diff_paths(
    left: Any,
    right: Any,
    *,
    prefix: str = "",
) -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: set[str] = set()
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                paths.add(path)
            else:
                paths |= _diff_paths(left[key], right[key], prefix=path)
        return paths
    if left != right:
        return {prefix}
    return set()


def test_v2_1_overlay_expands_to_hash_bound_v2_contract() -> None:
    source = _overlay_document()
    base = load_pilot_contract(BASE_PATH)
    amended = load_pilot_contract(OVERLAY_PATH)
    launch = load_pilot_contract(LAUNCH_PATH)

    assert source["schema_version"] == (
        PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_1
    )
    assert source["integrity"]["declared_sha256"] == (
        canonical_contract_sha256(source)
    )
    assert source["status"] == "frozen"
    assert base.canonical_hash == PILOT_CONTRACT_V2_CANONICAL_SHA256
    assert base.contract_id == PILOT_CONTRACT_ID_V2
    assert amended.schema_version == PILOT_CONTRACT_SCHEMA_VERSION_V2
    assert amended.contract_id == PILOT_CONTRACT_ID_V2_1
    assert amended.status == "frozen"
    assert amended.declared_sha256 == amended.canonical_hash
    assert amended.canonical_hash != base.canonical_hash
    assert launch.to_dict() == amended.to_dict()
    assert launch.canonical_hash == amended.canonical_hash
    assert amended.implementation["required_git_tag"] == (
        PILOT_CONTRACT_TAG_V2_1
    )
    assert amended.release_requirements is not None
    assert amended.release_requirements.tag == PILOT_CONTRACT_TAG_V2_1
    assert dict(amended.release_requirements.expected_ci) == {
        "test_count": 686,
        "test_collection_sha256": (
            "5bf1976e5bc11dff4ac4d6ae436fb4c4d48d54d9f811a1bf204637e862f6e23b"
        ),
        "compiled_source_count": 138,
        "compiled_source_inventory_sha256": (
            "bcf420bf723521d4f54c3debde7e8aad08652206e2421ed9dd0c59aa28662f66"
        ),
        "sealed_manifest_inventory_sha256": (
            "b5c5a817d09d10752c1f5f00ba556b417d16e06c64b5fcbb15671e49a1d81952"
        ),
    }
    assert len(amended.expand()) == len(base.expand()) == 174
    assert all(
        run.contract_id == PILOT_CONTRACT_ID_V2_1
        and run.run_id.startswith(f"{PILOT_CONTRACT_ID_V2_1}--")
        for run in amended.expand()
    )


def test_v2_1_only_changes_registered_operational_fields() -> None:
    base = load_pilot_contract(BASE_PATH).to_dict()
    amended_contract = load_pilot_contract(LAUNCH_PATH)
    amended = amended_contract.to_dict()

    for field in SCIENCE_DESIGN_FIELDS:
        assert amended[field] == base[field]
    base_denominator = dict(base["denominator_policy"])
    amended_denominator = dict(amended["denominator_policy"])
    assert base_denominator.pop("policy_id") == "finevo-pilot-v2-itt"
    assert amended_denominator.pop("policy_id") == "finevo-pilot-v2.1-itt"
    assert amended_denominator == base_denominator

    assert _diff_paths(base, amended) == {
        "budgets.stage_usd_caps.capability",
        "budgets.stage_usd_caps.cross_model",
        "contract_id",
        "denominator_policy.policy_id",
        "implementation.required_git_tag",
        "integrity.declared_sha256",
        "operational_amendment",
        "release_requirements.expected_ci.compiled_source_count",
        (
            "release_requirements.expected_ci."
            "compiled_source_inventory_sha256"
        ),
        "release_requirements.expected_ci.test_collection_sha256",
        "release_requirements.expected_ci.test_count",
        "release_requirements.tag",
    }
    assert dict(amended_contract.budgets["stage_usd_caps"]) == {
        "capability": 3.0701145,
        "calibration": 3.0,
        "core": 13.0,
        "cross_model": 4.9298855,
        "manual_reserve": 1.0,
    }
    assert amended_contract.budgets["total_usd"] == 25.0
    assert amended_contract.budgets["automatic_reserve_usd"] == 1.0


def test_v2_1_operational_amendment_binds_failure_retry_and_parent() -> None:
    contract = load_pilot_contract(LAUNCH_PATH)
    amendment = contract.operational_amendment
    assert amendment is not None

    assert amendment["parent"]["contract_sha256"] == (
        PILOT_CONTRACT_V2_CANONICAL_SHA256
    )
    assert amendment["failure"]["error_type"] == "APIConnectionError"
    assert amendment["failure"]["failure_count"] == 30
    assert amendment["failure"]["capability_sha256"] == (
        "da9076389db58fd682d213ccb932d66bb767f73423e5476abea788eb1f8fd294"
    )
    assert amendment["failure"]["gate_sha256"] == (
        "176547171d88dad5e757dc1795cef749bea57ea7e7291191a240c8cd92c57997"
    )
    assert amendment["failure"]["terminal_sha256"] == (
        "10b5ff7c78b4697b9754c809bed0e7d14380729a640632585085ad7f886704c6"
    )
    assert amendment["failure"]["secret_rotation_required"] is True
    assert tuple(amendment["retry_policy"]["eligible_model_ids"]) == (
        "gpt52_main",
    )
    assert amendment["retry_policy"]["preserve_parent_denominator"] is True
    assert amendment["retry_policy"]["failed_seed_replacement"] == "forbidden"
    assert amendment["retry_policy"]["outcome_inspected_for_retry"] is False
    assert amendment["budget_carry_forward"] == {
        "source_stage_bucket": "capability",
        "cost_usd": 1.0701145,
        "hosted_completions": 30,
        "storage_bytes": 479367,
    }
    inherited = amendment["inherited_results"]
    assert len(inherited) == 1
    assert inherited[0]["model_id"] == "llama33_local_controlled"
    assert inherited[0]["scores"] == {
        "utility_ranking": {"correct": 12, "denominator": 12},
        "rule_application": {"correct": 10, "denominator": 12},
        "rule_proposal": {"correct": 0, "denominator": 6},
    }


def test_v2_1_unresealed_overlay_tamper_is_rejected(tmp_path: Path) -> None:
    value = _overlay_document()
    value["changes"]["budgets"]["stage_usd_caps"]["capability"] = 3.1
    (tmp_path / "pilot_v2.yaml").write_text(
        BASE_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    path = tmp_path / "pilot_v2_1.yaml"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(PilotContractError, match="overlay hash mismatch"):
        load_pilot_contract(path)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["changes"].__setitem__("seeds", {}),
            "invalid V2.1 changes keys",
        ),
        (
            lambda value: value["base_contract"].__setitem__(
                "canonical_sha256",
                "1" * 64,
            ),
            "base contract binding drifted",
        ),
        (
            lambda value: value["changes"]["budgets"]["stage_usd_caps"].__setitem__(
                "cross_model",
                4.9,
            ),
            "overlay budget caps drifted",
        ),
        (
            lambda value: value["operational_amendment"]["failure"].__setitem__(
                "failure_count",
                29,
            ),
            "retry failure binding drifted",
        ),
        (
            lambda value: value["operational_amendment"]["failure"].__setitem__(
                "terminal_sha256",
                "f" * 64,
            ),
            "retry failure binding drifted",
        ),
        (
            lambda value: value["operational_amendment"][
                "retry_policy"
            ].__setitem__(
                "eligible_model_ids",
                ["gpt52_main", "llama33_local_controlled"],
            ),
            "operational retry policy drifted",
        ),
    ],
)
def test_v2_1_resealed_forbidden_amendment_is_rejected(
    tmp_path: Path,
    mutator: Any,
    message: str,
) -> None:
    value = _overlay_document()
    mutator(value)
    path = _write_resealed_overlay(tmp_path, value)

    with pytest.raises(PilotContractError, match=message):
        load_pilot_contract(path)


def test_expanded_v2_1_requires_operational_amendment() -> None:
    value = load_pilot_contract(LAUNCH_PATH).to_dict()
    value.pop("operational_amendment")
    value["integrity"]["declared_sha256"] = canonical_contract_sha256(value)

    with pytest.raises(PilotContractError, match="operational_amendment"):
        PilotContract.from_dict(value)


def _concrete_expected_ci() -> dict[str, Any]:
    return {
        "test_count": 679,
        "test_collection_sha256": "1" * 64,
        "compiled_source_count": 79,
        "compiled_source_inventory_sha256": "2" * 64,
        "sealed_manifest_inventory_sha256": "3" * 64,
    }


def test_v2_1_frozen_overlay_requires_and_accepts_all_concrete_ci(
    tmp_path: Path,
) -> None:
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
    release = ScientificReleaseRequirements.from_mapping(
        contract.release_requirements.to_dict()
    )
    assert release.expected_test_count == 679


@pytest.mark.parametrize("status", ["draft", "frozen"])
def test_v2_1_expected_ci_rejects_mixed_state(
    tmp_path: Path,
    status: str,
) -> None:
    value = _overlay_document()
    value["status"] = status
    value["changes"]["release_requirements"]["expected_ci"] = {
        field: None
        for field in value["changes"]["release_requirements"]["expected_ci"]
    }
    value["changes"]["release_requirements"]["expected_ci"]["test_count"] = 679
    path = _write_resealed_overlay(tmp_path, value)

    with pytest.raises(PilotContractError, match="exactly all-"):
        load_pilot_contract(path)


def test_v2_1_draft_is_rejected_by_paid_and_release_paths(
    tmp_path: Path,
) -> None:
    value = _overlay_document()
    value["status"] = "draft"
    value["changes"]["release_requirements"]["expected_ci"] = {
        field: None
        for field in value["changes"]["release_requirements"]["expected_ci"]
    }
    contract = load_pilot_contract(_write_resealed_overlay(tmp_path, value))
    assert contract.release_requirements is not None

    with pytest.raises(PilotContractError, match="draft contract"):
        contract.validate_provenance("1" * 40, PILOT_CONTRACT_TAG_V2_1)
    with pytest.raises(ScientificReleaseAttestationError):
        ScientificReleaseRequirements.from_mapping(
            contract.release_requirements.to_dict()
        )
