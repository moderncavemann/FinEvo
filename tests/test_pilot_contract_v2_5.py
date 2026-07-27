from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_4,
    PILOT_CONTRACT_ID_V2_5,
    PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_5,
    PILOT_CONTRACT_TAG_V2_5,
    PILOT_CONTRACT_V2_4_CANONICAL_SHA256,
    PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256,
    PilotContractError,
    canonical_contract_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
V24_PATH = EXPERIMENTS / "pilot_v2_4.yaml"
V24_SOURCE_PATH = EXPERIMENTS / "pilot_v2_4_parent_source_manifest.json"
OVERLAY_PATH = EXPERIMENTS / "pilot_v2_5_overlay.yaml"
SOURCE_PATH = EXPERIMENTS / "pilot_v2_5_source_manifest.json"


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
    for source in (V24_PATH, V24_SOURCE_PATH, SOURCE_PATH):
        (tmp_path / source.name).write_bytes(source.read_bytes())
    path = tmp_path / "pilot_v2_5_overlay.yaml"
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_v2_5_draft_is_a_new_release_identity_only() -> None:
    source = _overlay_document()
    v24 = load_pilot_contract(V24_PATH)
    v25 = load_pilot_contract(OVERLAY_PATH)

    assert source["schema_version"] == (
        PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_5
    )
    assert source["integrity"]["declared_sha256"] == (
        canonical_contract_sha256(source)
    )
    assert v24.contract_id == PILOT_CONTRACT_ID_V2_4
    assert v24.canonical_hash == PILOT_CONTRACT_V2_4_CANONICAL_SHA256
    assert v25.contract_id == PILOT_CONTRACT_ID_V2_5
    assert v25.status == "draft"
    assert v25.implementation["required_git_tag"] == PILOT_CONTRACT_TAG_V2_5
    assert v25.release_requirements is not None
    assert v25.release_requirements.tag == PILOT_CONTRACT_TAG_V2_5
    assert set(v25.release_requirements.expected_ci.values()) == {None}
    with pytest.raises(PilotContractError, match="draft contract"):
        v25.validate_provenance("1" * 40, PILOT_CONTRACT_TAG_V2_5)


def test_v2_5_preserves_the_exact_v2_4_211_cell_science_design() -> None:
    v24 = load_pilot_contract(V24_PATH)
    v25 = load_pilot_contract(OVERLAY_PATH)
    v24_specs = v24.expand()
    v25_specs = v25.expand()

    assert len(v24_specs) == len(v25_specs) == 211
    assert _normalized_specs(V24_PATH) == _normalized_specs(OVERLAY_PATH)
    assert v25.stage_ids == v24.stage_ids == (
        "parent-import",
        "q-ref-resolution",
        "stage0-calibration",
        "local-experiment-c",
        "local-experiment-a",
        "local-experiment-d",
        "local-experiment-b",
        "experiment-c",
        "experiment-a",
        "experiment-d",
        "experiment-b",
    )
    assert Counter(spec.stage_id for spec in v25_specs) == {
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
    assert science_design_sha256(v24.to_dict()) == (
        PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256
    )
    assert science_design_sha256(v25.to_dict()) == (
        PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256
    )
    assert v25.seeds == v24.seeds
    assert v25.arms == v24.arms
    assert v25.model_ids == v24.model_ids
    assert v25.provider_profiles == v24.provider_profiles
    assert v25.shocks == v24.shocks
    assert v25.utility == v24.utility
    assert v25.stop_go == v24.stop_go


def test_v2_5_keeps_the_authorized_budget_and_cumulative_debits() -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    amendment = contract.parent_import_retry_amendment

    assert amendment is not None
    assert contract.budgets["total_usd"] == 500.0
    assert contract.budgets["max_provider_completions"] == 7500
    assert contract.budgets["max_storage_bytes"] == 5_000_000_000
    assert dict(contract.budgets["stage_usd_caps"]) == {
        "parent_v23": 3.212770875,
        "local": 0.0,
        "hosted_confirmatory": 495.787229125,
        "manual_reserve": 1.0,
    }
    carry = amendment["budget_carry_forward"]
    assert carry["total_cap_usd"] == 500.0
    assert carry["max_provider_completions"] == 7500
    assert carry["max_storage_bytes"] == 5_000_000_000
    assert carry["v2_3"] == {
        "cost_usd": 3.212770875,
        "hosted_completions": 184,
        "storage_bytes": 4_196_087,
    }
    assert carry["v2_4_incremental"] == {
        "cost_usd": 0.0,
        "hosted_completions": 0,
        "storage_bytes": 518_235,
    }
    assert carry["budget_reset"] is False
    assert carry["debit_before_new_dispatch"] is True


def test_v2_5_binds_terminal_v2_4_and_original_v2_3_authority() -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    amendment = contract.parent_import_retry_amendment
    manifest = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))

    assert amendment is not None
    failure = amendment["failure_classification"]
    assert failure["parent_contract_id"] == PILOT_CONTRACT_ID_V2_4
    assert failure["parent_contract_sha256"] == (
        PILOT_CONTRACT_V2_4_CANONICAL_SHA256
    )
    assert failure["propagated_terminal_cells"] == 211
    assert failure["provider_calls"] == 0
    assert failure["scientific_effect_outcomes_available"] is False
    assert failure["scientific_effect_outcomes_inspected"] is False

    policy = amendment["correction_policy"]
    assert policy["historical_code_binding_verification"] == (
        "annotated-tag-peeled-commit-git-tree"
    )
    assert policy["historical_source_hashes_and_binding_hash_required"] is True
    assert policy["unconditional_strict_disable_forbidden"] is True
    assert policy["current_compatibility_replay_after_historical_gate"] is True
    assert policy["current_recomputed_exactness_must_equal_frozen"] is True
    assert policy["child_code_as_parent_binding_authority"] is False

    v24 = manifest["v2_4_terminal_parent"]
    assert v24["terminal_denominator"]["status_counts"] == {
        "integrity-stopped": 211
    }
    assert v24["parent_import_failure"]["provider_calls"] == 0
    assert v24["incremental_budget_debit"]["cost_usd"] == 0.0
    assert manifest["v2_4_published_evidence"]["status"] == (
        "complete-with-no-go"
    )
    assert manifest["v2_4_published_evidence"]["scientific_complete"] is False
    v23 = manifest["v2_3_authority_parent"]
    assert v23["science_tag"] == "pilot-v2.3-science"
    assert v23["science_tag_object"] == (
        "e985abd6749471363db6b27bda66485c0b578bb3"
    )
    assert v23["science_commit"] == (
        "ab32e3c9dcf581a40f3093652e144b56f853c782"
    )
    assert set(v23["observed_p95_sources"]) == {
        "gpt52_main",
        "llama33_local_controlled",
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["parent_import_retry_amendment"][
                "budget_carry_forward"
            ]["v2_4_incremental"].update({"cost_usd": 0.01}),
            "retry amendment drifted",
        ),
        (
            lambda value: value["parent_import_retry_amendment"][
                "failure_classification"
            ].update({"scientific_effect_outcomes_inspected": True}),
            "retry amendment drifted",
        ),
        (
            lambda value: value["parent_import_retry_amendment"][
                "correction_policy"
            ].update({"unconditional_strict_disable_forbidden": False}),
            "retry amendment drifted",
        ),
        (
            lambda value: value["changes"]["denominator_policy"].update(
                {"policy_id": "finevo-pilot-v2.4-itt"}
            ),
            "denominator identifier drifted",
        ),
    ],
)
def test_v2_5_resealed_operational_or_provenance_drift_fails(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    value = _overlay_document()
    mutate(value)
    path = _write_resealed_overlay(tmp_path, value)
    with pytest.raises(PilotContractError, match=message):
        load_pilot_contract(path)


def test_v2_5_source_manifest_file_hash_is_enforced(tmp_path: Path) -> None:
    path = _write_resealed_overlay(tmp_path, _overlay_document())
    manifest = tmp_path / SOURCE_PATH.name
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PilotContractError, match="manifest file hash drifted"):
        load_pilot_contract(path)


def test_v2_4_contract_remains_immutable_and_has_no_retry_amendment() -> None:
    contract = load_pilot_contract(V24_PATH)

    assert contract.contract_id == PILOT_CONTRACT_ID_V2_4
    assert contract.canonical_hash == PILOT_CONTRACT_V2_4_CANONICAL_SHA256
    assert len(contract.expand()) == 211
    assert contract.parent_import_retry_amendment is None
