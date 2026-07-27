from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_8,
    PILOT_CONTRACT_ID_V2_9,
    PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_9,
    PILOT_CONTRACT_TAG_V2_9,
    PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256,
    PILOT_CONTRACT_V2_8_CANONICAL_SHA256,
    PILOT_CONTRACT_V2_9_CANONICAL_SHA256,
    PILOT_V2_9_SOURCE_MANIFEST_CONTENT_SHA256,
    PILOT_V2_9_SOURCE_MANIFEST_FILE_SHA256,
    PilotContractError,
    canonical_contract_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
V28_PATH = EXPERIMENTS / "pilot_v2_8.yaml"
OVERLAY_PATH = EXPERIMENTS / "pilot_v2_9_overlay.yaml"
FULL_PATH = EXPERIMENTS / "pilot_v2_9.yaml"
DEPENDENCIES = (
    V28_PATH,
    EXPERIMENTS / "pilot_v2_4_parent_source_manifest.json",
    EXPERIMENTS / "pilot_v2_5_source_manifest.json",
    EXPERIMENTS / "pilot_v2_6_source_manifest.json",
    EXPERIMENTS / "pilot_v2_7_source_manifest.json",
    EXPERIMENTS / "pilot_v2_8_source_manifest.json",
    EXPERIMENTS / "pilot_v2_9_source_manifest.json",
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


def _write_overlay(
    tmp_path: Path,
    value: dict[str, Any],
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    value["integrity"]["declared_sha256"] = canonical_contract_sha256(value)
    for source in DEPENDENCIES:
        (tmp_path / source.name).write_bytes(source.read_bytes())
    path = tmp_path / "pilot_v2_9_overlay.yaml"
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_v2_9_frozen_overlay_and_expanded_contract_are_identical() -> None:
    source = _overlay_document()
    parent = load_pilot_contract(V28_PATH)
    overlay = load_pilot_contract(OVERLAY_PATH)
    full = load_pilot_contract(FULL_PATH)

    assert source["schema_version"] == (
        PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_9
    )
    assert source["integrity"]["declared_sha256"] == (
        canonical_contract_sha256(source)
    )
    assert parent.contract_id == PILOT_CONTRACT_ID_V2_8
    assert parent.canonical_hash == PILOT_CONTRACT_V2_8_CANONICAL_SHA256
    assert overlay.contract_id == full.contract_id == PILOT_CONTRACT_ID_V2_9
    assert overlay.status == full.status == "frozen"
    assert overlay.to_dict() == full.to_dict()
    assert overlay.canonical_hash == full.canonical_hash == (
        PILOT_CONTRACT_V2_9_CANONICAL_SHA256
    )
    assert overlay.declared_sha256 == overlay.canonical_hash
    assert overlay.implementation["required_git_tag"] == PILOT_CONTRACT_TAG_V2_9
    assert overlay.release_requirements is not None
    assert overlay.release_requirements.tag == PILOT_CONTRACT_TAG_V2_9
    assert overlay.release_requirements.expected_ci == {
        "test_count": 1065,
        "test_collection_sha256": (
            "03d18c63d9a28e81228c4d2ce0c3a811cb40047091c7c6c9fb092e0690fd0b3b"
        ),
        "compiled_source_count": 187,
        "compiled_source_inventory_sha256": (
            "52a0701bd7b3c21669e73efa889291c60540e83c3457a5b536b624f271578c08"
        ),
        "sealed_manifest_inventory_sha256": (
            "b5c5a817d09d10752c1f5f00ba556b417d16e06c64b5fcbb15671e49a1d81952"
        ),
    }
    amendment = overlay.qref_summary_equivalence_amendment
    assert amendment is not None
    assert amendment["source_manifest"] == {
        "path": "experiments/pilot_v2_9_source_manifest.json",
        "schema_version": "finevo-pilot-v2.9-source-manifest-v1",
        "file_sha256": PILOT_V2_9_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": PILOT_V2_9_SOURCE_MANIFEST_CONTENT_SHA256,
    }


def test_v2_9_preserves_v2_8_science_design_and_new_denominator() -> None:
    parent = load_pilot_contract(V28_PATH)
    contract = load_pilot_contract(OVERLAY_PATH)
    specs = contract.expand()

    assert len(parent.expand()) == len(specs) == 211
    assert _normalized_specs(V28_PATH) == _normalized_specs(OVERLAY_PATH)
    assert science_design_sha256(contract.to_dict()) == (
        PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256
    )
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
    assert contract.denominator_policy is not None
    assert contract.denominator_policy.policy_id == "finevo-pilot-v2.9-itt"
    assert all(spec.contract_id == PILOT_CONTRACT_ID_V2_9 for spec in specs)
    assert len({spec.run_id for spec in specs}) == 211


def test_v2_9_failure_budget_and_summary_policy_are_explicit() -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    amendment = contract.qref_summary_equivalence_amendment
    assert amendment is not None
    failure = amendment["failure_classification"]
    assert failure["status_counts"] == {
        "complete": 1,
        "failed": 1,
        "integrity-stopped": 209,
    }
    assert failure["scripted_diagnostic_calls"] == 48
    assert failure["hosted_provider_calls"] == 0
    assert failure["incremental_cost_usd"] == 0.0
    assert failure["a_d_treatment_effect_outcomes_generated"] is False
    assert failure["root_cause_code"] == (
        "qref-raw-summary-equivalence-included-identity-and-monotonic-time"
    )

    carry = amendment["budget_carry_forward"]
    assert carry["total_cap_usd"] == 500.0
    assert carry["max_provider_completions"] == 7_500
    assert carry["max_storage_bytes"] == 5_000_000_000
    assert carry["cumulative_prior"]["cost_usd"] == 3.212770875
    assert carry["cumulative_prior"]["hosted_completions"] == 184
    assert carry["cumulative_prior"]["record_sha256"] == (
        "0944138d9b47f7cf720681eb0ea8feda0b612a912992d78434c6bbda0d560fd0"
    )
    assert carry["v2_8_incremental"] == {
        "cost_usd": 0.0,
        "hosted_completions": 0,
        "scripted_diagnostic_calls": 48,
    }

    policy = amendment["run_summary_equivalence"]
    assert policy["comparison_mode"] == (
        "identity-bound-allowlist-normalization-then-exact"
    )
    assert policy["expected_completion_rows"] == 48
    assert policy["expected_observed_scalar_paths"] == 1002
    assert policy["expected_allowed_difference_paths"] == 195
    assert policy["tokens_cost_models_labels_tags_retained_exactly"] is True
    assert tuple(policy["identity_paths"]) == (
        "$.run_id",
        "$.api.budget_id",
        "$.api.completions[*].budget_id",
    )
    assert tuple(policy["monotonic_time_paths"]) == (
        "$.api.elapsed_seconds",
        "$.api.completions[*].started_elapsed_seconds",
        "$.api.completions[*].finished_elapsed_seconds",
        "$.api.completions[*].elapsed_seconds",
    )


def test_v2_9_overlay_rejects_science_or_policy_tampering(
    tmp_path: Path,
) -> None:
    source = _overlay_document()
    changed_seed = deepcopy(source)
    # The compact overlay does not expose science fields, so tamper its exact
    # parent binding and summary policy instead.
    changed_seed["base_contract"]["canonical_sha256"] = "1" * 64
    with pytest.raises(PilotContractError, match="base contract binding"):
        load_pilot_contract(_write_overlay(tmp_path / "base", changed_seed))

    changed_policy = deepcopy(source)
    changed_policy["qref_summary_equivalence_amendment"][
        "run_summary_equivalence"
    ]["tokens_cost_models_labels_tags_retained_exactly"] = False
    with pytest.raises(
        PilotContractError,
        match="summary-equivalence amendment drifted",
    ):
        load_pilot_contract(_write_overlay(tmp_path / "policy", changed_policy))
