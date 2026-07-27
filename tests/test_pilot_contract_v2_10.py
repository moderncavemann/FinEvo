from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import pytest

import scripts.render_pilot_v210_contract as render_v210
from verified_memory import pilot_contract as contract_module
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_9,
    PILOT_CONTRACT_ID_V2_10,
    PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10,
    PILOT_CONTRACT_SCHEMA_VERSION_V2,
    PILOT_CONTRACT_TAG_V2_10,
    PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256,
    PILOT_CONTRACT_V2_9_CANONICAL_SHA256,
    PILOT_CONTRACT_V2_10_CANONICAL_SHA256,
    PILOT_V2_10_SOURCE_MANIFEST_CONTENT_SHA256,
    PILOT_V2_10_SOURCE_MANIFEST_FILE_SHA256,
    PilotContractError,
    _v2_10_expected_p95_runner_binding_retry_amendment,
    canonical_contract_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
V29_PATH = EXPERIMENTS / "pilot_v2_9.yaml"
DEPENDENCIES = (
    V29_PATH,
    EXPERIMENTS / "pilot_v2_4_parent_source_manifest.json",
    EXPERIMENTS / "pilot_v2_5_source_manifest.json",
    EXPERIMENTS / "pilot_v2_6_source_manifest.json",
    EXPERIMENTS / "pilot_v2_7_source_manifest.json",
    EXPERIMENTS / "pilot_v2_8_source_manifest.json",
    EXPERIMENTS / "pilot_v2_9_source_manifest.json",
)


def _draft_overlay() -> dict[str, Any]:
    value = {
        "schema_version": PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10,
        "contract_id": PILOT_CONTRACT_ID_V2_10,
        "status": "draft",
        "base_contract": {
            "path": "pilot_v2_9.yaml",
            "schema_version": PILOT_CONTRACT_SCHEMA_VERSION_V2,
            "contract_id": PILOT_CONTRACT_ID_V2_9,
            "canonical_sha256": PILOT_CONTRACT_V2_9_CANONICAL_SHA256,
        },
        "changes": {
            "implementation": {
                "required_git_tag": PILOT_CONTRACT_TAG_V2_10,
            },
            "release_requirements": {
                "tag": PILOT_CONTRACT_TAG_V2_10,
                "expected_ci": {
                    "test_count": None,
                    "test_collection_sha256": None,
                    "compiled_source_count": None,
                    "compiled_source_inventory_sha256": None,
                    "sealed_manifest_inventory_sha256": None,
                },
            },
            "denominator_policy": {
                "policy_id": "finevo-pilot-v2.10-itt",
            },
        },
        "p95_runner_binding_retry_amendment": (
            _v2_10_expected_p95_runner_binding_retry_amendment(
                status="draft",
            )
        ),
        "integrity": {
            "canonicalization": "json-sort-keys-utf8-v1",
            "declared_sha256": "0" * 64,
        },
    }
    value["integrity"]["declared_sha256"] = canonical_contract_sha256(value)
    return value


def _copy_dependencies(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for source in DEPENDENCIES:
        (target / source.name).write_bytes(source.read_bytes())


def _write_overlay(target: Path, value: dict[str, Any]) -> Path:
    _copy_dependencies(target)
    value["integrity"]["declared_sha256"] = canonical_contract_sha256(value)
    path = target / "pilot_v2_10_overlay.yaml"
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _normalized_specs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in load_pilot_contract(path).expand():
        row = spec.to_dict()
        row.pop("contract_id")
        row.pop("run_id")
        rows.append(row)
    return rows


def test_v2_10_draft_overlay_preserves_exact_211_cell_matrix(
    tmp_path: Path,
) -> None:
    path = _write_overlay(tmp_path, _draft_overlay())
    parent = load_pilot_contract(tmp_path / "pilot_v2_9.yaml")
    contract = load_pilot_contract(path)

    assert contract.contract_id == PILOT_CONTRACT_ID_V2_10
    assert contract.status == "draft"
    assert contract.implementation["required_git_tag"] == (PILOT_CONTRACT_TAG_V2_10)
    assert contract.release_requirements is not None
    assert contract.release_requirements.tag == PILOT_CONTRACT_TAG_V2_10
    assert contract.denominator_policy is not None
    assert contract.denominator_policy.policy_id == ("finevo-pilot-v2.10-itt")
    assert len(parent.expand()) == len(contract.expand()) == 211
    assert _normalized_specs(path) == _normalized_specs(tmp_path / "pilot_v2_9.yaml")
    assert science_design_sha256(contract.to_dict()) == (
        PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256
    )
    assert Counter(spec.stage_id for spec in contract.expand()) == {
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


def test_v2_10_amendment_freezes_import_freshness_and_budget() -> None:
    amendment = _v2_10_expected_p95_runner_binding_retry_amendment(status="draft")

    assert amendment["source_manifest"] == {
        "path": "experiments/pilot_v2_10_source_manifest.json",
        "schema_version": "finevo-pilot-v2.10-source-manifest-v1",
        "file_sha256": None,
        "content_sha256": None,
    }
    assert amendment["prerequisite_import"]["imported_complete_cells"] == 16
    assert amendment["prerequisite_import"]["imported_cell_breakdown"] == {
        "parent-import": 1,
        "q-ref-resolution": 1,
        "stage0-calibration": 14,
    }
    assert amendment["fresh_science_dispatch"]["a_d_cells"] == 195
    assert amendment["fresh_science_dispatch"]["stage_counts"] == {
        "local-experiment-c": 25,
        "local-experiment-a": 20,
        "local-experiment-d": 35,
        "local-experiment-b": 25,
        "experiment-c": 25,
        "experiment-a": 20,
        "experiment-d": 30,
        "experiment-b": 15,
    }
    assert (
        amendment["fresh_science_dispatch"]["v2_9_offline_candidate_admission_reuse"]
        == "forbidden"
    )
    assert (
        amendment["observation_boundary"][
            "offline_candidate_admission_outcomes_generated"
        ]
        == 10
    )
    assert (
        amendment["observation_boundary"]["all_a_d_outcomes_unobserved_claim_forbidden"]
        is True
    )
    carry = amendment["budget_carry_forward"]
    assert carry["total_cap_usd"] == 500.0
    assert carry["max_provider_completions"] == 7_500
    assert carry["max_storage_bytes"] == 5_000_000_000
    assert carry["cumulative_prior"] == {
        "stage_bucket": "parent_v23",
        "cost_usd": 3.212770875,
        "hosted_completions": 184,
        "parent_contract_sha256": PILOT_CONTRACT_V2_9_CANONICAL_SHA256,
        "parent_run_ledger_sha256": (
            "9cc948d75c37ffeb59a2d7ed569e140668a997fa314d523906a047375011e409"
        ),
        "parent_budget_ledger_sha256": (
            "7e75b9c58bccaa746bdc92b926352fc0d3e56adee8426d3962a80ae5ddd59e10"
        ),
        "storage_bytes": 50_425_235,
        "record_sha256": (
            "408b25171d23c172abcc3e5545d736ef0fdb6251524995ab0bb39b34b0b6a5e1"
        ),
    }


def test_v2_10_p95_repair_requires_current_release_flat_binding() -> None:
    repair = _v2_10_expected_p95_runner_binding_retry_amendment(status="draft")[
        "p95_runner_binding_repair"
    ]

    assert repair["runner_binding_required_fields"] == [
        "receipt_path",
        "receipt_file_sha256",
        "receipt_content_sha256",
        "git_commit",
        "reservations",
    ]
    assert repair["current_release_wrapper_required"] is True
    assert repair["runner_binding_validation_before_provider_construction"] is True
    assert repair["nested_to_flat_alias_without_current_reseal"] == ("forbidden")
    assert repair["source_reservation_values_unchanged"] is True


def test_v2_10_draft_rejects_parent_or_amendment_tampering(
    tmp_path: Path,
) -> None:
    parent = _draft_overlay()
    parent["base_contract"]["canonical_sha256"] = "1" * 64
    with pytest.raises(PilotContractError, match="base contract binding"):
        load_pilot_contract(_write_overlay(tmp_path / "parent", parent))

    amendment = _draft_overlay()
    amendment["p95_runner_binding_retry_amendment"]["fresh_science_dispatch"][
        "v2_9_offline_candidate_admission_reuse"
    ] = "allowed"
    with pytest.raises(PilotContractError, match="runner-binding retry"):
        load_pilot_contract(_write_overlay(tmp_path / "amendment", amendment))


def test_v2_10_source_manifest_is_sealed_before_contract_freeze() -> None:
    assert PILOT_CONTRACT_V2_10_CANONICAL_SHA256 == (
        "d1b54c14d016c2b157db9e334d054ab9c7e86371d3fb9662a95fb94e50ce964b"
    )
    assert PILOT_V2_10_SOURCE_MANIFEST_FILE_SHA256 == (
        "8540bde06f364aa9ccf2a6937b78dec1f0d3b2c66b9e4943f9a3d2e20e4b19a7"
    )
    assert PILOT_V2_10_SOURCE_MANIFEST_CONTENT_SHA256 == (
        "fc781697a9260fa63d0535eafa24b87a8386a76dca55f3ce95ba59e12ceb4224"
    )
    amendment = _v2_10_expected_p95_runner_binding_retry_amendment(
        status="frozen"
    )
    assert amendment["source_manifest"]["file_sha256"] == (
        PILOT_V2_10_SOURCE_MANIFEST_FILE_SHA256
    )
    contract = load_pilot_contract(EXPERIMENTS / "pilot_v2_10.yaml")
    assert contract.canonical_hash == PILOT_CONTRACT_V2_10_CANONICAL_SHA256
    assert contract.validate_provenance(
        "1" * 40,
        PILOT_CONTRACT_TAG_V2_10,
    )["git_tag"] == PILOT_CONTRACT_TAG_V2_10


def test_v2_10_renderer_writes_draft_overlay_and_expanded_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_dependencies(tmp_path)
    monkeypatch.setattr(render_v210, "EXPERIMENTS", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["render_pilot_v210_contract.py", "--status", "draft"],
    )

    assert render_v210.main() == 0
    overlay = load_pilot_contract(tmp_path / "pilot_v2_10_overlay.yaml")
    expanded = load_pilot_contract(tmp_path / "pilot_v2_10.yaml")
    assert overlay.to_dict() == expanded.to_dict()
    assert overlay.contract_id == PILOT_CONTRACT_ID_V2_10
    assert len(overlay.expand()) == 211


def test_v2_10_frozen_constant_guard_precedes_overlay_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = deepcopy(_draft_overlay())
    value["status"] = "frozen"
    value["changes"]["release_requirements"]["expected_ci"] = {
        "test_count": 1,
        "test_collection_sha256": "1" * 64,
        "compiled_source_count": 1,
        "compiled_source_inventory_sha256": "2" * 64,
        "sealed_manifest_inventory_sha256": "3" * 64,
    }
    monkeypatch.setattr(
        contract_module,
        "PILOT_CONTRACT_V2_10_CANONICAL_SHA256",
        None,
    )
    with pytest.raises(PilotContractError, match="canonical hash and CI"):
        load_pilot_contract(_write_overlay(tmp_path, value))


def test_v2_10_module_exports_public_identity_constants() -> None:
    assert contract_module.PILOT_CONTRACT_ID_V2_10 == "finevo-pilot-v2.10"
    assert contract_module.PILOT_CONTRACT_TAG_V2_10 == ("pilot-v2.10-science")
    assert contract_module.PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10 == (
        "finevo-pilot-contract-v2.10-p95-runner-binding-retry-overlay-v1"
    )
