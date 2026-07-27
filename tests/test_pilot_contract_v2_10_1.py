from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import pytest

import scripts.render_pilot_v2101_contract as render_v2101
from verified_memory import pilot_contract as contract_module
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_10,
    PILOT_CONTRACT_ID_V2_10_1,
    PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10_1,
    PILOT_CONTRACT_SCHEMA_VERSION_V2,
    PILOT_CONTRACT_TAG_V2_10_1,
    PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256,
    PILOT_CONTRACT_V2_10_CANONICAL_SHA256,
    PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256,
    PilotContractError,
    _v2_10_1_expected_qref_receipt_verifier_retry_amendment,
    canonical_contract_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
V210_PATH = EXPERIMENTS / "pilot_v2_10.yaml"
OVERLAY_PATH = EXPERIMENTS / "pilot_v2_10_1_overlay.yaml"
EXPANDED_PATH = EXPERIMENTS / "pilot_v2_10_1.yaml"
DEPENDENCIES = (
    V210_PATH,
    EXPERIMENTS / "pilot_v2_4_parent_source_manifest.json",
    EXPERIMENTS / "pilot_v2_5_source_manifest.json",
    EXPERIMENTS / "pilot_v2_6_source_manifest.json",
    EXPERIMENTS / "pilot_v2_7_source_manifest.json",
    EXPERIMENTS / "pilot_v2_8_source_manifest.json",
    EXPERIMENTS / "pilot_v2_9_source_manifest.json",
    EXPERIMENTS / "pilot_v2_10_source_manifest.json",
)


def _draft_overlay() -> dict[str, Any]:
    return json.loads(OVERLAY_PATH.read_text(encoding="utf-8"))


def _copy_dependencies(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for source in DEPENDENCIES:
        (target / source.name).write_bytes(source.read_bytes())


def _write_overlay(target: Path, value: dict[str, Any]) -> Path:
    _copy_dependencies(target)
    value["integrity"]["declared_sha256"] = canonical_contract_sha256(value)
    path = target / "pilot_v2_10_1_overlay.yaml"
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


def test_v2_10_1_frozen_preserves_exact_frozen_v2_10_design() -> None:
    parent = load_pilot_contract(V210_PATH)
    overlay = load_pilot_contract(OVERLAY_PATH)
    expanded = load_pilot_contract(EXPANDED_PATH)

    assert overlay.to_dict() == expanded.to_dict()
    assert overlay.contract_id == PILOT_CONTRACT_ID_V2_10_1
    assert overlay.status == "frozen"
    assert overlay.canonical_hash == PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256
    assert overlay.implementation["required_git_tag"] == (PILOT_CONTRACT_TAG_V2_10_1)
    assert overlay.release_requirements is not None
    assert overlay.release_requirements.tag == PILOT_CONTRACT_TAG_V2_10_1
    assert overlay.denominator_policy is not None
    assert overlay.denominator_policy.policy_id == ("finevo-pilot-v2.10.1-itt")
    assert len(parent.expand()) == len(overlay.expand()) == 211
    assert _normalized_specs(V210_PATH) == _normalized_specs(OVERLAY_PATH)
    assert science_design_sha256(overlay.to_dict()) == (
        PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256
    )
    assert overlay.budgets == parent.budgets
    assert Counter(spec.stage_id for spec in overlay.expand()) == {
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


def test_v2_10_1_amendment_freezes_terminal_no_go_and_hash_domain() -> None:
    amendment = _v2_10_1_expected_qref_receipt_verifier_retry_amendment(status="draft")

    assert amendment["source_manifest"] == {
        "path": "experiments/pilot_v2_10_1_source_manifest.json",
        "schema_version": "finevo-pilot-v2.10.1-source-manifest-v1",
        "file_sha256": None,
        "content_sha256": None,
    }
    failure = amendment["failure_classification"]
    assert failure["parent_contract_id"] == PILOT_CONTRACT_ID_V2_10
    assert failure["parent_contract_sha256"] == (PILOT_CONTRACT_V2_10_CANONICAL_SHA256)
    assert failure["status_counts"] == {
        "complete": 1,
        "integrity-stopped": 210,
    }
    assert failure["incremental_cost_usd"] == 0.0
    assert failure["incremental_hosted_completions"] == 0
    assert failure["parent_evidence_commit"] == (
        "1e96373fa847b44e3418a777c1ed74165ecf2bac"
    )
    assert amendment["source_lineage"]["amendment_parent_raw_inventory"] == {
        "schema_version": "finevo-raw-tree-inventory-v1",
        "canonicalization": "json-sort-keys-compact-utf8-v1",
        "file_count": 637,
        "storage_bytes": 20_126_496,
        "inventory_sha256": (
            "d8964a15abed0d77598d2c2cf80136e438b67559796cc93f8566dca17e584baa"
        ),
    }
    repair = amendment["qref_receipt_verifier_repair"]
    assert repair["artifact_schema_version"] == ("finevo-pilot-stage-receipt-v2")
    assert repair["content_hash_projection"] == (
        "canonical-json-of-artifact-after-removing-entire-integrity-object"
    )
    assert repair["generic_self_hash_convention_for_stage_receipt_v2"] == ("forbidden")
    assert repair["validation_before_provider_construction"] is True
    carry = amendment["budget_carry_forward"]
    assert carry["total_cap_usd"] == 500.0
    assert carry["cumulative_prior"]["cost_usd"] == 3.212770875
    assert carry["cumulative_prior"]["hosted_completions"] == 184
    assert carry["budget_reset"] is False


def test_v2_10_1_rejects_parent_amendment_and_design_tampering(
    tmp_path: Path,
) -> None:
    parent = _draft_overlay()
    parent["base_contract"]["canonical_sha256"] = "1" * 64
    with pytest.raises(PilotContractError, match="base contract binding"):
        load_pilot_contract(_write_overlay(tmp_path / "parent", parent))

    amendment = _draft_overlay()
    amendment["qref_receipt_verifier_retry_amendment"]["qref_receipt_verifier_repair"][
        "generic_self_hash_convention_for_stage_receipt_v2"
    ] = "allowed"
    with pytest.raises(PilotContractError, match="receipt-verifier retry"):
        load_pilot_contract(_write_overlay(tmp_path / "amendment", amendment))

    expanded = load_pilot_contract(EXPANDED_PATH).to_dict()
    expanded["seeds"]["sets"]["main"][0] += 1
    expanded["integrity"]["declared_sha256"] = canonical_contract_sha256(expanded)
    path = tmp_path / "tampered-expanded.yaml"
    path.write_text(
        json.dumps(expanded, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PilotContractError, match="science-design|seed"):
        load_pilot_contract(path)


def test_v2_10_1_renderer_writes_draft_overlay_and_expanded_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_dependencies(tmp_path)
    monkeypatch.setattr(render_v2101, "EXPERIMENTS", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["render_pilot_v2101_contract.py", "--status", "draft"],
    )

    assert render_v2101.main() == 0
    overlay = load_pilot_contract(tmp_path / "pilot_v2_10_1_overlay.yaml")
    expanded = load_pilot_contract(tmp_path / "pilot_v2_10_1.yaml")
    assert overlay.to_dict() == expanded.to_dict()
    assert overlay.contract_id == PILOT_CONTRACT_ID_V2_10_1
    assert len(overlay.expand()) == 211


def test_v2_10_1_frozen_validates_paid_provenance_identity() -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    provenance = contract.validate_provenance(
        "1" * 40,
        PILOT_CONTRACT_TAG_V2_10_1,
    )
    assert provenance["contract_sha256"] == (
        PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256
    )
    assert provenance["git_tag"] == PILOT_CONTRACT_TAG_V2_10_1
    with pytest.raises(PilotContractError, match="requires annotated tag"):
        contract.validate_provenance("1" * 40, "pilot-v2.10-science")
    assert PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256 is not None


def test_v2_10_1_module_exports_public_identity_constants() -> None:
    assert contract_module.PILOT_CONTRACT_ID_V2_10_1 == ("finevo-pilot-v2.10.1")
    assert contract_module.PILOT_CONTRACT_TAG_V2_10_1 == ("pilot-v2.10.1-science")
    assert contract_module.PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10_1 == (
        "finevo-pilot-contract-v2.10.1-qref-receipt-verifier-retry-overlay-v1"
    )
    assert "PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10_1" in (contract_module.__all__)
    assert PILOT_CONTRACT_SCHEMA_VERSION_V2 == "finevo-pilot-contract-v2"
