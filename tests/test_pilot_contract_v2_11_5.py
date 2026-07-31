from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.render_pilot_v2115_contract import (
    _assert_expanded_science_specs_match,
    _assert_v2114_science_delta,
    build_contract,
)
import verified_memory.pilot_contract as pilot_contract_module
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_11_5,
    PILOT_CONTRACT_TAG_V2_11_5,
    PILOT_CONTRACT_V2_11_5_SCIENCE_DESIGN_SHA256,
    PILOT_V2_11_5_SOURCE_MANIFEST_CONTENT_SHA256,
    PILOT_V2_11_5_SOURCE_MANIFEST_FILE_SHA256,
    PilotContract,
    PilotContractError,
    canonical_contract_sha256,
    canonical_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_5.yaml"
PARENT_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_4.yaml"
SOURCE_MANIFEST_PATH = (
    ROOT / "experiments" / "pilot_v2_11_5_source_manifest.json"
)
OPERATIONAL_STAGE_IDS = {
    "parent-import",
    "capability-gate",
    "long-context-preflight",
}


def _rehash(value: dict) -> dict:
    candidate = deepcopy(value)
    candidate["integrity"]["declared_sha256"] = "0" * 64
    candidate["integrity"]["declared_sha256"] = canonical_contract_sha256(
        candidate
    )
    return candidate


def test_v2115_tracked_draft_binds_no_go_lineage_and_fresh_matrix() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    specs = contract.expand()

    assert contract.contract_id == PILOT_CONTRACT_ID_V2_11_5
    assert contract.status == "frozen"
    assert contract.implementation["required_git_tag"] == PILOT_CONTRACT_TAG_V2_11_5
    assert contract.release_requirements is not None
    assert dict(contract.release_requirements.expected_ci) == {
        "test_count": 1808,
        "test_collection_sha256": (
            "a51152f5afbbba62740f404438e2aed579bd3c0a4f607989835b1c5ff99beb60"
        ),
        "compiled_source_count": 280,
        "compiled_source_inventory_sha256": (
            "06ccd348a6296d7796eef8898457d69df9a620e1e29d1bdb7b7f33b671903bd2"
        ),
        "sealed_manifest_inventory_sha256": (
            "b5c5a817d09d10752c1f5f00ba556b417d16e06c64b5fcbb15671e49a1d81952"
        ),
    }
    assert len(specs) == 136
    assert len({spec.run_id for spec in specs}) == 136
    assert Counter(spec.stage_id for spec in specs) == {
        "parent-import": 1,
        "capability-gate": 2,
        "long-context-preflight": 2,
        "experiment-c": 25,
        "experiment-a": 20,
        "experiment-d": 55,
        "experiment-b": 25,
        "cross-model": 6,
    }
    assert len([s for s in specs if s.stage_id in OPERATIONAL_STAGE_IDS]) == 5
    assert len([s for s in specs if s.stage_id not in OPERATIONAL_STAGE_IDS]) == 131

    boundary = contract.v2115_forward_boundary
    assert boundary is not None
    assert boundary["parent"]["status_counts"] == {
        "complete": 5,
        "scheduled": 131,
    }
    assert boundary["parent"]["acceptance_receipt_present"] is False
    assert boundary["parent"]["fresh_provider_calls"] == 0
    assert boundary["parent"]["fresh_cost_usd"] == 0.0
    assert boundary["parent_budget_debit"] == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": (
            "e898fe49935dae9ae7f0d7ac577dae943192953c1da581d70c334f8c64924e46"
        ),
        "parent_run_ledger_sha256": (
            "f0064120e279137fbd7dd5f5cec474aa384745b7c405d9371c90ac5c4f448656"
        ),
        "parent_budget_ledger_sha256": (
            "d4ce8beebe1e462003039db2d39e6616c76cd33897c5b3e77e45989ced9d8789"
        ),
        "stage_bucket": "parent_v2114",
        "cost_usd": 19.998220562500006,
        "hosted_completions": 1004,
        "storage_bytes": 222048702,
        "record_sha256": (
            "9595d037a21f429a59fd37febd4abd8283287e080ee9eb506ebf999e3d1e81a5"
        ),
    }
    assert boundary["matrix"]["scientific_cells"] == 131
    assert boundary["matrix"]["fresh_scientific_provider_calls"] == 5816
    assert contract.budgets["stage_usd_caps"] == {
        "parent_v2114": 19.998220562500006,
        "hosted_v2115": 480.0017794375,
        "manual_reserve": 0.0,
    }


def test_v2115_normalizes_exactly_nine_stable_and_eight_generation_fields() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    amendment = contract.v2115_consumer_authority_normalization_amendment
    assert amendment is not None
    repair = amendment["consumer_authority_normalization_repair"]

    assert repair["stable_field_count"] == 9
    assert len(repair["stable_authority_fields"]) == 9
    assert repair["generation_field_count"] == 8
    assert len(repair["generation_authority_fields"]) == 8
    assert set(repair["source_generation_values"]) == set(
        repair["generation_authority_fields"]
    )
    assert set(repair["parent_generation_values"]) == set(
        repair["generation_authority_fields"]
    )
    assert set(repair["current_generation_rules"]) == set(
        repair["generation_authority_fields"]
    )
    payload_hashes = repair["reservation_payload_sha256_by_model_call_kind"]
    assert set(payload_hashes) == {"gpt52_main", "gpt56_diagnostic"}
    assert all(set(rows) == {"action", "semantic"} for rows in payload_hashes.values())
    assert repair["reservation_payload_exact_equality_required"] is True
    assert repair["stable_field_exact_equality_required"] is True
    assert repair["validation_before_provider_construction"] is True
    assert repair["provider_calls"] == 0


def test_v2115_source_manifest_is_canonical_and_matches_contract_repair() -> None:
    raw = SOURCE_MANIFEST_PATH.read_bytes()
    manifest = json.loads(raw)
    content = deepcopy(manifest)
    declared = content["integrity"].pop("content_sha256")

    assert hashlib.sha256(raw).hexdigest() == (
        PILOT_V2_11_5_SOURCE_MANIFEST_FILE_SHA256
    )
    assert declared == PILOT_V2_11_5_SOURCE_MANIFEST_CONTENT_SHA256
    assert canonical_sha256(content) == declared
    assert raw == (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    contract = load_pilot_contract(CONTRACT_PATH)
    amendment = contract.v2115_consumer_authority_normalization_amendment
    assert amendment is not None
    thawed_amendment = contract.to_dict()[
        "v2115_consumer_authority_normalization_amendment"
    ]
    assert manifest["consumer_authority_normalization"] == (
        thawed_amendment["consumer_authority_normalization_repair"]
    )
    lineage = manifest["terminal_lineage_release"]
    assert lineage["publication_status"] == (
        "immutable-pre-dispatch-acceptance-no-go"
    )
    assert lineage["run_ledger"]["status_counts"] == {
        "complete": 5,
        "scheduled": 131,
    }
    assert lineage["scientific_dispatch_acceptance"]["present"] is False


def test_v2115_renderer_is_deterministic_and_science_specs_match_parent() -> None:
    tracked = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    rendered = build_contract(
        ROOT,
        status="frozen",
        expected_ci=tracked["release_requirements"]["expected_ci"],
    )
    assert rendered == tracked
    assert science_design_sha256(rendered) == (
        PILOT_CONTRACT_V2_11_5_SCIENCE_DESIGN_SHA256
    )

    parent = load_pilot_contract(PARENT_CONTRACT_PATH)
    child = PilotContract.from_dict(rendered)
    _assert_v2114_science_delta(parent.to_dict(), rendered)
    _assert_expanded_science_specs_match(parent, child)


def test_v2115_strict_contract_rejects_normalization_or_parent_fact_drift() -> None:
    source = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    changed = deepcopy(source)
    changed["v2115_consumer_authority_normalization_amendment"][
        "consumer_authority_normalization_repair"
    ]["stable_authority_fields"] = changed[
        "v2115_consumer_authority_normalization_amendment"
    ]["consumer_authority_normalization_repair"]["stable_authority_fields"][:-1]
    with pytest.raises(PilotContractError, match="normalization amendment drifted"):
        PilotContract.from_dict(_rehash(changed))

    changed = deepcopy(source)
    changed["v2115_forward_boundary"]["parent"]["status_counts"]["scheduled"] = 130
    with pytest.raises(PilotContractError, match="forward boundary drifted"):
        PilotContract.from_dict(_rehash(changed))


def test_v2115_frozen_requires_a_pinned_canonical_identity(monkeypatch) -> None:
    source = build_contract(ROOT, status="draft")
    source["status"] = "frozen"
    source["release_requirements"]["expected_ci"] = {
        "test_count": 1,
        "test_collection_sha256": "1" * 64,
        "compiled_source_count": 1,
        "compiled_source_inventory_sha256": "2" * 64,
        "sealed_manifest_inventory_sha256": "3" * 64,
    }
    source = _rehash(source)
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_5_CANONICAL_SHA256",
        None,
    )
    with pytest.raises(PilotContractError, match="cannot be frozen"):
        PilotContract.from_dict(source)
