from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import verified_memory.pilot_contract as pilot_contract_module
from scripts.render_pilot_v2117_contract import (
    FrozenCandidateBootstrapError,
    _assert_expanded_continuation_specs_match,
    _assert_v2116_recovery_delta,
    _normalized_continuation_specs,
    _parse_with_bootstrap_design_pin,
    build_contract,
)
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_11_7,
    PILOT_CONTRACT_TAG_V2_11_7,
    PilotContract,
    PilotContractError,
    canonical_contract_sha256,
    canonical_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_7.yaml"
PARENT_PATH = ROOT / "experiments" / "pilot_v2_11_6.yaml"


def _parse_draft(
    value: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> PilotContract:
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_7_SCIENCE_DESIGN_SHA256",
        science_design_sha256(value),
    )
    return PilotContract.from_dict(value)


def _reseal(value: dict[str, object]) -> None:
    integrity = value["integrity"]
    assert isinstance(integrity, dict)
    integrity["declared_sha256"] = canonical_contract_sha256(value)


def test_v2117_draft_round_trip_and_exact_87_cell_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = build_contract(ROOT, status="draft")
    tracked = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert tracked["status"] == "frozen"
    assert build_contract(
        ROOT,
        status="frozen",
        expected_ci=tracked["release_requirements"]["expected_ci"],
    ) == tracked
    contract = _parse_draft(rendered, monkeypatch)
    assert contract.to_dict() == rendered
    assert contract.contract_id == PILOT_CONTRACT_ID_V2_11_7
    assert contract.status == "draft"
    assert contract.implementation["required_git_tag"] == PILOT_CONTRACT_TAG_V2_11_7
    assert contract.stage_ids == (
        "parent-import",
        "experiment-d",
        "experiment-b",
        "cross-model",
    )
    assert {stage.stage_id: stage.budget_bucket for stage in contract.stages} == {
        "parent-import": "parent_v2116",
        "experiment-d": "hosted_v2117",
        "experiment-b": "hosted_v2117",
        "cross-model": "hosted_v2117",
    }
    assert {stage.stage_id: len(contract.expand(stage=stage.stage_id)) for stage in contract.stages} == {
        "parent-import": 1,
        "experiment-d": 55,
        "experiment-b": 25,
        "cross-model": 6,
    }
    assert len(contract.expand()) == 87
    assert load_pilot_contract(CONTRACT_PATH).to_dict() == tracked


def test_v2117_maps_exactly_the_v2116_86_logical_science_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = load_pilot_contract(PARENT_PATH)
    rendered = build_contract(ROOT, status="draft")
    child = _parse_draft(rendered, monkeypatch)
    _assert_v2116_recovery_delta(parent.to_dict(), rendered, parsed_child=child)
    _assert_expanded_continuation_specs_match(parent, child)
    parent_rows = _normalized_continuation_specs(parent)
    child_rows = _normalized_continuation_specs(child)
    assert parent_rows == child_rows
    assert len(parent_rows) == 86
    assert canonical_sha256(parent_rows) == (
        "9968bb55b9c56ced90f56826bc8e186f72299e0a8bb40dfdb4fbb1e637af1632"
    )


def test_v2117_boundary_preserves_denominator_and_binds_no_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _parse_draft(build_contract(ROOT, status="draft"), monkeypatch)
    boundary = dict(contract.v2117_recovery_boundary or {})
    no_go = boundary["failed_release_no_go"]
    assert no_go["provider_calls"] == 0
    assert no_go["resume_forbidden"] is True
    assert no_go["run_ledger"]["status_counts"] == {"integrity-stopped": 87}
    decomposition = boundary["authority_current_actual_decomposition"]
    assert decomposition["hosted_v2115"]["row_count"] == 47
    assert decomposition["operational_parent_v2114"]["row_count"] == 3
    assert decomposition["all_current"] == {
        "row_count": 50,
        "cost_usd": 43.1214245,
        "hosted_completions": 2436,
        "storage_bytes": 48_139_533,
    }
    matrix = boundary["continuation_matrix"]
    assert matrix["combined_registered_denominator"] == 136
    assert matrix["logical_registered_denominator_after_cross_release_dedup"] == 136
    assert matrix["logical_scientific_denominator_after_cross_release_dedup"] == 131
    assert matrix["canonical_86_row_mapping_sha256"] == (
        "88aef768d311653c8335f7ad769400c84e0c0430c9c82183611f87d0f6906fcd"
    )


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("failed_release_no_go", "provider_calls", 1),
        ("authority_current_actual_decomposition", "observed_storage_difference_bytes", 0),
        ("continuation_matrix", "logical_scientific_denominator_after_cross_release_dedup", 217),
        ("immutability", "v2116_resume_forbidden", False),
    ],
)
def test_v2117_strict_boundary_rejects_lineage_or_denominator_drift(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    replacement: object,
) -> None:
    changed = deepcopy(build_contract(ROOT, status="draft"))
    changed["v2117_recovery_boundary"][section][field] = replacement
    _reseal(changed)
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_7_SCIENCE_DESIGN_SHA256",
        science_design_sha256(changed),
    )
    with pytest.raises(PilotContractError, match="recovery boundary drifted"):
        PilotContract.from_dict(changed)


def test_v2117_bootstrap_is_draft_only_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = build_contract(ROOT, status="draft")
    assert first == build_contract(ROOT, status="draft")
    assert canonical_contract_sha256(first) == first["integrity"]["declared_sha256"]
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_7_SCIENCE_DESIGN_SHA256",
        None,
    )
    assert _parse_with_bootstrap_design_pin(first).canonical_hash == (
        first["integrity"]["declared_sha256"]
    )
    frozen = deepcopy(first)
    frozen["status"] = "frozen"
    _reseal(frozen)
    with pytest.raises(FrozenCandidateBootstrapError):
        _parse_with_bootstrap_design_pin(frozen)


def test_v2117_rejects_wrong_denominator_policy_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = deepcopy(build_contract(ROOT, status="draft"))
    changed["denominator_policy"]["policy_id"] = "finevo-pilot-v2.11.6-itt"
    _reseal(changed)
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_7_SCIENCE_DESIGN_SHA256",
        science_design_sha256(changed),
    )
    with pytest.raises(PilotContractError, match="policy identifier drifted"):
        PilotContract.from_dict(changed)


def test_v2117_explicit_frozen_candidate_bootstrap_round_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = build_contract(ROOT, status="draft")
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_7_SCIENCE_DESIGN_SHA256",
        science_design_sha256(draft),
    )
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_V2_11_7_SOURCE_MANIFEST_FILE_SHA256",
        "1" * 64,
    )
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_V2_11_7_SOURCE_MANIFEST_CONTENT_SHA256",
        "2" * 64,
    )
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_7_CANONICAL_SHA256",
        None,
    )
    frozen = build_contract(
        ROOT,
        status="frozen",
        expected_ci={
            "test_count": 1,
            "test_collection_sha256": "3" * 64,
            "compiled_source_count": 1,
            "compiled_source_inventory_sha256": "4" * 64,
            "sealed_manifest_inventory_sha256": "5" * 64,
        },
        frozen_candidate=True,
    )
    assert frozen["status"] == "frozen"
    assert canonical_contract_sha256(frozen) == frozen["integrity"]["declared_sha256"]
