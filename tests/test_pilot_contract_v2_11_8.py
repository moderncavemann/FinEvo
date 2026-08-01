from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import verified_memory.pilot_contract as pilot_contract_module
from scripts.render_pilot_v2118_contract import (
    FrozenCandidateBootstrapError,
    _assert_expanded_continuation_specs_match,
    _assert_v2117_recovery_delta,
    _normalized_continuation_specs,
    _parse_with_bootstrap_design_pin,
    build_contract,
)
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_11_8,
    PILOT_CONTRACT_TAG_V2_11_8,
    PilotContract,
    PilotContractError,
    canonical_contract_sha256,
    canonical_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_8.yaml"
PARENT_PATH = ROOT / "experiments" / "pilot_v2_11_7.yaml"
AUTHORITY_PATH = ROOT / "experiments" / "pilot_v2_11_5.yaml"


def _parse_draft(
    value: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> PilotContract:
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_8_SCIENCE_DESIGN_SHA256",
        science_design_sha256(value),
    )
    return PilotContract.from_dict(value)


def _reseal(value: dict[str, object]) -> None:
    integrity = value["integrity"]
    assert isinstance(integrity, dict)
    integrity["declared_sha256"] = canonical_contract_sha256(value)


def _authority_mapping_rows(child: PilotContract) -> list[dict[str, object]]:
    authority = load_pilot_contract(AUTHORITY_PATH)
    source_specs = tuple(
        spec
        for stage_id in ("experiment-d", "experiment-b", "cross-model")
        for spec in authority.expand(stage=stage_id)
    )
    child_specs = tuple(
        spec
        for stage_id in ("experiment-d", "experiment-b", "cross-model")
        for spec in child.expand(stage=stage_id)
    )
    child_by_id = {spec.run_id: spec.to_dict() for spec in child_specs}
    rows: list[dict[str, object]] = []
    for source_spec in sorted(source_specs, key=lambda spec: spec.run_id):
        source = source_spec.to_dict()
        expected_child = deepcopy(source)
        prefix = "finevo-pilot-v2.11.5--"
        assert expected_child["run_id"].startswith(prefix)
        expected_child["run_id"] = (
            "finevo-pilot-v2.11.8--" + expected_child["run_id"][len(prefix) :]
        )
        expected_child["contract_id"] = PILOT_CONTRACT_ID_V2_11_8
        expected_child["budget_bucket"] = "hosted_v2118"
        observed_child = child_by_id[expected_child["run_id"]]
        assert observed_child == expected_child
        logical = deepcopy(source)
        logical.pop("run_id")
        logical.pop("contract_id")
        logical["budget_bucket"] = "normalized-hosted-continuation"
        rows.append(
            {
                "source_run_id": source["run_id"],
                "child_run_id": observed_child["run_id"],
                "logical_cell_sha256": canonical_sha256(logical),
                "source_spec_sha256": canonical_sha256(source),
                "child_spec_sha256": canonical_sha256(observed_child),
                "normalized_spec": logical,
            }
        )
    return rows


def test_v2118_draft_round_trip_and_exact_87_cell_recovery(
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
    assert contract.contract_id == PILOT_CONTRACT_ID_V2_11_8
    assert contract.status == "draft"
    assert contract.implementation["required_git_tag"] == PILOT_CONTRACT_TAG_V2_11_8
    assert contract.stage_ids == (
        "parent-import",
        "experiment-d",
        "experiment-b",
        "cross-model",
    )
    assert {stage.stage_id: stage.budget_bucket for stage in contract.stages} == {
        "parent-import": "parent_v2117",
        "experiment-d": "hosted_v2118",
        "experiment-b": "hosted_v2118",
        "cross-model": "hosted_v2118",
    }
    assert {
        stage.stage_id: len(contract.expand(stage=stage.stage_id))
        for stage in contract.stages
    } == {
        "parent-import": 1,
        "experiment-d": 55,
        "experiment-b": 25,
        "cross-model": 6,
    }
    assert len(contract.expand()) == 87
    source = contract.v2118_recovery_boundary["source_manifest"]
    assert source["file_sha256"] == (
        pilot_contract_module.PILOT_V2_11_8_SOURCE_MANIFEST_FILE_SHA256
    )
    assert source["content_sha256"] == (
        pilot_contract_module.PILOT_V2_11_8_SOURCE_MANIFEST_CONTENT_SHA256
    )
    assert source["file_sha256"] is not None
    assert source["content_sha256"] is not None
    assert load_pilot_contract(CONTRACT_PATH).to_dict() == tracked


def test_v2118_preserves_v2117_normalized_science_and_maps_v2115_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = load_pilot_contract(PARENT_PATH)
    rendered = build_contract(ROOT, status="draft")
    child = _parse_draft(rendered, monkeypatch)
    _assert_v2117_recovery_delta(parent.to_dict(), rendered, parsed_child=child)
    _assert_expanded_continuation_specs_match(parent, child)
    assert _normalized_continuation_specs(parent) == _normalized_continuation_specs(
        child
    )
    rows = _authority_mapping_rows(child)
    assert len(rows) == 86
    assert len({row["source_run_id"] for row in rows}) == 86
    assert len({row["child_run_id"] for row in rows}) == 86
    assert canonical_sha256(rows) == (
        "812781508a1cbfb8a827de0a981c3b6e189cff497decb7e9343ce6f8aa4d4ca5"
    )


def test_v2118_boundary_preserves_no_go_denominator_and_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _parse_draft(build_contract(ROOT, status="draft"), monkeypatch)
    boundary = dict(contract.v2118_recovery_boundary or {})
    no_go = boundary["failed_release_no_go"]
    assert no_go["contract_id"] == "finevo-pilot-v2.11.7"
    assert no_go["provider_calls"] == 0
    assert no_go["provider_construction"] is False
    assert no_go["resume_forbidden"] is True
    assert no_go["run_ledger"]["status_counts"] == {"integrity-stopped": 87}
    assert no_go["budget_ledger"]["current_actual"] == {
        "cost_usd": 0.0,
        "hosted_completions": 0,
        "storage_bytes": 1797,
    }
    assert boundary["parent_budget_debit"] == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": (
            "376c41f7b2793d4039bae43a652d6ba73759cce7b9b3f04fc665c41a23659e3b"
        ),
        "parent_run_ledger_sha256": (
            "bb6d497308097cf6f348c282339f2f6d4cb6721950604744c1e6b0751e913681"
        ),
        "parent_budget_ledger_sha256": (
            "bc6cc622beaff05e2480e866408929f3edd7f02a7555bdb26202fe94ae3e9c77"
        ),
        "stage_bucket": "parent_v2117",
        "cost_usd": 63.1196450625,
        "hosted_completions": 3440,
        "storage_bytes": 270191728,
        "record_sha256": (
            "a8281fea88c404d504792b08d8bef75ee5d33d890ee5a44ed91962012ba87f1e"
        ),
    }
    budget = boundary["continuation_budget"]
    assert budget["hosted_cap_usd"] == 436.8803549375
    assert budget["projected_cumulative_storage_bytes"] == 1_290_191_728
    matrix = boundary["continuation_matrix"]
    assert matrix["combined_registered_denominator"] == 136
    assert matrix["logical_registered_denominator_after_cross_release_dedup"] == 136
    assert matrix["logical_scientific_denominator_after_cross_release_dedup"] == 131
    assert matrix["failed_v2117_rows_are_aborted_release_audit_only"] is True


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("failed_release_no_go", "provider_calls", 1),
        ("parent_budget_debit", "storage_bytes", 270_189_931),
        (
            "continuation_matrix",
            "logical_scientific_denominator_after_cross_release_dedup",
            217,
        ),
        ("immutability", "v2117_resume_forbidden", False),
    ],
)
def test_v2118_strict_boundary_rejects_lineage_or_denominator_drift(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    replacement: object,
) -> None:
    changed = deepcopy(build_contract(ROOT, status="draft"))
    changed["v2118_recovery_boundary"][section][field] = replacement
    _reseal(changed)
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_8_SCIENCE_DESIGN_SHA256",
        science_design_sha256(changed),
    )
    with pytest.raises(PilotContractError, match="recovery boundary drifted"):
        PilotContract.from_dict(changed)


def test_v2118_bootstrap_is_draft_only_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = build_contract(ROOT, status="draft")
    assert first == build_contract(ROOT, status="draft")
    assert canonical_contract_sha256(first) == first["integrity"]["declared_sha256"]
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_8_SCIENCE_DESIGN_SHA256",
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


def test_v2118_rejects_wrong_denominator_policy_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = deepcopy(build_contract(ROOT, status="draft"))
    changed["denominator_policy"]["policy_id"] = "finevo-pilot-v2.11.7-itt"
    _reseal(changed)
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_8_SCIENCE_DESIGN_SHA256",
        science_design_sha256(changed),
    )
    with pytest.raises(PilotContractError, match="policy identifier drifted"):
        PilotContract.from_dict(changed)


def test_v2118_explicit_frozen_candidate_bootstrap_round_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = build_contract(ROOT, status="draft")
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_8_SCIENCE_DESIGN_SHA256",
        science_design_sha256(draft),
    )
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_V2_11_8_SOURCE_MANIFEST_FILE_SHA256",
        "1" * 64,
    )
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_V2_11_8_SOURCE_MANIFEST_CONTENT_SHA256",
        "2" * 64,
    )
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_8_CANONICAL_SHA256",
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
