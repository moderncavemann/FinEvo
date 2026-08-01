from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import verified_memory.pilot_contract as pilot_contract_module
from scripts.render_pilot_v21110_contract import (
    FrozenCandidateBootstrapError,
    _assert_expanded_continuation_specs_match,
    _assert_v2119_recovery_delta,
    _normalized_continuation_specs,
    _parse_with_bootstrap_design_pin,
    build_contract,
)
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_11_10,
    PILOT_CONTRACT_TAG_V2_11_10,
    PilotContract,
    PilotContractError,
    canonical_contract_sha256,
    canonical_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "experiments" / "pilot_v2_11_9.yaml"
AUTHORITY_PATH = ROOT / "experiments" / "pilot_v2_11_5.yaml"


def _parse_draft(
    value: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> PilotContract:
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256",
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
            "finevo-pilot-v2.11.10--" + expected_child["run_id"][len(prefix) :]
        )
        expected_child["contract_id"] = PILOT_CONTRACT_ID_V2_11_10
        expected_child["budget_bucket"] = "hosted_v21110"
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


def test_v21110_draft_round_trip_and_exact_87_cell_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = build_contract(ROOT, status="draft")
    contract = _parse_draft(rendered, monkeypatch)
    assert contract.to_dict() == rendered
    assert contract.contract_id == PILOT_CONTRACT_ID_V2_11_10
    assert contract.status == "draft"
    assert contract.implementation["required_git_tag"] == PILOT_CONTRACT_TAG_V2_11_10
    assert contract.stage_ids == (
        "parent-import",
        "experiment-d",
        "experiment-b",
        "cross-model",
    )
    assert {stage.stage_id: stage.budget_bucket for stage in contract.stages} == {
        "parent-import": "parent_v2119",
        "experiment-d": "hosted_v21110",
        "experiment-b": "hosted_v21110",
        "cross-model": "hosted_v21110",
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
    specs = contract.expand()
    assert len(specs) == 87
    assert len({spec.run_id for spec in specs}) == 87
    assert all(spec.contract_id == PILOT_CONTRACT_ID_V2_11_10 for spec in specs)
    source = contract.v21110_recovery_boundary["source_manifest"]
    assert source["file_sha256"] == (
        pilot_contract_module.PILOT_V2_11_10_SOURCE_MANIFEST_FILE_SHA256
    )
    assert source["content_sha256"] == (
        pilot_contract_module.PILOT_V2_11_10_SOURCE_MANIFEST_CONTENT_SHA256
    )


def test_v21110_preserves_v2119_normalized_science_and_maps_v2115_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = load_pilot_contract(PARENT_PATH)
    rendered = build_contract(ROOT, status="draft")
    child = _parse_draft(rendered, monkeypatch)
    _assert_v2119_recovery_delta(parent.to_dict(), rendered, parsed_child=child)
    _assert_expanded_continuation_specs_match(parent, child)
    parent_rows = _normalized_continuation_specs(parent)
    assert parent_rows == _normalized_continuation_specs(child)
    assert canonical_sha256(parent_rows) == (
        "9968bb55b9c56ced90f56826bc8e186f72299e0a8bb40dfdb4fbb1e637af1632"
    )
    rows = _authority_mapping_rows(child)
    assert len(rows) == 86
    assert len({row["source_run_id"] for row in rows}) == 86
    assert len({row["child_run_id"] for row in rows}) == 86
    assert canonical_sha256(rows) == (
        "d876dbf6ae604a80e9cc6d29f857b944fbdcc58f0a6a279c85e88f5127468d15"
    )


def test_v21110_boundary_binds_v2119_no_go_and_cumulative_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _parse_draft(build_contract(ROOT, status="draft"), monkeypatch)
    boundary = dict(contract.v21110_recovery_boundary or {})
    no_go = boundary["failed_release_no_go"]
    assert no_go["contract_id"] == "finevo-pilot-v2.11.9"
    assert no_go["contract_sha256"] == (
        "ec16563bf906b8f6c1492d2a30f291d2c849cd639c2f314e7a1c8ac619e3fa3f"
    )
    assert no_go["science_tag_object"] == (
        "f0af244b64a69b3ee4571452df6d3611fd8c6220"
    )
    assert no_go["science_commit"] == (
        "d850902af6218c72a6b0e71275c62c81c9143fb9"
    )
    assert no_go["run_ledger"]["status_counts"] == {
        "complete": 1,
        "failed": 86,
    }
    assert no_go["budget_ledger"]["status_counts"] == {
        "complete": 1,
        "failed": 36,
    }
    assert no_go["budget_ledger"]["current_actual"] == {
        "cost_usd": 0.0,
        "hosted_completions": 0,
        "storage_bytes": 800_162,
    }
    assert no_go["provider_calls"] == 0
    assert no_go["hosted_completions"] == 0
    assert no_go["resume_forbidden"] is True
    assert no_go["failure_reclassification_forbidden"] is True
    assert set(no_go["stage_receipts"]) == {
        "parent-import",
        "experiment-d",
        "experiment-b",
        "cross-model",
    }
    assert boundary["parent_budget_debit"] == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": (
            "ec16563bf906b8f6c1492d2a30f291d2c849cd639c2f314e7a1c8ac619e3fa3f"
        ),
        "parent_run_ledger_sha256": (
            "b2891fb152825cac846955b9c2fe4a041e80eab8cbebef9bc4d861d2313fc923"
        ),
        "parent_budget_ledger_sha256": (
            "02adeb470b823664c67d09cd34df8787a68760e6270f46b59cca204701e3465d"
        ),
        "stage_bucket": "parent_v2119",
        "cost_usd": 63.1196450625,
        "hosted_completions": 3440,
        "storage_bytes": 270_993_662,
        "record_sha256": (
            "5e0c39817c32c845c2f771a02320c55e85e9a6bfb5f3e705046b822593b4c592"
        ),
    }
    budget = boundary["continuation_budget"]
    assert budget["hosted_cap_usd"] == 436.8803549375
    assert budget["fresh_registered_provider_calls"] == 3256
    assert budget["projected_cumulative_cost_usd"] == 212.4498325625
    assert budget["projected_cumulative_hosted_completions"] == 6696
    assert budget["projected_cumulative_storage_bytes"] == 1_290_993_662
    matrix = boundary["continuation_matrix"]
    assert matrix["canonical_86_row_mapping_sha256"] == (
        "d876dbf6ae604a80e9cc6d29f857b944fbdcc58f0a6a279c85e88f5127468d15"
    )
    assert matrix["logical_registered_denominator_after_cross_release_dedup"] == 136
    assert matrix["logical_scientific_denominator_after_cross_release_dedup"] == 131
    assert matrix["failed_v2119_rows_are_aborted_release_audit_only"] is True


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("failed_release_no_go", "provider_calls", 1),
        ("parent_budget_debit", "storage_bytes", 270_993_661),
        (
            "continuation_matrix",
            "logical_scientific_denominator_after_cross_release_dedup",
            217,
        ),
        ("immutability", "v2119_resume_forbidden", False),
    ],
)
def test_v21110_strict_boundary_rejects_lineage_or_denominator_drift(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    replacement: object,
) -> None:
    changed = deepcopy(build_contract(ROOT, status="draft"))
    changed["v21110_recovery_boundary"][section][field] = replacement
    _reseal(changed)
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256",
        science_design_sha256(changed),
    )
    with pytest.raises(PilotContractError, match="recovery boundary drifted"):
        PilotContract.from_dict(changed)


def test_v21110_bootstrap_is_draft_only_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = build_contract(ROOT, status="draft")
    assert first == build_contract(ROOT, status="draft")
    assert canonical_contract_sha256(first) == first["integrity"]["declared_sha256"]
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256",
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


def test_v21110_frozen_parse_fails_closed_while_pins_are_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = build_contract(ROOT, status="draft")
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256",
        science_design_sha256(draft),
    )
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_10_CANONICAL_SHA256",
        None,
    )
    with pytest.raises(
        PilotContractError,
        match="cannot be frozen before its canonical hash and CI inventory",
    ):
        build_contract(
            ROOT,
            status="frozen",
            expected_ci={
                "test_count": 1,
                "test_collection_sha256": "3" * 64,
                "compiled_source_count": 1,
                "compiled_source_inventory_sha256": "4" * 64,
                "sealed_manifest_inventory_sha256": "5" * 64,
            },
        )


def test_v21110_rejects_wrong_denominator_policy_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = deepcopy(build_contract(ROOT, status="draft"))
    changed["denominator_policy"]["policy_id"] = "finevo-pilot-v2.11.9-itt"
    _reseal(changed)
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256",
        science_design_sha256(changed),
    )
    with pytest.raises(PilotContractError, match="policy identifier drifted"):
        PilotContract.from_dict(changed)


def test_v21110_leaves_frozen_v2119_round_trip_unchanged() -> None:
    parent = load_pilot_contract(PARENT_PATH)
    assert parent.contract_id == "finevo-pilot-v2.11.9"
    assert PilotContract.from_dict(parent.to_dict()).to_dict() == parent.to_dict()
