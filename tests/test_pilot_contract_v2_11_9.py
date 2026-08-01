from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import verified_memory.pilot_contract as pilot_contract_module
from scripts.render_pilot_v2119_contract import (
    FrozenCandidateBootstrapError,
    _assert_expanded_continuation_specs_match,
    _assert_v2118_recovery_delta,
    _normalized_continuation_specs,
    _parse_with_bootstrap_design_pin,
    build_contract,
)
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_11_9,
    PILOT_CONTRACT_TAG_V2_11_9,
    PilotContract,
    PilotContractError,
    canonical_contract_sha256,
    canonical_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_9.yaml"
PARENT_PATH = ROOT / "experiments" / "pilot_v2_11_8.yaml"
AUTHORITY_PATH = ROOT / "experiments" / "pilot_v2_11_5.yaml"


def _parse_draft(
    value: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> PilotContract:
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_9_SCIENCE_DESIGN_SHA256",
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
            "finevo-pilot-v2.11.9--" + expected_child["run_id"][len(prefix) :]
        )
        expected_child["contract_id"] = PILOT_CONTRACT_ID_V2_11_9
        expected_child["budget_bucket"] = "hosted_v2119"
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


def test_v2119_draft_round_trip_and_exact_87_cell_recovery(
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
    assert pilot_contract_module.PILOT_CONTRACT_V2_11_9_CANONICAL_SHA256 == (
        "ec16563bf906b8f6c1492d2a30f291d2c849cd639c2f314e7a1c8ac619e3fa3f"
    )
    assert pilot_contract_module.PILOT_CONTRACT_V2_11_9_SCIENCE_DESIGN_SHA256 == (
        "ad2609dbc1b2d736560bcfc874d2af5899f7a048a0b6aeadbe2e350f91244e01"
    )
    contract = _parse_draft(rendered, monkeypatch)
    assert contract.to_dict() == rendered
    assert contract.contract_id == PILOT_CONTRACT_ID_V2_11_9
    assert contract.status == "draft"
    assert contract.implementation["required_git_tag"] == PILOT_CONTRACT_TAG_V2_11_9
    assert contract.stage_ids == (
        "parent-import",
        "experiment-d",
        "experiment-b",
        "cross-model",
    )
    assert {stage.stage_id: stage.budget_bucket for stage in contract.stages} == {
        "parent-import": "parent_v2118",
        "experiment-d": "hosted_v2119",
        "experiment-b": "hosted_v2119",
        "cross-model": "hosted_v2119",
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
    source = contract.v2119_recovery_boundary["source_manifest"]
    assert source["file_sha256"] == (
        pilot_contract_module.PILOT_V2_11_9_SOURCE_MANIFEST_FILE_SHA256
    )
    assert source["content_sha256"] == (
        pilot_contract_module.PILOT_V2_11_9_SOURCE_MANIFEST_CONTENT_SHA256
    )
    assert source["file_sha256"] == (
        "609adf9d12543b4caa7adb0cbddb8c8a9073a10f689adf52a8670608d16e9cb1"
    )
    assert source["content_sha256"] == (
        "36a790fe5edd6269218d6010046ec9293c3c418d8bc58a4dd5d89a6a70a547d6"
    )
    assert contract.v2119_recovery_boundary["runtime_input_binding"] == {
        "cwd_must_equal_release_root": True,
        "profile_path": "data/profiles.json",
        "profile_file_sha256": (
            "1bc90a92ef8e32f3da6e474f787207b79b1c82cc0b7b13c5ea3bd6cd1439b223"
        ),
        "profile_regular_non_symlink_required": True,
        "verification_points": (
            "scientific-dispatch-acceptance",
            "each-scientific-stage-before-provider-construction",
        ),
    }
    assert contract.v2119_recovery_boundary["source_coverage"] == {
        "complete_verified_memory_python_tree": True,
        "complete_foundation_python_tree": True,
        "llm_provider_and_unique_cli": True,
        "release_renderers": True,
        "contract_module_normalization": "three-literal-v2119-cycle-pins-only",
        "ci_module_normalization": "two-literal-v2119-source-anchor-pins-only",
    }


def test_v2119_preserves_v2118_normalized_science_and_maps_v2115_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = load_pilot_contract(PARENT_PATH)
    rendered = build_contract(ROOT, status="draft")
    child = _parse_draft(rendered, monkeypatch)
    _assert_v2118_recovery_delta(parent.to_dict(), rendered, parsed_child=child)
    _assert_expanded_continuation_specs_match(parent, child)
    assert _normalized_continuation_specs(parent) == _normalized_continuation_specs(
        child
    )
    rows = _authority_mapping_rows(child)
    assert len(rows) == 86
    assert len({row["source_run_id"] for row in rows}) == 86
    assert len({row["child_run_id"] for row in rows}) == 86
    assert canonical_sha256(rows) == (
        "7d958b7e1c9caf1ac7a2b019c534b6ff7b599b8ec420c2b9ab386ea678a70346"
    )


def test_v2119_boundary_preserves_no_go_denominator_and_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _parse_draft(build_contract(ROOT, status="draft"), monkeypatch)
    boundary = dict(contract.v2119_recovery_boundary or {})
    no_go = boundary["failed_release_no_go"]
    assert no_go["contract_id"] == "finevo-pilot-v2.11.8"
    assert no_go["contract_sha256"] == (
        "25d43667520633c5dfa299a693fd4a42736524c2737c2acf6422e2d32f0106c8"
    )
    assert no_go["science_tag_object"] == ("a5564d374762aed5ea2493706888e2950b6e97fa")
    assert no_go["science_commit"] == ("67aa0fcce68fa5ac43b48dd3b81b849112137093")
    assert no_go["contract_file_sha256"] == (
        "c355c1f1fe7eaa3571f4101f2770bd3c9ef8a5fc41553c337439b7aa1148390a"
    )
    assert no_go["source_manifest_file_sha256"] == (
        "acfc9dc6c751e8ab9f314133de856bae7a0a4021c067f693ed8ebff938b230a6"
    )
    assert no_go["source_manifest_content_sha256"] == (
        "104b63db289234820aebf14f42808c26cd01d9f8a19029fef793887bfff47cd3"
    )
    assert no_go["raw_inventory"] == {
        "root": "experiment_results/pilot-v2.11.8/raw",
        "canonicalization": "json-sort-keys-compact-utf8-v1",
        "excluded_operational_paths": (".real-stage-execution.lock",),
        "file_count": 5,
        "storage_bytes": 221847,
        "inventory_sha256": (
            "07919624f2bfaeef1c9c54883f089b543f454de4d3775bb73cdf2f7230427596"
        ),
    }
    assert no_go["provider_calls"] == 0
    assert no_go["provider_construction"] is False
    assert no_go["resume_forbidden"] is True
    assert no_go["run_ledger"]["status_counts"] == {"integrity-stopped": 87}
    assert no_go["run_ledger"]["ledger_sha256"] == (
        "ab419bf9db32a9948b3ebac6d1ccd055d6e622e3a28a03ba1aae33f0564b7237"
    )
    assert no_go["budget_ledger"]["current_actual"] == {
        "cost_usd": 0.0,
        "hosted_completions": 0,
        "storage_bytes": 1772,
    }
    assert no_go["stage_receipt"]["failure_error_type"] == (
        "V2118ParentImportIntegrityError"
    )
    assert no_go["stage_receipt"]["failure_cause_type"] == (
        "PilotV2118ContinuationError"
    )
    assert no_go["stage_receipt"]["failure_message"] == (
        "V2.11.5 acceptance revalidation failed: scientific-dispatch acceptance "
        "field 'release' differs from source recomputation"
    )
    assert boundary["parent_budget_debit"] == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": (
            "25d43667520633c5dfa299a693fd4a42736524c2737c2acf6422e2d32f0106c8"
        ),
        "parent_run_ledger_sha256": (
            "ab419bf9db32a9948b3ebac6d1ccd055d6e622e3a28a03ba1aae33f0564b7237"
        ),
        "parent_budget_ledger_sha256": (
            "341f2e448e2162895fc7a58870b629dda3ebaaad9add453d26fe031d430dc339"
        ),
        "stage_bucket": "parent_v2118",
        "cost_usd": 63.1196450625,
        "hosted_completions": 3440,
        "storage_bytes": 270193500,
        "record_sha256": (
            "e5d18a013b0f2cd2faa4bf0d95c62c191a76ce8a0dcdff4a4d684e27956e42cd"
        ),
    }
    budget = boundary["continuation_budget"]
    assert budget["hosted_cap_usd"] == 436.8803549375
    assert budget["fresh_registered_provider_calls"] == 3256
    assert budget["fresh_projected_cost_usd"] == 149.3301875
    assert budget["projected_cumulative_cost_usd"] == 212.4498325625
    assert budget["projected_cumulative_hosted_completions"] == 6696
    assert budget["projected_cumulative_storage_bytes"] == 1_290_193_500
    matrix = boundary["continuation_matrix"]
    assert matrix["combined_registered_denominator"] == 136
    assert matrix["logical_registered_denominator_after_cross_release_dedup"] == 136
    assert matrix["logical_scientific_denominator_after_cross_release_dedup"] == 131
    assert matrix["failed_v2118_rows_are_aborted_release_audit_only"] is True


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("failed_release_no_go", "provider_calls", 1),
        ("parent_budget_debit", "storage_bytes", 270_193_499),
        (
            "continuation_matrix",
            "logical_scientific_denominator_after_cross_release_dedup",
            217,
        ),
        ("immutability", "v2118_resume_forbidden", False),
    ],
)
def test_v2119_strict_boundary_rejects_lineage_or_denominator_drift(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    replacement: object,
) -> None:
    changed = deepcopy(build_contract(ROOT, status="draft"))
    changed["v2119_recovery_boundary"][section][field] = replacement
    _reseal(changed)
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_9_SCIENCE_DESIGN_SHA256",
        science_design_sha256(changed),
    )
    with pytest.raises(PilotContractError, match="recovery boundary drifted"):
        PilotContract.from_dict(changed)


def test_v2119_bootstrap_is_draft_only_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = build_contract(ROOT, status="draft")
    assert first == build_contract(ROOT, status="draft")
    assert canonical_contract_sha256(first) == first["integrity"]["declared_sha256"]
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_9_SCIENCE_DESIGN_SHA256",
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


def test_v2119_frozen_parse_fails_closed_while_pins_are_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = build_contract(ROOT, status="draft")
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_9_SCIENCE_DESIGN_SHA256",
        science_design_sha256(draft),
    )
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_9_CANONICAL_SHA256",
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


def test_v2119_rejects_wrong_denominator_policy_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = deepcopy(build_contract(ROOT, status="draft"))
    changed["denominator_policy"]["policy_id"] = "finevo-pilot-v2.11.8-itt"
    _reseal(changed)
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_9_SCIENCE_DESIGN_SHA256",
        science_design_sha256(changed),
    )
    with pytest.raises(PilotContractError, match="policy identifier drifted"):
        PilotContract.from_dict(changed)


def test_v2119_explicit_frozen_candidate_bootstrap_round_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = build_contract(ROOT, status="draft")
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_9_SCIENCE_DESIGN_SHA256",
        science_design_sha256(draft),
    )
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_V2_11_9_SOURCE_MANIFEST_FILE_SHA256",
        "1" * 64,
    )
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_V2_11_9_SOURCE_MANIFEST_CONTENT_SHA256",
        "2" * 64,
    )
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_9_CANONICAL_SHA256",
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
