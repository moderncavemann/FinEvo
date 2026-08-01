from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import verified_memory.pilot_contract as pilot_contract_module
from scripts.render_pilot_v2116_contract import (
    FrozenCandidateBootstrapError,
    _assert_expanded_continuation_specs_match,
    _assert_v2115_continuation_delta,
    _normalized_continuation_specs,
    _parse_with_bootstrap_design_pin,
    build_contract,
)
from verified_memory.pilot_budget import ParentBudgetDebit
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_11_6,
    PILOT_CONTRACT_TAG_V2_11_6,
    PilotContract,
    PilotContractError,
    canonical_contract_sha256,
    canonical_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_6.yaml"
PARENT_PATH = ROOT / "experiments" / "pilot_v2_11_5.yaml"


def _parse_draft(
    value: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> PilotContract:
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_6_SCIENCE_DESIGN_SHA256",
        science_design_sha256(value),
    )
    return PilotContract.from_dict(value)


def _reseal(value: dict[str, object]) -> None:
    integrity = value["integrity"]
    assert isinstance(integrity, dict)
    integrity["declared_sha256"] = canonical_contract_sha256(value)


def test_v2116_draft_round_trip_and_exact_87_cell_continuation(
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
    assert contract.contract_id == PILOT_CONTRACT_ID_V2_11_6
    assert contract.status == "draft"
    assert contract.implementation["required_git_tag"] == PILOT_CONTRACT_TAG_V2_11_6
    assert contract.stage_ids == (
        "parent-import",
        "experiment-d",
        "experiment-b",
        "cross-model",
    )
    assert {stage.stage_id: stage.prerequisites for stage in contract.stages} == {
        "parent-import": (),
        "experiment-d": ("parent-import",),
        "experiment-b": ("experiment-d",),
        "cross-model": ("experiment-b",),
    }
    assert {stage.stage_id: stage.budget_bucket for stage in contract.stages} == {
        "parent-import": "parent_v2115",
        "experiment-d": "hosted_v2116",
        "experiment-b": "hosted_v2116",
        "cross-model": "hosted_v2116",
    }
    specs = contract.expand()
    assert len(specs) == 87
    assert len({spec.run_id for spec in specs}) == 87
    assert {
        stage_id: len(contract.expand(stage=stage_id))
        for stage_id in contract.stage_ids
    } == {
        "parent-import": 1,
        "experiment-d": 55,
        "experiment-b": 25,
        "cross-model": 6,
    }
    assert {spec.execution_mode for spec in contract.expand(stage="experiment-d")} == {
        "checkpoint_continuation"
    }
    assert {spec.execution_mode for spec in contract.expand(stage="experiment-b")} == {
        "actor_run"
    }
    assert {spec.execution_mode for spec in contract.expand(stage="cross-model")} == {
        "actor_run"
    }
    assert load_pilot_contract(CONTRACT_PATH).to_dict() == tracked


def test_v2116_maps_all_and_only_86_parent_scheduled_science_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = load_pilot_contract(PARENT_PATH)
    rendered = build_contract(ROOT, status="draft")
    child = _parse_draft(rendered, monkeypatch)
    _assert_v2115_continuation_delta(parent.to_dict(), rendered)
    _assert_expanded_continuation_specs_match(parent, child)
    parent_rows = _normalized_continuation_specs(parent)
    child_rows = _normalized_continuation_specs(child)
    assert parent_rows == child_rows
    assert len(parent_rows) == 86
    assert canonical_sha256(parent_rows) == (
        "9968bb55b9c56ced90f56826bc8e186f72299e0a8bb40dfdb4fbb1e637af1632"
    )
    assert {
        stage_id: len([row for row in child_rows if row["stage_id"] == stage_id])
        for stage_id in ("experiment-d", "experiment-b", "cross-model")
    } == {"experiment-d": 55, "experiment-b": 25, "cross-model": 6}


def test_v2116_boundary_binds_parent_prefix_receipts_inventory_and_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _parse_draft(build_contract(ROOT, status="draft"), monkeypatch)
    boundary = dict(contract.v2116_continuation_boundary or {})
    parent = boundary["parent_release"]
    assert parent == {
        "contract_id": "finevo-pilot-v2.11.5",
        "contract_path": "experiments/pilot_v2_11_5.yaml",
        "contract_file_sha256": (
            "b96438430231f0c46fd6c5f15ba749713534feb15f964c496aa02606cf11103b"
        ),
        "contract_sha256": (
            "e1ecdec43e3f7a7b9a3d0977e2522d95861e826fc68781377d7eaceeb5e6e2ef"
        ),
        "science_tag": "pilot-v2.11.5-science",
        "science_tag_object": "bccfb13cee7d592470d1873cfacc3b12bed38be4",
        "science_commit": "2351ac2283f9fedb9dce70067174020be56ed9cc",
        "source_manifest_path": "experiments/pilot_v2_11_5_source_manifest.json",
        "source_manifest_file_sha256": (
            "fea5a276fb64fdd5bf0539014687ea39a891e9d305205b1d2046a2c15a892d16"
        ),
        "source_manifest_content_sha256": (
            "be84d33f561a5ab8927f13e0753f5109b5f018dc790ae180d5e0e6e0228af559"
        ),
    }
    prefix = boundary["parent_terminal_prefix"]
    assert prefix["run_ledger"]["prefix_event_count"] == 53
    assert prefix["run_ledger"]["ledger_sha256"] == (
        "8a86231f0906ea117626190cc7a2699933c968ce555612cb1bc6378473601fa7"
    )
    assert prefix["run_ledger"]["status_counts"] == {
        "complete": 47,
        "failed": 3,
        "scheduled": 86,
    }
    assert prefix["budget_ledger"]["prefix_event_count"] == 103
    assert prefix["budget_ledger"]["ledger_sha256"] == (
        "53e70f6c0b9053674408de385e1a5b5bf42ace7e82dc8e0c6f227ea124b7a38f"
    )
    inventory = prefix["source_raw_inventory"]
    assert tuple(inventory["excluded_operational_paths"]) == (
        ".real-stage-execution.lock",
    )
    assert (inventory["file_count"], inventory["storage_bytes"]) == (
        691,
        48_820_556,
    )
    assert inventory["inventory_sha256"] == (
        "f2fdb1ccedcb70e6793d3b8f3c87425f0d602552f0a3e0e7f35db9c5777c6746"
    )
    experiment_a = boundary["parent_stage_receipts"]["experiment-a"]
    assert experiment_a["file_sha256"] == (
        "8193f3449663f63c9cf0c881ee5e7759d2682f320f214c4941040489c81734f9"
    )
    assert experiment_a["content_sha256"] == (
        "177dc8ce4d1957eac0734bb1716279676f77931e30b3a1d10dd2c138a43a5457"
    )
    assert experiment_a["status_counts"] == {
        "complete": 17,
        "failed": 3,
    }
    experiment_c = boundary["parent_stage_receipts"]["experiment-c"]
    assert experiment_c["file_sha256"] == (
        "958cb161785c144c89861da3e9536e53069e8f1070a64c03f54647cbfe05b322"
    )
    assert experiment_c["content_sha256"] == (
        "39a9d35f4961fee4b0bc59ac67f7a9a2da0c3f95fddf77a418b92e518b6e2eba"
    )
    assert experiment_c["status_counts"] == {"complete": 25}
    debit = ParentBudgetDebit.from_dict(boundary["parent_budget_debit"])
    assert debit.cost_usd == 63.1196450625
    assert debit.hosted_completions == 3440
    assert debit.storage_bytes == 270_188_235
    assert debit.record_sha256 == (
        "bada157f174d33344370c621f0bd480d57cf8ff5adcde498d7e02426a4363270"
    )
    budget = boundary["continuation_budget"]
    assert budget["fresh_projected_cost_usd"] == 149.3301875
    assert budget["fresh_registered_provider_calls"] == 3256
    assert budget["fresh_storage_reservation_bytes"] == 1_020_000_000
    assert budget["projected_cumulative_cost_usd"] == 212.4498325625
    assert budget["projected_cumulative_hosted_completions"] == 6696
    assert budget["projected_cumulative_storage_bytes"] == 1_290_188_235
    assert budget["within_all_hard_caps"] is True


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("parent_release", "science_tag_object", "0" * 40),
        ("parent_budget_debit", "hosted_completions", 3439),
        ("continuation_matrix", "combined_registered_denominator", 87),
        ("continuation_matrix", "source_scheduled_rows", 85),
        ("immutability", "failed_seed_replacement", "allowed"),
    ],
)
def test_v2116_strict_boundary_rejects_parent_or_denominator_drift(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    replacement: object,
) -> None:
    changed = deepcopy(build_contract(ROOT, status="draft"))
    changed["v2116_continuation_boundary"][section][field] = replacement
    _reseal(changed)
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_6_SCIENCE_DESIGN_SHA256",
        science_design_sha256(changed),
    )
    with pytest.raises(PilotContractError, match="continuation boundary drifted"):
        PilotContract.from_dict(changed)


def test_v2116_parent_import_declares_continuation_not_row_reclassification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered = build_contract(ROOT, status="draft")
    parameters = rendered["arms"]["parent-import"]["parameters"]
    assert parameters == {
        "parent_artifacts_read_only": True,
        "parent_denominator_continued": True,
        "terminal_parent_rows_imported_as_child_rows": False,
        "mapped_scheduled_parent_rows": 86,
        "provider_calls": 0,
        "scientific_evidence": False,
    }
    changed = deepcopy(rendered)
    changed["arms"]["parent-import"]["parameters"][
        "terminal_parent_rows_imported_as_child_rows"
    ] = True
    _reseal(changed)
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_6_SCIENCE_DESIGN_SHA256",
        science_design_sha256(changed),
    )
    with pytest.raises(PilotContractError, match="parent-import arm drifted"):
        PilotContract.from_dict(changed)


def test_v2116_bootstrap_is_draft_only_and_renderer_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = build_contract(ROOT, status="draft")
    second = build_contract(ROOT, status="draft")
    assert first == second
    assert canonical_contract_sha256(first) == first["integrity"]["declared_sha256"]
    design = science_design_sha256(first)
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_6_SCIENCE_DESIGN_SHA256",
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
    assert design == "af846dbd5697c2dfbd09b860162f6bcbec929e8d7a5ad5a09350bc45a091ca87"


def test_v2116_explicit_frozen_candidate_bootstrap_round_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = build_contract(ROOT, status="draft")
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_6_SCIENCE_DESIGN_SHA256",
        science_design_sha256(draft),
    )
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_V2_11_6_SOURCE_MANIFEST_FILE_SHA256",
        "1" * 64,
    )
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_V2_11_6_SOURCE_MANIFEST_CONTENT_SHA256",
        "2" * 64,
    )
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_6_CANONICAL_SHA256",
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
