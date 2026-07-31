from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.render_pilot_v2114_contract import (
    _assert_expanded_science_specs_match,
    _assert_v2113_science_delta,
    build_contract,
    main as render_main,
)
import verified_memory.pilot_contract as pilot_contract_module
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_11_2,
    PILOT_CONTRACT_ID_V2_11_4,
    PILOT_CONTRACT_TAG_V2_11_4,
    PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256,
    PILOT_CONTRACT_V2_11_4_SCIENCE_DESIGN_SHA256,
    PILOT_V2_11_4_SOURCE_MANIFEST_CONTENT_SHA256,
    PILOT_V2_11_4_SOURCE_MANIFEST_FILE_SHA256,
    PilotContract,
    PilotContractError,
    canonical_contract_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_4.yaml"
PARENT_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_3.yaml"
OPERATIONAL_STAGE_IDS = {
    "parent-import",
    "capability-gate",
    "long-context-preflight",
}

FROZEN_CI_FIXTURE = {
    "test_count": 321,
    "test_collection_sha256": "1" * 64,
    "compiled_source_count": 123,
    "compiled_source_inventory_sha256": "2" * 64,
    "sealed_manifest_inventory_sha256": "3" * 64,
}


def _rehash(value: dict) -> dict:
    candidate = deepcopy(value)
    candidate["integrity"]["declared_sha256"] = "0" * 64
    candidate["integrity"]["declared_sha256"] = canonical_contract_sha256(candidate)
    return candidate


def _normalized_science_specs(contract: PilotContract) -> list[dict]:
    rows = []
    for spec in contract.expand():
        if spec.stage_id in OPERATIONAL_STAGE_IDS:
            continue
        row = spec.to_dict()
        row.pop("run_id")
        row.pop("contract_id")
        row["budget_bucket"] = "normalized-hosted-science"
        rows.append(row)
    return rows


def test_v2114_tracked_frozen_has_exact_denominator_budget_and_no_go_lineage() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    specs = contract.expand()

    assert contract.contract_id == PILOT_CONTRACT_ID_V2_11_4
    assert contract.status == "frozen"
    assert contract.canonical_hash == PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256
    assert contract.implementation["required_git_tag"] == PILOT_CONTRACT_TAG_V2_11_4
    assert contract.release_requirements is not None
    assert all(
        value is not None
        for value in contract.release_requirements.expected_ci.values()
    )
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
    assert (
        len(
            [
                spec
                for spec in specs
                if spec.stage_id not in OPERATIONAL_STAGE_IDS
                and spec.arm_id == "verified-error-candidate"
            ]
        )
        == 5
    )

    assert contract.budgets["total_usd"] == 500.0
    assert contract.budgets["max_provider_completions"] == 7500
    assert contract.budgets["max_storage_bytes"] == 5_000_000_000
    assert contract.budgets["stage_usd_caps"] == {
        "parent_v2113": 19.998220562500006,
        "hosted_v2114": 480.0017794375,
        "manual_reserve": 0.0,
    }

    boundary = contract.v2114_forward_boundary
    assert boundary is not None
    assert boundary["source_manifest"] == {
        "path": "experiments/pilot_v2_11_4_source_manifest.json",
        "schema_version": "finevo-pilot-v2.11.4-source-manifest-v1",
        "file_sha256": (
            "fd37e5f7a6cfa0178fa0baec74fb0d18f058a361586296d50d4bcf611e13839d"
        ),
        "content_sha256": (
            "594b1a00910a1dbecd5e36fcac4397df5341e92b5a9802ce4ca781434b747760"
        ),
    }
    parent = boundary["parent"]
    assert parent["contract_sha256"] == (
        "84c818348fabfdd0ddd0ed503c0a5610faf10098f4973d1748b795e2e65b56f1"
    )
    assert parent["science_commit"] == ("65c613cdc9598dfffecbdf3a375cbf6113246782")
    assert parent["science_tag_object"] == ("87a1911284177b627755faf361ad4ea6c8213958")
    assert parent["run_ledger_internal_sha256"] == (
        "97216e7b0a23b1b78a1e79d3ae166621147fab5582e5259434e1138c39946f40"
    )
    assert parent["budget_ledger_internal_sha256"] == (
        "366495f3cc4b8075e072c47fcf31c3eed40371996f0057efba64a1709ac5850a"
    )
    assert parent["preflight_stage_receipt_content_sha256"] == (
        "1044feb8cf050269c9aafb206bc5fc2c7b6f5b7c0d332d96663d4433ddf967ae"
    )
    assert parent["terminal_status_counts"] == {
        "complete": 3,
        "integrity-stopped": 133,
    }
    assert parent["fresh_provider_calls"] == 0
    assert parent["fresh_cost_usd"] == 0.0


def test_v2114_parent_debit_and_v2112_source_authority_are_separate() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    boundary = contract.v2114_forward_boundary
    assert boundary is not None

    assert boundary["parent_budget_debit"] == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": (
            "84c818348fabfdd0ddd0ed503c0a5610faf10098f4973d1748b795e2e65b56f1"
        ),
        "parent_run_ledger_sha256": (
            "97216e7b0a23b1b78a1e79d3ae166621147fab5582e5259434e1138c39946f40"
        ),
        "parent_budget_ledger_sha256": (
            "366495f3cc4b8075e072c47fcf31c3eed40371996f0057efba64a1709ac5850a"
        ),
        "stage_bucket": "parent_v2113",
        "cost_usd": 19.998220562500006,
        "hosted_completions": 1004,
        "storage_bytes": 221838685,
        "record_sha256": (
            "3f75623b4eb5b6c3c1c2e2a7e97687c215da025cbea309f94e861abee47f90ca"
        ),
    }
    authority = boundary["reservation_authority_source"]
    assert authority["contract_id"] == PILOT_CONTRACT_ID_V2_11_2
    assert authority["schema_version"] == (
        "finevo-pilot-v2.11.2-post-gate-authority-v1"
    )
    assert authority["file_sha256"] == (
        "52ade890b123cd030b3d7242aa8347d7dc3a7040fe5f56de0b95938daa029312"
    )
    assert authority["content_sha256"] == (
        "0d95374c3e2db9fc5bf5c6156fb7bcdf0a9c94e26ed9995f74a2a542a8961aaa"
    )
    assert authority["scientific_evidence"] is False


def test_v2114_operational_cells_remain_zero_provider_authority_imports() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)

    assert {
        spec.execution_mode for spec in contract.expand(stage="capability-gate")
    } == {"capability_authority_import"}
    assert {
        spec.execution_mode for spec in contract.expand(stage="long-context-preflight")
    } == {"preflight_authority_import"}
    preflight_arm = contract.arms["closed-loop-preflight"]
    assert preflight_arm["execution_mode"] == "preflight_authority_import"
    assert preflight_arm["parameters"]["provider_construction"] is False
    assert preflight_arm["parameters"]["provider_calls"] == 0
    assert preflight_arm["parameters"]["fresh_samples"] == 0
    assert preflight_arm["parameters"]["scientific_evidence"] is False

    amendment = contract.v2114_authority_normalization_amendment
    assert amendment is not None
    preflight = amendment["preflight_authority_import"]
    assert preflight["provider_construction"] is False
    assert preflight["provider_calls"] == 0
    assert preflight["scientific_evidence"] is False


def test_v2114_normalization_amendment_is_exact_and_fail_closed() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    amendment = contract.v2114_authority_normalization_amendment
    assert amendment is not None

    parent = amendment["parent_terminal_receipt"]
    assert parent["terminal_cells"] == 136
    assert parent["complete_cells"] == 3
    assert parent["integrity_stopped_cells"] == 133
    assert parent["fresh_provider_calls"] == 0
    assert parent["fresh_cost_usd"] == 0.0

    root_cause = amendment["root_cause"]
    assert root_cause["root_cause_id"] == (
        "resealed-authority-representation-comparison-mismatch"
    )
    assert root_cause["source_wrapper_reservations_sha256"] == (
        "06788d94b9259c24b753d467683ae88c8015f34ce4989fed47ba71e7eeb823da"
    )
    assert root_cause["enriched_runtime_reservations_sha256"] == (
        "8e3e514360cfaff6c40103838d7106078984606d92c779a1f0fb00f93dcb5770"
    )
    assert root_cause["reservation_payloads_drifted"] is False
    assert root_cause["non_allowlisted_authority_fields_drifted"] is False

    repair = amendment["authority_normalization_repair"]
    assert list(repair["reseal_only_authority_fields"]) == [
        "source_authority_receipt_content_sha256",
        "source_authority_receipt_file_sha256",
        "source_authority_receipt_path",
        "source_release_commit",
    ]
    assert repair["normalization_scope"] == ("each-call-kind-authority-object-only")
    assert repair["source_wrapper_mutation"] == "forbidden"
    assert repair["reservation_payload_exact_equality_required"] is True
    assert repair["non_allowlisted_authority_exact_equality_required"] is True
    assert repair["removed_fields_verified_against_source_gate_receipt"] is True
    assert repair["unexpected_extra_field_policy"] == (
        "stop-before-provider-construction"
    )
    assert repair["provider_calls"] == 0


def test_v2114_scientific_run_specs_match_v2113_without_reuse() -> None:
    parent = load_pilot_contract(PARENT_CONTRACT_PATH)
    child = load_pilot_contract(CONTRACT_PATH)

    _assert_v2113_science_delta(parent.to_dict(), child.to_dict())
    _assert_expanded_science_specs_match(parent, child)
    assert _normalized_science_specs(parent) == _normalized_science_specs(child)
    assert all(spec.contract_id == PILOT_CONTRACT_ID_V2_11_4 for spec in child.expand())
    assert not (
        {spec.run_id for spec in parent.expand()}
        & {spec.run_id for spec in child.expand()}
    )
    boundary = child.v2114_forward_boundary
    assert boundary is not None
    assert boundary["matrix"]["scientific_cells"] == 131
    assert boundary["matrix"]["provider_backed_scientific_cells"] == 126
    assert boundary["matrix"]["offline_scientific_cells"] == 5
    assert boundary["matrix"]["fresh_scientific_provider_calls"] == 5816


def test_v2114_draft_renderer_is_deterministic_and_not_paid() -> None:
    tracked = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    rendered_a = build_contract(ROOT, status="draft")
    rendered_b = build_contract(ROOT, status="draft")

    assert rendered_a == rendered_b
    assert rendered_a != tracked
    assert rendered_a["status"] == "draft"
    assert rendered_a["integrity"]["declared_sha256"] == (
        canonical_contract_sha256(rendered_a)
    )
    assert science_design_sha256(rendered_a) == (
        PILOT_CONTRACT_V2_11_4_SCIENCE_DESIGN_SHA256
    )
    contract = PilotContract.from_dict(rendered_a)
    with pytest.raises(PilotContractError, match="draft"):
        contract.validate_provenance(
            git_tag=PILOT_CONTRACT_TAG_V2_11_4,
            git_commit="1" * 40,
        )


def test_v2114_pinned_freeze_reproduces_and_unpinned_candidate_self_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    tracked_ci = tracked["release_requirements"]["expected_ci"]
    assert PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256 == (
        tracked["integrity"]["declared_sha256"]
    )
    assert PILOT_V2_11_4_SOURCE_MANIFEST_FILE_SHA256 == (
        "fd37e5f7a6cfa0178fa0baec74fb0d18f058a361586296d50d4bcf611e13839d"
    )
    assert PILOT_V2_11_4_SOURCE_MANIFEST_CONTENT_SHA256 == (
        "594b1a00910a1dbecd5e36fcac4397df5341e92b5a9802ce4ca781434b747760"
    )

    assert build_contract(
        ROOT,
        status="frozen",
        expected_ci=tracked_ci,
    ) == tracked

    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256",
        None,
    )
    with pytest.raises(PilotContractError, match="cannot be frozen"):
        build_contract(
            ROOT,
            status="frozen",
            expected_ci=FROZEN_CI_FIXTURE,
        )
    candidate_a = build_contract(
        ROOT,
        status="frozen",
        expected_ci=FROZEN_CI_FIXTURE,
        frozen_candidate=True,
    )
    candidate_b = build_contract(
        ROOT,
        status="frozen",
        expected_ci=FROZEN_CI_FIXTURE,
        frozen_candidate=True,
    )
    assert candidate_a == candidate_b
    assert candidate_a["integrity"]["declared_sha256"] == (
        canonical_contract_sha256(candidate_a)
    )
    assert pilot_contract_module.PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256 is None

    pinned = candidate_a["integrity"]["declared_sha256"]
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256",
        pinned,
    )
    assert (
        build_contract(
            ROOT,
            status="frozen",
            expected_ci=FROZEN_CI_FIXTURE,
        )
        == candidate_a
    )


def test_v2114_candidate_cli_cannot_overwrite_tracked_draft() -> None:
    before = CONTRACT_PATH.read_bytes()
    argv = [
        "--status",
        "frozen",
        "--frozen-candidate",
        "--test-count",
        str(FROZEN_CI_FIXTURE["test_count"]),
        "--test-collection-sha256",
        str(FROZEN_CI_FIXTURE["test_collection_sha256"]),
        "--compiled-source-count",
        str(FROZEN_CI_FIXTURE["compiled_source_count"]),
        "--compiled-source-inventory-sha256",
        str(FROZEN_CI_FIXTURE["compiled_source_inventory_sha256"]),
        "--sealed-manifest-inventory-sha256",
        str(FROZEN_CI_FIXTURE["sealed_manifest_inventory_sha256"]),
        "--output",
        str(CONTRACT_PATH),
    ]

    with pytest.raises(SystemExit, match="must not overwrite the tracked contract"):
        render_main(argv)

    assert CONTRACT_PATH.read_bytes() == before


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["v2114_forward_boundary"]["parent"].__setitem__(
            "fresh_provider_calls", 1
        ),
        lambda value: value["v2114_forward_boundary"][
            "parent_budget_debit"
        ].__setitem__("storage_bytes", 221838684),
        lambda value: value["v2114_forward_boundary"][
            "reservation_authority_source"
        ].__setitem__("contract_id", "finevo-pilot-v2.11.3"),
        lambda value: value["v2114_authority_normalization_amendment"][
            "authority_normalization_repair"
        ]["reseal_only_authority_fields"].append("reservation"),
        lambda value: value["v2114_authority_normalization_amendment"][
            "authority_normalization_repair"
        ].__setitem__("source_wrapper_mutation", "allowed"),
        lambda value: value["budgets"]["stage_usd_caps"].__setitem__(
            "hosted_v2114", 480.01
        ),
        lambda value: value["arms"]["full"]["parameters"].__setitem__(
            "context_mode", "prompt_only"
        ),
    ],
)
def test_v2114_rehashed_contract_drift_is_rejected(mutator) -> None:
    value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    mutator(value)

    with pytest.raises(PilotContractError):
        PilotContract.from_dict(_rehash(value))


def test_v2114_delta_guard_rejects_scientific_drift() -> None:
    parent = load_pilot_contract(PARENT_CONTRACT_PATH).to_dict()
    child = load_pilot_contract(CONTRACT_PATH).to_dict()
    child["shocks"]["registered-rate-shock"]["schedule"][1]["interest_rate"] = 0.07

    with pytest.raises(ValueError, match="scientific fields"):
        _assert_v2113_science_delta(parent, child)


def test_v2114_public_constants_export_sealed_source_and_pinned_release() -> None:
    expected = {
        "PILOT_CONTRACT_ID_V2_11_4",
        "PILOT_CONTRACT_TAG_V2_11_4",
        "PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256",
        "PILOT_CONTRACT_V2_11_4_SCIENCE_DESIGN_SHA256",
        "PILOT_V2_11_4_SOURCE_MANIFEST_FILE_SHA256",
        "PILOT_V2_11_4_SOURCE_MANIFEST_CONTENT_SHA256",
    }
    assert expected <= set(pilot_contract_module.__all__)
    assert PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256 == (
        "e898fe49935dae9ae7f0d7ac577dae943192953c1da581d70c334f8c64924e46"
    )
    assert PILOT_V2_11_4_SOURCE_MANIFEST_FILE_SHA256 == (
        "fd37e5f7a6cfa0178fa0baec74fb0d18f058a361586296d50d4bcf611e13839d"
    )
    assert PILOT_V2_11_4_SOURCE_MANIFEST_CONTENT_SHA256 == (
        "594b1a00910a1dbecd5e36fcac4397df5341e92b5a9802ce4ca781434b747760"
    )
    assert PILOT_CONTRACT_V2_11_4_SCIENCE_DESIGN_SHA256 == (
        "8cf696da20de1ee703ff5248a8e081eee0d331f28ab35676264609087b3f3658"
    )
