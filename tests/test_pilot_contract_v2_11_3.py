from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.render_pilot_v2113_contract import (
    FrozenCandidateBootstrapError,
    _assert_expanded_science_specs_match,
    _assert_v2112_science_delta,
    build_contract,
    main as render_main,
)
import verified_memory.pilot_contract as pilot_contract_module
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256,
    PILOT_CONTRACT_ID_V2_11_3,
    PILOT_CONTRACT_TAG_V2_11_3,
    PILOT_V2_11_3_SOURCE_MANIFEST_CONTENT_SHA256,
    PILOT_V2_11_3_SOURCE_MANIFEST_FILE_SHA256,
    PilotContract,
    PilotContractError,
    canonical_contract_sha256,
    load_pilot_contract,
)
from verified_memory.pilot_v2113_parent_import import (
    V2113_SOURCE_MANIFEST_CONTENT_SHA256,
    V2113_SOURCE_MANIFEST_FILE_SHA256,
    V2113_SOURCE_MANIFEST_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_3.yaml"
PARENT_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_2.yaml"
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


def test_v2113_tracked_frozen_contract_has_exact_denominator_budget_and_lineage() -> (
    None
):
    contract = load_pilot_contract(CONTRACT_PATH)
    specs = contract.expand()

    assert contract.contract_id == PILOT_CONTRACT_ID_V2_11_3
    assert contract.status == "frozen"
    assert PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256 is not None
    assert contract.canonical_hash == PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256
    assert contract.implementation["required_git_tag"] == PILOT_CONTRACT_TAG_V2_11_3
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
    assert len([spec for spec in specs if spec.stage_id in OPERATIONAL_STAGE_IDS]) == 5
    assert (
        len([spec for spec in specs if spec.stage_id not in OPERATIONAL_STAGE_IDS])
        == 131
    )
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
        "parent_v2112": 19.998220562500006,
        "hosted_v2113": 480.0017794375,
        "manual_reserve": 0.0,
    }

    boundary = contract.v2113_forward_boundary
    assert boundary is not None
    assert boundary["source_manifest"] == {
        "path": "experiments/pilot_v2_11_3_source_manifest.json",
        "schema_version": "finevo-pilot-v2.11.3-source-manifest-v1",
        "file_sha256": (
            "f05dbac4951e99476c06883e3c1b792e7ccb459c16eb4d78ac15ddf7905598de"
        ),
        "content_sha256": (
            "5c8e554d1a00803b81deb4f31b4a87ddf54a272861a7c750985cd72b18a95f00"
        ),
    }
    assert boundary["parent_budget_debit"] == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": (
            "c04f7d4c5ae0962a4a64b0ac543d890a1475b6f184f516534eeb8ff026505a37"
        ),
        "parent_run_ledger_sha256": (
            "686d7f528268e0d9d6ac97ae27d483af9c2eb93be53bd329b4fd621c0ec2ae25"
        ),
        "parent_budget_ledger_sha256": (
            "36dd9c62a56c7e87bb647feebeaa7f8d03b0a410d3c7d163834d5029f8da868b"
        ),
        "stage_bucket": "parent_v2112",
        "cost_usd": 19.998220562500006,
        "hosted_completions": 1004,
        "storage_bytes": 221668707,
        "record_sha256": (
            "3ddc22970ff30d1ad9fc3b9efbffe5e4de1f641851bc9e3398aa2fd0977154a1"
        ),
    }
    assert boundary["matrix"]["scientific_cells"] == 131
    assert boundary["matrix"]["provider_backed_scientific_cells"] == 126
    assert boundary["matrix"]["offline_scientific_cells"] == 5
    assert boundary["matrix"]["fresh_scientific_provider_calls"] == 5816


def test_v2113_operational_cells_are_zero_provider_authority_imports() -> None:
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

    boundary = contract.v2113_forward_boundary
    assert boundary is not None
    assert boundary["import_policy"]["provider_calls_during_import"] == 0
    assert boundary["import_policy"]["imported_effect_cells"] == 0
    amendment = contract.v2113_consumer_adapter_amendment
    assert amendment is not None
    assert amendment["preflight_authority_import"]["provider_calls"] == 0
    assert amendment["preflight_authority_import"]["scientific_evidence"] is False


def test_v2113_consumer_adapter_amendment_freezes_failure_and_repair() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    amendment = contract.v2113_consumer_adapter_amendment
    assert amendment is not None

    parent = amendment["parent_terminal_receipt"]
    assert parent["terminal_cells"] == 136
    assert parent["complete_cells"] == 10
    assert parent["failed_cells"] == 126
    assert parent["scientific_provider_calls"] == 0
    assert parent["preflight_provider_calls"] == 64
    assert amendment["root_cause"]["root_cause_id"] == (
        "observed-p95-consumer-schema-dispatch-gap"
    )
    repair = amendment["consumer_adapter_repair"]
    assert repair["registry_adapter_id"] == "v2.11.2-post-gate-authority"
    assert repair["generic_mapping_only_acceptance_for_current_schema"] is False
    assert repair["verification_before_provider_construction"] is True
    assert repair["source_authority"] == {
        "path": (
            "experiment_results/pilot-v2.11.2/raw/long-context-preflight/"
            "post_gate_authority.json"
        ),
        "schema_version": "finevo-pilot-v2.11.2-post-gate-authority-v1",
        "source_commit": "78870956b528946d415a9be5f5769b0893d16d74",
        "file_sha256": (
            "52ade890b123cd030b3d7242aa8347d7dc3a7040fe5f56de0b95938daa029312"
        ),
        "content_sha256": (
            "0d95374c3e2db9fc5bf5c6156fb7bcdf0a9c94e26ed9995f74a2a542a8961aaa"
        ),
    }
    observation = amendment["observation_boundary"]
    assert observation["actor_performance_outcomes_observed"] is False
    assert observation["global_a_to_d_outcome_blind"] is False
    assert observation["inspected_offline_candidate_cells"] == 5
    assert amendment["fresh_science_dispatch"] == {
        "new_raw_namespace_required": "experiment_results/pilot-v2.11.3/raw",
        "registered_scientific_cells": 131,
        "provider_backed_scientific_cells": 126,
        "offline_scientific_cells": 5,
        "registered_provider_calls": 5816,
        "reuse_v2112_scientific_cells": False,
        "reuse_v2112_provider_completions": False,
        "failed_seed_replacement": "forbidden",
        "matrix_shrink": "forbidden",
        "reasoning_or_cap_downgrade": "forbidden",
    }


def test_v2113_scientific_run_specs_match_v2112_without_reuse() -> None:
    parent = load_pilot_contract(PARENT_CONTRACT_PATH)
    child = load_pilot_contract(CONTRACT_PATH)

    _assert_v2112_science_delta(parent.to_dict(), child.to_dict())
    _assert_expanded_science_specs_match(parent, child)
    assert _normalized_science_specs(parent) == _normalized_science_specs(child)
    assert all(spec.contract_id == PILOT_CONTRACT_ID_V2_11_3 for spec in child.expand())
    assert not (
        {spec.run_id for spec in parent.expand()}
        & {spec.run_id for spec in child.expand()}
    )


def test_v2113_draft_renderer_is_deterministic_and_not_paid() -> None:
    tracked = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    rendered_a = build_contract(ROOT, status="draft")
    rendered_b = build_contract(ROOT, status="draft")

    assert rendered_a == rendered_b
    assert rendered_a != tracked
    assert rendered_a["status"] == "draft"
    assert rendered_a["integrity"]["declared_sha256"] == (
        canonical_contract_sha256(rendered_a)
    )
    contract = PilotContract.from_dict(rendered_a)
    with pytest.raises(PilotContractError, match="draft"):
        contract.validate_provenance(
            git_tag=PILOT_CONTRACT_TAG_V2_11_3,
            git_commit="1" * 40,
        )


def test_v2113_pinned_frozen_renderer_reproduces_tracked_contract() -> None:
    tracked = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    expected_ci = tracked["release_requirements"]["expected_ci"]

    rendered = build_contract(
        ROOT,
        status="frozen",
        expected_ci=expected_ci,
    )

    assert rendered == tracked
    assert rendered["integrity"]["declared_sha256"] == (
        PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256
    )


def test_v2113_unpinned_frozen_candidate_is_deterministic_and_self_validating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256",
        None,
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
    assert candidate_a["status"] == "frozen"
    assert candidate_a["release_requirements"]["expected_ci"] == FROZEN_CI_FIXTURE
    assert candidate_a["integrity"]["declared_sha256"] == (
        canonical_contract_sha256(candidate_a)
    )
    assert pilot_contract_module.PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256 is None


def test_v2113_normal_frozen_render_requires_then_reproduces_pinned_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256",
        None,
    )
    candidate = build_contract(
        ROOT,
        status="frozen",
        expected_ci=FROZEN_CI_FIXTURE,
        frozen_candidate=True,
    )
    with pytest.raises(PilotContractError, match="cannot be frozen"):
        build_contract(
            ROOT,
            status="frozen",
            expected_ci=FROZEN_CI_FIXTURE,
        )

    pinned = candidate["integrity"]["declared_sha256"]
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256",
        pinned,
    )
    frozen = build_contract(
        ROOT,
        status="frozen",
        expected_ci=FROZEN_CI_FIXTURE,
    )

    assert frozen == candidate
    assert PilotContract.from_dict(frozen).canonical_hash == pinned


def test_v2113_frozen_candidate_mode_is_explicit_and_unpinned_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        FrozenCandidateBootstrapError,
        match="requires status=frozen",
    ):
        build_contract(ROOT, status="draft", frozen_candidate=True)

    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256",
        "4" * 64,
    )
    with pytest.raises(
        FrozenCandidateBootstrapError,
        match="already pinned",
    ):
        build_contract(
            ROOT,
            status="frozen",
            expected_ci=FROZEN_CI_FIXTURE,
            frozen_candidate=True,
        )


def _frozen_renderer_argv(output: Path, *, candidate: bool) -> list[str]:
    result = [
        "--status",
        "frozen",
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
        str(output),
    ]
    if candidate:
        result.append("--frozen-candidate")
    return result


def test_v2113_candidate_cli_cannot_overwrite_tracked_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256",
        None,
    )
    before = CONTRACT_PATH.read_bytes()

    with pytest.raises(SystemExit, match="must not overwrite the tracked contract"):
        render_main(_frozen_renderer_argv(CONTRACT_PATH, candidate=True))

    assert CONTRACT_PATH.read_bytes() == before


def test_v2113_candidate_cli_writes_only_external_self_validated_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256",
        None,
    )
    output = tmp_path / "pilot_v2_11_3.frozen-candidate.json"
    tracked_before = CONTRACT_PATH.read_bytes()

    assert render_main(_frozen_renderer_argv(output, candidate=True)) == 0

    candidate = json.loads(output.read_text(encoding="utf-8"))
    assert candidate["status"] == "frozen"
    assert candidate["integrity"]["declared_sha256"] == (
        canonical_contract_sha256(candidate)
    )
    assert CONTRACT_PATH.read_bytes() == tracked_before


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["v2113_forward_boundary"]["matrix"].__setitem__(
            "fresh_scientific_provider_calls", 5815
        ),
        lambda value: value["v2113_forward_boundary"][
            "parent_budget_debit"
        ].__setitem__("hosted_completions", 1003),
        lambda value: value["v2113_consumer_adapter_amendment"][
            "consumer_adapter_repair"
        ].__setitem__("verification_before_provider_construction", False),
        lambda value: value["v2113_consumer_adapter_amendment"][
            "observation_boundary"
        ].__setitem__("global_a_to_d_outcome_blind", True),
        lambda value: value["arms"]["closed-loop-preflight"].__setitem__(
            "execution_mode", "closed_loop_preflight"
        ),
        lambda value: value["budgets"]["stage_usd_caps"].__setitem__(
            "hosted_v2113", 480.01
        ),
        lambda value: value["arms"]["full"]["parameters"].__setitem__(
            "context_mode", "prompt_only"
        ),
    ],
)
def test_v2113_rehashed_contract_drift_is_rejected(mutator) -> None:
    value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    mutator(value)

    with pytest.raises(PilotContractError):
        PilotContract.from_dict(_rehash(value))


def test_v2113_delta_guard_rejects_scientific_drift() -> None:
    parent = load_pilot_contract(PARENT_CONTRACT_PATH).to_dict()
    child = load_pilot_contract(CONTRACT_PATH).to_dict()
    child["shocks"]["registered-rate-shock"]["schedule"][1]["interest_rate"] = 0.07

    with pytest.raises(ValueError, match="scientific fields"):
        _assert_v2112_science_delta(parent, child)


def test_v2113_public_constants_are_exported() -> None:
    expected = {
        "PILOT_CONTRACT_ID_V2_11_3",
        "PILOT_CONTRACT_TAG_V2_11_3",
        "PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256",
        "PILOT_CONTRACT_V2_11_3_SCIENCE_DESIGN_SHA256",
        "PILOT_V2_11_3_SOURCE_MANIFEST_FILE_SHA256",
        "PILOT_V2_11_3_SOURCE_MANIFEST_CONTENT_SHA256",
    }
    assert expected <= set(pilot_contract_module.__all__)
    assert PILOT_V2_11_3_SOURCE_MANIFEST_FILE_SHA256 == (
        V2113_SOURCE_MANIFEST_FILE_SHA256
    )
    assert PILOT_V2_11_3_SOURCE_MANIFEST_CONTENT_SHA256 == (
        V2113_SOURCE_MANIFEST_CONTENT_SHA256
    )
    contract = load_pilot_contract(CONTRACT_PATH)
    assert contract.v2113_forward_boundary is not None
    source = contract.v2113_forward_boundary["source_manifest"]
    assert source["schema_version"] == V2113_SOURCE_MANIFEST_SCHEMA_VERSION
