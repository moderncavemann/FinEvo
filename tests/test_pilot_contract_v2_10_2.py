from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import pytest

import scripts.render_pilot_v2102_contract as render_v2102
from verified_memory import pilot_contract as contract_module
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_10_1,
    PILOT_CONTRACT_ID_V2_10_2,
    PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10_2,
    PILOT_CONTRACT_TAG_V2_10_2,
    PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256,
    PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256,
    PilotContractError,
    _v2_10_2_expected_p95_consumer_adapter_retry_amendment,
    canonical_contract_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
V2101_PATH = EXPERIMENTS / "pilot_v2_10_1.yaml"
OVERLAY_PATH = EXPERIMENTS / "pilot_v2_10_2_overlay.yaml"
EXPANDED_PATH = EXPERIMENTS / "pilot_v2_10_2.yaml"
DEPENDENCIES = tuple(EXPERIMENTS.glob("pilot_v2_*manifest.json")) + (
    EXPERIMENTS / "pilot_v2_10.yaml",
    V2101_PATH,
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
    path = target / "pilot_v2_10_2_overlay.yaml"
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


def test_v2102_draft_preserves_the_exact_v2101_science_design() -> None:
    parent = load_pilot_contract(V2101_PATH)
    overlay = load_pilot_contract(OVERLAY_PATH)
    expanded = load_pilot_contract(EXPANDED_PATH)

    assert overlay.to_dict() == expanded.to_dict()
    assert overlay.contract_id == PILOT_CONTRACT_ID_V2_10_2
    assert overlay.status == "frozen"
    assert overlay.implementation["required_git_tag"] == (
        PILOT_CONTRACT_TAG_V2_10_2
    )
    assert overlay.release_requirements is not None
    assert overlay.release_requirements.tag == PILOT_CONTRACT_TAG_V2_10_2
    assert overlay.denominator_policy is not None
    assert overlay.denominator_policy.policy_id == "finevo-pilot-v2.10.2-itt"
    assert overlay.release_requirements.expected_ci == {
        "test_count": 1255,
        "test_collection_sha256": (
            "9bafe8a4f96f891c80f692412b81f923a5c574c72c64f85e976f1197f71a7e40"
        ),
        "compiled_source_count": 209,
        "compiled_source_inventory_sha256": (
            "ce7a1ac162e73fb9954892c5f2865795c55beeb794a15f3d4f90f6092936390b"
        ),
        "sealed_manifest_inventory_sha256": (
            "b5c5a817d09d10752c1f5f00ba556b417d16e06c64b5fcbb15671e49a1d81952"
        ),
    }
    assert len(parent.expand()) == len(overlay.expand()) == 211
    assert _normalized_specs(V2101_PATH) == _normalized_specs(OVERLAY_PATH)
    assert science_design_sha256(overlay.to_dict()) == (
        PILOT_CONTRACT_V2_4_SCIENCE_DESIGN_SHA256
    )
    assert overlay.budgets == parent.budgets


def test_v2102_amendment_freezes_failure_observation_and_fresh_retry() -> None:
    amendment = _v2_10_2_expected_p95_consumer_adapter_retry_amendment(
        status="frozen"
    )
    assert amendment["source_manifest"] == {
        "path": "experiments/pilot_v2_10_2_source_manifest.json",
        "schema_version": "finevo-pilot-v2.10.2-source-manifest-v1",
        "file_sha256": (
            "f1d953f5b39ab1032ffeb37b73db7c80d54296fba046eddf7e2485e4dc1cc2bd"
        ),
        "content_sha256": (
            "cafbc2cef89c3d605b7242327b9e7aa418ef26ce14eb37c3c89cf2996600f130"
        ),
    }
    parent = amendment["parent_terminal_failure"]
    assert parent["contract_id"] == PILOT_CONTRACT_ID_V2_10_1
    assert parent["contract_sha256"] == (
        PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256
    )
    assert parent["status_counts"] == {"complete": 26, "failed": 185}
    assert parent["fresh_provider_calls"] == 0
    assert parent["offline_candidate_metrics_inspected"] is True
    assert parent["actor_performance_treatment_outcome_blind"] is True
    assert parent["global_a_d_outcome_blind"] is False

    fresh = amendment["fresh_science_dispatch"]
    assert fresh["registered_cells"] == 211
    assert fresh["a_d_cells"] == 195
    assert fresh["provider_backed_a_d_cells"] == 185
    assert fresh["offline_candidate_admission_cells"] == 10
    assert fresh["offline_candidate_stage_counts"] == {
        "experiment-c": 5,
        "local-experiment-c": 5,
    }
    assert fresh["fresh_provider_dispatch_for_provider_backed_cells"] == "required"
    assert fresh["offline_candidate_provider_dispatch"] == "forbidden"
    assert fresh["v2_10_1_a_d_cell_reuse"] == "forbidden"
    assert fresh["v2_10_1_offline_candidate_cell_reuse"] == "forbidden"

    budget = amendment["budget_carry_forward"]
    assert budget["total_cap_usd"] == 500.0
    assert budget["cumulative_prior"] == {
        "cost_usd": 3.212770875,
        "hosted_completions": 184,
        "storage_bytes": 92_541_342,
        "parent_contract_sha256": (
            PILOT_CONTRACT_V2_10_1_CANONICAL_SHA256
        ),
        "parent_run_ledger_sha256": (
            "75e91445745ec5480577327053a8d7eaefc4352cb6f3f176693460cc712d22b6"
        ),
        "parent_budget_ledger_sha256": (
            "87d313e4f96766f3137c5c0175b0adb6e8a24d4c7697e556e2e0e46f00525161"
        ),
        "stage_bucket": "parent_v23",
    }
    assert budget["manual_reserve_automatic_use"] is False


def test_v2102_rejects_parent_and_retry_tampering(tmp_path: Path) -> None:
    parent = _draft_overlay()
    parent["base_contract"]["canonical_sha256"] = "1" * 64
    with pytest.raises(PilotContractError, match="base contract binding"):
        load_pilot_contract(_write_overlay(tmp_path / "parent", parent))

    adapter = _draft_overlay()
    adapter["p95_consumer_adapter_retry_amendment"][
        "consumer_adapter_repair"
    ]["mapping_only_current_release_input"] = "accept"
    with pytest.raises(PilotContractError, match="consumer-adapter retry"):
        load_pilot_contract(_write_overlay(tmp_path / "adapter", adapter))

    reuse = _draft_overlay()
    reuse["p95_consumer_adapter_retry_amendment"][
        "fresh_science_dispatch"
    ]["v2_10_1_offline_candidate_cell_reuse"] = "allowed"
    with pytest.raises(PilotContractError, match="consumer-adapter retry"):
        load_pilot_contract(_write_overlay(tmp_path / "reuse", reuse))


def test_v2102_renderer_round_trips_a_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _copy_dependencies(tmp_path)
    monkeypatch.setattr(render_v2102, "EXPERIMENTS", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["render_pilot_v2102_contract.py", "--status", "draft"],
    )
    assert render_v2102.main() == 0
    overlay = load_pilot_contract(tmp_path / "pilot_v2_10_2_overlay.yaml")
    expanded = load_pilot_contract(tmp_path / "pilot_v2_10_2.yaml")
    assert overlay.to_dict() == expanded.to_dict()
    assert len(overlay.expand()) == 211

    original_overlay = (
        tmp_path / "pilot_v2_10_2_overlay.yaml"
    ).read_bytes()
    original_expanded = (tmp_path / "pilot_v2_10_2.yaml").read_bytes()

    def fail_validation(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("injected staged validation failure")

    monkeypatch.setattr(render_v2102, "load_pilot_contract", fail_validation)
    with pytest.raises(RuntimeError, match="injected staged validation failure"):
        render_v2102.main()
    assert (
        tmp_path / "pilot_v2_10_2_overlay.yaml"
    ).read_bytes() == original_overlay
    assert (tmp_path / "pilot_v2_10_2.yaml").read_bytes() == original_expanded


def test_v2102_module_exports_identity_constants() -> None:
    assert contract_module.PILOT_CONTRACT_ID_V2_10_2 == (
        "finevo-pilot-v2.10.2"
    )
    assert contract_module.PILOT_CONTRACT_TAG_V2_10_2 == (
        "pilot-v2.10.2-science"
    )
    assert contract_module.PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_10_2 == (
        "finevo-pilot-contract-v2.10.2-p95-consumer-adapter-retry-overlay-v1"
    )
    assert "PILOT_CONTRACT_ID_V2_10_2" in contract_module.__all__
