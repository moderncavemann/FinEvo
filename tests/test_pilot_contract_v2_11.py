from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

import scripts.render_pilot_v211_contract as render_v211
from scripts.render_pilot_v211_contract import build_contract
from verified_memory import pilot_contract as contract_module
from verified_memory.pilot_budget import ParentBudgetDebit
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_11,
    PILOT_CONTRACT_TAG_V2_11,
    PilotContract,
    PilotContractError,
    canonical_contract_sha256,
    load_pilot_contract,
)
from verified_memory.pilot_v211_projection import (
    V211_NEW_HOSTED_COMPLETIONS,
    V211_STAGE_CALLS,
    V211_STAGE_LEDGER_CELLS,
    V211_TOTAL_LEDGER_CELLS,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
CONTRACT_PATH = EXPERIMENTS / "pilot_v2_11.yaml"
EXPECTED_CI = {
    "test_count": 1399,
    "test_collection_sha256": (
        "5d39ac1ef9831e97751a2f316f4bf56d1499e880a2536010320be2f67e893b01"
    ),
    "compiled_source_count": 223,
    "compiled_source_inventory_sha256": (
        "b300ff6574932ab1d66f9df1dc3401ae642c58eb40cf4d735c9e54220de43c2c"
    ),
    "sealed_manifest_inventory_sha256": (
        "b5c5a817d09d10752c1f5f00ba556b417d16e06c64b5fcbb15671e49a1d81952"
    ),
}


def _raw_contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _write_rehashed(
    tmp_path: Path,
    value: dict[str, object],
    name: str = "pilot_v2_11.yaml",
) -> Path:
    integrity = value["integrity"]
    assert isinstance(integrity, dict)
    integrity["declared_sha256"] = canonical_contract_sha256(value)
    path = tmp_path / name
    (tmp_path / "pilot_v2_11_source_manifest.json").write_bytes(
        (EXPERIMENTS / "pilot_v2_11_source_manifest.json").read_bytes()
    )
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_v211_is_an_independently_expanded_frozen_contract() -> None:
    raw = _raw_contract()
    contract = load_pilot_contract(CONTRACT_PATH)

    assert contract.contract_id == PILOT_CONTRACT_ID_V2_11
    assert contract.status == "frozen"
    assert contract.to_dict() == raw
    assert canonical_contract_sha256(raw) == raw["integrity"]["declared_sha256"]
    assert contract.canonical_hash == raw["integrity"]["declared_sha256"]
    assert contract.v211_forward_boundary is not None
    assert not any(key.endswith("_amendment") for key in raw)
    assert contract.release_requirements is not None
    assert contract.release_requirements.tag == PILOT_CONTRACT_TAG_V2_11
    assert contract.release_requirements.expected_ci == EXPECTED_CI
    assert contract_module.PILOT_CONTRACT_V2_11_CANONICAL_SHA256 == (
        raw["integrity"]["declared_sha256"]
    )
    binding = contract.validate_provenance(
        "0" * 40,
        PILOT_CONTRACT_TAG_V2_11,
    )
    assert binding["contract_sha256"] == contract.canonical_hash
    assert binding["resolved_git_commit"] == "0" * 40


def test_v211_exact_136_cell_full_scope_matrix() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    specs = contract.expand()
    counts = Counter(spec.stage_id for spec in specs)

    assert len(specs) == V211_TOTAL_LEDGER_CELLS == 136
    assert counts == V211_STAGE_LEDGER_CELLS == {
        "parent-import": 1,
        "capability-gate": 2,
        "long-context-preflight": 2,
        "experiment-c": 25,
        "experiment-a": 20,
        "experiment-d": 55,
        "experiment-b": 25,
        "cross-model": 6,
    }
    assert sum(
        count
        for stage_id, count in counts.items()
        if stage_id
        not in {
            "parent-import",
            "capability-gate",
            "long-context-preflight",
        }
    ) == 131
    d_specs = contract.expand(stage="experiment-d")
    assert Counter(spec.arm_id for spec in d_specs) == {
        "matched-a": 5,
        "matched-b": 5,
        "no-memory": 5,
        "shuffled-episodic": 5,
        "wrong-context": 5,
        "error-verified": 5,
        "error-unverified": 5,
        "narrative-content": 20,
    }
    assert Counter(
        spec.narrative_id
        for spec in d_specs
        if spec.arm_id == "narrative-content"
    ) == {"none": 5, "aligned": 5, "paraphrase": 5, "opposite": 5}


def test_v211_exact_call_caps_profiles_and_cumulative_budget() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    boundary = dict(contract.v211_forward_boundary or {})
    interface = boundary["interface"]
    matrix = boundary["matrix"]
    debit = boundary["parent_budget_debit"]

    assert {
        task_id: task.max_completion_tokens
        for task_id, task in contract.task_output_contracts.items()
    } == {
        "capability-choice": 4096,
        "capability-proposal": 4096,
        "actor-action": 4096,
        "semantic-proposal": 4096,
    }
    assert interface["prompt_token_tier_ceiling"] == 200_000
    assert interface["prompt_token_upper_bound_method"] == (
        "utf8-bytes-plus-256-v1"
    )
    assert interface["prompt_token_ceiling_comparison"] == (
        "reject-upper-bound-greater-than-or-equal-to-ceiling"
    )
    assert matrix["hosted_provider_calls"] == V211_NEW_HOSTED_COMPLETIONS == 5_940
    assert matrix["action_calls"] == 4_944
    assert matrix["semantic_calls"] == 996
    assert matrix["action_calls"] + matrix["semantic_calls"] == 5_940
    assert sum(
        sum(sum(kinds.values()) for kinds in models.values())
        for models in V211_STAGE_CALLS.values()
    ) == 5_940
    assert contract.budgets["total_usd"] == 500.0
    assert contract.budgets["automatic_reserve_usd"] == 0.0
    assert contract.budgets["max_provider_completions"] == 7_500
    assert contract.budgets["max_storage_bytes"] == 5_000_000_000
    assert contract.budgets["stage_usd_caps"] == {
        "parent_v2102": 16.044922812500005,
        "hosted_v211": 483.9550771875,
        "manual_reserve": 0.0,
    }
    assert debit["cost_usd"] == 16.044922812500005
    assert debit["hosted_completions"] == 816
    assert debit["storage_bytes"] == 217_010_835
    assert ParentBudgetDebit(**debit).record_sha256 == (
        "c841dc4cbdfdb548c6917fbb2670c31ba3759f3d4f52ffb0fbb5b9d8bcbbc74d"
    )
    assert set(contract.provider_profiles) == {
        "gpt52_main",
        "gpt56_diagnostic",
        "qref_scripted",
    }
    gpt56 = contract.provider_profiles["gpt56_diagnostic"]
    assert gpt56.requested_model == gpt56.served_model == "gpt-5.6-sol"
    assert gpt56.provider_pin == ("OpenAI-direct",)
    assert gpt56.price_snapshot.dispatch_input == 5.0
    assert gpt56.price_snapshot.dispatch_cached_input == 0.5
    assert gpt56.price_snapshot.dispatch_output == 30.0


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["v211_forward_boundary"]["import_policy"].update(
                {"imported_effect_cells": 1}
            ),
            "forward boundary",
        ),
        (
            lambda value: value["v211_forward_boundary"]["import_policy"].update(
                {"imported_p95_authorities": ["v2.10.2"]}
            ),
            "forward boundary",
        ),
        (
            lambda value: value["task_output_contracts"]["actor-action"].update(
                {"max_completion_tokens": 8192}
            ),
            "contract-specific frozen caps",
        ),
        (
            lambda value: value["budgets"].update(
                {"automatic_reserve_usd": 1.0}
            ),
            "global budget limits",
        ),
        (
            lambda value: value["stages"][5]["cells"][1]["narratives"].pop(),
            "136-cell matrix",
        ),
    ],
)
def test_v211_rejects_rehashed_boundary_and_matrix_mutations(
    tmp_path: Path,
    mutation: object,
    match: str,
) -> None:
    value = _raw_contract()
    assert callable(mutation)
    mutation(value)
    with pytest.raises(PilotContractError, match=match):
        load_pilot_contract(_write_rehashed(tmp_path, value))


def test_v211_cannot_be_declared_frozen_before_release_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _raw_contract()
    monkeypatch.setattr(
        contract_module,
        "PILOT_CONTRACT_V2_11_CANONICAL_SHA256",
        None,
    )
    with pytest.raises(PilotContractError, match="cannot be frozen"):
        load_pilot_contract(_write_rehashed(tmp_path, value))


def test_v211_frozen_contract_rejects_all_null_expected_ci(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _raw_contract()
    expected_ci = value["release_requirements"]["expected_ci"]
    assert isinstance(expected_ci, dict)
    expected_ci.update({field: None for field in expected_ci})
    path = _write_rehashed(tmp_path, value)
    declared = value["integrity"]["declared_sha256"]
    assert isinstance(declared, str)
    monkeypatch.setattr(
        contract_module,
        "PILOT_CONTRACT_V2_11_CANONICAL_SHA256",
        declared,
    )

    with pytest.raises(
        PilotContractError,
        match="frozen expected_ci must be exactly all-concrete",
    ):
        load_pilot_contract(path)


def test_v211_draft_contract_rejects_mixed_expected_ci(
    tmp_path: Path,
) -> None:
    value = _raw_contract()
    value["status"] = "draft"
    expected_ci = value["release_requirements"]["expected_ci"]
    assert isinstance(expected_ci, dict)
    expected_ci.update({field: None for field in expected_ci})
    expected_ci["test_count"] = 1

    with pytest.raises(
        PilotContractError,
        match="draft expected_ci must be exactly all-null",
    ):
        load_pilot_contract(_write_rehashed(tmp_path, value))


def test_v211_source_manifest_file_is_hash_bound(tmp_path: Path) -> None:
    path = _write_rehashed(tmp_path, _raw_contract())
    manifest_path = tmp_path / "pilot_v2_11_source_manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    with pytest.raises(PilotContractError, match="manifest file hash drifted"):
        load_pilot_contract(path)


def test_v211_renderer_is_deterministic_and_self_contained() -> None:
    rendered = build_contract(
        ROOT,
        status="frozen",
        expected_ci=EXPECTED_CI,
    )
    assert rendered == _raw_contract()
    assert rendered["schema_version"] == "finevo-pilot-contract-v2"
    assert rendered["contract_id"] == PILOT_CONTRACT_ID_V2_11
    assert "base_contract" not in rendered


def test_v211_renderer_draft_writes_all_null_expected_ci(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "pilot_v2_11.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_pilot_v211_contract.py",
            "--status",
            "draft",
            "--output",
            str(output),
        ],
    )

    assert render_v211.main() == 0
    rendered = json.loads(output.read_text(encoding="utf-8"))
    assert rendered["status"] == "draft"
    assert all(
        value is None
        for value in rendered["release_requirements"]["expected_ci"].values()
    )


def test_v211_renderer_builds_prospective_frozen_contract_with_concrete_ci() -> None:
    expected_ci = {
        "test_count": 1,
        "test_collection_sha256": "1" * 64,
        "compiled_source_count": 1,
        "compiled_source_inventory_sha256": "2" * 64,
        "sealed_manifest_inventory_sha256": "3" * 64,
    }

    rendered = build_contract(
        ROOT,
        status="frozen",
        expected_ci=expected_ci,
    )

    assert rendered["status"] == "frozen"
    assert rendered["release_requirements"]["expected_ci"] == expected_ci
    assert canonical_contract_sha256(rendered) == (
        rendered["integrity"]["declared_sha256"]
    )


def test_v211_renderer_frozen_cli_writes_all_concrete_expected_ci(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "pilot_v2_11.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_pilot_v211_contract.py",
            "--status",
            "frozen",
            "--test-count",
            "1",
            "--test-collection-sha256",
            "1" * 64,
            "--compiled-source-count",
            "1",
            "--compiled-source-inventory-sha256",
            "2" * 64,
            "--sealed-manifest-inventory-sha256",
            "3" * 64,
            "--output",
            str(output),
        ],
    )

    assert render_v211.main() == 0
    rendered = json.loads(output.read_text(encoding="utf-8"))
    assert rendered["status"] == "frozen"
    assert rendered["release_requirements"]["expected_ci"] == {
        "test_count": 1,
        "test_collection_sha256": "1" * 64,
        "compiled_source_count": 1,
        "compiled_source_inventory_sha256": "2" * 64,
        "sealed_manifest_inventory_sha256": "3" * 64,
    }


@pytest.mark.parametrize(
    "missing_option",
    [
        "--test-count",
        "--test-collection-sha256",
        "--compiled-source-count",
        "--compiled-source-inventory-sha256",
        "--sealed-manifest-inventory-sha256",
    ],
)
def test_v211_renderer_frozen_rejects_each_missing_expected_ci_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_option: str,
) -> None:
    output = tmp_path / "pilot_v2_11.yaml"
    ci_arguments = {
        "--test-count": "1",
        "--test-collection-sha256": "1" * 64,
        "--compiled-source-count": "1",
        "--compiled-source-inventory-sha256": "2" * 64,
        "--sealed-manifest-inventory-sha256": "3" * 64,
    }
    argv = [
        "render_pilot_v211_contract.py",
        "--status",
        "frozen",
        "--output",
        str(output),
    ]
    for option, value in ci_arguments.items():
        if option != missing_option:
            argv.extend((option, value))
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(
        SystemExit,
        match="frozen rendering requires all expected-CI arguments",
    ):
        render_v211.main()
    assert not output.exists()


def test_v211_renderer_draft_rejects_concrete_expected_ci_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "pilot_v2_11.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_pilot_v211_contract.py",
            "--status",
            "draft",
            "--test-count",
            "1",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(
        SystemExit,
        match="draft rendering requires all expected-CI arguments omitted",
    ):
        render_v211.main()
    assert not output.exists()


@pytest.mark.parametrize(
    "name",
    [
        "pilot_v1.yaml",
        "pilot_v2.yaml",
        "pilot_v2_1.yaml",
        "pilot_v2_2.yaml",
        "pilot_v2_3.yaml",
        "pilot_v2_4.yaml",
        "pilot_v2_5.yaml",
        "pilot_v2_6.yaml",
        "pilot_v2_7.yaml",
        "pilot_v2_8.yaml",
        "pilot_v2_9.yaml",
        "pilot_v2_10.yaml",
        "pilot_v2_10_1.yaml",
        "pilot_v2_10_2.yaml",
    ],
)
def test_v211_parser_change_preserves_all_legacy_round_trips(name: str) -> None:
    path = EXPERIMENTS / name
    raw = json.loads(path.read_text(encoding="utf-8"))
    contract = load_pilot_contract(path)

    assert isinstance(contract, PilotContract)
    assert contract.v211_forward_boundary is None
    assert contract.to_dict() == raw
    assert contract.canonical_hash == raw["integrity"]["declared_sha256"]


def test_v211_identity_constants_are_public() -> None:
    assert contract_module.PILOT_CONTRACT_ID_V2_11 == "finevo-pilot-v2.11"
    assert contract_module.PILOT_CONTRACT_TAG_V2_11 == "pilot-v2.11-science"
    assert "PILOT_CONTRACT_ID_V2_11" in contract_module.__all__
    assert "PILOT_CONTRACT_TAG_V2_11" in contract_module.__all__
