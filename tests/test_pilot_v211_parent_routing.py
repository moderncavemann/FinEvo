from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_pilot
from verified_memory.m0_utility import UtilityConfig
from verified_memory.pilot_budget import ParentBudgetDebit, PilotBudgetLedger
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory import pilot_orchestrator as orchestrator


ROOT = Path(__file__).resolve().parents[1]
V211_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11.yaml"
V2102_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_10_2.yaml"


def _v211_receipt(contract) -> dict:
    boundary = contract.to_dict()["v211_forward_boundary"]
    calibration = boundary["calibration_allowlist"]
    profile = {
        "profile_id": calibration["utility_profile_id"],
        "rho": calibration["utility_profile"]["rho"],
        "labor_weight": calibration["utility_profile"]["labor_weight"],
        "inverse_frisch": calibration["utility_profile"]["inverse_frisch"],
        "consumption_scale": calibration["q_ref"],
        "discount_factor": calibration["utility_profile"]["discount_factor"],
        "budget_tolerance": 1e-8,
        "max_labor_hours": 168.0,
    }
    threshold = {
        "value": calibration["absolute_flow_utility_threshold"],
        "source_profile": calibration["utility_profile_id"],
        "treatment_outcomes_inspected": False,
        "row_count": 96,
        "source_seeds": [1942013315, 760687867],
        "source_manifests": [],
    }
    return {
        "schema_version": "finevo-pilot-v2.11-parent-import-v1",
        "source_manifest": {
            "path": boundary["source_manifest"]["path"],
            "file_sha256": boundary["source_manifest"]["file_sha256"],
            "content_sha256": boundary["source_manifest"]["content_sha256"],
        },
        "parent_release": {
            "contract_id": boundary["parent"]["contract_id"],
            "contract_sha256": boundary["parent"]["contract_sha256"],
            "resolved_git_commit": boundary["parent"]["science_commit"],
            "science_tag": boundary["parent"]["science_tag"],
            "run_ledger_sha256": boundary["parent"][
                "run_ledger_internal_sha256"
            ],
            "budget_ledger_sha256": boundary["parent"][
                "budget_ledger_internal_sha256"
            ],
        },
        "imported_prerequisites": {
            "q_ref": calibration["q_ref"],
            "selected_utility_profile": profile,
            "stage0_absolute_flow_utility_threshold": threshold,
            "source_bindings": {
                "q_ref_content_sha256": boundary["parent"][
                    "q_ref_content_sha256"
                ],
                "stage0_selection_content_sha256": boundary["parent"][
                    "stage0_selection_content_sha256"
                ],
            },
        },
        "cumulative_budget_debit": boundary["parent_budget_debit"],
        "import_policy": {
            "provider_construction": False,
            "provider_calls": 0,
            "imported_effect_cells": 0,
            "effect_metrics_observed": False,
            "effect_artifact_paths": [],
            "imported_p95_authorities": [],
            "raw_tree_copied": False,
            "copied_file_count": 0,
            "copied_byte_count": 0,
        },
        "scientific_evidence": False,
        "integrity": {
            "canonicalization": "json-sort-keys-utf8-v1",
            "content_sha256": "1" * 64,
        },
    }


def _zero_result(path: Path) -> dict:
    return {
        "receipt": str(path),
        "receipt_file_sha256": "2" * 64,
        "receipt_content_sha256": "1" * 64,
        "provider_construction": False,
        "provider_calls": 0,
        "imported_effect_cells": 0,
        "effect_metrics_observed": False,
        "effect_artifact_paths": [],
        "imported_p95_authorities": [],
        "raw_tree_copied": False,
        "copied_file_count": 0,
        "copied_byte_count": 0,
    }


def test_parent_predicates_keep_v211_fresh_and_v2102_compatible() -> None:
    v211 = load_pilot_contract(V211_CONTRACT_PATH)
    v2102 = load_pilot_contract(V2102_CONTRACT_PATH)

    assert len(v211.expand()) == 136
    assert v211.contract_id in orchestrator.PARENT_IMPORT_CONTRACT_IDS
    assert v211.contract_id not in (
        orchestrator.INHERITED_P95_PARENT_CONTRACT_IDS
    )
    assert v211.contract_id not in orchestrator.LOCAL_FIRST_PARENT_CONTRACT_IDS
    assert orchestrator._materializes_legacy_amendment_controls(v211) is False
    assert orchestrator._cross_model_science_stage_ids(v211) == (
        "cross-model",
    )

    assert v2102.contract_id in orchestrator.PARENT_IMPORT_CONTRACT_IDS
    assert v2102.contract_id in (
        orchestrator.INHERITED_P95_PARENT_CONTRACT_IDS
    )
    assert v2102.contract_id in orchestrator.LOCAL_FIRST_PARENT_CONTRACT_IDS
    assert orchestrator._materializes_legacy_amendment_controls(v2102) is False
    assert orchestrator._cross_model_science_stage_ids(v2102) == ()
    debit = orchestrator._parent_budget_debit(v2102)
    assert debit is not None
    assert debit.to_dict() == (
        orchestrator.parent_budget_debit_for_v2102(v2102).to_dict()
    )


def test_v211_utility_and_threshold_are_consumed_from_parent_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(V211_CONTRACT_PATH)
    receipt = _v211_receipt(contract)
    receipt_path = (
        tmp_path / "parent-import" / "parent_import_receipt.json"
    )
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        orchestrator,
        "verify_v211_parent_import_receipt",
        lambda *_args, **_kwargs: receipt,
    )

    calibration = orchestrator._load_verified_v211_calibration(
        contract,
        raw_root=tmp_path,
    )
    assert calibration["q_ref"] == 63.50397933257746
    assert calibration["selected_profile_id"] == "nu-0.5"
    assert calibration["absolute_flow_utility_threshold"]["value"] == (
        0.05617208967516696
    )
    selection = orchestrator._load_verified_stage0_selection(
        contract,
        raw_root=tmp_path,
        paid=None,
    )
    assert selection["selected_profile_id"] == "nu-0.5"
    assert selection["outcome_fields_used"] == []
    q_ref = orchestrator._load_verified_q_ref(
        contract,
        raw_root=tmp_path,
        paid=None,
    )
    assert q_ref["q_ref"] == 63.50397933257746

    spec = contract.expand(stage="experiment-a")[0]
    resolved = orchestrator.resolve_utility(
        contract,
        spec,
        raw_root=tmp_path,
    )
    assert resolved == UtilityConfig(
        rho=1.0,
        labor_weight=2.0,
        inverse_frisch=0.5,
        consumption_scale=63.50397933257746,
        discount_factor=0.99,
        max_labor_hours=168.0,
        budget_tolerance=1e-8,
    )


def test_v211_dedicated_parent_executor_is_zero_provider_and_keeps_136_itt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(V211_CONTRACT_PATH)
    raw_root = tmp_path / "experiment_results" / "pilot-v2.11" / "raw"
    raw_root.mkdir(parents=True)
    run_ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    run_ledger.register(contract.expand())
    debit = ParentBudgetDebit.from_dict(
        contract.to_dict()["v211_forward_boundary"]["parent_budget_debit"]
    )
    budget_ledger = PilotBudgetLedger(
        raw_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=debit,
    )
    receipt = _v211_receipt(contract)
    calls = {"build": 0, "persist": 0, "verify": 0, "provider": 0}

    def build(**_kwargs):
        calls["build"] += 1
        return receipt

    def persist(*, destination, **_kwargs):
        calls["persist"] += 1
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return _zero_result(path)

    def verify(*_args, **_kwargs):
        calls["verify"] += 1
        return receipt

    def fail_provider(*_args, **_kwargs):
        calls["provider"] += 1
        raise AssertionError("provider construction is forbidden")

    def write_terminal(path, *, payload, **_kwargs):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"payload": payload}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    calibration = {
        "receipt": receipt,
        "q_ref": 63.50397933257746,
        "selected_profile_id": "nu-0.5",
        "selected_utility": {
            "rho": 1.0,
            "labor_weight": 2.0,
            "inverse_frisch": 0.5,
            "consumption_scale": 63.50397933257746,
            "discount_factor": 0.99,
            "budget_tolerance": 1e-8,
            "max_labor_hours": 168.0,
        },
        "absolute_flow_utility_threshold": {
            "value": 0.05617208967516696,
        },
    }
    monkeypatch.setattr(orchestrator, "build_v211_parent_import", build)
    monkeypatch.setattr(orchestrator, "persist_v211_parent_import", persist)
    monkeypatch.setattr(
        orchestrator,
        "verify_v211_parent_import_receipt",
        verify,
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_verified_v211_calibration",
        lambda *_args, **_kwargs: calibration,
    )
    monkeypatch.setattr(orchestrator, "_provider_for_profile", fail_provider)
    monkeypatch.setattr(orchestrator, "write_terminal_summary", write_terminal)

    paid = orchestrator.GitProvenance(
        git_tag=contract.implementation["required_git_tag"],
        head_commit="a" * 40,
        tag_commit="a" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )
    result = orchestrator._execute_v211_parent_import_stage(
        contract,
        contract.expand(stage="parent-import"),
        raw_root=raw_root,
        repo_root=tmp_path,
        parent_repo_root=tmp_path / "parent-science",
        paid=paid,
        run_ledger=run_ledger,
        budget_ledger=budget_ledger,
    )

    assert result["status"] == "complete"
    assert calls == {"build": 1, "persist": 1, "verify": 1, "provider": 0}
    rows = run_ledger.snapshot()["runs"]
    assert len(rows) == 136
    assert list(row["status"] for row in rows.values()).count("complete") == 1
    assert list(row["status"] for row in rows.values()).count("scheduled") == 135
    totals = budget_ledger.snapshot()["committed"]
    assert totals["cost_usd"] == 16.044922812500005
    assert totals["completions"] == 816


def test_cli_routes_v211_parent_root_and_rejects_missing_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = run_pilot.build_parser().parse_args(
        [
            "--contract",
            str(V211_CONTRACT_PATH),
            "--stage",
            "parent-import",
        ]
    )
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="V2.11 parent-import requires --parent-repo-root",
    ):
        run_pilot.execute(missing)

    parent_root = tmp_path / "finevo-pilot-v2-10-2-science"
    args = run_pilot.build_parser().parse_args(
        [
            "--contract",
            str(V211_CONTRACT_PATH),
            "--stage",
            "parent-import",
            "--parent-repo-root",
            str(parent_root),
            "--resume",
        ]
    )
    observed = {}

    def execute_stage(**kwargs):
        observed.update(kwargs)
        return {"status": "complete", "provider_calls": 0}

    monkeypatch.setattr(run_pilot, "execute_stage", execute_stage)
    assert run_pilot.execute(args)["provider_calls"] == 0
    assert observed["parent_repo_root"] == parent_root
    assert observed["stage_id"] == "parent-import"
    assert "V2.11" in run_pilot.build_parser().format_help()
