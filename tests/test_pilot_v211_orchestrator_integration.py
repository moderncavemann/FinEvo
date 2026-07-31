from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import test_pilot_v211_gate as gate_fixtures
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.pilot_budget import PilotBudgetLedger, RunProjection
from verified_memory.pilot_contract import load_pilot_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11.yaml"


def _paid() -> orchestrator.GitProvenance:
    return orchestrator.GitProvenance(
        git_tag="pilot-v2.11-science",
        head_commit="2" * 40,
        tag_commit="2" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )


def test_v211_development_fake_matrix_covers_all_25_diagnostic_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("V2.11 fake matrix attempted a live provider")

    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden)
    monkeypatch.setattr(
        orchestrator,
        "parent_budget_debit_for_v211",
        forbidden,
    )

    result = orchestrator.run_development_fake_matrix(
        contract_path=CONTRACT_PATH,
        resume=False,
        raw_root=tmp_path,
    )

    assert result["status"] == "pass"
    assert result["registered_cells"] == 25
    assert result["status_counts"] == {"complete": 25}
    assert result["diagnostic_only"] is True
    assert result["scientific_evidence"] is False


def _write_gate_sources(
    *,
    raw_root: Path,
    contract,
    run_ledger: orchestrator.PilotRunLedger,
    gpt56_capability_pass: bool,
) -> None:
    for model_id, capability_pass in (
        ("gpt52_main", True),
        ("gpt56_diagnostic", gpt56_capability_pass),
    ):
        capability_spec = contract.expand(
            stage="capability-gate",
            model=model_id,
        )[0]
        capability_dir = (
            raw_root
            / capability_spec.stage_id
            / "runs"
            / capability_spec.run_id
        )
        capability_dir.mkdir(parents=True)
        orchestrator._atomic_json(
            capability_dir / "capability.json",
            {"pass": capability_pass},
        )
        run_ledger.finalize(
            capability_spec.run_id,
            status="complete" if capability_pass else "capability-no-go",
            artifact=str(capability_dir / "capability.json"),
            failure=(
                None
                if capability_pass
                else {"error_type": "CapabilityOrInterfaceNoGo"}
            ),
        )

        preflight_spec = contract.expand(
            stage="long-context-preflight",
            model=model_id,
        )[0]
        preflight_dir = (
            raw_root
            / preflight_spec.stage_id
            / "runs"
            / preflight_spec.run_id
        )
        preflight_dir.mkdir(parents=True)
        orchestrator._atomic_json(
            preflight_dir / "preflight_checkpoint.json",
            {
                "run_config": {
                    "run_id": f"{preflight_spec.run_id}--actor-preflight"
                }
            },
        )
        orchestrator._atomic_json(
            preflight_dir / "preflight_checkpoint_exactness.json",
            {"receipt_hash": "3" * 64},
        )
        checks = {"all_interface_checks": True}
        orchestrator._atomic_json(
            preflight_dir / "gate_receipt.json",
            {
                "capability_pass": capability_pass,
                "preflight_checks": checks,
                "go": capability_pass,
            },
        )
        run_ledger.finalize(
            preflight_spec.run_id,
            status="complete" if capability_pass else "capability-no-go",
            artifact=str(preflight_dir / "gate_receipt.json"),
            failure=(
                None
                if capability_pass
                else {"error_type": "CapabilityOrInterfaceNoGo"}
            ),
        )


def _write_real_builder_gate_sources(
    *,
    raw_root: Path,
    contract,
    run_ledger: orchestrator.PilotRunLedger,
) -> None:
    for model_id in ("gpt52_main", "gpt56_diagnostic"):
        capability_spec = contract.expand(
            stage="capability-gate",
            model=model_id,
        )[0]
        capability_dir = (
            raw_root
            / capability_spec.stage_id
            / "runs"
            / capability_spec.run_id
        )
        capability_dir.mkdir(parents=True)
        capability = deepcopy(
            gate_fixtures._capability_artifact(model_id)["payload"]
        )
        orchestrator._atomic_json(
            capability_dir / "capability.json",
            capability,
        )
        run_ledger.finalize(
            capability_spec.run_id,
            status="complete",
            artifact=str(capability_dir / "capability.json"),
        )

        preflight_spec = contract.expand(
            stage="long-context-preflight",
            model=model_id,
        )[0]
        preflight_dir = (
            raw_root
            / preflight_spec.stage_id
            / "runs"
            / preflight_spec.run_id
        )
        preflight_dir.mkdir(parents=True)
        fixture = gate_fixtures._preflight_artifact(model_id)
        checkpoint = deepcopy(fixture["checkpoint"])
        checkpoint_run_id = f"{preflight_spec.run_id}--actor-preflight"
        checkpoint["run_config"]["run_id"] = checkpoint_run_id
        checkpoint["run_config"]["pilot_contract_hash"] = (
            contract.canonical_hash
        )
        journal = checkpoint["provider_call_journal_binding"]
        journal["run_id"] = checkpoint_run_id
        journal["contract_hash"] = contract.canonical_hash
        checkpoint["provider_call_journal_binding_hash"] = (
            orchestrator.canonical_sha256(journal)
        )
        checkpoint.pop("checkpoint_hash")
        checkpoint["checkpoint_hash"] = orchestrator.canonical_sha256(
            checkpoint
        )
        exactness = deepcopy(fixture["exactness"])
        exactness["checkpoint_hash"] = checkpoint["checkpoint_hash"]
        exactness["provider_call_journal_binding_hash"] = checkpoint[
            "provider_call_journal_binding_hash"
        ]
        exactness.pop("receipt_hash")
        exactness["receipt_hash"] = orchestrator.canonical_sha256(exactness)
        orchestrator._atomic_json(
            preflight_dir / "preflight_checkpoint.json",
            checkpoint,
        )
        orchestrator._atomic_json(
            preflight_dir / "preflight_checkpoint_exactness.json",
            exactness,
        )
        orchestrator._atomic_json(
            preflight_dir / "gate_receipt.json",
            {
                "capability_pass": True,
                "preflight_checks": {"all_interface_checks": True},
                "go": True,
            },
        )
        run_ledger.finalize(
            preflight_spec.run_id,
            status="complete",
            artifact=str(preflight_dir / "gate_receipt.json"),
        )


def _finalize_pre_science_budget(
    *,
    raw_root: Path,
    contract,
) -> PilotBudgetLedger:
    budget = PilotBudgetLedger(
        raw_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
    )
    run_ids = [
        contract.expand(stage="parent-import")[0].run_id,
        *[
            spec.run_id
            for stage_id in ("capability-gate", "long-context-preflight")
            for spec in contract.expand(stage=stage_id)
        ],
    ]
    for run_id in run_ids:
        projection = RunProjection(
            run_id=run_id,
            stage_bucket=(
                "parent_v2102"
                if run_id == run_ids[0]
                else "hosted_v211"
            ),
            cost_usd=0.0,
            completions=0,
            storage_bytes=10,
            basis={"method": "test"},
        )
        budget.reserve(projection)
        budget.finalize(
            run_id,
            status="complete",
            cost_usd=0.0,
            completions=0,
            storage_bytes=10,
        )
    return budget


def _seal_real_builder_gate(tmp_path: Path):
    contract = load_pilot_contract(CONTRACT_PATH)
    raw_root = tmp_path / "experiment_results" / "pilot-v2.11" / "raw"
    run_ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    run_ledger.register(contract.expand())
    parent_spec = contract.expand(stage="parent-import")[0]
    run_ledger.finalize(
        parent_spec.run_id,
        status="complete",
        artifact="parent-import.json",
    )
    _write_real_builder_gate_sources(
        raw_root=raw_root,
        contract=contract,
        run_ledger=run_ledger,
    )
    budget = _finalize_pre_science_budget(
        raw_root=raw_root,
        contract=contract,
    )
    path, receipt = orchestrator._persist_v211_post_gate_authority(
        contract,
        raw_root=raw_root,
        paid=_paid(),
        budget_ledger=budget,
        run_ledger=run_ledger,
    )
    return contract, raw_root, run_ledger, budget, path, receipt


def test_v211_post_gate_orchestrator_seals_after_model_scoped_no_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw_root = tmp_path / "experiment_results" / "pilot-v2.11" / "raw"
    run_ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    run_ledger.register(contract.expand())
    parent_spec = contract.expand(stage="parent-import")[0]
    run_ledger.finalize(
        parent_spec.run_id,
        status="complete",
        artifact="parent-import.json",
    )
    _write_gate_sources(
        raw_root=raw_root,
        contract=contract,
        run_ledger=run_ledger,
        gpt56_capability_pass=False,
    )
    budget = _finalize_pre_science_budget(
        raw_root=raw_root,
        contract=contract,
    )
    observed_heads: list[str] = []

    def fake_build(**inputs):
        statuses = inputs["model_terminal_statuses"]
        if statuses != {
            "gpt52_main": "eligible",
            "gpt56_diagnostic": "capability-no-go",
        }:
            raise orchestrator.PilotV211GateError(
                "terminal status differs from recomputed gates"
            )
        observed_heads.append(inputs["ledger_event_chain_head"])
        eligible = [
            model_id
            for model_id, status in statuses.items()
            if status == "eligible"
        ]
        return {
            "receipt_sha256": "4" * 64,
            "go": bool(eligible),
            "reasons": [],
            "model_decisions": {
                model_id: {
                    "terminal_status": status,
                    "eligible_for_science_dispatch": status == "eligible",
                }
                for model_id, status in statuses.items()
            },
            "denominator": {"eligible_model_ids": eligible},
            "bindings": {
                "ledger_event_chain_head": inputs["ledger_event_chain_head"]
            },
        }

    monkeypatch.setattr(
        orchestrator,
        "build_v211_post_gate_authority",
        fake_build,
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v211_gate_receipt",
        lambda receipt, **_: receipt,
    )

    path, receipt = orchestrator._persist_v211_post_gate_authority(
        contract,
        raw_root=raw_root,
        paid=_paid(),
        budget_ledger=budget,
        run_ledger=run_ledger,
    )

    assert path.exists()
    assert receipt["denominator"]["eligible_model_ids"] == ["gpt52_main"]
    assert len(observed_heads) == 2
    assert observed_heads[0] != observed_heads[1]
    assert observed_heads[1] == run_ledger.snapshot()["events"][-1][
        "event_sha256"
    ]
    assert all(
        run_ledger.status(spec.run_id) == "capability-no-go"
        for spec in contract.expand()
        if spec.model_id == "gpt56_diagnostic"
        and spec.stage_id == "cross-model"
    )
    assert all(
        run_ledger.status(spec.run_id) == "scheduled"
        for spec in contract.expand()
        if spec.model_id == "gpt52_main"
        and spec.stage_id
        not in {
            "parent-import",
            "capability-gate",
            "long-context-preflight",
        }
    )

    first_science = contract.expand(
        stage="experiment-c",
        model="gpt52_main",
    )[0]
    run_ledger.finalize(
        first_science.run_id,
        status="complete",
        artifact="science-result.json",
    )
    resumed_path, resumed_receipt = (
        orchestrator._persist_v211_post_gate_authority(
            contract,
            raw_root=raw_root,
            paid=_paid(),
            budget_ledger=budget,
            run_ledger=run_ledger,
        )
    )
    assert resumed_path == path
    assert resumed_receipt == receipt
    assert len(observed_heads) == 3
    assert observed_heads[-1] == receipt["bindings"]["ledger_event_chain_head"]


def test_v211_pre_gate_conservative_projection_uses_exact_call_denominator() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    capability = contract.expand(
        stage="capability-gate",
        model="gpt52_main",
    )[0]
    preflight = contract.expand(
        stage="long-context-preflight",
        model="gpt52_main",
    )[0]

    assert orchestrator._max_call_projection(contract, capability) == (
        30,
        6_000_000,
        30 * 4_096,
    )
    assert orchestrator._max_call_projection(contract, preflight) == (
        32,
        6_400_000,
        32 * 4_096,
    )


def test_v211_status_inference_allows_global_interface_no_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "gpt52_main": "eligible",
        "gpt56_diagnostic": "interface-no-go",
    }

    def fake_build(**inputs):
        if inputs["model_terminal_statuses"] != expected:
            raise orchestrator.PilotV211GateError("status mismatch")
        return {"model_decisions": {}, "go": True}

    monkeypatch.setattr(
        orchestrator,
        "build_v211_post_gate_authority",
        fake_build,
    )
    receipt, statuses = orchestrator._build_v211_post_gate_with_inferred_statuses(
        {
            "capability_artifacts": {
                model_id: {"payload": {"pass": True}}
                for model_id in expected
            }
        }
    )

    assert receipt["go"] is True
    assert statuses == expected


def test_v211_orchestrator_envelopes_pass_the_real_gate_and_verifier(
    tmp_path: Path,
) -> None:
    contract, _, _, _, path, receipt = _seal_real_builder_gate(
        tmp_path
    )

    assert path.exists()
    assert receipt["go"] is True
    assert receipt["actuals"]["hosted_completions"] == 124
    assert receipt["denominator"]["eligible_model_ids"] == [
        "gpt52_main",
        "gpt56_diagnostic",
    ]
    assert orchestrator.verify_v211_gate_receipt(
        receipt,
        expected_contract_sha256=contract.canonical_hash,
        expected_git_commit=_paid().head_commit,
    ) == receipt


def test_v211_resumed_post_gate_rejects_bound_source_drift(
    tmp_path: Path,
) -> None:
    contract, raw_root, run_ledger, budget, _, _ = _seal_real_builder_gate(
        tmp_path
    )
    capability_spec = contract.expand(
        stage="capability-gate",
        model="gpt52_main",
    )[0]
    capability_path = (
        raw_root
        / capability_spec.stage_id
        / "runs"
        / capability_spec.run_id
        / "capability.json"
    )
    capability = orchestrator._read_json(capability_path)
    capability["resume_drift_probe"] = True
    orchestrator._atomic_json(capability_path, capability)

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="differs from current sealed sources",
    ):
        orchestrator._persist_v211_post_gate_authority(
            contract,
            raw_root=raw_root,
            paid=_paid(),
            budget_ledger=budget,
            run_ledger=run_ledger,
        )


def test_v211_resumed_post_gate_rejects_pre_science_storage_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, raw_root, run_ledger, budget, _, _ = _seal_real_builder_gate(
        tmp_path
    )
    original_snapshot = budget.snapshot
    parent_run_id = contract.expand(stage="parent-import")[0].run_id

    def drifted_snapshot():
        snapshot = original_snapshot()
        snapshot["runs"][parent_run_id]["actual"]["storage_bytes"] += 1
        return snapshot

    monkeypatch.setattr(budget, "snapshot", drifted_snapshot)
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="differs from current sealed sources",
    ):
        orchestrator._persist_v211_post_gate_authority(
            contract,
            raw_root=raw_root,
            paid=_paid(),
            budget_ledger=budget,
            run_ledger=run_ledger,
        )


def test_existing_v2_stage_receipt_returns_after_prefix_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    output = raw_root / "long-context-preflight" / "stage_receipt.json"
    existing = {
        "status": "complete-with-no-go",
        "go_models": ["gpt52_main"],
        "artifacts": {"post_gate_authority": {"go": True}},
        "failure": None,
        "diagnostic_only": False,
        "scientific_evidence": None,
    }
    orchestrator._atomic_json(output, existing)
    verified: list[dict] = []
    monkeypatch.setattr(
        orchestrator,
        "_verify_v2_stage_receipt",
        lambda *args, **kwargs: verified.append(existing) or existing,
    )
    monkeypatch.setattr(
        orchestrator,
        "_v2_recomputed_stage_fields",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("existing receipt must not be rebuilt at a newer head")
        ),
    )
    contract = SimpleNamespace(schema_version="finevo-pilot-contract-v2")

    result = orchestrator._write_stage_receipt(
        contract,
        "long-context-preflight",
        raw_root=raw_root,
        ledger=SimpleNamespace(),
        status="complete-with-no-go",
        go_models=["gpt52_main"],
        artifacts={"post_gate_authority": {"go": True}},
        paid=_paid(),
    )

    assert result == output
    assert verified == [existing]
