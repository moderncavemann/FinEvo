from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verified_memory import pilot_orchestrator as orchestrator
from verified_memory import pilot_v25_parent_import as parent_import
from verified_memory.pilot_contract import load_pilot_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_5_overlay.yaml"


def _paid() -> SimpleNamespace:
    return SimpleNamespace(
        git_tag=parent_import.V25_SCIENCE_TAG,
        head_commit="a" * 40,
    )


def test_v25_runtime_uses_parent_import_controls() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)

    assert orchestrator._materializes_legacy_amendment_controls(contract) is False
    assert orchestrator._cross_model_science_stage_ids(contract) == ()


def test_v25_orchestrator_uses_single_parent_import_path(tmp_path: Path) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw_root = tmp_path / "experiment_results" / "pilot-v2.5" / "raw"

    path = orchestrator._observed_p95_authority_receipt_path(
        contract,
        "gpt52_main",
        raw_root=raw_root,
    )

    assert path == (
        raw_root
        / "parent-import"
        / "observed_p95"
        / "gpt52_main"
        / "observed_p95_authority_receipt.json"
    )
    assert path.parts.count("parent-import") == 1


def test_v25_parent_debit_has_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    sentinel = object()
    monkeypatch.setattr(
        orchestrator,
        "parent_budget_debit_for_v25",
        lambda _contract: sentinel,
    )
    monkeypatch.setattr(
        orchestrator,
        "parent_budget_debit_for_v24",
        lambda _contract: (_ for _ in ()).throw(
            AssertionError("V2.4 fallback must not run for V2.5")
        ),
    )

    assert orchestrator._parent_budget_debit(contract) is sentinel


def test_v25_parent_import_routes_without_provider_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid()
    parent_root = tmp_path / "parent"
    parent_root.mkdir()
    calls: list[dict[str, Any]] = []
    ledger = SimpleNamespace(register=lambda _specs: None)

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("parent import must not construct a provider")

    def route(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "complete", "provider_calls": 0}

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: paid,
    )
    monkeypatch.setattr(
        orchestrator,
        "_persist_release_attestation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator,
        "PilotRunLedger",
        lambda *_args, **_kwargs: ledger,
    )
    monkeypatch.setattr(
        orchestrator,
        "_execute_v24_parent_import_stage",
        route,
    )
    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden_provider)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden_provider)
    monkeypatch.setattr(orchestrator, "PilotBudgetLedger", forbidden_provider)

    result = orchestrator._execute_stage_locked(
        contract_path=CONTRACT_PATH,
        stage_id="parent-import",
        resume=True,
        raw_root=tmp_path / "raw",
        repo_root=tmp_path,
        parent_repo_root=parent_root,
    )

    assert result == {"status": "complete", "provider_calls": 0}
    assert len(calls) == 1
    assert calls[0]["parent_repo_root"] == parent_root


def test_v25_parent_import_calls_v25_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="parent-import")[0]
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    parent_root = tmp_path / "parent"
    parent_root.mkdir()
    calls: list[dict[str, Any]] = []

    class Ledger:
        def __init__(self) -> None:
            self.terminal = False

        def is_terminal(self, _run_id: str) -> bool:
            return self.terminal

        def finalize(self, *_args: Any, **_kwargs: Any) -> None:
            self.terminal = True

    def persist(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "provider_calls": 0,
            "scientific_evidence": False,
            "receipt": "synthetic.json",
        }

    terminal_path = raw_root / "parent-import" / "summary.json"
    stage_receipt = raw_root / "parent-import" / "stage_receipt.json"

    def write_terminal(path: Path, **_kwargs: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return path

    def write_stage(*_args: Any, **_kwargs: Any) -> Path:
        stage_receipt.parent.mkdir(parents=True, exist_ok=True)
        stage_receipt.write_text(
            json.dumps({"status": "complete"}) + "\n",
            encoding="utf-8",
        )
        return stage_receipt

    monkeypatch.setattr(orchestrator, "persist_v25_parent_import", persist)
    monkeypatch.setattr(
        orchestrator,
        "persist_v24_parent_import",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("V2.5 must not call the V2.4 adapter")
        ),
    )
    monkeypatch.setattr(orchestrator, "write_terminal_summary", write_terminal)
    monkeypatch.setattr(orchestrator, "_write_stage_receipt", write_stage)

    result = orchestrator._execute_v24_parent_import_stage(
        contract,
        (spec,),
        raw_root=raw_root,
        repo_root=tmp_path,
        parent_repo_root=parent_root,
        paid=_paid(),
        run_ledger=Ledger(),
    )

    assert result == {"status": "complete"}
    assert len(calls) == 1
    assert calls[0]["raw_root"] == raw_root
    assert calls[0]["parent_repo_root"] == parent_root


def test_v25_import_failure_terminalizes_all_211_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw_root = tmp_path / "raw"
    ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())
    spec = contract.expand(stage="parent-import")[0]
    stage_receipt = raw_root / "parent-import" / "stage_receipt.json"

    def fail(**_kwargs: Any) -> None:
        raise parent_import.PilotV25ParentImportError("synthetic import failure")

    def write_stage(*_args: Any, **_kwargs: Any) -> Path:
        stage_receipt.parent.mkdir(parents=True, exist_ok=True)
        stage_receipt.write_text("{}\n", encoding="utf-8")
        return stage_receipt

    monkeypatch.setattr(orchestrator, "persist_v25_parent_import", fail)
    monkeypatch.setattr(orchestrator, "_write_stage_receipt", write_stage)

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="V2.5 parent import failed",
    ):
        orchestrator._execute_v24_parent_import_stage(
            contract,
            (spec,),
            raw_root=raw_root,
            repo_root=tmp_path,
            parent_repo_root=tmp_path,
            paid=_paid(),
            run_ledger=ledger,
        )

    rows = ledger.snapshot()["runs"]
    assert len(rows) == 211
    assert {row["status"] for row in rows.values()} == {"integrity-stopped"}
    assert all(
        row["failure"]["source_stage"] == "parent-import"
        for run_id, row in rows.items()
        if run_id != spec.run_id
    )


def test_v25_control_gate_uses_v25_receipt_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    path = tmp_path / "raw" / "parent-import" / "parent_import_receipt.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    calls: list[Path] = []

    def verify(receipt_path: Path, **_kwargs: Any) -> dict[str, Any]:
        calls.append(receipt_path)
        return {}

    monkeypatch.setattr(orchestrator, "verify_v25_parent_import_receipt", verify)
    monkeypatch.setattr(
        orchestrator,
        "verify_v24_parent_import_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("V2.5 control gate must not invoke V2.4 verifier")
        ),
    )

    assert (
        orchestrator._v2_control_gate_ok(
            contract,
            "parent-import",
            raw_root=tmp_path / "raw",
            paid=_paid(),
        )
        is True
    )
    assert calls == [path]
