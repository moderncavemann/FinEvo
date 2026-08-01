from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from verified_memory import pilot_orchestrator as orchestrator
from verified_memory import pilot_v21111_fresh_cohort as fresh_cohort
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_v21111_fresh_cohort import (
    V21111_FAULT_ACCEPTANCE_CHECKS,
    PilotV21111FreshCohortError,
    require_exact_diagnostics_namespace,
    require_exact_raw_namespace,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_11.yaml"
KEY_NAMES = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)


def _copy_draft_contract(repo: Path) -> Path:
    target = repo / "experiments" / "pilot_v2_11_11.yaml"
    target.parent.mkdir(parents=True)
    shutil.copyfile(CONTRACT_PATH, target)
    return target


def _clear_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in KEY_NAMES:
        monkeypatch.delenv(name, raising=False)


def _write_minimal_fault_acceptance_fixture(
    diagnostics: Path,
    contract,
) -> tuple[Path, dict]:
    root = diagnostics / "provider-free-fault-isolation-acceptance"
    root.mkdir(parents=True)
    scenario_bindings = {}
    for scenario in orchestrator.V21111_FAULT_SCENARIO_NAMES:
        scenario_root = root / scenario
        scenario_root.mkdir()
        run = orchestrator.PilotRunLedger(
            scenario_root / "run_ledger.json",
            contract_hash=contract.canonical_hash,
            tamper_evident=True,
            bind_terminal_artifacts=True,
        )
        budget = orchestrator.PilotBudgetLedger(
            scenario_root / "budget_ledger.json",
            contract_hash=contract.canonical_hash,
            caps=orchestrator._budget_caps(contract),
            tamper_evident=True,
            parent_debit=orchestrator._parent_budget_debit(contract),
        )
        scenario_bindings[scenario] = orchestrator._v21111_fault_scenario_binding(
            acceptance_root=root,
            scenario_root=scenario_root,
            run_ledger=run,
            budget_ledger=budget,
        )
    fault = {
        "schema_version": "finevo-pilot-v2.11.11-fake-fault-acceptance-v1",
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "status": "pass",
        "external_provider_calls": 0,
        "hosted_cost_usd": 0.0,
        "diagnostic_only": True,
        "scientific_evidence": False,
        "checks": {name: True for name in V21111_FAULT_ACCEPTANCE_CHECKS},
        "scenario_ledger_bindings": scenario_bindings,
    }
    fault["receipt_sha256"] = orchestrator.canonical_sha256(fault)
    fault_path = root / "acceptance_receipt.json"
    fault_path.write_text(
        json.dumps(fault, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return fault_path, fault


def test_v21111_raw_and_fake_diagnostics_namespaces_are_disjoint_and_nonsymlinked(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    contract_path = _copy_draft_contract(repo)
    raw = repo / "experiment_results" / "pilot-v2.11.11" / "raw"
    diagnostics = repo / "experiment_results" / "pilot-v2.11.11" / "diagnostics"

    assert (
        require_exact_raw_namespace(
            contract_path=contract_path,
            raw_root=raw,
        )
        == raw
    )
    assert (
        require_exact_diagnostics_namespace(
            contract_path=contract_path,
            diagnostics_root=diagnostics,
        )
        == diagnostics
    )
    assert not raw.exists()
    assert not diagnostics.exists()

    alias_repo = tmp_path / "alias-repo"
    alias_contract = _copy_draft_contract(alias_repo)
    historical = tmp_path / "historical-raw"
    historical.mkdir()
    pilot_parent = alias_repo / "experiment_results" / "pilot-v2.11.11"
    pilot_parent.parent.mkdir(parents=True)
    pilot_parent.symlink_to(historical, target_is_directory=True)

    with pytest.raises(PilotV21111FreshCohortError, match="symlink"):
        require_exact_raw_namespace(
            contract_path=alias_contract,
            raw_root=pilot_parent / "raw",
        )
    with pytest.raises(PilotV21111FreshCohortError, match="symlink"):
        require_exact_diagnostics_namespace(
            contract_path=alias_contract,
            diagnostics_root=pilot_parent / "diagnostics",
        )


@pytest.mark.parametrize("namespace", ("raw", "diagnostics"))
@pytest.mark.parametrize("symlink_location", ("intermediate", "final"))
def test_v21111_namespaces_reject_dangling_symlink_components(
    tmp_path: Path,
    namespace: str,
    symlink_location: str,
) -> None:
    repo = tmp_path / f"{namespace}-{symlink_location}" / "repo"
    contract_path = _copy_draft_contract(repo)
    pilot_root = repo / "experiment_results" / "pilot-v2.11.11"
    target = pilot_root / namespace
    dangling = tmp_path / f"absent-{namespace}-{symlink_location}"
    assert not dangling.exists()
    if symlink_location == "intermediate":
        pilot_root.parent.mkdir(parents=True)
        pilot_root.symlink_to(dangling, target_is_directory=True)
    else:
        pilot_root.mkdir(parents=True)
        target.symlink_to(dangling, target_is_directory=True)

    checker = (
        require_exact_raw_namespace
        if namespace == "raw"
        else require_exact_diagnostics_namespace
    )
    keyword = "raw_root" if namespace == "raw" else "diagnostics_root"
    with pytest.raises(PilotV21111FreshCohortError, match="symlink"):
        checker(
            contract_path=contract_path,
            **{keyword: target},
        )


def test_v21111_parent_import_prewrite_blocks_before_source_or_stage_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_keys(monkeypatch)
    contract = load_pilot_contract(CONTRACT_PATH)
    repo = tmp_path / "repo"
    contract_path = _copy_draft_contract(repo)
    raw = repo / "experiment_results" / "pilot-v2.11.11" / "raw"
    raw.mkdir(parents=True)
    (raw / "foreign-prewrite.json").write_text("{}\n", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(
        orchestrator,
        "load_pilot_contract",
        lambda _path: contract,
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v21111_parent_sources",
        lambda *_args, **_kwargs: calls.append("source"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_execute_stage_locked",
        lambda **_kwargs: calls.append("stage"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_exclusive_real_stage_lock",
        lambda *_args, **_kwargs: nullcontext(),
    )

    with pytest.raises(orchestrator.PilotOrchestrationError, match="must be empty"):
        orchestrator.execute_stage(
            contract_path=contract_path,
            stage_id="parent-import",
            resume=False,
            raw_root=raw,
            repo_root=repo,
            parent_repo_root=tmp_path / "v21110",
            authority_repo_root=tmp_path / "v2115",
        )
    assert calls == []


@pytest.mark.parametrize("crash_window", ("receipt-only", "run-marker-only"))
def test_v21111_acceptance_resume_repairs_only_missing_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_window: str,
) -> None:
    _clear_provider_keys(monkeypatch)
    contract = replace(load_pilot_contract(CONTRACT_PATH), status="frozen")
    repo = tmp_path / "repo"
    contract_path = _copy_draft_contract(repo)
    raw = repo / "experiment_results" / "pilot-v2.11.11" / "raw"
    diagnostics = repo / "experiment_results" / "pilot-v2.11.11" / "diagnostics"
    raw.mkdir(parents=True)
    diagnostics.mkdir(parents=True)

    run = orchestrator.PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    run.register(contract.expand())
    (parent,) = contract.expand(stage="parent-import")
    run.finalize(parent.run_id, status="complete", artifact=None)
    budget = orchestrator.PilotBudgetLedger(
        raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=orchestrator._parent_budget_debit(contract),
    )
    parent_projection = orchestrator._v21111_parent_import_projection(parent)
    budget.reserve(parent_projection)
    budget.finalize(
        parent.run_id,
        status="complete",
        cost_usd=0.0,
        completions=0,
        storage_bytes=0,
    )
    parent_stage_receipt = raw / "parent-import" / "stage_receipt.json"
    parent_stage_receipt.parent.mkdir(parents=True)
    parent_stage_receipt.write_text("{}\n", encoding="utf-8")

    full_fake_path = (
        diagnostics / "provider-free-full-fake-acceptance" / "acceptance_receipt.json"
    )
    full_fake_path.parent.mkdir(parents=True)
    full_fake_path.write_text("{}\n", encoding="utf-8")
    fault_path, fault = _write_minimal_fault_acceptance_fixture(
        diagnostics,
        contract,
    )
    paid = orchestrator.GitProvenance(
        git_tag="pilot-v2.11.11-science",
        head_commit="a" * 40,
        tag_commit="a" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )
    monkeypatch.setattr(fresh_cohort, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        fresh_cohort,
        "verify_parent_import_receipt",
        lambda *_args, **_kwargs: {"status": "complete"},
    )
    monkeypatch.setattr(
        fresh_cohort,
        "verify_full_fake_acceptance",
        lambda *_args, **_kwargs: {"receipt_sha256": "e" * 64},
    )
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
        "_verify_v2_stage_receipt",
        lambda *_args, **_kwargs: {"status": "complete"},
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v21111_fake_fault_acceptance",
        lambda *_args, **_kwargs: fault,
    )
    monkeypatch.setattr(
        orchestrator,
        "create_llm_provider",
        lambda *_args, **_kwargs: pytest.fail(
            "provider constructed during provider-free acceptance recovery"
        ),
    )

    original_run_bind = orchestrator.PilotRunLedger.bind_acceptance_receipt
    original_budget_bind = orchestrator.PilotBudgetLedger.bind_acceptance_receipt

    def crash_before_marker(self, **_kwargs):
        raise RuntimeError("fixture crash before acceptance marker")

    if crash_window == "receipt-only":
        monkeypatch.setattr(
            orchestrator.PilotRunLedger,
            "bind_acceptance_receipt",
            crash_before_marker,
        )
    else:
        monkeypatch.setattr(
            orchestrator.PilotBudgetLedger,
            "bind_acceptance_receipt",
            crash_before_marker,
        )
    launch = diagnostics / "scientific_launch_input.json"
    with pytest.raises(RuntimeError, match="before acceptance marker"):
        fresh_cohort.accept_scientific_dispatch(
            contract_path=contract_path,
            repo_root=repo,
            raw_root=raw,
            scientific_launch_input_path=launch,
            diagnostics_root=diagnostics,
        )
    monkeypatch.setattr(
        orchestrator.PilotRunLedger,
        "bind_acceptance_receipt",
        original_run_bind,
    )
    monkeypatch.setattr(
        orchestrator.PilotBudgetLedger,
        "bind_acceptance_receipt",
        original_budget_bind,
    )

    acceptance_path = raw / fresh_cohort.V21111_ACCEPTANCE_FILENAME
    acceptance_before = acceptance_path.read_bytes()
    run_before = (raw / "run_ledger.json").read_bytes()
    run_events_before = orchestrator.PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    ).snapshot()["events"]
    budget_events_before = orchestrator.PilotBudgetLedger(
        raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=orchestrator._parent_budget_debit(contract),
    ).snapshot()["events"]
    assert sum(
        event["event_type"] == "acceptance_receipt_bound" for event in run_events_before
    ) == (1 if crash_window == "run-marker-only" else 0)
    assert (
        sum(
            event["event_type"] == "acceptance_receipt_bound"
            for event in budget_events_before
        )
        == 0
    )

    recovered = fresh_cohort.accept_scientific_dispatch(
        contract_path=contract_path,
        repo_root=repo,
        raw_root=raw,
        scientific_launch_input_path=launch,
        diagnostics_root=diagnostics,
    )
    assert acceptance_path.read_bytes() == acceptance_before
    run_events_after = orchestrator.PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    ).snapshot()["events"]
    budget_events_after = orchestrator.PilotBudgetLedger(
        raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=orchestrator._parent_budget_debit(contract),
    ).snapshot()["events"]
    assert run_events_after[: len(run_events_before)] == run_events_before
    assert len(run_events_after) == len(run_events_before) + (
        0 if crash_window == "run-marker-only" else 1
    )
    assert run_events_after[-1]["event_type"] == "acceptance_receipt_bound"
    if crash_window == "run-marker-only":
        assert (raw / "run_ledger.json").read_bytes() == run_before
    assert budget_events_after[: len(budget_events_before)] == budget_events_before
    assert len(budget_events_after) == len(budget_events_before) + 1
    assert budget_events_after[-1]["event_type"] == "acceptance_receipt_bound"
    verified = fresh_cohort.verify_scientific_dispatch_acceptance(
        contract=contract,
        raw_root=raw,
        paid=paid,
    )
    assert verified["integrity"] == recovered["integrity"]


def test_v21111_fake_fault_verifier_rejects_empty_self_consistent_ledgers(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    diagnostics = tmp_path / "diagnostics"
    _write_minimal_fault_acceptance_fixture(diagnostics, contract)

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="run ID denominator drifted",
    ):
        orchestrator.verify_v21111_fake_fault_acceptance(
            diagnostics,
            contract=contract,
        )


def test_v21111_parent_resume_revalidates_sources_before_entering_stage_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_keys(monkeypatch)
    contract = load_pilot_contract(CONTRACT_PATH)
    repo = tmp_path / "repo"
    contract_path = _copy_draft_contract(repo)
    raw = repo / "experiment_results" / "pilot-v2.11.11" / "raw"
    raw.mkdir(parents=True)
    (raw / "existing-ledger.json").write_text("{}\n", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(
        orchestrator,
        "load_pilot_contract",
        lambda _path: contract,
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v21111_parent_sources",
        lambda *_args, **_kwargs: calls.append("source"),
    )

    def lock(*_args, **_kwargs):
        calls.append("lock")
        return nullcontext()

    monkeypatch.setattr(orchestrator, "_exclusive_real_stage_lock", lock)
    monkeypatch.setattr(
        orchestrator,
        "_execute_stage_locked",
        lambda **_kwargs: calls.append("stage") or {"status": "complete"},
    )

    result = orchestrator.execute_stage(
        contract_path=contract_path,
        stage_id="parent-import",
        resume=True,
        raw_root=raw,
        repo_root=repo,
        parent_repo_root=tmp_path / "v21110",
        authority_repo_root=tmp_path / "v2115",
    )
    assert result == {"status": "complete"}
    assert calls == ["source", "lock", "stage"]


def test_v21111_completed_parent_import_resume_is_idempotent_and_never_rewrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_keys(monkeypatch)
    contract = replace(load_pilot_contract(CONTRACT_PATH), status="frozen")
    raw = tmp_path / "raw"
    spec = contract.expand(stage="parent-import")[0]
    run_ledger = orchestrator.PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    run_ledger.register(contract.expand())
    terminal = raw / "parent-import" / "summaries" / f"{spec.run_id}.json"
    terminal.parent.mkdir(parents=True)
    terminal.write_text("{}\n", encoding="utf-8")
    run_ledger.finalize(
        spec.run_id,
        status="complete",
        artifact=str(terminal),
    )
    budget = orchestrator.PilotBudgetLedger(
        raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=orchestrator._parent_budget_debit(contract),
    )
    projection = orchestrator._v21111_parent_import_projection(spec)
    budget.reserve(projection)
    budget.finalize(
        spec.run_id,
        status="complete",
        cost_usd=0.0,
        completions=0,
        storage_bytes=0,
    )
    stage_receipt_path = orchestrator._stage_receipt_path(raw, "parent-import")
    stage_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    stage_receipt_path.write_text('{"status":"complete"}\n', encoding="utf-8")
    paid = orchestrator.GitProvenance(
        git_tag="pilot-v2.11.11-science",
        head_commit="a" * 40,
        tag_commit="a" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )
    verified = {"status": "complete", "resume_provider_calls": 0}
    verification_calls = 0

    def verify_parent(*_args, **_kwargs):
        nonlocal verification_calls
        verification_calls += 1
        return {"integrity": {"content_sha256": "b" * 64}}

    monkeypatch.setattr(
        orchestrator,
        "verify_v21111_parent_import_receipt",
        verify_parent,
    )
    monkeypatch.setattr(
        orchestrator,
        "_verify_v2_stage_receipt",
        lambda *_args, **_kwargs: verified,
    )
    monkeypatch.setattr(
        orchestrator,
        "write_v21111_parent_import_receipt",
        lambda *_args, **_kwargs: pytest.fail("completed parent import was rewritten"),
    )
    monkeypatch.setattr(
        orchestrator,
        "create_llm_provider",
        lambda *_args, **_kwargs: pytest.fail("completed parent import built provider"),
    )
    run_before = run_ledger.snapshot()
    budget_before = budget.snapshot()

    first = orchestrator._execute_v21111_parent_import_stage(
        contract,
        (spec,),
        raw_root=raw,
        repo_root=tmp_path,
        failed_repo_root=tmp_path / "v21110",
        authority_repo_root=tmp_path / "v2115",
        paid=paid,
        run_ledger=run_ledger,
        budget_ledger=budget,
    )
    second = orchestrator._execute_v21111_parent_import_stage(
        contract,
        (spec,),
        raw_root=raw,
        repo_root=tmp_path,
        failed_repo_root=tmp_path / "v21110",
        authority_repo_root=tmp_path / "v2115",
        paid=paid,
        run_ledger=run_ledger,
        budget_ledger=budget,
    )

    assert first == second == verified
    assert verification_calls == 2
    assert run_ledger.snapshot() == run_before
    assert budget.snapshot() == budget_before


def test_v21111_refresh_requires_acceptance_before_catalog_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_keys(monkeypatch)
    contract = replace(load_pilot_contract(CONTRACT_PATH), status="frozen")
    repo = tmp_path / "repo"
    contract_path = _copy_draft_contract(repo)
    raw = repo / "experiment_results" / "pilot-v2.11.11" / "raw"
    calls: list[str] = []
    paid = orchestrator.GitProvenance(
        git_tag="pilot-v2.11.11-science",
        head_commit="a" * 40,
        tag_commit="a" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: calls.append("paid") or paid,
    )

    def reject_acceptance(**_kwargs):
        calls.append("acceptance")
        raise orchestrator.PilotV21111FreshCohortError("fixture no-go")

    monkeypatch.setattr(
        orchestrator,
        "verify_v21111_scientific_dispatch_acceptance",
        reject_acceptance,
    )
    monkeypatch.setattr(
        orchestrator,
        "_exclusive_real_stage_lock",
        lambda *_args, **_kwargs: pytest.fail("lock entered before acceptance"),
    )
    monkeypatch.setattr(
        orchestrator,
        "validate_live_provider_catalog",
        lambda *_args, **_kwargs: pytest.fail("catalog queried before acceptance"),
    )
    monkeypatch.setattr(
        orchestrator,
        "create_llm_provider",
        lambda *_args, **_kwargs: pytest.fail("provider built before acceptance"),
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="refresh lacks provider-free acceptance",
    ):
        orchestrator.execute_v21111_dispatch_refresh_stage(
            contract_path=contract_path,
            resume=False,
            raw_root=raw,
            repo_root=repo,
        )
    assert calls == ["paid", "acceptance"]
    assert not raw.exists()


def test_v21111_science_requires_refresh_go_before_any_raw_write_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = replace(load_pilot_contract(CONTRACT_PATH), status="frozen")
    repo = tmp_path / "repo"
    contract_path = _copy_draft_contract(repo)
    raw = repo / "experiment_results" / "pilot-v2.11.11" / "raw"
    calls: list[str] = []
    paid = orchestrator.GitProvenance(
        git_tag="pilot-v2.11.11-science",
        head_commit="a" * 40,
        tag_commit="a" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: calls.append("paid") or paid,
    )

    def reject_refresh(**_kwargs):
        calls.append("refresh")
        raise orchestrator.PilotV21111DispatchRefreshError("fixture no-go")

    monkeypatch.setattr(
        orchestrator,
        "verify_v21111_dispatch_refresh_go",
        reject_refresh,
    )
    monkeypatch.setattr(
        orchestrator,
        "_persist_release_attestation",
        lambda *_args, **_kwargs: pytest.fail("raw write preceded refresh GO"),
    )
    monkeypatch.setattr(
        orchestrator,
        "PilotRunLedger",
        lambda *_args, **_kwargs: pytest.fail("ledger preceded refresh GO"),
    )
    monkeypatch.setattr(
        orchestrator,
        "create_llm_provider",
        lambda *_args, **_kwargs: pytest.fail("provider preceded refresh GO"),
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="dispatch refresh is not GO",
    ):
        orchestrator._execute_stage_locked(
            contract_path=contract_path,
            stage_id="experiment-b",
            resume=False,
            raw_root=raw,
            repo_root=repo,
        )
    assert calls == ["paid", "refresh"]
    assert not raw.exists()


def test_v21111_full_fake_and_fault_receipts_cover_actual_runner_without_science_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_keys(monkeypatch)
    repo = tmp_path / "repo"
    contract_path = _copy_draft_contract(repo)
    diagnostics = repo / "experiment_results" / "pilot-v2.11.11" / "diagnostics"
    raw = repo / "experiment_results" / "pilot-v2.11.11" / "raw"

    full = orchestrator.run_v21111_full_fake_acceptance(
        contract_path=contract_path,
        diagnostics_root=diagnostics,
        resume=False,
    )
    assert full["status"] == "pass"
    assert full["registered_science_cells"] == 86
    assert full["status_counts"] == {"complete": 86}
    assert full["simulated_science_calls"] == 3_256
    assert full["simulated_calls_by_stage"] == {
        "experiment-b": 1_440,
        "experiment-d": 1_480,
        "cross-model": 336,
    }
    assert full["fake_provider_adapter_calls"] == 3_256
    assert full["task_call_counts"] == {
        "actor-action": 2_928,
        "semantic-proposal": 328,
    }
    assert full["external_provider_calls"] == 0
    assert full["hosted_cost_usd"] == 0.0
    assert all(full["checks"].values())
    assert full["diagnostic_only"] is True
    assert full["scientific_evidence"] is False
    full_receipt = Path(full["receipt"])
    assert diagnostics in full_receipt.parents
    assert raw not in full_receipt.parents
    assert not raw.exists()
    full_bytes = full_receipt.read_bytes()

    resumed_full = orchestrator.run_v21111_full_fake_acceptance(
        contract_path=contract_path,
        diagnostics_root=diagnostics,
        resume=True,
    )
    assert resumed_full["resume_provider_calls"] == 0
    assert full_receipt.read_bytes() == full_bytes

    fault = orchestrator.run_v21111_fake_fault_acceptance(
        contract_path=contract_path,
        diagnostics_root=diagnostics,
        resume=False,
    )
    assert fault["status"] == "pass"
    assert set(fault["checks"]) == V21111_FAULT_ACCEPTANCE_CHECKS
    assert all(fault["checks"].values())
    assert fault["external_provider_calls"] == 0
    assert fault["hosted_cost_usd"] == 0.0
    assert fault["diagnostic_only"] is True
    assert fault["scientific_evidence"] is False
    assert fault["prefix_failure"]["status_counts"] == {
        "failed": 11,
        "complete": 2,
        "scheduled": 73,
    }
    assert fault["branch_failure"]["status_counts"] == {
        "failed": 1,
        "complete": 10,
    }
    assert fault["branch_interruption_resume"]["status_counts"] == {
        "integrity-stopped": 1,
        "complete": 10,
    }
    assert fault["prefix_interruption_resume"]["status_counts"] == {
        "integrity-stopped": 11,
    }
    assert fault["terminal_commit_window"]["status_counts"] == {"complete": 11}
    fault_receipt = Path(fault["receipt"])
    assert diagnostics in fault_receipt.parents
    assert raw not in fault_receipt.parents
    assert not raw.exists()
    fault_bytes = fault_receipt.read_bytes()

    resumed_fault = orchestrator.run_v21111_fake_fault_acceptance(
        contract_path=contract_path,
        diagnostics_root=diagnostics,
        resume=True,
    )
    assert resumed_fault["resume_provider_calls"] == 0
    assert fault_receipt.read_bytes() == fault_bytes

    # No fake output may identify itself as scientific evidence, including the
    # terminal branch artifacts nested below the diagnostics tree.
    for path in diagnostics.rglob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and "scientific_evidence" in value:
            assert value["scientific_evidence"] is False, path


def test_v21111_full_fake_resume_and_verifier_reject_mutated_complete_d_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_keys(monkeypatch)
    repo = tmp_path / "repo"
    contract_path = _copy_draft_contract(repo)
    contract = load_pilot_contract(contract_path)
    diagnostics = repo / "experiment_results" / "pilot-v2.11.11" / "diagnostics"

    completed = orchestrator.run_v21111_full_fake_acceptance(
        contract_path=contract_path,
        diagnostics_root=diagnostics,
        resume=False,
    )
    assert completed["status"] == "pass"
    root = diagnostics / "provider-free-full-fake-acceptance"
    run_payload = json.loads((root / "run_ledger.json").read_text(encoding="utf-8"))
    d_rows = [
        row
        for run_id, row in run_payload["runs"].items()
        if "--experiment-d--" in run_id
    ]
    assert len(d_rows) == 55
    branch_result = Path(d_rows[0]["artifact"])
    assert branch_result.name == "branch_result.json"
    original = branch_result.read_bytes()
    branch_result.write_bytes(original + b" ")
    assert branch_result.stat().st_size == len(original) + 1

    rejected: dict[str, str] = {}
    try:
        orchestrator.run_v21111_full_fake_acceptance(
            contract_path=contract_path,
            diagnostics_root=diagnostics,
            resume=True,
        )
    except (
        orchestrator.PilotOrchestrationError,
        PilotV21111FreshCohortError,
    ) as exc:
        rejected["resume"] = str(exc)
    try:
        fresh_cohort.verify_full_fake_acceptance(
            diagnostics,
            contract=contract,
        )
    except (
        orchestrator.PilotOrchestrationError,
        PilotV21111FreshCohortError,
    ) as exc:
        rejected["verify"] = str(exc)

    assert set(rejected) == {"resume", "verify"}
    assert all("artifact" in message.lower() for message in rejected.values())


def test_v21111_fake_fault_resume_and_scientific_acceptance_reject_mutated_d_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_keys(monkeypatch)
    repo = tmp_path / "repo"
    contract_path = _copy_draft_contract(repo)
    loaded_contract = load_pilot_contract(contract_path)
    frozen_contract = replace(loaded_contract, status="frozen")
    diagnostics = repo / "experiment_results" / "pilot-v2.11.11" / "diagnostics"
    raw = repo / "experiment_results" / "pilot-v2.11.11" / "raw"
    monkeypatch.setattr(
        orchestrator,
        "load_pilot_contract",
        lambda _path: frozen_contract,
    )

    completed = orchestrator.run_v21111_fake_fault_acceptance(
        contract_path=contract_path,
        diagnostics_root=diagnostics,
        resume=False,
    )
    assert completed["status"] == "pass"
    scenario_root = (
        diagnostics / "provider-free-fault-isolation-acceptance" / "branch-failure"
    )
    run_payload = json.loads(
        (scenario_root / "run_ledger.json").read_text(encoding="utf-8")
    )
    complete_d_rows = [
        row
        for run_id, row in run_payload["runs"].items()
        if "--experiment-d--" in run_id and row["status"] == "complete"
    ]
    assert len(complete_d_rows) == 10
    branch_result = Path(complete_d_rows[0]["artifact"])
    assert branch_result.name == "branch_result.json"
    original = branch_result.read_bytes()
    branch_result.write_bytes(original + b" ")
    assert branch_result.stat().st_size == len(original) + 1

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="artifact binding drifted",
    ):
        orchestrator.run_v21111_fake_fault_acceptance(
            contract_path=contract_path,
            diagnostics_root=diagnostics,
            resume=True,
        )

    monkeypatch.setattr(
        fresh_cohort,
        "load_pilot_contract",
        lambda _path: frozen_contract,
    )
    monkeypatch.setattr(
        fresh_cohort,
        "verify_parent_import_receipt",
        lambda *_args, **_kwargs: {"status": "complete"},
    )
    monkeypatch.setattr(
        fresh_cohort,
        "verify_full_fake_acceptance",
        lambda *_args, **_kwargs: {"receipt_sha256": "e" * 64},
    )
    with pytest.raises(
        PilotV21111FreshCohortError,
        match="artifact binding drifted",
    ):
        fresh_cohort.accept_scientific_dispatch(
            contract_path=contract_path,
            repo_root=repo,
            raw_root=raw,
            scientific_launch_input_path=(diagnostics / "scientific_launch_input.json"),
            diagnostics_root=diagnostics,
        )


def test_v21111_fake_fault_verifier_rejects_resealed_usage_and_transition_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_keys(monkeypatch)
    repo = tmp_path / "repo"
    contract_path = _copy_draft_contract(repo)
    contract = load_pilot_contract(contract_path)
    diagnostics = repo / "experiment_results" / "pilot-v2.11.11" / "diagnostics"
    orchestrator.run_v21111_fake_fault_acceptance(
        contract_path=contract_path,
        diagnostics_root=diagnostics,
        resume=False,
    )
    root = diagnostics / "provider-free-fault-isolation-acceptance"
    receipt_path = root / "acceptance_receipt.json"
    original_receipt = receipt_path.read_bytes()

    scenario = "branch-failure"
    budget_path = root / scenario / "budget_ledger.json"
    original_budget = budget_path.read_bytes()
    budget = json.loads(original_budget)
    failed_run_id = next(
        run_id for run_id, row in budget["runs"].items() if row["status"] == "failed"
    )
    budget["runs"][failed_run_id]["actual"]["completions"] = 0
    previous = "0" * 64
    for event in budget["events"]:
        if (
            event["event_type"] == "run_finalized"
            and event["payload"].get("run_id") == failed_run_id
        ):
            event["payload"]["actual_sha256"] = orchestrator.canonical_sha256(
                budget["runs"][failed_run_id]["actual"]
            )
        event["previous_event_sha256"] = previous
        unsigned_event = dict(event)
        unsigned_event.pop("event_sha256", None)
        event["event_sha256"] = orchestrator.canonical_sha256(unsigned_event)
        previous = event["event_sha256"]
    unsigned_budget = dict(budget)
    unsigned_budget.pop("ledger_sha256", None)
    budget["ledger_sha256"] = orchestrator.canonical_sha256(unsigned_budget)
    budget_path.write_text(
        json.dumps(budget, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = json.loads(original_receipt)
    receipt["scenario_ledger_bindings"][scenario]["budget"].update(
        {
            "file_sha256": orchestrator._file_sha256(budget_path),
            "ledger_sha256": budget["ledger_sha256"],
        }
    )
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = orchestrator.canonical_sha256(unsigned_receipt)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="budget actual usage drifted",
    ):
        orchestrator.verify_v21111_fake_fault_acceptance(
            diagnostics,
            contract=contract,
        )

    budget_path.write_bytes(original_budget)
    receipt_path.write_bytes(original_receipt)
    receipt = json.loads(original_receipt)
    receipt["terminal_commit_window"]["interrupted"] = {"status": "fabricated-accepted"}
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = orchestrator.canonical_sha256(unsigned_receipt)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="receipt denominator drifted",
    ):
        orchestrator.verify_v21111_fake_fault_acceptance(
            diagnostics,
            contract=contract,
        )
