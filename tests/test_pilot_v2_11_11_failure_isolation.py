from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import shutil

import pytest

from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.pilot_contract import load_pilot_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_11.yaml"


class _InjectedProcessCrash(BaseException):
    """Escape ``except Exception`` to model a process death at a commit boundary."""


def _frozen_contract():
    return replace(load_pilot_contract(CONTRACT_PATH), status="frozen")


def _copy_contract(repo: Path) -> Path:
    target = repo / "experiments" / "pilot_v2_11_11.yaml"
    target.parent.mkdir(parents=True)
    shutil.copyfile(CONTRACT_PATH, target)
    return target


def _paid():
    return SimpleNamespace(
        git_tag="pilot-v2.11.11-science",
        head_commit="a" * 40,
        tag_commit="a" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )


def _strict_pass_capability_wrapper(_contract, model_id: str, *, raw_root: Path):
    del raw_root
    return {
        "capability": {
            "model_id": model_id,
            "capability_pass": True,
            "interface_pass": True,
        }
    }


def _parent_complete_ledgers(contract, raw: Path):
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
    projection = orchestrator._v21111_parent_import_projection(parent)
    budget.reserve(projection)
    budget.finalize(
        parent.run_id,
        status="complete",
        cost_usd=0.0,
        completions=0,
        storage_bytes=0,
    )
    return run, budget


def _paid_d_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    scenario: str,
    fail_at_call: int | None,
):
    contract = _frozen_contract()
    raw = tmp_path / scenario
    raw.mkdir()
    orchestrator._bootstrap_v21111_fake_utility(contract, raw_root=raw)
    seed = int(contract.seeds["sets"]["main"][0])
    specs = tuple(
        spec
        for spec in contract.expand(stage="experiment-d")
        if spec.environment_seed == seed
    )
    assert len(specs) == 11
    representative = next(spec for spec in specs if spec.arm_id == "matched-a")
    base = orchestrator.config_for_spec(
        contract,
        representative,
        raw_root=raw,
        paid_provenance=None,
        diagnostic_override=True,
        verify_bound_inputs=False,
    )
    run, budget = _parent_complete_ledgers(contract, raw)
    paid = _paid()
    state = orchestrator.V21111FakeProviderState(fail_at_call=fail_at_call)
    llm = orchestrator.MultiModelLLM(
        orchestrator._V21111AcceptanceProvider(state),
        num_workers=1,
    )

    monkeypatch.setattr(orchestrator, "config_for_spec", lambda *_a, **_k: base)
    monkeypatch.setattr(
        orchestrator,
        "_runner_p95_reservations",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        orchestrator,
        "validate_preflight_p95_reservations",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        orchestrator,
        "_assert_v21111_local_release_guard",
        lambda *_a, **_k: None,
    )
    # The paid-path harness deliberately supplies the offline acceptance
    # adapter. Bind strict checkpoint verification to that adapter's served
    # model name without weakening production provider checks.
    monkeypatch.setattr(
        orchestrator,
        "_runtime_model_for_profile",
        lambda _profile: "diagnostic/scripted-v1",
    )
    provider_constructions = 0

    def fake_provider(*_args, **_kwargs):
        nonlocal provider_constructions
        provider_constructions += 1
        return llm

    monkeypatch.setattr(orchestrator, "_provider_for_profile", fake_provider)

    def split_projections(_contract, group, **_kwargs):
        prefix = orchestrator._v21111_fake_projection(
            representative,
            completions=32,
            run_id_override=(
                f"{contract.contract_id}--experiment-d--gpt52_main--"
                f"checkpoint-prefix--s{seed}"
            ),
        )
        return (
            prefix,
            *(
                orchestrator._v21111_fake_projection(spec, completions=24)
                for spec in group
            ),
        )

    monkeypatch.setattr(
        orchestrator,
        "_v21111_d_split_projections",
        split_projections,
    )
    return (
        contract,
        raw,
        specs,
        run,
        budget,
        paid,
        state,
        lambda: provider_constructions,
    )


def _d_prefix_projection(contract, specs, *, completions: int = 32):
    seed = specs[0].environment_seed
    representative = next(spec for spec in specs if spec.arm_id == "matched-a")
    return orchestrator._v21111_fake_projection(
        representative,
        completions=completions,
        run_id_override=(
            f"{contract.contract_id}--experiment-d--gpt52_main--"
            f"checkpoint-prefix--s{seed}"
        ),
    )


def _stage_paid_d_prefix_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    scenario: str,
):
    fixture = _paid_d_fixture(
        tmp_path,
        monkeypatch,
        scenario=scenario,
        fail_at_call=None,
    )
    contract, raw, specs, run, budget, paid, state, provider_constructions = fixture
    interrupted = orchestrator._execute_v21111_isolated_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        diagnostic=False,
        authority_repo_root=tmp_path,
        interrupt_after_reserve_branch="matched-a",
    )
    assert interrupted == {"status": "interrupted", "active_branch": "matched-a"}
    assert state.calls == 32
    prefix = _d_prefix_projection(contract, specs)
    original_prefix_row = budget.snapshot()["runs"][prefix.run_id]

    # Entry-state tests need a complete prefix with no active branch. Rebuild
    # the sealed coordinator instead of mutating its JSON so any rejection is
    # caused by the deliberate cross-ledger inconsistency under test.
    coordinator = orchestrator.DBranchCoordinator(
        seed=specs[0].environment_seed,
        prefix_status="complete",
    )
    orchestrator._write_v21111_d_coordinator(
        raw
        / "experiment-d"
        / "checkpoints"
        / f"s{specs[0].environment_seed}"
        / "branch_coordinator.json",
        coordinator,
    )
    return (*fixture, prefix, original_prefix_row)


def _replace_budget_with_parent_and_complete_prefix(
    contract,
    raw: Path,
    old_budget,
    prefix,
    original_prefix_row,
):
    old_budget.path.unlink()
    _, budget = _parent_complete_ledgers(contract, raw)
    budget.reserve(prefix)
    actual = original_prefix_row["actual"]
    budget.finalize(
        prefix.run_id,
        status="complete",
        cost_usd=float(actual["cost_usd"]),
        completions=int(actual["completions"]),
        storage_bytes=int(actual["storage_bytes"]),
    )
    return budget


def test_v21111_paid_d_seed_checkpoint_directory_symlink_is_rejected_pre_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract,
        raw,
        specs,
        run,
        budget,
        paid,
        state,
        provider_constructions,
    ) = _paid_d_fixture(
        tmp_path,
        monkeypatch,
        scenario="paid-seed-checkpoint-directory-symlink",
        fail_at_call=None,
    )
    seed = specs[0].environment_seed
    checkpoints = raw / "experiment-d" / "checkpoints"
    checkpoints.mkdir(parents=True)
    outside = tmp_path / "outside-seed-checkpoint"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_bytes(b"must-not-change\n")
    (checkpoints / f"s{seed}").symlink_to(outside, target_is_directory=True)

    run_before = run.snapshot()
    budget_before = budget.snapshot()
    outside_before = {
        path.relative_to(outside).as_posix(): path.read_bytes()
        for path in outside.rglob("*")
        if path.is_file()
    }
    constructions_before = provider_constructions()
    original_provider = orchestrator._provider_for_profile

    def crash_if_provider_is_constructed(*args, **kwargs):
        original_provider(*args, **kwargs)
        raise _InjectedProcessCrash("provider constructed through checkpoint symlink")

    monkeypatch.setattr(
        orchestrator,
        "_provider_for_profile",
        crash_if_provider_is_constructed,
    )
    error: BaseException | None = None
    try:
        orchestrator._execute_v21111_paid_d_seed(
            contract,
            specs,
            raw_root=raw,
            paid=paid,
            budget_ledger=budget,
            run_ledger=run,
            authority_repo_root=tmp_path,
        )
    except BaseException as exc:  # includes the injected process-crash sentinel
        error = exc

    outside_after = {
        path.relative_to(outside).as_posix(): path.read_bytes()
        for path in outside.rglob("*")
        if path.is_file()
    }
    safety = {
        "rejected_as_integrity_error": isinstance(
            error, orchestrator.PilotOrchestrationError
        ),
        "outside_tree_unchanged": outside_after == outside_before,
        "provider_not_constructed": provider_constructions() == constructions_before,
        "prefix_calls_zero": state.calls == 0,
        "run_ledger_unchanged": run.snapshot() == run_before,
        "budget_ledger_unchanged": budget.snapshot() == budget_before,
    }
    assert all(safety.values()), {**safety, "error": repr(error)}


@pytest.mark.parametrize("coordinator_status", ("scheduled", "running"))
@pytest.mark.parametrize("budget_status", ("reserved", "terminal"))
def test_v21111_paid_d_projection_drift_prefix_recovery_rejects_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    coordinator_status: str,
    budget_status: str,
) -> None:
    (
        contract,
        raw,
        specs,
        run,
        budget,
        paid,
        state,
        provider_constructions,
    ) = _paid_d_fixture(
        tmp_path,
        monkeypatch,
        scenario=f"paid-prefix-projection-drift-{coordinator_status}-{budget_status}",
        fail_at_call=None,
    )
    drifted_prefix = _d_prefix_projection(contract, specs, completions=31)
    budget.reserve(drifted_prefix)
    if budget_status == "terminal":
        budget.finalize(
            drifted_prefix.run_id,
            status="failed",
            cost_usd=0.0,
            completions=0,
            storage_bytes=0,
            failure={
                "error_type": "FixtureDriftedPrefixFailure",
                "message": "terminal budget was reserved against a drifted projection",
            },
        )

    coordinator_path = (
        raw
        / "experiment-d"
        / "checkpoints"
        / f"s{specs[0].environment_seed}"
        / "branch_coordinator.json"
    )
    if coordinator_status == "running":
        coordinator = orchestrator.DBranchCoordinator(
            seed=specs[0].environment_seed,
        )
        coordinator.start_prefix()
        orchestrator._write_v21111_d_coordinator(coordinator_path, coordinator)

    run_before = run.snapshot()
    budget_before = budget.snapshot()
    coordinator_before = (
        coordinator_path.read_bytes() if coordinator_path.is_file() else None
    )
    calls_before = state.calls
    constructions_before = provider_constructions()
    with pytest.raises(orchestrator.PilotOrchestrationError):
        orchestrator._execute_v21111_paid_d_seed(
            contract,
            specs,
            raw_root=raw,
            paid=paid,
            budget_ledger=budget,
            run_ledger=run,
            authority_repo_root=tmp_path,
        )

    assert state.calls == calls_before == 0
    assert provider_constructions() == constructions_before == 0
    assert run.snapshot() == run_before
    assert budget.snapshot() == budget_before
    coordinator_after = (
        coordinator_path.read_bytes() if coordinator_path.is_file() else None
    )
    assert coordinator_after == coordinator_before


def test_v21111_refresh_no_go_terminalizes_all_science_without_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _frozen_contract()
    raw = (tmp_path / "raw").absolute()
    raw.mkdir()
    run, budget = _parent_complete_ledgers(contract, raw)
    refresh_path = raw / "dispatch-refresh" / "dispatch_refresh_receipt.json"
    refresh_path.parent.mkdir()
    refresh_path.write_text('{"status":"no-go"}\n', encoding="utf-8")
    refresh_receipt = {
        "status": "no-go",
        "go": False,
        "contract_sha256": contract.canonical_hash,
        "provider_calls_attempted": 1,
        "integrity": {"content_sha256": "c" * 64},
        "authority_ledger_binding": {"ledger_sha256": "d" * 64},
    }
    stage_receipt_calls: list[str] = []

    def write_stage_receipt(_contract, stage_id, **_kwargs):
        stage_receipt_calls.append(stage_id)
        return raw / stage_id / "stage_receipt.json"

    monkeypatch.setattr(orchestrator, "_write_stage_receipt", write_stage_receipt)
    monkeypatch.setattr(
        orchestrator,
        "create_llm_provider",
        lambda *_args, **_kwargs: pytest.fail(
            "refresh no-go terminalization constructed a science provider"
        ),
    )
    result = orchestrator._terminalize_v21111_science_after_refresh_no_go(
        contract,
        raw_root=raw,
        paid=_paid(),
        refresh_receipt=refresh_receipt,
        authority_repo_root=tmp_path,
    )

    reopened = orchestrator.PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    science = tuple(
        spec
        for stage_id in ("experiment-b", "experiment-d", "cross-model")
        for spec in contract.expand(stage=stage_id)
    )
    assert len(science) == 86
    assert Counter(reopened.status(spec.run_id) for spec in science) == {
        "integrity-stopped": 86
    }
    (parent,) = contract.expand(stage="parent-import")
    assert reopened.status(parent.run_id) == "complete"
    assert (
        run.snapshot()["runs"][parent.run_id]
        == reopened.snapshot()["runs"][parent.run_id]
    )
    assert set(budget.snapshot()["runs"]) == {parent.run_id}
    assert result["registered_scientific_cells"] == 86
    assert result["scientific_provider_calls_attempted"] == 0
    assert stage_receipt_calls == ["experiment-b", "experiment-d", "cross-model"]

    ledger_bytes = (raw / "run_ledger.json").read_bytes()
    resumed = orchestrator._terminalize_v21111_science_after_refresh_no_go(
        contract,
        raw_root=raw,
        paid=_paid(),
        refresh_receipt=refresh_receipt,
        authority_repo_root=tmp_path,
    )
    assert resumed["status_counts"] == {"integrity-stopped": 86}
    assert (raw / "run_ledger.json").read_bytes() == ledger_bytes


def test_v21111_paid_d_prefix_failure_terminalizes_exact_eleven_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract,
        raw,
        specs,
        run,
        budget,
        paid,
        state,
        provider_constructions,
    ) = _paid_d_fixture(
        tmp_path,
        monkeypatch,
        scenario="paid-prefix-failure",
        fail_at_call=1,
    )

    result = orchestrator._execute_v21111_paid_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        authority_repo_root=tmp_path,
    )

    assert result == {"status": "prefix-failed", "failed_cells": 11}
    assert Counter(run.status(spec.run_id) for spec in specs) == {"failed": 11}
    assert state.calls == 4
    assert provider_constructions() == 1
    assert not (raw / "experiment-d" / "summaries").exists()


def test_v21111_paid_d_branch_failure_isolated_to_one_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract,
        raw,
        specs,
        run,
        budget,
        paid,
        state,
        provider_constructions,
    ) = _paid_d_fixture(
        tmp_path,
        monkeypatch,
        scenario="paid-branch-failure",
        fail_at_call=33,
    )

    result = orchestrator._execute_v21111_paid_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        authority_repo_root=tmp_path,
    )

    assert result["status"] == "complete"
    assert Counter(run.status(spec.run_id) for spec in specs) == {
        "failed": 1,
        "complete": 10,
    }
    matched_a = next(spec for spec in specs if spec.arm_id == "matched-a")
    assert run.status(matched_a.run_id) == "failed"
    assert state.calls == 276
    assert provider_constructions() == 1
    summaries = raw / "experiment-d" / "summaries"
    assert len(tuple(summaries.glob("*.json"))) == 10
    for path in summaries.glob("*.json"):
        value = __import__("json").loads(path.read_text(encoding="utf-8"))
        assert value["scientific_evidence"] is True
        assert value["payload"]["gate_evidence"]["execution_mode"] == (
            "isolated-branch"
        )


def test_v21111_paid_d_resume_stops_only_inflight_branch_and_runs_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract,
        raw,
        specs,
        run,
        budget,
        paid,
        state,
        provider_constructions,
    ) = _paid_d_fixture(
        tmp_path,
        monkeypatch,
        scenario="paid-branch-resume",
        fail_at_call=None,
    )

    interrupted = orchestrator._execute_v21111_isolated_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        diagnostic=False,
        authority_repo_root=tmp_path,
        interrupt_after_reserve_branch="matched-a",
    )
    assert interrupted == {"status": "interrupted", "active_branch": "matched-a"}
    assert state.calls == 32

    resumed = orchestrator._execute_v21111_paid_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        authority_repo_root=tmp_path,
    )
    assert resumed["status"] == "complete"
    assert Counter(run.status(spec.run_id) for spec in specs) == {
        "integrity-stopped": 1,
        "complete": 10,
    }
    matched_a = next(spec for spec in specs if spec.arm_id == "matched-a")
    assert run.status(matched_a.run_id) == "integrity-stopped"
    assert state.calls == 272
    assert provider_constructions() == 2

    calls_before_replay = state.calls
    provider_constructions_before_replay = provider_constructions()
    run_before_replay = run.snapshot()
    budget_before_replay = budget.snapshot()
    replayed = orchestrator._execute_v21111_paid_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        authority_repo_root=tmp_path,
    )
    assert replayed["status"] == "complete"
    assert state.calls == calls_before_replay
    assert provider_constructions() == provider_constructions_before_replay
    assert run.snapshot() == run_before_replay
    assert budget.snapshot() == budget_before_replay


@pytest.mark.parametrize(
    "prefix_budget_state",
    ("missing", "reserved", "failed", "projection-drift"),
)
def test_v21111_paid_d_complete_prefix_requires_exact_complete_budget_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefix_budget_state: str,
) -> None:
    (
        contract,
        raw,
        specs,
        run,
        old_budget,
        paid,
        state,
        provider_constructions,
        prefix,
        original_prefix_row,
    ) = _stage_paid_d_prefix_complete(
        tmp_path,
        monkeypatch,
        scenario=f"paid-prefix-entry-{prefix_budget_state}",
    )
    old_budget.path.unlink()
    _, budget = _parent_complete_ledgers(contract, raw)
    if prefix_budget_state != "missing":
        reserved = (
            _d_prefix_projection(contract, specs, completions=31)
            if prefix_budget_state == "projection-drift"
            else prefix
        )
        budget.reserve(reserved)
        if prefix_budget_state != "reserved":
            actual = original_prefix_row["actual"]
            budget.finalize(
                reserved.run_id,
                status=("failed" if prefix_budget_state == "failed" else "complete"),
                cost_usd=float(actual["cost_usd"]),
                completions=min(int(actual["completions"]), int(reserved.completions)),
                storage_bytes=min(
                    int(actual["storage_bytes"]), int(reserved.storage_bytes)
                ),
                failure=(
                    {"error_type": "FixturePrefixFailure"}
                    if prefix_budget_state == "failed"
                    else None
                ),
            )

    calls_before_resume = state.calls
    constructions_before_resume = provider_constructions()
    with pytest.raises(orchestrator.PilotOrchestrationError):
        orchestrator._execute_v21111_paid_d_seed(
            contract,
            specs,
            raw_root=raw,
            paid=paid,
            budget_ledger=budget,
            run_ledger=run,
            authority_repo_root=tmp_path,
        )

    assert state.calls == calls_before_resume
    assert provider_constructions() == constructions_before_resume
    assert Counter(run.status(spec.run_id) for spec in specs) == {"scheduled": 11}


@pytest.mark.parametrize(
    "inconsistency",
    (
        "complete-run-missing",
        "complete-artifact-missing",
        "failed-budget-missing",
        "integrity-budget-status-mismatch",
    ),
)
def test_v21111_paid_d_non_scheduled_branch_requires_run_budget_artifact_agreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inconsistency: str,
) -> None:
    (
        contract,
        raw,
        specs,
        run,
        old_budget,
        paid,
        state,
        provider_constructions,
        prefix,
        original_prefix_row,
    ) = _stage_paid_d_prefix_complete(
        tmp_path,
        monkeypatch,
        scenario=f"paid-branch-entry-{inconsistency}",
    )
    budget = _replace_budget_with_parent_and_complete_prefix(
        contract,
        raw,
        old_budget,
        prefix,
        original_prefix_row,
    )
    spec = next(spec for spec in specs if spec.arm_id == "matched-a")
    projection = orchestrator._v21111_fake_projection(spec, completions=24)
    coordinator_status = inconsistency.split("-", maxsplit=1)[0]
    if coordinator_status == "integrity":
        coordinator_status = "integrity-stopped"
    coordinator = orchestrator.DBranchCoordinator(
        seed=spec.environment_seed,
        prefix_status="complete",
    )
    assert coordinator.branch_status is not None
    coordinator.branch_status["matched-a"] = coordinator_status
    orchestrator._write_v21111_d_coordinator(
        raw
        / "experiment-d"
        / "checkpoints"
        / f"s{spec.environment_seed}"
        / "branch_coordinator.json",
        coordinator,
    )

    if inconsistency == "complete-artifact-missing":
        artifact = (
            raw / "experiment-d" / "summaries" / f"{spec.run_id}.json"
        ).absolute()
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text('{"fixture":true}\n', encoding="utf-8")
        budget.reserve(projection)
        budget.finalize(
            spec.run_id,
            status="complete",
            cost_usd=0.0,
            completions=0,
            storage_bytes=0,
        )
        run.finalize(spec.run_id, status="complete", artifact=str(artifact))
        artifact.unlink()
    elif inconsistency == "failed-budget-missing":
        run.finalize(
            spec.run_id,
            status="failed",
            artifact=None,
            failure={"error_type": "FixtureBranchFailure"},
        )
    elif inconsistency == "integrity-budget-status-mismatch":
        run.finalize(
            spec.run_id,
            status="integrity-stopped",
            artifact=None,
            failure={"error_type": "FixtureIntegrityStop"},
        )
        budget.reserve(projection)
        budget.finalize(
            spec.run_id,
            status="failed",
            cost_usd=0.0,
            completions=0,
            storage_bytes=0,
            failure={"error_type": "FixtureBudgetFailure"},
        )

    calls_before_resume = state.calls
    constructions_before_resume = provider_constructions()
    with pytest.raises(orchestrator.PilotOrchestrationError):
        orchestrator._execute_v21111_paid_d_seed(
            contract,
            specs,
            raw_root=raw,
            paid=paid,
            budget_ledger=budget,
            run_ledger=run,
            authority_repo_root=tmp_path,
        )

    assert state.calls == calls_before_resume
    assert provider_constructions() == constructions_before_resume


def test_v21111_paid_d_run_finalize_commit_window_repairs_without_redispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract,
        raw,
        specs,
        run,
        budget,
        paid,
        state,
        provider_constructions,
    ) = _paid_d_fixture(
        tmp_path,
        monkeypatch,
        scenario="paid-run-finalize-commit-window",
        fail_at_call=None,
    )
    matched_a = next(spec for spec in specs if spec.arm_id == "matched-a")

    interrupted = orchestrator._execute_v21111_isolated_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        diagnostic=False,
        authority_repo_root=tmp_path,
        interrupt_after_run_finalize_branch="matched-a",
    )
    assert interrupted == {
        "status": "interrupted-after-run-finalize",
        "active_branch": "matched-a",
    }
    assert run.status(matched_a.run_id) == "complete"
    assert budget.snapshot()["runs"][matched_a.run_id]["status"] == "complete"
    assert state.calls == 56
    matched_row_before_resume = run.snapshot()["runs"][matched_a.run_id]
    matched_artifact = Path(matched_row_before_resume["artifact"])
    matched_artifact_bytes = matched_artifact.read_bytes()

    resumed = orchestrator._execute_v21111_paid_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        authority_repo_root=tmp_path,
    )

    assert resumed["status"] == "complete"
    assert Counter(run.status(spec.run_id) for spec in specs) == {"complete": 11}
    assert state.calls == 296
    assert provider_constructions() == 2
    assert run.snapshot()["runs"][matched_a.run_id] == matched_row_before_resume
    assert matched_artifact.read_bytes() == matched_artifact_bytes

    calls_before_replay = state.calls
    constructions_before_replay = provider_constructions()
    run_before_replay = run.snapshot()
    budget_before_replay = budget.snapshot()
    replayed = orchestrator._execute_v21111_paid_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        authority_repo_root=tmp_path,
    )
    assert replayed["status"] == "complete"
    assert state.calls == calls_before_replay
    assert provider_constructions() == constructions_before_replay
    assert run.snapshot() == run_before_replay
    assert budget.snapshot() == budget_before_replay


def test_v21111_paid_d_prefix_budget_terminal_before_itt_recovers_without_redispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract,
        raw,
        specs,
        run,
        budget,
        paid,
        state,
        provider_constructions,
    ) = _paid_d_fixture(
        tmp_path,
        monkeypatch,
        scenario="paid-prefix-budget-terminal-before-itt",
        fail_at_call=1,
    )
    prefix = _d_prefix_projection(contract, specs)
    coordinator_path = (
        raw
        / "experiment-d"
        / "checkpoints"
        / f"s{specs[0].environment_seed}"
        / "branch_coordinator.json"
    )
    original_finalize_many = run.finalize_many

    def crash_before_itt(_rows):
        raise _InjectedProcessCrash("prefix budget terminal before ITT")

    with monkeypatch.context() as crash:
        crash.setattr(run, "finalize_many", crash_before_itt)
        with pytest.raises(
            _InjectedProcessCrash,
            match="prefix budget terminal before ITT",
        ):
            orchestrator._execute_v21111_paid_d_seed(
                contract,
                specs,
                raw_root=raw,
                paid=paid,
                budget_ledger=budget,
                run_ledger=run,
                authority_repo_root=tmp_path,
            )

    assert run.finalize_many == original_finalize_many
    prefix_row_before_resume = budget.snapshot()["runs"][prefix.run_id]
    assert prefix_row_before_resume["status"] == "failed"
    assert Counter(run.status(spec.run_id) for spec in specs) == {"scheduled": 11}
    coordinator = orchestrator._load_v21111_d_coordinator(
        coordinator_path,
        seed=specs[0].environment_seed,
    )
    assert coordinator.prefix_status == "running"
    calls_before_resume = state.calls
    constructions_before_resume = provider_constructions()

    resumed = orchestrator._execute_v21111_paid_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        authority_repo_root=tmp_path,
    )

    assert resumed["status"] == "prefix-interrupted"
    assert state.calls == calls_before_resume
    assert provider_constructions() == constructions_before_resume
    assert budget.snapshot()["runs"][prefix.run_id] == prefix_row_before_resume
    assert Counter(run.status(spec.run_id) for spec in specs) == {"failed": 11}
    run_rows = run.snapshot()["runs"]
    assert all(run_rows[spec.run_id]["artifact"] is None for spec in specs)
    coordinator = orchestrator._load_v21111_d_coordinator(
        coordinator_path,
        seed=specs[0].environment_seed,
    )
    assert coordinator.prefix_status == "failed"
    assert Counter((coordinator.branch_status or {}).values()) == {"failed-prefix": 11}
    assert not (raw / "experiment-d" / "summaries").exists()

    run_before_replay = run.snapshot()
    budget_before_replay = budget.snapshot()
    coordinator_before_replay = coordinator_path.read_bytes()
    replayed = orchestrator._execute_v21111_paid_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        authority_repo_root=tmp_path,
    )
    assert replayed["status"] == "prefix-interrupted"
    assert state.calls == calls_before_resume
    assert provider_constructions() == constructions_before_resume
    assert run.snapshot() == run_before_replay
    assert budget.snapshot() == budget_before_replay
    assert coordinator_path.read_bytes() == coordinator_before_replay


def test_v21111_paid_d_partial_prefix_itt_failure_must_match_terminal_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract,
        raw,
        specs,
        run,
        budget,
        paid,
        state,
        provider_constructions,
    ) = _paid_d_fixture(
        tmp_path,
        monkeypatch,
        scenario="paid-prefix-partial-itt-failure-mismatch",
        fail_at_call=1,
    )
    prefix = _d_prefix_projection(contract, specs)
    coordinator_path = (
        raw
        / "experiment-d"
        / "checkpoints"
        / f"s{specs[0].environment_seed}"
        / "branch_coordinator.json"
    )

    def crash_after_mismatched_first_itt(rows):
        first = dict(rows[0])
        first_failure = dict(first["failure"])
        first_failure.update(
            {
                "error_type": "InjectedPartialITTMismatch",
                "message": "partial ITT row does not match terminal prefix budget",
            }
        )
        run.finalize(
            first["run_id"],
            status=first["status"],
            artifact=first["artifact"],
            failure=first_failure,
        )
        raise _InjectedProcessCrash("mismatched partial prefix ITT")

    with monkeypatch.context() as crash:
        crash.setattr(run, "finalize_many", crash_after_mismatched_first_itt)
        with pytest.raises(
            _InjectedProcessCrash,
            match="mismatched partial prefix ITT",
        ):
            orchestrator._execute_v21111_paid_d_seed(
                contract,
                specs,
                raw_root=raw,
                paid=paid,
                budget_ledger=budget,
                run_ledger=run,
                authority_repo_root=tmp_path,
            )

    prefix_row = budget.snapshot()["runs"][prefix.run_id]
    assert prefix_row["status"] == "failed"
    partial_rows = run.snapshot()["runs"]
    assert Counter(partial_rows[spec.run_id]["status"] for spec in specs) == {
        "failed": 1,
        "scheduled": 10,
    }
    partial_failure = next(
        partial_rows[spec.run_id]["failure"]
        for spec in specs
        if partial_rows[spec.run_id]["status"] == "failed"
    )
    assert partial_failure["error_type"] == "InjectedPartialITTMismatch"
    assert partial_failure != prefix_row["failure"]
    run_before_resume = run.snapshot()
    budget_before_resume = budget.snapshot()
    coordinator_before_resume = coordinator_path.read_bytes()
    calls_before_resume = state.calls
    constructions_before_resume = provider_constructions()

    with pytest.raises(orchestrator.PilotOrchestrationError):
        orchestrator._execute_v21111_paid_d_seed(
            contract,
            specs,
            raw_root=raw,
            paid=paid,
            budget_ledger=budget,
            run_ledger=run,
            authority_repo_root=tmp_path,
        )

    assert state.calls == calls_before_resume
    assert provider_constructions() == constructions_before_resume
    assert run.snapshot() == run_before_resume
    assert budget.snapshot() == budget_before_resume
    assert coordinator_path.read_bytes() == coordinator_before_resume


def test_v21111_paid_d_stale_prefix_commit_window_recovers_itt_from_coordinator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract,
        raw,
        specs,
        run,
        budget,
        paid,
        state,
        provider_constructions,
    ) = _paid_d_fixture(
        tmp_path,
        monkeypatch,
        scenario="paid-stale-prefix-itt-coordinator-window",
        fail_at_call=None,
    )
    checkpoint_dir = (
        raw / "experiment-d" / "checkpoints" / f"s{specs[0].environment_seed}"
    )
    checkpoint_dir.mkdir(parents=True)
    stale_checkpoint = checkpoint_dir / "checkpoint.json"
    stale_checkpoint.write_text('{"stale":true}\n', encoding="utf-8")
    coordinator_path = checkpoint_dir / "branch_coordinator.json"
    original_write_coordinator = orchestrator._write_v21111_d_coordinator

    def crash_after_failed_coordinator(path, coordinator):
        original_write_coordinator(path, coordinator)
        if coordinator.prefix_status == "failed":
            raise _InjectedProcessCrash("stale prefix coordinator before ITT")

    with monkeypatch.context() as crash:
        crash.setattr(
            orchestrator,
            "_write_v21111_d_coordinator",
            crash_after_failed_coordinator,
        )
        with pytest.raises(
            _InjectedProcessCrash,
            match="stale prefix coordinator before ITT",
        ):
            orchestrator._execute_v21111_paid_d_seed(
                contract,
                specs,
                raw_root=raw,
                paid=paid,
                budget_ledger=budget,
                run_ledger=run,
                authority_repo_root=tmp_path,
            )

    assert Counter(run.status(spec.run_id) for spec in specs) == {"scheduled": 11}
    coordinator = orchestrator._load_v21111_d_coordinator(
        coordinator_path,
        seed=specs[0].environment_seed,
    )
    assert coordinator.prefix_status == "failed"
    assert Counter((coordinator.branch_status or {}).values()) == {"failed-prefix": 11}
    budget_before_resume = budget.snapshot()
    calls_before_resume = state.calls
    constructions_before_resume = provider_constructions()

    resumed = orchestrator._execute_v21111_paid_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        authority_repo_root=tmp_path,
    )

    assert resumed["status"] == "prefix-interrupted"
    assert state.calls == calls_before_resume == 0
    assert provider_constructions() == constructions_before_resume == 0
    assert budget.snapshot() == budget_before_resume
    assert Counter(run.status(spec.run_id) for spec in specs) == {
        "integrity-stopped": 11
    }
    run_rows = run.snapshot()["runs"]
    failures = {run_rows[spec.run_id]["failure"]["error_type"] for spec in specs}
    assert failures == {"PreDispatchTargetNotFresh"}
    assert all(run_rows[spec.run_id]["artifact"] is None for spec in specs)
    assert stale_checkpoint.read_text(encoding="utf-8") == '{"stale":true}\n'
    assert not (raw / "experiment-d" / "summaries").exists()

    run_before_replay = run.snapshot()
    coordinator_before_replay = coordinator_path.read_bytes()
    replayed = orchestrator._execute_v21111_paid_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        authority_repo_root=tmp_path,
    )
    assert replayed["status"] == "prefix-interrupted"
    assert state.calls == 0
    assert provider_constructions() == 0
    assert run.snapshot() == run_before_replay
    assert budget.snapshot() == budget_before_resume
    assert coordinator_path.read_bytes() == coordinator_before_replay


@pytest.mark.parametrize("symlink_parent", ("run-dir", "summaries-parent"))
def test_v21111_paid_d_active_branch_symlink_recovery_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink_parent: str,
) -> None:
    (
        contract,
        raw,
        specs,
        run,
        budget,
        paid,
        state,
        provider_constructions,
    ) = _paid_d_fixture(
        tmp_path,
        monkeypatch,
        scenario=f"paid-active-branch-{symlink_parent}-symlink",
        fail_at_call=None,
    )
    matched_a = next(spec for spec in specs if spec.arm_id == "matched-a")
    interrupted = orchestrator._execute_v21111_isolated_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        diagnostic=False,
        authority_repo_root=tmp_path,
        interrupt_after_reserve_branch="matched-a",
    )
    assert interrupted == {"status": "interrupted", "active_branch": "matched-a"}
    assert state.calls == 32

    outside = tmp_path / f"outside-{symlink_parent}"
    outside.mkdir()
    guarded_paths: tuple[Path, ...]
    if symlink_parent == "run-dir":
        run_dir = raw / "experiment-d" / "runs" / matched_a.run_id
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.symlink_to(outside, target_is_directory=True)
        source = outside / "branch_result.json"
        source.write_text(
            '{"scientific_evidence":true,"outside":"unchanged"}\n',
            encoding="utf-8",
        )
        pending = outside / "pending_non_scientific_summary.json"
        pending.write_text('{"outside":"keep"}\n', encoding="utf-8")
        guarded_paths = (source, pending)
    else:
        summaries = raw / "experiment-d" / "summaries"
        summaries.parent.mkdir(parents=True, exist_ok=True)
        summaries.symlink_to(outside, target_is_directory=True)
        summary = outside / f"{matched_a.run_id}.json"
        summary.write_text(
            '{"scientific_evidence":true,"outside":"unchanged"}\n',
            encoding="utf-8",
        )
        guarded_paths = (summary,)

    def outside_snapshot() -> dict[str, bytes | None]:
        return {
            path.name: path.read_bytes() if path.is_file() else None
            for path in guarded_paths
        }

    outside_before = outside_snapshot()
    run_before_resume = run.snapshot()
    budget_before_resume = budget.snapshot()
    calls_before_resume = state.calls
    constructions_before_resume = provider_constructions()
    rejected = False
    try:
        orchestrator._execute_v21111_paid_d_seed(
            contract,
            specs,
            raw_root=raw,
            paid=paid,
            budget_ledger=budget,
            run_ledger=run,
            authority_repo_root=tmp_path,
        )
    except orchestrator.PilotOrchestrationError:
        rejected = True

    safety = {
        "rejected": rejected,
        "outside_unchanged": outside_snapshot() == outside_before,
        "provider_calls_unchanged": state.calls == calls_before_resume,
        "provider_constructions_unchanged": (
            provider_constructions() == constructions_before_resume
        ),
        "run_ledger_unchanged": run.snapshot() == run_before_resume,
        "budget_ledger_unchanged": budget.snapshot() == budget_before_resume,
    }
    assert all(safety.values()), safety


def test_v21111_paid_d_stale_branch_coordinator_before_itt_recovers_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract,
        raw,
        specs,
        run,
        budget,
        paid,
        state,
        provider_constructions,
    ) = _paid_d_fixture(
        tmp_path,
        monkeypatch,
        scenario="paid-stale-branch-coordinator-itt-window",
        fail_at_call=None,
    )
    matched_a = next(spec for spec in specs if spec.arm_id == "matched-a")
    matched_run_dir = raw / "experiment-d" / "runs" / matched_a.run_id
    matched_run_dir.mkdir(parents=True)
    stale_output = matched_run_dir / "stale-output.json"
    stale_output.write_text('{"stale":true}\n', encoding="utf-8")
    coordinator_path = (
        raw
        / "experiment-d"
        / "checkpoints"
        / f"s{matched_a.environment_seed}"
        / "branch_coordinator.json"
    )
    original_write_coordinator = orchestrator._write_v21111_d_coordinator

    def crash_after_running_branch_coordinator(path, coordinator):
        original_write_coordinator(path, coordinator)
        statuses = coordinator.branch_status or {}
        if (
            coordinator.active_branch == "matched-a"
            and statuses.get("matched-a") == "running"
        ):
            raise _InjectedProcessCrash("stale branch coordinator before ITT")

    with monkeypatch.context() as crash:
        crash.setattr(
            orchestrator,
            "_write_v21111_d_coordinator",
            crash_after_running_branch_coordinator,
        )
        with pytest.raises(
            _InjectedProcessCrash,
            match="stale branch coordinator before ITT",
        ):
            orchestrator._execute_v21111_paid_d_seed(
                contract,
                specs,
                raw_root=raw,
                paid=paid,
                budget_ledger=budget,
                run_ledger=run,
                authority_repo_root=tmp_path,
            )

    assert state.calls == 32
    assert Counter(run.status(spec.run_id) for spec in specs) == {"scheduled": 11}
    assert matched_a.run_id not in budget.snapshot()["runs"]
    coordinator = orchestrator._load_v21111_d_coordinator(
        coordinator_path,
        seed=matched_a.environment_seed,
    )
    assert coordinator.prefix_status == "complete"
    assert coordinator.active_branch == "matched-a"
    assert coordinator.branch_status is not None
    assert coordinator.branch_status["matched-a"] == "running"
    calls_before_resume = state.calls
    constructions_before_resume = provider_constructions()

    resumed = orchestrator._execute_v21111_paid_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        authority_repo_root=tmp_path,
    )

    assert resumed["status"] == "complete"
    assert state.calls - calls_before_resume == 240
    assert provider_constructions() == constructions_before_resume + 1
    assert Counter(run.status(spec.run_id) for spec in specs) == {
        "integrity-stopped": 1,
        "complete": 10,
    }
    matched_row = run.snapshot()["runs"][matched_a.run_id]
    assert matched_row["artifact"] is None
    assert matched_row["failure"]["error_type"] == "PreDispatchTargetNotFresh"
    assert matched_row["failure"]["provider_dispatch_started"] is False
    assert matched_a.run_id not in budget.snapshot()["runs"]
    assert stale_output.read_text(encoding="utf-8") == '{"stale":true}\n'
    coordinator = orchestrator._load_v21111_d_coordinator(
        coordinator_path,
        seed=matched_a.environment_seed,
    )
    assert coordinator.active_branch is None
    assert coordinator.branch_status is not None
    assert coordinator.branch_status["matched-a"] == "integrity-stopped"

    calls_before_replay = state.calls
    constructions_before_replay = provider_constructions()
    run_before_replay = run.snapshot()
    budget_before_replay = budget.snapshot()
    coordinator_before_replay = coordinator_path.read_bytes()
    replayed = orchestrator._execute_v21111_paid_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        authority_repo_root=tmp_path,
    )
    assert replayed["status"] == "complete"
    assert state.calls == calls_before_replay
    assert provider_constructions() == constructions_before_replay
    assert run.snapshot() == run_before_replay
    assert budget.snapshot() == budget_before_replay
    assert coordinator_path.read_bytes() == coordinator_before_replay


def test_v21111_paid_d_published_branch_before_itt_is_demoted_without_redispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract,
        raw,
        specs,
        run,
        budget,
        paid,
        state,
        provider_constructions,
    ) = _paid_d_fixture(
        tmp_path,
        monkeypatch,
        scenario="paid-branch-published-before-itt",
        fail_at_call=None,
    )
    matched_a = next(spec for spec in specs if spec.arm_id == "matched-a")
    projection = orchestrator._v21111_fake_projection(matched_a, completions=24)
    branch_source = (
        raw / "experiment-d" / "runs" / matched_a.run_id / "branch_result.json"
    )
    terminal_summary = raw / "experiment-d" / "summaries" / f"{matched_a.run_id}.json"
    pending_summary = branch_source.parent / "pending_non_scientific_summary.json"
    coordinator_path = (
        raw
        / "experiment-d"
        / "checkpoints"
        / f"s{matched_a.environment_seed}"
        / "branch_coordinator.json"
    )
    original_finalize = run.finalize

    def crash_before_itt(run_id, *args, **kwargs):
        if run_id == matched_a.run_id:
            raise _InjectedProcessCrash("published branch before ITT")
        return original_finalize(run_id, *args, **kwargs)

    with monkeypatch.context() as crash:
        crash.setattr(run, "finalize", crash_before_itt)
        with pytest.raises(
            _InjectedProcessCrash,
            match="published branch before ITT",
        ):
            orchestrator._execute_v21111_paid_d_seed(
                contract,
                specs,
                raw_root=raw,
                paid=paid,
                budget_ledger=budget,
                run_ledger=run,
                authority_repo_root=tmp_path,
            )

    assert run.status(matched_a.run_id) == "scheduled"
    matched_budget_before_resume = budget.snapshot()["runs"][matched_a.run_id]
    assert matched_budget_before_resume["status"] == "complete"
    assert matched_budget_before_resume["reservation"] == projection.to_dict()
    assert (
        json.loads(branch_source.read_text(encoding="utf-8"))["scientific_evidence"]
        is True
    )
    assert (
        json.loads(terminal_summary.read_text(encoding="utf-8"))["scientific_evidence"]
        is True
    )
    assert not pending_summary.exists()
    calls_before_resume = state.calls
    constructions_before_resume = provider_constructions()

    resumed = orchestrator._execute_v21111_paid_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        authority_repo_root=tmp_path,
    )

    assert resumed["status"] == "complete"
    assert state.calls - calls_before_resume == 240
    assert provider_constructions() == constructions_before_resume + 1
    assert Counter(run.status(spec.run_id) for spec in specs) == {
        "integrity-stopped": 1,
        "complete": 10,
    }
    matched_run = run.snapshot()["runs"][matched_a.run_id]
    assert matched_run["artifact"] is None
    assert matched_run["failure"]["error_type"] == "BudgetFinalizedBeforeITT"
    assert budget.snapshot()["runs"][matched_a.run_id] == (matched_budget_before_resume)
    assert (
        json.loads(branch_source.read_text(encoding="utf-8"))["scientific_evidence"]
        is False
    )
    assert not terminal_summary.exists()
    assert not pending_summary.exists()
    assert len(tuple((raw / "experiment-d" / "summaries").glob("*.json"))) == 10
    coordinator = orchestrator._load_v21111_d_coordinator(
        coordinator_path,
        seed=matched_a.environment_seed,
    )
    assert coordinator.branch_status is not None
    assert coordinator.branch_status["matched-a"] == "integrity-stopped"

    calls_before_replay = state.calls
    constructions_before_replay = provider_constructions()
    run_before_replay = run.snapshot()
    budget_before_replay = budget.snapshot()
    branch_source_before_replay = branch_source.read_bytes()
    replayed = orchestrator._execute_v21111_paid_d_seed(
        contract,
        specs,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        run_ledger=run,
        authority_repo_root=tmp_path,
    )
    assert replayed["status"] == "complete"
    assert state.calls == calls_before_replay
    assert provider_constructions() == constructions_before_replay
    assert run.snapshot() == run_before_replay
    assert budget.snapshot() == budget_before_replay
    assert branch_source.read_bytes() == branch_source_before_replay


@pytest.mark.parametrize("drift_kind", ("terminal-artifact", "release"))
def test_v21111_d_stage_revalidates_after_last_branch_before_final_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift_kind: str,
) -> None:
    contract = _frozen_contract()
    repo = tmp_path / f"repo-{drift_kind}"
    contract_path = _copy_contract(repo)
    raw = repo / "experiment_results" / "pilot-v2.11.11" / "raw"
    raw.mkdir(parents=True)
    _parent_complete_ledgers(contract, raw)
    paid = _paid()
    state = {
        "groups": 0,
        "last_group_complete": False,
        "receipt_calls": 0,
        "provider_constructions": 0,
    }

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: paid,
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v21111_dispatch_refresh_go",
        lambda **_kwargs: {"status": "go", "go": True},
    )
    monkeypatch.setattr(
        orchestrator,
        "verified_capability_wrapper_for_v21111",
        _strict_pass_capability_wrapper,
    )
    monkeypatch.setattr(
        orchestrator,
        "_persist_release_attestation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator,
        "_assert_prerequisites",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        orchestrator,
        "validate_live_provider_catalog",
        lambda *_args, **_kwargs: {"rows": [{"profile_id": "gpt52_main"}]},
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_provider_catalog_receipt",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        orchestrator,
        "_remaining_core_projections",
        lambda *_args, **_kwargs: (),
    )

    def local_release_guard(*_args, **_kwargs):
        if drift_kind == "release" and state["last_group_complete"]:
            raise orchestrator.PilotOrchestrationError("fixture release drift")

    monkeypatch.setattr(
        orchestrator,
        "_assert_v21111_local_release_guard",
        local_release_guard,
    )

    def no_provider(*_args, **_kwargs):
        state["provider_constructions"] += 1
        pytest.fail("D final-receipt guard test constructed a provider")

    monkeypatch.setattr(orchestrator, "_provider_for_profile", no_provider)
    monkeypatch.setattr(orchestrator, "create_llm_provider", no_provider)

    def fake_d_executor(
        _contract,
        group,
        *,
        raw_root,
        paid,
        budget_ledger,
        run_ledger,
        authority_repo_root,
    ):
        del paid, budget_ledger, authority_repo_root
        state["groups"] += 1
        last_artifact = None
        for spec in group:
            artifact = (
                raw_root / spec.stage_id / "summaries" / f"{spec.run_id}.json"
            ).absolute()
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(
                json.dumps({"run_id": spec.run_id}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            run_ledger.finalize(
                spec.run_id,
                status="complete",
                artifact=str(artifact),
            )
            last_artifact = artifact
        if state["groups"] == 5:
            state["last_group_complete"] = True
            if drift_kind == "terminal-artifact":
                assert last_artifact is not None
                last_artifact.write_text(
                    '{"tampered_after_last_branch":true}\n',
                    encoding="utf-8",
                )
        return {"status": "complete"}

    monkeypatch.setattr(
        orchestrator,
        "_execute_v21111_paid_d_seed",
        fake_d_executor,
    )

    def forbidden_receipt(*_args, **_kwargs):
        state["receipt_calls"] += 1
        pytest.fail("D stage receipt was written without a final integrity guard")

    monkeypatch.setattr(orchestrator, "_write_stage_receipt", forbidden_receipt)

    expected = (
        "artifact binding drifted"
        if drift_kind == "terminal-artifact"
        else "fixture release drift"
    )
    with pytest.raises(orchestrator.PilotOrchestrationError, match=expected):
        orchestrator._execute_stage_locked(
            contract_path=contract_path,
            stage_id="experiment-d",
            resume=False,
            raw_root=raw,
            repo_root=repo,
        )

    assert state["groups"] == 5
    assert state["last_group_complete"] is True
    assert state["receipt_calls"] == 0
    assert state["provider_constructions"] == 0


def test_v21111_b_projection_failure_terminalizes_only_b_not_d_or_cross(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _frozen_contract()
    repo = tmp_path / "repo"
    contract_path = _copy_contract(repo)
    raw = repo / "experiment_results" / "pilot-v2.11.11" / "raw"
    raw.mkdir(parents=True)
    _parent_complete_ledgers(contract, raw)
    paid = _paid()

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: paid,
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v21111_dispatch_refresh_go",
        lambda **_kwargs: {"status": "go", "go": True},
    )
    monkeypatch.setattr(
        orchestrator,
        "verified_capability_wrapper_for_v21111",
        _strict_pass_capability_wrapper,
    )
    monkeypatch.setattr(
        orchestrator,
        "_persist_release_attestation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator,
        "_assert_prerequisites",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        orchestrator,
        "validate_live_provider_catalog",
        lambda *_args, **_kwargs: {"rows": [{"profile_id": "gpt52_main"}]},
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_provider_catalog_receipt",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        orchestrator,
        "_write_stage_receipt",
        lambda *_args, **_kwargs: raw / "experiment-b" / "stage_receipt.json",
    )
    monkeypatch.setattr(
        orchestrator,
        "projection_from_preflight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            orchestrator.PilotBudgetError("fixture B projection no-go")
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_provider_for_profile",
        lambda *_args, **_kwargs: pytest.fail(
            "provider constructed after a pre-dispatch projection failure"
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "create_llm_provider",
        lambda *_args, **_kwargs: pytest.fail(
            "provider constructed after a pre-dispatch projection failure"
        ),
    )

    original_remaining = orchestrator._remaining_core_projections
    requested_stage_sets: list[tuple[str, ...] | None] = []

    def capture_remaining(*args, **kwargs):
        stage_ids = kwargs.get("stage_ids")
        requested_stage_sets.append(None if stage_ids is None else tuple(stage_ids))
        return original_remaining(*args, **kwargs)

    monkeypatch.setattr(
        orchestrator,
        "_remaining_core_projections",
        capture_remaining,
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="full stage projection failed before dispatch",
    ):
        orchestrator._execute_stage_locked(
            contract_path=contract_path,
            stage_id="experiment-b",
            resume=False,
            raw_root=raw,
            repo_root=repo,
        )

    ledger = orchestrator.PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    assert requested_stage_sets == [("experiment-b",)]
    assert Counter(
        ledger.status(spec.run_id) for spec in contract.expand(stage="experiment-b")
    ) == {"budget-stopped": 25}
    assert Counter(
        ledger.status(spec.run_id) for spec in contract.expand(stage="experiment-d")
    ) == {"scheduled": 55}
    assert Counter(
        ledger.status(spec.run_id) for spec in contract.expand(stage="cross-model")
    ) == {"scheduled": 6}
    (parent,) = contract.expand(stage="parent-import")
    assert ledger.status(parent.run_id) == "complete"
