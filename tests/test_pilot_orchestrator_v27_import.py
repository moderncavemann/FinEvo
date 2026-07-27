from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verified_memory import pilot_evidence as evidence
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory import pilot_v27_stage0_import as v27_import
from verified_memory.pilot_contract import canonical_sha256, load_pilot_contract
from verified_memory.runner import VerifiedRunResult


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_7.yaml"


def _paid() -> SimpleNamespace:
    return SimpleNamespace(
        git_tag="pilot-v2.7-science",
        head_commit="a" * 40,
    )


def test_v27_runtime_uses_parent_import_controls_and_debit() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)

    assert contract.contract_id in orchestrator.PARENT_IMPORT_CONTRACT_IDS
    assert orchestrator._materializes_legacy_amendment_controls(contract) is False
    assert orchestrator._cross_model_science_stage_ids(contract) == ()
    assert orchestrator._parent_budget_debit(contract) is not None


def test_v27_parent_qref_and_stage0_routes_construct_no_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid()
    parent_root = tmp_path / "parent"
    parent_root.mkdir()
    routed: list[str] = []

    class Ledger:
        def register(self, _specs: Any) -> None:
            return None

        def status(self, _run_id: str) -> str:
            return "scheduled"

    class Budget:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("V2.7 import stages must not construct a provider")

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
    monkeypatch.setattr(orchestrator, "PilotRunLedger", lambda *_a, **_k: Ledger())
    monkeypatch.setattr(orchestrator, "PilotBudgetLedger", Budget)
    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden_provider)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden_provider)

    def parent_route(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["budget_ledger"] is not None
        routed.append("parent-import")
        return {"status": "complete"}

    def qref_route(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        routed.append("q-ref-resolution")
        return {"status": "complete"}

    def stage0_route(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["budget_ledger"] is not None
        routed.append("stage0-calibration")
        return {"status": "complete"}

    monkeypatch.setattr(
        orchestrator, "_execute_v24_parent_import_stage", parent_route
    )
    monkeypatch.setattr(
        orchestrator, "_execute_v27_q_ref_import_stage", qref_route
    )
    monkeypatch.setattr(
        orchestrator, "_execute_v27_stage0_import_stage", stage0_route
    )

    for stage_id in (
        "parent-import",
        "q-ref-resolution",
        "stage0-calibration",
    ):
        result = orchestrator._execute_stage_locked(
            contract_path=CONTRACT_PATH,
            stage_id=stage_id,
            resume=True,
            raw_root=tmp_path / "experiment_results" / "pilot-v2.7" / "raw",
            repo_root=tmp_path,
            parent_repo_root=parent_root if stage_id == "parent-import" else None,
        )
        assert result["status"] == "complete"

    assert routed == [
        "parent-import",
        "q-ref-resolution",
        "stage0-calibration",
    ]


def test_v27_stage0_reader_decodes_only_frozen_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="stage0-calibration")[0]
    records = {
        "actions": tuple({"row": index} for index in range(48)),
        "utility_ledger": tuple({"row": index} for index in range(48)),
        "errors": (),
        "macro_steps": ({"forbidden": True},),
    }
    result = VerifiedRunResult(
        config={"seed": spec.environment_seed, "max_labor_hours": 168.0},
        summary={},
        validation_status={"status": "pass"},
        budget_snapshot={},
        records=records,
    )
    observed: list[set[str]] = []

    monkeypatch.setattr(
        orchestrator,
        "_load_v27_imported_result",
        lambda *_a, **_k: (
            result,
            {
                "source_run_id": "source-run",
                "source_spec": {"run_id": "source-run"},
                "source_artifacts": {
                    "manifest": {"file_sha256": "b" * 64},
                    "actor_journal": {"file_sha256": "c" * 64},
                },
            },
            tmp_path / "source-run",
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_v27_source_binding_payload",
        lambda **_k: {
            "parent_import_receipt": {"content_sha256": "d" * 64},
            "source_artifacts": {
                "manifest": {"file_sha256": "b" * 64}
            },
        },
    )

    def summarize(rows: Any, **_kwargs: Any) -> dict[str, Any]:
        observed.append(set(rows))
        return {"gate": "pass"}

    monkeypatch.setattr(orchestrator, "summarize_stage0_run", summarize)
    monkeypatch.setattr(
        orchestrator,
        "summarize_run",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("shock/recovery reader must not run for Stage0")
        ),
    )

    envelope, *_ = orchestrator._build_v27_stage0_envelope(
        contract,
        spec,
        raw_root=tmp_path,
        paid=_paid(),
        source_manifest={"integrity": {"content_sha256": "e" * 64}},
    )

    assert observed == [{"actions", "utility_ledger", "errors"}]
    assert envelope["reader"]["function"] == "summarize_stage0_run"
    assert (
        envelope["execution_disposition"]
        == "immutable-parent-import-offline-resummary"
    )
    assert envelope["provider_calls_current_attempt"] == 0


def test_v27_qref_rejects_source_environment_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    source_qref = tmp_path / "source_q_ref_resolution.json"
    source_qref.write_text(
        json.dumps(
            {
                "q_ref": 1.0,
                "status": "pass",
                "scientific_evidence": False,
                "bindings": {
                    "environment_source_hash": "0" * 64,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    source_binding = {
        "source_run_id": "v2.6-qref",
        "source_artifacts": {
            "q_ref_resolution": {
                "path": (
                    "experiment_results/pilot-v2.6/raw/"
                    "q-ref-resolution/q_ref_resolution.json"
                ),
                "file_sha256": evidence._sha256_file(source_qref),
            }
        },
    }
    monkeypatch.setattr(
        orchestrator,
        "_v27_import_authority",
        lambda *_args, **_kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_v27_imported_result",
        lambda *_args, **_kwargs: (
            object(),
            source_binding,
            tmp_path / "source-run",
        ),
    )
    expected_environment_hash = orchestrator._file_sha256(
        orchestrator.DEFAULT_ENV_CONFIG
    )
    monkeypatch.setattr(
        orchestrator,
        "build_q_ref_resolution",
        lambda *_args, **_kwargs: {
            "schema_version": "finevo-q-ref-resolution-v1",
            "q_ref": 1.0,
            "status": "pass",
            "bindings": {
                "environment_source_hash": expected_environment_hash,
            },
        },
    )
    monkeypatch.setattr(
        v27_import,
        "snapshot_path_for_source_artifact",
        lambda *_args, **_kwargs: source_qref,
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="q-ref recomputation differs",
    ):
        orchestrator._expected_v27_q_ref_resolution(
            contract,
            raw_root=tmp_path,
            paid=_paid(),
        )


@pytest.mark.parametrize("failure_mode", ["receipt", "budget"])
def test_v27_qref_no_go_resume_preserves_failure_and_reconciles_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    run_ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    run_ledger.register(contract.expand())
    parent_spec = contract.expand(stage="parent-import")[0]
    parent_artifact = raw_root / "parent-import" / "parent.json"
    parent_artifact.parent.mkdir()
    parent_artifact.write_text("{}\n", encoding="utf-8")
    run_ledger.finalize(
        parent_spec.run_id,
        status="complete",
        artifact=str(parent_artifact),
        failure=None,
    )
    spec = contract.expand(stage="q-ref-resolution")[0]

    class Budget:
        def __init__(self) -> None:
            self.rows: dict[str, dict[str, Any]] = {}
            self.fail_once = failure_mode == "budget"

        def snapshot(self) -> dict[str, Any]:
            return {"runs": self.rows}

        def reserve(self, projection: Any) -> None:
            self.rows[projection.run_id] = {
                "reservation": projection.to_dict(),
                "status": "reserved",
                "actual": None,
                "failure": None,
            }

        def finalize(self, run_id: str, **kwargs: Any) -> None:
            if self.fail_once:
                self.fail_once = False
                raise OSError("injected q-ref budget finalization failure")
            self.rows[run_id].update(
                {
                    "status": kwargs["status"],
                    "actual": {
                        "cost_usd": kwargs["cost_usd"],
                        "completions": kwargs["completions"],
                        "storage_bytes": kwargs["storage_bytes"],
                    },
                    "failure": kwargs["failure"],
                }
            )

    budget = Budget()
    monkeypatch.setattr(
        orchestrator,
        "_assert_prerequisites",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        orchestrator,
        "_expected_v27_q_ref_resolution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected q-ref import verification failure")
        ),
    )
    original_write_receipt = orchestrator._write_stage_receipt
    receipt_failed = False

    def write_receipt(*args: Any, **kwargs: Any) -> Path:
        nonlocal receipt_failed
        if (
            failure_mode == "receipt"
            and not receipt_failed
            and len(args) > 1
            and args[1] == "q-ref-resolution"
        ):
            receipt_failed = True
            raise OSError("injected q-ref receipt publication failure")
        return original_write_receipt(*args, **kwargs)

    monkeypatch.setattr(
        orchestrator,
        "_write_stage_receipt",
        write_receipt,
    )

    with pytest.raises(
        OSError,
        match="injected q-ref (budget finalization|receipt publication) failure",
    ):
        orchestrator._execute_v27_q_ref_import_stage(
            contract,
            (spec,),
            raw_root=raw_root,
            paid=_paid(),
            run_ledger=run_ledger,
            budget_ledger=budget,
        )
    original_failure = run_ledger.snapshot()["runs"][spec.run_id]["failure"]
    assert run_ledger.status(spec.run_id) == "integrity-stopped"
    assert all(
        run_ledger.is_terminal(candidate.run_id)
        for candidate in contract.expand()
    )
    assert budget.rows[spec.run_id]["status"] == "reserved"

    reconciled = orchestrator._execute_v27_q_ref_import_stage(
        contract,
        (spec,),
        raw_root=raw_root,
        paid=_paid(),
        run_ledger=run_ledger,
        budget_ledger=budget,
    )
    assert reconciled["status"] == "integrity-stopped"
    assert run_ledger.snapshot()["runs"][spec.run_id]["failure"] == (
        original_failure
    )
    assert budget.rows[spec.run_id]["status"] == "integrity-stopped"
    assert budget.rows[spec.run_id]["failure"] == original_failure


def test_v27_upstream_no_go_does_not_fabricate_qref_or_stage0_budget_rows(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    run_ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    run_ledger.register(contract.expand())
    parent_spec = contract.expand(stage="parent-import")[0]
    failure = {
        "kind": "upstream-parent-no-go",
        "message": "test-only propagated no-go",
    }
    run_ledger.finalize(
        parent_spec.run_id,
        status="integrity-stopped",
        artifact=None,
        failure=failure,
    )
    orchestrator._propagate_stage_no_go(
        contract,
        source_stage="parent-import",
        ledger=run_ledger,
        failure=failure,
    )

    class Budget:
        def __init__(self) -> None:
            self.rows: dict[str, dict[str, Any]] = {}

        def snapshot(self) -> dict[str, Any]:
            return {"runs": self.rows}

        def reserve(self, projection: Any) -> None:
            self.rows[projection.run_id] = {
                "reservation": projection.to_dict(),
                "status": "reserved",
            }

    budget = Budget()
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="upstream no-go has no dispatch authority",
    ):
        orchestrator._execute_v27_q_ref_import_stage(
            contract,
            contract.expand(stage="q-ref-resolution"),
            raw_root=raw_root,
            paid=_paid(),
            run_ledger=run_ledger,
            budget_ledger=budget,
        )
    assert budget.rows == {}

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="upstream no-go has no dispatch authority",
    ):
        orchestrator._execute_v27_stage0_import_stage(
            contract,
            contract.expand(stage="stage0-calibration"),
            raw_root=raw_root,
            paid=_paid(),
            run_ledger=run_ledger,
            budget_ledger=budget,
        )
    assert budget.rows == {}


def test_v27_stage0_no_go_resume_rebuilds_missing_receipt_and_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    run_ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    run_ledger.register(contract.expand())
    for stage_id in ("parent-import", "q-ref-resolution"):
        spec = contract.expand(stage=stage_id)[0]
        artifact = raw_root / stage_id / f"{spec.run_id}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}\n", encoding="utf-8")
        run_ledger.finalize(
            spec.run_id,
            status="complete",
            artifact=str(artifact),
            failure=None,
        )

    class Budget:
        def __init__(self) -> None:
            self.rows: dict[str, dict[str, Any]] = {}

        def snapshot(self) -> dict[str, Any]:
            return {"runs": self.rows}

        def reserve(self, projection: Any) -> None:
            self.rows[projection.run_id] = {
                "reservation": projection.to_dict(),
                "status": "reserved",
                "actual": None,
                "failure": None,
            }

        def finalize(self, run_id: str, **kwargs: Any) -> None:
            self.rows[run_id].update(
                {
                    "status": kwargs["status"],
                    "actual": {
                        "cost_usd": kwargs["cost_usd"],
                        "completions": kwargs["completions"],
                        "storage_bytes": kwargs["storage_bytes"],
                    },
                    "failure": kwargs["failure"],
                }
            )

    budget = Budget()
    monkeypatch.setattr(
        orchestrator,
        "_assert_prerequisites",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected Stage-0 prerequisite verification failure")
        ),
    )
    original_write_receipt = orchestrator._write_stage_receipt
    failed = False

    def write_receipt(*args: Any, **kwargs: Any) -> Path:
        nonlocal failed
        if not failed and len(args) > 1 and args[1] == "stage0-calibration":
            failed = True
            raise OSError("injected Stage-0 no-go receipt publication failure")
        return original_write_receipt(*args, **kwargs)

    monkeypatch.setattr(
        orchestrator,
        "_write_stage_receipt",
        write_receipt,
    )
    specs = contract.expand(stage="stage0-calibration")
    with pytest.raises(
        OSError,
        match="injected Stage-0 no-go receipt publication failure",
    ):
        orchestrator._execute_v27_stage0_import_stage(
            contract,
            specs,
            raw_root=raw_root,
            paid=_paid(),
            run_ledger=run_ledger,
            budget_ledger=budget,
        )
    assert {
        run_ledger.status(spec.run_id) for spec in specs
    } == {"integrity-stopped"}
    assert {
        row["status"] for row in budget.rows.values()
    } == {"reserved"}

    reconciled = orchestrator._execute_v27_stage0_import_stage(
        contract,
        specs,
        raw_root=raw_root,
        paid=_paid(),
        run_ledger=run_ledger,
        budget_ledger=budget,
    )
    assert reconciled["status"] == "integrity-stopped"
    assert {
        row["status"] for row in budget.rows.values()
    } == {"integrity-stopped"}


@pytest.mark.parametrize("failure_mode", ["receipt", "budget"])
def test_v27_stage0_partial_publication_is_auditable_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw_root = tmp_path / "raw"
    ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())
    for stage_id in ("parent-import", "q-ref-resolution"):
        spec = contract.expand(stage=stage_id)[0]
        artifact = raw_root / stage_id / f"{spec.run_id}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}\n", encoding="utf-8")
        ledger.finalize(
            spec.run_id,
            status="complete",
            artifact=str(artifact),
            failure=None,
        )

    cells = []
    for spec in contract.expand(stage="stage0-calibration"):
        envelope = raw_root / spec.stage_id / "runs" / spec.run_id / "envelope.json"
        terminal = raw_root / spec.stage_id / "summaries" / f"{spec.run_id}.json"
        envelope.parent.mkdir(parents=True, exist_ok=True)
        terminal.parent.mkdir(parents=True, exist_ok=True)
        envelope.write_text("{}\n", encoding="utf-8")
        terminal.write_text("{}\n", encoding="utf-8")
        cells.append((spec, envelope, terminal, {"gate": "pass"}))
    selection = {
        "selected_profile_id": "nu-0.5",
        "integrity": {"content_sha256": "f" * 64},
    }
    receipt_path = raw_root / "stage0-calibration" / "stage_receipt.json"

    monkeypatch.setattr(
        orchestrator, "_assert_prerequisites", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        orchestrator,
        "_expected_v27_stage0_selection",
        lambda *_a, **_k: (selection, tuple(cells)),
    )
    monkeypatch.setattr(
        orchestrator, "_remaining_core_projections", lambda *_a, **_k: ()
    )
    monkeypatch.setattr(
        orchestrator, "_assert_projection_matrix_fits", lambda *_a, **_k: None
    )

    receipt_attempts = 0

    def write_receipt(*_args: Any, **_kwargs: Any) -> Path:
        nonlocal receipt_attempts
        receipt_attempts += 1
        if failure_mode == "receipt" and receipt_attempts == 1:
            raise OSError("injected Stage-0 receipt publication failure")
        receipt_path.write_text(
            json.dumps({"status": "complete"}) + "\n",
            encoding="utf-8",
        )
        return receipt_path

    monkeypatch.setattr(orchestrator, "_write_stage_receipt", write_receipt)

    class Budget:
        def __init__(self) -> None:
            self.rows: dict[str, dict[str, Any]] = {}
            self.finalize_calls = 0
            self.injected = False

        def snapshot(self) -> dict[str, Any]:
            return {"runs": self.rows}

        def reserve(self, projection: Any) -> None:
            self.rows[projection.run_id] = {
                "reservation": projection.to_dict(),
                "status": "reserved",
            }

        def finalize(self, run_id: str, **_kwargs: Any) -> None:
            self.finalize_calls += 1
            if (
                failure_mode == "budget"
                and not self.injected
                and self.finalize_calls == 5
            ):
                self.injected = True
                raise OSError(
                    "injected Stage-0 budget finalization failure"
                )
            self.rows[run_id]["status"] = "complete"

    budget = Budget()
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="requires audited --resume",
    ):
        orchestrator._execute_v27_stage0_import_stage(
            contract,
            contract.expand(stage="stage0-calibration"),
            raw_root=raw_root,
            paid=_paid(),
            run_ledger=ledger,
            budget_ledger=budget,
        )
    interim_statuses: dict[str, int] = {}
    for row in ledger.snapshot()["runs"].values():
        interim_statuses[row["status"]] = (
            interim_statuses.get(row["status"], 0) + 1
        )
    assert interim_statuses == {"complete": 16, "scheduled": 195}
    assert (
        raw_root
        / "stage0-calibration"
        / "stage0_commit_intent.json"
    ).is_file()
    interim_budget_statuses = {
        row["status"] for row in budget.snapshot()["runs"].values()
    }
    assert "reserved" in interim_budget_statuses

    result = orchestrator._execute_v27_stage0_import_stage(
        contract,
        contract.expand(stage="stage0-calibration"),
        raw_root=raw_root,
        paid=_paid(),
        run_ledger=ledger,
        budget_ledger=budget,
    )

    statuses: dict[str, int] = {}
    for row in ledger.snapshot()["runs"].values():
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    assert result["status"] == "complete"
    assert statuses == {"complete": 16, "scheduled": 195}


def test_v27_parent_import_accounts_incremental_snapshot_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="parent-import")[0]
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    class RunLedger:
        terminal = False

        def is_terminal(self, _run_id: str) -> bool:
            return self.terminal

        def status(self, _run_id: str) -> str:
            return "complete" if self.terminal else "scheduled"

        def finalize(self, *_args: Any, **_kwargs: Any) -> None:
            self.terminal = True

    class BudgetLedger:
        def __init__(self) -> None:
            self.row: dict[str, Any] | None = None
            self.finalized: dict[str, Any] | None = None

        def snapshot(self) -> dict[str, Any]:
            return {"runs": {} if self.row is None else {spec.run_id: self.row}}

        def reserve(self, projection: Any) -> None:
            self.row = {
                "reservation": projection.to_dict(),
                "status": "reserved",
            }

        def finalize(self, _run_id: str, **kwargs: Any) -> None:
            self.finalized = kwargs

    budget = BudgetLedger()
    materializations: list[str] = []
    verifications: list[str] = []

    def persist(**_kwargs: Any) -> dict[str, Any]:
        materializations.append("importer")
        return {
            "resealed_p95_profiles": {
                "sentinel": "materialized-by-importer",
            },
            "provider_calls": 0,
            "scientific_evidence": False,
        }

    monkeypatch.setattr(
        orchestrator,
        "persist_v27_parent_import",
        persist,
    )

    def verify(
        _contract: Any,
        *,
        importer_result: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        verifications.append("read-only")
        return importer_result["resealed_p95_profiles"]

    monkeypatch.setattr(
        orchestrator,
        "_verify_v27_importer_p95_profiles",
        verify,
    )

    def terminal(path: Path, **_kwargs: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return path

    receipt_path = raw_root / "parent-import" / "stage_receipt.json"

    def receipt(*_args: Any, **_kwargs: Any) -> Path:
        receipt_path.write_text(
            json.dumps({"status": "complete"}) + "\n",
            encoding="utf-8",
        )
        return receipt_path

    monkeypatch.setattr(orchestrator, "write_terminal_summary", terminal)
    monkeypatch.setattr(orchestrator, "_write_stage_receipt", receipt)

    result = orchestrator._execute_v24_parent_import_stage(
        contract,
        (spec,),
        raw_root=raw_root,
        repo_root=tmp_path,
        parent_repo_root=tmp_path,
        paid=_paid(),
        run_ledger=RunLedger(),
        budget_ledger=budget,
    )

    assert result["status"] == "complete"
    assert budget.row is not None
    assert budget.row["reservation"]["storage_bytes"] > 12_877_797
    assert budget.finalized is not None
    assert budget.finalized["cost_usd"] == 0
    assert budget.finalized["completions"] == 0
    assert budget.finalized["storage_bytes"] > 0
    assert materializations == ["importer"]
    assert verifications == ["read-only"]


@pytest.mark.parametrize("failure_mode", ["receipt", "budget"])
def test_v27_parent_post_commit_failure_resumes_without_no_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="parent-import")[0]
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    run_ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    run_ledger.register(contract.expand())

    class Budget:
        def __init__(self) -> None:
            self.rows: dict[str, dict[str, Any]] = {}
            self.fail_once = failure_mode == "budget"

        def snapshot(self) -> dict[str, Any]:
            return {"runs": self.rows}

        def reserve(self, projection: Any) -> None:
            self.rows[projection.run_id] = {
                "reservation": projection.to_dict(),
                "status": "reserved",
                "actual": None,
                "failure": None,
            }

        def finalize(self, run_id: str, **kwargs: Any) -> None:
            if self.fail_once:
                self.fail_once = False
                raise OSError(
                    "injected parent success budget finalization failure"
                )
            self.rows[run_id].update(
                {
                    "status": kwargs["status"],
                    "actual": {
                        "cost_usd": kwargs["cost_usd"],
                        "completions": kwargs["completions"],
                        "storage_bytes": kwargs["storage_bytes"],
                    },
                    "failure": kwargs["failure"],
                }
            )

    budget = Budget()
    importer_calls: list[str] = []

    def persist(**_kwargs: Any) -> dict[str, Any]:
        importer_calls.append("materialized")
        return {
            "receipt": str(
                raw_root / "parent-import" / "parent_import_receipt.json"
            ),
            "snapshot_inventory_sha256": "a" * 64,
            "resealed_p95_profiles": {
                "sentinel": "importer-materialized",
            },
            "provider_calls": 0,
            "scientific_evidence": False,
        }

    monkeypatch.setattr(
        orchestrator,
        "persist_v27_parent_import",
        persist,
    )
    monkeypatch.setattr(
        orchestrator,
        "_verify_v27_importer_p95_profiles",
        lambda _contract, *, importer_result, **_kwargs: importer_result[
            "resealed_p95_profiles"
        ],
    )
    receipt_failed = False

    def write_receipt(*args: Any, **kwargs: Any) -> Path:
        nonlocal receipt_failed
        if (
            failure_mode == "receipt"
            and not receipt_failed
            and len(args) > 1
            and args[1] == "parent-import"
        ):
            receipt_failed = True
            raise OSError(
                "injected parent success receipt publication failure"
            )
        path = raw_root / "parent-import" / "stage_receipt.json"
        path.write_text(
            json.dumps({"status": "complete"}) + "\n",
            encoding="utf-8",
        )
        return path

    monkeypatch.setattr(
        orchestrator,
        "_write_stage_receipt",
        write_receipt,
    )
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="requires audited --resume",
    ):
        orchestrator._execute_v24_parent_import_stage(
            contract,
            (spec,),
            raw_root=raw_root,
            repo_root=tmp_path,
            parent_repo_root=tmp_path,
            paid=_paid(),
            run_ledger=run_ledger,
            budget_ledger=budget,
        )
    statuses: dict[str, int] = {}
    for row in run_ledger.snapshot()["runs"].values():
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    assert statuses == {"complete": 1, "scheduled": 210}
    assert budget.rows[spec.run_id]["status"] == "reserved"
    assert not (
        raw_root / "parent-import" / "failure_receipt.json"
    ).exists()
    assert (
        raw_root
        / "parent-import"
        / "parent_import_commit_intent.json"
    ).is_file()

    monkeypatch.setattr(
        orchestrator,
        "persist_v27_parent_import",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected parent resume verification failure")
        ),
    )
    with pytest.raises(
        RuntimeError,
        match="injected parent resume verification failure",
    ):
        orchestrator._execute_v24_parent_import_stage(
            contract,
            (spec,),
            raw_root=raw_root,
            repo_root=tmp_path,
            parent_repo_root=tmp_path,
            paid=_paid(),
            run_ledger=run_ledger,
            budget_ledger=budget,
        )
    assert {
        row["status"]
        for run_id, row in run_ledger.snapshot()["runs"].items()
        if run_id != spec.run_id
    } == {"scheduled"}
    assert run_ledger.status(spec.run_id) == "complete"
    assert budget.rows[spec.run_id]["status"] == "reserved"
    assert not (
        raw_root / "parent-import" / "failure_receipt.json"
    ).exists()
    monkeypatch.setattr(
        orchestrator,
        "persist_v27_parent_import",
        persist,
    )

    resumed = orchestrator._execute_v24_parent_import_stage(
        contract,
        (spec,),
        raw_root=raw_root,
        repo_root=tmp_path,
        parent_repo_root=tmp_path,
        paid=_paid(),
        run_ledger=run_ledger,
        budget_ledger=budget,
    )
    assert resumed["status"] == "complete"
    assert budget.rows[spec.run_id]["status"] == "complete"
    assert budget.rows[spec.run_id]["failure"] is None
    assert importer_calls == ["materialized", "materialized"]


@pytest.mark.parametrize("failure_mode", ["receipt", "budget"])
def test_v27_parent_no_go_resume_reconciles_reserved_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="parent-import")[0]
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    run_ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    run_ledger.register(contract.expand())

    class Budget:
        def __init__(self) -> None:
            self.rows: dict[str, dict[str, Any]] = {}
            self.fail_once = failure_mode == "budget"

        def snapshot(self) -> dict[str, Any]:
            return {"runs": self.rows}

        def reserve(self, projection: Any) -> None:
            self.rows[projection.run_id] = {
                "reservation": projection.to_dict(),
                "status": "reserved",
                "actual": None,
                "failure": None,
            }

        def finalize(self, run_id: str, **kwargs: Any) -> None:
            if self.fail_once:
                self.fail_once = False
                raise OSError("injected parent budget finalization failure")
            self.rows[run_id].update(
                {
                    "status": kwargs["status"],
                    "actual": {
                        "cost_usd": kwargs["cost_usd"],
                        "completions": kwargs["completions"],
                        "storage_bytes": kwargs["storage_bytes"],
                    },
                    "failure": kwargs["failure"],
                }
            )

    budget = Budget()
    monkeypatch.setattr(
        orchestrator,
        "persist_v27_parent_import",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected immutable parent verification failure")
        ),
    )
    original_write_receipt = orchestrator._write_stage_receipt
    receipt_failed = False

    def write_receipt(*args: Any, **kwargs: Any) -> Path:
        nonlocal receipt_failed
        if (
            failure_mode == "receipt"
            and not receipt_failed
            and len(args) > 1
            and args[1] == "parent-import"
        ):
            receipt_failed = True
            raise OSError("injected parent receipt publication failure")
        return original_write_receipt(*args, **kwargs)

    monkeypatch.setattr(
        orchestrator,
        "_write_stage_receipt",
        write_receipt,
    )

    with pytest.raises(
        OSError,
        match="injected parent (budget finalization|receipt publication) failure",
    ):
        orchestrator._execute_v24_parent_import_stage(
            contract,
            (spec,),
            raw_root=raw_root,
            repo_root=tmp_path,
            parent_repo_root=tmp_path,
            paid=_paid(),
            run_ledger=run_ledger,
            budget_ledger=budget,
        )
    assert run_ledger.status(spec.run_id) == "integrity-stopped"
    assert all(
        run_ledger.is_terminal(candidate.run_id)
        for candidate in contract.expand()
    )
    assert budget.rows[spec.run_id]["status"] == "reserved"
    assert (
        raw_root / "parent-import" / "stage_receipt.json"
    ).is_file() is (failure_mode == "budget")

    reconciled = orchestrator._execute_v24_parent_import_stage(
        contract,
        (spec,),
        raw_root=raw_root,
        repo_root=tmp_path,
        parent_repo_root=tmp_path,
        paid=_paid(),
        run_ledger=run_ledger,
        budget_ledger=budget,
    )
    assert reconciled["status"] == "integrity-stopped"
    assert budget.rows[spec.run_id]["status"] == "integrity-stopped"
    assert budget.rows[spec.run_id]["actual"]["cost_usd"] == 0.0
    assert budget.rows[spec.run_id]["actual"]["completions"] == 0


def test_v27_release_controls_validate_imported_stage0_envelope_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw_root = tmp_path / "raw"
    stage_root = raw_root / "stage0-calibration"
    stage_root.mkdir(parents=True)
    commit = "a" * 40
    source_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    verifier_calls: list[str] = []

    for index, spec in enumerate(
        contract.expand(stage="stage0-calibration")
    ):
        source_run_id = f"v26-source-{index}"
        source_manifest_sha256 = f"{index + 1:064x}"
        envelope_content_sha256 = f"{index + 101:064x}"
        terminal_content_sha256 = f"{index + 201:064x}"
        envelope_path = (
            stage_root
            / "runs"
            / spec.run_id
            / "imported_run_envelope.json"
        )
        envelope_path.parent.mkdir(parents=True)
        envelope = {
            "schema_version": (
                orchestrator.PILOT_V27_IMPORTED_RUN_ENVELOPE_SCHEMA_VERSION
            ),
            "source_import": {
                "source_run_id": source_run_id,
                "source_artifacts": {
                    "manifest": {
                        "file_sha256": source_manifest_sha256,
                    }
                },
            },
            "integrity": {
                "content_sha256": envelope_content_sha256,
            },
        }
        envelope_path.write_text(
            json.dumps(envelope, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        envelope_file_sha256 = evidence._sha256_file(envelope_path)
        envelope_binding = {
            "path": str(envelope_path),
            "file_sha256": envelope_file_sha256,
            "content_sha256": envelope_content_sha256,
            "schema_version": envelope["schema_version"],
        }
        metrics = {"profile": spec.utility_profile_id, "seed": spec.environment_seed}
        gate_evidence = {
            "status": "pass",
            "execution_disposition": (
                "immutable-parent-import-offline-resummary"
            ),
            "provider_calls_current_attempt": 0,
            "imported_run_envelope": envelope_binding,
        }
        terminal_path = stage_root / "summaries" / f"{spec.run_id}.json"
        terminal_path.parent.mkdir(parents=True, exist_ok=True)
        terminal = {
            "payload": {
                "metrics": metrics,
                "gate_evidence": gate_evidence,
            },
            "integrity": {
                "content_sha256": terminal_content_sha256,
            },
        }
        terminal_path.write_text(
            json.dumps(terminal, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        terminal_file_sha256 = evidence._sha256_file(terminal_path)
        source_rows.append(
            {
                "run_id": spec.run_id,
                "utility_profile_id": spec.utility_profile_id,
                "environment_seed": spec.environment_seed,
                "execution_disposition": (
                    "immutable-parent-import-offline-resummary"
                ),
                "envelope": str(envelope_path),
                "envelope_file_sha256": envelope_file_sha256,
                "envelope_content_sha256": envelope_content_sha256,
                "terminal_summary": str(terminal_path),
                "terminal_summary_file_sha256": terminal_file_sha256,
                "terminal_summary_content_sha256": terminal_content_sha256,
                "source_run_id": source_run_id,
                "source_manifest_sha256": source_manifest_sha256,
                "provider_calls_current_attempt": 0,
            }
        )
        aggregate_rows.append(
            {
                **spec.to_dict(),
                "status": "complete",
                "artifact_kind": "imported-stage0-run-envelope",
                "artifact_sha256": terminal_file_sha256,
                "metrics": metrics,
                "gate_evidence": gate_evidence,
            }
        )

    selection = {
        "schema_version": "finevo-stage0-selection-v1",
        "contract_sha256": contract.canonical_hash,
        "selected_profile_id": "rho-1",
        "selected_utility": {"rho": 1.0},
        "absolute_flow_utility_threshold": {"value": 0.5},
        "outcome_fields_used": [],
        "bindings": {
            "contract_sha256": contract.canonical_hash,
            "git_tag": contract.implementation["required_git_tag"],
            "git_commit": commit,
            "source_envelopes": source_rows,
        },
        "integrity": {
            "canonicalization": "json-sort-keys-utf8-v1",
            "content_sha256": "",
        },
    }
    selection["integrity"]["content_sha256"] = evidence._bound_artifact_hash(
        selection
    )
    (stage_root / "stage0_selection.json").write_text(
        json.dumps(selection, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    receipt = {
        "schema_version": evidence.PILOT_STAGE_RECEIPT_SCHEMA_VERSION_V2,
        "contract_sha256": contract.canonical_hash,
        "stage_id": "stage0-calibration",
        "status": "complete",
        "go": True,
        "terminal": True,
        "registered_run_count": len(source_rows),
        "complete_cell_count": len(source_rows),
        "bindings": {
            "contract_sha256": contract.canonical_hash,
            "run_ledger_schema_version": (
                evidence.PILOT_RUN_LEDGER_SCHEMA_VERSION_V2
            ),
            "stage_specs_sha256": canonical_sha256(
                [
                    spec.to_dict()
                    for spec in contract.expand(stage="stage0-calibration")
                ]
            ),
            "stage_rows_sha256": "b" * 64,
            "ledger_event_chain_head": "c" * 64,
            "source_files_sha256": "d" * 64,
        },
    }
    receipt["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(receipt),
    }
    (stage_root / "stage_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def verify(
        _contract: Any,
        spec: Any,
        terminal_path: Path,
        *_args: Any,
    ) -> dict[str, Any]:
        verifier_calls.append(spec.run_id)
        value = json.loads(terminal_path.read_text(encoding="utf-8"))
        gate = value["payload"]["gate_evidence"]
        return {
            "metrics": value["payload"]["metrics"],
            "envelope_binding": gate["imported_run_envelope"],
            "execution_disposition": gate["execution_disposition"],
            "provider_calls_current_attempt": 0,
            "scientific_evidence": True,
        }

    monkeypatch.setattr(
        orchestrator,
        "verify_v27_imported_stage0_terminal",
        verify,
    )
    replayed_selection = json.loads(json.dumps(selection))
    monkeypatch.setattr(
        orchestrator,
        "verify_v27_stage0_selection",
        lambda *_args, **_kwargs: replayed_selection,
    )
    controls = evidence._validated_release_controls(
        contract,
        raw_root=raw_root,
        rows=aggregate_rows,
        common_commit=commit,
    )
    assert controls["stage0_selection"]["pass"] is True
    assert controls["stage0_selection"]["checks"]["complete_source_matrix"] is True
    assert (
        controls["stage0_selection"]["checks"]["selection_semantic_replay"]
        is True
    )
    assert (
        "selection_is_outcome_blind"
        not in controls["stage0_selection"]["checks"]
    )
    assert (
        controls["stage0_selection"]["checks"][
            "selection_uses_no_a_d_treatment_outcome_fields"
        ]
        is True
    )
    assert set(verifier_calls) == {row["run_id"] for row in source_rows}

    aggregate_rows[0]["artifact_sha256"] = "f" * 64
    tampered = evidence._validated_release_controls(
        contract,
        raw_root=raw_root,
        rows=aggregate_rows,
        common_commit=commit,
    )
    assert tampered["stage0_selection"]["pass"] is False
    assert (
        tampered["stage0_selection"]["checks"]["complete_source_matrix"]
        is False
    )

    aggregate_rows[0]["artifact_sha256"] = source_rows[0][
        "terminal_summary_file_sha256"
    ]
    tampered_selection = json.loads(json.dumps(selection))
    tampered_selection["selected_profile_id"] = "attacker-selected-profile"
    tampered_selection["integrity"]["content_sha256"] = (
        evidence._bound_artifact_hash(tampered_selection)
    )
    (stage_root / "stage0_selection.json").write_text(
        json.dumps(tampered_selection, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resealed_tamper = evidence._validated_release_controls(
        contract,
        raw_root=raw_root,
        rows=aggregate_rows,
        common_commit=commit,
    )
    assert resealed_tamper["stage0_selection"]["pass"] is False
    assert (
        resealed_tamper["stage0_selection"]["checks"]["sealed_selection"]
        is True
    )
    assert (
        resealed_tamper["stage0_selection"]["checks"][
            "complete_source_matrix"
        ]
        is True
    )
    assert (
        resealed_tamper["stage0_selection"]["checks"][
            "selection_semantic_replay"
        ]
        is False
    )

    threshold_tamper = json.loads(json.dumps(selection))
    threshold_tamper["absolute_flow_utility_threshold"]["value"] = 999.0
    threshold_tamper["integrity"]["content_sha256"] = (
        evidence._bound_artifact_hash(threshold_tamper)
    )
    (stage_root / "stage0_selection.json").write_text(
        json.dumps(threshold_tamper, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    threshold_rejected = evidence._validated_release_controls(
        contract,
        raw_root=raw_root,
        rows=aggregate_rows,
        common_commit=commit,
    )
    assert threshold_rejected["stage0_selection"]["pass"] is False
    assert (
        threshold_rejected["stage0_selection"]["checks"]["sealed_selection"]
        is True
    )
    assert (
        threshold_rejected["stage0_selection"]["checks"][
            "complete_source_matrix"
        ]
        is True
    )
    assert (
        threshold_rejected["stage0_selection"]["checks"][
            "selection_semantic_replay"
        ]
        is False
    )


def test_v27_budget_validator_accepts_exact_import_rows_and_rejects_extra(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    parent_debit = orchestrator._parent_budget_debit(contract)
    assert parent_debit is not None
    assert parent_debit.cost_usd == pytest.approx(3.212770875)
    assert parent_debit.hosted_completions == 184
    assert parent_debit.storage_bytes == 19_181_432
    assert evidence._expected_parent_budget_debit(contract) == (
        parent_debit.to_dict()
    )

    budget = orchestrator.PilotBudgetLedger(
        raw_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=parent_debit,
    )
    imported_specs = tuple(contract.expand(stage="parent-import")) + tuple(
        contract.expand(stage="q-ref-resolution")
    ) + tuple(contract.expand(stage="stage0-calibration"))
    assert len(imported_specs) == 16
    for spec in imported_specs:
        projection = orchestrator.RunProjection(
            run_id=spec.run_id,
            stage_bucket=spec.budget_bucket,
            cost_usd=0.0,
            completions=0,
            storage_bytes=0,
            basis={"method": "test-exact-v2.7-import-row"},
        )
        budget.reserve(projection)
        budget.finalize(
            spec.run_id,
            status="complete",
            cost_usd=0.0,
            completions=0,
            storage_bytes=0,
            failure=None,
        )

    controls = evidence._validated_release_controls(
        contract,
        raw_root=raw_root,
        rows=[],
        common_commit=None,
    )
    budget_control = controls["budget_ledger"]
    assert budget_control["pass"] is True
    assert budget_control["checks"]["parent_debit_exact"] is True
    assert budget_control["checks"]["valid_finalized_dispatch_units"] is True
    assert budget_control["actual_totals"] == {
        "cost_usd": pytest.approx(3.212770875),
        "completions": 184,
        "storage_bytes": 19_181_432,
    }

    extra = orchestrator.RunProjection(
        run_id="finevo-pilot-v2.7--unexpected-budget-row",
        stage_bucket=imported_specs[-1].budget_bucket,
        cost_usd=0.0,
        completions=0,
        storage_bytes=0,
        basis={"method": "test-unregistered-row"},
    )
    budget.reserve(extra)
    budget.finalize(
        extra.run_id,
        status="complete",
        cost_usd=0.0,
        completions=0,
        storage_bytes=0,
        failure=None,
    )
    rejected = evidence._validated_release_controls(
        contract,
        raw_root=raw_root,
        rows=[],
        common_commit=None,
    )
    assert rejected["budget_ledger"]["pass"] is False
    assert (
        rejected["budget_ledger"]["checks"][
            "valid_finalized_dispatch_units"
        ]
        is False
    )

    missing_parent_root = tmp_path / "missing-parent"
    missing_parent_root.mkdir()
    missing_parent = orchestrator.PilotBudgetLedger(
        missing_parent_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=parent_debit,
    )
    for spec in imported_specs[1:]:
        projection = orchestrator.RunProjection(
            run_id=spec.run_id,
            stage_bucket=spec.budget_bucket,
            cost_usd=0.0,
            completions=0,
            storage_bytes=0,
            basis={"method": "test-resealed-without-parent-row"},
        )
        missing_parent.reserve(projection)
        missing_parent.finalize(
            spec.run_id,
            status="complete",
            cost_usd=0.0,
            completions=0,
            storage_bytes=0,
            failure=None,
        )
    missing_parent_controls = evidence._validated_release_controls(
        contract,
        raw_root=missing_parent_root,
        rows=[],
        common_commit=None,
    )
    assert (
        missing_parent_controls["budget_ledger"]["checks"][
            "self_hash_and_event_chain"
        ]
        is True
    )
    assert (
        missing_parent_controls["budget_ledger"]["checks"][
            "parent_debit_exact"
        ]
        is True
    )
    assert (
        missing_parent_controls["budget_ledger"]["checks"][
            "valid_finalized_dispatch_units"
        ]
        is False
    )
