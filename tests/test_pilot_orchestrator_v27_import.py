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


class _ParentBudget:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.finalize_calls = 0

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
        self.finalize_calls += 1
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


def _install_stable_parent_success(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raw_root: Path,
    spec: Any,
    calls: list[str],
) -> None:
    def persist(**_kwargs: Any) -> dict[str, Any]:
        calls.append("importer")
        return {
            "receipt": str(
                raw_root / "parent-import" / "parent_import_receipt.json"
            ),
            "snapshot_inventory_sha256": "a" * 64,
            "resealed_p95_profiles": {"sentinel": "stable"},
            "provider_calls": 0,
            "scientific_evidence": False,
        }

    def terminal(
        _contract: Any,
        _spec: Any,
        *,
        result: dict[str, Any],
        **_kwargs: Any,
    ) -> Path:
        path = (
            raw_root
            / "parent-import"
            / "summaries"
            / f"{spec.run_id}.json"
        )
        expected = {
            "schema_version": "test-parent-terminal-v1",
            "result_sha256": canonical_sha256(result),
        }
        if path.exists():
            assert json.loads(path.read_text(encoding="utf-8")) == expected
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(expected, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return path

    monkeypatch.setattr(orchestrator, "persist_v27_parent_import", persist)
    monkeypatch.setattr(
        orchestrator,
        "_verify_v27_importer_p95_profiles",
        lambda _contract, *, importer_result, **_kwargs: importer_result[
            "resealed_p95_profiles"
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "_materialize_or_verify_v27_parent_terminal",
        terminal,
    )


def _write_v27_failure_intent(
    contract: Any,
    raw_root: Path,
    failure: dict[str, Any],
    *,
    provider_calls: int = 0,
) -> Path:
    path = orchestrator._v27_parent_failure_receipt_path(
        raw_root=raw_root
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    value = orchestrator._seal_bound_payload(
        {
            "schema_version": (
                orchestrator.PILOT_V27_PARENT_FAILURE_INTENT_SCHEMA_VERSION
            ),
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "stage_id": "parent-import",
            "provider_calls": provider_calls,
            "scientific_evidence": False,
            "failure": failure,
            "bindings": {
                "contract_sha256": contract.canonical_hash,
                "git_tag": _paid().git_tag,
                "git_commit": _paid().head_commit,
            },
        }
    )
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _assert_exact_parent_no_go(
    contract: Any,
    spec: Any,
    *,
    raw_root: Path,
    ledger: Any,
) -> None:
    rows = ledger.snapshot()["runs"]
    assert len(rows) == 211
    assert {row["status"] for row in rows.values()} == {
        "integrity-stopped"
    }
    failure = rows[spec.run_id]["failure"]
    orchestrator._assert_v27_parent_no_go_ledger_boundary(
        contract,
        spec,
        raw_root=raw_root,
        run_ledger=ledger,
        failure=failure,
    )


def _new_parent_ledger(contract: Any, raw_root: Path) -> Any:
    ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())
    return ledger


def _execute_parent(
    contract: Any,
    spec: Any,
    *,
    raw_root: Path,
    repo_root: Path,
    ledger: Any,
    budget: Any,
) -> dict[str, Any]:
    return orchestrator._execute_v24_parent_import_stage(
        contract,
        (spec,),
        raw_root=raw_root,
        repo_root=repo_root,
        parent_repo_root=repo_root,
        paid=_paid(),
        run_ledger=ledger,
        budget_ledger=budget,
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
        def __init__(self) -> None:
            self.rows = {
                candidate.run_id: {
                    "status": "scheduled",
                    "artifact": None,
                    "failure": None,
                }
                for candidate in contract.expand()
            }

        def is_terminal(self, run_id: str) -> bool:
            return self.rows[run_id]["status"] != "scheduled"

        def status(self, run_id: str) -> str:
            return self.rows[run_id]["status"]

        def finalize(self, run_id: str, **kwargs: Any) -> None:
            self.rows[run_id] = {
                "status": kwargs["status"],
                "artifact": kwargs["artifact"],
                "failure": kwargs["failure"],
            }

        def snapshot(self) -> dict[str, Any]:
            return {"runs": self.rows}

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
    fault_observations: list[tuple[str, str]] = []

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
                fault_observations.append(
                    ("budget", run_ledger.status(spec.run_id))
                )
                assert run_ledger.status(spec.run_id) == "complete"
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
    terminal_observations: list[str] = []

    def terminal(
        _contract: Any,
        _spec: Any,
        *,
        result: dict[str, Any],
        **_kwargs: Any,
    ) -> Path:
        path = (
            raw_root
            / "parent-import"
            / "summaries"
            / f"{spec.run_id}.json"
        )
        expected = {
            "schema_version": "test-parent-terminal-v1",
            "result_sha256": canonical_sha256(result),
        }
        if path.exists():
            assert json.loads(path.read_text(encoding="utf-8")) == expected
            terminal_observations.append("verified")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(expected, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            terminal_observations.append("created")
        return path

    monkeypatch.setattr(
        orchestrator,
        "_materialize_or_verify_v27_parent_terminal",
        terminal,
    )
    boundary_observations: list[dict[str, int]] = []
    original_boundary = (
        orchestrator._assert_v27_parent_success_ledger_boundary
    )

    def assert_boundary(*args: Any, **kwargs: Any) -> None:
        statuses: dict[str, int] = {}
        for row in run_ledger.snapshot()["runs"].values():
            status = row["status"]
            statuses[status] = statuses.get(status, 0) + 1
        boundary_observations.append(statuses)
        original_boundary(*args, **kwargs)

    monkeypatch.setattr(
        orchestrator,
        "_assert_v27_parent_success_ledger_boundary",
        assert_boundary,
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
            fault_observations.append(
                ("receipt", run_ledger.status(spec.run_id))
            )
            assert run_ledger.status(spec.run_id) == "complete"
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
    assert fault_observations == [(failure_mode, "complete")]

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
    assert terminal_observations == ["created", "verified"]
    assert boundary_observations == [
        {"complete": 1, "scheduled": 210},
        {"complete": 1, "scheduled": 210},
    ]


def test_v27_complete_parent_missing_intent_fails_before_any_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="parent-import")[0]
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())
    terminal = (
        raw_root
        / "parent-import"
        / "summaries"
        / f"{spec.run_id}.json"
    )
    terminal.parent.mkdir(parents=True)
    terminal.write_text("{}\n", encoding="utf-8")
    ledger.finalize(
        spec.run_id,
        status="complete",
        artifact=str(terminal),
        failure=None,
    )
    side_effects: list[str] = []

    class UntouchedBudget:
        def snapshot(self) -> dict[str, Any]:
            side_effects.append("budget-snapshot")
            return {"runs": {}}

        def reserve(self, _projection: Any) -> None:
            side_effects.append("budget-reserve")

        def finalize(self, _run_id: str, **_kwargs: Any) -> None:
            side_effects.append("budget-finalize")

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        side_effects.append("publication")
        raise AssertionError("publication must not start")

    monkeypatch.setattr(orchestrator, "persist_v27_parent_import", forbidden)
    monkeypatch.setattr(orchestrator, "_write_stage_receipt", forbidden)
    monkeypatch.setattr(orchestrator, "_propagate_stage_no_go", forbidden)

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="lacks its exclusive pre-commit success intent",
    ):
        _execute_parent(
            contract,
            spec,
            raw_root=raw_root,
            repo_root=tmp_path,
            ledger=ledger,
            budget=UntouchedBudget(),
        )

    assert side_effects == []
    assert not orchestrator._v27_parent_commit_intent_path(
        raw_root=raw_root
    ).exists()
    assert not (raw_root / "parent-import" / "stage_receipt.json").exists()
    assert not orchestrator._v27_parent_failure_receipt_path(
        raw_root=raw_root
    ).exists()
    assert ledger.status(spec.run_id) == "complete"
    assert {
        row["status"]
        for run_id, row in ledger.snapshot()["runs"].items()
        if run_id != spec.run_id
    } == {"scheduled"}


def test_v27_resealed_post_commit_entry_state_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="parent-import")[0]
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    terminal = (
        raw_root
        / "parent-import"
        / "summaries"
        / f"{spec.run_id}.json"
    )
    terminal.parent.mkdir(parents=True)
    terminal.write_text("{}\n", encoding="utf-8")
    result = {"provider_calls": 0, "scientific_evidence": False}
    intent_path = orchestrator._persist_v27_parent_commit_intent(
        contract,
        spec,
        raw_root=raw_root,
        paid=_paid(),
        terminal_path=terminal,
        result=result,
        projection=orchestrator._v27_parent_import_projection(spec),
        entry_run_status="scheduled",
    )
    resealed = json.loads(intent_path.read_text(encoding="utf-8"))
    resealed["entry_run_status"] = "complete"
    resealed = orchestrator._seal_bound_payload(resealed)
    intent_path.chmod(0o644)
    intent_path.write_text(
        json.dumps(resealed, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ledger = _new_parent_ledger(contract, raw_root)
    ledger.finalize(
        spec.run_id,
        status="complete",
        artifact=str(terminal),
        failure=None,
    )
    side_effects: list[str] = []

    class UntouchedBudget:
        def snapshot(self) -> dict[str, Any]:
            side_effects.append("budget")
            return {"runs": {}}

    monkeypatch.setattr(
        orchestrator,
        "persist_v27_parent_import",
        lambda **_kwargs: side_effects.append("importer"),
    )
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="must record the original scheduled entry state",
    ):
        _execute_parent(
            contract,
            spec,
            raw_root=raw_root,
            repo_root=tmp_path,
            ledger=ledger,
            budget=UntouchedBudget(),
        )
    assert side_effects == []
    assert not (raw_root / "parent-import" / "stage_receipt.json").exists()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("success-symlink", "success intent must not be a symlink"),
        ("failure-symlink", "failure intent must not be a symlink"),
        ("failure-hash", "content hash mismatch"),
        ("failure-malformed", "failure receipt is malformed"),
        ("dual-intents", "success and failure intents cannot coexist"),
    ],
)
def test_v27_parent_intent_state_rejects_unsafe_artifacts(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw_root = tmp_path / "raw"
    parent_root = raw_root / "parent-import"
    parent_root.mkdir(parents=True)
    if case == "dual-intents":
        spec = contract.expand(stage="parent-import")[0]
        terminal = parent_root / "summaries" / f"{spec.run_id}.json"
        terminal.parent.mkdir(parents=True)
        terminal.write_text("{}\n", encoding="utf-8")
        orchestrator._persist_v27_parent_commit_intent(
            contract,
            spec,
            raw_root=raw_root,
            paid=_paid(),
            terminal_path=terminal,
            result={"provider_calls": 0},
            projection=orchestrator._v27_parent_import_projection(spec),
            entry_run_status="scheduled",
        )
        _write_v27_failure_intent(
            contract,
            raw_root,
            {"error_type": "TestFailure", "message": "sealed"},
        )
    elif case.endswith("symlink"):
        target = tmp_path / f"{case}.json"
        target.write_text("{}\n", encoding="utf-8")
        path = (
            orchestrator._v27_parent_commit_intent_path(raw_root=raw_root)
            if case.startswith("success")
            else orchestrator._v27_parent_failure_receipt_path(
                raw_root=raw_root
            )
        )
        path.symlink_to(target)
    else:
        path = _write_v27_failure_intent(
            contract,
            raw_root,
            {"error_type": "TestFailure", "message": "sealed"},
            provider_calls=1 if case == "failure-malformed" else 0,
        )
        if case == "failure-hash":
            value = json.loads(path.read_text(encoding="utf-8"))
            value["failure"]["message"] = "tampered-after-seal"
            path.write_text(
                json.dumps(value, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=message,
    ):
        orchestrator._classify_v27_parent_entry_state(
            contract,
            raw_root=raw_root,
            paid=_paid(),
            entry_run_status="scheduled",
        )


def test_v27_parent_success_intent_survives_parent_ledger_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="parent-import")[0]
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    ledger = _new_parent_ledger(contract, raw_root)
    budget = _ParentBudget()
    importer_calls: list[str] = []
    _install_stable_parent_success(
        monkeypatch,
        raw_root=raw_root,
        spec=spec,
        calls=importer_calls,
    )

    def write_receipt(*_args: Any, **_kwargs: Any) -> Path:
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
    original_finalize = ledger.finalize
    fail_once = True

    def crash_before_success_commit(run_id: str, **kwargs: Any) -> None:
        nonlocal fail_once
        if (
            fail_once
            and run_id == spec.run_id
            and kwargs["status"] == "complete"
        ):
            fail_once = False
            raise OSError("injected parent success ledger commit failure")
        original_finalize(run_id, **kwargs)

    monkeypatch.setattr(ledger, "finalize", crash_before_success_commit)
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="requires audited --resume",
    ):
        _execute_parent(
            contract,
            spec,
            raw_root=raw_root,
            repo_root=tmp_path,
            ledger=ledger,
            budget=budget,
        )
    assert orchestrator._v27_parent_commit_intent_path(
        raw_root=raw_root
    ).is_file()
    assert not orchestrator._v27_parent_failure_receipt_path(
        raw_root=raw_root
    ).exists()
    reloaded = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    assert {row["status"] for row in reloaded.snapshot()["runs"].values()} == {
        "scheduled"
    }

    resumed = _execute_parent(
        contract,
        spec,
        raw_root=raw_root,
        repo_root=tmp_path,
        ledger=reloaded,
        budget=budget,
    )
    statuses = {
        row["status"] for row in reloaded.snapshot()["runs"].values()
    }
    assert resumed["status"] == "complete"
    assert statuses == {"complete", "scheduled"}
    assert importer_calls == ["importer", "importer"]
    assert budget.rows[spec.run_id]["status"] == "complete"


def test_v27_parent_success_rejects_terminalized_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="parent-import")[0]
    descendant = contract.expand(stage="q-ref-resolution")[0]
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    ledger = _new_parent_ledger(contract, raw_root)
    ledger.finalize(
        descendant.run_id,
        status="integrity-stopped",
        artifact=None,
        failure={"error_type": "InjectedDescendantTerminal"},
    )
    budget = _ParentBudget()
    importer_calls: list[str] = []
    _install_stable_parent_success(
        monkeypatch,
        raw_root=raw_root,
        spec=spec,
        calls=importer_calls,
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="requires audited --resume",
    ):
        _execute_parent(
            contract,
            spec,
            raw_root=raw_root,
            repo_root=tmp_path,
            ledger=ledger,
            budget=budget,
        )
    assert ledger.status(spec.run_id) == "complete"
    assert budget.rows[spec.run_id]["status"] == "reserved"
    assert budget.finalize_calls == 0
    assert not (raw_root / "parent-import" / "stage_receipt.json").exists()

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="only pristine scheduled descendants",
    ):
        _execute_parent(
            contract,
            spec,
            raw_root=raw_root,
            repo_root=tmp_path,
            ledger=ledger,
            budget=budget,
        )
    assert importer_calls == ["importer", "importer"]
    assert budget.finalize_calls == 0
    assert not (raw_root / "parent-import" / "stage_receipt.json").exists()


def test_v27_parent_partial_no_go_propagation_resumes_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="parent-import")[0]
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    ledger_path = raw_root / "run_ledger.json"
    ledger = _new_parent_ledger(contract, raw_root)
    budget = _ParentBudget()
    importer_calls: list[str] = []

    def fail_import(**_kwargs: Any) -> None:
        importer_calls.append("importer")
        raise RuntimeError("injected immutable parent failure")

    monkeypatch.setattr(
        orchestrator,
        "persist_v27_parent_import",
        fail_import,
    )
    original_finalize = ledger.finalize
    propagated = 0

    def flaky_finalize(run_id: str, **kwargs: Any) -> None:
        nonlocal propagated
        if run_id != spec.run_id and kwargs["status"] == "integrity-stopped":
            propagated += 1
            if propagated == 7:
                raise OSError("injected descendant ledger write failure")
        original_finalize(run_id, **kwargs)

    monkeypatch.setattr(ledger, "finalize", flaky_finalize)
    with pytest.raises(
        OSError,
        match="injected descendant ledger write failure",
    ):
        _execute_parent(
            contract,
            spec,
            raw_root=raw_root,
            repo_root=tmp_path,
            ledger=ledger,
            budget=budget,
        )
    persisted = orchestrator.PilotRunLedger(
        ledger_path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    interim = persisted.snapshot()["runs"]
    assert interim[spec.run_id]["status"] == "integrity-stopped"
    assert 1 < sum(
        row["status"] == "integrity-stopped" for row in interim.values()
    ) < 211
    assert budget.rows[spec.run_id]["status"] == "reserved"
    assert budget.finalize_calls == 0
    assert not (raw_root / "parent-import" / "stage_receipt.json").exists()

    resumed = _execute_parent(
        contract,
        spec,
        raw_root=raw_root,
        repo_root=tmp_path,
        ledger=persisted,
        budget=budget,
    )
    assert resumed["status"] == "integrity-stopped"
    assert importer_calls == ["importer"]
    _assert_exact_parent_no_go(
        contract,
        spec,
        raw_root=raw_root,
        ledger=persisted,
    )
    assert budget.rows[spec.run_id]["status"] == "integrity-stopped"
    assert budget.finalize_calls == 1


def test_v27_parent_failure_intent_survives_parent_ledger_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="parent-import")[0]
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    ledger_path = raw_root / "run_ledger.json"
    ledger = _new_parent_ledger(contract, raw_root)
    budget = _ParentBudget()
    importer_calls: list[str] = []

    def fail_import(**_kwargs: Any) -> None:
        importer_calls.append("importer")
        raise RuntimeError("injected parent verification failure")

    monkeypatch.setattr(
        orchestrator,
        "persist_v27_parent_import",
        fail_import,
    )
    original_finalize = ledger.finalize

    def crash_before_parent_commit(run_id: str, **kwargs: Any) -> None:
        if (
            run_id == spec.run_id
            and kwargs["status"] == "integrity-stopped"
        ):
            raise OSError("injected parent ledger commit failure")
        original_finalize(run_id, **kwargs)

    monkeypatch.setattr(
        ledger,
        "finalize",
        crash_before_parent_commit,
    )
    with pytest.raises(
        OSError,
        match="injected parent ledger commit failure",
    ):
        _execute_parent(
            contract,
            spec,
            raw_root=raw_root,
            repo_root=tmp_path,
            ledger=ledger,
            budget=budget,
        )
    failure_path = orchestrator._v27_parent_failure_receipt_path(
        raw_root=raw_root
    )
    assert failure_path.is_file()
    reloaded = orchestrator.PilotRunLedger(
        ledger_path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    assert {row["status"] for row in reloaded.snapshot()["runs"].values()} == {
        "scheduled"
    }
    assert budget.rows[spec.run_id]["status"] == "reserved"

    resumed = _execute_parent(
        contract,
        spec,
        raw_root=raw_root,
        repo_root=tmp_path,
        ledger=reloaded,
        budget=budget,
    )
    assert resumed["status"] == "integrity-stopped"
    assert importer_calls == ["importer"]
    _assert_exact_parent_no_go(
        contract,
        spec,
        raw_root=raw_root,
        ledger=reloaded,
    )
    assert budget.rows[spec.run_id]["status"] == "integrity-stopped"
    assert budget.finalize_calls == 1


def test_v27_parent_no_go_conflicting_terminal_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="parent-import")[0]
    descendant = contract.expand(stage="q-ref-resolution")[0]
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    ledger_path = raw_root / "run_ledger.json"
    ledger = _new_parent_ledger(contract, raw_root)
    ledger.finalize(
        descendant.run_id,
        status="integrity-stopped",
        artifact=None,
        failure={"error_type": "ConflictingTerminal"},
    )
    budget = _ParentBudget()
    importer_calls: list[str] = []

    def fail_import(**_kwargs: Any) -> None:
        importer_calls.append("importer")
        raise RuntimeError("injected parent verification failure")

    monkeypatch.setattr(
        orchestrator,
        "persist_v27_parent_import",
        fail_import,
    )
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="descendant terminal state conflicts",
    ):
        _execute_parent(
            contract,
            spec,
            raw_root=raw_root,
            repo_root=tmp_path,
            ledger=ledger,
            budget=budget,
        )
    assert not (raw_root / "parent-import" / "stage_receipt.json").exists()
    assert budget.finalize_calls == 0

    reloaded = orchestrator.PilotRunLedger(
        ledger_path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="descendant terminal state conflicts",
    ):
        _execute_parent(
            contract,
            spec,
            raw_root=raw_root,
            repo_root=tmp_path,
            ledger=reloaded,
            budget=budget,
        )
    assert importer_calls == ["importer"]
    assert budget.rows[spec.run_id]["status"] == "reserved"
    assert budget.finalize_calls == 0
    assert not (raw_root / "parent-import" / "stage_receipt.json").exists()


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


@pytest.mark.parametrize(
    ("contract_path", "envelope_schema"),
    (
        (
            CONTRACT_PATH,
            orchestrator.PILOT_V27_IMPORTED_RUN_ENVELOPE_SCHEMA_VERSION,
        ),
        (
            ROOT / "experiments" / "pilot_v2_8.yaml",
            orchestrator.PILOT_V28_IMPORTED_RUN_ENVELOPE_SCHEMA_VERSION,
        ),
        (
            ROOT / "experiments" / "pilot_v2_9_overlay.yaml",
            orchestrator.PILOT_V29_IMPORTED_RUN_ENVELOPE_SCHEMA_VERSION,
        ),
    ),
)
def test_imported_stage0_release_controls_validate_envelope_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_path: Path,
    envelope_schema: str,
) -> None:
    contract = load_pilot_contract(contract_path)
    raw_root = tmp_path / "raw"
    stage_root = raw_root / "stage0-calibration"
    stage_root.mkdir(parents=True)
    commit = "a" * 40
    source_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    verifier_calls: list[str] = []
    verifier_kwargs_calls: list[dict[str, Any]] = []

    def verify(
        _contract: Any,
        spec: Any,
        terminal_path: Path,
        *_args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        run_id = spec["run_id"] if isinstance(spec, dict) else spec.run_id
        verifier_calls.append(run_id)
        verifier_kwargs_calls.append(kwargs)
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
    specs = tuple(contract.expand(stage="stage0-calibration"))
    terminal_paths: list[Path] = []
    for index, spec in enumerate(specs):
        source_run_id = f"v26-source-{index}"
        source_manifest_sha256 = f"{index + 1:064x}"
        envelope_content_sha256 = f"{index + 101:064x}"
        envelope_path = (
            stage_root
            / "runs"
            / spec.run_id
            / "imported_run_envelope.json"
        )
        envelope_path.parent.mkdir(parents=True)
        envelope = {
            "schema_version": envelope_schema,
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
        evidence.write_terminal_summary(
            terminal_path,
            contract=contract,
            run_spec=spec,
            resolved_git_commit=commit,
            git_tag=contract.implementation["required_git_tag"],
            payload={
                "metrics": metrics,
                "gate_evidence": gate_evidence,
            },
            scientific_evidence=True,
            diagnostic_only=False,
            evidence_scope=evidence.CURRENT_SCIENTIFIC_SCOPE,
        )
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        terminal_content_sha256 = terminal["integrity"]["content_sha256"]
        terminal_file_sha256 = evidence._sha256_file(terminal_path)
        terminal_paths.append(terminal_path)
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

    completed = evidence._load_completed_artifact(
        contract,
        specs[0].to_dict(),
        raw_root=raw_root,
        artifact=str(terminal_paths[0]),
        source_repo_root=ROOT,
    )
    assert completed["artifact_kind"] == "imported-stage0-run-envelope"
    verifier_calls.clear()
    verifier_kwargs_calls.clear()

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

    replayed_selection = json.loads(json.dumps(selection))
    selection_kwargs_calls: list[dict[str, Any]] = []

    def verify_selection(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        selection_kwargs_calls.append(kwargs)
        return replayed_selection

    monkeypatch.setattr(
        orchestrator,
        "verify_v27_stage0_selection",
        verify_selection,
    )
    external_source_root = tmp_path / "science-source"
    external_source_root.mkdir()
    controls = evidence._validated_release_controls(
        contract,
        raw_root=raw_root,
        rows=aggregate_rows,
        common_commit=commit,
        source_repo_root=external_source_root,
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
    expected_verifier_kwargs = (
        [{"authority_repo_root": external_source_root}] * len(specs)
        if contract.contract_id == orchestrator.V29_CONTRACT_ID
        else [{}] * len(specs)
    )
    assert verifier_kwargs_calls == expected_verifier_kwargs
    assert selection_kwargs_calls == (
        [{"authority_repo_root": external_source_root}]
        if contract.contract_id == orchestrator.V29_CONTRACT_ID
        else [{}]
    )

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


def test_v29_imported_stage0_terminal_enforces_versioned_envelope_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(
        ROOT / "experiments" / "pilot_v2_9_overlay.yaml"
    )
    spec = contract.expand(stage="stage0-calibration")[0]
    raw_root = tmp_path / "raw"
    commit = "a" * 40
    envelope_path = orchestrator._v27_stage0_envelope_path(raw_root, spec)
    envelope_path.parent.mkdir(parents=True)
    expected = orchestrator._seal_bound_payload(
        {
            "schema_version": (
                orchestrator.PILOT_V29_IMPORTED_RUN_ENVELOPE_SCHEMA_VERSION
            ),
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "run_spec": spec.to_dict(),
            "execution_disposition": (
                "immutable-parent-import-offline-resummary"
            ),
            "reader": {
                "summary": {
                    "profile": spec.utility_profile_id,
                    "seed": spec.environment_seed,
                }
            },
            "claim_boundary": "imported Stage-0 calibration only",
            "bindings": {
                "contract_sha256": contract.canonical_hash,
                "git_tag": contract.implementation["required_git_tag"],
                "git_commit": commit,
            },
        }
    )
    envelope_path.write_text(
        json.dumps(expected, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    terminal_path = orchestrator._v27_stage0_terminal_path(raw_root, spec)
    evidence.write_terminal_summary(
        terminal_path,
        contract=contract,
        run_spec=spec,
        resolved_git_commit=commit,
        git_tag=contract.implementation["required_git_tag"],
        payload=orchestrator._v27_stage0_terminal_payload(
            expected,
            envelope_path,
        ),
        scientific_evidence=True,
        diagnostic_only=False,
        evidence_scope=evidence.CURRENT_SCIENTIFIC_SCOPE,
    )
    monkeypatch.setattr(
        orchestrator,
        "_v29_import_authority",
        lambda *_args, **_kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        orchestrator,
        "_build_v27_stage0_envelope",
        lambda *_args, **_kwargs: (expected, None, {}, raw_root),
    )

    verified = orchestrator.verify_v27_imported_stage0_terminal(
        contract,
        spec,
        terminal_path,
        raw_root,
        commit,
        contract.implementation["required_git_tag"],
    )
    assert verified["scientific_evidence"] is True
    assert verified["provider_calls_current_attempt"] == 0

    wrong_version = json.loads(json.dumps(expected))
    wrong_version["schema_version"] = (
        orchestrator.PILOT_V28_IMPORTED_RUN_ENVELOPE_SCHEMA_VERSION
    )
    wrong_version = orchestrator._seal_bound_payload(wrong_version)
    envelope_path.write_text(
        json.dumps(wrong_version, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="unsupported schema version",
    ):
        orchestrator.verify_v27_imported_stage0_terminal(
            contract,
            spec,
            terminal_path,
            raw_root,
            commit,
            contract.implementation["required_git_tag"],
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
