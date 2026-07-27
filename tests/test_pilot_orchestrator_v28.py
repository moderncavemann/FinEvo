from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.pilot_calibration import q_ref_run_config
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.runner import VerifiedRunResult


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_8.yaml"


def _paid() -> SimpleNamespace:
    value = SimpleNamespace(
        git_tag="pilot-v2.8-science",
        head_commit="a" * 40,
    )
    value.to_dict = lambda: {
        "git_tag": value.git_tag,
        "head_commit": value.head_commit,
    }
    return value


def test_v28_frozen_denominator_and_import_partition() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)

    assert contract.contract_id == orchestrator.V28_CONTRACT_ID
    assert len(contract.expand()) == 211
    assert len(contract.expand(stage="parent-import")) == 1
    assert len(contract.expand(stage="q-ref-resolution")) == 1
    assert len(contract.expand(stage="stage0-calibration")) == 14
    assert contract.budgets["total_usd"] == pytest.approx(500.0)


def test_v28_parent_and_stage0_routes_never_construct_hosted_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    routed: list[str] = []

    class Ledger:
        def register(self, _specs: Any) -> None:
            return None

        def status(self, _run_id: str) -> str:
            return "scheduled"

    class Budget:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("V2.8 prerequisite stages constructed a provider")

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _p: contract)
    monkeypatch.setattr(
        orchestrator, "verify_paid_provenance", lambda *_a, **_k: _paid()
    )
    monkeypatch.setattr(
        orchestrator, "_persist_release_attestation", lambda *_a, **_k: None
    )
    monkeypatch.setattr(orchestrator, "PilotRunLedger", lambda *_a, **_k: Ledger())
    monkeypatch.setattr(orchestrator, "PilotBudgetLedger", Budget)
    monkeypatch.setattr(orchestrator, "_parent_budget_debit", lambda _c: object())
    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden)

    def parent(*_a: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["budget_ledger"] is not None
        routed.append("parent-import")
        return {"status": "complete"}

    def stage0(*_a: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["budget_ledger"] is not None
        routed.append("stage0-calibration")
        return {"status": "complete"}

    monkeypatch.setattr(orchestrator, "_execute_v24_parent_import_stage", parent)
    monkeypatch.setattr(orchestrator, "_execute_imported_stage0_stage", stage0)
    for stage_id in ("parent-import", "stage0-calibration"):
        result = orchestrator._execute_stage_locked(
            contract_path=CONTRACT_PATH,
            stage_id=stage_id,
            resume=True,
            raw_root=tmp_path / "experiment_results" / "pilot-v2.8" / "raw",
            repo_root=tmp_path,
            parent_repo_root=tmp_path,
        )
        assert result["status"] == "complete"
    assert routed == ["parent-import", "stage0-calibration"]


def test_v28_qref_is_fresh_scripted_and_zero_hosted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="q-ref-resolution")[0]
    config = q_ref_run_config(run_id=spec.run_id)
    result = VerifiedRunResult(
        config=config.to_dict(),
        summary={"complete": True},
        validation_status={"status": "pass"},
        budget_snapshot={},
        records={
            "actions": tuple({"row": i} for i in range(48)),
            "utility_ledger": tuple({"row": i} for i in range(48)),
            "shock_events": (),
        },
    )
    observed_provider: list[str] = []

    def forbidden(*_a: Any, **_k: Any) -> None:
        raise AssertionError("fresh q-ref constructed a hosted provider")

    def multi(provider: Any, *, num_workers: int) -> object:
        observed_provider.append(type(provider).__name__)
        assert num_workers == 4
        return object()

    def write_artifacts(run_dir: Path, *_a: Any, **_k: Any) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "manifest.json"
        path.write_text("{}\n", encoding="utf-8")
        return path

    def terminal(path: Path, **_kwargs: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return path

    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden)
    monkeypatch.setattr(orchestrator, "MultiModelLLM", multi)
    monkeypatch.setattr(
        orchestrator, "run_verified_experiment", lambda *_a, **_k: result
    )
    monkeypatch.setattr(
        orchestrator, "write_verified_run_artifacts", write_artifacts
    )
    monkeypatch.setattr(orchestrator, "write_terminal_summary", terminal)
    monkeypatch.setattr(
        orchestrator,
        "build_q_ref_resolution",
        lambda *_a, **_k: {
            "schema_version": "finevo-q-ref-resolution-v1",
            "status": "pass",
            "q_ref": 1.25,
            "row_count": 48,
            "checks": {"complete": True},
            "bindings": {
                "contract_hash": contract.canonical_hash,
                "source_config_hash": "b" * 64,
                "run_summary_hash": "c" * 64,
                "ledger_hash": "d" * 64,
                "environment_source_hash": orchestrator._file_sha256(
                    orchestrator.DEFAULT_ENV_CONFIG
                ),
            },
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "_build_v28_qref_equivalence_receipt",
        lambda *_a, **_k: orchestrator._seal_bound_payload(
            {
                "schema_version": (
                    orchestrator.PILOT_V28_QREF_EQUIVALENCE_SCHEMA_VERSION
                ),
                "status": "pass",
            }
        ),
    )
    projection = orchestrator.RunProjection(
        run_id=spec.run_id,
        stage_bucket=spec.budget_bucket,
        cost_usd=0.0,
        completions=0,
        storage_bytes=2_000_000,
        basis={"method": "scripted"},
    )
    _, _, resolution = orchestrator._execute_q_ref(
        contract,
        spec,
        raw_root=tmp_path,
        paid=_paid(),
        projection=projection,
    )
    assert observed_provider == ["ScriptedDiagnosticProvider"]
    assert resolution["provider_calls_current_attempt"] == 48
    assert resolution["hosted_provider_calls_current_attempt"] == 0
    assert resolution["hosted_cost_usd_current_attempt"] == 0.0
    assert resolution["q_ref_audit_equivalence"]["status"] == "pass"


def test_v28_qref_equivalence_replays_fixture_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="q-ref-resolution")[0]
    historical_id = "v26-historical-qref"
    current_config = q_ref_run_config(run_id=spec.run_id).to_dict()
    historical_config = {**current_config, "run_id": historical_id}
    actions = tuple({"period": i, "action": i % 3} for i in range(48))
    ledger = tuple({"period": i, "flow_utility": float(i)} for i in range(48))
    shocks = tuple({"period": i, "interest_rate": 0.03} for i in range(12))
    result = VerifiedRunResult(
        config=current_config,
        summary={"complete": True},
        validation_status={"status": "pass"},
        budget_snapshot={},
        records={
            "actions": actions,
            "utility_ledger": ledger,
            "shock_events": shocks,
        },
    )
    source_declared_root = (
        "experiment_results/pilot-v2.7/raw/"
        "parent-import/v2_6_raw_snapshot/q-ref-resolution/runs/"
        f"{historical_id}"
    )
    source_resolution_declared = (
        "experiment_results/pilot-v2.7/raw/"
        "parent-import/v2_6_raw_snapshot/q-ref-resolution/"
        "q_ref_resolution.json"
    )
    source_run_dir = (
        tmp_path
        / "parent-import"
        / "v2_7_raw_snapshot"
        / "parent-import"
        / "v2_6_raw_snapshot"
        / "q-ref-resolution"
        / "runs"
        / historical_id
    )
    source_run_dir.mkdir(parents=True)
    source_manifest_path = source_run_dir / "manifest.json"
    source_config_path = source_run_dir / "config.json"
    source_manifest_path.write_text("{}\n", encoding="utf-8")
    source_config_path.write_text(
        json.dumps(historical_config, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    streams = source_run_dir / "streams"
    streams.mkdir()
    for name, rows in (
        ("actions", actions),
        ("utility_ledger", ledger),
        ("shock_events", shocks),
    ):
        (streams / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    environment_hash = orchestrator._file_sha256(
        orchestrator.DEFAULT_ENV_CONFIG
    )
    common_resolution = {
        "schema_version": "finevo-q-ref-resolution-v1",
        "status": "pass",
        "scientific_evidence": False,
        "q_ref": 1.25,
        "row_count": 48,
        "run_contract": {"agents": 4, "months": 12},
        "checks": {"complete": True},
        "source": {"run_summary": {"complete": True}},
    }
    source_resolution = orchestrator._seal_bound_payload(
        {
            **common_resolution,
            "bindings": {
                "ledger_hash": "d" * 64,
                "environment_source_hash": environment_hash,
                "source_config_hash": "1" * 64,
            },
        }
    )
    source_resolution_path = (
        tmp_path
        / "parent-import"
        / "v2_7_raw_snapshot"
        / "parent-import"
        / "v2_6_raw_snapshot"
        / "q-ref-resolution"
        / "q_ref_resolution.json"
    )
    source_resolution_path.parent.mkdir(parents=True, exist_ok=True)
    source_resolution_path.write_text(
        json.dumps(source_resolution, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    current_manifest = tmp_path / "current-manifest.json"
    current_manifest.write_text("{}\n", encoding="utf-8")
    parent_receipt_path = (
        tmp_path / "parent-import" / "parent_import_receipt.json"
    )
    parent_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    parent_receipt_path.write_text("{}\n", encoding="utf-8")
    source_manifest = {
        "integrity": {"content_sha256": "e" * 64},
        "q_ref_audit_equivalence_reference": {
            "source_run_root": source_declared_root,
            "source_manifest": {
                "file_sha256": orchestrator._file_sha256(
                    source_manifest_path
                )
            },
            "source_config": {
                "file_sha256": orchestrator._file_sha256(source_config_path)
            },
            "q_ref_resolution": {
                "path": source_resolution_declared,
                "file_sha256": orchestrator._file_sha256(
                    source_resolution_path
                ),
            },
            "q_ref": 1.25,
        },
    }
    parent_receipt = {"integrity": {"content_sha256": "f" * 64}}
    monkeypatch.setattr(
        orchestrator,
        "_v28_import_authority",
        lambda *_a, **_k: (source_manifest, parent_receipt),
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_manifest",
        lambda _path: SimpleNamespace(
            valid=True,
            manifest_sha256=orchestrator._file_sha256(source_manifest_path),
        ),
    )
    original_file_hash = orchestrator._file_sha256

    def file_hash(path: Path) -> str:
        if path.name == "pilot_v2_8_source_manifest.json" and not path.exists():
            return "9" * 64
        return original_file_hash(path)

    monkeypatch.setattr(orchestrator, "_file_sha256", file_hash)
    current_resolution = {
        **common_resolution,
        "bindings": {
            "ledger_hash": "d" * 64,
            "environment_source_hash": environment_hash,
            "source_config_hash": "2" * 64,
        },
    }
    receipt = orchestrator._build_v28_qref_equivalence_receipt(
        contract,
        result,
        current_resolution,
        raw_root=tmp_path,
        paid=_paid(),
        current_manifest=current_manifest,
    )
    assert receipt["status"] == "pass"
    assert all(receipt["comparison"].values())
    assert receipt["provider_boundary"] == {
        "scripted_diagnostic_calls": 48,
        "hosted_provider_calls": 0,
        "hosted_cost_usd": 0.0,
        "hosted_provider_construction": False,
    }

    tampered = VerifiedRunResult(
        config=current_config,
        summary=result.summary,
        validation_status=result.validation_status,
        budget_snapshot=result.budget_snapshot,
        records={**result.records, "actions": ({"tampered": True},) + actions[1:]},
    )
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="actions_exact",
    ):
        orchestrator._build_v28_qref_equivalence_receipt(
            contract,
            tampered,
            current_resolution,
            raw_root=tmp_path,
            paid=_paid(),
            current_manifest=current_manifest,
        )


def test_v28_p95_binding_uses_v28_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    model_id = next(iter(orchestrator.V28_ALLOWED_P95_PROFILES))
    receipt = (
        tmp_path
        / "experiment_results"
        / "pilot-v2.8"
        / "raw"
        / "p95"
        / f"{model_id}.json"
    )
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    observed: list[tuple[str, Path, str]] = []

    monkeypatch.setattr(
        orchestrator,
        "_observed_p95_authority_receipt_path",
        lambda *_a, **_k: receipt,
    )

    def verify(
        relative: str,
        *,
        repo_root: Path,
        expected_git_commit: str,
    ) -> dict[str, Any]:
        observed.append((relative, repo_root, expected_git_commit))
        return {
            "receipt_path": relative,
            "git_commit": expected_git_commit,
        }

    monkeypatch.setattr(
        orchestrator,
        "verified_v28_observed_p95_authority_binding",
        verify,
    )
    binding = orchestrator._verified_observed_p95_binding(
        contract,
        model_id,
        raw_root=receipt.parents[2],
        paid=_paid(),
        authority_repo_root=tmp_path,
    )
    assert binding["git_commit"] == "a" * 40
    assert observed == [
        (
            receipt.relative_to(tmp_path).as_posix(),
            tmp_path,
            "a" * 40,
        )
    ]


def test_v28_parent_post_commit_failure_requires_audited_resume(
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
    importer_calls: list[str] = []
    result = {
        "receipt": str(
            raw_root / "parent-import" / "parent_import_receipt.json"
        ),
        "snapshot_inventory_sha256": "1" * 64,
        "resealed_p95_profiles": {"sentinel": "stable"},
        "provider_calls_during_import": 0,
        "scientific_evidence": False,
    }

    def persist(**_kwargs: Any) -> dict[str, Any]:
        importer_calls.append("verified")
        return dict(result)

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
        expected = json.dumps(result, sort_keys=True) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_text(encoding="utf-8") == expected
        else:
            path.write_text(expected, encoding="utf-8")
        return path

    receipt_attempts = 0

    def stage_receipt(*_args: Any, **_kwargs: Any) -> Path:
        nonlocal receipt_attempts
        receipt_attempts += 1
        if receipt_attempts == 1:
            raise OSError("injected V2.8 post-commit receipt failure")
        path = raw_root / "parent-import" / "stage_receipt.json"
        path.write_text('{"status":"complete"}\n', encoding="utf-8")
        return path

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

    def forbidden(*_a: Any, **_k: Any) -> None:
        raise AssertionError("V2.8 parent import constructed a provider")

    monkeypatch.setattr(orchestrator, "persist_v28_parent_import", persist)
    monkeypatch.setattr(
        orchestrator,
        "_verify_v28_importer_p95_profiles",
        lambda _contract, *, importer_result, **_kwargs: importer_result[
            "resealed_p95_profiles"
        ],
    )
    monkeypatch.setattr(
        orchestrator,
        "_materialize_or_verify_v27_parent_terminal",
        terminal,
    )
    monkeypatch.setattr(orchestrator, "_write_stage_receipt", stage_receipt)
    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden)
    budget = Budget()
    kwargs = {
        "raw_root": raw_root,
        "repo_root": tmp_path,
        "parent_repo_root": tmp_path,
        "paid": _paid(),
        "run_ledger": ledger,
        "budget_ledger": budget,
    }
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="V2.8 parent publication requires audited --resume",
    ):
        orchestrator._execute_v24_parent_import_stage(
            contract,
            (spec,),
            **kwargs,
        )
    assert ledger.status(spec.run_id) == "complete"
    assert sum(
        row["status"] == "scheduled"
        for run_id, row in ledger.snapshot()["runs"].items()
        if run_id != spec.run_id
    ) == 210
    assert (
        raw_root / "parent-import" / "parent_import_commit_intent.json"
    ).is_file()
    assert not (
        raw_root / "parent-import" / "failure_receipt.json"
    ).exists()

    value = orchestrator._execute_v24_parent_import_stage(
        contract,
        (spec,),
        **kwargs,
    )
    assert value["status"] == "complete"
    assert budget.rows[spec.run_id]["status"] == "complete"
    assert importer_calls == ["verified", "verified"]


def test_v28_stage0_commits_exactly_fourteen_imported_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    specs = tuple(contract.expand(stage="stage0-calibration"))
    ledger = orchestrator.PilotRunLedger(
        tmp_path / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())
    for stage_id in ("parent-import", "q-ref-resolution"):
        prereq = contract.expand(stage=stage_id)[0]
        artifact = tmp_path / stage_id / f"{prereq.run_id}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}\n", encoding="utf-8")
        ledger.finalize(
            prereq.run_id,
            status="complete",
            artifact=str(artifact),
            failure=None,
        )

    cells = []
    for spec in specs:
        envelope = tmp_path / spec.stage_id / "runs" / spec.run_id / "envelope.json"
        terminal = tmp_path / spec.stage_id / "summaries" / f"{spec.run_id}.json"
        envelope.parent.mkdir(parents=True, exist_ok=True)
        terminal.parent.mkdir(parents=True, exist_ok=True)
        envelope.write_text("{}\n", encoding="utf-8")
        terminal.write_text("{}\n", encoding="utf-8")
        cells.append((spec, envelope, terminal, {"status": "pass"}))
    selection = {
        "selected_profile_id": "center",
        "integrity": {"content_sha256": "f" * 64},
    }

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

        def finalize(self, run_id: str, **kwargs: Any) -> None:
            self.rows[run_id]["status"] = kwargs["status"]

    def forbidden(*_a: Any, **_k: Any) -> None:
        raise AssertionError("V2.8 imported Stage-0 constructed a provider")

    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden)
    monkeypatch.setattr(orchestrator, "_assert_prerequisites", lambda *_a, **_k: {})
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

    def receipt(*_a: Any, **_k: Any) -> Path:
        path = tmp_path / "stage0-calibration" / "stage_receipt.json"
        path.write_text('{"status":"complete"}\n', encoding="utf-8")
        return path

    monkeypatch.setattr(orchestrator, "_write_stage_receipt", receipt)
    budget = Budget()
    value = orchestrator._execute_imported_stage0_stage(
        contract,
        specs,
        raw_root=tmp_path,
        paid=_paid(),
        run_ledger=ledger,
        budget_ledger=budget,
    )
    rows = ledger.snapshot()["runs"]
    assert value["status"] == "complete"
    assert sum(row["status"] == "complete" for row in rows.values()) == 16
    assert sum(row["status"] == "scheduled" for row in rows.values()) == 195
    assert len(budget.rows) == 14
    assert {row["status"] for row in budget.rows.values()} == {"complete"}
