from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from llm_providers import MultiModelLLM
import pytest

from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.budget import BudgetLimits, RunBudget
from verified_memory.pilot_calibration import (
    build_q_ref_resolution,
    q_ref_run_config,
)
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.runner import VerifiedRunResult, run_verified_experiment
from verified_memory.runner_artifacts import write_verified_run_artifacts
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_9.yaml"
HISTORICAL_QREF_RUN_ID = (
    "finevo-pilot-v2.8--q-ref-resolution--qref_scripted--"
    "qref-scripted--none--provider-preflight-default--s2010922376"
)


def _paid() -> SimpleNamespace:
    value = SimpleNamespace(
        git_tag="pilot-v2.9-science",
        head_commit="a" * 40,
    )
    value.to_dict = lambda: {
        "git_tag": value.git_tag,
        "head_commit": value.head_commit,
    }
    return value


def _qref_budget(run_id: str) -> RunBudget:
    return RunBudget(
        BudgetLimits(
            max_calls=48,
            max_prompt_tokens=500_000,
            max_completion_tokens=100_000,
            max_total_tokens=600_000,
            max_cost_usd=1e-9,
            max_elapsed_seconds=3_600.0,
        ),
        budget_id=f"{run_id}-budget",
    )


def _scripted_qref(run_id: str) -> VerifiedRunResult:
    return run_verified_experiment(
        q_ref_run_config(run_id=run_id),
        llm=MultiModelLLM(
            ScriptedDiagnosticProvider(),
            num_workers=4,
        ),
        budget=_qref_budget(run_id),
        env_config_source=orchestrator.DEFAULT_ENV_CONFIG,
    )


def _jsonl_binding(path: Path, *, declared: str) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return {
        "path": declared,
        "file_sha256": orchestrator._file_sha256(path),
        "byte_size": path.stat().st_size,
        "row_count": len(rows),
    }


def test_v29_frozen_denominator_and_fresh_partition() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    stages: dict[str, int] = {}
    for spec in contract.expand():
        stages[spec.stage_id] = stages.get(spec.stage_id, 0) + 1

    assert contract.contract_id == orchestrator.V29_CONTRACT_ID
    assert len(contract.expand()) == 211
    assert stages["parent-import"] == 1
    assert stages["q-ref-resolution"] == 1
    assert stages["stage0-calibration"] == 14
    assert sum(
        count
        for stage, count in stages.items()
        if stage.startswith("experiment-")
        or stage.startswith("local-experiment-")
    ) == 195
    amendment = contract.qref_summary_equivalence_amendment
    assert amendment["q_ref_regeneration"]["source_result_reuse"] == "forbidden"
    assert (
        amendment["q_ref_regeneration"][
            "fresh_zero_hosted_provider_regeneration"
        ]
        is True
    )
    assert amendment["stage0_import"]["imported_complete_cells"] == 14
    assert contract.budgets["total_usd"] == pytest.approx(500.0)


def test_v29_parent_and_stage0_routes_never_construct_hosted_provider(
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
        raise AssertionError("V2.9 prerequisite route constructed a provider")

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _p: contract)
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_a, **_k: _paid(),
    )
    monkeypatch.setattr(
        orchestrator,
        "_persist_release_attestation",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(orchestrator, "PilotRunLedger", lambda *_a, **_k: Ledger())
    monkeypatch.setattr(orchestrator, "PilotBudgetLedger", Budget)
    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden)

    def parent(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["budget_ledger"] is not None
        routed.append("parent-import")
        return {"status": "complete"}

    def stage0(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["budget_ledger"] is not None
        routed.append("stage0-calibration")
        return {"status": "complete"}

    monkeypatch.setattr(orchestrator, "_execute_v24_parent_import_stage", parent)
    monkeypatch.setattr(orchestrator, "_execute_imported_stage0_stage", stage0)
    raw_root = tmp_path / "experiment_results" / "pilot-v2.9" / "raw"
    for stage_id in ("parent-import", "stage0-calibration"):
        result = orchestrator._execute_stage_locked(
            contract_path=CONTRACT_PATH,
            stage_id=stage_id,
            resume=True,
            raw_root=raw_root,
            repo_root=tmp_path,
            parent_repo_root=tmp_path,
        )
        assert result["status"] == "complete"
    assert routed == ["parent-import", "stage0-calibration"]


def test_v29_stage0_import_rejects_anything_but_exact_fourteen_cells(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    specs = tuple(contract.expand(stage="stage0-calibration"))
    assert len(specs) == 14
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="exact 7x2 actor matrix",
    ):
        orchestrator._execute_imported_stage0_stage(
            contract,
            specs[:-1],
            raw_root=tmp_path,
            paid=_paid(),
            run_ledger=object(),
            budget_ledger=object(),
        )


def test_v29_qref_projection_and_exact_stream_receipts_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="q-ref-resolution")[0]
    current = _scripted_qref(spec.run_id)
    historical = _scripted_qref(HISTORICAL_QREF_RUN_ID)
    raw_root = tmp_path / "raw"
    current_run_dir = (
        raw_root
        / "q-ref-resolution"
        / "runs"
        / spec.run_id
    )
    historical_run_dir = tmp_path / "historical" / HISTORICAL_QREF_RUN_ID
    current_manifest = write_verified_run_artifacts(
        current_run_dir,
        current,
        provenance={
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "run_spec": spec.to_dict(),
        },
        git_commit="a" * 40,
        git_dirty=False,
    )
    source_contract_sha256 = "8" * 64
    source_spec = {
        **spec.to_dict(),
        "contract_id": orchestrator.V28_CONTRACT_ID,
        "run_id": HISTORICAL_QREF_RUN_ID,
    }
    write_verified_run_artifacts(
        historical_run_dir,
        historical,
        provenance={
            "contract_id": orchestrator.V28_CONTRACT_ID,
            "contract_sha256": source_contract_sha256,
            "run_spec": source_spec,
        },
        git_commit=orchestrator.V29_PARENT_COMMIT,
        git_dirty=False,
    )

    declared_root = (
        "experiment_results/pilot-v2.8/raw/q-ref-resolution/runs/"
        f"{HISTORICAL_QREF_RUN_ID}"
    )
    path_map: dict[str, Path] = {declared_root: historical_run_dir}
    fixed: dict[str, dict[str, Any]] = {}
    for name, suffix in (
        ("manifest", "manifest.json"),
        ("config", "config.json"),
        ("provenance", "provenance.json"),
    ):
        declared = f"{declared_root}/{suffix}"
        path = historical_run_dir / suffix
        path_map[declared] = path
        fixed[name] = {
            "path": declared,
            "file_sha256": orchestrator._file_sha256(path),
            "byte_size": path.stat().st_size,
        }
    streams: dict[str, dict[str, Any]] = {}
    for name in (
        "summary",
        "actions",
        "api_usage",
        "utility_ledger",
        "shock_events",
    ):
        declared = f"{declared_root}/streams/{name}.jsonl"
        path = historical_run_dir / "streams" / f"{name}.jsonl"
        path_map[declared] = path
        streams[name] = _jsonl_binding(path, declared=declared)
    reference = {
        "imported": False,
        "source_result_reuse": "forbidden",
        "source_contract_id": orchestrator.V28_CONTRACT_ID,
        "source_contract_sha256": source_contract_sha256,
        "source_contract_cell_run_id": HISTORICAL_QREF_RUN_ID,
        "source_run_root": declared_root,
        "source_spec": source_spec,
        "verified_runner": {
            **fixed,
            "streams": streams,
            "provider_accounting": {
                "provider_kind": "scripted-diagnostic",
                "model": "diagnostic/scripted-v1",
                "scripted_diagnostic_calls": 48,
                "hosted_provider_calls": 0,
                "hosted_cost_usd": 0.0,
                "total_tokens": 15905,
            },
        },
        "ancestral_v2_6_scalar_reference": {
            "q_ref": 63.50397933257746,
        },
    }
    source_manifest = {
        "integrity": {"content_sha256": "e" * 64},
        "q_ref_audit_reference": reference,
    }
    parent_receipt = {"integrity": {"content_sha256": "f" * 64}}
    parent_receipt_path = (
        raw_root / "parent-import" / "parent_import_receipt.json"
    )
    parent_receipt_path.parent.mkdir(parents=True)
    parent_receipt_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        orchestrator,
        "_v29_import_authority",
        lambda *_a, **_k: (source_manifest, parent_receipt),
    )
    monkeypatch.setattr(
        orchestrator,
        "q_ref_audit_reference_v29",
        lambda value: deepcopy(value["q_ref_audit_reference"]),
    )
    monkeypatch.setattr(
        orchestrator,
        "snapshot_path_for_v28_source_artifact_v29",
        lambda _raw, declared: path_map[declared],
    )
    resolution = build_q_ref_resolution(
        current,
        contract_hash=contract.canonical_hash,
        environment_source_hash=orchestrator._file_sha256(
            orchestrator.DEFAULT_ENV_CONFIG
        ),
    )

    summary_receipt, audit_receipt = (
        orchestrator._build_v29_qref_equivalence_receipts(
            contract,
            current,
            resolution,
            raw_root=raw_root,
            paid=_paid(),
            current_manifest=current_manifest,
        )
    )
    assert summary_receipt["comparison"]["leaf_path_count"] == 1002
    assert summary_receipt["comparison"]["normalized_leaf_path_count"] == 195
    assert summary_receipt["current"]["provider_boundary"][
        "hosted_provider_calls"
    ] == 0
    assert summary_receipt["current"]["accounting"]["accounted_usage"] == {
        "prompt_tokens": 14657,
        "completion_tokens": 1248,
        "total_tokens": 15905,
        "cost_usd": 0.0,
    }
    assert all(audit_receipt["comparison"].values())

    resolution["bindings"]["source_manifest_sha256"] = (
        orchestrator._file_sha256(current_manifest)
    )
    resolution["bindings"]["contract_sha256"] = contract.canonical_hash
    resolution["bindings"]["git_tag"] = _paid().git_tag
    resolution["bindings"]["git_commit"] = _paid().head_commit
    resolution["source_manifest"] = str(current_manifest)
    resolution["scientific_evidence"] = False
    resolution["provider_calls_current_attempt"] = 48
    resolution["hosted_provider_calls_current_attempt"] = 0
    resolution["hosted_cost_usd_current_attempt"] = 0.0
    resolution["q_ref_summary_equivalence"] = summary_receipt
    resolution["q_ref_audit_equivalence"] = audit_receipt
    resolution = orchestrator._seal_bound_payload(resolution)
    resolution_path = raw_root / "q-ref-resolution" / "q_ref_resolution.json"
    resolution_path.write_text(
        json.dumps(resolution, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    replayed = orchestrator.verify_v29_qref_resolution(
        contract,
        resolution_path,
        raw_root,
        _paid().head_commit,
        _paid().git_tag,
    )
    assert replayed == resolution

    current_actions = current_run_dir / "streams" / "actions.jsonl"
    current_actions_bytes = current_actions.read_bytes()
    current_actions.unlink()
    with pytest.raises(orchestrator.PilotOrchestrationError):
        orchestrator.verify_v29_qref_resolution(
            contract,
            resolution_path,
            raw_root,
            _paid().head_commit,
            _paid().git_tag,
        )
    current_actions.write_bytes(current_actions_bytes)

    historical_actions = historical_run_dir / "streams" / "actions.jsonl"
    historical_actions_bytes = historical_actions.read_bytes()
    historical_actions.unlink()
    with pytest.raises(orchestrator.PilotOrchestrationError):
        orchestrator.verify_v29_qref_resolution(
            contract,
            resolution_path,
            raw_root,
            _paid().head_commit,
            _paid().git_tag,
        )
    historical_actions.write_bytes(historical_actions_bytes)

    parent_receipt_bytes = parent_receipt_path.read_bytes()
    parent_receipt_path.unlink()
    with pytest.raises(orchestrator.PilotOrchestrationError):
        orchestrator.verify_v29_qref_resolution(
            contract,
            resolution_path,
            raw_root,
            _paid().head_commit,
            _paid().git_tag,
        )
    parent_receipt_path.write_bytes(parent_receipt_bytes)

    source_manifest_path = orchestrator.V29_SOURCE_MANIFEST_PATH
    monkeypatch.setattr(
        orchestrator,
        "V29_SOURCE_MANIFEST_PATH",
        Path("experiments/missing-v2.9-source-manifest.json"),
    )
    with pytest.raises(orchestrator.PilotOrchestrationError):
        orchestrator.verify_v29_qref_resolution(
            contract,
            resolution_path,
            raw_root,
            _paid().head_commit,
            _paid().git_tag,
        )
    monkeypatch.setattr(
        orchestrator,
        "V29_SOURCE_MANIFEST_PATH",
        source_manifest_path,
    )

    tampered = VerifiedRunResult(
        config=current.config,
        summary=current.summary,
        validation_status=current.validation_status,
        budget_snapshot=current.budget_snapshot,
        records={
            **current.records,
            "actions": ({"tampered": True},) + current.stream("actions")[1:],
        },
    )
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="actions_exact",
    ):
        orchestrator._build_v29_qref_equivalence_receipts(
            contract,
            tampered,
            resolution,
            raw_root=raw_root,
            paid=_paid(),
            current_manifest=current_manifest,
        )

    reference["verified_runner"]["streams"]["summary"]["byte_size"] += 1
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="summary binding drifted",
    ):
        orchestrator._build_v29_qref_equivalence_receipts(
            contract,
            current,
            resolution,
            raw_root=raw_root,
            paid=_paid(),
            current_manifest=current_manifest,
        )


def test_v29_qref_terminal_binds_full_embedded_summary_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="q-ref-resolution")[0]
    result = VerifiedRunResult(
        config=q_ref_run_config(run_id=spec.run_id).to_dict(),
        summary={"unused": True},
        validation_status={"status": "pass"},
        budget_snapshot={},
        records={},
    )
    captured: dict[str, Any] = {}

    def write_artifacts(run_dir: Path, *_args: Any, **_kwargs: Any) -> Path:
        run_dir.mkdir(parents=True)
        path = run_dir / "manifest.json"
        path.write_text("{}\n", encoding="utf-8")
        return path

    def write_terminal(path: Path, **kwargs: Any) -> Path:
        captured.update(kwargs["payload"])
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")
        return path

    summary_receipt = {
        "schema_version": (
            orchestrator.QREF_RUN_SUMMARY_EQUIVALENCE_SCHEMA_VERSION
        ),
        "status": "pass",
        "integrity": {"content_sha256": "1" * 64},
    }
    audit_receipt = orchestrator._seal_bound_payload(
        {
            "schema_version": (
                orchestrator.PILOT_V29_QREF_EQUIVALENCE_SCHEMA_VERSION
            ),
            "status": "pass",
        }
    )
    monkeypatch.setattr(
        orchestrator,
        "run_verified_experiment",
        lambda *_a, **_k: result,
    )
    monkeypatch.setattr(
        orchestrator,
        "write_verified_run_artifacts",
        write_artifacts,
    )
    monkeypatch.setattr(orchestrator, "write_terminal_summary", write_terminal)
    monkeypatch.setattr(
        orchestrator,
        "build_q_ref_resolution",
        lambda *_a, **_k: {
            "schema_version": "finevo-q-ref-resolution-v1",
            "status": "pass",
            "q_ref": 63.50397933257746,
            "row_count": 48,
            "checks": {"complete": True},
            "bindings": {
                "contract_hash": contract.canonical_hash,
                "source_config_hash": "2" * 64,
                "run_summary_hash": "3" * 64,
                "ledger_hash": "4" * 64,
                "environment_source_hash": orchestrator._file_sha256(
                    orchestrator.DEFAULT_ENV_CONFIG
                ),
            },
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "_build_v29_qref_equivalence_receipts",
        lambda *_a, **_k: (summary_receipt, audit_receipt),
    )
    projection = orchestrator.conservative_projection(
        contract,
        spec,
        diagnostic=True,
    )
    _, _, resolution = orchestrator._execute_q_ref(
        contract,
        spec,
        raw_root=tmp_path,
        paid=_paid(),
        projection=projection,
    )

    assert resolution["q_ref_summary_equivalence"] == summary_receipt
    assert resolution["q_ref_audit_equivalence"] == audit_receipt
    binding = captured["q_ref_resolution"]["summary_equivalence"]
    assert binding["embedded_key"] == "q_ref_summary_equivalence"
    assert binding["path"] == str(
        tmp_path / "q-ref-resolution" / "q_ref_resolution.json"
    )
    assert binding["file_sha256"] == orchestrator._file_sha256(
        Path(binding["path"])
    )
    assert binding["content_sha256"] == resolution["integrity"]["content_sha256"]


def test_v29_imported_p95_is_read_only_and_tamper_evident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    model_id = "gpt52_main"
    profile = contract.provider_profiles[model_id]
    runtime_model = orchestrator._runtime_model_for_profile(profile)
    reservation = {
        "sample_count": 1,
        "raw_p95": {
            "prompt_tokens": 1.0,
            "completion_tokens": 1.0,
            "total_tokens": 2.0,
            "cost_usd": 0.0,
        },
        "reserved_p95": {
            "prompt_tokens": 2,
            "completion_tokens": 2,
            "total_tokens": 4,
            "cost_usd": 0.0,
        },
        "reserve_multiplier": 1.25,
    }
    projection_rows = {
        f"{profile.served_model}::{kind}": deepcopy(reservation)
        for kind in ("action", "semantic")
    }
    source_contract_sha256 = "8" * 64
    payload = orchestrator._seal_bound_payload(
        {
            "schema_version": orchestrator.PILOT_PROJECTION_SCHEMA_VERSION,
            "model_id": model_id,
            "served_model": profile.served_model,
            "projection": projection_rows,
            "bindings": {
                "contract_sha256": source_contract_sha256,
                "git_commit": orchestrator.V29_PARENT_COMMIT,
            },
        }
    )
    projection_path = tmp_path / "projection_p95.json"
    receipt_path = tmp_path / "observed_p95_authority_receipt.json"
    projection_path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_path.write_text("{}\n", encoding="utf-8")
    binding = {
        "profile_id": model_id,
        "source_contract_id": orchestrator.V28_CONTRACT_ID,
        "source_contract_sha256": source_contract_sha256,
        "source_git_commit": orchestrator.V29_PARENT_COMMIT,
        "source_git_tag": "pilot-v2.8-science",
        "authority": {
            "path": "authority.json",
            "schema_version": "authority-v1",
            "file_sha256": orchestrator._file_sha256(receipt_path),
            "content_sha256": "7" * 64,
        },
        "projection": {
            "path": "projection.json",
            "schema_version": orchestrator.PILOT_PROJECTION_SCHEMA_VERSION,
            "file_sha256": orchestrator._file_sha256(projection_path),
            "content_sha256": payload["integrity"]["content_sha256"],
        },
        "runtime_model": runtime_model,
        "served_model": profile.served_model,
        "reservations": {
            runtime_model: {
                kind: {"reservation": deepcopy(reservation)}
                for kind in ("action", "semantic")
            }
        },
    }
    monkeypatch.setattr(
        orchestrator,
        "_v29_import_authority",
        lambda *_a, **_k: ({}, {}),
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v29_imported_v28_observed_p95",
        lambda *_a, **_k: deepcopy(binding),
    )
    monkeypatch.setattr(
        orchestrator,
        "v29_observed_p95_projection_path",
        lambda *_a, **_k: projection_path,
    )
    monkeypatch.setattr(
        orchestrator,
        "v29_observed_p95_receipt_path",
        lambda *_a, **_k: receipt_path,
    )

    loaded, path = orchestrator._load_verified_projection(
        contract,
        model_id,
        raw_root=tmp_path,
        paid=_paid(),
    )
    assert path == projection_path
    assert loaded == payload

    projection_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="binding drifted",
    ):
        orchestrator._load_verified_projection(
            contract,
            model_id,
            raw_root=tmp_path,
            paid=_paid(),
        )


def test_v29_parent_receipt_rejects_hosted_import_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    receipt = {
        "provider_calls_during_import": 0,
        "hosted_provider_calls_during_import": 1,
        "scripted_diagnostic_calls_during_import": 0,
        "provider_construction_during_import": False,
        "scientific_evidence": False,
        "source_parent": {
            "terminal_status": "complete-with-no-go",
            "scientific_complete": False,
        },
        "q_ref": {
            "imported": False,
            "source_result_reuse": "forbidden",
            "fresh_v2_9_regeneration_required": True,
            "scripted_diagnostic_calls": 48,
            "hosted_provider_calls": 0,
            "hosted_cost_usd": 0.0,
        },
    }
    monkeypatch.setattr(
        orchestrator,
        "load_v29_source_manifest",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v29_parent_import_receipt",
        lambda **_kwargs: receipt,
    )
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="import/q-ref boundary",
    ):
        orchestrator._v29_import_authority(
            contract,
            raw_root=tmp_path,
            paid=_paid(),
        )
