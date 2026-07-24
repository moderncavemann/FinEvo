from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verified_memory.budget import BudgetLimits, RunBudget
from verified_memory.pilot_budget import RunProjection
from verified_memory.pilot_contract import canonical_sha256, load_pilot_contract
from verified_memory.pilot_evidence import write_terminal_summary
from verified_memory.pilot_orchestrator import (
    PILOT_SCIENTIFIC_LAUNCH_INPUT_SCHEMA_VERSION,
    PILOT_STAGE_RECEIPT_SCHEMA_VERSION,
    GitProvenance,
    PilotOrchestrationError,
    PilotRunLedger,
    _assert_prerequisites,
    _build_experiment_c_sensitivity,
    _cross_model_science_stage_ids,
    _persist_release_attestation,
    _preflight_config,
    _preflight_stage_for_model,
    _propagate_capability_no_go,
    _remaining_scientific_projections,
    _scientific_stage_ids,
    _v2_capability_semantic_go,
    _verify_v2_stage_receipt,
    _verified_provider_call_journal_binding,
    _write_execution_failure_receipt,
    _write_stage_receipt,
    verify_paid_provenance,
)
from verified_memory.scientific_release_attestation import (
    SCIENTIFIC_RELEASE_ATTESTATION_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2.yaml"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _launch_input(contract) -> dict:
    payload = {
        "schema_version": PILOT_SCIENTIFIC_LAUNCH_INPUT_SCHEMA_VERSION,
        "contract_sha256": contract.canonical_hash,
        "ci_run_selection": {
            "run_id": 101,
            "run_attempt": 1,
            "jobs": [
                {
                    "name": "Python 3.12.7 / ubuntu-24.04",
                    "database_id": 201,
                },
                {
                    "name": "Python 3.12.7 / macos-14",
                    "database_id": 202,
                },
            ],
        },
        "contract_binding": {
            "contract_canonical_sha256": contract.canonical_hash,
        },
    }
    return {
        **payload,
        "launch_input_sha256": canonical_sha256(payload),
    }


def _paid(contract) -> GitProvenance:
    commit = "1" * 40
    payload = {
        "schema_version": SCIENTIFIC_RELEASE_ATTESTATION_SCHEMA_VERSION,
        "status": "pass",
        "head_commit": commit,
    }
    return GitProvenance(
        git_tag="pilot-v2-science",
        head_commit=commit,
        tag_commit=commit,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding=contract.validate_provenance(commit, "pilot-v2-science"),
        release_attestation={
            **payload,
            "attestation_sha256": canonical_sha256(payload),
        },
    )


def test_v2_release_requires_explicit_self_hashed_launch_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    with pytest.raises(PilotOrchestrationError, match="explicit scientific"):
        verify_paid_provenance(contract, repo_root=tmp_path)

    launch = _launch_input(contract)
    launch["ci_run_selection"]["run_id"] = 999
    path = tmp_path / "scientific_launch_input.json"
    _write_json(path, launch)
    with pytest.raises(PilotOrchestrationError, match="self-hash"):
        verify_paid_provenance(
            contract,
            repo_root=tmp_path,
            scientific_launch_input_path=path,
        )


def test_v2_release_dispatches_exact_launch_selection_to_science_attestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    launch = _launch_input(contract)
    path = tmp_path / "scientific_launch_input.json"
    _write_json(path, launch)
    commit = "1" * 40

    def fake_git(_root, *args, **_kwargs):
        if args[:2] == ("cat-file", "-t"):
            return "tag"
        if args[:2] == ("status", "--porcelain"):
            return ""
        if args[:2] == ("rev-parse", "HEAD"):
            return commit
        if args[0] == "rev-parse":
            return commit
        raise AssertionError(args)

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._git",
        fake_git,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    observed: dict = {}

    def attest(_root, **kwargs):
        observed.update(kwargs)
        payload = {
            "schema_version": SCIENTIFIC_RELEASE_ATTESTATION_SCHEMA_VERSION,
            "status": "pass",
            "head_commit": commit,
        }
        return SimpleNamespace(
            to_dict=lambda: {
                **payload,
                "attestation_sha256": canonical_sha256(payload),
            }
        )

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator." "verify_scientific_release_attestation",
        attest,
    )
    paid = verify_paid_provenance(
        contract,
        repo_root=tmp_path,
        scientific_launch_input_path=path,
    )
    assert paid.head_commit == commit
    assert observed["ci_run_selection"] == launch["ci_run_selection"]
    assert observed["contract_binding"] == launch["contract_binding"]
    assert observed["release_requirements"] == (contract.release_requirements.to_dict())


def test_release_receipt_persistence_accepts_v2_and_rejects_drift(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid(contract)
    path = _persist_release_attestation(tmp_path, paid)
    assert path.is_file()
    assert _persist_release_attestation(tmp_path, paid) == path

    drifted_payload = dict(paid.release_attestation)
    drifted_payload["attestation_sha256"] = "0" * 64
    drifted = GitProvenance(
        **{
            **paid.to_dict(),
            "release_attestation": drifted_payload,
        }
    )
    with pytest.raises(PilotOrchestrationError, match="content hash"):
        _persist_release_attestation(tmp_path, drifted)


def test_journal_binding_requires_terminal_hash_chain_and_exact_identity(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    unsigned = {
        "schema_version": "finevo-provider-call-journal-v1",
        "run_id": "v2-journal-test",
        "contract_hash": contract.canonical_hash,
        "events": [],
    }
    journal = {
        **unsigned,
        "journal_sha256": canonical_sha256(unsigned),
    }
    path = tmp_path / "provider-call-journal.json"
    _write_json(path, journal)
    binding = _verified_provider_call_journal_binding(
        path,
        expected_run_id="v2-journal-test",
        expected_contract_hash=contract.canonical_hash,
    )
    assert binding["journal_sha256"] == journal["journal_sha256"]
    assert binding["file_sha256"] != binding["journal_sha256"]
    assert binding["terminal_dispositions_verified"] is True

    journal["run_id"] = "tampered"
    journal["journal_sha256"] = canonical_sha256(
        {key: value for key, value in journal.items() if key != "journal_sha256"}
    )
    _write_json(path, journal)
    with pytest.raises(Exception, match="run binding mismatch"):
        _verified_provider_call_journal_binding(
            path,
            expected_run_id="v2-journal-test",
            expected_contract_hash=contract.canonical_hash,
        )


def test_v2_preflight_keeps_record_and_skip_and_failure_receipt_binds_journal(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = _paid(contract)
    spec = contract.expand(
        stage="closed-loop-preflight",
        model="gpt52_main",
    )[0]
    config = _preflight_config(contract, spec, paid=paid)
    assert config.semantic_parse_failure_policy == "record-and-skip"
    assert config.pilot_contract_hash == contract.canonical_hash

    run_dir = tmp_path / spec.stage_id / "runs" / spec.run_id
    journal_path = (
        tmp_path
        / spec.stage_id
        / "provider_call_journals"
        / f"{spec.run_id}--preflight.json"
    )
    unsigned = {
        "schema_version": "finevo-provider-call-journal-v1",
        "run_id": config.run_id,
        "contract_hash": contract.canonical_hash,
        "events": [],
    }
    _write_json(
        journal_path,
        {
            **unsigned,
            "journal_sha256": canonical_sha256(unsigned),
        },
    )
    manifest = _write_execution_failure_receipt(
        run_dir / "failure_receipt",
        scope=f"finevo-pilot/{spec.stage_id}/{spec.execution_mode}",
        error=RuntimeError("fixture preflight failure"),
        contract=contract,
        projection=RunProjection(
            run_id=spec.run_id,
            stage_bucket=spec.budget_bucket,
            cost_usd=0.0,
            completions=16,
            storage_bytes=1_000,
            basis={"method": "v2-preflight-failure-fixture"},
        ),
        budget=RunBudget(
            BudgetLimits(max_calls=16, max_cost_usd=1.0),
            budget_id=f"{spec.run_id}-budget",
        ),
        specs=(spec,),
        paid=paid,
        diagnostic=False,
    )
    failure = json.loads((manifest.parent / "failure.json").read_text(encoding="utf-8"))
    bindings = failure["config"]["provider_call_journals"]
    assert len(bindings) == 1
    assert bindings[0]["run_id"] == config.run_id
    assert bindings[0]["contract_hash"] == contract.canonical_hash


def test_v2_gate_resolution_and_model_specific_no_go_propagation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    assert _preflight_stage_for_model(contract, "gpt52_main") == (
        "closed-loop-preflight"
    )
    assert (
        _preflight_stage_for_model(contract, "gemini35_flash_diagnostic")
        == "secondary-closed-loop-preflight"
    )
    assert _cross_model_science_stage_ids(contract) == (
        "controlled-second",
        "cross-model-diagnostics",
    )
    assert _scientific_stage_ids(contract) == (
        "stage0-calibration",
        "experiment-a",
        "experiment-c",
        "experiment-d",
        "experiment-b",
        "controlled-second",
        "cross-model-diagnostics",
    )

    ledger = PilotRunLedger(
        tmp_path / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._v2_capability_semantic_go",
        lambda _contract, spec, row, **_kwargs: (
            spec.model_id == "gpt52_main" and row.get("status") == "complete"
        ),
    )
    ledger.register(contract.expand())
    for spec in contract.expand(stage="capability-gate"):
        ledger.finalize(
            spec.run_id,
            status=(
                "complete" if spec.model_id == "gpt52_main" else "capability-no-go"
            ),
            artifact=None,
        )
    _propagate_capability_no_go(
        contract,
        ledger=ledger,
        source_stage="capability-gate",
    )
    local_preflight = contract.expand(
        stage="closed-loop-preflight",
        model="llama33_local_controlled",
    )[0]
    controlled = contract.expand(stage="controlled-second")
    primary_preflight = contract.expand(
        stage="closed-loop-preflight",
        model="gpt52_main",
    )[0]
    assert ledger.status(local_preflight.run_id) == "capability-no-go"
    assert {ledger.status(spec.run_id) for spec in controlled} == {"capability-no-go"}
    assert ledger.status(primary_preflight.run_id) == "scheduled"
    assert {
        ledger.status(spec.run_id)
        for spec in contract.expand(stage="secondary-capability-gate")
    } == {"scheduled"}

    receipt_path = _write_stage_receipt(
        contract,
        "capability-gate",
        raw_root=tmp_path,
        ledger=ledger,
        status="complete-with-no-go",
        go_models=("gpt52_main",),
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["go_models"] == ["gpt52_main"]
    assert receipt["execution_progression_go"] is True
    _assert_prerequisites(
        contract,
        "closed-loop-preflight",
        raw_root=tmp_path,
    )


def test_v2_full_science_projection_includes_both_cross_model_scopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    ledger = PilotRunLedger(
        tmp_path / "run_ledger.json",
        contract_hash=contract.canonical_hash,
    )
    ledger.register(contract.expand())
    for stage_id in (
        "closed-loop-preflight",
        "secondary-closed-loop-preflight",
    ):
        for spec in contract.expand(stage=stage_id):
            ledger.finalize(spec.run_id, status="complete", artifact=None)

    def projection(_contract, spec, **_kwargs):
        return RunProjection(
            run_id=spec.run_id,
            stage_bucket=spec.budget_bucket,
            cost_usd=0.0,
            completions=0,
            storage_bytes=1,
            basis={"method": "v2-projection-fixture"},
        )

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.projection_from_preflight",
        projection,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._remaining_core_projections",
        lambda *args, **kwargs: (),
    )
    projections = _remaining_scientific_projections(
        contract,
        raw_root=tmp_path,
        paid=_paid(contract),
        run_ledger=ledger,
    )
    projected = {row.run_id for row in projections}
    expected = {
        spec.run_id
        for stage_id in (
            "stage0-calibration",
            "controlled-second",
            "cross-model-diagnostics",
        )
        for spec in contract.expand(stage=stage_id)
    }
    assert projected == expected
    assert {row.stage_bucket for row in projections} == {"calibration", "cross_model"}


def test_v2_run_ledger_rejects_stale_and_rehashed_row_tampering(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    path = tmp_path / "run_ledger.json"
    ledger = PilotRunLedger(
        path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    specs = contract.expand(stage="experiment-a")[:2]
    ledger.register(specs)
    original = json.loads(path.read_text(encoding="utf-8"))

    stale = json.loads(json.dumps(original))
    stale["updated_at"] = "tampered"
    _write_json(path, stale)
    with pytest.raises(PilotOrchestrationError, match="self-hash"):
        PilotRunLedger(
            path,
            contract_hash=contract.canonical_hash,
            tamper_evident=True,
        )

    rehashed = json.loads(json.dumps(original))
    rehashed["runs"][specs[0].run_id]["status"] = "complete"
    unsigned = dict(rehashed)
    unsigned.pop("ledger_sha256")
    rehashed["ledger_sha256"] = canonical_sha256(unsigned)
    _write_json(path, rehashed)
    with pytest.raises(PilotOrchestrationError, match="event head"):
        PilotRunLedger(
            path,
            contract_hash=contract.canonical_hash,
            tamper_evident=True,
        )


def test_v2_run_ledger_resume_is_event_idempotent(tmp_path: Path) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    path = tmp_path / "run_ledger.json"
    spec = contract.expand(stage="experiment-a")[0]
    ledger = PilotRunLedger(
        path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register((spec,))
    after_register = ledger.snapshot()
    ledger.register((spec,))
    assert ledger.snapshot()["events"] == after_register["events"]

    ledger.finalize(spec.run_id, status="failed", artifact=None)
    after_finalize = ledger.snapshot()
    ledger.finalize(spec.run_id, status="failed", artifact=None)
    assert ledger.snapshot()["events"] == after_finalize["events"]


def test_v2_rehashed_stage_receipt_go_models_tamper_fails_semantic_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    ledger = PilotRunLedger(
        tmp_path / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._v2_capability_semantic_go",
        lambda _contract, spec, row, **_kwargs: (
            spec.model_id == "gpt52_main" and row.get("status") == "complete"
        ),
    )
    for spec in contract.expand(stage="capability-gate"):
        ledger.finalize(
            spec.run_id,
            status=(
                "complete" if spec.model_id == "gpt52_main" else "capability-no-go"
            ),
            artifact=None,
        )
    receipt_path = _write_stage_receipt(
        contract,
        "capability-gate",
        raw_root=tmp_path,
        ledger=ledger,
        status="complete-with-no-go",
        go_models=("gpt52_main",),
    )
    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["go_models"] = [
        "gpt52_main",
        "llama33_local_controlled",
    ]
    unsigned = dict(tampered)
    unsigned.pop("integrity")
    tampered["integrity"]["content_sha256"] = canonical_sha256(unsigned)
    receipt_path.chmod(0o644)
    _write_json(receipt_path, tampered)

    with pytest.raises(PilotOrchestrationError, match="go_models"):
        _assert_prerequisites(
            contract,
            "closed-loop-preflight",
            raw_root=tmp_path,
            ledger=ledger,
        )


def test_v2_capability_gate_is_recomputed_from_terminal_and_source_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(
        stage="capability-gate",
        model="gpt52_main",
    )[0]
    capability = {
        "schema_version": "finevo-capability-gate-v3",
        "contract_sha256": contract.canonical_hash,
        "run_spec": spec.to_dict(),
        "pass": True,
        "preflight_go": False,
        "interface_gate": {"pass": True},
        "capability_assessment": {"status": "pass"},
    }
    gate = {
        "capability_pass": True,
        "capability_status": "pass",
        "interface_pass": True,
        "preflight_run": None,
        "go": True,
        "reason": None,
    }
    run_dir = tmp_path / spec.stage_id / "runs" / spec.run_id
    _write_json(run_dir / "capability.json", capability)
    _write_json(run_dir / "gate_receipt.json", gate)
    paid = _paid(contract)
    terminal = write_terminal_summary(
        tmp_path / spec.stage_id / "summaries" / f"{spec.run_id}.json",
        contract=contract,
        run_spec=spec,
        resolved_git_commit=paid.head_commit,
        git_tag=paid.git_tag,
        payload={
            "metrics": {},
            "gate_evidence": gate,
            "capability": capability,
        },
        scientific_evidence=False,
        diagnostic_only=False,
        evidence_scope="preregistered_task_capability_gate",
    )
    ledger = PilotRunLedger(
        tmp_path / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register((spec,))
    ledger.finalize(spec.run_id, status="complete", artifact=str(terminal))
    row = ledger.snapshot()["runs"][spec.run_id]
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._validate_capability_v3",
        lambda _value: None,
    )
    assert _v2_capability_semantic_go(
        contract,
        spec,
        row,
        raw_root=tmp_path,
        paid=paid,
    )

    _write_json(run_dir / "gate_receipt.json", {**gate, "go": False})
    with pytest.raises(PilotOrchestrationError, match="terminal gate"):
        _v2_capability_semantic_go(
            contract,
            spec,
            row,
            raw_root=tmp_path,
            paid=paid,
        )


def test_v2_stage_receipt_recomputes_bound_artifact_hashes(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    ledger = PilotRunLedger(
        tmp_path / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())
    artifacts: list[Path] = []
    for spec in contract.expand(stage="experiment-a"):
        artifact = tmp_path / "experiment-a" / "fixture-artifacts" / f"{spec.run_id}.json"
        _write_json(artifact, {"run_id": spec.run_id, "observed": True})
        artifacts.append(artifact)
        ledger.finalize(
            spec.run_id,
            status="complete",
            artifact=str(artifact),
        )
    _write_stage_receipt(
        contract,
        "experiment-a",
        raw_root=tmp_path,
        ledger=ledger,
        status="complete",
    )

    _write_json(
        artifacts[0],
        {"run_id": contract.expand(stage="experiment-a")[0].run_id, "observed": False},
    )
    with pytest.raises(PilotOrchestrationError, match="source binding"):
        _verify_v2_stage_receipt(
            contract,
            "experiment-a",
            json.loads(
                (
                    tmp_path / "experiment-a" / "stage_receipt.json"
                ).read_text(encoding="utf-8")
            ),
            raw_root=tmp_path,
            ledger=ledger,
            paid=None,
        )


def test_v2_c_sensitivity_reads_c_full_before_later_experiment_b(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator." "_load_verified_stage0_selection",
        lambda *args, **kwargs: {
            "absolute_flow_utility_threshold": {"value": 0.0},
            "integrity": {"content_sha256": "a" * 64},
        },
    )
    observed: list[Path] = []

    def stop_at_first_source(run_dir: Path):
        observed.append(Path(run_dir))
        raise RuntimeError("source probe")

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.verify_manifest",
        stop_at_first_source,
    )
    with pytest.raises(RuntimeError, match="source probe"):
        _build_experiment_c_sensitivity(
            contract,
            raw_root=tmp_path,
            git_tag="pilot-v2-science",
            git_commit="1" * 40,
        )
    assert len(observed) == 1
    assert observed[0].parts[-3] == "experiment-c"
    assert "--experiment-c--" in observed[0].name
