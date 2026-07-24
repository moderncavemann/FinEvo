from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from verified_memory.pilot_amendment import (
    GPT52_PROFILE_ID,
    LLAMA33_PROFILE_ID,
    PARENT_DEBIT_RECORD_SHA256,
)
from verified_memory.pilot_contract import (
    PilotContract,
    canonical_sha256,
    load_pilot_contract,
)
from verified_memory.pilot_evaluation_amendment import (
    CAPABILITY_IMPORT_SCHEMA_VERSION,
    EVALUATOR_CORRECTION_RECEIPT_RELATIVE_PATH,
    PilotEvaluationAmendmentError,
    V22_PARENT_DEBIT_RECORD_SHA256,
    evaluator_amendment_control_path,
    validate_capability_import,
)
from verified_memory.pilot_orchestrator import (
    GitProvenance,
    PilotOrchestrationError,
    execute_stage,
)
from verified_memory.scientific_release_attestation import (
    SCIENTIFIC_RELEASE_ATTESTATION_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
V21_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_1.yaml"
V22_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_2_overlay.yaml"
V22_TAG = "pilot-v2.2-science"
TEST_COMMIT = "1" * 40


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_tampered_json(path: Path, value: dict[str, Any]) -> None:
    path.chmod(0o600)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _all_file_hashes(raw_root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(raw_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(raw_root.rglob("*"))
        if path.is_file() and path.name != ".real-stage-execution.lock"
    }


def _draft_provenance_binding(contract: PilotContract) -> dict[str, Any]:
    return {
        "git_tag": V22_TAG,
        "resolved_git_commit": TEST_COMMIT,
        "commit_resolution": contract.implementation["commit_resolution"],
        "p0_base_commit": contract.implementation["p0_base_commit"],
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
    }


def _paid(contract: PilotContract, *, git_tag: str) -> GitProvenance:
    binding = (
        _draft_provenance_binding(contract)
        if contract.status == "draft"
        else contract.validate_provenance(TEST_COMMIT, git_tag)
    )
    attestation = {
        "schema_version": SCIENTIFIC_RELEASE_ATTESTATION_SCHEMA_VERSION,
        "status": "pass",
        "head_commit": TEST_COMMIT,
    }
    return GitProvenance(
        git_tag=git_tag,
        head_commit=TEST_COMMIT,
        tag_commit=TEST_COMMIT,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding=binding,
        release_attestation={
            **attestation,
            "attestation_sha256": canonical_sha256(attestation),
        },
    )


def _install_v22_paid_zero_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    contract: PilotContract,
) -> tuple[GitProvenance, dict[str, list[Any]]]:
    paid = _paid(contract, git_tag=V22_TAG)
    calls: dict[str, list[Any]] = {
        "paid_provenance": [],
        "provider": [],
        "catalog": [],
        "capability_runner": [],
        "budget_reservation": [],
        "budget_finalization": [],
    }

    def validate_draft_provenance(
        observed: PilotContract,
        resolved_commit: str,
        git_tag: str,
    ) -> dict[str, Any]:
        assert observed.contract_id == contract.contract_id
        assert observed.canonical_hash == contract.canonical_hash
        assert resolved_commit == TEST_COMMIT
        assert git_tag == V22_TAG
        return _draft_provenance_binding(observed)

    def paid_provenance(
        observed: PilotContract,
        *,
        repo_root: Path,
        scientific_launch_input_path: Path,
    ) -> GitProvenance:
        calls["paid_provenance"].append(
            {
                "contract": observed,
                "repo_root": repo_root,
                "scientific_launch_input_path": scientific_launch_input_path,
            }
        )
        assert observed.contract_id == contract.contract_id
        assert observed.canonical_hash == contract.canonical_hash
        assert Path(repo_root).resolve() == ROOT
        return paid

    def provider_dispatch(*args: Any, **kwargs: Any) -> None:
        calls["provider"].append((args, kwargs))
        pytest.fail("V2.2 evaluator import attempted provider construction")

    def catalog_dispatch(*args: Any, **kwargs: Any) -> None:
        calls["catalog"].append((args, kwargs))
        pytest.fail("V2.2 evaluator import attempted a live catalog request")

    def capability_runner(*args: Any, **kwargs: Any) -> None:
        calls["capability_runner"].append((args, kwargs))
        pytest.fail("V2.2 evaluator import attempted capability dispatch")

    def reserve_budget(*args: Any, **kwargs: Any) -> None:
        calls["budget_reservation"].append((args, kwargs))
        pytest.fail("V2.2 evaluator import attempted a new budget reservation")

    def finalize_budget(*args: Any, **kwargs: Any) -> None:
        calls["budget_finalization"].append((args, kwargs))
        pytest.fail("V2.2 evaluator import attempted to record current usage")

    monkeypatch.setattr(
        PilotContract,
        "validate_provenance",
        validate_draft_provenance,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.verify_paid_provenance",
        paid_provenance,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._provider_for_profile",
        provider_dispatch,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.validate_live_provider_catalog",
        catalog_dispatch,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._execute_capability_preflight",
        capability_runner,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.PilotBudgetLedger.reserve",
        reserve_budget,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.PilotBudgetLedger.finalize",
        finalize_budget,
    )
    return paid, calls


def _run_v22(
    raw_root: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    return execute_stage(
        contract_path=V22_CONTRACT_PATH,
        stage_id="capability-gate",
        resume=resume,
        raw_root=raw_root,
        repo_root=ROOT,
    )


def _assert_no_current_attempt_dispatch(calls: dict[str, list[Any]]) -> None:
    assert calls["provider"] == []
    assert calls["catalog"] == []
    assert calls["capability_runner"] == []
    assert calls["budget_reservation"] == []
    assert calls["budget_finalization"] == []


def test_v22_paid_capability_import_is_zero_dispatch_and_budget_neutral(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(V22_CONTRACT_PATH)
    paid, calls = _install_v22_paid_zero_dispatch(monkeypatch, contract)
    raw_root = tmp_path / "raw"

    receipt = _run_v22(raw_root, resume=False)

    _assert_no_current_attempt_dispatch(calls)
    assert len(calls["paid_provenance"]) == 1
    assert calls["paid_provenance"][0]["scientific_launch_input_path"] == (
        raw_root.resolve() / "scientific_launch_input.json"
    )
    assert receipt["status"] == "complete"
    assert receipt["go"] is True
    assert receipt["execution_progression_go"] is True
    assert receipt["denominator_terminal"] is True
    assert receipt["complete_cell_count"] == 2
    assert receipt["hard_stop_cell_count"] == 0
    assert receipt["registered_run_count"] == 2
    assert len(contract.expand()) == 174
    assert receipt["status_counts"] == {"complete": 2}
    assert receipt["go_models"] == [GPT52_PROFILE_ID, LLAMA33_PROFILE_ID]
    assert receipt["artifacts"]["provider_calls_current_attempt"] == 0
    assert receipt["artifacts"]["imported_model_count"] == 2

    release = _json(raw_root / "release_attestation.json")
    assert release == paid.release_attestation

    tracked_amendment = ROOT / EVALUATOR_CORRECTION_RECEIPT_RELATIVE_PATH
    persisted_amendment = evaluator_amendment_control_path(raw_root=raw_root)
    assert persisted_amendment.read_bytes() == tracked_amendment.read_bytes()
    amendment = _json(persisted_amendment)
    assert receipt["artifacts"]["evaluator_amendment_receipt"] == str(
        persisted_amendment
    )

    expected_control_paths = {persisted_amendment.resolve()}
    ledger = _json(raw_root / "run_ledger.json")
    assert len(ledger["runs"]) == 174
    for spec in contract.expand(stage="capability-gate"):
        row = ledger["runs"][spec.run_id]
        assert row["status"] == "complete"
        assert row["failure"] is None

        run_dir = raw_root / spec.stage_id / "runs" / spec.run_id
        capability_path = run_dir / "capability.json"
        gate_path = run_dir / "gate_receipt.json"
        terminal_path = (
            raw_root / spec.stage_id / "summaries" / f"{spec.run_id}.json"
        )
        expected_control_paths.update(
            {
                capability_path.resolve(),
                gate_path.resolve(),
                terminal_path.resolve(),
            }
        )
        capability = _json(capability_path)
        validate_capability_import(capability, contract, spec, amendment)
        assert capability["schema_version"] == CAPABILITY_IMPORT_SCHEMA_VERSION
        assert capability["pass"] is True
        assert capability["provider_calls_current_attempt"] == 0
        gate = _json(gate_path)
        assert gate["go"] is True
        assert gate["provider_calls_current_attempt"] == 0
        assert gate["evaluator_amendment_receipt"] == str(persisted_amendment)
        terminal = _json(terminal_path)
        assert terminal["scientific_evidence"] is False
        assert terminal["diagnostic_only"] is False
        assert terminal["evidence_scope"] == (
            "preregistered_task_capability_gate"
        )
        assert terminal["payload"]["capability"] == capability
        assert terminal["payload"]["gate_evidence"] == gate

    source_bindings = receipt["bindings"]["source_files"]
    assert {Path(row["path"]).resolve() for row in source_bindings} == (
        expected_control_paths
    )
    for row in source_bindings:
        path = Path(row["path"])
        assert row["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    budget = _json(raw_root / "budget_ledger.json")
    assert budget["runs"] == {}
    assert budget["parent_debit"]["record_sha256"] == (
        V22_PARENT_DEBIT_RECORD_SHA256
    )
    assert budget["parent_debit"]["cost_usd"] == pytest.approx(1.53775475)
    assert budget["parent_debit"]["hosted_completions"] == 60
    assert budget["parent_debit"]["storage_bytes"] == 715_860
    assert [row["event_type"] for row in budget["events"]] == [
        "genesis",
        "parent_debit_imported",
    ]
    assert not (raw_root / "capability-gate" / "provider_catalog").exists()
    assert list(raw_root.rglob("*provider_call_journal*.json")) == []


def test_v22_paid_capability_import_resume_is_byte_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(V22_CONTRACT_PATH)
    _, calls = _install_v22_paid_zero_dispatch(monkeypatch, contract)
    raw_root = tmp_path / "raw"

    first = _run_v22(raw_root, resume=False)
    first_hashes = _all_file_hashes(raw_root)
    second = _run_v22(raw_root, resume=True)
    second_hashes = _all_file_hashes(raw_root)

    assert second == first
    assert second_hashes == first_hashes
    assert len(calls["paid_provenance"]) == 2
    _assert_no_current_attempt_dispatch(calls)
    budget = _json(raw_root / "budget_ledger.json")
    assert budget["runs"] == {}
    assert [row["event_type"] for row in budget["events"]] == [
        "genesis",
        "parent_debit_imported",
    ]
    capability_events = [
        row
        for row in _json(raw_root / "run_ledger.json")["events"]
        if row["event_type"] == "run_finalized"
        and "capability-gate" in row["payload"]["run_id"]
    ]
    assert len(capability_events) == 2


def test_v22_resume_rejects_tampered_persisted_amendment_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(V22_CONTRACT_PATH)
    _, calls = _install_v22_paid_zero_dispatch(monkeypatch, contract)
    raw_root = tmp_path / "raw"
    _run_v22(raw_root, resume=False)
    ledger_hashes = {
        name: hashlib.sha256((raw_root / name).read_bytes()).hexdigest()
        for name in ("run_ledger.json", "budget_ledger.json")
    }
    stage_receipt_hash = hashlib.sha256(
        (raw_root / "capability-gate" / "stage_receipt.json").read_bytes()
    ).hexdigest()

    amendment_path = evaluator_amendment_control_path(raw_root=raw_root)
    tampered = _json(amendment_path)
    tampered["status"] = "tampered"
    _write_tampered_json(amendment_path, tampered)

    with pytest.raises(
        PilotEvaluationAmendmentError,
        match="persisted evaluator amendment receipt differs from tracked source",
    ):
        _run_v22(raw_root, resume=True)

    _assert_no_current_attempt_dispatch(calls)
    assert {
        name: hashlib.sha256((raw_root / name).read_bytes()).hexdigest()
        for name in ("run_ledger.json", "budget_ledger.json")
    } == ledger_hashes
    assert hashlib.sha256(
        (raw_root / "capability-gate" / "stage_receipt.json").read_bytes()
    ).hexdigest() == stage_receipt_hash


def test_v22_resume_rejects_fully_rehashed_import_tamper_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(V22_CONTRACT_PATH)
    _, calls = _install_v22_paid_zero_dispatch(monkeypatch, contract)
    raw_root = tmp_path / "raw"
    _run_v22(raw_root, resume=False)
    spec = contract.expand(
        stage="capability-gate",
        model=GPT52_PROFILE_ID,
    )[0]
    run_dir = raw_root / spec.stage_id / "runs" / spec.run_id
    capability_path = run_dir / "capability.json"
    terminal_path = (
        raw_root / spec.stage_id / "summaries" / f"{spec.run_id}.json"
    )

    capability = _json(capability_path)
    capability["provider_calls_current_attempt"] = 1
    capability.pop("integrity")
    capability["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(capability),
    }
    _write_tampered_json(capability_path, capability)

    terminal = _json(terminal_path)
    terminal["payload"]["capability"] = capability
    terminal.pop("integrity")
    terminal["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(terminal),
    }
    _write_tampered_json(terminal_path, terminal)

    control_hashes = {
        name: hashlib.sha256((raw_root / name).read_bytes()).hexdigest()
        for name in (
            "run_ledger.json",
            "budget_ledger.json",
            "evaluator_amendment_receipt.json",
        )
    }
    stage_receipt_hash = hashlib.sha256(
        (raw_root / "capability-gate" / "stage_receipt.json").read_bytes()
    ).hexdigest()

    with pytest.raises(
        PilotOrchestrationError,
        match="capability evidence failed semantic recomputation",
    ):
        _run_v22(raw_root, resume=True)

    _assert_no_current_attempt_dispatch(calls)
    assert {
        name: hashlib.sha256((raw_root / name).read_bytes()).hexdigest()
        for name in (
            "run_ledger.json",
            "budget_ledger.json",
            "evaluator_amendment_receipt.json",
        )
    } == control_hashes
    assert hashlib.sha256(
        (raw_root / "capability-gate" / "stage_receipt.json").read_bytes()
    ).hexdigest() == stage_receipt_hash


def test_v21_capability_no_go_does_not_enter_v22_import_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(V21_CONTRACT_PATH)
    paid = _paid(contract, git_tag="pilot-v2.1-science")
    raw_root = tmp_path / "raw"
    provider_attempts: list[str] = []
    catalog_attempts: list[Any] = []
    evaluator_attempts: list[Any] = []
    budget_reservations: list[Any] = []

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.verify_paid_provenance",
        lambda *_args, **_kwargs: paid,
    )

    def reject_hosted_credential(profile: Any, **_kwargs: Any) -> None:
        provider_attempts.append(profile.profile_id)
        raise ValueError("credential rejected fixture")

    def forbidden_catalog(*args: Any, **kwargs: Any) -> None:
        catalog_attempts.append((args, kwargs))
        pytest.fail("V2.1 catalog ran after local credential rejection")

    def forbidden_evaluator_import(*args: Any, **kwargs: Any) -> None:
        evaluator_attempts.append((args, kwargs))
        pytest.fail("V2.1 entered the V2.2 evaluator import path")

    def forbidden_reservation(*args: Any, **kwargs: Any) -> None:
        budget_reservations.append((args, kwargs))
        pytest.fail("V2.1 no-go attempted a budget reservation")

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._provider_for_profile",
        reject_hosted_credential,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.validate_live_provider_catalog",
        forbidden_catalog,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator."
        "_execute_evaluator_capability_import_stage",
        forbidden_evaluator_import,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.PilotBudgetLedger.reserve",
        forbidden_reservation,
    )

    receipt = execute_stage(
        contract_path=V21_CONTRACT_PATH,
        stage_id="capability-gate",
        resume=False,
        raw_root=raw_root,
        repo_root=ROOT,
    )

    assert provider_attempts == [GPT52_PROFILE_ID]
    assert catalog_attempts == []
    assert evaluator_attempts == []
    assert budget_reservations == []
    assert receipt["status"] == "complete-with-no-go"
    assert receipt["go"] is False
    assert receipt["execution_progression_go"] is False
    assert receipt["go_models"] == []
    assert receipt["status_counts"] == {"capability-no-go": 2}

    ledger = _json(raw_root / "run_ledger.json")
    gpt_spec = contract.expand(
        stage="capability-gate",
        model=GPT52_PROFILE_ID,
    )[0]
    local_spec = contract.expand(
        stage="capability-gate",
        model=LLAMA33_PROFILE_ID,
    )[0]
    assert ledger["runs"][gpt_spec.run_id]["status"] == "capability-no-go"
    assert ledger["runs"][gpt_spec.run_id]["failure"]["error_type"] == (
        "ValueError"
    )
    assert ledger["runs"][local_spec.run_id]["status"] == "capability-no-go"
    assert ledger["runs"][local_spec.run_id]["failure"]["error_type"] == (
        "InheritedCapabilityNoGo"
    )
    assert ledger["runs"][local_spec.run_id]["failure"][
        "provider_calls_current_attempt"
    ] == 0

    budget = _json(raw_root / "budget_ledger.json")
    assert budget["runs"] == {}
    assert budget["parent_debit"]["record_sha256"] == PARENT_DEBIT_RECORD_SHA256
    assert (raw_root / "operational_amendment_receipt.json").is_file()
    assert not evaluator_amendment_control_path(raw_root=raw_root).exists()
