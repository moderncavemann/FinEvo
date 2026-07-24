from __future__ import annotations

import json
from pathlib import Path

import pytest

from verified_memory.pilot_amendment import (
    EXECUTION_INVARIANTS_BASE_SHA256,
    GPT52_PROFILE_ID,
    LLAMA33_PROFILE_ID,
    PARENT_DEBIT_RECORD_SHA256,
    SCIENCE_DESIGN_BASE_SHA256,
    PilotAmendmentError,
    apply_inherited_capability_no_go,
    assert_amended_capability_dispatch_scope,
    execution_invariants_sha256,
    load_operational_failure_receipt,
    parent_budget_debit_for_contract,
    persist_operational_failure_receipt,
    science_design_sha256,
)
from verified_memory.pilot_budget import (
    ParentBudgetDebit,
    PilotBudgetError,
    PilotBudgetLedger,
)
from verified_memory.pilot_contract import canonical_sha256, load_pilot_contract
from verified_memory.pilot_orchestrator import (
    GitProvenance,
    PilotRunLedger,
    _budget_caps,
    _execute_stage_locked,
    _v2_stage_control_paths,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_1.yaml"
BASE_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2.yaml"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_v21_receipt_binds_contract_science_and_parent_debit() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    receipt, path = load_operational_failure_receipt(
        repo_root=ROOT,
        contract=contract,
    )
    debit = parent_budget_debit_for_contract(contract)

    assert path == ROOT / "experiments" / "pilot_v2_failure_receipt.json"
    assert receipt["operational_amendment"] == contract.to_dict()[
        "operational_amendment"
    ]
    assert science_design_sha256(contract.to_dict()) == (
        SCIENCE_DESIGN_BASE_SHA256
    )
    assert execution_invariants_sha256(contract.to_dict()) == (
        EXECUTION_INVARIANTS_BASE_SHA256
    )
    assert debit is not None
    assert debit.record_sha256 == PARENT_DEBIT_RECORD_SHA256
    freeze = receipt["science_freeze"]
    assert freeze["science_design_field_changes"] == []
    assert freeze["operational_budget_cap_changes"] == [
        {
            "path": "budgets.stage_usd_caps.capability",
            "parent_value": 2.0,
            "amended_value": 3.0701145,
        },
        {
            "path": "budgets.stage_usd_caps.cross_model",
            "parent_value": 6.0,
            "amended_value": 4.9298855,
        },
    ]
    assert freeze["scientific_effect_outcomes_inspected_for_retry"] is False
    assert freeze["capability_gate_outcomes_inspected"] is True


def test_v21_rehashed_receipt_drift_is_rejected(tmp_path: Path) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    receipt = json.loads(
        (ROOT / "experiments" / "pilot_v2_failure_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    receipt["operational_amendment"]["retry_policy"]["eligible_model_ids"] = [
        LLAMA33_PROFILE_ID
    ]
    unsigned = dict(receipt)
    unsigned.pop("integrity")
    receipt["integrity"]["content_sha256"] = canonical_sha256(unsigned)
    _write_json(
        tmp_path / "experiments" / "pilot_v2_failure_receipt.json",
        receipt,
    )

    with pytest.raises(PilotAmendmentError, match="differs from the loaded"):
        load_operational_failure_receipt(
            repo_root=tmp_path,
            contract=contract,
        )


def test_v21_receipt_persistence_is_exact_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    _, persisted = persist_operational_failure_receipt(
        repo_root=ROOT,
        raw_root=tmp_path,
        contract=contract,
    )
    tracked = ROOT / "experiments" / "pilot_v2_failure_receipt.json"
    assert persisted.read_bytes() == tracked.read_bytes()

    persisted.write_text("{}", encoding="utf-8")
    with pytest.raises(PilotAmendmentError, match="differs from tracked"):
        persist_operational_failure_receipt(
            repo_root=ROOT,
            raw_root=tmp_path,
            contract=contract,
        )


def test_v21_receipt_persistence_reuses_the_validated_source_buffer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    source = ROOT / "experiments" / "pilot_v2_failure_receipt.json"
    source_bytes = source.read_bytes()
    repo = tmp_path / "repo"
    copied = repo / "experiments" / "pilot_v2_failure_receipt.json"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(source_bytes)
    raw = tmp_path / "raw"

    from verified_memory import pilot_amendment

    real_read = pilot_amendment._read_regular_bytes_once

    def mutate_after_single_read(path: Path, *, name: str) -> bytes:
        observed = real_read(path, name=name)
        if path == copied:
            copied.write_bytes(b"{}")
        return observed

    monkeypatch.setattr(
        pilot_amendment,
        "_read_regular_bytes_once",
        mutate_after_single_read,
    )
    _, persisted = persist_operational_failure_receipt(
        repo_root=repo,
        raw_root=raw,
        contract=contract,
    )

    assert copied.read_bytes() == b"{}"
    assert persisted.read_bytes() == source_bytes


def test_v21_inherits_local_no_go_idempotently_and_only_gpt_can_dispatch(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    receipt, _ = load_operational_failure_receipt(
        repo_root=ROOT,
        contract=contract,
    )
    ledger = PilotRunLedger(
        tmp_path / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())

    local = apply_inherited_capability_no_go(
        contract=contract,
        run_ledger=ledger,
        receipt=receipt,
    )
    apply_inherited_capability_no_go(
        contract=contract,
        run_ledger=ledger,
        receipt=receipt,
    )
    assert_amended_capability_dispatch_scope(
        contract=contract,
        run_ledger=ledger,
    )

    snapshot = ledger.snapshot()
    assert ledger.status(local.run_id) == "capability-no-go"
    assert snapshot["runs"][local.run_id]["failure"][
        "provider_calls_current_attempt"
    ] == 0
    gpt = contract.expand(
        stage="capability-gate",
        model=GPT52_PROFILE_ID,
    )[0]
    assert ledger.status(gpt.run_id) == "scheduled"


def test_v21_fully_rehashed_forged_parent_debit_is_rejected_on_reload(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    correct = parent_budget_debit_for_contract(contract)
    assert correct is not None
    path = tmp_path / "budget_ledger.json"
    PilotBudgetLedger(
        path,
        contract_hash=contract.canonical_hash,
        caps=_budget_caps(contract),
        tamper_evident=True,
        parent_debit=correct,
    )
    forged = ParentBudgetDebit(
        parent_contract_sha256=correct.parent_contract_sha256,
        parent_run_ledger_sha256=correct.parent_run_ledger_sha256,
        parent_budget_ledger_sha256="e" * 64,
        stage_bucket=correct.stage_bucket,
        cost_usd=correct.cost_usd,
        hosted_completions=correct.hosted_completions,
        storage_bytes=correct.storage_bytes,
    )
    assert forged.record_sha256 != correct.record_sha256

    with pytest.raises(PilotBudgetError, match="differs from frozen import"):
        PilotBudgetLedger(
            path,
            contract_hash=contract.canonical_hash,
            caps=_budget_caps(contract),
            tamper_evident=True,
            parent_debit=forged,
        )


def test_nonamended_v2_has_no_parent_debit() -> None:
    contract = load_pilot_contract(BASE_CONTRACT_PATH)
    assert parent_budget_debit_for_contract(contract) is None


def test_v21_capability_stage_binds_persisted_amendment_receipt(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    _, persisted = persist_operational_failure_receipt(
        repo_root=ROOT,
        raw_root=tmp_path,
        contract=contract,
    )
    assert persisted in _v2_stage_control_paths(
        contract,
        "capability-gate",
        raw_root=tmp_path,
    )


def test_v21_bad_hosted_credential_stops_before_catalog_and_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = GitProvenance(
        git_tag="pilot-v2.1-science",
        head_commit="1" * 40,
        tag_commit="1" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
        release_attestation={},
    )
    provider_attempts: list[str] = []

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.verify_paid_provenance",
        lambda *_args, **_kwargs: paid,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._persist_release_attestation",
        lambda *_args, **_kwargs: tmp_path / "release_attestation.json",
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._assert_prerequisites",
        lambda *_args, **_kwargs: {},
    )

    def reject_credential(profile, **_kwargs):
        provider_attempts.append(profile.profile_id)
        raise ValueError("credential contains forbidden whitespace")

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._provider_for_profile",
        reject_credential,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.validate_live_provider_catalog",
        lambda *_args, **_kwargs: pytest.fail(
            "live catalog must not run after local credential rejection"
        ),
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._propagate_capability_no_go",
        lambda *_args, **_kwargs: None,
    )

    def write_receipt(_contract, _stage_id, *, raw_root, **_kwargs):
        path = Path(raw_root) / "capability-gate" / "stage_receipt.json"
        _write_json(path, {"execution_progression_go": False})
        return path

    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._write_stage_receipt",
        write_receipt,
    )

    _execute_stage_locked(
        contract_path=CONTRACT_PATH,
        stage_id="capability-gate",
        resume=False,
        raw_root=tmp_path,
        repo_root=ROOT,
    )

    assert provider_attempts == [GPT52_PROFILE_ID]
    budget = json.loads(
        (tmp_path / "budget_ledger.json").read_text(encoding="utf-8")
    )
    assert budget["runs"] == {}
    assert budget["parent_debit"]["record_sha256"] == (
        PARENT_DEBIT_RECORD_SHA256
    )
