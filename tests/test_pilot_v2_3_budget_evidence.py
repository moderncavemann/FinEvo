from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from verified_memory.pilot_budget import PilotBudgetLedger
from verified_memory.pilot_contract import (
    PilotContract,
    canonical_sha256,
    load_pilot_contract,
)
from verified_memory.pilot_evaluation_amendment import (
    parent_budget_debit_for_evaluator_amendment,
)
from verified_memory.pilot_evidence import (
    _validate_v2_budget_hash_chain,
    _validated_release_controls,
)
from verified_memory.pilot_orchestrator import (
    GitProvenance,
    _budget_caps,
    _parent_budget_debit,
    execute_stage,
)
from verified_memory.pilot_preflight_amendment import (
    V23_PARENT_DEBIT_RECORD_SHA256,
    PilotPreflightAmendmentError,
    parent_budget_debit_for_preflight_amendment,
)
from verified_memory.scientific_release_attestation import (
    SCIENTIFIC_RELEASE_ATTESTATION_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
V23_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_3_overlay.yaml"
V23_TAG = "pilot-v2.3-science"
TEST_COMMIT = "1" * 40


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _draft_provenance_binding(contract: PilotContract) -> dict[str, Any]:
    return {
        "git_tag": V23_TAG,
        "resolved_git_commit": TEST_COMMIT,
        "commit_resolution": contract.implementation["commit_resolution"],
        "p0_base_commit": contract.implementation["p0_base_commit"],
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
    }


def _paid(contract: PilotContract) -> GitProvenance:
    attestation = {
        "schema_version": SCIENTIFIC_RELEASE_ATTESTATION_SCHEMA_VERSION,
        "status": "pass",
        "head_commit": TEST_COMMIT,
    }
    return GitProvenance(
        git_tag=V23_TAG,
        head_commit=TEST_COMMIT,
        tag_commit=TEST_COMMIT,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding=_draft_provenance_binding(contract),
        release_attestation={
            **attestation,
            "attestation_sha256": canonical_sha256(attestation),
        },
    )


def test_v23_parent_debit_prioritizes_latest_cumulative_parent_and_hash() -> None:
    contract = load_pilot_contract(V23_CONTRACT_PATH)

    preflight_debit = parent_budget_debit_for_preflight_amendment(contract)
    evaluator_debit = parent_budget_debit_for_evaluator_amendment(contract)
    selected = _parent_budget_debit(contract)

    assert preflight_debit is not None
    assert evaluator_debit is not None
    assert selected.to_dict() == preflight_debit.to_dict()
    assert selected.record_sha256 == V23_PARENT_DEBIT_RECORD_SHA256
    assert selected.parent_contract_sha256 == (
        "72f9a4f7b687e6711d54d1bed45350963b2039e79565fffb39b03b8c6c66b493"
    )
    assert selected.parent_run_ledger_sha256 == (
        "19c89a56e6b2317bf97eccd631a472fd2772fd37deede0117d2723b827ed9d42"
    )
    assert selected.parent_budget_ledger_sha256 == (
        "021d451e5b06a893d466848fd6313555dc24c13269376de06e398ff47b3bd998"
    )
    assert selected.cost_usd == pytest.approx(1.53775475)
    assert selected.hosted_completions == 60
    assert selected.storage_bytes == 751_437

    # V2.3 must import the final V2.2 cumulative ledger, not silently fall
    # back to the earlier evaluator-amendment snapshot.
    assert evaluator_debit.record_sha256 != selected.record_sha256
    assert evaluator_debit.storage_bytes == 715_860


def test_v23_parent_debit_rejects_rehashed_cumulative_total_drift() -> None:
    contract = load_pilot_contract(V23_CONTRACT_PATH)
    tampered = contract.to_dict()
    tampered["preflight_bootstrap_amendment"]["budget_carry_forward"][
        "storage_bytes"
    ] += 1

    with pytest.raises(
        PilotPreflightAmendmentError,
        match="frozen cumulative debit",
    ):
        parent_budget_debit_for_preflight_amendment(tampered)


def test_v23_imported_capabilities_are_zero_call_budget_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(V23_CONTRACT_PATH)
    paid = _paid(contract)
    raw_root = tmp_path / "raw"
    dispatch_attempts: list[str] = []

    def validate_draft_provenance(
        observed: PilotContract,
        resolved_commit: str,
        git_tag: str,
    ) -> dict[str, Any]:
        assert observed.contract_id == contract.contract_id
        assert observed.canonical_hash == contract.canonical_hash
        assert resolved_commit == TEST_COMMIT
        assert git_tag == V23_TAG
        return _draft_provenance_binding(observed)

    def paid_provenance(
        observed: PilotContract,
        *,
        repo_root: Path,
        scientific_launch_input_path: Path,
    ) -> GitProvenance:
        assert observed.canonical_hash == contract.canonical_hash
        assert Path(repo_root).resolve() == ROOT
        assert scientific_launch_input_path == (
            raw_root.resolve() / "scientific_launch_input.json"
        )
        return paid

    def forbid_dispatch(*args: Any, **kwargs: Any) -> None:
        dispatch_attempts.append("dispatch")
        pytest.fail("V2.3 capability import attempted a provider operation")

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
        forbid_dispatch,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator.validate_live_provider_catalog",
        forbid_dispatch,
    )
    monkeypatch.setattr(
        "verified_memory.pilot_orchestrator._execute_capability_preflight",
        forbid_dispatch,
    )

    receipt = execute_stage(
        contract_path=V23_CONTRACT_PATH,
        stage_id="capability-gate",
        resume=False,
        raw_root=raw_root,
        repo_root=ROOT,
    )

    assert dispatch_attempts == []
    assert receipt["status"] == "complete"
    assert receipt["go"] is True
    assert receipt["artifacts"]["provider_calls_current_attempt"] == 0
    assert receipt["artifacts"]["imported_model_count"] == 2
    assert set(receipt["artifacts"]["bootstrap_projections"]) == {
        "gpt52_main",
        "llama33_local_controlled",
    }

    persisted = _json(raw_root / "budget_ledger.json")
    assert persisted["parent_debit"]["record_sha256"] == (
        V23_PARENT_DEBIT_RECORD_SHA256
    )
    assert [row["event_type"] for row in persisted["events"]] == [
        "genesis",
        "parent_debit_imported",
        "run_reserved",
        "run_finalized",
        "run_reserved",
        "run_finalized",
    ]

    actual_storage = 0
    capability_specs = contract.expand(stage="capability-gate")
    assert len(capability_specs) == 2
    for spec in capability_specs:
        row = persisted["runs"][spec.run_id]
        assert row["status"] == "complete"
        assert row["failure"] is None
        assert row["reservation"]["cost_usd"] == 0.0
        assert row["reservation"]["completions"] == 0
        assert row["reservation"]["storage_bytes"] == 5_000_000
        assert row["reservation"]["basis"] == {
            "method": "validated-capability-import-zero-provider-call",
            "hosted_completion_cap_counted": True,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        assert row["actual"]["cost_usd"] == 0.0
        assert row["actual"]["completions"] == 0
        assert 0 < row["actual"]["storage_bytes"] <= 5_000_000
        actual_storage += row["actual"]["storage_bytes"]

        run_dir = raw_root / spec.stage_id / "runs" / spec.run_id
        capability = _json(run_dir / "capability.json")
        gate = _json(run_dir / "gate_receipt.json")
        assert capability["provider_calls_current_attempt"] == 0
        assert gate["provider_calls_current_attempt"] == 0
        assert (run_dir / "bootstrap_projection_p95.json").is_file()

    # Reload through the budget API to verify cumulative semantics: the
    # imported V2.2 spend is counted once, while the V2.3 zero-call rows add
    # only their actual artifact bytes.
    snapshot = PilotBudgetLedger(
        raw_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=_budget_caps(contract),
        tamper_evident=True,
        parent_debit=_parent_budget_debit(contract),
    ).snapshot()
    for totals_key in ("committed", "committed_plus_reserved"):
        totals = snapshot[totals_key]
        assert totals["cost_usd"] == pytest.approx(1.53775475)
        assert totals["completions"] == 60
        assert totals["storage_bytes"] == 751_437 + actual_storage
        assert totals["stage_cost_usd"]["capability"] == pytest.approx(
            1.53775475
        )

    # Exercise the evidence publisher's independent budget admission path,
    # including the V2.3 cumulative parent debit.  The parent is counted once
    # even though the imported capability rows are present in the child
    # ledger, and the sealed event chain is required for admission.
    controls = _validated_release_controls(
        contract,
        raw_root=raw_root,
        rows=[],
        common_commit=None,
    )
    evidence_budget = controls["budget_ledger"]
    assert evidence_budget["pass"] is True
    assert evidence_budget["checks"] == {
        "schema_and_contract": True,
        "self_hash_and_event_chain": True,
        "parent_debit_exact": True,
        "exact_frozen_caps": True,
        "valid_finalized_dispatch_units": True,
        "all_artifact_backed_dispatches_accounted": True,
        "actual_totals_within_caps": True,
    }
    assert evidence_budget["actual_totals"] == {
        "cost_usd": pytest.approx(1.53775475),
        "completions": 60,
        "storage_bytes": 751_437 + actual_storage,
    }
    assert evidence_budget["actual_stage_cost_usd"]["capability"] == (
        pytest.approx(1.53775475)
    )
    assert _validate_v2_budget_hash_chain(contract, persisted) is True

    # A stale event mutation must fail both the direct hash-chain verifier and
    # the evidence controls, even if all visible cumulative totals are left
    # untouched.
    chain_tampered = json.loads(json.dumps(persisted))
    chain_tampered["events"][-1]["payload"]["status"] = "failed"
    (raw_root / "budget_ledger.json").write_text(
        json.dumps(chain_tampered, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert _validate_v2_budget_hash_chain(contract, chain_tampered) is False
    chain_controls = _validated_release_controls(
        contract,
        raw_root=raw_root,
        rows=[],
        common_commit=None,
    )["budget_ledger"]
    assert chain_controls["pass"] is False
    assert chain_controls["checks"]["self_hash_and_event_chain"] is False
    assert chain_controls["checks"]["parent_debit_exact"] is True

    # Rehashing only the top-level ledger cannot launder a changed cumulative
    # parent debit: the frozen record, genesis binding, and adjacent import
    # event remain independent checks in pilot_evidence.
    parent_tampered = json.loads(json.dumps(persisted))
    parent_tampered["parent_debit"]["storage_bytes"] += 1
    unsigned_parent_tampered = dict(parent_tampered)
    unsigned_parent_tampered.pop("ledger_sha256")
    parent_tampered["ledger_sha256"] = canonical_sha256(
        unsigned_parent_tampered
    )
    (raw_root / "budget_ledger.json").write_text(
        json.dumps(parent_tampered, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert _validate_v2_budget_hash_chain(contract, parent_tampered) is False
    parent_controls = _validated_release_controls(
        contract,
        raw_root=raw_root,
        rows=[],
        common_commit=None,
    )["budget_ledger"]
    assert parent_controls["pass"] is False
    assert parent_controls["checks"]["self_hash_and_event_chain"] is False
    assert parent_controls["checks"]["parent_debit_exact"] is False
    assert parent_controls["actual_totals"] == {
        "cost_usd": pytest.approx(1.53775475),
        "completions": 60,
        "storage_bytes": 751_437 + actual_storage,
    }

    assert not (raw_root / "capability-gate" / "provider_catalog").exists()
    assert list(raw_root.rglob("*provider_call_journal*.json")) == []
