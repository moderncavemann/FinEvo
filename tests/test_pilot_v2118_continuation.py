from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verified_memory import pilot_v2118_continuation as continuation
from verified_memory.pilot_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]


def _draft_boundary() -> dict[str, Any]:
    document = json.loads(
        (ROOT / "experiments/pilot_v2_11_8.yaml").read_text(encoding="utf-8")
    )
    return document["v2118_recovery_boundary"]


def _contract() -> SimpleNamespace:
    return SimpleNamespace(
        contract_id=continuation.V2118_CONTRACT_ID,
        canonical_hash="9" * 64,
        implementation={"required_git_tag": continuation.V2118_SCIENCE_TAG},
        v2118_recovery_boundary=_draft_boundary(),
    )


def test_frozen_v2117_no_go_boundary_and_six_file_inventory_are_exact() -> None:
    boundary = _draft_boundary()

    assert (
        continuation._expected_v2117_failed_release_no_go()
        == boundary["failed_release_no_go"]
    )
    assert continuation._expected_v2115_parent_release() == boundary["parent_release"]
    assert continuation.V2117_COMPLETE_RAW_FILE_COUNT == 6
    assert continuation.V2117_COMPLETE_RAW_STORAGE_BYTES == 224_211
    assert continuation.V2117_COMPLETE_RAW_INVENTORY_SHA256 == (
        "13d7cc64beebafaf82aed90ebe4fd1abd1c00c300352a0bdba14fff492b1c7cf"
    )
    assert continuation.V2117_EVIDENCE_RAW_FILE_COUNT == 5
    assert continuation.V2117_EVIDENCE_RAW_STORAGE_BYTES == 224_071
    assert continuation.V2117_EVIDENCE_RAW_INVENTORY_SHA256 == (
        "af4053b3e7fc2b706707f47d552d56ac25dfff4fbf5df5d58a6739e375f160ec"
    )
    assert set(continuation.V2117_RAW_FILE_BINDINGS) == {
        ".real-stage-execution.lock",
        "budget_ledger.json",
        "parent-import/stage_receipt.json",
        "release_attestation.json",
        "run_ledger.json",
        "scientific_launch_input.json",
    }
    assert boundary["failed_release_no_go"]["run_ledger"]["status_counts"] == {
        "integrity-stopped": 87
    }
    assert boundary["failed_release_no_go"]["provider_construction"] is False
    assert boundary["failed_release_no_go"]["provider_calls"] == 0
    assert boundary["failed_release_no_go"]["science_reservations"] == 0


def test_parent_debit_includes_v2117_actual_storage() -> None:
    debit = continuation.parent_budget_debit_for_v2118(_contract())

    assert debit.to_dict() == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": continuation.V2117_CONTRACT_SHA256,
        "parent_run_ledger_sha256": continuation.V2117_RUN_LEDGER_SHA256,
        "parent_budget_ledger_sha256": continuation.V2117_BUDGET_LEDGER_SHA256,
        "stage_bucket": "parent_v2117",
        "cost_usd": 63.1196450625,
        "hosted_completions": 3_440,
        "storage_bytes": 270_191_728,
        "record_sha256": continuation.V2118_PARENT_DEBIT_RECORD_SHA256,
    }
    assert debit.storage_bytes == 270_189_931 + 1_797

    tampered = _contract()
    tampered.v2118_recovery_boundary = deepcopy(tampered.v2118_recovery_boundary)
    tampered.v2118_recovery_boundary["parent_budget_debit"]["storage_bytes"] -= 1
    with pytest.raises(
        continuation.PilotV2118ContinuationError,
        match="parent budget debit drifted",
    ):
        continuation.parent_budget_debit_for_v2118(tampered)


def test_six_file_inventory_rejects_lock_or_json_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path.joinpath(*continuation.V2117_FAILED_RAW_ROOT.parts)
    raw.mkdir(parents=True)
    payloads = {
        ".real-stage-execution.lock": b"lock\n",
        "budget_ledger.json": b"budget\n",
        "parent-import/stage_receipt.json": b"stage\n",
        "release_attestation.json": b"attestation\n",
        "run_ledger.json": b"runs\n",
        "scientific_launch_input.json": b"launch\n",
    }
    for relative, payload in payloads.items():
        path = raw / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    rows = [
        {
            "path": relative,
            "byte_size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for relative, payload in sorted(payloads.items())
    ]
    evidence = [row for row in rows if row["path"] != ".real-stage-execution.lock"]
    monkeypatch.setattr(
        continuation,
        "V2117_RAW_FILE_BINDINGS",
        {
            row["path"]: (row["byte_size"], row["sha256"])
            for row in rows
        },
    )
    monkeypatch.setattr(continuation, "V2117_COMPLETE_RAW_FILE_COUNT", len(rows))
    monkeypatch.setattr(
        continuation,
        "V2117_COMPLETE_RAW_STORAGE_BYTES",
        sum(row["byte_size"] for row in rows),
    )
    monkeypatch.setattr(
        continuation, "V2117_COMPLETE_RAW_INVENTORY_SHA256", canonical_sha256(rows)
    )
    monkeypatch.setattr(continuation, "V2117_EVIDENCE_RAW_FILE_COUNT", len(evidence))
    monkeypatch.setattr(
        continuation,
        "V2117_EVIDENCE_RAW_STORAGE_BYTES",
        sum(row["byte_size"] for row in evidence),
    )
    monkeypatch.setattr(
        continuation,
        "V2117_EVIDENCE_RAW_INVENTORY_SHA256",
        canonical_sha256(evidence),
    )

    inventory = continuation._v2117_raw_inventory(tmp_path)
    assert inventory["complete"]["file_count"] == 6
    assert inventory["evidence"]["file_count"] == 5

    (raw / ".real-stage-execution.lock").write_bytes(b"changed\n")
    with pytest.raises(
        continuation.PilotV2118ContinuationError,
        match="six-file raw binding drifted",
    ):
        continuation._v2117_raw_inventory(tmp_path)


def test_dual_root_roles_cannot_alias(tmp_path: Path) -> None:
    with pytest.raises(
        continuation.PilotV2118ContinuationError,
        match="must be distinct",
    ):
        continuation.verify_v2117_terminal_no_go(
            failed_repo_root=tmp_path,
            authority_repo_root=tmp_path,
        )


def test_parent_import_receipt_is_zero_provider_and_tamper_evident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in continuation._PROVIDER_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    contract = _contract()
    state = {
        "run_snapshot": {"ledger_sha256": continuation.V2117_RUN_LEDGER_SHA256},
        "budget_snapshot": {
            "ledger_sha256": continuation.V2117_BUDGET_LEDGER_SHA256
        },
        "stage_receipt": {
            "integrity": {
                "content_sha256": (
                    continuation.V2117_PARENT_IMPORT_RECEIPT_CONTENT_SHA256
                )
            }
        },
    }
    monkeypatch.setattr(
        continuation,
        "verify_v2117_terminal_no_go",
        lambda **_kwargs: state,
    )
    monkeypatch.setattr(
        continuation,
        "_verify_v2115_acceptance_with_authority_context",
        lambda _root: {"status": "pass"},
    )
    paid = SimpleNamespace(
        git_tag=continuation.V2118_SCIENCE_TAG,
        head_commit="8" * 40,
        tag_commit="8" * 40,
        tag_object_type="tag",
        worktree_clean=True,
    )

    receipt = continuation.build_v2118_parent_import_receipt(
        contract=contract,
        failed_repo_root="failed-v2117",
        authority_repo_root="authority-v2115",
        paid=paid,
    )
    verified = continuation.verify_v2118_parent_import_receipt(
        receipt, contract=contract
    )

    assert verified["status"] == "complete"
    assert verified["go"] is True
    assert verified["denominator_continuation"] == {
        "failed_registered_rows": 87,
        "failed_integrity_stopped_rows": 87,
        "failed_rows_reclassified_or_redispatched": 0,
        "child_operational_rows": 1,
        "child_scientific_rows": 86,
    }
    assert verified["import_policy"]["provider_construction"] is False
    assert verified["import_policy"]["provider_calls"] == 0
    assert verified["import_policy"]["imported_effect_cells"] == 0
    assert verified["scientific_evidence"] is False

    tampered = deepcopy(receipt)
    tampered["failed_release_no_go"]["run_ledger"]["status_counts"] = {
        "complete": 87
    }
    with pytest.raises(
        continuation.PilotV2118ContinuationError,
        match="self-hash mismatch",
    ):
        continuation.verify_v2118_parent_import_receipt(tampered, contract=contract)


def test_provider_keys_must_be_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in continuation._PROVIDER_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    continuation.require_v2118_provider_keys_absent()

    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret-never-read")
    with pytest.raises(
        continuation.PilotV2118ContinuationError,
        match="before provider credentials are loaded",
    ):
        continuation.require_v2118_provider_keys_absent()


def test_unsealed_current_authority_and_acceptance_paths_fail_closed(
    tmp_path: Path,
) -> None:
    raw = tmp_path.joinpath(*continuation.V2118_RAW_ROOT.parts)
    raw.mkdir(parents=True)
    assert continuation.current_authority_path(raw) == (
        raw / "parent-import/current_authority/post_gate_authority.json"
    )
    with pytest.raises(
        continuation.PilotV2118ContinuationError,
        match="must be a regular non-symlink file",
    ):
        continuation.verify_v2118_current_authority(
            contract=_contract(),
            repo_root=tmp_path,
            raw_root=raw,
            paid=SimpleNamespace(),
        )
    with pytest.raises(
        continuation.PilotV2118ContinuationError,
        match="must be a regular non-symlink file",
    ):
        continuation.verify_v2118_scientific_dispatch_acceptance(
            raw / continuation.V2118_ACCEPTANCE_FILENAME,
            contract=_contract(),
            repo_root=tmp_path,
            raw_root=raw,
            paid=SimpleNamespace(),
            run_ledger=None,
            budget_ledger=None,
        )
