from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_2,
    canonical_sha256,
    load_pilot_contract,
)
from verified_memory.pilot_evaluation_amendment import (
    CAPABILITY_IMPORT_SCHEMA_VERSION,
    EVALUATOR_CORRECTION_RECEIPT_RELATIVE_PATH,
    EXPECTED_EVALUATOR_AMENDMENT_SHA256,
    EXPECTED_EVALUATOR_RECEIPT_CONTENT_SHA256,
    EXPECTED_EVALUATOR_RECEIPT_FILE_SHA256,
    EXPECTED_MODEL_IDS,
    EXPECTED_PROPOSAL_TASK_IDS,
    EXPECTED_TASK_IDS,
    PilotEvaluationAmendmentError,
    V22_PARENT_DEBIT_RECORD_SHA256,
    build_capability_import,
    evaluator_amendment_control_path,
    load_evaluator_amendment_receipt,
    model_import_records,
    parent_budget_debit_for_evaluator_amendment,
    persist_evaluator_correction_receipt,
    validate_capability_import,
    validate_evaluator_amendment_mapping,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_2_overlay.yaml"
V21_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_1.yaml"
RECEIPT_PATH = ROOT / EVALUATOR_CORRECTION_RECEIPT_RELATIVE_PATH


@pytest.fixture
def contract():
    return load_pilot_contract(CONTRACT_PATH)


@pytest.fixture
def receipt(contract):
    value, path = load_evaluator_amendment_receipt(
        repo_root=ROOT,
        contract=contract,
    )
    assert path == RECEIPT_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        EXPECTED_EVALUATOR_RECEIPT_FILE_SHA256
    )
    return value


def _receipt_document() -> dict[str, Any]:
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def _write_receipt_repo(tmp_path: Path, value: dict[str, Any]) -> Path:
    path = tmp_path / EVALUATOR_CORRECTION_RECEIPT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _reseal_receipt(value: dict[str, Any]) -> None:
    unsigned = deepcopy(value)
    unsigned.pop("integrity")
    value["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(unsigned),
    }


def _reseal_import(value: dict[str, Any]) -> None:
    unsigned = deepcopy(value)
    unsigned.pop("integrity")
    value["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(unsigned),
    }


def _capability_spec(contract, model_id: str):
    specs = contract.expand(stage="capability-gate", model=model_id)
    assert len(specs) == 1
    return specs[0]


def test_tracked_receipt_loads_exact_bytes_and_hashes(contract) -> None:
    receipt, path = load_evaluator_amendment_receipt(
        repo_root=ROOT,
        contract=contract,
    )

    assert path == RECEIPT_PATH
    assert receipt["status"] == "frozen"
    unsigned = deepcopy(receipt)
    integrity = unsigned.pop("integrity")
    assert integrity == {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": EXPECTED_EVALUATOR_RECEIPT_CONTENT_SHA256,
    }
    assert canonical_sha256(unsigned) == (
        EXPECTED_EVALUATOR_RECEIPT_CONTENT_SHA256
    )
    assert canonical_sha256(receipt["evaluator_amendment"]) == (
        EXPECTED_EVALUATOR_AMENDMENT_SHA256
    )
    assert receipt["evaluator_amendment"] == contract.to_dict()[
        "evaluator_amendment"
    ]
    validate_evaluator_amendment_mapping(receipt["evaluator_amendment"])


def test_tracked_receipt_rejects_contract_mismatch() -> None:
    v21 = load_pilot_contract(V21_CONTRACT_PATH)

    with pytest.raises(
        PilotEvaluationAmendmentError,
        match="differs from the loaded contract",
    ):
        load_evaluator_amendment_receipt(repo_root=ROOT, contract=v21)


def test_tracked_receipt_rejects_rehashed_contract_amendment_drift(
    contract,
) -> None:
    contract_mapping = contract.to_dict()
    contract_mapping["evaluator_amendment"]["source_attempts"][0][
        "capability_sha256"
    ] = "f" * 64

    with pytest.raises(
        PilotEvaluationAmendmentError,
        match="differs from the loaded contract",
    ):
        load_evaluator_amendment_receipt(
            repo_root=ROOT,
            contract=contract_mapping,
        )


def test_tracked_receipt_rejects_symlink_source(
    tmp_path: Path,
    contract,
) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(RECEIPT_PATH.read_bytes())
    target = tmp_path / EVALUATOR_CORRECTION_RECEIPT_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.symlink_to(source)

    with pytest.raises(
        PilotEvaluationAmendmentError,
        match="cannot safely read|must not be a symlink",
    ):
        load_evaluator_amendment_receipt(
            repo_root=tmp_path,
            contract=contract,
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["model_audits"][0]["proposal_rows"][0].__setitem__(
            "candidate_status",
            "rejected",
        ),
        lambda value: value["model_audits"][1]["proposal_rows"].pop(),
        lambda value: value["usage_projection_rows"][0]["rows"][0][
            "usage"
        ].__setitem__("total_tokens", 0),
        lambda value: value["usage_projection_rows"][1]["rows"][0].__setitem__(
            "response_model",
            "wrong-model",
        ),
        lambda value: value["audit"].__setitem__("provider_calls", 1),
        lambda value: value["limitations"].clear(),
    ],
)
def test_rehashed_receipt_tamper_is_rejected(
    contract,
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    value = _receipt_document()
    mutator(value)
    _reseal_receipt(value)

    with pytest.raises(
        PilotEvaluationAmendmentError,
        match="receipt hash mismatch",
    ):
        model_import_records(contract, value)


def test_model_import_records_bind_both_complete_denominators(
    contract,
    receipt,
) -> None:
    records = model_import_records(contract, receipt)

    assert tuple(row["model_id"] for row in records) == EXPECTED_MODEL_IDS
    by_model = {row["model_id"]: row for row in records}
    assert set(by_model) == set(EXPECTED_MODEL_IDS)
    for model_id, row in by_model.items():
        assert row["provider_calls_current_attempt"] == 0
        assert row["pass"] is True
        assert row["capability_assessment"] == {
            "status": "pass",
            "pass": True,
        }
        assert row["checks"] == {
            "utility-ranking": True,
            "rule-application": True,
            "rule-proposal": True,
        }
        assert tuple(
            proposal["task_id"] for proposal in row["proposal_rows"]
        ) == EXPECTED_PROPOSAL_TASK_IDS
        assert all(
            proposal["semantic_candidate_accepted"] is True
            and proposal["candidate_status"] == "provisional"
            and len(proposal["supporting_episode_ids"]) == 3
            for proposal in row["proposal_rows"]
        )
        assert tuple(
            usage["task_id"] for usage in row["usage_projection_rows"]
        ) == EXPECTED_TASK_IDS
        assert row["corrected_scores"]["rule-proposal"] == {
            "correct": 6,
            "denominator": 6,
            "required": 5,
        }
        assert row["old_diagnostic"]["scores"]["rule-proposal"]["correct"] == 0
        assert row["old_diagnostic"][
            "semantic_match_disposition"
        ] == "diagnostic-only"
        assert len(row["taskset_sha256"]) == 64
        assert set(row["source_hashes"]) == {
            "contract_sha256",
            "capability_file_sha256",
            "gate_receipt_file_sha256",
            "summary_file_sha256",
            "summary_content_sha256",
        }
        assert model_id in {"gpt52_main", "llama33_local_controlled"}


@pytest.mark.parametrize("model_id", EXPECTED_MODEL_IDS)
def test_build_and_validate_capability_import(
    contract,
    receipt,
    model_id: str,
) -> None:
    spec = _capability_spec(contract, model_id)
    imported = build_capability_import(contract, spec, receipt)

    assert imported["schema_version"] == CAPABILITY_IMPORT_SCHEMA_VERSION
    assert imported["contract_id"] == PILOT_CONTRACT_ID_V2_2
    assert imported["contract_sha256"] == contract.canonical_hash
    assert imported["target_run_id"] == spec.run_id
    assert imported["target_execution_mode"] == "capability_probe"
    assert imported["model_id"] == model_id
    assert len(imported["proposal_rows"]) == 6
    assert len(imported["usage_projection_rows"]) == 30
    assert imported["category_totals"]["rule-proposal"][
        "registered_correct"
    ] == 6
    assert imported["checks"] == {
        "utility-ranking": True,
        "rule-application": True,
        "rule-proposal": True,
    }
    assert imported["interface_gate"] == {"pass": True, "failure_count": 0}
    assert imported["capability_assessment"] == {
        "status": "pass",
        "pass": True,
    }
    assert imported["pass"] is True
    assert imported["provider_calls_current_attempt"] == 0
    assert imported["scientific_evidence"] is False
    unsigned = deepcopy(imported)
    integrity = unsigned.pop("integrity")
    assert integrity["content_sha256"] == canonical_sha256(unsigned)
    validate_capability_import(imported, contract, spec, receipt)


def test_build_import_rejects_unregistered_model(contract, receipt) -> None:
    spec = SimpleNamespace(
        model_id="not-registered",
        execution_mode="capability_probe",
        run_id="not-registered-run",
    )

    with pytest.raises(
        PilotEvaluationAmendmentError,
        match="no corrected capability import",
    ):
        build_capability_import(contract, spec, receipt)


def test_build_import_rejects_non_capability_execution_mode(
    contract,
    receipt,
) -> None:
    spec = SimpleNamespace(
        model_id="gpt52_main",
        execution_mode="scientific_simulation",
        run_id="wrong-mode-run",
    )

    with pytest.raises(
        PilotEvaluationAmendmentError,
        match="requires a capability/preflight spec",
    ):
        build_capability_import(contract, spec, receipt)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.__setitem__(
            "provider_calls_current_attempt",
            1,
        ),
        lambda value: value["source_hashes"].__setitem__(
            "capability_file_sha256",
            "f" * 64,
        ),
        lambda value: value["proposal_rows"][0].__setitem__(
            "candidate_status",
            "rejected",
        ),
        lambda value: value["usage_projection_rows"][0]["usage"].__setitem__(
            "total_tokens",
            0,
        ),
        lambda value: value["category_totals"]["rule-proposal"].__setitem__(
            "registered_correct",
            5,
        ),
        lambda value: value["old_diagnostic"].__setitem__(
            "semantic_match_disposition",
            "gate",
        ),
    ],
)
def test_rehashed_capability_import_tamper_is_rejected(
    contract,
    receipt,
    mutator: Callable[[dict[str, Any]], None],
) -> None:
    spec = _capability_spec(contract, "gpt52_main")
    imported = build_capability_import(contract, spec, receipt)
    mutator(imported)
    _reseal_import(imported)

    with pytest.raises(
        PilotEvaluationAmendmentError,
        match="differs from the frozen evaluator receipt",
    ):
        validate_capability_import(imported, contract, spec, receipt)


def test_stale_capability_import_hash_is_rejected(contract, receipt) -> None:
    spec = _capability_spec(contract, "gpt52_main")
    imported = build_capability_import(contract, spec, receipt)
    imported["provider_calls_current_attempt"] = 1

    with pytest.raises(
        PilotEvaluationAmendmentError,
        match="content hash mismatch",
    ):
        validate_capability_import(imported, contract, spec, receipt)


def test_parent_budget_debit_is_exact_and_v21_has_no_new_import(
    contract,
) -> None:
    debit = parent_budget_debit_for_evaluator_amendment(contract)
    assert debit is not None
    assert debit.parent_contract_sha256 == (
        contract.evaluator_amendment["parent"]["contract_sha256"]
    )
    assert debit.parent_run_ledger_sha256 == (
        contract.evaluator_amendment["parent"]["run_ledger_internal_sha256"]
    )
    assert debit.parent_budget_ledger_sha256 == (
        contract.evaluator_amendment["parent"]["budget_ledger_internal_sha256"]
    )
    assert debit.stage_bucket == "capability"
    assert debit.cost_usd == pytest.approx(1.53775475)
    assert debit.hosted_completions == 60
    assert debit.storage_bytes == 715860
    assert debit.record_sha256 == V22_PARENT_DEBIT_RECORD_SHA256

    v21 = load_pilot_contract(V21_CONTRACT_PATH)
    assert parent_budget_debit_for_evaluator_amendment(v21) is None


def test_persist_receipt_is_exact_and_idempotent(
    tmp_path: Path,
    contract,
) -> None:
    raw_root = tmp_path / "raw"
    first, first_path = persist_evaluator_correction_receipt(
        repo_root=ROOT,
        raw_root=raw_root,
        contract=contract,
    )
    before = first_path.stat()
    second, second_path = persist_evaluator_correction_receipt(
        repo_root=ROOT,
        raw_root=raw_root,
        contract=contract,
    )
    after = second_path.stat()

    assert first == second
    assert first_path == second_path == evaluator_amendment_control_path(
        raw_root=raw_root
    )
    assert first_path.read_bytes() == RECEIPT_PATH.read_bytes()
    assert before.st_ino == after.st_ino
    assert before.st_size == after.st_size
    assert before.st_mtime_ns == after.st_mtime_ns
    assert os.stat(first_path).st_mode & 0o777 == 0o600


def test_persist_receipt_rejects_existing_different_bytes(
    tmp_path: Path,
    contract,
) -> None:
    raw_root = tmp_path / "raw"
    _, path = persist_evaluator_correction_receipt(
        repo_root=ROOT,
        raw_root=raw_root,
        contract=contract,
    )
    path.write_bytes(b"{}\n")

    with pytest.raises(
        PilotEvaluationAmendmentError,
        match="differs from tracked source",
    ):
        persist_evaluator_correction_receipt(
            repo_root=ROOT,
            raw_root=raw_root,
            contract=contract,
        )
