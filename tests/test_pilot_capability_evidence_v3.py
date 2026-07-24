from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from test_pilot_capability import (
    CapabilityFixtureProvider,
    EquivalentConditionToleranceProvider,
    GeminiLengthProvider,
    ProviderFailureProvider,
    RecoveryProvider,
    WrongSemanticEvidenceProvider,
    _contracts,
    _run_gate,
)
from verified_memory.pilot_contract import canonical_sha256, load_pilot_contract
from verified_memory.pilot_evidence import (
    PilotEvidenceError,
    _validate_capability_v3,
    _validate_capability_v4,
    _validate_terminal_payload_marker,
)
from verified_memory.pilot_evaluation_amendment import (
    build_capability_import,
    evaluator_amendment_control_path,
    persist_evaluator_correction_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v1.yaml"
V22_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_2_overlay.yaml"


def _v22_capability_import_fixture(
    tmp_path: Path,
    *,
    execution_mode: str,
) -> tuple[object, object, dict]:
    contract = load_pilot_contract(V22_CONTRACT_PATH)
    receipt, _ = persist_evaluator_correction_receipt(
        repo_root=ROOT,
        raw_root=tmp_path,
        contract=contract,
    )
    stage = (
        "capability-gate"
        if execution_mode == "capability_probe"
        else "closed-loop-preflight"
    )
    spec = next(
        item
        for item in contract.expand(stage=stage, model="gpt52_main")
        if item.execution_mode == execution_mode
    )
    return contract, spec, build_capability_import(contract, spec, receipt)


def _rehash_capability_import(value: dict) -> None:
    unsigned = deepcopy(value)
    unsigned.pop("integrity")
    value["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(unsigned),
    }


def _valid_v4() -> dict:
    result = _run_gate(
        CapabilityFixtureProvider(),
        budget_id="capability-v4-evidence",
        contracts=_contracts(),
    )
    _validate_capability_v4(result)
    return result


def _historical_v3_semantic_no_go() -> dict:
    result = _run_gate(
        EquivalentConditionToleranceProvider(),
        budget_id="capability-v3-historical-semantic-no-go",
        contracts=_contracts(),
    )
    result["schema_version"] = "finevo-capability-gate-v3"
    proposal_rows = [
        row for row in result["rows"] if row["task_kind"] == "rule_proposal"
    ]
    for row in result["rows"]:
        row["schema_version"] = "finevo-capability-gate-v3"
    for row in proposal_rows:
        assert row["semantic_candidate_accepted"] is True
        assert row["semantic_match"] is False
        row["legal"] = False
        row["correct"] = False

    totals = result["category_totals"]["rule-proposal"]
    totals.update(
        {
            "correct": 0,
            "registered_correct": 0,
            "conditional_correct": 0,
            "conditional_accuracy": 0.0,
        }
    )
    result["checks"]["rule-proposal"] = False
    result["capability_assessment"] = {
        "status": "fail",
        "pass": False,
        "checks": {
            "utility-ranking": True,
            "rule-application": True,
            "rule-proposal": False,
        },
    }
    result["pass"] = False
    _validate_capability_v3(result)
    return result


def _row(result: dict, task_id: str) -> dict:
    return next(row for row in result["rows"] if row["task_id"] == task_id)


class _PreResponseFailureProvider(ProviderFailureProvider):
    """Model a transport failure before any served-model response exists."""

    def mutate(self, task, completion):
        failed = super().mutate(task, completion)
        if task.task_id != "action-01":
            return failed
        return replace(
            failed,
            response_model=None,
            finish_reason=None,
            native_finish_reason=None,
            response_completed=None,
            output_disposition="unavailable_due_to_provider_error",
        )


class _InvalidFinishProvider(CapabilityFixtureProvider):
    def mutate(self, task, completion):
        if task.task_id != "action-01":
            return completion
        return replace(
            completion,
            finish_reason="content_filter",
            native_finish_reason="content_filter",
            response_completed=False,
            output_disposition="discarded_invalid_finish",
        )


def _pre_response_failure_v4() -> dict:
    return _run_gate(
        _PreResponseFailureProvider(),
        budget_id="capability-v4-pre-response-failure",
        contracts=_contracts(),
    )


def _valid_v2() -> dict:
    taskset = hashlib.sha256(b"read-only-v2-fixture").hexdigest()
    categories = (
        ("utility-ranking", 12, 10),
        ("rule-application", 12, 10),
        ("rule-proposal", 6, 5),
    )
    rows = []
    totals = {}
    checks = {}
    for category, denominator, required in categories:
        for index in range(denominator):
            rows.append(
                {
                    "schema_version": "finevo-capability-gate-v2",
                    "taskset_sha256": taskset,
                    "task_id": f"{category}-{index:02d}",
                    "category": category,
                    "correct": True,
                    "evaluable": True,
                    "interface_status": "pass",
                    "provider_error": None,
                    "provider_error_details": None,
                    "parse_error": None,
                    "parse_error_code": None,
                    "parse_error_offset": None,
                }
            )
        totals[category] = {
            "correct": denominator,
            "denominator": denominator,
            "required": required,
            "registered_correct": denominator,
            "registered_total": denominator,
            "evaluable_count": denominator,
            "conditional_correct": denominator,
            "conditional_accuracy": 1.0,
            "interface_failure_count": 0,
        }
        checks[category] = True
    return {
        "schema_version": "finevo-capability-gate-v2",
        "taskset_sha256": taskset,
        "pass": True,
        "preflight_go": True,
        "checks": checks,
        "category_totals": totals,
        "interface_gate": {"pass": True, "failure_count": 0},
        "capability_assessment": {
            "status": "pass",
            "pass": True,
            "checks": {category: True for category, _, _ in categories},
        },
        "provider_failure_count": 0,
        "parse_failure_count": 0,
        "rows": rows,
    }


def test_historical_v3_semantic_mismatch_remains_a_valid_no_go() -> None:
    result = _historical_v3_semantic_no_go()
    proposals = [
        row for row in result["rows"] if row["task_kind"] == "rule_proposal"
    ]

    assert len(proposals) == 6
    assert all(row["semantic_candidate_accepted"] is True for row in proposals)
    assert all(row["candidate_status"] == "provisional" for row in proposals)
    assert all(row["semantic_match"] is False for row in proposals)
    assert all(row["legal"] is False for row in proposals)
    assert all(row["correct"] is False for row in proposals)
    assert result["category_totals"]["rule-proposal"]["correct"] == 0
    assert result["pass"] is False
    _validate_capability_v3(result)


def test_terminal_capability_entry_routes_v2_v3_and_v4_readers(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="capability-preflight")[0]

    for capability in (_valid_v2(), _historical_v3_semantic_no_go(), _valid_v4()):
        capability["preflight_go"] = True
        _validate_terminal_payload_marker(
            contract,
            spec.to_dict(),
            {
                "metrics": {},
                "gate_evidence": {"go": True},
                "capability": capability,
            },
            raw_root=tmp_path,
        )


def test_terminal_marker_accepts_exact_capability_import(
    tmp_path: Path,
) -> None:
    contract, spec, imported = _v22_capability_import_fixture(
        tmp_path,
        execution_mode="capability_probe",
    )

    _validate_terminal_payload_marker(
        contract,
        spec.to_dict(),
        {
            "metrics": {},
            "gate_evidence": {"go": True},
            "capability": imported,
        },
        raw_root=tmp_path,
    )


def test_closed_loop_import_without_preflight_artifacts_fails_closed(
    tmp_path: Path,
) -> None:
    contract, spec, imported = _v22_capability_import_fixture(
        tmp_path,
        execution_mode="closed_loop_preflight",
    )

    with pytest.raises(
        PilotEvidenceError,
        match="closed-loop preflight lacks its exact checks and artifacts",
    ):
        _validate_terminal_payload_marker(
            contract,
            spec.to_dict(),
            {
                "metrics": {},
                "gate_evidence": {"go": False},
                "capability": imported,
            },
            raw_root=tmp_path,
        )


def test_capability_import_rehashed_tampering_fails_closed(tmp_path: Path) -> None:
    contract, spec, imported = _v22_capability_import_fixture(
        tmp_path,
        execution_mode="capability_probe",
    )
    imported["category_totals"]["rule-proposal"]["correct"] = 5
    _rehash_capability_import(imported)

    with pytest.raises(PilotEvidenceError, match="capability import validation failed"):
        _validate_terminal_payload_marker(
            contract,
            spec.to_dict(),
            {
                "metrics": {},
                "gate_evidence": {"go": True},
                "capability": imported,
            },
            raw_root=tmp_path,
        )


def test_capability_import_requires_persisted_evaluator_receipt(
    tmp_path: Path,
) -> None:
    contract, spec, imported = _v22_capability_import_fixture(
        tmp_path,
        execution_mode="capability_probe",
    )
    evaluator_amendment_control_path(raw_root=tmp_path).unlink()

    with pytest.raises(PilotEvidenceError, match="required evidence file is missing"):
        _validate_terminal_payload_marker(
            contract,
            spec.to_dict(),
            {
                "metrics": {},
                "gate_evidence": {"go": True},
                "capability": imported,
            },
            raw_root=tmp_path,
        )


def test_capability_import_cannot_be_rebound_to_another_run(
    tmp_path: Path,
) -> None:
    contract, _, imported = _v22_capability_import_fixture(
        tmp_path,
        execution_mode="capability_probe",
    )
    closed_loop_spec = contract.expand(
        stage="closed-loop-preflight",
        model="gpt52_main",
    )[0]

    with pytest.raises(PilotEvidenceError, match="capability import validation failed"):
        _validate_terminal_payload_marker(
            contract,
            closed_loop_spec.to_dict(),
            {
                "metrics": {},
                "gate_evidence": {"go": False},
                "capability": imported,
            },
            raw_root=tmp_path,
        )


def test_closed_loop_capability_import_can_go_only_with_all_checks(
    tmp_path: Path,
) -> None:
    contract, spec, imported = _v22_capability_import_fixture(
        tmp_path,
        execution_mode="closed_loop_preflight",
    )
    imported["preflight_go"] = True
    imported["preflight_checks"] = {
        "action_parse_success": True,
        "provider_route_complete": True,
    }
    _rehash_capability_import(imported)

    _validate_terminal_payload_marker(
        contract,
        spec.to_dict(),
        {
            "metrics": {},
            "gate_evidence": {
                "go": True,
                "preflight_checks": imported["preflight_checks"],
                "preflight_manifest": None,
                "preflight_checkpoint": "checkpoint.json",
                "preflight_checkpoint_exactness": (
                    "preflight_checkpoint_exactness.json"
                ),
                "projection": "projection_p95.json",
            },
            "capability": imported,
        },
        raw_root=tmp_path,
    )


@pytest.mark.parametrize(
    "preflight_checks",
    (
        None,
        {},
        {"action_parse_success": True, "provider_route_complete": False},
    ),
)
def test_closed_loop_capability_import_go_fails_without_all_checks(
    tmp_path: Path,
    preflight_checks,
) -> None:
    contract, spec, imported = _v22_capability_import_fixture(
        tmp_path,
        execution_mode="closed_loop_preflight",
    )
    imported["preflight_go"] = True
    imported["preflight_checks"] = preflight_checks
    _rehash_capability_import(imported)

    with pytest.raises(
        PilotEvidenceError,
        match="capability import .*requires all.*checks|capability import validation failed",
    ):
        _validate_terminal_payload_marker(
            contract,
            spec.to_dict(),
            {
                "metrics": {},
                "gate_evidence": {"go": True},
                "capability": imported,
            },
            raw_root=tmp_path,
        )


def test_capability_import_gate_marker_cannot_disagree(tmp_path: Path) -> None:
    contract, spec, imported = _v22_capability_import_fixture(
        tmp_path,
        execution_mode="capability_probe",
    )

    with pytest.raises(PilotEvidenceError, match="sealed gate receipt"):
        _validate_terminal_payload_marker(
            contract,
            spec.to_dict(),
            {
                "metrics": {},
                "gate_evidence": {"go": False},
                "capability": imported,
            },
            raw_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("name", "mutate"),
    (
        (
            "taskset",
            lambda value: value.__setitem__("taskset_sha256", "0" * 64),
        ),
        (
            "task_kind",
            lambda value: _row(value, "action-01").__setitem__(
                "task_kind", "rule_application"
            ),
        ),
        (
            "prompt_hash",
            lambda value: _row(value, "action-01").__setitem__(
                "prompt_sha256", "0" * 64
            ),
        ),
        (
            "evaluable",
            lambda value: _row(value, "action-01").__setitem__(
                "evaluable", False
            ),
        ),
        (
            "truncation",
            lambda value: _row(value, "action-01").__setitem__(
                "truncation", True
            ),
        ),
        (
            "visible_tokens",
            lambda value: _row(value, "action-01").__setitem__(
                "visible_completion_tokens",
                _row(value, "action-01")["visible_completion_tokens"] + 1,
            ),
        ),
        (
            "visible_bytes",
            lambda value: _row(value, "action-01").__setitem__(
                "within_visible_limit", False
            ),
        ),
        (
            "finish",
            lambda value: _row(value, "action-01").__setitem__(
                "finish_reason", "length"
            ),
        ),
        (
            "route",
            lambda value: _row(value, "action-01").__setitem__(
                "response_route", ["forged"]
            ),
        ),
        (
            "cost",
            lambda value: _row(value, "action-01").__setitem__(
                "cost_usd", 99.0
            ),
        ),
        (
            "category_total",
            lambda value: value["category_totals"]["utility-ranking"].__setitem__(
                "registered_correct", 0
            ),
        ),
        (
            "strict_counter",
            lambda value: value.__setitem__("strict_parse_count", 0),
        ),
        (
            "budget_usage",
            lambda value: value["budget"]["accounted_usage"].__setitem__(
                "completion_tokens", 0
            ),
        ),
        (
            "action_hash",
            lambda value: _row(value, "action-01")["action"].__setitem__(
                "raw_output_hash", "0" * 64
            ),
        ),
    ),
)
def test_v4_validator_rejects_row_and_aggregate_tampering(
    name: str,
    mutate,
) -> None:
    result = _valid_v4()
    mutate(result)

    with pytest.raises(PilotEvidenceError, match="capability v3|capability v4"):
        _validate_capability_v4(result)


def test_recovered_json_is_reportable_but_cannot_be_promoted_to_success() -> None:
    result = _run_gate(
        RecoveryProvider(),
        budget_id="capability-v4-recovery-evidence",
        contracts=_contracts(),
    )
    _validate_capability_v4(result)

    recovered = _row(result, "action-01")
    assert recovered["parse_mode"] == "fenced_recovery"
    assert recovered["evaluable"] is True
    assert recovered["correct"] is False

    recovered["correct"] = True
    with pytest.raises(PilotEvidenceError, match="success is inconsistent"):
        _validate_capability_v4(result)


def test_pre_response_failure_allows_null_served_model_and_keeps_itt() -> None:
    result = _pre_response_failure_v4()
    failed = _row(result, "action-01")

    assert failed["served_model"] is None
    assert failed["interface_status"] == "provider_error"
    assert failed["evaluable"] is False
    assert failed["correct"] is False
    assert len(result["rows"]) == 30
    assert result["budget"]["completed_calls"] == 30
    assert result["provider_failure_count"] == 1
    assert {
        category: totals["registered_total"]
        for category, totals in result["category_totals"].items()
    } == {
        "utility-ranking": 12,
        "rule-application": 12,
        "rule-proposal": 6,
    }

    _validate_capability_v4(result)


@pytest.mark.parametrize("served_model", ("", [], 0))
def test_provider_failure_rejects_invalid_non_null_served_model(
    served_model,
) -> None:
    result = _pre_response_failure_v4()
    _row(result, "action-01")["served_model"] = served_model

    with pytest.raises(PilotEvidenceError, match="invalid served_model"):
        _validate_capability_v4(result)


@pytest.mark.parametrize(
    "provider_factory",
    (
        CapabilityFixtureProvider,
        GeminiLengthProvider,
        _InvalidFinishProvider,
    ),
    ids=("success", "truncation", "invalid-finish"),
)
def test_non_provider_failure_requires_served_model(provider_factory) -> None:
    result = _run_gate(
        provider_factory(),
        budget_id=f"capability-v4-served-model-{provider_factory.__name__}",
        contracts=_contracts(),
    )
    _row(result, "action-01")["served_model"] = None

    with pytest.raises(PilotEvidenceError, match="lacks served_model"):
        _validate_capability_v4(result)


def test_provider_failure_requires_explicit_served_model_field() -> None:
    result = _pre_response_failure_v4()
    del _row(result, "action-01")["served_model"]

    with pytest.raises(PilotEvidenceError, match="lacks served_model field"):
        _validate_capability_v4(result)


@pytest.mark.parametrize(
    "mutation",
    ("status", "evidence", "insufficient-evidence", "duplicate-evidence"),
)
def test_proposal_admission_and_grounding_cannot_be_forged(
    mutation: str,
) -> None:
    result = _valid_v4()
    proposal = _row(result, "proposal-01")
    if mutation == "status":
        proposal["candidate_status"] = "rejected"
    elif mutation == "evidence":
        proposal["candidate_supporting_episode_ids"] = [
            "invented-episode-1",
            "invented-episode-2",
        ]
    elif mutation == "insufficient-evidence":
        proposal["candidate_supporting_episode_ids"] = proposal[
            "candidate_supporting_episode_ids"
        ][:1]
    else:
        episode_id = proposal["candidate_supporting_episode_ids"][0]
        proposal["candidate_supporting_episode_ids"] = [episode_id, episode_id]

    with pytest.raises(
        PilotEvidenceError,
        match="forges admission status|forges evidence grounding",
    ):
        _validate_capability_v4(result)


def test_v4_validator_accepts_legal_hidden_semantic_mismatch() -> None:
    result = _run_gate(
        EquivalentConditionToleranceProvider(),
        budget_id="capability-v4-equivalent-tolerance-evidence",
        contracts=_contracts(),
    )

    proposals = [
        row for row in result["rows"] if row["task_kind"] == "rule_proposal"
    ]
    assert all(row["semantic_match"] is False for row in proposals)
    assert all(row["legal"] is True for row in proposals)
    assert all(row["correct"] is True for row in proposals)
    _validate_capability_v4(result)


def test_v4_validator_preserves_verifier_rejected_evidence_as_illegal() -> None:
    result = _run_gate(
        WrongSemanticEvidenceProvider("episode_id"),
        budget_id="capability-v4-rejected-evidence-validator",
        contracts=_contracts(),
    )

    proposal = _row(result, "proposal-01")
    assert proposal["semantic_candidate_accepted"] is False
    assert proposal["candidate_status"] == "rejected"
    assert proposal["legal"] is False
    assert proposal["correct"] is False
    _validate_capability_v4(result)
