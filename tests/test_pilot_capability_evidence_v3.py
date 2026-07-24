from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from test_pilot_capability import (
    CapabilityFixtureProvider,
    RecoveryProvider,
    _contracts,
    _run_gate,
)
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_evidence import (
    PilotEvidenceError,
    _validate_capability_v3,
    _validate_terminal_payload_marker,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v1.yaml"


def _valid_v3() -> dict:
    result = _run_gate(
        CapabilityFixtureProvider(),
        budget_id="capability-v3-evidence",
        contracts=_contracts(),
    )
    _validate_capability_v3(result)
    return result


def _row(result: dict, task_id: str) -> dict:
    return next(row for row in result["rows"] if row["task_id"] == task_id)


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


def test_terminal_capability_entry_routes_v2_and_v3_readers(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="capability-preflight")[0]

    for capability in (_valid_v2(), _valid_v3()):
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
def test_v3_validator_rejects_row_and_aggregate_tampering(
    name: str,
    mutate,
) -> None:
    result = _valid_v3()
    mutate(result)

    with pytest.raises(PilotEvidenceError, match="capability v3"):
        _validate_capability_v3(result)


def test_recovered_json_is_reportable_but_cannot_be_promoted_to_success() -> None:
    result = _run_gate(
        RecoveryProvider(),
        budget_id="capability-v3-recovery-evidence",
        contracts=_contracts(),
    )
    _validate_capability_v3(result)

    recovered = _row(result, "action-01")
    assert recovered["parse_mode"] == "fenced_recovery"
    assert recovered["evaluable"] is True
    assert recovered["correct"] is False

    recovered["correct"] = True
    with pytest.raises(PilotEvidenceError, match="success is inconsistent"):
        _validate_capability_v3(result)


@pytest.mark.parametrize("mutation", ("status", "evidence"))
def test_proposal_admission_and_grounding_cannot_be_forged(
    mutation: str,
) -> None:
    result = _valid_v3()
    proposal = _row(result, "proposal-01")
    if mutation == "status":
        proposal["candidate_status"] = "rejected"
    else:
        proposal["candidate_supporting_episode_ids"] = [
            "invented-episode-1",
            "invented-episode-2",
        ]

    with pytest.raises(
        PilotEvidenceError,
        match="forges admission status|forges evidence grounding",
    ):
        _validate_capability_v3(result)
