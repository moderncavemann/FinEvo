from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from verified_memory.pilot_capability import (
    CAPABILITY_SCHEMA_VERSION,
    CAPABILITY_TASKSET_SHA256,
    build_capability_tasks,
)
from verified_memory.pilot_checkpoint import (
    PILOT_CHECKPOINT_SCHEMA_VERSION_V4,
    V211_LONG_CONTEXT_PREFLIGHT_CHECKPOINT_PURPOSE,
)
from verified_memory.pilot_v211_gate import (
    PilotV211GateError,
    build_v211_post_gate_authority,
    canonical_sha256,
    runner_reservations_from_v211_gate,
    verified_v211_gate_authority_binding,
    verify_v211_gate_receipt,
)
from verified_memory.pilot_v211_parent_import import (
    V211_SOURCE_MANIFEST_CONTENT_SHA256,
    V211_SOURCE_MANIFEST_FILE_SHA256,
)


CONTRACT_SHA256 = "1" * 64
RELEASE_COMMIT = "2" * 40
LEDGER_HEAD = "3" * 64
SOURCE_MANIFEST_HASHES = {
    "file_sha256": V211_SOURCE_MANIFEST_FILE_SHA256,
    "content_sha256": V211_SOURCE_MANIFEST_CONTENT_SHA256,
}
PROFILES = {
    "gpt52_main": {
        "runtime_model": "openai/gpt-5.2-2025-12-11",
        "requested_model": "gpt-5.2-2025-12-11",
        "served_model": "gpt-5.2-2025-12-11",
        "price_source": (
            "https://developers.openai.com/api/docs/models/gpt-5.2"
        ),
        "price_captured_at": "2026-07-22",
    },
    "gpt56_diagnostic": {
        "runtime_model": "openai/gpt-5.6-sol",
        "requested_model": "gpt-5.6-sol",
        "served_model": "gpt-5.6-sol",
        "price_source": (
            "https://developers.openai.com/api/docs/models/gpt-5.6-sol"
        ),
        "price_captured_at": "2026-07-31",
    },
}


def _usage(
    *,
    prompt_tokens: int = 100,
    completion_tokens: int = 100,
    cost_usd: float = 0.01,
) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost_usd,
    }


def _profile_fields(model_id: str) -> dict:
    profile = PROFILES[model_id]
    return {
        "provider": "openai",
        "request_profile_id": model_id,
        "request_provider_pin": ["OpenAI-direct"],
        "request_artifact_identity": {
            "route": "OpenAI-direct",
            "served_snapshot": profile["served_model"],
        },
        "request_price_snapshot_source": profile["price_source"],
        "request_price_snapshot_captured_at": (
            profile["price_captured_at"]
        ),
        "attempts": 1,
        "temperature_dispatch": "omitted_unsupported",
        "request_parameters": [
            "max_completion_tokens",
            "messages",
            "model",
            "reasoning_effort",
            "response_format",
        ],
        "provider_sdk_name": "openai-python",
        "provider_sdk_version": "1.2.3",
        "provider_request_id": "req-fixture",
        "cached_prompt_tokens": 0,
        "reasoning_tokens": 60,
        "visible_completion_tokens": 40,
        "finish_reason": "stop",
        "native_finish_reason": "stop",
        "response_completed": True,
        "output_disposition": "accepted",
    }


def _capability_artifact(
    model_id: str,
    *,
    capability_pass: bool = True,
    parse_failure_index: int | None = None,
) -> dict:
    profile = PROFILES[model_id]
    rows = []
    completions = []
    frozen_tasks = build_capability_tasks()
    deliberately_incorrect = {28, 29} if not capability_pass else set()
    for index in range(30):
        frozen_task = frozen_tasks[index]
        semantic = index >= 24
        task_kind = frozen_task.task_kind
        usage = _usage()
        task_id = frozen_task.task_id
        parse_success = index != parse_failure_index
        legal = parse_success and index not in deliberately_incorrect
        correct = parse_success and legal
        category = frozen_task.category
        row = {
            "schema_version": CAPABILITY_SCHEMA_VERSION,
            "task_id": task_id,
            "category": category,
            "task_kind": task_kind,
            "output_contract_id": frozen_task.output_contract_id,
            "taskset_sha256": CAPABILITY_TASKSET_SHA256,
            "prompt_sha256": hashlib.sha256(
                frozen_task.prompt.encode("utf-8")
            ).hexdigest(),
            "correct": correct,
            "legal": legal,
            "interface_valid": True,
            "evaluable": True,
            "truncation": False,
            "finish_contract_valid": True,
            "within_visible_limit": True,
            "requested_model": profile["requested_model"],
            "served_model": profile["served_model"],
            "response_provider": "OpenAI-direct",
            "response_route": "direct",
            **_profile_fields(model_id),
            "usage": deepcopy(usage),
            "provider_reported_usage": deepcopy(usage),
            "provider_reported_usage_available": True,
            "budget_accounted_usage": deepcopy(usage),
            "request_max_completion_tokens": 4_096,
            "visible_json_max_bytes": 4_096 if semantic else 512,
            "output_bytes": 64,
            "provider_error": None,
            "parse_mode": (
                "parse_failure" if not parse_success else "exact_json"
            ),
            "strict_parse": parse_success,
            "accepted_parse_mode": parse_success,
            "action": None if semantic else {"clipped": False},
        }
        rows.append(row)
        completions.append(
            {
                "label": f"capability:{task_id}",
                "usage": deepcopy(usage),
            }
        )
    category_totals = {}
    thresholds = {
        "utility-ranking": 10,
        "rule-application": 10,
        "rule-proposal": 5,
    }
    for category, required in thresholds.items():
        category_rows = [row for row in rows if row["category"] == category]
        registered_correct = sum(row["correct"] for row in category_rows)
        category_totals[category] = {
            "correct": registered_correct,
            "denominator": len(category_rows),
            "required": required,
            "registered_correct": registered_correct,
            "registered_total": len(category_rows),
            "evaluable_count": len(category_rows),
            "conditional_correct": registered_correct,
            "conditional_accuracy": registered_correct / len(category_rows),
            "interface_failure_count": 0,
        }
    checks = {
        category: row["registered_correct"] >= row["required"]
        for category, row in category_totals.items()
    }
    conditional_checks = {
        category: (
            row["conditional_accuracy"]
            >= row["required"] / row["registered_total"]
        )
        for category, row in category_totals.items()
    }
    recomputed_capability_pass = all(checks.values())
    assert recomputed_capability_pass is capability_pass
    accounted = {
        "prompt_tokens": 3_000,
        "completion_tokens": 3_000,
        "total_tokens": 6_000,
        "cost_usd": pytest.approx(0.30),
    }
    # Replace pytest's comparison helper with its exact numeric target before
    # hashing the JSON artifact.
    accounted["cost_usd"] = 0.30
    payload = {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "taskset_sha256": CAPABILITY_TASKSET_SHA256,
        "provider_model": profile["runtime_model"],
        "pass": recomputed_capability_pass,
        "checks": checks,
        "category_totals": category_totals,
        "task_output_contracts": {
            "actor-action": {
                "request_max_completion_tokens": 4_096,
                "visible_json_max_bytes": 512,
                "accepted_parse_modes": ["exact_json"],
                "required_finish_reason": "stop",
            },
            "semantic-proposal": {
                "request_max_completion_tokens": 4_096,
                "visible_json_max_bytes": 4_096,
                "accepted_parse_modes": ["exact_json"],
                "required_finish_reason": "stop",
            },
        },
        "prompt_tier_gate": {
            "upper_bound_method": "utf8-bytes-plus-256-v1",
            "ceiling_tokens": 200_000,
            "maximum_upper_bound_tokens": 12_000,
            "passed": True,
        },
        "interface_gate": {"pass": True, "failure_count": 0},
        "capability_assessment": {
            "status": "pass" if recomputed_capability_pass else "fail",
            "pass": recomputed_capability_pass,
            "checks": conditional_checks,
        },
        "provider_failure_count": 0,
        "parse_failure_count": (
            0 if parse_failure_index is None else 1
        ),
        "recovered_parse_count": 0,
        "strict_parse_count": (
            30 if parse_failure_index is None else 29
        ),
        "truncation_count": 0,
        "rows": rows,
        "budget": {
            "completed_calls": 30,
            "active_calls": 0,
            "accounted_usage": accounted,
            "completions": completions,
        },
    }
    return {
        "run_id": f"capability-{model_id}",
        "artifact_sha256": canonical_sha256(payload),
        "payload": payload,
    }


def _preflight_artifact(
    model_id: str,
    *,
    semantic_parse_failure_ordinal: int | None = None,
) -> dict:
    profile = PROFILES[model_id]
    ledger_run_id = f"preflight-{model_id}"
    checkpoint_run_id = f"{ledger_run_id}--actor-preflight"
    rows = []
    proposal_outcomes = []
    for index in range(32):
        semantic = index >= 24
        usage = _usage(
            prompt_tokens=200,
            completion_tokens=200,
            cost_usd=0.02,
        )
        row = {
                "call_index": index,
                "call_kind": "semantic" if semantic else "action",
                "decision_t": (
                    (3, 6, 9, 12)[(index - 24) // 2]
                    if semantic
                    else index // 2
                ),
                "agent_id": index % 2,
                "model": profile["requested_model"],
                "served_model": profile["served_model"],
                "served_provider": "OpenAI-direct",
                "served_route": "direct",
                **_profile_fields(model_id),
                "usage": usage,
                "reasoning_tokens": 120,
                "visible_completion_tokens": 80,
                "error_type": None,
                "task_cap": {
                    "max_visible_tokens": 4_096,
                    "max_visible_json_bytes": (
                        4_096 if semantic else 1_024
                    ),
                },
                "parse_disposition": {
                    "parse_status": (
                        "failure"
                        if semantic
                        and index - 24 == semantic_parse_failure_ordinal
                        else "success"
                    ),
                    "parse_mode": (
                        "parse_failure"
                        if semantic
                        and index - 24 == semantic_parse_failure_ordinal
                        else "exact_json"
                    ),
                    "accepted": not (
                        semantic
                        and index - 24 == semantic_parse_failure_ordinal
                    ),
                    **({} if semantic else {"clipped": False}),
                },
                "prompt_token_upper_bound": 20_000,
                "prompt_token_upper_bound_method": (
                    "utf8-bytes-plus-256-v1"
                ),
                "prompt_tier_ceiling_tokens": 200_000,
                "prompt_hash": hashlib.sha256(
                    f"prompt-{model_id}-{index}".encode()
                ).hexdigest(),
                "raw_output_hash": hashlib.sha256(
                    f"output-{model_id}-{index}".encode()
                ).hexdigest(),
            }
        rows.append(row)
        if semantic:
            failed = index - 24 == semantic_parse_failure_ordinal
            events = [] if failed else [{"event": f"accepted-{index}"}]
            proposal_outcomes.append(
                {
                    "call_index": index,
                    "current_t": row["decision_t"],
                    "agent_id": row["agent_id"],
                    "prompt_hash": row["prompt_hash"],
                    "raw_output_hash": row["raw_output_hash"],
                    "candidate_parse_status": (
                        "failure" if failed else "success"
                    ),
                    "candidate_parse_mode": (
                        "parse_failure" if failed else "exact_json"
                    ),
                    "failure_reason": (
                        "recorded candidate parse failure"
                        if failed
                        else None
                    ),
                    "semantic_events": events,
                    "semantic_events_hash": canonical_sha256(events),
                }
            )
    provider_calls_hash = canonical_sha256(rows)
    provider_denominator = {
        "planned_calls": 32,
        "observed_calls": 32,
        "successful_terminal_calls": 32,
        "failed_calls": 0,
        "action_calls": 24,
        "semantic_calls": 8,
        "semantic_candidate_parse_failures": (
            0 if semantic_parse_failure_ordinal is None else 1
        ),
    }
    provider_totals = {
        "call_count": 32,
        "action_call_count": 24,
        "semantic_call_count": 8,
        "prompt_tokens": 6_400,
        "completion_tokens": 6_400,
        "reasoning_tokens": 3_840,
        "visible_completion_tokens": 2_560,
        "visible_output_bytes": 2_048,
        "cost_usd": 0.64,
        "hosted": True,
        "requested_models": [profile["requested_model"]],
        "served_models": [profile["served_model"]],
        "served_providers": ["OpenAI-direct"],
        "served_routes": ["direct"],
    }
    budget = {
        "completed_calls": 32,
        "active_calls": 0,
        "accounted_usage": {
            "prompt_tokens": 6_400,
            "completion_tokens": 6_400,
            "total_tokens": 12_800,
            "cost_usd": 0.64,
        },
    }
    journal = {
        "enabled": True,
        "journal_sha256": hashlib.sha256(
            f"journal-{model_id}".encode()
        ).hexdigest(),
        "event_count": 64,
        "completion_event_count": 32,
        "parse_disposition_event_count": 32,
        "run_id": checkpoint_run_id,
        "contract_hash": CONTRACT_SHA256,
        "path_name": f"{checkpoint_run_id}-provider-calls.json",
    }
    checkpoint = {
        "schema_version": PILOT_CHECKPOINT_SCHEMA_VERSION_V4,
        "checkpoint_purpose": (
            V211_LONG_CONTEXT_PREFLIGHT_CHECKPOINT_PURPOSE
        ),
        "run_config": {
            "run_id": checkpoint_run_id,
            "pilot_contract_hash": CONTRACT_SHA256,
            "num_agents": 2,
            "episode_length": 12,
            "action_max_tokens": 4_096,
            "rule_max_tokens": 4_096,
            "action_max_visible_json_bytes": 1_024,
            "rule_max_visible_json_bytes": 4_096,
            "prompt_tier_ceiling_tokens": 200_000,
            "accepted_action_parse_modes": ["exact_json"],
            "accepted_semantic_parse_modes": ["exact_json"],
            "fail_on_clipped_action": True,
        },
        "provider_calls": rows,
        "provider_calls_hash": provider_calls_hash,
        "proposal_outcomes": proposal_outcomes,
        "proposal_outcomes_hash": canonical_sha256(proposal_outcomes),
        "provider_denominator": provider_denominator,
        "provider_totals": provider_totals,
        "provider_totals_hash": canonical_sha256(provider_totals),
        "budget_snapshot_at_checkpoint": budget,
        "budget_snapshot_hash": canonical_sha256(budget),
        "provider_call_journal_binding": journal,
        "provider_call_journal_binding_hash": canonical_sha256(journal),
    }
    checkpoint["checkpoint_hash"] = canonical_sha256(checkpoint)
    exactness = {
        "schema_version": (
            "finevo-v2.11-long-context-preflight-exactness-receipt-v1"
        ),
        "checkpoint_hash": checkpoint["checkpoint_hash"],
        "num_agents": 2,
        "completed_months": 12,
        "provider_calls_during_verification": 0,
        "verified_components": {"all_fixture_components": True},
        "provider_calls_hash": checkpoint["provider_calls_hash"],
        "proposal_outcomes_hash": checkpoint["proposal_outcomes_hash"],
        "provider_totals_hash": checkpoint["provider_totals_hash"],
        "budget_snapshot_hash": checkpoint["budget_snapshot_hash"],
        "provider_call_journal_binding_hash": checkpoint[
            "provider_call_journal_binding_hash"
        ],
        "provider_denominator": provider_denominator,
    }
    exactness["receipt_hash"] = canonical_sha256(exactness)
    return {
        "ledger_run_id": ledger_run_id,
        "checkpoint_run_id": checkpoint_run_id,
        "run_spec_sha256": hashlib.sha256(
            f"spec-{model_id}".encode()
        ).hexdigest(),
        "checkpoint_artifact_sha256": canonical_sha256(checkpoint),
        "checkpoint": checkpoint,
        "exactness_artifact_sha256": canonical_sha256(exactness),
        "exactness": exactness,
    }


def _science_run_ids() -> dict[str, list[str]]:
    return {
        "gpt52_main": [
            f"gpt52-science-{index:03d}" for index in range(125)
        ],
        "gpt56_diagnostic": [
            f"gpt56-science-{index:03d}" for index in range(6)
        ],
    }


def _inputs(
    *,
    statuses: dict[str, str] | None = None,
) -> dict:
    return {
        "contract_sha256": CONTRACT_SHA256,
        "release_tag": "pilot-v2.11-science",
        "release_commit": RELEASE_COMMIT,
        "parent_import_run_id": "parent-import",
        "capability_artifacts": {
            model_id: _capability_artifact(
                model_id,
                capability_pass=(
                    statuses is None
                    or statuses[model_id] != "capability-no-go"
                ),
            )
            for model_id in PROFILES
        },
        "preflight_artifacts": {
            model_id: _preflight_artifact(model_id)
            for model_id in PROFILES
        },
        "model_terminal_statuses": statuses
        or {
            "gpt52_main": "eligible",
            "gpt56_diagnostic": "eligible",
        },
        "pre_science_actual_storage_bytes": 100_000_000,
        "ledger_event_chain_head": LEDGER_HEAD,
        "science_run_ids_by_model": _science_run_ids(),
        "source_manifest_hashes": SOURCE_MANIFEST_HASHES,
    }


def _rehash_capability(envelope: dict) -> None:
    envelope["artifact_sha256"] = canonical_sha256(envelope["payload"])


def test_gate_builds_exact_124_call_authority_and_file_binding(
    tmp_path: Path,
) -> None:
    receipt = build_v211_post_gate_authority(**_inputs())

    verified = verify_v211_gate_receipt(
        receipt,
        expected_contract_sha256=CONTRACT_SHA256,
        expected_git_commit=RELEASE_COMMIT,
    )
    assert verified["receipt_sha256"] == receipt["receipt_sha256"]
    assert receipt["go"] is True
    assert receipt["actuals"]["hosted_completions"] == 124
    assert receipt["denominator"]["registered_hosted_calls"] == 5_940
    assert (
        receipt["projection"]["remaining_science_hosted_completions"]
        == 5_816
    )
    assert receipt["projection"]["new_hosted_completions"] == 5_940
    assert receipt["denominator"]["eligible_model_ids"] == [
        "gpt52_main",
        "gpt56_diagnostic",
    ]
    ledger_ids = ["parent-import"]
    checkpoint_ids = []
    for model_id in sorted(PROFILES):
        bindings = receipt["bindings"]["gate_artifacts"][model_id]
        ledger_ids.extend(
            [
                bindings["capability"]["run_id"],
                bindings["preflight"]["ledger_run_id"],
            ]
        )
        checkpoint_ids.append(bindings["preflight"]["checkpoint_run_id"])
    for model_id in sorted(PROFILES):
        ledger_ids.extend(_science_run_ids()[model_id])
    assert len(ledger_ids) == 136
    assert not set(checkpoint_ids) & set(ledger_ids)
    assert receipt["denominator"]["all_ledger_run_ids_sha256"] == (
        canonical_sha256(ledger_ids)
    )
    for model_id in PROFILES:
        decision = receipt["model_decisions"][model_id]
        assert decision["sample_counts"] == {
            "action": 48,
            "semantic": 14,
        }
        assert decision["interface_pass"] is True
        for call_kind, expected_count in (
            ("action", 48),
            ("semantic", 14),
        ):
            reservation = receipt["dispatch_reservations"][model_id][
                call_kind
            ]
            assert reservation["sample_count"] == expected_count
            assert reservation["reserve_multiplier"] == 1.25
            assert reservation["reserved_p95"]["completion_tokens"] == 250

    relative = Path(
        "experiment_results/pilot-v2.11/raw/post-gate/gate.json"
    )
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    binding = verified_v211_gate_authority_binding(
        relative.as_posix(),
        repo_root=tmp_path,
        expected_git_commit=RELEASE_COMMIT,
        expected_contract_sha256=CONTRACT_SHA256,
    )
    assert set(binding) == {
        "receipt_path",
        "receipt_file_sha256",
        "receipt_content_sha256",
        "git_commit",
        "reservations",
    }
    assert set(binding["reservations"]) == {
        "openai/gpt-5.2-2025-12-11",
        "openai/gpt-5.6-sol",
    }
    runner_rows = runner_reservations_from_v211_gate(
        relative.as_posix(),
        repo_root=tmp_path,
        expected_git_commit=RELEASE_COMMIT,
        expected_contract_sha256=CONTRACT_SHA256,
    )
    authority = runner_rows["openai/gpt-5.2-2025-12-11"]["action"][
        "authority"
    ]
    assert authority["source_authority_receipt_path"] == relative.as_posix()
    assert authority["source_authority_receipt_content_sha256"] == receipt[
        "receipt_sha256"
    ]
    assert authority["source_release_commit"] == RELEASE_COMMIT


def test_gpt56_capability_no_go_is_model_scoped_after_all_124_calls() -> None:
    inputs = _inputs(
        statuses={
            "gpt52_main": "eligible",
            "gpt56_diagnostic": "capability-no-go",
        }
    )
    receipt = build_v211_post_gate_authority(**inputs)

    assert receipt["go"] is True
    assert receipt["actuals"]["hosted_completions"] == 124
    assert receipt["denominator"]["eligible_model_ids"] == ["gpt52_main"]
    assert receipt["denominator"]["no_go_ledger_cells_by_model"] == {
        "gpt52_main": 0,
        "gpt56_diagnostic": 6,
    }
    assert len(
        receipt["denominator"]["no_go_science_run_ids_by_model"][
            "gpt56_diagnostic"
        ]
    ) == 6
    assert receipt["projection"]["registered_hosted_completions"] == 5_940
    assert receipt["projection"]["new_hosted_completions"] == 5_604
    assert (
        receipt["projection"]["remaining_science_hosted_completions"]
        == 5_480
    )
    assert set(receipt["dispatch_reservations"]) == {"gpt52_main"}
    assert len(
        receipt["denominator"]["science_run_ids_by_model"]["gpt56_diagnostic"]
    ) == 6


def test_both_model_no_go_preserves_matrix_but_emits_global_no_go() -> None:
    receipt = build_v211_post_gate_authority(
        **_inputs(
            statuses={
                "gpt52_main": "capability-no-go",
                "gpt56_diagnostic": "capability-no-go",
            }
        )
    )

    assert receipt["go"] is False
    assert receipt["reasons"] == ["no-dispatch-eligible-models"]
    assert receipt["actuals"]["hosted_completions"] == 124
    assert receipt["projection"]["new_hosted_completions"] == 124
    assert receipt["projection"]["remaining_science_hosted_completions"] == 0
    assert receipt["denominator"]["no_go_ledger_cells_by_model"] == {
        "gpt52_main": 125,
        "gpt56_diagnostic": 6,
    }
    assert receipt["denominator"]["registered_ledger_cells"] == 136


def test_capability_parse_failure_keeps_5_of_6_threshold_separate() -> None:
    inputs = _inputs(
        statuses={
            "gpt52_main": "eligible",
            "gpt56_diagnostic": "interface-no-go",
        }
    )
    inputs["capability_artifacts"]["gpt56_diagnostic"] = (
        _capability_artifact(
            "gpt56_diagnostic",
            capability_pass=True,
            parse_failure_index=29,
        )
    )

    receipt = build_v211_post_gate_authority(**inputs)

    decision = receipt["model_decisions"]["gpt56_diagnostic"]
    assert decision["terminal_status"] == "interface-no-go"
    assert decision["capability_pass"] is True
    assert decision["interface_pass"] is False
    semantic_gate = decision["interface_gates"]["semantic"]
    assert semantic_gate["observed_sample_count"] == 14
    assert semantic_gate["failed_sample_indices"] == [5]
    assert "parse-not-exact" in semantic_gate["reasons"]
    assert len(
        receipt["denominator"]["science_run_ids_by_model"][
            "gpt56_diagnostic"
        ]
    ) == 6


def test_preflight_record_and_skip_candidate_failure_is_fully_accounted() -> None:
    inputs = _inputs()
    inputs["preflight_artifacts"]["gpt56_diagnostic"] = (
        _preflight_artifact(
            "gpt56_diagnostic",
            semantic_parse_failure_ordinal=7,
        )
    )

    receipt = build_v211_post_gate_authority(**inputs)

    decision = receipt["model_decisions"]["gpt56_diagnostic"]
    assert decision["terminal_status"] == "eligible"
    assert decision["interface_pass"] is True
    assert (
        decision[
            "recorded_preflight_semantic_candidate_parse_failures"
        ]
        == 1
    )
    assert (
        decision["interface_gates"]["semantic"]["observed_sample_count"]
        == 14
    )
    assert decision["interface_gates"]["semantic"]["failed_sample_indices"] == []


def test_gate_rejects_payload_tamper_before_projection() -> None:
    inputs = _inputs()
    envelope = inputs["capability_artifacts"]["gpt52_main"]
    envelope["payload"]["rows"][0]["finish_reason"] = "length"

    with pytest.raises(PilotV211GateError, match="artifact canonical hash"):
        build_v211_post_gate_authority(**inputs)


def test_gate_rejects_missing_registered_call_even_if_artifact_rehashed() -> None:
    inputs = _inputs()
    envelope = inputs["capability_artifacts"]["gpt52_main"]
    envelope["payload"]["rows"].pop()
    _rehash_capability(envelope)

    with pytest.raises(PilotV211GateError, match="exactly 30 calls"):
        build_v211_post_gate_authority(**inputs)


def test_gate_rejects_checkpoint_run_id_used_as_ledger_cell() -> None:
    inputs = _inputs()
    envelope = inputs["preflight_artifacts"]["gpt52_main"]
    envelope["ledger_run_id"] = envelope["checkpoint_run_id"]

    with pytest.raises(PilotV211GateError, match="not bound to its ledger cell"):
        build_v211_post_gate_authority(**inputs)


def test_gate_rejects_frozen_profile_mismatch() -> None:
    inputs = _inputs()
    envelope = inputs["capability_artifacts"]["gpt52_main"]
    envelope["payload"]["rows"][0][
        "request_profile_id"
    ] = "gpt56_diagnostic"
    _rehash_capability(envelope)

    with pytest.raises(PilotV211GateError, match="profile mismatch"):
        build_v211_post_gate_authority(**inputs)


def test_gate_rejects_budget_double_count_even_if_artifact_rehashed() -> None:
    inputs = _inputs()
    envelope = inputs["capability_artifacts"]["gpt52_main"]
    accounted = envelope["payload"]["budget"]["accounted_usage"]
    accounted["prompt_tokens"] += 100
    accounted["total_tokens"] += 100
    _rehash_capability(envelope)

    with pytest.raises(PilotV211GateError, match="double-count/mismatch"):
        build_v211_post_gate_authority(**inputs)


def test_gate_rejects_self_consistent_substituted_source_manifest() -> None:
    inputs = _inputs()
    inputs["source_manifest_hashes"] = {
        "file_sha256": "a" * 64,
        "content_sha256": "b" * 64,
    }

    with pytest.raises(PilotV211GateError, match="frozen V2.11 manifest"):
        build_v211_post_gate_authority(**inputs)


def test_receipt_verifier_rejects_rehashed_source_manifest_substitution() -> None:
    receipt = build_v211_post_gate_authority(**_inputs())
    receipt["bindings"]["source_manifest_hashes"] = {
        "file_sha256": "a" * 64,
        "content_sha256": "b" * 64,
    }
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_sha256(receipt)

    with pytest.raises(PilotV211GateError, match="frozen V2.11 manifest"):
        verify_v211_gate_receipt(receipt)


def test_receipt_tamper_and_symlink_path_fail_closed(tmp_path: Path) -> None:
    receipt = build_v211_post_gate_authority(**_inputs())
    tampered = deepcopy(receipt)
    tampered["projection"]["new_hosted_completions"] -= 1

    with pytest.raises(PilotV211GateError, match="self-hash mismatch"):
        verify_v211_gate_receipt(tampered)

    real = tmp_path / "real.json"
    real.write_text(
        json.dumps(receipt, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    relative = Path("experiment_results/pilot-v2.11/raw/gate.json")
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.symlink_to(real)
    with pytest.raises(PilotV211GateError, match="symlink"):
        verified_v211_gate_authority_binding(
            relative.as_posix(),
            repo_root=tmp_path,
            expected_git_commit=RELEASE_COMMIT,
        )

    symlink_root = tmp_path / "directory-symlink-root"
    external = tmp_path / "external"
    external.mkdir()
    symlink_root.mkdir()
    (symlink_root / "experiment_results").symlink_to(
        external,
        target_is_directory=True,
    )
    with pytest.raises(PilotV211GateError, match="symlinks or unsafe nodes"):
        verified_v211_gate_authority_binding(
            "experiment_results/pilot-v2.11/gate.json",
            repo_root=symlink_root,
            expected_git_commit=RELEASE_COMMIT,
        )
