from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import tests.test_pilot_v2111_gate as v2111_fixtures
import tests.test_pilot_v211_gate as v211_fixtures
from verified_memory.budget import UsageRecord
from verified_memory.pilot_budget import ParentBudgetDebit
import verified_memory.pilot_v2112_gate as gate_module
from verified_memory.pilot_v2112_gate import (
    PilotV2112GateError,
    V2112_CAPABILITY_EVIDENCE_USE,
    V2112_GATE_RELEASE_TAG,
    V2112_PREFLIGHT_EXACTNESS_SCHEMA_VERSION,
    build_v2112_post_gate_authority,
    canonical_sha256,
    runner_reservations_from_v2112_gate_binding,
    verified_v2112_gate_authority_binding,
    verify_v2112_gate_receipt,
)
from verified_memory.runner import (
    LONG_CONTEXT_PREFLIGHT_AUTHORITY_BY_ID,
    PROVIDER_CALL_JOURNAL_SCHEMA_VERSION,
    ContractEnvelopeBootstrapReservation,
    PreflightP95Reservation,
    V2111_CONTRACT_ENVELOPE_AUTHORITY_ID,
    V2112_CONTRACT_ENVELOPE_AUTHORITY_ID,
)


CONTRACT_SHA256 = "1" * 64
RELEASE_COMMIT = "2" * 40
LEDGER_HEAD = "3" * 64
MANIFEST_FILE_SHA256 = "4" * 64
MANIFEST_CONTENT_SHA256 = "5" * 64


@pytest.fixture(autouse=True)
def _frozen_fixture_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gate_module,
        "V2112_SOURCE_MANIFEST_FILE_SHA256",
        MANIFEST_FILE_SHA256,
    )
    monkeypatch.setattr(
        gate_module,
        "V2112_SOURCE_MANIFEST_CONTENT_SHA256",
        MANIFEST_CONTENT_SHA256,
    )
    monkeypatch.setattr(v211_fixtures, "CONTRACT_SHA256", CONTRACT_SHA256)
    monkeypatch.setattr(v2111_fixtures, "CONTRACT_SHA256", CONTRACT_SHA256)


def _source_run_id(model_id: str) -> str:
    return (
        "finevo-pilot-v2.11.1--capability-gate--"
        f"{model_id}--capability-probe--none--"
        "provider-preflight-default--s2010922376"
    )


def _preflight_ledger_run_id(model_id: str) -> str:
    return (
        "finevo-pilot-v2.11.2--long-context-preflight--"
        f"{model_id}--closed-loop-preflight--none--stage0-selected--"
        "s2010922376"
    )


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


def _seal(value: dict) -> dict:
    result = deepcopy(value)
    result["integrity"] = {"canonicalization": "json-sort-keys-utf8-v1"}
    result["integrity"]["content_sha256"] = canonical_sha256(result)
    return result


def _rehash_wrapper(binding: dict) -> None:
    wrapper = binding["payload"]
    wrapper["integrity"].pop("content_sha256", None)
    wrapper["integrity"]["content_sha256"] = canonical_sha256(wrapper)
    binding["wrapper_content_sha256"] = wrapper["integrity"]["content_sha256"]


def _capability_binding(model_id: str) -> dict:
    profile = gate_module._MODEL_PROFILES[model_id]
    capability = deepcopy(
        v2111_fixtures._capability_binding(model_id)["payload"]["capability"]
    )
    run_id = _source_run_id(model_id)
    capability["run_id"] = (
        "finevo-pilot-v2.11--capability-gate--"
        f"{model_id}--capability-probe--none--"
        "provider-preflight-default--s2010922376"
    )
    source_run_spec = {
        "contract_id": gate_module.V211_PARENT_CONTRACT_ID,
        "run_id": run_id,
        "stage_id": "capability-gate",
        "model_id": model_id,
        "runtime_model": profile["runtime_model"],
    }
    wrapper = _seal(
        {
            "schema_version": gate_module.V2112_CAPABILITY_WRAPPER_SCHEMA_VERSION,
            "child_release": {
                "contract_id": gate_module.V2112_GATE_CONTRACT_ID,
                "contract_sha256": CONTRACT_SHA256,
                "git_tag": V2112_GATE_RELEASE_TAG,
                "resolved_git_commit": RELEASE_COMMIT,
            },
            "parent_release": {
                "contract_id": gate_module.V211_PARENT_CONTRACT_ID,
                "contract_sha256": gate_module.V211_PARENT_CONTRACT_SHA256,
                "git_tag": gate_module.V211_PARENT_SCIENCE_TAG,
                "git_tag_object": gate_module.V211_PARENT_SCIENCE_TAG_OBJECT,
                "resolved_git_commit": gate_module.V211_PARENT_SCIENCE_COMMIT,
            },
            "source_manifest": {
                "path": gate_module.V2112_SOURCE_MANIFEST_PATH.as_posix(),
                "file_sha256": MANIFEST_FILE_SHA256,
                "content_sha256": MANIFEST_CONTENT_SHA256,
            },
            "source_capability_wrapper": {
                "path": (
                    "experiment_results/pilot-v2.11.1/raw/capability-gate/"
                    f"runs/{run_id}/capability.json"
                ),
                "byte_size": 25_000,
                "file_sha256": "6" * 64,
                "content_sha256": "7" * 64,
                "schema_version": (gate_module.V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION),
                "run_id": run_id,
                "run_spec": source_run_spec,
                "historical_source_calls": 30,
            },
            "capability": capability,
            "provider_construction_current_attempt": False,
            "provider_calls_current_attempt": 0,
            "hosted_provider_calls_current_attempt": 0,
            "current_attempt_usage": _usage(
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
            ),
            "imported_effect_cells": 0,
            "imported_preflight_samples": 0,
            "imported_checkpoint_artifacts": [],
            "imported_p95_authorities": [],
            "scientific_evidence": False,
            "evidence_scope": "preregistered_task_capability_gate",
            "evidence_use": V2112_CAPABILITY_EVIDENCE_USE,
        }
    )
    return {
        "model_id": model_id,
        "wrapper_content_sha256": wrapper["integrity"]["content_sha256"],
        "payload": wrapper,
        "provider_construction_during_verification": False,
        "provider_calls_during_verification": 0,
    }


def _bootstrap_reservations(
    model_id: str,
    *,
    target_run_spec_sha256: str,
    source_run_spec_sha256: str,
) -> dict[str, dict]:
    authority = LONG_CONTEXT_PREFLIGHT_AUTHORITY_BY_ID[
        V2112_CONTRACT_ENVELOPE_AUTHORITY_ID
    ]
    runtime_model = str(gate_module._MODEL_PROFILES[model_id]["runtime_model"])
    common = {
        "authority_id": authority.authority_id,
        "target_contract_id": authority.target_contract_id,
        "pilot_contract_hash": CONTRACT_SHA256,
        "pilot_tag": authority.target_release_tag,
        "pilot_commit": RELEASE_COMMIT,
        "model_id": model_id,
        "authorized_run_id": authority.target_run_id(model_id),
        "authorized_seed": authority.preflight_seed,
        "authorized_config_sha256": "8" * 64,
        "target_run_spec_sha256": target_run_spec_sha256,
        "source_contract_id": authority.source_contract_id,
        "source_contract_hash": gate_module.V211_PARENT_CONTRACT_SHA256,
        "source_tag": authority.source_release_tag,
        "source_commit": gate_module.V211_PARENT_SCIENCE_COMMIT,
        "source_run_id": authority.source_run_id(model_id),
        "source_run_spec_sha256": source_run_spec_sha256,
        "source_capability_file_sha256": "6" * 64,
        "source_capability_payload_sha256": "9" * 64,
        "source_group_sha256": "a" * 64,
        "capability_projection_sha256": "b" * 64,
        "policy_sha256": "c" * 64,
        "provider_profile_sha256": "d" * 64,
        "price_snapshot_sha256": "e" * 64,
        "source_projection_sha256": "f" * 64,
    }
    result = {}
    for call_kind in ("action", "semantic"):
        result[call_kind] = ContractEnvelopeBootstrapReservation(
            capability_projection=PreflightP95Reservation(
                model=runtime_model,
                call_kind=call_kind,
                sample_count=authority.capability_sample_counts[call_kind],
                raw_prompt_tokens=80.0,
                raw_completion_tokens=80.0,
                raw_cost_usd=0.008,
                reserved_usage=UsageRecord(
                    prompt_tokens=100,
                    completion_tokens=100,
                    cost_usd=0.01,
                ),
            ),
            envelope_usage=UsageRecord(
                prompt_tokens=authority.prompt_envelope_tokens,
                completion_tokens=authority.completion_envelope_tokens,
                cost_usd=authority.envelope_cost_usd[runtime_model],
            ),
            **common,
        ).to_dict()
    return {runtime_model: result}


def _journal(checkpoint: dict) -> dict:
    events = []
    previous = "0" * 64
    for row in checkpoint["provider_calls"]:
        completion_payload = {
            key: deepcopy(row[key])
            for key in (
                "call_kind",
                "decision_t",
                "agent_id",
                "prompt_hash",
                "raw_output_hash",
                "usage",
            )
        }
        disposition_payload = {
            key: deepcopy(row[key])
            for key in (
                "call_kind",
                "decision_t",
                "agent_id",
                "prompt_hash",
                "raw_output_hash",
            )
        }
        disposition_payload.update(deepcopy(row["parse_disposition"]))
        for event_type, payload in (
            ("completion_received", completion_payload),
            ("parse_disposition", disposition_payload),
        ):
            event = {
                "event_index": len(events),
                "event_type": event_type,
                "previous_event_sha256": previous,
                "payload": payload,
            }
            event["event_sha256"] = canonical_sha256(event)
            previous = event["event_sha256"]
            events.append(event)
    value = {
        "schema_version": PROVIDER_CALL_JOURNAL_SCHEMA_VERSION,
        "run_id": checkpoint["run_config"]["run_id"],
        "contract_hash": CONTRACT_SHA256,
        "events": events,
    }
    value["journal_sha256"] = canonical_sha256(value)
    return value


def _rehash_checkpoint_and_exactness(envelope: dict) -> None:
    checkpoint = envelope["checkpoint"]
    checkpoint["provider_calls_hash"] = canonical_sha256(checkpoint["provider_calls"])
    checkpoint["proposal_outcomes_hash"] = canonical_sha256(
        checkpoint["proposal_outcomes"]
    )
    checkpoint["provider_totals_hash"] = canonical_sha256(checkpoint["provider_totals"])
    checkpoint["budget_snapshot_hash"] = canonical_sha256(
        checkpoint["budget_snapshot_at_checkpoint"]
    )
    checkpoint["provider_call_journal_binding_hash"] = canonical_sha256(
        checkpoint["provider_call_journal_binding"]
    )
    checkpoint.pop("checkpoint_hash", None)
    checkpoint["checkpoint_hash"] = canonical_sha256(checkpoint)
    envelope["checkpoint_artifact_sha256"] = canonical_sha256(checkpoint)

    exactness = envelope["exactness"]
    exactness.update(
        {
            "schema_version": V2112_PREFLIGHT_EXACTNESS_SCHEMA_VERSION,
            "checkpoint_hash": checkpoint["checkpoint_hash"],
            "provider_calls_hash": checkpoint["provider_calls_hash"],
            "proposal_outcomes_hash": checkpoint["proposal_outcomes_hash"],
            "provider_totals_hash": checkpoint["provider_totals_hash"],
            "budget_snapshot_hash": checkpoint["budget_snapshot_hash"],
            "provider_call_journal_binding_hash": checkpoint[
                "provider_call_journal_binding_hash"
            ],
        }
    )
    exactness.pop("receipt_hash", None)
    exactness["receipt_hash"] = canonical_sha256(exactness)
    envelope["exactness_artifact_sha256"] = canonical_sha256(exactness)


def _preflight_artifact(model_id: str) -> dict:
    envelope = deepcopy(v211_fixtures._preflight_artifact(model_id))
    ledger_run_id = _preflight_ledger_run_id(model_id)
    checkpoint_run_id = f"{ledger_run_id}--actor-preflight"
    run_spec_sha256 = canonical_sha256(
        {
            "contract_id": gate_module.V2112_GATE_CONTRACT_ID,
            "run_id": ledger_run_id,
            "stage_id": "long-context-preflight",
            "model_id": model_id,
        }
    )
    source_spec_sha256 = canonical_sha256(
        _capability_binding(model_id)["payload"]["source_capability_wrapper"][
            "run_spec"
        ]
    )
    envelope["ledger_run_id"] = ledger_run_id
    envelope["checkpoint_run_id"] = checkpoint_run_id
    envelope["run_spec_sha256"] = run_spec_sha256
    checkpoint = envelope["checkpoint"]
    run_config = checkpoint["run_config"]
    run_config.update(
        {
            "run_id": checkpoint_run_id,
            "pilot_contract_hash": CONTRACT_SHA256,
            "pilot_tag": V2112_GATE_RELEASE_TAG,
            "seed": 2010922376,
            "preflight_p95_reservations": {},
            "contract_bootstrap_reservations": _bootstrap_reservations(
                model_id,
                target_run_spec_sha256=run_spec_sha256,
                source_run_spec_sha256=source_spec_sha256,
            ),
        }
    )
    journal = _journal(checkpoint)
    checkpoint["provider_call_journal_binding"].update(
        {
            "journal_sha256": journal["journal_sha256"],
            "run_id": checkpoint_run_id,
            "contract_hash": CONTRACT_SHA256,
            "path_name": f"{checkpoint_run_id}-provider-calls.json",
        }
    )
    envelope["provider_call_journal"] = journal
    envelope["provider_call_journal_artifact_sha256"] = canonical_sha256(journal)
    _rehash_checkpoint_and_exactness(envelope)
    return envelope


def _parent_debit() -> dict:
    return ParentBudgetDebit(
        parent_contract_sha256=gate_module.V211_PARENT_CONTRACT_SHA256,
        parent_run_ledger_sha256=gate_module.V211_PARENT_RUN_LEDGER_SHA256,
        parent_budget_ledger_sha256=gate_module.V211_PARENT_BUDGET_LEDGER_SHA256,
        stage_bucket="parent_v2111",
        cost_usd=gate_module.V211_PARENT_CUMULATIVE_COST_USD,
        hosted_completions=gate_module.V211_PARENT_CUMULATIVE_COMPLETIONS,
        storage_bytes=gate_module.V211_PARENT_CUMULATIVE_STORAGE_BYTES,
        record_sha256=gate_module.V211_PARENT_DEBIT_RECORD_SHA256,
    ).to_dict()


def _science_run_ids() -> dict[str, list[str]]:
    return {
        "gpt52_main": [f"v2112-gpt52-{index:03d}" for index in range(125)],
        "gpt56_diagnostic": [f"v2112-gpt56-{index:03d}" for index in range(6)],
    }


def _inputs() -> dict:
    return {
        "contract_sha256": CONTRACT_SHA256,
        "release_tag": V2112_GATE_RELEASE_TAG,
        "release_commit": RELEASE_COMMIT,
        "parent_import_receipt_binding": {
            "path": (
                "experiment_results/pilot-v2.11.2/raw/parent-import/"
                "parent_import_receipt.json"
            ),
            "file_sha256": "a" * 64,
            "content_sha256": "b" * 64,
        },
        "parent_budget_debit": _parent_debit(),
        "inherited_capability_bindings": {
            model_id: _capability_binding(model_id)
            for model_id in gate_module._MODEL_PROFILES
        },
        "fresh_preflight_artifacts": {
            model_id: _preflight_artifact(model_id)
            for model_id in gate_module._MODEL_PROFILES
        },
        "model_terminal_statuses": {
            "gpt52_main": "eligible",
            "gpt56_diagnostic": "eligible",
        },
        "current_attempt_pre_science_storage_bytes": 100_000_000,
        "ledger_event_chain_head": LEDGER_HEAD,
        "science_run_ids_by_model": _science_run_ids(),
        "source_manifest_hashes": {
            "file_sha256": MANIFEST_FILE_SHA256,
            "content_sha256": MANIFEST_CONTENT_SHA256,
        },
    }


def _rehash_receipt(receipt: dict) -> None:
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = canonical_sha256(receipt)


def test_gate_builds_exact_fresh_and_cumulative_denominators() -> None:
    receipt = build_v2112_post_gate_authority(**_inputs())
    verified = verify_v2112_gate_receipt(
        receipt,
        expected_contract_sha256=CONTRACT_SHA256,
        expected_git_commit=RELEASE_COMMIT,
    )

    denominator = verified["denominator"]
    assert denominator["inherited_capability_evidence_calls"] == 60
    assert denominator["fresh_preflight_calls"] == 64
    assert denominator["gate_evidence_calls"] == 124
    assert denominator["registered_remaining_science_calls"] == 5_816
    assert denominator["registered_fresh_full_matrix_calls"] == 5_880
    assert denominator["parent_debit_calls"] == 940
    assert denominator["cumulative_full_matrix_calls"] == 6_820
    assert denominator["registered_call_headroom"] == 680
    assert denominator["eligible_model_ids"] == [
        "gpt52_main",
        "gpt56_diagnostic",
    ]
    assert verified["projection"]["caps"] == {
        "cost_usd": 500.0,
        "hosted_completions": 7_500,
        "storage_bytes": 5_000_000_000,
    }
    assert (
        verified["projection"]["cumulative_full_matrix"][
            "remaining_hosted_completion_headroom"
        ]
        == 680
    )
    assert verified["provider_calls_during_authority"] == 0
    assert verified["go"] is True
    for by_kind in verified["observed_reservations"].values():
        assert by_kind["action"]["sample_count"] == 24
        assert by_kind["semantic"]["sample_count"] == 8


def test_historical_capability_usage_never_changes_dispatch_p95() -> None:
    baseline = build_v2112_post_gate_authority(**_inputs())
    inputs = _inputs()
    for binding in inputs["inherited_capability_bindings"].values():
        capability = binding["payload"]["capability"]
        for row in capability["usage_rows"]:
            row["usage"]["cost_usd"] = 0.20
        capability["actual_usage"] = _usage(
            prompt_tokens=3_000,
            completion_tokens=3_000,
            cost_usd=6.0,
        )
        _rehash_wrapper(binding)

    changed = build_v2112_post_gate_authority(**inputs)

    assert changed["observed_reservations"] == baseline["observed_reservations"]
    assert changed["dispatch_reservations"] == baseline["dispatch_reservations"]
    assert changed["projection"] == baseline["projection"]
    assert (
        changed["evidence_actuals"]["inherited_capability"]
        != baseline["evidence_actuals"]["inherited_capability"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("imported_preflight_samples", 1),
        ("imported_checkpoint_artifacts", ["failed-checkpoint"]),
        ("imported_p95_authorities", [{"historical": True}]),
    ],
)
def test_gate_rejects_rehashed_v2111_failed_evidence_import(
    field: str,
    value: object,
) -> None:
    inputs = _inputs()
    binding = inputs["inherited_capability_bindings"]["gpt52_main"]
    binding["payload"][field] = value
    _rehash_wrapper(binding)

    with pytest.raises(PilotV2112GateError, match="scope drifted"):
        build_v2112_post_gate_authority(**inputs)


def test_gate_rejects_rehashed_failed_journal_field_in_wrapper() -> None:
    inputs = _inputs()
    binding = inputs["inherited_capability_bindings"]["gpt52_main"]
    binding["payload"]["failed_preflight_journal"] = {"events": []}
    _rehash_wrapper(binding)

    with pytest.raises(PilotV2112GateError, match="fields or binding hash"):
        build_v2112_post_gate_authority(**inputs)


def test_gate_rejects_historical_bootstrap_even_after_checkpoint_rehash() -> None:
    inputs = _inputs()
    envelope = inputs["fresh_preflight_artifacts"]["gpt52_main"]
    runtime_model = gate_module._MODEL_PROFILES["gpt52_main"]["runtime_model"]
    envelope["checkpoint"]["run_config"]["contract_bootstrap_reservations"][
        runtime_model
    ]["action"]["authority"]["authority_id"] = V2111_CONTRACT_ENVELOPE_AUTHORITY_ID
    _rehash_checkpoint_and_exactness(envelope)

    with pytest.raises(PilotV2112GateError, match="bootstrap is invalid"):
        build_v2112_post_gate_authority(**inputs)


def test_gate_rejects_historical_exactness_and_projection_injection() -> None:
    inputs = _inputs()
    envelope = inputs["fresh_preflight_artifacts"]["gpt52_main"]
    envelope["exactness"][
        "schema_version"
    ] = "finevo-v2.11-long-context-preflight-exactness-receipt-v1"
    envelope["exactness"].pop("receipt_hash")
    envelope["exactness"]["receipt_hash"] = canonical_sha256(envelope["exactness"])
    envelope["exactness_artifact_sha256"] = canonical_sha256(envelope["exactness"])

    with pytest.raises(PilotV2112GateError, match="not fresh V2.11.2"):
        build_v2112_post_gate_authority(**inputs)

    inputs = _inputs()
    inputs["fresh_preflight_artifacts"]["gpt52_main"]["projection_p95"] = {}
    with pytest.raises(PilotV2112GateError, match="P95 injection"):
        build_v2112_post_gate_authority(**inputs)


def test_gate_rejects_rehashed_journal_checkpoint_mismatch() -> None:
    inputs = _inputs()
    envelope = inputs["fresh_preflight_artifacts"]["gpt52_main"]
    journal = envelope["provider_call_journal"]
    journal["events"][0]["payload"]["usage"]["cost_usd"] = 0.03
    previous = "0" * 64
    for index, event in enumerate(journal["events"]):
        event["event_index"] = index
        event["previous_event_sha256"] = previous
        event.pop("event_sha256", None)
        event["event_sha256"] = canonical_sha256(event)
        previous = event["event_sha256"]
    journal.pop("journal_sha256")
    journal["journal_sha256"] = canonical_sha256(journal)
    envelope["provider_call_journal_artifact_sha256"] = canonical_sha256(journal)
    envelope["checkpoint"]["provider_call_journal_binding"]["journal_sha256"] = journal[
        "journal_sha256"
    ]
    _rehash_checkpoint_and_exactness(envelope)

    with pytest.raises(PilotV2112GateError, match="journal/checkpoint"):
        build_v2112_post_gate_authority(**inputs)


def test_mixed_model_terminal_status_is_matrix_wide_no_go() -> None:
    inputs = _inputs()
    envelope = inputs["fresh_preflight_artifacts"]["gpt56_diagnostic"]
    envelope["checkpoint"]["provider_calls"][0]["finish_reason"] = "length"
    _rehash_checkpoint_and_exactness(envelope)
    inputs["model_terminal_statuses"]["gpt56_diagnostic"] = "interface-no-go"

    receipt = build_v2112_post_gate_authority(**inputs)
    verify_v2112_gate_receipt(receipt)

    assert receipt["denominator"]["locally_eligible_model_ids"] == ["gpt52_main"]
    assert receipt["denominator"]["eligible_model_ids"] == []
    assert receipt["dispatch_reservations"] == {}
    assert receipt["go"] is False
    assert receipt["reasons"] == ["mixed-model-terminal-status-global-no-go"]
    assert (
        sum(
            len(rows)
            for rows in receipt["denominator"][
                "no_go_science_run_ids_by_model"
            ].values()
        )
        == 131
    )


def test_full_matrix_projection_stops_above_cost_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate_module, "V2112_TOTAL_HARD_CAP_USD", 18.60)
    receipt = build_v2112_post_gate_authority(**_inputs())
    verify_v2112_gate_receipt(receipt)

    assert receipt["projection"]["go"] is False
    assert receipt["go"] is False
    assert receipt["denominator"]["eligible_model_ids"] == []
    assert receipt["dispatch_reservations"] == {}
    assert receipt["reasons"] == ["cumulative-cost-exceeds-hard-cap"]
    assert receipt["projection"]["cumulative_full_matrix"]["projected_cost_usd"] > 18.60


def test_receipt_verifier_rejects_rehashed_call_headroom_tamper() -> None:
    receipt = build_v2112_post_gate_authority(**_inputs())
    receipt["denominator"]["registered_call_headroom"] = 681
    _rehash_receipt(receipt)

    with pytest.raises(PilotV2112GateError, match="registered_call_headroom"):
        verify_v2112_gate_receipt(receipt)


def test_file_binding_exports_both_models_and_rejects_global_no_go(
    tmp_path: Path,
) -> None:
    receipt = build_v2112_post_gate_authority(**_inputs())
    relative = Path(
        "experiment_results/pilot-v2.11.2/raw/long-context-preflight/"
        "post_gate_authority.json"
    )
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    binding = verified_v2112_gate_authority_binding(
        relative.as_posix(),
        repo_root=tmp_path,
        expected_git_commit=RELEASE_COMMIT,
        expected_contract_sha256=CONTRACT_SHA256,
    )
    reservations = runner_reservations_from_v2112_gate_binding(binding)
    assert set(reservations) == {
        "openai/gpt-5.2-2025-12-11",
        "openai/gpt-5.6-sol",
    }

    no_go_inputs = _inputs()
    no_go_inputs["current_attempt_pre_science_storage_bytes"] = 5_000_000_000
    no_go = build_v2112_post_gate_authority(**no_go_inputs)
    target.write_text(
        json.dumps(no_go, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PilotV2112GateError, match="global no-go"):
        verified_v2112_gate_authority_binding(
            relative.as_posix(),
            repo_root=tmp_path,
            expected_git_commit=RELEASE_COMMIT,
            expected_contract_sha256=CONTRACT_SHA256,
        )
