from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import tests.test_pilot_v211_gate as v211_fixtures
from verified_memory.pilot_budget import ParentBudgetDebit
import verified_memory.pilot_v2111_gate as gate_module
from verified_memory.pilot_v2111_gate import (
    PilotV2111GateError,
    build_v2111_post_gate_authority,
    canonical_sha256,
    runner_reservations_from_v2111_gate_binding,
    verified_v2111_gate_authority_binding,
    verify_v2111_gate_receipt,
)


CONTRACT_SHA256 = "1" * 64
RELEASE_COMMIT = "2" * 40
LEDGER_HEAD = "3" * 64
MANIFEST_FILE_SHA256 = "4" * 64
MANIFEST_CONTENT_SHA256 = "5" * 64


@pytest.fixture(autouse=True)
def _frozen_manifest_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gate_module,
        "V2111_SOURCE_MANIFEST_FILE_SHA256",
        MANIFEST_FILE_SHA256,
    )
    monkeypatch.setattr(
        gate_module,
        "V2111_SOURCE_MANIFEST_CONTENT_SHA256",
        MANIFEST_CONTENT_SHA256,
    )
    monkeypatch.setattr(
        v211_fixtures,
        "CONTRACT_SHA256",
        CONTRACT_SHA256,
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


def _capability_binding(model_id: str) -> dict:
    profile = gate_module._MODEL_PROFILES[model_id]
    samples = {"action": [], "semantic": []}
    usage_rows = []
    for call_kind, count in (("action", 24), ("semantic", 6)):
        for _ in range(count):
            usage = _usage()
            samples[call_kind].append(
                {
                    "finish_reason": "stop",
                    "response_completed": True,
                    "output_disposition": "accepted",
                    "error_type": None,
                    "parse_success": True,
                    "clipped": False,
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "reasoning_tokens": 60,
                    "visible_completion_tokens": 40,
                }
            )
            usage_rows.append(
                {
                    "response_model": profile["served_model"],
                    "call_kind": call_kind,
                    "usage": usage,
                }
            )
    actual = _usage(
        prompt_tokens=3_000,
        completion_tokens=3_000,
        cost_usd=0.30,
    )
    source_manifest = {
        "path": "experiments/pilot_v2_11_1_source_manifest.json",
        "file_sha256": MANIFEST_FILE_SHA256,
        "content_sha256": MANIFEST_CONTENT_SHA256,
    }
    wrapper = _seal(
        {
            "schema_version": (
                "finevo-pilot-v2.11.1-imported-capability-wrapper-v1"
            ),
            "child_release": {
                "contract_id": "finevo-pilot-v2.11.1",
                "contract_sha256": CONTRACT_SHA256,
                "git_tag": "pilot-v2.11.1-science",
                "resolved_git_commit": RELEASE_COMMIT,
            },
            "parent_release": {
                "contract_id": gate_module.V211_CONTRACT_ID,
                "contract_sha256": gate_module.V211_CONTRACT_SHA256,
                "git_tag": gate_module.V211_SCIENCE_TAG,
                "git_tag_object": gate_module.V211_SCIENCE_TAG_OBJECT,
                "resolved_git_commit": gate_module.V211_SCIENCE_COMMIT,
            },
            "source_manifest": source_manifest,
            "source_artifacts": {
                "historical_source_calls": 30,
                "action_sample_count": 24,
                "semantic_sample_count": 6,
                "runtime_model": profile["runtime_model"],
                "actual_usage": actual,
                "capability_pass": True,
                "interface_pass": True,
                "scientific_evidence": False,
            },
            "capability": {
                "model_id": model_id,
                "run_id": f"v211-capability-{model_id}",
                "runtime_model": profile["runtime_model"],
                "requested_model": profile["requested_model"],
                "served_model": profile["served_model"],
                "historical_source_calls": 30,
                "action_sample_count": 24,
                "semantic_sample_count": 6,
                "actual_usage": actual,
                "samples": samples,
                "usage_rows": usage_rows,
                "capability_pass": True,
                "interface_pass": True,
            },
            "provider_construction_current_attempt": False,
            "provider_calls_current_attempt": 0,
            "hosted_provider_calls_current_attempt": 0,
            "current_attempt_usage": _usage(
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
            ),
            "imported_effect_cells": 0,
            "imported_p95_authorities": [],
            "scientific_evidence": False,
            "evidence_scope": "preregistered_task_capability_gate",
        }
    )
    return {
        "model_id": model_id,
        "wrapper_content_sha256": wrapper["integrity"]["content_sha256"],
        "payload": wrapper,
        "provider_construction_during_verification": False,
        "provider_calls_during_verification": 0,
    }


def _parent_debit(*, cost_delta: float = 0.0) -> dict:
    return ParentBudgetDebit(
        parent_contract_sha256=gate_module.V211_CONTRACT_SHA256,
        parent_run_ledger_sha256=gate_module.V211_RUN_LEDGER_SHA256,
        parent_budget_ledger_sha256=gate_module.V211_BUDGET_LEDGER_SHA256,
        stage_bucket="parent_v211",
        cost_usd=gate_module.V211_CUMULATIVE_COST_USD + cost_delta,
        hosted_completions=gate_module.V211_CUMULATIVE_COMPLETIONS,
        storage_bytes=gate_module.V211_CUMULATIVE_STORAGE_BYTES,
        record_sha256=(
            gate_module.V211_PARENT_DEBIT_RECORD_SHA256
            if cost_delta == 0.0
            else None
        ),
    ).to_dict()


def _science_run_ids() -> dict[str, list[str]]:
    return {
        "gpt52_main": [
            f"v2111-gpt52-science-{index:03d}" for index in range(125)
        ],
        "gpt56_diagnostic": [
            f"v2111-gpt56-science-{index:03d}" for index in range(6)
        ],
    }


def _inputs() -> dict:
    return {
        "contract_sha256": CONTRACT_SHA256,
        "release_tag": "pilot-v2.11.1-science",
        "release_commit": RELEASE_COMMIT,
        "parent_import_receipt_binding": {
            "path": (
                "experiment_results/pilot-v2.11.1/raw/parent-import/"
                "parent_import_receipt.json"
            ),
            "file_sha256": "6" * 64,
            "content_sha256": "7" * 64,
        },
        "parent_budget_debit": _parent_debit(),
        "inherited_capability_bindings": {
            model_id: _capability_binding(model_id)
            for model_id in gate_module._MODEL_PROFILES
        },
        "fresh_preflight_artifacts": {
            model_id: v211_fixtures._preflight_artifact(model_id)
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


def _rehash_wrapper(binding: dict) -> None:
    wrapper = binding["payload"]
    wrapper["integrity"].pop("content_sha256", None)
    wrapper["integrity"]["content_sha256"] = canonical_sha256(wrapper)
    binding["wrapper_content_sha256"] = wrapper["integrity"][
        "content_sha256"
    ]


def _rehash_receipt(receipt: dict) -> None:
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = canonical_sha256(receipt)


def test_gate_separates_124_evidence_calls_from_64_fresh_budget_calls() -> None:
    receipt = build_v2111_post_gate_authority(**_inputs())
    verified = verify_v2111_gate_receipt(
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
    assert denominator["parent_debit_calls"] == 876
    assert denominator["cumulative_full_matrix_calls"] == 6_756
    assert verified["projection"]["parent_debit"]["hosted_completions"] == 876
    assert (
        verified["projection"]["fresh_preflight"]["hosted_completions"]
        == 64
    )
    assert (
        verified["evidence_actuals"]["combined_gate_evidence"][
            "hosted_completions"
        ]
        == 124
    )
    assert verified["go"] is True


def test_gate_uses_combined_observed_p95_and_never_bootstrap_envelope() -> None:
    receipt = build_v2111_post_gate_authority(**_inputs())

    for by_kind in receipt["observed_reservations"].values():
        assert by_kind["action"]["sample_count"] == 48
        assert by_kind["semantic"]["sample_count"] == 14
        assert by_kind["action"]["reserve_multiplier"] == 1.25
        assert by_kind["semantic"]["reserve_multiplier"] == 1.25
    assert receipt["bootstrap_envelope_used_as_observed_p95"] is False
    assert receipt["provider_construction_during_authority"] is False
    assert receipt["provider_calls_during_authority"] == 0
    assert receipt["scientific_evidence"] is False


def test_gate_rejects_rehashed_capability_count_tamper() -> None:
    inputs = _inputs()
    binding = inputs["inherited_capability_bindings"]["gpt52_main"]
    binding["payload"]["capability"]["usage_rows"].pop()
    _rehash_wrapper(binding)

    with pytest.raises(
        PilotV2111GateError,
        match="usage denominator",
    ):
        build_v2111_post_gate_authority(**inputs)


def test_gate_rejects_capability_wrapper_content_tamper() -> None:
    inputs = _inputs()
    binding = inputs["inherited_capability_bindings"]["gpt52_main"]
    binding["payload"]["capability"]["samples"]["action"][0][
        "finish_reason"
    ] = "length"

    with pytest.raises(PilotV2111GateError, match="self-hash"):
        build_v2111_post_gate_authority(**inputs)


def test_gate_rejects_self_consistent_parent_budget_substitution() -> None:
    inputs = _inputs()
    inputs["parent_budget_debit"] = _parent_debit(cost_delta=1.0)

    with pytest.raises(
        PilotV2111GateError,
        match="frozen V2.11 at cost_usd",
    ):
        build_v2111_post_gate_authority(**inputs)


def test_full_matrix_projection_stops_before_dispatch_above_500_dollars() -> None:
    inputs = _inputs()
    for binding in inputs["inherited_capability_bindings"].values():
        wrapper = binding["payload"]
        for row in wrapper["capability"]["usage_rows"]:
            row["usage"]["cost_usd"] = 0.20
        expensive_actual = _usage(
            prompt_tokens=3_000,
            completion_tokens=3_000,
            cost_usd=6.0,
        )
        wrapper["capability"]["actual_usage"] = expensive_actual
        wrapper["source_artifacts"]["actual_usage"] = expensive_actual
        _rehash_wrapper(binding)

    receipt = build_v2111_post_gate_authority(**inputs)

    assert receipt["projection"]["go"] is False
    assert receipt["go"] is False
    assert receipt["reasons"] == ["cumulative-cost-exceeds-hard-cap"]
    assert (
        receipt["projection"]["cumulative_full_matrix"][
            "projected_cost_usd"
        ]
        > 500.0
    )


def test_receipt_verifier_rejects_rehashed_source_manifest_substitution() -> None:
    receipt = build_v2111_post_gate_authority(**_inputs())
    receipt["bindings"]["source_manifest"]["file_sha256"] = "8" * 64
    _rehash_receipt(receipt)

    with pytest.raises(
        PilotV2111GateError,
        match="source manifest binding",
    ):
        verify_v2111_gate_receipt(receipt)


def test_file_binding_is_strict_runner_authority_and_rejects_symlink(
    tmp_path: Path,
) -> None:
    receipt = build_v2111_post_gate_authority(**_inputs())
    relative = Path(
        "experiment_results/pilot-v2.11.1/raw/long-context-preflight/"
        "post_gate_authority.json"
    )
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    binding = verified_v2111_gate_authority_binding(
        relative.as_posix(),
        repo_root=tmp_path,
        expected_git_commit=RELEASE_COMMIT,
        expected_contract_sha256=CONTRACT_SHA256,
    )
    reservations = runner_reservations_from_v2111_gate_binding(binding)

    assert set(reservations) == {
        "openai/gpt-5.2-2025-12-11",
        "openai/gpt-5.6-sol",
    }
    for by_kind in reservations.values():
        assert set(by_kind) == {"action", "semantic"}
        for row in by_kind.values():
            assert row["authority"]["pilot_tag"] == (
                "pilot-v2.11.1-science"
            )
            assert row["authority"]["source_kind"] == (
                "sealed-closed-loop-observed-p95"
            )

    target.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    target.symlink_to(outside)
    with pytest.raises(PilotV2111GateError, match="non-symlink"):
        verified_v2111_gate_authority_binding(
            relative.as_posix(),
            repo_root=tmp_path,
            expected_git_commit=RELEASE_COMMIT,
            expected_contract_sha256=CONTRACT_SHA256,
        )


def test_zero_provider_flags_cannot_be_laundered_by_rehash() -> None:
    receipt = build_v2111_post_gate_authority(**_inputs())
    receipt["provider_calls_during_authority"] = 1
    _rehash_receipt(receipt)

    with pytest.raises(PilotV2111GateError, match="zero-provider"):
        verify_v2111_gate_receipt(receipt)
