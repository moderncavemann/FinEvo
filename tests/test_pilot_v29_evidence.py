from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import pytest

import run_pilot
from verified_memory import pilot_v24_evidence as evidence
from verified_memory import pilot_evidence as core_evidence
from verified_memory.pilot_contract import canonical_sha256, load_pilot_contract
from verified_memory.pilot_evidence import PilotEvidenceError
from verified_memory.pilot_v29_qref_projection import (
    EXACT_RETAINED_PATHS,
    IDENTITY_NORMALIZED_PATHS,
    QREF_RUN_SUMMARY_EQUIVALENCE_SCHEMA_VERSION,
    QREF_RUN_SUMMARY_PROJECTION_SCHEMA_VERSION,
    VALIDATED_VOLATILE_PATHS,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_9_overlay.yaml"
PREREQUISITE_STAGES = {
    "parent-import",
    "q-ref-resolution",
    "stage0-calibration",
}
PARENT_TERMINAL_STATUSES = {
    "complete": 1,
    "failed": 1,
    "integrity-stopped": 209,
}


def _qref_usage() -> dict[str, Any]:
    return {
        "prompt_tokens": 14_657,
        "completion_tokens": 1_248,
        "total_tokens": 15_905,
        "cost_usd": 0.0,
    }


def _qref_accounting() -> dict[str, Any]:
    usage = _qref_usage()
    return {
        "accounted_usage": usage,
        "active_calls": 0,
        "completed_calls": 48,
        "completion_usage_sum": usage,
        "effective_usage": usage,
        "reserved_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        },
        "rolled_back_calls": 0,
        "stop_reasons": ["call_limit"],
        "stopped": True,
    }


def _qref_provider_boundary() -> dict[str, Any]:
    return {
        "provider_model": "diagnostic/scripted-v1",
        "scripted_diagnostic_calls": 48,
        "hosted_provider_calls": 0,
        "hosted_cost_usd": 0.0,
        "completion_models": {"diagnostic/scripted-v1": 48},
        "call_kind_counts": {"action": 48},
    }


def _seal_summary_receipt(value: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(value)
    value.pop("integrity", None)
    value["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(value),
    }
    return value


def _seal_bound_receipt(value: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(value)
    value.pop("integrity", None)
    value["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
    }
    value["integrity"]["content_sha256"] = canonical_sha256(value)
    return value


def _qref_summary_receipt(run_id: str) -> dict[str, Any]:
    projection = "7" * 64
    provider = _qref_provider_boundary()
    accounting = _qref_accounting()
    return _seal_summary_receipt(
        {
            "schema_version": QREF_RUN_SUMMARY_EQUIVALENCE_SCHEMA_VERSION,
            "status": "pass",
            "policy": {
                "projection_schema_version": (
                    QREF_RUN_SUMMARY_PROJECTION_SCHEMA_VERSION
                ),
                "mode": "allowlist-first-fail-closed",
                "exact_retained_paths": list(EXACT_RETAINED_PATHS),
                "identity_normalized_paths": list(IDENTITY_NORMALIZED_PATHS),
                "validated_volatile_paths": list(VALIDATED_VOLATILE_PATHS),
                "unknown_paths": "reject",
                "completion_order": "exact",
                "raw_summary_hash_basis": ("full-unprojected-summary-canonical-json"),
            },
            "comparison": {
                "identity_relations_validated_before_projection": True,
                "timing_values_validated_before_omission": True,
                "deterministic_projection_exact": True,
                "provider_boundary_exact": True,
                "api_accounting_exact": True,
                "leaf_path_count": 1002,
                "normalized_leaf_path_count": 195,
            },
            "current": {
                "run_id": run_id,
                "budget_id": f"{run_id}-budget",
                "raw_summary_sha256": "1" * 64,
                "projection_sha256": projection,
                "provider_boundary": provider,
                "accounting": accounting,
            },
            "historical_reference": {
                "run_id": "q-ref-resolution-s2010922376",
                "budget_id": "q-ref-resolution-s2010922376-budget",
                "raw_summary_sha256": "2" * 64,
                "projection_sha256": projection,
                "provider_boundary": provider,
                "accounting": accounting,
            },
            "common_projection_sha256": projection,
            "raw_summaries_reused_as_projection": False,
        }
    )


def _qref_audit_receipt(
    *,
    contract,
    run_id: str,
    resolved_commit: str,
    source_manifest: str,
    summary_receipt: dict[str, Any],
) -> dict[str, Any]:
    historical = summary_receipt["historical_reference"]
    stream_hashes = {
        "actions_sha256": "4" * 64,
        "utility_ledger_sha256": "5" * 64,
        "shock_events_sha256": "6" * 64,
    }
    return _seal_bound_receipt(
        {
            "schema_version": ("finevo-pilot-v2.9-qref-audit-equivalence-v1"),
            "status": "pass",
            "comparison": {
                "fresh_config_run_id_matches_contract_cell": True,
                "historical_config_run_id_matches_reference": True,
                "config_equal_except_run_id": True,
                "actions_exact": True,
                "utility_ledger_exact": True,
                "shock_events_exact": True,
                "q_ref_exact": True,
                "row_count_exact": True,
                "run_contract_exact": True,
                "checks_exact": True,
                "ledger_hash_exact": True,
                "environment_hash_exact": True,
                "source_config_hash_identity_only_difference": True,
                "summary_projection_exact": True,
                "provider_accounting_exact": True,
            },
            "current": {
                "run_id": run_id,
                "budget_id": f"{run_id}-budget",
                "manifest": source_manifest,
                "manifest_file_sha256": "3" * 64,
                **stream_hashes,
                "q_ref": 63.50397933257746,
                "ledger_hash": "7" * 64,
                "source_config_hash": "8" * 64,
            },
            "historical_reference": {
                "run_id": historical["run_id"],
                "budget_id": historical["budget_id"],
                "source_run_root": (
                    "parent-import/v2_8_raw_snapshot/q-ref-resolution/runs/"
                    "historical"
                ),
                "source_manifest_file_sha256": "a" * 64,
                **stream_hashes,
                "summary_file_sha256": "b" * 64,
                "q_ref": 63.50397933257746,
                "ledger_hash": "7" * 64,
                "source_config_hash": "9" * 64,
                "source_result_reused": False,
            },
            "summary_equivalence": {
                "schema_version": summary_receipt["schema_version"],
                "content_sha256": summary_receipt["integrity"]["content_sha256"],
                "common_projection_sha256": summary_receipt["common_projection_sha256"],
                "embedded_key": "q_ref_summary_equivalence",
            },
            "provider_boundary": {
                "scripted_diagnostic_calls": 48,
                "hosted_provider_calls": 0,
                "hosted_cost_usd": 0.0,
                "hosted_provider_construction": False,
                "total_tokens": 15_905,
            },
            "bindings": {
                "contract_sha256": contract.canonical_hash,
                "git_tag": contract.implementation["required_git_tag"],
                "git_commit": resolved_commit,
                "source_manifest_file_sha256": "a" * 64,
                "source_manifest_content_sha256": "b" * 64,
                "parent_import_receipt_file_sha256": "c" * 64,
                "parent_import_receipt_content_sha256": "d" * 64,
            },
        }
    )


def _write_qref_resolution(
    path: Path,
    *,
    contract,
    spec,
    resolved_commit: str,
    receipt_mutator=None,
    audit_mutator=None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _qref_summary_receipt(spec.run_id)
    if receipt_mutator is not None:
        receipt_mutator(receipt)
        receipt = _seal_summary_receipt(receipt)
    source_manifest = str(path.parent / "runs/source/manifest.json")
    audit_receipt = _qref_audit_receipt(
        contract=contract,
        run_id=spec.run_id,
        resolved_commit=resolved_commit,
        source_manifest=source_manifest,
        summary_receipt=receipt,
    )
    if audit_mutator is not None:
        audit_mutator(audit_receipt)
        audit_receipt = _seal_bound_receipt(audit_receipt)
    resolution = {
        "schema_version": "finevo-q-ref-resolution-v1",
        "status": "pass",
        "q_ref": 63.50397933257746,
        "row_count": 48,
        "source_manifest": source_manifest,
        "scientific_evidence": False,
        "provider_calls_current_attempt": 48,
        "hosted_provider_calls_current_attempt": 0,
        "hosted_cost_usd_current_attempt": 0.0,
        "bindings": {
            "source_manifest_sha256": "3" * 64,
            "contract_sha256": contract.canonical_hash,
            "git_tag": contract.implementation["required_git_tag"],
            "git_commit": resolved_commit,
        },
        "q_ref_summary_equivalence": receipt,
        "q_ref_audit_equivalence": audit_receipt,
        "integrity": {
            "canonicalization": "json-sort-keys-utf8-v1",
        },
    }
    resolution["integrity"]["content_sha256"] = canonical_sha256(resolution)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(resolution, sort_keys=True),
        encoding="utf-8",
    )
    marker = {
        "q_ref": resolution["q_ref"],
        "row_count": resolution["row_count"],
        "source_manifest": resolution["source_manifest"],
        "source_manifest_sha256": resolution["bindings"]["source_manifest_sha256"],
        "resolution_artifact": str(path),
        "summary_equivalence": {
            "path": str(path),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "content_sha256": resolution.get("integrity", {}).get("content_sha256"),
            "embedded_key": "q_ref_summary_equivalence",
        },
    }
    payload = {
        "metrics": {},
        "gate_evidence": {"go": True},
        "q_ref_resolution": marker,
    }
    return resolution, payload


def _rows(
    contract,
    *,
    parent_like_no_go: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in contract.expand():
        if parent_like_no_go:
            if spec.stage_id == "parent-import":
                status = "complete"
            elif spec.stage_id == "q-ref-resolution":
                status = "failed"
            else:
                status = "integrity-stopped"
        else:
            status = "complete"
        prerequisite = spec.stage_id in PREREQUISITE_STAGES
        scientific_eligible = bool(
            status == "complete"
            and (spec.stage_id == "stage0-calibration" if prerequisite else True)
        )
        rows.append(
            {
                **spec.to_dict(),
                "status": status,
                "failure": (
                    None if status == "complete" else {"kind": "fixture-terminal-no-go"}
                ),
                "artifact_kind": ("terminal-summary" if status == "complete" else None),
                "artifact_sha256": ("a" * 64 if status == "complete" else None),
                "scientific_eligible": scientific_eligible,
                "metrics": {},
                "gate_evidence": {},
                "capability": {},
                "narrative": {},
            }
        )
    return rows


def _denominator(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    return {
        "expected_count": 211,
        "observed_ledger_count": 211,
        "all_rows_present": True,
        "all_rows_terminal": True,
        "status_counts": dict(sorted(counts.items())),
        "all_completed_artifacts_validated": True,
        "pass": True,
    }


def _release_controls() -> dict[str, Any]:
    return {
        "pass": True,
        "budget_ledger": {
            "pass": True,
            "raw_root_storage_bytes": 0,
            "checks": {"parent_debit_exact": True},
            "actual_totals": {
                "cost_usd": 3.212770875,
                "completions": 184,
                "storage_bytes": 32_158_175,
            },
            "actual_stage_cost_usd": {
                "parent_v23": 3.212770875,
            },
        },
    }


def _install_supported_gate_fixtures(
    monkeypatch: pytest.MonkeyPatch,
) -> list[set[str]]:
    observed_stage_sets: list[set[str]] = []

    def record(rows) -> None:
        observed_stage_sets.append({str(row["stage_id"]) for row in rows})

    def c_gate(_contract, rows, *, stage_id, model_id):
        record(rows)
        return {
            "status": "supported",
            "scientific_evidence_complete": True,
            "same_direction_counts": {"false_activation": 5},
            "claim_action": f"retain {model_id}/{stage_id}",
        }

    def a_gate(_contract, rows, *, stage_id, model_id):
        record(rows)
        return {
            "status": "supported",
            "scientific_evidence_complete": True,
            "primary_contrast": {
                "raw_paired_deltas": {
                    str(seed): 1.0 for seed in _contract.seeds["sets"]["main"]
                }
            },
            "threshold_gate": {"same_direction_count": 5},
            "claim_action": f"retain {model_id}/{stage_id}",
        }

    def d_gate(_contract, rows, *, stage_id, model_id, arms):
        record(rows)
        return {
            "status": "supported",
            "scientific_evidence_complete": True,
            "supported_treatments": ["no-memory"],
            "treatment_gates": {
                "no-memory": {
                    "six_step_discounted_utility_gate": {
                        "treatment_deltas": {
                            str(seed): 1.0 for seed in _contract.seeds["sets"]["main"]
                        }
                    }
                }
            },
            "claim_action": f"retain {model_id}/{stage_id}/{tuple(arms)!r}",
        }

    def b_summary(rows, *, stage_id, model_id, arms):
        record(rows)
        return {
            "comparison_type": "descriptive_preregistered_architecture_arms",
            "selection_rule": "do not select a winner",
            "arms": {arm: {} for arm in arms},
            "binding": f"{model_id}/{stage_id}",
        }

    monkeypatch.setattr(evidence, "_experiment_c_gate", c_gate)
    monkeypatch.setattr(evidence, "_experiment_a_gate", a_gate)
    monkeypatch.setattr(evidence, "_experiment_d_gate", d_gate)
    monkeypatch.setattr(evidence, "_experiment_b_summary", b_summary)
    return observed_stage_sets


def test_v29_contract_uses_lane_adapter_and_cumulative_v28_debit() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)

    assert core_evidence._stage_sets(contract) == (
        core_evidence.V24_NON_SCIENTIFIC_STAGES,
        core_evidence.V24_SCIENTIFIC_STAGES,
    )
    assert core_evidence._evidence_namespace(contract) == ("current_v2/pilot-v2.9")
    debit = core_evidence._expected_parent_budget_debit(contract)
    assert debit == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": (
            "948eac04516dd2c292d68beb732f97532b13e667a180e8c2db16fbb927f92f19"
        ),
        "parent_run_ledger_sha256": (
            "9b5f4bd1acdc5a525fb58b04b02ba29e31b05b594bfc411863e7baf3eb11f0d9"
        ),
        "parent_budget_ledger_sha256": (
            "07c936d61a7c38e6a7877ffaeeaf6c8ecb7fd4f495dbe8ed012a9a2861004b8f"
        ),
        "stage_bucket": "parent_v23",
        "cost_usd": 3.212770875,
        "hosted_completions": 184,
        "storage_bytes": 32_158_175,
        "record_sha256": (
            "0944138d9b47f7cf720681eb0ea8feda0b612a912992d78434c6bbda0d560fd0"
        ),
    }


def test_v29_parent_marker_adapts_default_verifier_signature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from verified_memory import pilot_v29_stage0_import

    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="parent-import")[0]
    resolved_commit = "b" * 40
    receipt_hash = "c" * 64
    receipt_path = tmp_path / "parent-import/parent_import_receipt.json"
    calls: list[dict[str, Any]] = []

    def verify(
        *,
        receipt_path,
        child_repo_root,
        contract,
        expected_git_commit,
    ):
        calls.append(
            {
                "receipt_path": receipt_path,
                "child_repo_root": child_repo_root,
                "contract_id": contract.contract_id,
                "expected_git_commit": expected_git_commit,
            }
        )
        return {"integrity": {"content_sha256": receipt_hash}}

    monkeypatch.setattr(
        pilot_v29_stage0_import,
        "verify_v29_parent_import_receipt",
        verify,
    )
    payload = {
        "metrics": {},
        "gate_evidence": {
            "receipt": str(receipt_path),
            "receipt_content_sha256": receipt_hash,
            "provider_calls_during_import": 0,
            "scientific_evidence": False,
        },
        "provider_calls": 0,
    }
    core_evidence._validate_terminal_payload_marker(
        contract,
        spec.to_dict(),
        payload,
        raw_root=tmp_path,
        resolved_git_commit=resolved_commit,
        source_repo_root=tmp_path,
    )

    assert calls == [
        {
            "receipt_path": str(receipt_path),
            "child_repo_root": tmp_path,
            "contract_id": "finevo-pilot-v2.9",
            "expected_git_commit": resolved_commit,
        }
    ]


def test_v29_qref_terminal_requires_full_summary_equivalence_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verified_memory import pilot_orchestrator

    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="q-ref-resolution")[0]
    resolved_commit = "b" * 40
    resolution, payload = _write_qref_resolution(
        tmp_path / "q-ref-resolution/q_ref_resolution.json",
        contract=contract,
        spec=spec,
        resolved_commit=resolved_commit,
    )
    monkeypatch.setattr(
        pilot_orchestrator,
        "verify_v29_qref_resolution",
        lambda *_args, **_kwargs: resolution,
    )

    core_evidence._validate_terminal_payload_marker(
        contract,
        spec.to_dict(),
        payload,
        raw_root=tmp_path,
        resolved_git_commit=resolved_commit,
    )


def test_v29_qref_terminal_rejects_missing_exact_runner_source(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="q-ref-resolution")[0]
    resolved_commit = "b" * 40
    _, payload = _write_qref_resolution(
        tmp_path / "q-ref-resolution/q_ref_resolution.json",
        contract=contract,
        spec=spec,
        resolved_commit=resolved_commit,
    )

    with pytest.raises(PilotEvidenceError, match="exact source replay failed"):
        core_evidence._validate_terminal_payload_marker(
            contract,
            spec.to_dict(),
            payload,
            raw_root=tmp_path,
            resolved_git_commit=resolved_commit,
        )


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda receipt: receipt["comparison"].__setitem__(
                "leaf_path_count",
                998,
            ),
            "1002/195 comparison",
        ),
        (
            lambda receipt: receipt["current"]["accounting"][
                "accounted_usage"
            ].__setitem__("total_tokens", 15_904),
            "48-call/15905-token",
        ),
        (
            lambda receipt: receipt["current"]["provider_boundary"].__setitem__(
                "hosted_provider_calls",
                1,
            ),
            "provider boundary",
        ),
    ],
)
def test_v29_qref_summary_equivalence_tamper_fails_closed(
    tmp_path: Path,
    mutator,
    match: str,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="q-ref-resolution")[0]
    resolved_commit = "b" * 40
    _, payload = _write_qref_resolution(
        tmp_path / "q-ref-resolution/q_ref_resolution.json",
        contract=contract,
        spec=spec,
        resolved_commit=resolved_commit,
        receipt_mutator=mutator,
    )

    with pytest.raises(PilotEvidenceError, match=match):
        core_evidence._validate_terminal_payload_marker(
            contract,
            spec.to_dict(),
            payload,
            raw_root=tmp_path,
            resolved_git_commit=resolved_commit,
        )


def test_v29_qref_missing_or_rebound_receipt_fails_closed(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="q-ref-resolution")[0]
    resolved_commit = "b" * 40
    _, payload = _write_qref_resolution(
        tmp_path / "q-ref-resolution/q_ref_resolution.json",
        contract=contract,
        spec=spec,
        resolved_commit=resolved_commit,
    )
    missing = deepcopy(payload)
    missing["q_ref_resolution"].pop("summary_equivalence")
    with pytest.raises(PilotEvidenceError, match="must be an object"):
        core_evidence._validate_terminal_payload_marker(
            contract,
            spec.to_dict(),
            missing,
            raw_root=tmp_path,
            resolved_git_commit=resolved_commit,
        )

    rebound = deepcopy(payload)
    rebound["q_ref_resolution"]["summary_equivalence"]["content_sha256"] = "f" * 64
    with pytest.raises(PilotEvidenceError, match="terminal summary-equivalence"):
        core_evidence._validate_terminal_payload_marker(
            contract,
            spec.to_dict(),
            rebound,
            raw_root=tmp_path,
            resolved_git_commit=resolved_commit,
        )


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda audit: audit["comparison"].__setitem__(
                "actions_exact",
                False,
            ),
            "exact actions/utility/shocks/q_ref checks",
        ),
        (
            lambda audit: audit["historical_reference"].__setitem__(
                "utility_ledger_sha256",
                "e" * 64,
            ),
            "exact stream/scalar hash binding",
        ),
        (
            lambda audit: audit["historical_reference"].__setitem__(
                "q_ref",
                1.0,
            ),
            "exact stream/scalar hash binding",
        ),
        (
            lambda audit: audit["provider_boundary"].__setitem__(
                "total_tokens",
                15_904,
            ),
            "audit provider/accounting boundary",
        ),
    ],
)
def test_v29_qref_exact_audit_receipt_tamper_fails_closed(
    tmp_path: Path,
    mutator,
    match: str,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="q-ref-resolution")[0]
    resolved_commit = "b" * 40
    _, payload = _write_qref_resolution(
        tmp_path / "q-ref-resolution/q_ref_resolution.json",
        contract=contract,
        spec=spec,
        resolved_commit=resolved_commit,
        audit_mutator=mutator,
    )

    with pytest.raises(PilotEvidenceError, match=match):
        core_evidence._validate_terminal_payload_marker(
            contract,
            spec.to_dict(),
            payload,
            raw_root=tmp_path,
            resolved_git_commit=resolved_commit,
        )


def test_v29_complete_fixture_uses_only_fresh_a_d_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract)
    observed_stage_sets = _install_supported_gate_fixtures(monkeypatch)

    aggregate = evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator=_denominator(rows),
        release_controls=_release_controls(),
    )

    assert aggregate["schema_version"] == ("finevo-pilot-v2.9-evidence-package-v1")
    assert aggregate["evidence_namespace"] == "current_v2/pilot-v2.9"
    assert aggregate["publication_status"] == "complete"
    assert aggregate["scientific_complete"] is True
    assert aggregate["itt_row_preservation"] == {
        "registered_rows": 211,
        "retained_rows": 211,
        "failed_or_stopped_rows": 0,
        "status_counts": {"complete": 211},
        "all_registered_rows_retained": True,
        "failures_retained": True,
    }
    prerequisites = aggregate["prerequisites"]
    assert prerequisites["registered_cells"] == 16
    assert prerequisites["all_prerequisites_complete"] is True
    assert prerequisites["stage0_imported_cells"] == 14
    assert prerequisites["q_ref_provider_accounting"] == {
        "hosted_provider_calls": 0,
        "hosted_cost_usd": 0.0,
        "scripted_diagnostic_calls": 48,
    }
    assert (
        prerequisites["stages"]["q-ref-resolution"]["origin"]
        == "fresh-v2.9-scripted-diagnostic"
    )
    assert aggregate["effect_aggregation_scope"]["v2_9_a_d_cells_only"] is True
    assert (
        set(aggregate["effect_aggregation_scope"]["prerequisite_stage_ids_excluded"])
        == PREREQUISITE_STAGES
    )
    assert observed_stage_sets
    assert all(
        not (stage_ids & PREREQUISITE_STAGES) for stage_ids in observed_stage_sets
    )
    lineage = aggregate["parent_evidence_lineage"]
    assert lineage["source_contract_id"] == "finevo-pilot-v2.8"
    assert lineage["parent_status_counts"] == PARENT_TERMINAL_STATUSES
    assert lineage["parent_rows_imported_into_v2_9_effect_aggregate"] == 0
    budget = aggregate["inherited_budget_boundary"]
    assert budget["source_contract_id"] == "finevo-pilot-v2.8"
    assert budget["total_cap_usd"] == 500.0
    assert budget["expected_cumulative_prior"]["cost_usd"] == 3.212770875
    assert budget["v2_8_incremental"] == {
        "cost_usd": 0.0,
        "hosted_completions": 0,
        "scripted_diagnostic_calls": 48,
    }


def test_v29_terminal_parent_like_denominator_is_complete_with_no_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract, parent_like_no_go=True)
    _install_supported_gate_fixtures(monkeypatch)

    aggregate = evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator=_denominator(rows),
        release_controls=_release_controls(),
    )

    assert aggregate["denominator"]["status_counts"] == PARENT_TERMINAL_STATUSES
    assert aggregate["itt_row_preservation"]["retained_rows"] == 211
    assert aggregate["itt_row_preservation"]["failed_or_stopped_rows"] == 210
    assert aggregate["prerequisites"]["all_prerequisites_complete"] is False
    assert aggregate["scientific_matrix_complete"] is False
    assert aggregate["scientific_complete"] is False
    assert aggregate["publication_status"] == "complete-with-no-go"
    evidence._require_publishable_terminal_denominator(aggregate)


def test_v29_parent_reference_revalidates_without_rewriting_v28() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    parent_root = ROOT / "evidence" / "current_v2" / "pilot-v2.8"
    before = {
        path.relative_to(parent_root)
        .as_posix(): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in parent_root.rglob("*")
        if path.is_file()
    }

    reference = evidence._validated_v29_parent_evidence_reference(
        contract,
        contract_path=CONTRACT_PATH,
    )

    after = {
        path.relative_to(parent_root)
        .as_posix(): hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in parent_root.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert reference is not None
    assert reference["reference_kind"] == "immutable-external-package-reference"
    assert reference["source_package_copied"] is False
    assert reference["source_package_path"] == ("evidence/current_v2/pilot-v2.8")
    assert reference["inventory_verified"] is True
    assert reference["semantic_binding_verified"] is True
    assert reference["parent_status_counts"] == PARENT_TERMINAL_STATUSES


def test_v29_parent_reference_checksum_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    repository = tmp_path / "repo"
    experiments = repository / "experiments"
    experiments.mkdir(parents=True)
    fake_contract_path = experiments / CONTRACT_PATH.name
    shutil.copyfile(CONTRACT_PATH, fake_contract_path)
    parent_copy = repository / "evidence" / "current_v2" / "pilot-v2.8"
    shutil.copytree(
        ROOT / "evidence" / "current_v2" / "pilot-v2.8",
        parent_copy,
    )
    aggregate_path = parent_copy / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["publication_status"] = "complete"
    aggregate_path.write_text(
        json.dumps(aggregate, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(
        PilotEvidenceError,
        match="checksum verification failed",
    ):
        evidence._validated_v29_parent_evidence_reference(
            contract,
            contract_path=fake_contract_path,
        )


def test_run_pilot_routes_v29_publish_to_lane_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    package = SimpleNamespace(
        package_dir=tmp_path / "current_v2/pilot-v2.9",
        manifest_path=tmp_path / "current_v2/pilot-v2.9/package_manifest.json",
        checksums_path=tmp_path / "current_v2/pilot-v2.9/checksums.json",
        contract_hash="b" * 64,
        scientific_complete=False,
        claim_gates={"lanes": {}},
    )

    def build(**kwargs):
        calls.append(kwargs)
        return package

    monkeypatch.setattr(run_pilot, "build_pilot_v24_evidence_package", build)
    args = Namespace(
        contract=CONTRACT_PATH,
        stage="publish-evidence",
        resume=False,
        development_fake=False,
        raw_root=tmp_path / "raw",
        evidence_root=tmp_path,
        parent_repo_root=None,
        source_repo_root=None,
    )
    result = run_pilot.execute(args)

    assert len(calls) == 1
    assert calls[0]["contract_path"] == CONTRACT_PATH
    assert result["status"] == "complete-with-no-go"
    assert result["provider_calls"] == 0
    assert result["scientific_complete"] is False


def test_v29_prerequisite_and_budget_tamper_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract)
    _install_supported_gate_fixtures(monkeypatch)
    bad_rows = deepcopy(rows)
    qref = next(row for row in bad_rows if row["stage_id"] == "q-ref-resolution")
    qref["scientific_eligible"] = True
    with pytest.raises(PilotEvidenceError, match="eligibility boundary"):
        evidence.aggregate_v24_evidence(
            contract,
            bad_rows,
            denominator=_denominator(rows),
            release_controls=_release_controls(),
        )

    bad_release = _release_controls()
    bad_release["budget_ledger"]["checks"]["parent_debit_exact"] = False
    with pytest.raises(PilotEvidenceError, match="inherited debit/denominator"):
        evidence.aggregate_v24_evidence(
            contract,
            rows,
            denominator=_denominator(rows),
            release_controls=bad_release,
        )
