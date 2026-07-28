from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verified_memory import pilot_v2101_parent_import as parent_import
from verified_memory import pilot_v24_evidence as evidence
from verified_memory.failure_artifacts import write_failure_receipt
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_evidence import PilotEvidenceError


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_10_1.yaml"
PREREQUISITE_STAGES = {
    "parent-import",
    "q-ref-resolution",
    "stage0-calibration",
}


def _rows(contract) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in contract.expand():
        prerequisite = spec.stage_id in PREREQUISITE_STAGES
        status = "complete" if prerequisite else "integrity-stopped"
        rows.append(
            {
                **spec.to_dict(),
                "status": status,
                "failure": (
                    None
                    if prerequisite
                    else {"kind": "fixture-terminal-integrity-stop"}
                ),
                "artifact_kind": ("terminal-summary" if prerequisite else None),
                "artifact_sha256": "a" * 64 if prerequisite else None,
                "scientific_eligible": bool(
                    prerequisite and spec.stage_id == "stage0-calibration"
                ),
                "metrics": {},
                "gate_evidence": {},
                "capability": {},
                "narrative": {},
            }
        )
    return rows


def _early_qref_failure_rows(contract) -> list[dict[str, Any]]:
    rows = _rows(contract)
    for row in rows:
        if row["stage_id"] == "parent-import":
            continue
        row.update(
            {
                "status": "integrity-stopped",
                "failure": {"kind": "fixture-current-qref-failure"},
                "artifact_kind": None,
                "artifact_sha256": None,
                "scientific_eligible": False,
            }
        )
    return rows


def _terminal_implementation_failure_rows(contract) -> list[dict[str, Any]]:
    rows = _rows(contract)
    failure = {
        "error_type": "ValueError",
        "message": (
            "source-backed observed p95 receipt verification failed: "
            "observed-p95 receipt top-level shape or schema drifted"
        ),
        "message_bytes": 110,
        "message_sha256": (
            "39cb7f19f94e435d9eb4873df49beac2507703522f2ad9ffa7f688a5f6b92ef7"
        ),
        "message_truncated": False,
    }
    offline_reliability = {
        "unsupported_candidate_rejected": True,
        "false_rule_ever_active": False,
        "unverified_false_rule_ever_active": True,
        "same_candidate_content": True,
        "provider_calls": 0,
    }
    for row in rows:
        if row["stage_id"] in PREREQUISITE_STAGES:
            continue
        is_offline_candidate = (
            row["stage_id"] in {"local-experiment-c", "experiment-c"}
            and row["arm_id"] == "verified-error-candidate"
        )
        if is_offline_candidate:
            row.update(
                {
                    "status": "complete",
                    "failure": None,
                    "artifact_kind": "terminal-summary",
                    "artifact_sha256": "b" * 64,
                    "scientific_eligible": True,
                    "metrics": {
                        "rule_reliability": deepcopy(offline_reliability)
                    },
                    "gate_evidence": deepcopy(offline_reliability),
                }
            )
        else:
            row.update(
                {
                    "status": "failed",
                    "failure": deepcopy(failure),
                    "artifact_kind": None,
                    "artifact_sha256": None,
                    "scientific_eligible": False,
                    "metrics": {},
                    "gate_evidence": {},
                    "capability": {},
                    "narrative": {},
                }
            )
    return rows


def _zero_failure_budget_snapshot(run_id: str) -> dict[str, Any]:
    zero_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }
    return {
        "budget_id": f"{run_id}-budget",
        "limits": {
            "max_calls": 16,
            "max_prompt_tokens": None,
            "max_completion_tokens": None,
            "max_total_tokens": None,
            "max_cost_usd": 1.0,
            "max_elapsed_seconds": None,
        },
        "accounted_usage": deepcopy(zero_usage),
        "reserved_usage": deepcopy(zero_usage),
        "effective_usage": deepcopy(zero_usage),
        "completed_calls": 0,
        "active_calls": 0,
        "rolled_back_calls": 0,
        "elapsed_seconds": 0.0,
        "stopped": False,
        "stop_reasons": [],
        "active_reservations": [],
        "completions": [],
    }


def _build_failure_receipt_fixture(
    tmp_path: Path,
    contract,
    rows: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    release_attestation = {
        "schema_version": "synthetic-v2.10.1-release-attestation-v1",
        "status": "pass",
        "attestation_sha256": "a" * 64,
    }
    (raw_root / "release_attestation.json").write_text(
        json.dumps(
            release_attestation,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    expected_specs = {spec.run_id: spec.to_dict() for spec in contract.expand()}
    failed = [row for row in rows if row["status"] == "failed"]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in failed:
        if str(row["stage_id"]).endswith("experiment-d"):
            key = (
                row["stage_id"],
                row["model_id"],
                int(row["environment_seed"]),
            )
        else:
            key = (row["run_id"],)
        grouped.setdefault(key, []).append(row)

    paid = {
        "git_tag": "pilot-v2.10.1-science",
        "head_commit": "b5bfa9b86d3cdb706cea5be707597bef8ac85aed",
        "tag_commit": "b5bfa9b86d3cdb706cea5be707597bef8ac85aed",
        "tag_object_type": "tag",
        "worktree_clean": True,
        "contract_binding": contract.validate_provenance(
            "b5bfa9b86d3cdb706cea5be707597bef8ac85aed",
            "pilot-v2.10.1-science",
        ),
        "release_attestation": deepcopy(release_attestation),
    }
    ledger_runs: dict[str, Any] = {}
    for group_rows in grouped.values():
        group_rows = sorted(group_rows, key=lambda row: str(row["run_id"]))
        first = group_rows[0]
        stage_id = str(first["stage_id"])
        run_ids = [str(row["run_id"]) for row in group_rows]
        if stage_id.endswith("experiment-d"):
            receipt_dir = (
                raw_root
                / stage_id
                / "checkpoints"
                / f"s{int(first['environment_seed'])}"
                / "failure_receipt"
            )
            scope = f"finevo-pilot/{stage_id}/shared-checkpoint-group"
            projection_run_id = (
                f"{contract.contract_id}--{stage_id}--"
                f"{first['model_id']}--checkpoint-group--"
                f"s{int(first['environment_seed'])}"
            )
        else:
            receipt_dir = (
                raw_root
                / stage_id
                / "runs"
                / run_ids[0]
                / "failure_receipt"
            )
            scope = (
                f"finevo-pilot/{stage_id}/"
                f"{expected_specs[run_ids[0]]['execution_mode']}"
            )
            projection_run_id = run_ids[0]
        model_ids = sorted({str(row["model_id"]) for row in group_rows})
        manifest = write_failure_receipt(
            receipt_dir,
            scope=scope,
            error=ValueError(
                "source-backed observed p95 receipt verification failed: "
                "observed-p95 receipt top-level shape or schema drifted"
            ),
            budget_snapshot=_zero_failure_budget_snapshot(projection_run_id),
            config={
                "schema_version": "finevo-pilot-failure-config-v1",
                "contract_id": contract.contract_id,
                "contract_sha256": contract.canonical_hash,
                "projection": {
                    "run_id": projection_run_id,
                    "stage_bucket": first["budget_bucket"],
                    "cost_usd": 0.0,
                    "completions": 0,
                    "storage_bytes": 1_000_000,
                    "basis": {"method": "synthetic-pre-provider-fixture"},
                },
                "run_specs": [expected_specs[run_id] for run_id in run_ids],
                "provider_request_profiles": {
                    model_id: contract.provider_profiles[model_id].to_dict()
                    for model_id in model_ids
                },
                "provider_call_journals": [],
            },
            provenance={
                "contract_id": contract.contract_id,
                "contract_sha256": contract.canonical_hash,
                "paid_provenance": deepcopy(paid),
                "diagnostic_only": False,
                "scientific_evidence": False,
                "evidence_use": "failure denominator and audit only",
            },
            git_commit="b5bfa9b86d3cdb706cea5be707597bef8ac85aed",
            git_dirty=False,
        )
        relative = manifest.relative_to(raw_root).as_posix()
        for row in group_rows:
            run_id = str(row["run_id"])
            ledger_runs[run_id] = {
                "spec": expected_specs[run_id],
                "status": "failed",
                "artifact": relative,
                "failure": deepcopy(row["failure"]),
            }
    return raw_root, {"runs": ledger_runs}


def _fixture_release_controls(raw_root: Path) -> dict[str, Any]:
    return {
        "release_attestation": {
            "pass": True,
            "path": str(raw_root / "release_attestation.json"),
            "checks": {
                "schema_and_hash": True,
                "commit_bound": True,
                "exact_linux_macos_ci": True,
            },
            "reasons": [],
        }
    }


def _rewrite_failure_receipt(
    manifest_path: Path,
    mutate,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt_path = manifest_path.parent / "failure.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutate(receipt)
    receipt_bytes = (
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    receipt_path.write_bytes(receipt_bytes)
    manifest["failure_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    manifest["failure_size_bytes"] = len(receipt_bytes)
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _denominator(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "expected_count": 211,
        "observed_ledger_count": 211,
        "all_rows_present": True,
        "all_rows_terminal": True,
        "status_counts": dict(sorted(statuses.items())),
        "all_completed_artifacts_validated": True,
        "pass": True,
    }


def _release_controls() -> dict[str, Any]:
    sensitivities = {}
    for lane_id in evidence._V210_C_SENSITIVITY_FILES:
        sensitivities[lane_id] = {
            **evidence._v210_sensitivity_lane_definition(lane_id),
            "pass": False,
            "available": False,
            "provider_calls": 0,
            "descriptive_only": True,
            "effectiveness_gate": False,
            "reason": (
                f"{evidence._v210_sensitivity_lane_definition(lane_id)['stage_id']} "
                "ITT cells are not all complete and scientifically eligible"
            ),
        }
    return {
        # This is the exact post-sensitivity build shape: all base release
        # controls pass, while the historical V2.10-family top-level flag is
        # false because both C sensitivity artifacts are unavailable.
        "pass": False,
        "release_attestation": {
            "pass": True,
            "path": "synthetic/release_attestation.json",
            "checks": {
                "schema_and_hash": True,
                "static_release_requirements_frozen": True,
                "release_requirements_exact": True,
                "commit_and_annotated_tag_bound": True,
                "workflow_exact": True,
                "ci_selection_exact": True,
                "ci_run_success": True,
                "exact_linux_macos_ci_jobs": True,
                "ci_receipt_hash_chain": True,
                "ci_measurements_exact": True,
                "contract_and_policy_hashes": True,
                "sealed_manifest_inventory_hash": True,
            },
            "reasons": [],
        },
        "stage0_selection": {
            "pass": True,
            "path": "synthetic/stage0_selection.json",
            "stage_receipt_path": "synthetic/stage_receipt.json",
            "checks": {
                "sealed_selection": True,
                "complete_source_matrix": True,
                "stage_receipt_go": True,
                "selection_semantic_replay": True,
                "selection_uses_no_a_d_treatment_outcome_fields": True,
            },
            "reasons": [],
        },
        "experiment_c_rule_sensitivities": sensitivities,
        "budget_ledger": {
            "pass": True,
            "raw_root_storage_bytes": 70_035_938,
            "checks": {
                "schema_and_contract": True,
                "self_hash_and_event_chain": True,
                "exact_frozen_caps": True,
                "parent_debit_exact": True,
                "valid_finalized_dispatch_units": True,
                "all_artifact_backed_dispatches_accounted": True,
                "actual_totals_within_caps": True,
            },
            "actual_totals": {
                "cost_usd": 3.212770875,
                "completions": 184,
                "storage_bytes": 70_035_938,
            },
            "actual_stage_cost_usd": {
                "parent_v23": 3.212770875,
                "hosted_confirmatory": 0.0,
                "local": 0.0,
            },
        },
    }


def _synthetic_failure_receipt_control(contract) -> dict[str, Any]:
    return {
        "schema_version": (
            "finevo-pilot-v2.10.1-failure-receipt-control-v1"
        ),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "release_commit": "b5bfa9b86d3cdb706cea5be707597bef8ac85aed",
        "failed_cell_count": 185,
        "failed_stage_counts": {
            "experiment-a": 20,
            "experiment-b": 15,
            "experiment-c": 20,
            "experiment-d": 30,
            "local-experiment-a": 20,
            "local-experiment-b": 25,
            "local-experiment-c": 20,
            "local-experiment-d": 35,
        },
        "unique_receipt_count": 130,
        "unique_receipt_stage_counts": {
            "experiment-a": 20,
            "experiment-b": 15,
            "experiment-c": 20,
            "experiment-d": 5,
            "local-experiment-a": 20,
            "local-experiment-b": 25,
            "local-experiment-c": 20,
            "local-experiment-d": 5,
        },
        "failed_run_ids_sha256": "c" * 64,
        "cell_to_receipt_mapping_sha256": "d" * 64,
        "unique_receipt_inventory_sha256": "e" * 64,
        "release_attestation_file_sha256": "f" * 64,
        "provider_boundary": {
            "fresh_actor_provider_calls": 0,
            "accounted_reserved_effective_usage_zero": True,
            "provider_journals_present": 0,
            "partial_actor_streams_persisted": False,
        },
        "checks": {
            "ledger_artifact_mapping_exact": True,
            "receipt_manifest_schema_and_self_hash": True,
            "receipt_payload_schema_and_file_hash": True,
            "contract_run_spec_release_exact": True,
            "provider_journals_empty": True,
            "zero_budget_snapshots": True,
            "partial_actor_streams_absent": True,
            "unique_receipt_grouping_exact": True,
        },
        "pass": True,
    }


def _attach_failure_receipt_control(
    aggregate: dict[str, Any],
    contract,
) -> dict[str, Any]:
    aggregate["release_controls"]["v2_10_1_failure_receipts"] = (
        _synthetic_failure_receipt_control(contract)
    )
    return aggregate


def _aggregate(contract):
    rows = _rows(contract)
    aggregate = evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator=_denominator(rows),
        release_controls=_release_controls(),
    )
    return rows, aggregate


def _fake_parent_audit(contract) -> dict[str, Any]:
    return {
        "source_contract": SimpleNamespace(
            contract_id=evidence.PILOT_V210_CONTRACT_ID,
            canonical_hash=(
                contract.qref_receipt_verifier_retry_amendment[
                    "failure_classification"
                ]["parent_contract_sha256"]
            ),
        ),
        "evidence": {
            "publication_commit": ("1e96373fa847b44e3418a777c1ed74165ecf2bac"),
            "merge_commit": ("2c4f4750d02c9c6b90051cfaa4f16b8ab16aa637"),
            "root": "evidence/current_v2/pilot-v2.10",
            "package_manifest_file_sha256": (
                "9aa7d07d1d813a5acdea39401e017d5cefe9d85f9127917b119d2453ff972806"
            ),
            "checksums_file_sha256": (
                "b117c3e9d2555af9582c22de08b6e39f1366876d9bc0c6a84b37728533748695"
            ),
            "terminal_status": "complete-with-no-go",
            "status_counts": {"complete": 1, "integrity-stopped": 210},
            "v2_10_hosted_completions": 0,
            "v2_10_hosted_stage_cost_usd": 0.0,
            "scientific_claim_gates_supported": False,
        },
        "qref_failure_receipt": {
            "status": "integrity-stopped",
            "integrity": {
                "content_sha256": (
                    "48ae5807da2c3175b3fd427cc023796e7"
                    "bd81c5b77695789a900474e023da098"
                )
            },
        },
        "raw_inventory": {
            "file_count": 637,
            "storage_bytes": 20_126_496,
            "inventory_sha256": (
                "d8964a15abed0d77598d2c2cf80136e438b67559796cc93f8566dca17e584baa"
            ),
        },
        "provider_construction_during_import": False,
        "provider_calls_during_import": 0,
        "hosted_provider_calls_during_import": 0,
        "hosted_cost_usd_during_import": 0.0,
    }


def test_v2101_terminal_fake_matrix_is_publishable_no_go() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows, aggregate = _aggregate(contract)

    assert aggregate["schema_version"] == ("finevo-pilot-v2.10.1-evidence-package-v1")
    assert aggregate["publication_status"] == "complete-with-no-go"
    assert aggregate["scientific_complete"] is False
    assert aggregate["denominator"]["status_counts"] == {
        "complete": 16,
        "integrity-stopped": 195,
    }
    assert aggregate["itt_row_preservation"]["registered_rows"] == 211
    assert aggregate["itt_row_preservation"]["prerequisite_rows"] == 16
    assert aggregate["itt_row_preservation"]["fresh_a_d_rows"] == 195
    assert aggregate["effect_aggregation_scope"]["fresh_v2_10_1_a_d_cells_only"] is True
    assert aggregate["prerequisites"]["imported_a_d_effect_cells"] == 0
    assert (
        aggregate["parent_evidence_lineage"][
            "parent_rows_imported_into_v2_10_1_effect_aggregate"
        ]
        == 0
    )
    assert len(rows) == 211
    evidence._require_publishable_terminal_denominator(aggregate)


def test_v2101_implementation_failure_summary_is_exact_and_scoped() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _terminal_implementation_failure_rows(contract)
    aggregate = _attach_failure_receipt_control(
        evidence.aggregate_v24_evidence(
            contract,
            rows,
            denominator=_denominator(rows),
            release_controls=_release_controls(),
        ),
        contract,
    )

    summary = evidence._v2101_implementation_failure_summary(
        aggregate,
        rows,
        resolved_git_commit="b5bfa9b86d3cdb706cea5be707597bef8ac85aed",
    )

    assert summary is not None
    assert summary["schema_version"] == (
        "finevo-pilot-v2.10.1-implementation-failure-summary-v1"
    )
    assert summary["classification"] == "implementation-interface-no-go"
    assert summary["root_cause_code"] == (
        "observed-p95-consumer-schema-dispatch-gap"
    )
    assert summary["source_audit"]["producer"].endswith(
        "build_v2101_resealed_observed_p95_authority"
    )
    assert summary["source_audit"]["dedicated_verifier"].endswith(
        "verified_v2101_observed_p95_authority_binding"
    )
    assert summary["source_audit"]["consumer"].endswith(
        "_verify_source_backed_observed_p95_rows"
    )
    assert summary["observed_failure"]["failed_cell_count"] == 185
    assert summary["observed_failure"]["message_sha256"] == (
        "39cb7f19f94e435d9eb4873df49beac2507703522f2ad9ffa7f688a5f6b92ef7"
    )
    assert summary["provider_boundary"] == {
        "failure_phase": "before-provider-construction-and-dispatch",
        "v2_10_1_incremental_local_stage_cost_usd": 0.0,
        "v2_10_1_incremental_hosted_stage_cost_usd": 0.0,
        "v2_10_1_incremental_hosted_cost_usd": 0.0,
        "v2_10_1_incremental_hosted_completions": 0,
        "v2_10_1_fresh_provider_calls": 0,
        "v2_10_1_fresh_actor_provider_calls": 0,
        "v2_10_1_offline_candidate_provider_calls": 0,
        "partial_actor_streams_persisted": False,
    }
    assert summary["raw_failure_receipt_control"][
        "cell_to_receipt_mapping_sha256"
    ] == "d" * 64
    assert summary["raw_failure_receipt_control"]["failed_cell_count"] == 185
    assert summary["raw_failure_receipt_control"]["unique_receipt_count"] == 130
    assert summary["storage_accounting_boundary"] == {
        "budget_ledger_actual_totals_storage_bytes": 70_035_938,
        "canonical_raw_inventory_bound_here": False,
        "canonical_raw_inventory_policy": (
            "compute file_count/storage_bytes/inventory_sha256 separately "
            "after all 211 cells are terminal and no stage process remains"
        ),
    }
    assert summary["outcome_boundary"][
        "actor_action_utility_rule_exposure_outcomes_generated"
    ] is False
    assert summary["outcome_boundary"][
        "actor_performance_treatment_outcome_blind"
    ] is True
    assert (
        summary["outcome_boundary"]["offline_candidate_admission_cells_generated"]
        == 10
    )
    assert (
        summary["outcome_boundary"]["offline_candidate_scientific_use"]
        == "descriptive-only"
    )
    assert summary["outcome_boundary"]["offline_candidate_metrics_observed"] is True
    assert summary["outcome_boundary"]["offline_candidate_metrics_inspected"] is True
    assert summary["outcome_boundary"]["global_a_d_outcome_blind"] is False
    assert summary["outcome_boundary"][
        "offline_candidate_model_capability_evidence"
    ] is False
    assert summary["outcome_boundary"][
        "offline_candidate_treatment_effect_evidence"
    ] is False
    assert summary["retry_boundary"] == {
        "successor_contract_id": "finevo-pilot-v2.10.2",
        "offline_candidate_cells_imported": 0,
        "fresh_a_d_cells_required": 195,
        "offline_candidate_cells_fresh_rerun_required": 10,
        "claim": (
            "V2.10.1 offline candidate metrics are parent evidence only; "
            "V2.10.2 must rerun all 195 A-D cells and cannot import them."
        ),
    }
    assert (
        evidence._implementation_failure_summary_for_contract(
            aggregate,
            rows,
            resolved_git_commit=(
                "b5bfa9b86d3cdb706cea5be707597bef8ac85aed"
            ),
        )
        == summary
    )

    nonterminal_rows = _rows(contract)
    nonterminal_aggregate = evidence.aggregate_v24_evidence(
        contract,
        nonterminal_rows,
        denominator=_denominator(nonterminal_rows),
        release_controls=_release_controls(),
    )
    assert (
        evidence._implementation_failure_summary_for_contract(
            nonterminal_aggregate,
            nonterminal_rows,
            resolved_git_commit=None,
        )
        is None
    )


def test_v2101_post_sensitivity_build_shape_publishes_terminal_no_go() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _terminal_implementation_failure_rows(contract)
    release_controls = _release_controls()

    # Mirror build_pilot_v24_evidence_package after it has validated both
    # sensitivity lanes and folded their availability into the historical
    # V2.10-family top-level release flag.
    release_controls["pass"] = True
    release_controls["pass"] = bool(
        release_controls["pass"]
        and all(
            control["pass"]
            for control in release_controls[
                "experiment_c_rule_sensitivities"
            ].values()
        )
    )
    assert release_controls["pass"] is False
    assert all(
        control["available"] is False and control["pass"] is False
        for control in release_controls[
            "experiment_c_rule_sensitivities"
        ].values()
    )

    aggregate = _attach_failure_receipt_control(
        evidence.aggregate_v24_evidence(
            contract,
            rows,
            denominator=_denominator(rows),
            release_controls=release_controls,
        ),
        contract,
    )
    summary = evidence._v2101_implementation_failure_summary(
        aggregate,
        rows,
        resolved_git_commit="b5bfa9b86d3cdb706cea5be707597bef8ac85aed",
    )

    assert aggregate["release_controls"]["pass"] is False
    assert aggregate["publication_status"] == "complete-with-no-go"
    assert summary is not None
    assert summary["release_boundary"]["base_controls_pass"] == {
        "release_attestation": True,
        "stage0_selection": True,
        "budget_ledger": True,
        "failure_receipts": True,
    }
    assert (
        summary["release_boundary"]["top_level_release_controls_pass"]
        is False
    )
    assert all(
        control["available"] is False and control["pass"] is False
        for control in summary["release_boundary"][
            "experiment_c_rule_sensitivities"
        ].values()
    )


@pytest.mark.parametrize(
    ("tamper", "expected_match"),
    [
        (
            lambda rows, aggregate: rows[16]["failure"].__setitem__(
                "message", "different failure"
            ),
            "implementation-failure summary differs",
        ),
        (
            lambda rows, aggregate: rows[16]["failure"].__setitem__(
                "message_sha256", "0" * 64
            ),
            "implementation-failure summary differs",
        ),
        (
            lambda rows, aggregate: aggregate["budget"]["actual_totals"].__setitem__(
                "completions", 185
            ),
            "implementation-failure summary differs",
        ),
        (
            lambda rows, aggregate: aggregate["budget"].__setitem__(
                "pass", False
            ),
            "implementation-failure summary differs",
        ),
        (
            lambda rows, aggregate: aggregate["budget"]["checks"].__setitem__(
                "parent_debit_exact", False
            ),
            "implementation-failure summary differs",
        ),
        (
            lambda rows, aggregate: aggregate["budget"]["checks"].pop(
                "schema_and_contract"
            ),
            "implementation-failure summary differs",
        ),
        (
            lambda rows, aggregate: aggregate[
                "inherited_budget_boundary"
            ].__setitem__("pass", False),
            "implementation-failure summary differs",
        ),
        (
            lambda rows, aggregate: aggregate[
                "inherited_budget_boundary"
            ]["checks"].__setitem__("denominator_exact", False),
            "implementation-failure summary differs",
        ),
        (
            lambda rows, aggregate: aggregate["release_controls"].__setitem__(
                "pass", True
            ),
            "implementation-failure summary differs",
        ),
        (
            lambda rows, aggregate: aggregate["release_controls"][
                "release_attestation"
            ].__setitem__("pass", False),
            "implementation-failure summary differs",
        ),
        (
            lambda rows, aggregate: aggregate["release_controls"][
                "release_attestation"
            ]["checks"].pop("schema_and_hash"),
            "implementation-failure summary differs",
        ),
        (
            lambda rows, aggregate: aggregate["release_controls"][
                "stage0_selection"
            ].__setitem__("pass", False),
            "implementation-failure summary differs",
        ),
        (
            lambda rows, aggregate: aggregate["release_controls"][
                "stage0_selection"
            ]["checks"].pop("sealed_selection"),
            "implementation-failure summary differs",
        ),
        (
            lambda rows, aggregate: aggregate["release_controls"][
                "budget_ledger"
            ].__setitem__("pass", False),
            "implementation-failure summary differs",
        ),
        (
            lambda rows, aggregate: aggregate["release_controls"][
                "experiment_c_rule_sensitivities"
            ]["local"].__setitem__("available", True),
            "implementation-failure summary differs",
        ),
        (
            lambda rows, aggregate: aggregate["release_controls"][
                "v2_10_1_failure_receipts"
            ]["checks"].__setitem__("provider_journals_empty", False),
            "implementation-failure summary differs",
        ),
        (
            lambda rows, aggregate: next(
                row
                for row in rows
                if row["arm_id"] == "verified-error-candidate"
                and row["stage_id"] == "experiment-c"
            )["metrics"]["rule_reliability"].__setitem__("provider_calls", 1),
            "implementation-failure summary differs",
        ),
        (
            lambda rows, aggregate: next(
                row for row in rows if row["status"] == "failed"
            ).__setitem__("metrics", {"flow_utility": 1.0}),
            "implementation-failure summary differs",
        ),
    ],
)
def test_v2101_implementation_failure_summary_tamper_fails_closed(
    tamper,
    expected_match: str,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _terminal_implementation_failure_rows(contract)
    aggregate = _attach_failure_receipt_control(
        evidence.aggregate_v24_evidence(
            contract,
            rows,
            denominator=_denominator(rows),
            release_controls=_release_controls(),
        ),
        contract,
    )
    tamper(rows, aggregate)

    with pytest.raises(PilotEvidenceError, match=expected_match):
        evidence._v2101_implementation_failure_summary(
            aggregate,
            rows,
            resolved_git_commit=(
                "b5bfa9b86d3cdb706cea5be707597bef8ac85aed"
            ),
        )


def test_v2101_implementation_failure_summary_rejects_wrong_release_commit() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _terminal_implementation_failure_rows(contract)
    aggregate = _attach_failure_receipt_control(
        evidence.aggregate_v24_evidence(
            contract,
            rows,
            denominator=_denominator(rows),
            release_controls=_release_controls(),
        ),
        contract,
    )

    with pytest.raises(
        PilotEvidenceError,
        match="implementation-failure summary differs",
    ):
        evidence._v2101_implementation_failure_summary(
            aggregate,
            rows,
            resolved_git_commit="0" * 40,
        )


def test_v2101_failure_receipt_control_verifies_185_cells_and_130_receipts(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _terminal_implementation_failure_rows(contract)
    raw_root, ledger = _build_failure_receipt_fixture(
        tmp_path,
        contract,
        rows,
    )

    control = evidence._validated_v2101_failure_receipt_control(
        contract,
        ledger=ledger,
        raw_root=raw_root,
        rows=rows,
        resolved_git_commit="b5bfa9b86d3cdb706cea5be707597bef8ac85aed",
        release_controls=_fixture_release_controls(raw_root),
    )

    assert control is not None
    assert control["pass"] is True
    assert control["failed_cell_count"] == 185
    assert control["unique_receipt_count"] == 130
    assert control["unique_receipt_stage_counts"] == {
        "experiment-a": 20,
        "experiment-b": 15,
        "experiment-c": 20,
        "experiment-d": 5,
        "local-experiment-a": 20,
        "local-experiment-b": 25,
        "local-experiment-c": 20,
        "local-experiment-d": 5,
    }
    assert control["provider_boundary"] == {
        "fresh_actor_provider_calls": 0,
        "accounted_reserved_effective_usage_zero": True,
        "provider_journals_present": 0,
        "partial_actor_streams_persisted": False,
    }


@pytest.mark.parametrize(
    "tamper_kind",
    [
        "completed-call",
        "dirty-release",
        "declared-journal",
        "embedded-release-attestation",
        "projection-run-id",
        "experiment-d-spec-group",
        "undeclared-journal",
        "manifest-self-hash",
    ],
)
def test_v2101_failure_receipt_control_tamper_fails_closed(
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _terminal_implementation_failure_rows(contract)
    raw_root, ledger = _build_failure_receipt_fixture(
        tmp_path,
        contract,
        rows,
    )
    ledger_runs = ledger["runs"]
    first_run_id = sorted(ledger_runs)[0]
    first_manifest = raw_root / ledger_runs[first_run_id]["artifact"]

    if tamper_kind == "completed-call":
        _rewrite_failure_receipt(
            first_manifest,
            lambda receipt: receipt["budget_snapshot"].__setitem__(
                "completed_calls", 1
            ),
        )
    elif tamper_kind == "dirty-release":
        _rewrite_failure_receipt(
            first_manifest,
            lambda receipt: receipt["git"].__setitem__("dirty", True),
        )
    elif tamper_kind == "declared-journal":
        _rewrite_failure_receipt(
            first_manifest,
            lambda receipt: receipt["config"][
                "provider_call_journals"
            ].append({"unexpected": True}),
        )
    elif tamper_kind == "embedded-release-attestation":
        _rewrite_failure_receipt(
            first_manifest,
            lambda receipt: receipt["provenance"]["paid_provenance"][
                "release_attestation"
            ].__setitem__("status", "tampered"),
        )
    elif tamper_kind == "projection-run-id":
        _rewrite_failure_receipt(
            first_manifest,
            lambda receipt: receipt["config"]["projection"].__setitem__(
                "run_id", "wrong-run-id"
            ),
        )
    elif tamper_kind == "experiment-d-spec-group":
        d_run_id = next(
            run_id
            for run_id, source in ledger_runs.items()
            if source["spec"]["stage_id"] == "local-experiment-d"
        )
        d_manifest = raw_root / ledger_runs[d_run_id]["artifact"]
        _rewrite_failure_receipt(
            d_manifest,
            lambda receipt: receipt["config"]["run_specs"].pop(),
        )
    elif tamper_kind == "undeclared-journal":
        source = ledger_runs[first_run_id]
        journal = (
            raw_root
            / source["spec"]["stage_id"]
            / "provider_call_journals"
            / f"{first_run_id}--actor.json"
        )
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text("{}\n", encoding="utf-8")
    elif tamper_kind == "manifest-self-hash":
        manifest = json.loads(first_manifest.read_text(encoding="utf-8"))
        manifest["manifest_sha256"] = "0" * 64
        first_manifest.write_text(
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    else:  # pragma: no cover - parameter list is frozen above
        raise AssertionError(tamper_kind)

    with pytest.raises(PilotEvidenceError):
        evidence._validated_v2101_failure_receipt_control(
            contract,
            ledger=ledger,
            raw_root=raw_root,
            rows=rows,
            resolved_git_commit=(
                "b5bfa9b86d3cdb706cea5be707597bef8ac85aed"
            ),
            release_controls=_fixture_release_controls(raw_root),
        )


def test_v2101_budget_boundary_is_exact_and_tamper_fails_closed() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract)
    denominator = _denominator(rows)
    release = _release_controls()

    budget = evidence._v2101_inherited_budget_boundary(
        contract,
        denominator=denominator,
        release_controls=release,
    )
    assert budget is not None
    assert budget["expected_cumulative_prior"]["cost_usd"] == 3.212770875
    assert budget["expected_cumulative_prior"]["hosted_completions"] == 184
    assert budget["expected_cumulative_prior"]["storage_bytes"] == 70_035_938
    assert budget["total_cap_usd"] == 500.0
    assert budget["v2_10_incremental"] == {
        "cost_usd": 0.0,
        "hosted_completions": 0,
    }

    tampered = deepcopy(release)
    tampered["budget_ledger"]["actual_totals"]["storage_bytes"] = 1
    with pytest.raises(
        PilotEvidenceError,
        match="inherited debit/denominator",
    ):
        evidence._v2101_inherited_budget_boundary(
            contract,
            denominator=denominator,
            release_controls=tampered,
        )


def test_v2101_parent_reference_and_lineage_tamper_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    audit = _fake_parent_audit(contract)
    monkeypatch.setattr(
        parent_import,
        "verify_v210_terminal_lineage",
        lambda **_kwargs: audit,
    )

    reference = evidence._validated_v2101_parent_evidence_reference(
        contract,
        contract_path=CONTRACT_PATH,
    )
    assert reference is not None
    assert reference["source_contract_id"] == evidence.PILOT_V210_CONTRACT_ID
    assert reference["source_package_path"] == ("evidence/current_v2/pilot-v2.10")
    assert reference["source_package_copied"] is False
    assert reference["qref_failure_verified"] is True

    tampered_audit = deepcopy(audit)
    tampered_audit["evidence"]["status_counts"] = {
        "complete": 2,
        "integrity-stopped": 209,
    }
    monkeypatch.setattr(
        parent_import,
        "verify_v210_terminal_lineage",
        lambda **_kwargs: tampered_audit,
    )
    with pytest.raises(PilotEvidenceError, match="semantic binding mismatch"):
        evidence._validated_v2101_parent_evidence_reference(
            contract,
            contract_path=CONTRACT_PATH,
        )

    amendment = contract.to_dict()["qref_receipt_verifier_retry_amendment"]
    amendment["failure_classification"]["evidence_package_manifest_file_sha256"] = (
        "0" * 64
    )
    tampered_contract = SimpleNamespace(
        contract_id=evidence.PILOT_V2101_CONTRACT_ID,
        qref_receipt_verifier_retry_amendment=amendment,
    )
    with pytest.raises(PilotEvidenceError, match="immutable V2.10"):
        evidence._v2101_parent_evidence_lineage(tampered_contract)


def _patch_package_lineage(
    monkeypatch: pytest.MonkeyPatch,
    contract,
) -> list[str]:
    chain = evidence._v2101_source_manifest_amendment_chain(contract)
    expected_names = [
        "pilot_v2_10_1_source_manifest.json",
        "pilot_v2_10_source_manifest.json",
        "pilot_v2_9_source_manifest.json",
        "pilot_v2_8_source_manifest.json",
        "pilot_v2_7_source_manifest.json",
        "pilot_v2_6_source_manifest.json",
        "pilot_v2_5_source_manifest.json",
    ]
    assert [name for _, name in chain] == expected_names

    # Supply the renderer's exact final source-manifest binding so this stays
    # a zero-provider package-writer fixture.  The writer itself must continue
    # to reject a draft contract's null/unsealed binding.
    bound_chain = list(chain)
    newest_amendment = contract.to_dict()["qref_receipt_verifier_retry_amendment"]
    newest_source = CONTRACT_PATH.with_name(expected_names[0])
    newest_payload = json.loads(newest_source.read_text(encoding="utf-8"))
    newest_amendment["source_manifest"]["file_sha256"] = hashlib.sha256(
        newest_source.read_bytes()
    ).hexdigest()
    newest_amendment["source_manifest"]["content_sha256"] = newest_payload["integrity"][
        "content_sha256"
    ]
    bound_chain[0] = (newest_amendment, expected_names[0])
    monkeypatch.setattr(
        evidence,
        "_v2101_source_manifest_amendment_chain",
        lambda _contract: tuple(bound_chain),
    )

    lineage = evidence._v2101_parent_evidence_lineage(contract)
    assert lineage is not None
    monkeypatch.setattr(
        evidence,
        "_validated_v2101_parent_evidence_reference",
        lambda *_args, **_kwargs: {
            **lineage,
            "reference_kind": "immutable-external-package-reference",
            "source_package_path": "evidence/current_v2/pilot-v2.10",
            "source_package_copied": False,
            "inventory_verified": True,
            "semantic_binding_verified": True,
        },
    )
    return expected_names


def test_v2101_fake_package_contains_complete_list_driven_source_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows, aggregate = _aggregate(contract)
    expected_names = _patch_package_lineage(monkeypatch, contract)

    package_root = tmp_path / "package"
    manifest_path, checksums_path = evidence._write_v24_package(
        package_root,
        contract_path=CONTRACT_PATH,
        contract=contract,
        rows=rows,
        aggregate=aggregate,
        common_commit=None,
        experiment_c_sensitivities={},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == ("finevo-pilot-v2.10.1-evidence-package-v1")
    assert manifest["publication_status"] == "complete-with-no-go"
    assert manifest["base_contract"]["contract_id"] == (evidence.PILOT_V210_CONTRACT_ID)
    assert [
        Path(entry["package_path"]).name for entry in manifest["source_manifest_chain"]
    ] == expected_names
    assert (
        len({entry["package_path"] for entry in manifest["source_manifest_chain"]}) == 7
    )
    for name in expected_names:
        package_path = f"contract/{name}"
        assert package_path in manifest["published_files"]
        assert (package_root / package_path).is_file()
    checksum_paths = {row["path"] for row in checksums["files"]}
    assert set(manifest["published_files"]).issubset(checksum_paths)
    report = (package_root / "reviewer_report.md").read_text(encoding="utf-8")
    assert (
        "The 16 schema- and hash-verified V2.9 parent/q-ref/Stage-0 "
        "prerequisites are reverified"
    ) in report
    assert "every one of the 195 V2.10.1 A-D cells is fresh" in report


def test_v2101_implementation_failure_is_wired_to_all_package_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _terminal_implementation_failure_rows(contract)
    aggregate = _attach_failure_receipt_control(
        evidence.aggregate_v24_evidence(
            contract,
            rows,
            denominator=_denominator(rows),
            release_controls=_release_controls(),
        ),
        contract,
    )
    _patch_package_lineage(monkeypatch, contract)

    package_root = tmp_path / "implementation-failure-package"
    evidence._write_v24_package(
        package_root,
        contract_path=CONTRACT_PATH,
        contract=contract,
        rows=rows,
        aggregate=aggregate,
        common_commit="b5bfa9b86d3cdb706cea5be707597bef8ac85aed",
        experiment_c_sensitivities={},
    )

    published = json.loads(
        (package_root / "aggregate.json").read_text(encoding="utf-8")
    )
    claim = json.loads(
        (package_root / "claim_metric_artifact.json").read_text(
            encoding="utf-8"
        )
    )
    failure = json.loads(
        (package_root / "failure_ledger.json").read_text(encoding="utf-8")
    )
    report = (package_root / "reviewer_report.md").read_text(encoding="utf-8")

    for payload in (published, claim, failure):
        assert payload["implementation_failure"]["schema_version"] == (
            "finevo-pilot-v2.10.1-implementation-failure-summary-v1"
        )
        assert payload["implementation_failure"]["evidence_use"].endswith(
            "not model-capability, actor-reasoning, or A-D treatment-effect "
            "evidence"
        )
    assert len(failure["rows"]) == 185
    assert published["denominator"]["status_counts"] == {
        "complete": 26,
        "failed": 185,
    }
    assert "observed-p95-consumer-schema-dispatch-gap" in report
    assert "`0` fresh hosted completions and `0` fresh actor provider calls" in report
    assert "budget-ledger actual total only" in report
    assert "canonical raw `file_count/storage_bytes/inventory_sha256`" in report
    assert "generated, observed, and inspected" in report
    assert "global A-D record is not outcome-blind" in report
    assert "map to `130` unique strictly verified receipts" in report
    assert "must freshly rerun all `195` A-D cells" in report
    assert (
        "this is not a model-capability failure or a negative A-D effect result"
        in report
    )


def test_v2101_exact_early_qref_failure_package_preserves_no_outcome_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _early_qref_failure_rows(contract)
    aggregate = evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator=_denominator(rows),
        release_controls=_release_controls(),
    )
    _patch_package_lineage(monkeypatch, contract)

    assert aggregate["denominator"]["status_counts"] == {
        "complete": 1,
        "integrity-stopped": 210,
    }
    assert aggregate["prerequisites"]["all_prerequisites_complete"] is False
    assert aggregate["prerequisites"]["complete_prerequisite_cells"] == 1
    assert aggregate["prerequisites"]["fresh_a_d_complete_cells"] == 0
    assert (
        "incomplete prerequisites are not described as reverified"
        in aggregate["prerequisites"]["claim_boundary"]
    )
    assert (
        "no V2.10.1 A-D outcome was generated"
        in aggregate["prerequisites"]["claim_boundary"]
    )

    package_root = tmp_path / "early-qref-no-go-package"
    evidence._write_v24_package(
        package_root,
        contract_path=CONTRACT_PATH,
        contract=contract,
        rows=rows,
        aggregate=aggregate,
        common_commit=None,
        experiment_c_sensitivities={},
    )

    published = json.loads(
        (package_root / "aggregate.json").read_text(encoding="utf-8")
    )
    failures = json.loads(
        (package_root / "failure_ledger.json").read_text(encoding="utf-8")
    )
    report = (package_root / "reviewer_report.md").read_text(encoding="utf-8")
    assert published["denominator"]["status_counts"] == {
        "complete": 1,
        "integrity-stopped": 210,
    }
    assert len(failures["rows"]) == 210
    assert (
        "All 16 V2.9-derived prerequisite identities remain in the V2.10.1 "
        "denominator, but only 1/16 completed"
    ) in report
    assert (
        "The 195 A-D identities remain registered fresh-only identities, not "
        "imported outcomes"
    ) in report
    assert "no V2.10.1 A-D outcome was generated" in report
    assert (
        "The 16 schema- and hash-verified V2.9 parent/q-ref/Stage-0 "
        "prerequisites are reverified"
    ) not in report
    assert "every one of the 195 V2.10.1 A-D cells is fresh" not in report
