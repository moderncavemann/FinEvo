from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import pytest

from verified_memory import pilot_evidence
from verified_memory import pilot_orchestrator
from verified_memory import pilot_v2102_parent_import as parent_import
from verified_memory import pilot_v24_evidence as evidence
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_evidence import PilotEvidenceError


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_10_2.yaml"
PREREQUISITE_STAGES = {
    "parent-import",
    "q-ref-resolution",
    "stage0-calibration",
}


def _rows(contract: Any) -> list[dict[str, Any]]:
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
                    else {"kind": "fixture-v2.10.2-terminal-no-go"}
                ),
                "artifact_kind": (
                    "terminal-summary" if prerequisite else None
                ),
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


def _denominator(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "expected_count": 211,
        "observed_ledger_count": len(rows),
        "all_rows_present": len(rows) == 211,
        "all_rows_terminal": True,
        "status_counts": dict(sorted(statuses.items())),
        "all_completed_artifacts_validated": True,
        "pass": len(rows) == 211,
    }


def _release_controls(
    *,
    base_pass: bool = True,
    sensitivity_available: bool = False,
) -> dict[str, Any]:
    sensitivities = {
        lane_id: {
            **evidence._v210_sensitivity_lane_definition(lane_id),
            "pass": sensitivity_available,
            "available": sensitivity_available,
            "provider_calls": 0,
            "descriptive_only": True,
            "effectiveness_gate": False,
            **(
                {
                    "path": (
                        f"/fixture/{lane_id}/"
                        "experiment_c_rule_sensitivity.json"
                    ),
                    "file_sha256": "a" * 64,
                    "content_sha256": "b" * 64,
                    "source_run_count": 5,
                    "grid_cell_count": 9,
                }
                if sensitivity_available
                else {
                    "reason": "fixture A-D rows are terminal but incomplete"
                }
            ),
        }
        for lane_id in evidence._V210_C_SENSITIVITY_FILES
    }
    return {
        "pass": base_pass,
        "experiment_c_rule_sensitivities": sensitivities,
        "budget_ledger": {
            "pass": True,
            "raw_root_storage_bytes": 92_541_342,
            "checks": {
                "parent_debit_exact": True,
            },
            "actual_totals": {
                "cost_usd": 3.212770875,
                "completions": 184,
                "storage_bytes": 92_541_342,
            },
            "actual_stage_cost_usd": {
                "parent_v23": 3.212770875,
                "hosted_confirmatory": 0.0,
                "local": 0.0,
            },
        },
    }


def _fake_parent_audit(contract: Any) -> dict[str, Any]:
    lineage = evidence._v2102_parent_evidence_lineage(contract)
    assert lineage is not None
    return {
        "source_contract": SimpleNamespace(
            contract_id=evidence.PILOT_V2101_CONTRACT_ID,
            canonical_hash=lineage["source_contract_sha256"],
        ),
        "raw_inventory": {
            "file_count": 966,
            "storage_bytes": 23_559_957,
            "inventory_sha256": (
                "63385589f81342822f705c47fe09ce106"
                "29a1ccc667ec13e47e7de36cec31413"
            ),
        },
        "evidence": {
            "publication_commit": lineage["source_evidence_commit"],
            "merge_commit": lineage["source_evidence_merge_commit"],
            "root": lineage["source_evidence_namespace"],
            "checksums_file_sha256": lineage["checksums_file_sha256"],
            "package_manifest_file_sha256": (
                lineage["package_manifest_file_sha256"]
            ),
            "aggregate_file_sha256": lineage["aggregate_file_sha256"],
            "failure_ledger_file_sha256": (
                lineage["failure_ledger_file_sha256"]
            ),
            "reviewer_report_file_sha256": (
                lineage["reviewer_report_file_sha256"]
            ),
            "terminal_status": "complete-with-no-go",
            "status_counts": {"complete": 26, "failed": 185},
            "v2_10_1_incremental_hosted_completions": 0,
            "v2_10_1_incremental_hosted_stage_cost_usd": 0.0,
            "offline_candidate_admission_cells_observed": 10,
            "actor_performance_treatment_outcome_blind": True,
            "scientific_claim_gates_supported": False,
        },
        "provider_construction_during_import": False,
        "provider_calls_during_import": 0,
        "hosted_provider_calls_during_import": 0,
        "hosted_cost_usd_during_import": 0.0,
    }


def test_v2102_is_active_prerequisite_family_with_own_wire_identity() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)

    assert pilot_evidence._is_v2102_contract(contract)
    assert pilot_evidence._is_v210_prerequisite_family_contract(contract)
    assert pilot_evidence._v210_prerequisite_wire_identity(contract) == (
        "finevo-pilot-v2.10.2-imported-qref-resolution-v1",
        "immutable-v2.9-prerequisite-import-offline-v2.10.2-reseal",
    )
    assert pilot_evidence._stage_sets(contract) == (
        pilot_evidence.V24_NON_SCIENTIFIC_STAGES,
        pilot_evidence.V24_SCIENTIFIC_STAGES,
    )
    expected_parent = pilot_evidence._expected_parent_budget_debit(contract)
    assert expected_parent is not None
    assert expected_parent["cost_usd"] == 3.212770875
    assert expected_parent["hosted_completions"] == 184
    assert expected_parent["storage_bytes"] == 92_541_342


def test_v2102_aggregate_uses_only_fresh_child_effect_rows() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract)
    historical_model_boundaries = (
        evidence._validated_v2102_historical_model_boundaries(
            contract,
            repository_root=ROOT,
        )
    )
    aggregate = evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator=_denominator(rows),
        release_controls=_release_controls(),
        historical_model_boundaries=historical_model_boundaries,
    )

    assert aggregate["schema_version"] == (
        "finevo-pilot-v2.10.2-evidence-package-v1"
    )
    assert aggregate["publication_status"] == "complete-with-no-go"
    assert aggregate["denominator"]["status_counts"] == {
        "complete": 16,
        "integrity-stopped": 195,
    }
    assert aggregate["itt_row_preservation"]["registered_rows"] == 211
    assert aggregate["itt_row_preservation"]["prerequisite_rows"] == 16
    assert aggregate["itt_row_preservation"]["fresh_a_d_rows"] == 195
    assert (
        aggregate["effect_aggregation_scope"]["fresh_v2_10_2_a_d_cells_only"]
        is True
    )
    assert aggregate["prerequisites"]["terminal_parent_contract_id"] == (
        evidence.PILOT_V2101_CONTRACT_ID
    )
    assert aggregate["prerequisites"]["imported_a_d_effect_cells"] == 0
    assert aggregate["parent_evidence_lineage"]["parent_status_counts"] == {
        "complete": 26,
        "failed": 185,
    }
    assert (
        aggregate["parent_evidence_lineage"][
            "parent_rows_imported_into_v2_10_2_effect_aggregate"
        ]
        == 0
    )
    assert aggregate["inherited_budget_boundary"][
        "expected_cumulative_prior"
    ]["storage_bytes"] == 92_541_342
    assert aggregate["release_controls"]["pass"] is True
    assert "release-stage0-budget" not in {
        row["scope"] for row in aggregate["claim_narrowing"]
    }
    assert {
        row["scope"]
        for row in aggregate["claim_narrowing"]
        if row["scope"].endswith("/experiment-c-sensitivity")
    } == {
        "local/experiment-c-sensitivity",
        "gpt52/experiment-c-sensitivity",
    }
    sensitivity_claims = [
        row
        for row in aggregate["claims"]
        if row["lane"] in {"local", "gpt52"}
        and "zero-API Experiment C rule sensitivity" in row["claim"]
    ]
    assert len(sensitivity_claims) == 2
    assert all(row["status"] == "no-go" for row in sensitivity_claims)
    assert all("is available" not in row["claim"] for row in sensitivity_claims)
    assert all(
        row["artifact"].startswith(
            "aggregate.json#/experiment_c_rule_sensitivities/"
        )
        for row in sensitivity_claims
    )
    gpt56 = aggregate["historical_model_boundaries"]["gpt56_diagnostic"]
    assert gpt56["capability_tasks_passed"] == 30
    assert gpt56["closed_loop_preflight_calls_accounted"] == 16
    assert gpt56["directional_cell_status_counts"] == {
        "budget-stopped": 6
    }
    assert gpt56["matched_a_a_null_registered"] is False
    assert gpt56["paired_delta"] is None
    assert gpt56["directional_micro_pilot_replication"] is False
    assert gpt56["v2_10_2_redispatched"] is False
    assert gpt56["v2_10_2_effect_rows_imported"] == 0
    assert aggregate["claim_narrowing"][-1]["scope"] == (
        "historical-model/gpt56_diagnostic"
    )
    assert "implementation_failure" not in aggregate
    report = evidence._report_markdown(aggregate)
    assert "V2.10.1 generated no actor performance treatment-effect outcome" in report
    assert "## Terminal implementation failure" not in report
    assert "## Frozen model choice and historical GPT-5.6 boundary" in report
    assert "`frozen historical diagnostic only`" in report
    assert "not a V2.10.2 treatment lane" in report
    assert "`6/6 budget-stopped`" in report
    assert "no paired delta, no matched A/A null" in report
    assert "GPT-5.6 was not redispatched" in report
    assert "not a negative effect result" in report
    assert "prospective registered GPT-5.6 replication lane" in report
    assert "backbone-independent claim" in report
    evidence._require_publishable_terminal_denominator(aggregate)


def test_v2102_sensitivity_controls_do_not_mask_a_real_base_failure() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract)
    aggregate = evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator=_denominator(rows),
        release_controls=_release_controls(
            base_pass=False,
            sensitivity_available=True,
        ),
    )

    assert aggregate["release_controls"]["pass"] is False
    assert {
        row["scope"]
        for row in aggregate["claim_narrowing"]
        if row["scope"] == "release-stage0-budget"
    } == {"release-stage0-budget"}
    assert not any(
        row["scope"].endswith("/experiment-c-sensitivity")
        for row in aggregate["claim_narrowing"]
    )


def test_v2102_available_sensitivities_remain_descriptive() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract)
    aggregate = evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator=_denominator(rows),
        release_controls=_release_controls(sensitivity_available=True),
    )

    assert aggregate["release_controls"]["pass"] is True
    claims = [
        row
        for row in aggregate["claims"]
        if row["lane"] in {"local", "gpt52"}
        and "zero-API Experiment C rule sensitivity" in row["claim"]
    ]
    assert len(claims) == 2
    assert all(row["status"] == "complete-descriptive" for row in claims)
    assert all(row["artifact"].endswith(".json") for row in claims)
    assert not any(
        row["scope"].endswith("/experiment-c-sensitivity")
        for row in aggregate["claim_narrowing"]
    )


def test_v2102_aggregate_rejects_forged_gpt56_effect_boundary() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract)
    historical = evidence._validated_v2102_historical_model_boundaries(
        contract,
        repository_root=ROOT,
    )
    assert historical is not None
    forged = deepcopy(historical)
    forged["gpt56_diagnostic"]["v2_10_2_effect_rows_imported"] = 1

    with pytest.raises(
        PilotEvidenceError,
        match="historical GPT-5.6 boundary summary drifted",
    ):
        evidence.aggregate_v24_evidence(
            contract,
            rows,
            denominator=_denominator(rows),
            release_controls=_release_controls(),
            historical_model_boundaries=forged,
        )


def test_v2102_historical_gpt56_boundary_rejects_tampered_v23_aggregate(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    repository_root = tmp_path / "repo"
    source_manifest = (
        repository_root
        / "experiments"
        / "pilot_v2_4_parent_source_manifest.json"
    )
    source_manifest.parent.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "experiments" / "pilot_v2_4_parent_source_manifest.json",
        source_manifest,
    )
    package_root = (
        repository_root / "evidence" / "current_v2" / "pilot-v2.3"
    )
    shutil.copytree(
        ROOT / "evidence" / "current_v2" / "pilot-v2.3",
        package_root,
    )
    aggregate_path = package_root / "aggregate.json"
    aggregate_path.write_bytes(aggregate_path.read_bytes() + b"\n")

    with pytest.raises(
        PilotEvidenceError,
        match="aggregate file hash mismatch",
    ):
        evidence._validated_v2102_historical_model_boundaries(
            contract,
            repository_root=repository_root,
        )


def test_v2101_failure_signature_cannot_become_v2102_failure_summary() -> None:
    failure = {
        "message": (
            "source-backed observed p95 receipt verification failed: "
            "observed-p95 receipt top-level shape or schema drifted"
        ),
        "message_sha256": (
            "39cb7f19f94e435d9eb4873df49beac"
            "2507703522f2ad9ffa7f688a5f6b92ef7"
        ),
    }
    aggregate = {
        "contract_id": evidence.PILOT_V2102_CONTRACT_ID,
        "denominator": {"status_counts": {"complete": 26, "failed": 185}},
    }

    assert (
        evidence._implementation_failure_summary_for_contract(
            aggregate,
            [{"failure": deepcopy(failure)}],
            resolved_git_commit=parent_import.V2101_PARENT_SCIENCE_COMMIT,
        )
        is None
    )


def test_v2102_parent_reference_is_immutable_lineage_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    audit = _fake_parent_audit(contract)
    monkeypatch.setattr(
        parent_import,
        "verify_v2101_terminal_lineage",
        lambda **_kwargs: deepcopy(audit),
    )

    reference = evidence._validated_v2102_parent_evidence_reference(
        contract,
        contract_path=CONTRACT_PATH,
    )

    assert reference is not None
    assert reference["source_contract_id"] == evidence.PILOT_V2101_CONTRACT_ID
    assert reference["reference_kind"] == (
        "immutable-external-package-reference"
    )
    assert reference["source_package_copied"] is False
    assert reference["implementation_no_go_verified"] is True
    assert reference["offline_candidate_disclosure_verified"] is True
    assert reference["parent_rows_imported_into_v2_10_2_effect_aggregate"] == 0

    tampered = deepcopy(audit)
    tampered["evidence"]["status_counts"] = {"complete": 27, "failed": 184}
    monkeypatch.setattr(
        parent_import,
        "verify_v2101_terminal_lineage",
        lambda **_kwargs: deepcopy(tampered),
    )
    with pytest.raises(
        PilotEvidenceError,
        match="parent evidence semantic binding mismatch",
    ):
        evidence._validated_v2102_parent_evidence_reference(
            contract,
            contract_path=CONTRACT_PATH,
        )


def test_v2102_source_manifest_chain_is_complete_and_newest_first() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    chain = evidence._v2102_source_manifest_amendment_chain(contract)
    names = [name for _, name in chain]

    assert names == [
        "pilot_v2_10_2_source_manifest.json",
        "pilot_v2_10_1_source_manifest.json",
        "pilot_v2_10_source_manifest.json",
        "pilot_v2_9_source_manifest.json",
        "pilot_v2_8_source_manifest.json",
        "pilot_v2_7_source_manifest.json",
        "pilot_v2_6_source_manifest.json",
        "pilot_v2_5_source_manifest.json",
    ]
    assert all(amendment is not None for amendment, _ in chain)


def test_v2102_qref_wire_roundtrips_and_rejects_v2101_disposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="q-ref-resolution")[0].to_dict()
    raw_root = tmp_path / "experiment_results" / "pilot-v2.10.2" / "raw"
    source_path = (
        raw_root
        / "parent-import"
        / "v2_9_raw_snapshot"
        / "q-ref-resolution"
        / "q_ref_resolution.json"
    )
    source_path.parent.mkdir(parents=True)
    source_path.write_text('{"source":"immutable-v2.9"}\n', encoding="utf-8")
    current_path = raw_root / "q-ref-resolution" / "q_ref_resolution.json"
    current_path.parent.mkdir(parents=True)
    verified = {
        "schema_version": (
            "finevo-pilot-v2.10.2-imported-qref-resolution-v1"
        ),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "q_ref": 63.50397933257746,
        "row_count": 48,
        "provider_calls_current_attempt": 0,
        "hosted_provider_calls_current_attempt": 0,
        "provider_construction_current_attempt": False,
        "scientific_evidence": False,
        "source_import": {
            "source_artifacts": {
                "q_ref_resolution": {
                    "snapshot_path": str(source_path),
                    "file_sha256": pilot_evidence._sha256_file(source_path),
                }
            }
        },
        "integrity": {
            "canonicalization": "json-sort-keys-utf8-v1",
        },
    }
    verified["integrity"]["content_sha256"] = (
        pilot_evidence._bound_artifact_hash(verified)
    )
    current_path.write_text(
        json.dumps(verified, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = {
        "metrics": {"q_ref": verified["q_ref"]},
        "gate_evidence": {
            "status": "pass",
            "execution_disposition": (
                "immutable-v2.9-prerequisite-import-offline-v2.10.2-reseal"
            ),
            "q_ref_resolution": {
                "path": str(current_path),
                "file_sha256": pilot_evidence._sha256_file(current_path),
                "content_sha256": verified["integrity"]["content_sha256"],
            },
            "provider_calls_current_attempt": 0,
        },
        "q_ref_resolution": {
            "q_ref": verified["q_ref"],
            "row_count": verified["row_count"],
            "source_resolution": str(source_path),
            "source_resolution_sha256": pilot_evidence._sha256_file(
                source_path
            ),
            "resolution_artifact": str(current_path),
        },
        "provider_calls": 0,
    }
    monkeypatch.setattr(
        pilot_orchestrator,
        "_load_verified_q_ref",
        lambda *_args, **_kwargs: deepcopy(verified),
    )

    pilot_evidence._validate_terminal_payload_marker(
        contract,
        spec,
        payload,
        raw_root=raw_root,
    )

    cross_version = deepcopy(payload)
    cross_version["gate_evidence"]["execution_disposition"] = (
        "immutable-v2.9-prerequisite-import-offline-v2.10.1-reseal"
    )
    with pytest.raises(
        PilotEvidenceError,
        match="zero-call disposition drifted",
    ):
        pilot_evidence._validate_terminal_payload_marker(
            contract,
            spec,
            cross_version,
            raw_root=raw_root,
        )


def test_v2102_parent_terminal_uses_new_receipt_verifier_boundary(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="parent-import")[0].to_dict()
    observed: list[dict[str, Any]] = []

    def verify(
        receipt_path: str,
        *,
        repo_root: Path,
        contract: Any,
        expected_git_commit: str,
    ) -> dict[str, Any]:
        observed.append(
            {
                "receipt_path": receipt_path,
                "repo_root": repo_root,
                "contract_id": contract.contract_id,
                "commit": expected_git_commit,
            }
        )
        return {"integrity": {"content_sha256": "a" * 64}}

    pilot_evidence._validate_terminal_payload_marker(
        contract,
        spec,
        {
            "metrics": {},
            "gate_evidence": {
                "receipt": "synthetic/v2.10.2-parent-receipt.json",
                "receipt_content_sha256": "a" * 64,
                "provider_calls_during_import": 0,
                "scientific_evidence": False,
            },
            "provider_calls": 0,
        },
        raw_root=tmp_path,
        resolved_git_commit="b" * 40,
        parent_import_receipt_verifier=verify,
        source_repo_root=tmp_path,
    )

    assert observed == [
        {
            "receipt_path": "synthetic/v2.10.2-parent-receipt.json",
            "repo_root": tmp_path,
            "contract_id": evidence.PILOT_V2102_CONTRACT_ID,
            "commit": "b" * 40,
        }
    ]


def test_v2102_runner_rebuild_forwards_external_authority_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="local-experiment-a")[0]
    source_root = (tmp_path / "science-source").resolve()
    source_root.mkdir()
    raw_root = source_root / "experiment_results/pilot-v2.10.2/raw"
    raw_root.mkdir(parents=True)
    observed: dict[str, Any] = {
        "reservation_calls": 0,
        "config_calls": 0,
    }

    def reservations(*args: Any, **kwargs: Any) -> dict[str, Any]:
        observed["reservation_calls"] += 1
        observed["reservation_authority"] = kwargs.get("authority_repo_root")
        return {}

    def config_for_spec(*args: Any, **kwargs: Any) -> dict[str, Any]:
        observed["config_calls"] += 1
        observed["config_authority"] = kwargs.get("authority_repo_root")
        return {"fixture": True}

    monkeypatch.setattr(
        pilot_orchestrator,
        "_runner_p95_reservations",
        reservations,
    )
    monkeypatch.setattr(
        pilot_orchestrator,
        "config_for_spec",
        config_for_spec,
    )
    monkeypatch.setattr(
        "verified_memory.runner.build_sealed_run_config",
        lambda *args, **kwargs: {"schema_version": "fixture-runner-v3"},
    )
    monkeypatch.setattr(
        pilot_evidence,
        "_validate_provider_usage_rows",
        lambda *args, **kwargs: None,
    )

    profile = contract.provider_profiles[spec.model_id]
    commit = "b" * 40
    for _ in range(2):
        pilot_evidence._validate_standard_run_contract(
            contract,
            spec.to_dict(),
            config={"schema_version": "fixture-runner-v3"},
            summary={"provider_model": f"ollama/{profile.requested_model}"},
            records={"api_usage": []},
            provenance_git={
                "git_tag": contract.implementation["required_git_tag"],
                "head_commit": commit,
                "tag_commit": commit,
                "tag_object_type": "tag",
                "worktree_clean": True,
                "contract_binding": {},
            },
            raw_root=raw_root,
            source_repo_root=source_root,
        )

    assert observed == {
        "reservation_calls": 2,
        "config_calls": 2,
        "reservation_authority": source_root,
        "config_authority": source_root,
    }


def test_v2102_config_forwards_external_authority_to_utility_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="local-experiment-a")[0]
    source_root = (tmp_path / "science-source").resolve()
    source_root.mkdir()
    raw_root = source_root / "experiment_results/pilot-v2.10.2/raw"
    raw_root.mkdir(parents=True)
    observed: dict[str, Any] = {}

    def stage0_selection(
        called_contract: Any,
        *,
        raw_root: Path,
        paid: Any,
        authority_repo_root: Path,
    ) -> dict[str, Any]:
        observed.update(
            {
                "contract": called_contract,
                "raw_root": raw_root,
                "paid": paid,
                "authority_repo_root": authority_repo_root,
            }
        )
        return {
            "selected_utility": {
                "rho": 1.0,
                "labor_weight": 2.0,
                "inverse_frisch": 0.5,
                "consumption_scale": 63.50397933257746,
                "discount_factor": 0.99,
            }
        }

    monkeypatch.setattr(
        pilot_orchestrator,
        "_load_verified_stage0_selection",
        stage0_selection,
    )

    config = pilot_orchestrator.config_for_spec(
        contract,
        spec,
        raw_root=raw_root,
        paid_provenance=None,
        authority_repo_root=source_root,
        diagnostic_override=True,
    )

    assert observed == {
        "contract": contract,
        "raw_root": raw_root,
        "paid": None,
        "authority_repo_root": source_root,
    }
    assert config.utility.consumption_scale == 63.50397933257746
