from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
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


def _release_controls() -> dict[str, Any]:
    sensitivities = {
        lane_id: {
            **evidence._v210_sensitivity_lane_definition(lane_id),
            "pass": False,
            "available": False,
            "provider_calls": 0,
            "descriptive_only": True,
            "effectiveness_gate": False,
            "reason": "fixture A-D rows are terminal but incomplete",
        }
        for lane_id in evidence._V210_C_SENSITIVITY_FILES
    }
    return {
        "pass": False,
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
    aggregate = evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator=_denominator(rows),
        release_controls=_release_controls(),
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
    assert "implementation_failure" not in aggregate
    report = evidence._report_markdown(aggregate)
    assert "V2.10.1 generated no actor performance treatment-effect outcome" in report
    assert "## Terminal implementation failure" not in report
    evidence._require_publishable_terminal_denominator(aggregate)


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
