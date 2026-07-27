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
            "reason": "fixture terminal no-go has no complete C source rows",
        }
    return {
        "pass": True,
        "experiment_c_rule_sensitivities": sensitivities,
        "budget_ledger": {
            "pass": True,
            "raw_root_storage_bytes": 70_035_938,
            "checks": {"parent_debit_exact": True},
            "actual_totals": {
                "cost_usd": 3.212770875,
                "completions": 184,
                "storage_bytes": 70_035_938,
            },
            "actual_stage_cost_usd": {
                "parent_v23": 3.212770875,
            },
        },
    }


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
