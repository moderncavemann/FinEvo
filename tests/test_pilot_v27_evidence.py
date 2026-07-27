from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from verified_memory import pilot_v24_evidence as evidence
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_evidence import PilotEvidenceError


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_7.yaml"
EXPECTED_COMMIT = "a" * 40
IMPORTED_STAGES = {
    "parent-import",
    "q-ref-resolution",
    "stage0-calibration",
}


def _rows(contract, *, imported_no_go: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in contract.expand():
        imported = spec.stage_id in IMPORTED_STAGES
        stage0 = spec.stage_id == "stage0-calibration"
        status = "integrity-stopped" if imported_no_go and imported else "complete"
        rows.append(
            {
                **spec.to_dict(),
                "status": status,
                "failure": (
                    {"kind": "upstream-no-go"} if status != "complete" else None
                ),
                "artifact_kind": "terminal-summary",
                "artifact_sha256": "a" * 64,
                "scientific_eligible": (
                    (stage0 and status == "complete")
                    if imported
                    else status == "complete"
                ),
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


def _release_controls(*, passed: bool = True) -> dict[str, Any]:
    return {
        "pass": passed,
        "budget_ledger": {
            "pass": True,
            "raw_root_storage_bytes": 0,
            "checks": {"parent_debit_exact": True},
            "actual_totals": {
                "cost_usd": 3.212770875,
                "completions": 184,
                "storage_bytes": 19_181_432,
            },
            "actual_stage_cost_usd": {
                "parent_v23": 3.212770875,
            },
        },
    }


def _install_gate_fixtures(
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


def test_v27_aggregate_separates_imported_calibration_from_a_d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract)
    observed_stage_sets = _install_gate_fixtures(monkeypatch)
    aggregate = evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator=_denominator(rows),
        release_controls=_release_controls(),
    )

    imported = aggregate["imported_prerequisites"]
    assert imported["imported_registered_cells"] == 16
    assert imported["all_imported_complete"] is True
    assert imported["stages"]["parent-import"]["scientific_eligible_cells"] == 0
    assert imported["stages"]["q-ref-resolution"]["scientific_eligible_cells"] == 0
    assert imported["stages"]["stage0-calibration"]["scientific_eligible_cells"] == 14
    assert imported["stages"]["stage0-calibration"]["evidence_scope"] == (
        "stage0-baseline-calibration"
    )
    assert imported["used_in_a_d_effect_gates"] is False
    assert aggregate["inherited_budget_boundary"]["pass"] is True
    assert aggregate["inherited_budget_boundary"]["checks"]["denominator_exact"] is True
    assert observed_stage_sets
    assert all(not (stages & IMPORTED_STAGES) for stages in observed_stage_sets)


def test_v27_import_failure_is_terminal_complete_with_no_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract, imported_no_go=True)
    _install_gate_fixtures(monkeypatch)
    aggregate = evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator=_denominator(rows),
        release_controls=_release_controls(passed=False),
    )

    imported = aggregate["imported_prerequisites"]
    assert imported["all_imported_complete"] is False
    assert imported["stages"]["stage0-calibration"]["status_counts"] == {
        "integrity-stopped": 14
    }
    assert imported["stages"]["stage0-calibration"]["scientific_eligible_cells"] == 0
    assert aggregate["publication_status"] == "complete-with-no-go"
    assert aggregate["scientific_complete"] is False
    real_sha256_file = evidence._sha256_file
    monkeypatch.setattr(
        evidence,
        "_sha256_file",
        lambda path: (
            None
            if Path(path).name == "pilot_v2_7_source_manifest.json"
            else real_sha256_file(path)
        ),
    )
    manifest_path, _ = evidence._write_v24_package(
        tmp_path / "no-go-package",
        contract_path=CONTRACT_PATH,
        contract=contract,
        rows=rows,
        aggregate=aggregate,
        common_commit=EXPECTED_COMMIT,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["publication_status"] == "complete-with-no-go"
    assert manifest["scientific_complete"] is False


def test_v27_import_and_budget_tamper_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    _install_gate_fixtures(monkeypatch)
    rows = _rows(contract)
    denominator = _denominator(rows)
    release = _release_controls()

    imported_index = next(
        index
        for index, row in enumerate(rows)
        if row["stage_id"] == "stage0-calibration"
    )
    missing = [row for index, row in enumerate(rows) if index != imported_index]
    wrong_stage0 = deepcopy(rows)
    wrong_stage0[imported_index]["scientific_eligible"] = False
    parent_index = next(
        index for index, row in enumerate(rows) if row["stage_id"] == "parent-import"
    )
    wrong_parent = deepcopy(rows)
    wrong_parent[parent_index]["scientific_eligible"] = True

    for tampered in (missing, wrong_stage0, wrong_parent):
        with pytest.raises(PilotEvidenceError):
            evidence.aggregate_v24_evidence(
                contract,
                tampered,
                denominator=denominator,
                release_controls=release,
            )

    wrong_check = deepcopy(release)
    wrong_check["budget_ledger"]["checks"]["parent_debit_exact"] = False
    wrong_cost = deepcopy(release)
    wrong_cost["budget_ledger"]["actual_stage_cost_usd"]["parent_v23"] = 0.0
    for tampered_release in (wrong_check, wrong_cost):
        with pytest.raises(
            PilotEvidenceError,
            match="inherited debit/denominator",
        ):
            evidence.aggregate_v24_evidence(
                contract,
                rows,
                denominator=denominator,
                release_controls=tampered_release,
            )


def test_v27_package_copies_complete_contract_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract)
    _install_gate_fixtures(monkeypatch)
    aggregate = evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator=_denominator(rows),
        release_controls=_release_controls(),
    )
    real_sha256_file = evidence._sha256_file

    def draft_hash(path: Path) -> str | None:
        if Path(path).name == "pilot_v2_7_source_manifest.json":
            return None
        return real_sha256_file(path)

    monkeypatch.setattr(evidence, "_sha256_file", draft_hash)
    manifest_path, checksums_path = evidence._write_v24_package(
        tmp_path / "package",
        contract_path=CONTRACT_PATH,
        contract=contract,
        rows=rows,
        aggregate=aggregate,
        common_commit=EXPECTED_COMMIT,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    expected = {
        "contract/pilot_v2_7.yaml",
        "contract/pilot_v2_7_source_manifest.json",
        "contract/pilot_v2_6.yaml",
        "contract/pilot_v2_6_source_manifest.json",
        "contract/pilot_v2_5_source_manifest.json",
        "contract/pilot_v2_4_parent_source_manifest.json",
    }
    assert expected.issubset(set(manifest["published_files"]))
    assert expected.issubset({row["path"] for row in checksums["files"]})
    assert manifest["retry_source_manifest"]["package_path"] == (
        "contract/pilot_v2_7_source_manifest.json"
    )
    assert manifest["inherited_retry_source_manifest"]["package_path"] == (
        "contract/pilot_v2_6_source_manifest.json"
    )
    assert manifest["ancestral_retry_source_manifest"]["package_path"] == (
        "contract/pilot_v2_5_source_manifest.json"
    )
