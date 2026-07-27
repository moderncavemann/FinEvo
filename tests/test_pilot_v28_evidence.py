from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from verified_memory import pilot_v24_evidence as evidence
from verified_memory import pilot_evidence as core_evidence
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_evidence import PilotEvidenceError


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_8.yaml"
PREREQUISITE_STAGES = {
    "parent-import",
    "q-ref-resolution",
    "stage0-calibration",
}


def test_v28_uses_exact_lane_separated_stage_partition() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)

    assert core_evidence._stage_sets(contract) == (
        core_evidence.V24_NON_SCIENTIFIC_STAGES,
        core_evidence.V24_SCIENTIFIC_STAGES,
    )
    assert core_evidence._expected_parent_budget_debit(contract) == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "stage_bucket": "parent_v23",
        "cost_usd": 3.212770875,
        "hosted_completions": 184,
        "storage_bytes": 32_158_175,
        "parent_contract_sha256": (
            "938627d42ec8ec78e8424793797593736b79936b00813b81259af54e6df6779f"
        ),
        "parent_run_ledger_sha256": (
            "ab532bb56232efbc42d6e5f48c9f80c451f461a732cf2607774f6055de9deb4a"
        ),
        "parent_budget_ledger_sha256": (
            "70ff3f40bbebaea766c6403fc1f2879af9002faff287a112a39c2ce405d92170"
        ),
        "record_sha256": (
            "a5caad9515eb797a035c26d32d0a0cf7bfd7f0df210e7362bd3b93da18ff3ff7"
        ),
    }


def test_v28_parent_import_marker_accepts_v28_zero_call_field(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="parent-import")[0]
    receipt_hash = "c" * 64
    resolved_commit = "b" * 40
    calls: list[dict[str, Any]] = []

    def verifier(
        receipt_path,
        *,
        repo_root,
        contract,
        expected_git_commit,
    ):
        calls.append(
            {
                "receipt_path": receipt_path,
                "repo_root": repo_root,
                "contract_id": contract.contract_id,
                "expected_git_commit": expected_git_commit,
            }
        )
        return {"integrity": {"content_sha256": receipt_hash}}

    payload = {
        "metrics": {},
        "gate_evidence": {
            "receipt": str(tmp_path / "parent-import-receipt.json"),
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
        parent_import_receipt_verifier=verifier,
    )

    assert calls == [
        {
            "receipt_path": payload["gate_evidence"]["receipt"],
            "repo_root": ROOT,
            "contract_id": "finevo-pilot-v2.8",
            "expected_git_commit": resolved_commit,
        }
    ]
    bad_payload = deepcopy(payload)
    bad_payload["gate_evidence"]["provider_calls_during_import"] = 1
    with pytest.raises(PilotEvidenceError, match="exact zero-call marker"):
        core_evidence._validate_terminal_payload_marker(
            contract,
            spec.to_dict(),
            bad_payload,
            raw_root=tmp_path,
            resolved_git_commit=resolved_commit,
            parent_import_receipt_verifier=verifier,
        )


def _rows(contract) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in contract.expand():
        prerequisite = spec.stage_id in PREREQUISITE_STAGES
        rows.append(
            {
                **spec.to_dict(),
                "status": "complete",
                "failure": None,
                "artifact_kind": "terminal-summary",
                "artifact_sha256": "a" * 64,
                "scientific_eligible": (
                    spec.stage_id == "stage0-calibration"
                    if prerequisite
                    else True
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


def _install_gate_fixtures(
    monkeypatch: pytest.MonkeyPatch,
    *,
    unsupported_stage: str | None = None,
) -> list[set[str]]:
    observed_stage_sets: list[set[str]] = []

    def record(rows) -> None:
        observed_stage_sets.append({str(row["stage_id"]) for row in rows})

    def status(stage: str) -> str:
        return "unsupported" if stage == unsupported_stage else "supported"

    def c_gate(_contract, rows, *, stage_id, model_id):
        record(rows)
        state = status("experiment-c")
        return {
            "status": state,
            "scientific_evidence_complete": True,
            "same_direction_counts": {"false_activation": 5},
            "claim_action": (
                f"retain {model_id}/{stage_id}"
                if state == "supported"
                else "withdraw rule-reliability claim"
            ),
        }

    def a_gate(_contract, rows, *, stage_id, model_id):
        record(rows)
        state = status("experiment-a")
        return {
            "status": state,
            "scientific_evidence_complete": True,
            "primary_contrast": {
                "raw_paired_deltas": {
                    str(seed): 1.0 for seed in _contract.seeds["sets"]["main"]
                }
            },
            "threshold_gate": {"same_direction_count": 5},
            "claim_action": (
                f"retain {model_id}/{stage_id}"
                if state == "supported"
                else "retain route traceability only"
            ),
        }

    def d_gate(_contract, rows, *, stage_id, model_id, arms):
        record(rows)
        state = status("experiment-d")
        return {
            "status": state,
            "scientific_evidence_complete": True,
            "supported_treatments": ["no-memory"] if state == "supported" else [],
            "treatment_gates": {
                "no-memory": {
                    "six_step_discounted_utility_gate": {
                        "treatment_deltas": {
                            str(seed): 1.0
                            for seed in _contract.seeds["sets"]["main"]
                        }
                    }
                }
            },
            "claim_action": (
                f"retain {model_id}/{stage_id}/{tuple(arms)!r}"
                if state == "supported"
                else "report prompt sensitivity only"
            ),
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


def test_v28_aggregate_has_separate_namespace_prerequisites_and_budget(
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

    assert aggregate["schema_version"] == (
        "finevo-pilot-v2.8-evidence-package-v1"
    )
    assert aggregate["evidence_namespace"] == "current_v2/pilot-v2.8"
    assert aggregate["denominator"]["expected_count"] == 211
    assert len(evidence._sanitized_rows(rows)) == 211
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
    assert prerequisites["stage0_imported_cells"] == 14
    assert prerequisites["q_ref_provider_accounting"] == {
        "hosted_provider_calls": 0,
        "hosted_cost_usd": 0.0,
        "scripted_diagnostic_calls": 48,
    }
    assert prerequisites["used_in_a_d_effect_gates"] is False
    assert prerequisites["stages"]["parent-import"]["treatment_effect_evidence"] is False
    assert (
        prerequisites["stages"]["q-ref-resolution"]["origin"]
        == "fresh-v2.8-scripted-diagnostic"
    )
    assert (
        prerequisites["stages"]["stage0-calibration"]["scientific_eligible_cells"]
        == 14
    )
    budget = aggregate["inherited_budget_boundary"]
    assert budget["total_cap_usd"] == 500.0
    assert budget["expected_cumulative_prior"]["cost_usd"] == 3.212770875
    assert budget["expected_cumulative_prior"]["hosted_completions"] == 184
    assert budget["q_ref_incremental"]["hosted_provider_calls"] == 0
    assert budget["q_ref_incremental"]["scripted_diagnostic_calls"] == 48
    lineage = aggregate["parent_evidence_lineage"]
    assert lineage["source_evidence_status"] == "complete-with-no-go"
    assert lineage["source_evidence_commit"] == (
        "f15a26418264b5de31f53dbe7c46c1949761fcb6"
    )
    assert lineage["source_evidence_merge_commit"] == (
        "e951aa865186a7c2e841316fc6bb08a716aeaf80"
    )
    assert lineage["root_cause"]["code"] == (
        "qref-contract-cell-id-conflated-with-runner-execution-id"
    )
    assert aggregate["effect_aggregation_scope"]["v2_8_a_d_cells_only"] is True
    assert set(
        aggregate["effect_aggregation_scope"]["prerequisite_stage_ids_excluded"]
    ) == PREREQUISITE_STAGES
    assert observed_stage_sets
    assert all(not (stages & PREREQUISITE_STAGES) for stages in observed_stage_sets)

    report = evidence._report_markdown(aggregate)
    assert "$500.0" in report
    assert "`0` hosted provider calls" in report
    assert "`48` scripted diagnostic calls" in report
    assert "qref-contract-cell-id-conflated-with-runner-execution-id" in report
    assert "e951aa865186a7c2e841316fc6bb08a716aeaf80" in report


@pytest.mark.parametrize(
    "unsupported_stage",
    ["experiment-c", "experiment-a", "experiment-d"],
)
def test_v28_any_c_a_d_gate_failure_is_complete_with_narrowed_claim(
    monkeypatch: pytest.MonkeyPatch,
    unsupported_stage: str,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract)
    stopped = next(row for row in rows if row["stage_id"] == unsupported_stage)
    stopped["status"] = "integrity-stopped"
    stopped["failure"] = {"kind": "fixture-gate-failure"}
    stopped["scientific_eligible"] = False
    _install_gate_fixtures(monkeypatch, unsupported_stage=unsupported_stage)
    aggregate = evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator=_denominator(rows),
        release_controls=_release_controls(),
    )

    assert aggregate["scientific_matrix_complete"] is True
    assert aggregate["scientific_claim_gates_supported"] is False
    assert aggregate["scientific_complete"] is False
    assert aggregate["publication_status"] == "complete-with-no-go"
    narrowed = {
        item["scope"]: item for item in aggregate["claim_narrowing"]
    }
    assert f"local/{unsupported_stage}" in narrowed
    assert f"gpt52/{unsupported_stage}" in narrowed
    assert "preregistered mechanism gate was not supported" in (
        narrowed[f"local/{unsupported_stage}"]["reason"]
    )
    assert len(evidence._sanitized_rows(rows)) == 211
    assert aggregate["itt_row_preservation"]["retained_rows"] == 211
    assert aggregate["itt_row_preservation"]["failed_or_stopped_rows"] == 1
    assert aggregate["itt_row_preservation"]["status_counts"] == {
        "complete": 210,
        "integrity-stopped": 1,
    }


def test_v28_parent_reference_revalidates_without_rewriting_v27() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    parent_root = ROOT / "evidence" / "current_v2" / "pilot-v2.7"
    before = {
        path.relative_to(parent_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in parent_root.rglob("*")
        if path.is_file()
    }

    reference = evidence._validated_v28_parent_evidence_reference(
        contract,
        contract_path=CONTRACT_PATH,
    )

    after = {
        path.relative_to(parent_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in parent_root.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert reference is not None
    assert reference["reference_kind"] == "immutable-external-package-reference"
    assert reference["source_package_copied"] is False
    assert reference["inventory_verified"] is True
    assert reference["semantic_binding_verified"] is True
    assert reference["package_manifest_file_sha256"] == (
        "1b44a8984b61f00cbae4851a599674fb3e0479ca60d3259961460f99519e23bb"
    )


def test_v28_parent_reference_checksum_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    repository = tmp_path / "repo"
    experiments = repository / "experiments"
    experiments.mkdir(parents=True)
    fake_contract_path = experiments / CONTRACT_PATH.name
    shutil.copyfile(CONTRACT_PATH, fake_contract_path)
    parent_copy = repository / "evidence" / "current_v2" / "pilot-v2.7"
    shutil.copytree(
        ROOT / "evidence" / "current_v2" / "pilot-v2.7",
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
        evidence._validated_v28_parent_evidence_reference(
            contract,
            contract_path=fake_contract_path,
        )


def test_v28_prerequisite_or_budget_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract)
    _install_gate_fixtures(monkeypatch)
    bad_rows = deepcopy(rows)
    qref = next(
        row for row in bad_rows if row["stage_id"] == "q-ref-resolution"
    )
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
