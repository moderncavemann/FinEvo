from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from verified_memory.pilot_contract import load_pilot_contract
import verified_memory.pilot_evidence as core_evidence
import verified_memory.pilot_v24_evidence as lane_evidence


ROOT = Path(__file__).resolve().parents[1]
V24_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_4.yaml"
V25_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_5_overlay.yaml"


def _rows(contract) -> list[dict[str, Any]]:
    _, scientific_stages = core_evidence._stage_sets(contract)
    return [
        {
            **spec.to_dict(),
            "status": "complete",
            "failure": None,
            "artifact_kind": "terminal-summary",
            "artifact_sha256": "a" * 64,
            "scientific_eligible": spec.stage_id in scientific_stages,
            "metrics": {},
            "gate_evidence": {},
            "capability": {},
            "narrative": {},
        }
        for spec in contract.expand()
    ]


def _denominator(contract) -> dict[str, Any]:
    count = len(contract.expand())
    return {
        "expected_count": count,
        "observed_ledger_count": count,
        "all_rows_present": True,
        "all_rows_terminal": True,
        "status_counts": {"complete": count},
        "all_completed_artifacts_validated": True,
        "pass": True,
    }


def _release_controls() -> dict[str, Any]:
    return {
        "pass": True,
        "release_attestation": {"pass": True},
        "stage0_selection": {"pass": True},
        "budget_ledger": {
            "pass": True,
            "actual_totals": {
                "cost_usd": 3.212770875,
                "completions": 184,
                "storage_bytes": 4_714_322,
            },
            "actual_stage_cost_usd": {
                "parent_v23": 3.212770875,
                "local": 0.0,
                "hosted_confirmatory": 0.0,
                "manual_reserve": 0.0,
            },
            "raw_root_storage_bytes": 4_714_322,
        },
    }


def _install_gate_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    def c_gate(_contract, _rows, *, stage_id, model_id):
        return {
            "status": "supported",
            "scientific_evidence_complete": True,
            "same_direction_counts": {"false_activation": 5},
            "claim_action": f"retain {model_id}/{stage_id} rule claim",
        }

    def a_gate(_contract, _rows, *, stage_id, model_id):
        return {
            "status": "supported",
            "scientific_evidence_complete": True,
            "primary_contrast": {
                "raw_paired_deltas": {
                    str(seed): float(index + 1)
                    for index, seed in enumerate(
                        (
                            1099057501,
                            1421875452,
                            1769977770,
                            959809858,
                            617806385,
                        )
                    )
                }
            },
            "threshold_gate": {"same_direction_count": 5},
            "claim_action": f"retain {model_id}/{stage_id} retrieval claim",
        }

    def d_gate(_contract, _rows, *, stage_id, model_id, arms):
        return {
            "status": "supported",
            "scientific_evidence_complete": True,
            "supported_treatments": ["no-memory"],
            "treatment_gates": {
                "no-memory": {
                    "six_step_discounted_utility_gate": {
                        "treatment_deltas": {
                            str(seed): 1.0
                            for seed in (
                                1099057501,
                                1421875452,
                                1769977770,
                                959809858,
                                617806385,
                            )
                        }
                    }
                }
            },
            "claim_action": (
                f"retain {model_id}/{stage_id} pulse claim for {tuple(arms)!r}"
            ),
        }

    def b_summary(_rows, *, stage_id, model_id, arms):
        return {
            "comparison_type": "descriptive_preregistered_architecture_arms",
            "selection_rule": "do not select a winner",
            "arms": {arm: {} for arm in arms},
            "binding": f"{model_id}/{stage_id}",
        }

    monkeypatch.setattr(lane_evidence, "_experiment_c_gate", c_gate)
    monkeypatch.setattr(lane_evidence, "_experiment_a_gate", a_gate)
    monkeypatch.setattr(lane_evidence, "_experiment_d_gate", d_gate)
    monkeypatch.setattr(lane_evidence, "_experiment_b_summary", b_summary)


def test_v25_uses_the_exact_v24_lane_stage_partition() -> None:
    v24 = load_pilot_contract(V24_CONTRACT_PATH)
    v25 = load_pilot_contract(V25_CONTRACT_PATH)

    assert len(v24.expand()) == 211
    assert len(v25.expand()) == 211
    assert core_evidence._stage_sets(v25) == core_evidence._stage_sets(v24)
    assert core_evidence._evidence_namespace(v25) == "current_v2/pilot-v2.5"


def test_v25_aggregate_keeps_matrix_but_uses_v25_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(V25_CONTRACT_PATH)
    _install_gate_fixtures(monkeypatch)

    aggregate = lane_evidence.aggregate_v24_evidence(
        contract,
        _rows(contract),
        denominator=_denominator(contract),
        release_controls=_release_controls(),
    )

    assert aggregate["schema_version"] == (
        lane_evidence.PILOT_V25_EVIDENCE_SCHEMA_VERSION
    )
    assert aggregate["evidence_namespace"] == "current_v2/pilot-v2.5"
    assert aggregate["contract_id"] == "finevo-pilot-v2.5"
    assert aggregate["contract_sha256"] == contract.canonical_hash
    assert aggregate["pilot_tag"] == "pilot-v2.5-science"
    assert aggregate["denominator"]["expected_count"] == 211
    assert aggregate["narrative"]["claim_boundary"].startswith("no V2.5 ")
    assert lane_evidence.aggregate_lane_separated_evidence is (
        lane_evidence.aggregate_v24_evidence
    )


def test_v25_parent_import_marker_accepts_injected_v25_verifier(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(V25_CONTRACT_PATH)
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
            "provider_calls": 0,
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
            "contract_id": "finevo-pilot-v2.5",
            "expected_git_commit": resolved_commit,
        }
    ]


def test_v25_package_copies_retry_and_overlay_contract_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(V25_CONTRACT_PATH)
    _install_gate_fixtures(monkeypatch)
    rows = _rows(contract)
    aggregate = lane_evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator=_denominator(contract),
        release_controls=_release_controls(),
    )

    manifest_path, checksums_path = lane_evidence._write_v24_package(
        tmp_path / "package",
        contract_path=V25_CONTRACT_PATH,
        contract=contract,
        rows=rows,
        aggregate=aggregate,
        common_commit="b" * 40,
    )

    package = manifest_path.parent
    copied_contract = package / "contract" / V25_CONTRACT_PATH.name
    assert load_pilot_contract(copied_contract).canonical_hash == (
        contract.canonical_hash
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == (
        lane_evidence.PILOT_V25_EVIDENCE_SCHEMA_VERSION
    )
    assert manifest["evidence_namespace"] == "current_v2/pilot-v2.5"
    assert manifest["contract_id"] == "finevo-pilot-v2.5"
    assert manifest["pilot_tag"] == "pilot-v2.5-science"
    assert manifest["retry_source_manifest"]["package_path"] == (
        "contract/pilot_v2_5_source_manifest.json"
    )
    assert manifest["base_contract"]["package_path"] == (
        "contract/pilot_v2_4.yaml"
    )
    assert set(manifest["published_files"]) >= {
        "contract/pilot_v2_5_overlay.yaml",
        "contract/pilot_v2_5_source_manifest.json",
        "contract/pilot_v2_4_parent_source_manifest.json",
        "contract/pilot_v2_4.yaml",
    }
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    assert {row["path"] for row in checksums["files"]} >= set(
        manifest["published_files"]
    )
