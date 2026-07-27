from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_evidence import _validate_terminal_payload_marker
import verified_memory.pilot_v24_evidence as evidence
import verified_memory.pilot_v24_parent_import as parent_import


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_4.yaml"


def _rows(contract) -> list[dict[str, Any]]:
    scientific_stages = {
        "stage0-calibration",
        "local-experiment-c",
        "local-experiment-a",
        "local-experiment-d",
        "local-experiment-b",
        "experiment-c",
        "experiment-a",
        "experiment-d",
        "experiment-b",
    }
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


def _denominator(contract, *, status_counts=None) -> dict[str, Any]:
    count = len(contract.expand())
    return {
        "expected_count": count,
        "observed_ledger_count": count,
        "all_rows_present": True,
        "all_rows_terminal": True,
        "status_counts": status_counts or {"complete": count},
        "all_completed_artifacts_validated": True,
        "pass": True,
    }


def _release_controls(*, passed: bool = True) -> dict[str, Any]:
    return {
        "pass": passed,
        "release_attestation": {"pass": passed},
        "stage0_selection": {"pass": passed},
        "budget_ledger": {
            "pass": passed,
            "actual_totals": {
                "cost_usd": 3.212770875,
                "completions": 184,
                "storage_bytes": 4_196_087,
            },
            "actual_stage_cost_usd": {
                "parent_v23": 3.212770875,
                "local": 0.0,
                "hosted_confirmatory": 0.0,
                "manual_reserve": 0.0,
            },
            "raw_root_storage_bytes": 4_196_087,
        },
    }


def _install_gate_fixtures(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str, str, tuple[str, ...] | None]]:
    calls: list[tuple[str, str, str, tuple[str, ...] | None]] = []

    def c_gate(_contract, _rows, *, stage_id, model_id):
        calls.append(("experiment-c", stage_id, model_id, None))
        return {
            "status": "supported",
            "scientific_evidence_complete": True,
            "same_direction_counts": {
                "false_activation": (4 if model_id == "llama33_local_controlled" else 5)
            },
            "claim_action": f"retain {model_id} rule-reliability claim",
        }

    def a_gate(_contract, _rows, *, stage_id, model_id):
        calls.append(("experiment-a", stage_id, model_id, None))
        return {
            "status": "supported",
            "scientific_evidence_complete": True,
            "primary_contrast": {
                "raw_paired_deltas": {
                    str(seed): float(index + 1)
                    for index, seed in enumerate(
                        (1099057501, 1421875452, 1769977770, 959809858, 617806385)
                    )
                }
            },
            "threshold_gate": {
                "same_direction_count": (
                    4 if model_id == "llama33_local_controlled" else 5
                )
            },
            "claim_action": f"retain {model_id} retrieval-effect claim",
        }

    def d_gate(_contract, _rows, *, stage_id, model_id, arms):
        calls.append(("experiment-d", stage_id, model_id, tuple(arms)))
        return {
            "status": "supported",
            "scientific_evidence_complete": True,
            "supported_treatments": ["no-memory"],
            "treatment_gates": {
                "no-memory": {
                    "six_step_discounted_utility_gate": {
                        "treatment_deltas": {
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
                    }
                }
            },
            "claim_action": f"retain {model_id} named pulse effect",
        }

    def b_summary(_rows, *, stage_id, model_id, arms):
        calls.append(("experiment-b", stage_id, model_id, tuple(arms)))
        return {
            "comparison_type": "descriptive_preregistered_architecture_arms",
            "selection_rule": "do not select a winner",
            "arms": {arm: {} for arm in arms},
        }

    monkeypatch.setattr(evidence, "_experiment_c_gate", c_gate)
    monkeypatch.setattr(evidence, "_experiment_a_gate", a_gate)
    monkeypatch.setattr(evidence, "_experiment_d_gate", d_gate)
    monkeypatch.setattr(evidence, "_experiment_b_summary", b_summary)
    return calls


def test_v24_aggregate_keeps_lane_gates_and_directions_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract)
    calls = _install_gate_fixtures(monkeypatch)

    aggregate = evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator=_denominator(contract),
        release_controls=_release_controls(),
    )

    assert aggregate["fixed_matrix_order"] == [
        "experiment-c",
        "experiment-a",
        "experiment-d",
        "experiment-b",
    ]
    assert aggregate["scientific_complete"] is True
    assert aggregate["publication_status"] == "complete"
    assert aggregate["narrative"] == {
        "status": "deferred-unregistered",
        "registered_cells": 0,
        "claim_boundary": "no V2.4 narrative or real-news-understanding claim",
    }
    assert aggregate["cross_lane_policy"]["direction_counts_merged"] is False
    assert aggregate["cross_lane_policy"]["effect_estimates_pooled"] is False

    local = aggregate["lanes"]["local"]
    gpt = aggregate["lanes"]["gpt52"]
    assert local["paired_matrix_complete"] is True
    assert gpt["paired_matrix_complete"] is True
    assert all(
        gate["complete_pair_count"] == 5 for gate in local["paired_seed_gates"].values()
    )
    assert all(
        gate["complete_pair_count"] == 5 for gate in gpt["paired_seed_gates"].values()
    )
    assert local["gates"]["experiment-c"]["same_direction_counts"] == {
        "false_activation": 4
    }
    assert gpt["gates"]["experiment-c"]["same_direction_counts"] == {
        "false_activation": 5
    }
    assert "same_direction_counts" not in aggregate
    assert "directions" not in aggregate["cross_lane_policy"]
    comparison = {
        row["stage"]: row
        for row in aggregate["cross_lane_mechanism_comparison"]["rows"]
    }
    assert comparison["experiment-c"]["classification"] == (
        "same-direction-in-two-backbone-micro-pilots"
    )
    assert comparison["experiment-a"]["classification"] == (
        "same-direction-in-two-backbone-micro-pilots"
    )
    assert comparison["experiment-d"]["classification"] == (
        "same-direction-in-two-backbone-micro-pilots"
    )
    assert comparison["experiment-d"]["local_only_registered_treatments"] == [
        "shuffled-episodic"
    ]
    assert comparison["experiment-d"]["common_direction_qualified_treatments"] == [
        "no-memory"
    ]
    assert comparison["experiment-b"]["classification"] == "inconclusive"
    assert (
        aggregate["cross_lane_mechanism_comparison"]["direction_counts_merged"] is False
    )

    local_d_call = next(
        call
        for call in calls
        if call[:3]
        == (
            "experiment-d",
            "local-experiment-d",
            "llama33_local_controlled",
        )
    )
    gpt_d_call = next(
        call
        for call in calls
        if call[:3] == ("experiment-d", "experiment-d", "gpt52_main")
    )
    assert local_d_call[3] == (
        "matched-a",
        "matched-b",
        "no-memory",
        "shuffled-episodic",
        "wrong-context",
        "error-verified",
        "error-unverified",
    )
    assert gpt_d_call[3] == (
        "matched-a",
        "matched-b",
        "no-memory",
        "wrong-context",
        "error-verified",
        "error-unverified",
    )


def test_v24_aggregate_preserves_failures_and_narrows_only_failed_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows = _rows(contract)
    _install_gate_fixtures(monkeypatch)
    failed_seeds = set(contract.seeds["sets"]["main"][:2])
    failed = 0
    for row in rows:
        if (
            row["stage_id"] == "local-experiment-c"
            and row["arm_id"] == "verified-error-forced"
            and row["environment_seed"] in failed_seeds
        ):
            row["status"] = "failed"
            row["failure"] = {
                "error_type": "FixtureFailure",
                "message": "registered terminal failure",
            }
            row["scientific_eligible"] = False
            failed += 1
    count = len(contract.expand())
    denominator = _denominator(
        contract,
        status_counts={"complete": count - failed, "failed": failed},
    )

    aggregate = evidence.aggregate_v24_evidence(
        contract,
        rows,
        denominator=denominator,
        release_controls=_release_controls(),
    )

    local_c = aggregate["lanes"]["local"]["paired_seed_gates"]["experiment-c"]
    gpt_c = aggregate["lanes"]["gpt52"]["paired_seed_gates"]["experiment-c"]
    assert local_c["complete_pair_count"] == 3
    assert local_c["pass"] is False
    assert gpt_c["complete_pair_count"] == 5
    assert gpt_c["pass"] is True
    assert aggregate["denominator"]["status_counts"]["failed"] == 2
    assert aggregate["denominator"]["pass"] is True
    assert aggregate["scientific_matrix_complete"] is False
    assert aggregate["scientific_complete"] is False
    assert aggregate["publication_status"] == "complete-with-no-go"
    assert {item["scope"] for item in aggregate["claim_narrowing"]} >= {
        "local/experiment-c",
        "narrative",
        "cross-lane",
    }
    assert not any(
        item["scope"] == "gpt52/experiment-c" for item in aggregate["claim_narrowing"]
    )
    local_claim = next(
        claim
        for claim in aggregate["claims"]
        if claim["lane"] == "local" and claim["artifact"].endswith("/experiment-c")
    )
    gpt_claim = next(
        claim
        for claim in aggregate["claims"]
        if claim["lane"] == "gpt52" and claim["artifact"].endswith("/experiment-c")
    )
    assert local_claim["artifact"] != gpt_claim["artifact"]
    cross_c = next(
        row
        for row in aggregate["cross_lane_mechanism_comparison"]["rows"]
        if row["stage"] == "experiment-c"
    )
    assert cross_c["classification"] == "inconclusive"


def test_v24_claim_map_marks_narrative_deferred_and_backbone_claim_prohibited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    _install_gate_fixtures(monkeypatch)
    aggregate = evidence.aggregate_v24_evidence(
        contract,
        _rows(contract),
        denominator=_denominator(contract),
        release_controls=_release_controls(),
    )

    narrative = next(
        claim
        for claim in aggregate["claims"]
        if claim["status"] == "deferred-unregistered"
    )
    backbone = next(
        claim for claim in aggregate["claims"] if claim["status"] == "prohibited"
    )
    assert narrative["lane"] == "not-applicable"
    assert "real-news" in narrative["boundary"]
    assert backbone["lane"] == "cross-lane"
    assert "never pool" in backbone["boundary"]


def test_parent_import_marker_uses_terminal_tag_binding_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(stage="parent-import")[0]
    resolved_commit = "b" * 40
    receipt_hash = "c" * 64
    calls = []

    def fake_verify(
        receipt_path,
        *,
        repo_root,
        contract: Any,
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

    monkeypatch.setattr(
        parent_import,
        "verify_v24_parent_import_receipt",
        fake_verify,
    )
    payload = {
        "metrics": {},
        "gate_evidence": {
            "receipt": str(tmp_path / "parent-import" / "parent_import_receipt.json"),
            "receipt_content_sha256": receipt_hash,
            "provider_calls": 0,
            "scientific_evidence": False,
        },
        "provider_calls": 0,
    }

    _validate_terminal_payload_marker(
        contract,
        spec.to_dict(),
        payload,
        raw_root=tmp_path,
        resolved_git_commit=resolved_commit,
    )

    assert calls == [
        {
            "receipt_path": payload["gate_evidence"]["receipt"],
            "repo_root": ROOT,
            "contract_id": contract.contract_id,
            "expected_git_commit": resolved_commit,
        }
    ]
    assert contract.implementation["required_git_commit"] is None
    with pytest.raises(
        evidence.PilotEvidenceError,
        match="exact zero-call marker",
    ):
        _validate_terminal_payload_marker(
            contract,
            spec.to_dict(),
            payload,
            raw_root=tmp_path,
        )


def test_atomic_no_replace_rejects_destination_injection(
    tmp_path: Path,
) -> None:
    source = tmp_path / ".pilot-v2.4-build"
    target = tmp_path / "pilot-v2.4"
    source.mkdir()
    (source / "package_manifest.json").write_text(
        '{"candidate":true}\n',
        encoding="utf-8",
    )
    target.mkdir()
    sentinel = target / "injected.txt"
    sentinel.write_text("must survive\n", encoding="utf-8")

    with pytest.raises(
        evidence.PilotEvidenceError,
        match="refusing to overwrite",
    ):
        evidence._atomic_install_directory_no_replace(source, target)

    assert sentinel.read_text(encoding="utf-8") == "must survive\n"
    assert (source / "package_manifest.json").is_file()
    assert not (target / "package_manifest.json").exists()


def test_atomic_no_replace_installs_new_package_once(
    tmp_path: Path,
) -> None:
    source = tmp_path / ".pilot-v2.4-build"
    target = tmp_path / "pilot-v2.4"
    source.mkdir()
    (source / "package_manifest.json").write_text(
        '{"candidate":true}\n',
        encoding="utf-8",
    )

    evidence._atomic_install_directory_no_replace(source, target)

    assert not source.exists()
    assert (target / "package_manifest.json").read_text(
        encoding="utf-8"
    ) == '{"candidate":true}\n'


def test_incomplete_denominator_cannot_enter_immutable_publisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    _install_gate_fixtures(monkeypatch)
    count = len(contract.expand())
    denominator = _denominator(
        contract,
        status_counts={"complete": count - 1, "running": 1},
    )
    denominator["all_rows_terminal"] = False
    denominator["pass"] = False

    aggregate = evidence.aggregate_v24_evidence(
        contract,
        _rows(contract),
        denominator=denominator,
        release_controls=_release_controls(),
    )

    assert aggregate["publication_status"] == "incomplete"
    assert aggregate["scientific_complete"] is False
    with pytest.raises(
        evidence.PilotEvidenceError,
        match="all 211 ITT cells",
    ):
        evidence._require_publishable_terminal_denominator(aggregate)
    narrowing = next(
        row
        for row in aggregate["claim_narrowing"]
        if row["scope"] == "full-denominator"
    )
    assert "do not publish" in narrowing["required_wording"]


def test_cross_lane_d_compares_only_common_qualified_treatments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    _install_gate_fixtures(monkeypatch)
    scenario = {"name": "no-common"}

    def d_gate(_contract, _rows, *, stage_id, model_id, arms):
        if scenario["name"] == "no-common":
            treatment = (
                "shuffled-episodic"
                if model_id == "llama33_local_controlled"
                else "no-memory"
            )
            delta = 1.0
        else:
            treatment = "no-memory"
            delta = 1.0 if model_id == "llama33_local_controlled" else -1.0
        return {
            "status": "supported",
            "scientific_evidence_complete": True,
            "supported_treatments": [treatment],
            "treatment_gates": {
                treatment: {
                    "six_step_discounted_utility_gate": {
                        "treatment_deltas": {
                            str(seed): delta for seed in contract.seeds["sets"]["main"]
                        }
                    }
                }
            },
            "claim_action": f"retain {model_id} named pulse effect",
        }

    monkeypatch.setattr(evidence, "_experiment_d_gate", d_gate)

    no_common = evidence.aggregate_v24_evidence(
        contract,
        _rows(contract),
        denominator=_denominator(contract),
        release_controls=_release_controls(),
    )
    no_common_d = next(
        row
        for row in no_common["cross_lane_mechanism_comparison"]["rows"]
        if row["stage"] == "experiment-d"
    )
    assert no_common_d["classification"] == "inconclusive"
    assert no_common_d["common_direction_qualified_treatments"] == []
    assert no_common_d["local_only_registered_treatments"] == ["shuffled-episodic"]

    scenario["name"] = "opposite-common"
    opposite = evidence.aggregate_v24_evidence(
        contract,
        _rows(contract),
        denominator=_denominator(contract),
        release_controls=_release_controls(),
    )
    opposite_d = next(
        row
        for row in opposite["cross_lane_mechanism_comparison"]["rows"]
        if row["stage"] == "experiment-d"
    )
    assert opposite_d["classification"] == "backbone-interaction"
    assert opposite_d["common_direction_qualified_treatments"] == ["no-memory"]
