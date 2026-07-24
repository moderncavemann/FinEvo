from __future__ import annotations

import json
from pathlib import Path

import pytest

from verified_memory.pilot_contract import canonical_sha256, load_pilot_contract
from verified_memory import pilot_evidence
from verified_memory.pilot_evidence import (
    PilotEvidenceError,
    V2_NON_SCIENTIFIC_STAGES,
    V2_SCIENTIFIC_STAGES,
    _cross_model_summary,
    _narrative_gate,
    _stage_sets,
    _validate_branch_provider_journal_binding,
    _validate_memory_pulse_binding,
    build_pilot_evidence_package,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2.yaml"


def _write_empty_ledger(path: Path, contract_hash: str) -> None:
    event = {
        "event_index": 0,
        "event_type": "genesis",
        "created_at": "fixture",
        "previous_event_sha256": "0" * 64,
        "payload": {
            "contract_hash": contract_hash,
            "runs_sha256": canonical_sha256({}),
        },
    }
    event["event_sha256"] = canonical_sha256(event)
    ledger = {
        "schema_version": "finevo-pilot-run-ledger-v2",
        "contract_hash": contract_hash,
        "created_at": "fixture",
        "updated_at": "fixture",
        "runs": {},
        "events": [event],
    }
    ledger["ledger_sha256"] = canonical_sha256(ledger)
    path.write_text(
        json.dumps(ledger, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def test_v2_stage_partition_matches_contract_without_v1_aliases() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    non_scientific, scientific = _stage_sets(contract)

    assert non_scientific == V2_NON_SCIENTIFIC_STAGES
    assert scientific == V2_SCIENTIFIC_STAGES
    assert non_scientific | scientific == set(contract.stage_ids)
    assert "capability-preflight" not in non_scientific
    assert "cross-model-sentinels" not in scientific


def test_v2_missing_matrix_builds_explicit_no_go_package(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    raw = tmp_path / "raw"
    raw.mkdir()
    ledger = raw / "run_ledger.json"
    _write_empty_ledger(ledger, contract.canonical_hash)

    receipt = build_pilot_evidence_package(
        contract_path=CONTRACT_PATH,
        run_ledger_path=ledger,
        raw_root=raw,
        build_root=tmp_path / "evidence",
    )

    assert receipt.package_dir == (
        tmp_path / "evidence" / "current_v2" / "pilot-v2"
    )
    assert receipt.scientific_complete is False
    aggregate = json.loads(
        (receipt.package_dir / "aggregate.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (receipt.package_dir / "package_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    failures = json.loads(
        (receipt.package_dir / "failure_ledger.json").read_text(
            encoding="utf-8"
        )
    )

    expected = len(contract.expand())
    assert aggregate["evidence_namespace"] == "current_v2/pilot-v2"
    assert aggregate["denominator"]["expected_count"] == expected
    assert aggregate["denominator"]["status_counts"] == {"missing": expected}
    assert all(row["scientific_eligible"] is False for row in aggregate["rows"])
    assert len(failures["rows"]) == expected
    assert manifest["scientific_matrix_complete"] is False
    assert manifest["scientific_complete"] is False

    opus = aggregate["cross_model"]["opus48_no_go"]
    assert opus["registered_dispatch_cell_count"] == 0
    assert opus["observed_dispatch_row_count"] == 0
    assert opus["direction"] == "not-dispatched"
    assert opus["directional_micro_pilot_replication"] is False


def _narrative_row(
    *,
    seed: int,
    narrative_id: str,
    consumption_rate: float,
    labor_hours: float,
) -> dict[str, object]:
    contract = load_pilot_contract(CONTRACT_PATH)
    shared = {
        "checkpoint_hash": f"{seed:064x}",
        "prefix_hash": f"{seed + 1:064x}",
        "shock_schedule_hash": f"{seed + 2:064x}",
        "pre_generated_rng_hashes": [f"{seed + 3:064x}"],
        "rng_schedule_binding": {"schedule": seed},
        "shared_result_hash": f"{seed + 4:064x}",
        "fixture_hash": contract.stop_go["experiment_d"][
            "narrative_fixture_hash"
        ],
        "narrative_pulse_contract": dict(
            contract.stop_go["experiment_d"]["narrative_pulse_contract"]
        ),
        "branch_narrative_id": narrative_id,
    }
    return {
        "stage_id": "experiment-d",
        "model_id": "gpt52_main",
        "arm_id": "narrative-content",
        "narrative_id": narrative_id,
        "environment_seed": seed,
        "status": "complete",
        "scientific_eligible": True,
        "gate_evidence": {"causal_bindings": shared},
        "metrics": {
            "narrative": {
                "first_labor_hours": labor_hours,
                "first_consumption_rate": consumption_rate,
                "immediate_flow_utility": 1.0,
                "six_step_discounted_flow_utility": 10.0,
                "final_wealth": 5.0,
            }
        },
    }


def _matched_narrative_null_row(
    *,
    seed: int,
    arm_id: str,
) -> dict[str, object]:
    shared = {
        "checkpoint_hash": f"{seed:064x}",
        "prefix_hash": f"{seed + 1:064x}",
        "shock_schedule_hash": f"{seed + 2:064x}",
        "pre_generated_rng_hashes": [f"{seed + 3:064x}"],
        "rng_schedule_binding": {"schedule": seed},
        "shared_result_hash": f"{seed + 5:064x}",
        "error_common_start_equal": True,
        "error_common_start_hash": f"{seed + 6:064x}",
    }
    return {
        "stage_id": "experiment-d",
        "model_id": "gpt52_main",
        "arm_id": arm_id,
        "narrative_id": "none",
        "environment_seed": seed,
        "status": "complete",
        "scientific_eligible": True,
        "gate_evidence": {
            "matched_replay_equal": True,
            "causal_bindings": shared,
        },
        "metrics": {
            "continuation": {
                "focal": {
                    "first_step": {
                        "labor_hours": 8.0,
                        "consumption_rate": 0.5,
                        "immediate_flow_utility": 1.0,
                    },
                    "discounted_flow_utility_sum": 10.0,
                    "final_wealth": 5.0,
                }
            }
        },
    }


def test_v2_narrative_wrong_consumption_direction_is_no_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    rows: list[dict[str, object]] = []
    for seed in contract.seeds["sets"]["main"]:
        seed = int(seed)
        rows.extend(
            [
                _narrative_row(
                    seed=seed,
                    narrative_id="none",
                    consumption_rate=0.5,
                    labor_hours=8.0,
                ),
                _narrative_row(
                    seed=seed,
                    narrative_id="aligned",
                    consumption_rate=0.7,
                    labor_hours=24.0,
                ),
                _narrative_row(
                    seed=seed,
                    narrative_id="paraphrase",
                    consumption_rate=0.7,
                    labor_hours=24.0,
                ),
                _narrative_row(
                    seed=seed,
                    narrative_id="opposite",
                    consumption_rate=0.5,
                    labor_hours=8.0,
                ),
                _matched_narrative_null_row(seed=seed, arm_id="matched-a"),
                _matched_narrative_null_row(seed=seed, arm_id="matched-b"),
            ]
        )

    # This test isolates the registered directional gate. Causal-binding
    # tampering is covered independently below.
    monkeypatch.setattr(
        pilot_evidence,
        "_validate_causal_bindings",
        lambda value, **_: value,
    )

    gate = _narrative_gate(contract, rows)

    consumption = gate["aligned_vs_opposite_action_gates"][
        "consumption_rate"
    ]
    assert consumption["checks"]["exceeds_matched_null"] is True
    assert consumption["checks"]["exceeds_one_action_bin"] is True
    assert consumption["negative_direction_count"] == 0
    assert consumption["expected_direction_pass"] is False
    assert consumption["passes"] is False
    assert gate["semantic_response"] is False
    assert gate["status"] == "no-go"


def test_v2_seed_unsupported_model_without_matched_null_is_uncalibrated() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    seeds = [int(value) for value in contract.seeds["sets"]["cross-model"]]
    rows = []
    for seed in seeds:
        for arm_id, utility in (("full", 2.0), ("no-memory", 1.0)):
            rows.append(
                {
                    "stage_id": "cross-model-diagnostics",
                    "model_id": "gpt56_diagnostic",
                    "arm_id": arm_id,
                    "environment_seed": seed,
                    "status": "complete",
                    "scientific_eligible": True,
                    "failure": None,
                    "metrics": {
                        "utility": {
                            "shock_recovery_discounted": utility,
                        }
                    },
                }
            )
    capability = {
        "gpt56_diagnostic": {
            "ledger_status": "complete",
            "artifact_validated": True,
            "capability": {
                "pass": True,
                "preflight_go": True,
                "category_totals": {},
                "parse_failure_count": 0,
                "provider_failure_count": 0,
            },
        }
    }

    summary = _cross_model_summary(contract, rows, capability)[
        "gpt56_diagnostic"
    ]

    assert summary["direction"] == "positive"
    assert summary["paired_delta"]["direction_count"] == 3
    assert summary["matched_a_a_null_registered"] is False
    assert (
        summary["matched_null_resolution"]
        == "uncalibrated-diagnostic-no-registered-matched-a-a-null"
    )
    assert summary["directional_micro_pilot_replication"] is False
    assert "uncalibrated diagnostic only" in summary["claim_boundary"]


def test_v2_memory_pulse_duration_tamper_is_rejected() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    experiment_d = contract.stop_go["experiment_d"]
    pulse = {
        "schema_version": "finevo-pilot-d-memory-pulse-binding-v1",
        "kind": "no-memory",
        "checkpoint_hash": "1" * 64,
        "focal_agent_id": 0,
        "decision_t": 6,
        "pulse_only": True,
        "duration_decisions": 2,
        "original_memory_hash": "2" * 64,
        "treated_memory_hash": "3" * 64,
        "shuffle_binding": None,
        "wrong_context_source_agent_id": None,
    }

    with pytest.raises(PilotEvidenceError, match="frozen memory pulse"):
        _validate_memory_pulse_binding(
            pulse,
            name="tampered pulse",
            treatment="no-memory",
            checkpoint_hash="1" * 64,
            pulse_contract=experiment_d["memory_pulse_contract"],
            shuffle_policy=experiment_d["shuffle_policy"],
        )


def test_v2_provider_journal_denominator_tamper_is_rejected() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    binding = {
        "enabled": True,
        "path": "provider_calls.jsonl",
        "file_sha256": "1" * 64,
        "journal_sha256": "2" * 64,
        "run_id": "run",
        "contract_hash": contract.canonical_hash,
        "event_count": 47,
        "completion_event_count": 24,
        "parse_disposition_event_count": 24,
        "terminal_dispositions_verified": True,
    }

    with pytest.raises(PilotEvidenceError, match="24-call branch journal"):
        _validate_branch_provider_journal_binding(
            binding,
            name="tampered journal",
            contract_hash=contract.canonical_hash,
        )
