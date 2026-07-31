from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_evidence import (
    PilotEvidenceError,
    _cross_model_summary,
    _stage_sets,
)
from verified_memory.pilot_orchestrator import GitProvenance, PilotRunLedger
from verified_memory import pilot_v2112_evidence as evidence


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_2.yaml"


def _draft_contract():
    return load_pilot_contract(CONTRACT_PATH)


def _frozen_test_contract():
    # The production freeze rewrites the tracked contract and hash.  Unit tests
    # need only exercise the downstream all-terminal publication boundary.
    return replace(_draft_contract(), status="frozen")


def test_v2112_stage_partition_is_release_specific() -> None:
    non_scientific, scientific = _stage_sets(_draft_contract())
    assert non_scientific == {
        "parent-import",
        "capability-gate",
        "long-context-preflight",
    }
    assert scientific == {
        "experiment-a",
        "experiment-b",
        "experiment-c",
        "experiment-d",
        "cross-model",
    }


def test_v2112_draft_publish_fails_before_raw_access(tmp_path: Path) -> None:
    with pytest.raises(PilotEvidenceError, match="requires the frozen contract"):
        evidence.build_pilot_v2112_evidence_package(
            contract_path=CONTRACT_PATH,
            run_ledger_path=tmp_path / "missing" / "run_ledger.json",
            raw_root=tmp_path / "missing",
            build_root=tmp_path / "evidence",
        )


def test_v2112_nonterminal_ledger_is_not_publishable(tmp_path: Path) -> None:
    contract = _frozen_test_contract()
    raw = tmp_path / "raw"
    ledger_path = raw / "run_ledger.json"
    ledger = PilotRunLedger(
        ledger_path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())
    value = json.loads(ledger_path.read_text(encoding="utf-8"))
    with pytest.raises(PilotEvidenceError, match="all-terminal ITT denominator"):
        evidence._normalize_v2112_ledger(
            contract,
            value,
            raw_root=raw,
            expected_commit="a" * 40,
        )


def test_v2112_cross_model_direction_requires_capability_and_three_pairs() -> None:
    contract = _draft_contract()
    seeds = tuple(int(seed) for seed in contract.seeds["sets"]["cross-model"])
    rows = []
    for seed in seeds:
        for arm, utility in (("full", 2.0), ("no-memory", 1.0)):
            rows.append(
                {
                    "stage_id": "cross-model",
                    "model_id": "gpt56_diagnostic",
                    "arm_id": arm,
                    "environment_seed": seed,
                    "status": "complete",
                    "scientific_eligible": True,
                    "failure": None,
                    "metrics": {
                        "utility": {"shock_recovery_discounted": utility},
                        "memory": {"proposal_parse_status_counts": {}},
                    },
                }
            )
    capability = {
        "gpt56_diagnostic": {
            "ledger_status": "complete",
            "artifact_validated": True,
            "capability": {
                "preflight_go": True,
                "capability": {
                    "capability_pass": True,
                    "parse_failure_count": 0,
                    "provider_failure_count": 0,
                    "category_totals": {},
                },
            },
        }
    }
    result = _cross_model_summary(contract, rows, capability)
    assert result["gpt56_diagnostic"]["usable_paired_seeds"] == list(seeds)
    assert result["gpt56_diagnostic"]["direction"] == "positive"
    assert result["gpt56_diagnostic"]["directional_micro_pilot_replication"] is True
    assert "backbone" not in result["gpt56_diagnostic"]["claim_boundary"]


def test_provider_free_terminal_failure_fixture_preserves_all_itt_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _frozen_test_contract()
    raw = tmp_path / "raw"
    ledger_path = raw / "run_ledger.json"
    ledger = PilotRunLedger(
        ledger_path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    specs = contract.expand()
    ledger.register(specs)
    failure = {
        "error_type": "FixtureIntegrityStop",
        "message": "provider-free terminal denominator fixture",
        "provider_calls": 0,
    }
    ledger.finalize_many(
        [
            {
                "run_id": spec.run_id,
                "status": "integrity-stopped",
                "artifact": None,
                "failure": failure,
            }
            for spec in specs
        ]
    )
    (raw / "budget_ledger.json").write_text("{}\n", encoding="utf-8")

    paid = GitProvenance(
        git_tag="pilot-v2.11.2-science",
        head_commit="a" * 40,
        tag_commit="a" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
        release_attestation={},
    )
    monkeypatch.setattr(evidence, "load_pilot_contract", lambda _: contract)
    monkeypatch.setattr(
        evidence,
        "_validate_release",
        lambda *_args, **_kwargs: (
            {"pass": True, "file_sha256": "b" * 64},
            "a" * 40,
            paid,
        ),
    )
    monkeypatch.setattr(
        evidence,
        "_validate_stage_receipts",
        lambda *_args, **_kwargs: {
            "schema_version": evidence.V2112_STAGE_RECEIPTS_SCHEMA_VERSION,
            "all_terminal": True,
            "receipts": {},
        },
    )
    monkeypatch.setattr(
        evidence,
        "_validate_post_gate",
        lambda *_args, **_kwargs: {
            "available": False,
            "go": False,
            "reason": "fixture preflight no-go",
        },
    )
    monkeypatch.setattr(
        evidence,
        "_validate_budget",
        lambda *_args, **_kwargs: {
            "schema_version": evidence.V2112_BUDGET_RECEIPT_SCHEMA_VERSION,
            "pass": True,
            "raw_root_storage_bytes": 0,
        },
    )
    monkeypatch.setattr(
        evidence,
        "_validated_experiment_c_sensitivity",
        lambda *_args, **_kwargs: (
            None,
            {"pass": False, "available": False},
        ),
    )

    package = evidence.build_pilot_v2112_evidence_package(
        contract_path=CONTRACT_PATH,
        run_ledger_path=ledger_path,
        raw_root=raw,
        build_root=tmp_path / "evidence",
    )
    assert package.scientific_complete is False
    failures = json.loads(
        (package.package_dir / "failure_ledger.json").read_text(encoding="utf-8")
    )
    assert len(failures["rows"]) == 136
    assert failures["denominator"]["all_rows_terminal"] is True
    assert failures["denominator"]["itt_failures_retained"] == 136
    manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
    assert manifest["scientific_matrix_complete"] is False
    assert any(
        "V2.11.1 failed-preflight" in item for item in manifest["excluded_sources"]
    )
    aggregate = json.loads(
        (package.package_dir / "aggregate.json").read_text(encoding="utf-8")
    )
    assert aggregate["release_controls"]["historical_import_boundary"] == {
        "source_contract": "finevo-pilot-v2.11.1",
        "failed_preflight_calls_retained_for_budget_audit": 64,
        "failed_preflight_samples_admitted": 0,
        "failed_preflight_checkpoints_admitted": 0,
        "failed_preflight_p95_authorities_admitted": 0,
        "treatment_effect_cells_imported": 0,
        "scientific_evidence": False,
    }
