from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.render_pilot_v2117_contract import build_contract
from verified_memory import pilot_contract as pilot_contract_module
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory import pilot_v2117_continuation as continuation
from verified_memory.pilot_budget import PilotBudgetLedger
from verified_memory.pilot_contract import (
    PilotContract,
    canonical_sha256,
    load_pilot_contract,
    science_design_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
V2115_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_5.yaml"
EXPECTED_MAPPING_SHA256 = (
    "88aef768d311653c8335f7ad769400c84e0c0430c9c82183611f87d0f6906fcd"
)


def _contract(monkeypatch: pytest.MonkeyPatch) -> PilotContract:
    document = build_contract(ROOT, status="draft")
    monkeypatch.setattr(
        pilot_contract_module,
        "PILOT_CONTRACT_V2_11_7_SCIENCE_DESIGN_SHA256",
        science_design_sha256(document),
    )
    return PilotContract.from_dict(document)


def _exact_v2115_current_budget_rows() -> dict[str, Any]:
    runs: dict[str, Any] = {}
    for index in range(47):
        runs[f"hosted-{index}"] = {
            "stage_bucket": "hosted_v2115",
            "actual": {
                "cost_usd": 43.1214245 if index == 0 else 0.0,
                "completions": 2_436 if index == 0 else 0,
                "storage_bytes": 47_975_380 if index == 0 else 0,
            },
        }
    for index, storage_bytes in enumerate((41_006, 40_439, 82_708)):
        runs[f"operational-{index}"] = {
            "stage_bucket": "parent_v2114",
            "actual": {
                "cost_usd": 0.0,
                "completions": 0,
                "storage_bytes": storage_bytes,
            },
        }
    return {"runs": runs}


def _run_snapshot_before_science(contract: PilotContract) -> dict[str, Any]:
    parent_id = contract.expand(stage="parent-import")[0].run_id
    return {
        "events": [{"event_sha256": "1" * 64}],
        "ledger_sha256": "2" * 64,
        "runs": {
            spec.run_id: {
                "spec": spec.to_dict(),
                "status": "complete" if spec.run_id == parent_id else "scheduled",
            }
            for spec in contract.expand()
        },
    }


def _budget_snapshot_before_science(contract: PilotContract) -> dict[str, Any]:
    spec = contract.expand(stage="parent-import")[0]
    return {
        "events": [{"event_sha256": "3" * 64}],
        "ledger_sha256": "4" * 64,
        "runs": {
            spec.run_id: {
                "status": "complete",
                "stage_bucket": "parent_v2116",
                "reservation": orchestrator._v2117_parent_import_projection(
                    spec
                ).to_dict(),
                "actual": {
                    "cost_usd": 0.0,
                    "completions": 0,
                    "storage_bytes": 1,
                },
            }
        },
    }


def test_canonical_mapping_binds_86_unique_source_child_and_logical_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _contract(monkeypatch)
    authority = load_pilot_contract(V2115_CONTRACT_PATH)

    mapping = continuation._canonical_remaining_cell_mapping(child, authority)
    rows = mapping["rows"]

    assert mapping["row_count"] == len(rows) == 86
    assert mapping["mapping_sha256"] == EXPECTED_MAPPING_SHA256
    assert mapping["mapping_sha256"] == canonical_sha256(rows)
    assert len({row["source_run_id"] for row in rows}) == 86
    assert len({row["child_run_id"] for row in rows}) == 86
    assert len({row["logical_cell_sha256"] for row in rows}) == 86
    assert all(
        row["source_run_id"].startswith("finevo-pilot-v2.11.5--")
        and row["child_run_id"].startswith("finevo-pilot-v2.11.7--")
        and canonical_sha256(row["normalized_spec"])
        == row["logical_cell_sha256"]
        for row in rows
    )


def test_canonical_mapping_rejects_one_tampered_normalized_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _contract(monkeypatch)
    authority = load_pilot_contract(V2115_CONTRACT_PATH)
    original = continuation._normalize_parent_spec
    changed = False

    def tampered(spec: Any) -> dict[str, Any]:
        nonlocal changed
        value = original(spec)
        if not changed:
            value["utility_profile_id"] = "tampered-profile"
            changed = True
        return value

    monkeypatch.setattr(continuation, "_normalize_parent_spec", tampered)
    with pytest.raises(
        continuation.PilotV2117ContinuationError,
        match="child normalized spec differs",
    ):
        continuation._canonical_remaining_cell_mapping(child, authority)


def test_source_manifest_requires_both_real_roots_and_rejects_swapped_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(monkeypatch)
    missing = tmp_path / "missing-v2116"
    with pytest.raises(
        continuation.PilotV2117ContinuationError,
        match="V2.11.6 failed repository is unavailable",
    ):
        continuation.build_v2117_source_manifest(
            contract=contract,
            repo_root=ROOT,
            failed_repo_root=missing,
            authority_repo_root=tmp_path,
        )

    failed = tmp_path / "v2116"
    authority = tmp_path / "v2115"
    failed.mkdir()
    authority.mkdir()

    def exact_failed_role(root: Path) -> dict[str, str]:
        if root != failed:
            raise continuation.PilotV2117ContinuationError(
                "V2.11.6 failed release role drifted"
            )
        return {}

    monkeypatch.setattr(continuation, "_verify_failed_git", exact_failed_role)
    with pytest.raises(
        continuation.PilotV2117ContinuationError,
        match="failed release role drifted",
    ):
        continuation.build_v2117_source_manifest(
            contract=contract,
            repo_root=ROOT,
            failed_repo_root=authority,
            authority_repo_root=failed,
        )


def test_v2115_current_debit_counts_all_50_rows_not_only_47_hosted() -> None:
    snapshot = _exact_v2115_current_budget_rows()

    decomposition = continuation._v2115_current_actual_decomposition(snapshot)

    assert decomposition["hosted_v2115"] == {
        "row_count": 47,
        "cost_usd": 43.1214245,
        "hosted_completions": 2_436,
        "storage_bytes": 47_975_380,
    }
    assert decomposition["operational_parent_v2114"] == {
        "row_count": 3,
        "cost_usd": 0.0,
        "hosted_completions": 0,
        "storage_bytes": 164_153,
    }
    assert decomposition["all_current"] == {
        "row_count": 50,
        "cost_usd": 43.1214245,
        "hosted_completions": 2_436,
        "storage_bytes": 48_139_533,
    }

    hosted_only = deepcopy(snapshot)
    hosted_only["runs"] = {
        run_id: row
        for run_id, row in hosted_only["runs"].items()
        if row["stage_bucket"] == "hosted_v2115"
    }
    with pytest.raises(
        continuation.PilotV2117ContinuationError,
        match="actual debit decomposition drifted",
    ):
        continuation._v2115_current_actual_decomposition(hosted_only)


def test_v2116_terminal_no_go_is_bound_and_cannot_be_reclassified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(monkeypatch)
    no_go = continuation._expected_failed_release_no_go()
    assert no_go == continuation._json_copy(
        contract.v2117_recovery_boundary["failed_release_no_go"]
    )
    assert no_go["raw_inventory"] == {
        "root": "experiment_results/pilot-v2.11.6/raw",
        "canonicalization": "json-sort-keys-compact-utf8-v1",
        "excluded_operational_paths": [".real-stage-execution.lock"],
        "file_count": 5,
        "storage_bytes": 215_033,
        "inventory_sha256": (
            "0dbf0a293b9b2c00c642aa8d7724eb7b585e0b23ad327504e15edaf63e5e234d"
        ),
    }
    assert no_go["run_ledger"]["status_counts"] == {"integrity-stopped": 87}
    assert no_go["acceptance_receipt_present"] is False
    assert no_go["science_reservations"] == 0
    assert no_go["provider_construction"] is False
    assert no_go["provider_calls"] == 0
    assert no_go["resume_forbidden"] is True
    assert no_go["failure_reclassification_forbidden"] is True

    tampered = deepcopy(no_go)
    tampered["run_ledger"]["status_counts"] = {"complete": 87}
    monkeypatch.setattr(
        continuation,
        "validate_v2117_source_manifest",
        lambda **_kwargs: {"fixture": True},
    )
    monkeypatch.setattr(
        continuation, "_expected_failed_release_no_go", lambda: tampered
    )
    with pytest.raises(
        continuation.PilotV2117ContinuationError,
        match="failed-release contract boundary drifted",
    ):
        continuation._failed_state(
            contract=contract,
            repo_root=tmp_path,
            failed_repo_root=tmp_path,
            authority_repo_root=tmp_path,
        )


def test_parent_debit_and_projection_are_exactly_zero_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(monkeypatch)
    spec = contract.expand(stage="parent-import")[0]

    debit = continuation.parent_budget_debit_for_v2117(contract)
    projection = orchestrator._v2117_parent_import_projection(spec)

    assert debit.to_dict() == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": continuation.V2116_CONTRACT_SHA256,
        "parent_run_ledger_sha256": continuation.V2116_RUN_LEDGER_SHA256,
        "parent_budget_ledger_sha256": continuation.V2116_BUDGET_LEDGER_SHA256,
        "stage_bucket": "parent_v2116",
        "cost_usd": 63.1196450625,
        "hosted_completions": 3_440,
        "storage_bytes": 270_189_931,
        "record_sha256": (
            "1118a572ce7fe713f0428bbddd155808e20db6ac7bd845f8a180145c50f7b46a"
        ),
    }
    assert projection.run_id == spec.run_id
    assert projection.stage_bucket == "parent_v2116"
    assert (projection.cost_usd, projection.completions) == (0.0, 0)
    assert projection.basis["provider_construction"] is False
    assert projection.basis["provider_calls"] == 0
    assert projection.basis["failed_v2116_terminal_rows_bound"] == 87
    assert projection.basis["mapped_v2115_scheduled_cells"] == 86
    assert projection.basis["imported_parent_terminal_cells_as_child_rows"] == 0


def test_parent_import_failure_is_atomic_and_terminalizes_the_full_itt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(monkeypatch)
    specs = contract.expand(stage="parent-import")
    spec = specs[0]
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    run_ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    run_ledger.register(contract.expand())
    budget_ledger = PilotBudgetLedger(
        raw_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=continuation.parent_budget_debit_for_v2117(contract),
    )

    def fail_import(**_kwargs: Any) -> Any:
        raise continuation.PilotV2117ContinuationError("fixture lineage mismatch")

    def write_failure_receipt(
        _contract: PilotContract,
        stage_id: str,
        **_kwargs: Any,
    ) -> Path:
        path = raw_root / stage_id / "stage_receipt.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        return path

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("provider construction was attempted")

    monkeypatch.setattr(
        orchestrator, "build_v2117_parent_import_receipt", fail_import
    )
    monkeypatch.setattr(orchestrator, "_write_stage_receipt", write_failure_receipt)
    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden_provider)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden_provider)
    monkeypatch.setattr(orchestrator, "MultiModelLLM", forbidden_provider)
    monkeypatch.setattr(
        orchestrator, "validate_live_provider_catalog", forbidden_provider
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="V2.11.7 parent import failed",
    ):
        orchestrator._execute_v2117_parent_import_stage(
            contract,
            specs,
            raw_root=raw_root,
            repo_root=tmp_path,
            failed_repo_root=tmp_path / "failed-v2116",
            authority_repo_root=tmp_path / "authority-v2115",
            paid=SimpleNamespace(),
            run_ledger=run_ledger,
            budget_ledger=budget_ledger,
        )

    run_snapshot = run_ledger.snapshot()
    assert Counter(row["status"] for row in run_snapshot["runs"].values()) == {
        "integrity-stopped": 87
    }
    assert all(
        row["artifact"] is None
        and row["failure"]["provider_calls"] == 0
        and row["failure"]["provider_construction"] is False
        for row in run_snapshot["runs"].values()
    )
    budget_snapshot = budget_ledger.snapshot()
    assert set(budget_snapshot["runs"]) == {spec.run_id}
    budget_row = budget_snapshot["runs"][spec.run_id]
    assert budget_row["status"] == "integrity-stopped"
    assert budget_row["actual"]["cost_usd"] == 0.0
    assert budget_row["actual"]["completions"] == 0
    assert budget_row["failure"]["provider_calls"] == 0
    assert budget_row["failure"]["provider_construction"] is False
    assert not (raw_root / continuation.V2117_ACCEPTANCE_FILENAME).exists()


def test_acceptance_denominator_and_ledger_prefixes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract(monkeypatch)
    run_snapshot = _run_snapshot_before_science(contract)
    budget_snapshot = _budget_snapshot_before_science(contract)

    denominator = continuation._audit_acceptance_denominator(
        contract, run_snapshot, budget_snapshot
    )
    assert denominator["ledger_cells"] == 87
    assert denominator["fresh_scientific_cells"] == 86
    assert denominator["status_counts"] == {"complete": 1, "scheduled": 86}

    prefix = continuation._ledger_prefix(run_snapshot)
    assert prefix == {
        "event_count": 1,
        "event_chain_head": "1" * 64,
        "ledger_sha256": "2" * 64,
        "runs_sha256": canonical_sha256(run_snapshot["runs"]),
    }
    tampered = deepcopy(run_snapshot)
    next(iter(tampered["runs"].values()))["status"] = "failed"
    with pytest.raises(
        continuation.PilotV2117ContinuationError,
        match="unmarked acceptance ledger differs from sealed prefix",
    ):
        continuation._verify_acceptance_prefix_state(
            tampered,
            prefix=prefix,
            receipt={"ledger_prefixes": {}},
            receipt_path=(
                "experiment_results/pilot-v2.11.7/raw/"
                "scientific_dispatch_acceptance.json"
            ),
            budget=False,
        )


def test_acceptance_rejects_resealed_tampered_marker() -> None:
    receipt = continuation._seal(
        {
            "schema_version": continuation.V2117_ACCEPTANCE_SCHEMA_VERSION,
            "ledger_prefixes": {
                "run_ledger": {
                    "event_count": 1,
                    "event_chain_head": "1" * 64,
                },
                "budget_ledger": {
                    "event_count": 1,
                    "event_chain_head": "2" * 64,
                },
            },
        }
    )
    prefix = {
        "event_count": 1,
        "event_chain_head": "1" * 64,
        "ledger_sha256": "3" * 64,
        "runs_sha256": "4" * 64,
    }
    snapshot = {
        "events": [
            {"event_sha256": "1" * 64},
            {
                "event_type": "acceptance_receipt_bound",
                "payload": {"tampered": True},
            },
        ]
    }

    with pytest.raises(
        continuation.PilotV2117ContinuationError,
        match="acceptance ledger marker drifted",
    ):
        continuation._verify_acceptance_marker(
            snapshot,
            prefix=prefix,
            receipt=receipt,
            receipt_path=(
                "experiment_results/pilot-v2.11.7/raw/"
                "scientific_dispatch_acceptance.json"
            ),
            budget=False,
        )


def test_acceptance_requires_key_absence_and_blocks_provider_constructors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in continuation._PROVIDER_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    continuation.require_v2117_provider_keys_absent()
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret-never-read")
    with pytest.raises(
        continuation.PilotV2117ContinuationError,
        match="before provider credentials are loaded",
    ):
        continuation.require_v2117_provider_keys_absent()
    monkeypatch.delenv("OPENAI_API_KEY")

    original_factory = orchestrator.create_llm_provider
    original_multi = orchestrator.MultiModelLLM
    original_catalog = orchestrator.validate_live_provider_catalog
    with continuation._acceptance_provider_sentinels():
        for call in (
            lambda: orchestrator.create_llm_provider(),
            lambda: orchestrator.MultiModelLLM(),
            lambda: orchestrator.validate_live_provider_catalog(),
        ):
            with pytest.raises(
                continuation.PilotV2117ContinuationError,
                match="forbidden during acceptance",
            ):
                call()
    assert orchestrator.create_llm_provider is original_factory
    assert orchestrator.MultiModelLLM is original_multi
    assert orchestrator.validate_live_provider_catalog is original_catalog
