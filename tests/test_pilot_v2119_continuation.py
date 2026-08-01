from __future__ import annotations

import ast
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from verified_memory import pilot_v2119_continuation as continuation
from verified_memory.pilot_contract import canonical_sha256, load_pilot_contract


ROOT = Path(__file__).resolve().parents[1]


def _draft_boundary() -> dict[str, Any]:
    document = json.loads(
        (ROOT / "experiments/pilot_v2_11_9.yaml").read_text(encoding="utf-8")
    )
    return document["v2119_recovery_boundary"]


def _contract() -> SimpleNamespace:
    return SimpleNamespace(
        contract_id=continuation.V2119_CONTRACT_ID,
        canonical_hash="9" * 64,
        implementation={"required_git_tag": continuation.V2119_SCIENCE_TAG},
        v2119_recovery_boundary=_draft_boundary(),
    )


def test_frozen_v2118_no_go_boundary_and_six_file_inventory_are_exact() -> None:
    boundary = _draft_boundary()

    assert (
        continuation._expected_v2118_failed_release_no_go()
        == boundary["failed_release_no_go"]
    )
    assert continuation._expected_v2115_parent_release() == boundary["parent_release"]
    assert continuation.V2118_COMPLETE_RAW_FILE_COUNT == 6
    assert continuation.V2118_COMPLETE_RAW_STORAGE_BYTES == 221_987
    assert continuation.V2118_COMPLETE_RAW_INVENTORY_SHA256 == (
        "aded9bfbdd3cc8ac1f4d4ce83b23b614528dc3848dcf39f744606e3aed2654ca"
    )
    assert continuation.V2118_EVIDENCE_RAW_FILE_COUNT == 5
    assert continuation.V2118_EVIDENCE_RAW_STORAGE_BYTES == 221_847
    assert continuation.V2118_EVIDENCE_RAW_INVENTORY_SHA256 == (
        "07919624f2bfaeef1c9c54883f089b543f454de4d3775bb73cdf2f7230427596"
    )
    assert set(continuation.V2118_RAW_FILE_BINDINGS) == {
        ".real-stage-execution.lock",
        "budget_ledger.json",
        "parent-import/stage_receipt.json",
        "release_attestation.json",
        "run_ledger.json",
        "scientific_launch_input.json",
    }
    assert boundary["failed_release_no_go"]["run_ledger"]["status_counts"] == {
        "integrity-stopped": 87
    }
    assert boundary["failed_release_no_go"]["provider_construction"] is False
    assert boundary["failed_release_no_go"]["provider_calls"] == 0
    assert boundary["failed_release_no_go"]["science_reservations"] == 0


def test_parent_debit_includes_v2118_actual_storage() -> None:
    debit = continuation.parent_budget_debit_for_v2119(_contract())

    assert debit.to_dict() == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": continuation.V2118_CONTRACT_SHA256,
        "parent_run_ledger_sha256": continuation.V2118_RUN_LEDGER_SHA256,
        "parent_budget_ledger_sha256": continuation.V2118_BUDGET_LEDGER_SHA256,
        "stage_bucket": "parent_v2118",
        "cost_usd": 63.1196450625,
        "hosted_completions": 3_440,
        "storage_bytes": 270_193_500,
        "record_sha256": continuation.V2119_PARENT_DEBIT_RECORD_SHA256,
    }
    assert debit.storage_bytes == 270_191_728 + 1_772

    tampered = _contract()
    tampered.v2119_recovery_boundary = deepcopy(tampered.v2119_recovery_boundary)
    tampered.v2119_recovery_boundary["parent_budget_debit"]["storage_bytes"] -= 1
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="parent budget debit drifted",
    ):
        continuation.parent_budget_debit_for_v2119(tampered)


@pytest.mark.parametrize(
    ("location", "field", "value"),
    (
        ("actual", "cost_usd", False),
        ("actual", "completions", False),
        ("actual", "storage_bytes", True),
        ("reservation", "completions", False),
    ),
)
def test_parent_import_budget_rejects_bool_numeric_confusion(
    location: str,
    field: str,
    value: Any,
) -> None:
    from verified_memory import pilot_orchestrator as orchestrator

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    spec = contract.expand(stage="parent-import")[0]
    projection = orchestrator._v2119_parent_import_projection(spec).to_dict()
    row = {
        "stage_bucket": projection["stage_bucket"],
        "reservation": deepcopy(projection),
        "status": "complete",
        "actual": {"cost_usd": 0.0, "completions": 0, "storage_bytes": 1},
    }
    row[location][field] = value

    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="zero-provider projection",
    ):
        continuation._verified_parent_import_budget_actual(contract, row)


def test_science_budget_replay_rejects_resealed_actual_type_confusion() -> None:
    parent = SimpleNamespace(run_id="parent")
    science = SimpleNamespace(
        run_id="science-b",
        model_id="gpt52_main",
        environment_seed=1,
    )

    def expand(*, stage: str) -> tuple[Any, ...]:
        return {
            "parent-import": (parent,),
            "experiment-d": (),
            "experiment-b": (science,),
            "cross-model": (),
        }[stage]

    contract = SimpleNamespace(
        contract_id=continuation.V2119_CONTRACT_ID,
        expand=expand,
    )
    reservation = {
        "run_id": science.run_id,
        "stage_bucket": "hosted_v2119",
        "cost_usd": 2.0,
        "completions": 4,
        "storage_bytes": 100,
        "basis": {"provider_calls": 4},
    }
    actual = {"cost_usd": 1.0, "completions": 2, "storage_bytes": 50}
    row = {
        "stage_bucket": "hosted_v2119",
        "reservation": reservation,
        "actual": actual,
        "status": "complete",
        "reserved_at": "2026-08-01T00:00:00+00:00",
        "finalized_at": "2026-08-01T00:01:00+00:00",
        "failure": None,
    }
    budget = {
        "runs": {science.run_id: row},
        "events": [
            {
                "event_type": "run_reserved",
                "payload": {
                    "run_id": science.run_id,
                    "projection_sha256": canonical_sha256(reservation),
                },
            },
            {
                "event_type": "run_finalized",
                "payload": {
                    "run_id": science.run_id,
                    "status": "complete",
                    "actual_sha256": canonical_sha256(actual),
                    "failure_sha256": None,
                },
            },
        ],
    }
    receipt = {
        "budget_projection": {
            "projection_sha256_by_run_id": {
                science.run_id: canonical_sha256(reservation)
            }
        }
    }
    run_snapshot = {"runs": {science.run_id: {"status": "complete"}}}
    continuation._verify_current_accepted_budget_rows(
        contract,
        receipt,
        budget,
        run_snapshot,
    )

    row["actual"]["completions"] = False
    budget["events"][1]["payload"]["actual_sha256"] = canonical_sha256(row["actual"])
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="terminal science budget row/event drifted",
    ):
        continuation._verify_current_accepted_budget_rows(
            contract,
            receipt,
            budget,
            run_snapshot,
        )


def test_science_budget_replay_allows_partial_d_only_while_reserved() -> None:
    parent = SimpleNamespace(run_id="parent")
    d_specs = (
        SimpleNamespace(run_id="d-a", model_id="gpt52_main", environment_seed=1),
        SimpleNamespace(run_id="d-b", model_id="gpt52_main", environment_seed=1),
    )

    def expand(*, stage: str) -> tuple[Any, ...]:
        return {
            "parent-import": (parent,),
            "experiment-d": d_specs,
            "experiment-b": (),
            "cross-model": (),
        }[stage]

    contract = SimpleNamespace(
        contract_id=continuation.V2119_CONTRACT_ID,
        expand=expand,
    )
    group_id = (
        f"{contract.contract_id}--experiment-d--gpt52_main--" "checkpoint-group--s1"
    )
    reservation = {
        "run_id": group_id,
        "stage_bucket": "hosted_v2119",
        "cost_usd": 2.0,
        "completions": 4,
        "storage_bytes": 100,
        "basis": {"provider_calls": 4},
    }
    row = {
        "stage_bucket": "hosted_v2119",
        "reservation": reservation,
        "actual": None,
        "status": "reserved",
        "reserved_at": "2026-08-01T00:00:00+00:00",
        "finalized_at": None,
    }
    budget = {
        "runs": {group_id: row},
        "events": [
            {
                "event_type": "run_reserved",
                "payload": {
                    "run_id": group_id,
                    "projection_sha256": canonical_sha256(reservation),
                },
            }
        ],
    }
    receipt = {
        "budget_projection": {
            "projection_sha256_by_run_id": {group_id: canonical_sha256(reservation)}
        }
    }
    run_snapshot = {
        "runs": {
            "d-a": {"status": "failed"},
            "d-b": {"status": "scheduled"},
        }
    }

    continuation._verify_current_accepted_budget_rows(
        contract,
        receipt,
        budget,
        run_snapshot,
    )


@pytest.mark.parametrize("stage_id", ("experiment-b", "experiment-d"))
def test_interrupted_reservation_is_reconciled_before_redispatch(
    tmp_path: Path,
    stage_id: str,
) -> None:
    from verified_memory.pilot_budget import PilotBudgetLedger, RunProjection
    from verified_memory import pilot_orchestrator as orchestrator

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    run_ledger = orchestrator.PilotRunLedger(
        tmp_path / f"{stage_id}-runs.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    run_ledger.register(contract.expand())
    budget_ledger = PilotBudgetLedger(
        tmp_path / f"{stage_id}-budget.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
    )
    if stage_id == "experiment-b":
        specs = (contract.expand(stage=stage_id)[0],)
        projection_id = specs[0].run_id
    else:
        first = contract.expand(stage=stage_id)[0]
        specs = tuple(
            spec
            for spec in contract.expand(stage=stage_id)
            if spec.model_id == first.model_id
            and spec.environment_seed == first.environment_seed
        )
        projection_id = (
            f"{contract.contract_id}--experiment-d--{first.model_id}--"
            f"checkpoint-group--s{first.environment_seed}"
        )
    budget_ledger.reserve(
        RunProjection(
            run_id=projection_id,
            stage_bucket="hosted_v2119",
            cost_usd=1.0,
            completions=1,
            storage_bytes=100,
            basis={"fixture": True},
        )
    )
    recovered = orchestrator._recover_v2119_interrupted_reservations_before_dispatch(
        contract,
        stage_id=stage_id,
        budget_ledger=budget_ledger,
        run_ledger=run_ledger,
    )
    assert recovered == (projection_id,)
    assert budget_ledger.snapshot()["runs"][projection_id]["status"] == (
        "integrity-stopped"
    )
    assert all(run_ledger.status(spec.run_id) == "integrity-stopped" for spec in specs)
    assert all(
        run_ledger.snapshot()["runs"][spec.run_id]["artifact_binding"] is None
        for spec in specs
    )


def test_first_resume_recovers_finalized_b_budget_with_scheduled_itt(
    tmp_path: Path,
) -> None:
    from verified_memory.pilot_budget import PilotBudgetLedger, RunProjection
    from verified_memory import pilot_orchestrator as orchestrator

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    spec = contract.expand(stage="experiment-b")[0]
    run_ledger = orchestrator.PilotRunLedger(
        tmp_path / "b-runs.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    run_ledger.register(contract.expand())
    budget_ledger = PilotBudgetLedger(
        tmp_path / "b-budget.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
    )
    projection = RunProjection(
        run_id=spec.run_id,
        stage_bucket=spec.budget_bucket,
        cost_usd=1.0,
        completions=2,
        storage_bytes=100,
        basis={"fixture": "finalized-before-itt"},
    )
    budget_ledger.reserve(projection)
    budget_ledger.finalize(
        spec.run_id,
        status="complete",
        cost_usd=0.5,
        completions=1,
        storage_bytes=50,
    )
    receipt = {
        "budget_projection": {
            "projection_sha256_by_run_id": {
                spec.run_id: canonical_sha256(projection.to_dict())
            }
        }
    }

    continuation._verify_current_accepted_budget_rows(
        contract,
        receipt,
        budget_ledger.snapshot(),
        run_ledger.snapshot(),
    )
    budget_before_recovery = budget_ledger.snapshot()
    recovered = orchestrator._recover_v2119_interrupted_reservations_before_dispatch(
        contract,
        stage_id="experiment-b",
        budget_ledger=budget_ledger,
        run_ledger=run_ledger,
    )

    assert recovered == (spec.run_id,)
    assert budget_ledger.snapshot() == budget_before_recovery
    recovered_row = run_ledger.snapshot()["runs"][spec.run_id]
    assert recovered_row["status"] == "integrity-stopped"
    assert recovered_row["failure"]["error_type"] == "BudgetFinalizedBeforeITT"
    assert recovered_row["artifact"] is None
    assert recovered_row["artifact_binding"] is None
    continuation._verify_current_accepted_budget_rows(
        contract,
        receipt,
        budget_ledger.snapshot(),
        run_ledger.snapshot(),
    )
    tampered_recovery = run_ledger.snapshot()
    tampered_recovery["runs"][spec.run_id]["failure"][
        "message"
    ] = "arbitrary single-run recovery"
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="finalized science budget/ITT terminality is invalid",
    ):
        continuation._verify_current_accepted_budget_rows(
            contract,
            receipt,
            budget_ledger.snapshot(),
            tampered_recovery,
        )


def test_reserved_b_recovery_replays_only_with_exact_interrupted_fingerprint(
    tmp_path: Path,
) -> None:
    from verified_memory.pilot_budget import PilotBudgetLedger, RunProjection
    from verified_memory import pilot_orchestrator as orchestrator

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    spec = contract.expand(stage="experiment-b")[0]
    run_path = tmp_path / "reserved-b-runs.json"
    budget_path = tmp_path / "reserved-b-budget.json"
    run_ledger = orchestrator.PilotRunLedger(
        run_path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    run_ledger.register(contract.expand())
    budget_ledger = PilotBudgetLedger(
        budget_path,
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
    )
    projection = RunProjection(
        run_id=spec.run_id,
        stage_bucket=spec.budget_bucket,
        cost_usd=1.0,
        completions=2,
        storage_bytes=100,
        basis={"fixture": "reserved-before-b-itt"},
    )
    budget_ledger.reserve(projection)
    receipt = {
        "budget_projection": {
            "projection_sha256_by_run_id": {
                spec.run_id: canonical_sha256(projection.to_dict())
            }
        }
    }

    continuation._verify_current_accepted_budget_rows(
        contract,
        receipt,
        budget_ledger.snapshot(),
        run_ledger.snapshot(),
    )
    recovered = orchestrator._recover_v2119_interrupted_reservations_before_dispatch(
        contract,
        stage_id="experiment-b",
        budget_ledger=budget_ledger,
        run_ledger=run_ledger,
    )
    assert recovered == (spec.run_id,)

    reloaded_runs = orchestrator.PilotRunLedger(
        run_path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    reloaded_budget = PilotBudgetLedger(
        budget_path,
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
    )
    budget_row = reloaded_budget.snapshot()["runs"][spec.run_id]
    run_row = reloaded_runs.snapshot()["runs"][spec.run_id]
    assert budget_row["status"] == run_row["status"] == "integrity-stopped"
    assert budget_row["failure"] == run_row["failure"]
    assert budget_row["failure"] == {
        "error_type": "InterruptedReservation",
        "message": (
            "a prior process created budget state without a terminal ITT cell; "
            "the cell is retained and is not redispatched"
        ),
        "accounting_basis": "unreconciled-conservative-reservation",
    }
    assert run_row["artifact"] is None
    assert run_row["artifact_binding"] is None
    continuation._verify_current_accepted_budget_rows(
        contract,
        receipt,
        reloaded_budget.snapshot(),
        reloaded_runs.snapshot(),
    )

    tampered_recovery = reloaded_runs.snapshot()
    tampered_recovery["runs"][spec.run_id]["failure"][
        "accounting_basis"
    ] = "tampered-basis"
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="finalized science budget/ITT terminality is invalid",
    ):
        continuation._verify_current_accepted_budget_rows(
            contract,
            receipt,
            reloaded_budget.snapshot(),
            tampered_recovery,
        )


def test_reserved_b_after_terminal_recovery_replays_bound_original_itt(
    tmp_path: Path,
) -> None:
    from verified_memory.pilot_budget import PilotBudgetLedger, RunProjection
    from verified_memory import pilot_orchestrator as orchestrator

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    spec = contract.expand(stage="experiment-b")[0]
    run_path = tmp_path / "reserved-after-itt-b-runs.json"
    budget_path = tmp_path / "reserved-after-itt-b-budget.json"
    run_ledger = orchestrator.PilotRunLedger(
        run_path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    run_ledger.register(contract.expand())
    budget_ledger = PilotBudgetLedger(
        budget_path,
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
    )
    projection = RunProjection(
        run_id=spec.run_id,
        stage_bucket=spec.budget_bucket,
        cost_usd=1.0,
        completions=2,
        storage_bytes=100,
        basis={"fixture": "reserved-after-b-itt"},
    )
    budget_ledger.reserve(projection)
    run_ledger.finalize(
        spec.run_id,
        status="failed",
        artifact=None,
        failure={"error_type": "ProviderFailure", "message": "bound ITT"},
    )
    original_terminal = deepcopy(run_ledger.snapshot()["runs"][spec.run_id])
    receipt = {
        "budget_projection": {
            "projection_sha256_by_run_id": {
                spec.run_id: canonical_sha256(projection.to_dict())
            }
        }
    }

    continuation._verify_current_accepted_budget_rows(
        contract,
        receipt,
        budget_ledger.snapshot(),
        run_ledger.snapshot(),
    )
    recovered = orchestrator._recover_v2119_interrupted_reservations_before_dispatch(
        contract,
        stage_id="experiment-b",
        budget_ledger=budget_ledger,
        run_ledger=run_ledger,
    )
    assert recovered == (spec.run_id,)
    assert run_ledger.snapshot()["runs"][spec.run_id] == original_terminal

    reloaded_runs = orchestrator.PilotRunLedger(
        run_path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    reloaded_budget = PilotBudgetLedger(
        budget_path,
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
    )
    budget_row = reloaded_budget.snapshot()["runs"][spec.run_id]
    assert budget_row["status"] == "integrity-stopped"
    assert budget_row["failure"] == {
        "error_type": "InterruptedReservationAfterITT",
        "message": (
            "a terminal ITT row retained an unreconciled reservation; the "
            "conservative reservation was charged before stopping"
        ),
        "accounting_basis": "unreconciled-conservative-reservation",
    }
    continuation._verify_current_accepted_budget_rows(
        contract,
        receipt,
        reloaded_budget.snapshot(),
        reloaded_runs.snapshot(),
    )

    tampered_original = reloaded_runs.snapshot()
    tampered_original["runs"][spec.run_id]["failure"]["message"] = "tampered ITT"
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="finalized science budget/ITT terminality is invalid",
    ):
        continuation._verify_current_accepted_budget_rows(
            contract,
            receipt,
            reloaded_budget.snapshot(),
            tampered_original,
        )


def test_first_resume_recovers_finalized_d_budget_with_partial_itt(
    tmp_path: Path,
) -> None:
    from verified_memory.pilot_budget import PilotBudgetLedger, RunProjection
    from verified_memory import pilot_orchestrator as orchestrator

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    first = contract.expand(stage="experiment-d")[0]
    group = tuple(
        spec
        for spec in contract.expand(stage="experiment-d")
        if spec.model_id == first.model_id
        and spec.environment_seed == first.environment_seed
    )
    group_id = (
        f"{contract.contract_id}--experiment-d--{first.model_id}--"
        f"checkpoint-group--s{first.environment_seed}"
    )
    run_ledger = orchestrator.PilotRunLedger(
        tmp_path / "d-runs.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    run_ledger.register(contract.expand())
    budget_failure = {
        "error_type": "ProviderFailure",
        "message": "budget finalized before all ITT rows",
    }
    run_ledger.finalize(
        group[0].run_id,
        status="failed",
        artifact=None,
        failure=budget_failure,
    )
    budget_ledger = PilotBudgetLedger(
        tmp_path / "d-budget.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
    )
    projection = RunProjection(
        run_id=group_id,
        stage_bucket=first.budget_bucket,
        cost_usd=1.0,
        completions=2,
        storage_bytes=100,
        basis={"fixture": "finalized-before-partial-itt"},
    )
    budget_ledger.reserve(projection)
    budget_ledger.finalize(
        group_id,
        status="failed",
        cost_usd=0.5,
        completions=1,
        storage_bytes=50,
        failure=budget_failure,
    )
    receipt = {
        "budget_projection": {
            "projection_sha256_by_run_id": {
                group_id: canonical_sha256(projection.to_dict())
            }
        }
    }

    continuation._verify_current_accepted_budget_rows(
        contract,
        receipt,
        budget_ledger.snapshot(),
        run_ledger.snapshot(),
    )
    invalid_mixed_statuses = run_ledger.snapshot()
    invalid_mixed_statuses["runs"][group[1].run_id]["status"] = "integrity-stopped"
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="finalized science budget/ITT terminality is invalid",
    ):
        continuation._verify_current_accepted_budget_rows(
            contract,
            receipt,
            budget_ledger.snapshot(),
            invalid_mixed_statuses,
        )
    budget_before_recovery = budget_ledger.snapshot()
    recovered = orchestrator._recover_v2119_interrupted_reservations_before_dispatch(
        contract,
        stage_id="experiment-d",
        budget_ledger=budget_ledger,
        run_ledger=run_ledger,
    )

    assert recovered == (group_id,)
    assert budget_ledger.snapshot() == budget_before_recovery
    assert run_ledger.status(group[0].run_id) == "failed"
    for spec in group[1:]:
        recovered_row = run_ledger.snapshot()["runs"][spec.run_id]
        assert recovered_row["status"] == "integrity-stopped"
        assert recovered_row["failure"]["error_type"] == ("BudgetFinalizedBeforeITT")
        assert recovered_row["artifact"] is None
        assert recovered_row["artifact_binding"] is None

    recovered_snapshot = run_ledger.snapshot()
    continuation._verify_current_accepted_budget_rows(
        contract,
        receipt,
        budget_ledger.snapshot(),
        recovered_snapshot,
    )
    tampered_recovery = deepcopy(recovered_snapshot)
    tampered_recovery["runs"][group[1].run_id]["failure"][
        "message"
    ] = "arbitrary terminal mixture"
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="finalized science budget/ITT terminality is invalid",
    ):
        continuation._verify_current_accepted_budget_rows(
            contract,
            receipt,
            budget_ledger.snapshot(),
            tampered_recovery,
        )


def test_reserved_partial_d_recovery_replays_with_exact_interrupted_fingerprint(
    tmp_path: Path,
) -> None:
    from verified_memory.pilot_budget import PilotBudgetLedger, RunProjection
    from verified_memory import pilot_orchestrator as orchestrator

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    first = contract.expand(stage="experiment-d")[0]
    group = tuple(
        spec
        for spec in contract.expand(stage="experiment-d")
        if spec.model_id == first.model_id
        and spec.environment_seed == first.environment_seed
    )
    group_id = (
        f"{contract.contract_id}--experiment-d--{first.model_id}--"
        f"checkpoint-group--s{first.environment_seed}"
    )
    run_path = tmp_path / "reserved-d-runs.json"
    budget_path = tmp_path / "reserved-d-budget.json"
    run_ledger = orchestrator.PilotRunLedger(
        run_path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    run_ledger.register(contract.expand())
    run_ledger.finalize(
        group[0].run_id,
        status="failed",
        artifact=None,
        failure={"error_type": "CrashFixture", "message": "partial D ITT"},
    )
    original_terminal = deepcopy(run_ledger.snapshot()["runs"][group[0].run_id])
    budget_ledger = PilotBudgetLedger(
        budget_path,
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
    )
    projection = RunProjection(
        run_id=group_id,
        stage_bucket=first.budget_bucket,
        cost_usd=1.0,
        completions=2,
        storage_bytes=100,
        basis={"fixture": "reserved-before-partial-itt"},
    )
    budget_ledger.reserve(projection)
    receipt = {
        "budget_projection": {
            "projection_sha256_by_run_id": {
                group_id: canonical_sha256(projection.to_dict())
            }
        }
    }

    continuation._verify_current_accepted_budget_rows(
        contract,
        receipt,
        budget_ledger.snapshot(),
        run_ledger.snapshot(),
    )
    recovered = orchestrator._recover_v2119_interrupted_reservations_before_dispatch(
        contract,
        stage_id="experiment-d",
        budget_ledger=budget_ledger,
        run_ledger=run_ledger,
    )

    assert recovered == (group_id,)
    assert run_ledger.snapshot()["runs"][group[0].run_id] == original_terminal
    budget_row = budget_ledger.snapshot()["runs"][group_id]
    assert budget_row["status"] == "integrity-stopped"
    assert budget_row["failure"]["error_type"] == "InterruptedReservation"
    assert budget_row["failure"]["accounting_basis"] == (
        "unreconciled-conservative-reservation"
    )
    for spec in group[1:]:
        recovered_row = run_ledger.snapshot()["runs"][spec.run_id]
        assert recovered_row["status"] == "integrity-stopped"
        assert recovered_row["failure"] == budget_row["failure"]
        assert recovered_row["artifact"] is None
        assert recovered_row["artifact_binding"] is None

    reloaded_runs = orchestrator.PilotRunLedger(
        run_path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    reloaded_budget = PilotBudgetLedger(
        budget_path,
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
    )
    continuation._verify_current_accepted_budget_rows(
        contract,
        receipt,
        reloaded_budget.snapshot(),
        reloaded_runs.snapshot(),
    )

    for field, value in (
        ("model_id", "tampered-model"),
        ("environment_seed", -1),
        ("accounting_basis", "tampered-basis"),
    ):
        tampered_recovery = reloaded_runs.snapshot()
        tampered_recovery["runs"][group[1].run_id]["failure"][field] = value
        with pytest.raises(
            continuation.PilotV2119ContinuationError,
            match="finalized science budget/ITT terminality is invalid",
        ):
            continuation._verify_current_accepted_budget_rows(
                contract,
                receipt,
                reloaded_budget.snapshot(),
                tampered_recovery,
            )
    for field, value in (
        ("artifact", "/tmp/tampered-artifact.json"),
        ("artifact_binding", {"path": "/tmp/tampered-artifact.json"}),
    ):
        tampered_recovery = reloaded_runs.snapshot()
        tampered_recovery["runs"][group[1].run_id][field] = value
        with pytest.raises(
            continuation.PilotV2119ContinuationError,
            match="finalized science budget/ITT terminality is invalid",
        ):
            continuation._verify_current_accepted_budget_rows(
                contract,
                receipt,
                reloaded_budget.snapshot(),
                tampered_recovery,
            )


def test_execute_stage_first_resume_stops_at_partial_d_recovery_before_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verified_memory.pilot_budget import PilotBudgetLedger, RunProjection
    from verified_memory import pilot_orchestrator as orchestrator

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    first = contract.expand(stage="experiment-d")[0]
    group = tuple(
        spec
        for spec in contract.expand(stage="experiment-d")
        if spec.model_id == first.model_id
        and spec.environment_seed == first.environment_seed
    )
    group_id = (
        f"{contract.contract_id}--experiment-d--{first.model_id}--"
        f"checkpoint-group--s{first.environment_seed}"
    )
    raw = tmp_path / "raw"
    run_ledger = orchestrator.PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    run_ledger.register(contract.expand())
    budget_failure = {
        "error_type": "ProviderFailure",
        "message": "sealed failure",
    }
    run_ledger.finalize(
        group[0].run_id,
        status="failed",
        artifact=None,
        failure=budget_failure,
    )
    budget_ledger = PilotBudgetLedger(
        raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=orchestrator._parent_budget_debit(contract),
    )
    projection = RunProjection(
        run_id=group_id,
        stage_bucket=first.budget_bucket,
        cost_usd=1.0,
        completions=2,
        storage_bytes=100,
        basis={"fixture": "entrypoint-partial-d"},
    )
    budget_ledger.reserve(projection)
    budget_ledger.finalize(
        group_id,
        status="failed",
        cost_usd=0.5,
        completions=1,
        storage_bytes=50,
        failure=budget_failure,
    )
    receipt = {
        "budget_projection": {
            "projection_sha256_by_run_id": {
                group_id: canonical_sha256(projection.to_dict())
            }
        }
    }
    budget_before_recovery = budget_ledger.snapshot()
    order: list[str] = []
    forbidden_calls: list[str] = []
    real_recovery = orchestrator._recover_v2119_interrupted_reservations_before_dispatch

    def acceptance(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        order.append("acceptance")
        continuation._verify_current_accepted_budget_rows(
            contract,
            receipt,
            kwargs["budget_ledger"].snapshot(),
            kwargs["run_ledger"].snapshot(),
        )
        return {"status": "go"}

    def recovery(*args: Any, **kwargs: Any) -> tuple[str, ...]:
        order.append("recovery")
        return real_recovery(*args, **kwargs)

    def forbidden(name: str):
        def fail(*_args: Any, **_kwargs: Any) -> Any:
            forbidden_calls.append(name)
            raise AssertionError(f"{name} must remain unreachable")

        return fail

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(orchestrator, "_persist_release_attestation", lambda *_a: None)
    monkeypatch.setattr(
        orchestrator,
        "verify_v2119_scientific_dispatch_acceptance",
        acceptance,
    )
    monkeypatch.setattr(
        orchestrator,
        "_recover_v2119_interrupted_reservations_before_dispatch",
        recovery,
    )
    monkeypatch.setattr(orchestrator, "_assert_prerequisites", forbidden("prereq"))
    monkeypatch.setattr(
        orchestrator,
        "audit_v2119_scientific_stage_namespace",
        forbidden("namespace"),
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v2119_terminal_scientific_artifacts",
        forbidden("terminal-replay"),
    )
    monkeypatch.setattr(
        orchestrator,
        "validate_live_provider_catalog",
        forbidden("catalog"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_provider_for_profile",
        forbidden("provider"),
    )
    monkeypatch.setattr(orchestrator, "_propagate_stage_no_go", lambda *_a, **_k: None)
    monkeypatch.setattr(
        orchestrator,
        "_write_stage_receipt",
        lambda *_a, **_k: raw / "diagnostic-no-go-receipt.json",
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError, match="prerequisites failed"
    ):
        orchestrator._execute_stage_locked(
            contract_path=ROOT / "experiments/pilot_v2_11_9.yaml",
            stage_id="experiment-d",
            resume=True,
            raw_root=raw,
            repo_root=ROOT,
        )

    assert order == ["acceptance", "recovery"]
    assert forbidden_calls == []
    reloaded_runs = orchestrator.PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    assert all(reloaded_runs.is_terminal(spec.run_id) for spec in group)
    assert reloaded_runs.status(group[0].run_id) == "failed"
    reloaded_budget = PilotBudgetLedger(
        raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=orchestrator._parent_budget_debit(contract),
    )
    assert reloaded_budget.snapshot() == budget_before_recovery
    continuation._verify_current_accepted_budget_rows(
        contract,
        receipt,
        reloaded_budget.snapshot(),
        reloaded_runs.snapshot(),
    )


def test_execute_d_resume_rejects_future_b_orphan_without_mutating_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verified_memory.pilot_budget import PilotBudgetLedger, RunProjection
    from verified_memory import pilot_orchestrator as orchestrator

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    b_spec = contract.expand(stage="experiment-b")[0]
    raw = tmp_path / "raw"
    run_ledger = orchestrator.PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    run_ledger.register(contract.expand())
    b_before_recovery = deepcopy(run_ledger.snapshot()["runs"][b_spec.run_id])
    budget_ledger = PilotBudgetLedger(
        raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=orchestrator._parent_budget_debit(contract),
    )
    projection = RunProjection(
        run_id=b_spec.run_id,
        stage_bucket=b_spec.budget_bucket,
        cost_usd=1.0,
        completions=2,
        storage_bytes=100,
        basis={"fixture": "future-b-orphan"},
    )
    budget_ledger.reserve(projection)
    budget_ledger.finalize(
        b_spec.run_id,
        status="complete",
        cost_usd=0.5,
        completions=1,
        storage_bytes=50,
    )
    receipt = {
        "budget_projection": {
            "projection_sha256_by_run_id": {
                b_spec.run_id: canonical_sha256(projection.to_dict())
            }
        }
    }
    budget_before_recovery = budget_ledger.snapshot()
    order: list[str] = []
    forbidden_calls: list[str] = []
    real_recovery = orchestrator._recover_v2119_interrupted_reservations_before_dispatch

    def acceptance(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        order.append("acceptance")
        continuation._verify_current_accepted_budget_rows(
            contract,
            receipt,
            kwargs["budget_ledger"].snapshot(),
            kwargs["run_ledger"].snapshot(),
        )
        return {"status": "go"}

    def recovery(*args: Any, **kwargs: Any) -> tuple[str, ...]:
        order.append("recovery")
        return real_recovery(*args, **kwargs)

    def forbidden(name: str):
        def fail(*_args: Any, **_kwargs: Any) -> Any:
            forbidden_calls.append(name)
            raise AssertionError(f"{name} must remain unreachable")

        return fail

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(orchestrator, "_persist_release_attestation", lambda *_a: None)
    monkeypatch.setattr(
        orchestrator,
        "verify_v2119_scientific_dispatch_acceptance",
        acceptance,
    )
    monkeypatch.setattr(
        orchestrator,
        "_recover_v2119_interrupted_reservations_before_dispatch",
        recovery,
    )
    monkeypatch.setattr(orchestrator, "_assert_prerequisites", forbidden("prereq"))
    monkeypatch.setattr(
        orchestrator,
        "audit_v2119_scientific_stage_namespace",
        forbidden("namespace"),
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v2119_terminal_scientific_artifacts",
        forbidden("terminal-replay"),
    )
    monkeypatch.setattr(
        orchestrator,
        "validate_live_provider_catalog",
        forbidden("catalog"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_provider_for_profile",
        forbidden("provider"),
    )
    monkeypatch.setattr(orchestrator, "_propagate_stage_no_go", lambda *_a, **_k: None)
    monkeypatch.setattr(
        orchestrator,
        "_write_stage_receipt",
        lambda *_a, **_k: raw / "diagnostic-no-go-receipt.json",
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError, match="prerequisites failed"
    ):
        orchestrator._execute_stage_locked(
            contract_path=ROOT / "experiments/pilot_v2_11_9.yaml",
            stage_id="experiment-d",
            resume=True,
            raw_root=raw,
            repo_root=ROOT,
        )

    assert order == ["acceptance", "recovery"]
    assert forbidden_calls == []
    reloaded_runs = orchestrator.PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    assert reloaded_runs.snapshot()["runs"][b_spec.run_id] == b_before_recovery
    assert (
        PilotBudgetLedger(
            raw / "budget_ledger.json",
            contract_hash=contract.canonical_hash,
            caps=orchestrator._budget_caps(contract),
            tamper_evident=True,
            parent_debit=orchestrator._parent_budget_debit(contract),
        ).snapshot()
        == budget_before_recovery
    )


def test_interrupted_reservation_recovery_blocks_other_stage_without_mutation(
    tmp_path: Path,
) -> None:
    from verified_memory.pilot_budget import PilotBudgetLedger, RunProjection
    from verified_memory import pilot_orchestrator as orchestrator

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    first = contract.expand(stage="experiment-d")[0]
    group = tuple(
        spec
        for spec in contract.expand(stage="experiment-d")
        if spec.model_id == first.model_id
        and spec.environment_seed == first.environment_seed
    )
    group_id = (
        f"{contract.contract_id}--experiment-d--{first.model_id}--"
        f"checkpoint-group--s{first.environment_seed}"
    )
    run_ledger = orchestrator.PilotRunLedger(
        tmp_path / "runs.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    run_ledger.register(contract.expand())
    budget_ledger = PilotBudgetLedger(
        tmp_path / "budget.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
    )
    budget_ledger.reserve(
        RunProjection(
            run_id=group_id,
            stage_bucket="hosted_v2119",
            cost_usd=1.0,
            completions=1,
            storage_bytes=100,
            basis={"fixture": True},
        )
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="outside the requested stage",
    ):
        orchestrator._recover_v2119_interrupted_reservations_before_dispatch(
            contract,
            stage_id="experiment-b",
            budget_ledger=budget_ledger,
            run_ledger=run_ledger,
        )

    assert budget_ledger.snapshot()["runs"][group_id]["status"] == "reserved"
    assert all(run_ledger.status(spec.run_id) == "scheduled" for spec in group)


def test_partial_d_interrupted_reservation_recovers_in_one_pass(
    tmp_path: Path,
) -> None:
    from verified_memory.pilot_budget import PilotBudgetLedger, RunProjection
    from verified_memory import pilot_orchestrator as orchestrator

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    first = contract.expand(stage="experiment-d")[0]
    group = tuple(
        spec
        for spec in contract.expand(stage="experiment-d")
        if spec.model_id == first.model_id
        and spec.environment_seed == first.environment_seed
    )
    group_id = (
        f"{contract.contract_id}--experiment-d--{first.model_id}--"
        f"checkpoint-group--s{first.environment_seed}"
    )
    run_ledger = orchestrator.PilotRunLedger(
        tmp_path / "runs.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    run_ledger.register(contract.expand())
    run_ledger.finalize(
        group[0].run_id,
        status="integrity-stopped",
        artifact=None,
        failure={"error_type": "CrashFixture", "message": "partial group"},
    )
    budget_ledger = PilotBudgetLedger(
        tmp_path / "budget.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
    )
    budget_ledger.reserve(
        RunProjection(
            run_id=group_id,
            stage_bucket="hosted_v2119",
            cost_usd=1.0,
            completions=1,
            storage_bytes=100,
            basis={"fixture": True},
        )
    )

    recovered = orchestrator._recover_v2119_interrupted_reservations_before_dispatch(
        contract,
        stage_id="experiment-d",
        budget_ledger=budget_ledger,
        run_ledger=run_ledger,
    )

    assert recovered == (group_id,)
    assert budget_ledger.snapshot()["runs"][group_id]["status"] == ("integrity-stopped")
    assert all(run_ledger.is_terminal(spec.run_id) for spec in group)
    assert run_ledger.status(group[0].run_id) == "integrity-stopped"


def test_historical_v2115_contract_binding_rejects_empty_binding() -> None:
    empty = SimpleNamespace(validate_provenance=lambda *_args: {})

    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="contract provenance binding differs",
    ):
        continuation._historical_v2115_contract_binding(empty)


def test_v2115_acceptance_adapter_uses_exact_historical_contract_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verified_memory import pilot_orchestrator as orchestrator
    from verified_memory import pilot_v2115_acceptance as v2115_acceptance

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_5.yaml")
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(
        continuation,
        "_v2115_authority_state",
        lambda _root: {"contract": contract, "raw_root": raw},
    )
    monkeypatch.setattr(
        continuation,
        "_strict_json",
        lambda _path, *, name: {"parent_debit": None, "name": name},
    )

    class _Ledger:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    monkeypatch.setattr(orchestrator, "PilotRunLedger", _Ledger)
    monkeypatch.setattr(orchestrator, "_budget_caps", lambda _contract: object())
    monkeypatch.setattr(continuation, "PilotBudgetLedger", _Ledger)
    captured: dict[str, Any] = {}

    def _verify(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        captured["paid"] = kwargs["paid"]
        return {"status": "go"}

    monkeypatch.setattr(
        v2115_acceptance,
        "verify_v2115_scientific_dispatch_acceptance",
        _verify,
    )

    result = continuation._verify_v2115_acceptance_with_authority_context(tmp_path)
    binding = captured["paid"].contract_binding

    assert result == {"status": "go"}
    assert binding == contract.validate_provenance(
        continuation.v2117.V2115_SCIENCE_COMMIT,
        continuation.v2117.V2115_SCIENCE_TAG,
    )
    assert canonical_sha256(binding) == continuation.V2115_CONTRACT_BINDING_SHA256
    assert canonical_sha256(binding) != canonical_sha256({})


def test_six_file_inventory_rejects_lock_or_json_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path.joinpath(*continuation.V2118_FAILED_RAW_ROOT.parts)
    raw.mkdir(parents=True)
    payloads = {
        ".real-stage-execution.lock": b"lock\n",
        "budget_ledger.json": b"budget\n",
        "parent-import/stage_receipt.json": b"stage\n",
        "release_attestation.json": b"attestation\n",
        "run_ledger.json": b"runs\n",
        "scientific_launch_input.json": b"launch\n",
    }
    for relative, payload in payloads.items():
        path = raw / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    rows = [
        {
            "path": relative,
            "byte_size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for relative, payload in sorted(payloads.items())
    ]
    evidence = [row for row in rows if row["path"] != ".real-stage-execution.lock"]
    monkeypatch.setattr(
        continuation,
        "V2118_RAW_FILE_BINDINGS",
        {row["path"]: (row["byte_size"], row["sha256"]) for row in rows},
    )
    monkeypatch.setattr(continuation, "V2118_COMPLETE_RAW_FILE_COUNT", len(rows))
    monkeypatch.setattr(
        continuation,
        "V2118_COMPLETE_RAW_STORAGE_BYTES",
        sum(row["byte_size"] for row in rows),
    )
    monkeypatch.setattr(
        continuation, "V2118_COMPLETE_RAW_INVENTORY_SHA256", canonical_sha256(rows)
    )
    monkeypatch.setattr(continuation, "V2118_EVIDENCE_RAW_FILE_COUNT", len(evidence))
    monkeypatch.setattr(
        continuation,
        "V2118_EVIDENCE_RAW_STORAGE_BYTES",
        sum(row["byte_size"] for row in evidence),
    )
    monkeypatch.setattr(
        continuation,
        "V2118_EVIDENCE_RAW_INVENTORY_SHA256",
        canonical_sha256(evidence),
    )

    inventory = continuation._v2118_raw_inventory(tmp_path)
    assert inventory["complete"]["file_count"] == 6
    assert inventory["evidence"]["file_count"] == 5

    (raw / ".real-stage-execution.lock").write_bytes(b"changed\n")
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="six-file raw binding drifted",
    ):
        continuation._v2118_raw_inventory(tmp_path)


def test_dual_root_roles_cannot_alias(tmp_path: Path) -> None:
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="must be distinct",
    ):
        continuation.verify_v2118_terminal_no_go(
            failed_repo_root=tmp_path,
            authority_repo_root=tmp_path,
        )


def test_all_three_release_root_roles_reject_parent_symlink_aliases(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(tmp_path, target_is_directory=True)
    aliased = continuation._real_root(alias_parent / "real", name="aliased")
    resolved = continuation._real_root(real, name="resolved")

    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="child and failed roots must be distinct",
    ):
        continuation._require_distinct_roots(
            child=resolved,
            failed=aliased,
            authority=tmp_path,
        )


def test_parent_import_receipt_is_zero_provider_and_tamper_evident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in continuation._PROVIDER_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    contract = _contract()
    state = {
        "run_snapshot": {"ledger_sha256": continuation.V2118_RUN_LEDGER_SHA256},
        "budget_snapshot": {"ledger_sha256": continuation.V2118_BUDGET_LEDGER_SHA256},
        "stage_receipt": {
            "integrity": {
                "content_sha256": (
                    continuation.V2118_PARENT_IMPORT_RECEIPT_CONTENT_SHA256
                )
            }
        },
    }
    monkeypatch.setattr(
        continuation,
        "verify_v2118_terminal_no_go",
        lambda **_kwargs: state,
    )
    monkeypatch.setattr(
        continuation,
        "_verify_v2115_acceptance_with_authority_context",
        lambda _root: {"status": "pass"},
    )
    paid = SimpleNamespace(
        git_tag=continuation.V2119_SCIENCE_TAG,
        head_commit="8" * 40,
        tag_commit="8" * 40,
        tag_object_type="tag",
        worktree_clean=True,
    )

    receipt = continuation.build_v2119_parent_import_receipt(
        contract=contract,
        failed_repo_root="failed-v2118",
        authority_repo_root="authority-v2115",
        paid=paid,
    )
    verified = continuation.verify_v2119_parent_import_receipt(
        receipt, contract=contract
    )

    assert verified["status"] == "complete"
    assert verified["go"] is True
    assert verified["denominator_continuation"] == {
        "failed_registered_rows": 87,
        "failed_integrity_stopped_rows": 87,
        "failed_rows_reclassified_or_redispatched": 0,
        "child_operational_rows": 1,
        "child_scientific_rows": 86,
    }
    assert verified["import_policy"]["provider_construction"] is False
    assert verified["import_policy"]["provider_calls"] == 0
    assert verified["import_policy"]["imported_effect_cells"] == 0
    assert verified["scientific_evidence"] is False

    tampered = deepcopy(receipt)
    tampered["failed_release_no_go"]["run_ledger"]["status_counts"] = {"complete": 87}
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="self-hash mismatch",
    ):
        continuation.verify_v2119_parent_import_receipt(tampered, contract=contract)

    resealed_mutations = []
    extra = deepcopy(receipt)
    extra["unexpected"] = True
    resealed_mutations.append(continuation._seal(extra))
    wrong_policy = deepcopy(receipt)
    wrong_policy["import_policy"]["hosted_provider_calls"] = False
    resealed_mutations.append(continuation._seal(wrong_policy))
    wrong_denominator = deepcopy(receipt)
    wrong_denominator["denominator_continuation"]["child_scientific_rows"] = 86.0
    resealed_mutations.append(continuation._seal(wrong_denominator))
    wrong_claim = deepcopy(receipt)
    wrong_claim["claim_boundary"] = "broader claim"
    resealed_mutations.append(continuation._seal(wrong_claim))
    for mutation in resealed_mutations:
        with pytest.raises(
            continuation.PilotV2119ContinuationError,
            match="parent-import receipt drifted",
        ):
            continuation.verify_v2119_parent_import_receipt(
                mutation,
                contract=contract,
            )


def test_parent_import_source_replay_precedes_lineage_and_raw_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in continuation._PROVIDER_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    repository = tmp_path / "release"
    raw = repository.joinpath(*continuation.V2119_RAW_ROOT.parts)
    raw.mkdir(parents=True)
    calls: list[str] = []

    def fail_replay(**_kwargs: Any) -> dict[str, Any]:
        calls.append("source-replay")
        raise continuation.PilotV2119ContinuationError("fixture replay drift")

    def forbidden_lineage(**_kwargs: Any) -> dict[str, Any]:
        calls.append("lineage")
        raise AssertionError("lineage must follow source replay")

    monkeypatch.setattr(continuation, "validate_v2119_source_manifest", fail_replay)
    monkeypatch.setattr(
        continuation,
        "verify_v2118_terminal_no_go",
        forbidden_lineage,
    )
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="fixture replay drift",
    ):
        continuation.build_v2119_parent_import_receipt(
            contract=_contract(),
            repo_root=repository,
            raw_root=raw,
            failed_repo_root=tmp_path / "failed",
            authority_repo_root=tmp_path / "authority",
            paid=SimpleNamespace(),
        )
    assert calls == ["source-replay"]
    assert list(raw.iterdir()) == []


def test_provider_keys_must_be_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in continuation._PROVIDER_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    continuation.require_v2119_provider_keys_absent()

    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret-never-read")
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="before provider credentials are loaded",
    ):
        continuation.require_v2119_provider_keys_absent()


def test_pre_science_namespace_accepts_exact_tree_and_rejects_stale_paths(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    raw = tmp_path / "exact"
    raw.mkdir()
    allowed_files, _ = continuation._expected_pre_science_paths(contract)
    for relative in sorted(allowed_files):
        path = raw / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture\n")
    continuation._audit_pre_science_namespace(raw, contract)

    stale = raw / "experiment-d/runs/stale.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="pre-science raw namespace contains unexpected paths",
    ):
        continuation._audit_pre_science_namespace(raw, contract)

    for index, relative in enumerate(
        (
            "provider_call_journals/stale.json",
            "provider_catalog/stale.json",
            "development-fake/stale.json",
        )
    ):
        candidate = tmp_path / f"reserved-{index}"
        path = candidate / relative
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")
        with pytest.raises(
            continuation.PilotV2119ContinuationError,
            match="pre-science raw namespace contains unexpected paths",
        ):
            continuation._audit_pre_science_namespace(candidate, contract)


def test_pre_science_namespace_rejects_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    raw = tmp_path / "raw"
    raw.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "marker.txt"
    marker.write_text("unchanged\n", encoding="utf-8")
    (raw / "experiment-b").symlink_to(external, target_is_directory=True)

    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="pre-science raw namespace contains a symlink",
    ):
        continuation._audit_pre_science_namespace(raw, contract)
    assert marker.read_text(encoding="utf-8") == "unchanged\n"


def test_scientific_stage_namespace_rejects_b_and_d_collisions_but_allows_terminal(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    raw = tmp_path / "raw"
    raw.mkdir()

    class _Ledger:
        def __init__(self, terminal: set[str] | None = None) -> None:
            self.terminal = set() if terminal is None else terminal
            self.verified: list[str] = []

        def is_terminal(self, run_id: str) -> bool:
            return run_id in self.terminal

        def verify_terminal_artifact_binding(self, run_id: str) -> None:
            assert run_id in self.terminal
            self.verified.append(run_id)

    ledger = _Ledger()
    report = continuation.audit_v2119_scientific_stage_namespace(
        contract,
        raw_root=raw,
        stage_id="experiment-b",
        run_ledger=ledger,
    )
    assert report["fresh_pending_cells"] == 25
    b_spec = contract.expand(stage="experiment-b")[0]
    stale_run = raw / "experiment-b/runs" / b_spec.run_id
    stale_run.mkdir(parents=True)
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="stale dispatch outputs",
    ):
        continuation.audit_v2119_scientific_stage_namespace(
            contract,
            raw_root=raw,
            stage_id="experiment-b",
            run_ledger=ledger,
        )

    ledger.terminal.add(b_spec.run_id)
    summary = raw / "experiment-b/summaries" / f"{b_spec.run_id}.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}\n", encoding="utf-8")
    report = continuation.audit_v2119_scientific_stage_namespace(
        contract,
        raw_root=raw,
        stage_id="experiment-b",
        run_ledger=ledger,
    )
    assert report["terminal_cells"] == 1
    assert report["verified_terminal_artifact_bindings"] == 1
    assert ledger.verified[-1] == b_spec.run_id

    diagnostic = raw / "experiment-b/diagnostic_summaries" / f"{b_spec.run_id}.json"
    diagnostic.parent.mkdir(parents=True)
    diagnostic.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="non-scientific or unexpected roots",
    ):
        continuation.audit_v2119_scientific_stage_namespace(
            contract,
            raw_root=raw,
            stage_id="experiment-b",
            run_ledger=ledger,
        )
    diagnostic.unlink()
    diagnostic.parent.rmdir()

    development = raw / "development-fake"
    development.mkdir()
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="non-scientific or unexpected roots",
    ):
        continuation.audit_v2119_scientific_stage_namespace(
            contract,
            raw_root=raw,
            stage_id="experiment-b",
            run_ledger=ledger,
        )
    development.rmdir()

    future_spec = contract.expand(stage="cross-model")[0]
    future_run = raw / "cross-model/runs" / future_spec.run_id
    future_run.mkdir(parents=True)
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="stale dispatch outputs",
    ):
        continuation.audit_v2119_scientific_stage_namespace(
            contract,
            raw_root=raw,
            stage_id="experiment-b",
            run_ledger=ledger,
        )
    future_run.rmdir()
    future_run.parent.rmdir()
    future_run.parent.parent.rmdir()

    summary.unlink()
    summary.mkdir()
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="stale dispatch outputs",
    ):
        continuation.audit_v2119_scientific_stage_namespace(
            contract,
            raw_root=raw,
            stage_id="experiment-b",
            run_ledger=ledger,
        )
    summary.rmdir()
    summary.write_text("{}\n", encoding="utf-8")

    d_spec = contract.expand(stage="experiment-d")[0]
    d_group = raw / "experiment-d/checkpoints" / f"s{d_spec.environment_seed}"
    d_group.mkdir(parents=True)
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="stale dispatch outputs",
    ):
        continuation.audit_v2119_scientific_stage_namespace(
            contract,
            raw_root=raw,
            stage_id="experiment-d",
            run_ledger=_Ledger(),
        )


def test_v2119_run_ledger_binds_terminal_artifact_bytes(
    tmp_path: Path,
) -> None:
    from verified_memory.pilot_orchestrator import PilotRunLedger

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    spec = contract.expand(stage="experiment-b")[0]
    ledger_path = tmp_path / "run_ledger.json"
    artifact = tmp_path / "manifest.json"
    artifact.write_text('{"value":1}\n', encoding="utf-8")
    ledger = PilotRunLedger(
        ledger_path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    ledger.register((spec,))
    ledger.finalize(
        spec.run_id,
        status="complete",
        artifact=str(artifact),
    )
    binding = ledger.verify_terminal_artifact_binding(spec.run_id)
    assert binding is not None
    assert binding["file_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()

    artifact.write_text('{"value":2}\n', encoding="utf-8")
    resumed = PilotRunLedger(
        ledger_path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    with pytest.raises(Exception, match="artifact binding drifted"):
        resumed.verify_terminal_artifact_binding(spec.run_id)


def test_provider_no_go_receipt_failure_must_equal_ledger_failure(
    tmp_path: Path,
) -> None:
    from verified_memory.pilot_provider_catalog import (
        PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION,
    )

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    spec = contract.expand(stage="experiment-b")[0]
    failure = {
        "error_type": "ProviderCatalogError",
        "message": "sealed catalog no-go fixture",
        "model_id": spec.model_id,
        "paid_completions": 0,
    }
    receipt = {
        "schema_version": PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION,
        "captured_at": "2026-08-01T00:00:00+00:00",
        "contract_sha256": contract.canonical_hash,
        "status": "no-go",
        "paid_completions": 0,
        "model_id": spec.model_id,
        "rows": [],
        "failure": failure,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    path = tmp_path / spec.stage_id / "provider_catalog" / f"{spec.model_id}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    row = {
        "status": "integrity-stopped",
        "artifact": str(path),
        "failure": deepcopy(failure),
    }
    continuation._verify_v2119_failure_artifact(
        contract,
        spec,
        row,
        raw_root=tmp_path,
    )

    row["failure"]["message"] = "ledger-only tamper"
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="provider no-go receipt drifted",
    ):
        continuation._verify_v2119_failure_artifact(
            contract,
            spec,
            row,
            raw_root=tmp_path,
        )


def test_v2119_development_matrix_is_complete_resumable_and_tamper_evident(
    tmp_path: Path,
) -> None:
    from verified_memory import pilot_orchestrator as orchestrator

    contract_path = ROOT / "experiments/pilot_v2_11_9.yaml"
    first = orchestrator.run_development_fake_matrix(
        contract_path=contract_path,
        resume=False,
        raw_root=tmp_path,
    )
    assert first["status"] == "pass"
    assert first["registered_cells"] == 18
    assert first["status_counts"] == {"complete": 18}

    resumed = orchestrator.run_development_fake_matrix(
        contract_path=contract_path,
        resume=True,
        raw_root=tmp_path,
    )
    assert resumed["status"] == "pass"
    ledger_path = tmp_path / "development-fake/run_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(ledger["runs"]) == 18
    assert all(
        isinstance(row.get("artifact_binding"), Mapping)
        for row in ledger["runs"].values()
    )

    actor_row = next(
        row
        for row in ledger["runs"].values()
        if row["spec"]["stage_id"] == "experiment-b"
    )
    artifact = Path(actor_row["artifact"])
    artifact.chmod(0o600)
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    with pytest.raises(Exception, match="artifact binding drifted"):
        orchestrator.run_development_fake_matrix(
            contract_path=contract_path,
            resume=True,
            raw_root=tmp_path,
        )


def test_v2119_evidence_uses_exact_recovery_stage_partition() -> None:
    from verified_memory import pilot_evidence

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    non_scientific, scientific = pilot_evidence._stage_sets(contract)
    assert non_scientific == {"parent-import"}
    assert scientific == {"experiment-d", "experiment-b", "cross-model"}


def test_terminal_scientific_verifier_replays_completed_b_before_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verified_memory import pilot_evidence
    from verified_memory.pilot_orchestrator import PilotRunLedger

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    raw = tmp_path / "raw"
    raw.mkdir()
    spec = contract.expand(stage="experiment-b")[0]
    manifest = raw / spec.stage_id / "runs" / spec.run_id / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    ledger = PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    ledger.register(contract.expand())
    ledger.finalize(spec.run_id, status="complete", artifact=str(manifest))
    replayed: list[str] = []
    journals: list[str] = []

    def replay(
        _contract: Any,
        run_spec: Mapping[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        replayed.append(str(run_spec["run_id"]))
        return {"scientific_eligible": True, "gate_evidence": {}}

    monkeypatch.setattr(pilot_evidence, "_load_completed_artifact", replay)
    monkeypatch.setattr(
        continuation,
        "_verify_v2119_actor_journal",
        lambda _contract, actor_spec, **_kwargs: journals.append(actor_spec.run_id),
    )
    report = continuation.verify_v2119_terminal_scientific_artifacts(
        contract,
        repo_root=ROOT,
        raw_root=raw,
        run_ledger=ledger,
    )
    assert report["terminal_artifact_bindings"] == 1
    assert report["completed_semantic_replays"] == 1
    assert replayed == [spec.run_id]
    assert journals == [spec.run_id]


def test_terminal_scientific_verifier_rejects_partial_d_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verified_memory import pilot_evidence
    from verified_memory.pilot_orchestrator import PilotRunLedger

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    raw = tmp_path / "raw"
    raw.mkdir()
    spec = contract.expand(stage="experiment-d")[0]
    summary = raw / spec.stage_id / "summaries" / f"{spec.run_id}.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}\n", encoding="utf-8")
    ledger = PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    ledger.register(contract.expand())
    ledger.finalize(spec.run_id, status="complete", artifact=str(summary))
    monkeypatch.setattr(
        pilot_evidence,
        "_load_completed_artifact",
        lambda *_args, **_kwargs: {
            "scientific_eligible": True,
            "gate_evidence": {
                "checkpoint_hash": "a" * 64,
                "prefix_hash": "b" * 64,
            },
        },
    )

    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="only partly terminal",
    ):
        continuation.verify_v2119_terminal_scientific_artifacts(
            contract,
            repo_root=ROOT,
            raw_root=raw,
            run_ledger=ledger,
        )


@pytest.mark.parametrize(
    ("status", "failure"),
    (
        ("complete", {"error_type": "ImpossibleCompleteFailure"}),
        ("failed", None),
    ),
)
def test_terminal_scientific_verifier_rejects_invalid_failure_disposition(
    tmp_path: Path,
    status: str,
    failure: Mapping[str, Any] | None,
) -> None:
    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    spec = contract.expand(stage="experiment-b")[0]
    rows = {
        item.run_id: {
            "status": "scheduled",
            "artifact": None,
            "failure": None,
        }
        for item in contract.expand()
    }
    rows[spec.run_id] = {
        "status": status,
        "artifact": None,
        "failure": failure,
    }
    ledger = SimpleNamespace(
        snapshot=lambda: {"runs": rows},
        verify_terminal_artifact_binding=lambda _run_id: None,
    )
    raw = tmp_path / "raw"
    raw.mkdir()

    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="invalid failure disposition",
    ):
        continuation.verify_v2119_terminal_scientific_artifacts(
            contract,
            repo_root=ROOT,
            raw_root=raw,
            run_ledger=ledger,
        )


def test_terminal_failure_replay_binds_failure_json_bytes(tmp_path: Path) -> None:
    from verified_memory.failure_artifacts import write_failure_receipt

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    spec = contract.expand(stage="experiment-b")[0]
    raw = tmp_path / "raw"
    raw.mkdir()
    commit = "a" * 40
    (raw / continuation.V2119_ACCEPTANCE_FILENAME).write_text(
        json.dumps({"release": {"git_commit": commit}}) + "\n",
        encoding="utf-8",
    )
    failure_dir = raw / spec.stage_id / "runs" / spec.run_id / "failure_receipt"
    artifact = write_failure_receipt(
        failure_dir,
        scope=f"finevo-pilot/{spec.stage_id}/{spec.execution_mode}",
        error=RuntimeError("fixture"),
        budget_snapshot={},
        config={
            "schema_version": "finevo-pilot-failure-config-v1",
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "projection": {},
            "run_specs": [spec.to_dict()],
            "provider_request_profiles": {},
            "provider_call_journals": [],
        },
        provenance={
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "paid_provenance": {
                "git_tag": continuation.V2119_SCIENCE_TAG,
                "head_commit": commit,
                "tag_commit": commit,
                "tag_object_type": "tag",
                "worktree_clean": True,
            },
            "diagnostic_only": False,
            "scientific_evidence": False,
        },
        git_commit=commit,
        git_dirty=False,
    )
    row = {
        "status": "failed",
        "artifact": str(artifact),
        "failure": {"error_type": "RuntimeError", "message": "fixture"},
    }
    continuation._verify_v2119_failure_artifact(
        contract,
        spec,
        row,
        raw_root=raw,
    )

    failure_json = failure_dir / "failure.json"
    original = failure_json.read_bytes()
    external = tmp_path / "external-failure.json"
    external.write_bytes(original)
    failure_json.unlink()
    failure_json.symlink_to(external)
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="payload path is invalid",
    ):
        continuation._verify_v2119_failure_artifact(
            contract,
            spec,
            row,
            raw_root=raw,
        )
    failure_json.unlink()
    failure_json.write_bytes(original)
    failure_json.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failure receipt hash mismatch"):
        continuation._verify_v2119_failure_artifact(
            contract,
            spec,
            row,
            raw_root=raw,
        )


def test_scientific_dispatch_target_symlink_fails_freshness_gate(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    spec = contract.expand(stage="experiment-b")[0]
    raw = tmp_path / "raw"
    target = tmp_path / "external"
    target.mkdir()
    run_path = raw / "experiment-b/runs" / spec.run_id
    run_path.parent.mkdir(parents=True)
    run_path.symlink_to(target, target_is_directory=True)
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="dispatch target is not fresh",
    ):
        continuation.assert_v2119_dispatch_target_fresh(
            contract,
            raw_root=raw,
            spec=spec,
        )


def test_d_checkpoint_collision_fails_before_projection_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verified_memory import pilot_orchestrator as orchestrator

    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    first = contract.expand(stage="experiment-d")[0]
    specs = tuple(
        spec
        for spec in contract.expand(stage="experiment-d")
        if spec.environment_seed == first.environment_seed
    )
    raw = tmp_path / "raw"
    group = raw / "experiment-d/checkpoints" / f"s{first.environment_seed}"
    group.mkdir(parents=True)
    provider_entries = 0

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal provider_entries
        provider_entries += 1
        raise AssertionError("projection/provider must remain unreachable")

    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden)
    monkeypatch.setattr(orchestrator, "_d_group_projection", forbidden)

    class _Ledger:
        @staticmethod
        def is_terminal(_run_id: str) -> bool:
            return False

    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="dispatch target is not fresh",
    ):
        orchestrator._execute_d_seed(
            contract,
            specs,
            raw_root=raw,
            paid=SimpleNamespace(),
            diagnostic=False,
            budget_ledger=SimpleNamespace(),
            run_ledger=_Ledger(),
        )
    assert provider_entries == 0


def test_acceptance_identity_rejects_extra_fields_and_every_boundary_drift() -> None:
    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    paid = SimpleNamespace(
        git_tag=continuation.V2119_SCIENCE_TAG,
        head_commit="a" * 40,
        tag_commit="a" * 40,
        tag_object_type="tag",
        worktree_clean=True,
    )
    receipt = {key: None for key in continuation._ACCEPTANCE_TOP_LEVEL_FIELDS}
    receipt.update(
        {
            "schema_version": continuation.V2119_ACCEPTANCE_SCHEMA_VERSION,
            "status": "go",
            "go": True,
            "contract_id": continuation.V2119_CONTRACT_ID,
            "contract_sha256": contract.canonical_hash,
            "release": continuation._release_binding(contract, paid),
            "raw_namespace": continuation.V2119_RAW_ROOT.as_posix(),
            "denominator": continuation._expected_acceptance_denominator(contract),
            "provider_boundary": deepcopy(continuation._ACCEPTANCE_PROVIDER_BOUNDARY),
            "scientific_evidence": False,
            "claim_boundary": continuation._ACCEPTANCE_CLAIM_BOUNDARY,
        }
    )
    continuation._verify_acceptance_identity(receipt, contract=contract, paid=paid)

    mutations: list[dict[str, Any]] = []
    extra = deepcopy(receipt)
    extra["unexpected"] = True
    mutations.append(extra)
    wrong_claim = deepcopy(receipt)
    wrong_claim["claim_boundary"] = "broader claim"
    mutations.append(wrong_claim)
    for field in continuation._ACCEPTANCE_PROVIDER_BOUNDARY:
        changed = deepcopy(receipt)
        changed["provider_boundary"][field] = None
        mutations.append(changed)
    type_confusions = {
        "provider_construction": 0,
        "provider_calls": False,
        "hosted_cost_usd": 0,
    }
    for field, value in type_confusions.items():
        changed = deepcopy(receipt)
        changed["provider_boundary"][field] = value
        mutations.append(changed)
    changed_denominator = deepcopy(receipt)
    changed_denominator["denominator"]["fresh_scientific_cells"] = 86.0
    mutations.append(changed_denominator)
    for changed in mutations:
        with pytest.raises(
            continuation.PilotV2119ContinuationError,
            match="acceptance identity drifted",
        ):
            continuation._verify_acceptance_identity(
                changed,
                contract=contract,
                paid=paid,
            )


def test_contract_module_binding_covers_imports_assignments_and_only_cycle_pins(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pilot_contract.py"

    def write(*, imported: str = "os", value: int = 1, design: str = "d" * 64) -> None:
        source.write_text(
            "\n".join(
                (
                    f"import {imported}",
                    f"VALUE = {value}",
                    "PILOT_CONTRACT_V2_11_9_CANONICAL_SHA256: str = " f"'{('a' * 64)}'",
                    "PILOT_CONTRACT_V2_11_9_SCIENCE_DESIGN_SHA256: str = "
                    f"'{design}'",
                    "PILOT_V2_11_9_SOURCE_MANIFEST_FILE_SHA256: str = "
                    f"'{('b' * 64)}'",
                    "PILOT_V2_11_9_SOURCE_MANIFEST_CONTENT_SHA256: str = "
                    f"'{('c' * 64)}'",
                    "",
                )
            ),
            encoding="utf-8",
        )

    write()
    baseline = continuation._normalized_contract_module_ast_binding(
        source,
        require_v2119_cycle_pins=True,
    )
    source.write_text(
        source.read_text(encoding="utf-8")
        .replace("'" + "a" * 64 + "'", "'" + "e" * 64 + "'")
        .replace("'" + "b" * 64 + "'", "'" + "f" * 64 + "'")
        .replace("'" + "c" * 64 + "'", "'" + "0" * 64 + "'"),
        encoding="utf-8",
    )
    assert (
        continuation._normalized_contract_module_ast_binding(
            source,
            require_v2119_cycle_pins=True,
        )
        == baseline
    )

    source.write_text(
        source.read_text(encoding="utf-8").replace("'" + "e" * 64 + "'", "None"),
        encoding="utf-8",
    )
    assert (
        continuation._normalized_contract_module_ast_binding(
            source,
            require_v2119_cycle_pins=True,
        )
        == baseline
    )

    write()
    source.write_text(
        source.read_text(encoding="utf-8").replace("'" + "b" * 64 + "'", "None"),
        encoding="utf-8",
    )
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="only the canonical pin may be literal None",
    ):
        continuation._normalized_contract_module_ast_binding(
            source,
            require_v2119_cycle_pins=True,
        )

    write(imported="sys")
    assert (
        continuation._normalized_contract_module_ast_binding(
            source,
            require_v2119_cycle_pins=True,
        )["normalized_ast_sha256"]
        != baseline["normalized_ast_sha256"]
    )
    write(value=2)
    assert (
        continuation._normalized_contract_module_ast_binding(
            source,
            require_v2119_cycle_pins=True,
        )["normalized_ast_sha256"]
        != baseline["normalized_ast_sha256"]
    )
    write(design="changed-design")
    assert (
        continuation._normalized_contract_module_ast_binding(
            source,
            require_v2119_cycle_pins=True,
        )["normalized_ast_sha256"]
        != baseline["normalized_ast_sha256"]
    )

    write()
    source.write_text(
        source.read_text(encoding="utf-8").replace("'" + "a" * 64 + "'", "dangerous()"),
        encoding="utf-8",
    )
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="must be one literal lowercase SHA-256",
    ):
        continuation._normalized_contract_module_ast_binding(
            source,
            require_v2119_cycle_pins=True,
        )


def test_source_manifest_binds_all_transitive_release_dependencies() -> None:
    audit = continuation.verify_v2119_frozen_release_source_manifest(ROOT)
    manifest = audit["manifest"]
    runtime = manifest["current_runtime_sources"]
    paths = {row["path"] for row in runtime["full_file_bindings"]}
    release_paths = set(runtime["release_python_source_paths"])
    assert tuple(runtime["release_python_source_paths"]) == tuple(
        audit["release_python_source_paths"]
    )
    assert release_paths == paths | continuation._V2119_NORMALIZED_AST_SOURCE_PATHS
    assert "verified_memory/pilot_v21110_continuation.py" not in release_paths
    assert "verified_memory/pilot_v21110_evidence.py" not in release_paths
    assert "verified_memory/__init__.py" in paths
    assert {
        "verified_memory/pilot_evidence.py",
        "verified_memory/pilot_v2115_gate.py",
        "verified_memory/pilot_v2115_parent_import.py",
        "verified_memory/pilot_v2117_continuation.py",
        "verified_memory/pilot_v2118_continuation.py",
    } <= paths
    entries = continuation._v2119_frozen_tree_entries(ROOT)
    for row in runtime["full_file_bindings"]:
        assert row == continuation._v2119_frozen_source_binding(
            ROOT,
            row["path"],
            entries=entries,
        )
    child = runtime["pilot_contract_complete_module_ast_bindings"]["child"]
    assert child["replaced_cycle_pins"] == sorted(
        continuation._CYCLIC_V2119_CONTRACT_PIN_NAMES
    )
    assert runtime["ci_release_receipt_complete_module_ast_binding"] == (
        continuation._normalized_ci_release_module_ast_source_binding(
            continuation._v2119_frozen_blob(
                ROOT,
                "verified_memory/ci_release_receipt.py",
                entries=entries,
            ).decode("utf-8"),
            filename=(
                f"{continuation.V2119_SCIENCE_COMMIT}:"
                "verified_memory/ci_release_receipt.py"
            ),
        )
    )
    assert runtime["bound_data_files"] == [
        continuation._v2119_frozen_source_binding(
            ROOT,
            relative,
            entries=entries,
        )
        for relative in continuation._V2119_BOUND_DATA_PATHS
    ]
    equivalence = manifest["remaining_science_implementation_equivalence"]
    environment_paths = {
        row["path"] for row in equivalence["environment_byte_identical_files"]
    }
    frozen_foundation_paths = {
        relative
        for relative in entries
        if relative.startswith("ai_economist/foundation/")
        and relative.endswith(".py")
    }
    assert environment_paths == frozen_foundation_paths | {
        continuation.V2119_PROFILE_PATH.as_posix()
    }

    # The successor legitimately repairs this shared module.  Historical
    # release evidence must still come from the V2.11.9 tag blob, not from the
    # current checkout or from a resealed manifest.
    authority_path = "verified_memory/observed_p95_authority.py"
    historical = next(
        row for row in runtime["full_file_bindings"] if row["path"] == authority_path
    )
    assert historical != continuation._source_file_binding(ROOT, authority_path)


def test_release_source_inventory_rejects_nested_directory_symlink(
    tmp_path: Path,
) -> None:
    package = tmp_path / "ai_economist/foundation"
    package.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (external / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "linked").symlink_to(external, target_is_directory=True)

    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="source inventory contains a symlink",
    ):
        continuation._v2119_foundation_source_paths(tmp_path)


def test_run_pilot_local_import_closure_is_inside_release_inventory() -> None:
    audit = continuation.verify_v2119_frozen_release_source_manifest(ROOT)
    release_paths = set(audit["release_python_source_paths"])
    entries = continuation._v2119_frozen_tree_entries(ROOT)
    candidate_paths = {
        relative for relative in entries if relative.endswith(".py")
    }

    def module_name(relative: str) -> str:
        parts = list(Path(relative).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    module_paths = {
        module_name(relative): relative
        for relative in candidate_paths
        if module_name(relative)
    }

    def resolve(name: str) -> str | None:
        while name:
            if name in module_paths:
                return module_paths[name]
            name = name.rpartition(".")[0]
        return None

    def local_imports(relative: str) -> set[str]:
        current = module_name(relative)
        package = (
            current
            if Path(relative).name == "__init__.py"
            else current.rpartition(".")[0]
        )
        found: set[str] = set()
        tree = ast.parse(
            continuation._v2119_frozen_blob(
                ROOT,
                relative,
                entries=entries,
            ).decode("utf-8")
        )
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    prefix = package.split(".") if package else []
                    prefix = prefix[: len(prefix) - (node.level - 1)]
                    if node.module:
                        prefix.extend(node.module.split("."))
                    base = ".".join(prefix)
                else:
                    base = node.module or ""
                if base:
                    names.append(base)
                names.extend(
                    f"{base}.{alias.name}"
                    for alias in node.names
                    if alias.name != "*" and base
                )
            found.update(path for name in names if (path := resolve(name)) is not None)
        return found

    closure: set[str] = set()
    pending = ["run_pilot.py"]
    while pending:
        relative = pending.pop()
        if relative in closure:
            continue
        closure.add(relative)
        pending.extend(sorted(local_imports(relative) - closure))

    assert closure <= release_paths
    assert "llm_providers.py" in closure
    assert "verified_memory/__init__.py" in release_paths
    assert "verified_memory/pilot_v2115_parent_import.py" in closure


def test_ci_module_binding_ignores_only_v2119_source_hash_cycle(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ci_release_receipt.py"
    original = (ROOT / "verified_memory/ci_release_receipt.py").read_text(
        encoding="utf-8"
    )
    source.write_text(original, encoding="utf-8")
    baseline = continuation._normalized_ci_release_module_ast_binding(source)
    manifest = json.loads(
        (ROOT / "experiments/pilot_v2_11_9_source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source_hashes = (
        manifest["integrity"]["content_sha256"],
        hashlib.sha256(
            (ROOT / "experiments/pilot_v2_11_9_source_manifest.json").read_bytes()
        ).hexdigest(),
    )
    changed = original
    for value in source_hashes:
        changed = changed.replace(value, "f" * 64)
    source.write_text(changed, encoding="utf-8")
    assert continuation._normalized_ci_release_module_ast_binding(source) == baseline

    dangerous = original.replace(
        '"' + source_hashes[0] + '"',
        "dangerous()",
        1,
    )
    source.write_text(dangerous, encoding="utf-8")
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="must be one literal lowercase SHA-256",
    ):
        continuation._normalized_ci_release_module_ast_binding(source)

    source.write_text(
        original.replace("import argparse", "import csv"), encoding="utf-8"
    )
    assert (
        continuation._normalized_ci_release_module_ast_binding(source)[
            "normalized_ast_sha256"
        ]
        != baseline["normalized_ast_sha256"]
    )


def test_bound_profile_requires_release_cwd_regular_file_and_exact_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "release"
    profile = repository.joinpath(*continuation.V2119_PROFILE_PATH.parts)
    profile.parent.mkdir(parents=True)
    profile.write_bytes((ROOT / continuation.V2119_PROFILE_PATH).read_bytes())

    poison_root = tmp_path / "outside"
    poison = poison_root.joinpath(*continuation.V2119_PROFILE_PATH.parts)
    poison.parent.mkdir(parents=True)
    poison.write_text('{"poison": true}\n', encoding="utf-8")
    monkeypatch.chdir(poison_root)
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="must run from the release repository root",
    ):
        continuation._verify_v2119_bound_working_directory(repository)

    monkeypatch.chdir(repository)
    binding = continuation._verify_v2119_bound_working_directory(repository)
    assert binding["file_sha256"] == continuation.V2119_PROFILE_FILE_SHA256

    profile.write_bytes(b"tampered\n")
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="profile input hash drifted",
    ):
        continuation._verify_v2119_bound_working_directory(repository)

    profile.unlink()
    external = tmp_path / "external-profile.json"
    external.write_bytes((ROOT / continuation.V2119_PROFILE_PATH).read_bytes())
    profile.symlink_to(external)
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="regular non-symlink file",
    ):
        continuation._verify_v2119_bound_working_directory(repository)


def test_acceptance_stale_namespace_leaves_ledgers_unmodified_and_no_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verified_memory import pilot_orchestrator as orchestrator

    for name in continuation._PROVIDER_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_11_9.yaml")
    repository = tmp_path / "release"
    contract_path = repository / "experiments/pilot_v2_11_9.yaml"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text("fixture\n", encoding="utf-8")
    raw = repository.joinpath(*continuation.V2119_RAW_ROOT.parts)
    raw.mkdir(parents=True)
    run_path = raw / "run_ledger.json"
    budget_path = raw / "budget_ledger.json"
    launch = raw / "scientific_launch_input.json"
    run_path.write_bytes(b"run-prefix\n")
    budget_path.write_bytes(b"budget-prefix\n")
    launch.write_text("{}\n", encoding="utf-8")
    stale = raw / "experiment-d/runs/stale.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(continuation, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        continuation,
        "_verify_v2119_bound_working_directory",
        lambda _root: {},
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    class _Ledger:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

    monkeypatch.setattr(orchestrator, "PilotRunLedger", _Ledger)
    monkeypatch.setattr(continuation, "PilotBudgetLedger", _Ledger)
    sentinel_entries = 0

    @contextmanager
    def _sentinels():
        nonlocal sentinel_entries
        sentinel_entries += 1
        yield

    monkeypatch.setattr(
        continuation.v2117, "_acceptance_provider_sentinels", _sentinels
    )
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="pre-science raw namespace contains unexpected paths",
    ):
        continuation.accept_v2119_scientific_dispatch(
            contract_path=contract_path,
            repo_root=repository,
            raw_root=raw,
            scientific_launch_input_path=launch,
        )
    assert run_path.read_bytes() == b"run-prefix\n"
    assert budget_path.read_bytes() == b"budget-prefix\n"
    assert sentinel_entries == 0


def test_unsealed_current_authority_and_acceptance_paths_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path.joinpath(*continuation.V2119_RAW_ROOT.parts)
    raw.mkdir(parents=True)
    assert continuation.current_authority_path(raw) == (
        raw / "parent-import/current_authority/post_gate_authority.json"
    )
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="must be a regular non-symlink file",
    ):
        continuation.verify_v2119_current_authority(
            contract=_contract(),
            repo_root=tmp_path,
            raw_root=raw,
            paid=SimpleNamespace(),
        )
    monkeypatch.setattr(
        continuation,
        "_verify_v2119_bound_working_directory",
        lambda _root: {},
    )
    with pytest.raises(
        continuation.PilotV2119ContinuationError,
        match="must be a regular non-symlink file",
    ):
        continuation.verify_v2119_scientific_dispatch_acceptance(
            raw / continuation.V2119_ACCEPTANCE_FILENAME,
            contract=_contract(),
            repo_root=tmp_path,
            raw_root=raw,
            paid=SimpleNamespace(),
            run_ledger=None,
            budget_ledger=None,
        )
