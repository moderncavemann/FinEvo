from __future__ import annotations

import json
from pathlib import Path

import pytest

from verified_memory.pilot_budget import (
    PARENT_BUDGET_DEBIT_SCHEMA_VERSION,
    ParentBudgetDebit,
    PilotBudgetCaps,
    PilotBudgetError,
    PilotBudgetLedger,
    RunProjection,
)


CONTRACT_HASH = "a" * 64
PARENT_CONTRACT_HASH = (
    "980deddf2f82a762db7d73baa6ee0428c5e653298f4f275c5b3a5b23a95865c5"
)
PARENT_RUN_LEDGER_HASH = (
    "9d54ac1f22a56bafbe59164c7074d87bf914d290bc1282c631d50a4529f41fff"
)
PARENT_BUDGET_LEDGER_HASH = (
    "d9ec2c1bdfcc407aeb555ba71ee9d5e274d924e9a98791c5839ec749f3b1a0f2"
)
PARENT_DEBIT_RECORD_HASH = (
    "b2bb046832de4aa324a641920ab6a2f7dcb48168706a88dbd0d4fae54d1a4e50"
)
IMPORTED_COST_USD = 1.0701145
IMPORTED_HOSTED_COMPLETIONS = 30
IMPORTED_STORAGE_BYTES = 479_367


def _caps(
    *,
    capability_usd: float = 2.0,
    core_usd: float = 1.0,
    max_completions: int = 100,
    max_storage_bytes: int = 1_000_000,
) -> PilotBudgetCaps:
    return PilotBudgetCaps(
        total_usd=capability_usd + core_usd + 1.0,
        max_completions=max_completions,
        max_storage_bytes=max_storage_bytes,
        stage_usd_caps={
            "capability": capability_usd,
            "core": core_usd,
            "manual_reserve": 1.0,
        },
        automatic_reserve_usd=1.0,
    )


def _parent_debit(
    *,
    stage_bucket: str = "capability",
    cost_usd: float = IMPORTED_COST_USD,
    hosted_completions: int = IMPORTED_HOSTED_COMPLETIONS,
    storage_bytes: int = IMPORTED_STORAGE_BYTES,
    parent_budget_ledger_sha256: str = PARENT_BUDGET_LEDGER_HASH,
) -> ParentBudgetDebit:
    return ParentBudgetDebit(
        parent_contract_sha256=PARENT_CONTRACT_HASH,
        parent_run_ledger_sha256=PARENT_RUN_LEDGER_HASH,
        parent_budget_ledger_sha256=parent_budget_ledger_sha256,
        stage_bucket=stage_bucket,
        cost_usd=cost_usd,
        hosted_completions=hosted_completions,
        storage_bytes=storage_bytes,
    )


def _projection(
    run_id: str,
    *,
    stage_bucket: str = "capability",
    cost_usd: float = 0.0,
    completions: int = 0,
    storage_bytes: int = 0,
) -> RunProjection:
    return RunProjection(
        run_id=run_id,
        stage_bucket=stage_bucket,
        cost_usd=cost_usd,
        completions=completions,
        storage_bytes=storage_bytes,
        basis={"source": "v2.1-test"},
    )


def test_parent_debit_record_requires_parent_hashes_and_is_self_bound() -> None:
    debit = _parent_debit()
    value = debit.to_dict()

    assert value["schema_version"] == PARENT_BUDGET_DEBIT_SCHEMA_VERSION
    assert value["record_sha256"] == PARENT_DEBIT_RECORD_HASH
    assert ParentBudgetDebit.from_dict(value) == debit

    missing_hash = dict(value)
    missing_hash.pop("parent_run_ledger_sha256")
    with pytest.raises(ValueError, match="missing"):
        ParentBudgetDebit.from_dict(missing_hash)

    changed_without_rehash = dict(value)
    changed_without_rehash["parent_budget_ledger_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="record hash mismatch"):
        ParentBudgetDebit.from_dict(changed_without_rehash)


def test_parent_debit_is_genesis_adjacent_and_in_initial_totals(
    tmp_path: Path,
) -> None:
    debit = _parent_debit()
    ledger = PilotBudgetLedger(
        tmp_path / "budget.json",
        contract_hash=CONTRACT_HASH,
        caps=_caps(),
        tamper_evident=True,
        parent_debit=debit.to_dict(),
    )

    snapshot = ledger.snapshot()
    assert snapshot["parent_debit"] == debit.to_dict()
    assert snapshot["runs"] == {}
    assert [row["event_type"] for row in snapshot["events"]] == [
        "genesis",
        "parent_debit_imported",
    ]
    assert (
        snapshot["events"][0]["payload"]["parent_debit_sha256"]
        == debit.record_sha256
    )
    assert snapshot["events"][1]["payload"] == {
        "parent_debit": debit.to_dict()
    }
    for totals_key in ("committed", "committed_plus_reserved"):
        totals = snapshot[totals_key]
        assert totals["cost_usd"] == pytest.approx(IMPORTED_COST_USD)
        assert totals["completions"] == IMPORTED_HOSTED_COMPLETIONS
        assert totals["storage_bytes"] == IMPORTED_STORAGE_BYTES
        assert totals["stage_cost_usd"]["capability"] == pytest.approx(
            IMPORTED_COST_USD
        )


def test_parent_debit_precedes_and_adds_to_reservations(tmp_path: Path) -> None:
    ledger = PilotBudgetLedger(
        tmp_path / "budget.json",
        contract_hash=CONTRACT_HASH,
        caps=_caps(),
        tamper_evident=True,
        parent_debit=_parent_debit(),
    )
    ledger.reserve(
        _projection(
            "new-run",
            cost_usd=0.5,
            completions=5,
            storage_bytes=1_000,
        )
    )

    snapshot = ledger.snapshot()
    assert snapshot["committed"]["cost_usd"] == pytest.approx(
        IMPORTED_COST_USD
    )
    projected = snapshot["committed_plus_reserved"]
    assert projected["cost_usd"] == pytest.approx(IMPORTED_COST_USD + 0.5)
    assert projected["completions"] == IMPORTED_HOSTED_COMPLETIONS + 5
    assert projected["storage_bytes"] == IMPORTED_STORAGE_BYTES + 1_000
    assert projected["stage_cost_usd"]["capability"] == pytest.approx(
        IMPORTED_COST_USD + 0.5
    )


def test_parent_debit_reload_is_idempotent_and_rejects_different_import(
    tmp_path: Path,
) -> None:
    path = tmp_path / "budget.json"
    caps = _caps()
    debit = _parent_debit()
    original = PilotBudgetLedger(
        path,
        contract_hash=CONTRACT_HASH,
        caps=caps,
        tamper_evident=True,
        parent_debit=debit,
    ).snapshot()

    with pytest.raises(PilotBudgetError, match="expected parent debit"):
        PilotBudgetLedger(
            path,
            contract_hash=CONTRACT_HASH,
            caps=caps,
            tamper_evident=True,
        )
    identical = PilotBudgetLedger(
        path,
        contract_hash=CONTRACT_HASH,
        caps=caps,
        tamper_evident=True,
        parent_debit=debit.to_dict(),
    ).snapshot()
    assert identical["ledger_sha256"] == original["ledger_sha256"]

    different = _parent_debit(parent_budget_ledger_sha256="e" * 64)
    with pytest.raises(PilotBudgetError, match="differs from frozen import"):
        PilotBudgetLedger(
            path,
            contract_hash=CONTRACT_HASH,
            caps=caps,
            tamper_evident=True,
            parent_debit=different,
        )


@pytest.mark.parametrize("target", ["record", "event"])
def test_parent_debit_tampering_is_rejected(
    tmp_path: Path,
    target: str,
) -> None:
    path = tmp_path / "budget.json"
    caps = _caps()
    debit = _parent_debit()
    PilotBudgetLedger(
        path,
        contract_hash=CONTRACT_HASH,
        caps=caps,
        tamper_evident=True,
        parent_debit=debit,
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    if target == "record":
        value["parent_debit"]["cost_usd"] = 0.5
    else:
        value["events"][1]["payload"]["parent_debit"]["storage_bytes"] = 1
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(PilotBudgetError, match="hash|chain"):
        PilotBudgetLedger(
            path,
            contract_hash=CONTRACT_HASH,
            caps=caps,
            tamper_evident=True,
            parent_debit=debit,
        )


def test_parent_debit_is_counted_against_usd_caps(tmp_path: Path) -> None:
    with pytest.raises(PilotBudgetError, match="capability USD"):
        PilotBudgetLedger(
            tmp_path / "stage-over.json",
            contract_hash=CONTRACT_HASH,
            caps=_caps(capability_usd=1.0),
            tamper_evident=True,
            parent_debit=_parent_debit(),
        )

    ledger = PilotBudgetLedger(
        tmp_path / "reservation-over.json",
        contract_hash=CONTRACT_HASH,
        caps=_caps(),
        tamper_evident=True,
        parent_debit=_parent_debit(),
    )
    with pytest.raises(
        PilotBudgetError,
        match="dispatchable global USD.*capability USD",
    ):
        ledger.reserve(_projection("too-expensive", cost_usd=2.0))


def test_parent_debit_is_counted_against_completion_cap(
    tmp_path: Path,
) -> None:
    ledger = PilotBudgetLedger(
        tmp_path / "completion-cap.json",
        contract_hash=CONTRACT_HASH,
        caps=_caps(max_completions=IMPORTED_HOSTED_COMPLETIONS),
        tamper_evident=True,
        parent_debit=_parent_debit(),
    )

    with pytest.raises(PilotBudgetError, match="completion count"):
        ledger.reserve(_projection("one-too-many", completions=1))


def test_parent_debit_is_counted_against_storage_cap(tmp_path: Path) -> None:
    ledger = PilotBudgetLedger(
        tmp_path / "storage-cap.json",
        contract_hash=CONTRACT_HASH,
        caps=_caps(max_storage_bytes=IMPORTED_STORAGE_BYTES),
        tamper_evident=True,
        parent_debit=_parent_debit(),
    )

    with pytest.raises(PilotBudgetError, match="storage"):
        ledger.reserve(_projection("one-byte-too-many", storage_bytes=1))


def test_parent_debit_requires_v2_and_manual_reserve_stays_nondispatchable(
    tmp_path: Path,
) -> None:
    debit = _parent_debit()
    with pytest.raises(PilotBudgetError, match="tamper-evident"):
        PilotBudgetLedger(
            tmp_path / "v1.json",
            contract_hash=CONTRACT_HASH,
            caps=_caps(),
            parent_debit=debit,
        )
    with pytest.raises(PilotBudgetError, match="manual reserve"):
        PilotBudgetLedger(
            tmp_path / "manual-import.json",
            contract_hash=CONTRACT_HASH,
            caps=_caps(),
            tamper_evident=True,
            parent_debit=_parent_debit(stage_bucket="manual_reserve"),
        )

    ledger = PilotBudgetLedger(
        tmp_path / "manual-run.json",
        contract_hash=CONTRACT_HASH,
        caps=_caps(),
        tamper_evident=True,
        parent_debit=debit,
    )
    with pytest.raises(PilotBudgetError, match="manual reserve"):
        ledger.reserve(
            _projection(
                "forbidden",
                stage_bucket="manual_reserve",
                cost_usd=0.1,
            )
        )
