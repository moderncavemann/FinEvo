from copy import deepcopy
import json

import pytest

from verified_memory.m0_utility import EnvironmentLedger, UtilityConfig


def _pre(
    wealth: float,
    production: float,
    *,
    work: float = 0.5,
    consumption: float = 0.5,
    executed_labor: float = 168.0,
    executed_consumption: float = 0.5,
    interest_rate: float = 0.03,
) -> dict[str, float]:
    return {
        "wealth": wealth,
        "cumulative_production": production,
        "price": 2.0,
        "interest_rate": interest_rate,
        "proposed_work_propensity": work,
        "proposed_consumption_fraction": consumption,
        "executed_labor_hours": executed_labor,
        "executed_consumption_rate": executed_consumption,
    }


def _post(
    wealth: float,
    production: float,
    *,
    tax: float,
    transfer: float,
    spend: float,
    quantity: float,
    interest_applied: bool = False,
    interest_credit: float = 0.0,
) -> dict[str, float | bool]:
    return {
        "wealth": wealth,
        "cumulative_production": production,
        "tax_paid": tax,
        "lump_sum_transfer": transfer,
        "realized_consumption_spend": spend,
        "realized_consumption_quantity": quantity,
        "interest_applied": interest_applied,
        "interest_credit": interest_credit,
    }


def test_budget_identity_supply_rationing_and_proposed_vs_executed() -> None:
    ledger = EnvironmentLedger(
        UtilityConfig(consumption_scale=10.0, discount_factor=0.9)
    )
    pre = {
        "0": _pre(
            100.0,
            10.0,
            work=0.5,
            consumption=0.55,
            executed_labor=168.0,
            executed_consumption=0.5,
        )
    }
    # Gross income 40, tax 10, transfer 2 -> 132 cash. Requested spend is
    # 66, but only 60 is realized, leaving 72 cash.
    post = {
        "0": _post(
            72.0,
            50.0,
            tax=10.0,
            transfer=2.0,
            spend=60.0,
            quantity=30.0,
        )
    }
    ledger.capture_pre(0, pre)
    row = ledger.capture_post(0, post)[0]

    assert row.gross_labor_income == pytest.approx(40.0)
    assert row.cash_before_consumption == pytest.approx(132.0)
    assert row.requested_consumption_spend == pytest.approx(66.0)
    assert row.supply_rationed is True
    assert row.proposed_labor_hours == pytest.approx(84.0)
    assert row.executed_labor_hours == pytest.approx(168.0)
    assert row.proposed_consumption_fraction == pytest.approx(0.55)
    assert row.executed_consumption_rate == pytest.approx(0.5)
    assert row.budget_residual == pytest.approx(0.0)
    assert row.borrowing_supported is False
    assert row.debt_balance is None
    assert row.defaulted is None
    assert row.debt_default_status == "not_applicable"


def test_budget_identity_includes_interest_credit() -> None:
    ledger = EnvironmentLedger(UtilityConfig(consumption_scale=10.0))
    ledger.capture_pre(
        0,
        {
            "0": _pre(
                72.0,
                50.0,
                consumption=0.25,
                executed_consumption=0.25,
            )
        },
    )
    # Gross 30 - tax 5 + transfer 1 -> 98; spend 24.5 -> 73.5;
    # interest is 0.03 * 73.5 = 2.205; post wealth is 75.705.
    row = ledger.capture_post(
        0,
        {
            "0": _post(
                75.705,
                80.0,
                tax=5.0,
                transfer=1.0,
                spend=24.5,
                quantity=12.25,
                interest_applied=True,
                interest_credit=2.205,
            )
        },
    )[0]
    assert row.cash_before_interest == pytest.approx(73.5)
    assert row.interest_credit == pytest.approx(2.205)
    assert row.expected_wealth_post == pytest.approx(75.705)
    assert row.budget_residual == pytest.approx(0.0, abs=1e-10)

    bad = EnvironmentLedger(UtilityConfig())
    bad.capture_pre(0, {"0": _pre(72.0, 50.0, executed_consumption=0.0)})
    with pytest.raises(ValueError, match="interest credit is inconsistent"):
        bad.capture_post(
            0,
            {
                "0": _post(
                    73.0,
                    50.0,
                    tax=0.0,
                    transfer=0.0,
                    spend=0.0,
                    quantity=0.0,
                    interest_applied=False,
                    interest_credit=1.0,
                )
            },
        )


def test_explicit_noop_inputs_do_not_carry_stale_flow_values() -> None:
    ledger = EnvironmentLedger(UtilityConfig(consumption_scale=10.0))
    ledger.capture_pre(0, {"0": _pre(0.0, 0.0)})
    ledger.capture_post(
        0,
        {
            "0": _post(
                50.0,
                100.0,
                tax=20.0,
                transfer=20.0,
                spend=50.0,
                quantity=25.0,
            )
        },
    )

    # The simulator may still store prior 168-hour/50-dollar values after a
    # NO-OP.  M0 receives the current executed flows explicitly and records zero.
    ledger.capture_pre(
        1,
        {
            "0": _pre(
                50.0,
                100.0,
                work=0.0,
                consumption=0.0,
                executed_labor=0.0,
                executed_consumption=0.0,
            )
        },
    )
    row = ledger.capture_post(
        1,
        {
            "0": _post(
                50.0,
                100.0,
                tax=0.0,
                transfer=0.0,
                spend=0.0,
                quantity=0.0,
            )
        },
    )[0]
    assert row.executed_labor_hours == 0.0
    assert row.realized_consumption_spend == 0.0
    assert row.realized_consumption_quantity == 0.0
    assert row.flow_utility == pytest.approx(0.0)


def test_t_by_n_alignment_serialization_and_no_input_mutation() -> None:
    ledger = EnvironmentLedger(
        UtilityConfig(consumption_scale=10.0, discount_factor=0.5)
    )
    for period in range(2):
        pre = {
            "0": _pre(10.0, 0.0, executed_labor=0.0, executed_consumption=0.0),
            "1": _pre(20.0, 0.0, executed_labor=0.0, executed_consumption=0.0),
        }
        post = {
            "0": _post(
                10.0,
                0.0,
                tax=0.0,
                transfer=0.0,
                spend=0.0,
                quantity=0.0,
            ),
            "1": _post(
                20.0,
                0.0,
                tax=0.0,
                transfer=0.0,
                spend=0.0,
                quantity=0.0,
            ),
        }
        pre_before = deepcopy(pre)
        post_before = deepcopy(post)
        ledger.capture_pre(period, pre)
        ledger.capture_post(period, post)
        assert pre == pre_before
        assert post == post_before

    rows = ledger.finalize(expected_periods=2)
    assert len(rows) == 4
    assert ledger.num_agents == 2
    assert ledger.num_periods == 2
    assert [(row.period, row.agent_id) for row in rows] == [
        (0, "0"),
        (0, "1"),
        (1, "0"),
        (1, "1"),
    ]

    lines = ledger.to_jsonl().splitlines()
    assert len(lines) == 4
    decoded = [json.loads(line) for line in lines]
    assert all(row["schema_version"] == "m0.utility-ledger.v1" for row in decoded)
    assert all(row["debt_balance"] is None for row in decoded)
    assert all(row["defaulted"] is None for row in decoded)

    # Serialized records are fresh dictionaries and cannot mutate stored rows.
    records = ledger.records()
    records[0]["wealth_post"] = -999
    assert ledger.rows()[0].wealth_post == 10.0


def test_alignment_and_snapshot_schema_are_strict() -> None:
    ledger = EnvironmentLedger(UtilityConfig())
    with pytest.raises(ValueError, match="expected period 0"):
        ledger.capture_pre(1, {"0": _pre(0.0, 0.0)})

    invalid = _pre(0.0, 0.0)
    invalid["stale_labor_state"] = 168.0
    with pytest.raises(TypeError, match="unexpected keyword"):
        ledger.capture_pre(0, {"0": invalid})

    ledger.capture_pre(0, {"0": _pre(0.0, 0.0)})
    with pytest.raises(ValueError, match="cohort mismatch"):
        ledger.capture_post(
            0,
            {
                "1": _post(
                    0.0,
                    0.0,
                    tax=0.0,
                    transfer=0.0,
                    spend=0.0,
                    quantity=0.0,
                )
            },
        )
