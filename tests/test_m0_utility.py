from dataclasses import FrozenInstanceError
import math

import pytest

from verified_memory.m0_utility import (
    DiscountedUtilityAccumulator,
    UtilityConfig,
    realized_flow_utility,
)


def test_zero_consumption_and_labor_is_finite_and_zero() -> None:
    for rho in (0.0, 0.5, 1.0, 2.0):
        result = realized_flow_utility(
            0.0, 0.0, UtilityConfig(rho=rho, consumption_scale=10.0)
        )
        assert result.consumption_utility == pytest.approx(0.0)
        assert result.labor_disutility == pytest.approx(0.0)
        assert result.flow_utility == pytest.approx(0.0)
        assert math.isfinite(result.flow_utility)


def test_realized_utility_is_monotone_in_consumption_and_labor() -> None:
    config = UtilityConfig(
        rho=1.0,
        labor_weight=1.5,
        inverse_frisch=1.0,
        consumption_scale=20.0,
        max_labor_hours=168.0,
    )
    low_consumption = realized_flow_utility(5.0, 40.0, config)
    high_consumption = realized_flow_utility(10.0, 40.0, config)
    low_labor = realized_flow_utility(10.0, 20.0, config)
    high_labor = realized_flow_utility(10.0, 160.0, config)

    assert high_consumption.flow_utility > low_consumption.flow_utility
    assert high_labor.flow_utility < low_labor.flow_utility


def test_log_and_crra_branches_match_closed_forms() -> None:
    quantity = 10.0
    scale = 10.0

    log_result = realized_flow_utility(
        quantity, 0.0, UtilityConfig(rho=1.0, consumption_scale=scale)
    )
    assert log_result.consumption_utility == pytest.approx(math.log(2.0))

    sqrt_result = realized_flow_utility(
        quantity, 0.0, UtilityConfig(rho=0.5, consumption_scale=scale)
    )
    assert sqrt_result.consumption_utility == pytest.approx(
        (2.0**0.5 - 1.0) / 0.5
    )

    inverse_result = realized_flow_utility(
        quantity, 0.0, UtilityConfig(rho=2.0, consumption_scale=scale)
    )
    assert inverse_result.consumption_utility == pytest.approx(0.5)


def test_config_is_immutable_and_strictly_validated() -> None:
    config = UtilityConfig()
    with pytest.raises(FrozenInstanceError):
        config.rho = 2.0  # type: ignore[misc]

    for kwargs in (
        {"rho": -0.1},
        {"consumption_scale": 0.0},
        {"max_labor_hours": 0.0},
        {"discount_factor": 0.0},
        {"discount_factor": 1.1},
        {"budget_tolerance": float("nan")},
    ):
        with pytest.raises((TypeError, ValueError)):
            UtilityConfig(**kwargs)

    with pytest.raises(ValueError):
        realized_flow_utility(-1.0, 0.0, config)
    with pytest.raises(ValueError):
        realized_flow_utility(1.0, 169.0, config)


def test_discounted_accumulator_requires_consecutive_agent_periods() -> None:
    accumulator = DiscountedUtilityAccumulator(0.5)
    assert accumulator.add("a", 0, 2.0) == pytest.approx(2.0)
    assert accumulator.add("a", 1, 2.0) == pytest.approx(1.0)
    assert accumulator.add("b", 0, -1.0) == pytest.approx(-1.0)

    assert accumulator.to_dict() == {
        "a": {
            "periods": 2,
            "undiscounted_total": 4.0,
            "discounted_total": 3.0,
        },
        "b": {
            "periods": 1,
            "undiscounted_total": -1.0,
            "discounted_total": -1.0,
        },
    }
    with pytest.raises(ValueError, match="expected utility period"):
        accumulator.add("a", 1, 0.0)
