from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from verified_memory.actions import parse_direct_action
from verified_memory.foundation_adapter import (
    build_foundation_actions,
    capture_foundation_snapshots,
    derive_foundation_transitions,
    foundation_action_for_decision,
    locate_component,
    m0_snapshot_batches,
    prepare_foundation_env_config,
)


def _root_config() -> dict:
    return {
        "env": {
            "n_agents": 100,
            "episode_length": 240,
            "flatten_masks": True,
            "flatten_observations": True,
            "multi_action_mode_agents": True,
            "multi_action_mode_planner": True,
            "components": [
                {"SimpleLabor": {"labor_step": 168, "num_labor_hours": 168}},
                {"PeriodicBracketTax": {"period": 1}},
                {"SimpleConsumption": {"consumption_rate_step": 0.02}},
                {"SimpleSaving": {"saving_rate": 0.0}},
            ],
        },
        "trainer": {"seed": None},
    }


def test_config_is_deep_copied_loaded_and_components_are_found_by_name(tmp_path) -> None:
    source = _root_config()
    source_before = deepcopy(source)
    prepared = prepare_foundation_env_config(
        source, n_agents=3, episode_length=12
    )
    assert source == source_before
    assert prepared["n_agents"] == 3
    assert prepared["episode_length"] == 12
    assert prepared["flatten_masks"] is False
    assert prepared["flatten_observations"] is False
    index, labor = locate_component(prepared, "SimpleLabor")
    assert index == 0
    assert labor["labor_step"] == 8.0
    assert labor["num_labor_hours"] == 168.0

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    loaded = prepare_foundation_env_config(
        path, n_agents=2, episode_length=6, labor_step=8
    )
    assert loaded["n_agents"] == 2
    assert locate_component(loaded, "SimpleConsumption")[0] == 2
    assert source == source_before

    with pytest.raises(ValueError, match="divide"):
        prepare_foundation_env_config(
            source, n_agents=2, episode_length=6, labor_step=10
        )


def test_direct_action_mapping_matches_foundation_indices_exactly() -> None:
    decision = parse_direct_action(
        '{"reflection":"balanced", "work":0.5,"consumption":0.33}',
        max_labor_hours=168,
        labor_step=8,
        consumption_step=0.02,
    )
    assert decision.labor_action_index == 11
    assert decision.executed_labor_hours == 88.0
    assert decision.consumption_action_index == 17
    assert decision.executed_consumption_rate == pytest.approx(0.34)
    assert foundation_action_for_decision(decision) == [11, 17]

    decisions = {1: decision, 0: decision}
    original = dict(decisions)
    actions = build_foundation_actions(decisions)
    assert actions == {"0": [11, 17], "1": [11, 17], "p": [0]}
    assert decisions == original

    tampered = replace(decision, executed_labor_hours=80.0)
    with pytest.raises(ValueError, match="labor hours"):
        foundation_action_for_decision(tampered)


class StubAgent:
    def __init__(
        self,
        idx: int,
        *,
        wealth: float,
        production: float,
        income: float,
        spend: float,
        quantity: float,
        job: str,
    ) -> None:
        self.idx = idx
        self.inventory = {"Coin": wealth, "Products": 0.0}
        self.income = {"Coin": income, "Products": 0.0}
        self.consumption = {"Coin": spend, "Products": quantity}
        self.endogenous = {"job": job}
        self.state = {
            "inventory": self.inventory,
            "income": self.income,
            "consumption": self.consumption,
            "endogenous": self.endogenous,
            "production": production,
        }


class StubEnv:
    def __init__(self, agents) -> None:
        self.world = SimpleNamespace(
            timestep=0,
            period=12,
            n_agents=len(agents),
            agents=agents,
            price=[100.0],
            interest_rate=[0.03],
            inflation=[],
            unemployment=[0],
        )
        self._components_dict = {
            "PeriodicBracketTax": SimpleNamespace(taxes=[]),
        }


def test_snapshot_capture_is_t0_safe_and_uses_explicit_current_fields() -> None:
    agent = StubAgent(
        0,
        wealth=0.0,
        production=0.0,
        income=0.0,
        spend=0.0,
        quantity=0.0,
        job="Unemployment",
    )
    env = StubEnv([agent])
    snapshot = capture_foundation_snapshots(env, expected_timestamp=0)["0"]
    assert snapshot.timestamp == 0
    assert snapshot.price == 100.0
    assert snapshot.interest_rate == 0.03
    assert snapshot.inflation == 0.0
    assert snapshot.unemployment_rate == 0.0
    assert snapshot.wealth == 0.0
    assert snapshot.income == 0.0
    assert snapshot.consumption_spend == 0.0
    assert snapshot.job == "Unemployment"
    assert snapshot.employed is False
    assert snapshot.to_m2_state()["low_labor_rate"] == 0.0

    env.world.unemployment[0] = np.int64(3)
    numpy_snapshot = capture_foundation_snapshots(env, expected_timestamp=0)["0"]
    assert numpy_snapshot.unemployment_rate == pytest.approx(0.25)

    with pytest.raises(ValueError, match="timestamp mismatch"):
        capture_foundation_snapshots(env, expected_timestamp=1)


def test_post_transition_is_aligned_stale_safe_and_budget_complete() -> None:
    agent = StubAgent(
        0,
        wealth=100.0,
        production=10.0,
        income=9.0,
        spend=8.0,
        quantity=4.0,
        job="Analyst",
    )
    env = StubEnv([agent])
    pre = capture_foundation_snapshots(env, expected_timestamp=0)
    decision = parse_direct_action(
        '{"work":0.5,"consumption":0.5}', labor_step=8
    )

    # Simulate one non-interest transition: +40 production, -10 tax, +2 transfer,
    # requested consumption 66 but only 60 realized -> post wealth 72.
    env.world.timestep = 1
    agent.state["production"] = 50.0
    agent.inventory["Coin"] = 72.0
    agent.income["Coin"] = 40.0
    agent.consumption["Coin"] = 60.0
    agent.consumption["Products"] = 0.6
    env.world.unemployment[0] = 0
    env._components_dict["PeriodicBracketTax"].taxes.append(
        {"0": {"tax_paid": 10.0, "lump_sum": 2.0}}
    )
    transitions = derive_foundation_transitions(
        env,
        pre_snapshots=pre,
        decisions={"0": decision},
        expected_outcome_t=1,
    )
    transition = transitions["0"]
    assert transition.decision_t == 0
    assert transition.outcome_t == 1
    assert transition.gross_labor_income == 40.0
    assert transition.requested_consumption_spend == 66.0
    assert transition.realized_consumption_spend == 60.0
    assert transition.supply_rationed is True
    assert transition.interest_applied is False
    assert transition.interest_credit == 0.0
    assert transition.budget_residual == 0.0
    assert transition.to_m2_outcome(decision)["wealth_change"] == -28.0
    assert transition.to_m2_outcome(decision)["labor_hours"] == 88.0

    pre_batch, post_batch = m0_snapshot_batches(transitions, {"0": decision})
    assert pre_batch["0"]["executed_labor_hours"] == 88.0
    assert post_batch["0"]["realized_consumption_spend"] == 60.0
    assert post_batch["0"]["interest_applied"] is False

    # Timestamp alignment is fail-closed.
    with pytest.raises(ValueError, match="timestamp mismatch"):
        derive_foundation_transitions(
            env,
            pre_snapshots=pre,
            decisions={"0": decision},
            expected_outcome_t=2,
        )


def test_noop_post_capture_zeros_stale_income_and_consumption() -> None:
    agent = StubAgent(
        0,
        wealth=50.0,
        production=100.0,
        income=40.0,
        spend=25.0,
        quantity=0.25,
        job="Analyst",
    )
    env = StubEnv([agent])
    noop = parse_direct_action('{"work":0,"consumption":0}', labor_step=8)
    snapshots = capture_foundation_snapshots(
        env, expected_timestamp=0, current_decisions={"0": noop}
    )
    assert snapshots["0"].income == 0.0
    assert snapshots["0"].consumption_spend == 0.0
    assert snapshots["0"].consumption_quantity == 0.0


def test_transition_derives_interest_on_foundation_period_boundary() -> None:
    agent = StubAgent(
        0,
        wealth=72.0,
        production=50.0,
        income=30.0,
        spend=24.5,
        quantity=0.245,
        job="Analyst",
    )
    env = StubEnv([agent])
    env.world.timestep = 11
    pre = capture_foundation_snapshots(env, expected_timestamp=11)
    decision = parse_direct_action(
        '{"work":0.5,"consumption":0.25}', labor_step=8
    )

    # 72 + 30 - 5 + 1 = 98 cash; 24.5 consumption leaves 73.5;
    # Foundation credits 3% at t=12, giving 75.705.
    env.world.timestep = 12
    agent.state["production"] = 80.0
    agent.inventory["Coin"] = 75.705
    agent.income["Coin"] = 30.0
    agent.consumption["Coin"] = 24.5
    agent.consumption["Products"] = 0.245
    env._components_dict["PeriodicBracketTax"].taxes.append(
        {"0": {"tax_paid": 5.0, "lump_sum": 1.0}}
    )
    transition = derive_foundation_transitions(
        env,
        pre_snapshots=pre,
        decisions={"0": decision},
        expected_outcome_t=12,
    )["0"]
    assert transition.interest_applied is True
    assert transition.interest_credit == pytest.approx(2.205)
    assert transition.expected_wealth_post == pytest.approx(75.705)
    assert transition.budget_residual == pytest.approx(0.0, abs=1e-12)
