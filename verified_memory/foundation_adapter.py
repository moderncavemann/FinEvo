"""Isolated adapter between the verified runner and legacy Foundation environments.

The adapter imports neither ``simulate.py`` nor provider code.  It makes a private
copy of configuration, maps deterministic verified actions to Foundation's
multi-action list, and turns explicit pre/post environment states into aligned M0
and M2 records.

Foundation action-index assumptions
-----------------------------------
* Mobile agents run in multi-action mode with ``SimpleLabor`` registered before
  ``SimpleConsumption``.  Their action is therefore
  ``[labor_action_index, consumption_action_index]``.
* Index zero is NO-OP.  Positive labor index ``k`` executes
  ``k * labor_step`` hours.  Positive consumption index ``j`` executes rate
  ``j * consumption_step``.
* The verified environment uses direct hours (default step 8, maximum 168), not
  Bernoulli sampling.  The fixed planner action is ``[0]``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml

from .actions import ACTION_SCHEMA_VERSION, ActionDecision
from .m0_utility import PostStepSnapshot, PreStepSnapshot


DEFAULT_LABOR_STEP = 8.0
DEFAULT_MAX_LABOR_HOURS = 168.0
DEFAULT_CONSUMPTION_STEP = 0.02
_TOLERANCE = 1e-8


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _nonnegative(value: Any, name: str) -> float:
    value = _finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _load_config_source(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(source, Mapping):
        loaded: Any = deepcopy(dict(source))
    elif isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        loaded = deepcopy(loaded)
    else:
        raise TypeError("config source must be a mapping or YAML path")
    if not isinstance(loaded, dict):
        raise ValueError("configuration root must be a mapping")
    return loaded


def locate_component(
    env_config: Mapping[str, Any], component_name: str
) -> tuple[int, Mapping[str, Any]]:
    """Return a component's position and configuration without mutating it."""

    if not isinstance(component_name, str) or not component_name:
        raise TypeError("component_name must be a non-empty string")
    components = env_config.get("components")
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
        raise ValueError("env configuration must contain a component sequence")
    matches: list[tuple[int, Mapping[str, Any]]] = []
    for index, specification in enumerate(components):
        if not isinstance(specification, Mapping) or len(specification) != 1:
            raise ValueError(f"component specification {index} must be a one-key mapping")
        name, values = next(iter(specification.items()))
        if name == component_name:
            if not isinstance(values, Mapping):
                raise ValueError(f"{component_name} configuration must be a mapping")
            matches.append((index, values))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {component_name} component, found {len(matches)}"
        )
    return matches[0]


def _mutable_component(
    env_config: MutableMapping[str, Any], component_name: str
) -> tuple[int, MutableMapping[str, Any]]:
    index, values = locate_component(env_config, component_name)
    if not isinstance(values, MutableMapping):
        raise TypeError(f"{component_name} configuration is not mutable in copied config")
    return index, values


def prepare_foundation_env_config(
    source: Mapping[str, Any] | str | Path,
    *,
    n_agents: int,
    episode_length: int,
    labor_step: float = DEFAULT_LABOR_STEP,
    max_labor_hours: float = DEFAULT_MAX_LABOR_HOURS,
) -> dict[str, Any]:
    """Load and deep-copy an env configuration for the verified direct-hours runner.

    ``source`` may be either the full repository YAML (with an ``env`` key) or an
    environment mapping.  The returned object never aliases the caller's mapping.
    """

    root = _load_config_source(source)
    source_env = root.get("env", root)
    if not isinstance(source_env, Mapping):
        raise ValueError("env configuration must be a mapping")
    env_config = deepcopy(dict(source_env))

    n_agents = _positive_int(n_agents, "n_agents")
    episode_length = _positive_int(episode_length, "episode_length")
    labor_step = _finite(labor_step, "labor_step")
    max_labor_hours = _finite(max_labor_hours, "max_labor_hours")
    if labor_step <= 0 or max_labor_hours <= 0:
        raise ValueError("labor_step and max_labor_hours must be positive")
    action_count = max_labor_hours / labor_step
    if not math.isclose(action_count, round(action_count), abs_tol=1e-12):
        raise ValueError("labor_step must divide max_labor_hours exactly")

    labor_index, labor_config = _mutable_component(env_config, "SimpleLabor")
    consumption_index, _ = _mutable_component(env_config, "SimpleConsumption")
    # Foundation zips an action list to registered agent action spaces.  Enforce
    # the ordering required for [labor_index, consumption_index].
    if labor_index >= consumption_index:
        raise ValueError("SimpleLabor must precede SimpleConsumption in component order")

    env_config["n_agents"] = n_agents
    env_config["episode_length"] = episode_length
    env_config["flatten_masks"] = False
    env_config["flatten_observations"] = False
    env_config["multi_action_mode_agents"] = True
    env_config["multi_action_mode_planner"] = True
    labor_config["labor_step"] = labor_step
    labor_config["num_labor_hours"] = max_labor_hours
    return env_config


def foundation_action_for_decision(
    decision: ActionDecision,
    *,
    labor_step: float = DEFAULT_LABOR_STEP,
    max_labor_hours: float = DEFAULT_MAX_LABOR_HOURS,
    consumption_step: float = DEFAULT_CONSUMPTION_STEP,
) -> list[int]:
    """Map one verified direct decision to Foundation action indices exactly."""

    if not isinstance(decision, ActionDecision):
        raise TypeError("decision must be an ActionDecision")
    if decision.schema_version != ACTION_SCHEMA_VERSION:
        raise ValueError("unsupported action decision schema")
    labor_step = _finite(labor_step, "labor_step")
    max_labor_hours = _finite(max_labor_hours, "max_labor_hours")
    consumption_step = _finite(consumption_step, "consumption_step")
    if labor_step <= 0 or max_labor_hours <= 0 or consumption_step <= 0:
        raise ValueError("action steps and maximum labor must be positive")

    max_labor_index = int(max_labor_hours // labor_step)
    max_consumption_index = int(round(1.0 / consumption_step))
    if not 0 <= decision.labor_action_index <= max_labor_index:
        raise ValueError("labor action index is outside Foundation action space")
    if not 0 <= decision.consumption_action_index <= max_consumption_index:
        raise ValueError("consumption action index is outside Foundation action space")
    expected_hours = decision.labor_action_index * labor_step
    expected_rate = decision.consumption_action_index * consumption_step
    if not math.isclose(
        decision.executed_labor_hours, expected_hours, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("executed labor hours do not match labor action index")
    if not math.isclose(
        decision.executed_consumption_rate,
        expected_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("executed consumption rate does not match action index")
    return [decision.labor_action_index, decision.consumption_action_index]


def build_foundation_actions(
    decisions: Mapping[Any, ActionDecision],
    *,
    planner_action: int = 0,
    labor_step: float = DEFAULT_LABOR_STEP,
    max_labor_hours: float = DEFAULT_MAX_LABOR_HOURS,
    consumption_step: float = DEFAULT_CONSUMPTION_STEP,
) -> dict[str, list[int]]:
    """Build a fresh Foundation action dictionary, including planner ``[0]``."""

    if not isinstance(decisions, Mapping) or not decisions:
        raise ValueError("decisions must be a non-empty agent mapping")
    if isinstance(planner_action, bool) or not isinstance(planner_action, int):
        raise TypeError("planner_action must be an integer")
    if planner_action != 0:
        raise ValueError("verified fixed-planner runs require planner action zero")
    normalized = {str(agent_id): decision for agent_id, decision in decisions.items()}
    if len(normalized) != len(decisions) or "p" in normalized:
        raise ValueError("mobile agent identifiers must be unique and must not be 'p'")
    actions = {
        agent_id: foundation_action_for_decision(
            normalized[agent_id],
            labor_step=labor_step,
            max_labor_hours=max_labor_hours,
            consumption_step=consumption_step,
        )
        for agent_id in sorted(normalized)
    }
    actions["p"] = [planner_action]
    return actions


def _mapping_value(container: Any, key: str, name: str) -> float:
    if not isinstance(container, Mapping) or key not in container:
        raise ValueError(f"agent is missing {name} field {key!r}")
    return _finite(container[key], name)


def _agent_state_mapping(agent: Any, field: str) -> Mapping[str, Any]:
    direct = getattr(agent, field, None)
    if isinstance(direct, Mapping):
        return direct
    state = getattr(agent, "state", None)
    if isinstance(state, Mapping) and isinstance(state.get(field), Mapping):
        return state[field]
    raise ValueError(f"agent is missing mapping {field!r}")


def _agent_scalar(agent: Any, field: str) -> float:
    state = getattr(agent, "state", None)
    if not isinstance(state, Mapping) or field not in state:
        raise ValueError(f"agent is missing scalar state {field!r}")
    return _finite(state[field], field)


def _world_last(world: Any, field: str, *, default: float | None = None) -> float:
    values = getattr(world, field, None)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and values:
        return _finite(values[-1], f"world.{field}")
    if default is not None:
        return default
    raise ValueError(f"world.{field} must be a non-empty sequence")


def _unemployment_rate(world: Any, timestamp: int) -> float:
    period = _positive_int(getattr(world, "period", None), "world.period")
    n_agents = _positive_int(getattr(world, "n_agents", None), "world.n_agents")
    values = getattr(world, "unemployment", None)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        return 0.0
    year = 0 if timestamp == 0 else (timestamp - 1) // period
    if year < 0 or year >= len(values):
        return 0.0
    count = _nonnegative(values[year], f"world.unemployment[{year}]")
    return count / period / n_agents


@dataclass(frozen=True, slots=True)
class FoundationEconomicSnapshot:
    timestamp: int
    agent_id: str
    wealth: float
    cumulative_production: float
    income: float
    consumption_spend: float
    consumption_quantity: float
    job: str
    employed: bool
    price: float
    interest_rate: float
    inflation: float
    unemployment_rate: float

    def __post_init__(self) -> None:
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, int):
            raise TypeError("timestamp must be an integer")
        if self.timestamp < 0:
            raise ValueError("timestamp must be nonnegative")
        if not isinstance(self.agent_id, str) or not self.agent_id:
            raise TypeError("agent_id must be a non-empty string")
        for name in (
            "wealth",
            "cumulative_production",
            "income",
            "consumption_spend",
            "consumption_quantity",
            "interest_rate",
            "unemployment_rate",
        ):
            value = _nonnegative(getattr(self, name), name)
            object.__setattr__(self, name, value)
        price = _finite(self.price, "price")
        if price <= 0:
            raise ValueError("price must be positive")
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "inflation", _finite(self.inflation, "inflation"))
        if not isinstance(self.job, str) or not self.job:
            raise TypeError("job must be a non-empty string")
        if not isinstance(self.employed, bool):
            raise TypeError("employed must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_m2_state(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "price": self.price,
            "interest_rate": self.interest_rate,
            "unemployment_rate": self.unemployment_rate,
            "low_labor_rate": self.unemployment_rate,
            "inflation": self.inflation,
            "wealth": self.wealth,
            "income": self.income,
            "consumption_spend": self.consumption_spend,
            "consumption_quantity": self.consumption_quantity,
            "job": self.job,
            "employed": self.employed,
        }


def capture_foundation_snapshots(
    env: Any,
    *,
    expected_timestamp: int | None = None,
    current_decisions: Mapping[Any, ActionDecision] | None = None,
) -> dict[str, FoundationEconomicSnapshot]:
    """Capture explicit current state, optionally zeroing current NO-OP flows.

    Passing ``current_decisions`` for a post-step capture prevents legacy retained
    income/consumption fields from being mistaken for current-period flows.
    """

    world = getattr(env, "world", None)
    if world is None:
        raise ValueError("env must expose world")
    timestamp = getattr(world, "timestep", None)
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise ValueError("world.timestep must be a nonnegative integer")
    if expected_timestamp is not None and timestamp != expected_timestamp:
        raise ValueError(
            f"timestamp mismatch: expected {expected_timestamp}, observed {timestamp}"
        )

    price = _world_last(world, "price")
    interest_rate = _world_last(world, "interest_rate")
    inflation = _world_last(world, "inflation", default=0.0)
    unemployment_rate = _unemployment_rate(world, timestamp)
    agents = getattr(world, "agents", None)
    if not isinstance(agents, Sequence) or isinstance(agents, (str, bytes)):
        raise ValueError("world.agents must be a sequence")

    normalized_decisions: dict[str, ActionDecision] | None = None
    if current_decisions is not None:
        normalized_decisions = {
            str(agent_id): decision for agent_id, decision in current_decisions.items()
        }
        if len(normalized_decisions) != len(current_decisions):
            raise ValueError("decision agent identifiers collide after normalization")

    snapshots: dict[str, FoundationEconomicSnapshot] = {}
    for agent in agents:
        agent_id = str(getattr(agent, "idx", ""))
        if not agent_id:
            raise ValueError("agent.idx must be present")
        if agent_id in snapshots:
            raise ValueError(f"duplicate agent id {agent_id}")
        inventory = _agent_state_mapping(agent, "inventory")
        income_values = _agent_state_mapping(agent, "income")
        consumption = _agent_state_mapping(agent, "consumption")
        endogenous = _agent_state_mapping(agent, "endogenous")
        if "job" not in endogenous or not isinstance(endogenous["job"], str):
            raise ValueError("agent endogenous state must contain string job")

        income = _mapping_value(income_values, "Coin", "income")
        consumption_spend = _mapping_value(consumption, "Coin", "consumption")
        consumption_quantity = _mapping_value(
            consumption, "Products", "consumption quantity"
        )
        if normalized_decisions is not None:
            if agent_id not in normalized_decisions:
                raise ValueError(f"missing current decision for agent {agent_id}")
            decision = normalized_decisions[agent_id]
            foundation_action_for_decision(decision)
            if decision.labor_action_index == 0:
                income = 0.0
            if decision.consumption_action_index == 0:
                consumption_spend = 0.0
                consumption_quantity = 0.0

        job = endogenous["job"]
        snapshots[agent_id] = FoundationEconomicSnapshot(
            timestamp=timestamp,
            agent_id=agent_id,
            wealth=_mapping_value(inventory, "Coin", "wealth"),
            cumulative_production=_agent_scalar(agent, "production"),
            income=income,
            consumption_spend=consumption_spend,
            consumption_quantity=consumption_quantity,
            job=job,
            employed=job != "Unemployment",
            price=price,
            interest_rate=interest_rate,
            inflation=inflation,
            unemployment_rate=unemployment_rate,
        )
    if normalized_decisions is not None and set(normalized_decisions) != set(snapshots):
        raise ValueError("current decision cohort does not match environment agents")
    return snapshots


def _env_component(env: Any, name: str) -> Any:
    components = getattr(env, "_components_dict", None)
    if not isinstance(components, Mapping) or name not in components:
        raise ValueError(f"environment is missing component {name}")
    return components[name]


@dataclass(frozen=True, slots=True)
class FoundationTransition:
    decision_t: int
    outcome_t: int
    agent_id: str
    pre: FoundationEconomicSnapshot
    post: FoundationEconomicSnapshot
    gross_labor_income: float
    tax_paid: float
    lump_sum_transfer: float
    requested_consumption_spend: float
    realized_consumption_spend: float
    realized_consumption_quantity: float
    supply_rationed: bool
    interest_applied: bool
    interest_credit: float
    expected_wealth_post: float
    budget_residual: float

    def to_m0_pre(self, decision: ActionDecision) -> PreStepSnapshot:
        return PreStepSnapshot(
            period=self.decision_t,
            agent_id=self.agent_id,
            wealth=self.pre.wealth,
            cumulative_production=self.pre.cumulative_production,
            price=self.pre.price,
            interest_rate=self.pre.interest_rate,
            proposed_work_propensity=decision.proposed_work_fraction,
            proposed_consumption_fraction=decision.proposed_consumption_fraction,
            executed_labor_hours=decision.executed_labor_hours,
            executed_consumption_rate=decision.executed_consumption_rate,
        )

    def to_m0_post(self) -> PostStepSnapshot:
        return PostStepSnapshot(
            period=self.decision_t,
            agent_id=self.agent_id,
            wealth=self.post.wealth,
            cumulative_production=self.post.cumulative_production,
            tax_paid=self.tax_paid,
            lump_sum_transfer=self.lump_sum_transfer,
            realized_consumption_spend=self.realized_consumption_spend,
            realized_consumption_quantity=self.realized_consumption_quantity,
            interest_applied=self.interest_applied,
            interest_credit=self.interest_credit,
        )

    def to_m2_outcome(self, decision: ActionDecision) -> dict[str, Any]:
        return {
            "decision_t": self.decision_t,
            "outcome_t": self.outcome_t,
            "wealth_change": self.post.wealth - self.pre.wealth,
            "gross_labor_income": self.gross_labor_income,
            "tax_paid": self.tax_paid,
            "lump_sum_transfer": self.lump_sum_transfer,
            "consumption_spend": self.realized_consumption_spend,
            "consumption_quantity": self.realized_consumption_quantity,
            "labor_hours": decision.executed_labor_hours,
            "interest_credit": self.interest_credit,
            "supply_rationed": self.supply_rationed,
            "employment_changed": self.pre.employed != self.post.employed,
            "budget_residual": self.budget_residual,
        }


def derive_foundation_transitions(
    env: Any,
    *,
    pre_snapshots: Mapping[str, FoundationEconomicSnapshot],
    decisions: Mapping[Any, ActionDecision],
    expected_outcome_t: int,
    tolerance: float = _TOLERANCE,
) -> dict[str, FoundationTransition]:
    """Capture state at ``t+1`` and derive aligned M0/M2 budget outcomes."""

    if isinstance(expected_outcome_t, bool) or not isinstance(expected_outcome_t, int):
        raise TypeError("expected_outcome_t must be an integer")
    if expected_outcome_t < 1:
        raise ValueError("expected_outcome_t must be at least one")
    tolerance = _finite(tolerance, "tolerance")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    normalized_decisions = {
        str(agent_id): decision for agent_id, decision in decisions.items()
    }
    if set(pre_snapshots) != set(normalized_decisions):
        raise ValueError("pre-snapshot and decision cohorts differ")
    decision_t = expected_outcome_t - 1
    for agent_id, snapshot in pre_snapshots.items():
        if not isinstance(snapshot, FoundationEconomicSnapshot):
            raise TypeError("pre_snapshots must contain FoundationEconomicSnapshot")
        if snapshot.agent_id != agent_id or snapshot.timestamp != decision_t:
            raise ValueError(
                f"pre snapshot timestamp mismatch for {agent_id}: "
                f"observed {snapshot.timestamp}, decision_t={decision_t}"
            )

    post_snapshots = capture_foundation_snapshots(
        env,
        expected_timestamp=expected_outcome_t,
        current_decisions=normalized_decisions,
    )
    if set(post_snapshots) != set(pre_snapshots):
        raise ValueError("post-snapshot cohort differs from pre-snapshot cohort")

    tax_component = _env_component(env, "PeriodicBracketTax")
    taxes = getattr(tax_component, "taxes", None)
    if not isinstance(taxes, Sequence) or not taxes or not isinstance(taxes[-1], Mapping):
        raise ValueError("PeriodicBracketTax has no current tax record")
    tax_record = taxes[-1]
    world = env.world
    period = _positive_int(getattr(world, "period", None), "world.period")
    interest_applied = expected_outcome_t % period == 0

    transitions: dict[str, FoundationTransition] = {}
    for agent_id in sorted(pre_snapshots):
        pre = pre_snapshots[agent_id]
        post = post_snapshots[agent_id]
        decision = normalized_decisions[agent_id]
        foundation_action_for_decision(decision)
        agent_tax = tax_record.get(agent_id)
        if not isinstance(agent_tax, Mapping):
            raise ValueError(f"tax record missing agent {agent_id}")
        tax_paid = _nonnegative(agent_tax.get("tax_paid"), "tax_paid")
        transfer = _nonnegative(agent_tax.get("lump_sum"), "lump_sum_transfer")
        gross_income = post.cumulative_production - pre.cumulative_production
        if gross_income < -tolerance:
            raise ValueError("cumulative production decreased")
        if abs(gross_income) <= tolerance:
            gross_income = 0.0

        cash_before_consumption = pre.wealth + gross_income - tax_paid + transfer
        if cash_before_consumption < -tolerance:
            raise ValueError("cash before consumption is negative")
        requested_spend = decision.executed_consumption_rate * cash_before_consumption
        realized_spend = post.consumption_spend
        realized_quantity = post.consumption_quantity
        if realized_spend > requested_spend + tolerance:
            raise ValueError("realized consumption exceeds executed-rate budget")
        supply_rationed = realized_spend < requested_spend - tolerance
        cash_before_interest = cash_before_consumption - realized_spend
        if cash_before_interest < -tolerance:
            raise ValueError("cash before interest is negative")
        expected_interest = pre.interest_rate * cash_before_interest if interest_applied else 0.0
        interest_credit = post.wealth - cash_before_interest
        if abs(interest_credit) <= tolerance:
            interest_credit = 0.0
        if interest_credit < -tolerance:
            raise ValueError("derived interest credit is negative")
        if not math.isclose(
            interest_credit, expected_interest, rel_tol=0.0, abs_tol=tolerance
        ):
            raise ValueError(
                f"interest mismatch for agent {agent_id}: "
                f"derived {interest_credit}, expected {expected_interest}"
            )
        expected_wealth = cash_before_interest + interest_credit
        residual = post.wealth - expected_wealth
        if not math.isclose(post.wealth, expected_wealth, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(f"budget identity failed for agent {agent_id}")

        transitions[agent_id] = FoundationTransition(
            decision_t=decision_t,
            outcome_t=expected_outcome_t,
            agent_id=agent_id,
            pre=pre,
            post=post,
            gross_labor_income=gross_income,
            tax_paid=tax_paid,
            lump_sum_transfer=transfer,
            requested_consumption_spend=requested_spend,
            realized_consumption_spend=realized_spend,
            realized_consumption_quantity=realized_quantity,
            supply_rationed=supply_rationed,
            interest_applied=interest_applied,
            interest_credit=interest_credit,
            expected_wealth_post=expected_wealth,
            budget_residual=residual,
        )
    return transitions


def m0_snapshot_batches(
    transitions: Mapping[str, FoundationTransition],
    decisions: Mapping[Any, ActionDecision],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return fresh mapping batches accepted by ``EnvironmentLedger``."""

    normalized = {str(agent_id): decision for agent_id, decision in decisions.items()}
    if set(transitions) != set(normalized):
        raise ValueError("transition and decision cohorts differ")
    pre_batch: dict[str, dict[str, Any]] = {}
    post_batch: dict[str, dict[str, Any]] = {}
    for agent_id in sorted(transitions):
        transition = transitions[agent_id]
        pre_values = transition.to_m0_pre(normalized[agent_id]).to_dict()
        post_values = transition.to_m0_post().to_dict()
        pre_values.pop("period")
        pre_values.pop("agent_id")
        post_values.pop("period")
        post_values.pop("agent_id")
        pre_batch[agent_id] = pre_values
        post_batch[agent_id] = post_values
    return pre_batch, post_batch


__all__ = [
    "DEFAULT_CONSUMPTION_STEP",
    "DEFAULT_LABOR_STEP",
    "DEFAULT_MAX_LABOR_HOURS",
    "FoundationEconomicSnapshot",
    "FoundationTransition",
    "build_foundation_actions",
    "capture_foundation_snapshots",
    "derive_foundation_transitions",
    "foundation_action_for_decision",
    "locate_component",
    "m0_snapshot_batches",
    "prepare_foundation_env_config",
]
