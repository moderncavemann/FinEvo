"""Evaluation-only utility and cash-flow accounting for the economic simulator.

This module deliberately has no dependency on the simulator.  Callers take explicit
snapshots immediately before and after ``env.step`` and pass them to
``EnvironmentLedger``.  In particular, the ledger never reads persistent simulator
fields such as ``endogenous["Labor"]`` or ``consumption["Coin"]`` whose NO-OP values
may describe an earlier period.

The module is an observer: it does not choose actions, mutate supplied mappings,
consume randomness, or alter environment rewards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral, Real
from typing import Any, Mapping


SCHEMA_VERSION = "m0.utility-ledger.v1"
NOT_APPLICABLE = "not_applicable"


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _nonnegative(name: str, value: Any) -> float:
    value = _finite_number(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")
    return value


def _unit_interval(name: str, value: Any) -> float:
    value = _finite_number(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must lie in [0, 1], got {value}")
    return value


def _period(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("period must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError("period must be nonnegative")
    return value


def _agent_id(value: Any) -> str:
    value = str(value)
    if not value:
        raise ValueError("agent_id must not be empty")
    return value


def _close(a: float, b: float, tolerance: float) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)


@dataclass(frozen=True, slots=True)
class UtilityConfig:
    r"""Immutable parameters for the ex-post realized utility index.

    ``consumption_scale`` makes realized product consumption dimensionless.
    ``inverse_frisch`` is :math:`\nu` in the labor exponent ``1 + nu``.
    ``discount_factor`` is per simulator period and must not exceed one.
    """

    rho: float = 1.0
    labor_weight: float = 1.0
    inverse_frisch: float = 1.0
    consumption_scale: float = 1.0
    max_labor_hours: float = 168.0
    discount_factor: float = 1.0
    budget_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        rho = _nonnegative("rho", self.rho)
        labor_weight = _nonnegative("labor_weight", self.labor_weight)
        inverse_frisch = _nonnegative("inverse_frisch", self.inverse_frisch)
        consumption_scale = _finite_number("consumption_scale", self.consumption_scale)
        max_labor_hours = _finite_number("max_labor_hours", self.max_labor_hours)
        discount_factor = _finite_number("discount_factor", self.discount_factor)
        budget_tolerance = _finite_number("budget_tolerance", self.budget_tolerance)

        if consumption_scale <= 0:
            raise ValueError("consumption_scale must be positive")
        if max_labor_hours <= 0:
            raise ValueError("max_labor_hours must be positive")
        if not 0 < discount_factor <= 1:
            raise ValueError("discount_factor must lie in (0, 1]")
        if budget_tolerance <= 0:
            raise ValueError("budget_tolerance must be positive")

        # Normalize accepted integer inputs to floats while preserving immutability.
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "labor_weight", labor_weight)
        object.__setattr__(self, "inverse_frisch", inverse_frisch)
        object.__setattr__(self, "consumption_scale", consumption_scale)
        object.__setattr__(self, "max_labor_hours", max_labor_hours)
        object.__setattr__(self, "discount_factor", discount_factor)
        object.__setattr__(self, "budget_tolerance", budget_tolerance)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UtilityBreakdown:
    consumption_utility: float
    labor_disutility: float
    flow_utility: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def realized_flow_utility(
    consumption_quantity: float,
    executed_labor_hours: float,
    config: UtilityConfig,
) -> UtilityBreakdown:
    """Compute realized shifted-CRRA consumption utility less labor disutility.

    The shifted CRRA term is finite at zero consumption:

    ``((1 + q/q0)**(1-rho) - 1) / (1-rho)`` for ``rho != 1`` and
    ``log1p(q/q0)`` for ``rho == 1``.
    """

    if not isinstance(config, UtilityConfig):
        raise TypeError("config must be a UtilityConfig")
    quantity = _nonnegative("consumption_quantity", consumption_quantity)
    labor = _nonnegative("executed_labor_hours", executed_labor_hours)
    if labor > config.max_labor_hours:
        raise ValueError(
            "executed_labor_hours exceeds configured max_labor_hours: "
            f"{labor} > {config.max_labor_hours}"
        )

    scaled_quantity = quantity / config.consumption_scale
    if math.isclose(config.rho, 1.0, rel_tol=0.0, abs_tol=1e-12):
        consumption_utility = math.log1p(scaled_quantity)
    else:
        one_minus_rho = 1.0 - config.rho
        # expm1/log1p is stable when rho is close to one or q is close to zero.
        consumption_utility = math.expm1(
            one_minus_rho * math.log1p(scaled_quantity)
        ) / one_minus_rho

    labor_share = labor / config.max_labor_hours
    labor_disutility = (
        config.labor_weight
        / (1.0 + config.inverse_frisch)
        * labor_share ** (1.0 + config.inverse_frisch)
    )
    flow_utility = consumption_utility - labor_disutility

    if not all(
        math.isfinite(value)
        for value in (consumption_utility, labor_disutility, flow_utility)
    ):
        raise ValueError("utility calculation produced a non-finite value")
    return UtilityBreakdown(
        consumption_utility=consumption_utility,
        labor_disutility=labor_disutility,
        flow_utility=flow_utility,
    )


@dataclass(frozen=True, slots=True)
class PreStepSnapshot:
    period: int
    agent_id: str
    wealth: float
    cumulative_production: float
    price: float
    interest_rate: float
    proposed_work_propensity: float
    proposed_consumption_fraction: float
    executed_labor_hours: float
    executed_consumption_rate: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "period", _period(self.period))
        object.__setattr__(self, "agent_id", _agent_id(self.agent_id))
        object.__setattr__(self, "wealth", _nonnegative("wealth", self.wealth))
        object.__setattr__(
            self,
            "cumulative_production",
            _nonnegative("cumulative_production", self.cumulative_production),
        )
        price = _finite_number("price", self.price)
        if price <= 0:
            raise ValueError("price must be positive")
        object.__setattr__(self, "price", price)
        object.__setattr__(
            self, "interest_rate", _nonnegative("interest_rate", self.interest_rate)
        )
        object.__setattr__(
            self,
            "proposed_work_propensity",
            _unit_interval("proposed_work_propensity", self.proposed_work_propensity),
        )
        object.__setattr__(
            self,
            "proposed_consumption_fraction",
            _unit_interval(
                "proposed_consumption_fraction", self.proposed_consumption_fraction
            ),
        )
        object.__setattr__(
            self,
            "executed_labor_hours",
            _nonnegative("executed_labor_hours", self.executed_labor_hours),
        )
        object.__setattr__(
            self,
            "executed_consumption_rate",
            _unit_interval("executed_consumption_rate", self.executed_consumption_rate),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PostStepSnapshot:
    period: int
    agent_id: str
    wealth: float
    cumulative_production: float
    tax_paid: float
    lump_sum_transfer: float
    realized_consumption_spend: float
    realized_consumption_quantity: float
    interest_applied: bool
    interest_credit: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "period", _period(self.period))
        object.__setattr__(self, "agent_id", _agent_id(self.agent_id))
        object.__setattr__(self, "wealth", _nonnegative("wealth", self.wealth))
        object.__setattr__(
            self,
            "cumulative_production",
            _nonnegative("cumulative_production", self.cumulative_production),
        )
        object.__setattr__(self, "tax_paid", _nonnegative("tax_paid", self.tax_paid))
        object.__setattr__(
            self,
            "lump_sum_transfer",
            _nonnegative("lump_sum_transfer", self.lump_sum_transfer),
        )
        object.__setattr__(
            self,
            "realized_consumption_spend",
            _nonnegative(
                "realized_consumption_spend", self.realized_consumption_spend
            ),
        )
        object.__setattr__(
            self,
            "realized_consumption_quantity",
            _nonnegative(
                "realized_consumption_quantity", self.realized_consumption_quantity
            ),
        )
        if not isinstance(self.interest_applied, bool):
            raise TypeError("interest_applied must be boolean")
        object.__setattr__(
            self,
            "interest_credit",
            _nonnegative("interest_credit", self.interest_credit),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BudgetLedgerRow:
    schema_version: str
    period: int
    agent_id: str
    wealth_pre: float
    wealth_post: float
    cumulative_production_pre: float
    cumulative_production_post: float
    gross_labor_income: float
    tax_paid: float
    lump_sum_transfer: float
    cash_before_consumption: float
    proposed_work_propensity: float
    proposed_labor_hours: float
    executed_labor_hours: float
    proposed_consumption_fraction: float
    executed_consumption_rate: float
    requested_consumption_spend: float
    realized_consumption_spend: float
    realized_consumption_quantity: float
    supply_rationed: bool
    price: float
    interest_rate: float
    interest_applied: bool
    interest_credit: float
    cash_before_interest: float
    expected_wealth_post: float
    budget_residual: float
    consumption_utility: float
    labor_disutility: float
    flow_utility: float
    discount_weight: float
    discounted_flow_utility: float
    borrowing_supported: bool
    debt_balance: None
    defaulted: None
    debt_default_status: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        _validate_json_value(result)
        return result

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, allow_nan=False, separators=(",", ":")
        )


def _validate_json_value(value: Any, path: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON value at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {path} is not a string")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


class DiscountedUtilityAccumulator:
    """Strict per-agent accumulator with exactly one contribution per period."""

    def __init__(self, discount_factor: float) -> None:
        discount_factor = _finite_number("discount_factor", discount_factor)
        if not 0 < discount_factor <= 1:
            raise ValueError("discount_factor must lie in (0, 1]")
        self._discount_factor = discount_factor
        self._next_period: dict[str, int] = {}
        self._discounted_total: dict[str, float] = {}
        self._undiscounted_total: dict[str, float] = {}

    @property
    def discount_factor(self) -> float:
        return self._discount_factor

    def add(self, agent_id: Any, period: int, flow_utility: float) -> float:
        agent_id = _agent_id(agent_id)
        period = _period(period)
        flow_utility = _finite_number("flow_utility", flow_utility)
        expected = self._next_period.get(agent_id, 0)
        if period != expected:
            raise ValueError(
                f"agent {agent_id} expected utility period {expected}, got {period}"
            )
        contribution = self._discount_factor**period * flow_utility
        self._discounted_total[agent_id] = (
            self._discounted_total.get(agent_id, 0.0) + contribution
        )
        self._undiscounted_total[agent_id] = (
            self._undiscounted_total.get(agent_id, 0.0) + flow_utility
        )
        self._next_period[agent_id] = period + 1
        return contribution

    def to_dict(self) -> dict[str, dict[str, float | int]]:
        return {
            agent_id: {
                "periods": self._next_period[agent_id],
                "undiscounted_total": self._undiscounted_total[agent_id],
                "discounted_total": self._discounted_total[agent_id],
            }
            for agent_id in sorted(self._next_period)
        }


def _snapshot_from_mapping(
    snapshot_type: type[PreStepSnapshot] | type[PostStepSnapshot],
    period: int,
    agent_id: str,
    values: Mapping[str, Any],
) -> PreStepSnapshot | PostStepSnapshot:
    if not isinstance(values, Mapping):
        raise TypeError(f"snapshot for agent {agent_id} must be a mapping")
    copied = dict(values)
    supplied_period = copied.pop("period", period)
    supplied_agent = copied.pop("agent_id", agent_id)
    if supplied_period != period:
        raise ValueError(
            f"snapshot period mismatch for agent {agent_id}: {supplied_period} != {period}"
        )
    if str(supplied_agent) != agent_id:
        raise ValueError(
            f"snapshot agent_id mismatch: {supplied_agent!r} != {agent_id!r}"
        )
    # Dataclass construction supplies strict missing/unknown-field validation.
    return snapshot_type(period=period, agent_id=agent_id, **copied)


def build_budget_ledger_row(
    pre: PreStepSnapshot,
    post: PostStepSnapshot,
    config: UtilityConfig,
) -> BudgetLedgerRow:
    """Validate one transition and return its immutable accounting row."""

    if not isinstance(pre, PreStepSnapshot):
        raise TypeError("pre must be a PreStepSnapshot")
    if not isinstance(post, PostStepSnapshot):
        raise TypeError("post must be a PostStepSnapshot")
    if not isinstance(config, UtilityConfig):
        raise TypeError("config must be a UtilityConfig")
    if (pre.period, pre.agent_id) != (post.period, post.agent_id):
        raise ValueError("pre/post period and agent_id must match")
    if pre.executed_labor_hours > config.max_labor_hours:
        raise ValueError("executed labor exceeds configured maximum")

    tolerance = config.budget_tolerance
    gross_income = post.cumulative_production - pre.cumulative_production
    if gross_income < -tolerance:
        raise ValueError("cumulative production decreased across the transition")
    if abs(gross_income) <= tolerance:
        gross_income = 0.0

    cash_before_consumption = (
        pre.wealth + gross_income - post.tax_paid + post.lump_sum_transfer
    )
    if cash_before_consumption < -tolerance:
        raise ValueError("taxes make cash before consumption negative")
    if abs(cash_before_consumption) <= tolerance:
        cash_before_consumption = 0.0

    requested_consumption_spend = (
        pre.executed_consumption_rate * cash_before_consumption
    )
    if post.realized_consumption_spend > requested_consumption_spend + tolerance:
        raise ValueError(
            "realized consumption spend exceeds the executed-rate budget: "
            f"{post.realized_consumption_spend} > {requested_consumption_spend}"
        )
    supply_rationed = (
        post.realized_consumption_spend
        < requested_consumption_spend - tolerance
    )

    cash_before_interest = cash_before_consumption - post.realized_consumption_spend
    if cash_before_interest < -tolerance:
        raise ValueError("realized consumption makes pre-interest cash negative")
    if abs(cash_before_interest) <= tolerance:
        cash_before_interest = 0.0

    expected_interest = (
        pre.interest_rate * cash_before_interest if post.interest_applied else 0.0
    )
    if not _close(post.interest_credit, expected_interest, tolerance):
        raise ValueError(
            "interest credit is inconsistent with the pre-step rate and cash base: "
            f"{post.interest_credit} != {expected_interest}"
        )

    expected_wealth_post = cash_before_interest + post.interest_credit
    residual = post.wealth - expected_wealth_post
    if not _close(post.wealth, expected_wealth_post, tolerance):
        raise ValueError(
            "budget identity failed: "
            f"post wealth {post.wealth}, expected {expected_wealth_post}, "
            f"residual {residual}"
        )

    utility = realized_flow_utility(
        post.realized_consumption_quantity,
        pre.executed_labor_hours,
        config,
    )
    discount_weight = config.discount_factor**pre.period
    return BudgetLedgerRow(
        schema_version=SCHEMA_VERSION,
        period=pre.period,
        agent_id=pre.agent_id,
        wealth_pre=pre.wealth,
        wealth_post=post.wealth,
        cumulative_production_pre=pre.cumulative_production,
        cumulative_production_post=post.cumulative_production,
        gross_labor_income=gross_income,
        tax_paid=post.tax_paid,
        lump_sum_transfer=post.lump_sum_transfer,
        cash_before_consumption=cash_before_consumption,
        proposed_work_propensity=pre.proposed_work_propensity,
        proposed_labor_hours=(
            pre.proposed_work_propensity * config.max_labor_hours
        ),
        executed_labor_hours=pre.executed_labor_hours,
        proposed_consumption_fraction=pre.proposed_consumption_fraction,
        executed_consumption_rate=pre.executed_consumption_rate,
        requested_consumption_spend=requested_consumption_spend,
        realized_consumption_spend=post.realized_consumption_spend,
        realized_consumption_quantity=post.realized_consumption_quantity,
        supply_rationed=supply_rationed,
        price=pre.price,
        interest_rate=pre.interest_rate,
        interest_applied=post.interest_applied,
        interest_credit=post.interest_credit,
        cash_before_interest=cash_before_interest,
        expected_wealth_post=expected_wealth_post,
        budget_residual=residual,
        consumption_utility=utility.consumption_utility,
        labor_disutility=utility.labor_disutility,
        flow_utility=utility.flow_utility,
        discount_weight=discount_weight,
        discounted_flow_utility=discount_weight * utility.flow_utility,
        borrowing_supported=False,
        debt_balance=None,
        defaulted=None,
        debt_default_status=NOT_APPLICABLE,
    )


class EnvironmentLedger:
    """Side-effect-free batch recorder designed to bracket ``env.step``.

    Example::

        ledger.capture_pre(period, explicit_pre_snapshots)
        env.step(actions)
        rows = ledger.capture_post(period, explicit_post_snapshots)

    The first pre-step batch fixes the agent cohort.  Subsequent batches must have
    identical agents and consecutive periods, which guarantees ``T * N`` alignment.
    """

    def __init__(self, config: UtilityConfig) -> None:
        if not isinstance(config, UtilityConfig):
            raise TypeError("config must be a UtilityConfig")
        self._config = config
        self._agent_ids: tuple[str, ...] | None = None
        self._next_period = 0
        self._pending_period: int | None = None
        self._pending: dict[str, PreStepSnapshot] = {}
        self._rows: list[BudgetLedgerRow] = []
        self._accumulator = DiscountedUtilityAccumulator(config.discount_factor)

    @property
    def config(self) -> UtilityConfig:
        return self._config

    @property
    def num_agents(self) -> int:
        return len(self._agent_ids or ())

    @property
    def num_periods(self) -> int:
        return self._next_period

    def capture_pre(
        self,
        period: int,
        snapshots: Mapping[Any, Mapping[str, Any]],
    ) -> tuple[PreStepSnapshot, ...]:
        period = _period(period)
        if period != self._next_period:
            raise ValueError(f"expected period {self._next_period}, got {period}")
        if self._pending_period is not None:
            raise RuntimeError("capture_post must complete the pending period first")
        if not isinstance(snapshots, Mapping) or not snapshots:
            raise ValueError("snapshots must be a non-empty agent mapping")

        normalized = {_agent_id(key): value for key, value in snapshots.items()}
        if len(normalized) != len(snapshots):
            raise ValueError("agent identifiers collide after string normalization")
        agent_ids = tuple(sorted(normalized))
        if self._agent_ids is None:
            cohort = agent_ids
        else:
            cohort = self._agent_ids
            if agent_ids != cohort:
                raise ValueError(
                    f"agent cohort changed: expected {cohort}, got {agent_ids}"
                )

        built = {
            agent_id: _snapshot_from_mapping(
                PreStepSnapshot, period, agent_id, normalized[agent_id]
            )
            for agent_id in cohort
        }
        self._agent_ids = cohort
        self._pending = built  # freshly constructed immutable values
        self._pending_period = period
        return tuple(built[agent_id] for agent_id in cohort)

    def capture_post(
        self,
        period: int,
        snapshots: Mapping[Any, Mapping[str, Any]],
    ) -> tuple[BudgetLedgerRow, ...]:
        period = _period(period)
        if self._pending_period != period:
            raise ValueError(
                f"no matching pre-step batch for period {period}; "
                f"pending={self._pending_period}"
            )
        if not isinstance(snapshots, Mapping):
            raise TypeError("snapshots must be an agent mapping")
        normalized = {_agent_id(key): value for key, value in snapshots.items()}
        if len(normalized) != len(snapshots):
            raise ValueError("agent identifiers collide after string normalization")
        agent_ids = tuple(sorted(normalized))
        if agent_ids != self._agent_ids:
            raise ValueError(
                f"post-step cohort mismatch: expected {self._agent_ids}, got {agent_ids}"
            )

        # Build and validate the entire batch before changing recorder state.
        posts = {
            agent_id: _snapshot_from_mapping(
                PostStepSnapshot, period, agent_id, normalized[agent_id]
            )
            for agent_id in self._agent_ids
        }
        rows = tuple(
            build_budget_ledger_row(
                self._pending[agent_id], posts[agent_id], self._config
            )
            for agent_id in self._agent_ids
        )

        for row in rows:
            contribution = self._accumulator.add(
                row.agent_id, row.period, row.flow_utility
            )
            if not _close(
                contribution,
                row.discounted_flow_utility,
                self._config.budget_tolerance,
            ):
                raise AssertionError("discount accumulator disagrees with ledger row")
        self._rows.extend(rows)
        self._pending = {}
        self._pending_period = None
        self._next_period += 1
        return rows

    def rows(self) -> tuple[BudgetLedgerRow, ...]:
        return tuple(self._rows)

    def records(self) -> list[dict[str, Any]]:
        return [row.to_dict() for row in self._rows]

    def utility_totals(self) -> dict[str, dict[str, float | int]]:
        return self._accumulator.to_dict()

    def finalize(self, expected_periods: int | None = None) -> tuple[BudgetLedgerRow, ...]:
        if self._pending_period is not None:
            raise RuntimeError(
                f"period {self._pending_period} has pre-step but no post-step snapshot"
            )
        if expected_periods is not None:
            expected_periods = _period(expected_periods)
            if self._next_period != expected_periods:
                raise ValueError(
                    f"expected {expected_periods} periods, recorded {self._next_period}"
                )
        expected_rows = self._next_period * self.num_agents
        if len(self._rows) != expected_rows:
            raise AssertionError(
                f"ledger alignment failed: {len(self._rows)} != "
                f"{self._next_period} * {self.num_agents}"
            )
        return self.rows()

    def to_jsonl(self) -> str:
        return "".join(f"{row.to_json()}\n" for row in self._rows)


__all__ = [
    "BudgetLedgerRow",
    "DiscountedUtilityAccumulator",
    "EnvironmentLedger",
    "NOT_APPLICABLE",
    "PostStepSnapshot",
    "PreStepSnapshot",
    "SCHEMA_VERSION",
    "UtilityBreakdown",
    "UtilityConfig",
    "build_budget_ledger_row",
    "realized_flow_utility",
]
