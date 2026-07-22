import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import pytest

from verified_memory.budget import (
    BudgetExceeded,
    BudgetLimits,
    RunBudget,
    StopReason,
    UsageRecord,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_call_limit_reservation_and_dispatch_rollback() -> None:
    budget = RunBudget(BudgetLimits(max_calls=1), budget_id="test-budget")
    first = budget.reserve_call(label="decision")

    with pytest.raises(BudgetExceeded) as caught:
        budget.reserve_call(label="blocked")
    assert caught.value.reason is StopReason.CALL_LIMIT

    budget.rollback_call(first)
    replacement = budget.reserve_call(label="replacement")
    record = budget.complete_call(replacement, UsageRecord())

    assert record.label == "replacement"
    snapshot = budget.snapshot()
    assert snapshot.completed_calls == 1
    assert snapshot.active_calls == 0
    assert snapshot.rolled_back_calls == 1
    assert snapshot.stop_reasons == (StopReason.CALL_LIMIT,)


@pytest.mark.parametrize(
    ("limits", "estimate", "reason"),
    [
        (BudgetLimits(max_prompt_tokens=10), UsageRecord(prompt_tokens=11), StopReason.PROMPT_TOKEN_LIMIT),
        (
            BudgetLimits(max_completion_tokens=4),
            UsageRecord(completion_tokens=5),
            StopReason.COMPLETION_TOKEN_LIMIT,
        ),
        (BudgetLimits(max_total_tokens=7), UsageRecord(4, 4), StopReason.TOTAL_TOKEN_LIMIT),
        (BudgetLimits(max_cost_usd=0.20), UsageRecord(cost_usd=0.21), StopReason.COST_LIMIT),
    ],
)
def test_estimated_usage_is_rejected_before_dispatch(limits, estimate, reason) -> None:
    budget = RunBudget(limits)
    with pytest.raises(BudgetExceeded) as caught:
        budget.reserve_call(estimated_usage=estimate)
    assert reason in caught.value.reasons
    assert budget.snapshot().completed_calls == 0
    assert budget.snapshot().active_calls == 0


def test_reserved_tokens_block_concurrent_reservation_and_rollback_frees_them() -> None:
    budget = RunBudget(BudgetLimits(max_prompt_tokens=10))
    first = budget.reserve_call(estimated_usage=UsageRecord(prompt_tokens=6))

    with pytest.raises(BudgetExceeded) as caught:
        budget.reserve_call(estimated_usage=UsageRecord(prompt_tokens=5))
    assert caught.value.reason is StopReason.PROMPT_TOKEN_LIMIT

    budget.rollback_call(first)
    second = budget.reserve_call(estimated_usage=UsageRecord(prompt_tokens=5))
    budget.complete_call(second, UsageRecord(prompt_tokens=5))
    assert budget.snapshot().accounted_usage.prompt_tokens == 5


def test_actual_overage_is_accounted_before_cost_exception() -> None:
    budget = RunBudget(BudgetLimits(max_cost_usd=1.0))
    reservation = budget.reserve_call()

    with pytest.raises(BudgetExceeded) as caught:
        budget.complete_call(reservation, UsageRecord(10, 5, 1.25))

    assert caught.value.reason is StopReason.COST_LIMIT
    snapshot = caught.value.snapshot
    assert snapshot.completed_calls == 1
    assert snapshot.accounted_usage == UsageRecord(10, 5, 1.25)
    assert len(snapshot.completions) == 1


def test_exact_token_limits_allow_completion_but_stop_next_dispatch() -> None:
    budget = RunBudget(
        BudgetLimits(
            max_prompt_tokens=7,
            max_completion_tokens=5,
            max_total_tokens=12,
        )
    )
    reservation = budget.reserve_call(estimated_usage=UsageRecord(7, 5))
    budget.complete_call(reservation, UsageRecord(7, 5))

    assert budget.stop_reasons == (
        StopReason.PROMPT_TOKEN_LIMIT,
        StopReason.COMPLETION_TOKEN_LIMIT,
        StopReason.TOTAL_TOKEN_LIMIT,
    )
    with pytest.raises(BudgetExceeded):
        budget.reserve_call()


def test_elapsed_time_is_checked_on_reserve_and_after_completion() -> None:
    clock = FakeClock()
    budget = RunBudget(BudgetLimits(max_elapsed_seconds=5.0), clock=clock)
    reservation = budget.reserve_call()
    clock.advance(5.5)

    with pytest.raises(BudgetExceeded) as caught:
        budget.complete_call(reservation, UsageRecord(prompt_tokens=1))
    assert caught.value.reason is StopReason.ELAPSED_TIME_LIMIT
    assert caught.value.snapshot.accounted_usage.prompt_tokens == 1

    with pytest.raises(BudgetExceeded) as second:
        budget.reserve_call()
    assert second.value.reason is StopReason.ELAPSED_TIME_LIMIT


def test_records_are_immutable_and_snapshot_is_json_serializable() -> None:
    budget = RunBudget(BudgetLimits(max_calls=2), budget_id="serial-budget")
    reservation = budget.reserve_call(
        estimated_usage=UsageRecord(3, 2, 0.01),
        label="rule_generation",
        model="stub",
        tags={"month": 3, "agent": 0},
    )
    record = budget.complete_call(reservation, UsageRecord(4, 1, 0.012))

    with pytest.raises(FrozenInstanceError):
        record.label = "changed"
    with pytest.raises(FrozenInstanceError):
        record.usage = UsageRecord()

    payload = json.loads(budget.to_json())
    assert payload["budget_id"] == "serial-budget"
    assert payload["accounted_usage"]["total_tokens"] == 5
    assert payload["completions"][0]["tags"] == {"agent": "0", "month": "3"}


def test_thread_pool_never_exceeds_call_limit() -> None:
    budget = RunBudget(BudgetLimits(max_calls=17))

    def attempt(index: int) -> int | None:
        try:
            reservation = budget.reserve_call(tags={"index": index})
        except BudgetExceeded:
            return None
        budget.complete_call(reservation, UsageRecord(prompt_tokens=1))
        return reservation.reservation_id

    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(attempt, range(200)))

    successful = [value for value in results if value is not None]
    assert len(successful) == 17
    assert len(set(successful)) == 17
    snapshot = budget.snapshot()
    assert snapshot.completed_calls == 17
    assert snapshot.active_calls == 0
    assert snapshot.accounted_usage.prompt_tokens == 17


def test_thread_pool_respects_reserved_cost_limit() -> None:
    budget = RunBudget(BudgetLimits(max_calls=100, max_cost_usd=0.50))
    estimate = UsageRecord(cost_usd=0.10)

    def attempt(_: int) -> bool:
        try:
            reservation = budget.reserve_call(estimated_usage=estimate)
        except BudgetExceeded:
            return False
        budget.complete_call(reservation, estimate)
        return True

    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(attempt, range(100)))

    assert sum(results) == 5
    snapshot = budget.snapshot()
    assert snapshot.completed_calls == 5
    assert snapshot.accounted_usage.cost_usd == pytest.approx(0.50)
    assert snapshot.stop_reasons == (StopReason.COST_LIMIT,)


def test_invalid_limits_and_usage_are_rejected() -> None:
    with pytest.raises(ValueError):
        BudgetLimits(max_calls=-1)
    with pytest.raises(TypeError):
        BudgetLimits(max_total_tokens=1.5)
    with pytest.raises(ValueError):
        UsageRecord(prompt_tokens=-1)
    with pytest.raises(TypeError):
        UsageRecord(cost_usd=float("nan"))
