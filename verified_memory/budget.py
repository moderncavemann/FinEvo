"""Thread-safe API budget accounting for bounded experiment runs.

The budget has a deliberately two-phase interface:

1. :meth:`RunBudget.reserve_call` atomically reserves a dispatch slot and,
   optionally, a conservative token/cost estimate.
2. :meth:`RunBudget.complete_call` replaces that estimate with the provider's
   actual usage, or :meth:`RunBudget.rollback_call` releases it when dispatch
   never happened.

This separation matters for ``ThreadPoolExecutor`` callers.  Checking a limit
and incrementing a counter in separate operations permits every worker to pass
the same stale check.  ``RunBudget`` keeps both operations under one lock.
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Optional, Tuple


_FLOAT_EPSILON = 1e-12


class StopReason(str, Enum):
    """Stable machine-readable reasons why no further call may be dispatched."""

    CALL_LIMIT = "call_limit"
    PROMPT_TOKEN_LIMIT = "prompt_token_limit"
    COMPLETION_TOKEN_LIMIT = "completion_token_limit"
    TOTAL_TOKEN_LIMIT = "total_token_limit"
    COST_LIMIT = "cost_limit"
    ELAPSED_TIME_LIMIT = "elapsed_time_limit"
    CLOSED = "closed"


# More explicit alias for callers that prefer the longer public name.
BudgetStopReason = StopReason


@dataclass(frozen=True)
class BudgetLimits:
    """Hard limits for one run. ``None`` means that dimension is unlimited."""

    max_calls: Optional[int] = None
    max_prompt_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    max_total_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    max_elapsed_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        for name in (
            "max_calls",
            "max_prompt_tokens",
            "max_completion_tokens",
            "max_total_tokens",
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise TypeError(f"{name} must be an int or None")
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

        for name in ("max_cost_usd", "max_elapsed_seconds"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TypeError(f"{name} must be a finite number or None")
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_calls": self.max_calls,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "max_total_tokens": self.max_total_tokens,
            "max_cost_usd": self.max_cost_usd,
            "max_elapsed_seconds": self.max_elapsed_seconds,
        }


@dataclass(frozen=True)
class UsageRecord:
    """Immutable provider usage for one call or a reservation estimate."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "completion_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            isinstance(self.cost_usd, bool)
            or not isinstance(self.cost_usd, (int, float))
            or not math.isfinite(float(self.cost_usd))
        ):
            raise TypeError("cost_usd must be a finite number")
        if self.cost_usd < 0:
            raise ValueError("cost_usd must be non-negative")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": float(self.cost_usd),
        }


_ZERO_USAGE = UsageRecord()


def _add_usage(left: UsageRecord, right: UsageRecord) -> UsageRecord:
    return UsageRecord(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        cost_usd=float(left.cost_usd) + float(right.cost_usd),
    )


def _freeze_tags(tags: Optional[Mapping[str, object]]) -> Tuple[Tuple[str, str], ...]:
    if not tags:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in tags.items()))


@dataclass(frozen=True)
class CallReservation:
    """Opaque immutable proof that a caller owns one pre-dispatch slot."""

    budget_id: str
    reservation_id: int
    label: str
    model: str
    created_elapsed_seconds: float
    estimated_usage: UsageRecord
    tags: Tuple[Tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "budget_id": self.budget_id,
            "reservation_id": self.reservation_id,
            "label": self.label,
            "model": self.model,
            "created_elapsed_seconds": self.created_elapsed_seconds,
            "estimated_usage": self.estimated_usage.to_dict(),
            "tags": dict(self.tags),
        }


@dataclass(frozen=True)
class CompletionRecord:
    """Immutable accounting record produced after a provider call returns."""

    budget_id: str
    reservation_id: int
    label: str
    model: str
    started_elapsed_seconds: float
    finished_elapsed_seconds: float
    estimated_usage: UsageRecord
    usage: UsageRecord
    tags: Tuple[Tuple[str, str], ...] = ()

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self.finished_elapsed_seconds - self.started_elapsed_seconds)

    def to_dict(self) -> dict[str, object]:
        return {
            "budget_id": self.budget_id,
            "reservation_id": self.reservation_id,
            "label": self.label,
            "model": self.model,
            "started_elapsed_seconds": self.started_elapsed_seconds,
            "finished_elapsed_seconds": self.finished_elapsed_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_usage": self.estimated_usage.to_dict(),
            "usage": self.usage.to_dict(),
            "tags": dict(self.tags),
        }


@dataclass(frozen=True)
class BudgetSnapshot:
    """Immutable point-in-time view suitable for JSON serialization."""

    budget_id: str
    limits: BudgetLimits
    accounted_usage: UsageRecord
    reserved_usage: UsageRecord
    completed_calls: int
    active_calls: int
    rolled_back_calls: int
    elapsed_seconds: float
    stop_reasons: Tuple[StopReason, ...]
    active_reservations: Tuple[CallReservation, ...]
    completions: Tuple[CompletionRecord, ...]

    @property
    def effective_usage(self) -> UsageRecord:
        """Accounted usage plus conservative estimates for in-flight calls."""

        return _add_usage(self.accounted_usage, self.reserved_usage)

    @property
    def stopped(self) -> bool:
        return bool(self.stop_reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "budget_id": self.budget_id,
            "limits": self.limits.to_dict(),
            "accounted_usage": self.accounted_usage.to_dict(),
            "reserved_usage": self.reserved_usage.to_dict(),
            "effective_usage": self.effective_usage.to_dict(),
            "completed_calls": self.completed_calls,
            "active_calls": self.active_calls,
            "rolled_back_calls": self.rolled_back_calls,
            "elapsed_seconds": self.elapsed_seconds,
            "stopped": self.stopped,
            "stop_reasons": [reason.value for reason in self.stop_reasons],
            "active_reservations": [item.to_dict() for item in self.active_reservations],
            "completions": [item.to_dict() for item in self.completions],
        }


class BudgetExceeded(RuntimeError):
    """Raised when a dispatch or completed call violates a run budget."""

    def __init__(self, reasons: Tuple[StopReason, ...], snapshot: BudgetSnapshot):
        if not reasons:
            raise ValueError("BudgetExceeded requires at least one stop reason")
        self.reasons = tuple(reasons)
        self.reason = self.reasons[0]
        self.snapshot = snapshot
        joined = ", ".join(reason.value for reason in self.reasons)
        super().__init__(f"run budget exhausted: {joined}")


# Backward-friendly explicit suffix without creating a second exception type.
BudgetExceededError = BudgetExceeded


class RunBudget:
    """Atomic, thread-safe accounting for one bounded provider run."""

    def __init__(
        self,
        limits: BudgetLimits,
        *,
        clock: Callable[[], float] = time.monotonic,
        budget_id: Optional[str] = None,
    ) -> None:
        if not isinstance(limits, BudgetLimits):
            raise TypeError("limits must be a BudgetLimits instance")
        self.limits = limits
        self.budget_id = budget_id or uuid.uuid4().hex
        self._clock = clock
        self._started_at = float(clock())
        self._lock = threading.RLock()
        self._next_reservation_id = 1
        self._active: dict[int, CallReservation] = {}
        self._completions: list[CompletionRecord] = []
        self._accounted_usage = _ZERO_USAGE
        self._rolled_back_calls = 0
        self._closed = False

    def _elapsed_locked(self) -> float:
        return max(0.0, float(self._clock()) - self._started_at)

    def _reserved_usage_locked(self) -> UsageRecord:
        usage = _ZERO_USAGE
        for reservation in self._active.values():
            usage = _add_usage(usage, reservation.estimated_usage)
        return usage

    def _effective_usage_locked(self) -> UsageRecord:
        return _add_usage(self._accounted_usage, self._reserved_usage_locked())

    def _stop_reasons_locked(self) -> Tuple[StopReason, ...]:
        reasons: list[StopReason] = []
        limits = self.limits
        usage = self._effective_usage_locked()
        calls = len(self._completions) + len(self._active)

        if self._closed:
            reasons.append(StopReason.CLOSED)
        if limits.max_calls is not None and calls >= limits.max_calls:
            reasons.append(StopReason.CALL_LIMIT)
        if (
            limits.max_prompt_tokens is not None
            and usage.prompt_tokens >= limits.max_prompt_tokens
        ):
            reasons.append(StopReason.PROMPT_TOKEN_LIMIT)
        if (
            limits.max_completion_tokens is not None
            and usage.completion_tokens >= limits.max_completion_tokens
        ):
            reasons.append(StopReason.COMPLETION_TOKEN_LIMIT)
        if limits.max_total_tokens is not None and usage.total_tokens >= limits.max_total_tokens:
            reasons.append(StopReason.TOTAL_TOKEN_LIMIT)
        if (
            limits.max_cost_usd is not None
            and float(usage.cost_usd) >= float(limits.max_cost_usd) - _FLOAT_EPSILON
        ):
            reasons.append(StopReason.COST_LIMIT)
        if (
            limits.max_elapsed_seconds is not None
            and self._elapsed_locked() >= float(limits.max_elapsed_seconds)
        ):
            reasons.append(StopReason.ELAPSED_TIME_LIMIT)
        return tuple(reasons)

    def _strict_violations_locked(self) -> Tuple[StopReason, ...]:
        """Return overages caused by an already-dispatched call.

        Reaching a limit exactly is valid for that call but blocks the next one.
        Exceeding a token/cost limit, or finishing after the time limit, raises
        immediately after the actual usage has been durably accounted.
        """

        reasons: list[StopReason] = []
        limits = self.limits
        usage = self._effective_usage_locked()
        if (
            limits.max_prompt_tokens is not None
            and usage.prompt_tokens > limits.max_prompt_tokens
        ):
            reasons.append(StopReason.PROMPT_TOKEN_LIMIT)
        if (
            limits.max_completion_tokens is not None
            and usage.completion_tokens > limits.max_completion_tokens
        ):
            reasons.append(StopReason.COMPLETION_TOKEN_LIMIT)
        if limits.max_total_tokens is not None and usage.total_tokens > limits.max_total_tokens:
            reasons.append(StopReason.TOTAL_TOKEN_LIMIT)
        if (
            limits.max_cost_usd is not None
            and float(usage.cost_usd) > float(limits.max_cost_usd) + _FLOAT_EPSILON
        ):
            reasons.append(StopReason.COST_LIMIT)
        if (
            limits.max_elapsed_seconds is not None
            and self._elapsed_locked() > float(limits.max_elapsed_seconds)
        ):
            reasons.append(StopReason.ELAPSED_TIME_LIMIT)
        return tuple(reasons)

    def _snapshot_locked(self) -> BudgetSnapshot:
        active = tuple(self._active[key] for key in sorted(self._active))
        return BudgetSnapshot(
            budget_id=self.budget_id,
            limits=self.limits,
            accounted_usage=self._accounted_usage,
            reserved_usage=self._reserved_usage_locked(),
            completed_calls=len(self._completions),
            active_calls=len(active),
            rolled_back_calls=self._rolled_back_calls,
            elapsed_seconds=self._elapsed_locked(),
            stop_reasons=self._stop_reasons_locked(),
            active_reservations=active,
            completions=tuple(self._completions),
        )

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return self._snapshot_locked()

    @property
    def stopped(self) -> bool:
        return self.snapshot().stopped

    @property
    def stop_reasons(self) -> Tuple[StopReason, ...]:
        return self.snapshot().stop_reasons

    def reserve_call(
        self,
        *,
        estimated_usage: Optional[UsageRecord] = None,
        label: str = "",
        model: str = "",
        tags: Optional[Mapping[str, object]] = None,
    ) -> CallReservation:
        """Atomically reserve capacity before submitting work to an executor."""

        estimate = estimated_usage or _ZERO_USAGE
        if not isinstance(estimate, UsageRecord):
            raise TypeError("estimated_usage must be a UsageRecord or None")

        with self._lock:
            reasons = self._stop_reasons_locked()
            if reasons:
                raise BudgetExceeded(reasons, self._snapshot_locked())

            prospective = _add_usage(self._effective_usage_locked(), estimate)
            prospective_calls = len(self._completions) + len(self._active) + 1
            reasons_list: list[StopReason] = []
            if self.limits.max_calls is not None and prospective_calls > self.limits.max_calls:
                reasons_list.append(StopReason.CALL_LIMIT)
            if (
                self.limits.max_prompt_tokens is not None
                and prospective.prompt_tokens > self.limits.max_prompt_tokens
            ):
                reasons_list.append(StopReason.PROMPT_TOKEN_LIMIT)
            if (
                self.limits.max_completion_tokens is not None
                and prospective.completion_tokens > self.limits.max_completion_tokens
            ):
                reasons_list.append(StopReason.COMPLETION_TOKEN_LIMIT)
            if (
                self.limits.max_total_tokens is not None
                and prospective.total_tokens > self.limits.max_total_tokens
            ):
                reasons_list.append(StopReason.TOTAL_TOKEN_LIMIT)
            if (
                self.limits.max_cost_usd is not None
                and float(prospective.cost_usd)
                > float(self.limits.max_cost_usd) + _FLOAT_EPSILON
            ):
                reasons_list.append(StopReason.COST_LIMIT)
            if reasons_list:
                raise BudgetExceeded(tuple(reasons_list), self._snapshot_locked())

            reservation = CallReservation(
                budget_id=self.budget_id,
                reservation_id=self._next_reservation_id,
                label=str(label),
                model=str(model),
                created_elapsed_seconds=self._elapsed_locked(),
                estimated_usage=estimate,
                tags=_freeze_tags(tags),
            )
            self._next_reservation_id += 1
            self._active[reservation.reservation_id] = reservation
            return reservation

    def _pop_reservation_locked(self, reservation: CallReservation) -> CallReservation:
        if not isinstance(reservation, CallReservation):
            raise TypeError("reservation must be a CallReservation")
        if reservation.budget_id != self.budget_id:
            raise ValueError("reservation belongs to a different budget")
        active = self._active.get(reservation.reservation_id)
        if active is None:
            raise KeyError(f"reservation {reservation.reservation_id} is not active")
        if active != reservation:
            raise ValueError("reservation contents do not match the active reservation")
        return self._active.pop(reservation.reservation_id)

    def complete_call(
        self,
        reservation: CallReservation,
        usage: UsageRecord,
    ) -> CompletionRecord:
        """Commit actual usage and return an immutable completion record.

        Actual usage is recorded before an overage exception is raised, so an
        unexpectedly expensive response can never disappear from the ledger.
        """

        if not isinstance(usage, UsageRecord):
            raise TypeError("usage must be a UsageRecord")
        with self._lock:
            active = self._pop_reservation_locked(reservation)
            finished = self._elapsed_locked()
            record = CompletionRecord(
                budget_id=self.budget_id,
                reservation_id=active.reservation_id,
                label=active.label,
                model=active.model,
                started_elapsed_seconds=active.created_elapsed_seconds,
                finished_elapsed_seconds=finished,
                estimated_usage=active.estimated_usage,
                usage=usage,
                tags=active.tags,
            )
            self._accounted_usage = _add_usage(self._accounted_usage, usage)
            self._completions.append(record)
            violations = self._strict_violations_locked()
            if violations:
                raise BudgetExceeded(violations, self._snapshot_locked())
            return record

    def rollback_call(self, reservation: CallReservation) -> None:
        """Release a reservation when dispatch itself failed and used no API quota."""

        with self._lock:
            self._pop_reservation_locked(reservation)
            self._rolled_back_calls += 1

    def ensure_available(self) -> None:
        """Raise :class:`BudgetExceeded` if a new dispatch is currently forbidden."""

        with self._lock:
            reasons = self._stop_reasons_locked()
            if reasons:
                raise BudgetExceeded(reasons, self._snapshot_locked())

    def close(self) -> None:
        """Prevent future reservations without altering in-flight accounting."""

        with self._lock:
            self._closed = True

    def to_dict(self) -> dict[str, object]:
        return self.snapshot().to_dict()

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


__all__ = [
    "BudgetExceeded",
    "BudgetExceededError",
    "BudgetLimits",
    "BudgetSnapshot",
    "BudgetStopReason",
    "CallReservation",
    "CompletionRecord",
    "RunBudget",
    "StopReason",
    "UsageRecord",
]
