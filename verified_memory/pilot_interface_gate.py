"""Outcome-blind provider interface gates for prospective pilot contracts.

The historical V2.10.2 contract intentionally remains readable and unchanged.
This module is for later contracts that must prove their request/output
interface has enough capacity *before* constructing a paid provider.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


class PilotInterfaceGateError(ValueError):
    """Raised when a sealed interface reservation is malformed."""


def _finite_number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PilotInterfaceGateError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise PilotInterfaceGateError(
            f"{name} must be finite and at least {minimum}"
        )
    return number


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PilotInterfaceGateError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class CompletionCapacityGate:
    """Auditable result of comparing a wire cap with sealed observed usage."""

    call_kind: str
    wire_cap_tokens: int
    raw_p95_completion_tokens: int
    reserved_p95_completion_tokens: int
    reserve_multiplier: float
    sample_count: int
    minimum_sample_count: int
    minimum_headroom_fraction: float
    headroom_tokens: int
    headroom_fraction: float
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_kind": self.call_kind,
            "wire_cap_tokens": self.wire_cap_tokens,
            "raw_p95_completion_tokens": self.raw_p95_completion_tokens,
            "reserved_p95_completion_tokens": (
                self.reserved_p95_completion_tokens
            ),
            "reserve_multiplier": self.reserve_multiplier,
            "sample_count": self.sample_count,
            "minimum_sample_count": self.minimum_sample_count,
            "minimum_headroom_fraction": self.minimum_headroom_fraction,
            "headroom_tokens": self.headroom_tokens,
            "headroom_fraction": self.headroom_fraction,
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class InterfaceSampleGate:
    """Terminal sample and capacity decision for one model/call kind."""

    call_kind: str
    expected_sample_count: int
    observed_sample_count: int
    prompt_tier_ceiling_tokens: int
    maximum_prompt_tokens: int
    capacity: CompletionCapacityGate
    passed: bool
    reasons: tuple[str, ...]
    failed_sample_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_kind": self.call_kind,
            "expected_sample_count": self.expected_sample_count,
            "observed_sample_count": self.observed_sample_count,
            "prompt_tier_ceiling_tokens": self.prompt_tier_ceiling_tokens,
            "maximum_prompt_tokens": self.maximum_prompt_tokens,
            "capacity": self.capacity.to_dict(),
            "passed": self.passed,
            "reasons": list(self.reasons),
            "failed_sample_indices": list(self.failed_sample_indices),
        }


def completion_capacity_gate(
    *,
    call_kind: str,
    wire_cap_tokens: int,
    reservation: Mapping[str, Any],
    minimum_sample_count: int,
    minimum_headroom_fraction: float = 0.25,
) -> CompletionCapacityGate:
    """Compare a proposed request cap with an outcome-blind P95 reservation.

    Two independent checks are deliberate:

    * the request cap must cover the already-reserved P95 (raw P95 times the
      sealed reserve multiplier); and
    * the raw P95 must not sit too close to the hard wire cap.

    The second check catches the V2.10.2 failure mode where successful
    completions clustered just below the cap while later reasoning-only
    completions exhausted it without producing visible JSON.
    """

    if not isinstance(call_kind, str) or not call_kind.strip():
        raise PilotInterfaceGateError("call_kind must be non-empty text")
    wire_cap = _positive_integer(wire_cap_tokens, "wire_cap_tokens")
    minimum_samples = _positive_integer(
        minimum_sample_count, "minimum_sample_count"
    )
    minimum_headroom = _finite_number(
        minimum_headroom_fraction,
        "minimum_headroom_fraction",
    )
    if minimum_headroom >= 1.0:
        raise PilotInterfaceGateError(
            "minimum_headroom_fraction must be smaller than 1"
        )
    if not isinstance(reservation, Mapping):
        raise PilotInterfaceGateError("reservation must be a mapping")
    expected_keys = {
        "raw_p95",
        "reserve_multiplier",
        "reserved_p95",
        "sample_count",
    }
    if set(reservation) != expected_keys:
        raise PilotInterfaceGateError(
            "reservation must contain exactly raw_p95, reserve_multiplier, "
            "reserved_p95, and sample_count"
        )
    raw = reservation["raw_p95"]
    reserved = reservation["reserved_p95"]
    if not isinstance(raw, Mapping) or not isinstance(reserved, Mapping):
        raise PilotInterfaceGateError(
            "raw_p95 and reserved_p95 must be mappings"
        )
    raw_completion = _finite_number(
        raw.get("completion_tokens"),
        "raw_p95.completion_tokens",
        minimum=1.0,
    )
    reserved_completion = _positive_integer(
        reserved.get("completion_tokens"),
        "reserved_p95.completion_tokens",
    )
    multiplier = _finite_number(
        reservation["reserve_multiplier"],
        "reserve_multiplier",
        minimum=1.0,
    )
    if multiplier <= 1.0:
        raise PilotInterfaceGateError(
            "reserve_multiplier must be greater than 1"
        )
    expected_reserved = math.ceil(raw_completion * multiplier)
    if reserved_completion != expected_reserved:
        raise PilotInterfaceGateError(
            "reserved_p95.completion_tokens does not equal "
            "ceil(raw_p95.completion_tokens * reserve_multiplier)"
        )
    sample_count = _positive_integer(
        reservation["sample_count"], "sample_count"
    )

    raw_ceiling = math.ceil(raw_completion)
    headroom_tokens = wire_cap - raw_ceiling
    headroom_fraction = headroom_tokens / wire_cap
    reasons: list[str] = []
    if sample_count < minimum_samples:
        reasons.append("insufficient-samples")
    if wire_cap < reserved_completion:
        reasons.append("wire-cap-below-reserved-p95")
    if headroom_fraction < minimum_headroom:
        reasons.append("raw-p95-too-close-to-wire-cap")

    return CompletionCapacityGate(
        call_kind=call_kind.strip(),
        wire_cap_tokens=wire_cap,
        raw_p95_completion_tokens=raw_ceiling,
        reserved_p95_completion_tokens=reserved_completion,
        reserve_multiplier=multiplier,
        sample_count=sample_count,
        minimum_sample_count=minimum_samples,
        minimum_headroom_fraction=minimum_headroom,
        headroom_tokens=headroom_tokens,
        headroom_fraction=headroom_fraction,
        passed=not reasons,
        reasons=tuple(reasons),
    )


_INTERFACE_SAMPLE_FIELDS = frozenset(
    {
        "finish_reason",
        "response_completed",
        "output_disposition",
        "error_type",
        "parse_success",
        "clipped",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "visible_completion_tokens",
    }
)


def _interface_sample(
    value: Any,
    *,
    index: int,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if not isinstance(value, Mapping) or set(value) != _INTERFACE_SAMPLE_FIELDS:
        raise PilotInterfaceGateError(
            f"interface sample {index} must contain the exact frozen fields"
        )
    prompt = _positive_integer(
        value["prompt_tokens"],
        f"interface sample {index}.prompt_tokens",
    )
    completion = _positive_integer(
        value["completion_tokens"],
        f"interface sample {index}.completion_tokens",
    )
    reasoning = value["reasoning_tokens"]
    visible = value["visible_completion_tokens"]
    for field, item in (
        ("reasoning_tokens", reasoning),
        ("visible_completion_tokens", visible),
    ):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise PilotInterfaceGateError(
                f"interface sample {index}.{field} must be a "
                "nonnegative integer"
            )
    if reasoning + visible != completion:
        raise PilotInterfaceGateError(
            f"interface sample {index} visible/reasoning usage is not additive"
        )
    response_completed = value["response_completed"]
    parse_success = value["parse_success"]
    clipped = value["clipped"]
    if response_completed is not None and not isinstance(
        response_completed, bool
    ):
        raise PilotInterfaceGateError(
            f"interface sample {index}.response_completed must be boolean or null"
        )
    if not isinstance(parse_success, bool) or not isinstance(clipped, bool):
        raise PilotInterfaceGateError(
            f"interface sample {index} parse/clipping fields must be boolean"
        )
    for field in (
        "finish_reason",
        "output_disposition",
        "error_type",
    ):
        item = value[field]
        if item is not None and not isinstance(item, str):
            raise PilotInterfaceGateError(
                f"interface sample {index}.{field} must be text or null"
            )

    reasons: list[str] = []
    if value["error_type"] is not None:
        reasons.append("provider-error")
    if value["finish_reason"] != "stop":
        reasons.append("non-stop-finish")
    if response_completed is not True:
        reasons.append("response-not-complete")
    if value["output_disposition"] != "accepted":
        reasons.append("output-not-accepted")
    if parse_success is not True:
        reasons.append("parse-not-exact")
    if clipped is True:
        reasons.append("action-clipped")
    if visible == 0:
        reasons.append("no-visible-output")
    return (
        {
            **dict(value),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "reasoning_tokens": reasoning,
            "visible_completion_tokens": visible,
        },
        tuple(reasons),
    )


def interface_sample_gate(
    *,
    call_kind: str,
    wire_cap_tokens: int,
    reservation: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    expected_sample_count: int,
    prompt_tier_ceiling_tokens: int = 200_000,
    minimum_headroom_fraction: float = 0.25,
) -> InterfaceSampleGate:
    """Fail closed over every capability/preflight denominator sample.

    P95 alone cannot reveal a reasoning-only truncation if failed calls were
    omitted from the projection.  This gate therefore requires the exact
    registered sample count and verifies every terminal disposition before
    accepting the capacity result.  ``prompt_tier_ceiling_tokens`` is strict:
    an observation equal to the ceiling is rejected so later prompt growth
    cannot silently enter a different pricing tier.
    """

    expected = _positive_integer(
        expected_sample_count,
        "expected_sample_count",
    )
    tier_ceiling = _positive_integer(
        prompt_tier_ceiling_tokens,
        "prompt_tier_ceiling_tokens",
    )
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise PilotInterfaceGateError("samples must be a sequence")

    normalized: list[dict[str, Any]] = []
    failed_indices: list[int] = []
    sample_failure_reasons: set[str] = set()
    for index, row in enumerate(samples):
        normalized_row, row_reasons = _interface_sample(row, index=index)
        normalized.append(normalized_row)
        if row_reasons:
            failed_indices.append(index)
            sample_failure_reasons.update(row_reasons)

    reasons: list[str] = []
    if len(normalized) != expected:
        reasons.append("sample-count-mismatch")
    reasons.extend(sorted(sample_failure_reasons))
    maximum_prompt_tokens = max(
        (int(row["prompt_tokens"]) for row in normalized),
        default=0,
    )
    if any(
        int(row["prompt_tokens"]) >= tier_ceiling for row in normalized
    ):
        reasons.append("prompt-tier-ceiling-reached")
    if any(
        int(row["completion_tokens"]) > wire_cap_tokens
        for row in normalized
    ):
        reasons.append("completion-exceeds-wire-cap")

    capacity = completion_capacity_gate(
        call_kind=call_kind,
        wire_cap_tokens=wire_cap_tokens,
        reservation=reservation,
        minimum_sample_count=expected,
        minimum_headroom_fraction=minimum_headroom_fraction,
    )
    reasons.extend(capacity.reasons)
    # Preserve deterministic, non-duplicated reason order for receipts.
    unique_reasons = tuple(dict.fromkeys(reasons))
    return InterfaceSampleGate(
        call_kind=call_kind,
        expected_sample_count=expected,
        observed_sample_count=len(normalized),
        prompt_tier_ceiling_tokens=tier_ceiling,
        maximum_prompt_tokens=maximum_prompt_tokens,
        capacity=capacity,
        passed=not unique_reasons,
        reasons=unique_reasons,
        failed_sample_indices=tuple(failed_indices),
    )


__all__ = [
    "CompletionCapacityGate",
    "InterfaceSampleGate",
    "PilotInterfaceGateError",
    "completion_capacity_gate",
    "interface_sample_gate",
]
