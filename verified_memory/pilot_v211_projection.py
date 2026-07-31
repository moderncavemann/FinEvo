"""Outcome-blind full-matrix budget projection for the V2.11 successor pilot.

This module deliberately contains no provider construction and no outcome
logic.  After capability and long-context preflight have terminalized, it
combines their ledger *actuals* with sealed fresh-p95 reservations for the
still-undispatched science matrix.  The resulting whole-matrix receipt must
pass before any A--D or cross-model cell may dispatch.

The V2.10.2 cumulative debits are immutable parent inputs.  They are included
in every hard-cap decision but are never reclassified as V2.11 effects.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


class PilotV211ProjectionError(ValueError):
    """Raised when the prospective matrix or its reservations are incomplete."""


V211_TOTAL_HARD_CAP_USD = 500.0
V211_TOTAL_HOSTED_COMPLETION_CAP = 7_500
V211_TOTAL_STORAGE_CAP_BYTES = 5_000_000_000

V2102_PARENT_COST_USD = 16.044922812500005
V2102_PARENT_HOSTED_COMPLETIONS = 816
V2102_PARENT_STORAGE_BYTES = 217_010_835

# The user's complete A--D mechanism micro-pilot, plus fresh capability and
# 2-agent x 12-month long-context preflight for GPT-5.2/GPT-5.6 and the
# registered three-seed GPT-5.6 full/no-memory family sentinel.
#
# Counts are provider calls after grouping Experiment D by seed.  The
# offline candidate-admission cells in C remain in the ITT matrix but issue
# zero provider calls.
V211_STAGE_CALLS: Mapping[str, Mapping[str, Mapping[str, int]]] = {
    "capability-gate": {
        "gpt52_main": {"action": 24, "semantic": 6},
        "gpt56_diagnostic": {"action": 24, "semantic": 6},
    },
    "long-context-preflight": {
        "gpt52_main": {"action": 24, "semantic": 8},
        "gpt56_diagnostic": {"action": 24, "semantic": 8},
    },
    "experiment-a": {
        "gpt52_main": {"action": 960, "semantic": 320},
    },
    "experiment-b": {
        "gpt52_main": {"action": 1_200, "semantic": 240},
    },
    "experiment-c": {
        "gpt52_main": {"action": 960, "semantic": 320},
    },
    "experiment-d": {
        "gpt52_main": {"action": 1_440, "semantic": 40},
    },
    "cross-model": {
        "gpt56_diagnostic": {"action": 288, "semantic": 48},
    },
}
V211_PREFLIGHT_STAGE_IDS = (
    "capability-gate",
    "long-context-preflight",
)
V211_SCIENCE_STAGE_IDS = (
    "experiment-a",
    "experiment-b",
    "experiment-c",
    "experiment-d",
    "cross-model",
)

# Ledger cells preserve the preregistered denominator.  D has 7 continuation
# arms plus 4 narrative fixtures across five paired seeds.  The five D seed
# groups, rather than 55 independent branches, determine storage reservations.
V211_STAGE_LEDGER_CELLS: Mapping[str, int] = {
    "parent-import": 1,
    "capability-gate": 2,
    "long-context-preflight": 2,
    "experiment-a": 20,
    "experiment-b": 25,
    "experiment-c": 25,
    "experiment-d": 55,
    "cross-model": 6,
}

V211_NEW_HOSTED_COMPLETIONS = 5_940
V211_PREFLIGHT_ACTUAL_COMPLETIONS = 124
V211_REMAINING_SCIENCE_COMPLETIONS = 5_816
V211_TOTAL_LEDGER_CELLS = 136
V211_REMAINING_SCIENCE_STORAGE_RESERVATION_BYTES = 1_920_000_000
V211_MODEL_SCIENCE_LEDGER_CELLS: Mapping[str, int] = {
    "gpt52_main": 125,
    "gpt56_diagnostic": 6,
}
V211_MODEL_SCIENCE_STORAGE_RESERVATION_BYTES: Mapping[str, int] = {
    # Existing runner reservations: A=400 MB, B=500 MB, C=500 MB, and
    # five grouped-D seed checkpoints at 80 MB each.
    "gpt52_main": 1_800_000_000,
    # Six cross-model cells at the ordinary 20 MB run reservation.
    "gpt56_diagnostic": 120_000_000,
}
V211_MODEL_GATE_REGISTERED_CALLS: Mapping[str, int] = {
    "gpt52_main": 62,
    "gpt56_diagnostic": 62,
}
V211_MODEL_CAPABILITY_CALLS: Mapping[str, int] = {
    "gpt52_main": 30,
    "gpt56_diagnostic": 30,
}


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PilotV211ProjectionError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise PilotV211ProjectionError(f"{name} must be finite and nonnegative")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PilotV211ProjectionError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PilotV211ProjectionError(
            f"{name} must be a nonnegative integer"
        )
    return value


def _reservation_fields(
    row: Any,
    *,
    model_id: str,
    call_kind: str,
) -> dict[str, float | int]:
    if not isinstance(row, Mapping):
        raise PilotV211ProjectionError(
            f"{model_id}.{call_kind} reservation must be a mapping"
        )
    expected = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
    }
    if set(row) != expected:
        raise PilotV211ProjectionError(
            f"{model_id}.{call_kind} reservation fields are incomplete"
        )
    prompt = _positive_integer(
        row["prompt_tokens"],
        f"{model_id}.{call_kind}.prompt_tokens",
    )
    completion = _positive_integer(
        row["completion_tokens"],
        f"{model_id}.{call_kind}.completion_tokens",
    )
    total = _positive_integer(
        row["total_tokens"],
        f"{model_id}.{call_kind}.total_tokens",
    )
    if total != prompt + completion:
        raise PilotV211ProjectionError(
            f"{model_id}.{call_kind}.total_tokens is not additive"
        )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cost_usd": _finite_nonnegative(
            row["cost_usd"],
            f"{model_id}.{call_kind}.cost_usd",
        ),
    }


@dataclass(frozen=True, slots=True)
class PilotV211Projection:
    """Whole-matrix projection and cumulative hard-cap decision."""

    stage_calls: Mapping[str, int]
    model_calls: Mapping[str, int]
    projected_stage_calls: Mapping[str, int]
    projected_model_calls: Mapping[str, int]
    eligible_model_ids: tuple[str, ...]
    gate_actual_calls_by_model: Mapping[str, int]
    no_go_ledger_cells_by_model: Mapping[str, int]
    registered_hosted_completions: int
    new_hosted_completions: int
    cumulative_hosted_completions: int
    new_cost_usd: float
    cumulative_cost_usd: float
    new_storage_bytes: int
    cumulative_storage_bytes: int
    pre_science_actual_usage: Mapping[str, Any]
    pre_science_actual_hosted_completions: int
    pre_science_actual_storage_bytes: int
    remaining_science_hosted_completions: int
    registered_remaining_science_hosted_completions: int
    remaining_science_storage_reservation_bytes: int
    registered_remaining_science_storage_reservation_bytes: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    go: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "finevo-pilot-v2.11-full-matrix-projection-v1",
            "stage_calls": dict(self.stage_calls),
            "model_calls": dict(self.model_calls),
            "projected_stage_calls": dict(self.projected_stage_calls),
            "projected_model_calls": dict(self.projected_model_calls),
            "eligible_model_ids": list(self.eligible_model_ids),
            "gate_actual_calls_by_model": dict(
                self.gate_actual_calls_by_model
            ),
            "no_go_ledger_cells_by_model": dict(
                self.no_go_ledger_cells_by_model
            ),
            "registered_hosted_completions": (
                self.registered_hosted_completions
            ),
            "new_hosted_completions": self.new_hosted_completions,
            "cumulative_hosted_completions": (
                self.cumulative_hosted_completions
            ),
            "new_cost_usd": self.new_cost_usd,
            "cumulative_cost_usd": self.cumulative_cost_usd,
            "new_storage_bytes": self.new_storage_bytes,
            "cumulative_storage_bytes": self.cumulative_storage_bytes,
            "pre_science_actual_usage": dict(
                self.pre_science_actual_usage
            ),
            "pre_science_actual_hosted_completions": (
                self.pre_science_actual_hosted_completions
            ),
            "pre_science_actual_storage_bytes": (
                self.pre_science_actual_storage_bytes
            ),
            "remaining_science_hosted_completions": (
                self.remaining_science_hosted_completions
            ),
            "registered_remaining_science_hosted_completions": (
                self.registered_remaining_science_hosted_completions
            ),
            "remaining_science_storage_reservation_bytes": (
                self.remaining_science_storage_reservation_bytes
            ),
            "registered_remaining_science_storage_reservation_bytes": (
                self.registered_remaining_science_storage_reservation_bytes
            ),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "hard_caps": {
                "cost_usd": V211_TOTAL_HARD_CAP_USD,
                "hosted_completions": V211_TOTAL_HOSTED_COMPLETION_CAP,
                "storage_bytes": V211_TOTAL_STORAGE_CAP_BYTES,
            },
            "parent_v2_10_2_debits": {
                "cost_usd": V2102_PARENT_COST_USD,
                "hosted_completions": V2102_PARENT_HOSTED_COMPLETIONS,
                "storage_bytes": V2102_PARENT_STORAGE_BYTES,
            },
            "go": self.go,
            "reasons": list(self.reasons),
            "claim_boundary": (
                "This post-preflight receipt uses gate ledger actuals plus "
                "fresh reserved-p95 science projections. It is an "
                "outcome-blind dispatch budget decision, not scientific "
                "evidence and not a reclassification of V2.10.2."
            ),
        }


def project_v211_full_matrix(
    reservations: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    pre_science_actual_usage: Mapping[str, Any],
    pre_science_actual_hosted_completions: int,
    pre_science_actual_storage_bytes: int,
    eligible_model_ids: Sequence[str] | None = None,
    gate_actual_calls_by_model: Mapping[str, int] | None = None,
    no_go_ledger_cells_by_model: Mapping[str, int] | None = None,
) -> PilotV211Projection:
    """Project the exact remaining matrix from fresh reserved-p95 rows.

    ``reservations`` must contain exactly the dispatch-eligible forward models
    and exactly ``action``/``semantic`` rows for each.  Each row is the
    already-reserved p95 (raw p95 times the frozen 1.25 multiplier), so no
    second multiplier is applied here.  Capability/preflight calls have
    already happened for both registered models by this point, so their exact
    ledger usage and storage are required separately and are never replaced
    by, or double-counted with, p95 estimates.
    """

    expected_models = {"gpt52_main", "gpt56_diagnostic"}
    if eligible_model_ids is None:
        eligible_models = tuple(sorted(expected_models))
    else:
        if isinstance(eligible_model_ids, (str, bytes)) or not isinstance(
            eligible_model_ids, Sequence
        ):
            raise PilotV211ProjectionError(
                "eligible_model_ids must be a sequence"
            )
        eligible_models = tuple(eligible_model_ids)
        if (
            any(not isinstance(value, str) for value in eligible_models)
            or len(eligible_models) != len(set(eligible_models))
            or not set(eligible_models) <= expected_models
        ):
            raise PilotV211ProjectionError(
                "eligible_model_ids must be unique registered model IDs"
            )
        eligible_models = tuple(sorted(eligible_models))
    eligible_set = set(eligible_models)

    if gate_actual_calls_by_model is None:
        gate_calls = dict(V211_MODEL_GATE_REGISTERED_CALLS)
    else:
        if (
            not isinstance(gate_actual_calls_by_model, Mapping)
            or set(gate_actual_calls_by_model) != expected_models
        ):
            raise PilotV211ProjectionError(
                "gate_actual_calls_by_model must contain both registered models"
            )
        gate_calls = {}
        for model_id in sorted(expected_models):
            value = gate_actual_calls_by_model[model_id]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != V211_MODEL_GATE_REGISTERED_CALLS[model_id]
            ):
                raise PilotV211ProjectionError(
                    f"{model_id} gate actuals must be exactly 62 calls"
                )
            gate_calls[model_id] = value

    expected_no_go_cells: dict[str, int] = {}
    for model_id in sorted(expected_models):
        count = 0
        if model_id not in eligible_set:
            count += V211_MODEL_SCIENCE_LEDGER_CELLS[model_id]
        expected_no_go_cells[model_id] = count
    if no_go_ledger_cells_by_model is None:
        no_go_cells = expected_no_go_cells
    else:
        if (
            not isinstance(no_go_ledger_cells_by_model, Mapping)
            or set(no_go_ledger_cells_by_model) != expected_models
        ):
            raise PilotV211ProjectionError(
                "no_go_ledger_cells_by_model must contain both registered models"
            )
        no_go_cells = {}
        for model_id in sorted(expected_models):
            value = no_go_ledger_cells_by_model[model_id]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise PilotV211ProjectionError(
                    f"{model_id} no-go ledger-cell count is invalid"
                )
            no_go_cells[model_id] = value
        if no_go_cells != expected_no_go_cells:
            raise PilotV211ProjectionError(
                "no-go ledger-cell accounting differs from model eligibility"
            )

    if not isinstance(reservations, Mapping) or set(reservations) != eligible_set:
        raise PilotV211ProjectionError(
            "reservations must contain exactly the dispatch-eligible models"
        )
    normalized: dict[str, dict[str, dict[str, float | int]]] = {}
    for model_id in eligible_models:
        by_kind = reservations[model_id]
        if not isinstance(by_kind, Mapping) or set(by_kind) != {
            "action",
            "semantic",
        }:
            raise PilotV211ProjectionError(
                f"{model_id} must contain exactly action and semantic"
            )
        normalized[model_id] = {
            call_kind: _reservation_fields(
                by_kind[call_kind],
                model_id=model_id,
                call_kind=call_kind,
            )
            for call_kind in ("action", "semantic")
        }

    actual_usage = _reservation_fields(
        pre_science_actual_usage,
        model_id="pre_science",
        call_kind="actual_usage",
    )
    actual_calls = _positive_integer(
        pre_science_actual_hosted_completions,
        "pre_science_actual_hosted_completions",
    )
    expected_actual_calls = sum(gate_calls.values())
    if actual_calls != expected_actual_calls:
        raise PilotV211ProjectionError(
            "post-gate authority requires exactly 124 terminal "
            "capability/preflight calls"
        )
    actual_storage = _nonnegative_integer(
        pre_science_actual_storage_bytes,
        "pre_science_actual_storage_bytes",
    )

    stage_calls: dict[str, int] = {}
    projected_stage_calls: dict[str, int] = {}
    model_calls = {model_id: 0 for model_id in sorted(expected_models)}
    projected_model_calls = dict(gate_calls)
    prompt_tokens = int(actual_usage["prompt_tokens"])
    completion_tokens = int(actual_usage["completion_tokens"])
    total_tokens = int(actual_usage["total_tokens"])
    new_cost_usd = float(actual_usage["cost_usd"])
    remaining_science_calls = 0
    for stage_id, stage_models in V211_STAGE_CALLS.items():
        current_stage_calls = 0
        current_projected_stage_calls = 0
        for model_id, call_counts in stage_models.items():
            if set(call_counts) != {"action", "semantic"}:
                raise PilotV211ProjectionError(
                    f"{stage_id}.{model_id} call kinds drifted"
                )
            for call_kind, count in call_counts.items():
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise PilotV211ProjectionError(
                        f"{stage_id}.{model_id}.{call_kind} count is invalid"
                    )
                current_stage_calls += count
                model_calls[model_id] += count
                if stage_id in V211_PREFLIGHT_STAGE_IDS:
                    continue
                if model_id not in eligible_set:
                    continue
                current_projected_stage_calls += count
                projected_model_calls[model_id] += count
                reservation = normalized[model_id][call_kind]
                prompt_tokens += count * int(reservation["prompt_tokens"])
                completion_tokens += count * int(
                    reservation["completion_tokens"]
                )
                total_tokens += count * int(reservation["total_tokens"])
                new_cost_usd += count * float(reservation["cost_usd"])
                remaining_science_calls += count
        stage_calls[stage_id] = current_stage_calls
        if stage_id == "capability-gate":
            current_projected_stage_calls = sum(
                V211_MODEL_CAPABILITY_CALLS.values()
            )
        elif stage_id == "long-context-preflight":
            current_projected_stage_calls = sum(
                gate_calls[model_id]
                - V211_MODEL_CAPABILITY_CALLS[model_id]
                for model_id in sorted(expected_models)
            )
        projected_stage_calls[stage_id] = current_projected_stage_calls

    registered_hosted_completions = sum(stage_calls.values())
    if registered_hosted_completions != V211_NEW_HOSTED_COMPLETIONS:
        raise PilotV211ProjectionError(
            "full-matrix hosted completion count drifted"
        )
    if sum(V211_STAGE_LEDGER_CELLS.values()) != V211_TOTAL_LEDGER_CELLS:
        raise PilotV211ProjectionError("full-matrix ledger denominator drifted")
    expected_remaining_science_calls = sum(
        model_calls[model_id] - V211_MODEL_GATE_REGISTERED_CALLS[model_id]
        for model_id in eligible_models
    )
    if remaining_science_calls != expected_remaining_science_calls:
        raise PilotV211ProjectionError(
            "remaining science hosted completion count drifted"
        )
    remaining_science_storage = sum(
        V211_MODEL_SCIENCE_STORAGE_RESERVATION_BYTES[model_id]
        for model_id in eligible_models
    )
    storage = actual_storage + remaining_science_storage
    new_hosted_completions = expected_actual_calls + remaining_science_calls
    if new_hosted_completions != sum(projected_stage_calls.values()):
        raise PilotV211ProjectionError(
            "projected stage call count differs from model eligibility"
        )
    if new_hosted_completions != sum(projected_model_calls.values()):
        raise PilotV211ProjectionError(
            "projected model call count differs from model eligibility"
        )
    cumulative_calls = V2102_PARENT_HOSTED_COMPLETIONS + new_hosted_completions
    cumulative_cost = V2102_PARENT_COST_USD + new_cost_usd
    cumulative_storage = V2102_PARENT_STORAGE_BYTES + storage
    reasons: list[str] = []
    if cumulative_cost > V211_TOTAL_HARD_CAP_USD:
        reasons.append("cumulative-cost-exceeds-hard-cap")
    if cumulative_calls > V211_TOTAL_HOSTED_COMPLETION_CAP:
        reasons.append("cumulative-hosted-completions-exceed-hard-cap")
    if cumulative_storage > V211_TOTAL_STORAGE_CAP_BYTES:
        reasons.append("cumulative-storage-exceeds-hard-cap")

    return PilotV211Projection(
        stage_calls=stage_calls,
        model_calls=model_calls,
        projected_stage_calls=projected_stage_calls,
        projected_model_calls=projected_model_calls,
        eligible_model_ids=eligible_models,
        gate_actual_calls_by_model=gate_calls,
        no_go_ledger_cells_by_model=no_go_cells,
        registered_hosted_completions=registered_hosted_completions,
        new_hosted_completions=new_hosted_completions,
        cumulative_hosted_completions=cumulative_calls,
        new_cost_usd=new_cost_usd,
        cumulative_cost_usd=cumulative_cost,
        new_storage_bytes=storage,
        cumulative_storage_bytes=cumulative_storage,
        pre_science_actual_usage=actual_usage,
        pre_science_actual_hosted_completions=actual_calls,
        pre_science_actual_storage_bytes=actual_storage,
        remaining_science_hosted_completions=remaining_science_calls,
        registered_remaining_science_hosted_completions=(
            V211_REMAINING_SCIENCE_COMPLETIONS
        ),
        remaining_science_storage_reservation_bytes=(
            remaining_science_storage
        ),
        registered_remaining_science_storage_reservation_bytes=(
            V211_REMAINING_SCIENCE_STORAGE_RESERVATION_BYTES
        ),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        go=not reasons,
        reasons=tuple(reasons),
    )


__all__ = [
    "PilotV211Projection",
    "PilotV211ProjectionError",
    "V2102_PARENT_COST_USD",
    "V2102_PARENT_HOSTED_COMPLETIONS",
    "V2102_PARENT_STORAGE_BYTES",
    "V211_NEW_HOSTED_COMPLETIONS",
    "V211_MODEL_CAPABILITY_CALLS",
    "V211_MODEL_GATE_REGISTERED_CALLS",
    "V211_MODEL_SCIENCE_LEDGER_CELLS",
    "V211_MODEL_SCIENCE_STORAGE_RESERVATION_BYTES",
    "V211_PREFLIGHT_ACTUAL_COMPLETIONS",
    "V211_PREFLIGHT_STAGE_IDS",
    "V211_REMAINING_SCIENCE_COMPLETIONS",
    "V211_REMAINING_SCIENCE_STORAGE_RESERVATION_BYTES",
    "V211_SCIENCE_STAGE_IDS",
    "V211_STAGE_CALLS",
    "V211_STAGE_LEDGER_CELLS",
    "V211_TOTAL_HARD_CAP_USD",
    "V211_TOTAL_HOSTED_COMPLETION_CAP",
    "V211_TOTAL_LEDGER_CELLS",
    "V211_TOTAL_STORAGE_CAP_BYTES",
    "project_v211_full_matrix",
]
