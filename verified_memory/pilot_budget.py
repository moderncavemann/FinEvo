"""Durable cross-stage budget accounting for the FinEvo pilot.

``RunBudget`` protects one provider run.  This module protects the entire
preregistered matrix across restarts.  Reservations are idempotent and the
non-automatic USD reserve is excluded from dispatchable capacity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


PILOT_BUDGET_SCHEMA_VERSION = "finevo-pilot-budget-ledger-v1"
PILOT_BUDGET_SCHEMA_VERSION_V2 = "finevo-pilot-budget-ledger-v2"
DEFAULT_STAGE_USD_CAPS = {
    "capability_preflight": 2.0,
    "calibration": 3.0,
    "core": 13.0,
    "cross_model": 6.0,
    "manual_reserve": 1.0,
}
TERMINAL_STATUSES = frozenset(
    {"complete", "failed", "budget-stopped", "integrity-stopped"}
)


class PilotBudgetError(RuntimeError):
    """Raised before dispatch when a global or stage cap would be exceeded."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(value)


@dataclass(frozen=True, slots=True)
class PilotBudgetCaps:
    total_usd: float = 25.0
    max_completions: int = 7_500
    completion_scope: str = "hosted-api-only"
    max_storage_bytes: int = 5_000_000_000
    stage_usd_caps: Mapping[str, float] | None = None
    automatic_reserve_usd: float = 1.0

    def __post_init__(self) -> None:
        total = _finite_nonnegative(self.total_usd, "total_usd")
        completions = _nonnegative_int(self.max_completions, "max_completions")
        storage = _nonnegative_int(self.max_storage_bytes, "max_storage_bytes")
        if self.completion_scope != "hosted-api-only":
            raise ValueError("completion_scope must be hosted-api-only")
        reserve = _finite_nonnegative(
            self.automatic_reserve_usd, "automatic_reserve_usd"
        )
        if total <= 0 or completions <= 0 or storage <= 0:
            raise ValueError("pilot budget caps must be positive")
        if reserve >= total:
            raise ValueError("automatic reserve must be smaller than total USD")
        source = (
            DEFAULT_STAGE_USD_CAPS
            if self.stage_usd_caps is None
            else self.stage_usd_caps
        )
        if not isinstance(source, Mapping) or not source:
            raise TypeError("stage_usd_caps must be a non-empty mapping")
        normalized = {
            str(stage): _finite_nonnegative(cap, f"stage_usd_caps.{stage}")
            for stage, cap in source.items()
        }
        if any(not stage for stage in normalized):
            raise ValueError("stage names must be non-empty")
        if not math.isclose(sum(normalized.values()), total, abs_tol=1e-9):
            raise ValueError("stage USD caps must sum to the total USD cap")
        manual = normalized.get("manual_reserve")
        if manual is None or not math.isclose(manual, reserve, abs_tol=1e-9):
            raise ValueError("manual_reserve must equal automatic_reserve_usd")
        object.__setattr__(self, "total_usd", total)
        object.__setattr__(self, "max_completions", completions)
        object.__setattr__(self, "max_storage_bytes", storage)
        object.__setattr__(self, "automatic_reserve_usd", reserve)
        object.__setattr__(self, "stage_usd_caps", normalized)

    @property
    def dispatchable_usd(self) -> float:
        return self.total_usd - self.automatic_reserve_usd

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["stage_usd_caps"] = dict(self.stage_usd_caps or {})
        result["dispatchable_usd"] = self.dispatchable_usd
        return result


@dataclass(frozen=True, slots=True)
class RunProjection:
    run_id: str
    stage_bucket: str
    cost_usd: float
    completions: int
    storage_bytes: int
    basis: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if not isinstance(self.stage_bucket, str) or not self.stage_bucket.strip():
            raise ValueError("stage_bucket must be non-empty")
        object.__setattr__(
            self, "cost_usd", _finite_nonnegative(self.cost_usd, "cost_usd")
        )
        object.__setattr__(
            self, "completions", _nonnegative_int(self.completions, "completions")
        )
        object.__setattr__(
            self,
            "storage_bytes",
            _nonnegative_int(self.storage_bytes, "storage_bytes"),
        )
        if not isinstance(self.basis, Mapping):
            raise TypeError("projection basis must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage_bucket": self.stage_bucket,
            "cost_usd": self.cost_usd,
            "completions": self.completions,
            "storage_bytes": self.storage_bytes,
            "basis": dict(self.basis),
        }


class PilotBudgetLedger:
    """Atomic, resumable budget ledger with one reservation per run."""

    def __init__(
        self,
        path: str | Path,
        *,
        contract_hash: str,
        caps: PilotBudgetCaps | None = None,
        tamper_evident: bool = False,
    ) -> None:
        self.path = Path(path)
        if not isinstance(contract_hash, str) or len(contract_hash) != 64:
            raise ValueError("contract_hash must be a SHA-256 hex digest")
        try:
            int(contract_hash, 16)
        except ValueError as exc:
            raise ValueError("contract_hash must be hexadecimal") from exc
        self.contract_hash = contract_hash
        self.caps = caps or PilotBudgetCaps()
        if not isinstance(tamper_evident, bool):
            raise TypeError("tamper_evident must be boolean")
        self.tamper_evident = tamper_evident
        self.schema_version = (
            PILOT_BUDGET_SCHEMA_VERSION_V2
            if tamper_evident
            else PILOT_BUDGET_SCHEMA_VERSION
        )
        if self.path.exists():
            self._state = self._load()
        else:
            self._state = {
                "schema_version": self.schema_version,
                "contract_hash": contract_hash,
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "caps": self.caps.to_dict(),
                "runs": {},
            }
            if self.tamper_evident:
                self._state["events"] = []
                self._append_event(
                    "genesis",
                    {
                        "contract_hash": contract_hash,
                        "caps_sha256": _canonical_sha256(self.caps.to_dict()),
                    },
                )
            self._write()

    def _load(self) -> dict[str, Any]:
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise PilotBudgetError("budget ledger root must be an object")
        if value.get("schema_version") != self.schema_version:
            raise PilotBudgetError("unsupported pilot budget ledger schema")
        if value.get("contract_hash") != self.contract_hash:
            raise PilotBudgetError("budget ledger contract hash mismatch")
        if value.get("caps") != self.caps.to_dict():
            raise PilotBudgetError("budget ledger caps differ from frozen caps")
        if not isinstance(value.get("runs"), dict):
            raise PilotBudgetError("budget ledger runs must be an object")
        if self.tamper_evident:
            self._verify_event_chain(value)
            expected = value.get("ledger_sha256")
            unsigned = dict(value)
            unsigned.pop("ledger_sha256", None)
            if expected != _canonical_sha256(unsigned):
                raise PilotBudgetError("budget ledger self-hash mismatch")
        return value

    def _append_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if not self.tamper_evident:
            return
        events = self._state.setdefault("events", [])
        if not isinstance(events, list):
            raise PilotBudgetError("budget ledger events must be an array")
        previous = events[-1]["event_sha256"] if events else "0" * 64
        event = {
            "event_index": len(events),
            "event_type": str(event_type),
            "created_at": _utc_now(),
            "previous_event_sha256": previous,
            "payload": json.loads(
                json.dumps(payload, sort_keys=True, allow_nan=False)
            ),
        }
        event["event_sha256"] = _canonical_sha256(event)
        events.append(event)

    @staticmethod
    def _verify_event_chain(value: Mapping[str, Any]) -> None:
        events = value.get("events")
        if not isinstance(events, list) or not events:
            raise PilotBudgetError("budget ledger v2 requires a non-empty event chain")
        previous = "0" * 64
        for index, row in enumerate(events):
            if not isinstance(row, Mapping):
                raise PilotBudgetError("budget ledger event must be an object")
            unsigned = dict(row)
            digest = unsigned.pop("event_sha256", None)
            if (
                row.get("event_index") != index
                or row.get("previous_event_sha256") != previous
                or digest != _canonical_sha256(unsigned)
            ):
                raise PilotBudgetError("budget ledger event chain mismatch")
            previous = str(digest)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state["updated_at"] = _utc_now()
        if self.tamper_evident:
            unsigned = dict(self._state)
            unsigned.pop("ledger_sha256", None)
            self._state["ledger_sha256"] = _canonical_sha256(unsigned)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._state, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def _totals(self, *, include_reservations: bool) -> dict[str, Any]:
        usd = 0.0
        completions = 0
        storage = 0
        by_stage = {stage: 0.0 for stage in self.caps.stage_usd_caps or {}}
        for row in self._state["runs"].values():
            actual = row.get("actual")
            reservation = row.get("reservation")
            selected = (
                actual
                if isinstance(actual, Mapping)
                else reservation if include_reservations else None
            )
            if not isinstance(selected, Mapping):
                continue
            usd += float(selected["cost_usd"])
            completions += int(selected["completions"])
            storage += int(selected["storage_bytes"])
            by_stage[str(row["stage_bucket"])] += float(selected["cost_usd"])
        return {
            "cost_usd": usd,
            "completions": completions,
            "storage_bytes": storage,
            "stage_cost_usd": by_stage,
        }

    def reserve(self, projection: RunProjection) -> None:
        if projection.stage_bucket == "manual_reserve":
            raise PilotBudgetError("manual reserve cannot be dispatched automatically")
        if projection.stage_bucket not in (self.caps.stage_usd_caps or {}):
            raise PilotBudgetError(
                f"unknown stage budget bucket {projection.stage_bucket!r}"
            )
        existing = self._state["runs"].get(projection.run_id)
        if existing is not None:
            if existing.get("reservation") != projection.to_dict():
                raise PilotBudgetError(
                    f"run {projection.run_id} already has a different reservation"
                )
            return

        totals = self._totals(include_reservations=True)
        projected_usd = totals["cost_usd"] + projection.cost_usd
        projected_completions = totals["completions"] + projection.completions
        projected_storage = totals["storage_bytes"] + projection.storage_bytes
        projected_stage = (
            totals["stage_cost_usd"][projection.stage_bucket]
            + projection.cost_usd
        )
        stage_cap = float(
            (self.caps.stage_usd_caps or {})[projection.stage_bucket]
        )
        violations = []
        if projected_usd > self.caps.dispatchable_usd + 1e-12:
            violations.append("dispatchable global USD")
        if projected_stage > stage_cap + 1e-12:
            violations.append(f"{projection.stage_bucket} USD")
        if projected_completions > self.caps.max_completions:
            violations.append("completion count")
        if projected_storage > self.caps.max_storage_bytes:
            violations.append("storage")
        if violations:
            raise PilotBudgetError(
                "reservation would exceed " + ", ".join(violations)
            )
        self._state["runs"][projection.run_id] = {
            "stage_bucket": projection.stage_bucket,
            "reservation": projection.to_dict(),
            "actual": None,
            "status": "reserved",
            "reserved_at": _utc_now(),
            "finalized_at": None,
        }
        self._append_event(
            "run_reserved",
            {
                "run_id": projection.run_id,
                "projection_sha256": _canonical_sha256(projection.to_dict()),
            },
        )
        self._write()

    def finalize(
        self,
        run_id: str,
        *,
        status: str,
        cost_usd: float,
        completions: int,
        storage_bytes: int,
        failure: Mapping[str, Any] | None = None,
    ) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError("status must be terminal")
        row = self._state["runs"].get(run_id)
        if row is None:
            raise PilotBudgetError(f"run {run_id} has no reservation")
        actual = {
            "cost_usd": _finite_nonnegative(cost_usd, "cost_usd"),
            "completions": _nonnegative_int(completions, "completions"),
            "storage_bytes": _nonnegative_int(storage_bytes, "storage_bytes"),
        }
        if row.get("actual") is not None:
            if (
                row["actual"] != actual
                or row.get("status") != status
                or row.get("failure") != (dict(failure) if failure else None)
            ):
                raise PilotBudgetError(
                    f"run {run_id} was already finalized differently"
                )
            return
        reservation = row["reservation"]
        overages = []
        if actual["cost_usd"] > float(reservation["cost_usd"]) + 1e-12:
            overages.append("cost")
        if actual["completions"] > int(reservation["completions"]):
            overages.append("completion count")
        if actual["storage_bytes"] > int(reservation["storage_bytes"]):
            overages.append("storage")
        if overages and status != "integrity-stopped":
            raise PilotBudgetError(
                "actual run exceeded reserved " + ", ".join(overages)
            )
        if overages and not failure:
            raise PilotBudgetError(
                "integrity-stopped overage requires failure evidence"
            )
        row["actual"] = actual
        row["status"] = status
        row["failure"] = dict(failure) if failure else None
        row["finalized_at"] = _utc_now()
        self._append_event(
            "run_finalized",
            {
                "run_id": run_id,
                "status": status,
                "actual_sha256": _canonical_sha256(actual),
                "failure_sha256": (
                    None if failure is None else _canonical_sha256(dict(failure))
                ),
            },
        )
        self._write()

    def snapshot(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "contract_hash": self.contract_hash,
            "caps": self.caps.to_dict(),
            "committed": self._totals(include_reservations=False),
            "committed_plus_reserved": self._totals(include_reservations=True),
            "runs": json.loads(
                json.dumps(self._state["runs"], sort_keys=True, allow_nan=False)
            ),
        }
        if self.tamper_evident:
            result["events"] = json.loads(
                json.dumps(self._state["events"], sort_keys=True, allow_nan=False)
            )
            result["event_chain_head"] = self._state["events"][-1][
                "event_sha256"
            ]
            result["ledger_sha256"] = self._state["ledger_sha256"]
        return result


def _conservative_observed_p95(values: Sequence[float]) -> float:
    """Return a nearest-rank p95 with an observed-maximum safety floor.

    The pilot preflight deliberately has only twelve action observations and
    four semantic observations per model.  Interpolated quantiles can therefore
    fall materially below a value that was already observed.  A dispatch cap
    must never reserve less than its own source evidence, so the frozen p95 is
    floored at the observed maximum.
    """

    if not values:
        raise ValueError("p95 requires at least one observation")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    nearest_rank = ordered[rank - 1]
    return max(nearest_rank, ordered[-1])


def preflight_p95(
    usage_rows: Sequence[Mapping[str, Any]],
    *,
    reserve_multiplier: float = 1.25,
) -> dict[str, Any]:
    """Compute model x call-kind p95 projections from complete preflight rows."""

    reserve_multiplier = _finite_nonnegative(
        reserve_multiplier, "reserve_multiplier"
    )
    if reserve_multiplier < 1:
        raise ValueError("reserve_multiplier must be at least one")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in usage_rows:
        if not isinstance(row, Mapping):
            raise TypeError("usage rows must be mappings")
        model = str(row.get("response_model") or row.get("model") or "")
        call_kind = str(row.get("call_kind") or "")
        usage = row.get("usage")
        if not model or not call_kind or not isinstance(usage, Mapping):
            raise ValueError("usage rows require model, call_kind, and usage")
        grouped.setdefault((model, call_kind), []).append(usage)
    if not grouped:
        raise ValueError("preflight p95 requires at least one usage row")
    result: dict[str, Any] = {}
    for (model, call_kind), rows in sorted(grouped.items()):
        key = f"{model}::{call_kind}"
        raw = {
            field: _conservative_observed_p95(
                [_finite_nonnegative(row.get(field), field) for row in rows]
            )
            for field in (
                "prompt_tokens",
                "completion_tokens",
                "cost_usd",
            )
        }
        # A dispatch reservation must remain internally additive even though
        # marginal p95s need not occur on the same sample row.
        raw["total_tokens"] = (
            raw["prompt_tokens"] + raw["completion_tokens"]
        )
        reserved_prompt = math.ceil(
            raw["prompt_tokens"] * reserve_multiplier
        )
        reserved_completion = math.ceil(
            raw["completion_tokens"] * reserve_multiplier
        )
        result[key] = {
            "sample_count": len(rows),
            "raw_p95": raw,
            "reserved_p95": {
                "prompt_tokens": reserved_prompt,
                "completion_tokens": reserved_completion,
                "total_tokens": reserved_prompt + reserved_completion,
                "cost_usd": raw["cost_usd"] * reserve_multiplier,
            },
            "reserve_multiplier": reserve_multiplier,
        }
    return result


__all__ = [
    "DEFAULT_STAGE_USD_CAPS",
    "PILOT_BUDGET_SCHEMA_VERSION",
    "PilotBudgetCaps",
    "PilotBudgetError",
    "PilotBudgetLedger",
    "RunProjection",
    "preflight_p95",
]
