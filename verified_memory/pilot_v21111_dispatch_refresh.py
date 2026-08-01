"""V2.11.11 paid dispatch-identity refresh before scientific execution.

The refresh is an independently registered operational authority: twenty fixed
calls, five probes for each OpenAI profile and call kind.  It is neither part
of the scientific ITT denominator nor an observed-p95 estimator.  Any failed
or interrupted row permanently makes the shared receipt a no-go.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from llm_providers import StructuredCompletion

from .pilot_budget import (
    TERMINAL_STATUSES as BUDGET_TERMINAL_STATUSES,
    PilotBudgetCaps,
    PilotBudgetLedger,
    RunProjection,
)
from .pilot_contract import PilotContract, ProviderRequestProfile, canonical_sha256
from .pilot_provider_catalog import (
    ProviderCatalogError,
    verify_provider_catalog_receipt,
)
from .pilot_v21111_fresh_cohort import (
    V21111_ACCEPTANCE_FILENAME,
    V21111_CONTRACT_ID,
    parent_budget_debit_for_v21111,
    verify_scientific_dispatch_acceptance,
)


V21111_REFRESH_SCHEMA = "finevo-pilot-v2.11.11-dispatch-refresh-receipt-v1"
V21111_REFRESH_LEDGER_SCHEMA = "finevo-pilot-v2.11.11-dispatch-refresh-ledger-v1"
V21111_REFRESH_DIRECTORY = "dispatch-refresh"
V21111_REFRESH_LEDGER_FILENAME = "authority_ledger.json"
V21111_REFRESH_RECEIPT_FILENAME = "dispatch_refresh_receipt.json"
_TERMINAL = frozenset({"complete", "failed", "skipped"})
_REFRESH_CHECKS = frozenset(
    {
        "provider_ok",
        "attempts_one",
        "requested_model_exact",
        "served_model_exact",
        "provider_exact",
        "provider_route_exact",
        "request_id_present",
        "finish_reason_stop",
        "response_completed",
        "exact_json",
        "prompt_below_short_context_ceiling",
        "request_parameters_exact_minimum",
        "service_tier_default",
        "cost_matches_frozen_price",
    }
)


class PilotV21111DispatchRefreshError(RuntimeError):
    """Raised when refresh authority cannot safely authorize science."""


class RefreshProvider(Protocol):
    def get_structured_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
        max_retries: int | None = None,
        seed: int | None = None,
    ) -> StructuredCompletion: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy(value: Any) -> Any:
    """Return a JSON-native copy of recursively frozen contract values."""

    if isinstance(value, Mapping):
        thawed: Any = {str(key): _copy(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        thawed = [_copy(item) for item in value]
    else:
        thawed = value
    return json.loads(json.dumps(thawed, sort_keys=True, allow_nan=False))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any], *, new: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink() or (new and path.exists()):
        raise PilotV21111DispatchRefreshError(
            f"refusing unsafe or duplicate refresh write: {path.name}"
        )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if new:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    except FileExistsError as exc:
        raise PilotV21111DispatchRefreshError(
            f"concurrent refresh writer created {path.name}"
        ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _refresh_plan(contract: PilotContract) -> dict[str, Any]:
    if contract.contract_id != V21111_CONTRACT_ID:
        raise PilotV21111DispatchRefreshError("refresh uses the wrong contract")
    boundary = contract.v21111_fresh_cohort_boundary
    if not isinstance(boundary, Mapping):
        raise PilotV21111DispatchRefreshError("fresh-cohort boundary is absent")
    plan = boundary.get("dispatch_refresh")
    if (
        not isinstance(plan, Mapping)
        or plan.get("provider_calls") != 20
        or len(plan.get("rows", ())) != 20
        or len(plan.get("definitions", ())) != 10
        or plan.get("failure_policy") != "global-science-no-go-no-retry-no-replacement"
    ):
        raise PilotV21111DispatchRefreshError("dispatch refresh plan drifted")
    return _copy(plan)


def refresh_projections(contract: PilotContract) -> tuple[RunProjection, ...]:
    """Materialize the exact twenty conservative one-call reservations."""

    plan = _refresh_plan(contract)
    result = tuple(
        RunProjection(
            run_id=str(row["run_id"]),
            stage_bucket="dispatch_refresh",
            cost_usd=float(row["reserved_cost_usd"]),
            completions=1,
            storage_bytes=1_000_000,
            basis={
                "method": "fixed-full-cap-price-times-1.25",
                "profile_id": row["profile_id"],
                "probe_id": row["probe_id"],
                "call_kind": row["call_kind"],
                "prompt_token_upper_bound": row["prompt_token_upper_bound"],
                "max_completion_tokens": row["max_completion_tokens"],
                "reserve_multiplier": plan["reserve_multiplier"],
                "cached_input_discount_assumed": False,
                "scientific_evidence": False,
            },
        )
        for row in plan["rows"]
    )
    if (
        len(result) != 20
        or len({row.run_id for row in result}) != 20
        or not math.isclose(
            sum(row.cost_usd for row in result),
            float(plan["reserved_cost_usd"]),
            abs_tol=1e-12,
        )
    ):
        raise PilotV21111DispatchRefreshError("refresh projections drifted")
    return result


def _budget_caps(contract: PilotContract) -> PilotBudgetCaps:
    budgets = contract.budgets
    return PilotBudgetCaps(
        total_usd=float(budgets["total_usd"]),
        max_completions=int(budgets["max_provider_completions"]),
        completion_scope=str(budgets["completion_scope"]),
        max_storage_bytes=int(budgets["max_storage_bytes"]),
        stage_usd_caps={
            str(key): float(value) for key, value in budgets["stage_usd_caps"].items()
        },
        automatic_reserve_usd=float(budgets["automatic_reserve_usd"]),
    )


def _fixed_capacity(contract: PilotContract) -> dict[str, Any]:
    """Recompute the frozen parent + refresh + science hard-cap arithmetic."""

    plan = _refresh_plan(contract)
    boundary = contract.v21111_fresh_cohort_boundary
    assert isinstance(boundary, Mapping)
    envelope = boundary["budget_envelope"]
    parent = parent_budget_debit_for_v21111(contract)
    science_cost = float(envelope["fresh_full_cap_reserve_usd"]["total"])
    science_calls = int(boundary["fresh_cohort"]["simulated_provider_calls"])
    refresh_cost = float(plan["reserved_cost_usd"])
    refresh_calls = int(plan["provider_calls"])
    total_cost = float(contract.budgets["total_usd"])
    total_calls = int(contract.budgets["max_provider_completions"])
    final = {
        "cost_usd": float(envelope["remaining_cost_usd"]),
        "hosted_completions": int(envelope["remaining_hosted_completions"]),
    }
    initial = {
        "cost_usd": final["cost_usd"] + refresh_cost,
        "hosted_completions": final["hosted_completions"] + refresh_calls,
    }
    expected_cumulative_cost = parent.cost_usd + refresh_cost + science_cost
    expected_cumulative_calls = (
        parent.hosted_completions + refresh_calls + science_calls
    )
    if (
        not math.isclose(
            expected_cumulative_cost,
            float(envelope["projected_cumulative_cost_usd"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or expected_cumulative_calls
        != int(envelope["projected_cumulative_hosted_completions"])
        or not math.isclose(
            final["cost_usd"],
            float(envelope["remaining_cost_usd"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or final["hosted_completions"] != int(envelope["remaining_hosted_completions"])
        or initial["cost_usd"] < refresh_cost - 1e-12
        or initial["hosted_completions"] < refresh_calls
    ):
        raise PilotV21111DispatchRefreshError(
            "dispatch refresh fixed-capacity arithmetic drifted"
        )
    return {
        "parent": {
            "cost_usd": parent.cost_usd,
            "hosted_completions": parent.hosted_completions,
        },
        "science_reserve": {
            "cost_usd": science_cost,
            "hosted_completions": science_calls,
        },
        "initial_remaining_after_parent_plus_science": initial,
        "refresh_reserve": {
            "cost_usd": refresh_cost,
            "hosted_completions": refresh_calls,
        },
        "final_remaining_after_refresh": final,
    }


def _assert_whole_plan_fits(
    *,
    contract: PilotContract,
    budget_ledger: PilotBudgetLedger,
    projections: Mapping[str, RunProjection],
) -> dict[str, Any]:
    """Fail before call one unless all 20 refresh rows and all science fit."""

    fixed = _fixed_capacity(contract)
    snapshot = budget_ledger.snapshot()
    if snapshot.get("caps") != _budget_caps(contract).to_dict():
        raise PilotV21111DispatchRefreshError("refresh budget caps drifted")
    parent = parent_budget_debit_for_v21111(contract).to_dict()
    if snapshot.get("parent_debit") != parent:
        raise PilotV21111DispatchRefreshError("refresh parent debit drifted")

    parent_run_ids = {spec.run_id for spec in contract.expand(stage="parent-import")}
    allowed = parent_run_ids | set(projections)
    runs = snapshot.get("runs")
    if not isinstance(runs, Mapping) or not set(runs) <= allowed:
        raise PilotV21111DispatchRefreshError(
            "non-refresh budget rows exist before refresh GO"
        )
    for run_id in parent_run_ids:
        row = runs.get(run_id)
        actual = row.get("actual") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or row.get("status") != "complete"
            or not isinstance(actual, Mapping)
            or actual.get("cost_usd") != 0.0
            or actual.get("completions") != 0
            or isinstance(actual.get("storage_bytes"), bool)
            or not isinstance(actual.get("storage_bytes"), int)
            or int(actual["storage_bytes"]) < 0
        ):
            raise PilotV21111DispatchRefreshError(
                "parent-import budget row drifted before refresh"
            )

    # Validate every extant refresh reservation before doing arithmetic.  A
    # conflicting row is not something resume may repair or silently replace.
    for run_id, row in runs.items():
        if run_id not in projections:
            continue
        if (
            not isinstance(row, Mapping)
            or row.get("stage_bucket") != "dispatch_refresh"
            or row.get("reservation") != projections[run_id].to_dict()
        ):
            raise PilotV21111DispatchRefreshError("refresh budget reservation drifted")

    totals = snapshot["committed_plus_reserved"]
    absent = [
        projection for run_id, projection in projections.items() if run_id not in runs
    ]
    science = fixed["science_reserve"]
    projected_cost = (
        float(totals["cost_usd"])
        + sum(row.cost_usd for row in absent)
        + float(science["cost_usd"])
    )
    projected_calls = (
        int(totals["completions"])
        + sum(row.completions for row in absent)
        + int(science["hosted_completions"])
    )
    projected_refresh_stage = float(totals["stage_cost_usd"]["dispatch_refresh"]) + sum(
        row.cost_usd for row in absent
    )
    caps = _budget_caps(contract)
    if (
        projected_cost > caps.dispatchable_usd + 1e-12
        or projected_calls > caps.max_completions
        or projected_refresh_stage
        > float(caps.stage_usd_caps["dispatch_refresh"]) + 1e-12
        or float(science["cost_usd"])
        > float(caps.stage_usd_caps["hosted_v21111"]) + 1e-12
    ):
        raise PilotV21111DispatchRefreshError(
            "complete refresh plus scientific matrix no longer fits hard caps"
        )
    return fixed


def _validate_catalog_evidence(
    contract: PilotContract,
    evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(evidence) != {"gpt52_main", "gpt56_diagnostic"}:
        raise PilotV21111DispatchRefreshError("refresh catalog evidence is incomplete")
    result: dict[str, Any] = {}
    for profile_id in ("gpt52_main", "gpt56_diagnostic"):
        profile = contract.provider_profiles[profile_id]
        row = evidence[profile_id]
        expected = {
            "profile_id": profile_id,
            "captured_at": profile.price_snapshot.captured_at,
            "price_source_url": profile.price_snapshot.source,
            "model_reference_url": profile.price_snapshot.model_reference,
        }
        if any(row.get(key) != value for key, value in expected.items()) or any(
            not isinstance(row.get(key), str)
            or len(str(row[key])) != 64
            or any(character not in "0123456789abcdef" for character in str(row[key]))
            for key in ("price_source_sha256", "model_reference_sha256")
        ):
            raise PilotV21111DispatchRefreshError(
                f"{profile_id} catalog evidence drifted"
            )
        result[profile_id] = _copy(row)
    return result


class DispatchRefreshLedger:
    """Small append-only state machine for the independent refresh denominator."""

    def __init__(self, path: Path, *, contract: PilotContract) -> None:
        self.path = path
        self.contract = contract
        self.plan = _refresh_plan(contract)
        if path.exists():
            self.state = json.loads(path.read_text(encoding="utf-8"))
            self._verify()
            return
        rows = {
            str(row["run_id"]): {
                "plan": _copy(row),
                "status": "scheduled",
                "request": None,
                "response": None,
                "failure": None,
                "started_at": None,
                "terminal_at": None,
            }
            for row in self.plan["rows"]
        }
        self.state = {
            "schema_version": V21111_REFRESH_LEDGER_SCHEMA,
            "contract_sha256": contract.canonical_hash,
            "plan_sha256": canonical_sha256(self.plan),
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "rows": rows,
            "events": [],
        }
        self._event("genesis", {"rows_sha256": canonical_sha256(rows)})
        self._write()

    def _event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        previous = (
            self.state["events"][-1]["event_sha256"]
            if self.state["events"]
            else "0" * 64
        )
        event = {
            "event_index": len(self.state["events"]),
            "event_type": event_type,
            "created_at": _utc_now(),
            "previous_event_sha256": previous,
            "payload": _copy(payload),
        }
        event["event_sha256"] = canonical_sha256(event)
        self.state["events"].append(event)

    def _write(self) -> None:
        self.state["updated_at"] = _utc_now()
        unsigned = dict(self.state)
        unsigned.pop("ledger_sha256", None)
        self.state["ledger_sha256"] = canonical_sha256(unsigned)
        _atomic_json(self.path, self.state)

    def _verify(self) -> None:
        value = self.state
        expected_rows = {str(row["run_id"]): _copy(row) for row in self.plan["rows"]}
        unsigned = dict(value)
        claimed = unsigned.pop("ledger_sha256", None)
        if (
            value.get("schema_version") != V21111_REFRESH_LEDGER_SCHEMA
            or value.get("contract_sha256") != self.contract.canonical_hash
            or value.get("plan_sha256") != canonical_sha256(self.plan)
            or set(value.get("rows", {})) != set(expected_rows)
            or claimed != canonical_sha256(unsigned)
        ):
            raise PilotV21111DispatchRefreshError("refresh authority ledger drifted")
        previous = "0" * 64
        for index, event in enumerate(value.get("events", ())):
            event_unsigned = dict(event)
            digest = event_unsigned.pop("event_sha256", None)
            if (
                event.get("event_index") != index
                or event.get("previous_event_sha256") != previous
                or digest != canonical_sha256(event_unsigned)
            ):
                raise PilotV21111DispatchRefreshError(
                    "refresh authority event chain drifted"
                )
            previous = str(digest)
        expected_row_fields = {
            "plan",
            "status",
            "request",
            "response",
            "failure",
            "started_at",
            "terminal_at",
        }
        for run_id, row in value["rows"].items():
            if (
                not isinstance(row, Mapping)
                or set(row) != expected_row_fields
                or row.get("plan") != expected_rows[run_id]
            ):
                raise PilotV21111DispatchRefreshError(
                    "refresh authority ledger drifted"
                )
            status = row.get("status")
            if status not in {"scheduled", "running", "returned", *_TERMINAL}:
                raise PilotV21111DispatchRefreshError(
                    f"refresh row {run_id} has invalid status"
                )
            if status == "scheduled" and any(
                row.get(field) is not None
                for field in (
                    "request",
                    "response",
                    "failure",
                    "started_at",
                    "terminal_at",
                )
            ):
                raise PilotV21111DispatchRefreshError(
                    "refresh authority ledger drifted"
                )
            if status in {"running", "returned", "complete"} and (
                not isinstance(row.get("request"), Mapping)
                or not isinstance(row.get("started_at"), str)
            ):
                raise PilotV21111DispatchRefreshError(
                    "refresh authority ledger drifted"
                )
            if status == "returned" and not isinstance(row.get("response"), Mapping):
                raise PilotV21111DispatchRefreshError(
                    "refresh authority ledger drifted"
                )
            if status in _TERMINAL and not isinstance(row.get("terminal_at"), str):
                raise PilotV21111DispatchRefreshError(
                    "refresh authority ledger drifted"
                )

    def begin(self, run_id: str, request: Mapping[str, Any]) -> None:
        row = self.state["rows"][run_id]
        if row["status"] != "scheduled":
            raise PilotV21111DispatchRefreshError("refresh row cannot be redispatched")
        row.update(status="running", request=_copy(request), started_at=_utc_now())
        self._event("dispatch_started", {"run_id": run_id})
        self._write()

    def returned(self, run_id: str, response: Mapping[str, Any]) -> None:
        row = self.state["rows"][run_id]
        if row["status"] != "running":
            raise PilotV21111DispatchRefreshError("refresh return state drifted")
        row.update(status="returned", response=_copy(response))
        self._event(
            "provider_returned",
            {"run_id": run_id, "response_sha256": canonical_sha256(response)},
        )
        self._write()

    def finalize(
        self,
        run_id: str,
        *,
        status: str,
        failure: Mapping[str, Any] | None,
    ) -> None:
        if status not in _TERMINAL:
            raise PilotV21111DispatchRefreshError("invalid refresh terminal status")
        row = self.state["rows"][run_id]
        if row["status"] in _TERMINAL:
            if row["status"] != status or row.get("failure") != (
                None if failure is None else dict(failure)
            ):
                raise PilotV21111DispatchRefreshError(
                    "refresh row was finalized differently"
                )
            return
        row.update(
            status=status,
            failure=None if failure is None else _copy(failure),
            terminal_at=_utc_now(),
        )
        self._event(
            "row_finalized",
            {
                "run_id": run_id,
                "status": status,
                "row_sha256": canonical_sha256(row),
            },
        )
        self._write()

    def invalidate_complete(
        self,
        run_id: str,
        *,
        failure: Mapping[str, Any],
    ) -> None:
        """Fail closed when a completed authority row lacks matching budget state."""

        row = self.state["rows"][run_id]
        normalized = _copy(failure)
        if row["status"] == "failed" and row.get("failure") == normalized:
            return
        if row["status"] != "complete":
            raise PilotV21111DispatchRefreshError(
                "only a completed refresh row can be invalidated"
            )
        row.update(status="failed", failure=normalized, terminal_at=_utc_now())
        self._event(
            "completed_row_invalidated",
            {
                "run_id": run_id,
                "failure_sha256": canonical_sha256(normalized),
                "row_sha256": canonical_sha256(row),
            },
        )
        self._write()

    def snapshot(self) -> dict[str, Any]:
        return _copy(self.state)


def _definition_by_id(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["probe_id"]): row for row in plan["definitions"]}


def _request_record(
    *,
    profile: ProviderRequestProfile,
    row: Mapping[str, Any],
    definition: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "profile_id": profile.profile_id,
        "profile_sha256": canonical_sha256(profile.to_dict()),
        "requested_model": profile.requested_model,
        "served_model": profile.served_model,
        "service_tier": profile.service_tier,
        "prompt_messages_sha256": definition["messages_sha256"],
        "prompt_token_upper_bound": definition["prompt_token_upper_bound"],
        "short_context_prompt_token_ceiling": (
            profile.short_context_prompt_token_ceiling
        ),
        "max_completion_tokens": row["max_completion_tokens"],
        "response_format": {"type": "json_object"},
        "reasoning_effort": profile.reasoning.effort,
        "request_timeout_seconds": 300,
        "max_attempts": 1,
        "seed": None,
    }


def _expected_usage_cost(
    profile: ProviderRequestProfile,
    completion: StructuredCompletion,
) -> float:
    cached = int(completion.cached_prompt_tokens)
    uncached = int(completion.usage.prompt_tokens) - cached
    price = profile.price_snapshot
    cached_rate = (
        price.dispatch_input
        if price.dispatch_cached_input is None
        else price.dispatch_cached_input
    )
    assert price.dispatch_input is not None
    assert price.dispatch_output is not None
    return (
        uncached * float(price.dispatch_input)
        + cached * float(cached_rate)
        + completion.usage.completion_tokens * float(price.dispatch_output)
    ) / 1_000_000.0


def _completion_record(
    *,
    profile: ProviderRequestProfile,
    definition: Mapping[str, Any],
    completion: StructuredCompletion,
) -> tuple[dict[str, Any], bool, dict[str, Any] | None]:
    parsed: Any = None
    parse_error: str | None = None
    try:
        parsed = json.loads(completion.text)
    except (TypeError, json.JSONDecodeError) as exc:
        parse_error = type(exc).__name__
    expected_cost = _expected_usage_cost(profile, completion)
    required_parameters = {
        "model",
        "messages",
        "response_format",
        "reasoning_effort",
        "service_tier",
        "max_completion_tokens",
    }
    checks = {
        "provider_ok": completion.ok,
        "attempts_one": completion.attempts == 1,
        "requested_model_exact": completion.model == profile.requested_model,
        "served_model_exact": completion.response_model == profile.served_model,
        "provider_exact": completion.provider == "openai",
        "provider_route_exact": (
            completion.response_provider == "OpenAI-direct"
            and completion.response_route == "direct"
        ),
        "request_id_present": completion.request_id is not None,
        "finish_reason_stop": completion.finish_reason == "stop",
        "response_completed": completion.response_completed is True,
        "exact_json": parsed == definition["expected_json"],
        "prompt_below_short_context_ceiling": (
            completion.usage.prompt_tokens
            <= int(profile.short_context_prompt_token_ceiling or 0)
        ),
        "request_parameters_exact_minimum": required_parameters
        <= set(completion.request_parameters),
        "service_tier_default": profile.service_tier == "default",
        "cost_matches_frozen_price": math.isclose(
            completion.cost,
            expected_cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
    passed = all(checks.values())
    failure = None
    if not passed:
        failure = {
            "error_type": completion.error_type or "DispatchRefreshValidationError",
            "failed_checks": sorted(key for key, value in checks.items() if not value),
            "json_error_type": parse_error,
        }
    return (
        {
            "completion_audit": completion.safe_audit_dict(),
            "requested_model": profile.requested_model,
            "response_model": completion.response_model,
            "response_provider": completion.response_provider,
            "response_route": completion.response_route,
            "request_id_sha256": (
                None
                if completion.request_id is None
                else _sha256_bytes(completion.request_id.encode("utf-8"))
            ),
            "expected_json_sha256": definition["expected_json_sha256"],
            "parsed_json_sha256": (
                None if parsed is None else canonical_sha256(parsed)
            ),
            "json_error_type": parse_error,
            "usage": {
                **completion.usage.to_dict(),
                "cached_prompt_tokens": completion.cached_prompt_tokens,
                "reasoning_tokens": completion.reasoning_tokens,
                "frozen_price_cost_usd": expected_cost,
            },
            "checks": checks,
            "validation_failure": failure,
        },
        passed,
        failure,
    )


def _stored_response_outcome(
    *,
    profile: ProviderRequestProfile,
    definition: Mapping[str, Any],
    response: Mapping[str, Any],
) -> tuple[bool, dict[str, Any] | None, float]:
    """Revalidate a durable provider return without replaying the call."""

    expected_fields = {
        "completion_audit",
        "requested_model",
        "response_model",
        "response_provider",
        "response_route",
        "request_id_sha256",
        "expected_json_sha256",
        "parsed_json_sha256",
        "json_error_type",
        "usage",
        "checks",
        "validation_failure",
    }
    if set(response) != expected_fields:
        raise PilotV21111DispatchRefreshError("stored refresh response schema drifted")
    audit = response.get("completion_audit")
    usage = response.get("usage")
    checks = response.get("checks")
    if (
        not isinstance(audit, Mapping)
        or not isinstance(usage, Mapping)
        or not isinstance(checks, Mapping)
        or set(checks) != _REFRESH_CHECKS
        or any(not isinstance(value, bool) for value in checks.values())
    ):
        raise PilotV21111DispatchRefreshError("stored refresh response audit drifted")

    integer_fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_prompt_tokens",
        "reasoning_tokens",
    )
    if any(
        isinstance(usage.get(field), bool)
        or not isinstance(usage.get(field), int)
        or int(usage[field]) < 0
        for field in integer_fields
    ):
        raise PilotV21111DispatchRefreshError("stored refresh usage drifted")
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage["completion_tokens"])
    cached_tokens = int(usage["cached_prompt_tokens"])
    if (
        cached_tokens > prompt_tokens
        or int(usage["total_tokens"]) != prompt_tokens + completion_tokens
    ):
        raise PilotV21111DispatchRefreshError("stored refresh usage drifted")
    try:
        stored_cost = float(usage["frozen_price_cost_usd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PilotV21111DispatchRefreshError(
            "stored refresh frozen-price cost drifted"
        ) from exc
    if not math.isfinite(stored_cost) or stored_cost < 0:
        raise PilotV21111DispatchRefreshError(
            "stored refresh frozen-price cost drifted"
        )
    price = profile.price_snapshot
    cached_rate = (
        price.dispatch_input
        if price.dispatch_cached_input is None
        else price.dispatch_cached_input
    )
    expected_cost = (
        (prompt_tokens - cached_tokens) * float(price.dispatch_input)
        + cached_tokens * float(cached_rate)
        + completion_tokens * float(price.dispatch_output)
    ) / 1_000_000.0

    request_parameters = audit.get("request_parameters")
    required_parameters = {
        "model",
        "messages",
        "response_format",
        "reasoning_effort",
        "service_tier",
        "max_completion_tokens",
    }
    recomputed = {
        "provider_ok": audit.get("error_type") is None,
        "attempts_one": audit.get("attempts") == 1,
        "requested_model_exact": (
            response.get("requested_model") == profile.requested_model
            and audit.get("model") == profile.requested_model
        ),
        "served_model_exact": response.get("response_model") == profile.served_model,
        "provider_exact": audit.get("provider") == "openai",
        "provider_route_exact": (
            response.get("response_provider") == "OpenAI-direct"
            and response.get("response_route") == "direct"
        ),
        "request_id_present": (
            isinstance(response.get("request_id_sha256"), str)
            and len(str(response["request_id_sha256"])) == 64
            and audit.get("request_id_present") is True
        ),
        "finish_reason_stop": audit.get("finish_reason") == "stop",
        "response_completed": audit.get("response_completed") is True,
        "exact_json": (
            response.get("expected_json_sha256") == definition["expected_json_sha256"]
            and response.get("parsed_json_sha256") == definition["expected_json_sha256"]
        ),
        "prompt_below_short_context_ceiling": (
            prompt_tokens <= int(profile.short_context_prompt_token_ceiling or 0)
        ),
        "request_parameters_exact_minimum": (
            isinstance(request_parameters, list)
            and required_parameters <= set(request_parameters)
        ),
        "service_tier_default": profile.service_tier == "default",
        "cost_matches_frozen_price": math.isclose(
            stored_cost,
            expected_cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
    audit_usage = audit.get("usage")
    if (
        dict(checks) != recomputed
        or not isinstance(audit_usage, Mapping)
        or audit_usage.get("prompt_tokens") != prompt_tokens
        or audit_usage.get("completion_tokens") != completion_tokens
        or audit_usage.get("total_tokens") != int(usage["total_tokens"])
        or audit.get("cached_prompt_tokens") != cached_tokens
        or audit.get("reasoning_tokens") != int(usage["reasoning_tokens"])
        or audit.get("request_profile_id") != profile.profile_id
        or not isinstance(audit.get("output_disposition"), str)
        or not audit.get("output_disposition")
    ):
        raise PilotV21111DispatchRefreshError(
            "stored refresh response validation drifted"
        )

    passed = all(recomputed.values())
    if passed and audit.get("output_disposition") != "accepted":
        raise PilotV21111DispatchRefreshError(
            "stored refresh response validation drifted"
        )
    failure = None
    if not passed:
        failure = {
            "error_type": audit.get("error_type") or "DispatchRefreshValidationError",
            "failed_checks": sorted(
                key for key, value in recomputed.items() if not value
            ),
            "json_error_type": response.get("json_error_type"),
        }
    if response.get("validation_failure") != failure:
        raise PilotV21111DispatchRefreshError(
            "stored refresh validation outcome drifted"
        )
    return passed, failure, expected_cost


def _validate_complete_row(
    *,
    plan_row: Mapping[str, Any],
    definition: Mapping[str, Any],
    profile: ProviderRequestProfile,
    authority_row: Mapping[str, Any],
    budget_row: Mapping[str, Any],
    projection: RunProjection,
) -> float:
    expected_request = _request_record(
        profile=profile,
        row=plan_row,
        definition=definition,
    )
    if (
        authority_row.get("status") != "complete"
        or authority_row.get("plan") != _copy(plan_row)
        or authority_row.get("request") != expected_request
        or authority_row.get("failure") is not None
        or not isinstance(authority_row.get("response"), Mapping)
        or budget_row.get("stage_bucket") != "dispatch_refresh"
        or budget_row.get("reservation") != projection.to_dict()
        or budget_row.get("status") != "complete"
        or budget_row.get("failure") is not None
        or not isinstance(budget_row.get("actual"), Mapping)
    ):
        raise PilotV21111DispatchRefreshError("complete refresh row drifted")
    passed, failure, expected_cost = _stored_response_outcome(
        profile=profile,
        definition=definition,
        response=authority_row["response"],
    )
    actual = budget_row["actual"]
    if (
        passed is not True
        or failure is not None
        or actual.get("completions") != 1
        or actual.get("storage_bytes") != 0
        or not math.isclose(
            float(actual.get("cost_usd", -1)),
            expected_cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or expected_cost > projection.cost_usd + 1e-12
    ):
        raise PilotV21111DispatchRefreshError("complete refresh row drifted")
    return expected_cost


def _write_receipt(
    *,
    contract: PilotContract,
    raw_root: Path,
    paid: Any,
    acceptance: Mapping[str, Any],
    plan: Mapping[str, Any],
    ledger: DispatchRefreshLedger,
    budget_ledger: PilotBudgetLedger,
    catalog_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = ledger.snapshot()
    statuses = [row["status"] for row in snapshot["rows"].values()]
    budget = budget_ledger.snapshot()
    budget_events = budget["events"]
    committed = budget["committed"]
    fixed = _fixed_capacity(contract)
    live_headroom = {
        "cost_usd": round(
            float(contract.budgets["total_usd"]) - float(committed["cost_usd"]),
            12,
        ),
        "hosted_completions": int(contract.budgets["max_provider_completions"])
        - int(committed["completions"]),
    }
    capacity_fits = (
        fixed["final_remaining_after_refresh"]["cost_usd"] >= -1e-12
        and fixed["final_remaining_after_refresh"]["hosted_completions"] >= 0
        and live_headroom["cost_usd"] + 1e-12
        >= float(fixed["science_reserve"]["cost_usd"])
        and live_headroom["hosted_completions"]
        >= int(fixed["science_reserve"]["hosted_completions"])
    )
    go = (
        len(statuses) == 20
        and all(status == "complete" for status in statuses)
        and capacity_fits
    )
    receipt = {
        "schema_version": V21111_REFRESH_SCHEMA,
        "status": "go" if go else "no-go",
        "go": go,
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "release_provenance": {
            "git_tag": getattr(paid, "git_tag", None),
            "git_commit": getattr(paid, "head_commit", None),
            "worktree_clean": getattr(paid, "worktree_clean", None),
        },
        "acceptance_binding": {
            "path": str(raw_root / V21111_ACCEPTANCE_FILENAME),
            "content_sha256": acceptance["integrity"]["content_sha256"],
        },
        "plan_sha256": canonical_sha256(plan),
        "profile_bindings": {
            profile_id: {
                "profile": contract.provider_profiles[profile_id].to_dict(),
                "profile_sha256": canonical_sha256(
                    contract.provider_profiles[profile_id].to_dict()
                ),
                "catalog_evidence": _copy(catalog_evidence[profile_id]),
            }
            for profile_id in plan["model_profiles"]
        },
        "provider_calls_attempted": sum(
            row["status"] in {"running", "returned", "complete", "failed"}
            and row.get("started_at") is not None
            for row in snapshot["rows"].values()
        ),
        "row_status_counts": {
            status: statuses.count(status) for status in sorted(set(statuses))
        },
        "authority_ledger_binding": {
            "path": str(ledger.path),
            "file_sha256": _sha256_bytes(ledger.path.read_bytes()),
            "ledger_sha256": snapshot["ledger_sha256"],
            "event_count": len(snapshot["events"]),
            "event_chain_head": snapshot["events"][-1]["event_sha256"],
        },
        "budget_ledger_prefix": {
            "event_count": len(budget_events),
            "event_chain_head": budget_events[-1]["event_sha256"],
            "ledger_sha256": budget["ledger_sha256"],
            "refresh_reserved_cost_usd": plan["reserved_cost_usd"],
            "refresh_actual_cost_usd": sum(
                float(row["actual"]["cost_usd"])
                for run_id, row in budget["runs"].items()
                if "--dispatch-refresh--" in run_id
                and isinstance(row.get("actual"), Mapping)
            ),
        },
        "science_capacity": {
            **fixed,
            "live_global_headroom_before_science": live_headroom,
            "complete_matrix_fits": capacity_fits,
        },
        "scientific_evidence": False,
        "claim_boundary": "identity-interface-length-cap-refresh-not-p95",
    }
    receipt["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(receipt),
    }
    path = raw_root / V21111_REFRESH_DIRECTORY / V21111_REFRESH_RECEIPT_FILENAME
    _atomic_json(path, receipt, new=True)
    return {**receipt, "receipt_path": str(path)}


def _verify_catalog_binding(
    *,
    contract: PilotContract,
    root: Path,
    profile_bindings: Mapping[str, Any],
) -> None:
    if set(profile_bindings) != {"gpt52_main", "gpt56_diagnostic"}:
        raise PilotV21111DispatchRefreshError("refresh profile bindings drifted")
    evidence: dict[str, Mapping[str, Any]] = {}
    for profile_id in ("gpt52_main", "gpt56_diagnostic"):
        binding = profile_bindings.get(profile_id)
        profile = contract.provider_profiles[profile_id]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"profile", "profile_sha256", "catalog_evidence"}
            or binding.get("profile") != profile.to_dict()
            or binding.get("profile_sha256") != canonical_sha256(profile.to_dict())
            or not isinstance(binding.get("catalog_evidence"), Mapping)
        ):
            raise PilotV21111DispatchRefreshError("refresh profile bindings drifted")
        evidence[profile_id] = binding["catalog_evidence"]
    checked = _validate_catalog_evidence(contract, evidence)

    # Production evidence binds the independently persisted zero-completion
    # catalog receipt.  Isolated tests may intentionally provide only the two
    # document hashes; in production these two fields are all-or-nothing.
    receipt_fields = {
        "catalog_receipt_file_sha256",
        "catalog_receipt_sha256",
    }
    presence = {
        profile_id: receipt_fields <= set(row) for profile_id, row in checked.items()
    }
    if any(presence.values()) and not all(presence.values()):
        raise PilotV21111DispatchRefreshError("refresh catalog binding drifted")
    if not all(presence.values()):
        return
    catalog_path = root / V21111_REFRESH_DIRECTORY / "provider_catalog.json"
    if catalog_path.is_symlink() or not catalog_path.is_file():
        raise PilotV21111DispatchRefreshError("refresh catalog binding drifted")
    try:
        catalog = verify_provider_catalog_receipt(
            json.loads(catalog_path.read_text(encoding="utf-8")),
            contract_hash=contract.canonical_hash,
        )
    except (OSError, json.JSONDecodeError, ProviderCatalogError) as exc:
        raise PilotV21111DispatchRefreshError(
            "refresh catalog binding drifted"
        ) from exc
    rows = {
        str(row.get("profile_id")): row
        for row in catalog.get("rows", ())
        if isinstance(row, Mapping)
    }
    if set(rows) != {"gpt52_main", "gpt56_diagnostic"}:
        raise PilotV21111DispatchRefreshError("refresh catalog binding drifted")
    file_sha256 = _sha256_bytes(catalog_path.read_bytes())
    for profile_id, bound in checked.items():
        row = rows[profile_id]
        expected = {
            "price_source_url": row.get("price_source_url"),
            "price_source_sha256": row.get("price_source_sha256"),
            "model_reference_url": row.get("model_reference_url"),
            "model_reference_sha256": row.get("model_reference_sha256"),
            "catalog_receipt_file_sha256": file_sha256,
            "catalog_receipt_sha256": catalog.get("receipt_sha256"),
        }
        if any(bound.get(key) != value for key, value in expected.items()):
            raise PilotV21111DispatchRefreshError("refresh catalog binding drifted")


def _verify_budget_prefix(
    *,
    contract: PilotContract,
    root: Path,
    receipt: Mapping[str, Any],
    authority: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    budget_path = root / "budget_ledger.json"
    try:
        budget = PilotBudgetLedger(
            budget_path,
            contract_hash=contract.canonical_hash,
            caps=_budget_caps(contract),
            tamper_evident=True,
            parent_debit=parent_budget_debit_for_v21111(contract),
        ).snapshot()
    except Exception as exc:
        raise PilotV21111DispatchRefreshError("refresh budget ledger drifted") from exc
    prefix = receipt.get("budget_ledger_prefix")
    events = budget.get("events")
    if not isinstance(prefix, Mapping) or not isinstance(events, list):
        raise PilotV21111DispatchRefreshError("refresh budget prefix drifted")
    count = prefix.get("event_count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or len(events) < count
        or events[count - 1].get("event_sha256") != prefix.get("event_chain_head")
        or not math.isclose(
            float(prefix.get("refresh_reserved_cost_usd", -1)),
            float(plan["reserved_cost_usd"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or (
            len(events) == count
            and prefix.get("ledger_sha256") != budget.get("ledger_sha256")
        )
    ):
        raise PilotV21111DispatchRefreshError("refresh budget prefix drifted")

    projections = {row.run_id: row for row in refresh_projections(contract)}
    definitions = _definition_by_id(plan)
    budget_rows = budget.get("runs")
    if not isinstance(budget_rows, Mapping):
        raise PilotV21111DispatchRefreshError("refresh budget rows drifted")
    actual_cost = 0.0
    for plan_row in plan["rows"]:
        run_id = str(plan_row["run_id"])
        authority_row = authority["rows"][run_id]
        budget_row = budget_rows.get(run_id)
        status = authority_row.get("status")
        profile = contract.provider_profiles[str(plan_row["profile_id"])]
        definition = definitions[str(plan_row["probe_id"])]
        expected_request = _request_record(
            profile=profile,
            row=plan_row,
            definition=definition,
        )
        if status == "complete":
            if not isinstance(budget_row, Mapping):
                raise PilotV21111DispatchRefreshError(
                    "complete refresh budget row is absent"
                )
            actual_cost += _validate_complete_row(
                plan_row=plan_row,
                definition=definition,
                profile=profile,
                authority_row=authority_row,
                budget_row=budget_row,
                projection=projections[run_id],
            )
        elif status == "failed":
            actual = (
                budget_row.get("actual") if isinstance(budget_row, Mapping) else None
            )
            if (
                not isinstance(authority_row.get("failure"), Mapping)
                or not isinstance(budget_row, Mapping)
                or budget_row.get("reservation") != projections[run_id].to_dict()
                or budget_row.get("status") not in BUDGET_TERMINAL_STATUSES
                or not isinstance(actual, Mapping)
                or actual.get("completions") != 1
                or actual.get("storage_bytes") != 0
                or float(actual.get("cost_usd", -1)) < 0
                or float(actual.get("cost_usd", -1))
                > projections[run_id].cost_usd + 1e-12
                or (
                    authority_row.get("failure", {}).get("error_type")
                    != "RefreshStateMismatch"
                    and budget_row.get("failure") != authority_row.get("failure")
                )
                or (
                    authority_row.get("request") is not None
                    and authority_row.get("request") != expected_request
                    and authority_row.get("failure", {}).get("error_type")
                    != "RefreshStateMismatch"
                )
            ):
                raise PilotV21111DispatchRefreshError("failed refresh row drifted")
            if isinstance(authority_row.get("response"), Mapping):
                passed, failure, _ = _stored_response_outcome(
                    profile=profile,
                    definition=definition,
                    response=authority_row["response"],
                )
                if (
                    passed
                    and authority_row.get("failure")
                    != {"error_type": "RefreshStateMismatch"}
                ) or (not passed and failure != authority_row.get("failure")):
                    raise PilotV21111DispatchRefreshError(
                        "failed refresh response drifted"
                    )
            actual_cost += float(actual["cost_usd"])
        elif status == "skipped":
            if (
                budget_row is not None
                or authority_row.get("request") is not None
                or authority_row.get("response") is not None
                or authority_row.get("failure") != {"error_type": "RefreshGlobalNoGo"}
            ):
                raise PilotV21111DispatchRefreshError("skipped refresh row drifted")
        else:
            raise PilotV21111DispatchRefreshError(
                "refresh receipt does not bind a terminal denominator"
            )
    if not math.isclose(
        float(prefix.get("refresh_actual_cost_usd", -1)),
        actual_cost,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PilotV21111DispatchRefreshError("refresh actual cost drifted")


def verify_dispatch_refresh_terminal(
    *,
    contract: PilotContract,
    raw_root: str | Path,
    paid: Any,
) -> dict[str, Any]:
    """Deeply verify the immutable terminal refresh receipt, GO or no-go."""

    root = Path(raw_root).absolute()
    path = root / V21111_REFRESH_DIRECTORY / V21111_REFRESH_RECEIPT_FILENAME
    if path.is_symlink() or not path.is_file():
        raise PilotV21111DispatchRefreshError("dispatch refresh GO receipt is absent")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    integrity = receipt.get("integrity")
    unsigned = dict(receipt)
    unsigned.pop("integrity", None)
    try:
        plan = _refresh_plan(contract)
        expected_fields = {
            "schema_version",
            "status",
            "go",
            "contract_id",
            "contract_sha256",
            "release_provenance",
            "acceptance_binding",
            "plan_sha256",
            "profile_bindings",
            "provider_calls_attempted",
            "row_status_counts",
            "authority_ledger_binding",
            "budget_ledger_prefix",
            "science_capacity",
            "scientific_evidence",
            "claim_boundary",
            "integrity",
        }
        release = receipt.get("release_provenance")
        if (
            set(receipt) != expected_fields
            or not isinstance(integrity, Mapping)
            or integrity.get("canonicalization") != "json-sort-keys-utf8-v1"
            or integrity.get("content_sha256") != canonical_sha256(unsigned)
            or receipt.get("schema_version") != V21111_REFRESH_SCHEMA
            or receipt.get("status") not in {"go", "no-go"}
            or receipt.get("go") is not (receipt.get("status") == "go")
            or receipt.get("contract_id") != contract.contract_id
            or receipt.get("contract_sha256") != contract.canonical_hash
            or receipt.get("plan_sha256") != canonical_sha256(plan)
            or receipt.get("scientific_evidence") is not False
            or receipt.get("claim_boundary")
            != "identity-interface-length-cap-refresh-not-p95"
            or not isinstance(release, Mapping)
            or release.get("git_tag") != getattr(paid, "git_tag", None)
            or release.get("git_commit") != getattr(paid, "head_commit", None)
            or release.get("worktree_clean") is not True
        ):
            raise PilotV21111DispatchRefreshError("dispatch refresh receipt drifted")

        acceptance = verify_scientific_dispatch_acceptance(
            contract=contract,
            raw_root=root,
            paid=paid,
        )
        acceptance_binding = receipt.get("acceptance_binding")
        if (
            not isinstance(acceptance_binding, Mapping)
            or acceptance_binding.get("path") != str(root / V21111_ACCEPTANCE_FILENAME)
            or acceptance_binding.get("content_sha256")
            != acceptance["integrity"]["content_sha256"]
        ):
            raise PilotV21111DispatchRefreshError("dispatch refresh receipt drifted")

        profile_bindings = receipt.get("profile_bindings")
        if not isinstance(profile_bindings, Mapping):
            raise PilotV21111DispatchRefreshError("dispatch refresh receipt drifted")
        _verify_catalog_binding(
            contract=contract,
            root=root,
            profile_bindings=profile_bindings,
        )

        ledger_path = root / V21111_REFRESH_DIRECTORY / V21111_REFRESH_LEDGER_FILENAME
        ledger = DispatchRefreshLedger(ledger_path, contract=contract).snapshot()
        binding = receipt.get("authority_ledger_binding")
        if (
            not isinstance(binding, Mapping)
            or set(binding)
            != {
                "path",
                "file_sha256",
                "ledger_sha256",
                "event_count",
                "event_chain_head",
            }
            or binding.get("path") != str(ledger_path)
            or binding.get("file_sha256") != _sha256_bytes(ledger_path.read_bytes())
            or binding.get("ledger_sha256") != ledger["ledger_sha256"]
            or binding.get("event_count") != len(ledger["events"])
            or binding.get("event_chain_head") != ledger["events"][-1]["event_sha256"]
        ):
            raise PilotV21111DispatchRefreshError("dispatch refresh receipt drifted")
        statuses = [row["status"] for row in ledger["rows"].values()]
        counts = {status: statuses.count(status) for status in sorted(set(statuses))}
        expected_go = len(statuses) == 20 and all(
            status == "complete" for status in statuses
        )
        expected_attempted = sum(
            row["status"] in {"running", "returned", "complete", "failed"}
            and row.get("started_at") is not None
            for row in ledger["rows"].values()
        )
        if (
            receipt.get("row_status_counts") != counts
            or receipt.get("provider_calls_attempted") != expected_attempted
        ):
            raise PilotV21111DispatchRefreshError("dispatch refresh receipt drifted")

        _verify_budget_prefix(
            contract=contract,
            root=root,
            receipt=receipt,
            authority=ledger,
            plan=plan,
        )
        capacity = receipt.get("science_capacity")
        fixed = _fixed_capacity(contract)
        prefix = receipt["budget_ledger_prefix"]
        actual_cost = float(prefix["refresh_actual_cost_usd"])
        expected_live = {
            "cost_usd": round(
                float(contract.budgets["total_usd"])
                - fixed["parent"]["cost_usd"]
                - actual_cost,
                12,
            ),
            "hosted_completions": int(contract.budgets["max_provider_completions"])
            - fixed["parent"]["hosted_completions"]
            - sum(status != "skipped" for status in statuses),
        }
        expected_capacity = {
            **fixed,
            "live_global_headroom_before_science": expected_live,
            "complete_matrix_fits": (
                fixed["final_remaining_after_refresh"]["cost_usd"] >= -1e-12
                and fixed["final_remaining_after_refresh"]["hosted_completions"] >= 0
                and expected_live["cost_usd"] + 1e-12
                >= fixed["science_reserve"]["cost_usd"]
                and expected_live["hosted_completions"]
                >= fixed["science_reserve"]["hosted_completions"]
            ),
        }
        if capacity != expected_capacity:
            raise PilotV21111DispatchRefreshError("dispatch refresh receipt drifted")
        expected_go = expected_go and expected_capacity["complete_matrix_fits"]
        if receipt.get("go") is not expected_go:
            raise PilotV21111DispatchRefreshError("dispatch refresh receipt drifted")
        return receipt
    except PilotV21111DispatchRefreshError as exc:
        if str(exc) == "dispatch refresh receipt is absent":
            raise
        raise PilotV21111DispatchRefreshError(
            f"dispatch refresh receipt drifted: {exc}"
        ) from exc
    except Exception as exc:
        raise PilotV21111DispatchRefreshError(
            f"dispatch refresh receipt drifted: {type(exc).__name__}"
        ) from exc


def verify_dispatch_refresh_go(
    *,
    contract: PilotContract,
    raw_root: str | Path,
    paid: Any,
) -> dict[str, Any]:
    """Require one deeply verified terminal GO shared by all science stages."""

    receipt = verify_dispatch_refresh_terminal(
        contract=contract,
        raw_root=raw_root,
        paid=paid,
    )
    if receipt.get("status") != "go" or receipt.get("go") is not True:
        raise PilotV21111DispatchRefreshError(
            "dispatch refresh receipt drifted: terminal receipt is no-go"
        )
    return receipt


def _terminal_budget_row(row: Any) -> bool:
    return (
        isinstance(row, Mapping)
        and row.get("status") in BUDGET_TERMINAL_STATUSES
        and isinstance(row.get("actual"), Mapping)
    )


def _budget_actual_matches(
    row: Mapping[str, Any],
    *,
    projection: RunProjection,
    status: str,
    cost_usd: float,
    failure: Mapping[str, Any] | None,
) -> bool:
    actual = row.get("actual")
    return (
        row.get("reservation") == projection.to_dict()
        and row.get("status") == status
        and row.get("failure") == (None if failure is None else dict(failure))
        and isinstance(actual, Mapping)
        and actual.get("completions") == 1
        and actual.get("storage_bytes") == 0
        and math.isclose(
            float(actual.get("cost_usd", -1)),
            cost_usd,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


def _reconcile_existing_rows(
    *,
    contract: PilotContract,
    ledger: DispatchRefreshLedger,
    budget_ledger: PilotBudgetLedger,
    plan: Mapping[str, Any],
    projections: Mapping[str, RunProjection],
    definitions: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Close every durable crash window without ever redispatching a row."""

    global_no_go = False
    for plan_row in plan["rows"]:
        run_id = str(plan_row["run_id"])
        authority_row = ledger.snapshot()["rows"][run_id]
        budget_row = budget_ledger.snapshot()["runs"].get(run_id)
        projection = projections[run_id]
        profile = contract.provider_profiles[str(plan_row["profile_id"])]
        definition = definitions[str(plan_row["probe_id"])]
        expected_request = _request_record(
            profile=profile,
            row=plan_row,
            definition=definition,
        )
        authority_status = authority_row["status"]
        budget_status = None if budget_row is None else str(budget_row.get("status"))

        if isinstance(budget_row, Mapping) and (
            budget_row.get("reservation") != projection.to_dict()
            or budget_row.get("stage_bucket") != "dispatch_refresh"
        ):
            raise PilotV21111DispatchRefreshError("refresh budget reservation drifted")

        if authority_status == "scheduled":
            if budget_row is None:
                continue
            failure = {
                "error_type": (
                    "InterruptedDispatchRefresh"
                    if budget_status == "reserved"
                    else "RefreshStateMismatch"
                )
            }
            if budget_status == "reserved":
                budget_ledger.finalize(
                    run_id,
                    status="integrity-stopped",
                    cost_usd=projection.cost_usd,
                    completions=1,
                    storage_bytes=0,
                    failure=failure,
                )
            elif not _terminal_budget_row(budget_row):
                raise PilotV21111DispatchRefreshError("refresh budget state drifted")
            ledger.finalize(run_id, status="failed", failure=failure)
            global_no_go = True
            continue

        if authority_status == "running":
            if authority_row.get("request") != expected_request or (
                budget_row is not None and budget_status != "reserved"
            ):
                # A durable marker with a non-frozen request can never be
                # rehabilitated into an accepted terminal receipt.
                failure = {"error_type": "RefreshStateMismatch"}
            else:
                failure = {"error_type": "InterruptedDispatchRefresh"}
            if budget_row is None:
                budget_ledger.reserve(projection)
                budget_row = budget_ledger.snapshot()["runs"][run_id]
                budget_status = "reserved"
            if budget_status == "reserved":
                budget_ledger.finalize(
                    run_id,
                    status="integrity-stopped",
                    cost_usd=projection.cost_usd,
                    completions=1,
                    storage_bytes=0,
                    failure=failure,
                )
            elif not _terminal_budget_row(budget_row):
                raise PilotV21111DispatchRefreshError("refresh budget state drifted")
            ledger.finalize(run_id, status="failed", failure=failure)
            global_no_go = True
            continue

        if authority_status == "returned":
            if authority_row.get("request") != expected_request:
                raise PilotV21111DispatchRefreshError(
                    "returned refresh request drifted"
                )
            response = authority_row.get("response")
            if not isinstance(response, Mapping):
                raise PilotV21111DispatchRefreshError(
                    "returned refresh response is absent"
                )
            passed, failure, expected_cost = _stored_response_outcome(
                profile=profile,
                definition=definition,
                response=response,
            )
            terminal_budget_status = "complete" if passed else "integrity-stopped"
            charged_cost = expected_cost if passed else projection.cost_usd
            if budget_row is None:
                budget_ledger.reserve(projection)
                budget_row = budget_ledger.snapshot()["runs"][run_id]
                budget_status = "reserved"
            if budget_status == "reserved":
                budget_ledger.finalize(
                    run_id,
                    status=terminal_budget_status,
                    cost_usd=charged_cost,
                    completions=1,
                    storage_bytes=0,
                    failure=failure,
                )
            elif not _terminal_budget_row(budget_row) or not _budget_actual_matches(
                budget_row,
                projection=projection,
                status=terminal_budget_status,
                cost_usd=charged_cost,
                failure=failure,
            ):
                raise PilotV21111DispatchRefreshError(
                    "returned refresh budget outcome drifted"
                )
            ledger.finalize(
                run_id,
                status="complete" if passed else "failed",
                failure=failure,
            )
            global_no_go = global_no_go or not passed
            continue

        if authority_status == "complete":
            if not isinstance(budget_row, Mapping):
                failure = {"error_type": "RefreshStateMismatch"}
                budget_ledger.reserve(projection)
                budget_ledger.finalize(
                    run_id,
                    status="integrity-stopped",
                    cost_usd=projection.cost_usd,
                    completions=1,
                    storage_bytes=0,
                    failure=failure,
                )
                ledger.invalidate_complete(
                    run_id,
                    failure=failure,
                )
                global_no_go = True
                continue
            try:
                _validate_complete_row(
                    plan_row=plan_row,
                    definition=definition,
                    profile=profile,
                    authority_row=authority_row,
                    budget_row=budget_row,
                    projection=projection,
                )
            except PilotV21111DispatchRefreshError:
                failure = {"error_type": "RefreshStateMismatch"}
                if budget_status == "reserved":
                    budget_ledger.finalize(
                        run_id,
                        status="integrity-stopped",
                        cost_usd=projection.cost_usd,
                        completions=1,
                        storage_bytes=0,
                        failure=failure,
                    )
                ledger.invalidate_complete(
                    run_id,
                    failure=failure,
                )
                global_no_go = True
            continue

        if authority_status in {"failed", "skipped"}:
            global_no_go = True
            continue

        raise PilotV21111DispatchRefreshError("refresh authority state drifted")
    return global_no_go


def _skip_after_global_no_go(ledger: DispatchRefreshLedger) -> None:
    for run_id, row in ledger.snapshot()["rows"].items():
        if row["status"] == "scheduled":
            ledger.finalize(
                run_id,
                status="skipped",
                failure={"error_type": "RefreshGlobalNoGo"},
            )


def execute_dispatch_refresh(
    *,
    contract: PilotContract,
    raw_root: str | Path,
    paid: Any,
    budget_ledger: PilotBudgetLedger,
    provider_factory: Callable[[str, ProviderRequestProfile], RefreshProvider],
    catalog_evidence: Mapping[str, Mapping[str, Any]],
    resume: bool,
) -> dict[str, Any]:
    """Execute or conservatively recover the frozen twenty-call authority."""

    if contract.status != "frozen":
        raise PilotV21111DispatchRefreshError("refresh requires a frozen contract")
    root = Path(raw_root).absolute()
    acceptance = verify_scientific_dispatch_acceptance(
        contract=contract,
        raw_root=root,
        paid=paid,
    )
    evidence = _validate_catalog_evidence(contract, catalog_evidence)
    plan = _refresh_plan(contract)
    receipt_path = root / V21111_REFRESH_DIRECTORY / V21111_REFRESH_RECEIPT_FILENAME
    if receipt_path.exists():
        if not resume:
            raise PilotV21111DispatchRefreshError(
                "dispatch refresh is terminal; use --resume for verification"
            )
        return verify_dispatch_refresh_go(contract=contract, raw_root=root, paid=paid)

    ledger = DispatchRefreshLedger(
        root / V21111_REFRESH_DIRECTORY / V21111_REFRESH_LEDGER_FILENAME,
        contract=contract,
    )
    projections = {row.run_id: row for row in refresh_projections(contract)}
    definitions = _definition_by_id(plan)
    _assert_whole_plan_fits(
        contract=contract,
        budget_ledger=budget_ledger,
        projections=projections,
    )

    # Reconcile all durable windows before considering another dispatch.  A
    # returned response is committed from its stored audit; a merely running
    # marker consumes the full reservation and permanently fails the refresh.
    if _reconcile_existing_rows(
        contract=contract,
        ledger=ledger,
        budget_ledger=budget_ledger,
        plan=plan,
        projections=projections,
        definitions=definitions,
    ):
        _skip_after_global_no_go(ledger)
        return _write_receipt(
            contract=contract,
            raw_root=root,
            paid=paid,
            acceptance=acceptance,
            plan=plan,
            ledger=ledger,
            budget_ledger=budget_ledger,
            catalog_evidence=evidence,
        )

    providers: dict[str, RefreshProvider] = {}
    failed = False
    for plan_row in plan["rows"]:
        run_id = str(plan_row["run_id"])
        current_status = ledger.snapshot()["rows"][run_id]["status"]
        if current_status == "complete":
            continue
        if failed:
            if current_status == "scheduled":
                ledger.finalize(
                    run_id,
                    status="skipped",
                    failure={"error_type": "RefreshGlobalNoGo"},
                )
            continue
        if current_status != "scheduled":
            raise PilotV21111DispatchRefreshError(
                "refresh reconciliation left a non-dispatchable row"
            )
        projection = projections[run_id]
        profile_id = str(plan_row["profile_id"])
        profile = contract.provider_profiles[profile_id]
        definition = definitions[str(plan_row["probe_id"])]
        budget_ledger.reserve(projection)
        request = _request_record(
            profile=profile,
            row=plan_row,
            definition=definition,
        )
        ledger.begin(run_id, request)
        completion: StructuredCompletion | None = None
        failure: dict[str, Any] | None = None
        try:
            provider = providers.get(profile_id)
            if provider is None:
                provider = provider_factory(profile_id, profile)
                providers[profile_id] = provider
            completion = provider.get_structured_completion(
                _copy(definition["messages"]),
                temperature=0.0,
                top_p=1.0,
                max_tokens=int(plan_row["max_completion_tokens"]),
                max_retries=1,
                seed=None,
            )
            if not isinstance(completion, StructuredCompletion):
                raise TypeError("refresh provider returned a non-structured result")
            response, passed, failure = _completion_record(
                profile=profile,
                definition=definition,
                completion=completion,
            )
            ledger.returned(run_id, response)
            actual_cost = float(completion.cost) if passed else projection.cost_usd
            budget_ledger.finalize(
                run_id,
                status="complete" if passed else "integrity-stopped",
                cost_usd=actual_cost,
                completions=1,
                storage_bytes=0,
                failure=failure,
            )
            ledger.finalize(
                run_id,
                status="complete" if passed else "failed",
                failure=failure,
            )
            failed = not passed
        except Exception:
            # Re-enter through the same durable-state reconciliation used by a
            # restarted process.  This covers running, returned, budget-
            # finalized, and authority-finalized exception windows without a
            # contradictory second finalize or a redispatch.
            failed = _reconcile_existing_rows(
                contract=contract,
                ledger=ledger,
                budget_ledger=budget_ledger,
                plan=plan,
                projections=projections,
                definitions=definitions,
            )

    if failed:
        _skip_after_global_no_go(ledger)

    return _write_receipt(
        contract=contract,
        raw_root=root,
        paid=paid,
        acceptance=acceptance,
        plan=plan,
        ledger=ledger,
        budget_ledger=budget_ledger,
        catalog_evidence=evidence,
    )


__all__ = [
    "DispatchRefreshLedger",
    "PilotV21111DispatchRefreshError",
    "V21111_REFRESH_DIRECTORY",
    "V21111_REFRESH_LEDGER_FILENAME",
    "V21111_REFRESH_RECEIPT_FILENAME",
    "execute_dispatch_refresh",
    "refresh_projections",
    "verify_dispatch_refresh_go",
    "verify_dispatch_refresh_terminal",
]
