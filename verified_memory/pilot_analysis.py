"""Preregistered analyses for the FinEvo mechanism micro-pilot.

The functions in this module are deliberately pure: they consume sealed stream
rows and return JSON-serialisable summaries.  They never choose a favourable
seed, tune a utility parameter, or silently remove a failed run.  This keeps the
analysis contract usable both by the no-network mini matrix and by the hosted
pilot after ``pilot-v1`` is tagged.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


PILOT_ANALYSIS_SCHEMA_VERSION = "finevo-pilot-analysis-v1"
DEFAULT_SHOCK_SCHEDULE = (
    {"start": 0, "end": 4, "interest_rate": 0.03, "phase": "pre-shock"},
    {"start": 5, "end": 7, "interest_rate": 0.08, "phase": "shock"},
    {"start": 8, "end": 11, "interest_rate": 0.03, "phase": "recovery"},
)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _rows(records: Mapping[str, Sequence[Mapping[str, Any]]], name: str) -> tuple[Mapping[str, Any], ...]:
    value = records.get(name, ())
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"records[{name!r}] must be a sequence")
    rows = tuple(value)
    if any(not isinstance(row, Mapping) for row in rows):
        raise TypeError(f"records[{name!r}] must contain mappings")
    return rows


def normalize_shock_schedule(
    schedule: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Validate an exhaustive, non-overlapping integer-period shock schedule."""

    source = DEFAULT_SHOCK_SCHEDULE if schedule is None else schedule
    if isinstance(source, (str, bytes)) or not isinstance(source, Sequence):
        raise TypeError("shock schedule must be a sequence")
    normalized: list[dict[str, Any]] = []
    occupied: set[int] = set()
    for index, row in enumerate(source):
        if not isinstance(row, Mapping):
            raise TypeError(f"shock schedule row {index} must be a mapping")
        start = _integer(row.get("start"), f"shock[{index}].start")
        end = _integer(row.get("end"), f"shock[{index}].end")
        if start < 0 or end < start:
            raise ValueError(f"shock schedule row {index} has an invalid interval")
        rate = _finite(row.get("interest_rate"), f"shock[{index}].interest_rate")
        if rate < 0:
            raise ValueError("interest rates in the frozen schedule must be nonnegative")
        phase = row.get("phase")
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError(f"shock[{index}].phase must be non-empty")
        periods = set(range(start, end + 1))
        if occupied & periods:
            raise ValueError("shock schedule intervals overlap")
        occupied |= periods
        normalized.append(
            {
                "start": start,
                "end": end,
                "interest_rate": rate,
                "phase": phase,
            }
        )
    normalized.sort(key=lambda row: row["start"])
    if normalized and normalized[0]["start"] != 0:
        raise ValueError("shock schedule must begin at decision_t=0")
    for left, right in zip(normalized, normalized[1:]):
        if left["end"] + 1 != right["start"]:
            raise ValueError("shock schedule must not contain gaps")
    return tuple(normalized)


def phase_for_period(
    decision_t: int,
    schedule: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    decision_t = _integer(decision_t, "decision_t")
    for row in normalize_shock_schedule(schedule):
        if row["start"] <= decision_t <= row["end"]:
            return str(row["phase"])
    raise ValueError(f"decision_t={decision_t} lies outside the shock schedule")


def _period_means(
    rows: Iterable[Mapping[str, Any]], field: str
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        period = _integer(row.get("period"), "utility period")
        grouped[period].append(_finite(row.get(field), f"utility.{field}"))
    return {period: mean(values) for period, values in sorted(grouped.items())}


def _recovery_time(
    period_utility: Mapping[int, float],
    *,
    baseline: float,
    recovery_start: int,
    tolerance_fraction: float = 0.10,
    consecutive_periods: int = 2,
) -> int | None:
    tolerance = max(abs(baseline) * tolerance_fraction, 1e-12)
    ordered = sorted(period for period in period_utility if period >= recovery_start)
    for start in ordered:
        window = tuple(range(start, start + consecutive_periods))
        if all(
            period in period_utility
            and abs(period_utility[period] - baseline) <= tolerance
            for period in window
        ):
            return start - recovery_start
    return None


def _action_row_value(row: Mapping[str, Any], field: str) -> float:
    decision = row.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("action row is missing decision")
    return _finite(decision.get(field), f"action.decision.{field}")


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= 1e-12:
        return None
    return numerator / denominator


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _agent_integer(value: Any, name: str) -> int:
    """Accept the integer/string agent-ID split used by runner streams."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.strip().isdigit():
        result = int(value.strip())
    else:
        raise TypeError(f"{name} must be a non-negative integer")
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _optional_integer(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, name)


def _run_identity_from_episodes(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[str | None, int | None]:
    """Recover run/seed provenance when full EpisodeRecord rows are available."""

    episodes = _rows(records, "episodes")
    run_ids = {
        _nonempty_string(row["run_id"], "episode.run_id")
        for row in episodes
        if row.get("run_id") is not None
    }
    seeds = {
        _integer(row["seed"], "episode.seed")
        for row in episodes
        if row.get("seed") is not None
    }
    if len(run_ids) > 1:
        raise ValueError("episode rows contain multiple run IDs")
    if len(seeds) > 1:
        raise ValueError("episode rows contain multiple seeds")
    return (
        next(iter(run_ids)) if run_ids else None,
        next(iter(seeds)) if seeds else None,
    )


def _rule_reliability_by_agent_rule_family(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    action_rows: Sequence[Mapping[str, Any]],
    utility_rows: Sequence[Mapping[str, Any]],
    context_rows: Sequence[Mapping[str, Any]],
    rules: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    injected_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Retain the seed-agent-rule-family unit behind aggregate C diagnostics.

    A lifecycle event can establish that a rule was active, but it is not an
    actor exposure.  Exposure is counted only when that same agent's action row
    names a member of the family in ``selected_rule_ids``.
    """

    run_id, seed = _run_identity_from_episodes(records)
    rule_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    family_rules: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    family_by_rule: dict[tuple[int, str], tuple[int, str]] = {}
    for index, rule in enumerate(rules):
        agent_id = _agent_integer(
            rule.get("agent_id"), f"semantic_rules[{index}].agent_id"
        )
        rule_id = _nonempty_string(
            rule.get("rule_id"), f"semantic_rules[{index}].rule_id"
        )
        family_id = _nonempty_string(
            rule.get("rule_family_id"),
            f"semantic_rules[{index}].rule_family_id",
        )
        key = (agent_id, rule_id)
        if key in rule_by_key:
            raise ValueError(
                f"duplicate semantic rule identity agent={agent_id}, rule={rule_id}"
            )
        family_key = (agent_id, family_id)
        rule_by_key[key] = rule
        family_by_rule[key] = family_key
        family_rules[family_key].append(rule)

    injection_by_rule: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for index, injection in enumerate(injected_rows):
        agent_id = _agent_integer(
            injection.get("agent_id"),
            f"error_rule_injections[{index}].agent_id",
        )
        rule_id = _nonempty_string(
            injection.get("rule_id"),
            f"error_rule_injections[{index}].rule_id",
        )
        key = (agent_id, rule_id)
        if key not in rule_by_key:
            raise ValueError(
                "error-rule injection references an unknown agent/rule identity"
            )
        injection_by_rule[key].append(injection)

    events_by_family: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for index, event in enumerate(events):
        rule_id_value = event.get("rule_id")
        if rule_id_value is None:
            continue
        agent_id = _agent_integer(
            event.get("agent_id"), f"semantic_rule_events[{index}].agent_id"
        )
        rule_id = _nonempty_string(
            rule_id_value, f"semantic_rule_events[{index}].rule_id"
        )
        key = (agent_id, rule_id)
        family_key = family_by_rule.get(key)
        if family_key is None:
            raise ValueError(
                "semantic rule event references an unknown agent/rule identity"
            )
        events_by_family[family_key].append(event)

    context_by_decision: dict[tuple[int, int], Mapping[str, Any]] = {}
    for index, row in enumerate(context_rows):
        key = (
            _integer(
                row.get("decision_t"), f"context_trace[{index}].decision_t"
            ),
            _agent_integer(
                row.get("agent_id"), f"context_trace[{index}].agent_id"
            ),
        )
        if key in context_by_decision:
            raise ValueError("duplicate context trace decision")
        context_by_decision[key] = row

    utility_by_decision: dict[tuple[int, int], Mapping[str, Any]] = {}
    for index, row in enumerate(utility_rows):
        key = (
            _integer(row.get("period"), f"utility_ledger[{index}].period"),
            _agent_integer(
                row.get("agent_id"), f"utility_ledger[{index}].agent_id"
            ),
        )
        if key in utility_by_decision:
            raise ValueError("duplicate utility row for an agent-period")
        utility_by_decision[key] = row

    exposures_by_family: dict[
        tuple[int, str], list[dict[str, Any]]
    ] = defaultdict(list)
    observed_action_keys: set[tuple[int, int]] = set()
    for index, action in enumerate(action_rows):
        decision_t = _integer(
            action.get("decision_t"), f"actions[{index}].decision_t"
        )
        agent_id = _agent_integer(
            action.get("agent_id"), f"actions[{index}].agent_id"
        )
        action_key = (decision_t, agent_id)
        if action_key in observed_action_keys:
            raise ValueError("duplicate action row for an agent-decision")
        observed_action_keys.add(action_key)
        selected = action.get("selected_rule_ids", ())
        if isinstance(selected, (str, bytes)) or not isinstance(selected, Sequence):
            raise ValueError("actions.selected_rule_ids must be a sequence")
        context = context_by_decision.get(action_key, {})
        context_selected = context.get("selected_rule_ids")
        if context_selected is not None:
            if isinstance(context_selected, (str, bytes)) or not isinstance(
                context_selected, Sequence
            ):
                raise ValueError(
                    "context_trace.selected_rule_ids must be a sequence"
                )
            if [str(value) for value in context_selected] != [
                str(value) for value in selected
            ]:
                raise ValueError(
                    "action/context selected_rule_ids disagree for an "
                    "agent-decision"
                )
        selected_by_family: dict[tuple[int, str], set[str]] = defaultdict(set)
        for value in selected:
            rule_id = _nonempty_string(value, "actions.selected_rule_ids item")
            rule_key = (agent_id, rule_id)
            family_key = family_by_rule.get(rule_key)
            if family_key is None:
                owners = sorted(
                    owner for owner, observed_id in rule_by_key if observed_id == rule_id
                )
                if owners:
                    raise ValueError(
                        f"agent {agent_id} selected rule {rule_id!r} owned by "
                        f"agent(s) {owners}"
                    )
                raise ValueError(f"action selected unknown rule {rule_id!r}")
            selected_by_family[family_key].add(rule_id)

        utility = utility_by_decision.get(action_key, {})
        for family_key, selected_ids in sorted(selected_by_family.items()):
            exposure: dict[str, Any] = {
                "decision_t": decision_t,
                "agent_id": agent_id,
                "selected_rule_ids": sorted(selected_ids),
                "action_prompt_hash": action.get("prompt_hash"),
                "context_hash": context.get("context_hash"),
                "context_mode": context.get("context_mode"),
                "flow_utility": (
                    _finite(utility["flow_utility"], "utility.flow_utility")
                    if utility.get("flow_utility") is not None
                    else None
                ),
                "discounted_flow_utility": (
                    _finite(
                        utility["discounted_flow_utility"],
                        "utility.discounted_flow_utility",
                    )
                    if utility.get("discounted_flow_utility") is not None
                    else None
                ),
            }
            exposures_by_family[family_key].append(exposure)

    result: list[dict[str, Any]] = []
    for (agent_id, family_id), family in sorted(family_rules.items()):
        ordered_rules = sorted(
            family,
            key=lambda row: (
                _optional_integer(row.get("rule_version"), "rule.rule_version")
                or 0,
                _optional_integer(row.get("created_at"), "rule.created_at")
                or 0,
                str(row.get("rule_id")),
            ),
        )
        family_events = sorted(
            events_by_family.get((agent_id, family_id), ()),
            key=lambda row: (
                _integer(row.get("timestamp"), "semantic event timestamp"),
                str(row.get("event_id", "")),
            ),
        )
        family_exposures = sorted(
            exposures_by_family.get((agent_id, family_id), ()),
            key=lambda row: (row["decision_t"], row["agent_id"]),
        )
        family_rule_ids = {
            _nonempty_string(row.get("rule_id"), "rule.rule_id")
            for row in ordered_rules
        }
        family_injections = sorted(
            (
                injection
                for rule_id in family_rule_ids
                for injection in injection_by_rule.get((agent_id, rule_id), ())
            ),
            key=lambda row: (
                _integer(row.get("decision_t"), "injection.decision_t"),
                str(row.get("rule_id")),
            ),
        )
        activation_events = [
            event
            for event in family_events
            if (
                event.get("event_type")
                in {"rule_activated", "experimental_rule_injected_active"}
                or event.get("to_status") == "active"
                or event.get("from_status") == "active"
            )
        ]
        active_injection = any(
            injection.get("rule_status") == "active"
            for injection in family_injections
        )
        active_terminal = any(
            rule.get("status") == "active" for rule in ordered_rules
        )
        ever_active = bool(
            activation_events
            or active_injection
            or active_terminal
            or family_exposures
        )

        harmful_events = [
            event
            for event in family_events
            if event.get("event_type") == "harmful_compliance_evidence_added"
        ]
        retirement_events = [
            event
            for event in family_events
            if event.get("event_type") == "rule_retired"
        ]
        first_harmful = harmful_events[0] if harmful_events else None
        first_retirement = retirement_events[0] if retirement_events else None
        matched_retirement = None
        if first_harmful is not None:
            harmful_t = _integer(
                first_harmful.get("timestamp"), "first harmful timestamp"
            )
            harmful_rule_id = first_harmful.get("rule_id")
            matched_retirement = next(
                (
                    event
                    for event in retirement_events
                    if event.get("rule_id") == harmful_rule_id
                    and _integer(
                        event.get("timestamp"), "retirement timestamp"
                    )
                    >= harmful_t
                ),
                None,
            )
        else:
            harmful_t = None
            matched_retirement = first_retirement
        retirement_t = (
            _integer(
                matched_retirement.get("timestamp"), "matched retirement timestamp"
            )
            if matched_retirement is not None
            else None
        )

        latest_rule = ordered_rules[-1]
        rule_identities = [
            {
                "rule_id": _nonempty_string(rule.get("rule_id"), "rule.rule_id"),
                "rule_version": _optional_integer(
                    rule.get("rule_version"), "rule.rule_version"
                ),
                "status": str(rule.get("status", "missing")),
                "created_at": _optional_integer(
                    rule.get("created_at"), "rule.created_at"
                ),
                "updated_at": _optional_integer(
                    rule.get("updated_at"), "rule.updated_at"
                ),
                "storage_injected": bool(rule.get("injected", False)),
                "injection_provenance": (
                    dict(rule["injection_provenance"])
                    if isinstance(rule.get("injection_provenance"), Mapping)
                    else None
                ),
            }
            for rule in ordered_rules
        ]
        injection_provenance = [
            {
                "decision_t": _integer(
                    injection.get("decision_t"), "injection.decision_t"
                ),
                "rule_id": str(injection.get("rule_id")),
                "mode": injection.get("mode"),
                "semantic_policy": injection.get("semantic_policy"),
                "fixed_rule_hash": injection.get("fixed_rule_hash"),
                "verifier_bypassed": injection.get("verifier_bypassed"),
                "rule_status_at_injection": injection.get("rule_status"),
            }
            for injection in family_injections
        ]
        is_injected = bool(family_injections)
        unit_seed = str(seed) if seed is not None else "unknown"
        unit_run = run_id if run_id is not None else "unknown"
        result.append(
            {
                "unit_id": (
                    f"{unit_run}:s{unit_seed}:a{agent_id}:family:{family_id}"
                ),
                "run_id": run_id,
                "seed": seed,
                "agent_id": agent_id,
                "rule_family_id": family_id,
                "rule_ids": [
                    _nonempty_string(rule.get("rule_id"), "rule.rule_id")
                    for rule in ordered_rules
                ],
                "rule_identities": rule_identities,
                "source": "injected" if is_injected else "natural",
                "injected": is_injected,
                "natural": not is_injected,
                "ever_active": ever_active,
                "false_rule_ever_active": ever_active if is_injected else None,
                "first_active_t": (
                    min(
                        [
                            _integer(
                                event.get("timestamp"), "activation timestamp"
                            )
                            for event in activation_events
                        ]
                        + [
                            _integer(
                                injection.get("decision_t"),
                                "injection decision_t",
                            )
                            for injection in family_injections
                            if injection.get("rule_status") == "active"
                        ]
                    )
                    if activation_events or active_injection
                    else (
                        _optional_integer(
                            latest_rule.get("created_at"), "rule.created_at"
                        )
                        if active_terminal
                        else None
                    )
                ),
                "active_exposure_steps": len(family_exposures),
                "actor_exposure_steps": family_exposures,
                "first_harmful_compliance_t": harmful_t,
                "first_harmful_t": harmful_t,
                "first_harmful_compliance": (
                    {
                        "event_id": first_harmful.get("event_id"),
                        "rule_id": first_harmful.get("rule_id"),
                        "timestamp": harmful_t,
                        "episode_ids": list(
                            first_harmful.get("episode_ids", ())
                        ),
                    }
                    if first_harmful is not None
                    else None
                ),
                "harmful_compliance_event_count": len(harmful_events),
                "first_retirement_t": (
                    _integer(
                        first_retirement.get("timestamp"),
                        "first retirement timestamp",
                    )
                    if first_retirement is not None
                    else None
                ),
                "retirement_t": retirement_t,
                "harmful_to_retirement_delay": (
                    retirement_t - harmful_t
                    if harmful_t is not None and retirement_t is not None
                    else None
                ),
                "retirement_event_count": len(retirement_events),
                "terminal_rule_id": _nonempty_string(
                    latest_rule.get("rule_id"), "rule.rule_id"
                ),
                "terminal_status": str(latest_rule.get("status", "missing")),
                "terminal_status_counts": dict(
                    sorted(
                        Counter(
                            str(rule.get("status", "missing"))
                            for rule in ordered_rules
                        ).items()
                    )
                ),
                "lifecycle_event_type_counts": dict(
                    sorted(
                        Counter(
                            str(event.get("event_type", "missing"))
                            for event in family_events
                        ).items()
                    )
                ),
                "lifecycle_event_ids": [
                    str(event.get("event_id"))
                    for event in family_events
                    if event.get("event_id") is not None
                ],
                "injection_provenance": injection_provenance,
            }
        )
    return result


def summarize_run(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_labor_hours: float,
    schedule: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute the frozen per-run metrics used by Stage 0 and Experiments A--C."""

    max_labor_hours = _finite(max_labor_hours, "max_labor_hours")
    if max_labor_hours <= 0:
        raise ValueError("max_labor_hours must be positive")
    frozen_schedule = normalize_shock_schedule(schedule)
    utility_rows = _rows(records, "utility_ledger")
    action_rows = _rows(records, "actions")
    if not utility_rows or not action_rows:
        raise ValueError("a completed pilot run requires utility and action rows")

    period_utility = _period_means(utility_rows, "flow_utility")
    period_discounted = _period_means(utility_rows, "discounted_flow_utility")
    pre_periods = [
        period
        for period in period_utility
        if phase_for_period(period, frozen_schedule) == "pre-shock"
    ]
    if not pre_periods:
        raise ValueError("run has no pre-shock utility observations")
    baseline = mean(period_utility[period] for period in pre_periods)
    shock_recovery_periods = [
        period
        for period in period_utility
        if phase_for_period(period, frozen_schedule) in {"shock", "recovery"}
    ]
    discounted_shock_recovery = sum(
        period_discounted[period] for period in shock_recovery_periods
    )
    utility_deficit_auc = sum(
        max(0.0, baseline - period_utility[period])
        for period in shock_recovery_periods
    )
    recovery_start = min(
        row["start"] for row in frozen_schedule if row["phase"] == "recovery"
    )

    labor = [
        _action_row_value(row, "executed_labor_hours") for row in action_rows
    ]
    consumption = [
        _action_row_value(row, "executed_consumption_rate") for row in action_rows
    ]
    clipped = [
        bool(
            row.get("decision", {}).get("clipped")
            if isinstance(row.get("decision"), Mapping)
            else False
        )
        for row in action_rows
    ]
    component_ratios = [
        ratio
        for row in utility_rows
        if (
            ratio := _safe_ratio(
                _finite(row.get("labor_disutility"), "labor_disutility"),
                _finite(row.get("consumption_utility"), "consumption_utility"),
            )
        )
        is not None
    ]
    residuals = [
        abs(_finite(row.get("budget_residual"), "budget_residual"))
        for row in utility_rows
    ]

    proposals = _rows(records, "semantic_proposals")
    proposal_status = Counter(
        str(row.get("candidate_parse_status", "missing")) for row in proposals
    )
    rules = _rows(records, "semantic_rules")
    rule_status = Counter(str(row.get("status", "missing")) for row in rules)
    events = _rows(records, "semantic_rule_events")
    event_types = Counter(str(row.get("event_type", "missing")) for row in events)
    context_rows = _rows(records, "context_trace")
    route_modes = Counter(str(row.get("context_mode", "missing")) for row in context_rows)
    route_trace_top5 = [
        {
            "decision_t": _integer(row.get("decision_t"), "context_trace.decision_t"),
            "agent_id": _integer(row.get("agent_id"), "context_trace.agent_id"),
            "retrieved_episode_ids": [
                str(item)
                for item in list(row.get("retrieved_episode_ids", ()))[:5]
            ],
        }
        for row in context_rows
    ]
    injected_rows = _rows(records, "error_rule_injections")
    injected_ids = {str(row.get("rule_id")) for row in injected_rows}
    injected_rules = [
        row for row in rules if str(row.get("rule_id")) in injected_ids
    ]
    injected_events = [
        row for row in events if str(row.get("rule_id")) in injected_ids
    ]
    harmful_times = [
        _integer(row.get("timestamp"), "semantic event timestamp")
        for row in injected_events
        if row.get("event_type") == "harmful_compliance_evidence_added"
    ]
    retirement_times = [
        _integer(row.get("timestamp"), "semantic event timestamp")
        for row in injected_events
        if row.get("event_type") == "rule_retired"
    ]
    first_harmful = min(harmful_times) if harmful_times else None
    first_retirement = min(retirement_times) if retirement_times else None
    errors = _rows(records, "errors")
    provider_failures = sum(
        row.get("error_type") not in {None, "CandidateParseError"} for row in errors
    )
    by_agent_rule_family = _rule_reliability_by_agent_rule_family(
        records,
        action_rows=action_rows,
        utility_rows=utility_rows,
        context_rows=context_rows,
        rules=rules,
        events=events,
        injected_rows=injected_rows,
    )

    return {
        "schema_version": PILOT_ANALYSIS_SCHEMA_VERSION,
        "row_counts": {
            "actions": len(action_rows),
            "utility": len(utility_rows),
            "proposals": len(proposals),
            "rules": len(rules),
            "errors": len(errors),
        },
        "utility": {
            "pre_shock_mean": baseline,
            "shock_recovery_discounted": discounted_shock_recovery,
            "utility_deficit_auc": utility_deficit_auc,
            "recovery_periods_to_within_10pct_for_two": _recovery_time(
                period_utility,
                baseline=baseline,
                recovery_start=recovery_start,
            ),
            "period_means": {
                str(period): value for period, value in period_utility.items()
            },
        },
        "actions": {
            "labor_hours_counts": dict(
                sorted(Counter(f"{value:g}" for value in labor).items())
            ),
            "consumption_rate_counts": dict(
                sorted(Counter(f"{value:g}" for value in consumption).items())
            ),
            "zero_labor_rate": sum(value == 0 for value in labor) / len(labor),
            "ceiling_labor_rate": sum(
                math.isclose(value, max_labor_hours, abs_tol=1e-12)
                for value in labor
            )
            / len(labor),
            "interior_labor_rate": sum(
                0 < value < max_labor_hours for value in labor
            )
            / len(labor),
            "interior_consumption_rate": sum(
                0 < value < 1 for value in consumption
            )
            / len(consumption),
            "clipping_count": sum(clipped),
            "clipping_rate": sum(clipped) / len(clipped),
        },
        "guardrails": {
            "max_abs_budget_residual": max(residuals),
            "median_labor_disutility_to_consumption_utility": (
                median(component_ratios) if component_ratios else None
            ),
            "provider_failure_count": provider_failures,
        },
        "memory": {
            "episodic_retrieval_count": sum(
                len(row.get("retrieved_episode_ids", ())) for row in action_rows
            ),
            "active_rule_retrieval_count": sum(
                len(row.get("selected_rule_ids", ())) for row in action_rows
            ),
            "proposal_parse_status_counts": dict(sorted(proposal_status.items())),
            "rule_status_counts": dict(sorted(rule_status.items())),
            "rule_event_type_counts": dict(sorted(event_types.items())),
            "route_mode_counts": dict(sorted(route_modes.items())),
            "context_to_prompt_count": sum(
                bool(row.get("context_to_prompt")) for row in context_rows
            ),
            "context_to_retrieval_count": sum(
                bool(row.get("context_to_retrieval")) for row in context_rows
            ),
            "route_relevance_at_5": route_relevance_at_k(
                records,
                schedule=frozen_schedule,
                k=5,
            ),
            "route_trace_top5": route_trace_top5,
        },
        "rule_reliability": {
            "fixed_rule_injection_count": len(injected_rows),
            "false_rule_ever_active": any(
                row.get("rule_status") == "active" for row in injected_rows
            ),
            "active_exposure_steps": sum(
                any(str(rule_id) in injected_ids for rule_id in row.get("selected_rule_ids", ()))
                for row in action_rows
            ),
            "harmful_compliance_events": len(harmful_times),
            "first_harmful_t": first_harmful,
            "retirement_t": first_retirement,
            "harmful_to_retirement_delay": (
                first_retirement - first_harmful
                if first_harmful is not None and first_retirement is not None
                else None
            ),
            "final_status_counts": dict(
                sorted(
                    Counter(str(row.get("status", "missing")) for row in injected_rules).items()
                )
            ),
            "by_agent_rule_family": by_agent_rule_family,
        },
    }


def stage0_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the preregistered utility-calibration guardrails without outcome tuning."""

    actions = summary.get("actions")
    guardrails = summary.get("guardrails")
    if not isinstance(actions, Mapping) or not isinstance(guardrails, Mapping):
        raise ValueError("summary is missing actions or guardrails")
    ratio = guardrails.get(
        "median_labor_disutility_to_consumption_utility"
    )
    checks = {
        "budget_residual": _finite(
            guardrails.get("max_abs_budget_residual"), "max_abs_budget_residual"
        )
        <= 1e-8,
        "clipping_zero": _integer(
            actions.get("clipping_count"), "clipping_count"
        )
        == 0,
        "ceiling_labor_at_most_50pct": _finite(
            actions.get("ceiling_labor_rate"), "ceiling_labor_rate"
        )
        <= 0.50,
        "zero_labor_at_most_25pct": _finite(
            actions.get("zero_labor_rate"), "zero_labor_rate"
        )
        <= 0.25,
        "interior_labor_at_least_50pct": _finite(
            actions.get("interior_labor_rate"), "interior_labor_rate"
        )
        >= 0.50,
        "interior_consumption_at_least_75pct": _finite(
            actions.get("interior_consumption_rate"), "interior_consumption_rate"
        )
        >= 0.75,
        "component_balance": (
            ratio is not None and 0.5 <= _finite(ratio, "component ratio") <= 2.0
        ),
    }
    return {
        "schema_version": PILOT_ANALYSIS_SCHEMA_VERSION,
        "pass": all(checks.values()),
        "checks": checks,
    }


def route_relevance_at_k(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    schedule: Sequence[Mapping[str, Any]] | None = None,
    k: int = 5,
) -> dict[str, Any]:
    """Measure whether retrieved episodes came from the current shock phase."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    frozen_schedule = normalize_shock_schedule(schedule)
    episode_phase = {
        str(row["episode_id"]): phase_for_period(
            _integer(row["decision_t"], "episode.decision_t"), frozen_schedule
        )
        for row in _rows(records, "episodes")
    }
    numerators: Counter[str] = Counter()
    denominators: Counter[str] = Counter()
    for row in _rows(records, "context_trace"):
        current_phase = phase_for_period(
            _integer(row["decision_t"], "context_trace.decision_t"),
            frozen_schedule,
        )
        retrieved = row.get("retrieved_episode_ids", ())
        if not isinstance(retrieved, Sequence) or isinstance(retrieved, (str, bytes)):
            raise ValueError("retrieved_episode_ids must be a sequence")
        for episode_id in list(retrieved)[:k]:
            if str(episode_id) not in episode_phase:
                raise ValueError("context trace references an unknown episode")
            denominators[current_phase] += 1
            numerators[current_phase] += (
                episode_phase[str(episode_id)] == current_phase
            )
    phases = sorted(set(denominators) | set(numerators))
    return {
        "schema_version": PILOT_ANALYSIS_SCHEMA_VERSION,
        "k": k,
        "by_phase": {
            phase: {
                "relevant": numerators[phase],
                "retrieved": denominators[phase],
                "relevance": (
                    numerators[phase] / denominators[phase]
                    if denominators[phase]
                    else None
                ),
            }
            for phase in phases
        },
    }


def topk_overlap(
    left: Mapping[str, Sequence[Mapping[str, Any]]],
    right: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Return per-decision Jaccard overlap for two paired retrieval routes."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

    def index(records: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[tuple[int, int], set[str]]:
        result: dict[tuple[int, int], set[str]] = {}
        for row in _rows(records, "context_trace"):
            key = (
                _integer(row["decision_t"], "context_trace.decision_t"),
                _integer(row["agent_id"], "context_trace.agent_id"),
            )
            if key in result:
                raise ValueError("duplicate context trace decision")
            values = row.get("retrieved_episode_ids", ())
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError("retrieved_episode_ids must be a sequence")
            result[key] = {str(item) for item in list(values)[:k]}
        return result

    left_index = index(left)
    right_index = index(right)
    if set(left_index) != set(right_index):
        raise ValueError("paired retrieval routes do not share a decision index")
    rows = []
    for key in sorted(left_index):
        union = left_index[key] | right_index[key]
        overlap = 1.0 if not union else len(left_index[key] & right_index[key]) / len(union)
        rows.append(
            {
                "decision_t": key[0],
                "agent_id": key[1],
                "jaccard": overlap,
            }
        )
    return {
        "schema_version": PILOT_ANALYSIS_SCHEMA_VERSION,
        "k": k,
        "mean_jaccard": mean(row["jaccard"] for row in rows) if rows else None,
        "rows": rows,
    }


def paired_delta_summary(
    treatment: Mapping[int, float],
    control: Mapping[int, float],
    *,
    positive_direction: bool = True,
) -> dict[str, Any]:
    """Summarise every paired seed and retain the full ITT seed identity."""

    if set(treatment) != set(control):
        raise ValueError("treatment and control must contain the same paired seeds")
    if not treatment:
        raise ValueError("paired contrast requires at least one seed")
    deltas = {
        int(seed): _finite(treatment[seed], f"treatment[{seed}]")
        - _finite(control[seed], f"control[{seed}]")
        for seed in sorted(treatment)
    }
    values = list(deltas.values())
    direction = {
        seed: (value > 0 if positive_direction else value < 0)
        for seed, value in deltas.items()
    }
    relative = [
        value / abs(_finite(control[seed], f"control[{seed}]"))
        for seed, value in deltas.items()
        if abs(_finite(control[seed], f"control[{seed}]")) > 1e-12
    ]
    return {
        "schema_version": PILOT_ANALYSIS_SCHEMA_VERSION,
        "raw_paired_deltas": {str(seed): value for seed, value in deltas.items()},
        "mean": mean(values),
        "median": median(values),
        "range": [min(values), max(values)],
        "direction_count": sum(direction.values()),
        "pair_count": len(values),
        "median_relative_effect": median(relative) if relative else None,
    }


def retrieval_effect_gate(delta_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply Experiment A's frozen 4/5 and median-relative-effect thresholds."""

    pair_count = _integer(delta_summary.get("pair_count"), "pair_count")
    direction_count = _integer(
        delta_summary.get("direction_count"), "direction_count"
    )
    relative = delta_summary.get("median_relative_effect")
    checks = {
        "at_least_four_complete_pairs": pair_count >= 4,
        "at_least_four_same_direction": direction_count >= 4,
        "median_relative_effect_at_least_5pct": (
            relative is not None
            and _finite(relative, "median_relative_effect") >= 0.05
        ),
    }
    return {
        "schema_version": PILOT_ANALYSIS_SCHEMA_VERSION,
        "support_retrieval_effect": all(checks.values()),
        "checks": checks,
    }


def continuation_effect_gate(
    treatment_deltas: Mapping[int, float],
    *,
    matched_null_deltas: Mapping[int, float],
    action_bin_width: float,
) -> dict[str, Any]:
    """Apply D's matched-null, action-bin, and 4/5 direction gate."""

    if set(treatment_deltas) != set(matched_null_deltas):
        raise ValueError("D treatment and matched-null seeds must match")
    action_bin_width = _finite(action_bin_width, "action_bin_width")
    if action_bin_width <= 0:
        raise ValueError("action_bin_width must be positive")
    null_max = max(abs(_finite(value, "matched null")) for value in matched_null_deltas.values())
    direction_count = sum(
        _finite(value, "treatment delta") > 0
        for value in treatment_deltas.values()
    )
    reverse_count = sum(
        _finite(value, "treatment delta") < 0
        for value in treatment_deltas.values()
    )
    same_direction = max(direction_count, reverse_count)
    magnitude = median(
        abs(_finite(value, "treatment delta"))
        for value in treatment_deltas.values()
    )
    checks = {
        "at_least_four_complete_pairs": len(treatment_deltas) >= 4,
        "at_least_four_same_direction": same_direction >= 4,
        "exceeds_matched_null": magnitude > null_max,
        "exceeds_one_action_bin": magnitude > action_bin_width,
    }
    return {
        "schema_version": PILOT_ANALYSIS_SCHEMA_VERSION,
        "passes": all(checks.values()),
        "checks": checks,
        "matched_null_max_abs": null_max,
        "median_abs_treatment_delta": magnitude,
    }


def validate_itt_denominator(
    expected_run_ids: Sequence[str],
    ledger_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require one terminal ledger row for every preregistered run ID."""

    expected = tuple(str(item) for item in expected_run_ids)
    if len(expected) != len(set(expected)):
        raise ValueError("expected run IDs must be unique")
    observed: dict[str, Mapping[str, Any]] = {}
    for row in ledger_rows:
        if not isinstance(row, Mapping):
            raise TypeError("ledger rows must be mappings")
        run_id = str(row.get("run_id", ""))
        if not run_id:
            raise ValueError("ledger row is missing run_id")
        if run_id in observed:
            raise ValueError(f"duplicate terminal ledger row for {run_id}")
        status = row.get("status")
        if status not in {
            "complete",
            "failed",
            "budget-stopped",
            "integrity-stopped",
            "capability-no-go",
        }:
            raise ValueError(f"run {run_id} has a non-terminal ledger status")
        observed[run_id] = row
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    counts = Counter(str(row["status"]) for row in observed.values())
    return {
        "schema_version": PILOT_ANALYSIS_SCHEMA_VERSION,
        "pass": not missing and not unexpected,
        "expected_count": len(expected),
        "observed_count": len(observed),
        "missing_run_ids": missing,
        "unexpected_run_ids": unexpected,
        "status_counts": dict(sorted(counts.items())),
    }


__all__ = [
    "DEFAULT_SHOCK_SCHEDULE",
    "PILOT_ANALYSIS_SCHEMA_VERSION",
    "continuation_effect_gate",
    "normalize_shock_schedule",
    "paired_delta_summary",
    "phase_for_period",
    "retrieval_effect_gate",
    "route_relevance_at_k",
    "stage0_gate",
    "summarize_run",
    "topk_overlap",
    "validate_itt_denominator",
]
