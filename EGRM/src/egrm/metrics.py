"""Predeclared rule-lifecycle metrics for controlled EGRM evaluations."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


ADMITTED_STATUSES = frozenset({"provisional", "active", "retired"})


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def compute_lifecycle_metrics(
    *,
    rules: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    supported_rule_ids: Mapping[str, bool],
    known_episode_ids: set[str],
    observation_end_t: int | None = None,
) -> dict[str, Any]:
    """Compute metrics without dropping rejected rules or zero denominators."""

    rule_rows = [dict(rule) for rule in rules]
    event_rows = [dict(event) for event in events]
    rule_ids = {str(rule["rule_id"]) for rule in rule_rows}
    if rule_ids != set(supported_rule_ids):
        raise ValueError("ground-truth labels must cover exactly the evaluated rules")

    admitted = [rule for rule in rule_rows if rule["status"] in ADMITTED_STATUSES]
    admitted_supported = sum(
        int(supported_rule_ids[str(rule["rule_id"])]) for rule in admitted
    )
    activated_ids = {
        str(event["rule_id"])
        for event in event_rows
        if event.get("event_type")
        in {
            "rule_activated",
            "experimental_rule_injected_active",
        }
        and event.get("rule_id") is not None
    }
    unsupported_activated = {
        rule_id for rule_id in activated_ids if not supported_rule_ids[rule_id]
    }
    unsupported_registered = {
        rule_id for rule_id, supported in supported_rule_ids.items() if not supported
    }

    event_counts = Counter(str(event.get("event_type")) for event in event_rows)
    event_times = [
        int(event["timestamp"])
        for event in event_rows
        if event.get("timestamp") is not None
    ]
    if observation_end_t is None:
        end_t = max(event_times, default=0)
    elif isinstance(observation_end_t, bool) or not isinstance(observation_end_t, int):
        raise ValueError("observation_end_t must be an integer")
    else:
        end_t = observation_end_t
    if event_times and end_t < max(event_times):
        raise ValueError("observation_end_t precedes a recorded event")

    retirement_latencies: list[int] = []
    retirement_latency_records: list[dict[str, Any]] = []
    for rule_id in sorted(activated_ids):
        harmful_times = [
            int(event["timestamp"])
            for event in event_rows
            if event.get("rule_id") == rule_id
            and event.get("event_type") == "harmful_compliance_evidence_added"
        ]
        retired_times = [
            int(event["timestamp"])
            for event in event_rows
            if event.get("rule_id") == rule_id
            and event.get("event_type") == "rule_retired"
        ]
        if harmful_times:
            first_harm_t = min(harmful_times)
            retirement_after_harm = [
                timestamp for timestamp in retired_times if timestamp >= first_harm_t
            ]
            if retired_times and not retirement_after_harm:
                raise ValueError("a rule retirement precedes its harmful evidence")
            if retirement_after_harm:
                terminal_t = min(retirement_after_harm)
                latency = terminal_t - first_harm_t
                retirement_latencies.append(latency)
                censored = False
            else:
                terminal_t = end_t
                latency = terminal_t - first_harm_t
                censored = True
            retirement_latency_records.append(
                {
                    "rule_id": rule_id,
                    "first_harmful_compliance_t": first_harm_t,
                    "terminal_t": terminal_t,
                    "latency_steps": latency,
                    "right_censored": censored,
                }
            )

    link_fields = (
        "supporting_episode_ids",
        "harmful_compliance_episode_ids",
        "alternative_success_episode_ids",
        "alternative_failure_episode_ids",
        "irrelevant_episode_ids",
    )
    evidence_links: list[str] = []
    for rule in rule_rows:
        for field in link_fields:
            evidence_links.extend(str(item) for item in rule.get(field, []))
    valid_links = sum(int(link in known_episode_ids) for link in evidence_links)
    return {
        "candidate_count": len(rule_rows),
        "admitted_count": len(admitted),
        "rule_admission_precision": _safe_ratio(admitted_supported, len(admitted)),
        "ever_active_count": len(activated_ids),
        "unsupported_registered_count": len(unsupported_registered),
        "unsupported_ever_active_count": len(unsupported_activated),
        "unsupported_activation_rate": _safe_ratio(
            len(unsupported_activated), len(unsupported_registered)
        ),
        "unsupported_share_among_activated": _safe_ratio(
            len(unsupported_activated), len(activated_ids)
        ),
        "harmful_compliance_events": event_counts["harmful_compliance_evidence_added"],
        "alternative_success_events": event_counts[
            "alternative_success_evidence_added"
        ],
        "retired_rule_count": event_counts["rule_retired"],
        "retirement_latency_steps": retirement_latencies,
        "retirement_latency_records": retirement_latency_records,
        "provenance_link_count": len(evidence_links),
        "provenance_coverage": _safe_ratio(valid_links, len(evidence_links)),
        "event_type_counts": dict(sorted(event_counts.items())),
    }
