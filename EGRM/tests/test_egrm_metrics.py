import pytest

from egrm.metrics import compute_lifecycle_metrics


def rule(rule_id: str, status: str) -> dict[str, object]:
    return {"rule_id": rule_id, "status": status}


def event(rule_id: str, event_type: str, timestamp: int) -> dict[str, object]:
    return {
        "rule_id": rule_id,
        "event_type": event_type,
        "timestamp": timestamp,
    }


def test_unsupported_activation_rate_uses_registered_unsupported_candidates():
    metrics = compute_lifecycle_metrics(
        rules=[
            rule("supported", "active"),
            rule("unsupported-active", "active"),
            rule("unsupported-rejected", "rejected"),
        ],
        events=[
            event("supported", "rule_activated", 1),
            event("unsupported-active", "rule_activated", 1),
        ],
        supported_rule_ids={
            "supported": True,
            "unsupported-active": False,
            "unsupported-rejected": False,
        },
        known_episode_ids=set(),
    )
    assert metrics["unsupported_registered_count"] == 2
    assert metrics["unsupported_activation_rate"] == 0.5
    assert metrics["unsupported_share_among_activated"] == 0.5


def test_retirement_latency_records_include_right_censoring():
    metrics = compute_lifecycle_metrics(
        rules=[rule("retired", "retired"), rule("survived", "active")],
        events=[
            event("retired", "rule_activated", 1),
            event("retired", "harmful_compliance_evidence_added", 3),
            event("retired", "rule_retired", 5),
            event("survived", "rule_activated", 1),
            event("survived", "harmful_compliance_evidence_added", 4),
        ],
        supported_rule_ids={"retired": False, "survived": False},
        known_episode_ids=set(),
        observation_end_t=7,
    )
    assert metrics["retirement_latency_steps"] == [2]
    assert metrics["retirement_latency_records"] == [
        {
            "rule_id": "retired",
            "first_harmful_compliance_t": 3,
            "terminal_t": 5,
            "latency_steps": 2,
            "right_censored": False,
        },
        {
            "rule_id": "survived",
            "first_harmful_compliance_t": 4,
            "terminal_t": 7,
            "latency_steps": 3,
            "right_censored": True,
        },
    ]


def test_retirement_before_harm_is_rejected():
    with pytest.raises(ValueError, match="precedes"):
        compute_lifecycle_metrics(
            rules=[rule("broken", "retired")],
            events=[
                event("broken", "rule_activated", 0),
                event("broken", "rule_retired", 1),
                event("broken", "harmful_compliance_evidence_added", 2),
            ],
            supported_rule_ids={"broken": False},
            known_episode_ids=set(),
        )
