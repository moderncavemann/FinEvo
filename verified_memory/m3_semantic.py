"""M3: evidence-verified semantic rules with an auditable lifecycle.

An LLM may *propose* a structured rule, but it cannot activate one.  Candidate
support IDs are checked against finalized M2 episodes, counterevidence is
searched in the same ledger, and a rule remains provisional until distinct
post-proposal evidence validates it.  Active rules can later be retired by
contradictory evidence or confidence decay.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
from typing import Any, Mapping, Optional, Sequence

from verified_memory.m2_episodic import EpisodeRecord, EvidenceLinkedEpisodicTrack


SCHEMA_VERSION = "m3-verified-semantic-v1"
CANDIDATE_SCHEMA_VERSION = "m3-rule-candidate-v1"
RULE_SCHEMA_VERSION = "m3-verified-rule-v1"
EVENT_SCHEMA_VERSION = "m3-rule-event-v1"

RULE_STATUSES = frozenset({"provisional", "active", "retired", "rejected"})
CONDITION_OPERATORS = frozenset({">", ">=", "<", "<=", "=="})
ACTION_DIRECTIONS = frozenset({"increase", "decrease", "maintain"})
OUTCOME_METRICS = frozenset(
    {"utility_advantage", "flow_utility", "reward", "wealth_change"}
)
OUTCOME_OPERATORS = frozenset({">", ">=", "<", "<=", "=="})

DEFAULT_CONDITION_FIELDS = (
    "price",
    "interest_rate",
    "low_labor_rate",
    "unemployment_rate",
    "inflation",
    "sentiment",
    "wealth",
    "income",
)
DEFAULT_ACTION_TARGETS = (
    "labor_hours",
    "consumption_fraction",
    "work_propensity",
)


class CandidateParseError(ValueError):
    """Raised when an LLM candidate is not valid against the strict schema."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _strict_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(f"invalid {name} keys: {', '.join(details)}")


def _tuple_of_unique_strings(
    values: Sequence[Any], name: str, *, permit_duplicates: bool = False
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a list")
    result = tuple(_nonempty(value, f"{name} item") for value in values)
    if not permit_duplicates and len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate IDs")
    return result


def _compare(actual: float, operator: str, expected: float, tolerance: float) -> bool:
    if operator == ">":
        return actual > expected
    if operator == ">=":
        return actual >= expected
    if operator == "<":
        return actual < expected
    if operator == "<=":
        return actual <= expected
    if operator == "==":
        return abs(actual - expected) <= tolerance
    raise ValueError(f"unsupported operator: {operator}")


@dataclass(frozen=True)
class ConditionPredicate:
    field: str
    operator: str
    value: float
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", _nonempty(self.field, "condition.field"))
        if self.operator not in CONDITION_OPERATORS:
            raise ValueError(
                f"condition.operator must be one of {sorted(CONDITION_OPERATORS)}"
            )
        object.__setattr__(self, "value", _finite(self.value, "condition.value"))
        tolerance = _finite(self.tolerance, "condition.tolerance")
        if tolerance < 0:
            raise ValueError("condition.tolerance must be non-negative")
        object.__setattr__(self, "tolerance", tolerance)

    def matches(self, state: Mapping[str, Any]) -> bool:
        if self.field not in state:
            return False
        try:
            actual = _finite(state[self.field], f"state[{self.field!r}]")
        except ValueError:
            return False
        return _compare(actual, self.operator, self.value, self.tolerance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConditionPredicate":
        _strict_keys(value, {"field", "operator", "value", "tolerance"}, "condition")
        return cls(**value)


@dataclass(frozen=True)
class ActionGuidance:
    target: str
    direction: str
    threshold: float
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _nonempty(self.target, "action.target"))
        if self.direction not in ACTION_DIRECTIONS:
            raise ValueError(
                f"action.direction must be one of {sorted(ACTION_DIRECTIONS)}"
            )
        object.__setattr__(
            self, "threshold", _finite(self.threshold, "action.threshold")
        )
        tolerance = _finite(self.tolerance, "action.tolerance")
        if tolerance < 0:
            raise ValueError("action.tolerance must be non-negative")
        object.__setattr__(self, "tolerance", tolerance)

    def is_consistent(self, action: Mapping[str, Any]) -> bool:
        if self.target not in action:
            return False
        try:
            actual = _finite(action[self.target], f"action[{self.target!r}]")
        except ValueError:
            return False
        if self.direction == "increase":
            return actual >= self.threshold - self.tolerance
        if self.direction == "decrease":
            return actual <= self.threshold + self.tolerance
        return abs(actual - self.threshold) <= self.tolerance

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "direction": self.direction,
            "threshold": self.threshold,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionGuidance":
        _strict_keys(
            value,
            {"target", "direction", "threshold", "tolerance"},
            "action_guidance",
        )
        return cls(**value)


@dataclass(frozen=True)
class OutcomeCriterion:
    metric: str
    operator: str
    value: float
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.metric not in OUTCOME_METRICS:
            raise ValueError(
                f"outcome.metric must be one of {sorted(OUTCOME_METRICS)}"
            )
        if self.operator not in OUTCOME_OPERATORS:
            raise ValueError(
                f"outcome.operator must be one of {sorted(OUTCOME_OPERATORS)}"
            )
        object.__setattr__(self, "value", _finite(self.value, "outcome.value"))
        tolerance = _finite(self.tolerance, "outcome.tolerance")
        if tolerance < 0:
            raise ValueError("outcome.tolerance must be non-negative")
        object.__setattr__(self, "tolerance", tolerance)

    def observed_value(self, episode: EpisodeRecord) -> Optional[float]:
        if self.metric == "wealth_change":
            raw = episode.outcome.get("wealth_change")
        else:
            raw = getattr(episode, self.metric, None)
        try:
            return _finite(raw, f"episode.{self.metric}")
        except ValueError:
            return None

    def passes(self, episode: EpisodeRecord) -> bool:
        actual = self.observed_value(episode)
        if actual is None:
            return False
        return _compare(actual, self.operator, self.value, self.tolerance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "operator": self.operator,
            "value": self.value,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OutcomeCriterion":
        _strict_keys(
            value, {"metric", "operator", "value", "tolerance"}, "outcome_criterion"
        )
        return cls(**value)


@dataclass(frozen=True)
class RuleCandidate:
    candidate_id: str
    rule_key: str
    condition: ConditionPredicate
    action_guidance: ActionGuidance
    outcome_criterion: OutcomeCriterion
    rationale: str
    supporting_episode_ids: tuple[str, ...]
    generator_id: str
    raw_response_hash: str
    schema_version: str = CANDIDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CANDIDATE_SCHEMA_VERSION:
            raise ValueError("unsupported candidate schema version")
        _nonempty(self.candidate_id, "candidate_id")
        _nonempty(self.rule_key, "rule_key")
        object.__setattr__(self, "rationale", _nonempty(self.rationale, "rationale"))
        object.__setattr__(
            self,
            "supporting_episode_ids",
            _tuple_of_unique_strings(
                self.supporting_episode_ids,
                "supporting_episode_ids",
                permit_duplicates=True,
            ),
        )
        _nonempty(self.generator_id, "generator_id")
        _nonempty(self.raw_response_hash, "raw_response_hash")
        expected_rule_key = VerifiedSemanticRuleTrack._rule_key(
            self.condition, self.action_guidance, self.outcome_criterion
        )
        if self.rule_key != expected_rule_key:
            raise ValueError("candidate rule_key does not match semantic contents")
        content = {
            "condition": self.condition.to_dict(),
            "action_guidance": self.action_guidance.to_dict(),
            "outcome_criterion": self.outcome_criterion.to_dict(),
            "rationale": self.rationale,
            "supporting_episode_ids": list(self.supporting_episode_ids),
            "generator_id": self.generator_id,
        }
        if self.candidate_id != f"cand-{_hash(content)[:20]}":
            raise ValueError("candidate_id does not match candidate contents")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "rule_key": self.rule_key,
            "condition": self.condition.to_dict(),
            "action_guidance": self.action_guidance.to_dict(),
            "outcome_criterion": self.outcome_criterion.to_dict(),
            "rationale": self.rationale,
            "supporting_episode_ids": list(self.supporting_episode_ids),
            "generator_id": self.generator_id,
            "raw_response_hash": self.raw_response_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuleCandidate":
        _strict_keys(
            value,
            {
                "schema_version",
                "candidate_id",
                "rule_key",
                "condition",
                "action_guidance",
                "outcome_criterion",
                "rationale",
                "supporting_episode_ids",
                "generator_id",
                "raw_response_hash",
            },
            "rule candidate",
        )
        return cls(
            schema_version=value["schema_version"],
            candidate_id=value["candidate_id"],
            rule_key=value["rule_key"],
            condition=ConditionPredicate.from_dict(value["condition"]),
            action_guidance=ActionGuidance.from_dict(value["action_guidance"]),
            outcome_criterion=OutcomeCriterion.from_dict(value["outcome_criterion"]),
            rationale=value["rationale"],
            supporting_episode_ids=tuple(value["supporting_episode_ids"]),
            generator_id=value["generator_id"],
            raw_response_hash=value["raw_response_hash"],
        )


@dataclass(frozen=True)
class VerifiedRule:
    rule_id: str
    rule_key: str
    condition: ConditionPredicate
    action_guidance: ActionGuidance
    outcome_criterion: OutcomeCriterion
    rationale: str
    status: str
    supporting_episode_ids: tuple[str, ...]
    contradicting_episode_ids: tuple[str, ...]
    support_score: int
    contradiction_score: int
    margin: int
    confidence: float
    consecutive_failures: int
    post_proposal_evidence_count: int
    candidate_ids: tuple[str, ...]
    created_at: int
    updated_at: int
    verification_reasons: tuple[str, ...]
    injected: bool = False
    injection_provenance: Optional[Mapping[str, Any]] = None
    schema_version: str = RULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RULE_SCHEMA_VERSION:
            raise ValueError("unsupported verified-rule schema version")
        if self.status not in RULE_STATUSES:
            raise ValueError(f"unknown rule status {self.status!r}")
        for name in (
            "support_score",
            "contradiction_score",
            "consecutive_failures",
            "post_proposal_evidence_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.margin != self.support_score - self.contradiction_score:
            raise ValueError("margin must equal support_score - contradiction_score")
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self,
            "supporting_episode_ids",
            _tuple_of_unique_strings(
                self.supporting_episode_ids, "supporting_episode_ids"
            ),
        )
        object.__setattr__(
            self,
            "contradicting_episode_ids",
            _tuple_of_unique_strings(
                self.contradicting_episode_ids, "contradicting_episode_ids"
            ),
        )
        if set(self.supporting_episode_ids) & set(self.contradicting_episode_ids):
            raise ValueError("supporting and contradicting evidence must be disjoint")
        object.__setattr__(
            self,
            "candidate_ids",
            _tuple_of_unique_strings(self.candidate_ids, "candidate_ids")
            if self.candidate_ids
            else (),
        )
        object.__setattr__(
            self,
            "verification_reasons",
            tuple(str(reason) for reason in self.verification_reasons),
        )
        if self.injection_provenance is not None:
            provenance = json.loads(_canonical_json(dict(self.injection_provenance)))
            object.__setattr__(self, "injection_provenance", provenance)

    def to_prompt_text(self) -> str:
        return (
            f"When {self.condition.field} {self.condition.operator} "
            f"{self.condition.value:g}, {self.action_guidance.direction} "
            f"{self.action_guidance.target} toward "
            f"{self.action_guidance.threshold:g} "
            f"(confidence {self.confidence:.0%}, rule {self.rule_id})."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rule_id": self.rule_id,
            "rule_key": self.rule_key,
            "condition": self.condition.to_dict(),
            "action_guidance": self.action_guidance.to_dict(),
            "outcome_criterion": self.outcome_criterion.to_dict(),
            "rationale": self.rationale,
            "status": self.status,
            "supporting_episode_ids": list(self.supporting_episode_ids),
            "contradicting_episode_ids": list(self.contradicting_episode_ids),
            "support_score": self.support_score,
            "contradiction_score": self.contradiction_score,
            "margin": self.margin,
            "confidence": self.confidence,
            "consecutive_failures": self.consecutive_failures,
            "post_proposal_evidence_count": self.post_proposal_evidence_count,
            "candidate_ids": list(self.candidate_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "verification_reasons": list(self.verification_reasons),
            "injected": self.injected,
            "injection_provenance": self.injection_provenance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerifiedRule":
        return cls(
            **{
                **dict(value),
                "condition": ConditionPredicate.from_dict(value["condition"]),
                "action_guidance": ActionGuidance.from_dict(
                    value["action_guidance"]
                ),
                "outcome_criterion": OutcomeCriterion.from_dict(
                    value["outcome_criterion"]
                ),
                "supporting_episode_ids": tuple(value["supporting_episode_ids"]),
                "contradicting_episode_ids": tuple(
                    value["contradicting_episode_ids"]
                ),
                "candidate_ids": tuple(value["candidate_ids"]),
                "verification_reasons": tuple(value["verification_reasons"]),
            }
        )


@dataclass(frozen=True)
class RuleEvent:
    event_id: str
    timestamp: int
    event_type: str
    rule_id: Optional[str]
    candidate_id: Optional[str]
    from_status: Optional[str]
    to_status: Optional[str]
    episode_ids: tuple[str, ...]
    reason: str
    metrics: Mapping[str, Any]
    provenance: Mapping[str, Any]
    schema_version: str = EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError("unsupported rule-event schema version")
        _nonempty(self.event_id, "event_id")
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, int):
            raise ValueError("event timestamp must be an integer")
        _nonempty(self.event_type, "event_type")
        for name, status in (
            ("from_status", self.from_status),
            ("to_status", self.to_status),
        ):
            if status is not None and status not in RULE_STATUSES:
                raise ValueError(f"event {name} has unknown status {status!r}")
        object.__setattr__(
            self,
            "episode_ids",
            _tuple_of_unique_strings(self.episode_ids, "event.episode_ids")
            if self.episode_ids
            else (),
        )
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(
            self, "metrics", json.loads(_canonical_json(dict(self.metrics)))
        )
        object.__setattr__(
            self, "provenance", json.loads(_canonical_json(dict(self.provenance)))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "rule_id": self.rule_id,
            "candidate_id": self.candidate_id,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "episode_ids": list(self.episode_ids),
            "reason": self.reason,
            "metrics": dict(self.metrics),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuleEvent":
        values = dict(value)
        values["episode_ids"] = tuple(values["episode_ids"])
        return cls(**values)


class VerifiedSemanticRuleTrack:
    """Verified semantic-rule lifecycle backed by finalized M2 evidence."""

    def __init__(
        self,
        episodic_track: EvidenceLinkedEpisodicTrack,
        *,
        allowed_condition_fields: Sequence[str] = DEFAULT_CONDITION_FIELDS,
        allowed_action_targets: Sequence[str] = DEFAULT_ACTION_TARGETS,
        min_candidate_support: int = 2,
        activation_min_support: int = 3,
        activation_min_margin: int = 1,
        activation_confidence_threshold: float = 0.60,
        proposal_confidence_floor: float = 0.50,
        retirement_patience: int = 2,
        retirement_confidence_threshold: float = 0.45,
    ) -> None:
        self.episodic_track = episodic_track
        self.allowed_condition_fields = _tuple_of_unique_strings(
            allowed_condition_fields, "allowed_condition_fields"
        )
        self.allowed_action_targets = _tuple_of_unique_strings(
            allowed_action_targets, "allowed_action_targets"
        )
        for name, value, minimum in (
            ("min_candidate_support", min_candidate_support, 1),
            ("activation_min_support", activation_min_support, 1),
            ("retirement_patience", retirement_patience, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if activation_min_support < min_candidate_support:
            raise ValueError(
                "activation_min_support cannot be below min_candidate_support"
            )
        if isinstance(activation_min_margin, bool) or not isinstance(
            activation_min_margin, int
        ):
            raise ValueError("activation_min_margin must be an integer")
        for name, value in (
            ("activation_confidence_threshold", activation_confidence_threshold),
            ("proposal_confidence_floor", proposal_confidence_floor),
            ("retirement_confidence_threshold", retirement_confidence_threshold),
        ):
            numeric = _finite(value, name)
            if not 0.0 <= numeric <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            setattr(self, name, numeric)
        self.min_candidate_support = min_candidate_support
        self.activation_min_support = activation_min_support
        self.activation_min_margin = activation_min_margin
        self.retirement_patience = retirement_patience
        self._candidates: dict[str, RuleCandidate] = {}
        self._rules: dict[str, VerifiedRule] = {}
        self._events: list[RuleEvent] = []

    @staticmethod
    def _confidence(support: int, contradiction: int) -> float:
        return (support + 1.0) / (support + contradiction + 2.0)

    @staticmethod
    def _rule_key(
        condition: ConditionPredicate,
        guidance: ActionGuidance,
        criterion: OutcomeCriterion,
    ) -> str:
        payload = {
            "condition": condition.to_dict(),
            "action_guidance": guidance.to_dict(),
            "outcome_criterion": criterion.to_dict(),
        }
        return f"rule-{_hash(payload)[:20]}"

    @property
    def rules(self) -> tuple[VerifiedRule, ...]:
        return tuple(self._rules.values())

    @property
    def events(self) -> tuple[RuleEvent, ...]:
        return tuple(self._events)

    def get(self, rule_id: str) -> Optional[VerifiedRule]:
        return self._rules.get(rule_id)

    def _append_event(
        self,
        *,
        timestamp: int,
        event_type: str,
        rule_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        from_status: Optional[str] = None,
        to_status: Optional[str] = None,
        episode_ids: Sequence[str] = (),
        reason: str = "",
        metrics: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> RuleEvent:
        core = {
            "index": len(self._events),
            "timestamp": int(timestamp),
            "event_type": event_type,
            "rule_id": rule_id,
            "candidate_id": candidate_id,
            "from_status": from_status,
            "to_status": to_status,
            "episode_ids": list(episode_ids),
            "reason": reason,
            "metrics": dict(metrics or {}),
            "provenance": dict(provenance or {}),
        }
        event = RuleEvent(
            event_id=f"rle-{len(self._events):06d}-{_hash(core)[:12]}",
            timestamp=int(timestamp),
            event_type=event_type,
            rule_id=rule_id,
            candidate_id=candidate_id,
            from_status=from_status,
            to_status=to_status,
            episode_ids=tuple(episode_ids),
            reason=reason,
            metrics=dict(metrics or {}),
            provenance=dict(provenance or {}),
        )
        self._events.append(event)
        return event

    def _metrics(self, rule: VerifiedRule) -> dict[str, Any]:
        return {
            "support_score": rule.support_score,
            "contradiction_score": rule.contradiction_score,
            "margin": rule.margin,
            "confidence": rule.confidence,
            "consecutive_failures": rule.consecutive_failures,
            "post_proposal_evidence_count": rule.post_proposal_evidence_count,
        }

    def build_proposal_prompt(
        self,
        episode_ids: Optional[Sequence[str]] = None,
        *,
        max_episodes: int = 6,
        observed_through: Optional[int] = None,
    ) -> str:
        if max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        if episode_ids is None:
            episodes = [
                episode
                for episode in self.episodic_track.finalized_episodes
                if observed_through is None or episode.outcome_t <= observed_through
            ][-max_episodes:]
        else:
            episodes = []
            for episode_id in episode_ids:
                episode = self.episodic_track.get(episode_id)
                if episode is None:
                    raise KeyError(f"unknown episode ID in proposal prompt: {episode_id}")
                if (
                    observed_through is not None
                    and episode.outcome_t > observed_through
                ):
                    raise ValueError(
                        f"episode {episode_id} is not finalized by "
                        f"observed_through={observed_through}"
                    )
                episodes.append(episode)
            episodes = episodes[-max_episodes:]

        evidence = []
        for episode in episodes:
            evidence.append(
                {
                    "episode_id": episode.episode_id,
                    "pre_state": {
                        field: episode.pre_state[field]
                        for field in self.allowed_condition_fields
                        if field in episode.pre_state
                    },
                    "executed_action": {
                        target: episode.executed_action[target]
                        for target in self.allowed_action_targets
                        if target in episode.executed_action
                    },
                    "outcome": {
                        "utility_advantage": episode.utility_advantage,
                        "flow_utility": episode.flow_utility,
                        "reward": episode.reward,
                        "wealth_change": episode.outcome.get("wealth_change"),
                    },
                }
            )
        schema = {
            "condition": {
                "field": f"one of {list(self.allowed_condition_fields)}",
                "operator": f"one of {sorted(CONDITION_OPERATORS)}",
                "value": "finite number",
                "tolerance": "non-negative number",
            },
            "action_guidance": {
                "target": f"one of {list(self.allowed_action_targets)}",
                "direction": f"one of {sorted(ACTION_DIRECTIONS)}",
                "threshold": "finite number",
                "tolerance": "non-negative number",
            },
            "outcome_criterion": {
                "metric": f"one of {sorted(OUTCOME_METRICS)}",
                "operator": f"one of {sorted(OUTCOME_OPERATORS)}",
                "value": "finite number",
                "tolerance": "non-negative number",
            },
            "rationale": "brief evidence-grounded explanation",
            "supporting_episode_ids": "at least two IDs copied exactly from evidence",
        }
        return (
            "Propose one semantic decision rule using only the finalized evidence below. "
            "Return exactly one JSON object and no additional keys. Do not invent or "
            "duplicate episode IDs. A proposal is never activated directly; it is "
            "verified and must receive later distinct evidence. Every claimed support "
            "episode must independently satisfy all three checks: (1) the condition is "
            "true in pre_state, (2) the executed action satisfies the absolute guidance "
            "threshold, and (3) the outcome passes the criterion. The action directions "
            "are absolute verifier predicates: increase means executed value >= threshold, "
            "decrease means executed value <= threshold, and maintain means within "
            "tolerance of threshold. Do not describe a comparison between two episodes "
            "unless both episodes themselves pass the rule. Before returning JSON, check "
            "each copied support ID against those three predicates.\n"
            f"Allowed JSON schema:\n{json.dumps(schema, indent=2, sort_keys=True)}\n"
            f"Allowed episode IDs: {[item['episode_id'] for item in evidence]}\n"
            f"Evidence:\n{json.dumps(evidence, indent=2, sort_keys=True)}"
        )

    @staticmethod
    def _extract_json_object(raw_response: str) -> Mapping[str, Any]:
        if not isinstance(raw_response, str) or not raw_response.strip():
            raise CandidateParseError("candidate response is empty")
        text = raw_response.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise CandidateParseError("candidate response contains no JSON object")
            try:
                value = json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise CandidateParseError(f"malformed candidate JSON: {exc.msg}") from exc
        if not isinstance(value, Mapping):
            raise CandidateParseError("candidate JSON root must be an object")
        return value

    def parse_candidate(
        self, raw_response: str, *, generator_id: str = "llm-candidate-generator"
    ) -> RuleCandidate:
        try:
            value = self._extract_json_object(raw_response)
            _strict_keys(
                value,
                {
                    "condition",
                    "action_guidance",
                    "outcome_criterion",
                    "rationale",
                    "supporting_episode_ids",
                },
                "candidate JSON",
            )
            condition = ConditionPredicate.from_dict(value["condition"])
            guidance = ActionGuidance.from_dict(value["action_guidance"])
            criterion = OutcomeCriterion.from_dict(value["outcome_criterion"])
            if condition.field not in self.allowed_condition_fields:
                raise ValueError(f"unsupported condition field {condition.field!r}")
            if guidance.target not in self.allowed_action_targets:
                raise ValueError(f"unsupported action target {guidance.target!r}")
            rationale = _nonempty(value["rationale"], "rationale")
            support_ids = _tuple_of_unique_strings(
                value["supporting_episode_ids"],
                "supporting_episode_ids",
                permit_duplicates=True,
            )
        except (TypeError, ValueError, KeyError) as exc:
            if isinstance(exc, CandidateParseError):
                raise
            raise CandidateParseError(str(exc)) from exc

        content = {
            "condition": condition.to_dict(),
            "action_guidance": guidance.to_dict(),
            "outcome_criterion": criterion.to_dict(),
            "rationale": rationale,
            "supporting_episode_ids": list(support_ids),
            "generator_id": generator_id,
        }
        rule_key = self._rule_key(condition, guidance, criterion)
        return RuleCandidate(
            candidate_id=f"cand-{_hash(content)[:20]}",
            rule_key=rule_key,
            condition=condition,
            action_guidance=guidance,
            outcome_criterion=criterion,
            rationale=rationale,
            supporting_episode_ids=support_ids,
            generator_id=_nonempty(generator_id, "generator_id"),
            raw_response_hash=_hash({"raw_response": raw_response}),
        )

    @staticmethod
    def _classification(
        candidate_or_rule: RuleCandidate | VerifiedRule, episode: EpisodeRecord
    ) -> str:
        if not candidate_or_rule.condition.matches(episode.pre_state):
            return "irrelevant"
        if not candidate_or_rule.action_guidance.is_consistent(
            episode.executed_action
        ):
            return "contradiction"
        if not candidate_or_rule.outcome_criterion.passes(episode):
            return "contradiction"
        return "support"

    def propose(
        self,
        raw_response: str,
        *,
        current_t: int,
        generator_id: str = "llm-candidate-generator",
    ) -> VerifiedRule:
        try:
            candidate = self.parse_candidate(
                raw_response, generator_id=generator_id
            )
        except CandidateParseError as exc:
            self._append_event(
                timestamp=current_t,
                event_type="candidate_parse_rejected",
                reason=str(exc),
                provenance={"raw_response_hash": _hash({"raw_response": raw_response})},
            )
            raise
        return self.submit_candidate(candidate, current_t=current_t)

    def submit_candidate(
        self, candidate: RuleCandidate, *, current_t: int
    ) -> VerifiedRule:
        self._candidates[candidate.candidate_id] = candidate
        rule_id = candidate.rule_key
        if rule_id in self._rules:
            existing = self._rules[rule_id]
            self._append_event(
                timestamp=current_t,
                event_type="duplicate_semantic_candidate_ignored",
                rule_id=rule_id,
                candidate_id=candidate.candidate_id,
                from_status=existing.status,
                to_status=existing.status,
                reason="an equivalent condition/action/outcome rule already exists",
                metrics=self._metrics(existing),
            )
            return existing

        requested_ids = candidate.supporting_episode_ids
        unique_requested = tuple(dict.fromkeys(requested_ids))
        reasons: list[str] = []
        if len(unique_requested) != len(requested_ids):
            reasons.append("candidate contains duplicate supporting episode IDs")
        if len(unique_requested) < self.min_candidate_support:
            reasons.append(
                f"candidate requires at least {self.min_candidate_support} unique support IDs"
            )

        valid_support: list[str] = []
        contradictions: list[str] = []
        existing_requested: list[str] = []
        for episode_id in unique_requested:
            episode = self.episodic_track.get(episode_id)
            if episode is None:
                reasons.append(f"supporting episode does not exist: {episode_id}")
                continue
            if episode.outcome_t > current_t:
                reasons.append(
                    f"supporting episode is not observable at current_t={current_t}: "
                    f"{episode_id}"
                )
                continue
            existing_requested.append(episode_id)
            classification = self._classification(candidate, episode)
            if classification == "support":
                valid_support.append(episode_id)
            elif classification == "irrelevant":
                reasons.append(
                    f"condition does not hold in claimed support episode: {episode_id}"
                )
            else:
                contradictions.append(episode_id)
                if not candidate.action_guidance.is_consistent(
                    episode.executed_action
                ):
                    reasons.append(
                        f"action guidance is inconsistent with claimed support: {episode_id}"
                    )
                else:
                    reasons.append(
                        f"outcome criterion fails in claimed support: {episode_id}"
                    )

        # Search all other condition-matched episodes for counterexamples.  Positive
        # unlisted episodes are not silently promoted to candidate support.
        for episode in self.episodic_track.finalized_episodes:
            if episode.outcome_t > current_t:
                continue
            if episode.episode_id in unique_requested:
                continue
            if self._classification(candidate, episode) == "contradiction":
                contradictions.append(episode.episode_id)
        contradictions = list(dict.fromkeys(contradictions))
        if len(valid_support) < self.min_candidate_support:
            reasons.append(
                f"only {len(valid_support)} claimed episodes pass all verification checks"
            )

        support_score = len(valid_support)
        contradiction_score = len(contradictions)
        confidence = self._confidence(support_score, contradiction_score)
        margin = support_score - contradiction_score
        if confidence < self.proposal_confidence_floor:
            reasons.append(
                "candidate confidence is below the provisional admission floor"
            )
        rejected = bool(reasons) or confidence < self.proposal_confidence_floor
        status = "rejected" if rejected else "provisional"
        rule = VerifiedRule(
            rule_id=rule_id,
            rule_key=candidate.rule_key,
            condition=candidate.condition,
            action_guidance=candidate.action_guidance,
            outcome_criterion=candidate.outcome_criterion,
            rationale=candidate.rationale,
            status=status,
            supporting_episode_ids=tuple(valid_support),
            contradicting_episode_ids=tuple(contradictions),
            support_score=support_score,
            contradiction_score=contradiction_score,
            margin=margin,
            confidence=confidence,
            consecutive_failures=0,
            post_proposal_evidence_count=0,
            candidate_ids=(candidate.candidate_id,),
            created_at=int(current_t),
            updated_at=int(current_t),
            verification_reasons=tuple(reasons),
        )
        self._rules[rule.rule_id] = rule
        self._append_event(
            timestamp=current_t,
            event_type=("candidate_rejected" if rejected else "candidate_verified"),
            rule_id=rule.rule_id,
            candidate_id=candidate.candidate_id,
            from_status=None,
            to_status=status,
            episode_ids=tuple(existing_requested),
            reason="; ".join(reasons) if reasons else "all candidate checks passed",
            metrics=self._metrics(rule),
            provenance={
                "generator_id": candidate.generator_id,
                "raw_response_hash": candidate.raw_response_hash,
                "searched_counterevidence": True,
                "requested_support_ids": list(requested_ids),
            },
        )
        return rule

    def observe_episode(
        self, rule_id: str, episode_id: str, *, current_t: int
    ) -> VerifiedRule:
        rule = self._rules.get(rule_id)
        if rule is None:
            raise KeyError(f"unknown rule: {rule_id}")
        episode = self.episodic_track.get(episode_id)
        if episode is None:
            raise KeyError(f"unknown finalized episode: {episode_id}")
        if episode.outcome_t > current_t:
            raise ValueError(
                f"episode {episode_id} is not observable at current_t={current_t}"
            )
        if rule.status in {"rejected", "retired"}:
            self._append_event(
                timestamp=current_t,
                event_type="evidence_ignored_terminal_rule",
                rule_id=rule_id,
                from_status=rule.status,
                to_status=rule.status,
                episode_ids=(episode_id,),
                reason="terminal rules do not accept new evidence",
                metrics=self._metrics(rule),
            )
            return rule
        if episode_id in rule.supporting_episode_ids or episode_id in rule.contradicting_episode_ids:
            self._append_event(
                timestamp=current_t,
                event_type="duplicate_evidence_ignored",
                rule_id=rule_id,
                from_status=rule.status,
                to_status=rule.status,
                episode_ids=(episode_id,),
                reason="evidence ID was already counted",
                metrics=self._metrics(rule),
            )
            return rule

        # "Observed after proposal" is an evidence-time property, not merely the
        # time at which a caller invokes this method. Otherwise an older, unlisted
        # episode can be replayed immediately after proposal and falsely satisfy
        # the activation delay.
        if episode.outcome_t <= rule.created_at:
            raise ValueError(
                f"episode {episode_id} is not post-proposal evidence for "
                f"rule created_at={rule.created_at}"
            )

        classification = self._classification(rule, episode)
        if classification == "irrelevant":
            self._append_event(
                timestamp=current_t,
                event_type="irrelevant_evidence_ignored",
                rule_id=rule_id,
                from_status=rule.status,
                to_status=rule.status,
                episode_ids=(episode_id,),
                reason="rule condition does not hold",
                metrics=self._metrics(rule),
            )
            return rule

        supports = list(rule.supporting_episode_ids)
        contradictions = list(rule.contradicting_episode_ids)
        if classification == "support":
            supports.append(episode_id)
            failures = 0
        else:
            contradictions.append(episode_id)
            failures = rule.consecutive_failures + 1
        support_score = len(supports)
        contradiction_score = len(contradictions)
        updated = replace(
            rule,
            supporting_episode_ids=tuple(supports),
            contradicting_episode_ids=tuple(contradictions),
            support_score=support_score,
            contradiction_score=contradiction_score,
            margin=support_score - contradiction_score,
            confidence=self._confidence(support_score, contradiction_score),
            consecutive_failures=failures,
            post_proposal_evidence_count=rule.post_proposal_evidence_count + 1,
            updated_at=int(current_t),
        )
        self._rules[rule_id] = updated
        self._append_event(
            timestamp=current_t,
            event_type=("support_evidence_added" if classification == "support" else "contradiction_evidence_added"),
            rule_id=rule_id,
            from_status=rule.status,
            to_status=rule.status,
            episode_ids=(episode_id,),
            reason=(
                "condition, action guidance, and outcome criterion all pass"
                if classification == "support"
                else "condition matched but action guidance or outcome criterion failed"
            ),
            metrics=self._metrics(updated),
        )

        should_retire = (
            updated.consecutive_failures >= self.retirement_patience
            or updated.confidence < self.retirement_confidence_threshold
        )
        should_activate = (
            updated.status == "provisional"
            and updated.post_proposal_evidence_count >= 1
            and updated.support_score >= self.activation_min_support
            and updated.margin >= self.activation_min_margin
            and updated.confidence >= self.activation_confidence_threshold
        )
        if should_retire:
            retired = replace(updated, status="retired", updated_at=int(current_t))
            self._rules[rule_id] = retired
            self._append_event(
                timestamp=current_t,
                event_type="rule_retired",
                rule_id=rule_id,
                from_status=updated.status,
                to_status="retired",
                episode_ids=(episode_id,),
                reason=(
                    "retirement patience exhausted"
                    if updated.consecutive_failures >= self.retirement_patience
                    else "confidence fell below retirement threshold"
                ),
                metrics=self._metrics(retired),
            )
            return retired
        if should_activate:
            active = replace(updated, status="active", updated_at=int(current_t))
            self._rules[rule_id] = active
            self._append_event(
                timestamp=current_t,
                event_type="rule_activated",
                rule_id=rule_id,
                from_status="provisional",
                to_status="active",
                episode_ids=(episode_id,),
                reason="distinct post-proposal evidence passed activation thresholds",
                metrics=self._metrics(active),
            )
            return active
        return updated

    def inject_active_rule(
        self,
        *,
        condition: ConditionPredicate,
        action_guidance: ActionGuidance,
        outcome_criterion: OutcomeCriterion,
        rationale: str,
        current_t: int,
        injection_id: str,
        provenance: Mapping[str, Any],
        initial_confidence: float = 1.0,
    ) -> VerifiedRule:
        injection_id = _nonempty(injection_id, "injection_id")
        provenance_copy = json.loads(_canonical_json(dict(provenance)))
        if not provenance_copy:
            raise ValueError("injection provenance must not be empty")
        confidence = _finite(initial_confidence, "initial_confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("initial_confidence must be in [0, 1]")
        if condition.field not in self.allowed_condition_fields:
            raise ValueError(f"unsupported injected condition field {condition.field!r}")
        if action_guidance.target not in self.allowed_action_targets:
            raise ValueError(
                f"unsupported injected action target {action_guidance.target!r}"
            )
        semantic_key = self._rule_key(condition, action_guidance, outcome_criterion)
        rule_id = f"{semantic_key}:injected:{injection_id}"
        if rule_id in self._rules:
            raise ValueError(f"duplicate injected rule ID: {rule_id}")
        rule = VerifiedRule(
            rule_id=rule_id,
            rule_key=semantic_key,
            condition=condition,
            action_guidance=action_guidance,
            outcome_criterion=outcome_criterion,
            rationale=_nonempty(rationale, "rationale"),
            status="active",
            supporting_episode_ids=(),
            contradicting_episode_ids=(),
            support_score=0,
            contradiction_score=0,
            margin=0,
            confidence=confidence,
            consecutive_failures=0,
            post_proposal_evidence_count=0,
            candidate_ids=(),
            created_at=int(current_t),
            updated_at=int(current_t),
            verification_reasons=("experimental injection bypass",),
            injected=True,
            injection_provenance={
                **provenance_copy,
                "injection_id": injection_id,
            },
        )
        self._rules[rule_id] = rule
        self._append_event(
            timestamp=current_t,
            event_type="experimental_rule_injected_active",
            rule_id=rule_id,
            from_status=None,
            to_status="active",
            reason="explicit experimental verifier bypass",
            metrics=self._metrics(rule),
            provenance=rule.injection_provenance,
        )
        return rule

    def retrieve(
        self,
        current_state: Mapping[str, Any],
        *,
        current_t: int,
        limit: int = 3,
        log_selection: bool = True,
    ) -> tuple[VerifiedRule, ...]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        relevant = [
            rule
            for rule in self._rules.values()
            if rule.status == "active" and rule.condition.matches(current_state)
        ]
        relevant.sort(
            key=lambda rule: (
                -rule.confidence,
                -rule.margin,
                -rule.updated_at,
                rule.rule_id,
            )
        )
        selected = tuple(relevant[:limit])
        if log_selection:
            for rule in selected:
                self._append_event(
                    timestamp=current_t,
                    event_type="active_rule_retrieved",
                    rule_id=rule.rule_id,
                    from_status="active",
                    to_status="active",
                    reason="active rule condition matched current state",
                    metrics=self._metrics(rule),
                )
        return selected

    def validate_referential_integrity(self) -> None:
        episodes_by_id = {
            episode.episode_id: episode
            for episode in self.episodic_track.finalized_episodes
        }
        known_episode_ids = set(episodes_by_id)
        event_ids: set[str] = set()
        for rule in self._rules.values():
            expected_rule_key = self._rule_key(
                rule.condition, rule.action_guidance, rule.outcome_criterion
            )
            if rule.rule_key != expected_rule_key:
                raise ValueError(f"rule {rule.rule_id} has inconsistent rule_key")
            if not rule.injected and rule.rule_id != rule.rule_key:
                raise ValueError(f"rule {rule.rule_id} has inconsistent rule_id")
            evidence = set(rule.supporting_episode_ids) | set(
                rule.contradicting_episode_ids
            )
            missing = sorted(evidence - known_episode_ids)
            if missing:
                raise ValueError(
                    f"rule {rule.rule_id} references missing episode IDs: {missing}"
                )
            if rule.support_score != len(rule.supporting_episode_ids):
                raise ValueError(f"rule {rule.rule_id} has inconsistent support_score")
            if rule.contradiction_score != len(rule.contradicting_episode_ids):
                raise ValueError(
                    f"rule {rule.rule_id} has inconsistent contradiction_score"
                )
            if rule.updated_at < rule.created_at:
                raise ValueError(f"rule {rule.rule_id} was updated before creation")
            evidence_episodes = [episodes_by_id[episode_id] for episode_id in evidence]
            observed_after_creation = [
                episode
                for episode in evidence_episodes
                if episode.outcome_t > rule.created_at
            ]
            if len(observed_after_creation) != rule.post_proposal_evidence_count:
                raise ValueError(
                    f"rule {rule.rule_id} has inconsistent post-proposal evidence count"
                )
            not_yet_observable = [
                episode.episode_id
                for episode in evidence_episodes
                if episode.outcome_t > rule.updated_at
            ]
            if not_yet_observable:
                raise ValueError(
                    f"rule {rule.rule_id} contains evidence newer than updated_at: "
                    f"{sorted(not_yet_observable)}"
                )
            if not rule.injected or rule.post_proposal_evidence_count > 0:
                expected_confidence = self._confidence(
                    rule.support_score, rule.contradiction_score
                )
                if not math.isclose(rule.confidence, expected_confidence, abs_tol=1e-12):
                    raise ValueError(f"rule {rule.rule_id} has inconsistent confidence")
            missing_candidates = set(rule.candidate_ids) - set(self._candidates)
            if missing_candidates:
                raise ValueError(
                    f"rule {rule.rule_id} references missing candidates: "
                    f"{sorted(missing_candidates)}"
                )
            if rule.status == "active" and not rule.injected:
                if (
                    rule.post_proposal_evidence_count < 1
                    or rule.support_score < self.activation_min_support
                    or rule.margin < self.activation_min_margin
                    or rule.confidence < self.activation_confidence_threshold
                ):
                    raise ValueError(
                        f"active rule {rule.rule_id} does not satisfy activation invariants"
                    )

        for event_index, event in enumerate(self._events):
            if event.event_id in event_ids:
                raise ValueError(f"duplicate rule-event ID: {event.event_id}")
            event_ids.add(event.event_id)
            if event.rule_id is not None and event.rule_id not in self._rules:
                raise ValueError(f"event references missing rule: {event.rule_id}")
            if (
                event.candidate_id is not None
                and event.candidate_id not in self._candidates
            ):
                raise ValueError(
                    f"event references missing candidate: {event.candidate_id}"
                )
            missing = set(event.episode_ids) - known_episode_ids
            if missing:
                raise ValueError(
                    f"event {event.event_id} references missing evidence: {sorted(missing)}"
                )
            core = {
                "index": event_index,
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "rule_id": event.rule_id,
                "candidate_id": event.candidate_id,
                "from_status": event.from_status,
                "to_status": event.to_status,
                "episode_ids": list(event.episode_ids),
                "reason": event.reason,
                "metrics": dict(event.metrics),
                "provenance": dict(event.provenance),
            }
            expected_event_id = f"rle-{event_index:06d}-{_hash(core)[:12]}"
            if event.event_id != expected_event_id:
                raise ValueError(f"event ledger hash mismatch: {event.event_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "config": {
                "allowed_condition_fields": list(self.allowed_condition_fields),
                "allowed_action_targets": list(self.allowed_action_targets),
                "min_candidate_support": self.min_candidate_support,
                "activation_min_support": self.activation_min_support,
                "activation_min_margin": self.activation_min_margin,
                "activation_confidence_threshold": self.activation_confidence_threshold,
                "proposal_confidence_floor": self.proposal_confidence_floor,
                "retirement_patience": self.retirement_patience,
                "retirement_confidence_threshold": self.retirement_confidence_threshold,
            },
            "candidates": [candidate.to_dict() for candidate in self._candidates.values()],
            "rules": [rule.to_dict() for rule in self._rules.values()],
            "events": [event.to_dict() for event in self._events],
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        episodic_track: EvidenceLinkedEpisodicTrack,
    ) -> "VerifiedSemanticRuleTrack":
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported M3 schema version")
        config = dict(value["config"])
        track = cls(episodic_track, **config)
        for candidate_value in value.get("candidates", []):
            candidate = RuleCandidate.from_dict(candidate_value)
            if candidate.candidate_id in track._candidates:
                raise ValueError(f"duplicate candidate ID: {candidate.candidate_id}")
            track._candidates[candidate.candidate_id] = candidate
        for rule_value in value.get("rules", []):
            rule = VerifiedRule.from_dict(rule_value)
            if rule.rule_id in track._rules:
                raise ValueError(f"duplicate rule ID: {rule.rule_id}")
            track._rules[rule.rule_id] = rule
        track._events = [
            RuleEvent.from_dict(event_value) for event_value in value.get("events", [])
        ]
        track.validate_referential_integrity()
        return track


# Descriptive alias for callers that name the module rather than the track.
M3VerifiedSemanticMemory = VerifiedSemanticRuleTrack
