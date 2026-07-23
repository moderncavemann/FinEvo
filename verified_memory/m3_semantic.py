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


SCHEMA_VERSION = "m3-verified-semantic-v2"
CANDIDATE_SCHEMA_VERSION = "m3-rule-candidate-v2"
RULE_SCHEMA_VERSION = "m3-verified-rule-v2"
EVENT_SCHEMA_VERSION = "m3-rule-event-v2"

RULE_STATUSES = frozenset({"provisional", "active", "retired", "rejected"})
CONDITION_OPERATORS = frozenset({">", ">=", "<", "<=", "=="})
ACTION_DIRECTIONS = frozenset({"at_least", "at_most", "approximately"})
LEGACY_ACTION_DIRECTIONS = {
    "increase": "at_least",
    "decrease": "at_most",
    "maintain": "approximately",
}
EVIDENCE_TYPES = frozenset(
    {
        "support",
        "harmful_compliance",
        "alternative_success",
        "alternative_failure",
        "irrelevant",
    }
)
DEFAULT_EVIDENCE_WEIGHTS = {
    # One compliant success contributes one full unit of positive evidence.
    "support": 1.0,
    # A bad outcome after compliance is the strongest negative evidence.
    "harmful_compliance": 1.0,
    # A successful off-policy action is competitive but weaker evidence.
    "alternative_success": 0.5,
    # Neither off-policy failure nor out-of-scope data identifies rule quality.
    "alternative_failure": 0.0,
    "irrelevant": 0.0,
}
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


def _timestamp(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


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
class ContextScope:
    """Auditable state/context boundary within which a rule is applicable."""

    scope_id: str
    predicates: tuple[ConditionPredicate, ...] = ()

    def __post_init__(self) -> None:
        scope_id = _nonempty(self.scope_id, "context_scope.scope_id")
        object.__setattr__(self, "scope_id", scope_id)
        predicates = tuple(self.predicates)
        if any(not isinstance(item, ConditionPredicate) for item in predicates):
            raise ValueError(
                "context_scope.predicates must contain ConditionPredicate values"
            )
        if len(predicates) != len(set(predicates)):
            raise ValueError("context_scope.predicates must not contain duplicates")
        if scope_id == "global" and predicates:
            raise ValueError("global context scope must not contain predicates")
        if scope_id != "global" and not predicates:
            raise ValueError(
                "a non-global context scope requires at least one predicate"
            )
        object.__setattr__(self, "predicates", predicates)

    @classmethod
    def global_scope(cls) -> "ContextScope":
        return cls(scope_id="global", predicates=())

    def matches(self, state: Mapping[str, Any]) -> bool:
        return all(predicate.matches(state) for predicate in self.predicates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "predicates": [predicate.to_dict() for predicate in self.predicates],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContextScope":
        _strict_keys(value, {"scope_id", "predicates"}, "context_scope")
        predicates = value["predicates"]
        if isinstance(predicates, (str, bytes)) or not isinstance(predicates, Sequence):
            raise ValueError("context_scope.predicates must be a list")
        return cls(
            scope_id=value["scope_id"],
            predicates=tuple(ConditionPredicate.from_dict(item) for item in predicates),
        )


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
        if self.direction == "at_least":
            return actual >= self.threshold - self.tolerance
        if self.direction == "at_most":
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
        if value.get("direction") in LEGACY_ACTION_DIRECTIONS:
            raise ValueError(
                "legacy action direction is not accepted by canonical parsing; "
                "use parse_candidate() for an explicitly recorded migration"
            )
        return cls(**value)

    @classmethod
    def from_candidate_dict(
        cls, value: Mapping[str, Any]
    ) -> tuple["ActionGuidance", Optional[str]]:
        """Parse candidate guidance and explicitly record legacy migration.

        Version-1 prompts used relative-sounding names for absolute predicates.
        Candidate ingestion may migrate those values, but persisted v2 records and
        direct construction remain fail-closed so the semantic change is never
        silent.
        """

        _strict_keys(
            value,
            {"target", "direction", "threshold", "tolerance"},
            "action_guidance",
        )
        raw_direction = value.get("direction")
        if raw_direction not in LEGACY_ACTION_DIRECTIONS:
            return cls(**value), None
        canonical = LEGACY_ACTION_DIRECTIONS[str(raw_direction)]
        migrated = {**dict(value), "direction": canonical}
        note = (
            f"legacy action.direction={raw_direction!r} explicitly migrated to "
            f"the absolute predicate {canonical!r}"
        )
        return cls(**migrated), note


@dataclass(frozen=True)
class OutcomeCriterion:
    metric: str
    operator: str
    value: float
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.metric not in OUTCOME_METRICS:
            raise ValueError(f"outcome.metric must be one of {sorted(OUTCOME_METRICS)}")
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


DEFAULT_REGISTERED_OUTCOME_CRITERION = OutcomeCriterion(
    metric="utility_advantage",
    operator=">",
    value=0.0,
    tolerance=0.0,
)


@dataclass(frozen=True)
class RuleCandidate:
    candidate_id: str
    rule_key: str
    rule_family_id: str
    context_scope: ContextScope
    condition: ConditionPredicate
    action_guidance: ActionGuidance
    outcome_criterion: OutcomeCriterion
    rationale: str
    supporting_episode_ids: tuple[str, ...]
    generator_id: str
    raw_response_hash: str
    migration_notes: tuple[str, ...] = ()
    schema_version: str = CANDIDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CANDIDATE_SCHEMA_VERSION:
            raise ValueError("unsupported candidate schema version")
        _nonempty(self.candidate_id, "candidate_id")
        _nonempty(self.rule_key, "rule_key")
        _nonempty(self.rule_family_id, "rule_family_id")
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
        object.__setattr__(
            self,
            "migration_notes",
            tuple(_nonempty(note, "migration note") for note in self.migration_notes),
        )
        expected_rule_key = VerifiedSemanticRuleTrack._rule_key(
            self.context_scope,
            self.condition,
            self.action_guidance,
            self.outcome_criterion,
        )
        if self.rule_key != expected_rule_key:
            raise ValueError("candidate rule_key does not match semantic contents")
        expected_family_id = VerifiedSemanticRuleTrack._rule_family_id(
            self.context_scope,
            self.condition,
            self.action_guidance,
            self.outcome_criterion,
        )
        if self.rule_family_id != expected_family_id:
            raise ValueError(
                "candidate rule_family_id does not match semantic contents"
            )
        content = {
            "context_scope": self.context_scope.to_dict(),
            "condition": self.condition.to_dict(),
            "action_guidance": self.action_guidance.to_dict(),
            "outcome_criterion": self.outcome_criterion.to_dict(),
            "rationale": self.rationale,
            "supporting_episode_ids": list(self.supporting_episode_ids),
            "generator_id": self.generator_id,
            "raw_response_hash": self.raw_response_hash,
            "migration_notes": list(self.migration_notes),
        }
        if self.candidate_id != f"cand-{_hash(content)[:20]}":
            raise ValueError("candidate_id does not match candidate contents")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "rule_key": self.rule_key,
            "rule_family_id": self.rule_family_id,
            "context_scope": self.context_scope.to_dict(),
            "condition": self.condition.to_dict(),
            "action_guidance": self.action_guidance.to_dict(),
            "outcome_criterion": self.outcome_criterion.to_dict(),
            "rationale": self.rationale,
            "supporting_episode_ids": list(self.supporting_episode_ids),
            "generator_id": self.generator_id,
            "raw_response_hash": self.raw_response_hash,
            "migration_notes": list(self.migration_notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuleCandidate":
        _strict_keys(
            value,
            {
                "schema_version",
                "candidate_id",
                "rule_key",
                "rule_family_id",
                "context_scope",
                "condition",
                "action_guidance",
                "outcome_criterion",
                "rationale",
                "supporting_episode_ids",
                "generator_id",
                "raw_response_hash",
                "migration_notes",
            },
            "rule candidate",
        )
        return cls(
            schema_version=value["schema_version"],
            candidate_id=value["candidate_id"],
            rule_key=value["rule_key"],
            rule_family_id=value["rule_family_id"],
            context_scope=ContextScope.from_dict(value["context_scope"]),
            condition=ConditionPredicate.from_dict(value["condition"]),
            action_guidance=ActionGuidance.from_dict(value["action_guidance"]),
            outcome_criterion=OutcomeCriterion.from_dict(value["outcome_criterion"]),
            rationale=value["rationale"],
            supporting_episode_ids=tuple(value["supporting_episode_ids"]),
            generator_id=value["generator_id"],
            raw_response_hash=value["raw_response_hash"],
            migration_notes=tuple(value["migration_notes"]),
        )


@dataclass(frozen=True)
class VerifiedRule:
    rule_id: str
    rule_key: str
    rule_family_id: str
    rule_version: int
    context_scope: ContextScope
    condition: ConditionPredicate
    action_guidance: ActionGuidance
    outcome_criterion: OutcomeCriterion
    rationale: str
    status: str
    supporting_episode_ids: tuple[str, ...]
    harmful_compliance_episode_ids: tuple[str, ...]
    alternative_success_episode_ids: tuple[str, ...]
    alternative_failure_episode_ids: tuple[str, ...]
    irrelevant_episode_ids: tuple[str, ...]
    contradicting_episode_ids: tuple[str, ...]
    support_score: float
    contradiction_score: float
    margin: float
    confidence: float
    consecutive_failures: int
    post_proposal_evidence_count: int
    post_proposal_support_count: int
    post_proposal_contradiction_count: int
    post_proposal_neutral_count: int
    post_proposal_irrelevant_count: int
    activation_episode_id: Optional[str]
    candidate_ids: tuple[str, ...]
    supersedes_rule_id: Optional[str]
    derived_from_rule_ids: tuple[str, ...]
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
        _nonempty(self.rule_id, "rule_id")
        _nonempty(self.rule_key, "rule_key")
        _nonempty(self.rule_family_id, "rule_family_id")
        if (
            isinstance(self.rule_version, bool)
            or not isinstance(self.rule_version, int)
            or self.rule_version < 1
        ):
            raise ValueError("rule_version must be an integer >= 1")
        created_at = _timestamp(self.created_at, "created_at")
        updated_at = _timestamp(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at cannot be earlier than created_at")
        for name in (
            "consecutive_failures",
            "post_proposal_evidence_count",
            "post_proposal_support_count",
            "post_proposal_contradiction_count",
            "post_proposal_neutral_count",
            "post_proposal_irrelevant_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        support_score = _finite(self.support_score, "support_score")
        contradiction_score = _finite(self.contradiction_score, "contradiction_score")
        if support_score < 0 or contradiction_score < 0:
            raise ValueError("evidence scores must be non-negative")
        margin = _finite(self.margin, "margin")
        if not math.isclose(margin, support_score - contradiction_score, abs_tol=1e-12):
            raise ValueError("margin must equal support_score - contradiction_score")
        object.__setattr__(self, "support_score", support_score)
        object.__setattr__(self, "contradiction_score", contradiction_score)
        object.__setattr__(self, "margin", margin)
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
        for name in (
            "harmful_compliance_episode_ids",
            "alternative_success_episode_ids",
            "alternative_failure_episode_ids",
            "irrelevant_episode_ids",
        ):
            values = getattr(self, name)
            object.__setattr__(
                self,
                name,
                _tuple_of_unique_strings(values, name) if values else (),
            )
        expected_contradictions = set(self.harmful_compliance_episode_ids) | set(
            self.alternative_success_episode_ids
        )
        if set(self.contradicting_episode_ids) != expected_contradictions:
            raise ValueError(
                "contradicting evidence must equal harmful_compliance plus "
                "alternative_success evidence"
            )
        evidence_sets = (
            set(self.supporting_episode_ids),
            set(self.harmful_compliance_episode_ids),
            set(self.alternative_success_episode_ids),
            set(self.alternative_failure_episode_ids),
            set(self.irrelevant_episode_ids),
        )
        if sum(len(values) for values in evidence_sets) != len(
            set().union(*evidence_sets)
        ):
            raise ValueError("evidence taxonomy categories must be disjoint")
        if self.activation_episode_id is not None:
            activation_episode_id = _nonempty(
                self.activation_episode_id, "activation_episode_id"
            )
            if activation_episode_id not in self.supporting_episode_ids:
                raise ValueError("activation_episode_id must name supporting evidence")
        object.__setattr__(
            self,
            "candidate_ids",
            (
                _tuple_of_unique_strings(self.candidate_ids, "candidate_ids")
                if self.candidate_ids
                else ()
            ),
        )
        if self.supersedes_rule_id is not None:
            object.__setattr__(
                self,
                "supersedes_rule_id",
                _nonempty(self.supersedes_rule_id, "supersedes_rule_id"),
            )
        object.__setattr__(
            self,
            "derived_from_rule_ids",
            (
                _tuple_of_unique_strings(
                    self.derived_from_rule_ids, "derived_from_rule_ids"
                )
                if self.derived_from_rule_ids
                else ()
            ),
        )
        object.__setattr__(
            self,
            "verification_reasons",
            tuple(str(reason) for reason in self.verification_reasons),
        )
        if self.injection_provenance is not None:
            provenance = json.loads(_canonical_json(dict(self.injection_provenance)))
            object.__setattr__(self, "injection_provenance", provenance)
        if self.injected:
            if self.candidate_ids:
                raise ValueError("injected rules must not claim candidate provenance")
            if not self.injection_provenance:
                raise ValueError("injected rules require non-empty injection provenance")
        else:
            if len(self.candidate_ids) != 1:
                raise ValueError(
                    "non-injected rules require exactly one creation candidate ID"
                )
            if self.injection_provenance is not None:
                raise ValueError(
                    "non-injected rules must not contain injection provenance"
                )

    def to_prompt_text(self) -> str:
        relation = {
            "at_least": "keep at least",
            "at_most": "keep at most",
            "approximately": "keep approximately",
        }[self.action_guidance.direction]
        return (
            f"When {self.condition.field} {self.condition.operator} "
            f"{self.condition.value:g} within scope {self.context_scope.scope_id}, "
            f"{relation} {self.action_guidance.target} "
            f"{self.action_guidance.threshold:g} "
            f"(confidence {self.confidence:.0%}, family {self.rule_family_id} "
            f"v{self.rule_version})."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rule_id": self.rule_id,
            "rule_key": self.rule_key,
            "rule_family_id": self.rule_family_id,
            "rule_version": self.rule_version,
            "context_scope": self.context_scope.to_dict(),
            "condition": self.condition.to_dict(),
            "action_guidance": self.action_guidance.to_dict(),
            "outcome_criterion": self.outcome_criterion.to_dict(),
            "rationale": self.rationale,
            "status": self.status,
            "supporting_episode_ids": list(self.supporting_episode_ids),
            "harmful_compliance_episode_ids": list(self.harmful_compliance_episode_ids),
            "alternative_success_episode_ids": list(
                self.alternative_success_episode_ids
            ),
            "alternative_failure_episode_ids": list(
                self.alternative_failure_episode_ids
            ),
            "irrelevant_episode_ids": list(self.irrelevant_episode_ids),
            "contradicting_episode_ids": list(self.contradicting_episode_ids),
            "support_score": self.support_score,
            "contradiction_score": self.contradiction_score,
            "margin": self.margin,
            "confidence": self.confidence,
            "consecutive_failures": self.consecutive_failures,
            "post_proposal_evidence_count": self.post_proposal_evidence_count,
            "post_proposal_support_count": self.post_proposal_support_count,
            "post_proposal_contradiction_count": (
                self.post_proposal_contradiction_count
            ),
            "post_proposal_neutral_count": self.post_proposal_neutral_count,
            "post_proposal_irrelevant_count": self.post_proposal_irrelevant_count,
            "activation_episode_id": self.activation_episode_id,
            "candidate_ids": list(self.candidate_ids),
            "supersedes_rule_id": self.supersedes_rule_id,
            "derived_from_rule_ids": list(self.derived_from_rule_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "verification_reasons": list(self.verification_reasons),
            "injected": self.injected,
            "injection_provenance": (
                json.loads(_canonical_json(self.injection_provenance))
                if self.injection_provenance is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VerifiedRule":
        _strict_keys(
            value,
            {
                "schema_version",
                "rule_id",
                "rule_key",
                "rule_family_id",
                "rule_version",
                "context_scope",
                "condition",
                "action_guidance",
                "outcome_criterion",
                "rationale",
                "status",
                "supporting_episode_ids",
                "harmful_compliance_episode_ids",
                "alternative_success_episode_ids",
                "alternative_failure_episode_ids",
                "irrelevant_episode_ids",
                "contradicting_episode_ids",
                "support_score",
                "contradiction_score",
                "margin",
                "confidence",
                "consecutive_failures",
                "post_proposal_evidence_count",
                "post_proposal_support_count",
                "post_proposal_contradiction_count",
                "post_proposal_neutral_count",
                "post_proposal_irrelevant_count",
                "activation_episode_id",
                "candidate_ids",
                "supersedes_rule_id",
                "derived_from_rule_ids",
                "created_at",
                "updated_at",
                "verification_reasons",
                "injected",
                "injection_provenance",
            },
            "verified rule",
        )
        return cls(
            **{
                **dict(value),
                "context_scope": ContextScope.from_dict(value["context_scope"]),
                "condition": ConditionPredicate.from_dict(value["condition"]),
                "action_guidance": ActionGuidance.from_dict(value["action_guidance"]),
                "outcome_criterion": OutcomeCriterion.from_dict(
                    value["outcome_criterion"]
                ),
                "supporting_episode_ids": tuple(value["supporting_episode_ids"]),
                "harmful_compliance_episode_ids": tuple(
                    value["harmful_compliance_episode_ids"]
                ),
                "alternative_success_episode_ids": tuple(
                    value["alternative_success_episode_ids"]
                ),
                "alternative_failure_episode_ids": tuple(
                    value["alternative_failure_episode_ids"]
                ),
                "irrelevant_episode_ids": tuple(value["irrelevant_episode_ids"]),
                "contradicting_episode_ids": tuple(value["contradicting_episode_ids"]),
                "candidate_ids": tuple(value["candidate_ids"]),
                "derived_from_rule_ids": tuple(value["derived_from_rule_ids"]),
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
        _timestamp(self.timestamp, "event timestamp")
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
            (
                _tuple_of_unique_strings(self.episode_ids, "event.episode_ids")
                if self.episode_ids
                else ()
            ),
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
            "metrics": json.loads(_canonical_json(self.metrics)),
            "provenance": json.loads(_canonical_json(self.provenance)),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuleEvent":
        _strict_keys(
            value,
            {
                "schema_version",
                "event_id",
                "timestamp",
                "event_type",
                "rule_id",
                "candidate_id",
                "from_status",
                "to_status",
                "episode_ids",
                "reason",
                "metrics",
                "provenance",
            },
            "rule event",
        )
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
        activation_min_margin: float = 1,
        activation_confidence_threshold: float = 0.60,
        proposal_confidence_floor: float = 0.50,
        retirement_patience: int = 2,
        retirement_confidence_threshold: float = 0.45,
        registered_outcome_criterion: (
            OutcomeCriterion | Mapping[str, Any] | None
        ) = None,
        evidence_weights: Optional[Mapping[str, Any]] = None,
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
        activation_margin = _finite(activation_min_margin, "activation_min_margin")
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
        self.activation_min_margin = activation_margin
        self.retirement_patience = retirement_patience
        if registered_outcome_criterion is None:
            criterion = DEFAULT_REGISTERED_OUTCOME_CRITERION
        elif isinstance(registered_outcome_criterion, OutcomeCriterion):
            criterion = registered_outcome_criterion
        elif isinstance(registered_outcome_criterion, Mapping):
            criterion = OutcomeCriterion.from_dict(registered_outcome_criterion)
        else:
            raise TypeError(
                "registered_outcome_criterion must be an OutcomeCriterion, mapping, "
                "or None"
            )
        self.registered_outcome_criterion = criterion
        weights_input = dict(
            DEFAULT_EVIDENCE_WEIGHTS if evidence_weights is None else evidence_weights
        )
        _strict_keys(weights_input, set(EVIDENCE_TYPES), "evidence_weights")
        weights = {
            evidence_type: _finite(
                weights_input[evidence_type],
                f"evidence_weights[{evidence_type!r}]",
            )
            for evidence_type in sorted(EVIDENCE_TYPES)
        }
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("evidence weights must be non-negative")
        if weights["support"] <= 0 or weights["harmful_compliance"] <= 0:
            raise ValueError("support and harmful_compliance weights must be positive")
        if weights["alternative_success"] > weights["harmful_compliance"]:
            raise ValueError("alternative_success cannot outweigh harmful_compliance")
        if weights["alternative_failure"] != 0 or weights["irrelevant"] != 0:
            raise ValueError(
                "alternative_failure and irrelevant must have zero evidentiary weight"
            )
        self.evidence_weights = weights
        self._candidates: dict[str, RuleCandidate] = {}
        self._rules: dict[str, VerifiedRule] = {}
        self._events: list[RuleEvent] = []

    @staticmethod
    def _confidence(support: float, contradiction: float) -> float:
        return (support + 1.0) / (support + contradiction + 2.0)

    @staticmethod
    def _rule_key(
        context_scope: ContextScope,
        condition: ConditionPredicate,
        guidance: ActionGuidance,
        criterion: OutcomeCriterion,
    ) -> str:
        payload = {
            "context_scope": context_scope.to_dict(),
            "condition": condition.to_dict(),
            "action_guidance": guidance.to_dict(),
            "outcome_criterion": criterion.to_dict(),
        }
        return f"rule-{_hash(payload)[:20]}"

    @staticmethod
    def _rule_family_id(
        context_scope: ContextScope,
        condition: ConditionPredicate,
        guidance: ActionGuidance,
        criterion: OutcomeCriterion,
    ) -> str:
        """Return a stable family identity across numeric-threshold revisions.

        Predicate/action *shape* is part of family identity.  Otherwise opposite
        policies such as ``inflation > x -> at_most`` and
        ``inflation < y -> at_least`` collapse into one live family and the
        second policy is silently treated as a duplicate.  Numeric values and
        tolerances remain outside the identity so evidence-backed revisions can
        adjust thresholds without losing lineage.
        """

        predicate_shape = sorted(
            (
                {"field": predicate.field, "operator": predicate.operator}
                for predicate in context_scope.predicates
            ),
            key=_canonical_json,
        )

        payload = {
            # ``scope_id`` is an LLM-authored display label.  It must never
            # fragment otherwise identical semantic scopes into separate rule
            # families.  Global remains an explicit semantic case; non-global
            # identity comes only from the canonical predicate shape.
            "context_scope_kind": (
                "global" if context_scope.scope_id == "global" else "predicate_scope"
            ),
            "context_scope_predicate_shape": predicate_shape,
            "condition_field": condition.field,
            "condition_operator": condition.operator,
            "action_target": guidance.target,
            "action_direction": guidance.direction,
            "outcome_metric": criterion.metric,
            "outcome_operator": criterion.operator,
        }
        return f"family-{_hash(payload)[:20]}"

    @property
    def rules(self) -> tuple[VerifiedRule, ...]:
        return tuple(self._clone_rule(rule) for rule in self._rules.values())

    @property
    def events(self) -> tuple[RuleEvent, ...]:
        return tuple(self._clone_event(event) for event in self._events)

    def get(self, rule_id: str) -> Optional[VerifiedRule]:
        rule = self._rules.get(rule_id)
        return self._clone_rule(rule) if rule is not None else None

    @staticmethod
    def _clone_rule(rule: VerifiedRule) -> VerifiedRule:
        return VerifiedRule.from_dict(rule.to_dict())

    @staticmethod
    def _clone_event(event: RuleEvent) -> RuleEvent:
        return RuleEvent.from_dict(event.to_dict())

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
        timestamp = self._validate_event_timestamp(timestamp)
        core = {
            "index": len(self._events),
            "timestamp": timestamp,
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
            timestamp=timestamp,
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

    def _validate_event_timestamp(self, value: Any) -> int:
        timestamp = _timestamp(value, "semantic current_t")
        if self._events and timestamp < self._events[-1].timestamp:
            raise ValueError(
                "semantic operations must not move backward before the latest "
                "lifecycle event"
            )
        return timestamp

    def _metrics(self, rule: VerifiedRule) -> dict[str, Any]:
        return {
            "rule_family_id": rule.rule_family_id,
            "rule_version": rule.rule_version,
            "support_score": rule.support_score,
            "contradiction_score": rule.contradiction_score,
            "margin": rule.margin,
            "confidence": rule.confidence,
            "consecutive_failures": rule.consecutive_failures,
            "post_proposal_evidence_count": rule.post_proposal_evidence_count,
            "post_proposal_support_count": rule.post_proposal_support_count,
            "post_proposal_contradiction_count": (
                rule.post_proposal_contradiction_count
            ),
            "post_proposal_neutral_count": rule.post_proposal_neutral_count,
            "post_proposal_irrelevant_count": (
                rule.post_proposal_irrelevant_count
            ),
            "evidence_type_counts": {
                "support": len(rule.supporting_episode_ids),
                "harmful_compliance": len(rule.harmful_compliance_episode_ids),
                "alternative_success": len(rule.alternative_success_episode_ids),
                "alternative_failure": len(rule.alternative_failure_episode_ids),
                "irrelevant": len(rule.irrelevant_episode_ids),
            },
            "evidence_weights": dict(self.evidence_weights),
        }

    def _scores_from_categories(
        self,
        *,
        support_count: int,
        harmful_compliance_count: int,
        alternative_success_count: int,
    ) -> tuple[float, float, float, float]:
        support_score = support_count * self.evidence_weights["support"]
        contradiction_score = (
            harmful_compliance_count * self.evidence_weights["harmful_compliance"]
            + alternative_success_count * self.evidence_weights["alternative_success"]
        )
        margin = support_score - contradiction_score
        confidence = self._confidence(support_score, contradiction_score)
        return support_score, contradiction_score, margin, confidence

    def _family_rules(self, family_id: str) -> list[VerifiedRule]:
        return sorted(
            (rule for rule in self._rules.values() if rule.rule_family_id == family_id),
            key=lambda rule: rule.rule_version,
        )

    @staticmethod
    def _all_evidence_ids(rule: VerifiedRule) -> set[str]:
        return (
            set(rule.supporting_episode_ids)
            | set(rule.harmful_compliance_episode_ids)
            | set(rule.alternative_success_episode_ids)
            | set(rule.alternative_failure_episode_ids)
            | set(rule.irrelevant_episode_ids)
        )

    def _validate_evidence_event_sequence(
        self,
        rule: VerifiedRule,
        episodes_by_id: Mapping[str, EpisodeRecord],
    ) -> None:
        """Rebuild every historical metric from the append-only event log."""

        category_ids = {
            "support": tuple(rule.supporting_episode_ids),
            "harmful_compliance": tuple(rule.harmful_compliance_episode_ids),
            "alternative_success": tuple(rule.alternative_success_episode_ids),
            "alternative_failure": tuple(rule.alternative_failure_episode_ids),
            "irrelevant": tuple(rule.irrelevant_episode_ids),
        }
        expected_post = {
            episode_id: evidence_type
            for evidence_type, episode_ids in category_ids.items()
            for episode_id in episode_ids
            if episodes_by_id[episode_id].outcome_t > rule.created_at
        }
        event_type_to_evidence = {
            f"{evidence_type}_evidence_added": evidence_type
            for evidence_type in category_ids
        }
        observed_post: dict[str, str] = {}
        evidence_events: list[tuple[int, RuleEvent, str, str]] = []
        for event_index, event in enumerate(self._events):
            if event.rule_id != rule.rule_id:
                continue
            evidence_type = event_type_to_evidence.get(event.event_type)
            if evidence_type is None:
                continue
            if len(event.episode_ids) != 1:
                raise ValueError(
                    f"rule {rule.rule_id} evidence event must name exactly one episode"
                )
            episode_id = event.episode_ids[0]
            if episode_id in observed_post:
                raise ValueError(
                    f"rule {rule.rule_id} repeats an evidence-added event for "
                    f"{episode_id}"
                )
            if expected_post.get(episode_id) != evidence_type:
                raise ValueError(
                    f"rule {rule.rule_id} evidence event is not bound to its "
                    "post-proposal taxonomy"
                )
            if (
                event.provenance.get("evidence_type") != evidence_type
                or event.provenance.get("evidence_weight")
                != self.evidence_weights[evidence_type]
                or event.provenance.get("registered_outcome_criterion")
                != self.registered_outcome_criterion.to_dict()
            ):
                raise ValueError(
                    f"rule {rule.rule_id} evidence event provenance is inconsistent"
                )
            observed_post[episode_id] = evidence_type
            evidence_events.append((event_index, event, evidence_type, episode_id))
        if observed_post != expected_post:
            raise ValueError(
                f"rule {rule.rule_id} post-proposal evidence is not exactly "
                "represented by lifecycle events"
            )

        counts = {
            evidence_type: sum(
                episodes_by_id[episode_id].outcome_t <= rule.created_at
                for episode_id in episode_ids
            )
            for evidence_type, episode_ids in category_ids.items()
        }
        post_counts = {evidence_type: 0 for evidence_type in category_ids}
        consecutive_failures = 0
        confidence_override: Optional[float] = None
        if rule.injected:
            provenance = rule.injection_provenance or {}
            confidence_override = _finite(
                provenance.get("initial_confidence"),
                "injection_provenance.initial_confidence",
            )
            if not 0.0 <= confidence_override <= 1.0:
                raise ValueError(
                    "injection_provenance.initial_confidence must be in [0, 1]"
                )

        def expected_metrics() -> dict[str, Any]:
            support_score, contradiction_score, margin, confidence = (
                self._scores_from_categories(
                    support_count=counts["support"],
                    harmful_compliance_count=counts["harmful_compliance"],
                    alternative_success_count=counts["alternative_success"],
                )
            )
            if confidence_override is not None:
                confidence = confidence_override
            return {
                "rule_family_id": rule.rule_family_id,
                "rule_version": rule.rule_version,
                "support_score": support_score,
                "contradiction_score": contradiction_score,
                "margin": margin,
                "confidence": confidence,
                "consecutive_failures": consecutive_failures,
                "post_proposal_evidence_count": sum(post_counts.values()),
                "post_proposal_support_count": post_counts["support"],
                "post_proposal_contradiction_count": (
                    post_counts["harmful_compliance"]
                    + post_counts["alternative_success"]
                ),
                "post_proposal_neutral_count": post_counts["alternative_failure"],
                "post_proposal_irrelevant_count": post_counts["irrelevant"],
                "evidence_type_counts": dict(counts),
                "evidence_weights": dict(self.evidence_weights),
            }

        expected_transition_indexes: set[int] = set()
        evidence_by_index = {
            event_index: (event, evidence_type, episode_id)
            for event_index, event, evidence_type, episode_id in evidence_events
        }
        rule_event_indexes = [
            event_index
            for event_index, event in enumerate(self._events)
            if event.rule_id == rule.rule_id
        ]
        for event_index in rule_event_indexes:
            event = self._events[event_index]
            evidence_item = evidence_by_index.get(event_index)
            if evidence_item is None:
                if dict(event.metrics) != expected_metrics():
                    raise ValueError(
                        f"rule {rule.rule_id} event metrics do not reproduce at "
                        f"{event.event_id}"
                    )
                continue

            _, evidence_type, _ = evidence_item
            if (
                event.from_status not in {"provisional", "active"}
                or event.to_status != event.from_status
            ):
                raise ValueError(
                    f"rule {rule.rule_id} evidence event changes lifecycle status"
                )
            counts[evidence_type] += 1
            post_counts[evidence_type] += 1
            if evidence_type == "harmful_compliance":
                consecutive_failures += 1
            elif evidence_type != "irrelevant":
                consecutive_failures = 0
            if evidence_type != "irrelevant":
                confidence_override = None
            support_score, contradiction_score, margin, confidence = (
                self._scores_from_categories(
                    support_count=counts["support"],
                    harmful_compliance_count=counts["harmful_compliance"],
                    alternative_success_count=counts["alternative_success"],
                )
            )
            if dict(event.metrics) != expected_metrics():
                raise ValueError(
                    f"rule {rule.rule_id} event metrics do not reproduce at "
                    f"{event.event_id}"
                )
            should_retire = evidence_type != "irrelevant" and (
                consecutive_failures >= self.retirement_patience
                or confidence < self.retirement_confidence_threshold
            )
            should_activate = (
                event.from_status == "provisional"
                and evidence_type == "support"
                and post_counts["support"] >= 1
                and counts["support"] >= self.activation_min_support
                and margin >= self.activation_min_margin
                and confidence >= self.activation_confidence_threshold
            )
            expected_transition = (
                "rule_retired"
                if should_retire
                else "rule_activated" if should_activate else None
            )
            next_event = (
                self._events[event_index + 1]
                if event_index + 1 < len(self._events)
                else None
            )
            if expected_transition is not None:
                if (
                    next_event is None
                    or next_event.rule_id != rule.rule_id
                    or next_event.event_type != expected_transition
                    or next_event.timestamp != event.timestamp
                    or next_event.from_status != event.from_status
                    or next_event.to_status
                    != (
                        "retired"
                        if expected_transition == "rule_retired"
                        else "active"
                    )
                    or next_event.episode_ids != event.episode_ids
                ):
                    raise ValueError(
                        f"rule {rule.rule_id} does not follow its registered "
                        "activation/retirement schedule"
                    )
                expected_transition_indexes.add(event_index + 1)

        observed_transition_indexes = {
            event_index
            for event_index, event in enumerate(self._events)
            if event.rule_id == rule.rule_id
            and event.event_type in {"rule_activated", "rule_retired"}
        }
        if observed_transition_indexes != expected_transition_indexes:
            raise ValueError(
                f"rule {rule.rule_id} has an unearned or misplaced lifecycle "
                "transition"
            )

        if consecutive_failures != rule.consecutive_failures:
            raise ValueError(
                f"rule {rule.rule_id} consecutive_failures does not match the "
                "chronological evidence-event streak"
            )
        if expected_metrics() != self._metrics(rule):
            raise ValueError(
                f"rule {rule.rule_id} final metrics do not reproduce from its "
                "chronological event ledger"
            )

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
                    raise KeyError(
                        f"unknown episode ID in proposal prompt: {episode_id}"
                    )
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
            "context_scope": {
                "scope_id": "global, or a stable descriptive scope name",
                "predicates": [
                    {
                        "field": f"one of {list(self.allowed_condition_fields)}",
                        "operator": f"one of {sorted(CONDITION_OPERATORS)}",
                        "value": "finite number",
                        "tolerance": "non-negative number",
                    }
                ],
            },
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
            "rationale": "brief evidence-grounded explanation",
            "supporting_episode_ids": (
                f"at least {self.min_candidate_support} IDs copied exactly from evidence"
            ),
        }
        return (
            "Propose one semantic decision rule using only the finalized evidence below. "
            "Return exactly one JSON object and no additional keys. Do not invent or "
            "duplicate episode IDs. A proposal is never activated directly; it is "
            "verified and must receive later distinct evidence. Every claimed support "
            "episode must independently satisfy all three checks: (1) the condition is "
            "true in pre_state and context_scope, (2) the executed action satisfies the "
            "absolute guidance threshold, and (3) the outcome passes the verifier's "
            "pre-registered criterion. Do not output or choose an outcome criterion. "
            "The action directions are exact absolute predicates: at_least means "
            "executed value >= threshold, at_most means executed value <= threshold, "
            "and approximately means within tolerance of threshold. Do not describe a "
            "comparison between two episodes "
            "unless both episodes themselves pass the rule. Before returning JSON, check "
            "each copied support ID against those three predicates.\n"
            'For context_scope, use exactly {"scope_id": "global", '
            '"predicates": []} unless the evidence exposes every field needed by '
            "a narrower scope.\n"
            "Verifier-registered outcome criterion (not candidate-controlled):\n"
            f"{json.dumps(self.registered_outcome_criterion.to_dict(), sort_keys=True)}\n"
            f"Allowed JSON schema:\n{json.dumps(schema, indent=2, sort_keys=True)}\n"
            f"Allowed episode IDs: {[item['episode_id'] for item in evidence]}\n"
            f"Evidence:\n{json.dumps(evidence, indent=2, sort_keys=True)}"
        )

    @staticmethod
    def _extract_json_object(raw_response: str) -> Mapping[str, Any]:
        if not isinstance(raw_response, str) or not raw_response.strip():
            raise CandidateParseError("candidate response is empty")
        text = raw_response.strip()
        fenced = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE
        )
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
                raise CandidateParseError(
                    f"malformed candidate JSON: {exc.msg}"
                ) from exc
        if not isinstance(value, Mapping):
            raise CandidateParseError("candidate JSON root must be an object")
        return value

    def parse_candidate(
        self, raw_response: str, *, generator_id: str = "llm-candidate-generator"
    ) -> RuleCandidate:
        try:
            value = self._extract_json_object(raw_response)
            required = {
                "condition",
                "action_guidance",
                "rationale",
                "supporting_episode_ids",
            }
            optional_legacy = {"outcome_criterion", "context_scope"}
            missing = sorted(required - set(value))
            extra = sorted(set(value) - required - optional_legacy)
            if missing or extra:
                details = []
                if missing:
                    details.append(f"missing={missing}")
                if extra:
                    details.append(f"extra={extra}")
                raise ValueError(f"invalid candidate JSON keys: {', '.join(details)}")
            migration_notes: list[str] = []
            condition = ConditionPredicate.from_dict(value["condition"])
            guidance, direction_migration = ActionGuidance.from_candidate_dict(
                value["action_guidance"]
            )
            if direction_migration is not None:
                migration_notes.append(direction_migration)
            if "context_scope" in value:
                context_scope = ContextScope.from_dict(value["context_scope"])
            else:
                context_scope = ContextScope.global_scope()
                migration_notes.append(
                    "legacy candidate without context_scope explicitly migrated to "
                    "global scope"
                )
            criterion = self.registered_outcome_criterion
            if "outcome_criterion" in value:
                legacy_criterion = OutcomeCriterion.from_dict(
                    value["outcome_criterion"]
                )
                if legacy_criterion.to_dict() != criterion.to_dict():
                    raise ValueError(
                        "candidate outcome_criterion is not the verifier-registered "
                        "criterion; candidates cannot choose metrics or thresholds"
                    )
                migration_notes.append(
                    "legacy candidate outcome_criterion matched the pre-registered "
                    "criterion and was removed from candidate control"
                )
            if condition.field not in self.allowed_condition_fields:
                raise ValueError(f"unsupported condition field {condition.field!r}")
            unsupported_scope_fields = sorted(
                {
                    predicate.field
                    for predicate in context_scope.predicates
                    if predicate.field not in self.allowed_condition_fields
                }
            )
            if unsupported_scope_fields:
                raise ValueError(
                    "unsupported context scope fields " f"{unsupported_scope_fields}"
                )
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

        raw_response_hash = _hash({"raw_response": raw_response})
        content = {
            "context_scope": context_scope.to_dict(),
            "condition": condition.to_dict(),
            "action_guidance": guidance.to_dict(),
            "outcome_criterion": criterion.to_dict(),
            "rationale": rationale,
            "supporting_episode_ids": list(support_ids),
            "generator_id": generator_id,
            "raw_response_hash": raw_response_hash,
            "migration_notes": list(migration_notes),
        }
        rule_key = self._rule_key(context_scope, condition, guidance, criterion)
        family_id = self._rule_family_id(context_scope, condition, guidance, criterion)
        return RuleCandidate(
            candidate_id=f"cand-{_hash(content)[:20]}",
            rule_key=rule_key,
            rule_family_id=family_id,
            context_scope=context_scope,
            condition=condition,
            action_guidance=guidance,
            outcome_criterion=criterion,
            rationale=rationale,
            supporting_episode_ids=support_ids,
            generator_id=_nonempty(generator_id, "generator_id"),
            raw_response_hash=raw_response_hash,
            migration_notes=tuple(migration_notes),
        )

    @staticmethod
    def _classification(
        candidate_or_rule: RuleCandidate | VerifiedRule, episode: EpisodeRecord
    ) -> str:
        if not candidate_or_rule.context_scope.matches(episode.pre_state):
            return "irrelevant"
        if not candidate_or_rule.condition.matches(episode.pre_state):
            return "irrelevant"
        compliant = candidate_or_rule.action_guidance.is_consistent(
            episode.executed_action
        )
        successful = candidate_or_rule.outcome_criterion.passes(episode)
        if compliant and successful:
            return "support"
        if compliant and not successful:
            return "harmful_compliance"
        if not compliant and successful:
            return "alternative_success"
        return "alternative_failure"

    def propose(
        self,
        raw_response: str,
        *,
        current_t: int,
        generator_id: str = "llm-candidate-generator",
    ) -> VerifiedRule:
        try:
            candidate = self.parse_candidate(raw_response, generator_id=generator_id)
        except CandidateParseError as exc:
            self._append_event(
                timestamp=current_t,
                event_type="candidate_parse_rejected",
                reason=str(exc),
                provenance={"raw_response_hash": _hash({"raw_response": raw_response})},
            )
            raise
        return self.submit_candidate(candidate, current_t=current_t)

    def propose_unverified_immediate(
        self,
        raw_response: str,
        *,
        current_t: int,
        generator_id: str = "llm-candidate-generator",
    ) -> VerifiedRule:
        """Parse a candidate, then activate it without evidence admission.

        This intentionally unsafe control is used only for the preregistered
        unverified-memory arm.  Candidate parsing and field allow-lists remain
        enforced so the actor receives the same rule language, but claimed
        episode IDs are not admitted as evidence and later episodes cannot
        update or retire the rule.  The bypass is explicit in immutable
        injection provenance rather than being disguised as verifier support.
        """

        current_t = self._validate_event_timestamp(current_t)
        try:
            candidate = self.parse_candidate(
                raw_response,
                generator_id=generator_id,
            )
        except CandidateParseError as exc:
            self._append_event(
                timestamp=current_t,
                event_type="candidate_parse_rejected",
                reason=str(exc),
                provenance={"raw_response_hash": _hash({"raw_response": raw_response})},
            )
            raise

        existing_candidate = self._candidates.get(candidate.candidate_id)
        if existing_candidate is not None and existing_candidate != candidate:
            raise ValueError(
                "candidate_id collision would overwrite immutable candidate provenance"
            )
        if existing_candidate is None:
            self._candidates[candidate.candidate_id] = candidate
        for rule in self._rules.values():
            provenance = dict(rule.injection_provenance or {})
            if provenance.get("source_candidate_id") == candidate.candidate_id:
                self._append_event(
                    timestamp=current_t,
                    event_type="duplicate_unverified_candidate_ignored",
                    rule_id=rule.rule_id,
                    from_status=rule.status,
                    to_status=rule.status,
                    reason="the same parsed unverified candidate is already active",
                    metrics=self._metrics(rule),
                    provenance={
                        "source_candidate_id": candidate.candidate_id,
                        "semantic_policy": "unverified-immediate",
                    },
                )
                return self._clone_rule(rule)

        return self.inject_active_rule(
            condition=candidate.condition,
            action_guidance=candidate.action_guidance,
            outcome_criterion=candidate.outcome_criterion,
            rationale=candidate.rationale,
            current_t=current_t,
            injection_id=f"unverified:{candidate.candidate_id}",
            provenance={
                "semantic_policy": "unverified-immediate",
                "source_candidate_id": candidate.candidate_id,
                "generator_id": candidate.generator_id,
                "raw_response_hash": candidate.raw_response_hash,
                "requested_support_ids": list(candidate.supporting_episode_ids),
                "evidence_admission": False,
                "retirement_enabled": False,
            },
            initial_confidence=1.0,
            context_scope=candidate.context_scope,
        )

    def submit_candidate(
        self, candidate: RuleCandidate, *, current_t: int
    ) -> VerifiedRule:
        current_t = self._validate_event_timestamp(current_t)
        if (
            candidate.outcome_criterion.to_dict()
            != self.registered_outcome_criterion.to_dict()
        ):
            raise ValueError(
                "candidate outcome criterion does not match the verifier-registered "
                "criterion"
            )

        existing_candidate = self._candidates.get(candidate.candidate_id)
        if existing_candidate is not None and existing_candidate != candidate:
            raise ValueError(
                "candidate_id collision would overwrite immutable candidate provenance"
            )
        if existing_candidate is None:
            self._candidates[candidate.candidate_id] = candidate

        family_rules = self._family_rules(candidate.rule_family_id)
        live_rules = [
            rule for rule in family_rules if rule.status in {"provisional", "active"}
        ]
        if live_rules:
            existing = live_rules[-1]
            self._append_event(
                timestamp=current_t,
                event_type="duplicate_semantic_candidate_ignored",
                rule_id=existing.rule_id,
                candidate_id=candidate.candidate_id,
                from_status=existing.status,
                to_status=existing.status,
                reason="the rule family already has a live version",
                metrics=self._metrics(existing),
                provenance={"candidate_rule_key": candidate.rule_key},
            )
            return self._clone_rule(existing)

        latest = family_rules[-1] if family_rules else None
        if latest is not None:
            prior_evidence: set[str] = set()
            for prior_rule in family_rules:
                prior_evidence.update(self._all_evidence_ids(prior_rule))
            qualifying_new_support = []
            for episode_id in dict.fromkeys(candidate.supporting_episode_ids):
                episode = self.episodic_track.get(episode_id)
                if (
                    episode is not None
                    and episode.outcome_t <= current_t
                    and episode.outcome_t > latest.updated_at
                    and episode_id not in prior_evidence
                    and self._classification(candidate, episode) == "support"
                ):
                    qualifying_new_support.append(episode_id)
            if not qualifying_new_support:
                self._append_event(
                    timestamp=current_t,
                    event_type="terminal_family_candidate_without_new_support_ignored",
                    rule_id=latest.rule_id,
                    candidate_id=candidate.candidate_id,
                    from_status=latest.status,
                    to_status=latest.status,
                    reason=(
                        "a rejected or retired family requires at least one new, "
                        "observable supporting episode with outcome_t later than "
                        "the latest terminal rule update before a new version"
                    ),
                    metrics=self._metrics(latest),
                    provenance={
                        "candidate_rule_key": candidate.rule_key,
                        "latest_terminal_updated_at": latest.updated_at,
                    },
                )
                return self._clone_rule(latest)

        rule_version = 1 if latest is None else latest.rule_version + 1
        rule_id = f"{candidate.rule_family_id}:v{rule_version}"
        supersedes_rule_id = latest.rule_id if latest is not None else None
        derived_from_rule_ids = (latest.rule_id,) if latest is not None else ()

        requested_ids = candidate.supporting_episode_ids
        unique_requested = tuple(dict.fromkeys(requested_ids))
        reasons: list[str] = []
        if len(unique_requested) != len(requested_ids):
            reasons.append("candidate contains duplicate supporting episode IDs")
        if len(unique_requested) < self.min_candidate_support:
            reasons.append(
                f"candidate requires at least {self.min_candidate_support} unique support IDs"
            )

        evidence_ids: dict[str, list[str]] = {
            evidence_type: [] for evidence_type in EVIDENCE_TYPES
        }
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
            evidence_ids[classification].append(episode_id)
            if classification == "support":
                continue
            elif classification == "irrelevant":
                reasons.append(
                    "condition or context scope does not hold in claimed support "
                    f"episode: {episode_id}"
                )
            else:
                if classification in {"alternative_success", "alternative_failure"}:
                    reasons.append(
                        f"action guidance is inconsistent with claimed support: {episode_id}"
                    )
                else:
                    reasons.append(
                        f"outcome criterion fails in claimed support: {episode_id}"
                    )

        # Search all other finalized episodes for negative, neutral, or
        # out-of-scope observations. Positive unlisted episodes are never
        # promoted to candidate support, which keeps proposal evidence explicit;
        # irrelevant observations are retained at zero weight so the five-way
        # audit denominator remains reproducible.
        for episode in self.episodic_track.finalized_episodes:
            if episode.outcome_t > current_t:
                continue
            if episode.episode_id in unique_requested:
                continue
            classification = self._classification(candidate, episode)
            if classification != "support":
                evidence_ids[classification].append(episode.episode_id)
        for evidence_type in EVIDENCE_TYPES:
            evidence_ids[evidence_type] = list(
                dict.fromkeys(evidence_ids[evidence_type])
            )
        if len(evidence_ids["support"]) < self.min_candidate_support:
            reasons.append(
                f"only {len(evidence_ids['support'])} claimed episodes pass all "
                "verification checks"
            )

        support_score, contradiction_score, margin, confidence = (
            self._scores_from_categories(
                support_count=len(evidence_ids["support"]),
                harmful_compliance_count=len(evidence_ids["harmful_compliance"]),
                alternative_success_count=len(evidence_ids["alternative_success"]),
            )
        )
        if confidence < self.proposal_confidence_floor:
            reasons.append(
                "candidate confidence is below the provisional admission floor"
            )
        rejected = bool(reasons) or confidence < self.proposal_confidence_floor
        status = "rejected" if rejected else "provisional"
        rule = VerifiedRule(
            rule_id=rule_id,
            rule_key=candidate.rule_key,
            rule_family_id=candidate.rule_family_id,
            rule_version=rule_version,
            context_scope=candidate.context_scope,
            condition=candidate.condition,
            action_guidance=candidate.action_guidance,
            outcome_criterion=candidate.outcome_criterion,
            rationale=candidate.rationale,
            status=status,
            supporting_episode_ids=tuple(evidence_ids["support"]),
            harmful_compliance_episode_ids=tuple(evidence_ids["harmful_compliance"]),
            alternative_success_episode_ids=tuple(evidence_ids["alternative_success"]),
            alternative_failure_episode_ids=tuple(evidence_ids["alternative_failure"]),
            irrelevant_episode_ids=tuple(evidence_ids["irrelevant"]),
            contradicting_episode_ids=tuple(
                evidence_ids["harmful_compliance"] + evidence_ids["alternative_success"]
            ),
            support_score=support_score,
            contradiction_score=contradiction_score,
            margin=margin,
            confidence=confidence,
            consecutive_failures=0,
            post_proposal_evidence_count=0,
            post_proposal_support_count=0,
            post_proposal_contradiction_count=0,
            post_proposal_neutral_count=0,
            post_proposal_irrelevant_count=0,
            activation_episode_id=None,
            candidate_ids=(candidate.candidate_id,),
            supersedes_rule_id=supersedes_rule_id,
            derived_from_rule_ids=derived_from_rule_ids,
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
                "registered_outcome_criterion": (
                    self.registered_outcome_criterion.to_dict()
                ),
                "migration_notes": list(candidate.migration_notes),
                "supersedes_rule_id": supersedes_rule_id,
                "derived_from_rule_ids": list(derived_from_rule_ids),
            },
        )
        return self._clone_rule(rule)

    def observe_episode(
        self, rule_id: str, episode_id: str, *, current_t: int
    ) -> VerifiedRule:
        current_t = self._validate_event_timestamp(current_t)
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
            return self._clone_rule(rule)
        provenance = dict(rule.injection_provenance or {})
        if (
            rule.injected
            and provenance.get("evidence_admission") is False
            and provenance.get("retirement_enabled") is False
        ):
            self._append_event(
                timestamp=current_t,
                event_type="evidence_ignored_unverified_policy",
                rule_id=rule_id,
                from_status=rule.status,
                to_status=rule.status,
                episode_ids=(episode_id,),
                reason=(
                    "unverified-immediate control disables evidence admission "
                    "and retirement"
                ),
                metrics=self._metrics(rule),
                provenance={
                    "semantic_policy": "unverified-immediate",
                    "evidence_admission": False,
                    "retirement_enabled": False,
                },
            )
            return self._clone_rule(rule)
        if episode_id in self._all_evidence_ids(rule):
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
            return self._clone_rule(rule)

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
            updated = replace(
                rule,
                irrelevant_episode_ids=(
                    *rule.irrelevant_episode_ids,
                    episode_id,
                ),
                post_proposal_evidence_count=(
                    rule.post_proposal_evidence_count + 1
                ),
                post_proposal_irrelevant_count=(
                    rule.post_proposal_irrelevant_count + 1
                ),
                updated_at=int(current_t),
            )
            self._rules[rule_id] = updated
            self._append_event(
                timestamp=current_t,
                event_type="irrelevant_evidence_added",
                rule_id=rule_id,
                from_status=rule.status,
                to_status=rule.status,
                episode_ids=(episode_id,),
                reason=(
                    "rule condition or context scope does not hold; retained as "
                    "a zero-weight audit observation"
                ),
                metrics=self._metrics(updated),
                provenance={
                    "evidence_type": classification,
                    "evidence_weight": self.evidence_weights[classification],
                    "registered_outcome_criterion": (
                        self.registered_outcome_criterion.to_dict()
                    ),
                },
            )
            return self._clone_rule(updated)

        supports = list(rule.supporting_episode_ids)
        harmful = list(rule.harmful_compliance_episode_ids)
        alternative_success = list(rule.alternative_success_episode_ids)
        alternative_failure = list(rule.alternative_failure_episode_ids)
        if classification == "support":
            supports.append(episode_id)
            failures = 0
        elif classification == "harmful_compliance":
            harmful.append(episode_id)
            failures = rule.consecutive_failures + 1
        elif classification == "alternative_success":
            alternative_success.append(episode_id)
            failures = 0
        else:
            alternative_failure.append(episode_id)
            failures = 0
        contradictions = harmful + alternative_success
        support_score, contradiction_score, margin, confidence = (
            self._scores_from_categories(
                support_count=len(supports),
                harmful_compliance_count=len(harmful),
                alternative_success_count=len(alternative_success),
            )
        )
        contradiction_increment = int(
            classification in {"harmful_compliance", "alternative_success"}
        )
        neutral_increment = int(classification == "alternative_failure")
        updated = replace(
            rule,
            supporting_episode_ids=tuple(supports),
            harmful_compliance_episode_ids=tuple(harmful),
            alternative_success_episode_ids=tuple(alternative_success),
            alternative_failure_episode_ids=tuple(alternative_failure),
            contradicting_episode_ids=tuple(contradictions),
            support_score=support_score,
            contradiction_score=contradiction_score,
            margin=margin,
            confidence=confidence,
            consecutive_failures=failures,
            post_proposal_evidence_count=rule.post_proposal_evidence_count + 1,
            post_proposal_support_count=(
                rule.post_proposal_support_count + int(classification == "support")
            ),
            post_proposal_contradiction_count=(
                rule.post_proposal_contradiction_count + contradiction_increment
            ),
            post_proposal_neutral_count=(
                rule.post_proposal_neutral_count + neutral_increment
            ),
            updated_at=int(current_t),
        )
        self._rules[rule_id] = updated
        self._append_event(
            timestamp=current_t,
            event_type=f"{classification}_evidence_added",
            rule_id=rule_id,
            from_status=rule.status,
            to_status=rule.status,
            episode_ids=(episode_id,),
            reason={
                "support": "guidance was followed and the registered outcome passed",
                "harmful_compliance": (
                    "guidance was followed but the registered outcome failed"
                ),
                "alternative_success": (
                    "guidance was not followed and the registered outcome passed"
                ),
                "alternative_failure": (
                    "guidance was not followed and the registered outcome failed; "
                    "this is neutral evidence about the rule"
                ),
            }[classification],
            metrics=self._metrics(updated),
            provenance={
                "evidence_type": classification,
                "evidence_weight": self.evidence_weights[classification],
                "registered_outcome_criterion": (
                    self.registered_outcome_criterion.to_dict()
                ),
            },
        )

        should_retire = (
            updated.consecutive_failures >= self.retirement_patience
            or updated.confidence < self.retirement_confidence_threshold
        )
        should_activate = (
            updated.status == "provisional"
            and classification == "support"
            and updated.post_proposal_support_count >= 1
            and len(updated.supporting_episode_ids) >= self.activation_min_support
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
            return self._clone_rule(retired)
        if should_activate:
            active = replace(
                updated,
                status="active",
                activation_episode_id=episode_id,
                updated_at=int(current_t),
            )
            self._rules[rule_id] = active
            self._append_event(
                timestamp=current_t,
                event_type="rule_activated",
                rule_id=rule_id,
                from_status="provisional",
                to_status="active",
                episode_ids=(episode_id,),
                reason=(
                    "distinct post-proposal support passed activation thresholds; "
                    "non-support evidence cannot activate a rule"
                ),
                metrics=self._metrics(active),
            )
            return self._clone_rule(active)
        return self._clone_rule(updated)

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
        context_scope: Optional[ContextScope] = None,
    ) -> VerifiedRule:
        current_t = self._validate_event_timestamp(current_t)
        injection_id = _nonempty(injection_id, "injection_id")
        provenance_copy = json.loads(_canonical_json(dict(provenance)))
        if not provenance_copy:
            raise ValueError("injection provenance must not be empty")
        confidence = _finite(initial_confidence, "initial_confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("initial_confidence must be in [0, 1]")
        if condition.field not in self.allowed_condition_fields:
            raise ValueError(
                f"unsupported injected condition field {condition.field!r}"
            )
        if action_guidance.target not in self.allowed_action_targets:
            raise ValueError(
                f"unsupported injected action target {action_guidance.target!r}"
            )
        scope = context_scope or ContextScope.global_scope()
        unsupported_scope_fields = sorted(
            {
                predicate.field
                for predicate in scope.predicates
                if predicate.field not in self.allowed_condition_fields
            }
        )
        if unsupported_scope_fields:
            raise ValueError(
                f"unsupported injected context scope fields {unsupported_scope_fields}"
            )
        semantic_key = self._rule_key(
            scope, condition, action_guidance, outcome_criterion
        )
        base_family_id = self._rule_family_id(
            scope, condition, action_guidance, outcome_criterion
        )
        family_id = f"{base_family_id}:injected:{_hash(injection_id)[:12]}"
        rule_id = f"{family_id}:v1"
        if rule_id in self._rules:
            raise ValueError(f"duplicate injected rule ID: {rule_id}")
        rule = VerifiedRule(
            rule_id=rule_id,
            rule_key=semantic_key,
            rule_family_id=family_id,
            rule_version=1,
            context_scope=scope,
            condition=condition,
            action_guidance=action_guidance,
            outcome_criterion=outcome_criterion,
            rationale=_nonempty(rationale, "rationale"),
            status="active",
            supporting_episode_ids=(),
            harmful_compliance_episode_ids=(),
            alternative_success_episode_ids=(),
            alternative_failure_episode_ids=(),
            irrelevant_episode_ids=(),
            contradicting_episode_ids=(),
            support_score=0,
            contradiction_score=0,
            margin=0,
            confidence=confidence,
            consecutive_failures=0,
            post_proposal_evidence_count=0,
            post_proposal_support_count=0,
            post_proposal_contradiction_count=0,
            post_proposal_neutral_count=0,
            post_proposal_irrelevant_count=0,
            activation_episode_id=None,
            candidate_ids=(),
            supersedes_rule_id=None,
            derived_from_rule_ids=(),
            created_at=int(current_t),
            updated_at=int(current_t),
            verification_reasons=("experimental injection bypass",),
            injected=True,
            injection_provenance={
                **provenance_copy,
                "injection_id": injection_id,
                "initial_confidence": confidence,
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
        return self._clone_rule(rule)

    def retrieve(
        self,
        current_state: Mapping[str, Any],
        *,
        current_t: int,
        limit: int = 3,
        log_selection: bool = True,
        log_empty_selection: bool = False,
    ) -> tuple[VerifiedRule, ...]:
        current_t = self._validate_event_timestamp(current_t)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be non-negative")
        if not isinstance(log_selection, bool) or not isinstance(
            log_empty_selection, bool
        ):
            raise ValueError("retrieval logging flags must be boolean")
        relevant = [
            rule
            for rule in self._rules.values()
            if (
                rule.status == "active"
                and rule.context_scope.matches(current_state)
                and rule.condition.matches(current_state)
            )
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
        if not selected and log_empty_selection:
            self._append_event(
                timestamp=current_t,
                event_type="active_rule_retrieval_empty",
                reason="no active rule matched within the retrieval limit",
                provenance={
                    "limit": limit,
                    "state_hash": _hash(dict(current_state)),
                },
            )
        return tuple(self._clone_rule(rule) for rule in selected)

    def validate_referential_integrity(self) -> None:
        episodes_by_id = {
            episode.episode_id: episode
            for episode in self.episodic_track.finalized_episodes
        }
        known_episode_ids = set(episodes_by_id)
        event_ids: set[str] = set()
        versions_by_family: dict[str, list[int]] = {}
        for rule in self._rules.values():
            expected_rule_key = self._rule_key(
                rule.context_scope,
                rule.condition,
                rule.action_guidance,
                rule.outcome_criterion,
            )
            if rule.rule_key != expected_rule_key:
                raise ValueError(f"rule {rule.rule_id} has inconsistent rule_key")
            expected_rule_id = f"{rule.rule_family_id}:v{rule.rule_version}"
            if rule.rule_id != expected_rule_id:
                raise ValueError(f"rule {rule.rule_id} has inconsistent rule_id")
            base_family_id = self._rule_family_id(
                rule.context_scope,
                rule.condition,
                rule.action_guidance,
                rule.outcome_criterion,
            )
            if rule.injected:
                provenance = rule.injection_provenance or {}
                injection_id = provenance.get("injection_id")
                if not isinstance(injection_id, str) or not injection_id:
                    raise ValueError(
                        f"injected rule {rule.rule_id} lacks a valid injection_id"
                    )
                expected_family_id = (
                    f"{base_family_id}:injected:{_hash(injection_id)[:12]}"
                )
                if rule.rule_family_id != expected_family_id:
                    raise ValueError(
                        f"injected rule {rule.rule_id} has inconsistent "
                        "rule_family_id"
                    )
            else:
                expected_family_id = base_family_id
                if rule.rule_family_id != expected_family_id:
                    raise ValueError(
                        f"rule {rule.rule_id} has inconsistent rule_family_id"
                    )
                if (
                    rule.outcome_criterion.to_dict()
                    != self.registered_outcome_criterion.to_dict()
                ):
                    raise ValueError(
                        f"rule {rule.rule_id} does not use the registered outcome "
                        "criterion"
                    )
            versions_by_family.setdefault(rule.rule_family_id, []).append(
                rule.rule_version
            )
            classified_ids = {
                "support": rule.supporting_episode_ids,
                "harmful_compliance": rule.harmful_compliance_episode_ids,
                "alternative_success": rule.alternative_success_episode_ids,
                "alternative_failure": rule.alternative_failure_episode_ids,
                "irrelevant": rule.irrelevant_episode_ids,
            }
            evidence = self._all_evidence_ids(rule)
            missing = sorted(evidence - known_episode_ids)
            if missing:
                raise ValueError(
                    f"rule {rule.rule_id} references missing episode IDs: {missing}"
                )
            for evidence_type, episode_ids in classified_ids.items():
                misclassified = [
                    episode_id
                    for episode_id in episode_ids
                    if self._classification(rule, episodes_by_id[episode_id])
                    != evidence_type
                ]
                if misclassified:
                    raise ValueError(
                        f"rule {rule.rule_id} has misclassified {evidence_type} "
                        f"evidence: {sorted(misclassified)}"
                    )
            expected_scores = self._scores_from_categories(
                support_count=len(rule.supporting_episode_ids),
                harmful_compliance_count=len(rule.harmful_compliance_episode_ids),
                alternative_success_count=len(rule.alternative_success_episode_ids),
            )
            if not math.isclose(rule.support_score, expected_scores[0], abs_tol=1e-12):
                raise ValueError(f"rule {rule.rule_id} has inconsistent support_score")
            if not math.isclose(
                rule.contradiction_score, expected_scores[1], abs_tol=1e-12
            ):
                raise ValueError(
                    f"rule {rule.rule_id} has inconsistent contradiction_score"
                )
            if rule.updated_at < rule.created_at:
                raise ValueError(f"rule {rule.rule_id} was updated before creation")
            evidence_episodes = [episodes_by_id[episode_id] for episode_id in evidence]
            post_counts = {
                evidence_type: sum(
                    episodes_by_id[episode_id].outcome_t > rule.created_at
                    for episode_id in episode_ids
                )
                for evidence_type, episode_ids in classified_ids.items()
            }
            expected_post_support = post_counts["support"]
            expected_post_contradiction = (
                post_counts["harmful_compliance"] + post_counts["alternative_success"]
            )
            expected_post_neutral = post_counts["alternative_failure"]
            expected_post_irrelevant = post_counts["irrelevant"]
            expected_post_total = (
                expected_post_support
                + expected_post_contradiction
                + expected_post_neutral
                + expected_post_irrelevant
            )
            if expected_post_total != rule.post_proposal_evidence_count:
                raise ValueError(
                    f"rule {rule.rule_id} has inconsistent post-proposal evidence count"
                )
            if expected_post_support != rule.post_proposal_support_count:
                raise ValueError(
                    f"rule {rule.rule_id} has inconsistent post-proposal support count"
                )
            if expected_post_contradiction != rule.post_proposal_contradiction_count:
                raise ValueError(
                    f"rule {rule.rule_id} has inconsistent post-proposal contradiction "
                    "count"
                )
            if expected_post_neutral != rule.post_proposal_neutral_count:
                raise ValueError(
                    f"rule {rule.rule_id} has inconsistent post-proposal neutral count"
                )
            if expected_post_irrelevant != rule.post_proposal_irrelevant_count:
                raise ValueError(
                    f"rule {rule.rule_id} has inconsistent post-proposal "
                    "irrelevant count"
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
            substantive_post_count = (
                rule.post_proposal_evidence_count
                - rule.post_proposal_irrelevant_count
            )
            if not rule.injected or substantive_post_count > 0:
                expected_confidence = expected_scores[3]
                if not math.isclose(
                    rule.confidence, expected_confidence, abs_tol=1e-12
                ):
                    raise ValueError(f"rule {rule.rule_id} has inconsistent confidence")
            missing_candidates = set(rule.candidate_ids) - set(self._candidates)
            if missing_candidates:
                raise ValueError(
                    f"rule {rule.rule_id} references missing candidates: "
                    f"{sorted(missing_candidates)}"
                )
            if not rule.injected:
                candidate = self._candidates[rule.candidate_ids[0]]
                if (
                    candidate.rule_key != rule.rule_key
                    or candidate.rule_family_id != rule.rule_family_id
                    or candidate.context_scope != rule.context_scope
                    or candidate.condition != rule.condition
                    or candidate.action_guidance != rule.action_guidance
                    or candidate.outcome_criterion != rule.outcome_criterion
                    or candidate.rationale != rule.rationale
                ):
                    raise ValueError(
                        f"rule {rule.rule_id} is not semantically bound to its "
                        "creation candidate"
                    )
                expected_initial_ids = {
                    evidence_type: [] for evidence_type in EVIDENCE_TYPES
                }
                unique_requested = tuple(
                    dict.fromkeys(candidate.supporting_episode_ids)
                )
                for episode_id in unique_requested:
                    episode = episodes_by_id.get(episode_id)
                    if episode is None or episode.outcome_t > rule.created_at:
                        continue
                    expected_initial_ids[
                        self._classification(candidate, episode)
                    ].append(episode_id)
                for episode in self.episodic_track.finalized_episodes:
                    if (
                        episode.outcome_t > rule.created_at
                        or episode.episode_id in unique_requested
                    ):
                        continue
                    classification = self._classification(candidate, episode)
                    if classification != "support":
                        expected_initial_ids[classification].append(
                            episode.episode_id
                        )
                actual_initial_ids = {
                    evidence_type: [
                        episode_id
                        for episode_id in episode_ids
                        if episodes_by_id[episode_id].outcome_t <= rule.created_at
                    ]
                    for evidence_type, episode_ids in classified_ids.items()
                }
                if actual_initial_ids != expected_initial_ids:
                    raise ValueError(
                        f"rule {rule.rule_id} creation evidence search is not "
                        "exhaustive and reproducible"
                    )
            if rule.status == "active" and not rule.injected:
                if (
                    rule.post_proposal_support_count < 1
                    or len(rule.supporting_episode_ids) < self.activation_min_support
                    or rule.margin < self.activation_min_margin
                    or rule.confidence < self.activation_confidence_threshold
                    or rule.activation_episode_id is None
                    or episodes_by_id[rule.activation_episode_id].outcome_t
                    <= rule.created_at
                ):
                    raise ValueError(
                        f"active rule {rule.rule_id} does not satisfy activation invariants"
                    )

        for candidate in self._candidates.values():
            if (
                candidate.outcome_criterion.to_dict()
                != self.registered_outcome_criterion.to_dict()
            ):
                raise ValueError(
                    f"candidate {candidate.candidate_id} does not use the registered "
                    "outcome criterion"
                )

        for family_id, versions in versions_by_family.items():
            if len(versions) != len(set(versions)):
                raise ValueError(f"rule family {family_id} has duplicate versions")
            if sorted(versions) != list(range(1, max(versions) + 1)):
                raise ValueError(f"rule family {family_id} has non-contiguous versions")
            live_versions = [
                rule.rule_version
                for rule in self._rules.values()
                if (
                    rule.rule_family_id == family_id
                    and rule.status in {"provisional", "active"}
                )
            ]
            if len(live_versions) > 1:
                raise ValueError(f"rule family {family_id} has multiple live versions")
        for rule in self._rules.values():
            lineage_ids = set(rule.derived_from_rule_ids)
            if rule.supersedes_rule_id is not None:
                lineage_ids.add(rule.supersedes_rule_id)
            missing_lineage = lineage_ids - set(self._rules)
            if missing_lineage:
                raise ValueError(
                    f"rule {rule.rule_id} references missing lineage: "
                    f"{sorted(missing_lineage)}"
                )
            for parent_id in lineage_ids:
                parent = self._rules[parent_id]
                if (
                    parent.rule_family_id != rule.rule_family_id
                    or parent.rule_version >= rule.rule_version
                ):
                    raise ValueError(
                        f"rule {rule.rule_id} has invalid lineage parent {parent_id}"
                    )
            if rule.rule_version > 1 and rule.supersedes_rule_id is None:
                raise ValueError(
                    f"rule {rule.rule_id} version > 1 lacks supersedes lineage"
                )
            if rule.rule_version == 1 and lineage_ids:
                raise ValueError(
                    f"rule {rule.rule_id} version 1 must not claim prior lineage"
                )
            if rule.supersedes_rule_id is not None:
                parent = self._rules[rule.supersedes_rule_id]
                if parent.rule_version != rule.rule_version - 1:
                    raise ValueError(
                        f"rule {rule.rule_id} does not supersede the immediately prior "
                        "version"
                    )
                if parent.status not in {"rejected", "retired"}:
                    raise ValueError(
                        f"rule {rule.rule_id} supersedes non-terminal rule "
                        f"{parent.rule_id}"
                    )
                if rule.supersedes_rule_id not in rule.derived_from_rule_ids:
                    raise ValueError(
                        f"rule {rule.rule_id} supersedes a rule not recorded in "
                        "derived_from_rule_ids"
                    )
                if rule.created_at <= parent.updated_at:
                    raise ValueError(
                        f"rule {rule.rule_id} was created before or at its terminal "
                        f"parent update"
                    )
                prior_family_rules = [
                    prior
                    for prior in self._rules.values()
                    if prior.rule_family_id == rule.rule_family_id
                    and prior.rule_version < rule.rule_version
                ]
                prior_evidence: set[str] = set()
                for prior in prior_family_rules:
                    prior_evidence.update(self._all_evidence_ids(prior))
                creation_candidate = self._candidates[rule.candidate_ids[0]]
                qualifying_new_support = [
                    episode_id
                    for episode_id in creation_candidate.supporting_episode_ids
                    if episode_id in rule.supporting_episode_ids
                    and episode_id not in prior_evidence
                    and episodes_by_id[episode_id].outcome_t > parent.updated_at
                    and episodes_by_id[episode_id].outcome_t <= rule.created_at
                    and self._classification(rule, episodes_by_id[episode_id])
                    == "support"
                ]
                if not qualifying_new_support:
                    raise ValueError(
                        f"rule {rule.rule_id} lacks new creation support after its "
                        "terminal parent update"
                    )

        lifecycle_status: dict[str, str] = {}
        creation_events: dict[str, list[RuleEvent]] = {}
        mutating_event_times: dict[str, list[int]] = {}
        previous_event_timestamp: Optional[int] = None
        for event_index, event in enumerate(self._events):
            if event.event_id in event_ids:
                raise ValueError(f"duplicate rule-event ID: {event.event_id}")
            event_ids.add(event.event_id)
            if (
                previous_event_timestamp is not None
                and event.timestamp < previous_event_timestamp
            ):
                raise ValueError("semantic event timestamps move backward")
            previous_event_timestamp = event.timestamp
            if event.event_type == "active_rule_retrieved" and event.rule_id is None:
                raise ValueError(
                    "active-rule retrieval event must reference one active rule"
                )
            if event.event_type == "active_rule_retrieval_empty":
                provenance = dict(event.provenance)
                limit = provenance.get("limit")
                state_hash = provenance.get("state_hash")
                if (
                    event.rule_id is not None
                    or event.candidate_id is not None
                    or event.from_status is not None
                    or event.to_status is not None
                    or event.episode_ids
                    or event.reason
                    != "no active rule matched within the retrieval limit"
                    or dict(event.metrics)
                    or set(provenance) != {"limit", "state_hash"}
                    or isinstance(limit, bool)
                    or not isinstance(limit, int)
                    or limit < 0
                    or not isinstance(state_hash, str)
                    or re.fullmatch(r"[0-9a-f]{64}", state_hash) is None
                ):
                    raise ValueError(
                        "active-rule empty retrieval marker has malformed exact "
                        "shape or provenance"
                    )
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
            future_evidence = [
                episode_id
                for episode_id in event.episode_ids
                if episodes_by_id[episode_id].outcome_t > event.timestamp
            ]
            if future_evidence:
                raise ValueError(
                    f"event {event.event_id} references evidence newer than its "
                    f"timestamp: {sorted(future_evidence)}"
                )
            if event.rule_id is not None:
                prior_status = lifecycle_status.get(event.rule_id)
                if prior_status is None:
                    if event.from_status is not None:
                        raise ValueError(
                            f"rule {event.rule_id} lifecycle does not start from null"
                        )
                elif event.from_status != prior_status:
                    raise ValueError(
                        f"rule {event.rule_id} lifecycle status is discontinuous"
                    )
                if event.to_status is not None:
                    lifecycle_status[event.rule_id] = event.to_status
                if event.event_type == "active_rule_retrieved" and (
                    event.candidate_id is not None
                    or event.from_status != "active"
                    or event.to_status != "active"
                    or event.episode_ids
                ):
                    raise ValueError(
                        f"rule {event.rule_id} has malformed active-rule retrieval "
                        "provenance"
                    )
                if event.event_type in {
                    "candidate_verified",
                    "candidate_rejected",
                    "experimental_rule_injected_active",
                }:
                    creation_events.setdefault(event.rule_id, []).append(event)
                if event.event_type in {
                    "candidate_verified",
                    "candidate_rejected",
                    "experimental_rule_injected_active",
                    "rule_activated",
                    "rule_retired",
                } or event.event_type.endswith("_evidence_added"):
                    mutating_event_times.setdefault(event.rule_id, []).append(
                        event.timestamp
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

        for rule in self._rules.values():
            if lifecycle_status.get(rule.rule_id) != rule.status:
                raise ValueError(
                    f"rule {rule.rule_id} final status does not match its lifecycle"
                )
            mutation_times = mutating_event_times.get(rule.rule_id, [])
            if not mutation_times or max(mutation_times) != rule.updated_at:
                raise ValueError(
                    f"rule {rule.rule_id} updated_at is not bound to its lifecycle"
                )
            rule_creation_events = creation_events.get(rule.rule_id, [])
            if len(rule_creation_events) != 1:
                raise ValueError(
                    f"rule {rule.rule_id} must have exactly one creation event"
                )
            creation_event = rule_creation_events[0]
            if creation_event.timestamp != rule.created_at:
                raise ValueError(
                    f"rule {rule.rule_id} creation timestamp is not event-bound"
                )
            self._validate_evidence_event_sequence(rule, episodes_by_id)
            if rule.injected:
                if (
                    creation_event.event_type != "experimental_rule_injected_active"
                    or creation_event.candidate_id is not None
                    or creation_event.from_status is not None
                    or creation_event.to_status != "active"
                    or dict(creation_event.provenance)
                    != dict(rule.injection_provenance or {})
                ):
                    raise ValueError(
                        f"injected rule {rule.rule_id} lacks exact creation provenance"
                    )
                continue

            candidate = self._candidates[rule.candidate_ids[0]]
            expected_event_type = (
                "candidate_rejected" if rule.status == "rejected" else "candidate_verified"
            )
            expected_initial_status = (
                "rejected" if rule.status == "rejected" else "provisional"
            )
            provenance = creation_event.provenance
            expected_creation_episode_ids = tuple(
                episode_id
                for episode_id in dict.fromkeys(candidate.supporting_episode_ids)
                if (
                    episode_id in episodes_by_id
                    and episodes_by_id[episode_id].outcome_t <= rule.created_at
                )
            )
            if (
                creation_event.event_type != expected_event_type
                or creation_event.candidate_id != candidate.candidate_id
                or creation_event.from_status is not None
                or creation_event.to_status != expected_initial_status
                or creation_event.episode_ids != expected_creation_episode_ids
                or provenance.get("generator_id") != candidate.generator_id
                or provenance.get("raw_response_hash") != candidate.raw_response_hash
                or provenance.get("searched_counterevidence") is not True
                or provenance.get("requested_support_ids")
                != list(candidate.supporting_episode_ids)
                or provenance.get("registered_outcome_criterion")
                != self.registered_outcome_criterion.to_dict()
                or provenance.get("migration_notes")
                != list(candidate.migration_notes)
                or provenance.get("supersedes_rule_id")
                != rule.supersedes_rule_id
                or provenance.get("derived_from_rule_ids")
                != list(rule.derived_from_rule_ids)
            ):
                raise ValueError(
                    f"rule {rule.rule_id} creation candidate/raw provenance is not exact"
                )
            if rule.activation_episode_id is not None:
                activation_events = [
                    event
                    for event in self._events
                    if event.rule_id == rule.rule_id
                    and event.event_type == "rule_activated"
                ]
                if (
                    len(activation_events) != 1
                    or activation_events[0].episode_ids
                    != (rule.activation_episode_id,)
                ):
                    raise ValueError(
                        f"rule {rule.rule_id} activation episode is not event-bound"
                    )

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
                "registered_outcome_criterion": (
                    self.registered_outcome_criterion.to_dict()
                ),
                "evidence_weights": dict(self.evidence_weights),
            },
            "candidates": [
                candidate.to_dict() for candidate in self._candidates.values()
            ],
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
        if not isinstance(value, Mapping):
            raise ValueError("M3 snapshot must be an object")
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "unsupported M3 schema version; v1 snapshots are fail-closed "
                "because legacy action semantics and evidence categories cannot be "
                "silently reinterpreted"
            )
        _strict_keys(
            value,
            {"schema_version", "config", "candidates", "rules", "events"},
            "M3 snapshot",
        )
        if not isinstance(value["config"], Mapping):
            raise ValueError("M3 config must be an object")
        config = dict(value["config"])
        _strict_keys(
            config,
            {
                "allowed_condition_fields",
                "allowed_action_targets",
                "min_candidate_support",
                "activation_min_support",
                "activation_min_margin",
                "activation_confidence_threshold",
                "proposal_confidence_floor",
                "retirement_patience",
                "retirement_confidence_threshold",
                "registered_outcome_criterion",
                "evidence_weights",
            },
            "M3 config",
        )
        for name in ("candidates", "rules", "events"):
            if not isinstance(value[name], list):
                raise ValueError(f"M3 {name} must be an array")
        track = cls(episodic_track, **config)
        for candidate_value in value["candidates"]:
            candidate = RuleCandidate.from_dict(candidate_value)
            if candidate.candidate_id in track._candidates:
                raise ValueError(f"duplicate candidate ID: {candidate.candidate_id}")
            track._candidates[candidate.candidate_id] = candidate
        for rule_value in value["rules"]:
            rule = VerifiedRule.from_dict(rule_value)
            if rule.rule_id in track._rules:
                raise ValueError(f"duplicate rule ID: {rule.rule_id}")
            track._rules[rule.rule_id] = rule
        track._events = [
            RuleEvent.from_dict(event_value) for event_value in value["events"]
        ]
        track.validate_referential_integrity()
        return track


# Descriptive alias for callers that name the module rather than the track.
M3VerifiedSemanticMemory = VerifiedSemanticRuleTrack
