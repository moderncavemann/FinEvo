"""Artifact schemas and persistence for :mod:`verified_memory.runner`."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import json

from .artifacts import (
    ArtifactValidationError,
    JsonField,
    JsonlStreamSchema,
    RunArtifactWriter,
    verify_manifest,
)
from .actions import parse_direct_action
from .foundation_adapter import locate_component
from .m1_context import CausalContextRouter, ContextPacket
from .m0_utility import (
    PostStepSnapshot,
    PreStepSnapshot,
    SCHEMA_VERSION as M0_SCHEMA_VERSION,
    UtilityConfig,
    build_budget_ledger_row,
)
from .m2_episodic import (
    EpisodeRecord,
    EvidenceLinkedEpisodicTrack,
    SCHEMA_VERSION as M2_SCHEMA_VERSION,
    _utility_statistics,
)
from .m3_semantic import (
    EVENT_SCHEMA_VERSION as M3_EVENT_SCHEMA_VERSION,
    RULE_SCHEMA_VERSION as M3_RULE_SCHEMA_VERSION,
    RuleEvent,
    VerifiedRule,
    VerifiedSemanticRuleTrack,
)
from .prompts import (
    PREVIOUS_PROMPT_SCHEMA_VERSION,
    PROMPT_SCHEMA_VERSION,
    compose_decision_prompt,
)
from .runner import (
    CONTEXT_FEATURES,
    ERROR_RULE_INJECTION_SCHEMA_VERSION,
    RUNNER_SCHEMA_VERSION,
    SHOCK_EVENT_SCHEMA_VERSION,
    VerifiedRunResult,
    _semantic_parse_mode,
    serialized_has_sealed_observed_p95_authority,
)
from .system import SYSTEM_SCHEMA_VERSION


LEGACY_RUNNER_SCHEMA_VERSION = "verified-simulation-runner-v1"
PREVIOUS_RUNNER_SCHEMA_VERSION = "verified-simulation-runner-v2"
_MODERN_RUNNER_SCHEMA_VERSIONS = frozenset(
    {PREVIOUS_RUNNER_SCHEMA_VERSION, RUNNER_SCHEMA_VERSION}
)
_RULE_STATUSES = ("provisional", "active", "rejected", "retired")
_PARSE_MODES = (
    "exact_json",
    "fenced_recovery",
    "substring_recovery",
    "parse_failure",
)

_CURRENT_STREAM_SCHEMA_VERSIONS = {
    "actions": RUNNER_SCHEMA_VERSION,
    "api_usage": RUNNER_SCHEMA_VERSION,
    "context_trace": SYSTEM_SCHEMA_VERSION,
    "decision_snapshots": RUNNER_SCHEMA_VERSION,
    "episodes": M2_SCHEMA_VERSION,
    "utility_ledger": M0_SCHEMA_VERSION,
    "macro_steps": RUNNER_SCHEMA_VERSION,
    "semantic_proposals": RUNNER_SCHEMA_VERSION,
    "semantic_rule_events": M3_EVENT_SCHEMA_VERSION,
    "semantic_rules": M3_RULE_SCHEMA_VERSION,
    "shock_events": SHOCK_EVENT_SCHEMA_VERSION,
    "error_rule_injections": ERROR_RULE_INJECTION_SCHEMA_VERSION,
    "errors": RUNNER_SCHEMA_VERSION,
}
_PREVIOUS_STREAM_SCHEMA_VERSIONS = {
    "actions": PREVIOUS_RUNNER_SCHEMA_VERSION,
    "api_usage": PREVIOUS_RUNNER_SCHEMA_VERSION,
    "context_trace": SYSTEM_SCHEMA_VERSION,
    "decision_snapshots": PREVIOUS_RUNNER_SCHEMA_VERSION,
    "episodes": M2_SCHEMA_VERSION,
    "utility_ledger": M0_SCHEMA_VERSION,
    "macro_steps": PREVIOUS_RUNNER_SCHEMA_VERSION,
    "semantic_proposals": PREVIOUS_RUNNER_SCHEMA_VERSION,
    "semantic_rule_events": M3_EVENT_SCHEMA_VERSION,
    "semantic_rules": M3_RULE_SCHEMA_VERSION,
    "errors": PREVIOUS_RUNNER_SCHEMA_VERSION,
}
_LEGACY_STREAM_SCHEMA_VERSIONS = {
    "actions": LEGACY_RUNNER_SCHEMA_VERSION,
    "api_usage": LEGACY_RUNNER_SCHEMA_VERSION,
    "context_trace": "verified-dual-track-system-v1",
    "decision_snapshots": LEGACY_RUNNER_SCHEMA_VERSION,
    "episodes": "m2-episodic-v1",
    "utility_ledger": M0_SCHEMA_VERSION,
    "macro_steps": LEGACY_RUNNER_SCHEMA_VERSION,
    "semantic_proposals": LEGACY_RUNNER_SCHEMA_VERSION,
    "semantic_rule_events": "m3-rule-event-v1",
    "semantic_rules": "m3-verified-rule-v1",
    "errors": LEGACY_RUNNER_SCHEMA_VERSION,
}


def _schema(
    name: str,
    fields: tuple[JsonField, ...],
    *,
    required: bool = True,
) -> JsonlStreamSchema:
    return JsonlStreamSchema(
        name=name,
        relative_path=f"streams/{name}.jsonl",
        fields=fields,
        required=required,
        min_records=1 if required else 0,
        allow_extra_fields=True,
    )


def verified_run_schemas(
    *,
    semantic_required: bool,
    run_schema_version: str = RUNNER_SCHEMA_VERSION,
) -> tuple[JsonlStreamSchema, ...]:
    """Return the complete declared stream contract for a verified run."""

    if run_schema_version not in {
        LEGACY_RUNNER_SCHEMA_VERSION,
        PREVIOUS_RUNNER_SCHEMA_VERSION,
        RUNNER_SCHEMA_VERSION,
    }:
        raise ValueError(f"unsupported runner schema {run_schema_version!r}")

    core = (
        _schema(
            "actions",
            (
                JsonField("schema_version", "string"),
                JsonField("decision_t", "integer"),
                JsonField("agent_id", "integer"),
                JsonField("decision", "object"),
            ),
        ),
        _schema(
            "api_usage",
            (
                JsonField("schema_version", "string"),
                JsonField("call_kind", "string"),
                JsonField("decision_t", "integer"),
                JsonField("agent_id", "integer"),
                JsonField("usage", "object"),
            ),
        ),
        _schema(
            "context_trace",
            (
                JsonField("schema_version", "string"),
                JsonField("decision_t", "integer"),
                JsonField("agent_id", "integer"),
                JsonField("context_hash", "string"),
            ),
        ),
        _schema(
            "decision_snapshots",
            (
                JsonField("schema_version", "string"),
                JsonField("decision_t", "integer"),
                JsonField("agent_id", "integer"),
                JsonField("environment_state_hash", "string"),
                JsonField("prompt_schema_version", "string"),
                JsonField("base_prompt_hash", "string"),
                JsonField("memory_hash", "string"),
            ),
        ),
        _schema(
            "episodes",
            (
                JsonField("schema_version", "string"),
                JsonField("episode_id", "string"),
                JsonField("decision_t", "integer"),
                JsonField("outcome_t", "integer"),
                JsonField("record_hash", "string"),
            ),
        ),
        _schema(
            "utility_ledger",
            (
                JsonField("schema_version", "string"),
                JsonField("period", "integer"),
                JsonField("agent_id", "string"),
                JsonField("flow_utility", "number"),
                JsonField("budget_residual", "number"),
            ),
        ),
        _schema(
            "macro_steps",
            (
                JsonField("schema_version", "string"),
                JsonField("decision_t", "integer"),
                JsonField("outcome_t", "integer"),
                JsonField("low_labor_rate", "number"),
            ),
        ),
        _schema(
            "summary",
            (
                JsonField("schema_version", "string"),
                JsonField("run_id", "string"),
                JsonField("result_complete", "boolean"),
                JsonField("validation", "object"),
            ),
        ),
    )
    semantic = (
        _schema(
            "semantic_proposals",
            (
                JsonField("schema_version", "string"),
                JsonField("current_t", "integer"),
                JsonField("agent_id", "integer"),
            ),
            required=False,
        ),
        _schema(
            "semantic_rule_events",
            (
                JsonField("schema_version", "string"),
                JsonField("event_id", "string"),
                JsonField("agent_id", "integer"),
            ),
            required=False,
        ),
        _schema(
            "semantic_rules",
            (
                JsonField("schema_version", "string"),
                JsonField("rule_id", "string"),
                JsonField("agent_id", "integer"),
                JsonField("status", "string"),
            ),
            required=False,
        ),
        _schema(
            "errors",
            (JsonField("error_type", "string", required=False, nullable=True),),
            required=False,
        ),
    )
    pilot = ()
    if run_schema_version == RUNNER_SCHEMA_VERSION:
        pilot = (
            _schema(
                "shock_events",
                (
                    JsonField("schema_version", "string"),
                    JsonField("decision_t", "integer"),
                    JsonField("phase", "string"),
                    JsonField("interest_rate", "number"),
                    JsonField("applied_before_prompt", "boolean"),
                    JsonField("applied_before_step", "boolean"),
                ),
                required=False,
            ),
            _schema(
                "error_rule_injections",
                (
                    JsonField("schema_version", "string"),
                    JsonField("decision_t", "integer"),
                    JsonField("agent_id", "integer"),
                    JsonField("mode", "string"),
                    JsonField("semantic_policy", "string"),
                    JsonField("fixed_rule_hash", "string"),
                    JsonField("verifier_bypassed", "boolean"),
                    JsonField("rule_id", "string"),
                    JsonField("rule_status", "string"),
                ),
                required=False,
            ),
        )
    return core + semantic + pilot


def _contract_error(message: str) -> ArtifactValidationError:
    return ArtifactValidationError(f"verified run artifact contract: {message}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _contract_error(f"{label} must be an object")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _contract_error(f"{label} must be a non-negative integer")
    return value


def _reconcile_count(label: str, declared: Any, actual: int) -> None:
    declared_count = _nonnegative_int(declared, label)
    if declared_count != actual:
        raise _contract_error(
            f"{label} mismatch: summary declares {declared_count}, stream has {actual}"
        )


def _list_field(row: Mapping[str, Any], field: str, stream: str) -> list[Any]:
    value = row.get(field)
    if not isinstance(value, list):
        raise _contract_error(f"{stream}.{field} must be an array")
    return value


def _reconcile_identity_grid(
    stream_name: str,
    rows: tuple[Mapping[str, Any], ...],
    *,
    time_field: str,
    num_agents: int,
    completed_periods: int,
    string_agent_ids: bool = False,
) -> None:
    expected = Counter(
        (
            period,
            str(agent_id) if string_agent_ids else agent_id,
        )
        for period in range(completed_periods)
        for agent_id in range(num_agents)
    )
    actual = Counter((row.get(time_field), row.get("agent_id")) for row in rows)
    if actual != expected:
        missing = list((expected - actual).elements())[:5]
        unexpected = list((actual - expected).elements())[:5]
        raise _contract_error(
            f"{stream_name} identity grid mismatch; "
            f"missing={missing}, unexpected_or_duplicate={unexpected}"
        )


def _api_identity(
    *,
    call_kind: Any,
    decision_t: Any,
    agent_id: Any,
    model: Any,
    usage: Any,
) -> tuple[Any, ...]:
    return (
        call_kind,
        decision_t,
        str(agent_id),
        model,
        json.dumps(usage, sort_keys=True, separators=(",", ":"), allow_nan=False),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _m3_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _gini(values: list[float]) -> float:
    ordered = sorted(max(float(value), 0.0) for value in values)
    if not ordered or sum(ordered) == 0:
        return 0.0
    count = len(ordered)
    total = sum(ordered)
    weighted = sum(
        (index + 1) * value for index, value in enumerate(ordered)
    )
    return 2 * weighted / (count * total) - (count + 1) / count


def _finite_rate(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _contract_error(f"{field} must be a finite numeric rate")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise _contract_error(f"{field} must lie in [0, 1]")
    return numeric


def _validate_rate_availability(
    state: Mapping[str, Any],
    *,
    available: bool,
    label: str,
) -> None:
    """Keep bootstrap missingness distinct from an observed zero rate."""

    expected_mask = 1.0 if available else 0.0
    for field in ("low_labor_rate", "unemployment_rate"):
        mask_field = f"{field}_available"
        mask = _finite_rate(state.get(mask_field), f"{label}.{mask_field}")
        if mask != expected_mask:
            raise _contract_error(
                f"{label}.{mask_field} must be {expected_mask:g}"
            )
        if available:
            if field not in state:
                raise _contract_error(
                    f"{label}.{field} is required when its availability mask is 1"
                )
            _finite_rate(state[field], f"{label}.{field}")
        elif field in state:
            raise _contract_error(
                f"{label}.{field} must be absent when its availability mask is 0"
            )


def _validate_current_semantic_rows(
    config: Mapping[str, Any],
    episode_objects: Mapping[str, EpisodeRecord],
    parsed_rules: Mapping[tuple[Any, Any], VerifiedRule],
    events_by_agent: Mapping[Any, list[RuleEvent]],
) -> None:
    effective = config.get("effective_semantic_verifier")
    if parsed_rules and not isinstance(effective, Mapping):
        raise _contract_error("semantic rules require effective verifier config")
    if not isinstance(effective, Mapping):
        effective = {}
    weights = effective.get("evidence_weights", {})
    if parsed_rules and not isinstance(weights, Mapping):
        raise _contract_error("effective semantic evidence_weights must be an object")

    versions_by_family: dict[tuple[Any, str], list[int]] = {}
    for (agent_id, _), rule in parsed_rules.items():
        rule_key_payload = {
            "context_scope": rule.context_scope.to_dict(),
            "condition": rule.condition.to_dict(),
            "action_guidance": rule.action_guidance.to_dict(),
            "outcome_criterion": rule.outcome_criterion.to_dict(),
        }
        expected_rule_key = f"rule-{_m3_hash(rule_key_payload)[:20]}"
        if rule.rule_key != expected_rule_key:
            raise _contract_error(f"rule {rule.rule_id} has inconsistent rule_key")
        expected_rule_id = f"{rule.rule_family_id}:v{rule.rule_version}"
        if rule.rule_id != expected_rule_id:
            raise _contract_error(f"rule {rule.rule_id} has inconsistent rule_id")
        # Keep the artifact verifier coupled to the authoritative family
        # canonicalizer.  In v2 a family intentionally fixes categorical
        # semantics (including predicate/operator shape) while leaving numeric
        # thresholds/tolerances revisionable.
        base_family_id = VerifiedSemanticRuleTrack._rule_family_id(
            rule.context_scope,
            rule.condition,
            rule.action_guidance,
            rule.outcome_criterion,
        )
        if rule.injected:
            provenance = rule.injection_provenance
            if not isinstance(provenance, Mapping) or not isinstance(
                provenance.get("injection_id"), str
            ):
                raise _contract_error(
                    f"injected rule {rule.rule_id} lacks injection provenance"
                )
            expected_family_id = (
                f"{base_family_id}:injected:"
                f"{_m3_hash(provenance['injection_id'])[:12]}"
            )
            if rule.candidate_ids:
                raise _contract_error("injected rules may not claim candidate IDs")
        else:
            expected_family_id = base_family_id
            if rule.outcome_criterion.to_dict() != config.get(
                "registered_outcome_criterion"
            ):
                raise _contract_error(
                    f"rule {rule.rule_id} does not use registered outcome criterion"
                )
        if rule.rule_family_id != expected_family_id:
            raise _contract_error(
                f"rule {rule.rule_id} has inconsistent rule_family_id"
            )
        versions_by_family.setdefault((agent_id, rule.rule_family_id), []).append(
            rule.rule_version
        )

        classified_ids = {
            "support": rule.supporting_episode_ids,
            "harmful_compliance": rule.harmful_compliance_episode_ids,
            "alternative_success": rule.alternative_success_episode_ids,
            "alternative_failure": rule.alternative_failure_episode_ids,
            "irrelevant": rule.irrelevant_episode_ids,
        }
        evidence_ids = set().union(*(set(values) for values in classified_ids.values()))
        for evidence_type, episode_ids in classified_ids.items():
            for episode_id in episode_ids:
                episode = episode_objects.get(episode_id)
                if episode is None or episode.agent_id != agent_id:
                    raise _contract_error(
                        f"rule {rule.rule_id} references missing/cross-agent evidence"
                    )
                if (
                    not rule.context_scope.matches(episode.pre_state)
                    or not rule.condition.matches(episode.pre_state)
                ):
                    classification = "irrelevant"
                else:
                    compliant = rule.action_guidance.is_consistent(
                        episode.executed_action
                    )
                    successful = rule.outcome_criterion.passes(episode)
                    if compliant and successful:
                        classification = "support"
                    elif compliant:
                        classification = "harmful_compliance"
                    elif successful:
                        classification = "alternative_success"
                    else:
                        classification = "alternative_failure"
                if classification != evidence_type:
                    raise _contract_error(
                        f"rule {rule.rule_id} has misclassified {evidence_type} evidence"
                    )
        if set(rule.contradicting_episode_ids) != (
            set(rule.harmful_compliance_episode_ids)
            | set(rule.alternative_success_episode_ids)
        ):
            raise _contract_error(
                f"rule {rule.rule_id} has inconsistent contradicting evidence"
            )
        support_score = len(rule.supporting_episode_ids) * float(
            weights.get("support", 0.0)
        )
        contradiction_score = len(rule.harmful_compliance_episode_ids) * float(
            weights.get("harmful_compliance", 0.0)
        ) + len(rule.alternative_success_episode_ids) * float(
            weights.get("alternative_success", 0.0)
        )
        expected_confidence = (support_score + 1.0) / (
            support_score + contradiction_score + 2.0
        )
        if (
            not math.isclose(rule.support_score, support_score, abs_tol=1e-12)
            or not math.isclose(
                rule.contradiction_score, contradiction_score, abs_tol=1e-12
            )
            or not math.isclose(
                rule.margin, support_score - contradiction_score, abs_tol=1e-12
            )
        ):
            raise _contract_error(f"rule {rule.rule_id} has inconsistent evidence scores")
        if rule.updated_at < rule.created_at:
            raise _contract_error(f"rule {rule.rule_id} was updated before creation")
        if any(episode_objects[item].outcome_t > rule.updated_at for item in evidence_ids):
            raise _contract_error(f"rule {rule.rule_id} contains future evidence")
        post_counts = {
            evidence_type: sum(
                episode_objects[item].outcome_t > rule.created_at
                for item in episode_ids
            )
            for evidence_type, episode_ids in classified_ids.items()
        }
        expected_post_support = post_counts["support"]
        expected_post_contradiction = (
            post_counts["harmful_compliance"] + post_counts["alternative_success"]
        )
        expected_post_neutral = post_counts["alternative_failure"]
        expected_post_irrelevant = post_counts["irrelevant"]
        if (
            rule.post_proposal_support_count != expected_post_support
            or rule.post_proposal_contradiction_count != expected_post_contradiction
            or rule.post_proposal_neutral_count != expected_post_neutral
            or rule.post_proposal_irrelevant_count != expected_post_irrelevant
            or rule.post_proposal_evidence_count
            != (
                expected_post_support
                + expected_post_contradiction
                + expected_post_neutral
                + expected_post_irrelevant
            )
        ):
            raise _contract_error(
                f"rule {rule.rule_id} has inconsistent post-proposal evidence counts"
            )
        substantive_post_count = (
            rule.post_proposal_evidence_count
            - rule.post_proposal_irrelevant_count
        )
        if (
            not rule.injected or substantive_post_count > 0
        ) and not math.isclose(rule.confidence, expected_confidence, abs_tol=1e-12):
            raise _contract_error(f"rule {rule.rule_id} has inconsistent confidence")
        if rule.status == "active" and not rule.injected:
            activation_id = rule.activation_episode_id
            if (
                rule.post_proposal_support_count < 1
                or len(rule.supporting_episode_ids)
                < int(effective.get("activation_min_support", 0))
                or rule.margin < float(effective.get("activation_min_margin", 0.0))
                or rule.confidence
                < float(effective.get("activation_confidence_threshold", 0.0))
                or activation_id is None
                or episode_objects[activation_id].outcome_t <= rule.created_at
            ):
                raise _contract_error(
                    f"active rule {rule.rule_id} fails activation invariants"
                )

    for family_identity, versions in versions_by_family.items():
        if sorted(versions) != list(range(1, max(versions) + 1)):
            raise _contract_error(
                f"rule family {family_identity!r} has non-contiguous versions"
            )
        live = [
            rule
            for (agent_id, _), rule in parsed_rules.items()
            if (agent_id, rule.rule_family_id) == family_identity
            and rule.status in {"provisional", "active"}
        ]
        if len(live) > 1:
            raise _contract_error(
                f"rule family {family_identity!r} has multiple live versions"
            )
    creation_events_by_rule: dict[tuple[Any, str], list[RuleEvent]] = {}
    activation_events_by_rule: dict[tuple[Any, str], list[RuleEvent]] = {}
    mutating_event_times_by_rule: dict[tuple[Any, str], list[int]] = {}
    for agent_id, events in events_by_agent.items():
        candidate_ids_by_rule: dict[str, set[str]] = {}
        seen_event_ids: set[str] = set()
        previous_timestamp = -1
        for index, event in enumerate(events):
            if event.timestamp < previous_timestamp:
                raise _contract_error(
                    f"semantic event timestamps move backward for agent {agent_id}"
                )
            previous_timestamp = event.timestamp
            if event.event_id in seen_event_ids:
                raise _contract_error(f"duplicate semantic event ID for agent {agent_id}")
            seen_event_ids.add(event.event_id)
            if event.rule_id is not None and (agent_id, event.rule_id) not in parsed_rules:
                raise _contract_error("semantic event references missing same-agent rule")
            for episode_id in event.episode_ids:
                episode = episode_objects.get(episode_id)
                if (
                    episode is None
                    or episode.agent_id != agent_id
                    or episode.outcome_t > event.timestamp
                ):
                    raise _contract_error(
                        "semantic event references missing, cross-agent, or future evidence"
                    )
            core = {
                "index": index,
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
            expected_event_id = f"rle-{index:06d}-{_m3_hash(core)[:12]}"
            if event.event_id != expected_event_id:
                raise _contract_error(
                    f"semantic event ledger hash mismatch for agent {agent_id}"
                )
            if event.rule_id is not None and event.candidate_id is not None:
                candidate_ids_by_rule.setdefault(event.rule_id, set()).add(
                    event.candidate_id
                )
            if event.event_type in {"candidate_verified", "candidate_rejected"}:
                if event.rule_id is None:
                    raise _contract_error(
                        "semantic creation event must reference a rule"
                    )
                creation_events_by_rule.setdefault(
                    (agent_id, event.rule_id), []
                ).append(event)
            if event.event_type in {
                "rule_activated",
                "experimental_rule_injected_active",
            }:
                if event.rule_id is None:
                    raise _contract_error(
                        "semantic activation event must reference a rule"
                    )
                activation_events_by_rule.setdefault(
                    (agent_id, event.rule_id), []
                ).append(event)
            if event.rule_id is not None and (
                event.event_type
                in {
                    "candidate_verified",
                    "candidate_rejected",
                    "experimental_rule_injected_active",
                    "rule_activated",
                    "rule_retired",
                }
                or event.event_type.endswith("_evidence_added")
            ):
                mutating_event_times_by_rule.setdefault(
                    (agent_id, event.rule_id), []
                ).append(event.timestamp)
            if event.event_type == "active_rule_retrieved":
                activation_events = activation_events_by_rule.get(
                    (agent_id, event.rule_id or ""), []
                )
                if (
                    len(activation_events) != 1
                    or activation_events[0].timestamp > event.timestamp
                ):
                    raise _contract_error(
                        "active rule was retrieved before its unique activation event"
                    )
        for (rule_agent, _), rule in parsed_rules.items():
            if rule_agent != agent_id:
                continue
            if not set(rule.candidate_ids).issubset(
                candidate_ids_by_rule.get(rule.rule_id, set())
            ):
                raise _contract_error(
                    f"rule {rule.rule_id} candidate IDs lack event provenance"
                )
        agent_rules = [
            rule
            for (rule_agent, _), rule in parsed_rules.items()
            if rule_agent == agent_id
        ]
        if agent_rules:
            replay_validator = VerifiedSemanticRuleTrack(
                EvidenceLinkedEpisodicTrack(
                    run_id=str(config.get("run_id")),
                    seed=int(config.get("seed")),
                    agent_id=int(agent_id),
                    prompt_capacity=int(
                        config.get("episodic_prompt_capacity", 24)
                    ),
                ),
                **dict(effective),
            )
            replay_validator._events = list(events)
            for rule in agent_rules:
                try:
                    replay_validator._validate_evidence_event_sequence(
                        rule,
                        episode_objects,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise _contract_error(
                        f"rule {rule.rule_id} evidence lifecycle does not reproduce: "
                        f"{exc}"
                    ) from exc

    family_rules_by_identity: dict[tuple[Any, str], list[VerifiedRule]] = {}
    for (agent_id, _), rule in parsed_rules.items():
        family_rules_by_identity.setdefault(
            (agent_id, rule.rule_family_id), []
        ).append(rule)

    for family_identity, family_rules in family_rules_by_identity.items():
        family_rules.sort(key=lambda item: item.rule_version)
        prior_evidence_ids: set[str] = set()
        for index, rule in enumerate(family_rules):
            agent_id = family_identity[0]
            lineage_ids = set(rule.derived_from_rule_ids)
            if rule.supersedes_rule_id is not None:
                lineage_ids.add(rule.supersedes_rule_id)
            for parent_id in lineage_ids:
                parent = parsed_rules.get((agent_id, parent_id))
                if (
                    parent is None
                    or parent.rule_family_id != rule.rule_family_id
                    or parent.rule_version >= rule.rule_version
                ):
                    raise _contract_error(f"rule {rule.rule_id} has invalid lineage")

            creation_events = creation_events_by_rule.get(
                (agent_id, rule.rule_id), []
            )
            if rule.injected:
                if creation_events:
                    raise _contract_error(
                        f"injected rule {rule.rule_id} may not claim candidate creation"
                    )
                activation_events = activation_events_by_rule.get(
                    (agent_id, rule.rule_id), []
                )
                if (
                    len(activation_events) != 1
                    or activation_events[0].event_type
                    != "experimental_rule_injected_active"
                    or activation_events[0].timestamp != rule.created_at
                    or activation_events[0].from_status is not None
                    or activation_events[0].to_status != "active"
                    or activation_events[0].candidate_id is not None
                    or activation_events[0].episode_ids
                    or dict(activation_events[0].provenance)
                    != dict(rule.injection_provenance or {})
                ):
                    raise _contract_error(
                        f"injected rule {rule.rule_id} lacks its unique causal "
                        "activation event"
                    )
            else:
                if len(rule.candidate_ids) != 1:
                    raise _contract_error(
                        f"non-injected rule {rule.rule_id} must bind exactly one "
                        "creation candidate ID"
                    )
                if len(creation_events) != 1:
                    raise _contract_error(
                        f"non-injected rule {rule.rule_id} must have exactly one "
                        "candidate creation event"
                    )
                creation_event = creation_events[0]
                expected_creation_type = (
                    "candidate_rejected"
                    if creation_event.to_status == "rejected"
                    else "candidate_verified"
                )
                if (
                    creation_event.candidate_id != rule.candidate_ids[0]
                    or creation_event.timestamp != rule.created_at
                    or creation_event.from_status is not None
                    or creation_event.to_status not in {"provisional", "rejected"}
                    or creation_event.event_type != expected_creation_type
                    or creation_event.provenance.get("searched_counterevidence")
                    is not True
                    or creation_event.provenance.get(
                        "registered_outcome_criterion"
                    )
                    != config.get("registered_outcome_criterion")
                    or creation_event.provenance.get("supersedes_rule_id")
                    != rule.supersedes_rule_id
                    or creation_event.provenance.get("derived_from_rule_ids")
                    != list(rule.derived_from_rule_ids)
                ):
                    raise _contract_error(
                        f"rule {rule.rule_id} creation candidate/event provenance "
                        "is inconsistent"
                    )
                activation_events = activation_events_by_rule.get(
                    (agent_id, rule.rule_id), []
                )
                if rule.activation_episode_id is None:
                    if activation_events:
                        raise _contract_error(
                            f"rule {rule.rule_id} claims an activation event without "
                            "an activation episode"
                        )
                elif (
                    len(activation_events) != 1
                    or activation_events[0].event_type != "rule_activated"
                    or activation_events[0].from_status != "provisional"
                    or activation_events[0].to_status != "active"
                    or activation_events[0].candidate_id is not None
                    or activation_events[0].episode_ids
                    != (rule.activation_episode_id,)
                    or activation_events[0].timestamp
                    < episode_objects[rule.activation_episode_id].outcome_t
                ):
                    raise _contract_error(
                        f"rule {rule.rule_id} activation episode/event provenance "
                        "is inconsistent"
                    )

            mutation_times = mutating_event_times_by_rule.get(
                (agent_id, rule.rule_id), []
            )
            if not mutation_times or max(mutation_times) != rule.updated_at:
                raise _contract_error(
                    f"rule {rule.rule_id} updated_at is not bound to its lifecycle"
                )

            if index == 0:
                if rule.rule_version != 1 or lineage_ids:
                    raise _contract_error(
                        f"rule {rule.rule_id} v1 claims prior lineage"
                    )
            else:
                parent = family_rules[index - 1]
                if (
                    rule.supersedes_rule_id != parent.rule_id
                    or tuple(rule.derived_from_rule_ids) != (parent.rule_id,)
                ):
                    raise _contract_error(
                        f"rule {rule.rule_id} must derive from and supersede its "
                        "immediate family parent"
                    )
                if parent.status not in {"rejected", "retired"}:
                    raise _contract_error(
                        f"rule {rule.rule_id} parent must be terminal before revision"
                    )
                if rule.created_at <= parent.updated_at:
                    raise _contract_error(
                        f"rule {rule.rule_id} must be created strictly after its "
                        "terminal parent update"
                    )
                creation_event = creation_events[0]
                requested_support_ids = creation_event.provenance.get(
                    "requested_support_ids"
                )
                if not isinstance(requested_support_ids, list):
                    raise _contract_error(
                        f"rule {rule.rule_id} creation event lacks requested support "
                        "provenance"
                    )
                qualifying_new_support = {
                    episode_id
                    for episode_id in rule.supporting_episode_ids
                    if (
                        episode_id not in prior_evidence_ids
                        and episode_id in requested_support_ids
                        and episode_objects[episode_id].outcome_t > parent.updated_at
                        and episode_objects[episode_id].outcome_t <= rule.created_at
                    )
                }
                if not qualifying_new_support:
                    raise _contract_error(
                        f"rule {rule.rule_id} revision lacks genuinely new support "
                        "observed after its terminal parent update"
                    )

            prior_evidence_ids.update(rule.supporting_episode_ids)
            prior_evidence_ids.update(rule.harmful_compliance_episode_ids)
            prior_evidence_ids.update(rule.alternative_success_episode_ids)
            prior_evidence_ids.update(rule.alternative_failure_episode_ids)
            prior_evidence_ids.update(rule.irrelevant_episode_ids)


def _validate_current_semantic_proposals(
    config: Mapping[str, Any],
    records: Mapping[str, tuple[Mapping[str, Any], ...]],
    events_by_agent: Mapping[Any, list[RuleEvent]],
    parsed_rules: Mapping[tuple[Any, Any], VerifiedRule],
) -> None:
    proposals = records.get("semantic_proposals", ())
    if not proposals:
        if any(not rule.injected for rule in parsed_rules.values()):
            raise _contract_error(
                "non-injected semantic rules require reparsable creation proposals"
            )
        return
    effective = config.get("effective_semantic_verifier")
    if not isinstance(effective, Mapping):
        raise _contract_error("semantic proposals require effective verifier config")
    semantic_api = {
        (row.get("decision_t"), row.get("agent_id")): row
        for row in records.get("api_usage", ())
        if row.get("call_kind") == "semantic"
    }
    parsers: dict[int, VerifiedSemanticRuleTrack] = {}
    episode_objects = {
        str(row.get("episode_id")): EpisodeRecord.from_dict(row)
        for row in records.get("episodes", ())
    }
    episodes_by_agent: dict[Any, list[EpisodeRecord]] = {}
    for episode in episode_objects.values():
        episodes_by_agent.setdefault(episode.agent_id, []).append(episode)
    for agent_episodes in episodes_by_agent.values():
        agent_episodes.sort(key=lambda episode: episode.decision_t)
    reparsed_creation_candidates: set[tuple[Any, str, str]] = set()
    for proposal in proposals:
        current_t = proposal.get("current_t")
        agent_id = proposal.get("agent_id")
        api_row = semantic_api[(current_t, agent_id)]
        raw_output = proposal.get("raw_output")
        if not isinstance(raw_output, str):
            raise _contract_error("semantic proposal raw_output must be a string")
        raw_output_hash = _sha256_text(raw_output)
        if (
            proposal.get("prompt_hash") != api_row.get("prompt_hash")
            or proposal.get("raw_output_hash") != raw_output_hash
            or api_row.get("raw_output_hash") != raw_output_hash
            or proposal.get("provider_error") != api_row.get("error_type")
        ):
            raise _contract_error(
                "semantic proposal prompt/raw/provider receipt is not bound to api_usage"
            )
        provider_model = f"{api_row.get('provider')}/{api_row.get('model')}"
        parser = parsers.get(agent_id)
        if parser is None:
            episodic = EvidenceLinkedEpisodicTrack(
                run_id=str(config.get("run_id")),
                seed=int(config.get("seed")),
                agent_id=int(agent_id),
                prompt_capacity=int(config.get("episodic_prompt_capacity", 24)),
            )
            parser = VerifiedSemanticRuleTrack(episodic, **dict(effective))
            parsers[int(agent_id)] = parser
        parse_status = proposal.get("candidate_parse_status")
        try:
            candidate = parser.parse_candidate(
                raw_output,
                generator_id=provider_model,
            )
        except (TypeError, ValueError) as exc:
            if parse_status != "failure" or proposal.get("parse_error") != str(exc):
                raise _contract_error(
                    "semantic proposal parse status/error does not reproduce"
                ) from exc
            if proposal.get("candidate_parse_mode") != "parse_failure":
                raise _contract_error(
                    "semantic proposal candidate_parse_mode does not reproduce from "
                    "raw_output"
                )
            matching_events = [
                event
                for event in events_by_agent.get(agent_id, [])
                if event.timestamp == current_t
                and event.event_type == "candidate_parse_rejected"
                and event.rule_id is None
            ]
            if (
                len(matching_events) != 1
                or matching_events[0].reason != str(exc)
                or matching_events[0].provenance.get("raw_response_hash")
                != _m3_hash({"raw_response": raw_output})
            ):
                raise _contract_error(
                    "failed semantic proposal lacks matching parse-rejection provenance"
                )
            continue
        if parse_status != "success" or proposal.get("parse_error") is not None:
            raise _contract_error(
                "semantic proposal marked failure despite successful deterministic parse"
            )
        if proposal.get("candidate_parse_mode") != _semantic_parse_mode(raw_output):
            raise _contract_error(
                "semantic proposal candidate_parse_mode does not reproduce from "
                "raw_output"
            )
        semantic_policy = proposal.get(
            "semantic_policy", "evidence-grounded"
        )
        if semantic_policy == "unverified-immediate":
            matching_events = [
                event
                for event in events_by_agent.get(agent_id, [])
                if event.timestamp == current_t
                and event.rule_id == proposal.get("rule_id")
                and event.event_type
                in {
                    "experimental_rule_injected_active",
                    "duplicate_unverified_candidate_ignored",
                }
                and event.candidate_id is None
                and event.provenance.get("semantic_policy")
                == "unverified-immediate"
                and event.provenance.get("source_candidate_id")
                == candidate.candidate_id
                and event.to_status == proposal.get("rule_status")
            ]
        elif semantic_policy == "evidence-grounded":
            matching_events = [
                event
                for event in events_by_agent.get(agent_id, [])
                if event.timestamp == current_t
                and event.rule_id == proposal.get("rule_id")
                and event.candidate_id == candidate.candidate_id
                and event.to_status == proposal.get("rule_status")
            ]
        else:
            raise _contract_error(
                "semantic proposal has an unknown semantic_policy"
            )
        if len(matching_events) != 1:
            raise _contract_error(
                "successful semantic proposal lacks content-addressed candidate event"
            )
        event = matching_events[0]
        if semantic_policy == "unverified-immediate":
            rule_id = proposal.get("rule_id")
            rule = parsed_rules.get((agent_id, rule_id))
            provenance = (
                {} if rule is None else dict(rule.injection_provenance or {})
            )
            if (
                rule is None
                or not rule.injected
                or rule.status != "active"
                or proposal.get("rule_status") != "active"
                or candidate.context_scope != rule.context_scope
                or candidate.condition != rule.condition
                or candidate.action_guidance != rule.action_guidance
                or candidate.outcome_criterion != rule.outcome_criterion
                or candidate.rationale != rule.rationale
                or provenance.get("semantic_policy")
                != "unverified-immediate"
                or provenance.get("source_candidate_id")
                != candidate.candidate_id
                or provenance.get("generator_id") != candidate.generator_id
                or provenance.get("raw_response_hash")
                != candidate.raw_response_hash
                or provenance.get("requested_support_ids")
                != list(candidate.supporting_episode_ids)
                or provenance.get("evidence_admission") is not False
                or provenance.get("retirement_enabled") is not False
            ):
                raise _contract_error(
                    "unverified semantic proposal does not exactly bind its "
                    "parsed candidate and explicit bypass provenance"
                )
            if (
                event.event_type == "experimental_rule_injected_active"
                and event.timestamp != rule.created_at
            ):
                raise _contract_error(
                    "unverified rule creation timestamp does not match proposal"
                )
            reparsed_creation_candidates.add(
                (agent_id, rule.rule_id, candidate.candidate_id)
            )
            continue
        candidate_rule_key = event.provenance.get("candidate_rule_key")
        if (
            candidate_rule_key is not None
            and candidate_rule_key != candidate.rule_key
        ):
            raise _contract_error("candidate event rule_key does not match parsed proposal")
        if event.event_type in {"candidate_verified", "candidate_rejected"}:
            rule_id = proposal.get("rule_id")
            rule = parsed_rules.get((agent_id, rule_id))
            if rule is None or rule.injected:
                raise _contract_error(
                    "candidate creation proposal does not bind a non-injected rule"
                )
            if (
                tuple(rule.candidate_ids) != (candidate.candidate_id,)
                or candidate.rule_key != rule.rule_key
                or candidate.rule_family_id != rule.rule_family_id
                or candidate.context_scope != rule.context_scope
                or candidate.condition != rule.condition
                or candidate.action_guidance != rule.action_guidance
                or candidate.outcome_criterion != rule.outcome_criterion
                or candidate.rationale != rule.rationale
                or event.timestamp != rule.created_at
                or event.provenance.get("generator_id") != candidate.generator_id
                or event.provenance.get("raw_response_hash")
                != candidate.raw_response_hash
                or event.provenance.get("requested_support_ids")
                != list(candidate.supporting_episode_ids)
                or event.provenance.get("migration_notes")
                != list(candidate.migration_notes)
            ):
                raise _contract_error(
                    f"rule {rule.rule_id} is not exactly bound to its reparsed "
                    "creation candidate"
                )
            unique_requested = tuple(
                dict.fromkeys(candidate.supporting_episode_ids)
            )
            expected_creation_episode_ids = tuple(
                episode_id
                for episode_id in unique_requested
                if (
                    episode_id in episode_objects
                    and episode_objects[episode_id].agent_id == agent_id
                    and episode_objects[episode_id].outcome_t <= rule.created_at
                )
            )
            if event.episode_ids != expected_creation_episode_ids:
                raise _contract_error(
                    f"rule {rule.rule_id} creation event evidence IDs do not "
                    "reproduce"
                )
            expected_initial_ids = {
                evidence_type: []
                for evidence_type in (
                    "support",
                    "harmful_compliance",
                    "alternative_success",
                    "alternative_failure",
                    "irrelevant",
                )
            }
            for episode_id in unique_requested:
                episode = episode_objects.get(episode_id)
                if (
                    episode is None
                    or episode.agent_id != agent_id
                    or episode.outcome_t > rule.created_at
                ):
                    continue
                expected_initial_ids[
                    parser._classification(candidate, episode)
                ].append(episode_id)
            for episode in episodes_by_agent.get(agent_id, []):
                if (
                    episode.outcome_t > rule.created_at
                    or episode.episode_id in unique_requested
                ):
                    continue
                classification = parser._classification(candidate, episode)
                if classification != "support":
                    expected_initial_ids[classification].append(episode.episode_id)
            classified_ids = {
                "support": rule.supporting_episode_ids,
                "harmful_compliance": rule.harmful_compliance_episode_ids,
                "alternative_success": rule.alternative_success_episode_ids,
                "alternative_failure": rule.alternative_failure_episode_ids,
                "irrelevant": rule.irrelevant_episode_ids,
            }
            actual_initial_ids = {
                evidence_type: [
                    episode_id
                    for episode_id in episode_ids
                    if episode_objects[episode_id].outcome_t <= rule.created_at
                ]
                for evidence_type, episode_ids in classified_ids.items()
            }
            if actual_initial_ids != expected_initial_ids:
                raise _contract_error(
                    f"rule {rule.rule_id} creation evidence search does not "
                    "reproduce"
                )
            reparsed_creation_candidates.add(
                (agent_id, rule.rule_id, candidate.candidate_id)
            )

    expected_creation_candidates = set()
    for (agent_id, _), rule in parsed_rules.items():
        if not rule.injected:
            expected_creation_candidates.add(
                (agent_id, rule.rule_id, rule.candidate_ids[0])
            )
            continue
        provenance = dict(rule.injection_provenance or {})
        if provenance.get("semantic_policy") == "unverified-immediate":
            # The preregistered forced-active error control is an explicit
            # non-proposal injection and is independently reconciled against
            # error_rule_injections.  Only ordinary unverified candidates must
            # bind a reparsed source proposal.
            if provenance.get("error_rule_mode") == "forced-active":
                continue
            source_candidate_id = provenance.get("source_candidate_id")
            if not isinstance(source_candidate_id, str):
                raise _contract_error(
                    "unverified injected rule lacks source_candidate_id"
                )
            expected_creation_candidates.add(
                (agent_id, rule.rule_id, source_candidate_id)
            )
    if reparsed_creation_candidates != expected_creation_candidates:
        raise _contract_error(
            "every non-injected semantic rule must bind exactly one reparsed "
            "creation proposal"
        )


def _validate_current_semantic_retrievals(
    config: Mapping[str, Any],
    actions_by_key: Mapping[tuple[Any, Any], Mapping[str, Any]],
    traces_by_key: Mapping[tuple[Any, Any], Mapping[str, Any]],
    episodes_by_key: Mapping[tuple[Any, Any], EpisodeRecord],
    parsed_rules: Mapping[tuple[Any, Any], VerifiedRule],
    events_by_agent: Mapping[Any, list[RuleEvent]],
) -> None:
    """Replay each v2 M3 eligibility/ranking query from its exact event prefix."""

    retrieval_event_types = {
        "active_rule_retrieved",
        "active_rule_retrieval_empty",
    }
    mutating_event_types = {
        "candidate_verified",
        "candidate_rejected",
        "experimental_rule_injected_active",
        "rule_activated",
        "rule_retired",
    }
    semantic_enabled = bool(config.get("enable_semantic"))
    if not semantic_enabled:
        if parsed_rules or any(events_by_agent.values()):
            raise _contract_error(
                "semantic-disabled current run contains rules or retrieval events"
            )
        for key, action in actions_by_key.items():
            selected = _list_field(action, "selected_rule_ids", "actions")
            trace_selected = _list_field(
                traces_by_key[key], "selected_rule_ids", "context_trace"
            )
            if selected or trace_selected or episodes_by_key[key].selected_rule_ids:
                raise _contract_error(
                    "semantic-disabled current run selected a semantic rule"
                )
        return

    rule_budget = _nonnegative_int(config.get("rule_budget"), "config.rule_budget")
    rules_by_agent: dict[Any, list[VerifiedRule]] = {}
    for (agent_id, _), rule in parsed_rules.items():
        rules_by_agent.setdefault(agent_id, []).append(rule)

    decision_keys_by_agent: dict[Any, list[tuple[Any, Any]]] = {}
    for key in actions_by_key:
        decision_keys_by_agent.setdefault(key[1], []).append(key)

    for agent_id, decision_keys in decision_keys_by_agent.items():
        events = list(events_by_agent.get(agent_id, ()))
        all_retrieval_indices = {
            index
            for index, event in enumerate(events)
            if event.event_type in retrieval_event_types
        }
        bound_retrieval_indices: set[int] = set()
        for key in sorted(decision_keys):
            decision_t, _ = key
            episode = episodes_by_key[key]
            action_selected = _list_field(
                actions_by_key[key], "selected_rule_ids", "actions"
            )
            trace_selected = _list_field(
                traces_by_key[key], "selected_rule_ids", "context_trace"
            )
            if not (
                action_selected
                == trace_selected
                == list(episode.selected_rule_ids)
            ):
                raise _contract_error(
                    "selected rule IDs are not bound across action, context, and episode"
                )

            retrieval_entries = [
                (index, event)
                for index, event in enumerate(events)
                if event.timestamp == decision_t
                and event.event_type in retrieval_event_types
            ]
            if action_selected:
                selected_entries = [
                    (index, event)
                    for index, event in retrieval_entries
                    if event.event_type == "active_rule_retrieved"
                ]
                selected_indices = [index for index, _ in selected_entries]
                if (
                    len(retrieval_entries) != len(action_selected)
                    or [event.rule_id for _, event in selected_entries]
                    != action_selected
                    or selected_indices
                    != list(
                        range(
                            selected_indices[0],
                            selected_indices[0] + len(selected_indices),
                        )
                    )
                ):
                    raise _contract_error(
                        "selected_rule_ids do not bind the exact same-time M3 "
                        "retrieval block"
                    )
                block_start = selected_indices[0]
            else:
                if (
                    len(retrieval_entries) != 1
                    or retrieval_entries[0][1].event_type
                    != "active_rule_retrieval_empty"
                ):
                    raise _contract_error(
                        "empty semantic selection lacks its exact same-time M3 marker"
                    )
                block_start = retrieval_entries[0][0]

            historical_status: dict[str, str] = {}
            historical_metrics: dict[str, Mapping[str, Any]] = {}
            historical_updated_at: dict[str, int] = {}
            for event in events[:block_start]:
                rule_id = event.rule_id
                if rule_id is None:
                    continue
                historical_metrics[rule_id] = dict(event.metrics)
                if (
                    event.event_type in mutating_event_types
                    or event.event_type.endswith("_evidence_added")
                ):
                    historical_updated_at[rule_id] = event.timestamp
                if event.to_status is not None:
                    historical_status[rule_id] = event.to_status

            try:
                eligible = [
                    rule
                    for rule in rules_by_agent.get(agent_id, ())
                    if (
                        historical_status.get(rule.rule_id) == "active"
                        and rule.context_scope.matches(episode.pre_state)
                        and rule.condition.matches(episode.pre_state)
                    )
                ]
                eligible.sort(
                    key=lambda rule: (
                        -float(historical_metrics[rule.rule_id]["confidence"]),
                        -float(historical_metrics[rule.rule_id]["margin"]),
                        -historical_updated_at[rule.rule_id],
                        rule.rule_id,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise _contract_error(
                    "M3 event prefix cannot reproduce historical rule ranking"
                ) from exc
            expected_rule_ids = [
                rule.rule_id for rule in eligible[:rule_budget]
            ]
            if action_selected != expected_rule_ids:
                raise _contract_error(
                    "selected_rule_ids do not reproduce the exact M3 "
                    "eligibility/ranking query"
                )

            if action_selected:
                for _, event in retrieval_entries:
                    if (
                        event.event_type != "active_rule_retrieved"
                        or event.candidate_id is not None
                        or event.from_status != "active"
                        or event.to_status != "active"
                        or event.episode_ids
                        or event.reason
                        != "active rule condition matched current state"
                        or dict(event.provenance)
                    ):
                        raise _contract_error(
                            "active-rule retrieval event has malformed exact shape"
                        )
            else:
                marker = retrieval_entries[0][1]
                expected_provenance = {
                    "limit": rule_budget,
                    "state_hash": _m3_hash(dict(episode.pre_state)),
                }
                if (
                    marker.rule_id is not None
                    or marker.candidate_id is not None
                    or marker.from_status is not None
                    or marker.to_status is not None
                    or marker.episode_ids
                    or marker.reason
                    != "no active rule matched within the retrieval limit"
                    or dict(marker.metrics)
                    or dict(marker.provenance) != expected_provenance
                ):
                    raise _contract_error(
                        "empty M3 retrieval marker does not match the exact "
                        "decision state/budget"
                    )
            bound_retrieval_indices.update(index for index, _ in retrieval_entries)

        if bound_retrieval_indices != all_retrieval_indices:
            raise _contract_error(
                "M3 retrieval ledger contains an event or empty marker not bound "
                "to exactly one decision"
            )


def _validate_current_identity_bindings(
    config: Mapping[str, Any],
    records: Mapping[str, tuple[Mapping[str, Any], ...]],
) -> None:
    """Bind every current decision receipt to the same prompt and memory inputs."""

    actions_by_key = {
        (row.get("decision_t"), row.get("agent_id")): row
        for row in records.get("actions", ())
    }
    traces_by_key = {
        (row.get("decision_t"), row.get("agent_id")): row
        for row in records.get("context_trace", ())
    }
    snapshots_by_key = {
        (row.get("decision_t"), row.get("agent_id")): row
        for row in records.get("decision_snapshots", ())
    }
    episode_rows_by_key = {
        (row.get("decision_t"), row.get("agent_id")): row
        for row in records.get("episodes", ())
    }
    action_api_by_key = {
        (row.get("decision_t"), row.get("agent_id")): row
        for row in records.get("api_usage", ())
        if row.get("call_kind") == "action"
    }

    episode_objects: dict[str, EpisodeRecord] = {}
    safe_run_id = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(config.get("run_id"))
    )
    for key, row in episode_rows_by_key.items():
        try:
            episode = EpisodeRecord.from_dict(row)
        except (KeyError, TypeError, ValueError) as exc:
            raise _contract_error(
                f"episodes{key!r} failed record_hash/integrity validation: {exc}"
            ) from exc
        decision_t, agent_id = key
        expected_episode_id = (
            f"{safe_run_id}:s{int(config.get('seed'))}:"
            f"a{agent_id}:t{decision_t}"
        )
        if (
            episode.run_id != config.get("run_id")
            or episode.seed != config.get("seed")
            or episode.agent_id != agent_id
            or episode.decision_t != decision_t
            or episode.episode_id != expected_episode_id
            or episode.decision_id != f"D:{expected_episode_id}"
        ):
            raise _contract_error(
                f"episodes{key!r} identity does not match the sealed run"
            )
        if episode.episode_id in episode_objects:
            raise _contract_error(f"duplicate episode_id {episode.episode_id!r}")
        _validate_rate_availability(
            episode.pre_state,
            available=decision_t > 0,
            label=f"episodes{key!r}.pre_state",
        )
        _validate_rate_availability(
            episode.next_state,
            available=True,
            label=f"episodes{key!r}.next_state",
        )
        episode_objects[episode.episode_id] = episode

    # Reconstruct the append-only per-agent M2 ledger in sealed stream order.
    # The record hash alone cannot prove that utility normalization used only
    # the causal prefix: a forged row can simply be re-hashed.  Recomputing the
    # median/MAD-derived fields here keeps M3 evidence invariant under restore.
    last_decision_t: dict[Any, int] = {}
    for row in records.get("episodes", ()):
        episode = episode_objects[str(row.get("episode_id"))]
        agent_id = episode.agent_id
        previous_t = last_decision_t.get(agent_id)
        if previous_t is not None and episode.decision_t <= previous_t:
            raise _contract_error(
                "episodes are not in strictly increasing per-agent causal order"
            )
        last_decision_t[agent_id] = episode.decision_t

    causal_utilities: dict[Any, list[float]] = {}
    for row in records.get("episodes", ()):
        episode = episode_objects[str(row.get("episode_id"))]
        agent_id = episode.agent_id
        previous_utilities = causal_utilities.setdefault(agent_id, [])
        expected_advantage, expected_importance = _utility_statistics(
            previous_utilities,
            episode.flow_utility,
        )
        if not math.isclose(
            episode.utility_advantage,
            expected_advantage,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise _contract_error(
                f"episode {episode.episode_id} utility_advantage does not match "
                "its per-agent causal prefix"
            )
        if not math.isclose(
            episode.importance,
            expected_importance,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise _contract_error(
                f"episode {episode.episode_id} importance does not match its "
                "per-agent causal prefix"
            )
        previous_utilities.append(episode.flow_utility)

    parsed_rules: dict[tuple[Any, Any], VerifiedRule] = {}
    for row in records.get("semantic_rules", ()):
        agent_id = row.get("agent_id")
        payload = {key: value for key, value in row.items() if key != "agent_id"}
        try:
            rule = VerifiedRule.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise _contract_error(f"semantic rule failed structural validation: {exc}") from exc
        if rule.to_dict() != payload:
            raise _contract_error("semantic rule normalization changed sealed fields")
        parsed_rules[(agent_id, rule.rule_id)] = rule

    retrieval_events: dict[tuple[Any, Any, Any], RuleEvent] = {}
    events_by_agent: dict[Any, list[RuleEvent]] = {}
    for row in records.get("semantic_rule_events", ()):
        agent_id = row.get("agent_id")
        payload = {key: value for key, value in row.items() if key != "agent_id"}
        try:
            event = RuleEvent.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise _contract_error(f"semantic event failed structural validation: {exc}") from exc
        if event.to_dict() != payload:
            raise _contract_error("semantic event normalization changed sealed fields")
        events_by_agent.setdefault(agent_id, []).append(event)
        if event.event_type == "active_rule_retrieved":
            key = (agent_id, event.timestamp, event.rule_id)
            if key in retrieval_events:
                raise _contract_error(f"duplicate active-rule retrieval event {key!r}")
            retrieval_events[key] = event

    _validate_current_semantic_rows(
        config,
        episode_objects,
        parsed_rules,
        events_by_agent,
    )
    _validate_current_semantic_proposals(
        config,
        records,
        events_by_agent,
        parsed_rules,
    )
    episodes_by_key = {
        key: episode_objects[str(row.get("episode_id"))]
        for key, row in episode_rows_by_key.items()
    }
    _validate_current_semantic_retrievals(
        config,
        actions_by_key,
        traces_by_key,
        episodes_by_key,
        parsed_rules,
        events_by_agent,
    )

    route_contract = {
        "no-context": (False, False),
        "prompt-only": (False, True),
        "retrieval-only": (True, False),
        "full": (True, True),
    }
    expected_route = route_contract.get(config.get("context_mode"))
    if expected_route is None:
        raise _contract_error("config.context_mode is not registered")
    if tuple(config.get("context_features", ())) != tuple(CONTEXT_FEATURES):
        raise _contract_error("config.context_features does not match runner v2")
    context_router = CausalContextRouter(
        base_feature_names=CONTEXT_FEATURES,
        window_size=6,
        mode=str(config.get("context_mode")),
    )
    context_history_by_agent: dict[Any, list[dict[str, Any]]] = {
        agent_id: [] for agent_id in {key[1] for key in actions_by_key}
    }
    retrieval_tracks = {
        agent_id: EvidenceLinkedEpisodicTrack(
            run_id=str(config.get("run_id")),
            seed=int(config.get("seed")),
            agent_id=int(agent_id),
            prompt_capacity=int(config.get("episodic_prompt_capacity", 24)),
        )
        for agent_id in context_history_by_agent
    }

    for key in sorted(actions_by_key):
        action = actions_by_key[key]
        decision_t, agent_id = key
        trace = traces_by_key[key]
        snapshot = snapshots_by_key[key]
        episode_row = episode_rows_by_key[key]
        episode = episode_objects[str(episode_row.get("episode_id"))]
        api_row = action_api_by_key[key]

        context_packet_value = _mapping(
            trace.get("context_packet"), "context_trace.context_packet"
        )
        try:
            packet = ContextPacket.from_dict(context_packet_value)
        except (KeyError, TypeError, ValueError) as exc:
            raise _contract_error(f"context packet failed integrity validation: {exc}") from exc
        if packet.decision_t != decision_t:
            raise _contract_error("context packet decision_t does not match stream identity")
        low_labor_available = decision_t > 0
        context_observation = {
            "timestamp": decision_t,
            "log_price": math.log1p(float(episode.pre_state["price"])),
            "interest_rate": float(episode.pre_state["interest_rate"]),
            "prior_low_labor_rate": (
                float(episode.pre_state["low_labor_rate"])
                if low_labor_available
                else 0.0
            ),
            "prior_low_labor_rate_available": float(low_labor_available),
            "inflation": float(episode.pre_state["inflation"]),
            "log_wealth": math.log1p(float(episode.pre_state["wealth"])),
            "employed": float(bool(episode.pre_state["employed"])),
        }
        context_history_by_agent[agent_id].append(context_observation)
        expected_packet = context_router.encode(
            context_history_by_agent[agent_id],
            decision_t=decision_t,
            observed_through=decision_t,
        )
        if expected_packet.to_dict() != packet.to_dict():
            raise _contract_error(
                "context packet does not reproduce from the causal episode history"
            )
        raw_context = dict(zip(packet.feature_names, packet.raw_features))
        required_low_labor_features = {
            "prior_low_labor_rate.last",
            "prior_low_labor_rate.mean",
            "prior_low_labor_rate.slope",
            "prior_low_labor_rate_available.last",
            "prior_low_labor_rate_available.mean",
            "prior_low_labor_rate_available.slope",
        }
        if not required_low_labor_features.issubset(raw_context):
            raise _contract_error(
                "context packet lacks the registered low-labor value/mask features"
            )
        expected_available = 0.0 if decision_t == 0 else 1.0
        if raw_context["prior_low_labor_rate_available.last"] != expected_available:
            raise _contract_error(
                "context packet current low-labor availability disagrees with decision_t"
            )
        if decision_t == 0:
            if (
                packet.observed_through != 0
                or packet.history_start != 0
                or packet.observation_count != 1
                or any(
                    raw_context[name] != 0.0
                    for name in required_low_labor_features
                )
                or packet.prompt_summary.count(
                    "prior_low_labor_rate=unavailable"
                )
                != 1
            ):
                raise _contract_error(
                    "t0 context packet must preserve unavailable prior low labor"
                )
        else:
            if (
                packet.observed_through != decision_t
                or not math.isclose(
                    raw_context["prior_low_labor_rate.last"],
                    float(episode.pre_state["low_labor_rate"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or "prior_low_labor_rate=unavailable" in packet.prompt_summary
            ):
                raise _contract_error(
                    "context packet current low-labor observation is not bound "
                    "to episode pre_state"
                )
        if (
            trace.get("context_id") != packet.context_id
            or trace.get("context_hash") != packet.context_hash
            or snapshot.get("context_packet_id") != packet.context_id
            or snapshot.get("context_packet_hash") != packet.context_hash
            or episode.context_id != packet.context_id
            or tuple(episode.context_vector) != tuple(packet.context_vector)
        ):
            raise _contract_error(
                "context packet identity/vector is not bound across trace, snapshot, "
                "and episode"
            )
        expected_to_retrieval, expected_to_prompt = expected_route
        if (
            trace.get("context_mode") != config.get("context_mode")
            or trace.get("context_to_retrieval") is not expected_to_retrieval
            or trace.get("context_to_prompt") is not expected_to_prompt
        ):
            raise _contract_error("context trace route does not match config.context_mode")

        action_retrieved = _list_field(action, "retrieved_episode_ids", "actions")
        action_selected = _list_field(action, "selected_rule_ids", "actions")
        trace_retrieved = _list_field(
            trace, "retrieved_episode_ids", "context_trace"
        )
        trace_selected = _list_field(trace, "selected_rule_ids", "context_trace")
        if not (
            action_retrieved
            == trace_retrieved
            == list(episode.retrieved_episode_ids)
            and action_selected
            == trace_selected
            == list(episode.selected_rule_ids)
        ):
            raise _contract_error(
                "retrieved/selected IDs are not bound across action, context, and episode"
            )
        if len(trace.get("retrieval_scores", [])) != len(trace_retrieved) or len(
            trace.get("score_components", [])
        ) != len(trace_retrieved):
            raise _contract_error(
                "context retrieval scores/components do not align with retrieved IDs"
            )
        expected_hits = ()
        if bool(config.get("enable_episodic_retrieval")):
            expected_hits = retrieval_tracks[agent_id].retrieve(
                current_t=decision_t,
                current_state=episode.pre_state,
                context_vector=(
                    packet.context_vector if expected_to_retrieval else None
                ),
                use_context=expected_to_retrieval,
                k=int(config.get("retrieval_k")),
            )
        expected_retrieved = [hit.episode.episode_id for hit in expected_hits]
        expected_scores = [hit.score for hit in expected_hits]
        expected_components = [dict(hit.components) for hit in expected_hits]
        if (
            trace_retrieved != expected_retrieved
            or trace.get("retrieval_scores") != expected_scores
            or trace.get("score_components") != expected_components
        ):
            raise _contract_error(
                "episodic retrieval IDs/scores/components do not reproduce from M2"
            )
        for episode_id in trace_retrieved:
            retrieved = episode_objects.get(episode_id)
            if (
                retrieved is None
                or retrieved.agent_id != agent_id
                or retrieved.outcome_t > decision_t
            ):
                raise _contract_error(
                    "retrieved episode is missing, cross-agent, or not yet observable"
                )
        retrieval_tracks[agent_id]._ledger[episode.episode_id] = episode
        retrieval_tracks[agent_id]._prompt_ids.append(episode.episode_id)

        expected_memory_parts: list[str] = []
        if trace_retrieved:
            expected_memory_parts.append("Finalized experience evidence:")
            expected_memory_parts.extend(
                f"- {episode_objects[episode_id].to_prompt_text()}"
                for episode_id in trace_retrieved
            )
        if trace_selected:
            expected_memory_parts.append("Verified active rules:")
            for rule_id in trace_selected:
                rule = parsed_rules.get((agent_id, rule_id))
                event = retrieval_events.get((agent_id, decision_t, rule_id))
                if rule is None or event is None:
                    raise _contract_error(
                        "selected rule lacks same-agent rule/retrieval provenance"
                    )
                confidence = event.metrics.get("confidence")
                try:
                    prompt_rule = replace(rule, confidence=float(confidence))
                except (TypeError, ValueError) as exc:
                    raise _contract_error(
                        "active-rule retrieval event has invalid confidence"
                    ) from exc
                expected_memory_parts.append(f"- {prompt_rule.to_prompt_text()}")
        expected_memory = " ".join(expected_memory_parts)
        if snapshot.get("memory_text") != expected_memory:
            raise _contract_error(
                "decision snapshot memory_text does not reconstruct from selected evidence"
            )

        protected_context = packet.prompt_summary if expected_to_prompt else ""
        if snapshot.get("protected_context_text") != protected_context:
            raise _contract_error(
                "protected context text does not match the routed context packet"
            )
        base_prompt = snapshot.get("base_prompt")
        memory_text = snapshot.get("memory_text")
        if not isinstance(base_prompt, str) or not isinstance(memory_text, str):
            raise _contract_error("decision snapshot prompt fields must be strings")
        try:
            prompt = compose_decision_prompt(base_prompt, memory_text)
        except (TypeError, ValueError) as exc:
            raise _contract_error(f"decision prompt reconstruction failed: {exc}") from exc
        expected_prompt_schema = (
            PROMPT_SCHEMA_VERSION
            if config.get("schema_version") == RUNNER_SCHEMA_VERSION
            else PREVIOUS_PROMPT_SCHEMA_VERSION
        )
        if (
            snapshot.get("prompt_schema_version") != expected_prompt_schema
            or snapshot.get("base_prompt_hash") != prompt.base_prompt_hash
            or snapshot.get("memory_hash") != prompt.memory_hash
            or snapshot.get("full_prompt_hash") != prompt.full_prompt_hash
            or snapshot.get("protected_context_hash")
            != _sha256_text(protected_context)
        ):
            raise _contract_error("decision snapshot prompt/hash fields do not reconstruct")
        context_marker_prefix = "Causal context summary:"
        if expected_to_prompt:
            marker = f"{context_marker_prefix} {protected_context}"
            if base_prompt.count(context_marker_prefix) != 1 or marker not in base_prompt:
                raise _contract_error(
                    "base prompt must contain exactly the routed context summary once"
                )
        elif context_marker_prefix in base_prompt:
            raise _contract_error(
                "non-prompt context route contains a causal context summary marker"
            )

        combined_parts = []
        if protected_context:
            combined_parts.append(f"Causal context summary: {protected_context}")
        if memory_text:
            combined_parts.append(memory_text)
        combined_prompt = " ".join(combined_parts)
        if (
            trace.get("protected_context_prompt_hash")
            != _sha256_text(protected_context)
            or trace.get("memory_prompt_hash") != _sha256_text(memory_text)
            or trace.get("combined_prompt_hash") != _sha256_text(combined_prompt)
        ):
            raise _contract_error("context trace prompt hashes do not reconstruct")
        bundle_payload = {
            "decision_t": decision_t,
            "context_id": packet.context_id,
            "context_mode": trace.get("context_mode"),
            "episode_ids": trace_retrieved,
            "episode_scores": trace.get("retrieval_scores"),
            "rule_ids": trace_selected,
            "protected_context_prompt": protected_context,
            "memory_prompt": memory_text,
            "prompt": combined_prompt,
        }
        if trace.get("memory_bundle_hash") != _sha256_json(bundle_payload):
            raise _contract_error("context trace memory_bundle_hash does not reconstruct")

        if snapshot.get("environment_state_hash") != _sha256_json(episode.pre_state):
            raise _contract_error(
                "snapshot environment_state_hash does not match episode pre_state"
            )
        if (
            action.get("prompt_hash") != prompt.full_prompt_hash
            or api_row.get("prompt_hash") != prompt.full_prompt_hash
            or action.get("provider") != api_row.get("provider")
            or action.get("model") != api_row.get("model")
            or snapshot.get("provider_model")
            != f"{action.get('provider')}/{action.get('model')}"
        ):
            raise _contract_error(
                "action/API provider, model, or prompt hash is not bound to snapshot"
            )
        raw_output = action.get("raw_output")
        if not isinstance(raw_output, str):
            raise _contract_error("action.raw_output must be a string")
        try:
            parsed_action = parse_direct_action(
                raw_output,
                max_labor_hours=float(config.get("max_labor_hours")),
                labor_step=float(config.get("labor_step")),
                consumption_step=float(config.get("consumption_step")),
            )
        except (TypeError, ValueError) as exc:
            raise _contract_error(f"sealed action cannot be reparsed: {exc}") from exc
        decision = _mapping(action.get("decision"), "actions.decision")
        if parsed_action.to_dict() != dict(decision):
            raise _contract_error("action decision does not match reparsed raw_output")
        if api_row.get("raw_output_hash") != parsed_action.raw_output_hash:
            raise _contract_error("action/API raw output hashes do not match")

        expected_proposed = {
            "work_propensity": parsed_action.proposed_work_fraction,
            "consumption_fraction": parsed_action.proposed_consumption_fraction,
        }
        expected_executed = {
            "labor_hours": parsed_action.executed_labor_hours,
            "work_propensity": parsed_action.proposed_work_fraction,
            "consumption_fraction": parsed_action.executed_consumption_rate,
        }
        if (
            episode.proposed_action != expected_proposed
            or episode.executed_action != expected_executed
            or episode.reflection != parsed_action.reflection
        ):
            raise _contract_error(
                "episode proposed/executed action or reflection does not match action row"
            )


def _require_close(label: str, actual: Any, expected: float) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise _contract_error(f"{label} must be numeric")
    numeric = float(actual)
    if not math.isfinite(numeric) or not math.isclose(
        numeric,
        float(expected),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise _contract_error(
            f"{label} mismatch: declared={actual!r}, recomputed={expected!r}"
        )


def _validate_current_foundation_contract(
    config: Mapping[str, Any],
    *,
    num_agents: int,
    episode_length: int,
) -> None:
    """Bind the sealed Foundation environment to the registered v2 runner config."""

    foundation_env = _mapping(
        config.get("foundation_env"), "config.foundation_env"
    )
    expected_hash = _sha256_json(foundation_env)
    if config.get("foundation_env_hash") != expected_hash:
        raise _contract_error(
            "config.foundation_env_hash does not match the sealed Foundation environment"
        )

    env_num_agents = _nonnegative_int(
        foundation_env.get("n_agents"), "config.foundation_env.n_agents"
    )
    env_episode_length = _nonnegative_int(
        foundation_env.get("episode_length"),
        "config.foundation_env.episode_length",
    )
    if env_num_agents != num_agents:
        raise _contract_error(
            "config.foundation_env.n_agents does not match config.num_agents"
        )
    if env_episode_length != episode_length:
        raise _contract_error(
            "config.foundation_env.episode_length does not match config.episode_length"
        )

    try:
        labor_index, labor_config = locate_component(
            foundation_env, "SimpleLabor"
        )
        consumption_index, consumption_config = locate_component(
            foundation_env, "SimpleConsumption"
        )
    except (TypeError, ValueError) as exc:
        raise _contract_error(
            f"config.foundation_env action components are invalid: {exc}"
        ) from exc
    if labor_index >= consumption_index:
        raise _contract_error(
            "config.foundation_env SimpleLabor must precede SimpleConsumption"
        )
    _require_close(
        "config.foundation_env.SimpleLabor.labor_step",
        labor_config.get("labor_step"),
        float(config.get("labor_step")),
    )
    _require_close(
        "config.foundation_env.SimpleLabor.num_labor_hours",
        labor_config.get("num_labor_hours"),
        float(config.get("max_labor_hours")),
    )
    _require_close(
        "config.foundation_env.SimpleConsumption.consumption_rate_step",
        consumption_config.get("consumption_rate_step"),
        float(config.get("consumption_step")),
    )


def _validate_current_utility_and_result_contract(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    validation_status: Mapping[str, Any],
    records: Mapping[str, tuple[Mapping[str, Any], ...]],
    *,
    num_agents: int,
    macro_count: int,
    episode_length: int,
) -> None:
    """Recompute accounting, headline metrics, and every validation gate."""

    if macro_count <= 0:
        raise _contract_error("current verified runs must contain a completed period")
    utility_value = _mapping(config.get("utility"), "config.utility")
    try:
        utility_config = UtilityConfig(**dict(utility_value))
    except (TypeError, ValueError) as exc:
        raise _contract_error(f"config.utility is invalid: {exc}") from exc
    _require_close(
        "config.utility.max_labor_hours",
        utility_config.max_labor_hours,
        float(config.get("max_labor_hours")),
    )

    episode_by_key = {
        (row.get("decision_t"), row.get("agent_id")): EpisodeRecord.from_dict(row)
        for row in records.get("episodes", ())
    }
    action_by_key = {
        (row.get("decision_t"), row.get("agent_id")): row
        for row in records.get("actions", ())
    }
    ledger_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    previous_by_agent: dict[int, Mapping[str, Any]] = {}
    for sealed_row in records.get("utility_ledger", ()):
        period = sealed_row.get("period")
        raw_agent_id = sealed_row.get("agent_id")
        if (
            isinstance(period, bool)
            or not isinstance(period, int)
            or not isinstance(raw_agent_id, str)
            or not raw_agent_id.isdigit()
        ):
            raise _contract_error("utility_ledger has invalid period/agent identity")
        agent_id = int(raw_agent_id)
        key = (period, agent_id)
        try:
            pre = PreStepSnapshot(
                period=period,
                agent_id=raw_agent_id,
                wealth=sealed_row["wealth_pre"],
                cumulative_production=sealed_row["cumulative_production_pre"],
                price=sealed_row["price"],
                interest_rate=sealed_row["interest_rate"],
                proposed_work_propensity=sealed_row[
                    "proposed_work_propensity"
                ],
                proposed_consumption_fraction=sealed_row[
                    "proposed_consumption_fraction"
                ],
                executed_labor_hours=sealed_row["executed_labor_hours"],
                executed_consumption_rate=sealed_row[
                    "executed_consumption_rate"
                ],
            )
            post = PostStepSnapshot(
                period=period,
                agent_id=raw_agent_id,
                wealth=sealed_row["wealth_post"],
                cumulative_production=sealed_row["cumulative_production_post"],
                tax_paid=sealed_row["tax_paid"],
                lump_sum_transfer=sealed_row["lump_sum_transfer"],
                realized_consumption_spend=sealed_row[
                    "realized_consumption_spend"
                ],
                realized_consumption_quantity=sealed_row[
                    "realized_consumption_quantity"
                ],
                interest_applied=sealed_row["interest_applied"],
                interest_credit=sealed_row["interest_credit"],
            )
            recomputed = build_budget_ledger_row(pre, post, utility_config)
        except (KeyError, TypeError, ValueError) as exc:
            raise _contract_error(
                f"utility_ledger{key!r} failed authoritative reconstruction: {exc}"
            ) from exc
        if recomputed.to_dict() != dict(sealed_row):
            raise _contract_error(
                f"utility_ledger{key!r} does not match authoritative accounting"
            )

        previous = previous_by_agent.get(agent_id)
        if previous is not None:
            _require_close(
                f"utility_ledger{key!r}.wealth_pre",
                sealed_row["wealth_pre"],
                float(previous["wealth_post"]),
            )
            _require_close(
                f"utility_ledger{key!r}.cumulative_production_pre",
                sealed_row["cumulative_production_pre"],
                float(previous["cumulative_production_post"]),
            )
        previous_by_agent[agent_id] = sealed_row

        episode = episode_by_key[key]
        action = action_by_key[key]
        decision = _mapping(action.get("decision"), "actions.decision")
        bound_values = {
            "episode.flow_utility": (
                episode.flow_utility,
                sealed_row["flow_utility"],
            ),
            "episode.pre_state.wealth": (
                episode.pre_state.get("wealth"),
                sealed_row["wealth_pre"],
            ),
            "episode.pre_state.price": (
                episode.pre_state.get("price"),
                sealed_row["price"],
            ),
            "episode.pre_state.interest_rate": (
                episode.pre_state.get("interest_rate"),
                sealed_row["interest_rate"],
            ),
            "episode.next_state.wealth": (
                episode.next_state.get("wealth"),
                sealed_row["wealth_post"],
            ),
            "episode.next_state.income": (
                episode.next_state.get("income"),
                sealed_row["gross_labor_income"],
            ),
            "episode.next_state.consumption_spend": (
                episode.next_state.get("consumption_spend"),
                sealed_row["realized_consumption_spend"],
            ),
            "episode.next_state.consumption_quantity": (
                episode.next_state.get("consumption_quantity"),
                sealed_row["realized_consumption_quantity"],
            ),
            "episode.outcome.gross_labor_income": (
                episode.outcome.get("gross_labor_income"),
                sealed_row["gross_labor_income"],
            ),
            "episode.outcome.tax_paid": (
                episode.outcome.get("tax_paid"),
                sealed_row["tax_paid"],
            ),
            "episode.outcome.lump_sum_transfer": (
                episode.outcome.get("lump_sum_transfer"),
                sealed_row["lump_sum_transfer"],
            ),
            "episode.outcome.consumption_spend": (
                episode.outcome.get("consumption_spend"),
                sealed_row["realized_consumption_spend"],
            ),
            "episode.outcome.consumption_quantity": (
                episode.outcome.get("consumption_quantity"),
                sealed_row["realized_consumption_quantity"],
            ),
            "episode.outcome.labor_hours": (
                episode.outcome.get("labor_hours"),
                sealed_row["executed_labor_hours"],
            ),
            "episode.outcome.interest_credit": (
                episode.outcome.get("interest_credit"),
                sealed_row["interest_credit"],
            ),
            "episode.outcome.budget_residual": (
                episode.outcome.get("budget_residual"),
                sealed_row["budget_residual"],
            ),
            "episode.outcome.wealth_change": (
                episode.outcome.get("wealth_change"),
                float(sealed_row["wealth_post"])
                - float(sealed_row["wealth_pre"]),
            ),
            "action.executed_labor_hours": (
                decision.get("executed_labor_hours"),
                sealed_row["executed_labor_hours"],
            ),
            "action.executed_consumption_rate": (
                decision.get("executed_consumption_rate"),
                sealed_row["executed_consumption_rate"],
            ),
            "action.proposed_work_propensity": (
                decision.get("proposed_work_fraction"),
                sealed_row["proposed_work_propensity"],
            ),
            "action.proposed_consumption_fraction": (
                decision.get("proposed_consumption_fraction"),
                sealed_row["proposed_consumption_fraction"],
            ),
        }
        for label, (actual, expected) in bound_values.items():
            _require_close(f"{label}{key!r}", actual, float(expected))
        if episode.outcome.get("supply_rationed") is not sealed_row[
            "supply_rationed"
        ]:
            raise _contract_error(
                f"episode.outcome.supply_rationed{key!r} disagrees with ledger"
            )
        ledger_by_key[key] = sealed_row

    macro_rows = records.get("macro_steps", ())
    for period, macro in enumerate(macro_rows):
        period_ledger = [
            ledger_by_key[(period, agent_id)] for agent_id in range(num_agents)
        ]
        period_actions = [
            action_by_key[(period, agent_id)] for agent_id in range(num_agents)
        ]
        period_episodes = [
            episode_by_key[(period, agent_id)] for agent_id in range(num_agents)
        ]
        wealths = [float(row["wealth_post"]) for row in period_ledger]
        low_labor_rate = sum(
            float(
                _mapping(action.get("decision"), "actions.decision").get(
                    "executed_labor_hours"
                )
                < float(config.get("low_labor_threshold_hours"))
            )
            for action in period_actions
        ) / num_agents
        _require_close(
            f"macro_steps[{period}].average_wealth",
            macro.get("average_wealth"),
            sum(wealths) / num_agents,
        )
        _require_close(
            f"macro_steps[{period}].low_labor_rate",
            macro.get("low_labor_rate"),
            low_labor_rate,
        )
        prices = {float(episode.next_state["price"]) for episode in period_episodes}
        inflations = {
            float(episode.next_state["inflation"]) for episode in period_episodes
        }
        if len(prices) != 1 or len(inflations) != 1:
            raise _contract_error(
                f"episodes at period {period} disagree on shared macro state"
            )
        _require_close(
            f"macro_steps[{period}].price", macro.get("price"), prices.pop()
        )
        _require_close(
            f"macro_steps[{period}].monthly_inflation",
            macro.get("monthly_inflation"),
            inflations.pop(),
        )

    final_wealths = [
        float(ledger_by_key[(macro_count - 1, agent_id)]["wealth_post"])
        for agent_id in range(num_agents)
    ]
    ordered_wealths = sorted(final_wealths)
    midpoint = len(ordered_wealths) // 2
    median_wealth = (
        ordered_wealths[midpoint]
        if len(ordered_wealths) % 2
        else (ordered_wealths[midpoint - 1] + ordered_wealths[midpoint]) / 2.0
    )
    utility_values = [
        float(row["flow_utility"])
        for row in records.get("utility_ledger", ())
    ]
    low_labor_values = [float(row["low_labor_rate"]) for row in macro_rows]
    expected_metrics = {
        "average_wealth": sum(final_wealths) / len(final_wealths),
        "median_wealth": median_wealth,
        "gini": _gini(final_wealths),
        "average_flow_utility": sum(utility_values) / len(utility_values),
        "average_low_labor_rate": sum(low_labor_values) / len(low_labor_values),
    }
    final_metrics = _mapping(summary.get("final_metrics"), "summary.final_metrics")
    if set(final_metrics) != set(expected_metrics):
        raise _contract_error(
            "summary.final_metrics must contain exactly the registered metrics"
        )
    for metric, expected in expected_metrics.items():
        _require_close(
            f"summary.final_metrics.{metric}", final_metrics.get(metric), expected
        )

    proposals = records.get("semantic_proposals", ())
    provider_errors = [
        row
        for row in records.get("errors", ())
        if row.get("error_type") != "CandidateParseError"
    ]
    expected_checks = {
        "completed_all_periods": macro_count == episode_length,
        "action_count_t_by_n": len(records.get("actions", ()))
        == num_agents * episode_length,
        "episode_count_t_by_n": len(records.get("episodes", ()))
        == num_agents * episode_length,
        "utility_count_t_by_n": len(records.get("utility_ledger", ()))
        == num_agents * episode_length,
        "no_provider_errors": not provider_errors,
        "semantic_parse_outcomes_accounted": all(
            row.get("candidate_parse_status") in {"success", "failure"}
            for row in proposals
        ),
        "causal_context": all(
            _mapping(row.get("context_packet"), "context_trace.context_packet").get(
                "observed_through"
            )
            <= row.get("decision_t")
            for row in records.get("context_trace", ())
        ),
        "episode_alignment": all(
            row.get("outcome_t") == row.get("decision_t") + 1
            for row in records.get("episodes", ())
        ),
        "budget_identity": all(
            abs(float(row["budget_residual"]))
            <= utility_config.budget_tolerance
            for row in records.get("utility_ledger", ())
        ),
    }
    if config.get("schema_version") == RUNNER_SCHEMA_VERSION:
        unverified_rule_ids = {
            row.get("rule_id")
            for row in records.get("semantic_rules", ())
            if _mapping(
                row.get("injection_provenance") or {},
                "semantic_rules.injection_provenance",
            ).get("semantic_policy")
            == "unverified-immediate"
        }
        freeze_after = config.get("freeze_new_proposals_after")
        expected_checks.update(
            {
                "shock_schedule_applied_exactly": list(
                    records.get("shock_events", ())
                )
                == list(config.get("shock_schedule", ())),
                "no_future_shock_in_prompt": all(
                    row.get("shock_event") is None
                    or _mapping(
                        row.get("shock_event"),
                        "decision_snapshots.shock_event",
                    ).get("decision_t")
                    == row.get("decision_t")
                    for row in records.get("decision_snapshots", ())
                ),
                "proposal_freeze_respected": (
                    freeze_after is None
                    or all(
                        row.get("current_t") <= freeze_after
                        for row in records.get("semantic_proposals", ())
                    )
                ),
                "error_rule_injection_accounted": len(
                    records.get("error_rule_injections", ())
                )
                == (
                    0
                    if config.get("error_rule_mode") == "none"
                    else num_agents
                ),
                "unverified_policy_has_no_evidence_or_retirement": all(
                    not (
                        row.get("rule_id") in unverified_rule_ids
                        and (
                            str(row.get("event_type", "")).endswith(
                                "_evidence_added"
                            )
                            or row.get("event_type") == "rule_retired"
                        )
                    )
                    for row in records.get("semantic_rule_events", ())
                ),
            }
        )
    provider_models = {
        f"{row.get('provider')}/{row.get('model')}"
        for row in records.get("api_usage", ())
    }
    if len(provider_models) != 1:
        raise _contract_error("current run must use exactly one provider/model")
    provider_model = next(iter(provider_models))
    diagnostic_only = provider_model.startswith("diagnostic/")
    scientific_evidence = bool(
        config.get("schema_version") == RUNNER_SCHEMA_VERSION
        and all(expected_checks.values())
        and not diagnostic_only
        and config.get("scientific_scope")
        == "preregistered_mechanism_micro_pilot"
        and config.get("allow_scientific_scope") is True
        and config.get("pilot_contract_hash")
        and config.get("pilot_tag")
        and config.get("preflight_measurement_role") is None
        and serialized_has_sealed_observed_p95_authority(config)
    )
    expected_validation = {
        "status": "pass" if all(expected_checks.values()) else "fail",
        "checks": expected_checks,
        "diagnostic_only": diagnostic_only,
        "scientific_evidence": scientific_evidence,
    }
    if dict(validation_status) != expected_validation:
        raise _contract_error(
            "validation_status does not reproduce from the sealed streams"
        )
    if (
        summary.get("provider_model") != provider_model
        or summary.get("diagnostic_only") is not diagnostic_only
        or summary.get("scientific_evidence") is not scientific_evidence
        or summary.get("result_scope")
        != (
            (
                "preregistered_capability_preflight"
                if config.get("preflight_measurement_role") is not None
                else config.get("scientific_scope")
            )
            if config.get("schema_version") == RUNNER_SCHEMA_VERSION
            else "bounded_method_smoke"
        )
    ):
        raise _contract_error(
            "summary provider/evidence scope does not match the sealed run"
        )


def _validate_schema_versions(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    records: Mapping[str, tuple[Mapping[str, Any], ...]],
    *,
    for_write: bool,
) -> str:
    run_schema_version = config.get("schema_version")
    if for_write and run_schema_version != RUNNER_SCHEMA_VERSION:
        raise _contract_error(
            "new writes require exact current config schema "
            f"{RUNNER_SCHEMA_VERSION!r}; got {run_schema_version!r}"
        )
    if run_schema_version == RUNNER_SCHEMA_VERSION:
        stream_versions = _CURRENT_STREAM_SCHEMA_VERSIONS
    elif run_schema_version == PREVIOUS_RUNNER_SCHEMA_VERSION:
        stream_versions = _PREVIOUS_STREAM_SCHEMA_VERSIONS
    elif run_schema_version == LEGACY_RUNNER_SCHEMA_VERSION:
        stream_versions = _LEGACY_STREAM_SCHEMA_VERSIONS
    else:
        raise _contract_error(
            f"unsupported config schema_version {run_schema_version!r}"
        )

    if summary.get("schema_version") != run_schema_version:
        raise _contract_error(
            "summary schema_version must exactly match config schema_version"
        )

    current_schemas = {
        schema.name: schema
        for schema in verified_run_schemas(
            semantic_required=bool(config.get("enable_semantic")),
            run_schema_version=str(run_schema_version),
        )
    }
    for stream_name, expected_version in stream_versions.items():
        for index, row in enumerate(records.get(stream_name, ())):
            if not isinstance(row, Mapping):
                raise _contract_error(
                    f"{stream_name}[{index}] must be an object"
                )
            if row.get("schema_version") != expected_version:
                raise _contract_error(
                    f"{stream_name}[{index}].schema_version must be "
                    f"{expected_version!r}; got {row.get('schema_version')!r}"
                )
            # Current records must also satisfy the current structural schema.
            # Legacy snapshots predate a required prompt_schema_version field,
            # so their sealed v1 shape is handled by the explicit version map.
            if run_schema_version == RUNNER_SCHEMA_VERSION:
                current_schemas[stream_name].validate_record(row)
    if run_schema_version == RUNNER_SCHEMA_VERSION:
        current_schemas["summary"].validate_record(summary)
    return str(run_schema_version)


def _validate_cross_stream_contract(
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    validation_status: Mapping[str, Any],
    budget_snapshot: Mapping[str, Any],
    records: Mapping[str, tuple[Mapping[str, Any], ...]],
    manifest: Mapping[str, Any] | None,
    for_write: bool,
) -> None:
    """Fail closed when sealed summaries disagree with their source streams."""

    run_schema_version = _validate_schema_versions(
        config, summary, records, for_write=for_write
    )
    num_agents = _nonnegative_int(config.get("num_agents"), "config.num_agents")
    episode_length = _nonnegative_int(
        config.get("episode_length"), "config.episode_length"
    )
    if num_agents == 0 or episode_length == 0:
        raise _contract_error("num_agents and episode_length must be positive")
    _reconcile_count("summary.num_agents", summary.get("num_agents"), num_agents)
    _reconcile_count(
        "summary.episode_length", summary.get("episode_length"), episode_length
    )
    if summary.get("run_id") != config.get("run_id"):
        raise _contract_error("summary.run_id does not match config.run_id")
    if run_schema_version in _MODERN_RUNNER_SCHEMA_VERSIONS:
        _validate_current_foundation_contract(
            config,
            num_agents=num_agents,
            episode_length=episode_length,
        )

    macro_count = len(records.get("macro_steps", ()))
    result_complete = summary.get("result_complete")
    if not isinstance(result_complete, bool):
        raise _contract_error("summary.result_complete must be a boolean")
    if result_complete != (macro_count == episode_length):
        raise _contract_error(
            "summary.result_complete does not match the macro_steps stream"
        )
    expected_agent_rows = num_agents * macro_count
    for stream_name in (
        "actions",
        "context_trace",
        "decision_snapshots",
        "episodes",
        "utility_ledger",
    ):
        actual = len(records.get(stream_name, ()))
        if actual != expected_agent_rows:
            raise _contract_error(
                f"{stream_name} row count must equal num_agents * completed periods "
                f"({expected_agent_rows}); got {actual}"
            )

    for stream_name in (
        "actions",
        "context_trace",
        "decision_snapshots",
        "episodes",
    ):
        _reconcile_identity_grid(
            stream_name,
            records.get(stream_name, ()),
            time_field="decision_t",
            num_agents=num_agents,
            completed_periods=macro_count,
        )
    _reconcile_identity_grid(
        "utility_ledger",
        records.get("utility_ledger", ()),
        time_field="period",
        num_agents=num_agents,
        completed_periods=macro_count,
        string_agent_ids=True,
    )
    expected_macro_keys = [
        (decision_t, decision_t + 1) for decision_t in range(macro_count)
    ]
    actual_macro_keys = [
        (row.get("decision_t"), row.get("outcome_t"))
        for row in records.get("macro_steps", ())
    ]
    if actual_macro_keys != expected_macro_keys:
        raise _contract_error(
            "macro_steps must be the ordered causal sequence "
            "(decision_t=t, outcome_t=t+1)"
        )
    if run_schema_version in _MODERN_RUNNER_SCHEMA_VERSIONS and result_complete:
        expected_done = [False] * (macro_count - 1) + [True]
        actual_done = [row.get("done") for row in records.get("macro_steps", ())]
        if any(not isinstance(value, bool) for value in actual_done):
            raise _contract_error("macro_steps.done must be a boolean")
        if actual_done != expected_done:
            raise _contract_error(
                "complete fixed-duration macro_steps must have done=false before "
                "the final period and done=true at the final period"
            )
    for row in records.get("episodes", ()):
        decision_t = row.get("decision_t")
        outcome_t = row.get("outcome_t")
        if (
            isinstance(decision_t, bool)
            or not isinstance(decision_t, int)
            or outcome_t != decision_t + 1
        ):
            raise _contract_error("episodes must satisfy outcome_t = decision_t + 1")

    actions = records.get("actions", ())
    rules = records.get("semantic_rules", ())
    proposals = records.get("semantic_proposals", ())
    rule_events = records.get("semantic_rule_events", ())
    api_usage = records.get("api_usage", ())
    memory_diagnostics = _mapping(
        summary.get("memory_diagnostics"), "summary.memory_diagnostics"
    )

    declared_status_counts = _mapping(
        memory_diagnostics.get("semantic_rule_status_counts"),
        "summary.memory_diagnostics.semantic_rule_status_counts",
    )
    actual_status_counts = Counter(row.get("status") for row in rules)
    unknown_statuses = sorted(
        str(status) for status in actual_status_counts if status not in _RULE_STATUSES
    )
    if unknown_statuses:
        raise _contract_error(
            f"semantic_rules contains unknown statuses: {unknown_statuses}"
        )
    for status in _RULE_STATUSES:
        _reconcile_count(
            f"summary.memory_diagnostics.semantic_rule_status_counts.{status}",
            declared_status_counts.get(status),
            actual_status_counts[status],
        )
    if set(declared_status_counts) != set(_RULE_STATUSES):
        raise _contract_error(
            "semantic_rule_status_counts must contain exactly the registered statuses"
        )

    actual_active_retrievals = sum(
        len(_list_field(row, "selected_rule_ids", "actions")) for row in actions
    )
    _reconcile_count(
        "summary.memory_diagnostics.active_rule_retrieval_count",
        memory_diagnostics.get("active_rule_retrieval_count"),
        actual_active_retrievals,
    )
    actual_episode_retrievals = sum(
        len(_list_field(row, "retrieved_episode_ids", "actions")) for row in actions
    )
    _reconcile_count(
        "summary.memory_diagnostics.episodic_retrieval_count",
        memory_diagnostics.get("episodic_retrieval_count"),
        actual_episode_retrievals,
    )

    rules_by_identity: dict[tuple[Any, Any], Mapping[str, Any]] = {}
    for rule in rules:
        identity = (rule.get("agent_id"), rule.get("rule_id"))
        if identity in rules_by_identity:
            raise _contract_error(
                f"semantic_rules contains duplicate (agent_id, rule_id) {identity!r}"
            )
        rules_by_identity[identity] = rule

    lifecycle_status: dict[tuple[Any, Any], Any] = {}
    retrieval_events: Counter[tuple[Any, Any, Any]] = Counter()
    for event in rule_events:
        rule_id = event.get("rule_id")
        if rule_id is None:
            continue
        identity = (event.get("agent_id"), rule_id)
        if identity not in rules_by_identity:
            raise _contract_error(
                "semantic_rule_events references a rule absent from the same agent's "
                f"semantic_rules row: {identity!r}"
            )
        from_status = event.get("from_status")
        to_status = event.get("to_status")
        if identity in lifecycle_status:
            if from_status != lifecycle_status[identity]:
                raise _contract_error(
                    f"semantic rule lifecycle discontinuity for {identity!r}"
                )
        elif from_status is not None:
            raise _contract_error(
                f"semantic rule lifecycle for {identity!r} does not start from null"
            )
        if event.get("event_type") == "active_rule_retrieved":
            if from_status != "active" or to_status != "active":
                raise _contract_error(
                    "active_rule_retrieved event must preserve active status"
                )
            retrieval_events[
                (event.get("agent_id"), event.get("timestamp"), rule_id)
            ] += 1
        if to_status is not None:
            lifecycle_status[identity] = to_status
    for identity, rule in rules_by_identity.items():
        if lifecycle_status.get(identity) != rule.get("status"):
            raise _contract_error(
                f"final semantic rule status does not match lifecycle events for {identity!r}"
            )

    selected_rule_events: Counter[tuple[Any, Any, Any]] = Counter()
    for action in actions:
        for rule_id in _list_field(action, "selected_rule_ids", "actions"):
            identity = (action.get("agent_id"), rule_id)
            if identity not in rules_by_identity:
                raise _contract_error(
                    "actions selected a rule absent from the same agent's semantic_rules: "
                    f"{identity!r}"
                )
            selected_rule_events[
                (action.get("agent_id"), action.get("decision_t"), rule_id)
            ] += 1
    if selected_rule_events != retrieval_events:
        raise _contract_error(
            "selected_rule_ids must exactly match same-agent active_rule_retrieved "
            "lifecycle events at the decision timestamp"
        )

    semantic_api_calls = sum(row.get("call_kind") == "semantic" for row in api_usage)
    if semantic_api_calls != len(proposals):
        raise _contract_error(
            "semantic_proposals row count must match semantic api_usage calls; "
            f"got {len(proposals)} proposals and {semantic_api_calls} calls"
        )
    action_api_calls = sum(row.get("call_kind") == "action" for row in api_usage)
    if action_api_calls != len(actions):
        raise _contract_error(
            "actions row count must match action api_usage calls; "
            f"got {len(actions)} actions and {action_api_calls} calls"
        )
    if action_api_calls + semantic_api_calls != len(api_usage):
        raise _contract_error("api_usage contains an unregistered call_kind")
    summary_api = _mapping(summary.get("api"), "summary.api")
    _reconcile_count(
        "summary.api.completed_calls",
        summary_api.get("completed_calls"),
        len(api_usage),
    )
    _reconcile_count(
        "budget_snapshot.completed_calls",
        budget_snapshot.get("completed_calls"),
        len(api_usage),
    )
    summary_api_stable = dict(summary_api)
    budget_stable = dict(budget_snapshot)
    summary_api_stable.pop("elapsed_seconds", None)
    budget_stable.pop("elapsed_seconds", None)
    if summary_api_stable != budget_stable:
        raise _contract_error(
            "summary.api and budget_snapshot stable fields differ "
            "(only elapsed_seconds may advance)"
        )

    completion_rows = budget_snapshot.get("completions")
    if not isinstance(completion_rows, list):
        raise _contract_error("budget_snapshot.completions must be an array")
    api_identities = Counter(
        _api_identity(
            call_kind=row.get("call_kind"),
            decision_t=row.get("decision_t"),
            agent_id=row.get("agent_id"),
            model=f"{row.get('provider')}/{row.get('model')}",
            usage=row.get("usage"),
        )
        for row in api_usage
    )
    completion_identities: Counter[tuple[Any, ...]] = Counter()
    for completion in completion_rows:
        completion = _mapping(completion, "budget_snapshot.completions[]")
        tags = _mapping(
            completion.get("tags"), "budget_snapshot.completions[].tags"
        )
        call_kind = tags.get("call_kind")
        time_key = "current_t" if call_kind == "semantic" else "decision_t"
        raw_time = tags.get(time_key)
        try:
            decision_t = int(raw_time)
        except (TypeError, ValueError) as exc:
            raise _contract_error(
                "budget completion tags contain an invalid decision timestamp"
            ) from exc
        completion_identities[
            _api_identity(
                call_kind=call_kind,
                decision_t=decision_t,
                agent_id=tags.get("agent_id"),
                model=completion.get("model"),
                usage=completion.get("usage"),
            )
        ] += 1
    if completion_identities != api_identities:
        raise _contract_error(
            "budget completion identities/usages do not match api_usage rows"
        )

    proposal_call_keys = Counter(
        (row.get("current_t"), row.get("agent_id")) for row in proposals
    )
    semantic_call_keys = Counter(
        (row.get("decision_t"), row.get("agent_id"))
        for row in api_usage
        if row.get("call_kind") == "semantic"
    )
    if proposal_call_keys != semantic_call_keys:
        raise _contract_error(
            "semantic proposal identities do not match semantic api_usage identities"
        )
    action_record_keys = Counter(
        (row.get("decision_t"), row.get("agent_id")) for row in actions
    )
    action_call_keys = Counter(
        (row.get("decision_t"), row.get("agent_id"))
        for row in api_usage
        if row.get("call_kind") == "action"
    )
    if action_record_keys != action_call_keys:
        raise _contract_error(
            "action identities do not match action api_usage identities"
        )

    if run_schema_version in _MODERN_RUNNER_SCHEMA_VERSIONS:
        parse_summary = _mapping(
            memory_diagnostics.get("semantic_candidate_parse"),
            "summary.memory_diagnostics.semantic_candidate_parse",
        )
        parse_statuses = Counter(row.get("candidate_parse_status") for row in proposals)
        if set(parse_statuses) - {"success", "failure"}:
            raise _contract_error(
                "completed semantic_proposals must have success or failure parse status"
            )
        _reconcile_count(
            "summary.memory_diagnostics.semantic_candidate_parse.attempt_count",
            parse_summary.get("attempt_count"),
            len(proposals),
        )
        _reconcile_count(
            "summary.memory_diagnostics.semantic_candidate_parse.success_count",
            parse_summary.get("success_count"),
            parse_statuses["success"],
        )
        _reconcile_count(
            "summary.memory_diagnostics.semantic_candidate_parse.failure_count",
            parse_summary.get("failure_count"),
            parse_statuses["failure"],
        )
        expected_failure_rate = (
            parse_statuses["failure"] / len(proposals) if proposals else 0.0
        )
        failure_rate = parse_summary.get("failure_rate")
        if (
            isinstance(failure_rate, bool)
            or not isinstance(failure_rate, (int, float))
            or not math.isfinite(float(failure_rate))
            or not math.isclose(
                float(failure_rate), expected_failure_rate, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise _contract_error(
                "semantic_candidate_parse.failure_rate does not match proposal rows"
            )
        declared_mode_counts = _mapping(
            parse_summary.get("mode_counts"),
            "summary.memory_diagnostics.semantic_candidate_parse.mode_counts",
        )
        actual_mode_counts = Counter(row.get("candidate_parse_mode") for row in proposals)
        if set(actual_mode_counts) - set(_PARSE_MODES):
            raise _contract_error(
                "semantic_proposals contains an unregistered candidate_parse_mode"
            )
        for mode in _PARSE_MODES:
            _reconcile_count(
                "summary.memory_diagnostics.semantic_candidate_parse."
                f"mode_counts.{mode}",
                declared_mode_counts.get(mode),
                actual_mode_counts[mode],
            )
        if set(declared_mode_counts) != set(_PARSE_MODES):
            raise _contract_error(
                "semantic_candidate_parse.mode_counts must contain exactly the "
                "registered modes"
            )

        failed_proposal_keys: Counter[tuple[Any, Any]] = Counter()
        parse_rejection_event_keys = Counter(
            (event.get("timestamp"), event.get("agent_id"))
            for event in rule_events
            if event.get("event_type") == "candidate_parse_rejected"
            and event.get("rule_id") is None
        )
        for proposal in proposals:
            status = proposal.get("candidate_parse_status")
            mode = proposal.get("candidate_parse_mode")
            rule_id = proposal.get("rule_id")
            rule_status = proposal.get("rule_status")
            parse_error = proposal.get("parse_error")
            identity = (proposal.get("agent_id"), rule_id)
            if proposal.get("provider_error") is not None:
                raise _contract_error(
                    "completed semantic proposal rows cannot contain provider_error"
                )
            if status == "success":
                if (
                    not isinstance(rule_id, str)
                    or identity not in rules_by_identity
                    or rule_status not in _RULE_STATUSES
                    or parse_error is not None
                    or mode == "parse_failure"
                ):
                    raise _contract_error(
                        "successful semantic proposal must link to a same-agent rule "
                        "and have non-failure parse provenance"
                    )
                semantic_policy = proposal.get(
                    "semantic_policy", "evidence-grounded"
                )
                if semantic_policy == "unverified-immediate":
                    matching_candidate_event = any(
                        event.get("agent_id") == proposal.get("agent_id")
                        and event.get("timestamp") == proposal.get("current_t")
                        and event.get("rule_id") == rule_id
                        and event.get("event_type")
                        in {
                            "experimental_rule_injected_active",
                            "duplicate_unverified_candidate_ignored",
                        }
                        and _mapping(
                            event.get("provenance"),
                            "semantic_rule_events.provenance",
                        ).get("source_candidate_id")
                        is not None
                        and event.get("to_status") == "active"
                        and rule_status == "active"
                        for event in rule_events
                    )
                elif semantic_policy == "evidence-grounded":
                    matching_candidate_event = any(
                        event.get("agent_id") == proposal.get("agent_id")
                        and event.get("timestamp") == proposal.get("current_t")
                        and event.get("rule_id") == rule_id
                        and event.get("candidate_id") is not None
                        and event.get("to_status") == rule_status
                        for event in rule_events
                    )
                else:
                    raise _contract_error(
                        "semantic proposal has an unknown semantic_policy"
                    )
                if not matching_candidate_event:
                    raise _contract_error(
                        "successful semantic proposal lacks its candidate lifecycle event"
                    )
            elif status == "failure":
                if (
                    rule_id is not None
                    or rule_status is not None
                    or not isinstance(parse_error, str)
                    or not parse_error
                    or mode != "parse_failure"
                ):
                    raise _contract_error(
                        "failed semantic proposal must preserve parse error and have no rule"
                    )
                failed_proposal_keys[
                    (proposal.get("current_t"), proposal.get("agent_id"))
                ] += 1
        if failed_proposal_keys != parse_rejection_event_keys:
            raise _contract_error(
                "failed semantic proposals must exactly match candidate_parse_rejected "
                "events"
            )

        selected_rule_ids = {
            rule_id
            for row in actions
            for rule_id in _list_field(row, "selected_rule_ids", "actions")
        }
        expected_activation = bool(selected_rule_ids)
        if memory_diagnostics.get("semantic_activation_observed") is not expected_activation:
            raise _contract_error(
                "semantic_activation_observed does not match rule/action streams"
            )
        _validate_current_identity_bindings(config, records)
        _validate_current_utility_and_result_contract(
            config,
            summary,
            validation_status,
            records,
            num_agents=num_agents,
            macro_count=macro_count,
            episode_length=episode_length,
        )

    action_diagnostics = _mapping(
        summary.get("action_diagnostics"), "summary.action_diagnostics"
    )
    executed_hours = [
        float(_mapping(row.get("decision"), "actions.decision")["executed_labor_hours"])
        for row in actions
    ]
    max_labor_hours = float(config.get("max_labor_hours"))
    _reconcile_count(
        "summary.action_diagnostics.intermediate_action_count",
        action_diagnostics.get("intermediate_action_count"),
        sum(0.0 < hours < max_labor_hours for hours in executed_hours),
    )
    _reconcile_count(
        "summary.action_diagnostics.clipped_action_count",
        action_diagnostics.get("clipped_action_count"),
        sum(bool(_mapping(row.get("decision"), "actions.decision").get("clipped")) for row in actions),
    )
    if action_diagnostics.get("unique_labor_hours") != sorted(set(executed_hours)):
        raise _contract_error("unique_labor_hours does not match the actions stream")
    if "ceiling_labor_count" in action_diagnostics:
        _reconcile_count(
            "summary.action_diagnostics.ceiling_labor_count",
            action_diagnostics.get("ceiling_labor_count"),
            sum(hours == max_labor_hours for hours in executed_hours),
        )
    if "labor_hours_counts" in action_diagnostics:
        declared_labor_counts = _mapping(
            action_diagnostics.get("labor_hours_counts"),
            "summary.action_diagnostics.labor_hours_counts",
        )
        actual_labor_counts = Counter(f"{hours:g}" for hours in executed_hours)
        if dict(declared_labor_counts) != dict(actual_labor_counts):
            raise _contract_error("labor_hours_counts does not match the actions stream")
    if run_schema_version in _MODERN_RUNNER_SCHEMA_VERSIONS:
        expected_intermediate = sum(
            0.0 < hours < max_labor_hours for hours in executed_hours
        )
        expected_ceiling = sum(hours == max_labor_hours for hours in executed_hours)
        expected_action_diagnostic_keys = {
            "unique_labor_hours",
            "intermediate_action_count",
            "intermediate_action_observed",
            "labor_hours_counts",
            "ceiling_labor_count",
            "ceiling_labor_rate",
            "clipped_action_count",
        }
        if set(action_diagnostics) != expected_action_diagnostic_keys:
            raise _contract_error(
                "summary.action_diagnostics must contain exactly the registered fields"
            )
        if action_diagnostics.get("intermediate_action_observed") is not bool(
            expected_intermediate
        ):
            raise _contract_error(
                "intermediate_action_observed does not match the actions stream"
            )
        _require_close(
            "summary.action_diagnostics.ceiling_labor_rate",
            action_diagnostics.get("ceiling_labor_rate"),
            expected_ceiling / len(executed_hours),
        )

    if dict(summary.get("validation", {})) != dict(validation_status):
        raise _contract_error("summary.validation does not match validation_status")

    if manifest is not None:
        manifest_result = _mapping(manifest.get("result"), "manifest.result")
        if manifest_result.get("complete") is not result_complete:
            raise _contract_error(
                "manifest.result.complete does not match summary.result_complete"
            )
        declared_stream_counts = _mapping(
            manifest_result.get("stream_line_counts"),
            "manifest.result.stream_line_counts",
        )
        actual_stream_counts = {
            schema.name: (1 if schema.name == "summary" else len(records.get(schema.name, ())))
            for schema in verified_run_schemas(
                semantic_required=bool(config.get("enable_semantic")),
                run_schema_version=str(config.get("schema_version")),
            )
        }
        if dict(declared_stream_counts) != actual_stream_counts:
            raise _contract_error(
                "manifest stream_line_counts does not match loaded streams"
            )


def write_verified_run_artifacts(
    run_dir: str | Path,
    result: VerifiedRunResult,
    *,
    provenance: Mapping[str, Any],
    git_commit: str,
    git_dirty: bool,
) -> Path:
    """Persist, seal, and independently re-hash one completed run."""

    if not isinstance(result, VerifiedRunResult):
        raise TypeError("result must be VerifiedRunResult")
    _validate_cross_stream_contract(
        config=result.config,
        summary=result.summary,
        validation_status=result.validation_status,
        budget_snapshot=result.budget_snapshot,
        records=result.records,
        manifest=None,
        for_write=True,
    )
    semantic_required = bool(result.config.get("enable_semantic"))
    writer = RunArtifactWriter.create(
        Path(run_dir),
        verified_run_schemas(
            semantic_required=semantic_required,
            run_schema_version=RUNNER_SCHEMA_VERSION,
        ),
        config=result.config,
        provenance=dict(provenance),
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    for stream_name, rows in result.records.items():
        if stream_name not in {schema.name for schema in verified_run_schemas(
            semantic_required=semantic_required,
            run_schema_version=RUNNER_SCHEMA_VERSION,
        )}:
            raise ValueError(f"runner produced undeclared stream: {stream_name}")
        for row in rows:
            writer.append(stream_name, row)
    writer.append("summary", result.summary)
    manifest_path = writer.finalize(
        validation_status=result.validation_status,
        budget_snapshot=result.budget_snapshot,
        result_complete=bool(result.summary["result_complete"]),
    )
    verification = verify_manifest(Path(run_dir))
    if not verification.valid:
        raise RuntimeError("artifact manifest verification unexpectedly failed")
    return manifest_path


def load_verified_run_artifacts(run_dir: str | Path) -> VerifiedRunResult:
    """Load a sealed run only after its manifest and every file hash verify."""

    root = Path(run_dir)
    verify_manifest(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records: dict[str, tuple[Mapping[str, Any], ...]] = {}
    summary: Mapping[str, Any] | None = None
    for schema in verified_run_schemas(
        semantic_required=bool(config.get("enable_semantic")),
        run_schema_version=str(config.get("schema_version")),
    ):
        path = root / schema.relative_path
        rows: tuple[Mapping[str, Any], ...] = ()
        if path.exists():
            rows = tuple(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            )
        if schema.name == "summary":
            if len(rows) != 1:
                raise ValueError("sealed verified run must contain exactly one summary")
            summary = rows[0]
        else:
            records[schema.name] = rows
    if summary is None:
        raise ValueError("sealed verified run is missing summary")
    result = VerifiedRunResult(
        config=config,
        summary=summary,
        validation_status=manifest["validation_status"],
        budget_snapshot=manifest["budget_snapshot"],
        records=records,
    )
    _validate_cross_stream_contract(
        config=result.config,
        summary=result.summary,
        validation_status=result.validation_status,
        budget_snapshot=result.budget_snapshot,
        records=result.records,
        manifest=manifest,
        for_write=False,
    )
    return result


__all__ = [
    "load_verified_run_artifacts",
    "verified_run_schemas",
    "write_verified_run_artifacts",
]
