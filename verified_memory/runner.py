"""End-to-end verified-memory simulation runner.

This is a new execution path.  It intentionally does not call or mutate the legacy
``simulate.py`` pipeline, so existing paper runs remain reproducible.  The runner
uses direct labor hours, causal context, aligned M2 transitions, verified M3 rules,
an ex-post M0 utility ledger, and structured provider accounting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import re
from statistics import mean
from typing import Any, Mapping, Optional

import numpy as np

import ai_economist.foundation as foundation
from llm_providers import MODEL_COSTS, MultiModelLLM, StructuredCompletion

from .actions import ActionDecision, parse_direct_action
from .budget import RunBudget, UsageRecord
from .foundation_adapter import (
    capture_foundation_snapshots,
    build_foundation_actions,
    derive_foundation_transitions,
    prepare_foundation_env_config,
)
from .m0_utility import EnvironmentLedger, UtilityConfig
from .m1_context import CONTEXT_MODES, CausalContextRouter
from .m3_semantic import (
    DEFAULT_EVIDENCE_WEIGHTS,
    DEFAULT_REGISTERED_OUTCOME_CRITERION,
    ActionGuidance,
    CandidateParseError,
    ConditionPredicate,
    ContextScope,
    OutcomeCriterion,
)
from .prompts import (
    PROMPT_SCHEMA_VERSION,
    DecisionPromptState,
    build_base_decision_prompt,
    compose_decision_prompt,
)
from .scripted_provider import DIAGNOSTIC_PROVIDER_NAME
from .system import MemoryBundle, VerifiedDualTrackMemory


RUNNER_SCHEMA_VERSION = "verified-simulation-runner-v3"
SHOCK_EVENT_SCHEMA_VERSION = "finevo-shock-event-v1"
ERROR_RULE_INJECTION_SCHEMA_VERSION = "finevo-error-rule-injection-v1"
SEMANTIC_PARSE_FAILURE_POLICIES = frozenset({"record-and-skip", "fail-run"})
SEMANTIC_POLICIES = frozenset({"evidence-grounded", "unverified-immediate"})
ERROR_RULE_MODES = frozenset({"none", "candidate-admission", "forced-active"})
SCIENTIFIC_SCOPES = frozenset(
    {"bounded_method_smoke", "preregistered_mechanism_micro_pilot"}
)
PREFLIGHT_P95_CALL_KINDS = frozenset({"action", "semantic"})
PREFLIGHT_P95_RESERVE_MULTIPLIER = 1.25
CONTEXT_FEATURES = (
    "log_price",
    "interest_rate",
    "prior_low_labor_rate",
    "prior_low_labor_rate_available",
    "inflation",
    "log_wealth",
    "employed",
)

FIXED_ERRONEOUS_RULE = {
    "context_scope": {"scope_id": "global", "predicates": []},
    "condition": {
        "field": "interest_rate",
        "operator": ">=",
        "value": 0.0,
        "tolerance": 0.0,
    },
    "action_guidance": {
        "target": "consumption_fraction",
        "direction": "at_most",
        "threshold": 0.0,
        "tolerance": 0.0,
    },
    "rationale": (
        "Preregistered erroneous control rule: whenever the interest rate is "
        "nonnegative, consume none of available cash."
    ),
}


@dataclass(frozen=True, slots=True)
class RunnerFailure:
    """Structured failure details preserved by the CLI failure receipt."""

    schema_version: str
    error_stage: str
    call_kind: str
    decision_t: int
    agent_id: int
    error_type: str
    message: str
    prompt_hash: str
    raw_output_hash: str
    provider: str
    model: str
    attempts: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VerifiedRunError(RuntimeError):
    """Raised when a fail-closed simulation gate is violated."""

    def __init__(self, message: str, *, failure: RunnerFailure | None = None) -> None:
        self.failure = failure
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PreflightP95Reservation:
    """One immutable observed-p95 reservation bound to a model and call kind.

    Token p95 values can be fractional for a small calibration sample, while
    :class:`UsageRecord` deliberately permits only integral token counts.  The
    raw values are therefore retained separately and the dispatch reservation
    is the ceiling of ``raw_p95 * 1.25``.  This makes the exact preflight basis
    and the conservative reservation independently auditable.
    """

    model: str
    call_kind: str
    sample_count: int
    raw_prompt_tokens: float
    raw_completion_tokens: float
    raw_cost_usd: float
    reserved_usage: UsageRecord
    reserve_multiplier: float = PREFLIGHT_P95_RESERVE_MULTIPLIER

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("preflight p95 model must be a non-empty string")
        object.__setattr__(self, "model", self.model.strip())
        if not isinstance(self.call_kind, str):
            raise TypeError("preflight p95 call_kind must be a string")
        call_kind = self.call_kind.strip().lower().replace("_", "-")
        if call_kind not in PREFLIGHT_P95_CALL_KINDS:
            raise ValueError(
                "preflight p95 call_kind must be action or semantic"
            )
        object.__setattr__(self, "call_kind", call_kind)
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 1
        ):
            raise ValueError("preflight p95 sample_count must be a positive integer")
        for name in (
            "raw_prompt_tokens",
            "raw_completion_tokens",
            "raw_cost_usd",
            "reserve_multiplier",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, numeric)
        if not math.isclose(
            self.reserve_multiplier,
            PREFLIGHT_P95_RESERVE_MULTIPLIER,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("preflight p95 reserve_multiplier must equal 1.25")
        if not isinstance(self.reserved_usage, UsageRecord):
            raise TypeError("preflight p95 reserved_usage must be a UsageRecord")
        expected_prompt = math.ceil(
            self.raw_prompt_tokens * self.reserve_multiplier
        )
        expected_completion = math.ceil(
            self.raw_completion_tokens * self.reserve_multiplier
        )
        expected_cost = self.raw_cost_usd * self.reserve_multiplier
        if self.reserved_usage.prompt_tokens != expected_prompt:
            raise ValueError(
                "preflight p95 prompt reservation must ceil raw_p95 * 1.25"
            )
        if self.reserved_usage.completion_tokens != expected_completion:
            raise ValueError(
                "preflight p95 completion reservation must ceil raw_p95 * 1.25"
            )
        if not math.isclose(
            self.reserved_usage.cost_usd,
            expected_cost,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "preflight p95 cost reservation must equal raw_p95 * 1.25"
            )

    @classmethod
    def from_dict(
        cls,
        *,
        model: str,
        call_kind: str,
        value: Mapping[str, Any],
    ) -> "PreflightP95Reservation":
        if not isinstance(value, Mapping):
            raise TypeError("preflight p95 reservation entry must be a mapping")
        expected = {
            "sample_count",
            "raw_p95",
            "reserved_p95",
            "reserve_multiplier",
        }
        if set(value) != expected:
            raise ValueError(
                "preflight p95 reservation entry must contain exactly "
                f"{sorted(expected)}"
            )
        raw = value["raw_p95"]
        reserved = value["reserved_p95"]
        if not isinstance(raw, Mapping) or not isinstance(reserved, Mapping):
            raise TypeError("preflight raw_p95 and reserved_p95 must be mappings")
        for name, payload in (("raw_p95", raw), ("reserved_p95", reserved)):
            required = {
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cost_usd",
            }
            if set(payload) != required:
                raise ValueError(
                    f"preflight {name} must contain exactly {sorted(required)}"
                )
        if (
            isinstance(value["sample_count"], bool)
            or not isinstance(value["sample_count"], int)
        ):
            raise TypeError("preflight sample_count must be an integer")
        for field_name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cost_usd",
        ):
            raw_value = raw[field_name]
            if isinstance(raw_value, bool) or not isinstance(
                raw_value, (int, float)
            ):
                raise TypeError(f"preflight raw_p95 {field_name} must be numeric")
        for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            reserved_value = reserved[field_name]
            if isinstance(reserved_value, bool) or not isinstance(
                reserved_value, int
            ):
                raise TypeError(
                    f"preflight reserved_p95 {field_name} must be an integer"
                )
        if isinstance(reserved["cost_usd"], bool) or not isinstance(
            reserved["cost_usd"], (int, float)
        ):
            raise TypeError("preflight reserved_p95 cost_usd must be numeric")
        raw_total = float(raw["prompt_tokens"]) + float(raw["completion_tokens"])
        if not math.isclose(
            float(raw["total_tokens"]),
            raw_total,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("preflight raw_p95 total_tokens is inconsistent")
        reserved_usage = UsageRecord(
            prompt_tokens=int(reserved["prompt_tokens"]),
            completion_tokens=int(reserved["completion_tokens"]),
            cost_usd=float(reserved["cost_usd"]),
        )
        if int(reserved["total_tokens"]) != reserved_usage.total_tokens:
            raise ValueError("preflight reserved_p95 total_tokens is inconsistent")
        return cls(
            model=model,
            call_kind=call_kind,
            sample_count=value["sample_count"],
            raw_prompt_tokens=float(raw["prompt_tokens"]),
            raw_completion_tokens=float(raw["completion_tokens"]),
            raw_cost_usd=float(raw["cost_usd"]),
            reserved_usage=reserved_usage,
            reserve_multiplier=float(value["reserve_multiplier"]),
        )

    def to_dict(self) -> dict[str, Any]:
        raw_total = self.raw_prompt_tokens + self.raw_completion_tokens
        return {
            "sample_count": self.sample_count,
            "raw_p95": {
                "prompt_tokens": self.raw_prompt_tokens,
                "completion_tokens": self.raw_completion_tokens,
                "total_tokens": raw_total,
                "cost_usd": self.raw_cost_usd,
            },
            "reserved_p95": self.reserved_usage.to_dict(),
            "reserve_multiplier": self.reserve_multiplier,
        }


def _normalize_preflight_p95_reservations(
    value: Any,
) -> tuple[PreflightP95Reservation, ...]:
    if value is None:
        return ()
    entries: list[PreflightP95Reservation] = []
    if isinstance(value, Mapping):
        for model, by_kind in value.items():
            if not isinstance(by_kind, Mapping):
                raise TypeError(
                    "preflight_p95_reservations model values must be mappings"
                )
            for call_kind, item in by_kind.items():
                entries.append(
                    PreflightP95Reservation.from_dict(
                        model=str(model),
                        call_kind=str(call_kind),
                        value=item,
                    )
                )
    else:
        if isinstance(value, (str, bytes)):
            raise TypeError(
                "preflight_p95_reservations must be a mapping or iterable"
            )
        try:
            raw_entries = tuple(value)
        except TypeError as exc:
            raise TypeError(
                "preflight_p95_reservations must be a mapping or iterable"
            ) from exc
        if any(not isinstance(item, PreflightP95Reservation) for item in raw_entries):
            raise TypeError(
                "preflight_p95_reservations iterable must contain "
                "PreflightP95Reservation values"
            )
        entries.extend(raw_entries)
    keys = [(item.model, item.call_kind) for item in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate model x call-kind preflight p95 reservation")
    return tuple(sorted(entries, key=lambda item: (item.model, item.call_kind)))


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _gini(values: list[float]) -> float:
    ordered = sorted(max(float(value), 0.0) for value in values)
    if not ordered or sum(ordered) == 0:
        return 0.0
    n = len(ordered)
    total = sum(ordered)
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    return 2 * weighted / (n * total) - (n + 1) / n


@dataclass(frozen=True, slots=True)
class ShockEvent:
    """One causally scheduled interest-rate intervention.

    The exact same current-period event is applied before prompt construction
    and immediately before ``env.step``.  Future schedule rows are never passed
    to the prompt, M1 packet, or environment hook.
    """

    decision_t: int
    phase: str
    interest_rate: float
    applied_before_prompt: bool = True
    applied_before_step: bool = True
    schema_version: str = SHOCK_EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SHOCK_EVENT_SCHEMA_VERSION:
            raise ValueError("unsupported shock-event schema version")
        if isinstance(self.decision_t, bool) or not isinstance(self.decision_t, int):
            raise TypeError("shock decision_t must be an integer")
        if self.decision_t < 0:
            raise ValueError("shock decision_t must be nonnegative")
        if not isinstance(self.phase, str) or not self.phase.strip():
            raise ValueError("shock phase must be a non-empty string")
        object.__setattr__(self, "phase", " ".join(self.phase.split()))
        if isinstance(self.interest_rate, bool) or not isinstance(
            self.interest_rate, (int, float)
        ):
            raise TypeError("shock interest_rate must be numeric")
        rate = float(self.interest_rate)
        if not math.isfinite(rate) or rate < 0:
            raise ValueError("shock interest_rate must be finite and nonnegative")
        object.__setattr__(self, "interest_rate", rate)
        if self.applied_before_prompt is not True:
            raise ValueError("shock must be applied before prompt construction")
        if self.applied_before_step is not True:
            raise ValueError("shock must be applied immediately before env.step")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision_t": self.decision_t,
            "phase": self.phase,
            "interest_rate": self.interest_rate,
            "applied_before_prompt": self.applied_before_prompt,
            "applied_before_step": self.applied_before_step,
        }

    def to_prompt_text(self) -> str:
        return (
            f"Preregistered current phase is {self.phase}; the current savings "
            f"interest rate is set to {self.interest_rate:.4%}."
        )


@dataclass(frozen=True, slots=True)
class VerifiedRunConfig:
    run_id: str
    seed: int = 7
    num_agents: int = 2
    episode_length: int = 6
    context_mode: str = "retrieval-only"
    enable_episodic_retrieval: bool = True
    enable_semantic: bool = True
    retrieval_k: int = 5
    episodic_prompt_capacity: int = 24
    rule_budget: int = 3
    semantic_proposal_after: int = 3
    semantic_proposal_interval: int = 3
    max_rule_proposals_per_agent: int = 1
    freeze_new_proposals_after: Optional[int] = None
    semantic_policy: str = "evidence-grounded"
    error_rule_mode: str = "none"
    error_rule_injection_t: int = 5
    labor_step: float = 8.0
    max_labor_hours: float = 168.0
    consumption_step: float = 0.02
    low_labor_threshold_hours: float = 1.0
    send_decoding_seed: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    action_max_tokens: int = 220
    rule_max_tokens: int = 450
    max_retries: int = 1
    fail_on_clipped_action: bool = True
    semantic_parse_failure_policy: str = "record-and-skip"
    min_candidate_support: int = 2
    activation_min_support: int = 3
    activation_min_margin: float = 1.0
    activation_confidence_threshold: float = 0.60
    proposal_confidence_floor: float = 0.50
    retirement_patience: int = 2
    retirement_confidence_threshold: float = 0.45
    evidence_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_EVIDENCE_WEIGHTS)
    )
    registered_outcome_criterion: OutcomeCriterion = field(
        default_factory=lambda: DEFAULT_REGISTERED_OUTCOME_CRITERION
    )
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    shock_schedule: tuple[ShockEvent, ...] = ()
    scientific_scope: str = "bounded_method_smoke"
    pilot_contract_hash: Optional[str] = None
    pilot_tag: Optional[str] = None
    allow_scientific_scope: bool = False
    preflight_p95_reservations: tuple[PreflightP95Reservation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        for name in (
            "seed",
            "num_agents",
            "episode_length",
            "retrieval_k",
            "episodic_prompt_capacity",
            "rule_budget",
            "semantic_proposal_after",
            "semantic_proposal_interval",
            "max_rule_proposals_per_agent",
            "error_rule_injection_t",
            "min_candidate_support",
            "activation_min_support",
            "retirement_patience",
            "action_max_tokens",
            "rule_max_tokens",
            "max_retries",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.num_agents < 2:
            raise ValueError("Foundation requires num_agents >= 2")
        if self.episode_length < 1:
            raise ValueError("episode_length must be positive")
        if self.retrieval_k < 0 or self.rule_budget < 0:
            raise ValueError("retrieval_k and rule_budget must be nonnegative")
        if self.episodic_prompt_capacity < 1:
            raise ValueError("episodic_prompt_capacity must be positive")
        if self.semantic_proposal_after < 2:
            raise ValueError("semantic proposals require at least two completed periods")
        for name in (
            "semantic_proposal_interval",
            "max_rule_proposals_per_agent",
            "min_candidate_support",
            "activation_min_support",
            "retirement_patience",
            "action_max_tokens",
            "rule_max_tokens",
            "max_retries",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_retries != 1:
            raise ValueError(
                "verified runs require max_retries=1 so each provider attempt "
                "consumes one hard-budget call"
            )
        if self.activation_min_support < self.min_candidate_support:
            raise ValueError(
                "activation_min_support cannot be below min_candidate_support"
            )
        if self.error_rule_injection_t < 0:
            raise ValueError("error_rule_injection_t must be nonnegative")
        if self.freeze_new_proposals_after is not None:
            if (
                isinstance(self.freeze_new_proposals_after, bool)
                or not isinstance(self.freeze_new_proposals_after, int)
                or self.freeze_new_proposals_after < 0
            ):
                raise ValueError(
                    "freeze_new_proposals_after must be a nonnegative integer or None"
                )
        normalized_mode = self.context_mode.strip().lower().replace("_", "-")
        if normalized_mode not in CONTEXT_MODES:
            raise ValueError(f"unsupported context mode: {self.context_mode}")
        object.__setattr__(self, "context_mode", normalized_mode)
        if not isinstance(self.semantic_parse_failure_policy, str):
            raise TypeError("semantic_parse_failure_policy must be a string")
        normalized_parse_policy = (
            self.semantic_parse_failure_policy.strip().lower().replace("_", "-")
        )
        if normalized_parse_policy not in SEMANTIC_PARSE_FAILURE_POLICIES:
            raise ValueError(
                "semantic_parse_failure_policy must be one of "
                f"{sorted(SEMANTIC_PARSE_FAILURE_POLICIES)}"
            )
        object.__setattr__(
            self, "semantic_parse_failure_policy", normalized_parse_policy
        )
        for name, allowed in (
            ("semantic_policy", SEMANTIC_POLICIES),
            ("error_rule_mode", ERROR_RULE_MODES),
            ("scientific_scope", SCIENTIFIC_SCOPES),
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            normalized = value.strip().lower().replace("_", "-")
            if name == "scientific_scope":
                normalized = normalized.replace("-", "_")
            if normalized not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}")
            object.__setattr__(self, name, normalized)
        if self.error_rule_mode != "none":
            if not self.enable_semantic:
                raise ValueError("error-rule injection requires semantic memory")
            if self.error_rule_injection_t >= self.episode_length:
                raise ValueError(
                    "error_rule_injection_t must be inside the simulated horizon"
                )
        if (
            self.error_rule_mode == "candidate-admission"
            and self.semantic_policy != "evidence-grounded"
        ):
            raise ValueError(
                "candidate-admission is the evidence-grounded verifier arm; "
                "unverified controls must use forced-active"
            )
        for name in (
            "labor_step",
            "max_labor_hours",
            "consumption_step",
            "low_labor_threshold_hours",
            "temperature",
            "top_p",
            "activation_min_margin",
            "activation_confidence_threshold",
            "proposal_confidence_floor",
            "retirement_confidence_threshold",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.labor_step <= 0 or self.max_labor_hours <= 0:
            raise ValueError("labor step and maximum must be positive")
        if not math.isclose(
            self.max_labor_hours / self.labor_step,
            round(self.max_labor_hours / self.labor_step),
            abs_tol=1e-12,
        ):
            raise ValueError("labor_step must divide max_labor_hours")
        if self.consumption_step <= 0 or self.consumption_step > 1:
            raise ValueError("consumption_step must lie in (0, 1]")
        if not math.isclose(
            1.0 / self.consumption_step,
            round(1.0 / self.consumption_step),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("consumption_step must divide one")
        if not 0 <= self.low_labor_threshold_hours <= self.max_labor_hours:
            raise ValueError("low labor threshold is outside feasible hours")
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError("invalid decoding parameters")
        for name in (
            "activation_confidence_threshold",
            "proposal_confidence_floor",
            "retirement_confidence_threshold",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not isinstance(self.utility, UtilityConfig):
            raise TypeError("utility must be UtilityConfig")
        if not isinstance(self.registered_outcome_criterion, OutcomeCriterion):
            raise TypeError("registered_outcome_criterion must be an OutcomeCriterion")
        if not math.isclose(
            self.utility.max_labor_hours, self.max_labor_hours, abs_tol=1e-12
        ):
            raise ValueError("utility and action maximum labor hours must match")
        if not isinstance(self.evidence_weights, Mapping):
            raise TypeError("evidence_weights must be a mapping")
        if set(self.evidence_weights) != set(DEFAULT_EVIDENCE_WEIGHTS):
            raise ValueError(
                "evidence_weights must define the exact registered evidence taxonomy"
            )
        normalized_weights: dict[str, float] = {}
        for evidence_type, value in self.evidence_weights.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"evidence weight {evidence_type!r} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(
                    f"evidence weight {evidence_type!r} must be finite and nonnegative"
                )
            normalized_weights[str(evidence_type)] = numeric
        if (
            normalized_weights["support"] <= 0
            or normalized_weights["harmful_compliance"] <= 0
            or normalized_weights["alternative_success"]
            > normalized_weights["harmful_compliance"]
            or normalized_weights["alternative_failure"] != 0
            or normalized_weights["irrelevant"] != 0
        ):
            raise ValueError("evidence_weights violate verifier taxonomy constraints")
        object.__setattr__(
            self,
            "evidence_weights",
            dict(sorted(normalized_weights.items())),
        )
        if not isinstance(self.shock_schedule, tuple):
            raise TypeError("shock_schedule must be a tuple of ShockEvent values")
        if any(not isinstance(event, ShockEvent) for event in self.shock_schedule):
            raise TypeError("shock_schedule must contain only ShockEvent values")
        shock_times = [event.decision_t for event in self.shock_schedule]
        if shock_times != sorted(shock_times) or len(shock_times) != len(
            set(shock_times)
        ):
            raise ValueError(
                "shock_schedule decision_t values must be unique and increasing"
            )
        if any(decision_t >= self.episode_length for decision_t in shock_times):
            raise ValueError("shock_schedule contains an event outside the horizon")
        if not isinstance(self.allow_scientific_scope, bool):
            raise TypeError("allow_scientific_scope must be boolean")
        if not isinstance(self.send_decoding_seed, bool):
            raise TypeError("send_decoding_seed must be boolean")
        object.__setattr__(
            self,
            "preflight_p95_reservations",
            _normalize_preflight_p95_reservations(
                self.preflight_p95_reservations
            ),
        )
        if self.scientific_scope == "preregistered_mechanism_micro_pilot":
            if not self.allow_scientific_scope:
                raise ValueError(
                    "micro-pilot scope requires explicit allow_scientific_scope=True"
                )
            if (
                not isinstance(self.pilot_contract_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.pilot_contract_hash) is None
            ):
                raise ValueError(
                    "micro-pilot scope requires a lowercase SHA-256 pilot_contract_hash"
                )
            if not isinstance(self.pilot_tag, str) or not self.pilot_tag.strip():
                raise ValueError("micro-pilot scope requires a non-empty pilot_tag")
            if shock_times != list(range(self.episode_length)):
                raise ValueError(
                    "micro-pilot scope requires one shock event for every period"
                )
        elif (
            self.allow_scientific_scope
            or self.pilot_contract_hash is not None
            or self.pilot_tag is not None
        ):
            raise ValueError(
                "pilot authorization fields are only valid for the preregistered "
                "mechanism micro-pilot scope"
            )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["utility"] = self.utility.to_dict()
        result["registered_outcome_criterion"] = (
            self.registered_outcome_criterion.to_dict()
        )
        result["evidence_weights"] = dict(self.evidence_weights)
        result["shock_schedule"] = [
            event.to_dict() for event in self.shock_schedule
        ]
        reservations: dict[str, dict[str, Any]] = {}
        for item in self.preflight_p95_reservations:
            reservations.setdefault(item.model, {})[item.call_kind] = item.to_dict()
        result["preflight_p95_reservations"] = reservations
        result["schema_version"] = RUNNER_SCHEMA_VERSION
        return result


@dataclass(frozen=True, slots=True)
class VerifiedRunResult:
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    validation_status: Mapping[str, Any]
    budget_snapshot: Mapping[str, Any]
    records: Mapping[str, tuple[Mapping[str, Any], ...]]

    def stream(self, name: str) -> tuple[Mapping[str, Any], ...]:
        if name not in self.records:
            raise KeyError(name)
        return self.records[name]


def _estimate_usage(
    prompt: str, *, max_tokens: int, provider_model_name: str
) -> UsageRecord:
    prompt_tokens = max(1, math.ceil(len(prompt) / 4))
    _, _, model = provider_model_name.partition("/")
    costs = MODEL_COSTS.get(model or provider_model_name)
    if costs is None and not _zero_cost_is_permitted(provider_model_name):
        raise VerifiedRunError(
            "cannot reserve a hosted provider call with an unknown price: "
            f"{provider_model_name}"
        )
    cost = 0.0
    if costs:
        cost = (
            prompt_tokens / 1000 * float(costs["prompt"])
            + max_tokens / 1000 * float(costs["completion"])
        )
    return UsageRecord(
        prompt_tokens=prompt_tokens,
        completion_tokens=max_tokens,
        cost_usd=cost,
    )


def required_preflight_p95_call_kinds(
    config: VerifiedRunConfig,
) -> tuple[str, ...]:
    """Return the provider call kinds that can occur in this run."""

    if not isinstance(config, VerifiedRunConfig):
        raise TypeError("config must be a VerifiedRunConfig")
    required = ["action"]
    semantic_due = (
        config.enable_semantic
        and any(
            current_t >= config.semantic_proposal_after
            and (current_t - config.semantic_proposal_after)
            % config.semantic_proposal_interval
            == 0
            and (
                config.freeze_new_proposals_after is None
                or current_t <= config.freeze_new_proposals_after
            )
            for current_t in range(1, config.episode_length + 1)
        )
    )
    if semantic_due:
        required.append("semantic")
    return tuple(required)


def _reservation_index(
    config: VerifiedRunConfig,
) -> dict[tuple[str, str], PreflightP95Reservation]:
    return {
        (item.model, item.call_kind): item
        for item in config.preflight_p95_reservations
    }


def _zero_cost_is_permitted(provider_model_name: str) -> bool:
    provider, _, _ = provider_model_name.partition("/")
    return provider in {
        DIAGNOSTIC_PROVIDER_NAME,
        "ollama",
        "local",
    }


def preflight_p95_reservation_for_call(
    config: VerifiedRunConfig,
    *,
    provider_model_name: str,
    call_kind: str,
    prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> UsageRecord:
    """Return the exact pre-dispatch reservation for one runner call.

    Scientific runs are fail-closed: there is no price-table or zero-cost
    fallback when the exact model x call-kind preflight estimate is absent.
    Bounded diagnostic/legacy runs may continue to use the historical local
    estimate, preserving the non-scientific compatibility path.
    """

    if not isinstance(config, VerifiedRunConfig):
        raise TypeError("config must be a VerifiedRunConfig")
    if not isinstance(provider_model_name, str) or not provider_model_name.strip():
        raise ValueError("provider_model_name must be a non-empty string")
    if not isinstance(call_kind, str):
        raise TypeError("call_kind must be a string")
    normalized_kind = call_kind.strip().lower().replace("_", "-")
    if normalized_kind not in PREFLIGHT_P95_CALL_KINDS:
        raise ValueError("call_kind must be action or semantic")
    key = (provider_model_name.strip(), normalized_kind)
    item = _reservation_index(config).get(key)
    if item is not None:
        usage = item.reserved_usage
        if usage.prompt_tokens < 1 or usage.completion_tokens < 1:
            raise VerifiedRunError(
                "preflight p95 reservation must reserve positive prompt and "
                f"completion tokens for {key[0]}::{key[1]}"
            )
        if usage.cost_usd <= 0 and not _zero_cost_is_permitted(key[0]):
            raise VerifiedRunError(
                "hosted preflight p95 reservation must have a positive cost for "
                f"{key[0]}::{key[1]}"
            )
        return usage
    if (
        config.scientific_scope == "preregistered_mechanism_micro_pilot"
        and not provider_model_name.startswith(f"{DIAGNOSTIC_PROVIDER_NAME}/")
    ):
        raise VerifiedRunError(
            "scientific dispatch lacks an exact observed+25% preflight p95 "
            f"reservation for {key[0]}::{key[1]}"
        )
    if prompt is None or max_tokens is None:
        raise VerifiedRunError(
            "legacy reservation fallback requires prompt and max_tokens"
        )
    return _estimate_usage(
        prompt,
        max_tokens=max_tokens,
        provider_model_name=provider_model_name,
    )


def validate_preflight_p95_reservations(
    config: VerifiedRunConfig,
    *,
    provider_model_name: str,
) -> dict[str, UsageRecord]:
    """Validate every potentially dispatched call kind before a run starts."""

    result: dict[str, UsageRecord] = {}
    for call_kind in required_preflight_p95_call_kinds(config):
        if (
            config.scientific_scope == "preregistered_mechanism_micro_pilot"
            and not provider_model_name.startswith(
                f"{DIAGNOSTIC_PROVIDER_NAME}/"
            )
        ):
            result[call_kind] = preflight_p95_reservation_for_call(
                config,
                provider_model_name=provider_model_name,
                call_kind=call_kind,
            )
    return result


def _provider_row(
    result: StructuredCompletion,
    *,
    call_kind: str,
    decision_t: int,
    agent_id: int,
    prompt_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "call_kind": call_kind,
        "decision_t": int(decision_t),
        "agent_id": int(agent_id),
        "prompt_hash": prompt_hash,
        "provider": result.provider,
        "model": result.model,
        "attempts": result.attempts,
        "latency_seconds": result.latency_seconds,
        "error_type": result.error_type,
        "provider_error_details": (
            result.provider_error_details.to_dict()
            if result.provider_error_details is not None
            else None
        ),
        "usage": result.usage.to_dict(),
        "request_seed": result.request_seed,
        "system_fingerprint": result.system_fingerprint,
        "response_model": result.response_model,
        "cached_prompt_tokens": result.cached_prompt_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "provider_request_id": result.request_id,
        "response_provider": result.response_provider,
        "response_route": result.response_route,
        "request_profile_id": result.request_profile_id,
        "request_provider_pin": list(result.request_provider_pin),
        "request_artifact_identity": dict(result.request_artifact_identity),
        "request_price_snapshot_source": result.request_price_snapshot_source,
        "request_price_snapshot_captured_at": (
            result.request_price_snapshot_captured_at
        ),
        "finish_reason": result.finish_reason,
        "native_finish_reason": result.native_finish_reason,
        "response_completed": result.response_completed,
        "provider_sdk_name": result.provider_sdk_name,
        "provider_sdk_version": result.provider_sdk_version,
        "route_attestation_code": result.route_attestation_code,
        "route_attestation_path": result.route_attestation_path,
        "route_attestation_source": result.route_attestation_source,
        "request_parameters": list(result.request_parameters),
        "temperature_dispatch": result.temperature_dispatch,
        "output_disposition": result.output_disposition,
        "raw_output_bytes": len(result.text.encode("utf-8")),
        "raw_output_hash": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
    }


def _semantic_parse_mode(raw_response: str) -> str:
    """Classify how a successfully parsed candidate reached the JSON parser."""

    stripped = raw_response.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, Mapping):
        return "exact_json"
    if stripped.startswith("```") and stripped.endswith("```"):
        return "fenced_recovery"
    if "{" in stripped and "}" in stripped:
        return "substring_recovery"
    return "parse_failure"


def _monthly_inflation(world: Any) -> float:
    prices = getattr(world, "price", None)
    if not isinstance(prices, list) or len(prices) < 2:
        return 0.0
    previous, current = float(prices[-2]), float(prices[-1])
    return current / previous - 1.0 if previous > 0 else 0.0


def _context_observation(
    *,
    decision_t: int,
    price: float,
    interest_rate: float,
    low_labor_rate: Optional[float],
    inflation: float,
    wealth: float,
    employed: bool,
) -> dict[str, Any]:
    low_labor_observed = low_labor_rate is not None
    low_labor_value = 0.0 if low_labor_rate is None else float(low_labor_rate)
    return {
        "timestamp": int(decision_t),
        "log_price": math.log1p(float(price)),
        "interest_rate": float(interest_rate),
        "prior_low_labor_rate": low_labor_value,
        "prior_low_labor_rate_available": float(low_labor_observed),
        "inflation": float(inflation),
        "log_wealth": math.log1p(float(wealth)),
        "employed": float(bool(employed)),
    }


def _m2_state(
    snapshot: Any, *, low_labor_rate: Optional[float], inflation: float
) -> dict[str, Any]:
    state = snapshot.to_m2_state()
    if low_labor_rate is None:
        # Unknown prior labor behavior must not participate in state similarity
        # or satisfy a semantic predicate as though it were an observed zero.
        state.pop("low_labor_rate", None)
        # Foundation's annual unemployment accumulator is not an observed
        # period outcome at reset; its numeric zero is only initialization.
        state.pop("unemployment_rate", None)
        state["unemployment_rate_available"] = 0.0
    else:
        state["low_labor_rate"] = float(low_labor_rate)
    state["low_labor_rate_available"] = float(low_labor_rate is not None)
    state["inflation"] = float(inflation)
    return state


def _prompt_state(
    env: Any,
    *,
    agent_id: int,
    decision_t: int,
    snapshot: Any,
    last_transition: Optional[Any],
    last_decision: Optional[ActionDecision],
    max_labor_hours: float,
) -> DecisionPromptState:
    if (last_transition is None) != (last_decision is None):
        raise ValueError("prior transition and action availability must match")
    agent = env.get_agent(str(agent_id))
    endogenous = agent.endogenous
    return DecisionPromptState(
        decision_t=decision_t,
        agent_id=agent_id,
        name=str(endogenous.get("name") or f"Agent {agent_id}"),
        age=int(endogenous.get("age") or 0),
        city=str(endogenous.get("city") or "Unknown city"),
        job=str(endogenous.get("job") or "Unemployment"),
        offer=str(endogenous.get("offer") or "No current offer"),
        wealth=float(snapshot.wealth),
        skill=float(agent.state["skill"]),
        price=float(snapshot.price),
        interest_rate=float(snapshot.interest_rate),
        last_consumption_quantity=(
            0.0
            if last_transition is None
            else float(last_transition.realized_consumption_quantity)
        ),
        last_labor_hours=(
            0.0 if last_decision is None else last_decision.executed_labor_hours
        ),
        last_tax_paid=(0.0 if last_transition is None else last_transition.tax_paid),
        last_lump_sum=(
            0.0 if last_transition is None else last_transition.lump_sum_transfer
        ),
        previous_period_available=last_transition is not None,
        max_labor_hours=max_labor_hours,
    )


def _prepare_memories(config: VerifiedRunConfig) -> dict[int, VerifiedDualTrackMemory]:
    systems: dict[int, VerifiedDualTrackMemory] = {}
    for agent_id in range(config.num_agents):
        router = CausalContextRouter(
            base_feature_names=CONTEXT_FEATURES,
            window_size=6,
            mode=config.context_mode,
        )
        systems[agent_id] = VerifiedDualTrackMemory(
            run_id=config.run_id,
            seed=config.seed,
            agent_id=agent_id,
            context_router=router,
            context_mode=config.context_mode,
            episodic_capacity=config.episodic_prompt_capacity,
            enable_episodic_retrieval=config.enable_episodic_retrieval,
            enable_semantic=config.enable_semantic,
            semantic_config={
                "min_candidate_support": config.min_candidate_support,
                "activation_min_support": config.activation_min_support,
                "activation_min_margin": config.activation_min_margin,
                "activation_confidence_threshold": (
                    config.activation_confidence_threshold
                ),
                "proposal_confidence_floor": config.proposal_confidence_floor,
                "retirement_patience": config.retirement_patience,
                "retirement_confidence_threshold": (
                    config.retirement_confidence_threshold
                ),
                "registered_outcome_criterion": (
                    config.registered_outcome_criterion
                ),
                "evidence_weights": config.evidence_weights,
            },
        )
    return systems


def _sealed_config_payload(
    config: VerifiedRunConfig,
    *,
    foundation_config: Mapping[str, Any],
    memories: Mapping[int, VerifiedDualTrackMemory],
) -> dict[str, Any]:
    """Build the exact config envelope persisted beside current runner streams."""

    sealed_config = config.to_dict()
    sealed_config["context_features"] = list(CONTEXT_FEATURES)
    sealed_config["prompt_schema_version"] = PROMPT_SCHEMA_VERSION
    semantic_configs = [
        memory.semantic.to_dict()["config"]
        for memory in memories.values()
        if memory.semantic is not None
    ]
    if semantic_configs and any(
        item != semantic_configs[0] for item in semantic_configs[1:]
    ):
        raise VerifiedRunError("agents do not share one semantic verifier config")
    sealed_config["effective_semantic_verifier"] = (
        _json_copy(semantic_configs[0]) if semantic_configs else None
    )
    sealed_config["foundation_env"] = _json_copy(foundation_config)
    sealed_config["foundation_env_hash"] = _sha256(foundation_config)
    return _json_copy(sealed_config)


def build_sealed_run_config(
    config: VerifiedRunConfig,
    *,
    env_config_source: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Purely reconstruct the config envelope used by artifact publication.

    Evidence publication calls this same builder to bind the registered base
    config to the effective semantic verifier, prompt/context schemas, and
    normalized Foundation environment without invoking a provider.
    """

    if not isinstance(config, VerifiedRunConfig):
        raise TypeError("config must be VerifiedRunConfig")
    foundation_config = prepare_foundation_env_config(
        env_config_source,
        n_agents=config.num_agents,
        episode_length=config.episode_length,
        labor_step=config.labor_step,
        max_labor_hours=config.max_labor_hours,
        consumption_step=config.consumption_step,
    )
    return _sealed_config_payload(
        config,
        foundation_config=foundation_config,
        memories=_prepare_memories(config),
    )


def _apply_shock_interest_rate(env: Any, event: ShockEvent) -> None:
    rates = getattr(getattr(env, "world", None), "interest_rate", None)
    if not isinstance(rates, list) or not rates:
        raise VerifiedRunError(
            "Foundation environment has no mutable current interest-rate state"
        )
    rates[-1] = float(event.interest_rate)


def _shock_prompt_is_causal(
    prompt: str,
    *,
    decision_t: int,
    schedule: tuple[ShockEvent, ...],
) -> bool:
    current = next(
        (event for event in schedule if event.decision_t == decision_t),
        None,
    )
    current_text = "" if current is None else current.to_prompt_text()
    if current_text and current_text not in prompt:
        return False
    return all(
        event.decision_t <= decision_t
        or event.to_prompt_text() == current_text
        or event.to_prompt_text() not in prompt
        for event in schedule
    )


def _fixed_error_candidate_response(
    memory: VerifiedDualTrackMemory,
    *,
    min_support: int,
) -> tuple[str, tuple[str, ...]]:
    support_ids = tuple(
        episode.episode_id
        for episode in memory.episodic.finalized_episodes[:min_support]
    )
    payload = {
        **FIXED_ERRONEOUS_RULE,
        "supporting_episode_ids": list(support_ids),
    }
    return json.dumps(payload, sort_keys=True), support_ids


def _inject_fixed_error_rule(
    memory: VerifiedDualTrackMemory,
    *,
    config: VerifiedRunConfig,
    agent_id: int,
    current_t: int,
) -> dict[str, Any]:
    if memory.semantic is None:
        raise VerifiedRunError("error-rule injection requires an enabled M3 track")
    raw_candidate, support_ids = _fixed_error_candidate_response(
        memory,
        min_support=config.min_candidate_support,
    )
    if config.error_rule_mode == "candidate-admission":
        rule = memory.semantic.propose(
            raw_candidate,
            current_t=current_t,
            generator_id="preregistered-fixed-error-rule-v1",
        )
        verifier_bypassed = False
    elif config.error_rule_mode == "forced-active":
        evidence_enabled = config.semantic_policy == "evidence-grounded"
        rule = memory.semantic.inject_active_rule(
            condition=ConditionPredicate.from_dict(FIXED_ERRONEOUS_RULE["condition"]),
            action_guidance=ActionGuidance.from_dict(
                FIXED_ERRONEOUS_RULE["action_guidance"]
            ),
            outcome_criterion=config.registered_outcome_criterion,
            rationale=str(FIXED_ERRONEOUS_RULE["rationale"]),
            current_t=current_t,
            injection_id=(
                f"{config.run_id}:agent-{agent_id}:fixed-error:"
                f"t-{config.error_rule_injection_t}"
            ),
            provenance={
                "error_rule_mode": config.error_rule_mode,
                "fixed_rule_schema": ERROR_RULE_INJECTION_SCHEMA_VERSION,
                "fixed_rule_hash": _sha256(FIXED_ERRONEOUS_RULE),
                "agent_id": agent_id,
                "decision_t": current_t,
                "verifier_bypassed": True,
                "semantic_policy": config.semantic_policy,
                "evidence_admission": evidence_enabled,
                "retirement_enabled": evidence_enabled,
            },
            initial_confidence=1.0,
            context_scope=ContextScope.global_scope(),
        )
        verifier_bypassed = True
    else:
        raise VerifiedRunError(
            f"cannot inject fixed rule in mode {config.error_rule_mode!r}"
        )
    return {
        "schema_version": ERROR_RULE_INJECTION_SCHEMA_VERSION,
        "decision_t": current_t,
        "agent_id": agent_id,
        "mode": config.error_rule_mode,
        "semantic_policy": config.semantic_policy,
        "fixed_rule": _json_copy(FIXED_ERRONEOUS_RULE),
        "fixed_rule_hash": _sha256(FIXED_ERRONEOUS_RULE),
        "requested_support_ids": list(support_ids),
        "raw_candidate_hash": hashlib.sha256(
            raw_candidate.encode("utf-8")
        ).hexdigest(),
        "verifier_bypassed": verifier_bypassed,
        "rule_id": rule.rule_id,
        "rule_status": rule.status,
    }


def run_verified_experiment(
    config: VerifiedRunConfig,
    *,
    llm: MultiModelLLM,
    budget: RunBudget,
    env_config_source: Mapping[str, Any] | str,
) -> VerifiedRunResult:
    """Run a bounded verified experiment and return finite in-memory records."""

    if not isinstance(config, VerifiedRunConfig):
        raise TypeError("config must be VerifiedRunConfig")
    if not isinstance(llm, MultiModelLLM):
        raise TypeError("llm must be MultiModelLLM")
    if not isinstance(budget, RunBudget):
        raise TypeError("budget must be RunBudget")

    provider_model_name = llm.get_model_name()
    diagnostic_only = provider_model_name.startswith(f"{DIAGNOSTIC_PROVIDER_NAME}/")
    # Validate the complete scientific reservation surface before initializing
    # the environment, and critically before the first provider dispatch.
    validate_preflight_p95_reservations(
        config,
        provider_model_name=provider_model_name,
    )

    np.random.seed(config.seed)
    random.seed(config.seed)
    foundation_config = prepare_foundation_env_config(
        env_config_source,
        n_agents=config.num_agents,
        episode_length=config.episode_length,
        labor_step=config.labor_step,
        max_labor_hours=config.max_labor_hours,
        consumption_step=config.consumption_step,
    )
    env = foundation.make_env_instance(**foundation_config)
    env.reset()
    ledger = EnvironmentLedger(config.utility)
    memories = _prepare_memories(config)
    shocks_by_t = {event.decision_t: event for event in config.shock_schedule}

    records: dict[str, list[dict[str, Any]]] = {
        "actions": [],
        "api_usage": [],
        "context_trace": [],
        "decision_snapshots": [],
        "episodes": [],
        "utility_ledger": [],
        "semantic_rule_events": [],
        "semantic_rules": [],
        "semantic_proposals": [],
        "shock_events": [],
        "error_rule_injections": [],
        "macro_steps": [],
        "errors": [],
    }
    last_decisions: dict[str, ActionDecision] = {}
    last_transitions: dict[str, Any] = {}
    proposals_made = {agent_id: 0 for agent_id in range(config.num_agents)}
    semantic_event_offsets = {agent_id: 0 for agent_id in range(config.num_agents)}
    # There is no prior executed-labor cohort before t=0.  Encode that fact with
    # a neutral numeric value plus an explicit observation mask instead of
    # injecting an artificial 100% low-labor signal.
    previous_low_labor_rate: Optional[float] = None
    completed_periods = 0

    for decision_t in range(config.episode_length):
        current_shock = shocks_by_t.get(decision_t)
        if current_shock is not None:
            _apply_shock_interest_rate(env, current_shock)
        if (
            config.error_rule_mode != "none"
            and decision_t == config.error_rule_injection_t
        ):
            for agent_id, memory in memories.items():
                records["error_rule_injections"].append(
                    _inject_fixed_error_rule(
                        memory,
                        config=config,
                        agent_id=agent_id,
                        current_t=decision_t,
                    )
                )
        pre_snapshots = capture_foundation_snapshots(
            env,
            expected_timestamp=decision_t,
            labor_step=config.labor_step,
            max_labor_hours=config.max_labor_hours,
            consumption_step=config.consumption_step,
        )
        current_inflation = _monthly_inflation(env.world)
        bundles: dict[int, MemoryBundle] = {}
        prompt_rows: dict[int, Any] = {}
        dialogs: list[list[dict[str, str]]] = []

        for agent_id in range(config.num_agents):
            agent_key = str(agent_id)
            snapshot = pre_snapshots[agent_key]
            retrieval_state = _m2_state(
                snapshot,
                low_labor_rate=previous_low_labor_rate,
                inflation=current_inflation,
            )
            context = _context_observation(
                decision_t=decision_t,
                price=snapshot.price,
                interest_rate=snapshot.interest_rate,
                low_labor_rate=previous_low_labor_rate,
                inflation=current_inflation,
                wealth=snapshot.wealth,
                employed=snapshot.employed,
            )
            bundle = memories[agent_id].prepare_decision(
                decision_t=decision_t,
                context_observation=context,
                retrieval_state=retrieval_state,
                retrieval_k=config.retrieval_k,
                rule_budget=config.rule_budget,
            )
            base_prompt = build_base_decision_prompt(
                _prompt_state(
                    env,
                    agent_id=agent_id,
                    decision_t=decision_t,
                    snapshot=snapshot,
                    last_transition=last_transitions.get(agent_key),
                    last_decision=last_decisions.get(agent_key),
                    max_labor_hours=config.max_labor_hours,
                ),
                config.utility,
                event_text=(
                    current_shock.to_prompt_text()
                    if current_shock is not None
                    else ""
                ),
                causal_context_summary=bundle.protected_context_prompt,
            )
            prompt = compose_decision_prompt(base_prompt, bundle.memory_prompt)
            bundles[agent_id] = bundle
            prompt_rows[agent_id] = prompt
            dialogs.append([{"role": "user", "content": prompt.full_prompt}])
            trace = bundle.to_trace()
            trace.update(
                {
                    "agent_id": agent_id,
                    "context_packet": bundle.context_packet.to_dict(),
                }
            )
            records["context_trace"].append(trace)
            records["decision_snapshots"].append(
                {
                    "schema_version": RUNNER_SCHEMA_VERSION,
                    "prompt_schema_version": PROMPT_SCHEMA_VERSION,
                    "decision_t": decision_t,
                    "agent_id": agent_id,
                    "environment_state_hash": _sha256(retrieval_state),
                    "base_prompt": prompt.base_prompt,
                    "memory_text": prompt.memory_text,
                    "protected_context_text": bundle.protected_context_prompt,
                    "protected_context_hash": hashlib.sha256(
                        bundle.protected_context_prompt.encode("utf-8")
                    ).hexdigest(),
                    "full_prompt_hash": prompt.full_prompt_hash,
                    "base_prompt_hash": prompt.base_prompt_hash,
                    "memory_hash": prompt.memory_hash,
                    "context_packet_id": bundle.context_packet.context_id,
                    "context_packet_hash": bundle.context_packet.context_hash,
                    "provider_model": provider_model_name,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "diagnostic_only": diagnostic_only,
                    "shock_event": (
                        current_shock.to_dict()
                        if current_shock is not None
                        else None
                    ),
                }
            )

        estimates = [
            preflight_p95_reservation_for_call(
                config,
                provider_model_name=provider_model_name,
                call_kind="action",
                prompt=prompt_rows[index].full_prompt,
                max_tokens=config.action_max_tokens,
            )
            for index in range(config.num_agents)
        ]
        completions = llm.get_multiple_structured_completions(
            dialogs,
            temperature=config.temperature,
            max_tokens=config.action_max_tokens,
            top_p=config.top_p,
            budget=budget,
            labels=[f"action:t{decision_t}:a{index}" for index in range(config.num_agents)],
            tags=[
                {"call_kind": "action", "decision_t": decision_t, "agent_id": index}
                for index in range(config.num_agents)
            ],
            estimated_usages=estimates,
            max_retries=config.max_retries,
            seed=config.seed if config.send_decoding_seed else None,
        )
        decisions: dict[str, ActionDecision] = {}
        for agent_id, completion in enumerate(completions):
            prompt = prompt_rows[agent_id]
            usage_row = _provider_row(
                completion,
                call_kind="action",
                decision_t=decision_t,
                agent_id=agent_id,
                prompt_hash=prompt.full_prompt_hash,
            )
            records["api_usage"].append(usage_row)
            if not completion.ok or completion.text == "Error":
                records["errors"].append(usage_row)
                raise VerifiedRunError(
                    f"provider action failure at t={decision_t}, agent={agent_id}: "
                    f"{completion.error_type}"
                )
            decision = parse_direct_action(
                completion.text,
                max_labor_hours=config.max_labor_hours,
                labor_step=config.labor_step,
                consumption_step=config.consumption_step,
            )
            if config.fail_on_clipped_action and decision.clipped:
                raise VerifiedRunError(
                    f"clipped action at t={decision_t}, agent={agent_id}"
                )
            decisions[str(agent_id)] = decision
            bundle = bundles[agent_id]
            pre_state = _m2_state(
                pre_snapshots[str(agent_id)],
                low_labor_rate=previous_low_labor_rate,
                inflation=current_inflation,
            )
            memories[agent_id].begin_episode(
                decision_t=decision_t,
                pre_state=pre_state,
                proposed_action={
                    "work_propensity": decision.proposed_work_fraction,
                    "consumption_fraction": decision.proposed_consumption_fraction,
                },
                executed_action={
                    "labor_hours": decision.executed_labor_hours,
                    "work_propensity": decision.proposed_work_fraction,
                    "consumption_fraction": decision.executed_consumption_rate,
                },
                reflection=decision.reflection,
            )
            records["actions"].append(
                {
                    "schema_version": RUNNER_SCHEMA_VERSION,
                    "decision_t": decision_t,
                    "agent_id": agent_id,
                    "provider": completion.provider,
                    "model": completion.model,
                    "prompt_hash": prompt.full_prompt_hash,
                    "raw_output": completion.text,
                    "decision": decision.to_dict(),
                    "retrieved_episode_ids": list(bundle.retrieved_episode_ids),
                    "selected_rule_ids": list(bundle.selected_rule_ids),
                    "diagnostic_only": diagnostic_only,
                }
            )

        pre_batch = {
            agent_id: {
                "wealth": snapshot.wealth,
                "cumulative_production": snapshot.cumulative_production,
                "price": snapshot.price,
                "interest_rate": snapshot.interest_rate,
                "proposed_work_propensity": decisions[agent_id].proposed_work_fraction,
                "proposed_consumption_fraction": decisions[
                    agent_id
                ].proposed_consumption_fraction,
                "executed_labor_hours": decisions[agent_id].executed_labor_hours,
                "executed_consumption_rate": decisions[
                    agent_id
                ].executed_consumption_rate,
            }
            for agent_id, snapshot in pre_snapshots.items()
        }
        ledger.capture_pre(decision_t, pre_batch)
        env_actions = build_foundation_actions(
            decisions,
            labor_step=config.labor_step,
            max_labor_hours=config.max_labor_hours,
            consumption_step=config.consumption_step,
        )
        if current_shock is not None:
            _apply_shock_interest_rate(env, current_shock)
            records["shock_events"].append(current_shock.to_dict())
        _, rewards, done, _ = env.step(env_actions)
        transitions = derive_foundation_transitions(
            env,
            pre_snapshots=pre_snapshots,
            decisions=decisions,
            expected_outcome_t=decision_t + 1,
            labor_step=config.labor_step,
            max_labor_hours=config.max_labor_hours,
            consumption_step=config.consumption_step,
        )
        post_batch: dict[str, dict[str, Any]] = {}
        for agent_id, transition in transitions.items():
            post = transition.to_m0_post().to_dict()
            post.pop("period")
            post.pop("agent_id")
            post_batch[agent_id] = post
        utility_rows = ledger.capture_post(decision_t, post_batch)
        rows_by_agent = {row.agent_id: row for row in utility_rows}
        current_low_labor_rate = mean(
            float(decision.executed_labor_hours < config.low_labor_threshold_hours)
            for decision in decisions.values()
        )
        realized_inflation = _monthly_inflation(env.world)

        for agent_id in range(config.num_agents):
            agent_key = str(agent_id)
            transition = transitions[agent_key]
            decision = decisions[agent_key]
            utility_row = rows_by_agent[agent_key]
            next_state = _m2_state(
                transition.post,
                low_labor_rate=current_low_labor_rate,
                inflation=realized_inflation,
            )
            outcome = transition.to_m2_outcome(decision)
            episode = memories[agent_id].finalize_episode(
                decision_t=decision_t,
                next_state=next_state,
                outcome=outcome,
                reward=float(rewards[agent_key]),
                flow_utility=utility_row.flow_utility,
            )
            records["episodes"].append(episode.to_dict())
            records["utility_ledger"].append(utility_row.to_dict())
            last_decisions[agent_key] = decision
            last_transitions[agent_key] = transition

        current_t = decision_t + 1
        proposal_due = (
            config.enable_semantic
            and current_t >= config.semantic_proposal_after
            and (current_t - config.semantic_proposal_after)
            % config.semantic_proposal_interval
            == 0
            and (
                config.freeze_new_proposals_after is None
                or current_t <= config.freeze_new_proposals_after
            )
        )
        eligible = [
            agent_id
            for agent_id in range(config.num_agents)
            if proposal_due
            and proposals_made[agent_id] < config.max_rule_proposals_per_agent
        ]
        if eligible:
            proposal_prompts = [
                memories[agent_id].build_rule_proposal_prompt(max_episodes=6)
                for agent_id in eligible
            ]
            proposal_results = llm.get_multiple_structured_completions(
                [[{"role": "user", "content": prompt}] for prompt in proposal_prompts],
                temperature=0.0,
                max_tokens=config.rule_max_tokens,
                top_p=1.0,
                budget=budget,
                labels=[f"semantic:t{current_t}:a{agent_id}" for agent_id in eligible],
                tags=[
                    {"call_kind": "semantic", "current_t": current_t, "agent_id": agent_id}
                    for agent_id in eligible
                ],
                estimated_usages=[
                    preflight_p95_reservation_for_call(
                        config,
                        provider_model_name=provider_model_name,
                        call_kind="semantic",
                        prompt=prompt,
                        max_tokens=config.rule_max_tokens,
                    )
                    for prompt in proposal_prompts
                ],
                max_retries=config.max_retries,
                seed=config.seed if config.send_decoding_seed else None,
            )
            for agent_id, prompt, completion in zip(
                eligible, proposal_prompts, proposal_results
            ):
                proposals_made[agent_id] += 1
                prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                usage_row = _provider_row(
                    completion,
                    call_kind="semantic",
                    decision_t=current_t,
                    agent_id=agent_id,
                    prompt_hash=prompt_hash,
                )
                records["api_usage"].append(usage_row)
                proposal_row = {
                    "schema_version": RUNNER_SCHEMA_VERSION,
                    "current_t": current_t,
                    "agent_id": agent_id,
                    "prompt_hash": prompt_hash,
                    "raw_output": completion.text,
                    "raw_output_hash": usage_row["raw_output_hash"],
                    "provider_error": completion.error_type,
                    "rule_id": None,
                    "rule_status": None,
                    "parse_error": None,
                    "candidate_parse_status": "not_attempted",
                    "candidate_parse_mode": "not_attempted",
                    "semantic_policy": config.semantic_policy,
                    "diagnostic_only": diagnostic_only,
                }
                if completion.ok and completion.text != "Error":
                    try:
                        rule = memories[agent_id].submit_rule_proposal(
                            completion.text,
                            current_t=current_t,
                            generator_id=provider_model_name,
                            semantic_policy=config.semantic_policy,
                        )
                        proposal_row["rule_id"] = rule.rule_id
                        proposal_row["rule_status"] = rule.status
                        proposal_row["candidate_parse_status"] = "success"
                        proposal_row["candidate_parse_mode"] = _semantic_parse_mode(
                            completion.text
                        )
                    except CandidateParseError as exc:
                        proposal_row["parse_error"] = str(exc)
                        proposal_row["candidate_parse_status"] = "failure"
                        proposal_row["candidate_parse_mode"] = "parse_failure"
                        records["errors"].append(
                            {
                                **usage_row,
                                "error_type": "CandidateParseError",
                                "message": str(exc),
                            }
                        )
                        records["semantic_proposals"].append(proposal_row)
                        if config.semantic_parse_failure_policy == "fail-run":
                            raise VerifiedRunError(
                                str(exc),
                                failure=RunnerFailure(
                                    schema_version=RUNNER_SCHEMA_VERSION,
                                    error_stage="semantic_candidate_parser",
                                    call_kind="semantic",
                                    decision_t=current_t,
                                    agent_id=agent_id,
                                    error_type="CandidateParseError",
                                    message=str(exc),
                                    prompt_hash=prompt_hash,
                                    raw_output_hash=usage_row["raw_output_hash"],
                                    provider=completion.provider,
                                    model=completion.model,
                                    attempts=completion.attempts,
                                ),
                            ) from exc
                        continue
                else:
                    records["errors"].append(usage_row)
                    records["semantic_proposals"].append(proposal_row)
                    raise VerifiedRunError(
                        (
                            f"provider semantic failure at t={current_t}, "
                            f"agent={agent_id}: {completion.error_type}"
                        ),
                        failure=RunnerFailure(
                            schema_version=RUNNER_SCHEMA_VERSION,
                            error_stage="semantic_provider",
                            call_kind="semantic",
                            decision_t=current_t,
                            agent_id=agent_id,
                            error_type=str(completion.error_type or "ProviderError"),
                            message=str(completion.error_type or "provider failure"),
                            prompt_hash=prompt_hash,
                            raw_output_hash=usage_row["raw_output_hash"],
                            provider=completion.provider,
                            model=completion.model,
                            attempts=completion.attempts,
                        ),
                    )
                records["semantic_proposals"].append(proposal_row)

        for agent_id, memory in memories.items():
            if memory.semantic is None:
                continue
            events = memory.semantic.events
            offset = semantic_event_offsets[agent_id]
            for event in events[offset:]:
                event_row = event.to_dict()
                event_row["agent_id"] = agent_id
                records["semantic_rule_events"].append(event_row)
            semantic_event_offsets[agent_id] = len(events)

        wealths = [
            float(env.get_agent(str(agent_id)).inventory["Coin"])
            for agent_id in range(config.num_agents)
        ]
        records["macro_steps"].append(
            {
                "schema_version": RUNNER_SCHEMA_VERSION,
                "decision_t": decision_t,
                "outcome_t": current_t,
                "price": float(env.world.price[-1]),
                "monthly_inflation": realized_inflation,
                "low_labor_rate": current_low_labor_rate,
                "average_wealth": mean(wealths),
                "done": bool(done["__all__"]),
            }
        )
        previous_low_labor_rate = current_low_labor_rate
        completed_periods = current_t

    for agent_id, memory in memories.items():
        memory.validate()
        if memory.semantic is not None:
            for rule in memory.semantic.rules:
                row = rule.to_dict()
                row["agent_id"] = agent_id
                records["semantic_rules"].append(row)

    expected_rows = config.num_agents * config.episode_length
    final_wealths = [
        float(env.get_agent(str(agent_id)).inventory["Coin"])
        for agent_id in range(config.num_agents)
    ]
    action_rows = records["actions"]
    intermediate_actions = [
        row["decision"]["executed_labor_hours"]
        for row in action_rows
        if 0 < row["decision"]["executed_labor_hours"] < config.max_labor_hours
    ]
    selected_rule_ids = {
        rule_id
        for row in action_rows
        for rule_id in row["selected_rule_ids"]
    }
    parse_attempts = records["semantic_proposals"]
    parse_successes = [
        row for row in parse_attempts if row["candidate_parse_status"] == "success"
    ]
    parse_failures = [
        row for row in parse_attempts if row["candidate_parse_status"] == "failure"
    ]
    provider_errors = [
        row
        for row in records["errors"]
        if row.get("error_type") != "CandidateParseError"
    ]
    unverified_rule_ids = {
        row["rule_id"]
        for row in records["semantic_rules"]
        if (row.get("injection_provenance") or {}).get("semantic_policy")
        == "unverified-immediate"
    }
    checks = {
        "completed_all_periods": completed_periods == config.episode_length,
        "action_count_t_by_n": len(action_rows) == expected_rows,
        "episode_count_t_by_n": len(records["episodes"]) == expected_rows,
        "utility_count_t_by_n": len(records["utility_ledger"]) == expected_rows,
        "no_provider_errors": len(provider_errors) == 0,
        "semantic_parse_outcomes_accounted": (
            len(parse_successes) + len(parse_failures) == len(parse_attempts)
        ),
        "causal_context": all(
            row["context_packet"]["observed_through"] <= row["decision_t"]
            for row in records["context_trace"]
        ),
        "episode_alignment": all(
            row["outcome_t"] == row["decision_t"] + 1
            for row in records["episodes"]
        ),
        "budget_identity": all(
            abs(row["budget_residual"]) <= config.utility.budget_tolerance
            for row in records["utility_ledger"]
        ),
        "shock_schedule_applied_exactly": records["shock_events"]
        == [event.to_dict() for event in config.shock_schedule],
        "no_future_shock_in_prompt": all(
            (
                row["shock_event"] is None
                or row["shock_event"]["decision_t"] == row["decision_t"]
            )
            and _shock_prompt_is_causal(
                row["base_prompt"],
                decision_t=row["decision_t"],
                schedule=config.shock_schedule,
            )
            for row in records["decision_snapshots"]
        ),
        "proposal_freeze_respected": (
            config.freeze_new_proposals_after is None
            or all(
                row["current_t"] <= config.freeze_new_proposals_after
                for row in records["semantic_proposals"]
            )
        ),
        "error_rule_injection_accounted": len(
            records["error_rule_injections"]
        )
        == (
            0 if config.error_rule_mode == "none" else config.num_agents
        ),
        "unverified_policy_has_no_evidence_or_retirement": all(
            not (
                row.get("rule_id") in unverified_rule_ids
                and (
                    str(row.get("event_type", "")).endswith("_evidence_added")
                    or row.get("event_type") == "rule_retired"
                )
            )
            for row in records["semantic_rule_events"]
        ),
    }
    # This is an observed diagnostic, not a pass/fail gate.  In particular,
    # disabling M3 must not be serialized as if an activation had occurred.
    semantic_activation_observed = bool(selected_rule_ids)
    validation_pass = all(checks.values())
    scientific_evidence = bool(
        validation_pass
        and not diagnostic_only
        and config.scientific_scope
        == "preregistered_mechanism_micro_pilot"
        and config.allow_scientific_scope
        and config.pilot_contract_hash
        and config.pilot_tag
    )
    validation_status = {
        "status": "pass" if validation_pass else "fail",
        "checks": checks,
        "diagnostic_only": diagnostic_only,
        "scientific_evidence": scientific_evidence,
    }
    utility_values = [row["flow_utility"] for row in records["utility_ledger"]]
    labor_hours_counts = {
        f"{hours:g}": sum(
            row["decision"]["executed_labor_hours"] == hours
            for row in action_rows
        )
        for hours in sorted(
            {row["decision"]["executed_labor_hours"] for row in action_rows}
        )
    }
    ceiling_labor_count = sum(
        row["decision"]["executed_labor_hours"] == config.max_labor_hours
        for row in action_rows
    )
    summary = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "run_id": config.run_id,
        "provider_model": provider_model_name,
        "diagnostic_only": diagnostic_only,
        "scientific_evidence": scientific_evidence,
        "result_scope": config.scientific_scope,
        "result_complete": completed_periods == config.episode_length,
        "num_agents": config.num_agents,
        "episode_length": config.episode_length,
        "final_metrics": {
            "average_wealth": mean(final_wealths),
            "median_wealth": float(np.median(final_wealths)),
            "gini": _gini(final_wealths),
            "average_flow_utility": mean(utility_values),
            "average_low_labor_rate": mean(
                row["low_labor_rate"] for row in records["macro_steps"]
            ),
        },
        "action_diagnostics": {
            "unique_labor_hours": sorted(
                {row["decision"]["executed_labor_hours"] for row in action_rows}
            ),
            "intermediate_action_count": len(intermediate_actions),
            "intermediate_action_observed": bool(intermediate_actions),
            "labor_hours_counts": labor_hours_counts,
            "ceiling_labor_count": ceiling_labor_count,
            "ceiling_labor_rate": (
                ceiling_labor_count / len(action_rows) if action_rows else 0.0
            ),
            "clipped_action_count": sum(
                bool(row["decision"]["clipped"]) for row in action_rows
            ),
        },
        "memory_diagnostics": {
            "semantic_policy": config.semantic_policy,
            "error_rule_mode": config.error_rule_mode,
            "error_rule_injection_count": len(
                records["error_rule_injections"]
            ),
            "freeze_new_proposals_after": config.freeze_new_proposals_after,
            "registered_outcome_criterion": (
                config.registered_outcome_criterion.to_dict()
            ),
            "semantic_rule_status_counts": {
                status: sum(
                    row["status"] == status for row in records["semantic_rules"]
                )
                for status in ("provisional", "active", "rejected", "retired")
            },
            "active_rule_retrieval_count": sum(
                len(row["selected_rule_ids"]) for row in action_rows
            ),
            "episodic_retrieval_count": sum(
                len(row["retrieved_episode_ids"]) for row in action_rows
            ),
            "semantic_activation_observed": semantic_activation_observed,
            "semantic_candidate_parse": {
                "attempt_count": len(parse_attempts),
                "success_count": len(parse_successes),
                "failure_count": len(parse_failures),
                "failure_rate": (
                    len(parse_failures) / len(parse_attempts)
                    if parse_attempts
                    else 0.0
                ),
                "mode_counts": {
                    mode: sum(
                        row["candidate_parse_mode"] == mode
                        for row in parse_attempts
                    )
                    for mode in (
                        "exact_json",
                        "fenced_recovery",
                        "substring_recovery",
                        "parse_failure",
                    )
                },
            },
        },
        "api": budget.snapshot().to_dict(),
        "validation": validation_status,
    }
    frozen_records = {
        name: tuple(_json_copy(row) for row in rows) for name, rows in records.items()
    }
    sealed_config = _sealed_config_payload(
        config,
        foundation_config=foundation_config,
        memories=memories,
    )
    return VerifiedRunResult(
        config=_json_copy(sealed_config),
        summary=_json_copy(summary),
        validation_status=_json_copy(validation_status),
        budget_snapshot=_json_copy(budget.snapshot().to_dict()),
        records=frozen_records,
    )


__all__ = [
    "build_sealed_run_config",
    "CONTEXT_FEATURES",
    "ERROR_RULE_INJECTION_SCHEMA_VERSION",
    "ERROR_RULE_MODES",
    "FIXED_ERRONEOUS_RULE",
    "PREFLIGHT_P95_CALL_KINDS",
    "PREFLIGHT_P95_RESERVE_MULTIPLIER",
    "PreflightP95Reservation",
    "RUNNER_SCHEMA_VERSION",
    "RunnerFailure",
    "SCIENTIFIC_SCOPES",
    "SEMANTIC_PARSE_FAILURE_POLICIES",
    "SEMANTIC_POLICIES",
    "SHOCK_EVENT_SCHEMA_VERSION",
    "ShockEvent",
    "VerifiedRunConfig",
    "VerifiedRunError",
    "VerifiedRunResult",
    "preflight_p95_reservation_for_call",
    "required_preflight_p95_call_kinds",
    "run_verified_experiment",
    "validate_preflight_p95_reservations",
]
