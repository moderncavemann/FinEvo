"""Production-shaped 30-call actor/proposer capability gate.

The gate deliberately exercises the same decision prompt, action parser, episodic
evidence, semantic proposal prompt, and verifier admission path used by a pilot
run.  It is still a capability/interface gate, not scientific-effect evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence

from llm_providers import MultiModelLLM, StructuredCompletion

from .actions import ActionParseError, parse_direct_action
from .budget import RunBudget, UsageRecord
from .m0_utility import UtilityConfig
from .m2_episodic import EvidenceLinkedEpisodicTrack
from .m3_semantic import (
    ActionGuidance,
    CandidateParseError,
    ConditionPredicate,
    ContextScope,
    OutcomeCriterion,
    VerifiedSemanticRuleTrack,
)
from .prompts import (
    DecisionPromptState,
    build_base_decision_prompt,
    compose_decision_prompt,
)


CAPABILITY_SCHEMA_VERSION = "finevo-capability-gate-v4"
CAPABILITY_THRESHOLDS = {
    # The legacy category name remains stable for v1/v2 aggregate readers.  V3+
    # rows identify the actual task as ``task_kind=action_generation``.
    "utility-ranking": 10,
    "rule-application": 10,
    "rule-proposal": 5,
}
_OUTPUT_CONTRACT_IDS = {
    "action_generation": "actor-action",
    "rule_application": "actor-action",
    "rule_proposal": "semantic-proposal",
}
_DEFAULT_TASK_OUTPUT_CONTRACTS: dict[str, dict[str, Any]] = {
    # Compatibility defaults only.  Scientific pilot-v2 dispatch passes the
    # frozen contract mapping explicitly.
    "actor-action": {
        "request_max_completion_tokens": 512,
        "visible_json_max_bytes": 4096,
        "accepted_parse_modes": ("exact_json",),
        "required_finish_reason": "stop",
    },
    "semantic-proposal": {
        "request_max_completion_tokens": 1200,
        "visible_json_max_bytes": 8192,
        "accepted_parse_modes": ("exact_json",),
        "required_finish_reason": "stop",
    },
}
_PARSE_MODES = frozenset(
    {"exact_json", "fenced_recovery", "substring_recovery", "parse_failure"}
)


@dataclass(frozen=True, slots=True)
class CapabilityTask:
    task_id: str
    category: str
    task_kind: str
    output_contract_id: str
    prompt: str
    expected_choice: str | None = None
    expected_condition: Mapping[str, Any] | None = None
    expected_guidance: Mapping[str, Any] | None = None
    expected_scope: Mapping[str, Any] | None = None
    allowed_episode_ids: tuple[str, ...] = ()
    proposal_track: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _TaskOutputContract:
    request_max_completion_tokens: int
    visible_json_max_bytes: int
    accepted_parse_modes: tuple[str, ...]
    required_finish_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_max_completion_tokens": self.request_max_completion_tokens,
            "visible_json_max_bytes": self.visible_json_max_bytes,
            "accepted_parse_modes": list(self.accepted_parse_modes),
            "required_finish_reason": self.required_finish_reason,
        }


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utility_config() -> UtilityConfig:
    return UtilityConfig(
        rho=1.0,
        labor_weight=2.0,
        inverse_frisch=1.0,
        consumption_scale=1.0,
        max_labor_hours=168.0,
        discount_factor=0.99,
    )


def _decision_state(
    index: int,
    *,
    wealth: float = 80.0,
    price: float = 1.2,
    interest_rate: float = 0.03,
    decision_t: int | None = None,
) -> DecisionPromptState:
    return DecisionPromptState(
        decision_t=index if decision_t is None else decision_t,
        agent_id=index % 4,
        name=f"Capability Agent {index:02d}",
        age=25 + index,
        city="Pilot City",
        job="employed",
        offer="monthly production job",
        wealth=wealth,
        skill=1.0 + 0.05 * index,
        price=price,
        interest_rate=interest_rate,
        last_consumption_quantity=1.0 + 0.1 * index,
        last_labor_hours=float(40 + 8 * (index % 10)),
        last_tax_paid=2.0 + index,
        last_lump_sum=1.0,
        previous_period_available=index != 1,
    )


def _action_tasks() -> tuple[CapabilityTask, ...]:
    states = (
        (60.0, 1.00, 0.03),
        (20.0, 1.30, 0.03),
        (120.0, 0.90, 0.08),
        (8.0, 2.10, 0.08),
        (45.0, 1.70, 0.01),
        (200.0, 2.40, 0.03),
        (15.0, 0.75, 0.03),
        (90.0, 1.10, 0.10),
        (35.0, 3.00, 0.06),
        (300.0, 1.50, 0.02),
        (5.0, 0.65, 0.08),
        (75.0, 2.00, 0.03),
    )
    tasks: list[CapabilityTask] = []
    for index, (wealth, price, interest_rate) in enumerate(states, start=1):
        base = build_base_decision_prompt(
            _decision_state(
                index,
                wealth=wealth,
                price=price,
                interest_rate=interest_rate,
            ),
            _utility_config(),
            event_text=(
                "The current interest-rate regime and goods price are observable "
                "now; do not assume future values."
            ),
            causal_context_summary=(
                "Recent macro observations through this decision month show "
                f"prices near {price:g} and the current rate at "
                f"{interest_rate:.1%}."
            ),
        )
        prompt = compose_decision_prompt(base).full_prompt
        tasks.append(
            CapabilityTask(
                task_id=f"action-{index:02d}",
                category="utility-ranking",
                task_kind="action_generation",
                output_contract_id="actor-action",
                prompt=prompt,
            )
        )
    return tuple(tasks)


def _append_episode(
    track: EvidenceLinkedEpisodicTrack,
    *,
    t: int,
    pre_state: Mapping[str, Any],
    labor_hours: float,
    consumption_fraction: float,
    flow_utility: float,
) -> Any:
    action = {
        "labor_hours": float(labor_hours),
        "consumption_fraction": float(consumption_fraction),
        "work_propensity": float(labor_hours) / 168.0,
    }
    decision_id = track.begin_episode(
        decision_t=t,
        pre_state=pre_state,
        context_id=f"cap-context-{t}",
        context_vector=(float(pre_state.get("price", 1.0)), float(t)),
        retrieved_episode_ids=(),
        selected_rule_ids=(),
        proposed_action=action,
        executed_action=action,
        reflection="frozen capability evidence",
    )
    next_wealth = float(pre_state.get("wealth", 50.0)) + 1.0
    return track.finalize_episode(
        decision_id,
        outcome_t=t + 1,
        next_state={**dict(pre_state), "wealth": next_wealth},
        outcome={"wealth_change": 1.0},
        reward=flow_utility,
        flow_utility=flow_utility,
    )


def _rule_cases() -> tuple[tuple[Any, ...], ...]:
    return (
        ("interest_rate", ">=", 0.08, "consumption_fraction", "at_most", 0.25, 0.01),
        ("interest_rate", ">=", 0.08, "labor_hours", "at_least", 80.0, 0.0),
        ("price", ">=", 2.0, "consumption_fraction", "at_most", 0.40, 0.01),
        ("wealth", "<=", 10.0, "consumption_fraction", "approximately", 0.30, 0.05),
        ("wealth", ">=", 100.0, "labor_hours", "at_least", 84.0, 0.0),
        ("price", "<=", 0.80, "consumption_fraction", "at_least", 0.50, 0.01),
        ("wealth", "<=", 10.0, "labor_hours", "at_least", 100.0, 0.0),
        ("price", ">=", 2.0, "labor_hours", "approximately", 84.0, 8.0),
        ("interest_rate", "<=", 0.02, "consumption_fraction", "approximately", 0.60, 0.05),
        ("wealth", ">=", 100.0, "consumption_fraction", "at_most", 0.35, 0.01),
        ("price", "<=", 1.0, "consumption_fraction", "at_least", 0.55, 0.01),
        ("interest_rate", ">=", 0.08, "labor_hours", "approximately", 120.0, 8.0),
    )


def _current_state_for_rule(
    field: str, value: float
) -> tuple[DecisionPromptState, dict[str, float]]:
    state_values = {"wealth": 80.0, "price": 1.2, "interest_rate": 0.03}
    if field in state_values:
        state_values[field] = value
    state = _decision_state(
        20,
        wealth=state_values["wealth"],
        price=state_values["price"],
        interest_rate=state_values["interest_rate"],
        decision_t=12,
    )
    retrieval_state = {
        "wealth": state.wealth,
        "price": state.price,
        "interest_rate": state.interest_rate,
        "inflation": 0.03,
        "sentiment": 0.0,
        "low_labor_rate": 0.25,
        "unemployment_rate": 0.10,
        "income": state.skill * 84.0,
    }
    return state, retrieval_state


def _rule_tasks() -> tuple[CapabilityTask, ...]:
    tasks: list[CapabilityTask] = []
    for index, (
        field,
        operator,
        value,
        target,
        direction,
        threshold,
        tolerance,
    ) in enumerate(_rule_cases(), start=1):
        state, retrieval_state = _current_state_for_rule(field, value)
        track = EvidenceLinkedEpisodicTrack(
            run_id=f"capability-rule-{index}",
            seed=2010922376,
            agent_id=index % 4,
            prompt_capacity=20,
        )
        for t in range(8):
            _append_episode(
                track,
                t=t,
                pre_state={
                    **retrieval_state,
                    "wealth": 40.0 + t,
                    "price": 1.0 + 0.05 * t,
                },
                labor_hours=float(48 + 8 * (t % 8)),
                consumption_fraction=float(0.20 + 0.04 * (t % 6)),
                flow_utility=float(0.5 + 0.1 * t),
            )
        semantic = VerifiedSemanticRuleTrack(track)
        condition = ConditionPredicate(
            field=field, operator=operator, value=value, tolerance=1e-9
        )
        guidance = ActionGuidance(
            target=target,
            direction=direction,
            threshold=threshold,
            tolerance=tolerance,
        )
        rule = semantic.inject_active_rule(
            condition=condition,
            action_guidance=guidance,
            outcome_criterion=OutcomeCriterion(
                metric="utility_advantage", operator=">", value=0.0, tolerance=0.0
            ),
            rationale="Frozen active rule for a production-shaped capability task.",
            current_t=8,
            injection_id=f"capability-rule-{index}",
            provenance={"source": "capability-v3-fixture"},
            initial_confidence=1.0,
            context_scope=ContextScope.global_scope(),
        )
        m1_summary = " ".join(
            f"month {t}: price={1.0 + 0.05 * t:.2f}, rate="
            f"{state.interest_rate:.2%}, wealth trend observed;"
            for t in range(8)
        )
        base = build_base_decision_prompt(
            state,
            _utility_config(),
            event_text="Use only information observed through the current month.",
            causal_context_summary=m1_summary,
        )
        memory_text = " ".join(
            [
                "Finalized experience evidence:",
                *(
                    f"- {episode.to_prompt_text()}"
                    for episode in track.finalized_episodes
                ),
                "Verified active rules:",
                f"- {rule.to_prompt_text()}",
            ]
        )
        prompt = compose_decision_prompt(base, memory_text).full_prompt
        tasks.append(
            CapabilityTask(
                task_id=f"rule-{index:02d}",
                category="rule-application",
                task_kind="rule_application",
                output_contract_id="actor-action",
                prompt=prompt,
                expected_condition=condition.to_dict(),
                expected_guidance=guidance.to_dict(),
                expected_scope=ContextScope.global_scope().to_dict(),
            )
        )
    return tuple(tasks)


def _condition_false_value(operator: str, value: float) -> float:
    if operator in {">", ">="}:
        return value - max(0.1, abs(value) * 0.5 + 0.01)
    if operator in {"<", "<="}:
        return value + max(0.1, abs(value) * 0.5 + 0.01)
    return value + 1.0


def _proposal_tasks() -> tuple[CapabilityTask, ...]:
    specifications = (
        ("inflation", ">=", 0.04, "consumption_fraction", "at_most", 0.30, 0.01),
        ("interest_rate", ">=", 0.08, "consumption_fraction", "at_most", 0.25, 0.01),
        ("wealth", "<=", 10.0, "consumption_fraction", "approximately", 0.30, 0.05),
        ("low_labor_rate", ">=", 0.50, "labor_hours", "at_least", 80.0, 0.0),
        ("price", ">=", 2.0, "consumption_fraction", "at_most", 0.40, 0.01),
        ("sentiment", "<=", -0.20, "labor_hours", "approximately", 84.0, 8.0),
    )
    tasks: list[CapabilityTask] = []
    for index, (
        field,
        operator,
        value,
        target,
        direction,
        threshold,
        tolerance,
    ) in enumerate(specifications, start=1):
        track = EvidenceLinkedEpisodicTrack(
            run_id=f"capability-proposal-{index}",
            seed=2010922376,
            agent_id=index % 4,
            prompt_capacity=12,
        )
        supports = []
        false_value = _condition_false_value(operator, value)
        utilities = (0.1, 1.0, 2.0, 3.0, 0.2, 0.1)
        for t in range(6):
            condition_value = value if 1 <= t <= 3 else false_value
            pre_state = {
                "price": 1.0,
                "interest_rate": 0.03,
                "low_labor_rate": 0.20,
                "unemployment_rate": 0.10,
                "inflation": 0.02,
                "sentiment": 0.10,
                "wealth": 50.0,
                "income": 80.0,
                field: condition_value,
            }
            labor_hours = threshold if target == "labor_hours" else 84.0
            consumption = (
                threshold if target == "consumption_fraction" else 0.30
            )
            episode = _append_episode(
                track,
                t=t,
                pre_state=pre_state,
                labor_hours=labor_hours,
                consumption_fraction=consumption,
                flow_utility=utilities[t],
            )
            if 1 <= t <= 3:
                supports.append(episode.episode_id)
        semantic = VerifiedSemanticRuleTrack(track)
        prompt = semantic.build_proposal_prompt(max_episodes=6, observed_through=6)
        condition = ConditionPredicate(
            field=field, operator=operator, value=value, tolerance=1e-9
        )
        guidance = ActionGuidance(
            target=target,
            direction=direction,
            threshold=threshold,
            tolerance=tolerance,
        )
        tasks.append(
            CapabilityTask(
                task_id=f"proposal-{index:02d}",
                category="rule-proposal",
                task_kind="rule_proposal",
                output_contract_id="semantic-proposal",
                prompt=prompt,
                expected_condition=condition.to_dict(),
                expected_guidance=guidance.to_dict(),
                expected_scope=ContextScope.global_scope().to_dict(),
                allowed_episode_ids=tuple(supports),
                proposal_track=track.to_dict(),
            )
        )
    return tuple(tasks)


@lru_cache(maxsize=1)
def build_capability_tasks() -> tuple[CapabilityTask, ...]:
    tasks = _action_tasks() + _rule_tasks() + _proposal_tasks()
    if len(tasks) != 30 or len({task.task_id for task in tasks}) != 30:
        raise RuntimeError("capability fixture must contain 30 unique tasks")
    return tasks


CAPABILITY_TASKSET_SHA256 = _canonical_hash(
    [task.to_dict() for task in build_capability_tasks()]
)


def _parse_mode(raw: str) -> str:
    stripped = raw.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, Mapping):
        return "exact_json"
    if re.fullmatch(
        r"```(?:json)?\s*.*?\s*```", stripped, re.DOTALL | re.IGNORECASE
    ):
        return "fenced_recovery"
    if "{" in stripped and "}" in stripped:
        return "substring_recovery"
    return "parse_failure"


def _exact_mapping(raw: str) -> Mapping[str, Any] | None:
    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, Mapping) else None


def _strict_action_schema(raw: str) -> bool:
    value = _exact_mapping(raw)
    if value is None or set(value) != {"reflection", "work", "consumption"}:
        return False
    if not isinstance(value["reflection"], str):
        return False
    for field in ("work", "consumption"):
        item = value[field]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not 0.0 <= float(item) <= 1.0
        ):
            return False
    return True


def _strict_candidate_schema(raw: str) -> bool:
    value = _exact_mapping(raw)
    return value is not None and set(value) == {
        "context_scope",
        "condition",
        "action_guidance",
        "rationale",
        "supporting_episode_ids",
    }


def _action_mapping(decision: Any) -> dict[str, float]:
    return {
        "labor_hours": decision.executed_labor_hours,
        "consumption_fraction": decision.executed_consumption_rate,
        "work_propensity": decision.proposed_work_fraction,
    }


def _score_action(
    task: CapabilityTask, raw: str
) -> tuple[bool, bool, dict[str, Any], str | None]:
    details: dict[str, Any] = {
        "action_parser_valid": False,
        "strict_schema_valid": False,
        "rule_compliant": None,
        "action": None,
        "semantic_candidate_accepted": None,
    }
    try:
        decision = parse_direct_action(raw)
    except (ActionParseError, TypeError, ValueError) as exc:
        return False, False, details, f"{type(exc).__name__}: {exc}"
    details["action_parser_valid"] = True
    details["strict_schema_valid"] = _strict_action_schema(raw)
    details["action"] = decision.to_dict()
    legal = not decision.clipped
    if task.task_kind == "rule_application":
        if task.expected_guidance is None:
            raise AssertionError("rule task is missing expected guidance")
        guidance = ActionGuidance.from_dict(task.expected_guidance)
        compliant = guidance.is_consistent(_action_mapping(decision))
        details["rule_compliant"] = compliant
        legal = legal and compliant
    return legal, details["strict_schema_valid"], details, None


def _score_proposal(
    task: CapabilityTask, raw: str
) -> tuple[bool, bool, dict[str, Any], str | None]:
    details: dict[str, Any] = {
        "action_parser_valid": None,
        "strict_schema_valid": _strict_candidate_schema(raw),
        "rule_compliant": None,
        "action": None,
        "semantic_candidate_accepted": False,
        "candidate_status": None,
        "candidate_supporting_episode_ids": [],
        "semantic_match": False,
    }
    if task.proposal_track is None:
        raise AssertionError("proposal task is missing its M2 fixture")
    track = EvidenceLinkedEpisodicTrack.from_dict(task.proposal_track)
    semantic = VerifiedSemanticRuleTrack(track)
    try:
        candidate = semantic.parse_candidate(
            raw, generator_id="capability-model"
        )
        details["candidate_supporting_episode_ids"] = list(
            candidate.supporting_episode_ids
        )
        rule = semantic.propose(
            raw,
            current_t=max(item.outcome_t for item in track.finalized_episodes),
            generator_id="capability-model",
        )
    except (CandidateParseError, TypeError, ValueError, KeyError) as exc:
        return (
            False,
            details["strict_schema_valid"],
            details,
            f"{type(exc).__name__}: {exc}",
        )
    support_ids = set(candidate.supporting_episode_ids)
    semantic_match = (
        task.expected_condition is not None
        and candidate.condition.to_dict() == dict(task.expected_condition)
        and task.expected_guidance is not None
        and candidate.action_guidance.to_dict() == dict(task.expected_guidance)
        and task.expected_scope is not None
        and candidate.context_scope.to_dict() == dict(task.expected_scope)
    )
    admitted = rule.status == "provisional"
    grounded_support_ids = set(rule.supporting_episode_ids)
    grounded_citations = (
        len(candidate.supporting_episode_ids) == len(support_ids)
        and len(support_ids) >= 2
        and support_ids <= grounded_support_ids
    )
    details["candidate_status"] = rule.status
    details["semantic_candidate_accepted"] = admitted
    # Exact recovery of the hidden fixture condition/guidance/scope is useful
    # diagnostically, but it is not proposal legality.  Production legality is
    # the verifier's provisional admission of at least two grounded citations.
    details["semantic_match"] = semantic_match
    return (
        admitted and grounded_citations,
        details["strict_schema_valid"],
        details,
        None,
    )


def _validated_output_contract(
    value: Mapping[str, Any], *, contract_id: str
) -> _TaskOutputContract:
    required = {
        "request_max_completion_tokens",
        "visible_json_max_bytes",
        "accepted_parse_modes",
        "required_finish_reason",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(
            f"task output contract {contract_id!r} is missing {missing}"
        )
    max_tokens = value["request_max_completion_tokens"]
    max_bytes = value["visible_json_max_bytes"]
    for name, item in (
        ("request_max_completion_tokens", max_tokens),
        ("visible_json_max_bytes", max_bytes),
    ):
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"{contract_id}.{name} must be a positive integer")
    raw_modes = value["accepted_parse_modes"]
    if isinstance(raw_modes, (str, bytes)) or not isinstance(raw_modes, Sequence):
        raise ValueError(f"{contract_id}.accepted_parse_modes must be a list")
    modes = tuple(str(item) for item in raw_modes)
    if not modes or len(modes) != len(set(modes)) or not set(modes) <= _PARSE_MODES:
        raise ValueError(
            f"{contract_id}.accepted_parse_modes contains invalid values"
        )
    finish = value["required_finish_reason"]
    if not isinstance(finish, str) or not finish:
        raise ValueError(
            f"{contract_id}.required_finish_reason must be non-empty"
        )
    return _TaskOutputContract(
        request_max_completion_tokens=max_tokens,
        visible_json_max_bytes=max_bytes,
        accepted_parse_modes=modes,
        required_finish_reason=finish,
    )


def _resolve_output_contracts(
    task_output_contracts: Mapping[str, Any] | None,
    caps: Mapping[str, Any] | None,
) -> dict[str, _TaskOutputContract]:
    source: dict[str, dict[str, Any]] = {
        key: dict(value) for key, value in _DEFAULT_TASK_OUTPUT_CONTRACTS.items()
    }
    if task_output_contracts is not None:
        if not isinstance(task_output_contracts, Mapping):
            raise TypeError("task_output_contracts must be a mapping or None")
        for contract_id in ("actor-action", "semantic-proposal"):
            value = task_output_contracts.get(contract_id)
            if not isinstance(value, Mapping) and hasattr(value, "to_dict"):
                value = value.to_dict()
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"task_output_contracts must define {contract_id!r}"
                )
            normalized = dict(value)
            # Pilot-contract-v2 names the frozen fields from the contract's
            # perspective.  The gate receipt uses explicit request/visible
            # names.  Accept both representations and normalize once here.
            if "max_completion_tokens" in normalized:
                normalized["request_max_completion_tokens"] = normalized[
                    "max_completion_tokens"
                ]
            if "max_visible_json_bytes" in normalized:
                normalized["visible_json_max_bytes"] = normalized[
                    "max_visible_json_bytes"
                ]
            if "accepted_parse_modes" not in normalized:
                if normalized.get("science_parse_mode") != "exact_json_only":
                    raise ValueError(
                        f"{contract_id}.science_parse_mode must be exact_json_only"
                    )
                normalized["accepted_parse_modes"] = ["exact_json"]
            source[contract_id] = normalized
    if caps is not None:
        if not isinstance(caps, Mapping):
            raise TypeError("caps must be a mapping or None")
        aliases = {
            "actor-action": "actor-action",
            "semantic-proposal": "semantic-proposal",
            "action_generation": "actor-action",
            "rule_application": "actor-action",
            "rule_proposal": "semantic-proposal",
            "utility-ranking": "actor-action",
            "rule-application": "actor-action",
            "rule-proposal": "semantic-proposal",
        }
        for key, value in caps.items():
            contract_id = aliases.get(str(key))
            if contract_id is None:
                continue
            if isinstance(value, int) and not isinstance(value, bool):
                source[contract_id]["request_max_completion_tokens"] = value
            elif isinstance(value, Mapping):
                source[contract_id].update(dict(value))
            else:
                raise ValueError(f"caps[{key!r}] must be an integer or mapping")
    return {
        contract_id: _validated_output_contract(value, contract_id=contract_id)
        for contract_id, value in source.items()
    }


def _completion_row(
    task: CapabilityTask,
    completion: StructuredCompletion,
    output_contract: _TaskOutputContract,
) -> dict[str, Any]:
    raw_bytes = len(completion.text.encode("utf-8"))
    visible_completion_tokens = max(
        completion.usage.completion_tokens - completion.reasoning_tokens, 0
    )
    parse_mode = _parse_mode(completion.text)
    parse_error: str | None = None
    parse_error_code: str | None = None
    legal = False
    strict_schema_valid = False
    scoring_details: dict[str, Any] = {}
    if task.task_kind == "rule_proposal":
        legal, strict_schema_valid, scoring_details, parse_error = _score_proposal(
            task, completion.text
        )
    else:
        legal, strict_schema_valid, scoring_details, parse_error = _score_action(
            task, completion.text
        )
    if parse_error is not None:
        parse_error_code = parse_error.split(":", 1)[0]
        parse_mode = "parse_failure"

    truncation = (
        completion.error_type == "IncompleteCompletionError"
        or completion.finish_reason == "length"
        or completion.native_finish_reason == "length"
    )
    provider_error = (
        not completion.ok
        or completion.text == "Error"
    ) and not truncation
    finish_valid = (
        completion.finish_reason == output_contract.required_finish_reason
        and completion.response_completed is True
    )
    within_visible_limit = raw_bytes <= output_contract.visible_json_max_bytes
    interface_valid = (
        not provider_error
        and not truncation
        and finish_valid
        and within_visible_limit
    )
    strict_parse = (
        parse_mode == "exact_json"
        and strict_schema_valid
        and parse_error is None
    )
    accepted_parse_mode = parse_mode in output_contract.accepted_parse_modes
    correct = (
        interface_valid
        and strict_parse
        and accepted_parse_mode
        and legal
    )

    if provider_error:
        interface_status = "provider_error"
    elif truncation:
        interface_status = "incomplete"
    elif not finish_valid:
        interface_status = "invalid_finish"
    elif not within_visible_limit:
        interface_status = "visible_limit_exceeded"
    elif parse_error is not None:
        interface_status = "parse_error"
    elif parse_mode != "exact_json":
        interface_status = "recovered_parse"
    else:
        interface_status = "pass"
    # Parse and semantic answer failures remain evaluable ITT outcomes.  A
    # transport, truncation, finish-contract, or visible-size failure does not.
    evaluable = interface_valid
    provider_error_details = (
        completion.provider_error_details.to_dict()
        if completion.provider_error_details is not None
        else None
    )
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "task_id": task.task_id,
        "category": task.category,
        "task_kind": task.task_kind,
        "output_contract_id": task.output_contract_id,
        "taskset_sha256": CAPABILITY_TASKSET_SHA256,
        "prompt_sha256": hashlib.sha256(task.prompt.encode("utf-8")).hexdigest(),
        "expected_choice": task.expected_choice,
        "predicted_choice": None,
        "correct": correct,
        "legal": legal,
        "parse_mode": parse_mode,
        "strict_parse": strict_parse,
        "accepted_parse_mode": accepted_parse_mode,
        "strict_schema_valid": strict_schema_valid,
        "parse_error": parse_error,
        "parse_error_code": parse_error_code,
        "parse_error_offset": None,
        "interface_status": interface_status,
        "interface_valid": interface_valid,
        "evaluable": evaluable,
        "truncation": truncation,
        "finish_contract_valid": finish_valid,
        "within_visible_limit": within_visible_limit,
        "request_max_completion_tokens": (
            output_contract.request_max_completion_tokens
        ),
        "visible_json_max_bytes": output_contract.visible_json_max_bytes,
        "visible_completion_tokens": visible_completion_tokens,
        "reasoning_tokens": completion.reasoning_tokens,
        **scoring_details,
        "provider_error": completion.error_type,
        "provider_error_details": provider_error_details,
        "provider": completion.provider,
        "requested_model": completion.model,
        "served_model": completion.response_model,
        "response_provider": completion.response_provider,
        "response_route": completion.response_route,
        "finish_reason": completion.finish_reason,
        "native_finish_reason": completion.native_finish_reason,
        "response_completed": completion.response_completed,
        "provider_sdk_name": completion.provider_sdk_name,
        "provider_sdk_version": completion.provider_sdk_version,
        "route_attestation_code": completion.route_attestation_code,
        "route_attestation_path": completion.route_attestation_path,
        "route_attestation_source": completion.route_attestation_source,
        "request_parameters": list(completion.request_parameters),
        "temperature_dispatch": completion.temperature_dispatch,
        "output_disposition": completion.output_disposition,
        "request_profile_id": completion.request_profile_id,
        "request_provider_pin": list(completion.request_provider_pin),
        "request_artifact_identity": dict(completion.request_artifact_identity),
        "request_price_snapshot_source": completion.request_price_snapshot_source,
        "request_price_snapshot_captured_at": (
            completion.request_price_snapshot_captured_at
        ),
        "request_seed": completion.request_seed,
        "attempts": completion.attempts,
        "usage": completion.usage.to_dict(),
        "cost_usd": completion.usage.cost_usd,
        "output_bytes": raw_bytes,
        "raw_output_sha256": hashlib.sha256(
            completion.text.encode("utf-8")
        ).hexdigest(),
    }


def run_capability_gate(
    *,
    llm: MultiModelLLM,
    budget: RunBudget,
    seed: int | None,
    estimate_usage: Callable[[str, int], UsageRecord],
    task_output_contracts: Mapping[str, Any] | None = None,
    caps: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all 30 fixed tasks; every provider/parse result stays in the ITT."""

    if not isinstance(llm, MultiModelLLM):
        raise TypeError("llm must be MultiModelLLM")
    if not isinstance(budget, RunBudget):
        raise TypeError("budget must be RunBudget")
    contracts = _resolve_output_contracts(task_output_contracts, caps)
    tasks = build_capability_tasks()

    def dispatch(
        selected: Sequence[CapabilityTask], contract_id: str
    ) -> list[StructuredCompletion]:
        max_tokens = contracts[contract_id].request_max_completion_tokens
        return llm.get_multiple_structured_completions(
            [[{"role": "user", "content": task.prompt}] for task in selected],
            temperature=0.0,
            max_tokens=max_tokens,
            top_p=1.0,
            budget=budget,
            labels=[f"capability:{task.task_id}" for task in selected],
            tags=[
                {
                    "call_kind": "capability",
                    "category": task.category,
                    "task_kind": task.task_kind,
                    "output_contract_id": contract_id,
                }
                for task in selected
            ],
            estimated_usages=[
                estimate_usage(task.prompt, max_tokens) for task in selected
            ],
            max_retries=1,
            seed=seed,
        )

    actor_tasks = tuple(
        task for task in tasks if task.output_contract_id == "actor-action"
    )
    proposal_tasks = tuple(
        task for task in tasks if task.output_contract_id == "semantic-proposal"
    )
    completion_by_id: dict[str, StructuredCompletion] = {}
    for task, completion in zip(
        actor_tasks, dispatch(actor_tasks, "actor-action")
    ):
        completion_by_id[task.task_id] = completion
    for task, completion in zip(
        proposal_tasks, dispatch(proposal_tasks, "semantic-proposal")
    ):
        completion_by_id[task.task_id] = completion
    rows = [
        _completion_row(
            task,
            completion_by_id[task.task_id],
            contracts[task.output_contract_id],
        )
        for task in tasks
    ]

    totals: dict[str, dict[str, Any]] = {}
    for category, threshold in CAPABILITY_THRESHOLDS.items():
        category_rows = [row for row in rows if row["category"] == category]
        registered_correct = sum(bool(row["correct"]) for row in category_rows)
        registered_total = len(category_rows)
        evaluable_count = sum(bool(row["evaluable"]) for row in category_rows)
        conditional_correct = sum(
            bool(row["correct"]) for row in category_rows if row["evaluable"]
        )
        conditional_accuracy = (
            conditional_correct / evaluable_count if evaluable_count else None
        )
        totals[category] = {
            "correct": registered_correct,
            "denominator": registered_total,
            "required": threshold,
            "registered_correct": registered_correct,
            "registered_total": registered_total,
            "evaluable_count": evaluable_count,
            "conditional_correct": conditional_correct,
            "conditional_accuracy": conditional_accuracy,
            "interface_failure_count": registered_total - evaluable_count,
        }
    checks = {
        category: value["registered_correct"] >= value["required"]
        for category, value in totals.items()
    }
    conditional_checks = {
        category: (
            None
            if value["conditional_accuracy"] is None
            else value["conditional_accuracy"]
            >= value["required"] / value["registered_total"]
        )
        for category, value in totals.items()
    }
    interface_failure_count = sum(not row["evaluable"] for row in rows)
    if interface_failure_count > 0:
        capability_status = "not_evaluable"
        capability_pass = None
    else:
        capability_pass = all(checks.values())
        capability_status = "pass" if capability_pass else "fail"
    interface_gate = {
        "pass": interface_failure_count == 0,
        "failure_count": interface_failure_count,
    }
    capability_assessment = {
        "status": capability_status,
        "pass": capability_pass,
        "checks": conditional_checks,
    }
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "taskset_sha256": CAPABILITY_TASKSET_SHA256,
        "provider_model": llm.get_model_name(),
        "seed": seed,
        "pass": interface_gate["pass"] and capability_pass is True,
        "checks": checks,
        "category_totals": totals,
        "task_output_contracts": {
            key: value.to_dict() for key, value in contracts.items()
        },
        "interface_gate": interface_gate,
        "capability_assessment": capability_assessment,
        "provider_failure_count": sum(
            row["provider_error"] is not None for row in rows
        ),
        "parse_failure_count": sum(
            row["parse_mode"] == "parse_failure" for row in rows
        ),
        "recovered_parse_count": sum(
            row["parse_mode"] in {"fenced_recovery", "substring_recovery"}
            for row in rows
        ),
        "strict_parse_count": sum(bool(row["strict_parse"]) for row in rows),
        "truncation_count": sum(bool(row["truncation"]) for row in rows),
        "rows": rows,
        "budget": budget.snapshot().to_dict(),
    }


__all__ = [
    "CAPABILITY_SCHEMA_VERSION",
    "CAPABILITY_TASKSET_SHA256",
    "CAPABILITY_THRESHOLDS",
    "CapabilityTask",
    "build_capability_tasks",
    "run_capability_gate",
]
