"""Fixed 30-item actor/proposer capability gate for the pilot model matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from llm_providers import MultiModelLLM, StructuredCompletion

from .budget import RunBudget, UsageRecord
from .m0_utility import UtilityConfig, realized_flow_utility
from .m2_episodic import EvidenceLinkedEpisodicTrack
from .m3_semantic import CandidateParseError, VerifiedSemanticRuleTrack


CAPABILITY_SCHEMA_VERSION = "finevo-capability-gate-v2"
CAPABILITY_THRESHOLDS = {
    "utility-ranking": 10,
    "rule-application": 10,
    "rule-proposal": 5,
}


@dataclass(frozen=True, slots=True)
class CapabilityTask:
    task_id: str
    category: str
    prompt: str
    expected_choice: str | None
    allowed_episode_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utility_tasks() -> tuple[CapabilityTask, ...]:
    config = UtilityConfig(
        rho=1.0,
        labor_weight=2.0,
        inverse_frisch=1.0,
        consumption_scale=1.0,
        max_labor_hours=168.0,
        discount_factor=0.99,
    )
    pairs = (
        ((1.0, 42.0), (0.5, 42.0)),
        ((0.8, 84.0), (0.8, 126.0)),
        ((2.0, 84.0), (1.0, 42.0)),
        ((0.5, 0.0), (1.0, 168.0)),
        ((4.0, 126.0), (2.0, 84.0)),
        ((0.25, 42.0), (0.5, 84.0)),
        ((1.5, 168.0), (0.75, 42.0)),
        ((3.0, 84.0), (2.0, 42.0)),
        ((0.1, 0.0), (0.2, 84.0)),
        ((5.0, 168.0), (2.5, 84.0)),
        ((1.25, 63.0), (1.0, 42.0)),
        ((0.75, 21.0), (1.0, 84.0)),
    )
    tasks = []
    for index, (left, right) in enumerate(pairs, start=1):
        left_utility = realized_flow_utility(left[0], left[1], config).flow_utility
        right_utility = realized_flow_utility(right[0], right[1], config).flow_utility
        if abs(left_utility - right_utility) <= 1e-6:
            raise RuntimeError("frozen utility capability task is ambiguous")
        expected = "A" if left_utility > right_utility else "B"
        prompt = (
            "Use exactly this realized monthly utility: "
            "log(1+q/1) - 2/(1+1)*(h/168)^(1+1). "
            f"Option A has q={left[0]:g}, h={left[1]:g}; "
            f"Option B has q={right[0]:g}, h={right[1]:g}. "
            'Return ONLY JSON {"choice":"A"} or {"choice":"B"} for the '
            "higher-utility option."
        )
        tasks.append(
            CapabilityTask(
                task_id=f"utility-{index:02d}",
                category="utility-ranking",
                prompt=prompt,
                expected_choice=expected,
            )
        )
    return tuple(tasks)


def _rule_tasks() -> tuple[CapabilityTask, ...]:
    cases = (
        ("interest_rate >= 0.08", "consumption_fraction at_most 0.25", 0.08, ("0.20", "0.60"), "A"),
        ("interest_rate >= 0.08", "consumption_fraction at_most 0.25", 0.10, ("0.50", "0.10"), "B"),
        ("inflation >= 0.04", "labor_hours at_least 84", 0.05, ("40", "88"), "B"),
        ("inflation >= 0.04", "labor_hours at_least 84", 0.08, ("128", "64"), "A"),
        ("wealth <= 10", "consumption_fraction approximately 0.30 tolerance 0.05", 8.0, ("0.32", "0.80"), "A"),
        ("wealth <= 10", "consumption_fraction approximately 0.30 tolerance 0.05", 2.0, ("0.05", "0.28"), "B"),
        ("low_labor_rate >= 0.50", "labor_hours at_least 80", 0.75, ("88", "0"), "A"),
        ("low_labor_rate >= 0.50", "labor_hours at_least 80", 0.60, ("40", "120"), "B"),
        ("price >= 2", "consumption_fraction at_most 0.40", 3.0, ("0.80", "0.35"), "B"),
        ("unemployment_rate >= 0.20", "work_propensity at_least 0.50", 0.25, ("0.25", "0.75"), "B"),
        ("sentiment <= -0.20", "consumption_fraction at_most 0.20", -0.5, ("0.10", "0.50"), "A"),
        ("income >= 100", "labor_hours approximately 84 tolerance 8", 120.0, ("84", "140"), "A"),
    )
    tasks = []
    for index, (condition, guidance, state, actions, expected) in enumerate(
        cases, start=1
    ):
        prompt = (
            f"A verified rule is active: IF {condition} THEN {guidance}. "
            f"The current condition field value is {state:g}, so the rule applies. "
            f"Candidate action A sets the target to {actions[0]}; candidate action B "
            f"sets it to {actions[1]}. Return ONLY JSON "
            '{"choice":"A"} or {"choice":"B"} for the action consistent with the rule.'
        )
        tasks.append(
            CapabilityTask(
                task_id=f"rule-{index:02d}",
                category="rule-application",
                prompt=prompt,
                expected_choice=expected,
            )
        )
    return tuple(tasks)


def _proposal_tasks() -> tuple[CapabilityTask, ...]:
    specifications = (
        ("interest_rate", ">=", 0.08, "consumption_fraction", "at_most", 0.30),
        ("inflation", ">=", 0.04, "labor_hours", "at_least", 84.0),
        ("wealth", "<=", 10.0, "consumption_fraction", "approximately", 0.30),
        ("low_labor_rate", ">=", 0.50, "work_propensity", "at_least", 0.50),
        ("price", ">=", 2.0, "consumption_fraction", "at_most", 0.40),
        ("sentiment", "<=", -0.20, "labor_hours", "approximately", 84.0),
    )
    tasks = []
    for index, (field, operator, value, target, direction, threshold) in enumerate(
        specifications, start=1
    ):
        episode_ids = (f"cap-{index}-a", f"cap-{index}-b")
        evidence = [
            {
                "episode_id": episode_ids[0],
                "pre_state": {field: value},
                "executed_action": {target: threshold},
                "utility_advantage": 0.5,
            },
            {
                "episode_id": episode_ids[1],
                "pre_state": {field: value + (0.01 if operator in {">", ">="} else -0.01)},
                "executed_action": {target: threshold},
                "utility_advantage": 0.2,
            },
        ]
        schema = {
            "context_scope": {"scope_id": "global", "predicates": []},
            "condition": {
                "field": field,
                "operator": operator,
                "value": value,
                "tolerance": 0.0,
            },
            "action_guidance": {
                "target": target,
                "direction": direction,
                "threshold": threshold,
                "tolerance": 0.05,
            },
            "rationale": "one non-empty sentence",
            "supporting_episode_ids": list(episode_ids),
        }
        prompt = (
            "Propose one semantic decision rule from the two successful episodes. "
            "Return ONLY one JSON object matching this key/type template; do not add "
            f"an outcome criterion:\n{json.dumps(schema, sort_keys=True)}\n"
            f"Evidence:\n{json.dumps(evidence, sort_keys=True)}"
        )
        tasks.append(
            CapabilityTask(
                task_id=f"proposal-{index:02d}",
                category="rule-proposal",
                prompt=prompt,
                expected_choice=None,
                allowed_episode_ids=episode_ids,
            )
        )
    return tuple(tasks)


def build_capability_tasks() -> tuple[CapabilityTask, ...]:
    tasks = _utility_tasks() + _rule_tasks() + _proposal_tasks()
    if len(tasks) != 30 or len({task.task_id for task in tasks}) != 30:
        raise RuntimeError("capability fixture must contain 30 unique tasks")
    return tasks


CAPABILITY_TASKSET_SHA256 = _canonical_hash(
    [task.to_dict() for task in build_capability_tasks()]
)


def _exact_choice(raw: str) -> str:
    value = json.loads(raw.strip())
    if not isinstance(value, Mapping) or set(value) != {"choice"}:
        raise ValueError("choice response must contain exactly one choice key")
    choice = value["choice"]
    if choice not in {"A", "B"}:
        raise ValueError("choice must be A or B")
    return str(choice)


def _proposal_is_legal(raw: str, allowed_episode_ids: Sequence[str]) -> bool:
    track = VerifiedSemanticRuleTrack(
        EvidenceLinkedEpisodicTrack(
            run_id="capability-fixture",
            seed=0,
            agent_id=0,
        )
    )
    candidate = track.parse_candidate(raw, generator_id="capability-model")
    supplied = set(candidate.supporting_episode_ids)
    allowed = set(allowed_episode_ids)
    return len(supplied) >= 2 and supplied <= allowed


def _completion_row(
    task: CapabilityTask, completion: StructuredCompletion
) -> dict[str, Any]:
    predicted = None
    parse_error = None
    parse_error_code = None
    parse_error_offset = None
    correct = False
    legal = False
    incomplete = (
        completion.error_type == "IncompleteCompletionError"
        or completion.finish_reason == "length"
    )
    if incomplete:
        interface_status = "incomplete"
    elif (
        not completion.ok
        or completion.text == "Error"
    ):
        interface_status = "provider_error"
    elif (
        completion.finish_reason != "stop"
        or completion.response_completed is not True
    ):
        interface_status = "invalid_finish"
    else:
        interface_status = "pass"
        try:
            if task.category == "rule-proposal":
                legal = _proposal_is_legal(
                    completion.text, task.allowed_episode_ids
                )
                correct = legal
            else:
                predicted = _exact_choice(completion.text)
                legal = True
                correct = predicted == task.expected_choice
        except (CandidateParseError, json.JSONDecodeError, TypeError, ValueError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            parse_error_code = type(exc).__name__
            offset = getattr(exc, "pos", None)
            if isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0:
                parse_error_offset = offset
            interface_status = "parse_error"
    # A malformed model answer is still an evaluable registered task outcome:
    # it is scored incorrect in the fixed ITT denominator.  Only transport/
    # contract failures and incomplete completions make capability itself
    # unevaluable.  This preserves the preregistered 10/12, 10/12, and 5/6
    # thresholds instead of silently imposing a new 30/30 parseability gate.
    evaluable = interface_status in {"pass", "parse_error"}
    provider_error_details = (
        completion.provider_error_details.to_dict()
        if completion.provider_error_details is not None
        else None
    )
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "task_id": task.task_id,
        "category": task.category,
        "taskset_sha256": CAPABILITY_TASKSET_SHA256,
        "prompt_sha256": hashlib.sha256(task.prompt.encode("utf-8")).hexdigest(),
        "expected_choice": task.expected_choice,
        "predicted_choice": predicted,
        "correct": correct,
        "legal": legal,
        "parse_error": parse_error,
        "parse_error_code": parse_error_code,
        "parse_error_offset": parse_error_offset,
        "interface_status": interface_status,
        "evaluable": evaluable,
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
        "request_artifact_identity": dict(
            completion.request_artifact_identity
        ),
        "request_price_snapshot_source": (
            completion.request_price_snapshot_source
        ),
        "request_price_snapshot_captured_at": (
            completion.request_price_snapshot_captured_at
        ),
        "request_seed": completion.request_seed,
        "attempts": completion.attempts,
        "usage": completion.usage.to_dict(),
        "output_bytes": len(completion.text.encode("utf-8")),
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
) -> dict[str, Any]:
    """Run all 30 fixed tasks; every provider/parse result stays in the denominator."""

    if not isinstance(llm, MultiModelLLM):
        raise TypeError("llm must be MultiModelLLM")
    if not isinstance(budget, RunBudget):
        raise TypeError("budget must be RunBudget")
    tasks = build_capability_tasks()
    choice_tasks = tasks[:24]
    proposal_tasks = tasks[24:]

    def dispatch(
        selected: Sequence[CapabilityTask], max_tokens: int
    ) -> list[StructuredCompletion]:
        return llm.get_multiple_structured_completions(
            [[{"role": "user", "content": task.prompt}] for task in selected],
            temperature=0.0,
            max_tokens=max_tokens,
            top_p=1.0,
            budget=budget,
            labels=[f"capability:{task.task_id}" for task in selected],
            tags=[
                {"call_kind": "capability", "category": task.category}
                for task in selected
            ],
            estimated_usages=[
                estimate_usage(task.prompt, max_tokens) for task in selected
            ],
            max_retries=1,
            seed=seed,
        )

    completions = dispatch(choice_tasks, 80) + dispatch(proposal_tasks, 450)
    rows = [
        _completion_row(task, completion)
        for task, completion in zip(tasks, completions)
    ]
    totals = {}
    for category, threshold in CAPABILITY_THRESHOLDS.items():
        category_rows = [row for row in rows if row["category"] == category]
        registered_correct = sum(row["correct"] for row in category_rows)
        registered_total = len(category_rows)
        evaluable_count = sum(row["evaluable"] for row in category_rows)
        conditional_correct = sum(
            row["correct"] for row in category_rows if row["evaluable"]
        )
        conditional_accuracy = (
            conditional_correct / evaluable_count if evaluable_count else None
        )
        totals[category] = {
            # Compatibility aliases retained for v1 readers.
            "correct": registered_correct,
            "denominator": registered_total,
            "required": threshold,
            # V2 separates registered ITT accounting from evaluable answers.
            "registered_correct": registered_correct,
            "registered_total": registered_total,
            "evaluable_count": evaluable_count,
            "conditional_correct": conditional_correct,
            "conditional_accuracy": conditional_accuracy,
            "interface_failure_count": registered_total - evaluable_count,
        }
    checks = {
        category: value["correct"] >= value["required"]
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
        capability_pass = all(conditional_checks.values())
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
    overall_pass = interface_gate["pass"] and capability_pass is True
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "taskset_sha256": CAPABILITY_TASKSET_SHA256,
        "provider_model": llm.get_model_name(),
        "seed": seed,
        "pass": overall_pass,
        "checks": checks,
        "category_totals": totals,
        "interface_gate": interface_gate,
        "capability_assessment": capability_assessment,
        "provider_failure_count": sum(
            row["provider_error"] is not None for row in rows
        ),
        "parse_failure_count": sum(row["parse_error"] is not None for row in rows),
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
