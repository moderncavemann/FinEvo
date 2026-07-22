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
import random
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
from .m3_semantic import CandidateParseError
from .prompts import (
    DecisionPromptState,
    build_base_decision_prompt,
    compose_decision_prompt,
)
from .scripted_provider import DIAGNOSTIC_PROVIDER_NAME
from .system import MemoryBundle, VerifiedDualTrackMemory


RUNNER_SCHEMA_VERSION = "verified-simulation-runner-v1"
CONTEXT_FEATURES = (
    "log_price",
    "interest_rate",
    "low_labor_rate",
    "inflation",
    "log_wealth",
    "employed",
)


class VerifiedRunError(RuntimeError):
    """Raised when a fail-closed simulation gate is violated."""


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
class VerifiedRunConfig:
    run_id: str
    seed: int = 7
    num_agents: int = 2
    episode_length: int = 6
    context_mode: str = "retrieval-only"
    enable_episodic_retrieval: bool = True
    enable_semantic: bool = True
    retrieval_k: int = 5
    rule_budget: int = 3
    semantic_proposal_after: int = 3
    semantic_proposal_interval: int = 3
    max_rule_proposals_per_agent: int = 1
    labor_step: float = 8.0
    max_labor_hours: float = 168.0
    consumption_step: float = 0.02
    low_labor_threshold_hours: float = 1.0
    temperature: float = 0.0
    top_p: float = 1.0
    action_max_tokens: int = 220
    rule_max_tokens: int = 450
    max_retries: int = 1
    fail_on_clipped_action: bool = True
    fail_on_rule_parse_error: bool = False
    utility: UtilityConfig = field(default_factory=UtilityConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        for name in (
            "seed",
            "num_agents",
            "episode_length",
            "retrieval_k",
            "rule_budget",
            "semantic_proposal_after",
            "semantic_proposal_interval",
            "max_rule_proposals_per_agent",
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
        if self.semantic_proposal_after < 2:
            raise ValueError("semantic proposals require at least two completed periods")
        for name in (
            "semantic_proposal_interval",
            "max_rule_proposals_per_agent",
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
        normalized_mode = self.context_mode.strip().lower().replace("_", "-")
        if normalized_mode not in CONTEXT_MODES:
            raise ValueError(f"unsupported context mode: {self.context_mode}")
        object.__setattr__(self, "context_mode", normalized_mode)
        for name in (
            "labor_step",
            "max_labor_hours",
            "consumption_step",
            "low_labor_threshold_hours",
            "temperature",
            "top_p",
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
        if not 0 <= self.low_labor_threshold_hours <= self.max_labor_hours:
            raise ValueError("low labor threshold is outside feasible hours")
        if self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError("invalid decoding parameters")
        if not isinstance(self.utility, UtilityConfig):
            raise TypeError("utility must be UtilityConfig")
        if not math.isclose(
            self.utility.max_labor_hours, self.max_labor_hours, abs_tol=1e-12
        ):
            raise ValueError("utility and action maximum labor hours must match")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["utility"] = self.utility.to_dict()
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
        "usage": result.usage.to_dict(),
        "request_seed": result.request_seed,
        "system_fingerprint": result.system_fingerprint,
        "response_model": result.response_model,
        "cached_prompt_tokens": result.cached_prompt_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "provider_request_id": result.request_id,
        "raw_output_hash": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
    }


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
    low_labor_rate: float,
    inflation: float,
    wealth: float,
    employed: bool,
) -> dict[str, Any]:
    return {
        "timestamp": int(decision_t),
        "log_price": math.log1p(float(price)),
        "interest_rate": float(interest_rate),
        "low_labor_rate": float(low_labor_rate),
        "inflation": float(inflation),
        "log_wealth": math.log1p(float(wealth)),
        "employed": float(bool(employed)),
    }


def _m2_state(snapshot: Any, *, low_labor_rate: float, inflation: float) -> dict[str, Any]:
    state = snapshot.to_m2_state()
    state["low_labor_rate"] = float(low_labor_rate)
    state["unemployment_rate"] = float(low_labor_rate)
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
            enable_episodic_retrieval=config.enable_episodic_retrieval,
            enable_semantic=config.enable_semantic,
        )
    return systems


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

    np.random.seed(config.seed)
    random.seed(config.seed)
    foundation_config = prepare_foundation_env_config(
        env_config_source,
        n_agents=config.num_agents,
        episode_length=config.episode_length,
        labor_step=config.labor_step,
        max_labor_hours=config.max_labor_hours,
    )
    env = foundation.make_env_instance(**foundation_config)
    env.reset()
    ledger = EnvironmentLedger(config.utility)
    memories = _prepare_memories(config)
    provider_model_name = llm.get_model_name()
    diagnostic_only = provider_model_name.startswith(f"{DIAGNOSTIC_PROVIDER_NAME}/")

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
        "macro_steps": [],
        "errors": [],
    }
    last_decisions: dict[str, ActionDecision] = {}
    last_transitions: dict[str, Any] = {}
    proposals_made = {agent_id: 0 for agent_id in range(config.num_agents)}
    semantic_event_offsets = {agent_id: 0 for agent_id in range(config.num_agents)}
    previous_low_labor_rate = 1.0
    completed_periods = 0

    for decision_t in range(config.episode_length):
        pre_snapshots = capture_foundation_snapshots(
            env, expected_timestamp=decision_t
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
            )
            prompt = compose_decision_prompt(base_prompt, bundle.prompt)
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
                    "decision_t": decision_t,
                    "agent_id": agent_id,
                    "environment_state_hash": _sha256(retrieval_state),
                    "base_prompt": prompt.base_prompt,
                    "memory_text": prompt.memory_text,
                    "full_prompt_hash": prompt.full_prompt_hash,
                    "base_prompt_hash": prompt.base_prompt_hash,
                    "memory_hash": prompt.memory_hash,
                    "context_packet_id": bundle.context_packet.context_id,
                    "context_packet_hash": bundle.context_packet.context_hash,
                    "provider_model": provider_model_name,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "diagnostic_only": diagnostic_only,
                }
            )

        estimates = [
            _estimate_usage(
                prompt_rows[index].full_prompt,
                max_tokens=config.action_max_tokens,
                provider_model_name=provider_model_name,
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
            seed=config.seed,
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
        _, rewards, done, _ = env.step(env_actions)
        transitions = derive_foundation_transitions(
            env,
            pre_snapshots=pre_snapshots,
            decisions=decisions,
            expected_outcome_t=decision_t + 1,
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
                    _estimate_usage(
                        prompt,
                        max_tokens=config.rule_max_tokens,
                        provider_model_name=provider_model_name,
                    )
                    for prompt in proposal_prompts
                ],
                max_retries=config.max_retries,
                seed=config.seed,
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
                    "diagnostic_only": diagnostic_only,
                }
                if completion.ok and completion.text != "Error":
                    try:
                        rule = memories[agent_id].submit_rule_proposal(
                            completion.text,
                            current_t=current_t,
                            generator_id=provider_model_name,
                        )
                        proposal_row["rule_id"] = rule.rule_id
                        proposal_row["rule_status"] = rule.status
                    except CandidateParseError as exc:
                        proposal_row["parse_error"] = str(exc)
                        records["errors"].append(
                            {
                                **usage_row,
                                "error_type": "CandidateParseError",
                                "message": str(exc),
                            }
                        )
                        if config.fail_on_rule_parse_error:
                            raise VerifiedRunError(str(exc)) from exc
                else:
                    records["errors"].append(usage_row)
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
    active_rule_ids = {
        row["rule_id"] for row in records["semantic_rules"] if row["status"] == "active"
    }
    selected_rule_ids = {
        rule_id
        for row in action_rows
        for rule_id in row["selected_rule_ids"]
    }
    checks = {
        "completed_all_periods": completed_periods == config.episode_length,
        "action_count_t_by_n": len(action_rows) == expected_rows,
        "episode_count_t_by_n": len(records["episodes"]) == expected_rows,
        "utility_count_t_by_n": len(records["utility_ledger"]) == expected_rows,
        "no_provider_or_parse_errors": len(records["errors"]) == 0,
        "causal_context": all(
            row["context_packet"]["observed_through"] <= row["decision_t"]
            for row in records["context_trace"]
        ),
        "episode_alignment": all(
            row["outcome_t"] == row["decision_t"] + 1
            for row in records["episodes"]
        ),
        "direct_intermediate_labor_observed": bool(intermediate_actions),
        "budget_identity": all(
            abs(row["budget_residual"]) <= config.utility.budget_tolerance
            for row in records["utility_ledger"]
        ),
    }
    semantic_check = (
        not config.enable_semantic
        or bool(active_rule_ids & selected_rule_ids)
    )
    checks["semantic_rule_activated_and_retrieved"] = semantic_check
    validation_pass = all(checks.values())
    validation_status = {
        "status": "pass" if validation_pass else "fail",
        "checks": checks,
        "diagnostic_only": diagnostic_only,
        "scientific_evidence": False,
    }
    utility_values = [row["flow_utility"] for row in records["utility_ledger"]]
    summary = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "run_id": config.run_id,
        "provider_model": provider_model_name,
        "diagnostic_only": diagnostic_only,
        "scientific_evidence": False,
        "result_scope": "bounded_method_smoke",
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
            "clipped_action_count": sum(
                bool(row["decision"]["clipped"]) for row in action_rows
            ),
        },
        "memory_diagnostics": {
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
        },
        "api": budget.snapshot().to_dict(),
        "validation": validation_status,
    }
    frozen_records = {
        name: tuple(_json_copy(row) for row in rows) for name, rows in records.items()
    }
    sealed_config = config.to_dict()
    sealed_config["foundation_env"] = _json_copy(foundation_config)
    sealed_config["foundation_env_hash"] = _sha256(foundation_config)
    return VerifiedRunResult(
        config=_json_copy(sealed_config),
        summary=_json_copy(summary),
        validation_status=_json_copy(validation_status),
        budget_snapshot=_json_copy(budget.snapshot().to_dict()),
        records=frozen_records,
    )


__all__ = [
    "CONTEXT_FEATURES",
    "RUNNER_SCHEMA_VERSION",
    "VerifiedRunConfig",
    "VerifiedRunError",
    "VerifiedRunResult",
    "run_verified_experiment",
]
