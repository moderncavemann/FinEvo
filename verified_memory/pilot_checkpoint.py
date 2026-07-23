"""Auditable Foundation checkpoint/replay primitives for the FinEvo pilot.

The checkpoint is intentionally a replay recipe instead of a pickle.  Foundation
already records the NumPy state immediately before reset and every step; restoring
those states and replaying the exact action prefix reconstructs the environment
without depending on private object layout.  Economically relevant full-state
hashes, the utility ledger, every agent's memory snapshot, and the experiment
contract are checked before a continuation is released.

This module is a standalone pilot path.  It does not alter the verified runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any, Mapping, Optional, Sequence

import numpy as np

import ai_economist.foundation as foundation
from llm_providers import MultiModelLLM

from .actions import ActionDecision, parse_direct_action
from .budget import RunBudget
from .foundation_adapter import (
    FoundationTransition,
    build_foundation_actions,
    capture_foundation_snapshots,
    derive_foundation_transitions,
    prepare_foundation_env_config,
)
from .m0_utility import EnvironmentLedger, UtilityConfig
from .m3_semantic import CandidateParseError, OutcomeCriterion
from .prompts import build_base_decision_prompt, compose_decision_prompt
from .runner import (
    ShockEvent,
    VerifiedRunConfig,
    _context_observation,
    _m2_state,
    _monthly_inflation,
    _prepare_memories,
    _prompt_state,
    preflight_p95_reservation_for_call,
    validate_preflight_p95_reservations,
)
from .system import VerifiedDualTrackMemory


PILOT_CHECKPOINT_SCHEMA_VERSION = "finevo-pilot-checkpoint-v1"
DEFAULT_BRANCH_DECISION_T = 6
_CODE_FILES = (
    "verified_memory/actions.py",
    "verified_memory/foundation_adapter.py",
    "verified_memory/m0_utility.py",
    "verified_memory/m1_context.py",
    "verified_memory/m2_episodic.py",
    "verified_memory/m3_semantic.py",
    "verified_memory/prompts.py",
    "verified_memory/runner.py",
    "verified_memory/system.py",
    "verified_memory/pilot_checkpoint.py",
    "verified_memory/pilot_continuation.py",
    "ai_economist/foundation/base/base_env.py",
)


class PilotCheckpointError(ValueError):
    """Raised when a checkpoint cannot be constructed or restored exactly."""


def canonical_hash(value: Any) -> str:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )


def encode_numpy_rng_state(state: Sequence[Any]) -> dict[str, Any]:
    """Encode ``np.random.get_state()`` without losing integer precision."""

    if not isinstance(state, (tuple, list)) or len(state) != 5:
        raise TypeError("NumPy RNG state must contain five fields")
    return {
        "bit_generator": str(state[0]),
        "keys": [int(value) for value in np.asarray(state[1], dtype=np.uint32)],
        "position": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def decode_numpy_rng_state(value: Mapping[str, Any]) -> tuple[Any, ...]:
    expected = {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PilotCheckpointError("invalid encoded NumPy RNG state")
    return (
        str(value["bit_generator"]),
        np.asarray(value["keys"], dtype=np.uint32),
        int(value["position"]),
        int(value["has_gauss"]),
        float(value["cached_gaussian"]),
    )


def encode_python_rng_state(state: tuple[Any, ...]) -> dict[str, Any]:
    if not isinstance(state, tuple) or len(state) != 3:
        raise TypeError("Python RNG state must contain three fields")
    return {
        "version": int(state[0]),
        "internal": [int(value) for value in state[1]],
        "gauss_next": None if state[2] is None else float(state[2]),
    }


def decode_python_rng_state(value: Mapping[str, Any]) -> tuple[Any, ...]:
    if not isinstance(value, Mapping) or set(value) != {
        "version",
        "internal",
        "gauss_next",
    }:
        raise PilotCheckpointError("invalid encoded Python RNG state")
    return (
        int(value["version"]),
        tuple(int(item) for item in value["internal"]),
        None if value["gauss_next"] is None else float(value["gauss_next"]),
    )


def _jsonable(value: Any) -> Any:
    """Canonicalize Foundation state while rejecting opaque mutable objects."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise PilotCheckpointError("non-finite Foundation state")
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        converted = [_jsonable(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
    raise PilotCheckpointError(
        f"opaque mutable Foundation state type: {type(value).__name__}"
    )


def capture_environment_state(env: Any) -> dict[str, Any]:
    """Capture the replay-validated Foundation state used by this pilot.

    Agent/planner state, all serializable world state, map tensors, and component
    state are included. Back-pointers, CUDA managers, and static action-space
    objects are excluded because replay reconstructs them from the bound config.
    """

    world = env.world
    world_state: dict[str, Any] = {}
    for name, value in vars(world).items():
        if name in {"_agents", "_planner", "maps", "cuda_data_manager", "cuda_function_manager"}:
            continue
        world_state[name] = _jsonable(value)
    world_state["maps"] = _jsonable(world.maps.state_dict)
    agents = {
        str(agent.idx): {
            "state": _jsonable(agent.state),
            "action": _jsonable(agent.action),
        }
        for agent in world.agents
    }
    planner = {
        "state": _jsonable(world.planner.state),
        "action": _jsonable(world.planner.action),
    }
    component_state: dict[str, Any] = {}
    for component in env._components:
        values: dict[str, Any] = {}
        for name, value in vars(component).items():
            if name in {
                "_world",
                "common_mask_off",
                "common_mask_on",
                "_planner_masks",
            }:
                continue
            try:
                values[name] = _jsonable(value)
            except PilotCheckpointError:
                # Static schedules/callables are code/config bound.  Opaque values
                # are never used as the sole proof of a successful restore.
                continue
        component_state[type(component).__name__] = values
    return {
        "timestep": int(world.timestep),
        "world": world_state,
        "agents": agents,
        "planner": planner,
        "components": component_state,
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def current_code_binding() -> dict[str, Any]:
    root = _repo_root()
    hashes: dict[str, str] = {}
    for relative in _CODE_FILES:
        path = root / relative
        if not path.is_file():
            raise PilotCheckpointError(f"missing code-binding file: {relative}")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    binding = {"source_hashes": hashes}
    binding["binding_hash"] = canonical_hash(binding)
    return binding


def config_from_dict(value: Mapping[str, Any]) -> VerifiedRunConfig:
    if not isinstance(value, Mapping):
        raise TypeError("run config must be a mapping")
    config = dict(value)
    config.pop("schema_version", None)
    config["utility"] = UtilityConfig(**dict(config["utility"]))
    config["registered_outcome_criterion"] = OutcomeCriterion.from_dict(
        config["registered_outcome_criterion"]
    )
    config["shock_schedule"] = tuple(
        ShockEvent(**dict(event))
        for event in config.get("shock_schedule", ())
    )
    return VerifiedRunConfig(**config)


def _decision_from_dict(value: Mapping[str, Any]) -> ActionDecision:
    return ActionDecision(**dict(value))


def _snapshot_dicts(snapshots: Mapping[str, Any]) -> dict[str, Any]:
    return {agent_id: snapshot.to_dict() for agent_id, snapshot in snapshots.items()}


def _shock_by_decision_t(
    config: VerifiedRunConfig,
) -> dict[int, ShockEvent]:
    return {event.decision_t: event for event in config.shock_schedule}


def _apply_shock(env: Any, event: Optional[ShockEvent]) -> None:
    if event is None:
        return
    rates = getattr(getattr(env, "world", None), "interest_rate", None)
    if not isinstance(rates, list) or not rates:
        raise PilotCheckpointError(
            "Foundation environment has no current interest-rate state"
        )
    rates[-1] = float(event.interest_rate)


def _shock_prompt_text(event: Optional[ShockEvent]) -> str:
    return "" if event is None else event.to_prompt_text()


def _assert_no_future_shock_leak(
    base_prompt: str,
    *,
    current_event: Optional[ShockEvent],
    schedule: Sequence[ShockEvent],
    decision_t: int,
) -> None:
    current_text = _shock_prompt_text(current_event)
    if current_text and current_text not in base_prompt:
        raise PilotCheckpointError(
            f"current shock event missing from prompt at t={decision_t}"
        )
    for future in schedule:
        if future.decision_t <= decision_t:
            continue
        future_text = future.to_prompt_text()
        if future_text != current_text and future_text in base_prompt:
            raise PilotCheckpointError(
                f"future shock event leaked into prompt at t={decision_t}"
            )


def _pre_ledger_batch(
    snapshots: Mapping[str, Any], decisions: Mapping[str, ActionDecision]
) -> dict[str, dict[str, Any]]:
    return {
        agent_id: {
            "wealth": snapshot.wealth,
            "cumulative_production": snapshot.cumulative_production,
            "price": snapshot.price,
            "interest_rate": snapshot.interest_rate,
            "proposed_work_propensity": decisions[
                agent_id
            ].proposed_work_fraction,
            "proposed_consumption_fraction": decisions[
                agent_id
            ].proposed_consumption_fraction,
            "executed_labor_hours": decisions[agent_id].executed_labor_hours,
            "executed_consumption_rate": decisions[
                agent_id
            ].executed_consumption_rate,
        }
        for agent_id, snapshot in snapshots.items()
    }


def _post_ledger_batch(
    transitions: Mapping[str, FoundationTransition],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for agent_id, transition in transitions.items():
        row = transition.to_m0_post().to_dict()
        row.pop("period")
        row.pop("agent_id")
        result[agent_id] = row
    return result


@dataclass(frozen=True)
class PilotCheckpoint:
    """Validated immutable wrapper around a JSON checkpoint payload."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        copied = _json_copy(dict(self.payload))
        object.__setattr__(self, "payload", copied)
        self.validate()

    @property
    def checkpoint_hash(self) -> str:
        return str(self.payload["checkpoint_hash"])

    @property
    def next_decision_t(self) -> int:
        return int(self.payload["next_decision_t"])

    def validate(self) -> None:
        if self.payload.get("schema_version") != PILOT_CHECKPOINT_SCHEMA_VERSION:
            raise PilotCheckpointError("unsupported pilot checkpoint schema")
        body = dict(self.payload)
        claimed = body.pop("checkpoint_hash", None)
        if claimed != canonical_hash(body):
            raise PilotCheckpointError("checkpoint hash mismatch")
        prefix_steps = self.payload.get("prefix_steps")
        if not isinstance(prefix_steps, list) or not prefix_steps:
            raise PilotCheckpointError("checkpoint requires a non-empty action prefix")
        if [row.get("decision_t") for row in prefix_steps] != list(
            range(len(prefix_steps))
        ):
            raise PilotCheckpointError("prefix decision timestamps are not contiguous")
        if self.next_decision_t != len(prefix_steps):
            raise PilotCheckpointError("next_decision_t does not follow prefix")
        if self.payload.get("prefix_hash") != canonical_hash(prefix_steps):
            raise PilotCheckpointError("prefix hash mismatch")
        memories = self.payload.get("memories")
        if not isinstance(memories, Mapping) or len(memories) != int(
            self.payload["run_config"]["num_agents"]
        ):
            raise PilotCheckpointError("checkpoint memory cohort mismatch")
        if self.payload.get("memories_hash") != canonical_hash(memories):
            raise PilotCheckpointError("memory snapshot hash mismatch")
        if self.payload.get("ledger_hash") != canonical_hash(
            self.payload.get("ledger_records")
        ):
            raise PilotCheckpointError("ledger hash mismatch")
        if self.next_decision_t != DEFAULT_BRANCH_DECISION_T:
            raise PilotCheckpointError(
                "checkpoint must branch after t=5/before decision 6"
            )
        expected_agents = {
            str(agent_id)
            for agent_id in range(int(self.payload["run_config"]["num_agents"]))
        }
        proposals_made = self.payload.get("proposals_made")
        if (
            not isinstance(proposals_made, Mapping)
            or set(proposals_made) != expected_agents
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != 2
                for value in proposals_made.values()
            )
        ):
            raise PilotCheckpointError(
                "checkpoint must bind two proposal attempts per agent"
            )
        if self.payload.get("previous_state_hash") != canonical_hash(
            self.payload.get("previous_state")
        ):
            raise PilotCheckpointError("previous environment state hash mismatch")
        contract = {
            "schema_version": PILOT_CHECKPOINT_SCHEMA_VERSION,
            "run_config": self.payload["run_config"],
            "foundation_env_config": self.payload["foundation_env_config"],
            "branch_after_decision_t": self.next_decision_t - 1,
        }
        if self.payload.get("contract_hash") != canonical_hash(contract):
            raise PilotCheckpointError("canonical contract hash mismatch")

    def to_dict(self) -> dict[str, Any]:
        return _json_copy(self.payload)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PilotCheckpoint":
        return cls(payload=value)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                self.payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "PilotCheckpoint":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(value)


@dataclass
class RestoredPilotState:
    config: VerifiedRunConfig
    foundation_env_config: Mapping[str, Any]
    env: Any
    memories: dict[int, VerifiedDualTrackMemory]
    ledger: EnvironmentLedger
    proposals_made: dict[int, int]
    previous_low_labor_rate: Optional[float]
    last_decisions: dict[str, ActionDecision]
    last_transitions: dict[str, FoundationTransition]
    next_decision_t: int
    prefix_hash: str
    checkpoint_hash: str


def _provider_binding(llm: MultiModelLLM) -> dict[str, Any]:
    provider = llm.provider
    state = None
    checkpoint_state = getattr(provider, "checkpoint_state", None)
    if callable(checkpoint_state):
        state = _json_copy(checkpoint_state())
    return {
        "model_name": llm.get_model_name(),
        "state": state,
        "state_hash": None if state is None else canonical_hash(state),
    }


def _complete_actions(
    *,
    config: VerifiedRunConfig,
    llm: MultiModelLLM,
    budget: RunBudget,
    decision_t: int,
    prompts: Sequence[str],
) -> list[Any]:
    estimates = [
        preflight_p95_reservation_for_call(
            config,
            provider_model_name=llm.get_model_name(),
            call_kind="action",
            prompt=prompt,
            max_tokens=config.action_max_tokens,
        )
        for prompt in prompts
    ]
    return llm.get_multiple_structured_completions(
        [[{"role": "user", "content": prompt}] for prompt in prompts],
        temperature=config.temperature,
        max_tokens=config.action_max_tokens,
        top_p=config.top_p,
        budget=budget,
        labels=[
            f"pilot-prefix-action:t{decision_t}:a{agent_id}"
            for agent_id in range(config.num_agents)
        ],
        tags=[
            {
                "call_kind": "pilot_prefix_action",
                "decision_t": decision_t,
                "agent_id": agent_id,
            }
            for agent_id in range(config.num_agents)
        ],
        estimated_usages=estimates,
        max_retries=config.max_retries,
        seed=config.seed if config.send_decoding_seed else None,
    )


def build_pilot_checkpoint(
    config: VerifiedRunConfig,
    *,
    llm: MultiModelLLM,
    budget: RunBudget,
    env_config_source: Mapping[str, Any] | str | Path,
    prefix_periods: int = DEFAULT_BRANCH_DECISION_T,
) -> PilotCheckpoint:
    """Run an exact prefix and freeze it immediately before ``decision_t=6``.

    The default is the preregistered Experiment-D branch point: after outcomes
    for decisions 0..5 have been finalized and before any decision-6 retrieval.
    """

    if not isinstance(config, VerifiedRunConfig):
        raise TypeError("config must be VerifiedRunConfig")
    if not isinstance(llm, MultiModelLLM):
        raise TypeError("llm must be MultiModelLLM")
    if not isinstance(budget, RunBudget):
        raise TypeError("budget must be RunBudget")
    validate_preflight_p95_reservations(
        config,
        provider_model_name=llm.get_model_name(),
    )
    if isinstance(prefix_periods, bool) or not isinstance(prefix_periods, int):
        raise TypeError("prefix_periods must be an integer")
    if prefix_periods != DEFAULT_BRANCH_DECISION_T:
        raise ValueError(
            "finevo-pilot-checkpoint-v1 fixes the branch after t=5/before decision 6"
        )
    if config.num_agents != 4:
        raise ValueError("Experiment D checkpoint requires exactly four agents")
    if config.episode_length < prefix_periods:
        raise ValueError("episode_length is shorter than checkpoint prefix")
    if getattr(config, "error_rule_mode", "none") != "none":
        raise ValueError(
            "Experiment D checkpoint must precede standalone erroneous-rule "
            "branch injection; set error_rule_mode='none'"
        )
    if getattr(config, "semantic_policy", "evidence-grounded") != (
        "evidence-grounded"
    ):
        raise ValueError(
            "Experiment D source checkpoint requires evidence-grounded semantics"
        )
    if not config.enable_semantic:
        raise ValueError("Experiment D source checkpoint requires semantic memory")
    if (
        config.semantic_proposal_after != 3
        or config.semantic_proposal_interval != 3
    ):
        raise ValueError(
            "Experiment D checkpoint fixes semantic proposals at outcome "
            "months 3 and 6 before branching"
        )
    if config.max_rule_proposals_per_agent < 2:
        raise ValueError(
            "Experiment D checkpoint requires capacity for the outcome-month "
            "3 and outcome-month 6 proposals"
        )
    if (
        config.freeze_new_proposals_after is not None
        and config.freeze_new_proposals_after < prefix_periods
    ):
        raise ValueError(
            "Experiment D proposal freeze must start no earlier than the "
            "outcome-month 6 proposal"
        )

    np.random.seed(config.seed)
    random.seed(config.seed)
    python_rng_at_start = encode_python_rng_state(random.getstate())
    foundation_config = prepare_foundation_env_config(
        env_config_source,
        n_agents=config.num_agents,
        episode_length=config.episode_length,
        labor_step=config.labor_step,
        max_labor_hours=config.max_labor_hours,
        consumption_step=config.consumption_step,
    )
    numpy_rng_before_env_construction = encode_numpy_rng_state(
        np.random.get_state()
    )
    env = foundation.make_env_instance(**foundation_config)
    env.reset()
    reset_seed_state = encode_numpy_rng_state(
        env.replay_log["reset"]["seed_state"]
    )
    memories = _prepare_memories(config)
    ledger = EnvironmentLedger(config.utility)
    proposals_made = {agent_id: 0 for agent_id in range(config.num_agents)}
    last_decisions: dict[str, ActionDecision] = {}
    last_transitions: dict[str, FoundationTransition] = {}
    previous_low_labor_rate: Optional[float] = None
    prefix_steps: list[dict[str, Any]] = []
    shocks = _shock_by_decision_t(config)

    for decision_t in range(prefix_periods):
        shock = shocks.get(decision_t)
        _apply_shock(env, shock)
        pre_environment_state = capture_environment_state(env)
        pre_snapshots = capture_foundation_snapshots(
            env,
            expected_timestamp=decision_t,
            labor_step=config.labor_step,
            max_labor_hours=config.max_labor_hours,
            consumption_step=config.consumption_step,
        )
        current_inflation = _monthly_inflation(env.world)
        bundles: dict[int, Any] = {}
        prompts: list[str] = []
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
            base = build_base_decision_prompt(
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
                event_text=_shock_prompt_text(shock),
                causal_context_summary=bundle.protected_context_prompt,
            )
            _assert_no_future_shock_leak(
                base,
                current_event=shock,
                schedule=config.shock_schedule,
                decision_t=decision_t,
            )
            prompt = compose_decision_prompt(base, bundle.memory_prompt)
            bundles[agent_id] = bundle
            prompts.append(prompt.full_prompt)

        completions = _complete_actions(
            config=config,
            llm=llm,
            budget=budget,
            decision_t=decision_t,
            prompts=prompts,
        )
        decisions: dict[str, ActionDecision] = {}
        for agent_id, completion in enumerate(completions):
            if not completion.ok or completion.text == "Error":
                raise PilotCheckpointError(
                    f"prefix provider failure at t={decision_t}, agent={agent_id}"
                )
            decision = parse_direct_action(
                completion.text,
                max_labor_hours=config.max_labor_hours,
                labor_step=config.labor_step,
                consumption_step=config.consumption_step,
            )
            if config.fail_on_clipped_action and decision.clipped:
                raise PilotCheckpointError(
                    f"clipped prefix action at t={decision_t}, agent={agent_id}"
                )
            agent_key = str(agent_id)
            decisions[agent_key] = decision
            pre_state = _m2_state(
                pre_snapshots[agent_key],
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

        pre_batch = _pre_ledger_batch(pre_snapshots, decisions)
        ledger.capture_pre(decision_t, pre_batch)
        env_actions = build_foundation_actions(
            decisions,
            labor_step=config.labor_step,
            max_labor_hours=config.max_labor_hours,
            consumption_step=config.consumption_step,
        )
        # Re-apply immediately before the transition so prompt and environment
        # consume the same current-period shock even if intervening code changes
        # a mutable Foundation list.
        _apply_shock(env, shock)
        python_step_seed_state = encode_python_rng_state(random.getstate())
        _, rewards, done, _ = env.step(env_actions)
        step_seed_state = encode_numpy_rng_state(
            env.replay_log["step"][-1]["seed_state"]
        )
        transitions = derive_foundation_transitions(
            env,
            pre_snapshots=pre_snapshots,
            decisions=decisions,
            expected_outcome_t=decision_t + 1,
            labor_step=config.labor_step,
            max_labor_hours=config.max_labor_hours,
            consumption_step=config.consumption_step,
        )
        post_batch = _post_ledger_batch(transitions)
        utility_rows = ledger.capture_post(decision_t, post_batch)
        rows_by_agent = {row.agent_id: row for row in utility_rows}
        current_low_labor_rate = mean(
            float(
                decision.executed_labor_hours
                < config.low_labor_threshold_hours
            )
            for decision in decisions.values()
        )
        realized_inflation = _monthly_inflation(env.world)
        for agent_id in range(config.num_agents):
            agent_key = str(agent_id)
            decision = decisions[agent_key]
            transition = transitions[agent_key]
            memories[agent_id].finalize_episode(
                decision_t=decision_t,
                next_state=_m2_state(
                    transition.post,
                    low_labor_rate=current_low_labor_rate,
                    inflation=realized_inflation,
                ),
                outcome=transition.to_m2_outcome(decision),
                reward=float(rewards[agent_key]),
                flow_utility=rows_by_agent[agent_key].flow_utility,
            )
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
        proposal_not_frozen = (
            config.freeze_new_proposals_after is None
            or current_t <= config.freeze_new_proposals_after
        )
        eligible = [
            agent_id
            for agent_id in range(config.num_agents)
            if proposal_due
            and proposal_not_frozen
            and proposals_made[agent_id]
            < config.max_rule_proposals_per_agent
        ]
        if eligible:
            proposal_prompts = [
                memories[agent_id].build_rule_proposal_prompt(max_episodes=6)
                for agent_id in eligible
            ]
            proposal_results = llm.get_multiple_structured_completions(
                [
                    [{"role": "user", "content": prompt}]
                    for prompt in proposal_prompts
                ],
                temperature=0.0,
                max_tokens=config.rule_max_tokens,
                top_p=1.0,
                budget=budget,
                labels=[
                    f"pilot-prefix-semantic:t{current_t}:a{agent_id}"
                    for agent_id in eligible
                ],
                tags=[
                    {
                        "call_kind": "pilot_prefix_semantic",
                        "current_t": current_t,
                        "agent_id": agent_id,
                    }
                    for agent_id in eligible
                ],
                estimated_usages=[
                    preflight_p95_reservation_for_call(
                        config,
                        provider_model_name=llm.get_model_name(),
                        call_kind="semantic",
                        prompt=prompt,
                        max_tokens=config.rule_max_tokens,
                    )
                    for prompt in proposal_prompts
                ],
                max_retries=config.max_retries,
                seed=config.seed if config.send_decoding_seed else None,
            )
            for agent_id, completion in zip(eligible, proposal_results):
                proposals_made[agent_id] += 1
                if not completion.ok or completion.text == "Error":
                    raise PilotCheckpointError(
                        f"prefix semantic provider failure at t={current_t}, "
                        f"agent={agent_id}"
                    )
                try:
                    memories[agent_id].submit_rule_proposal(
                        completion.text,
                        current_t=current_t,
                        generator_id=llm.get_model_name(),
                    )
                except CandidateParseError:
                    if config.semantic_parse_failure_policy == "fail-run":
                        raise

        post_environment_state = capture_environment_state(env)
        prefix_steps.append(
            {
                "decision_t": decision_t,
                "shock_event": None if shock is None else shock.to_dict(),
                "shock_event_hash": canonical_hash(
                    None if shock is None else shock.to_dict()
                ),
                "shock_prompt_text": _shock_prompt_text(shock),
                "pre_environment_hash": canonical_hash(pre_environment_state),
                "post_environment_hash": canonical_hash(post_environment_state),
                "pre_snapshots": _snapshot_dicts(pre_snapshots),
                "decisions": {
                    key: value.to_dict() for key, value in decisions.items()
                },
                "foundation_actions": _json_copy(env_actions),
                "step_seed_state": step_seed_state,
                "python_step_seed_state": python_step_seed_state,
                "pre_ledger_batch": pre_batch,
                "post_ledger_batch": post_batch,
                "rewards": _jsonable(rewards),
                "done": bool(done["__all__"]),
                "low_labor_rate": current_low_labor_rate,
                "monthly_inflation": realized_inflation,
            }
        )
        previous_low_labor_rate = current_low_labor_rate

    expected_proposal_count = sum(
        current_t >= config.semantic_proposal_after
        and (current_t - config.semantic_proposal_after)
        % config.semantic_proposal_interval
        == 0
        and (
            config.freeze_new_proposals_after is None
            or current_t <= config.freeze_new_proposals_after
        )
        for current_t in range(1, prefix_periods + 1)
    )
    expected_proposal_count = min(
        expected_proposal_count, config.max_rule_proposals_per_agent
    )
    if expected_proposal_count != 2 or any(
        count != expected_proposal_count for count in proposals_made.values()
    ):
        raise PilotCheckpointError(
            "Experiment D checkpoint must contain exactly the outcome-month "
            "3 and outcome-month 6 proposal attempts for every agent"
        )

    for memory in memories.values():
        memory.validate()
    previous_state = capture_environment_state(env)
    memory_rows = {
        str(agent_id): memory.to_dict()
        for agent_id, memory in memories.items()
    }
    ledger_records = ledger.records()
    run_config = config.to_dict()
    contract = {
        "schema_version": PILOT_CHECKPOINT_SCHEMA_VERSION,
        "run_config": run_config,
        "foundation_env_config": foundation_config,
        "branch_after_decision_t": prefix_periods - 1,
    }
    payload: dict[str, Any] = {
        "schema_version": PILOT_CHECKPOINT_SCHEMA_VERSION,
        "next_decision_t": prefix_periods,
        "run_config": run_config,
        "foundation_env_config": foundation_config,
        "contract_hash": canonical_hash(contract),
        "code_binding": current_code_binding(),
        "provider_binding": _provider_binding(llm),
        "numpy_rng_before_env_construction": (
            numpy_rng_before_env_construction
        ),
        "foundation_reset_seed_state": reset_seed_state,
        "python_rng_at_start": python_rng_at_start,
        "numpy_rng_after_prefix": encode_numpy_rng_state(np.random.get_state()),
        "python_rng_after_prefix": encode_python_rng_state(random.getstate()),
        "prefix_steps": prefix_steps,
        "prefix_hash": canonical_hash(prefix_steps),
        "previous_state": previous_state,
        "previous_state_hash": canonical_hash(previous_state),
        "last_decisions": {
            key: value.to_dict() for key, value in last_decisions.items()
        },
        "memories": memory_rows,
        "memories_hash": canonical_hash(memory_rows),
        "proposals_made": {
            str(key): value for key, value in proposals_made.items()
        },
        "previous_low_labor_rate": previous_low_labor_rate,
        "ledger_records": ledger_records,
        "ledger_hash": canonical_hash(ledger_records),
    }
    payload["checkpoint_hash"] = canonical_hash(payload)
    return PilotCheckpoint(payload)


def restore_pilot_checkpoint(
    checkpoint: PilotCheckpoint | Mapping[str, Any],
    *,
    strict_code_binding: bool = True,
) -> RestoredPilotState:
    """Restore and independently verify environment, memory, and ledger state."""

    if not isinstance(checkpoint, PilotCheckpoint):
        checkpoint = PilotCheckpoint.from_dict(checkpoint)
    payload = checkpoint.payload
    if strict_code_binding and payload["code_binding"] != current_code_binding():
        raise PilotCheckpointError("checkpoint code binding differs from current code")
    config = config_from_dict(payload["run_config"])
    foundation_config = _json_copy(payload["foundation_env_config"])
    random.setstate(
        decode_python_rng_state(payload["python_rng_at_start"])
    )
    np.random.set_state(
        decode_numpy_rng_state(
            payload["numpy_rng_before_env_construction"]
        )
    )
    env = foundation.make_env_instance(**foundation_config)
    env.reset(
        seed_state=decode_numpy_rng_state(
            payload["foundation_reset_seed_state"]
        )
    )
    ledger = EnvironmentLedger(config.utility)
    last_decisions: dict[str, ActionDecision] = {}
    last_transitions: dict[str, FoundationTransition] = {}

    for step in payload["prefix_steps"]:
        decision_t = int(step["decision_t"])
        expected_shock = _shock_by_decision_t(config).get(decision_t)
        expected_shock_row = (
            None if expected_shock is None else expected_shock.to_dict()
        )
        if step.get("shock_event") != expected_shock_row:
            raise PilotCheckpointError(
                f"checkpoint shock binding mismatch at t={decision_t}"
            )
        if step.get("shock_event_hash") != canonical_hash(expected_shock_row):
            raise PilotCheckpointError(
                f"checkpoint shock hash mismatch at t={decision_t}"
            )
        if step.get("shock_prompt_text") != _shock_prompt_text(expected_shock):
            raise PilotCheckpointError(
                f"checkpoint shock prompt binding mismatch at t={decision_t}"
            )
        _apply_shock(env, expected_shock)
        pre_state = capture_environment_state(env)
        if canonical_hash(pre_state) != step["pre_environment_hash"]:
            raise PilotCheckpointError(
                f"pre-generated environment mismatch at t={decision_t}"
            )
        pre_snapshots = capture_foundation_snapshots(
            env,
            expected_timestamp=decision_t,
            labor_step=config.labor_step,
            max_labor_hours=config.max_labor_hours,
            consumption_step=config.consumption_step,
        )
        if _snapshot_dicts(pre_snapshots) != step["pre_snapshots"]:
            raise PilotCheckpointError(
                f"pre-generated economic snapshot mismatch at t={decision_t}"
            )
        decisions = {
            key: _decision_from_dict(value)
            for key, value in step["decisions"].items()
        }
        generated_actions = build_foundation_actions(
            decisions,
            labor_step=config.labor_step,
            max_labor_hours=config.max_labor_hours,
            consumption_step=config.consumption_step,
        )
        if generated_actions != step["foundation_actions"]:
            raise PilotCheckpointError(
                f"Foundation action mapping mismatch at t={decision_t}"
            )
        ledger.capture_pre(decision_t, step["pre_ledger_batch"])
        _apply_shock(env, expected_shock)
        random.setstate(
            decode_python_rng_state(step["python_step_seed_state"])
        )
        if encode_python_rng_state(random.getstate()) != step[
            "python_step_seed_state"
        ]:
            raise PilotCheckpointError(
                f"pre-generated Python RNG mismatch at t={decision_t}"
            )
        _, rewards, done, _ = env.step(
            generated_actions,
            seed_state=decode_numpy_rng_state(step["step_seed_state"]),
        )
        replay_seed = encode_numpy_rng_state(
            env.replay_log["step"][-1]["seed_state"]
        )
        if replay_seed != step["step_seed_state"]:
            raise PilotCheckpointError(
                f"pre-generated RNG mismatch at t={decision_t}"
            )
        transitions = derive_foundation_transitions(
            env,
            pre_snapshots=pre_snapshots,
            decisions=decisions,
            expected_outcome_t=decision_t + 1,
            labor_step=config.labor_step,
            max_labor_hours=config.max_labor_hours,
            consumption_step=config.consumption_step,
        )
        generated_post = _post_ledger_batch(transitions)
        if generated_post != step["post_ledger_batch"]:
            raise PilotCheckpointError(
                f"post-step accounting snapshot mismatch at t={decision_t}"
            )
        ledger.capture_post(decision_t, generated_post)
        if _jsonable(rewards) != step["rewards"] or bool(done["__all__"]) != step["done"]:
            raise PilotCheckpointError(
                f"reward/done replay mismatch at t={decision_t}"
            )
        post_state = capture_environment_state(env)
        if canonical_hash(post_state) != step["post_environment_hash"]:
            raise PilotCheckpointError(
                f"post-generated environment mismatch at t={decision_t}"
            )
        last_decisions = decisions
        last_transitions = transitions

    final_state = capture_environment_state(env)
    if final_state != payload["previous_state"]:
        raise PilotCheckpointError("restored Foundation state is not exact")
    if ledger.records() != payload["ledger_records"]:
        raise PilotCheckpointError("restored utility ledger is not exact")
    memories = {
        int(agent_id): VerifiedDualTrackMemory.from_dict(value)
        for agent_id, value in payload["memories"].items()
    }
    restored_memory_rows = {
        str(agent_id): memory.to_dict()
        for agent_id, memory in memories.items()
    }
    if restored_memory_rows != payload["memories"]:
        raise PilotCheckpointError("restored memory snapshot is not exact")
    np.random.set_state(
        decode_numpy_rng_state(payload["numpy_rng_after_prefix"])
    )
    random.setstate(
        decode_python_rng_state(payload["python_rng_after_prefix"])
    )
    return RestoredPilotState(
        config=config,
        foundation_env_config=foundation_config,
        env=env,
        memories=memories,
        ledger=ledger,
        proposals_made={
            int(key): int(value)
            for key, value in payload["proposals_made"].items()
        },
        previous_low_labor_rate=payload["previous_low_labor_rate"],
        last_decisions=last_decisions,
        last_transitions=last_transitions,
        next_decision_t=checkpoint.next_decision_t,
        prefix_hash=str(payload["prefix_hash"]),
        checkpoint_hash=checkpoint.checkpoint_hash,
    )


__all__ = [
    "DEFAULT_BRANCH_DECISION_T",
    "PILOT_CHECKPOINT_SCHEMA_VERSION",
    "PilotCheckpoint",
    "PilotCheckpointError",
    "RestoredPilotState",
    "build_pilot_checkpoint",
    "canonical_hash",
    "capture_environment_state",
    "config_from_dict",
    "current_code_binding",
    "decode_numpy_rng_state",
    "encode_numpy_rng_state",
    "restore_pilot_checkpoint",
]
