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
from llm_providers import MultiModelLLM, StructuredCompletion

from .actions import ActionDecision, ActionParseError, parse_direct_action
from .budget import BudgetExceeded, RunBudget
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
    _append_provider_call_journal,
    _action_parse_mode,
    _context_observation,
    _m2_state,
    _monthly_inflation,
    _prepare_memories,
    _prompt_state,
    _provider_row,
    _semantic_parse_mode,
    preflight_p95_reservation_for_call,
    validate_preflight_p95_reservations,
    verify_provider_call_journal,
)
from .system import VerifiedDualTrackMemory


PILOT_CHECKPOINT_SCHEMA_VERSION = "finevo-pilot-checkpoint-v1"
PILOT_CHECKPOINT_SCHEMA_VERSION_V2 = "finevo-pilot-checkpoint-v2"
CLOSED_LOOP_PREFLIGHT_CHECKPOINT_PURPOSE = "closed-loop-preflight"
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


def _supported_checkpoint_schema(value: Any) -> str:
    if value not in {
        PILOT_CHECKPOINT_SCHEMA_VERSION,
        PILOT_CHECKPOINT_SCHEMA_VERSION_V2,
    }:
        raise PilotCheckpointError("unsupported pilot checkpoint schema")
    return str(value)


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


def _validate_v2_provider_evidence(
    payload: Mapping[str, Any],
    *,
    run_config: Mapping[str, Any],
) -> None:
    rows = payload.get("provider_calls")
    if (
        not isinstance(rows, list)
        or len(rows) != 16
        or payload.get("provider_calls_hash") != canonical_hash(rows)
    ):
        raise PilotCheckpointError(
            "v2 provider-call denominator/hash is invalid"
        )
    if [row.get("call_index") for row in rows] != list(range(16)):
        raise PilotCheckpointError("v2 provider call indices are not exact")
    expected_action_cells = {
        (decision_t, agent_id)
        for decision_t in range(6)
        for agent_id in range(2)
    }
    expected_semantic_cells = {
        (current_t, agent_id)
        for current_t in (3, 6)
        for agent_id in range(2)
    }
    observed_action_cells: set[tuple[int, int]] = set()
    observed_semantic_cells: set[tuple[int, int]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise PilotCheckpointError("v2 provider row must be an object")
        call_kind = row.get("call_kind")
        try:
            cell = (int(row["decision_t"]), int(row["agent_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise PilotCheckpointError(
                "v2 provider row lacks a valid task cell"
            ) from exc
        if call_kind == "action":
            observed_action_cells.add(cell)
            max_tokens = int(run_config["action_max_tokens"])
            max_bytes = int(run_config["action_max_visible_json_bytes"])
        elif call_kind == "semantic":
            observed_semantic_cells.add(cell)
            max_tokens = int(run_config["rule_max_tokens"])
            max_bytes = int(run_config["rule_max_visible_json_bytes"])
        else:
            raise PilotCheckpointError("v2 provider row has an invalid call kind")
        usage = row.get("usage")
        disposition = row.get("parse_disposition")
        dispatch = row.get("parameter_dispatch")
        if (
            not isinstance(usage, Mapping)
            or int(usage.get("prompt_tokens", 0)) < 1
            or int(usage.get("completion_tokens", 0)) < 1
            or not isinstance(disposition, Mapping)
            or row.get("attempts") != 1
            or row.get("finish_reason") != "stop"
            or row.get("response_completed") is not True
            or row.get("output_disposition") != "accepted"
        ):
            raise PilotCheckpointError(
                "v2 provider row is not a successful terminal call"
            )
        exact_accepted = (
            disposition.get("parse_status") == "success"
            and disposition.get("parse_mode") == "exact_json"
            and disposition.get("accepted") is True
        )
        recorded_semantic_failure = (
            call_kind == "semantic"
            and disposition.get("parse_status") == "failure"
            and disposition.get("accepted") is False
            and disposition.get("parse_mode")
            in {
                "exact_json",
                "fenced_recovery",
                "substring_recovery",
                "parse_failure",
            }
        )
        if not exact_accepted and not recorded_semantic_failure:
            raise PilotCheckpointError(
                "v2 parse disposition violates exact-action/"
                "record-and-skip semantics"
            )
        if (
            not isinstance(dispatch, Mapping)
            or set(dispatch) != _V2_PARAMETER_DISPATCH_FIELDS
            or not set(dispatch.values())
            <= _V2_PARAMETER_DISPATCH_STATUSES
        ):
            raise PilotCheckpointError(
                "v2 provider parameter dispatch is incomplete"
            )
        if any(
            not isinstance(row.get(name), str) or not row[name].strip()
            for name in (
                "model",
                "provider",
                "served_model",
                "served_provider",
                "served_route",
                "request_profile_id",
                "request_price_snapshot_source",
                "request_price_snapshot_captured_at",
                "native_finish_reason",
                "provider_sdk_name",
                "provider_sdk_version",
                "temperature_dispatch",
                "prompt_hash",
                "raw_output_hash",
            )
        ):
            raise PilotCheckpointError(
                "v2 provider requested/served/route metadata is incomplete"
            )
        if (
            not isinstance(row.get("request_provider_pin"), list)
            or not row["request_provider_pin"]
            or not isinstance(row.get("request_artifact_identity"), Mapping)
            or not row["request_artifact_identity"]
            or not isinstance(row.get("request_parameters"), list)
            or not row["request_parameters"]
        ):
            raise PilotCheckpointError(
                "v2 provider request metadata is incomplete"
            )
        task_cap = row.get("task_cap")
        if task_cap != {
            "max_visible_tokens": max_tokens,
            "max_visible_json_bytes": max_bytes,
        }:
            raise PilotCheckpointError(
                "v2 provider row task cap differs from the run config"
            )
        if (
            int(row.get("visible_completion_tokens", -1)) < 0
            or int(row["visible_completion_tokens"]) > max_tokens
            or int(row.get("visible_output_bytes", -1)) < 0
            or int(row["visible_output_bytes"]) > max_bytes
            or row.get("raw_output_bytes") != row.get("visible_output_bytes")
        ):
            raise PilotCheckpointError(
                "v2 provider row exceeds its visible output cap"
            )
        provider = str(row["provider"]).strip().lower()
        if (
            provider not in {"diagnostic", "local", "ollama"}
            and float(usage.get("cost_usd", 0.0)) <= 0
        ):
            raise PilotCheckpointError(
                "hosted v2 provider row lacks positive cost"
            )
        if (
            provider not in {"diagnostic", "local", "ollama"}
            and (
                not isinstance(row.get("provider_request_id"), str)
                or not row["provider_request_id"]
            )
        ):
            raise PilotCheckpointError(
                "hosted v2 provider row lacks request id"
            )
    if (
        observed_action_cells != expected_action_cells
        or observed_semantic_cells != expected_semantic_cells
    ):
        raise PilotCheckpointError(
            "v2 provider rows do not cover the registered 12+4 task cells"
        )

    outcomes = payload.get("proposal_outcomes")
    semantic_indices = {
        row["call_index"] for row in rows if row["call_kind"] == "semantic"
    }
    if (
        not isinstance(outcomes, list)
        or len(outcomes) != 4
        or payload.get("proposal_outcomes_hash") != canonical_hash(outcomes)
        or {row.get("call_index") for row in outcomes} != semantic_indices
    ):
        raise PilotCheckpointError(
            "v2 proposal outcomes are missing or unbound"
        )
    semantic_by_index = {
        row["call_index"]: row for row in rows if row["call_kind"] == "semantic"
    }
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            raise PilotCheckpointError("v2 proposal outcome is incomplete")
        source = semantic_by_index[outcome["call_index"]]
        events = outcome.get("semantic_events")
        if (
            outcome.get("current_t") != source["decision_t"]
            or outcome.get("agent_id") != source["agent_id"]
            or outcome.get("prompt_hash") != source["prompt_hash"]
            or outcome.get("raw_output_hash") != source["raw_output_hash"]
            or not isinstance(events, list)
            or outcome.get("semantic_events_hash") != canonical_hash(events)
        ):
            raise PilotCheckpointError(
                "v2 proposal outcome source binding is incomplete"
            )
        if outcome.get("candidate_parse_status") == "success":
            valid = (
                outcome.get("candidate_parse_mode") == "exact_json"
                and not isinstance(outcome.get("failure_reason"), str)
                and isinstance(outcome.get("rule_id"), str)
                and bool(outcome["rule_id"])
                and isinstance(outcome.get("rule_status"), str)
                and bool(events)
                and source["parse_disposition"].get("accepted") is True
            )
        elif outcome.get("candidate_parse_status") == "failure":
            valid = (
                outcome.get("candidate_parse_mode")
                in {
                    "exact_json",
                    "fenced_recovery",
                    "substring_recovery",
                    "parse_failure",
                }
                and isinstance(outcome.get("failure_reason"), str)
                and bool(outcome["failure_reason"])
                and outcome.get("rule_id") is None
                and outcome.get("rule_status") is None
                and events == []
                and source["parse_disposition"].get("parse_status")
                == "failure"
                and source["parse_disposition"].get("accepted") is False
            )
        else:
            valid = False
        if not valid:
            raise PilotCheckpointError("v2 proposal outcome is incomplete")

    denominator = payload.get("provider_denominator")
    semantic_parse_failures = sum(
        outcome.get("candidate_parse_status") == "failure"
        for outcome in outcomes
    )
    if denominator != {
        "planned_calls": 16,
        "observed_calls": 16,
        "successful_terminal_calls": 16,
        "failed_calls": 0,
        "action_calls": 12,
        "semantic_calls": 4,
        "semantic_candidate_parse_failures": semantic_parse_failures,
    }:
        raise PilotCheckpointError("v2 provider denominator is not exact")
    totals = {
        "call_count": 16,
        "action_call_count": 12,
        "semantic_call_count": 4,
        "prompt_tokens": sum(
            int(row["usage"]["prompt_tokens"]) for row in rows
        ),
        "completion_tokens": sum(
            int(row["usage"]["completion_tokens"]) for row in rows
        ),
        "reasoning_tokens": sum(
            int(row["reasoning_tokens"]) for row in rows
        ),
        "visible_completion_tokens": sum(
            int(row["visible_completion_tokens"]) for row in rows
        ),
        "visible_output_bytes": sum(
            int(row["visible_output_bytes"]) for row in rows
        ),
        "cost_usd": sum(
            float(row["usage"]["cost_usd"]) for row in rows
        ),
        "hosted": any(
            str(row["provider"]).strip().lower()
            not in {"diagnostic", "local", "ollama"}
            for row in rows
        ),
        "requested_models": sorted({str(row["model"]) for row in rows}),
        "served_models": sorted({str(row["served_model"]) for row in rows}),
        "served_providers": sorted(
            {str(row["served_provider"]) for row in rows}
        ),
        "served_routes": sorted({str(row["served_route"]) for row in rows}),
    }
    if (
        payload.get("provider_totals") != totals
        or payload.get("provider_totals_hash") != canonical_hash(totals)
    ):
        raise PilotCheckpointError("v2 provider totals/hash mismatch")
    if totals["hosted"] and totals["cost_usd"] <= 0:
        raise PilotCheckpointError("hosted v2 total cost is not positive")

    budget_snapshot = payload.get("budget_snapshot_at_checkpoint")
    if (
        not isinstance(budget_snapshot, Mapping)
        or payload.get("budget_snapshot_hash")
        != canonical_hash(budget_snapshot)
        or budget_snapshot.get("completed_calls") != 16
        or budget_snapshot.get("active_calls") != 0
        or not isinstance(budget_snapshot.get("accounted_usage"), Mapping)
    ):
        raise PilotCheckpointError("v2 checkpoint budget snapshot is invalid")
    accounted = budget_snapshot["accounted_usage"]
    if (
        int(accounted.get("prompt_tokens", -1)) != totals["prompt_tokens"]
        or int(accounted.get("completion_tokens", -1))
        != totals["completion_tokens"]
        or abs(float(accounted.get("cost_usd", -1.0)) - totals["cost_usd"])
        > 1e-12
    ):
        raise PilotCheckpointError(
            "v2 budget snapshot differs from provider totals"
        )
    journal_binding = payload.get("provider_call_journal_binding")
    if (
        not isinstance(journal_binding, Mapping)
        or payload.get("provider_call_journal_binding_hash")
        != canonical_hash(journal_binding)
        or journal_binding.get("run_id") != run_config.get("run_id")
        or journal_binding.get("contract_hash")
        != run_config.get("pilot_contract_hash")
    ):
        raise PilotCheckpointError(
            "v2 provider journal binding is invalid"
        )
    if journal_binding.get("enabled") is True:
        if (
            not isinstance(journal_binding.get("journal_sha256"), str)
            or len(journal_binding["journal_sha256"]) != 64
            or journal_binding.get("event_count") != 32
            or journal_binding.get("completion_event_count") != 16
            or journal_binding.get("parse_disposition_event_count") != 16
            or not isinstance(journal_binding.get("path_name"), str)
            or not journal_binding["path_name"]
        ):
            raise PilotCheckpointError(
                "v2 enabled provider journal binding is incomplete"
            )
    elif journal_binding.get("enabled") is False:
        if (
            run_config.get("scientific_scope")
            == "preregistered_mechanism_micro_pilot"
            or journal_binding.get("journal_sha256") is not None
            or journal_binding.get("event_count") != 0
            or journal_binding.get("completion_event_count") != 0
            or journal_binding.get("parse_disposition_event_count") != 0
            or journal_binding.get("path_name") is not None
        ):
            raise PilotCheckpointError(
                "v2 disabled provider journal binding is inconsistent"
            )
    else:
        raise PilotCheckpointError(
            "v2 provider journal enabled flag is invalid"
        )


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
        schema_version = _supported_checkpoint_schema(
            self.payload.get("schema_version")
        )
        body = dict(self.payload)
        claimed = body.pop("checkpoint_hash", None)
        if claimed != canonical_hash(body):
            raise PilotCheckpointError("checkpoint hash mismatch")
        run_config = self.payload.get("run_config")
        if not isinstance(run_config, Mapping):
            raise PilotCheckpointError("checkpoint run config must be an object")
        try:
            num_agents = int(run_config["num_agents"])
            episode_length = int(run_config["episode_length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PilotCheckpointError(
                "checkpoint run config lacks a valid cohort/length"
            ) from exc
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
        if not isinstance(memories, Mapping) or len(memories) != num_agents:
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
            str(agent_id) for agent_id in range(num_agents)
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
        previous_state = self.payload.get("previous_state")
        if (
            not isinstance(previous_state, Mapping)
            or not isinstance(previous_state.get("agents"), Mapping)
            or set(previous_state["agents"]) != expected_agents
        ):
            raise PilotCheckpointError(
                "previous environment state agent cohort mismatch"
            )
        last_decisions = self.payload.get("last_decisions")
        if not isinstance(last_decisions, Mapping) or set(
            last_decisions
        ) != expected_agents:
            raise PilotCheckpointError("last-decision cohort mismatch")
        for step in prefix_steps:
            if (
                not isinstance(step, Mapping)
                or not isinstance(step.get("decisions"), Mapping)
                or set(step["decisions"]) != expected_agents
                or not isinstance(step.get("foundation_actions"), Mapping)
                or set(step["foundation_actions"]) != expected_agents | {"p"}
                or not isinstance(step.get("pre_snapshots"), Mapping)
                or set(step["pre_snapshots"]) != expected_agents
                or not isinstance(step.get("pre_ledger_batch"), Mapping)
                or set(step["pre_ledger_batch"]) != expected_agents
                or not isinstance(step.get("post_ledger_batch"), Mapping)
                or set(step["post_ledger_batch"]) != expected_agents
            ):
                raise PilotCheckpointError(
                    "prefix step cohort/actions are not exact"
                )
        contract = {
            "schema_version": schema_version,
            "run_config": self.payload["run_config"],
            "foundation_env_config": self.payload["foundation_env_config"],
            "branch_after_decision_t": self.next_decision_t - 1,
        }
        if schema_version == PILOT_CHECKPOINT_SCHEMA_VERSION_V2:
            purpose = self.payload.get("checkpoint_purpose")
            if purpose != CLOSED_LOOP_PREFLIGHT_CHECKPOINT_PURPOSE:
                raise PilotCheckpointError(
                    "v2 checkpoint purpose must be closed-loop-preflight"
                )
            if num_agents != 2 or episode_length != 6:
                raise PilotCheckpointError(
                    "closed-loop preflight checkpoint must bind 2 agents x 6 months"
                )
            ledger_records = self.payload.get("ledger_records")
            if not isinstance(ledger_records, list) or len(ledger_records) != 12:
                raise PilotCheckpointError(
                    "closed-loop preflight ledger must contain 12 rows"
                )
            if self.payload.get("proposal_counters_hash") != canonical_hash(
                proposals_made
            ):
                raise PilotCheckpointError("proposal-counter hash mismatch")
            if self.payload.get("last_decisions_hash") != canonical_hash(
                last_decisions
            ):
                raise PilotCheckpointError("last-decision hash mismatch")
            prefix_actions = [
                step["foundation_actions"] for step in prefix_steps
            ]
            if self.payload.get("prefix_actions_hash") != canonical_hash(
                prefix_actions
            ):
                raise PilotCheckpointError("prefix-action hash mismatch")
            rng_binding = {
                "numpy_rng_before_env_construction": self.payload.get(
                    "numpy_rng_before_env_construction"
                ),
                "foundation_reset_seed_state": self.payload.get(
                    "foundation_reset_seed_state"
                ),
                "python_rng_at_start": self.payload.get("python_rng_at_start"),
                "step_seed_states": [
                    step.get("step_seed_state") for step in prefix_steps
                ],
                "python_step_seed_states": [
                    step.get("python_step_seed_state") for step in prefix_steps
                ],
                "numpy_rng_after_prefix": self.payload.get(
                    "numpy_rng_after_prefix"
                ),
                "python_rng_after_prefix": self.payload.get(
                    "python_rng_after_prefix"
                ),
            }
            if self.payload.get("rng_binding_hash") != canonical_hash(
                rng_binding
            ):
                raise PilotCheckpointError("checkpoint RNG binding hash mismatch")
            try:
                decode_numpy_rng_state(
                    self.payload["numpy_rng_before_env_construction"]
                )
                decode_numpy_rng_state(
                    self.payload["foundation_reset_seed_state"]
                )
                decode_python_rng_state(self.payload["python_rng_at_start"])
                decode_numpy_rng_state(self.payload["numpy_rng_after_prefix"])
                decode_python_rng_state(self.payload["python_rng_after_prefix"])
                for step in prefix_steps:
                    decode_numpy_rng_state(step["step_seed_state"])
                    decode_python_rng_state(step["python_step_seed_state"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PilotCheckpointError(
                    "checkpoint contains an invalid RNG state"
                ) from exc
            if (
                run_config.get("accepted_action_parse_modes")
                != ["exact_json"]
                or run_config.get("accepted_semantic_parse_modes")
                != ["exact_json"]
                or run_config.get("semantic_parse_failure_policy")
                != "record-and-skip"
                or run_config.get("fail_on_clipped_action") is not True
            ):
                raise PilotCheckpointError(
                    "v2 checkpoint does not bind exact actions and "
                    "record-and-skip semantic parsing"
                )
            _validate_v2_provider_evidence(
                self.payload,
                run_config=run_config,
            )
            contract["checkpoint_purpose"] = purpose
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


_V2_PARAMETER_DISPATCH_FIELDS = frozenset(
    {"reasoning", "response_format", "seed", "temperature", "top_p"}
)
_V2_PARAMETER_DISPATCH_STATUSES = frozenset(
    {"explicit_supported", "omitted_unsupported"}
)


def _validate_v2_preflight_config(config: VerifiedRunConfig) -> None:
    if config.accepted_action_parse_modes != ("exact_json",):
        raise ValueError(
            "v2 closed-loop preflight requires exact-only action parsing"
        )
    if config.accepted_semantic_parse_modes != ("exact_json",):
        raise ValueError(
            "v2 closed-loop preflight requires exact-only semantic parsing"
        )
    if config.semantic_parse_failure_policy != "record-and-skip":
        raise ValueError(
            "v2 closed-loop preflight semantic parse failures must be "
            "recorded and skipped"
        )
    if config.fail_on_clipped_action is not True:
        raise ValueError(
            "v2 closed-loop preflight must fail on clipped actions"
        )
    if (
        config.action_max_tokens < 1
        or config.rule_max_tokens < 1
        or config.action_max_visible_json_bytes < 1
        or config.rule_max_visible_json_bytes < 1
    ):
        raise ValueError("v2 closed-loop preflight task caps must be positive")


def _hosted_provider(provider: str) -> bool:
    return provider.strip().lower() not in {
        "diagnostic",
        "local",
        "ollama",
    }


def _v2_completion_audit_row(
    completion: StructuredCompletion,
    *,
    call_index: int,
    call_kind: str,
    decision_t: int,
    agent_id: int,
    prompt_hash: str,
    max_visible_tokens: int,
    max_visible_json_bytes: int,
) -> dict[str, Any]:
    """Validate and retain one terminal V2 completion without raw output."""

    row = _provider_row(
        completion,
        call_kind=call_kind,
        decision_t=decision_t,
        agent_id=agent_id,
        prompt_hash=prompt_hash,
    )
    row.update(
        {
            "call_index": int(call_index),
            "served_model": completion.response_model,
            "served_provider": completion.response_provider,
            "served_route": completion.response_route,
            "visible_output_bytes": row["raw_output_bytes"],
            "task_cap": {
                "max_visible_tokens": int(max_visible_tokens),
                "max_visible_json_bytes": int(max_visible_json_bytes),
            },
        }
    )
    if not completion.ok or completion.text == "Error":
        raise PilotCheckpointError(
            f"v2 provider failure for {call_kind} call {call_index}: "
            f"{completion.error_type}"
        )
    if completion.attempts != 1:
        raise PilotCheckpointError(
            "v2 closed-loop preflight requires exactly one provider attempt"
        )
    if (
        completion.finish_reason != "stop"
        or completion.response_completed is not True
        or completion.output_disposition != "accepted"
    ):
        raise PilotCheckpointError(
            f"v2 completion is truncated or non-terminal for {call_kind} "
            f"call {call_index}"
        )
    for name in (
        "response_model",
        "response_provider",
        "response_route",
        "request_profile_id",
        "request_price_snapshot_source",
        "request_price_snapshot_captured_at",
        "native_finish_reason",
        "provider_sdk_name",
        "provider_sdk_version",
        "temperature_dispatch",
    ):
        value = getattr(completion, name)
        if not isinstance(value, str) or not value.strip():
            raise PilotCheckpointError(
                f"v2 completion lacks {name} for {call_kind} call {call_index}"
            )
    if (
        not completion.request_provider_pin
        or not completion.request_artifact_identity
        or not completion.request_parameters
    ):
        raise PilotCheckpointError(
            "v2 completion lacks request pin/artifact/parameter metadata"
        )
    if (
        _hosted_provider(completion.provider)
        and not isinstance(completion.request_id, str)
    ):
        raise PilotCheckpointError(
            "hosted v2 completion lacks a provider request id"
        )
    dispatch = dict(completion.parameter_dispatch)
    if (
        set(dispatch) != _V2_PARAMETER_DISPATCH_FIELDS
        or not set(dispatch.values()) <= _V2_PARAMETER_DISPATCH_STATUSES
    ):
        raise PilotCheckpointError(
            "v2 completion parameter dispatch is incomplete or invalid"
        )
    request_parameters = set(completion.request_parameters)
    provider_name = completion.provider.strip().lower()
    direct_wire_names: dict[str, str]
    if provider_name == "openai":
        direct_wire_names = {
            "temperature": "temperature",
            "top_p": "top_p",
            "seed": "seed",
            "reasoning": "reasoning_effort",
            "response_format": "response_format",
        }
    elif provider_name in {"thirdparty", "openrouter"}:
        direct_wire_names = {
            "temperature": "temperature",
            "top_p": "top_p",
            "seed": "seed",
            "response_format": "response_format",
        }
    elif provider_name == "ollama":
        direct_wire_names = {"response_format": "format"}
    else:
        direct_wire_names = {}
    for field, wire_name in direct_wire_names.items():
        explicit = dispatch[field] == "explicit_supported"
        if explicit != (wire_name in request_parameters):
            raise PilotCheckpointError(
                "v2 completion parameter dispatch differs from wire parameters"
            )
    usage = completion.usage
    if usage.prompt_tokens < 1 or usage.completion_tokens < 1:
        raise PilotCheckpointError(
            "v2 completion usage must contain positive prompt/completion tokens"
        )
    if _hosted_provider(completion.provider) and usage.cost_usd <= 0:
        raise PilotCheckpointError(
            "hosted v2 completion must record a positive cost"
        )
    if row["visible_completion_tokens"] > max_visible_tokens:
        raise PilotCheckpointError(
            f"{call_kind} completion exceeds its visible-token cap"
        )
    if row["visible_output_bytes"] > max_visible_json_bytes:
        raise PilotCheckpointError(
            f"{call_kind} completion exceeds its visible-JSON byte cap"
        )
    return row


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
    _schema_version: str = PILOT_CHECKPOINT_SCHEMA_VERSION,
    _checkpoint_purpose: Optional[str] = None,
    _call_journal_path: Optional[str | Path] = None,
) -> PilotCheckpoint:
    """Run an exact prefix and freeze it immediately before ``decision_t=6``.

    The default is the preregistered Experiment-D branch point: after outcomes
    for decisions 0..5 have been finalized and before any decision-6 retrieval.
    """

    schema_version = _supported_checkpoint_schema(_schema_version)
    if schema_version == PILOT_CHECKPOINT_SCHEMA_VERSION:
        if _checkpoint_purpose is not None:
            raise ValueError("v1 checkpoints do not accept a checkpoint purpose")
        if _call_journal_path is not None:
            raise ValueError("v1 checkpoints do not accept a provider journal")
        expected_num_agents = 4
    else:
        if _checkpoint_purpose != CLOSED_LOOP_PREFLIGHT_CHECKPOINT_PURPOSE:
            raise ValueError(
                "v2 checkpoint purpose must be closed-loop-preflight"
            )
        expected_num_agents = 2
    if not isinstance(config, VerifiedRunConfig):
        raise TypeError("config must be VerifiedRunConfig")
    if schema_version == PILOT_CHECKPOINT_SCHEMA_VERSION_V2:
        _validate_v2_preflight_config(config)
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
            f"{schema_version} fixes the branch after t=5/before decision 6"
        )
    if config.num_agents != expected_num_agents:
        raise ValueError(
            f"{schema_version} requires exactly {expected_num_agents} agents"
        )
    if (
        schema_version == PILOT_CHECKPOINT_SCHEMA_VERSION_V2
        and config.episode_length != prefix_periods
    ):
        raise ValueError(
            "closed-loop preflight checkpoint requires exactly six months"
        )
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
    provider_call_rows: list[dict[str, Any]] = []
    proposal_outcomes: list[dict[str, Any]] = []
    call_journal_path = (
        None
        if _call_journal_path is None
        else Path(_call_journal_path).resolve()
    )
    if (
        schema_version == PILOT_CHECKPOINT_SCHEMA_VERSION_V2
        and config.scientific_scope
        == "preregistered_mechanism_micro_pilot"
        and call_journal_path is None
    ):
        raise ValueError(
            "scientific v2 preflight requires a provider call journal path"
        )
    if call_journal_path is not None and call_journal_path.exists():
        raise ValueError(
            "v2 provider call journal already exists; restore/resume the "
            "sealed checkpoint instead of redispatching"
        )

    def journal_completion(
        completion: StructuredCompletion,
        *,
        call_kind: str,
        decision_t: int,
        agent_id: int,
        prompt_hash: str,
    ) -> dict[str, Any]:
        row = _provider_row(
            completion,
            call_kind=call_kind,
            decision_t=decision_t,
            agent_id=agent_id,
            prompt_hash=prompt_hash,
        )
        _append_provider_call_journal(
            call_journal_path,
            run_id=config.run_id,
            contract_hash=config.pilot_contract_hash,
            event_type="completion_received",
            payload=row,
        )
        return row

    def journal_disposition(
        completion_row: Mapping[str, Any],
        *,
        parse_status: str,
        parse_mode: str,
        accepted: bool,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        payload = {
            "call_kind": completion_row["call_kind"],
            "decision_t": completion_row["decision_t"],
            "agent_id": completion_row["agent_id"],
            "prompt_hash": completion_row["prompt_hash"],
            "raw_output_hash": completion_row["raw_output_hash"],
            "parse_status": parse_status,
            "parse_mode": parse_mode,
            "accepted": accepted,
        }
        payload.update(dict(extra or {}))
        _append_provider_call_journal(
            call_journal_path,
            run_id=config.run_id,
            contract_hash=config.pilot_contract_hash,
            event_type="parse_disposition",
            payload=payload,
        )

    def journal_budget_failed_batch(
        error: BudgetExceeded,
        *,
        call_kind: str,
        decision_t: int,
        agent_ids: Sequence[int],
        prompts: Sequence[str],
    ) -> None:
        """Terminalize every settled response before propagating an overage."""

        if schema_version != PILOT_CHECKPOINT_SCHEMA_VERSION_V2:
            return
        completions = getattr(error, "structured_completions", None)
        if (
            not isinstance(completions, tuple)
            or len(completions) != len(agent_ids)
            or len(prompts) != len(agent_ids)
            or any(
                not isinstance(completion, StructuredCompletion)
                for completion in completions
            )
        ):
            raise PilotCheckpointError(
                "budget failure did not expose every dispatched completion "
                "for terminal journaling"
            ) from error
        reasons = [
            getattr(reason, "value", str(reason))
            for reason in getattr(error, "reasons", ())
        ]
        for completion, agent_id, prompt in zip(
            completions,
            agent_ids,
            prompts,
            strict=True,
        ):
            completion_row = journal_completion(
                completion,
                call_kind=call_kind,
                decision_t=decision_t,
                agent_id=agent_id,
                prompt_hash=hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
            )
            journal_disposition(
                completion_row,
                parse_status="not_evaluated",
                parse_mode="budget_failure",
                accepted=False,
                extra={
                    "rejection": "run_budget_exceeded",
                    "budget_stop_reasons": reasons,
                },
            )

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

        try:
            completions = _complete_actions(
                config=config,
                llm=llm,
                budget=budget,
                decision_t=decision_t,
                prompts=prompts,
            )
        except BudgetExceeded as exc:
            journal_budget_failed_batch(
                exc,
                call_kind="action",
                decision_t=decision_t,
                agent_ids=tuple(range(config.num_agents)),
                prompts=prompts,
            )
            raise
        if len(completions) != config.num_agents:
            raise PilotCheckpointError(
                f"action denominator mismatch at t={decision_t}"
            )
        action_journal_rows: list[dict[str, Any]] = []
        if schema_version == PILOT_CHECKPOINT_SCHEMA_VERSION_V2:
            for agent_id, completion in enumerate(completions):
                prompt_hash = hashlib.sha256(
                    prompts[agent_id].encode("utf-8")
                ).hexdigest()
                action_journal_rows.append(
                    journal_completion(
                        completion,
                        call_kind="action",
                        decision_t=decision_t,
                        agent_id=agent_id,
                        prompt_hash=prompt_hash,
                    )
                )

        def close_later_action_dispositions(current_agent_id: int) -> None:
            for pending_row in action_journal_rows[current_agent_id + 1 :]:
                journal_disposition(
                    pending_row,
                    parse_status="not_evaluated",
                    parse_mode="prior_batch_failure",
                    accepted=False,
                )

        decisions: dict[str, ActionDecision] = {}
        for agent_id, completion in enumerate(completions):
            completion_row: Optional[dict[str, Any]] = None
            journal_row: Optional[dict[str, Any]] = None
            if schema_version == PILOT_CHECKPOINT_SCHEMA_VERSION_V2:
                prompt_hash = hashlib.sha256(
                    prompts[agent_id].encode("utf-8")
                ).hexdigest()
                journal_row = action_journal_rows[agent_id]
                try:
                    completion_row = _v2_completion_audit_row(
                        completion,
                        call_index=len(provider_call_rows),
                        call_kind="action",
                        decision_t=decision_t,
                        agent_id=agent_id,
                        prompt_hash=prompt_hash,
                        max_visible_tokens=config.action_max_tokens,
                        max_visible_json_bytes=(
                            config.action_max_visible_json_bytes
                        ),
                    )
                except PilotCheckpointError:
                    journal_disposition(
                        journal_row,
                        parse_status=(
                            "failure" if completion.ok else "unavailable"
                        ),
                        parse_mode=completion.output_disposition,
                        accepted=False,
                    )
                    close_later_action_dispositions(agent_id)
                    raise
            if not completion.ok or completion.text == "Error":
                raise PilotCheckpointError(
                    f"prefix provider failure at t={decision_t}, agent={agent_id}"
                )
            try:
                decision = parse_direct_action(
                    completion.text,
                    max_labor_hours=config.max_labor_hours,
                    labor_step=config.labor_step,
                    consumption_step=config.consumption_step,
                )
            except ActionParseError as exc:
                if journal_row is not None:
                    journal_disposition(
                        journal_row,
                        parse_status="failure",
                        parse_mode="parse_failure",
                        accepted=False,
                    )
                    close_later_action_dispositions(agent_id)
                raise PilotCheckpointError(
                    f"prefix action parse failure at t={decision_t}, "
                    f"agent={agent_id}: {exc}"
                ) from exc
            action_parse_mode = _action_parse_mode(
                completion.text, decision
            )
            if (
                schema_version == PILOT_CHECKPOINT_SCHEMA_VERSION_V2
                and action_parse_mode != "exact_json"
            ):
                assert journal_row is not None
                journal_disposition(
                    journal_row,
                    parse_status="success",
                    parse_mode=action_parse_mode,
                    accepted=False,
                )
                close_later_action_dispositions(agent_id)
                raise PilotCheckpointError(
                    f"v2 action is not exact JSON at t={decision_t}, "
                    f"agent={agent_id}"
                )
            if config.fail_on_clipped_action and decision.clipped:
                if journal_row is not None:
                    journal_disposition(
                        journal_row,
                        parse_status="success",
                        parse_mode=action_parse_mode,
                        accepted=False,
                        extra={"rejection": "clipped_action"},
                    )
                    close_later_action_dispositions(agent_id)
                raise PilotCheckpointError(
                    f"clipped prefix action at t={decision_t}, agent={agent_id}"
                )
            if schema_version == PILOT_CHECKPOINT_SCHEMA_VERSION_V2:
                if decision.clipped:
                    raise PilotCheckpointError(
                        f"v2 clipped action at t={decision_t}, "
                        f"agent={agent_id}"
                    )
                assert completion_row is not None
                assert journal_row is not None
                journal_disposition(
                    journal_row,
                    parse_status="success",
                    parse_mode=action_parse_mode,
                    accepted=True,
                )
                completion_row["parse_disposition"] = {
                    "parse_status": "success",
                    "parse_mode": action_parse_mode,
                    "accepted": True,
                    "clipped": False,
                }
                provider_call_rows.append(completion_row)
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
            try:
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
            except BudgetExceeded as exc:
                journal_budget_failed_batch(
                    exc,
                    call_kind="semantic",
                    decision_t=current_t,
                    agent_ids=eligible,
                    prompts=proposal_prompts,
                )
                raise
            if len(proposal_results) != len(eligible):
                raise PilotCheckpointError(
                    f"semantic denominator mismatch at t={current_t}"
                )
            semantic_journal_rows: list[dict[str, Any]] = []
            if schema_version == PILOT_CHECKPOINT_SCHEMA_VERSION_V2:
                for agent_id, prompt, completion in zip(
                    eligible,
                    proposal_prompts,
                    proposal_results,
                    strict=True,
                ):
                    semantic_journal_rows.append(
                        journal_completion(
                            completion,
                            call_kind="semantic",
                            decision_t=current_t,
                            agent_id=agent_id,
                            prompt_hash=hashlib.sha256(
                                prompt.encode("utf-8")
                            ).hexdigest(),
                        )
                    )

            def close_later_semantic_dispositions(
                current_index: int,
            ) -> None:
                for pending_row in semantic_journal_rows[
                    current_index + 1 :
                ]:
                    journal_disposition(
                        pending_row,
                        parse_status="not_evaluated",
                        parse_mode="prior_batch_failure",
                        accepted=False,
                    )

            for batch_index, (agent_id, completion) in enumerate(
                zip(eligible, proposal_results, strict=True)
            ):
                proposals_made[agent_id] += 1
                prompt = proposal_prompts[batch_index]
                prompt_hash = hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest()
                completion_row: Optional[dict[str, Any]] = None
                journal_row: Optional[dict[str, Any]] = None
                candidate_parse_mode: Optional[str] = None
                if schema_version == PILOT_CHECKPOINT_SCHEMA_VERSION_V2:
                    journal_row = semantic_journal_rows[batch_index]
                    try:
                        completion_row = _v2_completion_audit_row(
                            completion,
                            call_index=len(provider_call_rows),
                            call_kind="semantic",
                            decision_t=current_t,
                            agent_id=agent_id,
                            prompt_hash=prompt_hash,
                            max_visible_tokens=config.rule_max_tokens,
                            max_visible_json_bytes=(
                                config.rule_max_visible_json_bytes
                            ),
                        )
                    except PilotCheckpointError:
                        journal_disposition(
                            journal_row,
                            parse_status=(
                                "failure" if completion.ok else "unavailable"
                            ),
                            parse_mode=completion.output_disposition,
                            accepted=False,
                        )
                        close_later_semantic_dispositions(batch_index)
                        raise
                    candidate_parse_mode = _semantic_parse_mode(
                        completion.text
                    )
                    if candidate_parse_mode != "exact_json":
                        completion_row["parse_disposition"] = {
                            "parse_status": "failure",
                            "parse_mode": candidate_parse_mode,
                            "accepted": False,
                            "rejection": "non_exact_json",
                        }
                        journal_disposition(
                            journal_row,
                            parse_status="failure",
                            parse_mode=candidate_parse_mode,
                            accepted=False,
                            extra={"rejection": "non_exact_json"},
                        )
                        provider_call_rows.append(completion_row)
                        empty_events: list[dict[str, Any]] = []
                        proposal_outcomes.append(
                            {
                                "call_index": completion_row["call_index"],
                                "current_t": current_t,
                                "agent_id": agent_id,
                                "prompt_hash": prompt_hash,
                                "raw_output_hash": completion_row[
                                    "raw_output_hash"
                                ],
                                "candidate_parse_status": "failure",
                                "candidate_parse_mode": candidate_parse_mode,
                                "failure_reason": "non_exact_json",
                                "rule_id": None,
                                "rule_status": None,
                                "semantic_events": empty_events,
                                "semantic_events_hash": canonical_hash(
                                    empty_events
                                ),
                            }
                        )
                        continue
                if not completion.ok or completion.text == "Error":
                    raise PilotCheckpointError(
                        f"prefix semantic provider failure at t={current_t}, "
                        f"agent={agent_id}"
                    )
                before_events = (
                    len(memories[agent_id].semantic.events)
                    if memories[agent_id].semantic is not None
                    else 0
                )
                try:
                    rule = memories[agent_id].submit_rule_proposal(
                        completion.text,
                        current_t=current_t,
                        generator_id=llm.get_model_name(),
                        semantic_policy=config.semantic_policy,
                    )
                except CandidateParseError as exc:
                    if schema_version == PILOT_CHECKPOINT_SCHEMA_VERSION_V2:
                        assert completion_row is not None
                        assert journal_row is not None
                        assert candidate_parse_mode is not None
                        completion_row["parse_disposition"] = {
                            "parse_status": "failure",
                            "parse_mode": candidate_parse_mode,
                            "accepted": False,
                            "rejection": "candidate_parse_failure",
                        }
                        journal_disposition(
                            journal_row,
                            parse_status="failure",
                            parse_mode=candidate_parse_mode,
                            accepted=False,
                            extra={"rejection": "candidate_parse_failure"},
                        )
                        provider_call_rows.append(completion_row)
                        empty_events = []
                        proposal_outcomes.append(
                            {
                                "call_index": completion_row["call_index"],
                                "current_t": current_t,
                                "agent_id": agent_id,
                                "prompt_hash": prompt_hash,
                                "raw_output_hash": completion_row[
                                    "raw_output_hash"
                                ],
                                "candidate_parse_status": "failure",
                                "candidate_parse_mode": candidate_parse_mode,
                                "failure_reason": "candidate_parse_failure",
                                "rule_id": None,
                                "rule_status": None,
                                "semantic_events": empty_events,
                                "semantic_events_hash": canonical_hash(
                                    empty_events
                                ),
                            }
                        )
                        continue
                    if config.semantic_parse_failure_policy == "fail-run":
                        raise
                    continue
                except Exception as exc:
                    if schema_version == PILOT_CHECKPOINT_SCHEMA_VERSION_V2:
                        assert journal_row is not None
                        journal_disposition(
                            journal_row,
                            parse_status="failure",
                            parse_mode=(
                                candidate_parse_mode or "parse_failure"
                            ),
                            accepted=False,
                            extra={"rejection": "semantic_lifecycle_failure"},
                        )
                        close_later_semantic_dispositions(batch_index)
                        raise PilotCheckpointError(
                            f"v2 semantic lifecycle failure at t={current_t}, "
                            f"agent={agent_id}: {exc}"
                        ) from exc
                    raise
                if schema_version == PILOT_CHECKPOINT_SCHEMA_VERSION_V2:
                    assert completion_row is not None
                    assert candidate_parse_mode is not None
                    assert journal_row is not None
                    semantic = memories[agent_id].semantic
                    if semantic is None:
                        journal_disposition(
                            journal_row,
                            parse_status="failure",
                            parse_mode=candidate_parse_mode,
                            accepted=False,
                            extra={"rejection": "missing_semantic_state"},
                        )
                        close_later_semantic_dispositions(batch_index)
                        raise PilotCheckpointError(
                            "v2 semantic proposal completed without M3 state"
                        )
                    event_rows = [
                        event.to_dict()
                        for event in semantic.events[before_events:]
                    ]
                    if not event_rows:
                        journal_disposition(
                            journal_row,
                            parse_status="failure",
                            parse_mode=candidate_parse_mode,
                            accepted=False,
                            extra={"rejection": "missing_lifecycle_outcome"},
                        )
                        close_later_semantic_dispositions(batch_index)
                        raise PilotCheckpointError(
                            "v2 semantic proposal has no lifecycle outcome"
                        )
                    proposal_outcome = {
                        "call_index": completion_row["call_index"],
                        "current_t": current_t,
                        "agent_id": agent_id,
                        "prompt_hash": prompt_hash,
                        "raw_output_hash": completion_row["raw_output_hash"],
                        "candidate_parse_status": "success",
                        "candidate_parse_mode": candidate_parse_mode,
                        "rule_id": rule.rule_id,
                        "rule_status": rule.status,
                        "semantic_events": event_rows,
                        "semantic_events_hash": canonical_hash(event_rows),
                    }
                    completion_row["parse_disposition"] = {
                        "parse_status": "success",
                        "parse_mode": candidate_parse_mode,
                        "accepted": True,
                        "rule_id": rule.rule_id,
                        "rule_status": rule.status,
                    }
                    journal_disposition(
                        journal_row,
                        parse_status="success",
                        parse_mode=candidate_parse_mode,
                        accepted=True,
                        extra={
                            "rule_id": rule.rule_id,
                            "rule_status": rule.status,
                        },
                    )
                    provider_call_rows.append(completion_row)
                    proposal_outcomes.append(proposal_outcome)

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
    v2_provider_totals: Optional[dict[str, Any]] = None
    v2_budget_snapshot: Optional[dict[str, Any]] = None
    v2_journal_binding: Optional[dict[str, Any]] = None
    if schema_version == PILOT_CHECKPOINT_SCHEMA_VERSION_V2:
        action_rows = [
            row for row in provider_call_rows if row["call_kind"] == "action"
        ]
        semantic_rows = [
            row
            for row in provider_call_rows
            if row["call_kind"] == "semantic"
        ]
        if (
            len(provider_call_rows) != 16
            or len(action_rows) != 12
            or len(semantic_rows) != 4
            or [row["call_index"] for row in provider_call_rows]
            != list(range(16))
        ):
            raise PilotCheckpointError(
                "v2 closed-loop preflight must retain the exact 16-call denominator"
            )
        if (
            len(proposal_outcomes) != 4
            or {
                row["call_index"] for row in proposal_outcomes
            }
            != {row["call_index"] for row in semantic_rows}
        ):
            raise PilotCheckpointError(
                "v2 semantic proposal outcomes are not complete"
            )
        for outcome in proposal_outcomes:
            if outcome["semantic_events_hash"] != canonical_hash(
                outcome["semantic_events"]
            ):
                raise PilotCheckpointError(
                    "v2 semantic proposal outcome hash mismatch"
                )
        if any(
            not isinstance(row.get("parse_disposition"), Mapping)
            or row["parse_disposition"].get("accepted") is not True
            or row["parse_disposition"].get("parse_status") != "success"
            or row["parse_disposition"].get("parse_mode") != "exact_json"
            for row in action_rows
        ):
            raise PilotCheckpointError(
                "v2 action denominator contains an unaccepted parse outcome"
            )
        semantic_parse_failures = sum(
            row["parse_disposition"].get("parse_status") == "failure"
            and row["parse_disposition"].get("accepted") is False
            for row in semantic_rows
        )
        if any(
            not isinstance(row.get("parse_disposition"), Mapping)
            or (
                not (
                    row["parse_disposition"].get("parse_status") == "success"
                    and row["parse_disposition"].get("parse_mode")
                    == "exact_json"
                    and row["parse_disposition"].get("accepted") is True
                )
                and not (
                    row["parse_disposition"].get("parse_status") == "failure"
                    and row["parse_disposition"].get("accepted") is False
                    and row["parse_disposition"].get("parse_mode")
                    in {
                        "exact_json",
                        "fenced_recovery",
                        "substring_recovery",
                        "parse_failure",
                    }
                )
            )
            for row in semantic_rows
        ):
            raise PilotCheckpointError(
                "v2 semantic denominator contains a nonterminal parse outcome"
            )
        hosted = any(
            _hosted_provider(str(row["provider"]))
            for row in provider_call_rows
        )
        v2_provider_totals = {
            "call_count": len(provider_call_rows),
            "action_call_count": len(action_rows),
            "semantic_call_count": len(semantic_rows),
            "prompt_tokens": sum(
                int(row["usage"]["prompt_tokens"])
                for row in provider_call_rows
            ),
            "completion_tokens": sum(
                int(row["usage"]["completion_tokens"])
                for row in provider_call_rows
            ),
            "reasoning_tokens": sum(
                int(row["reasoning_tokens"])
                for row in provider_call_rows
            ),
            "visible_completion_tokens": sum(
                int(row["visible_completion_tokens"])
                for row in provider_call_rows
            ),
            "visible_output_bytes": sum(
                int(row["visible_output_bytes"])
                for row in provider_call_rows
            ),
            "cost_usd": sum(
                float(row["usage"]["cost_usd"])
                for row in provider_call_rows
            ),
            "hosted": hosted,
            "requested_models": sorted(
                {str(row["model"]) for row in provider_call_rows}
            ),
            "served_models": sorted(
                {str(row["served_model"]) for row in provider_call_rows}
            ),
            "served_providers": sorted(
                {str(row["served_provider"]) for row in provider_call_rows}
            ),
            "served_routes": sorted(
                {str(row["served_route"]) for row in provider_call_rows}
            ),
        }
        if hosted and v2_provider_totals["cost_usd"] <= 0:
            raise PilotCheckpointError(
                "hosted v2 preflight total cost must be positive"
            )
        if any(
            len(v2_provider_totals[name]) != 1
            for name in (
                "requested_models",
                "served_models",
                "served_providers",
                "served_routes",
            )
        ):
            raise PilotCheckpointError(
                "v2 preflight model/provider/route changed within the run"
            )
        v2_budget_snapshot = budget.snapshot().to_dict()
        accounted = v2_budget_snapshot["accounted_usage"]
        if (
            v2_budget_snapshot["completed_calls"] != 16
            or v2_budget_snapshot["active_calls"] != 0
            or int(accounted["prompt_tokens"])
            != v2_provider_totals["prompt_tokens"]
            or int(accounted["completion_tokens"])
            != v2_provider_totals["completion_tokens"]
            or abs(
                float(accounted["cost_usd"])
                - float(v2_provider_totals["cost_usd"])
            )
            > 1e-12
        ):
            raise PilotCheckpointError(
                "v2 provider rows differ from the run budget ledger"
            )
        if call_journal_path is None:
            v2_journal_binding = {
                "enabled": False,
                "journal_sha256": None,
                "event_count": 0,
                "completion_event_count": 0,
                "parse_disposition_event_count": 0,
                "run_id": config.run_id,
                "contract_hash": config.pilot_contract_hash,
                "path_name": None,
            }
        else:
            journal = verify_provider_call_journal(
                call_journal_path,
                expected_run_id=config.run_id,
                expected_contract_hash=config.pilot_contract_hash,
                require_terminal_dispositions=True,
            )
            events = journal["events"]
            completion_events = [
                event
                for event in events
                if event["event_type"] == "completion_received"
            ]
            disposition_events = [
                event
                for event in events
                if event["event_type"] == "parse_disposition"
            ]
            if (
                len(events) != 32
                or len(completion_events) != 16
                or len(disposition_events) != 16
            ):
                raise PilotCheckpointError(
                    "v2 provider journal must contain 16 terminal call pairs"
                )
            for row, event in zip(
                provider_call_rows, completion_events, strict=True
            ):
                if any(
                    row.get(key) != value
                    for key, value in event["payload"].items()
                ):
                    raise PilotCheckpointError(
                        "v2 provider journal completion differs from checkpoint row"
                    )
            for row, event in zip(
                provider_call_rows,
                disposition_events,
                strict=True,
            ):
                disposition = row["parse_disposition"]
                if any(
                    event["payload"].get(key) != disposition.get(key)
                    for key in ("parse_status", "parse_mode", "accepted")
                ):
                    raise PilotCheckpointError(
                        "v2 provider journal disposition differs from "
                        "checkpoint evidence"
                    )
            v2_journal_binding = {
                "enabled": True,
                "journal_sha256": journal["journal_sha256"],
                "event_count": len(events),
                "completion_event_count": len(completion_events),
                "parse_disposition_event_count": len(disposition_events),
                "run_id": journal["run_id"],
                "contract_hash": journal["contract_hash"],
                "path_name": call_journal_path.name,
            }
    contract = {
        "schema_version": schema_version,
        "run_config": run_config,
        "foundation_env_config": foundation_config,
        "branch_after_decision_t": prefix_periods - 1,
    }
    if schema_version == PILOT_CHECKPOINT_SCHEMA_VERSION_V2:
        contract["checkpoint_purpose"] = _checkpoint_purpose
    payload: dict[str, Any] = {
        "schema_version": schema_version,
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
    if schema_version == PILOT_CHECKPOINT_SCHEMA_VERSION_V2:
        rng_binding = {
            "numpy_rng_before_env_construction": (
                numpy_rng_before_env_construction
            ),
            "foundation_reset_seed_state": reset_seed_state,
            "python_rng_at_start": python_rng_at_start,
            "step_seed_states": [
                step["step_seed_state"] for step in prefix_steps
            ],
            "python_step_seed_states": [
                step["python_step_seed_state"] for step in prefix_steps
            ],
            "numpy_rng_after_prefix": payload["numpy_rng_after_prefix"],
            "python_rng_after_prefix": payload["python_rng_after_prefix"],
        }
        payload.update(
            {
                "checkpoint_purpose": _checkpoint_purpose,
                "proposal_counters_hash": canonical_hash(
                    payload["proposals_made"]
                ),
                "last_decisions_hash": canonical_hash(
                    payload["last_decisions"]
                ),
                "prefix_actions_hash": canonical_hash(
                    [
                        step["foundation_actions"]
                        for step in prefix_steps
                    ]
                ),
                "rng_binding_hash": canonical_hash(rng_binding),
                "provider_calls": provider_call_rows,
                "provider_calls_hash": canonical_hash(
                    provider_call_rows
                ),
                "proposal_outcomes": proposal_outcomes,
                "proposal_outcomes_hash": canonical_hash(
                    proposal_outcomes
                ),
                "provider_denominator": {
                    "planned_calls": 16,
                    "observed_calls": len(provider_call_rows),
                    "successful_terminal_calls": len(provider_call_rows),
                    "failed_calls": 0,
                    "action_calls": 12,
                    "semantic_calls": 4,
                    "semantic_candidate_parse_failures": (
                        semantic_parse_failures
                    ),
                },
                "provider_totals": v2_provider_totals,
                "provider_totals_hash": canonical_hash(
                    v2_provider_totals
                ),
                "budget_snapshot_at_checkpoint": v2_budget_snapshot,
                "budget_snapshot_hash": canonical_hash(
                    v2_budget_snapshot
                ),
                "provider_call_journal_binding": v2_journal_binding,
                "provider_call_journal_binding_hash": canonical_hash(
                    v2_journal_binding
                ),
            }
        )
    payload["checkpoint_hash"] = canonical_hash(payload)
    return PilotCheckpoint(payload)


def build_closed_loop_preflight_checkpoint(
    config: VerifiedRunConfig,
    *,
    llm: MultiModelLLM,
    budget: RunBudget,
    env_config_source: Mapping[str, Any] | str | Path,
    call_journal_path: Optional[str | Path] = None,
) -> PilotCheckpoint:
    """Run and seal the registered 2-agent x 6-month preflight once.

    This is the checkpoint-producing preflight execution itself, not a second
    pass over a completed preflight.  Restoring and verifying its result never
    receives an LLM/provider object and therefore cannot issue another provider
    request.
    """

    return build_pilot_checkpoint(
        config,
        llm=llm,
        budget=budget,
        env_config_source=env_config_source,
        prefix_periods=DEFAULT_BRANCH_DECISION_T,
        _schema_version=PILOT_CHECKPOINT_SCHEMA_VERSION_V2,
        _checkpoint_purpose=CLOSED_LOOP_PREFLIGHT_CHECKPOINT_PURPOSE,
        _call_journal_path=call_journal_path,
    )


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
    restored_last_decisions = {
        key: decision.to_dict() for key, decision in last_decisions.items()
    }
    if restored_last_decisions != payload["last_decisions"]:
        raise PilotCheckpointError("restored last-decision snapshot is not exact")
    replayed_numpy_after_prefix = encode_numpy_rng_state(np.random.get_state())
    replayed_python_after_prefix = encode_python_rng_state(random.getstate())
    if replayed_numpy_after_prefix != payload["numpy_rng_after_prefix"]:
        raise PilotCheckpointError(
            "restored Foundation/NumPy continuation RNG is not exact"
        )
    if replayed_python_after_prefix != payload["python_rng_after_prefix"]:
        raise PilotCheckpointError(
            "restored Python continuation RNG is not exact"
        )
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


def _continuation_rng_preview(
    *,
    numpy_state: Mapping[str, Any],
    python_state: Mapping[str, Any],
    draws: int,
) -> dict[str, Any]:
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    np.random.set_state(decode_numpy_rng_state(numpy_state))
    random.setstate(decode_python_rng_state(python_state))
    numpy_values = [float(value) for value in np.random.random_sample(draws)]
    python_values = [float(random.random()) for _ in range(draws)]
    preview = {
        "draws": draws,
        "numpy_values": numpy_values,
        "python_values": python_values,
        "numpy_state_after_preview": encode_numpy_rng_state(
            np.random.get_state()
        ),
        "python_state_after_preview": encode_python_rng_state(
            random.getstate()
        ),
    }
    preview["preview_hash"] = canonical_hash(preview)
    return preview


def verify_closed_loop_preflight_checkpoint(
    checkpoint: PilotCheckpoint | Mapping[str, Any],
    *,
    rng_preview_draws: int = 16,
    strict_code_binding: bool = True,
) -> dict[str, Any]:
    """Restore twice and emit an exactness receipt without provider access.

    The two restores independently replay Foundation from the bound reset/step
    RNG states and action prefix.  The receipt binds every state family needed
    by the closed-loop preflight: Foundation and agent state, M1/M2/M3 memory,
    proposal counters, utility ledger, previous state, action prefix, and the
    pre-generated continuation RNG stream.
    """

    if not isinstance(checkpoint, PilotCheckpoint):
        checkpoint = PilotCheckpoint.from_dict(checkpoint)
    if (
        checkpoint.payload["schema_version"]
        != PILOT_CHECKPOINT_SCHEMA_VERSION_V2
        or checkpoint.payload.get("checkpoint_purpose")
        != CLOSED_LOOP_PREFLIGHT_CHECKPOINT_PURPOSE
    ):
        raise PilotCheckpointError(
            "closed-loop exactness verification requires a v2 preflight checkpoint"
        )
    if (
        isinstance(rng_preview_draws, bool)
        or not isinstance(rng_preview_draws, int)
        or rng_preview_draws < 1
    ):
        raise ValueError("rng_preview_draws must be a positive integer")

    original_numpy_state = np.random.get_state()
    original_python_state = random.getstate()
    restored_rows: list[dict[str, Any]] = []
    try:
        for _ in range(2):
            restored = restore_pilot_checkpoint(
                checkpoint,
                strict_code_binding=strict_code_binding,
            )
            current_numpy_state = encode_numpy_rng_state(
                np.random.get_state()
            )
            current_python_state = encode_python_rng_state(
                random.getstate()
            )
            if current_numpy_state != checkpoint.payload[
                "numpy_rng_after_prefix"
            ]:
                raise PilotCheckpointError(
                    "restored NumPy continuation RNG differs from checkpoint"
                )
            if current_python_state != checkpoint.payload[
                "python_rng_after_prefix"
            ]:
                raise PilotCheckpointError(
                    "restored Python continuation RNG differs from checkpoint"
                )
            memories = {
                str(agent_id): memory.to_dict()
                for agent_id, memory in restored.memories.items()
            }
            proposals = {
                str(agent_id): count
                for agent_id, count in restored.proposals_made.items()
            }
            last_decisions = {
                agent_id: decision.to_dict()
                for agent_id, decision in restored.last_decisions.items()
            }
            state_row = {
                "environment_hash": canonical_hash(
                    capture_environment_state(restored.env)
                ),
                "foundation_agents_hash": canonical_hash(
                    capture_environment_state(restored.env)["agents"]
                ),
                "memories_hash": canonical_hash(memories),
                "proposal_counters_hash": canonical_hash(proposals),
                "ledger_hash": canonical_hash(restored.ledger.records()),
                "last_decisions_hash": canonical_hash(last_decisions),
                "prefix_hash": restored.prefix_hash,
                "prefix_actions_hash": canonical_hash(
                    [
                        step["foundation_actions"]
                        for step in checkpoint.payload["prefix_steps"]
                    ]
                ),
                "numpy_rng_state": current_numpy_state,
                "python_rng_state": current_python_state,
                "rng_preview": _continuation_rng_preview(
                    numpy_state=current_numpy_state,
                    python_state=current_python_state,
                    draws=rng_preview_draws,
                ),
            }
            restored_rows.append(state_row)
    finally:
        np.random.set_state(original_numpy_state)
        random.setstate(original_python_state)

    if restored_rows[0] != restored_rows[1]:
        raise PilotCheckpointError(
            "independent checkpoint restores are not bit-exact"
        )
    row = restored_rows[0]
    expected = {
        "environment_hash": checkpoint.payload["previous_state_hash"],
        "foundation_agents_hash": canonical_hash(
            checkpoint.payload["previous_state"]["agents"]
        ),
        "memories_hash": checkpoint.payload["memories_hash"],
        "proposal_counters_hash": checkpoint.payload[
            "proposal_counters_hash"
        ],
        "ledger_hash": checkpoint.payload["ledger_hash"],
        "last_decisions_hash": checkpoint.payload["last_decisions_hash"],
        "prefix_hash": checkpoint.payload["prefix_hash"],
        "prefix_actions_hash": checkpoint.payload["prefix_actions_hash"],
        "numpy_rng_state": checkpoint.payload["numpy_rng_after_prefix"],
        "python_rng_state": checkpoint.payload["python_rng_after_prefix"],
    }
    observed = {
        key: row[key] for key in expected
    }
    if observed != expected:
        raise PilotCheckpointError(
            "restored checkpoint components differ from sealed bindings"
        )
    receipt = {
        "schema_version": "finevo-checkpoint-exactness-receipt-v1",
        "checkpoint_schema_version": checkpoint.payload["schema_version"],
        "checkpoint_purpose": checkpoint.payload["checkpoint_purpose"],
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "num_agents": 2,
        "completed_months": 6,
        "provider_calls_during_verification": 0,
        "verified_components": {
            "foundation_environment_and_agents": True,
            "foundation_reset_and_step_rng": True,
            "dual_track_agent_memories": True,
            "proposal_counters": True,
            "utility_ledger": True,
            "previous_state": True,
            "prefix_actions_and_hash": True,
            "last_decisions": True,
            "pre_generated_continuation_rng_equality": True,
            "provider_16_call_denominator": True,
            "provider_usage_cost_route_finish_dispatch": True,
            "exact_actions_record_skip_semantics_and_task_caps": True,
            "semantic_proposal_outcomes": True,
            "provider_budget_reconciliation": True,
            "provider_call_journal_binding": True,
        },
        "component_hashes": {
            key: value
            for key, value in expected.items()
            if key not in {"numpy_rng_state", "python_rng_state"}
        },
        "rng_binding_hash": checkpoint.payload["rng_binding_hash"],
        "provider_calls_hash": checkpoint.payload["provider_calls_hash"],
        "proposal_outcomes_hash": checkpoint.payload[
            "proposal_outcomes_hash"
        ],
        "provider_totals_hash": checkpoint.payload[
            "provider_totals_hash"
        ],
        "budget_snapshot_hash": checkpoint.payload[
            "budget_snapshot_hash"
        ],
        "provider_call_journal_binding_hash": checkpoint.payload[
            "provider_call_journal_binding_hash"
        ],
        "provider_denominator": _json_copy(
            checkpoint.payload["provider_denominator"]
        ),
        "continuation_rng_preview_hash": row["rng_preview"][
            "preview_hash"
        ],
        "rng_preview_draws": rng_preview_draws,
    }
    receipt["receipt_hash"] = canonical_hash(receipt)
    return receipt


__all__ = [
    "CLOSED_LOOP_PREFLIGHT_CHECKPOINT_PURPOSE",
    "DEFAULT_BRANCH_DECISION_T",
    "PILOT_CHECKPOINT_SCHEMA_VERSION",
    "PILOT_CHECKPOINT_SCHEMA_VERSION_V2",
    "PilotCheckpoint",
    "PilotCheckpointError",
    "RestoredPilotState",
    "build_closed_loop_preflight_checkpoint",
    "build_pilot_checkpoint",
    "canonical_hash",
    "capture_environment_state",
    "config_from_dict",
    "current_code_binding",
    "decode_numpy_rng_state",
    "encode_numpy_rng_state",
    "restore_pilot_checkpoint",
    "verify_closed_loop_preflight_checkpoint",
]
