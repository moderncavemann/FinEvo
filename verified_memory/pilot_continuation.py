"""Matched continuation branches from ``finevo-pilot-checkpoint-v1``.

Experiment D is a one-time memory intervention at decision 6 followed by a
six-period closed-loop continuation of all four agents.  A matched-A branch
pre-generates the NumPy state used before each Foundation step; every other
branch is forced onto that same common-random-number schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

from llm_providers import MultiModelLLM

from .actions import ActionDecision, ActionParseError, parse_direct_action
from .budget import BudgetExceeded, RunBudget
from .foundation_adapter import (
    build_foundation_actions,
    capture_foundation_snapshots,
    derive_foundation_transitions,
)
from .m3_semantic import (
    ActionGuidance,
    ConditionPredicate,
    ContextScope,
)
from .pilot_checkpoint import (
    PilotCheckpoint,
    _apply_shock,
    _assert_no_future_shock_leak,
    _post_ledger_batch,
    _pre_ledger_batch,
    _shock_by_decision_t,
    _shock_prompt_text,
    canonical_hash,
    capture_environment_state,
    config_from_dict,
    decode_numpy_rng_state,
    decode_python_rng_state,
    encode_numpy_rng_state,
    encode_python_rng_state,
    restore_pilot_checkpoint,
)
from .prompts import build_base_decision_prompt, compose_decision_prompt
from .runner import (
    FIXED_ERRONEOUS_RULE,
    RUNNER_SCHEMA_VERSION,
    _action_parse_mode,
    _append_provider_call_journal,
    _context_observation,
    _m2_state,
    _monthly_inflation,
    _provider_row,
    _prompt_state,
    preflight_p95_reservation_for_call,
    validate_preflight_p95_reservations,
    verify_provider_call_journal,
)


PILOT_CONTINUATION_SCHEMA_VERSION = "finevo-pilot-continuation-v2"
PILOT_NARRATIVE_SCHEMA_VERSION = "finevo-pilot-narrative-v2"
DEFAULT_CONTINUATION_HORIZON = 6
MEMORY_PULSE_TREATMENTS = (
    "no-memory",
    "shuffled-episodic",
    "wrong-context",
)
MEMORY_PULSE_CONTRACT = {
    "schema_version": "finevo-pilot-d-memory-pulse-v1",
    "treatment_arms": list(MEMORY_PULSE_TREATMENTS),
    "focal_agent_id": 0,
    "wrong_context_source_agent_id": 1,
    "decision_t": 6,
    "duration_decisions": 1,
    "continuation_horizon_steps": DEFAULT_CONTINUATION_HORIZON,
    "pulse_at_first_continuation_step": True,
    "direct_treatment_only_at_pulse": True,
    "claim_label": (
        "focal-agent decision-6 memory pulse with six-step downstream "
        "continuation"
    ),
}
NARRATIVE_PULSE_CONTRACT = {
    "schema_version": "finevo-pilot-d-narrative-pulse-v1",
    "treatment_narratives": ["aligned", "paraphrase", "opposite"],
    "focal_agent_id": 0,
    "decision_t": 6,
    "duration_decisions": 1,
    "continuation_horizon_steps": DEFAULT_CONTINUATION_HORIZON,
    "pulse_at_first_continuation_step": True,
    "direct_treatment_only_at_pulse": True,
}
SHUFFLE_ALGORITHM = "checkpoint-bound-sha256-rank-permutation-v1"
DEFAULT_TREATMENTS = (
    "matched-a",
    "matched-b",
    "no-memory",
    "shuffled-episodic",
    "wrong-context",
    "erroneous-verified",
    "erroneous-unverified",
)
DEFAULT_NARRATIVES = {
    "none": "",
    "aligned": (
        "Savings now earn a high interest rate. Preserving cash and reducing "
        "current consumption is directionally consistent with this environment."
    ),
    "paraphrase": (
        "Returns on saved funds are elevated, so conserving available cash and "
        "lowering near-term spending points in the same direction."
    ),
    "opposite": (
        "A high interest rate makes immediate spending more attractive; increase "
        "current consumption rather than preserving cash."
    ),
}


class PilotContinuationError(RuntimeError):
    """Raised when a preregistered paired-continuation invariant fails."""


def _gini(values: Sequence[float]) -> float:
    ordered = sorted(max(float(value), 0.0) for value in values)
    if not ordered or sum(ordered) == 0:
        return 0.0
    count = len(ordered)
    total = sum(ordered)
    weighted = sum(
        (index + 1) * value for index, value in enumerate(ordered)
    )
    return 2 * weighted / (count * total) - (count + 1) / count


def _bad_rule_candidate(memory: Any) -> str:
    episodes = list(memory.episodic.finalized_episodes)
    min_support = int(memory.semantic.min_candidate_support)
    if len(episodes) < min_support:
        raise PilotContinuationError(
            "erroneous-rule treatment has insufficient finalized focal episodes"
        )
    payload = json.loads(json.dumps(FIXED_ERRONEOUS_RULE, sort_keys=True))
    payload["supporting_episode_ids"] = [
        episode.episode_id for episode in episodes[:min_support]
    ]
    return json.dumps(
        payload,
        sort_keys=True,
    )


def _apply_semantic_treatment(
    treatment: str,
    *,
    memory: Any,
    decision_t: int,
) -> dict[str, Any]:
    if treatment not in {"erroneous-verified", "erroneous-unverified"}:
        return {
            "kind": treatment,
            "applied": treatment
            in {"no-memory", "shuffled-episodic", "wrong-context"},
        }
    if memory.semantic is None:
        raise PilotContinuationError(
            f"{treatment} requires the semantic track to be enabled"
        )
    verifier_enabled = treatment == "erroneous-verified"
    # Both causal arms receive byte-identical rule and creation-event state.
    # The verifier assignment is deliberately kept outside this injected state
    # and is applied only when post-intervention outcomes are finalized.
    common_provenance = {
        "experiment": "D",
        "checkpoint_decision_t": decision_t,
        "fixed_rule_hash": canonical_hash(FIXED_ERRONEOUS_RULE),
        "forced_active_common_start": True,
        "lifecycle_assignment": "after-common-start",
        "lifecycle_policy_at_injection": "unassigned",
    }
    rule = memory.semantic.inject_active_rule(
        condition=ConditionPredicate.from_dict(
            FIXED_ERRONEOUS_RULE["condition"]
        ),
        action_guidance=ActionGuidance.from_dict(
            FIXED_ERRONEOUS_RULE["action_guidance"]
        ),
        outcome_criterion=memory.semantic.registered_outcome_criterion,
        rationale=str(FIXED_ERRONEOUS_RULE["rationale"]),
        current_t=decision_t,
        injection_id=(
            f"experiment-d-shared-error:{memory.run_id}:{memory.seed}:"
            f"a{memory.agent_id}:t{decision_t}"
        ),
        provenance=common_provenance,
        initial_confidence=1.0,
        context_scope=ContextScope.global_scope(),
    )
    rule_payload = rule.to_dict()
    start_payload = {
        "memory": memory.to_dict(),
        "injected_rule": rule_payload,
    }
    return {
        "kind": treatment,
        "applied": True,
        "rule_id": rule.rule_id,
        "rule_status": rule.status,
        "verifier_bypassed": not verifier_enabled,
        "verifier_enabled_after_start": verifier_enabled,
        "lifecycle_policy": (
            "observe-and-retire"
            if verifier_enabled
            else "skip-injected-rule-evidence-and-retirement"
        ),
        "forced_active_common_start": True,
        "forced_active_rule_hash": canonical_hash(rule_payload),
        "forced_active_memory_hash": canonical_hash(memory.to_dict()),
        "forced_active_start_hash": canonical_hash(start_payload),
    }


def _checkpoint_bound_shuffle(
    bundle: Any,
    *,
    checkpoint_hash: str,
    focal_agent_id: int,
    decision_t: int,
) -> tuple[list[Any], dict[str, Any]]:
    """Return a non-trivial, checkpoint-bound permutation and its receipt."""

    hits = list(bundle.episodic_hits)
    if len(hits) < 2:
        raise PilotContinuationError(
            "shuffled-episodic pulse requires at least two retrieved episodes"
        )
    original_ids = [str(hit.episode.episode_id) for hit in hits]
    ranked = sorted(
        range(len(hits)),
        key=lambda index: (
            hashlib.sha256(
                json.dumps(
                    {
                        "domain": SHUFFLE_ALGORITHM,
                        "checkpoint_hash": checkpoint_hash,
                        "decision_t": int(decision_t),
                        "focal_agent_id": int(focal_agent_id),
                        "episode_id": original_ids[index],
                        "original_index": index,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            index,
        ),
    )
    identity = list(range(len(hits)))
    reversed_identity = list(reversed(identity))
    adjustment = "none"
    if ranked == identity:
        ranked = ranked[1:] + ranked[:1]
        adjustment = "rotate-left-to-avoid-identity"
    if len(ranked) > 2 and ranked == reversed_identity:
        ranked = ranked[1:] + ranked[:1]
        adjustment = "rotate-left-to-avoid-fixed-reversal"
    shuffled = [hits[index] for index in ranked]
    binding = {
        "schema_version": "finevo-pilot-d-shuffle-binding-v1",
        "algorithm": SHUFFLE_ALGORITHM,
        "checkpoint_hash": checkpoint_hash,
        "decision_t": int(decision_t),
        "focal_agent_id": int(focal_agent_id),
        "original_episode_ids": original_ids,
        "permutation": ranked,
        "shuffled_episode_ids": [
            str(hit.episode.episode_id) for hit in shuffled
        ],
        "non_identity": ranked != identity,
        "not_fixed_reversal": (
            len(ranked) <= 2 or ranked != reversed_identity
        ),
        "deterministic_adjustment": adjustment,
    }
    binding["permutation_hash"] = canonical_hash(binding)
    return shuffled, binding


def _shuffled_memory_text(
    bundle: Any,
    *,
    checkpoint_hash: str,
    focal_agent_id: int,
    decision_t: int,
) -> tuple[str, dict[str, Any]]:
    shuffled_hits, shuffle_binding = _checkpoint_bound_shuffle(
        bundle,
        checkpoint_hash=checkpoint_hash,
        focal_agent_id=focal_agent_id,
        decision_t=decision_t,
    )
    parts: list[str] = []
    if shuffled_hits:
        parts.append("Finalized experience evidence:")
        parts.extend(
            f"- {hit.episode.to_prompt_text()}"
            for hit in shuffled_hits
        )
    if bundle.active_rules:
        parts.append("Verified active rules:")
        parts.extend(
            f"- {rule.to_prompt_text()}" for rule in bundle.active_rules
        )
    return " ".join(parts), shuffle_binding


def _memory_text_for_treatment(
    treatment: str,
    *,
    checkpoint_hash: str,
    focal_agent_id: int,
    agent_id: int,
    bundles: Mapping[int, Any],
    intervention_t: int,
    decision_t: int,
) -> tuple[str, Mapping[str, Any] | None]:
    bundle = bundles[agent_id]
    if agent_id != focal_agent_id or decision_t != intervention_t:
        return bundle.memory_prompt, None
    if treatment == "no-memory":
        treated = ""
        return treated, {
            "schema_version": "finevo-pilot-d-memory-pulse-binding-v1",
            "kind": treatment,
            "checkpoint_hash": checkpoint_hash,
            "focal_agent_id": focal_agent_id,
            "decision_t": decision_t,
            "pulse_only": True,
            "duration_decisions": 1,
            "original_memory_hash": canonical_hash(bundle.memory_prompt),
            "treated_memory_hash": canonical_hash(treated),
            "shuffle_binding": None,
            "wrong_context_source_agent_id": None,
        }
    if treatment == "shuffled-episodic":
        treated, shuffle_binding = _shuffled_memory_text(
            bundle,
            checkpoint_hash=checkpoint_hash,
            focal_agent_id=focal_agent_id,
            decision_t=decision_t,
        )
        return treated, {
            "schema_version": "finevo-pilot-d-memory-pulse-binding-v1",
            "kind": treatment,
            "checkpoint_hash": checkpoint_hash,
            "focal_agent_id": focal_agent_id,
            "decision_t": decision_t,
            "pulse_only": True,
            "duration_decisions": 1,
            "original_memory_hash": canonical_hash(bundle.memory_prompt),
            "treated_memory_hash": canonical_hash(treated),
            "shuffle_binding": shuffle_binding,
            "wrong_context_source_agent_id": None,
        }
    if treatment == "wrong-context":
        donor = next(
            candidate
            for candidate in sorted(bundles)
            if candidate != focal_agent_id
        )
        treated = bundles[donor].memory_prompt
        return treated, {
            "schema_version": "finevo-pilot-d-memory-pulse-binding-v1",
            "kind": treatment,
            "checkpoint_hash": checkpoint_hash,
            "focal_agent_id": focal_agent_id,
            "decision_t": decision_t,
            "pulse_only": True,
            "duration_decisions": 1,
            "original_memory_hash": canonical_hash(bundle.memory_prompt),
            "treated_memory_hash": canonical_hash(treated),
            "shuffle_binding": None,
            "wrong_context_source_agent_id": donor,
        }
    return bundle.memory_prompt, None


def _provider_state_restore(
    checkpoint: PilotCheckpoint, llm: MultiModelLLM
) -> None:
    binding = checkpoint.payload["provider_binding"]
    if llm.get_model_name() != binding["model_name"]:
        raise PilotContinuationError(
            "continuation provider differs from checkpoint provider"
        )
    state = binding.get("state")
    if state is None:
        return
    restore = getattr(llm.provider, "restore_checkpoint_state", None)
    if not callable(restore):
        raise PilotContinuationError(
            "stateful checkpoint provider has no restore_checkpoint_state()"
        )
    restore(state)


def _pre_generate_rng_schedule(
    checkpoint: PilotCheckpoint,
    *,
    horizon: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create a checkpoint-bound CRN schedule without touching provider/global RNG.

    Each period receives independent NumPy and Python RNG states derived through
    a domain-separated SHA-256 seed.  The derivation reads only immutable
    checkpoint material and runs before the first branch provider call.
    """

    source = {
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "numpy_rng_after_prefix": checkpoint.payload[
            "numpy_rng_after_prefix"
        ],
        "python_rng_after_prefix": checkpoint.payload[
            "python_rng_after_prefix"
        ],
    }
    source_hash = canonical_hash(source)
    schedule: list[dict[str, Any]] = []
    for offset in range(horizon):
        material = json.dumps(
            {
                "domain": "finevo-pilot-d-environment-rng-v1",
                "source_hash": source_hash,
                "offset": offset,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(material).digest()
        numpy_seed = int.from_bytes(digest[:4], "big", signed=False)
        python_seed = int.from_bytes(digest[4:12], "big", signed=False)
        numpy_rng = np.random.RandomState(numpy_seed)
        python_rng = random.Random(python_seed)
        schedule.append(
            {
                "offset": offset,
                "derivation_hash": hashlib.sha256(material).hexdigest(),
                "numpy": encode_numpy_rng_state(numpy_rng.get_state()),
                "python": encode_python_rng_state(python_rng.getstate()),
            }
        )
    binding = {
        "schema_version": "finevo-pilot-d-rng-schedule-v1",
        "derivation": "checkpoint-bound-domain-separated-sha256",
        "generated_before_provider_calls": True,
        "source_hash": source_hash,
        "schedule_hash": canonical_hash(schedule),
        "horizon": horizon,
    }
    return schedule, binding


def _completion_usage_row(
    completion: Any,
    *,
    treatment: str,
    decision_t: int,
    agent_id: int,
    prompt_hash: str,
) -> dict[str, Any]:
    row = _provider_row(
        completion,
        call_kind="pilot_continuation_action",
        decision_t=decision_t,
        agent_id=agent_id,
        prompt_hash=prompt_hash,
    )
    row["treatment"] = treatment
    return row


def _normalize_journal_target(
    value: Mapping[str, Any] | None,
) -> tuple[Path, str, str | None] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "run_id",
        "contract_hash",
    }:
        raise PilotContinuationError(
            "provider journal target must contain path/run_id/contract_hash"
        )
    path = Path(value["path"]).resolve()
    run_id = value["run_id"]
    contract_hash = value["contract_hash"]
    if not isinstance(run_id, str) or not run_id:
        raise PilotContinuationError("provider journal target run_id is invalid")
    if contract_hash is not None and (
        not isinstance(contract_hash, str) or len(contract_hash) != 64
    ):
        raise PilotContinuationError(
            "provider journal target contract_hash is invalid"
        )
    if path.exists():
        raise PilotContinuationError(
            "provider journal already exists; refusing branch redispatch"
        )
    return path, run_id, contract_hash


def _journal_event(
    target: tuple[Path, str, str | None] | None,
    *,
    event_type: str,
    payload: Mapping[str, Any],
) -> None:
    if target is None:
        return
    path, run_id, contract_hash = target
    _append_provider_call_journal(
        path,
        run_id=run_id,
        contract_hash=contract_hash,
        event_type=event_type,
        payload=payload,
    )


def _parse_disposition(
    usage_row: Mapping[str, Any],
    *,
    parse_status: str,
    parse_mode: str,
    accepted: bool,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "call_kind": usage_row["call_kind"],
        "decision_t": usage_row["decision_t"],
        "agent_id": usage_row["agent_id"],
        "prompt_hash": usage_row["prompt_hash"],
        "raw_output_hash": usage_row["raw_output_hash"],
        "parse_status": parse_status,
        "parse_mode": parse_mode,
        "accepted": accepted,
    }
    payload.update(dict(extra or {}))
    return payload


def _journal_binding(
    target: tuple[Path, str, str | None] | None,
    *,
    expected_completions: int,
) -> dict[str, Any]:
    if target is None:
        return {
            "enabled": False,
            "path": None,
            "file_sha256": None,
            "journal_sha256": None,
            "run_id": None,
            "contract_hash": None,
            "event_count": 0,
            "completion_event_count": 0,
            "parse_disposition_event_count": 0,
            "terminal_dispositions_verified": False,
        }
    path, run_id, contract_hash = target
    journal = verify_provider_call_journal(
        path,
        expected_run_id=run_id,
        expected_contract_hash=contract_hash,
        require_terminal_dispositions=True,
    )
    completions = [
        event
        for event in journal["events"]
        if event["event_type"] == "completion_received"
    ]
    dispositions = [
        event
        for event in journal["events"]
        if event["event_type"] == "parse_disposition"
    ]
    if (
        len(completions) != expected_completions
        or len(dispositions) != expected_completions
    ):
        raise PilotContinuationError(
            "provider journal does not contain the complete branch denominator"
        )
    return {
        "enabled": True,
        "path": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "journal_sha256": journal["journal_sha256"],
        "run_id": run_id,
        "contract_hash": contract_hash,
        "event_count": len(journal["events"]),
        "completion_event_count": len(completions),
        "parse_disposition_event_count": len(dispositions),
        "terminal_dispositions_verified": True,
    }


def _finalize_episode_with_lifecycle_policy(
    *,
    memory: Any,
    decision_t: int,
    next_state: Mapping[str, Any],
    outcome: Mapping[str, Any],
    reward: float,
    flow_utility: float,
    excluded_rule_id: str | None,
) -> Any:
    """Finalize M2 while optionally withholding one injected rule from M3.

    This pilot-only path mirrors ``VerifiedDualTrackMemory.finalize_episode``.
    It keeps all other rules on the normal evidence path while ensuring the
    unverified D arm starts from the same injected state and disables its
    verifier only for subsequent outcomes.
    """

    if excluded_rule_id is None:
        return memory.finalize_episode(
            decision_t=decision_t,
            next_state=next_state,
            outcome=outcome,
            reward=reward,
            flow_utility=flow_utility,
        )
    if decision_t not in memory._decision_ids:
        raise PilotContinuationError(
            "begin_episode must precede policy-controlled finalization"
        )
    decision_id = memory._decision_ids[decision_t]
    record = memory.episodic.finalize_episode(
        decision_id,
        outcome_t=int(decision_t) + 1,
        next_state=next_state,
        outcome=outcome,
        reward=reward,
        flow_utility=flow_utility,
    )
    if memory.semantic is None:
        raise PilotContinuationError(
            "policy-controlled finalization requires semantic memory"
        )
    observed_rule_ids = {
        rule.rule_id for rule in tuple(memory.semantic.rules)
    }
    if excluded_rule_id not in observed_rule_ids:
        raise PilotContinuationError(
            "unverified lifecycle exclusion rule is missing"
        )
    for rule in tuple(memory.semantic.rules):
        if rule.rule_id == excluded_rule_id:
            continue
        memory.semantic.observe_episode(
            rule.rule_id,
            record.episode_id,
            current_t=int(decision_t) + 1,
        )
    memory._decision_ids.pop(decision_t)
    return record


def _complete_actions(
    *,
    config: Any,
    llm: MultiModelLLM,
    budget: RunBudget,
    treatment: str,
    decision_t: int,
    prompts: Sequence[str],
) -> list[Any]:
    return llm.get_multiple_structured_completions(
        [[{"role": "user", "content": prompt}] for prompt in prompts],
        temperature=config.temperature,
        max_tokens=config.action_max_tokens,
        top_p=config.top_p,
        budget=budget,
        labels=[
            f"pilot-D:{treatment}:t{decision_t}:a{agent_id}"
            for agent_id in range(config.num_agents)
        ],
        tags=[
            {
                "call_kind": "pilot_continuation_action",
                "treatment": treatment,
                "decision_t": decision_t,
                "agent_id": agent_id,
            }
            for agent_id in range(config.num_agents)
        ],
        estimated_usages=[
            preflight_p95_reservation_for_call(
                config,
                provider_model_name=llm.get_model_name(),
                call_kind="action",
                prompt=prompt,
                max_tokens=config.action_max_tokens,
            )
            for prompt in prompts
        ],
        max_retries=config.max_retries,
        seed=config.seed if config.send_decoding_seed else None,
    )


def _metrics(
    *,
    state: Any,
    continuation_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    focal_agent_id: int,
    initial_wealths: Mapping[str, float],
) -> dict[str, Any]:
    focal = str(focal_agent_id)
    focal_rows = [
        row for row in continuation_rows if row["agent_id"] == focal
    ]
    focal_actions = [
        row["decisions"][focal] for row in action_rows
    ]
    final_wealths = {
        str(agent_id): float(
            state.env.get_agent(str(agent_id)).inventory["Coin"]
        )
        for agent_id in range(state.config.num_agents)
    }
    all_flows = [float(row["flow_utility"]) for row in continuation_rows]
    focal_flows = [float(row["flow_utility"]) for row in focal_rows]
    first_action_row = action_rows[0]
    first_focal_decision = first_action_row["decisions"][focal]
    first_focal_ledger = next(
        row
        for row in first_action_row["ledger_rows"]
        if str(row["agent_id"]) == focal
    )
    first_population_ledger = list(first_action_row["ledger_rows"])
    first_population_wealth = [
        float(row["wealth_post"]) for row in first_population_ledger
    ]
    return {
        "focal": {
            "agent_id": focal_agent_id,
            "initial_wealth": float(initial_wealths[focal]),
            "final_wealth": final_wealths[focal],
            "wealth_change": final_wealths[focal]
            - float(initial_wealths[focal]),
            "flow_utility_sum": sum(focal_flows),
            "discounted_flow_utility_sum": sum(
                float(row["discounted_flow_utility"])
                for row in focal_rows
            ),
            "mean_labor_hours": mean(
                float(row["executed_labor_hours"])
                for row in focal_actions
            ),
            "mean_consumption_rate": mean(
                float(row["executed_consumption_rate"])
                for row in focal_actions
            ),
            "first_step": {
                "labor_hours": float(
                    first_focal_decision["executed_labor_hours"]
                ),
                "consumption_rate": float(
                    first_focal_decision["executed_consumption_rate"]
                ),
                "immediate_flow_utility": float(
                    first_focal_ledger["flow_utility"]
                ),
                "next_wealth": float(first_focal_ledger["wealth_post"]),
                "next_cumulative_production": float(
                    first_focal_ledger["cumulative_production_post"]
                ),
            },
        },
        "population": {
            "num_agents": state.config.num_agents,
            "average_initial_wealth": mean(initial_wealths.values()),
            "average_final_wealth": mean(final_wealths.values()),
            "average_wealth_change": mean(
                final_wealths[key] - float(initial_wealths[key])
                for key in final_wealths
            ),
            "gini_final_wealth": _gini(list(final_wealths.values())),
            "flow_utility_sum": sum(all_flows),
            "mean_agent_period_flow_utility": mean(all_flows),
            "mean_low_labor_rate": mean(
                float(row["low_labor_rate"]) for row in action_rows
            ),
            "first_step": {
                "average_next_wealth": mean(first_population_wealth),
                "gini_next_wealth": _gini(first_population_wealth),
                "flow_utility_sum": sum(
                    float(row["flow_utility"])
                    for row in first_population_ledger
                ),
                "low_labor_rate": float(
                    first_action_row["low_labor_rate"]
                ),
            },
        },
    }


def _run_branch(
    *,
    checkpoint: PilotCheckpoint,
    llm: MultiModelLLM,
    budget: RunBudget,
    treatment: str,
    horizon: int,
    focal_agent_id: int,
    common_rng_schedule: Sequence[Mapping[str, Any]],
    strict_code_binding: bool,
    narrative_text: str = "",
    narrative_id: str | None = None,
    provider_call_journal: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _provider_state_restore(checkpoint, llm)
    state = restore_pilot_checkpoint(
        checkpoint, strict_code_binding=strict_code_binding
    )
    config = state.config
    intervention_t = state.next_decision_t
    semantic_intervention = _apply_semantic_treatment(
        treatment,
        memory=state.memories[focal_agent_id],
        decision_t=intervention_t,
    )
    excluded_rule_id = (
        str(semantic_intervention["rule_id"])
        if treatment == "erroneous-unverified"
        else None
    )
    initial_proposals = dict(state.proposals_made)
    initial_wealths = {
        str(agent_id): float(
            state.env.get_agent(str(agent_id)).inventory["Coin"]
        )
        for agent_id in range(config.num_agents)
    }
    branch_rows: list[dict[str, Any]] = []
    continuation_ledger_rows: list[dict[str, Any]] = []
    api_usage_rows: list[dict[str, Any]] = []
    rng_schedule: list[dict[str, Any]] = []
    memory_pulse_binding: Mapping[str, Any] | None = None
    journal_target = _normalize_journal_target(provider_call_journal)
    shocks = _shock_by_decision_t(config)
    if len(common_rng_schedule) != horizon:
        raise PilotContinuationError(
            "pre-generated RNG schedule length differs from continuation horizon"
        )

    for offset in range(horizon):
        decision_t = intervention_t + offset
        shock = shocks.get(decision_t)
        _apply_shock(state.env, shock)
        pre_snapshots = capture_foundation_snapshots(
            state.env,
            expected_timestamp=decision_t,
            labor_step=config.labor_step,
            max_labor_hours=config.max_labor_hours,
            consumption_step=config.consumption_step,
        )
        current_inflation = _monthly_inflation(state.env.world)
        bundles: dict[int, Any] = {}
        base_prompts: dict[int, str] = {}
        for agent_id in range(config.num_agents):
            agent_key = str(agent_id)
            snapshot = pre_snapshots[agent_key]
            retrieval_state = _m2_state(
                snapshot,
                low_labor_rate=state.previous_low_labor_rate,
                inflation=current_inflation,
            )
            context = _context_observation(
                decision_t=decision_t,
                price=snapshot.price,
                interest_rate=snapshot.interest_rate,
                low_labor_rate=state.previous_low_labor_rate,
                inflation=current_inflation,
                wealth=snapshot.wealth,
                employed=snapshot.employed,
            )
            bundle = state.memories[agent_id].prepare_decision(
                decision_t=decision_t,
                context_observation=context,
                retrieval_state=retrieval_state,
                retrieval_k=config.retrieval_k,
                rule_budget=config.rule_budget,
            )
            bundles[agent_id] = bundle
            base_prompts[agent_id] = build_base_decision_prompt(
                _prompt_state(
                    state.env,
                    agent_id=agent_id,
                    decision_t=decision_t,
                    snapshot=snapshot,
                    last_transition=state.last_transitions.get(agent_key),
                    last_decision=state.last_decisions.get(agent_key),
                    max_labor_hours=config.max_labor_hours,
                ),
                config.utility,
                event_text=_shock_prompt_text(shock),
                causal_context_summary=bundle.protected_context_prompt,
            )
            if (
                narrative_text
                and agent_id == focal_agent_id
                and decision_t == intervention_t
            ):
                base_prompts[agent_id] += (
                    "\nControlled narrative fixture (content intervention only): "
                    + narrative_text
                )
            _assert_no_future_shock_leak(
                base_prompts[agent_id],
                current_event=shock,
                schedule=config.shock_schedule,
                decision_t=decision_t,
            )

        composed = []
        memory_texts: dict[str, str] = {}
        memory_pulse_bindings: dict[str, Mapping[str, Any]] = {}
        for agent_id in range(config.num_agents):
            memory_text, pulse_binding = _memory_text_for_treatment(
                treatment,
                checkpoint_hash=checkpoint.checkpoint_hash,
                focal_agent_id=focal_agent_id,
                agent_id=agent_id,
                bundles=bundles,
                intervention_t=intervention_t,
                decision_t=decision_t,
            )
            if pulse_binding is not None:
                memory_pulse_bindings[str(agent_id)] = pulse_binding
                if memory_pulse_binding is not None:
                    raise PilotContinuationError(
                        "memory pulse was applied more than once in one branch"
                    )
                memory_pulse_binding = pulse_binding
            prompt = compose_decision_prompt(
                base_prompts[agent_id], memory_text
            )
            composed.append(prompt)
            memory_texts[str(agent_id)] = memory_text
        prompt_hashes = [
            prompt.full_prompt_hash for prompt in composed
        ]
        try:
            completions = _complete_actions(
                config=config,
                llm=llm,
                budget=budget,
                treatment=treatment,
                decision_t=decision_t,
                prompts=[prompt.full_prompt for prompt in composed],
            )
        except Exception as exc:
            settled = getattr(exc, "structured_completions", None)
            if settled is not None:
                if (
                    not isinstance(settled, tuple)
                    or len(settled) != config.num_agents
                ):
                    raise PilotContinuationError(
                        "failed continuation batch exposed an incomplete "
                        "completion denominator"
                    ) from exc
                failure_mode = (
                    "budget_failure"
                    if isinstance(exc, BudgetExceeded)
                    else "batch_failure"
                )
                for agent_id, completion in enumerate(settled):
                    usage_row = _completion_usage_row(
                        completion,
                        treatment=treatment,
                        decision_t=decision_t,
                        agent_id=agent_id,
                        prompt_hash=prompt_hashes[agent_id],
                    )
                    _journal_event(
                        journal_target,
                        event_type="completion_received",
                        payload=usage_row,
                    )
                    _journal_event(
                        journal_target,
                        event_type="parse_disposition",
                        payload=_parse_disposition(
                            usage_row,
                            parse_status="not_evaluated",
                            parse_mode=failure_mode,
                            accepted=False,
                            extra={
                                "rejection": type(exc).__name__,
                            },
                        ),
                    )
            raise
        if len(completions) != config.num_agents:
            raise PilotContinuationError(
                "continuation action batch denominator mismatch"
            )
        usage_rows: list[dict[str, Any]] = []
        for agent_id, completion in enumerate(completions):
            usage_row = _completion_usage_row(
                completion,
                treatment=treatment,
                decision_t=decision_t,
                agent_id=agent_id,
                prompt_hash=prompt_hashes[agent_id],
            )
            api_usage_rows.append(usage_row)
            usage_rows.append(usage_row)
            _journal_event(
                journal_target,
                event_type="completion_received",
                payload=usage_row,
            )

        def close_later_dispositions(current_agent_id: int) -> None:
            for pending_row in usage_rows[current_agent_id + 1 :]:
                _journal_event(
                    journal_target,
                    event_type="parse_disposition",
                    payload=_parse_disposition(
                        pending_row,
                        parse_status="not_evaluated",
                        parse_mode="prior_batch_failure",
                        accepted=False,
                    ),
                )

        decisions: dict[str, ActionDecision] = {}
        for agent_id, (completion, usage_row) in enumerate(
            zip(completions, usage_rows, strict=True)
        ):
            if not completion.ok or completion.text == "Error":
                _journal_event(
                    journal_target,
                    event_type="parse_disposition",
                    payload=_parse_disposition(
                        usage_row,
                        parse_status="unavailable",
                        parse_mode=completion.output_disposition,
                        accepted=False,
                        extra={"rejection": "provider_failure"},
                    ),
                )
                close_later_dispositions(agent_id)
                raise PilotContinuationError(
                    f"provider failure in {treatment} at t={decision_t}, "
                    f"agent={agent_id}"
                )
            if (
                len(completion.text.encode("utf-8"))
                > config.action_max_visible_json_bytes
            ):
                _journal_event(
                    journal_target,
                    event_type="parse_disposition",
                    payload=_parse_disposition(
                        usage_row,
                        parse_status="failure",
                        parse_mode="visible_limit_exceeded",
                        accepted=False,
                    ),
                )
                close_later_dispositions(agent_id)
                raise PilotContinuationError(
                    f"continuation action exceeds visible JSON limit at "
                    f"t={decision_t}, agent={agent_id}"
                )
            try:
                decision = parse_direct_action(
                    completion.text,
                    max_labor_hours=config.max_labor_hours,
                    labor_step=config.labor_step,
                    consumption_step=config.consumption_step,
                )
            except ActionParseError as exc:
                _journal_event(
                    journal_target,
                    event_type="parse_disposition",
                    payload=_parse_disposition(
                        usage_row,
                        parse_status="failure",
                        parse_mode="parse_failure",
                        accepted=False,
                    ),
                )
                close_later_dispositions(agent_id)
                raise PilotContinuationError(
                    f"continuation action parse failure at t={decision_t}, "
                    f"agent={agent_id}: {exc}"
                ) from exc
            action_parse_mode = _action_parse_mode(completion.text, decision)
            if action_parse_mode not in config.accepted_action_parse_modes:
                _journal_event(
                    journal_target,
                    event_type="parse_disposition",
                    payload=_parse_disposition(
                        usage_row,
                        parse_status="success",
                        parse_mode=action_parse_mode,
                        accepted=False,
                        extra={"rejection": "unaccepted_parse_mode"},
                    ),
                )
                close_later_dispositions(agent_id)
                raise PilotContinuationError(
                    f"continuation action parse mode {action_parse_mode!r} "
                    f"is not accepted at t={decision_t}, agent={agent_id}"
                )
            if config.fail_on_clipped_action and decision.clipped:
                _journal_event(
                    journal_target,
                    event_type="parse_disposition",
                    payload=_parse_disposition(
                        usage_row,
                        parse_status="success",
                        parse_mode=action_parse_mode,
                        accepted=False,
                        extra={"rejection": "clipped_action"},
                    ),
                )
                close_later_dispositions(agent_id)
                raise PilotContinuationError(
                    f"clipped continuation action at t={decision_t}, "
                    f"agent={agent_id}"
                )
            _journal_event(
                journal_target,
                event_type="parse_disposition",
                payload=_parse_disposition(
                    usage_row,
                    parse_status="success",
                    parse_mode=action_parse_mode,
                    accepted=True,
                ),
            )
            agent_key = str(agent_id)
            decisions[agent_key] = decision
            pre_state = _m2_state(
                pre_snapshots[agent_key],
                low_labor_rate=state.previous_low_labor_rate,
                inflation=current_inflation,
            )
            state.memories[agent_id].begin_episode(
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
        state.ledger.capture_pre(decision_t, pre_batch)
        env_actions = build_foundation_actions(
            decisions,
            labor_step=config.labor_step,
            max_labor_hours=config.max_labor_hours,
            consumption_step=config.consumption_step,
        )
        seed_state = dict(common_rng_schedule[offset])
        _apply_shock(state.env, shock)
        random.setstate(decode_python_rng_state(seed_state["python"]))
        if encode_python_rng_state(random.getstate()) != seed_state["python"]:
            raise PilotContinuationError(
                f"common Python RNG seed mismatch in {treatment} "
                f"at t={decision_t}"
            )
        _, rewards, done, _ = state.env.step(
            env_actions,
            seed_state=decode_numpy_rng_state(seed_state["numpy"]),
        )
        observed_seed = encode_numpy_rng_state(
            state.env.replay_log["step"][-1]["seed_state"]
        )
        if observed_seed != seed_state["numpy"]:
            raise PilotContinuationError(
                f"common RNG seed mismatch in {treatment} at t={decision_t}"
            )
        rng_schedule.append(seed_state)
        transitions = derive_foundation_transitions(
            state.env,
            pre_snapshots=pre_snapshots,
            decisions=decisions,
            expected_outcome_t=decision_t + 1,
            labor_step=config.labor_step,
            max_labor_hours=config.max_labor_hours,
            consumption_step=config.consumption_step,
        )
        post_batch = _post_ledger_batch(transitions)
        utility_rows = state.ledger.capture_post(decision_t, post_batch)
        rows_by_agent = {row.agent_id: row for row in utility_rows}
        continuation_ledger_rows.extend(
            row.to_dict() for row in utility_rows
        )
        current_low_labor_rate = mean(
            float(
                decision.executed_labor_hours
                < config.low_labor_threshold_hours
            )
            for decision in decisions.values()
        )
        realized_inflation = _monthly_inflation(state.env.world)
        for agent_id in range(config.num_agents):
            agent_key = str(agent_id)
            decision = decisions[agent_key]
            transition = transitions[agent_key]
            _finalize_episode_with_lifecycle_policy(
                memory=state.memories[agent_id],
                decision_t=decision_t,
                next_state=_m2_state(
                    transition.post,
                    low_labor_rate=current_low_labor_rate,
                    inflation=realized_inflation,
                ),
                outcome=transition.to_m2_outcome(decision),
                reward=float(rewards[agent_key]),
                flow_utility=rows_by_agent[agent_key].flow_utility,
                excluded_rule_id=(
                    excluded_rule_id
                    if agent_id == focal_agent_id
                    else None
                ),
            )
            state.last_decisions[agent_key] = decision
            state.last_transitions[agent_key] = transition
        state.previous_low_labor_rate = current_low_labor_rate
        branch_rows.append(
            {
                "decision_t": decision_t,
                "outcome_t": decision_t + 1,
                "shock_event": None if shock is None else shock.to_dict(),
                "shock_event_hash": canonical_hash(
                    None if shock is None else shock.to_dict()
                ),
                "shock_prompt_text": _shock_prompt_text(shock),
                "rng_pre_step_hash": canonical_hash(seed_state),
                "decisions": {
                    key: value.to_dict() for key, value in decisions.items()
                },
                "prompt_hashes": {
                    str(agent_id): composed[agent_id].full_prompt_hash
                    for agent_id in range(config.num_agents)
                },
                "memory_hashes": {
                    str(agent_id): composed[agent_id].memory_hash
                    for agent_id in range(config.num_agents)
                },
                "memory_texts": memory_texts,
                "memory_pulse_bindings": memory_pulse_bindings,
                "ledger_rows": [
                    row.to_dict() for row in utility_rows
                ],
                "environment_state_hash": canonical_hash(
                    capture_environment_state(state.env)
                ),
                "low_labor_rate": current_low_labor_rate,
                "monthly_inflation": realized_inflation,
                "done": bool(done["__all__"]),
            }
        )

    if state.proposals_made != initial_proposals:
        raise PilotContinuationError(
            f"semantic proposal counters changed in frozen branch {treatment}"
        )
    if treatment in MEMORY_PULSE_TREATMENTS:
        if (
            memory_pulse_binding is None
            or memory_pulse_binding.get("pulse_only") is not True
            or memory_pulse_binding.get("decision_t") != intervention_t
            or memory_pulse_binding.get("focal_agent_id") != focal_agent_id
        ):
            raise PilotContinuationError(
                f"{treatment} lacks its exact focal decision-6 pulse binding"
            )
    elif memory_pulse_binding is not None:
        raise PilotContinuationError(
            f"non-memory-pulse branch {treatment} recorded a memory pulse"
        )
    for memory in state.memories.values():
        memory.validate()
    if semantic_intervention.get("applied") and semantic_intervention.get(
        "rule_id"
    ):
        final_rule = next(
            (
                rule
                for rule in state.memories[focal_agent_id].semantic.rules
                if rule.rule_id == semantic_intervention["rule_id"]
            ),
            None,
        )
        if final_rule is None:
            raise PilotContinuationError(
                "injected erroneous rule disappeared from the semantic ledger"
            )
        semantic_intervention = {
            **semantic_intervention,
            "final_rule_status": final_rule.status,
            "final_rule_hash": canonical_hash(final_rule.to_dict()),
            "lifecycle_event_types": [
                event.event_type
                for event in state.memories[focal_agent_id].semantic.events
                if event.rule_id == final_rule.rule_id
            ],
        }
    metrics = _metrics(
        state=state,
        continuation_rows=continuation_ledger_rows,
        action_rows=branch_rows,
        focal_agent_id=focal_agent_id,
        initial_wealths=initial_wealths,
    )
    trajectory_payload = {
        "prefix_hash": state.prefix_hash,
        "rows": branch_rows,
        "final_memory_hashes": {
            str(agent_id): canonical_hash(memory.to_dict())
            for agent_id, memory in state.memories.items()
        },
        "final_ledger_hash": canonical_hash(state.ledger.records()),
    }
    provider_call_journal_binding = _journal_binding(
        journal_target,
        expected_completions=config.num_agents * horizon,
    )
    return (
        {
            "treatment": treatment,
            "intervention": {
                **semantic_intervention,
                "focal_agent_id": focal_agent_id,
                "decision_t": intervention_t,
                "pulse_only": treatment in MEMORY_PULSE_TREATMENTS,
                "memory_pulse_binding": memory_pulse_binding,
                "continuation_horizon_steps": horizon,
            },
            "narrative": {
                "narrative_id": narrative_id,
                "text": narrative_text,
                "text_hash": canonical_hash(narrative_text),
                "focal_agent_id": focal_agent_id,
                "decision_t": intervention_t,
                "pulse_only": narrative_id
                in NARRATIVE_PULSE_CONTRACT["treatment_narratives"],
                "continuation_horizon_steps": horizon,
            },
            "prefix_hash": state.prefix_hash,
            "checkpoint_hash": state.checkpoint_hash,
            "freeze_proposals": True,
            "shock_schedule_hash": canonical_hash(
                checkpoint.payload["run_config"].get("shock_schedule", [])
            ),
            "proposal_counters_before": {
                str(key): value for key, value in initial_proposals.items()
            },
            "proposal_counters_after": {
                str(key): value for key, value in state.proposals_made.items()
            },
            "rng_pre_step_hashes": [
                canonical_hash(value) for value in rng_schedule
            ],
            "api_usage": api_usage_rows,
            "api_usage_hash": canonical_hash(api_usage_rows),
            "provider_call_journal": provider_call_journal_binding,
            "trajectory": branch_rows,
            "trajectory_hash": canonical_hash(trajectory_payload),
            "metrics": metrics,
        },
        rng_schedule,
    )


@dataclass(frozen=True)
class PilotContinuationResult:
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(
                self.payload,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )


def run_pilot_continuations(
    checkpoint: PilotCheckpoint | Mapping[str, Any],
    *,
    llm: MultiModelLLM,
    budget: RunBudget,
    horizon: int = DEFAULT_CONTINUATION_HORIZON,
    focal_agent_id: int = 0,
    treatments: Sequence[str] = DEFAULT_TREATMENTS,
    strict_code_binding: bool = True,
    provider_call_journals: Mapping[str, Mapping[str, Any]] | None = None,
) -> PilotContinuationResult:
    """Run paired 4-agent Experiment-D continuations with frozen proposals."""

    if not isinstance(checkpoint, PilotCheckpoint):
        checkpoint = PilotCheckpoint.from_dict(checkpoint)
    if not isinstance(llm, MultiModelLLM):
        raise TypeError("llm must be MultiModelLLM")
    if not isinstance(budget, RunBudget):
        raise TypeError("budget must be RunBudget")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if horizon != DEFAULT_CONTINUATION_HORIZON:
        raise ValueError(
            "Experiment D preregistration fixes a six-period continuation"
        )
    if checkpoint.next_decision_t != 6:
        raise PilotContinuationError(
            "Experiment D must branch after t=5/before decision 6"
        )
    if checkpoint.payload["run_config"]["num_agents"] != 4:
        raise PilotContinuationError(
            "Experiment D requires exactly four continuing agents"
        )
    validate_preflight_p95_reservations(
        config_from_dict(checkpoint.payload["run_config"]),
        provider_model_name=llm.get_model_name(),
    )
    if not 0 <= focal_agent_id < 4:
        raise ValueError("focal_agent_id must identify one of four agents")
    if focal_agent_id != MEMORY_PULSE_CONTRACT["focal_agent_id"]:
        raise ValueError("Experiment D freezes focal_agent_id=0")
    normalized = tuple(str(value) for value in treatments)
    if normalized != DEFAULT_TREATMENTS:
        raise ValueError(
            "Experiment D requires the preregistered ordered treatment set"
        )
    if provider_call_journals is not None and set(provider_call_journals) != set(
        normalized
    ):
        raise ValueError(
            "continuation provider journals must cover every registered branch"
        )
    journal_paths = [
        str(Path(value["path"]).resolve())
        for value in (provider_call_journals or {}).values()
    ]
    if len(journal_paths) != len(set(journal_paths)):
        raise ValueError("continuation provider journal paths must be unique")

    branches: dict[str, Any] = {}
    common_rng_schedule, rng_schedule_binding = _pre_generate_rng_schedule(
        checkpoint, horizon=horizon
    )
    for treatment in normalized:
        branch, observed_schedule = _run_branch(
            checkpoint=checkpoint,
            llm=llm,
            budget=budget,
            treatment=treatment,
            horizon=horizon,
            focal_agent_id=focal_agent_id,
            common_rng_schedule=common_rng_schedule,
            strict_code_binding=strict_code_binding,
            provider_call_journal=(
                None
                if provider_call_journals is None
                else provider_call_journals[treatment]
            ),
        )
        if observed_schedule != common_rng_schedule:
            raise PilotContinuationError(
                f"pre-generated RNG schedule differs in branch {treatment}"
            )
        branches[treatment] = branch

    verified_start = branches["erroneous-verified"]["intervention"]
    unverified_start = branches["erroneous-unverified"]["intervention"]
    common_start_fields = (
        "rule_id",
        "forced_active_rule_hash",
        "forced_active_memory_hash",
        "forced_active_start_hash",
    )
    if any(
        verified_start[field] != unverified_start[field]
        for field in common_start_fields
    ):
        raise PilotContinuationError(
            "verified and unverified erroneous-rule arms do not share an "
            "exact forced-active start"
        )
    if (
        branches["erroneous-verified"]["trajectory"][0]["prompt_hashes"][
            str(focal_agent_id)
        ]
        != branches["erroneous-unverified"]["trajectory"][0][
            "prompt_hashes"
        ][str(focal_agent_id)]
    ):
        raise PilotContinuationError(
            "verifier assignment changed the intervention-period actor prompt"
        )

    if (
        branches["matched-a"]["trajectory_hash"]
        != branches["matched-b"]["trajectory_hash"]
    ):
        # This is an explicit diagnostic, not silently averaged away.
        matched_replay_equal = False
    else:
        matched_replay_equal = True
    baseline = branches["matched-a"]["metrics"]
    for branch in branches.values():
        branch["delta_vs_matched_a"] = {
            "focal_first_labor_hours": (
                branch["metrics"]["focal"]["first_step"]["labor_hours"]
                - baseline["focal"]["first_step"]["labor_hours"]
            ),
            "focal_first_consumption_rate": (
                branch["metrics"]["focal"]["first_step"][
                    "consumption_rate"
                ]
                - baseline["focal"]["first_step"]["consumption_rate"]
            ),
            "focal_immediate_flow_utility": (
                branch["metrics"]["focal"]["first_step"][
                    "immediate_flow_utility"
                ]
                - baseline["focal"]["first_step"]["immediate_flow_utility"]
            ),
            "focal_next_wealth": (
                branch["metrics"]["focal"]["first_step"]["next_wealth"]
                - baseline["focal"]["first_step"]["next_wealth"]
            ),
            "focal_final_wealth": (
                branch["metrics"]["focal"]["final_wealth"]
                - baseline["focal"]["final_wealth"]
            ),
            "focal_flow_utility_sum": (
                branch["metrics"]["focal"]["flow_utility_sum"]
                - baseline["focal"]["flow_utility_sum"]
            ),
            "focal_discounted_flow_utility_sum": (
                branch["metrics"]["focal"][
                    "discounted_flow_utility_sum"
                ]
                - baseline["focal"]["discounted_flow_utility_sum"]
            ),
            "population_first_step_flow_utility_sum": (
                branch["metrics"]["population"]["first_step"][
                    "flow_utility_sum"
                ]
                - baseline["population"]["first_step"]["flow_utility_sum"]
            ),
            "population_next_average_wealth": (
                branch["metrics"]["population"]["first_step"][
                    "average_next_wealth"
                ]
                - baseline["population"]["first_step"][
                    "average_next_wealth"
                ]
            ),
            "population_next_gini": (
                branch["metrics"]["population"]["first_step"][
                    "gini_next_wealth"
                ]
                - baseline["population"]["first_step"][
                    "gini_next_wealth"
                ]
            ),
            "population_next_low_labor_rate": (
                branch["metrics"]["population"]["first_step"][
                    "low_labor_rate"
                ]
                - baseline["population"]["first_step"][
                    "low_labor_rate"
                ]
            ),
            "population_average_final_wealth": (
                branch["metrics"]["population"]["average_final_wealth"]
                - baseline["population"]["average_final_wealth"]
            ),
            "population_final_gini": (
                branch["metrics"]["population"]["gini_final_wealth"]
                - baseline["population"]["gini_final_wealth"]
            ),
            "population_mean_low_labor_rate": (
                branch["metrics"]["population"]["mean_low_labor_rate"]
                - baseline["population"]["mean_low_labor_rate"]
            ),
            "population_flow_utility_sum": (
                branch["metrics"]["population"]["flow_utility_sum"]
                - baseline["population"]["flow_utility_sum"]
            ),
        }
    payload = {
        "schema_version": PILOT_CONTINUATION_SCHEMA_VERSION,
        "checkpoint_schema_version": checkpoint.payload["schema_version"],
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "prefix_hash": checkpoint.payload["prefix_hash"],
        "branch_after_decision_t": 5,
        "first_continuation_decision_t": 6,
        "horizon": horizon,
        "num_agents": 4,
        "focal_agent_id": focal_agent_id,
        "wrong_context_source_agent_id": 1,
        "memory_pulse_contract": dict(MEMORY_PULSE_CONTRACT),
        "action_grid": {
            "labor_step_hours": float(
                checkpoint.payload["run_config"]["labor_step"]
            ),
            "consumption_step": float(
                checkpoint.payload["run_config"]["consumption_step"]
            ),
        },
        "freeze_proposals": True,
        "shock_schedule_hash": canonical_hash(
            checkpoint.payload["run_config"].get("shock_schedule", [])
        ),
        "treatments": list(normalized),
        "pre_generated_rng_hashes": [
            canonical_hash(value) for value in common_rng_schedule
        ],
        "rng_schedule_binding": rng_schedule_binding,
        "erroneous_forced_active_common_start": {
            "equal": True,
            **{
                field: verified_start[field]
                for field in common_start_fields
            },
            "verifier_assignment_timing": "after-common-start",
        },
        "matched_replay_equal": matched_replay_equal,
        "branches": branches,
    }
    payload["result_hash"] = canonical_hash(payload)
    return PilotContinuationResult(payload)


def _narrative_branch_metrics(branch: Mapping[str, Any]) -> dict[str, float]:
    trajectory = branch["trajectory"]
    if not trajectory:
        raise PilotContinuationError("narrative branch has no continuation rows")
    first = trajectory[0]
    focal = str(branch["narrative"]["focal_agent_id"])
    decision = first["decisions"][focal]
    ledger = next(
        row for row in first["ledger_rows"] if str(row["agent_id"]) == focal
    )
    return {
        "first_labor_hours": float(decision["executed_labor_hours"]),
        "first_consumption_rate": float(
            decision["executed_consumption_rate"]
        ),
        "immediate_flow_utility": float(ledger["flow_utility"]),
        "six_step_discounted_flow_utility": float(
            branch["metrics"]["focal"]["discounted_flow_utility_sum"]
        ),
        "final_wealth": float(branch["metrics"]["focal"]["final_wealth"]),
    }


def run_pilot_narratives(
    checkpoint: PilotCheckpoint | Mapping[str, Any],
    *,
    llm: MultiModelLLM,
    budget: RunBudget,
    horizon: int = DEFAULT_CONTINUATION_HORIZON,
    focal_agent_id: int = 0,
    narratives: Mapping[str, str] = DEFAULT_NARRATIVES,
    strict_code_binding: bool = True,
    provider_call_journals: Mapping[str, Mapping[str, Any]] | None = None,
) -> PilotContinuationResult:
    """Run four content interventions from one hash-bound checkpoint."""

    if not isinstance(checkpoint, PilotCheckpoint):
        checkpoint = PilotCheckpoint.from_dict(checkpoint)
    if dict(narratives) != DEFAULT_NARRATIVES:
        raise ValueError("narrative fixtures differ from the preregistration")
    if focal_agent_id != NARRATIVE_PULSE_CONTRACT["focal_agent_id"]:
        raise ValueError("Experiment D freezes narrative focal_agent_id=0")
    if checkpoint.next_decision_t != 6:
        raise PilotContinuationError(
            "narrative branches must start before decision 6"
        )
    validate_preflight_p95_reservations(
        config_from_dict(checkpoint.payload["run_config"]),
        provider_model_name=llm.get_model_name(),
    )
    if provider_call_journals is not None and set(provider_call_journals) != set(
        DEFAULT_NARRATIVES
    ):
        raise ValueError(
            "narrative provider journals must cover every registered branch"
        )
    journal_paths = [
        str(Path(value["path"]).resolve())
        for value in (provider_call_journals or {}).values()
    ]
    if len(journal_paths) != len(set(journal_paths)):
        raise ValueError("narrative provider journal paths must be unique")
    branches: dict[str, Any] = {}
    common_rng_schedule, rng_schedule_binding = _pre_generate_rng_schedule(
        checkpoint, horizon=horizon
    )
    for narrative_id, narrative_text in DEFAULT_NARRATIVES.items():
        branch, observed_schedule = _run_branch(
            checkpoint=checkpoint,
            llm=llm,
            budget=budget,
            treatment=f"narrative-{narrative_id}",
            horizon=horizon,
            focal_agent_id=focal_agent_id,
            common_rng_schedule=common_rng_schedule,
            strict_code_binding=strict_code_binding,
            narrative_text=narrative_text,
            narrative_id=narrative_id,
            provider_call_journal=(
                None
                if provider_call_journals is None
                else provider_call_journals[narrative_id]
            ),
        )
        if observed_schedule != common_rng_schedule:
            raise PilotContinuationError(
                f"narrative RNG schedule differs in {narrative_id}"
            )
        branches[narrative_id] = branch

    metrics = {
        narrative_id: _narrative_branch_metrics(branch)
        for narrative_id, branch in branches.items()
    }

    def delta(left: str, right: str) -> dict[str, float]:
        return {
            field: metrics[left][field] - metrics[right][field]
            for field in metrics[left]
        }

    config = checkpoint.payload["run_config"]
    equivalence = {
        "labor_within_one_bin": abs(
            metrics["aligned"]["first_labor_hours"]
            - metrics["paraphrase"]["first_labor_hours"]
        )
        <= float(config["labor_step"]),
        "consumption_within_one_bin": abs(
            metrics["aligned"]["first_consumption_rate"]
            - metrics["paraphrase"]["first_consumption_rate"]
        )
        <= float(config["consumption_step"]),
    }
    payload = {
        "schema_version": PILOT_NARRATIVE_SCHEMA_VERSION,
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "prefix_hash": checkpoint.payload["prefix_hash"],
        "focal_agent_id": focal_agent_id,
        "horizon": horizon,
        "narrative_pulse_contract": dict(NARRATIVE_PULSE_CONTRACT),
        "shock_schedule_hash": canonical_hash(
            checkpoint.payload["run_config"].get("shock_schedule", [])
        ),
        "action_grid": {
            "labor_step_hours": float(config["labor_step"]),
            "consumption_step": float(config["consumption_step"]),
        },
        "fixtures": dict(DEFAULT_NARRATIVES),
        "fixture_hash": canonical_hash(DEFAULT_NARRATIVES),
        "pre_generated_rng_hashes": [
            canonical_hash(value) for value in common_rng_schedule
        ],
        "rng_schedule_binding": rng_schedule_binding,
        "branches": branches,
        "metrics": metrics,
        "aligned_vs_opposite_delta": delta("aligned", "opposite"),
        "aligned_vs_none_delta": delta("aligned", "none"),
        "paraphrase_vs_aligned_delta": delta("paraphrase", "aligned"),
        "semantic_equivalence_within_one_action_bin": {
            **equivalence,
            "pass": all(equivalence.values()),
        },
        "claim_boundary": (
            "controlled semantic response only; this is not evidence of real-news "
            "understanding"
        ),
    }
    payload["result_hash"] = canonical_hash(payload)
    return PilotContinuationResult(payload)


__all__ = [
    "DEFAULT_CONTINUATION_HORIZON",
    "DEFAULT_NARRATIVES",
    "DEFAULT_TREATMENTS",
    "MEMORY_PULSE_CONTRACT",
    "MEMORY_PULSE_TREATMENTS",
    "NARRATIVE_PULSE_CONTRACT",
    "PILOT_CONTINUATION_SCHEMA_VERSION",
    "PILOT_NARRATIVE_SCHEMA_VERSION",
    "PilotContinuationError",
    "PilotContinuationResult",
    "SHUFFLE_ALGORITHM",
    "run_pilot_continuations",
    "run_pilot_narratives",
]
