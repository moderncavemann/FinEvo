"""Deterministic regime-switch fixture for EGRM implementation validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import load_contract
from .m2_episodic import EvidenceLinkedEpisodicTrack, EpisodeRecord
from .m3_semantic import VerifiedSemanticRuleTrack
from .metrics import compute_lifecycle_metrics


OUTPUT_SCHEMA_VERSION = "egrm-controlled-fixture-result-v1"


def _append_episode(
    track: EvidenceLinkedEpisodicTrack, row: Mapping[str, Any]
) -> EpisodeRecord:
    decision_t = int(row["decision_t"])
    interest_rate = float(row["interest_rate"])
    consumption = float(row["consumption_fraction"])
    flow_utility = float(row["flow_utility"])
    pre_state = {
        "interest_rate": interest_rate,
        "wealth": 1000.0 + decision_t,
        "price": 100.0,
    }
    decision_id = track.begin_episode(
        decision_t=decision_t,
        pre_state=pre_state,
        context_id=f"controlled-regime-{decision_t}",
        context_vector=(interest_rate, float(decision_t)),
        retrieved_episode_ids=(),
        selected_rule_ids=(),
        proposed_action={"consumption_fraction": consumption, "labor_hours": 100.0},
        executed_action={"consumption_fraction": consumption, "labor_hours": 100.0},
    )
    return track.finalize_episode(
        decision_id,
        outcome_t=decision_t + 1,
        next_state={**pre_state, "wealth": pre_state["wealth"] + flow_utility},
        outcome={"wealth_change": flow_utility},
        reward=flow_utility,
        flow_utility=flow_utility,
    )


def _candidate(episode_ids: list[str], rule: Mapping[str, Any], rationale: str) -> str:
    return json.dumps(
        {
            "context_scope": {"scope_id": "global", "predicates": []},
            "condition": {
                "field": "interest_rate",
                "operator": rule["condition_operator"],
                "value": rule["condition_value"],
                "tolerance": 1e-9,
            },
            "action_guidance": {
                "target": "consumption_fraction",
                "direction": rule["action_direction"],
                "threshold": rule["action_threshold"],
                "tolerance": 0.01,
            },
            "rationale": rationale,
            "supporting_episode_ids": episode_ids,
        },
        sort_keys=True,
    )


def _make_tracks(run_id: str, seed: int, lifecycle: Mapping[str, Any]):
    episodes = EvidenceLinkedEpisodicTrack(
        run_id=run_id, seed=seed, agent_id=0, prompt_capacity=20
    )
    semantic = VerifiedSemanticRuleTrack(
        episodes,
        min_candidate_support=int(lifecycle["min_candidate_support"]),
        activation_min_support=int(lifecycle["activation_min_support"]),
        activation_min_margin=float(lifecycle["activation_min_margin"]),
        activation_confidence_threshold=float(
            lifecycle["activation_confidence_threshold"]
        ),
        proposal_confidence_floor=float(lifecycle["proposal_confidence_floor"]),
        retirement_patience=int(lifecycle["retirement_patience"]),
        retirement_confidence_threshold=float(
            lifecycle["retirement_confidence_threshold"]
        ),
    )
    return episodes, semantic


def run_controlled_fixture(contract_path: str | Path) -> dict[str, Any]:
    contract, contract_hash = load_contract(contract_path)
    if contract["status"] != "implementation_fixture":
        raise ValueError(
            "controlled fixture runner accepts implementation fixtures only"
        )
    fixture = contract["fixture"]
    required_fixture_keys = {
        "lifecycle",
        "discovery_episodes",
        "post_proposal_episodes",
        "supported_rule",
        "unsupported_rule",
        "expected",
    }
    if set(fixture) != required_fixture_keys:
        raise ValueError("fixture keys are not exact")
    seed = int(contract["randomization"]["seeds"][0])

    episodes, semantic = _make_tracks("egrm-controlled", seed, fixture["lifecycle"])
    discovery = [
        _append_episode(episodes, row) for row in fixture["discovery_episodes"]
    ]
    support_ids = [episode.episode_id for episode in discovery]
    proposal_t = max(episode.outcome_t for episode in discovery)
    supported_raw = _candidate(
        support_ids,
        fixture["supported_rule"],
        "Registered finalized outcomes support this provisional rule.",
    )
    unsupported_raw = _candidate(
        support_ids,
        fixture["unsupported_rule"],
        "This deliberately unsupported candidate tests evidence admission.",
    )
    supported_initial = semantic.propose(supported_raw, current_t=proposal_t)
    unsupported = semantic.propose(unsupported_raw, current_t=proposal_t)

    lifecycle_statuses = [supported_initial.status]
    post = []
    for row in fixture["post_proposal_episodes"]:
        episode = _append_episode(episodes, row)
        post.append(episode)
        updated = semantic.observe_episode(
            supported_initial.rule_id,
            episode.episode_id,
            current_t=episode.outcome_t,
        )
        lifecycle_statuses.append(updated.status)
    semantic.validate_referential_integrity()

    final_supported = semantic.get(supported_initial.rule_id)
    if final_supported is None:
        raise RuntimeError("supported rule disappeared")
    verified_rules = [final_supported.to_dict(), unsupported.to_dict()]
    verified_events = [event.to_dict() for event in semantic.events]
    supported_labels = {
        final_supported.rule_id: True,
        unsupported.rule_id: False,
    }
    known_ids = {episode.episode_id for episode in episodes.finalized_episodes}
    verified_metrics = compute_lifecycle_metrics(
        rules=verified_rules,
        events=verified_events,
        supported_rule_ids=supported_labels,
        known_episode_ids=known_ids,
    )

    baseline_episodes, baseline = _make_tracks(
        "egrm-unverified-control", seed, fixture["lifecycle"]
    )
    baseline_discovery = [
        _append_episode(baseline_episodes, row) for row in fixture["discovery_episodes"]
    ]
    baseline_ids = [episode.episode_id for episode in baseline_discovery]
    baseline_supported = baseline.propose_unverified_immediate(
        _candidate(
            baseline_ids,
            fixture["supported_rule"],
            "Unverified control candidate.",
        ),
        current_t=proposal_t,
    )
    baseline_unsupported = baseline.propose_unverified_immediate(
        _candidate(
            baseline_ids,
            fixture["unsupported_rule"],
            "Unverified control unsupported candidate.",
        ),
        current_t=proposal_t,
    )
    for row in fixture["post_proposal_episodes"]:
        episode = _append_episode(baseline_episodes, row)
        for rule in (baseline_supported, baseline_unsupported):
            baseline.observe_episode(
                rule.rule_id, episode.episode_id, current_t=episode.outcome_t
            )
    baseline.validate_referential_integrity()
    baseline_rules = [
        baseline.get(baseline_supported.rule_id).to_dict(),
        baseline.get(baseline_unsupported.rule_id).to_dict(),
    ]
    baseline_events = [event.to_dict() for event in baseline.events]
    baseline_labels = {
        baseline_supported.rule_id: True,
        baseline_unsupported.rule_id: False,
    }
    baseline_metrics = compute_lifecycle_metrics(
        rules=baseline_rules,
        events=baseline_events,
        supported_rule_ids=baseline_labels,
        known_episode_ids={
            episode.episode_id for episode in baseline_episodes.finalized_episodes
        },
    )

    observed = {
        "supported_lifecycle": lifecycle_statuses,
        "unsupported_final_status": unsupported.status,
        "verified_unsupported_ever_active_count": verified_metrics[
            "unsupported_ever_active_count"
        ],
        "unverified_unsupported_ever_active_count": baseline_metrics[
            "unsupported_ever_active_count"
        ],
    }
    fixture_passed = observed == fixture["expected"]
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "study_id": contract["study_id"],
        "contract_hash": contract_hash,
        "scope": "deterministic_implementation_fixture",
        "scientific_evidence": False,
        "fixture_passed": fixture_passed,
        "observed": observed,
        "verified_policy_metrics": verified_metrics,
        "unverified_policy_metrics": baseline_metrics,
        "verified_rules": verified_rules,
        "verified_events": verified_events,
        "episode_ids": sorted(known_ids),
        "claim_boundary": contract["claim_boundary"],
    }


def write_result(result: Mapping[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
