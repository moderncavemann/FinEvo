from __future__ import annotations

from types import SimpleNamespace

from verified_memory.pilot_sensitivity import (
    ALTERNATIVE_SUCCESS_WEIGHTS,
    OUTCOME_DEFINITIONS,
    _replay_one,
)


def test_sensitivity_grid_is_frozen_to_nine_cells() -> None:
    assert ALTERNATIVE_SUCCESS_WEIGHTS == (0.25, 0.50, 0.75)
    assert OUTCOME_DEFINITIONS == (
        "utility_advantage_positive",
        "absolute_flow_utility",
        "three_period_cumulative_advantage_positive",
    )
    assert len(ALTERNATIVE_SUCCESS_WEIGHTS) * len(OUTCOME_DEFINITIONS) == 9


class _AlwaysMatches:
    def matches(self, value):
        return True


class _AlwaysCompliant:
    def is_consistent(self, value):
        return True


def test_sensitivity_retirement_uses_runtime_or_semantics() -> None:
    episodes = [
        SimpleNamespace(
            episode_id="support-a",
            agent_id=0,
            outcome_t=2,
            utility_advantage=1.0,
            flow_utility=1.0,
            pre_state={},
            executed_action={},
        ),
        SimpleNamespace(
            episode_id="harm-a",
            agent_id=0,
            outcome_t=3,
            utility_advantage=-1.0,
            flow_utility=-1.0,
            pre_state={},
            executed_action={},
        ),
    ]
    rule = SimpleNamespace(
        rule_id="r1",
        created_at=1,
        supporting_episode_ids=("support-a",),
        context_scope=_AlwaysMatches(),
        condition=_AlwaysMatches(),
        action_guidance=_AlwaysCompliant(),
    )
    result = _replay_one(
        rule=rule,
        episodes=episodes,
        definition="utility_advantage_positive",
        absolute_flow_threshold=0.0,
        alternative_success_weight=0.5,
        verifier={
            "min_candidate_support": 1,
            "activation_min_support": 1,
            "activation_min_margin": 0.0,
            "activation_confidence_threshold": 0.5,
            "retirement_patience": 1,
            # Confidence cannot independently retire, so patience must.
            "retirement_confidence_threshold": 0.0,
        },
        rolling_advantage={(0, 2): 1.0, (0, 3): 0.0},
    )
    assert result["activation_t"] == 2
    assert result["retirement_t"] == 3
    assert result["harmful_to_retirement_delay"] == 0
    assert result["active_exposure_steps"] == 1
    assert result["timeline"][0]["active_before_outcome"] is False
    assert result["timeline"][1]["active_before_outcome"] is True


def test_activation_outcome_is_not_counted_as_actor_exposure() -> None:
    episodes = [
        SimpleNamespace(
            episode_id="support-a",
            agent_id=0,
            outcome_t=2,
            utility_advantage=1.0,
            flow_utility=1.0,
            pre_state={},
            executed_action={},
        ),
    ]
    rule = SimpleNamespace(
        rule_id="r1",
        created_at=1,
        supporting_episode_ids=("support-a",),
        context_scope=_AlwaysMatches(),
        condition=_AlwaysMatches(),
        action_guidance=_AlwaysCompliant(),
    )
    result = _replay_one(
        rule=rule,
        episodes=episodes,
        definition="utility_advantage_positive",
        absolute_flow_threshold=0.0,
        alternative_success_weight=0.5,
        verifier={
            "min_candidate_support": 1,
            "activation_min_support": 1,
            "activation_min_margin": 0.0,
            "activation_confidence_threshold": 0.5,
            "retirement_patience": 2,
            "retirement_confidence_threshold": 0.0,
        },
        rolling_advantage={(0, 2): 1.0},
    )
    assert result["activation_t"] == 2
    assert result["active_exposure_steps"] == 0
    assert result["timeline"][0]["active"] is True
