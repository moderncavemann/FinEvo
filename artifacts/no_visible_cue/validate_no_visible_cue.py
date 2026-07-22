#!/usr/bin/env python3
"""Static, matrix, and one-month smoke validation for cue visibility."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from run_emnlp_experiments import MODELS, _matrix, _simulate_cmd  # noqa: E402
from market_sentiment import MarketSentiment  # noqa: E402
from simulate import (  # noqa: E402
    _regime_cue_prompt,
    agent_decision,
    env_config,
    foundation,
)
from utils import prettify_document  # noqa: E402


class ProbeSentiment:
    def __init__(self) -> None:
        self.calls = 0

    def get_sentiment_prompt(self) -> str:
        self.calls += 1
        return "Market sentiment is pessimistic (index: -0.75)."


class ProbeMemory:
    def __init__(self) -> None:
        self.retrieval_states: list[dict[str, object]] = []

    def generate_memory_prompt(self, current_state, **kwargs):
        self.retrieval_states.append(dict(current_state))
        return "", {
            "retrieved_episode_ids": [],
            "retrieval_scores": [],
            "score_components": [],
            "selected_rule_ids": [],
        }


class StubLLM:
    def __init__(self) -> None:
        self.dialog_batches = []

    def get_multiple_completions(self, dialogs, **kwargs):
        self.dialog_batches.append(copy.deepcopy(dialogs))
        return [
            '{"reflection":"smoke","work":0.98,"consumption":0.20}'
            for _ in dialogs
        ], 0.0


def one_month_smoke(show_regime_cue: bool):
    np.random.seed(314159)
    config = copy.deepcopy(env_config)
    config["n_agents"] = 2
    config["episode_length"] = 1
    config["flatten_masks"] = False
    config["flatten_observations"] = False
    config["components"][0]["SimpleLabor"]["scale_obs"] = False
    config["components"][1]["PeriodicBracketTax"]["scale_obs"] = False
    config["components"][3]["SimpleSaving"]["scale_obs"] = False

    env = foundation.make_env_instance(**config)
    obs = env.reset()
    sentiment = MarketSentiment(n_agents=2)
    sentiment.reset()
    memories = {idx: ProbeMemory() for idx in range(2)}
    llm = StubLLM()
    dialog_queue = [deque(maxlen=3) for _ in range(2)]
    reflection_queue = [deque(maxlen=7) for _ in range(2)]

    with tempfile.TemporaryDirectory(prefix="finevo-cue-smoke-") as temp_dir:
        actions, errors, cost = agent_decision(
            env,
            obs,
            dialog_queue,
            reflection_queue,
            memories,
            sentiment,
            llm,
            str(Path(temp_dir) / "dialogs"),
            0,
            0.0,
            use_gap_fixes=True,
            use_sentiment=True,
            show_regime_cue=show_regime_cue,
            use_episodic=True,
            use_semantic=True,
            use_reflection=True,
            temperature=0.2,
            top_p=1.0,
            prompt_style="default",
            output_format="json",
            event_entry=None,
            seed=314159,
            model_display="stub",
            setting_label="finevo",
            variant_label="visible" if show_regime_cue else "hidden",
        )
        env.step(actions)

    assert errors == 0
    assert cost == 0.0
    assert len(llm.dialog_batches) == 1
    prompts = [dialog[-1]["content"] for dialog in llm.dialog_batches[0]]
    retrieval_states = [
        state for memory in memories.values() for state in memory.retrieval_states
    ]
    assert len(retrieval_states) == 2
    assert all("sentiment" in state for state in retrieval_states)
    return prompts, actions, retrieval_states


def main() -> None:
    specs = [
        spec
        for spec in _matrix()
        if spec.exp_id == "E9"
    ]
    assert {spec.variant for spec in specs} == {
        "visible-cue-control",
        "no-visible-cue",
    }
    by_variant = {spec.variant: spec for spec in specs}
    visible_spec = by_variant["visible-cue-control"]
    hidden_spec = by_variant["no-visible-cue"]
    for spec in specs:
        assert spec.model_key == "gpt52"
        assert spec.num_agents == 10
        assert spec.num_months == 24
        assert spec.gap_fixes is True
    assert visible_spec.flags == {"show_regime_cue": True}
    assert hidden_spec.flags == {"show_regime_cue": False}

    commands = {}
    for spec in specs:
        command = _simulate_cmd(spec, MODELS[spec.model_key], seed=13, workers=1)
        flag_index = command.index("--show_regime_cue")
        assert command[flag_index + 1] == (
            "True" if spec.variant == "visible-cue-control" else "False"
        )
        commands[spec.variant] = command

    probe = ProbeSentiment()
    assert _regime_cue_prompt(probe, use_sentiment=True, show_regime_cue=False) == ""
    assert probe.calls == 0, "hidden-cue path must not render or fetch cue text"
    visible = _regime_cue_prompt(probe, use_sentiment=True, show_regime_cue=True)
    assert "index: -0.75" in visible
    assert probe.calls == 1
    assert _regime_cue_prompt(probe, use_sentiment=False, show_regime_cue=True) == ""
    assert probe.calls == 1

    visible_prompts, visible_actions, visible_states = one_month_smoke(True)
    hidden_prompts, hidden_actions, hidden_states = one_month_smoke(False)
    assert visible_actions == hidden_actions
    assert visible_states == hidden_states
    neutral_cue = MarketSentiment(n_agents=2).get_sentiment_prompt()
    for visible_prompt, hidden_prompt in zip(visible_prompts, hidden_prompts):
        assert neutral_cue in visible_prompt
        assert neutral_cue not in hidden_prompt
        assert prettify_document(visible_prompt.replace(neutral_cue, "", 1)) == hidden_prompt

    print(
        json.dumps(
            {
                "evidence_scope": "historical_pre_p0_v1",
                "current_method_scientific_evidence": False,
                "method_implementation": "legacy_simulate_py_deterministic_template_memory",
                "status": "pass",
                "variants": sorted(by_variant),
                "model": MODELS[hidden_spec.model_key].display,
                "num_agents": hidden_spec.num_agents,
                "num_months": hidden_spec.num_months,
                "gap_fixes": hidden_spec.gap_fixes,
                "flags": {
                    variant: by_variant[variant].flags for variant in sorted(by_variant)
                },
                "sample_seed": 13,
                "dry_run_commands": commands,
                "one_month_smoke": {
                    "status": "pass",
                    "agents": 2,
                    "months": 1,
                    "same_actions": True,
                    "same_internal_retrieval_state": True,
                    "only_prompt_difference": "explicit regime cue text",
                },
                "model_api_called": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
