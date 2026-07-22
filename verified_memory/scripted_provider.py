"""Deterministic no-network provider for integration diagnostics only.

Outputs from this provider are synthetic fixtures.  They validate plumbing,
temporal alignment, semantic-rule lifecycle, and artifact completeness; they must
never be reported as model performance or scientific evidence.
"""

from __future__ import annotations

import json
import re
from time import monotonic
from typing import Any, Mapping

from llm_providers import LLMProvider, StructuredCompletion
from .budget import UsageRecord


DIAGNOSTIC_PROVIDER_NAME = "diagnostic"
DIAGNOSTIC_MODEL_NAME = "scripted-v1"


class ScriptedDiagnosticProvider(LLMProvider):
    """Return valid actions and evidence-linked proposals without network calls."""

    _WORK_SCHEDULE = (0.25, 0.50, 0.75, 0.50)
    _CONSUMPTION_SCHEDULE = (0.30, 0.35, 0.30, 0.25)

    def get_model_name(self) -> str:
        return f"{DIAGNOSTIC_PROVIDER_NAME}/{DIAGNOSTIC_MODEL_NAME}"

    @staticmethod
    def _prompt(messages: list[dict[str, Any]]) -> str:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must contain at least one entry")
        content = messages[-1].get("content")
        if not isinstance(content, str):
            raise TypeError("last message content must be a string")
        return content

    @staticmethod
    def _proposal(prompt: str) -> str:
        marker = "Evidence:\n"
        if marker not in prompt:
            raise ValueError("semantic proposal prompt is missing evidence")
        evidence = json.loads(prompt.split(marker, 1)[1])
        if not isinstance(evidence, list) or len(evidence) < 2:
            raise ValueError("diagnostic proposal requires two finalized episodes")
        episode_ids = [str(item["episode_id"]) for item in evidence[:2]]
        payload: Mapping[str, Any] = {
            "condition": {
                "field": "wealth",
                "operator": ">=",
                "value": 0.0,
                "tolerance": 0.0,
            },
            "action_guidance": {
                "target": "labor_hours",
                "direction": "maintain",
                "threshold": 84.0,
                "tolerance": 84.0,
            },
            "outcome_criterion": {
                "metric": "flow_utility",
                "operator": ">=",
                "value": -1000.0,
                "tolerance": 0.0,
            },
            "rationale": "Synthetic lifecycle fixture grounded in the listed episodes.",
            "supporting_episode_ids": episode_ids,
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def _action(cls, prompt: str) -> str:
        match = re.search(r"monthly decision t=(\d+)", prompt)
        if not match:
            raise ValueError("decision prompt is missing its timestamp")
        decision_t = int(match.group(1))
        work = cls._WORK_SCHEDULE[decision_t % len(cls._WORK_SCHEDULE)]
        consumption = cls._CONSUMPTION_SCHEDULE[
            decision_t % len(cls._CONSUMPTION_SCHEDULE)
        ]
        return json.dumps(
            {
                "reflection": "Synthetic diagnostic action; not empirical evidence.",
                "work": work,
                "consumption": consumption,
            },
            sort_keys=True,
        )

    def get_structured_completion(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
        max_retries: int | None = None,
        seed: int | None = None,
    ) -> StructuredCompletion:
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an int or None")
        started = monotonic()
        prompt = self._prompt(messages)
        if "Propose one semantic decision rule" in prompt:
            text = self._proposal(prompt)
        else:
            text = self._action(prompt)
        return StructuredCompletion(
            text=text,
            usage=UsageRecord(
                prompt_tokens=max(1, len(prompt) // 4),
                completion_tokens=max(1, len(text) // 4),
                cost_usd=0.0,
            ),
            model=DIAGNOSTIC_MODEL_NAME,
            provider=DIAGNOSTIC_PROVIDER_NAME,
            attempts=1,
            latency_seconds=monotonic() - started,
            request_seed=seed,
            response_model=DIAGNOSTIC_MODEL_NAME,
        )

    def get_completion(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
    ) -> tuple[str, float]:
        result = self.get_structured_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return result.text, 0.0


__all__ = [
    "DIAGNOSTIC_MODEL_NAME",
    "DIAGNOSTIC_PROVIDER_NAME",
    "ScriptedDiagnosticProvider",
]
