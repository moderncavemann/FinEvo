from __future__ import annotations

import json

from llm_providers import MultiModelLLM, StructuredCompletion

from verified_memory.budget import BudgetLimits, RunBudget, UsageRecord
from verified_memory.pilot_capability import (
    CAPABILITY_TASKSET_SHA256,
    build_capability_tasks,
    run_capability_gate,
)
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


class CapabilityFixtureProvider(ScriptedDiagnosticProvider):
    def get_structured_completion(self, messages, **kwargs):
        prompt = self._prompt(messages)
        task = next(task for task in build_capability_tasks() if task.prompt == prompt)
        if task.category == "rule-proposal":
            marker = "template; do not add an outcome criterion:\n"
            template = json.loads(
                prompt.split(marker, 1)[1].split("\nEvidence:\n", 1)[0]
            )
            template["rationale"] = "Both successful episodes support this rule."
            text = json.dumps(template, sort_keys=True)
        else:
            text = json.dumps({"choice": task.expected_choice})
        shape = super().get_structured_completion(
            [
                {
                    "role": "user",
                    "content": (
                        "monthly decision t=0 Return ONLY JSON with work and consumption"
                    ),
                }
            ],
            **kwargs,
        )
        return StructuredCompletion(
            text=text,
            usage=UsageRecord(prompt_tokens=10, completion_tokens=5),
            model=shape.model,
            provider=shape.provider,
            attempts=shape.attempts,
            latency_seconds=shape.latency_seconds,
            request_seed=kwargs.get("seed"),
            response_model=shape.response_model,
        )


def test_fixed_capability_taskset_shape_and_hash() -> None:
    tasks = build_capability_tasks()
    assert len(tasks) == 30
    assert [task.category for task in tasks].count("utility-ranking") == 12
    assert [task.category for task in tasks].count("rule-application") == 12
    assert [task.category for task in tasks].count("rule-proposal") == 6
    assert len(CAPABILITY_TASKSET_SHA256) == 64


def test_scripted_capability_fixture_passes_all_denominators() -> None:
    result = run_capability_gate(
        llm=MultiModelLLM(CapabilityFixtureProvider(), num_workers=4),
        budget=RunBudget(
            BudgetLimits(max_calls=30, max_cost_usd=0.01),
            budget_id="capability-test",
        ),
        seed=2010922376,
        estimate_usage=lambda prompt, max_tokens: UsageRecord(),
    )
    assert result["pass"] is True
    assert len(result["rows"]) == 30
    assert result["provider_failure_count"] == 0
    assert result["parse_failure_count"] == 0
    assert result["budget"]["completed_calls"] == 30
