from __future__ import annotations

from dataclasses import replace
import json
import threading

import pytest

from llm_providers import MultiModelLLM, StructuredCompletion
from verified_memory.budget import BudgetLimits, RunBudget, UsageRecord
from verified_memory.pilot_capability import (
    CAPABILITY_SCHEMA_VERSION,
    CAPABILITY_TASKSET_SHA256,
    CapabilityTask,
    build_capability_tasks,
    run_capability_gate,
)
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


def _proposal_response(task: CapabilityTask) -> str:
    assert task.expected_scope is not None
    assert task.expected_condition is not None
    assert task.expected_guidance is not None
    return json.dumps(
        {
            "context_scope": dict(task.expected_scope),
            "condition": dict(task.expected_condition),
            "action_guidance": dict(task.expected_guidance),
            "rationale": "The cited finalized M2 episodes support this rule.",
            "supporting_episode_ids": list(task.allowed_episode_ids[:2]),
        },
        sort_keys=True,
    )


def _compliant_action(task: CapabilityTask) -> str:
    work = 0.50
    consumption = 0.40
    if task.expected_guidance is not None:
        guidance = task.expected_guidance
        target = guidance["target"]
        direction = guidance["direction"]
        threshold = float(guidance["threshold"])
        if target == "consumption_fraction":
            if direction == "at_most":
                consumption = max(0.0, threshold - 0.02)
            elif direction == "at_least":
                consumption = min(1.0, threshold + 0.02)
            else:
                consumption = threshold
        elif target == "labor_hours":
            desired = threshold
            if direction == "at_least":
                desired = min(168.0, threshold + 8.0)
            work = desired / 168.0
    return json.dumps(
        {
            "reflection": "Use the current state and verified evidence.",
            "work": work,
            "consumption": consumption,
        },
        sort_keys=True,
    )


def _ideal_response(task: CapabilityTask) -> str:
    if task.task_kind == "rule_proposal":
        return _proposal_response(task)
    return _compliant_action(task)


class CapabilityFixtureProvider(ScriptedDiagnosticProvider):
    def __init__(self) -> None:
        super().__init__()
        self.dispatched: list[tuple[str, int]] = []
        self._lock = threading.Lock()

    def mutate(
        self,
        task: CapabilityTask,
        completion: StructuredCompletion,
    ) -> StructuredCompletion:
        return completion

    def get_structured_completion(self, messages, **kwargs):
        prompt = self._prompt(messages)
        task = next(
            task for task in build_capability_tasks() if task.prompt == prompt
        )
        with self._lock:
            self.dispatched.append((task.task_kind, kwargs["max_tokens"]))
        shape = super().get_structured_completion(
            [
                {
                    "role": "user",
                    "content": (
                        "monthly decision t=0 Return ONLY JSON with work and "
                        "consumption"
                    ),
                }
            ],
            **kwargs,
        )
        completion = StructuredCompletion(
            text=_ideal_response(task),
            usage=UsageRecord(
                prompt_tokens=120,
                completion_tokens=24,
                cost_usd=0.0001,
            ),
            model=shape.model,
            provider=shape.provider,
            attempts=shape.attempts,
            latency_seconds=shape.latency_seconds,
            request_seed=kwargs.get("seed"),
            response_model=shape.response_model,
            response_provider=shape.response_provider,
            response_route=shape.response_route,
            reasoning_tokens=4,
            finish_reason="stop",
            native_finish_reason="stop",
            response_completed=True,
        )
        return self.mutate(task, completion)


class RecoveryProvider(CapabilityFixtureProvider):
    def mutate(self, task, completion):
        if task.task_id == "action-01":
            return replace(
                completion,
                text=f"```json\n{completion.text}\n```",
            )
        if task.task_id == "action-02":
            return replace(
                completion,
                text=f"Here is the requested object: {completion.text}",
            )
        return completion


class WrongSemanticEvidenceProvider(CapabilityFixtureProvider):
    def __init__(self, mutation: str = "episode_id") -> None:
        super().__init__()
        self.mutation = mutation

    def mutate(self, task, completion):
        if task.task_id != "proposal-01":
            return completion
        payload = json.loads(completion.text)
        if self.mutation == "episode_id":
            assert task.proposal_track is not None
            all_ids = [
                row["episode_id"] for row in task.proposal_track["episodes"]
            ]
            wrong_id = next(
                episode_id
                for episode_id in all_ids
                if episode_id not in task.allowed_episode_ids
            )
            payload["supporting_episode_ids"] = [
                task.allowed_episode_ids[0],
                wrong_id,
            ]
        elif self.mutation == "condition":
            payload["condition"]["value"] = 0.03
        elif self.mutation == "guidance":
            payload["action_guidance"]["threshold"] = 0.31
        elif self.mutation == "scope":
            payload["context_scope"] = {
                "scope_id": "price-positive",
                "predicates": [
                    {
                        "field": "price",
                        "operator": ">=",
                        "value": 0.5,
                        "tolerance": 0.0,
                    }
                ],
            }
        else:
            raise AssertionError(f"unknown test mutation {self.mutation!r}")
        return replace(completion, text=json.dumps(payload, sort_keys=True))


class EquivalentConditionToleranceProvider(CapabilityFixtureProvider):
    """Use a verifier-equivalent inequality tolerance, not the hidden literal."""

    def mutate(self, task, completion):
        if task.task_kind != "rule_proposal":
            return completion
        assert task.expected_condition is not None
        assert task.expected_condition["operator"] in {">=", "<="}
        assert task.expected_condition["tolerance"] == pytest.approx(1e-9)
        payload = json.loads(completion.text)
        payload["condition"]["tolerance"] = 0.0
        return replace(completion, text=json.dumps(payload, sort_keys=True))


class GeminiLengthProvider(CapabilityFixtureProvider):
    def mutate(self, task, completion):
        if task.task_id != "action-01":
            return completion
        return replace(
            completion,
            text='{"reflection":"unfinished","work":0.5',
            usage=UsageRecord(
                prompt_tokens=500,
                completion_tokens=128,
                cost_usd=0.001,
            ),
            reasoning_tokens=120,
            error_type="IncompleteCompletionError",
            finish_reason="length",
            native_finish_reason="length",
            response_completed=False,
        )


class OversizeVisibleJSONProvider(CapabilityFixtureProvider):
    def mutate(self, task, completion):
        if task.task_id != "action-01":
            return completion
        payload = json.loads(completion.text)
        payload["reflection"] = "x" * 500
        return replace(completion, text=json.dumps(payload, sort_keys=True))


class ProviderFailureProvider(CapabilityFixtureProvider):
    def mutate(self, task, completion):
        if task.task_id != "action-01":
            return completion
        return replace(
            completion,
            text="Error",
            error_type="ProviderUnavailableError",
            finish_reason="error",
            native_finish_reason="error",
            response_completed=False,
        )


def _contracts(
    *,
    actor_tokens: int = 512,
    proposal_tokens: int = 1200,
    actor_bytes: int = 4096,
    proposal_bytes: int = 8192,
) -> dict[str, dict[str, object]]:
    return {
        "actor-action": {
            "request_max_completion_tokens": actor_tokens,
            "visible_json_max_bytes": actor_bytes,
            "accepted_parse_modes": ["exact_json"],
            "required_finish_reason": "stop",
        },
        "semantic-proposal": {
            "request_max_completion_tokens": proposal_tokens,
            "visible_json_max_bytes": proposal_bytes,
            "accepted_parse_modes": ["exact_json"],
            "required_finish_reason": "stop",
        },
    }


def _run_gate(
    provider: CapabilityFixtureProvider,
    *,
    budget_id: str,
    contracts: dict[str, dict[str, object]] | None = None,
):
    return run_capability_gate(
        llm=MultiModelLLM(provider, num_workers=4),
        budget=RunBudget(
            BudgetLimits(
                max_calls=30,
                max_completion_tokens=100_000,
                max_cost_usd=0.1,
            ),
            budget_id=budget_id,
        ),
        seed=2010922376,
        estimate_usage=lambda prompt, max_tokens: UsageRecord(),
        task_output_contracts=contracts,
    )


def test_v4_taskset_uses_production_prompts_and_real_evidence() -> None:
    tasks = build_capability_tasks()

    assert len(tasks) == 30
    assert [task.category for task in tasks].count("utility-ranking") == 12
    assert [task.category for task in tasks].count("rule-application") == 12
    assert [task.category for task in tasks].count("rule-proposal") == 6
    assert [task.task_kind for task in tasks].count("action_generation") == 12
    assert [task.task_kind for task in tasks].count("rule_application") == 12
    assert [task.task_kind for task in tasks].count("rule_proposal") == 6
    assert CAPABILITY_SCHEMA_VERSION == "finevo-capability-gate-v4"
    assert len(CAPABILITY_TASKSET_SHA256) == 64

    action = tasks[0]
    assert "verified-decision-prompt" not in action.prompt
    assert "q0=1" in action.prompt
    assert "<<<VERIFIED_MEMORY_START>>>" in action.prompt
    assert "Return ONLY JSON with keys reflection" in action.prompt

    rule = tasks[12]
    assert rule.output_contract_id == "actor-action"
    assert "Finalized experience evidence:" in rule.prompt
    assert "Verified active rules:" in rule.prompt
    assert len(rule.prompt) > 2_000

    proposal = tasks[24]
    assert proposal.output_contract_id == "semantic-proposal"
    assert "Allowed JSON schema" in proposal.prompt
    assert proposal.proposal_track is not None
    assert len(proposal.proposal_track["episodes"]) == 6
    assert len(proposal.allowed_episode_ids) == 3


def test_scripted_production_capability_fixture_passes_30_call_gate() -> None:
    provider = CapabilityFixtureProvider()
    result = _run_gate(
        provider,
        budget_id="capability-v4-pass",
        contracts=_contracts(actor_tokens=333, proposal_tokens=777),
    )

    assert result["schema_version"] == "finevo-capability-gate-v4"
    assert result["pass"] is True
    assert result["interface_gate"] == {"pass": True, "failure_count": 0}
    assert result["strict_parse_count"] == 30
    assert result["recovered_parse_count"] == 0
    assert result["parse_failure_count"] == 0
    assert result["truncation_count"] == 0
    assert len(result["rows"]) == 30
    assert result["budget"]["completed_calls"] == 30
    assert all(row["correct"] for row in result["rows"])
    assert all(row["interface_valid"] for row in result["rows"])
    assert all(row["strict_parse"] for row in result["rows"])
    assert all(row["parse_mode"] == "exact_json" for row in result["rows"])
    assert all(row["reasoning_tokens"] == 4 for row in result["rows"])
    assert all(row["visible_completion_tokens"] == 20 for row in result["rows"])
    assert all(row["cost_usd"] == pytest.approx(0.0001) for row in result["rows"])
    assert [cap for kind, cap in provider.dispatched if kind != "rule_proposal"] == [
        333
    ] * 24
    assert [cap for kind, cap in provider.dispatched if kind == "rule_proposal"] == [
        777
    ] * 6


def test_fenced_and_substring_recovery_are_itt_but_not_strict_success() -> None:
    result = _run_gate(
        RecoveryProvider(),
        budget_id="capability-v4-recovery",
        contracts=_contracts(),
    )
    rows = {row["task_id"]: row for row in result["rows"]}

    assert rows["action-01"]["parse_mode"] == "fenced_recovery"
    assert rows["action-02"]["parse_mode"] == "substring_recovery"
    for task_id in ("action-01", "action-02"):
        assert rows[task_id]["interface_valid"] is True
        assert rows[task_id]["interface_status"] == "recovered_parse"
        assert rows[task_id]["strict_parse"] is False
        assert rows[task_id]["correct"] is False
        assert rows[task_id]["evaluable"] is True
    assert result["recovered_parse_count"] == 2
    assert result["category_totals"]["utility-ranking"]["correct"] == 10
    assert result["pass"] is True


def test_verifier_rejected_evidence_remains_illegal() -> None:
    result = _run_gate(
        WrongSemanticEvidenceProvider("episode_id"),
        budget_id="capability-v4-rejected-evidence",
        contracts=_contracts(),
    )
    row = next(row for row in result["rows"] if row["task_id"] == "proposal-01")

    assert row["parse_mode"] == "exact_json"
    assert row["strict_parse"] is True
    assert row["semantic_candidate_accepted"] is False
    assert row["candidate_status"] == "rejected"
    assert row["semantic_match"] is True
    assert row["legal"] is False
    assert row["correct"] is False
    assert result["category_totals"]["rule-proposal"]["correct"] == 5
    assert result["pass"] is True


@pytest.mark.parametrize("mutation", ("condition", "guidance", "scope"))
def test_hidden_semantic_mismatch_does_not_override_verifier_admission(
    mutation: str,
) -> None:
    result = _run_gate(
        WrongSemanticEvidenceProvider(mutation),
        budget_id=f"capability-v4-wrong-{mutation}",
        contracts=_contracts(),
    )
    row = next(row for row in result["rows"] if row["task_id"] == "proposal-01")

    assert row["parse_mode"] == "exact_json"
    assert row["strict_parse"] is True
    assert row["semantic_match"] is False
    assert row["semantic_candidate_accepted"] is True
    assert row["candidate_status"] == "provisional"
    assert row["legal"] is True
    assert row["correct"] is True
    assert result["category_totals"]["rule-proposal"]["correct"] == 6
    assert result["pass"] is True


def test_semantically_equivalent_condition_tolerance_is_legal() -> None:
    result = _run_gate(
        EquivalentConditionToleranceProvider(),
        budget_id="capability-v4-equivalent-tolerance",
        contracts=_contracts(),
    )
    rows = [row for row in result["rows"] if row["task_kind"] == "rule_proposal"]

    assert len(rows) == 6
    assert all(row["interface_valid"] for row in rows)
    assert all(row["strict_parse"] for row in rows)
    assert all(row["semantic_candidate_accepted"] for row in rows)
    assert all(row["candidate_status"] == "provisional" for row in rows)
    assert all(row["semantic_match"] is False for row in rows)
    assert all(row["legal"] for row in rows)
    assert all(row["correct"] for row in rows)
    assert result["category_totals"]["rule-proposal"]["correct"] == 6
    assert result["pass"] is True


def test_hidden_reasoning_and_length_are_recorded_as_truncation() -> None:
    result = _run_gate(
        GeminiLengthProvider(),
        budget_id="capability-v4-gemini-length",
        contracts=_contracts(),
    )
    row = next(row for row in result["rows"] if row["task_id"] == "action-01")

    assert row["reasoning_tokens"] == 120
    assert row["usage"]["completion_tokens"] == 128
    assert row["visible_completion_tokens"] == 8
    assert row["truncation"] is True
    assert row["finish_reason"] == "length"
    assert row["native_finish_reason"] == "length"
    assert row["interface_status"] == "incomplete"
    assert row["interface_valid"] is False
    assert row["evaluable"] is False
    assert result["interface_gate"] == {"pass": False, "failure_count": 1}
    assert result["capability_assessment"]["status"] == "not_evaluable"
    assert result["pass"] is False


def test_visible_json_byte_cap_is_fail_closed() -> None:
    result = _run_gate(
        OversizeVisibleJSONProvider(),
        budget_id="capability-v4-visible-limit",
        contracts=_contracts(actor_bytes=256),
    )
    row = next(row for row in result["rows"] if row["task_id"] == "action-01")

    assert row["output_bytes"] > row["visible_json_max_bytes"]
    assert row["within_visible_limit"] is False
    assert row["interface_status"] == "visible_limit_exceeded"
    assert row["interface_valid"] is False
    assert row["strict_parse"] is True
    assert row["correct"] is False
    assert result["capability_assessment"]["status"] == "not_evaluable"


def test_provider_failure_stays_in_registered_denominator() -> None:
    result = _run_gate(
        ProviderFailureProvider(),
        budget_id="capability-v4-provider-failure",
        contracts=_contracts(),
    )
    row = next(row for row in result["rows"] if row["task_id"] == "action-01")

    assert row["interface_status"] == "provider_error"
    assert row["interface_valid"] is False
    assert row["evaluable"] is False
    assert row["correct"] is False
    assert result["provider_failure_count"] == 1
    totals = result["category_totals"]["utility-ranking"]
    assert totals["registered_total"] == 12
    assert totals["evaluable_count"] == 11
    assert totals["interface_failure_count"] == 1
    assert result["pass"] is False


def test_v2_compatibility_defaults_and_caps_override_remain_bounded() -> None:
    provider = CapabilityFixtureProvider()
    result = _run_gate(
        provider,
        budget_id="capability-v4-defaults",
        contracts=None,
    )
    assert result["pass"] is True
    assert result["task_output_contracts"]["actor-action"][
        "request_max_completion_tokens"
    ] == 512

    provider = CapabilityFixtureProvider()
    result = run_capability_gate(
        llm=MultiModelLLM(provider, num_workers=4),
        budget=RunBudget(
            BudgetLimits(max_calls=30, max_cost_usd=0.1),
            budget_id="capability-v4-caps-alias",
        ),
        seed=2010922376,
        estimate_usage=lambda prompt, max_tokens: UsageRecord(),
        caps={"action_generation": 222, "rule_proposal": 666},
    )
    assert result["pass"] is True
    assert result["task_output_contracts"]["actor-action"][
        "request_max_completion_tokens"
    ] == 222
    assert result["task_output_contracts"]["semantic-proposal"][
        "request_max_completion_tokens"
    ] == 666


def test_explicit_v2_contract_requires_both_scientific_call_kinds() -> None:
    with pytest.raises(ValueError, match="semantic-proposal"):
        _run_gate(
            CapabilityFixtureProvider(),
            budget_id="capability-v4-missing-contract",
            contracts={"actor-action": _contracts()["actor-action"]},
        )


def test_pilot_contract_v2_field_names_are_normalized() -> None:
    contracts = {}
    for task_id, max_tokens, max_bytes in (
        ("actor-action", 2048, 1024),
        ("semantic-proposal", 4096, 4096),
    ):
        contracts[task_id] = {
            "task_id": task_id,
            "max_completion_tokens": max_tokens,
            "max_visible_json_bytes": max_bytes,
            "visible_token_count_required": True,
            "reasoning_token_count_required": True,
            "science_parse_mode": "exact_json_only",
            "report_recovery_modes": ["fenced_json", "substring_json"],
            "recovered_output_scientific_success": False,
            "required_finish_reason": "stop",
        }
    provider = CapabilityFixtureProvider()
    result = _run_gate(
        provider,
        budget_id="capability-v4-contract-field-aliases",
        contracts=contracts,
    )

    assert result["pass"] is True
    assert result["task_output_contracts"]["actor-action"][
        "request_max_completion_tokens"
    ] == 2048
    assert result["task_output_contracts"]["actor-action"][
        "visible_json_max_bytes"
    ] == 1024
