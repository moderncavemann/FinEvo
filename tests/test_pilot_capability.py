from __future__ import annotations

from dataclasses import replace
import json

import pytest

from llm_providers import (
    MultiModelLLM,
    ProviderErrorDetails,
    StructuredCompletion,
)

from verified_memory.budget import BudgetLimits, RunBudget, UsageRecord
from verified_memory.pilot_capability import (
    CAPABILITY_SCHEMA_VERSION,
    CAPABILITY_TASKSET_SHA256,
    build_capability_tasks,
    run_capability_gate,
)
from verified_memory.pilot_evidence import (
    PilotEvidenceError,
    _validate_capability_v2,
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
            finish_reason="stop",
            native_finish_reason="stop",
            response_completed=True,
        )


class MixedInterfaceCapabilityProvider(CapabilityFixtureProvider):
    def get_structured_completion(self, messages, **kwargs):
        prompt = self._prompt(messages)
        task = next(task for task in build_capability_tasks() if task.prompt == prompt)
        completion = super().get_structured_completion(messages, **kwargs)
        if task.task_id == "utility-01":
            return replace(
                completion,
                text="Error",
                error_type="BadRequestError",
                provider_error_details=ProviderErrorDetails(
                    error_type="BadRequestError",
                    stage="openai.chat.completions.create",
                    sdk_name="openai-python",
                    sdk_version="test",
                    http_status=400,
                    code="unsupported_parameter",
                    param="temperature",
                    request_id="req_test_1",
                ),
                finish_reason="error",
                native_finish_reason="error",
                response_completed=False,
            )
        if task.task_id == "utility-02":
            return replace(completion, text="not JSON")
        if task.task_id == "utility-03":
            return replace(
                completion,
                text="unfinished output",
                error_type="IncompleteCompletionError",
                provider_error_details=ProviderErrorDetails(
                    error_type="IncompleteCompletionError",
                    stage="provider.response_contract",
                    sdk_name="test-provider",
                    sdk_version="1",
                ),
                finish_reason="length",
                native_finish_reason="length",
                response_completed=False,
            )
        if task.task_id == "utility-04":
            wrong = "B" if task.expected_choice == "A" else "A"
            return replace(completion, text=json.dumps({"choice": wrong}))
        return completion


class ProviderFailureCapabilityProvider(CapabilityFixtureProvider):
    def get_structured_completion(self, messages, **kwargs):
        completion = super().get_structured_completion(messages, **kwargs)
        return replace(
            completion,
            text="Error",
            error_type="ProviderUnavailableError",
            provider_error_details=ProviderErrorDetails(
                error_type="ProviderUnavailableError",
                stage="provider.dispatch",
                sdk_name="test-provider",
                sdk_version="1",
            ),
            finish_reason="error",
            native_finish_reason="error",
            response_completed=False,
        )


class MissingFinishCapabilityProvider(CapabilityFixtureProvider):
    def get_structured_completion(self, messages, **kwargs):
        prompt = self._prompt(messages)
        task = next(task for task in build_capability_tasks() if task.prompt == prompt)
        completion = super().get_structured_completion(messages, **kwargs)
        if task.task_id == "utility-01":
            return replace(
                completion,
                finish_reason=None,
                native_finish_reason=None,
                response_completed=None,
            )
        return completion


class SingleParseFailureCapabilityProvider(CapabilityFixtureProvider):
    def get_structured_completion(self, messages, **kwargs):
        prompt = self._prompt(messages)
        task = next(task for task in build_capability_tasks() if task.prompt == prompt)
        completion = super().get_structured_completion(messages, **kwargs)
        if task.task_id == "utility-01":
            return replace(completion, text="not JSON")
        return completion


def _run_gate(provider, *, budget_id: str):
    return run_capability_gate(
        llm=MultiModelLLM(provider, num_workers=4),
        budget=RunBudget(
            BudgetLimits(max_calls=30, max_cost_usd=0.01),
            budget_id=budget_id,
        ),
        seed=2010922376,
        estimate_usage=lambda prompt, max_tokens: UsageRecord(),
    )


def test_fixed_capability_taskset_shape_and_hash() -> None:
    tasks = build_capability_tasks()
    assert len(tasks) == 30
    assert [task.category for task in tasks].count("utility-ranking") == 12
    assert [task.category for task in tasks].count("rule-application") == 12
    assert [task.category for task in tasks].count("rule-proposal") == 6
    assert len(CAPABILITY_TASKSET_SHA256) == 64
    assert CAPABILITY_SCHEMA_VERSION == "finevo-capability-gate-v2"


def test_scripted_capability_fixture_passes_all_denominators() -> None:
    result = _run_gate(
        CapabilityFixtureProvider(),
        budget_id="capability-test",
    )
    assert result["schema_version"] == "finevo-capability-gate-v2"
    assert result["pass"] is True
    assert result["interface_gate"] == {"pass": True, "failure_count": 0}
    assert result["capability_assessment"] == {
        "status": "pass",
        "pass": True,
        "checks": {
            "utility-ranking": True,
            "rule-application": True,
            "rule-proposal": True,
        },
    }
    assert len(result["rows"]) == 30
    assert result["provider_failure_count"] == 0
    assert result["parse_failure_count"] == 0
    assert result["budget"]["completed_calls"] == 30
    assert all(row["interface_status"] == "pass" for row in result["rows"])
    assert all(row["evaluable"] is True for row in result["rows"])
    for category, totals in result["category_totals"].items():
        assert totals["correct"] == totals["registered_correct"]
        assert totals["denominator"] == totals["registered_total"]
        assert totals["evaluable_count"] == totals["registered_total"]
        assert totals["conditional_correct"] == totals["registered_correct"]
        assert totals["conditional_accuracy"] == 1.0
        assert totals["interface_failure_count"] == 0
        assert result["checks"][category] is True
    _validate_capability_v2(result)

    forged = json.loads(json.dumps(result))
    forged["category_totals"]["utility-ranking"]["registered_correct"] = 11
    with pytest.raises(PilotEvidenceError, match="totals"):
        _validate_capability_v2(forged)
    forged = json.loads(json.dumps(result))
    forged["rows"][0]["evaluable"] = False
    with pytest.raises(PilotEvidenceError, match="row schema/status"):
        _validate_capability_v2(forged)


def test_interface_failures_are_separate_from_conditional_capability() -> None:
    result = _run_gate(
        MixedInterfaceCapabilityProvider(),
        budget_id="capability-mixed-interface-test",
    )

    assert result["pass"] is False
    assert result["interface_gate"] == {"pass": False, "failure_count": 2}
    assert result["capability_assessment"]["status"] == "not_evaluable"
    assert result["capability_assessment"]["pass"] is None
    assert result["capability_assessment"]["checks"]["utility-ranking"] is False
    assert result["checks"]["utility-ranking"] is False
    utility = result["category_totals"]["utility-ranking"]
    assert utility == {
        "correct": 8,
        "denominator": 12,
        "required": 10,
        "registered_correct": 8,
        "registered_total": 12,
        "evaluable_count": 10,
        "conditional_correct": 8,
        "conditional_accuracy": pytest.approx(8 / 10),
        "interface_failure_count": 2,
    }

    rows = {row["task_id"]: row for row in result["rows"]}
    provider_error = rows["utility-01"]
    assert provider_error["interface_status"] == "provider_error"
    assert provider_error["evaluable"] is False
    assert provider_error["provider_error_details"]["http_status"] == 400
    assert provider_error["provider_error_details"]["param"] == "temperature"
    assert provider_error["finish_reason"] == "error"

    parse_error = rows["utility-02"]
    assert parse_error["interface_status"] == "parse_error"
    assert parse_error["evaluable"] is True
    assert parse_error["parse_error_code"] == "JSONDecodeError"
    assert parse_error["parse_error_offset"] == 0
    assert parse_error["provider_error_details"] is None
    assert parse_error["output_bytes"] == len("not JSON".encode("utf-8"))

    incomplete = rows["utility-03"]
    assert incomplete["interface_status"] == "incomplete"
    assert incomplete["evaluable"] is False
    assert incomplete["finish_reason"] == "length"
    assert incomplete["native_finish_reason"] == "length"
    assert incomplete["response_completed"] is False
    assert incomplete["output_bytes"] == len("unfinished output".encode("utf-8"))

    wrong = rows["utility-04"]
    assert wrong["interface_status"] == "pass"
    assert wrong["evaluable"] is True
    assert wrong["legal"] is True
    assert wrong["correct"] is False
    _validate_capability_v2(result)


def test_single_parse_failure_stays_in_registered_threshold_denominator() -> None:
    result = _run_gate(
        SingleParseFailureCapabilityProvider(),
        budget_id="capability-single-parse-test",
    )

    assert result["pass"] is True
    assert result["interface_gate"] == {"pass": True, "failure_count": 0}
    assert result["capability_assessment"]["status"] == "pass"
    assert result["checks"]["utility-ranking"] is True
    utility = result["category_totals"]["utility-ranking"]
    assert utility["registered_correct"] == 11
    assert utility["registered_total"] == 12
    assert utility["evaluable_count"] == 12
    assert utility["conditional_correct"] == 11
    assert utility["conditional_accuracy"] == pytest.approx(11 / 12)
    assert utility["interface_failure_count"] == 0
    row = next(row for row in result["rows"] if row["task_id"] == "utility-01")
    assert row["interface_status"] == "parse_error"
    assert row["evaluable"] is True
    assert row["correct"] is False
    _validate_capability_v2(result)

    forged = json.loads(json.dumps(result))
    forged_row = next(
        row for row in forged["rows"] if row["task_id"] == "utility-01"
    )
    forged_row["correct"] = True
    with pytest.raises(PilotEvidenceError, match="non-pass row cannot be correct"):
        _validate_capability_v2(forged)


def test_missing_finish_metadata_is_not_evaluable_capability() -> None:
    result = _run_gate(
        MissingFinishCapabilityProvider(),
        budget_id="capability-missing-finish-test",
    )

    assert result["pass"] is False
    assert result["interface_gate"] == {"pass": False, "failure_count": 1}
    assert result["capability_assessment"]["status"] == "not_evaluable"
    row = next(row for row in result["rows"] if row["task_id"] == "utility-01")
    assert row["interface_status"] == "invalid_finish"
    assert row["evaluable"] is False
    assert row["finish_reason"] is None
    assert row["response_completed"] is None
    _validate_capability_v2(result)

    forged = json.loads(json.dumps(result))
    forged_row = next(
        row for row in forged["rows"] if row["task_id"] == "utility-01"
    )
    forged_row["provider_error"] = "ForgedProviderError"
    with pytest.raises(PilotEvidenceError, match="invalid-finish"):
        _validate_capability_v2(forged)


def test_all_interface_failures_make_capability_not_evaluable() -> None:
    result = _run_gate(
        ProviderFailureCapabilityProvider(),
        budget_id="capability-not-evaluable-test",
    )

    assert result["pass"] is False
    assert result["interface_gate"] == {"pass": False, "failure_count": 30}
    assert result["capability_assessment"] == {
        "status": "not_evaluable",
        "pass": None,
        "checks": {
            "utility-ranking": None,
            "rule-application": None,
            "rule-proposal": None,
        },
    }
    assert result["provider_failure_count"] == 30
    assert result["parse_failure_count"] == 0
    for totals in result["category_totals"].values():
        assert totals["registered_correct"] == 0
        assert totals["registered_total"] == totals["denominator"]
        assert totals["evaluable_count"] == 0
        assert totals["conditional_correct"] == 0
        assert totals["conditional_accuracy"] is None
        assert totals["interface_failure_count"] == totals["registered_total"]
