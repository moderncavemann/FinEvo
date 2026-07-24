import hashlib
import json
import threading
import time
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import httpx
import openai
import pytest

from llm_providers import (
    LLMProvider,
    MODEL_COSTS,
    MultiModelLLM,
    OpenAIProvider,
    ProviderErrorDetails,
    StructuredCompletion,
    _usage_record,
)
from verified_memory.budget import BudgetExceeded, BudgetLimits, RunBudget, UsageRecord


def dialog(index: int):
    return [{"role": "user", "content": str(index)}]


def test_usage_cost_never_falls_below_frozen_token_price_estimate() -> None:
    costs = {
        "prompt": 0.01,
        "cached_prompt": 0.005,
        "completion": 0.02,
    }

    underreported = _usage_record(
        1000,
        1000,
        costs,
        reported_cost=0.0,
    )
    higher_reported = _usage_record(
        1000,
        1000,
        costs,
        reported_cost=0.04,
    )

    assert underreported.cost_usd == pytest.approx(0.03)
    assert higher_reported.cost_usd == pytest.approx(0.04)


def _openai_provider_raising(exc: Exception) -> OpenAIProvider:
    def fail(**_):
        raise exc

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.model = "gpt-5.2"
    provider.costs = {"prompt": 0.003, "completion": 0.012}
    provider.max_retries = 1
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail))
    )
    return provider


def _openai_bad_request(
    *,
    message: str,
    code: str,
    param: str,
    request_id: str,
    body_message: str,
) -> openai.BadRequestError:
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
    )
    response = httpx.Response(
        400,
        request=request,
        headers={"x-request-id": request_id},
    )
    return openai.BadRequestError(
        message,
        response=response,
        body={
            "code": code,
            "param": param,
            "type": "invalid_request_error",
            "message": body_message,
        },
    )


class LegacyTupleProvider(LLMProvider):
    def __init__(self):
        self.calls = 0

    def get_completion(self, messages, temperature=0, max_tokens=800, top_p=1.0):
        self.calls += 1
        return f"legacy-{messages[-1]['content']}", 0.25

    def get_model_name(self):
        return "legacy/test-model"


class StructuredStubProvider(LLMProvider):
    def __init__(
        self,
        *,
        usage=UsageRecord(prompt_tokens=7, completion_tokens=3, cost_usd=0.01),
        error_type=None,
        delay=0.0,
    ):
        self.usage = usage
        self.error_type = error_type
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.retry_values = []
        self.seed_values = []
        self._lock = threading.Lock()

    def get_completion(self, messages, temperature=0, max_tokens=800, top_p=1.0):
        result = self.get_structured_completion(messages, temperature, max_tokens, top_p)
        return result.text, result.cost

    def get_structured_completion(
        self,
        messages,
        temperature=0,
        max_tokens=800,
        top_p=1.0,
        max_retries=None,
        seed=None,
    ):
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.retry_values.append(max_retries)
            self.seed_values.append(seed)
        try:
            if self.delay:
                time.sleep(self.delay)
            value = messages[-1]["content"]
            attempts = 1 if max_retries is None else max_retries
            return StructuredCompletion(
                text="Error" if self.error_type else f"structured-{value}",
                usage=self.usage,
                model="stub-model",
                provider="stub",
                attempts=attempts,
                latency_seconds=self.delay,
                error_type=self.error_type,
            )
        finally:
            with self._lock:
                self.active -= 1

    def get_model_name(self):
        return "stub/stub-model"


class ExplodingProvider(LLMProvider):
    def __init__(self):
        self.calls = 0

    def get_completion(self, messages, temperature=0, max_tokens=800, top_p=1.0):
        raise RuntimeError("legacy explosion")

    def get_structured_completion(
        self,
        messages,
        temperature=0,
        max_tokens=800,
        top_p=1.0,
        max_retries=None,
        seed=None,
    ):
        self.calls += 1
        raise RuntimeError("dispatch may have occurred")

    def get_model_name(self):
        return "exploding/test-model"


def test_legacy_single_and_batch_tuple_api_are_unchanged() -> None:
    provider = LegacyTupleProvider()
    llm = MultiModelLLM(provider, num_workers=2)

    assert llm.get_completion(dialog(1)) == ("legacy-1", 0.25)
    texts, cost = llm.get_multiple_completions([dialog(2), dialog(3)])

    assert texts == ["legacy-2", "legacy-3"]
    assert cost == 0.50
    assert provider.calls == 3


def test_legacy_provider_gets_structured_compatibility_record() -> None:
    provider = LegacyTupleProvider()
    result = provider.get_structured_completion(dialog(7), max_retries=2)

    assert result.text == "legacy-7"
    assert result.provider == "legacy"
    assert result.model == "test-model"
    assert result.usage == UsageRecord(cost_usd=0.25)
    assert result.error_type is None


def test_structured_single_accounts_actual_usage_and_metadata() -> None:
    provider = StructuredStubProvider()
    llm = MultiModelLLM(provider)
    budget = RunBudget(BudgetLimits(max_calls=2, max_cost_usd=0.10))

    result = llm.get_structured_completion(
        dialog(4),
        budget=budget,
        estimated_usage=UsageRecord(8, 4, 0.02),
        label="decision",
        tags={"agent": 2, "month": 5},
        max_retries=1,
    )

    assert result.text == "structured-4"
    assert result.usage == provider.usage
    assert result.attempts == 1
    snapshot = budget.snapshot()
    assert snapshot.completed_calls == 1
    assert snapshot.active_calls == 0
    assert snapshot.accounted_usage == provider.usage
    assert snapshot.completions[0].label == "decision"
    assert dict(snapshot.completions[0].tags) == {"agent": "2", "month": "5"}
    assert provider.retry_values == [1]


def test_stopped_budget_raises_before_provider_fallback_or_dispatch() -> None:
    provider = StructuredStubProvider()
    llm = MultiModelLLM(provider)
    budget = RunBudget(BudgetLimits(max_calls=0))

    with pytest.raises(BudgetExceeded):
        llm.get_structured_completion(dialog(0), budget=budget)
    assert provider.calls == 0


def test_batch_reserves_every_call_before_dispatch_and_rolls_back_on_failure() -> None:
    provider = StructuredStubProvider()
    llm = MultiModelLLM(provider, num_workers=3)
    budget = RunBudget(BudgetLimits(max_calls=2))

    with pytest.raises(BudgetExceeded):
        llm.get_multiple_structured_completions(
            [dialog(0), dialog(1), dialog(2)],
            budget=budget,
        )

    snapshot = budget.snapshot()
    assert provider.calls == 0
    assert snapshot.completed_calls == 0
    assert snapshot.active_calls == 0
    assert snapshot.rolled_back_calls == 2


def test_structured_batch_is_concurrent_ordered_and_fully_accounted() -> None:
    provider = StructuredStubProvider(delay=0.003)
    llm = MultiModelLLM(provider, num_workers=8)
    budget = RunBudget(BudgetLimits(max_calls=40, max_cost_usd=0.40))

    results = llm.get_multiple_structured_completions(
        [dialog(index) for index in range(40)],
        budget=budget,
        labels=[f"decision-{index}" for index in range(40)],
        tags={"phase": "smoke"},
        estimated_usages=provider.usage,
        max_retries=1,
    )

    assert [item.text for item in results] == [f"structured-{index}" for index in range(40)]
    assert provider.max_active > 1
    assert provider.retry_values == [1] * 40
    snapshot = budget.snapshot()
    assert snapshot.completed_calls == 40
    assert snapshot.active_calls == 0
    assert snapshot.accounted_usage.prompt_tokens == 280
    assert snapshot.accounted_usage.completion_tokens == 120
    assert snapshot.accounted_usage.cost_usd == pytest.approx(0.40)
    assert dict(snapshot.completions[0].tags)["phase"] == "smoke"


def test_actual_batch_cost_overage_raises_after_all_calls_are_accounted() -> None:
    usage = UsageRecord(prompt_tokens=1, completion_tokens=1, cost_usd=0.20)
    provider = StructuredStubProvider(usage=usage, delay=0.002)
    llm = MultiModelLLM(provider, num_workers=2)
    budget = RunBudget(BudgetLimits(max_calls=2, max_cost_usd=0.30))

    with pytest.raises(BudgetExceeded) as caught:
        llm.get_multiple_structured_completions(
            [dialog(0), dialog(1)],
            budget=budget,
            estimated_usages=UsageRecord(cost_usd=0.05),
        )

    snapshot = budget.snapshot()
    assert provider.calls == 2
    assert snapshot.completed_calls == 2
    assert snapshot.active_calls == 0
    assert snapshot.accounted_usage.cost_usd == pytest.approx(0.40)
    assert [item.text for item in caught.value.structured_completions] == [
        "structured-0",
        "structured-1",
    ]


def test_provider_error_is_returned_as_immutable_record_and_accounted() -> None:
    provider = StructuredStubProvider(
        usage=UsageRecord(prompt_tokens=3, cost_usd=0.01),
        error_type="StubProviderError",
    )
    llm = MultiModelLLM(provider)
    budget = RunBudget(BudgetLimits(max_calls=1, max_cost_usd=0.10))

    result = llm.get_structured_completion(dialog(0), budget=budget, max_retries=1)

    assert result.text == "Error"
    assert result.error_type == "StubProviderError"
    assert result.ok is False
    assert result.attempts == 1
    assert budget.snapshot().completed_calls == 1
    assert budget.snapshot().rolled_back_calls == 0
    assert budget.snapshot().accounted_usage.prompt_tokens == 3
    with pytest.raises(FrozenInstanceError):
        result.text = "changed"


def test_unexpected_post_dispatch_exception_keeps_conservative_reservation() -> None:
    provider = ExplodingProvider()
    llm = MultiModelLLM(provider)
    estimate = UsageRecord(
        prompt_tokens=17,
        completion_tokens=9,
        cost_usd=0.07,
    )
    budget = RunBudget(BudgetLimits(max_calls=2, max_cost_usd=0.10))

    result = llm.get_structured_completion(
        dialog(0),
        budget=budget,
        estimated_usage=estimate,
    )

    assert result.text == "Error"
    assert result.error_type == "RuntimeError"
    assert result.provider == "exploding"
    assert result.usage == estimate
    assert provider.calls == 1
    snapshot = budget.snapshot()
    assert snapshot.completed_calls == 1
    assert snapshot.active_calls == 0
    assert snapshot.rolled_back_calls == 0
    assert snapshot.accounted_usage == estimate


def test_returned_provider_error_uses_componentwise_conservative_usage() -> None:
    provider = StructuredStubProvider(
        usage=UsageRecord(prompt_tokens=20, completion_tokens=1, cost_usd=0.01),
        error_type="StubProviderError",
    )
    llm = MultiModelLLM(provider)
    estimate = UsageRecord(
        prompt_tokens=10,
        completion_tokens=8,
        cost_usd=0.04,
    )
    budget = RunBudget(BudgetLimits(max_calls=1, max_cost_usd=0.10))

    result = llm.get_structured_completion(
        dialog(0),
        budget=budget,
        estimated_usage=estimate,
        max_retries=1,
    )

    expected = UsageRecord(
        prompt_tokens=20,
        completion_tokens=8,
        cost_usd=0.04,
    )
    assert result.error_type == "StubProviderError"
    assert result.usage == expected
    assert budget.snapshot().accounted_usage == expected


def test_builtin_exception_does_not_print_sensitive_text_and_keeps_reservation(
    capsys,
) -> None:
    def fail(**_):
        raise RuntimeError("sensitive-provider-response")

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.model = "gpt-5.2"
    provider.costs = {"prompt": 0.003, "completion": 0.012}
    provider.max_retries = 1
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail))
    )
    estimate = UsageRecord(
        prompt_tokens=13,
        completion_tokens=7,
        cost_usd=0.05,
    )
    budget = RunBudget(BudgetLimits(max_calls=1, max_cost_usd=0.10))

    result = MultiModelLLM(provider).get_structured_completion(
        dialog(0),
        budget=budget,
        estimated_usage=estimate,
        max_retries=1,
    )

    captured = capsys.readouterr()
    assert "sensitive-provider-response" not in captured.out
    assert "sensitive-provider-response" not in captured.err
    assert "sensitive-provider-response" not in str(result.to_dict())
    assert result.error_type == "RuntimeError"
    assert result.provider_error_details is not None
    assert result.provider_error_details.error_type == "RuntimeError"
    assert (
        result.provider_error_details.stage
        == "openai.chat.completions.create"
    )
    assert result.provider_error_details.sdk_name == "openai-python"
    assert result.provider_error_details.sdk_version == openai.__version__
    assert result.provider_error_details.http_status is None
    assert result.provider_error_details.code is None
    assert result.provider_error_details.param is None
    assert result.provider_error_details.request_id is None
    assert result.usage == estimate
    assert budget.snapshot().accounted_usage == estimate


def test_openai_bad_request_retains_only_allowlisted_error_details(capsys) -> None:
    provider_message = "SECRET_PROVIDER_MESSAGE_sk-provider"
    body_message = "SECRET_PROVIDER_BODY_sk-body"
    error = _openai_bad_request(
        message=provider_message,
        code="unsupported_parameter",
        param="temperature",
        request_id="req_stub_123",
        body_message=body_message,
    )
    provider = _openai_provider_raising(error)

    result = provider.get_structured_completion(
        [{"role": "user", "content": "SECRET_PROMPT_sk-prompt"}],
        max_retries=1,
    )

    details = result.provider_error_details
    assert result.error_type == "BadRequestError"
    assert details is not None
    assert details.error_type == "BadRequestError"
    assert details.http_status == 400
    assert details.code == "unsupported_parameter"
    assert details.param == "temperature"
    assert details.request_id == "req_stub_123"
    assert details.stage == "openai.chat.completions.create"
    assert details.sdk_name == "openai-python"
    assert details.sdk_version == openai.__version__
    assert details.schema_version == "finevo-provider-error-v1"
    assert details.redaction_policy == "allowlist-v1"

    serialized = json.dumps(result.to_dict(), sort_keys=True)
    captured = capsys.readouterr()
    for secret in (
        provider_message,
        body_message,
        "sk-provider",
        "sk-body",
        "sk-prompt",
    ):
        assert secret not in captured.out
        assert secret not in captured.err
        assert secret not in serialized


def test_openai_bad_request_rejects_hostile_error_tokens(capsys) -> None:
    provider_message = "HOSTILE_PROVIDER_MESSAGE_sk-hostile-message"
    body_message = "HOSTILE_PROVIDER_BODY_sk-hostile-body"
    hostile_request_id = "sk-hostile-request-" + "x" * 180
    error = _openai_bad_request(
        message=provider_message,
        code="sk-hostile-code with spaces",
        param="temperature\nsk-hostile-param",
        request_id=hostile_request_id,
        body_message=body_message,
    )
    provider = _openai_provider_raising(error)

    result = provider.get_structured_completion(
        [{"role": "user", "content": "sk-hostile-prompt"}],
        max_retries=1,
    )

    details = result.provider_error_details
    assert details is not None
    assert details.http_status == 400
    assert details.code is None
    assert details.param is None
    assert details.request_id is None
    assert details.stage == "openai.chat.completions.create"
    assert details.sdk_name == "openai-python"
    assert details.sdk_version == openai.__version__

    serialized = json.dumps(result.to_dict(), sort_keys=True)
    captured = capsys.readouterr()
    for secret in (
        "sk-hostile-message",
        "sk-hostile-body",
        "sk-hostile-code",
        "sk-hostile-param",
        "sk-hostile-request",
        "sk-hostile-prompt",
    ):
        assert secret not in captured.out
        assert secret not in captured.err
        assert secret not in serialized


def test_hostile_exception_class_name_is_replaced_before_serialization() -> None:
    hostile_marker = "SecretExceptionTypeSkClass"
    hostile_type = type(hostile_marker, (Exception,), {})
    provider = _openai_provider_raising(hostile_type("provider body secret"))

    result = provider.get_structured_completion(
        [{"role": "user", "content": "prompt secret"}],
        max_retries=1,
    )

    assert result.error_type == "ProviderError"
    assert result.provider_error_details is not None
    assert result.provider_error_details.error_type == "ProviderError"
    assert hostile_marker not in json.dumps(result.safe_audit_dict())
    with pytest.raises(ValueError, match="error_type is not allowlisted"):
        ProviderErrorDetails(
            error_type=hostile_marker,
            stage="provider.dispatch",
            sdk_name="test",
            sdk_version="1",
        )


def test_safe_audit_dict_replaces_provider_text_with_hash_and_byte_count() -> None:
    provider_text = "SECRET_OUTPUT_sk-output"
    result = StructuredCompletion(
        text=provider_text,
        usage=UsageRecord(prompt_tokens=3, completion_tokens=4, cost_usd=0.01),
        model="stub-model",
        provider="stub",
        attempts=1,
        latency_seconds=0.0,
    )

    audit = result.safe_audit_dict()

    assert "text" not in audit
    assert audit["output_bytes"] == len(provider_text.encode("utf-8"))
    assert audit["output_sha256"] == hashlib.sha256(
        provider_text.encode("utf-8")
    ).hexdigest()
    assert provider_text not in json.dumps(audit, sort_keys=True)


def test_safe_audit_dict_hashes_all_provider_controlled_metadata() -> None:
    marker = "SensitiveMetadataMarker123"
    result = StructuredCompletion(
        text="Error",
        usage=UsageRecord(prompt_tokens=3, completion_tokens=0, cost_usd=0.01),
        model="requested-model",
        provider="openai",
        attempts=1,
        latency_seconds=0.0,
        error_type="BadRequestError",
        system_fingerprint=marker,
        request_id=marker,
        response_model=marker,
        response_provider=marker,
        response_route=marker,
        native_finish_reason=marker,
        provider_error_details=ProviderErrorDetails(
            error_type="BadRequestError",
            stage="openai.chat.completions.create",
            sdk_name="openai-python",
            sdk_version="2.46.0",
            code=marker,
            param=marker,
            request_id=marker,
        ),
    )

    audit = result.safe_audit_dict()
    serialized = json.dumps(audit, sort_keys=True)

    assert audit["schema_version"] == "finevo-provider-completion-audit-v2"
    assert marker not in serialized
    for field in (
        "system_fingerprint",
        "request_id",
        "response_model",
        "response_provider",
        "response_route",
        "native_finish_reason",
    ):
        assert field not in audit
        assert audit[f"{field}_present"] is True
        assert audit[f"{field}_sha256"] == hashlib.sha256(
            marker.encode("utf-8")
        ).hexdigest()
    details = audit["provider_error_details"]
    for field in ("code", "param", "request_id"):
        assert field not in details
        assert details[f"{field}_present"] is True
        assert details[f"{field}_sha256"] == hashlib.sha256(
            marker.encode("utf-8")
        ).hexdigest()


def test_budget_usage_replace_preserves_sanitized_openai_error_details() -> None:
    error = _openai_bad_request(
        message="SECRET_REPLACE_MESSAGE",
        code="unsupported_parameter",
        param="temperature",
        request_id="req_replace_123",
        body_message="SECRET_REPLACE_BODY",
    )
    provider = _openai_provider_raising(error)
    returned = {}
    original_get_structured_completion = provider.get_structured_completion

    def record_provider_result(*args, **kwargs):
        result = original_get_structured_completion(*args, **kwargs)
        returned["result"] = result
        return result

    provider.get_structured_completion = record_provider_result
    estimate = UsageRecord(
        prompt_tokens=13,
        completion_tokens=7,
        cost_usd=0.05,
    )
    budget = RunBudget(BudgetLimits(max_calls=1, max_cost_usd=0.10))

    result = MultiModelLLM(provider).get_structured_completion(
        dialog(0),
        budget=budget,
        estimated_usage=estimate,
        max_retries=1,
    )

    pre_replace = returned["result"]
    assert pre_replace.usage == UsageRecord()
    assert result.usage == estimate
    assert result is not pre_replace
    assert result.provider_error_details is pre_replace.provider_error_details
    assert result.provider_error_details is not None
    assert result.provider_error_details.to_dict() == {
        "schema_version": "finevo-provider-error-v1",
        "error_type": "BadRequestError",
        "http_status": 400,
        "code": "unsupported_parameter",
        "param": "temperature",
        "request_id": "req_replace_123",
        "stage": "openai.chat.completions.create",
        "sdk": {
            "name": "openai-python",
            "version": openai.__version__,
        },
        "redaction_policy": "allowlist-v1",
    }
    assert budget.snapshot().accounted_usage == estimate


def test_batch_argument_validation_happens_before_any_reservation() -> None:
    provider = StructuredStubProvider()
    llm = MultiModelLLM(provider)
    budget = RunBudget(BudgetLimits(max_calls=10))

    with pytest.raises(ValueError):
        llm.get_multiple_structured_completions(
            [dialog(0), dialog(1)],
            budget=budget,
            labels=["only-one"],
        )
    assert provider.calls == 0
    assert budget.snapshot().active_calls == 0
    assert budget.snapshot().rolled_back_calls == 0


def test_builtin_structured_provider_uses_exposed_usage_without_network() -> None:
    request_kwargs = []

    def complete(**kwargs):
        request_kwargs.append(kwargs)
        return response

    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4, cost=0.123),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok"),
                finish_reason="stop",
            )
        ],
        system_fingerprint="fp-test",
        model="gpt-5.2-2025-12-11",
    )
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.model = "gpt-5.2"
    provider.costs = {"prompt": 0.003, "completion": 0.012}
    provider.max_retries = 20
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=complete),
        )
    )

    result = provider.get_structured_completion(dialog(0), max_retries=2, seed=17)

    assert result.text == "ok"
    assert result.usage == UsageRecord(11, 4, 0.123)
    assert result.attempts == 1
    assert result.provider == "openai"
    assert result.request_seed == 17
    assert result.system_fingerprint == "fp-test"
    assert result.response_model == "gpt-5.2-2025-12-11"
    assert request_kwargs[0]["seed"] == 17

    # The old tuple path retains its historical local price-table estimate.
    text, legacy_cost = provider.get_completion(dialog(0))
    assert text == "ok"
    assert legacy_cost == pytest.approx(11 / 1000 * 0.003 + 4 / 1000 * 0.012)
    assert "seed" not in request_kwargs[1]


def test_gpt52_cost_uses_cached_token_details() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=1000,
            prompt_tokens_details=SimpleNamespace(cached_tokens=400),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=250),
        ),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok"),
                finish_reason="stop",
            )
        ],
        system_fingerprint="fp-price",
        model="gpt-5.2-2025-12-11",
    )
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.model = "gpt-5.2"
    provider.costs = {
        "prompt": 0.00175,
        "cached_prompt": 0.000175,
        "completion": 0.014,
    }
    provider.max_retries = 1
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: response),
        )
    )

    result = provider.get_structured_completion(dialog(0), max_retries=1)

    assert result.cached_prompt_tokens == 400
    assert result.reasoning_tokens == 250
    assert result.cost == pytest.approx(0.6 * 0.00175 + 0.4 * 0.000175 + 0.014)
    assert MODEL_COSTS["gpt-5.2-2025-12-11"] == MODEL_COSTS["gpt-5.2"]


def test_multimodel_forwards_seed_before_budgeted_dispatch() -> None:
    provider = StructuredStubProvider()
    llm = MultiModelLLM(provider)
    budget = RunBudget(BudgetLimits(max_calls=1))

    llm.get_structured_completion(dialog(0), budget=budget, seed=23)

    assert provider.seed_values == [23]


def test_budgeted_provider_retries_are_rejected_before_reservation() -> None:
    provider = StructuredStubProvider()
    llm = MultiModelLLM(provider)
    budget = RunBudget(BudgetLimits(max_calls=2))

    with pytest.raises(ValueError, match="max_retries=1"):
        llm.get_structured_completion(dialog(0), budget=budget, max_retries=2)

    assert provider.calls == 0
    assert budget.snapshot().active_calls == 0


def test_invalid_seed_fails_before_budget_reservation() -> None:
    provider = StructuredStubProvider()
    llm = MultiModelLLM(provider)
    budget = RunBudget(BudgetLimits(max_calls=1))

    with pytest.raises(TypeError, match="seed"):
        llm.get_structured_completion(dialog(0), budget=budget, seed=True)

    assert provider.calls == 0
    assert budget.snapshot().active_calls == 0


def test_builtin_retry_override_caps_attempts_without_network(monkeypatch) -> None:
    attempts = 0

    def fail(**_):
        nonlocal attempts
        attempts += 1
        raise TimeoutError("stub timeout")

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.model = "gpt-5.2"
    provider.costs = {"prompt": 0.003, "completion": 0.012}
    provider.max_retries = 20
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fail))
    )
    monkeypatch.setattr("llm_providers.time.sleep", lambda _: None)

    result = provider.get_structured_completion(dialog(0), max_retries=2)

    assert attempts == 2
    assert result.text == "Error"
    assert result.error_type == "TimeoutError"
    assert result.attempts == 2
