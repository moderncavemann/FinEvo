from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_providers import OpenAIProvider, STRICT_OPENAI_BASE_URL
from verified_memory.pilot_contract import load_pilot_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_11.yaml"


def _sdk_response(model: str):
    return SimpleNamespace(
        id="fixture-request-1",
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"status":"ok"}'),
                finish_reason="stop",
            )
        ],
        system_fingerprint="fixture-fingerprint",
        model=model,
    )


def _wired_openai_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile_id: str,
) -> tuple[OpenAIProvider, dict[str, object], list[dict[str, object]]]:
    contract = load_pilot_contract(CONTRACT_PATH)
    profile = contract.provider_profiles[profile_id]
    client_options: dict[str, object] = {}
    requests: list[dict[str, object]] = []

    def create(**kwargs):
        requests.append(kwargs)
        return _sdk_response(profile.served_model)

    def client(**kwargs):
        client_options.update(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            )
        )

    import openai

    monkeypatch.setattr(openai, "OpenAI", client)
    provider = OpenAIProvider(
        "fixture-key-never-dispatched",
        model=profile.requested_model,
        max_retries=1,
        request_profile=profile,
        request_timeout_seconds=300,
    )
    return provider, client_options, requests


@pytest.mark.parametrize("profile_id", ("gpt52_main", "gpt56_diagnostic"))
def test_v21111_openai_service_tier_default_is_on_actual_wire_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    profile_id: str,
) -> None:
    provider, client_options, requests = _wired_openai_provider(
        monkeypatch,
        profile_id=profile_id,
    )
    result = provider.get_structured_completion(
        [{"role": "user", "content": "return JSON"}],
        max_tokens=8_192,
        seed=None,
    )

    assert client_options == {
        "api_key": "fixture-key-never-dispatched",
        "timeout": 300.0,
        "max_retries": 0,
        "base_url": STRICT_OPENAI_BASE_URL,
    }
    assert result.ok is True
    assert result.response_model == provider.request_profile.served_model
    assert result.request_profile_id == profile_id
    assert len(requests) == 1
    request = requests[0]
    assert request == {
        "model": provider.request_profile.requested_model,
        "messages": [{"role": "user", "content": "return JSON"}],
        "response_format": {"type": "json_object"},
        "reasoning_effort": "medium",
        "service_tier": "default",
        "max_completion_tokens": 8_192,
    }
    assert set(result.request_parameters) == set(request)
    assert result.temperature_dispatch == "omitted_unsupported"


def test_v21111_272k_short_context_gate_fails_before_sdk_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _, requests = _wired_openai_provider(
        monkeypatch,
        profile_id="gpt56_diagnostic",
    )
    result = provider.get_structured_completion(
        [{"role": "user", "content": "x" * 272_001}],
        max_tokens=8_192,
        seed=None,
    )

    assert result.ok is False
    assert result.error_type == "PilotContractError"
    assert result.provider_error_details is not None
    assert result.provider_error_details.stage == "openai.request.build"
    assert result.request_parameters == ()
    assert result.usage.total_tokens == 0
    assert result.cost == 0.0
    assert requests == []
