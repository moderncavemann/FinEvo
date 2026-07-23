from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_providers import (
    OllamaProvider,
    OpenAIProvider,
    ThirdPartyProvider,
    create_llm_provider,
)
from verified_memory.pilot_contract import (
    PilotContractError,
    ProviderRequestProfile,
    load_pilot_contract,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def profiles():
    return load_pilot_contract(
        ROOT / "experiments" / "pilot_v1.yaml"
    ).provider_profiles


def _sdk_response(model: str, *, cost=None):
    usage = SimpleNamespace(prompt_tokens=20, completion_tokens=5)
    if cost is not None:
        usage.cost = cost
    provider = {
        "anthropic": "Anthropic",
        "google": "Google",
        "meta-llama": "DeepInfra",
    }.get(model.split("/", 1)[0])
    return SimpleNamespace(
        id="request-1",
        usage=usage,
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
        system_fingerprint="fingerprint-1",
        model=model,
        provider=provider,
    )


def _openai_compatible_provider(provider_type, profile, response):
    requests = []

    def complete(**kwargs):
        requests.append(kwargs)
        return response

    provider = provider_type.__new__(provider_type)
    provider.model = profile.requested_model
    provider.costs = profile.price_snapshot.costs_per_1k()
    provider.max_retries = 1
    provider.request_profile = profile
    if provider_type is ThirdPartyProvider:
        selected_model = dict(profile.artifact_identity)["served_snapshot"]
        response.openrouter_metadata = {
            "requested": profile.requested_model,
            "strategy": "direct",
            "attempt": 1,
            "endpoints": {
                "available": [
                    {
                        "provider": profile.provider_pin[0],
                        "model": selected_model,
                        "selected": True,
                    }
                ]
            },
            "attempts": [
                {
                    "provider": profile.provider_pin[0],
                    "model": selected_model,
                    "status": 200,
                }
            ],
        }
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=complete),
        )
    )
    return provider, requests


def test_openrouter_request_is_json_pinned_nonfallback_and_single_attempt(
    profiles,
) -> None:
    profile = profiles["gemini35_flash_sentinel"]
    provider, requests = _openai_compatible_provider(
        ThirdPartyProvider,
        profile,
        _sdk_response(profile.served_model, cost=0.01),
    )

    result = provider.get_structured_completion(
        [{"role": "user", "content": "return JSON"}],
        max_retries=1,
        seed=1099057501,
    )

    assert result.ok is True
    assert result.response_model == profile.served_model
    assert result.cost == 0.01
    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == "google/gemini-3.5-flash"
    assert request["seed"] == 1099057501
    assert request["response_format"] == {"type": "json_object"}
    assert request["extra_body"] == {
        "provider": {
            "order": ["Google"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        "reasoning": {"effort": "medium", "exclude": True},
    }


def test_omitted_reasoning_is_absent_from_openrouter_wire_request(
    profiles,
) -> None:
    profile = profiles["llama4_maverick_sentinel"]
    provider, requests = _openai_compatible_provider(
        ThirdPartyProvider,
        profile,
        _sdk_response(profile.served_model),
    )

    result = provider.get_structured_completion(
        [{"role": "user", "content": "return JSON"}],
        seed=1421875452,
    )

    assert result.ok is True
    assert requests[0]["extra_body"] == {
        "provider": {
            "order": ["DeepInfra"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    }


def test_unsupported_seed_and_retry_override_fail_before_dispatch(
    profiles,
) -> None:
    profile = profiles["opus48_sentinel"]
    provider, requests = _openai_compatible_provider(
        ThirdPartyProvider,
        profile,
        _sdk_response(profile.served_model),
    )

    with pytest.raises(PilotContractError, match="does not support"):
        provider.get_structured_completion(
            [{"role": "user", "content": "return JSON"}],
            seed=17,
        )
    with pytest.raises(PilotContractError, match="exactly one"):
        provider.get_structured_completion(
            [{"role": "user", "content": "return JSON"}],
            max_retries=2,
            seed=None,
        )
    assert requests == []

    result = provider.get_structured_completion(
        [{"role": "user", "content": "return JSON"}],
        seed=None,
    )
    assert result.ok is True
    assert "seed" not in requests[0]


def test_unknown_endpoint_price_fails_before_dispatch(profiles) -> None:
    payload = profiles["llama4_maverick_sentinel"].to_dict()
    payload["price_snapshot"]["endpoint_output"] = None
    profile = ProviderRequestProfile.from_dict(payload)

    requests = []
    provider = ThirdPartyProvider.__new__(ThirdPartyProvider)
    provider.model = profile.requested_model
    provider.costs = {"prompt": 0.0, "completion": 0.0}
    provider.max_retries = 1
    provider.request_profile = profile
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: requests.append(kwargs)
            )
        )
    )

    with pytest.raises(PilotContractError, match="price is unknown"):
        provider.get_structured_completion(
            [{"role": "user", "content": "return JSON"}],
            seed=1099057501,
        )
    assert requests == []


def test_served_model_mismatch_is_an_accounted_contract_error(profiles) -> None:
    profile = profiles["gemini35_flash_sentinel"]
    provider, requests = _openai_compatible_provider(
        ThirdPartyProvider,
        profile,
        _sdk_response("google/gemini-other", cost=0.02),
    )

    result = provider.get_structured_completion(
        [{"role": "user", "content": "return JSON"}],
        seed=1769977770,
    )

    assert len(requests) == 1
    assert result.ok is False
    assert result.text == "Error"
    assert result.error_type == "PilotContractError"
    assert result.response_model == "google/gemini-other"
    assert result.usage.prompt_tokens == 20
    assert result.usage.completion_tokens == 5
    assert result.cost == 0.02


def test_direct_openai_profile_adds_json_reasoning_and_frozen_seed(
    profiles,
) -> None:
    profile = profiles["gpt52_main"]
    provider, requests = _openai_compatible_provider(
        OpenAIProvider,
        profile,
        _sdk_response(profile.served_model),
    )

    result = provider.get_structured_completion(
        [{"role": "user", "content": "return JSON"}],
        seed=959809858,
    )

    assert result.ok is True
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert requests[0]["reasoning_effort"] == "medium"
    assert requests[0]["seed"] == 959809858
    assert requests[0]["max_completion_tokens"] == 800
    assert "max_tokens" not in requests[0]


def test_factory_uses_profile_model_one_attempt_and_snapshot_price(
    profiles, monkeypatch
) -> None:
    import openai

    clients = []

    def fake_client(**kwargs):
        clients.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(openai, "OpenAI", fake_client)
    profile = profiles["llama4_maverick_sentinel"]
    provider = create_llm_provider(
        "thirdparty",
        api_key="test-key",
        request_profile=profile,
    )

    assert provider.model == profile.requested_model
    assert provider.max_retries == 1
    assert provider.costs == {
        "prompt": 0.0002,
        "cached_prompt": 0.0002,
        "completion": 0.0008,
    }
    assert clients[0]["base_url"] == "https://openrouter.ai/api/v1"
    assert clients[0]["max_retries"] == 0
    assert clients[0]["default_headers"]["X-OpenRouter-Metadata"] == "enabled"

    with pytest.raises(PilotContractError, match="exactly one"):
        create_llm_provider(
            "thirdparty",
            api_key="test-key",
            max_retries=2,
            request_profile=profile,
        )
    assert len(clients) == 1


def test_profileless_openrouter_constructor_does_not_opt_in_metadata(
    monkeypatch,
) -> None:
    import openai

    clients = []
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kwargs: clients.append(kwargs) or SimpleNamespace(),
    )

    create_llm_provider(
        "thirdparty",
        model="legacy/model",
        api_key="test-key",
    )

    assert len(clients) == 1
    assert "X-OpenRouter-Metadata" not in clients[0]["default_headers"]


def test_ollama_profile_forwards_deterministic_seed(
    profiles, monkeypatch
) -> None:
    import requests

    sent = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "llama3.3:70b-instruct-q4_K_M",
                "message": {"content": '{"ok":true}'},
                "prompt_eval_count": 10,
                "eval_count": 4,
            }

    def post(url, **kwargs):
        sent.append((url, kwargs))
        return Response()

    monkeypatch.setattr(requests, "post", post)
    payload = profiles["llama33_local_sentinel"].to_dict()
    payload["json_mode"] = "json_object"
    profile = ProviderRequestProfile.from_dict(payload)
    provider = OllamaProvider(
        model=profile.requested_model,
        request_profile=profile,
    )
    result = provider.get_structured_completion(
        [{"role": "user", "content": "return JSON"}],
        seed=617806385,
    )

    assert result.ok is True
    assert provider.max_retries == 1
    assert sent[0][1]["json"]["options"]["seed"] == 617806385
    assert sent[0][1]["json"]["format"] == "json"


def test_profileless_ollama_request_does_not_force_json_format(
    monkeypatch,
) -> None:
    import requests

    sent = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "legacy-local",
                "message": {"content": "plain text"},
                "prompt_eval_count": 2,
                "eval_count": 2,
            }

    monkeypatch.setattr(
        requests,
        "post",
        lambda url, **kwargs: sent.append((url, kwargs)) or Response(),
    )
    provider = OllamaProvider(model="legacy-local", max_retries=1)

    result = provider.get_structured_completion(
        [{"role": "user", "content": "plain text"}],
    )

    assert result.ok is True
    assert "format" not in sent[0][1]["json"]


def test_legacy_thirdparty_request_shape_remains_compatible() -> None:
    response = _sdk_response("legacy/model")
    profileless = SimpleNamespace()
    requests = []

    def complete(**kwargs):
        requests.append(kwargs)
        return response

    profileless.model = "legacy/model"
    profileless.costs = {"prompt": 0.0, "completion": 0.0}
    profileless.max_retries = 20
    profileless.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=complete),
        )
    )

    result = ThirdPartyProvider.get_structured_completion(
        profileless,
        [{"role": "user", "content": "legacy"}],
    )

    assert result.ok is True
    assert "response_format" not in requests[0]
    assert "extra_body" not in requests[0]
    assert "seed" not in requests[0]
