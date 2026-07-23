from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_providers import (
    MultiModelLLM,
    StructuredCompletion,
    ThirdPartyProvider,
    _validate_openrouter_response_route,
)
from verified_memory.budget import BudgetLimits, RunBudget, UsageRecord
from verified_memory.pilot_checkpoint import config_from_dict
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.runner import (
    PreflightP95Reservation,
    ShockEvent,
    VerifiedRunConfig,
    VerifiedRunError,
    _provider_row,
    preflight_p95_reservation_for_call,
    run_verified_experiment,
)


ROOT = Path(__file__).resolve().parents[1]


def _p95_entry(
    *,
    raw_prompt: float = 8.0,
    raw_completion: float = 4.0,
    raw_cost: float = 0.02,
) -> dict:
    return {
        "sample_count": 12,
        "raw_p95": {
            "prompt_tokens": raw_prompt,
            "completion_tokens": raw_completion,
            "total_tokens": raw_prompt + raw_completion,
            "cost_usd": raw_cost,
        },
        "reserved_p95": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost_usd": 0.025,
        },
        "reserve_multiplier": 1.25,
    }


def _scientific_config(
    *,
    model: str,
    reservations: dict | None,
    episode_length: int = 1,
    semantic_after: int = 3,
) -> VerifiedRunConfig:
    return VerifiedRunConfig(
        run_id="strict-p95",
        num_agents=2,
        episode_length=episode_length,
        semantic_proposal_after=semantic_after,
        shock_schedule=tuple(
            ShockEvent(
                decision_t=decision_t,
                phase="baseline",
                interest_rate=0.03,
            )
            for decision_t in range(episode_length)
        ),
        scientific_scope="preregistered_mechanism_micro_pilot",
        pilot_contract_hash="a" * 64,
        pilot_tag="pilot-v1",
        allow_scientific_scope=True,
        preflight_p95_reservations=reservations or {},
    )


class _NeverDispatchProvider:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls = 0

    def get_model_name(self) -> str:
        return self.model_name

    def get_completion(self, *args, **kwargs):
        raise AssertionError("tuple dispatch is not expected")

    def get_structured_completion(self, *args, **kwargs) -> StructuredCompletion:
        self.calls += 1
        raise AssertionError("scientific p95 validation must precede dispatch")


def test_scientific_unknown_model_cannot_fall_back_to_zero_cost_before_dispatch(
) -> None:
    provider = _NeverDispatchProvider("hosted/unknown-model")
    budget = RunBudget(BudgetLimits(max_calls=10, max_cost_usd=1.0))
    config = _scientific_config(
        model=provider.model_name,
        reservations=None,
    )

    with pytest.raises(VerifiedRunError, match=r"unknown-model::action"):
        run_verified_experiment(
            config,
            llm=MultiModelLLM(provider),
            budget=budget,
            env_config_source=ROOT / "config.yaml",
        )

    assert provider.calls == 0
    assert budget.snapshot().completed_calls == 0
    assert budget.snapshot().active_calls == 0


def test_bounded_hosted_unknown_model_also_cannot_reserve_zero_cost() -> None:
    config = VerifiedRunConfig(
        run_id="bounded-unknown-price",
        enable_semantic=False,
    )
    with pytest.raises(VerifiedRunError, match="unknown price"):
        preflight_p95_reservation_for_call(
            config,
            provider_model_name="hosted/unknown-model",
            call_kind="action",
            prompt="one prompt",
            max_tokens=10,
        )


def test_missing_semantic_p95_refuses_entire_scientific_run_before_action_dispatch(
) -> None:
    provider = _NeverDispatchProvider("hosted/frozen-model")
    config = _scientific_config(
        model=provider.model_name,
        episode_length=3,
        semantic_after=3,
        reservations={
            provider.model_name: {
                "action": _p95_entry(),
            }
        },
    )

    with pytest.raises(VerifiedRunError, match=r"frozen-model::semantic"):
        run_verified_experiment(
            config,
            llm=MultiModelLLM(provider),
            budget=RunBudget(BudgetLimits(max_calls=20, max_cost_usd=1.0)),
            env_config_source=ROOT / "config.yaml",
        )

    assert provider.calls == 0


def test_p95_mapping_is_immutable_exact_and_checkpoint_serializable() -> None:
    model = "hosted/frozen-model"
    config = _scientific_config(
        model=model,
        reservations={
            model: {
                "action": _p95_entry(),
            }
        },
    )

    usage = preflight_p95_reservation_for_call(
        config,
        provider_model_name=model,
        call_kind="action",
    )
    assert usage == UsageRecord(10, 5, 0.025)
    assert isinstance(config.preflight_p95_reservations, tuple)
    assert isinstance(
        config.preflight_p95_reservations[0],
        PreflightP95Reservation,
    )
    with pytest.raises(FrozenInstanceError):
        config.preflight_p95_reservations[0].sample_count = 99

    payload = config.to_dict()
    assert payload["preflight_p95_reservations"][model]["action"] == _p95_entry()
    assert json.loads(json.dumps(payload)) == payload
    restored = config_from_dict(payload)
    assert restored.preflight_p95_reservations == config.preflight_p95_reservations
    assert restored.to_dict() == payload


def test_malformed_or_zero_hosted_p95_never_becomes_a_dispatch_reservation() -> None:
    model = "hosted/frozen-model"
    malformed = _p95_entry()
    malformed["reserved_p95"]["cost_usd"] = 0.0
    with pytest.raises(ValueError, match=r"raw_p95 \* 1.25"):
        _scientific_config(
            model=model,
            reservations={model: {"action": malformed}},
        )

    zero = _p95_entry(raw_cost=0.0)
    zero["reserved_p95"]["cost_usd"] = 0.0
    config = _scientific_config(
        model=model,
        reservations={model: {"action": zero}},
    )
    with pytest.raises(VerifiedRunError, match="positive cost"):
        preflight_p95_reservation_for_call(
            config,
            provider_model_name=model,
            call_kind="action",
        )


def _route_metadata(
    profile,
    *,
    requested: str | None = None,
    strategy: str = "direct",
    router_attempt: int = 1,
    selected_provider: str | None = None,
    selected_model: str | None = None,
) -> dict:
    provider = selected_provider or profile.provider_pin[0]
    model = selected_model or dict(profile.artifact_identity)["served_snapshot"]
    return {
        "requested": requested or profile.requested_model,
        "strategy": strategy,
        "attempt": router_attempt,
        "endpoints": {
            "available": [
                {
                    "provider": provider,
                    "model": model,
                    "selected": True,
                }
            ]
        },
        "attempts": [
            {
                "provider": provider,
                "model": model,
                "status": 200,
            }
        ],
    }


def _sdk_response(
    profile,
    *,
    metadata: object | None,
    metadata_location: str = "attribute",
) -> SimpleNamespace:
    fields = {
        "id": "request-route-test",
        "usage": SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=5,
            cost=0.01,
        ),
        "choices": [
            SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))
        ],
        "system_fingerprint": "fingerprint-route",
        "model": profile.served_model,
    }
    if metadata is not None:
        if metadata_location == "attribute":
            fields["openrouter_metadata"] = metadata
        elif metadata_location == "model_extra":
            fields["model_extra"] = {"openrouter_metadata": metadata}
        else:
            raise AssertionError("unsupported metadata location")
    return SimpleNamespace(**fields)


def _thirdparty(
    profile, response: SimpleNamespace
) -> tuple[ThirdPartyProvider, list[dict]]:
    requests: list[dict] = []

    def complete(**kwargs):
        requests.append(kwargs)
        return response

    provider = ThirdPartyProvider.__new__(ThirdPartyProvider)
    provider.model = profile.requested_model
    provider.costs = profile.price_snapshot.costs_per_1k()
    provider.max_retries = 1
    provider.request_profile = profile
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=complete),
        )
    )
    return provider, requests


@pytest.mark.parametrize(
    "mutate",
    [
        None,
        lambda _profile, metadata: metadata.pop("requested"),
        lambda profile, metadata: metadata.update(
            requested=f"{profile.requested_model}-other"
        ),
        lambda _profile, metadata: metadata.pop("strategy"),
        lambda _profile, metadata: metadata.update(strategy="fallback"),
        lambda _profile, metadata: metadata.pop("attempt"),
        lambda _profile, metadata: metadata.update(attempt=2),
        lambda _profile, metadata: metadata.pop("endpoints"),
        lambda _profile, metadata: metadata["endpoints"].pop("available"),
        lambda _profile, metadata: metadata["endpoints"]["available"][0].pop(
            "selected"
        ),
        lambda _profile, metadata: metadata["endpoints"].update(available=[]),
        lambda _profile, metadata: metadata["endpoints"]["available"].append(
            dict(metadata["endpoints"]["available"][0])
        ),
        lambda _profile, metadata: metadata["endpoints"]["available"][0].pop(
            "provider"
        ),
        lambda _profile, metadata: metadata["endpoints"]["available"][0].update(
            provider="Together"
        ),
        lambda _profile, metadata: metadata["endpoints"]["available"][0].pop(
            "model"
        ),
        lambda _profile, metadata: metadata["endpoints"]["available"][0].update(
            model="meta-llama/unfrozen"
        ),
        lambda _profile, metadata: metadata.pop("attempts"),
        lambda _profile, metadata: metadata.update(attempts=[]),
        lambda _profile, metadata: metadata["attempts"][0].update(
            provider="Together"
        ),
        lambda _profile, metadata: metadata["attempts"][0].update(
            model="meta-llama/unfrozen"
        ),
        lambda _profile, metadata: metadata["attempts"][0].update(status=500),
    ],
    ids=[
        "metadata-missing",
        "requested-model-missing",
        "requested-model-mismatch",
        "strategy-missing",
        "strategy-not-direct",
        "router-attempt-missing",
        "router-attempt-not-one",
        "endpoints-missing",
        "available-endpoints-missing",
        "selection-flag-missing",
        "no-selected-endpoint",
        "ambiguous-selected-endpoints",
        "selected-provider-missing",
        "selected-provider-mismatch",
        "selected-model-missing",
        "selected-model-mismatch",
        "attempts-missing",
        "attempts-empty",
        "attempt-provider-mismatch",
        "attempt-model-mismatch",
        "attempt-not-successful",
    ],
)
def test_openrouter_invalid_route_attestation_is_accounted_contract_no_go(
    mutate,
) -> None:
    profile = load_pilot_contract(
        ROOT / "experiments" / "pilot_v1.yaml"
    ).provider_profiles["llama4_maverick_sentinel"]
    metadata = _route_metadata(profile)
    if mutate is None:
        response_metadata = None
    else:
        mutate(profile, metadata)
        response_metadata = metadata
    provider, requests = _thirdparty(
        profile,
        _sdk_response(profile, metadata=response_metadata),
    )
    result = provider.get_structured_completion(
        [{"role": "user", "content": "JSON"}],
        seed=7,
    )

    assert result.ok is False
    assert result.error_type == "PilotContractError"
    assert result.usage == UsageRecord(20, 5, 0.01)
    assert result.response_provider is None
    assert result.response_route is None
    assert len(requests) == 1


@pytest.mark.parametrize("metadata_location", ["attribute", "model_extra"])
def test_openrouter_exact_pin_captures_route_and_request_artifact_identity(
    metadata_location: str,
) -> None:
    profile = load_pilot_contract(
        ROOT / "experiments" / "pilot_v1.yaml"
    ).provider_profiles["llama4_maverick_sentinel"]
    provider, requests = _thirdparty(
        profile,
        _sdk_response(
            profile,
            metadata=_route_metadata(profile),
            metadata_location=metadata_location,
        ),
    )
    result = provider.get_structured_completion(
        [{"role": "user", "content": "JSON"}],
        seed=7,
    )

    assert result.ok is True
    assert result.response_provider == "DeepInfra"
    assert result.response_route == (
        "meta-llama/llama-4-maverick-17b-128e-instruct"
    )
    assert len(requests) == 1
    assert result.request_profile_id == profile.profile_id
    assert result.request_provider_pin == ("DeepInfra",)
    assert dict(result.request_artifact_identity) == dict(profile.artifact_identity)
    assert result.request_price_snapshot_source == profile.price_snapshot.source
    usage_row = _provider_row(
        result,
        call_kind="action",
        decision_t=0,
        agent_id=0,
        prompt_hash="b" * 64,
    )
    assert usage_row["response_provider"] == "DeepInfra"
    assert usage_row["response_route"] == (
        "meta-llama/llama-4-maverick-17b-128e-instruct"
    )
    assert usage_row["request_provider_pin"] == ["DeepInfra"]
    assert usage_row["request_artifact_identity"]["endpoint_tag"] == (
        "deepinfra/base"
    )


def test_openrouter_metadata_mapping_shape_is_supported() -> None:
    profile = load_pilot_contract(
        ROOT / "experiments" / "pilot_v1.yaml"
    ).provider_profiles["llama4_maverick_sentinel"]
    provider, model = _validate_openrouter_response_route(
        profile,
        {"openrouter_metadata": _route_metadata(profile)},
    )

    assert provider == "DeepInfra"
    assert model == "meta-llama/llama-4-maverick-17b-128e-instruct"


def test_openrouter_transport_error_does_not_echo_exception_or_secret(
    capsys,
) -> None:
    profile = load_pilot_contract(
        ROOT / "experiments" / "pilot_v1.yaml"
    ).provider_profiles["llama4_maverick_sentinel"]
    calls = 0

    def fail(**_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("never-print-sk-secret-material")

    provider = ThirdPartyProvider.__new__(ThirdPartyProvider)
    provider.model = profile.requested_model
    provider.costs = profile.price_snapshot.costs_per_1k()
    provider.max_retries = 1
    provider.request_profile = profile
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=fail),
        )
    )

    result = provider.get_structured_completion(
        [{"role": "user", "content": "JSON"}],
        seed=7,
    )

    captured = capsys.readouterr()
    assert calls == 1
    assert result.error_type == "RuntimeError"
    assert captured.out == ""
    assert captured.err == ""
