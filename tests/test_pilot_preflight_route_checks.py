from __future__ import annotations

from pathlib import Path

import pytest

from llm_providers import PINNED_PROVIDER_SDK_VERSIONS
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_orchestrator import _preflight_checks


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v1.yaml"


class _Result:
    def __init__(self, api_rows: list[dict]) -> None:
        self._api_rows = api_rows
        self.validation_status = {"status": "pass"}
        self.summary = {
            "scientific_evidence": False,
            "result_scope": "bounded_method_smoke",
        }

    def stream(self, name: str) -> list[dict]:
        if name == "actions":
            return [
                {"decision": {"clipped": False}} for _ in range(12)
            ]
        if name == "semantic_proposals":
            return [{"candidate_parse_status": "success"}]
        if name == "api_usage":
            return self._api_rows
        raise AssertionError(name)


def _row(profile) -> dict:
    if profile.transport == "openrouter":
        response_provider = profile.provider_pin[0]
        response_route = dict(profile.artifact_identity)["served_snapshot"]
        route_attestation_code = "OR_RA_PASS"
        provider_sdk_name = "openai-python"
        temperature_dispatch = "explicit"
        request_parameters = {
            "extra_body",
            "max_tokens",
            "messages",
            "model",
            "response_format",
            "temperature",
            "top_p",
        }
        if profile.seed_capability != "unsupported":
            request_parameters.add("seed")
    elif profile.transport == "openai":
        response_provider = "OpenAI-direct"
        response_route = "direct"
        route_attestation_code = None
        provider_sdk_name = "openai-python"
        temperature_dispatch = "omitted_unsupported"
        request_parameters = {
            "max_completion_tokens",
            "messages",
            "model",
            "reasoning_effort",
            "response_format",
            "top_p",
        }
        if profile.seed_capability != "unsupported":
            request_parameters.add("seed")
    elif profile.transport == "ollama":
        response_provider = "local-ollama"
        response_route = "local"
        route_attestation_code = None
        provider_sdk_name = "requests"
        temperature_dispatch = "explicit"
        request_parameters = {"messages", "model", "options", "stream"}
        if profile.json_mode == "json_object":
            request_parameters.add("format")
    else:
        raise AssertionError(f"unsupported test transport: {profile.transport}")

    return {
        "error_type": None,
        "provider_error_details": None,
        "response_model": profile.served_model,
        "finish_reason": "stop",
        "response_completed": True,
        "provider_sdk_name": provider_sdk_name,
        "provider_sdk_version": (
            PINNED_PROVIDER_SDK_VERSIONS["requests"]
            if profile.transport == "ollama"
            else PINNED_PROVIDER_SDK_VERSIONS["openai"]
        ),
        "request_parameters": sorted(request_parameters),
        "temperature_dispatch": temperature_dispatch,
        "output_disposition": "accepted",
        "attempts": 1,
        "request_profile_id": profile.profile_id,
        "request_provider_pin": list(profile.provider_pin),
        "request_artifact_identity": dict(profile.artifact_identity),
        "request_price_snapshot_source": profile.price_snapshot.source,
        "request_price_snapshot_captured_at": (
            profile.price_snapshot.captured_at
        ),
        "response_provider": response_provider,
        "response_route": response_route,
        "route_attestation_code": route_attestation_code,
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost_usd": 0.01,
        },
    }


def _assert_preflight_no_go(row: dict, profile) -> None:
    baseline = _preflight_checks(_Result([_row(profile)]), profile)
    assert all(baseline.values()), baseline

    checks = _preflight_checks(_Result([row]), profile)
    assert not all(checks.values()), checks


@pytest.mark.parametrize(
    "profile_id",
    [
        "gpt52_main",
        "llama4_maverick_sentinel",
        "llama33_local_sentinel",
    ],
)
def test_preflight_accepts_exact_transport_route_and_runtime_metadata(
    profile_id: str,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    profile = contract.provider_profiles[profile_id]

    checks = _preflight_checks(_Result([_row(profile)]), profile)

    assert all(checks.values()), checks


def test_preflight_requires_actual_route_and_full_request_profile_binding() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    profile = contract.provider_profiles["gpt52_main"]
    passing = _preflight_checks(_Result([_row(profile)]), profile)
    assert all(passing.values())

    missing_route = _row(profile)
    missing_route["response_route"] = None
    checks = _preflight_checks(_Result([missing_route]), profile)
    assert checks["response_route_metadata_complete"] is False

    wrong_profile = _row(profile)
    wrong_profile["request_profile_id"] = "different-profile"
    checks = _preflight_checks(_Result([wrong_profile]), profile)
    assert checks["request_profile_exact"] is False

    wrong_price = _row(profile)
    wrong_price["request_price_snapshot_source"] = "unfrozen-price"
    checks = _preflight_checks(_Result([wrong_price]), profile)
    assert checks["request_price_snapshot_exact"] is False


@pytest.mark.parametrize(
    "missing_field",
    [
        "response_provider",
        "response_route",
        "provider_sdk_name",
        "provider_sdk_version",
        "route_attestation_code",
        "provider_error_details",
    ],
)
def test_preflight_rejects_missing_openrouter_provenance_field(
    missing_field: str,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    profile = contract.provider_profiles["llama4_maverick_sentinel"]
    row = _row(profile)
    row.pop(missing_field)

    _assert_preflight_no_go(row, profile)


def test_preflight_rejects_openrouter_provider_outside_exact_pin() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    profile = contract.provider_profiles["llama4_maverick_sentinel"]
    row = _row(profile)
    row["response_provider"] = "Together"

    _assert_preflight_no_go(row, profile)


def test_preflight_rejects_openrouter_route_outside_served_snapshot() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    profile = contract.provider_profiles["llama4_maverick_sentinel"]
    row = _row(profile)
    row["response_route"] = "meta-llama/unfrozen"

    _assert_preflight_no_go(row, profile)


def test_preflight_rejects_openrouter_failed_route_attestation_code() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    profile = contract.provider_profiles["llama4_maverick_sentinel"]
    row = _row(profile)
    row["route_attestation_code"] = "OR_RA_007_PROVIDER_PIN_MISMATCH"

    _assert_preflight_no_go(row, profile)


@pytest.mark.parametrize(
    ("profile_id", "field", "invalid_value"),
    [
        ("gpt52_main", "response_provider", "OpenAI"),
        ("gpt52_main", "response_route", "OpenAI-direct"),
        ("llama33_local_sentinel", "response_provider", "Ollama"),
        ("llama33_local_sentinel", "response_route", "ollama"),
    ],
)
def test_preflight_rejects_direct_and_local_route_mismatch(
    profile_id: str,
    field: str,
    invalid_value: str,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    profile = contract.provider_profiles[profile_id]
    row = _row(profile)
    row[field] = invalid_value

    _assert_preflight_no_go(row, profile)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("provider_sdk_name", ""),
        ("provider_sdk_version", ""),
        ("provider_sdk_version", "unfrozen-sdk-version"),
        ("request_parameters", ["messages", "model"]),
        ("output_disposition", "discarded_invalid_finish"),
        ("finish_reason", "length"),
        ("finish_reason", "error"),
        ("response_completed", False),
        (
            "provider_error_details",
            {
                "error_type": "PilotContractError",
                "code": "served_model_mismatch",
            },
        ),
    ],
)
def test_preflight_rejects_incomplete_or_failed_success_metadata(
    field: str,
    invalid_value: object,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    profile = contract.provider_profiles["gpt52_main"]
    row = _row(profile)
    row[field] = invalid_value

    _assert_preflight_no_go(row, profile)
