from __future__ import annotations

from pathlib import Path

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
    return {
        "error_type": None,
        "response_model": profile.served_model,
        "attempts": 1,
        "request_profile_id": profile.profile_id,
        "request_provider_pin": list(profile.provider_pin),
        "request_artifact_identity": dict(profile.artifact_identity),
        "request_price_snapshot_source": profile.price_snapshot.source,
        "request_price_snapshot_captured_at": (
            profile.price_snapshot.captured_at
        ),
        "response_provider": profile.provider_pin[0],
        "response_route": (
            "direct"
            if profile.transport == "openai"
            else dict(profile.artifact_identity).get(
                "endpoint_tag",
                profile.routing_mode,
            )
        ),
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost_usd": 0.01,
        },
    }


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
