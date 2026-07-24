import hashlib
import json
from pathlib import Path

import pytest

from verified_memory.pilot_contract import ProviderRequestProfile, load_pilot_contract
from verified_memory.pilot_provider_catalog import (
    PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION,
    ProviderCatalogError,
    validate_live_provider_catalog,
    validate_local_ollama_profile,
    verify_provider_catalog_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_pilot_contract(ROOT / "experiments" / "pilot_v1.yaml")
CONTRACT_V2 = load_pilot_contract(ROOT / "experiments" / "pilot_v2.yaml")


def _hosted_catalogs(*, llama_output_price: float = 0.8) -> dict[str, bytes]:
    values: dict[str, bytes] = {}
    for profile_id in ("gpt52_main", "gpt56_upper"):
        profile = CONTRACT.provider_profiles[profile_id]
        input_price = float(profile.price_snapshot.dispatch_input)
        output_price = float(profile.price_snapshot.dispatch_output)
        values[profile.price_snapshot.source] = (
            f"{profile.requested_model} {dict(profile.artifact_identity)['served_snapshot']} "
            f"${input_price:g} ${output_price:g} Chat Completions"
        ).encode()
    for profile_id in (
        "opus48_sentinel",
        "gemini35_flash_sentinel",
        "llama4_maverick_sentinel",
    ):
        profile = CONTRACT.provider_profiles[profile_id]
        identity = dict(profile.artifact_identity)
        output_price = (
            llama_output_price
            if profile_id == "llama4_maverick_sentinel"
            else float(profile.price_snapshot.dispatch_output)
        )
        supported = [
            "max_tokens",
            "temperature",
            "top_p",
            "response_format",
        ]
        if profile.seed_capability != "unsupported":
            supported.append("seed")
        if profile.reasoning.mode == "fixed":
            supported.append("reasoning")
        endpoint = {
            "provider_name": profile.provider_pin[0],
            "tag": identity["endpoint_tag"],
            "name": f"{profile.provider_pin[0]} | {identity['served_snapshot']}",
            "quantization": identity.get("quantization", "unknown"),
            "supported_parameters": supported,
            "pricing": {
                "prompt": str(
                    float(profile.price_snapshot.dispatch_input) / 1_000_000
                ),
                "completion": str(output_price / 1_000_000),
            },
        }
        values[profile.price_snapshot.source] = json.dumps(
            {"data": {"endpoints": [endpoint]}}
        ).encode()
    return values


def test_live_catalog_gate_binds_hosted_routes_snapshots_parameters_and_prices():
    catalogs = _hosted_catalogs()
    receipt = validate_live_provider_catalog(
        CONTRACT,
        fetch_bytes=catalogs.__getitem__,
        model_ids=(
            "gpt52_main",
            "gpt56_upper",
            "opus48_sentinel",
            "gemini35_flash_sentinel",
            "llama4_maverick_sentinel",
        ),
    )

    assert receipt["schema_version"] == PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION
    assert receipt["contract_sha256"] == CONTRACT.canonical_hash
    assert receipt["status"] == "pass"
    assert receipt["paid_completions"] == 0
    assert len(receipt["rows"]) == 5
    assert len(receipt["receipt_sha256"]) == 64
    llama = next(
        row
        for row in receipt["rows"]
        if row["profile_id"] == "llama4_maverick_sentinel"
    )
    assert llama["provider_name"] == "DeepInfra"
    assert llama["endpoint_tag"] == "deepinfra/base"
    assert llama["quantization"] == "fp8"
    assert llama["live_input_per_million_usd"] == 0.2
    assert llama["live_output_per_million_usd"] == 0.8
    assert (
        verify_provider_catalog_receipt(
            receipt,
            contract_hash=CONTRACT.canonical_hash,
        )
        == receipt
    )

    tampered = json.loads(json.dumps(receipt))
    tampered["rows"][0]["status"] = "no-go"
    with pytest.raises(ProviderCatalogError, match="hash mismatch"):
        verify_provider_catalog_receipt(
            tampered,
            contract_hash=CONTRACT.canonical_hash,
        )


def test_live_catalog_gate_stops_on_price_drift_before_dispatch():
    catalogs = _hosted_catalogs(llama_output_price=0.9)
    with pytest.raises(ProviderCatalogError, match="price differs"):
        validate_live_provider_catalog(
            CONTRACT,
            fetch_bytes=catalogs.__getitem__,
            model_ids=("llama4_maverick_sentinel",),
        )


@pytest.mark.parametrize("field", ["prompt", "completion"])
def test_openrouter_live_catalog_rejects_zero_token_price(field: str):
    profile = CONTRACT.provider_profiles["llama4_maverick_sentinel"]
    catalogs = _hosted_catalogs()
    payload = json.loads(catalogs[profile.price_snapshot.source])
    payload["data"]["endpoints"][0]["pricing"][field] = "0"
    catalogs[profile.price_snapshot.source] = json.dumps(payload).encode()

    with pytest.raises(ProviderCatalogError, match="finite and positive"):
        validate_live_provider_catalog(
            CONTRACT,
            fetch_bytes=catalogs.__getitem__,
            model_ids=("llama4_maverick_sentinel",),
        )


def _v2_openrouter_catalog(profile_id: str, supported: list[str]) -> bytes:
    profile = CONTRACT_V2.provider_profiles[profile_id]
    identity = dict(profile.artifact_identity)
    endpoint = {
        "provider_name": profile.provider_pin[0],
        "tag": identity["endpoint_tag"],
        "name": f"{profile.provider_pin[0]} | {identity['served_snapshot']}",
        "quantization": identity.get("quantization", "unknown"),
        "supported_parameters": supported,
        "pricing": {
            "prompt": str(
                float(profile.price_snapshot.dispatch_input) / 1_000_000
            ),
            "completion": str(
                float(profile.price_snapshot.dispatch_output) / 1_000_000
            ),
        },
    }
    return json.dumps({"data": {"endpoints": [endpoint]}}).encode()


def test_v2_catalog_enforces_explicit_and_omitted_parameter_policy():
    explicit = [
        "max_tokens",
        "temperature",
        "top_p",
        "seed",
        "reasoning",
        "response_format",
    ]
    profile = CONTRACT_V2.provider_profiles["gemini35_flash_diagnostic"]
    receipt = validate_live_provider_catalog(
        CONTRACT_V2,
        fetch_bytes=lambda url: _v2_openrouter_catalog(
            "gemini35_flash_diagnostic", explicit
        ),
        model_ids=("gemini35_flash_diagnostic",),
    )
    row = receipt["rows"][0]
    assert row["required_parameters"] == sorted(explicit)
    assert row["omitted_unsupported_parameters"] == []

    with pytest.raises(ProviderCatalogError, match="explicit-supported.*seed"):
        validate_live_provider_catalog(
            CONTRACT_V2,
            fetch_bytes=lambda url: _v2_openrouter_catalog(
                "gemini35_flash_diagnostic",
                [field for field in explicit if field != "seed"],
            ),
            model_ids=(profile.profile_id,),
        )

    llama_supported = [
        "max_tokens",
        "temperature",
        "top_p",
        "seed",
        "response_format",
    ]
    llama_receipt = validate_live_provider_catalog(
        CONTRACT_V2,
        fetch_bytes=lambda url: _v2_openrouter_catalog(
            "llama4_maverick_diagnostic", llama_supported
        ),
        model_ids=("llama4_maverick_diagnostic",),
    )
    assert llama_receipt["rows"][0]["omitted_unsupported_parameters"] == [
        "reasoning"
    ]
    with pytest.raises(ProviderCatalogError, match="contradicts.*reasoning"):
        validate_live_provider_catalog(
            CONTRACT_V2,
            fetch_bytes=lambda url: _v2_openrouter_catalog(
                "llama4_maverick_diagnostic",
                [*llama_supported, "reasoning"],
            ),
            model_ids=("llama4_maverick_diagnostic",),
        )


def test_opus_profile_is_catalog_attested_but_frozen_budget_no_go():
    supported = ["max_tokens", "reasoning", "response_format"]
    receipt = validate_live_provider_catalog(
        CONTRACT_V2,
        fetch_bytes=lambda url: _v2_openrouter_catalog(
            "opus48_no_go", supported
        ),
        model_ids=("opus48_no_go",),
    )
    assert receipt["rows"][0]["omitted_unsupported_parameters"] == [
        "seed",
        "temperature",
        "top_p",
    ]
    profile = CONTRACT_V2.provider_profiles["opus48_no_go"]
    assert profile.dispatch_eligible is False
    assert profile.ineligibility_reason == (
        "cross_model_budget_no_go_under_nonshrink_policy"
    )


def test_local_manifest_and_model_layer_are_hash_and_size_bound(tmp_path: Path):
    model_bytes = b"frozen-local-model-fixture"
    model_digest = hashlib.sha256(model_bytes).hexdigest()
    manifest = {
        "schemaVersion": 2,
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": f"sha256:{model_digest}",
                "size": len(model_bytes),
            }
        ],
    }
    raw_manifest = json.dumps(manifest, sort_keys=True).encode()
    manifest_hash = hashlib.sha256(raw_manifest).hexdigest()
    manifest_path = (
        tmp_path
        / "manifests"
        / "registry.ollama.ai"
        / "library"
        / "fixture"
        / "tag"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(raw_manifest)
    blob = tmp_path / "blobs" / f"sha256-{model_digest}"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(model_bytes)

    payload = CONTRACT.provider_profiles["llama33_local_sentinel"].to_dict()
    payload["profile_id"] = "local-fixture"
    payload["requested_model"] = "fixture:tag"
    payload["served_model"] = "fixture:tag"
    payload["artifact_identity"] = {
        "manifest_sha256": manifest_hash,
        "model_layer_digest": f"sha256:{model_digest}",
    }
    profile = ProviderRequestProfile.from_dict(payload)
    row = validate_local_ollama_profile(profile, model_root=tmp_path)

    assert row["manifest_sha256"] == manifest_hash
    assert row["model_layer_digest"] == f"sha256:{model_digest}"
    assert row["model_layer_size"] == len(model_bytes)

    blob.write_bytes(b"truncated")
    with pytest.raises(ProviderCatalogError, match="absent or truncated"):
        validate_local_ollama_profile(profile, model_root=tmp_path)


def test_v2_local_profile_binds_frozen_layer_size(tmp_path: Path):
    model_bytes = b"v2-local-model-fixture"
    model_digest = hashlib.sha256(model_bytes).hexdigest()
    manifest = {
        "schemaVersion": 2,
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": f"sha256:{model_digest}",
                "size": len(model_bytes),
            }
        ],
    }
    raw_manifest = json.dumps(manifest, sort_keys=True).encode()
    manifest_path = (
        tmp_path
        / "manifests"
        / "registry.ollama.ai"
        / "library"
        / "fixture"
        / "tag"
    )
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(raw_manifest)
    blob = tmp_path / "blobs" / f"sha256-{model_digest}"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(model_bytes)

    payload = CONTRACT_V2.provider_profiles[
        "llama33_local_controlled"
    ].to_dict()
    payload.update(
        {
            "profile_id": "local-v2-fixture",
            "requested_model": "fixture:tag",
            "served_model": "fixture:tag",
            "artifact_identity": {
                "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                "model_layer_digest": f"sha256:{model_digest}",
                "model_layer_size_bytes": str(len(model_bytes)),
                "ollama_version": "0.15.4",
                "adapter": "ollama-python",
                "base_url": "http://127.0.0.1:11434",
            },
        }
    )
    profile = ProviderRequestProfile.from_dict(payload)
    row = validate_local_ollama_profile(profile, model_root=tmp_path)
    assert row["model_layer_size"] == len(model_bytes)
    assert row["runtime_ollama_version"] is None
    assert row["parameter_dispatch_policy"][
        "omitted_unsupported_parameters"
    ] == ["reasoning"]
    live_row = validate_local_ollama_profile(
        profile,
        model_root=tmp_path,
        fetch_bytes=lambda url: json.dumps(
            {"version": "0.15.4"}
        ).encode(),
    )
    assert live_row["runtime_ollama_version"] == "0.15.4"
    with pytest.raises(ProviderCatalogError, match="runtime version differs"):
        validate_local_ollama_profile(
            profile,
            model_root=tmp_path,
            fetch_bytes=lambda url: json.dumps(
                {"version": "0.15.5"}
            ).encode(),
        )

    payload["artifact_identity"]["model_layer_size_bytes"] = str(
        len(model_bytes) + 1
    )
    drifted = ProviderRequestProfile.from_dict(payload)
    with pytest.raises(ProviderCatalogError, match="size differs"):
        validate_local_ollama_profile(drifted, model_root=tmp_path)
