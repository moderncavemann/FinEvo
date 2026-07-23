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
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_pilot_contract(ROOT / "experiments" / "pilot_v1.yaml")


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


def test_live_catalog_gate_stops_on_price_drift_before_dispatch():
    catalogs = _hosted_catalogs(llama_output_price=0.9)
    with pytest.raises(ProviderCatalogError, match="price differs"):
        validate_live_provider_catalog(
            CONTRACT,
            fetch_bytes=catalogs.__getitem__,
            model_ids=("llama4_maverick_sentinel",),
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
