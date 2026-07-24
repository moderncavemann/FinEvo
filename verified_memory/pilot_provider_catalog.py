"""Live, no-completion provider identity and price gates for the pilot.

The frozen contract records a price and route snapshot.  Immediately before a
paid stage, this module re-reads the public model catalogs and fails closed if
the selected endpoint, served snapshot, required parameters, or prices have
drifted.  Catalog reads never send a model completion.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import urllib.request

from .pilot_contract import PilotContract, ProviderRequestProfile


PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION = "finevo-provider-catalog-receipt-v1"


class ProviderCatalogError(RuntimeError):
    """Raised before dispatch when a frozen provider identity cannot be proven."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _default_fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FinEvo-pilot-v1-catalog-gate/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _finite_price(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ProviderCatalogError(f"{name} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderCatalogError(f"{name} is not numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise ProviderCatalogError(f"{name} is not finite and positive")
    return result


def _prices_match(profile: ProviderRequestProfile, endpoint: Mapping[str, Any]) -> bool:
    pricing = endpoint.get("pricing")
    if not isinstance(pricing, Mapping):
        return False
    live_input = _finite_price(pricing.get("prompt"), "live prompt price") * 1_000_000
    live_output = (
        _finite_price(pricing.get("completion"), "live completion price") * 1_000_000
    )
    return math.isclose(
        live_input,
        float(profile.price_snapshot.dispatch_input),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) and math.isclose(
        live_output,
        float(profile.price_snapshot.dispatch_output),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _decoding_catalog_policy(
    profile: ProviderRequestProfile,
    supported_parameters: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Resolve the frozen per-field wire policy against catalog evidence.

    V1 profiles predate the explicit field policy and retain their historical
    required-parameter behavior.  V2 profiles distinguish parameters that must
    be sent from documented unsupported parameters that must be absent from the
    wire.  When a V2 field requires catalog evidence, an OpenRouter endpoint
    declaration must agree in both directions; unknown support therefore fails
    before a completion can be dispatched.
    """

    if not profile.decoding_fields:
        required = {"max_tokens", "temperature", "top_p", "response_format"}
        if profile.seed_capability != "unsupported":
            required.add("seed")
        if profile.reasoning.mode == "fixed":
            required.add("reasoning")
        return {
            "required_parameters": sorted(required),
            "omitted_unsupported_parameters": [],
            "catalog_evidence_parameters": sorted(required),
        }

    explicit: set[str] = set()
    omitted: set[str] = set()
    evidence: set[str] = set()
    for field, disposition in profile.decoding_fields:
        if disposition.dispatch_mode == "explicit_supported":
            explicit.add(field)
        else:
            omitted.add(field)
        if disposition.catalog_evidence_required:
            evidence.add(field)

    required = {"max_tokens", *explicit}
    if supported_parameters is not None:
        supported = {str(item) for item in supported_parameters}
        missing = sorted((explicit & evidence) - supported)
        contradicted = sorted((omitted & evidence) & supported)
        if missing:
            raise ProviderCatalogError(
                f"{profile.profile_id} endpoint lacks explicit-supported "
                f"parameters: {missing}"
            )
        if contradicted:
            raise ProviderCatalogError(
                f"{profile.profile_id} endpoint catalog contradicts frozen "
                f"unsupported omissions: {contradicted}"
            )

    return {
        "required_parameters": sorted(required),
        "omitted_unsupported_parameters": sorted(omitted),
        "catalog_evidence_parameters": sorted(evidence),
    }


def _openrouter_row(
    profile: ProviderRequestProfile,
    *,
    fetch_bytes: Callable[[str], bytes],
) -> dict[str, Any]:
    identity = dict(profile.artifact_identity)
    endpoint_tag = identity.get("endpoint_tag")
    served_snapshot = identity.get("served_snapshot")
    if not endpoint_tag or not served_snapshot:
        raise ProviderCatalogError(
            f"{profile.profile_id} lacks endpoint_tag or served_snapshot"
        )
    url = profile.price_snapshot.source
    raw = fetch_bytes(url)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderCatalogError(
            f"{profile.profile_id} endpoint catalog is not JSON"
        ) from exc
    data = payload.get("data") if isinstance(payload, Mapping) else None
    endpoints = data.get("endpoints") if isinstance(data, Mapping) else None
    if not isinstance(endpoints, list):
        raise ProviderCatalogError(
            f"{profile.profile_id} endpoint catalog has no endpoints"
        )
    matches = [
        endpoint
        for endpoint in endpoints
        if isinstance(endpoint, Mapping) and endpoint.get("tag") == endpoint_tag
    ]
    if len(matches) != 1:
        raise ProviderCatalogError(
            f"{profile.profile_id} expected one endpoint tag {endpoint_tag!r}; "
            f"observed {len(matches)}"
        )
    endpoint = matches[0]
    provider_name = endpoint.get("provider_name")
    if provider_name not in profile.provider_pin:
        raise ProviderCatalogError(
            f"{profile.profile_id} endpoint provider does not match its pin"
        )
    endpoint_name = str(endpoint.get("name") or "")
    if not endpoint_name.endswith(served_snapshot):
        raise ProviderCatalogError(
            f"{profile.profile_id} served snapshot changed: {endpoint_name!r}"
        )
    if not _prices_match(profile, endpoint):
        raise ProviderCatalogError(
            f"{profile.profile_id} live endpoint price differs from the contract"
        )
    supported = endpoint.get("supported_parameters")
    if not isinstance(supported, list):
        raise ProviderCatalogError(
            f"{profile.profile_id} has no supported-parameter declaration"
        )
    dispatch_policy = _decoding_catalog_policy(profile, supported)
    required = set(dispatch_policy["required_parameters"])
    missing = sorted(required - {str(item) for item in supported})
    if missing:
        raise ProviderCatalogError(
            f"{profile.profile_id} endpoint lacks required parameters: {missing}"
        )
    frozen_quantization = identity.get("quantization")
    if (
        frozen_quantization is not None
        and endpoint.get("quantization") != frozen_quantization
    ):
        raise ProviderCatalogError(
            f"{profile.profile_id} endpoint quantization changed"
        )
    pricing = endpoint["pricing"]
    return {
        "profile_id": profile.profile_id,
        "transport": profile.transport,
        "status": "pass",
        "catalog_url": url,
        "catalog_sha256": _sha256_bytes(raw),
        "provider_name": provider_name,
        "endpoint_tag": endpoint_tag,
        "served_snapshot": served_snapshot,
        "endpoint_name": endpoint_name,
        "quantization": endpoint.get("quantization"),
        "live_input_per_million_usd": (
            _finite_price(pricing["prompt"], "live prompt price") * 1_000_000
        ),
        "live_output_per_million_usd": (
            _finite_price(pricing["completion"], "live completion price") * 1_000_000
        ),
        **dispatch_policy,
    }


def _price_markers(profile: ProviderRequestProfile) -> tuple[str, str]:
    input_price = float(profile.price_snapshot.dispatch_input)
    output_price = float(profile.price_snapshot.dispatch_output)
    return (f"${input_price:g}", f"${output_price:g}")


def _openai_row(
    profile: ProviderRequestProfile,
    *,
    fetch_bytes: Callable[[str], bytes],
) -> dict[str, Any]:
    identity = dict(profile.artifact_identity)
    snapshot = identity.get("served_snapshot")
    if not snapshot or snapshot != profile.served_model:
        raise ProviderCatalogError(
            f"{profile.profile_id} lacks an exact direct served snapshot"
        )
    url = profile.price_snapshot.source
    raw = fetch_bytes(url)
    text = raw.decode("utf-8", "replace")
    input_marker, output_marker = _price_markers(profile)
    checks = {
        "model_id_present": profile.requested_model in text,
        "snapshot_present": snapshot in text,
        "input_price_present": input_marker in text,
        "output_price_present": output_marker in text,
        "chat_completions_present": "Chat Completions" in text,
    }
    if not all(checks.values()):
        failed = sorted(key for key, passed in checks.items() if not passed)
        raise ProviderCatalogError(
            f"{profile.profile_id} official model page drifted: {failed}"
        )
    return {
        "profile_id": profile.profile_id,
        "transport": profile.transport,
        "status": "pass",
        "catalog_url": url,
        "catalog_sha256": _sha256_bytes(raw),
        "provider_name": "OpenAI-direct",
        "served_snapshot": snapshot,
        "live_input_per_million_usd": float(
            profile.price_snapshot.dispatch_input
        ),
        "live_output_per_million_usd": float(
            profile.price_snapshot.dispatch_output
        ),
        "document_checks": checks,
        "parameter_dispatch_policy": _decoding_catalog_policy(profile),
    }


def _ollama_manifest_path(profile: ProviderRequestProfile, model_root: Path) -> Path:
    namespace, separator, tag = profile.requested_model.partition(":")
    if not separator:
        tag = "latest"
    parts = namespace.split("/")
    return model_root / "manifests" / "registry.ollama.ai" / "library" / Path(
        *parts
    ) / tag


def _ollama_row(
    profile: ProviderRequestProfile,
    *,
    model_root: Path,
    runtime_fetch_bytes: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    identity = dict(profile.artifact_identity)
    manifest_path = _ollama_manifest_path(profile, model_root)
    try:
        raw = manifest_path.read_bytes()
    except FileNotFoundError as exc:
        raise ProviderCatalogError(
            f"frozen local manifest is absent: {manifest_path}"
        ) from exc
    manifest_hash = _sha256_bytes(raw)
    if manifest_hash != identity.get("manifest_sha256"):
        raise ProviderCatalogError("local Ollama manifest hash changed")
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderCatalogError("local Ollama manifest is not JSON") from exc
    layers = manifest.get("layers") if isinstance(manifest, Mapping) else None
    if not isinstance(layers, list):
        raise ProviderCatalogError("local Ollama manifest has no layers")
    model_layers = [
        row
        for row in layers
        if isinstance(row, Mapping)
        and row.get("mediaType") == "application/vnd.ollama.image.model"
    ]
    if len(model_layers) != 1:
        raise ProviderCatalogError("local manifest requires exactly one model layer")
    layer = model_layers[0]
    digest = str(layer.get("digest") or "")
    if digest != identity.get("model_layer_digest"):
        raise ProviderCatalogError("local model-layer digest changed")
    if not digest.startswith("sha256:"):
        raise ProviderCatalogError("local model-layer digest is malformed")
    blob = model_root / "blobs" / f"sha256-{digest.split(':', 1)[1]}"
    if not blob.is_file() or blob.stat().st_size != int(layer.get("size", -1)):
        raise ProviderCatalogError("local model-layer blob is absent or truncated")
    frozen_size = identity.get("model_layer_size_bytes")
    if frozen_size is not None and blob.stat().st_size != int(frozen_size):
        raise ProviderCatalogError("local model-layer size differs from the contract")
    runtime_version = None
    if runtime_fetch_bytes is not None:
        base_url = identity.get("base_url")
        expected_version = identity.get("ollama_version")
        if not base_url or not expected_version:
            raise ProviderCatalogError(
                "local runtime check requires frozen base_url and Ollama version"
            )
        version_url = f"{base_url.rstrip('/')}/api/version"
        try:
            version_payload = json.loads(runtime_fetch_bytes(version_url))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProviderCatalogError(
                "local Ollama runtime version endpoint is unavailable"
            ) from exc
        if not isinstance(version_payload, Mapping):
            raise ProviderCatalogError(
                "local Ollama runtime version payload is invalid"
            )
        runtime_version = str(version_payload.get("version") or "")
        if runtime_version != expected_version:
            raise ProviderCatalogError(
                "local Ollama runtime version differs from the contract"
            )
    return {
        "profile_id": profile.profile_id,
        "transport": profile.transport,
        "status": "pass",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "model_layer_digest": digest,
        "model_layer_size": blob.stat().st_size,
        "served_snapshot": profile.served_model,
        "ollama_version": identity.get("ollama_version"),
        "runtime_ollama_version": runtime_version,
        "adapter": identity.get("adapter"),
        "base_url": identity.get("base_url"),
        "parameter_dispatch_policy": _decoding_catalog_policy(profile),
    }


def validate_live_provider_catalog(
    contract: PilotContract,
    *,
    fetch_bytes: Callable[[str], bytes] = _default_fetch_bytes,
    model_root: str | Path | None = None,
    model_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate every non-diagnostic model and return a hashable gate receipt."""

    if not isinstance(contract, PilotContract):
        raise TypeError("contract must be a PilotContract")
    root = (
        Path.home() / ".ollama" / "models"
        if model_root is None
        else Path(model_root)
    )
    selected = (
        tuple(contract.provider_profiles)
        if model_ids is None
        else tuple(str(item) for item in model_ids)
    )
    if not selected or len(selected) != len(set(selected)):
        raise ProviderCatalogError("model_ids must be non-empty and unique")
    unknown = sorted(set(selected) - set(contract.provider_profiles))
    if unknown:
        raise ProviderCatalogError(f"unknown provider profiles: {unknown}")
    rows = []
    for profile_id in selected:
        profile = contract.provider_profiles[profile_id]
        if profile.transport == "diagnostic":
            continue
        if profile.transport == "openrouter":
            row = _openrouter_row(profile, fetch_bytes=fetch_bytes)
        elif profile.transport == "openai":
            row = _openai_row(profile, fetch_bytes=fetch_bytes)
        elif profile.transport == "ollama":
            row = _ollama_row(
                profile,
                model_root=root,
                runtime_fetch_bytes=fetch_bytes,
            )
        else:  # pragma: no cover - contract validation owns the transport enum
            raise ProviderCatalogError(
                f"unsupported live catalog transport {profile.transport!r}"
            )
        rows.append(row)
    receipt = {
        "schema_version": PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION,
        "captured_at": _utc_now(),
        "contract_sha256": contract.canonical_hash,
        "status": "pass",
        "paid_completions": 0,
        "rows": rows,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return receipt


def verify_provider_catalog_receipt(
    value: Mapping[str, Any],
    *,
    contract_hash: str | None = None,
    require_pass: bool = True,
) -> dict[str, Any]:
    """Verify a persisted zero-completion catalog receipt without refetching.

    Catalog receipts are launch controls, not mutable caches.  A resumed stage
    may reuse the exact bytes only after this self-hash and contract binding
    pass; it must never silently overwrite a prior observation.
    """

    if not isinstance(value, Mapping):
        raise ProviderCatalogError("provider catalog receipt must be an object")
    payload = dict(value)
    expected = payload.pop("receipt_sha256", None)
    if expected != _sha256_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ):
        raise ProviderCatalogError("provider catalog receipt hash mismatch")
    if value.get("schema_version") != PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION:
        raise ProviderCatalogError("unsupported provider catalog receipt schema")
    if contract_hash is not None and value.get("contract_sha256") != contract_hash:
        raise ProviderCatalogError("provider catalog receipt contract mismatch")
    if value.get("paid_completions") != 0:
        raise ProviderCatalogError(
            "provider catalog receipt must have zero paid completions"
        )
    if require_pass and value.get("status") != "pass":
        raise ProviderCatalogError("provider catalog receipt is not passing")
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ProviderCatalogError("provider catalog receipt rows must be an array")
    result = json.loads(
        json.dumps(value, sort_keys=True, allow_nan=False)
    )
    return result


def validate_local_ollama_profile(
    profile: ProviderRequestProfile,
    *,
    model_root: str | Path,
    fetch_bytes: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    """Validate one local profile without constructing a full pilot contract."""

    if not isinstance(profile, ProviderRequestProfile):
        raise TypeError("profile must be a ProviderRequestProfile")
    if profile.transport != "ollama":
        raise ProviderCatalogError("profile is not an Ollama profile")
    return _ollama_row(
        profile,
        model_root=Path(model_root),
        runtime_fetch_bytes=fetch_bytes,
    )


__all__ = [
    "PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION",
    "ProviderCatalogError",
    "validate_local_ollama_profile",
    "validate_live_provider_catalog",
    "verify_provider_catalog_receipt",
]
