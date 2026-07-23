"""Frozen, hash-bound execution contract for the FinEvo mechanism pilot.

The contract is deliberately independent from the simulation runner.  It
validates the pre-registered model/request identities, seed registry, stage
matrix, interventions, budgets, and stop/go rules before a paid provider can
be constructed.  ``experiments/pilot_v1.yaml`` is JSON-compatible YAML and is
therefore parsed with the standard library rather than adding a configuration
dependency to the execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence


PILOT_CONTRACT_SCHEMA_VERSION = "finevo-pilot-contract-v1"
PILOT_CONTRACT_CANONICALIZATION = "json-sort-keys-utf8-v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SEED_CAPABILITIES = frozenset({"best_effort", "unsupported", "deterministic"})
_TRANSPORTS = frozenset({"openai", "openrouter", "ollama", "diagnostic"})
_JSON_MODES = frozenset({"json_object", "prompt_only"})
_REASONING_MODES = frozenset({"fixed", "omitted"})
_REASONING_EFFORTS = frozenset(
    {"max", "xhigh", "high", "medium", "low", "minimal", "none"}
)

PILOT_V1_ACTION_GRID = {
    "labor_step_hours": 8.0,
    "consumption_step": 0.02,
}
PILOT_V1_NARRATIVE_FIXTURES = {
    "none": "",
    "aligned": (
        "Savings now earn a high interest rate. Preserving cash and reducing "
        "current consumption is directionally consistent with this environment."
    ),
    "paraphrase": (
        "Returns on saved funds are elevated, so conserving available cash and "
        "lowering near-term spending points in the same direction."
    ),
    "opposite": (
        "A high interest rate makes immediate spending more attractive; increase "
        "current consumption rather than preserving cash."
    ),
}
PILOT_V1_SENSITIVITY_WEIGHTS = (0.25, 0.50, 0.75)
PILOT_V1_SENSITIVITY_OUTCOMES = (
    "utility_advantage_positive",
    "absolute_flow_utility",
    "three_period_cumulative_advantage_positive",
)


class PilotContractError(ValueError):
    """Raised when a pilot contract or runtime binding is not exact."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of canonical UTF-8 JSON."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _freeze_json(value: Any) -> Any:
    copied = _json_copy(value)

    def freeze(item: Any) -> Any:
        if isinstance(item, dict):
            return MappingProxyType({str(key): freeze(val) for key, val in item.items()})
        if isinstance(item, list):
            return tuple(freeze(val) for val in item)
        return item

    return freeze(copied)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotContractError(f"{name} must be an object")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: frozenset[str] = frozenset(),
    name: str,
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    extra = sorted(actual - required - optional)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise PilotContractError(f"invalid {name} keys: {', '.join(details)}")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PilotContractError(f"{name} must be a non-empty string")
    return value.strip()


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PilotContractError(f"{name} must be boolean")
    return value


def _integer(
    value: Any,
    name: str,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PilotContractError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise PilotContractError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise PilotContractError(f"{name} must be <= {maximum}")
    return result


def _optional_number(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PilotContractError(f"{name} must be numeric or null")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise PilotContractError(f"{name} must be finite and nonnegative")
    return result


def _string_tuple(value: Any, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PilotContractError(f"{name} must be an array")
    result = tuple(_text(item, f"{name} item") for item in value)
    if not result and not allow_empty:
        raise PilotContractError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise PilotContractError(f"{name} contains duplicates")
    return result


def _sha256(value: Any, name: str) -> str:
    result = _text(value, name).lower()
    if not _SHA256_RE.fullmatch(result):
        raise PilotContractError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _git_commit(value: Any, name: str) -> str:
    result = _text(value, name).lower()
    if not _GIT_COMMIT_RE.fullmatch(result):
        raise PilotContractError(f"{name} must be a lowercase 40-character commit")
    return result


@dataclass(frozen=True, slots=True)
class ReasoningProfile:
    """Frozen reasoning request semantics."""

    mode: str
    effort: Optional[str] = None
    exclude: bool = True

    def __post_init__(self) -> None:
        mode = _text(self.mode, "reasoning.mode")
        if mode not in _REASONING_MODES:
            raise PilotContractError(f"unsupported reasoning mode: {mode}")
        object.__setattr__(self, "mode", mode)
        _boolean(self.exclude, "reasoning.exclude")
        if mode == "fixed":
            effort = _text(self.effort, "reasoning.effort")
            if effort not in _REASONING_EFFORTS:
                raise PilotContractError(f"unsupported reasoning effort: {effort}")
            object.__setattr__(self, "effort", effort)
        elif self.effort is not None:
            raise PilotContractError("omitted reasoning cannot declare an effort")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReasoningProfile":
        value = _mapping(value, "reasoning")
        _strict_keys(
            value,
            required={"mode", "effort", "exclude"},
            name="reasoning",
        )
        return cls(
            mode=value["mode"],
            effort=value["effort"],
            exclude=value["exclude"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "effort": self.effort,
            "exclude": self.exclude,
        }

    def openrouter_payload(self) -> Optional[dict[str, Any]]:
        if self.mode == "omitted":
            return None
        return {"effort": self.effort, "exclude": self.exclude}


@dataclass(frozen=True, slots=True)
class PriceSnapshot:
    """Frozen catalog and dispatch-endpoint prices in USD per million tokens."""

    captured_at: str
    source: str
    currency: str
    unit: str
    dispatch_basis: str
    catalog_input: Optional[float]
    catalog_output: Optional[float]
    catalog_cached_input: Optional[float]
    endpoint_input: Optional[float]
    endpoint_output: Optional[float]
    endpoint_cached_input: Optional[float]

    def __post_init__(self) -> None:
        for name in ("captured_at", "source", "currency", "unit", "dispatch_basis"):
            object.__setattr__(self, name, _text(getattr(self, name), f"price.{name}"))
        if self.currency != "USD":
            raise PilotContractError("price currency must be USD")
        if self.unit != "per_million_tokens":
            raise PilotContractError("price unit must be per_million_tokens")
        if self.dispatch_basis not in {"catalog", "endpoint"}:
            raise PilotContractError("price dispatch_basis must be catalog or endpoint")
        for name in (
            "catalog_input",
            "catalog_output",
            "catalog_cached_input",
            "endpoint_input",
            "endpoint_output",
            "endpoint_cached_input",
        ):
            object.__setattr__(
                self, name, _optional_number(getattr(self, name), f"price.{name}")
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PriceSnapshot":
        value = _mapping(value, "price_snapshot")
        fields = {
            "captured_at",
            "source",
            "currency",
            "unit",
            "dispatch_basis",
            "catalog_input",
            "catalog_output",
            "catalog_cached_input",
            "endpoint_input",
            "endpoint_output",
            "endpoint_cached_input",
        }
        _strict_keys(value, required=fields, name="price_snapshot")
        return cls(**{field: value[field] for field in fields})

    @property
    def dispatch_input(self) -> Optional[float]:
        return (
            self.endpoint_input
            if self.dispatch_basis == "endpoint"
            else self.catalog_input
        )

    @property
    def dispatch_output(self) -> Optional[float]:
        return (
            self.endpoint_output
            if self.dispatch_basis == "endpoint"
            else self.catalog_output
        )

    @property
    def dispatch_cached_input(self) -> Optional[float]:
        return (
            self.endpoint_cached_input
            if self.dispatch_basis == "endpoint"
            else self.catalog_cached_input
        )

    @property
    def known_for_dispatch(self) -> bool:
        return self.dispatch_input is not None and self.dispatch_output is not None

    def assert_known_for_dispatch(self) -> None:
        if not self.known_for_dispatch:
            raise PilotContractError(
                "provider price is unknown for the frozen dispatch endpoint"
            )

    def costs_per_1k(self) -> dict[str, float]:
        self.assert_known_for_dispatch()
        prompt = float(self.dispatch_input) / 1000.0
        cached = self.dispatch_cached_input
        return {
            "prompt": prompt,
            "cached_prompt": prompt if cached is None else float(cached) / 1000.0,
            "completion": float(self.dispatch_output) / 1000.0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "source": self.source,
            "currency": self.currency,
            "unit": self.unit,
            "dispatch_basis": self.dispatch_basis,
            "catalog_input": self.catalog_input,
            "catalog_output": self.catalog_output,
            "catalog_cached_input": self.catalog_cached_input,
            "endpoint_input": self.endpoint_input,
            "endpoint_output": self.endpoint_output,
            "endpoint_cached_input": self.endpoint_cached_input,
        }


@dataclass(frozen=True, slots=True)
class ProviderRequestProfile:
    """Exact provider/model request identity used by a pilot matrix cell."""

    profile_id: str
    transport: str
    requested_model: str
    served_model: str
    provider_pin: tuple[str, ...]
    routing_mode: str
    seed_capability: str
    reasoning: ReasoningProfile
    json_mode: str
    price_snapshot: PriceSnapshot
    max_attempts: int = 1
    allow_fallbacks: bool = False
    require_parameters: bool = True
    artifact_identity: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "transport",
            "requested_model",
            "served_model",
            "routing_mode",
            "seed_capability",
            "json_mode",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.transport not in _TRANSPORTS:
            raise PilotContractError(f"unsupported transport: {self.transport}")
        if self.seed_capability not in _SEED_CAPABILITIES:
            raise PilotContractError(
                f"unsupported seed capability: {self.seed_capability}"
            )
        if self.json_mode not in _JSON_MODES:
            raise PilotContractError(f"unsupported JSON mode: {self.json_mode}")
        if not isinstance(self.reasoning, ReasoningProfile):
            raise PilotContractError("reasoning must be a ReasoningProfile")
        if not isinstance(self.price_snapshot, PriceSnapshot):
            raise PilotContractError("price_snapshot must be a PriceSnapshot")
        _integer(self.max_attempts, "max_attempts", minimum=1, maximum=1)
        _boolean(self.allow_fallbacks, "allow_fallbacks")
        _boolean(self.require_parameters, "require_parameters")
        pins = tuple(_text(item, "provider_pin item") for item in self.provider_pin)
        if len(pins) != len(set(pins)):
            raise PilotContractError("provider_pin contains duplicates")
        object.__setattr__(self, "provider_pin", pins)
        identity = tuple(
            sorted(
                (
                    _text(key, "artifact_identity key"),
                    _text(value, f"artifact_identity[{key}]"),
                )
                for key, value in self.artifact_identity
            )
        )
        if len(identity) != len({key for key, _ in identity}):
            raise PilotContractError("artifact_identity contains duplicate keys")
        object.__setattr__(self, "artifact_identity", identity)

        if self.transport == "openrouter":
            if not self.provider_pin:
                raise PilotContractError("OpenRouter profiles require a provider pin")
            if self.allow_fallbacks:
                raise PilotContractError("OpenRouter pilot profiles forbid fallbacks")
            if not self.require_parameters:
                raise PilotContractError(
                    "OpenRouter pilot profiles require parameter support"
                )
            if self.json_mode != "json_object":
                raise PilotContractError("OpenRouter pilot profiles require JSON mode")
            if self.routing_mode != "standard":
                raise PilotContractError(
                    "OpenRouter pilot profiles require standard non-fast routing"
                )
            if self.requested_model.endswith((":nitro", ":floor")):
                raise PilotContractError(
                    "OpenRouter fast/floor aliases are not permitted in the pilot"
                )
        elif self.routing_mode not in {"direct", "local", "diagnostic"}:
            raise PilotContractError("non-OpenRouter routing mode is invalid")

        if self.transport == "ollama":
            keys = dict(self.artifact_identity)
            if set(keys) != {"manifest_sha256", "model_layer_digest"}:
                raise PilotContractError(
                    "local model profile requires manifest and model-layer digests"
                )
            _sha256(keys["manifest_sha256"], "local manifest_sha256")
            layer = keys["model_layer_digest"]
            if not layer.startswith("sha256:"):
                raise PilotContractError("local model layer digest must use sha256:")
            _sha256(layer.split(":", 1)[1], "local model layer digest")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProviderRequestProfile":
        value = _mapping(value, "provider request profile")
        fields = {
            "profile_id",
            "transport",
            "requested_model",
            "served_model",
            "provider_pin",
            "routing_mode",
            "seed_capability",
            "reasoning",
            "json_mode",
            "price_snapshot",
            "max_attempts",
            "allow_fallbacks",
            "require_parameters",
            "artifact_identity",
        }
        _strict_keys(value, required=fields, name="provider request profile")
        artifact = _mapping(value["artifact_identity"], "artifact_identity")
        return cls(
            profile_id=value["profile_id"],
            transport=value["transport"],
            requested_model=value["requested_model"],
            served_model=value["served_model"],
            provider_pin=_string_tuple(
                value["provider_pin"], "provider_pin", allow_empty=True
            ),
            routing_mode=value["routing_mode"],
            seed_capability=value["seed_capability"],
            reasoning=ReasoningProfile.from_dict(value["reasoning"]),
            json_mode=value["json_mode"],
            price_snapshot=PriceSnapshot.from_dict(value["price_snapshot"]),
            max_attempts=value["max_attempts"],
            allow_fallbacks=value["allow_fallbacks"],
            require_parameters=value["require_parameters"],
            artifact_identity=tuple(
                (str(key), str(item)) for key, item in artifact.items()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "transport": self.transport,
            "requested_model": self.requested_model,
            "served_model": self.served_model,
            "provider_pin": list(self.provider_pin),
            "routing_mode": self.routing_mode,
            "seed_capability": self.seed_capability,
            "reasoning": self.reasoning.to_dict(),
            "json_mode": self.json_mode,
            "price_snapshot": self.price_snapshot.to_dict(),
            "max_attempts": self.max_attempts,
            "allow_fallbacks": self.allow_fallbacks,
            "require_parameters": self.require_parameters,
            "artifact_identity": dict(self.artifact_identity),
        }

    def validate_provider_configuration(
        self,
        *,
        transport: str,
        model: str,
        max_attempts: int,
    ) -> None:
        if transport != self.transport:
            raise PilotContractError(
                f"profile {self.profile_id} requires transport {self.transport}, "
                f"not {transport}"
            )
        if model != self.requested_model:
            raise PilotContractError(
                f"profile {self.profile_id} requested-model mismatch"
            )
        if max_attempts != self.max_attempts:
            raise PilotContractError(
                f"profile {self.profile_id} requires exactly one provider attempt"
            )
        self.price_snapshot.assert_known_for_dispatch()

    def validate_dispatch(
        self,
        *,
        transport: str,
        model: str,
        seed: Optional[int],
        max_attempts: int,
    ) -> None:
        self.validate_provider_configuration(
            transport=transport,
            model=model,
            max_attempts=max_attempts,
        )
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise PilotContractError("decoding seed must be an integer or null")
        if self.seed_capability == "unsupported" and seed is not None:
            raise PilotContractError(
                f"profile {self.profile_id} does not support a decoding seed"
            )
        if self.seed_capability != "unsupported" and seed is None:
            raise PilotContractError(
                f"profile {self.profile_id} requires the frozen decoding seed"
            )

    def validate_served_model(self, served_model: Any) -> str:
        actual = _text(served_model, "served model")
        if actual != self.served_model:
            raise PilotContractError(
                f"served model {actual!r} does not match frozen "
                f"{self.served_model!r}"
            )
        return actual

    def openrouter_request_options(self) -> dict[str, Any]:
        if self.transport != "openrouter":
            raise PilotContractError("profile is not an OpenRouter request")
        provider = {
            "order": list(self.provider_pin),
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        extra_body: dict[str, Any] = {"provider": provider}
        reasoning = self.reasoning.openrouter_payload()
        if reasoning is not None:
            extra_body["reasoning"] = reasoning
        return {
            "response_format": {"type": "json_object"},
            "extra_body": extra_body,
        }

    def openai_request_options(self) -> dict[str, Any]:
        if self.transport != "openai":
            raise PilotContractError("profile is not a direct OpenAI request")
        result: dict[str, Any] = {}
        if self.json_mode == "json_object":
            result["response_format"] = {"type": "json_object"}
        if self.reasoning.mode == "fixed":
            result["reasoning_effort"] = self.reasoning.effort
        return result


@dataclass(frozen=True, slots=True)
class PilotStageCell:
    models: tuple[str, ...]
    arms: tuple[str, ...]
    narratives: tuple[str, ...]
    execution_mode: str = "actor_run"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PilotStageCell":
        value = _mapping(value, "stage cell")
        _strict_keys(
            value,
            required={"models", "arms", "narratives", "execution_mode"},
            name="stage cell",
        )
        return cls(
            models=_string_tuple(value["models"], "stage cell models"),
            arms=_string_tuple(value["arms"], "stage cell arms"),
            narratives=_string_tuple(value["narratives"], "stage cell narratives"),
            execution_mode=_text(value["execution_mode"], "execution_mode"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": list(self.models),
            "arms": list(self.arms),
            "narratives": list(self.narratives),
            "execution_mode": self.execution_mode,
        }


@dataclass(frozen=True, slots=True)
class PilotStage:
    stage_id: str
    enabled: bool
    budget_bucket: str
    num_agents: int
    episode_length: int
    seed_set: str
    utility_profiles: tuple[str, ...]
    shock_id: str
    cells: tuple[PilotStageCell, ...]
    prerequisites: tuple[str, ...] = ()
    reuse: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("stage_id", "budget_bucket", "seed_set", "shock_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        _boolean(self.enabled, "stage.enabled")
        _integer(self.num_agents, "stage.num_agents", minimum=2)
        _integer(self.episode_length, "stage.episode_length", minimum=1)
        if not self.utility_profiles:
            raise PilotContractError("stage utility_profiles must not be empty")
        if not self.cells:
            raise PilotContractError("stage cells must not be empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PilotStage":
        value = _mapping(value, "stage")
        fields = {
            "stage_id",
            "enabled",
            "budget_bucket",
            "num_agents",
            "episode_length",
            "seed_set",
            "utility_profiles",
            "shock_id",
            "cells",
            "prerequisites",
            "reuse",
        }
        _strict_keys(value, required=fields, name="stage")
        cells = value["cells"]
        if isinstance(cells, (str, bytes)) or not isinstance(cells, Sequence):
            raise PilotContractError("stage cells must be an array")
        return cls(
            stage_id=value["stage_id"],
            enabled=value["enabled"],
            budget_bucket=value["budget_bucket"],
            num_agents=value["num_agents"],
            episode_length=value["episode_length"],
            seed_set=value["seed_set"],
            utility_profiles=_string_tuple(
                value["utility_profiles"], "stage utility_profiles"
            ),
            shock_id=value["shock_id"],
            cells=tuple(PilotStageCell.from_dict(cell) for cell in cells),
            prerequisites=_string_tuple(
                value["prerequisites"], "stage prerequisites", allow_empty=True
            ),
            reuse=_string_tuple(value["reuse"], "stage reuse", allow_empty=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "enabled": self.enabled,
            "budget_bucket": self.budget_bucket,
            "num_agents": self.num_agents,
            "episode_length": self.episode_length,
            "seed_set": self.seed_set,
            "utility_profiles": list(self.utility_profiles),
            "shock_id": self.shock_id,
            "cells": [cell.to_dict() for cell in self.cells],
            "prerequisites": list(self.prerequisites),
            "reuse": list(self.reuse),
        }


@dataclass(frozen=True, slots=True)
class PilotRunSpec:
    contract_id: str
    stage_id: str
    model_id: str
    requested_model: str
    arm_id: str
    narrative_id: str
    environment_seed: int
    decoding_seed: Optional[int]
    utility_profile_id: str
    shock_id: str
    budget_bucket: str
    num_agents: int
    episode_length: int
    execution_mode: str

    @property
    def run_id(self) -> str:
        fields = (
            self.contract_id,
            self.stage_id,
            self.model_id,
            self.arm_id,
            self.narrative_id,
            self.utility_profile_id,
            f"s{self.environment_seed}",
        )
        return "--".join(field.replace("/", "_").replace(":", "_") for field in fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "contract_id": self.contract_id,
            "stage_id": self.stage_id,
            "model_id": self.model_id,
            "requested_model": self.requested_model,
            "arm_id": self.arm_id,
            "narrative_id": self.narrative_id,
            "environment_seed": self.environment_seed,
            "decoding_seed": self.decoding_seed,
            "utility_profile_id": self.utility_profile_id,
            "shock_id": self.shock_id,
            "budget_bucket": self.budget_bucket,
            "num_agents": self.num_agents,
            "episode_length": self.episode_length,
            "execution_mode": self.execution_mode,
        }


def _contract_hash_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _json_copy(value)
    integrity = _mapping(payload.get("integrity"), "integrity")
    integrity = dict(integrity)
    integrity.pop("declared_sha256", None)
    payload["integrity"] = integrity
    return payload


def canonical_contract_sha256(value: Mapping[str, Any]) -> str:
    """Hash a contract while excluding its self-declared digest field."""

    return canonical_sha256(_contract_hash_payload(value))


@dataclass(frozen=True, slots=True)
class PilotContract:
    schema_version: str
    contract_id: str
    status: str
    implementation: Mapping[str, Any]
    seeds: Mapping[str, Any]
    provider_profiles: Mapping[str, ProviderRequestProfile]
    arms: Mapping[str, Any]
    narratives: Mapping[str, Any]
    shocks: Mapping[str, Any]
    utility: Mapping[str, Any]
    budgets: Mapping[str, Any]
    stop_go: Mapping[str, Any]
    stages: tuple[PilotStage, ...]
    non_claims: tuple[str, ...]
    canonicalization: str
    declared_sha256: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PilotContract":
        value = _mapping(value, "pilot contract")
        fields = {
            "schema_version",
            "contract_id",
            "status",
            "implementation",
            "seeds",
            "provider_profiles",
            "arms",
            "narratives",
            "shocks",
            "utility",
            "budgets",
            "stop_go",
            "stages",
            "non_claims",
            "integrity",
        }
        _strict_keys(value, required=fields, name="pilot contract")
        if value["schema_version"] != PILOT_CONTRACT_SCHEMA_VERSION:
            raise PilotContractError("unsupported pilot contract schema")
        if value["status"] != "frozen":
            raise PilotContractError("pilot contract status must be frozen")

        implementation = _mapping(value["implementation"], "implementation")
        _strict_keys(
            implementation,
            required={
                "required_git_tag",
                "commit_resolution",
                "required_git_commit",
                "p0_base_commit",
                "require_clean_worktree",
            },
            name="implementation",
        )
        if implementation["commit_resolution"] != "annotated_tag_peel":
            raise PilotContractError(
                "implementation commit_resolution must be annotated_tag_peel"
            )
        _text(implementation["required_git_tag"], "required_git_tag")
        _git_commit(implementation["p0_base_commit"], "p0_base_commit")
        _boolean(implementation["require_clean_worktree"], "require_clean_worktree")
        required_commit = implementation["required_git_commit"]
        if required_commit is not None:
            _git_commit(required_commit, "required_git_commit")
        elif implementation["commit_resolution"] != "annotated_tag_peel":
            raise PilotContractError(
                "null required_git_commit requires annotated_tag_peel"
            )

        profiles_value = _mapping(value["provider_profiles"], "provider_profiles")
        profiles: dict[str, ProviderRequestProfile] = {}
        for profile_id, row in profiles_value.items():
            profile = ProviderRequestProfile.from_dict(
                _mapping(row, f"provider_profiles.{profile_id}")
            )
            if profile.profile_id != profile_id:
                raise PilotContractError(
                    f"provider profile key {profile_id!r} does not match profile_id"
                )
            profile.price_snapshot.assert_known_for_dispatch()
            profiles[profile_id] = profile
        if not profiles:
            raise PilotContractError("provider_profiles must not be empty")

        seeds = _mapping(value["seeds"], "seeds")
        _strict_keys(
            seeds,
            required={"generation", "preflight_seed", "sets"},
            name="seeds",
        )
        generation = _mapping(seeds["generation"], "seeds.generation")
        _strict_keys(
            generation,
            required={"method", "salt", "generated_before_results", "values"},
            name="seeds.generation",
        )
        seed_method = _text(generation["method"], "seed generation method")
        if seed_method not in {"sha256-counter-v1", "user-preregistered-v1"}:
            raise PilotContractError(
                "seed generation method must be sha256-counter-v1 or "
                "user-preregistered-v1"
            )
        salt = _text(generation["salt"], "seed generation salt")
        if not _boolean(
            generation["generated_before_results"], "generated_before_results"
        ):
            raise PilotContractError("seeds must be frozen before results")
        raw_values = generation["values"]
        if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
            raise PilotContractError("seed generation values must be an array")
        main_values = tuple(
            _integer(item, "seed", minimum=0, maximum=2**31 - 2)
            for item in raw_values
        )
        if len(main_values) != 5 or len(set(main_values)) != 5:
            raise PilotContractError("the main pilot requires five unique frozen seeds")
        if seed_method == "sha256-counter-v1":
            derived = tuple(
                int.from_bytes(
                    hashlib.sha256(f"{salt}|{index}".encode("utf-8")).digest()[:8],
                    "big",
                )
                % (2**31 - 1)
                for index in range(5)
            )
            if main_values != derived:
                raise PilotContractError(
                    "frozen seed values do not match their derivation"
                )
        preflight_seed = _integer(
            seeds["preflight_seed"],
            "preflight_seed",
            minimum=0,
            maximum=2**31 - 2,
        )
        if preflight_seed in main_values:
            raise PilotContractError("preflight seed must be distinct from main seeds")
        seed_sets = _mapping(seeds["sets"], "seed sets")
        normalized_seed_sets: dict[str, tuple[int, ...]] = {}
        for set_id, items in seed_sets.items():
            if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
                raise PilotContractError(f"seed set {set_id} must be an array")
            normalized = tuple(
                _integer(item, f"seed_sets.{set_id}", minimum=0, maximum=2**31 - 2)
                for item in items
            )
            if not normalized or len(normalized) != len(set(normalized)):
                raise PilotContractError(f"seed set {set_id} is empty or duplicated")
            normalized_seed_sets[str(set_id)] = normalized
        if set(normalized_seed_sets) != {
            "preflight",
            "q-ref",
            "calibration",
            "main",
            "cross-model",
        }:
            raise PilotContractError("seed sets must match the frozen pilot registry")
        if normalized_seed_sets["preflight"] != (preflight_seed,):
            raise PilotContractError("preflight seed set does not match preflight_seed")
        if normalized_seed_sets["q-ref"] != (preflight_seed,):
            raise PilotContractError("q-ref seed set must reuse the preflight seed")
        if normalized_seed_sets["main"] != main_values:
            raise PilotContractError("main seed set does not match frozen main values")
        if normalized_seed_sets["cross-model"] != main_values[:3]:
            raise PilotContractError(
                "cross-model seed set must use the first three main seeds"
            )
        calibration_values = normalized_seed_sets["calibration"]
        if len(calibration_values) != 2:
            raise PilotContractError("calibration seed set requires exactly two seeds")
        if set(calibration_values) & {*main_values, preflight_seed}:
            raise PilotContractError(
                "calibration seeds must be distinct from preflight and main seeds"
            )

        arms = _mapping(value["arms"], "arms")
        narratives = _mapping(value["narratives"], "narratives")
        shocks = _mapping(value["shocks"], "shocks")
        utility = _mapping(value["utility"], "utility")
        budgets = _mapping(value["budgets"], "budgets")
        stop_go = _mapping(value["stop_go"], "stop_go")
        if not all((arms, narratives, shocks, utility, budgets, stop_go)):
            raise PilotContractError("contract sections must not be empty")
        if budgets.get("completion_scope") != "hosted-api-only":
            raise PilotContractError(
                "pilot provider-completion cap must use hosted-api-only scope"
            )
        for arm_id, arm in arms.items():
            row = _mapping(arm, f"arms.{arm_id}")
            if row.get("arm_id") != arm_id:
                raise PilotContractError(f"arm key {arm_id!r} does not match arm_id")
        for narrative_id, narrative in narratives.items():
            row = _mapping(narrative, f"narratives.{narrative_id}")
            _strict_keys(
                row,
                required={"narrative_id", "relation_to_shock", "text"},
                name=f"narratives.{narrative_id}",
            )
            if row.get("narrative_id") != narrative_id:
                raise PilotContractError(
                    f"narrative key {narrative_id!r} does not match narrative_id"
                )
        narrative_texts = {
            str(narrative_id): row.get("text")
            for narrative_id, narrative in narratives.items()
            for row in (_mapping(narrative, f"narratives.{narrative_id}"),)
        }
        if narrative_texts != PILOT_V1_NARRATIVE_FIXTURES:
            raise PilotContractError(
                "pilot-v1 narrative fixture text drifted from the frozen "
                "continuation intervention"
            )
        for shock_id, shock in shocks.items():
            row = _mapping(shock, f"shocks.{shock_id}")
            if row.get("shock_id") != shock_id:
                raise PilotContractError(
                    f"shock key {shock_id!r} does not match shock_id"
                )
        utility_profiles = _mapping(utility.get("profiles"), "utility.profiles")

        experiment_c = _mapping(
            stop_go.get("experiment_c"), "stop_go.experiment_c"
        )
        sensitivity = _mapping(
            experiment_c.get("zero_api_sensitivity"),
            "stop_go.experiment_c.zero_api_sensitivity",
        )
        _strict_keys(
            sensitivity,
            required={
                "alternative_success_weights",
                "outcome_definitions",
                "absolute_flow_threshold",
                "effectiveness_gate",
                "descriptive_only",
            },
            name="stop_go.experiment_c.zero_api_sensitivity",
        )
        sensitivity_weights = sensitivity["alternative_success_weights"]
        if (
            isinstance(sensitivity_weights, (str, bytes))
            or not isinstance(sensitivity_weights, Sequence)
            or tuple(sensitivity_weights) != PILOT_V1_SENSITIVITY_WEIGHTS
        ):
            raise PilotContractError(
                "pilot-v1 sensitivity weights differ from the frozen 3-cell grid"
            )
        if _string_tuple(
            sensitivity["outcome_definitions"],
            "sensitivity outcome definitions",
        ) != PILOT_V1_SENSITIVITY_OUTCOMES:
            raise PilotContractError(
                "pilot-v1 sensitivity outcomes differ from the frozen 3-cell grid"
            )
        threshold = _mapping(
            sensitivity["absolute_flow_threshold"],
            "stop_go.experiment_c.zero_api_sensitivity.absolute_flow_threshold",
        )
        expected_threshold = {
            "source_stage": "stage0-calibration",
            "source_profile": "selected-profile-only",
            "source_seeds": "all-two-calibration-seeds",
            "field": "flow_utility",
            "aggregation": "median",
            "derived_after_profile_selection": True,
            "treatment_outcomes_inspected": False,
        }
        if dict(threshold) != expected_threshold:
            raise PilotContractError(
                "absolute flow-utility threshold derivation is not the frozen "
                "Stage-0 selected-profile median"
            )
        if (
            _boolean(
                sensitivity["effectiveness_gate"],
                "sensitivity.effectiveness_gate",
            )
            is not False
            or _boolean(
                sensitivity["descriptive_only"],
                "sensitivity.descriptive_only",
            )
            is not True
        ):
            raise PilotContractError(
                "zero-API sensitivity must remain descriptive and outside the "
                "effectiveness gate"
            )

        experiment_d = _mapping(
            stop_go.get("experiment_d"), "stop_go.experiment_d"
        )
        action_grid = _mapping(
            experiment_d.get("action_grid"),
            "stop_go.experiment_d.action_grid",
        )
        if dict(action_grid) != PILOT_V1_ACTION_GRID:
            raise PilotContractError(
                "pilot-v1 Experiment D action grid drifted from the frozen bins"
            )
        fixture_hash = _sha256(
            experiment_d.get("narrative_fixture_hash"),
            "stop_go.experiment_d.narrative_fixture_hash",
        )
        if fixture_hash != canonical_sha256(PILOT_V1_NARRATIVE_FIXTURES):
            raise PilotContractError(
                "pilot-v1 narrative fixture hash does not match the exact texts"
            )

        stages_value = value["stages"]
        if isinstance(stages_value, (str, bytes)) or not isinstance(
            stages_value, Sequence
        ):
            raise PilotContractError("stages must be an array")
        stages = tuple(PilotStage.from_dict(stage) for stage in stages_value)
        stage_ids = tuple(stage.stage_id for stage in stages)
        if len(stage_ids) != len(set(stage_ids)):
            raise PilotContractError("stage IDs must be unique")
        for stage in stages:
            if stage.seed_set not in normalized_seed_sets:
                raise PilotContractError(
                    f"stage {stage.stage_id} references unknown seed set"
                )
            if stage.shock_id not in shocks:
                raise PilotContractError(
                    f"stage {stage.stage_id} references unknown shock"
                )
            if stage.budget_bucket not in _mapping(
                budgets.get("stage_usd_caps"), "budgets.stage_usd_caps"
            ):
                raise PilotContractError(
                    f"stage {stage.stage_id} references unknown budget bucket"
                )
            if not set(stage.utility_profiles) <= set(utility_profiles):
                raise PilotContractError(
                    f"stage {stage.stage_id} references unknown utility profile"
                )
            for prerequisite in stage.prerequisites:
                if prerequisite not in stage_ids:
                    raise PilotContractError(
                        f"stage {stage.stage_id} has unknown prerequisite"
                    )
            for cell in stage.cells:
                if not set(cell.models) <= set(profiles):
                    raise PilotContractError(
                        f"stage {stage.stage_id} references unknown model profile"
                    )
                if not set(cell.arms) <= set(arms):
                    raise PilotContractError(
                        f"stage {stage.stage_id} references unknown arm"
                    )
                if not set(cell.narratives) <= set(narratives):
                    raise PilotContractError(
                        f"stage {stage.stage_id} references unknown narrative"
                    )

        integrity = _mapping(value["integrity"], "integrity")
        _strict_keys(
            integrity,
            required={"canonicalization", "declared_sha256"},
            name="integrity",
        )
        if integrity["canonicalization"] != PILOT_CONTRACT_CANONICALIZATION:
            raise PilotContractError("unsupported contract canonicalization")
        declared = _sha256(integrity["declared_sha256"], "declared_sha256")
        actual = canonical_contract_sha256(value)
        if declared != actual:
            raise PilotContractError(
                f"pilot contract hash mismatch: declared {declared}, actual {actual}"
            )

        non_claims = _string_tuple(value["non_claims"], "non_claims")
        return cls(
            schema_version=value["schema_version"],
            contract_id=_text(value["contract_id"], "contract_id"),
            status=value["status"],
            implementation=_freeze_json(implementation),
            seeds=_freeze_json(
                {
                    **dict(seeds),
                    "sets": {
                        key: list(items) for key, items in normalized_seed_sets.items()
                    },
                }
            ),
            provider_profiles=MappingProxyType(dict(profiles)),
            arms=_freeze_json(arms),
            narratives=_freeze_json(narratives),
            shocks=_freeze_json(shocks),
            utility=_freeze_json(utility),
            budgets=_freeze_json(budgets),
            stop_go=_freeze_json(stop_go),
            stages=stages,
            non_claims=non_claims,
            canonicalization=integrity["canonicalization"],
            declared_sha256=declared,
        )

    @property
    def canonical_hash(self) -> str:
        return canonical_contract_sha256(self.to_dict())

    @property
    def stage_ids(self) -> tuple[str, ...]:
        return tuple(stage.stage_id for stage in self.stages)

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(self.provider_profiles)

    @property
    def arm_ids(self) -> tuple[str, ...]:
        return tuple(self.arms)

    def stage(self, stage_id: str) -> PilotStage:
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(f"unknown pilot stage: {stage_id}")

    def models_for_stage(self, stage_id: str) -> tuple[str, ...]:
        stage = self.stage(stage_id)
        return tuple(
            dict.fromkeys(model for cell in stage.cells for model in cell.models)
        )

    def arms_for_stage(self, stage_id: str) -> tuple[str, ...]:
        stage = self.stage(stage_id)
        return tuple(dict.fromkeys(arm for cell in stage.cells for arm in cell.arms))

    def expand(
        self,
        *,
        stage: Optional[str] = None,
        model: Optional[str] = None,
        arm: Optional[str] = None,
        include_disabled: bool = False,
    ) -> tuple[PilotRunSpec, ...]:
        """Expand the frozen stage/model/arm/seed/utility/narrative matrix."""

        if stage is not None and stage not in self.stage_ids:
            raise KeyError(f"unknown pilot stage: {stage}")
        if model is not None and model not in self.provider_profiles:
            raise KeyError(f"unknown pilot model: {model}")
        if arm is not None and arm not in self.arms:
            raise KeyError(f"unknown pilot arm: {arm}")
        seed_sets = _mapping(self.seeds["sets"], "seed sets")
        result: list[PilotRunSpec] = []
        for stage_spec in self.stages:
            if stage is not None and stage_spec.stage_id != stage:
                continue
            if not stage_spec.enabled and not include_disabled:
                continue
            seeds = tuple(int(item) for item in seed_sets[stage_spec.seed_set])
            for cell in stage_spec.cells:
                for model_id in cell.models:
                    if model is not None and model_id != model:
                        continue
                    profile = self.provider_profiles[model_id]
                    for arm_id in cell.arms:
                        if arm is not None and arm_id != arm:
                            continue
                        arm_row = _mapping(self.arms[arm_id], f"arms.{arm_id}")
                        execution_mode = str(
                            arm_row.get("execution_mode", cell.execution_mode)
                        )
                        for narrative_id in cell.narratives:
                            for utility_id in stage_spec.utility_profiles:
                                for seed_value in seeds:
                                    result.append(
                                        PilotRunSpec(
                                            contract_id=self.contract_id,
                                            stage_id=stage_spec.stage_id,
                                            model_id=model_id,
                                            requested_model=profile.requested_model,
                                            arm_id=arm_id,
                                            narrative_id=narrative_id,
                                            environment_seed=seed_value,
                                            decoding_seed=(
                                                None
                                                if profile.seed_capability
                                                == "unsupported"
                                                else seed_value
                                            ),
                                            utility_profile_id=utility_id,
                                            shock_id=stage_spec.shock_id,
                                            budget_bucket=stage_spec.budget_bucket,
                                            num_agents=stage_spec.num_agents,
                                            episode_length=stage_spec.episode_length,
                                            execution_mode=execution_mode,
                                        )
                                    )
        run_ids = [item.run_id for item in result]
        if len(run_ids) != len(set(run_ids)):
            raise PilotContractError("expanded pilot matrix contains duplicate run IDs")
        if any(selector is not None for selector in (stage, model, arm)) and not result:
            raise KeyError("pilot selector combination matches no run")
        return tuple(result)

    def validate_provenance(
        self, git_commit: str, git_tag: str
    ) -> dict[str, Any]:
        """Validate a caller-resolved annotated-tag binding for a run manifest.

        The caller must verify that ``git_tag`` is annotated and peel it to
        ``git_commit``.  This pure method verifies the frozen identity and
        returns the exact manifest fields that bind the peeled commit to the
        contract hash.
        """

        resolved = _git_commit(git_commit, "git_commit")
        actual_tag = _text(git_tag, "git_tag")
        required_tag = str(self.implementation["required_git_tag"])
        if actual_tag != required_tag:
            raise PilotContractError(
                f"pilot requires annotated tag {required_tag!r}, not {actual_tag!r}"
            )
        required_commit = self.implementation["required_git_commit"]
        if required_commit is not None and resolved != required_commit:
            raise PilotContractError("git commit does not match frozen contract")
        return {
            "git_tag": actual_tag,
            "resolved_git_commit": resolved,
            "commit_resolution": self.implementation["commit_resolution"],
            "p0_base_commit": self.implementation["p0_base_commit"],
            "contract_id": self.contract_id,
            "contract_sha256": self.canonical_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "status": self.status,
            "implementation": _thaw_json(self.implementation),
            "seeds": _thaw_json(self.seeds),
            "provider_profiles": {
                key: profile.to_dict()
                for key, profile in self.provider_profiles.items()
            },
            "arms": _thaw_json(self.arms),
            "narratives": _thaw_json(self.narratives),
            "shocks": _thaw_json(self.shocks),
            "utility": _thaw_json(self.utility),
            "budgets": _thaw_json(self.budgets),
            "stop_go": _thaw_json(self.stop_go),
            "stages": [stage.to_dict() for stage in self.stages],
            "non_claims": list(self.non_claims),
            "integrity": {
                "canonicalization": self.canonicalization,
                "declared_sha256": self.declared_sha256,
            },
        }


def load_pilot_contract(path: str | Path) -> PilotContract:
    """Load a JSON-compatible YAML pilot contract and verify its declared hash."""

    source = Path(path)

    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PilotContractError(
                    f"pilot contract contains duplicate JSON key: {key!r}"
                )
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise PilotContractError(
            f"pilot contract contains non-finite JSON number: {value}"
        )

    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise PilotContractError(
            "pilot contract must use JSON-compatible YAML"
        ) from exc
    return PilotContract.from_dict(_mapping(value, "pilot contract"))


__all__ = [
    "PILOT_CONTRACT_CANONICALIZATION",
    "PILOT_CONTRACT_SCHEMA_VERSION",
    "PilotContract",
    "PilotContractError",
    "PilotRunSpec",
    "PilotStage",
    "PilotStageCell",
    "PriceSnapshot",
    "ProviderRequestProfile",
    "ReasoningProfile",
    "canonical_contract_sha256",
    "canonical_sha256",
    "load_pilot_contract",
]
