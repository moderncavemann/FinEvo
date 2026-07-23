"""
Multi-Model LLM Provider Interface - ECCV 2026

Supports:
- OpenAI (GPT-4o, GPT-5.2)
- Google Gemini (Gemini 3 Pro)
- Local Models via OpenAI-compatible API (MLX, vLLM, LM Studio)
- Ollama
"""

import os
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from verified_memory.budget import (
    BudgetExceeded,
    CallReservation,
    RunBudget,
    UsageRecord,
)
from verified_memory.pilot_contract import (
    PilotContractError,
    ProviderRequestProfile,
)


# Cost tracking (per 1k tokens)
MODEL_COSTS = {
    # OpenAI
    # USD per 1k tokens. GPT-5.2 pricing checked against the official model
    # page on 2026-07-22; prompt totals include cached prompt tokens.
    "gpt-5.2": {
        "prompt": 0.00175,
        "cached_prompt": 0.000175,
        "completion": 0.014,
    },
    "gpt-5.2-2025-12-11": {
        "prompt": 0.00175,
        "cached_prompt": 0.000175,
        "completion": 0.014,
    },
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4.1-mini": {"prompt": 0.001, "completion": 0.002},
    # Gemini
    "gemini-3-pro-preview": {"prompt": 0.00125, "completion": 0.005},
    "gemini-2.0-flash": {"prompt": 0.0001, "completion": 0.0004},
    "google/gemini-3-flash-preview": {"prompt": 0.00125, "completion": 0.005},
    # Local models - free
    "default_local": {"prompt": 0, "completion": 0},
}


def _validated_retry_count(value: Optional[int], default: int) -> int:
    retries = default if value is None else value
    if isinstance(retries, bool) or not isinstance(retries, int):
        raise TypeError("max_retries must be an int or None")
    if retries < 1:
        raise ValueError("max_retries must be at least 1")
    return retries


def _validated_seed(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("seed must be an int or None")
    return value


def _usage_value(usage: object, *names: str) -> int:
    """Read an integer usage field from SDK objects or JSON mappings."""

    for name in names:
        if isinstance(usage, Mapping):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                continue
    return 0


def _nested_usage_value(usage: object, parent: str, *names: str) -> int:
    if isinstance(usage, Mapping):
        details = usage.get(parent)
    else:
        details = getattr(usage, parent, None)
    return _usage_value(details, *names)


def _reported_cost(usage: object) -> Optional[float]:
    """Return provider-billed cost when an API exposes it directly."""

    candidates = []
    if isinstance(usage, Mapping):
        candidates.extend([usage.get("cost"), usage.get("cost_usd")])
    else:
        candidates.extend([getattr(usage, "cost", None), getattr(usage, "cost_usd", None)])
        model_extra = getattr(usage, "model_extra", None)
        if isinstance(model_extra, Mapping):
            candidates.extend([model_extra.get("cost"), model_extra.get("cost_usd")])
    for value in candidates:
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            return parsed
    return None


def _metadata_text(value: object, *names: str) -> Optional[str]:
    """Read one non-empty routing metadata string from SDK/JSON objects."""

    sources: list[object] = [value]
    if isinstance(value, Mapping):
        for name in ("model_extra", "metadata", "routing"):
            nested = value.get(name)
            if nested is not None:
                sources.append(nested)
    else:
        for name in ("model_extra", "metadata", "routing"):
            nested = getattr(value, name, None)
            if nested is not None:
                sources.append(nested)
    for source in sources:
        for name in names:
            if isinstance(source, Mapping):
                candidate = source.get(name)
            else:
                candidate = getattr(source, name, None)
            if isinstance(candidate, Mapping):
                candidate = (
                    candidate.get("name")
                    or candidate.get("id")
                    or candidate.get("slug")
                )
            if candidate is None:
                continue
            text = str(candidate).strip()
            if text:
                return text
    return None


_METADATA_MISSING = object()


def _sdk_metadata_field(value: object, name: str) -> object:
    """Read a field from mappings, SDK attributes, or Pydantic ``model_extra``.

    The OpenAI Python SDK preserves response fields that are not in its static
    schema inside ``model_extra``. OpenRouter's opt-in metadata is such a field
    on SDK versions that predate the router-metadata response extension.
    """

    direct: object = _METADATA_MISSING
    extra: object = _METADATA_MISSING
    if isinstance(value, Mapping):
        if name in value:
            direct = value[name]
        extra = value.get("model_extra", _METADATA_MISSING)
    else:
        try:
            direct = getattr(value, name, _METADATA_MISSING)
        except Exception:
            direct = _METADATA_MISSING
        try:
            extra = getattr(value, "model_extra", _METADATA_MISSING)
        except Exception:
            extra = _METADATA_MISSING
    if direct is not _METADATA_MISSING and direct is not None:
        return direct
    if isinstance(extra, Mapping) and name in extra and extra[name] is not None:
        return extra[name]
    return _METADATA_MISSING


def _required_metadata_text(value: object, name: str) -> str:
    candidate = _sdk_metadata_field(value, name)
    if (
        candidate is _METADATA_MISSING
        or not isinstance(candidate, str)
        or not candidate
        or candidate.strip() != candidate
    ):
        raise PilotContractError(
            f"OpenRouter route attestation field {name} is missing or invalid"
        )
    return candidate


def _required_metadata_sequence(value: object, name: str) -> Sequence[object]:
    candidate = _sdk_metadata_field(value, name)
    if (
        candidate is _METADATA_MISSING
        or not isinstance(candidate, Sequence)
        or isinstance(candidate, (str, bytes, bytearray))
    ):
        raise PilotContractError(
            f"OpenRouter route attestation field {name} is missing or invalid"
        )
    return candidate


def _legacy_openrouter_response_route(
    response: object,
) -> tuple[Optional[str], Optional[str]]:
    """Preserve best-effort route extraction for non-pilot callers."""

    provider = _metadata_text(
        response,
        "provider",
        "provider_name",
        "providerName",
    )
    route = _metadata_text(
        response,
        "route",
        "route_id",
        "endpoint",
        "endpoint_id",
        "endpoint_tag",
        "provider_endpoint",
    )
    return provider, route


def _validate_openrouter_response_route(
    profile: ProviderRequestProfile,
    response: object,
) -> tuple[str, str]:
    """Return the uniquely attested upstream provider/model or fail closed."""

    metadata = _sdk_metadata_field(response, "openrouter_metadata")
    if metadata is _METADATA_MISSING:
        raise PilotContractError("OpenRouter route attestation metadata is absent")

    requested = _required_metadata_text(metadata, "requested")
    if requested != profile.requested_model:
        raise PilotContractError(
            "OpenRouter route attestation requested-model mismatch"
        )
    strategy = _required_metadata_text(metadata, "strategy")
    if strategy != "direct":
        raise PilotContractError(
            "OpenRouter route attestation did not use direct routing"
        )
    router_attempt = _sdk_metadata_field(metadata, "attempt")
    if (
        isinstance(router_attempt, bool)
        or not isinstance(router_attempt, int)
        or router_attempt != 1
    ):
        raise PilotContractError(
            "OpenRouter route attestation did not complete on router attempt 1"
        )

    endpoints = _sdk_metadata_field(metadata, "endpoints")
    if endpoints is _METADATA_MISSING:
        raise PilotContractError(
            "OpenRouter route attestation endpoints are absent"
        )
    available = _required_metadata_sequence(endpoints, "available")
    selected: list[object] = []
    for endpoint in available:
        selected_flag = _sdk_metadata_field(endpoint, "selected")
        if not isinstance(selected_flag, bool):
            raise PilotContractError(
                "OpenRouter route attestation endpoint selection is invalid"
            )
        if selected_flag:
            selected.append(endpoint)
    if len(selected) != 1:
        raise PilotContractError(
            "OpenRouter route attestation must identify exactly one selected endpoint"
        )

    selected_provider = _required_metadata_text(selected[0], "provider")
    selected_model = _required_metadata_text(selected[0], "model")
    if selected_provider not in profile.provider_pin:
        raise PilotContractError(
            "OpenRouter route attestation selected-provider mismatch"
        )
    expected_model = dict(profile.artifact_identity).get("served_snapshot")
    if expected_model is None or selected_model != expected_model:
        raise PilotContractError(
            "OpenRouter route attestation selected-model mismatch"
        )

    attempts = _required_metadata_sequence(metadata, "attempts")
    if len(attempts) != 1:
        raise PilotContractError(
            "OpenRouter route attestation must contain exactly one upstream attempt"
        )
    attempt_provider = _required_metadata_text(attempts[0], "provider")
    attempt_model = _required_metadata_text(attempts[0], "model")
    attempt_status = _sdk_metadata_field(attempts[0], "status")
    if (
        attempt_provider != selected_provider
        or attempt_model != selected_model
        or isinstance(attempt_status, bool)
        or not isinstance(attempt_status, int)
        or attempt_status != 200
    ):
        raise PilotContractError(
            "OpenRouter route attestation upstream-attempt mismatch"
        )
    return selected_provider, selected_model


def _profile_completion_metadata(
    profile: Optional[ProviderRequestProfile],
) -> dict[str, object]:
    if profile is None:
        return {
            "request_profile_id": None,
            "request_provider_pin": (),
            "request_artifact_identity": (),
            "request_price_snapshot_source": None,
            "request_price_snapshot_captured_at": None,
        }
    return {
        "request_profile_id": profile.profile_id,
        "request_provider_pin": tuple(profile.provider_pin),
        "request_artifact_identity": tuple(profile.artifact_identity),
        "request_price_snapshot_source": profile.price_snapshot.source,
        "request_price_snapshot_captured_at": profile.price_snapshot.captured_at,
    }


def _usage_record(
    prompt_tokens: int,
    completion_tokens: int,
    costs: Mapping[str, float],
    *,
    reported_cost: Optional[float] = None,
    cached_prompt_tokens: int = 0,
) -> UsageRecord:
    cached_prompt_tokens = max(0, min(int(cached_prompt_tokens), prompt_tokens))
    cost = reported_cost
    if cost is None:
        uncached_prompt_tokens = prompt_tokens - cached_prompt_tokens
        cost = (
            uncached_prompt_tokens / 1000 * costs["prompt"]
            + cached_prompt_tokens
            / 1000
            * costs.get("cached_prompt", costs["prompt"])
            + completion_tokens / 1000 * costs["completion"]
        )
    return UsageRecord(prompt_tokens, completion_tokens, float(cost))


@dataclass(frozen=True)
class StructuredCompletion:
    """Immutable provider response with auditable usage and failure metadata."""

    text: str
    usage: UsageRecord
    model: str
    provider: str
    attempts: int
    latency_seconds: float
    error_type: Optional[str] = None
    request_seed: Optional[int] = None
    system_fingerprint: Optional[str] = None
    response_model: Optional[str] = None
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0
    request_id: Optional[str] = None
    response_provider: Optional[str] = None
    response_route: Optional[str] = None
    request_profile_id: Optional[str] = None
    request_provider_pin: tuple[str, ...] = ()
    request_artifact_identity: tuple[tuple[str, str], ...] = ()
    request_price_snapshot_source: Optional[str] = None
    request_price_snapshot_captured_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not isinstance(self.usage, UsageRecord):
            raise TypeError("usage must be a UsageRecord")
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int):
            raise TypeError("attempts must be an int")
        if self.attempts < 1:
            raise ValueError("attempts must be at least 1")
        if self.latency_seconds < 0:
            raise ValueError("latency_seconds must be non-negative")
        _validated_seed(self.request_seed)
        if self.system_fingerprint is not None and not isinstance(
            self.system_fingerprint, str
        ):
            raise TypeError("system_fingerprint must be a string or None")
        if self.response_model is not None and not isinstance(self.response_model, str):
            raise TypeError("response_model must be a string or None")
        for name in ("cached_prompt_tokens", "reasoning_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.cached_prompt_tokens > self.usage.prompt_tokens:
            raise ValueError("cached_prompt_tokens cannot exceed prompt_tokens")
        if self.request_id is not None and not isinstance(self.request_id, str):
            raise TypeError("request_id must be a string or None")
        for name in (
            "response_provider",
            "response_route",
            "request_profile_id",
            "request_price_snapshot_source",
            "request_price_snapshot_captured_at",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")
        if not isinstance(self.request_provider_pin, tuple) or any(
            not isinstance(item, str) or not item
            for item in self.request_provider_pin
        ):
            raise TypeError("request_provider_pin must be a tuple of non-empty strings")
        if not isinstance(self.request_artifact_identity, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(value, str) and value for value in item)
            for item in self.request_artifact_identity
        ):
            raise TypeError(
                "request_artifact_identity must be a tuple of string pairs"
            )

    @property
    def cost(self) -> float:
        return float(self.usage.cost_usd)

    @property
    def ok(self) -> bool:
        return self.error_type is None

    def to_dict(self) -> Dict[str, object]:
        return {
            "text": self.text,
            "usage": self.usage.to_dict(),
            "model": self.model,
            "provider": self.provider,
            "attempts": self.attempts,
            "latency_seconds": self.latency_seconds,
            "error_type": self.error_type,
            "request_seed": self.request_seed,
            "system_fingerprint": self.system_fingerprint,
            "response_model": self.response_model,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "request_id": self.request_id,
            "response_provider": self.response_provider,
            "response_route": self.response_route,
            "request_profile_id": self.request_profile_id,
            "request_provider_pin": list(self.request_provider_pin),
            "request_artifact_identity": dict(self.request_artifact_identity),
            "request_price_snapshot_source": self.request_price_snapshot_source,
            "request_price_snapshot_captured_at": (
                self.request_price_snapshot_captured_at
            ),
        }


# Alternative descriptive name for downstream callers.
CompletionResult = StructuredCompletion


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def get_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
    ) -> Tuple[str, float]:
        """
        Get completion from LLM

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response_text, cost)
        """
        pass

    def get_structured_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
        max_retries: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> StructuredCompletion:
        """Compatibility adapter for providers implementing only the tuple API."""

        _validated_retry_count(max_retries, 1)
        if _validated_seed(seed) is not None:
            raise NotImplementedError(
                "legacy tuple providers cannot attest that a decoding seed was applied"
            )
        started = time.monotonic()
        full_name = self.get_model_name()
        provider, _, model = full_name.partition("/")
        try:
            text, cost = self.get_completion(messages, temperature, max_tokens, top_p)
            error_type = "LegacyProviderError" if text == "Error" else None
            return StructuredCompletion(
                text=str(text),
                usage=UsageRecord(cost_usd=float(cost)),
                model=model or full_name,
                provider=provider or "unknown",
                attempts=1,
                latency_seconds=time.monotonic() - started,
                error_type=error_type,
                response_model=model or full_name,
            )
        except Exception as exc:
            return StructuredCompletion(
                text="Error",
                usage=UsageRecord(),
                model=model or full_name,
                provider=provider or "unknown",
                attempts=1,
                latency_seconds=time.monotonic() - started,
                error_type=type(exc).__name__,
            )

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model identifier string"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_retries: Optional[int] = None,
        request_profile: Optional[ProviderRequestProfile] = None,
    ):
        self.api_key = api_key
        self.model = model
        if request_profile is not None and not isinstance(
            request_profile, ProviderRequestProfile
        ):
            raise TypeError(
                "request_profile must be a ProviderRequestProfile or None"
            )
        self.request_profile = request_profile
        default_retries = (
            request_profile.max_attempts if request_profile is not None else 20
        )
        self.max_retries = _validated_retry_count(max_retries, default_retries)
        if request_profile is None:
            self.costs = MODEL_COSTS.get(
                model, {"prompt": 0.003, "completion": 0.012}
            )
        else:
            request_profile.validate_provider_configuration(
                transport="openai",
                model=model,
                max_attempts=self.max_retries,
            )
            self.costs = request_profile.price_snapshot.costs_per_1k()
        from openai import OpenAI
        client_options: Dict[str, object] = {"api_key": api_key}
        if request_profile is not None:
            # The SDK otherwise retries selected failures internally, outside
            # this provider's auditable attempt counter.
            client_options["max_retries"] = 0
        self.client = OpenAI(**client_options)

    def get_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
    ) -> Tuple[str, float]:
        result = self.get_structured_completion(messages, temperature, max_tokens, top_p)
        legacy_cost = (
            result.usage.prompt_tokens / 1000 * self.costs["prompt"]
            + result.usage.completion_tokens / 1000 * self.costs["completion"]
        )
        return result.text, legacy_cost

    def get_structured_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
        max_retries: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> StructuredCompletion:
        retry_count = _validated_retry_count(max_retries, self.max_retries)
        seed = _validated_seed(seed)
        request_profile = getattr(self, "request_profile", None)
        if request_profile is not None:
            request_profile.validate_dispatch(
                transport="openai",
                model=self.model,
                seed=seed,
                max_attempts=retry_count,
            )
        profile_metadata = _profile_completion_metadata(request_profile)
        started = time.monotonic()
        for i in range(retry_count):
            try:
                request: Dict[str, object] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                }
                if seed is not None:
                    request["seed"] = seed
                if request_profile is not None:
                    request.update(request_profile.openai_request_options())
                # GPT-5.x and newer models use max_completion_tokens
                if self.model.startswith("gpt-5") or self.model.startswith("o1") or self.model.startswith("o3"):
                    request["max_completion_tokens"] = max_tokens
                else:
                    request["max_tokens"] = max_tokens
                response = self.client.chat.completions.create(**request)
                prompt_tokens = _usage_value(response.usage, "prompt_tokens")
                completion_tokens = _usage_value(response.usage, "completion_tokens")
                cached_prompt_tokens = _nested_usage_value(
                    response.usage, "prompt_tokens_details", "cached_tokens"
                )
                reasoning_tokens = _nested_usage_value(
                    response.usage, "completion_tokens_details", "reasoning_tokens"
                )
                usage = _usage_record(
                    prompt_tokens,
                    completion_tokens,
                    self.costs,
                    reported_cost=_reported_cost(response.usage),
                    cached_prompt_tokens=cached_prompt_tokens,
                )
                response_model = (
                    str(response.model)
                    if getattr(response, "model", None) is not None
                    else None
                )
                if request_profile is not None:
                    try:
                        request_profile.validate_served_model(response_model)
                    except PilotContractError:
                        return StructuredCompletion(
                            text="Error",
                            usage=usage,
                            model=self.model,
                            provider="openai",
                            attempts=i + 1,
                            latency_seconds=time.monotonic() - started,
                            error_type="PilotContractError",
                            request_seed=seed,
                            response_model=response_model,
                            cached_prompt_tokens=cached_prompt_tokens,
                            reasoning_tokens=reasoning_tokens,
                            request_id=(
                                str(response.id)
                                if getattr(response, "id", None) is not None
                                else None
                            ),
                            response_provider="OpenAI-direct",
                            response_route="direct",
                            **profile_metadata,
                        )
                return StructuredCompletion(
                    text=response.choices[0].message.content or "",
                    usage=usage,
                    model=self.model,
                    provider="openai",
                    attempts=i + 1,
                    latency_seconds=time.monotonic() - started,
                    request_seed=seed,
                    system_fingerprint=(
                        str(response.system_fingerprint)
                        if getattr(response, "system_fingerprint", None) is not None
                        else None
                    ),
                    response_model=response_model,
                    cached_prompt_tokens=cached_prompt_tokens,
                    reasoning_tokens=reasoning_tokens,
                    request_id=(
                        str(response.id)
                        if getattr(response, "id", None) is not None
                        else None
                    ),
                    response_provider="OpenAI-direct",
                    response_route="direct",
                    **profile_metadata,
                )
            except Exception as e:
                if i < retry_count - 1:
                    time.sleep(2)
                else:
                    return StructuredCompletion(
                        text="Error",
                        usage=UsageRecord(),
                        model=self.model,
                        provider="openai",
                        attempts=i + 1,
                        latency_seconds=time.monotonic() - started,
                        error_type=type(e).__name__,
                        response_provider="OpenAI-direct",
                        response_route="direct",
                        **profile_metadata,
                    )

    def get_model_name(self) -> str:
        return f"openai/{self.model}"


class GeminiProvider(LLMProvider):
    """Google Gemini API provider with rate limiting"""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3-pro-preview",
        max_retries: int = 20,
    ):
        self.api_key = api_key
        self.model = model
        self.costs = MODEL_COSTS.get(model, {"prompt": 0.002, "completion": 0.008})
        self.max_retries = _validated_retry_count(max_retries, 20)

        # Rate limiting: 25 RPM = 2.4s per request, use 3s for safety margin
        self.min_request_interval = 3.0
        self.last_request_time = 0

        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

    def get_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
    ) -> Tuple[str, float]:
        result = self.get_structured_completion(messages, temperature, max_tokens, top_p)
        # Preserve the historical tuple API's character-based estimate even
        # when the structured path has exact SDK token metadata.
        if result.error_type is not None:
            return result.text, 0
        prompt_tokens = sum(
            len(str(message.get("content", ""))) // 4
            for message in messages
            if message.get("role") in {"user", "assistant"}
        )
        completion_tokens = len(result.text) // 4
        legacy_cost = (
            prompt_tokens / 1000 * self.costs["prompt"]
            + completion_tokens / 1000 * self.costs["completion"]
        )
        return result.text, legacy_cost

    def get_structured_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
        max_retries: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> StructuredCompletion:
        import google.generativeai as genai

        retry_count = _validated_retry_count(max_retries, self.max_retries)
        if _validated_seed(seed) is not None:
            raise NotImplementedError(
                "the installed Gemini SDK does not expose a decoding seed"
            )
        started = time.monotonic()

        # Rate limiting to avoid hitting RPM limits
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

        for i in range(retry_count):
            try:
                gemini_messages = []
                system_prompt = ""

                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]

                    if role == "system":
                        system_prompt = content
                    elif role == "user":
                        gemini_messages.append({"role": "user", "parts": [content]})
                    elif role == "assistant":
                        gemini_messages.append({"role": "model", "parts": [content]})

                chat = self.client.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])

                last_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""
                if system_prompt:
                    last_message = f"{system_prompt}\n\n{last_message}"

                response = chat.send_message(
                    last_message,
                    generation_config=genai.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        top_p=top_p,
                    )
                )

                usage_metadata = getattr(response, "usage_metadata", None)
                prompt_tokens = _usage_value(
                    usage_metadata,
                    "prompt_token_count",
                    "prompt_tokens",
                )
                completion_tokens = _usage_value(
                    usage_metadata,
                    "candidates_token_count",
                    "completion_tokens",
                )
                # Older Gemini SDK responses do not expose usage metadata.
                if prompt_tokens == 0:
                    prompt_tokens = sum(len(m["parts"][0]) // 4 for m in gemini_messages)
                if completion_tokens == 0:
                    completion_tokens = len(response.text) // 4
                usage = _usage_record(
                    prompt_tokens,
                    completion_tokens,
                    self.costs,
                    reported_cost=_reported_cost(usage_metadata),
                )

                return StructuredCompletion(
                    text=response.text,
                    usage=usage,
                    model=self.model,
                    provider="gemini",
                    attempts=i + 1,
                    latency_seconds=time.monotonic() - started,
                    response_model=self.model,
                )

            except Exception as e:
                if i < retry_count - 1:
                    time.sleep(2)
                else:
                    return StructuredCompletion(
                        text="Error",
                        usage=UsageRecord(),
                        model=self.model,
                        provider="gemini",
                        attempts=i + 1,
                        latency_seconds=time.monotonic() - started,
                        error_type=type(e).__name__,
                    )

    def get_model_name(self) -> str:
        return f"gemini/{self.model}"


class LocalAPIProvider(LLMProvider):
    """Local model provider via OpenAI-compatible API (MLX, vLLM, LM Studio)"""

    def __init__(
        self,
        model: str = "mlx-community/Llama-3.3-70B-Instruct-4bit",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        max_retries: int = 10,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.costs = {"prompt": 0, "completion": 0}
        self.max_retries = _validated_retry_count(max_retries, 10)

    def get_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
    ) -> Tuple[str, float]:
        result = self.get_structured_completion(messages, temperature, max_tokens, top_p)
        return result.text, 0

    def get_structured_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
        max_retries: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> StructuredCompletion:
        import requests

        retry_count = _validated_retry_count(max_retries, self.max_retries)
        seed = _validated_seed(seed)
        started = time.monotonic()
        for i in range(retry_count):
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                }
                if seed is not None:
                    payload["seed"] = seed
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=300
                )
                response.raise_for_status()
                result = response.json()
                raw_usage = result.get("usage") or {}
                prompt_tokens = _usage_value(raw_usage, "prompt_tokens")
                completion_tokens = _usage_value(raw_usage, "completion_tokens")
                cached_prompt_tokens = _nested_usage_value(
                    raw_usage, "prompt_tokens_details", "cached_tokens"
                )
                reasoning_tokens = _nested_usage_value(
                    raw_usage, "completion_tokens_details", "reasoning_tokens"
                )
                usage = _usage_record(
                    prompt_tokens,
                    completion_tokens,
                    self.costs,
                    reported_cost=_reported_cost(raw_usage),
                    cached_prompt_tokens=cached_prompt_tokens,
                )
                return StructuredCompletion(
                    text=result["choices"][0]["message"]["content"] or "",
                    usage=usage,
                    model=self.model,
                    provider="local",
                    attempts=i + 1,
                    latency_seconds=time.monotonic() - started,
                    request_seed=seed,
                    system_fingerprint=(
                        str(result["system_fingerprint"])
                        if result.get("system_fingerprint") is not None
                        else None
                    ),
                    response_model=str(result.get("model") or self.model),
                    cached_prompt_tokens=cached_prompt_tokens,
                    reasoning_tokens=reasoning_tokens,
                    request_id=(
                        str(result["id"]) if result.get("id") is not None else None
                    ),
                )

            except Exception as e:
                if i < retry_count - 1:
                    time.sleep(2)
                else:
                    return StructuredCompletion(
                        text="Error",
                        usage=UsageRecord(),
                        model=self.model,
                        provider="local",
                        attempts=i + 1,
                        latency_seconds=time.monotonic() - started,
                        error_type=type(e).__name__,
                    )

    def get_model_name(self) -> str:
        return f"local/{self.model}"


class ThirdPartyProvider(LLMProvider):
    """OpenAI-compatible third-party provider, e.g. OpenRouter."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        app_name: str = "FinEvo",
        max_retries: Optional[int] = None,
        request_profile: Optional[ProviderRequestProfile] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        if request_profile is not None and not isinstance(
            request_profile, ProviderRequestProfile
        ):
            raise TypeError(
                "request_profile must be a ProviderRequestProfile or None"
            )
        self.request_profile = request_profile
        default_retries = (
            request_profile.max_attempts if request_profile is not None else 20
        )
        self.max_retries = _validated_retry_count(max_retries, default_retries)
        if request_profile is None:
            self.costs = MODEL_COSTS.get(
                model, {"prompt": 0, "completion": 0}
            )
        else:
            if base_url.rstrip("/") != "https://openrouter.ai/api/v1":
                raise PilotContractError(
                    "OpenRouter pilot profiles require the frozen OpenRouter API URL"
                )
            request_profile.validate_provider_configuration(
                transport="openrouter",
                model=model,
                max_attempts=self.max_retries,
            )
            self.costs = request_profile.price_snapshot.costs_per_1k()

        from openai import OpenAI
        default_headers = {
            "HTTP-Referer": "https://github.com/moderncavemann/FinEvo",
            "X-Title": app_name,
        }
        if request_profile is not None:
            default_headers["X-OpenRouter-Metadata"] = "enabled"
        client_options: Dict[str, object] = {
            "api_key": api_key,
            "base_url": base_url,
            "default_headers": default_headers,
        }
        if request_profile is not None:
            # One pilot attempt means one wire attempt; disable SDK retries.
            client_options["max_retries"] = 0
        self.client = OpenAI(**client_options)

    def get_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
    ) -> Tuple[str, float]:
        result = self.get_structured_completion(messages, temperature, max_tokens, top_p)
        legacy_cost = (
            result.usage.prompt_tokens / 1000 * self.costs["prompt"]
            + result.usage.completion_tokens / 1000 * self.costs["completion"]
        )
        return result.text, legacy_cost

    def get_structured_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
        max_retries: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> StructuredCompletion:
        retry_count = _validated_retry_count(max_retries, self.max_retries)
        seed = _validated_seed(seed)
        request_profile = getattr(self, "request_profile", None)
        if request_profile is not None:
            request_profile.validate_dispatch(
                transport="openrouter",
                model=self.model,
                seed=seed,
                max_attempts=retry_count,
            )
        profile_metadata = _profile_completion_metadata(request_profile)
        started = time.monotonic()
        for i in range(retry_count):
            try:
                request: Dict[str, object] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                }
                if seed is not None:
                    request["seed"] = seed
                if request_profile is not None:
                    request.update(request_profile.openrouter_request_options())
                response = self.client.chat.completions.create(**request)
                prompt_tokens = _usage_value(response.usage, "prompt_tokens")
                completion_tokens = _usage_value(response.usage, "completion_tokens")
                cached_prompt_tokens = _nested_usage_value(
                    response.usage, "prompt_tokens_details", "cached_tokens"
                )
                reasoning_tokens = _nested_usage_value(
                    response.usage, "completion_tokens_details", "reasoning_tokens"
                )
                usage = _usage_record(
                    prompt_tokens,
                    completion_tokens,
                    self.costs,
                    reported_cost=_reported_cost(response.usage),
                    cached_prompt_tokens=cached_prompt_tokens,
                )
                response_model = (
                    str(response.model)
                    if getattr(response, "model", None) is not None
                    else None
                )
                if request_profile is not None:
                    response_provider: Optional[str] = None
                    response_route: Optional[str] = None
                    try:
                        request_profile.validate_served_model(response_model)
                        response_provider, response_route = (
                            _validate_openrouter_response_route(
                                request_profile,
                                response,
                            )
                        )
                    except PilotContractError:
                        return StructuredCompletion(
                            text="Error",
                            usage=usage,
                            model=self.model,
                            provider="thirdparty",
                            attempts=i + 1,
                            latency_seconds=time.monotonic() - started,
                            error_type="PilotContractError",
                            request_seed=seed,
                            response_model=response_model,
                            cached_prompt_tokens=cached_prompt_tokens,
                            reasoning_tokens=reasoning_tokens,
                            request_id=(
                                str(response.id)
                                if getattr(response, "id", None) is not None
                                else None
                            ),
                            response_provider=response_provider,
                            response_route=response_route,
                            **profile_metadata,
                        )
                else:
                    response_provider, response_route = (
                        _legacy_openrouter_response_route(response)
                    )
                return StructuredCompletion(
                    text=response.choices[0].message.content or "",
                    usage=usage,
                    model=self.model,
                    provider="thirdparty",
                    attempts=i + 1,
                    latency_seconds=time.monotonic() - started,
                    request_seed=seed,
                    system_fingerprint=(
                        str(response.system_fingerprint)
                        if getattr(response, "system_fingerprint", None) is not None
                        else None
                    ),
                    response_model=response_model,
                    cached_prompt_tokens=cached_prompt_tokens,
                    reasoning_tokens=reasoning_tokens,
                    request_id=(
                        str(response.id)
                        if getattr(response, "id", None) is not None
                        else None
                    ),
                    response_provider=response_provider,
                    response_route=response_route,
                    **profile_metadata,
                )
            except Exception as e:
                if i < retry_count - 1:
                    time.sleep(2)
                else:
                    return StructuredCompletion(
                        text="Error",
                        usage=UsageRecord(),
                        model=self.model,
                        provider="thirdparty",
                        attempts=i + 1,
                        latency_seconds=time.monotonic() - started,
                        error_type=type(e).__name__,
                        **profile_metadata,
                    )

    def get_model_name(self) -> str:
        return f"thirdparty/{self.model}"


class OllamaProvider(LLMProvider):
    """Ollama local model provider"""

    def __init__(
        self,
        model: str = "llama3:8b",
        host: str = "http://localhost:11434",
        max_retries: Optional[int] = None,
        request_profile: Optional[ProviderRequestProfile] = None,
    ):
        self.model = model
        self.host = host
        if request_profile is not None and not isinstance(
            request_profile, ProviderRequestProfile
        ):
            raise TypeError(
                "request_profile must be a ProviderRequestProfile or None"
            )
        self.request_profile = request_profile
        default_retries = (
            request_profile.max_attempts if request_profile is not None else 10
        )
        self.max_retries = _validated_retry_count(max_retries, default_retries)
        if request_profile is None:
            self.costs = {"prompt": 0, "completion": 0}
        else:
            request_profile.validate_provider_configuration(
                transport="ollama",
                model=model,
                max_attempts=self.max_retries,
            )
            self.costs = request_profile.price_snapshot.costs_per_1k()

    def get_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
    ) -> Tuple[str, float]:
        result = self.get_structured_completion(messages, temperature, max_tokens, top_p)
        return result.text, 0

    def get_structured_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
        max_retries: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> StructuredCompletion:
        import requests

        retry_count = _validated_retry_count(max_retries, self.max_retries)
        seed = _validated_seed(seed)
        request_profile = getattr(self, "request_profile", None)
        if request_profile is not None:
            request_profile.validate_dispatch(
                transport="ollama",
                model=self.model,
                seed=seed,
                max_attempts=retry_count,
            )
        profile_metadata = _profile_completion_metadata(request_profile)
        started = time.monotonic()
        for i in range(retry_count):
            try:
                options = {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": top_p,
                }
                if seed is not None:
                    options["seed"] = seed
                request_body: Dict[str, object] = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": options,
                }
                if (
                    request_profile is not None
                    and request_profile.json_mode == "json_object"
                ):
                    request_body["format"] = "json"
                response = requests.post(
                    f"{self.host}/api/chat",
                    json=request_body,
                    timeout=300
                )
                response.raise_for_status()
                result = response.json()
                prompt_tokens = _usage_value(result, "prompt_eval_count", "prompt_tokens")
                completion_tokens = _usage_value(result, "eval_count", "completion_tokens")
                usage = _usage_record(prompt_tokens, completion_tokens, self.costs)
                response_model = str(result.get("model") or self.model)
                if request_profile is not None:
                    try:
                        request_profile.validate_served_model(response_model)
                    except PilotContractError:
                        return StructuredCompletion(
                            text="Error",
                            usage=usage,
                            model=self.model,
                            provider="ollama",
                            attempts=i + 1,
                            latency_seconds=time.monotonic() - started,
                            error_type="PilotContractError",
                            request_seed=seed,
                            response_model=response_model,
                            response_provider="local-ollama",
                            response_route="local",
                            **profile_metadata,
                        )
                return StructuredCompletion(
                    text=result["message"]["content"] or "",
                    usage=usage,
                    model=self.model,
                    provider="ollama",
                    attempts=i + 1,
                    latency_seconds=time.monotonic() - started,
                    request_seed=seed,
                    response_model=response_model,
                    response_provider="local-ollama",
                    response_route="local",
                    **profile_metadata,
                )

            except Exception as e:
                if i < retry_count - 1:
                    time.sleep(2)
                else:
                    return StructuredCompletion(
                        text="Error",
                        usage=UsageRecord(),
                        model=self.model,
                        provider="ollama",
                        attempts=i + 1,
                        latency_seconds=time.monotonic() - started,
                        error_type=type(e).__name__,
                        response_provider="local-ollama",
                        response_route="local",
                        **profile_metadata,
                    )

    def get_model_name(self) -> str:
        return f"ollama/{self.model}"


class MultiModelLLM:
    """Multi-model LLM manager with parallel execution"""

    def __init__(self, provider: LLMProvider, num_workers: int = 10):
        self.provider = provider
        self.num_workers = num_workers

    def get_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
    ) -> Tuple[str, float]:
        return self.provider.get_completion(messages, temperature, max_tokens, top_p)

    def _provider_identity(self) -> Tuple[str, str]:
        full_name = self.provider.get_model_name()
        provider, _, model = full_name.partition("/")
        return provider or "unknown", model or full_name

    def _invoke_structured(
        self,
        messages: List[Dict],
        *,
        temperature: float,
        max_tokens: int,
        top_p: float,
        seed: Optional[int],
        max_retries: Optional[int],
        budget: Optional[RunBudget],
        reservation: Optional[CallReservation],
    ) -> StructuredCompletion:
        """Invoke a provider after reservation; never roll back dispatched work."""

        started = time.monotonic()
        provider_name, model_name = self._provider_identity()
        try:
            result = self.provider.get_structured_completion(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                max_retries=max_retries,
                seed=seed,
            )
            if not isinstance(result, StructuredCompletion):
                raise TypeError("provider structured API must return StructuredCompletion")
        except Exception as exc:
            # Invocation has begun, so the call may have reached the provider.  It
            # must consume its reserved call slot even when exact usage is unknown.
            result = StructuredCompletion(
                text="Error",
                usage=UsageRecord(),
                model=model_name,
                provider=provider_name,
                attempts=1,
                latency_seconds=time.monotonic() - started,
                error_type=type(exc).__name__,
            )

        if budget is not None:
            if reservation is None:
                raise RuntimeError("budgeted dispatch is missing its reservation")
            if result.error_type is not None:
                # Invocation already began, so exact provider billing may be
                # unavailable even when the wrapper returned zero/partial usage.
                # Never release the conservative reservation on this path.
                estimate = reservation.estimated_usage
                conservative_usage = UsageRecord(
                    prompt_tokens=max(
                        result.usage.prompt_tokens, estimate.prompt_tokens
                    ),
                    completion_tokens=max(
                        result.usage.completion_tokens,
                        estimate.completion_tokens,
                    ),
                    cost_usd=max(
                        float(result.usage.cost_usd),
                        float(estimate.cost_usd),
                    ),
                )
                if conservative_usage != result.usage:
                    result = replace(result, usage=conservative_usage)
            budget.complete_call(reservation, result.usage)
        return result

    def get_structured_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
        *,
        budget: Optional[RunBudget] = None,
        estimated_usage: Optional[UsageRecord] = None,
        label: str = "completion",
        tags: Optional[Mapping[str, object]] = None,
        max_retries: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> StructuredCompletion:
        """Return one structured response, optionally under a hard run budget."""

        if max_retries is not None:
            _validated_retry_count(max_retries, 1)
        if budget is not None and not isinstance(budget, RunBudget):
            raise TypeError("budget must be a RunBudget or None")
        if estimated_usage is not None and not isinstance(estimated_usage, UsageRecord):
            raise TypeError("estimated_usage must be a UsageRecord or None")
        seed = _validated_seed(seed)
        if budget is not None:
            if max_retries not in (None, 1):
                raise ValueError(
                    "budgeted calls require max_retries=1; provider-internal retries "
                    "cannot share one hard-budget reservation"
                )
            max_retries = 1

        reservation = None
        if budget is not None:
            reservation = budget.reserve_call(
                estimated_usage=estimated_usage,
                label=label,
                model=self.provider.get_model_name(),
                tags=tags,
            )

        # All validation and reservation work is complete.  From this point on,
        # provider invocation counts as dispatch and must be accounted, not rolled back.
        return self._invoke_structured(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            seed=seed,
            max_retries=max_retries,
            budget=budget,
            reservation=reservation,
        )

    def get_multiple_completions(
        self,
        dialogs: List[List[Dict]],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
    ) -> Tuple[List[str], float]:
        """
        Get completions for multiple dialogs in parallel

        Args:
            dialogs: List of dialog message lists
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (list of responses, total cost)
        """
        get_completion_partial = partial(
            self.provider.get_completion,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        results = [None] * len(dialogs)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_idx = {
                executor.submit(get_completion_partial, d): i
                for i, d in enumerate(dialogs)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        total_cost = sum(cost for _, cost in results)
        responses = [response for response, _ in results]

        return responses, total_cost

    @staticmethod
    def _normalize_labels(
        labels: Optional[Union[str, Sequence[str]]],
        count: int,
    ) -> List[str]:
        if labels is None:
            return [f"completion:{index}" for index in range(count)]
        if isinstance(labels, str):
            return [labels for _ in range(count)]
        normalized = [str(value) for value in labels]
        if len(normalized) != count:
            raise ValueError("labels length must match dialogs length")
        return normalized

    @staticmethod
    def _normalize_tags(
        tags: Optional[Union[Mapping[str, object], Sequence[Mapping[str, object]]]],
        count: int,
    ) -> List[Mapping[str, object]]:
        if tags is None:
            return [{"batch_index": index} for index in range(count)]
        if isinstance(tags, Mapping):
            return [dict(tags, batch_index=index) for index in range(count)]
        normalized = list(tags)
        if len(normalized) != count:
            raise ValueError("tags length must match dialogs length")
        if any(not isinstance(value, Mapping) for value in normalized):
            raise TypeError("each tags item must be a mapping")
        return [dict(value, batch_index=index) for index, value in enumerate(normalized)]

    @staticmethod
    def _normalize_estimates(
        estimates: Optional[Union[UsageRecord, Sequence[UsageRecord]]],
        count: int,
    ) -> List[UsageRecord]:
        if estimates is None:
            return [UsageRecord() for _ in range(count)]
        if isinstance(estimates, UsageRecord):
            return [estimates for _ in range(count)]
        normalized = list(estimates)
        if len(normalized) != count:
            raise ValueError("estimated_usages length must match dialogs length")
        if any(not isinstance(value, UsageRecord) for value in normalized):
            raise TypeError("each estimated usage must be a UsageRecord")
        return normalized

    def get_multiple_structured_completions(
        self,
        dialogs: List[List[Dict]],
        temperature: float = 0,
        max_tokens: int = 800,
        top_p: float = 1.0,
        *,
        budget: Optional[RunBudget] = None,
        labels: Optional[Union[str, Sequence[str]]] = None,
        tags: Optional[Union[Mapping[str, object], Sequence[Mapping[str, object]]]] = None,
        estimated_usages: Optional[Union[UsageRecord, Sequence[UsageRecord]]] = None,
        max_retries: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[StructuredCompletion]:
        """Return ordered structured responses for a bounded parallel batch.

        Every reservation is acquired before the first executor submission.  If
        the complete batch cannot fit, prior reservations are rolled back and
        no provider method is called.
        """

        if max_retries is not None:
            _validated_retry_count(max_retries, 1)
        if budget is not None and not isinstance(budget, RunBudget):
            raise TypeError("budget must be a RunBudget or None")
        seed = _validated_seed(seed)
        if budget is not None:
            if max_retries not in (None, 1):
                raise ValueError(
                    "budgeted calls require max_retries=1; provider-internal retries "
                    "cannot share one hard-budget reservation"
                )
            max_retries = 1

        count = len(dialogs)
        if count == 0:
            return []
        normalized_labels = self._normalize_labels(labels, count)
        normalized_tags = self._normalize_tags(tags, count)
        estimates = self._normalize_estimates(estimated_usages, count)

        reservations: List[Optional[CallReservation]] = [None] * count
        if budget is not None:
            acquired: List[CallReservation] = []
            try:
                for index in range(count):
                    reservation = budget.reserve_call(
                        estimated_usage=estimates[index],
                        label=normalized_labels[index],
                        model=self.provider.get_model_name(),
                        tags=normalized_tags[index],
                    )
                    reservations[index] = reservation
                    acquired.append(reservation)
            except Exception:
                # No task has been submitted yet, so these reservations are safe
                # to refund even when the triggering error is BudgetExceeded.
                for reservation in reversed(acquired):
                    budget.rollback_call(reservation)
                raise

        results: List[Optional[StructuredCompletion]] = [None] * count
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        future_to_idx = {}
        submitted_indexes = set()
        try:
            for index, dialog in enumerate(dialogs):
                future = executor.submit(
                    self._invoke_structured,
                    dialog,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    seed=seed,
                    max_retries=max_retries,
                    budget=budget,
                    reservation=reservations[index],
                )
                future_to_idx[future] = index
                submitted_indexes.add(index)
        except Exception:
            # Only reservations whose executor submission never happened are
            # refundable. Submitted workers own and settle all other slots.
            if budget is not None:
                for index, reservation in enumerate(reservations):
                    if index not in submitted_indexes and reservation is not None:
                        budget.rollback_call(reservation)
            executor.shutdown(wait=True)
            raise

        first_error: Optional[Exception] = None
        try:
            for future in as_completed(future_to_idx):
                index = future_to_idx[future]
                try:
                    results[index] = future.result()
                except Exception as exc:  # account all in-flight work before raising
                    if first_error is None:
                        first_error = exc
        finally:
            executor.shutdown(wait=True)

        if first_error is not None:
            raise first_error
        if any(result is None for result in results):
            raise RuntimeError("structured batch completed without a result for every dialog")
        return [result for result in results if result is not None]

    def get_model_name(self) -> str:
        return self.provider.get_model_name()


def create_llm_provider(
    provider_type: str,
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    max_retries: Optional[int] = None,
    request_profile: Optional[ProviderRequestProfile] = None,
) -> LLMProvider:
    """
    Factory function to create LLM provider instance

    Args:
        provider_type: "openai", "gemini", "ollama", "local", or "thirdparty"
        model: Model name/identifier
        api_key: API key (required for openai and gemini)
        base_url: Base URL for local API server
        request_profile: Optional frozen pilot request identity. Strict pilot
            profiles are supported for direct OpenAI, OpenRouter, and Ollama.

    Returns:
        LLMProvider instance
    """
    if request_profile is not None:
        if not isinstance(request_profile, ProviderRequestProfile):
            raise TypeError(
                "request_profile must be a ProviderRequestProfile or None"
            )
        if model is None:
            model = request_profile.requested_model
        elif model != request_profile.requested_model:
            raise PilotContractError(
                f"profile {request_profile.profile_id} requested-model mismatch"
            )

    if provider_type == "openai":
        if api_key is None:
            api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required")
        model = model or "gpt-4o"
        return OpenAIProvider(
            api_key=api_key,
            model=model,
            max_retries=max_retries,
            request_profile=request_profile,
        )

    elif provider_type == "gemini":
        if request_profile is not None:
            raise PilotContractError(
                "strict Gemini pilot profiles must use OpenRouter thirdparty transport"
            )
        if api_key is None:
            api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key required")
        model = model or "gemini-3-pro-preview"
        return GeminiProvider(
            api_key=api_key,
            model=model,
            max_retries=20 if max_retries is None else max_retries,
        )

    elif provider_type == "ollama":
        model = model or "llama3:8b"
        return OllamaProvider(
            model=model,
            max_retries=max_retries,
            request_profile=request_profile,
        )

    elif provider_type == "local":
        if request_profile is not None:
            raise PilotContractError(
                "strict local pilot profile requires the frozen Ollama transport"
            )
        model = model or "mlx-community/Llama-3.3-70B-Instruct-4bit"
        base_url = base_url or "http://localhost:8000/v1"
        return LocalAPIProvider(
            model=model,
            base_url=base_url,
            api_key=api_key or "not-needed",
            max_retries=10 if max_retries is None else max_retries,
        )

    elif provider_type == "thirdparty":
        if api_key is None:
            api_key = os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("Third-party API key required")
        model = model or "google/gemini-3-flash-preview"
        base_url = base_url or os.environ.get('OPENROUTER_BASE_URL', "https://openrouter.ai/api/v1")
        return ThirdPartyProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_retries=max_retries,
            request_profile=request_profile,
        )

    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
