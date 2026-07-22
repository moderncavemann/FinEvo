"""Pure, versioned decision prompts for the verified-memory runner.

The base prompt is constructed without memory so paired replay can hash-bind it once.
Memory is then placed inside an explicit, replaceable block.  Context intended for
retrieval only never enters the prompt through this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from .actions import action_contract_prompt
from .m0_utility import UtilityConfig


LEGACY_PROMPT_SCHEMA_VERSION = "verified-decision-prompt-v1"
PROMPT_SCHEMA_VERSION = "verified-decision-prompt-v2"
SUPPORTED_PROMPT_SCHEMA_VERSIONS = frozenset(
    {LEGACY_PROMPT_SCHEMA_VERSION, PROMPT_SCHEMA_VERSION}
)
MEMORY_START = "<<<VERIFIED_MEMORY_START>>>"
MEMORY_END = "<<<VERIFIED_MEMORY_END>>>"


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(name: str, value: Any) -> float:
    result = _finite(name, value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    result = " ".join(value.split())
    if MEMORY_START in result or MEMORY_END in result:
        raise ValueError(f"{name} contains a reserved memory delimiter")
    return result


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DecisionPromptState:
    decision_t: int
    agent_id: int
    name: str
    age: int
    city: str
    job: str
    offer: str
    wealth: float
    skill: float
    price: float
    interest_rate: float
    last_consumption_quantity: float
    last_labor_hours: float
    last_tax_paid: float
    last_lump_sum: float
    previous_period_available: bool = True
    max_labor_hours: float = 168.0

    def __post_init__(self) -> None:
        for field in ("decision_t", "agent_id", "age"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an integer")
        if self.decision_t < 0 or self.agent_id < 0 or self.age < 0:
            raise ValueError("decision_t, agent_id, and age must be nonnegative")
        if not isinstance(self.previous_period_available, bool):
            raise TypeError("previous_period_available must be boolean")
        for field in ("name", "city", "job", "offer"):
            object.__setattr__(self, field, _text(field, getattr(self, field)))
        for field in (
            "wealth",
            "skill",
            "price",
            "interest_rate",
            "last_consumption_quantity",
            "last_labor_hours",
            "last_tax_paid",
            "last_lump_sum",
            "max_labor_hours",
        ):
            object.__setattr__(
                self, field, _nonnegative(field, getattr(self, field))
            )
        if self.price <= 0 or self.max_labor_hours <= 0:
            raise ValueError("price and max_labor_hours must be positive")
        if self.last_labor_hours > self.max_labor_hours:
            raise ValueError("last_labor_hours exceeds max_labor_hours")

    @property
    def maximum_gross_income(self) -> float:
        return self.skill * self.max_labor_hours

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DecisionPrompt:
    schema_version: str
    base_prompt: str
    memory_text: str
    full_prompt: str
    base_prompt_hash: str
    memory_hash: str
    full_prompt_hash: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_base_decision_prompt(
    state: DecisionPromptState,
    utility_config: UtilityConfig,
    *,
    event_text: str = "",
    causal_context_summary: str = "",
) -> str:
    """Build the protected prompt bytes shared by memory interventions."""

    if not isinstance(state, DecisionPromptState):
        raise TypeError("state must be DecisionPromptState")
    if not isinstance(utility_config, UtilityConfig):
        raise TypeError("utility_config must be UtilityConfig")
    event = ""
    if event_text:
        event = _text("event_text", event_text)
    context = ""
    if causal_context_summary:
        context = _text("causal_context_summary", causal_context_summary)
    employment = (
        f"Current job: {state.job}. Current offer: {state.offer}."
    )
    prompt = (
        f"You are {state.name}, agent {state.agent_id}, age {state.age}, living in "
        f"{state.city}. This is monthly decision t={state.decision_t}. "
        f"{employment} Your wage skill is {state.skill:.6g} currency units per "
        f"hour, so working the maximum {state.max_labor_hours:.0f} hours would "
        f"produce gross income {state.maximum_gross_income:.2f} before tax. "
        f"Current savings are {state.wealth:.2f}; goods price is "
        f"{state.price:.6g}; the current savings interest rate is "
        f"{state.interest_rate:.4%}. "
    )
    if state.previous_period_available:
        prompt += (
            f"In the last completed month you worked "
            f"{state.last_labor_hours:.1f} hours, consumed "
            f"{state.last_consumption_quantity:.4f} units, paid "
            f"{state.last_tax_paid:.2f} in tax, and received "
            f"{state.last_lump_sum:.2f} in redistribution. "
        )
    else:
        prompt += (
            "No completed prior month is available. Prior labor, consumption, "
            "tax, and redistribution are unavailable and must not be interpreted "
            "as observed zeros. "
        )
    if event:
        prompt += f"Observed monthly event: {event} "
    if context:
        prompt += f"Causal context summary: {context} "
    prompt += action_contract_prompt(utility_config)
    if MEMORY_START in prompt or MEMORY_END in prompt:
        raise AssertionError("base prompt unexpectedly contains memory delimiters")
    return " ".join(prompt.split())


def compose_decision_prompt(base_prompt: str, memory_text: str = "") -> DecisionPrompt:
    """Place memory in the sole mutable prompt region and return bound hashes."""

    base = _text("base_prompt", base_prompt)
    if not isinstance(memory_text, str):
        raise TypeError("memory_text must be a string")
    memory = memory_text.strip()
    if MEMORY_START in memory or MEMORY_END in memory:
        raise ValueError("memory_text contains a reserved delimiter")
    full = f"{base}\n\n{MEMORY_START}\n{memory}\n{MEMORY_END}"
    return DecisionPrompt(
        schema_version=PROMPT_SCHEMA_VERSION,
        base_prompt=base,
        memory_text=memory,
        full_prompt=full,
        base_prompt_hash=_sha256(base),
        memory_hash=_sha256(memory),
        full_prompt_hash=_sha256(full),
    )


def state_from_mapping(value: Mapping[str, Any]) -> DecisionPromptState:
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping")
    return DecisionPromptState(**dict(value))


__all__ = [
    "DecisionPrompt",
    "DecisionPromptState",
    "MEMORY_END",
    "MEMORY_START",
    "LEGACY_PROMPT_SCHEMA_VERSION",
    "PROMPT_SCHEMA_VERSION",
    "SUPPORTED_PROMPT_SCHEMA_VERSIONS",
    "build_base_decision_prompt",
    "compose_decision_prompt",
    "state_from_mapping",
]
