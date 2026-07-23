"""Pure prompt/action functions shared by simulation and paired replay."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .m0_utility import UtilityConfig


ACTION_SCHEMA_VERSION = "verified-action-v1"


class ActionParseError(ValueError):
    """Raised when a provider response cannot satisfy the action contract."""


def _extract_json_object(content: str) -> tuple[Mapping[str, Any], int]:
    if not isinstance(content, str) or not content.strip():
        raise ActionParseError("empty action response")
    text = content.strip()
    repairs = 0
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        repairs += 1
    if "```json" in text:
        try:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        except IndexError as exc:
            raise ActionParseError("unterminated JSON code fence") from exc
        repairs += 1
    elif "```" in text:
        try:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        except IndexError as exc:
            raise ActionParseError("unterminated code fence") from exc
        repairs += 1
    elif not (text.startswith("{") and text.endswith("}")):
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ActionParseError("response does not contain a JSON object")
        text = text[start : end + 1]
        repairs += 1
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ActionParseError(f"invalid action JSON: {exc.msg}") from exc
    if not isinstance(decoded, Mapping):
        raise ActionParseError("action JSON root must be an object")
    return decoded, repairs


def _fraction(payload: Mapping[str, Any], key: str) -> tuple[float, bool]:
    if key not in payload:
        raise ActionParseError(f"action JSON is missing {key!r}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionParseError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ActionParseError(f"{key} must be finite")
    clipped = not 0.0 <= result <= 1.0
    return min(1.0, max(0.0, result)), clipped


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


@dataclass(frozen=True)
class ActionDecision:
    schema_version: str
    proposed_work_fraction: float
    proposed_consumption_fraction: float
    labor_action_index: int
    executed_labor_hours: float
    consumption_action_index: int
    executed_consumption_rate: float
    reflection: str
    repair_attempts: int
    clipped: bool
    raw_output_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def environment_action(self) -> list[int]:
        return [self.labor_action_index, self.consumption_action_index]


def parse_direct_action(
    content: str,
    *,
    max_labor_hours: float = 168.0,
    labor_step: float = 8.0,
    consumption_step: float = 0.02,
) -> ActionDecision:
    """Parse fractions and deterministically map them to environment actions.

    Unlike the legacy Bernoulli path, the work value is the intended fraction
    of maximum monthly hours.  The same mapping is used for every treatment,
    eliminating a hidden random draw from paired memory interventions.
    """

    for name, value in (
        ("max_labor_hours", max_labor_hours),
        ("labor_step", labor_step),
        ("consumption_step", consumption_step),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    max_labor_actions = int(float(max_labor_hours) // float(labor_step))
    consumption_actions = int(round(1.0 / float(consumption_step)))
    if max_labor_actions < 1 or consumption_actions < 1:
        raise ValueError("action discretization has no positive action")

    payload, repairs = _extract_json_object(content)
    work, work_clipped = _fraction(payload, "work")
    consumption, consumption_clipped = _fraction(payload, "consumption")
    labor_index = min(max_labor_actions, _round_half_up(work * max_labor_actions))
    consumption_index = min(
        consumption_actions,
        _round_half_up(consumption / float(consumption_step)),
    )
    executed_hours = labor_index * float(labor_step)
    executed_consumption = consumption_index * float(consumption_step)
    reflection = payload.get("reflection", "")
    if reflection is None:
        reflection = ""
    if not isinstance(reflection, str):
        raise ActionParseError("reflection must be a string when present")
    return ActionDecision(
        schema_version=ACTION_SCHEMA_VERSION,
        proposed_work_fraction=work,
        proposed_consumption_fraction=consumption,
        labor_action_index=labor_index,
        executed_labor_hours=executed_hours,
        consumption_action_index=consumption_index,
        executed_consumption_rate=executed_consumption,
        reflection=reflection.strip(),
        repair_attempts=repairs,
        clipped=work_clipped or consumption_clipped,
        raw_output_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def action_contract_prompt(config: UtilityConfig) -> str:
    """Return the versioned decision objective and strict JSON contract."""
    if not isinstance(config, UtilityConfig):
        raise TypeError("config must be UtilityConfig")
    rho_term = "log(1 + real consumption / q0)" if math.isclose(config.rho, 1.0) else (
        f"shifted CRRA(real consumption / q0, rho={config.rho:g})"
    )
    return (
        "Choose a monthly labor fraction and consumption fraction while balancing "
        "current consumption, labor effort, and future savings. Evaluation uses "
        f"realized flow utility {rho_term} - "
        f"{config.labor_weight:g}/(1+{config.inverse_frisch:g}) * "
        f"(labor_hours/{config.max_labor_hours:g})^(1+{config.inverse_frisch:g}); "
        "wealth and macro variables are diagnostics, not the sole objective. "
        "Return ONLY JSON with keys reflection (one short sentence), work (a "
        "number from 0 to 1 meaning fraction of maximum hours), and consumption "
        "(a number from 0 to 1 meaning fraction of available cash)."
    )


__all__ = [
    "ACTION_SCHEMA_VERSION",
    "ActionDecision",
    "ActionParseError",
    "action_contract_prompt",
    "parse_direct_action",
]
