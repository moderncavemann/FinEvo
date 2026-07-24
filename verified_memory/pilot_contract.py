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


PILOT_CONTRACT_SCHEMA_VERSION_V1 = "finevo-pilot-contract-v1"
PILOT_CONTRACT_SCHEMA_VERSION_V2 = "finevo-pilot-contract-v2"
PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_1 = (
    "finevo-pilot-contract-v2.1-amendment-overlay-v1"
)
# Backward-compatible public name.  V1 artifacts remain immutable/readable;
# callers that need the science contract should use the explicit V2 constant.
PILOT_CONTRACT_SCHEMA_VERSION = PILOT_CONTRACT_SCHEMA_VERSION_V1
PILOT_CONTRACT_CANONICALIZATION = "json-sort-keys-utf8-v1"

PILOT_CONTRACT_ID_V2 = "finevo-pilot-v2"
PILOT_CONTRACT_ID_V2_1 = "finevo-pilot-v2.1"
PILOT_CONTRACT_TAG_V2 = "pilot-v2-science"
PILOT_CONTRACT_TAG_V2_1 = "pilot-v2.1-science"
PILOT_CONTRACT_V2_CANONICAL_SHA256 = (
    "980deddf2f82a762db7d73baa6ee0428c5e653298f4f275c5b3a5b23a95865c5"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SEED_CAPABILITIES = frozenset({"best_effort", "unsupported", "deterministic"})
_TRANSPORTS = frozenset({"openai", "openrouter", "ollama", "diagnostic"})
_JSON_MODES = frozenset({"json_object", "prompt_only"})
_REASONING_MODES = frozenset({"fixed", "omitted"})
_REASONING_EFFORTS = frozenset(
    {"max", "xhigh", "high", "medium", "low", "minimal", "none"}
)
_DISPATCH_MODES = frozenset(
    {"explicit_supported", "documented_unsupported_omitted"}
)
_DECODING_FIELDS = frozenset(
    {"temperature", "top_p", "seed", "reasoning", "response_format"}
)
_MODEL_ROLES = frozenset(
    {
        "primary",
        "controlled_second",
        "secondary_diagnostic",
        "capability_no_go",
        "calibration_only",
    }
)
_SCIENCE_TASK_CAPS = {
    "capability-choice": (2048, 512),
    "capability-proposal": (4096, 4096),
    "actor-action": (2048, 1024),
    "semantic-proposal": (4096, 4096),
}

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
class DecodingFieldDispatch:
    """Per-profile disposition of one potentially unsupported request field."""

    requested_value: Any
    dispatch_mode: str
    catalog_evidence_required: bool

    def __post_init__(self) -> None:
        mode = _text(self.dispatch_mode, "decoding field dispatch_mode")
        if mode not in _DISPATCH_MODES:
            raise PilotContractError(f"unsupported dispatch mode: {mode}")
        object.__setattr__(self, "dispatch_mode", mode)
        _boolean(
            self.catalog_evidence_required,
            "decoding field catalog_evidence_required",
        )
        object.__setattr__(self, "requested_value", _freeze_json(self.requested_value))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecodingFieldDispatch":
        value = _mapping(value, "decoding field")
        _strict_keys(
            value,
            required={
                "requested_value",
                "dispatch_mode",
                "catalog_evidence_required",
            },
            name="decoding field",
        )
        return cls(
            requested_value=value["requested_value"],
            dispatch_mode=value["dispatch_mode"],
            catalog_evidence_required=value["catalog_evidence_required"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_value": _thaw_json(self.requested_value),
            "dispatch_mode": self.dispatch_mode,
            "catalog_evidence_required": self.catalog_evidence_required,
        }


@dataclass(frozen=True, slots=True)
class ParameterDispatchPolicy:
    """Uniform fail-closed policy applied to every V2 model profile."""

    policy_id: str
    fields: tuple[str, ...]
    allowed_modes: tuple[str, ...]
    unsupported_field_action: str
    unknown_support_action: str
    omission_receipt_status: str
    uniform_across_profiles: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        fields = _string_tuple(self.fields, "parameter dispatch fields")
        if frozenset(fields) != _DECODING_FIELDS:
            raise PilotContractError(
                "parameter dispatch policy must cover the five frozen decoding fields"
            )
        object.__setattr__(self, "fields", fields)
        modes = _string_tuple(self.allowed_modes, "parameter dispatch modes")
        if frozenset(modes) != _DISPATCH_MODES:
            raise PilotContractError(
                "parameter dispatch policy modes differ from the frozen V2 policy"
            )
        object.__setattr__(self, "allowed_modes", modes)
        if self.unsupported_field_action != "omit-before-dispatch":
            raise PilotContractError(
                "unsupported request fields must be omitted before dispatch"
            )
        if self.unknown_support_action != "stop-before-dispatch":
            raise PilotContractError(
                "unknown parameter support must stop before dispatch"
            )
        if self.omission_receipt_status != "omitted_unsupported":
            raise PilotContractError(
                "unsupported omissions must be recorded as omitted_unsupported"
            )
        if not _boolean(
            self.uniform_across_profiles, "parameter dispatch uniform_across_profiles"
        ):
            raise PilotContractError(
                "parameter dispatch policy must be uniform across profiles"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ParameterDispatchPolicy":
        value = _mapping(value, "parameter_dispatch_policy")
        _strict_keys(
            value,
            required={
                "policy_id",
                "fields",
                "allowed_modes",
                "unsupported_field_action",
                "unknown_support_action",
                "omission_receipt_status",
                "uniform_across_profiles",
            },
            name="parameter_dispatch_policy",
        )
        return cls(
            policy_id=value["policy_id"],
            fields=_string_tuple(value["fields"], "parameter dispatch fields"),
            allowed_modes=_string_tuple(
                value["allowed_modes"], "parameter dispatch modes"
            ),
            unsupported_field_action=value["unsupported_field_action"],
            unknown_support_action=value["unknown_support_action"],
            omission_receipt_status=value["omission_receipt_status"],
            uniform_across_profiles=value["uniform_across_profiles"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "fields": list(self.fields),
            "allowed_modes": list(self.allowed_modes),
            "unsupported_field_action": self.unsupported_field_action,
            "unknown_support_action": self.unknown_support_action,
            "omission_receipt_status": self.omission_receipt_status,
            "uniform_across_profiles": self.uniform_across_profiles,
        }


@dataclass(frozen=True, slots=True)
class TaskOutputContract:
    """Model-independent output cap and parser contract for one call role."""

    task_id: str
    max_completion_tokens: int
    max_visible_json_bytes: int
    visible_token_count_required: bool
    reasoning_token_count_required: bool
    science_parse_mode: str
    report_recovery_modes: tuple[str, ...]
    recovered_output_scientific_success: bool
    required_finish_reason: str

    def __post_init__(self) -> None:
        task_id = _text(self.task_id, "task output contract task_id")
        object.__setattr__(self, "task_id", task_id)
        expected = _SCIENCE_TASK_CAPS.get(task_id)
        if expected is None:
            raise PilotContractError(f"unknown science task output contract: {task_id}")
        cap = _integer(
            self.max_completion_tokens,
            f"{task_id}.max_completion_tokens",
            minimum=1,
        )
        byte_limit = _integer(
            self.max_visible_json_bytes,
            f"{task_id}.max_visible_json_bytes",
            minimum=2,
        )
        if (cap, byte_limit) != expected:
            raise PilotContractError(
                f"{task_id} output limits differ from the frozen V2 task cap"
            )
        if not _boolean(
            self.visible_token_count_required,
            f"{task_id}.visible_token_count_required",
        ):
            raise PilotContractError("visible token counts must be recorded")
        if not _boolean(
            self.reasoning_token_count_required,
            f"{task_id}.reasoning_token_count_required",
        ):
            raise PilotContractError("reasoning token counts must be recorded")
        if self.science_parse_mode != "exact_json_only":
            raise PilotContractError(
                "scientific success requires exact JSON without parser recovery"
            )
        recovery = _string_tuple(
            self.report_recovery_modes,
            f"{task_id}.report_recovery_modes",
        )
        if recovery != ("fenced_json", "substring_json"):
            raise PilotContractError(
                "V2 recovery reporting must cover fenced_json and substring_json"
            )
        object.__setattr__(self, "report_recovery_modes", recovery)
        if _boolean(
            self.recovered_output_scientific_success,
            f"{task_id}.recovered_output_scientific_success",
        ):
            raise PilotContractError(
                "recovered JSON cannot count as V2 scientific parse success"
            )
        if self.required_finish_reason != "stop":
            raise PilotContractError("V2 task outputs require finish_reason=stop")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskOutputContract":
        value = _mapping(value, "task output contract")
        fields = {
            "task_id",
            "max_completion_tokens",
            "max_visible_json_bytes",
            "visible_token_count_required",
            "reasoning_token_count_required",
            "science_parse_mode",
            "report_recovery_modes",
            "recovered_output_scientific_success",
            "required_finish_reason",
        }
        _strict_keys(value, required=fields, name="task output contract")
        return cls(
            task_id=value["task_id"],
            max_completion_tokens=value["max_completion_tokens"],
            max_visible_json_bytes=value["max_visible_json_bytes"],
            visible_token_count_required=value["visible_token_count_required"],
            reasoning_token_count_required=value["reasoning_token_count_required"],
            science_parse_mode=value["science_parse_mode"],
            report_recovery_modes=_string_tuple(
                value["report_recovery_modes"], "report recovery modes"
            ),
            recovered_output_scientific_success=value[
                "recovered_output_scientific_success"
            ],
            required_finish_reason=value["required_finish_reason"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "max_completion_tokens": self.max_completion_tokens,
            "max_visible_json_bytes": self.max_visible_json_bytes,
            "visible_token_count_required": self.visible_token_count_required,
            "reasoning_token_count_required": self.reasoning_token_count_required,
            "science_parse_mode": self.science_parse_mode,
            "report_recovery_modes": list(self.report_recovery_modes),
            "recovered_output_scientific_success": (
                self.recovered_output_scientific_success
            ),
            "required_finish_reason": self.required_finish_reason,
        }


@dataclass(frozen=True, slots=True)
class ModelRolePolicy:
    """Frozen scientific role and dispatch surface for one model profile."""

    profile_id: str
    role: str
    dispatch_eligible: bool
    ineligibility_reason: Optional[str]
    allowed_stages: tuple[str, ...]
    allowed_call_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _text(self.profile_id, "profile_id"))
        role = _text(self.role, "model role")
        if role not in _MODEL_ROLES:
            raise PilotContractError(f"unsupported model role: {role}")
        object.__setattr__(self, "role", role)
        eligible = _boolean(self.dispatch_eligible, "model role dispatch_eligible")
        reason = self.ineligibility_reason
        if eligible:
            if reason is not None:
                raise PilotContractError(
                    "dispatch-eligible model role cannot have an ineligibility reason"
                )
        else:
            reason = _text(reason, "model role ineligibility_reason")
            if self.allowed_stages or self.allowed_call_roles:
                raise PilotContractError(
                    "dispatch-ineligible model role cannot allow stages or call roles"
                )
        object.__setattr__(self, "ineligibility_reason", reason)
        object.__setattr__(
            self,
            "allowed_stages",
            _string_tuple(
                self.allowed_stages, "model role allowed_stages", allow_empty=not eligible
            ),
        )
        object.__setattr__(
            self,
            "allowed_call_roles",
            _string_tuple(
                self.allowed_call_roles,
                "model role allowed_call_roles",
                allow_empty=not eligible,
            ),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelRolePolicy":
        value = _mapping(value, "model role")
        _strict_keys(
            value,
            required={
                "profile_id",
                "role",
                "dispatch_eligible",
                "ineligibility_reason",
                "allowed_stages",
                "allowed_call_roles",
            },
            name="model role",
        )
        return cls(
            profile_id=value["profile_id"],
            role=value["role"],
            dispatch_eligible=value["dispatch_eligible"],
            ineligibility_reason=value["ineligibility_reason"],
            allowed_stages=_string_tuple(
                value["allowed_stages"],
                "model role allowed_stages",
                allow_empty=True,
            ),
            allowed_call_roles=_string_tuple(
                value["allowed_call_roles"],
                "model role allowed_call_roles",
                allow_empty=True,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "role": self.role,
            "dispatch_eligible": self.dispatch_eligible,
            "ineligibility_reason": self.ineligibility_reason,
            "allowed_stages": list(self.allowed_stages),
            "allowed_call_roles": list(self.allowed_call_roles),
        }


@dataclass(frozen=True, slots=True)
class DenominatorPolicy:
    """Typed ITT and inference-unit contract for the mechanism micro-pilot."""

    policy_id: str
    registered_cells_are_itt: bool
    parse_failure_outcome: str
    provider_budget_integrity_failure_outcome: str
    failed_seed_replacement: str
    seed_inference_unit: str
    rule_inference_unit: str
    checkpoint_inference_unit: str
    core_complete_pairs_min: int
    core_registered_pairs: int
    cross_model_complete_pairs_min: int
    cross_model_registered_pairs: int
    raw_paired_deltas_required: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "denominator policy"))
        if not _boolean(
            self.registered_cells_are_itt, "denominator registered_cells_are_itt"
        ):
            raise PilotContractError("all preregistered cells must remain in ITT")
        if self.parse_failure_outcome != "candidate_not_activated":
            raise PilotContractError(
                "parse failures must be counted as candidate_not_activated"
            )
        if (
            self.provider_budget_integrity_failure_outcome
            != "terminate_run_keep_denominator"
        ):
            raise PilotContractError(
                "provider/budget/integrity failures must terminate but remain in ITT"
            )
        if self.failed_seed_replacement != "forbidden":
            raise PilotContractError("failed pilot seeds cannot be replaced")
        expected_units = (
            (self.seed_inference_unit, "seed"),
            (self.rule_inference_unit, "seed-agent-family"),
            (self.checkpoint_inference_unit, "seed-checkpoint"),
        )
        if any(actual != expected for actual, expected in expected_units):
            raise PilotContractError("V2 inference units differ from preregistration")
        if (
            _integer(self.core_complete_pairs_min, "core_complete_pairs_min") != 4
            or _integer(self.core_registered_pairs, "core_registered_pairs") != 5
            or _integer(
                self.cross_model_complete_pairs_min,
                "cross_model_complete_pairs_min",
            )
            != 2
            or _integer(
                self.cross_model_registered_pairs, "cross_model_registered_pairs"
            )
            != 3
        ):
            raise PilotContractError("V2 denominator pair counts drifted")
        if not _boolean(
            self.raw_paired_deltas_required, "raw_paired_deltas_required"
        ):
            raise PilotContractError("raw paired deltas are required")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DenominatorPolicy":
        value = _mapping(value, "denominator_policy")
        fields = {
            "policy_id",
            "registered_cells_are_itt",
            "parse_failure_outcome",
            "provider_budget_integrity_failure_outcome",
            "failed_seed_replacement",
            "seed_inference_unit",
            "rule_inference_unit",
            "checkpoint_inference_unit",
            "core_complete_pairs_min",
            "core_registered_pairs",
            "cross_model_complete_pairs_min",
            "cross_model_registered_pairs",
            "raw_paired_deltas_required",
        }
        _strict_keys(value, required=fields, name="denominator_policy")
        return cls(**{key: value[key] for key in fields})

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "registered_cells_are_itt": self.registered_cells_are_itt,
            "parse_failure_outcome": self.parse_failure_outcome,
            "provider_budget_integrity_failure_outcome": (
                self.provider_budget_integrity_failure_outcome
            ),
            "failed_seed_replacement": self.failed_seed_replacement,
            "seed_inference_unit": self.seed_inference_unit,
            "rule_inference_unit": self.rule_inference_unit,
            "checkpoint_inference_unit": self.checkpoint_inference_unit,
            "core_complete_pairs_min": self.core_complete_pairs_min,
            "core_registered_pairs": self.core_registered_pairs,
            "cross_model_complete_pairs_min": self.cross_model_complete_pairs_min,
            "cross_model_registered_pairs": self.cross_model_registered_pairs,
            "raw_paired_deltas_required": self.raw_paired_deltas_required,
        }


@dataclass(frozen=True, slots=True)
class ReleaseRequirements:
    """Static CI identity and expected freeze values for a later attestor."""

    remote: str
    branch: str
    tag: str
    workflow_file: str
    workflow_name: str
    required_job_names: tuple[str, ...]
    expected_ci: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.remote != "origin":
            raise PilotContractError("V2 release remote must be origin")
        if self.branch != "main":
            raise PilotContractError("V2 release branch must be main")
        if self.tag not in {
            PILOT_CONTRACT_TAG_V2,
            PILOT_CONTRACT_TAG_V2_1,
        }:
            raise PilotContractError(
                "V2 release tag must be a registered annotated science tag"
            )
        if self.workflow_file != ".github/workflows/verified-memory-ci.yml":
            raise PilotContractError("V2 release workflow file drifted")
        if self.workflow_name != "Verified memory CI":
            raise PilotContractError("V2 release workflow name drifted")
        jobs = _string_tuple(self.required_job_names, "required_job_names")
        if jobs != (
            "Python 3.12.7 / ubuntu-24.04",
            "Python 3.12.7 / macos-14",
        ):
            raise PilotContractError("V2 release requires the frozen Linux/macOS jobs")
        object.__setattr__(self, "required_job_names", jobs)
        expected_ci = _mapping(self.expected_ci, "release expected_ci")
        expected_fields = {
            "test_count",
            "test_collection_sha256",
            "compiled_source_count",
            "compiled_source_inventory_sha256",
            "sealed_manifest_inventory_sha256",
        }
        _strict_keys(
            expected_ci,
            required=expected_fields,
            name="release expected_ci",
        )
        for name in ("test_count", "compiled_source_count"):
            value = expected_ci[name]
            if value is not None:
                _integer(value, name, minimum=1)
        for name in (
            "test_collection_sha256",
            "compiled_source_inventory_sha256",
            "sealed_manifest_inventory_sha256",
        ):
            value = expected_ci[name]
            if value is not None:
                _sha256(value, name)
        object.__setattr__(self, "expected_ci", _freeze_json(expected_ci))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReleaseRequirements":
        value = _mapping(value, "release_requirements")
        fields = {
            "remote",
            "branch",
            "tag",
            "workflow_file",
            "workflow_name",
            "required_job_names",
            "expected_ci",
        }
        _strict_keys(value, required=fields, name="release_requirements")
        return cls(
            remote=value["remote"],
            branch=value["branch"],
            tag=value["tag"],
            workflow_file=value["workflow_file"],
            workflow_name=value["workflow_name"],
            required_job_names=_string_tuple(
                value["required_job_names"], "required_job_names"
            ),
            expected_ci=_mapping(value["expected_ci"], "release expected_ci"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "remote": self.remote,
            "branch": self.branch,
            "tag": self.tag,
            "workflow_file": self.workflow_file,
            "workflow_name": self.workflow_name,
            "required_job_names": list(self.required_job_names),
            "expected_ci": _thaw_json(self.expected_ci),
        }


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

    def assert_positive_for_hosted_dispatch(self) -> None:
        """Require a conservative, nonzero frozen price for hosted dispatch."""

        self.assert_known_for_dispatch()
        if (
            float(self.dispatch_input) <= 0.0
            or float(self.dispatch_output) <= 0.0
        ):
            raise PilotContractError(
                "hosted provider dispatch input/output prices must be finite "
                "and positive"
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
    decoding_fields: tuple[tuple[str, DecodingFieldDispatch], ...] = ()
    dispatch_eligible: bool = True
    ineligibility_reason: Optional[str] = None

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
        decoding = tuple(sorted(self.decoding_fields, key=lambda item: item[0]))
        if decoding:
            decoding_keys = tuple(
                _text(key, "decoding_fields key") for key, _ in decoding
            )
            if len(decoding_keys) != len(set(decoding_keys)):
                raise PilotContractError("decoding_fields contains duplicate keys")
            if frozenset(decoding_keys) != _DECODING_FIELDS:
                raise PilotContractError(
                    "V2 profile decoding_fields must cover the five frozen fields"
                )
            if any(
                not isinstance(item, DecodingFieldDispatch) for _, item in decoding
            ):
                raise PilotContractError(
                    "decoding_fields values must be DecodingFieldDispatch objects"
                )
        object.__setattr__(self, "decoding_fields", decoding)
        eligible = _boolean(self.dispatch_eligible, "dispatch_eligible")
        reason = self.ineligibility_reason
        if eligible:
            if reason is not None:
                raise PilotContractError(
                    "dispatch-eligible profile cannot declare ineligibility_reason"
                )
        else:
            reason = _text(reason, "ineligibility_reason")
        object.__setattr__(self, "ineligibility_reason", reason)

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
            v1_keys = {"manifest_sha256", "model_layer_digest"}
            v2_keys = {
                "manifest_sha256",
                "model_layer_digest",
                "model_layer_size_bytes",
                "ollama_version",
                "adapter",
                "base_url",
            }
            key_set = frozenset(keys)
            if key_set not in {frozenset(v1_keys), frozenset(v2_keys)}:
                raise PilotContractError(
                    "local model profile requires the frozen V1 or V2 artifact identity"
                )
            _sha256(keys["manifest_sha256"], "local manifest_sha256")
            layer = keys["model_layer_digest"]
            if not layer.startswith("sha256:"):
                raise PilotContractError("local model layer digest must use sha256:")
            _sha256(layer.split(":", 1)[1], "local model layer digest")
            if key_set == frozenset(v2_keys):
                try:
                    layer_size = int(keys["model_layer_size_bytes"])
                except ValueError as exc:
                    raise PilotContractError(
                        "local model_layer_size_bytes must be an integer"
                    ) from exc
                _integer(
                    layer_size,
                    "local model_layer_size_bytes",
                    minimum=1,
                )
                _text(keys["ollama_version"], "local ollama_version")
                if keys["adapter"] != "ollama-python":
                    raise PilotContractError("local adapter must be ollama-python")
                if keys["base_url"] not in {
                    "http://127.0.0.1:11434",
                    "http://localhost:11434",
                }:
                    raise PilotContractError("local Ollama endpoint must be loopback")

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
        v2_fields = {
            "decoding_fields",
            "dispatch_eligible",
            "ineligibility_reason",
        }
        present_v2 = bool(set(value) & v2_fields)
        _strict_keys(
            value,
            required=fields | (v2_fields if present_v2 else set()),
            name="provider request profile",
        )
        artifact = _mapping(value["artifact_identity"], "artifact_identity")
        decoding: tuple[tuple[str, DecodingFieldDispatch], ...] = ()
        if present_v2:
            raw_decoding = _mapping(value["decoding_fields"], "decoding_fields")
            decoding = tuple(
                (
                    str(key),
                    DecodingFieldDispatch.from_dict(
                        _mapping(item, f"decoding_fields.{key}")
                    ),
                )
                for key, item in raw_decoding.items()
            )
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
            decoding_fields=decoding,
            dispatch_eligible=(
                value["dispatch_eligible"] if present_v2 else True
            ),
            ineligibility_reason=(
                value["ineligibility_reason"] if present_v2 else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
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
        if self.decoding_fields:
            result.update(
                {
                    "decoding_fields": {
                        key: item.to_dict() for key, item in self.decoding_fields
                    },
                    "dispatch_eligible": self.dispatch_eligible,
                    "ineligibility_reason": self.ineligibility_reason,
                }
            )
        return result

    def validate_provider_configuration(
        self,
        *,
        transport: str,
        model: str,
        max_attempts: int,
    ) -> None:
        if not self.dispatch_eligible:
            raise PilotContractError(
                f"profile {self.profile_id} is not dispatch eligible: "
                f"{self.ineligibility_reason}"
            )
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
        if self.transport in {"openai", "openrouter"}:
            self.price_snapshot.assert_positive_for_hosted_dispatch()
        else:
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
        if self.decoding_fields:
            seed_dispatch = dict(self.decoding_fields)["seed"].dispatch_mode
            if (
                seed_dispatch == "documented_unsupported_omitted"
                and seed is not None
            ):
                raise PilotContractError(
                    f"profile {self.profile_id} must omit decoding seed on the wire"
                )
            if seed_dispatch == "explicit_supported" and seed is None:
                raise PilotContractError(
                    f"profile {self.profile_id} requires the frozen decoding seed"
                )
        else:
            # Immutable V1 compatibility: the historical schema used the coarse
            # model capability field as the wire-dispatch decision.
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
    call_roles: tuple[str, ...] = ()

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
        object.__setattr__(
            self,
            "call_roles",
            _string_tuple(
                self.call_roles,
                "stage call_roles",
                allow_empty=True,
            ),
        )

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
        _strict_keys(
            value,
            required=fields,
            optional=frozenset({"call_roles"}),
            name="stage",
        )
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
            call_roles=_string_tuple(
                value.get("call_roles", ()),
                "stage call_roles",
                allow_empty=True,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
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
        if self.call_roles:
            result["call_roles"] = list(self.call_roles)
        return result


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


_V2_1_SCIENCE_DESIGN_FIELDS = (
    "seeds",
    "provider_profiles",
    "arms",
    "narratives",
    "shocks",
    "utility",
    "stop_go",
    "stages",
    "parameter_dispatch_policy",
    "task_output_contracts",
    "model_roles",
    "non_claims",
)
PILOT_CONTRACT_V2_SCIENCE_DESIGN_SHA256 = (
    "f3ea82bf587079dc5b999df71cd8bb748db7d56aa20bf759b370fd03bec82168"
)


def science_design_sha256(value: Mapping[str, Any]) -> str:
    """Hash the frozen experiment design, excluding operational budget caps."""

    payload = {
        field: _json_copy(value[field])
        for field in _V2_1_SCIENCE_DESIGN_FIELDS
    }
    denominator = _json_copy(value["denominator_policy"])
    denominator.pop("policy_id")
    payload["denominator_policy"] = denominator
    return canonical_sha256(payload)


_V2_1_EXPECTED_CI_FIELDS = {
    "test_count",
    "test_collection_sha256",
    "compiled_source_count",
    "compiled_source_inventory_sha256",
    "sealed_manifest_inventory_sha256",
}


def _validate_v2_1_expected_ci_state(
    value: Any,
    *,
    status: str,
    name: str,
) -> Mapping[str, Any]:
    """Require an all-null draft or an all-concrete frozen CI identity."""

    expected_ci = _mapping(value, name)
    _strict_keys(
        expected_ci,
        required=_V2_1_EXPECTED_CI_FIELDS,
        name=name,
    )
    null_fields = {
        field for field in _V2_1_EXPECTED_CI_FIELDS if expected_ci[field] is None
    }
    if status == "draft":
        if null_fields != _V2_1_EXPECTED_CI_FIELDS:
            raise PilotContractError(
                "V2.1 draft expected_ci must be exactly all-null"
            )
    elif status == "frozen":
        if null_fields:
            raise PilotContractError(
                "V2.1 frozen expected_ci must be exactly all-concrete"
            )
        _integer(expected_ci["test_count"], "test_count", minimum=1)
        _integer(
            expected_ci["compiled_source_count"],
            "compiled_source_count",
            minimum=1,
        )
        for field in (
            "test_collection_sha256",
            "compiled_source_inventory_sha256",
            "sealed_manifest_inventory_sha256",
        ):
            _sha256(expected_ci[field], field)
    else:
        raise PilotContractError("V2.1 status must be draft or frozen")
    return expected_ci


def _validate_v2_1_operational_amendment(
    value: Any,
) -> Mapping[str, Any]:
    """Validate the one authorized operational retry and its parent receipts."""

    amendment = _mapping(value, "operational_amendment")
    _strict_keys(
        amendment,
        required={
            "schema_version",
            "amendment_id",
            "parent",
            "failure",
            "inherited_results",
            "retry_policy",
            "budget_carry_forward",
        },
        name="operational_amendment",
    )
    if (
        amendment["schema_version"]
        != "finevo-pilot-operational-amendment-v1"
        or amendment["amendment_id"]
        != "finevo-pilot-v2.1-operational-retry-1"
    ):
        raise PilotContractError("V2.1 operational amendment identity drifted")

    parent = _mapping(amendment["parent"], "operational_amendment.parent")
    _strict_keys(
        parent,
        required={
            "contract_id",
            "contract_sha256",
            "release_tag",
            "release_commit",
            "launch_input_sha256",
            "release_attestation_sha256",
            "run_ledger_file_sha256",
            "run_ledger_internal_sha256",
            "budget_ledger_file_sha256",
            "budget_ledger_internal_sha256",
        },
        name="operational_amendment.parent",
    )
    expected_parent = {
        "contract_id": PILOT_CONTRACT_ID_V2,
        "contract_sha256": PILOT_CONTRACT_V2_CANONICAL_SHA256,
        "release_tag": PILOT_CONTRACT_TAG_V2,
        "release_commit": "3664778727813e5e8328b4b17b91a28c8122f87c",
        "launch_input_sha256": (
            "6516ce8660d588aaf13381353f67d9cacd991d5236a7d3ae8c41ef1c0a88d357"
        ),
        "release_attestation_sha256": (
            "54a11dc86df139a3656934ff81920ae1f10c9425afa44815e30c9befda583895"
        ),
        "run_ledger_file_sha256": (
            "34b1a763f4f1c5824249e4acaaa83334f2b254eb899b4def14a4c3365eefd60f"
        ),
        "run_ledger_internal_sha256": (
            "9d54ac1f22a56bafbe59164c7074d87bf914d290bc1282c631d50a4529f41fff"
        ),
        "budget_ledger_file_sha256": (
            "f1318f47977ad956e206dfb53ff8a14350338c691b17f44280121504a99d2882"
        ),
        "budget_ledger_internal_sha256": (
            "d9ec2c1bdfcc407aeb555ba71ee9d5e274d924e9a98791c5839ec749f3b1a0f2"
        ),
    }
    if _json_copy(parent) != expected_parent:
        raise PilotContractError("V2.1 parent release binding drifted")

    failure = _mapping(amendment["failure"], "operational_amendment.failure")
    _strict_keys(
        failure,
        required={
            "affected_run_id",
            "error_type",
            "failure_count",
            "capability_sha256",
            "gate_sha256",
            "terminal_sha256",
            "served_model_observation",
            "capability_status",
            "parent_terminal_status",
            "root_cause_codes",
            "secret_rotation_required",
        },
        name="operational_amendment.failure",
    )
    expected_failure = {
        "affected_run_id": (
            "finevo-pilot-v2--capability-gate--gpt52_main--capability-probe--"
            "none--provider-preflight-default--s2010922376"
        ),
        "error_type": "APIConnectionError",
        "failure_count": 30,
        "capability_sha256": (
            "da9076389db58fd682d213ccb932d66bb767f73423e5476abea788eb1f8fd294"
        ),
        "gate_sha256": (
            "176547171d88dad5e757dc1795cef749bea57ea7e7291191a240c8cd92c57997"
        ),
        "terminal_sha256": (
            "10b5ff7c78b4697b9754c809bed0e7d14380729a640632585085ad7f886704c6"
        ),
        "served_model_observation": "null-pre-response",
        "capability_status": "not_evaluable",
        "parent_terminal_status": "capability-no-go",
        "root_cause_codes": [
            "credential-header-trailing-whitespace",
            "capability-reader-nullability-drift",
        ],
        "secret_rotation_required": True,
    }
    if _json_copy(failure) != expected_failure:
        raise PilotContractError("V2.1 retry failure binding drifted")

    inherited = amendment["inherited_results"]
    if (
        isinstance(inherited, (str, bytes))
        or not isinstance(inherited, Sequence)
        or len(inherited) != 1
    ):
        raise PilotContractError(
            "V2.1 must inherit exactly one parent capability result"
        )
    inherited_result = _mapping(
        inherited[0],
        "operational_amendment.inherited_results[0]",
    )
    _strict_keys(
        inherited_result,
        required={
            "model_id",
            "run_id",
            "status",
            "capability_sha256",
            "gate_sha256",
            "terminal_sha256",
            "scores",
        },
        name="operational_amendment.inherited_results[0]",
    )
    scores = _mapping(
        inherited_result["scores"],
        "operational_amendment.inherited_results[0].scores",
    )
    _strict_keys(
        scores,
        required={
            "utility_ranking",
            "rule_application",
            "rule_proposal",
        },
        name="operational_amendment.inherited_results[0].scores",
    )
    for score_name in (
        "utility_ranking",
        "rule_application",
        "rule_proposal",
    ):
        _strict_keys(
            _mapping(
                scores[score_name],
                (
                    "operational_amendment.inherited_results[0].scores."
                    f"{score_name}"
                ),
            ),
            required={"correct", "denominator"},
            name=(
                "operational_amendment.inherited_results[0].scores."
                f"{score_name}"
            ),
        )
    expected_inherited = {
        "model_id": "llama33_local_controlled",
        "run_id": (
            "finevo-pilot-v2--capability-gate--llama33_local_controlled--"
            "capability-probe--none--provider-preflight-default--s2010922376"
        ),
        "status": "capability-no-go",
        "capability_sha256": (
            "4c4c864733f32166c286e22b446dc3849df624a267ad083426ee4a89e79052ca"
        ),
        "gate_sha256": (
            "01c61c25e7d25577975dbe3aae8a408f464d685210b071aa612f4bd46bb78eda"
        ),
        "terminal_sha256": (
            "544b409e6ce8538958ec6278f5311429f14cf24591f8479bff287512d02e7380"
        ),
        "scores": {
            "utility_ranking": {"correct": 12, "denominator": 12},
            "rule_application": {"correct": 10, "denominator": 12},
            "rule_proposal": {"correct": 0, "denominator": 6},
        },
    }
    if _json_copy(inherited_result) != expected_inherited:
        raise PilotContractError("V2.1 inherited capability result drifted")

    retry = _mapping(
        amendment["retry_policy"],
        "operational_amendment.retry_policy",
    )
    _strict_keys(
        retry,
        required={
            "eligible_model_ids",
            "ineligible_parent_terminal_model_ids",
            "preserve_parent_denominator",
            "retry_is_operational_amendment",
            "unchanged_science_fields",
            "failed_seed_replacement",
            "outcome_inspected_for_retry",
        },
        name="operational_amendment.retry_policy",
    )
    expected_retry = {
        "eligible_model_ids": ["gpt52_main"],
        "ineligible_parent_terminal_model_ids": [
            "llama33_local_controlled"
        ],
        "preserve_parent_denominator": True,
        "retry_is_operational_amendment": True,
        "unchanged_science_fields": "science-critical-v2-fieldset",
        "failed_seed_replacement": "forbidden",
        "outcome_inspected_for_retry": False,
    }
    if _json_copy(retry) != expected_retry:
        raise PilotContractError("V2.1 operational retry policy drifted")

    carry = _mapping(
        amendment["budget_carry_forward"],
        "operational_amendment.budget_carry_forward",
    )
    _strict_keys(
        carry,
        required={
            "source_stage_bucket",
            "cost_usd",
            "hosted_completions",
            "storage_bytes",
        },
        name="operational_amendment.budget_carry_forward",
    )
    expected_carry = {
        "source_stage_bucket": "capability",
        "cost_usd": 1.0701145,
        "hosted_completions": 30,
        "storage_bytes": 479367,
    }
    if _json_copy(carry) != expected_carry:
        raise PilotContractError("V2.1 parent budget carry-forward drifted")
    return _freeze_json(amendment)


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
    parameter_dispatch_policy: Optional[ParameterDispatchPolicy]
    task_output_contracts: Mapping[str, TaskOutputContract]
    model_roles: Mapping[str, ModelRolePolicy]
    denominator_policy: Optional[DenominatorPolicy]
    release_requirements: Optional[ReleaseRequirements]
    operational_amendment: Optional[Mapping[str, Any]]
    non_claims: tuple[str, ...]
    canonicalization: str
    declared_sha256: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PilotContract":
        value = _mapping(value, "pilot contract")
        base_fields = {
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
        v2_fields = {
            "parameter_dispatch_policy",
            "task_output_contracts",
            "model_roles",
            "denominator_policy",
            "release_requirements",
        }
        schema_version = value.get("schema_version")
        if schema_version == PILOT_CONTRACT_SCHEMA_VERSION_V1:
            fields = base_fields
            is_v2 = False
            is_v2_1 = False
        elif schema_version == PILOT_CONTRACT_SCHEMA_VERSION_V2:
            fields = base_fields | v2_fields
            is_v2 = True
            contract_id = value.get("contract_id")
            is_v2_1 = contract_id == PILOT_CONTRACT_ID_V2_1
            if is_v2_1:
                fields = fields | {"operational_amendment"}
        else:
            raise PilotContractError("unsupported pilot contract schema")
        _strict_keys(value, required=fields, name="pilot contract")
        if value["status"] != "frozen" and not (
            is_v2_1 and value["status"] == "draft"
        ):
            raise PilotContractError(
                "pilot contract status must be frozen, except a V2.1 draft"
            )
        if is_v2 and value["contract_id"] not in {
            PILOT_CONTRACT_ID_V2,
            PILOT_CONTRACT_ID_V2_1,
        }:
            raise PilotContractError("unsupported V2 contract_id")
        if (
            is_v2_1
            and science_design_sha256(value)
            != PILOT_CONTRACT_V2_SCIENCE_DESIGN_SHA256
        ):
            raise PilotContractError(
                "V2.1 science-design fieldset differs from frozen V2"
            )

        implementation = _mapping(value["implementation"], "implementation")
        implementation_fields = {
            "required_git_tag",
            "commit_resolution",
            "required_git_commit",
            "p0_base_commit",
            "require_clean_worktree",
        }
        if is_v2:
            implementation_fields.add("required_git_branch")
        _strict_keys(
            implementation,
            required=implementation_fields,
            name="implementation",
        )
        if implementation["commit_resolution"] != "annotated_tag_peel":
            raise PilotContractError(
                "implementation commit_resolution must be annotated_tag_peel"
            )
        _text(implementation["required_git_tag"], "required_git_tag")
        _git_commit(implementation["p0_base_commit"], "p0_base_commit")
        _boolean(implementation["require_clean_worktree"], "require_clean_worktree")
        if is_v2:
            expected_tag = (
                PILOT_CONTRACT_TAG_V2_1
                if is_v2_1
                else PILOT_CONTRACT_TAG_V2
            )
            if implementation["required_git_tag"] != expected_tag:
                raise PilotContractError(
                    f"{value['contract_id']} must require {expected_tag}"
                )
            if implementation["required_git_branch"] != "main":
                raise PilotContractError("V2 implementation branch must be main")
        required_commit = implementation["required_git_commit"]
        if required_commit is not None:
            _git_commit(required_commit, "required_git_commit")
        elif implementation["commit_resolution"] != "annotated_tag_peel":
            raise PilotContractError(
                "null required_git_commit requires annotated_tag_peel"
            )

        parameter_dispatch_policy: Optional[ParameterDispatchPolicy] = None
        task_output_contracts: dict[str, TaskOutputContract] = {}
        model_roles: dict[str, ModelRolePolicy] = {}
        denominator_policy: Optional[DenominatorPolicy] = None
        release_requirements: Optional[ReleaseRequirements] = None
        operational_amendment: Optional[Mapping[str, Any]] = None
        if is_v2:
            parameter_dispatch_policy = ParameterDispatchPolicy.from_dict(
                _mapping(
                    value["parameter_dispatch_policy"],
                    "parameter_dispatch_policy",
                )
            )
            task_rows = _mapping(
                value["task_output_contracts"], "task_output_contracts"
            )
            for task_id, row in task_rows.items():
                task = TaskOutputContract.from_dict(
                    _mapping(row, f"task_output_contracts.{task_id}")
                )
                if task.task_id != task_id:
                    raise PilotContractError(
                        f"task output key {task_id!r} does not match task_id"
                    )
                task_output_contracts[task_id] = task
            if set(task_output_contracts) != set(_SCIENCE_TASK_CAPS):
                raise PilotContractError(
                    "V2 task_output_contracts must define exactly four call roles"
                )
            role_rows = _mapping(value["model_roles"], "model_roles")
            for profile_id, row in role_rows.items():
                role = ModelRolePolicy.from_dict(
                    _mapping(row, f"model_roles.{profile_id}")
                )
                if role.profile_id != profile_id:
                    raise PilotContractError(
                        f"model role key {profile_id!r} does not match profile_id"
                    )
                model_roles[profile_id] = role
            denominator_policy = DenominatorPolicy.from_dict(
                _mapping(value["denominator_policy"], "denominator_policy")
            )
            release_requirements = ReleaseRequirements.from_dict(
                _mapping(value["release_requirements"], "release_requirements")
            )
            expected_tag = (
                PILOT_CONTRACT_TAG_V2_1
                if is_v2_1
                else PILOT_CONTRACT_TAG_V2
            )
            if release_requirements.tag != expected_tag:
                raise PilotContractError(
                    "release tag differs from implementation contract version"
                )
            if is_v2_1:
                operational_amendment = _validate_v2_1_operational_amendment(
                    value["operational_amendment"]
                )
                _validate_v2_1_expected_ci_state(
                    release_requirements.expected_ci,
                    status=str(value["status"]),
                    name="release expected_ci",
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
            if profile.transport in {"openai", "openrouter"}:
                profile.price_snapshot.assert_positive_for_hosted_dispatch()
            else:
                profile.price_snapshot.assert_known_for_dispatch()
            if is_v2 and not profile.decoding_fields:
                raise PilotContractError(
                    f"V2 profile {profile_id} lacks decoding_fields"
                )
            profiles[profile_id] = profile
        if not profiles:
            raise PilotContractError("provider_profiles must not be empty")
        if is_v2:
            if set(model_roles) != set(profiles):
                raise PilotContractError(
                    "V2 model_roles must cover every provider profile exactly"
                )
            for profile_id, profile in profiles.items():
                role = model_roles[profile_id]
                if profile.dispatch_eligible != role.dispatch_eligible:
                    raise PilotContractError(
                        f"profile/model-role dispatch eligibility differs for {profile_id}"
                    )
                if profile.ineligibility_reason != role.ineligibility_reason:
                    raise PilotContractError(
                        f"profile/model-role ineligibility reason differs for {profile_id}"
                    )
            opus = profiles.get("opus48_no_go")
            opus_role = model_roles.get("opus48_no_go")
            if (
                opus is None
                or opus_role is None
                or opus.dispatch_eligible
                or opus.ineligibility_reason
                != "cross_model_budget_no_go_under_nonshrink_policy"
                or opus_role.role != "capability_no_go"
            ):
                raise PilotContractError(
                    "Opus must remain zero-dispatch under the frozen "
                    "cross-model non-shrink budget gate"
                )
            for profile_id, profile in profiles.items():
                decoding = dict(profile.decoding_fields)
                if set(decoding) != set(parameter_dispatch_policy.fields):
                    raise PilotContractError(
                        f"profile {profile_id} does not implement uniform dispatch fields"
                    )
                if profile.transport in {"openai", "openrouter"} and any(
                    not field.catalog_evidence_required
                    for field in decoding.values()
                ):
                    raise PilotContractError(
                        f"hosted profile {profile_id} requires catalog evidence "
                        "for every dispatch disposition"
                    )
                seed_dispatch = decoding["seed"]
                if (
                    profile.seed_capability == "unsupported"
                    and seed_dispatch.dispatch_mode
                    != "documented_unsupported_omitted"
                ):
                    raise PilotContractError(
                        f"seed-unsupported profile {profile_id} must omit seed"
                    )
                response_dispatch = decoding["response_format"]
                if (
                    profile.json_mode == "json_object"
                    and response_dispatch.dispatch_mode != "explicit_supported"
                ):
                    raise PilotContractError(
                        f"JSON profile {profile_id} must explicitly dispatch "
                        "response_format"
                    )
            local = profiles.get("llama33_local_controlled")
            if (
                local is None
                or local.transport != "ollama"
                or local.json_mode != "json_object"
                or set(dict(local.artifact_identity))
                != {
                    "manifest_sha256",
                    "model_layer_digest",
                    "model_layer_size_bytes",
                    "ollama_version",
                    "adapter",
                    "base_url",
                }
            ):
                raise PilotContractError(
                    "controlled local Llama must freeze JSON mode and runtime identity"
                )

        seeds = _mapping(value["seeds"], "seeds")
        _strict_keys(
            seeds,
            required={
                "generation",
                "preflight_seed",
                "sets",
                *(("failed_seed_replacement",) if is_v2 else ()),
            },
            name="seeds",
        )
        if is_v2 and seeds["failed_seed_replacement"] != "forbidden":
            raise PilotContractError("V2 failed seeds cannot be replaced")
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
        if is_v2:
            if main_values != (
                1099057501,
                1421875452,
                1769977770,
                959809858,
                617806385,
            ):
                raise PilotContractError("V2 main seed registry drifted")
            if preflight_seed != 2010922376:
                raise PilotContractError("V2 preflight seed drifted")
            if calibration_values != (1942013315, 760687867):
                raise PilotContractError("V2 calibration seed registry drifted")

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
        if is_v2:
            shock = _mapping(
                shocks.get("registered-rate-shock"),
                "shocks.registered-rate-shock",
            )
            expected_schedule = (
                {
                    "start": 0,
                    "end": 4,
                    "interest_rate": 0.03,
                    "phase": "pre-shock",
                },
                {
                    "start": 5,
                    "end": 7,
                    "interest_rate": 0.08,
                    "phase": "shock",
                },
                {
                    "start": 8,
                    "end": 11,
                    "interest_rate": 0.03,
                    "phase": "recovery",
                },
            )
            schedule = shock.get("schedule")
            if (
                isinstance(schedule, (str, bytes))
                or not isinstance(schedule, Sequence)
                or tuple(dict(_mapping(row, "shock schedule row")) for row in schedule)
                != expected_schedule
            ):
                raise PilotContractError("V2 registered shock schedule drifted")
            hook = _mapping(shock.get("hook_semantics"), "shock hook_semantics")
            if dict(hook) != {
                "prompt_effective_before_decision": True,
                "environment_effective_before_step": True,
                "write_independent_event_stream": True,
                "future_values_hidden": True,
            }:
                raise PilotContractError("V2 shock hook semantics drifted")

            expected_budget = {
                "total_usd": 25.0,
                "max_provider_completions": 7500,
                "completion_scope": "hosted-api-only",
                "max_storage_bytes": 5_000_000_000,
                "automatic_reserve_usd": 1.0,
            }
            if any(budgets.get(key) != expected for key, expected in expected_budget.items()):
                raise PilotContractError("V2 global budget limits drifted")
            caps = _mapping(budgets.get("stage_usd_caps"), "budgets.stage_usd_caps")
            expected_caps = (
                {
                    "capability": 3.0701145,
                    "calibration": 3.0,
                    "core": 13.0,
                    "cross_model": 4.9298855,
                    "manual_reserve": 1.0,
                }
                if is_v2_1
                else {
                    "capability": 2.0,
                    "calibration": 3.0,
                    "core": 13.0,
                    "cross_model": 6.0,
                    "manual_reserve": 1.0,
                }
            )
            if dict(caps) != expected_caps:
                raise PilotContractError("V2 stage budget caps drifted")
            projection = _mapping(
                budgets.get("pre_dispatch_projection"),
                "budgets.pre_dispatch_projection",
            )
            if dict(projection) != {
                "required": True,
                "basis": "model-by-call-role preflight p95",
                "reserve_multiplier": 1.25,
                "unknown_price_policy": "stop-before-dispatch",
                "over_budget_policy": "no-go-no-matrix-shrink",
            }:
                raise PilotContractError("V2 budget projection policy drifted")

            q_ref = _mapping(utility.get("q_ref_resolution"), "utility.q_ref_resolution")
            if (
                q_ref.get("seed") != 2010922376
                or q_ref.get("num_agents") != 4
                or q_ref.get("episode_length") != 12
                or q_ref.get("aggregation") != "median"
                or q_ref.get("gate") != "finite_and_strictly_positive"
                or tuple(q_ref.get("work_fraction_cycle", ()))
                != (0.25, 0.5, 0.75, 0.5)
                or tuple(q_ref.get("consumption_fraction_cycle", ()))
                != (0.3, 0.35, 0.3, 0.25)
                or q_ref.get("expected_rows") != 48
            ):
                raise PilotContractError("V2 q_ref calibration contract drifted")
            selection = _mapping(utility.get("selection_rule"), "utility.selection_rule")
            if (
                selection.get("method") != "guardrail-then-registered-tiebreak-v1"
                or selection.get("outcome_blind") is not True
                or tuple(selection.get("tiebreak_order", ()))
                != (
                    "maximize mean interior action coverage",
                    "minimize component-balance log distance from one",
                    "minimize normalized center distance",
                    "declaration order only for an exact remaining tie",
                )
            ):
                raise PilotContractError(
                    "V2 utility selection must remain outcome-blind"
                )
            profile_fields = (
                "rho",
                "labor_weight",
                "inverse_frisch",
                "consumption_scale",
                "consumption_scale_multiplier_of_q_ref",
                "discount_factor",
                "evidence_use",
            )
            expected_profile_signatures = {
                "provider-preflight-default": (
                    1.0,
                    2.0,
                    1.0,
                    1.0,
                    None,
                    0.99,
                    "capability-only",
                ),
                "center": (
                    1.0,
                    2.0,
                    1.0,
                    None,
                    1.0,
                    0.99,
                    "stage0-candidate",
                ),
                "psi-1": (
                    1.0,
                    1.0,
                    1.0,
                    None,
                    1.0,
                    0.99,
                    "stage0-candidate",
                ),
                "psi-4": (
                    1.0,
                    4.0,
                    1.0,
                    None,
                    1.0,
                    0.99,
                    "stage0-candidate",
                ),
                "nu-0.5": (
                    1.0,
                    2.0,
                    0.5,
                    None,
                    1.0,
                    0.99,
                    "stage0-candidate",
                ),
                "nu-2": (
                    1.0,
                    2.0,
                    2.0,
                    None,
                    1.0,
                    0.99,
                    "stage0-candidate",
                ),
                "q0-0.5x": (
                    1.0,
                    2.0,
                    1.0,
                    None,
                    0.5,
                    0.99,
                    "stage0-candidate",
                ),
                "q0-2x": (
                    1.0,
                    2.0,
                    1.0,
                    None,
                    2.0,
                    0.99,
                    "stage0-candidate",
                ),
                "stage0-selected": (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "resolved-from-stage0-selection-artifact",
                ),
            }
            profile_signatures = {
                profile_id: tuple(
                    _mapping(row, f"utility.profiles.{profile_id}").get(field)
                    for field in profile_fields
                )
                for profile_id, row in utility_profiles.items()
            }
            if profile_signatures != expected_profile_signatures:
                raise PilotContractError("V2 utility OFAT profile grid drifted")
            calibration_gate = _mapping(
                stop_go.get("calibration"), "stop_go.calibration"
            )
            expected_calibration = {
                "max_abs_budget_residual": 1e-8,
                "clipping_count": 0,
                "ceiling_labor_rate_max": 0.5,
                "zero_labor_rate_max": 0.25,
                "interior_labor_rate_min": 0.5,
                "interior_consumption_rate_min": 0.75,
                "median_labor_disutility_to_consumption_utility": [0.5, 2.0],
                "no_candidate_action": "stop",
            }
            if _json_copy(calibration_gate) != expected_calibration:
                raise PilotContractError("V2 calibration stop/go contract drifted")
            expected_capability = {
                "required_task_families": [
                    "action-generation",
                    "m3-proposal",
                    "evidence-citation",
                    "context-scope",
                    "strict-json",
                    "long-memory-context",
                ],
                "interface_valid_required": True,
                "strict_parse_required": True,
                "semantic_candidate_acceptance_required": True,
                "all_provider_and_parse_outcomes_in_denominator": True,
                "recovery_is_report_only": True,
                "truncation_is_failure": True,
            }
            if _json_copy(
                _mapping(stop_go.get("capability"), "stop_go.capability")
            ) != expected_capability:
                raise PilotContractError("V2 capability gate contract drifted")
            expected_preflight = {
                "action_parse_success": "12/12",
                "semantic_proposals_all_accounted": True,
                "clipping_count": 0,
                "provider_failure_count": 0,
                "route_metadata_complete": True,
                "usage_metadata_complete": True,
                "cost_metadata_complete": True,
                "served_model_exact": True,
                "provider_pin_exact": True,
                "fallback_observed": False,
                "attempts_per_request": 1,
            }
            if _json_copy(
                _mapping(
                    stop_go.get("closed_loop_preflight"),
                    "stop_go.closed_loop_preflight",
                )
            ) != expected_preflight:
                raise PilotContractError("V2 closed-loop preflight gate drifted")
            expected_a = {
                "complete_pairs_min": 4,
                "same_direction_min": 4,
                "total_registered_pairs": 5,
                "median_relative_effect_min": 0.05,
                "route_manipulation_checks_required": True,
            }
            if _json_copy(
                _mapping(stop_go.get("experiment_a"), "stop_go.experiment_a")
            ) != expected_a:
                raise PilotContractError("V2 Experiment A stop/go drifted")
            if _json_copy(
                _mapping(
                    stop_go.get("core_completeness"),
                    "stop_go.core_completeness",
                )
            ) != {
                "complete_pairs_min": 4,
                "total_registered_pairs": 5,
                "failed_and_missing_runs_remain_in_itt_denominator": True,
            }:
                raise PilotContractError("V2 core completeness policy drifted")
            if _json_copy(
                _mapping(stop_go.get("cross_model"), "stop_go.cross_model")
            ) != {
                "reportable_complete_pairs_min": 2,
                "total_registered_pairs": 3,
                "direction_replication_complete_pairs": 3,
                "direction_replication_requires_capability_pass": True,
                "seed_unsupported_directional_replication_requires_registered_matched_a_a_null": True,
                "missing_matched_a_a_null_action": (
                    "uncalibrated-diagnostic-no-registered-matched-a-a-null"
                ),
            }:
                raise PilotContractError("V2 cross-model stop/go drifted")
            if _json_copy(
                _mapping(stop_go.get("global"), "stop_go.global")
            ) != {
                "contract_hash_match": True,
                "annotated_tag_peel_match": True,
                "clean_worktree_required": True,
                "all_registered_runs_have_terminal_ledger_rows": True,
                "provider_and_parse_failures_remain_in_denominator": True,
                "budget_or_storage_projection_failure": "stop-before-dispatch",
            }:
                raise PilotContractError("V2 global stop/go drifted")

        experiment_c = _mapping(
            stop_go.get("experiment_c"), "stop_go.experiment_c"
        )
        if is_v2:
            if tuple(experiment_c.get("required_directions", ())) != (
                "verifier lowers false activation",
                "verifier lowers harmful exposure",
                "verifier lowers cumulative utility loss",
            ) or experiment_c.get("failure_action") != (
                "withdraw-or-narrow-rule-reliability-claim"
            ):
                raise PilotContractError("V2 Experiment C stop/go drifted")
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
        if is_v2:
            expected_d_scalars = {
                "complete_pairs_min": 4,
                "same_direction_min": 4,
                "total_registered_pairs": 5,
                "effect_exceeds_matched_a_b_max_null": True,
                "effect_exceeds_one_action_bin": True,
            }
            if any(
                experiment_d.get(key) != expected
                for key, expected in expected_d_scalars.items()
            ):
                raise PilotContractError("V2 Experiment D stop/go drifted")
            expected_memory_pulse = {
                "schema_version": "finevo-pilot-d-memory-pulse-v1",
                "treatment_arms": [
                    "no-memory",
                    "shuffled-episodic",
                    "wrong-context",
                ],
                "focal_agent_id": 0,
                "wrong_context_source_agent_id": 1,
                "decision_t": 6,
                "duration_decisions": 1,
                "continuation_horizon_steps": 6,
                "pulse_at_first_continuation_step": True,
                "direct_treatment_only_at_pulse": True,
                "claim_label": (
                    "focal-agent decision-6 memory pulse with six-step "
                    "downstream continuation"
                ),
            }
            expected_shuffle = {
                "algorithm": (
                    "checkpoint-bound-sha256-rank-permutation-v1"
                ),
                "non_identity_required": True,
                "fixed_reversal_prohibited_for_three_or_more_items": True,
                "checkpoint_hash_bound": True,
            }
            expected_journal = {
                "required": True,
                "calls_per_branch": 24,
                "completion_events_per_branch": 24,
                "terminal_parse_dispositions_per_branch": 24,
                "raw_output_storage": "sha256-and-byte-count-only",
            }
            expected_narrative_pulse = {
                "schema_version": "finevo-pilot-d-narrative-pulse-v1",
                "treatment_narratives": [
                    "aligned",
                    "paraphrase",
                    "opposite",
                ],
                "focal_agent_id": 0,
                "decision_t": 6,
                "duration_decisions": 1,
                "continuation_horizon_steps": 6,
                "pulse_at_first_continuation_step": True,
                "direct_treatment_only_at_pulse": True,
            }
            expected_narrative_gate = {
                "primary_contrast": "aligned-minus-opposite",
                "directional_action_metric": (
                    "focal_first_consumption_rate"
                ),
                "expected_sign": "negative",
                "same_direction_min": 4,
                "must_exceed_matched_a_b_max_null": True,
                "must_exceed_one_consumption_action_bin": True,
                "labor_action_metric": "diagnostic-only",
                "paraphrase_equivalence": (
                    "aligned-within-one-labor-and-consumption-action-bin"
                ),
            }
            expected_nested = {
                "memory_pulse_contract": expected_memory_pulse,
                "shuffle_policy": expected_shuffle,
                "branch_provider_call_journal": expected_journal,
                "narrative_pulse_contract": expected_narrative_pulse,
                "narrative_semantic_gate": expected_narrative_gate,
                "source_schema_versions": {
                    "continuation": "finevo-pilot-continuation-v2",
                    "narrative": "finevo-pilot-narrative-v2",
                },
            }
            if any(
                _json_copy(
                    _mapping(
                        experiment_d.get(key),
                        f"stop_go.experiment_d.{key}",
                    )
                )
                != expected
                for key, expected in expected_nested.items()
            ):
                raise PilotContractError(
                    "V2 Experiment D pulse/journal/narrative contract drifted"
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
        if is_v2:
            expected_stage_order = (
                "capability-gate",
                "closed-loop-preflight",
                "secondary-capability-gate",
                "secondary-closed-loop-preflight",
                "q-ref-resolution",
                "stage0-calibration",
                "experiment-a",
                "experiment-c",
                "experiment-d",
                "experiment-b",
                "controlled-second",
                "cross-model-diagnostics",
            )
            if stage_ids != expected_stage_order:
                raise PilotContractError(
                    "V2 stages must keep capability/preflight split and A-C-D-B order"
                )
            stage_map = {stage.stage_id: stage for stage in stages}
            models_by_stage = {
                stage.stage_id: {
                    model for cell in stage.cells for model in cell.models
                }
                for stage in stages
            }
            primary_gate_models = {"gpt52_main", "llama33_local_controlled"}
            secondary_gate_models = {
                "gpt56_diagnostic",
                "gemini35_flash_diagnostic",
                "llama4_maverick_diagnostic",
            }
            if (
                models_by_stage["capability-gate"] != primary_gate_models
                or models_by_stage["closed-loop-preflight"] != primary_gate_models
                or models_by_stage["secondary-capability-gate"]
                != secondary_gate_models
                or models_by_stage["secondary-closed-loop-preflight"]
                != secondary_gate_models
            ):
                raise PilotContractError(
                    "V2 primary and secondary capability tiers drifted"
                )
            if (
                stage_map["capability-gate"].budget_bucket != "capability"
                or stage_map["closed-loop-preflight"].budget_bucket != "capability"
                or stage_map["secondary-capability-gate"].budget_bucket
                != "cross_model"
                or stage_map["secondary-closed-loop-preflight"].budget_bucket
                != "cross_model"
            ):
                raise PilotContractError(
                    "V2 capability tiers must use their frozen budget buckets"
                )
            if stage_map["secondary-capability-gate"].prerequisites != (
                "closed-loop-preflight",
            ) or stage_map[
                "secondary-closed-loop-preflight"
            ].prerequisites != ("secondary-capability-gate",):
                raise PilotContractError(
                    "V2 secondary gates must follow primary closed-loop preflight"
                )
            if "secondary-closed-loop-preflight" not in stage_map[
                "cross-model-diagnostics"
            ].prerequisites:
                raise PilotContractError(
                    "cross-model diagnostics require secondary preflight"
                )
            if "experiment-b" in stage_map["experiment-c"].prerequisites:
                raise PilotContractError("Experiment C cannot depend on Experiment B")
            if stage_map["experiment-d"].prerequisites != (
                "experiment-a",
                "experiment-c",
            ):
                raise PilotContractError("Experiment D must depend on A and C only")
            if "experiment-d" not in stage_map["experiment-b"].prerequisites:
                raise PilotContractError("Experiment B must run after Experiment D")

            expected_roles = {
                "gpt52_main": "primary",
                "llama33_local_controlled": "controlled_second",
                "gpt56_diagnostic": "secondary_diagnostic",
                "gemini35_flash_diagnostic": "secondary_diagnostic",
                "llama4_maverick_diagnostic": "secondary_diagnostic",
                "opus48_no_go": "capability_no_go",
                "qref_scripted": "calibration_only",
            }
            if {
                key: role.role for key, role in model_roles.items()
            } != expected_roles:
                raise PilotContractError("V2 model scientific roles drifted")
            allowed_special_roles = {
                "qref-scripted",
                "offline-verifier",
                "checkpoint-branch",
            }
            for role in model_roles.values():
                if not set(role.allowed_stages) <= set(stage_ids):
                    raise PilotContractError(
                        f"model role {role.profile_id} references an unknown stage"
                    )
                if not set(role.allowed_call_roles) <= {
                    *task_output_contracts,
                    *allowed_special_roles,
                }:
                    raise PilotContractError(
                        f"model role {role.profile_id} references an unknown call role"
                    )
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
            if is_v2:
                if not stage.call_roles:
                    raise PilotContractError(
                        f"V2 stage {stage.stage_id} must declare call_roles"
                    )
                if not set(stage.call_roles) <= {
                    *task_output_contracts,
                    "qref-scripted",
                    "offline-verifier",
                    "checkpoint-branch",
                }:
                    raise PilotContractError(
                        f"stage {stage.stage_id} has an unknown V2 call role"
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
                if is_v2:
                    for model_id in cell.models:
                        role = model_roles[model_id]
                        if not role.dispatch_eligible:
                            raise PilotContractError(
                                f"dispatch-ineligible profile {model_id} appears in "
                                f"stage {stage.stage_id}"
                            )
                        if stage.stage_id not in role.allowed_stages:
                            raise PilotContractError(
                                f"profile {model_id} is not eligible for stage "
                                f"{stage.stage_id}"
                            )
                        if not set(stage.call_roles) <= set(role.allowed_call_roles):
                            raise PilotContractError(
                                f"profile {model_id} is not eligible for all call "
                                f"roles in stage {stage.stage_id}"
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
            parameter_dispatch_policy=parameter_dispatch_policy,
            task_output_contracts=MappingProxyType(dict(task_output_contracts)),
            model_roles=MappingProxyType(dict(model_roles)),
            denominator_policy=denominator_policy,
            release_requirements=release_requirements,
            operational_amendment=operational_amendment,
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
                                                (
                                                    seed_value
                                                    if dict(
                                                        profile.decoding_fields
                                                    )["seed"].dispatch_mode
                                                    == "explicit_supported"
                                                    else None
                                                )
                                                if profile.decoding_fields
                                                else (
                                                    None
                                                    if profile.seed_capability
                                                    == "unsupported"
                                                    else seed_value
                                                )
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

        if self.status != "frozen":
            raise PilotContractError(
                "paid provenance cannot be validated from a draft contract"
            )
        if self.contract_id == PILOT_CONTRACT_ID_V2_1:
            if self.release_requirements is None:  # pragma: no cover - parser
                raise PilotContractError("V2.1 lacks release requirements")
            _validate_v2_1_expected_ci_state(
                self.release_requirements.expected_ci,
                status=self.status,
                name="release expected_ci",
            )

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
        result = {
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
        if self.schema_version == PILOT_CONTRACT_SCHEMA_VERSION_V2:
            if (
                self.parameter_dispatch_policy is None
                or self.denominator_policy is None
                or self.release_requirements is None
            ):
                raise PilotContractError("incomplete typed V2 contract")
            result.update(
                {
                    "parameter_dispatch_policy": (
                        self.parameter_dispatch_policy.to_dict()
                    ),
                    "task_output_contracts": {
                        key: item.to_dict()
                        for key, item in self.task_output_contracts.items()
                    },
                    "model_roles": {
                        key: item.to_dict() for key, item in self.model_roles.items()
                    },
                    "denominator_policy": self.denominator_policy.to_dict(),
                    "release_requirements": self.release_requirements.to_dict(),
                }
            )
            if self.contract_id == PILOT_CONTRACT_ID_V2_1:
                if self.operational_amendment is None:
                    raise PilotContractError(
                        "V2.1 contract lacks its operational amendment"
                    )
                result["operational_amendment"] = _thaw_json(
                    self.operational_amendment
                )
            elif self.operational_amendment is not None:
                raise PilotContractError(
                    "original V2 contract cannot carry an operational amendment"
                )
        return result


def _assert_v2_1_base_equivalence(
    base: Mapping[str, Any],
    expanded: Mapping[str, Any],
    *,
    overlay_status: str,
) -> None:
    """Fail closed if the amendment changes any scientific design field."""

    for field in _V2_1_SCIENCE_DESIGN_FIELDS:
        if _json_copy(expanded[field]) != _json_copy(base[field]):
            raise PilotContractError(
                f"V2.1 science-critical field {field!r} differs from V2"
            )

    base_denominator = _json_copy(base["denominator_policy"])
    expanded_denominator = _json_copy(expanded["denominator_policy"])
    base_denominator.pop("policy_id")
    expanded_denominator.pop("policy_id")
    if expanded_denominator != base_denominator:
        raise PilotContractError(
            "V2.1 denominator differs beyond its policy identifier"
        )

    base_implementation = _json_copy(base["implementation"])
    expanded_implementation = _json_copy(expanded["implementation"])
    base_implementation.pop("required_git_tag")
    expanded_implementation.pop("required_git_tag")
    if expanded_implementation != base_implementation:
        raise PilotContractError(
            "V2.1 implementation differs beyond its release tag"
        )

    base_release = _json_copy(base["release_requirements"])
    expanded_release = _json_copy(expanded["release_requirements"])
    for release in (base_release, expanded_release):
        release.pop("tag")
        release.pop("expected_ci")
    if expanded_release != base_release:
        raise PilotContractError(
            "V2.1 release requirements differ beyond tag/CI placeholders"
        )

    expected_budgets = _json_copy(base["budgets"])
    expected_budgets["total_usd"] = 25.0
    expected_budgets["automatic_reserve_usd"] = 1.0
    expected_budgets["stage_usd_caps"] = {
        "capability": 3.0701145,
        "calibration": 3.0,
        "core": 13.0,
        "cross_model": 4.9298855,
        "manual_reserve": 1.0,
    }
    if _json_copy(expanded["budgets"]) != expected_budgets:
        raise PilotContractError("V2.1 budget reallocation drifted")

    expected_ci = expanded["release_requirements"]["expected_ci"]
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=overlay_status,
        name="V2.1 expanded release expected_ci",
    )
    if (
        expanded["schema_version"] != base["schema_version"]
        or expanded["status"] != overlay_status
        or expanded["contract_id"] != PILOT_CONTRACT_ID_V2_1
        or expanded["implementation"]["required_git_tag"]
        != PILOT_CONTRACT_TAG_V2_1
        or expanded["release_requirements"]["tag"]
        != PILOT_CONTRACT_TAG_V2_1
        or expanded["denominator_policy"]["policy_id"]
        != "finevo-pilot-v2.1-itt"
        or set(expected_ci) != _V2_1_EXPECTED_CI_FIELDS
    ):
        raise PilotContractError("V2.1 allowed identifier/CI amendment drifted")


def _expand_v2_1_overlay(
    value: Mapping[str, Any],
    *,
    source: Path,
) -> Mapping[str, Any]:
    """Expand the compact V2.1 operational amendment over the frozen V2."""

    _strict_keys(
        value,
        required={
            "schema_version",
            "contract_id",
            "status",
            "base_contract",
            "changes",
            "operational_amendment",
            "integrity",
        },
        name="V2.1 amendment overlay",
    )
    if (
        value["schema_version"]
        != PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_1
        or value["contract_id"] != PILOT_CONTRACT_ID_V2_1
        or value["status"] not in {"draft", "frozen"}
    ):
        raise PilotContractError("V2.1 amendment overlay identity drifted")

    integrity = _mapping(value["integrity"], "V2.1 overlay integrity")
    _strict_keys(
        integrity,
        required={"canonicalization", "declared_sha256"},
        name="V2.1 overlay integrity",
    )
    if integrity["canonicalization"] != PILOT_CONTRACT_CANONICALIZATION:
        raise PilotContractError("unsupported V2.1 overlay canonicalization")
    declared = _sha256(
        integrity["declared_sha256"],
        "V2.1 overlay declared_sha256",
    )
    actual = canonical_contract_sha256(value)
    if declared != actual:
        raise PilotContractError(
            f"V2.1 overlay hash mismatch: declared {declared}, actual {actual}"
        )

    base_binding = _mapping(value["base_contract"], "V2.1 base_contract")
    _strict_keys(
        base_binding,
        required={
            "path",
            "schema_version",
            "contract_id",
            "canonical_sha256",
        },
        name="V2.1 base_contract",
    )
    if _json_copy(base_binding) != {
        "path": "pilot_v2.yaml",
        "schema_version": PILOT_CONTRACT_SCHEMA_VERSION_V2,
        "contract_id": PILOT_CONTRACT_ID_V2,
        "canonical_sha256": PILOT_CONTRACT_V2_CANONICAL_SHA256,
    }:
        raise PilotContractError("V2.1 base contract binding drifted")
    base_path = source.parent / str(base_binding["path"])
    if (
        base_path.name != "pilot_v2.yaml"
        or base_path.resolve().parent != source.parent.resolve()
    ):
        raise PilotContractError(
            "V2.1 base contract must be the sibling pilot_v2.yaml"
        )
    base_contract = load_pilot_contract(base_path)
    if (
        base_contract.schema_version != PILOT_CONTRACT_SCHEMA_VERSION_V2
        or base_contract.contract_id != PILOT_CONTRACT_ID_V2
        or base_contract.canonical_hash != PILOT_CONTRACT_V2_CANONICAL_SHA256
    ):
        raise PilotContractError("V2.1 resolved base contract identity drifted")

    changes = _mapping(value["changes"], "V2.1 changes")
    _strict_keys(
        changes,
        required={
            "implementation",
            "release_requirements",
            "budgets",
            "denominator_policy",
        },
        name="V2.1 changes",
    )
    implementation_change = _mapping(
        changes["implementation"],
        "V2.1 changes.implementation",
    )
    _strict_keys(
        implementation_change,
        required={"required_git_tag"},
        name="V2.1 changes.implementation",
    )
    if implementation_change["required_git_tag"] != PILOT_CONTRACT_TAG_V2_1:
        raise PilotContractError("V2.1 implementation tag drifted")

    release_change = _mapping(
        changes["release_requirements"],
        "V2.1 changes.release_requirements",
    )
    _strict_keys(
        release_change,
        required={"tag", "expected_ci"},
        name="V2.1 changes.release_requirements",
    )
    expected_ci = _mapping(
        release_change["expected_ci"],
        "V2.1 changes.release_requirements.expected_ci",
    )
    _strict_keys(
        expected_ci,
        required=_V2_1_EXPECTED_CI_FIELDS,
        name="V2.1 changes.release_requirements.expected_ci",
    )
    if release_change["tag"] != PILOT_CONTRACT_TAG_V2_1:
        raise PilotContractError(
            "V2.1 release tag drifted"
        )
    _validate_v2_1_expected_ci_state(
        expected_ci,
        status=str(value["status"]),
        name="V2.1 changes.release_requirements.expected_ci",
    )

    budget_change = _mapping(changes["budgets"], "V2.1 changes.budgets")
    _strict_keys(
        budget_change,
        required={
            "total_usd",
            "automatic_reserve_usd",
            "stage_usd_caps",
        },
        name="V2.1 changes.budgets",
    )
    expected_budget_change = {
        "total_usd": 25.0,
        "automatic_reserve_usd": 1.0,
        "stage_usd_caps": {
            "capability": 3.0701145,
            "calibration": 3.0,
            "core": 13.0,
            "cross_model": 4.9298855,
            "manual_reserve": 1.0,
        },
    }
    if _json_copy(budget_change) != expected_budget_change:
        raise PilotContractError("V2.1 overlay budget caps drifted")

    denominator_change = _mapping(
        changes["denominator_policy"],
        "V2.1 changes.denominator_policy",
    )
    _strict_keys(
        denominator_change,
        required={"policy_id"},
        name="V2.1 changes.denominator_policy",
    )
    if denominator_change["policy_id"] != "finevo-pilot-v2.1-itt":
        raise PilotContractError("V2.1 denominator policy identifier drifted")

    amendment = _validate_v2_1_operational_amendment(
        value["operational_amendment"]
    )
    expanded = base_contract.to_dict()
    expanded["status"] = value["status"]
    expanded["contract_id"] = PILOT_CONTRACT_ID_V2_1
    expanded["implementation"]["required_git_tag"] = PILOT_CONTRACT_TAG_V2_1
    expanded["release_requirements"]["tag"] = PILOT_CONTRACT_TAG_V2_1
    expanded["release_requirements"]["expected_ci"] = _json_copy(expected_ci)
    expanded["budgets"]["total_usd"] = 25.0
    expanded["budgets"]["automatic_reserve_usd"] = 1.0
    expanded["budgets"]["stage_usd_caps"] = _json_copy(
        budget_change["stage_usd_caps"]
    )
    expanded["denominator_policy"]["policy_id"] = "finevo-pilot-v2.1-itt"
    expanded["operational_amendment"] = _thaw_json(amendment)
    expanded["integrity"]["declared_sha256"] = "0" * 64
    expanded["integrity"]["declared_sha256"] = canonical_contract_sha256(
        expanded
    )
    _assert_v2_1_base_equivalence(
        base_contract.to_dict(),
        expanded,
        overlay_status=str(value["status"]),
    )
    return expanded


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
    document = _mapping(value, "pilot contract")
    if (
        document.get("schema_version")
        == PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_1
    ):
        document = _expand_v2_1_overlay(document, source=source)
    return PilotContract.from_dict(document)


__all__ = [
    "PILOT_CONTRACT_CANONICALIZATION",
    "PILOT_CONTRACT_SCHEMA_VERSION",
    "PILOT_CONTRACT_SCHEMA_VERSION_V1",
    "PILOT_CONTRACT_SCHEMA_VERSION_V2",
    "PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_1",
    "PILOT_CONTRACT_ID_V2",
    "PILOT_CONTRACT_ID_V2_1",
    "PILOT_CONTRACT_TAG_V2",
    "PILOT_CONTRACT_TAG_V2_1",
    "PILOT_CONTRACT_V2_CANONICAL_SHA256",
    "PILOT_CONTRACT_V2_SCIENCE_DESIGN_SHA256",
    "DecodingFieldDispatch",
    "DenominatorPolicy",
    "ModelRolePolicy",
    "ParameterDispatchPolicy",
    "PilotContract",
    "PilotContractError",
    "PilotRunSpec",
    "PilotStage",
    "PilotStageCell",
    "PriceSnapshot",
    "ProviderRequestProfile",
    "ReasoningProfile",
    "ReleaseRequirements",
    "TaskOutputContract",
    "canonical_contract_sha256",
    "canonical_sha256",
    "load_pilot_contract",
    "science_design_sha256",
]
