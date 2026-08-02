"""Fail-closed contracts for EGRM fixtures and scientific runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


SCHEMA_VERSION = "egrm-experiment-contract-v1"
STATUSES = frozenset({"implementation_fixture", "design_draft_not_run"})
ARM_IMPLEMENTATION_STATUSES = frozenset({"implemented", "planned", "manifest_only"})
EXPECTED_KEYS = {
    "schema_version",
    "study_id",
    "status",
    "scientific_evidence",
    "source",
    "randomization",
    "units",
    "arms",
    "metrics",
    "denominator",
    "stop_rules",
    "claim_boundary",
    "fixture",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _string_list(value: Any, name: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    if nonempty and not value:
        raise ValueError(f"{name} must not be empty")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    return value


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("contract must be a JSON object")
    actual = set(value)
    if actual != EXPECTED_KEYS:
        raise ValueError(
            "contract keys are not exact: "
            f"missing={sorted(EXPECTED_KEYS - actual)}, "
            f"extra={sorted(actual - EXPECTED_KEYS)}"
        )
    result = json.loads(canonical_json(value))
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if result["status"] not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}")
    if not isinstance(result["scientific_evidence"], bool):
        raise ValueError("scientific_evidence must be boolean")
    if result["scientific_evidence"]:
        raise ValueError("a not-run EGRM contract cannot be scientific evidence")
    if not isinstance(result["study_id"], str) or not result["study_id"].strip():
        raise ValueError("study_id must be a non-empty string")

    source = result["source"]
    if not isinstance(source, dict) or set(source) != {
        "extraction_repository",
        "extraction_commit",
        "provenance_manifest",
    }:
        raise ValueError("source keys are not exact")
    if not re.fullmatch(r"[0-9a-f]{40}", str(source["extraction_commit"])):
        raise ValueError("source.extraction_commit must be a full lowercase Git SHA")
    if any(not isinstance(source[key], str) or not source[key] for key in source):
        raise ValueError("source values must be non-empty strings")

    randomization = result["randomization"]
    if not isinstance(randomization, dict) or set(randomization) != {"seeds", "paired"}:
        raise ValueError("randomization keys are not exact")
    seeds = randomization["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("randomization.seeds must be a non-empty integer list")
    if len(set(seeds)) != len(seeds):
        raise ValueError("randomization.seeds must be fresh and unique")
    if not isinstance(randomization["paired"], bool):
        raise ValueError("randomization.paired must be boolean")

    units = result["units"]
    if not isinstance(units, dict) or set(units) != {"primary", "rule", "intervention"}:
        raise ValueError("units keys are not exact")
    if any(not isinstance(item, str) or not item for item in units.values()):
        raise ValueError("units values must be non-empty strings")

    arms = result["arms"]
    if (
        not isinstance(arms, list)
        or len(arms) < 2
        or any(
            not isinstance(arm, dict)
            or set(arm) != {"name", "policy", "implementation_status"}
            for arm in arms
        )
    ):
        raise ValueError(
            "arms must contain at least two exact name/policy/implementation objects"
        )
    arm_names = [arm["name"] for arm in arms]
    _string_list(arm_names, "arm names")
    if any(not isinstance(arm["policy"], str) or not arm["policy"] for arm in arms):
        raise ValueError("arm policies must be non-empty strings")
    if any(
        arm["implementation_status"] not in ARM_IMPLEMENTATION_STATUSES for arm in arms
    ):
        raise ValueError("arm implementation_status is invalid")

    metrics = result["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != {"primary", "secondary"}:
        raise ValueError("metrics keys are not exact")
    _string_list(metrics["primary"], "metrics.primary")
    _string_list(metrics["secondary"], "metrics.secondary", nonempty=False)

    denominator = result["denominator"]
    if not isinstance(denominator, dict) or set(denominator) != {
        "registered_cells",
        "include_parse_failures",
        "include_provider_failures",
        "failed_seed_replacement",
    }:
        raise ValueError("denominator keys are not exact")
    if denominator["include_parse_failures"] is not True:
        raise ValueError("parse failures must remain in the denominator")
    if denominator["include_provider_failures"] is not True:
        raise ValueError("provider failures must remain in the denominator")
    if denominator["failed_seed_replacement"] is not False:
        raise ValueError("failed seeds must not be replaced")
    declared_total = denominator["registered_cells"]
    if (
        isinstance(declared_total, bool)
        or not isinstance(declared_total, int)
        or declared_total <= 0
    ):
        raise ValueError("registered_cells must be a positive integer")

    _string_list(result["stop_rules"], "stop_rules")
    boundary = result["claim_boundary"]
    if not isinstance(boundary, dict) or set(boundary) != {"allowed", "forbidden"}:
        raise ValueError("claim_boundary keys are not exact")
    _string_list(boundary["allowed"], "claim_boundary.allowed")
    _string_list(boundary["forbidden"], "claim_boundary.forbidden")
    if not isinstance(result["fixture"], dict):
        raise ValueError("fixture must be an object")
    if result["status"] == "implementation_fixture":
        if len(seeds) != 1:
            raise ValueError("the deterministic fixture requires exactly one seed")
        if arm_names != ["egrm", "unverified-immediate"]:
            raise ValueError("implementation fixture arms are frozen")
        if any(arm["implementation_status"] != "implemented" for arm in arms):
            raise ValueError("implementation fixture arms must be implemented")
        if declared_total != len(arms):
            raise ValueError("implementation fixture denominator must equal its arms")
        if metrics["primary"] != [
            "rule_admission_precision",
            "unsupported_activation_rate",
            "harmful_compliance_events",
            "retirement_latency_steps",
        ]:
            raise ValueError("implementation fixture primary metrics are frozen")
        if metrics["secondary"] != [
            "provenance_coverage",
            "event_type_counts",
        ]:
            raise ValueError("implementation fixture secondary metrics are frozen")
    if result["status"] == "design_draft_not_run":
        design = result["fixture"]
        required = {
            "design_type",
            "hosted_calls_enabled",
            "models",
            "stages",
            "estimated_hosted_call_upper_bound",
            "budget_authorization",
            "release_binding",
        }
        if set(design) != required:
            raise ValueError("scientific design keys are not exact")
        if design["design_type"] != "fresh_egrm_scientific_design":
            raise ValueError("scientific design_type is invalid")
        if design["hosted_calls_enabled"] is not False:
            raise ValueError("hosted calls must remain disabled in this release")
        if not isinstance(design["models"], dict) or set(design["models"]) != {
            "primary",
            "diagnostic",
        }:
            raise ValueError("scientific models keys are not exact")
        if any(
            not isinstance(model, str) or not model
            for model in design["models"].values()
        ):
            raise ValueError("scientific model names must be non-empty strings")
        stages = design["stages"]
        if (
            not isinstance(stages, list)
            or not stages
            or any(
                not isinstance(stage, dict)
                or set(stage)
                != {
                    "name",
                    "registered_cells",
                    "description",
                    "unit_axis",
                    "unit_values",
                    "arms",
                    "model",
                }
                for stage in stages
            )
        ):
            raise ValueError("scientific stages have an invalid shape")
        stage_names = [stage["name"] for stage in stages]
        _string_list(stage_names, "scientific stage names")
        for stage in stages:
            if (
                isinstance(stage["registered_cells"], bool)
                or not isinstance(stage["registered_cells"], int)
                or stage["registered_cells"] <= 0
            ):
                raise ValueError("stage registered_cells must be positive integers")
            if not isinstance(stage["description"], str) or not stage["description"]:
                raise ValueError("stage description must be a non-empty string")
            if stage["unit_axis"] not in {"scenario_id", "seed"}:
                raise ValueError("stage unit_axis must be scenario_id or seed")
            unit_values = stage["unit_values"]
            if not isinstance(unit_values, list) or not unit_values:
                raise ValueError("stage unit_values must be a non-empty list")
            if len({canonical_json(item) for item in unit_values}) != len(unit_values):
                raise ValueError("stage unit_values must be unique")
            if stage["unit_axis"] == "scenario_id":
                _string_list(unit_values, "stage scenario ids")
            elif any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed not in seeds
                for seed in unit_values
            ):
                raise ValueError("stage seeds must be drawn from randomization.seeds")
            stage_arms = _string_list(stage["arms"], "stage arms")
            if any(arm not in arm_names for arm in stage_arms):
                raise ValueError("stage references an unknown arm")
            expected_cells = len(unit_values) * len(stage_arms)
            if stage["registered_cells"] != expected_cells:
                raise ValueError(
                    "stage registered_cells must equal unit-by-arm product"
                )
            if not isinstance(stage["model"], str) or not stage["model"]:
                raise ValueError("stage model must be a non-empty string")
        if declared_total != sum(stage["registered_cells"] for stage in stages):
            raise ValueError("registered_cells must equal the scientific stage sum")
        upper_bound = design["estimated_hosted_call_upper_bound"]
        if (
            isinstance(upper_bound, bool)
            or not isinstance(upper_bound, int)
            or upper_bound <= 0
        ):
            raise ValueError("estimated hosted call upper bound must be positive")
        if design["budget_authorization"] != "not_authorized_by_this_release":
            raise ValueError("this release must not authorize a hosted budget")
        release = design["release_binding"]
        if not isinstance(release, dict) or set(release) != {
            "status",
            "required_annotated_tag",
            "required_git_commit",
        }:
            raise ValueError("release_binding keys are not exact")
        if (
            release["status"] != "unbound_draft"
            or release["required_git_commit"] is not None
        ):
            raise ValueError(
                "the design draft must remain unbound and non-dispatchable"
            )
        if (
            not isinstance(release["required_annotated_tag"], str)
            or not release["required_annotated_tag"]
        ):
            raise ValueError("required_annotated_tag must be non-empty")
    return result


def load_contract(path: str | Path) -> tuple[dict[str, Any], str]:
    contract_path = Path(path)
    with contract_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    validated = validate_contract(value)
    return validated, content_hash(validated)
