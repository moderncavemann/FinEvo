#!/usr/bin/env python3
"""Extract and verify one deterministic, log-grounded FinEvo trace.

The artifact is an observational mechanism trace, not a single-agent causal
estimate.  It connects a decision-time state, the exact retrieved memories and
semantic rule, the parsed action, and the subsequently observed joint-system
transition.  Missing fields remain null and are recorded explicitly.

The normalized ``runs/`` files are checked against their native ``data/``
counterparts.  A raw memory checkpoint independently verifies decision-time
macro fields, the exact retrieval-query sentiment, and pre/post timing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import pickle
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = Path(__file__).resolve()

NUMBER_OR_NULL = {"type": ["number", "null"]}
STRING_OR_NULL = {"type": ["string", "null"]}

SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "FinEvo log-grounded observational closed-loop trace",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "evidence_scope",
        "current_method_scientific_evidence",
        "method_implementation",
        "trace_id",
        "trace_scope",
        "selection_rule",
        "run_metadata",
        "case_index",
        "shared_state",
        "private_state",
        "text_event",
        "retrieved_episodic_memory",
        "retrieved_semantic_rules",
        "parsed_action",
        "next_window_outcome",
        "provenance",
        "validation",
    ],
    "$defs": {
        "file_record": {
            "type": "object",
            "additionalProperties": False,
            "required": ["path", "sha256", "size_bytes", "role"],
            "properties": {
                "path": {"type": "string"},
                "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "size_bytes": {"type": "integer", "minimum": 0},
                "role": {"type": "string"},
            },
        },
        "macro_state": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "price_level",
                "interest_rate_pct",
                "unemployment_pct",
                "inflation_pct",
                "sentiment",
            ],
            "properties": {
                "price_level": NUMBER_OR_NULL,
                "interest_rate_pct": NUMBER_OR_NULL,
                "unemployment_pct": NUMBER_OR_NULL,
                "inflation_pct": NUMBER_OR_NULL,
                "sentiment": NUMBER_OR_NULL,
            },
        },
    },
    "properties": {
        "evidence_scope": {"const": "historical_pre_p0_v1"},
        "current_method_scientific_evidence": {"const": False},
        "method_implementation": {"const": "legacy_deterministic_template_memory"},
        "trace_id": {"type": "string", "minLength": 1},
        "trace_scope": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "classification",
                "supports_mechanism_trace",
                "supports_single_agent_causal_attribution",
                "interpretation",
            ],
            "properties": {
                "classification": {"const": "observational_closed_loop_trace"},
                "supports_mechanism_trace": {"const": True},
                "supports_single_agent_causal_attribution": {"const": False},
                "interpretation": {"type": "string"},
            },
        },
        "selection_rule": {
            "type": "object",
            "additionalProperties": False,
            "required": ["primary", "fallback_order", "agent_tie_break"],
            "properties": {
                "primary": {"type": "string"},
                "fallback_order": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string"},
                },
                "agent_tie_break": {"type": "string"},
            },
        },
        "run_metadata": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "exp_id",
                "model",
                "setting",
                "variant",
                "seed",
                "num_agents",
                "num_months",
                "source_run_dir",
                "raw_source_dir",
                "simulation_code_commit",
                "extractor_script_sha256",
                "legacy_export_note",
            ],
            "properties": {
                "exp_id": {"type": "string"},
                "model": {"type": "string"},
                "setting": {"type": "string"},
                "variant": {"type": "string"},
                "seed": {"type": "integer"},
                "num_agents": {"type": "integer", "minimum": 1},
                "num_months": {"type": "integer", "minimum": 1},
                "source_run_dir": {"type": "string"},
                "raw_source_dir": {"type": "string"},
                "simulation_code_commit": STRING_OR_NULL,
                "extractor_script_sha256": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{64}$",
                },
                "legacy_export_note": STRING_OR_NULL,
            },
        },
        "case_index": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "month",
                "agent_id",
                "selection_rule_used",
                "selected_rule_ids",
                "retrieved_episode_ids",
            ],
            "properties": {
                "month": {"type": "integer", "minimum": 0},
                "agent_id": {"type": "integer", "minimum": 0},
                "selection_rule_used": {"type": "string"},
                "selected_rule_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "retrieved_episode_ids": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string"},
                },
            },
        },
        "shared_state": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "month",
                "retrieval_query_state",
                "decision_prompt_sentiment_cue",
                "selection_signal",
                "exported_trajectory_snapshot",
                "source_alignment_note",
            ],
            "properties": {
                "month": {"type": "integer"},
                "retrieval_query_state": {"$ref": "#/$defs/macro_state"},
                "decision_prompt_sentiment_cue": NUMBER_OR_NULL,
                "selection_signal": {
                    "type": "object",
                    "required": ["name", "value", "rank_within_run", "crash_flag", "label"],
                },
                "exported_trajectory_snapshot": {"type": "object"},
                "source_alignment_note": {"type": "string"},
            },
        },
        "private_state": {
            "type": "object",
            "additionalProperties": False,
            "required": ["agent_id", "wealth", "income", "employed"],
            "properties": {
                "agent_id": {"type": "integer"},
                "wealth": NUMBER_OR_NULL,
                "income": NUMBER_OR_NULL,
                "employed": {"type": ["boolean", "null"]},
            },
        },
        "text_event": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "value",
                "event_id",
                "event_type",
                "sentiment_label",
                "v_t",
                "missing_reason",
            ],
            "properties": {
                "value": STRING_OR_NULL,
                "event_id": STRING_OR_NULL,
                "event_type": STRING_OR_NULL,
                "sentiment_label": STRING_OR_NULL,
                "v_t": NUMBER_OR_NULL,
                "missing_reason": STRING_OR_NULL,
            },
        },
        "retrieved_episodic_memory": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "retrieved_episode_ids",
                "retrieval_scores",
                "score_components",
                "episode_summaries",
                "memory_block_hash",
                "temporal_alignment_warning",
            ],
            "properties": {
                "retrieved_episode_ids": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string"},
                },
                "retrieval_scores": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "number"},
                },
                "score_components": {"type": "array", "minItems": 1},
                "episode_summaries": {"type": "array", "minItems": 1},
                "memory_block_hash": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{64}$",
                },
                "temporal_alignment_warning": {"type": "string"},
            },
        },
        "retrieved_semantic_rules": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "rule_id",
                    "condition",
                    "action_guidance",
                    "confidence",
                    "source_episode_ids",
                    "condition_evaluation",
                    "action_guidance_evaluation",
                    "validity_note",
                    "validity_note_interpretation",
                ],
            },
        },
        "parsed_action": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "work_propensity",
                "labor_hours_if_realized",
                "consumption_fraction",
                "rationale_excerpt",
                "parser_accepted",
                "repair_attempts",
                "strict_json_without_repair",
                "clipped",
                "used_memory_ids",
                "used_memory_ids_match_retrieval",
                "prompt_hash",
                "raw_action_hash",
            ],
            "properties": {
                "work_propensity": NUMBER_OR_NULL,
                "labor_hours_if_realized": NUMBER_OR_NULL,
                "consumption_fraction": NUMBER_OR_NULL,
                "rationale_excerpt": STRING_OR_NULL,
                "parser_accepted": {"type": ["boolean", "null"]},
                "repair_attempts": {"type": ["integer", "null"]},
                "strict_json_without_repair": {"type": ["boolean", "null"]},
                "clipped": {"type": ["boolean", "null"]},
                "used_memory_ids": {"type": "array", "items": {"type": "string"}},
                "used_memory_ids_match_retrieval": {"type": "boolean"},
                "prompt_hash": STRING_OR_NULL,
                "raw_action_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
        },
        "next_window_outcome": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "timing",
                "concurrent_population_action",
                "immediate_transition",
                "six_month_followup",
                "causal_interpretation",
            ],
            "properties": {
                "timing": {"type": "object"},
                "concurrent_population_action": {"type": "object"},
                "immediate_transition": {"type": "object"},
                "six_month_followup": {"type": "object"},
                "causal_interpretation": {
                    "type": "object",
                    "required": ["classification", "supports_single_agent_causal_attribution", "reason"],
                },
            },
        },
        "provenance": {
            "type": "object",
            "additionalProperties": False,
            "required": ["files", "pointers", "normalization_lineage"],
            "properties": {
                "files": {
                    "type": "object",
                    "additionalProperties": {"$ref": "#/$defs/file_record"},
                },
                "pointers": {"type": "object"},
                "normalization_lineage": {"type": "object"},
            },
        },
        "validation": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "schema_valid",
                "schema_errors",
                "provenance_paths_exist",
                "source_file_hashes_recorded",
                "native_logs_match_raw_source",
                "source_values_verified",
                "retrieval_score_components_sum",
                "retrieved_episodes_found_in_checkpoint",
                "selected_rules_found",
                "semantic_precondition_matches_query_state",
                "action_matches_rule_guidance",
                "action_memory_ids_match_retrieval",
                "parser_accepted",
                "strict_json_without_repair",
                "immediate_delta_matches_next_episode_record",
                "has_observed_transition",
                "supports_causal_attribution",
                "publishable_as_observational_trace",
                "missing_fields",
                "excluded_candidates",
                "warnings",
            ],
            "properties": {
                "schema_valid": {"type": "boolean"},
                "schema_errors": {"type": "array", "items": {"type": "string"}},
                "provenance_paths_exist": {"type": "boolean"},
                "source_file_hashes_recorded": {"type": "boolean"},
                "native_logs_match_raw_source": {"type": "boolean"},
                "source_values_verified": {"type": "boolean"},
                "retrieval_score_components_sum": {"type": "boolean"},
                "retrieved_episodes_found_in_checkpoint": {"type": "boolean"},
                "selected_rules_found": {"type": "boolean"},
                "semantic_precondition_matches_query_state": {"type": ["boolean", "null"]},
                "action_matches_rule_guidance": {"type": ["boolean", "null"]},
                "action_memory_ids_match_retrieval": {"type": "boolean"},
                "parser_accepted": {"type": "boolean"},
                "strict_json_without_repair": {"type": "boolean"},
                "immediate_delta_matches_next_episode_record": {"type": "boolean"},
                "has_observed_transition": {"type": "boolean"},
                "supports_causal_attribution": {"const": False},
                "publishable_as_observational_trace": {"type": "boolean"},
                "missing_fields": {"type": "array"},
                "excluded_candidates": {"type": "array"},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


@dataclass(frozen=True)
class JsonLine:
    line_index: int
    data: dict[str, Any]


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def rounded(value: Any, digits: int = 6, multiplier: float = 1.0) -> float | None:
    number = as_float(value)
    return None if number is None else round(number * multiplier, digits)


def numeric(row: dict[str, Any], key: str, multiplier: float = 1.0) -> float | None:
    return rounded(row.get(key), multiplier=multiplier)


def difference(end: Any, start: Any) -> float | None:
    end_number = as_float(end)
    start_number = as_float(start)
    if end_number is None or start_number is None:
        return None
    return round(end_number - start_number, 6)


def mean(values: list[Any]) -> float | None:
    numbers = [number for value in values if (number := as_float(value)) is not None]
    return None if not numbers else round(sum(numbers) / len(numbers), 6)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=True))


def short_text(value: Any, limit: int = 220) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).replace("\n", " ").split())
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row_index, row in enumerate(rows):
        row["_row_index"] = row_index
    return rows


def read_jsonl(path: Path) -> list[JsonLine]:
    rows: list[JsonLine] = []
    with path.open() as handle:
        for line_index, line in enumerate(handle):
            if line.strip():
                rows.append(JsonLine(line_index, json.loads(line)))
    return rows


def file_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def file_record(path: Path, role: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        "path": rel(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "role": role,
    }


def row_by_month(rows: list[dict[str, Any]], month: int) -> dict[str, Any] | None:
    for row in rows:
        if int(float(row["month"])) == month:
            return row
    return None


def rows_by_month(rows: list[dict[str, Any]], month: int) -> list[dict[str, Any]]:
    return [row for row in rows if int(float(row["month"])) == month]


def action_key(obj: dict[str, Any]) -> tuple[int, int]:
    return int(obj["month"]), int(obj["agent_id"])


def find_agent_state(
    rows: list[dict[str, Any]], month: int, agent_id: int
) -> dict[str, Any] | None:
    for row in rows:
        if int(float(row["month"])) == month and int(float(row["agent_id"])) == agent_id:
            return row
    return None


def candidate_dirs() -> list[Path]:
    patterns = [
        "runs/E1/GPT-5.2/finevo/default/seed_*",
        "runs/E2/GPT-5.2/finevo/default/seed_*",
        "runs/E5/GPT-5.2/finevo/default-prompt/seed_*",
        "runs/E1/GPT-4o/finevo/default/seed_13",
        "runs/E1/Gemini-3-Flash/finevo/default/seed_13",
        "runs/E1/Qwen3-235B/finevo/default/seed_13",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(ROOT.glob(pattern)))
    return [path for path in paths if path.is_dir()]


def source_dir_for_run(run_dir: Path) -> Path:
    manifest_path = run_dir / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_dir = (ROOT / str(manifest["source_dir"])).resolve()
    source_dir.relative_to(ROOT.resolve())
    return source_dir


def checkpoint_path(source_dir: Path, required_month: int) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in source_dir.glob("memory_*.pkl"):
        match = re.fullmatch(r"memory_([0-9]+)\.pkl", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    later = sorted((step, path) for step, path in candidates if step >= required_month)
    if later:
        return later[0][1]
    final_path = source_dir / "memory_final.pkl"
    return final_path if final_path.exists() else None


def check_candidate(run_dir: Path) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    required = [
        "config.yaml",
        "export_manifest.json",
        "trajectory.csv",
        "agent_state.csv",
        "actions.jsonl",
        "memory_retrieval.jsonl",
        "semantic_rules.jsonl",
        "metrics_summary.csv",
        "api_errors.jsonl",
    ]
    for name in required:
        path = run_dir / name
        if not path.exists():
            reasons.append(f"missing {name}")
        elif name != "api_errors.jsonl" and path.stat().st_size == 0:
            reasons.append(f"empty {name}")
    if reasons:
        return False, reasons
    try:
        source_dir = source_dir_for_run(run_dir)
    except Exception as exc:
        reasons.append(f"invalid export lineage: {exc}")
        return False, reasons
    for name in ["actions.jsonl", "memory_retrieval.jsonl", "semantic_rules.jsonl"]:
        source_path = source_dir / name
        if not source_path.exists():
            reasons.append(f"raw source missing {name}")
        elif sha256_file(source_path) != sha256_file(run_dir / name):
            reasons.append(f"normalized {name} differs from raw source")
    if checkpoint_path(source_dir, 6) is None:
        reasons.append("raw source has no memory checkpoint")
    try:
        first = read_jsonl(run_dir / "actions.jsonl")[0].data
        for key in ["valid_json", "repair_attempts", "clipped", "used_memory_ids"]:
            if key not in first:
                reasons.append(f"actions.jsonl lacks parser field {key}")
    except Exception as exc:
        reasons.append(f"actions.jsonl unreadable: {exc}")
    return not reasons, reasons


def action_change(current: dict[str, Any], previous: dict[str, Any] | None) -> float:
    if previous is None:
        return -1.0
    current_action = current.get("parsed_action") or {}
    previous_action = previous.get("parsed_action") or {}
    current_labor = as_float(current_action.get("labor_hours"))
    previous_labor = as_float(previous_action.get("labor_hours"))
    current_consumption = as_float(current_action.get("consumption_fraction"))
    previous_consumption = as_float(previous_action.get("consumption_fraction"))
    if None in [current_labor, previous_labor, current_consumption, previous_consumption]:
        return -1.0
    return abs(current_labor - previous_labor) / 168.0 + abs(
        current_consumption - previous_consumption
    )


def month_candidates(trajectory: list[dict[str, Any]]) -> list[tuple[str, int]]:
    sentiment_rows = [row for row in trajectory if as_float(row.get("global_sentiment")) is not None]
    if not sentiment_rows:
        return []
    lowest = min(
        sentiment_rows,
        key=lambda row: (float(row["global_sentiment"]), int(float(row["month"]))),
    )
    candidates = [("lowest_logged_sentiment", int(float(lowest["month"])))]
    jumps: list[tuple[float, int]] = []
    drawdowns: list[tuple[float, int]] = []
    for previous, current in zip(trajectory, trajectory[1:]):
        unemployment_delta = difference(
            current.get("unemployment_pct"), previous.get("unemployment_pct")
        )
        wealth_drawdown = difference(previous.get("avg_wealth"), current.get("avg_wealth"))
        month = int(float(current["month"]))
        if unemployment_delta is not None:
            jumps.append((unemployment_delta, month))
        if wealth_drawdown is not None:
            drawdowns.append((wealth_drawdown, month))
    if jumps:
        candidates.append(("largest_exported_unemployment_jump", max(jumps)[1]))
    if drawdowns:
        candidates.append(("largest_average_wealth_drawdown", max(drawdowns)[1]))
    return candidates


def find_rule(
    rules: list[JsonLine], agent_id: int, rule_ids: list[str], month: int
) -> list[JsonLine]:
    matches = [
        line
        for line in rules
        if int(line.data.get("agent_id", -1)) == agent_id
        and line.data.get("rule_id") in rule_ids
        and int(line.data.get("month", 10**9)) <= month
    ]
    if not matches:
        return []
    latest_month = max(int(line.data["month"]) for line in matches)
    return [line for line in matches if int(line.data["month"]) == latest_month]


def pick_case(run_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    trajectory = read_csv_dicts(run_dir / "trajectory.csv")
    actions = read_jsonl(run_dir / "actions.jsonl")
    retrievals = read_jsonl(run_dir / "memory_retrieval.jsonl")
    rules = read_jsonl(run_dir / "semantic_rules.jsonl")
    actions_by_key = {action_key(line.data): line for line in actions}
    retrievals_by_key = {action_key(line.data): line for line in retrievals}
    reasons: list[str] = []

    for rule_name, month in month_candidates(trajectory):
        eligible: list[tuple[float, int, JsonLine, JsonLine, list[JsonLine]]] = []
        for (action_month, agent_id), action_line in actions_by_key.items():
            if action_month != month:
                continue
            retrieval_line = retrievals_by_key.get((month, agent_id))
            if retrieval_line is None:
                continue
            episode_ids = retrieval_line.data.get("retrieved_episode_ids") or []
            rule_ids = retrieval_line.data.get("selected_rule_ids") or []
            matched_rules = find_rule(rules, agent_id, rule_ids, month)
            if not episode_ids or not matched_rules:
                continue
            previous = actions_by_key.get((month - 1, agent_id))
            eligible.append(
                (
                    action_change(action_line.data, previous.data if previous else None),
                    agent_id,
                    action_line,
                    retrieval_line,
                    matched_rules,
                )
            )
        if eligible:
            eligible.sort(key=lambda item: (-item[0], item[1]))
            _, agent_id, action_line, retrieval_line, rule_lines = eligible[0]
            return {
                "selection_rule_used": rule_name,
                "month": month,
                "agent_id": agent_id,
                "action_line": action_line,
                "retrieval_line": retrieval_line,
                "rule_lines": rule_lines,
            }, reasons
        reasons.append(
            f"{rule_name} month {month} lacked an agent with both retrieved episodes and a matched semantic rule"
        )
    return None, reasons


def load_checkpoint_episodes(
    source_dir: Path, agent_id: int, required_month: int
) -> tuple[Path, dict[int, dict[str, Any]]]:
    path = checkpoint_path(source_dir, required_month)
    if path is None:
        raise FileNotFoundError(f"no memory checkpoint at or after month {required_month}")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    with path.open("rb") as handle:
        memory_systems = pickle.load(handle)  # trusted local experiment artifact
    memory = memory_systems.get(agent_id, memory_systems.get(str(agent_id)))
    if memory is None:
        raise KeyError(f"agent {agent_id} absent from {path}")
    episodes: dict[int, dict[str, Any]] = {}
    for episode in getattr(memory, "episodic_memories", []):
        data = episode.to_dict() if hasattr(episode, "to_dict") else dict(vars(episode))
        episodes[int(data["timestamp"])] = data
    return path, episodes


def macro_state(episode: dict[str, Any]) -> dict[str, Any]:
    state = episode.get("economic_state") or {}
    return {
        "price_level": rounded(state.get("price")),
        "interest_rate_pct": rounded(state.get("interest_rate"), multiplier=100.0),
        "unemployment_pct": rounded(state.get("unemployment_rate"), multiplier=100.0),
        "inflation_pct": rounded(state.get("inflation"), multiplier=100.0),
        "sentiment": rounded(state.get("sentiment")),
    }


def episode_number(episode_id: str) -> int:
    match = re.fullmatch(r"E([0-9]+)", str(episode_id))
    if not match:
        raise ValueError(f"invalid episode id: {episode_id}")
    return int(match.group(1))


def build_episode_summaries(
    episode_ids: list[str], episodes: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for episode_id in episode_ids[:3]:
        month = episode_number(episode_id)
        episode = episodes[month]
        previous = episodes.get(month - 1)
        next_episode = episodes.get(month + 1)
        personal = episode.get("personal_state") or {}
        stored_outcome = episode.get("outcome") or {}
        entering_delta = (
            difference(
                personal.get("wealth"),
                (previous.get("personal_state") or {}).get("wealth"),
            )
            if previous
            else None
        )
        stored_delta = rounded(stored_outcome.get("wealth_change"))
        summaries.append(
            {
                "episode_id": episode_id,
                "month": month,
                "economic_state": macro_state(episode),
                "personal_state": {
                    "wealth": rounded(personal.get("wealth")),
                    "income": rounded(personal.get("income")),
                    "employed": bool(personal.get("employed"))
                    if personal.get("employed") is not None
                    else None,
                },
                "stored_executed_decision": episode.get("decision") or {},
                "stored_outcome": {
                    "wealth_change": stored_delta,
                    "matches_change_entering_same_month": (
                        stored_delta is not None
                        and entering_delta is not None
                        and abs(stored_delta - entering_delta) < 1e-5
                    ),
                    "wealth_change_realized_after_this_decision": rounded(
                        (next_episode.get("outcome") or {}).get("wealth_change")
                    )
                    if next_episode
                    else None,
                },
                "reflection_excerpt": short_text(episode.get("reflection"), 180),
            }
        )
    return summaries


def evaluate_condition(condition: Any, query_state: dict[str, Any]) -> dict[str, Any]:
    text = str(condition or "")
    inflation_match = re.search(r"inflation.*?>\s*([0-9]+(?:\.[0-9]+)?)\s*%", text, re.I)
    if inflation_match:
        threshold = float(inflation_match.group(1))
        observed = as_float(query_state.get("inflation_pct"))
        return {
            "variable": "inflation_pct",
            "observed_value": rounded(observed),
            "operator": ">",
            "threshold": threshold,
            "matches": observed > threshold if observed is not None else None,
            "method": "exact numeric predicate parsed from the logged condition",
        }
    return {
        "variable": None,
        "observed_value": None,
        "operator": None,
        "threshold": None,
        "matches": None,
        "method": "condition not machine-evaluated; no supported exact predicate",
    }


def evaluate_guidance(guidance: Any, action: dict[str, Any]) -> dict[str, Any]:
    text = str(guidance or "")
    consumption_match = re.search(
        r"(?:consumption\s+(?:to|around)\s+|consume\s+)([0-9]+(?:\.[0-9]+)?)\s*%",
        text,
        re.I,
    )
    if consumption_match:
        target = float(consumption_match.group(1)) / 100.0
        observed = as_float(action.get("consumption_fraction"))
        return {
            "variable": "consumption_fraction",
            "observed_value": rounded(observed),
            "target": round(target, 6),
            "matches": abs(observed - target) < 1e-9 if observed is not None else None,
            "method": "exact percentage target parsed from the logged guidance",
        }
    return {
        "variable": None,
        "observed_value": None,
        "target": None,
        "matches": None,
        "method": "guidance not machine-evaluated; no supported exact target",
    }


def load_event(run_dir: Path, month: int) -> tuple[dict[str, Any], dict[str, Any]]:
    path = run_dir / "event_log.csv"
    if path.exists():
        row = row_by_month(read_csv_dicts(path), month)
        if row:
            value = short_text(row.get("event_text"))
            return {
                "value": value,
                "event_id": row.get("event_id") or None,
                "event_type": row.get("event_type") or None,
                "sentiment_label": row.get("sentiment_label") or None,
                "v_t": rounded(row.get("v_t")),
                "missing_reason": None
                if value
                else "numeric-only or empty textual-event channel",
            }, {"path": rel(path), "row_index": row["_row_index"]}
    return {
        "value": None,
        "event_id": None,
        "event_type": None,
        "sentiment_label": None,
        "v_t": None,
        "missing_reason": "no event row found for the selected month",
    }, {"path": rel(path) if path.exists() else None, "row_index": None}


def native_copy_check(source_path: Path, normalized_path: Path) -> dict[str, Any]:
    source_hash = sha256_file(source_path)
    normalized_hash = sha256_file(normalized_path)
    return {
        "raw_path": rel(source_path),
        "normalized_path": rel(normalized_path),
        "raw_sha256": source_hash,
        "normalized_sha256": normalized_hash,
        "byte_identical": source_hash == normalized_hash,
    }


def build_trace(
    run_dir: Path, case: dict[str, Any], excluded: list[dict[str, Any]]
) -> dict[str, Any]:
    trajectory = read_csv_dicts(run_dir / "trajectory.csv")
    agent_rows = read_csv_dicts(run_dir / "agent_state.csv")
    metrics = read_csv_dicts(run_dir / "metrics_summary.csv")[0]
    config = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    source_dir = source_dir_for_run(run_dir)
    actions = read_jsonl(run_dir / "actions.jsonl")
    actions_by_key = {action_key(line.data): line for line in actions}
    api_errors_path = run_dir / "api_errors.jsonl"
    api_error_count = len(read_jsonl(api_errors_path)) if file_nonempty(api_errors_path) else 0

    month = int(case["month"])
    agent_id = int(case["agent_id"])
    immediate_month = month + 1
    max_month = max(int(float(row["month"])) for row in trajectory)
    end_month = min(month + 6, max_month)
    checkpoint, episodes = load_checkpoint_episodes(source_dir, agent_id, end_month)

    required_episode_months = {
        month,
        immediate_month,
        end_month,
        *[
            episode_number(value)
            for value in case["retrieval_line"].data["retrieved_episode_ids"]
        ],
    }
    missing_episode_months = sorted(required_episode_months - set(episodes))
    if missing_episode_months:
        raise ValueError(f"checkpoint missing required episodes: {missing_episode_months}")

    query_episode = episodes[month]
    immediate_episode = episodes[immediate_month]
    end_episode = episodes[end_month]
    query_state = macro_state(query_episode)
    prompt_sentiment = rounded(query_episode.get("sentiment"))
    query_personal = query_episode.get("personal_state") or {}

    shared_row = row_by_month(trajectory, month)
    immediate_row = row_by_month(trajectory, immediate_month)
    end_row = row_by_month(trajectory, end_month)
    private_row = find_agent_state(agent_rows, month, agent_id)
    immediate_private = find_agent_state(agent_rows, immediate_month, agent_id)
    end_private = find_agent_state(agent_rows, end_month, agent_id)
    if not all([shared_row, immediate_row, end_row, private_row, immediate_private, end_private]):
        raise ValueError("normalized trajectory or agent-state row missing for selected trace")

    action_line: JsonLine = case["action_line"]
    retrieval_line: JsonLine = case["retrieval_line"]
    rule_lines: list[JsonLine] = case["rule_lines"]
    parsed = action_line.data.get("parsed_action") or {}
    expected_memory_ids = (retrieval_line.data.get("retrieved_episode_ids") or []) + (
        retrieval_line.data.get("selected_rule_ids") or []
    )
    used_memory_ids = action_line.data.get("used_memory_ids") or []

    event, event_pointer = load_event(run_dir, month)
    episode_summaries = build_episode_summaries(
        retrieval_line.data.get("retrieved_episode_ids") or [], episodes
    )

    rule_entries: list[dict[str, Any]] = []
    for line in rule_lines:
        rule = line.data
        source_ids = rule.get("source_episode_ids") or []
        rule_entries.append(
            {
                "rule_id": rule.get("rule_id"),
                "condition": rule.get("condition"),
                "action_guidance": rule.get("action_guidance"),
                "confidence": rounded(rule.get("confidence")),
                "source_episode_ids": source_ids,
                "unique_source_episode_ids": list(dict.fromkeys(source_ids)),
                "condition_evaluation": evaluate_condition(rule.get("condition"), query_state),
                "action_guidance_evaluation": evaluate_guidance(
                    rule.get("action_guidance"), parsed
                ),
                "validity_note": rule.get("validity_note"),
                "validity_note_interpretation": (
                    "Literal emitter label from semantic_rules.jsonl; the extractor does not treat "
                    "this label as independent proof of rule validity."
                ),
            }
        )

    month_actions = [line for line in actions if int(line.data.get("month", -1)) == month]
    post_agent_rows = rows_by_month(agent_rows, immediate_month)
    parsed_labor = [
        (line.data.get("parsed_action") or {}).get("labor_hours") for line in month_actions
    ]
    parsed_consumption = [
        (line.data.get("parsed_action") or {}).get("consumption_fraction")
        for line in month_actions
    ]
    realized_labor = [row.get("labor_hours") for row in post_agent_rows]
    realized_consumption = [row.get("consumption_fraction") for row in post_agent_rows]

    immediate_agent_delta = difference(
        immediate_private.get("wealth"), private_row.get("wealth")
    )
    six_month_agent_delta = difference(end_private.get("wealth"), private_row.get("wealth"))
    next_episode_stored_delta = rounded(
        (immediate_episode.get("outcome") or {}).get("wealth_change")
    )

    source_files: dict[str, dict[str, Any]] = {
        "extractor_script": file_record(SCRIPT_PATH, "case-study extractor"),
        "config": file_record(run_dir / "config.yaml", "normalized run configuration"),
        "export_manifest": file_record(
            run_dir / "export_manifest.json", "normalized-to-raw lineage manifest"
        ),
        "trajectory": file_record(
            run_dir / "trajectory.csv", "normalized aggregate trajectory"
        ),
        "agent_state": file_record(
            run_dir / "agent_state.csv", "normalized per-agent state trajectory"
        ),
        "actions": file_record(run_dir / "actions.jsonl", "normalized native action log"),
        "memory_retrieval": file_record(
            run_dir / "memory_retrieval.jsonl", "normalized native retrieval log"
        ),
        "semantic_rules": file_record(
            run_dir / "semantic_rules.jsonl", "normalized native semantic-rule log"
        ),
        "event_log": file_record(run_dir / "event_log.csv", "normalized event log"),
        "api_errors": file_record(api_errors_path, "normalized API-error log"),
        "raw_summary": file_record(source_dir / "summary.json", "native run summary"),
        "raw_dense_log": file_record(source_dir / "dense_log.pkl", "native dense environment log"),
        "raw_actions": file_record(source_dir / "actions.jsonl", "native action log"),
        "raw_memory_retrieval": file_record(
            source_dir / "memory_retrieval.jsonl", "native retrieval log"
        ),
        "raw_semantic_rules": file_record(
            source_dir / "semantic_rules.jsonl", "native semantic-rule log"
        ),
        "raw_event_log": file_record(source_dir / "event_log.csv", "native event log"),
        "raw_memory_checkpoint": file_record(
            checkpoint, "native serialized dual-track memory checkpoint"
        ),
    }
    if (source_dir / "api_errors.jsonl").exists():
        source_files["raw_api_errors"] = file_record(
            source_dir / "api_errors.jsonl", "native API-error log"
        )

    native_pairs = [
        native_copy_check(source_dir / name, run_dir / name)
        for name in [
            "actions.jsonl",
            "memory_retrieval.jsonl",
            "semantic_rules.jsonl",
            "event_log.csv",
        ]
    ]
    if (source_dir / "api_errors.jsonl").exists():
        native_pairs.append(
            native_copy_check(source_dir / "api_errors.jsonl", api_errors_path)
        )

    retrieval_ids = retrieval_line.data.get("retrieved_episode_ids") or []
    retrieval_scores = retrieval_line.data.get("retrieval_scores") or []
    retrieval_components = retrieval_line.data.get("score_components") or []
    retrieval_components_sum = (
        len(retrieval_ids) == len(retrieval_scores) == len(retrieval_components)
        and all(
            component.get("episode_id") == episode_id
            and abs(
                float(score)
                - sum(
                    float(value)
                    for key, value in component.items()
                    if key != "episode_id"
                )
            )
            < 1e-9
            for episode_id, score, component in zip(
                retrieval_ids, retrieval_scores, retrieval_components
            )
        )
    )
    condition_matches = rule_entries[0]["condition_evaluation"]["matches"]
    guidance_matches = rule_entries[0]["action_guidance_evaluation"]["matches"]
    parser_accepted = bool(action_line.data.get("valid_json"))
    repairs = int(action_line.data.get("repair_attempts") or 0)
    strict_json = parser_accepted and repairs == 0
    raw_output = str(action_line.data.get("raw_output") or "")

    trace: dict[str, Any] = {
        "evidence_scope": "historical_pre_p0_v1",
        "current_method_scientific_evidence": False,
        "method_implementation": "legacy_deterministic_template_memory",
        "trace_id": (
            f"{metrics.get('model')}_seed{metrics.get('seed')}_m{month}_a{agent_id}"
        ).replace(" ", "_"),
        "trace_scope": {
            "classification": "observational_closed_loop_trace",
            "supports_mechanism_trace": True,
            "supports_single_agent_causal_attribution": False,
            "interpretation": (
                "The trace verifies the logged state-to-retrieval-to-rule-to-action chain and "
                "reports the subsequently observed joint-system transition. It contains no "
                "counterfactual intervention isolating the focal agent."
            ),
        },
        "selection_rule": {
            "primary": "lowest_logged_sentiment",
            "fallback_order": [
                "lowest_logged_sentiment",
                "largest_exported_unemployment_jump",
                "largest_average_wealth_drawdown",
            ],
            "agent_tie_break": (
                "Largest absolute parsed-action change from the prior month among agents with "
                "non-empty retrieved episodic memory and a matched semantic rule; ties by agent_id."
            ),
        },
        "run_metadata": {
            "exp_id": str(metrics.get("exp_id")),
            "model": str(metrics.get("model")),
            "setting": str(metrics.get("setting")),
            "variant": str(metrics.get("variant")),
            "seed": int(float(metrics.get("seed"))),
            "num_agents": int(float(metrics.get("num_agents"))),
            "num_months": int(float(metrics.get("num_months"))),
            "source_run_dir": rel(run_dir),
            "raw_source_dir": rel(source_dir),
            "simulation_code_commit": config.get("code_commit") or None,
            "extractor_script_sha256": sha256_file(SCRIPT_PATH),
            "legacy_export_note": config.get("legacy_export_note") or None,
        },
        "case_index": {
            "month": month,
            "agent_id": agent_id,
            "selection_rule_used": case["selection_rule_used"],
            "selected_rule_ids": retrieval_line.data.get("selected_rule_ids") or [],
            "retrieved_episode_ids": retrieval_line.data.get("retrieved_episode_ids") or [],
        },
        "shared_state": {
            "month": month,
            "retrieval_query_state": query_state,
            "decision_prompt_sentiment_cue": prompt_sentiment,
            "selection_signal": {
                "name": "global_sentiment",
                "value": numeric(shared_row, "global_sentiment"),
                "rank_within_run": 1,
                "crash_flag": int(float(shared_row.get("crash_flag", 0))),
                "label": "lowest logged sentiment; not an observed crash",
            },
            "exported_trajectory_snapshot": {
                "avg_wealth": numeric(shared_row, "avg_wealth"),
                "gini": numeric(shared_row, "gini"),
                "price_level": numeric(shared_row, "price_level"),
                "interest_rate_pct": numeric(
                    shared_row, "interest_rate", multiplier=100.0
                ),
                "unemployment_pct": numeric(shared_row, "unemployment_pct"),
                "inflation_pct": numeric(shared_row, "inflation_pct"),
                "gdp": numeric(shared_row, "gdp"),
                "global_sentiment": numeric(shared_row, "global_sentiment"),
            },
            "source_alignment_note": (
                "The corrected normalized trajectory agrees with the raw memory checkpoint on "
                "decision-time inflation, unemployment, interest, and price. The checkpoint also "
                "preserves the exact pre-update sentiment used as the retrieval query; the trajectory "
                "sentiment is the post-update cue shown in the decision prompt and used for selection."
            ),
        },
        "private_state": {
            "agent_id": agent_id,
            "wealth": rounded(query_personal.get("wealth")),
            "income": rounded(query_personal.get("income")),
            "employed": bool(query_personal.get("employed"))
            if query_personal.get("employed") is not None
            else None,
        },
        "text_event": event,
        "retrieved_episodic_memory": {
            "retrieved_episode_ids": retrieval_line.data.get("retrieved_episode_ids") or [],
            "retrieval_scores": retrieval_line.data.get("retrieval_scores") or [],
            "score_components": retrieval_line.data.get("score_components") or [],
            "episode_summaries": episode_summaries,
            "memory_block_hash": stable_hash(retrieval_line.data),
            "temporal_alignment_warning": (
                "The source implementation stores each current decision beside the wealth change "
                "already realized on entry to that month. Consequently, stored outcome E_t aligns "
                "with the preceding transition, while the transition after decision E_t is observed "
                "as the wealth change stored in E_(t+1). The extractor does not treat a same-record "
                "stored outcome as the effect of that record's decision."
            ),
        },
        "retrieved_semantic_rules": rule_entries,
        "parsed_action": {
            "work_propensity": rounded(parsed.get("work")),
            "labor_hours_if_realized": rounded(parsed.get("labor_hours")),
            "consumption_fraction": rounded(parsed.get("consumption_fraction")),
            "rationale_excerpt": short_text(action_line.data.get("rationale"), 240),
            "parser_accepted": action_line.data.get("valid_json"),
            "repair_attempts": repairs,
            "strict_json_without_repair": strict_json,
            "clipped": action_line.data.get("clipped"),
            "used_memory_ids": used_memory_ids,
            "used_memory_ids_match_retrieval": used_memory_ids == expected_memory_ids,
            "prompt_hash": action_line.data.get("prompt_hash") or None,
            "raw_action_hash": sha256_text(raw_output),
        },
        "next_window_outcome": {
            "timing": {
                "decision_month": month,
                "immediate_post_state_month": immediate_month,
                "followup_state_month": end_month,
                "followup_horizon_transitions": end_month - month,
            },
            "concurrent_population_action": {
                "parsed_agent_count": len(month_actions),
                "parsed_mean_labor_hours_if_realized": mean(parsed_labor),
                "parsed_mean_consumption_fraction": mean(parsed_consumption),
                "realized_agent_count": len(post_agent_rows),
                "realized_mean_labor_hours": mean(realized_labor),
                "realized_mean_consumption_fraction": mean(realized_consumption),
            },
            "immediate_transition": {
                "months": [month, immediate_month],
                "post_transition_macro_state": macro_state(immediate_episode),
                "focal_agent": {
                    "start_wealth": numeric(private_row, "wealth"),
                    "end_wealth": numeric(immediate_private, "wealth"),
                    "wealth_delta": immediate_agent_delta,
                    "start_employed": bool(int(float(private_row["employed"]))),
                    "end_employed": bool(int(float(immediate_private["employed"]))),
                    "realized_labor_hours": numeric(immediate_private, "labor_hours"),
                    "realized_consumption_fraction": numeric(
                        immediate_private, "consumption_fraction"
                    ),
                    "transition_reward": numeric(immediate_private, "reward"),
                },
                "aggregate": {
                    "start_avg_wealth": numeric(shared_row, "avg_wealth"),
                    "end_avg_wealth": numeric(immediate_row, "avg_wealth"),
                    "avg_wealth_delta": difference(
                        immediate_row.get("avg_wealth"), shared_row.get("avg_wealth")
                    ),
                    "gini_delta": difference(immediate_row.get("gini"), shared_row.get("gini")),
                },
            },
            "six_month_followup": {
                "months": [month, end_month],
                "end_macro_state": macro_state(end_episode),
                "focal_agent": {
                    "end_wealth": numeric(end_private, "wealth"),
                    "wealth_delta": six_month_agent_delta,
                    "end_employed": bool(int(float(end_private["employed"]))),
                },
                "aggregate": {
                    "end_avg_wealth": numeric(end_row, "avg_wealth"),
                    "avg_wealth_delta": difference(
                        end_row.get("avg_wealth"), shared_row.get("avg_wealth")
                    ),
                    "gini_delta": difference(end_row.get("gini"), shared_row.get("gini")),
                },
            },
            "causal_interpretation": {
                "classification": "observed joint-system follow-up",
                "supports_single_agent_causal_attribution": False,
                "reason": (
                    "All agents act concurrently and the environment evolves stochastically. "
                    "No matched counterfactual holds other agents and shocks fixed, so neither the "
                    "focal nor aggregate change is attributed to this one action."
                ),
            },
        },
        "provenance": {
            "files": source_files,
            "pointers": {
                "decision_state_checkpoint": {
                    "path": rel(checkpoint),
                    "agent_id": agent_id,
                    "episode_timestamp": month,
                },
                "retrieved_episode_checkpoint_records": {
                    "path": rel(checkpoint),
                    "agent_id": agent_id,
                    "episode_timestamps": [
                        episode_number(value)
                        for value in retrieval_line.data.get("retrieved_episode_ids", [])[:3]
                    ],
                },
                "action": {
                    "path": rel(run_dir / "actions.jsonl"),
                    "line_index_zero_based": action_line.line_index,
                },
                "retrieval": {
                    "path": rel(run_dir / "memory_retrieval.jsonl"),
                    "line_index_zero_based": retrieval_line.line_index,
                },
                "semantic_rules": [
                    {
                        "path": rel(run_dir / "semantic_rules.jsonl"),
                        "line_index_zero_based": line.line_index,
                    }
                    for line in rule_lines
                ],
                "selection_and_aggregate_states": {
                    "path": rel(run_dir / "trajectory.csv"),
                    "csv_data_row_indices_zero_based": [
                        shared_row["_row_index"],
                        immediate_row["_row_index"],
                        end_row["_row_index"],
                    ],
                },
                "focal_agent_states": {
                    "path": rel(run_dir / "agent_state.csv"),
                    "csv_data_row_indices_zero_based": [
                        private_row["_row_index"],
                        immediate_private["_row_index"],
                        end_private["_row_index"],
                    ],
                },
                "concurrent_parsed_actions": {
                    "path": rel(run_dir / "actions.jsonl"),
                    "line_indices_zero_based": [line.line_index for line in month_actions],
                },
                "concurrent_realized_actions": {
                    "path": rel(run_dir / "agent_state.csv"),
                    "csv_data_row_indices_zero_based": [
                        row["_row_index"] for row in post_agent_rows
                    ],
                },
                "text_event": event_pointer,
                "api_errors": {
                    "path": rel(api_errors_path),
                    "record_count": api_error_count,
                },
            },
            "normalization_lineage": {
                "native_copy_checks": native_pairs,
                "derived_table_note": (
                    "trajectory.csv and agent_state.csv are deterministic exports from raw "
                    "dense_log.pkl and summary.json. Native action, retrieval, semantic-rule, "
                    "event, and API-error logs are checked byte-for-byte where available."
                ),
            },
        },
        "validation": {
            "schema_valid": False,
            "schema_errors": [],
            "provenance_paths_exist": all(
                (ROOT / record["path"]).exists() for record in source_files.values()
            ),
            "source_file_hashes_recorded": all(
                bool(record["sha256"]) for record in source_files.values()
            ),
            "native_logs_match_raw_source": all(
                item["byte_identical"] for item in native_pairs
            ),
            "source_values_verified": (
                int(action_line.data["month"]) == month
                and int(action_line.data["agent_id"]) == agent_id
                and int(retrieval_line.data["month"]) == month
                and int(retrieval_line.data["agent_id"]) == agent_id
                and int(query_episode["timestamp"]) == month
                and abs(float(query_personal["wealth"]) - float(private_row["wealth"])) < 1e-5
                and abs(float(prompt_sentiment) - float(shared_row["global_sentiment"])) < 1e-5
                and abs(float(query_state["price_level"]) - float(shared_row["price_level"])) < 1e-5
                and abs(float(query_state["interest_rate_pct"]) - 100.0 * float(shared_row["interest_rate"])) < 1e-5
                and abs(float(query_state["unemployment_pct"]) - float(shared_row["unemployment_pct"])) < 1e-5
                and abs(float(query_state["inflation_pct"]) - float(shared_row["inflation_pct"])) < 1e-5
            ),
            "retrieval_score_components_sum": retrieval_components_sum,
            "retrieved_episodes_found_in_checkpoint": not missing_episode_months,
            "selected_rules_found": bool(rule_entries),
            "semantic_precondition_matches_query_state": condition_matches,
            "action_matches_rule_guidance": guidance_matches,
            "action_memory_ids_match_retrieval": used_memory_ids == expected_memory_ids,
            "parser_accepted": parser_accepted,
            "strict_json_without_repair": strict_json,
            "immediate_delta_matches_next_episode_record": (
                immediate_agent_delta is not None
                and next_episode_stored_delta is not None
                and abs(immediate_agent_delta - next_episode_stored_delta) < 1e-5
            ),
            "has_observed_transition": immediate_agent_delta is not None,
            "supports_causal_attribution": False,
            "publishable_as_observational_trace": False,
            "missing_fields": [
                {"field": "text_event.value", "reason": event["missing_reason"]}
            ]
            if event["value"] is None
            else [],
            "excluded_candidates": excluded,
            "warnings": [
                "The selected month is the lowest logged sentiment state, but crash_flag is 0; it must not be labeled a crash.",
                "Decision-time macro values are independently cross-checked between the corrected normalized trajectory and the raw memory checkpoint.",
                "Episodic same-record outcomes are one transition earlier than the decision stored beside them; the extractor uses the next state for action follow-up.",
                "The action was parser-accepted after one formatting repair and is not strict JSON without repair.",
                "Observed follow-up is not a single-agent causal effect because all agents act concurrently and there is no counterfactual replay.",
            ],
        },
    }

    preliminary_errors = sorted(
        Draft202012Validator(SCHEMA).iter_errors(trace), key=lambda error: list(error.path)
    )
    trace["validation"]["schema_errors"] = [error.message for error in preliminary_errors]
    trace["validation"]["schema_valid"] = not preliminary_errors
    trace["validation"]["publishable_as_observational_trace"] = (
        not preliminary_errors
        and trace["validation"]["provenance_paths_exist"]
        and trace["validation"]["native_logs_match_raw_source"]
        and trace["validation"]["source_values_verified"]
        and trace["validation"]["retrieval_score_components_sum"]
        and trace["validation"]["retrieved_episodes_found_in_checkpoint"]
        and trace["validation"]["selected_rules_found"]
        and trace["validation"]["parser_accepted"]
        and trace["validation"]["has_observed_transition"]
    )
    final_errors = sorted(
        Draft202012Validator(SCHEMA).iter_errors(trace), key=lambda error: list(error.path)
    )
    if final_errors:
        trace["validation"]["schema_valid"] = False
        trace["validation"]["schema_errors"] = [error.message for error in final_errors]
        trace["validation"]["publishable_as_observational_trace"] = False
    return trace


def select_trace() -> dict[str, Any]:
    excluded: list[dict[str, Any]] = []
    for run_dir in candidate_dirs():
        ok, reasons = check_candidate(run_dir)
        if not ok:
            excluded.append({"run_dir": rel(run_dir), "reasons": reasons})
            continue
        case, case_reasons = pick_case(run_dir)
        if case is None:
            excluded.append({"run_dir": rel(run_dir), "reasons": case_reasons})
            continue
        return build_trace(run_dir, case, excluded)
    raise SystemExit("No eligible log-grounded case-study run found.")


CSV_FIELDS = [
    "trace_id",
    "model",
    "seed",
    "month",
    "agent_id",
    "selection_label",
    "retrieval_query_inflation_pct",
    "retrieval_query_unemployment_pct",
    "decision_prompt_sentiment_cue",
    "retrieved_episode_ids",
    "semantic_rule_condition",
    "semantic_rule_guidance",
    "rule_confidence",
    "condition_matches",
    "action_labor_hours_if_realized",
    "action_consumption_fraction",
    "action_matches_guidance",
    "immediate_agent_wealth_delta",
    "six_month_agent_wealth_delta",
    "immediate_avg_wealth_delta",
    "six_month_avg_wealth_delta",
    "causal_attribution_supported",
    "source_run",
]


def csv_row(trace: dict[str, Any]) -> dict[str, Any]:
    rule = trace["retrieved_semantic_rules"][0]
    outcome = trace["next_window_outcome"]
    return {
        "trace_id": trace["trace_id"],
        "model": trace["run_metadata"]["model"],
        "seed": trace["run_metadata"]["seed"],
        "month": trace["case_index"]["month"],
        "agent_id": trace["case_index"]["agent_id"],
        "selection_label": trace["shared_state"]["selection_signal"]["label"],
        "retrieval_query_inflation_pct": trace["shared_state"]["retrieval_query_state"][
            "inflation_pct"
        ],
        "retrieval_query_unemployment_pct": trace["shared_state"]["retrieval_query_state"][
            "unemployment_pct"
        ],
        "decision_prompt_sentiment_cue": trace["shared_state"][
            "decision_prompt_sentiment_cue"
        ],
        "retrieved_episode_ids": ";".join(
            trace["retrieved_episodic_memory"]["retrieved_episode_ids"]
        ),
        "semantic_rule_condition": rule["condition"],
        "semantic_rule_guidance": rule["action_guidance"],
        "rule_confidence": rule["confidence"],
        "condition_matches": rule["condition_evaluation"]["matches"],
        "action_labor_hours_if_realized": trace["parsed_action"][
            "labor_hours_if_realized"
        ],
        "action_consumption_fraction": trace["parsed_action"]["consumption_fraction"],
        "action_matches_guidance": rule["action_guidance_evaluation"]["matches"],
        "immediate_agent_wealth_delta": outcome["immediate_transition"]["focal_agent"][
            "wealth_delta"
        ],
        "six_month_agent_wealth_delta": outcome["six_month_followup"]["focal_agent"][
            "wealth_delta"
        ],
        "immediate_avg_wealth_delta": outcome["immediate_transition"]["aggregate"][
            "avg_wealth_delta"
        ],
        "six_month_avg_wealth_delta": outcome["six_month_followup"]["aggregate"][
            "avg_wealth_delta"
        ],
        "causal_attribution_supported": trace["trace_scope"][
            "supports_single_agent_causal_attribution"
        ],
        "source_run": trace["run_metadata"]["source_run_dir"],
    }


def csv_text(trace: dict[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(csv_row(trace))
    return buffer.getvalue()


def wrap_multiline(value: str, width: int) -> str:
    return "\n".join(
        line
        for source_line in value.splitlines()
        for line in (textwrap.wrap(source_line, width=width, break_long_words=False) or [""])
    )


def figure_text(trace: dict[str, Any]) -> list[tuple[str, str]]:
    state = trace["shared_state"]
    query = state["retrieval_query_state"]
    memory = trace["retrieved_episodic_memory"]
    top_episode = memory["episode_summaries"][0]
    rule = trace["retrieved_semantic_rules"][0]
    action = trace["parsed_action"]
    outcome = trace["next_window_outcome"]
    immediate = outcome["immediate_transition"]
    followup = outcome["six_month_followup"]
    top_decision = top_episode["stored_executed_decision"]
    return [
        (
            "Lowest-sentiment decision state",
            f"Month {state['month']} (not a crash)\n"
            f"retrieval inflation {query['inflation_pct']:.2f}%\n"
            f"retrieval unemployment {query['unemployment_pct']:.2f}%\n"
            f"prompt sentiment cue {state['decision_prompt_sentiment_cue']:.3f}",
        ),
        (
            "Retrieved episodic memory",
            f"IDs: {', '.join(memory['retrieved_episode_ids'][:5])}\n"
            f"Top {top_episode['episode_id']}: consume "
            f"{100 * float(top_decision.get('consumption', 0)):.0f}%, "
            f"work {100 * float(top_decision.get('work', 0)):.0f}%",
        ),
        (
            "Retrieved semantic rule",
            f"{rule['condition']}\n{rule['action_guidance']}\n"
            f"confidence {rule['confidence']:.3f}; condition matched "
            f"{rule['condition_evaluation']['matches']}",
        ),
        (
            "Parsed focal action",
            f"labor if realized {action['labor_hours_if_realized']:.0f}h\n"
            f"consume {100 * action['consumption_fraction']:.0f}%\n"
            f"guidance matched {rule['action_guidance_evaluation']['matches']}\n"
            f"parser repairs {action['repair_attempts']}",
        ),
        (
            "Observed joint-system follow-up",
            f"m{state['month']}→m{state['month'] + 1}: focal wealth "
            f"{immediate['focal_agent']['wealth_delta']:+.1f}\n"
            f"m{state['month']}→m{followup['months'][1]}: focal wealth "
            f"{followup['focal_agent']['wealth_delta']:+.1f}\n"
            f"mean wealth {followup['aggregate']['avg_wealth_delta']:+.1f}\n"
            "observational; no focal causal attribution",
        ),
    ]


def write_figure(trace: dict[str, Any]) -> None:
    boxes = figure_text(trace)
    figure, axis = plt.subplots(figsize=(15.8, 4.8))
    axis.axis("off")
    positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = ["#e8f1fb", "#fff2cc", "#e8f5e9", "#fce4ec", "#eeeeee"]
    axis.text(
        0.5,
        0.965,
        "HISTORICAL PRE-P0 V1 EVIDENCE ONLY - legacy GPT-4o E1 trace, not current-method evidence",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="#8b1e1e",
        transform=axis.transAxes,
    )
    for index, ((title, body), x_value, color) in enumerate(zip(boxes, positions, colors)):
        label = f"{title}\n\n{wrap_multiline(body, 29)}"
        axis.text(
            x_value,
            0.59,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            linespacing=1.23,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": color,
                "edgecolor": "#444444",
                "linewidth": 0.9,
            },
            transform=axis.transAxes,
        )
        if index < 4:
            axis.annotate(
                "",
                xy=(positions[index + 1] - 0.095, 0.59),
                xytext=(x_value + 0.095, 0.59),
                xycoords=axis.transAxes,
                arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 1.2},
            )
    axis.text(
        0.5,
        0.105,
        "All displayed values are log-derived; normalized macro values are cross-checked against the raw memory checkpoint.",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#333333",
        transform=axis.transAxes,
    )
    axis.text(
        0.5,
        0.045,
        "The follow-up is observational: all 100 agents act concurrently and no counterfactual isolates the focal action.",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#333333",
        transform=axis.transAxes,
    )
    figure.tight_layout()
    figure.savefig(
        OUT_DIR / "case_study_trace_figure.pdf",
        bbox_inches="tight",
        metadata={
            "Title": "FinEvo historical observational case trace",
            "Subject": "HISTORICAL PRE-P0 V1 EVIDENCE ONLY",
            "CreationDate": None,
        },
    )
    figure.savefig(
        OUT_DIR / "case_study_trace_figure.png",
        bbox_inches="tight",
        dpi=220,
        metadata={
            "Title": "FinEvo historical observational case trace",
            "Description": "HISTORICAL PRE-P0 V1 EVIDENCE ONLY",
        },
    )
    plt.close(figure)


def report_text(trace: dict[str, Any]) -> str:
    validation = trace["validation"]
    rule = trace["retrieved_semantic_rules"][0]
    query = trace["shared_state"]["retrieval_query_state"]
    action = trace["parsed_action"]
    outcome = trace["next_window_outcome"]
    lines = [
        "> [!WARNING]",
        "> **HISTORICAL PRE-P0 V1 EVIDENCE ONLY**",
        ">",
        "> This trace comes from the legacy GPT-4o E1 deterministic-template system. It",
        "> is not a current-method trace and must not be used as evidence for Evidence-",
        "> Grounded Rule Memory v2.",
        "",
        "# Case Study Extraction and Validation Report",
        "",
        "## Verdict",
        "",
        "The selected trace is derived from real run artifacts, not a hand-filled template. Native action, retrieval, semantic-rule, event, and API-error logs are checked byte-for-byte against the raw run directory. The trace is publishable only as an **observational closed-loop mechanism trace**, not as a single-agent causal estimate.",
        "",
        "## Selected Trace",
        "",
        f"- Trace ID: `{trace['trace_id']}`",
        f"- Source run: `{trace['run_metadata']['source_run_dir']}`",
        f"- Raw source: `{trace['run_metadata']['raw_source_dir']}`",
        f"- Month / agent: `{trace['case_index']['month']}` / `{trace['case_index']['agent_id']}`",
        f"- Selection: `{trace['case_index']['selection_rule_used']}`; this month is **not** a crash (`crash_flag=0`).",
        f"- Retrieval query: inflation `{query['inflation_pct']:.6f}%`, unemployment `{query['unemployment_pct']:.6f}%`, sentiment `{query['sentiment']:.6f}`.",
        f"- Rule: `{rule['condition']}` → `{rule['action_guidance']}` at confidence `{rule['confidence']}`.",
        f"- Parsed action: labor-if-realized `{action['labor_hours_if_realized']}` hours, consumption `{action['consumption_fraction']}`.",
        f"- Rule precondition matched: `{rule['condition_evaluation']['matches']}`; action target matched: `{rule['action_guidance_evaluation']['matches']}`.",
        f"- Immediate focal wealth change (state m→m+1): `{outcome['immediate_transition']['focal_agent']['wealth_delta']}`.",
        f"- Six-transition focal / mean wealth changes: `{outcome['six_month_followup']['focal_agent']['wealth_delta']}` / `{outcome['six_month_followup']['aggregate']['avg_wealth_delta']}`.",
        "",
        "## Corrections Relative to the Previous Artifact",
        "",
        "- The figure now says “lowest-sentiment decision state,” not “crash state.”",
        "- The normalized trajectory exporter now carries the most recent annual macro values into non-boundary months; month 69 agrees with the raw checkpoint on inflation, unemployment, interest, and price.",
        "- The raw checkpoint remains necessary to distinguish the pre-update sentiment used for memory retrieval from the post-update sentiment cue shown in the decision prompt.",
        "- The action at month `t` is evaluated against the post-action state at `t+1`; no same-record episodic outcome is treated as its effect.",
        "- `valid_json=true` is reported as parser acceptance. Because one code-fence repair was required, the route is not described as strict JSON without repair.",
        "- The next-state and six-month values are labeled observational joint-system follow-up. They are not attributed causally to one agent.",
        "",
        "## Provenance and Validation",
        "",
        f"- Schema valid: `{validation['schema_valid']}`",
        f"- Source values verified: `{validation['source_values_verified']}`",
        f"- Native logs match raw source: `{validation['native_logs_match_raw_source']}`",
        f"- Retrieval score components sum to logged totals: `{validation['retrieval_score_components_sum']}`",
        f"- Immediate wealth delta matches the next checkpoint episode record: `{validation['immediate_delta_matches_next_episode_record']}`",
        f"- Publishable as observational trace: `{validation['publishable_as_observational_trace']}`",
        "",
        "Every source file is recorded with a SHA-256 digest and byte size under `case_study_trace.json → provenance.files`. Row and line pointers are zero-based and are stored under `provenance.pointers`.",
        "",
        "## Missing Fields",
        "",
    ]
    if validation["missing_fields"]:
        for item in validation["missing_fields"]:
            lines.append(f"- `{item['field']}`: {item['reason']}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Excluded Higher-Priority Candidates", ""])
    if validation["excluded_candidates"]:
        for item in validation["excluded_candidates"]:
            lines.append(f"- `{item['run_dir']}`: {'; '.join(item['reasons'])}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Known Source Limitation",
            "",
            "The simulator appends an episodic record before applying that month’s action. Its stored `wealth_change` is therefore the change entering the month, while the stored decision is the action chosen for the next transition. This artifact exposes that timing explicitly and uses state `t+1` for the observed follow-up. It does not silently relabel the same-record outcome.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python artifacts/case_study/extract_case_study.py",
            "python artifacts/case_study/extract_case_study.py --verify-only",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def verify_outputs(expected: dict[str, Any]) -> None:
    errors: list[str] = []
    trace_path = OUT_DIR / "case_study_trace.json"
    schema_path = OUT_DIR / "case_study_trace_schema.json"
    csv_path = OUT_DIR / "case_study_trace.csv"
    report_path = OUT_DIR / "case_study_extraction_report.md"
    try:
        actual = json.loads(trace_path.read_text())
    except Exception as exc:
        raise SystemExit(f"verification failed: cannot read {trace_path}: {exc}") from exc
    if actual != expected:
        errors.append("case_study_trace.json differs from a fresh source extraction")
    try:
        stored_schema = json.loads(schema_path.read_text())
        if stored_schema != SCHEMA:
            errors.append("case_study_trace_schema.json differs from extractor schema")
    except Exception as exc:
        errors.append(f"schema artifact unreadable: {exc}")
    if csv_path.read_text() != csv_text(expected):
        errors.append("case_study_trace.csv differs from the JSON-derived row")
    if report_path.read_text() != report_text(expected):
        errors.append("case_study_extraction_report.md differs from the JSON-derived report")
    schema_errors = list(Draft202012Validator(SCHEMA).iter_errors(actual))
    if schema_errors:
        errors.extend(f"schema: {error.message}" for error in schema_errors)
    for name, record in actual.get("provenance", {}).get("files", {}).items():
        path = ROOT / record["path"]
        if not path.exists():
            errors.append(f"provenance file missing ({name}): {record['path']}")
        elif sha256_file(path) != record["sha256"]:
            errors.append(f"provenance hash mismatch ({name}): {record['path']}")
    for name in ["case_study_trace_figure.pdf", "case_study_trace_figure.png"]:
        path = OUT_DIR / name
        if not file_nonempty(path):
            errors.append(f"missing or empty figure: {name}")
        elif b"HISTORICAL PRE-P0 V1 EVIDENCE ONLY" not in path.read_bytes():
            errors.append(f"historical evidence marker missing from figure: {name}")
    if errors:
        raise SystemExit("verification failed:\n- " + "\n- ".join(errors))
    print(
        f"verified {actual['trace_id']}: schema, fresh extraction, CSV/report, "
        f"{len(actual['provenance']['files'])} file hashes, and labeled figures "
        "[historical_pre_p0_v1; current_method_scientific_evidence=false]"
    )


def write_outputs(trace: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "case_study_trace_schema.json").write_text(
        json.dumps(SCHEMA, indent=2, ensure_ascii=False) + "\n"
    )
    (OUT_DIR / "case_study_trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False) + "\n"
    )
    (OUT_DIR / "case_study_trace.csv").write_text(csv_text(trace))
    (OUT_DIR / "case_study_extraction_report.md").write_text(report_text(trace))
    write_figure(trace)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="re-extract from sources and verify existing artifacts without rewriting them",
    )
    args = parser.parse_args()
    trace = select_trace()
    if args.verify_only:
        verify_outputs(trace)
        return
    write_outputs(trace)
    verify_outputs(trace)
    print(
        f"selected {trace['trace_id']} from {trace['run_metadata']['source_run_dir']}"
    )
    print(
        "publishable_as_observational_trace="
        f"{trace['validation']['publishable_as_observational_trace']} "
        f"supports_causal_attribution={trace['validation']['supports_causal_attribution']} "
        "evidence_scope=historical_pre_p0_v1 "
        "current_method_scientific_evidence=false"
    )


if __name__ == "__main__":
    main()
