#!/usr/bin/env python3
"""Extract one deterministic, log-grounded FinEvo case study.

The output intentionally contains short snippets and hashes rather than raw
prompts or full API outputs. All visible figure text is generated from the JSON
trace written by this script.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "FinEvo case-study trace",
    "type": "object",
    "required": [
        "trace_id",
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
    "properties": {
        "trace_id": {"type": "string"},
        "selection_rule": {
            "type": "object",
            "required": ["primary", "fallback_order", "agent_tie_break"],
        },
        "run_metadata": {"type": "object"},
        "case_index": {"type": "object"},
        "shared_state": {"type": "object"},
        "private_state": {"type": "object"},
        "text_event": {"type": "object"},
        "retrieved_episodic_memory": {"type": "object"},
        "retrieved_semantic_rules": {"type": "array"},
        "parsed_action": {"type": "object"},
        "next_window_outcome": {"type": "object"},
        "provenance": {"type": "object"},
        "validation": {
            "type": "object",
            "required": [
                "schema_valid",
                "json_compliant_route",
                "has_retrieved_memory",
                "has_semantic_rule",
                "api_errors_recorded",
                "parser_repairs_recorded",
                "acceptable",
                "missing_fields",
                "excluded_candidates",
            ],
        },
    },
}


@dataclass(frozen=True)
class JsonLine:
    line_index: int
    data: dict[str, Any]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def as_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        row["_row_index"] = i
    return rows


def read_jsonl(path: Path) -> list[JsonLine]:
    rows: list[JsonLine] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            rows.append(JsonLine(i, json.loads(line)))
    return rows


def file_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
    except Exception:
        return None
    return out.strip()


def candidate_dirs() -> list[Path]:
    patterns = [
        "runs/E1/GPT-5.2/finevo/default/seed_*",
        "runs/E2/GPT-5.2/finevo/default/seed_*",
        "runs/E5/GPT-5.2/finevo/default-prompt/seed_*",
        "runs/E1/GPT-4o/finevo/default/seed_13",
        "runs/E1/Gemini-3-Flash/finevo/default/seed_13",
        "runs/E1/Qwen3-235B/finevo/default/seed_13",
    ]
    out: list[Path] = []
    for pattern in patterns:
        out.extend(sorted(ROOT.glob(pattern)))
    return [p for p in out if p.is_dir()]


def check_candidate(run_dir: Path) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    required = [
        "config.yaml",
        "trajectory.csv",
        "agent_state.csv",
        "actions.jsonl",
        "memory_retrieval.jsonl",
        "metrics_summary.csv",
        "api_errors.jsonl",
    ]
    for name in required:
        path = run_dir / name
        if not path.exists():
            reasons.append(f"missing {name}")
        elif name not in {"api_errors.jsonl"} and path.stat().st_size == 0:
            reasons.append(f"empty {name}")
    if not (file_nonempty(run_dir / "semantic_rules.jsonl") or file_nonempty(run_dir / "reflection_logs.jsonl")):
        reasons.append("missing non-empty semantic_rules.jsonl or reflection_logs.jsonl")
    if file_nonempty(run_dir / "actions.jsonl"):
        try:
            first = read_jsonl(run_dir / "actions.jsonl")[0].data
            for key in ["valid_json", "repair_attempts", "clipped"]:
                if key not in first:
                    reasons.append(f"actions.jsonl lacks parser field {key}")
                    break
        except Exception as exc:
            reasons.append(f"actions.jsonl unreadable: {exc}")
    return not reasons, reasons


def row_by_month(rows: list[dict[str, Any]], month: int) -> dict[str, Any] | None:
    for row in rows:
        if int(float(row["month"])) == month:
            return row
    return None


def action_key(obj: dict[str, Any]) -> tuple[int, int]:
    return int(obj["month"]), int(obj["agent_id"])


def action_change(current: dict[str, Any], previous: dict[str, Any] | None) -> float:
    if previous is None:
        return -1.0
    cur = current.get("parsed_action") or {}
    prev = previous.get("parsed_action") or {}
    labor_delta = abs(float(cur.get("labor_hours", 0.0)) - float(prev.get("labor_hours", 0.0))) / 168.0
    cons_delta = abs(float(cur.get("consumption_fraction", 0.0)) - float(prev.get("consumption_fraction", 0.0)))
    return labor_delta + cons_delta


def month_candidates(trajectory: list[dict[str, Any]]) -> list[tuple[str, int]]:
    lowest = min(
        trajectory,
        key=lambda r: as_float(r.get("global_sentiment")) if as_float(r.get("global_sentiment")) is not None else 1e9,
    )
    out = [("lowest_regime_cue", int(float(lowest["month"])))]

    jumps: list[tuple[float, int]] = []
    drawdowns: list[tuple[float, int]] = []
    prev = None
    for row in trajectory:
        month = int(float(row["month"]))
        if prev is not None:
            unemp = (as_float(row.get("unemployment_pct")) or 0.0) - (
                as_float(prev.get("unemployment_pct")) or 0.0
            )
            wealth_drawdown = (as_float(prev.get("avg_wealth")) or 0.0) - (
                as_float(row.get("avg_wealth")) or 0.0
            )
            jumps.append((unemp, month))
            drawdowns.append((wealth_drawdown, month))
        prev = row
    if jumps:
        out.append(("largest_unemployment_jump", max(jumps)[1]))
    if drawdowns:
        out.append(("largest_wealth_drawdown", max(drawdowns)[1]))
    return out


def find_rule(
    rules: list[JsonLine], agent_id: int, rule_ids: list[str], month: int
) -> list[JsonLine]:
    matches = [
        r
        for r in rules
        if int(r.data.get("agent_id", -1)) == agent_id
        and r.data.get("rule_id") in rule_ids
        and int(r.data.get("month", 10**9)) <= month
    ]
    if not matches:
        return []
    latest_month = max(int(r.data["month"]) for r in matches)
    return [r for r in matches if int(r.data["month"]) == latest_month]


def find_agent_state(rows: list[dict[str, Any]], month: int, agent_id: int) -> dict[str, Any] | None:
    for row in rows:
        if int(float(row["month"])) == month and int(float(row["agent_id"])) == agent_id:
            return row
    return None


def pick_case(run_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    trajectory = read_csv_dicts(run_dir / "trajectory.csv")
    actions = read_jsonl(run_dir / "actions.jsonl")
    retrievals = read_jsonl(run_dir / "memory_retrieval.jsonl")
    rule_file = run_dir / "semantic_rules.jsonl"
    if not file_nonempty(rule_file):
        rule_file = run_dir / "reflection_logs.jsonl"
    rules = read_jsonl(rule_file)

    actions_by_key = {action_key(a.data): a for a in actions}
    retrievals_by_key = {action_key(r.data): r for r in retrievals}

    reasons: list[str] = []
    for rule_name, month in month_candidates(trajectory):
        eligible: list[tuple[float, int, JsonLine, JsonLine, list[JsonLine]]] = []
        for key, action_line in actions_by_key.items():
            action_month, agent_id = key
            if action_month != month:
                continue
            retrieval_line = retrievals_by_key.get((month, agent_id))
            if retrieval_line is None:
                continue
            retrieved_ids = retrieval_line.data.get("retrieved_episode_ids") or []
            selected_rule_ids = retrieval_line.data.get("selected_rule_ids") or []
            matched_rules = find_rule(rules, agent_id, selected_rule_ids, month)
            if not retrieved_ids or not matched_rules:
                continue
            previous = actions_by_key.get((month - 1, agent_id))
            eligible.append((action_change(action_line.data, previous.data if previous else None), agent_id, action_line, retrieval_line, matched_rules))
        if eligible:
            eligible.sort(key=lambda x: (-x[0], x[1]))
            _, agent_id, action_line, retrieval_line, matched_rules = eligible[0]
            return {
                "selection_rule_used": rule_name,
                "month": month,
                "agent_id": agent_id,
                "action_line": action_line,
                "retrieval_line": retrieval_line,
                "rule_lines": matched_rules,
                "rule_file": rule_file,
            }, reasons
        reasons.append(f"{rule_name} month {month} lacked an agent with non-empty retrieved memory and matched semantic rule")
    return None, reasons


def numeric(row: dict[str, Any], key: str) -> float | None:
    value = as_float(row.get(key))
    return None if value is None else round(value, 6)


def build_episode_sources(
    run_dir: Path,
    episode_ids: list[str],
    agent_id: int,
    trajectory: list[dict[str, Any]],
    agent_rows: list[dict[str, Any]],
    actions_by_key: dict[tuple[int, int], JsonLine],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    snippets: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for episode_id in episode_ids[:3]:
        try:
            month = int(str(episode_id).lstrip("E"))
        except ValueError:
            continue
        trow = row_by_month(trajectory, month)
        arow = find_agent_state(agent_rows, month, agent_id)
        action = actions_by_key.get((month, agent_id))
        snippet = {
            "episode_id": episode_id,
            "month": month,
            "macro": {
                "global_sentiment": numeric(trow or {}, "global_sentiment"),
                "unemployment_pct": numeric(trow or {}, "unemployment_pct"),
                "inflation_pct": numeric(trow or {}, "inflation_pct"),
                "gini": numeric(trow or {}, "gini"),
            },
            "agent_state": {
                "wealth": numeric(arow or {}, "wealth"),
                "income": numeric(arow or {}, "income"),
                "labor_hours": numeric(arow or {}, "labor_hours"),
                "consumption_fraction": numeric(arow or {}, "consumption_fraction"),
            },
            "prior_action": action.data.get("parsed_action") if action else None,
            "prior_rationale_excerpt": short_text(action.data.get("rationale"), 160) if action else None,
        }
        snippets.append(snippet)
        provenance.append(
            {
                "episode_id": episode_id,
                "trajectory": {
                    "path": rel(run_dir / "trajectory.csv"),
                    "row_index": trow.get("_row_index") if trow else None,
                },
                "agent_state": {
                    "path": rel(run_dir / "agent_state.csv"),
                    "row_index": arow.get("_row_index") if arow else None,
                },
                "action": {
                    "path": rel(run_dir / "actions.jsonl"),
                    "line_index": action.line_index if action else None,
                },
            }
        )
    return snippets, provenance


def load_event(run_dir: Path, month: int) -> tuple[dict[str, Any], dict[str, Any]]:
    for name in ["events.jsonl", "event_log.csv"]:
        path = run_dir / name
        if not path.exists():
            continue
        if name.endswith(".jsonl") and file_nonempty(path):
            for line in read_jsonl(path):
                if int(line.data.get("month", -1)) == month:
                    text = line.data.get("event_text")
                    text_value = short_text(text)
                    return {
                        "value": text_value,
                        "event_id": line.data.get("event_id"),
                        "event_type": line.data.get("event_type"),
                        "missing_reason": None if text_value else "event row contains no text",
                    }, {"path": rel(path), "line_index": line.line_index}
        if name.endswith(".csv"):
            rows = read_csv_dicts(path)
            row = row_by_month(rows, month)
            if row:
                text = row.get("event_text")
                text_value = short_text(text)
                return {
                    "value": text_value,
                    "event_id": row.get("event_id") or None,
                    "event_type": row.get("event_type") or None,
                    "missing_reason": None if text_value else "numeric-only or empty textual-event channel",
                }, {"path": rel(path), "row_index": row.get("_row_index")}
    return {
        "value": None,
        "event_id": None,
        "event_type": None,
        "missing_reason": "no event log found for this run",
    }, {"path": None, "row_index": None}


def build_trace(run_dir: Path, case: dict[str, Any], excluded: list[dict[str, Any]]) -> dict[str, Any]:
    trajectory = read_csv_dicts(run_dir / "trajectory.csv")
    agent_rows = read_csv_dicts(run_dir / "agent_state.csv")
    metrics = read_csv_dicts(run_dir / "metrics_summary.csv")[0]
    actions = read_jsonl(run_dir / "actions.jsonl")
    actions_by_key = {action_key(a.data): a for a in actions}
    api_errors_path = run_dir / "api_errors.jsonl"
    api_error_count = len(read_jsonl(api_errors_path)) if file_nonempty(api_errors_path) else 0

    month = int(case["month"])
    agent_id = int(case["agent_id"])
    action_line: JsonLine = case["action_line"]
    retrieval_line: JsonLine = case["retrieval_line"]
    rule_lines: list[JsonLine] = case["rule_lines"]

    shared = row_by_month(trajectory, month) or {}
    private = find_agent_state(agent_rows, month, agent_id) or {}
    next_month = min(month + 6, max(int(float(r["month"])) for r in trajectory))
    next_row = row_by_month(trajectory, next_month) or {}
    event, event_prov = load_event(run_dir, month)

    episode_sources, episode_provenance = build_episode_sources(
        run_dir,
        retrieval_line.data.get("retrieved_episode_ids") or [],
        agent_id,
        trajectory,
        agent_rows,
        actions_by_key,
    )

    rule_entries: list[dict[str, Any]] = []
    for line in rule_lines:
        r = line.data
        rule_entries.append(
            {
                "rule_id": r.get("rule_id"),
                "condition": r.get("condition"),
                "action_guidance": r.get("action_guidance"),
                "confidence": r.get("confidence"),
                "source_episode_ids": r.get("source_episode_ids"),
                "coded_category": r.get("coded_category"),
                "validity_note": r.get("validity_note"),
            }
        )

    parsed = action_line.data.get("parsed_action") or {}
    rationale = short_text(action_line.data.get("rationale"), 220)
    raw_action = action_line.data.get("raw_output") or ""

    trace = {
        "trace_id": f"{metrics.get('model')}_seed{metrics.get('seed')}_m{month}_a{agent_id}".replace(" ", "_"),
        "selection_rule": {
            "primary": case["selection_rule_used"],
            "fallback_order": ["lowest_regime_cue", "largest_unemployment_jump", "largest_wealth_drawdown"],
            "agent_tie_break": "largest absolute action change from previous month among agents with non-empty retrieved episodic memory and matched semantic rule; ties by smaller agent_id",
        },
        "run_metadata": {
            "exp_id": metrics.get("exp_id"),
            "model": metrics.get("model"),
            "setting": metrics.get("setting"),
            "variant": metrics.get("variant"),
            "seed": int(float(metrics.get("seed", 0))),
            "source_run_dir": rel(run_dir),
            "code_commit": git_commit(),
            "json_compliant_route": bool(action_line.data.get("valid_json")),
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
            "global_sentiment": numeric(shared, "global_sentiment"),
            "unemployment_pct": numeric(shared, "unemployment_pct"),
            "inflation_pct": numeric(shared, "inflation_pct"),
            "inflation_dev_abs_pct": numeric(shared, "inflation_dev_abs_pct"),
            "avg_wealth": numeric(shared, "avg_wealth"),
            "gini": numeric(shared, "gini"),
            "crash_flag": int(float(shared.get("crash_flag", 0))) if shared else None,
        },
        "private_state": {
            "agent_id": agent_id,
            "wealth": numeric(private, "wealth"),
            "income": numeric(private, "income"),
            "labor_hours": numeric(private, "labor_hours"),
            "consumption_fraction": numeric(private, "consumption_fraction"),
            "employed": numeric(private, "employed"),
            "private_sentiment": numeric(private, "private_sentiment"),
            "reward": numeric(private, "reward"),
        },
        "text_event": event,
        "retrieved_episodic_memory": {
            "retrieved_episode_ids": retrieval_line.data.get("retrieved_episode_ids") or [],
            "retrieval_scores": retrieval_line.data.get("retrieval_scores") or [],
            "score_components": retrieval_line.data.get("score_components") or [],
            "episode_summaries": episode_sources,
            "memory_block_hash": stable_hash(retrieval_line.data),
        },
        "retrieved_semantic_rules": rule_entries,
        "parsed_action": {
            "work": parsed.get("work"),
            "labor_hours": parsed.get("labor_hours"),
            "consumption_fraction": parsed.get("consumption_fraction"),
            "rationale_excerpt": rationale,
            "valid_json": action_line.data.get("valid_json"),
            "repair_attempts": action_line.data.get("repair_attempts"),
            "clipped": action_line.data.get("clipped"),
            "prompt_hash": action_line.data.get("prompt_hash"),
            "raw_action_hash": sha256_text(raw_action),
        },
        "next_window_outcome": {
            "window_months": [month, next_month],
            "avg_wealth_delta": round((as_float(next_row.get("avg_wealth")) or 0.0) - (as_float(shared.get("avg_wealth")) or 0.0), 6),
            "unemployment_delta_pct": round((as_float(next_row.get("unemployment_pct")) or 0.0) - (as_float(shared.get("unemployment_pct")) or 0.0), 6),
            "inflation_dev_delta_pct": round((as_float(next_row.get("inflation_dev_abs_pct")) or 0.0) - (as_float(shared.get("inflation_dev_abs_pct")) or 0.0), 6),
            "gini_delta": round((as_float(next_row.get("gini")) or 0.0) - (as_float(shared.get("gini")) or 0.0), 6),
            "end_state": {
                "month": next_month,
                "avg_wealth": numeric(next_row, "avg_wealth"),
                "unemployment_pct": numeric(next_row, "unemployment_pct"),
                "inflation_dev_abs_pct": numeric(next_row, "inflation_dev_abs_pct"),
                "gini": numeric(next_row, "gini"),
            },
        },
        "provenance": {
            "selection_rule": {
                "path": rel(run_dir / "trajectory.csv"),
                "rule_source": "deterministic ordering implemented in extract_case_study.py",
            },
            "run_metadata": {
                "config": {"path": rel(run_dir / "config.yaml")},
                "metrics": {"path": rel(run_dir / "metrics_summary.csv"), "row_index": metrics.get("_row_index")},
            },
            "case_index": {
                "retrieval": {
                    "path": rel(run_dir / "memory_retrieval.jsonl"),
                    "line_index": retrieval_line.line_index,
                },
                "action": {"path": rel(run_dir / "actions.jsonl"), "line_index": action_line.line_index},
            },
            "shared_state": {"path": rel(run_dir / "trajectory.csv"), "row_index": shared.get("_row_index")},
            "private_state": {"path": rel(run_dir / "agent_state.csv"), "row_index": private.get("_row_index")},
            "text_event": event_prov,
            "retrieved_episodic_memory": {
                "path": rel(run_dir / "memory_retrieval.jsonl"),
                "line_index": retrieval_line.line_index,
                "derived_episode_sources": episode_provenance,
            },
            "retrieved_semantic_rules": [
                {"path": rel(case["rule_file"]), "line_index": line.line_index} for line in rule_lines
            ],
            "parsed_action": {"path": rel(run_dir / "actions.jsonl"), "line_index": action_line.line_index},
            "next_window_outcome": {
                "path": rel(run_dir / "trajectory.csv"),
                "start_row_index": shared.get("_row_index"),
                "end_row_index": next_row.get("_row_index"),
            },
            "api_errors": {"path": rel(api_errors_path), "record_count": api_error_count},
            "parser_repairs": {
                "path": rel(run_dir / "actions.jsonl"),
                "line_index": action_line.line_index,
                "fields": ["valid_json", "repair_attempts", "clipped"],
            },
            "hashes": {
                "prompt_hash": action_line.data.get("prompt_hash"),
                "raw_action_hash": sha256_text(raw_action),
                "memory_block_hash": stable_hash(retrieval_line.data),
                "rule_block_hash": stable_hash([line.data for line in rule_lines]),
            },
        },
        "validation": {
            "schema_valid": False,
            "json_compliant_route": bool(action_line.data.get("valid_json")),
            "has_retrieved_memory": bool(retrieval_line.data.get("retrieved_episode_ids")),
            "has_semantic_rule": bool(rule_entries),
            "api_errors_recorded": True,
            "parser_repairs_recorded": all(k in action_line.data for k in ["valid_json", "repair_attempts", "clipped"]),
            "acceptable": False,
            "missing_fields": [],
            "excluded_candidates": excluded,
            "notes": [
                "The trace is a log-grounded example only and is not used for cross-model causal claims.",
                "Raw prompts and API outputs are redacted by hash.",
            ],
        },
    }

    missing = []
    if event.get("value") is None:
        missing.append({"field": "text_event.value", "reason": event.get("missing_reason")})
    trace["validation"]["missing_fields"] = missing

    validator = Draft202012Validator(SCHEMA)
    errors = sorted(validator.iter_errors(trace), key=lambda e: list(e.path))
    trace["validation"]["schema_valid"] = not errors
    trace["validation"]["schema_errors"] = [error.message for error in errors]
    trace["validation"]["acceptable"] = (
        trace["validation"]["schema_valid"]
        and trace["validation"]["json_compliant_route"]
        and trace["validation"]["has_retrieved_memory"]
        and trace["validation"]["has_semantic_rule"]
        and trace["validation"]["api_errors_recorded"]
        and trace["validation"]["parser_repairs_recorded"]
    )
    return trace


def figure_text(trace: dict[str, Any]) -> list[tuple[str, str]]:
    shared = trace["shared_state"]
    memory = trace["retrieved_episodic_memory"]
    rule = trace["retrieved_semantic_rules"][0]
    action = trace["parsed_action"]
    outcome = trace["next_window_outcome"]

    memory_line = "IDs: " + ", ".join(memory["retrieved_episode_ids"][:5])
    if memory.get("episode_summaries"):
        ep = memory["episode_summaries"][0]
        memory_line += (
            f"\nTop {ep['episode_id']}: sentiment {ep['macro']['global_sentiment']}, "
            f"infl. {ep['macro']['inflation_pct']}%, action {ep['prior_action']}"
        )

    return [
        (
            "Shared crash state",
            f"Month {shared['month']}\n"
            f"Regime cue {shared['global_sentiment']}\n"
            f"Inflation {shared['inflation_pct']}%\n"
            f"Unemp. {shared['unemployment_pct']}%\n"
            f"Gini {shared['gini']}",
        ),
        ("Retrieved episodic memory", memory_line),
        (
            "Semantic rule",
            f"{rule.get('condition')}\n{rule.get('action_guidance')}\n"
            f"confidence {rule.get('confidence')}",
        ),
        (
            "Parsed action",
            f"labor {action.get('labor_hours')}h\n"
            f"consumption {action.get('consumption_fraction')}\n"
            f"{action.get('rationale_excerpt')}",
        ),
        (
            "Next-window outcome",
            f"Months {outcome['window_months'][0]}->{outcome['window_months'][1]}\n"
            f"wealth delta {outcome['avg_wealth_delta']:.1f}\n"
            f"unemp. delta {outcome['unemployment_delta_pct']:.2f} pp\n"
            f"gini delta {outcome['gini_delta']:.4f}",
        ),
    ]


def write_figure(trace: dict[str, Any]) -> None:
    boxes = figure_text(trace)
    fig, ax = plt.subplots(figsize=(15.5, 4.3))
    ax.axis("off")
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = ["#e8f1fb", "#fff2cc", "#e8f5e9", "#fce4ec", "#eeeeee"]
    for i, ((title, body), x, color) in enumerate(zip(boxes, xs, colors)):
        wrapped = "\n".join(textwrap.wrap(body, width=28, break_long_words=False))
        label = f"{title}\n\n{wrapped}"
        ax.text(
            x,
            0.57,
            label,
            ha="center",
            va="center",
            fontsize=8.6,
            linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.55", facecolor=color, edgecolor="#444444", linewidth=0.9),
            transform=ax.transAxes,
        )
        if i < 4:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.095, 0.57),
                xytext=(x + 0.095, 0.57),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", color="#333333", linewidth=1.2),
            )
    ax.text(
        0.5,
        0.08,
        "All visible fields are extracted from exported logs; raw prompts and API outputs are redacted by hash.",
        ha="center",
        va="center",
        fontsize=8.0,
        color="#333333",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "case_study_trace_figure.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "case_study_trace_figure.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def write_report(trace: dict[str, Any]) -> None:
    excluded = trace["validation"]["excluded_candidates"]
    lines = [
        "# Case Study Extraction Report",
        "",
        "This report is generated by `extract_case_study.py`.",
        "",
        "## Selected Trace",
        "",
        f"- Trace ID: `{trace['trace_id']}`",
        f"- Source run: `{trace['run_metadata']['source_run_dir']}`",
        f"- Selection rule used: `{trace['case_index']['selection_rule_used']}`",
        f"- Month / agent: `{trace['case_index']['month']}` / `{trace['case_index']['agent_id']}`",
        f"- JSON-compliant route: `{trace['validation']['json_compliant_route']}`",
        f"- Acceptable: `{trace['validation']['acceptable']}`",
        "",
        "## Excluded Candidates",
        "",
    ]
    if excluded:
        for item in excluded:
            reason = "; ".join(item["reasons"])
            lines.append(f"- `{item['run_dir']}`: {reason}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "Every visible field in the figure is read from `case_study_trace.json`; the JSON stores source file paths and row or line indices under `provenance`.",
            "Raw prompts and full raw API outputs are not published. The JSON stores `prompt_hash`, `raw_action_hash`, `memory_block_hash`, and `rule_block_hash`.",
            "",
            "## Limitations",
            "",
            "- This is a single log-grounded trace and is not evidence for cross-model causal claims.",
            "- GPT-5.2 controlled runs were excluded from the main case-study figure because their exported `semantic_rules.jsonl` files are empty.",
            "- The selected run uses a numeric-only event channel, so `text_event.value` is null with the documented missing reason.",
        ]
    )
    (OUT_DIR / "case_study_extraction_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "case_study_trace_schema.json").write_text(json.dumps(SCHEMA, indent=2) + "\n")

    excluded: list[dict[str, Any]] = []
    selected_run: Path | None = None
    selected_case: dict[str, Any] | None = None

    for run_dir in candidate_dirs():
        ok, reasons = check_candidate(run_dir)
        if not ok:
            excluded.append({"run_dir": rel(run_dir), "reasons": reasons})
            continue
        case, case_reasons = pick_case(run_dir)
        if case is None:
            excluded.append({"run_dir": rel(run_dir), "reasons": case_reasons})
            continue
        selected_run = run_dir
        selected_case = case
        break

    if selected_run is None or selected_case is None:
        raise SystemExit("No eligible case-study run found.")

    trace = build_trace(selected_run, selected_case, excluded)
    trace_path = OUT_DIR / "case_study_trace.json"
    trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")
    write_figure(trace)
    write_report(trace)
    print(f"selected {trace['trace_id']} from {trace['run_metadata']['source_run_dir']}")
    print(f"acceptable={trace['validation']['acceptable']} schema_valid={trace['validation']['schema_valid']}")
    print(f"wrote {rel(trace_path)}")


if __name__ == "__main__":
    main()
