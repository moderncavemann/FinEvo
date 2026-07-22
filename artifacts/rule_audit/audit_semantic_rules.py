#!/usr/bin/env python3
"""Audit FinEvo semantic-rule provenance, confidence, reuse, and outcomes.

The audit is intentionally conservative:

* A zero-row semantic rule file is an observed zero, not missing data.
* Candidate proposals and rejected rules are not inferred because the current
  logger and implementation do not expose such events.
* Mechanical support checks apply only to the two rule templates implemented
  in ``memory_module.py``. They do not constitute a human semantic-quality or
  causal-effect judgment.
* Future outcomes after rule use are descriptive within-run associations.

The script reads trusted local pickle checkpoints. Do not run it on untrusted
pickle files.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import html
import json
import math
import pickle
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # Required so trusted historical pickles can resolve memory_module classes.
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_RUNS = [
    "data/openai-gpt-4o-E1_finevo-seed13-100agents-240months",
    "data/thirdparty-google_gemini-3-flash-preview-E1_finevo-seed13-100agents-240months",
    "data/openai-gpt-5.2-E2_full-seed13-10agents-24months",
    "data/openai-gpt-5.2-E2_full-seed21-10agents-24months",
    "data/openai-gpt-5.2-E2_full-seed42-10agents-24months",
    "data/openai-gpt-5.2-E2_full-seed87-10agents-24months",
    "data/openai-gpt-5.2-E2_full-seed2026-10agents-24months",
]

RULE_SPECS = {
    "high_inflation_strategy": {
        "condition": "inflation is high (>3%)",
        "kind": "high_inflation",
    },
    "high_unemployment_strategy": {
        "condition": "unemployment is high (>8%)",
        "kind": "high_unemployment",
    },
}

SNAPSHOT_FIELDS = [
    "run_id",
    "model",
    "seed",
    "agent_id",
    "rule_id",
    "month",
    "event_type",
    "observed_condition",
    "condition_schema_match",
    "observed_strategy",
    "expected_strategy",
    "strategy_match",
    "confidence_observed",
    "confidence_expected",
    "confidence_match",
    "confidence_direction",
    "cumulative_source_reference_count",
    "new_source_reference_count",
    "new_source_episode_ids",
    "new_source_missing_count",
    "new_source_future_count",
    "new_source_condition_mismatch_count",
    "new_source_reused_reference_count",
    "new_source_mean_wealth_change",
    "inferred_update_success",
    "support_status",
    "validity_note_observed",
    "rationale_present",
    "coded_category_present",
]

USE_FIELDS = [
    "run_id",
    "model",
    "seed",
    "month",
    "agent_id",
    "rule_id",
    "active_rule_month",
    "confidence",
    "confidence_band",
    "action_guidance",
    "consumption_fraction",
    "guidance_target_fraction",
    "guidance_adherence",
    "outcome_horizon_months",
    "wealth_start",
    "wealth_end",
    "wealth_delta",
    "wealth_delta_positive",
    "employed_end",
    "outcome_status",
]

INVENTORY_FIELDS = [
    "run_id",
    "source_dir",
    "model",
    "seed",
    "exp_id",
    "variant",
    "num_agents",
    "num_months",
    "semantic_module_enabled",
    "summary_present",
    "semantic_file_present",
    "semantic_snapshot_rows",
    "semantic_json_parse_errors",
    "memory_checkpoint_count",
    "episode_records_recovered",
    "episode_conflicts",
    "final_semantic_rule_count",
    "final_strategy_evolution_events",
    "inflation_trigger_episode_count",
    "inflation_trigger_month_count",
    "unemployment_trigger_episode_count",
    "unemployment_trigger_month_count",
    "reflection_episode_count",
    "reflection_nonempty_count",
    "retrieval_rows",
    "action_rows",
    "dense_log_present",
]

SUMMARY_FIELDS = [
    "run_id",
    "model",
    "seed",
    "num_agents",
    "num_months",
    "semantic_snapshot_rows",
    "create_events",
    "merge_update_events",
    "carried_snapshot_rows",
    "invalid_lineage_rows",
    "accepted_update_events_observed",
    "proposed_candidate_events",
    "rejected_candidate_events",
    "verified_supported_update_events",
    "mechanically_contradicted_update_events",
    "not_computable_update_events",
    "mechanical_contradiction_rate",
    "source_references_across_updates",
    "reused_source_references_across_updates",
    "source_reference_reuse_rate",
    "confidence_increase_updates",
    "confidence_decrease_updates",
    "confidence_unchanged_updates",
    "negative_evidence_updates",
    "selected_rule_decisions",
    "selected_rule_decisions_with_active_snapshot",
    "high_confidence_selected_decisions",
    "lower_confidence_selected_decisions",
    "high_confidence_observed_outcomes",
    "lower_confidence_observed_outcomes",
    "high_confidence_mean_next_h_wealth_delta",
    "lower_confidence_mean_next_h_wealth_delta",
    "high_confidence_positive_next_h_wealth_rate",
    "lower_confidence_positive_next_h_wealth_rate",
    "high_confidence_employed_at_h_rate",
    "lower_confidence_employed_at_h_rate",
    "guidance_adherence_rate",
    "rationale_nonempty_rate",
    "coded_category_nonempty_rate",
]


@dataclass
class RunAudit:
    source_dir: Path
    run_id: str
    summary: Dict[str, Any]
    inventory: Dict[str, Any]
    snapshot_rows: List[Dict[str, Any]]
    update_rows: List[Dict[str, Any]]
    use_rows: List[Dict[str, Any]]
    summary_row: Dict[str, Any]
    failure_examples: List[Dict[str, Any]]
    source_hashes: List[Dict[str, Any]]


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _bool_cell(value: Optional[bool]) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def _rate(values: Iterable[Optional[bool]]) -> Optional[float]:
    clean = [value for value in values if value is not None]
    return sum(bool(value) for value in clean) / len(clean) if clean else None


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return str(value)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    errors = 0
    if not path.exists():
        return rows, errors
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            if isinstance(value, dict):
                rows.append(value)
            else:
                errors += 1
    return rows, errors


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")


def _checkpoint_paths(source_dir: Path) -> List[Path]:
    numbered: List[Tuple[int, Path]] = []
    for path in source_dir.glob("memory_*.pkl"):
        match = re.fullmatch(r"memory_(\d+)\.pkl", path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    numbered.sort()
    if numbered:
        return [path for _, path in numbered]
    final_path = source_dir / "memory_final.pkl"
    return [final_path] if final_path.exists() else []


def _final_memory_path(source_dir: Path) -> Optional[Path]:
    final_path = source_dir / "memory_final.pkl"
    if final_path.exists():
        return final_path
    paths = _checkpoint_paths(source_dir)
    return paths[-1] if paths else None


def _episode_record(episode: Any) -> Dict[str, Any]:
    if hasattr(episode, "to_dict"):
        record = episode.to_dict()
    else:
        record = {
            "timestamp": getattr(episode, "timestamp", None),
            "economic_state": getattr(episode, "economic_state", {}) or {},
            "personal_state": getattr(episode, "personal_state", {}) or {},
            "decision": getattr(episode, "decision", {}) or {},
            "outcome": getattr(episode, "outcome", {}) or {},
            "sentiment": getattr(episode, "sentiment", None),
            "reflection": getattr(episode, "reflection", ""),
            "importance": getattr(episode, "importance", None),
        }
    return _jsonable(record)


def _memory_map(value: Any) -> Mapping[Any, Any]:
    return value if isinstance(value, Mapping) else {}


def _load_episode_catalog(source_dir: Path) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], int, int]:
    catalog: Dict[Tuple[int, int], Dict[str, Any]] = {}
    fingerprints: Dict[Tuple[int, int], str] = {}
    conflicts = 0
    paths = _checkpoint_paths(source_dir)
    for path in paths:
        memory_systems = _memory_map(_load_pickle(path))
        for raw_agent_id, memory in memory_systems.items():
            agent_id = _as_int(raw_agent_id)
            if agent_id is None:
                continue
            for episode in getattr(memory, "episodic_memories", []) or []:
                record = _episode_record(episode)
                timestamp = _as_int(record.get("timestamp"))
                if timestamp is None:
                    continue
                key = (agent_id, timestamp)
                fingerprint = hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()
                if key in fingerprints and fingerprints[key] != fingerprint:
                    conflicts += 1
                    continue
                fingerprints[key] = fingerprint
                catalog[key] = record
    return catalog, conflicts, len(paths)


def _final_memory_stats(source_dir: Path) -> Tuple[int, int, Mapping[Any, Any]]:
    path = _final_memory_path(source_dir)
    if path is None:
        return 0, 0, {}
    memory_systems = _memory_map(_load_pickle(path))
    final_rules = 0
    evolution_events = 0
    for memory in memory_systems.values():
        final_rules += len(getattr(memory, "semantic_memories", {}) or {})
        evolution_events += len(getattr(memory, "strategy_evolution", []) or [])
    return final_rules, evolution_events, memory_systems


def _parse_episode_ids(values: Any) -> Tuple[List[int], int]:
    if not isinstance(values, list):
        return [], 1
    parsed: List[int] = []
    errors = 0
    for value in values:
        if isinstance(value, int):
            parsed.append(value)
            continue
        match = re.fullmatch(r"E?(\d+)", str(value).strip())
        if not match:
            errors += 1
            continue
        parsed.append(int(match.group(1)))
    return parsed, errors


def _condition_match(rule_id: str, episode: Mapping[str, Any]) -> Optional[bool]:
    economic = episode.get("economic_state") or {}
    if rule_id == "high_inflation_strategy":
        value = _as_float(economic.get("inflation"))
        return None if value is None else value > 0.03
    if rule_id == "high_unemployment_strategy":
        value = _as_float(economic.get("unemployment_rate"))
        return None if value is None else value > 0.08
    return None


def _expected_update(rule_id: str, episodes: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not episodes or rule_id not in RULE_SPECS:
        return None
    if rule_id == "high_inflation_strategy":
        consumptions = [_as_float((episode.get("decision") or {}).get("consumption"), 0.5) for episode in episodes]
        outcomes = [_as_float((episode.get("outcome") or {}).get("wealth_change"), 0.0) for episode in episodes]
        # Match memory_module.py exactly. Boundary formatting such as 5.5% can
        # differ between numpy.mean and statistics.fmean by one ULP, which is
        # enough to change Python's ``:.0%`` rendering.
        avg_consumption = float(np.mean([float(value) for value in consumptions if value is not None]))
        avg_outcome = float(np.mean([float(value) for value in outcomes if value is not None]))
        if avg_outcome < 0:
            strategy = f"reduce consumption to {max(0.2, avg_consumption - 0.1):.0%}"
        else:
            strategy = f"maintain consumption around {avg_consumption:.0%}"
        return {
            "strategy": strategy,
            "success": avg_outcome > 0,
            "mean_wealth_change": avg_outcome,
        }

    works = [_as_float((episode.get("decision") or {}).get("work"), 0.5) for episode in episodes]
    avg_work = float(np.mean([float(value) for value in works if value is not None]))
    employed = [bool((episode.get("personal_state") or {}).get("employed", False)) for episode in episodes]
    success = sum(employed) > len(employed) / 2
    strategy = f"maintain high work propensity ({avg_work:.0%})" if success else "be flexible in job search"
    outcomes = [_as_float((episode.get("outcome") or {}).get("wealth_change")) for episode in episodes]
    return {
        "strategy": strategy,
        "success": success,
        "mean_wealth_change": _mean(outcomes),
    }


def _infer_rule_snapshots(
    run_id: str,
    model: str,
    seed: Any,
    semantic_rows: Sequence[Mapping[str, Any]],
    episodes: Mapping[Tuple[int, int], Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: MutableMapping[Tuple[int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in semantic_rows:
        agent_id = _as_int(row.get("agent_id"))
        rule_id = str(row.get("rule_id", ""))
        if agent_id is not None:
            grouped[(agent_id, rule_id)].append(row)

    output: List[Dict[str, Any]] = []
    updates: List[Dict[str, Any]] = []
    for (agent_id, rule_id), rows in sorted(grouped.items()):
        ordered = sorted(enumerate(rows), key=lambda item: (_as_int(item[1].get("month"), -1), item[0]))
        previous_ids: List[int] = []
        seen_source_ids: Counter[int] = Counter()
        update_count = 0
        inferred_success_count = 0
        prior_update_confidence: Optional[float] = None

        for _, source_row in ordered:
            month = _as_int(source_row.get("month"), -1)
            current_ids, id_parse_errors = _parse_episode_ids(source_row.get("source_episode_ids", []))
            if id_parse_errors:
                event_type = "invalid_source_id"
                new_ids: List[int] = []
            elif not previous_ids and current_ids:
                event_type = "create"
                new_ids = list(current_ids)
            elif current_ids == previous_ids:
                event_type = "carried_snapshot"
                new_ids = []
            elif len(current_ids) >= len(previous_ids) and current_ids[: len(previous_ids)] == previous_ids:
                event_type = "merge_update"
                new_ids = current_ids[len(previous_ids) :]
            else:
                event_type = "lineage_reset_or_mutation"
                new_ids = []

            condition_expected = RULE_SPECS.get(rule_id, {}).get("condition")
            condition_schema_match = (
                str(source_row.get("condition", "")) == condition_expected if condition_expected is not None else None
            )
            confidence_observed = _as_float(source_row.get("confidence"))
            confidence_expected: Optional[float] = None
            confidence_match: Optional[bool] = None
            confidence_direction = ""
            expected_strategy: Optional[str] = None
            strategy_match: Optional[bool] = None
            inferred_success: Optional[bool] = None
            source_mean_wealth_change: Optional[float] = None
            missing_count = 0
            future_count = 0
            condition_mismatch_count = 0
            reused_count = 0

            if event_type in {"create", "merge_update"}:
                resolved: List[Mapping[str, Any]] = []
                for episode_id in new_ids:
                    reused_count += int(seen_source_ids[episode_id] > 0)
                    episode = episodes.get((agent_id, episode_id))
                    if episode is None:
                        missing_count += 1
                        continue
                    resolved.append(episode)
                    if month is not None and episode_id > month:
                        future_count += 1
                    match = _condition_match(rule_id, episode)
                    if match is False or match is None:
                        condition_mismatch_count += 1

                expected = _expected_update(rule_id, resolved) if len(resolved) == len(new_ids) else None
                if expected is not None:
                    expected_strategy = str(expected["strategy"])
                    strategy_match = str(source_row.get("action_guidance", "")) == expected_strategy
                    inferred_success = bool(expected["success"])
                    source_mean_wealth_change = _as_float(expected.get("mean_wealth_change"))
                    update_count += 1
                    inferred_success_count += int(inferred_success)
                    confidence_expected = 0.5 if update_count == 1 else inferred_success_count / update_count
                    confidence_match = (
                        confidence_observed is not None
                        and math.isclose(confidence_observed, confidence_expected, rel_tol=1e-12, abs_tol=1e-12)
                    )
                elif rule_id in RULE_SPECS:
                    # Preserve the update count only when evidence is complete; otherwise
                    # later expected confidences are intentionally not fabricated.
                    pass

                if prior_update_confidence is None or confidence_observed is None:
                    confidence_direction = "initial"
                elif confidence_observed > prior_update_confidence + 1e-12:
                    confidence_direction = "increase"
                elif confidence_observed < prior_update_confidence - 1e-12:
                    confidence_direction = "decrease"
                else:
                    confidence_direction = "unchanged"
                prior_update_confidence = confidence_observed

                if rule_id not in RULE_SPECS:
                    support_status = "not_computable_unknown_rule_type"
                elif missing_count:
                    support_status = "not_computable_missing_episode"
                elif id_parse_errors or not new_ids:
                    support_status = "invalid_source_lineage"
                elif (
                    condition_schema_match is False
                    or future_count
                    or condition_mismatch_count
                    or strategy_match is False
                    or confidence_match is False
                ):
                    support_status = "mechanically_contradicted"
                else:
                    support_status = "verified_supported"
            elif event_type == "carried_snapshot":
                support_status = "carried_forward_snapshot_no_new_evidence"
            else:
                support_status = "invalid_source_lineage"

            record = {
                "run_id": run_id,
                "model": model,
                "seed": seed,
                "agent_id": agent_id,
                "rule_id": rule_id,
                "month": month,
                "event_type": event_type,
                "observed_condition": source_row.get("condition", ""),
                "condition_schema_match": _bool_cell(condition_schema_match),
                "observed_strategy": source_row.get("action_guidance", ""),
                "expected_strategy": expected_strategy,
                "strategy_match": _bool_cell(strategy_match),
                "confidence_observed": confidence_observed,
                "confidence_expected": confidence_expected,
                "confidence_match": _bool_cell(confidence_match),
                "confidence_direction": confidence_direction,
                "cumulative_source_reference_count": len(current_ids),
                "new_source_reference_count": len(new_ids),
                "new_source_episode_ids": ";".join(f"E{episode_id}" for episode_id in new_ids),
                "new_source_missing_count": missing_count,
                "new_source_future_count": future_count,
                "new_source_condition_mismatch_count": condition_mismatch_count,
                "new_source_reused_reference_count": reused_count,
                "new_source_mean_wealth_change": source_mean_wealth_change,
                "inferred_update_success": _bool_cell(inferred_success),
                "support_status": support_status,
                "validity_note_observed": source_row.get("validity_note", ""),
                "rationale_present": _bool_cell(bool(str(source_row.get("rationale", "")).strip())),
                "coded_category_present": _bool_cell(bool(source_row.get("coded_category"))),
            }
            output.append(record)
            if event_type in {"create", "merge_update"}:
                updates.append(record)
                for episode_id in new_ids:
                    seen_source_ids[episode_id] += 1
            if not id_parse_errors:
                previous_ids = current_ids

    return output, updates


def _state_index(dense_log: Mapping[str, Any]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    output: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for month, state in enumerate(dense_log.get("states") or []):
        if not isinstance(state, Mapping):
            continue
        for raw_agent_id, agent_state in state.items():
            agent_id = _as_int(raw_agent_id)
            if agent_id is None or not isinstance(agent_state, Mapping):
                continue
            inventory = agent_state.get("inventory") or {}
            endogenous = agent_state.get("endogenous") or {}
            output[(month, agent_id)] = {
                "wealth": _as_float(inventory.get("Coin")),
                "employed": bool(endogenous.get("job") != "Unemployment"),
            }
    return output


def _active_rule_before_month(
    updates_by_rule: Mapping[Tuple[int, str], Sequence[Mapping[str, Any]]],
    agent_id: int,
    rule_id: str,
    month: int,
) -> Optional[Mapping[str, Any]]:
    active: Optional[Mapping[str, Any]] = None
    for update in updates_by_rule.get((agent_id, rule_id), []):
        update_month = _as_int(update.get("month"), -1)
        if update_month is not None and update_month < month:
            active = update
        else:
            break
    return active


def _guidance_target(guidance: str) -> Optional[float]:
    match = re.search(r"(\d+)%", guidance)
    return int(match.group(1)) / 100 if match else None


def _guidance_adherence(guidance: str, consumption: Optional[float]) -> Optional[bool]:
    target = _guidance_target(guidance)
    if target is None or consumption is None:
        return None
    if guidance.startswith("reduce consumption"):
        return consumption <= target + 0.02 + 1e-12
    if guidance.startswith("maintain consumption"):
        return abs(consumption - target) <= 0.02 + 1e-12
    return None


def _audit_rule_use(
    run_id: str,
    model: str,
    seed: Any,
    retrieval_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    update_rows: Sequence[Mapping[str, Any]],
    dense_log: Mapping[str, Any],
    horizon: int,
    high_confidence: float,
) -> List[Dict[str, Any]]:
    actions: Dict[Tuple[int, int], Mapping[str, Any]] = {}
    for row in action_rows:
        month = _as_int(row.get("month"))
        agent_id = _as_int(row.get("agent_id"))
        if month is not None and agent_id is not None:
            actions[(month, agent_id)] = row

    updates_by_rule: MutableMapping[Tuple[int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in update_rows:
        updates_by_rule[(_as_int(row.get("agent_id"), -1), str(row.get("rule_id", "")))].append(row)
    for rows in updates_by_rule.values():
        rows.sort(key=lambda row: _as_int(row.get("month"), -1))

    states = _state_index(dense_log)
    output: List[Dict[str, Any]] = []
    for retrieval in retrieval_rows:
        selected = retrieval.get("selected_rule_ids")
        if not isinstance(selected, list) or not selected:
            continue
        month = _as_int(retrieval.get("month"))
        agent_id = _as_int(retrieval.get("agent_id"))
        if month is None or agent_id is None:
            continue
        action_row = actions.get((month, agent_id), {})
        parsed_action = action_row.get("parsed_action") or {}
        consumption = _as_float(parsed_action.get("consumption_fraction"))
        start = states.get((month, agent_id))
        end = states.get((month + horizon, agent_id))

        for raw_rule_id in selected:
            rule_id = str(raw_rule_id)
            active = _active_rule_before_month(updates_by_rule, agent_id, rule_id, month)
            confidence = _as_float(active.get("confidence_observed")) if active else None
            guidance = str(active.get("observed_strategy", "")) if active else ""
            wealth_start = start.get("wealth") if start else None
            wealth_end = end.get("wealth") if end else None
            wealth_delta = (
                float(wealth_end) - float(wealth_start)
                if wealth_start is not None and wealth_end is not None
                else None
            )
            if start is None or end is None:
                outcome_status = "not_computable_missing_state_or_horizon"
            else:
                outcome_status = "observed_descriptive"
            adherence = _guidance_adherence(guidance, consumption)
            output.append({
                "run_id": run_id,
                "model": model,
                "seed": seed,
                "month": month,
                "agent_id": agent_id,
                "rule_id": rule_id,
                "active_rule_month": active.get("month") if active else None,
                "confidence": confidence,
                "confidence_band": (
                    "high" if confidence is not None and confidence >= high_confidence else
                    "lower" if confidence is not None else "unavailable"
                ),
                "action_guidance": guidance,
                "consumption_fraction": consumption,
                "guidance_target_fraction": _guidance_target(guidance),
                "guidance_adherence": _bool_cell(adherence),
                "outcome_horizon_months": horizon,
                "wealth_start": wealth_start,
                "wealth_end": wealth_end,
                "wealth_delta": wealth_delta,
                "wealth_delta_positive": _bool_cell(wealth_delta > 0 if wealth_delta is not None else None),
                "employed_end": _bool_cell(end.get("employed") if end else None),
                "outcome_status": outcome_status,
            })
    return output


def _run_id(summary: Mapping[str, Any], source_dir: Path) -> str:
    exp_id = str(summary.get("exp_id") or "unknown")
    tag = str(summary.get("ablation_tag") or summary.get("variant") or source_dir.name)
    model = str(summary.get("model") or "unknown").split("/", 1)[-1]
    seed = summary.get("random_seed", "unknown")
    return f"{exp_id}:{model}:{tag}:seed{seed}"


def _inventory(
    source_dir: Path,
    summary: Mapping[str, Any],
    semantic_rows: Sequence[Mapping[str, Any]],
    semantic_errors: int,
    episodes: Mapping[Tuple[int, int], Mapping[str, Any]],
    episode_conflicts: int,
    checkpoint_count: int,
    final_rule_count: int,
    evolution_count: int,
    retrieval_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    inflation_episodes = [
        (key, episode) for key, episode in episodes.items()
        if _condition_match("high_inflation_strategy", episode) is True
    ]
    unemployment_episodes = [
        (key, episode) for key, episode in episodes.items()
        if _condition_match("high_unemployment_strategy", episode) is True
    ]
    reflections = [str(episode.get("reflection", "")) for episode in episodes.values()]
    modules = summary.get("modules") or {}
    return {
        "run_id": _run_id(summary, source_dir),
        "source_dir": str(source_dir.relative_to(REPO_ROOT)) if source_dir.is_relative_to(REPO_ROOT) else str(source_dir),
        "model": summary.get("model", ""),
        "seed": summary.get("random_seed", ""),
        "exp_id": summary.get("exp_id", ""),
        "variant": summary.get("variant", summary.get("ablation_tag", "")),
        "num_agents": summary.get("num_agents", ""),
        "num_months": summary.get("episode_length", ""),
        "semantic_module_enabled": bool(modules.get("semantic_memory", False)),
        "summary_present": (source_dir / "summary.json").exists(),
        "semantic_file_present": (source_dir / "semantic_rules.jsonl").exists(),
        "semantic_snapshot_rows": len(semantic_rows),
        "semantic_json_parse_errors": semantic_errors,
        "memory_checkpoint_count": checkpoint_count,
        "episode_records_recovered": len(episodes),
        "episode_conflicts": episode_conflicts,
        "final_semantic_rule_count": final_rule_count,
        "final_strategy_evolution_events": evolution_count,
        "inflation_trigger_episode_count": len(inflation_episodes),
        "inflation_trigger_month_count": len({key[1] for key, _ in inflation_episodes}),
        "unemployment_trigger_episode_count": len(unemployment_episodes),
        "unemployment_trigger_month_count": len({key[1] for key, _ in unemployment_episodes}),
        "reflection_episode_count": len(reflections),
        "reflection_nonempty_count": sum(bool(value.strip()) for value in reflections),
        "retrieval_rows": len(retrieval_rows),
        "action_rows": len(action_rows),
        "dense_log_present": (source_dir / "dense_log.pkl").exists(),
    }


def _summary_row(
    inventory: Mapping[str, Any],
    snapshot_rows: Sequence[Mapping[str, Any]],
    update_rows: Sequence[Mapping[str, Any]],
    use_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    event_counts = Counter(row.get("event_type") for row in snapshot_rows)
    support_counts = Counter(row.get("support_status") for row in update_rows)
    direction_counts = Counter(row.get("confidence_direction") for row in update_rows)
    source_refs = sum(_as_int(row.get("new_source_reference_count"), 0) or 0 for row in update_rows)
    reused_refs = sum(_as_int(row.get("new_source_reused_reference_count"), 0) or 0 for row in update_rows)
    contradicted = support_counts["mechanically_contradicted"]
    verified = support_counts["verified_supported"]
    not_computable = len(update_rows) - contradicted - verified

    high = [row for row in use_rows if row.get("confidence_band") == "high"]
    lower = [row for row in use_rows if row.get("confidence_band") == "lower"]
    high_observed = [row for row in high if row.get("outcome_status") == "observed_descriptive"]
    lower_observed = [row for row in lower if row.get("outcome_status") == "observed_descriptive"]
    with_active = [row for row in use_rows if row.get("confidence_band") in {"high", "lower"}]
    rationale_present = [row.get("rationale_present") == "true" for row in snapshot_rows]
    category_present = [row.get("coded_category_present") == "true" for row in snapshot_rows]
    adherence_values = [
        row.get("guidance_adherence") == "true"
        for row in use_rows
        if row.get("guidance_adherence") in {"true", "false"}
    ]

    def use_mean(rows: Sequence[Mapping[str, Any]], field: str) -> Optional[float]:
        return _mean(_as_float(row.get(field)) for row in rows)

    def use_rate(rows: Sequence[Mapping[str, Any]], field: str) -> Optional[float]:
        values: List[bool] = []
        for row in rows:
            if row.get(field) in {"true", "false"}:
                values.append(row.get(field) == "true")
        return sum(values) / len(values) if values else None

    return {
        "run_id": inventory["run_id"],
        "model": inventory["model"],
        "seed": inventory["seed"],
        "num_agents": inventory["num_agents"],
        "num_months": inventory["num_months"],
        "semantic_snapshot_rows": len(snapshot_rows),
        "create_events": event_counts["create"],
        "merge_update_events": event_counts["merge_update"],
        "carried_snapshot_rows": event_counts["carried_snapshot"],
        "invalid_lineage_rows": sum(
            count for event, count in event_counts.items()
            if event not in {"create", "merge_update", "carried_snapshot"}
        ),
        "accepted_update_events_observed": event_counts["create"] + event_counts["merge_update"],
        "proposed_candidate_events": None,
        "rejected_candidate_events": None,
        "verified_supported_update_events": verified,
        "mechanically_contradicted_update_events": contradicted,
        "not_computable_update_events": not_computable,
        "mechanical_contradiction_rate": _ratio(contradicted, verified + contradicted),
        "source_references_across_updates": source_refs,
        "reused_source_references_across_updates": reused_refs,
        "source_reference_reuse_rate": _ratio(reused_refs, source_refs),
        "confidence_increase_updates": direction_counts["increase"],
        "confidence_decrease_updates": direction_counts["decrease"],
        "confidence_unchanged_updates": direction_counts["unchanged"],
        "negative_evidence_updates": sum(row.get("inferred_update_success") == "false" for row in update_rows),
        "selected_rule_decisions": len(use_rows),
        "selected_rule_decisions_with_active_snapshot": len(with_active),
        "high_confidence_selected_decisions": len(high),
        "lower_confidence_selected_decisions": len(lower),
        "high_confidence_observed_outcomes": len(high_observed),
        "lower_confidence_observed_outcomes": len(lower_observed),
        "high_confidence_mean_next_h_wealth_delta": use_mean(high, "wealth_delta"),
        "lower_confidence_mean_next_h_wealth_delta": use_mean(lower, "wealth_delta"),
        "high_confidence_positive_next_h_wealth_rate": use_rate(high, "wealth_delta_positive"),
        "lower_confidence_positive_next_h_wealth_rate": use_rate(lower, "wealth_delta_positive"),
        "high_confidence_employed_at_h_rate": use_rate(high, "employed_end"),
        "lower_confidence_employed_at_h_rate": use_rate(lower, "employed_end"),
        "guidance_adherence_rate": sum(adherence_values) / len(adherence_values) if adherence_values else None,
        "rationale_nonempty_rate": sum(rationale_present) / len(rationale_present) if rationale_present else None,
        "coded_category_nonempty_rate": sum(category_present) / len(category_present) if category_present else None,
    }


def _failure_examples(
    update_rows: Sequence[Mapping[str, Any]],
    episodes: Mapping[Tuple[int, int], Mapping[str, Any]],
    limit: int = 2,
) -> List[Dict[str, Any]]:
    candidates = [
        row for row in update_rows
        if row.get("inferred_update_success") == "false" and row.get("support_status") == "verified_supported"
    ]
    candidates.sort(key=lambda row: (_as_float(row.get("new_source_mean_wealth_change"), 0.0), row.get("month")))
    output: List[Dict[str, Any]] = []
    for row in candidates[:limit]:
        agent_id = _as_int(row.get("agent_id"), -1)
        source_ids, _ = _parse_episode_ids(str(row.get("new_source_episode_ids", "")).split(";"))
        source_evidence = []
        for episode_id in source_ids:
            episode = episodes.get((agent_id, episode_id), {})
            source_evidence.append({
                "episode_id": f"E{episode_id}",
                "inflation": _as_float((episode.get("economic_state") or {}).get("inflation")),
                "unemployment_rate": _as_float((episode.get("economic_state") or {}).get("unemployment_rate")),
                "consumption": _as_float((episode.get("decision") or {}).get("consumption")),
                "work": _as_float((episode.get("decision") or {}).get("work")),
                "wealth_change": _as_float((episode.get("outcome") or {}).get("wealth_change")),
                "employed": (episode.get("personal_state") or {}).get("employed"),
            })
        output.append({
            "run_id": row.get("run_id"),
            "model": row.get("model"),
            "seed": row.get("seed"),
            "agent_id": agent_id,
            "month": row.get("month"),
            "rule_id": row.get("rule_id"),
            "observed_strategy": row.get("observed_strategy"),
            "confidence_observed": row.get("confidence_observed"),
            "confidence_expected": row.get("confidence_expected"),
            "confidence_direction": row.get("confidence_direction"),
            "source_mean_wealth_change": row.get("new_source_mean_wealth_change"),
            "source_evidence": source_evidence,
            "interpretation_scope": "Observed negative-evidence update; not proof that the semantic rule caused the outcome.",
        })
    return output


def _source_hash_rows(source_dir: Path, run_id: str) -> List[Dict[str, Any]]:
    paths = [
        source_dir / "summary.json",
        source_dir / "semantic_rules.jsonl",
        source_dir / "memory_retrieval.jsonl",
        source_dir / "actions.jsonl",
        source_dir / "dense_log.pkl",
        *_checkpoint_paths(source_dir),
    ]
    unique: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if path.exists() and resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return [{
        "run_id": run_id,
        "file": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    } for path in unique]


def audit_run(source_dir: Path, horizon: int, high_confidence: float) -> RunAudit:
    summary_path = source_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")
    summary = _read_json(summary_path)
    run_id = _run_id(summary, source_dir)
    model = str(summary.get("model", ""))
    seed = summary.get("random_seed", "")

    semantic_rows, semantic_errors = _read_jsonl(source_dir / "semantic_rules.jsonl")
    retrieval_rows, retrieval_errors = _read_jsonl(source_dir / "memory_retrieval.jsonl")
    action_rows, action_errors = _read_jsonl(source_dir / "actions.jsonl")
    if retrieval_errors or action_errors:
        raise ValueError(
            f"JSONL parse errors in {run_id}: retrieval={retrieval_errors}, actions={action_errors}"
        )
    episodes, episode_conflicts, checkpoint_count = _load_episode_catalog(source_dir)
    final_rule_count, evolution_count, _ = _final_memory_stats(source_dir)
    inventory = _inventory(
        source_dir,
        summary,
        semantic_rows,
        semantic_errors,
        episodes,
        episode_conflicts,
        checkpoint_count,
        final_rule_count,
        evolution_count,
        retrieval_rows,
        action_rows,
    )
    snapshot_rows, update_rows = _infer_rule_snapshots(run_id, model, seed, semantic_rows, episodes)
    dense_path = source_dir / "dense_log.pkl"
    dense_log = _load_pickle(dense_path) if dense_path.exists() else {}
    use_rows = _audit_rule_use(
        run_id,
        model,
        seed,
        retrieval_rows,
        action_rows,
        update_rows,
        dense_log if isinstance(dense_log, Mapping) else {},
        horizon,
        high_confidence,
    )
    summary_row = _summary_row(inventory, snapshot_rows, update_rows, use_rows)
    failure_examples = _failure_examples(update_rows, episodes)
    source_hashes = _source_hash_rows(source_dir, run_id)
    return RunAudit(
        source_dir=source_dir,
        run_id=run_id,
        summary=summary,
        inventory=inventory,
        snapshot_rows=snapshot_rows,
        update_rows=update_rows,
        use_rows=use_rows,
        summary_row=summary_row,
        failure_examples=failure_examples,
        source_hashes=source_hashes,
    )


def _fmt_int(value: Any) -> str:
    integer = _as_int(value)
    return f"{integer:,}" if integer is not None else "N/A"


def _fmt_pct(value: Any, decimals: int = 1) -> str:
    number = _as_float(value)
    return f"{number * 100:.{decimals}f}%" if number is not None else "N/A"


def _fmt_float(value: Any, decimals: int = 1) -> str:
    number = _as_float(value)
    return f"{number:,.{decimals}f}" if number is not None else "N/A"


def _short_model(model: str) -> str:
    lowered = model.lower()
    if "gpt-4o" in lowered:
        return "GPT-4o"
    if "gemini" in lowered:
        return "Gemini-3-Flash"
    if "gpt-5.2" in lowered:
        return "GPT-5.2"
    return model.split("/", 1)[-1]


def _tooltip(value: str, tooltip_id: str, datasets: str) -> str:
    return (
        f'<span class="source-tooltip" tabindex="0" aria-describedby="{tooltip_id}">{html.escape(value)}'
        f'<span class="source-tooltip-content" id="{tooltip_id}" role="tooltip">'
        f'Source: FinEvo local experiment artifact<br>Dataset: {html.escape(datasets)}</span></span>'
    )


def _render_bar_fallback(chart_rows: Sequence[Mapping[str, Any]]) -> str:
    width = 960
    left = 245
    right = 70
    top = 28
    row_height = 46
    height = top + row_height * len(chart_rows) + 42
    max_value = max([_as_float(row.get("value"), 0.0) or 0.0 for row in chart_rows] + [1.0])
    plot_width = width - left - right
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Accepted semantic-rule update events by audited run">']
    parts.append(f'<line x1="{left}" y1="18" x2="{left}" y2="{height - 25}" stroke="var(--grid)"/>')
    for index, row in enumerate(chart_rows):
        y = top + index * row_height
        value = _as_float(row.get("value"), 0.0) or 0.0
        bar_width = plot_width * value / max_value
        label = html.escape(str(row.get("run")))
        parts.append(f'<text x="{left - 12}" y="{y + 18}" text-anchor="end" fill="var(--secondary)" font-size="13">{label}</text>')
        if value > 0:
            parts.append(f'<rect x="{left}" y="{y + 4}" width="{bar_width:.1f}" height="20" rx="3" fill="#0169cc"/>')
        parts.append(f'<text x="{left + bar_width + 9:.1f}" y="{y + 19}" fill="var(--text)" font-size="13">{int(value):,}</text>')
    parts.append('</svg>')
    return "".join(parts)


def _html_report(
    audits: Sequence[RunAudit],
    output_dir: Path,
    horizon: int,
    high_confidence: float,
) -> Tuple[Path, Path]:
    summaries = [audit.summary_row for audit in audits]
    e1 = [row for row in summaries if (_as_int(row.get("accepted_update_events_observed"), 0) or 0) > 0]
    zero = [row for row in summaries if (_as_int(row.get("accepted_update_events_observed"), 0) or 0) == 0]
    total_updates = sum(_as_int(row.get("accepted_update_events_observed"), 0) or 0 for row in summaries)
    total_verified = sum(_as_int(row.get("verified_supported_update_events"), 0) or 0 for row in summaries)
    total_contradicted = sum(_as_int(row.get("mechanically_contradicted_update_events"), 0) or 0 for row in summaries)
    total_refs = sum(_as_int(row.get("source_references_across_updates"), 0) or 0 for row in summaries)
    total_reused = sum(_as_int(row.get("reused_source_references_across_updates"), 0) or 0 for row in summaries)
    total_selected = sum(_as_int(row.get("selected_rule_decisions"), 0) or 0 for row in summaries)
    total_negative = sum(_as_int(row.get("negative_evidence_updates"), 0) or 0 for row in summaries)
    total_confidence_decreases = sum(_as_int(row.get("confidence_decrease_updates"), 0) or 0 for row in summaries)

    chart_rows = []
    for audit in audits:
        row = audit.summary_row
        label = f"{_short_model(str(row['model']))} {audit.inventory.get('exp_id')} s{row['seed']}"
        chart_rows.append({
            "run": label,
            "value": _as_int(row.get("accepted_update_events_observed"), 0) or 0,
            "snapshots": _as_int(row.get("semantic_snapshot_rows"), 0) or 0,
            "selected_decisions": _as_int(row.get("selected_rule_decisions"), 0) or 0,
        })

    payload = {
        "charts": [{
            "id": "rule-update-events",
            "height": max(320, 68 + 46 * len(chart_rows)),
            "type": "bar",
            "dataset": {
                "id": "rule-update-events",
                "title": "Accepted semantic-rule update events by audited run",
                "data": chart_rows,
                "chart_spec": {
                    "id": "rule-update-events",
                    "dataset": "rule-update-events",
                    "title": "Accepted semantic-rule update events by audited run",
                    "type": "bar",
                    "encodings": {
                        "x": {"field": "run", "type": "nominal"},
                        "y": {"field": "value", "label": "Logged create or merge events", "type": "quantitative"},
                        "tooltip": [
                            {"field": "snapshots", "label": "Snapshot rows", "type": "quantitative"},
                            {"field": "selected_decisions", "label": "Selected decisions", "type": "quantitative"},
                        ],
                    },
                    "xAxisTitle": "",
                    "yAxisTitle": "Logged create or merge events",
                    "valueFormat": "number",
                    "settings": {"orientation": "horizontal", "groupMode": "grouped"},
                },
            },
        }]
    }
    payload_path = output_dir / "audit_report_payload.json"
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    e1_rows = []
    outcome_rows = []
    tooltip_counter = 0
    for row in e1:
        tooltip_counter += 1
        source = "semantic_rules.jsonl; memory_*.pkl; memory_retrieval.jsonl; actions.jsonl; dense_log.pkl"
        values = [
            html.escape(_short_model(str(row["model"]))),
            html.escape(str(row["seed"])),
            _tooltip(_fmt_int(row["accepted_update_events_observed"]), f"src-u-{tooltip_counter}-1", source),
            _tooltip(_fmt_int(row["verified_supported_update_events"]), f"src-u-{tooltip_counter}-2", source),
            _tooltip(_fmt_pct(row["mechanical_contradiction_rate"]), f"src-u-{tooltip_counter}-3", source),
            _tooltip(_fmt_pct(row["source_reference_reuse_rate"]), f"src-u-{tooltip_counter}-4", source),
            _tooltip(_fmt_int(row["confidence_decrease_updates"]), f"src-u-{tooltip_counter}-5", source),
            _tooltip(_fmt_int(row["selected_rule_decisions"]), f"src-u-{tooltip_counter}-6", source),
            _tooltip(_fmt_pct(row["guidance_adherence_rate"]), f"src-u-{tooltip_counter}-7", source),
        ]
        e1_rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
        for band, prefix in (("High", "high_confidence"), ("Lower", "lower_confidence")):
            tooltip_counter += 1
            outcome_source = "memory_retrieval.jsonl; semantic_rules.jsonl; dense_log.pkl"
            outcome_values = [
                html.escape(_short_model(str(row["model"]))),
                html.escape(band),
                _tooltip(_fmt_int(row[f"{prefix}_selected_decisions"]), f"src-o-{tooltip_counter}-1", outcome_source),
                _tooltip(_fmt_int(row[f"{prefix}_observed_outcomes"]), f"src-o-{tooltip_counter}-2", outcome_source),
                _tooltip(_fmt_float(row[f"{prefix}_mean_next_h_wealth_delta"]), f"src-o-{tooltip_counter}-3", outcome_source),
                _tooltip(_fmt_pct(row[f"{prefix}_positive_next_h_wealth_rate"]), f"src-o-{tooltip_counter}-4", outcome_source),
                _tooltip(_fmt_pct(row[f"{prefix}_employed_at_h_rate"]), f"src-o-{tooltip_counter}-5", outcome_source),
            ]
            outcome_rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in outcome_values) + "</tr>")

    inventory_rows = []
    for audit in audits:
        row = audit.inventory
        tooltip_counter += 1
        source = "summary.json; semantic_rules.jsonl; memory_*.pkl"
        status = "Observed rules" if row["semantic_snapshot_rows"] else "Observed zero rules"
        values = [
            html.escape(_short_model(str(row["model"]))),
            html.escape(str(row["exp_id"])),
            html.escape(str(row["seed"])),
            _tooltip(_fmt_int(row["num_agents"]), f"src-i-{tooltip_counter}-1", "summary.json"),
            _tooltip(_fmt_int(row["num_months"]), f"src-i-{tooltip_counter}-2", "summary.json"),
            _tooltip(_fmt_int(row["semantic_snapshot_rows"]), f"src-i-{tooltip_counter}-3", source),
            _tooltip(_fmt_int(row["final_semantic_rule_count"]), f"src-i-{tooltip_counter}-4", source),
            _tooltip(_fmt_int(row["inflation_trigger_episode_count"]), f"src-i-{tooltip_counter}-5", source),
            html.escape(status),
        ]
        inventory_rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")

    def metric_card(label: str, value: str, tooltip_id: str, datasets: str, note: str) -> str:
        return (
            '<div class="metric">'
            f'<div class="metric-label">{html.escape(label)}</div>'
            f'<div class="metric-value">{_tooltip(value, tooltip_id, datasets)}</div>'
            f'<div class="metric-note">{html.escape(note)}</div></div>'
        )

    chart_height = max(320, 68 + 46 * len(chart_rows))
    static_chart = _render_bar_fallback(chart_rows)
    shell = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>FinEvo semantic-rule reliability audit</title>
  <style>
    :root{{--surface:#fff;--bg:#f7f7f7;--surface-tertiary:#f3f3f3;--text:#0d0d0d;--secondary:#5d5d5d;--muted:#8f8f8f;--border:rgba(13,13,13,.08);--border-strong:rgba(13,13,13,.14);--grid:#e9e9e9;--blue:#0169cc;--purple:#8046d9;--warning:#e25507;--warning-bg:#fff5f0;--positive:#00692a;--positive-bg:#edfaf2;--ds-font:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;--ds-chart-body-height:{chart_height}px}}
    *{{box-sizing:border-box}}html,body{{max-width:100%;overflow-x:clip}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 var(--ds-font)}}.shell{{width:min(1140px,calc(100% - 32px));margin:18px auto 56px;overflow:hidden;border:1px solid var(--border);border-radius:24px;background:var(--surface);box-shadow:0 4px 16px rgba(0,0,0,.05)}}.topbar{{display:flex;min-height:48px;align-items:center;justify-content:space-between;padding:8px 16px;border-bottom:1px solid var(--border);font-weight:500}}.brand{{display:flex;align-items:center;gap:9px}}.mark{{width:18px;height:18px;border-radius:6px;background:linear-gradient(145deg,var(--blue),var(--purple))}}.meta{{max-width:45%;overflow:hidden;color:var(--muted);font-size:12px;text-overflow:ellipsis;white-space:nowrap}}main{{padding:48px}}.reading{{width:min(820px,100%);min-width:0;margin:0 auto}}.wide{{width:100%;margin:28px auto 46px}}.kicker{{margin-bottom:12px;color:var(--blue);font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase}}h1{{margin:0;font-size:36px;font-weight:500;letter-spacing:-2px;line-height:42px}}h2{{margin:0 0 10px;font-size:24px;font-weight:500;letter-spacing:-.7px;line-height:31px}}h3{{margin:0;font-size:14px;font-weight:600}}h1,h2,p,li,.deck,code{{overflow-wrap:anywhere}}p,li{{line-height:22px}}.deck{{margin:16px 0 30px;color:var(--secondary);font-size:16px;line-height:25px}}.summary{{display:grid;grid-template-columns:145px minmax(0,1fr);gap:24px;margin-bottom:30px;padding:24px 0;border-top:1px solid var(--border-strong);border-bottom:1px solid var(--border-strong)}}.summary>*,.summary-body{{min-width:0}}.summary-label{{color:var(--muted);font-size:12px;font-weight:600;letter-spacing:.06em;text-transform:uppercase}}.summary-body{{display:grid;gap:12px}}.summary-body p{{margin:0;color:var(--secondary)}}.summary-body strong{{color:var(--text)}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:46px}}.metric{{min-width:0;min-height:126px;padding:16px 20px;border:1px solid var(--border);border-radius:24px}}.metric-label{{color:var(--secondary);font-size:12px}}.metric-value{{margin:8px 0 5px;font-size:26px;font-weight:500;font-variant-numeric:tabular-nums;letter-spacing:-1px;line-height:34px}}.metric-note{{color:var(--muted);font-size:11px}}.narrative{{margin:44px 0 20px}}.narrative p,.narrative li{{color:var(--secondary)}}.card{{overflow:hidden;border:1px solid var(--border);border-radius:24px;background:var(--surface)}}.card-head{{padding:16px 20px 14px;border-bottom:1px solid var(--border)}}.card-head p{{margin:2px 0 0;color:var(--muted);font-size:12px}}.chart-wrap{{padding:16px 24px 16px 96px}}[data-recharts-chart]{{min-height:{chart_height}px}}[data-recharts-live]{{display:none;width:100%;min-height:{chart_height}px}}[data-recharts-ready="true"] [data-recharts-fallback]{{display:none}}[data-recharts-ready="true"] [data-recharts-live]{{display:block}}.chart-fallback{{width:100%;min-height:{chart_height}px}}.chart-fallback svg{{display:block;width:100%;height:auto;min-height:{chart_height}px;overflow:visible}}.chart-note{{padding:12px 20px 15px;border-top:1px solid var(--border);color:var(--muted);font-size:11px}}.table-card{{margin:12px 0 46px}}.table-scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:12px}}th{{padding:11px 14px;border-bottom:1px solid var(--border);color:var(--muted);font-weight:600;text-align:right;white-space:nowrap}}td{{padding:13px 14px;border-bottom:1px solid var(--border);font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}tbody tr:last-child td{{border-bottom:0}}.caveat{{margin:30px 0;padding:16px 18px;border-radius:14px;background:var(--surface-tertiary);color:var(--secondary);font-size:12px;line-height:19px}}.caveat strong{{color:var(--text)}}code{{padding:1px 5px;border-radius:5px;background:var(--surface-tertiary)}}.source-tooltip{{position:relative;display:inline-block;padding:0;border:0;background:none;color:inherit;font:inherit;cursor:help;text-decoration:underline dotted;text-underline-offset:.18em}}.source-tooltip:focus-visible{{outline:2px solid var(--purple);outline-offset:3px}}.source-tooltip-content{{position:fixed;z-index:1000;top:var(--source-tooltip-top,8px);left:var(--source-tooltip-left,8px);display:none;width:max-content;max-width:min(360px,calc(100vw - 16px));padding:8px 10px;border-radius:8px;background:#171411;color:#fff!important;font-size:12px;font-weight:400;line-height:1.4;white-space:normal;text-align:left;box-shadow:0 8px 24px rgba(23,20,17,.2);opacity:0;visibility:hidden;pointer-events:none}}.source-tooltip:hover .source-tooltip-content,.source-tooltip:focus .source-tooltip-content{{display:block;opacity:1;visibility:visible}}.source-figure{{position:relative}}.source-figure>.source-tooltip{{position:absolute;top:10px;right:10px;z-index:2;padding:4px 7px;border:1px solid var(--border-strong);border-radius:999px;background:var(--surface);color:var(--secondary);font-size:11px;font-weight:600;text-decoration:none}}
    @media(prefers-color-scheme:dark){{:root{{--surface:#212121;--bg:#171717;--surface-tertiary:#303030;--text:#fff;--secondary:#cdcdcd;--muted:#afafaf;--border:rgba(255,255,255,.07);--border-strong:rgba(255,255,255,.16);--grid:#3a3a3a;--blue:#48aaff;--purple:#b58cff;--warning:#ff9e6c;--warning-bg:#4a2206;--positive:#7bd99b;--positive-bg:#143b22}}}}
    @media(max-width:800px){{.shell{{width:100%;margin:0;border:0;border-radius:0}}main{{padding:32px 18px}}.metrics{{grid-template-columns:1fr 1fr}}.summary{{grid-template-columns:1fr;gap:10px}}.chart-wrap{{overflow-x:auto;padding:16px 18px}}.chart-fallback svg{{min-width:720px}}}}
    @media(max-width:600px){{.source-tooltip:hover .source-tooltip-content,.source-tooltip:focus .source-tooltip-content,.source-tooltip:focus-within .source-tooltip-content{{display:none;opacity:0;visibility:hidden}}.source-tooltip[data-source-tooltip-open] .source-tooltip-content{{position:fixed;top:auto;right:16px;bottom:16px;left:16px;display:block;width:auto;max-width:none;max-height:280px;padding:14px 16px;overflow-y:auto;border-radius:14px;font-size:14px;line-height:1.45;transform:none;opacity:1;visibility:visible;pointer-events:auto}}}}
    @media(max-width:520px){{h1{{font-size:32px;line-height:38px}}.metrics{{grid-template-columns:1fr}}}}
  </style>
</head>
<body><div class="shell"><header class="topbar"><div class="brand"><span class="mark"></span>FinEvo audit</div><div class="meta">Trusted local logs · deterministic regeneration</div></header>
<main data-report-audience="technical">
  <article class="reading">
    <div class="kicker">Technical evidence report</div>
    <header data-contract-section="title"><h1>FinEvo semantic-rule reliability audit</h1></header>
    <p class="deck">The available logs support a narrow mechanical-provenance claim for two single-seed E1 runs, but they do not support a claim that LLM reflection proposals were accepted or rejected by a hallucination filter.</p>
    <section class="summary" data-contract-section="technical-summary"><div class="summary-label">Technical summary</div><div class="summary-body">
      <p><strong>What is observed:</strong> {_tooltip(_fmt_int(total_updates), 'src-summary-1', 'semantic_rules.jsonl; memory_*.pkl')} accepted create/merge update events were reconstructed in GPT-4o and Gemini E1 seed 13. Of these, {_tooltip(_fmt_int(total_verified), 'src-summary-2', 'semantic_rules.jsonl; memory_*.pkl')} match the implemented condition, strategy, provenance, and confidence equations; {_tooltip(_fmt_int(total_contradicted), 'src-summary-3', 'semantic_rules.jsonl; memory_*.pkl')} are mechanically contradicted.</p>
      <p><strong>What is not observed:</strong> the current implementation has no LLM-generated candidate-rule stream or rejection event, so proposed/rejected/hallucinated-rule rates are not computable. All observed semantic rules are one hard-coded template, <code>high_inflation_strategy</code>.</p>
      <p><strong>Material risk:</strong> {_tooltip(_fmt_pct(_ratio(total_reused,total_refs)), 'src-summary-4', 'semantic_rules.jsonl; memory_*.pkl')} of source references in accepted updates reuse an episode already cited by an earlier update for the same agent/rule. Confidence updates therefore reuse overlapping evidence rather than independent quarterly samples.</p>
      <p><strong>GPT-5.2 limitation:</strong> all {_tooltip(_fmt_int(len(zero)), 'src-summary-5', 'semantic_rules.jsonl; memory_final.pkl')} audited E2 full seeds contain zero semantic rules because neither trigger threshold was reached. They cannot validate semantic-rule reliability or backbone differences.</p>
    </div></section>
    <section class="metrics">
      {metric_card('Accepted update events', _fmt_int(total_updates), 'src-card-1', 'semantic_rules.jsonl; memory_*.pkl', 'Create + merge events; no proposal denominator')}
      {metric_card('Mechanical contradictions', _fmt_int(total_contradicted), 'src-card-2', 'semantic_rules.jsonl; memory_*.pkl', 'Condition, strategy, provenance, confidence checks')}
      {metric_card('Reused source references', _fmt_pct(_ratio(total_reused,total_refs)), 'src-card-3', 'semantic_rules.jsonl; memory_*.pkl', 'Prior episode reused for same agent/rule')}
      {metric_card('Rule-selected decisions', _fmt_int(total_selected), 'src-card-4', 'memory_retrieval.jsonl', f'Descriptive next-{horizon}-month outcomes only')}
    </section>
  </article>

  <article class="reading" data-contract-section="key-findings"><section class="narrative"><h2>Only two E1 runs contain auditable semantic updates</h2><p>The chart separates observed zero activation from missing logs. Counts are update events inferred from monotonic source-provenance growth, not raw snapshot rows. The five GPT-5.2 E2 rows are genuine zeroes under their 24-month state trajectories.</p></section></article>
  <div class="wide"><figure class="card source-figure"><div class="card-head"><h3>Accepted semantic-rule update events by audited run</h3><p>Two E1 seed-13 runs and five GPT-5.2 E2 full seeds; counts start at zero.</p></div><div class="chart-wrap"><div data-recharts-chart="rule-update-events"><div class="chart-fallback" data-recharts-fallback>{static_chart}</div><div data-recharts-live aria-hidden="true"></div></div></div><figcaption class="chart-note">A bar is shown only for logged create or merge events. Zero is an observed lack of activation, not imputed data.</figcaption><button type="button" class="source-tooltip" aria-describedby="src-chart-1">Source<span class="source-tooltip-content" id="src-chart-1" role="tooltip">Source: FinEvo local experiment artifact<br>Dataset: semantic_rules.jsonl; memory_*.pkl</span></button></figure></div>

  <article class="reading">
    <section class="narrative"><h2>The observed rules pass code-level provenance checks, but evidence is repeatedly reused</h2><p>Every accepted E1 update could be reconstructed from checkpointed episodes. “Verified” means the source episodes satisfy the implemented threshold, the templated strategy recomputes exactly, and confidence matches the implementation. Confidence decreased on {_tooltip(_fmt_int(total_confidence_decreases), 'src-neg-1', 'semantic_rules.jsonl; memory_*.pkl')} of {_tooltip(_fmt_int(total_negative), 'src-neg-2', 'semantic_rules.jsonl; memory_*.pkl')} negative-evidence updates; first-creation neutral initialization and boundary cases explain the remainder. This does not mean the rule is economically correct, non-over-generalized, or causal.</p></section>
    <section class="card table-card"><div class="card-head"><h3>Rule-level audit by active run</h3><p>Exact counts; high confidence is defined as confidence ≥ {high_confidence:.2f}.</p></div><div class="table-scroll"><table><thead><tr><th>Model</th><th>Seed</th><th>Updates</th><th>Verified</th><th>Contradiction rate</th><th>Source reuse</th><th>Confidence decreases</th><th>Rule-selected decisions</th><th>Guidance adherence</th></tr></thead><tbody>{''.join(e1_rows)}</tbody></table></div></section>

    <section class="narrative"><h2>High confidence is not followed by uniformly better observed outcomes</h2><p>GPT-4o has a larger mean next-{horizon}-month wealth change in the high-confidence band, but slightly lower end-of-window employment. Gemini's mean wealth change is nearly unchanged while its positive-wealth-change rate is lower in the high-confidence band. These within-run associations are confounded by month and macro state and must not be reported as causal benefits.</p></section>
    <section class="card table-card"><div class="card-head"><h3>Descriptive outcomes after rule-selected decisions</h3><p>High confidence ≥ {high_confidence:.2f}; observed outcome N excludes decisions without a complete next-{horizon}-month window.</p></div><div class="table-scroll"><table><thead><tr><th>Model</th><th>Confidence band</th><th>Selected N</th><th>Observed outcome N</th><th>Mean wealth Δ</th><th>Positive wealth Δ</th><th>Employed at H</th></tr></thead><tbody>{''.join(outcome_rows)}</tbody></table></div></section>

    <section class="narrative" data-contract-section="scope-data-and-metric-definitions"><h2>Scope and definitions keep zeroes, missingness, and non-computability separate</h2><p>The audit covers exactly seven real runs: two 100-agent × 240-month E1 FinEvo runs and five 10-agent × 24-month GPT-5.2 E2 full runs. A “snapshot row” is a logger output; an “update event” is a create or cumulative-source extension; a “mechanical contradiction” is a mismatch against the checked-in rule equations. Proposal and rejection counts remain blank because no source event exists.</p></section>
    <section class="card table-card"><div class="card-head"><h3>Run and trigger inventory</h3><p>Observed source coverage and whether the semantic thresholds activated.</p></div><div class="table-scroll"><table><thead><tr><th>Model</th><th>Exp.</th><th>Seed</th><th>Agents</th><th>Months</th><th>Snapshots</th><th>Final rules</th><th>High-inflation episodes</th><th>Status</th></tr></thead><tbody>{''.join(inventory_rows)}</tbody></table></div></section>

    <section class="narrative" data-contract-section="methodology"><h2>Checkpoint overlap makes full episode provenance recoverable</h2><p>The script hashes every source file, unions the rolling 24-episode memories from six-month checkpoints, detects conflicting episode records, and then replays the two checked-in deterministic consolidation equations. For rule use, the active rule is the latest update strictly before the decision month. Next-{horizon}-month wealth and employment come from dense state logs; these outcomes are descriptive and are not a no-rule counterfactual.</p></section>

    <section class="narrative" data-contract-section="limitations-uncertainty-and-robustness-checks"><h2>The current logs cannot establish hallucination filtering or causal benefit</h2><ul><li>Semantic consolidation does not consume LLM reflection text; rules are hard-coded templates. A reflection-hallucination rate is therefore not applicable to the observed rule objects.</li><li>No candidate proposal, rejection reason, manual quality code, or counterfactual outcome is logged. Accepted/merged events are observable; proposed and rejected rates are not.</li><li>Only one seed per active E1 model has semantic rules. No cross-seed or matched cross-model inference is justified.</li><li><code>validity_note="supported_by_episode"</code> is logger-supplied for every row; this audit does not treat that label as evidence.</li><li>Rule-use outcomes are state- and time-confounded. They cannot be described as treatment effects.</li></ul></section>

    <section class="narrative" data-contract-section="recommended-next-steps"><h2>Instrument candidate rules before making a stronger rebuttal claim</h2><ol><li>Add immutable <code>proposal_id</code>, candidate text, source episode IDs, accept/reject decision, rejection reason, and rule-version IDs.</li><li>Stop double-counting overlapping source episodes in confidence evidence, or report effective unique evidence alongside update count.</li><li>Run a forced-trigger, five-seed small audit where semantic rules actually activate; include an episodic-only matched control and blinded manual coding.</li><li>Describe the present evidence narrowly: deterministic rule provenance is auditable in two E1 runs; LLM-reflection hallucination filtering is not yet evaluated.</li></ol></section>

    <section class="narrative" data-contract-section="further-questions"><h2>Questions that remain open</h2><ul><li>Should the intended method use LLM reflection to generate candidate rules, or should the paper describe deterministic template consolidation?</li><li>What constitutes an unsupported or over-generalized rule for blinded human annotation?</li><li>Should confidence count unique episodes, disjoint windows, or statistically weighted overlapping evidence?</li></ul></section>
    <section class="caveat"><strong>Interpretation boundary.</strong> This is a reproducible audit of checked-in code and local run artifacts, not a new simulation, economic-validity study, or causal experiment. Detailed CSV/JSONL outputs and SHA-256 hashes accompany this report.</section>
  </article>
</main></div><!-- DATA_ANALYTICS_HTML_REPORT_RUNTIME --></body></html>'''

    shell_path = output_dir / "audit_report_shell.html"
    shell_path.write_text(shell, encoding="utf-8")
    return shell_path, payload_path


def _find_embed_helper(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.exists() else None
    patterns = [
        str(Path.home() / ".codex/plugins/cache/openai-curated-remote/data-analytics/*/skills/build-report/scripts/embed_html_report_runtime.py"),
        str(Path.home() / ".codex/plugins/cache/claude-cowork/data/*/skills/build-report/scripts/embed_html_report_runtime.py"),
    ]
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(Path(path) for path in glob.glob(pattern))
    return sorted(candidates)[-1] if candidates else None


def _embed_report(shell_path: Path, payload_path: Path, output_path: Path, explicit_helper: Optional[str]) -> str:
    helper = _find_embed_helper(explicit_helper)
    if helper is None:
        output_path.write_bytes(shell_path.read_bytes())
        return "static_html_only_embed_helper_not_found"
    subprocess.run(
        [sys.executable, str(helper), "--input", str(shell_path), "--payload", str(payload_path), "--output", str(output_path)],
        check=True,
    )
    # Keep generated artifacts portable: the resolved helper commonly lives
    # below a user-specific plugin-cache path that should not leak into the
    # report manifest or source notes.
    return f"embedded_runtime:{helper.name}"


def _metric_availability_rows(audits: Sequence[RunAudit], horizon: int) -> List[Dict[str, Any]]:
    any_updates = any(audit.update_rows for audit in audits)
    any_use = any(audit.use_rows for audit in audits)
    return [
        {"analysis_item": "semantic snapshot rows", "status": "observed", "reason": "Direct JSONL rows; zero-row files remain observed zeroes."},
        {"analysis_item": "accepted / merged updates", "status": "computed" if any_updates else "observed_zero", "reason": "Inferred from first cumulative source list and monotonic source-list extensions."},
        {"analysis_item": "proposed candidate rules", "status": "not_computable", "reason": "No proposal event or candidate identifier in code or logs."},
        {"analysis_item": "rejected candidate rules", "status": "not_computable", "reason": "No rejection branch, reason, or event in code or logs."},
        {"analysis_item": "mechanically unsupported accepted updates", "status": "computed" if any_updates else "no_denominator", "reason": "Checks known template condition, source existence/time, strategy recomputation, and confidence equation."},
        {"analysis_item": "hallucinated / over-generalized rule rate", "status": "not_applicable_and_not_computable", "reason": "Observed semantic rules are deterministic templates; no LLM candidate text or human semantic labels."},
        {"analysis_item": "confidence decay / validation", "status": "computed" if any_updates else "no_denominator", "reason": "Observed confidence transitions and negative-evidence updates."},
        {"analysis_item": "high-confidence next-H outcome", "status": "computed_descriptive" if any_use else "no_rule_use", "reason": f"Next-{horizon}-month dense-log outcomes; not causal and no counterfactual."},
        {"analysis_item": "failure examples", "status": "computed_descriptive" if any_updates else "no_denominator", "reason": "Lowest observed negative-evidence updates with source episode fields."},
    ]


def _source_notes(audits: Sequence[RunAudit], horizon: int, high_confidence: float, embed_status: str) -> str:
    return f"""# FinEvo rule-audit source notes

## Report contract

- Audience: technical.
- Delivery mode: self-contained HTML.
- Primary question: do existing logs support a rule-level hallucination/error-reinforcement claim?
- Cohort: two E1 FinEvo seed-13 runs with semantic activity plus five GPT-5.2 E2 full seeds with observed zero activity.
- High-confidence threshold: `{high_confidence}`.
- Outcome window: next `{horizon}` simulation months.
- Runtime embedding: `{embed_status}`.

## Chart map

- Segment: key findings.
- Question: which audited runs contain accepted semantic-rule updates?
- Family/type: comparison / horizontal bar.
- Fields: run label, accepted create-or-merge event count; snapshot and selected-decision counts retained for tooltips.
- Takeaway: semantic activity exists only in the two long E1 runs; five GPT-5.2 E2 full seeds are observed zeroes.
- Palette: single blue root plus neutral axes and direct labels.
- Scale: absolute counts starting at zero.
- Static fallback: inline SVG generated from the same rows as `audit_report_payload.json`.

## Omitted charts

Confidence and outcome distributions are left as CSV detail because only one active seed exists per model and a polished comparative visual could imply unsupported cross-model inference. Exact audit lookup is the primary task for those fields.

## Required-structure mapping

The HTML preserves the technical report roles in order: title, technical summary, key findings, scope/data/metric definitions, methodology, limitations/uncertainty/robustness checks, recommended next steps, and further questions.

## Interpretation constraints

- `verified_supported` is a code-level replay result, not an economic-validity or human semantic-quality label.
- Proposal and rejection metrics are not recoverable and remain blank.
- `supported_by_episode` from the source logger is not accepted as audit evidence.
- Rule-use outcomes are descriptive associations without a no-rule counterfactual.
- Pickles are trusted local artifacts; their hashes are saved in `source_manifest.csv`.
"""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", dest="runs", help="Run directory; repeatable. Defaults to the audited E1/E2 cohort.")
    parser.add_argument("--output-dir", default="artifacts/rule_audit/results", help="Output directory relative to repository root.")
    parser.add_argument("--horizon", type=int, default=3, help="Forward outcome horizon in simulation months.")
    parser.add_argument("--high-confidence", type=float, default=0.75, help="High-confidence threshold in [0, 1].")
    parser.add_argument("--embed-helper", default=None, help="Optional path to embed_html_report_runtime.py.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if not 0 <= args.high_confidence <= 1:
        raise ValueError("--high-confidence must be in [0, 1]")
    run_values = args.runs or DEFAULT_RUNS
    run_dirs = [Path(value).expanduser() for value in run_values]
    run_dirs = [path if path.is_absolute() else REPO_ROOT / path for path in run_dirs]
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    audits = [audit_run(path.resolve(), args.horizon, args.high_confidence) for path in run_dirs]
    inventories = [audit.inventory for audit in audits]
    snapshots = [row for audit in audits for row in audit.snapshot_rows]
    updates = [row for audit in audits for row in audit.update_rows]
    uses = [row for audit in audits for row in audit.use_rows]
    summaries = [audit.summary_row for audit in audits]
    failures = [row for audit in audits for row in audit.failure_examples]
    hashes = [row for audit in audits for row in audit.source_hashes]
    availability = _metric_availability_rows(audits, args.horizon)

    _write_csv(output_dir / "run_inventory.csv", inventories, INVENTORY_FIELDS)
    _write_csv(output_dir / "rule_snapshot_audit.csv", snapshots, SNAPSHOT_FIELDS)
    _write_csv(output_dir / "rule_update_audit.csv", updates, SNAPSHOT_FIELDS)
    _write_csv(output_dir / "rule_use_outcomes.csv", uses, USE_FIELDS)
    _write_csv(output_dir / "summary_by_run.csv", summaries, SUMMARY_FIELDS)
    _write_csv(output_dir / "metric_availability.csv", availability, ["analysis_item", "status", "reason"])
    _write_csv(output_dir / "source_manifest.csv", hashes, ["run_id", "file", "bytes", "sha256"])
    _write_jsonl(output_dir / "negative_evidence_examples.jsonl", failures)

    shell_path, payload_path = _html_report(audits, output_dir, args.horizon, args.high_confidence)
    report_path = output_dir / "audit_report.html"
    embed_status = _embed_report(shell_path, payload_path, report_path, args.embed_helper)
    (output_dir / "source_notes.md").write_text(
        _source_notes(audits, args.horizon, args.high_confidence, embed_status),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "command": "python artifacts/rule_audit/audit_semantic_rules.py",
        "run_dirs": [str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path) for path in run_dirs],
        "horizon": args.horizon,
        "high_confidence_threshold": args.high_confidence,
        "embed_status": embed_status,
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        "counts": {
            "runs": len(audits),
            "semantic_snapshot_rows": len(snapshots),
            "accepted_update_events": len(updates),
            "rule_use_rows": len(uses),
            "negative_evidence_examples": len(failures),
        },
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({
        "report": str(report_path),
        "runs": len(audits),
        "semantic_snapshot_rows": len(snapshots),
        "accepted_update_events": len(updates),
        "rule_use_rows": len(uses),
        "mechanical_contradictions": sum(row.get("support_status") == "mechanically_contradicted" for row in updates),
        "embed_status": embed_status,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
