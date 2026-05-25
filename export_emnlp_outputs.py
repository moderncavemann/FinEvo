#!/usr/bin/env python3
"""Export EconAgent runs into the EMNLP experiment output schema.

The old rebuttal experiments wrote ``data/<run>/summary.json`` plus dense
pickle logs.  The EMNLP revision expects one normalized folder per seed:

    runs/<exp_id>/<model>/<setting>/<variant>/seed_<seed>/

This script converts existing runs without modifying the original artifacts.
It also works as a post-processing step after new ``simulate.py`` runs.
Fields that cannot be reconstructed from legacy logs are left blank or marked
with ``legacy_missing`` inside JSONL records.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import pickle as pkl
import re
import shutil
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


METRICS_FIELDS = [
    "exp_id",
    "model",
    "setting",
    "variant",
    "seed",
    "num_agents",
    "num_months",
    "avg_wealth_final",
    "median_wealth_final",
    "gini_final",
    "unemployment_final_pct",
    "inflation_final_pct",
    "inflation_dev_abs_pct",
    "gdp_final",
    "gdp_growth_mean_pct",
    "gdp_volatility_pct",
    "crash_count",
    "mean_crash_recovery_months",
    "invalid_action_rate",
    "api_error_rate",
    "rule_turnover",
    "rule_diversity",
    "mean_prompt_tokens",
    "mean_completion_tokens",
    "total_cost_usd",
    "wall_time_min",
]

TRAJECTORY_FIELDS = [
    "month",
    "seed",
    "model",
    "setting",
    "variant",
    "avg_wealth",
    "median_wealth",
    "gini",
    "unemployment_pct",
    "inflation_pct",
    "inflation_dev_abs_pct",
    "gdp",
    "gdp_growth_pct",
    "price_level",
    "wage_level",
    "interest_rate",
    "global_sentiment",
    "crash_flag",
    "num_invalid_actions",
    "num_api_errors",
]

AGENT_STATE_FIELDS = [
    "month",
    "seed",
    "model",
    "setting",
    "variant",
    "agent_id",
    "skill",
    "wealth",
    "income",
    "labor_hours",
    "consumption_fraction",
    "portfolio_risk_free",
    "portfolio_risky_asset",
    "employed",
    "private_sentiment",
    "tax_paid",
    "redistribution_received",
    "reward",
]

EVENT_FIELDS = [
    "month",
    "seed",
    "event_id",
    "event_text",
    "event_type",
    "source_macro_state",
    "sentiment_label",
    "v_t",
    "classifier_name",
    "classifier_confidence",
    "shuffled_from_month",
]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pkl.load(f)


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_slug(value: str) -> str:
    value = value.replace("/", "_").replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", value).strip("_") or "unknown"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[Any]) -> float:
    vals = [_as_float(v, math.nan) for v in values]
    vals = [v for v in vals if not math.isnan(v)]
    return statistics.mean(vals) if vals else 0.0


def _stdev(values: Iterable[Any]) -> float:
    vals = [_as_float(v, math.nan) for v in values]
    vals = [v for v in vals if not math.isnan(v)]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _gini(values: Iterable[Any]) -> float:
    wealth = sorted(max(_as_float(v), 0.0) for v in values)
    n = len(wealth)
    total = sum(wealth)
    if n == 0 or total == 0:
        return 0.0
    cumulative = sum((i + 1) * value for i, value in enumerate(wealth))
    return (2 * cumulative) / (n * total) - (n + 1) / n


def _state_wealths(state: Dict[str, Any]) -> List[float]:
    wealths = []
    for agent_id, agent_state in state.items():
        if agent_id == "p" or not isinstance(agent_state, dict):
            continue
        wealths.append(_as_float(agent_state.get("inventory", {}).get("Coin")))
    return wealths


def _world_value(world: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in world:
            return _as_float(world[key], default)
    return default


def _infer_seed(source_dir: Path, summary: Dict[str, Any], override: Optional[str]) -> str:
    if override is not None:
        return str(override)
    seed = summary.get("random_seed")
    if seed is not None and seed != "":
        return str(seed)
    match = re.search(r"-seed([0-9]+)", source_dir.name)
    return match.group(1) if match else "unknown"


def _infer_model(summary: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        return override
    model = summary.get("model", "unknown")
    return model.split("/", 1)[-1] if "/" in model else model


def _infer_setting_variant(summary: Dict[str, Any], override_setting: Optional[str], override_variant: Optional[str]) -> Tuple[str, str]:
    if override_setting and override_variant:
        return override_setting, override_variant

    tag = str(summary.get("ablation_tag") or "").lower()
    gap_fixes = bool(summary.get("gap_fixes", False))
    modules = summary.get("modules") or {}

    setting = "finevo" if gap_fixes or any(modules.values()) else "text-only"
    variant = tag or ("default" if setting == "finevo" else "baseline")

    aliases = {
        "gpt_full": ("finevo", "default"),
        "gpt_baseline": ("text-only", "baseline"),
        "gpt_sentonly": ("finevo", "sentiment-only"),
        "gpt_nosent": ("finevo", "without-sentiment"),
        "gpt_norefl": ("finevo", "without-reflection"),
        "gpt_nosem": ("finevo", "without-semantic-memory"),
        "gpt_noep": ("finevo", "without-episodic-memory"),
        "gap_fixed": ("finevo", "default"),
        "baseline": ("text-only", "baseline"),
    }
    if tag in aliases:
        setting, variant = aliases[tag]

    return override_setting or setting, override_variant or variant


def _load_dense_log(source_dir: Path) -> Dict[str, Any]:
    dense_path = source_dir / "dense_log.pkl"
    if not dense_path.exists():
        return {}
    return _read_pickle(dense_path)


def _latest_memory_path(source_dir: Path) -> Optional[Path]:
    candidates = sorted(source_dir.glob("memory_*.pkl"))
    if not candidates:
        return None

    def step(path: Path) -> int:
        match = re.search(r"memory_([0-9]+)\.pkl$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=step)


def _load_memory(source_dir: Path) -> Dict[int, Any]:
    path = _latest_memory_path(source_dir)
    return _read_pickle(path) if path else {}


def _crash_flag(unemployment_pct: float, gdp_growth_pct: float, sentiment: float) -> bool:
    return unemployment_pct >= 10.0 or gdp_growth_pct <= -5.0 or sentiment <= -0.50


def _trajectory_rows(
    dense_log: Dict[str, Any],
    summary: Dict[str, Any],
    seed: str,
    model: str,
    setting: str,
    variant: str,
) -> List[Dict[str, Any]]:
    worlds = dense_log.get("world") or []
    states = dense_log.get("states") or []
    sentiment = summary.get("final_metrics", {}).get("sentiment_history") or []
    rows = []

    for month, world in enumerate(worlds):
        state = states[month] if month < len(states) else {}
        wealths = _state_wealths(state)
        unemp_pct = _world_value(world, "Unemployment Rate") * 100
        infl_pct = _world_value(world, "Price Inflation") * 100
        gdp_growth_pct = _world_value(world, "Real GDP Growth", "Nominal GDP Growth") * 100
        sentiment_value = _as_float(sentiment[month], 0.0) if month < len(sentiment) else 0.0
        rows.append({
            "month": month,
            "seed": seed,
            "model": model,
            "setting": setting,
            "variant": variant,
            "avg_wealth": _mean(wealths),
            "median_wealth": statistics.median(wealths) if wealths else 0.0,
            "gini": _gini(wealths),
            "unemployment_pct": unemp_pct,
            "inflation_pct": infl_pct,
            "inflation_dev_abs_pct": abs(infl_pct - 2.0),
            "gdp": _world_value(world, "Real GDP", "Nominal GDP"),
            "gdp_growth_pct": gdp_growth_pct,
            "price_level": _world_value(world, "Price"),
            "wage_level": _world_value(world, "Wage", "Wage Level"),
            "interest_rate": _world_value(world, "Interest Rate"),
            "global_sentiment": sentiment_value,
            "crash_flag": int(_crash_flag(unemp_pct, gdp_growth_pct, sentiment_value)),
            "num_invalid_actions": "",
            "num_api_errors": "",
        })

    return rows


def _agent_state_rows(
    dense_log: Dict[str, Any],
    seed: str,
    model: str,
    setting: str,
    variant: str,
    max_labor_hours: int = 168,
) -> List[Dict[str, Any]]:
    states = dense_log.get("states") or []
    actions = dense_log.get("actions") or []
    rewards = dense_log.get("rewards") or []
    taxes = dense_log.get("PeriodicTax") or []
    rows = []

    for month, state in enumerate(states):
        action_month = actions[month - 1] if month > 0 and month - 1 < len(actions) else {}
        reward_month = rewards[month - 1] if month > 0 and month - 1 < len(rewards) else {}
        tax_month = taxes[month - 1] if month > 0 and month - 1 < len(taxes) else {}
        for agent_id, agent_state in state.items():
            if agent_id == "p" or not isinstance(agent_state, dict):
                continue
            action = action_month.get(agent_id, {}) if isinstance(action_month, dict) else {}
            tax = tax_month.get(agent_id, {}) if isinstance(tax_month, dict) else {}
            endogenous = agent_state.get("endogenous", {})
            inventory = agent_state.get("inventory", {})
            income = agent_state.get("income", {})
            labor_binary = _as_float(action.get("SimpleLabor"), 0.0)
            consumption_step = action.get("SimpleConsumption")
            consumption_fraction = (
                _as_float(consumption_step) * 0.02
                if consumption_step is not None
                else _as_float(endogenous.get("Consumption Rate"), 0.0)
            )
            rows.append({
                "month": month,
                "seed": seed,
                "model": model,
                "setting": setting,
                "variant": variant,
                "agent_id": agent_id,
                "skill": agent_state.get("skill", ""),
                "wealth": inventory.get("Coin", ""),
                "income": income.get("Coin", ""),
                "labor_hours": labor_binary * max_labor_hours if action else "",
                "consumption_fraction": consumption_fraction,
                "portfolio_risk_free": "",
                "portfolio_risky_asset": "",
                "employed": int(endogenous.get("job") != "Unemployment"),
                "private_sentiment": "",
                "tax_paid": tax.get("tax_paid", ""),
                "redistribution_received": tax.get("lump_sum", ""),
                "reward": reward_month.get(agent_id, "") if isinstance(reward_month, dict) else "",
            })

    return rows


def _legacy_action_rows(
    dense_log: Dict[str, Any],
    seed: str,
    model: str,
    setting: str,
    variant: str,
    max_labor_hours: int = 168,
) -> Iterable[Dict[str, Any]]:
    for month, action_month in enumerate(dense_log.get("actions") or [], start=1):
        if not isinstance(action_month, dict):
            continue
        for agent_id, action in action_month.items():
            if agent_id == "p" or not isinstance(action, dict):
                continue
            consumption_fraction = _as_float(action.get("SimpleConsumption")) * 0.02
            labor_hours = _as_float(action.get("SimpleLabor")) * max_labor_hours
            yield {
                "month": month,
                "seed": seed,
                "model": model,
                "setting": setting,
                "variant": variant,
                "agent_id": int(agent_id) if str(agent_id).isdigit() else agent_id,
                "prompt_hash": "",
                "temperature": "",
                "top_p": "",
                "raw_output": "",
                "parsed_action": {
                    "labor_hours": labor_hours,
                    "consumption_fraction": consumption_fraction,
                },
                "valid_json": None,
                "repair_attempts": "",
                "clipped": "",
                "rationale": "",
                "used_memory_ids": [],
                "legacy_missing": True,
            }


def _semantic_rule_rows(
    memory_systems: Dict[int, Any],
    seed: str,
    model: str,
) -> Iterable[Dict[str, Any]]:
    for agent_id, memory in (memory_systems or {}).items():
        semantic_memories = getattr(memory, "semantic_memories", {}) or {}
        for rule in semantic_memories.values():
            condition = getattr(rule, "condition", "")
            strategy = getattr(rule, "strategy", "")
            rule_id = getattr(rule, "rule_id", "")
            category = []
            if "inflation" in rule_id or "inflation" in condition:
                category.append("inflation_response")
            if "unemployment" in rule_id or "unemployment" in condition:
                category.append("labor_continuity")
            yield {
                "month": getattr(rule, "last_updated", ""),
                "seed": seed,
                "model": model,
                "agent_id": agent_id,
                "rule_id": rule_id,
                "condition": condition,
                "action_guidance": strategy,
                "rationale": "",
                "regime": _regime_from_condition(condition),
                "confidence": getattr(rule, "confidence", ""),
                "source_episode_ids": getattr(rule, "source_episodes", []),
                "coded_category": category,
                "validity_note": "legacy_final_memory_dump",
            }


def _regime_from_condition(condition: str) -> str:
    lowered = condition.lower()
    if "unemployment" in lowered or "high" in lowered:
        return "panic"
    return "neutral"


def _rule_stats(memory_systems: Dict[int, Any], num_agents: int, num_months: int) -> Tuple[float, int]:
    if not memory_systems:
        return 0.0, 0
    unique_rules = set()
    updates = 0
    for memory in memory_systems.values():
        semantic_memories = getattr(memory, "semantic_memories", {}) or {}
        unique_rules.update(semantic_memories.keys())
        updates += len(getattr(memory, "strategy_evolution", []) or [])
    denom = max(num_agents * num_months, 1)
    return updates / denom, len(unique_rules)


def _metrics_row(
    source_dir: Path,
    dense_log: Dict[str, Any],
    memory_systems: Dict[int, Any],
    summary: Dict[str, Any],
    exp_id: str,
    seed: str,
    model: str,
    setting: str,
    variant: str,
) -> Dict[str, Any]:
    final_metrics = summary.get("final_metrics", {})
    worlds = dense_log.get("world") or []
    states = dense_log.get("states") or []
    trajectory = _trajectory_rows(dense_log, summary, seed, model, setting, variant)
    final_world = worlds[-1] if worlds else {}
    final_state = states[-1] if states else {}
    final_wealths = _state_wealths(final_state)
    num_agents = int(summary.get("num_agents") or len(final_wealths) or 0)
    num_months = int(summary.get("episode_length") or max(len(worlds) - 1, 0))
    gdp_growth = [_world_value(world, "Real GDP Growth", "Nominal GDP Growth") * 100 for world in worlds if world]
    unemp_pct = _world_value(final_world, "Unemployment Rate", default=_as_float(final_metrics.get("avg_unemployment"))) * 100
    infl_pct = _world_value(final_world, "Price Inflation", default=_as_float(final_metrics.get("avg_inflation"))) * 100
    rule_turnover, rule_diversity = _rule_stats(memory_systems, num_agents, num_months)

    return {
        "exp_id": exp_id,
        "model": model,
        "setting": setting,
        "variant": variant,
        "seed": seed,
        "num_agents": num_agents,
        "num_months": num_months,
        "avg_wealth_final": final_metrics.get("avg_wealth", _mean(final_wealths)),
        "median_wealth_final": final_metrics.get("median_wealth", statistics.median(final_wealths) if final_wealths else ""),
        "gini_final": final_metrics.get("gini", _gini(final_wealths)),
        "unemployment_final_pct": unemp_pct,
        "inflation_final_pct": infl_pct,
        "inflation_dev_abs_pct": abs(infl_pct - 2.0),
        "gdp_final": _world_value(final_world, "Real GDP", "Nominal GDP"),
        "gdp_growth_mean_pct": _mean(gdp_growth),
        "gdp_volatility_pct": _stdev(gdp_growth),
        "crash_count": sum(int(row.get("crash_flag", 0)) for row in trajectory),
        "mean_crash_recovery_months": "",
        "invalid_action_rate": summary.get("error_rate", ""),
        "api_error_rate": "",
        "rule_turnover": rule_turnover,
        "rule_diversity": rule_diversity,
        "mean_prompt_tokens": "",
        "mean_completion_tokens": "",
        "total_cost_usd": summary.get("total_cost", ""),
        "wall_time_min": "",
        "legacy_source_dir": str(source_dir),
    }


def _config_doc(
    source_dir: Path,
    summary: Dict[str, Any],
    exp_id: str,
    seed: str,
    model: str,
    setting: str,
    variant: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    sentiment = summary.get("sentiment_config") or {}
    return {
        "exp_id": exp_id,
        "model_display_name": model,
        "model_exact_id": summary.get("model", model),
        "api_provider": str(summary.get("model", "")).split("/", 1)[0] if "/" in str(summary.get("model", "")) else "",
        "api_access_date": args.api_access_date or "",
        "code_commit": args.code_commit or "",
        "prompt_version": args.prompt_version,
        "reflection_prompt_version": args.reflection_prompt_version,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": seed,
        "num_agents": summary.get("num_agents", ""),
        "num_months": summary.get("episode_length", ""),
        "setting": setting,
        "variant": variant,
        "environment_params": {
            "alpha_sentiment": sentiment.get("inertia", 0.70),
            "beta_price": sentiment.get("inflation_sens", 0.15),
            "gamma_unemployment": sentiment.get("unemp_sens", -0.20),
            "delta_gdp": sentiment.get("gdp_sens", 0.10),
            "sigma_sentiment": sentiment.get("news_std", 0.05),
            "rho_diffusion": sentiment.get("diffusion", 0.30),
            "episodic_capacity": args.episodic_capacity,
            "semantic_capacity": args.semantic_capacity,
            "retrieval_k": args.retrieval_k,
            "rule_budget_m": args.rule_budget_m,
            "reflection_period_months": args.reflection_period_months,
            "tax_brackets": "us-federal-single-filer-2018-scaled",
        },
        "legacy_source_dir": str(source_dir),
        "legacy_export_note": "Some JSONL fields are blank because old rebuttal runs did not log raw prompts, raw outputs, retrieval traces, or API errors in the EMNLP schema.",
    }


def export_one(source_dir: Path, args: argparse.Namespace) -> Path:
    summary_path = source_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary.json: {summary_path}")

    summary = _read_json(summary_path)
    dense_log = _load_dense_log(source_dir)
    memory_systems = _load_memory(source_dir)
    exp_id = args.exp_id
    seed = _infer_seed(source_dir, summary, args.seed)
    model = _infer_model(summary, args.model)
    setting, variant = _infer_setting_variant(summary, args.setting, args.variant)
    out_dir = Path(args.runs_root) / exp_id / _safe_slug(model) / _safe_slug(setting) / _safe_slug(variant) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _config_doc(source_dir, summary, exp_id, seed, model, setting, variant, args)
    with (out_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    metrics = _metrics_row(source_dir, dense_log, memory_systems, summary, exp_id, seed, model, setting, variant)
    _write_csv(out_dir / "metrics_summary.csv", [metrics], METRICS_FIELDS)
    _write_csv(out_dir / "trajectory.csv", _trajectory_rows(dense_log, summary, seed, model, setting, variant), TRAJECTORY_FIELDS)
    _write_csv(out_dir / "agent_state.csv", _agent_state_rows(dense_log, seed, model, setting, variant), AGENT_STATE_FIELDS)

    native_actions = source_dir / "actions.jsonl"
    if native_actions.exists():
        shutil.copyfile(native_actions, out_dir / "actions.jsonl")
    else:
        _append_jsonl(out_dir / "actions.jsonl", _legacy_action_rows(dense_log, seed, model, setting, variant))

    for name in ["api_errors.jsonl", "memory_retrieval.jsonl"]:
        native = source_dir / name
        if native.exists():
            shutil.copyfile(native, out_dir / name)
        else:
            _append_jsonl(out_dir / name, [])

    native_semantic = source_dir / "semantic_rules.jsonl"
    if native_semantic.exists():
        shutil.copyfile(native_semantic, out_dir / "semantic_rules.jsonl")
    else:
        _append_jsonl(out_dir / "semantic_rules.jsonl", _semantic_rule_rows(memory_systems, seed, model))

    _write_csv(out_dir / "event_log.csv", summary.get("event_log", []), EVENT_FIELDS)
    _append_jsonl(out_dir / "case_study.jsonl", [])

    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(out_dir),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "dense_log_present": bool(dense_log),
        "memory_present": bool(memory_systems),
    }
    with (out_dir / "export_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    return out_dir


def _expand_sources(patterns: List[str]) -> List[Path]:
    sources: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            sources.extend(Path(match) for match in matches)
        else:
            sources.append(Path(pattern))
    return sorted(set(path for path in sources if path.is_dir()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", help="Run directory or glob containing summary.json")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--model")
    parser.add_argument("--setting")
    parser.add_argument("--variant")
    parser.add_argument("--seed")
    parser.add_argument("--api-access-date", default="")
    parser.add_argument("--code-commit", default="")
    parser.add_argument("--prompt-version", default="v1_default")
    parser.add_argument("--reflection-prompt-version", default="v1_default")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", default="")
    parser.add_argument("--episodic-capacity", type=int, default=24)
    parser.add_argument("--semantic-capacity", type=int, default=10)
    parser.add_argument("--retrieval-k", type=int, default=5)
    parser.add_argument("--rule-budget-m", type=int, default=3)
    parser.add_argument("--reflection-period-months", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exported = []
    for source in _expand_sources(args.sources):
        exported.append(export_one(source, args))
    for out_dir in exported:
        print(out_dir)


if __name__ == "__main__":
    main()
