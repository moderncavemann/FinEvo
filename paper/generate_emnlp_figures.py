#!/usr/bin/env python3
"""Generate EMNLP revision figures from normalized runs.

Primary input:
    runs/<exp_id>/<model>/<setting>/<variant>/seed_<seed>/

For continuity while the five-seed EMNLP runs are still pending, the script can
also read legacy rebuttal folders under ``data/``.  Legacy plots are useful for
drafting but should be treated as preliminary if they only contain seeds 0/1/2
or single-run large-scale results.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import pickle
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 160,
})


E2_LEGACY = {
    "GPT_Full": ("finevo", "default"),
    "GPT_noSent": ("finevo", "without-sentiment"),
    "GPT_noRefl": ("finevo", "without-reflection"),
    "GPT_noSem": ("finevo", "without-semantic-memory"),
    "GPT_noEp": ("finevo", "without-episodic-memory"),
    "GPT_SentOnly": ("finevo", "sentiment-only"),
    "GPT_Baseline": ("text-only", "baseline"),
}

E3_LEGACY = {
    "GPT_sB0_default": "default",
    "GPT_sB_alpha_hi": "alpha-plus-50",
    "GPT_sB_alpha_lo": "alpha-minus-50",
    "GPT_sB_beta_hi": "beta-plus-50",
    "GPT_sB_beta_lo": "beta-minus-50",
    "GPT_sB_gamma_hi": "gamma-plus-50",
    "GPT_sB_gamma_lo": "gamma-minus-50",
    "GPT_sB_delta_hi": "delta-plus-50",
    "GPT_sB_delta_lo": "delta-minus-50",
    "GPT_sB_no_macro": "no-macro-terms",
    "GPT_sB_random_S": "random-sentiment",
}

E1_LEGACY = [
    ("GPT-5.2", "text-only", "baseline", "openai-gpt-5.2-baseline-100agents-240months", "legacy-single"),
    ("GPT-5.2", "finevo", "default", "openai-gpt-5.2-gap_fixed-100agents-240months", "legacy-single"),
]

LABELS = {
    ("GPT-5.2", "text-only", "baseline"): "GPT-5.2 baseline",
    ("GPT-5.2", "finevo", "default"): "GPT-5.2 FinEvo",
}

COLORS = {
    "baseline": "#8e8e8e",
    "default": "#1f77b4",
    "without-sentiment": "#ff7f0e",
    "without-reflection": "#d62728",
    "without-semantic-memory": "#2ca02c",
    "without-episodic-memory": "#9467bd",
    "sentiment-only": "#17becf",
    "random-sentiment": "#d62728",
    "no-macro-terms": "#ff7f0e",
}

HISTORICAL_FIGURE_LABEL = (
    "HISTORICAL PRE-P0 V1 EVIDENCE ONLY - not current-method scientific evidence"
)


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def gini(values: Iterable[Any]) -> float:
    xs = sorted(max(as_float(value, 0.0), 0.0) for value in values)
    total = sum(xs)
    n = len(xs)
    if n == 0 or total == 0:
        return 0.0
    weighted = sum((i + 1) * value for i, value in enumerate(xs))
    return 2 * weighted / (n * total) - (n + 1) / n


def mean_ci(values: Iterable[Any]) -> Tuple[float, float, float, int]:
    xs = [as_float(value) for value in values]
    xs = [value for value in xs if not math.isnan(value)]
    if not xs:
        return math.nan, 0.0, 0.0, 0
    mean = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return mean, sd, ci, len(xs)


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.10))
    fig.text(
        0.5,
        0.012,
        HISTORICAL_FIGURE_LABEL,
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="#8b1e1e",
    )
    fig.savefig(
        out_dir / f"{name}.pdf",
        bbox_inches="tight",
        metadata={
            "Subject": "HISTORICAL PRE-P0 V1 EVIDENCE ONLY",
            "CreationDate": None,
        },
    )
    fig.savefig(
        out_dir / f"{name}.png",
        bbox_inches="tight",
        dpi=180,
        metadata={"Description": "HISTORICAL PRE-P0 V1 EVIDENCE ONLY"},
    )
    plt.close(fig)
    print(f"[fig] {out_dir}/{name}.pdf")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_runs(runs_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    data = {"metrics": [], "trajectory": [], "agent_state": [], "semantic_rules": []}
    for metrics_path in sorted(runs_root.glob("*/*/*/*/seed_*/metrics_summary.csv")):
        run_dir = metrics_path.parent
        for row in read_csv(metrics_path):
            row["source"] = "runs"
            row["run_dir"] = str(run_dir)
            data["metrics"].append(row)
        for row in read_csv(run_dir / "trajectory.csv") if (run_dir / "trajectory.csv").exists() else []:
            row["source"] = "runs"
            row["run_dir"] = str(run_dir)
            data["trajectory"].append(row)
        for row in read_csv(run_dir / "agent_state.csv") if (run_dir / "agent_state.csv").exists() else []:
            row["source"] = "runs"
            row["run_dir"] = str(run_dir)
            data["agent_state"].append(row)
        for row in load_jsonl(run_dir / "semantic_rules.jsonl"):
            row["source"] = "runs"
            row["run_dir"] = str(run_dir)
            data["semantic_rules"].append(row)
    return data


def load_summary(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def legacy_summary_metrics(data_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for tag, (setting, variant) in E2_LEGACY.items():
        for path in sorted(data_root.glob(f"openai-gpt-5.2-{tag}-seed*-10agents-24months/summary.json")):
            summary = load_summary(path)
            metrics = summary.get("final_metrics", {})
            rows.append({
                "source": "legacy",
                "exp_id": "E2",
                "model": "GPT-5.2",
                "setting": setting,
                "variant": variant,
                "seed": summary.get("random_seed", ""),
                "num_agents": summary.get("num_agents", 10),
                "num_months": summary.get("episode_length", 24),
                "avg_wealth_final": metrics.get("avg_wealth", ""),
                "median_wealth_final": metrics.get("median_wealth", ""),
                "gini_final": metrics.get("gini", ""),
                "unemployment_final_pct": as_float(metrics.get("avg_unemployment"), 0.0) * 100,
                "inflation_final_pct": as_float(metrics.get("avg_inflation"), 0.0) * 100,
                "inflation_dev_abs_pct": abs(as_float(metrics.get("avg_inflation"), 0.0) * 100 - 2.0),
                "invalid_action_rate": summary.get("error_rate", ""),
            })

    for tag, variant in E3_LEGACY.items():
        for path in sorted(data_root.glob(f"openai-gpt-5.2-{tag}-seed*-10agents-24months/summary.json")):
            summary = load_summary(path)
            metrics = summary.get("final_metrics", {})
            rows.append({
                "source": "legacy",
                "exp_id": "E3",
                "model": "GPT-5.2",
                "setting": "finevo",
                "variant": variant,
                "seed": summary.get("random_seed", ""),
                "num_agents": summary.get("num_agents", 10),
                "num_months": summary.get("episode_length", 24),
                "avg_wealth_final": metrics.get("avg_wealth", ""),
                "median_wealth_final": metrics.get("median_wealth", ""),
                "gini_final": metrics.get("gini", ""),
                "unemployment_final_pct": as_float(metrics.get("avg_unemployment"), 0.0) * 100,
                "inflation_final_pct": as_float(metrics.get("avg_inflation"), 0.0) * 100,
                "inflation_dev_abs_pct": abs(as_float(metrics.get("avg_inflation"), 0.0) * 100 - 2.0),
                "invalid_action_rate": summary.get("error_rate", ""),
            })

    return rows


def load_dense_log(path: Path) -> Dict[str, Any]:
    dense_path = path / "dense_log.pkl"
    with dense_path.open("rb") as f:
        return pickle.load(f)


def state_wealths(state: Dict[str, Any]) -> List[float]:
    return [
        as_float(agent_state.get("inventory", {}).get("Coin"), 0.0)
        for agent_id, agent_state in state.items()
        if agent_id != "p" and isinstance(agent_state, dict)
    ]


def legacy_trajectory_and_agents(data_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    trajectory: List[Dict[str, Any]] = []
    agent_state: List[Dict[str, Any]] = []

    for model, setting, variant, folder, seed in E1_LEGACY:
        run_dir = data_root / folder
        if not (run_dir / "dense_log.pkl").exists():
            continue
        summary = load_summary(run_dir / "summary.json")
        sentiment = summary.get("final_metrics", {}).get("sentiment_history", [])
        dense = load_dense_log(run_dir)
        states = dense.get("states", [])
        worlds = dense.get("world", [])

        for month, world in enumerate(worlds):
            state = states[month] if month < len(states) else {}
            wealths = state_wealths(state)
            infl_pct = as_float(world.get("Price Inflation"), 0.0) * 100
            gdp_growth = as_float(world.get("Real GDP Growth", world.get("Nominal GDP Growth", 0.0))) * 100
            trajectory.append({
                "source": "legacy",
                "exp_id": "E1",
                "model": model,
                "setting": setting,
                "variant": variant,
                "seed": seed,
                "month": month,
                "avg_wealth": statistics.mean(wealths) if wealths else 0.0,
                "median_wealth": statistics.median(wealths) if wealths else 0.0,
                "gini": gini(wealths),
                "unemployment_pct": as_float(world.get("Unemployment Rate"), 0.0) * 100,
                "inflation_pct": infl_pct,
                "inflation_dev_abs_pct": abs(infl_pct - 2.0),
                "gdp": as_float(world.get("Real GDP", world.get("Nominal GDP", 0.0))),
                "gdp_growth_pct": gdp_growth,
                "price_level": as_float(world.get("Price"), 0.0),
                "global_sentiment": sentiment[month] if month < len(sentiment) else 0.0,
            })

        if states:
            final_month = len(states) - 1
            for agent_id, agent in states[-1].items():
                if agent_id == "p" or not isinstance(agent, dict):
                    continue
                agent_state.append({
                    "source": "legacy",
                    "exp_id": "E1",
                    "model": model,
                    "setting": setting,
                    "variant": variant,
                    "seed": seed,
                    "month": final_month,
                    "agent_id": agent_id,
                    "wealth": agent.get("inventory", {}).get("Coin", 0.0),
                })

    return trajectory, agent_state


def add_legacy(data: Dict[str, List[Dict[str, Any]]], data_root: Path) -> None:
    data["metrics"].extend(legacy_summary_metrics(data_root))
    trajectory, agent_state = legacy_trajectory_and_agents(data_root)
    data["trajectory"].extend(trajectory)
    data["agent_state"].extend(agent_state)


def group_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return str(row.get("model", "")), str(row.get("setting", "")), str(row.get("variant", ""))


def group_label(key: Tuple[str, str, str]) -> str:
    return LABELS.get(key, " ".join(part for part in key if part))


def filter_rows(rows: List[Dict[str, Any]], exp_id: str) -> List[Dict[str, Any]]:
    return [row for row in rows if str(row.get("exp_id")) == exp_id]


def plot_component_ablation(data: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> bool:
    rows = [row for row in filter_rows(data["metrics"], "E2") if row.get("model") == "GPT-5.2"]
    if not rows:
        print("[skip] component ablation: no E2 metrics")
        return False

    order = [
        "baseline",
        "sentiment-only",
        "without-episodic-memory",
        "without-semantic-memory",
        "without-reflection",
        "without-sentiment",
        "default",
    ]
    labels = {
        "baseline": "Text-only",
        "sentiment-only": "Sentiment\nonly",
        "without-episodic-memory": "w/o\nEpisodic",
        "without-semantic-memory": "w/o\nSemantic",
        "without-reflection": "w/o\nReflection",
        "without-sentiment": "w/o\nSentiment",
        "default": "Full\nFinEvo",
    }
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    variants = [variant for variant in order if variant in by_variant]

    panels = [
        ("avg_wealth_final", "Avg Wealth (k)", 1 / 1000, "wealth"),
        ("gini_final", "Gini", 1, "gini"),
        ("unemployment_final_pct", "Unemployment (%)", 1, "unemp"),
        ("inflation_dev_abs_pct", "|Infl. - 2%|", 1, "infl"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3))
    x = np.arange(len(variants))
    for ax, (field, ylabel, scale, _) in zip(axes, panels):
        means, cis, ns = [], [], []
        for variant in variants:
            mean, sd, ci, n = mean_ci(as_float(row.get(field)) * scale for row in by_variant[variant])
            means.append(mean)
            cis.append(ci)
            ns.append(n)
        colors = [COLORS.get(variant, "#666666") for variant in variants]
        ax.bar(x, means, yerr=cis, capsize=3, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(v, v) for v in variants], rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.margins(y=0.18)
        for xi, mean, n in zip(x, means, ns):
            if not math.isnan(mean):
                ax.text(xi, mean, f"n={n}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Component Ablation (GPT-5.2, 10 agents / 24 months)", y=1.05)
    fig.tight_layout()
    save(fig, out_dir, "fig_component_ablation")
    return True


def plot_sentiment_robustness(data: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> bool:
    rows = [row for row in filter_rows(data["metrics"], "E3") if row.get("model") == "GPT-5.2"]
    if not rows:
        print("[skip] sentiment robustness: no E3 metrics")
        return False

    order = [
        "default",
        "alpha-plus-50",
        "alpha-minus-50",
        "beta-plus-50",
        "beta-minus-50",
        "gamma-plus-50",
        "gamma-minus-50",
        "delta-plus-50",
        "delta-minus-50",
        "no-macro-terms",
        "random-sentiment",
    ]
    labels = {
        "default": "Default",
        "alpha-plus-50": "alpha +50%",
        "alpha-minus-50": "alpha -50%",
        "beta-plus-50": "beta +50%",
        "beta-minus-50": "beta -50%",
        "gamma-plus-50": "gamma +50%",
        "gamma-minus-50": "gamma -50%",
        "delta-plus-50": "delta +50%",
        "delta-minus-50": "delta -50%",
        "no-macro-terms": "No macro",
        "random-sentiment": "Random S",
    }
    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    variants = [variant for variant in order if variant in by_variant]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    y = np.arange(len(variants))[::-1]

    for ax, field, xlabel, scale in [
        (axes[0], "avg_wealth_final", "Avg wealth (k)", 1 / 1000),
        (axes[1], "gini_final", "Gini", 1),
    ]:
        means, cis, colors = [], [], []
        for variant in variants:
            mean, sd, ci, _ = mean_ci(as_float(row.get(field)) * scale for row in by_variant[variant])
            means.append(mean)
            cis.append(ci)
            colors.append(COLORS.get(variant, "#1f77b4"))
        ax.errorbar(means, y, xerr=cis, fmt="o", color="#555555", ecolor="#777777", capsize=3)
        for mean, yi, color in zip(means, y, colors):
            ax.plot(mean, yi, "o", color=color, markeredgecolor="black", markeredgewidth=0.4)
        if "default" in by_variant:
            default_mean, _, _, _ = mean_ci(as_float(row.get(field)) * scale for row in by_variant["default"])
            ax.axvline(default_mean, color="#1f77b4", linestyle=":", linewidth=1.2)
            ax.text(default_mean, y[0] + 0.35, "default", color="#1f77b4", fontsize=7, ha="center")
        ax.set_xlabel(xlabel)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([labels.get(variant, variant) for variant in variants])
    axes[1].set_yticks(y)
    axes[1].tick_params(labelleft=False)
    fig.suptitle("Sentiment Parameter Robustness (95% CI)", y=1.02)
    fig.tight_layout()
    save(fig, out_dir, "fig3_sentiment_robustness")
    return True


def plot_trajectory_ribbons(data: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> bool:
    rows = filter_rows(data["trajectory"], "E1")
    if not rows:
        print("[skip] trajectory ribbons: no E1 trajectory rows")
        return False

    grouped: Dict[Tuple[str, str, str], Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[group_key(row)][str(row.get("seed", ""))].append(row)

    panels = [
        ("avg_wealth", "Avg Wealth", "coin"),
        ("unemployment_pct", "Unemployment", "%"),
        ("inflation_dev_abs_pct", "|Inflation - 2%|", "pp"),
        ("gini", "Gini", ""),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8))
    axes = axes.ravel()

    for ax, (field, title, ylabel) in zip(axes, panels):
        for key, by_seed in sorted(grouped.items()):
            seed_series = []
            for seed_rows in by_seed.values():
                seed_rows = sorted(seed_rows, key=lambda row: int(as_float(row.get("month"), 0)))
                seed_series.append(np.array([as_float(row.get(field), 0.0) for row in seed_rows], dtype=float))
            if not seed_series:
                continue
            min_len = min(len(series) for series in seed_series)
            matrix = np.vstack([series[:min_len] for series in seed_series])
            x = np.arange(min_len)
            mean = matrix.mean(axis=0)
            sd = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros(min_len)
            ci = 1.96 * sd / math.sqrt(matrix.shape[0]) if matrix.shape[0] > 1 else np.zeros(min_len)
            color = COLORS.get(key[2], "#1f77b4")
            linestyle = "--" if key[1] == "text-only" else "-"
            ax.plot(x, mean, label=f"{group_label(key)} (n={matrix.shape[0]})", color=color, linestyle=linestyle, linewidth=1.6)
            if matrix.shape[0] > 1:
                ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.14, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Month")
        ax.set_ylabel(ylabel)
        if field == "gini":
            ax.set_ylim(0, 1)
    axes[0].legend(loc="best", frameon=False)
    fig.suptitle("Trajectory Diagnostics with Seed Ribbons", y=1.02)
    fig.tight_layout()
    save(fig, out_dir, "fig2_trajectory_ribbons")
    return True


def lorenz_curve(wealths: List[float], grid: np.ndarray) -> np.ndarray:
    xs = np.sort(np.maximum(np.array(wealths, dtype=float), 0.0))
    if xs.size == 0 or xs.sum() == 0:
        return np.zeros_like(grid)
    cum_pop = np.linspace(0, 1, xs.size + 1)
    cum_wealth = np.concatenate([[0.0], np.cumsum(xs) / xs.sum()])
    return np.interp(grid, cum_pop, cum_wealth)


def plot_lorenz(data: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> bool:
    rows = filter_rows(data["agent_state"], "E1")
    if not rows:
        print("[skip] Lorenz: no E1 agent_state rows")
        return False

    grouped = defaultdict(lambda: defaultdict(list))
    max_month_by_group_seed = defaultdict(int)
    for row in rows:
        key = group_key(row)
        seed = str(row.get("seed", ""))
        month = int(as_float(row.get("month"), 0))
        max_month_by_group_seed[(key, seed)] = max(max_month_by_group_seed[(key, seed)], month)

    for row in rows:
        key = group_key(row)
        seed = str(row.get("seed", ""))
        if int(as_float(row.get("month"), 0)) == max_month_by_group_seed[(key, seed)]:
            grouped[key][seed].append(as_float(row.get("wealth"), 0.0))

    grid = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="Equality")
    for key, by_seed in sorted(grouped.items()):
        curves = np.vstack([lorenz_curve(wealths, grid) for wealths in by_seed.values() if wealths])
        if curves.size == 0:
            continue
        mean = curves.mean(axis=0)
        sd = curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean)
        ci = 1.96 * sd / math.sqrt(curves.shape[0]) if curves.shape[0] > 1 else np.zeros_like(mean)
        all_gini = [gini(wealths) for wealths in by_seed.values() if wealths]
        color = COLORS.get(key[2], "#1f77b4")
        linestyle = "--" if key[1] == "text-only" else "-"
        ax.plot(grid, mean, color=color, linestyle=linestyle, linewidth=1.7, label=f"{group_label(key)}; Gini={statistics.mean(all_gini):.3f}")
        if curves.shape[0] > 1:
            ax.fill_between(grid, mean - ci, mean + ci, color=color, alpha=0.14, linewidth=0)
    ax.set_xlabel("Cumulative share of agents")
    ax.set_ylabel("Cumulative share of wealth")
    ax.set_title("Final-Month Lorenz Curves")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, out_dir, "fig_lorenz")
    return True


def plot_rule_survival(data: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> bool:
    rows = [row for row in data["semantic_rules"] if row.get("month") not in ("", None)]
    if not rows:
        print("[skip] rule survival: no semantic_rules rows with months")
        return False

    by_regime_month = defaultdict(list)
    for row in rows:
        regime = str(row.get("regime") or "unknown")
        month = int(as_float(row.get("month"), 0))
        by_regime_month[(regime, month)].append(as_float(row.get("confidence"), math.nan))

    regimes = sorted(set(regime for regime, month in by_regime_month))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for regime in regimes:
        months = sorted(month for r, month in by_regime_month if r == regime)
        values = []
        for month in months:
            mean, _, _, _ = mean_ci(by_regime_month[(regime, month)])
            values.append(mean)
        ax.plot(months, values, marker="o", linewidth=1.5, label=regime)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean rule confidence")
    ax.set_title("Semantic Rule Confidence by Regime")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out_dir, "fig_rule_survival")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="figs/emnlp")
    parser.add_argument("--no-legacy-fallback", action="store_true")
    parser.add_argument(
        "--figures",
        default="all",
        help="Comma-separated: component,sentiment,trajectory,lorenz,rule or all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_runs(Path(args.runs_root))
    if not args.no_legacy_fallback:
        add_legacy(data, Path(args.data_root))

    print(
        f"[load] metrics={len(data['metrics'])}, trajectory={len(data['trajectory'])}, "
        f"agent_state={len(data['agent_state'])}, semantic_rules={len(data['semantic_rules'])}"
    )

    requested = {item.strip() for item in args.figures.split(",") if item.strip()}
    if "all" in requested:
        requested = {"component", "sentiment", "trajectory", "lorenz", "rule"}

    out_dir = Path(args.out)
    made = []
    if "component" in requested and plot_component_ablation(data, out_dir):
        made.append("component")
    if "sentiment" in requested and plot_sentiment_robustness(data, out_dir):
        made.append("sentiment")
    if "trajectory" in requested and plot_trajectory_ribbons(data, out_dir):
        made.append("trajectory")
    if "lorenz" in requested and plot_lorenz(data, out_dir):
        made.append("lorenz")
    if "rule" in requested and plot_rule_survival(data, out_dir):
        made.append("rule")

    print(f"[done] generated: {', '.join(made) if made else 'none'}")


if __name__ == "__main__":
    main()
