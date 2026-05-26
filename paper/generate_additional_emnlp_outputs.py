#!/usr/bin/env python3
"""Generate additional EMNLP tables and figures from completed E2--E5 runs.

This script does not read raw API data or start experiments.  It only consumes
normalized run exports under ``runs/`` and writes publication helper artifacts
under ``figs/emnlp/``, ``paper/generated_tables/``, and
``paper/generated_tex/``.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FIGS = ROOT / "figs" / "emnlp"
TABLES = ROOT / "paper" / "generated_tables"
TEX = ROOT / "paper" / "generated_tex"

FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
TEX.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 160,
})


COLORS = {
    "default": "#1f77b4",
    "without-sentiment": "#ff7f0e",
    "without-reflection": "#d62728",
    "without-semantic-memory": "#2ca02c",
    "without-episodic-memory": "#9467bd",
    "numeric-only": "#7f7f7f",
    "text-event": "#1f77b4",
    "shuffled-event": "#d62728",
    "random-event": "#2ca02c",
    "default-prompt": "#1f77b4",
    "concise-prompt": "#17becf",
    "risk-neutral-wording": "#9467bd",
    "no-fairness-wording": "#2ca02c",
    "temperature-0.0": "#ff7f0e",
    "temperature-0.7": "#d62728",
    "natural-language-plus-json": "#8c564b",
}


def read_all_metrics() -> pd.DataFrame:
    paths = sorted(RUNS.glob("**/seed_*/metrics_summary.csv"))
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = str(path.relative_to(ROOT))
        rows.append(df)
    if not rows:
        raise RuntimeError("No metrics_summary.csv files found under runs/")
    allm = pd.concat(rows, ignore_index=True)
    allm.to_csv(RUNS / "all_metrics_summary.csv", index=False)
    return allm


def summarize(df: pd.DataFrame, exp_id: str) -> pd.DataFrame:
    sub = df[df["exp_id"] == exp_id].copy()
    if sub.empty:
        return pd.DataFrame()
    metrics = [
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
        "total_cost_usd",
        "wall_time_min",
    ]
    rows = []
    for keys, g in sub.groupby(["exp_id", "model", "setting", "variant"], dropna=False):
        row = dict(zip(["exp_id", "model", "setting", "variant"], keys))
        row["n_runs"] = len(g)
        row["n_seeds"] = g["seed"].nunique()
        row["seeds"] = ",".join(sorted(str(x) for x in g["seed"].dropna().unique()))
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            if vals.empty:
                row[f"{metric}_mean"] = math.nan
                row[f"{metric}_std"] = math.nan
                row[f"{metric}_ci95"] = math.nan
                row[f"{metric}_n"] = 0
            else:
                std = vals.std(ddof=1) if len(vals) > 1 else 0.0
                row[f"{metric}_mean"] = vals.mean()
                row[f"{metric}_std"] = std
                row[f"{metric}_ci95"] = 1.96 * std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
                row[f"{metric}_n"] = len(vals)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["exp_id", "model", "setting", "variant"])


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"[fig] {FIGS / (name + '.pdf')}")


def mean_ci(values: Iterable[float]) -> tuple[float, float]:
    series = pd.Series(values, dtype="float64").dropna()
    if series.empty:
        return math.nan, 0.0
    if len(series) == 1:
        return float(series.mean()), 0.0
    return float(series.mean()), float(1.96 * series.std(ddof=1) / math.sqrt(len(series)))


def plot_grouped_summary(summary: pd.DataFrame, exp_id: str, order: list[str], labels: dict[str, str], name: str, title: str) -> None:
    df = summary[summary["variant"].isin(order)].copy()
    df["order"] = df["variant"].map({v: i for i, v in enumerate(order)})
    df = df.sort_values("order")

    panels = [
        ("avg_wealth_final", "Final wealth (k)", 1 / 1000, 1),
        ("gini_final", "Final Gini", 1, 3),
        ("inflation_dev_abs_pct", "Inflation dev. (%)", 1, 1),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    x = range(len(df))
    tick_labels = [labels.get(v, v) for v in df["variant"]]
    for ax, (metric, ylabel, scale, digits) in zip(axes, panels):
        means = pd.to_numeric(df[f"{metric}_mean"], errors="coerce") * scale
        stds = pd.to_numeric(df[f"{metric}_std"], errors="coerce").fillna(0.0) * scale
        colors = [COLORS.get(v, "#4c78a8") for v in df["variant"]]
        ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.45)
        for i, value in enumerate(means):
            if not math.isnan(value):
                ax.text(i, value + max(means.max(), 1.0) * 0.025, f"{value:.{digits}f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(x))
        ax.set_xticklabels(tick_labels, rotation=28, ha="right", rotation_mode="anchor")
        ax.margins(y=0.2)
    fig.suptitle(title, y=0.98)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.30, top=0.82, wspace=0.34)
    save_figure(fig, name)


def load_trajectories(exp_id: str) -> pd.DataFrame:
    rows = []
    for path in sorted((RUNS / exp_id).glob("**/trajectory.csv")):
        df = pd.read_csv(path)
        df["run_dir"] = str(path.parent.relative_to(ROOT))
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_e2_trajectory_diagnostics() -> None:
    traj = load_trajectories("E2")
    order = ["default", "without-sentiment", "without-reflection", "without-semantic-memory", "without-episodic-memory"]
    labels = {
        "default": "Full",
        "without-sentiment": "w/o sentiment",
        "without-reflection": "w/o reflection",
        "without-semantic-memory": "w/o semantic",
        "without-episodic-memory": "w/o episodic",
    }
    panels = [
        ("avg_wealth", "Avg wealth (k)", 1 / 1000),
        ("gini", "Gini", 1),
        ("unemployment_pct", "Unemployment (%)", 1),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7), sharex=True)
    for ax, (metric, ylabel, scale) in zip(axes, panels):
        for variant in order:
            sub = traj[traj["variant"] == variant]
            grouped = []
            for month, g in sub.groupby("month"):
                mean, ci = mean_ci(pd.to_numeric(g[metric], errors="coerce") * scale)
                grouped.append((month, mean, ci))
            if not grouped:
                continue
            plot_df = pd.DataFrame(grouped, columns=["month", "mean", "ci"]).sort_values("month")
            ax.plot(plot_df["month"], plot_df["mean"], label=labels[variant], color=COLORS.get(variant), linewidth=1.7)
            ax.fill_between(
                plot_df["month"].to_numpy(dtype=float),
                (plot_df["mean"] - plot_df["ci"]).to_numpy(dtype=float),
                (plot_df["mean"] + plot_df["ci"]).to_numpy(dtype=float),
                color=COLORS.get(variant),
                alpha=0.12,
                linewidth=0,
            )
        ax.set_xlabel("Month")
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="upper left", frameon=False, ncol=1)
    fig.suptitle("E2 component-ablation trajectories (mean +/- 95% CI across five seeds)", y=0.99)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.18, top=0.82, wspace=0.28)
    save_figure(fig, "fig_e2_trajectory_diagnostics")


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def write_cost_error_tables(allm: pd.DataFrame) -> None:
    sub = allm[allm["exp_id"].isin(["E2", "E3", "E4", "E5"])].copy()
    rows = []
    for keys, g in sub.groupby(["exp_id", "variant"], dropna=False):
        row = dict(zip(["exp_id", "variant"], keys))
        row["n_runs"] = len(g)
        for col in ["total_cost_usd", "wall_time_min", "invalid_action_rate", "api_error_rate"]:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"{col}_mean"] = vals.mean() if not vals.empty else math.nan
            row[f"{col}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["exp_id", "variant"])
    out.to_csv(TABLES / "appendix_cost_error_summary.csv", index=False)
    print(f"[table] {TABLES / 'appendix_cost_error_summary.csv'}")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Exp. & Variant & Runs & Cost/run (\$) & Wall time (min) & Invalid / API err. \\",
        r"\midrule",
    ]
    for _, row in out.iterrows():
        invalid = row["invalid_action_rate_mean"]
        api = row["api_error_rate_mean"]
        lines.append(
            f"{latex_escape(row['exp_id'])} & {latex_escape(row['variant'])} & "
            f"{int(row['n_runs'])} & {row['total_cost_usd_mean']:.3f} & "
            f"{row['wall_time_min_mean']:.2f} & {invalid:.4f} / {api:.4f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Cost, runtime, and interface-compliance summary for completed E2--E5 GPT-5.2 runs.}",
        r"\label{tab:appendix_cost_error}",
        r"\end{table*}",
    ]
    (TEX / "table_appendix_cost_error.tex").write_text("\n".join(lines) + "\n")
    print(f"[tex] {TEX / 'table_appendix_cost_error.tex'}")


def write_event_fixture_examples() -> None:
    rows = []
    variants = ["numeric-only", "text-event", "shuffled-event", "random-event"]
    for variant in variants:
        paths = sorted((RUNS / "E4" / "GPT-5.2" / "finevo" / variant).glob("seed_*/event_log.csv"))
        if not paths:
            continue
        df = pd.read_csv(paths[0])
        if variant == "numeric-only":
            rows.append({
                "variant": variant,
                "month": 0,
                "event_text": "(numeric sentiment only; no text event)",
                "sentiment_label": "none",
                "v_t": 0.0,
                "shuffled_from_month": "",
            })
            continue
        for _, row in df.head(3).iterrows():
            rows.append({
                "variant": variant,
                "month": int(row["month"]),
                "event_text": row.get("event_text", ""),
                "sentiment_label": row.get("sentiment_label", ""),
                "v_t": row.get("v_t", ""),
                "shuffled_from_month": row.get("shuffled_from_month", ""),
            })
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "E4_event_fixture_examples.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[table] {TABLES / 'E4_event_fixture_examples.csv'}")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llp{0.48\textwidth}rr}",
        r"\toprule",
        r"Variant & Label & Example event text & $v_t$ & Shuffled from \\",
        r"\midrule",
    ]
    for _, row in out.iterrows():
        shuffled = row["shuffled_from_month"]
        if pd.isna(shuffled):
            shuffled = ""
        lines.append(
            f"{latex_escape(row['variant'])} & {latex_escape(row['sentiment_label'])} & "
            f"{latex_escape(row['event_text'])} & {row['v_t']} & {latex_escape(str(shuffled))} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Examples from the synthetic event fixture used in E4.  The shuffled-event condition preserves event text but breaks month-level alignment.}",
        r"\label{tab:e4_event_fixture_examples}",
        r"\end{table*}",
    ]
    (TEX / "table_e4_event_fixture_examples.tex").write_text("\n".join(lines) + "\n")
    print(f"[tex] {TEX / 'table_e4_event_fixture_examples.tex'}")


def main() -> None:
    allm = read_all_metrics()
    summaries = {exp: summarize(allm, exp) for exp in ["E2", "E3", "E4", "E5"]}
    for exp, summary in summaries.items():
        summary.to_csv(TABLES / f"{exp}_summary.csv", index=False)

    plot_grouped_summary(
        summaries["E4"],
        "E4",
        ["numeric-only", "text-event", "shuffled-event", "random-event"],
        {
            "numeric-only": "Numeric only",
            "text-event": "Text event",
            "shuffled-event": "Shuffled",
            "random-event": "Random",
        },
        "fig_text_event_controls",
        "E4 textual event-channel controls",
    )
    plot_grouped_summary(
        summaries["E5"],
        "E5",
        [
            "default-prompt",
            "concise-prompt",
            "risk-neutral-wording",
            "no-fairness-wording",
            "temperature-0.0",
            "temperature-0.7",
            "natural-language-plus-json",
        ],
        {
            "default-prompt": "Default",
            "concise-prompt": "Concise",
            "risk-neutral-wording": "Risk-neutral",
            "no-fairness-wording": "No fairness",
            "temperature-0.0": "Temp 0.0",
            "temperature-0.7": "Temp 0.7",
            "natural-language-plus-json": "NL + JSON",
        },
        "fig_prompt_robustness",
        "E5 prompt and decoding robustness",
    )
    plot_e2_trajectory_diagnostics()
    write_cost_error_tables(allm)
    write_event_fixture_examples()
    print("[done] additional EMNLP outputs generated")


if __name__ == "__main__":
    main()
