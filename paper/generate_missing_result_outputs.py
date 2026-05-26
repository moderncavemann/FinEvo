#!/usr/bin/env python3
"""Generate integration artifacts for the remaining FinEvo result gaps.

This post-processing script does not start API runs. It consumes normalized
exports under ``runs/`` and writes compact CSV/LaTeX/figure artifacts for:

* the large-scale cross-model diagnostic table;
* the GPT-5.2 large-scale significance row, when matched seeds exist;
* a log-grounded crash-state case-study trace.

Reader-facing LaTeX generated here avoids internal experiment IDs.
"""

from __future__ import annotations

import csv
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
TABLES = ROOT / "paper" / "generated_tables"
TEX = ROOT / "paper" / "generated_tex"
FIGS = ROOT / "figs" / "emnlp"

TABLES.mkdir(parents=True, exist_ok=True)
TEX.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

METRICS = [
    "avg_wealth_final",
    "gini_final",
    "unemployment_final_pct",
    "inflation_dev_abs_pct",
    "gdp_volatility_pct",
    "invalid_action_rate",
    "api_error_rate",
    "wall_time_min",
    "total_cost_usd",
]


def latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def short(value: Any, limit: int = 170) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def read_e1_metrics() -> pd.DataFrame:
    rows = []
    for path in sorted((RUNS / "E1").glob("*/*/*/seed_*/metrics_summary.csv")):
        df = pd.read_csv(path)
        df["source_file"] = str(path.relative_to(ROOT))
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def mean_std(values: pd.Series) -> tuple[float, float, int]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return math.nan, math.nan, 0
    return float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0), int(len(vals))


def format_mean_std(mean: float, std: float, scale: float = 1.0, digits: int = 2, suffix: str = "") -> str:
    if math.isnan(mean):
        return "TBD"
    return f"{mean * scale:.{digits}f} $\\pm$ {std * scale:.{digits}f}{suffix}"


def format_single(value: Any, scale: float = 1.0, digits: int = 2, suffix: str = "") -> str:
    try:
        val = float(value) * scale
    except (TypeError, ValueError):
        return "TBD"
    return f"{val:.{digits}f}{suffix}"


def write_cross_model_diagnostic() -> None:
    e1 = read_e1_metrics()
    out_csv = TABLES / "E1_diagnostic_summary.csv"
    if e1.empty:
        e1.to_csv(out_csv, index=False)
        return

    keep = [
        "model",
        "setting",
        "variant",
        "seed",
        "num_agents",
        "num_months",
        "avg_wealth_final",
        "gini_final",
        "unemployment_final_pct",
        "inflation_dev_abs_pct",
        "invalid_action_rate",
        "api_error_rate",
        "wall_time_min",
        "total_cost_usd",
        "source_file",
    ]
    e1[[c for c in keep if c in e1.columns]].sort_values(["model", "setting", "variant", "seed"]).to_csv(out_csv, index=False)
    print(f"[table] {out_csv}")

    paired_rows = []
    pending_rows = []
    for model, g in e1.groupby("model"):
        baseline = g[(g["setting"] == "text-only") & (g["variant"] == "baseline")]
        finevo = g[(g["setting"] == "finevo") & (g["variant"] == "default")]
        seeds = sorted(set(baseline["seed"].astype(str)) & set(finevo["seed"].astype(str)))
        if not seeds:
            pending_rows.append({"model": model, "baseline_rows": len(baseline), "finevo_rows": len(finevo), "status": "missing paired FinEvo or baseline"})
            continue
        for seed in seeds:
            brow = baseline[baseline["seed"].astype(str) == seed].iloc[0]
            frow = finevo[finevo["seed"].astype(str) == seed].iloc[0]
            paired_rows.append({"model": model, "seed": seed, "baseline": brow, "finevo": frow})

    pd.DataFrame(pending_rows).to_csv(TABLES / "E1_diagnostic_pending.csv", index=False)
    print(f"[table] {TABLES / 'E1_diagnostic_pending.csv'}")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Seed & Setting & Wealth (k) $\uparrow$ & Gini $\downarrow$ & Unemp. $\downarrow$ & Infl. dev. $\downarrow$ & Invalid/API err. \\",
        r"\midrule",
    ]
    for row in paired_rows:
        for label, src in [("Text-only", row["baseline"]), ("FinEvo", row["finevo"])]:
            lines.append(
                f"{latex_escape(row['model'])} & {latex_escape(row['seed'])} & {label} & "
                f"{format_single(src.get('avg_wealth_final'), 1/1000, 1)} & "
                f"{format_single(src.get('gini_final'), 1, 3)} & "
                f"{format_single(src.get('unemployment_final_pct'), 1, 2, r'\%')} & "
                f"{format_single(src.get('inflation_dev_abs_pct'), 1, 2, r'\%')} & "
                f"{format_single(src.get('invalid_action_rate'), 1, 4)} / {format_single(src.get('api_error_rate'), 1, 4)} \\\\"
            )
        lines.append(r"\addlinespace")
    if not paired_rows:
        lines.append(r"\multicolumn{8}{c}{No paired large-scale diagnostic rows are currently exported.} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Large-scale cross-model diagnostic runs. These seed-matched diagnostics are used for behavioral comparison and qualitative analysis, not for causal claims about model architecture.}",
        r"\label{tab:cross_model_diagnostic}",
        r"\end{table*}",
    ]
    out_tex = TEX / "table_cross_model_diagnostic.tex"
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[tex] {out_tex}")


def read_all_metrics() -> pd.DataFrame:
    rows = []
    for path in sorted(RUNS.glob("*/*/*/*/seed_*/metrics_summary.csv")):
        df = pd.read_csv(path)
        df["source_file"] = str(path.relative_to(ROOT))
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_group(df: pd.DataFrame, label: str, status: str = "complete") -> dict[str, Any]:
    row: dict[str, Any] = {"row_label": label, "status": status, "n_runs": len(df)}
    row["seeds"] = ",".join(sorted(str(x) for x in df["seed"].dropna().unique())) if "seed" in df else ""
    for metric in METRICS:
        if metric in df:
            mean, std, n = mean_std(df[metric])
        else:
            mean, std, n = math.nan, math.nan, 0
        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
        row[f"{metric}_n"] = n
    return row


def paired_pvalues(baseline: pd.DataFrame, finevo: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if baseline.empty or finevo.empty:
        return out
    base = baseline.copy()
    fin = finevo.copy()
    base["seed_key"] = base["seed"].astype(str)
    fin["seed_key"] = fin["seed"].astype(str)
    merged = base.merge(fin, on="seed_key", suffixes=("_baseline", "_finevo"))
    out["paired_seeds"] = ",".join(sorted(merged["seed_key"].unique()))
    for metric in ["avg_wealth_final", "gini_final", "unemployment_final_pct", "inflation_dev_abs_pct"]:
        bcol = f"{metric}_baseline"
        fcol = f"{metric}_finevo"
        if bcol not in merged or fcol not in merged or len(merged) < 2:
            out[f"{metric}_paired_p"] = math.nan
            continue
        bvals = pd.to_numeric(merged[bcol], errors="coerce")
        fvals = pd.to_numeric(merged[fcol], errors="coerce")
        valid = ~(bvals.isna() | fvals.isna())
        if valid.sum() < 2:
            out[f"{metric}_paired_p"] = math.nan
        else:
            out[f"{metric}_paired_p"] = float(stats.ttest_rel(fvals[valid], bvals[valid]).pvalue)
    return out


def write_main_significance_and_ablation() -> None:
    allm = read_all_metrics()
    rows: list[dict[str, Any]] = []
    if allm.empty:
        pd.DataFrame(rows).to_csv(TABLES / "main_significance_and_ablation.csv", index=False)
        return

    large = allm[
        (allm["model"] == "GPT-5.2")
        & (allm["num_agents"] == 100)
        & (allm["num_months"] == 240)
    ].copy()
    baseline = large[(large["setting"] == "text-only") & (large["variant"] == "baseline")]
    finevo = large[(large["setting"] == "finevo") & (large["variant"] == "default")]

    if baseline.empty:
        rows.append({"row_label": "GPT-5.2 large-scale text-only baseline", "status": "pending: no exported 100-agent/240-month baseline rows", "n_runs": 0})
    else:
        rows.append(summarize_group(baseline, "GPT-5.2 large-scale text-only baseline"))

    if finevo.empty:
        rows.append({"row_label": "GPT-5.2 large-scale FinEvo", "status": "pending: no exported 100-agent/240-month FinEvo rows", "n_runs": 0})
    else:
        rows.append(summarize_group(finevo, "GPT-5.2 large-scale FinEvo"))

    pvals = paired_pvalues(baseline, finevo)
    for row in rows:
        row.update(pvals)

    e2 = allm[(allm["exp_id"] == "E2") & (allm["model"] == "GPT-5.2")].copy()
    order = [
        "default",
        "without-sentiment",
        "without-reflection",
        "without-semantic-memory",
        "without-episodic-memory",
    ]
    labels = {
        "default": "Controlled component: full FinEvo",
        "without-sentiment": "Controlled component: w/o regime cue",
        "without-reflection": "Controlled component: w/o reflection",
        "without-semantic-memory": "Controlled component: w/o semantic memory",
        "without-episodic-memory": "Controlled component: w/o episodic memory",
    }
    for variant in order:
        sub = e2[e2["variant"] == variant]
        if not sub.empty:
            rows.append(summarize_group(sub, labels[variant]))

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "main_significance_and_ablation.csv", index=False)
    print(f"[table] {TABLES / 'main_significance_and_ablation.csv'}")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Row & Runs & Wealth (k) $\uparrow$ & Gini $\downarrow$ & Unemp. $\downarrow$ & Infl. dev. $\downarrow$ & Status \\",
        r"\midrule",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"{latex_escape(row.get('row_label', ''))} & {int(row.get('n_runs', 0) or 0)} & "
            f"{format_mean_std(row.get('avg_wealth_final_mean', math.nan), row.get('avg_wealth_final_std', 0.0), 1/1000, 1)} & "
            f"{format_mean_std(row.get('gini_final_mean', math.nan), row.get('gini_final_std', 0.0), 1, 3)} & "
            f"{format_mean_std(row.get('unemployment_final_pct_mean', math.nan), row.get('unemployment_final_pct_std', 0.0), 1, 2, r'\%')} & "
            f"{format_mean_std(row.get('inflation_dev_abs_pct_mean', math.nan), row.get('inflation_dev_abs_pct_std', 0.0), 1, 2, r'\%')} & "
            f"{latex_escape(row.get('status', 'complete'))} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Main large-scale significance row and controlled component rows. Large-scale p-values should be reported only after matched GPT-5.2 text-only and FinEvo seeds are exported.}",
        r"\label{tab:main_significance_and_ablation}",
        r"\end{table*}",
    ]
    out_tex = TEX / "table_main_significance_and_ablation.tex"
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[tex] {out_tex}")


def choose_case_month(traj: pd.DataFrame) -> int:
    sub = traj.copy()
    if "global_sentiment" in sub and pd.to_numeric(sub["global_sentiment"], errors="coerce").notna().any():
        sentiments = pd.to_numeric(sub["global_sentiment"], errors="coerce")
        return int(sub.loc[sentiments.idxmin(), "month"])
    if "unemployment_pct" in sub:
        unemp = pd.to_numeric(sub["unemployment_pct"], errors="coerce")
        jump = unemp.diff().fillna(0)
        return int(sub.loc[jump.idxmax(), "month"])
    return int(sub["month"].min())


def find_action(actions: list[dict[str, Any]], month: int) -> dict[str, Any]:
    same_month = [row for row in actions if int(row.get("month", -1)) == month]
    if not same_month:
        prior = [row for row in actions if int(row.get("month", -1)) <= month]
        same_month = prior[-100:] if prior else actions[:100]
    for row in same_month:
        if row.get("used_memory_ids") or row.get("rationale"):
            return row
    return same_month[0] if same_month else {}


def find_row(rows: list[dict[str, Any]], month: int, agent_id: Any) -> dict[str, Any]:
    exact = [row for row in rows if int(row.get("month", -1)) == month and str(row.get("agent_id")) == str(agent_id)]
    if exact:
        return exact[0]
    prior = [row for row in rows if int(row.get("month", -1)) <= month and str(row.get("agent_id")) == str(agent_id)]
    return prior[-1] if prior else {}


def code_case(text: str) -> list[str]:
    lowered = text.lower()
    codes = []
    if any(term in lowered for term in ["work", "labor", "income", "employment", "job"]):
        codes.append("labor_continuity")
    if any(term in lowered for term in ["save", "saving", "preserve", "reduce consumption", "liquidity"]):
        codes.append("liquidity_preservation")
    if any(term in lowered for term in ["inflation", "price", "interest"]):
        codes.append("inflation_response")
    if any(term in lowered for term in ["risk", "volatile", "uncertain"]):
        codes.append("risk_management")
    return codes or ["general_budget_adjustment"]


def next_window_outcome(traj: pd.DataFrame, agent: pd.DataFrame, month: int, agent_id: Any) -> str:
    future_month = min(month + 6, int(traj["month"].max()))
    t0 = traj[traj["month"] == month]
    t1 = traj[traj["month"] == future_month]
    a0 = agent[(agent["month"] == month) & (agent["agent_id"].astype(str) == str(agent_id))]
    a1 = agent[(agent["month"] == future_month) & (agent["agent_id"].astype(str) == str(agent_id))]
    parts = [f"month+6={future_month}"]
    if not t0.empty and not t1.empty:
        parts.append(f"aggregate_wealth_delta={float(t1.iloc[0]['avg_wealth']) - float(t0.iloc[0]['avg_wealth']):.1f}")
        parts.append(f"unemployment_delta={float(t1.iloc[0]['unemployment_pct']) - float(t0.iloc[0]['unemployment_pct']):.2f}pp")
    if not a0.empty and not a1.empty:
        parts.append(f"agent_wealth_delta={float(a1.iloc[0]['wealth']) - float(a0.iloc[0]['wealth']):.1f}")
        parts.append(f"employed_t+6={a1.iloc[0].get('employed', '')}")
    return "; ".join(parts)


def write_case_study() -> None:
    rows = []
    preferred = ["GPT-4o", "Gemini-3-Flash", "Qwen3-235B", "GPT-5.2"]
    for model in preferred:
        run_dir = RUNS / "E1" / model / "finevo" / "default" / "seed_13"
        if not run_dir.exists():
            continue
        traj_path = run_dir / "trajectory.csv"
        agent_path = run_dir / "agent_state.csv"
        if not traj_path.exists() or not agent_path.exists():
            continue
        traj = pd.read_csv(traj_path)
        agent = pd.read_csv(agent_path)
        month = choose_case_month(traj)
        actions = read_jsonl(run_dir / "actions.jsonl")
        retrievals = read_jsonl(run_dir / "memory_retrieval.jsonl")
        rules = read_jsonl(run_dir / "semantic_rules.jsonl")
        action = find_action(actions, month)
        agent_id = action.get("agent_id", "")
        retrieval = find_row(retrievals, month, agent_id)
        rule = find_row(rules, month, agent_id)
        state = traj[traj["month"] == month].iloc[0].to_dict()
        agent_state = agent[(agent["month"] == month) & (agent["agent_id"].astype(str) == str(agent_id))]
        wealth = agent_state.iloc[0].get("wealth", "") if not agent_state.empty else ""
        state_snapshot = (
            f"month={month}; S_t={float(state.get('global_sentiment', 0.0)):.3f}; "
            f"unemp={float(state.get('unemployment_pct', 0.0)):.2f}%; "
            f"infl={float(state.get('inflation_pct', 0.0)):.2f}%; "
            f"agent_wealth={wealth}"
        )
        retrieved = (
            f"ids={retrieval.get('retrieved_episode_ids', [])}; "
            f"scores={retrieval.get('retrieval_scores', [])[:3]}; "
            f"rules={retrieval.get('selected_rule_ids', [])}"
        )
        rule_text = (
            f"{rule.get('rule_id', '')}: if {rule.get('condition', '')}, "
            f"then {rule.get('action_guidance', '')}; confidence={rule.get('confidence', '')}"
            if rule
            else ""
        )
        rationale = action.get("rationale") or action.get("raw_output", "")
        text_for_code = " ".join([str(rationale), str(rule_text)])
        rows.append({
            "model": model,
            "seed": 13,
            "month": month,
            "agent_id": agent_id,
            "state_snapshot": state_snapshot,
            "regime_or_sentiment": f"S_t={float(state.get('global_sentiment', 0.0)):.3f}",
            "retrieved_episode_excerpt": short(retrieved, 240),
            "semantic_rule_excerpt": short(rule_text, 240),
            "parsed_action": json.dumps(action.get("parsed_action", {}), ensure_ascii=False, sort_keys=True),
            "raw_rationale_excerpt": short(rationale, 240),
            "next_6_month_outcome": next_window_outcome(traj, agent, month, agent_id),
            "manual_code": ";".join(code_case(text_for_code)),
            "source_run": str(run_dir.relative_to(ROOT)),
        })

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "case_study_trace.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[table] {TABLES / 'case_study_trace.csv'}")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llp{0.22\textwidth}p{0.22\textwidth}p{0.20\textwidth}p{0.16\textwidth}}",
        r"\toprule",
        r"Model & Month/agent & State cue & Retrieved memory / rule & Action rationale & Outcome/code \\",
        r"\midrule",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"{latex_escape(row['model'])} & {int(row['month'])}/{latex_escape(row['agent_id'])} & "
            f"{latex_escape(row['state_snapshot'])} & "
            f"{latex_escape(row['retrieved_episode_excerpt'])} {latex_escape(row['semantic_rule_excerpt'])} & "
            f"{latex_escape(row['raw_rationale_excerpt'])} {latex_escape(row['parsed_action'])} & "
            f"{latex_escape(row['next_6_month_outcome'])}; {latex_escape(row['manual_code'])} \\\\"
        )
    if out.empty:
        lines.append(r"\multicolumn{6}{c}{No exported FinEvo large-scale logs are available.} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Log-grounded crash-state trace selected by the lowest global sentiment month in each completed FinEvo large-scale diagnostic run. Excerpts are copied from exported logs.}",
        r"\label{tab:case_study_trace}",
        r"\end{table*}",
    ]
    out_tex = TEX / "table_case_study_trace.tex"
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[tex] {out_tex}")

    if not out.empty:
        fig, axes = plt.subplots(len(out), 1, figsize=(11.5, max(3.0, 2.35 * len(out))))
        if len(out) == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, out.iterrows()):
            ax.axis("off")
            title = f"{row['model']} | seed {row['seed']} | month {row['month']} | agent {row['agent_id']}"
            body = (
                f"State: {row['state_snapshot']}\n"
                f"Retrieved: {row['retrieved_episode_excerpt']}\n"
                f"Rule: {row['semantic_rule_excerpt']}\n"
                f"Action: {row['parsed_action']}\n"
                f"Rationale: {row['raw_rationale_excerpt']}\n"
                f"Next window: {row['next_6_month_outcome']} | Code: {row['manual_code']}"
            )
            ax.text(0, 1.0, title, transform=ax.transAxes, va="top", ha="left", fontsize=10, fontweight="bold")
            ax.text(
                0,
                0.86,
                "\n".join(textwrap.wrap(body, width=145, replace_whitespace=False)),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f7f7", "edgecolor": "#bbbbbb", "linewidth": 0.6},
            )
        fig.subplots_adjust(left=0.035, right=0.99, top=0.96, bottom=0.04, hspace=0.35)
        fig.savefig(FIGS / "fig_case_study.pdf", bbox_inches="tight")
        fig.savefig(FIGS / "fig_case_study.png", bbox_inches="tight", dpi=180)
        plt.close(fig)
        print(f"[fig] {FIGS / 'fig_case_study.pdf'}")


def main() -> None:
    write_cross_model_diagnostic()
    write_main_significance_and_ablation()
    write_case_study()
    print("[done] missing-result integration artifacts generated")


if __name__ == "__main__":
    main()
