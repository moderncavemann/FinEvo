#!/usr/bin/env python3
"""Summarize the same-day E9 visible/no-visible cue pair by matched seed."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml

try:
    import scipy
    from scipy import stats as scipy_stats
except ImportError:  # Keep descriptive outputs available in minimal installs.
    scipy = None
    scipy_stats = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
VARIANTS = {
    "visible-cue-control": True,
    "no-visible-cue": False,
}
METRICS = {
    "avg_wealth_final": "higher",
    "gini_final": "lower",
    "unemployment_final_pct": "lower",
    "low_labor_h_lt_1_full_horizon_pct": "lower",
    "inflation_dev_abs_pct": "lower",
    "gdp_volatility_pct": "lower",
    "invalid_action_rate": "lower",
    "api_error_rate": "lower",
    "total_cost_usd": "lower",
    "wall_time_min": "lower",
}
PAIR_CONFIG_FIELDS = [
    "model_display_name",
    "model_exact_id",
    "api_provider",
    "api_access_date",
    "code_commit",
    "prompt_version",
    "reflection_prompt_version",
    "temperature",
    "top_p",
    "max_tokens",
    "decision_max_tokens",
    "reflection_max_tokens",
    "seed",
    "num_agents",
    "num_months",
    "setting",
    "environment_params",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--expected-seeds", default="13,21,42,87,2026")
    return parser.parse_args()


def read_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one metrics row in {path}, found {len(rows)}")
    return rows[0]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def full_horizon_low_labor_pct(run_dir: Path, metrics: dict[str, str]) -> float:
    path = run_dir / "agent_state.csv"
    with path.open(newline="") as handle:
        values = [
            float(row["labor_hours"])
            for row in csv.DictReader(handle)
            if row.get("labor_hours") not in {None, ""}
        ]
    expected = int(float(metrics["num_agents"])) * int(float(metrics["num_months"]))
    if len(values) != expected:
        raise ValueError(
            f"{path}: expected {expected} executed labor rows, found {len(values)}"
        )
    return 100.0 * sum(value < 1.0 for value in values) / len(values)


def as_float(value: Any) -> float:
    if value is None or value == "":
        raise ValueError("paired E9 metric is missing")
    return float(value)


def paired_inference(differences: list[float]) -> dict[str, float | bool]:
    """Return a Student-t CI and two-sided paired t-test for paired differences."""
    if scipy_stats is None or len(differences) < 2:
        return {
            "paired_inference_available": False,
            "paired_diff_ci95_low": math.nan,
            "paired_diff_ci95_high": math.nan,
            "paired_t_stat": math.nan,
            "paired_t_p_two_sided": math.nan,
        }

    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    if sd == 0.0:
        return {
            "paired_inference_available": True,
            "paired_diff_ci95_low": mean,
            "paired_diff_ci95_high": mean,
            "paired_t_stat": 0.0 if mean == 0.0 else math.copysign(math.inf, mean),
            "paired_t_p_two_sided": 1.0 if mean == 0.0 else 0.0,
        }

    sem = sd / math.sqrt(len(differences))
    critical = float(scipy_stats.t.ppf(0.975, df=len(differences) - 1))
    test = scipy_stats.ttest_1samp(differences, popmean=0.0)
    return {
        "paired_inference_available": True,
        "paired_diff_ci95_low": mean - critical * sem,
        "paired_diff_ci95_high": mean + critical * sem,
        "paired_t_stat": float(test.statistic),
        "paired_t_p_two_sided": float(test.pvalue),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        if path.exists():
            path.unlink()
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def discover(runs_root: Path) -> dict[str, dict[int, tuple[dict[str, str], dict[str, Any], Path]]]:
    base = runs_root / "E9" / "GPT-5.2" / "finevo"
    result: dict[str, dict[int, tuple[dict[str, str], dict[str, Any], Path]]] = {}
    for variant, expected_flag in VARIANTS.items():
        by_seed = {}
        for metric_path in sorted((base / variant).glob("seed_*/metrics_summary.csv")):
            seed_text = metric_path.parent.name.removeprefix("seed_")
            seed = int(seed_text)
            config_path = metric_path.parent / "config.yaml"
            config = read_yaml(config_path)
            if "show_regime_cue" not in config:
                raise ValueError(f"missing show_regime_cue in {config_path}")
            if bool(config["show_regime_cue"]) is not expected_flag:
                raise ValueError(
                    f"{config_path}: expected show_regime_cue={expected_flag}, "
                    f"found {config['show_regime_cue']!r}"
                )
            metrics = read_csv_row(metric_path)
            metrics["low_labor_h_lt_1_full_horizon_pct"] = str(
                full_horizon_low_labor_pct(metric_path.parent, metrics)
            )
            by_seed[seed] = (metrics, config, metric_path.parent)
        result[variant] = by_seed
    return result


def validate_pair(
    seed: int,
    visible: tuple[dict[str, str], dict[str, Any], Path],
    hidden: tuple[dict[str, str], dict[str, Any], Path],
) -> None:
    _, visible_config, visible_path = visible
    _, hidden_config, hidden_path = hidden
    mismatches = [
        field
        for field in PAIR_CONFIG_FIELDS
        if visible_config.get(field) != hidden_config.get(field)
    ]
    if mismatches:
        raise ValueError(
            f"seed {seed} is not a controlled pair; config fields differ: {mismatches}; "
            f"paths={visible_path},{hidden_path}"
        )
    if not visible_config.get("api_access_date"):
        raise ValueError(f"seed {seed}: missing api_access_date for same-day validation")


def analyze(
    runs: dict[str, dict[int, tuple[dict[str, str], dict[str, Any], Path]]],
    expected_seeds: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    visible = runs["visible-cue-control"]
    hidden = runs["no-visible-cue"]
    paired_seeds = sorted(set(visible) & set(hidden))

    by_seed: list[dict[str, Any]] = []
    for seed in paired_seeds:
        validate_pair(seed, visible[seed], hidden[seed])
        visible_metrics, visible_config, visible_path = visible[seed]
        hidden_metrics, _, hidden_path = hidden[seed]
        for metric, direction in METRICS.items():
            visible_value = as_float(visible_metrics.get(metric))
            hidden_value = as_float(hidden_metrics.get(metric))
            difference = hidden_value - visible_value
            by_seed.append(
                {
                    "seed": seed,
                    "api_access_date": visible_config["api_access_date"],
                    "metric": metric,
                    "preferred_direction": direction,
                    "visible_cue_control": visible_value,
                    "no_visible_cue": hidden_value,
                    "no_visible_minus_visible": difference,
                    "direction_adjusted_no_visible_benefit": difference
                    * (1.0 if direction == "higher" else -1.0),
                    "visible_source": str(visible_path.relative_to(REPO_ROOT)),
                    "hidden_source": str(hidden_path.relative_to(REPO_ROOT)),
                }
            )

    summary: list[dict[str, Any]] = []
    for metric, direction in METRICS.items():
        rows = [row for row in by_seed if row["metric"] == metric]
        if not rows:
            continue
        visible_values = [float(row["visible_cue_control"]) for row in rows]
        hidden_values = [float(row["no_visible_cue"]) for row in rows]
        differences = [float(row["no_visible_minus_visible"]) for row in rows]
        benefits = [float(row["direction_adjusted_no_visible_benefit"]) for row in rows]
        difference_mean = statistics.mean(differences)
        difference_sd = statistics.stdev(differences) if len(rows) > 1 else 0.0
        inference = paired_inference(differences)
        summary.append(
            {
                "metric": metric,
                "preferred_direction": direction,
                "n_seed_pairs": len(rows),
                "seeds": ",".join(str(row["seed"]) for row in rows),
                "visible_mean": statistics.mean(visible_values),
                "visible_sd": statistics.stdev(visible_values) if len(rows) > 1 else 0.0,
                "no_visible_mean": statistics.mean(hidden_values),
                "no_visible_sd": statistics.stdev(hidden_values) if len(rows) > 1 else 0.0,
                "no_visible_minus_visible_mean": difference_mean,
                "no_visible_minus_visible_sd": difference_sd,
                **inference,
                "direction_adjusted_no_visible_benefit_mean": statistics.mean(benefits),
            }
        )

    expected = set(expected_seeds)
    status = {
        "analysis": "e9_same_day_cue_visibility_pair_v1",
        "evidence_scope": "historical_pre_p0_v1",
        "current_method_scientific_evidence": False,
        "method_implementation": "legacy_simulate_py_deterministic_template_memory",
        "status": "complete" if set(paired_seeds) == expected else "pending_or_partial",
        "expected_seeds": expected_seeds,
        "visible_seeds": sorted(visible),
        "no_visible_seeds": sorted(hidden),
        "paired_seeds": paired_seeds,
        "missing_visible_seeds": sorted(expected - set(visible)),
        "missing_no_visible_seeds": sorted(expected - set(hidden)),
        "same_day_config_pairs_validated": len(paired_seeds),
        "paired_inference_available": scipy_stats is not None,
        "paired_inference_method": "two-sided one-sample Student-t test over matched-seed differences",
        "scipy_version": getattr(scipy, "__version__", None),
        "comparison_fallback_used": False,
        "note": "Only E9 visible-cue-control vs E9 no-visible-cue is compared; old E2 rows are never substituted.",
    }
    return status, by_seed, summary


def write_report(
    path: Path,
    status: dict[str, Any],
    by_seed: list[dict[str, Any]],
    summary: list[dict[str, Any]],
) -> None:
    lines = [
        "> [!WARNING]",
        "> **HISTORICAL PRE-P0 V1 EVIDENCE ONLY**",
        ">",
        "> This E9 pair used legacy `simulate.py` and deterministic-template memory. Its",
        "> completion does not constitute current M1 route-decomposition evidence.",
        "",
        "# E9 paired cue-visibility analysis",
        "",
        f"Status: **{status['status']}**",
        "",
        f"Paired seeds: {status['paired_seeds'] or 'none'}",
        "",
        "This analysis compares only the same-day `visible-cue-control` and "
        "`no-visible-cue` E9 rows. It does not fall back to older E2 runs.",
    ]
    if summary:
        lines += [
            "",
            "| Metric | Visible mean +/- SD | No-visible mean +/- SD | No-visible - visible [95% t CI] | Paired p |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in summary:
            lines.append(
                f"| {row['metric']} | {row['visible_mean']:.6g} +/- "
                f"{row['visible_sd']:.6g} | {row['no_visible_mean']:.6g} +/- "
                f"{row['no_visible_sd']:.6g} | "
                f"{row['no_visible_minus_visible_mean']:.6g} "
                f"[{row['paired_diff_ci95_low']:.6g}, "
                f"{row['paired_diff_ci95_high']:.6g}] | "
                f"{row['paired_t_p_two_sided']:.4g} |"
            )
    else:
        lines += [
            "",
            "No completed normalized seed pair is present yet. Run both E9 variants "
            "and export them before drawing any performance conclusion.",
        ]
    if status["status"] == "complete" and summary:
        indexed = {row["metric"]: row for row in summary}
        wealth = indexed["avg_wealth_final"]
        gini = indexed["gini_final"]
        low_labor = indexed["low_labor_h_lt_1_full_horizon_pct"]
        inflation = indexed["inflation_dev_abs_pct"]
        wealth_relative = (
            100.0
            * float(wealth["no_visible_minus_visible_mean"])
            / float(wealth["visible_mean"])
        )
        wealth_retained = 100.0 * float(wealth["no_visible_mean"]) / float(
            wealth["visible_mean"]
        )

        def direction_count(metric: str, predicate) -> int:
            return sum(
                predicate(float(row["no_visible_minus_visible"]))
                for row in by_seed
                if row["metric"] == metric
            )

        if status["paired_inference_available"]:
            primary = [wealth, gini, low_labor]
            primary_ci_crosses_zero = [
                float(row["paired_diff_ci95_low"]) <= 0.0
                <= float(row["paired_diff_ci95_high"])
                for row in primary
            ]
            inference_text = (
                "For wealth, Gini, and full-horizon low labor, the 95% paired "
                f"Student-t CIs "
                f"{'all include' if all(primary_ci_crosses_zero) else 'do not all include'} "
                "zero; their two-sided paired-test p-values are "
                f"{float(wealth['paired_t_p_two_sided']):.3f}, "
                f"{float(gini['paired_t_p_two_sided']):.3f}, and "
                f"{float(low_labor['paired_t_p_two_sided']):.3f}, respectively. "
                "With n=5 and no pre-specified non-inferiority margin, the result "
                "supports neither a statistically detectable cue effect nor an "
                "equivalence/non-inferiority claim. The defensible conclusion is "
                "that hiding the cue does not produce an obvious point-estimate "
                "collapse, while uncertainty remains large."
            )
        else:
            inference_text = (
                "SciPy is unavailable, so the paired Student-t intervals and tests "
                "were not computed. The descriptive results must not be treated as "
                "an equivalence or non-inferiority test."
            )

        lines += [
            "",
            "## Interpretation",
            "",
            f"Hiding the prompt-level cue reduced mean final wealth by "
            f"{abs(wealth_relative):.2f}% "
            f"({float(wealth['no_visible_minus_visible_mean']):.1f}; lower in "
            f"{direction_count('avg_wealth_final', lambda value: value < 0)}/5 seeds), "
            f"increased Gini by {float(gini['no_visible_minus_visible_mean']):.4f} "
            f"(worse in {direction_count('gini_final', lambda value: value > 0)}/5), "
            f"and increased the full-horizon low-labor rate by "
            f"{float(low_labor['no_visible_minus_visible_mean']):.3f} percentage "
            f"points (worse in "
            f"{direction_count('low_labor_h_lt_1_full_horizon_pct', lambda value: value > 0)}/5). "
            f"Absolute inflation deviation improved by "
            f"{abs(float(inflation['no_visible_minus_visible_mean'])):.3f} points.",
            "",
            "The point estimates suggest that the prompt cue helps, especially for "
            "distributional and labor outcomes. The no-visible condition retains "
            f"{wealth_retained:.2f}% of visible-control mean wealth and has a "
            f"{float(low_labor['no_visible_mean']):.2f}% full-horizon low-labor rate, "
            "but these point estimates alone do not establish robustness.",
            "",
            inference_text,
        ]
    lines += [
        "",
        "`low_labor_h_lt_1_full_horizon_pct` is reconstructed from all executed "
        "agent-month actions in `agent_state.csv`; it is distinct from the exported "
        "final-period `unemployment_final_pct` field.",
        "",
        "`unemployment_final_pct` is the final logged annual-window metric, not a "
        "full-horizon average. In these 24-month runs there is only one logged "
        "annual GDP-growth observation, so `gdp_volatility_pct = 0` is mechanical, "
        "not interpretable, and is not used in the conclusion.",
        "",
        "The experiment isolates explicit cue text only. Endogenous sentiment still "
        "enters the numeric state-similarity feature vector; the separate "
        "`regime_match` retrieval component is currently fixed at 0.0.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    expected_seeds = [
        int(value.strip()) for value in args.expected_seeds.split(",") if value.strip()
    ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover(args.runs_root.resolve())
    status, by_seed, summary = analyze(runs, expected_seeds)
    (output_dir / "paired_analysis_status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    write_csv(output_dir / "paired_metrics_by_seed.csv", by_seed)
    write_csv(output_dir / "paired_metrics_summary.csv", summary)
    write_report(output_dir / "paired_report.md", status, by_seed, summary)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
