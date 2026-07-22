#!/usr/bin/env python3
"""Audit labor-threshold sensitivity from matched GPT-5.2 legacy runs.

The simulator samples the LLM's continuous ``work`` propensity into a binary
environment action.  This analysis therefore reports two distinct quantities:

* ``executed``: labor actually applied to the environment (0 or 168 hours);
* ``proposed``: the continuous LLM work propensity multiplied by 168 hours.

Keeping the two bases separate prevents a threshold sweep over a binary action
from being misrepresented as independent robustness evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "ACL24-EconAgent" / "data"
MESSAGE_MARKER = re.compile(r"^>>>>>>>>>\s*(user|assistant):\s*", re.MULTILINE)
DECISION_TIME = re.compile(r"\bNow it's\s+(\d{4}\.\d{2})\b")

SETTINGS = {
    "text-only": "openai-gpt-5.2-GPT_Baseline-seed*-10agents-24months",
    "finevo": "openai-gpt-5.2-GPT_Full-seed*-10agents-24months",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing the matched legacy source folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory for CSV, JSON, and Markdown outputs.",
    )
    parser.add_argument(
        "--thresholds",
        default="1,20,40",
        help="Strict low-labor thresholds in monthly hours.",
    )
    parser.add_argument(
        "--max-labor-hours",
        type=float,
        default=168.0,
        help="Hours represented by work propensity 1.0 and executed action 1.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], q: float) -> float:
    """Return a linearly interpolated percentile for q in [0, 1]."""
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def parse_messages(text: str) -> list[tuple[str, str]]:
    matches = list(MESSAGE_MARKER.finditer(text))
    messages: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        messages.append((match.group(1), text[match.end() : end].strip()))
    return messages


def extract_work(response: str) -> float:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", response):
        try:
            value, _ = decoder.raw_decode(response[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("work"), (int, float)):
            work = float(value["work"])
            if 0.0 <= work <= 1.0:
                return work

    fallback = re.search(r'["\']work["\']\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))', response)
    if fallback:
        work = float(fallback.group(1))
        if 0.0 <= work <= 1.0:
            return work
    raise ValueError(f"could not parse a valid work propensity from: {response[:160]!r}")


def proposed_by_agent(dialog_dir: Path) -> dict[str, list[tuple[str, float]]]:
    result: dict[str, list[tuple[str, float]]] = {}
    for path in sorted(item for item in dialog_dir.iterdir() if item.is_file()):
        messages = parse_messages(path.read_text(errors="replace"))
        decisions: list[tuple[str, float]] = []
        for idx, (role, content) in enumerate(messages):
            if role != "user":
                continue
            timestamp = DECISION_TIME.search(content)
            if not timestamp:
                continue
            if idx + 1 >= len(messages) or messages[idx + 1][0] != "assistant":
                raise ValueError(f"decision prompt lacks assistant response: {path}")
            decisions.append((timestamp.group(1), extract_work(messages[idx + 1][1])))
        result[path.name] = decisions
    return result


def discover_runs(data_root: Path) -> dict[str, dict[int, Path]]:
    discovered: dict[str, dict[int, Path]] = {}
    for setting, pattern in SETTINGS.items():
        by_seed: dict[int, Path] = {}
        for path in sorted(data_root.glob(pattern)):
            summary = read_json(path / "summary.json")
            seed = int(summary["random_seed"])
            if seed in by_seed:
                raise ValueError(f"duplicate {setting} seed {seed}: {path}")
            if summary.get("model") != "openai/gpt-5.2":
                raise ValueError(f"unexpected model in {path}: {summary.get('model')}")
            if int(summary.get("num_agents", -1)) != 10 or int(summary.get("episode_length", -1)) != 24:
                raise ValueError(f"unexpected run dimensions in {path}")
            by_seed[seed] = path
        if not by_seed:
            raise FileNotFoundError(f"no source folders matched {data_root / pattern}")
        discovered[setting] = by_seed

    matched = set.intersection(*(set(runs) for runs in discovered.values()))
    if not matched:
        raise ValueError("baseline and FinEvo have no matched seeds")
    return {
        setting: {seed: runs[seed] for seed in sorted(matched)}
        for setting, runs in discovered.items()
    }


def load_agent_months(
    setting: str,
    seed: int,
    source_dir: Path,
    max_labor_hours: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = read_json(source_dir / "summary.json")
    with (source_dir / "dense_log.pkl").open("rb") as handle:
        dense_log = pickle.load(handle)

    states = dense_log.get("states") or []
    actions = dense_log.get("actions") or []
    expected_months = int(summary["episode_length"])
    if len(actions) != expected_months or not states:
        raise ValueError(
            f"{source_dir}: expected {expected_months} action months, found {len(actions)}"
        )

    initial_state = states[0]
    agent_ids = sorted(
        (key for key, value in initial_state.items() if key != "p" and isinstance(value, dict)),
        key=lambda value: int(value) if str(value).isdigit() else str(value),
    )
    if len(agent_ids) != int(summary["num_agents"]):
        raise ValueError(f"{source_dir}: agent count mismatch")

    id_by_name = {
        str(initial_state[agent_id].get("endogenous", {}).get("name", "")): agent_id
        for agent_id in agent_ids
    }
    if "" in id_by_name or len(id_by_name) != len(agent_ids):
        raise ValueError(f"{source_dir}: missing or duplicate agent names")

    proposed = proposed_by_agent(source_dir / "dialogs")
    if set(proposed) != set(id_by_name):
        missing = sorted(set(id_by_name) - set(proposed))
        extra = sorted(set(proposed) - set(id_by_name))
        raise ValueError(f"{source_dir}: dialog/agent mismatch; missing={missing}, extra={extra}")

    rows: list[dict[str, Any]] = []
    for name, agent_id in sorted(id_by_name.items(), key=lambda item: int(item[1])):
        decisions = proposed[name]
        if len(decisions) != expected_months:
            raise ValueError(
                f"{source_dir}/{name}: expected {expected_months} decision responses, "
                f"found {len(decisions)}"
            )
        for month, (timestamp, work_propensity) in enumerate(decisions):
            action = actions[month].get(str(agent_id), {})
            executed_code = float(action.get("SimpleLabor", 0.0))
            if executed_code not in {0.0, 1.0}:
                raise ValueError(
                    f"{source_dir}: expected binary SimpleLabor action, got {executed_code}"
                )
            rows.append(
                {
                    "setting": setting,
                    "seed": seed,
                    "agent_id": agent_id,
                    "agent_name": name,
                    "month": month,
                    "calendar_month": timestamp,
                    "proposed_work_propensity": work_propensity,
                    "proposed_labor_hours": work_propensity * max_labor_hours,
                    "executed_labor_action": int(executed_code),
                    "executed_labor_hours": executed_code * max_labor_hours,
                }
            )

    executed_low_rate = (
        sum(float(row["executed_labor_hours"]) < 1.0 for row in rows) / len(rows)
    )
    reported_unemployment = float(summary["final_metrics"]["avg_unemployment"])
    discrepancy = abs(executed_low_rate - reported_unemployment)
    if discrepancy > 1e-12:
        raise ValueError(
            f"{source_dir}: executed h<1 rate {executed_low_rate} does not match "
            f"reported avg_unemployment {reported_unemployment}"
        )

    source_record = {
        "setting": setting,
        "seed": seed,
        "source_dir": os.path.relpath(source_dir.resolve(), REPO_ROOT.resolve()),
        "num_agents": len(agent_ids),
        "num_months": expected_months,
        "num_agent_months": len(rows),
        "reported_avg_unemployment_pct": reported_unemployment * 100.0,
        "executed_h_lt_1_pct": executed_low_rate * 100.0,
        "unemployment_validation_abs_diff_pct": discrepancy * 100.0,
        "summary_sha256": sha256(source_dir / "summary.json"),
        "dense_log_sha256": sha256(source_dir / "dense_log.pkl"),
        "dialog_sha256": {
            path.name: sha256(path)
            for path in sorted((source_dir / "dialogs").iterdir())
            if path.is_file()
        },
    }
    return rows, source_record


def basis_values(rows: Iterable[dict[str, Any]], basis: str) -> list[float]:
    field = f"{basis}_labor_hours"
    return [float(row[field]) for row in rows]


def distribution(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean_hours": statistics.mean(values),
        "median_hours": statistics.median(values),
        "p05_hours": percentile(values, 0.05),
        "p25_hours": percentile(values, 0.25),
        "p75_hours": percentile(values, 0.75),
        "p95_hours": percentile(values, 0.95),
        "min_hours": min(values),
        "max_hours": max(values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_tables(
    agent_months: list[dict[str, Any]], thresholds: Sequence[float]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in agent_months:
        grouped[(str(row["setting"]), int(row["seed"]))].append(row)

    by_seed: list[dict[str, Any]] = []
    for (setting, seed), rows in sorted(grouped.items()):
        for basis in ("executed", "proposed"):
            values = basis_values(rows, basis)
            stats = distribution(values)
            for threshold in thresholds:
                by_seed.append(
                    {
                        "setting": setting,
                        "action_basis": basis,
                        "seed": seed,
                        "threshold_hours_strict_lt": threshold,
                        "low_labor_rate_pct": 100.0
                        * sum(value < threshold for value in values)
                        / len(values),
                        "n_agent_months": len(values),
                        **stats,
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    for setting in SETTINGS:
        setting_rows = [row for row in agent_months if row["setting"] == setting]
        seeds = sorted({int(row["seed"]) for row in setting_rows})
        for basis in ("executed", "proposed"):
            pooled = basis_values(setting_rows, basis)
            pooled_stats = distribution(pooled)
            for threshold in thresholds:
                seed_rates = [
                    float(row["low_labor_rate_pct"])
                    for row in by_seed
                    if row["setting"] == setting
                    and row["action_basis"] == basis
                    and float(row["threshold_hours_strict_lt"]) == threshold
                ]
                summary_rows.append(
                    {
                        "setting": setting,
                        "action_basis": basis,
                        "threshold_hours_strict_lt": threshold,
                        "n_seeds": len(seeds),
                        "seeds": ",".join(str(seed) for seed in seeds),
                        "n_agent_months": len(pooled),
                        "seed_rate_mean_pct": statistics.mean(seed_rates),
                        "seed_rate_sd_pct": statistics.stdev(seed_rates)
                        if len(seed_rates) > 1
                        else 0.0,
                        "pooled_rate_pct": 100.0
                        * sum(value < threshold for value in pooled)
                        / len(pooled),
                        **pooled_stats,
                    }
                )

    paired: list[dict[str, Any]] = []
    for basis in ("executed", "proposed"):
        for threshold in thresholds:
            baseline = {
                int(row["seed"]): float(row["low_labor_rate_pct"])
                for row in by_seed
                if row["setting"] == "text-only"
                and row["action_basis"] == basis
                and float(row["threshold_hours_strict_lt"]) == threshold
            }
            finevo = {
                int(row["seed"]): float(row["low_labor_rate_pct"])
                for row in by_seed
                if row["setting"] == "finevo"
                and row["action_basis"] == basis
                and float(row["threshold_hours_strict_lt"]) == threshold
            }
            seeds = sorted(set(baseline) & set(finevo))
            differences = [finevo[seed] - baseline[seed] for seed in seeds]
            paired.append(
                {
                    "action_basis": basis,
                    "threshold_hours_strict_lt": threshold,
                    "n_seed_pairs": len(seeds),
                    "seeds": ",".join(str(seed) for seed in seeds),
                    "text_only_seed_rate_mean_pct": statistics.mean(
                        baseline[seed] for seed in seeds
                    ),
                    "finevo_seed_rate_mean_pct": statistics.mean(
                        finevo[seed] for seed in seeds
                    ),
                    "finevo_minus_text_only_mean_pp": statistics.mean(differences),
                    "finevo_minus_text_only_sd_pp": statistics.stdev(differences)
                    if len(differences) > 1
                    else 0.0,
                    "paired_differences_pp_by_seed": ";".join(
                        f"{seed}:{finevo[seed] - baseline[seed]:.12g}" for seed in seeds
                    ),
                }
            )
    return by_seed, summary_rows, paired


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    thresholds: Sequence[float],
) -> None:
    by_key = {
        (
            str(row["setting"]),
            str(row["action_basis"]),
            float(row["threshold_hours_strict_lt"]),
        ): row
        for row in summary_rows
    }
    pair_key = {
        (str(row["action_basis"]), float(row["threshold_hours_strict_lt"])): row
        for row in paired
    }

    lines = [
        "> [!WARNING]",
        "> **HISTORICAL PRE-P0 V1 EVIDENCE ONLY**",
        ">",
        "> This audit covers the legacy binary 0/168-hour execution path. It is not",
        "> utility calibration or action-distribution evidence for current Evidence-",
        "> Grounded Rule Memory v2.",
        "",
        "# Labor-hours threshold sensitivity audit",
        "",
        "## Scope and provenance",
        "",
        "This audit uses the matched GPT-5.2 legacy runs `GPT_Baseline` and "
        "`GPT_Full`: seeds 0, 1, and 2; 10 agents; 24 months. Each setting "
        "therefore contributes 720 agent-month observations. Variability is reported "
        "across the three seeds; the agent-month rows are not treated as independent "
        "experimental replicates.",
        "",
        "## Executed labor (environment action)",
        "",
        "| Strict threshold | Text-only low-labor rate | FinEvo low-labor rate | FinEvo - text-only |",
        "|---:|---:|---:|---:|",
    ]
    for threshold in thresholds:
        baseline = by_key[("text-only", "executed", threshold)]
        finevo = by_key[("finevo", "executed", threshold)]
        comparison = pair_key[("executed", threshold)]
        lines.append(
            f"| h < {threshold:g} h | {fmt(float(baseline['seed_rate_mean_pct']))}% "
            f"+/- {fmt(float(baseline['seed_rate_sd_pct']))} | "
            f"{fmt(float(finevo['seed_rate_mean_pct']))}% +/- "
            f"{fmt(float(finevo['seed_rate_sd_pct']))} | "
            f"{fmt(float(comparison['finevo_minus_text_only_mean_pp']))} pp |"
        )

    lines += [
        "",
        "The three executed-action thresholds are identical by construction, not "
        "three independent robustness checks. The runner samples the LLM work "
        "propensity to a binary `SimpleLabor` action, and this configuration maps the "
        "two possible actions to 0 or 168 monthly hours. The executed `h < 1` rate "
        "exactly reproduces each run's stored `avg_unemployment` value.",
        "With only three matched seeds, these values are descriptive; no "
        "inferential p-value is reported.",
        "",
        "## Proposed labor (pre-sampling LLM propensity)",
        "",
        "| Strict threshold | Text-only low-labor rate | FinEvo low-labor rate |",
        "|---:|---:|---:|",
    ]
    for threshold in thresholds:
        baseline = by_key[("text-only", "proposed", threshold)]
        finevo = by_key[("finevo", "proposed", threshold)]
        lines.append(
            f"| h < {threshold:g} h | {fmt(float(baseline['seed_rate_mean_pct']))}% "
            f"+/- {fmt(float(baseline['seed_rate_sd_pct']))} | "
            f"{fmt(float(finevo['seed_rate_mean_pct']))}% +/- "
            f"{fmt(float(finevo['seed_rate_sd_pct']))} |"
        )

    lines += [
        "",
        "| Setting | Mean proposed hours | Median | IQR | Range |",
        "|---|---:|---:|---:|---:|",
    ]
    for setting, label in (("text-only", "Text-only"), ("finevo", "FinEvo")):
        row = by_key[(setting, "proposed", thresholds[0])]
        lines.append(
            f"| {label} | {fmt(float(row['mean_hours']))} | "
            f"{fmt(float(row['median_hours']))} | "
            f"{fmt(float(row['p25_hours']))}-{fmt(float(row['p75_hours']))} | "
            f"{fmt(float(row['min_hours']))}-{fmt(float(row['max_hours']))} |"
        )

    lines += [
        "",
        "No proposed action is near the 1/20/40-hour cutoffs. This rules out the "
        "specific artifact in which FinEvo merely moves agents from about 0 hours to "
        "2-5 hours. It does **not** establish realistic labor-market calibration: both "
        "systems operate close to the 168-hour ceiling, and only three matched seeds "
        "are available. The defensible paper claim is therefore narrow: the reported "
        "difference is not caused by agents clustering just above the original "
        "`h < 1` cutoff, while the binary action space and ceiling saturation remain "
        "limitations.",
        "",
        "## Reproduction",
        "",
        "Run from the `eccv26_EconAgent` repository root:",
        "",
        "```bash",
        "python artifacts/labor_threshold_sensitivity/analyze_labor_thresholds.py",
        "```",
        "",
        "Machine-readable outputs are `labor_agent_month.csv`, "
        "`labor_threshold_by_seed.csv`, `labor_threshold_summary.csv`, "
        "`labor_threshold_paired_differences.csv`, and `source_manifest.json`.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    thresholds = [float(value.strip()) for value in args.thresholds.split(",") if value.strip()]
    if not thresholds or any(value < 0 for value in thresholds):
        raise ValueError("thresholds must contain non-negative numeric values")
    thresholds = sorted(set(thresholds))

    runs = discover_runs(args.data_root.resolve())
    agent_months: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for setting, by_seed in runs.items():
        for seed, source_dir in by_seed.items():
            rows, source = load_agent_months(
                setting, seed, source_dir, args.max_labor_hours
            )
            agent_months.extend(rows)
            sources.append(source)

    by_seed, summary_rows, paired = make_tables(agent_months, thresholds)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "labor_agent_month.csv", agent_months)
    write_csv(output_dir / "labor_threshold_by_seed.csv", by_seed)
    write_csv(output_dir / "labor_threshold_summary.csv", summary_rows)
    write_csv(output_dir / "labor_threshold_paired_differences.csv", paired)
    with (output_dir / "source_manifest.json").open("w") as handle:
        json.dump(
            {
                "analysis": "labor_threshold_sensitivity_v1",
                "evidence_scope": "historical_pre_p0_v1",
                "current_method_scientific_evidence": False,
                "method_implementation": "legacy_binary_labor_simulate_py",
                "thresholds_strict_lt_hours": thresholds,
                "max_labor_hours": args.max_labor_hours,
                "matched_seeds": sorted(
                    set.intersection(*(set(by_seed) for by_seed in runs.values()))
                ),
                "sources": sources,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    write_report(output_dir / "report.md", summary_rows, paired, thresholds)

    print(f"wrote labor sensitivity artifacts to {output_dir}")
    print(f"validated {len(sources)} source runs and {len(agent_months)} agent-month rows")


if __name__ == "__main__":
    main()
