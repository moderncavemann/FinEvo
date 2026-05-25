#!/usr/bin/env python3
"""Run or print the EMNLP E1-E8 experiment matrix.

By default this script is a dry run and only prints commands. Use ``--run`` to
execute one or more experiment groups. Each successful simulation is exported
to the normalized ``runs/<exp_id>/<model>/<setting>/<variant>/seed_<seed>/``
layout with ``export_emnlp_outputs.py``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_SEEDS = [13, 21, 42, 87, 2026]


@dataclass(frozen=True)
class ModelSpec:
    display: str
    provider: str
    model: str
    api_base_env: Optional[str] = None
    api_key_env: Optional[str] = None
    local_url: Optional[str] = None


@dataclass(frozen=True)
class RunSpec:
    exp_id: str
    setting: str
    variant: str
    model_key: str = "gpt52"
    num_agents: int = 10
    num_months: int = 24
    tag: str = ""
    gap_fixes: bool = True
    flags: Dict[str, object] = field(default_factory=dict)
    supported: bool = True
    note: str = ""


MODELS: Dict[str, ModelSpec] = {
    "gpt52": ModelSpec("GPT-5.2", "openai", "gpt-5.2", api_key_env="OPENAI_API_KEY"),
    "gpt4o": ModelSpec("GPT-4o", "openai", "gpt-4o", api_key_env="OPENAI_API_KEY"),
    "llama4": ModelSpec(
        "Llama-4-Maverick",
        "local",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        local_url="http://localhost:8000/v1",
    ),
    "qwen3": ModelSpec(
        "Qwen3-235B",
        "local",
        "Qwen/Qwen3-VL-235B-A22B-Thinking",
        local_url="http://localhost:8001/v1",
    ),
    "gemini3flash": ModelSpec(
        "Gemini-3-Flash",
        "thirdparty",
        "google/gemini-3-flash-preview",
        api_base_env="OPENROUTER_BASE_URL",
        api_key_env="OPENROUTER_API_KEY",
    ),
}


def _matrix() -> List[RunSpec]:
    runs: List[RunSpec] = []

    # E1: large-scale paired baseline/FinEvo.
    for model_key in ["gpt52", "gpt4o", "llama4", "qwen3", "gemini3flash"]:
        runs.extend([
            RunSpec("E1", "text-only", "baseline", model_key, 100, 240, "E1_baseline", False),
            RunSpec("E1", "finevo", "default", model_key, 100, 240, "E1_finevo", True),
        ])

    # E2: component ablation. Current code supports the core rows; retrieval-only
    # variants need separate read/write/include flags and are listed as pending.
    e2_core = [
        ("default", "E2_full", True, {}),
        ("without-sentiment", "E2_without_sentiment", False, {
            "gap_fixes": False,
            "use_sentiment": False,
            "use_episodic": True,
            "use_semantic": True,
            "use_reflection": True,
        }),
        ("without-reflection", "E2_without_reflection", False, {
            "gap_fixes": False,
            "use_sentiment": True,
            "use_episodic": True,
            "use_semantic": True,
            "use_reflection": False,
        }),
        ("without-semantic-memory", "E2_without_semantic", False, {
            "gap_fixes": False,
            "use_sentiment": True,
            "use_episodic": True,
            "use_semantic": False,
            "use_reflection": True,
        }),
        ("without-episodic-memory", "E2_without_episodic", False, {
            "gap_fixes": False,
            "use_sentiment": True,
            "use_episodic": False,
            "use_semantic": False,
            "use_reflection": True,
        }),
    ]
    for variant, tag, gap_fixes, flags in e2_core:
        if "gap_fixes" in flags:
            gap_fixes = bool(flags.pop("gap_fixes"))
        runs.append(RunSpec("E2", "finevo", variant, "gpt52", 10, 24, tag, gap_fixes, flags))
    runs.extend([
        RunSpec("E2", "finevo", "episodic-only-retrieval", "gpt52", 10, 24, "E2_epi_only_retrieval", True, supported=False, note="Needs separate episodic write vs prompt-include flags."),
        RunSpec("E2", "finevo", "semantic-only-retrieval", "gpt52", 10, 24, "E2_sem_only_retrieval", True, supported=False, note="Needs semantic retrieval without episodic prompt inclusion."),
        RunSpec("E2", "finevo", "sliding-window-memory", "gpt52", 10, 24, "E2_sliding_window", True, supported=False, note="Needs a real sliding-window prompt baseline; current memory flags are not sufficient."),
    ])

    # E3: sentiment robustness. Adaptive weights are intentionally pending until
    # the learner is implemented and documented.
    sentiment_variants = [
        ("default", "E3_default", {}),
        ("alpha-plus-50", "E3_alpha_plus_50", {"s_inertia": 1.05}),
        ("alpha-minus-50", "E3_alpha_minus_50", {"s_inertia": 0.35}),
        ("beta-plus-50", "E3_beta_plus_50", {"s_inflation_sens": 0.225}),
        ("beta-minus-50", "E3_beta_minus_50", {"s_inflation_sens": 0.075}),
        ("gamma-plus-50", "E3_gamma_plus_50", {"s_unemp_sens": -0.30}),
        ("gamma-minus-50", "E3_gamma_minus_50", {"s_unemp_sens": -0.10}),
        ("delta-plus-50", "E3_delta_plus_50", {"s_gdp_sens": 0.15}),
        ("delta-minus-50", "E3_delta_minus_50", {"s_gdp_sens": 0.05}),
        ("no-macro-terms", "E3_no_macro", {"s_no_macro": True}),
        ("random-sentiment", "E3_random_sentiment", {"s_random": True}),
    ]
    for variant, tag, flags in sentiment_variants:
        runs.append(RunSpec("E3", "finevo", variant, "gpt52", 10, 24, tag, True, flags))
    runs.append(RunSpec("E3", "finevo", "adaptive-weights", "gpt52", 10, 24, "E3_adaptive_weights", True, supported=False, note="Implement and specify online learner before running."))

    # E4: text-event controls. These require simulator event logging support.
    for variant in ["numeric-only", "text-event", "shuffled-event", "random-event"]:
        runs.append(RunSpec("E4", "finevo", variant, "gpt52", 10, 24, f"E4_{variant.replace('-', '_')}", True, {"event_variant": variant}))

    # E5: prompt, decoding, and interface robustness.
    e5_variants = [
        ("default-prompt", "GPT-5.2", "gpt52", {"prompt_style": "default", "temperature": 0.2, "output_format": "json"}),
        ("concise-prompt", "GPT-5.2", "gpt52", {"prompt_style": "concise", "temperature": 0.2, "output_format": "json"}),
        ("risk-neutral-wording", "GPT-5.2", "gpt52", {"prompt_style": "risk_neutral", "temperature": 0.2, "output_format": "json"}),
        ("no-fairness-wording", "GPT-5.2", "gpt52", {"prompt_style": "no_fairness", "temperature": 0.2, "output_format": "json"}),
        ("temperature-0.0", "GPT-5.2", "gpt52", {"prompt_style": "default", "temperature": 0.0, "output_format": "json"}),
        ("temperature-0.7", "GPT-5.2", "gpt52", {"prompt_style": "default", "temperature": 0.7, "output_format": "json"}),
        ("natural-language-plus-json", "GPT-5.2", "gpt52", {"prompt_style": "default", "temperature": 0.2, "output_format": "natural_json"}),
        ("default-prompt", "Llama-4-Maverick", "llama4", {"prompt_style": "default", "temperature": 0.2, "output_format": "json"}),
        ("concise-prompt", "Llama-4-Maverick", "llama4", {"prompt_style": "concise", "temperature": 0.2, "output_format": "json"}),
        ("temperature-0.0", "Llama-4-Maverick", "llama4", {"prompt_style": "default", "temperature": 0.0, "output_format": "json"}),
        ("temperature-0.7", "Llama-4-Maverick", "llama4", {"prompt_style": "default", "temperature": 0.7, "output_format": "json"}),
        ("natural-language-plus-json", "Llama-4-Maverick", "llama4", {"prompt_style": "default", "temperature": 0.2, "output_format": "natural_json"}),
    ]
    for variant, model_label, model_key, flags in e5_variants:
        tag = f"E5_{model_label}_{variant}".replace(".", "").replace("-", "_").replace(" ", "_")
        runs.append(RunSpec("E5", "finevo", variant, model_key, 10, 24, tag, True, flags))

    # E6-E8 are kept visible but not executable until the simulator exposes the
    # required cognitive profiles, crash-state replay, and rule-survival modes.
    pending = [
        ("E6", "mixed-myopic-balanced-strategic", "Needs per-agent memory/reflection profiles and group metrics."),
        ("E7", "shared-crash-state-replay", "Needs frozen crash-state replay harness across models."),
        ("E8", "regime-conditioned-rule-survival", "Needs alternate confidence-update and no-pruning modes."),
    ]
    for exp_id, variant, note in pending:
        runs.append(RunSpec(exp_id, "finevo", variant, "gpt52", 10, 24, f"{exp_id}_{variant}", True, supported=False, note=note))

    return runs


def _bool_arg(value: bool) -> str:
    return "True" if value else "False"


def _model_safe_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def _source_dir(spec: RunSpec, model: ModelSpec, seed: int) -> Path:
    return Path("data") / f"{model.provider}-{_model_safe_name(model.model)}-{spec.tag}-seed{seed}-{spec.num_agents}agents-{spec.num_months}months"


def _simulate_cmd(spec: RunSpec, model: ModelSpec, seed: int, workers: int) -> List[str]:
    cmd = [
        sys.executable,
        "simulate.py",
        "--provider", model.provider,
        "--model", model.model,
        "--num_agents", str(spec.num_agents),
        "--episode_length", str(spec.num_months),
        "--gap_fixes", _bool_arg(spec.gap_fixes),
        "--seed", str(seed),
        "--tag", spec.tag,
        "--workers", str(workers),
        "--exp_id", spec.exp_id,
        "--setting", spec.setting,
        "--variant", spec.variant,
    ]
    if model.local_url:
        cmd.extend(["--local_url", model.local_url])
    if model.api_base_env and os.environ.get(model.api_base_env):
        cmd.extend(["--api_base", os.environ[model.api_base_env]])
    if "temperature" not in spec.flags:
        cmd.extend(["--temperature", "0.2"])
    for key, value in spec.flags.items():
        cmd.extend([f"--{key}", _bool_arg(value) if isinstance(value, bool) else str(value)])
    return cmd


def _export_cmd(spec: RunSpec, model: ModelSpec, seed: int, runs_root: str) -> List[str]:
    return [
        sys.executable,
        "export_emnlp_outputs.py",
        str(_source_dir(spec, model, seed)),
        "--runs-root", runs_root,
        "--exp-id", spec.exp_id,
        "--model", model.display,
        "--setting", spec.setting,
        "--variant", spec.variant,
        "--seed", str(seed),
        "--temperature", str(spec.flags.get("temperature", 0.2)),
        "--top-p", "1.0",
        "--max-tokens", "4000",
    ]


def _selected_specs(args: argparse.Namespace) -> List[RunSpec]:
    selected_exps = {item.strip() for item in args.exps.split(",") if item.strip()}
    selected_models = {item.strip() for item in args.models.split(",") if item.strip()} if args.models else set()
    selected_variants = {item.strip() for item in args.variants.split(",") if item.strip()} if args.variants else set()

    specs = []
    for spec in _matrix():
        if selected_exps and spec.exp_id not in selected_exps:
            continue
        if selected_models and spec.model_key not in selected_models and MODELS[spec.model_key].display not in selected_models:
            continue
        if selected_variants and spec.variant not in selected_variants and spec.tag not in selected_variants:
            continue
        if not spec.supported and not args.include_pending:
            continue
        specs.append(spec)
    return specs


def _print_command(title: str, cmd: List[str]) -> None:
    print(f"# {title}")
    print(" ".join(shlex.quote(part) for part in cmd))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exps", default="E2,E3,E4,E5", help="Comma-separated experiment ids; use E1 explicitly because it is expensive.")
    parser.add_argument("--models", default="", help="Optional comma-separated model keys/display names.")
    parser.add_argument("--variants", default="", help="Optional comma-separated variants or tags.")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run", action="store_true", help="Execute instead of printing commands.")
    parser.add_argument("--export-only", action="store_true", help="Skip simulation and export existing data dirs.")
    parser.add_argument("--include-pending", action="store_true", help="Show unsupported planned rows too.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    specs = _selected_specs(args)
    if not specs:
        print("No matching run specs.")
        return

    for spec in specs:
        model = MODELS[spec.model_key]
        for seed in seeds:
            if not spec.supported:
                print(f"# PENDING {spec.exp_id}/{model.display}/{spec.variant}: {spec.note}")
                continue
            sim_cmd = _simulate_cmd(spec, model, seed, args.workers)
            export_cmd = _export_cmd(spec, model, seed, args.runs_root)
            if not args.run:
                if not args.export_only:
                    _print_command(f"{spec.exp_id} {model.display} {spec.variant} seed {seed}", sim_cmd)
                _print_command(f"export {spec.exp_id} {model.display} {spec.variant} seed {seed}", export_cmd)
                continue

            if not args.export_only:
                subprocess.run(sim_cmd, cwd=Path(__file__).resolve().parent, check=True)
            subprocess.run(export_cmd, cwd=Path(__file__).resolve().parent, check=True)


if __name__ == "__main__":
    main()
