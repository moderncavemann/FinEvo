#!/usr/bin/env python3
"""CLI for bounded verified dual-track memory simulations.

Examples:
    python simulate_verified.py --provider diagnostic --run-id local-g0
    python simulate_verified.py --provider openai --model gpt-5.2 --run-id api-g4

Runs larger than four agents or twelve months require ``--allow-larger-run``.
That switch exists for later preregistered experiments, not for method debugging.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from llm_providers import MultiModelLLM, create_llm_provider
from verified_memory.budget import BudgetLimits, RunBudget
from verified_memory.failure_artifacts import write_failure_receipt
from verified_memory.m0_utility import UtilityConfig
from verified_memory.runner import VerifiedRunConfig, run_verified_experiment
from verified_memory.runner_artifacts import write_verified_run_artifacts
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


ROOT = Path(__file__).resolve().parent


def _git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", *command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"verified-smoke-{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("diagnostic", "openai", "gemini", "thirdparty", "local", "ollama"),
        default="diagnostic",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key", default="", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--episode-length", type=int, default=6)
    parser.add_argument(
        "--context-mode",
        choices=("no-context", "prompt-only", "retrieval-only", "full"),
        default="retrieval-only",
    )
    parser.add_argument("--disable-episodic-retrieval", action="store_true")
    parser.add_argument("--disable-semantic", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-calls", type=int, default=24)
    parser.add_argument("--max-prompt-tokens", type=int, default=60_000)
    parser.add_argument("--max-completion-tokens", type=int, default=12_000)
    parser.add_argument("--max-total-tokens", type=int, default=72_000)
    parser.add_argument("--max-cost-usd", type=float, default=0.25)
    parser.add_argument("--max-elapsed-seconds", type=float, default=900.0)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-larger-run", action="store_true")
    return parser


def _default_model(provider: str) -> str:
    return {
        "diagnostic": "scripted-v1",
        "openai": "gpt-5.2",
        "gemini": "gemini-3-pro-preview",
        "thirdparty": "google/gemini-3-flash-preview",
        "local": "mlx-community/Llama-3.3-70B-Instruct-4bit",
        "ollama": "llama3:8b",
    }[provider]


def _required_call_floor(args: argparse.Namespace) -> int:
    actions = args.num_agents * args.episode_length
    proposals = 0
    if not args.disable_semantic and args.episode_length >= 2:
        proposals = args.num_agents
    return actions + proposals


def execute(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if not args.allow_larger_run and (
        args.num_agents > 4 or args.episode_length > 12
    ):
        raise ValueError(
            "bounded validation permits at most 4 agents and 12 months; "
            "use --allow-larger-run only after the smoke gates pass"
        )
    if args.workers < 1:
        raise ValueError("workers must be positive")
    required_calls = _required_call_floor(args)
    if args.max_calls < required_calls:
        raise ValueError(
            f"max-calls={args.max_calls} cannot complete the declared run; "
            f"at least {required_calls} calls are required"
        )
    run_id = args.run_id or _default_run_id()
    output_dir = args.output_dir or ROOT / "artifacts" / "verified_runs" / run_id
    model = args.model or _default_model(args.provider)

    if args.provider == "diagnostic":
        provider = ScriptedDiagnosticProvider()
    else:
        provider = create_llm_provider(
            args.provider,
            model=model,
            api_key=args.api_key or None,
            base_url=args.api_base or None,
            max_retries=args.max_retries,
        )
    llm = MultiModelLLM(provider, num_workers=args.workers)
    budget = RunBudget(
        BudgetLimits(
            max_calls=args.max_calls,
            max_prompt_tokens=args.max_prompt_tokens,
            max_completion_tokens=args.max_completion_tokens,
            max_total_tokens=args.max_total_tokens,
            max_cost_usd=args.max_cost_usd,
            max_elapsed_seconds=args.max_elapsed_seconds,
        ),
        budget_id=f"{run_id}-budget",
    )
    config = VerifiedRunConfig(
        run_id=run_id,
        seed=args.seed,
        num_agents=args.num_agents,
        episode_length=args.episode_length,
        context_mode=args.context_mode,
        enable_episodic_retrieval=not args.disable_episodic_retrieval,
        enable_semantic=not args.disable_semantic,
        temperature=args.temperature,
        top_p=args.top_p,
        max_retries=args.max_retries,
        utility=UtilityConfig(max_labor_hours=168.0),
    )
    git_commit = _git(["rev-parse", "HEAD"])
    git_dirty = bool(_git(["status", "--porcelain", "--untracked-files=no"]))
    provenance = {
        "purpose": "bounded verified-memory method smoke",
        "scientific_evidence": False,
        "legacy_runner_unchanged": True,
        "provider": args.provider,
        "model": model,
        "config_path": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "python": sys.version.split()[0],
        "untracked_files_present": bool(
            _git(["ls-files", "--others", "--exclude-standard"])
        ),
        "environment_keys_present": {
            "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
            "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
            "OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY")),
        },
    }
    try:
        result = run_verified_experiment(
            config,
            llm=llm,
            budget=budget,
            env_config_source=args.config,
        )
        manifest_path = write_verified_run_artifacts(
            output_dir,
            result,
            provenance=provenance,
            git_commit=git_commit,
            git_dirty=git_dirty,
        )
    except Exception as exc:
        try:
            write_failure_receipt(
                output_dir,
                scope="verified_simulation",
                error=exc,
                budget_snapshot=budget.snapshot().to_dict(),
                config=config.to_dict(),
                provenance=provenance,
                git_commit=git_commit,
                git_dirty=git_dirty,
            )
        except Exception as receipt_error:
            exc.add_note(f"failure receipt could not be written: {type(receipt_error).__name__}")
        raise
    return manifest_path, dict(result.summary)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest_path, summary = execute(args)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": summary["validation"]["status"],
                "run_id": summary["run_id"],
                "manifest": str(manifest_path),
                "diagnostic_only": summary["diagnostic_only"],
                "scientific_evidence": summary["scientific_evidence"],
                "final_metrics": summary["final_metrics"],
                "memory_diagnostics": summary["memory_diagnostics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["validation"]["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
