#!/usr/bin/env python3
"""Run five integrity-matched memory interventions from a sealed verified run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from llm_providers import MultiModelLLM, create_llm_provider
from verified_memory.budget import BudgetLimits, RunBudget
from verified_memory.replay_experiment import (
    build_paired_snapshot,
    run_paired_replay,
    summarize_paired_replay,
    write_paired_replay_artifacts,
)
from verified_memory.runner_artifacts import load_verified_run_artifacts
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


ROOT = Path(__file__).resolve().parent


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decision-t", type=int, default=None)
    parser.add_argument("--agent-id", type=int, default=0)
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key", default="", help=argparse.SUPPRESS)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-cost-usd", type=float, default=0.15)
    parser.add_argument("--max-prompt-tokens", type=int, default=30_000)
    parser.add_argument("--max-completion-tokens", type=int, default=3_000)
    return parser


def execute(args: argparse.Namespace):
    source = load_verified_run_artifacts(args.source_run)
    source_provider, _, source_model = str(source.summary["provider_model"]).partition("/")
    provider_name = args.provider or source_provider
    model_name = args.model or source_model
    if (provider_name, model_name) != (source_provider, source_model):
        raise ValueError("paired replay must use the source run provider and model")
    selected_actions = [
        row
        for row in source.stream("actions")
        if row["agent_id"] == args.agent_id and row["selected_rule_ids"]
    ]
    if not selected_actions:
        raise ValueError("source run has no active-rule decision for this agent")
    decision_t = (
        max(row["decision_t"] for row in selected_actions)
        if args.decision_t is None
        else args.decision_t
    )
    if provider_name == "diagnostic":
        provider = ScriptedDiagnosticProvider()
    else:
        provider = create_llm_provider(
            provider_name,
            model=model_name,
            api_key=args.api_key or None,
            base_url=args.api_base or None,
            max_retries=args.max_retries,
        )
    llm = MultiModelLLM(provider, num_workers=1)
    snapshot = build_paired_snapshot(
        source,
        decision_t=decision_t,
        agent_id=args.agent_id,
        provider=provider_name,
        model=model_name,
    )
    budget = RunBudget(
        BudgetLimits(
            max_calls=5,
            max_prompt_tokens=args.max_prompt_tokens,
            max_completion_tokens=args.max_completion_tokens,
            max_total_tokens=args.max_prompt_tokens + args.max_completion_tokens,
            max_cost_usd=args.max_cost_usd,
            max_elapsed_seconds=600,
        ),
        budget_id=f"{snapshot.snapshot_id}-budget",
    )
    result = run_paired_replay(
        snapshot,
        llm=llm,
        budget=budget,
        max_retries=args.max_retries,
    )
    summary = summarize_paired_replay(result)
    manifest = write_paired_replay_artifacts(
        args.output_dir,
        result,
        budget_snapshot=budget.snapshot().to_dict(),
        provenance={
            "purpose": "bounded prompt-level paired memory replay",
            "source_run": str(args.source_run.resolve()),
            "source_manifest_sha256": hashlib.sha256(
                (args.source_run / "manifest.json").read_bytes()
            ).hexdigest(),
            "provider": provider_name,
            "model": model_name,
            "scientific_evidence": False,
            "environment_keys_present": {
                "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
                "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
                "OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY")),
            },
        },
        git_commit=_git(["rev-parse", "HEAD"]),
        git_dirty=bool(_git(["status", "--porcelain"])),
    )
    return manifest, summary, budget.snapshot().to_dict()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest, summary, budget = execute(args)
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
                "status": "pass" if summary["integrity_verified"] else "fail",
                "manifest": str(manifest),
                "summary": summary,
                "accounted_usage": budget["accounted_usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["integrity_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
