#!/usr/bin/env python3
"""Run one bounded, non-scientific FinEvo provider interface probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from llm_providers import SAFE_PROVIDER_ERROR_TYPES
from verified_memory.provider_diagnostics import (
    DEFAULT_INTERFACE_PROBE_MAX_TOKENS,
    run_provider_interface_probe,
)


ROOT = Path(__file__).resolve().parent


def _safe_error_type(exc: BaseException) -> str:
    candidate = type(exc).__name__
    return candidate if candidate in SAFE_PROVIDER_ERROR_TYPES else "ProviderError"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "experiments" / "pilot_v1.yaml",
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--required-tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_INTERFACE_PROBE_MAX_TOKENS,
        help=(
            "completion-token cap including provider reasoning tokens "
            f"(default: {DEFAULT_INTERFACE_PROBE_MAX_TOKENS})"
        ),
    )
    parser.add_argument("--max-cost-usd", type=float, default=0.05)
    parser.add_argument(
        "--force-json-object",
        action="store_true",
        help="diagnostic-only override for a prompt-only local profile",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_provider_interface_probe(
            contract_path=args.contract,
            model_id=args.model_id,
            output_path=args.output,
            repo_root=ROOT,
            required_tag=args.required_tag,
            max_tokens=args.max_tokens,
            max_cost_usd=args.max_cost_usd,
            force_json_object=args.force_json_object,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed-before-or-during-probe",
                    "error_type": _safe_error_type(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "diagnostic_only": True,
                "scientific_evidence": False,
                "receipt": str(args.output),
                "receipt_sha256": result["receipt_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
