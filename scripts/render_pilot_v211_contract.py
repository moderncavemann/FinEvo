#!/usr/bin/env python3
"""Render the independently expanded prospective V2.11 pilot contract.

The script copies stable, already-typed primitives (arms, shocks, utilities,
request-dispatch policy, and exact provider profile shapes) from the latest
readable contracts.  It deliberately discards the V2.1--V2.10.2 amendment
chain and replaces the stage matrix, model roles, budget, output limits,
lineage boundary, non-claims, and release identity with the prospective V2.11
registration.  The resulting JSON-compatible YAML is self-contained at
runtime; loading it never expands or executes a parent overlay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verified_memory.pilot_contract import (
    PILOT_CONTRACT_CANONICALIZATION,
    PILOT_CONTRACT_ID_V2_11,
    PILOT_CONTRACT_TAG_V2_11,
    _SCIENCE_TASK_CAPS_V2_11,
    _v2_11_expected_forward_boundary,
    _v2_11_expected_model_roles,
    _v2_11_expected_non_claims,
    _v2_11_expected_stages,
    _validate_v2_1_expected_ci_state,
    canonical_contract_sha256,
    load_pilot_contract,
)


_AMENDMENT_FIELDS = {
    "operational_amendment",
    "evaluator_amendment",
    "preflight_bootstrap_amendment",
    "matrix_amendment",
    "parent_import_retry_amendment",
    "p95_authority_retry_amendment",
    "stage0_evaluator_retry_amendment",
    "qref_identity_retry_amendment",
    "qref_summary_equivalence_amendment",
    "p95_runner_binding_retry_amendment",
    "qref_receipt_verifier_retry_amendment",
    "p95_consumer_adapter_retry_amendment",
}

_EXPECTED_CI_FIELDS = (
    "test_count",
    "test_collection_sha256",
    "compiled_source_count",
    "compiled_source_inventory_sha256",
    "sealed_manifest_inventory_sha256",
)


def _task_output_contracts(source: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(source))
    for task_id, (completion_cap, visible_byte_cap) in (
        _SCIENCE_TASK_CAPS_V2_11.items()
    ):
        result[task_id]["max_completion_tokens"] = completion_cap
        result[task_id]["max_visible_json_bytes"] = visible_byte_cap
    return result


def _expected_ci(
    *,
    status: str,
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized = (
        {field: None for field in _EXPECTED_CI_FIELDS}
        if value is None
        else dict(value)
    )
    _validate_v2_1_expected_ci_state(
        normalized,
        status=status,
        name="release expected_ci",
    )
    return normalized


def build_contract(
    repo_root: Path,
    *,
    status: str = "draft",
    expected_ci: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    release_ci = _expected_ci(status=status, value=expected_ci)
    parent = load_pilot_contract(
        repo_root / "experiments" / "pilot_v2_10_2.yaml"
    ).to_dict()
    v23 = load_pilot_contract(
        repo_root / "experiments" / "pilot_v2_3.yaml"
    ).to_dict()
    for field in _AMENDMENT_FIELDS:
        parent.pop(field, None)

    parent["contract_id"] = PILOT_CONTRACT_ID_V2_11
    parent["status"] = status
    parent["implementation"] = {
        "commit_resolution": "annotated_tag_peel",
        "p0_base_commit": parent["implementation"]["p0_base_commit"],
        "require_clean_worktree": True,
        "required_git_branch": "main",
        "required_git_commit": None,
        "required_git_tag": PILOT_CONTRACT_TAG_V2_11,
    }

    gpt52 = parent["provider_profiles"]["gpt52_main"]
    gpt56 = v23["provider_profiles"]["gpt56_diagnostic"]
    gpt56["price_snapshot"]["captured_at"] = "2026-07-31"
    qref = parent["provider_profiles"]["qref_scripted"]
    parent["provider_profiles"] = {
        "gpt52_main": gpt52,
        "gpt56_diagnostic": gpt56,
        "qref_scripted": qref,
    }
    parent["model_roles"] = _v2_11_expected_model_roles()
    parent["task_output_contracts"] = _task_output_contracts(
        parent["task_output_contracts"]
    )

    parent["arms"]["closed-loop-preflight"]["parameters"].update(
        {
            "action_calls_expected": 24,
            "episode_length": 12,
            "num_agents": 2,
            "semantic_proposals_expected": 8,
        }
    )
    parent["budgets"] = {
        "automatic_reserve_usd": 0.0,
        "completion_scope": "hosted-api-only",
        "max_provider_completions": 7500,
        "max_storage_bytes": 5_000_000_000,
        "pre_dispatch_projection": {
            "basis": "model-by-call-role preflight p95",
            "over_budget_policy": "no-go-no-matrix-shrink",
            "required": True,
            "reserve_multiplier": 1.25,
            "unknown_price_policy": "stop-before-dispatch",
        },
        "stage_usd_caps": {
            "parent_v2102": 16.044922812500005,
            "hosted_v211": 483.9550771875,
            "manual_reserve": 0.0,
        },
        "total_usd": 500.0,
    }
    parent["denominator_policy"][
        "policy_id"
    ] = "finevo-pilot-v2.11-itt"
    parent["stop_go"]["closed_loop_preflight"] = {
        "action_parse_success": "24/24",
        "semantic_proposal_outcomes_accounted": "8/8",
        "provider_rows": 32,
        "clipping_count": 0,
        "provider_failure_count": 0,
        "route_metadata_complete": True,
        "usage_metadata_complete": True,
        "cost_metadata_complete": True,
        "finish_reason_stop_required": True,
        "served_model_exact": True,
        "provider_pin_exact": True,
        "fallback_observed": False,
        "attempts_per_request": 1,
        "prompt_token_tier_ceiling": 200000,
    }
    parent["stop_go"]["cross_model"] = {
        "reportable_complete_pairs_min": 2,
        "total_registered_pairs": 3,
        "direction_replication_complete_pairs": 3,
        "direction_replication_requires_capability_pass": True,
        "seed_unsupported_directional_replication_requires_registered_matched_a_a_null": (
            False
        ),
        "missing_matched_a_a_null_action": (
            "directional-only-no-model-specific-repeatability-null"
        ),
    }
    parent["stages"] = _v2_11_expected_stages()
    parent["non_claims"] = _v2_11_expected_non_claims()
    parent["v211_forward_boundary"] = _v2_11_expected_forward_boundary()
    parent["release_requirements"] = {
        "remote": "origin",
        "branch": "main",
        "tag": PILOT_CONTRACT_TAG_V2_11,
        "workflow_file": ".github/workflows/verified-memory-ci.yml",
        "workflow_name": "Verified memory CI",
        "required_job_names": [
            "Python 3.12.7 / ubuntu-24.04",
            "Python 3.12.7 / macos-14",
        ],
        "expected_ci": release_ci,
    }
    parent["integrity"] = {
        "canonicalization": PILOT_CONTRACT_CANONICALIZATION,
        "declared_sha256": "0" * 64,
    }
    parent["integrity"]["declared_sha256"] = canonical_contract_sha256(parent)
    return parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        choices=("draft", "frozen"),
        required=True,
    )
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--test-collection-sha256")
    parser.add_argument("--compiled-source-count", type=int)
    parser.add_argument("--compiled-source-inventory-sha256")
    parser.add_argument("--sealed-manifest-inventory-sha256")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/pilot_v2_11.yaml"),
    )
    args = parser.parse_args()
    ci_values = {
        "test_count": args.test_count,
        "test_collection_sha256": args.test_collection_sha256,
        "compiled_source_count": args.compiled_source_count,
        "compiled_source_inventory_sha256": (
            args.compiled_source_inventory_sha256
        ),
        "sealed_manifest_inventory_sha256": (
            args.sealed_manifest_inventory_sha256
        ),
    }
    if args.status == "frozen" and any(
        value is None for value in ci_values.values()
    ):
        raise SystemExit(
            "frozen rendering requires all expected-CI arguments"
        )
    if args.status == "draft" and any(
        value is not None for value in ci_values.values()
    ):
        raise SystemExit(
            "draft rendering requires all expected-CI arguments omitted"
        )
    repo_root = ROOT
    contract = build_contract(
        repo_root,
        status=args.status,
        expected_ci=ci_values,
    )
    output = args.output
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            contract,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(contract["integrity"]["declared_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
