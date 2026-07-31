#!/usr/bin/env python3
"""Render the independently canonicalized V2.11.1 bootstrap repair contract."""

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
    PILOT_CONTRACT_ID_V2_11_1,
    PILOT_CONTRACT_TAG_V2_11_1,
    _v2_11_1_expected_forward_boundary,
    _v2_11_1_expected_model_roles,
    _v2_11_1_expected_non_claims,
    _v2_11_1_expected_preflight_bootstrap_amendment,
    _v2_11_1_expected_stages,
    _validate_v2_1_expected_ci_state,
    canonical_contract_sha256,
    load_pilot_contract,
)


_EXPECTED_CI_FIELDS = (
    "test_count",
    "test_collection_sha256",
    "compiled_source_count",
    "compiled_source_inventory_sha256",
    "sealed_manifest_inventory_sha256",
)


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
    """Build V2.11.1 without mutating or expanding the immutable parent."""

    release_ci = _expected_ci(status=status, value=expected_ci)
    contract = load_pilot_contract(
        repo_root / "experiments" / "pilot_v2_11.yaml"
    ).to_dict()
    contract.pop("v211_forward_boundary")
    contract["contract_id"] = PILOT_CONTRACT_ID_V2_11_1
    contract["status"] = status
    contract["implementation"] = {
        "commit_resolution": "annotated_tag_peel",
        "p0_base_commit": contract["implementation"]["p0_base_commit"],
        "require_clean_worktree": True,
        "required_git_branch": "main",
        "required_git_commit": None,
        "required_git_tag": PILOT_CONTRACT_TAG_V2_11_1,
    }
    contract["model_roles"] = _v2_11_1_expected_model_roles()
    contract["arms"]["capability-probe"][
        "execution_mode"
    ] = "capability_authority_import"
    contract["budgets"] = {
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
            "parent_v211": 17.166524062500006,
            "hosted_v2111": 482.8334759375,
            "manual_reserve": 0.0,
        },
        "total_usd": 500.0,
    }
    contract["denominator_policy"][
        "policy_id"
    ] = "finevo-pilot-v2.11.1-itt"
    contract["stages"] = _v2_11_1_expected_stages()
    contract["non_claims"] = _v2_11_1_expected_non_claims()
    contract["v2111_forward_boundary"] = (
        _v2_11_1_expected_forward_boundary()
    )
    contract["v2111_preflight_bootstrap_amendment"] = (
        _v2_11_1_expected_preflight_bootstrap_amendment()
    )
    contract["release_requirements"] = {
        "remote": "origin",
        "branch": "main",
        "tag": PILOT_CONTRACT_TAG_V2_11_1,
        "workflow_file": ".github/workflows/verified-memory-ci.yml",
        "workflow_name": "Verified memory CI",
        "required_job_names": [
            "Python 3.12.7 / ubuntu-24.04",
            "Python 3.12.7 / macos-14",
        ],
        "expected_ci": release_ci,
    }
    contract["integrity"] = {
        "canonicalization": PILOT_CONTRACT_CANONICALIZATION,
        "declared_sha256": "0" * 64,
    }
    contract["integrity"]["declared_sha256"] = canonical_contract_sha256(
        contract
    )
    return contract


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
        default=Path("experiments/pilot_v2_11_1.yaml"),
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
    contract = build_contract(
        ROOT,
        status=args.status,
        expected_ci=ci_values,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(contract["integrity"]["declared_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
