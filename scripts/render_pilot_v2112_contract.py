#!/usr/bin/env python3
"""Render the independently canonicalized V2.11.2 lifecycle repair contract."""

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
    PILOT_CONTRACT_ID_V2_11_2,
    PILOT_CONTRACT_TAG_V2_11_2,
    _v2_11_2_expected_forward_boundary,
    _v2_11_2_expected_model_roles,
    _v2_11_2_expected_non_claims,
    _v2_11_2_expected_recovery_amendment,
    _v2_11_2_expected_stages,
    _validate_v2_1_expected_ci_state,
    canonical_contract_sha256,
    canonical_sha256,
    load_pilot_contract,
)


_EXPECTED_CI_FIELDS = (
    "test_count",
    "test_collection_sha256",
    "compiled_source_count",
    "compiled_source_inventory_sha256",
    "sealed_manifest_inventory_sha256",
)

_SCIENCE_FIELDS_REQUIRED_EQUAL = (
    "schema_version",
    "seeds",
    "provider_profiles",
    "arms",
    "narratives",
    "shocks",
    "utility",
    "stop_go",
    "parameter_dispatch_policy",
    "task_output_contracts",
    "model_roles",
)


def _expected_ci(
    *,
    status: str,
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized = (
        {field: None for field in _EXPECTED_CI_FIELDS} if value is None else dict(value)
    )
    _validate_v2_1_expected_ci_state(
        normalized,
        status=status,
        name="release expected_ci",
    )
    return normalized


def _assert_v2111_science_delta(
    parent: Mapping[str, Any], child: Mapping[str, Any]
) -> None:
    """Fail closed if the repair changes any registered scientific choice."""

    drifted = [
        field
        for field in _SCIENCE_FIELDS_REQUIRED_EQUAL
        if parent.get(field) != child.get(field)
    ]
    if drifted:
        raise ValueError(
            "V2.11.2 changed frozen V2.11.1 scientific fields: " + ", ".join(drifted)
        )

    parent_denominator = dict(parent["denominator_policy"])
    child_denominator = dict(child["denominator_policy"])
    parent_denominator.pop("policy_id")
    child_denominator.pop("policy_id")
    if parent_denominator != child_denominator:
        raise ValueError("V2.11.2 changed the frozen denominator semantics")

    parent_stages = json.loads(json.dumps(parent["stages"]))
    child_stages = json.loads(json.dumps(child["stages"]))
    for row in parent_stages:
        row["budget_bucket"] = "normalized"
    for row in child_stages:
        row["budget_bucket"] = "normalized"
    if parent_stages != child_stages:
        raise ValueError("V2.11.2 changed the frozen 136-cell stage matrix")

    parent_specs = load_pilot_contract(
        ROOT / "experiments" / "pilot_v2_11_1.yaml"
    ).expand()
    if len(parent_specs) != 136 or len({spec.run_id for spec in parent_specs}) != 136:
        raise ValueError("V2.11.1 parent denominator is not the frozen 136 cells")
    if canonical_sha256(parent_stages) != canonical_sha256(child_stages):
        raise ValueError("V2.11.2 normalized stage hash differs from V2.11.1")


def build_contract(
    repo_root: Path,
    *,
    status: str = "draft",
    expected_ci: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build V2.11.2 without mutating or reopening the immutable parent."""

    release_ci = _expected_ci(status=status, value=expected_ci)
    parent = load_pilot_contract(
        repo_root / "experiments" / "pilot_v2_11_1.yaml"
    ).to_dict()
    contract = json.loads(json.dumps(parent))
    contract.pop("v2111_forward_boundary")
    contract.pop("v2111_preflight_bootstrap_amendment")
    contract["contract_id"] = PILOT_CONTRACT_ID_V2_11_2
    contract["status"] = status
    contract["implementation"] = {
        "commit_resolution": "annotated_tag_peel",
        "p0_base_commit": contract["implementation"]["p0_base_commit"],
        "require_clean_worktree": True,
        "required_git_branch": "main",
        "required_git_commit": None,
        "required_git_tag": PILOT_CONTRACT_TAG_V2_11_2,
    }
    contract["model_roles"] = _v2_11_2_expected_model_roles()
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
            "parent_v2111": 18.586399812500005,
            "hosted_v2112": 481.4136001875,
            "manual_reserve": 0.0,
        },
        "total_usd": 500.0,
    }
    contract["denominator_policy"]["policy_id"] = "finevo-pilot-v2.11.2-itt"
    contract["stages"] = _v2_11_2_expected_stages()
    contract["non_claims"] = _v2_11_2_expected_non_claims()
    contract["v2112_forward_boundary"] = _v2_11_2_expected_forward_boundary()
    contract["v2112_recovery_amendment"] = _v2_11_2_expected_recovery_amendment()
    contract["release_requirements"] = {
        "remote": "origin",
        "branch": "main",
        "tag": PILOT_CONTRACT_TAG_V2_11_2,
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
    _assert_v2111_science_delta(parent, contract)
    contract["integrity"]["declared_sha256"] = canonical_contract_sha256(contract)
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", choices=("draft", "frozen"), required=True)
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--test-collection-sha256")
    parser.add_argument("--compiled-source-count", type=int)
    parser.add_argument("--compiled-source-inventory-sha256")
    parser.add_argument("--sealed-manifest-inventory-sha256")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/pilot_v2_11_2.yaml"),
    )
    args = parser.parse_args()
    ci_values = {
        "test_count": args.test_count,
        "test_collection_sha256": args.test_collection_sha256,
        "compiled_source_count": args.compiled_source_count,
        "compiled_source_inventory_sha256": (args.compiled_source_inventory_sha256),
        "sealed_manifest_inventory_sha256": (args.sealed_manifest_inventory_sha256),
    }
    if args.status == "frozen" and any(value is None for value in ci_values.values()):
        raise SystemExit("frozen rendering requires all expected-CI arguments")
    if args.status == "draft" and any(
        value is not None for value in ci_values.values()
    ):
        raise SystemExit("draft rendering requires all expected-CI arguments omitted")
    contract = build_contract(ROOT, status=args.status, expected_ci=ci_values)
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
