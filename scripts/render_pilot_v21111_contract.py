#!/usr/bin/env python3
"""Render the prospective V2.11.11 fresh-seed recovery cohort contract."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import verified_memory.pilot_contract as pilot_contract_module
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_CANONICALIZATION,
    PILOT_CONTRACT_ID_V2_11_11,
    PILOT_CONTRACT_TAG_V2_11_11,
    PilotContract,
    _v2_11_11_expected_fresh_cohort_boundary,
    _v2_11_11_expected_dispatch_refresh,
    _v2_11_11_expected_model_roles,
    _v2_11_11_expected_non_claims,
    _v2_11_11_expected_parent_import_arm,
    _v2_11_11_expected_seed_generation,
    _v2_11_11_expected_stages,
    _validate_v2_1_expected_ci_state,
    canonical_contract_sha256,
    load_pilot_contract,
    science_design_sha256,
)


TRACKED_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_11.yaml"
FRESH_MAIN_SEEDS = (
    877_361,
    1_410_637_959,
    416_755_402,
    357_136_200,
    1_541_219_789,
)


class FrozenCandidateBootstrapError(ValueError):
    """Raised when an unpinned frozen candidate crosses the draft boundary."""


def _expected_ci(
    *,
    status: str,
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    fields = (
        "test_count",
        "test_collection_sha256",
        "compiled_source_count",
        "compiled_source_inventory_sha256",
        "sealed_manifest_inventory_sha256",
    )
    normalized = {field: None for field in fields} if value is None else dict(value)
    _validate_v2_1_expected_ci_state(
        normalized,
        status=status,
        name="release expected_ci",
    )
    return normalized


def _parse_with_bootstrap_design_pin(contract: Mapping[str, Any]) -> PilotContract:
    pinned = pilot_contract_module.PILOT_CONTRACT_V2_11_11_SCIENCE_DESIGN_SHA256
    if pinned is not None:
        return PilotContract.from_dict(contract)
    if contract.get("status") != "draft":
        raise FrozenCandidateBootstrapError(
            "V2.11.11 science-design pin must be set before frozen rendering"
        )
    candidate = science_design_sha256(contract)
    pilot_contract_module.PILOT_CONTRACT_V2_11_11_SCIENCE_DESIGN_SHA256 = candidate
    try:
        return PilotContract.from_dict(contract)
    finally:
        pilot_contract_module.PILOT_CONTRACT_V2_11_11_SCIENCE_DESIGN_SHA256 = pinned


def _assert_fresh_cohort(contract: PilotContract) -> None:
    expected_counts = {
        "parent-import": 1,
        "experiment-b": 25,
        "experiment-d": 55,
        "cross-model": 6,
    }
    observed_counts = {
        stage_id: len(contract.expand(stage=stage_id))
        for stage_id in contract.stage_ids
    }
    if observed_counts != expected_counts:
        raise ValueError(f"V2.11.11 exact stage denominator drifted: {observed_counts}")
    specs = contract.expand()
    if len(specs) != 87 or len({spec.run_id for spec in specs}) != 87:
        raise ValueError("V2.11.11 denominator must contain 87 unique cells")
    science = tuple(spec for spec in specs if spec.stage_id != "parent-import")
    if len(science) != 86:
        raise ValueError("V2.11.11 must contain exactly 86 scientific cells")
    observed_seeds = {
        spec.environment_seed
        for spec in science
        if spec.stage_id in {"experiment-b", "experiment-d"}
    }
    if observed_seeds != set(FRESH_MAIN_SEEDS):
        raise ValueError("V2.11.11 scientific seed registry drifted")
    old = {1099057501, 1421875452, 1769977770, 959809858, 617806385}
    if observed_seeds & old:
        raise ValueError("V2.11.11 fresh cohort overlaps an old scientific seed")
    stages = {stage.stage_id: stage for stage in contract.stages}
    if any(
        stages[stage_id].prerequisites != ("parent-import",)
        for stage_id in ("experiment-b", "experiment-d", "cross-model")
    ):
        raise ValueError("V2.11.11 scientific stages are not independent")
    if {stage.stage_id: stage.evidence_class for stage in contract.stages} != {
        "parent-import": "operational",
        "experiment-b": "scientific",
        "experiment-d": "scientific",
        "cross-model": "scientific",
    }:
        raise ValueError("V2.11.11 evidence-class partition drifted")
    calls = contract.v21111_fresh_cohort_boundary["fresh_cohort"]["calls_by_stage"]
    if (
        dict(calls)
        != {
            "experiment-b": 1440,
            "experiment-d": 1480,
            "cross-model": 336,
        }
        or sum(calls.values()) != 3256
    ):
        raise ValueError("V2.11.11 simulated call denominator drifted")


def build_contract(
    repo_root: Path,
    *,
    status: str = "draft",
    expected_ci: Mapping[str, Any] | None = None,
    frozen_candidate: bool = False,
) -> dict[str, Any]:
    """Build a fresh cohort without mutating or importing V2.11.10 effects."""

    if frozen_candidate and status != "frozen":
        raise FrozenCandidateBootstrapError(
            "frozen-candidate mode requires status=frozen"
        )
    release_ci = _expected_ci(status=status, value=expected_ci)
    parent = load_pilot_contract(
        repo_root / "experiments" / "pilot_v2_11_10.yaml"
    ).to_dict()
    contract = deepcopy(parent)
    contract.pop("v21110_recovery_boundary")
    contract["contract_id"] = PILOT_CONTRACT_ID_V2_11_11
    contract["status"] = status
    contract["implementation"] = {
        "commit_resolution": "annotated_tag_peel",
        "p0_base_commit": parent["implementation"]["p0_base_commit"],
        "require_clean_worktree": True,
        "required_git_branch": "main",
        "required_git_commit": None,
        "required_git_tag": PILOT_CONTRACT_TAG_V2_11_11,
    }
    contract["seeds"] = deepcopy(parent["seeds"])
    contract["seeds"]["generation"] = _v2_11_11_expected_seed_generation()
    contract["seeds"]["sets"]["main"] = list(FRESH_MAIN_SEEDS)
    contract["seeds"]["sets"]["cross-model"] = list(FRESH_MAIN_SEEDS[:3])
    contract["arms"]["parent-import"] = _v2_11_11_expected_parent_import_arm()
    contract["model_roles"] = _v2_11_11_expected_model_roles()
    contract["task_output_contracts"]["actor-action"]["max_completion_tokens"] = 8192
    refreshed_prices = {
        "gpt52_main": {
            "input": 1.75,
            "cached_input": 0.175,
            "output": 14.0,
            "model_reference": (
                "https://developers.openai.com/api/docs/models/gpt-5.2"
            ),
        },
        "gpt56_diagnostic": {
            "input": 5.0,
            "cached_input": 0.5,
            "output": 30.0,
            "model_reference": (
                "https://developers.openai.com/api/docs/models/gpt-5.6-sol"
            ),
        },
    }
    for profile_id, pricing in refreshed_prices.items():
        profile = contract["provider_profiles"][profile_id]
        profile["service_tier"] = "default"
        profile["short_context_prompt_token_ceiling"] = 272_000
        snapshot = profile["price_snapshot"]
        snapshot.update(
            {
                "captured_at": "2026-08-02",
                "source": "https://developers.openai.com/api/docs/pricing",
                "model_reference": pricing["model_reference"],
                "catalog_input": pricing["input"],
                "catalog_cached_input": pricing["cached_input"],
                "catalog_output": pricing["output"],
                "endpoint_input": pricing["input"],
                "endpoint_cached_input": pricing["cached_input"],
                "endpoint_output": pricing["output"],
            }
        )
    refresh_reserve = float(_v2_11_11_expected_dispatch_refresh()["reserved_cost_usd"])
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
            "parent_v21110": 78.3237413125,
            "dispatch_refresh": refresh_reserve,
            "hosted_v21111": 404.46984,
            "unallocated_headroom": round(
                500.0 - 78.3237413125 - refresh_reserve - 404.46984,
                12,
            ),
            "manual_reserve": 0.0,
        },
        "total_usd": 500.0,
    }
    contract["denominator_policy"]["policy_id"] = "finevo-pilot-v2.11.11-itt"
    contract["stages"] = _v2_11_11_expected_stages()
    contract["non_claims"] = _v2_11_11_expected_non_claims()
    contract["v21111_fresh_cohort_boundary"] = _v2_11_11_expected_fresh_cohort_boundary(
        status=status
    )
    contract["release_requirements"] = {
        "remote": "origin",
        "branch": "main",
        "tag": PILOT_CONTRACT_TAG_V2_11_11,
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
    contract["integrity"]["declared_sha256"] = canonical_contract_sha256(contract)
    contract["integrity"]["declared_sha256"] = canonical_contract_sha256(contract)

    if frozen_candidate:
        if (
            pilot_contract_module.PILOT_V2_11_11_SOURCE_MANIFEST_FILE_SHA256 is None
            or pilot_contract_module.PILOT_V2_11_11_SOURCE_MANIFEST_CONTENT_SHA256
            is None
        ):
            raise FrozenCandidateBootstrapError(
                "V2.11.11 source manifest must be sealed before frozen candidate"
            )
        pinned = pilot_contract_module.PILOT_CONTRACT_V2_11_11_CANONICAL_SHA256
        if pinned is not None:
            raise FrozenCandidateBootstrapError(
                "V2.11.11 canonical hash is already pinned; use normal frozen mode"
            )
        candidate_hash = contract["integrity"]["declared_sha256"]
        pilot_contract_module.PILOT_CONTRACT_V2_11_11_CANONICAL_SHA256 = candidate_hash
        try:
            parsed = PilotContract.from_dict(contract)
        finally:
            pilot_contract_module.PILOT_CONTRACT_V2_11_11_CANONICAL_SHA256 = pinned
    else:
        parsed = _parse_with_bootstrap_design_pin(contract)
    _assert_fresh_cohort(parsed)
    if parsed.to_dict() != contract:
        raise ValueError("V2.11.11 contract failed canonical round trip")
    return contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", choices=("draft", "frozen"), required=True)
    parser.add_argument(
        "--frozen-candidate",
        action="store_true",
        help="self-validate one unpinned frozen candidate outside the tracked path",
    )
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--test-collection-sha256")
    parser.add_argument("--compiled-source-count", type=int)
    parser.add_argument("--compiled-source-inventory-sha256")
    parser.add_argument("--sealed-manifest-inventory-sha256")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/pilot_v2_11_11.yaml"),
    )
    args = parser.parse_args(argv)
    ci_values = {
        "test_count": args.test_count,
        "test_collection_sha256": args.test_collection_sha256,
        "compiled_source_count": args.compiled_source_count,
        "compiled_source_inventory_sha256": args.compiled_source_inventory_sha256,
        "sealed_manifest_inventory_sha256": args.sealed_manifest_inventory_sha256,
    }
    if args.status == "frozen" and any(value is None for value in ci_values.values()):
        raise SystemExit("frozen rendering requires all expected-CI arguments")
    if args.status == "draft" and any(
        value is not None for value in ci_values.values()
    ):
        raise SystemExit("draft rendering requires expected-CI arguments omitted")
    output = args.output.resolve(strict=False)
    if args.frozen_candidate and output == TRACKED_CONTRACT_PATH.resolve(strict=False):
        raise SystemExit("an unpinned frozen candidate cannot overwrite the contract")
    contract = build_contract(
        ROOT,
        status=args.status,
        expected_ci=ci_values,
        frozen_candidate=args.frozen_candidate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(contract, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(contract["integrity"]["declared_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
