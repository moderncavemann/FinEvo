#!/usr/bin/env python3
"""Render the independently canonicalized V2.11.4 normalization draft."""

from __future__ import annotations

import argparse
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
    PILOT_CONTRACT_ID_V2_11_4,
    PILOT_CONTRACT_TAG_V2_11_4,
    PilotContract,
    _v2_11_4_expected_authority_normalization_amendment,
    _v2_11_4_expected_forward_boundary,
    _v2_11_4_expected_model_roles,
    _v2_11_4_expected_non_claims,
    _v2_11_4_expected_preflight_arm,
    _v2_11_4_expected_stages,
    _validate_v2_1_expected_ci_state,
    canonical_contract_sha256,
    load_pilot_contract,
    science_design_sha256,
)


TRACKED_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_4.yaml"


class FrozenCandidateBootstrapError(ValueError):
    """Raised when the explicit unpinned frozen-candidate boundary is crossed."""


_EXPECTED_CI_FIELDS = (
    "test_count",
    "test_collection_sha256",
    "compiled_source_count",
    "compiled_source_inventory_sha256",
    "sealed_manifest_inventory_sha256",
)

_SCIENCE_STAGE_IDS = frozenset(
    {
        "experiment-c",
        "experiment-a",
        "experiment-d",
        "experiment-b",
        "cross-model",
    }
)

_SCIENCE_FIELDS_REQUIRED_EQUAL = (
    "schema_version",
    "seeds",
    "provider_profiles",
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


def _science_arms(value: Mapping[str, Any]) -> dict[str, Any]:
    operational = {"parent-import", "capability-probe", "closed-loop-preflight"}
    return {
        arm_id: row
        for arm_id, row in dict(value["arms"]).items()
        if arm_id not in operational
    }


def _science_stages(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = json.loads(json.dumps(value["stages"]))
    result = []
    for row in rows:
        if row["stage_id"] not in _SCIENCE_STAGE_IDS:
            continue
        row["budget_bucket"] = "normalized-hosted-science"
        result.append(row)
    return result


def _normalized_science_specs(contract: PilotContract) -> list[dict[str, Any]]:
    result = []
    for spec in contract.expand():
        if spec.stage_id not in _SCIENCE_STAGE_IDS:
            continue
        row = spec.to_dict()
        row.pop("run_id")
        row.pop("contract_id")
        row["budget_bucket"] = "normalized-hosted-science"
        result.append(row)
    return result


def _assert_v2113_science_delta(
    parent: Mapping[str, Any], child: Mapping[str, Any]
) -> None:
    """Fail closed if normalization changes any registered science choice."""

    drifted = [
        field
        for field in _SCIENCE_FIELDS_REQUIRED_EQUAL
        if parent.get(field) != child.get(field)
    ]
    if drifted:
        raise ValueError(
            "V2.11.4 changed frozen V2.11.3 scientific fields: " + ", ".join(drifted)
        )
    if _science_arms(parent) != _science_arms(child):
        raise ValueError("V2.11.4 changed frozen V2.11.3 scientific arms")
    if _science_stages(parent) != _science_stages(child):
        raise ValueError("V2.11.4 changed frozen V2.11.3 scientific stages")

    parent_denominator = dict(parent["denominator_policy"])
    child_denominator = dict(child["denominator_policy"])
    parent_denominator.pop("policy_id")
    child_denominator.pop("policy_id")
    if parent_denominator != child_denominator:
        raise ValueError("V2.11.4 changed the frozen denominator semantics")

    expected_operational_modes = {
        "parent-import": "parent_authority_import",
        "capability-gate": "capability_authority_import",
        "long-context-preflight": "preflight_authority_import",
    }
    actual_operational_modes = {
        row["stage_id"]: row["cells"][0]["execution_mode"]
        for row in child["stages"]
        if row["stage_id"] in expected_operational_modes
    }
    if actual_operational_modes != expected_operational_modes:
        raise ValueError("V2.11.4 operational authority modes drifted")
    if child["arms"]["closed-loop-preflight"] != (_v2_11_4_expected_preflight_arm()):
        raise ValueError("V2.11.4 preflight authority arm drifted")


def _assert_expanded_science_specs_match(
    parent: PilotContract, child: PilotContract
) -> None:
    parent_specs = parent.expand()
    child_specs = child.expand()
    if len(parent_specs) != 136 or len(child_specs) != 136:
        raise ValueError("V2.11.3/V2.11.4 denominator is not exactly 136 cells")
    if len({spec.run_id for spec in child_specs}) != 136:
        raise ValueError("V2.11.4 denominator contains duplicate run IDs")
    if _normalized_science_specs(parent) != _normalized_science_specs(child):
        raise ValueError("V2.11.4 expanded scientific run specs drifted")


def _parse_with_bootstrap_design_pin(contract: Mapping[str, Any]) -> PilotContract:
    """Permit only the first deterministic draft to reveal its design digest."""

    pinned = pilot_contract_module.PILOT_CONTRACT_V2_11_4_SCIENCE_DESIGN_SHA256
    if pinned is not None:
        return PilotContract.from_dict(contract)
    if contract.get("status") != "draft":
        raise FrozenCandidateBootstrapError(
            "V2.11.4 science-design pin must be set before frozen rendering"
        )
    candidate = science_design_sha256(contract)
    pilot_contract_module.PILOT_CONTRACT_V2_11_4_SCIENCE_DESIGN_SHA256 = candidate
    try:
        return PilotContract.from_dict(contract)
    finally:
        pilot_contract_module.PILOT_CONTRACT_V2_11_4_SCIENCE_DESIGN_SHA256 = pinned


def build_contract(
    repo_root: Path,
    *,
    status: str = "draft",
    expected_ci: Mapping[str, Any] | None = None,
    frozen_candidate: bool = False,
) -> dict[str, Any]:
    """Build V2.11.4 without reopening the immutable V2.11.3 denominator."""

    if frozen_candidate and status != "frozen":
        raise FrozenCandidateBootstrapError(
            "frozen-candidate mode requires status=frozen"
        )

    release_ci = _expected_ci(status=status, value=expected_ci)
    parent_contract = load_pilot_contract(
        repo_root / "experiments" / "pilot_v2_11_3.yaml"
    )
    parent = parent_contract.to_dict()
    contract = json.loads(json.dumps(parent))
    contract.pop("v2113_forward_boundary")
    contract.pop("v2113_consumer_adapter_amendment")
    contract["contract_id"] = PILOT_CONTRACT_ID_V2_11_4
    contract["status"] = status
    contract["implementation"] = {
        "commit_resolution": "annotated_tag_peel",
        "p0_base_commit": contract["implementation"]["p0_base_commit"],
        "require_clean_worktree": True,
        "required_git_branch": "main",
        "required_git_commit": None,
        "required_git_tag": PILOT_CONTRACT_TAG_V2_11_4,
    }
    contract["arms"]["closed-loop-preflight"] = _v2_11_4_expected_preflight_arm()
    contract["model_roles"] = _v2_11_4_expected_model_roles()
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
            "parent_v2113": 19.998220562500006,
            "hosted_v2114": 480.0017794375,
            "manual_reserve": 0.0,
        },
        "total_usd": 500.0,
    }
    contract["denominator_policy"]["policy_id"] = "finevo-pilot-v2.11.4-itt"
    contract["stages"] = _v2_11_4_expected_stages()
    contract["non_claims"] = _v2_11_4_expected_non_claims()
    contract["v2114_forward_boundary"] = _v2_11_4_expected_forward_boundary(
        status=status
    )
    contract["v2114_authority_normalization_amendment"] = (
        _v2_11_4_expected_authority_normalization_amendment()
    )
    contract["release_requirements"] = {
        "remote": "origin",
        "branch": "main",
        "tag": PILOT_CONTRACT_TAG_V2_11_4,
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
    _assert_v2113_science_delta(parent, contract)
    contract["integrity"]["declared_sha256"] = canonical_contract_sha256(contract)

    if frozen_candidate:
        if (
            pilot_contract_module.PILOT_V2_11_4_SOURCE_MANIFEST_FILE_SHA256 is None
            or pilot_contract_module.PILOT_V2_11_4_SOURCE_MANIFEST_CONTENT_SHA256
            is None
        ):
            raise FrozenCandidateBootstrapError(
                "V2.11.4 source manifest must be sealed before frozen candidate"
            )
        pinned = pilot_contract_module.PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256
        if pinned is not None:
            raise FrozenCandidateBootstrapError(
                "V2.11.4 canonical hash is already pinned; use normal frozen mode"
            )
        candidate_hash = contract["integrity"]["declared_sha256"]
        pilot_contract_module.PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256 = candidate_hash
        try:
            parsed = PilotContract.from_dict(contract)
        finally:
            pilot_contract_module.PILOT_CONTRACT_V2_11_4_CANONICAL_SHA256 = pinned
        if (
            canonical_contract_sha256(contract) != candidate_hash
            or parsed.to_dict() != contract
        ):
            raise FrozenCandidateBootstrapError(
                "frozen candidate failed canonical round-trip self-validation"
            )
    else:
        parsed = _parse_with_bootstrap_design_pin(contract)
    _assert_expanded_science_specs_match(parent_contract, parsed)
    return contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", choices=("draft", "frozen"), required=True)
    parser.add_argument(
        "--frozen-candidate",
        "--bootstrap-frozen-candidate",
        dest="frozen_candidate",
        action="store_true",
        help=(
            "self-validate one unpinned frozen candidate; it must be written "
            "outside the tracked contract path"
        ),
    )
    parser.add_argument("--test-count", type=int)
    parser.add_argument("--test-collection-sha256")
    parser.add_argument("--compiled-source-count", type=int)
    parser.add_argument("--compiled-source-inventory-sha256")
    parser.add_argument("--sealed-manifest-inventory-sha256")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/pilot_v2_11_4.yaml"),
    )
    args = parser.parse_args(argv)
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
    if args.frozen_candidate and args.status != "frozen":
        raise SystemExit("--frozen-candidate requires --status frozen")
    output = args.output.resolve(strict=False)
    if args.frozen_candidate and output == TRACKED_CONTRACT_PATH.resolve(strict=False):
        raise SystemExit(
            "an unpinned frozen candidate must not overwrite the tracked contract"
        )
    contract = build_contract(
        ROOT,
        status=args.status,
        expected_ci=ci_values,
        frozen_candidate=args.frozen_candidate,
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
