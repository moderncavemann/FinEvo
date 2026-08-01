#!/usr/bin/env python3
"""Render the prospective V2.11.10 P95-authority-layer recovery contract."""

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
    PILOT_CONTRACT_ID_V2_11_10,
    PILOT_CONTRACT_TAG_V2_11_10,
    PilotContract,
    _v2_11_10_expected_model_roles,
    _v2_11_10_expected_non_claims,
    _v2_11_10_expected_parent_import_arm,
    _v2_11_10_expected_recovery_boundary,
    _v2_11_10_expected_stages,
    _validate_v2_1_expected_ci_state,
    canonical_contract_sha256,
    canonical_sha256,
    load_pilot_contract,
    science_design_sha256,
)


TRACKED_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_10.yaml"


class FrozenCandidateBootstrapError(ValueError):
    """Raised when the explicit unpinned frozen-candidate boundary is crossed."""


_EXPECTED_CI_FIELDS = (
    "test_count",
    "test_collection_sha256",
    "compiled_source_count",
    "compiled_source_inventory_sha256",
    "sealed_manifest_inventory_sha256",
)

_CONTINUATION_STAGE_IDS = (
    "experiment-d",
    "experiment-b",
    "cross-model",
)

_CONTINUATION_FIELDS_REQUIRED_EQUAL = (
    "schema_version",
    "seeds",
    "provider_profiles",
    "narratives",
    "shocks",
    "utility",
    "stop_go",
    "parameter_dispatch_policy",
    "task_output_contracts",
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


def _used_continuation_arm_ids(contract: PilotContract) -> set[str]:
    return {
        spec.arm_id
        for spec in contract.expand()
        if spec.stage_id in _CONTINUATION_STAGE_IDS
    }


def _normalized_continuation_specs(contract: PilotContract) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for spec in contract.expand():
        if spec.stage_id not in _CONTINUATION_STAGE_IDS:
            continue
        row = spec.to_dict()
        row.pop("run_id")
        row.pop("contract_id")
        row["budget_bucket"] = "normalized-hosted-continuation"
        result.append(row)
    return result


def _assert_v2119_recovery_delta(
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
    *,
    parsed_child: PilotContract | None = None,
) -> None:
    """Fail closed if the successor changes any remaining science choice."""

    drifted = [
        field
        for field in _CONTINUATION_FIELDS_REQUIRED_EQUAL
        if parent.get(field) != child.get(field)
    ]
    if drifted:
        raise ValueError(
            "V2.11.10 changed frozen V2.11.9 continuation fields: "
            + ", ".join(drifted)
        )

    parent_contract = PilotContract.from_dict(parent)
    child_contract = (
        _parse_with_bootstrap_design_pin(child)
        if parsed_child is None
        else parsed_child
    )
    arm_ids = _used_continuation_arm_ids(parent_contract)
    if {key: parent["arms"][key] for key in arm_ids} != {
        key: child["arms"][key] for key in arm_ids
    }:
        raise ValueError("V2.11.10 changed a remaining V2.11.9 scientific arm")
    if child["arms"]["parent-import"] != _v2_11_10_expected_parent_import_arm():
        raise ValueError("V2.11.10 recovery import semantics drifted")

    parent_roles = json.loads(json.dumps(parent["model_roles"]))
    child_roles = json.loads(json.dumps(child["model_roles"]))
    for role in parent_roles.values():
        role.pop("allowed_stages")
    for role in child_roles.values():
        role.pop("allowed_stages")
    if parent_roles != child_roles:
        raise ValueError("V2.11.10 changed a model role outside its stage allowlist")
    if child["model_roles"] != _v2_11_10_expected_model_roles():
        raise ValueError("V2.11.10 continuation stage allowlists drifted")

    parent_denominator = dict(parent["denominator_policy"])
    child_denominator = dict(child["denominator_policy"])
    parent_denominator.pop("policy_id")
    child_denominator.pop("policy_id")
    if parent_denominator != child_denominator:
        raise ValueError("V2.11.10 changed the frozen denominator semantics")

    if _normalized_continuation_specs(parent_contract) != (
        _normalized_continuation_specs(child_contract)
    ):
        raise ValueError("V2.11.10 changed a normalized remaining run spec")


def _assert_expanded_continuation_specs_match(
    parent: PilotContract,
    child: PilotContract,
) -> None:
    parent_rows = _normalized_continuation_specs(parent)
    child_rows = _normalized_continuation_specs(child)
    child_specs = child.expand()
    if len(parent.expand()) != 87:
        raise ValueError("V2.11.9 no-go denominator is not exactly 87 cells")
    if len(parent_rows) != 86 or len(child_rows) != 86:
        raise ValueError("V2.11.10 must map exactly 86 remaining science cells")
    if len(child_specs) != 87 or len({spec.run_id for spec in child_specs}) != 87:
        raise ValueError("V2.11.10 denominator is not 87 unique cells")
    if parent_rows != child_rows:
        raise ValueError("V2.11.10 normalized continuation specs drifted")
    if canonical_sha256(parent_rows) != (
        "9968bb55b9c56ced90f56826bc8e186f72299e0a8bb40dfdb4fbb1e637af1632"
    ):
        raise ValueError("V2.11.9 normalized continuation digest drifted")
    import_specs = [spec for spec in child_specs if spec.stage_id == "parent-import"]
    if len(import_specs) != 1 or import_specs[0].execution_mode != (
        "parent_authority_import"
    ):
        raise ValueError("V2.11.10 must contain one zero-provider import cell")


def _parse_with_bootstrap_design_pin(contract: Mapping[str, Any]) -> PilotContract:
    """Permit only the first deterministic draft to reveal its design digest."""

    pinned = pilot_contract_module.PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256
    if pinned is not None:
        return PilotContract.from_dict(contract)
    if contract.get("status") != "draft":
        raise FrozenCandidateBootstrapError(
            "V2.11.10 science-design pin must be set before frozen rendering"
        )
    candidate = science_design_sha256(contract)
    pilot_contract_module.PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256 = candidate
    try:
        return PilotContract.from_dict(contract)
    finally:
        pilot_contract_module.PILOT_CONTRACT_V2_11_10_SCIENCE_DESIGN_SHA256 = pinned


def build_contract(
    repo_root: Path,
    *,
    status: str = "draft",
    expected_ci: Mapping[str, Any] | None = None,
    frozen_candidate: bool = False,
) -> dict[str, Any]:
    """Build V2.11.10 without reopening the immutable V2.11.9 namespace."""

    if frozen_candidate and status != "frozen":
        raise FrozenCandidateBootstrapError(
            "frozen-candidate mode requires status=frozen"
        )

    release_ci = _expected_ci(status=status, value=expected_ci)
    parent_contract = load_pilot_contract(
        repo_root / "experiments" / "pilot_v2_11_9.yaml"
    )
    parent = parent_contract.to_dict()
    contract = json.loads(json.dumps(parent))
    contract.pop("v2119_recovery_boundary")
    contract["contract_id"] = PILOT_CONTRACT_ID_V2_11_10
    contract["status"] = status
    contract["implementation"] = {
        "commit_resolution": "annotated_tag_peel",
        "p0_base_commit": contract["implementation"]["p0_base_commit"],
        "require_clean_worktree": True,
        "required_git_branch": "main",
        "required_git_commit": None,
        "required_git_tag": PILOT_CONTRACT_TAG_V2_11_10,
    }
    contract["arms"]["parent-import"] = _v2_11_10_expected_parent_import_arm()
    contract["model_roles"] = _v2_11_10_expected_model_roles()
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
            "parent_v2119": 63.1196450625,
            "hosted_v21110": 436.8803549375,
            "manual_reserve": 0.0,
        },
        "total_usd": 500.0,
    }
    contract["denominator_policy"]["policy_id"] = "finevo-pilot-v2.11.10-itt"
    contract["stages"] = _v2_11_10_expected_stages()
    contract["non_claims"] = _v2_11_10_expected_non_claims()
    contract["v21110_recovery_boundary"] = _v2_11_10_expected_recovery_boundary(
        status=status
    )
    contract["release_requirements"] = {
        "remote": "origin",
        "branch": "main",
        "tag": PILOT_CONTRACT_TAG_V2_11_10,
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
            pilot_contract_module.PILOT_V2_11_10_SOURCE_MANIFEST_FILE_SHA256
            is None
            or pilot_contract_module.PILOT_V2_11_10_SOURCE_MANIFEST_CONTENT_SHA256
            is None
        ):
            raise FrozenCandidateBootstrapError(
                "V2.11.10 source manifest must be sealed before frozen candidate"
            )
        pinned = pilot_contract_module.PILOT_CONTRACT_V2_11_10_CANONICAL_SHA256
        if pinned is not None:
            raise FrozenCandidateBootstrapError(
                "V2.11.10 canonical hash is already pinned; use normal frozen mode"
            )
        candidate_hash = contract["integrity"]["declared_sha256"]
        pilot_contract_module.PILOT_CONTRACT_V2_11_10_CANONICAL_SHA256 = (
            candidate_hash
        )
        try:
            parsed = PilotContract.from_dict(contract)
            _assert_v2119_recovery_delta(parent, contract, parsed_child=parsed)
            _assert_expanded_continuation_specs_match(parent_contract, parsed)
        finally:
            pilot_contract_module.PILOT_CONTRACT_V2_11_10_CANONICAL_SHA256 = pinned
        if (
            canonical_contract_sha256(contract) != candidate_hash
            or parsed.to_dict() != contract
        ):
            raise FrozenCandidateBootstrapError(
                "frozen candidate failed canonical round-trip self-validation"
            )
    else:
        parsed = _parse_with_bootstrap_design_pin(contract)
        _assert_v2119_recovery_delta(parent, contract, parsed_child=parsed)
        _assert_expanded_continuation_specs_match(parent_contract, parsed)
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
        default=Path("experiments/pilot_v2_11_10.yaml"),
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
    if args.status == "draft" and any(value is not None for value in ci_values.values()):
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
