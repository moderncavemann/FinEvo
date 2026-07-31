from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.render_pilot_v2111_contract import build_contract
from verified_memory.scientific_release_attestation import (
    build_scientific_contract_binding,
)
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_11_1,
    PILOT_CONTRACT_TAG_V2_11_1,
    PILOT_CONTRACT_V2_11_1_CANONICAL_SHA256,
    PILOT_V2_11_1_SOURCE_MANIFEST_CONTENT_SHA256,
    PILOT_V2_11_1_SOURCE_MANIFEST_FILE_SHA256,
    PilotContract,
    PilotContractError,
    canonical_contract_sha256,
    canonical_sha256,
    load_pilot_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_1.yaml"
MANIFEST_PATH = (
    ROOT / "experiments" / "pilot_v2_11_1_source_manifest.json"
)


def _rehash(value: dict) -> dict:
    candidate = deepcopy(value)
    candidate["integrity"]["declared_sha256"] = "0" * 64
    candidate["integrity"]["declared_sha256"] = canonical_contract_sha256(
        candidate
    )
    return candidate


def test_v2111_tracked_frozen_contract_loads_with_exact_denominator() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)

    assert contract.contract_id == PILOT_CONTRACT_ID_V2_11_1
    assert contract.status == "frozen"
    assert contract.canonical_hash == (
        PILOT_CONTRACT_V2_11_1_CANONICAL_SHA256
    )
    assert contract.implementation["required_git_tag"] == (
        PILOT_CONTRACT_TAG_V2_11_1
    )
    assert len(contract.expand()) == 136
    assert len(
        tuple(
            spec
            for spec in contract.expand()
            if spec.stage_id
            not in {
                "parent-import",
                "capability-gate",
                "long-context-preflight",
            }
        )
    ) == 131
    assert {
        spec.execution_mode
        for spec in contract.expand(stage="capability-gate")
    } == {"capability_authority_import"}
    assert contract.budgets["stage_usd_caps"] == {
        "parent_v211": 17.166524062500006,
        "hosted_v2111": 482.8334759375,
        "manual_reserve": 0.0,
    }
    assert contract.v2111_forward_boundary["matrix"][
        "fresh_hosted_provider_calls"
    ] == 5880
    assert contract.v2111_preflight_bootstrap_amendment[
        "bootstrap_policy"
    ]["effective_contract_envelope"] == {
        "prompt_tokens_per_call": 200000,
        "completion_tokens_per_call": 4096,
        "cached_input_discount_assumed": False,
        "price_basis": "frozen-provider-profile-dispatch-endpoint",
    }


def test_v2111_source_manifest_binding_is_exact() -> None:
    payload = MANIFEST_PATH.read_bytes()
    value = json.loads(payload)

    assert hashlib.sha256(payload).hexdigest() == (
        PILOT_V2_11_1_SOURCE_MANIFEST_FILE_SHA256
    )
    assert value["integrity"]["content_sha256"] == (
        PILOT_V2_11_1_SOURCE_MANIFEST_CONTENT_SHA256
    )
    unsigned = deepcopy(value)
    unsigned["integrity"].pop("content_sha256")
    assert canonical_sha256(unsigned) == (
        PILOT_V2_11_1_SOURCE_MANIFEST_CONTENT_SHA256
    )


def test_v2111_draft_renderer_remains_independently_valid() -> None:
    rendered = build_contract(ROOT, status="draft")
    contract = PilotContract.from_dict(rendered)

    assert contract.status == "draft"
    with pytest.raises(PilotContractError, match="draft"):
        contract.validate_provenance(
            git_tag=PILOT_CONTRACT_TAG_V2_11_1,
            git_commit="1" * 40,
        )


def test_v2111_frozen_renderer_is_deterministic() -> None:
    tracked = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    rendered = build_contract(
        ROOT,
        status="frozen",
        expected_ci=tracked["release_requirements"]["expected_ci"],
    )

    assert rendered == tracked


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["v2111_forward_boundary"]["matrix"].__setitem__(
            "fresh_hosted_provider_calls", 5879
        ),
        lambda value: value["v2111_preflight_bootstrap_amendment"][
            "bootstrap_policy"
        ]["effective_contract_envelope"].__setitem__(
            "prompt_tokens_per_call", 199999
        ),
        lambda value: value["arms"]["capability-probe"].__setitem__(
            "execution_mode", "capability_probe"
        ),
        lambda value: value["budgets"]["stage_usd_caps"].__setitem__(
            "hosted_v2111", 482.84
        ),
    ],
)
def test_v2111_rehashed_contract_drift_is_rejected(mutator) -> None:
    value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    mutator(value)

    with pytest.raises(PilotContractError):
        PilotContract.from_dict(_rehash(value))


def test_v2111_frozen_provenance_binds_tag_commit_and_contract() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    commit = "1" * 40

    assert contract.validate_provenance(
        git_tag=PILOT_CONTRACT_TAG_V2_11_1,
        git_commit=commit,
    ) == {
        "git_tag": PILOT_CONTRACT_TAG_V2_11_1,
        "resolved_git_commit": commit,
        "commit_resolution": "annotated_tag_peel",
        "p0_base_commit": contract.implementation["p0_base_commit"],
        "contract_id": PILOT_CONTRACT_ID_V2_11_1,
        "contract_sha256": contract.canonical_hash,
    }


def test_v2111_release_binding_covers_contract_and_sealed_manifests() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    binding = build_scientific_contract_binding(
        ROOT,
        contract_path="experiments/pilot_v2_11_1.yaml",
    )

    assert binding["contract_path"] == "experiments/pilot_v2_11_1.yaml"
    assert binding["contract_canonical_sha256"] == contract.canonical_hash
    assert binding["sealed_manifest_paths"]
    assert binding["sealed_manifest_inventory_sha256"] == (
        contract.release_requirements.expected_ci[
            "sealed_manifest_inventory_sha256"
        ]
    )
