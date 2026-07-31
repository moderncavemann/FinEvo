from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.render_pilot_v2112_contract import (
    _assert_v2111_science_delta,
    build_contract,
)
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256,
    PILOT_CONTRACT_ID_V2_11_2,
    PILOT_CONTRACT_TAG_V2_11_2,
    PILOT_V2_11_2_SOURCE_MANIFEST_CONTENT_SHA256,
    PILOT_V2_11_2_SOURCE_MANIFEST_FILE_SHA256,
    PilotContract,
    PilotContractError,
    canonical_contract_sha256,
    canonical_sha256,
    load_pilot_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_2.yaml"
PARENT_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_1.yaml"
MANIFEST_PATH = ROOT / "experiments" / "pilot_v2_11_2_source_manifest.json"
EXPECTED_CI = {
    "test_count": 1553,
    "test_collection_sha256": (
        "e599e883988ad04dd1eb40f6810902cc60fb91495c042486c335fbe08ecc33fe"
    ),
    "compiled_source_count": 245,
    "compiled_source_inventory_sha256": (
        "4ca58622096459153ea503b26d2d7cfe41dd09b1e1ebc1e16901fcbdea9c55a1"
    ),
    "sealed_manifest_inventory_sha256": (
        "b5c5a817d09d10752c1f5f00ba556b417d16e06c64b5fcbb15671e49a1d81952"
    ),
}


def _rehash(value: dict) -> dict:
    candidate = deepcopy(value)
    candidate["integrity"]["declared_sha256"] = "0" * 64
    candidate["integrity"]["declared_sha256"] = canonical_contract_sha256(candidate)
    return candidate


def test_v2112_tracked_draft_contract_has_exact_denominator_and_budget() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    specs = contract.expand()

    assert contract.contract_id == PILOT_CONTRACT_ID_V2_11_2
    assert contract.status == "frozen"
    assert contract.canonical_hash == PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256
    assert contract.release_requirements is not None
    assert contract.release_requirements.expected_ci == EXPECTED_CI
    assert contract.implementation["required_git_tag"] == (PILOT_CONTRACT_TAG_V2_11_2)
    assert len(specs) == 136
    assert len({spec.run_id for spec in specs}) == 136
    assert Counter(spec.stage_id for spec in specs) == {
        "parent-import": 1,
        "capability-gate": 2,
        "long-context-preflight": 2,
        "experiment-c": 25,
        "experiment-a": 20,
        "experiment-d": 55,
        "experiment-b": 25,
        "cross-model": 6,
    }
    assert {
        spec.execution_mode for spec in contract.expand(stage="capability-gate")
    } == {"capability_authority_import"}
    assert contract.budgets["stage_usd_caps"] == {
        "parent_v2111": 18.586399812500005,
        "hosted_v2112": 481.4136001875,
        "manual_reserve": 0.0,
    }
    boundary = contract.v2112_forward_boundary
    assert boundary is not None
    assert boundary["parent_budget_debit"] == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": (
            "818607de5cd512cee60ece06c3f81612e6945cf7ff6d1e48ca643d2109cd7410"
        ),
        "parent_run_ledger_sha256": (
            "ed9c0210791627128dc0e9942df2cd46269acbbd28a6c52af33454172a4b76c9"
        ),
        "parent_budget_ledger_sha256": (
            "df9ccffbba39ac6d86375be433c4a34f4e3b51f60a4e0e10d42d31b7c2886330"
        ),
        "stage_bucket": "parent_v2111",
        "cost_usd": 18.586399812500005,
        "hosted_completions": 940,
        "storage_bytes": 217838625,
        "record_sha256": (
            "678fc5b795e66f1aa358ea7941bebb9167097158f2aed8cee4044567109c5582"
        ),
    }
    assert boundary["matrix"]["fresh_hosted_provider_calls"] == 5880
    assert boundary["matrix"]["scientific_cells"] == 131
    assert boundary["import_policy"]["imported_preflight_samples"] == 0
    assert tuple(boundary["import_policy"]["imported_checkpoint_artifacts"]) == ()
    assert tuple(boundary["import_policy"]["imported_p95_authorities"]) == ()


def test_v2112_science_configuration_matches_v2111_delta_allowlist() -> None:
    parent = load_pilot_contract(PARENT_CONTRACT_PATH).to_dict()
    child = load_pilot_contract(CONTRACT_PATH).to_dict()

    _assert_v2111_science_delta(parent, child)
    for field in (
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
    ):
        assert child[field] == parent[field]


def test_v2112_source_manifest_is_exact_and_failure_only() -> None:
    payload = MANIFEST_PATH.read_bytes()
    value = json.loads(payload)

    assert hashlib.sha256(payload).hexdigest() == (
        PILOT_V2_11_2_SOURCE_MANIFEST_FILE_SHA256
    )
    assert value["integrity"]["content_sha256"] == (
        PILOT_V2_11_2_SOURCE_MANIFEST_CONTENT_SHA256
    )
    unsigned = deepcopy(value)
    unsigned["integrity"].pop("content_sha256")
    assert canonical_sha256(unsigned) == (PILOT_V2_11_2_SOURCE_MANIFEST_CONTENT_SHA256)
    audit = value["failed_preflight_audit"]
    assert audit["historical_provider_calls"] == 64
    assert audit["samples_exported"] == 0
    assert audit["p95_authorities_exported"] == []
    assert audit["checkpoint_artifacts_exported"] == []
    assert all(
        row["journal_audit"]["samples_exported"] == 0
        and row["journal_audit"]["p95_authority"] is None
        and row["checkpoint_created"] is False
        and row["exactness_receipt_created"] is False
        for row in audit["models"].values()
    )


def test_v2112_draft_renderer_is_deterministic_and_not_paid() -> None:
    tracked = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    rendered = build_contract(ROOT, status="draft")
    frozen = build_contract(ROOT, status="frozen", expected_ci=EXPECTED_CI)

    assert rendered != tracked
    assert rendered["status"] == "draft"
    assert all(
        value is None
        for value in rendered["release_requirements"]["expected_ci"].values()
    )
    assert frozen == tracked
    assert frozen["integrity"]["declared_sha256"] == (
        PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256
    )
    contract = PilotContract.from_dict(rendered)
    with pytest.raises(PilotContractError, match="draft"):
        contract.validate_provenance(
            git_tag=PILOT_CONTRACT_TAG_V2_11_2,
            git_commit="1" * 40,
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["v2112_forward_boundary"]["matrix"].__setitem__(
            "fresh_hosted_provider_calls", 5879
        ),
        lambda value: value["v2112_forward_boundary"]["import_policy"].__setitem__(
            "imported_preflight_samples", 64
        ),
        lambda value: value["v2112_recovery_amendment"][
            "lifecycle_validator_repair"
        ].__setitem__("unchanged_retirement_policy", False),
        lambda value: value["v2112_recovery_amendment"]["bootstrap_policy"][
            "effective_contract_envelope"
        ].__setitem__("prompt_tokens_per_call", 199999),
        lambda value: value["arms"]["capability-probe"].__setitem__(
            "execution_mode", "capability_probe"
        ),
        lambda value: value["budgets"]["stage_usd_caps"].__setitem__(
            "hosted_v2112", 481.42
        ),
    ],
)
def test_v2112_rehashed_contract_drift_is_rejected(mutator) -> None:
    value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    mutator(value)

    with pytest.raises(PilotContractError):
        PilotContract.from_dict(_rehash(value))


def test_v2112_delta_guard_rejects_scientific_drift() -> None:
    parent = load_pilot_contract(PARENT_CONTRACT_PATH).to_dict()
    child = load_pilot_contract(CONTRACT_PATH).to_dict()
    child["shocks"]["registered-rate-shock"]["schedule"][1]["interest_rate"] = 0.07

    with pytest.raises(ValueError, match="scientific fields"):
        _assert_v2111_science_delta(parent, child)
