from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verified_memory.pilot_v2113_parent_import import (
    PilotV2113ParentImportError,
    V2113_CUMULATIVE_DEBIT,
    V2113_FORBIDDEN_IMPORT_PREFIXES,
    V2113_REUSABLE_AUTHORITY_KINDS,
    V2113_SOURCE_MANIFEST_CONTENT_SHA256,
    V2113_SOURCE_MANIFEST_FILE_SHA256,
    _seal,
    build_v2113_parent_import,
    build_v2113_source_manifest,
    parent_budget_debit_for_v2113,
    persist_v2113_parent_import,
    validate_v2113_parent_import_receipt,
)


REPO = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = REPO / "experiments" / "pilot_v2_11_3_source_manifest.json"


class FrozenContract:
    contract_id = "finevo-pilot-v2.11.3"
    status = "frozen"
    canonical_hash = "a" * 64
    implementation = {"required_git_tag": "pilot-v2.11.3-science"}


@pytest.fixture(scope="module")
def frozen_contract() -> FrozenContract:
    return FrozenContract()


def test_source_manifest_is_exact_replay_with_standard_seal(
    v2112_parent_release_root: Path,
) -> None:
    tracked = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    rebuilt = build_v2113_source_manifest(
        parent_science_root=v2112_parent_release_root,
        evidence_repo_root=REPO,
    )
    assert rebuilt == tracked
    assert tracked["integrity"]["content_sha256"] == (
        V2113_SOURCE_MANIFEST_CONTENT_SHA256
    )
    assert __import__("hashlib").sha256(SOURCE_MANIFEST.read_bytes()).hexdigest() == (
        V2113_SOURCE_MANIFEST_FILE_SHA256
    )
    assert tracked["reusable_authority_allowlist"] == list(
        V2113_REUSABLE_AUTHORITY_KINDS
    )
    assert tracked["forbidden_import_prefixes"] == list(V2113_FORBIDDEN_IMPORT_PREFIXES)
    assert tracked["terminal_denominator"]["status_counts"] == {
        "complete": 10,
        "failed": 126,
    }
    assert tracked["import_policy"]["imported_effect_cells"] == 0
    assert tracked["import_policy"]["provider_calls_during_import"] == 0


def test_parent_debit_is_exact_and_scoped(frozen_contract: FrozenContract) -> None:
    debit = parent_budget_debit_for_v2113(frozen_contract)
    assert debit == V2113_CUMULATIVE_DEBIT
    assert debit.to_dict() == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": (
            "c04f7d4c5ae0962a4a64b0ac543d890a1475b6f184f516534eeb8ff026505a37"
        ),
        "parent_run_ledger_sha256": (
            "686d7f528268e0d9d6ac97ae27d483af9c2eb93be53bd329b4fd621c0ec2ae25"
        ),
        "parent_budget_ledger_sha256": (
            "36dd9c62a56c7e87bb647feebeaa7f8d03b0a410d3c7d163834d5029f8da868b"
        ),
        "stage_bucket": "parent_v2112",
        "cost_usd": 19.998220562500006,
        "hosted_completions": 1004,
        "storage_bytes": 221668707,
        "record_sha256": (
            "3ddc22970ff30d1ad9fc3b9efbffe5e4de1f641851bc9e3398aa2fd0977154a1"
        ),
    }
    assert parent_budget_debit_for_v2113(SimpleNamespace(contract_id="other")) is None


def test_parent_receipt_rejects_resealed_scientific_substitution(
    frozen_contract: FrozenContract,
    v2112_parent_release_root: Path,
) -> None:
    receipt = build_v2113_parent_import(
        repo_root=REPO,
        contract=frozen_contract,
        child_git_commit="b" * 40,
        parent_science_root=v2112_parent_release_root,
        evidence_repo_root=REPO,
    )
    validate_v2113_parent_import_receipt(
        receipt,
        contract=frozen_contract,
        child_git_commit="b" * 40,
        repo_root=REPO,
    )

    tampered = deepcopy(receipt)
    tampered["terminal_parent_denominator"]["scientific_complete"] = True
    tampered = _seal(tampered)
    with pytest.raises(PilotV2113ParentImportError, match="claim boundary"):
        validate_v2113_parent_import_receipt(
            tampered,
            contract=frozen_contract,
            child_git_commit="b" * 40,
            repo_root=REPO,
        )

    tampered = deepcopy(receipt)
    tampered["preflight_authority_wrappers"]["gpt52_main"]["reservations"]["action"][
        "reservation"
    ]["sample_count"] = 25
    tampered["preflight_authority_wrappers"]["gpt52_main"] = _seal(
        tampered["preflight_authority_wrappers"]["gpt52_main"]
    )
    tampered = _seal(tampered)
    with pytest.raises(PilotV2113ParentImportError, match="preflight wrapper"):
        validate_v2113_parent_import_receipt(
            tampered,
            contract=frozen_contract,
            child_git_commit="b" * 40,
            repo_root=REPO,
        )


def test_parent_persist_owns_only_parent_receipt(
    tmp_path: Path,
    frozen_contract: FrozenContract,
    v2112_parent_release_root: Path,
) -> None:
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / SOURCE_MANIFEST.name).write_bytes(
        SOURCE_MANIFEST.read_bytes()
    )
    result = persist_v2113_parent_import(
        repo_root=tmp_path,
        raw_root=tmp_path / "experiment_results" / "pilot-v2.11.3" / "raw",
        contract=frozen_contract,
        git_commit="b" * 40,
        source_repo_root=v2112_parent_release_root,
        evidence_repo_root=REPO,
    )
    receipt = Path(result["receipt"])
    assert receipt == (
        tmp_path
        / "experiment_results"
        / "pilot-v2.11.3"
        / "raw"
        / "parent-import"
        / "parent_import_receipt.json"
    )
    assert receipt.is_file()
    assert not list(receipt.parent.glob("observed_p95/**/*"))
    assert result["provider_construction_during_import"] is False
    assert result["provider_calls_during_import"] == 0
