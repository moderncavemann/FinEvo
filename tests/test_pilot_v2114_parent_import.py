from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import verified_memory.pilot_v2114_parent_import as parent_module
from verified_memory.pilot_v2114_parent_import import (
    PilotV2114ParentImportError,
    V2113_PREFLIGHT_RECEIPT_CONTENT_SHA256,
    V2114_CUMULATIVE_DEBIT,
    V2114_FORBIDDEN_IMPORT_PREFIXES,
    V2114_REUSABLE_AUTHORITY_KINDS,
    V2114_SOURCE_MANIFEST_CONTENT_SHA256,
    V2114_SOURCE_MANIFEST_FILE_SHA256,
    _audit_v2113_terminal_lineage,
    _seal,
    build_v2114_parent_import,
    build_v2114_source_manifest,
    parent_budget_debit_for_v2114,
    persist_v2114_parent_import,
    validate_v2114_parent_import_receipt,
)


REPO = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = REPO / "experiments" / "pilot_v2_11_4_source_manifest.json"


class FrozenContract:
    contract_id = "finevo-pilot-v2.11.4"
    status = "frozen"
    canonical_hash = "a" * 64
    implementation = {"required_git_tag": "pilot-v2.11.4-science"}


@pytest.fixture(scope="module")
def frozen_contract() -> FrozenContract:
    return FrozenContract()


@pytest.fixture(scope="module")
def v2113_terminal_release_root() -> Path:
    configured = os.environ.get("FINEVO_V2113_TERMINAL_RELEASE_ROOT")
    root = (
        Path(configured).expanduser().absolute()
        if configured
        else REPO.parent / "finevo-pilot-v2-11-3-science"
    )
    if not (root / "experiment_results" / "pilot-v2.11.3" / "raw").is_dir():
        pytest.skip(
            "exact V2.11.3 lineage replay requires its ignored terminal raw tree"
        )
    return root


def test_terminal_v2113_lineage_is_exact_zero_call_no_go(
    v2113_terminal_release_root: Path,
) -> None:
    lineage = _audit_v2113_terminal_lineage(
        lineage_repo_root=v2113_terminal_release_root
    )
    assert lineage["release"] == {
        "science_tag": "pilot-v2.11.3-science",
        "science_tag_object": "87a1911284177b627755faf361ad4ea6c8213958",
        "resolved_git_commit": "65c613cdc9598dfffecbdf3a375cbf6113246782",
    }
    assert lineage["run_ledger"]["internal_sha256"] == (
        "97216e7b0a23b1b78a1e79d3ae166621147fab5582e5259434e1138c39946f40"
    )
    assert lineage["run_ledger"]["status_counts"] == {
        "complete": 3,
        "integrity-stopped": 133,
    }
    assert lineage["budget_ledger"]["internal_sha256"] == (
        "366495f3cc4b8075e072c47fcf31c3eed40371996f0057efba64a1709ac5850a"
    )
    assert lineage["budget_ledger"]["current_attempt"] == {
        "cost_usd": 0.0,
        "hosted_completions": 0,
        "storage_bytes": 169978,
        "provider_calls": 0,
    }
    assert lineage["stage_receipts"]["long-context-preflight"][
        "content_sha256"
    ] == V2113_PREFLIGHT_RECEIPT_CONTENT_SHA256
    assert lineage["terminal_denominator"]["globally_a_d_outcome_blind"] is True


def test_source_manifest_exactly_replays_both_frozen_roots(
    v2112_parent_release_root: Path,
    v2113_terminal_release_root: Path,
) -> None:
    tracked = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    rebuilt = build_v2114_source_manifest(
        source_repo_root=v2112_parent_release_root,
        lineage_repo_root=v2113_terminal_release_root,
        evidence_repo_root=REPO,
    )
    assert rebuilt == tracked
    assert tracked["integrity"]["content_sha256"] == (
        V2114_SOURCE_MANIFEST_CONTENT_SHA256
    )
    assert hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest() == (
        V2114_SOURCE_MANIFEST_FILE_SHA256
    )
    assert tracked["authority_parent_release"]["contract_id"] == (
        "finevo-pilot-v2.11.2"
    )
    assert tracked["terminal_lineage_release"]["contract_id"] == (
        "finevo-pilot-v2.11.3"
    )
    assert tracked["terminal_lineage_release"]["raw_inventory"] == {
        "file_count": 18,
        "inventory_canonicalization": "json-sort-keys-compact-utf8-v1",
        "inventory_schema_version": "finevo-raw-tree-inventory-v1",
        "inventory_sha256": (
            "691dd5f032d926d6c40cedf14ea380403163323c53bfbd9bc5c38d14e54decd3"
        ),
        "root": "experiment_results/pilot-v2.11.3/raw",
        "storage_bytes": 524325,
    }
    assert tracked["terminal_denominator"]["status_counts"] == {
        "complete": 3,
        "integrity-stopped": 133,
    }
    assert tracked["authority_source_denominator"]["status_counts"] == {
        "complete": 10,
        "failed": 126,
    }
    assert tracked["reusable_authority_allowlist"] == list(
        V2114_REUSABLE_AUTHORITY_KINDS
    )
    assert tracked["forbidden_import_prefixes"] == list(
        V2114_FORBIDDEN_IMPORT_PREFIXES
    )
    assert tracked["import_policy"]["imported_effect_cells"] == 0
    assert tracked["import_policy"]["v2113_scientific_cells_imported"] == 0
    assert tracked["import_policy"]["provider_calls_during_import"] == 0


def test_lineage_preflight_hash_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    v2113_terminal_release_root: Path,
) -> None:
    monkeypatch.setattr(
        parent_module,
        "_V2113_STAGE_RECEIPTS",
        {
            **parent_module._V2113_STAGE_RECEIPTS,
            "long-context-preflight": {
                **parent_module._V2113_STAGE_RECEIPTS[
                    "long-context-preflight"
                ],
                "content_sha256": "0" * 64,
            },
        },
    )
    with pytest.raises(PilotV2114ParentImportError, match="receipt drifted"):
        _audit_v2113_terminal_lineage(
            lineage_repo_root=v2113_terminal_release_root
        )


def test_parent_debit_is_bound_to_v2113_ledgers(
    frozen_contract: FrozenContract,
) -> None:
    debit = parent_budget_debit_for_v2114(frozen_contract)
    assert debit == V2114_CUMULATIVE_DEBIT
    assert debit.to_dict() == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": (
            "84c818348fabfdd0ddd0ed503c0a5610faf10098f4973d1748b795e2e65b56f1"
        ),
        "parent_run_ledger_sha256": (
            "97216e7b0a23b1b78a1e79d3ae166621147fab5582e5259434e1138c39946f40"
        ),
        "parent_budget_ledger_sha256": (
            "366495f3cc4b8075e072c47fcf31c3eed40371996f0057efba64a1709ac5850a"
        ),
        "stage_bucket": "parent_v2113",
        "cost_usd": 19.998220562500006,
        "hosted_completions": 1004,
        "storage_bytes": 221838685,
        "record_sha256": (
            "3f75623b4eb5b6c3c1c2e2a7e97687c215da025cbea309f94e861abee47f90ca"
        ),
    }
    assert parent_budget_debit_for_v2114(SimpleNamespace(contract_id="other")) is None


def test_parent_receipt_binds_authority_and_terminal_lineage(
    frozen_contract: FrozenContract,
    v2112_parent_release_root: Path,
    v2113_terminal_release_root: Path,
) -> None:
    receipt = build_v2114_parent_import(
        repo_root=REPO,
        contract=frozen_contract,
        child_git_commit="b" * 40,
        source_repo_root=v2112_parent_release_root,
        lineage_repo_root=v2113_terminal_release_root,
        evidence_repo_root=REPO,
    )
    validate_v2114_parent_import_receipt(
        receipt,
        contract=frozen_contract,
        child_git_commit="b" * 40,
        repo_root=REPO,
    )
    assert receipt["authority_parent_release"]["contract_id"] == (
        "finevo-pilot-v2.11.2"
    )
    assert receipt["terminal_lineage_release"]["contract_id"] == (
        "finevo-pilot-v2.11.3"
    )
    assert receipt["terminal_parent_denominator"]["status_counts"] == {
        "complete": 3,
        "integrity-stopped": 133,
    }

    tampered = deepcopy(receipt)
    tampered["terminal_lineage_release"]["preflight_receipt_content_sha256"] = (
        "0" * 64
    )
    tampered = _seal(tampered)
    with pytest.raises(PilotV2114ParentImportError, match="claim boundary"):
        validate_v2114_parent_import_receipt(
            tampered,
            contract=frozen_contract,
            child_git_commit="b" * 40,
            repo_root=REPO,
        )

    tampered = deepcopy(receipt)
    tampered["preflight_authority_wrappers"]["gpt52_main"]["reservations"][
        "action"
    ]["reservation"]["sample_count"] = 25
    tampered["preflight_authority_wrappers"]["gpt52_main"] = _seal(
        tampered["preflight_authority_wrappers"]["gpt52_main"]
    )
    tampered = _seal(tampered)
    with pytest.raises(PilotV2114ParentImportError, match="preflight wrapper"):
        validate_v2114_parent_import_receipt(
            tampered,
            contract=frozen_contract,
            child_git_commit="b" * 40,
            repo_root=REPO,
        )


def test_parent_persist_owns_only_v2114_parent_receipt(
    tmp_path: Path,
    frozen_contract: FrozenContract,
    v2112_parent_release_root: Path,
    v2113_terminal_release_root: Path,
) -> None:
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / SOURCE_MANIFEST.name).write_bytes(
        SOURCE_MANIFEST.read_bytes()
    )
    result = persist_v2114_parent_import(
        repo_root=tmp_path,
        raw_root=tmp_path / "experiment_results" / "pilot-v2.11.4" / "raw",
        contract=frozen_contract,
        git_commit="b" * 40,
        source_repo_root=v2112_parent_release_root,
        lineage_repo_root=v2113_terminal_release_root,
        evidence_repo_root=REPO,
    )
    receipt = Path(result["receipt"])
    assert receipt == (
        tmp_path
        / "experiment_results"
        / "pilot-v2.11.4"
        / "raw"
        / "parent-import"
        / "parent_import_receipt.json"
    )
    assert receipt.is_file()
    assert not list(receipt.parent.glob("observed_p95/**/*"))
    assert result["provider_construction_during_import"] is False
    assert result["provider_calls_during_import"] == 0
