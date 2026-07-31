from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import verified_memory.pilot_v2115_parent_import as parent_module
from verified_memory.pilot_v2115_parent_import import (
    PilotV2115ParentImportError,
    V2114_POST_GATE_CONTENT_SHA256,
    V2115_CUMULATIVE_DEBIT,
    V2115_FORBIDDEN_IMPORT_PREFIXES,
    V2115_REUSABLE_AUTHORITY_KINDS,
    V2115_SOURCE_MANIFEST_CONTENT_SHA256,
    V2115_SOURCE_MANIFEST_FILE_SHA256,
    _audit_v2114_terminal_lineage,
    _seal,
    build_v2115_parent_import,
    build_v2115_source_manifest,
    parent_budget_debit_for_v2115,
    persist_v2115_parent_import,
    validate_v2115_parent_import_receipt,
)


REPO = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = REPO / "experiments" / "pilot_v2_11_5_source_manifest.json"


class FrozenContract:
    contract_id = "finevo-pilot-v2.11.5"
    status = "frozen"
    canonical_hash = "a" * 64
    implementation = {"required_git_tag": "pilot-v2.11.5-science"}


@pytest.fixture(scope="module")
def frozen_contract() -> FrozenContract:
    return FrozenContract()


@pytest.fixture(scope="module")
def v2114_terminal_release_root() -> Path:
    configured = os.environ.get("FINEVO_V2114_SCIENCE_RELEASE_ROOT")
    root = (
        Path(configured).expanduser().absolute()
        if configured
        else REPO.parent / "finevo-pilot-v2-11-4-science"
    )
    if not (root / "experiment_results" / "pilot-v2.11.4" / "raw").is_dir():
        pytest.skip(
            "exact V2.11.4 lineage replay requires its ignored science raw tree"
        )
    return root


def test_v2114_lineage_is_exact_pre_dispatch_zero_call_no_go(
    v2114_terminal_release_root: Path,
) -> None:
    lineage = _audit_v2114_terminal_lineage(
        lineage_repo_root=v2114_terminal_release_root
    )
    assert lineage["release"] == {
        "science_tag": "pilot-v2.11.4-science",
        "science_tag_object": "d774465ad006e9ae974f927ff7b4de94fd5f5147",
        "resolved_git_commit": "74f6c05dafc58fadf8d1b658ef3764d244676f76",
    }
    assert lineage["run_ledger"]["internal_sha256"] == (
        "f0064120e279137fbd7dd5f5cec474aa384745b7c405d9371c90ac5c4f448656"
    )
    assert lineage["run_ledger"]["status_counts"] == {
        "complete": 5,
        "scheduled": 131,
    }
    assert lineage["budget_ledger"]["internal_sha256"] == (
        "d4ce8beebe1e462003039db2d39e6616c76cd33897c5b3e77e45989ced9d8789"
    )
    assert lineage["budget_ledger"]["current_attempt"] == {
        "cost_usd": 0.0,
        "hosted_completions": 0,
        "storage_bytes": 210017,
        "provider_calls": 0,
    }
    assert lineage["post_gate_authority"]["receipt_content_sha256"] == (
        V2114_POST_GATE_CONTENT_SHA256
    )
    assert lineage["acceptance_receipt"]["present"] is False
    assert lineage["parent_attempt_denominator"]["terminal_cells"] == 5
    assert lineage["parent_attempt_denominator"]["scheduled_cells"] == 131
    assert (
        lineage["parent_attempt_denominator"]["globally_a_d_outcome_blind"] is True
    )


def test_source_manifest_exactly_replays_both_frozen_roots(
    v2112_parent_release_root: Path,
    v2114_terminal_release_root: Path,
) -> None:
    tracked = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    rebuilt = build_v2115_source_manifest(
        source_repo_root=v2112_parent_release_root,
        lineage_repo_root=v2114_terminal_release_root,
        evidence_repo_root=REPO,
    )
    assert rebuilt == tracked
    assert tracked["integrity"]["content_sha256"] == (
        V2115_SOURCE_MANIFEST_CONTENT_SHA256
    )
    assert hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest() == (
        V2115_SOURCE_MANIFEST_FILE_SHA256
    )
    assert tracked["authority_parent_release"]["contract_id"] == (
        "finevo-pilot-v2.11.2"
    )
    assert tracked["terminal_lineage_release"]["contract_id"] == (
        "finevo-pilot-v2.11.4"
    )
    assert tracked["terminal_lineage_release"]["raw_inventory"] == {
        "file_count": 27,
        "inventory_canonicalization": "json-sort-keys-compact-utf8-v1",
        "inventory_schema_version": "finevo-raw-tree-inventory-v1",
        "inventory_sha256": (
            "b3e6228fc1873e64f792b4ab51af18104c38b402dbd13b2a7f14ef2ddcb0c89c"
        ),
        "root": "experiment_results/pilot-v2.11.4/raw",
        "storage_bytes": 427703,
    }
    assert tracked["parent_attempt_denominator"]["status_counts"] == {
        "complete": 5,
        "scheduled": 131,
    }
    assert tracked["authority_source_denominator"]["status_counts"] == {
        "complete": 10,
        "failed": 126,
    }
    assert tracked["reusable_authority_allowlist"] == list(
        V2115_REUSABLE_AUTHORITY_KINDS
    )
    assert tracked["forbidden_import_prefixes"] == list(
        V2115_FORBIDDEN_IMPORT_PREFIXES
    )
    assert tracked["import_policy"]["imported_effect_cells"] == 0
    assert tracked["import_policy"]["v2114_scientific_cells_imported"] == 0
    assert tracked["import_policy"]["provider_calls_during_import"] == 0
    assert tracked["consumer_authority_normalization"]["stable_field_count"] == 9
    assert tracked["consumer_authority_normalization"]["generation_field_count"] == 8


def test_lineage_preflight_hash_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    v2114_terminal_release_root: Path,
) -> None:
    monkeypatch.setattr(
        parent_module,
        "_V2114_STAGE_RECEIPTS",
        {
            **parent_module._V2114_STAGE_RECEIPTS,
            "long-context-preflight": {
                **parent_module._V2114_STAGE_RECEIPTS[
                    "long-context-preflight"
                ],
                "content_sha256": "0" * 64,
            },
        },
    )
    with pytest.raises(PilotV2115ParentImportError, match="receipt drifted"):
        _audit_v2114_terminal_lineage(
            lineage_repo_root=v2114_terminal_release_root
        )


def test_parent_debit_is_bound_to_v2114_ledgers(
    frozen_contract: FrozenContract,
) -> None:
    debit = parent_budget_debit_for_v2115(frozen_contract)
    assert debit == V2115_CUMULATIVE_DEBIT
    assert debit.to_dict() == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": (
            "e898fe49935dae9ae7f0d7ac577dae943192953c1da581d70c334f8c64924e46"
        ),
        "parent_run_ledger_sha256": (
            "f0064120e279137fbd7dd5f5cec474aa384745b7c405d9371c90ac5c4f448656"
        ),
        "parent_budget_ledger_sha256": (
            "d4ce8beebe1e462003039db2d39e6616c76cd33897c5b3e77e45989ced9d8789"
        ),
        "stage_bucket": "parent_v2114",
        "cost_usd": 19.998220562500006,
        "hosted_completions": 1004,
        "storage_bytes": 222048702,
        "record_sha256": (
            "9595d037a21f429a59fd37febd4abd8283287e080ee9eb506ebf999e3d1e81a5"
        ),
    }
    assert parent_budget_debit_for_v2115(SimpleNamespace(contract_id="other")) is None


def test_parent_receipt_binds_authority_and_terminal_lineage(
    frozen_contract: FrozenContract,
    v2112_parent_release_root: Path,
    v2114_terminal_release_root: Path,
) -> None:
    receipt = build_v2115_parent_import(
        repo_root=REPO,
        contract=frozen_contract,
        child_git_commit="b" * 40,
        source_repo_root=v2112_parent_release_root,
        lineage_repo_root=v2114_terminal_release_root,
        evidence_repo_root=REPO,
    )
    validate_v2115_parent_import_receipt(
        receipt,
        contract=frozen_contract,
        child_git_commit="b" * 40,
        repo_root=REPO,
    )
    assert receipt["authority_parent_release"]["contract_id"] == (
        "finevo-pilot-v2.11.2"
    )
    assert receipt["terminal_lineage_release"]["contract_id"] == (
        "finevo-pilot-v2.11.4"
    )
    assert receipt["parent_attempt_denominator"]["status_counts"] == {
        "complete": 5,
        "scheduled": 131,
    }

    tampered = deepcopy(receipt)
    tampered["terminal_lineage_release"]["post_gate_receipt_content_sha256"] = (
        "0" * 64
    )
    tampered = _seal(tampered)
    with pytest.raises(PilotV2115ParentImportError, match="claim boundary"):
        validate_v2115_parent_import_receipt(
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
    with pytest.raises(PilotV2115ParentImportError, match="preflight wrapper"):
        validate_v2115_parent_import_receipt(
            tampered,
            contract=frozen_contract,
            child_git_commit="b" * 40,
            repo_root=REPO,
        )


def test_parent_persist_owns_only_v2115_parent_receipt(
    tmp_path: Path,
    frozen_contract: FrozenContract,
    v2112_parent_release_root: Path,
    v2114_terminal_release_root: Path,
) -> None:
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / SOURCE_MANIFEST.name).write_bytes(
        SOURCE_MANIFEST.read_bytes()
    )
    result = persist_v2115_parent_import(
        repo_root=tmp_path,
        raw_root=tmp_path / "experiment_results" / "pilot-v2.11.5" / "raw",
        contract=frozen_contract,
        git_commit="b" * 40,
        source_repo_root=v2112_parent_release_root,
        lineage_repo_root=v2114_terminal_release_root,
        evidence_repo_root=REPO,
    )
    receipt = Path(result["receipt"])
    assert receipt == (
        tmp_path
        / "experiment_results"
        / "pilot-v2.11.5"
        / "raw"
        / "parent-import"
        / "parent_import_receipt.json"
    )
    assert receipt.is_file()
    assert not list(receipt.parent.glob("observed_p95/**/*"))
    assert result["provider_construction_during_import"] is False
    assert result["provider_calls_during_import"] == 0
