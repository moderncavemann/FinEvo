from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from verified_memory import pilot_v2112_parent_import as parent_import
from verified_memory.pilot_budget import ParentBudgetDebit


ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = ROOT.parent / "finevo-pilot-v2-11-1-science"
GRANDPARENT_ROOT = ROOT.parent / "finevo-pilot-v2-11-science"
CHILD_CONTRACT_SHA256 = "a" * 64
CHILD_GIT_TAG = "pilot-v2.11.2-science"
CHILD_GIT_COMMIT = "b" * 40
MODELS = {"gpt52_main", "gpt56_diagnostic"}


def _all_keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def _require_immutable_sources() -> None:
    missing = [path for path in (PARENT_ROOT, GRANDPARENT_ROOT) if not path.is_dir()]
    if missing:
        pytest.skip(
            "immutable V2.11.1/V2.11 science worktrees are unavailable: "
            + ", ".join(str(path) for path in missing)
        )


@pytest.fixture(scope="module")
def live_parent_import() -> dict[str, Any]:
    _require_immutable_sources()
    rebuilt_manifest = parent_import.build_v2112_source_manifest(
        parent_science_root=PARENT_ROOT,
        grandparent_science_root=GRANDPARENT_ROOT,
    )
    tracked_manifest = parent_import.load_v2112_source_manifest(repo_root=ROOT)
    assert rebuilt_manifest == tracked_manifest
    return parent_import.build_v2112_parent_import(
        repo_root=ROOT,
        parent_science_root=PARENT_ROOT,
        grandparent_science_root=GRANDPARENT_ROOT,
        child_contract_sha256=CHILD_CONTRACT_SHA256,
        child_git_tag=CHILD_GIT_TAG,
        child_git_commit=CHILD_GIT_COMMIT,
    )


def test_tracked_manifest_hash_and_terminal_denominator_are_frozen() -> None:
    path = ROOT / parent_import.V2112_SOURCE_MANIFEST_PATH
    raw = path.read_bytes()
    manifest = parent_import.load_v2112_source_manifest(repo_root=ROOT)

    assert hashlib.sha256(raw).hexdigest() == (
        parent_import.V2112_SOURCE_MANIFEST_FILE_SHA256
    )
    assert manifest["integrity"]["content_sha256"] == (
        parent_import.V2112_SOURCE_MANIFEST_CONTENT_SHA256
    )
    assert manifest["terminal_denominator"] == {
        "registered_cells": 136,
        "status_counts": {
            "complete": 3,
            "failed": 2,
            "integrity-stopped": 131,
        },
        "stage_status_counts": {
            "parent-import": {"complete": 1},
            "capability-gate": {"complete": 2},
            "long-context-preflight": {"failed": 2},
            "experiment-a": {"integrity-stopped": 20},
            "experiment-b": {"integrity-stopped": 25},
            "experiment-c": {"integrity-stopped": 25},
            "experiment-d": {"integrity-stopped": 55},
            "cross-model": {"integrity-stopped": 6},
        },
        "all_cells_terminal": True,
        "scientific_matrix_complete": False,
        "post_gate_authority_created": False,
        "preflight_checkpoint_count": 0,
        "preflight_exactness_receipt_count": 0,
    }


def test_manifest_keeps_failed_preflight_as_audit_only() -> None:
    manifest = parent_import.load_v2112_source_manifest(repo_root=ROOT)
    failed = manifest["failed_preflight_audit"]

    assert failed["historical_provider_calls"] == 64
    assert failed["historical_cost_usd"] == pytest.approx(1.41987575)
    assert failed["samples_exported"] == 0
    assert failed["checkpoint_artifacts_exported"] == []
    assert failed["p95_authorities_exported"] == []
    assert failed["authority_use"] == ("historical-failure-and-budget-audit-only")
    assert set(failed["models"]) == MODELS
    for row in failed["models"].values():
        audit = row["journal_audit"]
        assert row["status"] == "failed"
        assert row["provider_calls"] == 32
        assert row["samples_exported"] == 0
        assert row["checkpoint_created"] is False
        assert row["exactness_receipt_created"] is False
        assert row["p95_authority_created"] is False
        assert audit["event_count"] == 64
        assert audit["completion_event_count"] == 32
        assert audit["parse_disposition_count"] == 32
        assert audit["action_call_count"] == 24
        assert audit["semantic_call_count"] == 8
        assert audit["samples_exported"] == 0
        assert audit["p95_authority"] is None


def test_exact_parent_debit_is_returned_only_after_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_audit(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(parent_import, "_audit_sources", fake_audit)
    debit = parent_import.parent_budget_debit_for_v2112(
        SimpleNamespace(contract_id="finevo-pilot-v2.11.2"),
        repo_root=ROOT,
    )

    assert isinstance(debit, ParentBudgetDebit)
    assert calls and calls[0]["repo_root"] == ROOT
    assert debit.to_dict() == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "stage_bucket": "parent_v2111",
        "hosted_completions": 940,
        "cost_usd": 18.586399812500005,
        "storage_bytes": 217_838_625,
        "parent_contract_sha256": parent_import.V2111_CONTRACT_SHA256,
        "parent_run_ledger_sha256": parent_import.V2111_RUN_LEDGER_SHA256,
        "parent_budget_ledger_sha256": parent_import.V2111_BUDGET_LEDGER_SHA256,
        "record_sha256": parent_import.V2111_PARENT_DEBIT_RECORD_SHA256,
    }


@pytest.mark.parametrize(
    ("contract_sha256", "tag", "commit", "message"),
    [
        ("A" * 64, CHILD_GIT_TAG, CHILD_GIT_COMMIT, "lowercase SHA-256"),
        (
            CHILD_CONTRACT_SHA256,
            "pilot-v2.11.1-science",
            CHILD_GIT_COMMIT,
            "child_git_tag",
        ),
        (CHILD_CONTRACT_SHA256, CHILD_GIT_TAG, "B" * 40, "lowercase 40-hex"),
    ],
)
def test_child_release_binding_fails_closed_before_source_audit(
    contract_sha256: str,
    tag: str,
    commit: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        parent_import,
        "_audit_sources",
        lambda **_: pytest.fail("source audit must not run for an invalid child"),
    )
    with pytest.raises(parent_import.PilotV2112ParentImportError, match=message):
        parent_import.build_v2112_parent_import(
            repo_root=ROOT,
            child_contract_sha256=contract_sha256,
            child_git_tag=tag,
            child_git_commit=commit,
        )


def test_symlinked_parent_root_is_rejected(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(
        parent_import.PilotV2112ParentImportError,
        match="path contains a symlink",
    ):
        parent_import.verify_v2112_parent_sources(
            repo_root=ROOT,
            parent_science_root=linked_parent,
        )


def test_live_source_replay_and_parent_receipt_are_zero_call(
    live_parent_import: dict[str, Any],
) -> None:
    verified_sources = parent_import.verify_v2112_parent_sources(
        repo_root=ROOT,
        parent_science_root=PARENT_ROOT,
        grandparent_science_root=GRANDPARENT_ROOT,
    )
    verified = parent_import.verify_v2112_parent_import_receipt(
        live_parent_import,
        repo_root=ROOT,
        parent_science_root=PARENT_ROOT,
        grandparent_science_root=GRANDPARENT_ROOT,
        child_contract_sha256=CHILD_CONTRACT_SHA256,
        child_git_tag=CHILD_GIT_TAG,
        child_git_commit=CHILD_GIT_COMMIT,
    )

    assert verified_sources["cumulative_parent_budget_debit"] == (
        verified["cumulative_parent_budget_debit"]
    )
    assert verified["terminal_parent_denominator"]["status_counts"] == {
        "complete": 3,
        "failed": 2,
        "integrity-stopped": 131,
    }
    policy = verified["import_policy"]
    assert policy["provider_construction_during_import"] is False
    assert policy["provider_calls_during_import"] == 0
    assert policy["hosted_provider_calls_during_import"] == 0
    assert policy["imported_preflight_samples"] == 0
    assert policy["imported_checkpoint_artifacts"] == []
    assert policy["imported_p95_authorities"] == []
    assert policy["imported_effect_cells"] == 0
    assert policy["copied_file_count"] == 0
    assert policy["copied_byte_count"] == 0
    assert policy["raw_tree_copied"] is False

    failed_keys = set(_all_keys(verified["failed_preflight_audit"]))
    assert "samples" not in failed_keys
    assert "usage_rows" not in failed_keys
    assert "checkpoint" not in failed_keys
    assert "p95_authority" not in failed_keys


def test_live_rebind_exposes_only_calibration_and_capability_wrappers(
    live_parent_import: dict[str, Any],
) -> None:
    calibration = parent_import.calibration_wrapper_from_v2112_receipt(
        live_parent_import
    )
    wrappers = parent_import.capability_wrappers_from_v2112_receipt(live_parent_import)

    assert calibration["calibration"]["q_ref"] == 63.50397933257746
    assert calibration["calibration"]["selected_utility_profile"]["profile_id"] == (
        "nu-0.5"
    )
    assert calibration["provider_calls_current_attempt"] == 0
    assert set(wrappers) == MODELS
    for model_id, wrapper in wrappers.items():
        capability = wrapper["capability"]
        assert capability["model_id"] == model_id
        assert capability["historical_source_calls"] == 30
        assert len(capability["samples"]["action"]) == 24
        assert len(capability["samples"]["semantic"]) == 6
        assert len(capability["usage_rows"]) == 30
        assert wrapper["provider_construction_current_attempt"] is False
        assert wrapper["provider_calls_current_attempt"] == 0
        assert wrapper["imported_preflight_samples"] == 0
        assert wrapper["imported_checkpoint_artifacts"] == []
        assert wrapper["imported_p95_authorities"] == []


def test_live_bootstrap_source_is_capability_wrapper_not_failed_preflight(
    live_parent_import: dict[str, Any],
) -> None:
    for model_id in sorted(MODELS):
        source = parent_import.verified_v2112_capability_source(
            live_parent_import,
            model_id=model_id,
            repo_root=ROOT,
            parent_science_root=PARENT_ROOT,
            grandparent_science_root=GRANDPARENT_ROOT,
            child_contract_sha256=CHILD_CONTRACT_SHA256,
            child_git_tag=CHILD_GIT_TAG,
            child_git_commit=CHILD_GIT_COMMIT,
        )
        path = source["source_path"]
        assert source["schema_version"] == (
            parent_import.V2112_VERIFIED_CAPABILITY_SOURCE_SCHEMA_VERSION
        )
        assert source["payload"]["schema_version"] == (
            parent_import.V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION
        )
        assert path.endswith("/capability.json")
        assert "long-context-preflight" not in path
        assert "bootstrap" not in Path(path).name
        assert "journal" not in Path(path).name
        assert source["run_spec"]["stage_id"] == "capability-gate"
        assert source["provider_construction_during_verification"] is False
        assert source["provider_calls_during_verification"] == 0
        assert source["failed_preflight_samples_imported"] == 0
        assert source["checkpoint_artifacts_imported"] == []
        assert source["p95_authorities_imported"] == []

        inherited = parent_import.verified_v2112_inherited_capability_binding(
            live_parent_import,
            model_id=model_id,
            repo_root=ROOT,
            parent_science_root=PARENT_ROOT,
            grandparent_science_root=GRANDPARENT_ROOT,
            child_contract_sha256=CHILD_CONTRACT_SHA256,
            child_git_tag=CHILD_GIT_TAG,
            child_git_commit=CHILD_GIT_COMMIT,
        )
        assert inherited["payload"]["schema_version"] == (
            parent_import.V2112_CAPABILITY_WRAPPER_SCHEMA_VERSION
        )
        assert inherited["provider_construction_during_verification"] is False
        assert inherited["provider_calls_during_verification"] == 0


def test_resealed_receipt_scope_tamper_is_rejected(
    live_parent_import: dict[str, Any],
) -> None:
    tampered = copy.deepcopy(live_parent_import)
    tampered["failed_preflight_audit"]["gpt52_main"]["samples_exported"] = 1
    tampered.pop("integrity")
    tampered = parent_import._seal(tampered)

    with pytest.raises(
        parent_import.PilotV2112ParentImportError,
        match="differs from exact source replay",
    ):
        parent_import.verify_v2112_parent_import_receipt(
            tampered,
            repo_root=ROOT,
            parent_science_root=PARENT_ROOT,
            grandparent_science_root=GRANDPARENT_ROOT,
            child_contract_sha256=CHILD_CONTRACT_SHA256,
            child_git_tag=CHILD_GIT_TAG,
            child_git_commit=CHILD_GIT_COMMIT,
        )


def test_resealed_source_manifest_tamper_is_rejected() -> None:
    _require_immutable_sources()
    tampered = parent_import.load_v2112_source_manifest(repo_root=ROOT)
    tampered["terminal_denominator"]["scientific_matrix_complete"] = True
    tampered.pop("integrity")
    tampered = parent_import._seal(tampered)

    with pytest.raises(
        parent_import.PilotV2112ParentImportError,
        match="differs from immutable source replay",
    ):
        parent_import.validate_v2112_source_manifest(
            tampered,
            parent_science_root=PARENT_ROOT,
            grandparent_science_root=GRANDPARENT_ROOT,
        )
