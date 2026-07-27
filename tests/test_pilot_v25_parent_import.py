from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from verified_memory import pilot_v25_parent_import as parent_import
from verified_memory.pilot_contract import load_pilot_contract


ROOT = Path(__file__).resolve().parents[1]
OVERLAY_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_5_overlay.yaml"


def _install_expected_receipt_fixture(
    monkeypatch: Any,
) -> tuple[Any, dict[str, Any]]:
    contract = load_pilot_contract(OVERLAY_CONTRACT_PATH)
    expected = parent_import._seal(
        {
            "schema_version": parent_import.V25_PARENT_IMPORT_SCHEMA_VERSION,
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "child_release": {
                "git_tag": parent_import.V25_SCIENCE_TAG,
                "git_commit": "a" * 40,
            },
            "parent_run_ledger": {
                "ledger_sha256": "1" * 64,
                "event_count": 176,
                "event_chain_head": "2" * 64,
            },
            "parent_budget_ledger": {
                "ledger_sha256": "3" * 64,
                "event_count": 22,
                "event_chain_head": "4" * 64,
            },
            "imported_projection_profiles": {
                "gpt52_main": {},
                "llama33_local_controlled": {},
            },
            "boundary_only_profiles": {
                "gpt56_diagnostic": {
                    "dispatch_authority": False,
                    "boundary_reason": parent_import._BOUNDARY_REASON,
                    "snapshot_path": (
                        "experiment_results/pilot-v2.5/raw/parent-import/"
                        "parent_snapshots/"
                        "gpt56_diagnostic.observed_p95_parent.json"
                    ),
                }
            },
            "provider_calls": 0,
            "scientific_evidence": False,
            "scientific_outcomes_observed_before_amendment": False,
            "claim_boundary": parent_import._CLAIM_BOUNDARY,
        }
    )
    monkeypatch.setattr(
        parent_import,
        "_load_v25_source_manifest",
        lambda *_args, **_kwargs: ({"manifest": "v2.5"}, b"v2.5"),
    )
    monkeypatch.setattr(
        parent_import,
        "_validate_child_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        parent_import,
        "_verify_v24_terminal_evidence",
        lambda *_args, **_kwargs: {"terminal": "v2.4"},
    )
    monkeypatch.setattr(
        parent_import,
        "_load_v23_source_manifest",
        lambda *_args, **_kwargs: ({"parent": "v2.3"}, b"v2.3"),
    )
    monkeypatch.setattr(
        parent_import,
        "parent_budget_debit_for_v25",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        parent_import,
        "_rebuild_v25_parent_import_receipt",
        lambda **_kwargs: copy.deepcopy(expected),
    )
    return contract, expected


def test_inherited_paths_have_exactly_one_parent_import_component() -> None:
    raw_root = ROOT / "experiment_results" / "pilot-v2.5" / "raw"

    receipt = parent_import.inherited_p95_receipt_path(
        raw_root,
        "gpt52_main",
    )
    projection = parent_import.inherited_projection_path(
        raw_root,
        "gpt52_main",
    )

    expected_root = raw_root / "parent-import" / "observed_p95" / "gpt52_main"
    assert receipt == expected_root / "observed_p95_authority_receipt.json"
    assert projection == expected_root / "projection_p95.json"
    assert receipt.parts.count("parent-import") == 1
    assert projection.parts.count("parent-import") == 1


def test_draft_contract_imports_exact_cumulative_parent_debit(
    monkeypatch: Any,
) -> None:
    contract = load_pilot_contract(OVERLAY_CONTRACT_PATH)

    def forbidden_subprocess(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("budget debit lookup must not invoke git or providers")

    monkeypatch.setattr(
        "verified_memory.pilot_v24_parent_import.subprocess.run",
        forbidden_subprocess,
    )
    debit = parent_import.parent_budget_debit_for_v25(
        contract,
        repo_root=ROOT,
    )

    assert debit is not None
    assert debit.stage_bucket == "parent_v23"
    assert debit.cost_usd == 3.212770875
    assert debit.hosted_completions == 184
    assert debit.storage_bytes == 4_714_322
    assert debit.parent_run_ledger_sha256 == (
        "6ef976205f37fe675169b05fcec8806c16085aceffdafeaa4a471a002f194fd1"
    )
    assert debit.record_sha256 == parent_import.V25_PARENT_DEBIT_RECORD_SHA256


def test_v24_terminal_package_verifies_before_v23_authority(
    monkeypatch: Any,
) -> None:
    contract = load_pilot_contract(OVERLAY_CONTRACT_PATH)
    manifest, raw = parent_import._load_v25_source_manifest(ROOT)
    parent_import._validate_child_contract(
        contract,
        source_manifest=manifest,
        source_manifest_raw=raw,
        require_frozen=False,
    )
    calls: list[str] = []
    real_terminal_verifier = parent_import._verify_v24_terminal_evidence

    def terminal_first(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("v2.4-terminal")
        return real_terminal_verifier(*args, **kwargs)

    def stop_before_v23(*_args: Any, **_kwargs: Any) -> None:
        calls.append("v2.3-authority")
        raise RuntimeError("stop after ordering assertion")

    monkeypatch.setattr(
        parent_import,
        "_verify_v24_terminal_evidence",
        terminal_first,
    )
    monkeypatch.setattr(
        parent_import,
        "_validate_child_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        parent_import,
        "_load_v23_source_manifest",
        stop_before_v23,
    )

    try:
        parent_import._persist_v25_parent_import_impl(
            contract=contract,
            repo_root=ROOT,
            raw_root=ROOT / "experiment_results" / "pilot-v2.5" / "raw",
            parent_repo_root=ROOT,
            child_git_tag=parent_import.V25_SCIENCE_TAG,
            child_git_commit="a" * 40,
        )
    except RuntimeError as exc:
        assert str(exc) == "stop after ordering assertion"
    else:  # pragma: no cover - monkeypatch always stops before parent access
        raise AssertionError("synthetic ordering stop was not reached")

    assert calls == ["v2.4-terminal", "v2.3-authority"]


def test_resealed_parent_run_ledger_tamper_is_rejected(
    monkeypatch: Any,
) -> None:
    contract, expected = _install_expected_receipt_fixture(monkeypatch)
    assert parent_import.verify_v25_parent_import_receipt(
        expected,
        repo_root=ROOT,
        contract=contract,
        expected_git_commit="a" * 40,
    ) == expected

    tampered = copy.deepcopy(expected)
    tampered["parent_run_ledger"]["event_count"] += 1
    tampered = parent_import._seal(tampered)

    with pytest.raises(
        parent_import.PilotV25ParentImportError,
        match="differs from frozen sources",
    ):
        parent_import.verify_v25_parent_import_receipt(
            tampered,
            repo_root=ROOT,
            contract=contract,
            expected_git_commit="a" * 40,
        )


def test_resealed_boundary_profile_tamper_is_rejected(
    monkeypatch: Any,
) -> None:
    contract, expected = _install_expected_receipt_fixture(monkeypatch)
    assert parent_import.verify_v25_parent_import_receipt(
        expected,
        repo_root=ROOT,
        contract=contract,
        expected_git_commit="a" * 40,
    ) == expected

    tampered = copy.deepcopy(expected)
    tampered["boundary_only_profiles"]["gpt56_diagnostic"][
        "dispatch_authority"
    ] = True
    tampered = parent_import._seal(tampered)

    with pytest.raises(
        parent_import.PilotV25ParentImportError,
        match="differs from frozen sources",
    ):
        parent_import.verify_v25_parent_import_receipt(
            tampered,
            repo_root=ROOT,
            contract=contract,
            expected_git_commit="a" * 40,
        )


def test_parent_import_verifier_rejects_malformed_expected_commit() -> None:
    contract = load_pilot_contract(OVERLAY_CONTRACT_PATH)

    with pytest.raises(
        parent_import.PilotV25ParentImportError,
        match="exactly 40 lowercase hex",
    ):
        parent_import.verify_v25_parent_import_receipt(
            {},
            repo_root=ROOT,
            contract=contract,
            expected_git_commit="not-a-release-commit",
        )
