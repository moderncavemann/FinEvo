from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from verified_memory.pilot_budget import ParentBudgetDebit
from verified_memory.pilot_v24_parent_import import _seal
import verified_memory.pilot_v26_parent_import as v26


COMMIT = "b" * 40
CONTRACT_HASH = "c" * 64


def _raw_json(value: dict) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _fake_contract(manifest: dict, *, status: str = "frozen"):
    source_hashes = (
        (None, None)
        if status == "draft"
        else (
            v26.V26_SOURCE_MANIFEST_FILE_SHA256,
            v26.V26_SOURCE_MANIFEST_CONTENT_SHA256,
        )
    )
    cumulative_prior = dict(manifest["cumulative_budget_debit"])
    cumulative_prior.pop("schema_version", None)
    return SimpleNamespace(
        contract_id=v26.V26_CONTRACT_ID,
        status=status,
        canonical_hash=CONTRACT_HASH,
        implementation={"required_git_tag": v26.V26_SCIENCE_TAG},
        p95_authority_retry_amendment={
            "source_manifest": {
                "path": v26.V26_SOURCE_MANIFEST_PATH.as_posix(),
                "schema_version": v26.V26_SOURCE_MANIFEST_SCHEMA_VERSION,
                "file_sha256": source_hashes[0],
                "content_sha256": source_hashes[1],
            },
            "failure_classification": {
                "parent_contract_sha256": (
                    v26.V25_CONTRACT_CANONICAL_SHA256
                ),
                "run_ledger_internal_sha256": (
                    v26.V25_RUN_LEDGER_INTERNAL_SHA256
                ),
                "budget_ledger_internal_sha256": (
                    v26.V25_BUDGET_LEDGER_INTERNAL_SHA256
                ),
                "status_counts": manifest["v2_5_terminal_parent"][
                    "terminal_denominator"
                ]["status_counts"],
            },
            "budget_carry_forward": {
                "cumulative_prior": cumulative_prior,
                "budget_reset": False,
            },
        },
    )


def _parent_receipt(profile_id: str, runtime: str, served: str) -> dict:
    def one(call_kind: str) -> dict:
        return {
            "authority": {
                "pilot_contract_hash": "a" * 64,
                "pilot_tag": v26.V25_SCIENCE_TAG,
                "call_kind": call_kind,
            },
            "reservation": {
                "completion_tokens_p95": 101 if call_kind == "action" else 51,
                "input_tokens_p95": 202 if call_kind == "action" else 102,
            },
        }

    return _seal(
        {
            "schema_version": (
                v26.V25_INHERITED_P95_RECEIPT_SCHEMA_VERSION
            ),
            "contract": {"contract_id": v26.V25_CONTRACT_ID},
            "git": {
                "tag": v26.V25_SCIENCE_TAG,
                "commit": v26.V25_SCIENCE_COMMIT,
            },
            "model": {
                "model_id": profile_id,
                "runtime_model": runtime,
                "served_model": served,
            },
            "reservations": {
                runtime: {
                    "action": one("action"),
                    "semantic": one("semantic"),
                }
            },
            "scientific_evidence": False,
        }
    )


def _synthetic_manifest(parent_receipts: dict[str, tuple[dict, bytes]]) -> dict:
    inherited = {}
    for profile_id, (receipt, raw) in parent_receipts.items():
        model = receipt["model"]
        inherited[profile_id] = {
            "receipt": {
                "path": (
                    "experiment_results/pilot-v2.5/raw/parent-import/"
                    f"observed_p95/{profile_id}/"
                    "observed_p95_authority_receipt.json"
                ),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "content_sha256": receipt["integrity"]["content_sha256"],
                "byte_size": len(raw),
                "schema_version": (
                    v26.V25_INHERITED_P95_RECEIPT_SCHEMA_VERSION
                ),
            },
            "runtime_model": model["runtime_model"],
            "served_model": model["served_model"],
            "use": "test-only",
        }
    debit = ParentBudgetDebit(
        parent_contract_sha256=v26.V25_CONTRACT_CANONICAL_SHA256,
        parent_run_ledger_sha256=v26.V25_RUN_LEDGER_INTERNAL_SHA256,
        parent_budget_ledger_sha256=(
            v26.V25_BUDGET_LEDGER_INTERNAL_SHA256
        ),
        stage_bucket="parent_v23",
        cost_usd=3.212770875,
        hosted_completions=184,
        storage_bytes=6_303_635,
    )
    assert debit.record_sha256 == v26.V26_PARENT_DEBIT_RECORD_SHA256
    return {
        "schema_version": v26.V26_SOURCE_MANIFEST_SCHEMA_VERSION,
        "v2_5_terminal_parent": {
            "contract": {
                "contract_id": v26.V25_CONTRACT_ID,
                "path": "experiments/pilot_v2_5.yaml",
                "file_sha256": v26.V25_CONTRACT_FILE_SHA256,
                "canonical_sha256": v26.V25_CONTRACT_CANONICAL_SHA256,
            },
            "release": {
                "science_tag": v26.V25_SCIENCE_TAG,
                "science_tag_object": v26.V25_SCIENCE_TAG_OBJECT,
                "science_commit": v26.V25_SCIENCE_COMMIT,
                "raw_root": v26.V25_RAW_ROOT.as_posix(),
            },
            "ledgers": {
                "run": {
                    "path": "experiment_results/pilot-v2.5/raw/run_ledger.json",
                    "file_sha256": v26.V25_RUN_LEDGER_FILE_SHA256,
                    "internal_sha256": v26.V25_RUN_LEDGER_INTERNAL_SHA256,
                    "event_count": 213,
                    "event_chain_head": v26.V25_RUN_LEDGER_EVENT_HEAD,
                },
                "budget": {
                    "path": (
                        "experiment_results/pilot-v2.5/raw/budget_ledger.json"
                    ),
                    "file_sha256": v26.V25_BUDGET_LEDGER_FILE_SHA256,
                    "internal_sha256": (
                        v26.V25_BUDGET_LEDGER_INTERNAL_SHA256
                    ),
                    "event_count": 32,
                    "event_chain_head": v26.V25_BUDGET_LEDGER_EVENT_HEAD,
                },
            },
            "terminal_denominator": {
                "registered_cells": 211,
                "status_counts": {
                    "complete": 2,
                    "failed": 14,
                    "integrity-stopped": 195,
                },
            },
            "raw_snapshot": {
                "file_count": 61,
                "storage_bytes": 1_589_313,
                "inventory_sha256": v26.V25_RAW_INVENTORY_SHA256,
            },
        },
        "inherited_observed_p95_authority": inherited,
        "parent_import_receipt": {
            "path": (
                "experiment_results/pilot-v2.5/raw/parent-import/"
                "parent_import_receipt.json"
            ),
            "file_sha256": "",
            "content_sha256": (
                v26.V25_PARENT_IMPORT_RECEIPT_CONTENT_SHA256
            ),
            "byte_size": 0,
        },
        "cumulative_budget_debit": debit.to_dict(),
        "integrity": {
            "content_sha256": v26.V26_SOURCE_MANIFEST_CONTENT_SHA256
        },
    }


def _context() -> tuple[dict, bytes, dict[str, tuple[dict, bytes]]]:
    receipts = {}
    identities = {
        "gpt52_main": (
            "openai/gpt-5.2-2025-12-11",
            "gpt-5.2-2025-12-11",
        ),
        "llama33_local_controlled": (
            "ollama/llama3.3:70b-instruct-q4_K_M",
            "llama3.3:70b-instruct-q4_K_M",
        ),
    }
    for profile_id, (runtime, served) in identities.items():
        receipt = _parent_receipt(profile_id, runtime, served)
        receipts[profile_id] = (receipt, _raw_json(receipt))
    manifest = _synthetic_manifest(receipts)
    return manifest, b"synthetic-manifest", receipts


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    manifest: dict,
    manifest_raw: bytes,
    contract,
) -> None:
    monkeypatch.setattr(
        v26,
        "_load_source_manifest",
        lambda _root: (manifest, manifest_raw),
    )
    monkeypatch.setattr(v26, "_validate_child_contract", lambda *a, **k: None)
    monkeypatch.setattr(
        v26,
        "_load_child_contract_from_receipt",
        lambda *a, **k: contract,
    )
    monkeypatch.setattr(
        v26,
        "_child_contract_binding",
        lambda **k: {
            "path": v26.V26_EXPANDED_CONTRACT_PATH.as_posix(),
            "file_sha256": "d" * 64,
            "contract_id": v26.V26_CONTRACT_ID,
            "contract_sha256": CONTRACT_HASH,
        },
    )


def test_tracked_source_manifest_and_cumulative_debit_are_exact() -> None:
    root = Path(v26.__file__).resolve().parents[1]
    manifest, raw = v26._load_source_manifest(root)
    assert hashlib.sha256(raw).hexdigest() == (
        v26.V26_SOURCE_MANIFEST_FILE_SHA256
    )
    assert manifest["integrity"]["content_sha256"] == (
        v26.V26_SOURCE_MANIFEST_CONTENT_SHA256
    )
    debit = ParentBudgetDebit.from_dict(manifest["cumulative_budget_debit"])
    assert debit.stage_bucket == "parent_v23"
    assert debit.cost_usd == pytest.approx(3.212770875)
    assert debit.hosted_completions == 184
    assert debit.storage_bytes == 6_303_635
    assert debit.record_sha256 == v26.V26_PARENT_DEBIT_RECORD_SHA256


def test_published_evidence_is_complete_and_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    source_root = Path(v26.__file__).resolve().parents[1]
    manifest, _ = v26._load_source_manifest(source_root)
    v26._verify_published_evidence(source_root, manifest)

    copied_root = tmp_path / "child"
    evidence_relative = Path("evidence/current_v2/pilot-v2.5")
    shutil.copytree(
        source_root / evidence_relative,
        copied_root / evidence_relative,
    )
    aggregate = copied_root / evidence_relative / "aggregate.json"
    aggregate.write_bytes(aggregate.read_bytes() + b"\n")
    with pytest.raises(
        v26.PilotV26ParentImportError,
        match="published aggregate bytes drifted",
    ):
        v26._verify_published_evidence(copied_root, manifest)


def test_runtime_rebuild_uses_local_snapshot_and_fails_on_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, manifest_raw, receipts = _context()
    contract = _fake_contract(manifest)
    _patch_runtime(monkeypatch, manifest, manifest_raw, contract)
    root = tmp_path / "child"
    root.mkdir()
    raw_root = root.joinpath(*v26.V26_RAW_ROOT.parts)
    profile_id = "gpt52_main"
    parent_receipt, parent_raw = receipts[profile_id]
    snapshot = v26.parent_snapshot_path(raw_root, profile_id)
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(parent_raw)
    child = v26._build_child_p95_receipt(
        repo_root=root,
        contract=contract,
        child_git_tag=v26.V26_SCIENCE_TAG,
        child_git_commit=COMMIT,
        source_manifest=manifest,
        source_manifest_raw=manifest_raw,
        profile_id=profile_id,
        parent_receipt=parent_receipt,
        parent_snapshot=snapshot,
        parent_snapshot_raw=parent_raw,
    )

    reservations = v26.verify_v26_inherited_p95_receipt(
        child,
        repo_root=root,
        expected_git_commit=COMMIT,
    )
    assert set(reservations) == {"openai/gpt-5.2-2025-12-11"}

    snapshot.write_bytes(parent_raw + b"\n")
    with pytest.raises(v26.PilotV26ParentImportError, match="snapshot"):
        v26.verify_v26_inherited_p95_receipt(
            child,
            repo_root=root,
            expected_git_commit=COMMIT,
        )


def test_persist_calls_both_v25_verifiers_and_snapshots_exact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, manifest_raw, receipts = _context()
    contract = _fake_contract(manifest)
    child_root = tmp_path / "child"
    parent_root = tmp_path / "parent"
    child_root.mkdir()
    parent_root.mkdir()
    child_raw = child_root.joinpath(*v26.V26_RAW_ROOT.parts)
    child_raw.mkdir(parents=True)

    parent_import_raw = b"immutable-v2.5-parent-import"
    parent_import = manifest["parent_import_receipt"]
    parent_import["byte_size"] = len(parent_import_raw)
    parent_import["file_sha256"] = hashlib.sha256(
        parent_import_raw
    ).hexdigest()
    parent_import_path = parent_root / parent_import["path"]
    parent_import_path.parent.mkdir(parents=True)
    parent_import_path.write_bytes(parent_import_raw)
    for profile_id, (_, raw) in receipts.items():
        path = (
            parent_root
            / manifest["inherited_observed_p95_authority"][profile_id][
                "receipt"
            ]["path"]
        )
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)

    _patch_runtime(monkeypatch, manifest, manifest_raw, contract)
    monkeypatch.setattr(v26, "_verify_parent_git", lambda *a, **k: None)
    monkeypatch.setattr(
        v26,
        "_verify_parent_contract",
        lambda *a, **k: SimpleNamespace(),
    )
    monkeypatch.setattr(
        v26,
        "_verify_parent_ledgers",
        lambda *a, **k: (
            {
                "ledger_sha256": v26.V25_RUN_LEDGER_INTERNAL_SHA256,
                "events": [
                    {"event_sha256": v26.V25_RUN_LEDGER_EVENT_HEAD}
                ]
                * 213,
            },
            {
                "ledger_sha256": v26.V25_BUDGET_LEDGER_INTERNAL_SHA256,
                "events": [{}] * 32,
                "event_chain_head": v26.V25_BUDGET_LEDGER_EVENT_HEAD,
            },
        ),
    )
    monkeypatch.setattr(
        v26,
        "_git",
        lambda _root, *args: (
            v26.V25_SCIENCE_COMMIT if args == ("rev-parse", "HEAD") else ""
        ),
    )
    monkeypatch.setattr(
        v26,
        "_verify_parent_raw_inventory",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        v26,
        "_verify_published_evidence",
        lambda *a, **k: None,
    )

    calls: list[tuple[str, str]] = []

    def verify_parent(*args, **kwargs):
        calls.append(("parent", str(args[0])))
        return {
            "integrity": {
                "content_sha256": (
                    v26.V25_PARENT_IMPORT_RECEIPT_CONTENT_SHA256
                )
            }
        }

    def verify_child(path, **kwargs):
        profile_id = next(
            profile
            for profile in receipts
            if f"/{profile}/" in str(path)
        )
        calls.append(("child", profile_id))
        source = manifest["inherited_observed_p95_authority"][profile_id][
            "receipt"
        ]
        return {
            "receipt_file_sha256": source["file_sha256"],
            "receipt_content_sha256": source["content_sha256"],
        }

    monkeypatch.setattr(
        v26,
        "verify_v25_parent_import_receipt",
        verify_parent,
    )
    monkeypatch.setattr(
        v26,
        "verified_v25_inherited_p95_binding",
        verify_child,
    )
    monkeypatch.setattr(
        v26,
        "verify_v26_parent_import_receipt",
        lambda path, **k: json.loads(Path(path).read_text()),
    )

    result = v26.persist_v26_parent_import(
        contract=contract,
        repo_root=child_root,
        raw_root=child_raw,
        parent_repo_root=parent_root,
        child_git_tag=v26.V26_SCIENCE_TAG,
        child_git_commit=COMMIT,
    )

    assert result["provider_calls"] == 0
    assert result["scientific_evidence"] is False
    assert calls == [
        ("parent", manifest["parent_import_receipt"]["path"]),
        ("child", "gpt52_main"),
        ("child", "llama33_local_controlled"),
    ]
    for profile_id, (_, parent_raw) in receipts.items():
        assert (
            v26.parent_snapshot_path(child_raw, profile_id).read_bytes()
            == parent_raw
        )


def test_resealed_parent_import_receipt_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, manifest_raw, receipts = _context()
    contract = _fake_contract(manifest)
    _patch_runtime(monkeypatch, manifest, manifest_raw, contract)
    root = tmp_path / "child"
    root.mkdir()
    raw_root = root.joinpath(*v26.V26_RAW_ROOT.parts)
    for profile_id, (parent, parent_raw) in receipts.items():
        snapshot = v26.parent_snapshot_path(raw_root, profile_id)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(parent_raw)
        child = v26._build_child_p95_receipt(
            repo_root=root,
            contract=contract,
            child_git_tag=v26.V26_SCIENCE_TAG,
            child_git_commit=COMMIT,
            source_manifest=manifest,
            source_manifest_raw=manifest_raw,
            profile_id=profile_id,
            parent_receipt=parent,
            parent_snapshot=snapshot,
            parent_snapshot_raw=parent_raw,
        )
        child_path = v26.inherited_p95_receipt_path(raw_root, profile_id)
        child_path.parent.mkdir(parents=True, exist_ok=True)
        child_path.write_bytes(_raw_json(child))
        projection = v26._build_child_projection(
            contract=contract,
            child_git_tag=v26.V26_SCIENCE_TAG,
            child_git_commit=COMMIT,
            profile_id=profile_id,
            child_receipt=child,
            child_receipt_path=child_path,
        )
        v26.inherited_projection_path(
            raw_root,
            profile_id,
        ).write_bytes(_raw_json(projection))

    debit = ParentBudgetDebit.from_dict(manifest["cumulative_budget_debit"])
    receipt = v26._rebuild_v26_parent_import_receipt(
        repo_root=root,
        contract=contract,
        expected_git_commit=COMMIT,
        source_manifest=manifest,
        source_manifest_raw=manifest_raw,
        debit=debit,
    )
    assert (
        v26.verify_v26_parent_import_receipt(
            receipt,
            repo_root=root,
            contract=contract,
            expected_git_commit=COMMIT,
        )
        == receipt
    )
    tampered = dict(receipt)
    tampered["provider_calls"] = 1
    tampered = _seal(tampered)
    with pytest.raises(v26.PilotV26ParentImportError, match="sealed sources"):
        v26.verify_v26_parent_import_receipt(
            tampered,
            repo_root=root,
            contract=contract,
            expected_git_commit=COMMIT,
        )
