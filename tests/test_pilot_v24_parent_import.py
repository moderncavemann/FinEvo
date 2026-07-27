from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any

import pytest

from verified_memory import pilot_v24_parent_import as parent_import
from verified_memory.pilot_budget import ParentBudgetDebit
from verified_memory.pilot_contract import load_pilot_contract


ROOT = Path(__file__).resolve().parents[1]
FULL_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_4.yaml"
OVERLAY_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_4_overlay.yaml"
PARENT_SOURCE_PATH = (
    ROOT / "experiments" / "pilot_v2_4_parent_source_manifest.json"
)
PARENT_SOURCE_FILE_SHA256 = (
    "d6a867cd7add43818127af7778a447d579ac1ab31ed6d053bcd29d69b3cf0f33"
)
CHILD_COMMIT = "a" * 40


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _parent_p95_receipt(
    profile_id: str,
    runtime_model: str,
    served_model: str,
) -> dict[str, Any]:
    reservations: dict[str, Any] = {}
    for call_kind, prompt_tokens, completion_tokens in (
        ("action", 120, 24),
        ("semantic", 180, 40),
    ):
        reservations[call_kind] = {
            "authority": {
                "pilot_contract_hash": "b" * 64,
                "pilot_tag": "pilot-v2.3-science",
                "call_kind": call_kind,
            },
            "reservation": {
                "sample_count": 12,
                "reserve_multiplier": 1.25,
                "raw_p95": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost_usd": 0.01,
                },
                "reserved_p95": {
                    "prompt_tokens": int(prompt_tokens * 1.25),
                    "completion_tokens": int(completion_tokens * 1.25),
                    "cost_usd": 0.0125,
                },
            },
        }
    return parent_import._seal(
        {
            "schema_version": "synthetic-parent-observed-p95-v1",
            "model": {
                "model_id": profile_id,
                "runtime_model": runtime_model,
                "served_model": served_model,
            },
            "reservations": {runtime_model: reservations},
        }
    )


def _prepare_child_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    child_root = tmp_path / "child"
    contract_path = child_root / "experiments" / "pilot_v2_4.yaml"
    contract_path.parent.mkdir(parents=True)
    shutil.copyfile(FULL_CONTRACT_PATH, contract_path)
    shutil.copyfile(
        PARENT_SOURCE_PATH,
        contract_path.parent / "pilot_v2_4_parent_source_manifest.json",
    )
    contract = load_pilot_contract(contract_path)

    raw_root = child_root / "experiment_results" / "pilot-v2.4" / "raw"
    snapshot_path = (
        raw_root
        / "parent-import"
        / "parent_snapshots"
        / "gpt52_main.observed_p95_parent.json"
    )
    snapshot_path.parent.mkdir(parents=True)
    runtime_model = "openai/gpt-5.2-2025-12-11"
    served_model = "gpt-5.2-2025-12-11"
    parent_receipt = _parent_p95_receipt(
        "gpt52_main",
        runtime_model,
        served_model,
    )
    snapshot_raw = _json_bytes(parent_receipt)
    snapshot_path.write_bytes(snapshot_raw)
    source_manifest = {
        "parent": {
            "contract_canonical_sha256": "c" * 64,
            "science_tag": "pilot-v2.3-science",
            "science_commit": "d" * 40,
        },
        "observed_p95_sources": {
            "gpt52_main": {
                "use": "v2.4-scientific-projection-authority",
                "path": "experiment_results/synthetic/gpt52_main/receipt.json",
                "file_sha256": hashlib.sha256(snapshot_raw).hexdigest(),
                "content_sha256": parent_receipt["integrity"][
                    "content_sha256"
                ],
                "runtime_model": runtime_model,
                "served_model": served_model,
            }
        },
    }
    receipt = parent_import._build_child_p95_receipt(
        repo_root=child_root,
        raw_root=raw_root,
        contract=contract,
        child_git_tag=parent_import.V24_SCIENCE_TAG,
        child_git_commit=CHILD_COMMIT,
        source_manifest=source_manifest,
        profile_id="gpt52_main",
        parent_receipt=parent_receipt,
        parent_snapshot_path=snapshot_path,
        parent_snapshot_raw=snapshot_raw,
    )
    monkeypatch.setattr(
        parent_import,
        "_load_source_manifest",
        lambda _root: (source_manifest, b'{"synthetic":true}\n'),
    )
    monkeypatch.setattr(
        parent_import,
        "_validate_child_contract",
        lambda *_args, **_kwargs: None,
    )
    return {
        "child_root": child_root,
        "contract": contract,
        "raw_root": raw_root,
        "receipt": receipt,
        "snapshot_path": snapshot_path,
        "snapshot_raw": snapshot_raw,
        "source_manifest": source_manifest,
    }


def test_draft_contract_reads_exact_parent_debit_without_external_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(OVERLAY_CONTRACT_PATH)

    def forbidden_subprocess(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("draft parent-debit lookup must not run subprocesses")

    monkeypatch.setattr(parent_import.subprocess, "run", forbidden_subprocess)
    debit = parent_import.parent_budget_debit_for_v24(
        contract,
        repo_root=ROOT,
    )

    assert debit is not None
    assert debit.stage_bucket == "parent_v23"
    assert debit.cost_usd == 3.212770875
    assert debit.hosted_completions == 184
    assert debit.storage_bytes == 4_196_087
    assert debit.record_sha256 == parent_import.V24_PARENT_DEBIT_RECORD_SHA256


def test_json_copy_thaws_nested_mappingproxy_and_tuple() -> None:
    source = MappingProxyType(
        {
            "outer": (
                MappingProxyType(
                    {
                        "inner": (
                            1,
                            MappingProxyType({"leaf": ("x", "y")}),
                        )
                    }
                ),
                2,
            )
        }
    )

    copied = parent_import._json_copy(source)

    assert copied == {
        "outer": [
            {"inner": [1, {"leaf": ["x", "y"]}]},
            2,
        ]
    }
    assert isinstance(copied, dict)
    assert isinstance(copied["outer"], list)
    assert isinstance(copied["outer"][0]["inner"], list)
    json.dumps(copied, sort_keys=True, allow_nan=False)


def test_tracked_parent_manifest_self_hash_and_fixed_debit() -> None:
    manifest, raw = parent_import._load_source_manifest(ROOT)

    assert manifest["schema_version"] == (
        parent_import.V24_PARENT_SOURCE_MANIFEST_SCHEMA_VERSION
    )
    assert hashlib.sha256(raw).hexdigest() == PARENT_SOURCE_FILE_SHA256
    assert manifest["integrity"]["content_sha256"] == (
        parent_import.V24_PARENT_SOURCE_MANIFEST_CONTENT_SHA256
    )
    assert parent_import._bound_content_sha256(manifest) == (
        parent_import.V24_PARENT_SOURCE_MANIFEST_CONTENT_SHA256
    )
    debit = ParentBudgetDebit.from_dict(manifest["cumulative_budget_debit"])
    assert debit.to_dict() == manifest["cumulative_budget_debit"]
    assert debit.record_sha256 == parent_import.V24_PARENT_DEBIT_RECORD_SHA256
    assert (
        manifest["inheritance_policy"]["provider_calls"] == 0
    )


def test_boundary_gpt56_cannot_build_dispatch_authority(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(FULL_CONTRACT_PATH)
    message = "boundary-only parent model cannot create V2.4 dispatch authority"

    with pytest.raises(parent_import.PilotV24ParentImportError, match=message):
        parent_import._build_child_p95_receipt(
            repo_root=tmp_path,
            raw_root=tmp_path,
            contract=contract,
            child_git_tag=parent_import.V24_SCIENCE_TAG,
            child_git_commit=CHILD_COMMIT,
            source_manifest={},
            profile_id="gpt56_diagnostic",
            parent_receipt={},
            parent_snapshot_path=tmp_path / "unused.json",
            parent_snapshot_raw=b"",
        )
    with pytest.raises(parent_import.PilotV24ParentImportError, match=message):
        parent_import._build_child_projection(
            contract=contract,
            child_git_tag=parent_import.V24_SCIENCE_TAG,
            child_git_commit=CHILD_COMMIT,
            profile_id="gpt56_diagnostic",
            child_receipt={},
            child_receipt_path=tmp_path / "unused.json",
        )


@pytest.mark.parametrize(
    ("tamper_kind", "message"),
    [
        (
            "child-receipt",
            "receipt differs from its tracked parent source",
        ),
        (
            "snapshot",
            "parent snapshot binding drifted",
        ),
        (
            "bound-hash",
            "parent snapshot binding drifted",
        ),
    ],
)
def test_tampered_child_receipt_snapshot_or_hash_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
    message: str,
) -> None:
    prepared = _prepare_child_authority(tmp_path, monkeypatch)
    receipt = parent_import._json_copy(prepared["receipt"])
    if tamper_kind == "child-receipt":
        runtime_model = receipt["model"]["runtime_model"]
        receipt["reservations"][runtime_model]["action"]["reservation"][
            "reserved_p95"
        ]["prompt_tokens"] += 1
        receipt = parent_import._seal(receipt)
    elif tamper_kind == "snapshot":
        prepared["snapshot_path"].write_bytes(
            prepared["snapshot_raw"] + b"\n"
        )
    else:
        receipt["parent_source"]["parent_receipt_file_sha256"] = "f" * 64
        receipt = parent_import._seal(receipt)

    with pytest.raises(parent_import.PilotV24ParentImportError, match=message):
        parent_import.verify_v24_inherited_p95_receipt(
            receipt,
            repo_root=prepared["child_root"],
            expected_git_commit=CHILD_COMMIT,
        )


def test_parent_import_persists_without_provider_construction_or_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from verified_memory import observed_p95_authority, pilot_orchestrator

    child_root = tmp_path / "child"
    parent_root = tmp_path / "parent"
    child_contract_path = child_root / "experiments" / "pilot_v2_4.yaml"
    child_contract_path.parent.mkdir(parents=True)
    parent_root.mkdir()
    shutil.copyfile(FULL_CONTRACT_PATH, child_contract_path)
    shutil.copyfile(
        PARENT_SOURCE_PATH,
        child_contract_path.parent / "pilot_v2_4_parent_source_manifest.json",
    )
    contract = load_pilot_contract(child_contract_path)
    raw_root = child_root / "experiment_results" / "pilot-v2.4" / "raw"

    profile_models = {
        "gpt52_main": (
            "openai/gpt-5.2-2025-12-11",
            "gpt-5.2-2025-12-11",
            "v2.4-scientific-projection-authority",
        ),
        "llama33_local_controlled": (
            "ollama/llama3.3:70b-instruct-q4_K_M",
            "llama3.3:70b-instruct-q4_K_M",
            "v2.4-scientific-projection-authority",
        ),
        "gpt56_diagnostic": (
            "openai/gpt-5.6-sol",
            "gpt-5.6-sol",
            "parent-boundary-audit-only-no-v2.4-dispatch-authority",
        ),
    }
    sources: dict[str, Any] = {}
    for profile_id, (runtime_model, served_model, use) in profile_models.items():
        receipt = _parent_p95_receipt(
            profile_id,
            runtime_model,
            served_model,
        )
        raw = _json_bytes(receipt)
        relative = Path(
            "experiment_results",
            "synthetic-parent",
            profile_id,
            "observed_p95_authority_receipt.json",
        )
        path = parent_root / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
        sources[profile_id] = {
            "use": use,
            "path": relative.as_posix(),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "content_sha256": receipt["integrity"]["content_sha256"],
            "runtime_model": runtime_model,
            "served_model": served_model,
        }

    tracked_manifest = json.loads(PARENT_SOURCE_PATH.read_text(encoding="utf-8"))
    source_manifest = {
        "parent": {
            "contract_canonical_sha256": "b" * 64,
            "science_tag": "pilot-v2.3-science",
            "science_commit": "c" * 40,
        },
        "terminal_denominator": {
            "registered_cells": 174,
            "status_counts": {"complete": 174},
        },
        "cumulative_budget_debit": tracked_manifest[
            "cumulative_budget_debit"
        ],
        "observed_p95_sources": sources,
    }
    source_manifest_raw = b'{"synthetic":true}\n'
    run_snapshot = {
        "ledger_sha256": "d" * 64,
        "events": [{"event_sha256": "e" * 64}],
    }
    budget_snapshot = {
        "ledger_sha256": "f" * 64,
        "events": [{"event_sha256": "1" * 64}],
        "event_chain_head": "1" * 64,
        "committed": {
            "cost_usd": 3.212770875,
            "completions": 184,
            "storage_bytes": 4_196_087,
        },
    }

    monkeypatch.setattr(
        parent_import,
        "_load_source_manifest",
        lambda _root: (source_manifest, source_manifest_raw),
    )
    monkeypatch.setattr(
        parent_import,
        "_validate_child_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        parent_import,
        "_verify_parent_git",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        parent_import,
        "_verify_parent_contract",
        lambda *_args, **_kwargs: contract,
    )
    monkeypatch.setattr(
        parent_import,
        "_verify_parent_ledgers",
        lambda *_args, **_kwargs: (run_snapshot, budget_snapshot),
    )
    monkeypatch.setattr(
        parent_import,
        "_verify_parent_bound_files",
        lambda *_args, **_kwargs: None,
    )

    def synthetic_binding(
        path: str,
        **_kwargs: Any,
    ) -> dict[str, str]:
        row = next(
            value for value in sources.values() if value["path"] == path
        )
        return {
            "receipt_file_sha256": row["file_sha256"],
            "receipt_content_sha256": row["content_sha256"],
        }

    monkeypatch.setattr(
        observed_p95_authority,
        "verified_observed_p95_authority_binding",
        synthetic_binding,
    )
    provider_calls: list[str] = []

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> None:
        provider_calls.append("forbidden")
        raise AssertionError("parent import must not construct or call a provider")

    monkeypatch.setattr(
        pilot_orchestrator,
        "_provider_for_profile",
        forbidden_provider,
    )
    monkeypatch.setattr(
        pilot_orchestrator,
        "create_llm_provider",
        forbidden_provider,
    )

    result = parent_import.persist_v24_parent_import(
        contract=contract,
        repo_root=child_root,
        raw_root=raw_root,
        parent_repo_root=parent_root,
        child_git_tag=parent_import.V24_SCIENCE_TAG,
        child_git_commit=CHILD_COMMIT,
    )

    assert provider_calls == []
    assert result["provider_calls"] == 0
    assert result["scientific_evidence"] is False
    assert result["imported_profiles"] == [
        "gpt52_main",
        "llama33_local_controlled",
    ]
    assert result["boundary_only_profiles"] == ["gpt56_diagnostic"]
    assert not parent_import.inherited_p95_receipt_path(
        raw_root,
        "gpt56_diagnostic",
    ).exists()
    assert not parent_import.inherited_projection_path(
        raw_root,
        "gpt56_diagnostic",
    ).exists()
