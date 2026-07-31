from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import pytest

from verified_memory.pilot_budget import ParentBudgetDebit
from verified_memory.pilot_contract import canonical_sha256
from verified_memory import pilot_v211_parent_import as parent_import


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> bytes:
    raw = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _fake_audit(repo_root: Path) -> dict[str, object]:
    manifest = json.loads(
        (
            ROOT / "experiments" / "pilot_v2_11_source_manifest.json"
        ).read_text(encoding="utf-8")
    )
    return {
        "repo_root": repo_root,
        "manifest": manifest,
        "science_root": repo_root / "fake-science",
        "evidence_root": repo_root / "fake-evidence",
        "git": {
            "science_tag": parent_import.V2102_SCIENCE_TAG,
            "science_tag_object": parent_import.V2102_SCIENCE_TAG_OBJECT,
            "resolved_git_commit": parent_import.V2102_SCIENCE_COMMIT,
        },
        "prerequisites": {
            "q_ref_file_sha256": (
                "bc0ce9c3e4319d88b255b49c4d124b23f6f85965675f3b8a8685b07316de390f"
            ),
            "q_ref_content_sha256": (
                "50d75c846c5e9d2b58fb92faf674da8a06ebb3b0ba7f21a6b1b2ad689034c40c"
            ),
            "stage0_selection_file_sha256": (
                "701dff04abad00fb5e6d734168e6d4faeac6fd8b5ee4d8c1ed4aa0ce06bbc0eb"
            ),
            "stage0_selection_content_sha256": (
                "68c810055fc38683d3a8a7d597c54ffed4fb2c6332c2c02e1964b3ebfb61743c"
            ),
            "stage0_receipt_file_sha256": (
                "0b9f179ea4899da63acc6633e0e5bafefc4bddb7a23dfdfe8345eba8cdb2b0f1"
            ),
            "stage0_receipt_content_sha256": (
                "cb93b97e74a5a8ee9de6a3ce0320c4ebd1f774fd1250586b31da3b93f7fc5718"
            ),
        },
        "evidence": {
            "aggregate_json_sha256": (
                "4aad9e0030f8f7f66416fcb8471684e5ae1203eef645aa8060ca8838500d7869"
            ),
            "aggregate_csv_sha256": (
                "9a320b526b7b0ea43e2d37be7851ef68745c86940948039b12a9610b44e4f477"
            ),
            "checksums_sha256": (
                "b511d610e7b7f9201d6078acaced66e11ce0096ae34a5bb0ff3e318c1c041e03"
            ),
            "checksums_files_canonical_sha256": (
                "d27688de15fae4201685a726ce5a9fc34c674d8d292efdaa7ef1019b9f62647c"
            ),
            "package_manifest_sha256": (
                "a46cfeafcaccae02d50c11a7ae5671b46a732b9760ce4915017e98d43e82dd79"
            ),
            "publication_status": "complete-with-no-go",
            "denominator_status_counts": {"complete": 126, "failed": 85},
        },
    }


@pytest.fixture
def fake_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    repo_root = tmp_path / "child"
    repo_root.mkdir()
    audit = _fake_audit(repo_root)
    monkeypatch.setattr(
        parent_import,
        "_audit_sources",
        lambda **_kwargs: audit,
    )
    return repo_root


def test_tracked_source_manifest_is_exact_and_effect_blind() -> None:
    raw = (
        ROOT / "experiments" / "pilot_v2_11_source_manifest.json"
    ).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        parent_import.V211_SOURCE_MANIFEST_FILE_SHA256
    )
    manifest = parent_import._strict_json(
        raw,
        name="tracked source manifest",
    )
    verified = parent_import._verify_source_manifest(manifest)
    allowlist = verified["import_allowlist"]
    assert allowlist["q_ref"] == 63.50397933257746
    assert allowlist["selected_utility_profile"]["profile_id"] == "nu-0.5"
    assert allowlist["imported_effect_cells"] == 0
    assert allowlist["effect_metrics_observed"] is False
    assert allowlist["effect_artifact_paths"] == []
    assert allowlist["imported_p95_authorities"] == []
    assert allowlist["provider_construction"] is False
    assert allowlist["provider_calls"] == 0


def test_build_verify_and_budget_debit_are_zero_provider(
    fake_audit: Path,
) -> None:
    receipt = parent_import.build_v211_parent_import(
        repo_root=fake_audit
    )
    assert parent_import.verify_v211_parent_import_receipt(
        receipt,
        repo_root=fake_audit,
    ) == receipt
    assert receipt["import_policy"] == {
        "provider_construction": False,
        "provider_calls": 0,
        "imported_effect_cells": 0,
        "effect_metrics_observed": False,
        "effect_artifact_paths": [],
        "imported_p95_authorities": [],
        "raw_tree_copied": False,
        "copied_file_count": 0,
        "copied_byte_count": 0,
    }
    debit = parent_import.parent_budget_debit_for_v211(
        repo_root=fake_audit
    )
    assert isinstance(debit, ParentBudgetDebit)
    assert debit.cost_usd == 16.044922812500005
    assert debit.hosted_completions == 816
    assert debit.storage_bytes == 217010835
    assert debit.record_sha256 == (
        "c841dc4cbdfdb548c6917fbb2670c31ba3759f3d4f52ffb0fbb5b9d8bcbbc74d"
    )


def test_persist_is_small_idempotent_and_copies_no_raw_tree(
    fake_audit: Path,
) -> None:
    first = parent_import.persist_v211_parent_import(
        repo_root=fake_audit
    )
    second = parent_import.persist_v211_parent_import(
        repo_root=fake_audit
    )
    assert first == second
    path = Path(first["receipt"])
    assert path.is_file()
    assert path.stat().st_size < 20_000
    assert first["raw_tree_copied"] is False
    assert first["copied_file_count"] == 0
    assert first["copied_byte_count"] == 0
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize(
    ("field", "injected"),
    [
        ("imported_effect_cells", 1),
        ("effect_metrics_observed", True),
        ("effect_artifact_paths", ["aggregate.json"]),
        ("imported_p95_authorities", [{"model_id": "gpt52_main"}]),
        ("provider_construction", True),
        ("provider_calls", 1),
    ],
)
def test_receipt_rejects_effect_p95_or_provider_injection(
    fake_audit: Path,
    field: str,
    injected: object,
) -> None:
    receipt = parent_import.build_v211_parent_import(
        repo_root=fake_audit
    )
    unsigned = json.loads(json.dumps(receipt))
    unsigned.pop("integrity")
    unsigned["import_policy"][field] = injected
    tampered = parent_import._seal(unsigned)
    with pytest.raises(
        parent_import.PilotV211ParentImportError,
        match="differs from exact allowlist",
    ):
        parent_import.verify_v211_parent_import_receipt(
            tampered,
            repo_root=fake_audit,
        )


def test_source_manifest_rejects_effect_or_p95_injection() -> None:
    source = json.loads(
        (
            ROOT / "experiments" / "pilot_v2_11_source_manifest.json"
        ).read_text(encoding="utf-8")
    )
    source.pop("integrity")
    source["import_allowlist"]["imported_effect_cells"] = 1
    tampered = parent_import._seal(source)
    with pytest.raises(
        parent_import.PilotV211ParentImportError,
        match="allowlist drifted",
    ):
        parent_import._verify_source_manifest(tampered)


def test_guarded_source_rejects_path_escape_and_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    outside = tmp_path / "outside.json"
    root.mkdir()
    outside.write_text("{}\n", encoding="utf-8")
    (root / "evidence").mkdir()
    (root / "evidence" / "linked.json").symlink_to(outside)
    with pytest.raises(
        parent_import.PilotV211ParentImportError,
        match="escaped",
    ):
        parent_import._normalized_relative(
            "evidence/../outside.json",
            required_top="evidence",
            name="source",
        )
    with pytest.raises(
        parent_import.PilotV211ParentImportError,
        match="symlink",
    ):
        parent_import._guarded_file(
            root,
            PurePosixPath("evidence/linked.json"),
            name="source",
        )


def test_bound_source_rejects_byte_tamper(tmp_path: Path) -> None:
    root = tmp_path / "source"
    path = root / "evidence" / "aggregate.csv"
    path.parent.mkdir(parents=True)
    original = b"a,b\n1,2\n"
    path.write_bytes(original)
    binding = {
        "path": "evidence/aggregate.csv",
        "byte_size": len(original),
        "sha256": hashlib.sha256(original).hexdigest(),
    }
    assert parent_import._bound_json_or_bytes(
        root,
        binding,
        required_top="evidence",
        name="aggregate",
    ) == original
    path.write_bytes(b"a,b\n9,9\n")
    with pytest.raises(
        parent_import.PilotV211ParentImportError,
        match="identity drifted",
    ):
        parent_import._bound_json_or_bytes(
            root,
            binding,
            required_top="evidence",
            name="aggregate",
        )


def test_event_ledger_rejects_chain_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = {
        "event_index": 0,
        "event_type": "genesis",
        "payload": {"contract_hash": parent_import.V2102_CONTRACT_SHA256},
        "previous_event_sha256": "0" * 64,
    }
    event["event_sha256"] = canonical_sha256(event)
    ledger = {
        "schema_version": "finevo-pilot-run-ledger-v2",
        "contract_hash": parent_import.V2102_CONTRACT_SHA256,
        "created_at": "synthetic",
        "updated_at": "synthetic",
        "events": [event],
        "runs": {},
    }
    internal = canonical_sha256(ledger)
    ledger["ledger_sha256"] = internal
    binding = {
        "internal_sha256": internal,
        "event_count": 1,
        "event_head_sha256": event["event_sha256"],
        "run_count": 0,
    }
    monkeypatch.setattr(
        parent_import,
        "V2102_RUN_LEDGER_SHA256",
        internal,
    )
    parent_import._verify_event_ledger(
        ledger,
        binding,
        schema_version="finevo-pilot-run-ledger-v2",
        name="synthetic ledger",
    )
    tampered = json.loads(json.dumps(ledger))
    tampered["events"][0]["payload"]["contract_hash"] = "f" * 64
    with pytest.raises(
        parent_import.PilotV211ParentImportError,
        match="drifted",
    ):
        parent_import._verify_event_ledger(
            tampered,
            binding,
            schema_version="finevo-pilot-run-ledger-v2",
            name="synthetic ledger",
        )


def test_fake_evidence_checks_aggregate_csv_and_checksum_identities(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence-repo"
    package_root = root / "evidence" / "current_v2" / "pilot-v2.10.2"
    aggregate = {
        "schema_version": "finevo-pilot-v2.10.2-evidence-package-v1",
        "contract_id": parent_import.V2102_CONTRACT_ID,
        "contract_sha256": parent_import.V2102_CONTRACT_SHA256,
        "pilot_tag": parent_import.V2102_SCIENCE_TAG,
        "resolved_git_commit": parent_import.V2102_SCIENCE_COMMIT,
        "publication_status": "complete-with-no-go",
        "scientific_complete": False,
        "scientific_matrix_complete": False,
        "budget": {
            "actual_totals": {
                "cost_usd": 16.044922812500005,
                "completions": 816,
                "storage_bytes": 217010835,
            },
            "pass": True,
        },
        "denominator": {
            "expected_count": 211,
            "observed_ledger_count": 211,
            "status_counts": {"complete": 126, "failed": 85},
            "all_rows_terminal": True,
        },
    }
    aggregate_raw = _write_json(package_root / "aggregate.json", aggregate)
    csv_raw = b"run_id,status\nsynthetic,complete\n"
    (package_root / "aggregate.csv").write_bytes(csv_raw)
    package = {
        "schema_version": "finevo-pilot-v2.10.2-evidence-package-v1",
        "contract_id": parent_import.V2102_CONTRACT_ID,
        "contract_sha256": parent_import.V2102_CONTRACT_SHA256,
        "pilot_tag": parent_import.V2102_SCIENCE_TAG,
        "resolved_git_commit": parent_import.V2102_SCIENCE_COMMIT,
        "publication_status": "complete-with-no-go",
        "scientific_complete": False,
        "scientific_matrix_complete": False,
        "published_files": ["aggregate.csv", "aggregate.json"],
    }
    package_raw = _write_json(package_root / "package_manifest.json", package)
    checksum_files = [
        {
            "path": "aggregate.csv",
            "byte_size": len(csv_raw),
            "sha256": hashlib.sha256(csv_raw).hexdigest(),
        },
        {
            "path": "aggregate.json",
            "byte_size": len(aggregate_raw),
            "sha256": hashlib.sha256(aggregate_raw).hexdigest(),
        },
        {
            "path": "package_manifest.json",
            "byte_size": len(package_raw),
            "sha256": hashlib.sha256(package_raw).hexdigest(),
        },
    ]
    checksums = {
        "schema_version": "finevo-pilot-package-checksums-v1",
        "contract_sha256": parent_import.V2102_CONTRACT_SHA256,
        "files": checksum_files,
    }
    checksums_raw = _write_json(package_root / "checksums.json", checksums)
    prefix = "evidence/current_v2/pilot-v2.10.2"
    manifest = {
        "evidence_release": {
            "aggregate_json": {
                "path": f"{prefix}/aggregate.json",
                "byte_size": len(aggregate_raw),
                "sha256": hashlib.sha256(aggregate_raw).hexdigest(),
            },
            "aggregate_csv": {
                "path": f"{prefix}/aggregate.csv",
                "byte_size": len(csv_raw),
                "sha256": hashlib.sha256(csv_raw).hexdigest(),
            },
            "checksums": {
                "path": f"{prefix}/checksums.json",
                "byte_size": len(checksums_raw),
                "sha256": hashlib.sha256(checksums_raw).hexdigest(),
                "files_canonical_sha256": canonical_sha256(
                    checksum_files
                ),
            },
            "package_manifest": {
                "path": f"{prefix}/package_manifest.json",
                "byte_size": len(package_raw),
                "sha256": hashlib.sha256(package_raw).hexdigest(),
            },
        }
    }
    verified = parent_import._verify_evidence(root, manifest)
    assert verified["publication_status"] == "complete-with-no-go"
    aggregate["budget"]["actual_totals"]["completions"] = 817
    _write_json(package_root / "aggregate.json", aggregate)
    with pytest.raises(
        parent_import.PilotV211ParentImportError,
        match="file identity drifted",
    ):
        parent_import._verify_evidence(root, manifest)
