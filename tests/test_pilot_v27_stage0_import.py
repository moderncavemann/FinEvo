from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_v24_parent_import import _seal
import verified_memory.pilot_v27_stage0_import as v27


ROOT = Path(__file__).resolve().parents[1]
PARENT = Path(
    "/Users/guanghaowu/Develop/financial world/worktrees/"
    "finevo-pilot-v2-6-science"
)
COMMIT = "c" * 40


def _minimal_manifest() -> dict:
    row = {
        "stage_id": "stage0-calibration",
        "source_run_id": "finevo-pilot-v2.6--source",
        "target_run_id": "finevo-pilot-v2.7--target",
        "source_spec": {
            "run_id": "finevo-pilot-v2.6--source",
            "contract_id": v27.V26_CONTRACT_ID,
            "stage_id": "stage0-calibration",
        },
        "target_spec": {
            "run_id": "finevo-pilot-v2.7--target",
            "contract_id": v27.V27_CONTRACT_ID,
            "stage_id": "stage0-calibration",
        },
        "source_artifacts": {
            "run_root": (
                "experiment_results/pilot-v2.6/raw/"
                "stage0-calibration/runs/source"
            ),
            "manifest": {
                "path": (
                    "experiment_results/pilot-v2.6/raw/"
                    "stage0-calibration/runs/source/manifest.json"
                ),
                "file_sha256": "1" * 64,
                "byte_size": 1,
            },
        },
    }
    return _seal(
        {
            "schema_version": v27.V27_SOURCE_MANIFEST_SCHEMA_VERSION,
            "v2_6_terminal_parent": {"sentinel": True},
            "published_v2_6_evidence": {"sentinel": True},
            "v2_6_p95_sources_for_child_reseal": {"sentinel": True},
            "imported_complete_cells": [row],
            "cumulative_budget_debit": {"sentinel": True},
            "import_policy": {"sentinel": True},
            "observation_boundary": {"sentinel": True},
        }
    )


def test_real_v26_source_manifest_build_is_complete_when_parent_exists() -> None:
    if not (PARENT / "experiment_results/pilot-v2.6/raw").is_dir():
        pytest.skip("the immutable local V2.6 source checkout is unavailable")
    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_7.yaml")
    manifest = v27.build_v27_source_manifest(
        parent_repo_root=PARENT,
        child_repo_root=ROOT,
        target_contract=contract,
    )

    assert manifest["schema_version"] == (
        v27.V27_SOURCE_MANIFEST_SCHEMA_VERSION
    )
    assert len(manifest["imported_complete_cells"]) == 16
    assert {
        row["stage_id"] for row in manifest["imported_complete_cells"]
    } == {"parent-import", "q-ref-resolution", "stage0-calibration"}
    assert manifest["v2_6_terminal_parent"]["raw_snapshot"] == {
        "root": "experiment_results/pilot-v2.6/raw",
        "inventory_schema_version": "finevo-raw-tree-inventory-v1",
        "inventory_canonicalization":
        "json-sort-keys-compact-utf8-v1",
        "file_count": 228,
        "storage_bytes": 12_877_797,
        "inventory_sha256": v27.V26_RAW_INVENTORY_SHA256,
    }
    assert manifest["v2_6_terminal_parent"]["terminal_denominator"][
        "status_counts"
    ] == {"complete": 16, "integrity-stopped": 195}
    assert (
        manifest["v2_6_terminal_parent"]["stage0_failure"][
            "a_d_treatment_effect_outcomes_generated"
        ]
        is False
    )
    assert sorted(manifest["v2_6_p95_sources_for_child_reseal"]) == [
        "gpt52_main",
        "llama33_local_controlled",
    ]
    assert manifest["import_policy"]["provider_construction_during_import"] is False
    assert manifest["import_policy"]["scientific_evidence"] is False

    verified = v27.validate_v27_source_manifest(
        manifest,
        parent_repo_root=PARENT,
        child_repo_root=ROOT,
        target_contract=contract,
    )
    assert verified == manifest

    tampered = json.loads(json.dumps(manifest))
    tampered["import_policy"]["scientific_evidence"] = True
    tampered = _seal(tampered)
    with pytest.raises(
        v27.PilotV27Stage0ImportError,
        match="differs from verified V2.6 authority",
    ):
        v27.validate_v27_source_manifest(
            tampered,
            parent_repo_root=PARENT,
            child_repo_root=ROOT,
            target_contract=contract,
        )


def test_source_binding_and_snapshot_mapping_preserve_v26_identity(
    tmp_path: Path,
) -> None:
    manifest = _minimal_manifest()
    target = manifest["imported_complete_cells"][0]["target_spec"]
    binding = v27.source_binding_for_target(manifest, target)
    assert binding["source_run_id"] == "finevo-pilot-v2.6--source"
    assert binding["target_run_id"] == "finevo-pilot-v2.7--target"

    raw_root = tmp_path / "experiment_results/pilot-v2.7/raw"
    expected = (
        raw_root
        / "parent-import/v2_6_raw_snapshot"
        / "stage0-calibration/runs/source"
    )
    assert v27.imported_v26_run_dir(raw_root, target, manifest) == expected
    assert (
        v27.snapshot_path_for_source_artifact(
            raw_root,
            (
                "experiment_results/pilot-v2.6/raw/"
                "stage0-calibration/runs/source/manifest.json"
            ),
        )
        == expected / "manifest.json"
    )
    with pytest.raises(
        v27.PilotV27Stage0ImportError,
        match="outside the V2.6 raw namespace",
    ):
        v27.snapshot_path_for_source_artifact(
            raw_root, "experiment_results/pilot-v2.5/raw/file.json"
        )


def test_exact_snapshot_copy_is_idempotent_and_rejects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    payload = b"immutable-v2.6"
    (source / "one.bin").write_bytes(payload)
    rows, summary = v27._inventory(source)
    monkeypatch.setattr(v27, "V26_RAW_FILE_COUNT", 1)
    monkeypatch.setattr(v27, "V26_RAW_STORAGE_BYTES", len(payload))
    monkeypatch.setattr(
        v27, "V26_RAW_INVENTORY_SHA256", summary["inventory_sha256"]
    )

    v27._copy_exact_snapshot(
        source_root=source,
        destination_root=destination,
        inventory=rows,
    )
    v27._copy_exact_snapshot(
        source_root=source,
        destination_root=destination,
        inventory=rows,
    )
    assert (destination / "one.bin").read_bytes() == payload

    (destination / "one.bin").write_bytes(b"tampered")
    with pytest.raises(
        v27.PilotV27Stage0ImportError,
        match="immutable parent-import artifact differs on resume",
    ):
        v27._copy_exact_snapshot(
            source_root=source,
            destination_root=destination,
            inventory=rows,
        )


def test_parent_import_receipt_verifier_rebuilds_and_rejects_resealed_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    manifest = _minimal_manifest()
    contract = SimpleNamespace(
        contract_id=v27.V27_CONTRACT_ID,
        canonical_hash="a" * 64,
        status="frozen",
        implementation={"required_git_tag": v27.V27_SCIENCE_TAG},
        stage0_evaluator_retry_amendment={
            "schema_version":
            "finevo-pilot-stage0-evaluator-retry-amendment-v1"
        },
    )
    monkeypatch.setattr(v27, "_validate_target_contract", lambda *a, **k: None)
    monkeypatch.setattr(
        v27,
        "_child_contract_binding",
        lambda *a, **k: {
            "path": v27.V27_EXPANDED_CONTRACT_PATH.as_posix(),
            "file_sha256": "b" * 64,
            "contract_id": v27.V27_CONTRACT_ID,
            "contract_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        v27, "load_v27_source_manifest", lambda **kwargs: manifest
    )
    monkeypatch.setattr(
        v27, "_verify_exact_v26_inventory", lambda _root: []
    )
    receipt = v27._build_v27_parent_import_receipt(
        child_root=root,
        contract=contract,
        child_git_commit=COMMIT,
        manifest=manifest,
    )
    assert (
        v27.verify_v27_parent_import_receipt(
            receipt,
            repo_root=root,
            contract=contract,
            expected_git_commit=COMMIT,
        )
        == receipt
    )
    tampered = json.loads(json.dumps(receipt))
    tampered["provider_calls"] = 1
    tampered = _seal(tampered)
    with pytest.raises(
        v27.PilotV27Stage0ImportError,
        match="differs from sealed sources",
    ):
        v27.verify_v27_parent_import_receipt(
            tampered,
            repo_root=root,
            contract=contract,
            expected_git_commit=COMMIT,
        )


def test_v27_cumulative_parent_debit_is_exact() -> None:
    contract = load_pilot_contract(ROOT / "experiments/pilot_v2_7.yaml")
    debit = v27.parent_budget_debit_for_v27(contract)
    assert debit is not None
    assert debit.cost_usd == pytest.approx(3.212770875)
    assert debit.hosted_completions == 184
    assert debit.storage_bytes == 19_181_432
    assert debit.record_sha256 == (
        "6d5a9461485122a3770e9855229dfc120728ab6da1f4f9074c5150515a62285e"
    )


def test_source_manifest_draft_write_is_exact_and_idempotent(
    tmp_path: Path,
) -> None:
    value = _minimal_manifest()
    path = tmp_path / "pilot_v2_7_source_manifest.json"
    assert v27.write_v27_source_manifest_draft(path, value) == path
    v27.write_v27_source_manifest_draft(path, value)
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        v27._source_manifest_binding(value)["file_sha256"]
    )
    with pytest.raises(
        v27.PilotV27Stage0ImportError,
        match="immutable parent-import artifact differs on resume",
    ):
        v27.write_v27_source_manifest_draft(path, _seal({"schema_version": "x"}))
