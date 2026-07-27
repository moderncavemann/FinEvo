from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_v24_parent_import import _seal
import verified_memory.pilot_v29_stage0_import as v29


ROOT = Path(__file__).resolve().parents[1]
PARENT = Path(
    "/Users/guanghaowu/Develop/financial world/worktrees/"
    "finevo-pilot-v2-8-science"
)


@pytest.fixture(scope="module")
def draft_contract():
    return load_pilot_contract(ROOT / "experiments/pilot_v2_9.yaml")


@pytest.fixture(scope="module")
def live_manifest(draft_contract):
    if not (PARENT / v29.V28_RAW_ROOT).is_dir():
        pytest.skip("immutable local V2.8 source checkout is unavailable")
    return v29.build_v29_source_manifest(
        parent_repo_root=PARENT,
        child_repo_root=ROOT,
        target_contract=draft_contract,
    )


@pytest.fixture(scope="module")
def copied_child_raw(tmp_path_factory):
    if not (PARENT / v29.V28_RAW_ROOT).is_dir():
        pytest.skip("immutable local V2.8 source checkout is unavailable")
    guard = tmp_path_factory.mktemp("v29-exact-copy")
    child_raw = guard / v29.V29_RAW_ROOT
    snapshot = v29.imported_v28_raw_root_v29(child_raw)
    source = PARENT / v29.V28_RAW_ROOT
    inventory = v29._verify_exact_v28_inventory(source)
    v29._copy_exact_snapshot(
        source_root=source,
        destination_root=snapshot,
        destination_guard_root=guard,
        inventory=inventory,
    )
    return child_raw


def test_live_manifest_binds_exact_v28_no_go_and_inventory(
    live_manifest,
) -> None:
    parent = live_manifest["v2_8_terminal_parent"]
    assert parent["release"] == {
        "science_tag": v29.V28_SCIENCE_TAG,
        "science_tag_object": v29.V28_SCIENCE_TAG_OBJECT,
        "science_commit": v29.V28_SCIENCE_COMMIT,
        "tag_kind": "annotated",
        "raw_root": "experiment_results/pilot-v2.8/raw",
        "release_attestation": parent["release"]["release_attestation"],
    }
    assert parent["raw_snapshot"] == {
        "root": "experiment_results/pilot-v2.8/raw",
        "inventory_schema_version": "finevo-raw-tree-inventory-v1",
        "inventory_canonicalization": "json-sort-keys-compact-utf8-v1",
        "file_count": 271,
        "storage_bytes": 14_766_598,
        "inventory_sha256":
        "3dfdb24e52a1c2291bfc4882c1ff7d1dffed47c6c83a0f9c1f1eae825ec68e61",
    }
    assert parent["terminal_denominator"]["status_counts"] == {
        "complete": 1,
        "failed": 1,
        "integrity-stopped": 209,
    }
    assert parent["ledgers"]["run"]["internal_sha256"] == (
        v29.V28_RUN_LEDGER_INTERNAL_SHA256
    )
    assert parent["ledgers"]["budget"]["internal_sha256"] == (
        v29.V28_BUDGET_LEDGER_INTERNAL_SHA256
    )
    evidence = live_manifest["published_v2_8_evidence"]
    assert evidence["publication_commit"] == (
        "00cc7142ae7af603f7989804a43c4d509456bad2"
    )
    assert evidence["merge_commit"] == (
        "981e2af20372c0413600f2bbd1b732f2d643593e"
    )
    assert evidence["checksums"]["entry_count"] == 16
    assert evidence["publication_status"] == "complete-with-no-go"
    assert evidence["scientific_complete"] is False


def test_qref_is_verified_failed_audit_reference_not_import(
    live_manifest,
) -> None:
    qref = v29.q_ref_audit_reference_v29(live_manifest)
    assert qref["imported"] is False
    assert qref["source_result_reuse"] == "forbidden"
    assert qref["failed_prerequisite"] == {
        **qref["failed_prerequisite"],
        "status": "failed",
        "error_type": "PilotOrchestrationError",
        "error_message": (
            "V2.8 fresh q-ref differs from its audit reference: "
            "['run_summary_exact']"
        ),
        "error_message_sha256":
        "713b1d429fd939e74b2007d78d3c3789ce10376ee0ba970e1cfe1359503c246a",
    }
    runner = qref["verified_runner"]
    assert runner["identity_grid"] == {
        "agents": 4,
        "periods": 12,
        "action_rows": 48,
        "api_usage_rows": 48,
        "utility_ledger_rows": 48,
        "shock_rows": 12,
        "summary_rows": 1,
    }
    assert runner["provider_accounting"] == {
        "provider_kind": "scripted-diagnostic",
        "model": "diagnostic/scripted-v1",
        "scripted_diagnostic_calls": 48,
        "hosted_provider_calls": 0,
        "hosted_cost_usd": 0.0,
        "total_tokens": 15905,
    }
    assert set(runner["streams"]) == {
        "summary",
        "actions",
        "api_usage",
        "utility_ledger",
        "shock_events",
    }
    ancestral = qref["ancestral_v2_6_scalar_reference"]
    assert ancestral["source_contract_id"] == "finevo-pilot-v2.6"
    assert ancestral["q_ref"] == 63.50397933257746
    assert qref["fresh_v2_9_policy"]["source_result_reuse"] == "forbidden"


def test_only_parent_and_fourteen_stage0_cells_are_imported(
    live_manifest,
    draft_contract,
    tmp_path: Path,
) -> None:
    rows = live_manifest["imported_complete_cells"]
    assert len(rows) == 15
    assert [row["stage_id"] for row in rows].count("parent-import") == 1
    assert [row["stage_id"] for row in rows].count(
        "stage0-calibration"
    ) == 14
    assert not any(row["stage_id"] == "q-ref-resolution" for row in rows)
    stage0 = next(
        row for row in rows if row["stage_id"] == "stage0-calibration"
    )
    binding = v29.source_binding_for_target_v29(
        live_manifest,
        stage0["target_spec"],
    )
    assert binding["physical_source_contract_id"] == "finevo-pilot-v2.6"
    assert binding["source_authority_contract_id"] == "finevo-pilot-v2.8"
    assert binding["source_artifacts"]["run_root"].startswith(
        "experiment_results/pilot-v2.8/raw/"
        "parent-import/v2_7_raw_snapshot/"
        "parent-import/v2_6_raw_snapshot/"
    )
    raw_root = tmp_path / v29.V29_RAW_ROOT
    expected = v29.snapshot_path_for_v28_source_artifact_v29(
        raw_root,
        binding["source_artifacts"]["run_root"],
    )
    assert (
        v29.imported_v26_run_dir_v29(
            raw_root,
            stage0["target_spec"],
            live_manifest,
        )
        == expected
    )
    qref = draft_contract.expand(stage="q-ref-resolution")[0]
    with pytest.raises(
        v29.PilotV29Stage0ImportError,
        match="no unique imported",
    ):
        v29.source_binding_for_target_v29(live_manifest, qref)


def test_live_validator_and_budget_debit(
    live_manifest,
    draft_contract,
) -> None:
    assert v29.validate_v29_source_manifest(
        live_manifest,
        parent_repo_root=PARENT,
        child_repo_root=ROOT,
        target_contract=draft_contract,
    ) == live_manifest
    debit = v29.parent_budget_debit_for_v29(draft_contract)
    assert debit is not None
    assert debit.record_sha256 == (
        "0944138d9b47f7cf720681eb0ea8feda0b612a912992d78434c6bbda0d560fd0"
    )
    assert debit.parent_contract_sha256 == (
        "948eac04516dd2c292d68beb732f97532b13e667a180e8c2db16fbb927f92f19"
    )


def test_manifest_tamper_cannot_reclassify_qref(live_manifest) -> None:
    tampered = json.loads(json.dumps(live_manifest))
    tampered["q_ref_audit_reference"]["imported"] = True
    tampered["import_policy"]["q_ref_imported"] = True
    tampered = _seal(tampered)
    with pytest.raises(
        v29.PilotV29Stage0ImportError,
        match="authority boundary drifted",
    ):
        v29._validate_source_manifest_structure(tampered)


def test_manifest_tamper_cannot_forge_raw_inventory(live_manifest) -> None:
    tampered = json.loads(json.dumps(live_manifest))
    tampered["v2_8_terminal_parent"]["raw_snapshot"][
        "inventory_sha256"
    ] = "0" * 64
    tampered = _seal(tampered)
    with pytest.raises(
        v29.PilotV29Stage0ImportError,
        match="authority boundary drifted",
    ):
        v29._validate_source_manifest_structure(tampered)


def test_manifest_write_load_and_symlink_fail_closed(
    live_manifest,
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.json"
    v29.write_v29_source_manifest_draft(path, live_manifest)
    raw = path.read_bytes()
    loaded = v29.load_v29_source_manifest(
        path,
        expected_file_sha256=v29._sha256(raw),
        expected_content_sha256=live_manifest["integrity"][
            "content_sha256"
        ],
    )
    assert loaded == live_manifest
    link = tmp_path / "source-link.json"
    link.symlink_to(path)
    with pytest.raises(
        v29.PilotV29Stage0ImportError,
        match="symlink",
    ):
        v29.load_v29_source_manifest(link)


@pytest.mark.parametrize(
    "path",
    (
        "../experiment_results/pilot-v2.8/raw/run_ledger.json",
        "/tmp/experiment_results/pilot-v2.8/raw/run_ledger.json",
        "experiment_results/pilot-v2.7/raw/run_ledger.json",
        "experiment_results/pilot-v2.8/raw/../raw/run_ledger.json",
    ),
)
def test_snapshot_mapper_rejects_escape(path: str, tmp_path: Path) -> None:
    with pytest.raises(v29.PilotV29Stage0ImportError):
        v29.snapshot_path_for_v28_source_artifact_v29(tmp_path, path)


def test_inventory_rejects_symlink(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    (root / "linked.json").symlink_to(target)
    with pytest.raises(
        v29.PilotV29Stage0ImportError,
        match="symlink",
    ):
        v29._inventory(root, declared_root=v29.V28_RAW_ROOT)


def test_copy_rejects_destination_symlink_injection(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "first").write_bytes(b"authority")
    guard = tmp_path / "guard"
    guard.mkdir()
    destination = guard / "snapshot"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"do-not-overwrite")
    (destination / "first").symlink_to(outside)
    inventory = [
        {
            "path": "first",
            "byte_size": len(b"authority"),
            "sha256": v29._sha256(b"authority"),
        }
    ]
    with pytest.raises(
        v29.PilotV29Stage0ImportError,
        match="symlink|immutable",
    ):
        v29._copy_exact_snapshot(
            source_root=source,
            destination_root=destination,
            destination_guard_root=guard,
            inventory=inventory,
        )
    assert outside.read_bytes() == b"do-not-overwrite"


def test_live_and_copied_v28_p95_are_verified(
    copied_child_raw,
) -> None:
    for profile_id in v29.V29_ALLOWED_P95_PROFILES:
        live = v29.v2_8_p95_source_binding_v29(PARENT, profile_id)
        copied = v29.verify_v29_imported_v28_observed_p95(
            copied_child_raw,
            profile_id,
        )
        assert live["profile_id"] == copied["profile_id"] == profile_id
        assert live["source_git_commit"] == copied["source_git_commit"]
        assert live["reservations"] == copied["reservations"]


def test_parent_receipt_records_zero_import_dispatch(
    copied_child_raw,
    draft_contract,
    live_manifest,
) -> None:
    receipt = v29._build_v29_parent_import_receipt(
        child_root=ROOT,
        child_raw=copied_child_raw,
        contract=draft_contract,
        child_git_commit="c" * 40,
        manifest=live_manifest,
    )
    assert receipt["scripted_diagnostic_calls_during_import"] == 0
    assert receipt["hosted_provider_calls_during_import"] == 0
    assert receipt["provider_calls_during_import"] == 0
    assert receipt["provider_construction_during_import"] is False
    assert receipt["q_ref"]["scripted_diagnostic_calls"] == 48
    assert receipt["q_ref"]["hosted_provider_calls"] == 0
    assert receipt["q_ref"]["imported"] is False


def test_copied_p95_tamper_breaks_inventory(
    copied_child_raw,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "raw"
    shutil.copytree(copied_child_raw, copied)
    projection = v29.v29_observed_p95_projection_path(
        copied,
        "gpt52_main",
    )
    projection.write_bytes(projection.read_bytes() + b" ")
    with pytest.raises(
        v29.PilotV29Stage0ImportError,
        match="inventory drifted",
    ):
        v29.verify_v29_imported_v28_observed_p95(
            copied,
            "gpt52_main",
        )
