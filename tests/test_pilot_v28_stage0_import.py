from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_v24_parent_import import _seal
import verified_memory.pilot_v28_stage0_import as v28


ROOT = Path(__file__).resolve().parents[1]
PARENT = Path(
    "/Users/guanghaowu/Develop/financial world/worktrees/"
    "finevo-pilot-v2-7-source"
)
COMMIT = "c" * 40


@pytest.fixture(scope="module")
def draft_contract():
    return load_pilot_contract(ROOT / "experiments/pilot_v2_8.yaml")


@pytest.fixture(scope="module")
def live_manifest(draft_contract):
    if not (PARENT / "experiment_results/pilot-v2.7/raw").is_dir():
        pytest.skip("the immutable local V2.7 source checkout is unavailable")
    return v28.build_v28_source_manifest(
        parent_repo_root=PARENT,
        child_repo_root=ROOT,
        target_contract=draft_contract,
    )


def test_live_v27_source_manifest_binds_exact_lineage_and_qref_core(
    live_manifest,
) -> None:
    assert live_manifest["schema_version"] == (
        v28.V28_SOURCE_MANIFEST_SCHEMA_VERSION
    )
    assert len(live_manifest["imported_complete_cells"]) == 15
    assert {
        row["stage_id"] for row in live_manifest["imported_complete_cells"]
    } == {"parent-import", "stage0-calibration"}
    assert not any(
        row["stage_id"] == "q-ref-resolution"
        for row in live_manifest["imported_complete_cells"]
    )
    parent = live_manifest["v2_7_terminal_parent"]
    assert parent["release"] == {
        "raw_root": "experiment_results/pilot-v2.7/raw",
        "science_commit": v28.V27_SCIENCE_COMMIT,
        "science_tag": v28.V27_SCIENCE_TAG,
        "science_tag_object": v28.V27_SCIENCE_TAG_OBJECT,
        "tag_kind": "annotated",
    }
    assert parent["raw_snapshot"] == {
        "root": "experiment_results/pilot-v2.7/raw",
        "inventory_schema_version": "finevo-raw-tree-inventory-v1",
        "inventory_canonicalization": "json-sort-keys-compact-utf8-v1",
        "file_count": 242,
        "storage_bytes": 13_500_493,
        "inventory_sha256": v28.V27_RAW_INVENTORY_SHA256,
    }
    assert parent["terminal_denominator"]["status_counts"] == {
        "complete": 1,
        "integrity-stopped": 210,
    }
    assert parent["ledgers"]["run"]["internal_sha256"] == (
        v28.V27_RUN_LEDGER_INTERNAL_SHA256
    )
    assert parent["ledgers"]["budget"]["internal_sha256"] == (
        v28.V27_BUDGET_LEDGER_INTERNAL_SHA256
    )
    evidence = live_manifest["published_v2_7_evidence"]
    assert evidence["publication_commit"] == (
        v28.V27_EVIDENCE_PUBLICATION_COMMIT
    )
    assert evidence["merge_commit"] == v28.V27_EVIDENCE_MERGE_COMMIT
    assert evidence["checksums"]["entry_count"] == 14

    qref = live_manifest["q_ref_audit_equivalence_reference"]
    assert qref["imported"] is False
    assert qref["source_result_reuse"] == "forbidden"
    assert qref["source_runner_short_run_id"] == (
        "q-ref-resolution-s2010922376"
    )
    assert qref["scripted_diagnostic_calls"] == 48
    assert qref["hosted_provider_calls"] == 0
    assert qref["hosted_cost_usd"] == 0.0
    assert qref["source_core"]["identity_grid"] == {
        "agents": 4,
        "periods": 12,
        "action_rows": 48,
        "api_usage_rows": 48,
        "utility_ledger_rows": 48,
        "shock_rows": 12,
    }
    assert qref["source_core"]["streams"]["api_usage"]["row_count"] == 48
    assert qref["source_core"]["semantic_streams"] == {
        "semantic_proposals": 0,
        "semantic_rules": 0,
        "semantic_rule_events": 0,
    }
    stage0 = [
        row
        for row in live_manifest["imported_complete_cells"]
        if row["stage_id"] == "stage0-calibration"
    ]
    assert len(stage0) == 14
    assert {
        (
            row["source_spec"]["utility_profile_id"],
            row["source_spec"]["environment_seed"],
        )
        for row in stage0
    } == {
        (profile_id, seed)
        for profile_id in v28._STAGE0_PROFILES
        for seed in v28._STAGE0_SEEDS
    }
    assert all(
        row["physical_source_contract_id"] == v28.V26_CONTRACT_ID
        and row["source_authority_contract_id"] == v28.V27_CONTRACT_ID
        and row["source_artifacts"]["run_root"].endswith(
            f"/{row['source_run_id']}"
        )
        and row["source_artifacts"]["config"]["path"].endswith(
            f"/{row['source_run_id']}/config.json"
        )
        and row["source_artifacts"]["manifest"]["path"].endswith(
            f"/{row['source_run_id']}/manifest.json"
        )
        and row["source_artifacts"]["actor_journal"]["path"].endswith(
            f"/{row['source_run_id']}--actor.json"
        )
        for row in stage0
    )
    assert live_manifest["import_policy"][
        "provider_construction_during_import"
    ] is False


def test_source_binding_maps_only_parent_and_nested_stage0(
    live_manifest,
    tmp_path: Path,
) -> None:
    stage0 = next(
        row
        for row in live_manifest["imported_complete_cells"]
        if row["stage_id"] == "stage0-calibration"
    )
    binding = v28.source_binding_for_target_v28(
        live_manifest,
        stage0["target_spec"],
    )
    assert binding["source_run_id"].startswith(
        "finevo-pilot-v2.6--stage0-calibration--"
    )
    assert binding["target_run_id"].startswith(
        "finevo-pilot-v2.8--stage0-calibration--"
    )
    raw_root = tmp_path / "experiment_results/pilot-v2.8/raw"
    expected = (
        raw_root
        / "parent-import/v2_7_raw_snapshot"
        / "parent-import/v2_6_raw_snapshot"
        / "stage0-calibration/runs"
        / binding["source_run_id"]
    )
    assert (
        v28.imported_v26_run_dir_v28(
            raw_root,
            stage0["target_spec"],
            live_manifest,
        )
        == expected
    )
    qref_run_id = live_manifest["q_ref_audit_equivalence_reference"][
        "source_run_id"
    ].replace("finevo-pilot-v2.6", "finevo-pilot-v2.8", 1)
    with pytest.raises(
        v28.PilotV28Stage0ImportError,
        match="no unique imported",
    ):
        v28.source_binding_for_target_v28(live_manifest, qref_run_id)


def test_manifest_rejects_qref_reclassification(live_manifest) -> None:
    tampered = json.loads(json.dumps(live_manifest))
    tampered["q_ref_audit_equivalence_reference"]["imported"] = True
    tampered = _seal(tampered)
    with pytest.raises(
        v28.PilotV28Stage0ImportError,
        match="parent authority drifted",
    ):
        v28._validate_source_manifest_structure(tampered)


def test_manifest_rejects_qref_provider_accounting_tamper(
    live_manifest,
) -> None:
    tampered = json.loads(json.dumps(live_manifest))
    qref = tampered["q_ref_audit_equivalence_reference"]
    qref["hosted_provider_calls"] = 1
    qref["source_core"]["identity_grid"]["api_usage_rows"] = 47
    tampered = _seal(tampered)
    with pytest.raises(
        v28.PilotV28Stage0ImportError,
        match="parent authority drifted",
    ):
        v28._validate_source_manifest_structure(tampered)


def test_live_source_validator_rejects_resealed_binding_tamper(
    live_manifest,
    draft_contract,
) -> None:
    tampered = json.loads(json.dumps(live_manifest))
    stage0 = next(
        row
        for row in tampered["imported_complete_cells"]
        if row["stage_id"] == "stage0-calibration"
    )
    stage0["source_artifacts"]["config"]["file_sha256"] = "0" * 64
    tampered = _seal(tampered)
    with pytest.raises(
        v28.PilotV28Stage0ImportError,
        match="differs from verified V2.7 authority",
    ):
        v28.validate_v28_source_manifest(
            tampered,
            parent_repo_root=PARENT,
            child_repo_root=ROOT,
            target_contract=draft_contract,
        )


def test_stage0_config_run_id_must_equal_source_cell(
    tmp_path: Path,
) -> None:
    run_id = "finevo-pilot-v2.6--stage0-calibration--source"
    relative = Path("nested/stage0-calibration/runs") / run_id
    path = tmp_path / relative / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"run_id": "short-runner-id"}),
        encoding="utf-8",
    )
    with pytest.raises(
        v28.PilotV28Stage0ImportError,
        match="config.run_id differs from source cell ID",
    ):
        v28._verify_stage0_source_config(
            tmp_path,
            run_relative=v28.PurePosixPath(relative.as_posix()),
            source_run_id=run_id,
        )


def test_exact_snapshot_copy_is_idempotent_and_rejects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    payload = b"immutable-v2.7"
    (source / "one.bin").write_bytes(payload)
    rows, summary = v28._inventory(
        source,
        declared_root=v28.V27_RAW_ROOT,
    )
    monkeypatch.setattr(v28, "V27_RAW_FILE_COUNT", 1)
    monkeypatch.setattr(v28, "V27_RAW_STORAGE_BYTES", len(payload))
    monkeypatch.setattr(
        v28,
        "V27_RAW_INVENTORY_SHA256",
        summary["inventory_sha256"],
    )
    v28._copy_exact_snapshot(
        source_root=source,
        destination_root=destination,
        destination_guard_root=tmp_path,
        inventory=rows,
    )
    v28._copy_exact_snapshot(
        source_root=source,
        destination_root=destination,
        destination_guard_root=tmp_path,
        inventory=rows,
    )
    assert (destination / "one.bin").read_bytes() == payload
    (destination / "one.bin").write_bytes(b"tampered")
    with pytest.raises(
        v28.PilotV28Stage0ImportError,
        match="immutable parent-import artifact differs on resume",
    ):
        v28._copy_exact_snapshot(
            source_root=source,
            destination_root=destination,
            destination_guard_root=tmp_path,
            inventory=rows,
        )


def test_parent_persist_resume_reseals_both_p95_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    draft_contract,
    live_manifest,
) -> None:
    child = tmp_path / "child"
    shutil.copytree(
        ROOT / "experiments",
        child / "experiments",
    )
    (child / "experiments/pilot_v2_8_source_manifest.json").write_text(
        json.dumps(
            live_manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    raw_root = child / "experiment_results/pilot-v2.8/raw"
    raw_root.mkdir(parents=True)

    monkeypatch.setattr(
        v28,
        "_validate_target_contract",
        lambda contract, require_frozen: None,
    )
    monkeypatch.setattr(
        v28,
        "validate_v28_source_manifest",
        lambda value, **kwargs: dict(value),
    )
    monkeypatch.setattr(
        v28,
        "_git",
        lambda repo_root, *arguments: COMMIT,
    )

    first = v28.persist_v28_parent_import(
        contract=draft_contract,
        repo_root=child,
        raw_root=raw_root,
        parent_repo_root=PARENT,
        child_git_tag=v28.V28_SCIENCE_TAG,
        child_git_commit=COMMIT,
        source_manifest=live_manifest,
    )
    second = v28.persist_v28_parent_import(
        contract=draft_contract,
        repo_root=child,
        raw_root=raw_root,
        parent_repo_root=PARENT,
        child_git_tag=v28.V28_SCIENCE_TAG,
        child_git_commit=COMMIT,
        source_manifest=live_manifest,
    )
    assert first == second
    assert first["imported_cell_count"] == 15
    assert first["q_ref_imported"] is False
    assert set(first["resealed_p95_profiles"]) == set(
        v28.V28_ALLOWED_P95_PROFILES
    )
    for profile_id in v28.V28_ALLOWED_P95_PROFILES:
        receipt_path = v28.v28_observed_p95_receipt_path(
            raw_root,
            profile_id,
        )
        projection_path = v28.v28_observed_p95_projection_path(
            raw_root,
            profile_id,
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        projection = json.loads(projection_path.read_text(encoding="utf-8"))
        assert receipt["schema_version"] == (
            v28.V28_RESEALED_P95_AUTHORITY_SCHEMA_VERSION
        )
        assert receipt["git"] == {
            "tag": v28.V28_SCIENCE_TAG,
            "commit": COMMIT,
        }
        assert projection["bindings"]["source_kind"] == (
            v28.V28_RESEALED_P95_SOURCE_KIND
        )
        verified = v28.verify_v28_resealed_observed_p95_projection(
            projection_path,
            receipt_or_path=receipt_path,
            repo_root=child,
            expected_git_commit=COMMIT,
        )
        assert verified == projection

    copied_lock = (
        raw_root
        / "parent-import/v2_7_raw_snapshot/.real-stage-execution.lock"
    )
    copied_lock.write_bytes(b"tampered")
    with pytest.raises(
        v28.PilotV28Stage0ImportError,
        match="immutable parent-import artifact differs on resume",
    ):
        v28.persist_v28_parent_import(
            contract=draft_contract,
            repo_root=child,
            raw_root=raw_root,
            parent_repo_root=PARENT,
            child_git_tag=v28.V28_SCIENCE_TAG,
            child_git_commit=COMMIT,
            source_manifest=live_manifest,
        )


def test_parent_budget_debit_is_exact(draft_contract) -> None:
    debit = v28.parent_budget_debit_for_v28(draft_contract)
    assert debit is not None
    assert debit.cost_usd == 3.212770875
    assert debit.hosted_completions == 184
    assert debit.storage_bytes == 32_158_175
