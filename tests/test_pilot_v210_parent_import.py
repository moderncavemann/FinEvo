from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from verified_memory.pilot_contract import load_pilot_contract
from verified_memory import pilot_v210_parent_import as parent_import


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_10.yaml"
SOURCE_MANIFEST_PATH = ROOT / "experiments" / "pilot_v2_10_source_manifest.json"
V29_SCIENCE_ROOT = ROOT.parent / "finevo-pilot-v2-9-science"


def _contract():
    return load_pilot_contract(CONTRACT_PATH)


def _manifest():
    return parent_import.load_v210_source_manifest(
        SOURCE_MANIFEST_PATH,
        expected_file_sha256=(
            "8540bde06f364aa9ccf2a6937b78dec1f0d3b2c66b9e4943f9a3d2e20e4b19a7"
        ),
        expected_content_sha256=(
            "fc781697a9260fa63d0535eafa24b87a8386a76dca55f3ce95ba59e12ceb4224"
        ),
    )


def test_v29_published_evidence_hashes_and_failure_boundary_are_exact() -> None:
    verified = parent_import._verify_v29_evidence(ROOT)

    assert verified == {
        "publication_commit": ("51525614e138e5b7ac498d15b409048d5110b753"),
        "merge_commit": "08fcbc0dd9319fcc86c3f4e812c3db504a0c5a17",
        "root": "evidence/current_v2/pilot-v2.9",
        "checksums_file_sha256": (
            "b0de7185c710b69736ddfe1d331b7f6308165a9f03bb0c616f14ec1fd7a515db"
        ),
        "package_manifest_file_sha256": (
            "6d006ba59c5af6a1e0dd3931466b90d4599edc0ded47e2de3ea4f8ecd6c4831a"
        ),
        "aggregate_file_sha256": (
            "8cddead63df3b9e86703ef54056da87b26fdd3ba63841df29f1a4e5188aa1936"
        ),
        "failure_ledger_file_sha256": (
            "2c4ef904a4aff0fff9185d63c62736bcbf130853b7371e868f8a3328b6df9bdb"
        ),
        "reviewer_report_file_sha256": (
            "19991e751ea71c73cd0927ace8ae271e69ea69ab95098c2c030a6266ceb817c0"
        ),
        "terminal_status": "complete-with-no-go",
        "root_cause_code": "imported-p95-runner-binding-shape-mismatch",
        "v2_9_hosted_completions": 0,
        "v2_9_hosted_stage_cost_usd": 0.0,
        "actor_treatment_effect_outcomes_generated": False,
        "offline_candidate_admission_cells_generated": 10,
        "scientific_claim_gates_supported": False,
    }


def test_source_manifest_freezes_only_sixteen_prerequisites() -> None:
    manifest = _manifest()
    rows = manifest["imported_complete_cells"]

    assert len(rows) == 16
    assert {
        stage: len([row for row in rows if row["stage_id"] == stage])
        for stage in (
            "parent-import",
            "q-ref-resolution",
            "stage0-calibration",
        )
    } == {
        "parent-import": 1,
        "q-ref-resolution": 1,
        "stage0-calibration": 14,
    }
    assert manifest["import_policy"]["offline_candidate_admission_cells_imported"] == 0
    assert manifest["import_policy"]["provider_calls_during_import"] == 0
    assert manifest["import_policy"]["provider_construction_during_import"] is False
    assert manifest["import_policy"]["stage0_selected_profile_id"] == "nu-0.5"


def test_target_spec_to_snapshot_artifact_routes_are_exact() -> None:
    contract = _contract()
    manifest = _manifest()
    child_raw = ROOT / "experiment_results" / "pilot-v2.10" / "raw"

    qref = contract.expand(stage="q-ref-resolution")[0]
    qref_binding = parent_import.source_binding_for_target_v210(
        manifest,
        qref,
    )
    assert qref_binding["stage_id"] == "q-ref-resolution"
    assert qref_binding["source_run_id"].startswith(
        "finevo-pilot-v2.9--q-ref-resolution--"
    )
    qref_path = parent_import.imported_prerequisite_path_v210(
        child_raw,
        manifest,
        qref,
        "q_ref_resolution",
    )
    assert qref_path == (
        child_raw
        / "parent-import"
        / "v2_9_raw_snapshot"
        / "q-ref-resolution"
        / "q_ref_resolution.json"
    )

    stage0 = contract.expand(stage="stage0-calibration")[0]
    stage0_binding = parent_import.source_binding_for_target_v210(
        manifest,
        stage0,
    )
    assert stage0_binding["stage_id"] == "stage0-calibration"
    assert set(stage0_binding["source_artifacts"]) == {
        "summary",
        "imported_run_envelope",
        "stage0_selection",
        "stage_receipt",
        "run_root",
    }
    assert (
        parent_import.v210_imported_v29_run_dir(
            child_raw,
            manifest,
            stage0,
        ).name
        == stage0_binding["source_run_id"]
    )


@pytest.mark.parametrize(
    "profile_id",
    ("gpt52_main", "llama33_local_controlled"),
)
def test_nested_v29_p95_binding_is_normalized_to_exact_runner_fields(
    profile_id: str,
) -> None:
    manifest = _manifest()
    source = manifest["v2_9_p95_sources_for_current_release_reseal"][profile_id]
    nested = source["v2_8_observed_p95_origin"]

    flat = parent_import.normalize_v29_observed_p95_binding(
        nested,
        profile_id=profile_id,
    )

    assert set(flat) == {
        "receipt_path",
        "receipt_file_sha256",
        "receipt_content_sha256",
        "git_commit",
        "reservations",
    }
    assert flat == source["normalized_v2_9_binding"]
    assert flat["receipt_path"] == nested["authority"]["path"]
    assert flat["git_commit"] == nested["source_git_commit"]

    malformed = deepcopy(nested)
    malformed["receipt_path"] = malformed["authority"]["path"]
    with pytest.raises(
        parent_import.PilotV210ParentImportError,
        match="shape drifted",
    ):
        parent_import.normalize_v29_observed_p95_binding(
            malformed,
            profile_id=profile_id,
        )


@pytest.mark.parametrize(
    "profile_id",
    ("gpt52_main", "llama33_local_controlled"),
)
def test_current_release_p95_pair_rebuilds_and_tampering_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    profile_id: str,
) -> None:
    contract = _contract()
    manifest = _manifest()
    source = manifest["v2_9_p95_sources_for_current_release_reseal"][profile_id]
    raw_root = ROOT / "experiment_results" / "pilot-v2.10" / "raw"
    commit = "a" * 40

    monkeypatch.setattr(
        parent_import,
        "_validate_target_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        parent_import,
        "_verify_release_identity",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        parent_import,
        "v2_9_p95_source_binding_v210",
        lambda **_kwargs: deepcopy(source),
    )

    built = parent_import.build_v210_resealed_observed_p95_authority(
        repo_root=ROOT,
        contract=contract,
        raw_root=raw_root,
        profile_id=profile_id,
        expected_git_commit=commit,
        verified_v2_9_source_binding=source,
    )
    reservations = parent_import.verify_v210_resealed_observed_p95_authority(
        built["receipt"],
        repo_root=ROOT,
        raw_root=raw_root,
        expected_git_commit=commit,
        contract=contract,
    )
    assert reservations == built["receipt"]["reservations"]
    assert built["receipt"]["provider_boundary"] == {
        "provider_construction_during_reseal": False,
        "provider_calls_during_reseal": 0,
        "hosted_provider_calls_during_reseal": 0,
        "hosted_cost_usd_during_reseal": 0.0,
    }
    assert (
        parent_import.verify_v210_resealed_observed_p95_projection(
            built["projection"],
            receipt=built["receipt"],
            repo_root=ROOT,
            raw_root=raw_root,
            expected_git_commit=commit,
            contract=contract,
        )
        == built["projection"]
    )

    tampered_receipt = deepcopy(built["receipt"])
    tampered_receipt["evidence_use"] = "tampered"
    tampered_receipt = parent_import._seal(tampered_receipt)
    with pytest.raises(
        parent_import.PilotV210ParentImportError,
        match="differs from current source/release authority",
    ):
        parent_import.verify_v210_resealed_observed_p95_authority(
            tampered_receipt,
            repo_root=ROOT,
            raw_root=raw_root,
            expected_git_commit=commit,
            contract=contract,
        )

    tampered_projection = deepcopy(built["projection"])
    projection_key = next(iter(tampered_projection["projection"]))
    tampered_projection["projection"][projection_key]["sample_count"] += 1
    tampered_projection = parent_import._seal(tampered_projection)
    with pytest.raises(
        parent_import.PilotV210ParentImportError,
        match="differs from its receipt/source",
    ):
        parent_import.verify_v210_resealed_observed_p95_projection(
            tampered_projection,
            receipt=built["receipt"],
            repo_root=ROOT,
            raw_root=raw_root,
            expected_git_commit=commit,
            contract=contract,
        )


def test_source_manifest_hash_or_claim_tamper_fails_closed(tmp_path: Path) -> None:
    manifest = _manifest()
    tampered = deepcopy(manifest)
    tampered["import_policy"]["offline_candidate_admission_cells_imported"] = 1
    tampered = parent_import._seal(tampered)
    path = tmp_path / "pilot_v2_10_source_manifest.json"
    path.write_text(
        __import__("json").dumps(
            tampered,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        parent_import.PilotV210ParentImportError,
        match="authority/claim boundary drifted",
    ):
        parent_import.load_v210_source_manifest(path)

    assert hashlib.sha256(SOURCE_MANIFEST_PATH.read_bytes()).hexdigest() == (
        "8540bde06f364aa9ccf2a6937b78dec1f0d3b2c66b9e4943f9a3d2e20e4b19a7"
    )


def test_real_v29_raw_audit_is_zero_call_when_source_checkout_is_present() -> None:
    if not (V29_SCIENCE_ROOT / "experiment_results" / "pilot-v2.9" / "raw").is_dir():
        pytest.skip("immutable V2.9 science raw checkout is not installed")

    audit = parent_import.verify_v29_terminal_source(
        parent_repo_root=V29_SCIENCE_ROOT,
        evidence_repo_root=ROOT,
        target_contract=_contract(),
    )

    assert audit["raw_inventory"]["file_count"] == 623
    assert len(audit["imported_cells"]) == 16
    assert set(audit["p95_sources"]) == {
        "gpt52_main",
        "llama33_local_controlled",
    }
    assert audit["provider_construction_during_import"] is False
    assert audit["provider_calls_during_import"] == 0


@pytest.mark.parametrize(
    ("relative", "expected_content_sha256"),
    (
        (
            "q-ref-resolution/stage_receipt.json",
            "e9865c91ec078043489592813f62e72ca4f1d19239cf935a31699637e9f37d57",
        ),
        (
            "stage0-calibration/stage_receipt.json",
            "fc45635bbac056f9a72f3d8235286aa743592c793382192c156f7fc0c42c45d5",
        ),
    ),
)
def test_real_v29_v2_stage_receipts_use_integrity_excluded_hash_convention(
    relative: str,
    expected_content_sha256: str,
) -> None:
    path = V29_SCIENCE_ROOT / "experiment_results" / "pilot-v2.9" / "raw" / relative
    if not path.is_file():
        pytest.skip("immutable V2.9 science stage receipt is not installed")
    value = json.loads(path.read_text(encoding="utf-8"))

    parent_import._verify_bound_artifact_self_hash(
        value,
        name=f"real V2.9 {relative}",
    )

    assert value["integrity"]["content_sha256"] == expected_content_sha256
    unsigned = deepcopy(value)
    unsigned.pop("integrity")
    assert parent_import.canonical_sha256(unsigned) == expected_content_sha256


def test_v2_stage_receipt_tamper_and_wrong_hash_convention_fail_closed() -> None:
    path = (
        V29_SCIENCE_ROOT
        / "experiment_results"
        / "pilot-v2.9"
        / "raw"
        / "q-ref-resolution"
        / "stage_receipt.json"
    )
    if not path.is_file():
        pytest.skip("immutable V2.9 science stage receipt is not installed")
    value = json.loads(path.read_text(encoding="utf-8"))

    tampered = deepcopy(value)
    tampered["complete_cell_count"] = 0
    with pytest.raises(
        parent_import.PilotV210ParentImportError,
        match="schema or content hash mismatch",
    ):
        parent_import._verify_bound_artifact_self_hash(
            tampered,
            name="tampered V2.9 q-ref stage receipt",
        )

    wrongly_resealed = parent_import._seal(value)
    with pytest.raises(
        parent_import.PilotV210ParentImportError,
        match="schema or content hash mismatch",
    ):
        parent_import._verify_bound_artifact_self_hash(
            wrongly_resealed,
            name="wrong-convention V2.9 q-ref stage receipt",
        )
