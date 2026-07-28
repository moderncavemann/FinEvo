from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from verified_memory.pilot_contract import load_pilot_contract
from verified_memory import pilot_v2101_parent_import as v2101_parent_import
from verified_memory import pilot_v2102_parent_import as parent_import


ROOT = Path(__file__).resolve().parents[1]
V2102_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_10_2.yaml"
V2101_SOURCE_MANIFEST_PATH = (
    ROOT / "experiments" / "pilot_v2_10_1_source_manifest.json"
)
V2101_SCIENCE_ROOT = ROOT.parent / "finevo-pilot-v2-10-1-science"


def _contract():
    return load_pilot_contract(V2102_CONTRACT_PATH)


def _real_manifest():
    if not V2101_SCIENCE_ROOT.is_dir():
        pytest.skip("immutable V2.10.1 science checkout is not installed")
    return parent_import.build_v2102_source_manifest(
        parent_repo_root=V2101_SCIENCE_ROOT,
        evidence_repo_root=ROOT,
        target_contract=_contract(),
    )


def test_v2102_release_constants_and_parent_debit_are_exact() -> None:
    assert parent_import.V2102_CONTRACT_ID == "finevo-pilot-v2.10.2"
    assert parent_import.V2102_SCIENCE_TAG == "pilot-v2.10.2-science"
    assert parent_import.V2102_SOURCE_MANIFEST_PATH.as_posix() == (
        "experiments/pilot_v2_10_2_source_manifest.json"
    )
    assert parent_import.V2101_PARENT_SOURCE_MANIFEST_FILE_SHA256 == (
        "e9360d9754cd054386ff03264c331091404555379457a59a7b01344f4a8f2d8f"
    )
    assert parent_import.V2101_PARENT_SOURCE_MANIFEST_CONTENT_SHA256 == (
        "11447dd0c231140102411eb231b8716c8f1581d0fa1533e98ccc51c3afb31426"
    )
    assert parent_import.V2101_PARENT_RELEASE_ATTESTATION_FILE_SHA256 == (
        "d7ac625c787a0d7287eb6a20ca3172b251966487915dee376d391494b2670443"
    )
    assert parent_import.V2101_PARENT_RELEASE_ATTESTATION_CONTENT_SHA256 == (
        "c45f93e89cb5cb2e49539c3a1ba18ad8a193489fd2ab92a3bf7da0ba8869c3b3"
    )
    assert parent_import.V2102_CUMULATIVE_DEBIT.to_dict() == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": (
            "1f9c642c155d5256815cb14a68335b65a25497523c14210c36f89070b3c8d996"
        ),
        "parent_run_ledger_sha256": (
            "75e91445745ec5480577327053a8d7eaefc4352cb6f3f176693460cc712d22b6"
        ),
        "parent_budget_ledger_sha256": (
            "87d313e4f96766f3137c5c0175b0adb6e8a24d4c7697e556e2e0e46f00525161"
        ),
        "stage_bucket": "parent_v23",
        "cost_usd": 3.212770875,
        "hosted_completions": 184,
        "storage_bytes": 92_541_342,
        "record_sha256": (
            "4af5a2c29b3dcc417e261f25b7544e9ca3198f3c3b67d43ea6fbdf50e2ccdad9"
        ),
    }


def test_real_v2101_terminal_lineage_is_exact_and_zero_provider() -> None:
    if not V2101_SCIENCE_ROOT.is_dir():
        pytest.skip("immutable V2.10.1 science checkout is not installed")

    audit = parent_import.verify_v2101_terminal_lineage(
        parent_repo_root=V2101_SCIENCE_ROOT,
        evidence_repo_root=ROOT,
    )

    assert audit["raw_inventory"] == {
        "root": "experiment_results/pilot-v2.10.1/raw",
        "file_count": 966,
        "storage_bytes": 23_559_957,
        "inventory_sha256": (
            "63385589f81342822f705c47fe09ce10629a1ccc667ec13e47e7de36cec31413"
        ),
        "rows": audit["raw_inventory"]["rows"],
    }
    assert len(audit["raw_inventory"]["rows"]) == 966
    assert audit["run_ledger"]["ledger_sha256"] == (
        "75e91445745ec5480577327053a8d7eaefc4352cb6f3f176693460cc712d22b6"
    )
    assert audit["budget_ledger"]["ledger_sha256"] == (
        "87d313e4f96766f3137c5c0175b0adb6e8a24d4c7697e556e2e0e46f00525161"
    )
    assert audit["qref_receipt"]["status"] == "complete"
    assert audit["qref_receipt"]["integrity"]["content_sha256"] == (
        "8d22ec395608285dc96da65b0349389255fa6ca997b5de422b35686b496bc7db"
    )
    assert audit["source_manifest"] == {
        "path": "experiments/pilot_v2_10_1_source_manifest.json",
        "schema_version": "finevo-pilot-v2.10.1-source-manifest-v1",
        "file_sha256": (
            "e9360d9754cd054386ff03264c331091404555379457a59a7b01344f4a8f2d8f"
        ),
        "content_sha256": (
            "11447dd0c231140102411eb231b8716c8f1581d0fa1533e98ccc51c3afb31426"
        ),
    }
    assert audit["release_attestation"] == {
        "path": (
            "experiment_results/pilot-v2.10.1/raw/release_attestation.json"
        ),
        "schema_version": "finevo-scientific-release-attestation-v2",
        "file_sha256": (
            "d7ac625c787a0d7287eb6a20ca3172b251966487915dee376d391494b2670443"
        ),
        "content_sha256": (
            "c45f93e89cb5cb2e49539c3a1ba18ad8a193489fd2ab92a3bf7da0ba8869c3b3"
        ),
        "status": "pass",
        "science_tag": "pilot-v2.10.1-science",
        "science_tag_object": "2e6137cb5f4c3c8e5dc174efe8813cf04f2490f5",
        "science_commit": "b5bfa9b86d3cdb706cea5be707597bef8ac85aed",
    }
    assert audit["evidence"]["status_counts"] == {
        "complete": 26,
        "failed": 185,
    }
    assert audit["evidence"]["offline_candidate_admission_cells_observed"] == 10
    assert audit["evidence"]["actor_performance_treatment_outcome_blind"] is True
    assert audit["provider_construction_during_import"] is False
    assert audit["provider_calls_during_import"] == 0
    assert audit["hosted_provider_calls_during_import"] == 0
    assert audit["hosted_cost_usd_during_import"] == 0.0


def test_source_manifest_uses_v29_bytes_not_v2101_wrappers() -> None:
    manifest = _real_manifest()
    prior = v2101_parent_import.load_v2101_source_manifest(
        V2101_SOURCE_MANIFEST_PATH,
        expected_file_sha256=(
            "e9360d9754cd054386ff03264c331091404555379457a59a7b01344f4a8f2d8f"
        ),
        expected_content_sha256=(
            "11447dd0c231140102411eb231b8716c8f1581d0fa1533e98ccc51c3afb31426"
        ),
    )

    assert manifest["v2_10_1_terminal_parent"]["terminal_denominator"][
        "status_counts"
    ] == {"complete": 26, "failed": 185}
    assert manifest["v2_10_1_terminal_parent"]["source_manifest"] == {
        "path": "experiments/pilot_v2_10_1_source_manifest.json",
        "schema_version": "finevo-pilot-v2.10.1-source-manifest-v1",
        "file_sha256": (
            "e9360d9754cd054386ff03264c331091404555379457a59a7b01344f4a8f2d8f"
        ),
        "content_sha256": (
            "11447dd0c231140102411eb231b8716c8f1581d0fa1533e98ccc51c3afb31426"
        ),
    }
    assert prior["integrity"]["content_sha256"] == manifest[
        "v2_10_1_terminal_parent"
    ]["source_manifest"]["content_sha256"]
    assert manifest["v2_10_1_terminal_parent"]["release"][
        "release_attestation"
    ]["file_sha256"] == (
        "d7ac625c787a0d7287eb6a20ca3172b251966487915dee376d391494b2670443"
    )
    assert manifest["v2_9_exact_source"] == {
        "source_path_kind": (
            "byte-exact-v2.9-raw-inside-v2.10.1-terminal-snapshot"
        ),
        "source_path": (
            "experiment_results/pilot-v2.10.1/raw/"
            "parent-import/v2_9_raw_snapshot"
        ),
        "declared_root": "experiment_results/pilot-v2.9/raw",
        "file_count": 623,
        "storage_bytes": 19_288_343,
        "inventory_sha256": (
            "ae478634a83a98bd206bcafa03f87636fcc392f8dd1e8e234f84696f245ef22f"
        ),
        "v2_10_1_wrapper_is_current_authority": False,
    }
    current_by_source = {
        row["source_run_id"]: row for row in manifest["imported_complete_cells"]
    }
    prior_by_source = {
        row["source_run_id"]: row for row in prior["imported_complete_cells"]
    }
    assert set(current_by_source) == set(prior_by_source)
    for source_run_id, current in current_by_source.items():
        previous = prior_by_source[source_run_id]
        assert current["source_spec"] == previous["source_spec"]
        assert current["source_artifacts"] == previous["source_artifacts"]
        assert current["target_run_id"].startswith("finevo-pilot-v2.10.2--")
        assert current["target_spec"]["contract_id"] == "finevo-pilot-v2.10.2"
        assert current["source_path_kind"] == (
            "byte-exact-v2.9-raw-inside-v2.10.1-terminal-snapshot"
        )
    assert manifest["observation_boundary"][
        "v2_10_1_offline_candidate_admission_cells_observed"
    ] == 10
    assert manifest["observation_boundary"]["a_d_cells_fresh_in_v2_10_2"] == 195


def test_source_manifest_round_trip_and_parent_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    manifest = _real_manifest()
    path = tmp_path / "pilot_v2_10_2_source_manifest.json"
    parent_import.write_v2102_source_manifest_draft(path, manifest)
    loaded = parent_import.load_v2102_source_manifest(
        path,
        expected_file_sha256=parent_import._sha256(path.read_bytes()),
        expected_content_sha256=manifest["integrity"]["content_sha256"],
    )
    assert loaded == manifest

    tampered = deepcopy(manifest)
    tampered["v2_10_1_terminal_parent"]["terminal_denominator"][
        "status_counts"
    ] = {"complete": 211}
    tampered = parent_import._seal(tampered)
    with pytest.raises(
        parent_import.PilotV2102ParentImportError,
        match="authority/claim boundary drifted",
    ):
        parent_import._validate_source_manifest_structure(tampered)

    tampered_source = deepcopy(manifest)
    tampered_source["v2_10_1_terminal_parent"]["source_manifest"][
        "content_sha256"
    ] = "0" * 64
    tampered_source = parent_import._seal(tampered_source)
    with pytest.raises(
        parent_import.PilotV2102ParentImportError,
        match="authority/claim boundary drifted",
    ):
        parent_import._validate_source_manifest_structure(tampered_source)

    tampered_release = deepcopy(manifest)
    tampered_release["v2_10_1_terminal_parent"]["release"][
        "release_attestation"
    ]["content_sha256"] = "0" * 64
    tampered_release = parent_import._seal(tampered_release)
    with pytest.raises(
        parent_import.PilotV2102ParentImportError,
        match="authority/claim boundary drifted",
    ):
        parent_import._validate_source_manifest_structure(tampered_release)


@pytest.mark.parametrize(
    "profile_id",
    ("gpt52_main", "llama33_local_controlled"),
)
def test_p95_reseal_uses_exact_v29_values_and_rejects_v2101_wrapper_shape(
    monkeypatch: pytest.MonkeyPatch,
    profile_id: str,
) -> None:
    contract = _contract()
    manifest = _real_manifest()
    source = manifest["v2_9_p95_sources_for_current_release_reseal"][profile_id]
    raw_root = ROOT / "experiment_results" / "pilot-v2.10.2" / "raw"
    commit = "a" * 40

    monkeypatch.setattr(
        parent_import,
        "_validate_target_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        parent_import,
        "_contract_binding",
        lambda _root, selected: {
            "path": "experiments/pilot_v2_10_2.yaml",
            "file_sha256": "b" * 64,
            "contract_id": selected.contract_id,
            "contract_sha256": selected.canonical_hash,
        },
    )
    built = parent_import.build_v2102_resealed_observed_p95_authority(
        repo_root=ROOT,
        contract=contract,
        raw_root=raw_root,
        profile_id=profile_id,
        expected_git_commit=commit,
        verified_v2_9_source_binding=source,
    )

    nested = source["v2_8_observed_p95_origin"]
    runtime_model = nested["runtime_model"]
    for call_kind in ("action", "semantic"):
        assert (
            built["receipt"]["reservations"][runtime_model][call_kind]["reservation"]
            == source["normalized_v2_9_binding"]["reservations"][runtime_model][
                call_kind
            ]["reservation"]
        )
        authority = built["receipt"]["reservations"][runtime_model][call_kind][
            "authority"
        ]
        assert authority["pilot_contract_hash"] == contract.canonical_hash
        assert authority["pilot_tag"] == "pilot-v2.10.2-science"
    assert built["receipt"]["provider_boundary"] == {
        "provider_construction_during_reseal": False,
        "provider_calls_during_reseal": 0,
        "hosted_provider_calls_during_reseal": 0,
        "hosted_cost_usd_during_reseal": 0.0,
    }
    assert built["receipt"]["v2_10_1_terminal_lineage"]["source_manifest"] == (
        manifest["v2_10_1_terminal_parent"]["source_manifest"]
    )
    assert built["receipt"]["v2_10_1_terminal_lineage"][
        "release_attestation"
    ] == manifest["v2_10_1_terminal_parent"]["release"]["release_attestation"]

    wrapper = deepcopy(source)
    wrapper.pop("source_path_kind")
    wrapper["v2_10_1_current_wrapper"] = {
        "schema_version": (
            "finevo-pilot-v2.10.1-resealed-observed-p95-authority-v1"
        )
    }
    with pytest.raises(
        parent_import.PilotV2102ParentImportError,
        match="source lineage shape drifted",
    ):
        parent_import.build_v2102_resealed_observed_p95_authority(
            repo_root=ROOT,
            contract=contract,
            raw_root=raw_root,
            profile_id=profile_id,
            expected_git_commit=commit,
            verified_v2_9_source_binding=wrapper,
        )


def test_parent_budget_debit_routes_only_v2102_contract() -> None:
    contract = _contract()
    assert parent_import.parent_budget_debit_for_v2102(contract) == (
        parent_import.V2102_CUMULATIVE_DEBIT
    )

    class Other:
        contract_id = "finevo-pilot-v2.10.1"

    assert parent_import.parent_budget_debit_for_v2102(Other()) is None
