from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from verified_memory.pilot_contract import load_pilot_contract
from verified_memory import pilot_v2101_parent_import as parent_import
from verified_memory import pilot_v210_parent_import as v210_parent_import


ROOT = Path(__file__).resolve().parents[1]
V2101_CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_10_1.yaml"
V210_SOURCE_MANIFEST_PATH = ROOT / "experiments" / "pilot_v2_10_source_manifest.json"
V210_SCIENCE_ROOT = ROOT.parent / "finevo-pilot-v2-10-science"


def _contract():
    return load_pilot_contract(V2101_CONTRACT_PATH)


def _real_manifest():
    if not V210_SCIENCE_ROOT.is_dir():
        pytest.skip("immutable V2.10 science checkout is not installed")
    return parent_import.build_v2101_source_manifest(
        parent_repo_root=V210_SCIENCE_ROOT,
        evidence_repo_root=ROOT,
        target_contract=_contract(),
    )


def test_v2101_release_constants_and_parent_debit_are_exact() -> None:
    assert parent_import.V2101_CONTRACT_ID == "finevo-pilot-v2.10.1"
    assert parent_import.V2101_SCIENCE_TAG == "pilot-v2.10.1-science"
    assert parent_import.V2101_SOURCE_MANIFEST_PATH.as_posix() == (
        "experiments/pilot_v2_10_1_source_manifest.json"
    )
    assert parent_import.V2101_SOURCE_MANIFEST_SCHEMA_VERSION == (
        "finevo-pilot-v2.10.1-source-manifest-v1"
    )
    assert parent_import.V2101_CUMULATIVE_DEBIT.to_dict() == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": (
            "d1b54c14d016c2b157db9e334d054ab9c7e86371d3fb9662a95fb94e50ce964b"
        ),
        "parent_run_ledger_sha256": (
            "ef2a7a1d003e4b876749cf87e7a49bd5080e1096b7d6beedb797d6adde149db6"
        ),
        "parent_budget_ledger_sha256": (
            "a03b87a18aae4ddcbcc2f546bf142745969a6225922d7b2da327c7f9730d3f6a"
        ),
        "stage_bucket": "parent_v23",
        "cost_usd": 3.212770875,
        "hosted_completions": 184,
        "storage_bytes": 70_035_938,
        "record_sha256": (
            "4837821a5f059714ef8fa6f8b22522bc693c8adb0edc7603367823a870e94510"
        ),
    }


def test_real_v210_terminal_lineage_is_exact_and_zero_provider() -> None:
    if not V210_SCIENCE_ROOT.is_dir():
        pytest.skip("immutable V2.10 science checkout is not installed")

    audit = parent_import.verify_v210_terminal_lineage(
        parent_repo_root=V210_SCIENCE_ROOT,
        evidence_repo_root=ROOT,
    )

    assert audit["raw_inventory"] == {
        "root": "experiment_results/pilot-v2.10/raw",
        "file_count": 637,
        "storage_bytes": 20_126_496,
        "inventory_sha256": (
            "d8964a15abed0d77598d2c2cf80136e438b67559796cc93f8566dca17e584baa"
        ),
        "rows": audit["raw_inventory"]["rows"],
    }
    assert len(audit["raw_inventory"]["rows"]) == 637
    assert audit["run_ledger"]["ledger_sha256"] == (
        "ef2a7a1d003e4b876749cf87e7a49bd5080e1096b7d6beedb797d6adde149db6"
    )
    assert audit["budget_ledger"]["ledger_sha256"] == (
        "a03b87a18aae4ddcbcc2f546bf142745969a6225922d7b2da327c7f9730d3f6a"
    )
    assert audit["qref_failure_receipt"]["status"] == "integrity-stopped"
    assert audit["qref_failure_receipt"]["integrity"]["content_sha256"] == (
        "48ae5807da2c3175b3fd427cc023796e7bd81c5b77695789a900474e023da098"
    )
    assert audit["evidence"]["status_counts"] == {
        "complete": 1,
        "integrity-stopped": 210,
    }
    assert audit["provider_construction_during_import"] is False
    assert audit["provider_calls_during_import"] == 0
    assert audit["hosted_provider_calls_during_import"] == 0
    assert audit["hosted_cost_usd_during_import"] == 0.0


def test_real_source_manifest_uses_v29_bytes_not_v210_wrappers() -> None:
    manifest = _real_manifest()
    v210_manifest = v210_parent_import.load_v210_source_manifest(
        V210_SOURCE_MANIFEST_PATH,
        expected_file_sha256=(
            "8540bde06f364aa9ccf2a6937b78dec1f0d3b2c66b9e4943f9a3d2e20e4b19a7"
        ),
        expected_content_sha256=(
            "fc781697a9260fa63d0535eafa24b87a8386a76dca55f3ce95ba59e12ceb4224"
        ),
    )

    assert manifest["schema_version"] == ("finevo-pilot-v2.10.1-source-manifest-v1")
    assert manifest["v2_10_terminal_parent"]["terminal_denominator"][
        "status_counts"
    ] == {"complete": 1, "integrity-stopped": 210}
    assert manifest["v2_9_exact_source"] == {
        "source_path_kind": ("byte-exact-v2.9-raw-inside-v2.10-terminal-snapshot"),
        "source_path": (
            "experiment_results/pilot-v2.10/raw/" "parent-import/v2_9_raw_snapshot"
        ),
        "declared_root": "experiment_results/pilot-v2.9/raw",
        "file_count": 623,
        "storage_bytes": 19_288_343,
        "inventory_sha256": (
            "ae478634a83a98bd206bcafa03f87636fcc392f8dd1e8e234f84696f245ef22f"
        ),
        "v2_10_wrapper_is_current_authority": False,
    }
    assert len(manifest["imported_complete_cells"]) == 16

    prior_by_source = {
        row["source_run_id"]: row for row in v210_manifest["imported_complete_cells"]
    }
    current_by_source = {
        row["source_run_id"]: row for row in manifest["imported_complete_cells"]
    }
    assert set(current_by_source) == set(prior_by_source)
    for source_run_id, current in current_by_source.items():
        prior = prior_by_source[source_run_id]
        assert current["source_spec"] == prior["source_spec"]
        assert current["source_artifacts"] == prior["source_artifacts"]
        assert current["target_run_id"].startswith("finevo-pilot-v2.10.1--")
        assert current["target_spec"]["contract_id"] == ("finevo-pilot-v2.10.1")
        assert current["source_path_kind"] == (
            "byte-exact-v2.9-raw-inside-v2.10-terminal-snapshot"
        )

    for profile_id, source in manifest[
        "v2_9_p95_sources_for_current_release_reseal"
    ].items():
        prior = v210_manifest["v2_9_p95_sources_for_current_release_reseal"][profile_id]
        assert source["v2_8_observed_p95_origin"] == (prior["v2_8_observed_p95_origin"])
        assert source["normalized_v2_9_binding"] == (prior["normalized_v2_9_binding"])
        assert source["source_path_kind"] == (
            "byte-exact-v2.9-raw-inside-v2.10-terminal-snapshot"
        )


def test_source_manifest_round_trip_and_resealed_tamper_fail_closed(
    tmp_path: Path,
) -> None:
    manifest = _real_manifest()
    path = tmp_path / "pilot_v2_10_1_source_manifest.json"
    parent_import.write_v2101_source_manifest_draft(path, manifest)

    loaded = parent_import.load_v2101_source_manifest(
        path,
        expected_file_sha256=parent_import._sha256(path.read_bytes()),
        expected_content_sha256=manifest["integrity"]["content_sha256"],
    )
    assert loaded == manifest

    tampered = deepcopy(manifest)
    tampered["v2_10_terminal_parent"]["terminal_denominator"]["status_counts"] = {
        "complete": 2,
        "integrity-stopped": 209,
    }
    tampered = parent_import._seal(tampered)
    with pytest.raises(
        parent_import.PilotV2101ParentImportError,
        match="authority/claim boundary drifted",
    ):
        parent_import._validate_source_manifest_structure(tampered)


@pytest.mark.parametrize(
    "profile_id",
    ("gpt52_main", "llama33_local_controlled"),
)
def test_p95_reseal_uses_exact_v29_values_and_rejects_v210_wrapper_shape(
    monkeypatch: pytest.MonkeyPatch,
    profile_id: str,
) -> None:
    contract = _contract()
    manifest = _real_manifest()
    source = manifest["v2_9_p95_sources_for_current_release_reseal"][profile_id]
    raw_root = ROOT / "experiment_results" / "pilot-v2.10.1" / "raw"
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
            "path": "experiments/pilot_v2_10_1.yaml",
            "file_sha256": "b" * 64,
            "contract_id": selected.contract_id,
            "contract_sha256": selected.canonical_hash,
        },
    )
    built = parent_import.build_v2101_resealed_observed_p95_authority(
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
        assert authority["pilot_tag"] == "pilot-v2.10.1-science"
    assert built["receipt"]["provider_boundary"] == {
        "provider_construction_during_reseal": False,
        "provider_calls_during_reseal": 0,
        "hosted_provider_calls_during_reseal": 0,
        "hosted_cost_usd_during_reseal": 0.0,
    }
    assert built["projection"]["bindings"]["source_kind"] == (
        "v2.9-exact-raw-via-v2.10-terminal-v2.10.1"
    )

    wrapper = deepcopy(source)
    wrapper.pop("source_path_kind")
    wrapper["v2_10_current_wrapper"] = {
        "schema_version": ("finevo-pilot-v2.10-resealed-observed-p95-authority-v1")
    }
    with pytest.raises(
        parent_import.PilotV2101ParentImportError,
        match="source lineage shape drifted",
    ):
        parent_import.build_v2101_resealed_observed_p95_authority(
            repo_root=ROOT,
            contract=contract,
            raw_root=raw_root,
            profile_id=profile_id,
            expected_git_commit=commit,
            verified_v2_9_source_binding=wrapper,
        )


def test_parent_budget_debit_routes_only_v2101_contract() -> None:
    contract = _contract()
    assert parent_import.parent_budget_debit_for_v2101(contract) == (
        parent_import.V2101_CUMULATIVE_DEBIT
    )

    class Other:
        contract_id = "finevo-pilot-v2.10"

    assert parent_import.parent_budget_debit_for_v2101(Other()) is None


def test_imported_qref_binding_accepts_frozen_stage_receipt_v2_hash_domain() -> None:
    manifest = _real_manifest()
    contract = _contract()
    qref = contract.expand(stage="q-ref-resolution")[0]
    v210_raw = V210_SCIENCE_ROOT / "experiment_results" / "pilot-v2.10" / "raw"

    binding = parent_import.verified_v2101_imported_prerequisite_binding(
        v210_raw,
        manifest,
        qref,
    )

    assert binding["source_stage_id"] == "q-ref-resolution"
    assert binding["q_ref"] == 63.50397933257746
    assert binding["source_release"]["source_path_kind"] == (
        "byte-exact-v2.9-raw-inside-v2.10-terminal-snapshot"
    )
    assert binding["provider_construction_during_verification"] is False
    assert binding["provider_calls_during_verification"] == 0
    assert binding["source_artifacts"]["stage_receipt"]["content_sha256"] == (
        "e9865c91ec078043489592813f62e72ca4f1d19239cf935a31699637e9f37d57"
    )


@pytest.mark.parametrize(
    "profile_id",
    ("gpt52_main", "llama33_local_controlled"),
)
def test_p95_authority_and_projection_verifiers_rebuild_exact_source(
    monkeypatch: pytest.MonkeyPatch,
    profile_id: str,
) -> None:
    contract = _contract()
    manifest = _real_manifest()
    source = manifest["v2_9_p95_sources_for_current_release_reseal"][profile_id]
    raw_root = ROOT / "experiment_results" / "pilot-v2.10.1" / "raw"
    commit = "c" * 40

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
        "_load_current_contract",
        lambda _root, _selected: contract,
    )
    monkeypatch.setattr(
        parent_import,
        "_contract_binding",
        lambda _root, selected: {
            "path": "experiments/pilot_v2_10_1.yaml",
            "file_sha256": "d" * 64,
            "contract_id": selected.contract_id,
            "contract_sha256": selected.canonical_hash,
        },
    )
    monkeypatch.setattr(
        parent_import,
        "v2_9_p95_source_binding_v2101",
        lambda **_kwargs: deepcopy(source),
    )
    built = parent_import.build_v2101_resealed_observed_p95_authority(
        repo_root=ROOT,
        contract=contract,
        raw_root=raw_root,
        profile_id=profile_id,
        expected_git_commit=commit,
        verified_v2_9_source_binding=source,
    )

    reservations = parent_import.verify_v2101_resealed_observed_p95_authority(
        built["receipt"],
        repo_root=ROOT,
        raw_root=raw_root,
        expected_git_commit=commit,
        contract=contract,
    )
    verified_projection = parent_import.verify_v2101_resealed_observed_p95_projection(
        built["projection"],
        receipt=built["receipt"],
        repo_root=ROOT,
        raw_root=raw_root,
        expected_git_commit=commit,
        contract=contract,
    )

    assert reservations == built["receipt"]["reservations"]
    assert verified_projection == built["projection"]

    tampered = deepcopy(built["projection"])
    tampered["bindings"]["source_kind"] = "v2.10-current-wrapper-forbidden"
    tampered = parent_import._seal(tampered)
    with pytest.raises(
        parent_import.PilotV2101ParentImportError,
        match="differs from its receipt/source",
    ):
        parent_import.verify_v2101_resealed_observed_p95_projection(
            tampered,
            receipt=built["receipt"],
            repo_root=ROOT,
            raw_root=raw_root,
            expected_git_commit=commit,
            contract=contract,
        )
