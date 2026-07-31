from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from verified_memory.pilot_v2115_gate import (
    PilotV2115GateError,
    V2115_GATE_SCHEMA_VERSION,
    V2115_RESEALED_P95_AUTHORITY_SCHEMA_VERSION,
    build_v2115_post_gate_authority,
    build_v2115_resealed_observed_p95_authority,
    canonical_sha256,
    persist_v2115_post_gate_authority,
    persist_v2115_resealed_observed_p95_authority,
    runner_reservations_from_v2115_gate_binding,
    verified_v2115_gate_authority_binding,
    verified_v2115_observed_p95_authority_binding,
    verify_v2115_gate_receipt,
    verify_v2115_resealed_observed_p95_authority,
    v2115_observed_p95_projection_path,
    v2115_observed_p95_receipt_path,
)
from verified_memory.pilot_v2115_parent_import import (
    PilotV2115ParentImportError,
    V2115_SOURCE_MANIFEST_CONTENT_SHA256,
    V2115_SOURCE_MANIFEST_FILE_SHA256,
    _atomic_json,
    _seal,
    build_v2115_parent_import,
)


REPO = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = REPO / "experiments" / "pilot_v2_11_5_source_manifest.json"


class FrozenContract:
    contract_id = "finevo-pilot-v2.11.5"
    status = "frozen"
    canonical_hash = "a" * 64
    implementation = {"required_git_tag": "pilot-v2.11.5-science"}
    provider_profiles = {
        "gpt52_main": SimpleNamespace(
            transport="openai",
            requested_model="gpt-5.2-2025-12-11",
            served_model="gpt-5.2-2025-12-11",
        ),
        "gpt56_diagnostic": SimpleNamespace(
            transport="openai",
            requested_model="gpt-5.6-sol",
            served_model="gpt-5.6-sol",
        ),
    }

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_id": self.contract_id,
            "status": self.status,
            "implementation": dict(self.implementation),
            "v2115_forward_boundary": {
                "source_manifest": {
                    "path": "experiments/pilot_v2_11_5_source_manifest.json",
                    "schema_version": "finevo-pilot-v2.11.5-source-manifest-v1",
                    "file_sha256": V2115_SOURCE_MANIFEST_FILE_SHA256,
                    "content_sha256": V2115_SOURCE_MANIFEST_CONTENT_SHA256,
                }
            },
        }


@pytest.fixture(scope="module")
def v2114_terminal_release_root() -> Path:
    configured = os.environ.get("FINEVO_V2114_SCIENCE_RELEASE_ROOT")
    root = (
        Path(configured).expanduser().absolute()
        if configured
        else REPO.parent / "finevo-pilot-v2-11-4-science"
    )
    if not (root / "experiment_results" / "pilot-v2.11.4" / "raw").is_dir():
        pytest.skip(
            "exact V2.11.4 lineage replay requires its ignored science raw tree"
        )
    return root


@pytest.fixture(scope="module")
def release(
    tmp_path_factory: pytest.TempPathFactory,
    v2112_parent_release_root: Path,
    v2114_terminal_release_root: Path,
):
    root = tmp_path_factory.mktemp("v2115-release")
    (root / "experiments").mkdir()
    (root / "experiments" / SOURCE_MANIFEST.name).write_bytes(
        SOURCE_MANIFEST.read_bytes()
    )
    contract = FrozenContract()
    (root / "experiments" / "pilot_v2_11_5.yaml").write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "experiments"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-qm", "release inputs"], check=True
    )
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "tag",
            "-am",
            "V2.11.5 science",
            "pilot-v2.11.5-science",
        ],
        check=True,
        capture_output=True,
    )
    parent_receipt = build_v2115_parent_import(
        repo_root=root,
        contract=contract,
        child_git_commit=commit,
        source_repo_root=v2112_parent_release_root,
        lineage_repo_root=v2114_terminal_release_root,
        evidence_repo_root=REPO,
    )
    parent_path = (
        root
        / "experiment_results"
        / "pilot-v2.11.5"
        / "raw"
        / "parent-import"
        / "parent_import_receipt.json"
    )
    _atomic_json(parent_path, parent_receipt, repo_root=root)
    return root, contract, commit, parent_receipt


def test_per_model_reseal_preserves_four_source_provenance_fields(release) -> None:
    root, contract, commit, parent = release
    raw = root / "experiment_results" / "pilot-v2.11.5" / "raw"
    expected_provenance = {
        "source_authority_receipt_path": (
            "experiment_results/pilot-v2.11.2/raw/long-context-preflight/"
            "post_gate_authority.json"
        ),
        "source_authority_receipt_file_sha256": (
            "52ade890b123cd030b3d7242aa8347d7dc3a7040fe5f56de0b95938daa029312"
        ),
        "source_authority_receipt_content_sha256": (
            "0d95374c3e2db9fc5bf5c6156fb7bcdf0a9c94e26ed9995f74a2a542a8961aaa"
        ),
        "source_release_commit": "78870956b528946d415a9be5f5769b0893d16d74",
    }
    for model_id in ("gpt52_main", "gpt56_diagnostic"):
        result = persist_v2115_resealed_observed_p95_authority(
            repo_root=root,
            raw_root=raw,
            contract=contract,
            model_id=model_id,
            expected_git_commit=commit,
            parent_import_receipt=parent,
        )
        expected = (
            raw
            / "long-context-preflight"
            / "imported_observed_p95"
            / model_id
            / "observed_p95_authority_receipt.json"
        )
        assert Path(result["receipt"]) == expected
        assert Path(result["projection"]) == expected.with_name("projection_p95.json")
        assert result["provider_construction_during_reseal"] is False
        assert result["provider_calls_during_reseal"] == 0
        binding = verified_v2115_observed_p95_authority_binding(
            expected,
            repo_root=root,
            raw_root=raw,
            expected_git_commit=commit,
            contract=contract,
        )
        runtime = next(iter(binding["reservations"]))
        for call_kind, sample_count in (("action", 24), ("semantic", 8)):
            entry = binding["reservations"][runtime][call_kind]
            assert entry["reservation"]["sample_count"] == sample_count
            assert {
                key: entry["authority"][key] for key in expected_provenance
            } == expected_provenance
            assert entry["authority"]["pilot_contract_hash"] == (
                "c04f7d4c5ae0962a4a64b0ac543d890a1475b6f184f516534eeb8ff026505a37"
            )


def test_resealed_provenance_tamper_fails_closed(release) -> None:
    root, contract, commit, parent = release
    raw = root / "experiment_results" / "pilot-v2.11.5" / "raw"
    built = build_v2115_resealed_observed_p95_authority(
        repo_root=root,
        raw_root=raw,
        contract=contract,
        profile_id="gpt52_main",
        expected_git_commit=commit,
        parent_import_receipt=parent,
    )
    tampered = deepcopy(built["receipt"])
    runtime = next(iter(tampered["reservations"]))
    tampered["reservations"][runtime]["action"]["authority"][
        "source_release_commit"
    ] = "0" * 40
    tampered = _seal(tampered)
    with pytest.raises(PilotV2115GateError, match="differs from current parent-import replay"):
        verify_v2115_resealed_observed_p95_authority(
            tampered,
            repo_root=root,
            raw_root=raw,
            expected_git_commit=commit,
            contract=contract,
        )


def test_supplied_parent_substitution_is_rejected(release) -> None:
    root, contract, commit, parent = release
    raw = root / "experiment_results" / "pilot-v2.11.5" / "raw"
    substituted = deepcopy(parent)
    substituted["claim_boundary"] = "substituted"
    substituted = _seal(substituted)
    with pytest.raises(PilotV2115GateError, match="persisted bytes"):
        build_v2115_resealed_observed_p95_authority(
            repo_root=root,
            raw_root=raw,
            contract=contract,
            profile_id="gpt52_main",
            expected_git_commit=commit,
            parent_import_receipt=substituted,
        )


def test_global_gate_roundtrip_rebinds_current_release_provenance(release) -> None:
    root, contract, commit, parent = release
    raw = root / "experiment_results" / "pilot-v2.11.5" / "raw"
    bindings = {
        model_id: verified_v2115_observed_p95_authority_binding(
            v2115_observed_p95_receipt_path(raw, model_id),
            repo_root=root,
            raw_root=raw,
            expected_git_commit=commit,
            contract=contract,
        )
        for model_id in ("gpt52_main", "gpt56_diagnostic")
    }
    head = "c" * 64
    built = build_v2115_post_gate_authority(
        repo_root=root,
        contract=contract,
        expected_git_commit=commit,
        parent_import_receipt=parent,
        per_model_authority_bindings=bindings,
        ledger_event_chain_head=head,
    )
    assert built["schema_version"] == V2115_GATE_SCHEMA_VERSION
    verify_v2115_gate_receipt(
        built,
        expected_git_commit=commit,
        expected_contract_sha256=contract.canonical_hash,
    )
    path, persisted = persist_v2115_post_gate_authority(
        repo_root=root,
        raw_root=raw,
        contract=contract,
        expected_git_commit=commit,
        parent_import_receipt=parent,
        per_model_authority_bindings=bindings,
        ledger_event_chain_head=head,
    )
    assert path == raw / "long-context-preflight" / "post_gate_authority.json"
    assert persisted == built
    flat = verified_v2115_gate_authority_binding(
        path,
        repo_root=root,
        expected_git_commit=commit,
        expected_contract_sha256=contract.canonical_hash,
        contract=contract,
    )
    assert flat["receipt_content_sha256"] == built["receipt_sha256"]
    assert set(flat["reservations"]) == {
        "openai/gpt-5.2-2025-12-11",
        "openai/gpt-5.6-sol",
    }
    runner_rows = runner_reservations_from_v2115_gate_binding(flat)
    stable_fields = {
        "authority_id",
        "source_kind",
        "source_projection_schema_version",
        "source_preflight_run_id",
        "source_preflight_run_spec_sha256",
        "source_model_id",
        "source_served_model",
        "source_execution_artifact_sha256",
        "source_provider_call_journal_sha256",
    }
    generation_fields = {
        "pilot_contract_hash",
        "pilot_tag",
        "source_authority_receipt_content_sha256",
        "source_authority_receipt_file_sha256",
        "source_authority_receipt_path",
        "source_projection_content_sha256",
        "source_projection_file_sha256",
        "source_release_commit",
    }
    payload_hashes = {
        "gpt52_main": {
            "action": "fdbcf2d6696969452ea3e0aa1af7475618dcd058ad0f741f92886f23aa05a9f3",
            "semantic": "61c8bb94d82ec715eb7fc93d73b51fda9caf194c1e58627e737c5c13e69996de",
        },
        "gpt56_diagnostic": {
            "action": "4da97d868764ff87ac45cf517f8de3e08db1cb4debf4462cb3279dc75f4d9b3f",
            "semantic": "715708d6a4ad18a46d02c628c6688bf398d5f4ed5e88c519ade2bee642df9c7a",
        },
    }
    runtime_by_model = {
        "gpt52_main": "openai/gpt-5.2-2025-12-11",
        "gpt56_diagnostic": "openai/gpt-5.6-sol",
    }
    for model_id, runtime in runtime_by_model.items():
        source_rows = bindings[model_id]["reservations"][runtime]
        by_kind = runner_rows[runtime]
        for call_kind, row in by_kind.items():
            source = source_rows[call_kind]
            assert set(row["authority"]) == stable_fields | generation_fields
            assert {
                field: row["authority"][field] for field in stable_fields
            } == {
                field: source["authority"][field] for field in stable_fields
            }
            assert row["reservation"] == source["reservation"]
            assert canonical_sha256(row["reservation"]) == payload_hashes[model_id][
                call_kind
            ]
            assert source["authority"]["source_release_commit"] == (
                "78870956b528946d415a9be5f5769b0893d16d74"
            )
            assert row["authority"]["source_release_commit"] == commit
            assert row["authority"]["pilot_contract_hash"] == contract.canonical_hash
            assert row["authority"]["pilot_tag"] == "pilot-v2.11.5-science"
            assert row["authority"]["source_authority_receipt_path"] == (
                "experiment_results/pilot-v2.11.5/raw/long-context-preflight/"
                "post_gate_authority.json"
            )
            assert set(
                key
                for key in row["authority"]
                if key.startswith("source_authority_receipt_")
                or key == "source_release_commit"
            ) >= {
                "source_authority_receipt_path",
                "source_authority_receipt_file_sha256",
                "source_authority_receipt_content_sha256",
                "source_release_commit",
            }

    tampered = deepcopy(built)
    tampered["provider_boundary"]["provider_calls_during_authority_import"] = 1
    tampered["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    )
    with pytest.raises(PilotV2115GateError, match="zero-provider"):
        verify_v2115_gate_receipt(
            tampered,
            expected_git_commit=commit,
            expected_contract_sha256=contract.canonical_hash,
        )


def test_mapping_and_wrong_paths_fail_closed(release) -> None:
    root, contract, commit, parent = release
    raw = root / "experiment_results" / "pilot-v2.11.5" / "raw"
    built = build_v2115_resealed_observed_p95_authority(
        repo_root=root,
        raw_root=raw,
        contract=contract,
        profile_id="gpt52_main",
        expected_git_commit=commit,
        parent_import_receipt=parent,
    )
    assert built["receipt"]["schema_version"] == (
        V2115_RESEALED_P95_AUTHORITY_SCHEMA_VERSION
    )
    assert v2115_observed_p95_projection_path(raw, "gpt52_main") == (
        v2115_observed_p95_receipt_path(raw, "gpt52_main").with_name(
            "projection_p95.json"
        )
    )
    with pytest.raises(PilotV2115GateError, match="path drifted"):
        verified_v2115_observed_p95_authority_binding(
            raw / "parent-import" / "observed_p95" / "gpt52_main" / "x.json",
            repo_root=root,
            raw_root=raw,
            expected_git_commit=commit,
            contract=contract,
        )


def test_immutable_writer_rejects_symlink_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "experiment_results").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PilotV2115ParentImportError, match="without following symlinks"):
        _atomic_json(
            tmp_path / "experiment_results" / "escaped.json",
            {"escape": False},
            repo_root=tmp_path,
        )
    assert not (outside / "escaped.json").exists()
