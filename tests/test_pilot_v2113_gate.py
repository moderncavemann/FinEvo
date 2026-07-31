from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from verified_memory.pilot_v2113_gate import (
    PilotV2113GateError,
    V2113_GATE_SCHEMA_VERSION,
    V2113_RESEALED_P95_AUTHORITY_SCHEMA_VERSION,
    build_v2113_post_gate_authority,
    build_v2113_resealed_observed_p95_authority,
    canonical_sha256,
    persist_v2113_post_gate_authority,
    persist_v2113_resealed_observed_p95_authority,
    runner_reservations_from_v2113_gate_binding,
    verified_v2113_gate_authority_binding,
    verified_v2113_observed_p95_authority_binding,
    verify_v2113_gate_receipt,
    v2113_observed_p95_projection_path,
    v2113_observed_p95_receipt_path,
)
from verified_memory.pilot_v2113_parent_import import (
    PilotV2113ParentImportError,
    _atomic_json,
    _seal,
    build_v2113_parent_import,
)


REPO = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = REPO / "experiments" / "pilot_v2_11_3_source_manifest.json"


class FrozenContract:
    contract_id = "finevo-pilot-v2.11.3"
    status = "frozen"
    canonical_hash = "a" * 64
    implementation = {"required_git_tag": "pilot-v2.11.3-science"}
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
            "v2113_forward_boundary": {
                "source_manifest": {
                    "path": "experiments/pilot_v2_11_3_source_manifest.json",
                    "schema_version": "finevo-pilot-v2.11.3-source-manifest-v1",
                    "file_sha256": (
                        "f05dbac4951e99476c06883e3c1b792e7ccb459c16eb4d78ac15ddf7905598de"
                    ),
                    "content_sha256": (
                        "5c8e554d1a00803b81deb4f31b4a87ddf54a272861a7c750985cd72b18a95f00"
                    ),
                }
            },
        }


@pytest.fixture(scope="module")
def release(
    tmp_path_factory: pytest.TempPathFactory,
    v2112_parent_release_root: Path,
):
    root = tmp_path_factory.mktemp("v2113-release")
    (root / "experiments").mkdir()
    (root / "experiments" / SOURCE_MANIFEST.name).write_bytes(
        SOURCE_MANIFEST.read_bytes()
    )
    contract = FrozenContract()
    (root / "experiments" / "pilot_v2_11_3.yaml").write_text(
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
            "V2.11.3 science",
            "pilot-v2.11.3-science",
        ],
        check=True,
        capture_output=True,
    )
    parent_receipt = build_v2113_parent_import(
        repo_root=root,
        contract=contract,
        child_git_commit=commit,
        parent_science_root=v2112_parent_release_root,
        evidence_repo_root=REPO,
    )
    parent_path = (
        root
        / "experiment_results"
        / "pilot-v2.11.3"
        / "raw"
        / "parent-import"
        / "parent_import_receipt.json"
    )
    _atomic_json(parent_path, parent_receipt, repo_root=root)
    return root, contract, commit, parent_receipt


def test_per_model_reseal_uses_preflight_stage_and_zero_provider(release) -> None:
    root, contract, commit, parent = release
    raw = root / "experiment_results" / "pilot-v2.11.3" / "raw"
    for model_id in ("gpt52_main", "gpt56_diagnostic"):
        result = persist_v2113_resealed_observed_p95_authority(
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
        binding = verified_v2113_observed_p95_authority_binding(
            expected,
            repo_root=root,
            raw_root=raw,
            expected_git_commit=commit,
            contract=contract,
        )
        runtime = next(iter(binding["reservations"]))
        assert (
            binding["reservations"][runtime]["action"]["reservation"]["sample_count"]
            == 24
        )
        assert (
            binding["reservations"][runtime]["semantic"]["reservation"]["sample_count"]
            == 8
        )
        assert all(
            row["authority"]["pilot_contract_hash"]
            == "c04f7d4c5ae0962a4a64b0ac543d890a1475b6f184f516534eeb8ff026505a37"
            for row in binding["reservations"][runtime].values()
        )


def test_supplied_parent_substitution_is_rejected(release) -> None:
    root, contract, commit, parent = release
    raw = root / "experiment_results" / "pilot-v2.11.3" / "raw"
    substituted = deepcopy(parent)
    substituted["claim_boundary"] = "substituted"
    substituted = _seal(substituted)
    with pytest.raises(PilotV2113GateError, match="persisted bytes"):
        build_v2113_resealed_observed_p95_authority(
            repo_root=root,
            raw_root=raw,
            contract=contract,
            profile_id="gpt52_main",
            expected_git_commit=commit,
            parent_import_receipt=substituted,
        )


def test_global_gate_roundtrip_and_tamper_rejection(release) -> None:
    root, contract, commit, parent = release
    raw = root / "experiment_results" / "pilot-v2.11.3" / "raw"
    bindings = {
        model_id: verified_v2113_observed_p95_authority_binding(
            v2113_observed_p95_receipt_path(raw, model_id),
            repo_root=root,
            raw_root=raw,
            expected_git_commit=commit,
            contract=contract,
        )
        for model_id in ("gpt52_main", "gpt56_diagnostic")
    }
    head = "c" * 64
    built = build_v2113_post_gate_authority(
        repo_root=root,
        contract=contract,
        expected_git_commit=commit,
        parent_import_receipt=parent,
        per_model_authority_bindings=bindings,
        ledger_event_chain_head=head,
    )
    assert built["schema_version"] == V2113_GATE_SCHEMA_VERSION
    verify_v2113_gate_receipt(
        built,
        expected_git_commit=commit,
        expected_contract_sha256=contract.canonical_hash,
    )
    path, persisted = persist_v2113_post_gate_authority(
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
    flat = verified_v2113_gate_authority_binding(
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
    runner_rows = runner_reservations_from_v2113_gate_binding(flat)
    for by_kind in runner_rows.values():
        for row in by_kind.values():
            assert row["authority"]["source_release_commit"] == commit
            assert row["authority"]["source_authority_receipt_path"] == (
                "experiment_results/pilot-v2.11.3/raw/long-context-preflight/"
                "post_gate_authority.json"
            )

    tampered = deepcopy(built)
    tampered["provider_boundary"]["provider_calls_during_authority_import"] = 1
    tampered["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    )
    with pytest.raises(PilotV2113GateError, match="zero-provider"):
        verify_v2113_gate_receipt(
            tampered,
            expected_git_commit=commit,
            expected_contract_sha256=contract.canonical_hash,
        )


def test_mapping_and_wrong_paths_fail_closed(release) -> None:
    root, contract, commit, parent = release
    raw = root / "experiment_results" / "pilot-v2.11.3" / "raw"
    built = build_v2113_resealed_observed_p95_authority(
        repo_root=root,
        raw_root=raw,
        contract=contract,
        profile_id="gpt52_main",
        expected_git_commit=commit,
        parent_import_receipt=parent,
    )
    assert built["receipt"]["schema_version"] == (
        V2113_RESEALED_P95_AUTHORITY_SCHEMA_VERSION
    )
    assert v2113_observed_p95_projection_path(raw, "gpt52_main") == (
        v2113_observed_p95_receipt_path(raw, "gpt52_main").with_name(
            "projection_p95.json"
        )
    )
    with pytest.raises(PilotV2113GateError, match="path drifted"):
        verified_v2113_observed_p95_authority_binding(
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
    with pytest.raises(PilotV2113ParentImportError, match="without following symlinks"):
        _atomic_json(
            tmp_path / "experiment_results" / "escaped.json",
            {"escape": False},
            repo_root=tmp_path,
        )
    assert not (outside / "escaped.json").exists()
