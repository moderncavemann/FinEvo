from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import pytest

from verified_memory import observed_p95_authority as authority
from verified_memory import pilot_v27_stage0_import as stage0_import
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_V2_6_CANONICAL_SHA256,
    load_pilot_contract,
)
from verified_memory.runner import (
    OBSERVED_P95_AUTHORITY_ID,
    OBSERVED_P95_PROJECTION_SCHEMA_VERSION,
    OBSERVED_P95_SOURCE_KIND,
)


ROOT = Path(__file__).resolve().parents[1]
V27_CONTRACT = ROOT / "experiments" / "pilot_v2_7.yaml"
EXPECTED_COMMIT = "a" * 40
PROFILE_ID = "llama33_local_controlled"
RUNTIME_MODEL = "ollama/llama3.3:70b-instruct-q4_K_M"
SERVED_MODEL = "llama3.3:70b-instruct-q4_K_M"


def _reservation(
    *,
    prompt_tokens: float,
    completion_tokens: float,
    sample_count: int,
) -> dict[str, Any]:
    reserved_prompt = math.ceil(prompt_tokens * 1.25)
    reserved_completion = math.ceil(completion_tokens * 1.25)
    return {
        "sample_count": sample_count,
        "raw_p95": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": 0.0,
        },
        "reserved_p95": {
            "prompt_tokens": reserved_prompt,
            "completion_tokens": reserved_completion,
            "total_tokens": reserved_prompt + reserved_completion,
            "cost_usd": 0.0,
        },
        "reserve_multiplier": 1.25,
    }


def _source_authority() -> dict[str, Any]:
    return {
        "authority_id": OBSERVED_P95_AUTHORITY_ID,
        "source_kind": OBSERVED_P95_SOURCE_KIND,
        "pilot_contract_hash": PILOT_CONTRACT_V2_6_CANONICAL_SHA256,
        "pilot_tag": "pilot-v2.6-science",
        "source_projection_schema_version": (OBSERVED_P95_PROJECTION_SCHEMA_VERSION),
        "source_projection_file_sha256": "1" * 64,
        "source_projection_content_sha256": "2" * 64,
        "source_preflight_run_id": "fixture-v2.3-preflight",
        "source_preflight_run_spec_sha256": "3" * 64,
        "source_model_id": PROFILE_ID,
        "source_served_model": SERVED_MODEL,
        "source_execution_artifact_sha256": "4" * 64,
        "source_provider_call_journal_sha256": "5" * 64,
    }


@pytest.fixture
def v27_authority_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Any, Path, dict[str, Any]]:
    repo_root = tmp_path / "repo"
    contract_path = repo_root / "experiments" / "pilot_v2_7.yaml"
    contract_path.parent.mkdir(parents=True)
    for source_name in (
        "pilot_v2_7.yaml",
        "pilot_v2_4_parent_source_manifest.json",
        "pilot_v2_5_source_manifest.json",
        "pilot_v2_6_source_manifest.json",
    ):
        shutil.copyfile(
            ROOT / "experiments" / source_name,
            contract_path.parent / source_name,
        )
    contract = load_pilot_contract(contract_path)
    raw_root = repo_root / "experiment_results" / "pilot-v2.7" / "raw"
    snapshot_root = raw_root / "parent-import" / "v2_6_raw_snapshot" / "parent-import"
    authority_snapshot = (
        snapshot_root
        / "observed_p95"
        / PROFILE_ID
        / "observed_p95_authority_receipt.json"
    )
    projection_snapshot = (
        snapshot_root / "observed_p95" / PROFILE_ID / "projection_p95.json"
    )
    authority_snapshot.parent.mkdir(parents=True)
    source = _source_authority()
    reservations = {
        RUNTIME_MODEL: {
            "action": {
                "authority": deepcopy(source),
                "reservation": _reservation(
                    prompt_tokens=120.0,
                    completion_tokens=20.0,
                    sample_count=12,
                ),
            },
            "semantic": {
                "authority": deepcopy(source),
                "reservation": _reservation(
                    prompt_tokens=300.0,
                    completion_tokens=40.0,
                    sample_count=4,
                ),
            },
        }
    }
    source_receipt = authority._seal_v27(
        {
            "schema_version": ("finevo-pilot-v2.6-inherited-observed-p95-authority-v1"),
            "contract": {
                "contract_sha256": PILOT_CONTRACT_V2_6_CANONICAL_SHA256,
            },
            "git": {
                "tag": "pilot-v2.6-science",
                "commit": "0f59a15bc2cc3cce68f64de1dc1be78f7d74e214",
            },
            "model": {
                "model_id": PROFILE_ID,
                "runtime_model": RUNTIME_MODEL,
                "served_model": SERVED_MODEL,
            },
            "parent_source": {},
            "reservations": reservations,
            "scientific_evidence": False,
            "evidence_use": "fixture V2.6 authority",
        }
    )
    source_projection = authority._seal_v27(
        {
            "schema_version": OBSERVED_P95_PROJECTION_SCHEMA_VERSION,
            "model_id": PROFILE_ID,
            "served_model": SERVED_MODEL,
            "projection": {
                f"{SERVED_MODEL}::{call_kind}": deepcopy(
                    reservations[RUNTIME_MODEL][call_kind]["reservation"]
                )
                for call_kind in ("action", "semantic")
            },
            "bindings": {
                "contract_sha256": PILOT_CONTRACT_V2_6_CANONICAL_SHA256,
                "git_tag": "pilot-v2.6-science",
                "git_commit": ("0f59a15bc2cc3cce68f64de1dc1be78f7d74e214"),
                "source_kind": "v2.5-terminal-parent-import-v2.6",
                "source_authority_receipt": "/frozen/v2.6/authority.json",
                "source_authority_receipt_content_sha256": source_receipt["integrity"][
                    "content_sha256"
                ],
                "source_parent_manifest_content_sha256": "a" * 64,
            },
        }
    )
    _write_json(authority_snapshot, source_receipt)
    _write_json(projection_snapshot, source_projection)
    authority_raw = authority_snapshot.read_bytes()
    projection_raw = projection_snapshot.read_bytes()
    binding = {
        "source_contract_sha256": PILOT_CONTRACT_V2_6_CANONICAL_SHA256,
        "source_git_tag": "pilot-v2.6-science",
        "source_git_commit": ("0f59a15bc2cc3cce68f64de1dc1be78f7d74e214"),
        "model_id": PROFILE_ID,
        "runtime_model": RUNTIME_MODEL,
        "served_model": SERVED_MODEL,
        "authority": {
            "path": (
                "experiment_results/pilot-v2.6/raw/parent-import/"
                f"observed_p95/{PROFILE_ID}/"
                "observed_p95_authority_receipt.json"
            ),
            "schema_version": ("finevo-pilot-v2.6-inherited-observed-p95-authority-v1"),
            "file_sha256": hashlib.sha256(authority_raw).hexdigest(),
            "content_sha256": source_receipt["integrity"]["content_sha256"],
            "snapshot_path": authority_snapshot.relative_to(repo_root).as_posix(),
        },
        "projection": {
            "path": (
                "experiment_results/pilot-v2.6/raw/parent-import/"
                f"observed_p95/{PROFILE_ID}/projection_p95.json"
            ),
            "schema_version": OBSERVED_P95_PROJECTION_SCHEMA_VERSION,
            "file_sha256": hashlib.sha256(projection_raw).hexdigest(),
            "content_sha256": source_projection["integrity"]["content_sha256"],
            "snapshot_path": projection_snapshot.relative_to(repo_root).as_posix(),
        },
        "reservations": reservations,
    }
    monkeypatch.setattr(
        authority,
        "_V27_FROZEN_V26_P95_SOURCES",
        {
            PROFILE_ID: {
                kind: {
                    key: binding[kind][key]
                    for key in (
                        "path",
                        "schema_version",
                        "file_sha256",
                        "content_sha256",
                    )
                }
                for kind in ("authority", "projection")
            }
        },
    )
    return repo_root, contract, raw_root, binding


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_v27_builder_reseals_child_authority_and_projection(
    v27_authority_fixture: tuple[Path, Any, Path, dict[str, Any]],
) -> None:
    repo_root, contract, raw_root, binding = v27_authority_fixture
    built = authority.build_v27_resealed_observed_p95_authority(
        repo_root=repo_root,
        contract=contract,
        contract_path="experiments/pilot_v2_7.yaml",
        raw_root="experiment_results/pilot-v2.7/raw",
        profile_id=PROFILE_ID,
        expected_git_commit=EXPECTED_COMMIT,
        verified_v2_6_source_binding=binding,
    )

    receipt = built["receipt"]
    assert (
        receipt["schema_version"]
        == authority.V27_RESEALED_OBSERVED_P95_AUTHORITY_SCHEMA_VERSION
    )
    assert receipt["contract"]["contract_sha256"] == contract.canonical_hash
    assert receipt["git"] == {
        "tag": "pilot-v2.7-science",
        "commit": EXPECTED_COMMIT,
    }
    assert receipt["scientific_evidence"] is False
    assert (
        receipt["reservations"][RUNTIME_MODEL]["action"]["authority"][
            "pilot_contract_hash"
        ]
        == contract.canonical_hash
    )
    assert (
        receipt["reservations"][RUNTIME_MODEL]["action"]["authority"]["pilot_tag"]
        == "pilot-v2.7-science"
    )
    projection = built["projection"]
    assert projection["bindings"]["source_kind"] == "v2.6-terminal-stage0-import-v2.7"
    assert "pilot-v2.7/raw" in projection["bindings"]["source_authority_receipt"]
    assert "pilot-v2.6/raw" not in projection["bindings"]["source_authority_receipt"]


def test_v27_generic_reader_rebuilds_and_rejects_tamper(
    monkeypatch: pytest.MonkeyPatch,
    v27_authority_fixture: tuple[Path, Any, Path, dict[str, Any]],
) -> None:
    repo_root, contract, raw_root, binding = v27_authority_fixture
    monkeypatch.setattr(
        stage0_import,
        "v2_6_p95_source_binding",
        lambda **_: deepcopy(binding),
    )
    built = authority.build_v27_resealed_observed_p95_authority(
        repo_root=repo_root,
        contract=contract,
        contract_path="experiments/pilot_v2_7.yaml",
        raw_root="experiment_results/pilot-v2.7/raw",
        profile_id=PROFILE_ID,
        expected_git_commit=EXPECTED_COMMIT,
        verified_v2_6_source_binding=binding,
    )
    _write_json(built["receipt_path"], built["receipt"])
    _write_json(built["projection_path"], built["projection"])

    receipt_relative = built["receipt_path"].relative_to(repo_root).as_posix()
    verified = authority.verified_observed_p95_authority_binding(
        receipt_relative,
        repo_root=repo_root,
        expected_git_commit=EXPECTED_COMMIT,
    )
    assert set(verified["reservations"]) == {RUNTIME_MODEL}
    assert verified["git_commit"] == EXPECTED_COMMIT
    authority.verify_v27_resealed_observed_p95_projection(
        built["projection_path"],
        receipt_or_path=built["receipt_path"],
        repo_root=repo_root,
        expected_git_commit=EXPECTED_COMMIT,
    )

    for mutate in (
        lambda value: value["git"].update(tag="pilot-v2.6-science"),
        lambda value: value["contract"].update(contract_sha256="0" * 64),
        lambda value: value["parent_source"]["authority"].update(
            content_sha256="0" * 64
        ),
        lambda value: value["reservations"][RUNTIME_MODEL]["action"].update(
            reservation=_reservation(
                prompt_tokens=1.0,
                completion_tokens=20.0,
                sample_count=12,
            )
        ),
    ):
        tampered = deepcopy(built["receipt"])
        mutate(tampered)
        tampered = authority._seal_v27(tampered)
        with pytest.raises(authority.ObservedP95AuthorityError):
            authority.verify_observed_p95_authority_receipt(
                tampered,
                repo_root=repo_root,
                expected_git_commit=EXPECTED_COMMIT,
            )

    wrong_projection = deepcopy(built["projection"])
    wrong_projection["bindings"]["source_kind"] = "direct-v2.6-receipt"
    wrong_projection = authority._seal_v27(wrong_projection)
    with pytest.raises(authority.ObservedP95AuthorityError):
        authority.verify_v27_resealed_observed_p95_projection(
            wrong_projection,
            receipt_or_path=built["receipt"],
            repo_root=repo_root,
            expected_git_commit=EXPECTED_COMMIT,
        )


def test_v27_builder_rejects_fabricated_caps_and_snapshot_tamper(
    v27_authority_fixture: tuple[Path, Any, Path, dict[str, Any]],
) -> None:
    repo_root, contract, raw_root, binding = v27_authority_fixture
    fabricated = deepcopy(binding)
    fabricated["reservations"][RUNTIME_MODEL]["action"]["reservation"] = _reservation(
        prompt_tokens=1.0,
        completion_tokens=20.0,
        sample_count=12,
    )
    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="reservations are malformed",
    ):
        authority.build_v27_resealed_observed_p95_authority(
            repo_root=repo_root,
            contract=contract,
            contract_path="experiments/pilot_v2_7.yaml",
            raw_root=raw_root,
            profile_id=PROFILE_ID,
            expected_git_commit=EXPECTED_COMMIT,
            verified_v2_6_source_binding=fabricated,
        )

    snapshot = repo_root / binding["authority"]["snapshot_path"]
    snapshot.write_text('{"fabricated":true}\n', encoding="utf-8")
    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="snapshot hash or schema drifted",
    ):
        authority.build_v27_resealed_observed_p95_authority(
            repo_root=repo_root,
            contract=contract,
            contract_path="experiments/pilot_v2_7.yaml",
            raw_root=raw_root,
            profile_id=PROFILE_ID,
            expected_git_commit=EXPECTED_COMMIT,
            verified_v2_6_source_binding=binding,
        )


def test_v27_builder_rejects_stale_self_hash_with_updated_file_hash(
    v27_authority_fixture: tuple[Path, Any, Path, dict[str, Any]],
) -> None:
    repo_root, contract, raw_root, binding = v27_authority_fixture
    snapshot = repo_root / binding["authority"]["snapshot_path"]
    source_receipt = json.loads(snapshot.read_text(encoding="utf-8"))
    source_receipt["evidence_use"] = "fabricated while retaining stale integrity"
    _write_json(snapshot, source_receipt)
    fabricated = deepcopy(binding)
    fabricated["authority"]["file_sha256"] = hashlib.sha256(
        snapshot.read_bytes()
    ).hexdigest()

    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="snapshot hash or schema drifted",
    ):
        authority.build_v27_resealed_observed_p95_authority(
            repo_root=repo_root,
            contract=contract,
            contract_path="experiments/pilot_v2_7.yaml",
            raw_root=raw_root,
            profile_id=PROFILE_ID,
            expected_git_commit=EXPECTED_COMMIT,
            verified_v2_6_source_binding=fabricated,
        )


def test_v27_builder_rejects_symlinked_source_snapshot(
    v27_authority_fixture: tuple[Path, Any, Path, dict[str, Any]],
) -> None:
    repo_root, contract, raw_root, binding = v27_authority_fixture
    snapshot = repo_root / binding["authority"]["snapshot_path"]
    exact_copy = snapshot.with_name("exact-authority-copy.json")
    shutil.copyfile(snapshot, exact_copy)
    snapshot.unlink()
    snapshot.symlink_to(exact_copy.name)

    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="cannot be opened safely",
    ):
        authority.build_v27_resealed_observed_p95_authority(
            repo_root=repo_root,
            contract=contract,
            contract_path="experiments/pilot_v2_7.yaml",
            raw_root=raw_root,
            profile_id=PROFILE_ID,
            expected_git_commit=EXPECTED_COMMIT,
            verified_v2_6_source_binding=binding,
        )


def test_unknown_observed_p95_schema_remains_fail_closed(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    with pytest.raises(
        authority.ObservedP95AuthorityError,
        match="shape or schema drifted",
    ):
        authority.verify_observed_p95_authority_receipt(
            {"schema_version": "finevo-pilot-v2.8-unregistered"},
            repo_root=repo_root,
            expected_git_commit=EXPECTED_COMMIT,
        )
