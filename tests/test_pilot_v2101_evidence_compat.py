from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from verified_memory import pilot_evidence as evidence
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.pilot_contract import canonical_sha256, load_pilot_contract
from verified_memory.pilot_evidence import PilotEvidenceError


ROOT = Path(__file__).resolve().parents[1]
V29_CONTRACT = ROOT / "experiments" / "pilot_v2_9_overlay.yaml"
V210_CONTRACT = ROOT / "experiments" / "pilot_v2_10_overlay.yaml"
V2101_CONTRACT = ROOT / "experiments" / "pilot_v2_10_1_overlay.yaml"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _seal_bound(value: dict[str, Any]) -> dict[str, Any]:
    sealed = deepcopy(value)
    sealed.pop("integrity", None)
    sealed["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
    }
    sealed["integrity"]["content_sha256"] = canonical_sha256(sealed)
    return sealed


def _v210_wire_identity(contract: Any) -> tuple[str, str, str]:
    if contract.contract_id == orchestrator.V210_CONTRACT_ID:
        return (
            orchestrator.PILOT_V210_IMPORTED_QREF_SCHEMA_VERSION,
            orchestrator.PILOT_V210_IMPORTED_RUN_ENVELOPE_SCHEMA_VERSION,
            "immutable-v2.9-prerequisite-import-offline-reseal",
        )
    if contract.contract_id == orchestrator.V2101_CONTRACT_ID:
        return (
            orchestrator.PILOT_V2101_IMPORTED_QREF_SCHEMA_VERSION,
            orchestrator.PILOT_V2101_IMPORTED_RUN_ENVELOPE_SCHEMA_VERSION,
            "immutable-v2.9-prerequisite-import-offline-v2.10.1-reseal",
        )
    raise AssertionError("fixture requires a V2.10-family contract")


def _v210_qref_fixture(
    tmp_path: Path,
    contract_path: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = load_pilot_contract(contract_path)
    spec = contract.expand(stage="q-ref-resolution")[0]
    raw = tmp_path / contract.contract_id / "raw"
    source_path = (
        raw
        / "parent-import"
        / "v2_9_raw_snapshot"
        / "q-ref-resolution"
        / "q_ref_resolution.json"
    )
    _write_json(source_path, {"source": "immutable-v2.9-qref"})
    current_path = raw / "q-ref-resolution" / "q_ref_resolution.json"
    qref_schema, _, disposition = _v210_wire_identity(contract)
    verified = _seal_bound(
        {
            "schema_version": qref_schema,
            "status": "pass",
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "q_ref": 63.50397933257746,
            "row_count": 48,
            "provider_calls_current_attempt": 0,
            "hosted_provider_calls_current_attempt": 0,
            "local_model_calls_current_attempt": 0,
            "provider_construction_current_attempt": False,
            "scientific_evidence": False,
            "source_import": {
                "source_artifacts": {
                    "q_ref_resolution": {
                        "snapshot_path": str(source_path),
                        "file_sha256": _file_sha256(source_path),
                    }
                }
            },
        }
    )
    _write_json(current_path, verified)
    marker = {
        "q_ref": verified["q_ref"],
        "row_count": verified["row_count"],
        "source_resolution": str(source_path),
        "source_resolution_sha256": _file_sha256(source_path),
        "resolution_artifact": str(current_path),
    }
    payload = {
        "metrics": {"q_ref": verified["q_ref"]},
        "gate_evidence": {
            "status": "pass",
            "execution_disposition": disposition,
            "q_ref_resolution": {
                "path": str(current_path),
                "file_sha256": _file_sha256(current_path),
                "content_sha256": verified["integrity"]["content_sha256"],
            },
            "provider_calls_current_attempt": 0,
        },
        "q_ref_resolution": marker,
        "provider_calls": 0,
    }
    return contract, spec.to_dict(), payload, verified


@pytest.mark.parametrize(
    "contract_path",
    [V210_CONTRACT, V2101_CONTRACT],
)
def test_v210_family_qref_accepts_exact_source_resolution_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_path: Path,
) -> None:
    contract, spec, payload, verified = _v210_qref_fixture(
        tmp_path,
        contract_path,
    )
    observed: list[tuple[str, Path]] = []

    def replay(_contract, *, raw_root, paid, authority_repo_root):
        assert _contract is contract
        assert paid is None
        assert authority_repo_root is None
        observed.append((_contract.contract_id, raw_root))
        return deepcopy(verified)

    monkeypatch.setattr(orchestrator, "_load_verified_q_ref", replay)

    evidence._validate_terminal_payload_marker(
        contract,
        spec,
        payload,
        raw_root=tmp_path / contract.contract_id / "raw",
    )

    assert observed == [(contract.contract_id, tmp_path / contract.contract_id / "raw")]
    assert evidence._is_v210_prerequisite_family_contract(contract)
    assert evidence._is_imported_stage0_spec(
        contract,
        contract.expand(stage="stage0-calibration")[0].to_dict(),
    )
    assert evidence._stage_sets(contract) == (
        evidence.V24_NON_SCIENTIFIC_STAGES,
        evidence.V24_SCIENTIFIC_STAGES,
    )
    assert evidence._evidence_namespace(contract) == (
        f"current_v2/{contract.contract_id.removeprefix('finevo-')}"
    )


def test_v2101_qref_producer_roundtrips_evidence_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(V2101_CONTRACT)
    spec = contract.expand(stage="q-ref-resolution")[0]
    raw = tmp_path / "experiment_results" / "pilot-v2.10.1" / "raw"
    authority = tmp_path / "authority"
    paid = orchestrator.GitProvenance(
        git_tag=str(contract.implementation["required_git_tag"]),
        head_commit="1" * 40,
        tag_commit="1" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )
    source_path = (
        raw
        / "parent-import"
        / "v2_9_raw_snapshot"
        / "q-ref-resolution"
        / "q_ref_resolution.json"
    )
    _write_json(source_path, {"source": "immutable-v2.9-qref"})
    source_manifest = {
        "integrity": {
            "content_sha256": "2" * 64,
        }
    }
    source_manifest_path = authority.joinpath(
        *orchestrator.V2101_SOURCE_MANIFEST_PATH.parts
    )
    _write_json(source_manifest_path, source_manifest)
    parent_receipt = {
        "integrity": {
            "content_sha256": "3" * 64,
        }
    }
    parent_receipt_path = raw / "parent-import" / "parent_import_receipt.json"
    _write_json(parent_receipt_path, parent_receipt)
    source = {
        "source_stage_id": "q-ref-resolution",
        "source_run_id": "immutable-v2.9-qref",
        "target_spec": spec.to_dict(),
        "q_ref": 63.50397933257746,
        "q_ref_resolution": {
            "status": "pass",
            "q_ref": 63.50397933257746,
            "row_count": 48,
        },
        "source_artifacts": {
            "q_ref_resolution": {
                "snapshot_path": str(source_path),
                "file_sha256": _file_sha256(source_path),
                "content_sha256": "4" * 64,
            }
        },
        "source_release": {
            "contract_id": orchestrator.V29_CONTRACT_ID,
            "contract_sha256": "5" * 64,
            "raw_inventory_sha256": "6" * 64,
            "source_path_kind": ("byte-exact-v2.9-raw-inside-v2.10-terminal-snapshot"),
        },
        "source_terminal": {
            "file_sha256": "7" * 64,
            "content_sha256": "8" * 64,
        },
        "provider_construction_during_verification": False,
        "provider_calls_during_verification": 0,
        "treatment_effect_evidence": False,
    }
    monkeypatch.setattr(
        orchestrator,
        "_v2101_import_authority",
        lambda *_args, **_kwargs: (
            deepcopy(source_manifest),
            deepcopy(parent_receipt),
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "verified_v2101_imported_prerequisite_binding",
        lambda *_args, **_kwargs: deepcopy(source),
    )

    produced = orchestrator._expected_v2101_q_ref_resolution(
        contract,
        raw_root=raw,
        paid=paid,
        authority_repo_root=authority,
    )
    current_path = raw / "q-ref-resolution" / "q_ref_resolution.json"
    _write_json(current_path, produced)
    _, _, disposition = _v210_wire_identity(contract)
    payload = {
        "metrics": {"q_ref": produced["q_ref"]},
        "gate_evidence": {
            "status": "pass",
            "execution_disposition": disposition,
            "q_ref_resolution": {
                "path": str(current_path),
                "file_sha256": _file_sha256(current_path),
                "content_sha256": produced["integrity"]["content_sha256"],
            },
            "provider_calls_current_attempt": 0,
        },
        "q_ref_resolution": {
            "q_ref": produced["q_ref"],
            "row_count": produced["row_count"],
            "source_resolution": str(source_path),
            "source_resolution_sha256": _file_sha256(source_path),
            "resolution_artifact": str(current_path),
        },
        "provider_calls": 0,
    }
    monkeypatch.setattr(
        orchestrator,
        "_load_verified_q_ref",
        lambda *_args, **_kwargs: deepcopy(produced),
    )

    evidence._validate_terminal_payload_marker(
        contract,
        spec.to_dict(),
        payload,
        raw_root=raw,
        source_repo_root=authority,
    )


@pytest.mark.parametrize(
    ("contract_path", "other_contract_path"),
    [
        (V210_CONTRACT, V2101_CONTRACT),
        (V2101_CONTRACT, V210_CONTRACT),
    ],
)
def test_v210_family_qref_rejects_cross_version_schema_and_disposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_path: Path,
    other_contract_path: Path,
) -> None:
    contract, spec, payload, verified = _v210_qref_fixture(
        tmp_path,
        contract_path,
    )
    other = load_pilot_contract(other_contract_path)
    other_schema, _, other_disposition = _v210_wire_identity(other)
    raw = tmp_path / contract.contract_id / "raw"

    cross_disposition = deepcopy(payload)
    cross_disposition["gate_evidence"]["execution_disposition"] = other_disposition
    monkeypatch.setattr(
        orchestrator,
        "_load_verified_q_ref",
        lambda *_args, **_kwargs: deepcopy(verified),
    )
    with pytest.raises(PilotEvidenceError, match="marker shape"):
        evidence._validate_terminal_payload_marker(
            contract,
            spec,
            cross_disposition,
            raw_root=raw,
        )

    cross_schema = deepcopy(verified)
    cross_schema["schema_version"] = other_schema
    monkeypatch.setattr(
        orchestrator,
        "_load_verified_q_ref",
        lambda *_args, **_kwargs: deepcopy(cross_schema),
    )
    with pytest.raises(PilotEvidenceError, match="exact resealed resolution"):
        evidence._validate_terminal_payload_marker(
            contract,
            spec,
            payload,
            raw_root=raw,
        )


def test_v210_qref_rejects_mixed_legacy_shape_and_hash_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, spec, payload, verified = _v210_qref_fixture(
        tmp_path,
        V210_CONTRACT,
    )
    monkeypatch.setattr(
        orchestrator,
        "_load_verified_q_ref",
        lambda *_args, **_kwargs: deepcopy(verified),
    )
    raw = tmp_path / contract.contract_id / "raw"

    mixed = deepcopy(payload)
    mixed["q_ref_resolution"]["source_manifest"] = "legacy/manifest.json"
    mixed["q_ref_resolution"]["source_manifest_sha256"] = "a" * 64
    with pytest.raises(PilotEvidenceError, match="marker shape"):
        evidence._validate_terminal_payload_marker(
            contract,
            spec,
            mixed,
            raw_root=raw,
        )

    tampered = deepcopy(payload)
    tampered["q_ref_resolution"]["source_resolution_sha256"] = "b" * 64
    with pytest.raises(PilotEvidenceError, match="file bindings"):
        evidence._validate_terminal_payload_marker(
            contract,
            spec,
            tampered,
            raw_root=raw,
        )


def test_v29_qref_keeps_legacy_marker_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(V29_CONTRACT)
    spec = contract.expand(stage="q-ref-resolution")[0].to_dict()
    legacy = {
        "metrics": {},
        "gate_evidence": {"go": True},
        "q_ref_resolution": {
            "q_ref": 63.50397933257746,
            "row_count": 48,
            "source_manifest": "q-ref-resolution/runs/source/manifest.json",
            "source_manifest_sha256": "a" * 64,
            "resolution_artifact": ("q-ref-resolution/q_ref_resolution.json"),
        },
    }
    monkeypatch.setattr(
        evidence,
        "_validate_v29_qref_resolution_artifact",
        lambda *_args, **_kwargs: None,
    )

    evidence._validate_terminal_payload_marker(
        contract,
        spec,
        legacy,
        raw_root=tmp_path,
    )

    current_shape = deepcopy(legacy)
    marker = current_shape["q_ref_resolution"]
    marker["source_resolution"] = marker.pop("source_manifest")
    marker["source_resolution_sha256"] = marker.pop("source_manifest_sha256")
    with pytest.raises(PilotEvidenceError, match="sealed positive"):
        evidence._validate_terminal_payload_marker(
            contract,
            spec,
            current_shape,
            raw_root=tmp_path,
        )


def _stage0_source_fixture(
    tmp_path: Path,
    contract_path: Path,
    *,
    legacy: bool,
) -> tuple[Any, Any, Path, dict[str, Any], dict[str, Any]]:
    contract = load_pilot_contract(contract_path)
    spec = contract.expand(stage="stage0-calibration")[0]
    raw = tmp_path / contract.contract_id / "raw"
    source_run_id = f"immutable-source--{spec.run_id}"
    source_hash = "d" * 64
    source_import = {"source_run_id": source_run_id}
    if legacy:
        source_import["source_artifacts"] = {"manifest": {"file_sha256": source_hash}}
        lineage_key = "source_manifest_sha256"
        disposition = "immutable-parent-import-offline-resummary"
        envelope_schema = "finevo-pilot-v2.9-imported-run-envelope-v1"
    else:
        source_import["source_terminal"] = {"file_sha256": source_hash}
        lineage_key = "source_terminal_file_sha256"
        _, envelope_schema, disposition = _v210_wire_identity(contract)
    metrics = {
        "schema_version": "finevo-pilot-stage0-analysis-v1",
        "row_counts": {"actions": 48, "utility_ledger": 48},
    }
    envelope_path = (
        raw / "stage0-calibration" / "imports" / spec.run_id / "envelope.json"
    )
    envelope = _seal_bound(
        {
            "schema_version": envelope_schema,
            "execution_disposition": disposition,
            "source_import": source_import,
            "reader": {"summary": metrics},
        }
    )
    _write_json(envelope_path, envelope)
    envelope_binding = {
        "path": str(envelope_path),
        "file_sha256": _file_sha256(envelope_path),
        "content_sha256": envelope["integrity"]["content_sha256"],
        "schema_version": envelope_schema,
    }
    terminal_path = raw / "stage0-calibration" / "summaries" / f"{spec.run_id}.json"
    gate = {
        "status": "pass",
        "execution_disposition": disposition,
        "provider_calls_current_attempt": 0,
        "imported_run_envelope": envelope_binding,
    }
    terminal = _seal_bound(
        {
            "payload": {
                "metrics": metrics,
                "gate_evidence": gate,
            }
        }
    )
    _write_json(terminal_path, terminal)
    source = {
        "run_id": spec.run_id,
        "utility_profile_id": spec.utility_profile_id,
        "environment_seed": spec.environment_seed,
        "execution_disposition": disposition,
        "envelope": str(envelope_path),
        "envelope_file_sha256": _file_sha256(envelope_path),
        "envelope_content_sha256": envelope["integrity"]["content_sha256"],
        "terminal_summary": str(terminal_path),
        "terminal_summary_file_sha256": _file_sha256(terminal_path),
        "terminal_summary_content_sha256": terminal["integrity"]["content_sha256"],
        "source_run_id": source_run_id,
        lineage_key: source_hash,
        "provider_calls_current_attempt": 0,
    }
    row = {
        "status": "complete",
        "artifact_kind": "imported-stage0-run-envelope",
        "artifact_sha256": _file_sha256(terminal_path),
        "metrics": metrics,
        "gate_evidence": gate,
    }
    return contract, spec, raw, source, row


@pytest.mark.parametrize(
    ("contract_path", "legacy"),
    [
        (V29_CONTRACT, True),
        (V210_CONTRACT, False),
        (V2101_CONTRACT, False),
    ],
)
def test_imported_stage0_source_row_dispatches_exact_contract_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_path: Path,
    legacy: bool,
) -> None:
    contract, spec, raw, source, row = _stage0_source_fixture(
        tmp_path,
        contract_path,
        legacy=legacy,
    )
    gate = row["gate_evidence"]

    monkeypatch.setattr(
        orchestrator,
        "verify_v27_imported_stage0_terminal",
        lambda *_args, **_kwargs: {
            "execution_disposition": source["execution_disposition"],
            "provider_calls_current_attempt": 0,
            "scientific_evidence": True,
            "envelope_binding": gate["imported_run_envelope"],
            "metrics": row["metrics"],
        },
    )

    assert evidence._validated_imported_stage0_source_row(
        contract,
        raw_root=raw,
        source=source,
        row=row,
        spec=spec,
        common_commit="1" * 40,
    )


def test_v210_stage0_rejects_mixed_legacy_and_terminal_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, spec, raw, source, row = _stage0_source_fixture(
        tmp_path,
        V210_CONTRACT,
        legacy=False,
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v27_imported_stage0_terminal",
        lambda *_args, **_kwargs: {
            "execution_disposition": source["execution_disposition"],
            "provider_calls_current_attempt": 0,
            "scientific_evidence": True,
            "envelope_binding": row["gate_evidence"]["imported_run_envelope"],
            "metrics": row["metrics"],
        },
    )

    mixed = deepcopy(source)
    mixed["source_manifest_sha256"] = "e" * 64
    assert not evidence._validated_imported_stage0_source_row(
        contract,
        raw_root=raw,
        source=mixed,
        row=row,
        spec=spec,
        common_commit="1" * 40,
    )

    tampered = deepcopy(source)
    tampered["source_terminal_file_sha256"] = "e" * 64
    assert not evidence._validated_imported_stage0_source_row(
        contract,
        raw_root=raw,
        source=tampered,
        row=row,
        spec=spec,
        common_commit="1" * 40,
    )


@pytest.mark.parametrize(
    ("contract_path", "other_contract_path"),
    [
        (V210_CONTRACT, V2101_CONTRACT),
        (V2101_CONTRACT, V210_CONTRACT),
    ],
)
def test_v210_family_stage0_rejects_cross_version_disposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_path: Path,
    other_contract_path: Path,
) -> None:
    contract, spec, raw, source, row = _stage0_source_fixture(
        tmp_path,
        contract_path,
        legacy=False,
    )
    other = load_pilot_contract(other_contract_path)
    _, _, other_disposition = _v210_wire_identity(other)
    cross_source = deepcopy(source)
    cross_source["execution_disposition"] = other_disposition
    monkeypatch.setattr(
        orchestrator,
        "verify_v27_imported_stage0_terminal",
        lambda *_args, **_kwargs: {
            "execution_disposition": other_disposition,
            "provider_calls_current_attempt": 0,
            "scientific_evidence": True,
            "envelope_binding": row["gate_evidence"]["imported_run_envelope"],
            "metrics": row["metrics"],
        },
    )

    assert not evidence._validated_imported_stage0_source_row(
        contract,
        raw_root=raw,
        source=cross_source,
        row=row,
        spec=spec,
        common_commit="1" * 40,
    )
