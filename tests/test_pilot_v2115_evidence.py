from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_evidence import PilotEvidenceError, _stage_sets
from verified_memory import pilot_orchestrator
from verified_memory.pilot_orchestrator import PilotRunLedger
from verified_memory import pilot_v2115_evidence as evidence


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_5.yaml"


def _contract():
    return load_pilot_contract(CONTRACT_PATH)


@pytest.mark.parametrize("version", ("3", "4", "5"))
def test_v2113_through_v2115_use_exact_136_cell_stage_partition(
    version: str,
) -> None:
    contract = load_pilot_contract(ROOT / "experiments" / f"pilot_v2_11_{version}.yaml")
    non_scientific, scientific = _stage_sets(contract)
    assert non_scientific == {
        "parent-import",
        "capability-gate",
        "long-context-preflight",
    }
    assert scientific == {
        "experiment-a",
        "experiment-b",
        "experiment-c",
        "experiment-d",
        "cross-model",
    }
    assert len(contract.expand()) == 136


def test_future_v211_contract_is_not_implicitly_admitted() -> None:
    future = replace(_contract(), contract_id="finevo-pilot-v2.11.6")
    with pytest.raises(PilotEvidenceError, match="stage partition differs"):
        _stage_sets(future)


def test_v2115_frozen_contract_requires_exact_stage_order_and_denominator() -> None:
    contract = _contract()
    evidence._frozen_contract(contract)
    assert tuple(contract.stage_ids) == evidence._EXPECTED_STAGE_IDS
    assert {
        stage_id: len(contract.expand(stage=stage_id))
        for stage_id in contract.stage_ids
    } == evidence._EXPECTED_STAGE_COUNTS


def test_v2115_nonterminal_136_cell_ledger_is_not_publishable(
    tmp_path: Path,
) -> None:
    contract = _contract()
    raw = tmp_path / "raw"
    ledger_path = raw / "run_ledger.json"
    ledger = PilotRunLedger(
        ledger_path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())
    value = json.loads(ledger_path.read_text(encoding="utf-8"))
    with pytest.raises(PilotEvidenceError, match="all-terminal ITT denominator"):
        evidence._normalize_v2112_ledger(
            contract,
            value,
            raw_root=raw,
            expected_commit=evidence.V2115_SOURCE_COMMIT,
        )


def test_v2115_stage_receipt_adapter_replays_all_eight_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    calls: list[dict[str, Any]] = []

    def replay(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "schema_version": "old",
            "contract_sha256": contract.canonical_hash,
            "all_terminal": True,
            "receipts": {
                stage_id: {"status": "complete"} for stage_id in contract.stage_ids
            },
        }

    monkeypatch.setattr(evidence, "_validate_stage_receipts", replay)
    ledger = object()
    paid = object()
    result = evidence._normalized_stage_receipts(
        contract,
        raw_root=tmp_path / "raw",
        ledger=ledger,
        paid=paid,
        source_repo_root=tmp_path / "science",
    )

    assert result["schema_version"] == evidence.V2115_STAGE_RECEIPTS_SCHEMA_VERSION
    assert tuple(result["receipts"]) == tuple(contract.stage_ids)
    assert calls == [
        {
            "raw_root": tmp_path / "raw",
            "ledger": ledger,
            "paid": paid,
            "authority_repo_root": tmp_path / "science",
        }
    ]


def _source_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "science"
    contract = source / evidence.V2115_CONTRACT_RELATIVE
    manifest = source / evidence.V2115_SOURCE_MANIFEST_RELATIVE
    raw = source / evidence.V2115_RAW_RELATIVE
    ledger = raw / "run_ledger.json"
    contract.parent.mkdir(parents=True)
    raw.mkdir(parents=True)
    contract.write_text("{}", encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")
    ledger.write_text("{}", encoding="utf-8")
    return source, contract, raw, ledger


def test_v2115_source_paths_require_exact_in_place_namespace(tmp_path: Path) -> None:
    source, contract, raw, ledger = _source_tree(tmp_path)
    assert evidence._resolve_source_paths(
        source_repo_root=source,
        contract_path=contract,
        raw_root=raw,
        run_ledger_path=ledger,
    ) == tuple(path.resolve() for path in (source, contract, raw, ledger))

    outside = tmp_path / "outside.yaml"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(PilotEvidenceError, match="exact in-place science"):
        evidence._resolve_source_paths(
            source_repo_root=source,
            contract_path=outside,
            raw_root=raw,
            run_ledger_path=ledger,
        )


def test_v2115_source_paths_reject_symlinked_manifest(tmp_path: Path) -> None:
    source, contract, raw, ledger = _source_tree(tmp_path)
    manifest = source / evidence.V2115_SOURCE_MANIFEST_RELATIVE
    manifest.unlink()
    outside = tmp_path / "outside-manifest.json"
    outside.write_text("{}", encoding="utf-8")
    manifest.symlink_to(outside)

    with pytest.raises(PilotEvidenceError, match="crosses a symlink"):
        evidence._resolve_source_paths(
            source_repo_root=source,
            contract_path=contract,
            raw_root=raw,
            run_ledger_path=ledger,
        )


def _git_values(root: Path) -> dict[tuple[str, ...], str]:
    values = {
        ("rev-parse", "--show-toplevel"): str(root.resolve()),
        ("rev-parse", "HEAD"): evidence.V2115_SOURCE_COMMIT,
        (
            "cat-file",
            "-t",
            f"refs/tags/{evidence.V2115_SOURCE_TAG}",
        ): "tag",
        (
            "rev-parse",
            f"refs/tags/{evidence.V2115_SOURCE_TAG}^{{object}}",
        ): "1"
        * 40,
        (
            "rev-parse",
            f"refs/tags/{evidence.V2115_SOURCE_TAG}^{{commit}}",
        ): evidence.V2115_SOURCE_COMMIT,
        ("rev-parse", "--abbrev-ref", "HEAD"): "HEAD",
        ("status", "--porcelain", "--untracked-files=no"): "",
    }
    for index, relative in enumerate(evidence._SOURCE_REQUIRED_TRACKED_FILES):
        values[("ls-files", "--error-unmatch", "--", relative)] = relative
        values[("diff", "--quiet", "HEAD", "--", relative)] = ""
        values[("rev-parse", f"HEAD:{relative}")] = f"{index + 10:040x}"
    return values


def test_v2115_source_git_requires_detached_clean_annotated_exact_tag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _git_values(tmp_path)
    monkeypatch.setattr(
        evidence,
        "_git",
        lambda _root, *args: values[tuple(args)],
    )
    result = evidence._validate_source_git(
        tmp_path,
        _contract(),
        expected_commit=evidence.V2115_SOURCE_COMMIT,
    )
    assert result["detached_head"] is True
    assert result["tracked_worktree_clean"] is True


@pytest.mark.parametrize(
    ("key", "bad"),
    (
        (("rev-parse", "HEAD"), "2" * 40),
        (("status", "--porcelain", "--untracked-files=no"), " M run_pilot.py"),
        (
            (
                "rev-parse",
                f"refs/tags/{evidence.V2115_SOURCE_TAG}^{{commit}}",
            ),
            "3" * 40,
        ),
        (("rev-parse", "--abbrev-ref", "HEAD"), "main"),
    ),
)
def test_v2115_source_git_rejects_commit_dirty_tag_or_branch_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: tuple[str, ...],
    bad: str,
) -> None:
    values = _git_values(tmp_path)
    values[key] = bad
    monkeypatch.setattr(
        evidence,
        "_git",
        lambda _root, *args: values[tuple(args)],
    )
    with pytest.raises(PilotEvidenceError, match="exact detached"):
        evidence._validate_source_git(
            tmp_path,
            _contract(),
            expected_commit=evidence.V2115_SOURCE_COMMIT,
        )


def test_v2115_source_rejects_untracked_nested_directory_git_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent-repository"
    source_alias = parent / "untracked" / "science"
    source_alias.mkdir(parents=True)
    values = _git_values(parent)
    monkeypatch.setattr(
        evidence,
        "_git",
        lambda _root, *args: values[tuple(args)],
    )

    with pytest.raises(PilotEvidenceError, match="exact git repository top-level"):
        evidence._validate_source_git(
            source_alias,
            _contract(),
            expected_commit=evidence.V2115_SOURCE_COMMIT,
        )


def test_v2115_source_requires_contract_and_control_files_tracked_at_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _git_values(tmp_path)
    relative = evidence.V2115_CONTRACT_RELATIVE.as_posix()
    values[("ls-files", "--error-unmatch", "--", relative)] = ""
    monkeypatch.setattr(
        evidence,
        "_git",
        lambda _root, *args: values[tuple(args)],
    )
    with pytest.raises(PilotEvidenceError, match="not tracked exactly"):
        evidence._validate_source_git(
            tmp_path,
            _contract(),
            expected_commit=evidence.V2115_SOURCE_COMMIT,
        )


def test_required_provenance_file_must_match_its_head_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "verified_memory/pilot_v2115_evidence.py"

    def fake_git(_root: Path, *args: str) -> str:
        if args[0] == "ls-files":
            return relative
        if args[0] == "diff":
            raise PilotEvidenceError("working file differs from HEAD")
        raise AssertionError(args)

    monkeypatch.setattr(evidence, "_git", fake_git)
    with pytest.raises(PilotEvidenceError, match="differs from HEAD"):
        evidence._tracked_head_blobs(tmp_path, (relative,))


def test_v2115_publisher_is_a_clean_committed_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher_commit = "5" * 40
    values = {
        ("rev-parse", "--show-toplevel"): str(tmp_path.resolve()),
        ("rev-parse", "HEAD"): publisher_commit,
        ("status", "--porcelain"): "",
        (
            "merge-base",
            "--is-ancestor",
            evidence.V2115_SOURCE_COMMIT,
            publisher_commit,
        ): "",
        ("rev-parse", "--abbrev-ref", "HEAD"): (
            "codex/pilot-v2-11-5-evidence-consumer"
        ),
    }
    for index, relative in enumerate(evidence._PUBLISHER_REQUIRED_TRACKED_FILES):
        values[("ls-files", "--error-unmatch", "--", relative)] = relative
        values[("diff", "--quiet", "HEAD", "--", relative)] = ""
        values[("rev-parse", f"HEAD:{relative}")] = f"{index + 30:040x}"
    monkeypatch.setattr(
        evidence,
        "_git",
        lambda _root, *args: values[tuple(args)],
    )
    result = evidence._publisher_provenance(tmp_path)
    assert result["git_commit"] == publisher_commit
    assert result["tracked_worktree_clean"] is True
    assert result["provider_calls"] == 0


def test_v2115_publisher_rejects_untracked_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        ("rev-parse", "--show-toplevel"): str(tmp_path.resolve()),
        ("rev-parse", "HEAD"): "5" * 40,
        ("status", "--porcelain"): ("?? verified_memory/pilot_v2115_evidence.py"),
    }
    monkeypatch.setattr(
        evidence,
        "_git",
        lambda _root, *args: values[tuple(args)],
    )
    with pytest.raises(PilotEvidenceError, match="exact clean repository root"):
        evidence._publisher_provenance(tmp_path)


def _capability_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id in ("gpt52_main", "gpt56_diagnostic"):
        rows.append(
            {
                "stage_id": "capability-gate",
                "model_id": model_id,
                "status": "complete",
                "artifact_kind": "terminal-summary",
                "capability": {
                    "capability": {
                        "model_id": model_id,
                        "capability_pass": True,
                        "interface_pass": True,
                        "category_totals": {},
                    },
                    "provider_calls_current_attempt": 0,
                    "provider_construction_current_attempt": False,
                    "imported_effect_cells": 0,
                    "scientific_evidence": False,
                },
                "gate_evidence": {
                    "go": True,
                    "capability_pass": True,
                    "interface_pass": True,
                    "provider_calls_current_attempt": 0,
                    "provider_construction_current_attempt": False,
                },
            }
        )
        rows.append(
            {
                "stage_id": "long-context-preflight",
                "model_id": model_id,
                "status": "complete",
                "artifact_kind": "terminal-summary",
                "gate_evidence": {
                    "go": True,
                    "capability_pass": True,
                    "interface_pass": True,
                    "provider_calls_current_attempt": 0,
                    "provider_construction_current_attempt": False,
                },
            }
        )
    return rows


def test_v2115_imported_capability_and_preflight_are_combined_without_fresh_calls() -> (
    None
):
    result = evidence._v2115_capability_by_model(_capability_rows(), _contract())
    for model_id in ("gpt52_main", "gpt56_diagnostic"):
        row = result[model_id]
        assert row["ledger_status"] == "complete"
        assert row["artifact_validated"] is True
        assert row["registered_dispatch_cells"] == 2
        assert row["capability"]["preflight_go"] is True
        assert row["capability"]["provider_calls_current_attempt"] == 0
        assert row["capability"]["scientific_evidence"] is False


def test_v2115_imported_capability_rejects_current_attempt_calls() -> None:
    rows = _capability_rows()
    rows[0]["capability"]["provider_calls_current_attempt"] = 1
    with pytest.raises(PilotEvidenceError, match="imported capability/preflight"):
        evidence._v2115_capability_by_model(rows, _contract())


def _sealed_preflight_gate(
    contract: Any,
    spec: Any,
    authority: Path,
    projection: Path,
) -> dict[str, Any]:
    value = {
        "schema_version": evidence.V2115_PREFLIGHT_IMPORT_GATE_SCHEMA_VERSION,
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "model_id": spec.model_id,
        "capability_pass": True,
        "interface_pass": True,
        "go": True,
        "historical_action_samples": 24,
        "historical_semantic_samples": 8,
        "historical_provider_calls": 32,
        "provider_construction_current_attempt": False,
        "provider_calls_current_attempt": 0,
        "scientific_evidence": False,
        "authority_receipt": str(authority),
        "projection_p95": str(projection),
        "integrity": {"canonicalization": "json-sort-keys-utf8-v1"},
    }
    value["integrity"]["content_sha256"] = evidence.canonical_sha256(value)
    return value


def test_v2115_post_gate_normalizes_two_zero_call_authority_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    raw = tmp_path / "experiment_results/pilot-v2.11.5/raw"
    post_gate = raw / "long-context-preflight/post_gate_authority.json"
    post_gate.parent.mkdir(parents=True)
    post_gate.write_text("{}", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for spec in contract.expand(stage="long-context-preflight"):
        imported = raw / "long-context-preflight/imported_observed_p95" / spec.model_id
        authority = imported / "observed_p95_authority_receipt.json"
        projection = imported / "projection_p95.json"
        imported.mkdir(parents=True)
        authority.write_text("{}", encoding="utf-8")
        projection.write_text("{}", encoding="utf-8")
        rows.append(
            {
                **spec.to_dict(),
                "status": "complete",
                "artifact_kind": "terminal-summary",
                "gate_evidence": _sealed_preflight_gate(
                    contract, spec, authority, projection
                ),
            }
        )
    receipt = {
        "receipt_sha256": "6" * 64,
        "go": True,
        "denominator": {},
        "authority_sources": {},
        "reservations": {},
        "provider_boundary": {},
    }
    monkeypatch.setattr(
        evidence,
        "verified_v2115_gate_authority_binding",
        lambda *_args, **_kwargs: {"git_commit": evidence.V2115_SOURCE_COMMIT},
    )
    monkeypatch.setattr(
        evidence,
        "verify_v2115_gate_receipt",
        lambda *_args, **_kwargs: receipt,
    )

    result = evidence._validated_post_gate(
        contract,
        source_repo_root=tmp_path,
        raw_root=raw,
        commit=evidence.V2115_SOURCE_COMMIT,
        rows=rows,
    )

    assert result["go"] is True
    assert result["provider_calls_current_attempt"] == 0
    assert set(result["operational_imports"]) == {
        "gpt52_main",
        "gpt56_diagnostic",
    }
    assert all(
        row["scientific_evidence"] is False
        for row in result["operational_imports"].values()
    )


def test_v2115_c_sensitivity_replay_receives_external_source_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    rows = [
        {
            **spec.to_dict(),
            "status": "complete",
            "scientific_eligible": True,
        }
        for spec in contract.expand(stage="experiment-c")
    ]
    calls: list[dict[str, Any]] = []

    def sentinel(*_args: Any, **kwargs: Any) -> None:
        calls.append(kwargs)
        raise RuntimeError("sentinel after authority-root capture")

    monkeypatch.setattr(
        pilot_orchestrator,
        "_load_verified_experiment_c_sensitivity",
        sentinel,
    )
    source_root = tmp_path / "science"
    with pytest.raises(PilotEvidenceError, match="sentinel"):
        evidence._validated_experiment_c_sensitivity(
            contract,
            raw_root=tmp_path / "raw",
            rows=rows,
            common_commit=evidence.V2115_SOURCE_COMMIT,
            source_repo_root=source_root,
        )
    assert calls == [
        {
            "raw_root": tmp_path / "raw",
            "paid": None,
            "authority_repo_root": source_root,
        }
    ]


def _offline_fixture(
    tmp_path: Path,
) -> tuple[Any, Path, list[dict[str, Any]], dict[str, Any], list[Path]]:
    contract = _contract()
    raw = tmp_path / "raw"
    rows: list[dict[str, Any]] = []
    ledger: dict[str, Any] = {"runs": {}}
    details: list[Path] = []
    for spec in contract.expand(stage="experiment-c"):
        if spec.execution_mode != "offline_candidate_admission":
            continue
        payload = evidence._expected_offline_payload(contract, spec)
        detail = (
            raw
            / spec.stage_id
            / "runs"
            / spec.run_id
            / "offline_candidate_admission.json"
        )
        detail.parent.mkdir(parents=True)
        detail.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        summary = raw / spec.stage_id / "summaries" / f"{spec.run_id}.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(
            json.dumps(
                {
                    "payload": {
                        "metrics": {"rule_reliability": payload["check"]},
                        "gate_evidence": payload["check"],
                        "offline_source": str(detail.resolve()),
                    }
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        ledger["runs"][spec.run_id] = {"artifact": str(summary)}
        rows.append(
            {
                **spec.to_dict(),
                "status": "complete",
                "artifact_kind": "terminal-summary",
                "scientific_eligible": True,
                "metrics": {"rule_reliability": payload["check"]},
                "gate_evidence": payload["check"],
            }
        )
        details.append(detail)
    return contract, raw, rows, ledger, details


def test_v2115_offline_candidate_details_replay_and_seal_five_of_five(
    tmp_path: Path,
) -> None:
    contract, raw, rows, ledger, _details = _offline_fixture(tmp_path)
    audit, copies = evidence._validate_offline_candidate_admission(
        contract,
        raw_root=raw,
        rows=rows,
        ledger=ledger,
    )
    assert audit["pass"] is True
    assert audit["registered_cell_count"] == 5
    assert audit["publication_admitted_detail_count"] == 5
    assert audit["execution_time_full_detail_receipt_bound"] is False
    assert audit["publication_time_deterministic_revalidation"] is True
    assert len(copies) == 5


@pytest.mark.parametrize("failure", ("missing", "tamper", "symlink"))
def test_v2115_offline_candidate_details_fail_closed(
    tmp_path: Path,
    failure: str,
) -> None:
    contract, raw, rows, ledger, details = _offline_fixture(tmp_path)
    detail = details[0]
    if failure == "missing":
        detail.unlink()
    elif failure == "tamper":
        payload = json.loads(detail.read_text(encoding="utf-8"))
        payload["candidate"]["rationale"] = "tampered"
        detail.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    else:
        payload = detail.read_text(encoding="utf-8")
        detail.unlink()
        outside = tmp_path / "outside-offline.json"
        outside.write_text(payload, encoding="utf-8")
        detail.symlink_to(outside)

    with pytest.raises(PilotEvidenceError):
        evidence._validate_offline_candidate_admission(
            contract,
            raw_root=raw,
            rows=rows,
            ledger=ledger,
        )


def _no_go_gate(name: str) -> dict[str, Any]:
    value = {"status": "no-go", "claim_action": f"do not claim {name}"}
    if name == "narrative":
        value["claim_boundary"] = "do not claim semantic response"
    return value


def test_v2115_package_checksums_replay_and_existing_target_is_not_overwritten(
    tmp_path: Path,
) -> None:
    contract = _contract()
    rows = [
        {
            **spec.to_dict(),
            "status": "failed",
            "failure": {"error_type": "fixture-no-go"},
            "artifact_kind": None,
            "artifact_sha256": None,
            "scientific_eligible": False,
            "metrics": {},
            "gate_evidence": {},
            "capability": {},
            "narrative": {},
        }
        for spec in contract.expand()
    ]
    denominator = {
        "expected_count": 136,
        "observed_ledger_count": 136,
        "all_rows_present": True,
        "all_rows_terminal": True,
        "status_counts": {"failed": 136},
        "stage_status_counts": {},
        "all_completed_artifacts_validated": True,
        "itt_failures_retained": 136,
        "pass": True,
    }
    gates = {
        name: _no_go_gate(name)
        for name in ("experiment_a", "experiment_c", "experiment_d", "narrative")
    }
    release_controls = {
        "resolved_git_commit": evidence.V2115_SOURCE_COMMIT,
        "science_source": {"git_tag": evidence.V2115_SOURCE_TAG},
        "publisher": {"git_commit": "4" * 40},
        "stage_receipts": {},
        "run_ledger": {},
        "post_gate": {"available": False, "go": False},
        "budget": {"pass": True, "raw_root_storage_bytes": 0},
        "historical_import_boundary": {"scientific_evidence": False},
    }
    offline_audit = {
        "schema_version": evidence.V2115_OFFLINE_ADMISSION_AUDIT_SCHEMA_VERSION,
        "pass": False,
        "rows": [],
    }
    target = tmp_path / "package"
    manifest, checksums, scientific_complete = evidence._write_package(
        target,
        contract_path=CONTRACT_PATH,
        contract=contract,
        rows=rows,
        denominator=denominator,
        gates=gates,
        capability={},
        cross_model={},
        release_controls=release_controls,
        experiment_b={},
        rule_sensitivity=None,
        offline_audit=offline_audit,
        offline_payloads={},
        run_ledger={},
        budget_ledger={},
    )
    assert manifest.is_file()
    assert scientific_complete is False
    checksum_payload = json.loads(checksums.read_text(encoding="utf-8"))
    for row in checksum_payload["files"]:
        assert evidence._sha256_file(target / row["path"]) == row["sha256"]

    evidence_root = tmp_path / "evidence"
    package_target = evidence._new_package_target(evidence_root, contract)
    package_target.mkdir(parents=True)
    marker = package_target / "user-owned.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(PilotEvidenceError, match="refusing to overwrite"):
        evidence._new_package_target(evidence_root, contract)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_v2115_install_does_not_replace_target_created_after_initial_check(
    tmp_path: Path,
) -> None:
    temporary = tmp_path / ".package-build"
    target = tmp_path / "pilot-v2.11.5"
    temporary.mkdir()
    payload = temporary / "aggregate.json"
    payload.write_text("complete build", encoding="utf-8")
    # Simulate another process creating the target after _new_package_target.
    target.mkdir()
    marker = target / "other-process.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(PilotEvidenceError, match="refusing to overwrite"):
        evidence._install_package_no_overwrite(temporary, target)

    assert marker.read_text(encoding="utf-8") == "preserve"
    assert payload.read_text(encoding="utf-8") == "complete build"


def test_v2115_complete_matrix_with_negative_claim_gates_still_publishes_no_go(
    tmp_path: Path,
) -> None:
    contract = _contract()
    rows = [
        {
            **spec.to_dict(),
            "status": "complete",
            "failure": None,
            "artifact_kind": "terminal-summary",
            "artifact_sha256": "7" * 64,
            "scientific_eligible": spec.stage_id in evidence.V211_SCIENTIFIC_STAGES,
            "metrics": {},
            "gate_evidence": {},
            "capability": {},
            "narrative": {},
        }
        for spec in contract.expand()
    ]
    denominator = {
        "expected_count": 136,
        "observed_ledger_count": 136,
        "all_rows_present": True,
        "all_rows_terminal": True,
        "status_counts": {"complete": 136},
        "stage_status_counts": {},
        "all_completed_artifacts_validated": True,
        "itt_failures_retained": 0,
        "pass": True,
    }
    gates = {
        name: _no_go_gate(name)
        for name in ("experiment_a", "experiment_c", "experiment_d", "narrative")
    }
    post_gate_source = tmp_path / "post_gate_authority.json"
    post_gate_source.write_text("{}", encoding="utf-8")
    release_controls = {
        "resolved_git_commit": evidence.V2115_SOURCE_COMMIT,
        "science_source": {"git_tag": evidence.V2115_SOURCE_TAG},
        "publisher": {"git_commit": "8" * 40},
        "stage_receipts": {},
        "run_ledger": {},
        "post_gate": {
            "available": True,
            "go": True,
            "path": str(post_gate_source),
        },
        "budget": {"pass": True, "raw_root_storage_bytes": 0},
        "historical_import_boundary": {"scientific_evidence": False},
    }
    offline_audit = {
        "schema_version": evidence.V2115_OFFLINE_ADMISSION_AUDIT_SCHEMA_VERSION,
        "pass": True,
        "rows": [],
    }
    target = tmp_path / "negative-package"
    manifest_path, _checksums, scientific_complete = evidence._write_package(
        target,
        contract_path=CONTRACT_PATH,
        contract=contract,
        rows=rows,
        denominator=denominator,
        gates=gates,
        capability={},
        cross_model={},
        release_controls=release_controls,
        experiment_b={},
        rule_sensitivity=None,
        offline_audit=offline_audit,
        offline_payloads={},
        run_ledger={},
        budget_ledger={},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["scientific_matrix_complete"] is True
    assert manifest["scientific_claim_gates_supported"] is False
    assert manifest["scientific_complete"] is False
    assert scientific_complete is False
