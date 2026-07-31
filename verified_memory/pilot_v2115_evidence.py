"""Provider-free, fail-closed publication for the FinEvo V2.11.5 pilot.

The V2.11.5 science checkout is immutable.  This module is intentionally a
newer consumer: it reads the frozen contract and raw tree in place through an
explicit ``source_repo_root``, replays every terminal control, and writes a
separate reviewer package.  It never dispatches, retries, repairs, or mutates a
science cell.

V2.11.5 imported five operational authority cells with zero current provider
calls, then registered 131 fresh scientific cells.  Imported capability and
preflight payloads are therefore reported as dispatch gates, never as current
performance observations.  The five deterministic candidate-admission files
are re-executed and sealed at publication time because the execution-time
stage receipt binds their terminal summaries but not their full detail files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from .m2_episodic import EvidenceLinkedEpisodicTrack
from .m3_semantic import VerifiedSemanticRuleTrack
from .pilot_contract import PilotContract, canonical_sha256, load_pilot_contract
from .pilot_evidence import (
    HISTORICAL_SCOPE,
    PILOT_CHECKSUM_SCHEMA_VERSION,
    PILOT_FAILURE_LEDGER_SCHEMA_VERSION,
    PilotEvidenceError,
    PilotEvidencePackage,
    V211_SCIENTIFIC_STAGES,
    _aggregate_csv,
    _atomic_bytes,
    _claims,
    _cross_model_summary,
    _evidence_namespace,
    _experiment_a_gate,
    _experiment_b_summary,
    _experiment_c_gate,
    _experiment_d_gate,
    _json_copy,
    _method_scaffold,
    _narrative_gate,
    _pretty_bytes,
    _resolve_artifact,
    _sha256_file,
    _strict_json_load,
    _validated_experiment_c_sensitivity,
)
from .pilot_orchestrator import (
    PILOT_EXPERIMENT_C_SENSITIVITY_SCHEMA_VERSION,
    PILOT_OFFLINE_ADMISSION_SCHEMA_VERSION,
    PilotRunLedger,
    _build_experiment_c_sensitivity,
    _fixed_error_candidate,
)
from .pilot_v2112_evidence import (
    _normalize_v2112_ledger,
    _run_ledger_receipt,
    _validate_budget,
    _validate_release,
    _validate_stage_receipts,
)
from .pilot_v2115_gate import (
    PilotV2115GateError,
    V2115_PREFLIGHT_IMPORT_GATE_SCHEMA_VERSION,
    verified_v2115_gate_authority_binding,
    verify_v2115_gate_receipt,
)


V2115_CONTRACT_ID = "finevo-pilot-v2.11.5"
V2115_EVIDENCE_SCHEMA_VERSION = "finevo-pilot-v2.11.5-evidence-package-v1"
V2115_STAGE_RECEIPTS_SCHEMA_VERSION = "finevo-pilot-v2.11.5-stage-receipts-audit-v1"
V2115_RUN_LEDGER_RECEIPT_SCHEMA_VERSION = "finevo-pilot-v2.11.5-run-ledger-audit-v1"
V2115_BUDGET_RECEIPT_SCHEMA_VERSION = "finevo-pilot-v2.11.5-budget-audit-v1"
V2115_OFFLINE_ADMISSION_AUDIT_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.5-offline-candidate-admission-audit-v1"
)
V2115_C_SENSITIVITY_DIAGNOSTIC_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.5-publication-time-rule-sensitivity-diagnostic-v1"
)
V2115_SOURCE_TAG = "pilot-v2.11.5-science"
V2115_SOURCE_COMMIT = "2351ac2283f9fedb9dce70067174020be56ed9cc"
V2115_RAW_RELATIVE = Path("experiment_results/pilot-v2.11.5/raw")
V2115_CONTRACT_RELATIVE = Path("experiments/pilot_v2_11_5.yaml")
V2115_SOURCE_MANIFEST_RELATIVE = Path("experiments/pilot_v2_11_5_source_manifest.json")

_SOURCE_REQUIRED_TRACKED_FILES = (
    V2115_CONTRACT_RELATIVE.as_posix(),
    V2115_SOURCE_MANIFEST_RELATIVE.as_posix(),
    "run_pilot.py",
    "verified_memory/pilot_contract.py",
    "verified_memory/pilot_orchestrator.py",
    "verified_memory/pilot_v2115_acceptance.py",
    "verified_memory/pilot_v2115_gate.py",
)
_PUBLISHER_REQUIRED_TRACKED_FILES = (
    ".github/workflows/verified-memory-ci.yml",
    "run_pilot.py",
    "verified_memory/m2_episodic.py",
    "verified_memory/m3_semantic.py",
    "verified_memory/pilot_contract.py",
    "verified_memory/pilot_evidence.py",
    "verified_memory/pilot_orchestrator.py",
    "verified_memory/pilot_v2112_evidence.py",
    "verified_memory/pilot_v2115_evidence.py",
    "verified_memory/pilot_v2115_gate.py",
    "tests/test_pilot_v2115_evidence.py",
    "tests/test_run_pilot_v2115_evidence.py",
)

_EXPECTED_STAGE_IDS = (
    "parent-import",
    "capability-gate",
    "long-context-preflight",
    "experiment-c",
    "experiment-a",
    "experiment-d",
    "experiment-b",
    "cross-model",
)

_EXPECTED_C_SENSITIVITY_FAILURE = {
    "error_type": "PilotOrchestrationError",
    "message": "V2.11.5 imported Stage-0 selection requires paid provenance",
}
_EXPECTED_C_STAGE_RECEIPT_FILE_SHA256 = (
    "958cb161785c144c89861da3e9536e53069e8f1070a64c03f54647cbfe05b322"
)
_EXPECTED_C_STAGE_RECEIPT_CONTENT_SHA256 = (
    "39a9d35f4961fee4b0bc59ac67f7a9a2da0c3f95fddf77a418b92e518b6e2eba"
)
_EXPECTED_C_STAGE_RECEIPT_KEYS = frozenset(
    {
        "artifacts",
        "bindings",
        "complete_cell_count",
        "contract_id",
        "contract_sha256",
        "created_at",
        "denominator_terminal",
        "diagnostic_only",
        "execution_progression_go",
        "failure",
        "go",
        "go_models",
        "hard_stop_cell_count",
        "integrity",
        "registered_run_count",
        "schema_version",
        "scientific_evidence",
        "scientific_matrix_complete",
        "stage_id",
        "status",
        "status_counts",
        "terminal",
    }
)
_EXPECTED_C_STAGE_ARTIFACT_KEYS = frozenset(
    {"zero_api_rule_sensitivity_failure"}
)
_EXPECTED_STAGE_COUNTS = {
    "parent-import": 1,
    "capability-gate": 2,
    "long-context-preflight": 2,
    "experiment-c": 25,
    "experiment-a": 20,
    "experiment-d": 55,
    "experiment-b": 25,
    "cross-model": 6,
}
_OFFLINE_ADMISSION_ARM = "verified-error-candidate"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotEvidenceError(f"{name} must be an object")
    return value


def _frozen_contract(contract: PilotContract) -> None:
    if contract.contract_id != V2115_CONTRACT_ID:
        raise PilotEvidenceError(
            "V2.11.5 evidence adapter received a different contract"
        )
    if contract.status != "frozen":
        raise PilotEvidenceError(
            "V2.11.5 publish-evidence requires the frozen contract"
        )
    if tuple(contract.stage_ids) != _EXPECTED_STAGE_IDS:
        raise PilotEvidenceError("V2.11.5 evidence stage order drifted")
    counts = {
        stage_id: len(contract.expand(stage=stage_id))
        for stage_id in contract.stage_ids
    }
    if counts != _EXPECTED_STAGE_COUNTS or sum(counts.values()) != 136:
        raise PilotEvidenceError("V2.11.5 evidence denominator is not 136 cells")
    if contract.v2115_forward_boundary is None:
        raise PilotEvidenceError("V2.11.5 forward boundary is absent")
    if contract.v2115_consumer_authority_normalization_amendment is None:
        raise PilotEvidenceError(
            "V2.11.5 consumer-authority normalization amendment is absent"
        )


def _has_symlink_component(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _resolve_source_paths(
    *,
    source_repo_root: str | Path | None,
    contract_path: str | Path,
    raw_root: str | Path,
    run_ledger_path: str | Path,
) -> tuple[Path, Path, Path, Path]:
    if source_repo_root is None:
        raise PilotEvidenceError(
            "V2.11.5 publish-evidence requires --source-repo-root pointing "
            "to the immutable science checkout"
        )
    supplied_source = Path(source_repo_root)
    if supplied_source.is_symlink():
        raise PilotEvidenceError("V2.11.5 source repository cannot be a symlink")
    try:
        source = supplied_source.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PilotEvidenceError("V2.11.5 source repository does not exist") from exc
    if not source.is_dir():
        raise PilotEvidenceError("V2.11.5 source repository is not a directory")

    expected_contract = source / V2115_CONTRACT_RELATIVE
    expected_manifest = source / V2115_SOURCE_MANIFEST_RELATIVE
    expected_raw = source / V2115_RAW_RELATIVE
    expected_ledger = expected_raw / "run_ledger.json"
    supplied_contract = Path(contract_path)
    supplied_raw = Path(raw_root)
    supplied_ledger = Path(run_ledger_path)
    if any(
        path.is_symlink() for path in (supplied_contract, supplied_raw, supplied_ledger)
    ):
        raise PilotEvidenceError(
            "V2.11.5 supplied contract/raw/ledger cannot be symlinks"
        )
    if (
        Path(os.path.abspath(supplied_contract)) != expected_contract
        or Path(os.path.abspath(supplied_raw)) != expected_raw
        or Path(os.path.abspath(supplied_ledger)) != expected_ledger
    ):
        raise PilotEvidenceError(
            "V2.11.5 publisher requires the exact in-place science contract, "
            "raw namespace, source manifest, and raw-root ledger"
        )
    for path, name in (
        (expected_contract, "contract"),
        (expected_manifest, "source manifest"),
        (expected_raw, "raw root"),
        (expected_ledger, "run ledger"),
    ):
        if _has_symlink_component(source, path):
            raise PilotEvidenceError(f"V2.11.5 source {name} crosses a symlink")
    try:
        contract = supplied_contract.resolve(strict=True)
        raw = supplied_raw.resolve(strict=True)
        ledger = supplied_ledger.resolve(strict=True)
        manifest = expected_manifest.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PilotEvidenceError(
            "V2.11.5 source contract/raw/ledger/manifest is missing"
        ) from exc
    if (
        contract != expected_contract.resolve()
        or raw != expected_raw.resolve()
        or ledger != expected_ledger.resolve()
        or manifest != expected_manifest.resolve()
    ):
        raise PilotEvidenceError(
            "V2.11.5 publisher requires the exact in-place science contract, "
            "raw namespace, source manifest, and raw-root ledger"
        )
    if not contract.is_file() or not manifest.is_file() or not raw.is_dir():
        raise PilotEvidenceError("V2.11.5 source path types drifted")
    return source, contract, raw, ledger


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotEvidenceError(
            f"git provenance check failed in {repo_root}: {' '.join(args)}"
        ) from exc
    return result.stdout.strip()


def _tracked_head_blobs(
    repo_root: Path,
    relative_paths: Sequence[str],
) -> dict[str, str]:
    blobs: dict[str, str] = {}
    for relative in relative_paths:
        if (
            _git(
                repo_root,
                "ls-files",
                "--error-unmatch",
                "--",
                relative,
            )
            != relative
        ):
            raise PilotEvidenceError(
                f"required provenance file is not tracked exactly: {relative}"
            )
        # This checks index and worktree content against the committed blob.
        _git(repo_root, "diff", "--quiet", "HEAD", "--", relative)
        blobs[relative] = _git(repo_root, "rev-parse", f"HEAD:{relative}")
    return blobs


def _validate_source_git(
    source: Path,
    contract: PilotContract,
    *,
    expected_commit: str,
) -> dict[str, Any]:
    required_tag = str(contract.implementation["required_git_tag"])
    top_level = Path(_git(source, "rev-parse", "--show-toplevel")).resolve()
    if top_level != source:
        raise PilotEvidenceError(
            "V2.11.5 source path is not the exact git repository top-level"
        )
    head = _git(source, "rev-parse", "HEAD")
    tag_type = _git(source, "cat-file", "-t", f"refs/tags/{required_tag}")
    tag_object = _git(source, "rev-parse", f"refs/tags/{required_tag}^{{object}}")
    peeled = _git(source, "rev-parse", f"refs/tags/{required_tag}^{{commit}}")
    branch = _git(source, "rev-parse", "--abbrev-ref", "HEAD")
    tracked_status = _git(source, "status", "--porcelain", "--untracked-files=no")
    tracked_blobs = _tracked_head_blobs(source, _SOURCE_REQUIRED_TRACKED_FILES)
    if (
        required_tag != V2115_SOURCE_TAG
        or expected_commit != V2115_SOURCE_COMMIT
        or head != expected_commit
        or peeled != expected_commit
        or tag_type != "tag"
        or tag_object == expected_commit
        or branch != "HEAD"
        or tracked_status
    ):
        raise PilotEvidenceError(
            "V2.11.5 source checkout is not the exact detached, tracked-clean "
            "annotated science tag"
        )
    return {
        "source_repo_root": str(source),
        "contract_path": V2115_CONTRACT_RELATIVE.as_posix(),
        "source_manifest_path": V2115_SOURCE_MANIFEST_RELATIVE.as_posix(),
        "raw_root": V2115_RAW_RELATIVE.as_posix(),
        "git_tag": required_tag,
        "tag_object": tag_object,
        "resolved_git_commit": expected_commit,
        "detached_head": True,
        "tracked_worktree_clean": True,
        "required_tracked_head_blobs": tracked_blobs,
    }


def _publisher_provenance(code_root: Path) -> dict[str, Any]:
    top_level = Path(_git(code_root, "rev-parse", "--show-toplevel")).resolve()
    head = _git(code_root, "rev-parse", "HEAD")
    status = _git(code_root, "status", "--porcelain")
    if top_level != code_root or status:
        raise PilotEvidenceError(
            "V2.11.5 evidence publisher must be the exact clean repository root"
        )
    if head == V2115_SOURCE_COMMIT:
        raise PilotEvidenceError(
            "V2.11.5 evidence publisher must be a newer committed consumer"
        )
    _git(code_root, "merge-base", "--is-ancestor", V2115_SOURCE_COMMIT, head)
    tracked_blobs = _tracked_head_blobs(code_root, _PUBLISHER_REQUIRED_TRACKED_FILES)
    return {
        "repository_root": str(code_root),
        "git_commit": head,
        "branch": _git(code_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "tracked_worktree_clean": True,
        "required_tracked_head_blobs": tracked_blobs,
        "provider_calls": 0,
    }


def _normalized_stage_receipts(
    contract: PilotContract,
    *,
    raw_root: Path,
    ledger: PilotRunLedger,
    paid: Any,
    source_repo_root: Path,
) -> dict[str, Any]:
    value = _validate_stage_receipts(
        contract,
        raw_root=raw_root,
        ledger=ledger,
        paid=paid,
        authority_repo_root=source_repo_root,
    )
    value["schema_version"] = V2115_STAGE_RECEIPTS_SCHEMA_VERSION
    return value


def _authoritative_c_sensitivity_no_go(
    contract: PilotContract,
    *,
    raw_root: Path,
) -> dict[str, Any]:
    """Replay the exact immutable C-stage infrastructure no-go receipt."""

    path = raw_root / "experiment-c" / "stage_receipt.json"
    if _has_symlink_component(raw_root, path):
        raise PilotEvidenceError("Experiment C stage receipt crosses a symlink")
    receipt = _strict_json_load(path)
    integrity = _mapping(receipt.get("integrity"), "Experiment C receipt integrity")
    content = _json_copy(receipt)
    content.pop("integrity", None)
    artifacts = _mapping(receipt.get("artifacts"), "Experiment C receipt artifacts")
    failure = artifacts.get("zero_api_rule_sensitivity_failure")
    if (
        set(receipt) != _EXPECTED_C_STAGE_RECEIPT_KEYS
        or set(artifacts) != _EXPECTED_C_STAGE_ARTIFACT_KEYS
        or _sha256_file(path) != _EXPECTED_C_STAGE_RECEIPT_FILE_SHA256
        or receipt.get("schema_version") != "finevo-pilot-stage-receipt-v2"
        or receipt.get("contract_id") != contract.contract_id
        or receipt.get("contract_sha256") != contract.canonical_hash
        or receipt.get("stage_id") != "experiment-c"
        or receipt.get("status") != "complete-with-no-go"
        or receipt.get("terminal") is not True
        or receipt.get("go") is not False
        or receipt.get("execution_progression_go") is not True
        or receipt.get("denominator_terminal") is not True
        or receipt.get("scientific_matrix_complete") is not True
        or receipt.get("registered_run_count") != 25
        or receipt.get("complete_cell_count") != 25
        or receipt.get("hard_stop_cell_count") != 0
        or receipt.get("status_counts") != {"complete": 25}
        or receipt.get("failure") is not None
        or receipt.get("scientific_evidence") is not None
        or artifacts.get("zero_api_rule_sensitivity") is not None
        or failure != _EXPECTED_C_SENSITIVITY_FAILURE
        or integrity.get("canonicalization") != "json-sort-keys-utf8-v1"
        or integrity.get("content_sha256")
        != _EXPECTED_C_STAGE_RECEIPT_CONTENT_SHA256
        or canonical_sha256(content) != _EXPECTED_C_STAGE_RECEIPT_CONTENT_SHA256
    ):
        raise PilotEvidenceError(
            "Experiment C sensitivity no-go receipt differs from the frozen "
            "V2.11.5 infrastructure failure"
        )
    return {
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "content_sha256": integrity["content_sha256"],
        "status": receipt["status"],
        "go": receipt["go"],
        "execution_progression_go": receipt["execution_progression_go"],
        "scientific_matrix_complete": receipt["scientific_matrix_complete"],
        "complete_cell_count": receipt["complete_cell_count"],
        "failure": _json_copy(failure),
        "stage_authoritative": True,
    }


def _publication_time_c_sensitivity_replay(
    contract: PilotContract,
    *,
    raw_root: Path,
    rows: Sequence[Mapping[str, Any]],
    common_commit: str,
    source_repo_root: Path,
    paid: Any,
    stage_no_go: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the missing zero-provider C replay without changing stage authority."""

    value = _build_experiment_c_sensitivity(
        contract,
        raw_root=raw_root,
        git_tag=V2115_SOURCE_TAG,
        git_commit=common_commit,
        paid=paid,
        authority_repo_root=source_repo_root,
    )
    bindings = _mapping(value.get("bindings"), "diagnostic C replay bindings")
    cells = value.get("aggregate_cells")
    sensitivity_contract = _mapping(
        contract.stop_go["experiment_c"]["zero_api_sensitivity"],
        "Experiment C sensitivity contract",
    )
    expected_weights = list(sensitivity_contract["alternative_success_weights"])
    expected_outcomes = list(sensitivity_contract["outcome_definitions"])
    expected_grid = {
        (weight, outcome)
        for weight in expected_weights
        for outcome in expected_outcomes
    }
    expected_sources = {
        row["run_id"]: row["artifact_sha256"]
        for row in rows
        if (
            row["stage_id"] == "experiment-c"
            and row["model_id"] == "gpt52_main"
            and row["arm_id"] == "full"
            and row["status"] == "complete"
        )
    }
    source_rows = bindings.get("source_manifests")
    valid_source_sequence = (
        isinstance(source_rows, Sequence)
        and not isinstance(source_rows, (str, bytes))
        and len(source_rows) == 5
        and all(isinstance(source, Mapping) for source in source_rows)
    )
    observed_sources = (
        {
            str(source.get("run_id")): source.get("manifest_sha256")
            for source in source_rows
            if isinstance(source, Mapping)
        }
        if valid_source_sequence
        else {}
    )
    valid_cell_sequence = (
        isinstance(cells, Sequence)
        and not isinstance(cells, (str, bytes))
        and len(cells) == len(expected_grid)
        and all(
            isinstance(cell, Mapping) and cell.get("source_run_count") == 5
            for cell in cells
        )
    )
    observed_grid = (
        {
            (
                cell.get("alternative_success_weight"),
                cell.get("outcome_definition"),
            )
            for cell in cells
            if isinstance(cell, Mapping)
        }
        if valid_cell_sequence
        else set()
    )
    if (
        value.get("schema_version")
        != PILOT_EXPERIMENT_C_SENSITIVITY_SCHEMA_VERSION
        or value.get("status") != "pass"
        or value.get("terminal") is not True
        or value.get("provider_calls") != 0
        or value.get("descriptive_only") is not True
        or value.get("effectiveness_gate") is not False
        or value.get("source_run_count") != 5
        or value.get("alternative_success_weights") != expected_weights
        or value.get("outcome_definitions") != expected_outcomes
        or bindings.get("contract_sha256") != contract.canonical_hash
        or bindings.get("git_tag") != V2115_SOURCE_TAG
        or bindings.get("git_commit") != common_commit
        or bindings.get("stage0_selection_source_kind")
        != "v2.11.5-sealed-parent-import"
        or len(expected_sources) != 5
        or not valid_source_sequence
        or len(observed_sources) != 5
        or observed_sources != expected_sources
        or not valid_cell_sequence
        or observed_grid != expected_grid
    ):
        raise PilotEvidenceError(
            "publication-time Experiment C diagnostic replay is not bound to "
            "the frozen V2.11.5 inputs"
        )

    diagnostic = _json_copy(value)
    diagnostic.update(
        {
            "schema_version": V2115_C_SENSITIVITY_DIAGNOSTIC_SCHEMA_VERSION,
            "source_replay_schema_version": (
                PILOT_EXPERIMENT_C_SENSITIVITY_SCHEMA_VERSION
            ),
            "status": "diagnostic-replay-complete",
            "publication_time_replay": True,
            "stage_authoritative": False,
            "diagnostic_only": True,
            "scientific_evidence": False,
            "effectiveness_gate": False,
            "original_stage_no_go": _json_copy(stage_no_go),
            "publication_input_bindings": {
                "contract_sha256": contract.canonical_hash,
                "science_git_tag": V2115_SOURCE_TAG,
                "science_git_commit": common_commit,
                "experiment_c_stage_receipt_file_sha256": stage_no_go[
                    "file_sha256"
                ],
                "experiment_c_stage_receipt_content_sha256": stage_no_go[
                    "content_sha256"
                ],
                "parent_import_file_sha256": bindings[
                    "stage0_selection_file_sha256"
                ],
                "parent_import_content_sha256": bindings[
                    "stage0_selection_content_sha256"
                ],
                "source_matrix_sha256": bindings["source_matrix_sha256"],
            },
            "claim_boundary": (
                "Publication-time deterministic diagnostic only. The immutable "
                "Experiment C receipt remains complete-with-no-go/go=false; "
                "this replay is not stage-authoritative or scientific evidence "
                "and cannot restore the rule-reliability claim."
            ),
        }
    )
    diagnostic["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(diagnostic),
    }
    return diagnostic


def _validated_v2115_experiment_c_sensitivity(
    contract: PilotContract,
    *,
    raw_root: Path,
    rows: Sequence[Mapping[str, Any]],
    common_commit: str,
    source_repo_root: Path,
    paid: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Preserve C's no-go while optionally adding a diagnostic-only replay."""

    c_rows = [row for row in rows if row["stage_id"] == "experiment-c"]
    c_complete = bool(c_rows) and all(
        row["status"] == "complete" and row["scientific_eligible"] is True
        for row in c_rows
    )
    path = raw_root / "experiment-c" / "rule_sensitivity.json"
    if not c_complete:
        return _validated_experiment_c_sensitivity(
            contract,
            raw_root=raw_root,
            rows=rows,
            common_commit=common_commit,
            source_repo_root=source_repo_root,
        )
    if path.exists():
        # A source artifact is authoritative only when the source receipt also
        # registered it.  V2.11.5's frozen receipt instead registered failure,
        # so an added file would be unbound mutation and must fail closed.
        _authoritative_c_sensitivity_no_go(contract, raw_root=raw_root)
        raise PilotEvidenceError(
            "Experiment C sensitivity file exists despite its immutable "
            "failure receipt"
        )

    stage_no_go = _authoritative_c_sensitivity_no_go(
        contract,
        raw_root=raw_root,
    )
    control: dict[str, Any] = {
        "pass": False,
        "available": False,
        "path": str(path),
        "provider_calls": 0,
        "infrastructure_no_go": True,
        "publication_time_replay": True,
        "stage_authoritative": True,
        "scientific_evidence": False,
        "diagnostic_only": False,
        "original_stage_no_go": _json_copy(stage_no_go),
        "reason": (
            "the immutable Experiment C stage did not seal its preregistered "
            "zero-API sensitivity artifact"
        ),
    }
    try:
        diagnostic = _publication_time_c_sensitivity_replay(
            contract,
            raw_root=raw_root,
            rows=rows,
            common_commit=common_commit,
            source_repo_root=source_repo_root,
            paid=paid,
            stage_no_go=stage_no_go,
        )
    except Exception as exc:  # diagnostic failure cannot erase the ITT package
        control.update(
            {
                "diagnostic_replay_available": False,
                "diagnostic_replay_status": "failed",
                "diagnostic_replay_failure": {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                "claim_boundary": (
                    "Experiment C remains complete-with-no-go. Publication-time "
                    "diagnostic recovery also failed and contributes no evidence."
                ),
            }
        )
        return None, control
    control.update(
        {
            "diagnostic_replay_available": True,
            "diagnostic_replay_status": "complete",
            "diagnostic_content_sha256": diagnostic["integrity"][
                "content_sha256"
            ],
            "diagnostic_source_run_count": diagnostic.get("source_run_count"),
            "diagnostic_grid_cell_count": len(diagnostic["aggregate_cells"]),
            "claim_boundary": diagnostic["claim_boundary"],
        }
    )
    return diagnostic, control


def _apply_c_sensitivity_no_go(
    gate: Mapping[str, Any],
    sensitivity_control: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep the C claim fail-closed when its preregistered control is absent."""

    result = _json_copy(gate)
    if sensitivity_control.get("pass") is True:
        return result
    prior_status = result.get("status")
    prior_support = result.get("support_rule_reliability")
    stage_no_go = sensitivity_control.get("original_stage_no_go", {})
    reasons = list(result.get("reasons", []))
    reasons.append(
        "preregistered zero-API rule sensitivity was not sealed by the "
        "authoritative Experiment C stage"
    )
    result.update(
        {
            "status": "no-go",
            "scientific_evidence_complete": False,
            "support_rule_reliability": False,
            "core_effect_status_before_sensitivity_control": prior_status,
            "core_effect_support_before_sensitivity_control": prior_support,
            "formal_stage_status": (
                stage_no_go.get("status")
                if isinstance(stage_no_go, Mapping)
                else None
            ),
            "formal_stage_go": (
                stage_no_go.get("go")
                if isinstance(stage_no_go, Mapping)
                else None
            ),
            "experiment_c_sensitivity": _json_copy(sensitivity_control),
            "retirement_delay_boundary": (
                "Retirement delay is reported only when both paired values "
                "are observed; terminal-active/null values are never encoded "
                "as zero or infinity."
            ),
            "candidate_admission_boundary": (
                "The five deterministic candidate-admission fixture rows are "
                "a fixed mechanism check, not independent random repetitions."
            ),
            "claim_action": (
                "withdraw or narrow the rule-reliability claim; the immutable "
                "Experiment C receipt is complete-with-no-go and any "
                "publication-time replay is diagnostic only"
            ),
            "reasons": reasons,
        }
    )
    return result


def _validated_post_gate(
    contract: PilotContract,
    *,
    source_repo_root: Path,
    raw_root: Path,
    commit: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = raw_root / "long-context-preflight" / "post_gate_authority.json"
    complete_science = any(
        row["stage_id"] in V211_SCIENTIFIC_STAGES and row["status"] == "complete"
        for row in rows
    )
    if not path.exists():
        if complete_science:
            raise PilotEvidenceError(
                "scientific cells exist without a V2.11.5 post-gate authority"
            )
        return {
            "available": False,
            "go": False,
            "reason": "authority import stopped before global gate sealing",
            "scientific_evidence": False,
        }
    if _has_symlink_component(raw_root, path):
        raise PilotEvidenceError("V2.11.5 post-gate path crosses a symlink")
    try:
        binding = verified_v2115_gate_authority_binding(
            path,
            repo_root=source_repo_root,
            expected_git_commit=commit,
            expected_contract_sha256=contract.canonical_hash,
            contract=contract,
        )
        receipt = verify_v2115_gate_receipt(
            _strict_json_load(path),
            expected_git_commit=commit,
            expected_contract_sha256=contract.canonical_hash,
        )
    except (PilotV2115GateError, OSError, TypeError, ValueError) as exc:
        raise PilotEvidenceError(
            f"V2.11.5 post-gate authority failed replay: {exc}"
        ) from exc
    if complete_science and receipt.get("go") is not True:
        raise PilotEvidenceError(
            "scientific cells exist behind a V2.11.5 global authority no-go"
        )

    by_model = {
        str(row["model_id"]): row
        for row in rows
        if row["stage_id"] == "long-context-preflight"
    }
    imports: dict[str, Any] = {}
    for spec in contract.expand(stage="long-context-preflight"):
        row = by_model.get(spec.model_id)
        if not isinstance(row, Mapping) or row.get("status") != "complete":
            raise PilotEvidenceError(
                f"V2.11.5 global gate lacks completed import row {spec.model_id}"
            )
        gate = _mapping(row.get("gate_evidence"), "preflight import gate")
        body = _json_copy(gate)
        integrity = gate.get("integrity")
        if isinstance(integrity, Mapping):
            body["integrity"].pop("content_sha256", None)
        if (
            gate.get("schema_version") != V2115_PREFLIGHT_IMPORT_GATE_SCHEMA_VERSION
            or not isinstance(integrity, Mapping)
            or integrity.get("canonicalization") != "json-sort-keys-utf8-v1"
            or integrity.get("content_sha256") != canonical_sha256(body)
            or gate.get("model_id") != spec.model_id
            or gate.get("contract_id") != contract.contract_id
            or gate.get("contract_sha256") != contract.canonical_hash
            or gate.get("go") is not True
            or gate.get("capability_pass") is not True
            or gate.get("interface_pass") is not True
            or gate.get("historical_action_samples") != 24
            or gate.get("historical_semantic_samples") != 8
            or gate.get("historical_provider_calls") != 32
            or gate.get("provider_calls_current_attempt") != 0
            or gate.get("provider_construction_current_attempt") is not False
            or gate.get("scientific_evidence") is not False
        ):
            raise PilotEvidenceError(
                f"V2.11.5 preflight authority-import gate drifted for {spec.model_id}"
            )
        expected_root = (
            raw_root
            / "long-context-preflight"
            / "imported_observed_p95"
            / spec.model_id
        )
        authority_path = Path(str(gate.get("authority_receipt")))
        projection_path = Path(str(gate.get("projection_p95")))
        try:
            unsafe_path = (
                _has_symlink_component(raw_root, authority_path)
                or _has_symlink_component(raw_root, projection_path)
                or authority_path.resolve(strict=True)
                != (expected_root / "observed_p95_authority_receipt.json").resolve(
                    strict=True
                )
                or projection_path.resolve(strict=True)
                != (expected_root / "projection_p95.json").resolve(strict=True)
            )
        except FileNotFoundError as exc:
            raise PilotEvidenceError(
                f"V2.11.5 preflight authority path is missing for {spec.model_id}"
            ) from exc
        if unsafe_path:
            raise PilotEvidenceError(
                f"V2.11.5 preflight authority path drifted for {spec.model_id}"
            )
        imports[spec.model_id] = {
            "historical_action_samples": 24,
            "historical_semantic_samples": 8,
            "historical_provider_calls": 32,
            "provider_calls_current_attempt": 0,
            "provider_construction_current_attempt": False,
            "scientific_evidence": False,
            "claim_boundary": (
                "imported historical dispatch-budget authority; not a fresh "
                "V2.11.5 capability or performance sample"
            ),
        }
    return {
        "available": True,
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "content_sha256": receipt["receipt_sha256"],
        "go": receipt["go"],
        "denominator": _json_copy(receipt["denominator"]),
        "authority_sources": _json_copy(receipt["authority_sources"]),
        "reservations": _json_copy(receipt["reservations"]),
        "provider_boundary": _json_copy(receipt["provider_boundary"]),
        "binding": _json_copy(binding),
        "operational_imports": imports,
        "provider_calls_current_attempt": 0,
        "scientific_evidence": False,
    }


def _v2115_capability_by_model(
    rows: Sequence[Mapping[str, Any]],
    contract: PilotContract,
) -> dict[str, dict[str, Any]]:
    components: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if row["stage_id"] not in {"capability-gate", "long-context-preflight"}:
            continue
        key = "capability_gate" if row["stage_id"] == "capability-gate" else "preflight"
        components.setdefault(str(row["model_id"]), {})[key] = row

    result: dict[str, dict[str, Any]] = {}
    for model_id, role in contract.model_roles.items():
        if role.role == "calibration_only":
            continue
        if not role.dispatch_eligible:
            result[model_id] = {
                "ledger_status": "capability-no-go",
                "artifact_validated": False,
                "capability": {},
                "registered_dispatch_cells": 0,
                "contract_role": role.role,
                "dispatch_eligible": False,
                "ineligibility_reason": role.ineligibility_reason,
                "capability_gate": None,
                "closed_loop_preflight": None,
            }
            continue
        pair = components.get(model_id, {})
        gate_row = pair.get("capability_gate")
        preflight_row = pair.get("preflight")
        both_complete = bool(
            isinstance(gate_row, Mapping)
            and isinstance(preflight_row, Mapping)
            and gate_row.get("status") == "complete"
            and preflight_row.get("status") == "complete"
            and gate_row.get("artifact_kind") is not None
            and preflight_row.get("artifact_kind") is not None
        )
        capability: dict[str, Any] = {}
        gate_summary: dict[str, Any] | None = None
        preflight_summary: dict[str, Any] | None = None
        if both_complete:
            wrapper = _mapping(gate_row.get("capability"), "capability wrapper")
            inherited = _mapping(wrapper.get("capability"), "imported capability")
            gate_evidence = _mapping(
                gate_row.get("gate_evidence"), "capability gate evidence"
            )
            preflight_evidence = _mapping(
                preflight_row.get("gate_evidence"), "preflight gate evidence"
            )
            if (
                inherited.get("model_id") != model_id
                or inherited.get("capability_pass") is not True
                or inherited.get("interface_pass") is not True
                or wrapper.get("provider_calls_current_attempt") != 0
                or wrapper.get("provider_construction_current_attempt") is not False
                or wrapper.get("imported_effect_cells") != 0
                or wrapper.get("scientific_evidence") is not False
                or gate_evidence.get("go") is not True
                or gate_evidence.get("capability_pass") is not True
                or gate_evidence.get("interface_pass") is not True
                or gate_evidence.get("provider_calls_current_attempt") != 0
                or gate_evidence.get("provider_construction_current_attempt")
                is not False
                or preflight_evidence.get("go") is not True
                or preflight_evidence.get("capability_pass") is not True
                or preflight_evidence.get("interface_pass") is not True
                or preflight_evidence.get("provider_calls_current_attempt") != 0
                or preflight_evidence.get("provider_construction_current_attempt")
                is not False
            ):
                raise PilotEvidenceError(
                    f"V2.11.5 imported capability/preflight drifted for {model_id}"
                )
            capability = _json_copy(wrapper)
            # Keep the generic cross-model gate compatible while making the
            # imported source/current boundary explicit.
            capability["preflight_go"] = True
            capability["preflight_interface_pass"] = True
            capability["provider_calls_current_attempt"] = 0
            capability["scientific_evidence"] = False
            capability["claim_boundary"] = (
                "historical capability and P95 dispatch authority only; zero "
                "current V2.11.5 capability/preflight provider calls"
            )
            gate_summary = {
                "stage_id": "capability-gate",
                "ledger_status": "complete",
                "artifact_validated": True,
                "gate_evidence": _json_copy(gate_evidence),
            }
            preflight_summary = {
                "stage_id": "long-context-preflight",
                "ledger_status": "complete",
                "artifact_validated": True,
                "gate_evidence": _json_copy(preflight_evidence),
            }
        result[model_id] = {
            "ledger_status": "complete" if both_complete else "incomplete",
            "artifact_validated": both_complete,
            "capability": capability,
            "registered_dispatch_cells": sum(
                spec.model_id == model_id
                and spec.execution_mode
                in {"capability_authority_import", "preflight_authority_import"}
                for spec in contract.expand()
            ),
            "contract_role": role.role,
            "dispatch_eligible": True,
            "ineligibility_reason": None,
            "capability_gate": gate_summary,
            "closed_loop_preflight": preflight_summary,
        }
    return result


def _expected_offline_payload(
    contract: PilotContract,
    spec: Any,
) -> dict[str, Any]:
    verified_episodic = EvidenceLinkedEpisodicTrack(
        run_id=spec.run_id,
        seed=spec.environment_seed,
        agent_id=0,
    )
    verified_semantic = VerifiedSemanticRuleTrack(verified_episodic)
    raw_candidate = _fixed_error_candidate()
    candidate = verified_semantic.parse_candidate(
        raw_candidate,
        generator_id="fixed-preregistered-error",
    )
    verified_rule = verified_semantic.submit_candidate(candidate, current_t=5)
    unverified_episodic = EvidenceLinkedEpisodicTrack(
        run_id=f"{spec.run_id}--unverified-control",
        seed=spec.environment_seed,
        agent_id=0,
    )
    unverified_semantic = VerifiedSemanticRuleTrack(unverified_episodic)
    unverified_rule = unverified_semantic.propose_unverified_immediate(
        raw_candidate,
        current_t=5,
        generator_id="fixed-preregistered-error",
    )
    return {
        "schema_version": PILOT_OFFLINE_ADMISSION_SCHEMA_VERSION,
        "contract_sha256": contract.canonical_hash,
        "run_spec": spec.to_dict(),
        "candidate": candidate.to_dict(),
        "verified_rule": verified_rule.to_dict(),
        "unverified_rule": unverified_rule.to_dict(),
        "check": {
            "unsupported_candidate_rejected": verified_rule.status == "rejected",
            "false_rule_ever_active": verified_rule.status == "active",
            "unverified_false_rule_ever_active": unverified_rule.status == "active",
            "same_candidate_content": (
                unverified_rule.injection_provenance.get("raw_response_hash")
                == candidate.raw_response_hash
            ),
            "provider_calls": 0,
        },
        "diagnostic_only": False,
        "scientific_evidence": True,
    }


def _validate_offline_candidate_admission(
    contract: PilotContract,
    *,
    raw_root: Path,
    rows: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    observed = _mapping(ledger.get("runs"), "V2.11.5 run ledger rows")
    by_run = {str(row["run_id"]): row for row in rows}
    specs = tuple(
        spec
        for spec in contract.expand(stage="experiment-c")
        if spec.execution_mode == "offline_candidate_admission"
    )
    if len(specs) != 5 or {spec.arm_id for spec in specs} != {_OFFLINE_ADMISSION_ARM}:
        raise PilotEvidenceError(
            "V2.11.5 offline candidate-admission denominator drifted"
        )
    audit_rows: list[dict[str, Any]] = []
    copies: dict[str, dict[str, Any]] = {}
    for spec in specs:
        row = _mapping(by_run.get(spec.run_id), "offline normalized ledger row")
        source_row = _mapping(observed.get(spec.run_id), "offline source ledger row")
        detail = (
            raw_root
            / spec.stage_id
            / "runs"
            / spec.run_id
            / "offline_candidate_admission.json"
        )
        if row.get("status") != "complete":
            if detail.exists() or detail.is_symlink():
                raise PilotEvidenceError(
                    f"failed offline cell retains ambiguous detail: {spec.run_id}"
                )
            audit_rows.append(
                {
                    "run_id": spec.run_id,
                    "environment_seed": spec.environment_seed,
                    "status": row.get("status"),
                    "admitted": False,
                    "reason": "registered ITT cell did not complete",
                }
            )
            continue
        if (
            row.get("artifact_kind") is None
            or row.get("scientific_eligible") is not True
            or _has_symlink_component(raw_root, detail)
            or not detail.is_file()
        ):
            raise PilotEvidenceError(
                f"completed offline detail is missing or unsafe: {spec.run_id}"
            )
        payload = _strict_json_load(detail)
        expected = _expected_offline_payload(contract, spec)
        if payload != expected:
            raise PilotEvidenceError(
                f"offline candidate-admission deterministic replay drifted: {spec.run_id}"
            )
        artifact = source_row.get("artifact")
        terminal_path = _resolve_artifact(raw_root, artifact)
        terminal = _strict_json_load(terminal_path)
        terminal_payload = _mapping(
            terminal.get("payload"), "offline terminal summary payload"
        )
        terminal_metrics = _mapping(
            terminal_payload.get("metrics"), "offline terminal metrics"
        )
        if (
            terminal_metrics.get("rule_reliability") != expected["check"]
            or terminal_payload.get("gate_evidence") != expected["check"]
            or terminal_payload.get("offline_source") != str(detail.resolve())
            or row.get("metrics", {}).get("rule_reliability") != expected["check"]
            or row.get("gate_evidence") != expected["check"]
        ):
            raise PilotEvidenceError(
                f"offline detail/terminal-summary binding drifted: {spec.run_id}"
            )
        published_path = f"audit/offline_candidate_admission/{spec.run_id}.json"
        copies[published_path] = payload
        audit_rows.append(
            {
                "run_id": spec.run_id,
                "environment_seed": spec.environment_seed,
                "status": "complete",
                "admitted": True,
                "source_relative_path": detail.relative_to(raw_root).as_posix(),
                "source_file_sha256": _sha256_file(detail),
                "source_content_sha256": canonical_sha256(payload),
                "published_path": published_path,
                "candidate_id": payload["candidate"]["candidate_id"],
                "verified_rule_status": payload["verified_rule"]["status"],
                "unverified_rule_status": payload["unverified_rule"]["status"],
                "check": _json_copy(payload["check"]),
            }
        )
    admitted = sum(item["admitted"] is True for item in audit_rows)
    audit = {
        "schema_version": V2115_OFFLINE_ADMISSION_AUDIT_SCHEMA_VERSION,
        "contract_sha256": contract.canonical_hash,
        "registered_cell_count": 5,
        "publication_admitted_detail_count": admitted,
        "all_registered_cells_accounted": len(audit_rows) == 5,
        "pass": admitted == 5,
        "provider_calls": 0,
        "execution_time_full_detail_receipt_bound": False,
        "publication_time_deterministic_revalidation": True,
        "independent_random_repetitions": False,
        "claim_boundary": (
            "publication-time deterministic seal of the fixed zero-provider "
            "candidate-admission trace; the five seed-indexed fixture rows are "
            "not independent random repetitions and this is not an "
            "execution-time full-detail hash"
        ),
        "rows": audit_rows,
    }
    return audit, copies


def _report(
    contract: PilotContract,
    *,
    denominator: Mapping[str, Any],
    gates: Mapping[str, Any],
    capability: Mapping[str, Any],
    cross_model: Mapping[str, Any],
    release_controls: Mapping[str, Any],
) -> str:
    sensitivity_control = release_controls.get(
        "experiment_c_sensitivity",
        {"pass": False, "available": False},
    )
    lines = [
        "# FinEvo V2.11.5 preregistered mechanism micro-pilot",
        "",
        f"- Contract: `{contract.contract_id}` / `{contract.canonical_hash}`",
        "- Scale: 4 agents x 12 months; not the confirmatory 10x24x5 or 100x240 run.",
        (
            "- ITT denominator: 136/136 terminal; statuses "
            f"`{json.dumps(denominator['status_counts'], sort_keys=True)}`."
        ),
        (
            "- V2.11.4 is immutable pre-dispatch failure/budget history only; "
            "zero historical treatment-effect cells are reused."
        ),
        "",
        "## Claim decisions",
        "",
    ]
    for name in ("experiment_a", "experiment_c", "experiment_d", "narrative"):
        gate = gates[name]
        boundary = gate.get("claim_action") or gate.get("claim_boundary")
        lines.append(f"- `{name}`: `{gate['status']}` — {boundary}")
    lines.extend(
        [
            "",
            (
                "- Experiment C sensitivity control: "
                f"`{json.dumps(sensitivity_control, sort_keys=True)}`"
            ),
            (
                "- The source Experiment C receipt remains authoritative as "
                "`complete-with-no-go/go=false`; a publication-time replay, "
                "when available, is diagnostic only and cannot restore the claim. "
                "The exact source receipt is copied to "
                "`source_receipts/experiment-c-stage_receipt.json`."
            ),
            (
                "- Experiment C core effect gate before the missing sensitivity "
                "control: `"
                f"{gates['experiment_c'].get('core_effect_status_before_sensitivity_control')}`; "
                "formal publication decision: `no-go`."
            ),
            (
                "- Retirement-delay null/terminal-active values are not recoded "
                "as 0 or infinity; the five deterministic candidate-admission "
                "fixture rows are not independent random repetitions."
            ),
            "",
            "Experiment B is descriptive; no arm is selected by wealth alone.",
            "",
            "## Imported operational authority boundary",
            "",
            (
                "- Capability and P95 samples are historical dispatch authority; "
                "V2.11.5 made zero provider calls for the five operational imports."
            ),
            f"- Capability/preflight: `{json.dumps(capability, sort_keys=True)}`",
            f"- Cross-model: `{json.dumps(cross_model, sort_keys=True)}`",
            "- No result supports backbone-independent wording.",
            "",
            "## Provenance and controls",
            "",
            f"- Science source: `{json.dumps(release_controls['science_source'], sort_keys=True)}`",
            f"- Publisher: `{json.dumps(release_controls['publisher'], sort_keys=True)}`",
            f"- Post-gate: `{json.dumps(release_controls['post_gate'], sort_keys=True)}`",
            f"- Budget: `{json.dumps(release_controls['budget'], sort_keys=True)}`",
            (
                "- Candidate admission: publication-time deterministic replay and "
                "seal; the execution-time receipt did not hash the full detail."
            ),
            "",
            (
                "All ITT failures remain in `failure_ledger.json`; raw prompts and "
                "provider outputs are not copied."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _write_package(
    target: Path,
    *,
    contract_path: Path,
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
    denominator: Mapping[str, Any],
    gates: Mapping[str, Any],
    capability: Mapping[str, Any],
    cross_model: Mapping[str, Any],
    release_controls: Mapping[str, Any],
    experiment_b: Mapping[str, Any],
    rule_sensitivity: Mapping[str, Any] | None,
    offline_audit: Mapping[str, Any],
    offline_payloads: Mapping[str, Mapping[str, Any]],
    run_ledger: Mapping[str, Any],
    budget_ledger: Mapping[str, Any],
) -> tuple[Path, Path, bool]:
    target.mkdir(parents=True, exist_ok=True)
    contract_target = target / "contract" / contract_path.name
    contract_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(contract_path, contract_target)
    source_manifest = contract_path.with_name(V2115_SOURCE_MANIFEST_RELATIVE.name)
    source_manifest_target = contract_target.with_name(source_manifest.name)
    shutil.copyfile(source_manifest, source_manifest_target)
    if load_pilot_contract(contract_target).canonical_hash != contract.canonical_hash:
        raise PilotEvidenceError("copied V2.11.5 contract failed revalidation")

    sanitized_rows = [
        {
            key: _json_copy(row[key])
            for key in (
                "run_id",
                "contract_id",
                "stage_id",
                "model_id",
                "requested_model",
                "arm_id",
                "narrative_id",
                "environment_seed",
                "decoding_seed",
                "utility_profile_id",
                "shock_id",
                "budget_bucket",
                "num_agents",
                "episode_length",
                "execution_mode",
                "status",
                "failure",
                "artifact_kind",
                "artifact_sha256",
                "scientific_eligible",
                "metrics",
                "gate_evidence",
                "capability",
                "narrative",
            )
        }
        for row in rows
    ]
    claims = _claims(gates, denominator=denominator)
    aggregate = {
        "schema_version": V2115_EVIDENCE_SCHEMA_VERSION,
        "evidence_namespace": _evidence_namespace(contract),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "pilot_tag": contract.implementation["required_git_tag"],
        "resolved_git_commit": release_controls["resolved_git_commit"],
        "denominator": _json_copy(denominator),
        "claim_gates": _json_copy(gates),
        "claims": claims,
        "experiment_b": _json_copy(experiment_b),
        "model_capability": _json_copy(capability),
        "cross_model": _json_copy(cross_model),
        "release_controls": _json_copy(release_controls),
        "experiment_c_rule_sensitivity": _json_copy(rule_sensitivity),
        "offline_candidate_admission_audit": _json_copy(offline_audit),
        "rows": sanitized_rows,
    }
    _atomic_bytes(target / "aggregate.json", _pretty_bytes(aggregate))
    _atomic_bytes(target / "aggregate.csv", _aggregate_csv(sanitized_rows))
    failures = [
        {
            "run_id": row["run_id"],
            "stage_id": row["stage_id"],
            "model_id": row["model_id"],
            "arm_id": row["arm_id"],
            "narrative_id": row["narrative_id"],
            "environment_seed": row["environment_seed"],
            "status": row["status"],
            "failure": _json_copy(row["failure"]),
        }
        for row in rows
        if row["status"] != "complete"
    ]
    _atomic_bytes(
        target / "failure_ledger.json",
        _pretty_bytes(
            {
                "schema_version": PILOT_FAILURE_LEDGER_SCHEMA_VERSION,
                "contract_sha256": contract.canonical_hash,
                "denominator": _json_copy(denominator),
                "rows": failures,
            }
        ),
    )
    for name, value in (
        ("stage_receipts.json", release_controls["stage_receipts"]),
        ("run_ledger_receipt.json", release_controls["run_ledger"]),
        ("budget_receipt.json", release_controls["budget"]),
        ("offline_candidate_admission_audit.json", offline_audit),
    ):
        _atomic_bytes(target / name, _pretty_bytes(value))
    sensitivity_release = release_controls.get("experiment_c_sensitivity")
    if isinstance(sensitivity_release, Mapping):
        original_stage = sensitivity_release.get("original_stage_no_go")
        if isinstance(original_stage, Mapping) and isinstance(
            original_stage.get("path"), str
        ):
            source_receipt = Path(original_stage["path"])
            if (
                not source_receipt.is_file()
                or source_receipt.is_symlink()
                or _sha256_file(source_receipt) != original_stage.get("file_sha256")
            ):
                raise PilotEvidenceError(
                    "authoritative Experiment C no-go receipt changed before copy"
                )
            copied_receipt = (
                target / "source_receipts" / "experiment-c-stage_receipt.json"
            )
            _atomic_bytes(copied_receipt, source_receipt.read_bytes())
            if _sha256_file(copied_receipt) != original_stage.get("file_sha256"):
                raise PilotEvidenceError(
                    "authoritative Experiment C no-go receipt changed during copy"
                )
    _atomic_bytes(target / "audit" / "run_ledger.json", _pretty_bytes(run_ledger))
    _atomic_bytes(target / "audit" / "budget_ledger.json", _pretty_bytes(budget_ledger))
    for relative, payload in offline_payloads.items():
        _atomic_bytes(target / relative, _pretty_bytes(payload))
    if release_controls["post_gate"].get("available") is True:
        _atomic_bytes(
            target / "post_gate_authority.json",
            _pretty_bytes(
                _strict_json_load(Path(release_controls["post_gate"]["path"]))
            ),
        )
    if rule_sensitivity is not None:
        sensitivity_name = (
            "experiment_c_rule_sensitivity_diagnostic.json"
            if rule_sensitivity.get("publication_time_replay") is True
            else "experiment_c_rule_sensitivity.json"
        )
        _atomic_bytes(
            target / sensitivity_name,
            _pretty_bytes(rule_sensitivity),
        )
    _atomic_bytes(
        target / "method_differences_scaffold.json",
        _pretty_bytes(_method_scaffold(contract_path.name)),
    )
    _atomic_bytes(
        target / "reviewer_report.md",
        _report(
            contract,
            denominator=denominator,
            gates=gates,
            capability=capability,
            cross_model=cross_model,
            release_controls=release_controls,
        ).encode("utf-8"),
    )

    scientific_matrix_complete = bool(
        denominator["pass"]
        and all(
            row["status"] == "complete" and row["scientific_eligible"] is True
            for row in rows
            if row["stage_id"] in V211_SCIENTIFIC_STAGES
        )
        and release_controls["post_gate"].get("go") is True
        and release_controls["budget"].get("pass") is True
        and offline_audit.get("pass") is True
    )
    claim_gates_supported = all(
        gates[name].get("status") == "supported"
        for name in ("experiment_a", "experiment_c", "experiment_d", "narrative")
    )
    publication_controls_complete = bool(
        isinstance(sensitivity_release, Mapping)
        and sensitivity_release.get("pass") is True
    )
    scientific_complete = bool(
        scientific_matrix_complete
        and publication_controls_complete
        and claim_gates_supported
    )
    published_files = sorted(
        path.relative_to(target).as_posix()
        for path in target.rglob("*")
        if path.is_file()
    )
    manifest = {
        "schema_version": V2115_EVIDENCE_SCHEMA_VERSION,
        "evidence_namespace": _evidence_namespace(contract),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "pilot_tag": contract.implementation["required_git_tag"],
        "resolved_git_commit": release_controls["resolved_git_commit"],
        "publisher_commit": release_controls["publisher"]["git_commit"],
        "scientific_matrix_complete": scientific_matrix_complete,
        "publication_controls_complete": publication_controls_complete,
        "scientific_claim_gates_supported": claim_gates_supported,
        "scientific_complete": scientific_complete,
        "claim_gates": _json_copy(gates),
        "published_files": published_files,
        "historical_boundary": _json_copy(
            release_controls["historical_import_boundary"]
        ),
        "excluded_sources": [
            HISTORICAL_SCOPE,
            "V2.11.4 scientific outcomes (none completed)",
            "V2.11.4 provider completions as V2.11.5 outcomes",
            "imported capability/P95 samples as current performance evidence",
            "diagnostic artifacts as scientific evidence",
            "raw prompts and raw provider outputs",
        ],
        "offline_detail_exception": (
            "five deterministic zero-provider candidate-admission details are "
            "copied after publication-time replay"
        ),
        "experiment_c_sensitivity_boundary": _json_copy(
            sensitivity_release
            if isinstance(sensitivity_release, Mapping)
            else {"pass": False, "available": False}
        ),
    }
    manifest_path = target / "package_manifest.json"
    _atomic_bytes(manifest_path, _pretty_bytes(manifest))
    checksum_paths = sorted(path for path in target.rglob("*") if path.is_file())
    checksums = {
        "schema_version": PILOT_CHECKSUM_SCHEMA_VERSION,
        "contract_sha256": contract.canonical_hash,
        "files": [
            {
                "path": path.relative_to(target).as_posix(),
                "sha256": _sha256_file(path),
                "byte_size": path.stat().st_size,
            }
            for path in checksum_paths
        ],
    }
    checksums_path = target / "checksums.json"
    _atomic_bytes(checksums_path, _pretty_bytes(checksums))
    for row in checksums["files"]:
        if _sha256_file(target / row["path"]) != row["sha256"]:
            raise PilotEvidenceError("V2.11.5 package checksum replay failed")
    package_bytes = sum(
        path.stat().st_size for path in target.rglob("*") if path.is_file()
    )
    if package_bytes + int(release_controls["budget"]["raw_root_storage_bytes"]) > int(
        contract.budgets["max_storage_bytes"]
    ):
        raise PilotEvidenceError("raw plus V2.11.5 package exceeds storage cap")
    return manifest_path, checksums_path, scientific_complete


def _new_package_target(build_root: str | Path, contract: PilotContract) -> Path:
    target = Path(build_root).resolve() / _evidence_namespace(contract)
    if target.exists():
        raise PilotEvidenceError(f"refusing to overwrite evidence package: {target}")
    return target


def _install_package_no_overwrite(temporary: Path, target: Path) -> None:
    """Reserve the final directory atomically and never replace another path.

    Python's directory ``os.replace`` can replace a target that appeared as an
    empty directory after the initial existence check.  ``mkdir`` with
    ``exist_ok=False`` is the portable no-replace reservation on both CI
    platforms.  The checksum receipt is installed last, so an interrupted
    installation is visibly incomplete and remains fail-closed on a retry.
    """

    try:
        target.mkdir()
    except FileExistsError as exc:
        raise PilotEvidenceError(
            f"refusing to overwrite evidence package: {target}"
        ) from exc
    # Manifest is useful only together with the checksum receipt, so both are
    # installed after all payloads and checksums.json is the final move.
    children = sorted(
        temporary.iterdir(),
        key=lambda path: (
            (
                2
                if path.name == "checksums.json"
                else 1 if path.name == "package_manifest.json" else 0
            ),
            path.name,
        ),
    )
    for source in children:
        destination = target / source.name
        if destination.exists() or destination.is_symlink():
            raise PilotEvidenceError(
                f"evidence installation destination already exists: {destination}"
            )
        os.rename(source, destination)
    temporary.rmdir()


def build_pilot_v2115_evidence_package(
    *,
    contract_path: str | Path,
    run_ledger_path: str | Path,
    raw_root: str | Path,
    build_root: str | Path,
    source_repo_root: str | Path | None = None,
) -> PilotEvidencePackage:
    """Build a zero-provider reviewer package from immutable V2.11.5 evidence."""

    source, contract_source, raw, ledger_path = _resolve_source_paths(
        source_repo_root=source_repo_root,
        contract_path=contract_path,
        raw_root=raw_root,
        run_ledger_path=run_ledger_path,
    )
    contract = load_pilot_contract(contract_source)
    _frozen_contract(contract)
    release_control, commit, paid = _validate_release(contract, raw_root=raw)
    source_provenance = _validate_source_git(source, contract, expected_commit=commit)
    code_root = Path(__file__).resolve().parents[1]
    publisher_provenance = _publisher_provenance(code_root)

    ledger = _strict_json_load(ledger_path)
    rows, denominator = _normalize_v2112_ledger(
        contract,
        ledger,
        raw_root=raw,
        expected_commit=commit,
        source_repo_root=source,
    )
    ledger_object = PilotRunLedger(
        ledger_path,
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    stage_receipts = _normalized_stage_receipts(
        contract,
        raw_root=raw,
        ledger=ledger_object,
        paid=paid,
        source_repo_root=source,
    )
    post_gate = _validated_post_gate(
        contract,
        source_repo_root=source,
        raw_root=raw,
        commit=commit,
        rows=rows,
    )
    budget = _validate_budget(
        contract,
        raw_root=raw,
        repo_root=source,
        rows=rows,
    )
    budget["schema_version"] = V2115_BUDGET_RECEIPT_SCHEMA_VERSION
    run_receipt = _run_ledger_receipt(
        contract,
        ledger,
        denominator,
        path=ledger_path,
    )
    run_receipt["schema_version"] = V2115_RUN_LEDGER_RECEIPT_SCHEMA_VERSION
    offline_audit, offline_payloads = _validate_offline_candidate_admission(
        contract,
        raw_root=raw,
        rows=rows,
        ledger=ledger,
    )

    gates: dict[str, Any] = {
        "experiment_a": _experiment_a_gate(contract, rows),
        "experiment_c": _experiment_c_gate(contract, rows),
        "experiment_d": _experiment_d_gate(contract, rows),
        "narrative": _narrative_gate(contract, rows),
    }
    capability = _v2115_capability_by_model(rows, contract)
    cross_model = _cross_model_summary(contract, rows, capability)
    experiment_b = _experiment_b_summary(rows)
    rule_sensitivity, sensitivity_control = (
        _validated_v2115_experiment_c_sensitivity(
            contract,
            raw_root=raw,
            rows=rows,
            common_commit=commit,
            source_repo_root=source,
            paid=paid,
        )
    )
    gates["experiment_c"] = _apply_c_sensitivity_no_go(
        gates["experiment_c"],
        sensitivity_control,
    )
    release_controls = {
        "pass": True,
        "resolved_git_commit": commit,
        "science_source": source_provenance,
        "publisher": publisher_provenance,
        "release": release_control,
        "run_ledger": run_receipt,
        "stage_receipts": stage_receipts,
        "post_gate": post_gate,
        "budget": budget,
        "experiment_c_sensitivity": sensitivity_control,
        "offline_candidate_admission": _json_copy(offline_audit),
        "historical_import_boundary": {
            "source_contract": "finevo-pilot-v2.11.4",
            "source_classification": "immutable-pre-dispatch-acceptance-no-go",
            "source_registered_cells": 136,
            "source_complete_operational_cells": 5,
            "source_scientific_complete_cells": 0,
            "source_fresh_provider_calls": 0,
            "current_zero_call_operational_import_cells": 5,
            "current_imported_effect_cells": 0,
            "historical_treatment_effect_cells_reused": 0,
            "scientific_evidence": False,
        },
    }

    target = _new_package_target(build_root, contract)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}-build-", dir=target.parent)
    )
    try:
        manifest, checksums, scientific_complete = _write_package(
            temporary,
            contract_path=contract_source,
            contract=contract,
            rows=rows,
            denominator=denominator,
            gates=gates,
            capability=capability,
            cross_model=cross_model,
            release_controls=release_controls,
            experiment_b=experiment_b,
            rule_sensitivity=rule_sensitivity,
            offline_audit=offline_audit,
            offline_payloads=offline_payloads,
            run_ledger=ledger,
            budget_ledger=_strict_json_load(raw / "budget_ledger.json"),
        )
        _install_package_no_overwrite(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return PilotEvidencePackage(
        package_dir=target,
        manifest_path=target / manifest.name,
        checksums_path=target / checksums.name,
        contract_hash=contract.canonical_hash,
        scientific_complete=scientific_complete,
        claim_gates=_json_copy(gates),
    )


__all__ = [
    "V2115_C_SENSITIVITY_DIAGNOSTIC_SCHEMA_VERSION",
    "V2115_CONTRACT_ID",
    "V2115_EVIDENCE_SCHEMA_VERSION",
    "build_pilot_v2115_evidence_package",
]
