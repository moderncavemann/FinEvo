"""Provider-free V2.11.5 executed-labor threshold diagnostic.

This module consumes the immutable, tagged V2.11.5 publication package.  It
does not import a provider client, read credentials, dispatch a model, retry a
failed cell, or alter the source package.  The diagnostic is retrospective and
descriptive: it reports rates of *executed labor actions* below three frozen
thresholds.  Those rates are not measurements of unemployment.

The source denominator is preserved exactly.  Experiments A and C contain 45
registered cells: 40 actor runs and five deterministic candidate-admission
cells for which an action metric is structurally not applicable.  Every actor
run contributes 48 registered action opportunities.  Failed actor runs remain
in the output with null threshold results and no imputation.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence


DIAGNOSTIC_ID = "pilot-v2.11.5-labor-threshold-sensitivity-v1"
DIAGNOSTIC_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.5-labor-threshold-sensitivity-diagnostic-v1"
)
SOURCE_PROVENANCE_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.5-labor-threshold-source-provenance-v1"
)
FAILURE_LEDGER_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.5-labor-threshold-failure-ledger-v1"
)
PACKAGE_MANIFEST_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.5-labor-threshold-package-manifest-v1"
)
PACKAGE_CHECKSUM_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.5-labor-threshold-package-checksums-v1"
)

SOURCE_PACKAGE_RELATIVE = Path("evidence/current_v2/pilot-v2.11.5")
OUTPUT_RELATIVE = Path(
    "evidence/current_v2/pilot-v2.11.5-labor-threshold-sensitivity-v1"
)
SOURCE_AGGREGATE_RELATIVE = SOURCE_PACKAGE_RELATIVE / "aggregate.json"
SOURCE_CHECKSUMS_RELATIVE = SOURCE_PACKAGE_RELATIVE / "checksums.json"
SOURCE_MANIFEST_RELATIVE = SOURCE_PACKAGE_RELATIVE / "package_manifest.json"
SCHEMA_RELATIVE = Path(
    "schemas/pilot_v2115_labor_threshold_sensitivity_v1.schema.json"
)

SOURCE_EVIDENCE_TAG = "pilot-v2.11.5-diagnostic-evidence-v1"
SOURCE_EVIDENCE_TAG_OBJECT = "7b83cf3953c4f59e3c79051f40c1e40456b92ef2"
SOURCE_EVIDENCE_COMMIT = "34134f2624833e45f0e1f559332b0d11ea1942d6"
SOURCE_SCIENCE_TAG = "pilot-v2.11.5-science"
SOURCE_SCIENCE_COMMIT = "2351ac2283f9fedb9dce70067174020be56ed9cc"
SOURCE_CONTRACT_ID = "finevo-pilot-v2.11.5"
SOURCE_CONTRACT_SHA256 = (
    "e1ecdec43e3f7a7b9a3d0977e2522d95861e826fc68781377d7eaceeb5e6e2ef"
)
SOURCE_AGGREGATE_SHA256 = (
    "5b50767e7e6f6f53aee8cc64f7f99a7c83a61cf8d57f28c73b0a205e30ac0c97"
)
SOURCE_CHECKSUMS_SHA256 = (
    "1c57592cc14689eee3ed9832996cacb3a4edde764647e1d704f5765fe2920576"
)
SOURCE_MANIFEST_SHA256 = (
    "99d4db05cc4dbfd2b9339f9034748396d36f49d6db46f91c2b73970388d3b333"
)

SEEDS = (1099057501, 1421875452, 1769977770, 959809858, 617806385)
A_ARMS = ("no-context", "prompt-only", "retrieval-only", "full")
C_DYNAMIC_ARMS = (
    "full",
    "unverified-dual",
    "verified-error-forced",
    "unverified-error-forced",
)
C_STRUCTURAL_ARM = "verified-error-candidate"
PER_ACTOR_RUN_DENOMINATOR = 48
EXPECTED_LABOR_GRID_STEP = 8

THRESHOLDS = (
    {
        "threshold_id": "h_lt_1",
        "upper_bound_exclusive_hours": 1,
        "included_frozen_grid_hours": [0],
    },
    {
        "threshold_id": "h_lt_20",
        "upper_bound_exclusive_hours": 20,
        "included_frozen_grid_hours": [0, 8, 16],
    },
    {
        "threshold_id": "h_lt_40",
        "upper_bound_exclusive_hours": 40,
        "included_frozen_grid_hours": [0, 8, 16, 24, 32],
    },
)

PAIRED_CONTRASTS = (
    {
        "contrast_id": "a_full_minus_prompt_only",
        "stage_id": "experiment-a",
        "left_arm": "full",
        "right_arm": "prompt-only",
    },
    {
        "contrast_id": "a_retrieval_only_minus_no_context",
        "stage_id": "experiment-a",
        "left_arm": "retrieval-only",
        "right_arm": "no-context",
    },
    {
        "contrast_id": "c_full_minus_unverified_dual",
        "stage_id": "experiment-c",
        "left_arm": "full",
        "right_arm": "unverified-dual",
    },
    {
        "contrast_id": "c_verified_error_forced_minus_unverified_error_forced",
        "stage_id": "experiment-c",
        "left_arm": "verified-error-forced",
        "right_arm": "unverified-error-forced",
    },
)

_REQUIRED_PUBLISHER_FILES = (
    "docs/v2115_labor_threshold_sensitivity_v1.md",
    "schemas/pilot_v2115_labor_threshold_sensitivity_v1.schema.json",
    "scripts/build_v2115_labor_threshold_sensitivity.py",
    "tests/test_v2115_labor_threshold_sensitivity.py",
    "verified_memory/v2115_labor_threshold_sensitivity.py",
)

_OUTPUT_FILES = (
    "labor_threshold_sensitivity.json",
    "per_run.csv",
    "paired_contrasts.csv",
    "failure_ledger.json",
    "source_provenance.json",
    "schema.json",
    "report.md",
    "package_manifest.json",
    "checksums.json",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?:^|[\s\"'`(])/(?:Users|private|tmp|root|home|opt|var)/"),
    re.compile(r"file://", re.IGNORECASE),
    re.compile(r"(?:^|[\s\"'`(])[A-Za-z]:\\\\"),
)
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bsk-or-v1-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(
        r"(?i)(?:api[_-]?key|authorization|bearer|secret)\s*[:=]\s*[\"']?[^\s\"']{12,}"
    ),
)


class LaborThresholdDiagnosticError(RuntimeError):
    """Raised when a source or diagnostic invariant is not satisfied."""


@dataclass(frozen=True)
class LaborThresholdPackage:
    package_dir: Path
    diagnostic_path: Path
    checksums_path: Path
    content_sha256: str


def _reject_constant(value: str) -> None:
    raise LaborThresholdDiagnosticError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LaborThresholdDiagnosticError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_load(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_constant,
        )
    except LaborThresholdDiagnosticError:
        raise
    except Exception as exc:
        raise LaborThresholdDiagnosticError(f"invalid JSON at {path.name}: {exc}") from exc


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LaborThresholdDiagnosticError(f"value is not canonical JSON: {exc}") from exc


def _pretty_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LaborThresholdDiagnosticError(f"value is not finite JSON: {exc}") from exc


def _content_sha256(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LaborThresholdDiagnosticError(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise LaborThresholdDiagnosticError(f"{name} must be an array")
    return value


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LaborThresholdDiagnosticError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise LaborThresholdDiagnosticError(f"{name} must be >= {minimum}")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LaborThresholdDiagnosticError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LaborThresholdDiagnosticError(f"{name} must be finite")
    return result


def _git(repo_root: Path, *args: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "LANG": "C",
            "LC_ALL": "C",
        },
    )
    if check and process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise LaborThresholdDiagnosticError(
            f"git {' '.join(args)} failed: {detail}"
        )
    return process.stdout.strip()


def _assert_no_symlink_component(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise LaborThresholdDiagnosticError("path escaped repository root") from exc
    current = root
    if current.is_symlink():
        raise LaborThresholdDiagnosticError("repository root cannot be a symlink")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise LaborThresholdDiagnosticError(
                f"symlinked source/output component is forbidden: {relative.as_posix()}"
            )


def _validate_regular_file(root: Path, relative: str | Path) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise LaborThresholdDiagnosticError(f"unsafe relative path: {rel}")
    path = root / rel
    _assert_no_symlink_component(root, path)
    if not path.is_file():
        raise LaborThresholdDiagnosticError(f"required regular file is absent: {rel}")
    return path


def _tracked_blob_inventory(
    repo_root: Path,
    *,
    commit: str,
    relatives: Iterable[str],
) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for relative in sorted(set(relatives)):
        path = _validate_regular_file(repo_root, relative)
        expected_blob = _git(repo_root, "rev-parse", f"{commit}:{relative}")
        actual_blob = _git(repo_root, "hash-object", "--", relative)
        if actual_blob != expected_blob:
            raise LaborThresholdDiagnosticError(
                f"tracked file differs from {commit}: {relative}"
            )
        inventory[relative] = {
            "git_blob_sha1": expected_blob,
            "sha256": _sha256_file(path),
            "byte_size": path.stat().st_size,
        }
    return inventory


def _validate_source_git_anchor(repo_root: Path) -> dict[str, Any]:
    if _git(repo_root, "cat-file", "-t", SOURCE_EVIDENCE_TAG) != "tag":
        raise LaborThresholdDiagnosticError("source evidence tag is not annotated")
    tag_object = _git(repo_root, "rev-parse", SOURCE_EVIDENCE_TAG)
    tag_commit = _git(repo_root, "rev-parse", f"{SOURCE_EVIDENCE_TAG}^{{}}")
    if tag_object != SOURCE_EVIDENCE_TAG_OBJECT or tag_commit != SOURCE_EVIDENCE_COMMIT:
        raise LaborThresholdDiagnosticError("source evidence tag identity drifted")
    head = _git(repo_root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", tag_commit, head],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestor.returncode != 0:
        raise LaborThresholdDiagnosticError("source evidence tag is not an ancestor of HEAD")
    diff = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "diff",
            "--quiet",
            tag_commit,
            "--",
            SOURCE_PACKAGE_RELATIVE.as_posix(),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if diff.returncode != 0:
        raise LaborThresholdDiagnosticError(
            "tagged V2.11.5 source publication differs at current HEAD"
        )
    return {
        "git_tag": SOURCE_EVIDENCE_TAG,
        "git_tag_object": tag_object,
        "git_commit": tag_commit,
    }


def _publisher_provenance(repo_root: Path) -> dict[str, Any]:
    top = Path(_git(repo_root, "rev-parse", "--show-toplevel"))
    if top != repo_root:
        raise LaborThresholdDiagnosticError("publisher root must be the exact Git root")
    dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise LaborThresholdDiagnosticError(
            "publisher worktree must be clean before evidence generation"
        )
    commit = _git(repo_root, "rev-parse", "HEAD")
    if not _HEX40.fullmatch(commit):
        raise LaborThresholdDiagnosticError("publisher commit is invalid")
    return {
        "git_commit": commit,
        "tracked_worktree_clean": True,
        "required_tracked_files": _tracked_blob_inventory(
            repo_root,
            commit=commit,
            relatives=_REQUIRED_PUBLISHER_FILES,
        ),
        "provider_calls": 0,
        "hosted_cost_usd": 0.0,
        "credential_reads": 0,
    }


def _validate_source_checksums(repo_root: Path) -> dict[str, Any]:
    source_root = repo_root / SOURCE_PACKAGE_RELATIVE
    _assert_no_symlink_component(repo_root, source_root)
    if not source_root.is_dir():
        raise LaborThresholdDiagnosticError("V2.11.5 source package is absent")
    checksum_path = _validate_regular_file(repo_root, SOURCE_CHECKSUMS_RELATIVE)
    if _sha256_file(checksum_path) != SOURCE_CHECKSUMS_SHA256:
        raise LaborThresholdDiagnosticError("source checksums.json hash drifted")
    checksums = _mapping(_strict_json_load(checksum_path), "source checksums")
    if checksums.get("schema_version") != "finevo-pilot-package-checksums-v1":
        raise LaborThresholdDiagnosticError("source checksum schema drifted")
    rows = _list(checksums.get("files"), "source checksum files")
    records: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(rows):
        record = _mapping(raw, f"source checksum row {index}")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative or relative in records:
            raise LaborThresholdDiagnosticError("source checksum paths must be unique")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise LaborThresholdDiagnosticError("source checksum path escaped package")
        digest = record.get("sha256")
        byte_size = record.get("byte_size")
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise LaborThresholdDiagnosticError("source checksum digest is invalid")
        _integer(byte_size, "source checksum byte_size", minimum=0)
        path = _validate_regular_file(source_root, relative)
        if path.stat().st_size != byte_size or _sha256_file(path) != digest:
            raise LaborThresholdDiagnosticError(
                f"source package checksum mismatch: {relative}"
            )
        records[relative] = record
    actual = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    expected = set(records) | {"checksums.json"}
    if actual != expected:
        raise LaborThresholdDiagnosticError("source package file inventory drifted")
    aggregate_record = records.get("aggregate.json")
    manifest_record = records.get("package_manifest.json")
    if (
        aggregate_record is None
        or aggregate_record.get("sha256") != SOURCE_AGGREGATE_SHA256
        or manifest_record is None
        or manifest_record.get("sha256") != SOURCE_MANIFEST_SHA256
    ):
        raise LaborThresholdDiagnosticError("source anchor files are not checksum-bound")
    return {
        "schema_version": checksums["schema_version"],
        "file_count_including_checksums": len(expected),
        "checksums_sha256": SOURCE_CHECKSUMS_SHA256,
        "checksums_content_sha256": _content_sha256(checksums),
        "aggregate_sha256": SOURCE_AGGREGATE_SHA256,
        "aggregate_byte_size": aggregate_record["byte_size"],
        "package_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "package_manifest_byte_size": manifest_record["byte_size"],
        "inventory_content_sha256": _content_sha256(rows),
    }


def _validate_source_manifest(repo_root: Path) -> Mapping[str, Any]:
    path = _validate_regular_file(repo_root, SOURCE_MANIFEST_RELATIVE)
    if _sha256_file(path) != SOURCE_MANIFEST_SHA256:
        raise LaborThresholdDiagnosticError("source package manifest hash drifted")
    manifest = _mapping(_strict_json_load(path), "source package manifest")
    expected = {
        "contract_id": SOURCE_CONTRACT_ID,
        "contract_sha256": SOURCE_CONTRACT_SHA256,
        "pilot_tag": SOURCE_SCIENCE_TAG,
        "resolved_git_commit": SOURCE_SCIENCE_COMMIT,
        "evidence_namespace": "current_v2/pilot-v2.11.5",
        "scientific_complete": False,
        "scientific_matrix_complete": False,
        "scientific_claim_gates_supported": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise LaborThresholdDiagnosticError(
                f"source package manifest field drifted: {key}"
            )
    return manifest


def _source_row_hash(row: Mapping[str, Any]) -> str:
    return _content_sha256(row)


def _expected_run_id(stage: str, arm: str, seed: int) -> str:
    return (
        f"finevo-pilot-v2.11.5--{stage}--gpt52_main--{arm}--none--"
        f"stage0-selected--s{seed}"
    )


def _expected_cell_keys() -> set[tuple[str, str, int]]:
    keys = {
        ("experiment-a", arm, seed)
        for arm in A_ARMS
        for seed in SEEDS
    }
    keys.update(
        ("experiment-c", arm, seed)
        for arm in (*C_DYNAMIC_ARMS, C_STRUCTURAL_ARM)
        for seed in SEEDS
    )
    return keys


def _validate_common_row(
    row: Mapping[str, Any],
    *,
    stage: str,
    arm: str,
    seed: int,
) -> None:
    expected = {
        "contract_id": SOURCE_CONTRACT_ID,
        "stage_id": stage,
        "arm_id": arm,
        "model_id": "gpt52_main",
        "requested_model": "gpt-5.2-2025-12-11",
        "environment_seed": seed,
        "decoding_seed": None,
        "narrative_id": "none",
        "utility_profile_id": "stage0-selected",
        "num_agents": 4,
        "episode_length": 12,
        "shock_id": "registered-rate-shock",
        "budget_bucket": "hosted_v2115",
        "run_id": _expected_run_id(stage, arm, seed),
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise LaborThresholdDiagnosticError(
                f"source row field drifted for {stage}/{arm}/{seed}: {key}"
            )


def _labor_histogram(row: Mapping[str, Any]) -> dict[str, int]:
    metrics = _mapping(row.get("metrics"), "source row metrics")
    actions = _mapping(metrics.get("actions"), "source row action metrics")
    raw_counts = _mapping(actions.get("labor_hours_counts"), "labor_hours_counts")
    normalized: dict[int, int] = {}
    for raw_hour, raw_count in raw_counts.items():
        try:
            numeric = float(raw_hour)
        except (TypeError, ValueError) as exc:
            raise LaborThresholdDiagnosticError("labor-hour key is not numeric") from exc
        if not numeric.is_integer():
            raise LaborThresholdDiagnosticError("labor-hour key must be integral")
        hour = int(numeric)
        if (
            hour < 0
            or hour % EXPECTED_LABOR_GRID_STEP != 0
            or str(hour) != str(raw_hour)
        ):
            raise LaborThresholdDiagnosticError(f"labor-hour key is off-grid: {raw_hour}")
        count = _integer(raw_count, f"labor count at {hour}", minimum=0)
        if hour in normalized:
            raise LaborThresholdDiagnosticError("duplicate normalized labor-hour bin")
        normalized[hour] = count
    if sum(normalized.values()) != PER_ACTOR_RUN_DENOMINATOR:
        raise LaborThresholdDiagnosticError("complete actor histogram does not sum to 48")
    return {str(hour): normalized[hour] for hour in sorted(normalized)}


def _threshold_values(histogram: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for threshold in THRESHOLDS:
        threshold_id = str(threshold["threshold_id"])
        upper = _integer(
            threshold["upper_bound_exclusive_hours"],
            "threshold upper bound",
            minimum=0,
        )
        count = sum(
            value for hour, value in histogram.items() if int(hour) < upper
        )
        values[threshold_id] = {
            "below_threshold_count": count,
            "registered_action_denominator": PER_ACTOR_RUN_DENOMINATOR,
            "rate": count / PER_ACTOR_RUN_DENOMINATOR,
        }
    return values


def _validate_aggregate(aggregate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    expected = {
        "schema_version": "finevo-pilot-v2.11.5-evidence-package-v1",
        "contract_id": SOURCE_CONTRACT_ID,
        "contract_sha256": SOURCE_CONTRACT_SHA256,
        "pilot_tag": SOURCE_SCIENCE_TAG,
        "resolved_git_commit": SOURCE_SCIENCE_COMMIT,
        "evidence_namespace": "current_v2/pilot-v2.11.5",
    }
    for key, value in expected.items():
        if aggregate.get(key) != value:
            raise LaborThresholdDiagnosticError(f"source aggregate field drifted: {key}")
    rows = _list(aggregate.get("rows"), "source aggregate rows")
    if len(rows) != 136:
        raise LaborThresholdDiagnosticError("source aggregate denominator is not 136")
    run_ids: set[str] = set()
    selected: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"source row {index}")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in run_ids:
            raise LaborThresholdDiagnosticError("source run IDs must be unique strings")
        run_ids.add(run_id)
        stage_value = row.get("stage_id")
        if stage_value not in {"experiment-a", "experiment-c"}:
            continue
        if not isinstance(stage_value, str):
            raise LaborThresholdDiagnosticError("selected source stage is malformed")
        stage = stage_value
        arm = row.get("arm_id")
        seed = row.get("environment_seed")
        if not isinstance(arm, str) or isinstance(seed, bool) or not isinstance(seed, int):
            raise LaborThresholdDiagnosticError("selected source row key is malformed")
        cell_key = (stage, arm, seed)
        if cell_key in selected:
            raise LaborThresholdDiagnosticError("selected source cell is duplicated")
        selected[cell_key] = row
    if set(selected) != _expected_cell_keys():
        raise LaborThresholdDiagnosticError("A/C registered cell inventory drifted")

    for (stage, arm, seed), row in selected.items():
        _validate_common_row(row, stage=stage, arm=arm, seed=seed)
        structural = stage == "experiment-c" and arm == C_STRUCTURAL_ARM
        if structural:
            if (
                row.get("execution_mode") != "offline_candidate_admission"
                or row.get("status") != "complete"
                or row.get("artifact_kind") != "terminal-summary"
                or row.get("failure") is not None
            ):
                raise LaborThresholdDiagnosticError(
                    "candidate-admission structural cell status drifted"
                )
            metrics = _mapping(row.get("metrics"), "candidate metrics")
            if "actions" in metrics:
                raise LaborThresholdDiagnosticError(
                    "candidate-admission cell unexpectedly exposes action metrics"
                )
            continue
        if row.get("execution_mode") != "actor_run":
            raise LaborThresholdDiagnosticError("dynamic cell is not an actor run")
        status = row.get("status")
        if status == "complete":
            if (
                row.get("scientific_eligible") is not True
                or row.get("artifact_kind") != "verified-run-manifest"
                or row.get("failure") is not None
            ):
                raise LaborThresholdDiagnosticError("complete actor cell identity drifted")
            _labor_histogram(row)
        elif status == "failed":
            if (
                row.get("scientific_eligible") is not False
                or row.get("artifact_kind") != "failure-audit-artifact"
                or not isinstance(row.get("failure"), Mapping)
                or row.get("metrics") != {}
            ):
                raise LaborThresholdDiagnosticError("failed actor cell boundary drifted")
        else:
            raise LaborThresholdDiagnosticError("A/C actor cell is not terminal")

    statuses = Counter(
        row["status"]
        for (stage, arm, _), row in selected.items()
        if not (stage == "experiment-c" and arm == C_STRUCTURAL_ARM)
    )
    if statuses != {"complete": 37, "failed": 3}:
        raise LaborThresholdDiagnosticError("actor completion denominator drifted")
    expected_by_arm = {
        ("experiment-a", "full"): {"complete": 5},
        ("experiment-a", "prompt-only"): {"complete": 5},
        ("experiment-a", "no-context"): {"complete": 4, "failed": 1},
        ("experiment-a", "retrieval-only"): {"complete": 3, "failed": 2},
        **{
            ("experiment-c", arm): {"complete": 5}
            for arm in C_DYNAMIC_ARMS
        },
    }
    for arm_key, expected_counts in expected_by_arm.items():
        observed = Counter(
            row["status"]
            for selected_key, row in selected.items()
            if selected_key[:2] == arm_key
        )
        if dict(observed) != expected_counts:
            raise LaborThresholdDiagnosticError(
                f"source status inventory drifted for {arm_key[0]}/{arm_key[1]}"
            )
    return [selected[key] for key in sorted(selected, key=_cell_sort_key)]


def _cell_sort_key(key: tuple[str, str, int]) -> tuple[int, int, int]:
    stage, arm, seed = key
    stage_index = 0 if stage == "experiment-a" else 1
    arms = A_ARMS if stage == "experiment-a" else (*C_DYNAMIC_ARMS, C_STRUCTURAL_ARM)
    return stage_index, arms.index(arm), SEEDS.index(seed)


def _source_claim_boundaries(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    gates = _mapping(aggregate.get("claim_gates"), "source claim gates")
    a = _mapping(gates.get("experiment_a"), "Experiment A source gate")
    c = _mapping(gates.get("experiment_c"), "Experiment C source gate")
    primary = _mapping(a.get("primary_contrast"), "Experiment A primary contrast")
    threshold_gate = _mapping(a.get("threshold_gate"), "Experiment A threshold gate")
    a_boundary = {
        "status": a.get("status"),
        "support_retrieval_effect": a.get("support_retrieval_effect"),
        "scientific_evidence_complete": a.get("scientific_evidence_complete"),
        "primary_pair_count": primary.get("pair_count"),
        "primary_direction_count": primary.get("direction_count"),
        "primary_median_relative_effect": primary.get("median_relative_effect"),
        "threshold_checks": threshold_gate.get("checks"),
    }
    c_boundary = {
        "status": c.get("status"),
        "support_rule_reliability": c.get("support_rule_reliability"),
        "scientific_evidence_complete": c.get("scientific_evidence_complete"),
        "reasons": c.get("reasons"),
    }
    if a_boundary != {
        "status": "no-go",
        "support_retrieval_effect": False,
        "scientific_evidence_complete": False,
        "primary_pair_count": 5,
        "primary_direction_count": 3,
        "primary_median_relative_effect": 0.030620326693089294,
        "threshold_checks": {
            "at_least_four_complete_pairs": True,
            "at_least_four_same_direction": False,
            "median_relative_effect_at_least_5pct": False,
        },
    }:
        raise LaborThresholdDiagnosticError("Experiment A no-go boundary drifted")
    if (
        c_boundary["status"] != "no-go"
        or c_boundary["support_rule_reliability"] is not False
        or c_boundary["scientific_evidence_complete"] is not False
        or c_boundary["reasons"]
        != [
            "preregistered zero-API rule sensitivity was not sealed by the authoritative Experiment C stage"
        ]
    ):
        raise LaborThresholdDiagnosticError("Experiment C no-go boundary drifted")
    return {"experiment_a": a_boundary, "experiment_c": c_boundary}


def _run_records(selected_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in selected_rows:
        stage = str(row["stage_id"])
        arm = str(row["arm_id"])
        seed = int(row["environment_seed"])
        structural = stage == "experiment-c" and arm == C_STRUCTURAL_ARM
        base: dict[str, Any] = {
            "stage_id": stage,
            "arm_id": arm,
            "seed": seed,
            "run_id": row["run_id"],
            "source_status": row["status"],
            "source_scientific_eligible": row["scientific_eligible"],
            "source_artifact_kind": row["artifact_kind"],
            "source_artifact_sha256": row["artifact_sha256"],
            "source_row_content_sha256": _source_row_hash(row),
            "metric_name": "below-threshold executed labor action rate",
        }
        if structural:
            base.update(
                {
                    "cell_class": "structural-not-applicable",
                    "action_metric_applicable": False,
                    "registered_action_denominator": None,
                    "observed_action_count": None,
                    "missing_action_count": None,
                    "labor_hours_counts": None,
                    "threshold_results": None,
                    "missing_reason": (
                        "deterministic candidate-admission cell has no actor action stream"
                    ),
                    "failure": None,
                }
            )
        elif row["status"] == "complete":
            histogram = _labor_histogram(row)
            base.update(
                {
                    "cell_class": "actor-run",
                    "action_metric_applicable": True,
                    "registered_action_denominator": PER_ACTOR_RUN_DENOMINATOR,
                    "observed_action_count": PER_ACTOR_RUN_DENOMINATOR,
                    "missing_action_count": 0,
                    "labor_hours_counts": histogram,
                    "threshold_results": _threshold_values(histogram),
                    "missing_reason": None,
                    "failure": None,
                }
            )
        else:
            failure = _mapping(row["failure"], "failed source row failure")
            base.update(
                {
                    "cell_class": "actor-run",
                    "action_metric_applicable": True,
                    "registered_action_denominator": PER_ACTOR_RUN_DENOMINATOR,
                    "observed_action_count": 0,
                    "missing_action_count": PER_ACTOR_RUN_DENOMINATOR,
                    "labor_hours_counts": None,
                    "threshold_results": None,
                    "missing_reason": "source actor run failed; threshold values are null",
                    "failure": {
                        "error_type": failure.get("error_type"),
                        "message_sha256": failure.get("message_sha256"),
                        "message_bytes": failure.get("message_bytes"),
                        "message_truncated": failure.get("message_truncated"),
                    },
                }
            )
        records.append(base)
    return records


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "range": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "range": [min(values), max(values)],
    }


def _arm_summaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    arm_order = [
        *(('experiment-a', arm) for arm in A_ARMS),
        *(('experiment-c', arm) for arm in C_DYNAMIC_ARMS),
    ]
    for stage, arm in arm_order:
        group = [
            record
            for record in records
            if record["stage_id"] == stage and record["arm_id"] == arm
        ]
        by_seed = {int(record["seed"]): record for record in group}
        if set(by_seed) != set(SEEDS):
            raise LaborThresholdDiagnosticError("arm summary seed inventory drifted")
        complete = [seed for seed in SEEDS if by_seed[seed]["source_status"] == "complete"]
        failed = [seed for seed in SEEDS if by_seed[seed]["source_status"] == "failed"]
        threshold_summaries: dict[str, Any] = {}
        for threshold in THRESHOLDS:
            threshold_id = str(threshold["threshold_id"])
            rates_by_seed = {
                str(seed): (
                    by_seed[seed]["threshold_results"][threshold_id]["rate"]
                    if seed in complete
                    else None
                )
                for seed in SEEDS
            }
            observed_values: list[float] = []
            for complete_seed in complete:
                observed_values.append(
                    _finite_number(
                        rates_by_seed[str(complete_seed)],
                        "complete seed/run threshold rate",
                    )
                )
            threshold_summaries[threshold_id] = {
                "rates_by_seed": rates_by_seed,
                "summary_over_complete_seed_runs": _summary(observed_values),
            }
        summaries.append(
            {
                "stage_id": stage,
                "arm_id": arm,
                "registered_seed_count": 5,
                "complete_seed_count": len(complete),
                "failed_seed_count": len(failed),
                "complete_seeds": complete,
                "failed_seeds": failed,
                "registered_action_opportunities": 5 * PER_ACTOR_RUN_DENOMINATOR,
                "observed_action_count": len(complete) * PER_ACTOR_RUN_DENOMINATOR,
                "missing_action_count": len(failed) * PER_ACTOR_RUN_DENOMINATOR,
                "inference_unit": "seed/run",
                "thresholds": threshold_summaries,
            }
        )
    return summaries


def _paired_contrasts(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (record["stage_id"], record["arm_id"], int(record["seed"])): record
        for record in records
        if record["cell_class"] == "actor-run"
    }
    output: list[dict[str, Any]] = []
    for spec in PAIRED_CONTRASTS:
        stage = spec["stage_id"]
        left_arm = spec["left_arm"]
        right_arm = spec["right_arm"]
        usable: list[int] = []
        excluded: dict[str, Any] = {}
        for seed in SEEDS:
            left = index[(stage, left_arm, seed)]
            right = index[(stage, right_arm, seed)]
            if left["source_status"] == right["source_status"] == "complete":
                usable.append(seed)
            else:
                excluded[str(seed)] = {
                    "left_status": left["source_status"],
                    "right_status": right["source_status"],
                    "paired_delta": None,
                }
        thresholds: dict[str, Any] = {}
        for threshold in THRESHOLDS:
            threshold_id = str(threshold["threshold_id"])
            deltas = {
                str(seed): (
                    index[(stage, left_arm, seed)]["threshold_results"][threshold_id][
                        "rate"
                    ]
                    - index[(stage, right_arm, seed)]["threshold_results"][threshold_id][
                        "rate"
                    ]
                )
                for seed in usable
            }
            thresholds[threshold_id] = {
                "raw_paired_deltas_by_seed": deltas,
                "summary_over_complete_pairs": _summary(list(deltas.values())),
            }
        output.append(
            {
                **spec,
                "delta_definition": "left_arm rate minus right_arm rate",
                "registered_pair_count": 5,
                "complete_pair_count": len(usable),
                "complete_pair_seeds": usable,
                "excluded_pair_count": len(excluded),
                "excluded_pairs": excluded,
                "inference_unit": "paired seed/run",
                "thresholds": thresholds,
            }
        )
    return output


def _denominator(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    actor = [record for record in records if record["cell_class"] == "actor-run"]
    structural = [
        record for record in records if record["cell_class"] == "structural-not-applicable"
    ]
    complete = [record for record in actor if record["source_status"] == "complete"]
    failed = [record for record in actor if record["source_status"] == "failed"]
    denominator = {
        "registered_a_c_cell_count": len(records),
        "registered_actor_run_count": len(actor),
        "structural_not_applicable_cell_count": len(structural),
        "complete_actor_run_count": len(complete),
        "failed_actor_run_count": len(failed),
        "per_actor_run_registered_action_denominator": PER_ACTOR_RUN_DENOMINATOR,
        "registered_actor_action_opportunities": sum(
            int(record["registered_action_denominator"]) for record in actor
        ),
        "observed_actor_action_count": sum(
            int(record["observed_action_count"]) for record in actor
        ),
        "missing_actor_action_count": sum(
            int(record["missing_action_count"]) for record in actor
        ),
        "itt_failure_policy": (
            "all registered actor runs retained; failed runs have null threshold "
            "values and contribute 48 missing actions; no imputation"
        ),
        "structural_na_policy": (
            "candidate-admission cells are retained in the 45-cell inventory but "
            "excluded from the actor action-opportunity denominator"
        ),
    }
    expected = {
        "registered_a_c_cell_count": 45,
        "registered_actor_run_count": 40,
        "structural_not_applicable_cell_count": 5,
        "complete_actor_run_count": 37,
        "failed_actor_run_count": 3,
        "per_actor_run_registered_action_denominator": 48,
        "registered_actor_action_opportunities": 1920,
        "observed_actor_action_count": 1776,
        "missing_actor_action_count": 144,
    }
    for key, value in expected.items():
        if denominator[key] != value:
            raise LaborThresholdDiagnosticError(f"diagnostic denominator drifted: {key}")
    return denominator


def _selected_projection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "stage_id": row["stage_id"],
            "arm_id": row["arm_id"],
            "environment_seed": row["environment_seed"],
            "run_id": row["run_id"],
            "status": row["status"],
            "artifact_kind": row["artifact_kind"],
            "artifact_sha256": row["artifact_sha256"],
            "scientific_eligible": row["scientific_eligible"],
            "labor_hours_counts": (
                row.get("metrics", {}).get("actions", {}).get("labor_hours_counts")
                if isinstance(row.get("metrics"), Mapping)
                else None
            ),
            "source_row_content_sha256": _source_row_hash(row),
        }
        for row in rows
    ]


def _build_diagnostic_payload(
    aggregate: Mapping[str, Any],
    *,
    source_provenance: Mapping[str, Any],
    publisher_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    selected = _validate_aggregate(aggregate)
    records = _run_records(selected)
    payload: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "title": "V2.11.5 below-threshold executed labor action sensitivity",
        "classification": {
            "retrospective": True,
            "descriptive": True,
            "diagnostic_only": True,
            "publication_time_analysis": True,
            "preregistered": False,
            "scientific_evidence": False,
            "provider_calls": 0,
            "hosted_cost_usd": 0.0,
            "credential_reads": 0,
        },
        "metric": {
            "name": "below-threshold executed labor action rate",
            "formula": (
                "count(executed labor_hours < h) / 48 for each complete seed/run"
            ),
            "range": [0.0, 1.0],
            "direction": (
                "descriptive only; lower values mean fewer executed actions below "
                "the stated threshold, not lower unemployment"
            ),
            "aggregation": (
                "compute per complete seed/run, then report raw seed values and "
                "unweighted mean, median, and range across complete seed/runs"
            ),
            "missing_values": (
                "failed seed/runs remain null; no imputation or seed replacement"
            ),
            "inference_unit": "seed/run",
            "not_unemployment": True,
            "non_claim": (
                "This action-frequency diagnostic is not an unemployment rate, "
                "employment-state estimate, causal effect, or effectiveness result."
            ),
        },
        "thresholds": [dict(threshold) for threshold in THRESHOLDS],
        "denominator": _denominator(records),
        "run_records": records,
        "arm_summaries": _arm_summaries(records),
        "paired_contrasts": _paired_contrasts(records),
        "source_claim_boundaries": _source_claim_boundaries(aggregate),
        "claim_boundary": (
            "This zero-provider retrospective diagnostic cannot restore or reverse "
            "the authoritative Experiment A retrieval-effect no-go or Experiment C "
            "rule-reliability no-go."
        ),
        "source_binding": {
            "source_package": SOURCE_PACKAGE_RELATIVE.as_posix(),
            "source_aggregate_sha256": source_provenance["source_package"][
                "aggregate_sha256"
            ],
            "selected_row_count": len(selected),
            "selected_row_projection_sha256": _content_sha256(
                _selected_projection(selected)
            ),
        },
        "publisher_binding": {
            "git_commit": publisher_provenance["git_commit"],
            "provider_calls": 0,
            "hosted_cost_usd": 0.0,
            "credential_reads": 0,
        },
    }
    payload["content_sha256"] = _content_sha256(payload)
    return payload


def _validate_diagnostic_internal(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "diagnostic_id",
        "title",
        "classification",
        "metric",
        "thresholds",
        "denominator",
        "run_records",
        "arm_summaries",
        "paired_contrasts",
        "source_claim_boundaries",
        "claim_boundary",
        "source_binding",
        "publisher_binding",
        "content_sha256",
    }
    if set(payload) != required:
        raise LaborThresholdDiagnosticError("diagnostic top-level keys drifted")
    if payload.get("schema_version") != DIAGNOSTIC_SCHEMA_VERSION:
        raise LaborThresholdDiagnosticError("diagnostic schema version drifted")
    if payload.get("diagnostic_id") != DIAGNOSTIC_ID:
        raise LaborThresholdDiagnosticError("diagnostic ID drifted")
    classification = _mapping(payload.get("classification"), "classification")
    expected_flags = {
        "retrospective": True,
        "descriptive": True,
        "diagnostic_only": True,
        "publication_time_analysis": True,
        "preregistered": False,
        "scientific_evidence": False,
        "provider_calls": 0,
        "hosted_cost_usd": 0.0,
        "credential_reads": 0,
    }
    if dict(classification) != expected_flags:
        raise LaborThresholdDiagnosticError("diagnostic classification drifted")
    metric = _mapping(payload.get("metric"), "metric")
    if (
        metric.get("name") != "below-threshold executed labor action rate"
        or metric.get("not_unemployment") is not True
        or "unemployment rate" not in str(metric.get("non_claim"))
    ):
        raise LaborThresholdDiagnosticError("metric boundary drifted")
    if payload.get("thresholds") != [dict(value) for value in THRESHOLDS]:
        raise LaborThresholdDiagnosticError("threshold grid drifted")
    records = _list(payload.get("run_records"), "run_records")
    if len(records) != 45:
        raise LaborThresholdDiagnosticError("diagnostic must retain 45 A/C cells")
    keys: set[tuple[str, str, int]] = set()
    for index, raw in enumerate(records):
        record = _mapping(raw, f"run record {index}")
        key = (record.get("stage_id"), record.get("arm_id"), record.get("seed"))
        if key in keys:
            raise LaborThresholdDiagnosticError("diagnostic run record duplicated")
        keys.add(key)  # type: ignore[arg-type]
        if record.get("metric_name") != "below-threshold executed labor action rate":
            raise LaborThresholdDiagnosticError("run metric name drifted")
        if not isinstance(record.get("source_row_content_sha256"), str) or not _HEX64.fullmatch(
            str(record.get("source_row_content_sha256"))
        ):
            raise LaborThresholdDiagnosticError("source row binding is invalid")
        cell_class = record.get("cell_class")
        if cell_class == "actor-run" and record.get("source_status") == "complete":
            histogram = _mapping(record.get("labor_hours_counts"), "record histogram")
            normalized = {str(key): _integer(value, "record labor count", minimum=0) for key, value in histogram.items()}
            if sum(normalized.values()) != 48:
                raise LaborThresholdDiagnosticError("record histogram denominator drifted")
            expected_thresholds = _threshold_values(normalized)
            if record.get("threshold_results") != expected_thresholds:
                raise LaborThresholdDiagnosticError("record threshold computation drifted")
            if (
                record.get("observed_action_count") != 48
                or record.get("missing_action_count") != 0
                or record.get("failure") is not None
            ):
                raise LaborThresholdDiagnosticError("complete record denominator drifted")
        elif cell_class == "actor-run" and record.get("source_status") == "failed":
            if (
                record.get("threshold_results") is not None
                or record.get("labor_hours_counts") is not None
                or record.get("observed_action_count") != 0
                or record.get("missing_action_count") != 48
                or not isinstance(record.get("failure"), Mapping)
            ):
                raise LaborThresholdDiagnosticError("failed record was imputed")
        elif cell_class == "structural-not-applicable":
            if any(
                record.get(key) is not None
                for key in (
                    "registered_action_denominator",
                    "observed_action_count",
                    "missing_action_count",
                    "labor_hours_counts",
                    "threshold_results",
                )
            ):
                raise LaborThresholdDiagnosticError("structural N/A was quantified")
        else:
            raise LaborThresholdDiagnosticError("run record class/status is invalid")
    if keys != _expected_cell_keys():
        raise LaborThresholdDiagnosticError("diagnostic cell inventory drifted")
    if payload.get("denominator") != _denominator(records):
        raise LaborThresholdDiagnosticError("diagnostic denominator is not reproducible")
    if payload.get("arm_summaries") != _arm_summaries(records):
        raise LaborThresholdDiagnosticError("arm summaries are not reproducible")
    if payload.get("paired_contrasts") != _paired_contrasts(records):
        raise LaborThresholdDiagnosticError("paired contrasts are not reproducible")
    without_hash = dict(payload)
    observed_hash = without_hash.pop("content_sha256", None)
    if observed_hash != _content_sha256(without_hash):
        raise LaborThresholdDiagnosticError("diagnostic content hash mismatch")


def _source_provenance(
    repo_root: Path,
    *,
    publisher: Mapping[str, Any],
) -> dict[str, Any]:
    tag = _validate_source_git_anchor(repo_root)
    package = _validate_source_checksums(repo_root)
    _validate_source_manifest(repo_root)
    required = (
        SOURCE_AGGREGATE_RELATIVE.as_posix(),
        SOURCE_CHECKSUMS_RELATIVE.as_posix(),
        SOURCE_MANIFEST_RELATIVE.as_posix(),
    )
    tagged_blobs = _tracked_blob_inventory(
        repo_root,
        commit=SOURCE_EVIDENCE_COMMIT,
        relatives=required,
    )
    value: dict[str, Any] = {
        "schema_version": SOURCE_PROVENANCE_SCHEMA_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "source_package": {
            "logical_path": SOURCE_PACKAGE_RELATIVE.as_posix(),
            **tag,
            **package,
            "required_tagged_files": tagged_blobs,
            "science_anchor": {
                "contract_id": SOURCE_CONTRACT_ID,
                "contract_sha256": SOURCE_CONTRACT_SHA256,
                "git_tag": SOURCE_SCIENCE_TAG,
                "git_commit": SOURCE_SCIENCE_COMMIT,
            },
        },
        "publisher": dict(publisher),
        "publication_execution": {
            "provider_calls": 0,
            "hosted_cost_usd": 0.0,
            "credential_reads": 0,
            "source_mutations": 0,
        },
    }
    value["content_sha256"] = _content_sha256(value)
    return value


def _failure_ledger(payload: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    for record in payload["run_records"]:
        if record["cell_class"] == "actor-run" and record["source_status"] == "failed":
            failures.append(
                {
                    "stage_id": record["stage_id"],
                    "arm_id": record["arm_id"],
                    "seed": record["seed"],
                    "run_id": record["run_id"],
                    "source_artifact_sha256": record["source_artifact_sha256"],
                    "failure": record["failure"],
                    "registered_action_denominator": 48,
                    "observed_action_count": 0,
                    "missing_action_count": 48,
                    "threshold_results": None,
                    "imputed": False,
                    "replacement_seed": None,
                    "retried_for_diagnostic": False,
                }
            )
    value: dict[str, Any] = {
        "schema_version": FAILURE_LEDGER_SCHEMA_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "failure_count": len(failures),
        "missing_action_count": sum(item["missing_action_count"] for item in failures),
        "policy": "ITT retained; null outcomes; no imputation, retry, or replacement",
        "failures": failures,
        "provider_calls": 0,
        "scientific_evidence": False,
    }
    value["content_sha256"] = _content_sha256(value)
    return value


def _per_run_csv(payload: Mapping[str, Any]) -> bytes:
    threshold_ids = [item["threshold_id"] for item in THRESHOLDS]
    fields = [
        "stage_id",
        "arm_id",
        "seed",
        "run_id",
        "cell_class",
        "source_status",
        "action_metric_applicable",
        "registered_action_denominator",
        "observed_action_count",
        "missing_action_count",
        *(
            value
            for threshold_id in threshold_ids
            for value in (f"{threshold_id}_count", f"{threshold_id}_rate")
        ),
        "failure_error_type",
        "failure_message_sha256",
        "source_artifact_sha256",
        "source_row_content_sha256",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in payload["run_records"]:
        row: dict[str, Any] = {
            "stage_id": record["stage_id"],
            "arm_id": record["arm_id"],
            "seed": record["seed"],
            "run_id": record["run_id"],
            "cell_class": record["cell_class"],
            "source_status": record["source_status"],
            "action_metric_applicable": str(record["action_metric_applicable"]).lower(),
            "registered_action_denominator": record["registered_action_denominator"],
            "observed_action_count": record["observed_action_count"],
            "missing_action_count": record["missing_action_count"],
            "failure_error_type": (
                record["failure"].get("error_type") if record["failure"] else ""
            ),
            "failure_message_sha256": (
                record["failure"].get("message_sha256") if record["failure"] else ""
            ),
            "source_artifact_sha256": record["source_artifact_sha256"],
            "source_row_content_sha256": record["source_row_content_sha256"],
        }
        for threshold_id in threshold_ids:
            result = (
                record["threshold_results"].get(threshold_id)
                if record["threshold_results"]
                else None
            )
            row[f"{threshold_id}_count"] = (
                result["below_threshold_count"] if result else ""
            )
            row[f"{threshold_id}_rate"] = result["rate"] if result else ""
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _paired_csv(payload: Mapping[str, Any]) -> bytes:
    fields = [
        "contrast_id",
        "stage_id",
        "left_arm",
        "right_arm",
        "threshold_id",
        "seed",
        "pair_status",
        "left_status",
        "right_status",
        "left_rate",
        "right_rate",
        "left_minus_right_rate",
    ]
    record_index = {
        (record["stage_id"], record["arm_id"], record["seed"]): record
        for record in payload["run_records"]
        if record["cell_class"] == "actor-run"
    }
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for contrast in payload["paired_contrasts"]:
        for threshold in THRESHOLDS:
            threshold_id = threshold["threshold_id"]
            for seed in SEEDS:
                left = record_index[(contrast["stage_id"], contrast["left_arm"], seed)]
                right = record_index[(contrast["stage_id"], contrast["right_arm"], seed)]
                complete = left["source_status"] == right["source_status"] == "complete"
                left_rate = (
                    left["threshold_results"][threshold_id]["rate"]
                    if left["source_status"] == "complete"
                    else ""
                )
                right_rate = (
                    right["threshold_results"][threshold_id]["rate"]
                    if right["source_status"] == "complete"
                    else ""
                )
                paired_delta: float | str = ""
                if complete:
                    paired_delta = float(left_rate) - float(right_rate)
                writer.writerow(
                    {
                        "contrast_id": contrast["contrast_id"],
                        "stage_id": contrast["stage_id"],
                        "left_arm": contrast["left_arm"],
                        "right_arm": contrast["right_arm"],
                        "threshold_id": threshold_id,
                        "seed": seed,
                        "pair_status": "complete" if complete else "excluded-null",
                        "left_status": left["source_status"],
                        "right_status": right["source_status"],
                        "left_rate": left_rate,
                        "right_rate": right_rate,
                        "left_minus_right_rate": paired_delta,
                    }
                )
    return stream.getvalue().encode("utf-8")


def _pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def _report(payload: Mapping[str, Any]) -> bytes:
    lines = [
        "# V2.11.5 executed-labor threshold sensitivity diagnostic",
        "",
        "## Verdict",
        "",
        (
            "This is a retrospective, descriptive, zero-provider diagnostic. It "
            "does **not** restore or reverse the authoritative Experiment A or "
            "Experiment C no-go."
        ),
        "",
        "The metric is **below-threshold executed labor action rate**. It is not an unemployment rate, employment-state estimate, causal effect, or effectiveness result.",
        "",
        "## Frozen metric",
        "",
        "For each complete 4-agent × 12-month seed/run, the denominator is 48 executed actions. The frozen thresholds are `h < 1` (0 hours), `h < 20` (0/8/16), and `h < 40` (0/8/16/24/32). Rates are computed within each seed/run; summaries use the seed/run as the unit.",
        "",
        "## Denominator",
        "",
        "| Item | Count |",
        "|---|---:|",
    ]
    denominator = payload["denominator"]
    denominator_rows = (
        ("Registered A+C cells", denominator["registered_a_c_cell_count"]),
        ("Registered actor runs", denominator["registered_actor_run_count"]),
        ("Structural candidate-admission N/A cells", denominator["structural_not_applicable_cell_count"]),
        ("Complete actor runs", denominator["complete_actor_run_count"]),
        ("Failed actor runs retained", denominator["failed_actor_run_count"]),
        ("Registered actor action opportunities", denominator["registered_actor_action_opportunities"]),
        ("Observed actor actions", denominator["observed_actor_action_count"]),
        ("Missing actor actions", denominator["missing_actor_action_count"]),
    )
    lines.extend(f"| {label} | {value} |" for label, value in denominator_rows)
    lines.extend(
        [
            "",
            "All three failed A cells remain null and contribute 48 missing actions each. No failed cell was retried, replaced, removed, or imputed. The five candidate-admission cells are retained in the 45-cell inventory but are structurally N/A for an actor-action metric.",
            "",
            "## Arm summaries",
            "",
            "Values below are mean / median / range over complete seed/runs only; the complete/registered column exposes missingness.",
            "",
            "| Stage | Arm | Complete / registered | h<1 | h<20 | h<40 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for arm in payload["arm_summaries"]:
        cells = []
        for threshold in THRESHOLDS:
            stats = arm["thresholds"][threshold["threshold_id"]][
                "summary_over_complete_seed_runs"
            ]
            cells.append(
                f"{_pct(stats['mean'])} / {_pct(stats['median'])} / "
                f"[{_pct(stats['range'][0])}, {_pct(stats['range'][1])}]"
            )
        lines.append(
            f"| {arm['stage_id']} | {arm['arm_id']} | "
            f"{arm['complete_seed_count']}/{arm['registered_seed_count']} | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Paired descriptive contrasts",
            "",
            "Each delta is left-arm rate minus right-arm rate. No contrast is interpreted as causal or used as a replacement pass/fail gate.",
            "",
            "| Contrast | Complete pairs | h<1 median delta | h<20 median delta | h<40 median delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for contrast in payload["paired_contrasts"]:
        medians = [
            contrast["thresholds"][threshold["threshold_id"]][
                "summary_over_complete_pairs"
            ]["median"]
            for threshold in THRESHOLDS
        ]
        lines.append(
            f"| {contrast['contrast_id']} | {contrast['complete_pair_count']}/5 | "
            + " | ".join(_pct(value) for value in medians)
            + " |"
        )
    lines.extend(
        [
            "",
            "The A `full − prompt-only` contrast has 5/5 complete pairs. The A `retrieval-only − no-context` contrast has only 2/5 complete pairs; the other three registered pairs remain explicitly excluded-null because at least one source run failed.",
            "",
            "## Claim boundary",
            "",
            payload["claim_boundary"],
            "",
            "Experiment A remains a no-go (3/5 primary directions; median relative effect 3.062%, below the frozen 5% threshold, with a failed retrieval-only route manipulation check). Experiment C remains a no-go because its preregistered zero-API sensitivity artifact was not sealed by the authoritative stage. This publication-time labor diagnostic changes neither decision.",
            "",
            "## Provenance",
            "",
            f"Source aggregate SHA-256: `{payload['source_binding']['source_aggregate_sha256']}`.",
            "",
            f"Selected A/C row projection SHA-256: `{payload['source_binding']['selected_row_projection_sha256']}`.",
            "",
            f"Diagnostic content SHA-256: `{payload['content_sha256']}`.",
            "",
            "New provider calls: `0`; hosted cost: `$0`; credential reads: `0`.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _schema_bytes(repo_root: Path) -> bytes:
    path = _validate_regular_file(repo_root, SCHEMA_RELATIVE)
    schema = _strict_json_load(path)
    return _pretty_bytes(schema)


def _scan_publication_bytes(files: Mapping[str, bytes]) -> None:
    for relative, data in files.items():
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LaborThresholdDiagnosticError(
                f"publication file is not UTF-8: {relative}"
            ) from exc
        for pattern in _ABSOLUTE_PATH_PATTERNS:
            if pattern.search(text):
                raise LaborThresholdDiagnosticError(
                    f"absolute/local path leaked into publication: {relative}"
                )
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                raise LaborThresholdDiagnosticError(
                    f"possible secret leaked into publication: {relative}"
                )


def _build_file_bytes(
    repo_root: Path,
    *,
    aggregate: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
    publisher_provenance: Mapping[str, Any],
) -> dict[str, bytes]:
    payload = _build_diagnostic_payload(
        aggregate,
        source_provenance=source_provenance,
        publisher_provenance=publisher_provenance,
    )
    _validate_diagnostic_internal(payload)
    failures = _failure_ledger(payload)
    files: dict[str, bytes] = {
        "labor_threshold_sensitivity.json": _pretty_bytes(payload),
        "per_run.csv": _per_run_csv(payload),
        "paired_contrasts.csv": _paired_csv(payload),
        "failure_ledger.json": _pretty_bytes(failures),
        "source_provenance.json": _pretty_bytes(source_provenance),
        "schema.json": _schema_bytes(repo_root),
        "report.md": _report(payload),
    }
    manifest: dict[str, Any] = {
        "schema_version": PACKAGE_MANIFEST_SCHEMA_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "classification": "retrospective-descriptive-diagnostic-only",
        "scientific_evidence": False,
        "provider_calls": 0,
        "hosted_cost_usd": 0.0,
        "credential_reads": 0,
        "source_package": SOURCE_PACKAGE_RELATIVE.as_posix(),
        "source_evidence_tag": SOURCE_EVIDENCE_TAG,
        "source_evidence_commit": SOURCE_EVIDENCE_COMMIT,
        "source_aggregate_sha256": SOURCE_AGGREGATE_SHA256,
        "publisher_commit": publisher_provenance["git_commit"],
        "diagnostic_content_sha256": payload["content_sha256"],
        "published_files": list(_OUTPUT_FILES),
        "claim_boundary": payload["claim_boundary"],
    }
    manifest["content_sha256"] = _content_sha256(manifest)
    files["package_manifest.json"] = _pretty_bytes(manifest)
    checksum_rows = [
        {
            "path": relative,
            "sha256": sha256(data).hexdigest(),
            "byte_size": len(data),
        }
        for relative, data in sorted(files.items())
    ]
    checksums: dict[str, Any] = {
        "schema_version": PACKAGE_CHECKSUM_SCHEMA_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "files": checksum_rows,
    }
    checksums["content_sha256"] = _content_sha256(checksums)
    files["checksums.json"] = _pretty_bytes(checksums)
    if set(files) != set(_OUTPUT_FILES):
        raise LaborThresholdDiagnosticError("publication file inventory drifted")
    _scan_publication_bytes(files)
    return files


def _write_new_package(target: Path, files: Mapping[str, bytes]) -> None:
    if target.exists() or target.is_symlink():
        raise LaborThresholdDiagnosticError(
            f"diagnostic target already exists; refusing overwrite: {target.name}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-build-", dir=target.parent))
    claimed = False
    try:
        for relative, data in files.items():
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        # Atomic exclusive claim prevents two builders from overwriting one another.
        target.mkdir()
        claimed = True
        for relative in sorted(
            files,
            key=lambda value: (value == "checksums.json", value),
        ):
            source = temporary / relative
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            # A hard link is an atomic no-overwrite install on this filesystem.
            # It prevents a concurrent writer from replacing a destination after
            # this builder has exclusively claimed the package directory.
            os.link(source, destination, follow_symlinks=False)
            source.unlink()
        temporary.rmdir()
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if claimed:
            shutil.rmtree(target, ignore_errors=True)
        raise


def _source_snapshot(repo_root: Path) -> dict[str, tuple[int, str]]:
    source_root = repo_root / SOURCE_PACKAGE_RELATIVE
    return {
        path.relative_to(source_root).as_posix(): (path.stat().st_size, _sha256_file(path))
        for path in sorted(source_root.rglob("*"))
        if path.is_file()
    }


def build_v2115_labor_threshold_sensitivity(
    *,
    repo_root: str | Path,
    build_root: str | Path | None = None,
) -> LaborThresholdPackage:
    """Build the sealed, provider-free diagnostic in a new directory."""

    root = Path(repo_root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise LaborThresholdDiagnosticError("repo_root must be a real directory")
    if Path(_git(root, "rev-parse", "--show-toplevel")) != root:
        raise LaborThresholdDiagnosticError("repo_root must be the exact Git root")
    publisher = _publisher_provenance(root)
    before = _source_snapshot(root)
    provenance = _source_provenance(root, publisher=publisher)
    aggregate_path = _validate_regular_file(root, SOURCE_AGGREGATE_RELATIVE)
    if _sha256_file(aggregate_path) != SOURCE_AGGREGATE_SHA256:
        raise LaborThresholdDiagnosticError("source aggregate hash drifted")
    aggregate = _mapping(_strict_json_load(aggregate_path), "source aggregate")
    files = _build_file_bytes(
        root,
        aggregate=aggregate,
        source_provenance=provenance,
        publisher_provenance=publisher,
    )
    after = _source_snapshot(root)
    if before != after:
        raise LaborThresholdDiagnosticError("source package changed during diagnostic build")
    publisher_after = _tracked_blob_inventory(
        root,
        commit=str(publisher["git_commit"]),
        relatives=_REQUIRED_PUBLISHER_FILES,
    )
    if publisher_after != publisher["required_tracked_files"]:
        raise LaborThresholdDiagnosticError(
            "publisher implementation changed during diagnostic build"
        )
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise LaborThresholdDiagnosticError(
            "publisher worktree changed during diagnostic build"
        )
    base = Path(build_root).absolute() if build_root is not None else root / "evidence/current_v2"
    target = base / OUTPUT_RELATIVE.name
    try:
        target.relative_to(root / SOURCE_PACKAGE_RELATIVE)
    except ValueError:
        pass
    else:
        raise LaborThresholdDiagnosticError("diagnostic target cannot be inside source package")
    _write_new_package(target, files)
    try:
        validated = validate_v2115_labor_threshold_package(
            package_dir=target,
            repo_root=root,
        )
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return LaborThresholdPackage(
        package_dir=target,
        diagnostic_path=target / "labor_threshold_sensitivity.json",
        checksums_path=target / "checksums.json",
        content_sha256=validated["content_sha256"],
    )


def _verify_publisher_binding(repo_root: Path, provenance: Mapping[str, Any]) -> None:
    publisher = _mapping(provenance.get("publisher"), "publisher provenance")
    commit = publisher.get("git_commit")
    if not isinstance(commit, str) or not _HEX40.fullmatch(commit):
        raise LaborThresholdDiagnosticError("publisher commit is invalid")
    head = _git(repo_root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", commit, head],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestor.returncode != 0:
        raise LaborThresholdDiagnosticError("publisher commit is not an ancestor of HEAD")
    required = _mapping(publisher.get("required_tracked_files"), "publisher files")
    if set(required) != set(_REQUIRED_PUBLISHER_FILES):
        raise LaborThresholdDiagnosticError("publisher required-file inventory drifted")
    observed = _tracked_blob_inventory(
        repo_root,
        commit=commit,
        relatives=_REQUIRED_PUBLISHER_FILES,
    )
    if dict(required) != observed:
        raise LaborThresholdDiagnosticError("publisher tracked-file binding drifted")
    if (
        publisher.get("tracked_worktree_clean") is not True
        or publisher.get("provider_calls") != 0
        or publisher.get("hosted_cost_usd") != 0.0
        or publisher.get("credential_reads") != 0
    ):
        raise LaborThresholdDiagnosticError("publisher zero-provider boundary drifted")


def validate_v2115_labor_threshold_package(
    *,
    package_dir: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Validate checksums, provenance, denominators, and all derived values."""

    root = Path(repo_root).absolute()
    package = Path(package_dir).absolute()
    if package.is_symlink() or not package.is_dir():
        raise LaborThresholdDiagnosticError("diagnostic package directory is absent")
    if root.is_symlink() or Path(_git(root, "rev-parse", "--show-toplevel")) != root:
        raise LaborThresholdDiagnosticError("repo_root must be the exact real Git root")
    actual_files = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file()
    }
    if actual_files != set(_OUTPUT_FILES):
        raise LaborThresholdDiagnosticError("diagnostic package file inventory drifted")
    for relative in actual_files:
        _validate_regular_file(package, relative)
    checksums = _mapping(
        _strict_json_load(package / "checksums.json"), "diagnostic checksums"
    )
    if (
        checksums.get("schema_version") != PACKAGE_CHECKSUM_SCHEMA_VERSION
        or checksums.get("diagnostic_id") != DIAGNOSTIC_ID
    ):
        raise LaborThresholdDiagnosticError("diagnostic checksum header drifted")
    checksum_rows = _list(checksums.get("files"), "diagnostic checksum rows")
    checksum_index: dict[str, Mapping[str, Any]] = {}
    for raw in checksum_rows:
        row = _mapping(raw, "diagnostic checksum row")
        relative_value = row.get("path")
        if not isinstance(relative_value, str) or relative_value in checksum_index:
            raise LaborThresholdDiagnosticError("diagnostic checksum path duplicated")
        relative = relative_value
        checksum_index[relative] = row
    if set(checksum_index) != set(_OUTPUT_FILES) - {"checksums.json"}:
        raise LaborThresholdDiagnosticError("diagnostic checksum inventory drifted")
    for relative, row in checksum_index.items():
        path = _validate_regular_file(package, relative)
        if (
            row.get("byte_size") != path.stat().st_size
            or row.get("sha256") != _sha256_file(path)
        ):
            raise LaborThresholdDiagnosticError(f"diagnostic checksum mismatch: {relative}")
    without_hash = dict(checksums)
    checksum_content_hash = without_hash.pop("content_sha256", None)
    if checksum_content_hash != _content_sha256(without_hash):
        raise LaborThresholdDiagnosticError("checksums content hash mismatch")

    publication_bytes = {
        relative: (package / relative).read_bytes() for relative in actual_files
    }
    _scan_publication_bytes(publication_bytes)
    provenance = _mapping(
        _strict_json_load(package / "source_provenance.json"), "source provenance"
    )
    provenance_without_hash = dict(provenance)
    provenance_hash = provenance_without_hash.pop("content_sha256", None)
    if provenance_hash != _content_sha256(provenance_without_hash):
        raise LaborThresholdDiagnosticError("source provenance content hash mismatch")
    if provenance.get("schema_version") != SOURCE_PROVENANCE_SCHEMA_VERSION:
        raise LaborThresholdDiagnosticError("source provenance schema drifted")
    execution = _mapping(provenance.get("publication_execution"), "publication execution")
    if dict(execution) != {
        "provider_calls": 0,
        "hosted_cost_usd": 0.0,
        "credential_reads": 0,
        "source_mutations": 0,
    }:
        raise LaborThresholdDiagnosticError("publication execution boundary drifted")
    _verify_publisher_binding(root, provenance)
    current_source = _source_provenance(
        root,
        publisher=_mapping(provenance["publisher"], "publisher"),
    )
    if current_source != provenance:
        raise LaborThresholdDiagnosticError("source provenance no longer reproduces")

    aggregate_path = _validate_regular_file(root, SOURCE_AGGREGATE_RELATIVE)
    aggregate = _mapping(_strict_json_load(aggregate_path), "source aggregate")
    payload = _mapping(
        _strict_json_load(package / "labor_threshold_sensitivity.json"),
        "diagnostic payload",
    )
    _validate_diagnostic_internal(payload)
    expected_payload = _build_diagnostic_payload(
        aggregate,
        source_provenance=provenance,
        publisher_provenance=provenance["publisher"],
    )
    if payload != expected_payload:
        raise LaborThresholdDiagnosticError("diagnostic payload differs from source replay")
    if (package / "per_run.csv").read_bytes() != _per_run_csv(payload):
        raise LaborThresholdDiagnosticError("per-run CSV is not reproducible")
    if (package / "paired_contrasts.csv").read_bytes() != _paired_csv(payload):
        raise LaborThresholdDiagnosticError("paired contrast CSV is not reproducible")
    if _strict_json_load(package / "failure_ledger.json") != _failure_ledger(payload):
        raise LaborThresholdDiagnosticError("failure ledger is not reproducible")
    if (package / "report.md").read_bytes() != _report(payload):
        raise LaborThresholdDiagnosticError("reviewer report is not reproducible")
    if (package / "schema.json").read_bytes() != _schema_bytes(root):
        raise LaborThresholdDiagnosticError("published schema differs from tracked schema")

    manifest = _mapping(
        _strict_json_load(package / "package_manifest.json"), "package manifest"
    )
    manifest_without_hash = dict(manifest)
    manifest_hash = manifest_without_hash.pop("content_sha256", None)
    if manifest_hash != _content_sha256(manifest_without_hash):
        raise LaborThresholdDiagnosticError("package manifest content hash mismatch")
    if (
        manifest.get("schema_version") != PACKAGE_MANIFEST_SCHEMA_VERSION
        or manifest.get("diagnostic_id") != DIAGNOSTIC_ID
        or manifest.get("classification")
        != "retrospective-descriptive-diagnostic-only"
        or manifest.get("scientific_evidence") is not False
        or manifest.get("provider_calls") != 0
        or manifest.get("hosted_cost_usd") != 0.0
        or manifest.get("credential_reads") != 0
        or manifest.get("source_package") != SOURCE_PACKAGE_RELATIVE.as_posix()
        or manifest.get("source_evidence_tag") != SOURCE_EVIDENCE_TAG
        or manifest.get("source_evidence_commit") != SOURCE_EVIDENCE_COMMIT
        or manifest.get("source_aggregate_sha256") != SOURCE_AGGREGATE_SHA256
        or manifest.get("publisher_commit")
        != provenance["publisher"]["git_commit"]
        or manifest.get("published_files") != list(_OUTPUT_FILES)
        or manifest.get("diagnostic_content_sha256") != payload["content_sha256"]
        or manifest.get("claim_boundary") != payload["claim_boundary"]
    ):
        raise LaborThresholdDiagnosticError("package manifest boundary drifted")
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_id": DIAGNOSTIC_ID,
        "status": "valid",
        "content_sha256": payload["content_sha256"],
        "package_checksums_sha256": _sha256_file(package / "checksums.json"),
        "registered_cells": payload["denominator"]["registered_a_c_cell_count"],
        "registered_actor_actions": payload["denominator"][
            "registered_actor_action_opportunities"
        ],
        "observed_actor_actions": payload["denominator"]["observed_actor_action_count"],
        "missing_actor_actions": payload["denominator"]["missing_actor_action_count"],
        "provider_calls": 0,
        "scientific_evidence": False,
    }


__all__ = [
    "DIAGNOSTIC_ID",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "LaborThresholdDiagnosticError",
    "LaborThresholdPackage",
    "build_v2115_labor_threshold_sensitivity",
    "validate_v2115_labor_threshold_package",
]
