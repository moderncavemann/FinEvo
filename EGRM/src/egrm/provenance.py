"""Validate the source-pinned EGRM extraction and its Git source objects."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping


SCHEMA_VERSION = "egrm-source-provenance-v1"
EXPECTED_KEYS = {
    "schema_version",
    "extracted_on",
    "source_repository",
    "source_commit",
    "source_tag",
    "source_tree_state",
    "modules",
    "inherited_tests",
    "new_egrm_modules",
    "explicitly_excluded",
    "scientific_evidence_imported",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _relative_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{name} must be a normalized relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a normalized relative path")
    return path.as_posix()


def _digest(value: Any, name: str, length: int) -> str:
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"{name} must be a lowercase {length}-character digest")
    return value


def _git_output(repo_root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ValueError(f"Git source verification failed for {' '.join(args)}")
    return completed.stdout


def _validate_rows(
    rows: Any,
    *,
    label: str,
    egrm_root: Path,
    source_repo_root: Path | None,
    source_commit: str,
) -> list[str]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"source provenance requires {label} rows")
    checked: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "source",
            "destination",
            "git_blob",
            "source_sha256",
            "destination_sha256",
            "changes",
        }:
            raise ValueError(f"source provenance {label} keys are not exact")
        source = _relative_path(row["source"], f"{label}.source")
        destination_relative = _relative_path(
            row["destination"], f"{label}.destination"
        )
        git_blob = _digest(row["git_blob"], f"{label}.git_blob", 40)
        source_sha256 = _digest(row["source_sha256"], f"{label}.source_sha256", 64)
        destination_sha256 = _digest(
            row["destination_sha256"], f"{label}.destination_sha256", 64
        )
        changes = row["changes"]
        if not isinstance(changes, list) or any(
            not isinstance(change, str) or not change for change in changes
        ):
            raise ValueError(f"{label}.changes must be a string list")
        if (source_sha256 == destination_sha256) != (not changes):
            raise ValueError(
                f"source provenance {label} change declaration is inconsistent"
            )

        destination = egrm_root / destination_relative
        try:
            resolved_destination = destination.resolve(strict=True)
            resolved_destination.relative_to(egrm_root.resolve(strict=True))
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise ValueError(
                f"extracted file escapes the EGRM root or is missing: "
                f"{destination_relative}"
            ) from exc
        if destination.is_symlink() or not resolved_destination.is_file():
            raise ValueError(f"extracted file is not regular: {destination_relative}")
        observed_destination = file_sha256(resolved_destination)
        if observed_destination != destination_sha256:
            raise ValueError(f"extracted file hash mismatch: {destination_relative}")

        if source_repo_root is not None:
            revision = f"{source_commit}:{source}"
            observed_blob = (
                _git_output(source_repo_root, "rev-parse", revision)
                .decode("ascii", "strict")
                .strip()
            )
            if observed_blob != git_blob:
                raise ValueError(f"source Git blob mismatch: {source}")
            observed_source_sha = _sha256_bytes(
                _git_output(source_repo_root, "show", revision)
            )
            if observed_source_sha != source_sha256:
                raise ValueError(f"source file hash mismatch: {source}")
        checked.append(destination_relative)
    if len(set(checked)) != len(checked):
        raise ValueError(f"source provenance {label} destinations are duplicated")
    return checked


def validate_extraction(
    manifest_path: str | Path,
    *,
    egrm_root: str | Path | None = None,
    source_repo_root: str | Path | None = None,
    verify_source: bool = True,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    with manifest_file.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != EXPECTED_KEYS
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("source provenance schema is invalid")
    if manifest.get("scientific_evidence_imported") is not False:
        raise ValueError("source provenance must forbid imported scientific evidence")
    commit = manifest.get("source_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source provenance requires a full commit")
    root = Path(egrm_root) if egrm_root is not None else manifest_file.parents[1]
    if verify_source:
        source_root = (
            Path(source_repo_root).resolve()
            if source_repo_root is not None
            else root.parent.resolve()
        )
        observed_source_root = Path(
            _git_output(source_root, "rev-parse", "--show-toplevel")
            .decode("utf-8", "strict")
            .strip()
        ).resolve()
        if observed_source_root != source_root:
            raise ValueError("source_repo_root must be the Git repository root")
        observed_commit = (
            _git_output(source_root, "rev-parse", f"{commit}^{{commit}}")
            .decode("ascii", "strict")
            .strip()
        )
        if observed_commit != commit:
            raise ValueError("source commit does not resolve exactly")
        source_tag = manifest.get("source_tag")
        if not isinstance(source_tag, str) or not source_tag:
            raise ValueError("source_tag must be non-empty")
        tag_type = (
            _git_output(source_root, "cat-file", "-t", source_tag)
            .decode("ascii", "strict")
            .strip()
        )
        if tag_type != "tag":
            raise ValueError("source_tag must be annotated")
        tag_commit = (
            _git_output(source_root, "rev-parse", f"{source_tag}^{{commit}}")
            .decode("ascii", "strict")
            .strip()
        )
        if tag_commit != commit:
            raise ValueError("source_tag does not peel to source_commit")
    else:
        source_root = None

    checked_modules = _validate_rows(
        manifest.get("modules"),
        label="module",
        egrm_root=root,
        source_repo_root=source_root,
        source_commit=commit,
    )
    checked_tests = _validate_rows(
        manifest.get("inherited_tests"),
        label="inherited_test",
        egrm_root=root,
        source_repo_root=source_root,
        source_commit=commit,
    )
    if len(set((*checked_modules, *checked_tests))) != len(
        checked_modules
    ) + len(checked_tests):
        raise ValueError("source provenance destinations are globally duplicated")
    new_modules = manifest.get("new_egrm_modules")
    if not isinstance(new_modules, list) or not new_modules:
        raise ValueError("new_egrm_modules must be a non-empty list")
    normalized_new = [_relative_path(path, "new_egrm_modules") for path in new_modules]
    if normalized_new != sorted(normalized_new) or len(set(normalized_new)) != len(
        normalized_new
    ):
        raise ValueError("new_egrm_modules must be sorted and unique")
    missing_new = [path for path in normalized_new if not (root / path).is_file()]
    if missing_new:
        raise ValueError(f"missing new EGRM module: {missing_new[0]}")
    declared_package = set(checked_modules) | set(normalized_new)
    observed_package = {
        path.relative_to(root).as_posix()
        for path in (root / "src" / "egrm").glob("*.py")
        if path.is_file() and not path.is_symlink()
    }
    if declared_package != observed_package:
        raise ValueError("source provenance package inventory is not exact")
    if set(checked_tests) != {
        path.relative_to(root).as_posix()
        for path in (root / "tests").glob("test_extracted_*.py")
        if path.is_file() and not path.is_symlink()
    }:
        raise ValueError("source provenance inherited-test inventory is not exact")
    excluded = manifest.get("explicitly_excluded")
    if (
        not isinstance(excluded, list)
        or not excluded
        or len(set(excluded)) != len(excluded)
        or any(not isinstance(item, str) or not item.strip() for item in excluded)
    ):
        raise ValueError("explicitly_excluded must be a unique non-empty string list")
    return {
        "schema_version": "egrm-source-validation-v1",
        "source_commit": commit,
        "source_verified": verify_source,
        "checked_modules": checked_modules,
        "checked_module_count": len(checked_modules),
        "checked_inherited_tests": checked_tests,
        "checked_inherited_test_count": len(checked_tests),
        "new_module_count": len(normalized_new),
        "checked_count": len(checked_modules) + len(checked_tests),
        "scientific_evidence_imported": False,
    }
