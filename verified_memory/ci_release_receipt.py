"""Build safe, machine-readable CI receipts for scientific release gates.

The module has no provider credentials and never records environment values
other than an allow-listed set of GitHub identifiers.  Its final log line is
consumed by :mod:`verified_memory.scientific_release_attestation`.
"""

from __future__ import annotations

import argparse
import compileall
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

from .scientific_release_attestation import (
    CI_JOB_RECEIPT_LOG_PREFIX,
    CI_JOB_RECEIPT_SCHEMA_VERSION,
    ScientificReleaseAttestationError,
    canonical_sha256,
    sealed_manifest_inventory,
)


COLLECTION_INVENTORY_SCHEMA_VERSION = "finevo-ci-test-collection-v1"
COMPILED_SOURCE_INVENTORY_SCHEMA_VERSION = "finevo-ci-python-sources-v1"
_WORKFLOW_FILE = ".github/workflows/verified-memory-ci.yml"


class CIReleaseReceiptError(RuntimeError):
    """Raised when CI evidence is incomplete or internally inconsistent."""


def build_collection_inventory(nodeids: Sequence[str]) -> dict[str, Any]:
    """Return a deterministic inventory from pytest collection node IDs."""

    rows = list(nodeids)
    if (
        not rows
        or any(
            not isinstance(row, str)
            or not row
            or row != row.strip()
            or "\n" in row
            or "\r" in row
            for row in rows
        )
    ):
        raise CIReleaseReceiptError(
            "pytest collection must contain normalized non-empty node IDs"
        )
    if len(set(rows)) != len(rows):
        raise CIReleaseReceiptError(
            "pytest collection contains duplicate node IDs"
        )
    return {
        "schema_version": COLLECTION_INVENTORY_SCHEMA_VERSION,
        "test_count": len(rows),
        "test_collection_sha256": canonical_sha256(rows),
    }


def build_source_inventory(paths: Sequence[str]) -> dict[str, Any]:
    """Return the deterministic tracked-Python source inventory."""

    rows = list(paths)
    if (
        not rows
        or rows != sorted(rows)
        or len(set(rows)) != len(rows)
        or any(not row.endswith(".py") for row in rows)
    ):
        raise CIReleaseReceiptError(
            "tracked Python sources must be a non-empty sorted unique list"
        )
    return {
        "schema_version": COMPILED_SOURCE_INVENTORY_SCHEMA_VERSION,
        "compiled_source_count": len(rows),
        "compiled_source_inventory_sha256": canonical_sha256(rows),
    }


def parse_junit_summary(path: Path | str) -> dict[str, int]:
    """Count executed testcases and require a successful JUnit document."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise CIReleaseReceiptError("JUnit report is missing or not regular")
    try:
        root = ET.parse(source).getroot()
    except (ET.ParseError, OSError) as exc:
        raise CIReleaseReceiptError("JUnit report is not valid XML") from exc
    if root.tag not in {"testsuite", "testsuites"}:
        raise CIReleaseReceiptError("JUnit report has an unsupported root")
    cases = list(root.iter("testcase"))
    failures = len(list(root.iter("failure")))
    errors = len(list(root.iter("error")))
    skipped = len(list(root.iter("skipped")))
    if not cases or failures or errors:
        raise CIReleaseReceiptError(
            "JUnit report is empty or contains failed/error tests"
        )
    return {
        "executed_test_count": len(cases),
        "failure_count": failures,
        "error_count": errors,
        "skipped_count": skipped,
    }


def build_ci_job_receipt(
    repo_root: Path | str,
    *,
    collection_inventory: Mapping[str, Any],
    source_inventory: Mapping[str, Any],
    junit_summary: Mapping[str, Any],
    environment: Mapping[str, str],
    manifest_paths: Sequence[str],
    workflow_file: str = _WORKFLOW_FILE,
) -> dict[str, Any]:
    """Build and self-hash one successful CI matrix-job receipt."""

    root = Path(repo_root).resolve()
    collection = _validate_collection_inventory(collection_inventory)
    sources = _validate_source_inventory(source_inventory)
    executed = _positive_int(
        junit_summary.get("executed_test_count"), "executed_test_count"
    )
    if executed != collection["test_count"]:
        raise CIReleaseReceiptError(
            "executed test count differs from collected test count"
        )
    for key in ("failure_count", "error_count", "skipped_count"):
        value = junit_summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CIReleaseReceiptError(
                f"JUnit {key} must be a non-negative integer"
            )
    if junit_summary["failure_count"] or junit_summary["error_count"]:
        raise CIReleaseReceiptError("cannot seal a failing CI test receipt")

    env = {
        key: _environment_text(environment, key)
        for key in (
            "GITHUB_REPOSITORY",
            "GITHUB_SHA",
            "GITHUB_RUN_ID",
            "GITHUB_RUN_ATTEMPT",
            "GITHUB_JOB",
            "GITHUB_WORKFLOW",
            "GITHUB_WORKFLOW_REF",
            "GITHUB_WORKFLOW_SHA",
            "RUNNER_OS",
            "FINEVO_CI_JOB_NAME",
        )
    }
    run_id = _positive_decimal(env["GITHUB_RUN_ID"], "GITHUB_RUN_ID")
    run_attempt = _positive_decimal(
        env["GITHUB_RUN_ATTEMPT"], "GITHUB_RUN_ATTEMPT"
    )
    if env["GITHUB_WORKFLOW_SHA"] != env["GITHUB_SHA"]:
        raise CIReleaseReceiptError(
            "workflow source SHA must equal the checked-out release SHA"
        )
    workflow = root / workflow_file
    if workflow.is_symlink() or not workflow.is_file():
        raise CIReleaseReceiptError(
            "workflow file is missing or is not regular"
        )
    workflow_bytes = workflow.read_bytes()
    workflow_sha256 = hashlib.sha256(workflow_bytes).hexdigest()
    workflow_blob_oid = _git_line(
        root, ("git", "rev-parse", "--verify", f"HEAD:{workflow_file}")
    )
    checked_out_head = _git_line(
        root, ("git", "rev-parse", "--verify", "HEAD^{commit}")
    )
    if checked_out_head != env["GITHUB_SHA"]:
        raise CIReleaseReceiptError(
            "checked-out HEAD does not equal GITHUB_SHA"
        )

    rows, manifest_inventory_sha256 = sealed_manifest_inventory(
        root, manifest_paths
    )
    payload = {
        "schema_version": CI_JOB_RECEIPT_SCHEMA_VERSION,
        "status": "pass",
        "repository": env["GITHUB_REPOSITORY"],
        "head_sha": checked_out_head,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "job_name": env["FINEVO_CI_JOB_NAME"],
        "job_key": env["GITHUB_JOB"],
        "runner_os": env["RUNNER_OS"],
        "workflow_name": env["GITHUB_WORKFLOW"],
        "workflow_file": workflow_file,
        "workflow_ref": env["GITHUB_WORKFLOW_REF"],
        "workflow_source_sha": env["GITHUB_WORKFLOW_SHA"],
        "workflow_file_sha256": workflow_sha256,
        "workflow_blob_oid": workflow_blob_oid,
        "test_count": collection["test_count"],
        "test_collection_sha256": collection[
            "test_collection_sha256"
        ],
        "skipped_test_count": junit_summary["skipped_count"],
        "compiled_source_count": sources["compiled_source_count"],
        "compiled_source_inventory_sha256": sources[
            "compiled_source_inventory_sha256"
        ],
        "sealed_manifest_count": len(rows),
        "sealed_manifest_inventory_sha256": manifest_inventory_sha256,
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def discover_tracked_files(
    repo_root: Path | str, patterns: Sequence[str]
) -> tuple[str, ...]:
    """Return the sorted tracked files matching exact Git pathspecs."""

    root = Path(repo_root).resolve()
    completed = subprocess.run(
        ("git", "ls-files", "-z", "--", *patterns),
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0 or completed.stderr:
        raise CIReleaseReceiptError("git ls-files failed")
    try:
        rows = [
            raw.decode("utf-8", "strict")
            for raw in completed.stdout.split(b"\0")
            if raw
        ]
    except UnicodeDecodeError as exc:
        raise CIReleaseReceiptError(
            "tracked file inventory is not UTF-8"
        ) from exc
    if rows != sorted(rows) or len(set(rows)) != len(rows):
        raise CIReleaseReceiptError(
            "tracked file inventory is not sorted and unique"
        )
    return tuple(rows)


def collect_tests(output: Path | str) -> dict[str, Any]:
    """Collect tests through pytest's object model and persist only its hash."""

    try:
        import pytest
    except ImportError as exc:  # pragma: no cover - CI dependency guard
        raise CIReleaseReceiptError("pytest is required to collect tests") from exc

    class InventoryPlugin:
        nodeids: list[str] | None = None

        @staticmethod
        def pytest_collection_finish(session: Any) -> None:
            InventoryPlugin.nodeids = [item.nodeid for item in session.items]

    result = pytest.main(
        ["--collect-only", "-q", "-p", "no:cacheprovider"],
        plugins=[InventoryPlugin()],
    )
    if result != pytest.ExitCode.OK or InventoryPlugin.nodeids is None:
        raise CIReleaseReceiptError("pytest test collection failed")
    inventory = build_collection_inventory(InventoryPlugin.nodeids)
    _write_json(Path(output), inventory)
    return inventory


def compile_sources(
    repo_root: Path | str, output: Path | str
) -> dict[str, Any]:
    """Compile every tracked Python source and persist its inventory hash."""

    root = Path(repo_root).resolve()
    sources = discover_tracked_files(root, ("*.py",))
    inventory = build_source_inventory(sources)
    failures = [
        relative
        for relative in sources
        if not compileall.compile_file(
            root / relative, force=True, quiet=1
        )
    ]
    if failures:
        raise CIReleaseReceiptError(
            f"compileall failed for {len(failures)} tracked source(s)"
        )
    _write_json(Path(output), inventory)
    return inventory


def emit_ci_job_receipt(
    repo_root: Path | str,
    *,
    collection_path: Path | str,
    source_path: Path | str,
    junit_path: Path | str,
    output_path: Path | str,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Seal one job receipt, write it, and print one canonical safe log line."""

    root = Path(repo_root).resolve()
    collection = _read_json(Path(collection_path), "collection inventory")
    sources = _read_json(Path(source_path), "source inventory")
    junit = parse_junit_summary(junit_path)
    manifests = discover_tracked_files(
        root,
        (
            "artifacts/verified_replays/*/manifest.json",
            "artifacts/verified_runs/*/manifest.json",
        ),
    )
    receipt = build_ci_job_receipt(
        root,
        collection_inventory=collection,
        source_inventory=sources,
        junit_summary=junit,
        environment=os.environ if environment is None else environment,
        manifest_paths=manifests,
    )
    _write_json(Path(output_path), receipt)
    print(
        CI_JOB_RECEIPT_LOG_PREFIX
        + json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return receipt


def _validate_collection_inventory(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if set(value) != {
        "schema_version",
        "test_count",
        "test_collection_sha256",
    }:
        raise CIReleaseReceiptError("collection inventory keys mismatch")
    if value.get("schema_version") != COLLECTION_INVENTORY_SCHEMA_VERSION:
        raise CIReleaseReceiptError(
            "collection inventory schema mismatch"
        )
    return {
        "schema_version": COLLECTION_INVENTORY_SCHEMA_VERSION,
        "test_count": _positive_int(value.get("test_count"), "test_count"),
        "test_collection_sha256": _sha256(
            value.get("test_collection_sha256"),
            "test_collection_sha256",
        ),
    }


def _validate_source_inventory(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if set(value) != {
        "schema_version",
        "compiled_source_count",
        "compiled_source_inventory_sha256",
    }:
        raise CIReleaseReceiptError("source inventory keys mismatch")
    if (
        value.get("schema_version")
        != COMPILED_SOURCE_INVENTORY_SCHEMA_VERSION
    ):
        raise CIReleaseReceiptError("source inventory schema mismatch")
    return {
        "schema_version": COMPILED_SOURCE_INVENTORY_SCHEMA_VERSION,
        "compiled_source_count": _positive_int(
            value.get("compiled_source_count"),
            "compiled_source_count",
        ),
        "compiled_source_inventory_sha256": _sha256(
            value.get("compiled_source_inventory_sha256"),
            "compiled_source_inventory_sha256",
        ),
    }


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CIReleaseReceiptError(f"{name} is missing or not regular")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CIReleaseReceiptError(f"{name} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise CIReleaseReceiptError(f"{name} must be a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(path)


def _git_line(root: Path, argv: Sequence[str]) -> str:
    completed = subprocess.run(
        tuple(argv),
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0 or completed.stderr:
        raise CIReleaseReceiptError("read-only Git command failed")
    try:
        value = completed.stdout.decode("utf-8", "strict").strip()
    except UnicodeDecodeError as exc:
        raise CIReleaseReceiptError("Git output is not UTF-8") from exc
    if (
        len(completed.stdout.decode("utf-8", "strict").splitlines()) != 1
        or not value
    ):
        raise CIReleaseReceiptError("Git output is not one normalized line")
    return value


def _environment_text(environment: Mapping[str, str], key: str) -> str:
    value = environment.get(key)
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise CIReleaseReceiptError(f"CI environment lacks normalized {key}")
    return value


def _positive_decimal(value: str, name: str) -> int:
    if not value.isdecimal() or int(value) <= 0:
        raise CIReleaseReceiptError(f"{name} must be a positive decimal")
    return int(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CIReleaseReceiptError(f"{name} must be a positive integer")
    return int(value)


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CIReleaseReceiptError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect-tests")
    collect_parser.add_argument("--output", required=True)
    compile_parser = subparsers.add_parser("compile-sources")
    compile_parser.add_argument("--output", required=True)
    emit_parser = subparsers.add_parser("emit")
    emit_parser.add_argument("--collection", required=True)
    emit_parser.add_argument("--sources", required=True)
    emit_parser.add_argument("--junit", required=True)
    emit_parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "collect-tests":
            collect_tests(args.output)
        elif args.command == "compile-sources":
            compile_sources(Path.cwd(), args.output)
        else:
            emit_ci_job_receipt(
                Path.cwd(),
                collection_path=args.collection,
                source_path=args.sources,
                junit_path=args.junit,
                output_path=args.output,
            )
    except (
        CIReleaseReceiptError,
        ScientificReleaseAttestationError,
    ) as exc:
        print(f"CI release receipt failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CI
    raise SystemExit(main())


__all__ = [
    "CIReleaseReceiptError",
    "COLLECTION_INVENTORY_SCHEMA_VERSION",
    "COMPILED_SOURCE_INVENTORY_SCHEMA_VERSION",
    "build_ci_job_receipt",
    "build_collection_inventory",
    "build_source_inventory",
    "collect_tests",
    "compile_sources",
    "discover_tracked_files",
    "emit_ci_job_receipt",
    "parse_junit_summary",
]
