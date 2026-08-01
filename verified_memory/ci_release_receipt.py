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
from .pilot_contract import PilotContractError, load_pilot_contract


COLLECTION_INVENTORY_SCHEMA_VERSION = "finevo-ci-test-collection-v1"
COMPILED_SOURCE_INVENTORY_SCHEMA_VERSION = "finevo-ci-python-sources-v1"
SCIENTIFIC_SOURCE_MANIFEST_INVENTORY_SCHEMA_VERSION = (
    "finevo-ci-scientific-source-manifests-v1"
)
PUBLICATION_CONSUMER_CI_AUTHORITY_SCHEMA_VERSION = (
    "finevo-publication-consumer-ci-authority-v1"
)
PUBLICATION_CONSUMER_CI_RECEIPT_SCHEMA_VERSION = (
    "finevo-publication-consumer-ci-job-receipt-v1"
)
PUBLICATION_CONSUMER_CI_RECEIPT_LOG_PREFIX = (
    "FINEVO_PUBLICATION_CONSUMER_CI_RECEIPT="
)
PUBLICATION_CONSUMER_CI_AUTHORITY_RELATIVE = (
    "experiments/pilot_v2_11_5_publication_consumer_ci.json"
)
_V2115_SCIENCE_CONTRACT_RELATIVE = "experiments/pilot_v2_11_5.yaml"
_V2115_SCIENCE_CONTRACT_ID = "finevo-pilot-v2.11.5"
_V2115_SCIENCE_CONTRACT_SHA256 = (
    "e1ecdec43e3f7a7b9a3d0977e2522d95861e826fc68781377d7eaceeb5e6e2ef"
)
_V2115_SCIENCE_TAG = "pilot-v2.11.5-science"
_V2115_SCIENCE_COMMIT = "2351ac2283f9fedb9dce70067174020be56ed9cc"
V2115_SCIENCE_TAG_OBJECT = "bccfb13cee7d592470d1873cfacc3b12bed38be4"
_V2115_REPOSITORY = "moderncavemann/FinEvo"
_WORKFLOW_FILE = ".github/workflows/verified-memory-ci.yml"
_EXPECTED_CI_FIELDS = frozenset(
    {
        "test_count",
        "test_collection_sha256",
        "compiled_source_count",
        "compiled_source_inventory_sha256",
        "sealed_manifest_inventory_sha256",
    }
)
_CI_JOB_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "repository",
        "head_sha",
        "run_id",
        "run_attempt",
        "job_name",
        "job_key",
        "runner_os",
        "workflow_name",
        "workflow_file",
        "workflow_ref",
        "workflow_source_sha",
        "workflow_file_sha256",
        "workflow_blob_oid",
        "test_count",
        "test_collection_sha256",
        "skipped_test_count",
        "compiled_source_count",
        "compiled_source_inventory_sha256",
        "sealed_manifest_count",
        "sealed_manifest_inventory_sha256",
        "receipt_sha256",
    }
)

# Source manifests are preregistration/import authorities, not sealed run
# manifests.  Keep this inventory separate from the six historical manifests
# consumed by ``sealed_manifest_inventory`` and its stable receipt schema.
SCIENTIFIC_SOURCE_MANIFEST_ANCHORS: tuple[Mapping[str, Any], ...] = (
    {
        "path": "experiments/pilot_v2_11_3_source_manifest.json",
        "schema_version": "finevo-pilot-v2.11.3-source-manifest-v1",
        "file_sha256": (
            "f05dbac4951e99476c06883e3c1b792e7ccb459c16eb4d78ac15ddf7905598de"
        ),
        "content_sha256": (
            "5c8e554d1a00803b81deb4f31b4a87ddf54a272861a7c750985cd72b18a95f00"
        ),
    },
    {
        "path": "experiments/pilot_v2_11_4_source_manifest.json",
        "schema_version": "finevo-pilot-v2.11.4-source-manifest-v1",
        "file_sha256": (
            "fd37e5f7a6cfa0178fa0baec74fb0d18f058a361586296d50d4bcf611e13839d"
        ),
        "content_sha256": (
            "594b1a00910a1dbecd5e36fcac4397df5341e92b5a9802ce4ca781434b747760"
        ),
    },
    {
        "path": "experiments/pilot_v2_11_5_source_manifest.json",
        "schema_version": "finevo-pilot-v2.11.5-source-manifest-v1",
        "file_sha256": (
            "fea5a276fb64fdd5bf0539014687ea39a891e9d305205b1d2046a2c15a892d16"
        ),
        "content_sha256": (
            "be84d33f561a5ab8927f13e0753f5109b5f018dc790ae180d5e0e6e0228af559"
        ),
    },
    {
        "path": "experiments/pilot_v2_11_6_source_manifest.json",
        "schema_version": "finevo-pilot-v2.11.6-source-manifest-v1",
        "file_sha256": (
            "710db4414471005d088cd64fb1e1a7c4a46fd99f8852b05f3f17f2acaead240d"
        ),
        "content_sha256": (
            "c510941c565d1120604199139d193990948d6b65be15a823ba1d4850968f2ce0"
        ),
    },
    {
        "path": "experiments/pilot_v2_11_7_source_manifest.json",
        "schema_version": "finevo-pilot-v2.11.7-source-manifest-v1",
        "file_sha256": (
            "dd124c09359d0bd08411add3486cc43887cbee207fdbb6f9bc929e5c1eb81ef9"
        ),
        "content_sha256": (
            "64be1bf836d131d8ec0542e68388dbc328314af7e891549600f5871f8f61f2b0"
        ),
    },
)


class CIReleaseReceiptError(RuntimeError):
    """Raised when CI evidence is incomplete or internally inconsistent."""


def build_collection_inventory(nodeids: Sequence[str]) -> dict[str, Any]:
    """Return a deterministic inventory from pytest collection node IDs."""

    rows = list(nodeids)
    if not rows or any(
        not isinstance(row, str)
        or not row
        or row != row.strip()
        or "\n" in row
        or "\r" in row
        for row in rows
    ):
        raise CIReleaseReceiptError(
            "pytest collection must contain normalized non-empty node IDs"
        )
    if len(set(rows)) != len(rows):
        raise CIReleaseReceiptError("pytest collection contains duplicate node IDs")
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


def build_scientific_source_manifest_inventory(
    repo_root: Path | str,
    *,
    anchors: Sequence[Mapping[str, Any]] = SCIENTIFIC_SOURCE_MANIFEST_ANCHORS,
) -> dict[str, Any]:
    """Verify and inventory release-critical scientific source manifests.

    These JSON authorities use an embedded canonical-content seal and an
    exact file-byte anchor.  They deliberately remain outside the sealed-run
    manifest inventory because :func:`sealed_manifest_inventory` expects the
    artifact-manifest schema and re-hashes the referenced run directory.
    """

    root = Path(repo_root).resolve()
    normalized = tuple(_validate_source_manifest_anchor(row) for row in anchors)
    paths = tuple(row["path"] for row in normalized)
    if not paths or paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
        raise CIReleaseReceiptError(
            "scientific source manifest anchors must be non-empty, sorted, and unique"
        )
    tracked = discover_tracked_files(root, paths)
    if tracked != paths:
        raise CIReleaseReceiptError(
            "scientific source manifest inventory is not exactly tracked"
        )

    rows: list[dict[str, str]] = []
    for anchor in normalized:
        relative = anchor["path"]
        source = _guarded_regular_file(root, relative)
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != anchor["file_sha256"]:
            raise CIReleaseReceiptError(
                f"scientific source manifest file hash drifted: {relative}"
            )
        value = _strict_json_object(raw, f"scientific source manifest {relative}")
        if value.get("schema_version") != anchor["schema_version"]:
            raise CIReleaseReceiptError(
                f"scientific source manifest schema drifted: {relative}"
            )
        canonical = (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        if raw != canonical:
            raise CIReleaseReceiptError(
                f"scientific source manifest bytes are not canonical: {relative}"
            )
        integrity = value.get("integrity")
        if not isinstance(integrity, Mapping):
            raise CIReleaseReceiptError(
                f"scientific source manifest integrity is missing: {relative}"
            )
        claimed = integrity.get("content_sha256")
        if (
            set(integrity) != {"canonicalization", "content_sha256"}
            or integrity.get("canonicalization") != "json-sort-keys-utf8-v1"
            or claimed != anchor["content_sha256"]
        ):
            raise CIReleaseReceiptError(
                f"scientific source manifest content anchor drifted: {relative}"
            )
        candidate = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
        candidate["integrity"].pop("content_sha256")
        if canonical_sha256(candidate) != claimed:
            raise CIReleaseReceiptError(
                f"scientific source manifest content seal drifted: {relative}"
            )
        rows.append(dict(anchor))

    return {
        "schema_version": SCIENTIFIC_SOURCE_MANIFEST_INVENTORY_SCHEMA_VERSION,
        "source_manifest_count": len(rows),
        "source_manifests": rows,
        "source_manifest_inventory_sha256": canonical_sha256(rows),
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
            raise CIReleaseReceiptError(f"JUnit {key} must be a non-negative integer")
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
    run_attempt = _positive_decimal(env["GITHUB_RUN_ATTEMPT"], "GITHUB_RUN_ATTEMPT")
    if env["GITHUB_WORKFLOW_SHA"] != env["GITHUB_SHA"]:
        raise CIReleaseReceiptError(
            "workflow source SHA must equal the checked-out release SHA"
        )
    workflow = root / workflow_file
    if workflow.is_symlink() or not workflow.is_file():
        raise CIReleaseReceiptError("workflow file is missing or is not regular")
    workflow_bytes = workflow.read_bytes()
    workflow_sha256 = hashlib.sha256(workflow_bytes).hexdigest()
    workflow_blob_oid = _git_line(
        root, ("git", "rev-parse", "--verify", f"HEAD:{workflow_file}")
    )
    checked_out_head = _git_line(
        root, ("git", "rev-parse", "--verify", "HEAD^{commit}")
    )
    if checked_out_head != env["GITHUB_SHA"]:
        raise CIReleaseReceiptError("checked-out HEAD does not equal GITHUB_SHA")

    rows, manifest_inventory_sha256 = sealed_manifest_inventory(root, manifest_paths)
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
        "test_collection_sha256": collection["test_collection_sha256"],
        "skipped_test_count": junit_summary["skipped_count"],
        "compiled_source_count": sources["compiled_source_count"],
        "compiled_source_inventory_sha256": sources["compiled_source_inventory_sha256"],
        "sealed_manifest_count": len(rows),
        "sealed_manifest_inventory_sha256": manifest_inventory_sha256,
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def verify_expected_ci_matches_receipt(
    expected_ci: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Require one CI receipt to reproduce the frozen contract inventory.

    This check intentionally runs inside each matrix job, before a merge
    commit can be treated as taggable release evidence.  The later scientific
    launch attestation repeats the same comparison against downloaded job
    receipts; keeping both checks makes an inventory mistake fail before an
    immutable science tag is created.
    """

    if not isinstance(expected_ci, Mapping) or set(expected_ci) != _EXPECTED_CI_FIELDS:
        raise CIReleaseReceiptError(
            "frozen contract expected_ci fields differ from the CI schema"
        )
    if not isinstance(receipt, Mapping):
        raise CIReleaseReceiptError("CI receipt must be a mapping")
    normalized = {
        "test_count": _positive_int(expected_ci.get("test_count"), "test_count"),
        "test_collection_sha256": _sha256(
            expected_ci.get("test_collection_sha256"),
            "test_collection_sha256",
        ),
        "compiled_source_count": _positive_int(
            expected_ci.get("compiled_source_count"),
            "compiled_source_count",
        ),
        "compiled_source_inventory_sha256": _sha256(
            expected_ci.get("compiled_source_inventory_sha256"),
            "compiled_source_inventory_sha256",
        ),
        "sealed_manifest_inventory_sha256": _sha256(
            expected_ci.get("sealed_manifest_inventory_sha256"),
            "sealed_manifest_inventory_sha256",
        ),
    }
    observed = {field: receipt.get(field) for field in _EXPECTED_CI_FIELDS}
    if observed != normalized:
        drifted = sorted(
            field
            for field in _EXPECTED_CI_FIELDS
            if observed[field] != normalized[field]
        )
        raise CIReleaseReceiptError(
            "CI receipt differs from frozen expected_ci: " + ", ".join(drifted)
        )
    return normalized


def verify_contract_ci_receipt(
    contract_path: Path | str,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Load one pinned frozen contract and verify its CI inventory now."""

    source = Path(contract_path)
    if source.is_symlink() or not source.is_file():
        raise CIReleaseReceiptError(
            "release contract is missing or is not a regular file"
        )
    try:
        contract = load_pilot_contract(source)
    except (OSError, ValueError, PilotContractError) as exc:
        raise CIReleaseReceiptError("release contract failed validation") from exc
    if contract.status != "frozen" or contract.release_requirements is None:
        raise CIReleaseReceiptError(
            "CI inventory comparison requires a frozen release contract"
        )
    verify_expected_ci_matches_receipt(
        contract.release_requirements.expected_ci,
        receipt,
    )
    return {
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "status": "pass",
    }


def load_publication_consumer_ci_authority(
    repo_root: Path | str,
    authority_path: Path | str = PUBLICATION_CONSUMER_CI_AUTHORITY_RELATIVE,
) -> dict[str, Any]:
    """Load the tracked non-scientific CI authority for a descendant consumer.

    The V2.11.5 science contract remains authoritative only for its exact
    science commit.  Publication code added later necessarily has a different
    test/source inventory, so this separate authority binds that inventory to
    the immutable science tag without granting scientific-dispatch authority.
    """

    root = Path(repo_root).resolve()
    source = Path(authority_path)
    if source.is_absolute():
        try:
            relative = source.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise CIReleaseReceiptError(
                "publication consumer CI authority must be below the repository"
            ) from exc
    else:
        relative = source.as_posix()
        source = root / source
    if relative != PUBLICATION_CONSUMER_CI_AUTHORITY_RELATIVE:
        raise CIReleaseReceiptError(
            "publication consumer CI authority path is not the registered path"
        )
    source = _guarded_authority_file(root, relative)
    if discover_tracked_files(root, (relative,)) != (relative,):
        raise CIReleaseReceiptError(
            "publication consumer CI authority is not exactly tracked"
        )
    _git_success(
        root,
        ("git", "diff", "--quiet", "HEAD", "--", relative),
        "publication consumer CI authority differs from HEAD",
    )

    raw = source.read_bytes()
    value = _strict_json_object(raw, "publication consumer CI authority")
    canonical = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise CIReleaseReceiptError(
            "publication consumer CI authority bytes are not canonical"
        )
    if set(value) != {
        "schema_version",
        "status",
        "authority_id",
        "science_anchor",
        "scope",
        "expected_ci",
        "integrity",
    }:
        raise CIReleaseReceiptError(
            "publication consumer CI authority keys mismatch"
        )
    if (
        value.get("schema_version")
        != PUBLICATION_CONSUMER_CI_AUTHORITY_SCHEMA_VERSION
        or value.get("status") != "frozen"
        or value.get("authority_id")
        != "finevo-pilot-v2.11.5-evidence-consumer-ci"
    ):
        raise CIReleaseReceiptError(
            "publication consumer CI authority identity drifted"
        )

    science = value.get("science_anchor")
    expected_science = {
        "contract_path": _V2115_SCIENCE_CONTRACT_RELATIVE,
        "contract_id": _V2115_SCIENCE_CONTRACT_ID,
        "contract_sha256": _V2115_SCIENCE_CONTRACT_SHA256,
        "git_tag": _V2115_SCIENCE_TAG,
        "git_tag_object": V2115_SCIENCE_TAG_OBJECT,
        "git_commit": _V2115_SCIENCE_COMMIT,
    }
    if not isinstance(science, Mapping) or dict(science) != expected_science:
        raise CIReleaseReceiptError(
            "publication consumer CI science anchor drifted"
        )
    scope = value.get("scope")
    expected_scope = {
        "purpose": "publication-consumer-ci",
        "scientific_evidence": False,
        "provider_calls": 0,
        "science_dispatch_authority": False,
    }
    if not isinstance(scope, Mapping) or dict(scope) != expected_scope:
        raise CIReleaseReceiptError(
            "publication consumer CI scope must remain non-scientific and zero-call"
        )
    expected_ci = value.get("expected_ci")
    # Validate every expected field before any comparison against a job.
    normalized_ci = verify_expected_ci_matches_receipt(expected_ci, expected_ci)
    integrity = value.get("integrity")
    if (
        not isinstance(integrity, Mapping)
        or set(integrity) != {"canonicalization", "content_sha256"}
        or integrity.get("canonicalization") != "json-sort-keys-utf8-v1"
    ):
        raise CIReleaseReceiptError(
            "publication consumer CI authority integrity keys mismatch"
        )
    claimed_content = _sha256(
        integrity.get("content_sha256"),
        "publication consumer CI authority content_sha256",
    )
    candidate = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    candidate["integrity"].pop("content_sha256")
    if canonical_sha256(candidate) != claimed_content:
        raise CIReleaseReceiptError(
            "publication consumer CI authority self-hash mismatch"
        )

    contract_path = root / _V2115_SCIENCE_CONTRACT_RELATIVE
    try:
        contract = load_pilot_contract(contract_path)
    except (OSError, ValueError, PilotContractError) as exc:
        raise CIReleaseReceiptError(
            "publication consumer CI science contract failed validation"
        ) from exc
    if (
        contract.status != "frozen"
        or contract.contract_id != _V2115_SCIENCE_CONTRACT_ID
        or contract.canonical_hash != _V2115_SCIENCE_CONTRACT_SHA256
        or contract.release_requirements is None
        or contract.release_requirements.tag != _V2115_SCIENCE_TAG
    ):
        raise CIReleaseReceiptError(
            "publication consumer CI authority does not match the frozen contract"
        )

    tag_ref = f"refs/tags/{_V2115_SCIENCE_TAG}"
    if _git_line(root, ("git", "cat-file", "-t", tag_ref)) != "tag":
        raise CIReleaseReceiptError(
            "publication consumer CI science tag is not annotated"
        )
    tag_object = _git_line(
        root,
        ("git", "rev-parse", "--verify", f"{tag_ref}^{{object}}"),
    )
    if tag_object != V2115_SCIENCE_TAG_OBJECT:
        raise CIReleaseReceiptError(
            "publication consumer CI science tag object drifted"
        )
    resolved_science = _git_line(
        root,
        ("git", "rev-parse", "--verify", f"{tag_ref}^{{commit}}"),
    )
    if resolved_science != _V2115_SCIENCE_COMMIT:
        raise CIReleaseReceiptError(
            "publication consumer CI science tag resolves to the wrong commit"
        )
    head = _git_line(root, ("git", "rev-parse", "--verify", "HEAD^{commit}"))
    if head == _V2115_SCIENCE_COMMIT:
        raise CIReleaseReceiptError(
            "publication consumer CI authority requires a descendant consumer commit"
        )
    _git_success(
        root,
        ("git", "merge-base", "--is-ancestor", _V2115_SCIENCE_COMMIT, head),
        "publication consumer HEAD does not descend from the science commit",
    )
    return {
        "schema_version": PUBLICATION_CONSUMER_CI_AUTHORITY_SCHEMA_VERSION,
        "authority_status": "frozen",
        "validation_status": "pass",
        "ci_execution_status": "unverified",
        "authority_id": value["authority_id"],
        "authority_path": relative,
        "authority_file_sha256": hashlib.sha256(raw).hexdigest(),
        "authority_content_sha256": claimed_content,
        "science_anchor": dict(science),
        "consumer_head_sha": head,
        "expected_ci": normalized_ci,
        "scientific_evidence": False,
        "provider_calls": 0,
        "science_dispatch_authority": False,
    }


def verify_publication_consumer_ci_receipt(
    repo_root: Path | str,
    authority_path: Path | str,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a CI job against the descendant publication authority."""

    root = Path(repo_root).resolve()
    authority = load_publication_consumer_ci_authority(root, authority_path)
    normalized = _validate_complete_ci_job_receipt(root, receipt)
    verify_expected_ci_matches_receipt(authority["expected_ci"], normalized)
    if normalized["head_sha"] != authority["consumer_head_sha"]:
        raise CIReleaseReceiptError(
            "publication consumer CI receipt HEAD differs from the verified consumer"
        )
    return {
        **authority,
        "ci_execution_status": "current-job-pass",
        "verified_job": {
            "run_id": normalized["run_id"],
            "run_attempt": normalized["run_attempt"],
            "job_name": normalized["job_name"],
            "runner_os": normalized["runner_os"],
            "head_sha": normalized["head_sha"],
            "receipt_sha256": normalized["receipt_sha256"],
        },
    }


def _validate_complete_ci_job_receipt(
    repo_root: Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping) or set(receipt) != _CI_JOB_RECEIPT_FIELDS:
        raise CIReleaseReceiptError(
            "publication consumer CI job receipt keys mismatch"
        )
    try:
        normalized = json.loads(json.dumps(receipt, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise CIReleaseReceiptError(
            "publication consumer CI job receipt is not finite JSON"
        ) from exc
    claimed = _sha256(
        normalized.pop("receipt_sha256", None),
        "publication consumer CI job receipt receipt_sha256",
    )
    if canonical_sha256(normalized) != claimed:
        raise CIReleaseReceiptError(
            "publication consumer CI job receipt self-hash mismatch"
        )
    if (
        normalized.get("schema_version") != CI_JOB_RECEIPT_SCHEMA_VERSION
        or normalized.get("status") != "pass"
    ):
        raise CIReleaseReceiptError(
            "publication consumer CI job receipt identity or status drifted"
        )
    for key in (
        "repository",
        "head_sha",
        "job_name",
        "job_key",
        "runner_os",
        "workflow_name",
        "workflow_file",
        "workflow_ref",
        "workflow_source_sha",
        "workflow_blob_oid",
    ):
        _environment_text(normalized, key)
    for key in (
        "run_id",
        "run_attempt",
        "test_count",
        "compiled_source_count",
        "sealed_manifest_count",
    ):
        _positive_int(normalized.get(key), key)
    skipped = normalized.get("skipped_test_count")
    if (
        isinstance(skipped, bool)
        or not isinstance(skipped, int)
        or skipped < 0
        or skipped > normalized["test_count"]
    ):
        raise CIReleaseReceiptError(
            "publication consumer CI skipped_test_count is invalid"
        )
    for key in (
        "test_collection_sha256",
        "compiled_source_inventory_sha256",
        "sealed_manifest_inventory_sha256",
        "workflow_file_sha256",
    ):
        _sha256(normalized.get(key), key)
    for key in ("head_sha", "workflow_source_sha", "workflow_blob_oid"):
        value = normalized[key]
        if len(value) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise CIReleaseReceiptError(f"publication consumer CI {key} is invalid")
    if (
        normalized["repository"] != _V2115_REPOSITORY
        or normalized["workflow_name"] != "Verified memory CI"
        or normalized["workflow_file"] != _WORKFLOW_FILE
        or normalized["workflow_source_sha"] != normalized["head_sha"]
        or normalized["job_key"] != "verify"
    ):
        raise CIReleaseReceiptError(
            "publication consumer CI workflow identity drifted"
        )
    workflow_ref_prefix = f"{_V2115_REPOSITORY}/{_WORKFLOW_FILE}@refs/"
    if (
        not normalized["workflow_ref"].startswith(workflow_ref_prefix)
        or normalized["workflow_ref"] == workflow_ref_prefix
    ):
        raise CIReleaseReceiptError(
            "publication consumer CI workflow ref drifted"
        )
    expected_job_by_os = {
        "Linux": "Python 3.12.7 / ubuntu-24.04",
        "macOS": "Python 3.12.7 / macos-14",
    }
    if normalized["job_name"] != expected_job_by_os.get(normalized["runner_os"]):
        raise CIReleaseReceiptError(
            "publication consumer CI runner and job identity differ"
        )
    workflow = repo_root / _WORKFLOW_FILE
    if workflow.is_symlink() or not workflow.is_file():
        raise CIReleaseReceiptError(
            "publication consumer CI workflow is missing or not regular"
        )
    if hashlib.sha256(workflow.read_bytes()).hexdigest() != normalized[
        "workflow_file_sha256"
    ]:
        raise CIReleaseReceiptError(
            "publication consumer CI workflow file hash drifted"
        )
    workflow_blob = _git_line(
        repo_root,
        ("git", "rev-parse", "--verify", f"HEAD:{_WORKFLOW_FILE}"),
    )
    if workflow_blob != normalized["workflow_blob_oid"]:
        raise CIReleaseReceiptError(
            "publication consumer CI workflow blob drifted"
        )
    return {**normalized, "receipt_sha256": claimed}


def build_publication_consumer_ci_receipt(
    repo_root: Path | str,
    *,
    authority_path: Path | str,
    ci_job_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Wrap a passing job receipt in an explicitly non-scientific envelope."""

    authority = verify_publication_consumer_ci_receipt(
        repo_root,
        authority_path,
        ci_job_receipt,
    )
    payload = {
        "schema_version": PUBLICATION_CONSUMER_CI_RECEIPT_SCHEMA_VERSION,
        "status": "pass",
        "scientific_evidence": False,
        "provider_calls": 0,
        "science_dispatch_authority": False,
        "authority": authority,
        "ci_job_receipt": dict(ci_job_receipt),
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
        raise CIReleaseReceiptError("tracked file inventory is not UTF-8") from exc
    if rows != sorted(rows) or len(set(rows)) != len(rows):
        raise CIReleaseReceiptError("tracked file inventory is not sorted and unique")
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


def compile_sources(repo_root: Path | str, output: Path | str) -> dict[str, Any]:
    """Compile every tracked Python source and persist its inventory hash."""

    root = Path(repo_root).resolve()
    sources = discover_tracked_files(root, ("*.py",))
    inventory = build_source_inventory(sources)
    failures = [
        relative
        for relative in sources
        if not compileall.compile_file(root / relative, force=True, quiet=1)
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
    contract_path: Path | str,
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
    contract_source = Path(contract_path)
    if not contract_source.is_absolute():
        contract_source = root / contract_source
    verify_contract_ci_receipt(contract_source, receipt)
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


def emit_publication_consumer_ci_receipt(
    repo_root: Path | str,
    *,
    collection_path: Path | str,
    source_path: Path | str,
    junit_path: Path | str,
    output_path: Path | str,
    authority_path: Path | str,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Seal one descendant-consumer job under its non-scientific authority."""

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
    ci_job_receipt = build_ci_job_receipt(
        root,
        collection_inventory=collection,
        source_inventory=sources,
        junit_summary=junit,
        environment=os.environ if environment is None else environment,
        manifest_paths=manifests,
    )
    authority_source = Path(authority_path)
    if not authority_source.is_absolute():
        authority_source = root / authority_source
    receipt = build_publication_consumer_ci_receipt(
        root,
        authority_path=authority_source,
        ci_job_receipt=ci_job_receipt,
    )
    _write_json(Path(output_path), receipt)
    print(
        PUBLICATION_CONSUMER_CI_RECEIPT_LOG_PREFIX
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
        raise CIReleaseReceiptError("collection inventory schema mismatch")
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
    if value.get("schema_version") != COMPILED_SOURCE_INVENTORY_SCHEMA_VERSION:
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


def _strict_json_object(raw: bytes, name: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in rows:
            if key in value:
                raise CIReleaseReceiptError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CIReleaseReceiptError(f"{name} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise CIReleaseReceiptError(f"{name} must contain one JSON object")
    return value


def _validate_source_manifest_anchor(value: Mapping[str, Any]) -> dict[str, str]:
    if set(value) != {
        "path",
        "schema_version",
        "file_sha256",
        "content_sha256",
    }:
        raise CIReleaseReceiptError("scientific source manifest anchor keys mismatch")
    path = value.get("path")
    schema = value.get("schema_version")
    if (
        not isinstance(path, str)
        or not path.startswith("experiments/")
        or not path.endswith("_source_manifest.json")
        or "\\" in path
        or "\x00" in path
        or path != Path(path).as_posix()
        or Path(path).is_absolute()
        or any(part in {"", ".", ".."} for part in Path(path).parts)
    ):
        raise CIReleaseReceiptError(
            "scientific source manifest path must be normalized below experiments/"
        )
    if (
        not isinstance(schema, str)
        or not schema
        or schema != schema.strip()
        or "\n" in schema
        or "\r" in schema
    ):
        raise CIReleaseReceiptError(
            "scientific source manifest schema must be normalized text"
        )
    if value.get("file_sha256") is None or value.get("content_sha256") is None:
        raise CIReleaseReceiptError(
            "scientific source manifest anchor hashes must be sealed before CI"
        )
    return {
        "path": path,
        "schema_version": schema,
        "file_sha256": _sha256(value.get("file_sha256"), "file_sha256"),
        "content_sha256": _sha256(value.get("content_sha256"), "content_sha256"),
    }


def _guarded_regular_file(root: Path, relative: str) -> Path:
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise CIReleaseReceiptError(
                f"scientific source manifest path contains a symlink: {relative}"
            )
    if not current.is_file():
        raise CIReleaseReceiptError(
            f"scientific source manifest is missing or not regular: {relative}"
        )
    return current


def _guarded_authority_file(root: Path, relative: str) -> Path:
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise CIReleaseReceiptError(
                "publication consumer CI authority path contains a symlink"
            )
    if not current.is_file():
        raise CIReleaseReceiptError(
            "publication consumer CI authority is missing or not regular"
        )
    return current


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
    if len(completed.stdout.decode("utf-8", "strict").splitlines()) != 1 or not value:
        raise CIReleaseReceiptError("Git output is not one normalized line")
    return value


def _git_success(root: Path, argv: Sequence[str], message: str) -> None:
    completed = subprocess.run(
        tuple(argv),
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0 or completed.stdout or completed.stderr:
        raise CIReleaseReceiptError(message)


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
        raise CIReleaseReceiptError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect-tests")
    collect_parser.add_argument("--output", required=True)
    compile_parser = subparsers.add_parser("compile-sources")
    compile_parser.add_argument("--output", required=True)
    source_manifest_parser = subparsers.add_parser("verify-source-manifests")
    source_manifest_parser.add_argument("--output", required=True)
    emit_parser = subparsers.add_parser("emit")
    emit_parser.add_argument("--collection", required=True)
    emit_parser.add_argument("--sources", required=True)
    emit_parser.add_argument("--junit", required=True)
    emit_parser.add_argument("--output", required=True)
    emit_parser.add_argument("--contract", required=True)
    publication_parser = subparsers.add_parser("emit-publication-consumer")
    publication_parser.add_argument("--collection", required=True)
    publication_parser.add_argument("--sources", required=True)
    publication_parser.add_argument("--junit", required=True)
    publication_parser.add_argument("--output", required=True)
    publication_parser.add_argument("--authority", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "collect-tests":
            collect_tests(args.output)
        elif args.command == "compile-sources":
            compile_sources(Path.cwd(), args.output)
        elif args.command == "verify-source-manifests":
            inventory = build_scientific_source_manifest_inventory(Path.cwd())
            _write_json(Path(args.output), inventory)
        elif args.command == "emit":
            emit_ci_job_receipt(
                Path.cwd(),
                collection_path=args.collection,
                source_path=args.sources,
                junit_path=args.junit,
                output_path=args.output,
                contract_path=args.contract,
            )
        else:
            emit_publication_consumer_ci_receipt(
                Path.cwd(),
                collection_path=args.collection,
                source_path=args.sources,
                junit_path=args.junit,
                output_path=args.output,
                authority_path=args.authority,
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
    "PUBLICATION_CONSUMER_CI_AUTHORITY_RELATIVE",
    "PUBLICATION_CONSUMER_CI_AUTHORITY_SCHEMA_VERSION",
    "PUBLICATION_CONSUMER_CI_RECEIPT_LOG_PREFIX",
    "PUBLICATION_CONSUMER_CI_RECEIPT_SCHEMA_VERSION",
    "V2115_SCIENCE_TAG_OBJECT",
    "SCIENTIFIC_SOURCE_MANIFEST_ANCHORS",
    "SCIENTIFIC_SOURCE_MANIFEST_INVENTORY_SCHEMA_VERSION",
    "build_ci_job_receipt",
    "build_collection_inventory",
    "build_publication_consumer_ci_receipt",
    "build_scientific_source_manifest_inventory",
    "build_source_inventory",
    "verify_contract_ci_receipt",
    "verify_expected_ci_matches_receipt",
    "collect_tests",
    "compile_sources",
    "discover_tracked_files",
    "emit_ci_job_receipt",
    "emit_publication_consumer_ci_receipt",
    "load_publication_consumer_ci_authority",
    "parse_junit_summary",
    "verify_publication_consumer_ci_receipt",
]
