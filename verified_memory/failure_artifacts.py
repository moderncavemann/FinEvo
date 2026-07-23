"""Minimal sealed receipts for failed bounded provider runs.

Successful runs use the richer stream manifests.  A failure can occur before
those streams are returned to the CLI, but dispatched calls and their budget
ledger must still remain auditable.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


FAILURE_RECEIPT_SCHEMA_VERSION = "verified-failure-receipt-v1"


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_failure_receipt(
    run_dir: str | Path,
    *,
    scope: str,
    error: BaseException,
    budget_snapshot: Mapping[str, Any],
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    git_commit: str,
    git_dirty: bool,
) -> Path:
    """Write a content-addressed failure and budget receipt.

    This intentionally makes no claim that partial in-memory simulation streams
    were persisted.  It records that boundary explicitly.
    """

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise FileExistsError("refusing to add a failure receipt to a completed run")
    error_record: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    structured_failure = getattr(error, "failure", None)
    if structured_failure is not None and callable(
        getattr(structured_failure, "to_dict", None)
    ):
        error_record["details"] = structured_failure.to_dict()
    receipt = {
        "schema_version": FAILURE_RECEIPT_SCHEMA_VERSION,
        "status": "failed",
        "scope": str(scope),
        "error": error_record,
        "budget_snapshot": dict(budget_snapshot),
        "config": dict(config),
        "provenance": dict(provenance),
        "git": {"commit": str(git_commit), "dirty": bool(git_dirty)},
        "partial_streams_persisted": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_bytes = (_canonical(receipt) + "\n").encode("utf-8")
    receipt_path = root / "failure.json"
    receipt_path.write_bytes(receipt_bytes)
    manifest = {
        "schema_version": FAILURE_RECEIPT_SCHEMA_VERSION,
        "status": "failed",
        "failure_file": "failure.json",
        "failure_sha256": _sha256_bytes(receipt_bytes),
        "failure_size_bytes": len(receipt_bytes),
    }
    manifest_core = _canonical(manifest)
    manifest["manifest_sha256"] = _sha256_bytes(manifest_core.encode("utf-8"))
    manifest_path = root / "failure_manifest.json"
    manifest_path.write_text(_canonical(manifest) + "\n", encoding="utf-8")
    verify_failure_receipt(root)
    return manifest_path


def verify_failure_receipt(run_dir: str | Path) -> Mapping[str, Any]:
    root = Path(run_dir)
    manifest = json.loads((root / "failure_manifest.json").read_text(encoding="utf-8"))
    receipt_bytes = (root / str(manifest["failure_file"])).read_bytes()
    if _sha256_bytes(receipt_bytes) != manifest.get("failure_sha256"):
        raise ValueError("failure receipt hash mismatch")
    if len(receipt_bytes) != manifest.get("failure_size_bytes"):
        raise ValueError("failure receipt size mismatch")
    manifest_core = dict(manifest)
    expected_manifest_hash = manifest_core.pop("manifest_sha256", None)
    if _sha256_bytes(_canonical(manifest_core).encode("utf-8")) != expected_manifest_hash:
        raise ValueError("failure manifest hash mismatch")
    receipt = json.loads(receipt_bytes)
    if receipt.get("status") != "failed":
        raise ValueError("failure receipt has a non-failed status")
    return receipt


__all__ = [
    "FAILURE_RECEIPT_SCHEMA_VERSION",
    "verify_failure_receipt",
    "write_failure_receipt",
]
