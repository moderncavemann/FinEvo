"""Immutable V2.7 -> V2.8 prerequisite and Stage-0 import primitives.

V2.7 is a terminal ``complete-with-no-go`` release.  V2.8 preserves that
denominator and copies the *entire* V2.7 raw namespace byte-for-byte.  The
only reusable cells are the completed V2.7 parent prerequisite and the
fourteen completed V2.6 Stage-0 calibration cells inside V2.7's immutable
nested snapshot.

The nested V2.6 q-ref artifact is deliberately not imported.  It is bound
only as an audit-equivalence reference for V2.8's fresh, zero-hosted-provider
q-ref regeneration (48 deterministic ``ScriptedDiagnosticProvider`` calls).
No function in this module constructs a provider.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from statistics import median
import subprocess
from typing import Any, Mapping, Sequence

from .artifacts import verify_manifest
from .pilot_budget import ParentBudgetDebit
from .pilot_contract import (
    PilotContract,
    PilotRunSpec,
    canonical_sha256,
    load_pilot_contract,
)
from .pilot_v24_parent_import import (
    CANONICALIZATION,
    PilotV24ParentImportError,
    _atomic_exact_json,
    _git,
    _guarded_file,
    _json_copy,
    _normalized_relative,
    _real_root,
    _repo_relative,
    _seal,
    _sha256,
    _strict_json,
    _verify_self_hash,
)
from .pilot_v27_stage0_import import (
    PilotV27Stage0ImportError,
    _atomic_exact_bytes_no_follow as _v27_atomic_exact_bytes_no_follow,
    verify_v27_parent_import_receipt,
)
from .runner import PreflightP95Reservation, verify_provider_call_journal


V28_CONTRACT_ID = "finevo-pilot-v2.8"
V28_SCIENCE_TAG = "pilot-v2.8-science"
V28_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.8-source-manifest-v1"
V28_PARENT_IMPORT_SCHEMA_VERSION = "finevo-pilot-v2.8-parent-import-v1"
V28_RESEALED_P95_AUTHORITY_SCHEMA_VERSION = (
    "finevo-pilot-v2.8-inherited-observed-p95-authority-v1"
)
V28_RESEALED_P95_SOURCE_KIND = "v2.7-terminal-parent-import-v2.8"
V28_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_8_source_manifest.json"
)
V28_EXPANDED_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_8.yaml")
V28_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.8/raw")
V28_SNAPSHOT_RELATIVE = PurePosixPath("parent-import/v2_7_raw_snapshot")
V28_ALLOWED_P95_PROFILES = ("gpt52_main", "llama33_local_controlled")

V27_CONTRACT_ID = "finevo-pilot-v2.7"
V27_CONTRACT_CANONICAL_SHA256 = (
    "938627d42ec8ec78e8424793797593736b79936b00813b81259af54e6df6779f"
)
V27_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_7.yaml")
V27_CONTRACT_FILE_SHA256 = (
    "93bcabeca5be3cc66e30b28d3b616a3327ab77703c9462591aacecb02d1805fc"
)
V27_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_7_source_manifest.json"
)
V27_SOURCE_MANIFEST_FILE_SHA256 = (
    "ee0ef62f5dcde9fc820aef6d23d1ce5a8c5bca7b9f20486bf42233f18763a1c8"
)
V27_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "f195661d01d0aa6742d9e2f2658b6b1acb38715ddbd43e4e5fd375309d78dbe4"
)
V27_SCIENCE_TAG = "pilot-v2.7-science"
V27_SCIENCE_TAG_OBJECT = "1e72d4f03df0ed5d12436a2d008da508bcb938f2"
V27_SCIENCE_COMMIT = "60566410f38f7842169e93ae9822f180235b60b6"
V27_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.7/raw")
V27_RAW_FILE_COUNT = 242
V27_RAW_STORAGE_BYTES = 13_500_493
V27_RAW_INVENTORY_SHA256 = (
    "3fe3d46d6d3ae7c4d320ffc8ec8f69172684b22626e7c54c032c03017db45742"
)
V27_RUN_LEDGER_FILE_SHA256 = (
    "4ea0b36668066f86177f042d90b2f9c69ec6411bf1cb83b9bb89b4b607db66f8"
)
V27_RUN_LEDGER_INTERNAL_SHA256 = (
    "ab532bb56232efbc42d6e5f48c9f80c451f461a732cf2607774f6055de9deb4a"
)
V27_RUN_LEDGER_EVENT_COUNT = 213
V27_RUN_LEDGER_EVENT_HEAD = (
    "abc9e708cb1641e7ea699a618aa69c350ab03d592b4deb6d734b444d0c92b9b3"
)
V27_BUDGET_LEDGER_FILE_SHA256 = (
    "2436866c9968ebe7f64088e88c3fee90b188a2c6a87d24266408d3848ac5d72e"
)
V27_BUDGET_LEDGER_INTERNAL_SHA256 = (
    "70ff3f40bbebaea766c6403fc1f2879af9002faff287a112a39c2ce405d92170"
)
V27_BUDGET_LEDGER_EVENT_COUNT = 6
V27_BUDGET_LEDGER_EVENT_HEAD = (
    "33eeee00601b195d2b70a087687c82b69907295435b1f3fa8a1302d092a3af72"
)
V27_QREF_FAILURE_STAGE_RECEIPT_FILE_SHA256 = (
    "45ad6725749333852902cfade0b0858bfbe9a85af7648556343f7d350510201d"
)
V27_QREF_FAILURE_STAGE_RECEIPT_CONTENT_SHA256 = (
    "652106fcba0c3849979e39a1d84fb42487278e2efa1b43cb06c4cffb5f0786f8"
)
V27_PARENT_IMPORT_RECEIPT_FILE_SHA256 = (
    "1bb43cf6af6f74081a2413bb78add8dc08adef2db207d1bf5865d8d7772ad0ed"
)
V27_PARENT_IMPORT_RECEIPT_CONTENT_SHA256 = (
    "e63fee22f0b0108f3402f97e912bd90705dedce090c65081d3d09a2d9d2fa12d"
)

V27_EVIDENCE_ROOT = PurePosixPath("evidence/current_v2/pilot-v2.7")
V27_EVIDENCE_PUBLICATION_COMMIT = (
    "f15a26418264b5de31f53dbe7c46c1949761fcb6"
)
V27_EVIDENCE_MERGE_COMMIT = (
    "e951aa865186a7c2e841316fc6bb08a716aeaf80"
)
V27_EVIDENCE_CHECKSUMS_FILE_SHA256 = (
    "b28889b0fc590ec884c69fdf43f88b01ce8f384491168a031ac5fdb2a6b3caad"
)
V27_EVIDENCE_PACKAGE_FILE_SHA256 = (
    "1b44a8984b61f00cbae4851a599674fb3e0479ca60d3259961460f99519e23bb"
)
V27_EVIDENCE_AGGREGATE_FILE_SHA256 = (
    "dfd934df48e9d83c02abc63c7d365b776d88d9ac99eb94381768e3c506ae82fd"
)
V27_EVIDENCE_FAILURE_FILE_SHA256 = (
    "4906f40ca19854d68647859a89e44f432368a41d28d49461228f2bb56796df3e"
)

V26_CONTRACT_ID = "finevo-pilot-v2.6"
V26_CONTRACT_CANONICAL_SHA256 = (
    "bb6b12d71227c423e5a67452dc496f26843dec74e359b9b04bf096dc17d0c509"
)
V26_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_6.yaml")
V26_NESTED_RELATIVE = PurePosixPath("parent-import/v2_6_raw_snapshot")
V26_RAW_FILE_COUNT = 228
V26_RAW_STORAGE_BYTES = 12_877_797
V26_RAW_INVENTORY_SHA256 = (
    "cfc3365828ba8fe2f75f11bb5117d8282d8f68f52653c411605dd3849f712105"
)
V26_QREF_VALUE = 63.50397933257746
V26_QREF_FILE_SHA256 = (
    "bf8cba5fd34a30b3b78b681f5be8bd617d3c43f7714f5876166a9a92734ed454"
)
V26_QREF_CONTENT_SHA256 = (
    "cbecfcfaf9a85badb049f0e5024d9ffe896369432b5982b97a71dad6970830f2"
)
V26_QREF_MANIFEST_FILE_SHA256 = (
    "8d299f2cb1646a810eb4311cbf054acd72c643694925a7cc6043dda1f7201d2b"
)
V26_STAGE0_RECEIPT_FILE_SHA256 = (
    "56392110d442896c72732da999bb67b879d685ef4c7c8dd3add75b62abb92359"
)
V26_STAGE0_RECEIPT_CONTENT_SHA256 = (
    "615394abdd55f1f1cdbd2c9a52df2b6a9f91ef3888ab58d2952bb95777ca23c4"
)

V28_CUMULATIVE_DEBIT = ParentBudgetDebit(
    parent_contract_sha256=V27_CONTRACT_CANONICAL_SHA256,
    parent_run_ledger_sha256=V27_RUN_LEDGER_INTERNAL_SHA256,
    parent_budget_ledger_sha256=V27_BUDGET_LEDGER_INTERNAL_SHA256,
    stage_bucket="parent_v23",
    cost_usd=3.212770875,
    hosted_completions=184,
    storage_bytes=32_158_175,
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_STAGE0_PROFILES = (
    "center",
    "psi-1",
    "psi-4",
    "nu-0.5",
    "nu-2",
    "q0-0.5x",
    "q0-2x",
)
_STAGE0_SEEDS = (1942013315, 760687867)

_V27_P95_SOURCES: dict[str, dict[str, Any]] = {
    "gpt52_main": {
        "runtime_model": "openai/gpt-5.2-2025-12-11",
        "served_model": "gpt-5.2-2025-12-11",
        "authority": {
            "path": (
                "experiment_results/pilot-v2.7/raw/parent-import/observed_p95/"
                "gpt52_main/observed_p95_authority_receipt.json"
            ),
            "schema_version":
            "finevo-pilot-v2.7-inherited-observed-p95-authority-v1",
            "file_sha256":
            "bc9fd377a0f4032893824853f7800af347cea9932252edae14063777afeb64bd",
            "content_sha256":
            "c1bf68934b11ce6b61d692bab84d962aa337ae57a3e36518da209955908ee568",
        },
        "projection": {
            "path": (
                "experiment_results/pilot-v2.7/raw/parent-import/observed_p95/"
                "gpt52_main/projection_p95.json"
            ),
            "schema_version": "finevo-pilot-projection-p95-v1",
            "file_sha256":
            "9577ab95f50753546a904c2e9090b3201dc1733868e4ca79f79a8c4faa4b9af5",
            "content_sha256":
            "7419dc14b8e34910014e8a44976a60d32ef67551c40be0edc421b0086dc1e95e",
        },
    },
    "llama33_local_controlled": {
        "runtime_model": "ollama/llama3.3:70b-instruct-q4_K_M",
        "served_model": "llama3.3:70b-instruct-q4_K_M",
        "authority": {
            "path": (
                "experiment_results/pilot-v2.7/raw/parent-import/observed_p95/"
                "llama33_local_controlled/observed_p95_authority_receipt.json"
            ),
            "schema_version":
            "finevo-pilot-v2.7-inherited-observed-p95-authority-v1",
            "file_sha256":
            "ce335863d0b863eaf7fb37dd47d78a86d3b7c737777990a9010b01abec9e92dd",
            "content_sha256":
            "8681021c3a67d2a1a3f609dcf0c314c4be2136c81cba7d338666c051f08e654d",
        },
        "projection": {
            "path": (
                "experiment_results/pilot-v2.7/raw/parent-import/observed_p95/"
                "llama33_local_controlled/projection_p95.json"
            ),
            "schema_version": "finevo-pilot-projection-p95-v1",
            "file_sha256":
            "5a4c71ec90d2b06444614cb9c74f3484e3929392f884192db27e8e1c576ccf04",
            "content_sha256":
            "03fa77903fb50e5dae18d1410ab5d46120cc5e17425b6851a8fcef7e6a37ff75",
        },
    },
}


class PilotV28Stage0ImportError(RuntimeError):
    """Raised before immutable V2.7/V2.6 authority can enter V2.8."""


def _translate(exc: Exception) -> PilotV28Stage0ImportError:
    return PilotV28Stage0ImportError(str(exc))


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotV28Stage0ImportError(f"{name} must be an object")
    return value


def _verify_v2_stage_receipt_self_hash(
    value: Mapping[str, Any],
    *,
    name: str,
) -> None:
    """Verify the V2 stage-receipt convention (integrity fully excluded)."""

    unsigned = _json_copy(value)
    integrity = unsigned.pop("integrity", None)
    if (
        value.get("schema_version") != "finevo-pilot-stage-receipt-v2"
        or not isinstance(integrity, Mapping)
        or set(integrity) != {"canonicalization", "content_sha256"}
        or integrity.get("canonicalization") != CANONICALIZATION
        or integrity.get("content_sha256") != canonical_sha256(unsigned)
    ):
        raise PilotV28Stage0ImportError(f"{name} content hash mismatch")


def _strict_file(
    root: Path,
    relative: PurePosixPath,
    *,
    name: str,
    expected_sha256: str | None = None,
) -> tuple[Path, bytes, dict[str, Any]]:
    try:
        path, raw = _guarded_file(root, relative, name=name)
        value = _strict_json(raw, name=name)
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if expected_sha256 is not None and _sha256(raw) != expected_sha256:
        raise PilotV28Stage0ImportError(f"{name} file hash drifted")
    return path, raw, value


def _strict_jsonl_rows(
    root: Path,
    relative: PurePosixPath,
    *,
    name: str,
) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        _, raw = _guarded_file(root, relative, name=name)
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise PilotV28Stage0ImportError(
                f"{name} contains an empty JSONL row"
            )
        try:
            rows.append(
                _strict_json(line, name=f"{name} row {index}")
            )
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
    return raw, rows


def _inventory(
    root: Path,
    *,
    declared_root: PurePosixPath,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise PilotV28Stage0ImportError("raw snapshot root is unavailable")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PilotV28Stage0ImportError(
                "raw snapshot inventory contains a symlink"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise PilotV28Stage0ImportError(
                "raw snapshot inventory contains a non-regular entry"
            )
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise PilotV28Stage0ImportError(
                "raw snapshot file changed during inventory"
            )
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "byte_size": len(raw),
                "sha256": _sha256(raw),
            }
        )
    canonical = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return rows, {
        "root": declared_root.as_posix(),
        "inventory_schema_version": "finevo-raw-tree-inventory-v1",
        "inventory_canonicalization": "json-sort-keys-compact-utf8-v1",
        "file_count": len(rows),
        "storage_bytes": sum(row["byte_size"] for row in rows),
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _verify_exact_inventory(
    root: Path,
    *,
    declared_root: PurePosixPath,
    file_count: int,
    storage_bytes: int,
    inventory_sha256: str,
    name: str,
) -> list[dict[str, Any]]:
    rows, summary = _inventory(root, declared_root=declared_root)
    if summary != {
        "root": declared_root.as_posix(),
        "inventory_schema_version": "finevo-raw-tree-inventory-v1",
        "inventory_canonicalization": "json-sort-keys-compact-utf8-v1",
        "file_count": file_count,
        "storage_bytes": storage_bytes,
        "inventory_sha256": inventory_sha256,
    }:
        raise PilotV28Stage0ImportError(f"{name} raw-tree inventory drifted")
    return rows


def _verify_exact_v27_inventory(raw_root: Path) -> list[dict[str, Any]]:
    return _verify_exact_inventory(
        raw_root,
        declared_root=V27_RAW_ROOT,
        file_count=V27_RAW_FILE_COUNT,
        storage_bytes=V27_RAW_STORAGE_BYTES,
        inventory_sha256=V27_RAW_INVENTORY_SHA256,
        name="V2.7 complete",
    )


def _verify_exact_nested_v26_inventory(nested_root: Path) -> list[dict[str, Any]]:
    return _verify_exact_inventory(
        nested_root,
        declared_root=V27_RAW_ROOT / V26_NESTED_RELATIVE,
        file_count=V26_RAW_FILE_COUNT,
        storage_bytes=V26_RAW_STORAGE_BYTES,
        inventory_sha256=V26_RAW_INVENTORY_SHA256,
        name="nested V2.6",
    )


def _git_blob_bytes(
    repo_root: Path,
    *,
    commit: str,
    relative: PurePosixPath,
) -> bytes:
    if _COMMIT_RE.fullmatch(commit) is None:
        raise PilotV28Stage0ImportError("git blob commit must be lowercase 40-hex")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
        and not key.startswith("DYLD_")
        and key not in {"LD_LIBRARY_PATH", "LD_PRELOAD"}
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    binary = Path("/usr/bin/git")
    if not binary.is_file():
        raise PilotV28Stage0ImportError("system git binary is unavailable")
    completed = subprocess.run(
        (
            str(binary),
            "show",
            f"{commit}:{relative.as_posix()}",
        ),
        cwd=repo_root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise PilotV28Stage0ImportError(
            f"git blob is unavailable: {commit}:{relative.as_posix()}"
        )
    return completed.stdout


def _artifact_binding(
    root: Path,
    relative: PurePosixPath,
) -> dict[str, Any]:
    try:
        _, raw = _guarded_file(root, relative, name=relative.as_posix())
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    return {
        "path": relative.as_posix(),
        "file_sha256": _sha256(raw),
        "byte_size": len(raw),
    }


def _verify_v27_published_evidence(child_root: Path) -> dict[str, Any]:
    """Verify the tracked V2.7 package and both publication commits."""

    for commit in (
        V27_EVIDENCE_PUBLICATION_COMMIT,
        V27_EVIDENCE_MERGE_COMMIT,
    ):
        if _git(child_root, "cat-file", "-t", commit) != "commit":
            raise PilotV28Stage0ImportError(
                f"V2.7 evidence commit is unavailable: {commit}"
            )
    merge_parents = _git(
        child_root,
        "rev-list",
        "--parents",
        "-n",
        "1",
        V27_EVIDENCE_MERGE_COMMIT,
    ).split()
    if (
        not merge_parents
        or merge_parents[0] != V27_EVIDENCE_MERGE_COMMIT
        or V27_EVIDENCE_PUBLICATION_COMMIT not in merge_parents[1:]
    ):
        raise PilotV28Stage0ImportError(
            "V2.7 evidence merge does not bind the publication commit"
        )
    try:
        _git(
            child_root,
            "merge-base",
            "--is-ancestor",
            V27_EVIDENCE_MERGE_COMMIT,
            "HEAD",
        )
    except PilotV24ParentImportError as exc:
        raise PilotV28Stage0ImportError(
            "V2.7 evidence merge is not an ancestor of the child checkout"
        ) from exc

    fixed = {
        "checksums.json": V27_EVIDENCE_CHECKSUMS_FILE_SHA256,
        "package_manifest.json": V27_EVIDENCE_PACKAGE_FILE_SHA256,
        "aggregate.json": V27_EVIDENCE_AGGREGATE_FILE_SHA256,
        "failure_ledger.json": V27_EVIDENCE_FAILURE_FILE_SHA256,
    }
    loaded: dict[str, dict[str, Any]] = {}
    raw_by_name: dict[str, bytes] = {}
    for name, expected_hash in fixed.items():
        relative = V27_EVIDENCE_ROOT / name
        _, raw, value = _strict_file(
            child_root,
            relative,
            name=f"published V2.7 {name}",
            expected_sha256=expected_hash,
        )
        loaded[name] = value
        raw_by_name[name] = raw
        for commit in (
            V27_EVIDENCE_PUBLICATION_COMMIT,
            V27_EVIDENCE_MERGE_COMMIT,
        ):
            if _git_blob_bytes(
                child_root,
                commit=commit,
                relative=relative,
            ) != raw:
                raise PilotV28Stage0ImportError(
                    f"published V2.7 {name} differs at {commit}"
                )

    package = loaded["package_manifest.json"]
    if (
        package.get("schema_version")
        != "finevo-pilot-v2.7-evidence-package-v1"
        or package.get("contract_id") != V27_CONTRACT_ID
        or package.get("contract_sha256")
        != V27_CONTRACT_CANONICAL_SHA256
        or package.get("pilot_tag") != V27_SCIENCE_TAG
        or package.get("resolved_git_commit") != V27_SCIENCE_COMMIT
        or package.get("publication_status") != "complete-with-no-go"
        or package.get("scientific_complete") is not False
        or package.get("scientific_matrix_complete") is not False
        or package.get("scientific_claim_gates_supported") is not False
        or package.get("lane_separated") is not True
        or package.get("direction_counts_merged") is not False
    ):
        raise PilotV28Stage0ImportError(
            "published V2.7 evidence claim boundary drifted"
        )

    checksums = loaded["checksums.json"]
    rows = checksums.get("files")
    if (
        checksums.get("schema_version")
        != "finevo-pilot-package-checksums-v1"
        or checksums.get("contract_sha256")
        != V27_CONTRACT_CANONICAL_SHA256
        or not isinstance(rows, list)
        or len(rows) != 14
    ):
        raise PilotV28Stage0ImportError(
            "published V2.7 checksum inventory is malformed"
        )
    observed: set[str] = set()
    checksum_bindings: list[dict[str, Any]] = []
    for row_value in rows:
        row = _mapping(row_value, name="published V2.7 checksum row")
        try:
            relative = _normalized_relative(
                str(row.get("path", "")),
                required_top=None,
                name="published V2.7 checksum path",
            )
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
        if relative.as_posix() in observed:
            raise PilotV28Stage0ImportError(
                "published V2.7 checksum path is duplicated"
            )
        observed.add(relative.as_posix())
        package_relative = V27_EVIDENCE_ROOT / relative
        try:
            _, raw = _guarded_file(
                child_root,
                package_relative,
                name=f"published V2.7 file {relative.as_posix()}",
            )
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
        if len(raw) != row.get("byte_size") or _sha256(raw) != row.get("sha256"):
            raise PilotV28Stage0ImportError(
                f"published V2.7 checksum drifted for {relative.as_posix()}"
            )
        for commit in (
            V27_EVIDENCE_PUBLICATION_COMMIT,
            V27_EVIDENCE_MERGE_COMMIT,
        ):
            if _git_blob_bytes(
                child_root,
                commit=commit,
                relative=package_relative,
            ) != raw:
                raise PilotV28Stage0ImportError(
                    f"V2.7 evidence file differs at {commit}: "
                    f"{relative.as_posix()}"
                )
        checksum_bindings.append(_json_copy(row))
    if observed != set(package.get("published_files", ())) | {
        "package_manifest.json"
    }:
        raise PilotV28Stage0ImportError(
            "published V2.7 package/checksum inventory differs"
        )

    aggregate = loaded["aggregate.json"]
    denominator = _mapping(
        aggregate.get("denominator"),
        name="V2.7 aggregate denominator",
    )
    budget = _mapping(aggregate.get("budget"), name="V2.7 aggregate budget")
    if (
        aggregate.get("publication_status") != "complete-with-no-go"
        or denominator.get("expected_count") != 211
        or denominator.get("observed_ledger_count") != 211
        or denominator.get("all_rows_present") is not True
        or denominator.get("all_rows_terminal") is not True
        or denominator.get("status_counts")
        != {"complete": 1, "integrity-stopped": 210}
        or budget.get("pass") is not True
        or budget.get("raw_root_storage_bytes") != V27_RAW_STORAGE_BYTES
        or budget.get("actual_totals")
        != {
            "cost_usd": 3.212770875,
            "completions": 184,
            "storage_bytes": 32_158_175,
        }
    ):
        raise PilotV28Stage0ImportError(
            "published V2.7 denominator or budget drifted"
        )
    failure = loaded["failure_ledger.json"]
    if (
        failure.get("schema_version") != "finevo-pilot-failure-ledger-v1"
        or failure.get("contract_sha256")
        != V27_CONTRACT_CANONICAL_SHA256
        or failure.get("denominator") != denominator
        or not isinstance(failure.get("rows"), list)
        or len(failure["rows"]) != 210
        or any(
            row.get("status") != "integrity-stopped"
            for row in failure["rows"]
        )
    ):
        raise PilotV28Stage0ImportError(
            "published V2.7 failure denominator drifted"
        )
    return {
        "root": V27_EVIDENCE_ROOT.as_posix(),
        "schema_version": package["schema_version"],
        "publication_commit": V27_EVIDENCE_PUBLICATION_COMMIT,
        "merge_commit": V27_EVIDENCE_MERGE_COMMIT,
        "checksums": {
            "path": (V27_EVIDENCE_ROOT / "checksums.json").as_posix(),
            "file_sha256": V27_EVIDENCE_CHECKSUMS_FILE_SHA256,
            "entry_count": len(checksum_bindings),
            "files": checksum_bindings,
        },
        "package_manifest_file_sha256": V27_EVIDENCE_PACKAGE_FILE_SHA256,
        "aggregate_file_sha256": V27_EVIDENCE_AGGREGATE_FILE_SHA256,
        "failure_ledger_file_sha256": V27_EVIDENCE_FAILURE_FILE_SHA256,
        "publication_status": "complete-with-no-go",
        "scientific_complete": False,
        "scientific_matrix_complete": False,
        "scientific_claim_gates_supported": False,
    }


def _verify_v27_ledgers(raw_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _, _, run = _strict_file(
        raw_root,
        PurePosixPath("run_ledger.json"),
        name="V2.7 run ledger",
        expected_sha256=V27_RUN_LEDGER_FILE_SHA256,
    )
    _, _, budget = _strict_file(
        raw_root,
        PurePosixPath("budget_ledger.json"),
        name="V2.7 budget ledger",
        expected_sha256=V27_BUDGET_LEDGER_FILE_SHA256,
    )
    runs = _mapping(run.get("runs"), name="V2.7 run ledger rows")
    run_events = run.get("events")
    budget_events = budget.get("events")
    if (
        run.get("schema_version") != "finevo-pilot-run-ledger-v2"
        or run.get("contract_hash") != V27_CONTRACT_CANONICAL_SHA256
        or run.get("ledger_sha256") != V27_RUN_LEDGER_INTERNAL_SHA256
        or not isinstance(run_events, list)
        or len(run_events) != V27_RUN_LEDGER_EVENT_COUNT
        or run_events[-1].get("event_sha256") != V27_RUN_LEDGER_EVENT_HEAD
        or len(runs) != 211
        or Counter(str(row.get("status")) for row in runs.values())
        != Counter({"complete": 1, "integrity-stopped": 210})
        or budget.get("schema_version") != "finevo-pilot-budget-ledger-v2"
        or budget.get("contract_hash") != V27_CONTRACT_CANONICAL_SHA256
        or budget.get("ledger_sha256") != V27_BUDGET_LEDGER_INTERNAL_SHA256
        or not isinstance(budget_events, list)
        or len(budget_events) != V27_BUDGET_LEDGER_EVENT_COUNT
        or budget_events[-1].get("event_sha256")
        != V27_BUDGET_LEDGER_EVENT_HEAD
    ):
        raise PilotV28Stage0ImportError(
            "V2.7 terminal ledger identity or denominator drifted"
        )
    return run, budget


def _verify_v27_qref_failure(raw_root: Path) -> dict[str, Any]:
    _, raw, value = _strict_file(
        raw_root,
        PurePosixPath("q-ref-resolution/stage_receipt.json"),
        name="V2.7 q-ref failure stage receipt",
        expected_sha256=V27_QREF_FAILURE_STAGE_RECEIPT_FILE_SHA256,
    )
    _verify_v2_stage_receipt_self_hash(
        value,
        name="V2.7 q-ref failure stage receipt",
    )
    failure = _mapping(value.get("failure"), name="V2.7 q-ref failure")
    if (
        value.get("contract_id") != V27_CONTRACT_ID
        or value.get("contract_sha256")
        != V27_CONTRACT_CANONICAL_SHA256
        or value.get("status") != "integrity-stopped"
        or value.get("registered_run_count") != 1
        or value.get("complete_cell_count") != 0
        or value.get("go") is not False
        or value.get("execution_progression_go") is not False
        or value.get("artifacts") != {}
        or failure.get("error_type") != "PilotOrchestrationError"
        or failure.get("message") != (
            "finevo-pilot-v2.7--q-ref-resolution--qref_scripted--"
            "qref-scripted--none--provider-preflight-default--s2010922376 "
            "imported source run identity is malformed"
        )
        or value.get("integrity", {}).get("content_sha256")
        != V27_QREF_FAILURE_STAGE_RECEIPT_CONTENT_SHA256
    ):
        raise PilotV28Stage0ImportError(
            "V2.7 q-ref failure classification drifted"
        )
    return {
        "path": (
            V27_RAW_ROOT / "q-ref-resolution/stage_receipt.json"
        ).as_posix(),
        "file_sha256": _sha256(raw),
        "content_sha256": V27_QREF_FAILURE_STAGE_RECEIPT_CONTENT_SHA256,
        "status": "integrity-stopped",
        "root_cause_code":
        "qref-contract-cell-id-conflated-with-runner-execution-id",
        "root_cause_message": "imported source run identity is malformed",
    }


def _validate_target_contract(
    contract: PilotContract,
    *,
    require_frozen: bool,
) -> None:
    amendment = contract.qref_identity_retry_amendment
    if (
        contract.contract_id != V28_CONTRACT_ID
        or contract.implementation.get("required_git_tag") != V28_SCIENCE_TAG
        or (
            require_frozen
            and contract.status != "frozen"
        )
        or (
            not require_frozen
            and contract.status not in {"draft", "frozen"}
        )
        or not isinstance(amendment, Mapping)
        or amendment.get("schema_version")
        != "finevo-pilot-qref-identity-retry-amendment-v1"
    ):
        raise PilotV28Stage0ImportError(
            "V2.8 Stage-0 import requires its exact q-ref retry contract"
        )
    failure = _mapping(
        amendment.get("failure_classification"),
        name="V2.8 failure classification",
    )
    lineage = _mapping(
        amendment.get("source_lineage"),
        name="V2.8 source lineage",
    )
    stage0 = _mapping(
        amendment.get("stage0_import"),
        name="V2.8 Stage-0 import amendment",
    )
    qref = _mapping(
        amendment.get("q_ref_regeneration"),
        name="V2.8 q-ref regeneration amendment",
    )
    if (
        failure.get("parent_contract_id") != V27_CONTRACT_ID
        or failure.get("parent_contract_sha256")
        != V27_CONTRACT_CANONICAL_SHA256
        or failure.get("parent_release_tag") != V27_SCIENCE_TAG
        or failure.get("parent_release_commit") != V27_SCIENCE_COMMIT
        or failure.get("parent_evidence_commit")
        != V27_EVIDENCE_PUBLICATION_COMMIT
        or failure.get("parent_evidence_merge_commit")
        != V27_EVIDENCE_MERGE_COMMIT
        or lineage.get("amendment_parent_raw_namespace")
        != V27_RAW_ROOT.as_posix()
        or lineage.get("nested_stage0_source_contract_id")
        != V26_CONTRACT_ID
        or lineage.get("nested_stage0_source_contract_sha256")
        != V26_CONTRACT_CANONICAL_SHA256
        or lineage.get("nested_stage0_snapshot_namespace")
        != (V27_RAW_ROOT / V26_NESTED_RELATIVE).as_posix()
        or lineage.get("child_raw_namespace") != V28_RAW_ROOT.as_posix()
        or stage0.get("imported_complete_cells") != 14
        or stage0.get("provider_construction_during_import") is not False
        or stage0.get("provider_redispatch_for_imported_cells") != "forbidden"
        or qref.get("source_result_reuse") != "forbidden"
        or qref.get("fresh_zero_hosted_provider_regeneration") is not True
        or qref.get("scripted_diagnostic_calls") != 48
        or qref.get("hosted_provider_calls") != 0
        or qref.get("hosted_cost_usd") != 0.0
        or qref.get("config_run_id_policy")
        != "must-equal-current-contract-cell-run-id"
    ):
        raise PilotV28Stage0ImportError(
            "V2.8 q-ref/Stage-0 source-lineage policy drifted"
        )


def _normalized_spec(spec: PilotRunSpec | Mapping[str, Any]) -> dict[str, Any]:
    value = spec.to_dict() if isinstance(spec, PilotRunSpec) else _json_copy(spec)
    value.pop("contract_id", None)
    value.pop("run_id", None)
    return value


def _match_specs(
    source: Sequence[PilotRunSpec],
    target: Sequence[PilotRunSpec],
    *,
    name: str,
) -> list[tuple[PilotRunSpec, PilotRunSpec]]:
    source_map = {
        json.dumps(_normalized_spec(spec), sort_keys=True, allow_nan=False): spec
        for spec in source
    }
    target_map = {
        json.dumps(_normalized_spec(spec), sort_keys=True, allow_nan=False): spec
        for spec in target
    }
    if (
        len(source_map) != len(source)
        or len(target_map) != len(target)
        or set(source_map) != set(target_map)
    ):
        raise PilotV28Stage0ImportError(
            f"{name} source/target matrix is not an exact normalized match"
        )
    return [(source_map[key], target_map[key]) for key in sorted(source_map)]


def _load_verified_v27_contract(parent_root: Path) -> PilotContract:
    _, _, contract_value = _strict_file(
        parent_root,
        V27_CONTRACT_PATH,
        name="frozen V2.7 contract",
        expected_sha256=V27_CONTRACT_FILE_SHA256,
    )
    contract = PilotContract.from_dict(contract_value)
    if (
        contract.contract_id != V27_CONTRACT_ID
        or contract.canonical_hash != V27_CONTRACT_CANONICAL_SHA256
        or contract.status != "frozen"
        or contract.implementation.get("required_git_tag") != V27_SCIENCE_TAG
    ):
        raise PilotV28Stage0ImportError("frozen V2.7 contract identity drifted")
    _, _, source_manifest = _strict_file(
        parent_root,
        V27_SOURCE_MANIFEST_PATH,
        name="frozen V2.7 source manifest",
        expected_sha256=V27_SOURCE_MANIFEST_FILE_SHA256,
    )
    try:
        _verify_self_hash(
            source_manifest,
            schema_version="finevo-pilot-v2.7-source-manifest-v1",
            name="frozen V2.7 source manifest",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if (
        source_manifest.get("integrity", {}).get("content_sha256")
        != V27_SOURCE_MANIFEST_CONTENT_SHA256
    ):
        raise PilotV28Stage0ImportError(
            "frozen V2.7 source-manifest content drifted"
        )
    return contract


def _verify_v27_release_identity(parent_root: Path) -> None:
    tag_ref = f"refs/tags/{V27_SCIENCE_TAG}"
    if (
        _git(parent_root, "cat-file", "-t", tag_ref) != "tag"
        or _git(parent_root, "rev-parse", tag_ref) != V27_SCIENCE_TAG_OBJECT
        or _git(parent_root, "rev-parse", f"{tag_ref}^{{commit}}")
        != V27_SCIENCE_COMMIT
        or _git(parent_root, "rev-parse", "HEAD") != V27_SCIENCE_COMMIT
        or _git(
            parent_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        )
    ):
        raise PilotV28Stage0ImportError(
            "V2.7 annotated release or tracked checkout drifted"
        )


def _verify_v27_parent_receipt(
    parent_root: Path,
    contract: PilotContract,
) -> dict[str, Any]:
    raw_root = parent_root.joinpath(*V27_RAW_ROOT.parts)
    receipt_path, raw, value = _strict_file(
        raw_root,
        PurePosixPath("parent-import/parent_import_receipt.json"),
        name="V2.7 parent-import receipt",
        expected_sha256=V27_PARENT_IMPORT_RECEIPT_FILE_SHA256,
    )
    try:
        verified = verify_v27_parent_import_receipt(
            receipt_path,
            repo_root=parent_root,
            contract=contract,
            expected_git_commit=V27_SCIENCE_COMMIT,
        )
    except PilotV27Stage0ImportError as exc:
        raise _translate(exc) from exc
    if (
        verified != value
        or value.get("integrity", {}).get("content_sha256")
        != V27_PARENT_IMPORT_RECEIPT_CONTENT_SHA256
    ):
        raise PilotV28Stage0ImportError(
            "V2.7 parent-import receipt did not reproduce"
        )
    return {
        "path": (V27_RAW_ROOT / "parent-import/parent_import_receipt.json").as_posix(),
        "file_sha256": _sha256(raw),
        "content_sha256": V27_PARENT_IMPORT_RECEIPT_CONTENT_SHA256,
    }


def _verify_v27_p95_sources(parent_root: Path) -> dict[str, Any]:
    from .observed_p95_authority import (
        ObservedP95AuthorityError,
        verify_v27_resealed_observed_p95_authority,
        verify_v27_resealed_observed_p95_projection,
    )

    verified: dict[str, Any] = {}
    for profile_id in V28_ALLOWED_P95_PROFILES:
        source = _V27_P95_SOURCES[profile_id]
        values: dict[str, dict[str, Any]] = {}
        for kind in ("authority", "projection"):
            binding = source[kind]
            try:
                relative = _normalized_relative(
                    binding["path"],
                    required_top="experiment_results",
                    name=f"V2.7 {profile_id} {kind} path",
                )
            except PilotV24ParentImportError as exc:
                raise _translate(exc) from exc
            _, _, value = _strict_file(
                parent_root,
                relative,
                name=f"V2.7 {profile_id} {kind}",
                expected_sha256=binding["file_sha256"],
            )
            integrity = value.get("integrity")
            if (
                value.get("schema_version") != binding["schema_version"]
                or not isinstance(integrity, Mapping)
                or integrity.get("canonicalization") != CANONICALIZATION
                or integrity.get("content_sha256")
                != binding["content_sha256"]
            ):
                raise PilotV28Stage0ImportError(
                    f"V2.7 {profile_id} {kind} identity drifted"
                )
            values[kind] = value
        try:
            reservations = verify_v27_resealed_observed_p95_authority(
                source["authority"]["path"],
                repo_root=parent_root,
                expected_git_commit=V27_SCIENCE_COMMIT,
            )
            projection = verify_v27_resealed_observed_p95_projection(
                source["projection"]["path"],
                receipt_or_path=source["authority"]["path"],
                repo_root=parent_root,
                expected_git_commit=V27_SCIENCE_COMMIT,
            )
        except ObservedP95AuthorityError as exc:
            raise _translate(exc) from exc
        runtime_model = source["runtime_model"]
        served_model = source["served_model"]
        if (
            set(reservations) != {runtime_model}
            or set(reservations[runtime_model]) != {"action", "semantic"}
            or any(
                reservations[runtime_model][kind]["reservation"]
                != projection["projection"][f"{served_model}::{kind}"]
                for kind in ("action", "semantic")
            )
        ):
            raise PilotV28Stage0ImportError(
                f"V2.7 {profile_id} p95 receipt/projection differ"
            )
        verified[profile_id] = _json_copy(source)
    return verified


def _verify_qref_audit_reference(
    parent_root: Path,
    *,
    source_contract: PilotContract,
) -> dict[str, Any]:
    specs = source_contract.expand(stage="q-ref-resolution")
    if len(specs) != 1:
        raise PilotV28Stage0ImportError(
            "nested V2.6 q-ref reference lacks one source cell"
        )
    spec = specs[0]
    prefix = V27_RAW_ROOT / V26_NESTED_RELATIVE
    run_relative = prefix / "q-ref-resolution/runs" / spec.run_id
    run_dir = parent_root.joinpath(*run_relative.parts)
    verification = verify_manifest(run_dir)
    if not verification.valid:
        raise PilotV28Stage0ImportError(
            "nested V2.6 q-ref audit manifest failed verification"
        )
    manifest_binding = _artifact_binding(
        parent_root,
        run_relative / "manifest.json",
    )
    if manifest_binding["file_sha256"] != V26_QREF_MANIFEST_FILE_SHA256:
        raise PilotV28Stage0ImportError(
            "nested V2.6 q-ref source manifest drifted"
        )
    _, config_raw, config = _strict_file(
        parent_root,
        run_relative / "config.json",
        name="nested V2.6 q-ref config",
    )
    _, provenance_raw, provenance = _strict_file(
        parent_root,
        run_relative / "provenance.json",
        name="nested V2.6 q-ref provenance",
    )
    stream_rows: dict[str, list[dict[str, Any]]] = {}
    stream_bindings: dict[str, dict[str, Any]] = {}
    for stream_name, expected_count in (
        ("actions", 48),
        ("api_usage", 48),
        ("utility_ledger", 48),
        ("shock_events", 12),
        ("summary", 1),
    ):
        relative = run_relative / f"streams/{stream_name}.jsonl"
        raw, rows = _strict_jsonl_rows(
            parent_root,
            relative,
            name=f"nested V2.6 q-ref {stream_name}",
        )
        if len(rows) != expected_count:
            raise PilotV28Stage0ImportError(
                f"nested V2.6 q-ref {stream_name} row count drifted"
            )
        stream_rows[stream_name] = rows
        stream_bindings[stream_name] = {
            "path": relative.as_posix(),
            "file_sha256": _sha256(raw),
            "byte_size": len(raw),
            "row_count": len(rows),
        }
    semantic_paths = tuple(
        run_dir / f"streams/{name}.jsonl"
        for name in (
            "semantic_proposals",
            "semantic_rules",
            "semantic_rule_events",
        )
    )
    if any(path.exists() for path in semantic_paths):
        raise PilotV28Stage0ImportError(
            "nested V2.6 q-ref semantic streams are not empty"
        )
    expected_short_run_id = "q-ref-resolution-s2010922376"
    expected_shocks = [
        {
            "schema_version": "finevo-shock-event-v1",
            "decision_t": decision_t,
            "phase": "baseline",
            "interest_rate": 0.03,
            "applied_before_prompt": True,
            "applied_before_step": True,
        }
        for decision_t in range(12)
    ]
    if (
        config.get("run_id") != expected_short_run_id
        or config.get("seed") != 2010922376
        or config.get("num_agents") != 4
        or config.get("episode_length") != 12
        or config.get("context_mode") != "no-context"
        or config.get("enable_episodic_retrieval") is not False
        or config.get("enable_semantic") is not False
        or config.get("retrieval_k") != 0
        or config.get("rule_budget") != 0
        or config.get("shock_schedule") != expected_shocks
    ):
        raise PilotV28Stage0ImportError(
            "nested V2.6 q-ref config identity/design drifted"
        )
    provenance_details = _mapping(
        provenance.get("details"),
        name="nested V2.6 q-ref provenance details",
    )
    if (
        provenance_details.get("contract_id") != V26_CONTRACT_ID
        or provenance_details.get("contract_sha256")
        != V26_CONTRACT_CANONICAL_SHA256
        or provenance_details.get("run_spec") != spec.to_dict()
        or provenance_details.get("purpose")
        != "deterministic q_ref scale resolution"
        or provenance_details.get("diagnostic_only") is not True
        or provenance_details.get("scientific_evidence") is not False
        or provenance.get("git")
        != {
            "commit": "0f59a15bc2cc3cce68f64de1dc1be78f7d74e214",
            "dirty": False,
        }
    ):
        raise PilotV28Stage0ImportError(
            "nested V2.6 q-ref provenance/spec identity drifted"
        )
    actions = stream_rows["actions"]
    api_usage = stream_rows["api_usage"]
    ledger = stream_rows["utility_ledger"]
    shocks = stream_rows["shock_events"]
    summary = stream_rows["summary"][0]
    expected_keys = {
        (decision_t, agent_id)
        for decision_t in range(12)
        for agent_id in range(4)
    }
    action_keys = {
        (row.get("decision_t"), row.get("agent_id"))
        for row in actions
    }
    api_usage_keys = {
        (row.get("decision_t"), row.get("agent_id"))
        for row in api_usage
    }
    ledger_keys = {
        (row.get("period"), int(row.get("agent_id", -1)))
        for row in ledger
    }
    summary_api = _mapping(
        summary.get("api"),
        name="nested V2.6 q-ref summary API accounting",
    )
    summary_accounted = _mapping(
        summary_api.get("accounted_usage"),
        name="nested V2.6 q-ref accounted usage",
    )
    summary_effective = _mapping(
        summary_api.get("effective_usage"),
        name="nested V2.6 q-ref effective usage",
    )
    summary_limits = _mapping(
        summary_api.get("limits"),
        name="nested V2.6 q-ref API limits",
    )
    summary_completions = summary_api.get("completions")
    work_cycle = (0.25, 0.50, 0.75, 0.50)
    consumption_cycle = (0.30, 0.35, 0.30, 0.25)
    if (
        action_keys != expected_keys
        or api_usage_keys != expected_keys
        or ledger_keys != expected_keys
        or shocks != expected_shocks
        or any(
            row.get("schema_version") != "verified-simulation-runner-v3"
            or row.get("provider") != "diagnostic"
            or row.get("model") != "scripted-v1"
            or row.get("response_model") != "scripted-v1"
            or row.get("call_kind") != "action"
            or row.get("attempts") != 1
            or row.get("action_parse_mode") != "exact_json"
            or row.get("output_disposition") != "accepted"
            or row.get("error_type") is not None
            or row.get("provider_error_details") is not None
            or row.get("provider_request_id") is not None
            or row.get("response_provider") is not None
            or row.get("response_route") is not None
            or row.get("request_profile_id") is not None
            or row.get("request_provider_pin") != []
            or row.get("request_artifact_identity") != {}
            or row.get("request_seed") != 2010922376
            or _mapping(
                row.get("usage"),
                name="nested V2.6 q-ref API usage row",
            ).get("cost_usd")
            != 0.0
            for row in api_usage
        )
        or any(
            not math.isclose(
                float(row["decision"]["proposed_work_fraction"]),
                work_cycle[int(row["decision_t"]) % len(work_cycle)],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(row["decision"]["proposed_consumption_fraction"]),
                consumption_cycle[
                    int(row["decision_t"]) % len(consumption_cycle)
                ],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for row in actions
        )
        or summary.get("run_id") != expected_short_run_id
        or summary.get("result_complete") is not True
        or summary.get("diagnostic_only") is not True
        or summary.get("scientific_evidence") is not False
        or summary.get("provider_model") != "diagnostic/scripted-v1"
        or summary.get("num_agents") != 4
        or summary.get("episode_length") != 12
        or summary_api.get("completed_calls") != 48
        or summary_api.get("rolled_back_calls") != 0
        or summary_api.get("active_calls") != 0
        or summary_api.get("active_reservations") != []
        or summary_accounted.get("cost_usd") != 0.0
        or summary_effective.get("cost_usd") != 0.0
        or summary_limits.get("max_calls") != 48
        or not isinstance(summary_completions, list)
        or len(summary_completions) != 48
        or any(
            completion.get("model") != "diagnostic/scripted-v1"
            or _mapping(
                completion.get("usage"),
                name="nested V2.6 q-ref completion usage",
            ).get("cost_usd")
            != 0.0
            for completion in summary_completions
        )
    ):
        raise PilotV28Stage0ImportError(
            "nested V2.6 q-ref source streams/design drifted"
        )
    qref_relative = prefix / "q-ref-resolution/q_ref_resolution.json"
    _, _, qref = _strict_file(
        parent_root,
        qref_relative,
        name="nested V2.6 q-ref audit reference",
        expected_sha256=V26_QREF_FILE_SHA256,
    )
    try:
        _verify_self_hash(
            qref,
            schema_version="finevo-q-ref-resolution-v1",
            name="nested V2.6 q-ref audit reference",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if (
        qref.get("integrity", {}).get("content_sha256")
        != V26_QREF_CONTENT_SHA256
        or qref.get("q_ref") != V26_QREF_VALUE
        or qref.get("status") != "pass"
        or qref.get("scientific_evidence") is not False
    ):
        raise PilotV28Stage0ImportError(
            "nested V2.6 q-ref audit value or boundary drifted"
        )
    source = _mapping(qref.get("source"), name="nested V2.6 q-ref source core")
    bindings = _mapping(
        qref.get("bindings"),
        name="nested V2.6 q-ref source bindings",
    )
    checks = _mapping(qref.get("checks"), name="nested V2.6 q-ref checks")
    try:
        _, environment_raw = _guarded_file(
            parent_root,
            PurePosixPath("config.yaml"),
            name="V2.7 checkout environment config",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    quantities = [
        float(row["realized_consumption_quantity"])
        for row in ledger
    ]
    if (
        source.get("config") != config
        or source.get("run_summary") != summary
        or source.get("utility_ledger") != ledger
        or bindings.get("source_config_hash") != canonical_sha256(config)
        or bindings.get("run_summary_hash") != canonical_sha256(summary)
        or bindings.get("ledger_hash") != canonical_sha256(ledger)
        or bindings.get("environment_source_hash") != _sha256(environment_raw)
        or bindings.get("source_manifest_sha256")
        != V26_QREF_MANIFEST_FILE_SHA256
        or bindings.get("contract_hash") != V26_CONTRACT_CANONICAL_SHA256
        or bindings.get("contract_sha256") != V26_CONTRACT_CANONICAL_SHA256
        or qref.get("row_count") != 48
        or qref.get("ledger_field") != "realized_consumption_quantity"
        or qref.get("aggregation") != "median"
        or median(quantities) != V26_QREF_VALUE
        or not checks
        or any(passed is not True for passed in checks.values())
    ):
        raise PilotV28Stage0ImportError(
            "nested V2.6 q-ref resolution/source-core hashes drifted"
        )
    return {
        "source_contract_id": V26_CONTRACT_ID,
        "source_contract_sha256": V26_CONTRACT_CANONICAL_SHA256,
        "source_spec": spec.to_dict(),
        "source_run_id": spec.run_id,
        "source_contract_cell_run_id": spec.run_id,
        "source_runner_short_run_id": expected_short_run_id,
        "source_run_root": run_relative.as_posix(),
        "source_manifest": manifest_binding,
        "source_config": _artifact_binding(
            parent_root,
            run_relative / "config.json",
        ),
        "q_ref_resolution": _artifact_binding(parent_root, qref_relative),
        "q_ref": V26_QREF_VALUE,
        "reference_use":
        "audit-equivalence-only-for-fresh-zero-hosted-provider-regeneration",
        "imported": False,
        "source_result_reuse": "forbidden",
        "scripted_diagnostic_calls": 48,
        "hosted_provider_calls": 0,
        "hosted_cost_usd": 0.0,
        "source_core": {
            "config": {
                "path": (run_relative / "config.json").as_posix(),
                "file_sha256": _sha256(config_raw),
                "content_sha256": canonical_sha256(config),
            },
            "provenance": {
                "path": (run_relative / "provenance.json").as_posix(),
                "file_sha256": _sha256(provenance_raw),
                "content_sha256": canonical_sha256(provenance),
            },
            "streams": stream_bindings,
            "semantic_streams": {
                "semantic_proposals": 0,
                "semantic_rules": 0,
                "semantic_rule_events": 0,
            },
            "environment_config": {
                "path": "config.yaml",
                "file_sha256": _sha256(environment_raw),
            },
            "resolution_bindings": {
                key: bindings[key]
                for key in (
                    "source_config_hash",
                    "run_summary_hash",
                    "ledger_hash",
                    "environment_source_hash",
                    "source_manifest_sha256",
                )
            },
            "identity_grid": {
                "agents": 4,
                "periods": 12,
                "action_rows": 48,
                "api_usage_rows": 48,
                "utility_ledger_rows": 48,
                "shock_rows": 12,
            },
        },
    }


def _verify_stage0_source_config(
    parent_root: Path,
    *,
    run_relative: PurePosixPath,
    source_run_id: str,
) -> dict[str, Any]:
    _, _, config = _strict_file(
        parent_root,
        run_relative / "config.json",
        name=f"nested V2.6 Stage-0 config {source_run_id}",
    )
    if (
        run_relative.name != source_run_id
        or config.get("run_id") != source_run_id
    ):
        raise PilotV28Stage0ImportError(
            f"Stage-0 config.run_id differs from source cell ID: {source_run_id}"
        )
    return config


def _verify_imported_cells(
    parent_root: Path,
    *,
    v27_contract: PilotContract,
    v26_contract: PilotContract,
    target_contract: PilotContract,
    v27_run_ledger: Mapping[str, Any],
) -> list[dict[str, Any]]:
    parent_pairs = _match_specs(
        v27_contract.expand(stage="parent-import"),
        target_contract.expand(stage="parent-import"),
        name="V2.7 parent prerequisite",
    )
    stage0_pairs = _match_specs(
        v26_contract.expand(stage="stage0-calibration"),
        target_contract.expand(stage="stage0-calibration"),
        name="nested V2.6 Stage-0",
    )
    if len(parent_pairs) != 1 or len(stage0_pairs) != 14:
        raise PilotV28Stage0ImportError(
            "V2.8 imported prerequisite/Stage-0 cell count drifted"
        )
    runs = _mapping(v27_run_ledger.get("runs"), name="V2.7 run ledger rows")
    source_parent, target_parent = parent_pairs[0]
    parent_ledger_row = _mapping(
        runs.get(source_parent.run_id),
        name="V2.7 parent prerequisite ledger row",
    )
    if (
        parent_ledger_row.get("status") != "complete"
        or parent_ledger_row.get("spec") != source_parent.to_dict()
    ):
        raise PilotV28Stage0ImportError(
            "V2.7 parent prerequisite is not the unique complete source cell"
        )
    rows: list[dict[str, Any]] = [
        {
            "stage_id": "parent-import",
            "source_authority_contract_id": V27_CONTRACT_ID,
            "source_run_id": source_parent.run_id,
            "target_run_id": target_parent.run_id,
            "source_spec": source_parent.to_dict(),
            "target_spec": target_parent.to_dict(),
            "source_artifacts": {
                "receipt": _artifact_binding(
                    parent_root,
                    V27_RAW_ROOT
                    / "parent-import/parent_import_receipt.json",
                ),
                "stage_receipt": _artifact_binding(
                    parent_root,
                    V27_RAW_ROOT / "parent-import/stage_receipt.json",
                ),
            },
        }
    ]

    nested_prefix = V27_RAW_ROOT / V26_NESTED_RELATIVE
    for source_spec, target_spec in stage0_pairs:
        run_relative = (
            nested_prefix
            / "stage0-calibration/runs"
            / source_spec.run_id
        )
        run_dir = parent_root.joinpath(*run_relative.parts)
        _verify_stage0_source_config(
            parent_root,
            run_relative=run_relative,
            source_run_id=source_spec.run_id,
        )
        verification = verify_manifest(run_dir)
        if not verification.valid:
            raise PilotV28Stage0ImportError(
                f"nested V2.6 Stage-0 manifest is invalid: "
                f"{source_spec.run_id}"
            )
        journal_relative = (
            nested_prefix
            / "stage0-calibration/provider_call_journals"
            / f"{source_spec.run_id}--actor.json"
        )
        journal_path = parent_root.joinpath(*journal_relative.parts)
        try:
            journal = verify_provider_call_journal(
                journal_path,
                expected_run_id=source_spec.run_id,
                expected_contract_hash=V26_CONTRACT_CANONICAL_SHA256,
                require_terminal_dispositions=True,
            )
        except Exception as exc:
            raise PilotV28Stage0ImportError(
                f"nested V2.6 Stage-0 journal failed verification: "
                f"{source_spec.run_id}: {exc}"
            ) from exc
        events = journal.get("events")
        if (
            not isinstance(events, list)
            or len(events) != 96
            or Counter(event.get("event_type") for event in events)
            != Counter({"completion_received": 48, "parse_disposition": 48})
        ):
            raise PilotV28Stage0ImportError(
                f"nested V2.6 Stage-0 journal is incomplete: "
                f"{source_spec.run_id}"
            )
        rows.append(
            {
                "stage_id": "stage0-calibration",
                "source_authority_contract_id": V27_CONTRACT_ID,
                "physical_source_contract_id": V26_CONTRACT_ID,
                "source_run_id": source_spec.run_id,
                "target_run_id": target_spec.run_id,
                "source_spec": source_spec.to_dict(),
                "target_spec": target_spec.to_dict(),
                "source_artifacts": {
                    "run_root": run_relative.as_posix(),
                    "config": _artifact_binding(
                        parent_root,
                        run_relative / "config.json",
                    ),
                    "manifest": _artifact_binding(
                        parent_root,
                        run_relative / "manifest.json",
                    ),
                    "actor_journal": _artifact_binding(
                        parent_root,
                        journal_relative,
                    ),
                },
            }
        )
    if (
        len(rows) != 15
        or Counter(row["stage_id"] for row in rows)
        != Counter({"parent-import": 1, "stage0-calibration": 14})
        or {
            row["source_spec"]["utility_profile_id"]
            for row in rows
            if row["stage_id"] == "stage0-calibration"
        }
        != set(_STAGE0_PROFILES)
        or {
            row["source_spec"]["environment_seed"]
            for row in rows
            if row["stage_id"] == "stage0-calibration"
        }
        != set(_STAGE0_SEEDS)
    ):
        raise PilotV28Stage0ImportError(
            "V2.8 imported-cell profile/seed inventory drifted"
        )
    return sorted(rows, key=lambda row: row["target_run_id"])


def _verify_nested_stage0_receipt(parent_root: Path) -> dict[str, Any]:
    relative = (
        V27_RAW_ROOT / V26_NESTED_RELATIVE / "stage0-calibration/stage_receipt.json"
    )
    _, raw, value = _strict_file(
        parent_root,
        relative,
        name="nested V2.6 Stage-0 receipt",
        expected_sha256=V26_STAGE0_RECEIPT_FILE_SHA256,
    )
    _verify_v2_stage_receipt_self_hash(
        value,
        name="nested V2.6 Stage-0 receipt",
    )
    if (
        value.get("integrity", {}).get("content_sha256")
        != V26_STAGE0_RECEIPT_CONTENT_SHA256
        or value.get("status") != "complete-with-no-go"
        or value.get("registered_run_count") != 14
        or value.get("complete_cell_count") != 14
        or value.get("go") is not False
        or value.get("execution_progression_go") is not False
    ):
        raise PilotV28Stage0ImportError(
            "nested V2.6 Stage-0 receipt boundary drifted"
        )
    return {
        "path": relative.as_posix(),
        "file_sha256": _sha256(raw),
        "content_sha256": V26_STAGE0_RECEIPT_CONTENT_SHA256,
        "status": "complete-with-no-go",
        "registered_run_count": 14,
        "complete_cell_count": 14,
    }


def _audit_v27_source(
    *,
    parent_repo_root: str | Path,
    child_repo_root: str | Path,
    target_contract: PilotContract,
) -> dict[str, Any]:
    try:
        parent_root = _real_root(
            parent_repo_root,
            name="V2.7 source repository",
        )
        child_root = _real_root(
            child_repo_root,
            name="V2.8 child repository",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    _validate_target_contract(target_contract, require_frozen=False)
    _verify_v27_release_identity(parent_root)
    v27_contract = _load_verified_v27_contract(parent_root)
    v26_contract = load_pilot_contract(
        parent_root.joinpath(*V26_CONTRACT_PATH.parts)
    )
    if (
        v26_contract.contract_id != V26_CONTRACT_ID
        or v26_contract.canonical_hash != V26_CONTRACT_CANONICAL_SHA256
        or v26_contract.status != "frozen"
    ):
        raise PilotV28Stage0ImportError(
            "nested Stage-0 V2.6 contract identity drifted"
        )
    raw_root = parent_root.joinpath(*V27_RAW_ROOT.parts)
    inventory = _verify_exact_v27_inventory(raw_root)
    nested_root = raw_root.joinpath(*V26_NESTED_RELATIVE.parts)
    _verify_exact_nested_v26_inventory(nested_root)
    run_ledger, budget_ledger = _verify_v27_ledgers(raw_root)
    expanded = {spec.run_id: spec.to_dict() for spec in v27_contract.expand()}
    runs = _mapping(run_ledger.get("runs"), name="V2.7 run ledger rows")
    if (
        set(runs) != set(expanded)
        or any(row.get("spec") != expanded[run_id] for run_id, row in runs.items())
    ):
        raise PilotV28Stage0ImportError(
            "V2.7 run ledger differs from its frozen denominator"
        )
    parent_receipt = _verify_v27_parent_receipt(parent_root, v27_contract)
    qref_failure = _verify_v27_qref_failure(raw_root)
    nested_stage0_receipt = _verify_nested_stage0_receipt(parent_root)
    p95_sources = _verify_v27_p95_sources(parent_root)
    evidence = _verify_v27_published_evidence(child_root)
    qref_reference = _verify_qref_audit_reference(
        parent_root,
        source_contract=v26_contract,
    )
    imported = _verify_imported_cells(
        parent_root,
        v27_contract=v27_contract,
        v26_contract=v26_contract,
        target_contract=target_contract,
        v27_run_ledger=run_ledger,
    )
    return {
        "parent_root": parent_root,
        "child_root": child_root,
        "v27_contract": v27_contract,
        "v26_contract": v26_contract,
        "inventory": inventory,
        "parent_receipt": parent_receipt,
        "qref_failure": qref_failure,
        "nested_stage0_receipt": nested_stage0_receipt,
        "p95_sources": p95_sources,
        "evidence": evidence,
        "qref_reference": qref_reference,
        "imported_cells": imported,
        "budget_ledger": budget_ledger,
    }


def build_v28_source_manifest(
    *,
    parent_repo_root: str | Path,
    child_repo_root: str | Path,
    target_contract: PilotContract | None = None,
) -> dict[str, Any]:
    """Verify frozen V2.7 and build the deterministic V2.8 source manifest."""

    try:
        child_root = _real_root(
            child_repo_root,
            name="V2.8 child repository",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    contract = target_contract or load_pilot_contract(
        child_root.joinpath(*V28_EXPANDED_CONTRACT_PATH.parts)
    )
    audit = _audit_v27_source(
        parent_repo_root=parent_repo_root,
        child_repo_root=child_root,
        target_contract=contract,
    )
    return _seal(
        {
            "schema_version": V28_SOURCE_MANIFEST_SCHEMA_VERSION,
            "v2_7_terminal_parent": {
                "contract": {
                    "contract_id": V27_CONTRACT_ID,
                    "path": V27_CONTRACT_PATH.as_posix(),
                    "schema_version": "finevo-pilot-contract-v2",
                    "status": "frozen",
                    "file_sha256": V27_CONTRACT_FILE_SHA256,
                    "canonical_sha256": V27_CONTRACT_CANONICAL_SHA256,
                },
                "source_manifest": {
                    "path": V27_SOURCE_MANIFEST_PATH.as_posix(),
                    "file_sha256": V27_SOURCE_MANIFEST_FILE_SHA256,
                    "content_sha256": V27_SOURCE_MANIFEST_CONTENT_SHA256,
                },
                "release": {
                    "science_tag": V27_SCIENCE_TAG,
                    "science_tag_object": V27_SCIENCE_TAG_OBJECT,
                    "science_commit": V27_SCIENCE_COMMIT,
                    "tag_kind": "annotated",
                    "raw_root": V27_RAW_ROOT.as_posix(),
                },
                "raw_snapshot": {
                    "root": V27_RAW_ROOT.as_posix(),
                    "inventory_schema_version":
                    "finevo-raw-tree-inventory-v1",
                    "inventory_canonicalization":
                    "json-sort-keys-compact-utf8-v1",
                    "file_count": V27_RAW_FILE_COUNT,
                    "storage_bytes": V27_RAW_STORAGE_BYTES,
                    "inventory_sha256": V27_RAW_INVENTORY_SHA256,
                },
                "ledgers": {
                    "run": {
                        "path": f"{V27_RAW_ROOT.as_posix()}/run_ledger.json",
                        "file_sha256": V27_RUN_LEDGER_FILE_SHA256,
                        "internal_sha256": V27_RUN_LEDGER_INTERNAL_SHA256,
                        "event_count": V27_RUN_LEDGER_EVENT_COUNT,
                        "event_chain_head": V27_RUN_LEDGER_EVENT_HEAD,
                    },
                    "budget": {
                        "path": f"{V27_RAW_ROOT.as_posix()}/budget_ledger.json",
                        "file_sha256": V27_BUDGET_LEDGER_FILE_SHA256,
                        "internal_sha256": V27_BUDGET_LEDGER_INTERNAL_SHA256,
                        "event_count": V27_BUDGET_LEDGER_EVENT_COUNT,
                        "event_chain_head": V27_BUDGET_LEDGER_EVENT_HEAD,
                    },
                },
                "terminal_denominator": {
                    "registered_cells": 211,
                    "scientific_cells": 209,
                    "terminal_cells": 211,
                    "all_rows_present": True,
                    "all_rows_terminal": True,
                    "status_counts": {
                        "complete": 1,
                        "integrity-stopped": 210,
                    },
                    "completed_cell_breakdown": {
                        "parent-import": 1,
                        "q-ref-resolution": 0,
                        "stage0-calibration": 0,
                    },
                    "terminal_status": "complete-with-no-go",
                    "scientific_complete": False,
                    "scientific_matrix_complete": False,
                    "scientific_claim_gates_supported": False,
                },
                "parent_import_receipt": audit["parent_receipt"],
                "q_ref_failure_stage_receipt": audit["qref_failure"],
            },
            "published_v2_7_evidence": audit["evidence"],
            "nested_v2_6_stage0_source": {
                "contract": {
                    "contract_id": V26_CONTRACT_ID,
                    "path": V26_CONTRACT_PATH.as_posix(),
                    "canonical_sha256": V26_CONTRACT_CANONICAL_SHA256,
                },
                "physical_snapshot_root": (
                    V27_RAW_ROOT / V26_NESTED_RELATIVE
                ).as_posix(),
                "inventory": {
                    "file_count": V26_RAW_FILE_COUNT,
                    "storage_bytes": V26_RAW_STORAGE_BYTES,
                    "inventory_sha256": V26_RAW_INVENTORY_SHA256,
                },
                "stage0_receipt": audit["nested_stage0_receipt"],
                "source_via_v2_7_exact_snapshot": True,
            },
            "q_ref_audit_equivalence_reference": audit["qref_reference"],
            "v2_7_p95_sources_for_child_reseal": audit["p95_sources"],
            "imported_complete_cells": audit["imported_cells"],
            "cumulative_budget_debit": V28_CUMULATIVE_DEBIT.to_dict(),
            "import_policy": {
                "source_raw_namespace": V27_RAW_ROOT.as_posix(),
                "child_raw_namespace": V28_RAW_ROOT.as_posix(),
                "child_snapshot_namespace": (
                    V28_RAW_ROOT / V28_SNAPSHOT_RELATIVE
                ).as_posix(),
                "exact_full_v2_7_raw_snapshot_copy": True,
                "imported_cell_count": 15,
                "imported_cell_breakdown": {
                    "parent-import": 1,
                    "stage0-calibration": 14,
                },
                "q_ref_imported": False,
                "q_ref_fresh_zero_hosted_provider_regeneration_required": True,
                "q_ref_scripted_diagnostic_calls": 48,
                "q_ref_hosted_provider_calls": 0,
                "q_ref_hosted_cost_usd": 0.0,
                "source_manifests_rewritten": False,
                "source_journals_rewritten": False,
                "provider_construction_during_import": False,
                "provider_redispatch_for_imported_cells": "forbidden",
                "v2_7_no_go_preserved": True,
                "v2_7_terminal_rows_reclassified": False,
                "scientific_evidence": False,
            },
            "observation_boundary": {
                "q_ref_identity_failure_observed_before_amendment": True,
                "stage0_calibration_selection_observed_before_amendment": True,
                "stage0_guardrail_outputs_may_have_been_inspected": True,
                "stage0_candidate_winner_may_have_been_observed": True,
                "a_d_treatment_effect_outcomes_generated": False,
                "a_d_treatment_effect_outcomes_observed": False,
                "amendment_is_outcome_blind_with_respect_to_a_d_effects": True,
            },
        }
    )


def validate_v28_source_manifest(
    value: Mapping[str, Any],
    *,
    parent_repo_root: str | Path,
    child_repo_root: str | Path,
    target_contract: PilotContract | None = None,
) -> dict[str, Any]:
    """Rebuild and compare a proposed V2.8 source manifest exactly."""

    candidate = _json_copy(value)
    try:
        _verify_self_hash(
            candidate,
            schema_version=V28_SOURCE_MANIFEST_SCHEMA_VERSION,
            name="V2.8 source manifest",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    expected = build_v28_source_manifest(
        parent_repo_root=parent_repo_root,
        child_repo_root=child_repo_root,
        target_contract=target_contract,
    )
    if candidate != expected:
        raise PilotV28Stage0ImportError(
            "V2.8 source manifest differs from verified V2.7 authority"
        )
    return candidate


def write_v28_source_manifest_draft(
    path: str | Path,
    value: Mapping[str, Any],
) -> Path:
    """Write an exact manifest draft; this does not freeze a contract hash."""

    target = Path(path)
    try:
        _atomic_exact_json(target, value)
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    return target


def _validate_source_manifest_structure(
    value: Mapping[str, Any],
    *,
    contract: PilotContract | None = None,
    file_sha256: str | None = None,
) -> dict[str, Any]:
    candidate = _json_copy(value)
    try:
        _verify_self_hash(
            candidate,
            schema_version=V28_SOURCE_MANIFEST_SCHEMA_VERSION,
            name="V2.8 source manifest",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if set(candidate) != {
        "schema_version",
        "v2_7_terminal_parent",
        "published_v2_7_evidence",
        "nested_v2_6_stage0_source",
        "q_ref_audit_equivalence_reference",
        "v2_7_p95_sources_for_child_reseal",
        "imported_complete_cells",
        "cumulative_budget_debit",
        "import_policy",
        "observation_boundary",
        "integrity",
    }:
        raise PilotV28Stage0ImportError(
            "V2.8 source manifest fields drifted"
        )
    parent = _mapping(
        candidate.get("v2_7_terminal_parent"),
        name="V2.7 terminal parent",
    )
    raw = _mapping(parent.get("raw_snapshot"), name="V2.7 raw snapshot")
    denominator = _mapping(
        parent.get("terminal_denominator"),
        name="V2.7 terminal denominator",
    )
    nested = _mapping(
        candidate.get("nested_v2_6_stage0_source"),
        name="nested V2.6 Stage-0 source",
    )
    qref = _mapping(
        candidate.get("q_ref_audit_equivalence_reference"),
        name="V2.6 q-ref audit reference",
    )
    if (
        parent.get("contract", {}).get("canonical_sha256")
        != V27_CONTRACT_CANONICAL_SHA256
        or parent.get("release", {}).get("science_commit")
        != V27_SCIENCE_COMMIT
        or raw.get("file_count") != V27_RAW_FILE_COUNT
        or raw.get("storage_bytes") != V27_RAW_STORAGE_BYTES
        or raw.get("inventory_sha256") != V27_RAW_INVENTORY_SHA256
        or denominator.get("status_counts")
        != {"complete": 1, "integrity-stopped": 210}
        or denominator.get("terminal_status") != "complete-with-no-go"
        or denominator.get("scientific_complete") is not False
        or candidate.get("published_v2_7_evidence", {}).get("merge_commit")
        != V27_EVIDENCE_MERGE_COMMIT
        or nested.get("inventory", {}).get("inventory_sha256")
        != V26_RAW_INVENTORY_SHA256
        or qref.get("q_ref") != V26_QREF_VALUE
        or qref.get("imported") is not False
        or qref.get("source_result_reuse") != "forbidden"
        or qref.get("reference_use")
        != "audit-equivalence-only-for-fresh-zero-hosted-provider-regeneration"
        or qref.get("scripted_diagnostic_calls") != 48
        or qref.get("hosted_provider_calls") != 0
        or qref.get("hosted_cost_usd") != 0.0
        or qref.get("source_runner_short_run_id")
        != "q-ref-resolution-s2010922376"
        or qref.get("source_run_id")
        != qref.get("source_spec", {}).get("run_id")
        or qref.get("source_core", {}).get("identity_grid")
        != {
            "agents": 4,
            "periods": 12,
            "action_rows": 48,
            "api_usage_rows": 48,
            "utility_ledger_rows": 48,
            "shock_rows": 12,
        }
        or qref.get("source_core", {}).get("semantic_streams")
        != {
            "semantic_proposals": 0,
            "semantic_rules": 0,
            "semantic_rule_events": 0,
        }
        or candidate.get("v2_7_p95_sources_for_child_reseal")
        != _V27_P95_SOURCES
        or candidate.get("cumulative_budget_debit")
        != V28_CUMULATIVE_DEBIT.to_dict()
    ):
        raise PilotV28Stage0ImportError(
            "V2.8 source manifest parent authority drifted"
        )
    rows = candidate.get("imported_complete_cells")
    if (
        not isinstance(rows, list)
        or len(rows) != 15
        or len({row.get("target_run_id") for row in rows}) != 15
        or Counter(row.get("stage_id") for row in rows)
        != Counter({"parent-import": 1, "stage0-calibration": 14})
        or any(row.get("stage_id") == "q-ref-resolution" for row in rows)
    ):
        raise PilotV28Stage0ImportError(
            "V2.8 source manifest imported-cell inventory drifted"
        )
    policy = _mapping(
        candidate.get("import_policy"),
        name="V2.8 import policy",
    )
    boundary = _mapping(
        candidate.get("observation_boundary"),
        name="V2.8 observation boundary",
    )
    if (
        policy.get("provider_construction_during_import") is not False
        or policy.get("provider_redispatch_for_imported_cells") != "forbidden"
        or policy.get("q_ref_imported") is not False
        or policy.get("q_ref_fresh_zero_hosted_provider_regeneration_required")
        is not True
        or policy.get("q_ref_scripted_diagnostic_calls") != 48
        or policy.get("q_ref_hosted_provider_calls") != 0
        or policy.get("q_ref_hosted_cost_usd") != 0.0
        or policy.get("v2_7_no_go_preserved") is not True
        or policy.get("scientific_evidence") is not False
        or boundary.get("a_d_treatment_effect_outcomes_generated") is not False
        or boundary.get("a_d_treatment_effect_outcomes_observed") is not False
    ):
        raise PilotV28Stage0ImportError(
            "V2.8 source manifest observation/import boundary drifted"
        )
    if contract is not None:
        _validate_target_contract(contract, require_frozen=False)
        amendment = _mapping(
            contract.qref_identity_retry_amendment,
            name="V2.8 q-ref identity retry amendment",
        )
        binding = _mapping(
            amendment.get("source_manifest"),
            name="V2.8 contract source-manifest binding",
        )
        if (
            binding.get("path") != V28_SOURCE_MANIFEST_PATH.as_posix()
            or binding.get("schema_version")
            != V28_SOURCE_MANIFEST_SCHEMA_VERSION
        ):
            raise PilotV28Stage0ImportError(
                "V2.8 contract source-manifest path/schema drifted"
            )
        if contract.status == "frozen" and (
            file_sha256 != binding.get("file_sha256")
            or candidate["integrity"]["content_sha256"]
            != binding.get("content_sha256")
        ):
            raise PilotV28Stage0ImportError(
                "V2.8 frozen contract/source-manifest binding drifted"
            )
    return candidate


def load_v28_source_manifest(
    *,
    repo_root: str | Path,
    contract: PilotContract | None = None,
    parent_repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load the tracked manifest and optionally reverify live V2.7 authority."""

    try:
        root = _real_root(repo_root, name="V2.8 child repository")
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    _, raw, value = _strict_file(
        root,
        V28_SOURCE_MANIFEST_PATH,
        name="tracked V2.8 source manifest",
    )
    selected_contract = contract or load_pilot_contract(
        root.joinpath(*V28_EXPANDED_CONTRACT_PATH.parts)
    )
    candidate = _validate_source_manifest_structure(
        value,
        contract=selected_contract,
        file_sha256=_sha256(raw),
    )
    canonical_raw = (
        json.dumps(
            candidate,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if raw != canonical_raw:
        raise PilotV28Stage0ImportError(
            "tracked V2.8 source manifest is not canonical pretty JSON"
        )
    if parent_repo_root is not None:
        validate_v28_source_manifest(
            candidate,
            parent_repo_root=parent_repo_root,
            child_repo_root=root,
            target_contract=selected_contract,
        )
    return candidate


def source_binding_for_target_v28(
    source_manifest: Mapping[str, Any],
    target: PilotRunSpec | Mapping[str, Any] | str,
) -> dict[str, Any]:
    """Return the unique imported source for a V2.8 prerequisite/Stage-0 cell."""

    value = _validate_source_manifest_structure(source_manifest)
    target_run_id = (
        target
        if isinstance(target, str)
        else (
            target.run_id
            if isinstance(target, PilotRunSpec)
            else target.get("run_id")
        )
    )
    rows = [
        row
        for row in value["imported_complete_cells"]
        if isinstance(row, Mapping)
        and row.get("target_run_id") == target_run_id
    ]
    if len(rows) != 1:
        raise PilotV28Stage0ImportError(
            "target has no unique imported V2.7/V2.6 source binding"
        )
    row = _json_copy(rows[0])
    if row["stage_id"] not in {"parent-import", "stage0-calibration"}:
        raise PilotV28Stage0ImportError(
            "only parent/Stage-0 targets may use V2.8 imported sources"
        )
    if not isinstance(target, str):
        target_value = (
            target.to_dict()
            if isinstance(target, PilotRunSpec)
            else _json_copy(target)
        )
        if target_value != row["target_spec"]:
            raise PilotV28Stage0ImportError(
                "target spec differs from its V2.8 source-manifest binding"
            )
    return row


def imported_v27_raw_root_v28(raw_root: str | Path) -> Path:
    """Return the child-local root of the exact immutable V2.7 snapshot."""

    return Path(raw_root).joinpath(*V28_SNAPSHOT_RELATIVE.parts)


def snapshot_path_for_v27_source_artifact_v28(
    child_raw_root: str | Path,
    source_artifact_path: str,
) -> Path:
    """Map one V2.7 artifact path into V2.8's exact parent snapshot."""

    try:
        relative = _normalized_relative(
            source_artifact_path,
            required_top="experiment_results",
            name="V2.7 source artifact path",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    inside = PurePosixPath(*relative.parts[len(V27_RAW_ROOT.parts) :])
    if (
        tuple(relative.parts[: len(V27_RAW_ROOT.parts)])
        != V27_RAW_ROOT.parts
        or not inside.parts
    ):
        raise PilotV28Stage0ImportError(
            "source artifact is outside the V2.7 raw namespace"
        )
    return imported_v27_raw_root_v28(child_raw_root).joinpath(*inside.parts)


def imported_v26_run_dir_v28(
    raw_root: str | Path,
    v28_spec: PilotRunSpec | Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> Path:
    """Resolve an imported nested V2.6 Stage-0 run without relabelling it."""

    binding = source_binding_for_target_v28(source_manifest, v28_spec)
    if binding["stage_id"] != "stage0-calibration":
        raise PilotV28Stage0ImportError(
            "target is not a nested V2.6 Stage-0 runner directory"
        )
    run_root = binding["source_artifacts"].get("run_root")
    if not isinstance(run_root, str):
        raise PilotV28Stage0ImportError(
            "Stage-0 target lacks an imported V2.6 runner directory"
        )
    return snapshot_path_for_v27_source_artifact_v28(raw_root, run_root)


def parent_budget_debit_for_v28(
    contract: PilotContract,
) -> ParentBudgetDebit | None:
    """Return the exact cumulative V2.7 debit inherited by V2.8."""

    if contract.contract_id != V28_CONTRACT_ID:
        return None
    _validate_target_contract(contract, require_frozen=False)
    amendment = _mapping(
        contract.qref_identity_retry_amendment,
        name="V2.8 q-ref identity retry amendment",
    )
    carry = _mapping(
        amendment.get("budget_carry_forward"),
        name="V2.8 budget carry-forward",
    )
    expected_prior = V28_CUMULATIVE_DEBIT.to_dict()
    expected_prior.pop("schema_version")
    if (
        carry.get("cumulative_prior") != expected_prior
        or carry.get("budget_reset") is not False
        or carry.get("debit_before_new_dispatch") is not True
    ):
        raise PilotV28Stage0ImportError(
            "V2.8 cumulative parent debit drifted"
        )
    return V28_CUMULATIVE_DEBIT


def v28_observed_p95_receipt_path(
    raw_root: str | Path,
    profile_id: str,
) -> Path:
    return (
        Path(raw_root)
        / "parent-import"
        / "observed_p95"
        / profile_id
        / "observed_p95_authority_receipt.json"
    )


def v28_observed_p95_projection_path(
    raw_root: str | Path,
    profile_id: str,
) -> Path:
    return (
        Path(raw_root)
        / "parent-import"
        / "observed_p95"
        / profile_id
        / "projection_p95.json"
    )


def _child_contract_binding(
    child_root: Path,
    contract: PilotContract,
) -> dict[str, Any]:
    _, raw, value = _strict_file(
        child_root,
        V28_EXPANDED_CONTRACT_PATH,
        name="expanded V2.8 contract",
    )
    parsed = PilotContract.from_dict(value)
    if (
        parsed.contract_id != contract.contract_id
        or parsed.canonical_hash != contract.canonical_hash
        or parsed.to_dict() != contract.to_dict()
    ):
        raise PilotV28Stage0ImportError(
            "expanded V2.8 contract differs from selected contract"
        )
    return {
        "path": V28_EXPANDED_CONTRACT_PATH.as_posix(),
        "file_sha256": _sha256(raw),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
    }


def _source_manifest_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return {
        "path": V28_SOURCE_MANIFEST_PATH.as_posix(),
        "schema_version": V28_SOURCE_MANIFEST_SCHEMA_VERSION,
        "file_sha256": _sha256(raw),
        "content_sha256": value["integrity"]["content_sha256"],
    }


def _read_artifact_input(
    value_or_path: Mapping[str, Any] | str | Path,
    *,
    repo_root: Path,
    name: str,
) -> tuple[dict[str, Any], PurePosixPath | None, bytes | None]:
    if isinstance(value_or_path, Mapping):
        return _json_copy(value_or_path), None, None
    path = Path(value_or_path)
    if path.is_absolute():
        try:
            relative = PurePosixPath(*path.absolute().relative_to(repo_root).parts)
        except ValueError as exc:
            raise PilotV28Stage0ImportError(
                f"{name} escaped the repository"
            ) from exc
    else:
        try:
            relative = _normalized_relative(
                path,
                required_top="experiment_results",
                name=f"{name} path",
            )
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
    _, raw, value = _strict_file(repo_root, relative, name=name)
    return value, relative, raw


def v2_7_p95_source_binding_v28(
    *,
    repo_root: str | Path,
    child_raw_root: str | Path,
    profile_id: str,
) -> dict[str, Any]:
    """Verify one copied V2.7 p95 authority for V2.8 child resealing."""

    if profile_id not in V28_ALLOWED_P95_PROFILES:
        raise PilotV28Stage0ImportError(
            f"{profile_id} has no imported V2.7 p95 source"
        )
    try:
        root = _real_root(repo_root, name="V2.8 child repository")
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    contract = load_pilot_contract(
        root.joinpath(*V28_EXPANDED_CONTRACT_PATH.parts)
    )
    raw_root = Path(child_raw_root).absolute()
    if (
        _repo_relative(root, raw_root, name="V2.8 raw root")
        != V28_RAW_ROOT.as_posix()
    ):
        raise PilotV28Stage0ImportError(
            "V2.7 p95 source requires the exact V2.8 raw namespace"
        )
    expected_commit = _git(root, "rev-parse", "HEAD")
    verify_v28_parent_import_receipt(
        raw_root / "parent-import/parent_import_receipt.json",
        repo_root=root,
        contract=contract,
        expected_git_commit=expected_commit,
    )
    manifest = load_v28_source_manifest(repo_root=root, contract=contract)
    source = manifest["v2_7_p95_sources_for_child_reseal"][profile_id]
    output = {
        "source_contract_sha256": V27_CONTRACT_CANONICAL_SHA256,
        "source_git_tag": V27_SCIENCE_TAG,
        "source_git_commit": V27_SCIENCE_COMMIT,
        "model_id": profile_id,
        "runtime_model": source["runtime_model"],
        "served_model": source["served_model"],
    }
    values: dict[str, dict[str, Any]] = {}
    for kind in ("authority", "projection"):
        expected = source[kind]
        path = snapshot_path_for_v27_source_artifact_v28(
            raw_root,
            expected["path"],
        )
        if path.is_symlink() or not path.is_file():
            raise PilotV28Stage0ImportError(
                f"imported V2.7 {profile_id} {kind} is unavailable"
            )
        raw = path.read_bytes()
        try:
            value = _strict_json(
                raw,
                name=f"imported V2.7 {profile_id} {kind}",
            )
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
        integrity = value.get("integrity")
        if (
            _sha256(raw) != expected["file_sha256"]
            or value.get("schema_version") != expected["schema_version"]
            or not isinstance(integrity, Mapping)
            or integrity.get("canonicalization") != CANONICALIZATION
            or integrity.get("content_sha256")
            != expected["content_sha256"]
        ):
            raise PilotV28Stage0ImportError(
                f"imported V2.7 {profile_id} {kind} drifted"
            )
        values[kind] = value
        output[kind] = {
            **_json_copy(expected),
            "snapshot_path": _repo_relative(
                root,
                path,
                name=f"imported V2.7 {profile_id} {kind}",
            ),
        }
    reservations = values["authority"].get("reservations")
    runtime_model = source["runtime_model"]
    served_model = source["served_model"]
    projection = values["projection"].get("projection")
    if (
        not isinstance(reservations, Mapping)
        or set(reservations) != {runtime_model}
        or not isinstance(projection, Mapping)
    ):
        raise PilotV28Stage0ImportError(
            f"imported V2.7 {profile_id} p95 payload is malformed"
        )
    for call_kind in ("action", "semantic"):
        entry = reservations[runtime_model].get(call_kind)
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"authority", "reservation"}
            or not isinstance(entry.get("authority"), Mapping)
        ):
            raise PilotV28Stage0ImportError(
                f"imported V2.7 {profile_id}/{call_kind} p95 entry is malformed"
            )
        authority = entry["authority"]
        if (
            authority.get("pilot_contract_hash")
            != V27_CONTRACT_CANONICAL_SHA256
            or authority.get("pilot_tag") != V27_SCIENCE_TAG
            or authority.get("source_model_id") != profile_id
            or authority.get("source_served_model") != served_model
            or entry.get("reservation")
            != projection.get(f"{served_model}::{call_kind}")
        ):
            raise PilotV28Stage0ImportError(
                f"imported V2.7 {profile_id}/{call_kind} authority drifted"
            )
        try:
            parsed = PreflightP95Reservation.from_dict(
                model=runtime_model,
                call_kind=call_kind,
                value=entry["reservation"],
            )
        except (TypeError, ValueError) as exc:
            raise PilotV28Stage0ImportError(
                f"imported V2.7 {profile_id}/{call_kind} reservation is invalid"
            ) from exc
        if parsed.to_dict() != entry["reservation"]:
            raise PilotV28Stage0ImportError(
                f"imported V2.7 {profile_id}/{call_kind} reservation drifted"
            )
    output["reservations"] = _json_copy(reservations)
    return output


def build_v28_resealed_observed_p95_authority(
    *,
    repo_root: str | Path,
    contract: PilotContract,
    contract_path: str | Path,
    raw_root: str | Path,
    profile_id: str,
    expected_git_commit: str,
    verified_v2_7_source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Build, but do not persist, one V2.8 receipt and projection."""

    try:
        root = _real_root(repo_root, name="V2.8 child repository")
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    _validate_target_contract(contract, require_frozen=True)
    if (
        _COMMIT_RE.fullmatch(expected_git_commit) is None
        or profile_id not in V28_ALLOWED_P95_PROFILES
    ):
        raise PilotV28Stage0ImportError(
            "V2.8 p95 release commit or profile is invalid"
        )
    try:
        contract_relative = _normalized_relative(
            contract_path,
            required_top="experiments",
            name="V2.8 p95 contract path",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if contract_relative != V28_EXPANDED_CONTRACT_PATH:
        raise PilotV28Stage0ImportError(
            "V2.8 p95 requires the expanded contract path"
        )
    contract_binding = _child_contract_binding(root, contract)
    raw_path = Path(raw_root)
    if not raw_path.is_absolute():
        raw_path = root.joinpath(*PurePosixPath(str(raw_root)).parts)
    raw_path = raw_path.absolute()
    if (
        _repo_relative(root, raw_path, name="V2.8 p95 raw root")
        != V28_RAW_ROOT.as_posix()
    ):
        raise PilotV28Stage0ImportError(
            "V2.8 p95 requires the exact raw namespace"
        )
    source = _json_copy(verified_v2_7_source_binding)
    expected_source = _V27_P95_SOURCES[profile_id]
    if any(
        source.get(key) != expected
        for key, expected in {
            "source_contract_sha256": V27_CONTRACT_CANONICAL_SHA256,
            "source_git_tag": V27_SCIENCE_TAG,
            "source_git_commit": V27_SCIENCE_COMMIT,
            "model_id": profile_id,
            "runtime_model": expected_source["runtime_model"],
            "served_model": expected_source["served_model"],
        }.items()
    ):
        raise PilotV28Stage0ImportError(
            "verified V2.7 p95 source identity drifted"
        )
    for kind in ("authority", "projection"):
        if any(
            source.get(kind, {}).get(key) != expected_source[kind][key]
            for key in (
                "path",
                "schema_version",
                "file_sha256",
                "content_sha256",
            )
        ):
            raise PilotV28Stage0ImportError(
                f"verified V2.7 {profile_id} {kind} binding drifted"
            )
    profile = contract.provider_profiles[profile_id]
    runtime_model = expected_source["runtime_model"]
    expected_runtime = f"{profile.transport}/{profile.requested_model}"
    if (
        runtime_model != expected_runtime
        or profile.served_model != expected_source["served_model"]
    ):
        raise PilotV28Stage0ImportError(
            f"V2.8 {profile_id} provider identity differs from V2.7"
        )
    reservations = _json_copy(source.get("reservations"))
    if set(reservations) != {runtime_model}:
        raise PilotV28Stage0ImportError(
            f"verified V2.7 {profile_id} reservations are malformed"
        )
    for call_kind in ("action", "semantic"):
        entry = reservations[runtime_model].get(call_kind)
        if not isinstance(entry, dict) or not isinstance(
            entry.get("authority"), dict
        ):
            raise PilotV28Stage0ImportError(
                f"verified V2.7 {profile_id}/{call_kind} entry is malformed"
            )
        try:
            parsed = PreflightP95Reservation.from_dict(
                model=runtime_model,
                call_kind=call_kind,
                value=entry.get("reservation"),
            )
        except (TypeError, ValueError) as exc:
            raise PilotV28Stage0ImportError(
                f"verified V2.7 {profile_id}/{call_kind} reservation is invalid"
            ) from exc
        entry["reservation"] = parsed.to_dict()
        authority = entry["authority"]
        if (
            authority.get("pilot_contract_hash")
            != V27_CONTRACT_CANONICAL_SHA256
            or authority.get("pilot_tag") != V27_SCIENCE_TAG
            or authority.get("source_model_id") != profile_id
            or authority.get("source_served_model") != profile.served_model
        ):
            raise PilotV28Stage0ImportError(
                f"verified V2.7 {profile_id}/{call_kind} authority drifted"
            )
        authority["pilot_contract_hash"] = contract.canonical_hash
        authority["pilot_tag"] = V28_SCIENCE_TAG
    parent_source = {
        key: _json_copy(source[key])
        for key in (
            "source_contract_sha256",
            "source_git_tag",
            "source_git_commit",
            "model_id",
            "runtime_model",
            "served_model",
            "authority",
            "projection",
        )
    }
    receipt = _seal(
        {
            "schema_version": V28_RESEALED_P95_AUTHORITY_SCHEMA_VERSION,
            "contract": contract_binding,
            "raw_root": V28_RAW_ROOT.as_posix(),
            "git": {
                "tag": V28_SCIENCE_TAG,
                "commit": expected_git_commit,
            },
            "model": {
                "model_id": profile_id,
                "runtime_model": runtime_model,
                "served_model": profile.served_model,
            },
            "parent_source": parent_source,
            "reservations": reservations,
            "scientific_evidence": False,
            "evidence_use": (
                "V2.8 prospective budget authority only; copied V2.7/V2.6 "
                "prerequisites contribute no V2.8 A-D treatment effect."
            ),
        }
    )
    receipt_path = v28_observed_p95_receipt_path(raw_path, profile_id)
    receipt_relative = _repo_relative(
        root,
        receipt_path,
        name=f"{profile_id} V2.8 p95 receipt",
    )
    projection = {
        f"{profile.served_model}::{call_kind}": _json_copy(
            reservations[runtime_model][call_kind]["reservation"]
        )
        for call_kind in ("action", "semantic")
    }
    projection_value = _seal(
        {
            "schema_version": "finevo-pilot-projection-p95-v1",
            "model_id": profile_id,
            "served_model": profile.served_model,
            "projection": projection,
            "bindings": {
                "contract_sha256": contract.canonical_hash,
                "git_tag": V28_SCIENCE_TAG,
                "git_commit": expected_git_commit,
                "source_kind": V28_RESEALED_P95_SOURCE_KIND,
                "source_authority_receipt": receipt_relative,
                "source_authority_receipt_content_sha256": receipt["integrity"][
                    "content_sha256"
                ],
                "source_v2_7_authority_content_sha256": parent_source[
                    "authority"
                ]["content_sha256"],
                "source_v2_7_projection_content_sha256": parent_source[
                    "projection"
                ]["content_sha256"],
            },
        }
    )
    return {
        "receipt_path": receipt_path,
        "projection_path": v28_observed_p95_projection_path(
            raw_path,
            profile_id,
        ),
        "receipt": receipt,
        "projection": projection_value,
    }


def _rebuild_v28_p95_from_receipt(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_git_commit: str,
) -> dict[str, Any]:
    value = _json_copy(receipt)
    if (
        set(value)
        != {
            "schema_version",
            "contract",
            "raw_root",
            "git",
            "model",
            "parent_source",
            "reservations",
            "scientific_evidence",
            "evidence_use",
            "integrity",
        }
        or value.get("schema_version")
        != V28_RESEALED_P95_AUTHORITY_SCHEMA_VERSION
    ):
        raise PilotV28Stage0ImportError(
            "V2.8 observed-p95 receipt shape or schema drifted"
        )
    try:
        _verify_self_hash(
            value,
            schema_version=V28_RESEALED_P95_AUTHORITY_SCHEMA_VERSION,
            name="V2.8 observed-p95 receipt",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    git = _mapping(value.get("git"), name="V2.8 p95 git binding")
    model = _mapping(value.get("model"), name="V2.8 p95 model binding")
    contract_binding = _mapping(
        value.get("contract"),
        name="V2.8 p95 contract binding",
    )
    if (
        git.get("tag") != V28_SCIENCE_TAG
        or git.get("commit") != expected_git_commit
        or _COMMIT_RE.fullmatch(expected_git_commit) is None
        or value.get("raw_root") != V28_RAW_ROOT.as_posix()
    ):
        raise PilotV28Stage0ImportError(
            "V2.8 observed-p95 release binding drifted"
        )
    try:
        contract_relative = _normalized_relative(
            str(contract_binding.get("path", "")),
            required_top="experiments",
            name="V2.8 p95 receipt contract path",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if contract_relative != V28_EXPANDED_CONTRACT_PATH:
        raise PilotV28Stage0ImportError(
            "V2.8 observed-p95 contract path drifted"
        )
    _, contract_raw, contract_value = _strict_file(
        repo_root,
        contract_relative,
        name="V2.8 p95 receipt contract",
    )
    contract = PilotContract.from_dict(contract_value)
    if (
        contract_binding
        != {
            "path": contract_relative.as_posix(),
            "file_sha256": _sha256(contract_raw),
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
        }
        or contract.contract_id != V28_CONTRACT_ID
    ):
        raise PilotV28Stage0ImportError(
            "V2.8 observed-p95 contract binding drifted"
        )
    profile_id = str(model.get("model_id"))
    raw_root = repo_root.joinpath(*V28_RAW_ROOT.parts)
    source = v2_7_p95_source_binding_v28(
        repo_root=repo_root,
        child_raw_root=raw_root,
        profile_id=profile_id,
    )
    return build_v28_resealed_observed_p95_authority(
        repo_root=repo_root,
        contract=contract,
        contract_path=contract_relative.as_posix(),
        raw_root=raw_root,
        profile_id=profile_id,
        expected_git_commit=expected_git_commit,
        verified_v2_7_source_binding=source,
    )


def verify_v28_resealed_observed_p95_authority(
    receipt_or_path: Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Rebuild one V2.8 receipt from the exact copied V2.7 authority."""

    try:
        root = _real_root(repo_root, name="V2.8 child repository")
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    receipt, _, _ = _read_artifact_input(
        receipt_or_path,
        repo_root=root,
        name="V2.8 observed-p95 authority receipt",
    )
    rebuilt = _rebuild_v28_p95_from_receipt(
        receipt,
        repo_root=root,
        expected_git_commit=expected_git_commit,
    )
    if receipt != rebuilt["receipt"]:
        raise PilotV28Stage0ImportError(
            "V2.8 observed-p95 receipt differs from verified V2.7 source"
        )
    return _json_copy(rebuilt["receipt"]["reservations"])


def verify_v28_resealed_observed_p95_projection(
    projection_or_path: Mapping[str, Any] | str | Path,
    *,
    receipt_or_path: Mapping[str, Any] | str | Path,
    repo_root: str | Path,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Verify the paired V2.8 projection against its rebuilt receipt."""

    try:
        root = _real_root(repo_root, name="V2.8 child repository")
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    receipt, _, _ = _read_artifact_input(
        receipt_or_path,
        repo_root=root,
        name="V2.8 observed-p95 authority receipt",
    )
    rebuilt = _rebuild_v28_p95_from_receipt(
        receipt,
        repo_root=root,
        expected_git_commit=expected_git_commit,
    )
    if receipt != rebuilt["receipt"]:
        raise PilotV28Stage0ImportError(
            "V2.8 observed-p95 receipt differs from verified V2.7 source"
        )
    projection, _, _ = _read_artifact_input(
        projection_or_path,
        repo_root=root,
        name="V2.8 observed-p95 projection",
    )
    if projection != rebuilt["projection"]:
        raise PilotV28Stage0ImportError(
            "V2.8 observed-p95 projection differs from its receipt"
        )
    return _json_copy(projection)


def verified_v28_observed_p95_authority_binding(
    receipt_path: str | Path,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Return a guarded V2.8 receipt binding plus verified reservations."""

    try:
        root = _real_root(repo_root, name="V2.8 child repository")
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    receipt, relative, raw = _read_artifact_input(
        receipt_path,
        repo_root=root,
        name="V2.8 observed-p95 authority receipt",
    )
    if relative is None or raw is None:
        raise PilotV28Stage0ImportError(
            "V2.8 authority binding requires a receipt path"
        )
    reservations = verify_v28_resealed_observed_p95_authority(
        receipt,
        repo_root=root,
        expected_git_commit=expected_git_commit,
    )
    return {
        "receipt_path": relative.as_posix(),
        "receipt_file_sha256": _sha256(raw),
        "receipt_content_sha256": receipt["integrity"]["content_sha256"],
        "git_commit": expected_git_commit,
        "reservations": reservations,
    }


def _atomic_exact_json_no_follow(
    *,
    repo_root: Path,
    path: Path,
    value: Mapping[str, Any],
) -> None:
    raw = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        _v27_atomic_exact_bytes_no_follow(
            repo_root=repo_root,
            path=path,
            raw=raw,
        )
    except PilotV27Stage0ImportError as exc:
        raise _translate(exc) from exc


def _copy_exact_snapshot(
    *,
    source_root: Path,
    destination_root: Path,
    destination_guard_root: Path,
    inventory: Sequence[Mapping[str, Any]],
) -> None:
    for row in inventory:
        try:
            relative = _normalized_relative(
                str(row.get("path", "")),
                required_top=None,
                name="V2.7 raw inventory path",
            )
            _, raw = _guarded_file(
                source_root,
                relative,
                name=f"V2.7 raw {relative.as_posix()}",
            )
            _v27_atomic_exact_bytes_no_follow(
                repo_root=destination_guard_root,
                path=destination_root.joinpath(*relative.parts),
                raw=raw,
            )
        except (PilotV24ParentImportError, PilotV27Stage0ImportError) as exc:
            raise _translate(exc) from exc
        if len(raw) != row.get("byte_size") or _sha256(raw) != row.get("sha256"):
            raise PilotV28Stage0ImportError(
                f"V2.7 raw source changed during copy: {relative.as_posix()}"
            )
    copied_rows, copied = _inventory(
        destination_root,
        declared_root=V27_RAW_ROOT,
    )
    source_canonical = json.dumps(
        list(inventory),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    copied_canonical = json.dumps(
        copied_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if (
        source_canonical != copied_canonical
        or copied["file_count"] != V27_RAW_FILE_COUNT
        or copied["storage_bytes"] != V27_RAW_STORAGE_BYTES
        or copied["inventory_sha256"] != V27_RAW_INVENTORY_SHA256
    ):
        raise PilotV28Stage0ImportError(
            "V2.8 copied V2.7 raw snapshot differs from its source"
        )


def _build_v28_parent_import_receipt(
    *,
    child_root: Path,
    contract: PilotContract,
    child_git_commit: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return _seal(
        {
            "schema_version": V28_PARENT_IMPORT_SCHEMA_VERSION,
            "contract": _child_contract_binding(child_root, contract),
            "child_release": {
                "git_tag": V28_SCIENCE_TAG,
                "git_commit": child_git_commit,
            },
            "source_manifest": _source_manifest_binding(manifest),
            "v2_7_terminal_parent": _json_copy(
                manifest["v2_7_terminal_parent"]
            ),
            "published_v2_7_evidence": _json_copy(
                manifest["published_v2_7_evidence"]
            ),
            "nested_v2_6_stage0_source": _json_copy(
                manifest["nested_v2_6_stage0_source"]
            ),
            "q_ref_audit_equivalence_reference": _json_copy(
                manifest["q_ref_audit_equivalence_reference"]
            ),
            "v2_7_p95_sources_for_child_reseal": _json_copy(
                manifest["v2_7_p95_sources_for_child_reseal"]
            ),
            "cumulative_budget_debit": _json_copy(
                manifest["cumulative_budget_debit"]
            ),
            "imported_complete_cells": [
                {
                    "stage_id": row["stage_id"],
                    "source_authority_contract_id": row[
                        "source_authority_contract_id"
                    ],
                    "physical_source_contract_id": row.get(
                        "physical_source_contract_id"
                    ),
                    "source_run_id": row["source_run_id"],
                    "target_run_id": row["target_run_id"],
                    "source_artifacts": _json_copy(row["source_artifacts"]),
                }
                for row in manifest["imported_complete_cells"]
            ],
            "copied_raw_snapshot": {
                "source_root": V27_RAW_ROOT.as_posix(),
                "snapshot_root": (
                    V28_RAW_ROOT / V28_SNAPSHOT_RELATIVE
                ).as_posix(),
                "file_count": V27_RAW_FILE_COUNT,
                "storage_bytes": V27_RAW_STORAGE_BYTES,
                "inventory_sha256": V27_RAW_INVENTORY_SHA256,
                "exact_bytes": True,
                "nested_v2_6_inventory_sha256": V26_RAW_INVENTORY_SHA256,
            },
            "provider_calls_during_import": 0,
            "hosted_provider_calls_during_import": 0,
            "local_model_calls_during_import": 0,
            "q_ref_imported": False,
            "q_ref_regeneration": {
                "fresh": True,
                "scripted_diagnostic_calls": 48,
                "hosted_provider_calls": 0,
                "hosted_cost_usd": 0.0,
                "status": "required-after-parent-import",
            },
            "a_d_treatment_effect_outcomes_generated": False,
            "a_d_treatment_effect_outcomes_observed": False,
            "scientific_evidence": False,
            "v2_7_terminal_no_go_preserved": True,
            "source_artifacts_rewritten": False,
        }
    )


def verify_v28_parent_import_receipt(
    receipt_or_path: Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path,
    contract: PilotContract,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Verify V2.8 import receipt, tracked manifest, and copied raw snapshot."""

    try:
        root = _real_root(repo_root, name="V2.8 child repository")
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    _validate_target_contract(contract, require_frozen=True)
    if _COMMIT_RE.fullmatch(expected_git_commit) is None:
        raise PilotV28Stage0ImportError(
            "V2.8 expected commit must be 40 lowercase hex characters"
        )
    value, _, _ = _read_artifact_input(
        receipt_or_path,
        repo_root=root,
        name="V2.8 parent-import receipt",
    )
    try:
        _verify_self_hash(
            value,
            schema_version=V28_PARENT_IMPORT_SCHEMA_VERSION,
            name="V2.8 parent-import receipt",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    manifest = load_v28_source_manifest(repo_root=root, contract=contract)
    snapshot = root.joinpath(
        *V28_RAW_ROOT.parts,
        *V28_SNAPSHOT_RELATIVE.parts,
    )
    _verify_exact_v27_inventory(snapshot)
    _verify_exact_nested_v26_inventory(
        snapshot.joinpath(*V26_NESTED_RELATIVE.parts)
    )
    expected = _build_v28_parent_import_receipt(
        child_root=root,
        contract=contract,
        child_git_commit=expected_git_commit,
        manifest=manifest,
    )
    if value != expected:
        raise PilotV28Stage0ImportError(
            "V2.8 parent-import receipt differs from sealed sources"
        )
    return value


def _materialize_v28_resealed_p95(
    *,
    child_root: Path,
    child_raw: Path,
    contract: PilotContract,
    child_git_commit: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for profile_id in V28_ALLOWED_P95_PROFILES:
        source = v2_7_p95_source_binding_v28(
            repo_root=child_root,
            child_raw_root=child_raw,
            profile_id=profile_id,
        )
        built = build_v28_resealed_observed_p95_authority(
            repo_root=child_root,
            contract=contract,
            contract_path=V28_EXPANDED_CONTRACT_PATH.as_posix(),
            raw_root=child_raw,
            profile_id=profile_id,
            expected_git_commit=child_git_commit,
            verified_v2_7_source_binding=source,
        )
        receipt_path = Path(built["receipt_path"])
        projection_path = Path(built["projection_path"])
        _atomic_exact_json_no_follow(
            repo_root=child_root,
            path=receipt_path,
            value=built["receipt"],
        )
        _atomic_exact_json_no_follow(
            repo_root=child_root,
            path=projection_path,
            value=built["projection"],
        )
        reservations = verify_v28_resealed_observed_p95_authority(
            receipt_path,
            repo_root=child_root,
            expected_git_commit=child_git_commit,
        )
        projection = verify_v28_resealed_observed_p95_projection(
            projection_path,
            receipt_or_path=receipt_path,
            repo_root=child_root,
            expected_git_commit=child_git_commit,
        )
        receipt_raw = receipt_path.read_bytes()
        projection_raw = projection_path.read_bytes()
        output[profile_id] = {
            "receipt": {
                "path": _repo_relative(
                    child_root,
                    receipt_path,
                    name=f"{profile_id} V2.8 p95 receipt",
                ),
                "file_sha256": _sha256(receipt_raw),
                "content_sha256": built["receipt"]["integrity"][
                    "content_sha256"
                ],
            },
            "projection": {
                "path": _repo_relative(
                    child_root,
                    projection_path,
                    name=f"{profile_id} V2.8 p95 projection",
                ),
                "file_sha256": _sha256(projection_raw),
                "content_sha256": projection["integrity"]["content_sha256"],
            },
            "runtime_models": sorted(reservations),
            "source_kind": V28_RESEALED_P95_SOURCE_KIND,
        }
    return output


def persist_v28_parent_import(
    *,
    contract: PilotContract,
    repo_root: str | Path,
    raw_root: str | Path,
    parent_repo_root: str | Path,
    child_git_tag: str,
    child_git_commit: str,
    source_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy exact V2.7 raw, seal the import, and reseal both p95 profiles."""

    try:
        child_root = _real_root(repo_root, name="V2.8 child repository")
        parent_root = _real_root(
            parent_repo_root,
            name="V2.7 source repository",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    child_raw = Path(raw_root).absolute()
    if (
        _repo_relative(child_root, child_raw, name="V2.8 raw root")
        != V28_RAW_ROOT.as_posix()
    ):
        raise PilotV28Stage0ImportError(
            "V2.8 import requires its fresh pilot-v2.8 raw namespace"
        )
    _validate_target_contract(contract, require_frozen=True)
    if (
        child_git_tag != V28_SCIENCE_TAG
        or _COMMIT_RE.fullmatch(child_git_commit) is None
    ):
        raise PilotV28Stage0ImportError(
            "V2.8 child release tag or commit is malformed"
        )
    manifest = (
        validate_v28_source_manifest(
            source_manifest,
            parent_repo_root=parent_root,
            child_repo_root=child_root,
            target_contract=contract,
        )
        if source_manifest is not None
        else load_v28_source_manifest(
            repo_root=child_root,
            contract=contract,
            parent_repo_root=parent_root,
        )
    )
    source_raw = parent_root.joinpath(*V27_RAW_ROOT.parts)
    inventory = _verify_exact_v27_inventory(source_raw)
    snapshot = child_raw.joinpath(*V28_SNAPSHOT_RELATIVE.parts)
    _copy_exact_snapshot(
        source_root=source_raw,
        destination_root=snapshot,
        destination_guard_root=child_root,
        inventory=inventory,
    )
    receipt = _build_v28_parent_import_receipt(
        child_root=child_root,
        contract=contract,
        child_git_commit=child_git_commit,
        manifest=manifest,
    )
    receipt_path = child_raw / "parent-import/parent_import_receipt.json"
    _atomic_exact_json_no_follow(
        repo_root=child_root,
        path=receipt_path,
        value=receipt,
    )
    resealed_p95 = _materialize_v28_resealed_p95(
        child_root=child_root,
        child_raw=child_raw,
        contract=contract,
        child_git_commit=child_git_commit,
    )
    return {
        "receipt": str(receipt_path),
        "receipt_content_sha256": receipt["integrity"]["content_sha256"],
        "snapshot_root": str(snapshot),
        "snapshot_inventory_sha256": V27_RAW_INVENTORY_SHA256,
        "nested_v2_6_inventory_sha256": V26_RAW_INVENTORY_SHA256,
        "imported_cell_count": 15,
        "imported_stage0_cell_count": 14,
        "q_ref_imported": False,
        "imported_profiles": sorted(V28_ALLOWED_P95_PROFILES),
        "resealed_p95_profiles": resealed_p95,
        "provider_calls_during_import": 0,
        "scientific_evidence": False,
        "v2_7_terminal_no_go_preserved": True,
    }


__all__ = [
    "PilotV28Stage0ImportError",
    "V26_RAW_FILE_COUNT",
    "V26_RAW_INVENTORY_SHA256",
    "V26_RAW_STORAGE_BYTES",
    "V27_RAW_FILE_COUNT",
    "V27_RAW_INVENTORY_SHA256",
    "V27_RAW_STORAGE_BYTES",
    "V28_ALLOWED_P95_PROFILES",
    "V28_CONTRACT_ID",
    "V28_PARENT_IMPORT_SCHEMA_VERSION",
    "V28_RAW_ROOT",
    "V28_RESEALED_P95_AUTHORITY_SCHEMA_VERSION",
    "V28_RESEALED_P95_SOURCE_KIND",
    "V28_SCIENCE_TAG",
    "V28_SNAPSHOT_RELATIVE",
    "V28_SOURCE_MANIFEST_PATH",
    "V28_SOURCE_MANIFEST_SCHEMA_VERSION",
    "build_v28_resealed_observed_p95_authority",
    "build_v28_source_manifest",
    "imported_v26_run_dir_v28",
    "imported_v27_raw_root_v28",
    "load_v28_source_manifest",
    "parent_budget_debit_for_v28",
    "persist_v28_parent_import",
    "snapshot_path_for_v27_source_artifact_v28",
    "source_binding_for_target_v28",
    "v2_7_p95_source_binding_v28",
    "v28_observed_p95_projection_path",
    "v28_observed_p95_receipt_path",
    "validate_v28_source_manifest",
    "verified_v28_observed_p95_authority_binding",
    "verify_v28_parent_import_receipt",
    "verify_v28_resealed_observed_p95_authority",
    "verify_v28_resealed_observed_p95_projection",
    "write_v28_source_manifest_draft",
]
