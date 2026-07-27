"""Zero-provider-call V2.3 -> V2.4 parent authority import.

The V2.3 pilot is terminal and immutable.  V2.4 therefore cannot reopen its
cells or relabel the old capability/preflight work as treatment evidence.  It
may, however, inherit two inputs that were fixed before any A--D outcome was
observed:

* the complete cumulative budget debit; and
* the source-backed closed-loop p95 reservations for GPT-5.2 and the frozen
  local Llama-3.3 profile.

This module performs that import without constructing a provider.  The import
first revalidates the parent tag, contract, tamper-evident ledgers, terminal
denominator, release/stage receipts, published no-go package, and every source
p95 receipt.  Exact parent receipt bytes are then copied into the ignored
child raw tree and wrapped by a child-contract/tag-bound authority receipt.
The tracked parent-source manifest fixes all admissible hashes, so a raw-tree
receipt cannot be edited and merely rehashed after import.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Any, Mapping

from .pilot_budget import ParentBudgetDebit, PilotBudgetLedger
from .pilot_contract import PilotContract, canonical_sha256, load_pilot_contract


V24_CONTRACT_ID = "finevo-pilot-v2.4"
V24_SCIENCE_TAG = "pilot-v2.4-science"
V24_PARENT_IMPORT_SCHEMA_VERSION = "finevo-pilot-v2.4-parent-import-v1"
V24_INHERITED_P95_RECEIPT_SCHEMA_VERSION = (
    "finevo-inherited-observed-p95-authority-receipt-v1"
)
V24_PARENT_SOURCE_MANIFEST_SCHEMA_VERSION = (
    "finevo-pilot-v2.4-parent-source-manifest-v1"
)
V24_PARENT_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_4_parent_source_manifest.json"
)
V24_PARENT_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "7ae427fe6eac5aa6e04eddd3efa9e63405e128c782013ed3f67c35808be3cec5"
)
V24_PARENT_DEBIT_RECORD_SHA256 = (
    "8371a27428ad044a3e7f959815717bff6b4b13f754bab34670056d89b9981019"
)
V24_ALLOWED_P95_PROFILES = ("gpt52_main", "llama33_local_controlled")
V24_BOUNDARY_ONLY_P95_PROFILES = ("gpt56_diagnostic",)
CANONICALIZATION = "json-sort-keys-utf8-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class PilotV24ParentImportError(RuntimeError):
    """Raised before a parent authority can enter the V2.4 raw tree."""


def _json_copy(value: Any) -> Any:
    def thaw(candidate: Any) -> Any:
        if isinstance(candidate, Mapping):
            return {
                str(key): thaw(nested)
                for key, nested in candidate.items()
            }
        if isinstance(candidate, (list, tuple)):
            return [thaw(nested) for nested in candidate]
        return candidate

    try:
        return json.loads(
            json.dumps(
                thaw(value),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise PilotV24ParentImportError(
            "parent-import value is not canonical JSON"
        ) from exc


def _strict_json(raw: bytes, *, name: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PilotV24ParentImportError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise PilotV24ParentImportError(
            f"{name} contains non-finite JSON number {value}"
        )

    try:
        value = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotV24ParentImportError(
            f"{name} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise PilotV24ParentImportError(f"{name} must contain a JSON object")
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _bound_content_sha256(value: Mapping[str, Any]) -> str:
    copied = _json_copy(value)
    integrity = copied.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("content_sha256", None)
    return canonical_sha256(copied)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_copy(value)
    integrity = result.setdefault("integrity", {})
    if not isinstance(integrity, dict):
        raise PilotV24ParentImportError("bound integrity must be an object")
    integrity["canonicalization"] = CANONICALIZATION
    integrity.pop("content_sha256", None)
    integrity["content_sha256"] = _bound_content_sha256(result)
    return result


def _verify_self_hash(
    value: Mapping[str, Any],
    *,
    schema_version: str,
    name: str,
) -> None:
    integrity = value.get("integrity")
    if (
        value.get("schema_version") != schema_version
        or not isinstance(integrity, Mapping)
        or set(integrity) != {"canonicalization", "content_sha256"}
        or integrity.get("canonicalization") != CANONICALIZATION
        or integrity.get("content_sha256") != _bound_content_sha256(value)
    ):
        raise PilotV24ParentImportError(
            f"{name} schema or content hash mismatch"
        )


def _normalized_relative(
    value: str | Path,
    *,
    required_top: str | None,
    name: str,
) -> PurePosixPath:
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or "\x00" in text
        or Path(text).is_absolute()
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != text
        or (required_top is not None and path.parts[0] != required_top)
    ):
        raise PilotV24ParentImportError(
            f"{name} must be a normalized repository-relative POSIX path"
        )
    return path


def _real_root(value: str | Path, *, name: str) -> Path:
    root = Path(value).absolute()
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise PilotV24ParentImportError(f"{name} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PilotV24ParentImportError(
            f"{name} must be a real non-symlink directory"
        )
    return root


def _guarded_file(
    repo_root: Path,
    relative: PurePosixPath,
    *,
    name: str,
) -> tuple[Path, bytes]:
    current = repo_root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise PilotV24ParentImportError(
                f"required {name} is missing: {relative.as_posix()}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise PilotV24ParentImportError(
                f"{name} path contains a symlink: {relative.as_posix()}"
            )
        final = index == len(relative.parts) - 1
        if final:
            if not stat.S_ISREG(metadata.st_mode):
                raise PilotV24ParentImportError(
                    f"{name} must be a regular file"
                )
        elif not stat.S_ISDIR(metadata.st_mode):
            raise PilotV24ParentImportError(
                f"{name} parent is not a directory"
            )
    before = current.stat()
    raw = current.read_bytes()
    after = current.stat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise PilotV24ParentImportError(f"{name} changed during read")
    return current, raw


def _git(repo_root: Path, *arguments: str) -> str:
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
        raise PilotV24ParentImportError("system git binary is unavailable")
    try:
        completed = subprocess.run(
            (str(binary), *arguments),
            cwd=repo_root,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PilotV24ParentImportError(
            "parent git identity check failed"
        ) from exc
    if completed.returncode != 0:
        raise PilotV24ParentImportError(
            "parent git identity check returned a failure"
        )
    try:
        return completed.stdout.decode("utf-8", "strict").strip()
    except UnicodeDecodeError as exc:
        raise PilotV24ParentImportError(
            "parent git identity is not UTF-8"
        ) from exc


def _repo_relative(repo_root: Path, path: Path, *, name: str) -> str:
    try:
        relative = path.absolute().relative_to(repo_root)
    except ValueError as exc:
        raise PilotV24ParentImportError(
            f"{name} must stay within the child repository"
        ) from exc
    if not relative.parts or relative.parts[0] != "experiment_results":
        raise PilotV24ParentImportError(
            f"{name} must stay below experiment_results/"
        )
    return PurePosixPath(*relative.parts).as_posix()


def _atomic_exact_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise PilotV24ParentImportError(
                f"immutable parent-import artifact differs on resume: {path}"
            )
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise PilotV24ParentImportError(
                f"concurrent parent-import artifact differs: {path}"
            )
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _atomic_exact_json(path: Path, value: Mapping[str, Any]) -> None:
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
    _atomic_exact_bytes(path, raw)


def _load_source_manifest(repo_root: Path) -> tuple[dict[str, Any], bytes]:
    _, raw = _guarded_file(
        repo_root,
        V24_PARENT_SOURCE_MANIFEST_PATH,
        name="V2.4 parent source manifest",
    )
    value = _strict_json(raw, name="V2.4 parent source manifest")
    _verify_self_hash(
        value,
        schema_version=V24_PARENT_SOURCE_MANIFEST_SCHEMA_VERSION,
        name="V2.4 parent source manifest",
    )
    if (
        value["integrity"]["content_sha256"]
        != V24_PARENT_SOURCE_MANIFEST_CONTENT_SHA256
    ):
        raise PilotV24ParentImportError(
            "V2.4 parent source manifest differs from the code-bound hash"
        )
    return value, raw


def _matrix_amendment(contract: PilotContract) -> Mapping[str, Any]:
    value = getattr(contract, "matrix_amendment", None)
    if not isinstance(value, Mapping):
        raise PilotV24ParentImportError(
            "V2.4 contract lacks its prospective matrix amendment"
        )
    return value


def _validate_child_contract(
    contract: PilotContract,
    *,
    source_manifest: Mapping[str, Any],
    require_frozen: bool = True,
) -> None:
    if (
        contract.contract_id != V24_CONTRACT_ID
        or (
            require_frozen
            and contract.status != "frozen"
        )
        or (
            not require_frozen
            and contract.status not in {"draft", "frozen"}
        )
        or contract.implementation.get("required_git_tag") != V24_SCIENCE_TAG
    ):
        raise PilotV24ParentImportError(
            "parent import requires the frozen V2.4 science contract"
        )
    amendment = _matrix_amendment(contract)
    serialized = json.dumps(
        _json_copy(amendment),
        sort_keys=True,
        allow_nan=False,
    )
    required_literals = (
        str(source_manifest["parent"]["contract_canonical_sha256"]),
        str(source_manifest["parent"]["science_tag"]),
        str(source_manifest["parent"]["science_commit"]),
        V24_PARENT_SOURCE_MANIFEST_PATH.as_posix(),
        V24_PARENT_SOURCE_MANIFEST_CONTENT_SHA256,
    )
    if any(value not in serialized for value in required_literals):
        raise PilotV24ParentImportError(
            "V2.4 matrix amendment does not bind the complete parent authority"
        )
    if "scientific_outcomes_observed_before_amendment" not in serialized:
        raise PilotV24ParentImportError(
            "V2.4 amendment lacks the outcome-blind declaration"
        )


def parent_budget_debit_for_v24(
    contract: PilotContract,
    *,
    repo_root: str | Path | None = None,
) -> ParentBudgetDebit | None:
    """Return the exact cumulative V2.3 debit for V2.4 only."""

    if contract.contract_id != V24_CONTRACT_ID:
        return None
    root = _real_root(
        repo_root or Path(__file__).resolve().parents[1],
        name="child repository root",
    )
    manifest, _ = _load_source_manifest(root)
    _validate_child_contract(
        contract,
        source_manifest=manifest,
        require_frozen=False,
    )
    try:
        debit = ParentBudgetDebit.from_dict(
            manifest["cumulative_budget_debit"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PilotV24ParentImportError(
            "V2.4 cumulative parent debit is malformed"
        ) from exc
    if debit.record_sha256 != V24_PARENT_DEBIT_RECORD_SHA256:
        raise PilotV24ParentImportError(
            "V2.4 cumulative parent debit record drifted"
        )
    return debit


def inherited_p95_receipt_path(
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


def inherited_projection_path(
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


def _snapshot_path(raw_root: Path, profile_id: str) -> Path:
    return (
        raw_root
        / "parent-import"
        / "parent_snapshots"
        / f"{profile_id}.observed_p95_parent.json"
    )


def _child_contract_binding(
    *,
    repo_root: Path,
    contract: PilotContract,
) -> dict[str, Any]:
    relative = PurePosixPath("experiments/pilot_v2_4.yaml")
    _, raw = _guarded_file(
        repo_root,
        relative,
        name="expanded V2.4 contract",
    )
    parsed = _strict_json(raw, name="expanded V2.4 contract")
    parsed_contract = PilotContract.from_dict(parsed)
    if (
        parsed_contract.contract_id != contract.contract_id
        or parsed_contract.canonical_hash != contract.canonical_hash
        or parsed_contract.to_dict() != contract.to_dict()
    ):
        raise PilotV24ParentImportError(
            "expanded V2.4 contract differs from the selected contract"
        )
    return {
        "path": relative.as_posix(),
        "file_sha256": _sha256(raw),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
    }


def _build_child_p95_receipt(
    *,
    repo_root: Path,
    raw_root: Path,
    contract: PilotContract,
    child_git_tag: str,
    child_git_commit: str,
    source_manifest: Mapping[str, Any],
    profile_id: str,
    parent_receipt: Mapping[str, Any],
    parent_snapshot_path: Path,
    parent_snapshot_raw: bytes,
) -> dict[str, Any]:
    if profile_id not in V24_ALLOWED_P95_PROFILES:
        raise PilotV24ParentImportError(
            "boundary-only parent model cannot create V2.4 dispatch authority"
        )
    source_row = source_manifest["observed_p95_sources"][profile_id]
    parent_model = parent_receipt.get("model")
    parent_reservations = parent_receipt.get("reservations")
    if (
        not isinstance(parent_model, Mapping)
        or parent_model.get("model_id") != profile_id
        or parent_model.get("runtime_model") != source_row["runtime_model"]
        or parent_model.get("served_model") != source_row["served_model"]
        or not isinstance(parent_reservations, Mapping)
        or set(parent_reservations) != {source_row["runtime_model"]}
    ):
        raise PilotV24ParentImportError(
            f"parent p95 model identity drifted for {profile_id}"
        )
    transformed = _json_copy(parent_reservations)
    for call_kind in ("action", "semantic"):
        entry = transformed[source_row["runtime_model"]].get(call_kind)
        if (
            not isinstance(entry, dict)
            or set(entry) != {"authority", "reservation"}
            or not isinstance(entry.get("authority"), dict)
        ):
            raise PilotV24ParentImportError(
                f"parent p95 reservation is malformed for {profile_id}/{call_kind}"
            )
        authority = entry["authority"]
        authority["pilot_contract_hash"] = contract.canonical_hash
        authority["pilot_tag"] = child_git_tag
    return _seal(
        {
            "schema_version": V24_INHERITED_P95_RECEIPT_SCHEMA_VERSION,
            "contract": _child_contract_binding(
                repo_root=repo_root,
                contract=contract,
            ),
            "git": {
                "tag": child_git_tag,
                "commit": child_git_commit,
            },
            "model": {
                "model_id": profile_id,
                "runtime_model": source_row["runtime_model"],
                "served_model": source_row["served_model"],
            },
            "parent_source": {
                "manifest_path": V24_PARENT_SOURCE_MANIFEST_PATH.as_posix(),
                "manifest_content_sha256": (
                    V24_PARENT_SOURCE_MANIFEST_CONTENT_SHA256
                ),
                "parent_contract_sha256": source_manifest["parent"][
                    "contract_canonical_sha256"
                ],
                "parent_git_tag": source_manifest["parent"]["science_tag"],
                "parent_git_commit": source_manifest["parent"][
                    "science_commit"
                ],
                "parent_receipt_source_path": source_row["path"],
                "parent_receipt_snapshot_path": _repo_relative(
                    repo_root,
                    parent_snapshot_path,
                    name="parent p95 snapshot",
                ),
                "parent_receipt_file_sha256": _sha256(parent_snapshot_raw),
                "parent_receipt_content_sha256": source_row[
                    "content_sha256"
                ],
            },
            "reservations": transformed,
            "scientific_evidence": False,
            "evidence_use": (
                "Prospective budget authority only. Parent capability and "
                "preflight outputs are not V2.4 treatment-effect evidence."
            ),
        }
    )


def _build_child_projection(
    *,
    contract: PilotContract,
    child_git_tag: str,
    child_git_commit: str,
    profile_id: str,
    child_receipt: Mapping[str, Any],
    child_receipt_path: Path,
) -> dict[str, Any]:
    if profile_id not in V24_ALLOWED_P95_PROFILES:
        raise PilotV24ParentImportError(
            "boundary-only parent model cannot create V2.4 dispatch authority"
        )
    model = child_receipt["model"]
    runtime_model = model["runtime_model"]
    entries = child_receipt["reservations"][runtime_model]
    projection = {
        f"{model['served_model']}::{call_kind}": _json_copy(
            entries[call_kind]["reservation"]
        )
        for call_kind in ("action", "semantic")
    }
    return _seal(
        {
            "schema_version": "finevo-pilot-projection-p95-v1",
            "model_id": profile_id,
            "served_model": model["served_model"],
            "projection": projection,
            "bindings": {
                "contract_sha256": contract.canonical_hash,
                "git_tag": child_git_tag,
                "git_commit": child_git_commit,
                "source_kind": "v2.3-verified-parent-import",
                "source_parent_manifest_content_sha256": (
                    V24_PARENT_SOURCE_MANIFEST_CONTENT_SHA256
                ),
                "source_authority_receipt": str(child_receipt_path),
                "source_authority_receipt_content_sha256": child_receipt[
                    "integrity"
                ]["content_sha256"],
            },
        }
    )


def _verify_parent_git(
    parent_root: Path,
    source_manifest: Mapping[str, Any],
) -> None:
    parent = source_manifest["parent"]
    tag_ref = f"refs/tags/{parent['science_tag']}"
    if _git(parent_root, "cat-file", "-t", tag_ref) != "tag":
        raise PilotV24ParentImportError(
            "V2.3 science tag is missing or is not annotated"
        )
    if (
        _git(parent_root, "rev-parse", tag_ref)
        != parent["science_tag_object"]
        or _git(parent_root, "rev-parse", f"{tag_ref}^{{commit}}")
        != parent["science_commit"]
    ):
        raise PilotV24ParentImportError(
            "V2.3 annotated tag object or peeled commit drifted"
        )


def _verify_parent_contract(
    parent_root: Path,
    source_manifest: Mapping[str, Any],
) -> PilotContract:
    parent = source_manifest["parent"]
    relative = _normalized_relative(
        parent["contract_path"],
        required_top="experiments",
        name="parent contract path",
    )
    path, raw = _guarded_file(
        parent_root,
        relative,
        name="parent contract",
    )
    if _sha256(raw) != parent["contract_file_sha256"]:
        raise PilotV24ParentImportError("parent contract file hash drifted")
    contract = load_pilot_contract(path)
    if (
        contract.contract_id != parent["contract_id"]
        or contract.canonical_hash != parent["contract_canonical_sha256"]
        or contract.status != "frozen"
        or contract.implementation.get("required_git_tag")
        != parent["science_tag"]
    ):
        raise PilotV24ParentImportError(
            "parent contract identity or frozen release binding drifted"
        )
    return contract


def _verify_parent_ledgers(
    parent_root: Path,
    *,
    parent_contract: PilotContract,
    source_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Lazy import avoids a module cycle: the orchestrator calls this function
    # only after PilotRunLedger and its contract helpers are fully defined.
    from .pilot_orchestrator import (
        PilotRunLedger,
        _budget_caps,
        _parent_budget_debit,
    )

    ledger_manifest = source_manifest["ledgers"]
    run_path, run_raw = _guarded_file(
        parent_root,
        _normalized_relative(
            ledger_manifest["run"]["path"],
            required_top="experiment_results",
            name="parent run ledger path",
        ),
        name="parent run ledger",
    )
    budget_path, budget_raw = _guarded_file(
        parent_root,
        _normalized_relative(
            ledger_manifest["budget"]["path"],
            required_top="experiment_results",
            name="parent budget ledger path",
        ),
        name="parent budget ledger",
    )
    if (
        _sha256(run_raw) != ledger_manifest["run"]["file_sha256"]
        or _sha256(budget_raw) != ledger_manifest["budget"]["file_sha256"]
    ):
        raise PilotV24ParentImportError("parent ledger file hash drifted")
    run_ledger = PilotRunLedger(
        run_path,
        contract_hash=parent_contract.canonical_hash,
        tamper_evident=True,
    )
    budget_ledger = PilotBudgetLedger(
        budget_path,
        contract_hash=parent_contract.canonical_hash,
        caps=_budget_caps(parent_contract),
        tamper_evident=True,
        parent_debit=_parent_budget_debit(parent_contract),
    )
    run_snapshot = run_ledger.snapshot()
    budget_snapshot = budget_ledger.snapshot()
    for manifest_row, snapshot in (
        (ledger_manifest["run"], run_snapshot),
        (ledger_manifest["budget"], budget_snapshot),
    ):
        events = snapshot["events"]
        event_chain_head = (
            snapshot.get("event_chain_head")
            if isinstance(snapshot, Mapping)
            else None
        )
        if event_chain_head is None:
            event_chain_head = events[-1]["event_sha256"]
        if (
            snapshot["ledger_sha256"] != manifest_row["internal_sha256"]
            or len(events) != manifest_row["event_count"]
            or event_chain_head != manifest_row["event_chain_head"]
        ):
            raise PilotV24ParentImportError(
                "parent ledger self-hash or event-chain binding drifted"
            )
    statuses = [
        str(row["status"]) for row in run_snapshot["runs"].values()
    ]
    counts = {
        status: statuses.count(status) for status in sorted(set(statuses))
    }
    denominator = source_manifest["terminal_denominator"]
    if (
        len(statuses) != denominator["registered_cells"]
        or counts != denominator["status_counts"]
        or any(
            status
            not in {
                "complete",
                "failed",
                "budget-stopped",
                "integrity-stopped",
                "capability-no-go",
            }
            for status in statuses
        )
    ):
        raise PilotV24ParentImportError(
            "parent terminal denominator or status counts drifted"
        )
    expected_debit = source_manifest["cumulative_budget_debit"]
    committed = budget_snapshot["committed"]
    if (
        not math.isclose(
            float(committed["cost_usd"]),
            float(expected_debit["cost_usd"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or committed["completions"]
        != expected_debit["hosted_completions"]
        or committed["storage_bytes"] != expected_debit["storage_bytes"]
    ):
        raise PilotV24ParentImportError(
            "parent cumulative budget debit differs from the frozen manifest"
        )
    return run_snapshot, budget_snapshot


def _verify_parent_bound_files(
    parent_root: Path,
    child_root: Path,
    *,
    source_manifest: Mapping[str, Any],
) -> None:
    release = source_manifest["release_attestation"]
    _, raw = _guarded_file(
        parent_root,
        _normalized_relative(
            release["path"],
            required_top="experiment_results",
            name="parent release attestation path",
        ),
        name="parent release attestation",
    )
    value = _strict_json(raw, name="parent release attestation")
    if (
        _sha256(raw) != release["file_sha256"]
        or value.get("attestation_sha256") != release["content_sha256"]
        or value.get("status") != "pass"
        or value.get("head_commit")
        != source_manifest["parent"]["science_commit"]
        or value.get("release_requirements", {}).get("tag")
        != source_manifest["parent"]["science_tag"]
    ):
        raise PilotV24ParentImportError(
            "parent scientific release attestation drifted"
        )
    for stage_id, row in source_manifest["stage_receipts"].items():
        _, stage_raw = _guarded_file(
            parent_root,
            _normalized_relative(
                row["path"],
                required_top="experiment_results",
                name=f"{stage_id} receipt path",
            ),
            name=f"{stage_id} receipt",
        )
        stage = _strict_json(stage_raw, name=f"{stage_id} receipt")
        if (
            _sha256(stage_raw) != row["file_sha256"]
            or stage.get("stage_id") != stage_id
            or stage.get("contract_sha256")
            != source_manifest["parent"]["contract_canonical_sha256"]
            or stage.get("integrity", {}).get("content_sha256")
            != row["content_sha256"]
        ):
            raise PilotV24ParentImportError(
                f"parent {stage_id} receipt drifted"
            )
    for name, row in source_manifest["published_evidence"].items():
        _, evidence_raw = _guarded_file(
            child_root,
            _normalized_relative(
                row["path"],
                required_top="evidence",
                name=f"published {name} path",
            ),
            name=f"published {name}",
        )
        if _sha256(evidence_raw) != row["file_sha256"]:
            raise PilotV24ParentImportError(
                f"published V2.3 {name} hash drifted"
            )
    package_row = source_manifest["published_evidence"]["package_manifest"]
    _, package_raw = _guarded_file(
        child_root,
        _normalized_relative(
            package_row["path"],
            required_top="evidence",
            name="published package manifest path",
        ),
        name="published package manifest",
    )
    package = _strict_json(package_raw, name="published package manifest")
    if (
        package.get("contract_sha256")
        != source_manifest["parent"]["contract_canonical_sha256"]
        or package.get("resolved_git_commit")
        != source_manifest["parent"]["science_commit"]
        or package.get("pilot_tag")
        != source_manifest["parent"]["science_tag"]
        or package.get("scientific_complete") is not False
        or package.get("scientific_matrix_complete") is not False
    ):
        raise PilotV24ParentImportError(
            "published V2.3 no-go package was reinterpreted or drifted"
        )


def persist_v24_parent_import(
    *,
    contract: PilotContract,
    repo_root: str | Path,
    raw_root: str | Path,
    parent_repo_root: str | Path,
    child_git_tag: str,
    child_git_commit: str,
) -> dict[str, Any]:
    """Validate the immutable parent and persist one idempotent child import."""

    child_root = _real_root(repo_root, name="child repository root")
    parent_root = _real_root(parent_repo_root, name="parent repository root")
    child_raw_root = Path(raw_root).absolute()
    _repo_relative(child_root, child_raw_root, name="V2.4 raw root")
    if (
        child_git_tag != V24_SCIENCE_TAG
        or _COMMIT_RE.fullmatch(child_git_commit) is None
    ):
        raise PilotV24ParentImportError(
            "child release tag or commit is malformed"
        )
    source_manifest, source_manifest_raw = _load_source_manifest(child_root)
    _validate_child_contract(contract, source_manifest=source_manifest)
    _verify_parent_git(parent_root, source_manifest)
    parent_contract = _verify_parent_contract(parent_root, source_manifest)
    run_snapshot, budget_snapshot = _verify_parent_ledgers(
        parent_root,
        parent_contract=parent_contract,
        source_manifest=source_manifest,
    )
    _verify_parent_bound_files(
        parent_root,
        child_root,
        source_manifest=source_manifest,
    )

    from .observed_p95_authority import (
        ObservedP95AuthorityError,
        verified_observed_p95_authority_binding,
    )

    imported_profiles: dict[str, Any] = {}
    boundary_profiles: dict[str, Any] = {}
    for profile_id, source_row in source_manifest[
        "observed_p95_sources"
    ].items():
        relative = _normalized_relative(
            source_row["path"],
            required_top="experiment_results",
            name=f"{profile_id} parent p95 receipt path",
        )
        _, parent_raw = _guarded_file(
            parent_root,
            relative,
            name=f"{profile_id} parent p95 receipt",
        )
        if _sha256(parent_raw) != source_row["file_sha256"]:
            raise PilotV24ParentImportError(
                f"{profile_id} parent p95 receipt file hash drifted"
            )
        try:
            binding = verified_observed_p95_authority_binding(
                relative.as_posix(),
                repo_root=parent_root,
                expected_git_commit=source_manifest["parent"][
                    "science_commit"
                ],
            )
        except ObservedP95AuthorityError as exc:
            raise PilotV24ParentImportError(
                f"{profile_id} parent p95 source chain failed verification: {exc}"
            ) from exc
        if (
            binding["receipt_file_sha256"] != source_row["file_sha256"]
            or binding["receipt_content_sha256"]
            != source_row["content_sha256"]
        ):
            raise PilotV24ParentImportError(
                f"{profile_id} verified p95 binding drifted"
            )
        snapshot_path = _snapshot_path(child_raw_root, profile_id)
        _atomic_exact_bytes(snapshot_path, parent_raw)
        snapshot = _strict_json(
            parent_raw,
            name=f"{profile_id} parent p95 snapshot",
        )
        source_binding = {
            "parent_source_path": source_row["path"],
            "snapshot_path": _repo_relative(
                child_root,
                snapshot_path,
                name=f"{profile_id} parent p95 snapshot",
            ),
            "file_sha256": source_row["file_sha256"],
            "content_sha256": source_row["content_sha256"],
            "runtime_model": source_row["runtime_model"],
            "served_model": source_row["served_model"],
        }
        if profile_id in V24_ALLOWED_P95_PROFILES:
            receipt = _build_child_p95_receipt(
                repo_root=child_root,
                raw_root=child_raw_root,
                contract=contract,
                child_git_tag=child_git_tag,
                child_git_commit=child_git_commit,
                source_manifest=source_manifest,
                profile_id=profile_id,
                parent_receipt=snapshot,
                parent_snapshot_path=snapshot_path,
                parent_snapshot_raw=parent_raw,
            )
            receipt_path = inherited_p95_receipt_path(
                child_raw_root,
                profile_id,
            )
            _atomic_exact_json(receipt_path, receipt)
            projection = _build_child_projection(
                contract=contract,
                child_git_tag=child_git_tag,
                child_git_commit=child_git_commit,
                profile_id=profile_id,
                child_receipt=receipt,
                child_receipt_path=receipt_path,
            )
            projection_path = inherited_projection_path(
                child_raw_root,
                profile_id,
            )
            _atomic_exact_json(projection_path, projection)
            imported_profiles[profile_id] = {
                **source_binding,
                "child_authority_receipt": _repo_relative(
                    child_root,
                    receipt_path,
                    name=f"{profile_id} child p95 authority",
                ),
                "child_authority_receipt_file_sha256": _sha256(
                    receipt_path.read_bytes()
                ),
                "child_authority_receipt_content_sha256": receipt[
                    "integrity"
                ]["content_sha256"],
                "child_projection": _repo_relative(
                    child_root,
                    projection_path,
                    name=f"{profile_id} child p95 projection",
                ),
                "child_projection_file_sha256": _sha256(
                    projection_path.read_bytes()
                ),
                "child_projection_content_sha256": projection[
                    "integrity"
                ]["content_sha256"],
            }
        else:
            boundary_profiles[profile_id] = {
                **source_binding,
                "dispatch_authority_created": False,
                "boundary_status": source_row["use"],
            }
    if set(imported_profiles) != set(V24_ALLOWED_P95_PROFILES):
        raise PilotV24ParentImportError(
            "V2.4 import did not create the exact two permitted p95 authorities"
        )
    if set(boundary_profiles) != set(V24_BOUNDARY_ONLY_P95_PROFILES):
        raise PilotV24ParentImportError(
            "V2.4 parent boundary p95 inventory drifted"
        )

    receipt = _seal(
        {
            "schema_version": V24_PARENT_IMPORT_SCHEMA_VERSION,
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "child_release": {
                "git_tag": child_git_tag,
                "git_commit": child_git_commit,
            },
            "parent_release": _json_copy(source_manifest["parent"]),
            "source_manifest": {
                "path": V24_PARENT_SOURCE_MANIFEST_PATH.as_posix(),
                "file_sha256": _sha256(source_manifest_raw),
                "content_sha256": (
                    V24_PARENT_SOURCE_MANIFEST_CONTENT_SHA256
                ),
            },
            "terminal_denominator": _json_copy(
                source_manifest["terminal_denominator"]
            ),
            "cumulative_budget_debit": _json_copy(
                source_manifest["cumulative_budget_debit"]
            ),
            "parent_run_ledger": {
                "ledger_sha256": run_snapshot["ledger_sha256"],
                "event_count": len(run_snapshot["events"]),
                "event_chain_head": run_snapshot["events"][-1][
                    "event_sha256"
                ],
            },
            "parent_budget_ledger": {
                "ledger_sha256": budget_snapshot["ledger_sha256"],
                "event_count": len(budget_snapshot["events"]),
                "event_chain_head": budget_snapshot["event_chain_head"],
                "committed": _json_copy(budget_snapshot["committed"]),
            },
            "imported_projection_profiles": imported_profiles,
            "boundary_only_profiles": boundary_profiles,
            "provider_calls": 0,
            "scientific_evidence": False,
            "scientific_outcomes_observed_before_amendment": False,
            "claim_boundary": (
                "This receipt imports only frozen budget and p95 authority. "
                "It contains no V2.4 treatment outcome and does not convert "
                "V2.3 capability/preflight rows into scientific evidence."
            ),
        }
    )
    receipt_path = child_raw_root / "parent-import" / "parent_import_receipt.json"
    _atomic_exact_json(receipt_path, receipt)
    verified = verify_v24_parent_import_receipt(
        receipt_path,
        repo_root=child_root,
        contract=contract,
        expected_git_commit=child_git_commit,
    )
    if verified != receipt:
        raise PilotV24ParentImportError(
            "persisted V2.4 parent import differs after verification"
        )
    return {
        "receipt": str(receipt_path),
        "receipt_content_sha256": receipt["integrity"]["content_sha256"],
        "provider_calls": 0,
        "scientific_evidence": False,
        "imported_profiles": sorted(imported_profiles),
        "boundary_only_profiles": sorted(boundary_profiles),
    }


def _load_child_contract_from_receipt(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
) -> PilotContract:
    binding = receipt.get("contract")
    if not isinstance(binding, Mapping):
        raise PilotV24ParentImportError(
            "inherited p95 receipt lacks a child contract binding"
        )
    relative = _normalized_relative(
        binding.get("path", ""),
        required_top="experiments",
        name="inherited p95 child contract path",
    )
    _, raw = _guarded_file(
        repo_root,
        relative,
        name="inherited p95 child contract",
    )
    if _sha256(raw) != binding.get("file_sha256"):
        raise PilotV24ParentImportError(
            "inherited p95 child contract file hash drifted"
        )
    value = _strict_json(raw, name="inherited p95 child contract")
    contract = PilotContract.from_dict(value)
    if (
        contract.contract_id != binding.get("contract_id")
        or contract.canonical_hash != binding.get("contract_sha256")
    ):
        raise PilotV24ParentImportError(
            "inherited p95 child contract identity drifted"
        )
    return contract


def verify_v24_inherited_p95_receipt(
    receipt: Mapping[str, Any],
    *,
    repo_root: str | Path,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Verify a child p95 receipt from tracked hashes and copied parent bytes."""

    root = _real_root(repo_root, name="child repository root")
    value = _json_copy(receipt)
    _verify_self_hash(
        value,
        schema_version=V24_INHERITED_P95_RECEIPT_SCHEMA_VERSION,
        name="V2.4 inherited p95 receipt",
    )
    contract = _load_child_contract_from_receipt(value, repo_root=root)
    manifest, _ = _load_source_manifest(root)
    _validate_child_contract(contract, source_manifest=manifest)
    git = value.get("git")
    model = value.get("model")
    parent_source = value.get("parent_source")
    if (
        not isinstance(git, Mapping)
        or git.get("tag") != V24_SCIENCE_TAG
        or git.get("commit") != expected_git_commit
        or _COMMIT_RE.fullmatch(expected_git_commit) is None
        or not isinstance(model, Mapping)
        or not isinstance(parent_source, Mapping)
    ):
        raise PilotV24ParentImportError(
            "V2.4 inherited p95 child release/model binding is malformed"
        )
    profile_id = str(model.get("model_id"))
    if profile_id not in V24_ALLOWED_P95_PROFILES:
        raise PilotV24ParentImportError(
            "boundary-only parent model cannot create V2.4 dispatch authority"
        )
    source_row = manifest["observed_p95_sources"][profile_id]
    snapshot_relative = _normalized_relative(
        parent_source.get("parent_receipt_snapshot_path", ""),
        required_top="experiment_results",
        name="parent p95 snapshot path",
    )
    snapshot_path, snapshot_raw = _guarded_file(
        root,
        snapshot_relative,
        name="parent p95 snapshot",
    )
    if (
        _sha256(snapshot_raw) != source_row["file_sha256"]
        or parent_source.get("parent_receipt_file_sha256")
        != source_row["file_sha256"]
        or parent_source.get("parent_receipt_content_sha256")
        != source_row["content_sha256"]
        or parent_source.get("manifest_content_sha256")
        != V24_PARENT_SOURCE_MANIFEST_CONTENT_SHA256
        or parent_source.get("parent_contract_sha256")
        != manifest["parent"]["contract_canonical_sha256"]
        or parent_source.get("parent_git_tag")
        != manifest["parent"]["science_tag"]
        or parent_source.get("parent_git_commit")
        != manifest["parent"]["science_commit"]
    ):
        raise PilotV24ParentImportError(
            "V2.4 inherited p95 parent snapshot binding drifted"
        )
    parent_receipt = _strict_json(
        snapshot_raw,
        name="parent p95 snapshot",
    )
    if (
        parent_receipt.get("integrity", {}).get("content_sha256")
        != source_row["content_sha256"]
    ):
        raise PilotV24ParentImportError(
            "V2.4 inherited p95 parent receipt content hash drifted"
        )
    expected = _build_child_p95_receipt(
        repo_root=root,
        raw_root=snapshot_path.parents[2],
        contract=contract,
        child_git_tag=V24_SCIENCE_TAG,
        child_git_commit=expected_git_commit,
        source_manifest=manifest,
        profile_id=profile_id,
        parent_receipt=parent_receipt,
        parent_snapshot_path=snapshot_path,
        parent_snapshot_raw=snapshot_raw,
    )
    if value != expected:
        raise PilotV24ParentImportError(
            "V2.4 inherited p95 receipt differs from its tracked parent source"
        )
    reservations = value.get("reservations")
    if not isinstance(reservations, dict):
        raise PilotV24ParentImportError(
            "V2.4 inherited p95 reservations are malformed"
        )
    return _json_copy(reservations)


def verify_v24_parent_import_receipt(
    receipt_or_path: Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path,
    contract: PilotContract,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Verify the persisted zero-call import and every child p95 artifact."""

    root = _real_root(repo_root, name="child repository root")
    if isinstance(receipt_or_path, Mapping):
        value = _json_copy(receipt_or_path)
    else:
        path = Path(receipt_or_path)
        if path.is_absolute():
            try:
                relative = PurePosixPath(
                    *path.absolute().relative_to(root).parts
                )
            except ValueError as exc:
                raise PilotV24ParentImportError(
                    "parent import receipt escaped the child repository"
                ) from exc
        else:
            relative = _normalized_relative(
                path,
                required_top="experiment_results",
                name="parent import receipt path",
            )
        _, raw = _guarded_file(
            root,
            relative,
            name="parent import receipt",
        )
        value = _strict_json(raw, name="parent import receipt")
    _verify_self_hash(
        value,
        schema_version=V24_PARENT_IMPORT_SCHEMA_VERSION,
        name="V2.4 parent import receipt",
    )
    manifest, manifest_raw = _load_source_manifest(root)
    _validate_child_contract(contract, source_manifest=manifest)
    if (
        value.get("contract_id") != contract.contract_id
        or value.get("contract_sha256") != contract.canonical_hash
        or value.get("child_release")
        != {
            "git_tag": V24_SCIENCE_TAG,
            "git_commit": expected_git_commit,
        }
        or value.get("provider_calls") != 0
        or value.get("scientific_evidence") is not False
        or value.get("scientific_outcomes_observed_before_amendment")
        is not False
        or value.get("terminal_denominator")
        != manifest["terminal_denominator"]
        or value.get("cumulative_budget_debit")
        != manifest["cumulative_budget_debit"]
        or value.get("source_manifest")
        != {
            "path": V24_PARENT_SOURCE_MANIFEST_PATH.as_posix(),
            "file_sha256": _sha256(manifest_raw),
            "content_sha256": V24_PARENT_SOURCE_MANIFEST_CONTENT_SHA256,
        }
    ):
        raise PilotV24ParentImportError(
            "V2.4 parent import contract/outcome/source binding drifted"
        )
    imported = value.get("imported_projection_profiles")
    boundary = value.get("boundary_only_profiles")
    if (
        not isinstance(imported, Mapping)
        or set(imported) != set(V24_ALLOWED_P95_PROFILES)
        or not isinstance(boundary, Mapping)
        or set(boundary) != set(V24_BOUNDARY_ONLY_P95_PROFILES)
    ):
        raise PilotV24ParentImportError(
            "V2.4 parent import profile inventory drifted"
        )
    for profile_id, row in imported.items():
        if not isinstance(row, Mapping):
            raise PilotV24ParentImportError(
                "V2.4 imported p95 profile row is malformed"
            )
        receipt_relative = _normalized_relative(
            row.get("child_authority_receipt", ""),
            required_top="experiment_results",
            name=f"{profile_id} child authority path",
        )
        _, receipt_raw = _guarded_file(
            root,
            receipt_relative,
            name=f"{profile_id} child authority",
        )
        child_receipt = _strict_json(
            receipt_raw,
            name=f"{profile_id} child authority",
        )
        verify_v24_inherited_p95_receipt(
            child_receipt,
            repo_root=root,
            expected_git_commit=expected_git_commit,
        )
        projection_relative = _normalized_relative(
            row.get("child_projection", ""),
            required_top="experiment_results",
            name=f"{profile_id} child projection path",
        )
        projection_path, projection_raw = _guarded_file(
            root,
            projection_relative,
            name=f"{profile_id} child projection",
        )
        projection = _strict_json(
            projection_raw,
            name=f"{profile_id} child projection",
        )
        expected_projection = _build_child_projection(
            contract=contract,
            child_git_tag=V24_SCIENCE_TAG,
            child_git_commit=expected_git_commit,
            profile_id=profile_id,
            child_receipt=child_receipt,
            child_receipt_path=Path(root).joinpath(*receipt_relative.parts),
        )
        if (
            projection != expected_projection
            or _sha256(receipt_raw)
            != row["child_authority_receipt_file_sha256"]
            or child_receipt["integrity"]["content_sha256"]
            != row["child_authority_receipt_content_sha256"]
            or _sha256(projection_raw)
            != row["child_projection_file_sha256"]
            or projection["integrity"]["content_sha256"]
            != row["child_projection_content_sha256"]
            or projection_path
            != Path(root).joinpath(*projection_relative.parts)
        ):
            raise PilotV24ParentImportError(
                f"V2.4 imported p95 artifacts drifted for {profile_id}"
            )
    return value


__all__ = [
    "PilotV24ParentImportError",
    "V24_ALLOWED_P95_PROFILES",
    "V24_BOUNDARY_ONLY_P95_PROFILES",
    "V24_CONTRACT_ID",
    "V24_INHERITED_P95_RECEIPT_SCHEMA_VERSION",
    "V24_PARENT_DEBIT_RECORD_SHA256",
    "V24_PARENT_IMPORT_SCHEMA_VERSION",
    "V24_PARENT_SOURCE_MANIFEST_CONTENT_SHA256",
    "V24_PARENT_SOURCE_MANIFEST_PATH",
    "V24_SCIENCE_TAG",
    "inherited_p95_receipt_path",
    "inherited_projection_path",
    "parent_budget_debit_for_v24",
    "persist_v24_parent_import",
    "verify_v24_inherited_p95_receipt",
    "verify_v24_parent_import_receipt",
]
