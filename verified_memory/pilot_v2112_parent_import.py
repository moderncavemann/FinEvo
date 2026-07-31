"""Strict zero-provider parent boundary for the FinEvo V2.11.2 amendment.

V2.11.1 is an immutable terminal no-go release.  Its parent and capability
imports completed, both paid long-context preflights consumed their complete
32-call denominators and then failed while sealing an invalidly checked active
semantic-rule state, and all 131 science cells were integrity-stopped.

This module preserves that history exactly while allowing V2.11.2 to import
only three outcome-blind prerequisites:

* the frozen q-ref / Stage-0 utility calibration wrapper; and
* two capability/interface wrappers whose 60 historical calls predate the
  failed V2.11.1 preflights.

The two failed provider journals are verified only as audit and cumulative
budget evidence.  Their samples, missing checkpoints, and any derived P95 are
never returned by the import API.  Provider construction is intentionally
absent from this module.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence

from .pilot_budget import ParentBudgetDebit
from .pilot_contract import (
    canonical_contract_sha256,
    canonical_sha256,
    load_pilot_contract,
)
from .pilot_v2111_parent_import import (
    V2111_CALIBRATION_WRAPPER_SCHEMA_VERSION,
    V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION,
    V2111_PARENT_IMPORT_SCHEMA_VERSION,
    calibration_wrapper_from_v2111_receipt,
    capability_wrappers_from_v2111_receipt,
    verify_v2111_parent_import_receipt,
)


V2112_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.11.2-parent-source-manifest-v1"
V2112_PARENT_IMPORT_SCHEMA_VERSION = "finevo-pilot-v2.11.2-parent-import-v1"
V2112_CALIBRATION_WRAPPER_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.2-imported-calibration-wrapper-v1"
)
V2112_CAPABILITY_WRAPPER_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.2-imported-capability-wrapper-v1"
)
V2112_VERIFIED_CAPABILITY_SOURCE_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.2-verified-capability-source-v1"
)
V2112_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_11_2_source_manifest.json"
)
V2112_SOURCE_MANIFEST_FILE_SHA256 = (
    "f38fb442b04ab9a0a85a246954b486f11e6c6571434336d3650b89833c70e90f"
)
V2112_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "4143655b99feba414c319b951578dd652d4bcae550391ea83acdd4d74c00c9d3"
)
V2112_DEFAULT_RECEIPT_PATH = PurePosixPath(
    "experiment_results/pilot-v2.11.2/raw/parent-import/parent_import_receipt.json"
)

V2111_CONTRACT_ID = "finevo-pilot-v2.11.1"
V2111_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_11_1.yaml")
V2111_CONTRACT_FILE_SHA256 = (
    "ff36745de8b0d348c84ae97945e6609a042348eefccdeaeb0e649d223c27519b"
)
V2111_CONTRACT_SHA256 = (
    "818607de5cd512cee60ece06c3f81612e6945cf7ff6d1e48ca643d2109cd7410"
)
V2111_SCIENCE_TAG = "pilot-v2.11.1-science"
V2111_SCIENCE_TAG_OBJECT = "c12f6bd5b74cb676109b83fcbfdb4376adf7abdf"
V2111_SCIENCE_COMMIT = "e9871353ad307fdd134f3c74764d201efbc81081"
V2111_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.11.1/raw")
V2111_TRACKED_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_11_1_source_manifest.json"
)
V2111_TRACKED_SOURCE_MANIFEST_FILE_SHA256 = (
    "78f7910ddbd5aa1207b869fc68d45650576e2e370af0f06cc53d0bc7226b71c5"
)
V2111_TRACKED_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "7cf33945cff145fa5ca4cf6aae521acec8c96bf521abac076982ffc2e88b7812"
)
V2111_RELEASE_ATTESTATION_PATH = V2111_RAW_ROOT / "release_attestation.json"
V2111_RELEASE_ATTESTATION_FILE_SHA256 = (
    "749b8632458ad25634342237e98c792e83a5b466907d520ce8056a49c339ec8c"
)
V2111_RELEASE_ATTESTATION_SHA256 = (
    "a6499b45005d213377d24580a5820a2194b22d89559b4f2a78e45780a1c35f6a"
)
V2111_SCIENTIFIC_LAUNCH_INPUT_PATH = V2111_RAW_ROOT / "scientific_launch_input.json"
V2111_SCIENTIFIC_LAUNCH_INPUT_FILE_SHA256 = (
    "1af12c5185e99cba2e8408bc991e0198508ae785eacf8451f84f38c32a457b10"
)
V2111_SCIENTIFIC_LAUNCH_INPUT_SHA256 = (
    "d2e4e03f0ca9254a8e2611535a6185e729280d1b1da97a986fe8226c4ff19816"
)
V2111_RUN_LEDGER_PATH = V2111_RAW_ROOT / "run_ledger.json"
V2111_RUN_LEDGER_FILE_SHA256 = (
    "ddd35892843acb4e770c9572d24e95537cb9dee83c044abc321a580f122f97b9"
)
V2111_RUN_LEDGER_SHA256 = (
    "ed9c0210791627128dc0e9942df2cd46269acbbd28a6c52af33454172a4b76c9"
)
V2111_RUN_LEDGER_EVENT_COUNT = 138
V2111_RUN_LEDGER_EVENT_HEAD = (
    "8ae162eb98bfd49ecc00964176800df3a15d0c88142f00c5d64cdf45cb343bec"
)
V2111_BUDGET_LEDGER_PATH = V2111_RAW_ROOT / "budget_ledger.json"
V2111_BUDGET_LEDGER_FILE_SHA256 = (
    "d487edda8555e9c66b1d11fdc938daac04c5a62e02afc45b9c3c18fb5947013c"
)
V2111_BUDGET_LEDGER_SHA256 = (
    "df9ccffbba39ac6d86375be433c4a34f4e3b51f60a4e0e10d42d31b7c2886330"
)
V2111_BUDGET_LEDGER_EVENT_COUNT = 12
V2111_BUDGET_LEDGER_EVENT_HEAD = (
    "d62d92f38c5924b820291d27f5a69b9081afad306f2c82b22dfd5a9d2a43ef49"
)

V2111_PARENT_IMPORT_RECEIPT_PATH = (
    V2111_RAW_ROOT / "parent-import/parent_import_receipt.json"
)
V2111_PARENT_IMPORT_RECEIPT_FILE_SHA256 = (
    "e5d2f79e9f5a5c960aa213e38a38b0f9be513f954703ddbb0515024149f04aa0"
)
V2111_PARENT_IMPORT_RECEIPT_CONTENT_SHA256 = (
    "6e574f20f32d589597dc14dadfdcff6343554ac92493c3bfa742a848eb34873a"
)
V2111_PARENT_IMPORT_STAGE_RECEIPT_PATH = (
    V2111_RAW_ROOT / "parent-import/stage_receipt.json"
)
V2111_PARENT_IMPORT_STAGE_RECEIPT_FILE_SHA256 = (
    "5e02f9d69d5602476497307324d538ab83cb783aef6692bbd246da6a27f1887d"
)
V2111_PARENT_IMPORT_STAGE_RECEIPT_CONTENT_SHA256 = (
    "948e9fc23c8a6eb307a3532f4ebf6b2d87aa8922e55fb6d93baaed161d0e084a"
)
V2111_CAPABILITY_STAGE_RECEIPT_PATH = (
    V2111_RAW_ROOT / "capability-gate/stage_receipt.json"
)
V2111_CAPABILITY_STAGE_RECEIPT_FILE_SHA256 = (
    "9231eeedd6274f7618436817303c33f81b4da3316da6926385e169aae959410b"
)
V2111_CAPABILITY_STAGE_RECEIPT_CONTENT_SHA256 = (
    "03bdf116fca10bffa4e4fd0e7b5f2fedee5bf1617e005cf9a24fc6d11a9a8487"
)
V2111_PREFLIGHT_STAGE_RECEIPT_PATH = (
    V2111_RAW_ROOT / "long-context-preflight/stage_receipt.json"
)
V2111_PREFLIGHT_STAGE_RECEIPT_FILE_SHA256 = (
    "88463a813129b93073403b84f1b239869a72b5395290218404e5c75c3167455f"
)
V2111_PREFLIGHT_STAGE_RECEIPT_CONTENT_SHA256 = (
    "b729d7b7ba702cd4dd088f7188e26becd3994c5ce589112572f044b3be86a97d"
)

V2111_CUMULATIVE_COST_USD = 18.586399812500005
V2111_CUMULATIVE_COMPLETIONS = 940
V2111_CUMULATIVE_STORAGE_BYTES = 217_838_625
V2111_PARENT_DEBIT_RECORD_SHA256 = (
    "678fc5b795e66f1aa358ea7941bebb9167097158f2aed8cee4044567109c5582"
)
V2111_ATTEMPT_COST_USD = 1.4198757499999999
V2111_ATTEMPT_COMPLETIONS = 64
V2111_ATTEMPT_STORAGE_BYTES = 257_490

V2111_EXPECTED_STAGE_STATUS_COUNTS: Mapping[str, Mapping[str, int]] = {
    "parent-import": {"complete": 1},
    "capability-gate": {"complete": 2},
    "long-context-preflight": {"failed": 2},
    "experiment-c": {"integrity-stopped": 25},
    "experiment-a": {"integrity-stopped": 20},
    "experiment-d": {"integrity-stopped": 55},
    "experiment-b": {"integrity-stopped": 25},
    "cross-model": {"integrity-stopped": 6},
}
V2111_EXPECTED_STATUS_COUNTS = {
    "complete": 3,
    "failed": 2,
    "integrity-stopped": 131,
}

_ZERO_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "cost_usd": 0.0,
}
_ZERO_PROVIDER_POLICY = {
    "provider_construction_during_import": False,
    "provider_calls_during_import": 0,
    "hosted_provider_calls_during_import": 0,
    "imported_effect_cells": 0,
    "effect_metrics_observed": False,
    "imported_preflight_samples": 0,
    "imported_checkpoint_artifacts": [],
    "imported_p95_authorities": [],
    "raw_tree_copied": False,
    "copied_file_count": 0,
    "copied_byte_count": 0,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _capability_run_id(model_id: str) -> str:
    return (
        "finevo-pilot-v2.11.1--capability-gate--"
        f"{model_id}--capability-probe--none--provider-preflight-default--"
        "s2010922376"
    )


def _preflight_run_id(model_id: str) -> str:
    return (
        "finevo-pilot-v2.11.1--long-context-preflight--"
        f"{model_id}--closed-loop-preflight--none--stage0-selected--"
        "s2010922376"
    )


_MODEL_SOURCES: Mapping[str, Mapping[str, Any]] = {
    "gpt52_main": {
        "runtime_model": "openai/gpt-5.2-2025-12-11",
        "served_model": "gpt-5.2-2025-12-11",
        "capability_wrapper_file_sha256": (
            "d486da4eefb47d57d40ccb1ea86c7354b745b4d823e795c4110f16d6b786c508"
        ),
        "capability_wrapper_content_sha256": (
            "08454e2b881f4e199aaabd9b0ea22b2695e4ecaf6befeffe81844cbcdc85afd2"
        ),
        "capability_gate_file_sha256": (
            "a7d9f05144a441408fef8b197051abd652d28d9a62f7f0c95605b63fbc1944f5"
        ),
        "capability_summary_file_sha256": (
            "34900ad5b1c8336fe7ffdf59352bf92fe010f8ae8f2a61a112a8a1d380f122f7"
        ),
        "bootstrap_file_sha256": (
            "b3aadf4bf11afbf8a32ffd379f6e453e4ca4027b90801abcc3f26aa8541167fb"
        ),
        "provider_catalog_file_sha256": (
            "23ee07202584d8c0312209e03c092481bfa35bad892c91f70b419a1adcb53309"
        ),
        "run_intent_file_sha256": (
            "95b568faee38e509245f239ce18d08f7169ae1b2e6b7f9f0c15e891a1b7c5e66"
        ),
        "failure_file_sha256": (
            "540e14bc290d7b49dd361277975939bf00b502a94fd693cf626e3d314c988e6c"
        ),
        "failure_manifest_file_sha256": (
            "6052cd2875eed9fe4dae358e2fd1f95ca3198b3f8318d784175faa0aa59b86fe"
        ),
        "journal_file_sha256": (
            "5cf6b541c6137422b39ad6bcd905deb0294a0d79322aa6c9ead2b91ecb26b19d"
        ),
        "journal_sha256": (
            "156ec20e5f7539dec6952d97a78a1dfd87b0a14719b5ebeb2f2f547a78f8b44d"
        ),
        "journal_event_head": (
            "b9cf69f4c1634810ab1768b24d04f3dc2cef951a6ea1e96fcf16c7fc1598822e"
        ),
        "failure_message": (
            "active rule family-e42ecae1edcdf27de75b:v1 does not satisfy "
            "activation invariants"
        ),
        "actual": {
            "completions": 32,
            "cost_usd": 0.62165075,
            "storage_bytes": 40_621,
        },
    },
    "gpt56_diagnostic": {
        "runtime_model": "openai/gpt-5.6-sol",
        "served_model": "gpt-5.6-sol",
        "capability_wrapper_file_sha256": (
            "351126acf5291324de19433549ccb6f90679f3474b2d4aa12f3823b7412e920c"
        ),
        "capability_wrapper_content_sha256": (
            "aa2f881f9ad6b1c38f33dacdff4c63813329ae9ec738d3652f0d27a7bf60ecb9"
        ),
        "capability_gate_file_sha256": (
            "8b6bb0d53e7ee24a04c5ea885a84c4050c5a72288705a4a13e4e55e58d3e348a"
        ),
        "capability_summary_file_sha256": (
            "c04a8c2d1608164b139532e9d9fd85a57096356fda8186e356ee2d1f74b2ef25"
        ),
        "bootstrap_file_sha256": (
            "344d27820ed609db6c1c22adb2b86b61f5d84da450edef9a3c04636b043bd1d5"
        ),
        "provider_catalog_file_sha256": (
            "445cf11688f9b862f9795940c238d4621f83342458a12f009bacfd689060c775"
        ),
        "run_intent_file_sha256": (
            "f7fcf7e55580b4c4c55222d6056875e0d71a14a6a400c2c6572aa06772308f90"
        ),
        "failure_file_sha256": (
            "c8d4a5e506cf2a0ae67e9db4696014e081df5c1895bd4d5d4b5e81ee6fd1dbca"
        ),
        "failure_manifest_file_sha256": (
            "de9e6fa2cd626c4d44b77c904b190b43fc482d5928b03d0fa07c56b49d3bb310"
        ),
        "journal_file_sha256": (
            "128126b3ab73bc38c4711303cbc2205ce35feaa520923d4b6842a59f7ec6430d"
        ),
        "journal_sha256": (
            "05bca881ad7cbbce1546926488d29552c2eb2da6584f0ed54c8b8717b96f754d"
        ),
        "journal_event_head": (
            "4a46c6ed0922528cf72dffb45cd5928dc0232bb0987185f7be36c93293c9081f"
        ),
        "failure_message": (
            "active rule family-757c248bc4729875ca6b:v1 does not satisfy "
            "activation invariants"
        ),
        "actual": {
            "completions": 32,
            "cost_usd": 0.7982249999999999,
            "storage_bytes": 40_440,
        },
    },
}


class PilotV2112ParentImportError(RuntimeError):
    """Raised before any V2.11.1 parent evidence may cross the boundary."""


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except Exception as exc:
        raise PilotV2112ParentImportError(
            "value is not canonical-JSON compatible"
        ) from exc


def _strict_json(raw: bytes, *, name: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PilotV2112ParentImportError(
                    f"{name} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except PilotV2112ParentImportError:
        raise
    except Exception as exc:
        raise PilotV2112ParentImportError(f"{name} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise PilotV2112ParentImportError(f"{name} must be a JSON object")
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    candidate = _json_copy(dict(value))
    if "integrity" in candidate:
        raise PilotV2112ParentImportError("cannot seal a pre-sealed value")
    candidate["integrity"] = {"canonicalization": "json-sort-keys-utf8-v1"}
    candidate["integrity"]["content_sha256"] = canonical_sha256(candidate)
    return candidate


def _verify_seal(
    value: Mapping[str, Any],
    *,
    schema_version: str,
    name: str,
    integrity_in_hash: bool = True,
) -> None:
    candidate = _json_copy(dict(value))
    integrity = candidate.get("integrity")
    if isinstance(integrity, dict):
        claimed = integrity.pop("content_sha256", None)
        if not integrity_in_hash:
            candidate.pop("integrity", None)
    else:
        claimed = None
    if (
        value.get("schema_version") != schema_version
        or not isinstance(value.get("integrity"), Mapping)
        or set(value.get("integrity", {})) != {"canonicalization", "content_sha256"}
        or integrity.get("canonicalization") != "json-sort-keys-utf8-v1"
        or claimed != canonical_sha256(candidate)
    ):
        raise PilotV2112ParentImportError(f"{name} schema or content hash mismatch")


def _verify_field_hash(value: Mapping[str, Any], *, field: str, name: str) -> None:
    candidate = _json_copy(dict(value))
    claimed = candidate.pop(field, None)
    if not isinstance(claimed, str) or claimed != canonical_sha256(candidate):
        raise PilotV2112ParentImportError(f"{name} self-hash mismatch")


def _strict_root(value: str | Path, *, name: str) -> Path:
    path = Path(value).expanduser().absolute()
    for component in (path, *path.parents):
        try:
            if component.is_symlink():
                raise PilotV2112ParentImportError(f"{name} path contains a symlink")
        except OSError as exc:
            raise PilotV2112ParentImportError(f"{name} is unavailable") from exc
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PilotV2112ParentImportError(f"{name} is unavailable") from exc
    if not resolved.is_dir():
        raise PilotV2112ParentImportError(f"{name} must be a directory")
    return resolved


def _normalized_relative(value: Any, *, required_top: str, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PilotV2112ParentImportError(f"{name} path is malformed")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.parts[0] != required_top
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise PilotV2112ParentImportError(f"{name} path escaped its namespace")
    return relative


def _guarded_file(
    root: Path,
    relative: PurePosixPath,
    *,
    name: str,
) -> tuple[Path, bytes]:
    path = root.joinpath(*relative.parts)
    current = root
    try:
        for part in relative.parts:
            current = current / part
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise PilotV2112ParentImportError(f"{name} path contains a symlink")
        if not stat.S_ISREG(path.lstat().st_mode):
            raise PilotV2112ParentImportError(f"{name} must be a regular file")
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb", closefd=True) as handle:
            raw = handle.read()
    except PilotV2112ParentImportError:
        raise
    except (OSError, ValueError) as exc:
        raise PilotV2112ParentImportError(f"{name} is unavailable") from exc
    return path, raw


def _read_json(
    root: Path,
    relative: PurePosixPath,
    *,
    name: str,
) -> tuple[bytes, dict[str, Any]]:
    _, raw = _guarded_file(root, relative, name=name)
    return raw, _strict_json(raw, name=name)


def _binding(
    relative: PurePosixPath,
    raw: bytes,
    value: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": relative.as_posix(),
        "byte_size": len(raw),
        "file_sha256": _sha256(raw),
    }
    if isinstance(value, Mapping):
        integrity = value.get("integrity")
        if isinstance(integrity, Mapping) and isinstance(
            integrity.get("content_sha256"), str
        ):
            result["content_sha256"] = integrity["content_sha256"]
        else:
            for field in (
                "attestation_sha256",
                "launch_input_sha256",
                "ledger_sha256",
                "receipt_sha256",
                "manifest_sha256",
                "intent_sha256",
                "journal_sha256",
            ):
                if isinstance(value.get(field), str):
                    result["content_sha256"] = value[field]
                    break
    return result


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotV2112ParentImportError(
            "V2.11.1 git release identity is unavailable"
        ) from exc
    return result.stdout.strip()


def _verify_git_release(root: Path) -> dict[str, str]:
    head = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    tag_object = _git(
        root,
        "rev-parse",
        "--verify",
        f"refs/tags/{V2111_SCIENCE_TAG}^{{tag}}",
    )
    tag_commit = _git(
        root,
        "rev-parse",
        "--verify",
        f"refs/tags/{V2111_SCIENCE_TAG}^{{commit}}",
    )
    tracked = _git(root, "status", "--porcelain=v1", "--untracked-files=no")
    if (
        head != V2111_SCIENCE_COMMIT
        or tag_commit != V2111_SCIENCE_COMMIT
        or tag_object != V2111_SCIENCE_TAG_OBJECT
        or tracked
    ):
        raise PilotV2112ParentImportError(
            "V2.11.1 annotated tag, commit, or tracked worktree drifted"
        )
    return {
        "science_tag": V2111_SCIENCE_TAG,
        "science_tag_object": tag_object,
        "resolved_git_commit": tag_commit,
    }


def _verify_event_ledger(
    value: Mapping[str, Any],
    *,
    schema_version: str,
    internal_sha256: str,
    event_count: int,
    event_head: str,
    run_count: int,
    name: str,
) -> None:
    candidate = _json_copy(dict(value))
    claimed = candidate.pop("ledger_sha256", None)
    events = value.get("events")
    runs = value.get("runs")
    if (
        value.get("schema_version") != schema_version
        or value.get("contract_hash") != V2111_CONTRACT_SHA256
        or claimed != internal_sha256
        or canonical_sha256(candidate) != internal_sha256
        or not isinstance(events, list)
        or len(events) != event_count
        or not isinstance(runs, Mapping)
        or len(runs) != run_count
        or not events
        or events[-1].get("event_sha256") != event_head
    ):
        raise PilotV2112ParentImportError(f"{name} identity drifted")
    previous = "0" * 64
    for index, source in enumerate(events):
        if not isinstance(source, Mapping):
            raise PilotV2112ParentImportError(f"{name} event is malformed")
        row = _json_copy(dict(source))
        digest = row.pop("event_sha256", None)
        if (
            source.get("event_index") != index
            or source.get("previous_event_sha256") != previous
            or digest != canonical_sha256(row)
        ):
            raise PilotV2112ParentImportError(f"{name} event chain drifted")
        previous = str(digest)


def _verify_expected_absent(root: Path, relative: PurePosixPath) -> None:
    path = root.joinpath(*relative.parts)
    current = root
    try:
        for part in relative.parts[:-1]:
            current = current / part
            if current.exists() and current.is_symlink():
                raise PilotV2112ParentImportError(
                    f"expected-absent parent path contains symlink: {relative}"
                )
        if path.exists() or path.is_symlink():
            raise PilotV2112ParentImportError(
                f"forbidden V2.11.1 authority artifact exists: {relative}"
            )
    except PilotV2112ParentImportError:
        raise
    except OSError as exc:
        raise PilotV2112ParentImportError(
            f"cannot verify expected absence: {relative}"
        ) from exc


def _parent_budget_debit() -> ParentBudgetDebit:
    return ParentBudgetDebit(
        parent_contract_sha256=V2111_CONTRACT_SHA256,
        parent_run_ledger_sha256=V2111_RUN_LEDGER_SHA256,
        parent_budget_ledger_sha256=V2111_BUDGET_LEDGER_SHA256,
        stage_bucket="parent_v2111",
        cost_usd=V2111_CUMULATIVE_COST_USD,
        hosted_completions=V2111_CUMULATIVE_COMPLETIONS,
        storage_bytes=V2111_CUMULATIVE_STORAGE_BYTES,
        record_sha256=V2111_PARENT_DEBIT_RECORD_SHA256,
    )


def _capability_paths(model_id: str) -> dict[str, PurePosixPath]:
    run_id = _capability_run_id(model_id)
    root = V2111_RAW_ROOT / f"capability-gate/runs/{run_id}"
    return {
        "wrapper": root / "capability.json",
        "gate_receipt": root / "gate_receipt.json",
        "bootstrap": root / "v2111_contract_envelope_bootstrap.json",
        "summary": V2111_RAW_ROOT / f"capability-gate/summaries/{run_id}.json",
    }


def _preflight_paths(model_id: str) -> dict[str, PurePosixPath]:
    run_id = _preflight_run_id(model_id)
    run_root = V2111_RAW_ROOT / f"long-context-preflight/runs/{run_id}"
    return {
        "provider_catalog": (
            V2111_RAW_ROOT / f"long-context-preflight/provider_catalog/{model_id}.json"
        ),
        "journal": V2111_RAW_ROOT
        / (
            "long-context-preflight/provider_call_journals/" f"{run_id}--preflight.json"
        ),
        "run_intent": run_root / "preflight_checkpoint.json.run-intent.json",
        "failure": run_root / "failure_receipt/failure.json",
        "failure_manifest": run_root / "failure_receipt/failure_manifest.json",
    }


def _expected_absent_paths() -> tuple[PurePosixPath, ...]:
    rows: list[PurePosixPath] = [
        V2111_RAW_ROOT / "long-context-preflight/post_gate_authority.json",
    ]
    for model_id in sorted(_MODEL_SOURCES):
        preflight_root = (
            V2111_RAW_ROOT
            / f"long-context-preflight/runs/{_preflight_run_id(model_id)}"
        )
        capability_root = (
            V2111_RAW_ROOT / f"capability-gate/runs/{_capability_run_id(model_id)}"
        )
        rows.extend(
            [
                preflight_root / "preflight_checkpoint.json",
                preflight_root / "preflight_checkpoint_exactness.json",
                preflight_root / "projection_p95.json",
                capability_root / "projection_p95.json",
            ]
        )
    return tuple(rows)


def _verify_contract(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], tuple[Any, ...]]:
    raw, value = _read_json(root, V2111_CONTRACT_PATH, name="V2.11.1 contract")
    if _sha256(raw) != V2111_CONTRACT_FILE_SHA256:
        raise PilotV2112ParentImportError("V2.11.1 contract file hash drifted")
    if (
        value.get("contract_id") != V2111_CONTRACT_ID
        or value.get("status") != "frozen"
        or value.get("implementation", {}).get("required_git_tag") != V2111_SCIENCE_TAG
        or value.get("integrity", {}).get("declared_sha256") != V2111_CONTRACT_SHA256
        or canonical_contract_sha256(value) != V2111_CONTRACT_SHA256
    ):
        raise PilotV2112ParentImportError("V2.11.1 contract identity drifted")
    try:
        contract = load_pilot_contract(root.joinpath(*V2111_CONTRACT_PATH.parts))
        specs = tuple(contract.expand())
    except Exception as exc:
        raise PilotV2112ParentImportError("V2.11.1 contract expansion failed") from exc
    stage_counts = Counter(spec.stage_id for spec in specs)
    if (
        contract.canonical_hash != V2111_CONTRACT_SHA256
        or len(specs) != 136
        or dict(stage_counts)
        != {
            stage_id: sum(counts.values())
            for stage_id, counts in V2111_EXPECTED_STAGE_STATUS_COUNTS.items()
        }
    ):
        raise PilotV2112ParentImportError("V2.11.1 denominator drifted")
    return value, _binding(V2111_CONTRACT_PATH, raw, value), specs


def _verify_tracked_v2111_source_manifest(root: Path) -> dict[str, Any]:
    raw, value = _read_json(
        root,
        V2111_TRACKED_SOURCE_MANIFEST_PATH,
        name="V2.11.1 tracked source manifest",
    )
    _verify_seal(
        value,
        schema_version="finevo-pilot-v2.11.1-parent-source-manifest-v1",
        name="V2.11.1 tracked source manifest",
    )
    if (
        _sha256(raw) != V2111_TRACKED_SOURCE_MANIFEST_FILE_SHA256
        or value.get("integrity", {}).get("content_sha256")
        != V2111_TRACKED_SOURCE_MANIFEST_CONTENT_SHA256
    ):
        raise PilotV2112ParentImportError(
            "V2.11.1 tracked source manifest hash drifted"
        )
    return _binding(V2111_TRACKED_SOURCE_MANIFEST_PATH, raw, value)


def _verify_attestation(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, value = _read_json(
        root,
        V2111_RELEASE_ATTESTATION_PATH,
        name="V2.11.1 release attestation",
    )
    candidate = _json_copy(value)
    claimed = candidate.pop("attestation_sha256", None)
    expected_tag = {
        "kind": "annotated",
        "name": V2111_SCIENCE_TAG,
        "object_id": V2111_SCIENCE_TAG_OBJECT,
        "peeled_commit": V2111_SCIENCE_COMMIT,
    }
    if (
        _sha256(raw) != V2111_RELEASE_ATTESTATION_FILE_SHA256
        or value.get("schema_version") != "finevo-scientific-release-attestation-v2"
        or claimed != V2111_RELEASE_ATTESTATION_SHA256
        or canonical_sha256(candidate) != V2111_RELEASE_ATTESTATION_SHA256
        or value.get("status") != "pass"
        or value.get("head_commit") != V2111_SCIENCE_COMMIT
        or value.get("local_tag") != expected_tag
        or value.get("remote", {}).get("tag_object_id") != V2111_SCIENCE_TAG_OBJECT
        or value.get("remote", {}).get("tag_peeled_commit") != V2111_SCIENCE_COMMIT
        or value.get("remote", {}).get("branch_commit") != V2111_SCIENCE_COMMIT
        or value.get("contract", {}).get("canonical_sha256") != V2111_CONTRACT_SHA256
        or value.get("contract", {}).get("file_sha256") != V2111_CONTRACT_FILE_SHA256
        or value.get("github_actions", {}).get("ci_measurements", {}).get("test_count")
        != 1446
    ):
        raise PilotV2112ParentImportError("V2.11.1 release attestation drifted")
    return value, _binding(V2111_RELEASE_ATTESTATION_PATH, raw, value)


def _verify_launch_input(root: Path) -> dict[str, Any]:
    raw, value = _read_json(
        root,
        V2111_SCIENTIFIC_LAUNCH_INPUT_PATH,
        name="V2.11.1 scientific launch input",
    )
    _verify_field_hash(
        value,
        field="launch_input_sha256",
        name="V2.11.1 scientific launch input",
    )
    if (
        _sha256(raw) != V2111_SCIENTIFIC_LAUNCH_INPUT_FILE_SHA256
        or value.get("schema_version") != "finevo-scientific-launch-input-v1"
        or value.get("launch_input_sha256") != V2111_SCIENTIFIC_LAUNCH_INPUT_SHA256
        or value.get("contract_sha256") != V2111_CONTRACT_SHA256
        or value.get("contract_binding", {}).get("contract_file_sha256")
        != V2111_CONTRACT_FILE_SHA256
        or value.get("ci_run_selection", {}).get("run_id") != 30622571424
    ):
        raise PilotV2112ParentImportError("V2.11.1 scientific launch input drifted")
    return _binding(V2111_SCIENTIFIC_LAUNCH_INPUT_PATH, raw, value)


def _verify_run_ledger(
    root: Path,
    specs: Sequence[Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, value = _read_json(root, V2111_RUN_LEDGER_PATH, name="V2.11.1 run ledger")
    if _sha256(raw) != V2111_RUN_LEDGER_FILE_SHA256:
        raise PilotV2112ParentImportError("V2.11.1 run ledger file hash drifted")
    _verify_event_ledger(
        value,
        schema_version="finevo-pilot-run-ledger-v2",
        internal_sha256=V2111_RUN_LEDGER_SHA256,
        event_count=V2111_RUN_LEDGER_EVENT_COUNT,
        event_head=V2111_RUN_LEDGER_EVENT_HEAD,
        run_count=136,
        name="V2.11.1 run ledger",
    )
    expected_specs = {spec.run_id: spec.to_dict() for spec in specs}
    runs = value["runs"]
    if set(runs) != set(expected_specs):
        raise PilotV2112ParentImportError(
            "V2.11.1 run ledger differs from contract denominator"
        )
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    status_counts: Counter[str] = Counter()
    for run_id, row in runs.items():
        if not isinstance(row, Mapping) or row.get("spec") != expected_specs[run_id]:
            raise PilotV2112ParentImportError("V2.11.1 run spec binding drifted")
        status = str(row.get("status"))
        status_counts[status] += 1
        by_stage[expected_specs[run_id]["stage_id"]][status] += 1
    if dict(status_counts) != V2111_EXPECTED_STATUS_COUNTS or {
        stage_id: dict(counts) for stage_id, counts in by_stage.items()
    } != {
        stage_id: dict(counts)
        for stage_id, counts in V2111_EXPECTED_STAGE_STATUS_COUNTS.items()
    }:
        raise PilotV2112ParentImportError("V2.11.1 terminal status denominator drifted")
    return value, _binding(V2111_RUN_LEDGER_PATH, raw, value)


def _expected_budget_actuals() -> dict[str, dict[str, int | float]]:
    result: dict[str, dict[str, int | float]] = {
        (
            "finevo-pilot-v2.11.1--parent-import--qref_scripted--parent-import--"
            "none--provider-preflight-default--s2010922376"
        ): {"completions": 0, "cost_usd": 0.0, "storage_bytes": 73_620},
        _capability_run_id("gpt52_main"): {
            "completions": 0,
            "cost_usd": 0.0,
            "storage_bytes": 51_653,
        },
        _capability_run_id("gpt56_diagnostic"): {
            "completions": 0,
            "cost_usd": 0.0,
            "storage_bytes": 51_156,
        },
    }
    for model_id, source in _MODEL_SOURCES.items():
        result[_preflight_run_id(model_id)] = dict(source["actual"])
    return result


def _verify_budget_ledger(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, value = _read_json(
        root,
        V2111_BUDGET_LEDGER_PATH,
        name="V2.11.1 budget ledger",
    )
    if _sha256(raw) != V2111_BUDGET_LEDGER_FILE_SHA256:
        raise PilotV2112ParentImportError("V2.11.1 budget ledger file hash drifted")
    _verify_event_ledger(
        value,
        schema_version="finevo-pilot-budget-ledger-v2",
        internal_sha256=V2111_BUDGET_LEDGER_SHA256,
        event_count=V2111_BUDGET_LEDGER_EVENT_COUNT,
        event_head=V2111_BUDGET_LEDGER_EVENT_HEAD,
        run_count=5,
        name="V2.11.1 budget ledger",
    )
    expected = _expected_budget_actuals()
    runs = value["runs"]
    if set(runs) != set(expected):
        raise PilotV2112ParentImportError("V2.11.1 budget run denominator drifted")
    for run_id, actual in expected.items():
        if runs[run_id].get("actual") != actual:
            raise PilotV2112ParentImportError(
                f"V2.11.1 budget actual drifted for {run_id}"
            )
    try:
        inherited = ParentBudgetDebit.from_dict(value["parent_debit"])
    except Exception as exc:
        raise PilotV2112ParentImportError(
            "V2.11.1 inherited parent debit is malformed"
        ) from exc
    current_cost = math.fsum(float(row["actual"]["cost_usd"]) for row in runs.values())
    current_calls = sum(int(row["actual"]["completions"]) for row in runs.values())
    current_storage = sum(int(row["actual"]["storage_bytes"]) for row in runs.values())
    if (
        not math.isclose(
            current_cost,
            V2111_ATTEMPT_COST_USD,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or current_calls != V2111_ATTEMPT_COMPLETIONS
        or current_storage != V2111_ATTEMPT_STORAGE_BYTES
        or not math.isclose(
            inherited.cost_usd + current_cost,
            V2111_CUMULATIVE_COST_USD,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or inherited.hosted_completions + current_calls != V2111_CUMULATIVE_COMPLETIONS
        or inherited.storage_bytes + current_storage != V2111_CUMULATIVE_STORAGE_BYTES
    ):
        raise PilotV2112ParentImportError(
            "V2.11.1 attempt or cumulative budget debit drifted"
        )
    _parent_budget_debit()
    return value, _binding(V2111_BUDGET_LEDGER_PATH, raw, value)


def _verify_stage_receipt(
    root: Path,
    *,
    path: PurePosixPath,
    file_sha256: str,
    content_sha256: str,
    stage_id: str,
    registered: int,
    status: str,
    status_counts: Mapping[str, int],
    go: bool,
    go_models: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, value = _read_json(root, path, name=f"V2.11.1 {stage_id} stage receipt")
    _verify_seal(
        value,
        schema_version="finevo-pilot-stage-receipt-v2",
        name=f"V2.11.1 {stage_id} stage receipt",
        integrity_in_hash=False,
    )
    if (
        _sha256(raw) != file_sha256
        or value.get("integrity", {}).get("content_sha256") != content_sha256
        or value.get("contract_id") != V2111_CONTRACT_ID
        or value.get("contract_sha256") != V2111_CONTRACT_SHA256
        or value.get("stage_id") != stage_id
        or value.get("status") != status
        or value.get("terminal") is not True
        or value.get("denominator_terminal") is not True
        or value.get("registered_run_count") != registered
        or value.get("status_counts") != dict(status_counts)
        or value.get("go") is not go
        or value.get("go_models") != list(go_models)
        or value.get("execution_progression_go") is not go
    ):
        raise PilotV2112ParentImportError(f"V2.11.1 {stage_id} stage receipt drifted")
    return value, _binding(path, raw, value)


def _verify_existing_parent_import(
    root: Path,
    *,
    grandparent_science_root: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, observed = _read_json(
        root,
        V2111_PARENT_IMPORT_RECEIPT_PATH,
        name="V2.11.1 parent import receipt",
    )
    if (
        _sha256(raw) != V2111_PARENT_IMPORT_RECEIPT_FILE_SHA256
        or observed.get("integrity", {}).get("content_sha256")
        != V2111_PARENT_IMPORT_RECEIPT_CONTENT_SHA256
    ):
        raise PilotV2112ParentImportError("V2.11.1 parent import receipt hash drifted")
    try:
        verified = verify_v2111_parent_import_receipt(
            observed,
            repo_root=root,
            child_contract_sha256=V2111_CONTRACT_SHA256,
            child_git_tag=V2111_SCIENCE_TAG,
            child_git_commit=V2111_SCIENCE_COMMIT,
            parent_science_root=grandparent_science_root,
        )
    except Exception as exc:
        raise PilotV2112ParentImportError(
            "V2.11.1 parent import receipt failed exact source replay"
        ) from exc
    if (
        verified != observed
        or observed.get("schema_version") != V2111_PARENT_IMPORT_SCHEMA_VERSION
        or observed.get("scientific_evidence") is not False
        or observed.get("import_policy", {}).get("provider_calls_during_import") != 0
        or observed.get("import_policy", {}).get("provider_construction_during_import")
        is not False
        or observed.get("import_policy", {}).get("imported_p95_authorities") != []
    ):
        raise PilotV2112ParentImportError(
            "V2.11.1 parent import receipt semantics drifted"
        )
    return observed, _binding(V2111_PARENT_IMPORT_RECEIPT_PATH, raw, observed)


def _verify_capability_model(
    root: Path,
    *,
    model_id: str,
    receipt_wrapper: Mapping[str, Any],
    source_spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _MODEL_SOURCES[model_id]
    paths = _capability_paths(model_id)
    wrapper_raw, wrapper = _read_json(
        root,
        paths["wrapper"],
        name=f"V2.11.1 {model_id} capability wrapper",
    )
    _verify_seal(
        wrapper,
        schema_version=V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION,
        name=f"V2.11.1 {model_id} capability wrapper",
    )
    if (
        _sha256(wrapper_raw) != source["capability_wrapper_file_sha256"]
        or wrapper.get("integrity", {}).get("content_sha256")
        != source["capability_wrapper_content_sha256"]
        or wrapper != receipt_wrapper
        or wrapper.get("child_release")
        != {
            "contract_id": V2111_CONTRACT_ID,
            "contract_sha256": V2111_CONTRACT_SHA256,
            "git_tag": V2111_SCIENCE_TAG,
            "resolved_git_commit": V2111_SCIENCE_COMMIT,
        }
        or wrapper.get("provider_construction_current_attempt") is not False
        or wrapper.get("provider_calls_current_attempt") != 0
        or wrapper.get("hosted_provider_calls_current_attempt") != 0
        or wrapper.get("current_attempt_usage") != _ZERO_USAGE
        or wrapper.get("imported_effect_cells") != 0
        or wrapper.get("imported_p95_authorities") != []
        or wrapper.get("scientific_evidence") is not False
    ):
        raise PilotV2112ParentImportError(
            f"V2.11.1 {model_id} capability wrapper drifted"
        )
    capability = wrapper.get("capability")
    if not isinstance(capability, Mapping):
        raise PilotV2112ParentImportError(
            f"V2.11.1 {model_id} capability payload is absent"
        )
    samples = capability.get("samples")
    usage_rows = capability.get("usage_rows")
    if (
        capability.get("model_id") != model_id
        or capability.get("runtime_model") != source["runtime_model"]
        or capability.get("served_model") != source["served_model"]
        or capability.get("historical_source_calls") != 30
        or capability.get("action_sample_count") != 24
        or capability.get("semantic_sample_count") != 6
        or capability.get("capability_pass") is not True
        or capability.get("interface_pass") is not True
        or not isinstance(samples, Mapping)
        or set(samples) != {"action", "semantic"}
        or len(samples["action"]) != 24
        or len(samples["semantic"]) != 6
        or not isinstance(usage_rows, list)
        or len(usage_rows) != 30
    ):
        raise PilotV2112ParentImportError(
            f"V2.11.1 {model_id} capability denominator drifted"
        )

    gate_raw, gate = _read_json(
        root,
        paths["gate_receipt"],
        name=f"V2.11.1 {model_id} capability gate receipt",
    )
    if (
        _sha256(gate_raw) != source["capability_gate_file_sha256"]
        or gate.get("capability_pass") is not True
        or gate.get("capability_status") != "imported-pass"
        or gate.get("interface_pass") is not True
        or gate.get("go") is not True
        or gate.get("historical_source_calls") != 30
        or gate.get("provider_calls_current_attempt") != 0
        or gate.get("preflight_run") is not None
        or not str(gate.get("bootstrap_projection", "")).endswith(
            "/v2111_contract_envelope_bootstrap.json"
        )
    ):
        raise PilotV2112ParentImportError(f"V2.11.1 {model_id} capability gate drifted")

    bootstrap_raw, bootstrap = _read_json(
        root,
        paths["bootstrap"],
        name=f"V2.11.1 {model_id} bootstrap audit artifact",
    )
    _verify_seal(
        bootstrap,
        schema_version="finevo-pilot-v2.11.1-contract-envelope-bootstrap-v1",
        name=f"V2.11.1 {model_id} bootstrap audit artifact",
        integrity_in_hash=False,
    )
    if (
        _sha256(bootstrap_raw) != source["bootstrap_file_sha256"]
        or bootstrap.get("scientific_evidence") is not False
        or bootstrap.get("target", {}).get("contract_id") != V2111_CONTRACT_ID
    ):
        raise PilotV2112ParentImportError(
            f"V2.11.1 {model_id} bootstrap audit artifact drifted"
        )

    summary_raw, summary = _read_json(
        root,
        paths["summary"],
        name=f"V2.11.1 {model_id} capability summary",
    )
    _verify_seal(
        summary,
        schema_version="finevo-pilot-terminal-summary-v1",
        name=f"V2.11.1 {model_id} capability summary",
        integrity_in_hash=False,
    )
    if (
        _sha256(summary_raw) != source["capability_summary_file_sha256"]
        or summary.get("contract_id") != V2111_CONTRACT_ID
        or summary.get("contract_sha256") != V2111_CONTRACT_SHA256
        or summary.get("run_spec") != source_spec
        or summary.get("payload", {}).get("capability") != wrapper
        or summary.get("payload", {}).get("gate_evidence") != gate
        or summary.get("scientific_evidence") is not False
        or summary.get("evidence_scope") != "preregistered_task_capability_gate"
    ):
        raise PilotV2112ParentImportError(
            f"V2.11.1 {model_id} capability summary drifted"
        )
    return (
        {
            "model_id": model_id,
            "run_id": _capability_run_id(model_id),
            "run_spec": _json_copy(source_spec),
            "runtime_model": source["runtime_model"],
            "served_model": source["served_model"],
            "historical_source_calls": 30,
            "action_sample_count": 24,
            "semantic_sample_count": 6,
            "capability_pass": True,
            "interface_pass": True,
            "wrapper": _json_copy(wrapper),
            "wrapper_content_sha256": wrapper["integrity"]["content_sha256"],
        },
        {
            "wrapper": _binding(paths["wrapper"], wrapper_raw, wrapper),
            "gate_receipt": _binding(paths["gate_receipt"], gate_raw, gate),
            "summary": _binding(paths["summary"], summary_raw, summary),
            "historical_bootstrap_audit_only": _binding(
                paths["bootstrap"], bootstrap_raw, bootstrap
            ),
        },
    )


def _sum_usage(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    prompt = sum(int(row["prompt_tokens"]) for row in rows)
    completion = sum(int(row["completion_tokens"]) for row in rows)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cost_usd": math.fsum(float(row["cost_usd"]) for row in rows),
    }


def _usage_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        int(left.get("prompt_tokens", -1)) == int(right.get("prompt_tokens", -2))
        and int(left.get("completion_tokens", -1))
        == int(right.get("completion_tokens", -2))
        and int(left.get("total_tokens", -1)) == int(right.get("total_tokens", -2))
        and math.isclose(
            float(left.get("cost_usd", math.nan)),
            float(right.get("cost_usd", math.nan)),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


def _verify_journal(
    root: Path,
    *,
    model_id: str,
    path: PurePosixPath,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _MODEL_SOURCES[model_id]
    raw, value = _read_json(root, path, name=f"V2.11.1 {model_id} failed journal")
    _verify_field_hash(
        value,
        field="journal_sha256",
        name=f"V2.11.1 {model_id} failed journal",
    )
    events = value.get("events")
    if (
        _sha256(raw) != source["journal_file_sha256"]
        or value.get("schema_version") != "finevo-provider-call-journal-v1"
        or value.get("contract_hash") != V2111_CONTRACT_SHA256
        or value.get("journal_sha256") != source["journal_sha256"]
        or not isinstance(events, list)
        or len(events) != 64
        or events[-1].get("event_sha256") != source["journal_event_head"]
    ):
        raise PilotV2112ParentImportError(
            f"V2.11.1 {model_id} failed journal identity drifted"
        )
    previous = "0" * 64
    completion_rows: list[dict[str, Any]] = []
    event_types: Counter[str] = Counter()
    call_kinds: Counter[str] = Counter()
    for index, source_event in enumerate(events):
        if not isinstance(source_event, Mapping):
            raise PilotV2112ParentImportError("journal event is malformed")
        event = _json_copy(source_event)
        digest = event.pop("event_sha256", None)
        if (
            source_event.get("event_index") != index
            or source_event.get("previous_event_sha256") != previous
            or digest != canonical_sha256(event)
        ):
            raise PilotV2112ParentImportError(
                f"V2.11.1 {model_id} failed journal event chain drifted"
            )
        previous = str(digest)
        event_type = str(source_event.get("event_type"))
        event_types[event_type] += 1
        if event_type == "completion_received":
            payload = source_event.get("payload")
            if not isinstance(payload, Mapping):
                raise PilotV2112ParentImportError("completion payload is malformed")
            usage = payload.get("usage")
            call_kind = payload.get("call_kind")
            if (
                not isinstance(usage, Mapping)
                or call_kind not in {"action", "semantic"}
                or payload.get("response_completed") is not True
                or payload.get("finish_reason") != "stop"
                or payload.get("output_disposition") != "accepted"
                or payload.get("response_model") != source["served_model"]
                or usage.get("total_tokens")
                != usage.get("prompt_tokens") + usage.get("completion_tokens")
            ):
                raise PilotV2112ParentImportError(
                    f"V2.11.1 {model_id} completion journal row drifted"
                )
            completion_rows.append(dict(usage))
            call_kinds[str(call_kind)] += 1
    if event_types != {
        "completion_received": 32,
        "parse_disposition": 32,
    } or call_kinds != {
        "action": 24,
        "semantic": 8,
    }:
        raise PilotV2112ParentImportError(
            f"V2.11.1 {model_id} failed journal denominator drifted"
        )
    usage = _sum_usage(completion_rows)
    if not math.isclose(
        float(usage["cost_usd"]),
        float(source["actual"]["cost_usd"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PilotV2112ParentImportError(
            f"V2.11.1 {model_id} failed journal cost drifted"
        )
    # Deliberately return aggregate audit facts only.  No sample or usage row
    # crosses the V2.11.2 authority boundary.
    return (
        {
            "event_count": 64,
            "completion_event_count": 32,
            "parse_disposition_count": 32,
            "action_call_count": 24,
            "semantic_call_count": 8,
            "aggregate_usage": usage,
            "journal_sha256": value["journal_sha256"],
            "event_head_sha256": events[-1]["event_sha256"],
            "authority_use": "historical-failure-and-budget-audit-only",
            "samples_exported": 0,
            "p95_authority": None,
        },
        _binding(path, raw, value),
    )


def _verify_preflight_model(
    root: Path,
    *,
    model_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _MODEL_SOURCES[model_id]
    paths = _preflight_paths(model_id)

    catalog_raw, catalog = _read_json(
        root,
        paths["provider_catalog"],
        name=f"V2.11.1 {model_id} preflight provider catalog",
    )
    _verify_field_hash(
        catalog,
        field="receipt_sha256",
        name=f"V2.11.1 {model_id} preflight provider catalog",
    )
    rows = catalog.get("rows")
    if (
        _sha256(catalog_raw) != source["provider_catalog_file_sha256"]
        or catalog.get("schema_version") != "finevo-provider-catalog-receipt-v1"
        or catalog.get("contract_sha256") != V2111_CONTRACT_SHA256
        or catalog.get("paid_completions") != 0
        or catalog.get("status") != "pass"
        or not isinstance(rows, list)
        or len(rows) != 1
        or rows[0].get("profile_id") != model_id
        or rows[0].get("served_snapshot") != source["served_model"]
        or rows[0].get("status") != "pass"
    ):
        raise PilotV2112ParentImportError(
            f"V2.11.1 {model_id} provider catalog drifted"
        )

    intent_raw, intent = _read_json(
        root,
        paths["run_intent"],
        name=f"V2.11.1 {model_id} preflight run intent",
    )
    _verify_field_hash(
        intent,
        field="intent_sha256",
        name=f"V2.11.1 {model_id} preflight run intent",
    )
    if (
        _sha256(intent_raw) != source["run_intent_file_sha256"]
        or intent.get("schema_version")
        != "finevo-v2.11-long-context-preflight-run-intent-v1"
        or intent.get("pilot_contract_hash") != V2111_CONTRACT_SHA256
        or intent.get("checkpoint_schema_version") != "finevo-pilot-checkpoint-v4"
        or intent.get("num_agents") != 2
        or intent.get("completed_months") != 12
        or intent.get("provider_call_count") != 32
        or intent.get("action_call_count") != 24
        or intent.get("semantic_call_count") != 8
        or intent.get("prompt_tier_ceiling_tokens") != 200_000
    ):
        raise PilotV2112ParentImportError(
            f"V2.11.1 {model_id} preflight run intent drifted"
        )

    journal_audit, journal_binding = _verify_journal(
        root,
        model_id=model_id,
        path=paths["journal"],
    )

    failure_raw, failure = _read_json(
        root,
        paths["failure"],
        name=f"V2.11.1 {model_id} preflight failure",
    )
    manifest_raw, failure_manifest = _read_json(
        root,
        paths["failure_manifest"],
        name=f"V2.11.1 {model_id} preflight failure manifest",
    )
    _verify_field_hash(
        failure_manifest,
        field="manifest_sha256",
        name=f"V2.11.1 {model_id} preflight failure manifest",
    )
    if (
        _sha256(failure_raw) != source["failure_file_sha256"]
        or _sha256(manifest_raw) != source["failure_manifest_file_sha256"]
        or failure_manifest.get("schema_version") != "verified-failure-receipt-v1"
        or failure_manifest.get("status") != "failed"
        or failure_manifest.get("failure_file") != "failure.json"
        or failure_manifest.get("failure_sha256") != _sha256(failure_raw)
        or failure_manifest.get("failure_size_bytes") != len(failure_raw)
    ):
        raise PilotV2112ParentImportError(
            f"V2.11.1 {model_id} failure manifest drifted"
        )
    budget = failure.get("budget_snapshot")
    config = failure.get("config")
    journal_rows = (
        config.get("provider_call_journals") if isinstance(config, Mapping) else None
    )
    if (
        failure.get("schema_version") != "verified-failure-receipt-v1"
        or failure.get("status") != "failed"
        or failure.get("error", {}).get("type") != "ValueError"
        or failure.get("error", {}).get("message") != source["failure_message"]
        or failure.get("partial_streams_persisted") is not False
        or failure.get("git") != {"commit": V2111_SCIENCE_COMMIT, "dirty": False}
        or not isinstance(budget, Mapping)
        or budget.get("completed_calls") != 32
        or budget.get("active_calls") != 0
        or budget.get("rolled_back_calls") != 0
        or budget.get("stop_reasons") != ["call_limit"]
        or not _usage_equal(
            budget.get("accounted_usage", {}),
            journal_audit["aggregate_usage"],
        )
        or not isinstance(config, Mapping)
        or config.get("contract_id") != V2111_CONTRACT_ID
        or config.get("contract_sha256") != V2111_CONTRACT_SHA256
        or not isinstance(journal_rows, list)
        or len(journal_rows) != 1
        or journal_rows[0].get("file_sha256") != source["journal_file_sha256"]
        or journal_rows[0].get("journal_sha256") != source["journal_sha256"]
        or journal_rows[0].get("event_count") != 64
        or journal_rows[0].get("terminal_dispositions_verified") is not True
        or failure.get("provenance", {}).get("scientific_evidence") is not False
    ):
        raise PilotV2112ParentImportError(
            f"V2.11.1 {model_id} terminal preflight failure drifted"
        )
    return (
        {
            "model_id": model_id,
            "run_id": _preflight_run_id(model_id),
            "status": "failed",
            "provider_calls": 32,
            "cost_usd": source["actual"]["cost_usd"],
            "storage_bytes": source["actual"]["storage_bytes"],
            "failure_type": "ValueError",
            "failure_message": source["failure_message"],
            "journal_audit": journal_audit,
            "checkpoint_created": False,
            "exactness_receipt_created": False,
            "p95_authority_created": False,
            "authority_use": "historical-failure-and-budget-audit-only",
            "samples_exported": 0,
        },
        {
            "provider_catalog": _binding(
                paths["provider_catalog"], catalog_raw, catalog
            ),
            "run_intent": _binding(paths["run_intent"], intent_raw, intent),
            "journal_audit_only": journal_binding,
            "failure": _binding(paths["failure"], failure_raw, failure),
            "failure_manifest": _binding(
                paths["failure_manifest"], manifest_raw, failure_manifest
            ),
        },
    )


def _audit_parent_release(
    parent_science_root: str | Path,
    *,
    grandparent_science_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _strict_root(parent_science_root, name="V2.11.1 science source")
    git = _verify_git_release(root)
    _, contract_binding, specs = _verify_contract(root)
    tracked_manifest_binding = _verify_tracked_v2111_source_manifest(root)
    _, attestation_binding = _verify_attestation(root)
    launch_binding = _verify_launch_input(root)
    _, run_binding = _verify_run_ledger(root, specs)
    _, budget_binding = _verify_budget_ledger(root)

    parent_stage, parent_stage_binding = _verify_stage_receipt(
        root,
        path=V2111_PARENT_IMPORT_STAGE_RECEIPT_PATH,
        file_sha256=V2111_PARENT_IMPORT_STAGE_RECEIPT_FILE_SHA256,
        content_sha256=V2111_PARENT_IMPORT_STAGE_RECEIPT_CONTENT_SHA256,
        stage_id="parent-import",
        registered=1,
        status="complete",
        status_counts={"complete": 1},
        go=True,
        go_models=[],
    )
    del parent_stage
    capability_stage, capability_stage_binding = _verify_stage_receipt(
        root,
        path=V2111_CAPABILITY_STAGE_RECEIPT_PATH,
        file_sha256=V2111_CAPABILITY_STAGE_RECEIPT_FILE_SHA256,
        content_sha256=V2111_CAPABILITY_STAGE_RECEIPT_CONTENT_SHA256,
        stage_id="capability-gate",
        registered=2,
        status="complete",
        status_counts={"complete": 2},
        go=True,
        go_models=["gpt52_main", "gpt56_diagnostic"],
    )
    del capability_stage
    preflight_stage, preflight_stage_binding = _verify_stage_receipt(
        root,
        path=V2111_PREFLIGHT_STAGE_RECEIPT_PATH,
        file_sha256=V2111_PREFLIGHT_STAGE_RECEIPT_FILE_SHA256,
        content_sha256=V2111_PREFLIGHT_STAGE_RECEIPT_CONTENT_SHA256,
        stage_id="long-context-preflight",
        registered=2,
        status="complete-with-no-go",
        status_counts={"failed": 2},
        go=False,
        go_models=[],
    )
    if preflight_stage.get("scientific_matrix_complete") is not False:
        raise PilotV2112ParentImportError(
            "V2.11.1 preflight receipt misstates science completion"
        )

    parent_receipt, parent_receipt_binding = _verify_existing_parent_import(
        root,
        grandparent_science_root=grandparent_science_root,
    )
    calibration_wrapper = calibration_wrapper_from_v2111_receipt(parent_receipt)
    if (
        calibration_wrapper.get("schema_version")
        != V2111_CALIBRATION_WRAPPER_SCHEMA_VERSION
        or calibration_wrapper.get("provider_calls_current_attempt") != 0
        or calibration_wrapper.get("imported_p95_authorities") != []
        or calibration_wrapper.get("scientific_evidence") is not False
    ):
        raise PilotV2112ParentImportError("V2.11.1 calibration wrapper drifted")
    calibration = calibration_wrapper.get("calibration")
    if (
        not isinstance(calibration, Mapping)
        or calibration.get("q_ref") != 63.50397933257746
        or calibration.get("selected_utility_profile", {}).get("profile_id") != "nu-0.5"
        or calibration.get("stage0_absolute_flow_utility_threshold", {}).get("value")
        != 0.05617208967516696
    ):
        raise PilotV2112ParentImportError("V2.11.1 calibration values drifted")

    receipt_wrappers = capability_wrappers_from_v2111_receipt(parent_receipt)
    capability_specs = {
        spec.model_id: spec.to_dict()
        for spec in specs
        if spec.stage_id == "capability-gate"
    }
    if set(capability_specs) != set(_MODEL_SOURCES):
        raise PilotV2112ParentImportError("V2.11.1 capability specs drifted")
    capabilities: dict[str, dict[str, Any]] = {}
    capability_bindings: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}
    failure_bindings: dict[str, dict[str, Any]] = {}
    for model_id in sorted(_MODEL_SOURCES):
        capabilities[model_id], capability_bindings[model_id] = (
            _verify_capability_model(
                root,
                model_id=model_id,
                receipt_wrapper=receipt_wrappers[model_id],
                source_spec=capability_specs[model_id],
            )
        )
        failures[model_id], failure_bindings[model_id] = _verify_preflight_model(
            root,
            model_id=model_id,
        )
    for path in _expected_absent_paths():
        _verify_expected_absent(root, path)

    manifest = _seal(
        {
            "schema_version": V2112_SOURCE_MANIFEST_SCHEMA_VERSION,
            "parent_release": {
                "root_hint": "../finevo-pilot-v2-11-1-science",
                "contract_id": V2111_CONTRACT_ID,
                "contract_sha256": V2111_CONTRACT_SHA256,
                **git,
                "publication_status": "immutable-terminal-no-go",
                "contract": contract_binding,
                "tracked_v2111_source_manifest": tracked_manifest_binding,
                "release_attestation": attestation_binding,
                "scientific_launch_input": launch_binding,
                "run_ledger": {
                    **run_binding,
                    "internal_sha256": V2111_RUN_LEDGER_SHA256,
                    "event_count": V2111_RUN_LEDGER_EVENT_COUNT,
                    "event_head_sha256": V2111_RUN_LEDGER_EVENT_HEAD,
                    "run_count": 136,
                },
                "budget_ledger": {
                    **budget_binding,
                    "internal_sha256": V2111_BUDGET_LEDGER_SHA256,
                    "event_count": V2111_BUDGET_LEDGER_EVENT_COUNT,
                    "event_head_sha256": V2111_BUDGET_LEDGER_EVENT_HEAD,
                    "run_count": 5,
                },
            },
            "terminal_denominator": {
                "registered_cells": 136,
                "status_counts": dict(V2111_EXPECTED_STATUS_COUNTS),
                "stage_status_counts": {
                    stage_id: dict(counts)
                    for stage_id, counts in V2111_EXPECTED_STAGE_STATUS_COUNTS.items()
                },
                "all_cells_terminal": True,
                "scientific_matrix_complete": False,
                "post_gate_authority_created": False,
                "preflight_checkpoint_count": 0,
                "preflight_exactness_receipt_count": 0,
            },
            "parent_stage_receipt": parent_stage_binding,
            "calibration_source": {
                "parent_import_receipt": parent_receipt_binding,
                "parent_import_stage_receipt": parent_stage_binding,
                "source_wrapper_schema_version": (
                    V2111_CALIBRATION_WRAPPER_SCHEMA_VERSION
                ),
                "source_wrapper_content_sha256": calibration_wrapper["integrity"][
                    "content_sha256"
                ],
                "q_ref": calibration["q_ref"],
                "selected_profile_id": calibration["selected_utility_profile"][
                    "profile_id"
                ],
                "absolute_flow_utility_threshold": calibration[
                    "stage0_absolute_flow_utility_threshold"
                ]["value"],
                "scientific_evidence": False,
            },
            "capability_source": {
                "stage_receipt": capability_stage_binding,
                "models": {
                    model_id: {
                        **capability_bindings[model_id],
                        "run_id": capabilities[model_id]["run_id"],
                        "runtime_model": capabilities[model_id]["runtime_model"],
                        "historical_source_calls": 30,
                        "action_sample_count": 24,
                        "semantic_sample_count": 6,
                        "capability_pass": True,
                        "interface_pass": True,
                        "scientific_evidence": False,
                    }
                    for model_id in sorted(capabilities)
                },
            },
            "failed_preflight_audit": {
                "stage_receipt": preflight_stage_binding,
                "models": {
                    model_id: {
                        **failure_bindings[model_id],
                        **failures[model_id],
                    }
                    for model_id in sorted(failures)
                },
                "historical_provider_calls": 64,
                "historical_cost_usd": V2111_ATTEMPT_COST_USD,
                "samples_exported": 0,
                "checkpoint_artifacts_exported": [],
                "p95_authorities_exported": [],
                "authority_use": "historical-failure-and-budget-audit-only",
            },
            "expected_absent": [path.as_posix() for path in _expected_absent_paths()],
            "cumulative_parent_budget_debit": _parent_budget_debit().to_dict(),
            "import_policy": {
                **_ZERO_PROVIDER_POLICY,
                "imported_calibration_wrappers": 1,
                "imported_capability_wrappers": 2,
                "historical_capability_calls": 60,
                "historical_failed_preflight_calls": 64,
                "historical_failed_preflight_calls_used_for_budget_only": 64,
                "historical_failed_preflight_calls_used_for_p95": 0,
                "historical_effect_cells_imported": 0,
                "claim_boundary": (
                    "Calibration and capability/interface evidence only; no "
                    "V2.11.1 failed-preflight sample, checkpoint, P95 authority, "
                    "A-D effect, or cross-model effect is imported."
                ),
            },
        }
    )
    return {
        "parent_root": root,
        "manifest": manifest,
        "calibration_wrapper": _json_copy(calibration_wrapper),
        "capability_sources": capabilities,
        "failed_preflight_audit": failures,
    }


def build_v2112_source_manifest(
    *,
    parent_science_root: str | Path,
    grandparent_science_root: str | Path | None = None,
) -> dict[str, Any]:
    """Render the tracked manifest from immutable V2.11.1 source bytes."""

    return _audit_parent_release(
        parent_science_root,
        grandparent_science_root=grandparent_science_root,
    )["manifest"]


def _load_tracked_source_manifest(repo_root: Path) -> dict[str, Any]:
    _, raw = _guarded_file(
        repo_root,
        V2112_SOURCE_MANIFEST_PATH,
        name="tracked V2.11.2 parent source manifest",
    )
    if _sha256(raw) != V2112_SOURCE_MANIFEST_FILE_SHA256:
        raise PilotV2112ParentImportError(
            "tracked V2.11.2 source manifest file hash drifted"
        )
    value = _strict_json(raw, name="tracked V2.11.2 parent source manifest")
    _verify_seal(
        value,
        schema_version=V2112_SOURCE_MANIFEST_SCHEMA_VERSION,
        name="tracked V2.11.2 parent source manifest",
    )
    if (
        value.get("integrity", {}).get("content_sha256")
        != V2112_SOURCE_MANIFEST_CONTENT_SHA256
    ):
        raise PilotV2112ParentImportError(
            "tracked V2.11.2 source manifest content hash drifted"
        )
    return value


def _resolve_parent_root(
    repo_root: Path,
    manifest: Mapping[str, Any],
    *,
    parent_science_root: str | Path | None,
) -> Path:
    if parent_science_root is not None:
        return _strict_root(parent_science_root, name="V2.11.1 science source")
    hint = manifest.get("parent_release", {}).get("root_hint")
    if not isinstance(hint, str) or not hint:
        raise PilotV2112ParentImportError("V2.11.1 source root hint is malformed")
    return _strict_root(repo_root / hint, name="V2.11.1 science source")


def _audit_sources(
    *,
    repo_root: str | Path,
    parent_science_root: str | Path | None,
    grandparent_science_root: str | Path | None,
) -> dict[str, Any]:
    child_root = _strict_root(repo_root, name="V2.11.2 repository")
    tracked = _load_tracked_source_manifest(child_root)
    parent_root = _resolve_parent_root(
        child_root,
        tracked,
        parent_science_root=parent_science_root,
    )
    audit = _audit_parent_release(
        parent_root,
        grandparent_science_root=grandparent_science_root,
    )
    if audit["manifest"] != tracked:
        raise PilotV2112ParentImportError(
            "V2.11.1 source bytes differ from the tracked V2.11.2 manifest"
        )
    audit["repo_root"] = child_root
    return audit


def verify_v2112_parent_sources(
    *,
    repo_root: str | Path,
    parent_science_root: str | Path | None = None,
    grandparent_science_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the full immutable source without constructing a provider."""

    audit = _audit_sources(
        repo_root=repo_root,
        parent_science_root=parent_science_root,
        grandparent_science_root=grandparent_science_root,
    )
    return {
        "parent_root": str(audit["parent_root"]),
        "source_manifest": _json_copy(audit["manifest"]),
        "calibration_wrapper": _json_copy(audit["calibration_wrapper"]),
        "capability_wrapper_content_sha256": {
            model_id: source["wrapper_content_sha256"]
            for model_id, source in audit["capability_sources"].items()
        },
        "failed_preflight_audit": {
            model_id: {
                "run_id": row["run_id"],
                "status": row["status"],
                "provider_calls": row["provider_calls"],
                "cost_usd": row["cost_usd"],
                "failure_type": row["failure_type"],
                "failure_message": row["failure_message"],
                "journal_sha256": row["journal_audit"]["journal_sha256"],
                "samples_exported": 0,
                "p95_authority": None,
            }
            for model_id, row in audit["failed_preflight_audit"].items()
        },
        "cumulative_parent_budget_debit": _parent_budget_debit().to_dict(),
        **_ZERO_PROVIDER_POLICY,
    }


def _child_binding(
    *,
    child_contract_sha256: str,
    child_git_tag: str,
    child_git_commit: str,
) -> dict[str, str]:
    if _SHA256_RE.fullmatch(child_contract_sha256) is None:
        raise PilotV2112ParentImportError(
            "child_contract_sha256 must be a lowercase SHA-256"
        )
    if child_git_tag != "pilot-v2.11.2-science":
        raise PilotV2112ParentImportError("child_git_tag must be pilot-v2.11.2-science")
    if _COMMIT_RE.fullmatch(child_git_commit) is None:
        raise PilotV2112ParentImportError(
            "child_git_commit must be a lowercase 40-hex commit"
        )
    return {
        "contract_id": "finevo-pilot-v2.11.2",
        "contract_sha256": child_contract_sha256,
        "git_tag": child_git_tag,
        "resolved_git_commit": child_git_commit,
    }


def _source_manifest_receipt_binding() -> dict[str, str]:
    return {
        "path": V2112_SOURCE_MANIFEST_PATH.as_posix(),
        "file_sha256": V2112_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": V2112_SOURCE_MANIFEST_CONTENT_SHA256,
    }


def _parent_release_binding() -> dict[str, str]:
    return {
        "contract_id": V2111_CONTRACT_ID,
        "contract_sha256": V2111_CONTRACT_SHA256,
        "git_tag": V2111_SCIENCE_TAG,
        "git_tag_object": V2111_SCIENCE_TAG_OBJECT,
        "resolved_git_commit": V2111_SCIENCE_COMMIT,
    }


def _calibration_wrapper(
    audit: Mapping[str, Any],
    child: Mapping[str, str],
) -> dict[str, Any]:
    source = audit["calibration_wrapper"]
    return _seal(
        {
            "schema_version": V2112_CALIBRATION_WRAPPER_SCHEMA_VERSION,
            "child_release": _json_copy(child),
            "parent_release": _parent_release_binding(),
            "source_manifest": _source_manifest_receipt_binding(),
            "source_wrapper": {
                "schema_version": V2111_CALIBRATION_WRAPPER_SCHEMA_VERSION,
                "content_sha256": source["integrity"]["content_sha256"],
                "parent_import_receipt_path": (
                    V2111_PARENT_IMPORT_RECEIPT_PATH.as_posix()
                ),
            },
            "calibration": _json_copy(source["calibration"]),
            "provider_construction_current_attempt": False,
            "provider_calls_current_attempt": 0,
            "hosted_provider_calls_current_attempt": 0,
            "imported_effect_cells": 0,
            "imported_preflight_samples": 0,
            "imported_checkpoint_artifacts": [],
            "imported_p95_authorities": [],
            "scientific_evidence": False,
            "evidence_use": (
                "Outcome-blind q-ref, selected Stage-0 utility profile, and "
                "absolute flow-utility threshold only."
            ),
        }
    )


def _capability_wrapper(
    audit: Mapping[str, Any],
    child: Mapping[str, str],
    *,
    model_id: str,
) -> dict[str, Any]:
    source = audit["capability_sources"][model_id]
    source_binding = audit["manifest"]["capability_source"]["models"][model_id][
        "wrapper"
    ]
    return _seal(
        {
            "schema_version": V2112_CAPABILITY_WRAPPER_SCHEMA_VERSION,
            "child_release": _json_copy(child),
            "parent_release": _parent_release_binding(),
            "source_manifest": _source_manifest_receipt_binding(),
            "source_capability_wrapper": {
                **_json_copy(source_binding),
                "schema_version": V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION,
                "run_id": source["run_id"],
                "run_spec": _json_copy(source["run_spec"]),
                "historical_source_calls": 30,
            },
            "capability": _json_copy(source["wrapper"]["capability"]),
            "provider_construction_current_attempt": False,
            "provider_calls_current_attempt": 0,
            "hosted_provider_calls_current_attempt": 0,
            "current_attempt_usage": dict(_ZERO_USAGE),
            "imported_effect_cells": 0,
            "imported_preflight_samples": 0,
            "imported_checkpoint_artifacts": [],
            "imported_p95_authorities": [],
            "scientific_evidence": False,
            "evidence_scope": "preregistered_task_capability_gate",
            "evidence_use": (
                "Capability/interface gate and its original 30 historical "
                "capability samples only; no V2.11.1 failed-preflight sample, "
                "checkpoint, P95, or treatment effect."
            ),
        }
    )


def build_v2112_parent_import(
    *,
    repo_root: str | Path,
    child_contract_sha256: str,
    child_git_tag: str,
    child_git_commit: str,
    parent_science_root: str | Path | None = None,
    grandparent_science_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a compact V2.11.2 receipt after exact zero-provider replay."""

    child = _child_binding(
        child_contract_sha256=child_contract_sha256,
        child_git_tag=child_git_tag,
        child_git_commit=child_git_commit,
    )
    audit = _audit_sources(
        repo_root=repo_root,
        parent_science_root=parent_science_root,
        grandparent_science_root=grandparent_science_root,
    )
    return _seal(
        {
            "schema_version": V2112_PARENT_IMPORT_SCHEMA_VERSION,
            "child_release": child,
            "source_manifest": _source_manifest_receipt_binding(),
            "parent_release": {
                **_parent_release_binding(),
                "release_attestation_sha256": V2111_RELEASE_ATTESTATION_SHA256,
                "run_ledger_sha256": V2111_RUN_LEDGER_SHA256,
                "budget_ledger_sha256": V2111_BUDGET_LEDGER_SHA256,
                "publication_status": "immutable-terminal-no-go",
            },
            "terminal_parent_denominator": _json_copy(
                audit["manifest"]["terminal_denominator"]
            ),
            "failed_preflight_audit": {
                model_id: {
                    "run_id": row["run_id"],
                    "status": row["status"],
                    "provider_calls": row["provider_calls"],
                    "cost_usd": row["cost_usd"],
                    "failure_type": row["failure_type"],
                    "failure_message": row["failure_message"],
                    "journal_file_sha256": audit["manifest"]["failed_preflight_audit"][
                        "models"
                    ][model_id]["journal_audit_only"]["file_sha256"],
                    "journal_sha256": row["journal_audit"]["journal_sha256"],
                    "samples_exported": 0,
                    "checkpoint_exported": False,
                    "p95_authority_exported": False,
                    "authority_use": "historical-failure-and-budget-audit-only",
                }
                for model_id, row in sorted(audit["failed_preflight_audit"].items())
            },
            "expected_absent": [path.as_posix() for path in _expected_absent_paths()],
            "calibration_wrapper": _calibration_wrapper(audit, child),
            "capability_wrappers": {
                model_id: _capability_wrapper(audit, child, model_id=model_id)
                for model_id in sorted(audit["capability_sources"])
            },
            "cumulative_parent_budget_debit": _parent_budget_debit().to_dict(),
            "import_policy": {
                **_ZERO_PROVIDER_POLICY,
                "imported_calibration_wrappers": 1,
                "imported_capability_wrappers": 2,
                "historical_capability_calls": 60,
                "historical_failed_preflight_calls": 64,
                "historical_failed_preflight_calls_used_for_budget_only": 64,
                "historical_failed_preflight_calls_used_for_p95": 0,
                "historical_effect_cells_imported": 0,
                "validation_before_provider_construction": True,
            },
            "scientific_evidence": False,
            "claim_boundary": (
                "V2.11.2 imports calibration and two capability/interface "
                "wrappers only. V2.11.1 remains a terminal 136-cell no-go; "
                "its 64 failed-preflight calls remain budget/audit history and "
                "supply no sample, checkpoint, P95, or treatment-effect authority."
            ),
        }
    )


def _load_receipt(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _json_copy(dict(value))
    path = Path(value).expanduser().absolute()
    parent = _strict_root(path.parent, name="V2.11.2 parent receipt directory")
    _, raw = _guarded_file(
        parent,
        PurePosixPath(path.name),
        name="V2.11.2 parent import receipt",
    )
    return _strict_json(raw, name="V2.11.2 parent import receipt")


def verify_v2112_parent_import_receipt(
    receipt: Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path,
    child_contract_sha256: str,
    child_git_tag: str,
    child_git_commit: str,
    parent_science_root: str | Path | None = None,
    grandparent_science_root: str | Path | None = None,
) -> dict[str, Any]:
    """Replay every source check and require byte-semantic receipt equality."""

    observed = _load_receipt(receipt)
    _verify_seal(
        observed,
        schema_version=V2112_PARENT_IMPORT_SCHEMA_VERSION,
        name="V2.11.2 parent import receipt",
    )
    expected = build_v2112_parent_import(
        repo_root=repo_root,
        child_contract_sha256=child_contract_sha256,
        child_git_tag=child_git_tag,
        child_git_commit=child_git_commit,
        parent_science_root=parent_science_root,
        grandparent_science_root=grandparent_science_root,
    )
    if observed != expected:
        raise PilotV2112ParentImportError(
            "V2.11.2 parent import receipt differs from exact source replay"
        )
    return observed


def validate_v2112_source_manifest(
    value: Mapping[str, Any],
    *,
    parent_science_root: str | Path,
    grandparent_science_root: str | Path | None = None,
) -> dict[str, Any]:
    observed = _json_copy(dict(value))
    _verify_seal(
        observed,
        schema_version=V2112_SOURCE_MANIFEST_SCHEMA_VERSION,
        name="V2.11.2 parent source manifest",
    )
    expected = build_v2112_source_manifest(
        parent_science_root=parent_science_root,
        grandparent_science_root=grandparent_science_root,
    )
    if observed != expected:
        raise PilotV2112ParentImportError(
            "V2.11.2 source manifest differs from immutable source replay"
        )
    return observed


def load_v2112_source_manifest(*, repo_root: str | Path) -> dict[str, Any]:
    root = _strict_root(repo_root, name="V2.11.2 repository")
    return _load_tracked_source_manifest(root)


def _safe_destination(
    repo_root: Path,
    destination: str | Path | None,
) -> Path:
    if destination is None:
        relative = V2112_DEFAULT_RECEIPT_PATH
    else:
        candidate = Path(destination)
        if candidate.is_absolute():
            try:
                relative = PurePosixPath(
                    *candidate.absolute().relative_to(repo_root).parts
                )
            except ValueError as exc:
                raise PilotV2112ParentImportError(
                    "receipt destination escaped V2.11.2 repository"
                ) from exc
        else:
            relative = PurePosixPath(candidate.as_posix())
    normalized = _normalized_relative(
        relative.as_posix(),
        required_top="experiment_results",
        name="V2.11.2 receipt destination",
    )
    path = repo_root.joinpath(*normalized.parts)
    current = repo_root
    for part in normalized.parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise PilotV2112ParentImportError("receipt destination contains a symlink")
    if path.is_symlink():
        raise PilotV2112ParentImportError("receipt destination is a symlink")
    return path


def _persist_exact_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise PilotV2112ParentImportError("receipt destination contains a symlink")
        current = current.parent
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise PilotV2112ParentImportError("receipt destination is a symlink")
        if path.read_bytes() != raw:
            raise PilotV2112ParentImportError(
                "existing receipt differs; refusing to overwrite"
            )
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(temporary, flags, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def persist_v2112_parent_import(
    *,
    repo_root: str | Path,
    child_contract_sha256: str,
    child_git_tag: str,
    child_git_commit: str,
    parent_science_root: str | Path | None = None,
    grandparent_science_root: str | Path | None = None,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Persist only the compact receipt; never copy immutable parent raw."""

    root = _strict_root(repo_root, name="V2.11.2 repository")
    receipt = build_v2112_parent_import(
        repo_root=root,
        child_contract_sha256=child_contract_sha256,
        child_git_tag=child_git_tag,
        child_git_commit=child_git_commit,
        parent_science_root=parent_science_root,
        grandparent_science_root=grandparent_science_root,
    )
    path = _safe_destination(root, destination)
    _persist_exact_json(path, receipt)
    verified = verify_v2112_parent_import_receipt(
        path,
        repo_root=root,
        child_contract_sha256=child_contract_sha256,
        child_git_tag=child_git_tag,
        child_git_commit=child_git_commit,
        parent_science_root=parent_science_root,
        grandparent_science_root=grandparent_science_root,
    )
    raw = path.read_bytes()
    return {
        "receipt": str(path),
        "receipt_file_sha256": _sha256(raw),
        "receipt_content_sha256": verified["integrity"]["content_sha256"],
        "calibration_wrapper_content_sha256": verified["calibration_wrapper"][
            "integrity"
        ]["content_sha256"],
        "capability_wrapper_content_sha256": {
            model_id: wrapper["integrity"]["content_sha256"]
            for model_id, wrapper in verified["capability_wrappers"].items()
        },
        **_ZERO_PROVIDER_POLICY,
    }


def parent_budget_debit_for_v2112(
    contract: Any = None,
    *,
    repo_root: str | Path,
    parent_science_root: str | Path | None = None,
    grandparent_science_root: str | Path | None = None,
) -> ParentBudgetDebit:
    """Return the exact cumulative debit only after immutable-source replay."""

    contract_id = getattr(contract, "contract_id", None)
    if contract is not None and contract_id not in {
        "finevo-pilot-v2.11.2",
        "finevo-pilot-v2.11.2-prospective",
    }:
        raise PilotV2112ParentImportError("parent debit requires the V2.11.2 contract")
    _audit_sources(
        repo_root=repo_root,
        parent_science_root=parent_science_root,
        grandparent_science_root=grandparent_science_root,
    )
    return _parent_budget_debit()


def calibration_wrapper_from_v2112_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    value = _json_copy(dict(receipt))
    _verify_seal(
        value,
        schema_version=V2112_PARENT_IMPORT_SCHEMA_VERSION,
        name="V2.11.2 parent import receipt",
    )
    wrapper = value.get("calibration_wrapper")
    if not isinstance(wrapper, Mapping):
        raise PilotV2112ParentImportError("V2.11.2 calibration wrapper is absent")
    _verify_seal(
        wrapper,
        schema_version=V2112_CALIBRATION_WRAPPER_SCHEMA_VERSION,
        name="V2.11.2 calibration wrapper",
    )
    if (
        wrapper.get("provider_calls_current_attempt") != 0
        or wrapper.get("imported_preflight_samples") != 0
        or wrapper.get("imported_checkpoint_artifacts") != []
        or wrapper.get("imported_p95_authorities") != []
    ):
        raise PilotV2112ParentImportError(
            "V2.11.2 calibration wrapper authority scope drifted"
        )
    return _json_copy(dict(wrapper))


def capability_wrappers_from_v2112_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    value = _json_copy(dict(receipt))
    _verify_seal(
        value,
        schema_version=V2112_PARENT_IMPORT_SCHEMA_VERSION,
        name="V2.11.2 parent import receipt",
    )
    wrappers = value.get("capability_wrappers")
    if not isinstance(wrappers, Mapping) or set(wrappers) != set(_MODEL_SOURCES):
        raise PilotV2112ParentImportError(
            "V2.11.2 capability wrapper denominator drifted"
        )
    result: dict[str, dict[str, Any]] = {}
    for model_id in sorted(_MODEL_SOURCES):
        wrapper = wrappers[model_id]
        if not isinstance(wrapper, Mapping):
            raise PilotV2112ParentImportError(
                f"V2.11.2 {model_id} capability wrapper is malformed"
            )
        _verify_seal(
            wrapper,
            schema_version=V2112_CAPABILITY_WRAPPER_SCHEMA_VERSION,
            name=f"V2.11.2 {model_id} capability wrapper",
        )
        capability = wrapper.get("capability")
        samples = capability.get("samples") if isinstance(capability, Mapping) else None
        usage_rows = (
            capability.get("usage_rows") if isinstance(capability, Mapping) else None
        )
        if (
            not isinstance(capability, Mapping)
            or capability.get("model_id") != model_id
            or capability.get("historical_source_calls") != 30
            or capability.get("action_sample_count") != 24
            or capability.get("semantic_sample_count") != 6
            or not isinstance(samples, Mapping)
            or len(samples.get("action", [])) != 24
            or len(samples.get("semantic", [])) != 6
            or not isinstance(usage_rows, list)
            or len(usage_rows) != 30
            or wrapper.get("provider_calls_current_attempt") != 0
            or wrapper.get("imported_preflight_samples") != 0
            or wrapper.get("imported_checkpoint_artifacts") != []
            or wrapper.get("imported_p95_authorities") != []
            or wrapper.get("scientific_evidence") is not False
        ):
            raise PilotV2112ParentImportError(
                f"V2.11.2 {model_id} capability wrapper scope drifted"
            )
        result[model_id] = _json_copy(dict(wrapper))
    return result


def verified_v2112_capability_source(
    receipt: Mapping[str, Any] | str | Path,
    *,
    model_id: str,
    repo_root: str | Path,
    child_contract_sha256: str,
    child_git_tag: str,
    child_git_commit: str,
    parent_science_root: str | Path | None = None,
    grandparent_science_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return one exact immutable V2.11.1 capability wrapper for bootstrap.

    The returned payload is the V2.11.1 capability wrapper, not its historical
    bootstrap projection and not either failed-preflight journal.  This keeps
    the V2.11.2 preflight bootstrap source identical in kind to the V2.11.1
    pattern while preventing failed-preflight evidence laundering.
    """

    if model_id not in _MODEL_SOURCES:
        raise PilotV2112ParentImportError(
            f"unsupported V2.11.2 capability source model {model_id!r}"
        )
    verified_receipt = verify_v2112_parent_import_receipt(
        receipt,
        repo_root=repo_root,
        child_contract_sha256=child_contract_sha256,
        child_git_tag=child_git_tag,
        child_git_commit=child_git_commit,
        parent_science_root=parent_science_root,
        grandparent_science_root=grandparent_science_root,
    )
    child_wrapper = capability_wrappers_from_v2112_receipt(verified_receipt)[model_id]
    tracked = load_v2112_source_manifest(repo_root=repo_root)
    parent_root = _resolve_parent_root(
        _strict_root(repo_root, name="V2.11.2 repository"),
        tracked,
        parent_science_root=parent_science_root,
    )
    source_binding = child_wrapper["source_capability_wrapper"]
    relative = _normalized_relative(
        source_binding["path"],
        required_top="experiment_results",
        name="V2.11.1 capability wrapper source",
    )
    raw, source_wrapper = _read_json(
        parent_root,
        relative,
        name=f"V2.11.1 {model_id} bootstrap capability source",
    )
    _verify_seal(
        source_wrapper,
        schema_version=V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION,
        name=f"V2.11.1 {model_id} bootstrap capability source",
    )
    if (
        _sha256(raw) != source_binding["file_sha256"]
        or source_wrapper.get("integrity", {}).get("content_sha256")
        != source_binding["content_sha256"]
        or source_wrapper.get("capability") != child_wrapper["capability"]
        or source_binding.get("run_id") != _capability_run_id(model_id)
        or source_binding.get("run_spec", {}).get("run_id")
        != _capability_run_id(model_id)
        or relative.name != "capability.json"
        or "long-context-preflight" in relative.parts
        or "bootstrap" in relative.name
        or "journal" in relative.name
    ):
        raise PilotV2112ParentImportError(
            f"V2.11.1 {model_id} capability bootstrap source drifted"
        )
    return {
        "schema_version": V2112_VERIFIED_CAPABILITY_SOURCE_SCHEMA_VERSION,
        "model_id": model_id,
        "run_id": _capability_run_id(model_id),
        "run_spec": _json_copy(source_binding["run_spec"]),
        "source_release": _parent_release_binding(),
        "source_path": relative.as_posix(),
        "source_file_sha256": _sha256(raw),
        "source_content_sha256": source_wrapper["integrity"]["content_sha256"],
        "payload": _json_copy(source_wrapper),
        "provider_construction_during_verification": False,
        "provider_calls_during_verification": 0,
        "failed_preflight_samples_imported": 0,
        "checkpoint_artifacts_imported": [],
        "p95_authorities_imported": [],
    }


def verified_v2112_inherited_capability_binding(
    receipt: Mapping[str, Any] | str | Path,
    *,
    model_id: str,
    repo_root: str | Path,
    child_contract_sha256: str,
    child_git_tag: str,
    child_git_commit: str,
    parent_science_root: str | Path | None = None,
    grandparent_science_root: str | Path | None = None,
) -> dict[str, Any]:
    verified = verify_v2112_parent_import_receipt(
        receipt,
        repo_root=repo_root,
        child_contract_sha256=child_contract_sha256,
        child_git_tag=child_git_tag,
        child_git_commit=child_git_commit,
        parent_science_root=parent_science_root,
        grandparent_science_root=grandparent_science_root,
    )
    wrapper = capability_wrappers_from_v2112_receipt(verified).get(model_id)
    if wrapper is None:
        raise PilotV2112ParentImportError(
            f"V2.11.2 capability wrapper is absent for {model_id!r}"
        )
    return {
        "model_id": model_id,
        "wrapper_content_sha256": wrapper["integrity"]["content_sha256"],
        "payload": wrapper,
        "provider_construction_during_verification": False,
        "provider_calls_during_verification": 0,
    }


__all__ = [
    "PilotV2112ParentImportError",
    "V2111_BUDGET_LEDGER_SHA256",
    "V2111_CONTRACT_ID",
    "V2111_CONTRACT_SHA256",
    "V2111_CUMULATIVE_COMPLETIONS",
    "V2111_CUMULATIVE_COST_USD",
    "V2111_CUMULATIVE_STORAGE_BYTES",
    "V2111_PARENT_DEBIT_RECORD_SHA256",
    "V2111_RUN_LEDGER_SHA256",
    "V2111_SCIENCE_COMMIT",
    "V2111_SCIENCE_TAG",
    "V2111_SCIENCE_TAG_OBJECT",
    "V2112_CALIBRATION_WRAPPER_SCHEMA_VERSION",
    "V2112_CAPABILITY_WRAPPER_SCHEMA_VERSION",
    "V2112_PARENT_IMPORT_SCHEMA_VERSION",
    "V2112_SOURCE_MANIFEST_CONTENT_SHA256",
    "V2112_SOURCE_MANIFEST_FILE_SHA256",
    "V2112_SOURCE_MANIFEST_PATH",
    "V2112_SOURCE_MANIFEST_SCHEMA_VERSION",
    "V2112_VERIFIED_CAPABILITY_SOURCE_SCHEMA_VERSION",
    "build_v2112_parent_import",
    "build_v2112_source_manifest",
    "calibration_wrapper_from_v2112_receipt",
    "capability_wrappers_from_v2112_receipt",
    "load_v2112_source_manifest",
    "parent_budget_debit_for_v2112",
    "persist_v2112_parent_import",
    "validate_v2112_source_manifest",
    "verified_v2112_capability_source",
    "verified_v2112_inherited_capability_binding",
    "verify_v2112_parent_import_receipt",
    "verify_v2112_parent_sources",
]
