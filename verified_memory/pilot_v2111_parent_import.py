"""Strict, zero-provider V2.11 parent import for the V2.11.1 retry.

V2.11 is an immutable scientific no-go release.  Its two capability cells
completed, both long-context preflight cells failed before a provider call,
and the remaining 131 scientific cells were integrity-stopped.  This module
preserves that denominator exactly and imports only:

* the already imported, outcome-blind calibration prerequisites;
* two capability/interface gate payloads as child-bound wrapper evidence; and
* the cumulative V2.11 budget debit.

No V2.11 treatment-effect cell or observed-p95 authority crosses this
boundary.  Provider construction is intentionally absent from this module.
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
from typing import Any, Mapping

from .pilot_budget import ParentBudgetDebit
from .pilot_capability import (
    CAPABILITY_SCHEMA_VERSION,
    CAPABILITY_TASKSET_SHA256,
)
from .pilot_contract import (
    canonical_contract_sha256,
    canonical_sha256,
    load_pilot_contract,
)
from .pilot_v211_gate import PilotV211GateError, _capability_rows


V2111_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.11.1-parent-source-manifest-v1"
V2111_PARENT_IMPORT_SCHEMA_VERSION = "finevo-pilot-v2.11.1-parent-import-v1"
V2111_CALIBRATION_WRAPPER_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.1-imported-calibration-wrapper-v1"
)
V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.1-imported-capability-wrapper-v1"
)
V2111_VERIFIED_CAPABILITY_SOURCE_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.1-verified-capability-source-v1"
)
V2111_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_11_1_source_manifest.json"
)
V2111_SOURCE_MANIFEST_FILE_SHA256 = (
    "78f7910ddbd5aa1207b869fc68d45650576e2e370af0f06cc53d0bc7226b71c5"
)
V2111_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "7cf33945cff145fa5ca4cf6aae521acec8c96bf521abac076982ffc2e88b7812"
)
V2111_DEFAULT_RECEIPT_PATH = PurePosixPath(
    "experiment_results/pilot-v2.11.1/raw/parent-import/" "parent_import_receipt.json"
)

V211_CONTRACT_ID = "finevo-pilot-v2.11"
V211_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_11.yaml")
V211_CONTRACT_FILE_SHA256 = (
    "2417bff327cf9a38ae077e6c5b40265346ad6177a7a7a4c17a44769de92a03d8"
)
V211_CONTRACT_SHA256 = (
    "d65a60ccab684654979fce598f013b72c813f36f5d40d6063b81ac87557c2c36"
)
V211_SCIENCE_TAG = "pilot-v2.11-science"
V211_SCIENCE_TAG_OBJECT = "c4b457d0cc8e7e48f99c64f0283ab043877cc47f"
V211_SCIENCE_COMMIT = "5d6c7920bd4a872b02931fdee8a47b9ac4e7b352"
V211_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.11/raw")
V211_RELEASE_ATTESTATION_PATH = V211_RAW_ROOT / "release_attestation.json"
V211_RELEASE_ATTESTATION_FILE_SHA256 = (
    "db06b47051f0c98d47bca133f42bb2f9c114e7e4b759679ae9e590383b219697"
)
V211_RELEASE_ATTESTATION_SHA256 = (
    "d8de3dd480f395e9d406eaa06f569f7dc904fcef892052de33e3dd5ee2265baf"
)
V211_RUN_LEDGER_PATH = V211_RAW_ROOT / "run_ledger.json"
V211_RUN_LEDGER_FILE_SHA256 = (
    "b0a4a0af97ec3fbee3247ceb51b5c7e0241c8d02d7ac0fba55031654bf0b8dbb"
)
V211_RUN_LEDGER_SHA256 = (
    "d50d89535b0896f46f4ded93d9ca28062558a75a7fb8b9548f989d77233f20a1"
)
V211_RUN_LEDGER_EVENT_COUNT = 138
V211_RUN_LEDGER_EVENT_HEAD = (
    "e0177f58837c727e0d8173000db1df8bd7248bd5ce7caefa7b942b226476ca8c"
)
V211_BUDGET_LEDGER_PATH = V211_RAW_ROOT / "budget_ledger.json"
V211_BUDGET_LEDGER_FILE_SHA256 = (
    "842fba225918c472210042597925847566d0639a5d67331cca8ad8bb2c1cb366"
)
V211_BUDGET_LEDGER_SHA256 = (
    "be72be029f0e558153f0a81545ffe13347833031bb9b5611bcd929dc0b0408d8"
)
V211_BUDGET_LEDGER_EVENT_COUNT = 12
V211_BUDGET_LEDGER_EVENT_HEAD = (
    "8c6ba4853cee2347db9718c8155dbf9b2a211bee0b77703b0e20ac380ce29621"
)

V211_CUMULATIVE_COST_USD = 17.166524062500006
V211_CUMULATIVE_COMPLETIONS = 876
V211_CUMULATIVE_STORAGE_BYTES = 217_581_135
V211_PARENT_DEBIT_RECORD_SHA256 = (
    "e5b8406c636d5045040677ca0bd09dd72557afdef2998095f0f5775a0ead8b9c"
)

V211_PARENT_IMPORT_RECEIPT_PATH = (
    V211_RAW_ROOT / "parent-import/parent_import_receipt.json"
)
V211_PARENT_IMPORT_SUMMARY_PATH = V211_RAW_ROOT / (
    "parent-import/summaries/"
    "finevo-pilot-v2.11--parent-import--qref_scripted--parent-import--none--"
    "provider-preflight-default--s2010922376.json"
)
V211_PARENT_IMPORT_STAGE_RECEIPT_PATH = (
    V211_RAW_ROOT / "parent-import/stage_receipt.json"
)

V211_EXPECTED_ABSENT_PATHS = (
    V211_RAW_ROOT / "long-context-preflight/post_gate_authority.json",
    V211_RAW_ROOT / "long-context-preflight/stage_receipt.json",
)
V211_EXPECTED_STAGE_STATUS_COUNTS: Mapping[str, Mapping[str, int]] = {
    "parent-import": {"complete": 1},
    "capability-gate": {"complete": 2},
    "long-context-preflight": {"failed": 2},
    "experiment-c": {"integrity-stopped": 25},
    "experiment-a": {"integrity-stopped": 20},
    "experiment-d": {"integrity-stopped": 55},
    "experiment-b": {"integrity-stopped": 25},
    "cross-model": {"integrity-stopped": 6},
}
V211_EXPECTED_STATUS_COUNTS = {
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
    "imported_p95_authorities": [],
    "raw_tree_copied": False,
    "copied_file_count": 0,
    "copied_byte_count": 0,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _capability_run_id(model_id: str) -> str:
    return (
        "finevo-pilot-v2.11--capability-gate--"
        f"{model_id}--capability-probe--none--provider-preflight-default--"
        "s2010922376"
    )


def _preflight_run_id(model_id: str) -> str:
    return (
        "finevo-pilot-v2.11--long-context-preflight--"
        f"{model_id}--closed-loop-preflight--none--stage0-selected--"
        "s2010922376"
    )


_MODEL_SOURCES: Mapping[str, Mapping[str, Any]] = {
    "gpt52_main": {
        "runtime_model": "openai/gpt-5.2-2025-12-11",
        "requested_model": "gpt-5.2-2025-12-11",
        "served_model": "gpt-5.2-2025-12-11",
        "capability_cost_usd": 0.5358062499999999,
        "capability_prompt_tokens": 26_127,
        "capability_completion_tokens": 35_006,
        "capability_storage_bytes": 262_723,
        "preflight_cost_cap_usd": 13.035008,
        "preflight_failure_message": (
            "scientific dispatch lacks an exact observed+25% preflight p95 "
            "reservation for openai/gpt-5.2-2025-12-11::action"
        ),
        "capability_catalog": (
            V211_RAW_ROOT / "capability-gate/provider_catalog/gpt52_main.json"
        ),
        "capability_payload": V211_RAW_ROOT
        / (
            "capability-gate/runs/"
            f"{_capability_run_id('gpt52_main')}/capability.json"
        ),
        "capability_gate": V211_RAW_ROOT
        / (
            "capability-gate/runs/"
            f"{_capability_run_id('gpt52_main')}/gate_receipt.json"
        ),
        "capability_summary": V211_RAW_ROOT
        / ("capability-gate/summaries/" f"{_capability_run_id('gpt52_main')}.json"),
        "preflight_catalog": (
            V211_RAW_ROOT / "long-context-preflight/provider_catalog/gpt52_main.json"
        ),
        "preflight_intent": V211_RAW_ROOT
        / (
            "long-context-preflight/runs/"
            f"{_preflight_run_id('gpt52_main')}/"
            "preflight_checkpoint.json.run-intent.json"
        ),
        "preflight_failure": V211_RAW_ROOT
        / (
            "long-context-preflight/runs/"
            f"{_preflight_run_id('gpt52_main')}/"
            "failure_receipt/failure.json"
        ),
        "preflight_failure_manifest": V211_RAW_ROOT
        / (
            "long-context-preflight/runs/"
            f"{_preflight_run_id('gpt52_main')}/"
            "failure_receipt/failure_manifest.json"
        ),
    },
    "gpt56_diagnostic": {
        "runtime_model": "openai/gpt-5.6-sol",
        "requested_model": "gpt-5.6-sol",
        "served_model": "gpt-5.6-sol",
        "capability_cost_usd": 0.585795,
        "capability_prompt_tokens": 26_127,
        "capability_completion_tokens": 15_172,
        "capability_storage_bytes": 259_798,
        "preflight_cost_cap_usd": 35.93216,
        "preflight_failure_message": (
            "scientific dispatch lacks an exact observed+25% preflight p95 "
            "reservation for openai/gpt-5.6-sol::action"
        ),
        "capability_catalog": (
            V211_RAW_ROOT / "capability-gate/provider_catalog/gpt56_diagnostic.json"
        ),
        "capability_payload": V211_RAW_ROOT
        / (
            "capability-gate/runs/"
            f"{_capability_run_id('gpt56_diagnostic')}/capability.json"
        ),
        "capability_gate": V211_RAW_ROOT
        / (
            "capability-gate/runs/"
            f"{_capability_run_id('gpt56_diagnostic')}/gate_receipt.json"
        ),
        "capability_summary": V211_RAW_ROOT
        / (
            "capability-gate/summaries/"
            f"{_capability_run_id('gpt56_diagnostic')}.json"
        ),
        "preflight_catalog": (
            V211_RAW_ROOT
            / "long-context-preflight/provider_catalog/gpt56_diagnostic.json"
        ),
        "preflight_intent": V211_RAW_ROOT
        / (
            "long-context-preflight/runs/"
            f"{_preflight_run_id('gpt56_diagnostic')}/"
            "preflight_checkpoint.json.run-intent.json"
        ),
        "preflight_failure": V211_RAW_ROOT
        / (
            "long-context-preflight/runs/"
            f"{_preflight_run_id('gpt56_diagnostic')}/"
            "failure_receipt/failure.json"
        ),
        "preflight_failure_manifest": V211_RAW_ROOT
        / (
            "long-context-preflight/runs/"
            f"{_preflight_run_id('gpt56_diagnostic')}/"
            "failure_receipt/failure_manifest.json"
        ),
    },
}


class PilotV2111ParentImportError(RuntimeError):
    """Raised before any inherited V2.11 authority may be consumed."""


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except Exception as exc:
        raise PilotV2111ParentImportError(
            "value is not canonical-JSON compatible"
        ) from exc


def _strict_json(raw: bytes, *, name: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PilotV2111ParentImportError(
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
    except PilotV2111ParentImportError:
        raise
    except Exception as exc:
        raise PilotV2111ParentImportError(f"{name} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise PilotV2111ParentImportError(f"{name} must be a JSON object")
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    candidate = _json_copy(dict(value))
    if "integrity" in candidate:
        raise PilotV2111ParentImportError("cannot seal a pre-sealed value")
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
        raise PilotV2111ParentImportError(f"{name} schema or content hash mismatch")


def _verify_field_hash(
    value: Mapping[str, Any],
    *,
    field: str,
    name: str,
) -> None:
    candidate = _json_copy(dict(value))
    claimed = candidate.pop(field, None)
    if not isinstance(claimed, str) or claimed != canonical_sha256(candidate):
        raise PilotV2111ParentImportError(f"{name} self-hash mismatch")


def _strict_root(value: str | Path, *, name: str) -> Path:
    path = Path(value).expanduser().absolute()
    for component in (path, *path.parents):
        try:
            if component.is_symlink():
                raise PilotV2111ParentImportError(f"{name} path contains a symlink")
        except OSError as exc:
            raise PilotV2111ParentImportError(f"{name} is unavailable") from exc
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PilotV2111ParentImportError(f"{name} is unavailable") from exc
    if not resolved.is_dir():
        raise PilotV2111ParentImportError(f"{name} must be a directory")
    return resolved


def _normalized_relative(
    value: Any,
    *,
    required_top: str,
    name: str,
) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PilotV2111ParentImportError(f"{name} path is malformed")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.parts[0] != required_top
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise PilotV2111ParentImportError(f"{name} path escaped its allowed namespace")
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
                raise PilotV2111ParentImportError(f"{name} path contains a symlink")
        if not stat.S_ISREG(path.lstat().st_mode):
            raise PilotV2111ParentImportError(f"{name} must be a regular file")
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb", closefd=True) as handle:
            raw = handle.read()
    except PilotV2111ParentImportError:
        raise
    except (OSError, ValueError) as exc:
        raise PilotV2111ParentImportError(f"{name} is unavailable") from exc
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
        elif isinstance(value.get("attestation_sha256"), str):
            result["content_sha256"] = value["attestation_sha256"]
        elif isinstance(value.get("manifest_sha256"), str):
            result["content_sha256"] = value["manifest_sha256"]
        elif isinstance(value.get("receipt_sha256"), str):
            result["content_sha256"] = value["receipt_sha256"]
        elif isinstance(value.get("intent_sha256"), str):
            result["content_sha256"] = value["intent_sha256"]
    return result


def _verify_expected_absent(
    root: Path,
    relative: PurePosixPath,
    *,
    name: str,
) -> None:
    current = root
    try:
        for part in relative.parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise PilotV2111ParentImportError(
                    f"{name} parent path contains a symlink"
                )
        path = root.joinpath(*relative.parts)
        if path.exists() or path.is_symlink():
            raise PilotV2111ParentImportError(
                f"{name} must remain absent in immutable V2.11"
            )
    except PilotV2111ParentImportError:
        raise
    except OSError as exc:
        raise PilotV2111ParentImportError(f"{name} absence is unverifiable") from exc


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
        raise PilotV2111ParentImportError(
            "V2.11 git release identity is unavailable"
        ) from exc
    return result.stdout.strip()


def _verify_git_release(root: Path) -> dict[str, str]:
    head = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
    tag_object = _git(
        root,
        "rev-parse",
        "--verify",
        f"refs/tags/{V211_SCIENCE_TAG}^{{tag}}",
    )
    tag_commit = _git(
        root,
        "rev-parse",
        "--verify",
        f"refs/tags/{V211_SCIENCE_TAG}^{{commit}}",
    )
    tracked = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    )
    if (
        head != V211_SCIENCE_COMMIT
        or tag_commit != V211_SCIENCE_COMMIT
        or tag_object != V211_SCIENCE_TAG_OBJECT
        or tracked
    ):
        raise PilotV2111ParentImportError(
            "V2.11 annotated tag, commit, or tracked worktree drifted"
        )
    return {
        "science_tag": V211_SCIENCE_TAG,
        "science_tag_object": tag_object,
        "resolved_git_commit": tag_commit,
    }


def _verify_contract(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], tuple[Any, ...]]:
    raw, value = _read_json(root, V211_CONTRACT_PATH, name="V2.11 contract")
    if _sha256(raw) != V211_CONTRACT_FILE_SHA256:
        raise PilotV2111ParentImportError("V2.11 contract file hash drifted")
    if (
        value.get("contract_id") != V211_CONTRACT_ID
        or value.get("status") != "frozen"
        or value.get("implementation", {}).get("required_git_tag") != V211_SCIENCE_TAG
        or value.get("integrity", {}).get("declared_sha256") != V211_CONTRACT_SHA256
        or canonical_contract_sha256(value) != V211_CONTRACT_SHA256
    ):
        raise PilotV2111ParentImportError("V2.11 contract identity drifted")
    try:
        contract = load_pilot_contract(root.joinpath(*V211_CONTRACT_PATH.parts))
        specs = tuple(contract.expand())
    except Exception as exc:
        raise PilotV2111ParentImportError("V2.11 contract expansion failed") from exc
    if contract.canonical_hash != V211_CONTRACT_SHA256 or len(specs) != 136:
        raise PilotV2111ParentImportError("V2.11 registered denominator drifted")
    expected_counts = {
        stage_id: sum(spec.stage_id == stage_id for spec in specs)
        for stage_id in V211_EXPECTED_STAGE_STATUS_COUNTS
    }
    if expected_counts != {
        stage_id: sum(counts.values())
        for stage_id, counts in V211_EXPECTED_STAGE_STATUS_COUNTS.items()
    }:
        raise PilotV2111ParentImportError("V2.11 contract stage denominator drifted")
    return value, _binding(V211_CONTRACT_PATH, raw, value), specs


def _verify_attestation(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, value = _read_json(
        root,
        V211_RELEASE_ATTESTATION_PATH,
        name="V2.11 release attestation",
    )
    candidate = _json_copy(value)
    claimed = candidate.pop("attestation_sha256", None)
    expected_tag = {
        "kind": "annotated",
        "name": V211_SCIENCE_TAG,
        "object_id": V211_SCIENCE_TAG_OBJECT,
        "peeled_commit": V211_SCIENCE_COMMIT,
    }
    if (
        _sha256(raw) != V211_RELEASE_ATTESTATION_FILE_SHA256
        or value.get("schema_version") != "finevo-scientific-release-attestation-v2"
        or claimed != V211_RELEASE_ATTESTATION_SHA256
        or canonical_sha256(candidate) != V211_RELEASE_ATTESTATION_SHA256
        or value.get("status") != "pass"
        or value.get("head_commit") != V211_SCIENCE_COMMIT
        or value.get("local_tag") != expected_tag
        or value.get("contract", {}).get("canonical_sha256") != V211_CONTRACT_SHA256
        or value.get("contract", {}).get("file_sha256") != V211_CONTRACT_FILE_SHA256
        or value.get("remote", {}).get("tag_object_id") != V211_SCIENCE_TAG_OBJECT
        or value.get("remote", {}).get("tag_peeled_commit") != V211_SCIENCE_COMMIT
        or value.get("remote", {}).get("branch_commit") != V211_SCIENCE_COMMIT
        or value.get("github_actions", {}).get("ci_measurements", {}).get("test_count")
        != 1399
    ):
        raise PilotV2111ParentImportError(
            "V2.11 scientific release attestation drifted"
        )
    return value, _binding(V211_RELEASE_ATTESTATION_PATH, raw, value)


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
        or value.get("contract_hash") != V211_CONTRACT_SHA256
        or claimed != internal_sha256
        or canonical_sha256(candidate) != internal_sha256
        or not isinstance(events, list)
        or len(events) != event_count
        or not isinstance(runs, Mapping)
        or len(runs) != run_count
        or not events
        or events[-1].get("event_sha256") != event_head
    ):
        raise PilotV2111ParentImportError(f"{name} identity drifted")
    previous = "0" * 64
    for index, source in enumerate(events):
        if not isinstance(source, Mapping):
            raise PilotV2111ParentImportError(f"{name} event is malformed")
        row = _json_copy(dict(source))
        digest = row.pop("event_sha256", None)
        if (
            source.get("event_index") != index
            or source.get("previous_event_sha256") != previous
            or digest != canonical_sha256(row)
        ):
            raise PilotV2111ParentImportError(f"{name} event chain drifted")
        previous = str(digest)


def _verify_run_ledger(
    root: Path,
    specs: tuple[Any, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, value = _read_json(
        root,
        V211_RUN_LEDGER_PATH,
        name="V2.11 run ledger",
    )
    if _sha256(raw) != V211_RUN_LEDGER_FILE_SHA256:
        raise PilotV2111ParentImportError("V2.11 run ledger file hash drifted")
    _verify_event_ledger(
        value,
        schema_version="finevo-pilot-run-ledger-v2",
        internal_sha256=V211_RUN_LEDGER_SHA256,
        event_count=V211_RUN_LEDGER_EVENT_COUNT,
        event_head=V211_RUN_LEDGER_EVENT_HEAD,
        run_count=136,
        name="V2.11 run ledger",
    )
    runs = value["runs"]
    expected_specs = {spec.run_id: spec.to_dict() for spec in specs}
    if set(runs) != set(expected_specs):
        raise PilotV2111ParentImportError(
            "V2.11 run ledger denominator differs from contract"
        )
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    for run_id, row in runs.items():
        if not isinstance(row, Mapping) or row.get("spec") != expected_specs[run_id]:
            raise PilotV2111ParentImportError("V2.11 run ledger spec binding drifted")
        by_stage[expected_specs[run_id]["stage_id"]][str(row.get("status"))] += 1
    observed = {stage_id: dict(counts) for stage_id, counts in by_stage.items()}
    if observed != {
        stage_id: dict(counts)
        for stage_id, counts in V211_EXPECTED_STAGE_STATUS_COUNTS.items()
    }:
        raise PilotV2111ParentImportError("V2.11 terminal denominator status drifted")
    return value, _binding(V211_RUN_LEDGER_PATH, raw, value)


def _parent_budget_debit() -> ParentBudgetDebit:
    debit = ParentBudgetDebit(
        parent_contract_sha256=V211_CONTRACT_SHA256,
        parent_run_ledger_sha256=V211_RUN_LEDGER_SHA256,
        parent_budget_ledger_sha256=V211_BUDGET_LEDGER_SHA256,
        stage_bucket="parent_v211",
        cost_usd=V211_CUMULATIVE_COST_USD,
        hosted_completions=V211_CUMULATIVE_COMPLETIONS,
        storage_bytes=V211_CUMULATIVE_STORAGE_BYTES,
        record_sha256=V211_PARENT_DEBIT_RECORD_SHA256,
    )
    return debit


def _verify_budget_ledger(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, value = _read_json(
        root,
        V211_BUDGET_LEDGER_PATH,
        name="V2.11 budget ledger",
    )
    if _sha256(raw) != V211_BUDGET_LEDGER_FILE_SHA256:
        raise PilotV2111ParentImportError("V2.11 budget ledger file hash drifted")
    _verify_event_ledger(
        value,
        schema_version="finevo-pilot-budget-ledger-v2",
        internal_sha256=V211_BUDGET_LEDGER_SHA256,
        event_count=V211_BUDGET_LEDGER_EVENT_COUNT,
        event_head=V211_BUDGET_LEDGER_EVENT_HEAD,
        run_count=5,
        name="V2.11 budget ledger",
    )
    runs = value["runs"]
    expected_actuals = {
        _capability_run_id("gpt52_main"): {
            "completions": 30,
            "cost_usd": 0.5358062499999999,
            "storage_bytes": 262_723,
        },
        _capability_run_id("gpt56_diagnostic"): {
            "completions": 30,
            "cost_usd": 0.585795,
            "storage_bytes": 259_798,
        },
        _preflight_run_id("gpt52_main"): {
            "completions": 0,
            "cost_usd": 0.0,
            "storage_bytes": 18_626,
        },
        _preflight_run_id("gpt56_diagnostic"): {
            "completions": 0,
            "cost_usd": 0.0,
            "storage_bytes": 18_624,
        },
        (
            "finevo-pilot-v2.11--parent-import--qref_scripted--parent-import--"
            "none--provider-preflight-default--s2010922376"
        ): {
            "completions": 0,
            "cost_usd": 0.0,
            "storage_bytes": 10_529,
        },
    }
    if set(runs) != set(expected_actuals):
        raise PilotV2111ParentImportError(
            "V2.11 budget ledger operational denominator drifted"
        )
    for run_id, expected in expected_actuals.items():
        row = runs[run_id]
        if not isinstance(row, Mapping) or row.get("actual") != expected:
            raise PilotV2111ParentImportError("V2.11 budget ledger actuals drifted")
    try:
        parent = ParentBudgetDebit.from_dict(value["parent_debit"])
    except Exception as exc:
        raise PilotV2111ParentImportError(
            "V2.11 inherited V2.10.2 debit is malformed"
        ) from exc
    costs = [float(row["actual"]["cost_usd"]) for row in runs.values()]
    completions = sum(int(row["actual"]["completions"]) for row in runs.values())
    storage = sum(int(row["actual"]["storage_bytes"]) for row in runs.values())
    if (
        not math.isclose(
            parent.cost_usd + math.fsum(costs),
            V211_CUMULATIVE_COST_USD,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or parent.hosted_completions + completions != V211_CUMULATIVE_COMPLETIONS
        or parent.storage_bytes + storage != V211_CUMULATIVE_STORAGE_BYTES
    ):
        raise PilotV2111ParentImportError(
            "V2.11 cumulative parent budget debit drifted"
        )
    _parent_budget_debit()
    return value, _binding(V211_BUDGET_LEDGER_PATH, raw, value)


def _verify_calibration_source(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_raw, receipt = _read_json(
        root,
        V211_PARENT_IMPORT_RECEIPT_PATH,
        name="V2.11 parent import receipt",
    )
    _verify_seal(
        receipt,
        schema_version="finevo-pilot-v2.11-parent-import-v1",
        name="V2.11 parent import receipt",
    )
    imported = receipt.get("imported_prerequisites")
    policy = receipt.get("import_policy")
    if (
        not isinstance(imported, Mapping)
        or imported.get("q_ref") != 63.50397933257746
        or imported.get("selected_utility_profile")
        != {
            "profile_id": "nu-0.5",
            "rho": 1.0,
            "labor_weight": 2.0,
            "inverse_frisch": 0.5,
            "consumption_scale": 63.50397933257746,
            "discount_factor": 0.99,
            "budget_tolerance": 1e-8,
            "max_labor_hours": 168.0,
        }
        or imported.get("stage0_absolute_flow_utility_threshold", {}).get("value")
        != 0.05617208967516696
        or imported.get("stage0_absolute_flow_utility_threshold", {}).get(
            "treatment_outcomes_inspected"
        )
        is not False
        or receipt.get("scientific_evidence") is not False
        or not isinstance(policy, Mapping)
        or policy.get("provider_construction") is not False
        or policy.get("provider_calls") != 0
        or policy.get("imported_effect_cells") != 0
        or policy.get("effect_metrics_observed") is not False
        or policy.get("imported_p95_authorities") != []
    ):
        raise PilotV2111ParentImportError("V2.11 calibration import semantics drifted")
    try:
        inherited_debit = ParentBudgetDebit.from_dict(
            receipt["cumulative_budget_debit"]
        )
    except Exception as exc:
        raise PilotV2111ParentImportError(
            "V2.11 calibration receipt debit is malformed"
        ) from exc
    if (
        inherited_debit.cost_usd != 16.044922812500005
        or inherited_debit.hosted_completions != 816
        or inherited_debit.storage_bytes != 217_010_835
    ):
        raise PilotV2111ParentImportError(
            "V2.11 calibration receipt parent debit drifted"
        )

    summary_raw, summary = _read_json(
        root,
        V211_PARENT_IMPORT_SUMMARY_PATH,
        name="V2.11 parent import terminal summary",
    )
    _verify_seal(
        summary,
        schema_version="finevo-pilot-terminal-summary-v1",
        name="V2.11 parent import terminal summary",
        integrity_in_hash=False,
    )
    gate_evidence = summary.get("payload", {}).get("gate_evidence")
    if (
        summary.get("contract_id") != V211_CONTRACT_ID
        or summary.get("contract_sha256") != V211_CONTRACT_SHA256
        or summary.get("scientific_evidence") is not False
        or summary.get("evidence_scope") != "preregistered_parent_authority_import"
        or not isinstance(gate_evidence, Mapping)
        or gate_evidence.get("q_ref") != imported["q_ref"]
        or gate_evidence.get("selected_profile_id") != "nu-0.5"
        or gate_evidence.get("absolute_flow_utility_threshold") != 0.05617208967516696
        or summary.get("payload", {}).get("provider_calls") != 0
        or summary.get("payload", {}).get("provider_construction") is not False
        or summary.get("payload", {}).get("imported_effect_cells") != 0
        or summary.get("payload", {}).get("imported_p95_authorities") != []
    ):
        raise PilotV2111ParentImportError(
            "V2.11 parent import summary semantics drifted"
        )

    stage_raw, stage = _read_json(
        root,
        V211_PARENT_IMPORT_STAGE_RECEIPT_PATH,
        name="V2.11 parent import stage receipt",
    )
    _verify_seal(
        stage,
        schema_version="finevo-pilot-stage-receipt-v2",
        name="V2.11 parent import stage receipt",
        integrity_in_hash=False,
    )
    if (
        stage.get("contract_id") != V211_CONTRACT_ID
        or stage.get("contract_sha256") != V211_CONTRACT_SHA256
        or stage.get("stage_id") != "parent-import"
        or stage.get("status") != "complete"
        or stage.get("terminal") is not True
        or stage.get("denominator_terminal") is not True
        or stage.get("registered_run_count") != 1
        or stage.get("complete_cell_count") != 1
        or stage.get("status_counts") != {"complete": 1}
        or stage.get("go") is not True
        or stage.get("execution_progression_go") is not True
    ):
        raise PilotV2111ParentImportError("V2.11 parent import stage receipt drifted")
    return (
        {
            "q_ref": imported["q_ref"],
            "selected_utility_profile": _json_copy(
                imported["selected_utility_profile"]
            ),
            "stage0_absolute_flow_utility_threshold": _json_copy(
                imported["stage0_absolute_flow_utility_threshold"]
            ),
            "source_bindings": _json_copy(imported["source_bindings"]),
        },
        {
            "parent_import_receipt": _binding(
                V211_PARENT_IMPORT_RECEIPT_PATH,
                receipt_raw,
                receipt,
            ),
            "parent_import_summary": _binding(
                V211_PARENT_IMPORT_SUMMARY_PATH,
                summary_raw,
                summary,
            ),
            "parent_import_stage_receipt": _binding(
                V211_PARENT_IMPORT_STAGE_RECEIPT_PATH,
                stage_raw,
                stage,
            ),
        },
    )


def _verify_provider_catalog(
    value: Mapping[str, Any],
    *,
    model_id: str,
    name: str,
) -> None:
    _verify_field_hash(value, field="receipt_sha256", name=name)
    rows = value.get("rows")
    profile = _MODEL_SOURCES[model_id]
    if (
        value.get("schema_version") != "finevo-provider-catalog-receipt-v1"
        or value.get("contract_sha256") != V211_CONTRACT_SHA256
        or value.get("paid_completions") != 0
        or value.get("status") != "pass"
        or not isinstance(rows, list)
        or len(rows) != 1
        or rows[0].get("profile_id") != model_id
        or rows[0].get("provider_name") != "OpenAI-direct"
        or rows[0].get("served_snapshot") != profile["served_model"]
        or rows[0].get("transport") != "openai"
        or rows[0].get("status") != "pass"
        or not all(rows[0].get("document_checks", {}).values())
    ):
        raise PilotV2111ParentImportError(f"{name} semantics drifted")


def _verify_capability_stage_receipt(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = V211_RAW_ROOT / "capability-gate/stage_receipt.json"
    raw, value = _read_json(
        root,
        path,
        name="V2.11 capability stage receipt",
    )
    _verify_seal(
        value,
        schema_version="finevo-pilot-stage-receipt-v2",
        name="V2.11 capability stage receipt",
        integrity_in_hash=False,
    )
    if (
        value.get("contract_id") != V211_CONTRACT_ID
        or value.get("contract_sha256") != V211_CONTRACT_SHA256
        or value.get("stage_id") != "capability-gate"
        or value.get("status") != "complete"
        or value.get("terminal") is not True
        or value.get("denominator_terminal") is not True
        or value.get("registered_run_count") != 2
        or value.get("complete_cell_count") != 2
        or value.get("status_counts") != {"complete": 2}
        or value.get("go") is not True
        or value.get("go_models") != ["gpt52_main", "gpt56_diagnostic"]
        or value.get("execution_progression_go") is not True
    ):
        raise PilotV2111ParentImportError("V2.11 capability stage receipt drifted")
    return value, _binding(path, raw, value)


def _verify_capability_model(
    root: Path,
    *,
    model_id: str,
    stage_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _MODEL_SOURCES[model_id]
    catalog_raw, catalog = _read_json(
        root,
        source["capability_catalog"],
        name=f"V2.11 {model_id} capability provider catalog",
    )
    _verify_provider_catalog(
        catalog,
        model_id=model_id,
        name=f"V2.11 {model_id} capability provider catalog",
    )

    capability_raw, capability = _read_json(
        root,
        source["capability_payload"],
        name=f"V2.11 {model_id} capability artifact",
    )
    envelope = {
        "run_id": _capability_run_id(model_id),
        "artifact_sha256": canonical_sha256(capability),
        "payload": capability,
    }
    try:
        validated = _capability_rows(
            model_id=model_id,
            envelope=envelope,
            contract_sha256=V211_CONTRACT_SHA256,
        )
    except PilotV211GateError as exc:
        raise PilotV2111ParentImportError(
            f"V2.11 {model_id} capability validation failed: {exc}"
        ) from exc
    expected_total_tokens = (
        source["capability_prompt_tokens"] + source["capability_completion_tokens"]
    )
    actual_usage = validated["actual_usage"]
    if (
        validated["run_id"] != _capability_run_id(model_id)
        or validated["capability_pass"] is not True
        or actual_usage.get("prompt_tokens") != source["capability_prompt_tokens"]
        or actual_usage.get("completion_tokens")
        != source["capability_completion_tokens"]
        or actual_usage.get("total_tokens") != expected_total_tokens
        or not math.isclose(
            float(actual_usage.get("cost_usd", math.nan)),
            float(source["capability_cost_usd"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or capability.get("schema_version") != CAPABILITY_SCHEMA_VERSION
        or capability.get("taskset_sha256") != CAPABILITY_TASKSET_SHA256
        or capability.get("provider_model") != source["runtime_model"]
        or capability.get("pass") is not True
        or capability.get("provider_failure_count") != 0
        or capability.get("parse_failure_count") != 0
        or capability.get("recovered_parse_count") != 0
        or capability.get("strict_parse_count") != 30
        or capability.get("truncation_count") != 0
        or capability.get("category_totals", {})
        .get("utility-ranking", {})
        .get("registered_correct")
        != 12
        or capability.get("category_totals", {})
        .get("rule-application", {})
        .get("registered_correct")
        != 12
        or capability.get("category_totals", {})
        .get("rule-proposal", {})
        .get("registered_correct")
        != 6
    ):
        raise PilotV2111ParentImportError(
            f"V2.11 {model_id} capability semantics drifted"
        )

    gate_raw, gate = _read_json(
        root,
        source["capability_gate"],
        name=f"V2.11 {model_id} capability gate receipt",
    )
    if gate != {
        "capability_pass": True,
        "capability_status": "pass",
        "go": True,
        "interface_pass": True,
        "preflight_run": None,
        "reason": None,
    }:
        raise PilotV2111ParentImportError(
            f"V2.11 {model_id} capability gate receipt drifted"
        )

    summary_raw, summary = _read_json(
        root,
        source["capability_summary"],
        name=f"V2.11 {model_id} capability summary",
    )
    _verify_seal(
        summary,
        schema_version="finevo-pilot-terminal-summary-v1",
        name=f"V2.11 {model_id} capability summary",
        integrity_in_hash=False,
    )
    if (
        summary.get("contract_id") != V211_CONTRACT_ID
        or summary.get("contract_sha256") != V211_CONTRACT_SHA256
        or summary.get("scientific_evidence") is not False
        or summary.get("evidence_scope") != "preregistered_task_capability_gate"
        or summary.get("run_spec", {}).get("run_id") != _capability_run_id(model_id)
        or summary.get("payload", {}).get("capability") != capability
        or summary.get("payload", {}).get("gate_evidence") != gate
        or summary.get("provenance", {}).get("git_tag") != V211_SCIENCE_TAG
        or summary.get("provenance", {}).get("resolved_git_commit")
        != V211_SCIENCE_COMMIT
    ):
        raise PilotV2111ParentImportError(
            f"V2.11 {model_id} capability summary drifted"
        )

    projection = {
        "model_id": model_id,
        "run_id": _capability_run_id(model_id),
        "runtime_model": source["runtime_model"],
        "requested_model": source["requested_model"],
        "served_model": source["served_model"],
        "taskset_sha256": CAPABILITY_TASKSET_SHA256,
        "historical_source_calls": 30,
        "action_sample_count": 24,
        "semantic_sample_count": 6,
        "category_totals": _json_copy(capability["category_totals"]),
        "checks": _json_copy(capability["checks"]),
        "interface_gate": _json_copy(capability["interface_gate"]),
        "capability_assessment": _json_copy(capability["capability_assessment"]),
        "prompt_tier_gate": _json_copy(capability["prompt_tier_gate"]),
        "actual_usage": _json_copy(validated["actual_usage"]),
        "samples": _json_copy(validated["samples"]),
        "usage_rows": _json_copy(validated["usage_rows"]),
        "provider_failure_count": 0,
        "parse_failure_count": 0,
        "recovered_parse_count": 0,
        "strict_parse_count": 30,
        "truncation_count": 0,
        "capability_pass": True,
        "interface_pass": True,
        "stage_receipt_content_sha256": stage_receipt["integrity"]["content_sha256"],
    }
    bindings = {
        "provider_catalog": _binding(
            source["capability_catalog"],
            catalog_raw,
            catalog,
        ),
        "capability": _binding(
            source["capability_payload"],
            capability_raw,
            capability,
        ),
        "gate_receipt": _binding(
            source["capability_gate"],
            gate_raw,
            gate,
        ),
        "summary": _binding(
            source["capability_summary"],
            summary_raw,
            summary,
        ),
    }
    return projection, bindings


def _verify_preflight_model(
    root: Path,
    *,
    model_id: str,
    attestation: Mapping[str, Any],
) -> dict[str, Any]:
    source = _MODEL_SOURCES[model_id]
    catalog_raw, catalog = _read_json(
        root,
        source["preflight_catalog"],
        name=f"V2.11 {model_id} preflight provider catalog",
    )
    _verify_provider_catalog(
        catalog,
        model_id=model_id,
        name=f"V2.11 {model_id} preflight provider catalog",
    )

    intent_raw, intent = _read_json(
        root,
        source["preflight_intent"],
        name=f"V2.11 {model_id} preflight run intent",
    )
    _verify_field_hash(
        intent,
        field="intent_sha256",
        name=f"V2.11 {model_id} preflight run intent",
    )
    if (
        intent.get("schema_version")
        != "finevo-v2.11-long-context-preflight-run-intent-v1"
        or intent.get("pilot_contract_hash") != V211_CONTRACT_SHA256
        or intent.get("checkpoint_purpose") != "v2.11-long-context-preflight"
        or intent.get("checkpoint_schema_version") != "finevo-pilot-checkpoint-v4"
        or intent.get("num_agents") != 2
        or intent.get("completed_months") != 12
        or intent.get("provider_call_count") != 32
        or intent.get("action_call_count") != 24
        or intent.get("semantic_call_count") != 8
        or intent.get("prompt_tier_ceiling_tokens") != 200_000
        or intent.get("prompt_token_upper_bound_method") != "utf8-bytes-plus-256-v1"
    ):
        raise PilotV2111ParentImportError(
            f"V2.11 {model_id} preflight run intent drifted"
        )

    failure_raw, failure = _read_json(
        root,
        source["preflight_failure"],
        name=f"V2.11 {model_id} preflight failure",
    )
    manifest_raw, failure_manifest = _read_json(
        root,
        source["preflight_failure_manifest"],
        name=f"V2.11 {model_id} preflight failure manifest",
    )
    _verify_field_hash(
        failure_manifest,
        field="manifest_sha256",
        name=f"V2.11 {model_id} preflight failure manifest",
    )
    if (
        failure_manifest.get("schema_version") != "verified-failure-receipt-v1"
        or failure_manifest.get("status") != "failed"
        or failure_manifest.get("failure_file") != "failure.json"
        or failure_manifest.get("failure_sha256") != _sha256(failure_raw)
        or failure_manifest.get("failure_size_bytes") != len(failure_raw)
    ):
        raise PilotV2111ParentImportError(
            f"V2.11 {model_id} preflight failure manifest drifted"
        )
    snapshot = failure.get("budget_snapshot")
    config = failure.get("config")
    provenance = failure.get("provenance")
    if (
        failure.get("schema_version") != "verified-failure-receipt-v1"
        or failure.get("status") != "failed"
        or failure.get("error", {}).get("type") != "VerifiedRunError"
        or failure.get("error", {}).get("message")
        != source["preflight_failure_message"]
        or failure.get("partial_streams_persisted") is not False
        or failure.get("git") != {"commit": V211_SCIENCE_COMMIT, "dirty": False}
        or not isinstance(snapshot, Mapping)
        or snapshot.get("completed_calls") != 0
        or snapshot.get("active_calls") != 0
        or snapshot.get("completions") != []
        or snapshot.get("active_reservations") != []
        or snapshot.get("accounted_usage") != _ZERO_USAGE
        or snapshot.get("effective_usage") != _ZERO_USAGE
        or snapshot.get("reserved_usage") != _ZERO_USAGE
        or not isinstance(config, Mapping)
        or config.get("contract_id") != V211_CONTRACT_ID
        or config.get("contract_sha256") != V211_CONTRACT_SHA256
        or config.get("provider_call_journals") != []
        or config.get("projection", {}).get("completions") != 32
        or config.get("projection", {}).get("cost_usd")
        != source["preflight_cost_cap_usd"]
        or config.get("run_specs", [{}])[0].get("run_id") != _preflight_run_id(model_id)
        or not isinstance(provenance, Mapping)
        or provenance.get("contract_id") != V211_CONTRACT_ID
        or provenance.get("contract_sha256") != V211_CONTRACT_SHA256
        or provenance.get("scientific_evidence") is not False
        or provenance.get("evidence_use") != "failure denominator and audit only"
        or provenance.get("paid_provenance", {}).get("release_attestation")
        != attestation
    ):
        raise PilotV2111ParentImportError(
            f"V2.11 {model_id} preflight zero-call failure drifted"
        )
    return {
        "model_id": model_id,
        "run_id": _preflight_run_id(model_id),
        "status": "failed",
        "provider_calls": 0,
        "cost_usd": 0.0,
        "failure_reason": source["preflight_failure_message"],
        "bindings": {
            "provider_catalog": _binding(
                source["preflight_catalog"], catalog_raw, catalog
            ),
            "run_intent": _binding(source["preflight_intent"], intent_raw, intent),
            "failure": _binding(source["preflight_failure"], failure_raw, failure),
            "failure_manifest": _binding(
                source["preflight_failure_manifest"],
                manifest_raw,
                failure_manifest,
            ),
        },
    }


def _audit_parent_release(parent_science_root: str | Path) -> dict[str, Any]:
    root = _strict_root(parent_science_root, name="V2.11 science source")
    git = _verify_git_release(root)
    _, contract_binding, specs = _verify_contract(root)
    attestation, attestation_binding = _verify_attestation(root)
    _, run_binding = _verify_run_ledger(root, specs)
    _, budget_binding = _verify_budget_ledger(root)
    calibration, calibration_bindings = _verify_calibration_source(root)
    capability_stage, capability_stage_binding = _verify_capability_stage_receipt(root)
    capabilities: dict[str, dict[str, Any]] = {}
    capability_bindings: dict[str, dict[str, Any]] = {}
    preflight_failures: dict[str, dict[str, Any]] = {}
    for model_id in sorted(_MODEL_SOURCES):
        projection, bindings = _verify_capability_model(
            root,
            model_id=model_id,
            stage_receipt=capability_stage,
        )
        capabilities[model_id] = projection
        capability_bindings[model_id] = bindings
        preflight_failures[model_id] = _verify_preflight_model(
            root,
            model_id=model_id,
            attestation=attestation,
        )
    for path in V211_EXPECTED_ABSENT_PATHS:
        _verify_expected_absent(
            root,
            path,
            name=f"V2.11 expected-absent {path.as_posix()}",
        )

    manifest = _seal(
        {
            "schema_version": V2111_SOURCE_MANIFEST_SCHEMA_VERSION,
            "parent_release": {
                "root_hint": "../finevo-pilot-v2-11-science",
                "contract_id": V211_CONTRACT_ID,
                "contract_sha256": V211_CONTRACT_SHA256,
                **git,
                "publication_status": "immutable-no-go",
                "contract": contract_binding,
                "release_attestation": attestation_binding,
                "run_ledger": {
                    **run_binding,
                    "internal_sha256": V211_RUN_LEDGER_SHA256,
                    "event_count": V211_RUN_LEDGER_EVENT_COUNT,
                    "event_head_sha256": V211_RUN_LEDGER_EVENT_HEAD,
                    "run_count": 136,
                },
                "budget_ledger": {
                    **budget_binding,
                    "internal_sha256": V211_BUDGET_LEDGER_SHA256,
                    "event_count": V211_BUDGET_LEDGER_EVENT_COUNT,
                    "event_head_sha256": V211_BUDGET_LEDGER_EVENT_HEAD,
                    "run_count": 5,
                },
            },
            "calibration_source": {
                **calibration_bindings,
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
                "taskset_sha256": CAPABILITY_TASKSET_SHA256,
                "stage_receipt": capability_stage_binding,
                "models": {
                    model_id: {
                        **capability_bindings[model_id],
                        "run_id": capabilities[model_id]["run_id"],
                        "runtime_model": capabilities[model_id]["runtime_model"],
                        "historical_source_calls": 30,
                        "action_sample_count": 24,
                        "semantic_sample_count": 6,
                        "actual_usage": capabilities[model_id]["actual_usage"],
                        "capability_pass": True,
                        "interface_pass": True,
                        "scientific_evidence": False,
                    }
                    for model_id in sorted(capabilities)
                },
            },
            "failed_preflight_source": {
                "models": {
                    model_id: {
                        "run_id": preflight_failures[model_id]["run_id"],
                        "status": "failed",
                        "provider_calls": 0,
                        "cost_usd": 0.0,
                        "failure_reason": preflight_failures[model_id][
                            "failure_reason"
                        ],
                        **preflight_failures[model_id]["bindings"],
                    }
                    for model_id in sorted(preflight_failures)
                },
                "expected_absent": [
                    path.as_posix() for path in V211_EXPECTED_ABSENT_PATHS
                ],
            },
            "terminal_denominator": {
                "registered_cells": 136,
                "status_counts": dict(V211_EXPECTED_STATUS_COUNTS),
                "stage_status_counts": {
                    stage_id: dict(counts)
                    for stage_id, counts in (V211_EXPECTED_STAGE_STATUS_COUNTS.items())
                },
                "all_cells_terminal": True,
                "scientific_matrix_complete": False,
                "post_gate_authority_created": False,
                "stage_receipt_created": False,
            },
            "cumulative_parent_budget_debit": (_parent_budget_debit().to_dict()),
            "import_policy": {
                **_ZERO_PROVIDER_POLICY,
                "imported_calibration_wrappers": 1,
                "imported_capability_wrappers": 2,
                "historical_capability_calls": 60,
                "historical_preflight_calls": 0,
                "historical_effect_cells_imported": 0,
                "claim_boundary": (
                    "Calibration and capability/interface gate evidence only; "
                    "no V2.11 A-D or cross-model effect, no preflight sample, "
                    "and no observed-p95 authority."
                ),
            },
        }
    )
    return {
        "parent_root": root,
        "manifest": manifest,
        "calibration": calibration,
        "capabilities": capabilities,
        "preflight_failures": preflight_failures,
    }


def build_v2111_source_manifest(
    *,
    parent_science_root: str | Path,
) -> dict[str, Any]:
    """Render the exact tracked source manifest from immutable V2.11 bytes."""

    return _audit_parent_release(parent_science_root)["manifest"]


def _load_tracked_source_manifest(repo_root: Path) -> dict[str, Any]:
    _, raw = _guarded_file(
        repo_root,
        V2111_SOURCE_MANIFEST_PATH,
        name="tracked V2.11.1 parent source manifest",
    )
    if _sha256(raw) != V2111_SOURCE_MANIFEST_FILE_SHA256:
        raise PilotV2111ParentImportError(
            "tracked V2.11.1 parent source manifest file hash drifted"
        )
    value = _strict_json(
        raw,
        name="tracked V2.11.1 parent source manifest",
    )
    _verify_seal(
        value,
        schema_version=V2111_SOURCE_MANIFEST_SCHEMA_VERSION,
        name="tracked V2.11.1 parent source manifest",
    )
    if (
        value.get("integrity", {}).get("content_sha256")
        != V2111_SOURCE_MANIFEST_CONTENT_SHA256
    ):
        raise PilotV2111ParentImportError(
            "tracked V2.11.1 parent source manifest content hash drifted"
        )
    return value


def _resolve_parent_root(
    repo_root: Path,
    manifest: Mapping[str, Any],
    *,
    parent_science_root: str | Path | None,
) -> Path:
    if parent_science_root is not None:
        return _strict_root(
            parent_science_root,
            name="V2.11 science source",
        )
    hint = manifest.get("parent_release", {}).get("root_hint")
    if not isinstance(hint, str) or not hint:
        raise PilotV2111ParentImportError(
            "V2.11.1 parent source root hint is malformed"
        )
    return _strict_root(
        repo_root / hint,
        name="V2.11 science source",
    )


def _audit_sources(
    *,
    repo_root: str | Path,
    parent_science_root: str | Path | None,
) -> dict[str, Any]:
    child_root = _strict_root(repo_root, name="V2.11.1 repository")
    tracked = _load_tracked_source_manifest(child_root)
    parent_root = _resolve_parent_root(
        child_root,
        tracked,
        parent_science_root=parent_science_root,
    )
    audit = _audit_parent_release(parent_root)
    if audit["manifest"] != tracked:
        raise PilotV2111ParentImportError(
            "V2.11 source bytes differ from the frozen V2.11.1 manifest"
        )
    audit["repo_root"] = child_root
    return audit


def verify_v2111_parent_sources(
    *,
    repo_root: str | Path,
    parent_science_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify all immutable parent sources without constructing a provider."""

    audit = _audit_sources(
        repo_root=repo_root,
        parent_science_root=parent_science_root,
    )
    return {
        "parent_root": str(audit["parent_root"]),
        "source_manifest": _json_copy(audit["manifest"]),
        "calibration": _json_copy(audit["calibration"]),
        "capabilities": _json_copy(audit["capabilities"]),
        "preflight_failures": _json_copy(audit["preflight_failures"]),
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
        raise PilotV2111ParentImportError(
            "child_contract_sha256 must be a lowercase SHA-256"
        )
    if child_git_tag != "pilot-v2.11.1-science":
        raise PilotV2111ParentImportError("child_git_tag must be pilot-v2.11.1-science")
    if _COMMIT_RE.fullmatch(child_git_commit) is None:
        raise PilotV2111ParentImportError(
            "child_git_commit must be a lowercase 40-hex commit"
        )
    return {
        "contract_id": "finevo-pilot-v2.11.1",
        "contract_sha256": child_contract_sha256,
        "git_tag": child_git_tag,
        "resolved_git_commit": child_git_commit,
    }


def _source_manifest_receipt_binding() -> dict[str, str]:
    return {
        "path": V2111_SOURCE_MANIFEST_PATH.as_posix(),
        "file_sha256": V2111_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": V2111_SOURCE_MANIFEST_CONTENT_SHA256,
    }


def _calibration_wrapper(
    audit: Mapping[str, Any],
    child: Mapping[str, str],
) -> dict[str, Any]:
    manifest = audit["manifest"]
    return _seal(
        {
            "schema_version": V2111_CALIBRATION_WRAPPER_SCHEMA_VERSION,
            "child_release": _json_copy(child),
            "parent_release": {
                "contract_id": V211_CONTRACT_ID,
                "contract_sha256": V211_CONTRACT_SHA256,
                "git_tag": V211_SCIENCE_TAG,
                "git_tag_object": V211_SCIENCE_TAG_OBJECT,
                "resolved_git_commit": V211_SCIENCE_COMMIT,
            },
            "source_manifest": _source_manifest_receipt_binding(),
            "source_artifacts": _json_copy(manifest["calibration_source"]),
            "calibration": _json_copy(audit["calibration"]),
            "provider_construction_current_attempt": False,
            "provider_calls_current_attempt": 0,
            "hosted_provider_calls_current_attempt": 0,
            "imported_effect_cells": 0,
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
    manifest = audit["manifest"]
    projection = audit["capabilities"][model_id]
    return _seal(
        {
            "schema_version": V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION,
            "child_release": _json_copy(child),
            "parent_release": {
                "contract_id": V211_CONTRACT_ID,
                "contract_sha256": V211_CONTRACT_SHA256,
                "git_tag": V211_SCIENCE_TAG,
                "git_tag_object": V211_SCIENCE_TAG_OBJECT,
                "resolved_git_commit": V211_SCIENCE_COMMIT,
            },
            "source_manifest": _source_manifest_receipt_binding(),
            "source_artifacts": _json_copy(
                manifest["capability_source"]["models"][model_id]
            ),
            "capability": _json_copy(projection),
            "provider_construction_current_attempt": False,
            "provider_calls_current_attempt": 0,
            "hosted_provider_calls_current_attempt": 0,
            "current_attempt_usage": dict(_ZERO_USAGE),
            "imported_effect_cells": 0,
            "imported_p95_authorities": [],
            "scientific_evidence": False,
            "evidence_scope": "preregistered_task_capability_gate",
            "evidence_use": (
                "Capability/interface gate and capability-derived p95 samples "
                "only; not treatment-effect or model-performance evidence."
            ),
        }
    )


def build_v2111_parent_import(
    *,
    repo_root: str | Path,
    child_contract_sha256: str,
    child_git_tag: str,
    child_git_commit: str,
    parent_science_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a compact child-bound receipt after replaying all parent checks."""

    child = _child_binding(
        child_contract_sha256=child_contract_sha256,
        child_git_tag=child_git_tag,
        child_git_commit=child_git_commit,
    )
    audit = _audit_sources(
        repo_root=repo_root,
        parent_science_root=parent_science_root,
    )
    return _seal(
        {
            "schema_version": V2111_PARENT_IMPORT_SCHEMA_VERSION,
            "child_release": child,
            "source_manifest": _source_manifest_receipt_binding(),
            "parent_release": {
                "contract_id": V211_CONTRACT_ID,
                "contract_sha256": V211_CONTRACT_SHA256,
                "science_tag": V211_SCIENCE_TAG,
                "science_tag_object": V211_SCIENCE_TAG_OBJECT,
                "resolved_git_commit": V211_SCIENCE_COMMIT,
                "release_attestation_sha256": (V211_RELEASE_ATTESTATION_SHA256),
                "run_ledger_sha256": V211_RUN_LEDGER_SHA256,
                "budget_ledger_sha256": V211_BUDGET_LEDGER_SHA256,
                "publication_status": "immutable-no-go",
            },
            "terminal_parent_denominator": _json_copy(
                audit["manifest"]["terminal_denominator"]
            ),
            "failed_preflight": {
                model_id: {
                    "run_id": row["run_id"],
                    "status": row["status"],
                    "provider_calls": row["provider_calls"],
                    "cost_usd": row["cost_usd"],
                    "failure_reason": row["failure_reason"],
                }
                for model_id, row in sorted(audit["preflight_failures"].items())
            },
            "expected_absent": [path.as_posix() for path in V211_EXPECTED_ABSENT_PATHS],
            "calibration_wrapper": _calibration_wrapper(audit, child),
            "capability_wrappers": {
                model_id: _capability_wrapper(
                    audit,
                    child,
                    model_id=model_id,
                )
                for model_id in sorted(audit["capabilities"])
            },
            "cumulative_parent_budget_debit": (_parent_budget_debit().to_dict()),
            "import_policy": {
                **_ZERO_PROVIDER_POLICY,
                "imported_calibration_wrappers": 1,
                "imported_capability_wrappers": 2,
                "historical_capability_calls": 60,
                "historical_preflight_calls": 0,
                "historical_effect_cells_imported": 0,
                "validation_before_provider_construction": True,
            },
            "scientific_evidence": False,
            "claim_boundary": (
                "V2.11.1 inherits calibration and two capability/interface "
                "gates only. V2.11 remains a 136-cell immutable no-go; no "
                "effect cell, preflight sample, or p95 authority is imported."
            ),
        }
    )


def _load_receipt(
    value: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _json_copy(dict(value))
    path = Path(value).expanduser().absolute()
    try:
        parent = _strict_root(
            path.parent,
            name="V2.11.1 parent import receipt parent",
        )
    except PilotV2111ParentImportError:
        raise
    _, raw = _guarded_file(
        parent,
        PurePosixPath(path.name),
        name="V2.11.1 parent import receipt",
    )
    return _strict_json(raw, name="V2.11.1 parent import receipt")


def verify_v2111_parent_import_receipt(
    receipt: Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path,
    child_contract_sha256: str,
    child_git_tag: str,
    child_git_commit: str,
    parent_science_root: str | Path | None = None,
) -> dict[str, Any]:
    """Replay source verification and require byte-semantic receipt equality."""

    observed = _load_receipt(receipt)
    _verify_seal(
        observed,
        schema_version=V2111_PARENT_IMPORT_SCHEMA_VERSION,
        name="V2.11.1 parent import receipt",
    )
    expected = build_v2111_parent_import(
        repo_root=repo_root,
        child_contract_sha256=child_contract_sha256,
        child_git_tag=child_git_tag,
        child_git_commit=child_git_commit,
        parent_science_root=parent_science_root,
    )
    if observed != expected:
        raise PilotV2111ParentImportError(
            "V2.11.1 parent import receipt differs from exact parent replay"
        )
    return observed


def validate_v2111_source_manifest(
    value: Mapping[str, Any],
    *,
    parent_science_root: str | Path,
) -> dict[str, Any]:
    """Require a manifest to equal a fresh replay of the immutable source."""

    observed = _json_copy(dict(value))
    _verify_seal(
        observed,
        schema_version=V2111_SOURCE_MANIFEST_SCHEMA_VERSION,
        name="V2.11.1 parent source manifest",
    )
    expected = build_v2111_source_manifest(parent_science_root=parent_science_root)
    if observed != expected:
        raise PilotV2111ParentImportError(
            "V2.11.1 parent source manifest differs from source replay"
        )
    return observed


def load_v2111_source_manifest(
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Load the tracked manifest after exact file/content verification."""

    root = _strict_root(repo_root, name="V2.11.1 repository")
    return _load_tracked_source_manifest(root)


def _safe_destination(
    repo_root: Path,
    destination: str | Path | None,
) -> Path:
    if destination is None:
        relative = V2111_DEFAULT_RECEIPT_PATH
    else:
        candidate = Path(destination)
        if candidate.is_absolute():
            try:
                relative = PurePosixPath(
                    *candidate.absolute().relative_to(repo_root).parts
                )
            except ValueError as exc:
                raise PilotV2111ParentImportError(
                    "receipt destination escaped V2.11.1 repository"
                ) from exc
        else:
            relative = PurePosixPath(candidate.as_posix())
    normalized = _normalized_relative(
        relative.as_posix(),
        required_top="experiment_results",
        name="V2.11.1 receipt destination",
    )
    path = repo_root.joinpath(*normalized.parts)
    current = repo_root
    for part in normalized.parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise PilotV2111ParentImportError("receipt destination contains a symlink")
    if path.is_symlink():
        raise PilotV2111ParentImportError("receipt destination is a symlink")
    return path


def _persist_exact_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise PilotV2111ParentImportError("receipt destination contains a symlink")
        current = current.parent
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise PilotV2111ParentImportError("receipt destination is a symlink")
        if path.read_bytes() != raw:
            raise PilotV2111ParentImportError(
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


def persist_v2111_parent_import(
    *,
    repo_root: str | Path,
    child_contract_sha256: str,
    child_git_tag: str,
    child_git_commit: str,
    parent_science_root: str | Path | None = None,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Persist only the compact receipt; never copy immutable parent raw."""

    root = _strict_root(repo_root, name="V2.11.1 repository")
    receipt = build_v2111_parent_import(
        repo_root=root,
        child_contract_sha256=child_contract_sha256,
        child_git_tag=child_git_tag,
        child_git_commit=child_git_commit,
        parent_science_root=parent_science_root,
    )
    path = _safe_destination(root, destination)
    _persist_exact_json(path, receipt)
    verified = verify_v2111_parent_import_receipt(
        path,
        repo_root=root,
        child_contract_sha256=child_contract_sha256,
        child_git_tag=child_git_tag,
        child_git_commit=child_git_commit,
        parent_science_root=parent_science_root,
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


def parent_budget_debit_for_v2111(
    contract: Any = None,
    *,
    repo_root: str | Path,
    parent_science_root: str | Path | None = None,
) -> ParentBudgetDebit:
    """Return the cumulative V2.11 debit only after exact parent replay."""

    contract_id = getattr(contract, "contract_id", None)
    if contract is not None and contract_id not in {
        "finevo-pilot-v2.11.1",
        "finevo-pilot-v2.11.1-prospective",
    }:
        raise PilotV2111ParentImportError("parent debit requires the V2.11.1 contract")
    _audit_sources(
        repo_root=repo_root,
        parent_science_root=parent_science_root,
    )
    return _parent_budget_debit()


def calibration_wrapper_from_v2111_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Structurally verify and extract the embedded calibration wrapper."""

    value = _json_copy(dict(receipt))
    _verify_seal(
        value,
        schema_version=V2111_PARENT_IMPORT_SCHEMA_VERSION,
        name="V2.11.1 parent import receipt",
    )
    wrapper = value.get("calibration_wrapper")
    if not isinstance(wrapper, Mapping):
        raise PilotV2111ParentImportError("V2.11.1 calibration wrapper is absent")
    _verify_seal(
        wrapper,
        schema_version=V2111_CALIBRATION_WRAPPER_SCHEMA_VERSION,
        name="V2.11.1 calibration wrapper",
    )
    return _json_copy(dict(wrapper))


def capability_wrappers_from_v2111_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Structurally verify and extract both embedded capability wrappers."""

    value = _json_copy(dict(receipt))
    _verify_seal(
        value,
        schema_version=V2111_PARENT_IMPORT_SCHEMA_VERSION,
        name="V2.11.1 parent import receipt",
    )
    wrappers = value.get("capability_wrappers")
    if not isinstance(wrappers, Mapping) or set(wrappers) != set(_MODEL_SOURCES):
        raise PilotV2111ParentImportError(
            "V2.11.1 capability wrapper denominator drifted"
        )
    result: dict[str, dict[str, Any]] = {}
    for model_id in sorted(_MODEL_SOURCES):
        wrapper = wrappers[model_id]
        if not isinstance(wrapper, Mapping):
            raise PilotV2111ParentImportError(
                f"V2.11.1 {model_id} capability wrapper is malformed"
            )
        _verify_seal(
            wrapper,
            schema_version=V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION,
            name=f"V2.11.1 {model_id} capability wrapper",
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
            or not isinstance(samples.get("action"), list)
            or len(samples["action"]) != 24
            or not isinstance(samples.get("semantic"), list)
            or len(samples["semantic"]) != 6
            or not isinstance(usage_rows, list)
            or len(usage_rows) != 30
            or sum(
                row.get("call_kind") == "action"
                for row in usage_rows
                if isinstance(row, Mapping)
            )
            != 24
            or sum(
                row.get("call_kind") == "semantic"
                for row in usage_rows
                if isinstance(row, Mapping)
            )
            != 6
            or wrapper.get("provider_construction_current_attempt") is not False
            or wrapper.get("provider_calls_current_attempt") != 0
            or wrapper.get("hosted_provider_calls_current_attempt") != 0
            or wrapper.get("current_attempt_usage") != _ZERO_USAGE
            or wrapper.get("scientific_evidence") is not False
        ):
            raise PilotV2111ParentImportError(
                f"V2.11.1 {model_id} capability wrapper semantics drifted"
            )
        result[model_id] = _json_copy(dict(wrapper))
    return result


def verified_v2111_capability_source(
    receipt: Mapping[str, Any] | str | Path,
    *,
    model_id: str,
    repo_root: str | Path,
    child_contract_sha256: str,
    child_git_tag: str,
    child_git_commit: str,
    parent_science_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a replay-verified raw V5 capability source for bootstrap/gates.

    The parent root is optional.  Normal execution resolves the immutable
    checkout from the tracked source manifest's ``root_hint`` so downstream
    stages need only the child repository and persisted parent receipt.
    """

    if model_id not in _MODEL_SOURCES:
        raise PilotV2111ParentImportError(
            f"unsupported inherited capability model {model_id!r}"
        )
    verified = verify_v2111_parent_import_receipt(
        receipt,
        repo_root=repo_root,
        child_contract_sha256=child_contract_sha256,
        child_git_tag=child_git_tag,
        child_git_commit=child_git_commit,
        parent_science_root=parent_science_root,
    )
    wrappers = capability_wrappers_from_v2111_receipt(verified)
    audit = _audit_sources(
        repo_root=repo_root,
        parent_science_root=parent_science_root,
    )
    source = _MODEL_SOURCES[model_id]
    relative = source["capability_payload"]
    raw, payload = _read_json(
        audit["parent_root"],
        relative,
        name=f"V2.11 {model_id} replay-verified capability source",
    )
    summary_raw, summary = _read_json(
        audit["parent_root"],
        source["capability_summary"],
        name=f"V2.11 {model_id} replay-verified capability summary",
    )
    del summary_raw
    binding = audit["manifest"]["capability_source"]["models"][model_id]["capability"]
    if (
        binding.get("path") != relative.as_posix()
        or binding.get("byte_size") != len(raw)
        or binding.get("file_sha256") != _sha256(raw)
        or summary.get("payload", {}).get("capability") != payload
        or summary.get("run_spec", {}).get("run_id") != _capability_run_id(model_id)
    ):
        raise PilotV2111ParentImportError(
            f"V2.11 {model_id} raw capability source replay drifted"
        )
    wrapper = wrappers[model_id]
    return _seal(
        {
            "schema_version": V2111_VERIFIED_CAPABILITY_SOURCE_SCHEMA_VERSION,
            "model_id": model_id,
            "run_id": _capability_run_id(model_id),
            "spec": _json_copy(summary["run_spec"]),
            "payload": _json_copy(payload),
            "artifact_sha256": canonical_sha256(payload),
            "source": {
                "relative_path": relative.as_posix(),
                "resolved_path": str(audit["parent_root"].joinpath(*relative.parts)),
                "byte_size": len(raw),
                "file_sha256": _sha256(raw),
            },
            "parent_release": {
                "contract_sha256": V211_CONTRACT_SHA256,
                "git_tag": V211_SCIENCE_TAG,
                "git_tag_object": V211_SCIENCE_TAG_OBJECT,
                "resolved_git_commit": V211_SCIENCE_COMMIT,
            },
            "child_release": _json_copy(verified["child_release"]),
            "wrapper_content_sha256": wrapper["integrity"]["content_sha256"],
            "provider_construction_during_verification": False,
            "provider_calls_during_verification": 0,
            "scientific_evidence": False,
            "evidence_use": (
                "Raw V5 same-model capability replay for zero-call import and "
                "post-gate construction only."
            ),
        }
    )


def verified_v2111_inherited_capability_binding(
    receipt: Mapping[str, Any] | str | Path,
    *,
    model_id: str,
    repo_root: str | Path,
    child_contract_sha256: str,
    child_git_tag: str,
    child_git_commit: str,
    parent_science_root: str | Path | None = None,
) -> dict[str, Any]:
    """Replay-verify the receipt and return one gate-consumable wrapper."""

    if model_id not in _MODEL_SOURCES:
        raise PilotV2111ParentImportError(
            f"unsupported inherited capability model {model_id!r}"
        )
    verified = verify_v2111_parent_import_receipt(
        receipt,
        repo_root=repo_root,
        child_contract_sha256=child_contract_sha256,
        child_git_tag=child_git_tag,
        child_git_commit=child_git_commit,
        parent_science_root=parent_science_root,
    )
    wrapper = capability_wrappers_from_v2111_receipt(verified)[model_id]
    return {
        "model_id": model_id,
        "wrapper_content_sha256": wrapper["integrity"]["content_sha256"],
        "payload": wrapper,
        "provider_construction_during_verification": False,
        "provider_calls_during_verification": 0,
    }


# Receipt-oriented aliases preserve the naming pattern used by prior adapters.
build_v2111_parent_import_receipt = build_v2111_parent_import
verify_v2111_parent_import = verify_v2111_parent_import_receipt


__all__ = [
    "PilotV2111ParentImportError",
    "V2111_CALIBRATION_WRAPPER_SCHEMA_VERSION",
    "V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION",
    "V2111_VERIFIED_CAPABILITY_SOURCE_SCHEMA_VERSION",
    "V2111_DEFAULT_RECEIPT_PATH",
    "V2111_PARENT_IMPORT_SCHEMA_VERSION",
    "V2111_SOURCE_MANIFEST_CONTENT_SHA256",
    "V2111_SOURCE_MANIFEST_FILE_SHA256",
    "V2111_SOURCE_MANIFEST_PATH",
    "V2111_SOURCE_MANIFEST_SCHEMA_VERSION",
    "build_v2111_parent_import",
    "build_v2111_parent_import_receipt",
    "build_v2111_source_manifest",
    "calibration_wrapper_from_v2111_receipt",
    "capability_wrappers_from_v2111_receipt",
    "load_v2111_source_manifest",
    "parent_budget_debit_for_v2111",
    "persist_v2111_parent_import",
    "validate_v2111_source_manifest",
    "verified_v2111_inherited_capability_binding",
    "verified_v2111_capability_source",
    "verify_v2111_parent_import",
    "verify_v2111_parent_import_receipt",
    "verify_v2111_parent_sources",
]
