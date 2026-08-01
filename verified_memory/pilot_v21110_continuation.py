"""Immutable V2.11.9 no-go lineage for the V2.11.10 continuation.

This module is intentionally provider-free.  It binds the complete terminal
V2.11.9 failure as lineage and keeps the V2.11.5 science checkout as the only
dispatch-authority root.  No V2.11.9 row is resumed, reclassified, copied as an
effect, or charged as a fresh completion.

The functions here are read-only except for returning newly constructed JSON
objects.  Writing a V2.11.10 receipt belongs to the orchestrator's atomic stage
transaction, not to this provenance layer.
"""

from __future__ import annotations

import ast
from collections import Counter
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping

from .pilot_budget import ParentBudgetDebit, PilotBudgetLedger
from .pilot_contract import PilotContract, canonical_sha256, load_pilot_contract
from . import pilot_v2117_continuation as v2117
from . import pilot_v2119_continuation as v2119


V21110_CONTRACT_ID = "finevo-pilot-v2.11.10"
V21110_SCIENCE_TAG = "pilot-v2.11.10-science"
V21110_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_11_10_source_manifest.json"
)
V21110_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.11.10/raw")
V21110_PARENT_IMPORT_SCHEMA_VERSION = "finevo-pilot-v2.11.10-parent-import-v1"
V21110_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.11.10-source-manifest-v1"
V2115_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.11.5-source-manifest-v1"
V21110_CURRENT_AUTHORITY_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.10-continuation-observed-p95-authority-v1"
)
V21110_CURRENT_PROJECTION_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.10-continuation-projection-v1"
)
V21110_ACCEPTANCE_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.10-scientific-dispatch-acceptance-v1"
)
V21110_ACCEPTANCE_FILENAME = "scientific_dispatch_acceptance.json"
V21110_PROFILE_PATH = PurePosixPath("data/profiles.json")
V21110_PROFILE_FILE_SHA256 = (
    "1bc90a92ef8e32f3da6e474f787207b79b1c82cc0b7b13c5ea3bd6cd1439b223"
)

V2119_CONTRACT_ID = "finevo-pilot-v2.11.9"
V2119_SCIENCE_TAG = "pilot-v2.11.9-science"
V2119_SCIENCE_COMMIT = "d850902af6218c72a6b0e71275c62c81c9143fb9"
V2119_SCIENCE_TAG_OBJECT = "f0af244b64a69b3ee4571452df6d3611fd8c6220"
V2119_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_11_9.yaml")
V2119_CONTRACT_SHA256 = (
    "ec16563bf906b8f6c1492d2a30f291d2c849cd639c2f314e7a1c8ac619e3fa3f"
)
V2119_CONTRACT_FILE_SHA256 = (
    "160f82b4bd57ba3ccc8ae711542ffc1d41f26503b48b80031b277b1f199c5d47"
)
V2119_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_11_9_source_manifest.json"
)
V2119_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.11.9-source-manifest-v1"
V2119_SOURCE_MANIFEST_FILE_SHA256 = (
    "609adf9d12543b4caa7adb0cbddb8c8a9073a10f689adf52a8670608d16e9cb1"
)
V2119_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "36a790fe5edd6269218d6010046ec9293c3c418d8bc58a4dd5d89a6a70a547d6"
)
V2119_FAILED_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.11.9/raw")

V2119_RUN_LEDGER_FILE_SHA256 = (
    "ea1dbe11bdb229a89eae21e37c58bbd47283599de17d8402edec952e39482cc1"
)
V2119_RUN_LEDGER_SHA256 = (
    "b2891fb152825cac846955b9c2fe4a041e80eab8cbebef9bc4d861d2313fc923"
)
V2119_RUN_EVENT_COUNT = 90
V2119_RUN_EVENT_HEAD = (
    "3b255d0cfd972b5491f01588d3987006e5babc27ffafcff0dddd6c7c2f880bd6"
)
V2119_BUDGET_LEDGER_FILE_SHA256 = (
    "f4164290d4a66f56843bb3bb47bee2662de77fec98e911d601c51f9f06374267"
)
V2119_BUDGET_LEDGER_SHA256 = (
    "02adeb470b823664c67d09cd34df8787a68760e6270f46b59cca204701e3465d"
)
V2119_BUDGET_EVENT_COUNT = 77
V2119_BUDGET_EVENT_HEAD = (
    "ab37a7c425d7b5e16d96dd6fcb75fa1af95cd93f96a7bb0bb2182857a53eda51"
)
V2119_LEDGER_CELL_COUNT = 87
V2119_REMAINING_SCIENCE_CELL_COUNT = 86
V2119_CURRENT_ACTUAL_STORAGE_BYTES = 800_162
V2119_CUMULATIVE_COST_USD = 63.1196450625
V2119_CUMULATIVE_COMPLETIONS = 3_440
V2119_CUMULATIVE_STORAGE_BYTES = 270_993_662
V21110_PARENT_DEBIT_RECORD_SHA256 = (
    "5e0c39817c32c845c2f771a02320c55e85e9a6bfb5f3e705046b822593b4c592"
)

V2119_FAILURE_MESSAGE_BY_MODEL = {
    "gpt52_main": (
        "source-backed observed p95 source authority differs for "
        "openai/gpt-5.2-2025-12-11::action"
    ),
    "gpt56_diagnostic": (
        "source-backed observed p95 source authority differs for "
        "openai/gpt-5.6-sol::action"
    ),
}
V2119_FAILURE_ERROR_TYPE = "ValueError"
V2119_PARENT_IMPORT_RECEIPT_CONTENT_SHA256 = (
    "72a05abc25081990f65fba15cd6112a7c5a48f3878cc0852b50e5b92fb3bc7d2"
)
V2119_PARENT_IMPORT_RECEIPT_FILE_SHA256 = (
    "8659e5de1c5f79467bc42c66706aca8cb30f243a0917c4f94ec77c3367631bfb"
)
V2119_PARENT_STAGE_RECEIPT_CONTENT_SHA256 = (
    "2d5e4cbb2e08ef231537646e2ad31a2f432a0349c585b3645d544abdf9126754"
)
V2119_EXPERIMENT_D_STAGE_RECEIPT_CONTENT_SHA256 = (
    "111d77c6525b5d17cf3ad65da45649221f7dd5a17207fcbdf68c979f01cc01b6"
)
V2119_EXPERIMENT_B_STAGE_RECEIPT_CONTENT_SHA256 = (
    "cc710beb91422372cec53e77996ad53572006c294b7b24efbcd5fdd47fca56f1"
)
V2119_CROSS_MODEL_STAGE_RECEIPT_CONTENT_SHA256 = (
    "98d07a4db01d1e1011544e8b3e814471ec3e65f8121fa574520f07c02eed7e18"
)
V2119_ACCEPTANCE_FILE_SHA256 = (
    "f9d49c118b319e7612c0e84c428e5fdfa1571c33915d160afa6f93036864be94"
)
V2119_ACCEPTANCE_CONTENT_SHA256 = (
    "2cac2d2b5f9f9e97054473c982e94df8ae0ed823852a03c39f1cc59d9f5a4e06"
)
V2119_RELEASE_ATTESTATION_SHA256 = (
    "1b02503d7a7488b3d4b166413a5b4fd7ee376fe175e58cbff0b080d52ef765e9"
)
V2119_RELEASE_ATTESTATION_FILE_SHA256 = (
    "15a0a4b519466fc84fd08340e4643bd5a4b5fe7c7abdf38baee6601058397b38"
)
V2119_LAUNCH_INPUT_SHA256 = (
    "ad4bf87f65582f63ba3a675891b7cb9e2ef29a88ca2efa5e889a40461512c16d"
)
V2119_LAUNCH_INPUT_FILE_SHA256 = (
    "c8d92522fc02ce7f2fb4905cdd28f6e6ce759eeb29cce7f9221b4adbd5369dfa"
)

# These critical receipts are pinned individually, while the complete 90-file
# terminal namespace is bound below by its count, byte total, and inventory
# digest.  This preserves the failed parent's exact immutable denominator.
V2119_CRITICAL_RAW_FILE_SHA256: Mapping[str, str] = {
    "budget_ledger.json": V2119_BUDGET_LEDGER_FILE_SHA256,
    "cross-model/stage_receipt.json": (
        "fb25de2c70b619370b2e7a455e14bd17fe07f6a53b8480e716e29988000f928a"
    ),
    "experiment-b/stage_receipt.json": (
        "1278621c8a8d10f7e4d03bcdd46eb4647bb24eb61f615a978650b711e6cc93c9"
    ),
    "experiment-d/stage_receipt.json": (
        "5a66db97dbea5abb380a6ba43bdb632bf43c65226bb57b86b6fd4632185c2ff0"
    ),
    "parent-import/parent_import_receipt.json": (
        V2119_PARENT_IMPORT_RECEIPT_FILE_SHA256
    ),
    "parent-import/stage_receipt.json": (
        "49caccf869b8515baea84f5e016a53bd8bacdbb22aebc85a61324cd9799a3da4"
    ),
    "release_attestation.json": V2119_RELEASE_ATTESTATION_FILE_SHA256,
    "run_ledger.json": V2119_RUN_LEDGER_FILE_SHA256,
    "scientific_dispatch_acceptance.json": V2119_ACCEPTANCE_FILE_SHA256,
    "scientific_launch_input.json": V2119_LAUNCH_INPUT_FILE_SHA256,
}
V2119_COMPLETE_RAW_FILE_COUNT = 90
V2119_COMPLETE_RAW_STORAGE_BYTES = 1_219_229
V2119_COMPLETE_RAW_INVENTORY_SHA256 = (
    "268654caa5404b5e49729e418f4ebe1896f5b06bd52bba4de5a533021dd9e206"
)
V2119_EVIDENCE_RAW_FILE_COUNT = 89
V2119_EVIDENCE_RAW_STORAGE_BYTES = 1_219_091
V2119_EVIDENCE_RAW_INVENTORY_SHA256 = (
    "1f98a77a4262542e050003305600e07f0fef0e2e4e8eef8cc51a56fe9a8111bc"
)

V2115_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_11_5.yaml")
V2115_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_11_5_source_manifest.json"
)
V2115_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.11.5/raw")
V2115_ACCEPTANCE_FILE_SHA256 = (
    "10d9c591d7dea12dc0062fd59f091628f474e5be1720fa9840ddc9361fd6d72d"
)
V2115_ACCEPTANCE_CONTENT_SHA256 = (
    "41bae59ac6ac34182091aeb1720777f5c391b1f6d1e61df9cdb2d92e537894cf"
)
V2115_POST_GATE_FILE_SHA256 = (
    "08b33162c91a07b392bacefc40d6abee9d89633600608a3de798f171e427a35a"
)
V2115_POST_GATE_CONTENT_SHA256 = (
    "19e18c1641ecaf55e48340694e126416ff30f0b39307c66f5335c9c9e9a46abc"
)
V2115_CONTRACT_BINDING_SHA256 = (
    "a4808230e8aac15da5eb8bba0b844ed9d36e853ca9a2aafbebe23887a4a21d14"
)

_CURRENT_AUTHORITY_PATH = (
    V21110_RAW_ROOT / "parent-import/current_authority/post_gate_authority.json"
)

_PROVIDER_KEY_ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)
_ACCEPTANCE_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "go",
        "contract_id",
        "contract_sha256",
        "release",
        "raw_namespace",
        "denominator",
        "parent_import",
        "current_authority",
        "runner_configs",
        "budget_projection",
        "ledger_prefixes",
        "provider_boundary",
        "scientific_evidence",
        "claim_boundary",
        "integrity",
    }
)
_ACCEPTANCE_PROVIDER_BOUNDARY = {
    "credential_environment_names_checked": list(_PROVIDER_KEY_ENV_NAMES),
    "credential_values_present": False,
    "provider_construction": False,
    "provider_calls": 0,
    "provider_catalog_calls": 0,
    "hosted_provider_calls": 0,
    "hosted_cost_usd": 0.0,
    "validation_before_provider_construction": True,
}
_ACCEPTANCE_CLAIM_BOUNDARY = (
    "Pre-dispatch V2.11.10 integrity and budget acceptance only; "
    "no treatment outcome is created or imported."
)
_PARENT_IMPORT_BASE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "go",
        "contract_id",
        "contract_sha256",
        "release",
        "failed_release_no_go",
        "authority_release",
        "denominator_continuation",
        "cumulative_parent_budget_debit",
        "verified_terminal_bindings",
        "source_manifest_replay",
        "import_policy",
        "scientific_evidence",
        "claim_boundary",
        "integrity",
    }
)
_PARENT_IMPORT_PERSISTED_FIELDS = _PARENT_IMPORT_BASE_FIELDS | frozenset(
    {
        "source_manifest",
        "failed_terminal_no_go",
        "authority_import",
        "calibration_wrapper",
        "capability_authority",
        "dispatch_authority_source",
        "canonical_remaining_cell_mapping",
    }
)
_PARENT_IMPORT_POLICY = {
    "provider_construction": False,
    "provider_calls": 0,
    "hosted_provider_calls": 0,
    "hosted_cost_usd": 0.0,
    "decoded_completion_reuse": False,
    "imported_effect_cells": 0,
    "failed_raw_tree_copied": False,
    "authority_raw_tree_copied": False,
    "validation_before_provider_construction": True,
}
_PARENT_IMPORT_CLAIM_BOUNDARY = (
    "Immutable V2.11.9 terminal no-go lineage plus V2.11.5 dispatch "
    "authority only; no V2.11.9 outcome is resumed or reclassified."
)
_CURRENT_AUTHORITY_PROVIDER_BOUNDARY = {
    "provider_calls": 0,
    "hosted_provider_calls": 0,
    "hosted_cost_usd": 0.0,
    "provider_construction": False,
}
_CURRENT_AUTHORITY_CLAIM_BOUNDARY = (
    "V2.11.10 current-release dispatch-budget authority only; no decoded "
    "completion or A/C effect row is imported."
)
_CURRENT_AUTHORITY_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "contract_sha256",
        "release",
        "authority_release",
        "parent_import_content_sha256",
        "reservations",
        "stable_source_authorities",
        "provider_boundary",
        "scientific_evidence",
        "claim_boundary",
        "integrity",
    }
)
_CURRENT_PROJECTION_CLAIM_BOUNDARY = (
    "V2.11.10 current-release model/call-kind budget projection only; no "
    "provider completion or treatment outcome is created."
)
_CURRENT_PROJECTION_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "model_id",
        "runtime_model",
        "served_model",
        "projection",
        "bindings",
        "provider_calls",
        "provider_construction",
        "scientific_evidence",
        "claim_boundary",
        "integrity",
    }
)


class PilotV21110ContinuationError(RuntimeError):
    """Raised before V2.11.10 may construct a provider."""


def _json_copy(value: Any) -> Any:
    try:
        return v2119._json_copy(value)
    except v2119.PilotV2119ContinuationError as exc:
        raise PilotV21110ContinuationError(str(exc)) from exc


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        return v2119._strict_json(path, name=name)
    except v2119.PilotV2119ContinuationError as exc:
        raise PilotV21110ContinuationError(str(exc)) from exc


def _file_sha256(path: Path) -> str:
    try:
        return v2119._file_sha256(path)
    except (OSError, v2119.PilotV2119ContinuationError) as exc:
        raise PilotV21110ContinuationError(f"cannot hash {path}") from exc


def _real_root(value: str | Path, *, name: str) -> Path:
    lexical = Path(value).absolute()
    try:
        mode = lexical.lstat().st_mode
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise PilotV21110ContinuationError(f"{name} is unavailable") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise PilotV21110ContinuationError(
            f"{name} must be a real non-symlink directory"
        )
    return resolved


def _require_distinct_roots(**roots: Path) -> None:
    names = tuple(roots)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left = roots[left_name]
            right = roots[right_name]
            try:
                aliased = os.path.samefile(left, right)
            except OSError as exc:
                raise PilotV21110ContinuationError(
                    "cannot establish independent release roots"
                ) from exc
            if aliased:
                raise PilotV21110ContinuationError(
                    f"{left_name} and {right_name} roots must be distinct"
                )


def _verify_v21110_bound_working_directory(repo_root: str | Path) -> dict[str, Any]:
    """Bind the legacy cwd-relative profile input before provider construction."""

    repository = _real_root(repo_root, name="V2.11.10 repository")
    try:
        working_directory = Path.cwd().resolve(strict=True)
    except OSError as exc:
        raise PilotV21110ContinuationError(
            "V2.11.10 working directory is unavailable"
        ) from exc
    if working_directory != repository:
        raise PilotV21110ContinuationError(
            "V2.11.10 scientific dispatch must run from the release repository root"
        )
    profile = repository.joinpath(*V21110_PROFILE_PATH.parts)
    try:
        mode = profile.lstat().st_mode
    except OSError as exc:
        raise PilotV21110ContinuationError(
            "V2.11.10 bound profile input is unavailable"
        ) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise PilotV21110ContinuationError(
            "V2.11.10 bound profile input must be a regular non-symlink file"
        )
    observed = _file_sha256(profile)
    if observed != V21110_PROFILE_FILE_SHA256:
        raise PilotV21110ContinuationError("V2.11.10 bound profile input hash drifted")
    return {
        "path": V21110_PROFILE_PATH.as_posix(),
        "byte_size": profile.stat().st_size,
        "file_sha256": observed,
        "cwd_bound_to_release_root": True,
    }


def _verify_seal(value: Mapping[str, Any], *, name: str) -> None:
    try:
        v2119._verify_seal(value, name=name)
    except v2119.PilotV2119ContinuationError as exc:
        raise PilotV21110ContinuationError(str(exc)) from exc


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return v2119._seal(value)
    except v2119.PilotV2119ContinuationError as exc:
        raise PilotV21110ContinuationError(str(exc)) from exc


def _boundary(contract: Any) -> Mapping[str, Any]:
    value = getattr(contract, "v21110_recovery_boundary", None)
    if not isinstance(value, Mapping):
        raise PilotV21110ContinuationError("V2.11.10 recovery boundary is absent")
    return value


def _expected_v2115_authority_release() -> dict[str, Any]:
    return {
        "contract_id": v2117.V2115_CONTRACT_ID,
        "contract_path": V2115_CONTRACT_PATH.as_posix(),
        "contract_file_sha256": v2117.V2115_CONTRACT_FILE_SHA256,
        "contract_sha256": v2117.V2115_CONTRACT_SHA256,
        "science_tag": v2117.V2115_SCIENCE_TAG,
        "science_tag_object": v2117.V2115_SCIENCE_TAG_OBJECT,
        "science_commit": v2117.V2115_SCIENCE_COMMIT,
        "source_manifest_path": V2115_SOURCE_MANIFEST_PATH.as_posix(),
        "source_manifest_file_sha256": v2117.V2115_SOURCE_MANIFEST_FILE_SHA256,
        "source_manifest_content_sha256": (v2117.V2115_SOURCE_MANIFEST_CONTENT_SHA256),
        "raw_inventory": {
            "root": V2115_RAW_ROOT.as_posix(),
            "canonicalization": "json-sort-keys-compact-utf8-v1",
            "excluded_operational_paths": [".real-stage-execution.lock"],
            "file_count": v2117.V2115_RAW_INVENTORY_FILE_COUNT,
            "storage_bytes": v2117.V2115_RAW_INVENTORY_STORAGE_BYTES,
            "inventory_sha256": v2117.V2115_RAW_INVENTORY_SHA256,
        },
        "scientific_dispatch_acceptance": {
            "path": (V2115_RAW_ROOT / "scientific_dispatch_acceptance.json").as_posix(),
            "file_sha256": V2115_ACCEPTANCE_FILE_SHA256,
            "content_sha256": V2115_ACCEPTANCE_CONTENT_SHA256,
        },
        "preflight_authority": {
            "path": (
                V2115_RAW_ROOT / "long-context-preflight/post_gate_authority.json"
            ).as_posix(),
            "file_sha256": V2115_POST_GATE_FILE_SHA256,
            "content_sha256": V2115_POST_GATE_CONTENT_SHA256,
        },
    }


def _expected_v2115_parent_release() -> dict[str, Any]:
    """Exact V2.11.5 release shape used by the V2.11.10 contract."""

    return {
        key: value
        for key, value in _expected_v2115_authority_release().items()
        if key
        in {
            "contract_id",
            "contract_path",
            "contract_file_sha256",
            "contract_sha256",
            "science_tag",
            "science_tag_object",
            "science_commit",
            "source_manifest_path",
            "source_manifest_file_sha256",
            "source_manifest_content_sha256",
        }
    }


def _expected_v2119_failed_release_no_go() -> dict[str, Any]:
    return {
        "contract_id": V2119_CONTRACT_ID,
        "contract_path": V2119_CONTRACT_PATH.as_posix(),
        "contract_file_sha256": V2119_CONTRACT_FILE_SHA256,
        "contract_sha256": V2119_CONTRACT_SHA256,
        "science_tag": V2119_SCIENCE_TAG,
        "science_tag_object": V2119_SCIENCE_TAG_OBJECT,
        "science_commit": V2119_SCIENCE_COMMIT,
        "source_manifest_path": V2119_SOURCE_MANIFEST_PATH.as_posix(),
        "source_manifest_file_sha256": V2119_SOURCE_MANIFEST_FILE_SHA256,
        "source_manifest_content_sha256": V2119_SOURCE_MANIFEST_CONTENT_SHA256,
        "raw_inventory": {
            "root": V2119_FAILED_RAW_ROOT.as_posix(),
            "canonicalization": "json-sort-keys-compact-utf8-v1",
            "excluded_operational_paths": [".real-stage-execution.lock"],
            "file_count": V2119_EVIDENCE_RAW_FILE_COUNT,
            "storage_bytes": V2119_EVIDENCE_RAW_STORAGE_BYTES,
            "inventory_sha256": V2119_EVIDENCE_RAW_INVENTORY_SHA256,
        },
        "complete_raw_inventory": {
            "root": V2119_FAILED_RAW_ROOT.as_posix(),
            "canonicalization": "json-sort-keys-compact-utf8-v1",
            "excluded_operational_paths": [],
            "file_count": V2119_COMPLETE_RAW_FILE_COUNT,
            "storage_bytes": V2119_COMPLETE_RAW_STORAGE_BYTES,
            "inventory_sha256": V2119_COMPLETE_RAW_INVENTORY_SHA256,
        },
        "run_ledger": {
            "path": (V2119_FAILED_RAW_ROOT / "run_ledger.json").as_posix(),
            "file_sha256": V2119_RUN_LEDGER_FILE_SHA256,
            "ledger_sha256": V2119_RUN_LEDGER_SHA256,
            "event_count": V2119_RUN_EVENT_COUNT,
            "event_head_sha256": V2119_RUN_EVENT_HEAD,
            "registered_rows": V2119_LEDGER_CELL_COUNT,
            "status_counts": {"complete": 1, "failed": 86},
        },
        "budget_ledger": {
            "path": (V2119_FAILED_RAW_ROOT / "budget_ledger.json").as_posix(),
            "file_sha256": V2119_BUDGET_LEDGER_FILE_SHA256,
            "ledger_sha256": V2119_BUDGET_LEDGER_SHA256,
            "event_count": V2119_BUDGET_EVENT_COUNT,
            "event_head_sha256": V2119_BUDGET_EVENT_HEAD,
            "owner_rows": 37,
            "status_counts": {"complete": 1, "failed": 36},
            "current_actual": {
                "cost_usd": 0.0,
                "hosted_completions": 0,
                "storage_bytes": V2119_CURRENT_ACTUAL_STORAGE_BYTES,
            },
        },
        "stage_receipts": {
            "parent-import": {
                "path": (
                    V2119_FAILED_RAW_ROOT / "parent-import/stage_receipt.json"
                ).as_posix(),
                "file_sha256": V2119_CRITICAL_RAW_FILE_SHA256[
                    "parent-import/stage_receipt.json"
                ],
                "content_sha256": V2119_PARENT_STAGE_RECEIPT_CONTENT_SHA256,
                "status": "complete",
                "go": True,
                "execution_progression_go": True,
                "registered_run_count": 1,
                "status_counts": {"complete": 1},
                "scientific_matrix_complete": True,
            },
            "experiment-d": {
                "path": (
                    V2119_FAILED_RAW_ROOT / "experiment-d/stage_receipt.json"
                ).as_posix(),
                "file_sha256": V2119_CRITICAL_RAW_FILE_SHA256[
                    "experiment-d/stage_receipt.json"
                ],
                "content_sha256": V2119_EXPERIMENT_D_STAGE_RECEIPT_CONTENT_SHA256,
                "status": "complete-with-no-go",
                "go": False,
                "execution_progression_go": True,
                "registered_run_count": 55,
                "status_counts": {"failed": 55},
                "scientific_matrix_complete": False,
            },
            "experiment-b": {
                "path": (
                    V2119_FAILED_RAW_ROOT / "experiment-b/stage_receipt.json"
                ).as_posix(),
                "file_sha256": V2119_CRITICAL_RAW_FILE_SHA256[
                    "experiment-b/stage_receipt.json"
                ],
                "content_sha256": V2119_EXPERIMENT_B_STAGE_RECEIPT_CONTENT_SHA256,
                "status": "complete-with-no-go",
                "go": False,
                "execution_progression_go": True,
                "registered_run_count": 25,
                "status_counts": {"failed": 25},
                "scientific_matrix_complete": False,
            },
            "cross-model": {
                "path": (
                    V2119_FAILED_RAW_ROOT / "cross-model/stage_receipt.json"
                ).as_posix(),
                "file_sha256": V2119_CRITICAL_RAW_FILE_SHA256[
                    "cross-model/stage_receipt.json"
                ],
                "content_sha256": V2119_CROSS_MODEL_STAGE_RECEIPT_CONTENT_SHA256,
                "status": "complete-with-no-go",
                "go": False,
                "execution_progression_go": False,
                "registered_run_count": 6,
                "status_counts": {"failed": 6},
                "scientific_matrix_complete": False,
            },
        },
        "release_attestation": {
            "path": (V2119_FAILED_RAW_ROOT / "release_attestation.json").as_posix(),
            "file_sha256": V2119_RELEASE_ATTESTATION_FILE_SHA256,
            "attestation_sha256": V2119_RELEASE_ATTESTATION_SHA256,
            "status": "pass",
        },
        "scientific_launch_input": {
            "path": (V2119_FAILED_RAW_ROOT / "scientific_launch_input.json").as_posix(),
            "file_sha256": V2119_LAUNCH_INPUT_FILE_SHA256,
            "launch_input_sha256": V2119_LAUNCH_INPUT_SHA256,
        },
        "scientific_dispatch_acceptance": {
            "path": (
                V2119_FAILED_RAW_ROOT / "scientific_dispatch_acceptance.json"
            ).as_posix(),
            "file_sha256": V2119_ACCEPTANCE_FILE_SHA256,
            "content_sha256": V2119_ACCEPTANCE_CONTENT_SHA256,
            "status": "go",
            "go": True,
            "scientific_evidence": False,
        },
        "acceptance_receipt_present": True,
        "science_reservations": 36,
        "provider_construction": False,
        "provider_calls": 0,
        "hosted_completions": 0,
        "scientific_evidence": False,
        "resume_forbidden": True,
        "failure_reclassification_forbidden": True,
        "failure_profile": {
            "failure_manifest_count": 36,
            "failed_scientific_cells": 86,
            "gpt52_failure_count": 80,
            "gpt56_failure_count": 6,
            "gpt52_message": V2119_FAILURE_MESSAGE_BY_MODEL["gpt52_main"],
            "gpt56_message": V2119_FAILURE_MESSAGE_BY_MODEL["gpt56_diagnostic"],
        },
    }


def _verify_v2119_release_git(root: Path) -> dict[str, str]:
    try:
        return v2117._verify_release_git(
            root,
            name="V2.11.9 failed release",
            science_tag=V2119_SCIENCE_TAG,
            science_commit=V2119_SCIENCE_COMMIT,
            science_tag_object=V2119_SCIENCE_TAG_OBJECT,
        )
    except v2117.PilotV2117ContinuationError as exc:
        raise PilotV21110ContinuationError(str(exc)) from exc


def _verify_v2115_authority_git(root: Path) -> dict[str, str]:
    try:
        return v2117._verify_authority_git(root)
    except v2117.PilotV2117ContinuationError as exc:
        raise PilotV21110ContinuationError(str(exc)) from exc


def _v2119_raw_inventory(root: Path) -> dict[str, Any]:
    raw = root.joinpath(*V2119_FAILED_RAW_ROOT.parts)
    if raw.is_symlink() or not raw.is_dir():
        raise PilotV21110ContinuationError("V2.11.9 failed raw root is unavailable")
    rows: list[dict[str, Any]] = []
    for path in sorted(raw.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise PilotV21110ContinuationError("V2.11.9 raw tree contains a symlink")
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(raw).as_posix(),
                    "byte_size": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
        elif not path.is_dir():
            raise PilotV21110ContinuationError(
                "V2.11.9 raw tree contains a non-regular entry"
            )
    observed_hashes = {str(row["path"]): str(row["sha256"]) for row in rows}
    if any(
        observed_hashes.get(relative) != expected
        for relative, expected in V2119_CRITICAL_RAW_FILE_SHA256.items()
    ):
        raise PilotV21110ContinuationError(
            "V2.11.9 critical terminal raw binding drifted"
        )
    evidence_rows = [row for row in rows if row["path"] != ".real-stage-execution.lock"]
    complete = {
        "root": V2119_FAILED_RAW_ROOT.as_posix(),
        "canonicalization": "json-sort-keys-compact-utf8-v1",
        "excluded_operational_paths": [],
        "file_count": len(rows),
        "storage_bytes": sum(int(row["byte_size"]) for row in rows),
        "inventory_sha256": canonical_sha256(rows),
    }
    evidence = {
        "root": V2119_FAILED_RAW_ROOT.as_posix(),
        "canonicalization": "json-sort-keys-compact-utf8-v1",
        "excluded_operational_paths": [".real-stage-execution.lock"],
        "file_count": len(evidence_rows),
        "storage_bytes": sum(int(row["byte_size"]) for row in evidence_rows),
        "inventory_sha256": canonical_sha256(evidence_rows),
    }
    expected_evidence = _expected_v2119_failed_release_no_go()["raw_inventory"]
    expected_complete = {
        "root": V2119_FAILED_RAW_ROOT.as_posix(),
        "canonicalization": "json-sort-keys-compact-utf8-v1",
        "excluded_operational_paths": [],
        "file_count": V2119_COMPLETE_RAW_FILE_COUNT,
        "storage_bytes": V2119_COMPLETE_RAW_STORAGE_BYTES,
        "inventory_sha256": V2119_COMPLETE_RAW_INVENTORY_SHA256,
    }
    if complete != expected_complete or evidence != expected_evidence:
        raise PilotV21110ContinuationError("V2.11.9 raw inventory digest drifted")
    return {"complete": complete, "evidence": evidence, "rows": rows}


def _v2115_authority_state(root: Path) -> dict[str, Any]:
    release = _verify_v2115_authority_git(root)
    expected = _expected_v2115_authority_release()
    contract_path = root.joinpath(*V2115_CONTRACT_PATH.parts)
    contract = load_pilot_contract(contract_path)
    manifest_path = root.joinpath(*V2115_SOURCE_MANIFEST_PATH.parts)
    manifest = _strict_json(manifest_path, name="V2.11.5 source manifest")
    _verify_seal(manifest, name="V2.11.5 source manifest")
    if (
        contract.contract_id != v2117.V2115_CONTRACT_ID
        or contract.canonical_hash != v2117.V2115_CONTRACT_SHA256
        or _file_sha256(contract_path) != v2117.V2115_CONTRACT_FILE_SHA256
        or _file_sha256(manifest_path) != v2117.V2115_SOURCE_MANIFEST_FILE_SHA256
        or manifest.get("integrity", {}).get("content_sha256")
        != v2117.V2115_SOURCE_MANIFEST_CONTENT_SHA256
        or manifest.get("schema_version") != V2115_SOURCE_MANIFEST_SCHEMA_VERSION
    ):
        raise PilotV21110ContinuationError("V2.11.5 authority source drifted")
    try:
        raw_inventory = v2117._authority_raw_inventory(root)
    except v2117.PilotV2117ContinuationError as exc:
        raise PilotV21110ContinuationError(str(exc)) from exc
    raw = root.joinpath(*V2115_RAW_ROOT.parts)
    acceptance_path = raw / "scientific_dispatch_acceptance.json"
    acceptance = _strict_json(acceptance_path, name="V2.11.5 acceptance")
    _verify_seal(acceptance, name="V2.11.5 acceptance")
    gate_path = raw / "long-context-preflight/post_gate_authority.json"
    gate = _strict_json(gate_path, name="V2.11.5 post-gate authority")
    if (
        _file_sha256(acceptance_path) != V2115_ACCEPTANCE_FILE_SHA256
        or acceptance.get("integrity", {}).get("content_sha256")
        != V2115_ACCEPTANCE_CONTENT_SHA256
        or _file_sha256(gate_path) != V2115_POST_GATE_FILE_SHA256
        or gate.get("receipt_sha256") != V2115_POST_GATE_CONTENT_SHA256
        or gate.get("contract_id") != v2117.V2115_CONTRACT_ID
        or gate.get("contract_sha256") != v2117.V2115_CONTRACT_SHA256
        or gate.get("go") is not True
        or gate.get("scientific_evidence") is not False
    ):
        raise PilotV21110ContinuationError("V2.11.5 authority receipt drifted")
    from .pilot_v2115_gate import verified_v2115_gate_authority_binding

    try:
        gate_binding = verified_v2115_gate_authority_binding(
            gate_path.relative_to(root).as_posix(),
            repo_root=root,
            expected_git_commit=v2117.V2115_SCIENCE_COMMIT,
            expected_contract_sha256=v2117.V2115_CONTRACT_SHA256,
        )
    except Exception as exc:
        raise PilotV21110ContinuationError(
            f"V2.11.5 gate authority binding failed: {exc}"
        ) from exc
    parent_import_path = raw / "parent-import/parent_import_receipt.json"
    parent_import = _strict_json(
        parent_import_path, name="V2.11.5 parent-import receipt"
    )
    _verify_seal(parent_import, name="V2.11.5 parent-import receipt")
    if (
        _file_sha256(parent_import_path) != v2117.V2115_PARENT_IMPORT_FILE_SHA256
        or parent_import.get("integrity", {}).get("content_sha256")
        != v2117.V2115_PARENT_IMPORT_CONTENT_SHA256
    ):
        raise PilotV21110ContinuationError("V2.11.5 parent-import receipt drifted")
    return {
        "release": release,
        "contract": contract,
        "raw_root": raw,
        "raw_inventory": raw_inventory,
        "acceptance": acceptance,
        "post_gate_authority": gate,
        "gate_binding": gate_binding,
        "parent_import_receipt": parent_import,
        "binding": expected,
    }


def verify_v2119_terminal_no_go(
    *,
    failed_repo_root: str | Path,
    authority_repo_root: str | Path,
) -> dict[str, Any]:
    """Deep-verify V2.11.9 as an immutable zero-completion no-go.

    V2.11.9 accepted its 86-cell dispatch matrix, then every scientific cell
    failed before the first provider completion because the observed-P95
    authority adapter compared a 17-field envelope with a 13-field core.  The
    failures remain in the denominator and are never resumed by V2.11.10.
    """

    from .pilot_orchestrator import PilotRunLedger, _budget_caps

    failed_root = _real_root(failed_repo_root, name="V2.11.9 failed repository")
    authority_root = _real_root(
        authority_repo_root, name="V2.11.5 authority repository"
    )
    if failed_root == authority_root:
        raise PilotV21110ContinuationError(
            "V2.11.9 failed and V2.11.5 authority roots must be distinct"
        )
    failed_release = _verify_v2119_release_git(failed_root)
    authority = _v2115_authority_state(authority_root)
    raw_inventory = _v2119_raw_inventory(failed_root)

    contract_path = failed_root.joinpath(*V2119_CONTRACT_PATH.parts)
    failed_contract = load_pilot_contract(contract_path)
    manifest_path = failed_root.joinpath(*V2119_SOURCE_MANIFEST_PATH.parts)
    manifest = _strict_json(manifest_path, name="V2.11.9 source manifest")
    _verify_seal(manifest, name="V2.11.9 source manifest")
    if (
        failed_contract.contract_id != V2119_CONTRACT_ID
        or failed_contract.canonical_hash != V2119_CONTRACT_SHA256
        or _file_sha256(contract_path) != V2119_CONTRACT_FILE_SHA256
        or _file_sha256(manifest_path) != V2119_SOURCE_MANIFEST_FILE_SHA256
        or manifest.get("integrity", {}).get("content_sha256")
        != V2119_SOURCE_MANIFEST_CONTENT_SHA256
        or manifest.get("schema_version") != V2119_SOURCE_MANIFEST_SCHEMA_VERSION
        or manifest.get("contract_id") != V2119_CONTRACT_ID
        or manifest.get("release_tag") != V2119_SCIENCE_TAG
    ):
        raise PilotV21110ContinuationError("V2.11.9 release source drifted")

    raw = failed_root.joinpath(*V2119_FAILED_RAW_ROOT.parts)
    run_path = raw / "run_ledger.json"
    run_ledger = PilotRunLedger(
        run_path,
        contract_hash=failed_contract.canonical_hash,
        tamper_evident=True,
        bind_terminal_artifacts=True,
    )
    run_snapshot = run_ledger.snapshot()
    events = run_snapshot.get("events")
    runs = run_snapshot.get("runs")
    expected_specs = {spec.run_id: spec.to_dict() for spec in failed_contract.expand()}
    parent_specs = tuple(failed_contract.expand(stage="parent-import"))
    science_specs = tuple(
        spec
        for stage_id in ("experiment-d", "experiment-b", "cross-model")
        for spec in failed_contract.expand(stage=stage_id)
    )
    if len(parent_specs) != 1 or len(science_specs) != 86:
        raise PilotV21110ContinuationError("V2.11.9 failed denominator drifted")
    parent_run_id = parent_specs[0].run_id
    science_ids = {spec.run_id for spec in science_specs}
    if (
        _file_sha256(run_path) != V2119_RUN_LEDGER_FILE_SHA256
        or run_snapshot.get("ledger_sha256") != V2119_RUN_LEDGER_SHA256
        or not isinstance(events, list)
        or len(events) != V2119_RUN_EVENT_COUNT
        or events[-1].get("event_sha256") != V2119_RUN_EVENT_HEAD
        or Counter(event.get("event_type") for event in events)
        != Counter(
            {
                "genesis": 1,
                "runs_registered": 1,
                "run_finalized": 87,
                "acceptance_receipt_bound": 1,
            }
        )
        or not isinstance(runs, Mapping)
        or set(runs) != set(expected_specs)
        or runs[parent_run_id].get("status") != "complete"
        or runs[parent_run_id].get("failure") is not None
        or any(
            not isinstance(runs[run_id], Mapping)
            or runs[run_id].get("spec") != expected_specs[run_id]
            or runs[run_id].get("status") != "failed"
            or not isinstance(runs[run_id].get("failure"), Mapping)
            or runs[run_id]["failure"].get("error_type")
            != V2119_FAILURE_ERROR_TYPE
            or runs[run_id]["failure"].get("message")
            != V2119_FAILURE_MESSAGE_BY_MODEL[
                str(expected_specs[run_id]["model_id"])
            ]
            for run_id in science_ids
        )
    ):
        raise PilotV21110ContinuationError("V2.11.9 terminal run-ledger no-go drifted")
    for run_id in expected_specs:
        run_ledger.verify_terminal_artifact_binding(run_id)
    paid = type(
        "V2119Paid",
        (),
        {
            "git_tag": V2119_SCIENCE_TAG,
            "head_commit": V2119_SCIENCE_COMMIT,
            "tag_commit": V2119_SCIENCE_COMMIT,
            "tag_object_type": "tag",
            "worktree_clean": True,
        },
    )()
    terminal_artifacts = v2119.verify_v2119_terminal_scientific_artifacts(
        failed_contract,
        repo_root=failed_root,
        raw_root=raw,
        run_ledger=run_ledger,
        paid=paid,
    )
    if terminal_artifacts.get("terminal_artifact_bindings") != 86:
        raise PilotV21110ContinuationError(
            "V2.11.9 scientific failure-artifact denominator drifted"
        )

    budget_path = raw / "budget_ledger.json"
    stored_budget = _strict_json(budget_path, name="V2.11.9 budget ledger")
    imported_parent = v2119.parent_budget_debit_for_v2119(failed_contract)
    if stored_budget.get("parent_debit") != imported_parent.to_dict():
        raise PilotV21110ContinuationError("V2.11.9 imported parent debit drifted")
    budget_ledger = PilotBudgetLedger(
        budget_path,
        contract_hash=failed_contract.canonical_hash,
        caps=_budget_caps(failed_contract),
        tamper_evident=True,
        parent_debit=imported_parent,
    )
    budget = budget_ledger.snapshot()
    budget_events = budget.get("events")
    budget_runs = budget.get("runs")
    committed = budget.get("committed")
    committed_plus_reserved = budget.get("committed_plus_reserved")
    if (
        _file_sha256(budget_path) != V2119_BUDGET_LEDGER_FILE_SHA256
        or budget.get("ledger_sha256") != V2119_BUDGET_LEDGER_SHA256
        or not isinstance(budget_events, list)
        or len(budget_events) != V2119_BUDGET_EVENT_COUNT
        or budget_events[-1].get("event_sha256") != V2119_BUDGET_EVENT_HEAD
        or Counter(event.get("event_type") for event in budget_events)
        != Counter(
            {
                "genesis": 1,
                "parent_debit_imported": 1,
                "run_reserved": 37,
                "run_finalized": 37,
                "acceptance_receipt_bound": 1,
            }
        )
        or not isinstance(budget_runs, Mapping)
        or len(budget_runs) != 37
        or Counter(row.get("status") for row in budget_runs.values())
        != Counter({"complete": 1, "failed": 36})
        or not isinstance(committed, Mapping)
        or committed.get("cost_usd") != V2119_CUMULATIVE_COST_USD
        or committed.get("completions") != V2119_CUMULATIVE_COMPLETIONS
        or committed.get("storage_bytes") != V2119_CUMULATIVE_STORAGE_BYTES
        or committed_plus_reserved != committed
    ):
        raise PilotV21110ContinuationError(
            "V2.11.9 terminal budget-ledger no-go drifted"
        )
    current_actuals = [
        row.get("actual") for row in budget_runs.values() if isinstance(row, Mapping)
    ]
    if (
        any(not isinstance(actual, Mapping) for actual in current_actuals)
        or sum(float(actual.get("cost_usd", math.nan)) for actual in current_actuals)
        != 0.0
        or sum(int(actual.get("completions", -1)) for actual in current_actuals) != 0
        or sum(int(actual.get("storage_bytes", -1)) for actual in current_actuals)
        != V2119_CURRENT_ACTUAL_STORAGE_BYTES
    ):
        raise PilotV21110ContinuationError("V2.11.9 current actual debit drifted")

    parent_import_path = raw / "parent-import/parent_import_receipt.json"
    if (
        _file_sha256(parent_import_path) != V2119_PARENT_IMPORT_RECEIPT_FILE_SHA256
        or _strict_json(
            parent_import_path, name="V2.11.9 parent-import authority receipt"
        ).get("integrity", {}).get("content_sha256")
        != V2119_PARENT_IMPORT_RECEIPT_CONTENT_SHA256
    ):
        raise PilotV21110ContinuationError("V2.11.9 parent-import authority drifted")
    v2119.verify_v2119_parent_import_receipt(
        parent_import_path,
        contract=failed_contract,
        repo_root=failed_root,
        raw_root=raw,
        paid=paid,
    )

    stages: dict[str, Any] = {}
    expected_stages = _expected_v2119_failed_release_no_go()["stage_receipts"]
    for stage_id, expected in expected_stages.items():
        stage_path = raw / stage_id / "stage_receipt.json"
        stage = _strict_json(stage_path, name=f"V2.11.9 {stage_id} receipt")
        if (
            _file_sha256(stage_path) != expected["file_sha256"]
            or stage.get("integrity", {}).get("content_sha256")
            != expected["content_sha256"]
            or stage.get("contract_id") != V2119_CONTRACT_ID
            or stage.get("contract_sha256") != V2119_CONTRACT_SHA256
            or stage.get("stage_id") != stage_id
            or stage.get("status") != expected["status"]
            or stage.get("go") is not expected["go"]
            or stage.get("execution_progression_go")
            is not expected["execution_progression_go"]
            or stage.get("status_counts") != expected["status_counts"]
            or stage.get("terminal") is not True
            or stage.get("denominator_terminal") is not True
        ):
            raise PilotV21110ContinuationError(
                f"V2.11.9 {stage_id} terminal receipt drifted"
            )
        stages[stage_id] = stage

    acceptance_path = raw / "scientific_dispatch_acceptance.json"
    acceptance = _strict_json(acceptance_path, name="V2.11.9 acceptance")
    _verify_seal(acceptance, name="V2.11.9 acceptance")
    if (
        _file_sha256(acceptance_path) != V2119_ACCEPTANCE_FILE_SHA256
        or acceptance.get("integrity", {}).get("content_sha256")
        != V2119_ACCEPTANCE_CONTENT_SHA256
        or acceptance.get("status") != "go"
        or acceptance.get("go") is not True
        or acceptance.get("provider_boundary", {}).get("provider_calls") != 0
        or acceptance.get("provider_boundary", {}).get("provider_construction")
        is not False
    ):
        raise PilotV21110ContinuationError("V2.11.9 acceptance drifted")

    attestation_path = raw / "release_attestation.json"
    launch_path = raw / "scientific_launch_input.json"
    attestation = _strict_json(attestation_path, name="V2.11.9 attestation")
    launch = _strict_json(launch_path, name="V2.11.9 launch input")
    if (
        _file_sha256(attestation_path) != V2119_RELEASE_ATTESTATION_FILE_SHA256
        or _file_sha256(launch_path) != V2119_LAUNCH_INPUT_FILE_SHA256
        or attestation.get("status") != "pass"
        or attestation.get("attestation_sha256") != V2119_RELEASE_ATTESTATION_SHA256
        or attestation.get("head_commit") != V2119_SCIENCE_COMMIT
        or attestation.get("local_tag", {}).get("object_id") != V2119_SCIENCE_TAG_OBJECT
        or attestation.get("contract", {}).get("canonical_sha256")
        != V2119_CONTRACT_SHA256
        or launch.get("contract_sha256") != V2119_CONTRACT_SHA256
        or launch.get("launch_input_sha256") != V2119_LAUNCH_INPUT_SHA256
    ):
        raise PilotV21110ContinuationError("V2.11.9 launch provenance drifted")
    return {
        "failed_release": failed_release,
        "failed_contract": failed_contract,
        "failed_raw": raw,
        "source_manifest": manifest,
        "raw_inventory": raw_inventory,
        "run_snapshot": run_snapshot,
        "budget_snapshot": budget,
        "stage_receipts": stages,
        "parent_import_receipt": _strict_json(
            parent_import_path, name="V2.11.9 parent-import authority receipt"
        ),
        "acceptance": acceptance,
        "parent_run_id": parent_run_id,
        "science_run_ids": sorted(science_ids),
        "authority": authority,
        "terminal_artifacts": terminal_artifacts,
    }


def _historical_v2115_contract_binding(contract: PilotContract) -> dict[str, Any]:
    """Reconstruct the exact provenance binding used by V2.11.5 acceptance."""

    try:
        binding = contract.validate_provenance(
            v2117.V2115_SCIENCE_COMMIT,
            v2117.V2115_SCIENCE_TAG,
        )
    except Exception as exc:
        raise PilotV21110ContinuationError(
            f"V2.11.5 contract provenance binding failed: {exc}"
        ) from exc
    if (
        not isinstance(binding, Mapping)
        or not binding
        or canonical_sha256(binding) != V2115_CONTRACT_BINDING_SHA256
    ):
        raise PilotV21110ContinuationError(
            "V2.11.5 contract provenance binding differs from the immutable "
            "scientific-dispatch acceptance"
        )
    return dict(binding)


def _verify_v2115_acceptance_with_authority_context(
    authority_repo_root: str | Path,
) -> dict[str, Any]:
    """Rebuild V2.11.5 acceptance under its own source-authority context."""

    from .pilot_orchestrator import GitProvenance, PilotRunLedger, _budget_caps
    from .pilot_v2115_acceptance import verify_v2115_scientific_dispatch_acceptance
    from .runner import observed_p95_authority_repo_context

    root = _real_root(authority_repo_root, name="V2.11.5 authority repository")
    state = _v2115_authority_state(root)
    contract = state["contract"]
    raw = state["raw_root"]
    run_ledger = PilotRunLedger(
        raw / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    stored = _strict_json(raw / "budget_ledger.json", name="V2.11.5 budget ledger")
    budget_ledger = PilotBudgetLedger(
        raw / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=_budget_caps(contract),
        tamper_evident=True,
        parent_debit=stored.get("parent_debit"),
    )
    try:
        paid = GitProvenance(
            git_tag=v2117.V2115_SCIENCE_TAG,
            head_commit=v2117.V2115_SCIENCE_COMMIT,
            tag_commit=v2117.V2115_SCIENCE_COMMIT,
            tag_object_type="tag",
            worktree_clean=True,
            contract_binding=_historical_v2115_contract_binding(contract),
            release_attestation=_strict_json(
                raw / "release_attestation.json", name="V2.11.5 attestation"
            ),
        )
        with observed_p95_authority_repo_context(root):
            return verify_v2115_scientific_dispatch_acceptance(
                raw / "scientific_dispatch_acceptance.json",
                contract=contract,
                repo_root=root,
                raw_root=raw,
                paid=paid,
                run_ledger=run_ledger,
                budget_ledger=budget_ledger,
            )
    except Exception as exc:
        raise PilotV21110ContinuationError(
            f"V2.11.5 acceptance revalidation failed: {exc}"
        ) from exc


def require_v21110_provider_keys_absent() -> None:
    present = [name for name in _PROVIDER_KEY_ENV_NAMES if os.environ.get(name)]
    if present:
        raise PilotV21110ContinuationError(
            "V2.11.10 parent import must run before provider credentials are "
            f"loaded; present={sorted(present)}"
        )


def _expected_pre_science_paths(contract: PilotContract) -> tuple[set[str], set[str]]:
    parent_specs = tuple(contract.expand(stage="parent-import"))
    if len(parent_specs) != 1:
        raise PilotV21110ContinuationError(
            "pre-science namespace requires one parent-import cell"
        )
    parent_run_id = parent_specs[0].run_id
    files = {
        ".real-stage-execution.lock",
        "budget_ledger.json",
        "release_attestation.json",
        "run_ledger.json",
        V21110_ACCEPTANCE_FILENAME,
        f".{V21110_ACCEPTANCE_FILENAME}.pending",
        "scientific_launch_input.json",
        "parent-import/parent_import_receipt.json",
        "parent-import/stage_receipt.json",
        f"parent-import/summaries/{parent_run_id}.json",
        "parent-import/current_authority/post_gate_authority.json",
        "parent-import/current_authority/gpt52_main/projection_p95.json",
        "parent-import/current_authority/gpt56_diagnostic/projection_p95.json",
    }
    directories = {
        PurePosixPath(relative).parent.as_posix()
        for relative in files
        if PurePosixPath(relative).parent.as_posix() != "."
    }
    for relative in tuple(directories):
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            directories.add(parent.as_posix())
            parent = parent.parent
    return files, directories


def _audit_pre_science_namespace(raw_root: Path, contract: PilotContract) -> None:
    allowed_files, allowed_directories = _expected_pre_science_paths(contract)
    unexpected: list[str] = []
    for path in sorted(raw_root.rglob("*")):
        relative = path.relative_to(raw_root).as_posix()
        if path.is_symlink():
            raise PilotV21110ContinuationError(
                f"pre-science raw namespace contains a symlink: {relative}"
            )
        if path.is_file():
            if relative not in allowed_files:
                unexpected.append(relative)
        elif path.is_dir():
            if relative not in allowed_directories:
                unexpected.append(relative + "/")
        else:
            unexpected.append(relative)
    if unexpected:
        raise PilotV21110ContinuationError(
            "pre-science raw namespace contains unexpected paths: "
            + ", ".join(unexpected[:10])
        )


def _v21110_pending_output_paths(
    contract: PilotContract,
    raw_root: Path,
    spec: Any,
) -> tuple[Path, ...]:
    stage_root = raw_root / spec.stage_id
    paths = [
        stage_root / "runs" / spec.run_id,
        stage_root / "summaries" / f"{spec.run_id}.json",
        stage_root / "diagnostic_summaries" / f"{spec.run_id}.json",
        stage_root / "provider_call_journals" / f"{spec.run_id}--actor.json",
        stage_root / "provider_call_journals" / f"{spec.run_id}--preflight.json",
    ]
    if spec.stage_id == "experiment-d":
        checkpoint_root = stage_root / "checkpoints"
        if len(contract.models_for_stage(spec.stage_id)) > 1:
            checkpoint_root = checkpoint_root / spec.model_id
        checkpoint_group = checkpoint_root / f"s{spec.environment_seed}"
        prefix_group_id = (
            f"{contract.contract_id}--experiment-d--{spec.model_id}--"
            f"checkpoint-group--s{spec.environment_seed}"
        )
        paths.extend(
            (
                checkpoint_group,
                checkpoint_group / "checkpoint.json.run-intent.json",
                stage_root
                / "provider_call_journals"
                / f"{prefix_group_id}--prefix.json",
            )
        )
    return tuple(paths)


def assert_v21110_dispatch_target_fresh(
    contract: PilotContract,
    *,
    raw_root: str | Path,
    spec: Any,
) -> None:
    raw = _real_root(raw_root, name="V2.11.10 raw root")
    stale = [
        path.relative_to(raw).as_posix()
        for path in _v21110_pending_output_paths(contract, raw, spec)
        if path.exists() or path.is_symlink()
    ]
    if stale:
        raise PilotV21110ContinuationError(
            "V2.11.10 nonterminal dispatch target is not fresh: " + ", ".join(stale[:10])
        )


def audit_v21110_scientific_stage_namespace(
    contract: PilotContract,
    *,
    raw_root: str | Path,
    stage_id: str,
    run_ledger: Any,
) -> dict[str, Any]:
    """Reject post-acceptance output collisions before provider construction."""

    if stage_id not in {"experiment-d", "experiment-b", "cross-model"}:
        raise PilotV21110ContinuationError(
            "V2.11.10 namespace audit requires a scientific stage"
        )
    raw = _real_root(raw_root, name="V2.11.10 raw root")
    pre_science_files, pre_science_directories = _expected_pre_science_paths(contract)
    allowed_root_names = {
        PurePosixPath(relative).parts[0]
        for relative in (*pre_science_files, *pre_science_directories)
    } | {"experiment-d", "experiment-b", "cross-model"}
    root_stale: list[str] = []
    for child in raw.iterdir():
        if child.is_symlink() or child.name not in allowed_root_names:
            root_stale.append(child.relative_to(raw).as_posix())
    for scientific_stage in ("experiment-d", "experiment-b", "cross-model"):
        diagnostic_root = raw / scientific_stage / "diagnostic_summaries"
        if diagnostic_root.exists() or diagnostic_root.is_symlink():
            root_stale.append(diagnostic_root.relative_to(raw).as_posix())
    if root_stale:
        raise PilotV21110ContinuationError(
            "V2.11.10 scientific raw namespace contains non-scientific or "
            "unexpected roots: " + ", ".join(sorted(set(root_stale))[:10])
        )

    stage_order = ("experiment-d", "experiment-b", "cross-model")
    current_index = stage_order.index(stage_id)
    specs_by_stage = {
        candidate: tuple(contract.expand(stage=candidate)) for candidate in stage_order
    }
    terminal_ids_by_stage = {
        candidate: {
            spec.run_id
            for spec in candidate_specs
            if run_ledger.is_terminal(spec.run_id)
        }
        for candidate, candidate_specs in specs_by_stage.items()
    }
    specs = specs_by_stage[stage_id]
    terminal_ids = terminal_ids_by_stage[stage_id]
    terminal_binding_verifier = getattr(
        run_ledger, "verify_terminal_artifact_binding", None
    )
    if not callable(terminal_binding_verifier):
        raise PilotV21110ContinuationError(
            "V2.11.10 scientific resume requires immutable terminal artifact " "bindings"
        )
    try:
        for run_id in sorted(terminal_ids):
            terminal_binding_verifier(run_id)
    except Exception as exc:
        raise PilotV21110ContinuationError(
            "V2.11.10 terminal artifact binding failed before provider "
            f"construction: {exc}"
        ) from exc
    pending = tuple(spec for spec in specs if spec.run_id not in terminal_ids)
    stale: list[str] = []
    for candidate, candidate_specs in specs_by_stage.items():
        candidate_terminal = terminal_ids_by_stage[candidate]
        for spec in candidate_specs:
            if spec.run_id in candidate_terminal:
                continue
            stale.extend(
                path.relative_to(raw).as_posix()
                for path in _v21110_pending_output_paths(contract, raw, spec)
                if path.exists() or path.is_symlink()
            )

    def audit_direct_children(
        directory: Path,
        allowed: set[str],
        *,
        expected_kind: str,
    ) -> None:
        if directory.is_symlink():
            stale.append(directory.relative_to(raw).as_posix())
            return
        if not directory.exists():
            return
        if not directory.is_dir():
            stale.append(directory.relative_to(raw).as_posix())
            return
        for child in directory.iterdir():
            wrong_kind = (expected_kind == "directory" and not child.is_dir()) or (
                expected_kind == "regular-file" and not child.is_file()
            )
            if child.is_symlink() or child.name not in allowed or wrong_kind:
                stale.append(child.relative_to(raw).as_posix())

    for index, candidate in enumerate(stage_order):
        candidate_root = raw / candidate
        if candidate_root.is_symlink():
            stale.append(candidate)
            continue
        if not candidate_root.exists():
            continue
        if not candidate_root.is_dir():
            stale.append(candidate)
            continue
        if index > current_index:
            stale.append(candidate + "/")
            continue
        candidate_terminal = terminal_ids_by_stage[candidate]
        allowed_stage_children = {
            "runs",
            "summaries",
            "provider_call_journals",
            "provider_catalog",
        }
        if candidate == "experiment-d":
            allowed_stage_children.add("checkpoints")
        if len(candidate_terminal) == len(specs_by_stage[candidate]):
            allowed_stage_children.add("stage_receipt.json")
        for child in candidate_root.iterdir():
            if (
                child.is_symlink()
                or child.name not in allowed_stage_children
                or (child.name == "stage_receipt.json" and not child.is_file())
                or (child.name != "stage_receipt.json" and not child.is_dir())
            ):
                stale.append(child.relative_to(raw).as_posix())
        audit_direct_children(
            candidate_root / "runs",
            candidate_terminal,
            expected_kind="directory",
        )
        audit_direct_children(
            candidate_root / "summaries",
            {f"{run_id}.json" for run_id in candidate_terminal},
            expected_kind="regular-file",
        )
        audit_direct_children(
            candidate_root / "provider_call_journals",
            {
                f"{run_id}--{kind}.json"
                for run_id in candidate_terminal
                for kind in ("actor", "preflight")
            }
            | (
                {
                    (
                        f"{contract.contract_id}--experiment-d--{model_id}--"
                        f"checkpoint-group--s{seed}--prefix.json"
                    )
                    for model_id, seed in {
                        (spec.model_id, spec.environment_seed)
                        for spec in specs_by_stage[candidate]
                    }
                    if all(
                        spec.run_id in candidate_terminal
                        for spec in specs_by_stage[candidate]
                        if spec.model_id == model_id and spec.environment_seed == seed
                    )
                }
                if candidate == "experiment-d"
                else set()
            ),
            expected_kind="regular-file",
        )
        audit_direct_children(
            candidate_root / "provider_catalog",
            {f"{model_id}.json" for model_id in contract.models_for_stage(candidate)},
            expected_kind="regular-file",
        )
        if candidate == "experiment-d":
            d_models = tuple(contract.models_for_stage(candidate))
            terminal_groups = {
                (model_id, seed)
                for model_id, seed in {
                    (spec.model_id, spec.environment_seed)
                    for spec in specs_by_stage[candidate]
                }
                if all(
                    spec.run_id in candidate_terminal
                    for spec in specs_by_stage[candidate]
                    if spec.model_id == model_id and spec.environment_seed == seed
                )
            }
            checkpoint_root = candidate_root / "checkpoints"
            if len(d_models) > 1:
                audit_direct_children(
                    checkpoint_root,
                    {
                        model_id
                        for model_id in d_models
                        if any(
                            terminal_model == model_id
                            for terminal_model, _ in terminal_groups
                        )
                    },
                    expected_kind="directory",
                )
                for model_id in d_models:
                    audit_direct_children(
                        checkpoint_root / model_id,
                        {
                            f"s{seed}"
                            for terminal_model, seed in terminal_groups
                            if terminal_model == model_id
                        },
                        expected_kind="directory",
                    )
            else:
                audit_direct_children(
                    checkpoint_root,
                    {f"s{seed}" for _, seed in terminal_groups},
                    expected_kind="directory",
                )
    if stale:
        raise PilotV21110ContinuationError(
            "V2.11.10 scientific namespace contains stale dispatch outputs: "
            + ", ".join(sorted(set(stale))[:10])
        )
    raw_storage_bytes = 0
    invalid_storage_entries: list[str] = []
    for path in raw.rglob("*"):
        if path.is_symlink():
            invalid_storage_entries.append(path.relative_to(raw).as_posix())
        elif path.is_file():
            raw_storage_bytes += path.stat().st_size
        elif not path.is_dir():
            invalid_storage_entries.append(path.relative_to(raw).as_posix())
    max_storage = contract.budgets.get("max_storage_bytes")
    if (
        invalid_storage_entries
        or type(max_storage) is not int
        or raw_storage_bytes > max_storage
    ):
        raise PilotV21110ContinuationError(
            "V2.11.10 raw storage namespace violates its hard cap/type boundary: "
            + ", ".join(sorted(invalid_storage_entries)[:10])
        )
    return {
        "stage_id": stage_id,
        "registered_cells": len(specs),
        "terminal_cells": len(terminal_ids),
        "verified_terminal_artifact_bindings": len(terminal_ids),
        "fresh_pending_cells": len(pending),
        "scope": "collision-and-non-scientific-namespace-preflight",
        "raw_storage_bytes": raw_storage_bytes,
        "max_storage_bytes": max_storage,
        "provider_construction": False,
        "provider_calls": 0,
    }


def _verify_v21110_actor_journal(
    contract: PilotContract,
    spec: Any,
    *,
    repo_root: Path,
    raw_root: Path,
    artifact: str,
) -> None:
    """Bind an external actor journal back to its immutable run manifest."""

    from .runner import verify_provider_call_journal
    from .runner_artifacts import load_verified_run_artifacts

    manifest = Path(artifact)
    run_dir = manifest.parent if manifest.name == "manifest.json" else manifest
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise PilotV21110ContinuationError(
            f"{spec.run_id} completed runner directory is invalid"
        )
    result = load_verified_run_artifacts(
        run_dir,
        authority_repo_root=repo_root,
    )
    provenance = _strict_json(
        run_dir / "provenance.json", name=f"{spec.run_id} provenance"
    )
    details = provenance.get("details")
    journal_binding = (
        details.get("provider_call_journal") if isinstance(details, Mapping) else None
    )
    expected_path = (
        raw_root
        / spec.stage_id
        / "provider_call_journals"
        / f"{spec.run_id}--actor.json"
    )
    if (
        not isinstance(journal_binding, Mapping)
        or set(journal_binding)
        != {
            "path",
            "file_sha256",
            "journal_sha256",
            "run_id",
            "contract_hash",
            "event_count",
            "terminal_dispositions_verified",
        }
        or Path(str(journal_binding.get("path"))).absolute() != expected_path.absolute()
        or expected_path.is_symlink()
        or not expected_path.is_file()
        or expected_path.resolve() != expected_path.absolute()
    ):
        raise PilotV21110ContinuationError(
            f"{spec.run_id} actor journal binding is malformed"
        )
    journal = verify_provider_call_journal(
        expected_path,
        expected_run_id=spec.run_id,
        expected_contract_hash=contract.canonical_hash,
        require_terminal_dispositions=True,
    )
    recomputed = {
        "path": str(expected_path),
        "file_sha256": _file_sha256(expected_path),
        "journal_sha256": journal["journal_sha256"],
        "run_id": journal["run_id"],
        "contract_hash": journal["contract_hash"],
        "event_count": len(journal["events"]),
        "terminal_dispositions_verified": True,
    }
    completions = [
        event["payload"]
        for event in journal["events"]
        if event.get("event_type") == "completion_received"
    ]
    if canonical_sha256(journal_binding) != canonical_sha256(
        recomputed
    ) or canonical_sha256(completions) != canonical_sha256(
        result.records.get("api_usage")
    ):
        raise PilotV21110ContinuationError(
            f"{spec.run_id} actor journal differs from its sealed run"
        )


def _verify_v21110_failure_artifact(
    contract: PilotContract,
    spec: Any,
    row: Mapping[str, Any],
    *,
    raw_root: Path,
) -> None:
    """Verify both files and any partial journals behind a terminal failure."""

    from .failure_artifacts import verify_failure_receipt
    from .pilot_provider_catalog import verify_provider_catalog_receipt
    from .runner import verify_provider_call_journal

    artifact = row.get("artifact")
    failure = row.get("failure")
    if artifact is None:
        if not isinstance(failure, Mapping) or row.get("status") == "complete":
            raise PilotV21110ContinuationError(
                f"terminal run {spec.run_id} lacks its failure disposition"
            )
        return
    if not isinstance(artifact, str):
        raise PilotV21110ContinuationError(
            f"terminal run {spec.run_id} failure artifact is malformed"
        )
    path = Path(artifact)
    catalog_path = (
        raw_root / spec.stage_id / "provider_catalog" / f"{spec.model_id}.json"
    )
    if path.absolute() == catalog_path.absolute():
        receipt = verify_provider_catalog_receipt(
            _strict_json(path, name=f"{spec.run_id} provider no-go"),
            contract_hash=contract.canonical_hash,
            require_pass=False,
        )
        if (
            receipt.get("status") != "no-go"
            or receipt.get("model_id") != spec.model_id
            or type(receipt.get("paid_completions")) is not int
            or receipt.get("paid_completions") != 0
            or receipt.get("failure") != failure
        ):
            raise PilotV21110ContinuationError(
                f"{spec.run_id} provider no-go receipt drifted"
            )
        return

    if spec.stage_id == "experiment-d":
        group_root = raw_root / spec.stage_id / "checkpoints"
        if len(contract.models_for_stage(spec.stage_id)) > 1:
            group_root = group_root / spec.model_id
        expected = (
            group_root
            / f"s{spec.environment_seed}"
            / "failure_receipt"
            / "failure_manifest.json"
        )
    else:
        expected = (
            raw_root
            / spec.stage_id
            / "runs"
            / spec.run_id
            / "failure_receipt"
            / "failure_manifest.json"
        )
    if (
        path.absolute() != expected.absolute()
        or expected.is_symlink()
        or not expected.is_file()
        or expected.resolve() != expected.absolute()
    ):
        raise PilotV21110ContinuationError(
            f"{spec.run_id} uses a noncanonical failure artifact path"
        )
    failure_json = expected.parent / "failure.json"
    if (
        failure_json.is_symlink()
        or not failure_json.is_file()
        or failure_json.resolve() != failure_json.absolute()
        or not failure_json.resolve().is_relative_to(raw_root)
    ):
        raise PilotV21110ContinuationError(
            f"{spec.run_id} failure receipt payload path is invalid"
        )
    receipt = verify_failure_receipt(expected.parent)
    config = receipt.get("config")
    provenance = receipt.get("provenance")
    git = receipt.get("git")
    specs = config.get("run_specs") if isinstance(config, Mapping) else None
    paid = (
        provenance.get("paid_provenance") if isinstance(provenance, Mapping) else None
    )
    acceptance = _strict_json(
        raw_root / V21110_ACCEPTANCE_FILENAME,
        name="V2.11.10 scientific acceptance",
    )
    release = acceptance.get("release")
    if (
        not isinstance(config, Mapping)
        or config.get("contract_id") != contract.contract_id
        or config.get("contract_sha256") != contract.canonical_hash
        or not isinstance(specs, list)
        or canonical_sha256(spec.to_dict())
        not in {canonical_sha256(item) for item in specs}
        or not isinstance(provenance, Mapping)
        or provenance.get("contract_id") != contract.contract_id
        or provenance.get("contract_sha256") != contract.canonical_hash
        or provenance.get("diagnostic_only") is not False
        or provenance.get("scientific_evidence") is not False
        or not isinstance(paid, Mapping)
        or paid.get("git_tag") != V21110_SCIENCE_TAG
        or not isinstance(release, Mapping)
        or paid.get("head_commit") != release.get("git_commit")
        or paid.get("tag_commit") != release.get("git_commit")
        or paid.get("tag_object_type") != "tag"
        or paid.get("worktree_clean") is not True
        or not isinstance(git, Mapping)
        or git.get("commit") != release.get("git_commit")
        or git.get("dirty") is not False
    ):
        raise PilotV21110ContinuationError(
            f"{spec.run_id} failure receipt contract/release binding drifted"
        )
    journals = config.get("provider_call_journals")
    if not isinstance(journals, list):
        raise PilotV21110ContinuationError(
            f"{spec.run_id} failure receipt journal list is malformed"
        )
    for binding in journals:
        if not isinstance(binding, Mapping):
            raise PilotV21110ContinuationError(
                f"{spec.run_id} failure journal binding is malformed"
            )
        journal_path = Path(str(binding.get("path")))
        if (
            journal_path.is_symlink()
            or not journal_path.is_file()
            or journal_path.resolve() != journal_path.absolute()
            or not journal_path.resolve().is_relative_to(raw_root)
        ):
            raise PilotV21110ContinuationError(
                f"{spec.run_id} failure journal path is invalid"
            )
        journal = verify_provider_call_journal(
            journal_path,
            expected_run_id=str(binding.get("run_id")),
            expected_contract_hash=contract.canonical_hash,
            require_terminal_dispositions=True,
        )
        recomputed = {
            "path": str(journal_path),
            "file_sha256": _file_sha256(journal_path),
            "journal_sha256": journal["journal_sha256"],
            "run_id": journal["run_id"],
            "contract_hash": journal["contract_hash"],
            "event_count": len(journal["events"]),
            "terminal_dispositions_verified": True,
        }
        if canonical_sha256(binding) != canonical_sha256(recomputed):
            raise PilotV21110ContinuationError(
                f"{spec.run_id} failure journal binding drifted"
            )


def verify_v21110_terminal_scientific_artifacts(
    contract: PilotContract,
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    run_ledger: Any,
    paid: Any | None = None,
) -> dict[str, Any]:
    """Revalidate every completed scientific artifact before any new call."""

    from . import pilot_evidence
    from .pilot_checkpoint import (
        EXPERIMENT_D_SHARED_PREFIX_CHECKPOINT_PURPOSE,
        PILOT_CHECKPOINT_SCHEMA_VERSION_V3,
        PilotCheckpoint,
        current_code_binding,
        prepare_foundation_env_config,
    )

    repository = _real_root(repo_root, name="V2.11.10 repository")
    raw = _real_root(raw_root, name="V2.11.10 raw root")
    snapshot = run_ledger.snapshot()
    rows = snapshot.get("runs")
    if not isinstance(rows, Mapping):
        raise PilotV21110ContinuationError("V2.11.10 run ledger rows are malformed")
    completed: dict[str, tuple[Any, Mapping[str, Any], Mapping[str, Any]]] = {}
    checked_bindings = 0
    for stage_id in ("experiment-d", "experiment-b", "cross-model"):
        for spec in contract.expand(stage=stage_id):
            row = rows.get(spec.run_id)
            if not isinstance(row, Mapping):
                raise PilotV21110ContinuationError(
                    f"V2.11.10 ledger is missing {spec.run_id}"
                )
            status = row.get("status")
            failure = row.get("failure")
            if status not in {
                "complete",
                "failed",
                "capability-no-go",
                "budget-stopped",
                "integrity-stopped",
            }:
                continue
            if (status == "complete" and failure is not None) or (
                status != "complete" and not isinstance(failure, Mapping)
            ):
                raise PilotV21110ContinuationError(
                    f"terminal run {spec.run_id} has an invalid failure disposition"
                )
            run_ledger.verify_terminal_artifact_binding(spec.run_id)
            checked_bindings += 1
            if status != "complete":
                try:
                    _verify_v21110_failure_artifact(
                        contract,
                        spec,
                        row,
                        raw_root=raw,
                    )
                except Exception as exc:
                    if isinstance(exc, PilotV21110ContinuationError):
                        raise
                    raise PilotV21110ContinuationError(
                        f"terminal run {spec.run_id} failure replay failed: {exc}"
                    ) from exc
                continue
            artifact = row.get("artifact")
            if not isinstance(artifact, str):
                raise PilotV21110ContinuationError(
                    f"completed run {spec.run_id} lacks its artifact"
                )
            expected_artifact = (
                raw / spec.stage_id / "summaries" / f"{spec.run_id}.json"
                if spec.stage_id == "experiment-d"
                else raw / spec.stage_id / "runs" / spec.run_id / "manifest.json"
            )
            artifact_path = Path(artifact)
            if (
                artifact_path.absolute() != expected_artifact.absolute()
                or expected_artifact.is_symlink()
                or not expected_artifact.is_file()
                or expected_artifact.resolve() != expected_artifact.absolute()
            ):
                raise PilotV21110ContinuationError(
                    f"completed run {spec.run_id} uses a noncanonical artifact path"
                )
            try:
                evidence = pilot_evidence._load_completed_artifact(
                    contract,
                    spec.to_dict(),
                    raw_root=raw,
                    artifact=artifact,
                    source_repo_root=repository,
                )
            except Exception as exc:
                raise PilotV21110ContinuationError(
                    f"completed run {spec.run_id} failed semantic replay: {exc}"
                ) from exc
            if evidence.get("scientific_eligible") is not True:
                raise PilotV21110ContinuationError(
                    f"completed run {spec.run_id} is not scientific-eligible"
                )
            if spec.execution_mode in {"actor_run", "matched_duplicate"}:
                try:
                    _verify_v21110_actor_journal(
                        contract,
                        spec,
                        repo_root=repository,
                        raw_root=raw,
                        artifact=artifact,
                    )
                except Exception as exc:
                    if isinstance(exc, PilotV21110ContinuationError):
                        raise
                    raise PilotV21110ContinuationError(
                        f"completed run {spec.run_id} journal replay failed: {exc}"
                    ) from exc
            completed[spec.run_id] = (spec, row, evidence)

    d_specs = tuple(contract.expand(stage="experiment-d"))
    for model_id, seed in sorted(
        {(spec.model_id, spec.environment_seed) for spec in d_specs}
    ):
        group = tuple(
            spec
            for spec in d_specs
            if spec.model_id == model_id and spec.environment_seed == seed
        )
        terminal = tuple(
            spec
            for spec in group
            if rows[spec.run_id].get("status")
            in {
                "complete",
                "failed",
                "capability-no-go",
                "budget-stopped",
                "integrity-stopped",
            }
        )
        if terminal and len(terminal) != len(group):
            raise PilotV21110ContinuationError(
                f"Experiment D group {model_id}/s{seed} is only partly terminal"
            )
        if not terminal or any(
            rows[spec.run_id].get("status") != "complete" for spec in group
        ):
            continue
        if paid is None:
            raise PilotV21110ContinuationError(
                "Experiment D terminal replay requires paid release provenance"
            )
        group_root = raw / "experiment-d" / "checkpoints"
        if len(contract.models_for_stage("experiment-d")) > 1:
            group_root = group_root / model_id
        group_dir = group_root / f"s{seed}"
        checkpoint_path = group_dir / "checkpoint.json"
        if (
            group_dir.is_symlink()
            or not group_dir.is_dir()
            or checkpoint_path.is_symlink()
            or not checkpoint_path.is_file()
        ):
            raise PilotV21110ContinuationError(
                f"Experiment D group {model_id}/s{seed} lacks its checkpoint"
            )
        group_entries = {child.name: child for child in group_dir.iterdir()}
        expected_group_entries = {
            "checkpoint.json",
            "continuations.json",
            "narratives.json",
        }
        if set(group_entries) != expected_group_entries or any(
            child.is_symlink() or not child.is_file()
            for child in group_entries.values()
        ):
            raise PilotV21110ContinuationError(
                f"Experiment D group {model_id}/s{seed} does not contain its "
                "exact checkpoint/source inventory"
            )
        checkpoint = PilotCheckpoint.read_json(checkpoint_path)
        gates = [completed[spec.run_id][2].get("gate_evidence") for spec in group]
        if any(not isinstance(gate, Mapping) for gate in gates):
            raise PilotV21110ContinuationError(
                f"Experiment D group {model_id}/s{seed} lacks causal gates"
            )
        checkpoint_hashes = {gate.get("checkpoint_hash") for gate in gates}
        prefix_hashes = {gate.get("prefix_hash") for gate in gates}
        run_config = checkpoint.payload.get("run_config")
        from . import pilot_orchestrator as orch

        representative = next(spec for spec in group if spec.arm_id == "matched-a")
        base = orch.config_for_spec(
            contract,
            representative,
            raw_root=raw,
            paid_provenance=paid,
            authority_repo_root=repository,
            diagnostic_override=False,
            verify_bound_inputs=True,
            preflight_p95_reservations=orch._runner_p95_reservations(
                contract,
                representative.model_id,
                raw_root=raw,
                paid=paid,
                authority_repo_root=repository,
            ),
        )
        plan = orch.build_v21110_experiment_d_group_plan(
            contract,
            group,
            base_config=base,
        )
        if plan.prefix_config is None:
            raise PilotV21110ContinuationError(
                f"Experiment D group {model_id}/s{seed} lacks its prefix config"
            )
        expected_config = plan.prefix_config.to_dict()
        expected_foundation = prepare_foundation_env_config(
            repository / "config.yaml",
            n_agents=plan.prefix_config.num_agents,
            episode_length=plan.prefix_config.episode_length,
            labor_step=plan.prefix_config.labor_step,
            max_labor_hours=plan.prefix_config.max_labor_hours,
            consumption_step=plan.prefix_config.consumption_step,
        )
        prefix_journal_path = (
            raw
            / "experiment-d"
            / "provider_call_journals"
            / (
                f"{contract.contract_id}--experiment-d--{model_id}--"
                f"checkpoint-group--s{seed}--prefix.json"
            )
        )
        journal_binding = checkpoint.payload.get("provider_call_journal_binding")
        from .runner import verify_provider_call_journal

        if (
            prefix_journal_path.is_symlink()
            or not prefix_journal_path.is_file()
            or prefix_journal_path.resolve() != prefix_journal_path.absolute()
        ):
            raise PilotV21110ContinuationError(
                f"Experiment D group {model_id}/s{seed} prefix journal is invalid"
            )
        prefix_journal = verify_provider_call_journal(
            prefix_journal_path,
            expected_run_id=plan.prefix_config.run_id,
            expected_contract_hash=contract.canonical_hash,
            require_terminal_dispositions=True,
        )
        provider_binding = checkpoint.payload.get("provider_binding")
        if (
            checkpoint_hashes != {checkpoint.checkpoint_hash}
            or prefix_hashes != {checkpoint.payload.get("prefix_hash")}
            or checkpoint.payload.get("schema_version")
            != PILOT_CHECKPOINT_SCHEMA_VERSION_V3
            or checkpoint.payload.get("checkpoint_purpose")
            != EXPERIMENT_D_SHARED_PREFIX_CHECKPOINT_PURPOSE
            or not isinstance(run_config, Mapping)
            or canonical_sha256(run_config) != canonical_sha256(expected_config)
            or canonical_sha256(checkpoint.payload.get("foundation_env_config"))
            != canonical_sha256(expected_foundation)
            or canonical_sha256(checkpoint.payload.get("code_binding"))
            != canonical_sha256(current_code_binding())
            or not isinstance(provider_binding, Mapping)
            or provider_binding.get("model_name")
            != orch._runtime_model_for_profile(contract.provider_profiles[model_id])
            or not isinstance(journal_binding, Mapping)
            or journal_binding.get("path_name") != prefix_journal_path.name
            or journal_binding.get("journal_sha256") != prefix_journal["journal_sha256"]
            or journal_binding.get("event_count") != len(prefix_journal["events"])
        ):
            raise PilotV21110ContinuationError(
                f"Experiment D group {model_id}/s{seed} checkpoint binding drifted"
            )
        for spec in group:
            summary_path = Path(str(rows[spec.run_id]["artifact"]))
            summary = _strict_json(
                summary_path,
                name=f"{spec.run_id} Experiment D summary",
            )
            payload = summary.get("payload")
            expected_source = group_dir / (
                "narratives.json"
                if spec.arm_id == "narrative-content"
                else "continuations.json"
            )
            expected_branch_journal = (
                raw
                / "experiment-d"
                / "provider_call_journals"
                / f"{spec.run_id}--actor.json"
            )
            if (
                not isinstance(payload, Mapping)
                or Path(str(payload.get("shared_source"))).absolute()
                != expected_source.absolute()
                or not isinstance(payload.get("provider_call_journal"), Mapping)
                or Path(str(payload["provider_call_journal"].get("path"))).absolute()
                != expected_branch_journal.absolute()
            ):
                raise PilotV21110ContinuationError(
                    f"Experiment D run {spec.run_id} uses noncanonical shared paths"
                )
    return {
        "terminal_artifact_bindings": checked_bindings,
        "completed_semantic_replays": len(completed),
        "provider_construction": False,
        "provider_calls": 0,
    }


def parent_budget_debit_for_v21110(contract: Any) -> ParentBudgetDebit:
    if getattr(contract, "contract_id", None) != V21110_CONTRACT_ID:
        raise PilotV21110ContinuationError("parent debit requires V2.11.10")
    expected = {
        "parent_contract_sha256": V2119_CONTRACT_SHA256,
        "parent_run_ledger_sha256": V2119_RUN_LEDGER_SHA256,
        "parent_budget_ledger_sha256": V2119_BUDGET_LEDGER_SHA256,
        "stage_bucket": "parent_v2119",
        "cost_usd": V2119_CUMULATIVE_COST_USD,
        "hosted_completions": V2119_CUMULATIVE_COMPLETIONS,
        "storage_bytes": V2119_CUMULATIVE_STORAGE_BYTES,
    }
    declared = _boundary(contract).get("parent_budget_debit")
    if not isinstance(declared, Mapping) or any(
        declared.get(key) != value for key, value in expected.items()
    ):
        raise PilotV21110ContinuationError("V2.11.10 parent budget debit drifted")
    debit = ParentBudgetDebit(**expected)
    if debit.record_sha256 != V21110_PARENT_DEBIT_RECORD_SHA256:
        raise PilotV21110ContinuationError("V2.11.10 parent debit seal drifted")
    return debit


def current_authority_path(raw_root: str | Path) -> Path:
    return Path(raw_root) / "parent-import/current_authority/post_gate_authority.json"


def current_projection_path(raw_root: str | Path, model_id: str) -> Path:
    if model_id not in {"gpt52_main", "gpt56_diagnostic"}:
        raise PilotV21110ContinuationError(
            f"unsupported V2.11.10 continuation model {model_id}"
        )
    return (
        Path(raw_root)
        / f"parent-import/current_authority/{model_id}/projection_p95.json"
    )


def _tracked_source_manifest(
    contract: Any, *, repo_root: str | Path
) -> tuple[Path, dict[str, Any]]:
    repository = _real_root(repo_root, name="V2.11.10 repository")
    path = repository.joinpath(*V21110_SOURCE_MANIFEST_PATH.parts)
    value = _strict_json(path, name="V2.11.10 source manifest")
    _verify_seal(value, name="V2.11.10 source manifest")
    declared = _boundary(contract).get("source_manifest")
    if (
        not isinstance(declared, Mapping)
        or declared.get("path") != V21110_SOURCE_MANIFEST_PATH.as_posix()
        or declared.get("schema_version") != V21110_SOURCE_MANIFEST_SCHEMA_VERSION
        or declared.get("file_sha256") != _file_sha256(path)
        or declared.get("content_sha256")
        != value.get("integrity", {}).get("content_sha256")
        or value.get("schema_version") != V21110_SOURCE_MANIFEST_SCHEMA_VERSION
        or value.get("contract_id") != V21110_CONTRACT_ID
        or value.get("release_tag") != V21110_SCIENCE_TAG
    ):
        raise PilotV21110ContinuationError(
            "V2.11.10 tracked source-manifest binding drifted"
        )
    return path, value


def _normalize_authority_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    value = _json_copy(spec)
    run_id = str(value.get("run_id", ""))
    prefix = f"{v2117.V2115_CONTRACT_ID}--"
    if not run_id.startswith(prefix):
        raise PilotV21110ContinuationError("V2.11.5 scheduled run id is malformed")
    value["run_id"] = f"{V21110_CONTRACT_ID}--{run_id[len(prefix):]}"
    value["contract_id"] = V21110_CONTRACT_ID
    value["budget_bucket"] = "hosted_v21110"
    return value


def _canonical_remaining_cell_mapping(
    child_contract: Any, authority_contract: PilotContract
) -> dict[str, Any]:
    source_specs = tuple(
        spec
        for stage_id in ("experiment-d", "experiment-b", "cross-model")
        for spec in authority_contract.expand(stage=stage_id)
    )
    child_specs = tuple(
        spec
        for stage_id in ("experiment-d", "experiment-b", "cross-model")
        for spec in child_contract.expand(stage=stage_id)
    )
    child_by_id = {spec.run_id: spec.to_dict() for spec in child_specs}
    if (
        len(source_specs) != V2119_REMAINING_SCIENCE_CELL_COUNT
        or len(child_specs) != V2119_REMAINING_SCIENCE_CELL_COUNT
        or len(child_by_id) != V2119_REMAINING_SCIENCE_CELL_COUNT
    ):
        raise PilotV21110ContinuationError(
            "V2.11.10 canonical mapping denominator drifted"
        )
    rows: list[dict[str, Any]] = []
    for source_spec in sorted(source_specs, key=lambda item: item.run_id):
        source = source_spec.to_dict()
        child = _normalize_authority_spec(source)
        if child_by_id.get(child["run_id"]) != child:
            raise PilotV21110ContinuationError(
                f"V2.11.10 child spec differs for {source_spec.run_id}"
            )
        logical = _json_copy(source)
        logical.pop("run_id")
        logical.pop("contract_id")
        logical["budget_bucket"] = "normalized-hosted-continuation"
        rows.append(
            {
                "source_run_id": source_spec.run_id,
                "child_run_id": child["run_id"],
                "logical_cell_sha256": canonical_sha256(logical),
                "source_spec_sha256": canonical_sha256(source),
                "child_spec_sha256": canonical_sha256(child),
                "normalized_spec": logical,
            }
        )
    mapping = {
        "schema_version": "finevo-pilot-v2.11.10-canonical-cell-mapping-v1",
        "row_count": len(rows),
        "mapping_sha256": canonical_sha256(rows),
        "rows": rows,
    }
    declared = _boundary(child_contract)["continuation_matrix"].get(
        "canonical_86_row_mapping_sha256"
    )
    if (
        mapping["mapping_sha256"] != declared
        or len({row["source_run_id"] for row in rows}) != len(rows)
        or len({row["child_run_id"] for row in rows}) != len(rows)
        or len({row["logical_cell_sha256"] for row in rows}) != len(rows)
    ):
        raise PilotV21110ContinuationError("V2.11.10 canonical mapping identity drifted")
    return mapping


def _source_file_binding(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise PilotV21110ContinuationError(
            f"required V2.11.10 source is unavailable: {relative}"
        )
    return {
        "path": relative,
        "byte_size": path.stat().st_size,
        "file_sha256": _file_sha256(path),
    }


_CYCLIC_V21110_CONTRACT_PIN_NAMES = frozenset(
    {
        "PILOT_CONTRACT_V2_11_10_CANONICAL_SHA256",
        "PILOT_V2_11_10_SOURCE_MANIFEST_FILE_SHA256",
        "PILOT_V2_11_10_SOURCE_MANIFEST_CONTENT_SHA256",
    }
)

_V21110_NORMALIZED_AST_SOURCE_PATHS = frozenset(
    {
        "verified_memory/ci_release_receipt.py",
        "verified_memory/pilot_contract.py",
    }
)
_V21110_RELEASE_PYTHON_ENTRY_PATHS = frozenset(
    {
        "llm_providers.py",
        "run_pilot.py",
        "scripts/render_pilot_v21110_contract.py",
        "scripts/render_pilot_v21110_source_manifest.py",
    }
)
_V21110_BOUND_DATA_PATHS = (
    "config.yaml",
    V21110_PROFILE_PATH.as_posix(),
)


def _is_lower_sha256_constant(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and type(node.value) is str
        and len(node.value) == 64
        and set(node.value) <= set("0123456789abcdef")
    )


def _regular_python_tree_paths(root: Path, relative_directory: str) -> tuple[str, ...]:
    directory = root / relative_directory
    if directory.is_symlink() or not directory.is_dir():
        raise PilotV21110ContinuationError(
            f"required V2.11.10 source directory is unavailable: {relative_directory}"
        )
    paths: list[str] = []
    for path in directory.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise PilotV21110ContinuationError(
                f"V2.11.10 source inventory contains a symlink: {relative}"
            )
        if path.is_file() and path.suffix == ".py":
            paths.append(relative)
    return tuple(sorted(paths))


def _v21110_release_python_source_paths(root: Path) -> tuple[str, ...]:
    """Inventory the complete local Python release/runtime source surface."""

    paths = set(_V21110_RELEASE_PYTHON_ENTRY_PATHS)
    for directory in ("verified_memory", "ai_economist/foundation"):
        paths.update(_regular_python_tree_paths(root, directory))
    missing = sorted(relative for relative in paths if not (root / relative).is_file())
    if missing:
        raise PilotV21110ContinuationError(
            "V2.11.10 release Python source inventory is incomplete: "
            + ", ".join(missing[:10])
        )
    return tuple(sorted(paths))


def _v21110_foundation_source_paths(root: Path) -> tuple[str, ...]:
    return _regular_python_tree_paths(root, "ai_economist/foundation")


def _normalized_contract_module_ast_binding(
    path: Path,
    *,
    require_v21110_cycle_pins: bool,
) -> dict[str, Any]:
    """Hash the complete module AST while breaking only the three hash cycles."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise PilotV21110ContinuationError(
            "cannot normalize the pilot contract module AST"
        ) from exc
    replaced: list[str] = []
    bootstrap_none_fields: list[str] = []
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        elif isinstance(node, ast.Assign):
            names = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
        cyclic = [name for name in names if name in _CYCLIC_V21110_CONTRACT_PIN_NAMES]
        if not cyclic:
            continue
        if len(cyclic) != 1 or len(names) != 1:
            raise PilotV21110ContinuationError(
                "V2.11.10 cyclic contract pin assignment is ambiguous"
            )
        pin_name = cyclic[0]
        bootstrap_none = bool(
            isinstance(node.value, ast.Constant) and node.value.value is None
        )
        if not (_is_lower_sha256_constant(node.value) or bootstrap_none):
            raise PilotV21110ContinuationError(
                "V2.11.10 cyclic contract pin must be one literal lowercase SHA-256; "
                "literal None is allowed only while bootstrapping the release pins"
            )
        if bootstrap_none:
            bootstrap_none_fields.append(pin_name)
        node.value = ast.Constant(value="<v21110-release-cycle-pin>")
        replaced.extend(cyclic)
    if len(replaced) != len(set(replaced)):
        raise PilotV21110ContinuationError(
            "V2.11.10 cyclic contract pin is assigned more than once"
        )
    if require_v21110_cycle_pins and set(replaced) != set(
        _CYCLIC_V21110_CONTRACT_PIN_NAMES
    ):
        raise PilotV21110ContinuationError(
            "V2.11.10 cyclic contract pin set is incomplete"
        )
    if not require_v21110_cycle_pins and replaced:
        raise PilotV21110ContinuationError(
            "historical contract module unexpectedly contains V2.11.10 cycle pins"
        )
    allowed_bootstrap_none_sets = {
        frozenset(),
        frozenset(_CYCLIC_V21110_CONTRACT_PIN_NAMES),
        frozenset({"PILOT_CONTRACT_V2_11_10_CANONICAL_SHA256"}),
    }
    if frozenset(bootstrap_none_fields) not in allowed_bootstrap_none_sets:
        raise PilotV21110ContinuationError(
            "V2.11.10 contract source pins must both be None or both be strict "
            "SHA-256 values; the canonical pin may remain None during bootstrap"
        )
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return {
        "normalization_schema_version": (
            "finevo-pilot-v2.11.10-complete-module-ast-cycle-normalization-v1"
        ),
        "normalized_ast_sha256": canonical_sha256(normalized),
        "top_level_node_count": len(tree.body),
        "replaced_cycle_pins": sorted(replaced),
    }


def _normalized_ci_release_module_ast_binding(path: Path) -> dict[str, Any]:
    """Hash the complete CI module while breaking only its V2.11.10 anchor cycle."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise PilotV21110ContinuationError(
            "cannot normalize the CI release module AST"
        ) from exc
    assignments = []
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        elif isinstance(node, ast.Assign):
            names = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
        else:
            names = []
        if "SCIENTIFIC_SOURCE_MANIFEST_ANCHORS" in names:
            assignments.append(node)
    if len(assignments) != 1:
        raise PilotV21110ContinuationError(
            "CI source-manifest anchor assignment is ambiguous"
        )
    replaced: list[str] = []
    bootstrap_none_fields: list[str] = []
    matching_rows = 0
    for node in ast.walk(assignments[0]):
        if not isinstance(node, ast.Dict):
            continue
        indices = {
            key.value: index
            for index, key in enumerate(node.keys)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        path_index = indices.get("path")
        if path_index is None:
            continue
        path_value = node.values[path_index]
        if (
            not isinstance(path_value, ast.Constant)
            or path_value.value != V21110_SOURCE_MANIFEST_PATH.as_posix()
        ):
            continue
        matching_rows += 1
        for field in ("file_sha256", "content_sha256"):
            index = indices.get(field)
            if index is None:
                raise PilotV21110ContinuationError(
                    "V2.11.10 CI source-manifest anchor is incomplete"
                )
            bootstrap_none = bool(
                isinstance(node.values[index], ast.Constant)
                and node.values[index].value is None
            )
            if not (_is_lower_sha256_constant(node.values[index]) or bootstrap_none):
                raise PilotV21110ContinuationError(
                    "V2.11.10 CI source-manifest cycle pin must be one literal "
                    "lowercase SHA-256 or bootstrap None"
                )
            if bootstrap_none:
                bootstrap_none_fields.append(field)
            node.values[index] = ast.Constant(value="<v21110-release-cycle-pin>")
            replaced.append(field)
    if matching_rows != 1 or sorted(replaced) != ["content_sha256", "file_sha256"]:
        raise PilotV21110ContinuationError(
            "V2.11.10 CI source-manifest cycle pins are incomplete"
        )
    if bootstrap_none_fields and sorted(bootstrap_none_fields) != [
        "content_sha256",
        "file_sha256",
    ]:
        raise PilotV21110ContinuationError(
            "V2.11.10 CI source-manifest bootstrap pins must both be None or "
            "both be strict SHA-256 values"
        )
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return {
        "normalization_schema_version": (
            "finevo-pilot-v2.11.10-complete-ci-module-ast-cycle-normalization-v1"
        ),
        "normalized_ast_sha256": canonical_sha256(normalized),
        "top_level_node_count": len(tree.body),
        "replaced_cycle_pins": sorted(replaced),
    }


def _current_runtime_source_bindings(child: Path, authority: Path) -> dict[str, Any]:
    """Seal successor glue without creating a contract/manifest hash cycle."""

    python_source_paths = _v21110_release_python_source_paths(child)
    if not _V21110_NORMALIZED_AST_SOURCE_PATHS < set(python_source_paths):
        raise PilotV21110ContinuationError(
            "V2.11.10 normalized source paths are absent from the release inventory"
        )
    full_hash_paths = tuple(
        relative
        for relative in python_source_paths
        if relative not in _V21110_NORMALIZED_AST_SOURCE_PATHS
    )
    files = [_source_file_binding(child, relative) for relative in full_hash_paths]
    contract_relative = "verified_memory/pilot_contract.py"
    child_contract_binding = _normalized_contract_module_ast_binding(
        child / contract_relative,
        require_v21110_cycle_pins=True,
    )
    authority_contract_binding = _normalized_contract_module_ast_binding(
        authority / contract_relative,
        require_v21110_cycle_pins=False,
    )
    ci_relative = "verified_memory/ci_release_receipt.py"
    ci_binding = _normalized_ci_release_module_ast_binding(child / ci_relative)
    data_files = [
        _source_file_binding(child, relative) for relative in _V21110_BOUND_DATA_PATHS
    ]
    return {
        "release_python_source_paths": list(python_source_paths),
        "release_python_source_path_set_sha256": canonical_sha256(python_source_paths),
        "full_file_bindings": files,
        "full_file_binding_set_sha256": canonical_sha256(files),
        "pilot_contract_path": contract_relative,
        "pilot_contract_complete_module_ast_bindings": {
            "authority": authority_contract_binding,
            "child": child_contract_binding,
        },
        "ci_release_receipt_path": ci_relative,
        "ci_release_receipt_complete_module_ast_binding": ci_binding,
        "bound_data_files": data_files,
        "bound_data_file_set_sha256": canonical_sha256(data_files),
        "cycle_avoidance": (
            "pilot_contract.py and ci_release_receipt.py are bound by their "
            "complete normalized module ASTs; only the V2.11.10 canonical/source "
            "manifest hash-cycle values are replaced by a fixed sentinel"
        ),
    }


def _normalized_d_plan_receipts(
    child_contract: Any, authority_contract: PilotContract
) -> list[dict[str, Any]]:
    from . import pilot_orchestrator as orch

    rows: list[dict[str, Any]] = []
    for seed in child_contract.seeds["sets"]["main"]:
        child_specs = tuple(
            spec
            for spec in child_contract.expand(stage="experiment-d")
            if spec.environment_seed == seed
        )
        authority_specs = tuple(
            spec
            for spec in authority_contract.expand(stage="experiment-d")
            if spec.environment_seed == seed
        )
        child_plan = orch.build_v21110_experiment_d_group_plan(
            child_contract, child_specs
        )
        authority_plan = orch.build_v2115_experiment_d_group_plan(
            authority_contract, authority_specs
        )

        def normalized(plan: Any) -> dict[str, Any]:
            specs = tuple(plan.continuation_specs.values()) + tuple(
                plan.narrative_specs.values()
            )
            values: list[dict[str, Any]] = []
            for spec in specs:
                value = spec.to_dict()
                value.pop("run_id")
                value.pop("contract_id")
                value["budget_bucket"] = "normalized-hosted-continuation"
                values.append(value)
            return {
                "seed": plan.representative.environment_seed,
                "registered_treatments": list(plan.registered_treatments),
                "specs": sorted(
                    values,
                    key=lambda value: (value["arm_id"], value["narrative_id"]),
                ),
            }

        child_value = normalized(child_plan)
        authority_value = normalized(authority_plan)
        if child_value != authority_value:
            raise PilotV21110ContinuationError(
                f"V2.11.10 D plan differs from V2.11.5 at seed {seed}"
            )
        rows.append(
            {
                "seed": seed,
                "normalized_plan_sha256": canonical_sha256(child_value),
            }
        )
    if len(rows) != 5:
        raise PilotV21110ContinuationError("V2.11.10 D plan lacks five seeds")
    return rows


def _remaining_science_implementation_equivalence(
    child: Path,
    authority: Path,
    *,
    child_contract: Any,
    authority_contract: PilotContract,
) -> dict[str, Any]:
    identical: list[dict[str, Any]] = []
    for relative in v2117._BYTE_IDENTICAL_SCIENCE_PATHS:
        child_binding = _source_file_binding(child, relative)
        authority_binding = _source_file_binding(authority, relative)
        if child_binding["file_sha256"] != authority_binding["file_sha256"]:
            raise PilotV21110ContinuationError(
                f"remaining-science source differs from V2.11.5: {relative}"
            )
        identical.append(child_binding)
    child_foundation_paths = _v21110_foundation_source_paths(child)
    authority_foundation_paths = _v21110_foundation_source_paths(authority)
    if child_foundation_paths != authority_foundation_paths:
        raise PilotV21110ContinuationError(
            "V2.11.10 Foundation source inventory differs from V2.11.5"
        )
    environment_identical: list[dict[str, Any]] = []
    for relative in (*child_foundation_paths, V21110_PROFILE_PATH.as_posix()):
        child_binding = _source_file_binding(child, relative)
        authority_binding = _source_file_binding(authority, relative)
        if child_binding != authority_binding:
            raise PilotV21110ContinuationError(
                f"V2.11.10 environment source differs from V2.11.5: {relative}"
            )
        environment_identical.append(child_binding)
    orchestrator_relative = "verified_memory/pilot_orchestrator.py"
    try:
        child_functions = v2117._ast_function_digests(
            child / orchestrator_relative,
            v2117._UNCHANGED_ORCHESTRATOR_AST_FUNCTIONS,
        )
        authority_functions = v2117._ast_function_digests(
            authority / orchestrator_relative,
            v2117._UNCHANGED_ORCHESTRATOR_AST_FUNCTIONS,
        )
    except (OSError, SyntaxError, v2117.PilotV2117ContinuationError) as exc:
        raise PilotV21110ContinuationError(
            "cannot verify V2.11.10 science-function equivalence"
        ) from exc
    if child_functions != authority_functions:
        changed = sorted(
            name
            for name in child_functions
            if child_functions[name] != authority_functions.get(name)
        )
        raise PilotV21110ContinuationError(
            f"remaining-science orchestrator functions drifted: {changed}"
        )
    plans = _normalized_d_plan_receipts(child_contract, authority_contract)
    return {
        "policy": "science-core-equal-with-observed-p95-authority-adapter-v1",
        "byte_identical_files": identical,
        "byte_identical_files_sha256": canonical_sha256(identical),
        "environment_byte_identical_files": environment_identical,
        "environment_byte_identical_file_set_sha256": canonical_sha256(
            environment_identical
        ),
        "orchestrator_path": orchestrator_relative,
        "unchanged_orchestrator_function_sha256": child_functions,
        "unchanged_orchestrator_set_sha256": canonical_sha256(child_functions),
        "experiment_d_normalized_plan_receipts": plans,
        "experiment_d_normalized_plan_set_sha256": canonical_sha256(plans),
        "equivalence_claim": (
            "science_core_equal_with_observed_p95_authority_adapter"
        ),
        "full_runtime_byte_identity_claimed": False,
    }


def build_v21110_source_manifest(
    *,
    contract: PilotContract,
    repo_root: str | Path,
    failed_repo_root: str | Path,
    authority_repo_root: str | Path,
) -> dict[str, Any]:
    """Build the deterministic V2.11.10 dual-root source manifest."""

    if contract.contract_id != V21110_CONTRACT_ID:
        raise PilotV21110ContinuationError("source manifest requires V2.11.10")
    child = _real_root(repo_root, name="V2.11.10 repository")
    failed = _real_root(failed_repo_root, name="V2.11.9 failed repository")
    authority = _real_root(authority_repo_root, name="V2.11.5 authority repository")
    _require_distinct_roots(child=child, failed=failed, authority=authority)
    state = verify_v2119_terminal_no_go(
        failed_repo_root=failed,
        authority_repo_root=authority,
    )
    authority_state = state["authority"]
    authority_contract = authority_state["contract"]
    mapping = _canonical_remaining_cell_mapping(contract, authority_contract)
    expected_mapping = _boundary(contract)["continuation_matrix"].get(
        "canonical_86_row_mapping_sha256"
    )
    if mapping["mapping_sha256"] != expected_mapping:
        raise PilotV21110ContinuationError(
            "V2.11.10 source-manifest mapping binding drifted"
        )
    return _seal(
        {
            "schema_version": V21110_SOURCE_MANIFEST_SCHEMA_VERSION,
            "contract_id": V21110_CONTRACT_ID,
            # The frozen contract binds this manifest after bootstrap.  Keeping
            # its canonical hash out of this object avoids a hash cycle.
            "release_tag": str(contract.implementation["required_git_tag"]),
            "failed_release": {
                "contract_id": V2119_CONTRACT_ID,
                "contract_sha256": V2119_CONTRACT_SHA256,
                "contract_file_sha256": V2119_CONTRACT_FILE_SHA256,
                "source_manifest_path": V2119_SOURCE_MANIFEST_PATH.as_posix(),
                "source_manifest_file_sha256": (V2119_SOURCE_MANIFEST_FILE_SHA256),
                "source_manifest_content_sha256": (
                    V2119_SOURCE_MANIFEST_CONTENT_SHA256
                ),
                **state["failed_release"],
            },
            "failed_terminal_no_go": _expected_v2119_failed_release_no_go(),
            "failed_raw_inventory": state["raw_inventory"]["evidence"],
            "failed_complete_raw_inventory": state["raw_inventory"]["complete"],
            "authority_release": {
                **_expected_v2115_parent_release(),
                **authority_state["release"],
            },
            "authority_raw_inventory": authority_state["raw_inventory"],
            "authority_dispatch_artifacts": {
                "scientific_dispatch_acceptance": _expected_v2115_authority_release()[
                    "scientific_dispatch_acceptance"
                ],
                "preflight_authority": _expected_v2115_authority_release()[
                    "preflight_authority"
                ],
                "parent_import_receipt": {
                    "path": (
                        V2115_RAW_ROOT / "parent-import/parent_import_receipt.json"
                    ).as_posix(),
                    "file_sha256": v2117.V2115_PARENT_IMPORT_FILE_SHA256,
                    "content_sha256": v2117.V2115_PARENT_IMPORT_CONTENT_SHA256,
                },
            },
            "canonical_remaining_cell_mapping": mapping,
            "current_runtime_sources": _current_runtime_source_bindings(
                child, authority
            ),
            "remaining_science_implementation_equivalence": (
                _remaining_science_implementation_equivalence(
                    child,
                    authority,
                    child_contract=contract,
                    authority_contract=authority_contract,
                )
            ),
            "observed_p95_authority_adapter_recovery": {
                "failed_release_contract_id": V2119_CONTRACT_ID,
                "failure_error_type": V2119_FAILURE_ERROR_TYPE,
                "failure_message_by_model": dict(V2119_FAILURE_MESSAGE_BY_MODEL),
                "producer_core_authority_field_count": 13,
                "runner_envelope_authority_field_count": 17,
                "runner_receipt_envelope_field_count": 4,
                "authority_release_contract_id": v2117.V2115_CONTRACT_ID,
                "authority_gate_path": (
                    V2115_RAW_ROOT / "long-context-preflight/post_gate_authority.json"
                ).as_posix(),
                "authority_gate_content_sha256": V2115_POST_GATE_CONTENT_SHA256,
                "repair_changes_scientific_design": False,
                "scientific_outcomes_inspected_for_repair": False,
                "additional_source_release_required": False,
                "provider_construction": False,
                "provider_calls": 0,
            },
            "observation_boundary": {
                "failed_v2119_is_terminal_lineage_only": True,
                "failed_v2119_effect_rows_imported": 0,
                "authority_v2115_a_c_outcomes_are_frozen_external_evidence": True,
                "authority_v2115_a_c_rows_imported_into_child_ledger": 0,
                "authority_v2115_scheduled_cells_mapped_to_child": (
                    V2119_REMAINING_SCIENCE_CELL_COUNT
                ),
                "decoded_completion_reuse": False,
                "runtime_cwd_bound_to_release_root": True,
                "profile_input_path": V21110_PROFILE_PATH.as_posix(),
                "profile_input_file_sha256": V21110_PROFILE_FILE_SHA256,
                "profile_input_regular_non_symlink": True,
                "provider_calls": 0,
                "provider_construction": False,
            },
        }
    )


def validate_v21110_source_manifest(
    *,
    contract: PilotContract,
    repo_root: str | Path,
    failed_repo_root: str | Path,
    authority_repo_root: str | Path,
) -> dict[str, Any]:
    root = _real_root(repo_root, name="V2.11.10 repository")
    path = root.joinpath(*V21110_SOURCE_MANIFEST_PATH.parts)
    observed = _strict_json(path, name="V2.11.10 source manifest")
    _verify_seal(observed, name="V2.11.10 source manifest")
    expected = build_v21110_source_manifest(
        contract=contract,
        repo_root=root,
        failed_repo_root=failed_repo_root,
        authority_repo_root=authority_repo_root,
    )
    if observed != expected:
        raise PilotV21110ContinuationError("V2.11.10 source manifest replay drifted")
    declared = _boundary(contract).get("source_manifest")
    expected_binding = {
        "path": V21110_SOURCE_MANIFEST_PATH.as_posix(),
        "schema_version": V21110_SOURCE_MANIFEST_SCHEMA_VERSION,
        "file_sha256": _file_sha256(path),
        "content_sha256": expected["integrity"]["content_sha256"],
    }
    if (
        not isinstance(declared, Mapping)
        or {key: declared.get(key) for key in expected_binding} != expected_binding
    ):
        raise PilotV21110ContinuationError(
            "V2.11.10 contract source-manifest identity drifted"
        )
    return observed


def _capability_summary(wrapper: Mapping[str, Any], *, model_id: str) -> dict[str, Any]:
    try:
        return v2119._capability_summary(wrapper, model_id=model_id)
    except v2119.PilotV2119ContinuationError as exc:
        raise PilotV21110ContinuationError(str(exc)) from exc


def _dispatch_authority_source(authority: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return v2119._dispatch_authority_source(authority)
    except v2119.PilotV2119ContinuationError as exc:
        raise PilotV21110ContinuationError(str(exc)) from exc


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    try:
        v2119._write_once(path, value)
    except v2119.PilotV2119ContinuationError as exc:
        raise PilotV21110ContinuationError(str(exc)) from exc


def _build_current_authority(
    *,
    contract: Any,
    raw_root: Path,
    paid: Any,
    authority_state: Mapping[str, Any],
    parent_import_content_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    gate = authority_state.get("gate_binding")
    source = gate.get("reservations") if isinstance(gate, Mapping) else None
    if not isinstance(source, Mapping):
        raise PilotV21110ContinuationError("V2.11.5 p95 authority is malformed")
    numeric: dict[str, dict[str, Any]] = {}
    stable: dict[str, dict[str, Any]] = {}
    for runtime_model, by_kind in source.items():
        if not isinstance(by_kind, Mapping) or set(by_kind) != {"action", "semantic"}:
            raise PilotV21110ContinuationError("V2.11.5 call-kind denominator drifted")
        numeric[str(runtime_model)] = {}
        stable[str(runtime_model)] = {}
        for kind in ("action", "semantic"):
            entry = by_kind[kind]
            authority = entry.get("authority") if isinstance(entry, Mapping) else None
            reservation = (
                entry.get("reservation") if isinstance(entry, Mapping) else None
            )
            if not isinstance(authority, Mapping) or not isinstance(
                reservation, Mapping
            ):
                raise PilotV21110ContinuationError("V2.11.5 p95 row is malformed")
            numeric[str(runtime_model)][kind] = _json_copy(reservation)
            stable[str(runtime_model)][kind] = {
                key: value
                for key, value in authority.items()
                if key
                not in {
                    "pilot_contract_hash",
                    "pilot_tag",
                    "source_projection_file_sha256",
                    "source_projection_content_sha256",
                }
            }
    authority = _seal(
        {
            "schema_version": V21110_CURRENT_AUTHORITY_SCHEMA_VERSION,
            "contract_id": V21110_CONTRACT_ID,
            "contract_sha256": contract.canonical_hash,
            "release": {"git_tag": paid.git_tag, "git_commit": paid.head_commit},
            "authority_release": {
                "contract_id": v2117.V2115_CONTRACT_ID,
                "contract_sha256": v2117.V2115_CONTRACT_SHA256,
                "git_tag": v2117.V2115_SCIENCE_TAG,
                "git_commit": v2117.V2115_SCIENCE_COMMIT,
                "source_gate": {
                    "path": gate["receipt_path"],
                    "file_sha256": gate["receipt_file_sha256"],
                    "content_sha256": gate["receipt_content_sha256"],
                },
            },
            "parent_import_content_sha256": parent_import_content_sha256,
            "reservations": numeric,
            "stable_source_authorities": stable,
            "provider_boundary": _json_copy(_CURRENT_AUTHORITY_PROVIDER_BOUNDARY),
            "scientific_evidence": False,
            "claim_boundary": _CURRENT_AUTHORITY_CLAIM_BOUNDARY,
        }
    )
    authority_path = current_authority_path(raw_root)
    _write_once(authority_path, authority)
    authority_file = _file_sha256(authority_path)
    authority_content = authority["integrity"]["content_sha256"]
    projections: dict[str, dict[str, Any]] = {}
    for model_id in ("gpt52_main", "gpt56_diagnostic"):
        profile = contract.provider_profiles[model_id]
        runtime = (
            f"{profile.transport}/{profile.served_model}"
            if profile.transport != "openrouter"
            else f"thirdparty/{profile.served_model}"
        )
        rows = numeric.get(runtime)
        if not isinstance(rows, Mapping):
            raise PilotV21110ContinuationError(
                f"V2.11.10 current authority lacks {model_id}/{runtime}"
            )
        projection = _seal(
            {
                "schema_version": V21110_CURRENT_PROJECTION_SCHEMA_VERSION,
                "model_id": model_id,
                "runtime_model": runtime,
                "served_model": profile.served_model,
                "projection": {
                    f"{profile.served_model}::{kind}": _json_copy(rows[kind])
                    for kind in ("action", "semantic")
                },
                "bindings": {
                    "contract_sha256": contract.canonical_hash,
                    "git_tag": paid.git_tag,
                    "git_commit": paid.head_commit,
                    "authority_path": _CURRENT_AUTHORITY_PATH.as_posix(),
                    "authority_file_sha256": authority_file,
                    "authority_content_sha256": authority_content,
                    "parent_import_content_sha256": parent_import_content_sha256,
                },
                "provider_calls": 0,
                "provider_construction": False,
                "scientific_evidence": False,
                "claim_boundary": _CURRENT_PROJECTION_CLAIM_BOUNDARY,
            }
        )
        _write_once(current_projection_path(raw_root, model_id), projection)
        projections[model_id] = projection
    return authority, projections


def verify_v21110_current_authority(
    *,
    contract: Any,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: Any,
) -> dict[str, Any]:
    repository = _real_root(repo_root, name="V2.11.10 repository")
    raw = _real_root(raw_root, name="V2.11.10 raw root")
    parent = verify_v21110_parent_import_receipt(
        raw / "parent-import/parent_import_receipt.json",
        contract=contract,
        repo_root=repository,
        raw_root=raw,
        paid=paid,
    )
    path = current_authority_path(raw)
    authority = _strict_json(path, name="V2.11.10 current p95 authority")
    _verify_seal(authority, name="V2.11.10 current p95 authority")
    dispatch = parent.get("dispatch_authority_source")
    release = _release_binding(contract, paid)
    expected_source = (
        dispatch.get("source_gate") if isinstance(dispatch, Mapping) else None
    )
    expected_release = {
        "git_tag": release["git_tag"],
        "git_commit": release["git_commit"],
    }
    expected_authority_release = {
        "contract_id": v2117.V2115_CONTRACT_ID,
        "contract_sha256": v2117.V2115_CONTRACT_SHA256,
        "git_tag": v2117.V2115_SCIENCE_TAG,
        "git_commit": v2117.V2115_SCIENCE_COMMIT,
        "source_gate": expected_source,
    }
    expected_reservations = (
        dispatch.get("reservations") if isinstance(dispatch, Mapping) else None
    )
    expected_stable = (
        dispatch.get("stable_source_authorities")
        if isinstance(dispatch, Mapping)
        else None
    )
    if (
        set(authority) != _CURRENT_AUTHORITY_TOP_LEVEL_FIELDS
        or authority.get("schema_version") != V21110_CURRENT_AUTHORITY_SCHEMA_VERSION
        or authority.get("contract_id") != V21110_CONTRACT_ID
        or authority.get("contract_sha256") != contract.canonical_hash
        or canonical_sha256(authority.get("release"))
        != canonical_sha256(expected_release)
        or canonical_sha256(authority.get("authority_release"))
        != canonical_sha256(expected_authority_release)
        or authority.get("parent_import_content_sha256")
        != parent["integrity"]["content_sha256"]
        or canonical_sha256(authority.get("reservations"))
        != canonical_sha256(expected_reservations)
        or canonical_sha256(authority.get("stable_source_authorities"))
        != canonical_sha256(expected_stable)
        or canonical_sha256(authority.get("provider_boundary"))
        != canonical_sha256(_CURRENT_AUTHORITY_PROVIDER_BOUNDARY)
        or authority.get("scientific_evidence") is not False
        or authority.get("claim_boundary") != _CURRENT_AUTHORITY_CLAIM_BOUNDARY
    ):
        raise PilotV21110ContinuationError("V2.11.10 current p95 authority drifted")
    return authority


def verified_v21110_projection(
    contract: Any,
    model_id: str,
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: Any,
) -> tuple[dict[str, Any], Path]:
    authority = verify_v21110_current_authority(
        contract=contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
    )
    raw = _real_root(raw_root, name="V2.11.10 raw root")
    path = current_projection_path(raw, model_id)
    value = _strict_json(path, name=f"V2.11.10 {model_id} p95 projection")
    _verify_seal(value, name=f"V2.11.10 {model_id} p95 projection")
    profile = contract.provider_profiles[model_id]
    runtime = (
        f"{profile.transport}/{profile.served_model}"
        if profile.transport != "openrouter"
        else f"thirdparty/{profile.served_model}"
    )
    rows = authority.get("reservations", {}).get(runtime)
    expected_projection = (
        {
            f"{profile.served_model}::{kind}": rows[kind]
            for kind in ("action", "semantic")
        }
        if isinstance(rows, Mapping)
        else None
    )
    bindings = value.get("bindings")
    expected_bindings = {
        "contract_sha256": contract.canonical_hash,
        "git_tag": paid.git_tag,
        "git_commit": paid.head_commit,
        "authority_path": _CURRENT_AUTHORITY_PATH.as_posix(),
        "authority_file_sha256": _file_sha256(current_authority_path(raw)),
        "authority_content_sha256": authority["integrity"]["content_sha256"],
        "parent_import_content_sha256": authority["parent_import_content_sha256"],
    }
    if (
        set(value) != _CURRENT_PROJECTION_TOP_LEVEL_FIELDS
        or value.get("schema_version") != V21110_CURRENT_PROJECTION_SCHEMA_VERSION
        or value.get("model_id") != model_id
        or value.get("runtime_model") != runtime
        or value.get("served_model") != profile.served_model
        or canonical_sha256(value.get("projection"))
        != canonical_sha256(expected_projection)
        or canonical_sha256(bindings) != canonical_sha256(expected_bindings)
        or type(value.get("provider_calls")) is not int
        or value.get("provider_calls") != 0
        or value.get("provider_construction") is not False
        or value.get("scientific_evidence") is not False
        or value.get("claim_boundary") != _CURRENT_PROJECTION_CLAIM_BOUNDARY
    ):
        raise PilotV21110ContinuationError(
            f"V2.11.10 {model_id} current p95 projection drifted"
        )
    return value, path


def verified_v21110_calibration(
    contract: Any,
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: Any,
) -> dict[str, Any]:
    receipt = verify_v21110_parent_import_receipt(
        Path(raw_root) / "parent-import/parent_import_receipt.json",
        contract=contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
    )
    wrapper = receipt.get("calibration_wrapper")
    calibration = wrapper.get("calibration") if isinstance(wrapper, Mapping) else None
    selected = (
        calibration.get("selected_utility_profile")
        if isinstance(calibration, Mapping)
        else None
    )
    threshold = (
        calibration.get("stage0_absolute_flow_utility_threshold")
        if isinstance(calibration, Mapping)
        else None
    )
    if (
        not isinstance(calibration, Mapping)
        or calibration.get("q_ref") != v2117.V2115_Q_REF
        or not isinstance(selected, Mapping)
        or selected.get("profile_id") != "nu-0.5"
        or selected.get("rho") != 1.0
        or selected.get("labor_weight") != 2.0
        or selected.get("inverse_frisch") != 0.5
        or selected.get("consumption_scale") != v2117.V2115_Q_REF
        or selected.get("discount_factor") != 0.99
        or not isinstance(threshold, Mapping)
        or threshold.get("value") != v2117.V2115_ABSOLUTE_FLOW_UTILITY_THRESHOLD
        or threshold.get("treatment_outcomes_inspected") is not False
        or wrapper.get("provider_construction_during_import") is not False
        or wrapper.get("provider_calls_during_import") != 0
        or wrapper.get("imported_effect_cells") != 0
        or wrapper.get("scientific_evidence") is not False
    ):
        raise PilotV21110ContinuationError("V2.11.10 calibration authority drifted")
    return {
        "receipt": receipt,
        "wrapper": wrapper,
        "q_ref": v2117.V2115_Q_REF,
        "selected_profile_id": "nu-0.5",
        "selected_utility": {
            key: value for key, value in selected.items() if key != "profile_id"
        },
        "absolute_flow_utility_threshold": _json_copy(threshold),
    }


def verified_v21110_observed_p95_authority_binding(
    receipt_path: str | Path,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
    expected_contract_sha256: str,
) -> dict[str, Any]:
    repository = _real_root(repo_root, name="V2.11.10 repository")
    contract = load_pilot_contract(repository / "experiments/pilot_v2_11_10.yaml")
    if (
        contract.contract_id != V21110_CONTRACT_ID
        or contract.canonical_hash != expected_contract_sha256
    ):
        raise PilotV21110ContinuationError("V2.11.10 contract identity drifted")

    class _Paid:
        pass

    paid = _Paid()
    paid.git_tag = V21110_SCIENCE_TAG
    paid.head_commit = expected_git_commit
    paid.tag_commit = expected_git_commit
    paid.tag_object_type = "tag"
    paid.worktree_clean = True
    raw = repository.joinpath(*V21110_RAW_ROOT.parts)
    expected_path = current_authority_path(raw)
    requested = Path(receipt_path)
    if not requested.is_absolute():
        requested = repository.joinpath(*PurePosixPath(str(receipt_path)).parts)
    if requested.absolute() != expected_path:
        raise PilotV21110ContinuationError("V2.11.10 authority path drifted")
    authority = verify_v21110_current_authority(
        contract=contract,
        repo_root=repository,
        raw_root=raw,
        paid=paid,
    )
    file_hash = _file_sha256(expected_path)
    content_hash = authority["integrity"]["content_sha256"]
    reservations: dict[str, dict[str, Any]] = {}
    for runtime, numeric in authority["reservations"].items():
        reservations[runtime] = {}
        for kind in ("action", "semantic"):
            stable = authority["stable_source_authorities"][runtime][kind]
            current = {
                **_json_copy(stable),
                "pilot_contract_hash": contract.canonical_hash,
                "pilot_tag": V21110_SCIENCE_TAG,
                "source_projection_schema_version": "finevo-pilot-projection-p95-v1",
                "source_projection_file_sha256": file_hash,
                "source_projection_content_sha256": content_hash,
            }
            reservations[runtime][kind] = {
                "authority": current,
                "reservation": _json_copy(numeric[kind]),
            }
    return {
        "receipt_path": _CURRENT_AUTHORITY_PATH.as_posix(),
        "receipt_file_sha256": file_hash,
        "receipt_content_sha256": content_hash,
        "git_commit": expected_git_commit,
        "reservations": reservations,
    }


def runner_reservations_for_v21110(
    contract: Any,
    model_id: str,
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: Any,
) -> dict[str, dict[str, Any]]:
    projection, _ = verified_v21110_projection(
        contract,
        model_id,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
    )
    binding = verified_v21110_observed_p95_authority_binding(
        _CURRENT_AUTHORITY_PATH.as_posix(),
        repo_root=repo_root,
        expected_git_commit=paid.head_commit,
        expected_contract_sha256=contract.canonical_hash,
    )
    runtime = projection["runtime_model"]
    selected = binding["reservations"].get(runtime)
    if not isinstance(selected, Mapping) or set(selected) != {"action", "semantic"}:
        raise PilotV21110ContinuationError(
            "V2.11.10 runner authority denominator drifted"
        )
    for kind in ("action", "semantic"):
        key = f"{projection['served_model']}::{kind}"
        if selected[kind]["reservation"] != projection["projection"][key]:
            raise PilotV21110ContinuationError("V2.11.10 runner/projection drifted")
    serialized = _json_copy(selected)
    receipt_envelope = {
        "source_authority_receipt_path": binding["receipt_path"],
        "source_authority_receipt_file_sha256": binding["receipt_file_sha256"],
        "source_authority_receipt_content_sha256": binding[
            "receipt_content_sha256"
        ],
        "source_release_commit": binding["git_commit"],
    }
    for kind in ("action", "semantic"):
        serialized[kind]["authority"].update(receipt_envelope)
    return {runtime: serialized}


def _release_binding(contract: Any, paid: Any) -> dict[str, Any]:
    implementation = getattr(contract, "implementation", None)
    expected_tag = (
        implementation.get("required_git_tag")
        if isinstance(implementation, Mapping)
        else None
    )
    head = str(getattr(paid, "head_commit", ""))
    if (
        getattr(contract, "contract_id", None) != V21110_CONTRACT_ID
        or expected_tag != V21110_SCIENCE_TAG
        or getattr(paid, "git_tag", None) != V21110_SCIENCE_TAG
        or len(head) != 40
        or any(character not in "0123456789abcdef" for character in head)
        or getattr(paid, "tag_commit", None) != head
        or getattr(paid, "tag_object_type", None) != "tag"
        or getattr(paid, "worktree_clean", None) is not True
    ):
        raise PilotV21110ContinuationError(
            "V2.11.10 parent import requires its clean annotated release"
        )
    return {
        "git_tag": V21110_SCIENCE_TAG,
        "git_commit": head,
        "tag_object_type": "tag",
        "worktree_clean": True,
    }


def _expected_parent_import_denominator() -> dict[str, Any]:
    return {
        "failed_registered_rows": V2119_LEDGER_CELL_COUNT,
        "failed_operational_complete_rows": 1,
        "failed_scientific_failed_rows": V2119_REMAINING_SCIENCE_CELL_COUNT,
        "failed_rows_reclassified_or_redispatched": 0,
        "child_operational_rows": 1,
        "child_scientific_rows": V2119_REMAINING_SCIENCE_CELL_COUNT,
    }


def _expected_parent_terminal_bindings() -> dict[str, Any]:
    return {
        "run_ledger_sha256": V2119_RUN_LEDGER_SHA256,
        "budget_ledger_sha256": V2119_BUDGET_LEDGER_SHA256,
        "stage_receipt_content_sha256_by_stage": {
            "parent-import": V2119_PARENT_STAGE_RECEIPT_CONTENT_SHA256,
            "experiment-d": V2119_EXPERIMENT_D_STAGE_RECEIPT_CONTENT_SHA256,
            "experiment-b": V2119_EXPERIMENT_B_STAGE_RECEIPT_CONTENT_SHA256,
            "cross-model": V2119_CROSS_MODEL_STAGE_RECEIPT_CONTENT_SHA256,
        },
        "scientific_dispatch_acceptance_content_sha256": (
            V2119_ACCEPTANCE_CONTENT_SHA256
        ),
    }


def build_v21110_parent_import_receipt(
    *,
    contract: Any,
    repo_root: str | Path | None = None,
    raw_root: str | Path | None = None,
    failed_repo_root: str | Path,
    authority_repo_root: str | Path,
    paid: Any,
) -> dict[str, Any]:
    """Build the zero-provider V2.11.10 lineage and current authority bundle."""

    require_v21110_provider_keys_absent()
    if (repo_root is None) != (raw_root is None):
        raise PilotV21110ContinuationError(
            "V2.11.10 repo_root and raw_root must be supplied together"
        )
    repository: Path | None = None
    raw: Path | None = None
    source_path: Path | None = None
    source: dict[str, Any] | None = None
    source_replay: dict[str, Any] = {
        "performed": False,
        "recomputed_equal": False,
        "provider_construction": False,
        "provider_calls": 0,
    }
    if repo_root is not None and raw_root is not None:
        repository = _real_root(repo_root, name="V2.11.10 repository")
        raw = _real_root(raw_root, name="V2.11.10 raw root")
        if raw != repository.joinpath(*V21110_RAW_ROOT.parts):
            raise PilotV21110ContinuationError("V2.11.10 raw namespace drifted")
        source = validate_v21110_source_manifest(
            contract=contract,
            repo_root=repository,
            failed_repo_root=failed_repo_root,
            authority_repo_root=authority_repo_root,
        )
        source_path = repository.joinpath(*V21110_SOURCE_MANIFEST_PATH.parts)
        source_replay = {
            "performed": True,
            "recomputed_equal": True,
            "source_root_roles_pairwise_distinct": True,
            "file_sha256": _file_sha256(source_path),
            "content_sha256": source["integrity"]["content_sha256"],
            "provider_construction": False,
            "provider_calls": 0,
        }
    state = verify_v2119_terminal_no_go(
        failed_repo_root=failed_repo_root,
        authority_repo_root=authority_repo_root,
    )
    _verify_v2115_acceptance_with_authority_context(authority_repo_root)
    boundary = _boundary(contract)
    expected_failed = _expected_v2119_failed_release_no_go()
    expected_authority = _expected_v2115_parent_release()
    if (
        _json_copy(boundary.get("failed_release_no_go")) != expected_failed
        or _json_copy(boundary.get("parent_release")) != expected_authority
    ):
        raise PilotV21110ContinuationError("V2.11.10 lineage boundary drifted")
    debit = parent_budget_debit_for_v21110(contract)
    payload: dict[str, Any] = {
        "schema_version": V21110_PARENT_IMPORT_SCHEMA_VERSION,
        "status": "complete",
        "go": True,
        "contract_id": V21110_CONTRACT_ID,
        "contract_sha256": getattr(contract, "canonical_hash", None),
        "release": _release_binding(contract, paid),
        "failed_release_no_go": expected_failed,
        "authority_release": expected_authority,
        "denominator_continuation": _expected_parent_import_denominator(),
        "cumulative_parent_budget_debit": debit.to_dict(),
        "verified_terminal_bindings": {
            "run_ledger_sha256": state["run_snapshot"]["ledger_sha256"],
            "budget_ledger_sha256": state["budget_snapshot"]["ledger_sha256"],
            "stage_receipt_content_sha256_by_stage": {
                stage_id: stage["integrity"]["content_sha256"]
                for stage_id, stage in state["stage_receipts"].items()
            },
            "scientific_dispatch_acceptance_content_sha256": state["acceptance"][
                "integrity"
            ]["content_sha256"],
        },
        "source_manifest_replay": source_replay,
        "import_policy": _json_copy(_PARENT_IMPORT_POLICY),
        "scientific_evidence": False,
        "claim_boundary": _PARENT_IMPORT_CLAIM_BOUNDARY,
    }
    if repository is not None and raw is not None:
        assert source_path is not None and source is not None
        authority_state = state["authority"]
        parent_import = authority_state["parent_import_receipt"]
        calibration = parent_import.get("calibration_wrapper")
        capabilities = parent_import.get("capability_wrappers")
        if (
            not isinstance(calibration, Mapping)
            or calibration.get("integrity", {}).get("content_sha256")
            != v2117.V2115_CALIBRATION_CONTENT_SHA256
            or not isinstance(capabilities, Mapping)
            or set(capabilities) != {"gpt52_main", "gpt56_diagnostic"}
        ):
            raise PilotV21110ContinuationError(
                "V2.11.5 calibration/capability authority drifted"
            )
        capability_summaries = {
            model_id: _capability_summary(capabilities[model_id], model_id=model_id)
            for model_id in sorted(capabilities)
        }
        dispatch = _dispatch_authority_source(authority_state)
        mapping = _canonical_remaining_cell_mapping(
            contract, authority_state["contract"]
        )
        payload.update(
            {
                "source_manifest": {
                    "path": V21110_SOURCE_MANIFEST_PATH.as_posix(),
                    "file_sha256": _file_sha256(source_path),
                    "content_sha256": source["integrity"]["content_sha256"],
                },
                "failed_terminal_no_go": {
                    "registered_rows": V2119_LEDGER_CELL_COUNT,
                    "operational_complete_rows": 1,
                    "scientific_failed_rows": V2119_REMAINING_SCIENCE_CELL_COUNT,
                    "provider_calls": 0,
                    "provider_construction": False,
                    "hosted_completions": 0,
                    "hosted_cost_usd": 0.0,
                    "current_storage_bytes": V2119_CURRENT_ACTUAL_STORAGE_BYTES,
                    "scientific_acceptance_present": True,
                    "science_budget_reservation_count": 36,
                },
                "authority_import": {
                    "path": (
                        V2115_RAW_ROOT / "parent-import/parent_import_receipt.json"
                    ).as_posix(),
                    "file_sha256": v2117.V2115_PARENT_IMPORT_FILE_SHA256,
                    "content_sha256": v2117.V2115_PARENT_IMPORT_CONTENT_SHA256,
                },
                "calibration_wrapper": _json_copy(calibration),
                "capability_authority": capability_summaries,
                "dispatch_authority_source": dispatch,
                "canonical_remaining_cell_mapping": mapping,
            }
        )
    receipt = _seal(payload)
    if repository is not None and raw is not None:
        _write_once(raw / "parent-import/parent_import_receipt.json", receipt)
        _build_current_authority(
            contract=contract,
            raw_root=raw,
            paid=paid,
            authority_state=state["authority"],
            parent_import_content_sha256=receipt["integrity"]["content_sha256"],
        )
    return receipt


def verify_v21110_parent_import_receipt(
    receipt: Mapping[str, Any] | str | Path,
    *,
    contract: Any,
    repo_root: str | Path | None = None,
    raw_root: str | Path | None = None,
    paid: Any | None = None,
) -> dict[str, Any]:
    if isinstance(receipt, Mapping):
        value = _json_copy(receipt)
    else:
        if repo_root is None or raw_root is None or paid is None:
            raise PilotV21110ContinuationError(
                "persisted V2.11.10 receipt requires release provenance"
            )
        repository = _real_root(repo_root, name="V2.11.10 repository")
        raw = _real_root(raw_root, name="V2.11.10 raw root")
        path = Path(receipt).absolute()
        if (
            raw != repository.joinpath(*V21110_RAW_ROOT.parts)
            or path != raw / "parent-import/parent_import_receipt.json"
        ):
            raise PilotV21110ContinuationError("parent-import receipt path drifted")
        value = _strict_json(path, name="V2.11.10 parent-import receipt")
    _verify_seal(value, name="V2.11.10 parent-import receipt")
    expected_debit = parent_budget_debit_for_v21110(contract).to_dict()
    replay = value.get("source_manifest_replay")
    expected_fields = (
        _PARENT_IMPORT_PERSISTED_FIELDS
        if repo_root is not None
        else _PARENT_IMPORT_BASE_FIELDS
    )
    if (
        set(value) != expected_fields
        or value.get("schema_version") != V21110_PARENT_IMPORT_SCHEMA_VERSION
        or value.get("status") != "complete"
        or value.get("go") is not True
        or value.get("contract_id") != V21110_CONTRACT_ID
        or value.get("contract_sha256") != getattr(contract, "canonical_hash", None)
        or (
            paid is not None
            and canonical_sha256(value.get("release"))
            != canonical_sha256(_release_binding(contract, paid))
        )
        or canonical_sha256(value.get("failed_release_no_go"))
        != canonical_sha256(_expected_v2119_failed_release_no_go())
        or canonical_sha256(value.get("authority_release"))
        != canonical_sha256(_expected_v2115_parent_release())
        or canonical_sha256(value.get("cumulative_parent_budget_debit"))
        != canonical_sha256(expected_debit)
        or canonical_sha256(value.get("denominator_continuation"))
        != canonical_sha256(_expected_parent_import_denominator())
        or canonical_sha256(value.get("verified_terminal_bindings"))
        != canonical_sha256(_expected_parent_terminal_bindings())
        or canonical_sha256(value.get("import_policy"))
        != canonical_sha256(_PARENT_IMPORT_POLICY)
        or not isinstance(replay, Mapping)
        or replay.get("provider_construction") is not False
        or type(replay.get("provider_calls")) is not int
        or replay.get("provider_calls") != 0
        or value.get("scientific_evidence") is not False
        or value.get("claim_boundary") != _PARENT_IMPORT_CLAIM_BOUNDARY
    ):
        raise PilotV21110ContinuationError("V2.11.10 parent-import receipt drifted")
    if repo_root is not None:
        source_path, source = _tracked_source_manifest(contract, repo_root=repo_root)
        mapping = value.get("canonical_remaining_cell_mapping")
        dispatch = value.get("dispatch_authority_source")
        if (
            value.get("source_manifest")
            != {
                "path": V21110_SOURCE_MANIFEST_PATH.as_posix(),
                "file_sha256": _file_sha256(source_path),
                "content_sha256": source["integrity"]["content_sha256"],
            }
            or canonical_sha256(value.get("failed_terminal_no_go"))
            != canonical_sha256(
                {
                    "registered_rows": V2119_LEDGER_CELL_COUNT,
                    "operational_complete_rows": 1,
                    "scientific_failed_rows": V2119_REMAINING_SCIENCE_CELL_COUNT,
                    "provider_calls": 0,
                    "provider_construction": False,
                    "hosted_completions": 0,
                    "hosted_cost_usd": 0.0,
                    "current_storage_bytes": V2119_CURRENT_ACTUAL_STORAGE_BYTES,
                    "scientific_acceptance_present": True,
                    "science_budget_reservation_count": 36,
                }
            )
            or canonical_sha256(value.get("authority_import"))
            != canonical_sha256(
                {
                    "path": (
                        V2115_RAW_ROOT / "parent-import/parent_import_receipt.json"
                    ).as_posix(),
                    "file_sha256": v2117.V2115_PARENT_IMPORT_FILE_SHA256,
                    "content_sha256": v2117.V2115_PARENT_IMPORT_CONTENT_SHA256,
                }
            )
            or replay
            != {
                "performed": True,
                "recomputed_equal": True,
                "source_root_roles_pairwise_distinct": True,
                "file_sha256": _file_sha256(source_path),
                "content_sha256": source["integrity"]["content_sha256"],
                "provider_construction": False,
                "provider_calls": 0,
            }
            or not isinstance(mapping, Mapping)
            or mapping.get("row_count") != V2119_REMAINING_SCIENCE_CELL_COUNT
            or mapping.get("mapping_sha256")
            != _boundary(contract)["continuation_matrix"][
                "canonical_86_row_mapping_sha256"
            ]
            or canonical_sha256(mapping.get("rows")) != mapping.get("mapping_sha256")
            or not isinstance(dispatch, Mapping)
            or dispatch.get("reservation_set_sha256")
            != v2117.V2115_RESERVATION_SET_SHA256
            or dispatch.get("stable_authority_set_sha256")
            != v2117.V2115_STABLE_AUTHORITY_SET_SHA256
        ):
            raise PilotV21110ContinuationError(
                "V2.11.10 persisted parent-import authority drifted"
            )
    elif (
        replay.get("performed") is not False
        or replay.get("recomputed_equal") is not False
    ):
        raise PilotV21110ContinuationError(
            "unpersisted V2.11.10 parent receipt claims source replay"
        )
    return value


def _ledger_prefix(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return v2119._ledger_prefix(snapshot)
    except v2119.PilotV2119ContinuationError as exc:
        raise PilotV21110ContinuationError(str(exc)) from exc


def _expected_acceptance_denominator(contract: Any) -> dict[str, Any]:
    return {
        "ledger_cells": 87,
        "operational_import_cells": 1,
        "fresh_scientific_cells": 86,
        "status_counts": {"complete": 1, "scheduled": 86},
        "stage_cell_counts": {
            stage_id: len(contract.expand(stage=stage_id))
            for stage_id in (
                "parent-import",
                "experiment-d",
                "experiment-b",
                "cross-model",
            )
        },
        "a_c_child_cells": 0,
        "source_parent_terminal_rows": 50,
        "failed_v2119_terminal_rows": 87,
    }


def _verified_parent_import_budget_actual(contract: Any, row: Any) -> Mapping[str, Any]:
    from . import pilot_orchestrator as orch

    specs = tuple(contract.expand(stage="parent-import"))
    if len(specs) != 1:
        raise PilotV21110ContinuationError("parent-import budget denominator drifted")
    expected = orch._v21110_parent_import_projection(specs[0]).to_dict()
    actual = row.get("actual") if isinstance(row, Mapping) else None
    if (
        not isinstance(row, Mapping)
        or row.get("stage_bucket") != expected["stage_bucket"]
        or canonical_sha256(row.get("reservation")) != canonical_sha256(expected)
        or row.get("status") != "complete"
        or not isinstance(actual, Mapping)
        or type(actual.get("cost_usd")) is not float
        or actual.get("cost_usd") != 0.0
        or type(actual.get("completions")) is not int
        or actual.get("completions") != 0
        or type(actual.get("storage_bytes")) is not int
        or actual["storage_bytes"] < 1
        or actual["storage_bytes"] > expected["storage_bytes"]
    ):
        raise PilotV21110ContinuationError(
            "parent-import budget row differs from its zero-provider projection"
        )
    return actual


def _audit_acceptance_denominator(
    contract: Any,
    run_snapshot: Mapping[str, Any],
    budget_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    specs = tuple(contract.expand())
    rows = run_snapshot.get("runs")
    if (
        len(specs) != V2119_LEDGER_CELL_COUNT
        or not isinstance(rows, Mapping)
        or set(rows) != {spec.run_id for spec in specs}
        or any(
            canonical_sha256(rows[spec.run_id].get("spec"))
            != canonical_sha256(spec.to_dict())
            for spec in specs
        )
    ):
        raise PilotV21110ContinuationError("V2.11.10 ITT denominator drifted")
    parent = tuple(contract.expand(stage="parent-import"))
    science = tuple(
        spec
        for stage_id in ("experiment-d", "experiment-b", "cross-model")
        for spec in contract.expand(stage=stage_id)
    )
    if (
        len(parent) != 1
        or len(science) != V2119_REMAINING_SCIENCE_CELL_COUNT
        or rows[parent[0].run_id].get("status") != "complete"
        or Counter(rows[spec.run_id].get("status") for spec in specs)
        != Counter({"scheduled": 86, "complete": 1})
    ):
        raise PilotV21110ContinuationError(
            "acceptance must precede the first V2.11.10 science cell"
        )
    budget_rows = budget_snapshot.get("runs")
    if not isinstance(budget_rows, Mapping) or set(budget_rows) != {parent[0].run_id}:
        raise PilotV21110ContinuationError(
            "acceptance budget prefix contains a science reservation"
        )
    _verified_parent_import_budget_actual(contract, budget_rows[parent[0].run_id])
    return _expected_acceptance_denominator(contract)


def _acceptance_projections(
    contract: Any,
    *,
    repo_root: Path,
    raw_root: Path,
    paid: Any,
) -> tuple[Any, ...]:
    from . import pilot_orchestrator as orch

    projections: list[Any] = []
    d_specs = tuple(contract.expand(stage="experiment-d"))
    for model_id, seed in sorted(
        {(spec.model_id, spec.environment_seed) for spec in d_specs}
    ):
        group = tuple(
            spec
            for spec in d_specs
            if spec.model_id == model_id and spec.environment_seed == seed
        )
        matched = [spec for spec in group if spec.arm_id == "matched-a"]
        if len(group) != 11 or len(matched) != 1:
            raise PilotV21110ContinuationError("Experiment D group drifted")
        projections.append(
            orch._d_group_projection(
                contract,
                matched[0],
                raw_root=raw_root,
                paid=paid,
                authority_repo_root=repo_root,
            )
        )
    for stage_id in ("experiment-b", "cross-model"):
        for spec in contract.expand(stage=stage_id):
            projections.append(
                orch.projection_from_preflight(
                    contract,
                    spec,
                    raw_root=raw_root,
                    paid=paid,
                    authority_repo_root=repo_root,
                )
            )
    if len(projections) != 36:
        raise PilotV21110ContinuationError("V2.11.10 projection-unit denominator drifted")
    return tuple(projections)


def _validated_runner_config_material(
    contract: Any,
    *,
    repo_root: Path,
    raw_root: Path,
    paid: Any,
) -> dict[str, Any]:
    """Exercise the exact producer-to-runner authority path for all 86 cells."""

    from . import pilot_orchestrator as orch
    from .pilot_checkpoint import config_from_dict
    from .runner import (
        has_sealed_observed_p95_authority,
        observed_p95_authority_repo_context,
        serialized_has_sealed_observed_p95_authority,
        validate_preflight_p95_reservations,
    )

    science_specs = tuple(
        spec
        for stage_id in ("experiment-d", "experiment-b", "cross-model")
        for spec in contract.expand(stage=stage_id)
    )
    if len(science_specs) != V2119_REMAINING_SCIENCE_CELL_COUNT:
        raise PilotV21110ContinuationError(
            "V2.11.10 runner-config cell denominator drifted"
        )
    model_ids = tuple(sorted({spec.model_id for spec in science_specs}))
    if model_ids != ("gpt52_main", "gpt56_diagnostic"):
        raise PilotV21110ContinuationError(
            "V2.11.10 runner-config model denominator drifted"
        )

    runner_by_model: dict[str, Mapping[str, Any]] = {}
    authority_shape: dict[str, Any] = {}
    for model_id in model_ids:
        binding = verified_v21110_observed_p95_authority_binding(
            _CURRENT_AUTHORITY_PATH.as_posix(),
            repo_root=repo_root,
            expected_git_commit=paid.head_commit,
            expected_contract_sha256=contract.canonical_hash,
        )
        runner = runner_reservations_for_v21110(
            contract,
            model_id,
            repo_root=repo_root,
            raw_root=raw_root,
            paid=paid,
        )
        runtime = orch._runtime_model_for_profile(contract.provider_profiles[model_id])
        producer = binding.get("reservations", {}).get(runtime)
        selected = runner.get(runtime)
        if (
            not isinstance(producer, Mapping)
            or not isinstance(selected, Mapping)
            or set(producer) != {"action", "semantic"}
            or set(selected) != {"action", "semantic"}
        ):
            raise PilotV21110ContinuationError(
                f"V2.11.10 {model_id} runner authority denominator drifted"
            )
        for call_kind in ("action", "semantic"):
            producer_authority = producer[call_kind].get("authority")
            runner_authority = selected[call_kind].get("authority")
            if (
                not isinstance(producer_authority, Mapping)
                or not isinstance(runner_authority, Mapping)
                or len(producer_authority) != 13
                or len(runner_authority) != 17
                or {
                    key: value
                    for key, value in runner_authority.items()
                    if key
                    not in {
                        "source_authority_receipt_path",
                        "source_authority_receipt_file_sha256",
                        "source_authority_receipt_content_sha256",
                        "source_release_commit",
                    }
                }
                != producer_authority
            ):
                raise PilotV21110ContinuationError(
                    f"V2.11.10 {model_id}/{call_kind} authority layering drifted"
                )
        runner_by_model[model_id] = runner
        authority_shape[model_id] = {
            "runtime_model": runtime,
            "producer_core_field_count": 13,
            "runner_envelope_field_count": 17,
            "receipt_envelope_field_count": 4,
        }

    configs: dict[str, str] = {}
    validated_call_kinds: dict[str, list[str]] = {}
    execution_units: dict[str, list[str]] = {}
    with observed_p95_authority_repo_context(repo_root):
        for spec in science_specs:
            runtime = orch._runtime_model_for_profile(
                contract.provider_profiles[spec.model_id]
            )
            config = orch.config_for_spec(
                contract,
                spec,
                raw_root=raw_root,
                paid_provenance=paid,
                authority_repo_root=repo_root,
                verify_bound_inputs=True,
                preflight_p95_reservations=runner_by_model[spec.model_id],
            )
            payload = config.to_dict()
            restored = config_from_dict(payload)
            validated = validate_preflight_p95_reservations(
                restored,
                provider_model_name=runtime,
            )
            if (
                restored.to_dict() != payload
                or not validated
                or has_sealed_observed_p95_authority(config) is not True
                or has_sealed_observed_p95_authority(restored) is not True
                or serialized_has_sealed_observed_p95_authority(
                    payload,
                    authority_repo_root=repo_root,
                )
                is not True
            ):
                raise PilotV21110ContinuationError(
                    f"V2.11.10 runner config round-trip failed for {spec.run_id}"
                )
            configs[spec.run_id] = canonical_sha256(payload)
            validated_call_kinds[spec.run_id] = sorted(validated)
            execution_unit = (
                f"{contract.contract_id}--experiment-d--{spec.model_id}--"
                f"checkpoint-group--s{spec.environment_seed}"
                if spec.stage_id == "experiment-d"
                else spec.run_id
            )
            execution_units.setdefault(execution_unit, []).append(spec.run_id)

    if (
        len(configs) != V2119_REMAINING_SCIENCE_CELL_COUNT
        or len(execution_units) != 36
        or sum(len(run_ids) for run_ids in execution_units.values()) != 86
        or len([run_ids for run_ids in execution_units.values() if len(run_ids) == 11])
        != 5
    ):
        raise PilotV21110ContinuationError(
            "V2.11.10 runner config execution-unit coverage drifted"
        )
    normalized_units = {
        unit_id: sorted(run_ids) for unit_id, run_ids in sorted(execution_units.items())
    }
    return {
        "cell_count": len(configs),
        "execution_unit_count": len(normalized_units),
        "model_count": len(model_ids),
        "config_sha256_by_run_id": dict(sorted(configs.items())),
        "config_set_sha256": canonical_sha256(configs),
        "validated_call_kinds_by_run_id": dict(sorted(validated_call_kinds.items())),
        "execution_unit_run_ids": normalized_units,
        "execution_unit_set_sha256": canonical_sha256(normalized_units),
        "authority_layering_by_model": authority_shape,
        "serialize_restore_validated": True,
        "source_backed_validation_performed": True,
        "provider_construction": False,
        "provider_calls": 0,
    }


def _acceptance_material(
    contract: Any,
    *,
    repo_root: Path,
    raw_root: Path,
    paid: Any,
    budget_ledger: PilotBudgetLedger,
) -> dict[str, Any]:
    from . import pilot_orchestrator as orch

    parent = verify_v21110_parent_import_receipt(
        raw_root / "parent-import/parent_import_receipt.json",
        contract=contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
    )
    authority = verify_v21110_current_authority(
        contract=contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
    )
    projections = _acceptance_projections(
        contract, repo_root=repo_root, raw_root=raw_root, paid=paid
    )
    try:
        orch._assert_projection_matrix_fits(budget_ledger, projections)
    except Exception as exc:
        raise PilotV21110ContinuationError(
            f"complete V2.11.10 continuation exceeds a hard cap: {exc}"
        ) from exc
    calls = {"experiment-d": 0, "experiment-b": 0, "cross-model": 0}
    costs = {"experiment-d": 0.0, "experiment-b": 0.0, "cross-model": 0.0}
    storage = {"experiment-d": 0, "experiment-b": 0, "cross-model": 0}
    projection_rows: list[dict[str, Any]] = []
    for projection in projections:
        stage = next(stage for stage in calls if f"--{stage}--" in projection.run_id)
        calls[stage] += int(projection.completions)
        costs[stage] += float(projection.cost_usd)
        storage[stage] += int(projection.storage_bytes)
        projection_rows.append(projection.to_dict())
    budget_boundary = _boundary(contract)["continuation_budget"]
    total_cost = sum(costs.values())
    total_calls = sum(calls.values())
    total_storage = sum(storage.values())
    if (
        calls != {"experiment-d": 1480, "experiment-b": 1440, "cross-model": 336}
        or total_calls != budget_boundary["fresh_registered_provider_calls"]
        or total_storage != budget_boundary["fresh_storage_reservation_bytes"]
        or not math.isclose(
            total_cost,
            float(budget_boundary["fresh_projected_cost_usd"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise PilotV21110ContinuationError(
            "V2.11.10 full projection differs from preregistration"
        )
    runner_configs = _validated_runner_config_material(
        contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
    )
    parent_specs = tuple(contract.expand(stage="parent-import"))
    budget_rows = budget_ledger.snapshot().get("runs")
    parent_row = (
        budget_rows.get(parent_specs[0].run_id)
        if isinstance(budget_rows, Mapping) and len(parent_specs) == 1
        else None
    )
    actual = _verified_parent_import_budget_actual(contract, parent_row)
    rows_sorted = sorted(projection_rows, key=lambda item: item["run_id"])
    return {
        "parent_import": {
            "path": (
                V21110_RAW_ROOT / "parent-import/parent_import_receipt.json"
            ).as_posix(),
            "file_sha256": _file_sha256(
                raw_root / "parent-import/parent_import_receipt.json"
            ),
            "content_sha256": parent["integrity"]["content_sha256"],
        },
        "current_authority": {
            "path": _CURRENT_AUTHORITY_PATH.as_posix(),
            "file_sha256": _file_sha256(current_authority_path(raw_root)),
            "content_sha256": authority["integrity"]["content_sha256"],
        },
        "runner_configs": runner_configs,
        "budget_projection": {
            "projection_unit_count": len(projections),
            "fresh_provider_calls": total_calls,
            "fresh_calls_by_stage": calls,
            "fresh_projected_cost_usd": total_cost,
            "fresh_projected_cost_usd_by_stage": costs,
            "fresh_storage_bytes": total_storage,
            "fresh_storage_bytes_by_stage": storage,
            "projection_sha256_by_run_id": {
                row["run_id"]: canonical_sha256(row) for row in rows_sorted
            },
            "projection_set_sha256": canonical_sha256(rows_sorted),
            "projected_cumulative_cost_usd": budget_boundary[
                "projected_cumulative_cost_usd"
            ],
            "projected_cumulative_hosted_completions": budget_boundary[
                "projected_cumulative_hosted_completions"
            ],
            "projected_cumulative_storage_bytes": budget_boundary[
                "projected_cumulative_storage_bytes"
            ],
            "operational_import_storage_bytes": actual["storage_bytes"],
            "ledger_projected_cumulative_storage_bytes": (
                budget_boundary["projected_cumulative_storage_bytes"]
                + actual["storage_bytes"]
            ),
            "hard_caps": orch._budget_caps(contract).to_dict(),
            "full_matrix_fits": True,
        },
    }


def _acceptance_receipt(
    contract: Any,
    *,
    repo_root: Path,
    raw_root: Path,
    paid: Any,
    run_ledger: Any,
    budget_ledger: PilotBudgetLedger,
) -> dict[str, Any]:
    run_snapshot = run_ledger.snapshot()
    budget_snapshot = budget_ledger.snapshot()
    denominator = _audit_acceptance_denominator(contract, run_snapshot, budget_snapshot)
    material = _acceptance_material(
        contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
        budget_ledger=budget_ledger,
    )
    return _seal(
        {
            "schema_version": V21110_ACCEPTANCE_SCHEMA_VERSION,
            "status": "go",
            "go": True,
            "contract_id": V21110_CONTRACT_ID,
            "contract_sha256": contract.canonical_hash,
            "release": _release_binding(contract, paid),
            "raw_namespace": V21110_RAW_ROOT.as_posix(),
            "denominator": denominator,
            **material,
            "ledger_prefixes": {
                "run_ledger": _ledger_prefix(run_snapshot),
                "budget_ledger": _ledger_prefix(budget_snapshot),
            },
            "provider_boundary": _ACCEPTANCE_PROVIDER_BOUNDARY,
            "scientific_evidence": False,
            "claim_boundary": _ACCEPTANCE_CLAIM_BOUNDARY,
        }
    )


def _verify_acceptance_prefix(
    snapshot: Mapping[str, Any],
    *,
    prefix: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_path: str,
    budget: bool,
) -> bool:
    events = snapshot.get("events")
    runs = snapshot.get("runs")
    count = prefix.get("event_count")
    if (
        set(prefix)
        != {"event_count", "event_chain_head", "ledger_sha256", "runs_sha256"}
        or not isinstance(events, list)
        or not isinstance(runs, Mapping)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or len(events) < count
        or events[count - 1].get("event_sha256") != prefix.get("event_chain_head")
    ):
        raise PilotV21110ContinuationError("accepted ledger prefix drifted")
    if len(events) == count:
        if snapshot.get("ledger_sha256") != prefix.get(
            "ledger_sha256"
        ) or canonical_sha256(runs) != prefix.get("runs_sha256"):
            raise PilotV21110ContinuationError(
                "unmarked acceptance ledger differs from sealed prefix"
            )
        return False
    marker = events[count]
    expected = {
        "receipt_schema_version": V21110_ACCEPTANCE_SCHEMA_VERSION,
        "receipt_path": receipt_path,
        "receipt_content_sha256": receipt["integrity"]["content_sha256"],
        "accepted_run_event_count": receipt["ledger_prefixes"]["run_ledger"][
            "event_count"
        ],
        "accepted_run_event_chain_head": receipt["ledger_prefixes"]["run_ledger"][
            "event_chain_head"
        ],
        "accepted_budget_event_count": receipt["ledger_prefixes"]["budget_ledger"][
            "event_count"
        ],
        "accepted_budget_event_chain_head": receipt["ledger_prefixes"]["budget_ledger"][
            "event_chain_head"
        ],
        ("budget_runs_sha256" if budget else "runs_sha256"): prefix["runs_sha256"],
    }
    if (
        marker.get("event_type") != "acceptance_receipt_bound"
        or marker.get("payload") != expected
    ):
        raise PilotV21110ContinuationError("acceptance ledger marker drifted")
    return True


def _verify_current_accepted_budget_rows(
    contract: Any,
    receipt: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    run_snapshot: Mapping[str, Any],
) -> None:
    rows = snapshot.get("runs")
    events = snapshot.get("events")
    run_rows = run_snapshot.get("runs")
    accepted = receipt["budget_projection"]["projection_sha256_by_run_id"]
    parent = tuple(contract.expand(stage="parent-import"))
    if (
        not isinstance(rows, Mapping)
        or not isinstance(events, list)
        or not isinstance(run_rows, Mapping)
        or len(parent) != 1
    ):
        raise PilotV21110ContinuationError("accepted budget rows are malformed")
    reservations_by_id: dict[str, list[Mapping[str, Any]]] = {}
    finalizations_by_id: dict[str, list[Mapping[str, Any]]] = {}
    budget_to_run_ids: dict[str, tuple[str, ...]] = {
        spec.run_id: (spec.run_id,)
        for stage_id in ("experiment-b", "cross-model")
        for spec in contract.expand(stage=stage_id)
    }
    d_specs = tuple(contract.expand(stage="experiment-d"))
    for model_id, seed in sorted(
        {(spec.model_id, spec.environment_seed) for spec in d_specs}
    ):
        budget_to_run_ids[
            (
                f"{contract.contract_id}--experiment-d--{model_id}--"
                f"checkpoint-group--s{seed}"
            )
        ] = tuple(
            spec.run_id
            for spec in d_specs
            if spec.model_id == model_id and spec.environment_seed == seed
        )
    for event in events:
        if not isinstance(event, Mapping):
            raise PilotV21110ContinuationError("accepted budget event is malformed")
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            raise PilotV21110ContinuationError("accepted budget payload is malformed")
        if event.get("event_type") == "run_reserved":
            reservations_by_id.setdefault(str(payload.get("run_id")), []).append(
                payload
            )
        elif event.get("event_type") == "run_finalized":
            finalizations_by_id.setdefault(str(payload.get("run_id")), []).append(
                payload
            )
    for run_id, row in rows.items():
        if run_id == parent[0].run_id:
            _verified_parent_import_budget_actual(contract, row)
            continue
        reservation = row.get("reservation") if isinstance(row, Mapping) else None
        actual = row.get("actual") if isinstance(row, Mapping) else None
        status = row.get("status") if isinstance(row, Mapping) else None
        failure = row.get("failure") if isinstance(row, Mapping) else None
        reserved_events = reservations_by_id.get(str(run_id), [])
        finalized_events = finalizations_by_id.get(str(run_id), [])
        if (
            run_id not in accepted
            or not isinstance(reservation, Mapping)
            or canonical_sha256(reservation) != accepted[run_id]
            or reservation.get("run_id") != run_id
            or row.get("stage_bucket") != reservation.get("stage_bucket")
            or len(reserved_events) != 1
            or set(reserved_events[0]) != {"run_id", "projection_sha256"}
            or reserved_events[0].get("projection_sha256")
            != canonical_sha256(reservation)
        ):
            raise PilotV21110ContinuationError(
                f"unaccepted science reservation appeared: {run_id}"
            )
        linked_run_ids = budget_to_run_ids.get(str(run_id))
        if not linked_run_ids or any(
            linked_run_id not in run_rows for linked_run_id in linked_run_ids
        ):
            raise PilotV21110ContinuationError(
                f"science budget row has no exact ITT mapping: {run_id}"
            )
        linked_row_items = tuple(
            (linked_run_id, run_rows[linked_run_id]) for linked_run_id in linked_run_ids
        )
        linked_rows = tuple(linked_row for _, linked_row in linked_row_items)
        if any(not isinstance(linked_row, Mapping) for linked_row in linked_rows):
            raise PilotV21110ContinuationError(
                f"science budget/ITT rows are malformed: {run_id}"
            )
        linked_statuses = {linked_row.get("status") for linked_row in linked_rows}
        linked_terminal = {
            "complete",
            "failed",
            "capability-no-go",
            "budget-stopped",
            "integrity-stopped",
        }
        is_d_group = str(run_id).startswith(f"{contract.contract_id}--experiment-d--")

        def has_bound_terminal_event(
            linked_run_id: str,
            linked_row: Mapping[str, Any],
        ) -> bool:
            run_events = run_snapshot.get("events")
            if not isinstance(run_events, list):
                return False
            terminal_state = {
                "status": linked_row.get("status"),
                "artifact": linked_row.get("artifact"),
                "failure": linked_row.get("failure"),
            }
            if "artifact_binding" in linked_row:
                terminal_state["artifact_binding"] = linked_row.get("artifact_binding")
            matches = tuple(
                event
                for event in run_events
                if isinstance(event, Mapping)
                and event.get("event_type") == "run_finalized"
                and isinstance(event.get("payload"), Mapping)
                and event["payload"].get("run_id") == linked_run_id
            )
            return bool(
                len(matches) == 1
                and matches[0]["payload"].get("terminal_state_sha256")
                == canonical_sha256(terminal_state)
            )

        if status == "reserved":
            # A process may die after atomically reserving a shared D group
            # but while terminal ITT dispositions are being persisted.  Keep
            # that exact accepted reservation admissible long enough for the
            # pre-dispatch recovery boundary to conservatively charge it and
            # stop every remaining linked row.  A finalized budget row never
            # receives this exception.
            reserved_statuses_valid = bool(
                linked_statuses
                and linked_statuses <= ({"scheduled"} | linked_terminal)
                and (
                    linked_statuses == {"scheduled"}
                    or is_d_group
                    or (
                        len(linked_row_items) == 1
                        and linked_rows[0].get("status") in linked_terminal
                        and has_bound_terminal_event(
                            linked_row_items[0][0], linked_rows[0]
                        )
                    )
                )
            )
            if not reserved_statuses_valid:
                raise PilotV21110ContinuationError(
                    f"reserved science budget/ITT terminality is invalid: {run_id}"
                )
            if (
                set(row)
                != {
                    "stage_bucket",
                    "reservation",
                    "actual",
                    "status",
                    "reserved_at",
                    "finalized_at",
                }
                or actual is not None
                or not isinstance(row.get("reserved_at"), str)
                or not row["reserved_at"]
                or row.get("finalized_at") is not None
                or finalized_events
            ):
                raise PilotV21110ContinuationError(
                    f"reserved science budget row drifted: {run_id}"
                )
            continue
        if status not in {"complete", "failed", "budget-stopped", "integrity-stopped"}:
            raise PilotV21110ContinuationError(
                f"science budget row has an invalid status: {run_id}"
            )
        if (
            set(row)
            != {
                "stage_bucket",
                "reservation",
                "actual",
                "status",
                "reserved_at",
                "finalized_at",
                "failure",
            }
            or not isinstance(actual, Mapping)
            or set(actual) != {"cost_usd", "completions", "storage_bytes"}
            or type(actual.get("cost_usd")) is not float
            or not math.isfinite(actual["cost_usd"])
            or actual["cost_usd"] < 0.0
            or type(actual.get("completions")) is not int
            or actual["completions"] < 0
            or type(actual.get("storage_bytes")) is not int
            or actual["storage_bytes"] < 0
            or (failure is not None and not isinstance(failure, Mapping))
            or (status == "complete" and failure is not None)
            or (
                status in {"failed", "budget-stopped", "integrity-stopped"}
                and not isinstance(failure, Mapping)
            )
            or not isinstance(row.get("reserved_at"), str)
            or not row["reserved_at"]
            or not isinstance(row.get("finalized_at"), str)
            or not row["finalized_at"]
            or len(finalized_events) != 1
            or set(finalized_events[0])
            != {"run_id", "status", "actual_sha256", "failure_sha256"}
            or finalized_events[0].get("status") != status
            or finalized_events[0].get("actual_sha256") != canonical_sha256(actual)
            or finalized_events[0].get("failure_sha256")
            != (None if failure is None else canonical_sha256(failure))
        ):
            raise PilotV21110ContinuationError(
                f"terminal science budget row/event drifted: {run_id}"
            )
        if status != "integrity-stopped" and (
            actual["cost_usd"] > float(reservation["cost_usd"]) + 1e-12
            or actual["completions"] > int(reservation["completions"])
            or actual["storage_bytes"] > int(reservation["storage_bytes"])
        ):
            raise PilotV21110ContinuationError(
                f"terminal science budget row exceeds reservation: {run_id}"
            )
        # A process can crash after the accepted budget unit is durably
        # finalized but before its linked ITT disposition is persisted.  The
        # next resume must first validate the complete reservation/actual/
        # failure/event record above, then admit only the exact shapes that the
        # pre-dispatch recovery boundary can conservatively close without a
        # second budget finalization.  B/cross own one ITT row, so that row may
        # still be scheduled.  D owns a group and may expose either the first
        # scheduled-plus-original-terminal window or the terminal state after
        # recovery.  The latter is admitted only when every original row is
        # still bound to its immutable run-finalization event and every added
        # row has the exact recovery fingerprint.
        d_identity = None
        if is_d_group:
            for model_id, seed in sorted(
                {(spec.model_id, spec.environment_seed) for spec in d_specs}
            ):
                expected_group_id = (
                    f"{contract.contract_id}--experiment-d--{model_id}--"
                    f"checkpoint-group--s{seed}"
                )
                if expected_group_id == run_id:
                    d_identity = (str(model_id), int(seed))
                    break
            if d_identity is None:
                raise PilotV21110ContinuationError(
                    f"finalized Experiment D budget identity is unknown: {run_id}"
                )

        def is_original_terminal(linked_row: Mapping[str, Any]) -> bool:
            return bool(
                linked_row.get("status") == status
                and linked_row.get("failure") == failure
            )

        def is_exact_d_recovery(linked_row: Mapping[str, Any]) -> bool:
            if d_identity is None:
                return False
            model_id, seed = d_identity
            expected_failure = {
                "error_type": "BudgetFinalizedBeforeITT",
                "message": (
                    "a prior process created shared Experiment D budget state "
                    "without an exact terminal ITT group; no redispatch is permitted"
                ),
                "model_id": model_id,
                "environment_seed": seed,
                "provider_dispatch_started": False,
                "stop_origin": "pre-catalog-interrupted-reservation-recovery",
            }
            return bool(
                linked_row.get("status") == "integrity-stopped"
                and linked_row.get("artifact") is None
                and linked_row.get("failure") == expected_failure
                and (
                    "artifact_binding" not in linked_row
                    or linked_row.get("artifact_binding") is None
                )
            )

        def interrupted_d_failure() -> dict[str, Any] | None:
            if d_identity is None:
                return None
            model_id, seed = d_identity
            return {
                "error_type": "InterruptedReservation",
                "message": (
                    "a prior process created shared Experiment D budget state "
                    "without an exact terminal ITT group; no redispatch is permitted"
                ),
                "model_id": model_id,
                "environment_seed": seed,
                "provider_dispatch_started": False,
                "stop_origin": "pre-catalog-interrupted-reservation-recovery",
                "accounting_basis": "unreconciled-conservative-reservation",
            }

        expected_interrupted_d_failure = interrupted_d_failure()
        exact_interrupted_d_budget = bool(
            is_d_group
            and status == "integrity-stopped"
            and failure == expected_interrupted_d_failure
            and actual["cost_usd"] == float(reservation["cost_usd"])
            and actual["completions"] == int(reservation["completions"])
            and actual["storage_bytes"] == int(reservation["storage_bytes"])
        )

        def is_exact_interrupted_d_recovery(
            linked_row: Mapping[str, Any],
        ) -> bool:
            return bool(
                exact_interrupted_d_budget
                and linked_row.get("status") == "integrity-stopped"
                and "artifact" in linked_row
                and linked_row.get("artifact") is None
                and linked_row.get("failure") == expected_interrupted_d_failure
                and "artifact_binding" in linked_row
                and linked_row.get("artifact_binding") is None
            )

        def is_exact_single_recovery(linked_row: Mapping[str, Any]) -> bool:
            expected_failure = {
                "error_type": "BudgetFinalizedBeforeITT",
                "message": (
                    "a prior process created budget state without a terminal ITT "
                    "cell; the cell is retained and is not redispatched"
                ),
            }
            return bool(
                not is_d_group
                and linked_row.get("status") == "integrity-stopped"
                and linked_row.get("artifact") is None
                and linked_row.get("failure") == expected_failure
                and (
                    "artifact_binding" not in linked_row
                    or linked_row.get("artifact_binding") is None
                )
            )

        expected_interrupted_single_failure = {
            "error_type": "InterruptedReservation",
            "message": (
                "a prior process created budget state without a terminal ITT "
                "cell; the cell is retained and is not redispatched"
            ),
            "accounting_basis": "unreconciled-conservative-reservation",
        }
        exact_interrupted_single_budget = bool(
            not is_d_group
            and status == "integrity-stopped"
            and failure == expected_interrupted_single_failure
            and actual["cost_usd"] == float(reservation["cost_usd"])
            and actual["completions"] == int(reservation["completions"])
            and actual["storage_bytes"] == int(reservation["storage_bytes"])
        )

        def is_exact_interrupted_single_recovery(
            linked_row: Mapping[str, Any],
        ) -> bool:
            return bool(
                exact_interrupted_single_budget
                and linked_row.get("status") == "integrity-stopped"
                and "artifact" in linked_row
                and linked_row.get("artifact") is None
                and linked_row.get("failure") == expected_interrupted_single_failure
                and "artifact_binding" in linked_row
                and linked_row.get("artifact_binding") is None
            )

        expected_interrupted_after_itt_failure = {
            "error_type": "InterruptedReservationAfterITT",
            "message": (
                "a terminal ITT row retained an unreconciled reservation; the "
                "conservative reservation was charged before stopping"
            ),
            "accounting_basis": "unreconciled-conservative-reservation",
        }
        exact_interrupted_after_itt_budget = bool(
            not is_d_group
            and status == "integrity-stopped"
            and failure == expected_interrupted_after_itt_failure
            and actual["cost_usd"] == float(reservation["cost_usd"])
            and actual["completions"] == int(reservation["completions"])
            and actual["storage_bytes"] == int(reservation["storage_bytes"])
        )

        scheduled_rows = tuple(
            linked_row
            for linked_row in linked_rows
            if linked_row.get("status") == "scheduled"
        )
        original_rows = tuple(
            linked_row for linked_row in linked_rows if is_original_terminal(linked_row)
        )
        recovered_rows = tuple(
            linked_row for linked_row in linked_rows if is_exact_d_recovery(linked_row)
        )
        if scheduled_rows:
            # Recovery is atomic at this boundary: before it runs, every
            # non-scheduled D row must be an exact original budget terminal.
            finalized_statuses_valid = bool(
                (not is_d_group and len(scheduled_rows) == len(linked_rows) == 1)
                or (
                    is_d_group
                    and len(scheduled_rows) + len(original_rows) == len(linked_rows)
                )
            )
        elif is_d_group and exact_interrupted_d_budget:
            # Reserved-group recovery charges the full reservation first and
            # then atomically terminalizes only the scheduled ITT rows.  The
            # pre-existing terminals can differ from the new budget status,
            # but their original finalization events must still bind them.
            finalized_statuses_valid = bool(
                all(
                    linked_row.get("status") in linked_terminal
                    and has_bound_terminal_event(linked_run_id, linked_row)
                    and (
                        not (
                            linked_row.get("status") == status
                            and linked_row.get("failure") == failure
                        )
                        or is_exact_interrupted_d_recovery(linked_row)
                    )
                    for linked_run_id, linked_row in linked_row_items
                )
            )
        elif is_d_group and recovered_rows:
            # A fully scheduled group legitimately becomes all recovery rows;
            # a partial original group becomes exact original+recovery rows.
            finalized_statuses_valid = bool(
                len(original_rows) + len(recovered_rows) == len(linked_rows)
            )
        elif not is_d_group:
            linked_row = linked_rows[0]
            # A finalized single-run budget can precede its ITT write.  If
            # recovery changed the ITT status, require the exact conservative
            # recovery disposition; ordinary terminal replay remains valid
            # only when the ITT and budget statuses agree.  A reserved-run
            # recovery uses that same status on both ledgers, so its explicit
            # InterruptedReservation claim must be validated before the
            # ordinary same-status path.
            if (
                isinstance(failure, Mapping)
                and failure.get("error_type") == "InterruptedReservation"
            ):
                finalized_statuses_valid = is_exact_interrupted_single_recovery(
                    linked_row
                )
            elif (
                isinstance(failure, Mapping)
                and failure.get("error_type") == "InterruptedReservationAfterITT"
            ):
                finalized_statuses_valid = bool(
                    exact_interrupted_after_itt_budget
                    and linked_row.get("status") in linked_terminal
                    and has_bound_terminal_event(linked_row_items[0][0], linked_row)
                )
            else:
                finalized_statuses_valid = bool(
                    (
                        linked_row.get("status") == status
                        and linked_row.get("status") in linked_terminal
                    )
                    or is_exact_single_recovery(linked_row)
                )
        else:
            finalized_statuses_valid = bool(
                len(linked_statuses) == 1 and linked_statuses <= linked_terminal
            )
        if not finalized_statuses_valid:
            raise PilotV21110ContinuationError(
                f"finalized science budget/ITT terminality is invalid: {run_id}"
            )
    science_ids = set(rows) - {parent[0].run_id}
    if set(reservations_by_id) - {parent[0].run_id} != science_ids or set(
        finalizations_by_id
    ) - {parent[0].run_id} != {
        run_id for run_id in science_ids if rows[run_id].get("status") != "reserved"
    }:
        raise PilotV21110ContinuationError(
            "science budget row/event denominator drifted"
        )


def _verify_acceptance_identity(
    receipt: Mapping[str, Any],
    *,
    contract: Any,
    paid: Any,
) -> None:
    if (
        set(receipt) != _ACCEPTANCE_TOP_LEVEL_FIELDS
        or receipt.get("schema_version") != V21110_ACCEPTANCE_SCHEMA_VERSION
        or receipt.get("status") != "go"
        or receipt.get("go") is not True
        or receipt.get("contract_id") != V21110_CONTRACT_ID
        or receipt.get("contract_sha256") != contract.canonical_hash
        or canonical_sha256(receipt.get("release"))
        != canonical_sha256(_release_binding(contract, paid))
        or receipt.get("raw_namespace") != V21110_RAW_ROOT.as_posix()
        or canonical_sha256(receipt.get("denominator"))
        != canonical_sha256(_expected_acceptance_denominator(contract))
        or canonical_sha256(receipt.get("provider_boundary"))
        != canonical_sha256(_ACCEPTANCE_PROVIDER_BOUNDARY)
        or receipt.get("scientific_evidence") is not False
        or receipt.get("claim_boundary") != _ACCEPTANCE_CLAIM_BOUNDARY
    ):
        raise PilotV21110ContinuationError("V2.11.10 acceptance identity drifted")


def _verify_acceptance_core(
    receipt: Mapping[str, Any],
    *,
    contract: Any,
    repo_root: Path,
    raw_root: Path,
    paid: Any,
    run_ledger: Any,
    budget_ledger: PilotBudgetLedger,
    require_markers: bool,
) -> tuple[bool, bool]:
    _verify_seal(receipt, name="V2.11.10 scientific acceptance")
    _verify_acceptance_identity(receipt, contract=contract, paid=paid)
    material = _acceptance_material(
        contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
        budget_ledger=budget_ledger,
    )
    for key, value in material.items():
        if canonical_sha256(receipt.get(key)) != canonical_sha256(value):
            raise PilotV21110ContinuationError(
                f"V2.11.10 acceptance field {key!r} drifted"
            )
    run_snapshot = run_ledger.snapshot()
    budget_snapshot = budget_ledger.snapshot()
    rows = run_snapshot.get("runs")
    specs = tuple(contract.expand())
    if (
        not isinstance(rows, Mapping)
        or set(rows) != {spec.run_id for spec in specs}
        or any(rows[spec.run_id].get("spec") != spec.to_dict() for spec in specs)
    ):
        raise PilotV21110ContinuationError("accepted ITT denominator drifted")
    _verify_current_accepted_budget_rows(
        contract,
        receipt,
        budget_snapshot,
        run_snapshot,
    )
    prefixes = receipt.get("ledger_prefixes")
    if not isinstance(prefixes, Mapping) or set(prefixes) != {
        "run_ledger",
        "budget_ledger",
    }:
        raise PilotV21110ContinuationError("acceptance ledger prefixes are absent")
    relative = (V21110_RAW_ROOT / V21110_ACCEPTANCE_FILENAME).as_posix()
    run_marked = _verify_acceptance_prefix(
        run_snapshot,
        prefix=prefixes["run_ledger"],
        receipt=receipt,
        receipt_path=relative,
        budget=False,
    )
    budget_marked = _verify_acceptance_prefix(
        budget_snapshot,
        prefix=prefixes["budget_ledger"],
        receipt=receipt,
        receipt_path=relative,
        budget=True,
    )
    if require_markers and not (run_marked and budget_marked):
        raise PilotV21110ContinuationError(
            "both acceptance markers are required before dispatch"
        )
    return run_marked, budget_marked


def verify_v21110_scientific_dispatch_acceptance(
    receipt_path: str | Path,
    *,
    contract: Any,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: Any,
    run_ledger: Any,
    budget_ledger: PilotBudgetLedger,
) -> dict[str, Any]:
    repository = _real_root(repo_root, name="V2.11.10 repository")
    _verify_v21110_bound_working_directory(repository)
    raw = _real_root(raw_root, name="V2.11.10 raw root")
    path = Path(receipt_path).absolute()
    if path != raw / V21110_ACCEPTANCE_FILENAME:
        raise PilotV21110ContinuationError("V2.11.10 acceptance path drifted")
    receipt = _strict_json(path, name="V2.11.10 scientific acceptance")
    try:
        with v2117._acceptance_provider_sentinels():
            _verify_acceptance_core(
                receipt,
                contract=contract,
                repo_root=repository,
                raw_root=raw,
                paid=paid,
                run_ledger=run_ledger,
                budget_ledger=budget_ledger,
                require_markers=True,
            )
    except v2117.PilotV2117ContinuationError as exc:
        raise PilotV21110ContinuationError(str(exc)) from exc
    return receipt


def accept_v21110_scientific_dispatch(
    *,
    contract_path: str | Path,
    repo_root: str | Path,
    raw_root: str | Path,
    scientific_launch_input_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    from . import pilot_orchestrator as orch

    require_v21110_provider_keys_absent()
    repository = _real_root(repo_root, name="V2.11.10 repository")
    _verify_v21110_bound_working_directory(repository)
    candidate = Path(contract_path)
    if not candidate.is_absolute():
        candidate = repository / candidate
    expected_contract = repository / "experiments/pilot_v2_11_10.yaml"
    if candidate.absolute() != expected_contract:
        raise PilotV21110ContinuationError("acceptance contract path drifted")
    contract = load_pilot_contract(candidate)
    raw = _real_root(raw_root, name="V2.11.10 raw root")
    if raw != repository.joinpath(*V21110_RAW_ROOT.parts):
        raise PilotV21110ContinuationError("acceptance raw namespace drifted")
    output = raw / V21110_ACCEPTANCE_FILENAME
    if receipt_path is not None and Path(receipt_path).absolute() != output:
        raise PilotV21110ContinuationError("acceptance output path drifted")
    launch = Path(scientific_launch_input_path).absolute()
    if launch != raw / "scientific_launch_input.json":
        raise PilotV21110ContinuationError("scientific launch input path drifted")
    with orch._exclusive_real_stage_lock(
        raw, stage_id="scientific-dispatch-acceptance"
    ):
        paid = orch.verify_paid_provenance(
            contract,
            repo_root=repository,
            scientific_launch_input_path=launch,
        )
        run_ledger = orch.PilotRunLedger(
            raw / "run_ledger.json",
            contract_hash=contract.canonical_hash,
            tamper_evident=True,
            bind_terminal_artifacts=True,
        )
        budget_ledger = PilotBudgetLedger(
            raw / "budget_ledger.json",
            contract_hash=contract.canonical_hash,
            caps=orch._budget_caps(contract),
            tamper_evident=True,
            parent_debit=parent_budget_debit_for_v21110(contract),
        )
        _audit_pre_science_namespace(raw, contract)
        try:
            with v2117._acceptance_provider_sentinels():
                if output.exists():
                    receipt = _strict_json(output, name="V2.11.10 scientific acceptance")
                    run_marked, budget_marked = _verify_acceptance_core(
                        receipt,
                        contract=contract,
                        repo_root=repository,
                        raw_root=raw,
                        paid=paid,
                        run_ledger=run_ledger,
                        budget_ledger=budget_ledger,
                        require_markers=False,
                    )
                    if not run_marked and not budget_marked:
                        candidate_receipt = _acceptance_receipt(
                            contract,
                            repo_root=repository,
                            raw_root=raw,
                            paid=paid,
                            run_ledger=run_ledger,
                            budget_ledger=budget_ledger,
                        )
                        _write_once(output, candidate_receipt)
                        receipt = candidate_receipt
                else:
                    receipt = _acceptance_receipt(
                        contract,
                        repo_root=repository,
                        raw_root=raw,
                        paid=paid,
                        run_ledger=run_ledger,
                        budget_ledger=budget_ledger,
                    )
                    _write_once(output, receipt)
                    _verify_acceptance_core(
                        receipt,
                        contract=contract,
                        repo_root=repository,
                        raw_root=raw,
                        paid=paid,
                        run_ledger=run_ledger,
                        budget_ledger=budget_ledger,
                        require_markers=False,
                    )
        except v2117.PilotV2117ContinuationError as exc:
            raise PilotV21110ContinuationError(str(exc)) from exc
        prefixes = receipt["ledger_prefixes"]
        relative = (V21110_RAW_ROOT / V21110_ACCEPTANCE_FILENAME).as_posix()
        common = {
            "receipt_schema_version": V21110_ACCEPTANCE_SCHEMA_VERSION,
            "receipt_path": relative,
            "receipt_content_sha256": receipt["integrity"]["content_sha256"],
            "accepted_run_event_count": prefixes["run_ledger"]["event_count"],
            "accepted_run_event_chain_head": prefixes["run_ledger"]["event_chain_head"],
            "accepted_budget_event_count": prefixes["budget_ledger"]["event_count"],
            "accepted_budget_event_chain_head": prefixes["budget_ledger"][
                "event_chain_head"
            ],
        }
        run_ledger.bind_acceptance_receipt(**common)
        budget_ledger.bind_acceptance_receipt(**common)
        reloaded_run = orch.PilotRunLedger(
            raw / "run_ledger.json",
            contract_hash=contract.canonical_hash,
            tamper_evident=True,
            bind_terminal_artifacts=True,
        )
        reloaded_budget = PilotBudgetLedger(
            raw / "budget_ledger.json",
            contract_hash=contract.canonical_hash,
            caps=orch._budget_caps(contract),
            tamper_evident=True,
            parent_debit=parent_budget_debit_for_v21110(contract),
        )
        return verify_v21110_scientific_dispatch_acceptance(
            output,
            contract=contract,
            repo_root=repository,
            raw_root=raw,
            paid=paid,
            run_ledger=reloaded_run,
            budget_ledger=reloaded_budget,
        )


__all__ = [
    "PilotV21110ContinuationError",
    "V21110_CONTRACT_ID",
    "V21110_ACCEPTANCE_FILENAME",
    "V21110_ACCEPTANCE_SCHEMA_VERSION",
    "V21110_CURRENT_AUTHORITY_SCHEMA_VERSION",
    "V21110_PARENT_IMPORT_SCHEMA_VERSION",
    "V21110_RAW_ROOT",
    "V21110_SCIENCE_TAG",
    "V21110_SOURCE_MANIFEST_PATH",
    "V21110_SOURCE_MANIFEST_SCHEMA_VERSION",
    "build_v21110_source_manifest",
    "build_v21110_parent_import_receipt",
    "accept_v21110_scientific_dispatch",
    "current_authority_path",
    "current_projection_path",
    "parent_budget_debit_for_v21110",
    "require_v21110_provider_keys_absent",
    "runner_reservations_for_v21110",
    "verified_v21110_calibration",
    "verified_v21110_observed_p95_authority_binding",
    "verified_v21110_projection",
    "verify_v2119_terminal_no_go",
    "verify_v21110_current_authority",
    "verify_v21110_parent_import_receipt",
    "verify_v21110_scientific_dispatch_acceptance",
    "validate_v21110_source_manifest",
    "verify_v21110_terminal_scientific_artifacts",
]
