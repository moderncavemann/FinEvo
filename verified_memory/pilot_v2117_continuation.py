"""Zero-provider V2.11.7 recovery continuation and dispatch authority.

V2.11.7 is deliberately not a retry of either the terminal V2.11.5 A/C cells
or the terminal V2.11.6 integrity no-go.  It binds the complete V2.11.6
fail-closed receipt as lineage, imports the immutable V2.11.5 *science
authority view*, registers a fresh 87-cell ledger (one operational import plus
the same 86 untouched D/B/cross cells), and seals the remaining provider
matrix before credentials may be loaded.

This module never imports or constructs an LLM provider.  Runtime integration
in :mod:`verified_memory.pilot_orchestrator` calls these functions before its
provider-catalog/provider-construction boundary.
"""

from __future__ import annotations

import ast
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence
from unittest import mock

import llm_providers as canonical_llm_providers

from .pilot_budget import ParentBudgetDebit, PilotBudgetLedger
from .pilot_contract import PilotContract, canonical_sha256, load_pilot_contract
from . import pilot_provider_catalog as canonical_provider_catalog


V2117_CONTRACT_ID = "finevo-pilot-v2.11.7"
V2117_SCIENCE_TAG = "pilot-v2.11.7-science"
V2117_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_11_7_source_manifest.json"
)
V2117_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.11.7/raw")
V2117_PARENT_IMPORT_SCHEMA_VERSION = "finevo-pilot-v2.11.7-parent-import-v1"
V2117_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.11.7-source-manifest-v1"
V2117_CURRENT_AUTHORITY_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.7-continuation-observed-p95-authority-v1"
)
V2117_CURRENT_PROJECTION_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.7-continuation-projection-v1"
)
V2117_ACCEPTANCE_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.7-scientific-dispatch-acceptance-v1"
)
V2117_ACCEPTANCE_FILENAME = "scientific_dispatch_acceptance.json"

V2116_CONTRACT_ID = "finevo-pilot-v2.11.6"
V2116_SCIENCE_TAG = "pilot-v2.11.6-science"
V2116_SCIENCE_COMMIT = "0a7eb29a76c5f9c90486052a4c335ad1d2000bf0"
V2116_SCIENCE_TAG_OBJECT = "6355d2329d800c95595c89f5e41e032ba6129fb7"
V2116_CONTRACT_SHA256 = "879359813cf733e1aced869b28adcbeaffdb4dd4333226224601e82fa36f0fac"
V2116_CONTRACT_FILE_SHA256 = "8670a2c464214f8c63b5c4712baf946f26cb85fc727dec7f6c1c6a933979792a"
V2116_SOURCE_MANIFEST_FILE_SHA256 = (
    "710db4414471005d088cd64fb1e1a7c4a46fd99f8852b05f3f17f2acaead240d"
)
V2116_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "c510941c565d1120604199139d193990948d6b65be15a823ba1d4850968f2ce0"
)
V2116_RUN_LEDGER_FILE_SHA256 = (
    "e43ee536232369cabc27215e8c066a2d1834c27505da138652c76a578ea8af40"
)
V2116_RUN_LEDGER_SHA256 = (
    "0fcd05fea8cf93574e69fc2b7a0c94171a952e85e7932d035ea85784fbb8594d"
)
V2116_RUN_EVENT_COUNT = 89
V2116_RUN_EVENT_HEAD = (
    "d2749697e1139ea403e43a164901c0ff36b1f83814a8035a5d292ff47b0f2100"
)
V2116_BUDGET_LEDGER_FILE_SHA256 = (
    "d36b2cded3ea7ddd8e7354173aaa5ad26e72ad8430d5d65071ef59d121ff1b0a"
)
V2116_BUDGET_LEDGER_SHA256 = (
    "a35780543ba195257da51a66dbfc7f9f662c1b6546b728e97e9041920795afea"
)
V2116_BUDGET_EVENT_COUNT = 4
V2116_BUDGET_EVENT_HEAD = (
    "16a1a891adbee9b44e92ebce969485ec1ee8bb785aa8c70a5df4542e162bcbf2"
)
V2116_RAW_INVENTORY_FILE_COUNT = 5
V2116_RAW_INVENTORY_STORAGE_BYTES = 215_033
V2116_RAW_INVENTORY_SHA256 = (
    "0dbf0a293b9b2c00c642aa8d7724eb7b585e0b23ad327504e15edaf63e5e234d"
)
V2116_RAW_FILE_BINDINGS: Mapping[str, tuple[int, str]] = {
    "budget_ledger.json": (
        5_673,
        "d36b2cded3ea7ddd8e7354173aaa5ad26e72ad8430d5d65071ef59d121ff1b0a",
    ),
    "parent-import/stage_receipt.json": (
        1_696,
        "f91d6631b718265cc7ff8682089b184780a4fa390237484b335c05d580fa1d0f",
    ),
    "release_attestation.json": (
        15_558,
        "456ac803bd6120b5675417251041ebf8aa3b4ba1c928d195b0fb831c2c676cef",
    ),
    "run_ledger.json": (
        189_887,
        "e43ee536232369cabc27215e8c066a2d1834c27505da138652c76a578ea8af40",
    ),
    "scientific_launch_input.json": (
        2_219,
        "cb4df8813f13c297a78da2e9d7c4c280deb4fbb991c3e80f89d10fe3cade4c62",
    ),
}
V2116_PARENT_IMPORT_RECEIPT_CONTENT_SHA256 = (
    "52905fa9f9fcf3cb9e65579e49f5b8bca77dd0d252b314cbfbf7fd34ab0e6e69"
)
V2116_PARENT_IMPORT_ACTUAL_STORAGE_BYTES = 1_696
V2116_IMPORTED_PARENT_DEBIT_RECORD_SHA256 = (
    "bada157f174d33344370c621f0bd480d57cf8ff5adcde498d7e02426a4363270"
)
V2116_CUMULATIVE_COST_USD = 63.1196450625
V2116_CUMULATIVE_COMPLETIONS = 3_440
V2116_CUMULATIVE_STORAGE_BYTES = 270_189_931

V2115_CONTRACT_ID = "finevo-pilot-v2.11.5"
V2115_SCIENCE_TAG = "pilot-v2.11.5-science"
V2115_SCIENCE_COMMIT = "2351ac2283f9fedb9dce70067174020be56ed9cc"
V2115_SCIENCE_TAG_OBJECT = "bccfb13cee7d592470d1873cfacc3b12bed38be4"
V2115_CONTRACT_SHA256 = "e1ecdec43e3f7a7b9a3d0977e2522d95861e826fc68781377d7eaceeb5e6e2ef"
V2115_CONTRACT_FILE_SHA256 = "b96438430231f0c46fd6c5f15ba749713534feb15f964c496aa02606cf11103b"
V2115_SOURCE_MANIFEST_FILE_SHA256 = (
    "fea5a276fb64fdd5bf0539014687ea39a891e9d305205b1d2046a2c15a892d16"
)
V2115_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "be84d33f561a5ab8927f13e0753f5109b5f018dc790ae180d5e0e6e0228af559"
)
V2115_RUN_LEDGER_SHA256 = "8a86231f0906ea117626190cc7a2699933c968ce555612cb1bc6378473601fa7"
V2115_RUN_EVENT_COUNT = 53
V2115_RUN_EVENT_HEAD = "61489ef64e71400e603e2fb1110e5e8af3ba772ac083361338a4ccff9641022f"
V2115_BUDGET_LEDGER_SHA256 = (
    "53e70f6c0b9053674408de385e1a5b5bf42ace7e82dc8e0c6f227ea124b7a38f"
)
V2115_BUDGET_EVENT_COUNT = 103
V2115_BUDGET_EVENT_HEAD = (
    "f745a8c4087b310d5d5cfb74645df8ddb0f2f80ef3d269b9642d3715f7de5834"
)
V2115_RAW_INVENTORY_FILE_COUNT = 691
V2115_RAW_INVENTORY_STORAGE_BYTES = 48_820_556
V2115_RAW_INVENTORY_SHA256 = (
    "f2fdb1ccedcb70e6793d3b8f3c87425f0d602552f0a3e0e7f35db9c5777c6746"
)
V2115_CURRENT_COST_USD = 43.1214245
V2115_CURRENT_COMPLETIONS = 2_436
V2115_CURRENT_STORAGE_BYTES = 48_139_533
V2115_HOSTED_RUN_COUNT = 47
V2115_HOSTED_STORAGE_BYTES = 47_975_380
V2115_OPERATIONAL_RUN_COUNT = 3
V2115_OPERATIONAL_STORAGE_BYTES = 164_153
V2115_CUMULATIVE_COST_USD = 63.1196450625
V2115_CUMULATIVE_COMPLETIONS = 3_440
V2115_CUMULATIVE_STORAGE_BYTES = 270_188_235

V2117_REMAINING_CELL_COUNT = 86
V2117_LEDGER_CELL_COUNT = 87
V2117_REMAINING_PROVIDER_COMPLETIONS = 3_256
V2117_REMAINING_PROJECTED_COST_USD = 149.3301875
V2117_REMAINING_PROJECTED_STORAGE_BYTES = 1_020_000_000
V2117_PROJECTED_CUMULATIVE_COST_USD = 212.4498325625
V2117_PROJECTED_CUMULATIVE_COMPLETIONS = 6_696
V2117_PROJECTED_CUMULATIVE_STORAGE_BYTES = 1_290_189_931

V2115_PARENT_IMPORT_FILE_SHA256 = (
    "104966506803b93e009730fb3e2f742a1eded17e6f5c210faacff1f5ffc5ace8"
)
V2115_PARENT_IMPORT_CONTENT_SHA256 = (
    "3eb37262ca7a3df78436e964bc202768e7b5ca4417d947c7a61f419eec24a658"
)
V2115_CALIBRATION_CONTENT_SHA256 = (
    "50a238c086aa698badb69ae4b2fa22f465b8c6e49273897ba6227fbe4b459ffe"
)
V2115_CAPABILITY_CONTENT_SHA256 = {
    "gpt52_main": "be8684bd1208bb5049be744910c10bdaf5f48e69ad6f13ae086ecef9ce42e32f",
    "gpt56_diagnostic": "f3a3025347327545e33d42149efe1ed0d29c3279429b3f379ecc88c6cdeab863",
}
V2115_Q_REF = 63.50397933257746
V2115_ABSOLUTE_FLOW_UTILITY_THRESHOLD = 0.05617208967516696
V2115_RESERVATION_SET_SHA256 = (
    "efcc9d0b4e80306f16661b71d19221da3596b02889698a7bfb4382a40c5521f0"
)
V2115_STABLE_AUTHORITY_SET_SHA256 = (
    "f7e87d6f017d5072575be2f9d879b871c2df1925ac43a1a03724d8936b8db52e"
)
V2115_STAGE_RECEIPT_BINDINGS: Mapping[str, Mapping[str, Any]] = {
    "parent-import": {
        "file_sha256": "2ec91fe97e65f11dcabc3e15d539ae5dbd19ef9d7ca0ccf23cb4fce26749d8ba",
        "content_sha256": "e69c7b79b8ffa0c26bcb14904c9f1dd6051e4919760fcefd3df3983f3db0e703",
        "status": "complete", "go": True, "execution_progression_go": True,
        "scientific_matrix_complete": True, "registered_run_count": 1,
        "status_counts": {"complete": 1},
    },
    "capability-gate": {
        "file_sha256": "e43212244a98d670447d85de1b807e88e391d0cf44863bc83e6d4aed3826a18a",
        "content_sha256": "90f5d2e2a5eea9335a1aeb56556b21bd351a1af0aecbdd4b8d006bf7c1873023",
        "status": "complete", "go": True, "execution_progression_go": True,
        "scientific_matrix_complete": True, "registered_run_count": 2,
        "status_counts": {"complete": 2},
    },
    "long-context-preflight": {
        "file_sha256": "b79e6af1c5c56e46b979a3ce817ae51a79a387ab44c9c5e2d5300b601215edb6",
        "content_sha256": "db44a4a0d532411922cf22266fff0554d45c46a8bc8abae5aedecfedf5dae99c",
        "status": "complete", "go": True, "execution_progression_go": True,
        "scientific_matrix_complete": True, "registered_run_count": 2,
        "status_counts": {"complete": 2},
    },
    "experiment-c": {
        "file_sha256": "958cb161785c144c89861da3e9536e53069e8f1070a64c03f54647cbfe05b322",
        "content_sha256": "39a9d35f4961fee4b0bc59ac67f7a9a2da0c3f95fddf77a418b92e518b6e2eba",
        "status": "complete-with-no-go", "go": False,
        "execution_progression_go": True, "scientific_matrix_complete": True,
        "registered_run_count": 25, "status_counts": {"complete": 25},
    },
    "experiment-a": {
        "file_sha256": "8193f3449663f63c9cf0c881ee5e7759d2682f320f214c4941040489c81734f9",
        "content_sha256": "177dc8ce4d1957eac0734bb1716279676f77931e30b3a1d10dd2c138a43a5457",
        "status": "complete-with-no-go", "go": False,
        "execution_progression_go": True, "scientific_matrix_complete": False,
        "registered_run_count": 20,
        "status_counts": {"complete": 17, "failed": 3},
    },
}

_CANONICALIZATION = "json-sort-keys-utf8-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_FAILED_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_11_6.yaml")
_FAILED_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_11_6_source_manifest.json"
)
_FAILED_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.11.6/raw")
_AUTHORITY_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_11_5.yaml")
_AUTHORITY_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_11_5_source_manifest.json"
)
_AUTHORITY_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.11.5/raw")
_AUTHORITY_GATE_PATH = (
    _AUTHORITY_RAW_ROOT / "long-context-preflight/post_gate_authority.json"
)
# Compatibility aliases for the mechanically inherited V2.11.5 authority
# implementation below.  They never refer to the V2.11.6 failed lineage.
_PARENT_CONTRACT_PATH = _AUTHORITY_CONTRACT_PATH
_PARENT_SOURCE_MANIFEST_PATH = _AUTHORITY_SOURCE_MANIFEST_PATH
_PARENT_RAW_ROOT = _AUTHORITY_RAW_ROOT
_PARENT_GATE_PATH = _AUTHORITY_GATE_PATH
_CURRENT_AUTHORITY_PATH = (
    V2117_RAW_ROOT / "parent-import/current_authority/post_gate_authority.json"
)

# These execution sources must be byte-identical to the V2.11.5 science tag.
# Continuation glue, the contract parser, run_pilot.py, and the orchestrator are
# intentionally excluded and are instead constrained by tests/source manifests.
_BYTE_IDENTICAL_SCIENCE_PATHS = (
    "config.yaml",
    "llm_providers.py",
    "verified_memory/actions.py",
    "verified_memory/foundation_adapter.py",
    "verified_memory/artifacts.py",
    "verified_memory/budget.py",
    "verified_memory/failure_artifacts.py",
    "verified_memory/m0_utility.py",
    "verified_memory/m1_context.py",
    "verified_memory/m2_episodic.py",
    "verified_memory/m3_semantic.py",
    "verified_memory/pilot_analysis.py",
    "verified_memory/pilot_budget.py",
    "verified_memory/pilot_checkpoint.py",
    "verified_memory/pilot_continuation.py",
    "verified_memory/pilot_provider_catalog.py",
    "verified_memory/pilot_release_attestation.py",
    "verified_memory/prompts.py",
    "verified_memory/runner.py",
    "verified_memory/runner_artifacts.py",
    "verified_memory/scientific_release_attestation.py",
    "verified_memory/scripted_provider.py",
    "verified_memory/system.py",
)
_UNCHANGED_ORCHESTRATOR_AST_FUNCTIONS = (
    "_shock_events",
    "resolve_utility",
    "config_for_spec",
    "projection_from_preflight",
    "_execute_actor_run",
    "_d_continuation_causal_bindings",
    "_d_narrative_causal_bindings",
)
_REVIEWED_CHANGED_ORCHESTRATOR_FUNCTIONS = frozenset(
    {
        "_build_experiment_c_sensitivity",
        "_d_group_projection",
        "_execute_d_seed",
        "_execute_stage_locked",
        "_load_verified_experiment_c_sensitivity",
        "_load_verified_projection",
        "_load_verified_q_ref",
        "_load_verified_stage0_selection",
        "_observed_p95_authority_receipt_path",
        "_parent_budget_debit",
        "_runner_p95_reservations",
        "run_development_fake_matrix",
        "_v2_control_gate_ok",
        "_verified_observed_p95_binding",
        "_write_experiment_c_sensitivity",
    }
)
_V2116_NEW_ORCHESTRATOR_FUNCTIONS = frozenset(
    {
        "_execute_v2116_parent_import_stage",
        "_load_verified_v2116_calibration",
        "_v2116_parent_import_projection",
        "build_v2116_experiment_d_group_plan",
    }
)
_REVIEWED_NEW_ORCHESTRATOR_FUNCTIONS = frozenset(
    {
        *_V2116_NEW_ORCHESTRATOR_FUNCTIONS,
        "_execute_v2117_parent_import_stage",
        "_load_verified_v2117_calibration",
        "_v2117_parent_import_projection",
        "build_v2117_experiment_d_group_plan",
    }
)
_REVIEWED_CHANGED_CONTRACT_NODES = frozenset(
    {
        "PilotContract",
        "ReleaseRequirements",
        "load_pilot_contract",
    }
)
_REVIEWED_NEW_CONTRACT_NODES = frozenset(
    {
        "_v2_11_6_expected_continuation_boundary",
        "_v2_11_6_expected_model_roles",
        "_v2_11_6_expected_non_claims",
        "_v2_11_6_expected_parent_import_arm",
        "_v2_11_6_expected_stages",
        "_validate_v2_11_6_continuation_boundary",
        "_v2_11_7_expected_recovery_boundary",
        "_v2_11_7_expected_model_roles",
        "_v2_11_7_expected_non_claims",
        "_v2_11_7_expected_parent_import_arm",
        "_v2_11_7_expected_stages",
        "_validate_v2_11_7_recovery_boundary",
    }
)
_ACCEPTANCE_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version", "status", "go", "contract_id", "contract_sha256",
        "release", "raw_namespace", "denominator", "parent_import",
        "current_authority", "runner_configs", "budget_projection",
        "ledger_prefixes", "provider_boundary", "scientific_evidence",
        "claim_boundary", "integrity",
    }
)
_PROVIDER_KEY_ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
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
    "Pre-dispatch continuation integrity and budget acceptance only; no "
    "treatment outcome is created or imported."
)


class PilotV2117ContinuationError(RuntimeError):
    """Raised before V2.11.7 may construct a provider."""


def _json_copy(value: Any) -> Any:
    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: thaw(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [thaw(child) for child in item]
        return item

    try:
        return json.loads(
            json.dumps(thaw(value), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise PilotV2117ContinuationError("value is not canonical JSON") from exc


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PilotV2117ContinuationError(f"{name} must be a regular non-symlink file")

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise PilotV2117ContinuationError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise PilotV2117ContinuationError(f"{name} contains non-finite {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotV2117ContinuationError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PilotV2117ContinuationError(f"{name} must contain an object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_sha256(value: Mapping[str, Any]) -> str:
    unsigned = _json_copy(value)
    integrity = unsigned.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("content_sha256", None)
    else:
        unsigned.pop("receipt_sha256", None)
    return canonical_sha256(unsigned)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _json_copy(value)
    payload["integrity"] = {"canonicalization": _CANONICALIZATION}
    payload["integrity"]["content_sha256"] = _content_sha256(payload)
    return payload


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    pending = path.with_name(f".{path.name}.pending")
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise PilotV2117ContinuationError(f"immutable artifact drifted: {path}")
        if os.path.lexists(pending):
            if pending.is_symlink() or not pending.is_file():
                raise PilotV2117ContinuationError(
                    f"immutable artifact pending path drifted: {pending}"
                )
            pending.unlink()
            _fsync_directory(path.parent)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PilotV2117ContinuationError(
            f"immutable artifact parent is not a real directory: {path.parent}"
        )
    if os.path.lexists(pending):
        if pending.is_symlink() or not pending.is_file():
            raise PilotV2117ContinuationError(
                f"immutable artifact pending path drifted: {pending}"
            )
        # A killed writer can leave only this deterministic same-directory
        # staging file.  It has never been published, so discard and rebuild.
        pending.unlink()
        _fsync_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(pending, flags, 0o644)
    try:
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short write while staging immutable artifact")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    published = False
    try:
        try:
            # A same-directory hard link is an atomic create-if-absent publish;
            # unlike replace/rename it can never overwrite an immutable target.
            os.link(pending, path, follow_symlinks=False)
            published = True
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise PilotV2117ContinuationError(
                    f"immutable artifact drifted: {path}"
                )
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            if published and os.path.lexists(path):
                path.unlink()
            raise PilotV2117ContinuationError(
                f"immutable artifact failed post-publication verification: {path}"
            )
    finally:
        try:
            if os.path.lexists(pending):
                if pending.is_symlink() or not pending.is_file():
                    raise PilotV2117ContinuationError(
                        f"immutable artifact pending path drifted: {pending}"
                    )
                pending.unlink()
        finally:
            _fsync_directory(path.parent)


def _git(root: Path, *args: str) -> str:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_") and not key.startswith("DYLD_")
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
    result = subprocess.run(
        ("/usr/bin/git", *args),
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0 or result.stderr:
        raise PilotV2117ContinuationError(
            "read-only git identity check failed: "
            + (result.stderr.strip() or result.stdout.strip())
        )
    return result.stdout.strip()


def _real_root(value: str | Path, *, name: str) -> Path:
    root = Path(value).absolute()
    try:
        mode = root.lstat().st_mode
    except OSError as exc:
        raise PilotV2117ContinuationError(f"{name} is unavailable") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise PilotV2117ContinuationError(f"{name} must be a real directory")
    return root


def _verify_release_git(
    root: Path,
    *,
    name: str,
    science_tag: str,
    science_commit: str,
    science_tag_object: str,
) -> dict[str, str]:
    tag_ref = f"refs/tags/{science_tag}"
    observed = {
        "head_commit": _git(root, "rev-parse", "--verify", "HEAD"),
        "head_ref": _git(root, "rev-parse", "--symbolic-full-name", "HEAD"),
        "tag_type": _git(root, "cat-file", "-t", tag_ref),
        "tag_object": _git(root, "rev-parse", "--verify", tag_ref),
        "peeled_commit": _git(root, "rev-parse", "--verify", f"{tag_ref}^{{commit}}"),
        "status": _git(root, "status", "--porcelain=v1", "--untracked-files=all"),
    }
    expected = {
        "head_commit": science_commit,
        "head_ref": "HEAD",
        "tag_type": "tag",
        "tag_object": science_tag_object,
        "peeled_commit": science_commit,
        "status": "",
    }
    if observed != expected:
        raise PilotV2117ContinuationError(
            f"{name} is not the exact clean detached annotated science tag"
        )
    return {key: value for key, value in observed.items() if key != "status"} | {
        "science_tag": science_tag
    }


def _verify_failed_git(root: Path) -> dict[str, str]:
    return _verify_release_git(
        root,
        name="V2.11.6 failed release",
        science_tag=V2116_SCIENCE_TAG,
        science_commit=V2116_SCIENCE_COMMIT,
        science_tag_object=V2116_SCIENCE_TAG_OBJECT,
    )


def _verify_authority_git(root: Path) -> dict[str, str]:
    return _verify_release_git(
        root,
        name="V2.11.5 authority release",
        science_tag=V2115_SCIENCE_TAG,
        science_commit=V2115_SCIENCE_COMMIT,
        science_tag_object=V2115_SCIENCE_TAG_OBJECT,
    )


def _raw_inventory_rows(raw_root: Path, *, name: str) -> list[dict[str, Any]]:
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise PilotV2117ContinuationError(f"{name} raw root is unavailable")
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_root.rglob("*"), key=lambda item: item.as_posix()):
        if path == raw_root / ".real-stage-execution.lock":
            continue
        if path.is_symlink():
            raise PilotV2117ContinuationError(f"{name} raw tree contains a symlink")
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(raw_root).as_posix(),
                    "byte_size": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    return rows


def _raw_inventory(
    root: Path,
    *,
    raw_relative: PurePosixPath,
    name: str,
    expected_file_count: int,
    expected_storage_bytes: int,
    expected_inventory_sha256: str,
    expected_files: Mapping[str, tuple[int, str]] | None = None,
) -> dict[str, Any]:
    raw_root = root.joinpath(*raw_relative.parts)
    rows = _raw_inventory_rows(raw_root, name=name)
    result = {
        "root": raw_relative.as_posix(),
        "schema_version": "finevo-raw-tree-inventory-v1",
        "canonicalization": "json-sort-keys-compact-utf8-v1",
        "excluded_operational_paths": [".real-stage-execution.lock"],
        "file_count": len(rows),
        "storage_bytes": sum(int(row["byte_size"]) for row in rows),
        "inventory_sha256": canonical_sha256(rows),
    }
    if result != {
        "root": raw_relative.as_posix(),
        "schema_version": "finevo-raw-tree-inventory-v1",
        "canonicalization": "json-sort-keys-compact-utf8-v1",
        "excluded_operational_paths": [".real-stage-execution.lock"],
        "file_count": expected_file_count,
        "storage_bytes": expected_storage_bytes,
        "inventory_sha256": expected_inventory_sha256,
    }:
        raise PilotV2117ContinuationError(f"{name} raw inventory drifted")
    if expected_files is not None:
        observed_files = {
            str(row["path"]): (int(row["byte_size"]), str(row["sha256"]))
            for row in rows
        }
        if observed_files != dict(expected_files):
            raise PilotV2117ContinuationError(f"{name} raw file binding drifted")
    return result


def _failed_raw_inventory(root: Path) -> dict[str, Any]:
    return _raw_inventory(
        root,
        raw_relative=_FAILED_RAW_ROOT,
        name="V2.11.6 failed release",
        expected_file_count=V2116_RAW_INVENTORY_FILE_COUNT,
        expected_storage_bytes=V2116_RAW_INVENTORY_STORAGE_BYTES,
        expected_inventory_sha256=V2116_RAW_INVENTORY_SHA256,
        expected_files=V2116_RAW_FILE_BINDINGS,
    )


def _authority_raw_inventory(root: Path) -> dict[str, Any]:
    return _raw_inventory(
        root,
        raw_relative=_AUTHORITY_RAW_ROOT,
        name="V2.11.5 authority release",
        expected_file_count=V2115_RAW_INVENTORY_FILE_COUNT,
        expected_storage_bytes=V2115_RAW_INVENTORY_STORAGE_BYTES,
        expected_inventory_sha256=V2115_RAW_INVENTORY_SHA256,
    )


def _ast_function_digests(path: Path, names: Sequence[str]) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = {
        node.name: hashlib.sha256(
            ast.dump(node, annotate_fields=True, include_attributes=False).encode()
        ).hexdigest()
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in set(names)
    }
    if set(found) != set(names):
        raise PilotV2117ContinuationError(
            f"orchestrator AST allowlist is incomplete: {sorted(set(names)-set(found))}"
        )
    return dict(sorted(found.items()))


def _ast_top_level_inventory(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return dict(
        sorted(
            (
                node.name,
                hashlib.sha256(
                    ast.dump(
                        node, annotate_fields=True, include_attributes=False
                    ).encode()
                ).hexdigest(),
            )
            for node in tree.body
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
        )
    )


def _normalized_d_plan_receipts(
    child_contract: PilotContract,
    parent_contract: PilotContract,
) -> list[dict[str, Any]]:
    from . import pilot_orchestrator as orch

    rows: list[dict[str, Any]] = []
    for seed in child_contract.seeds["sets"]["main"]:
        child_specs = tuple(
            spec
            for spec in child_contract.expand(stage="experiment-d")
            if spec.environment_seed == seed
        )
        parent_specs = tuple(
            spec
            for spec in parent_contract.expand(stage="experiment-d")
            if spec.environment_seed == seed
        )
        if hasattr(orch, "build_v2117_experiment_d_group_plan"):
            child_plan = orch.build_v2117_experiment_d_group_plan(
                child_contract, child_specs
            )
        else:
            # Source-manifest rendering is also allowed before the V2.11.7
            # orchestrator adapter lands.  Exercise the reviewed V2.11.6
            # adapter against a contract-id-only compatibility view; the
            # normalized plan below must still match V2.11.5 byte-for-byte in
            # all science-relevant fields.  Once the V2.11.7 adapter exists,
            # the exact branch above is mandatory and its AST is sealed.
            child_plan = orch.build_v2116_experiment_d_group_plan(
                replace(child_contract, contract_id=V2116_CONTRACT_ID),
                child_specs,
            )
        parent_plan = orch.build_v2115_experiment_d_group_plan(
            parent_contract, parent_specs
        )

        def normalized(plan: Any) -> dict[str, Any]:
            specs = tuple(plan.continuation_specs.values()) + tuple(
                plan.narrative_specs.values()
            )
            normalized_specs: list[dict[str, Any]] = []
            for spec in specs:
                value = spec.to_dict()
                value.pop("run_id")
                value.pop("contract_id")
                value["budget_bucket"] = "normalized-hosted-continuation"
                normalized_specs.append(value)
            return {
                "seed": plan.representative.environment_seed,
                "registered_treatments": list(plan.registered_treatments),
                "specs": sorted(
                    normalized_specs,
                    key=lambda value: (value["arm_id"], value["narrative_id"]),
                ),
            }

        child_value = normalized(child_plan)
        parent_value = normalized(parent_plan)
        if child_value != parent_value:
            raise PilotV2117ContinuationError(
                f"V2.11.7 D plan differs from V2.11.5 at seed {seed}"
            )
        rows.append(
            {
                "seed": seed,
                "normalized_plan_sha256": canonical_sha256(child_value),
            }
        )
    if len(rows) != 5:
        raise PilotV2117ContinuationError("D plan equivalence lacks five seeds")
    return rows


def _implementation_equivalence(
    child: Path,
    parent: Path,
    *,
    child_contract: PilotContract,
    parent_contract: PilotContract,
) -> dict[str, Any]:
    identical: list[dict[str, Any]] = []
    for relative in _BYTE_IDENTICAL_SCIENCE_PATHS:
        child_path = child / relative
        parent_path = parent / relative
        if (
            child_path.is_symlink()
            or parent_path.is_symlink()
            or not child_path.is_file()
            or not parent_path.is_file()
        ):
            raise PilotV2117ContinuationError(
                f"required science source is unavailable: {relative}"
            )
        child_hash = _file_sha256(child_path)
        parent_hash = _file_sha256(parent_path)
        if child_hash != parent_hash:
            raise PilotV2117ContinuationError(
                f"remaining-science source differs from V2.11.5: {relative}"
            )
        identical.append({"path": relative, "file_sha256": child_hash})
    orchestrator_relative = "verified_memory/pilot_orchestrator.py"
    child_ast = _ast_function_digests(
        child / orchestrator_relative, _UNCHANGED_ORCHESTRATOR_AST_FUNCTIONS
    )
    parent_ast = _ast_function_digests(
        parent / orchestrator_relative, _UNCHANGED_ORCHESTRATOR_AST_FUNCTIONS
    )
    if child_ast != parent_ast:
        changed = sorted(name for name in child_ast if child_ast[name] != parent_ast[name])
        raise PilotV2117ContinuationError(
            f"remaining-science orchestrator functions drifted: {changed}"
        )
    child_inventory = _ast_top_level_inventory(child / orchestrator_relative)
    parent_inventory = _ast_top_level_inventory(parent / orchestrator_relative)
    changed_functions = {
        name
        for name in set(child_inventory) & set(parent_inventory)
        if child_inventory[name] != parent_inventory[name]
    }
    new_functions = set(child_inventory) - set(parent_inventory)
    removed_functions = set(parent_inventory) - set(child_inventory)
    if (
        changed_functions != _REVIEWED_CHANGED_ORCHESTRATOR_FUNCTIONS
        or (
            new_functions != _V2116_NEW_ORCHESTRATOR_FUNCTIONS
            and new_functions != _REVIEWED_NEW_ORCHESTRATOR_FUNCTIONS
        )
        or removed_functions
    ):
        raise PilotV2117ContinuationError(
            "orchestrator reviewed successor-delta inventory drifted: "
            f"changed={sorted(changed_functions)}, new={sorted(new_functions)}, "
            f"removed={sorted(removed_functions)}"
        )
    d_plans = _normalized_d_plan_receipts(child_contract, parent_contract)
    return {
        "policy": "science-core-equal-with-explicit-successor-adapter-v1",
        "byte_identical_files": identical,
        "byte_identical_files_sha256": canonical_sha256(identical),
        "orchestrator_path": orchestrator_relative,
        "unchanged_orchestrator_function_sha256": child_ast,
        "unchanged_orchestrator_set_sha256": canonical_sha256(child_ast),
        "orchestrator_full_file_sha256": {
            "parent": _file_sha256(parent / orchestrator_relative),
            "child": _file_sha256(child / orchestrator_relative),
        },
        "orchestrator_top_level_ast_inventory_sha256": {
            "parent": canonical_sha256(parent_inventory),
            "child": canonical_sha256(child_inventory),
        },
        "reviewed_changed_function_sha256": {
            name: {
                "parent": parent_inventory[name],
                "child": child_inventory[name],
            }
            for name in sorted(changed_functions)
        },
        "reviewed_new_function_sha256": {
            name: child_inventory[name] for name in sorted(new_functions)
        },
        "removed_top_level_functions": [],
        "experiment_d_normalized_plan_receipts": d_plans,
        "experiment_d_normalized_plan_set_sha256": canonical_sha256(d_plans),
        "equivalence_claim": "science_core_equal_with_explicit_successor_adapter",
        "full_runtime_byte_identity_claimed": False,
    }


def _current_runtime_source_bindings(root: Path, parent: Path) -> dict[str, Any]:
    full_hash_paths = (
        "run_pilot.py",
        "verified_memory/observed_p95_authority.py",
        "verified_memory/pilot_orchestrator.py",
        "verified_memory/pilot_v2117_continuation.py",
        "scripts/render_pilot_v2117_contract.py",
        "scripts/render_pilot_v2117_source_manifest.py",
    )
    files: list[dict[str, str]] = []
    for relative in full_hash_paths:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise PilotV2117ContinuationError(
                f"current runtime source is unavailable: {relative}"
            )
        files.append({"path": relative, "file_sha256": _file_sha256(path)})
    contract_relative = "verified_memory/pilot_contract.py"
    child_contract_inventory = _ast_top_level_inventory(root / contract_relative)
    parent_contract_inventory = _ast_top_level_inventory(parent / contract_relative)
    changed_contract_nodes = {
        name
        for name in set(child_contract_inventory) & set(parent_contract_inventory)
        if child_contract_inventory[name] != parent_contract_inventory[name]
    }
    new_contract_nodes = set(child_contract_inventory) - set(parent_contract_inventory)
    removed_contract_nodes = set(parent_contract_inventory) - set(
        child_contract_inventory
    )
    if (
        changed_contract_nodes != _REVIEWED_CHANGED_CONTRACT_NODES
        or new_contract_nodes != _REVIEWED_NEW_CONTRACT_NODES
        or removed_contract_nodes
    ):
        raise PilotV2117ContinuationError(
            "contract reviewed successor-delta inventory drifted: "
            f"changed={sorted(changed_contract_nodes)}, "
            f"new={sorted(new_contract_nodes)}, "
            f"removed={sorted(removed_contract_nodes)}"
        )
    evidence_relative = "verified_memory/pilot_evidence.py"
    evidence_functions = ("write_terminal_summary",)
    child_evidence = _ast_function_digests(
        root / evidence_relative, evidence_functions
    )
    parent_evidence = _ast_function_digests(
        parent / evidence_relative, evidence_functions
    )
    if child_evidence != parent_evidence:
        raise PilotV2117ContinuationError(
            "terminal artifact writer differs from V2.11.5"
        )
    return {
        "full_file_bindings": files,
        "full_file_binding_set_sha256": canonical_sha256(files),
        "pilot_contract_path": contract_relative,
        "pilot_contract_parent_full_file_sha256": _file_sha256(
            parent / contract_relative
        ),
        "pilot_contract_top_level_ast_inventory_sha256": {
            "parent": canonical_sha256(parent_contract_inventory),
            "child": canonical_sha256(child_contract_inventory),
        },
        "pilot_contract_top_level_node_counts": {
            "parent": len(parent_contract_inventory),
            "child": len(child_contract_inventory),
            "unchanged": len(
                set(child_contract_inventory)
                & set(parent_contract_inventory)
                - changed_contract_nodes
            ),
        },
        "pilot_contract_reviewed_changed_node_sha256": {
            name: {
                "parent": parent_contract_inventory[name],
                "child": child_contract_inventory[name],
            }
            for name in sorted(changed_contract_nodes)
        },
        "pilot_contract_reviewed_new_node_sha256": {
            name: child_contract_inventory[name]
            for name in sorted(new_contract_nodes)
        },
        "pilot_contract_removed_top_level_nodes": [],
        "terminal_artifact_writer": {
            "path": evidence_relative,
            "function_sha256": child_evidence,
            "parent_equal": True,
        },
        "cycle_avoidance": (
            "pilot_contract.py is fully top-level-AST-inventory bound with an exact "
            "reviewed delta because its assignments contain generated source-manifest "
            "hashes"
        ),
    }


def build_v2117_source_manifest(
    *,
    contract: PilotContract,
    repo_root: str | Path,
    failed_repo_root: str | Path,
    authority_repo_root: str | Path,
) -> dict[str, Any]:
    """Build deterministic child, failed-lineage, and authority identities."""

    if contract.contract_id != V2117_CONTRACT_ID:
        raise PilotV2117ContinuationError("source manifest requires V2.11.7")
    child = _real_root(repo_root, name="V2.11.7 repository")
    failed = _real_root(failed_repo_root, name="V2.11.6 failed repository")
    authority = _real_root(
        authority_repo_root, name="V2.11.5 authority repository"
    )

    failed_release = _verify_failed_git(failed)
    failed_contract_path = failed.joinpath(*_FAILED_CONTRACT_PATH.parts)
    failed_contract = load_pilot_contract(failed_contract_path)
    if (
        failed_contract.contract_id != V2116_CONTRACT_ID
        or failed_contract.canonical_hash != V2116_CONTRACT_SHA256
        or _file_sha256(failed_contract_path) != V2116_CONTRACT_FILE_SHA256
    ):
        raise PilotV2117ContinuationError(
            "V2.11.6 failed contract identity drifted"
        )
    failed_source_path = failed.joinpath(*_FAILED_SOURCE_MANIFEST_PATH.parts)
    failed_source = _strict_json(
        failed_source_path, name="V2.11.6 failed source manifest"
    )
    if (
        _file_sha256(failed_source_path) != V2116_SOURCE_MANIFEST_FILE_SHA256
        or failed_source.get("integrity", {}).get("content_sha256")
        != V2116_SOURCE_MANIFEST_CONTENT_SHA256
        or _content_sha256(failed_source)
        != V2116_SOURCE_MANIFEST_CONTENT_SHA256
    ):
        raise PilotV2117ContinuationError(
            "V2.11.6 failed source manifest drifted"
        )

    authority_release = _verify_authority_git(authority)
    authority_contract_path = authority.joinpath(*_AUTHORITY_CONTRACT_PATH.parts)
    authority_contract = load_pilot_contract(authority_contract_path)
    if (
        authority_contract.contract_id != V2115_CONTRACT_ID
        or authority_contract.canonical_hash != V2115_CONTRACT_SHA256
        or _file_sha256(authority_contract_path) != V2115_CONTRACT_FILE_SHA256
    ):
        raise PilotV2117ContinuationError(
            "V2.11.5 authority contract identity drifted"
        )
    authority_source_path = authority.joinpath(
        *_AUTHORITY_SOURCE_MANIFEST_PATH.parts
    )
    authority_source = _strict_json(
        authority_source_path, name="V2.11.5 authority source manifest"
    )
    if (
        _file_sha256(authority_source_path) != V2115_SOURCE_MANIFEST_FILE_SHA256
        or authority_source.get("integrity", {}).get("content_sha256")
        != V2115_SOURCE_MANIFEST_CONTENT_SHA256
        or _content_sha256(authority_source)
        != V2115_SOURCE_MANIFEST_CONTENT_SHA256
    ):
        raise PilotV2117ContinuationError(
            "V2.11.5 authority source manifest drifted"
        )
    cell_mapping = _canonical_remaining_cell_mapping(
        contract, authority_contract
    )
    declared_mapping = _boundary(contract).get("continuation_matrix", {}).get(
        "canonical_86_row_mapping_sha256"
    )
    if declared_mapping != cell_mapping["mapping_sha256"]:
        raise PilotV2117ContinuationError(
            "canonical 86-row mapping contract binding drifted"
        )
    return _seal(
        {
            "schema_version": V2117_SOURCE_MANIFEST_SCHEMA_VERSION,
            "contract_id": contract.contract_id,
            # The frozen contract binds this manifest's file/content hashes.
            # Do not put the contract hash back into the manifest: that would
            # create an unresolvable contract <-> manifest hash cycle.
            "release_tag": str(contract.implementation["required_git_tag"]),
            "failed_release": {
                "contract_id": V2116_CONTRACT_ID,
                "contract_sha256": V2116_CONTRACT_SHA256,
                "contract_file_sha256": V2116_CONTRACT_FILE_SHA256,
                "source_manifest_path": _FAILED_SOURCE_MANIFEST_PATH.as_posix(),
                "source_manifest_file_sha256": (
                    V2116_SOURCE_MANIFEST_FILE_SHA256
                ),
                "source_manifest_content_sha256": (
                    V2116_SOURCE_MANIFEST_CONTENT_SHA256
                ),
                **failed_release,
            },
            "failed_raw_inventory": _failed_raw_inventory(failed),
            "authority_release": {
                "contract_id": V2115_CONTRACT_ID,
                "contract_sha256": V2115_CONTRACT_SHA256,
                "contract_file_sha256": V2115_CONTRACT_FILE_SHA256,
                "source_manifest_path": (
                    _AUTHORITY_SOURCE_MANIFEST_PATH.as_posix()
                ),
                "source_manifest_file_sha256": V2115_SOURCE_MANIFEST_FILE_SHA256,
                "source_manifest_content_sha256": V2115_SOURCE_MANIFEST_CONTENT_SHA256,
                **authority_release,
            },
            "authority_raw_inventory": _authority_raw_inventory(authority),
            "canonical_remaining_cell_mapping": cell_mapping,
            "current_runtime_sources": _current_runtime_source_bindings(
                child, authority
            ),
            "remaining_science_implementation_equivalence": (
                _implementation_equivalence(
                    child,
                    authority,
                    child_contract=contract,
                    parent_contract=authority_contract,
                )
            ),
            "observation_boundary": {
                "failed_v2116_is_terminal_lineage_only": True,
                "failed_v2116_effect_rows_imported": 0,
                "authority_v2115_a_c_outcomes_are_frozen_external_evidence": True,
                "authority_v2115_a_c_rows_imported_into_child_ledger": 0,
                "authority_v2115_scheduled_cells_mapped_to_child": 86,
                "decoded_completion_reuse": False,
                "provider_calls": 0,
                "provider_construction": False,
            },
        }
    )


def _boundary(contract: PilotContract) -> Mapping[str, Any]:
    value = getattr(contract, "v2117_recovery_boundary", None)
    if not isinstance(value, Mapping):
        raise PilotV2117ContinuationError("V2.11.7 recovery boundary is absent")
    return value


def _expected_failed_release_no_go() -> dict[str, Any]:
    return {
        "contract_id": V2116_CONTRACT_ID,
        "contract_path": _FAILED_CONTRACT_PATH.as_posix(),
        "contract_file_sha256": V2116_CONTRACT_FILE_SHA256,
        "contract_sha256": V2116_CONTRACT_SHA256,
        "science_tag": V2116_SCIENCE_TAG,
        "science_tag_object": V2116_SCIENCE_TAG_OBJECT,
        "science_commit": V2116_SCIENCE_COMMIT,
        "source_manifest_path": _FAILED_SOURCE_MANIFEST_PATH.as_posix(),
        "source_manifest_file_sha256": V2116_SOURCE_MANIFEST_FILE_SHA256,
        "source_manifest_content_sha256": V2116_SOURCE_MANIFEST_CONTENT_SHA256,
        "raw_inventory": {
            "root": _FAILED_RAW_ROOT.as_posix(),
            "canonicalization": "json-sort-keys-compact-utf8-v1",
            "excluded_operational_paths": [".real-stage-execution.lock"],
            "file_count": V2116_RAW_INVENTORY_FILE_COUNT,
            "storage_bytes": V2116_RAW_INVENTORY_STORAGE_BYTES,
            "inventory_sha256": V2116_RAW_INVENTORY_SHA256,
        },
        "run_ledger": {
            "path": (_FAILED_RAW_ROOT / "run_ledger.json").as_posix(),
            "file_sha256": V2116_RUN_LEDGER_FILE_SHA256,
            "ledger_sha256": V2116_RUN_LEDGER_SHA256,
            "event_count": V2116_RUN_EVENT_COUNT,
            "event_head_sha256": V2116_RUN_EVENT_HEAD,
            "registered_rows": V2117_LEDGER_CELL_COUNT,
            "status_counts": {"integrity-stopped": V2117_LEDGER_CELL_COUNT},
        },
        "budget_ledger": {
            "path": (_FAILED_RAW_ROOT / "budget_ledger.json").as_posix(),
            "file_sha256": V2116_BUDGET_LEDGER_FILE_SHA256,
            "ledger_sha256": V2116_BUDGET_LEDGER_SHA256,
            "event_count": V2116_BUDGET_EVENT_COUNT,
            "event_head_sha256": V2116_BUDGET_EVENT_HEAD,
            "current_actual": {
                "cost_usd": 0.0,
                "hosted_completions": 0,
                "storage_bytes": V2116_PARENT_IMPORT_ACTUAL_STORAGE_BYTES,
            },
        },
        "stage_receipt": {
            "path": (
                _FAILED_RAW_ROOT / "parent-import/stage_receipt.json"
            ).as_posix(),
            "file_sha256": V2116_RAW_FILE_BINDINGS[
                "parent-import/stage_receipt.json"
            ][1],
            "content_sha256": V2116_PARENT_IMPORT_RECEIPT_CONTENT_SHA256,
            "status": "integrity-stopped",
            "go": False,
            "execution_progression_go": False,
            "failure_error_type": "V2116ParentImportIntegrityError",
            "failure_cause_type": "PilotV2116ContinuationError",
            "failure_message": "V2.11.5 current-release actual debit drifted",
        },
        "release_attestation": {
            "path": (_FAILED_RAW_ROOT / "release_attestation.json").as_posix(),
            "file_sha256": V2116_RAW_FILE_BINDINGS[
                "release_attestation.json"
            ][1],
            "attestation_sha256": (
                "7f6d245ebb010237248d51db44fc466a1f4061190d10d8fba5046ba103b7671d"
            ),
            "status": "pass",
        },
        "scientific_launch_input": {
            "path": (
                _FAILED_RAW_ROOT / "scientific_launch_input.json"
            ).as_posix(),
            "file_sha256": V2116_RAW_FILE_BINDINGS[
                "scientific_launch_input.json"
            ][1],
            "launch_input_sha256": (
                "660b549503d4209b7b1ed4e859504a9ecb2f01f6fc6a0abc01a86c058a8f6d00"
            ),
        },
        "acceptance_receipt_present": False,
        "science_reservations": 0,
        "provider_construction": False,
        "provider_calls": 0,
        "scientific_evidence": False,
        "resume_forbidden": True,
        "failure_reclassification_forbidden": True,
    }


def validate_v2117_source_manifest(
    *,
    contract: PilotContract,
    repo_root: str | Path,
    failed_repo_root: str | Path,
    authority_repo_root: str | Path,
) -> dict[str, Any]:
    root = _real_root(repo_root, name="V2.11.7 repository")
    path = root.joinpath(*V2117_SOURCE_MANIFEST_PATH.parts)
    observed = _strict_json(path, name="V2.11.7 source manifest")
    expected = build_v2117_source_manifest(
        contract=contract,
        repo_root=root,
        failed_repo_root=failed_repo_root,
        authority_repo_root=authority_repo_root,
    )
    if observed != expected:
        raise PilotV2117ContinuationError("V2.11.7 source manifest replay drifted")
    boundary = _boundary(contract)
    declared = boundary.get("source_manifest")
    if not isinstance(declared, Mapping):
        raise PilotV2117ContinuationError("contract source-manifest boundary is malformed")
    expected_binding = {
        "path": V2117_SOURCE_MANIFEST_PATH.as_posix(),
        "file_sha256": _file_sha256(path),
        "content_sha256": expected["integrity"]["content_sha256"],
    }
    if {key: declared.get(key) for key in expected_binding} != expected_binding:
        raise PilotV2117ContinuationError("contract source-manifest identity drifted")
    return observed


def parent_budget_debit_for_v2117(contract: PilotContract) -> ParentBudgetDebit:
    if contract.contract_id != V2117_CONTRACT_ID:
        raise PilotV2117ContinuationError("parent debit requires V2.11.7")
    boundary = _boundary(contract)
    declared = boundary.get("parent_budget_debit")
    expected = {
        "parent_contract_sha256": V2116_CONTRACT_SHA256,
        "parent_run_ledger_sha256": V2116_RUN_LEDGER_SHA256,
        "parent_budget_ledger_sha256": V2116_BUDGET_LEDGER_SHA256,
        "stage_bucket": "parent_v2116",
        "cost_usd": V2116_CUMULATIVE_COST_USD,
        "hosted_completions": V2116_CUMULATIVE_COMPLETIONS,
        "storage_bytes": V2116_CUMULATIVE_STORAGE_BYTES,
    }
    if not isinstance(declared, Mapping) or any(
        declared.get(key) != value for key, value in expected.items()
    ):
        raise PilotV2117ContinuationError("V2.11.7 parent budget debit drifted")
    return ParentBudgetDebit(**expected)


def current_authority_path(raw_root: str | Path) -> Path:
    return Path(raw_root) / "parent-import/current_authority/post_gate_authority.json"


def current_projection_path(raw_root: str | Path, model_id: str) -> Path:
    if model_id not in {"gpt52_main", "gpt56_diagnostic"}:
        raise PilotV2117ContinuationError(f"unsupported continuation model {model_id}")
    return Path(raw_root) / f"parent-import/current_authority/{model_id}/projection_p95.json"


def _parent_raw(parent_root: Path) -> Path:
    return parent_root.joinpath(*_PARENT_RAW_ROOT.parts)


def _parent_binding(path: Path, *, root: Path) -> dict[str, Any]:
    value = _strict_json(path, name=f"parent artifact {path.name}")
    content = value.get("integrity", {}).get("content_sha256")
    if content is None:
        content = value.get("receipt_sha256")
    binding = {
        "path": path.relative_to(root).as_posix(),
        "file_sha256": _file_sha256(path),
    }
    if content is not None:
        binding["content_sha256"] = content
    return binding


def _normalize_parent_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    value = _json_copy(spec)
    run_id = str(value.get("run_id", ""))
    prefix = f"{V2115_CONTRACT_ID}--"
    if not run_id.startswith(prefix):
        raise PilotV2117ContinuationError("parent scheduled run id is malformed")
    value["run_id"] = f"{V2117_CONTRACT_ID}--{run_id[len(prefix):]}"
    value["contract_id"] = V2117_CONTRACT_ID
    value["budget_bucket"] = "hosted_v2117"
    return value


def _canonical_remaining_cell_mapping(
    child_contract: PilotContract,
    authority_contract: PilotContract,
) -> dict[str, Any]:
    """Bind every untouched V2.11.5 cell to exactly one V2.11.7 cell."""

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
        len(source_specs) != V2117_REMAINING_CELL_COUNT
        or len(child_specs) != V2117_REMAINING_CELL_COUNT
        or len(child_by_id) != V2117_REMAINING_CELL_COUNT
    ):
        raise PilotV2117ContinuationError(
            "canonical remaining-cell mapping denominator drifted"
        )
    rows: list[dict[str, Any]] = []
    for source_spec in sorted(source_specs, key=lambda spec: spec.run_id):
        source = source_spec.to_dict()
        normalized_child = _normalize_parent_spec(source)
        child = child_by_id.get(normalized_child["run_id"])
        if child != normalized_child:
            raise PilotV2117ContinuationError(
                f"child normalized spec differs for {source_spec.run_id}"
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
    if (
        len({row["source_run_id"] for row in rows}) != len(rows)
        or len({row["child_run_id"] for row in rows}) != len(rows)
        or len({row["logical_cell_sha256"] for row in rows}) != len(rows)
        or {row["child_run_id"] for row in rows} != set(child_by_id)
    ):
        raise PilotV2117ContinuationError(
            "canonical remaining-cell mapping is not one-to-one"
        )
    return {
        "schema_version": "finevo-pilot-v2.11.7-canonical-cell-mapping-v1",
        "row_count": V2117_REMAINING_CELL_COUNT,
        "mapping_sha256": canonical_sha256(rows),
        "rows": rows,
    }


def _failed_state(
    *,
    contract: PilotContract,
    repo_root: Path,
    failed_repo_root: Path,
    authority_repo_root: Path,
) -> dict[str, Any]:
    """Deep-verify the terminal zero-provider V2.11.6 no-go lineage."""

    # Imports remain local so importing this zero-provider module cannot create
    # the orchestrator/provider construction cycle.
    from .pilot_orchestrator import PilotRunLedger, _budget_caps

    manifest = validate_v2117_source_manifest(
        contract=contract,
        repo_root=repo_root,
        failed_repo_root=failed_repo_root,
        authority_repo_root=authority_repo_root,
    )
    if _json_copy(_boundary(contract).get("failed_release_no_go")) != (
        _expected_failed_release_no_go()
    ):
        raise PilotV2117ContinuationError(
            "V2.11.6 failed-release contract boundary drifted"
        )
    failed_contract = load_pilot_contract(
        failed_repo_root / _FAILED_CONTRACT_PATH
    )
    failed_raw = failed_repo_root.joinpath(*_FAILED_RAW_ROOT.parts)
    run_path = failed_raw / "run_ledger.json"
    run_ledger = PilotRunLedger(
        run_path,
        contract_hash=failed_contract.canonical_hash,
        tamper_evident=True,
    )
    run_snapshot = run_ledger.snapshot()
    events = run_snapshot.get("events")
    runs = run_snapshot.get("runs")
    failed_specs = tuple(failed_contract.expand())
    expected_specs = {spec.run_id: spec.to_dict() for spec in failed_specs}
    if (
        _file_sha256(run_path) != V2116_RUN_LEDGER_FILE_SHA256
        or run_snapshot.get("ledger_sha256") != V2116_RUN_LEDGER_SHA256
        or not isinstance(events, list)
        or len(events) != V2116_RUN_EVENT_COUNT
        or events[-1].get("event_sha256") != V2116_RUN_EVENT_HEAD
        or Counter(event.get("event_type") for event in events)
        != Counter({"genesis": 1, "runs_registered": 1, "run_finalized": 87})
        or not isinstance(runs, Mapping)
        or set(runs) != set(expected_specs)
        or len(runs) != V2117_LEDGER_CELL_COUNT
        or any(
            not isinstance(row, Mapping)
            or row.get("spec") != expected_specs[run_id]
            or row.get("status") != "integrity-stopped"
            or row.get("artifact") is not None
            or not isinstance(row.get("failure"), Mapping)
            or row["failure"].get("provider_calls") != 0
            or row["failure"].get("provider_construction") is not False
            or row["failure"].get("message")
            != "V2.11.5 current-release actual debit drifted"
            for run_id, row in runs.items()
        )
    ):
        raise PilotV2117ContinuationError(
            "V2.11.6 terminal run-ledger no-go drifted"
        )

    parent_specs = tuple(failed_contract.expand(stage="parent-import"))
    science_specs = tuple(
        spec
        for stage_id in ("experiment-d", "experiment-b", "cross-model")
        for spec in failed_contract.expand(stage=stage_id)
    )
    if len(parent_specs) != 1 or len(science_specs) != V2117_REMAINING_CELL_COUNT:
        raise PilotV2117ContinuationError(
            "V2.11.6 failed denominator shape drifted"
        )
    parent_run_id = parent_specs[0].run_id
    parent_failure = runs[parent_run_id]["failure"]
    if (
        parent_failure.get("cause_type") != "PilotV2116ContinuationError"
        or parent_failure.get("error_type")
        != "V2116ParentImportIntegrityError"
    ):
        raise PilotV2117ContinuationError(
            "V2.11.6 parent-import failure identity drifted"
        )

    budget_path = failed_raw / "budget_ledger.json"
    stored_budget = _strict_json(
        budget_path, name="V2.11.6 failed budget ledger"
    )
    imported_parent = ParentBudgetDebit(
        parent_contract_sha256=V2115_CONTRACT_SHA256,
        parent_run_ledger_sha256=V2115_RUN_LEDGER_SHA256,
        parent_budget_ledger_sha256=V2115_BUDGET_LEDGER_SHA256,
        stage_bucket="parent_v2115",
        cost_usd=V2115_CUMULATIVE_COST_USD,
        hosted_completions=V2115_CUMULATIVE_COMPLETIONS,
        storage_bytes=V2115_CUMULATIVE_STORAGE_BYTES,
    )
    if (
        imported_parent.record_sha256
        != V2116_IMPORTED_PARENT_DEBIT_RECORD_SHA256
        or stored_budget.get("parent_debit") != imported_parent.to_dict()
    ):
        raise PilotV2117ContinuationError(
            "V2.11.6 imported V2.11.5 debit drifted"
        )
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
    expected_committed = {
        "cost_usd": V2116_CUMULATIVE_COST_USD,
        "completions": V2116_CUMULATIVE_COMPLETIONS,
        "storage_bytes": V2116_CUMULATIVE_STORAGE_BYTES,
        "stage_cost_usd": {
            "hosted_v2116": 0.0,
            "manual_reserve": 0.0,
            "parent_v2115": V2115_CUMULATIVE_COST_USD,
        },
    }
    if (
        _file_sha256(budget_path) != V2116_BUDGET_LEDGER_FILE_SHA256
        or budget.get("ledger_sha256") != V2116_BUDGET_LEDGER_SHA256
        or not isinstance(budget_events, list)
        or len(budget_events) != V2116_BUDGET_EVENT_COUNT
        or budget_events[-1].get("event_sha256") != V2116_BUDGET_EVENT_HEAD
        or Counter(event.get("event_type") for event in budget_events)
        != Counter(
            {
                "genesis": 1,
                "parent_debit_imported": 1,
                "run_reserved": 1,
                "run_finalized": 1,
            }
        )
        or budget.get("committed") != expected_committed
        or budget.get("committed_plus_reserved") != expected_committed
        or not isinstance(budget_runs, Mapping)
        or set(budget_runs) != {parent_run_id}
    ):
        raise PilotV2117ContinuationError(
            "V2.11.6 terminal budget-ledger no-go drifted"
        )
    budget_row = budget_runs[parent_run_id]
    reservation = budget_row.get("reservation")
    actual = budget_row.get("actual")
    failure = budget_row.get("failure")
    if (
        budget_row.get("status") != "integrity-stopped"
        or budget_row.get("stage_bucket") != "parent_v2115"
        or not isinstance(reservation, Mapping)
        or reservation.get("run_id") != parent_run_id
        or reservation.get("stage_bucket") != "parent_v2115"
        or reservation.get("cost_usd") != 0.0
        or reservation.get("completions") != 0
        or reservation.get("basis", {}).get("provider_calls") != 0
        or reservation.get("basis", {}).get("provider_construction") is not False
        or actual
        != {
            "cost_usd": 0.0,
            "completions": 0,
            "storage_bytes": V2116_PARENT_IMPORT_ACTUAL_STORAGE_BYTES,
        }
        or not isinstance(failure, Mapping)
        or failure.get("provider_calls") != 0
        or failure.get("provider_construction") is not False
        or {spec.run_id for spec in science_specs} & set(budget_runs)
    ):
        raise PilotV2117ContinuationError(
            "V2.11.6 no-science-reservation boundary drifted"
        )

    stage_path = failed_raw / "parent-import/stage_receipt.json"
    stage = _strict_json(stage_path, name="V2.11.6 parent-import no-go receipt")
    # Stage receipts use the orchestrator's receipt-hash convention rather
    # than this module's private `_seal` convention.  The exact file hash,
    # declared content hash, ledger bindings, and frozen raw inventory below
    # bind it without reinterpreting that schema.
    if (
        _file_sha256(stage_path)
        != V2116_RAW_FILE_BINDINGS["parent-import/stage_receipt.json"][1]
        or stage.get("integrity", {}).get("content_sha256")
        != V2116_PARENT_IMPORT_RECEIPT_CONTENT_SHA256
        or stage.get("contract_id") != V2116_CONTRACT_ID
        or stage.get("contract_sha256") != V2116_CONTRACT_SHA256
        or stage.get("stage_id") != "parent-import"
        or stage.get("status") != "integrity-stopped"
        or stage.get("go") is not False
        or stage.get("execution_progression_go") is not False
        or stage.get("terminal") is not True
        or stage.get("denominator_terminal") is not True
        or stage.get("scientific_matrix_complete") is not False
        or stage.get("registered_run_count") != 1
        or stage.get("status_counts") != {"integrity-stopped": 1}
        or stage.get("failure") != parent_failure
    ):
        raise PilotV2117ContinuationError(
            "V2.11.6 parent-import terminal receipt drifted"
        )
    acceptance_path = failed_raw / V2117_ACCEPTANCE_FILENAME
    if (
        acceptance_path.exists()
        or any(
            event.get("event_type") == "acceptance_receipt_bound"
            for event in (*events, *budget_events)
        )
    ):
        raise PilotV2117ContinuationError(
            "V2.11.6 unexpectedly contains scientific acceptance"
        )
    return {
        "source_manifest": manifest,
        "failed_contract": failed_contract,
        "failed_raw": failed_raw,
        "run_snapshot": run_snapshot,
        "budget_snapshot": budget,
        "stage_receipt": stage,
        "parent_run_id": parent_run_id,
        "science_run_ids": sorted(spec.run_id for spec in science_specs),
    }


def _v2115_current_actual_decomposition(
    budget_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all 50 V2.11.5 current rows, including operational storage."""

    runs = budget_snapshot.get("runs")
    if not isinstance(runs, Mapping):
        raise PilotV2117ContinuationError(
            "V2.11.5 current-release budget rows are malformed"
        )
    all_current_rows = [
        row
        for row in runs.values()
        if isinstance(row, Mapping) and isinstance(row.get("actual"), Mapping)
    ]
    hosted_rows = [
        row
        for row in all_current_rows
        if row.get("stage_bucket") == "hosted_v2115"
    ]
    operational_rows = [
        row
        for row in all_current_rows
        if row.get("stage_bucket") == "parent_v2114"
    ]

    def actual_sum(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "cost_usd": sum(float(row["actual"]["cost_usd"]) for row in rows),
            "completions": sum(
                int(row["actual"]["completions"]) for row in rows
            ),
            "storage_bytes": sum(
                int(row["actual"]["storage_bytes"]) for row in rows
            ),
        }

    hosted_actual = actual_sum(hosted_rows)
    operational_actual = actual_sum(operational_rows)
    current_actual = actual_sum(all_current_rows)
    if (
        len(all_current_rows) != 50
        or len(hosted_rows) != V2115_HOSTED_RUN_COUNT
        or hosted_actual
        != {
            "cost_usd": V2115_CURRENT_COST_USD,
            "completions": V2115_CURRENT_COMPLETIONS,
            "storage_bytes": V2115_HOSTED_STORAGE_BYTES,
        }
        or len(operational_rows) != V2115_OPERATIONAL_RUN_COUNT
        or operational_actual
        != {
            "cost_usd": 0.0,
            "completions": 0,
            "storage_bytes": V2115_OPERATIONAL_STORAGE_BYTES,
        }
        or current_actual
        != {
            "cost_usd": V2115_CURRENT_COST_USD,
            "completions": V2115_CURRENT_COMPLETIONS,
            "storage_bytes": V2115_CURRENT_STORAGE_BYTES,
        }
        or {row.get("stage_bucket") for row in all_current_rows}
        != {"hosted_v2115", "parent_v2114"}
    ):
        raise PilotV2117ContinuationError(
            "V2.11.5 current-release actual debit decomposition drifted"
        )
    return {
        "aggregation_scope": "all-current-budget-run-rows",
        "hosted_v2115": {
            "row_count": V2115_HOSTED_RUN_COUNT,
            "cost_usd": V2115_CURRENT_COST_USD,
            "hosted_completions": V2115_CURRENT_COMPLETIONS,
            "storage_bytes": V2115_HOSTED_STORAGE_BYTES,
        },
        "operational_parent_v2114": {
            "row_count": V2115_OPERATIONAL_RUN_COUNT,
            "cost_usd": 0.0,
            "hosted_completions": 0,
            "storage_bytes": V2115_OPERATIONAL_STORAGE_BYTES,
        },
        "all_current": {
            "row_count": 50,
            "cost_usd": V2115_CURRENT_COST_USD,
            "hosted_completions": V2115_CURRENT_COMPLETIONS,
            "storage_bytes": V2115_CURRENT_STORAGE_BYTES,
        },
        "inherited_parent": {
            "cost_usd": 19.998220562500006,
            "hosted_completions": 1_004,
            "storage_bytes": 222_048_702,
        },
        "cumulative_v2115": {
            "cost_usd": V2115_CUMULATIVE_COST_USD,
            "hosted_completions": V2115_CUMULATIVE_COMPLETIONS,
            "storage_bytes": V2115_CUMULATIVE_STORAGE_BYTES,
        },
        "observed_storage_difference_bytes": V2115_OPERATIONAL_STORAGE_BYTES,
        "repair_changes_scientific_design": False,
        "scientific_outcomes_inspected_for_repair": False,
    }


def _parent_state(
    *,
    contract: PilotContract,
    repo_root: Path,
    failed_repo_root: Path,
    parent_repo_root: Path,
) -> dict[str, Any]:
    """Deep-verify the immutable V2.11.5 ledgers and required receipts."""

    # Imports are local to avoid an orchestrator/module cycle.
    from .pilot_orchestrator import (
        GitProvenance,
        PilotRunLedger,
        _budget_caps,
        _verify_v2_stage_receipt,
    )
    from .pilot_v2115_acceptance import verify_v2115_scientific_dispatch_acceptance
    from .pilot_v2115_gate import verified_v2115_gate_authority_binding

    manifest = validate_v2117_source_manifest(
        contract=contract,
        repo_root=repo_root,
        failed_repo_root=failed_repo_root,
        authority_repo_root=parent_repo_root,
    )
    parent_contract = load_pilot_contract(parent_repo_root / _PARENT_CONTRACT_PATH)
    parent_raw = _parent_raw(parent_repo_root)
    run_ledger = PilotRunLedger(
        parent_raw / "run_ledger.json",
        contract_hash=parent_contract.canonical_hash,
        tamper_evident=True,
    )
    run_snapshot = run_ledger.snapshot()
    events = run_snapshot.get("events")
    runs = run_snapshot.get("runs")
    counts = Counter(
        row.get("status") for row in runs.values() if isinstance(row, Mapping)
    ) if isinstance(runs, Mapping) else Counter()
    if (
        _file_sha256(parent_raw / "run_ledger.json")
        != _boundary(contract)["parent_terminal_prefix"]["run_ledger"][
            "file_sha256"
        ]
        or
        run_snapshot.get("ledger_sha256") != V2115_RUN_LEDGER_SHA256
        or not isinstance(events, list)
        or len(events) != V2115_RUN_EVENT_COUNT
        or events[-1].get("event_sha256") != V2115_RUN_EVENT_HEAD
        or not isinstance(runs, Mapping)
        or len(runs) != 136
        or counts != Counter({"complete": 47, "failed": 3, "scheduled": 86})
    ):
        raise PilotV2117ContinuationError("V2.11.5 run-ledger terminal prefix drifted")

    stored_budget = _strict_json(parent_raw / "budget_ledger.json", name="parent budget ledger")
    budget_ledger = PilotBudgetLedger(
        parent_raw / "budget_ledger.json",
        contract_hash=parent_contract.canonical_hash,
        caps=_budget_caps(parent_contract),
        tamper_evident=True,
        parent_debit=stored_budget.get("parent_debit"),
    )
    budget = budget_ledger.snapshot()
    budget_events = budget.get("events")
    committed = budget.get("committed")
    if (
        _file_sha256(parent_raw / "budget_ledger.json")
        != _boundary(contract)["parent_terminal_prefix"]["budget_ledger"][
            "file_sha256"
        ]
        or
        budget.get("ledger_sha256") != V2115_BUDGET_LEDGER_SHA256
        or not isinstance(budget_events, list)
        or len(budget_events) != V2115_BUDGET_EVENT_COUNT
        or budget_events[-1].get("event_sha256") != V2115_BUDGET_EVENT_HEAD
        or committed
        != {
            "cost_usd": V2115_CUMULATIVE_COST_USD,
            "completions": V2115_CUMULATIVE_COMPLETIONS,
            "storage_bytes": V2115_CUMULATIVE_STORAGE_BYTES,
            "stage_cost_usd": {
                "hosted_v2115": V2115_CURRENT_COST_USD,
                "manual_reserve": 0.0,
                "parent_v2114": 19.998220562500006,
            },
        }
    ):
        raise PilotV2117ContinuationError("V2.11.5 budget-ledger terminal prefix drifted")
    expected_decomposition = _v2115_current_actual_decomposition(budget)
    if _json_copy(
        _boundary(contract).get("authority_current_actual_decomposition")
    ) != expected_decomposition:
        raise PilotV2117ContinuationError(
            "V2.11.5 current-debit repair boundary drifted"
        )

    release_attestation = _strict_json(
        parent_raw / "release_attestation.json", name="parent release attestation"
    )
    paid = GitProvenance(
        git_tag=V2115_SCIENCE_TAG,
        head_commit=V2115_SCIENCE_COMMIT,
        tag_commit=V2115_SCIENCE_COMMIT,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
        release_attestation=release_attestation,
    )
    stage_receipts: dict[str, Any] = {}
    for stage_id in (
        "parent-import",
        "capability-gate",
        "long-context-preflight",
        "experiment-c",
        "experiment-a",
    ):
        path = parent_raw / stage_id / "stage_receipt.json"
        receipt = _verify_v2_stage_receipt(
            parent_contract,
            stage_id,
            _strict_json(path, name=f"parent {stage_id} receipt"),
            raw_root=parent_raw,
            ledger=run_ledger,
            paid=paid,
            authority_repo_root=parent_repo_root,
        )
        stage_receipts[stage_id] = {
            **_parent_binding(path, root=parent_repo_root),
            "status": receipt["status"],
            "go": receipt["go"],
            "execution_progression_go": receipt["execution_progression_go"],
            "scientific_matrix_complete": receipt["scientific_matrix_complete"],
            "registered_run_count": receipt["registered_run_count"],
            "status_counts": _json_copy(receipt["status_counts"]),
        }
        expected_stage = {
            "path": path.relative_to(parent_repo_root).as_posix(),
            **_json_copy(V2115_STAGE_RECEIPT_BINDINGS[stage_id]),
        }
        if stage_receipts[stage_id] != expected_stage:
            raise PilotV2117ContinuationError(
                f"V2.11.5 {stage_id} receipt identity drifted"
            )
    if (
        stage_receipts["experiment-a"]["status"] != "complete-with-no-go"
        or stage_receipts["experiment-a"]["scientific_matrix_complete"] is not False
        or stage_receipts["experiment-c"]["status"] != "complete-with-no-go"
        or stage_receipts["experiment-c"]["scientific_matrix_complete"] is not True
        or stage_receipts["experiment-a"]["execution_progression_go"] is not True
        or stage_receipts["experiment-c"]["execution_progression_go"] is not True
    ):
        raise PilotV2117ContinuationError("V2.11.5 A/C terminal disposition drifted")

    verify_v2115_scientific_dispatch_acceptance(
        parent_raw / "scientific_dispatch_acceptance.json",
        contract=parent_contract,
        repo_root=parent_repo_root,
        raw_root=parent_raw,
        paid=paid,
        run_ledger=run_ledger,
        budget_ledger=budget_ledger,
    )
    gate_binding = verified_v2115_gate_authority_binding(
        _PARENT_GATE_PATH.as_posix(),
        repo_root=parent_repo_root,
        expected_git_commit=V2115_SCIENCE_COMMIT,
        expected_contract_sha256=V2115_CONTRACT_SHA256,
    )
    expected_gate_boundary = _boundary(contract)["imported_authority"]
    if (
        gate_binding.get("receipt_path")
        != expected_gate_boundary["preflight_authority_path"]
        or gate_binding.get("receipt_file_sha256")
        != expected_gate_boundary["preflight_authority_file_sha256"]
        or gate_binding.get("receipt_content_sha256")
        != expected_gate_boundary["preflight_authority_receipt_sha256"]
    ):
        raise PilotV2117ContinuationError("V2.11.5 gate authority binding drifted")

    parent_scheduled = {
        run_id: row
        for run_id, row in runs.items()
        if isinstance(row, Mapping) and row.get("status") == "scheduled"
    }
    child_specs = tuple(
        spec
        for stage_id in ("experiment-d", "experiment-b", "cross-model")
        for spec in contract.expand(stage=stage_id)
    )
    parent_remaining_specs = tuple(
        spec
        for stage_id in ("experiment-d", "experiment-b", "cross-model")
        for spec in parent_contract.expand(stage=stage_id)
    )
    if set(parent_scheduled) != {spec.run_id for spec in parent_remaining_specs}:
        raise PilotV2117ContinuationError(
            "parent scheduled ledger rows differ from its frozen remaining matrix"
        )
    if any(
        parent_scheduled[spec.run_id].get("spec") != spec.to_dict()
        for spec in parent_remaining_specs
    ):
        raise PilotV2117ContinuationError(
            "parent scheduled ledger spec payload drifted"
        )
    normalized_parent = {
        _normalize_parent_spec(parent_scheduled[spec.run_id]["spec"])["run_id"]: (
            _normalize_parent_spec(parent_scheduled[spec.run_id]["spec"])
        )
        for spec in parent_remaining_specs
    }
    expected_child = {spec.run_id: spec.to_dict() for spec in child_specs}
    if normalized_parent != expected_child or len(expected_child) != V2117_REMAINING_CELL_COUNT:
        raise PilotV2117ContinuationError(
            "V2.11.7 remaining denominator is not 1:1 with parent scheduled specs"
        )
    normalized_source_rows: list[dict[str, Any]] = []
    for spec in parent_remaining_specs:
        row = spec.to_dict()
        row.pop("run_id")
        row.pop("contract_id")
        row["budget_bucket"] = "normalized-hosted-continuation"
        normalized_source_rows.append(row)
    scheduled_mapping_sha256 = canonical_sha256(normalized_source_rows)
    if scheduled_mapping_sha256 != _boundary(contract)["continuation_matrix"][
        "normalized_source_spec_sha256"
    ]:
        raise PilotV2117ContinuationError(
            "parent scheduled-row normalized digest drifted"
        )
    cell_mapping = _canonical_remaining_cell_mapping(contract, parent_contract)
    if cell_mapping["mapping_sha256"] != _boundary(contract)[
        "continuation_matrix"
    ].get("canonical_86_row_mapping_sha256"):
        raise PilotV2117ContinuationError(
            "parent canonical 86-row mapping digest drifted"
        )
    return {
        "source_manifest": manifest,
        "parent_contract": parent_contract,
        "parent_raw": parent_raw,
        "run_snapshot": run_snapshot,
        "budget_snapshot": budget,
        "stage_receipts": stage_receipts,
        "gate_binding": gate_binding,
        "scheduled_mapping_sha256": scheduled_mapping_sha256,
        "canonical_remaining_cell_mapping": cell_mapping,
        "exact_child_spec_mapping_sha256": canonical_sha256(normalized_parent),
        "parent_import_receipt": _strict_json(
            parent_raw / "parent-import/parent_import_receipt.json",
            name="parent import receipt",
        ),
    }


def _build_current_authority(
    *,
    contract: PilotContract,
    raw_root: Path,
    paid: Any,
    parent: Mapping[str, Any],
    parent_import_content_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    gate = parent["gate_binding"]
    source_reservations = gate.get("reservations")
    if not isinstance(source_reservations, Mapping):
        raise PilotV2117ContinuationError("parent p95 authority is malformed")
    numeric: dict[str, dict[str, Any]] = {}
    stable_authorities: dict[str, dict[str, Any]] = {}
    for runtime_model, by_kind in source_reservations.items():
        if not isinstance(by_kind, Mapping) or set(by_kind) != {"action", "semantic"}:
            raise PilotV2117ContinuationError("parent p95 call-kind denominator drifted")
        numeric[str(runtime_model)] = {}
        stable_authorities[str(runtime_model)] = {}
        for call_kind in ("action", "semantic"):
            entry = by_kind[call_kind]
            authority = entry.get("authority")
            reservation = entry.get("reservation")
            if not isinstance(authority, Mapping) or not isinstance(reservation, Mapping):
                raise PilotV2117ContinuationError("parent p95 reservation is malformed")
            stable = {
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
            numeric[str(runtime_model)][call_kind] = _json_copy(reservation)
            stable_authorities[str(runtime_model)][call_kind] = stable
    authority = _seal(
        {
            "schema_version": V2117_CURRENT_AUTHORITY_SCHEMA_VERSION,
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "release": {"git_tag": paid.git_tag, "git_commit": paid.head_commit},
            "authority_release": {
                "contract_id": V2115_CONTRACT_ID,
                "contract_sha256": V2115_CONTRACT_SHA256,
                "git_tag": V2115_SCIENCE_TAG,
                "git_commit": V2115_SCIENCE_COMMIT,
                "source_gate": {
                    "path": gate["receipt_path"],
                    "file_sha256": gate["receipt_file_sha256"],
                    "content_sha256": gate["receipt_content_sha256"],
                },
            },
            "parent_import_content_sha256": parent_import_content_sha256,
            "reservations": numeric,
            "stable_source_authorities": stable_authorities,
            "provider_boundary": {
                "provider_calls": 0,
                "hosted_provider_calls": 0,
                "hosted_cost_usd": 0.0,
                "provider_construction": False,
            },
            "scientific_evidence": False,
            "claim_boundary": (
                "Current-release dispatch-budget authority only; no decoded "
                "completion or A/C effect row is imported."
            ),
        }
    )
    authority_path = current_authority_path(raw_root)
    _write_once(authority_path, authority)
    authority_file_hash = _file_sha256(authority_path)
    authority_content_hash = authority["integrity"]["content_sha256"]
    projections: dict[str, dict[str, Any]] = {}
    runtime_by_profile = {
        str(profile_id): (
            f"{profile.transport}/{profile.served_model}"
            if profile.transport != "openrouter"
            else f"thirdparty/{profile.served_model}"
        )
        for profile_id, profile in contract.provider_profiles.items()
    }
    for model_id in ("gpt52_main", "gpt56_diagnostic"):
        runtime_model = runtime_by_profile[model_id]
        rows = numeric.get(runtime_model)
        if not isinstance(rows, Mapping):
            raise PilotV2117ContinuationError(
                f"current p95 authority lacks {model_id}/{runtime_model}"
            )
        profile = contract.provider_profiles[model_id]
        projection = _seal(
            {
                "schema_version": V2117_CURRENT_PROJECTION_SCHEMA_VERSION,
                "model_id": model_id,
                "runtime_model": runtime_model,
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
                    "authority_file_sha256": authority_file_hash,
                    "authority_content_sha256": authority_content_hash,
                    "parent_import_content_sha256": parent_import_content_sha256,
                },
                "provider_calls": 0,
                "provider_construction": False,
                "scientific_evidence": False,
            }
        )
        path = current_projection_path(raw_root, model_id)
        _write_once(path, projection)
        projections[model_id] = projection
    return authority, projections


def _verify_seal(value: Mapping[str, Any], *, name: str) -> None:
    integrity = value.get("integrity")
    if (
        not isinstance(integrity, Mapping)
        or set(integrity) != {"canonicalization", "content_sha256"}
        or integrity.get("canonicalization") != _CANONICALIZATION
        or integrity.get("content_sha256") != _content_sha256(value)
    ):
        raise PilotV2117ContinuationError(f"{name} self-hash mismatch")


def _release_binding(contract: PilotContract, paid: Any) -> dict[str, Any]:
    expected_tag = str(contract.implementation["required_git_tag"])
    if (
        contract.contract_id != V2117_CONTRACT_ID
        or expected_tag != V2117_SCIENCE_TAG
        or paid.git_tag != expected_tag
        or not _COMMIT.fullmatch(str(paid.head_commit))
        or paid.tag_commit != paid.head_commit
        or paid.tag_object_type != "tag"
        or paid.worktree_clean is not True
    ):
        raise PilotV2117ContinuationError(
            "V2.11.7 continuation requires its exact clean annotated release"
        )
    return {
        "git_tag": paid.git_tag,
        "git_commit": paid.head_commit,
        "tag_object_type": paid.tag_object_type,
        "worktree_clean": paid.worktree_clean,
    }


def _tracked_source_manifest(
    contract: PilotContract,
    *,
    repo_root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    root = _real_root(repo_root, name="V2.11.7 repository")
    path = root.joinpath(*V2117_SOURCE_MANIFEST_PATH.parts)
    value = _strict_json(path, name="V2.11.7 source manifest")
    _verify_seal(value, name="V2.11.7 source manifest")
    declared = _boundary(contract).get("source_manifest")
    if (
        not isinstance(declared, Mapping)
        or declared.get("path") != V2117_SOURCE_MANIFEST_PATH.as_posix()
        or declared.get("schema_version") != V2117_SOURCE_MANIFEST_SCHEMA_VERSION
        or declared.get("file_sha256") != _file_sha256(path)
        or declared.get("content_sha256")
        != value["integrity"]["content_sha256"]
        or value.get("schema_version") != V2117_SOURCE_MANIFEST_SCHEMA_VERSION
        or value.get("contract_id") != contract.contract_id
        or value.get("release_tag") != V2117_SCIENCE_TAG
    ):
        raise PilotV2117ContinuationError(
            "V2.11.7 tracked source-manifest binding drifted"
        )
    return path, value


def _capability_summary(wrapper: Mapping[str, Any], *, model_id: str) -> dict[str, Any]:
    capability = wrapper.get("capability")
    integrity = wrapper.get("integrity")
    if not isinstance(capability, Mapping) or not isinstance(integrity, Mapping):
        raise PilotV2117ContinuationError("parent capability wrapper is malformed")
    expected_hash = V2115_CAPABILITY_CONTENT_SHA256[model_id]
    if integrity.get("content_sha256") != expected_hash:
        raise PilotV2117ContinuationError("parent capability wrapper identity drifted")
    result = {
        "model_id": model_id,
        "runtime_model": capability.get("runtime_model"),
        "requested_model": capability.get("requested_model"),
        "capability_pass": capability.get("capability_pass"),
        "interface_pass": capability.get("interface_pass"),
        "parse_failure_count": capability.get("parse_failure_count"),
        "provider_failure_count": capability.get("provider_failure_count"),
        "category_totals": _json_copy(capability.get("category_totals")),
        "source_wrapper_content_sha256": expected_hash,
    }
    if (
        result["capability_pass"] is not True
        or result["interface_pass"] is not True
        or result["parse_failure_count"] != 0
        or result["provider_failure_count"] != 0
    ):
        raise PilotV2117ContinuationError(
            f"parent capability/interface authority is no-go for {model_id}"
        )
    return result


def _dispatch_authority_source(parent: Mapping[str, Any]) -> dict[str, Any]:
    gate = parent.get("gate_binding")
    if not isinstance(gate, Mapping):
        raise PilotV2117ContinuationError("parent gate binding is malformed")
    rows = gate.get("reservations")
    if not isinstance(rows, Mapping):
        raise PilotV2117ContinuationError("parent gate reservations are malformed")
    numeric: dict[str, dict[str, Any]] = {}
    stable: dict[str, dict[str, Any]] = {}
    for runtime_model, by_kind in rows.items():
        if not isinstance(by_kind, Mapping) or set(by_kind) != {"action", "semantic"}:
            raise PilotV2117ContinuationError("parent gate call-kind set drifted")
        numeric[str(runtime_model)] = {}
        stable[str(runtime_model)] = {}
        for kind in ("action", "semantic"):
            entry = by_kind[kind]
            if not isinstance(entry, Mapping) or set(entry) != {"authority", "reservation"}:
                raise PilotV2117ContinuationError("parent gate reservation row drifted")
            authority = entry["authority"]
            if not isinstance(authority, Mapping):
                raise PilotV2117ContinuationError("parent authority row is malformed")
            numeric[str(runtime_model)][kind] = _json_copy(entry["reservation"])
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
    result = {
        "source_gate": {
            "path": gate["receipt_path"],
            "file_sha256": gate["receipt_file_sha256"],
            "content_sha256": gate["receipt_content_sha256"],
        },
        "reservations": numeric,
        "stable_source_authorities": stable,
        "reservation_set_sha256": canonical_sha256(numeric),
        "stable_authority_set_sha256": canonical_sha256(stable),
    }
    if (
        result["reservation_set_sha256"] != V2115_RESERVATION_SET_SHA256
        or result["stable_authority_set_sha256"]
        != V2115_STABLE_AUTHORITY_SET_SHA256
    ):
        raise PilotV2117ContinuationError("parent p95 authority set drifted")
    return result


def build_v2117_parent_import_receipt(
    *,
    contract: PilotContract,
    repo_root: str | Path,
    raw_root: str | Path,
    failed_repo_root: str | Path,
    authority_repo_root: str | Path,
    paid: Any,
) -> dict[str, Any]:
    """Bind V2.11.6 no-go and import V2.11.5 authority with zero calls."""

    require_v2117_provider_keys_absent()
    repository = _real_root(repo_root, name="V2.11.7 repository")
    raw = _real_root(raw_root, name="V2.11.7 raw root")
    if raw != repository.joinpath(*V2117_RAW_ROOT.parts):
        raise PilotV2117ContinuationError("V2.11.7 raw namespace drifted")
    release = _release_binding(contract, paid)
    failed_root = _real_root(
        failed_repo_root, name="V2.11.6 failed repository"
    )
    authority_root = _real_root(
        authority_repo_root, name="V2.11.5 authority repository"
    )
    failed = _failed_state(
        contract=contract,
        repo_root=repository,
        failed_repo_root=failed_root,
        authority_repo_root=authority_root,
    )
    parent = _parent_state(
        contract=contract,
        repo_root=repository,
        failed_repo_root=failed_root,
        parent_repo_root=authority_root,
    )
    source_path, source_manifest = _tracked_source_manifest(
        contract, repo_root=repository
    )
    parent_import = parent["parent_import_receipt"]
    if (
        parent_import.get("integrity", {}).get("content_sha256")
        != V2115_PARENT_IMPORT_CONTENT_SHA256
    ):
        raise PilotV2117ContinuationError("parent import authority identity drifted")
    calibration = parent_import.get("calibration_wrapper")
    if (
        not isinstance(calibration, Mapping)
        or calibration.get("integrity", {}).get("content_sha256")
        != V2115_CALIBRATION_CONTENT_SHA256
    ):
        raise PilotV2117ContinuationError("parent calibration authority drifted")
    capabilities = parent_import.get("capability_wrappers")
    if not isinstance(capabilities, Mapping) or set(capabilities) != {
        "gpt52_main",
        "gpt56_diagnostic",
    }:
        raise PilotV2117ContinuationError("parent capability denominator drifted")
    capability_summaries = {
        model_id: _capability_summary(capabilities[model_id], model_id=model_id)
        for model_id in sorted(capabilities)
    }
    dispatch_source = _dispatch_authority_source(parent)
    receipt = _seal(
        {
            "schema_version": V2117_PARENT_IMPORT_SCHEMA_VERSION,
            "status": "complete",
            "go": True,
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "release": release,
            "source_manifest": {
                "path": V2117_SOURCE_MANIFEST_PATH.as_posix(),
                "file_sha256": _file_sha256(source_path),
                "content_sha256": source_manifest["integrity"]["content_sha256"],
            },
            "failed_release_no_go": _json_copy(
                _boundary(contract)["failed_release_no_go"]
            ),
            "failed_terminal_no_go": {
                "registered_rows": V2117_LEDGER_CELL_COUNT,
                "integrity_stopped_rows": V2117_LEDGER_CELL_COUNT,
                "provider_calls": 0,
                "provider_construction": False,
                "scientific_acceptance_present": False,
                "science_budget_reservation_count": 0,
                "parent_import_stage_receipt": _parent_binding(
                    failed["failed_raw"] / "parent-import/stage_receipt.json",
                    root=failed_root,
                ),
            },
            "authority_release": _json_copy(
                _boundary(contract)["parent_release"]
            ),
            "authority_terminal_prefix": _json_copy(
                _boundary(contract)["parent_terminal_prefix"]
            ),
            "authority_current_actual_decomposition": _json_copy(
                _boundary(contract)["authority_current_actual_decomposition"]
            ),
            "authority_stage_receipts": _json_copy(parent["stage_receipts"]),
            "authority_import": {
                "path": (
                    _AUTHORITY_RAW_ROOT
                    / "parent-import/parent_import_receipt.json"
                ).as_posix(),
                "file_sha256": V2115_PARENT_IMPORT_FILE_SHA256,
                "content_sha256": V2115_PARENT_IMPORT_CONTENT_SHA256,
            },
            "calibration_wrapper": _json_copy(calibration),
            "capability_authority": capability_summaries,
            "dispatch_authority_source": dispatch_source,
            "canonical_remaining_cell_mapping": _json_copy(
                parent["canonical_remaining_cell_mapping"]
            ),
            "denominator_continuation": {
                "failed_registered_rows": V2117_LEDGER_CELL_COUNT,
                "failed_integrity_stopped_rows": V2117_LEDGER_CELL_COUNT,
                "failed_rows_reclassified_or_redispatched": 0,
                "authority_registered_rows": 136,
                "authority_terminal_rows": 50,
                "authority_scheduled_rows": 86,
                "child_registered_rows": V2117_LEDGER_CELL_COUNT,
                "child_operational_rows": 1,
                "child_scientific_rows": V2117_REMAINING_CELL_COUNT,
                "source_to_child_mapping": "normalized-one-to-one",
                "normalized_mapping_sha256": parent["scheduled_mapping_sha256"],
                "authority_a_c_rows_copied_to_child": 0,
                "authority_a_c_redispatched": False,
            },
            "cumulative_parent_budget_debit": (
                parent_budget_debit_for_v2117(contract).to_dict()
            ),
            "import_policy": {
                "provider_construction": False,
                "provider_calls": 0,
                "hosted_provider_calls": 0,
                "hosted_cost_usd": 0.0,
                "decoded_completion_reuse": False,
                "imported_effect_cells": 0,
                "imported_scientific_run_summaries": 0,
                "failed_raw_tree_copied": False,
                "authority_raw_tree_copied": False,
                "validation_before_provider_construction": True,
            },
            "scientific_evidence": False,
            "claim_boundary": (
                "Immutable V2.11.6 integrity no-go lineage plus V2.11.5 "
                "non-effect authority import only. No failed or A/C terminal "
                "cell is reclassified, copied, or redispatched."
            ),
        }
    )
    receipt_path = raw / "parent-import/parent_import_receipt.json"
    _write_once(receipt_path, receipt)
    _build_current_authority(
        contract=contract,
        raw_root=raw,
        paid=paid,
        parent=parent,
        parent_import_content_sha256=receipt["integrity"]["content_sha256"],
    )
    return receipt


def verify_v2117_parent_import_receipt(
    receipt_path: str | Path,
    *,
    contract: PilotContract,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: Any,
) -> dict[str, Any]:
    repository = _real_root(repo_root, name="V2.11.7 repository")
    raw = _real_root(raw_root, name="V2.11.7 raw root")
    path = Path(receipt_path).absolute()
    if (
        raw != repository.joinpath(*V2117_RAW_ROOT.parts)
        or path != raw / "parent-import/parent_import_receipt.json"
    ):
        raise PilotV2117ContinuationError("parent-import receipt path drifted")
    receipt = _strict_json(path, name="V2.11.7 parent-import receipt")
    _verify_seal(receipt, name="V2.11.7 parent-import receipt")
    release = _release_binding(contract, paid)
    source_path, source = _tracked_source_manifest(contract, repo_root=repository)
    boundary = _boundary(contract)
    calibration = receipt.get("calibration_wrapper")
    capabilities = receipt.get("capability_authority")
    dispatch = receipt.get("dispatch_authority_source")
    cell_mapping = receipt.get("canonical_remaining_cell_mapping")
    failed_no_go = receipt.get("failed_terminal_no_go")
    authority_stages = receipt.get("authority_stage_receipts")
    denominator = receipt.get("denominator_continuation")
    policy = receipt.get("import_policy")
    if (
        receipt.get("schema_version") != V2117_PARENT_IMPORT_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("go") is not True
        or receipt.get("contract_id") != contract.contract_id
        or receipt.get("contract_sha256") != contract.canonical_hash
        or receipt.get("release") != release
        or receipt.get("source_manifest")
        != {
            "path": V2117_SOURCE_MANIFEST_PATH.as_posix(),
            "file_sha256": _file_sha256(source_path),
            "content_sha256": source["integrity"]["content_sha256"],
        }
        or receipt.get("failed_release_no_go")
        != boundary["failed_release_no_go"]
        or failed_no_go
        != {
            "registered_rows": V2117_LEDGER_CELL_COUNT,
            "integrity_stopped_rows": V2117_LEDGER_CELL_COUNT,
            "provider_calls": 0,
            "provider_construction": False,
            "scientific_acceptance_present": False,
            "science_budget_reservation_count": 0,
            "parent_import_stage_receipt": {
                "path": (
                    _FAILED_RAW_ROOT / "parent-import/stage_receipt.json"
                ).as_posix(),
                "file_sha256": V2116_RAW_FILE_BINDINGS[
                    "parent-import/stage_receipt.json"
                ][1],
                "content_sha256": V2116_PARENT_IMPORT_RECEIPT_CONTENT_SHA256,
            },
        }
        or receipt.get("authority_release") != boundary["parent_release"]
        or receipt.get("authority_terminal_prefix")
        != boundary["parent_terminal_prefix"]
        or receipt.get("authority_current_actual_decomposition")
        != boundary["authority_current_actual_decomposition"]
        or not isinstance(authority_stages, Mapping)
        or set(authority_stages) != set(V2115_STAGE_RECEIPT_BINDINGS)
        or any(
            authority_stages[stage_id]
            != {
                "path": (
                    _AUTHORITY_RAW_ROOT / stage_id / "stage_receipt.json"
                ).as_posix(),
                **_json_copy(V2115_STAGE_RECEIPT_BINDINGS[stage_id]),
            }
            for stage_id in V2115_STAGE_RECEIPT_BINDINGS
        )
        or any(
            authority_stages[stage_id]
            != {
                **_json_copy(boundary["parent_stage_receipts"][stage_id]),
            }
            for stage_id in ("experiment-a", "experiment-c")
        )
        or receipt.get("authority_import")
        != {
            "path": (
                _AUTHORITY_RAW_ROOT / "parent-import/parent_import_receipt.json"
            ).as_posix(),
            "file_sha256": V2115_PARENT_IMPORT_FILE_SHA256,
            "content_sha256": V2115_PARENT_IMPORT_CONTENT_SHA256,
        }
        or not isinstance(calibration, Mapping)
        or calibration.get("integrity", {}).get("content_sha256")
        != V2115_CALIBRATION_CONTENT_SHA256
        or not isinstance(capabilities, Mapping)
        or set(capabilities) != {"gpt52_main", "gpt56_diagnostic"}
        or any(
            capabilities[model_id].get("source_wrapper_content_sha256")
            != V2115_CAPABILITY_CONTENT_SHA256[model_id]
            or capabilities[model_id].get("capability_pass") is not True
            or capabilities[model_id].get("interface_pass") is not True
            for model_id in capabilities
        )
        or not isinstance(dispatch, Mapping)
        or dispatch.get("reservation_set_sha256")
        != V2115_RESERVATION_SET_SHA256
        or dispatch.get("reservation_set_sha256")
        != canonical_sha256(dispatch.get("reservations"))
        or dispatch.get("stable_authority_set_sha256")
        != V2115_STABLE_AUTHORITY_SET_SHA256
        or dispatch.get("stable_authority_set_sha256")
        != canonical_sha256(dispatch.get("stable_source_authorities"))
        or dispatch.get("source_gate")
        != {
            "path": boundary["imported_authority"]["preflight_authority_path"],
            "file_sha256": boundary["imported_authority"][
                "preflight_authority_file_sha256"
            ],
            "content_sha256": boundary["imported_authority"][
                "preflight_authority_receipt_sha256"
            ],
        }
        or cell_mapping != source.get("canonical_remaining_cell_mapping")
        or not isinstance(cell_mapping, Mapping)
        or cell_mapping.get("schema_version")
        != "finevo-pilot-v2.11.7-canonical-cell-mapping-v1"
        or cell_mapping.get("row_count") != V2117_REMAINING_CELL_COUNT
        or cell_mapping.get("mapping_sha256")
        != boundary["continuation_matrix"].get(
            "canonical_86_row_mapping_sha256"
        )
        or not isinstance(cell_mapping.get("rows"), list)
        or canonical_sha256(cell_mapping["rows"])
        != cell_mapping.get("mapping_sha256")
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "source_run_id",
                "child_run_id",
                "logical_cell_sha256",
                "source_spec_sha256",
                "child_spec_sha256",
                "normalized_spec",
            }
            for row in cell_mapping["rows"]
        )
        or any(
            not str(row["source_run_id"]).startswith(
                f"{V2115_CONTRACT_ID}--"
            )
            or not str(row["child_run_id"]).startswith(
                f"{V2117_CONTRACT_ID}--"
            )
            or not _SHA256.fullmatch(str(row["logical_cell_sha256"]))
            or not _SHA256.fullmatch(str(row["source_spec_sha256"]))
            or not _SHA256.fullmatch(str(row["child_spec_sha256"]))
            or canonical_sha256(row["normalized_spec"])
            != row["logical_cell_sha256"]
            for row in cell_mapping["rows"]
        )
        or len({row.get("source_run_id") for row in cell_mapping["rows"]})
        != V2117_REMAINING_CELL_COUNT
        or len({row.get("child_run_id") for row in cell_mapping["rows"]})
        != V2117_REMAINING_CELL_COUNT
        or len({row.get("logical_cell_sha256") for row in cell_mapping["rows"]})
        != V2117_REMAINING_CELL_COUNT
        or denominator
        != {
            "failed_registered_rows": V2117_LEDGER_CELL_COUNT,
            "failed_integrity_stopped_rows": V2117_LEDGER_CELL_COUNT,
            "failed_rows_reclassified_or_redispatched": 0,
            "authority_registered_rows": 136,
            "authority_terminal_rows": 50,
            "authority_scheduled_rows": 86,
            "child_registered_rows": V2117_LEDGER_CELL_COUNT,
            "child_operational_rows": 1,
            "child_scientific_rows": V2117_REMAINING_CELL_COUNT,
            "source_to_child_mapping": "normalized-one-to-one",
            "normalized_mapping_sha256": boundary["continuation_matrix"][
                "normalized_source_spec_sha256"
            ],
            "authority_a_c_rows_copied_to_child": 0,
            "authority_a_c_redispatched": False,
        }
        or policy
        != {
            "provider_construction": False,
            "provider_calls": 0,
            "hosted_provider_calls": 0,
            "hosted_cost_usd": 0.0,
            "decoded_completion_reuse": False,
            "imported_effect_cells": 0,
            "imported_scientific_run_summaries": 0,
            "failed_raw_tree_copied": False,
            "authority_raw_tree_copied": False,
            "validation_before_provider_construction": True,
        }
        or receipt.get("scientific_evidence") is not False
    ):
        raise PilotV2117ContinuationError("V2.11.7 parent-import receipt drifted")
    expected_debit = parent_budget_debit_for_v2117(contract).to_dict()
    if receipt.get("cumulative_parent_budget_debit") != expected_debit:
        raise PilotV2117ContinuationError("parent debit receipt binding drifted")
    return receipt


def verify_v2117_current_authority(
    *,
    contract: PilotContract,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: Any,
) -> dict[str, Any]:
    repository = _real_root(repo_root, name="V2.11.7 repository")
    raw = _real_root(raw_root, name="V2.11.7 raw root")
    parent = verify_v2117_parent_import_receipt(
        raw / "parent-import/parent_import_receipt.json",
        contract=contract,
        repo_root=repository,
        raw_root=raw,
        paid=paid,
    )
    path = current_authority_path(raw)
    authority = _strict_json(path, name="V2.11.7 current p95 authority")
    _verify_seal(authority, name="V2.11.7 current p95 authority")
    dispatch = parent["dispatch_authority_source"]
    release = _release_binding(contract, paid)
    if (
        authority.get("schema_version") != V2117_CURRENT_AUTHORITY_SCHEMA_VERSION
        or authority.get("contract_id") != contract.contract_id
        or authority.get("contract_sha256") != contract.canonical_hash
        or authority.get("release")
        != {"git_tag": release["git_tag"], "git_commit": release["git_commit"]}
        or authority.get("authority_release")
        != {
            "contract_id": V2115_CONTRACT_ID,
            "contract_sha256": V2115_CONTRACT_SHA256,
            "git_tag": V2115_SCIENCE_TAG,
            "git_commit": V2115_SCIENCE_COMMIT,
            "source_gate": dispatch["source_gate"],
        }
        or authority.get("parent_import_content_sha256")
        != parent["integrity"]["content_sha256"]
        or authority.get("reservations") != dispatch["reservations"]
        or authority.get("stable_source_authorities")
        != dispatch["stable_source_authorities"]
        or authority.get("provider_boundary")
        != {
            "provider_calls": 0,
            "hosted_provider_calls": 0,
            "hosted_cost_usd": 0.0,
            "provider_construction": False,
        }
        or authority.get("scientific_evidence") is not False
    ):
        raise PilotV2117ContinuationError("V2.11.7 current p95 authority drifted")
    return authority


def verified_v2117_projection(
    contract: PilotContract,
    model_id: str,
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: Any,
) -> tuple[dict[str, Any], Path]:
    authority = verify_v2117_current_authority(
        contract=contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
    )
    raw = _real_root(raw_root, name="V2.11.7 raw root")
    path = current_projection_path(raw, model_id)
    value = _strict_json(path, name=f"V2.11.7 {model_id} p95 projection")
    _verify_seal(value, name=f"V2.11.7 {model_id} p95 projection")
    profile = contract.provider_profiles[model_id]
    runtime_model = (
        f"{profile.transport}/{profile.served_model}"
        if profile.transport != "openrouter"
        else f"thirdparty/{profile.served_model}"
    )
    rows = authority["reservations"].get(runtime_model)
    expected_projection = {
        f"{profile.served_model}::{kind}": rows[kind]
        for kind in ("action", "semantic")
    } if isinstance(rows, Mapping) else None
    bindings = value.get("bindings")
    if (
        value.get("schema_version") != V2117_CURRENT_PROJECTION_SCHEMA_VERSION
        or value.get("model_id") != model_id
        or value.get("runtime_model") != runtime_model
        or value.get("served_model") != profile.served_model
        or value.get("projection") != expected_projection
        or bindings
        != {
            "contract_sha256": contract.canonical_hash,
            "git_tag": paid.git_tag,
            "git_commit": paid.head_commit,
            "authority_path": _CURRENT_AUTHORITY_PATH.as_posix(),
            "authority_file_sha256": _file_sha256(
                current_authority_path(raw)
            ),
            "authority_content_sha256": authority["integrity"]["content_sha256"],
            "parent_import_content_sha256": authority[
                "parent_import_content_sha256"
            ],
        }
        or value.get("provider_calls") != 0
        or value.get("provider_construction") is not False
        or value.get("scientific_evidence") is not False
    ):
        raise PilotV2117ContinuationError(
            f"V2.11.7 {model_id} current p95 projection drifted"
        )
    return value, path


def verified_v2117_calibration(
    contract: PilotContract,
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: Any,
) -> dict[str, Any]:
    receipt = verify_v2117_parent_import_receipt(
        Path(raw_root) / "parent-import/parent_import_receipt.json",
        contract=contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
    )
    wrapper = receipt["calibration_wrapper"]
    calibration = wrapper.get("calibration")
    if not isinstance(calibration, Mapping):
        raise PilotV2117ContinuationError("V2.11.7 calibration wrapper is malformed")
    selected = calibration.get("selected_utility_profile")
    threshold = calibration.get("stage0_absolute_flow_utility_threshold")
    if (
        calibration.get("q_ref") != V2115_Q_REF
        or not isinstance(selected, Mapping)
        or selected.get("profile_id") != "nu-0.5"
        or selected.get("rho") != 1.0
        or selected.get("labor_weight") != 2.0
        or selected.get("inverse_frisch") != 0.5
        or selected.get("consumption_scale") != V2115_Q_REF
        or selected.get("discount_factor") != 0.99
        or not isinstance(threshold, Mapping)
        or threshold.get("value") != V2115_ABSOLUTE_FLOW_UTILITY_THRESHOLD
        or threshold.get("treatment_outcomes_inspected") is not False
        or wrapper.get("provider_construction_during_import") is not False
        or wrapper.get("provider_calls_during_import") != 0
        or wrapper.get("imported_effect_cells") != 0
        or wrapper.get("scientific_evidence") is not False
    ):
        raise PilotV2117ContinuationError("V2.11.7 calibration authority drifted")
    return {
        "receipt": receipt,
        "wrapper": wrapper,
        "q_ref": V2115_Q_REF,
        "selected_profile_id": "nu-0.5",
        "selected_utility": {
            key: value for key, value in selected.items() if key != "profile_id"
        },
        "absolute_flow_utility_threshold": _json_copy(threshold),
    }


def verified_v2117_observed_p95_authority_binding(
    receipt_path: str | Path,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
    expected_contract_sha256: str,
) -> dict[str, Any]:
    """Return runner-ready rows from the current continuation authority."""

    repository = _real_root(repo_root, name="V2.11.7 repository")
    contract_path = repository / "experiments/pilot_v2_11_7.yaml"
    contract = load_pilot_contract(contract_path)
    if (
        contract.contract_id != V2117_CONTRACT_ID
        or contract.canonical_hash != expected_contract_sha256
    ):
        raise PilotV2117ContinuationError("V2.11.7 contract identity drifted")
    # This verifier is called from the generic runner adapter, which has
    # already checked the paid git commit.  Construct the minimal immutable
    # provenance view needed by the same current-authority verifier.
    class _Paid:
        pass

    paid = _Paid()
    paid.git_tag = V2117_SCIENCE_TAG
    paid.head_commit = expected_git_commit
    paid.tag_commit = expected_git_commit
    paid.tag_object_type = "tag"
    paid.worktree_clean = True

    raw = repository.joinpath(*V2117_RAW_ROOT.parts)
    expected_path = current_authority_path(raw)
    requested = Path(receipt_path)
    if not requested.is_absolute():
        requested = repository.joinpath(*PurePosixPath(str(receipt_path)).parts)
    if requested.absolute() != expected_path:
        raise PilotV2117ContinuationError("V2.11.7 authority path drifted")
    authority = verify_v2117_current_authority(
        contract=contract,
        repo_root=repository,
        raw_root=raw,
        paid=paid,
    )
    receipt_file_hash = _file_sha256(expected_path)
    receipt_content_hash = authority["integrity"]["content_sha256"]
    reservations: dict[str, dict[str, Any]] = {}
    for runtime_model, numeric in authority["reservations"].items():
        reservations[runtime_model] = {}
        for kind in ("action", "semantic"):
            stable = authority["stable_source_authorities"][runtime_model][kind]
            current_authority = {
                **_json_copy(stable),
                "pilot_contract_hash": contract.canonical_hash,
                "pilot_tag": V2117_SCIENCE_TAG,
                "source_projection_schema_version": (
                    "finevo-pilot-projection-p95-v1"
                ),
                "source_projection_file_sha256": receipt_file_hash,
                "source_projection_content_sha256": receipt_content_hash,
                "source_authority_receipt_path": _CURRENT_AUTHORITY_PATH.as_posix(),
                "source_authority_receipt_file_sha256": receipt_file_hash,
                "source_authority_receipt_content_sha256": receipt_content_hash,
                "source_release_commit": expected_git_commit,
            }
            reservations[runtime_model][kind] = {
                "authority": current_authority,
                "reservation": _json_copy(numeric[kind]),
            }
    return {
        "receipt_path": _CURRENT_AUTHORITY_PATH.as_posix(),
        "receipt_file_sha256": receipt_file_hash,
        "receipt_content_sha256": receipt_content_hash,
        "git_commit": expected_git_commit,
        "reservations": reservations,
    }


def runner_reservations_for_v2117(
    contract: PilotContract,
    model_id: str,
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: Any,
) -> dict[str, dict[str, Any]]:
    projection, _ = verified_v2117_projection(
        contract,
        model_id,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
    )
    binding = verified_v2117_observed_p95_authority_binding(
        _CURRENT_AUTHORITY_PATH.as_posix(),
        repo_root=repo_root,
        expected_git_commit=paid.head_commit,
        expected_contract_sha256=contract.canonical_hash,
    )
    runtime_model = projection["runtime_model"]
    selected = binding["reservations"].get(runtime_model)
    if not isinstance(selected, Mapping) or set(selected) != {"action", "semantic"}:
        raise PilotV2117ContinuationError("V2.11.7 runner authority denominator drifted")
    for kind in ("action", "semantic"):
        key = f"{projection['served_model']}::{kind}"
        if selected[kind]["reservation"] != projection["projection"][key]:
            raise PilotV2117ContinuationError("V2.11.7 runner/projection drifted")
    return {runtime_model: _json_copy(selected)}


def _ledger_prefix(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    events = snapshot.get("events")
    runs = snapshot.get("runs")
    if not isinstance(events, list) or not events or not isinstance(runs, Mapping):
        raise PilotV2117ContinuationError("acceptance ledger prefix is malformed")
    return {
        "event_count": len(events),
        "event_chain_head": events[-1]["event_sha256"],
        "ledger_sha256": snapshot["ledger_sha256"],
        "runs_sha256": canonical_sha256(runs),
    }


def require_v2117_provider_keys_absent() -> None:
    present = [name for name in _PROVIDER_KEY_ENV_NAMES if os.environ.get(name)]
    if present:
        raise PilotV2117ContinuationError(
            "acceptance must run before provider credentials are loaded; present="
            + ",".join(present)
        )


@contextmanager
def _acceptance_provider_sentinels() -> Any:
    """Make accidental provider/catalog construction fail before side effects."""

    from . import pilot_orchestrator as orch

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise PilotV2117ContinuationError(
            "provider/catalog construction is forbidden during acceptance"
        )

    with ExitStack() as stack:
        for name in (
            "_provider_for_profile",
            "create_llm_provider",
            "MultiModelLLM",
            "validate_live_provider_catalog",
        ):
            if hasattr(orch, name):
                stack.enter_context(mock.patch.object(orch, name, forbidden))
        for module, names in (
            (canonical_llm_providers, ("create_llm_provider", "MultiModelLLM")),
            (canonical_provider_catalog, ("validate_live_provider_catalog",)),
        ):
            for name in names:
                if hasattr(module, name):
                    stack.enter_context(mock.patch.object(module, name, forbidden))
        yield


def _expected_pre_science_paths(contract: PilotContract) -> tuple[set[str], set[str]]:
    parent_specs = tuple(contract.expand(stage="parent-import"))
    if len(parent_specs) != 1:
        raise PilotV2117ContinuationError(
            "pre-science namespace requires one parent-import cell"
        )
    parent_run_id = parent_specs[0].run_id
    files = {
        ".real-stage-execution.lock",
        "budget_ledger.json",
        "release_attestation.json",
        "run_ledger.json",
        V2117_ACCEPTANCE_FILENAME,
        f".{V2117_ACCEPTANCE_FILENAME}.pending",
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


def _audit_pre_science_namespace(
    raw_root: Path, contract: PilotContract
) -> None:
    allowed_files, allowed_directories = _expected_pre_science_paths(contract)
    unexpected: list[str] = []
    for path in sorted(raw_root.rglob("*")):
        relative = path.relative_to(raw_root).as_posix()
        if path.is_symlink():
            raise PilotV2117ContinuationError(
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
        raise PilotV2117ContinuationError(
            "pre-science raw namespace contains unexpected paths: "
            + ", ".join(unexpected[:10])
        )


def _acceptance_projections(
    contract: PilotContract,
    *,
    repo_root: Path,
    raw_root: Path,
    paid: Any,
) -> tuple[Any, ...]:
    from . import pilot_orchestrator as orch

    projections: list[Any] = []
    d_specs = tuple(contract.expand(stage="experiment-d"))
    groups = sorted({(spec.model_id, spec.environment_seed) for spec in d_specs})
    for model_id, seed in groups:
        group = tuple(
            spec
            for spec in d_specs
            if spec.model_id == model_id and spec.environment_seed == seed
        )
        representatives = [spec for spec in group if spec.arm_id == "matched-a"]
        if len(group) != 11 or len(representatives) != 1:
            raise PilotV2117ContinuationError("Experiment D acceptance group drifted")
        projections.append(
            orch._d_group_projection(
                contract,
                representatives[0],
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
        raise PilotV2117ContinuationError("V2.11.7 projection-unit denominator drifted")
    return tuple(projections)


def _acceptance_material(
    contract: PilotContract,
    *,
    repo_root: Path,
    raw_root: Path,
    paid: Any,
    budget_ledger: PilotBudgetLedger,
) -> dict[str, Any]:
    from . import pilot_orchestrator as orch

    parent = verify_v2117_parent_import_receipt(
        raw_root / "parent-import/parent_import_receipt.json",
        contract=contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
    )
    authority = verify_v2117_current_authority(
        contract=contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
    )
    projections = _acceptance_projections(
        contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
    )
    try:
        orch._assert_projection_matrix_fits(budget_ledger, projections)
    except Exception as exc:
        raise PilotV2117ContinuationError(
            f"complete V2.11.7 continuation exceeds a hard cap: {exc}"
        ) from exc
    calls_by_stage = {"experiment-d": 0, "experiment-b": 0, "cross-model": 0}
    cost_by_stage = {"experiment-d": 0.0, "experiment-b": 0.0, "cross-model": 0.0}
    storage_by_stage = {"experiment-d": 0, "experiment-b": 0, "cross-model": 0}
    projection_rows: list[dict[str, Any]] = []
    for projection in projections:
        stage_id = next(
            stage
            for stage in calls_by_stage
            if f"--{stage}--" in projection.run_id
        )
        calls_by_stage[stage_id] += int(projection.completions)
        cost_by_stage[stage_id] += float(projection.cost_usd)
        storage_by_stage[stage_id] += int(projection.storage_bytes)
        projection_rows.append(projection.to_dict())
    total_cost = sum(cost_by_stage.values())
    total_calls = sum(calls_by_stage.values())
    total_storage = sum(storage_by_stage.values())
    if (
        calls_by_stage
        != {"experiment-d": 1480, "experiment-b": 1440, "cross-model": 336}
        or total_calls != V2117_REMAINING_PROVIDER_COMPLETIONS
        or total_storage != V2117_REMAINING_PROJECTED_STORAGE_BYTES
        or not math.isclose(
            total_cost,
            V2117_REMAINING_PROJECTED_COST_USD,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise PilotV2117ContinuationError(
            "V2.11.7 full remaining projection differs from preregistration"
        )

    config_hashes: dict[str, str] = {}
    for stage_id in ("experiment-d", "experiment-b", "cross-model"):
        for spec in contract.expand(stage=stage_id):
            config = orch.config_for_spec(
                contract,
                spec,
                raw_root=raw_root,
                paid_provenance=paid,
                authority_repo_root=repo_root,
                verify_bound_inputs=True,
            )
            config_hashes[spec.run_id] = canonical_sha256(config.to_dict())
    if len(config_hashes) != V2117_REMAINING_CELL_COUNT:
        raise PilotV2117ContinuationError("V2.11.7 config denominator drifted")
    parent_specs = tuple(contract.expand(stage="parent-import"))
    budget_rows = budget_ledger.snapshot().get("runs")
    parent_row = (
        budget_rows.get(parent_specs[0].run_id)
        if isinstance(budget_rows, Mapping) and len(parent_specs) == 1
        else None
    )
    parent_actual = parent_row.get("actual") if isinstance(parent_row, Mapping) else None
    if (
        not isinstance(parent_actual, Mapping)
        or not isinstance(parent_actual.get("storage_bytes"), int)
        or parent_actual["storage_bytes"] < 1
        or parent_actual["storage_bytes"] > 5_000_000
    ):
        raise PilotV2117ContinuationError(
            "parent-import actual storage is unavailable for projected ledger total"
        )
    operational_import_storage = int(parent_actual["storage_bytes"])
    return {
        "parent_import": {
            "path": (V2117_RAW_ROOT / "parent-import/parent_import_receipt.json").as_posix(),
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
        "runner_configs": {
            "cell_count": len(config_hashes),
            "config_sha256_by_run_id": dict(sorted(config_hashes.items())),
            "config_set_sha256": canonical_sha256(config_hashes),
        },
        "budget_projection": {
            "projection_unit_count": len(projections),
            "fresh_provider_calls": total_calls,
            "fresh_calls_by_stage": calls_by_stage,
            "fresh_projected_cost_usd": total_cost,
            "fresh_projected_cost_usd_by_stage": cost_by_stage,
            "fresh_storage_bytes": total_storage,
            "fresh_storage_bytes_by_stage": storage_by_stage,
            "projection_sha256_by_run_id": {
                row["run_id"]: canonical_sha256(row)
                for row in sorted(projection_rows, key=lambda item: item["run_id"])
            },
            "projection_set_sha256": canonical_sha256(
                sorted(projection_rows, key=lambda item: item["run_id"])
            ),
            "projected_cumulative_cost_usd": V2117_PROJECTED_CUMULATIVE_COST_USD,
            "projected_cumulative_hosted_completions": (
                V2117_PROJECTED_CUMULATIVE_COMPLETIONS
            ),
            "projected_cumulative_storage_bytes": (
                V2117_PROJECTED_CUMULATIVE_STORAGE_BYTES
            ),
            "contract_preregistered_cumulative_storage_bytes": (
                V2117_PROJECTED_CUMULATIVE_STORAGE_BYTES
            ),
            "operational_import_storage_bytes": operational_import_storage,
            "ledger_projected_cumulative_storage_bytes": (
                V2117_PROJECTED_CUMULATIVE_STORAGE_BYTES
                + operational_import_storage
            ),
            "hard_caps": orch._budget_caps(contract).to_dict(),
            "full_matrix_fits": True,
        },
    }


def _expected_acceptance_denominator(contract: PilotContract) -> dict[str, Any]:
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
    }


def _verified_parent_import_budget_actual(
    contract: PilotContract, row: Any
) -> Mapping[str, Any]:
    from . import pilot_orchestrator as orch

    parent_specs = tuple(contract.expand(stage="parent-import"))
    if len(parent_specs) != 1:
        raise PilotV2117ContinuationError("parent-import budget denominator drifted")
    expected = orch._v2117_parent_import_projection(parent_specs[0]).to_dict()
    actual = row.get("actual") if isinstance(row, Mapping) else None
    if (
        not isinstance(row, Mapping)
        or row.get("stage_bucket") != expected["stage_bucket"]
        or row.get("reservation") != expected
        or row.get("status") != "complete"
        or not isinstance(actual, Mapping)
        or actual.get("cost_usd") != 0.0
        or actual.get("completions") != 0
        or not isinstance(actual.get("storage_bytes"), int)
        or actual["storage_bytes"] < 1
        or actual["storage_bytes"] > expected["storage_bytes"]
    ):
        raise PilotV2117ContinuationError(
            "parent-import budget row differs from its exact zero-provider projection"
        )
    return actual


def _audit_acceptance_denominator(
    contract: PilotContract,
    run_snapshot: Mapping[str, Any],
    budget_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    specs = tuple(contract.expand())
    rows = run_snapshot.get("runs")
    if (
        len(specs) != V2117_LEDGER_CELL_COUNT
        or not isinstance(rows, Mapping)
        or set(rows) != {spec.run_id for spec in specs}
        or any(rows[spec.run_id].get("spec") != spec.to_dict() for spec in specs)
    ):
        raise PilotV2117ContinuationError("V2.11.7 ITT ledger denominator drifted")
    parent_specs = tuple(contract.expand(stage="parent-import"))
    if len(parent_specs) != 1 or rows[parent_specs[0].run_id].get("status") != "complete":
        raise PilotV2117ContinuationError("parent import is not terminal complete")
    scientific = tuple(
        spec
        for stage_id in ("experiment-d", "experiment-b", "cross-model")
        for spec in contract.expand(stage=stage_id)
    )
    if len(scientific) != 86:
        raise PilotV2117ContinuationError("scientific continuation denominator drifted")
    status_counts = Counter(rows[spec.run_id].get("status") for spec in specs)
    if status_counts != Counter({"scheduled": 86, "complete": 1}):
        raise PilotV2117ContinuationError(
            "acceptance must occur before the first continuation science cell"
        )
    budget_runs = budget_snapshot.get("runs")
    if not isinstance(budget_runs, Mapping) or any(
        run_id != parent_specs[0].run_id for run_id in budget_runs
    ):
        raise PilotV2117ContinuationError(
            "acceptance budget prefix contains a scientific reservation"
        )
    _verified_parent_import_budget_actual(
        contract, budget_runs.get(parent_specs[0].run_id)
    )
    return _expected_acceptance_denominator(contract)


def _acceptance_receipt(
    contract: PilotContract,
    *,
    repo_root: Path,
    raw_root: Path,
    paid: Any,
    run_ledger: Any,
    budget_ledger: PilotBudgetLedger,
) -> dict[str, Any]:
    run_snapshot = run_ledger.snapshot()
    budget_snapshot = budget_ledger.snapshot()
    denominator = _audit_acceptance_denominator(
        contract, run_snapshot, budget_snapshot
    )
    material = _acceptance_material(
        contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
        budget_ledger=budget_ledger,
    )
    return _seal(
        {
            "schema_version": V2117_ACCEPTANCE_SCHEMA_VERSION,
            "status": "go",
            "go": True,
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "release": _release_binding(contract, paid),
            "raw_namespace": V2117_RAW_ROOT.as_posix(),
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


def _verify_acceptance_marker(
    snapshot: Mapping[str, Any],
    *,
    prefix: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_path: str,
    budget: bool,
) -> None:
    events = snapshot.get("events")
    count = prefix.get("event_count")
    if (
        not isinstance(events, list)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or len(events) <= count
        or events[count - 1].get("event_sha256") != prefix.get("event_chain_head")
    ):
        raise PilotV2117ContinuationError("accepted ledger event prefix is absent")
    marker = events[count]
    expected = {
        "receipt_schema_version": V2117_ACCEPTANCE_SCHEMA_VERSION,
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
        "accepted_budget_event_chain_head": receipt["ledger_prefixes"][
            "budget_ledger"
        ]["event_chain_head"],
        ("budget_runs_sha256" if budget else "runs_sha256"): prefix["runs_sha256"],
    }
    if marker.get("event_type") != "acceptance_receipt_bound" or marker.get(
        "payload"
    ) != expected:
        raise PilotV2117ContinuationError("acceptance ledger marker drifted")


def _verify_acceptance_prefix_state(
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
        set(prefix) != {
            "event_count",
            "event_chain_head",
            "ledger_sha256",
            "runs_sha256",
        }
        or not isinstance(events, list)
        or not isinstance(runs, Mapping)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or len(events) < count
        or events[count - 1].get("event_sha256") != prefix.get("event_chain_head")
    ):
        raise PilotV2117ContinuationError("accepted ledger prefix drifted")
    if len(events) == count:
        if (
            snapshot.get("ledger_sha256") != prefix.get("ledger_sha256")
            or canonical_sha256(runs) != prefix.get("runs_sha256")
        ):
            raise PilotV2117ContinuationError(
                "unmarked acceptance ledger differs from sealed prefix"
            )
        return False
    _verify_acceptance_marker(
        snapshot,
        prefix=prefix,
        receipt=receipt,
        receipt_path=receipt_path,
        budget=budget,
    )
    return True


def _verify_current_accepted_budget_rows(
    contract: PilotContract,
    receipt: Mapping[str, Any],
    budget_snapshot: Mapping[str, Any],
) -> None:
    rows = budget_snapshot.get("runs")
    projection = receipt["budget_projection"]
    accepted = projection["projection_sha256_by_run_id"]
    parent_specs = tuple(contract.expand(stage="parent-import"))
    if not isinstance(rows, Mapping) or len(parent_specs) != 1:
        raise PilotV2117ContinuationError("accepted budget rows are malformed")
    parent_id = parent_specs[0].run_id
    for run_id, row in rows.items():
        if run_id == parent_id:
            _verified_parent_import_budget_actual(contract, row)
            continue
        if not isinstance(row, Mapping) or run_id not in accepted:
            raise PilotV2117ContinuationError(
                f"unaccepted scientific budget row appeared: {run_id}"
            )
        reservation = row.get("reservation")
        if not isinstance(reservation, Mapping) or canonical_sha256(
            reservation
        ) != accepted[run_id]:
            raise PilotV2117ContinuationError(
                f"scientific budget reservation drifted: {run_id}"
            )


def _verify_v2117_acceptance_core(
    receipt: Mapping[str, Any],
    *,
    contract: PilotContract,
    repo_root: Path,
    raw_root: Path,
    paid: Any,
    run_ledger: Any,
    budget_ledger: PilotBudgetLedger,
    require_markers: bool,
) -> tuple[bool, bool]:
    _verify_seal(receipt, name="V2.11.7 scientific acceptance")
    if (
        set(receipt) != _ACCEPTANCE_TOP_LEVEL_FIELDS
        or receipt.get("schema_version") != V2117_ACCEPTANCE_SCHEMA_VERSION
        or receipt.get("status") != "go"
        or receipt.get("go") is not True
        or receipt.get("contract_id") != contract.contract_id
        or receipt.get("contract_sha256") != contract.canonical_hash
        or receipt.get("release") != _release_binding(contract, paid)
        or receipt.get("raw_namespace") != V2117_RAW_ROOT.as_posix()
        or receipt.get("denominator") != _expected_acceptance_denominator(contract)
        or receipt.get("provider_boundary") != _ACCEPTANCE_PROVIDER_BOUNDARY
        or receipt.get("scientific_evidence") is not False
        or receipt.get("claim_boundary") != _ACCEPTANCE_CLAIM_BOUNDARY
    ):
        raise PilotV2117ContinuationError("V2.11.7 acceptance identity drifted")
    material = _acceptance_material(
        contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
        budget_ledger=budget_ledger,
    )
    for key, value in material.items():
        if receipt.get(key) != value:
            raise PilotV2117ContinuationError(
                f"V2.11.7 acceptance field {key!r} drifted"
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
        raise PilotV2117ContinuationError("accepted ITT denominator no longer matches")
    _verify_current_accepted_budget_rows(contract, receipt, budget_snapshot)
    prefixes = receipt.get("ledger_prefixes")
    if not isinstance(prefixes, Mapping) or set(prefixes) != {
        "run_ledger",
        "budget_ledger",
    }:
        raise PilotV2117ContinuationError("acceptance ledger prefixes are absent")
    relative = (V2117_RAW_ROOT / V2117_ACCEPTANCE_FILENAME).as_posix()
    run_marked = _verify_acceptance_prefix_state(
        run_snapshot,
        prefix=prefixes["run_ledger"],
        receipt=receipt,
        receipt_path=relative,
        budget=False,
    )
    budget_marked = _verify_acceptance_prefix_state(
        budget_snapshot,
        prefix=prefixes["budget_ledger"],
        receipt=receipt,
        receipt_path=relative,
        budget=True,
    )
    if require_markers and not (run_marked and budget_marked):
        raise PilotV2117ContinuationError(
            "both acceptance ledger markers are required before dispatch"
        )
    return run_marked, budget_marked


def verify_v2117_scientific_dispatch_acceptance(
    receipt_path: str | Path,
    *,
    contract: PilotContract,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: Any,
    run_ledger: Any,
    budget_ledger: PilotBudgetLedger,
) -> dict[str, Any]:
    repository = _real_root(repo_root, name="V2.11.7 repository")
    raw = _real_root(raw_root, name="V2.11.7 raw root")
    path = Path(receipt_path).absolute()
    if path != raw / V2117_ACCEPTANCE_FILENAME:
        raise PilotV2117ContinuationError("V2.11.7 acceptance path drifted")
    receipt = _strict_json(path, name="V2.11.7 scientific acceptance")
    with _acceptance_provider_sentinels():
        _verify_v2117_acceptance_core(
            receipt,
            contract=contract,
            repo_root=repository,
            raw_root=raw,
            paid=paid,
            run_ledger=run_ledger,
            budget_ledger=budget_ledger,
            require_markers=True,
        )
    return receipt


def accept_v2117_scientific_dispatch(
    *,
    contract_path: str | Path,
    repo_root: str | Path,
    raw_root: str | Path,
    scientific_launch_input_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Seal and bind the exact 86-cell continuation before provider access."""

    from . import pilot_orchestrator as orch

    require_v2117_provider_keys_absent()
    repository = _real_root(repo_root, name="V2.11.7 repository")
    contract_candidate = Path(contract_path)
    if not contract_candidate.is_absolute():
        contract_candidate = repository / contract_candidate
    expected_contract = repository / "experiments/pilot_v2_11_7.yaml"
    if contract_candidate.absolute() != expected_contract:
        raise PilotV2117ContinuationError("acceptance contract path drifted")
    contract = load_pilot_contract(contract_candidate)
    raw = _real_root(raw_root, name="V2.11.7 raw root")
    if raw != repository.joinpath(*V2117_RAW_ROOT.parts):
        raise PilotV2117ContinuationError("acceptance raw namespace drifted")
    output = raw / V2117_ACCEPTANCE_FILENAME
    if receipt_path is not None and Path(receipt_path).absolute() != output:
        raise PilotV2117ContinuationError("acceptance output path drifted")
    launch = Path(scientific_launch_input_path).absolute()
    if launch != raw / "scientific_launch_input.json":
        raise PilotV2117ContinuationError("scientific launch input path drifted")
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
        )
        budget_ledger = PilotBudgetLedger(
            raw / "budget_ledger.json",
            contract_hash=contract.canonical_hash,
            caps=orch._budget_caps(contract),
            tamper_evident=True,
            parent_debit=parent_budget_debit_for_v2117(contract),
        )
        _audit_pre_science_namespace(raw, contract)
        with _acceptance_provider_sentinels():
            if output.exists():
                receipt = _strict_json(
                    output, name="V2.11.7 scientific acceptance"
                )
                run_marked, budget_marked = _verify_v2117_acceptance_core(
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
                    # Rebuild the exact candidate before binding an existing
                    # ignored-namespace file.  A pre-planted resealed receipt
                    # must compare byte-for-byte or fail without ledger writes.
                    candidate = _acceptance_receipt(
                        contract,
                        repo_root=repository,
                        raw_root=raw,
                        paid=paid,
                        run_ledger=run_ledger,
                        budget_ledger=budget_ledger,
                    )
                    _write_once(output, candidate)
                    receipt = candidate
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
                _verify_v2117_acceptance_core(
                    receipt,
                    contract=contract,
                    repo_root=repository,
                    raw_root=raw,
                    paid=paid,
                    run_ledger=run_ledger,
                    budget_ledger=budget_ledger,
                    require_markers=False,
                )
        prefixes = receipt.get("ledger_prefixes")
        if not isinstance(prefixes, Mapping):
            raise PilotV2117ContinuationError("acceptance prefixes are malformed")
        relative = (V2117_RAW_ROOT / V2117_ACCEPTANCE_FILENAME).as_posix()
        common = {
            "receipt_schema_version": V2117_ACCEPTANCE_SCHEMA_VERSION,
            "receipt_path": relative,
            "receipt_content_sha256": receipt["integrity"]["content_sha256"],
            "accepted_run_event_count": prefixes["run_ledger"]["event_count"],
            "accepted_run_event_chain_head": prefixes["run_ledger"][
                "event_chain_head"
            ],
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
        )
        reloaded_budget = PilotBudgetLedger(
            raw / "budget_ledger.json",
            contract_hash=contract.canonical_hash,
            caps=orch._budget_caps(contract),
            tamper_evident=True,
            parent_debit=parent_budget_debit_for_v2117(contract),
        )
        return verify_v2117_scientific_dispatch_acceptance(
            output,
            contract=contract,
            repo_root=repository,
            raw_root=raw,
            paid=paid,
            run_ledger=reloaded_run,
            budget_ledger=reloaded_budget,
        )


__all__ = [
    "PilotV2117ContinuationError",
    "V2117_ACCEPTANCE_FILENAME",
    "V2117_ACCEPTANCE_SCHEMA_VERSION",
    "V2117_CURRENT_AUTHORITY_SCHEMA_VERSION",
    "V2117_SOURCE_MANIFEST_PATH",
    "accept_v2117_scientific_dispatch",
    "build_v2117_parent_import_receipt",
    "build_v2117_source_manifest",
    "current_authority_path",
    "current_projection_path",
    "parent_budget_debit_for_v2117",
    "require_v2117_provider_keys_absent",
    "runner_reservations_for_v2117",
    "validate_v2117_source_manifest",
    "verified_v2117_calibration",
    "verified_v2117_observed_p95_authority_binding",
    "verified_v2117_projection",
    "verify_v2117_current_authority",
    "verify_v2117_parent_import_receipt",
    "verify_v2117_scientific_dispatch_acceptance",
]
