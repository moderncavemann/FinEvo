"""Immutable V2.11.7 no-go lineage for the V2.11.8 continuation.

This module is intentionally provider-free.  It binds the complete terminal
V2.11.7 failure as lineage and keeps the V2.11.5 science checkout as the only
dispatch-authority root.  No V2.11.7 row is resumed, reclassified, copied as an
effect, or charged as a fresh completion.

The functions here are read-only except for returning newly constructed JSON
objects.  Writing a V2.11.8 receipt belongs to the orchestrator's atomic stage
transaction, not to this provenance layer.
"""

from __future__ import annotations

from collections import Counter
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .pilot_budget import ParentBudgetDebit, PilotBudgetLedger
from .pilot_contract import PilotContract, canonical_sha256, load_pilot_contract
from . import pilot_v2117_continuation as v2117


V2118_CONTRACT_ID = "finevo-pilot-v2.11.8"
V2118_SCIENCE_TAG = "pilot-v2.11.8-science"
V2118_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_11_8_source_manifest.json"
)
V2118_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.11.8/raw")
V2118_PARENT_IMPORT_SCHEMA_VERSION = "finevo-pilot-v2.11.8-parent-import-v1"
V2118_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.11.8-source-manifest-v1"
V2115_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.11.5-source-manifest-v1"
V2118_CURRENT_AUTHORITY_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.8-continuation-observed-p95-authority-v1"
)
V2118_CURRENT_PROJECTION_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.8-continuation-projection-v1"
)
V2118_ACCEPTANCE_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.8-scientific-dispatch-acceptance-v1"
)
V2118_ACCEPTANCE_FILENAME = "scientific_dispatch_acceptance.json"

V2117_CONTRACT_ID = "finevo-pilot-v2.11.7"
V2117_SCIENCE_TAG = "pilot-v2.11.7-science"
V2117_SCIENCE_COMMIT = "57c53588440dc2647f6b6ffae519049db4cd4844"
V2117_SCIENCE_TAG_OBJECT = "6ce166fecfb126c07788bc87c31fcdc6ecb42078"
V2117_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_11_7.yaml")
V2117_CONTRACT_SHA256 = (
    "376c41f7b2793d4039bae43a652d6ba73759cce7b9b3f04fc665c41a23659e3b"
)
V2117_CONTRACT_FILE_SHA256 = (
    "4b570b212f391c1887b4a7cb3554ab65e6fac77b6d35e0a3aa8b0509e84c8d85"
)
V2117_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_11_7_source_manifest.json"
)
V2117_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.11.7-source-manifest-v1"
V2117_SOURCE_MANIFEST_FILE_SHA256 = (
    "dd124c09359d0bd08411add3486cc43887cbee207fdbb6f9bc929e5c1eb81ef9"
)
V2117_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "64be1bf836d131d8ec0542e68388dbc328314af7e891549600f5871f8f61f2b0"
)
V2117_FAILED_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.11.7/raw")

V2117_RUN_LEDGER_FILE_SHA256 = (
    "8dc580d7030f7aab182429bc7dd7bc72c6a0d61e7944477bce8a52836ac324cc"
)
V2117_RUN_LEDGER_SHA256 = (
    "bb6d497308097cf6f348c282339f2f6d4cb6721950604744c1e6b0751e913681"
)
V2117_RUN_EVENT_COUNT = 89
V2117_RUN_EVENT_HEAD = (
    "5446a8981e9f8579893c1faea39a0722972ee709967967e99086be614414dab6"
)
V2117_BUDGET_LEDGER_FILE_SHA256 = (
    "f9e0216cbc8e5d3ea6ceb7728ca1a4df0fd71ecfb49124a1ccacd3b0758d9272"
)
V2117_BUDGET_LEDGER_SHA256 = (
    "bc6cc622beaff05e2480e866408929f3edd7f02a7555bdb26202fe94ae3e9c77"
)
V2117_BUDGET_EVENT_COUNT = 4
V2117_BUDGET_EVENT_HEAD = (
    "66333a703087c3aae041171eb2d2a96ff2f7e3ff60aa454d8b55a08bda2f5fdd"
)
V2117_LEDGER_CELL_COUNT = 87
V2117_REMAINING_SCIENCE_CELL_COUNT = 86
V2117_PARENT_IMPORT_ACTUAL_STORAGE_BYTES = 1_797
V2117_CUMULATIVE_COST_USD = 63.1196450625
V2117_CUMULATIVE_COMPLETIONS = 3_440
V2117_CUMULATIVE_STORAGE_BYTES = 270_191_728
V2118_PARENT_DEBIT_RECORD_SHA256 = (
    "a8281fea88c404d504792b08d8bef75ee5d33d890ee5a44ed91962012ba87f1e"
)

V2117_FAILURE_MESSAGE = (
    "Experiment D group gpt52_main/617806385 failed validation: "
    "source-backed observed p95 release commit differs from the annotated "
    "tag or current HEAD"
)
V2117_FAILURE_ERROR_TYPE = "V2117ParentImportIntegrityError"
V2117_FAILURE_CAUSE_TYPE = "PilotV2115AcceptanceError"
V2117_PARENT_IMPORT_RECEIPT_CONTENT_SHA256 = (
    "914ee31e0b2f1102d77b18b26e2ae133247df13ae403c9df97545be80637abba"
)
V2117_RELEASE_ATTESTATION_SHA256 = (
    "df033a800e85d3a0a918e10b4627f01bba3bb8aa046c04a3a3440370e6ea226c"
)
V2117_LAUNCH_INPUT_SHA256 = (
    "2ae474f0958a31e070b36d85a4cbe28a6b2301c8f9bf2b8a47da2e30303b8f2b"
)

# The operational lock is deliberately included here.  The conventional
# evidence inventory remains the five JSON artifacts below, but V2.11.8 also
# binds the exact six-file terminal namespace requested by the recovery
# contract so the stale lock cannot be silently added, removed, or replaced.
V2117_RAW_FILE_BINDINGS: Mapping[str, tuple[int, str]] = {
    ".real-stage-execution.lock": (
        140,
        "f2be7ac0a92cff13e173925fa39bef2d22595599bcd5b5e0f3501900851dd716",
    ),
    "budget_ledger.json": (5_823, V2117_BUDGET_LEDGER_FILE_SHA256),
    "parent-import/stage_receipt.json": (
        1_797,
        "d8e276f8eaa725b1b32666f09508c984d536c9e5eb05e54cf6bb6faaff2b0ddc",
    ),
    "release_attestation.json": (
        15_558,
        "07805c14c5a6673805de1f0d1d3a423a98270d9b420227c5ac1865e2ffc64a7e",
    ),
    "run_ledger.json": (198_674, V2117_RUN_LEDGER_FILE_SHA256),
    "scientific_launch_input.json": (
        2_219,
        "0a29ceb2d7d21f3d99032d40543c074cc0095f5ac5acebf206e3c37cb8215e27",
    ),
}
V2117_COMPLETE_RAW_FILE_COUNT = 6
V2117_COMPLETE_RAW_STORAGE_BYTES = 224_211
V2117_COMPLETE_RAW_INVENTORY_SHA256 = (
    "13d7cc64beebafaf82aed90ebe4fd1abd1c00c300352a0bdba14fff492b1c7cf"
)
V2117_EVIDENCE_RAW_FILE_COUNT = 5
V2117_EVIDENCE_RAW_STORAGE_BYTES = 224_071
V2117_EVIDENCE_RAW_INVENTORY_SHA256 = (
    "af4053b3e7fc2b706707f47d552d56ac25dfff4fbf5df5d58a6739e375f160ec"
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

_CURRENT_AUTHORITY_PATH = (
    V2118_RAW_ROOT / "parent-import/current_authority/post_gate_authority.json"
)

_PROVIDER_KEY_ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)


class PilotV2118ContinuationError(RuntimeError):
    """Raised before V2.11.8 may construct a provider."""


def _json_copy(value: Any) -> Any:
    try:
        return v2117._json_copy(value)
    except v2117.PilotV2117ContinuationError as exc:
        raise PilotV2118ContinuationError(str(exc)) from exc


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        return v2117._strict_json(path, name=name)
    except v2117.PilotV2117ContinuationError as exc:
        raise PilotV2118ContinuationError(str(exc)) from exc


def _file_sha256(path: Path) -> str:
    try:
        return v2117._file_sha256(path)
    except (OSError, v2117.PilotV2117ContinuationError) as exc:
        raise PilotV2118ContinuationError(f"cannot hash {path}") from exc


def _real_root(value: str | Path, *, name: str) -> Path:
    try:
        return v2117._real_root(value, name=name)
    except v2117.PilotV2117ContinuationError as exc:
        raise PilotV2118ContinuationError(str(exc)) from exc


def _verify_seal(value: Mapping[str, Any], *, name: str) -> None:
    try:
        v2117._verify_seal(value, name=name)
    except v2117.PilotV2117ContinuationError as exc:
        raise PilotV2118ContinuationError(str(exc)) from exc


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return v2117._seal(value)
    except v2117.PilotV2117ContinuationError as exc:
        raise PilotV2118ContinuationError(str(exc)) from exc


def _boundary(contract: Any) -> Mapping[str, Any]:
    value = getattr(contract, "v2118_recovery_boundary", None)
    if not isinstance(value, Mapping):
        raise PilotV2118ContinuationError("V2.11.8 recovery boundary is absent")
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
        "source_manifest_content_sha256": (
            v2117.V2115_SOURCE_MANIFEST_CONTENT_SHA256
        ),
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
    """Exact V2.11.5 release shape used by the V2.11.8 contract."""

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


def _expected_v2117_failed_release_no_go() -> dict[str, Any]:
    return {
        "contract_id": V2117_CONTRACT_ID,
        "contract_path": V2117_CONTRACT_PATH.as_posix(),
        "contract_file_sha256": V2117_CONTRACT_FILE_SHA256,
        "contract_sha256": V2117_CONTRACT_SHA256,
        "science_tag": V2117_SCIENCE_TAG,
        "science_tag_object": V2117_SCIENCE_TAG_OBJECT,
        "science_commit": V2117_SCIENCE_COMMIT,
        "source_manifest_path": V2117_SOURCE_MANIFEST_PATH.as_posix(),
        "source_manifest_file_sha256": V2117_SOURCE_MANIFEST_FILE_SHA256,
        "source_manifest_content_sha256": V2117_SOURCE_MANIFEST_CONTENT_SHA256,
        "raw_inventory": {
            "root": V2117_FAILED_RAW_ROOT.as_posix(),
            "canonicalization": "json-sort-keys-compact-utf8-v1",
            "excluded_operational_paths": [".real-stage-execution.lock"],
            "file_count": V2117_EVIDENCE_RAW_FILE_COUNT,
            "storage_bytes": V2117_EVIDENCE_RAW_STORAGE_BYTES,
            "inventory_sha256": V2117_EVIDENCE_RAW_INVENTORY_SHA256,
        },
        "run_ledger": {
            "path": (V2117_FAILED_RAW_ROOT / "run_ledger.json").as_posix(),
            "file_sha256": V2117_RUN_LEDGER_FILE_SHA256,
            "ledger_sha256": V2117_RUN_LEDGER_SHA256,
            "event_count": V2117_RUN_EVENT_COUNT,
            "event_head_sha256": V2117_RUN_EVENT_HEAD,
            "registered_rows": V2117_LEDGER_CELL_COUNT,
            "status_counts": {"integrity-stopped": V2117_LEDGER_CELL_COUNT},
        },
        "budget_ledger": {
            "path": (V2117_FAILED_RAW_ROOT / "budget_ledger.json").as_posix(),
            "file_sha256": V2117_BUDGET_LEDGER_FILE_SHA256,
            "ledger_sha256": V2117_BUDGET_LEDGER_SHA256,
            "event_count": V2117_BUDGET_EVENT_COUNT,
            "event_head_sha256": V2117_BUDGET_EVENT_HEAD,
            "current_actual": {
                "cost_usd": 0.0,
                "hosted_completions": 0,
                "storage_bytes": V2117_PARENT_IMPORT_ACTUAL_STORAGE_BYTES,
            },
        },
        "stage_receipt": {
            "path": (
                V2117_FAILED_RAW_ROOT / "parent-import/stage_receipt.json"
            ).as_posix(),
            "file_sha256": V2117_RAW_FILE_BINDINGS[
                "parent-import/stage_receipt.json"
            ][1],
            "content_sha256": V2117_PARENT_IMPORT_RECEIPT_CONTENT_SHA256,
            "status": "integrity-stopped",
            "go": False,
            "execution_progression_go": False,
            "failure_error_type": V2117_FAILURE_ERROR_TYPE,
            "failure_cause_type": V2117_FAILURE_CAUSE_TYPE,
            "failure_message": V2117_FAILURE_MESSAGE,
        },
        "release_attestation": {
            "path": (V2117_FAILED_RAW_ROOT / "release_attestation.json").as_posix(),
            "file_sha256": V2117_RAW_FILE_BINDINGS["release_attestation.json"][1],
            "attestation_sha256": V2117_RELEASE_ATTESTATION_SHA256,
            "status": "pass",
        },
        "scientific_launch_input": {
            "path": (
                V2117_FAILED_RAW_ROOT / "scientific_launch_input.json"
            ).as_posix(),
            "file_sha256": V2117_RAW_FILE_BINDINGS[
                "scientific_launch_input.json"
            ][1],
            "launch_input_sha256": V2117_LAUNCH_INPUT_SHA256,
        },
        "acceptance_receipt_present": False,
        "science_reservations": 0,
        "provider_construction": False,
        "provider_calls": 0,
        "scientific_evidence": False,
        "resume_forbidden": True,
        "failure_reclassification_forbidden": True,
    }


def _verify_v2117_release_git(root: Path) -> dict[str, str]:
    try:
        return v2117._verify_release_git(
            root,
            name="V2.11.7 failed release",
            science_tag=V2117_SCIENCE_TAG,
            science_commit=V2117_SCIENCE_COMMIT,
            science_tag_object=V2117_SCIENCE_TAG_OBJECT,
        )
    except v2117.PilotV2117ContinuationError as exc:
        raise PilotV2118ContinuationError(str(exc)) from exc


def _verify_v2115_authority_git(root: Path) -> dict[str, str]:
    try:
        return v2117._verify_authority_git(root)
    except v2117.PilotV2117ContinuationError as exc:
        raise PilotV2118ContinuationError(str(exc)) from exc


def _v2117_raw_inventory(root: Path) -> dict[str, Any]:
    raw = root.joinpath(*V2117_FAILED_RAW_ROOT.parts)
    if raw.is_symlink() or not raw.is_dir():
        raise PilotV2118ContinuationError("V2.11.7 failed raw root is unavailable")
    rows: list[dict[str, Any]] = []
    for path in sorted(raw.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise PilotV2118ContinuationError("V2.11.7 raw tree contains a symlink")
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(raw).as_posix(),
                    "byte_size": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
        elif not path.is_dir():
            raise PilotV2118ContinuationError(
                "V2.11.7 raw tree contains a non-regular entry"
            )
    observed = {
        str(row["path"]): (int(row["byte_size"]), str(row["sha256"]))
        for row in rows
    }
    if observed != dict(V2117_RAW_FILE_BINDINGS):
        raise PilotV2118ContinuationError("V2.11.7 six-file raw binding drifted")
    evidence_rows = [
        row for row in rows if row["path"] != ".real-stage-execution.lock"
    ]
    complete = {
        "root": V2117_FAILED_RAW_ROOT.as_posix(),
        "canonicalization": "json-sort-keys-compact-utf8-v1",
        "excluded_operational_paths": [],
        "file_count": len(rows),
        "storage_bytes": sum(int(row["byte_size"]) for row in rows),
        "inventory_sha256": canonical_sha256(rows),
    }
    evidence = {
        "root": V2117_FAILED_RAW_ROOT.as_posix(),
        "canonicalization": "json-sort-keys-compact-utf8-v1",
        "excluded_operational_paths": [".real-stage-execution.lock"],
        "file_count": len(evidence_rows),
        "storage_bytes": sum(int(row["byte_size"]) for row in evidence_rows),
        "inventory_sha256": canonical_sha256(evidence_rows),
    }
    expected_evidence = _expected_v2117_failed_release_no_go()["raw_inventory"]
    expected_complete = {
        "root": V2117_FAILED_RAW_ROOT.as_posix(),
        "canonicalization": "json-sort-keys-compact-utf8-v1",
        "excluded_operational_paths": [],
        "file_count": V2117_COMPLETE_RAW_FILE_COUNT,
        "storage_bytes": V2117_COMPLETE_RAW_STORAGE_BYTES,
        "inventory_sha256": V2117_COMPLETE_RAW_INVENTORY_SHA256,
    }
    if complete != expected_complete or evidence != expected_evidence:
        raise PilotV2118ContinuationError("V2.11.7 raw inventory digest drifted")
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
        raise PilotV2118ContinuationError("V2.11.5 authority source drifted")
    try:
        raw_inventory = v2117._authority_raw_inventory(root)
    except v2117.PilotV2117ContinuationError as exc:
        raise PilotV2118ContinuationError(str(exc)) from exc
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
        raise PilotV2118ContinuationError("V2.11.5 authority receipt drifted")
    from .pilot_v2115_gate import verified_v2115_gate_authority_binding

    try:
        gate_binding = verified_v2115_gate_authority_binding(
            gate_path.relative_to(root).as_posix(),
            repo_root=root,
            expected_git_commit=v2117.V2115_SCIENCE_COMMIT,
            expected_contract_sha256=v2117.V2115_CONTRACT_SHA256,
        )
    except Exception as exc:
        raise PilotV2118ContinuationError(
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
        raise PilotV2118ContinuationError("V2.11.5 parent-import receipt drifted")
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


def verify_v2117_terminal_no_go(
    *,
    failed_repo_root: str | Path,
    authority_repo_root: str | Path,
) -> dict[str, Any]:
    """Deep-verify V2.11.7 as a terminal zero-provider no-go.

    ``failed_repo_root`` must be the clean V2.11.7 annotated-tag checkout;
    ``authority_repo_root`` must independently be the clean V2.11.5 checkout.
    The two roles are intentionally not interchangeable.
    """

    from .pilot_orchestrator import PilotRunLedger, _budget_caps

    failed_root = _real_root(failed_repo_root, name="V2.11.7 failed repository")
    authority_root = _real_root(
        authority_repo_root, name="V2.11.5 authority repository"
    )
    if failed_root == authority_root:
        raise PilotV2118ContinuationError(
            "V2.11.7 failed and V2.11.5 authority roots must be distinct"
        )
    failed_release = _verify_v2117_release_git(failed_root)
    authority = _v2115_authority_state(authority_root)
    raw_inventory = _v2117_raw_inventory(failed_root)

    contract_path = failed_root.joinpath(*V2117_CONTRACT_PATH.parts)
    failed_contract = load_pilot_contract(contract_path)
    manifest_path = failed_root.joinpath(*V2117_SOURCE_MANIFEST_PATH.parts)
    manifest = _strict_json(manifest_path, name="V2.11.7 source manifest")
    _verify_seal(manifest, name="V2.11.7 source manifest")
    if (
        failed_contract.contract_id != V2117_CONTRACT_ID
        or failed_contract.canonical_hash != V2117_CONTRACT_SHA256
        or _file_sha256(contract_path) != V2117_CONTRACT_FILE_SHA256
        or _file_sha256(manifest_path) != V2117_SOURCE_MANIFEST_FILE_SHA256
        or manifest.get("integrity", {}).get("content_sha256")
        != V2117_SOURCE_MANIFEST_CONTENT_SHA256
        or manifest.get("schema_version") != V2117_SOURCE_MANIFEST_SCHEMA_VERSION
        or manifest.get("contract_id") != V2117_CONTRACT_ID
        or manifest.get("release_tag") != V2117_SCIENCE_TAG
    ):
        raise PilotV2118ContinuationError("V2.11.7 release source drifted")

    raw = failed_root.joinpath(*V2117_FAILED_RAW_ROOT.parts)
    run_path = raw / "run_ledger.json"
    run_ledger = PilotRunLedger(
        run_path,
        contract_hash=failed_contract.canonical_hash,
        tamper_evident=True,
    )
    run_snapshot = run_ledger.snapshot()
    events = run_snapshot.get("events")
    runs = run_snapshot.get("runs")
    expected_specs = {spec.run_id: spec.to_dict() for spec in failed_contract.expand()}
    if (
        _file_sha256(run_path) != V2117_RUN_LEDGER_FILE_SHA256
        or run_snapshot.get("ledger_sha256") != V2117_RUN_LEDGER_SHA256
        or not isinstance(events, list)
        or len(events) != V2117_RUN_EVENT_COUNT
        or events[-1].get("event_sha256") != V2117_RUN_EVENT_HEAD
        or Counter(event.get("event_type") for event in events)
        != Counter({"genesis": 1, "runs_registered": 1, "run_finalized": 87})
        or not isinstance(runs, Mapping)
        or len(runs) != V2117_LEDGER_CELL_COUNT
        or set(runs) != set(expected_specs)
        or any(
            not isinstance(row, Mapping)
            or row.get("spec") != expected_specs[run_id]
            or row.get("status") != "integrity-stopped"
            or row.get("artifact") is not None
            or not isinstance(row.get("failure"), Mapping)
            or row["failure"].get("provider_calls") != 0
            or row["failure"].get("provider_construction") is not False
            or row["failure"].get("error_type") != V2117_FAILURE_ERROR_TYPE
            or row["failure"].get("cause_type") != V2117_FAILURE_CAUSE_TYPE
            or row["failure"].get("message") != V2117_FAILURE_MESSAGE
            for run_id, row in runs.items()
        )
    ):
        raise PilotV2118ContinuationError(
            "V2.11.7 terminal run-ledger no-go drifted"
        )

    parent_specs = tuple(failed_contract.expand(stage="parent-import"))
    science_specs = tuple(
        spec
        for stage_id in ("experiment-d", "experiment-b", "cross-model")
        for spec in failed_contract.expand(stage=stage_id)
    )
    if len(parent_specs) != 1 or len(science_specs) != V2117_REMAINING_SCIENCE_CELL_COUNT:
        raise PilotV2118ContinuationError("V2.11.7 failed denominator drifted")
    parent_run_id = parent_specs[0].run_id
    parent_failure = runs[parent_run_id]["failure"]

    budget_path = raw / "budget_ledger.json"
    stored_budget = _strict_json(budget_path, name="V2.11.7 budget ledger")
    imported_parent = v2117.parent_budget_debit_for_v2117(failed_contract)
    if stored_budget.get("parent_debit") != imported_parent.to_dict():
        raise PilotV2118ContinuationError("V2.11.7 imported parent debit drifted")
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
    committed = {
        "cost_usd": V2117_CUMULATIVE_COST_USD,
        "completions": V2117_CUMULATIVE_COMPLETIONS,
        "storage_bytes": V2117_CUMULATIVE_STORAGE_BYTES,
        "stage_cost_usd": {
            "hosted_v2117": 0.0,
            "manual_reserve": 0.0,
            "parent_v2116": V2117_CUMULATIVE_COST_USD,
        },
    }
    if (
        _file_sha256(budget_path) != V2117_BUDGET_LEDGER_FILE_SHA256
        or budget.get("ledger_sha256") != V2117_BUDGET_LEDGER_SHA256
        or not isinstance(budget_events, list)
        or len(budget_events) != V2117_BUDGET_EVENT_COUNT
        or budget_events[-1].get("event_sha256") != V2117_BUDGET_EVENT_HEAD
        or Counter(event.get("event_type") for event in budget_events)
        != Counter(
            {
                "genesis": 1,
                "parent_debit_imported": 1,
                "run_reserved": 1,
                "run_finalized": 1,
            }
        )
        or budget.get("committed") != committed
        or budget.get("committed_plus_reserved") != committed
        or not isinstance(budget_runs, Mapping)
        or set(budget_runs) != {parent_run_id}
    ):
        raise PilotV2118ContinuationError(
            "V2.11.7 terminal budget-ledger no-go drifted"
        )
    budget_row = budget_runs[parent_run_id]
    reservation = budget_row.get("reservation")
    actual = budget_row.get("actual")
    if (
        budget_row.get("status") != "integrity-stopped"
        or budget_row.get("stage_bucket") != "parent_v2116"
        or not isinstance(reservation, Mapping)
        or reservation.get("cost_usd") != 0.0
        or reservation.get("completions") != 0
        or reservation.get("basis", {}).get("provider_calls") != 0
        or reservation.get("basis", {}).get("provider_construction") is not False
        or actual
        != {
            "cost_usd": 0.0,
            "completions": 0,
            "storage_bytes": V2117_PARENT_IMPORT_ACTUAL_STORAGE_BYTES,
        }
        or budget_row.get("failure") != parent_failure
        or {spec.run_id for spec in science_specs} & set(budget_runs)
    ):
        raise PilotV2118ContinuationError(
            "V2.11.7 no-science-reservation boundary drifted"
        )

    stage_path = raw / "parent-import/stage_receipt.json"
    stage = _strict_json(stage_path, name="V2.11.7 parent-import receipt")
    if (
        stage.get("integrity", {}).get("content_sha256")
        != V2117_PARENT_IMPORT_RECEIPT_CONTENT_SHA256
        or stage.get("contract_id") != V2117_CONTRACT_ID
        or stage.get("contract_sha256") != V2117_CONTRACT_SHA256
        or stage.get("stage_id") != "parent-import"
        or stage.get("status") != "integrity-stopped"
        or stage.get("go") is not False
        or stage.get("execution_progression_go") is not False
        or stage.get("terminal") is not True
        or stage.get("denominator_terminal") is not True
        or stage.get("scientific_matrix_complete") is not False
        or stage.get("registered_run_count") != 1
        or stage.get("complete_cell_count") != 0
        or stage.get("hard_stop_cell_count") != 1
        or stage.get("status_counts") != {"integrity-stopped": 1}
        or stage.get("failure") != parent_failure
    ):
        raise PilotV2118ContinuationError(
            "V2.11.7 parent-import terminal receipt drifted"
        )

    attestation = _strict_json(raw / "release_attestation.json", name="V2.11.7 attestation")
    launch = _strict_json(raw / "scientific_launch_input.json", name="V2.11.7 launch input")
    if (
        attestation.get("status") != "pass"
        or attestation.get("attestation_sha256") != V2117_RELEASE_ATTESTATION_SHA256
        or attestation.get("head_commit") != V2117_SCIENCE_COMMIT
        or attestation.get("local_tag", {}).get("object_id")
        != V2117_SCIENCE_TAG_OBJECT
        or attestation.get("contract", {}).get("canonical_sha256")
        != V2117_CONTRACT_SHA256
        or launch.get("contract_sha256") != V2117_CONTRACT_SHA256
        or launch.get("launch_input_sha256") != V2117_LAUNCH_INPUT_SHA256
    ):
        raise PilotV2118ContinuationError("V2.11.7 launch provenance drifted")
    if any(
        event.get("event_type") == "acceptance_receipt_bound"
        for event in (*events, *budget_events)
    ):
        raise PilotV2118ContinuationError(
            "V2.11.7 unexpectedly contains scientific acceptance"
        )
    return {
        "failed_release": failed_release,
        "failed_contract": failed_contract,
        "failed_raw": raw,
        "source_manifest": manifest,
        "raw_inventory": raw_inventory,
        "run_snapshot": run_snapshot,
        "budget_snapshot": budget,
        "stage_receipt": stage,
        "parent_run_id": parent_run_id,
        "science_run_ids": sorted(spec.run_id for spec in science_specs),
        "authority": authority,
    }


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
    paid = GitProvenance(
        git_tag=v2117.V2115_SCIENCE_TAG,
        head_commit=v2117.V2115_SCIENCE_COMMIT,
        tag_commit=v2117.V2115_SCIENCE_COMMIT,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
        release_attestation=_strict_json(
            raw / "release_attestation.json", name="V2.11.5 attestation"
        ),
    )
    try:
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
        raise PilotV2118ContinuationError(
            f"V2.11.5 acceptance revalidation failed: {exc}"
        ) from exc


def require_v2118_provider_keys_absent() -> None:
    present = [name for name in _PROVIDER_KEY_ENV_NAMES if os.environ.get(name)]
    if present:
        raise PilotV2118ContinuationError(
            "V2.11.8 parent import must run before provider credentials are "
            f"loaded; present={sorted(present)}"
        )


def parent_budget_debit_for_v2118(contract: Any) -> ParentBudgetDebit:
    if getattr(contract, "contract_id", None) != V2118_CONTRACT_ID:
        raise PilotV2118ContinuationError("parent debit requires V2.11.8")
    expected = {
        "parent_contract_sha256": V2117_CONTRACT_SHA256,
        "parent_run_ledger_sha256": V2117_RUN_LEDGER_SHA256,
        "parent_budget_ledger_sha256": V2117_BUDGET_LEDGER_SHA256,
        "stage_bucket": "parent_v2117",
        "cost_usd": V2117_CUMULATIVE_COST_USD,
        "hosted_completions": V2117_CUMULATIVE_COMPLETIONS,
        "storage_bytes": V2117_CUMULATIVE_STORAGE_BYTES,
    }
    declared = _boundary(contract).get("parent_budget_debit")
    if not isinstance(declared, Mapping) or any(
        declared.get(key) != value for key, value in expected.items()
    ):
        raise PilotV2118ContinuationError("V2.11.8 parent budget debit drifted")
    debit = ParentBudgetDebit(**expected)
    if debit.record_sha256 != V2118_PARENT_DEBIT_RECORD_SHA256:
        raise PilotV2118ContinuationError("V2.11.8 parent debit seal drifted")
    return debit


def current_authority_path(raw_root: str | Path) -> Path:
    return Path(raw_root) / "parent-import/current_authority/post_gate_authority.json"


def current_projection_path(raw_root: str | Path, model_id: str) -> Path:
    if model_id not in {"gpt52_main", "gpt56_diagnostic"}:
        raise PilotV2118ContinuationError(
            f"unsupported V2.11.8 continuation model {model_id}"
        )
    return Path(raw_root) / f"parent-import/current_authority/{model_id}/projection_p95.json"


def _tracked_source_manifest(
    contract: Any, *, repo_root: str | Path
) -> tuple[Path, dict[str, Any]]:
    repository = _real_root(repo_root, name="V2.11.8 repository")
    path = repository.joinpath(*V2118_SOURCE_MANIFEST_PATH.parts)
    value = _strict_json(path, name="V2.11.8 source manifest")
    _verify_seal(value, name="V2.11.8 source manifest")
    declared = _boundary(contract).get("source_manifest")
    if (
        not isinstance(declared, Mapping)
        or declared.get("path") != V2118_SOURCE_MANIFEST_PATH.as_posix()
        or declared.get("schema_version") != V2118_SOURCE_MANIFEST_SCHEMA_VERSION
        or declared.get("file_sha256") != _file_sha256(path)
        or declared.get("content_sha256")
        != value.get("integrity", {}).get("content_sha256")
        or value.get("schema_version") != V2118_SOURCE_MANIFEST_SCHEMA_VERSION
        or value.get("contract_id") != V2118_CONTRACT_ID
        or value.get("release_tag") != V2118_SCIENCE_TAG
    ):
        raise PilotV2118ContinuationError(
            "V2.11.8 tracked source-manifest binding drifted"
        )
    return path, value


def _normalize_authority_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    value = _json_copy(spec)
    run_id = str(value.get("run_id", ""))
    prefix = f"{v2117.V2115_CONTRACT_ID}--"
    if not run_id.startswith(prefix):
        raise PilotV2118ContinuationError("V2.11.5 scheduled run id is malformed")
    value["run_id"] = f"{V2118_CONTRACT_ID}--{run_id[len(prefix):]}"
    value["contract_id"] = V2118_CONTRACT_ID
    value["budget_bucket"] = "hosted_v2118"
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
        len(source_specs) != V2117_REMAINING_SCIENCE_CELL_COUNT
        or len(child_specs) != V2117_REMAINING_SCIENCE_CELL_COUNT
        or len(child_by_id) != V2117_REMAINING_SCIENCE_CELL_COUNT
    ):
        raise PilotV2118ContinuationError(
            "V2.11.8 canonical mapping denominator drifted"
        )
    rows: list[dict[str, Any]] = []
    for source_spec in sorted(source_specs, key=lambda item: item.run_id):
        source = source_spec.to_dict()
        child = _normalize_authority_spec(source)
        if child_by_id.get(child["run_id"]) != child:
            raise PilotV2118ContinuationError(
                f"V2.11.8 child spec differs for {source_spec.run_id}"
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
        "schema_version": "finevo-pilot-v2.11.8-canonical-cell-mapping-v1",
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
        raise PilotV2118ContinuationError(
            "V2.11.8 canonical mapping identity drifted"
        )
    return mapping


def _source_file_binding(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise PilotV2118ContinuationError(
            f"required V2.11.8 source is unavailable: {relative}"
        )
    return {
        "path": relative,
        "byte_size": path.stat().st_size,
        "file_sha256": _file_sha256(path),
    }


def _current_runtime_source_bindings(
    child: Path, authority: Path
) -> dict[str, Any]:
    """Seal successor glue without creating a contract/manifest hash cycle."""

    full_hash_paths = (
        "run_pilot.py",
        "verified_memory/observed_p95_authority.py",
        "verified_memory/pilot_orchestrator.py",
        "verified_memory/pilot_v2115_acceptance.py",
        "verified_memory/pilot_v2118_continuation.py",
        "scripts/render_pilot_v2118_contract.py",
        "scripts/render_pilot_v2118_source_manifest.py",
    )
    files = [_source_file_binding(child, relative) for relative in full_hash_paths]
    contract_relative = "verified_memory/pilot_contract.py"
    try:
        child_contract_inventory = v2117._ast_top_level_inventory(
            child / contract_relative
        )
        authority_contract_inventory = v2117._ast_top_level_inventory(
            authority / contract_relative
        )
    except (OSError, SyntaxError, v2117.PilotV2117ContinuationError) as exc:
        raise PilotV2118ContinuationError(
            "cannot seal the V2.11.8 contract parser AST"
        ) from exc
    return {
        "full_file_bindings": files,
        "full_file_binding_set_sha256": canonical_sha256(files),
        "pilot_contract_path": contract_relative,
        "pilot_contract_top_level_ast_inventory_sha256": {
            "authority": canonical_sha256(authority_contract_inventory),
            "child": canonical_sha256(child_contract_inventory),
        },
        "pilot_contract_top_level_node_counts": {
            "authority": len(authority_contract_inventory),
            "child": len(child_contract_inventory),
        },
        "cycle_avoidance": (
            "pilot_contract.py is bound by its complete top-level function/class "
            "AST inventory; generated contract and source-manifest hash constants "
            "are intentionally excluded"
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
        child_plan = orch.build_v2118_experiment_d_group_plan(
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
            raise PilotV2118ContinuationError(
                f"V2.11.8 D plan differs from V2.11.5 at seed {seed}"
            )
        rows.append(
            {
                "seed": seed,
                "normalized_plan_sha256": canonical_sha256(child_value),
            }
        )
    if len(rows) != 5:
        raise PilotV2118ContinuationError("V2.11.8 D plan lacks five seeds")
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
            raise PilotV2118ContinuationError(
                f"remaining-science source differs from V2.11.5: {relative}"
            )
        identical.append(child_binding)
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
        raise PilotV2118ContinuationError(
            "cannot verify V2.11.8 science-function equivalence"
        ) from exc
    if child_functions != authority_functions:
        changed = sorted(
            name
            for name in child_functions
            if child_functions[name] != authority_functions.get(name)
        )
        raise PilotV2118ContinuationError(
            f"remaining-science orchestrator functions drifted: {changed}"
        )
    plans = _normalized_d_plan_receipts(child_contract, authority_contract)
    return {
        "policy": "science-core-equal-with-context-recovery-adapter-v1",
        "byte_identical_files": identical,
        "byte_identical_files_sha256": canonical_sha256(identical),
        "orchestrator_path": orchestrator_relative,
        "unchanged_orchestrator_function_sha256": child_functions,
        "unchanged_orchestrator_set_sha256": canonical_sha256(child_functions),
        "experiment_d_normalized_plan_receipts": plans,
        "experiment_d_normalized_plan_set_sha256": canonical_sha256(plans),
        "equivalence_claim": "science_core_equal_with_context_recovery_adapter",
        "full_runtime_byte_identity_claimed": False,
    }


def build_v2118_source_manifest(
    *,
    contract: PilotContract,
    repo_root: str | Path,
    failed_repo_root: str | Path,
    authority_repo_root: str | Path,
) -> dict[str, Any]:
    """Build the deterministic V2.11.8 dual-root source manifest."""

    if contract.contract_id != V2118_CONTRACT_ID:
        raise PilotV2118ContinuationError("source manifest requires V2.11.8")
    child = _real_root(repo_root, name="V2.11.8 repository")
    failed = _real_root(failed_repo_root, name="V2.11.7 failed repository")
    authority = _real_root(
        authority_repo_root, name="V2.11.5 authority repository"
    )
    state = verify_v2117_terminal_no_go(
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
        raise PilotV2118ContinuationError(
            "V2.11.8 source-manifest mapping binding drifted"
        )
    return _seal(
        {
            "schema_version": V2118_SOURCE_MANIFEST_SCHEMA_VERSION,
            "contract_id": V2118_CONTRACT_ID,
            # The frozen contract binds this manifest after bootstrap.  Keeping
            # its canonical hash out of this object avoids a hash cycle.
            "release_tag": str(contract.implementation["required_git_tag"]),
            "failed_release": {
                "contract_id": V2117_CONTRACT_ID,
                "contract_sha256": V2117_CONTRACT_SHA256,
                "contract_file_sha256": V2117_CONTRACT_FILE_SHA256,
                "source_manifest_path": V2117_SOURCE_MANIFEST_PATH.as_posix(),
                "source_manifest_file_sha256": (
                    V2117_SOURCE_MANIFEST_FILE_SHA256
                ),
                "source_manifest_content_sha256": (
                    V2117_SOURCE_MANIFEST_CONTENT_SHA256
                ),
                **state["failed_release"],
            },
            "failed_terminal_no_go": _expected_v2117_failed_release_no_go(),
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
            "observed_p95_context_recovery": {
                "failed_release_contract_id": V2117_CONTRACT_ID,
                "failure_error_type": V2117_FAILURE_ERROR_TYPE,
                "failure_cause_type": V2117_FAILURE_CAUSE_TYPE,
                "failure_message": V2117_FAILURE_MESSAGE,
                "authority_release_contract_id": v2117.V2115_CONTRACT_ID,
                "authority_gate_path": (
                    V2115_RAW_ROOT
                    / "long-context-preflight/post_gate_authority.json"
                ).as_posix(),
                "authority_gate_content_sha256": V2115_POST_GATE_CONTENT_SHA256,
                "repair_changes_scientific_design": False,
                "scientific_outcomes_inspected_for_repair": False,
                "additional_source_release_required": False,
                "provider_construction": False,
                "provider_calls": 0,
            },
            "observation_boundary": {
                "failed_v2117_is_terminal_lineage_only": True,
                "failed_v2117_effect_rows_imported": 0,
                "authority_v2115_a_c_outcomes_are_frozen_external_evidence": True,
                "authority_v2115_a_c_rows_imported_into_child_ledger": 0,
                "authority_v2115_scheduled_cells_mapped_to_child": (
                    V2117_REMAINING_SCIENCE_CELL_COUNT
                ),
                "decoded_completion_reuse": False,
                "provider_calls": 0,
                "provider_construction": False,
            },
        }
    )


def validate_v2118_source_manifest(
    *,
    contract: PilotContract,
    repo_root: str | Path,
    failed_repo_root: str | Path,
    authority_repo_root: str | Path,
) -> dict[str, Any]:
    root = _real_root(repo_root, name="V2.11.8 repository")
    path = root.joinpath(*V2118_SOURCE_MANIFEST_PATH.parts)
    observed = _strict_json(path, name="V2.11.8 source manifest")
    _verify_seal(observed, name="V2.11.8 source manifest")
    expected = build_v2118_source_manifest(
        contract=contract,
        repo_root=root,
        failed_repo_root=failed_repo_root,
        authority_repo_root=authority_repo_root,
    )
    if observed != expected:
        raise PilotV2118ContinuationError(
            "V2.11.8 source manifest replay drifted"
        )
    declared = _boundary(contract).get("source_manifest")
    expected_binding = {
        "path": V2118_SOURCE_MANIFEST_PATH.as_posix(),
        "schema_version": V2118_SOURCE_MANIFEST_SCHEMA_VERSION,
        "file_sha256": _file_sha256(path),
        "content_sha256": expected["integrity"]["content_sha256"],
    }
    if (
        not isinstance(declared, Mapping)
        or {key: declared.get(key) for key in expected_binding}
        != expected_binding
    ):
        raise PilotV2118ContinuationError(
            "V2.11.8 contract source-manifest identity drifted"
        )
    return observed


def _capability_summary(wrapper: Mapping[str, Any], *, model_id: str) -> dict[str, Any]:
    try:
        return v2117._capability_summary(wrapper, model_id=model_id)
    except v2117.PilotV2117ContinuationError as exc:
        raise PilotV2118ContinuationError(str(exc)) from exc


def _dispatch_authority_source(authority: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return v2117._dispatch_authority_source(authority)
    except v2117.PilotV2117ContinuationError as exc:
        raise PilotV2118ContinuationError(str(exc)) from exc


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    try:
        v2117._write_once(path, value)
    except v2117.PilotV2117ContinuationError as exc:
        raise PilotV2118ContinuationError(str(exc)) from exc


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
        raise PilotV2118ContinuationError("V2.11.5 p95 authority is malformed")
    numeric: dict[str, dict[str, Any]] = {}
    stable: dict[str, dict[str, Any]] = {}
    for runtime_model, by_kind in source.items():
        if not isinstance(by_kind, Mapping) or set(by_kind) != {"action", "semantic"}:
            raise PilotV2118ContinuationError("V2.11.5 call-kind denominator drifted")
        numeric[str(runtime_model)] = {}
        stable[str(runtime_model)] = {}
        for kind in ("action", "semantic"):
            entry = by_kind[kind]
            authority = entry.get("authority") if isinstance(entry, Mapping) else None
            reservation = entry.get("reservation") if isinstance(entry, Mapping) else None
            if not isinstance(authority, Mapping) or not isinstance(reservation, Mapping):
                raise PilotV2118ContinuationError("V2.11.5 p95 row is malformed")
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
            "schema_version": V2118_CURRENT_AUTHORITY_SCHEMA_VERSION,
            "contract_id": V2118_CONTRACT_ID,
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
            "provider_boundary": {
                "provider_calls": 0,
                "hosted_provider_calls": 0,
                "hosted_cost_usd": 0.0,
                "provider_construction": False,
            },
            "scientific_evidence": False,
            "claim_boundary": (
                "V2.11.8 current-release dispatch-budget authority only; no "
                "decoded completion or A/C effect row is imported."
            ),
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
            raise PilotV2118ContinuationError(
                f"V2.11.8 current authority lacks {model_id}/{runtime}"
            )
        projection = _seal(
            {
                "schema_version": V2118_CURRENT_PROJECTION_SCHEMA_VERSION,
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
            }
        )
        _write_once(current_projection_path(raw_root, model_id), projection)
        projections[model_id] = projection
    return authority, projections


def verify_v2118_current_authority(
    *,
    contract: Any,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: Any,
) -> dict[str, Any]:
    repository = _real_root(repo_root, name="V2.11.8 repository")
    raw = _real_root(raw_root, name="V2.11.8 raw root")
    parent = verify_v2118_parent_import_receipt(
        raw / "parent-import/parent_import_receipt.json",
        contract=contract,
        repo_root=repository,
        raw_root=raw,
        paid=paid,
    )
    path = current_authority_path(raw)
    authority = _strict_json(path, name="V2.11.8 current p95 authority")
    _verify_seal(authority, name="V2.11.8 current p95 authority")
    dispatch = parent.get("dispatch_authority_source")
    release = _release_binding(contract, paid)
    expected_source = (
        dispatch.get("source_gate") if isinstance(dispatch, Mapping) else None
    )
    if (
        authority.get("schema_version") != V2118_CURRENT_AUTHORITY_SCHEMA_VERSION
        or authority.get("contract_id") != V2118_CONTRACT_ID
        or authority.get("contract_sha256") != contract.canonical_hash
        or authority.get("release")
        != {"git_tag": release["git_tag"], "git_commit": release["git_commit"]}
        or authority.get("authority_release")
        != {
            "contract_id": v2117.V2115_CONTRACT_ID,
            "contract_sha256": v2117.V2115_CONTRACT_SHA256,
            "git_tag": v2117.V2115_SCIENCE_TAG,
            "git_commit": v2117.V2115_SCIENCE_COMMIT,
            "source_gate": expected_source,
        }
        or authority.get("parent_import_content_sha256")
        != parent["integrity"]["content_sha256"]
        or authority.get("reservations")
        != (dispatch.get("reservations") if isinstance(dispatch, Mapping) else None)
        or authority.get("stable_source_authorities")
        != (
            dispatch.get("stable_source_authorities")
            if isinstance(dispatch, Mapping)
            else None
        )
        or authority.get("provider_boundary")
        != {
            "provider_calls": 0,
            "hosted_provider_calls": 0,
            "hosted_cost_usd": 0.0,
            "provider_construction": False,
        }
        or authority.get("scientific_evidence") is not False
    ):
        raise PilotV2118ContinuationError(
            "V2.11.8 current p95 authority drifted"
        )
    return authority


def verified_v2118_projection(
    contract: Any,
    model_id: str,
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: Any,
) -> tuple[dict[str, Any], Path]:
    authority = verify_v2118_current_authority(
        contract=contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
    )
    raw = _real_root(raw_root, name="V2.11.8 raw root")
    path = current_projection_path(raw, model_id)
    value = _strict_json(path, name=f"V2.11.8 {model_id} p95 projection")
    _verify_seal(value, name=f"V2.11.8 {model_id} p95 projection")
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
    if (
        value.get("schema_version") != V2118_CURRENT_PROJECTION_SCHEMA_VERSION
        or value.get("model_id") != model_id
        or value.get("runtime_model") != runtime
        or value.get("served_model") != profile.served_model
        or value.get("projection") != expected_projection
        or bindings
        != {
            "contract_sha256": contract.canonical_hash,
            "git_tag": paid.git_tag,
            "git_commit": paid.head_commit,
            "authority_path": _CURRENT_AUTHORITY_PATH.as_posix(),
            "authority_file_sha256": _file_sha256(current_authority_path(raw)),
            "authority_content_sha256": authority["integrity"]["content_sha256"],
            "parent_import_content_sha256": authority[
                "parent_import_content_sha256"
            ],
        }
        or value.get("provider_calls") != 0
        or value.get("provider_construction") is not False
        or value.get("scientific_evidence") is not False
    ):
        raise PilotV2118ContinuationError(
            f"V2.11.8 {model_id} current p95 projection drifted"
        )
    return value, path


def verified_v2118_calibration(
    contract: Any,
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: Any,
) -> dict[str, Any]:
    receipt = verify_v2118_parent_import_receipt(
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
        raise PilotV2118ContinuationError(
            "V2.11.8 calibration authority drifted"
        )
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


def verified_v2118_observed_p95_authority_binding(
    receipt_path: str | Path,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
    expected_contract_sha256: str,
) -> dict[str, Any]:
    repository = _real_root(repo_root, name="V2.11.8 repository")
    contract = load_pilot_contract(repository / "experiments/pilot_v2_11_8.yaml")
    if (
        contract.contract_id != V2118_CONTRACT_ID
        or contract.canonical_hash != expected_contract_sha256
    ):
        raise PilotV2118ContinuationError("V2.11.8 contract identity drifted")

    class _Paid:
        pass

    paid = _Paid()
    paid.git_tag = V2118_SCIENCE_TAG
    paid.head_commit = expected_git_commit
    paid.tag_commit = expected_git_commit
    paid.tag_object_type = "tag"
    paid.worktree_clean = True
    raw = repository.joinpath(*V2118_RAW_ROOT.parts)
    expected_path = current_authority_path(raw)
    requested = Path(receipt_path)
    if not requested.is_absolute():
        requested = repository.joinpath(*PurePosixPath(str(receipt_path)).parts)
    if requested.absolute() != expected_path:
        raise PilotV2118ContinuationError("V2.11.8 authority path drifted")
    authority = verify_v2118_current_authority(
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
                "pilot_tag": V2118_SCIENCE_TAG,
                "source_projection_schema_version": "finevo-pilot-projection-p95-v1",
                "source_projection_file_sha256": file_hash,
                "source_projection_content_sha256": content_hash,
                "source_authority_receipt_path": _CURRENT_AUTHORITY_PATH.as_posix(),
                "source_authority_receipt_file_sha256": file_hash,
                "source_authority_receipt_content_sha256": content_hash,
                "source_release_commit": expected_git_commit,
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


def runner_reservations_for_v2118(
    contract: Any,
    model_id: str,
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: Any,
) -> dict[str, dict[str, Any]]:
    projection, _ = verified_v2118_projection(
        contract,
        model_id,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
    )
    binding = verified_v2118_observed_p95_authority_binding(
        _CURRENT_AUTHORITY_PATH.as_posix(),
        repo_root=repo_root,
        expected_git_commit=paid.head_commit,
        expected_contract_sha256=contract.canonical_hash,
    )
    runtime = projection["runtime_model"]
    selected = binding["reservations"].get(runtime)
    if not isinstance(selected, Mapping) or set(selected) != {"action", "semantic"}:
        raise PilotV2118ContinuationError(
            "V2.11.8 runner authority denominator drifted"
        )
    for kind in ("action", "semantic"):
        key = f"{projection['served_model']}::{kind}"
        if selected[kind]["reservation"] != projection["projection"][key]:
            raise PilotV2118ContinuationError("V2.11.8 runner/projection drifted")
    return {runtime: _json_copy(selected)}


def _release_binding(contract: Any, paid: Any) -> dict[str, Any]:
    implementation = getattr(contract, "implementation", None)
    expected_tag = (
        implementation.get("required_git_tag")
        if isinstance(implementation, Mapping)
        else None
    )
    head = str(getattr(paid, "head_commit", ""))
    if (
        getattr(contract, "contract_id", None) != V2118_CONTRACT_ID
        or expected_tag != V2118_SCIENCE_TAG
        or getattr(paid, "git_tag", None) != V2118_SCIENCE_TAG
        or len(head) != 40
        or any(character not in "0123456789abcdef" for character in head)
        or getattr(paid, "tag_commit", None) != head
        or getattr(paid, "tag_object_type", None) != "tag"
        or getattr(paid, "worktree_clean", None) is not True
    ):
        raise PilotV2118ContinuationError(
            "V2.11.8 parent import requires its clean annotated release"
        )
    return {
        "git_tag": V2118_SCIENCE_TAG,
        "git_commit": head,
        "tag_object_type": "tag",
        "worktree_clean": True,
    }


def build_v2118_parent_import_receipt(
    *,
    contract: Any,
    repo_root: str | Path | None = None,
    raw_root: str | Path | None = None,
    failed_repo_root: str | Path,
    authority_repo_root: str | Path,
    paid: Any,
) -> dict[str, Any]:
    """Build the zero-provider V2.11.8 lineage and current authority bundle."""

    require_v2118_provider_keys_absent()
    state = verify_v2117_terminal_no_go(
        failed_repo_root=failed_repo_root,
        authority_repo_root=authority_repo_root,
    )
    _verify_v2115_acceptance_with_authority_context(authority_repo_root)
    boundary = _boundary(contract)
    expected_failed = _expected_v2117_failed_release_no_go()
    expected_authority = _expected_v2115_parent_release()
    if (
        _json_copy(boundary.get("failed_release_no_go")) != expected_failed
        or _json_copy(boundary.get("parent_release")) != expected_authority
    ):
        raise PilotV2118ContinuationError("V2.11.8 lineage boundary drifted")
    debit = parent_budget_debit_for_v2118(contract)
    payload: dict[str, Any] = {
            "schema_version": V2118_PARENT_IMPORT_SCHEMA_VERSION,
            "status": "complete",
            "go": True,
            "contract_id": V2118_CONTRACT_ID,
            "contract_sha256": getattr(contract, "canonical_hash", None),
            "release": _release_binding(contract, paid),
            "failed_release_no_go": expected_failed,
            "authority_release": expected_authority,
            "denominator_continuation": {
                "failed_registered_rows": V2117_LEDGER_CELL_COUNT,
                "failed_integrity_stopped_rows": V2117_LEDGER_CELL_COUNT,
                "failed_rows_reclassified_or_redispatched": 0,
                "child_operational_rows": 1,
                "child_scientific_rows": V2117_REMAINING_SCIENCE_CELL_COUNT,
            },
            "cumulative_parent_budget_debit": debit.to_dict(),
            "verified_terminal_bindings": {
                "run_ledger_sha256": state["run_snapshot"]["ledger_sha256"],
                "budget_ledger_sha256": state["budget_snapshot"]["ledger_sha256"],
                "stage_receipt_content_sha256": state["stage_receipt"][
                    "integrity"
                ]["content_sha256"],
            },
            "import_policy": {
                "provider_construction": False,
                "provider_calls": 0,
                "hosted_provider_calls": 0,
                "hosted_cost_usd": 0.0,
                "decoded_completion_reuse": False,
                "imported_effect_cells": 0,
                "failed_raw_tree_copied": False,
                "authority_raw_tree_copied": False,
                "validation_before_provider_construction": True,
            },
            "scientific_evidence": False,
            "claim_boundary": (
                "Immutable V2.11.7 integrity no-go lineage plus V2.11.5 "
                "dispatch authority only; no V2.11.7 outcome is resumed or "
                "reclassified."
            ),
    }
    if (repo_root is None) != (raw_root is None):
        raise PilotV2118ContinuationError(
            "V2.11.8 repo_root and raw_root must be supplied together"
        )
    if repo_root is not None and raw_root is not None:
        repository = _real_root(repo_root, name="V2.11.8 repository")
        raw = _real_root(raw_root, name="V2.11.8 raw root")
        if raw != repository.joinpath(*V2118_RAW_ROOT.parts):
            raise PilotV2118ContinuationError("V2.11.8 raw namespace drifted")
        source_path, source = _tracked_source_manifest(
            contract, repo_root=repository
        )
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
            raise PilotV2118ContinuationError(
                "V2.11.5 calibration/capability authority drifted"
            )
        capability_summaries = {
            model_id: _capability_summary(
                capabilities[model_id], model_id=model_id
            )
            for model_id in sorted(capabilities)
        }
        dispatch = _dispatch_authority_source(authority_state)
        mapping = _canonical_remaining_cell_mapping(
            contract, authority_state["contract"]
        )
        payload.update(
            {
                "source_manifest": {
                    "path": V2118_SOURCE_MANIFEST_PATH.as_posix(),
                    "file_sha256": _file_sha256(source_path),
                    "content_sha256": source["integrity"]["content_sha256"],
                },
                "failed_terminal_no_go": {
                    "registered_rows": V2117_LEDGER_CELL_COUNT,
                    "integrity_stopped_rows": V2117_LEDGER_CELL_COUNT,
                    "provider_calls": 0,
                    "provider_construction": False,
                    "scientific_acceptance_present": False,
                    "science_budget_reservation_count": 0,
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
    if repo_root is not None and raw_root is not None:
        raw = Path(raw_root).resolve(strict=True)
        _write_once(raw / "parent-import/parent_import_receipt.json", receipt)
        _build_current_authority(
            contract=contract,
            raw_root=raw,
            paid=paid,
            authority_state=state["authority"],
            parent_import_content_sha256=receipt["integrity"]["content_sha256"],
        )
    return receipt


def verify_v2118_parent_import_receipt(
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
            raise PilotV2118ContinuationError(
                "persisted V2.11.8 receipt requires release provenance"
            )
        repository = _real_root(repo_root, name="V2.11.8 repository")
        raw = _real_root(raw_root, name="V2.11.8 raw root")
        path = Path(receipt).absolute()
        if (
            raw != repository.joinpath(*V2118_RAW_ROOT.parts)
            or path != raw / "parent-import/parent_import_receipt.json"
        ):
            raise PilotV2118ContinuationError("parent-import receipt path drifted")
        value = _strict_json(path, name="V2.11.8 parent-import receipt")
    _verify_seal(value, name="V2.11.8 parent-import receipt")
    expected_debit = parent_budget_debit_for_v2118(contract).to_dict()
    if (
        value.get("schema_version") != V2118_PARENT_IMPORT_SCHEMA_VERSION
        or value.get("status") != "complete"
        or value.get("go") is not True
        or value.get("contract_id") != V2118_CONTRACT_ID
        or value.get("contract_sha256") != getattr(contract, "canonical_hash", None)
        or (
            paid is not None
            and value.get("release") != _release_binding(contract, paid)
        )
        or value.get("failed_release_no_go")
        != _expected_v2117_failed_release_no_go()
        or value.get("authority_release") != _expected_v2115_parent_release()
        or value.get("cumulative_parent_budget_debit") != expected_debit
        or value.get("scientific_evidence") is not False
        or value.get("import_policy", {}).get("provider_construction") is not False
        or value.get("import_policy", {}).get("provider_calls") != 0
        or value.get("import_policy", {}).get("imported_effect_cells") != 0
    ):
        raise PilotV2118ContinuationError("V2.11.8 parent-import receipt drifted")
    if repo_root is not None:
        source_path, source = _tracked_source_manifest(contract, repo_root=repo_root)
        mapping = value.get("canonical_remaining_cell_mapping")
        dispatch = value.get("dispatch_authority_source")
        if (
            value.get("source_manifest")
            != {
                "path": V2118_SOURCE_MANIFEST_PATH.as_posix(),
                "file_sha256": _file_sha256(source_path),
                "content_sha256": source["integrity"]["content_sha256"],
            }
            or not isinstance(mapping, Mapping)
            or mapping.get("row_count") != V2117_REMAINING_SCIENCE_CELL_COUNT
            or mapping.get("mapping_sha256")
            != _boundary(contract)["continuation_matrix"][
                "canonical_86_row_mapping_sha256"
            ]
            or canonical_sha256(mapping.get("rows"))
            != mapping.get("mapping_sha256")
            or not isinstance(dispatch, Mapping)
            or dispatch.get("reservation_set_sha256")
            != v2117.V2115_RESERVATION_SET_SHA256
            or dispatch.get("stable_authority_set_sha256")
            != v2117.V2115_STABLE_AUTHORITY_SET_SHA256
        ):
            raise PilotV2118ContinuationError(
                "V2.11.8 persisted parent-import authority drifted"
            )
    return value


def _ledger_prefix(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return v2117._ledger_prefix(snapshot)
    except v2117.PilotV2117ContinuationError as exc:
        raise PilotV2118ContinuationError(str(exc)) from exc


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
        "failed_v2117_terminal_rows": 87,
    }


def _verified_parent_import_budget_actual(
    contract: Any, row: Any
) -> Mapping[str, Any]:
    from . import pilot_orchestrator as orch

    specs = tuple(contract.expand(stage="parent-import"))
    if len(specs) != 1:
        raise PilotV2118ContinuationError("parent-import budget denominator drifted")
    expected = orch._v2118_parent_import_projection(specs[0]).to_dict()
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
        raise PilotV2118ContinuationError(
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
        len(specs) != V2117_LEDGER_CELL_COUNT
        or not isinstance(rows, Mapping)
        or set(rows) != {spec.run_id for spec in specs}
        or any(rows[spec.run_id].get("spec") != spec.to_dict() for spec in specs)
    ):
        raise PilotV2118ContinuationError("V2.11.8 ITT denominator drifted")
    parent = tuple(contract.expand(stage="parent-import"))
    science = tuple(
        spec
        for stage_id in ("experiment-d", "experiment-b", "cross-model")
        for spec in contract.expand(stage=stage_id)
    )
    if (
        len(parent) != 1
        or len(science) != V2117_REMAINING_SCIENCE_CELL_COUNT
        or rows[parent[0].run_id].get("status") != "complete"
        or Counter(rows[spec.run_id].get("status") for spec in specs)
        != Counter({"scheduled": 86, "complete": 1})
    ):
        raise PilotV2118ContinuationError(
            "acceptance must precede the first V2.11.8 science cell"
        )
    budget_rows = budget_snapshot.get("runs")
    if not isinstance(budget_rows, Mapping) or set(budget_rows) != {parent[0].run_id}:
        raise PilotV2118ContinuationError(
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
            raise PilotV2118ContinuationError("Experiment D group drifted")
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
        raise PilotV2118ContinuationError(
            "V2.11.8 projection-unit denominator drifted"
        )
    return tuple(projections)


def _acceptance_material(
    contract: Any,
    *,
    repo_root: Path,
    raw_root: Path,
    paid: Any,
    budget_ledger: PilotBudgetLedger,
) -> dict[str, Any]:
    from . import pilot_orchestrator as orch

    parent = verify_v2118_parent_import_receipt(
        raw_root / "parent-import/parent_import_receipt.json",
        contract=contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
    )
    authority = verify_v2118_current_authority(
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
        raise PilotV2118ContinuationError(
            f"complete V2.11.8 continuation exceeds a hard cap: {exc}"
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
        raise PilotV2118ContinuationError(
            "V2.11.8 full projection differs from preregistration"
        )
    configs: dict[str, str] = {}
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
            configs[spec.run_id] = canonical_sha256(config.to_dict())
    if len(configs) != V2117_REMAINING_SCIENCE_CELL_COUNT:
        raise PilotV2118ContinuationError("V2.11.8 config denominator drifted")
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
                V2118_RAW_ROOT / "parent-import/parent_import_receipt.json"
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
        "runner_configs": {
            "cell_count": len(configs),
            "config_sha256_by_run_id": dict(sorted(configs.items())),
            "config_set_sha256": canonical_sha256(configs),
        },
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
            "schema_version": V2118_ACCEPTANCE_SCHEMA_VERSION,
            "status": "go",
            "go": True,
            "contract_id": V2118_CONTRACT_ID,
            "contract_sha256": contract.canonical_hash,
            "release": _release_binding(contract, paid),
            "raw_namespace": V2118_RAW_ROOT.as_posix(),
            "denominator": denominator,
            **material,
            "ledger_prefixes": {
                "run_ledger": _ledger_prefix(run_snapshot),
                "budget_ledger": _ledger_prefix(budget_snapshot),
            },
            "provider_boundary": {
                "credential_environment_names_checked": list(
                    _PROVIDER_KEY_ENV_NAMES
                ),
                "credential_values_present": False,
                "provider_construction": False,
                "provider_calls": 0,
                "provider_catalog_calls": 0,
                "hosted_provider_calls": 0,
                "hosted_cost_usd": 0.0,
                "validation_before_provider_construction": True,
            },
            "scientific_evidence": False,
            "claim_boundary": (
                "Pre-dispatch V2.11.8 integrity and budget acceptance only; "
                "no treatment outcome is created or imported."
            ),
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
        raise PilotV2118ContinuationError("accepted ledger prefix drifted")
    if len(events) == count:
        if (
            snapshot.get("ledger_sha256") != prefix.get("ledger_sha256")
            or canonical_sha256(runs) != prefix.get("runs_sha256")
        ):
            raise PilotV2118ContinuationError(
                "unmarked acceptance ledger differs from sealed prefix"
            )
        return False
    marker = events[count]
    expected = {
        "receipt_schema_version": V2118_ACCEPTANCE_SCHEMA_VERSION,
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
        raise PilotV2118ContinuationError("acceptance ledger marker drifted")
    return True


def _verify_current_accepted_budget_rows(
    contract: Any,
    receipt: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> None:
    rows = snapshot.get("runs")
    accepted = receipt["budget_projection"]["projection_sha256_by_run_id"]
    parent = tuple(contract.expand(stage="parent-import"))
    if not isinstance(rows, Mapping) or len(parent) != 1:
        raise PilotV2118ContinuationError("accepted budget rows are malformed")
    for run_id, row in rows.items():
        if run_id == parent[0].run_id:
            _verified_parent_import_budget_actual(contract, row)
            continue
        reservation = row.get("reservation") if isinstance(row, Mapping) else None
        if (
            run_id not in accepted
            or not isinstance(reservation, Mapping)
            or canonical_sha256(reservation) != accepted[run_id]
        ):
            raise PilotV2118ContinuationError(
                f"unaccepted science reservation appeared: {run_id}"
            )


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
    _verify_seal(receipt, name="V2.11.8 scientific acceptance")
    if (
        receipt.get("schema_version") != V2118_ACCEPTANCE_SCHEMA_VERSION
        or receipt.get("status") != "go"
        or receipt.get("go") is not True
        or receipt.get("contract_id") != V2118_CONTRACT_ID
        or receipt.get("contract_sha256") != contract.canonical_hash
        or receipt.get("release") != _release_binding(contract, paid)
        or receipt.get("raw_namespace") != V2118_RAW_ROOT.as_posix()
        or receipt.get("denominator") != _expected_acceptance_denominator(contract)
        or receipt.get("provider_boundary", {}).get("provider_construction")
        is not False
        or receipt.get("provider_boundary", {}).get("provider_calls") != 0
        or receipt.get("scientific_evidence") is not False
    ):
        raise PilotV2118ContinuationError("V2.11.8 acceptance identity drifted")
    material = _acceptance_material(
        contract,
        repo_root=repo_root,
        raw_root=raw_root,
        paid=paid,
        budget_ledger=budget_ledger,
    )
    for key, value in material.items():
        if receipt.get(key) != value:
            raise PilotV2118ContinuationError(
                f"V2.11.8 acceptance field {key!r} drifted"
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
        raise PilotV2118ContinuationError("accepted ITT denominator drifted")
    _verify_current_accepted_budget_rows(contract, receipt, budget_snapshot)
    prefixes = receipt.get("ledger_prefixes")
    if not isinstance(prefixes, Mapping) or set(prefixes) != {
        "run_ledger",
        "budget_ledger",
    }:
        raise PilotV2118ContinuationError("acceptance ledger prefixes are absent")
    relative = (V2118_RAW_ROOT / V2118_ACCEPTANCE_FILENAME).as_posix()
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
        raise PilotV2118ContinuationError(
            "both acceptance markers are required before dispatch"
        )
    return run_marked, budget_marked


def verify_v2118_scientific_dispatch_acceptance(
    receipt_path: str | Path,
    *,
    contract: Any,
    repo_root: str | Path,
    raw_root: str | Path,
    paid: Any,
    run_ledger: Any,
    budget_ledger: PilotBudgetLedger,
) -> dict[str, Any]:
    repository = _real_root(repo_root, name="V2.11.8 repository")
    raw = _real_root(raw_root, name="V2.11.8 raw root")
    path = Path(receipt_path).absolute()
    if path != raw / V2118_ACCEPTANCE_FILENAME:
        raise PilotV2118ContinuationError("V2.11.8 acceptance path drifted")
    receipt = _strict_json(path, name="V2.11.8 scientific acceptance")
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
        raise PilotV2118ContinuationError(str(exc)) from exc
    return receipt


def accept_v2118_scientific_dispatch(
    *,
    contract_path: str | Path,
    repo_root: str | Path,
    raw_root: str | Path,
    scientific_launch_input_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    from . import pilot_orchestrator as orch

    require_v2118_provider_keys_absent()
    repository = _real_root(repo_root, name="V2.11.8 repository")
    candidate = Path(contract_path)
    if not candidate.is_absolute():
        candidate = repository / candidate
    expected_contract = repository / "experiments/pilot_v2_11_8.yaml"
    if candidate.absolute() != expected_contract:
        raise PilotV2118ContinuationError("acceptance contract path drifted")
    contract = load_pilot_contract(candidate)
    raw = _real_root(raw_root, name="V2.11.8 raw root")
    if raw != repository.joinpath(*V2118_RAW_ROOT.parts):
        raise PilotV2118ContinuationError("acceptance raw namespace drifted")
    output = raw / V2118_ACCEPTANCE_FILENAME
    if receipt_path is not None and Path(receipt_path).absolute() != output:
        raise PilotV2118ContinuationError("acceptance output path drifted")
    launch = Path(scientific_launch_input_path).absolute()
    if launch != raw / "scientific_launch_input.json":
        raise PilotV2118ContinuationError("scientific launch input path drifted")
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
            parent_debit=parent_budget_debit_for_v2118(contract),
        )
        try:
            with v2117._acceptance_provider_sentinels():
                if output.exists():
                    receipt = _strict_json(
                        output, name="V2.11.8 scientific acceptance"
                    )
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
            raise PilotV2118ContinuationError(str(exc)) from exc
        prefixes = receipt["ledger_prefixes"]
        relative = (V2118_RAW_ROOT / V2118_ACCEPTANCE_FILENAME).as_posix()
        common = {
            "receipt_schema_version": V2118_ACCEPTANCE_SCHEMA_VERSION,
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
            parent_debit=parent_budget_debit_for_v2118(contract),
        )
        return verify_v2118_scientific_dispatch_acceptance(
            output,
            contract=contract,
            repo_root=repository,
            raw_root=raw,
            paid=paid,
            run_ledger=reloaded_run,
            budget_ledger=reloaded_budget,
        )


__all__ = [
    "PilotV2118ContinuationError",
    "V2118_CONTRACT_ID",
    "V2118_ACCEPTANCE_FILENAME",
    "V2118_ACCEPTANCE_SCHEMA_VERSION",
    "V2118_CURRENT_AUTHORITY_SCHEMA_VERSION",
    "V2118_PARENT_IMPORT_SCHEMA_VERSION",
    "V2118_RAW_ROOT",
    "V2118_SCIENCE_TAG",
    "V2118_SOURCE_MANIFEST_PATH",
    "V2118_SOURCE_MANIFEST_SCHEMA_VERSION",
    "build_v2118_source_manifest",
    "build_v2118_parent_import_receipt",
    "accept_v2118_scientific_dispatch",
    "current_authority_path",
    "current_projection_path",
    "parent_budget_debit_for_v2118",
    "require_v2118_provider_keys_absent",
    "runner_reservations_for_v2118",
    "verified_v2118_calibration",
    "verified_v2118_observed_p95_authority_binding",
    "verified_v2118_projection",
    "verify_v2117_terminal_no_go",
    "verify_v2118_current_authority",
    "verify_v2118_parent_import_receipt",
    "verify_v2118_scientific_dispatch_acceptance",
    "validate_v2118_source_manifest",
]
