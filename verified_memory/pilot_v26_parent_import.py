"""Zero-provider V2.6 reseal of immutable V2.5 p95 authorities.

The external V2.5 checkout is consulted only by :func:`persist_v26_parent_import`.
That import verifies the frozen V2.5 release, complete raw-tree inventory,
parent-import receipt, and both p95 receipts before copying their exact bytes
into the fresh V2.6 namespace.  Normal V2.6 readers rebuild their authority
from the tracked V2.6 source manifest plus those local snapshots and never
reopen the external parent.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from .pilot_budget import ParentBudgetDebit
from .pilot_contract import PilotContract
from .pilot_v24_parent_import import (
    PilotV24ParentImportError,
    _atomic_exact_bytes,
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
    _verify_parent_contract,
    _verify_parent_git,
    _verify_parent_ledgers,
    _verify_self_hash,
)
from .pilot_v25_parent_import import (
    PilotV25ParentImportError,
    V25_ALLOWED_P95_PROFILES,
    V25_CONTRACT_ID,
    V25_INHERITED_P95_RECEIPT_SCHEMA_VERSION,
    V25_PARENT_IMPORT_SCHEMA_VERSION,
    V25_SCIENCE_TAG,
    verified_v25_inherited_p95_binding,
    verify_v25_parent_import_receipt,
)


V26_CONTRACT_ID = "finevo-pilot-v2.6"
V26_SCIENCE_TAG = "pilot-v2.6-science"
V26_PARENT_IMPORT_SCHEMA_VERSION = "finevo-pilot-v2.6-parent-import-v1"
V26_INHERITED_P95_RECEIPT_SCHEMA_VERSION = (
    "finevo-pilot-v2.6-inherited-observed-p95-authority-v1"
)
V26_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.6-source-manifest-v1"
V26_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_6_source_manifest.json"
)
V26_EXPANDED_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_6.yaml")
V26_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.6/raw")
V26_ALLOWED_P95_PROFILES = V25_ALLOWED_P95_PROFILES

V25_CONTRACT_CANONICAL_SHA256 = (
    "1f9809062684a1a2afb96b7342b88a06810e0e87ac883aa63a858a65a81d188d"
)
V25_CONTRACT_FILE_SHA256 = (
    "c5ed0c5792cee405be365f62956b1e0718b533080477f84980a67f6e6d513ebc"
)
V25_SCIENCE_TAG_OBJECT = "fe5e33f221d431c85301e7d101497188018914c0"
V25_SCIENCE_COMMIT = "a3ec8d96162b50e41e7d4700e0534ce33c1958c3"
V25_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.5/raw")
V25_PARENT_IMPORT_RECEIPT_FILE_SHA256 = (
    "9c057bd85d9e8f6e0cd2d8262faab65664193ad80a2e270c541110c0572cea85"
)
V25_PARENT_IMPORT_RECEIPT_CONTENT_SHA256 = (
    "d19774e3113b88a2cfa210fa6c33bbc349d4b337c69dd513d3be3472fff27443"
)
V25_RUN_LEDGER_FILE_SHA256 = (
    "158aa234af442ff9b04520c2153f89213d6a1960bb57c6d997fa76febf496ca1"
)
V25_RUN_LEDGER_INTERNAL_SHA256 = (
    "7d223ddc2cc46b022f051217b9f6767bf9264fb66212b1a63a3498fb6447220f"
)
V25_RUN_LEDGER_EVENT_HEAD = (
    "a95b8af98d789259a0916ec9ad4599de331118bac80ebf7aaad74b3722c8c10a"
)
V25_BUDGET_LEDGER_FILE_SHA256 = (
    "a4ccc451e00e83b668406d911aeb78eb502d2ae1f1bca129d66dda3627fb3a3e"
)
V25_BUDGET_LEDGER_INTERNAL_SHA256 = (
    "7b448a0ebc002b932150c68f2c4e552e940ce186ea5e58afed8673af627d9162"
)
V25_BUDGET_LEDGER_EVENT_HEAD = (
    "c79501630cd45bf190f79346d75e71c1459a533cd407ae5d99950f3525d64065"
)
V26_PARENT_DEBIT_RECORD_SHA256 = (
    "4f445491738ea756280fca0b8c5c82823f4cefe7574cd368ed0c2c51c6a48802"
)
V26_SOURCE_MANIFEST_FILE_SHA256 = (
    "f84778ed279b8ca98b9b61e26619669fade54b95d0c3e4f17874733acbc84efe"
)
V26_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "78d42a49f16cbbee4fc5e76de17ff26c501a5dcb04a5eb1f79cbe080d2b1b669"
)
V25_RAW_INVENTORY_SHA256 = (
    "49645ac177b04e26655a44ee4c6c627cfaa54fd0d0c2c2448bb751c60904a541"
)

_V25_SOURCE_ROWS = {
    "gpt52_main": {
        "path": (
            "experiment_results/pilot-v2.5/raw/parent-import/observed_p95/"
            "gpt52_main/observed_p95_authority_receipt.json"
        ),
        "file_sha256": (
            "5fae513f4c082ecc21e867838188fd2bcac4af267955bf46b128931e74abcabb"
        ),
        "content_sha256": (
            "1fb036f8d0cfe56faa3206235716a195dbdf69b98447208a8ac54d8bef994632"
        ),
        "byte_size": 6310,
        "schema_version": V25_INHERITED_P95_RECEIPT_SCHEMA_VERSION,
        "runtime_model": "openai/gpt-5.2-2025-12-11",
        "served_model": "gpt-5.2-2025-12-11",
    },
    "llama33_local_controlled": {
        "path": (
            "experiment_results/pilot-v2.5/raw/parent-import/observed_p95/"
            "llama33_local_controlled/observed_p95_authority_receipt.json"
        ),
        "file_sha256": (
            "fc673570e4fef22055baf805f3304fd2fd3740f4760cdd401d4cef8a4d48db5b"
        ),
        "content_sha256": (
            "c5bd81af76fad87f600f777dc1d7c0417dbefc8eadfb3fff61317b3464636063"
        ),
        "byte_size": 6419,
        "schema_version": V25_INHERITED_P95_RECEIPT_SCHEMA_VERSION,
        "runtime_model": "ollama/llama3.3:70b-instruct-q4_K_M",
        "served_model": "llama3.3:70b-instruct-q4_K_M",
    },
}

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CLAIM_BOUNDARY = (
    "This V2.6 receipt imports only verified V2.5 budget and p95 authority. "
    "V2.5 remains a terminal no-go and contributes no treatment outcome."
)


class PilotV26ParentImportError(RuntimeError):
    """Raised before immutable V2.5 authority can enter V2.6."""


def _translate_v24(exc: PilotV24ParentImportError) -> PilotV26ParentImportError:
    return PilotV26ParentImportError(str(exc))


def _translate_v25(exc: PilotV25ParentImportError) -> PilotV26ParentImportError:
    return PilotV26ParentImportError(str(exc))


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotV26ParentImportError(f"{name} must be an object")
    return value


def _source_row(
    manifest: Mapping[str, Any],
    profile_id: str,
) -> dict[str, Any]:
    inventory = _mapping(
        manifest.get("inherited_observed_p95_authority"),
        name="V2.5 inherited p95 inventory",
    )
    row = _mapping(
        inventory.get(profile_id),
        name=f"{profile_id} V2.5 inherited p95 source",
    )
    receipt = _mapping(
        row.get("receipt"),
        name=f"{profile_id} V2.5 inherited p95 receipt",
    )
    return {
        **_json_copy(receipt),
        "runtime_model": row.get("runtime_model"),
        "served_model": row.get("served_model"),
    }


def _validate_source_manifest(value: Mapping[str, Any]) -> None:
    if set(value) != {
        "schema_version",
        "v2_5_terminal_parent",
        "v2_5_published_evidence",
        "inherited_observed_p95_authority",
        "parent_import_receipt",
        "cumulative_budget_debit",
        "cumulative_budget_debit_basis",
        "inheritance_policy",
        "integrity",
    }:
        raise PilotV26ParentImportError("V2.6 source manifest fields drifted")
    terminal = _mapping(
        value.get("v2_5_terminal_parent"),
        name="V2.5 terminal parent",
    )
    contract = _mapping(terminal.get("contract"), name="V2.5 parent contract")
    release = _mapping(terminal.get("release"), name="V2.5 parent release")
    if _json_copy(contract) != {
        "contract_id": V25_CONTRACT_ID,
        "path": "experiments/pilot_v2_5.yaml",
        "schema_version": "finevo-pilot-contract-v2",
        "status": "frozen",
        "file_sha256": V25_CONTRACT_FILE_SHA256,
        "canonical_sha256": V25_CONTRACT_CANONICAL_SHA256,
    } or _json_copy(release) != {
        "science_tag": V25_SCIENCE_TAG,
        "science_tag_object": V25_SCIENCE_TAG_OBJECT,
        "science_commit": V25_SCIENCE_COMMIT,
        "tag_kind": "annotated",
        "raw_root": V25_RAW_ROOT.as_posix(),
    }:
        raise PilotV26ParentImportError("V2.5 parent release identity drifted")

    ledgers = _mapping(terminal.get("ledgers"), name="V2.5 ledgers")
    expected_ledgers = {
        "run": {
            "path": f"{V25_RAW_ROOT.as_posix()}/run_ledger.json",
            "file_sha256": V25_RUN_LEDGER_FILE_SHA256,
            "internal_sha256": V25_RUN_LEDGER_INTERNAL_SHA256,
            "event_count": 213,
            "event_chain_head": V25_RUN_LEDGER_EVENT_HEAD,
        },
        "budget": {
            "path": f"{V25_RAW_ROOT.as_posix()}/budget_ledger.json",
            "file_sha256": V25_BUDGET_LEDGER_FILE_SHA256,
            "internal_sha256": V25_BUDGET_LEDGER_INTERNAL_SHA256,
            "event_count": 32,
            "event_chain_head": V25_BUDGET_LEDGER_EVENT_HEAD,
        },
    }
    if _json_copy(ledgers) != expected_ledgers:
        raise PilotV26ParentImportError("V2.5 ledger identity drifted")

    denominator = _mapping(
        terminal.get("terminal_denominator"),
        name="V2.5 terminal denominator",
    )
    if (
        denominator.get("registered_cells") != 211
        or denominator.get("scientific_cells") != 209
        or denominator.get("terminal_cells") != 211
        or denominator.get("all_rows_present") is not True
        or denominator.get("all_rows_terminal") is not True
        or denominator.get("status_counts")
        != {
            "complete": 2,
            "failed": 14,
            "integrity-stopped": 195,
        }
        or denominator.get("terminal_status") != "complete-with-no-go"
        or denominator.get("scientific_complete") is not False
        or denominator.get("scientific_matrix_complete") is not False
        or denominator.get("scientific_claim_gates_supported") is not False
    ):
        raise PilotV26ParentImportError("V2.5 terminal denominator drifted")

    inventory = _mapping(
        value.get("inherited_observed_p95_authority"),
        name="V2.5 inherited p95 inventory",
    )
    if set(inventory) != set(_V25_SOURCE_ROWS):
        raise PilotV26ParentImportError("V2.5 inherited p95 inventory drifted")
    for profile_id, expected in _V25_SOURCE_ROWS.items():
        row = _mapping(inventory[profile_id], name=f"{profile_id} p95 source")
        if (
            _source_row(value, profile_id) != expected
            or row.get("use")
            != (
                "v2.6-scientific-projection-authority-after-source-backed-"
                "v2.5-adapter-verification"
            )
        ):
            raise PilotV26ParentImportError(
                f"{profile_id} inherited p95 source drifted"
            )

    parent_receipt = _mapping(
        value.get("parent_import_receipt"),
        name="V2.5 parent-import receipt",
    )
    if (
        parent_receipt.get("path")
        != f"{V25_RAW_ROOT.as_posix()}/parent-import/parent_import_receipt.json"
        or parent_receipt.get("schema_version")
        != V25_PARENT_IMPORT_SCHEMA_VERSION
        or parent_receipt.get("file_sha256")
        != V25_PARENT_IMPORT_RECEIPT_FILE_SHA256
        or parent_receipt.get("content_sha256")
        != V25_PARENT_IMPORT_RECEIPT_CONTENT_SHA256
        or parent_receipt.get("byte_size") != 8326
        or parent_receipt.get("provider_calls") != 0
        or parent_receipt.get("scientific_evidence") is not False
        or parent_receipt.get(
            "scientific_outcomes_observed_before_amendment"
        )
        is not False
        or parent_receipt.get("contains_v2_4_or_v2_5_treatment_outcomes")
        is not False
    ):
        raise PilotV26ParentImportError("V2.5 parent-import receipt drifted")

    raw_snapshot = _mapping(
        terminal.get("raw_snapshot"),
        name="V2.5 raw-tree inventory",
    )
    if _json_copy(raw_snapshot) != {
        "root": V25_RAW_ROOT.as_posix(),
        "file_count": 61,
        "storage_bytes": 1_589_313,
        "inventory_schema_version": "finevo-raw-tree-inventory-v1",
        "inventory_canonicalization": "json-sort-keys-compact-utf8-v1",
        "inventory_sha256": V25_RAW_INVENTORY_SHA256,
    }:
        raise PilotV26ParentImportError("V2.5 raw-tree inventory drifted")

    try:
        debit = ParentBudgetDebit.from_dict(value["cumulative_budget_debit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PilotV26ParentImportError(
            "V2.6 cumulative parent debit is malformed"
        ) from exc
    if (
        debit.parent_contract_sha256 != V25_CONTRACT_CANONICAL_SHA256
        or debit.parent_run_ledger_sha256 != V25_RUN_LEDGER_INTERNAL_SHA256
        or debit.parent_budget_ledger_sha256
        != V25_BUDGET_LEDGER_INTERNAL_SHA256
        or debit.stage_bucket != "parent_v23"
        or not math.isclose(debit.cost_usd, 3.212770875, abs_tol=1e-12)
        or debit.hosted_completions != 184
        or debit.storage_bytes != 6_303_635
        or debit.record_sha256 != V26_PARENT_DEBIT_RECORD_SHA256
    ):
        raise PilotV26ParentImportError("V2.6 cumulative parent debit drifted")


def _load_source_manifest(repo_root: Path) -> tuple[dict[str, Any], bytes]:
    try:
        _, raw = _guarded_file(
            repo_root,
            V26_SOURCE_MANIFEST_PATH,
            name="V2.6 source manifest",
        )
        value = _strict_json(raw, name="V2.6 source manifest")
        _verify_self_hash(
            value,
            schema_version=V26_SOURCE_MANIFEST_SCHEMA_VERSION,
            name="V2.6 source manifest",
        )
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    _validate_source_manifest(value)
    if (
        _sha256(raw) != V26_SOURCE_MANIFEST_FILE_SHA256
        or value["integrity"]["content_sha256"]
        != V26_SOURCE_MANIFEST_CONTENT_SHA256
    ):
        raise PilotV26ParentImportError(
            "V2.6 source manifest differs from its code-bound hashes"
        )
    from . import pilot_contract as contract_module

    configured_file = contract_module.PILOT_V2_6_SOURCE_MANIFEST_FILE_SHA256
    configured_content = (
        contract_module.PILOT_V2_6_SOURCE_MANIFEST_CONTENT_SHA256
    )
    if configured_file not in {None, V26_SOURCE_MANIFEST_FILE_SHA256} or (
        configured_content not in {None, V26_SOURCE_MANIFEST_CONTENT_SHA256}
    ):
        raise PilotV26ParentImportError(
            "V2.6 contract constants bind a different source manifest"
        )
    return value, raw


def _validate_child_contract(
    contract: PilotContract,
    *,
    source_manifest: Mapping[str, Any],
    source_manifest_raw: bytes,
    require_frozen: bool = True,
) -> None:
    allowed_status = {"frozen"} if require_frozen else {"draft", "frozen"}
    if (
        contract.contract_id != V26_CONTRACT_ID
        or contract.status not in allowed_status
        or contract.implementation.get("required_git_tag") != V26_SCIENCE_TAG
    ):
        raise PilotV26ParentImportError(
            "parent import requires the V2.6 science contract"
        )
    amendment = _mapping(
        getattr(contract, "p95_authority_retry_amendment", None),
        name="V2.6 p95-authority retry amendment",
    )
    source = _mapping(
        amendment.get("source_manifest"),
        name="V2.6 source-manifest contract binding",
    )
    expected_hashes = (
        (None, None)
        if contract.status == "draft"
        else (
            V26_SOURCE_MANIFEST_FILE_SHA256,
            V26_SOURCE_MANIFEST_CONTENT_SHA256,
        )
    )
    if (
        source.get("path") != V26_SOURCE_MANIFEST_PATH.as_posix()
        or source.get("schema_version") != V26_SOURCE_MANIFEST_SCHEMA_VERSION
        or (source.get("file_sha256"), source.get("content_sha256"))
        != expected_hashes
        or _sha256(source_manifest_raw) != V26_SOURCE_MANIFEST_FILE_SHA256
        or source_manifest["integrity"]["content_sha256"]
        != V26_SOURCE_MANIFEST_CONTENT_SHA256
    ):
        raise PilotV26ParentImportError(
            "V2.6 contract does not bind the exact source manifest"
        )
    failure = _mapping(
        amendment.get("failure_classification"),
        name="V2.6 failure classification",
    )
    carry = _mapping(
        amendment.get("budget_carry_forward"),
        name="V2.6 budget carry-forward",
    )
    expected_cumulative = _json_copy(
        source_manifest["cumulative_budget_debit"]
    )
    expected_cumulative.pop("schema_version", None)
    if (
        failure.get("parent_contract_sha256")
        != V25_CONTRACT_CANONICAL_SHA256
        or failure.get("run_ledger_internal_sha256")
        != V25_RUN_LEDGER_INTERNAL_SHA256
        or failure.get("budget_ledger_internal_sha256")
        != V25_BUDGET_LEDGER_INTERNAL_SHA256
        or failure.get("status_counts")
        != source_manifest["v2_5_terminal_parent"][
            "terminal_denominator"
        ]["status_counts"]
        or _json_copy(carry.get("cumulative_prior"))
        != expected_cumulative
        or carry.get("budget_reset") is not False
    ):
        raise PilotV26ParentImportError(
            "V2.6 failure or cumulative budget binding drifted"
        )


def parent_budget_debit_for_v26(
    contract: PilotContract,
    *,
    repo_root: str | Path | None = None,
) -> ParentBudgetDebit | None:
    """Return the exact cumulative V2.5 boundary debit for V2.6."""

    if contract.contract_id != V26_CONTRACT_ID:
        return None
    root = _real_root(
        repo_root or Path(__file__).resolve().parents[1],
        name="child repository root",
    )
    manifest, raw = _load_source_manifest(root)
    _validate_child_contract(
        contract,
        source_manifest=manifest,
        source_manifest_raw=raw,
        require_frozen=False,
    )
    debit = ParentBudgetDebit.from_dict(manifest["cumulative_budget_debit"])
    if debit.record_sha256 != V26_PARENT_DEBIT_RECORD_SHA256:
        raise PilotV26ParentImportError("V2.6 parent debit record drifted")
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


def parent_snapshot_path(raw_root: str | Path, profile_id: str) -> Path:
    return (
        Path(raw_root)
        / "parent-import"
        / "parent_snapshots"
        / f"{profile_id}.v2_5_observed_p95_parent.json"
    )


def parent_import_receipt_path(raw_root: str | Path) -> Path:
    return Path(raw_root) / "parent-import" / "parent_import_receipt.json"


def _child_contract_binding(
    *,
    repo_root: Path,
    contract: PilotContract,
) -> dict[str, Any]:
    try:
        _, raw = _guarded_file(
            repo_root,
            V26_EXPANDED_CONTRACT_PATH,
            name="expanded V2.6 contract",
        )
        parsed = _strict_json(raw, name="expanded V2.6 contract")
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    parsed_contract = PilotContract.from_dict(parsed)
    if (
        parsed_contract.contract_id != contract.contract_id
        or parsed_contract.canonical_hash != contract.canonical_hash
        or parsed_contract.to_dict() != contract.to_dict()
    ):
        raise PilotV26ParentImportError(
            "expanded V2.6 contract differs from the selected contract"
        )
    return {
        "path": V26_EXPANDED_CONTRACT_PATH.as_posix(),
        "file_sha256": _sha256(raw),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
    }


def _build_child_p95_receipt(
    *,
    repo_root: Path,
    contract: PilotContract,
    child_git_tag: str,
    child_git_commit: str,
    source_manifest: Mapping[str, Any],
    source_manifest_raw: bytes,
    profile_id: str,
    parent_receipt: Mapping[str, Any],
    parent_snapshot: Path,
    parent_snapshot_raw: bytes,
) -> dict[str, Any]:
    if profile_id not in V26_ALLOWED_P95_PROFILES:
        raise PilotV26ParentImportError(
            f"{profile_id} has no V2.6 inherited dispatch authority"
        )
    source = _source_row(source_manifest, profile_id)
    model = _mapping(parent_receipt.get("model"), name="V2.5 p95 model")
    reservations = _mapping(
        parent_receipt.get("reservations"),
        name="V2.5 p95 reservations",
    )
    if (
        model.get("model_id") != profile_id
        or model.get("runtime_model") != source["runtime_model"]
        or model.get("served_model") != source["served_model"]
        or set(reservations) != {source["runtime_model"]}
    ):
        raise PilotV26ParentImportError(
            f"V2.5 p95 model identity drifted for {profile_id}"
        )
    transformed = _json_copy(reservations)
    for call_kind in ("action", "semantic"):
        entry = transformed[source["runtime_model"]].get(call_kind)
        if (
            not isinstance(entry, dict)
            or set(entry) != {"authority", "reservation"}
            or not isinstance(entry.get("authority"), dict)
        ):
            raise PilotV26ParentImportError(
                f"V2.5 p95 reservation drifted for {profile_id}/{call_kind}"
            )
        entry["authority"]["pilot_contract_hash"] = contract.canonical_hash
        entry["authority"]["pilot_tag"] = child_git_tag
    parent = source_manifest["v2_5_terminal_parent"]
    source_parent_receipt = source_manifest["parent_import_receipt"]
    return _seal(
        {
            "schema_version": V26_INHERITED_P95_RECEIPT_SCHEMA_VERSION,
            "contract": _child_contract_binding(
                repo_root=repo_root,
                contract=contract,
            ),
            "git": {"tag": child_git_tag, "commit": child_git_commit},
            "model": {
                "model_id": profile_id,
                "runtime_model": source["runtime_model"],
                "served_model": source["served_model"],
            },
            "parent_source": {
                "source_manifest_path": V26_SOURCE_MANIFEST_PATH.as_posix(),
                "source_manifest_file_sha256": _sha256(source_manifest_raw),
                "source_manifest_content_sha256": source_manifest[
                    "integrity"
                ]["content_sha256"],
                "parent_contract_sha256": parent["contract"][
                    "canonical_sha256"
                ],
                "parent_git_tag": parent["release"]["science_tag"],
                "parent_git_tag_object": parent["release"][
                    "science_tag_object"
                ],
                "parent_git_commit": parent["release"]["science_commit"],
                "parent_import_receipt_path": source_parent_receipt["path"],
                "parent_import_receipt_file_sha256": source_parent_receipt[
                    "file_sha256"
                ],
                "parent_import_receipt_content_sha256": source_parent_receipt[
                    "content_sha256"
                ],
                "parent_receipt_source_path": source["path"],
                "parent_receipt_snapshot_path": _repo_relative(
                    repo_root,
                    parent_snapshot,
                    name=f"{profile_id} V2.5 p95 snapshot",
                ),
                "parent_receipt_schema_version": source["schema_version"],
                "parent_receipt_file_sha256": _sha256(parent_snapshot_raw),
                "parent_receipt_content_sha256": source["content_sha256"],
            },
            "reservations": transformed,
            "scientific_evidence": False,
            "evidence_use": (
                "V2.6 prospective budget authority only; the V2.5 terminal "
                "no-go contributes no treatment effect."
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
    if profile_id not in V26_ALLOWED_P95_PROFILES:
        raise PilotV26ParentImportError(
            f"{profile_id} has no V2.6 inherited dispatch authority"
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
                "source_kind": "v2.5-terminal-parent-import-v2.6",
                "source_parent_manifest_content_sha256": child_receipt[
                    "parent_source"
                ]["source_manifest_content_sha256"],
                "source_authority_receipt": str(child_receipt_path),
                "source_authority_receipt_content_sha256": child_receipt[
                    "integrity"
                ]["content_sha256"],
            },
        }
    )


def _load_child_contract_from_receipt(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
) -> PilotContract:
    binding = _mapping(
        receipt.get("contract"),
        name="V2.6 inherited contract binding",
    )
    relative = _normalized_relative(
        binding.get("path", ""),
        required_top="experiments",
        name="V2.6 inherited contract path",
    )
    try:
        _, raw = _guarded_file(
            repo_root,
            relative,
            name="V2.6 inherited contract",
        )
        value = _strict_json(raw, name="V2.6 inherited contract")
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    if _sha256(raw) != binding.get("file_sha256"):
        raise PilotV26ParentImportError("V2.6 contract file hash drifted")
    contract = PilotContract.from_dict(value)
    if (
        contract.contract_id != binding.get("contract_id")
        or contract.canonical_hash != binding.get("contract_sha256")
    ):
        raise PilotV26ParentImportError("V2.6 contract identity drifted")
    return contract


def _verified_parent_snapshot(
    *,
    repo_root: Path,
    raw_root: Path,
    profile_id: str,
    source_manifest: Mapping[str, Any],
) -> tuple[Path, bytes, dict[str, Any]]:
    source = _source_row(source_manifest, profile_id)
    path = parent_snapshot_path(raw_root, profile_id)
    relative = _normalized_relative(
        _repo_relative(
            repo_root,
            path,
            name=f"{profile_id} V2.5 p95 snapshot",
        ),
        required_top="experiment_results",
        name=f"{profile_id} V2.5 p95 snapshot path",
    )
    try:
        guarded, raw = _guarded_file(
            repo_root,
            relative,
            name=f"{profile_id} V2.5 p95 snapshot",
        )
        receipt = _strict_json(raw, name=f"{profile_id} V2.5 p95 snapshot")
        _verify_self_hash(
            receipt,
            schema_version=V25_INHERITED_P95_RECEIPT_SCHEMA_VERSION,
            name=f"{profile_id} V2.5 p95 snapshot",
        )
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    if (
        guarded != path
        or len(raw) != source["byte_size"]
        or _sha256(raw) != source["file_sha256"]
        or receipt["integrity"]["content_sha256"]
        != source["content_sha256"]
    ):
        raise PilotV26ParentImportError(
            f"{profile_id} V2.5 p95 snapshot drifted"
        )
    return path, raw, receipt


def verify_v26_inherited_p95_receipt(
    receipt: Mapping[str, Any],
    *,
    repo_root: str | Path,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Rebuild a V2.6 receipt from tracked manifest and exact local snapshot."""

    root = _real_root(repo_root, name="child repository root")
    value = _json_copy(receipt)
    try:
        _verify_self_hash(
            value,
            schema_version=V26_INHERITED_P95_RECEIPT_SCHEMA_VERSION,
            name="V2.6 inherited p95 receipt",
        )
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    contract = _load_child_contract_from_receipt(value, repo_root=root)
    manifest, manifest_raw = _load_source_manifest(root)
    _validate_child_contract(
        contract,
        source_manifest=manifest,
        source_manifest_raw=manifest_raw,
    )
    git = _mapping(value.get("git"), name="V2.6 inherited git binding")
    model = _mapping(value.get("model"), name="V2.6 inherited model binding")
    if (
        git.get("tag") != V26_SCIENCE_TAG
        or git.get("commit") != expected_git_commit
        or _COMMIT_RE.fullmatch(expected_git_commit) is None
    ):
        raise PilotV26ParentImportError(
            "V2.6 inherited release binding is malformed"
        )
    profile_id = str(model.get("model_id"))
    if profile_id not in V26_ALLOWED_P95_PROFILES:
        raise PilotV26ParentImportError(
            f"{profile_id} has no V2.6 inherited dispatch authority"
        )
    raw_root = root.joinpath(*V26_RAW_ROOT.parts)
    snapshot, snapshot_raw, parent_receipt = _verified_parent_snapshot(
        repo_root=root,
        raw_root=raw_root,
        profile_id=profile_id,
        source_manifest=manifest,
    )
    expected = _build_child_p95_receipt(
        repo_root=root,
        contract=contract,
        child_git_tag=V26_SCIENCE_TAG,
        child_git_commit=expected_git_commit,
        source_manifest=manifest,
        source_manifest_raw=manifest_raw,
        profile_id=profile_id,
        parent_receipt=parent_receipt,
        parent_snapshot=snapshot,
        parent_snapshot_raw=snapshot_raw,
    )
    if value != expected:
        raise PilotV26ParentImportError(
            "V2.6 inherited p95 receipt differs from its tracked snapshot"
        )
    return _json_copy(value["reservations"])


def verified_v26_inherited_p95_binding(
    receipt_path: str | Path,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Return one guarded V2.6 receipt binding and verified reservations."""

    root = _real_root(repo_root, name="child repository root")
    path = Path(receipt_path)
    if path.is_absolute():
        try:
            relative = PurePosixPath(*path.absolute().relative_to(root).parts)
        except ValueError as exc:
            raise PilotV26ParentImportError(
                "V2.6 inherited p95 receipt escaped the repository"
            ) from exc
    else:
        relative = _normalized_relative(
            path,
            required_top="experiment_results",
            name="V2.6 inherited p95 receipt path",
        )
    try:
        _, raw = _guarded_file(
            root,
            relative,
            name="V2.6 inherited p95 receipt",
        )
        receipt = _strict_json(raw, name="V2.6 inherited p95 receipt")
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    reservations = verify_v26_inherited_p95_receipt(
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


def _rebuild_v26_parent_import_receipt(
    *,
    repo_root: Path,
    contract: PilotContract,
    expected_git_commit: str,
    source_manifest: Mapping[str, Any],
    source_manifest_raw: bytes,
    debit: ParentBudgetDebit,
) -> dict[str, Any]:
    raw_root = repo_root.joinpath(*V26_RAW_ROOT.parts)
    imported: dict[str, Any] = {}
    for profile_id in sorted(V26_ALLOWED_P95_PROFILES):
        source = _source_row(source_manifest, profile_id)
        snapshot, snapshot_raw, parent_receipt = _verified_parent_snapshot(
            repo_root=repo_root,
            raw_root=raw_root,
            profile_id=profile_id,
            source_manifest=source_manifest,
        )
        receipt_path = inherited_p95_receipt_path(raw_root, profile_id)
        projection_path = inherited_projection_path(raw_root, profile_id)
        try:
            receipt_relative = _normalized_relative(
                _repo_relative(
                    repo_root,
                    receipt_path,
                    name=f"{profile_id} V2.6 child authority",
                ),
                required_top="experiment_results",
                name=f"{profile_id} V2.6 child authority path",
            )
            projection_relative = _normalized_relative(
                _repo_relative(
                    repo_root,
                    projection_path,
                    name=f"{profile_id} V2.6 projection",
                ),
                required_top="experiment_results",
                name=f"{profile_id} V2.6 projection path",
            )
            _, receipt_raw = _guarded_file(
                repo_root,
                receipt_relative,
                name=f"{profile_id} V2.6 child authority",
            )
            _, projection_raw = _guarded_file(
                repo_root,
                projection_relative,
                name=f"{profile_id} V2.6 projection",
            )
            child = _strict_json(
                receipt_raw,
                name=f"{profile_id} V2.6 child authority",
            )
            projection = _strict_json(
                projection_raw,
                name=f"{profile_id} V2.6 projection",
            )
        except PilotV24ParentImportError as exc:
            raise _translate_v24(exc) from exc
        expected_child = _build_child_p95_receipt(
            repo_root=repo_root,
            contract=contract,
            child_git_tag=V26_SCIENCE_TAG,
            child_git_commit=expected_git_commit,
            source_manifest=source_manifest,
            source_manifest_raw=source_manifest_raw,
            profile_id=profile_id,
            parent_receipt=parent_receipt,
            parent_snapshot=snapshot,
            parent_snapshot_raw=snapshot_raw,
        )
        expected_projection = _build_child_projection(
            contract=contract,
            child_git_tag=V26_SCIENCE_TAG,
            child_git_commit=expected_git_commit,
            profile_id=profile_id,
            child_receipt=expected_child,
            child_receipt_path=receipt_path,
        )
        if child != expected_child or projection != expected_projection:
            raise PilotV26ParentImportError(
                f"V2.6 imported p95 artifact drifted for {profile_id}"
            )
        imported[profile_id] = {
            "parent_source_path": source["path"],
            "snapshot_path": _repo_relative(
                repo_root,
                snapshot,
                name=f"{profile_id} V2.5 p95 snapshot",
            ),
            "parent_file_sha256": source["file_sha256"],
            "parent_content_sha256": source["content_sha256"],
            "runtime_model": source["runtime_model"],
            "served_model": source["served_model"],
            "child_authority_receipt": receipt_relative.as_posix(),
            "child_authority_receipt_file_sha256": _sha256(receipt_raw),
            "child_authority_receipt_content_sha256": child["integrity"][
                "content_sha256"
            ],
            "child_projection": projection_relative.as_posix(),
            "child_projection_file_sha256": _sha256(projection_raw),
            "child_projection_content_sha256": projection["integrity"][
                "content_sha256"
            ],
        }
    terminal = source_manifest["v2_5_terminal_parent"]
    return _seal(
        {
            "schema_version": V26_PARENT_IMPORT_SCHEMA_VERSION,
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "child_release": {
                "git_tag": V26_SCIENCE_TAG,
                "git_commit": expected_git_commit,
            },
            "v2_5_parent_release": {
                "contract": _json_copy(terminal["contract"]),
                "release": _json_copy(terminal["release"]),
                "terminal_denominator": _json_copy(
                    terminal["terminal_denominator"]
                ),
                "raw_snapshot": _json_copy(terminal["raw_snapshot"]),
            },
            "source_manifest": {
                "path": V26_SOURCE_MANIFEST_PATH.as_posix(),
                "file_sha256": _sha256(source_manifest_raw),
                "content_sha256": source_manifest["integrity"][
                    "content_sha256"
                ],
            },
            "verified_v2_5_parent_import": _json_copy(
                source_manifest["parent_import_receipt"]
            ),
            "cumulative_budget_debit": debit.to_dict(),
            "parent_run_ledger": {
                "ledger_sha256": V25_RUN_LEDGER_INTERNAL_SHA256,
                "event_count": 213,
                "event_chain_head": V25_RUN_LEDGER_EVENT_HEAD,
            },
            "parent_budget_ledger": {
                "ledger_sha256": V25_BUDGET_LEDGER_INTERNAL_SHA256,
                "event_count": 32,
                "event_chain_head": V25_BUDGET_LEDGER_EVENT_HEAD,
            },
            "imported_projection_profiles": imported,
            "provider_calls": 0,
            "scientific_evidence": False,
            "scientific_outcomes_observed_before_amendment": False,
            "claim_boundary": _CLAIM_BOUNDARY,
        }
    )


def _parent_helper_manifest(
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    terminal = source_manifest["v2_5_terminal_parent"]
    contract = terminal["contract"]
    release = terminal["release"]
    return {
        "parent": {
            "contract_id": contract["contract_id"],
            "contract_path": contract["path"],
            "contract_file_sha256": contract["file_sha256"],
            "contract_canonical_sha256": contract["canonical_sha256"],
            "science_tag": release["science_tag"],
            "science_tag_object": release["science_tag_object"],
            "science_commit": release["science_commit"],
            "raw_root": release["raw_root"],
        },
        "ledgers": _json_copy(terminal["ledgers"]),
        "terminal_denominator": {
            "registered_cells": 211,
            "status_counts": {
                "complete": 2,
                "failed": 14,
                "integrity-stopped": 195,
            },
        },
        # This is what the V2.5 tamper-evident budget ledger commits.  The
        # V2.6 debit separately adds the complete 1,589,313-byte raw snapshot.
        "cumulative_budget_debit": {
            "cost_usd": 3.212770875,
            "hosted_completions": 184,
            "storage_bytes": 5_712_571,
        },
    }


def _verify_parent_raw_inventory(
    parent_root: Path,
    source_manifest: Mapping[str, Any],
) -> None:
    raw = parent_root.joinpath(*V25_RAW_ROOT.parts)
    if not raw.is_dir() or raw.is_symlink():
        raise PilotV26ParentImportError(
            "V2.5 immutable raw root is unavailable"
        )
    rows: list[dict[str, Any]] = []
    for path in sorted(raw.rglob("*")):
        if path.is_symlink():
            raise PilotV26ParentImportError(
                "V2.5 raw inventory contains a symlink"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise PilotV26ParentImportError(
                "V2.5 raw inventory contains a non-regular entry"
            )
        payload = path.read_bytes()
        rows.append(
            {
                "path": path.relative_to(raw).as_posix(),
                "byte_size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    canonical = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    expected = source_manifest["v2_5_terminal_parent"]["raw_snapshot"]
    if (
        len(rows) != expected["file_count"]
        or sum(row["byte_size"] for row in rows) != expected["storage_bytes"]
        or hashlib.sha256(canonical).hexdigest()
        != expected["inventory_sha256"]
    ):
        raise PilotV26ParentImportError(
            "V2.5 complete raw-tree inventory drifted"
        )


def _verify_published_evidence(
    child_root: Path,
    source_manifest: Mapping[str, Any],
) -> None:
    """Verify the complete tracked V2.5 no-go package before parent access."""

    published = _mapping(
        source_manifest.get("v2_5_published_evidence"),
        name="V2.5 published evidence binding",
    )
    if (
        published.get("root") != "evidence/current_v2/pilot-v2.5"
        or published.get("schema_version")
        != "finevo-pilot-v2.5-evidence-package-v1"
        or published.get("publication_status") != "complete-with-no-go"
        or published.get("contract_id") != V25_CONTRACT_ID
        or published.get("contract_sha256")
        != V25_CONTRACT_CANONICAL_SHA256
        or published.get("pilot_tag") != V25_SCIENCE_TAG
        or published.get("resolved_git_commit") != V25_SCIENCE_COMMIT
        or published.get("scientific_complete") is not False
        or published.get("scientific_matrix_complete") is not False
        or published.get("scientific_claim_gates_supported") is not False
    ):
        raise PilotV26ParentImportError(
            "V2.5 published evidence release boundary drifted"
        )

    loaded: dict[str, dict[str, Any]] = {}
    loaded_raw: dict[str, bytes] = {}
    for name in ("package_manifest", "checksums", "aggregate", "failure_ledger"):
        binding = _mapping(
            published.get(name),
            name=f"V2.5 published {name} binding",
        )
        relative = _normalized_relative(
            binding.get("path", ""),
            required_top="evidence",
            name=f"V2.5 published {name} path",
        )
        try:
            _, raw = _guarded_file(
                child_root,
                relative,
                name=f"V2.5 published {name}",
            )
            value = _strict_json(raw, name=f"V2.5 published {name}")
        except PilotV24ParentImportError as exc:
            raise _translate_v24(exc) from exc
        if (
            len(raw) != binding.get("byte_size")
            or _sha256(raw) != binding.get("file_sha256")
        ):
            raise PilotV26ParentImportError(
                f"V2.5 published {name} bytes drifted"
            )
        loaded[name] = value
        loaded_raw[name] = raw

    package = loaded["package_manifest"]
    if (
        package.get("schema_version")
        != "finevo-pilot-v2.5-evidence-package-v1"
        or package.get("contract_id") != V25_CONTRACT_ID
        or package.get("contract_sha256")
        != V25_CONTRACT_CANONICAL_SHA256
        or package.get("pilot_tag") != V25_SCIENCE_TAG
        or package.get("resolved_git_commit") != V25_SCIENCE_COMMIT
        or package.get("publication_status") != "complete-with-no-go"
        or package.get("scientific_complete") is not False
        or package.get("scientific_matrix_complete") is not False
        or package.get("scientific_claim_gates_supported") is not False
        or package.get("lane_separated") is not True
        or package.get("direction_counts_merged") is not False
        or package.get("narrative_status") != "deferred-unregistered"
    ):
        raise PilotV26ParentImportError(
            "V2.5 package manifest claim boundary drifted"
        )

    checksums = loaded["checksums"]
    rows = checksums.get("files")
    if (
        checksums.get("schema_version")
        != "finevo-pilot-package-checksums-v1"
        or checksums.get("contract_sha256")
        != V25_CONTRACT_CANONICAL_SHA256
        or not isinstance(rows, list)
        or not rows
    ):
        raise PilotV26ParentImportError(
            "V2.5 package checksum inventory is malformed"
        )
    checksum_paths: set[str] = set()
    for row_value in rows:
        row = _mapping(row_value, name="V2.5 package checksum row")
        relative_inside = _normalized_relative(
            row.get("path", ""),
            required_top=None,
            name="V2.5 package checksum relative path",
        )
        if relative_inside.as_posix() in checksum_paths:
            raise PilotV26ParentImportError(
                "V2.5 package checksum path is duplicated"
            )
        checksum_paths.add(relative_inside.as_posix())
        package_relative = _normalized_relative(
            f"{published['root']}/{relative_inside.as_posix()}",
            required_top="evidence",
            name="V2.5 checksummed package path",
        )
        try:
            _, raw = _guarded_file(
                child_root,
                package_relative,
                name=f"V2.5 package file {relative_inside.as_posix()}",
            )
        except PilotV24ParentImportError as exc:
            raise _translate_v24(exc) from exc
        if (
            len(raw) != row.get("byte_size")
            or _sha256(raw) != row.get("sha256")
        ):
            raise PilotV26ParentImportError(
                f"V2.5 package checksum drifted for "
                f"{relative_inside.as_posix()}"
            )
    if (
        set(package.get("published_files", ())) | {"package_manifest.json"}
        != checksum_paths
    ):
        raise PilotV26ParentImportError(
            "V2.5 package manifest and checksum inventory differ"
        )

    copied_binding = _mapping(
        published.get("copied_v2_5_source_manifest"),
        name="V2.5 copied source-manifest binding",
    )
    copied_relative = _normalized_relative(
        copied_binding.get("path", ""),
        required_top="evidence",
        name="V2.5 copied source-manifest path",
    )
    try:
        _, copied_raw = _guarded_file(
            child_root,
            copied_relative,
            name="V2.5 copied source manifest",
        )
        copied = _strict_json(
            copied_raw,
            name="V2.5 copied source manifest",
        )
        _verify_self_hash(
            copied,
            schema_version="finevo-pilot-v2.5-source-manifest-v1",
            name="V2.5 copied source manifest",
        )
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    if (
        _sha256(copied_raw) != copied_binding.get("file_sha256")
        or copied["integrity"]["content_sha256"]
        != copied_binding.get("content_sha256")
    ):
        raise PilotV26ParentImportError(
            "V2.5 copied source manifest drifted"
        )

    aggregate = loaded["aggregate"]
    denominator = _mapping(
        aggregate.get("denominator"),
        name="V2.5 aggregate denominator",
    )
    budget = _mapping(
        aggregate.get("budget"),
        name="V2.5 aggregate budget",
    )
    totals = _mapping(
        budget.get("actual_totals"),
        name="V2.5 aggregate budget totals",
    )
    release_controls = _mapping(
        aggregate.get("release_controls"),
        name="V2.5 aggregate release controls",
    )
    stage0 = _mapping(
        release_controls.get("stage0_selection"),
        name="V2.5 Stage-0 selection control",
    )
    if (
        aggregate.get("schema_version")
        != "finevo-pilot-v2.5-evidence-package-v1"
        or aggregate.get("contract_id") != V25_CONTRACT_ID
        or aggregate.get("contract_sha256")
        != V25_CONTRACT_CANONICAL_SHA256
        or aggregate.get("pilot_tag") != V25_SCIENCE_TAG
        or aggregate.get("resolved_git_commit") != V25_SCIENCE_COMMIT
        or aggregate.get("publication_status") != "complete-with-no-go"
        or aggregate.get("scientific_complete") is not False
        or aggregate.get("scientific_matrix_complete") is not False
        or aggregate.get("scientific_claim_gates_supported") is not False
        or denominator.get("expected_count") != 211
        or denominator.get("observed_ledger_count") != 211
        or denominator.get("all_rows_present") is not True
        or denominator.get("all_rows_terminal") is not True
        or denominator.get("status_counts")
        != {
            "complete": 2,
            "failed": 14,
            "integrity-stopped": 195,
        }
        or budget.get("pass") is not True
        or totals
        != {
            "cost_usd": 3.212770875,
            "completions": 184,
            "storage_bytes": 5_712_571,
        }
        or budget.get("raw_root_storage_bytes") != 1_589_313
        or any(value is not True for value in budget.get("checks", {}).values())
        or release_controls.get("pass") is not False
        or stage0.get("pass") is not False
        or not any(
            "required evidence file is missing" in str(reason)
            for reason in stage0.get("reasons", ())
        )
        or aggregate.get("narrative", {}).get("status")
        != "deferred-unregistered"
    ):
        raise PilotV26ParentImportError(
            "V2.5 aggregate no-go, budget, or Stage-0 boundary drifted"
        )

    failure = loaded["failure_ledger"]
    failure_denominator = _mapping(
        failure.get("denominator"),
        name="V2.5 failure-ledger denominator",
    )
    failure_rows = failure.get("rows")
    if (
        failure.get("schema_version") != "finevo-pilot-failure-ledger-v1"
        or failure.get("contract_sha256")
        != V25_CONTRACT_CANONICAL_SHA256
        or failure_denominator != denominator
        or not isinstance(failure_rows, list)
        or len(failure_rows) != 209
    ):
        raise PilotV26ParentImportError(
            "V2.5 failure ledger identity or denominator drifted"
        )
    direct = [row for row in failure_rows if row.get("status") == "failed"]
    blocked = [
        row for row in failure_rows if row.get("status") == "integrity-stopped"
    ]
    if (
        len(direct) != 14
        or len(blocked) != 195
        or any(row.get("stage_id") != "stage0-calibration" for row in direct)
        or any(
            row.get("failure", {}).get("message_sha256")
            != "39cb7f19f94e435d9eb4873df49beac2507703522f2ad9ffa7f688a5f6b92ef7"
            for row in direct
        )
        or any(
            row.get("failure", {}).get("source_stage")
            != "stage0-calibration"
            for row in blocked
        )
    ):
        raise PilotV26ParentImportError(
            "V2.5 Stage-0 failure propagation boundary drifted"
        )


def _persist_v26_parent_import_impl(
    *,
    contract: PilotContract,
    repo_root: str | Path,
    raw_root: str | Path,
    parent_repo_root: str | Path,
    child_git_tag: str,
    child_git_commit: str,
) -> dict[str, Any]:
    child_root = _real_root(repo_root, name="child repository root")
    parent_root = _real_root(parent_repo_root, name="V2.5 parent repository")
    child_raw = Path(raw_root).absolute()
    if (
        _repo_relative(child_root, child_raw, name="V2.6 raw root")
        != V26_RAW_ROOT.as_posix()
    ):
        raise PilotV26ParentImportError(
            "V2.6 import requires its fresh pilot-v2.6 raw namespace"
        )
    if (
        child_git_tag != V26_SCIENCE_TAG
        or _COMMIT_RE.fullmatch(child_git_commit) is None
    ):
        raise PilotV26ParentImportError(
            "V2.6 child release tag or commit is malformed"
        )
    manifest, manifest_raw = _load_source_manifest(child_root)
    _validate_child_contract(
        contract,
        source_manifest=manifest,
        source_manifest_raw=manifest_raw,
    )
    _verify_published_evidence(child_root, manifest)

    helper_manifest = _parent_helper_manifest(manifest)
    try:
        _verify_parent_git(parent_root, helper_manifest)
        parent_contract = _verify_parent_contract(
            parent_root,
            helper_manifest,
        )
        run_snapshot, budget_snapshot = _verify_parent_ledgers(
            parent_root,
            parent_contract=parent_contract,
            source_manifest=helper_manifest,
        )
        if (
            _git(parent_root, "rev-parse", "HEAD") != V25_SCIENCE_COMMIT
            or _git(
                parent_root,
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
            )
        ):
            raise PilotV26ParentImportError(
                "V2.5 parent HEAD or tracked worktree drifted"
            )
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    _verify_parent_raw_inventory(parent_root, manifest)

    parent_import = manifest["parent_import_receipt"]
    parent_import_relative = _normalized_relative(
        parent_import["path"],
        required_top="experiment_results",
        name="V2.5 parent-import receipt path",
    )
    try:
        _, parent_import_raw = _guarded_file(
            parent_root,
            parent_import_relative,
            name="V2.5 parent-import receipt",
        )
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    if (
        len(parent_import_raw) != parent_import["byte_size"]
        or _sha256(parent_import_raw) != parent_import["file_sha256"]
    ):
        raise PilotV26ParentImportError(
            "V2.5 parent-import receipt bytes drifted"
        )
    try:
        verified_parent_import = verify_v25_parent_import_receipt(
            parent_import_relative.as_posix(),
            repo_root=parent_root,
            contract=parent_contract,
            expected_git_commit=V25_SCIENCE_COMMIT,
        )
    except PilotV25ParentImportError as exc:
        raise _translate_v25(exc) from exc
    if (
        verified_parent_import["integrity"]["content_sha256"]
        != V25_PARENT_IMPORT_RECEIPT_CONTENT_SHA256
    ):
        raise PilotV26ParentImportError(
            "verified V2.5 parent-import receipt identity drifted"
        )

    for profile_id in sorted(V26_ALLOWED_P95_PROFILES):
        source = _source_row(manifest, profile_id)
        try:
            binding = verified_v25_inherited_p95_binding(
                source["path"],
                repo_root=parent_root,
                expected_git_commit=V25_SCIENCE_COMMIT,
            )
        except PilotV25ParentImportError as exc:
            raise _translate_v25(exc) from exc
        if (
            binding["receipt_file_sha256"] != source["file_sha256"]
            or binding["receipt_content_sha256"] != source["content_sha256"]
        ):
            raise PilotV26ParentImportError(
                f"{profile_id} verified V2.5 p95 binding drifted"
            )
        try:
            _, parent_raw = _guarded_file(
                parent_root,
                _normalized_relative(
                    source["path"],
                    required_top="experiment_results",
                    name=f"{profile_id} V2.5 p95 source path",
                ),
                name=f"{profile_id} V2.5 p95 source",
            )
        except PilotV24ParentImportError as exc:
            raise _translate_v24(exc) from exc
        snapshot = parent_snapshot_path(child_raw, profile_id)
        try:
            _atomic_exact_bytes(snapshot, parent_raw)
        except PilotV24ParentImportError as exc:
            raise _translate_v24(exc) from exc
        parent_receipt = _strict_json(
            parent_raw,
            name=f"{profile_id} V2.5 p95 source",
        )
        child = _build_child_p95_receipt(
            repo_root=child_root,
            contract=contract,
            child_git_tag=child_git_tag,
            child_git_commit=child_git_commit,
            source_manifest=manifest,
            source_manifest_raw=manifest_raw,
            profile_id=profile_id,
            parent_receipt=parent_receipt,
            parent_snapshot=snapshot,
            parent_snapshot_raw=parent_raw,
        )
        child_path = inherited_p95_receipt_path(child_raw, profile_id)
        projection = _build_child_projection(
            contract=contract,
            child_git_tag=child_git_tag,
            child_git_commit=child_git_commit,
            profile_id=profile_id,
            child_receipt=child,
            child_receipt_path=child_path,
        )
        try:
            _atomic_exact_json(child_path, child)
            _atomic_exact_json(
                inherited_projection_path(child_raw, profile_id),
                projection,
            )
        except PilotV24ParentImportError as exc:
            raise _translate_v24(exc) from exc

    if (
        run_snapshot["ledger_sha256"] != V25_RUN_LEDGER_INTERNAL_SHA256
        or len(run_snapshot["events"]) != 213
        or run_snapshot["events"][-1]["event_sha256"]
        != V25_RUN_LEDGER_EVENT_HEAD
        or budget_snapshot["ledger_sha256"]
        != V25_BUDGET_LEDGER_INTERNAL_SHA256
        or len(budget_snapshot["events"]) != 32
        or budget_snapshot["event_chain_head"]
        != V25_BUDGET_LEDGER_EVENT_HEAD
    ):
        raise PilotV26ParentImportError(
            "verified V2.5 ledger snapshots drifted"
        )
    debit = parent_budget_debit_for_v26(contract, repo_root=child_root)
    if debit is None:  # pragma: no cover
        raise PilotV26ParentImportError("V2.6 parent debit is unavailable")
    receipt = _rebuild_v26_parent_import_receipt(
        repo_root=child_root,
        contract=contract,
        expected_git_commit=child_git_commit,
        source_manifest=manifest,
        source_manifest_raw=manifest_raw,
        debit=debit,
    )
    receipt_path = parent_import_receipt_path(child_raw)
    try:
        _atomic_exact_json(receipt_path, receipt)
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    verified = verify_v26_parent_import_receipt(
        receipt_path,
        repo_root=child_root,
        contract=contract,
        expected_git_commit=child_git_commit,
    )
    if verified != receipt:
        raise PilotV26ParentImportError(
            "persisted V2.6 parent import differs after verification"
        )
    return {
        "receipt": str(receipt_path),
        "receipt_content_sha256": receipt["integrity"]["content_sha256"],
        "provider_calls": 0,
        "scientific_evidence": False,
        "imported_profiles": sorted(V26_ALLOWED_P95_PROFILES),
    }


def persist_v26_parent_import(**kwargs: Any) -> dict[str, Any]:
    """Verify immutable V2.5 and persist one idempotent zero-call import."""

    try:
        return _persist_v26_parent_import_impl(**kwargs)
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    except PilotV25ParentImportError as exc:
        raise _translate_v25(exc) from exc


def verify_v26_parent_import_receipt(
    receipt_or_path: Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path,
    contract: PilotContract,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Verify the V2.6 import using only tracked/local V2.6 artifacts."""

    root = _real_root(repo_root, name="child repository root")
    if _COMMIT_RE.fullmatch(expected_git_commit) is None:
        raise PilotV26ParentImportError(
            "V2.6 expected commit must be 40 lowercase hex characters"
        )
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
                raise PilotV26ParentImportError(
                    "V2.6 parent-import receipt escaped the repository"
                ) from exc
        else:
            relative = _normalized_relative(
                path,
                required_top="experiment_results",
                name="V2.6 parent-import receipt path",
            )
        try:
            _, raw = _guarded_file(
                root,
                relative,
                name="V2.6 parent-import receipt",
            )
            value = _strict_json(raw, name="V2.6 parent-import receipt")
        except PilotV24ParentImportError as exc:
            raise _translate_v24(exc) from exc
    try:
        _verify_self_hash(
            value,
            schema_version=V26_PARENT_IMPORT_SCHEMA_VERSION,
            name="V2.6 parent-import receipt",
        )
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    manifest, manifest_raw = _load_source_manifest(root)
    _validate_child_contract(
        contract,
        source_manifest=manifest,
        source_manifest_raw=manifest_raw,
    )
    if (
        value.get("contract_id") != contract.contract_id
        or value.get("contract_sha256") != contract.canonical_hash
    ):
        raise PilotV26ParentImportError(
            "V2.6 parent-import contract identity drifted"
        )
    debit = parent_budget_debit_for_v26(contract, repo_root=root)
    if debit is None:  # pragma: no cover
        raise PilotV26ParentImportError("V2.6 parent debit is unavailable")
    expected = _rebuild_v26_parent_import_receipt(
        repo_root=root,
        contract=contract,
        expected_git_commit=expected_git_commit,
        source_manifest=manifest,
        source_manifest_raw=manifest_raw,
        debit=debit,
    )
    if value != expected:
        raise PilotV26ParentImportError(
            "V2.6 parent-import receipt differs from its sealed sources"
        )
    return value


__all__ = [
    "PilotV26ParentImportError",
    "V26_ALLOWED_P95_PROFILES",
    "V26_CONTRACT_ID",
    "V26_INHERITED_P95_RECEIPT_SCHEMA_VERSION",
    "V26_PARENT_IMPORT_SCHEMA_VERSION",
    "V26_PARENT_DEBIT_RECORD_SHA256",
    "V26_RAW_ROOT",
    "V26_SCIENCE_TAG",
    "V26_SOURCE_MANIFEST_CONTENT_SHA256",
    "V26_SOURCE_MANIFEST_FILE_SHA256",
    "V26_SOURCE_MANIFEST_PATH",
    "inherited_p95_receipt_path",
    "inherited_projection_path",
    "parent_budget_debit_for_v26",
    "parent_import_receipt_path",
    "parent_snapshot_path",
    "persist_v26_parent_import",
    "verified_v26_inherited_p95_binding",
    "verify_v26_inherited_p95_receipt",
    "verify_v26_parent_import_receipt",
]
