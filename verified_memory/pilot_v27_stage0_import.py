"""Immutable V2.6 -> V2.7 Stage-0 source import primitives.

V2.6 is a terminal ``complete-with-no-go`` release.  V2.7 may reuse only the
sixteen cells that completed before that no-go (parent import, q-ref
resolution, and fourteen Stage-0 calibration cells).  This module verifies
the complete frozen V2.6 release, constructs the tracked V2.7 source manifest,
copies the *entire* V2.6 raw tree byte-for-byte into the fresh V2.7 namespace,
and seals a zero-provider-call import receipt.

The copied manifests and journals retain their V2.6 identities.  They are
source evidence for the corrected offline Stage-0 reader; they are never
rewritten or relabelled as V2.7 treatment outcomes.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from .artifacts import verify_manifest
from .pilot_budget import ParentBudgetDebit
from .pilot_contract import PilotContract, PilotRunSpec, load_pilot_contract
from .pilot_v24_parent_import import (
    CANONICALIZATION,
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
from .pilot_v26_parent_import import (
    PilotV26ParentImportError,
    verify_v26_parent_import_receipt,
)
from .runner import verify_provider_call_journal
from .runner_artifacts import load_verified_run_artifacts


V27_CONTRACT_ID = "finevo-pilot-v2.7"
V27_SCIENCE_TAG = "pilot-v2.7-science"
V27_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.7-source-manifest-v1"
V27_PARENT_IMPORT_SCHEMA_VERSION = "finevo-pilot-v2.7-parent-import-v1"
V27_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_7_source_manifest.json"
)
V27_EXPANDED_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_7.yaml")
V27_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.7/raw")
V27_SNAPSHOT_RELATIVE = PurePosixPath(
    "parent-import/v2_6_raw_snapshot"
)

V26_CONTRACT_ID = "finevo-pilot-v2.6"
V26_SCIENCE_TAG = "pilot-v2.6-science"
V26_SCIENCE_TAG_OBJECT = "ff0ac3882dbb06fd5ad61694888249829f61f903"
V26_SCIENCE_COMMIT = "0f59a15bc2cc3cce68f64de1dc1be78f7d74e214"
V26_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_6.yaml")
V26_CONTRACT_FILE_SHA256 = (
    "bb23062851c7b0014e9c1b07527c77be5bd70e15c492e2097cef2e4654b9cbea"
)
V26_CONTRACT_CANONICAL_SHA256 = (
    "bb6b12d71227c423e5a67452dc496f26843dec74e359b9b04bf096dc17d0c509"
)
V26_SOURCE_MANIFEST_FILE_SHA256 = (
    "f84778ed279b8ca98b9b61e26619669fade54b95d0c3e4f17874733acbc84efe"
)
V26_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "78d42a49f16cbbee4fc5e76de17ff26c501a5dcb04a5eb1f79cbe080d2b1b669"
)
V26_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.6/raw")
V26_RAW_FILE_COUNT = 228
V26_RAW_STORAGE_BYTES = 12_877_797
V26_RAW_INVENTORY_SHA256 = (
    "cfc3365828ba8fe2f75f11bb5117d8282d8f68f52653c411605dd3849f712105"
)

V26_RUN_LEDGER_FILE_SHA256 = (
    "61ce4ff174934edadbb317f606e034d8cac5bcf13b4f0ad968029c25c7f358a8"
)
V26_RUN_LEDGER_INTERNAL_SHA256 = (
    "cca42a01c6685994fa8b22e0fc7c7fb2067e4b1973fd37acf7c47be4591337d4"
)
V26_RUN_LEDGER_EVENT_COUNT = 213
V26_RUN_LEDGER_EVENT_HEAD = (
    "9b000791ad5d44ab9514386e10d50a9436c79d031314a17a835bf12071dbc725"
)
V26_BUDGET_LEDGER_FILE_SHA256 = (
    "242b92e994b1b3a172328928f71620138f7221e3c6c4542070dca3fa9a8d930b"
)
V26_BUDGET_LEDGER_INTERNAL_SHA256 = (
    "73f218feb368c5770908a4107d78037e11871e1c68e1fdd8461b94953549cdba"
)
V26_BUDGET_LEDGER_EVENT_COUNT = 32
V26_BUDGET_LEDGER_EVENT_HEAD = (
    "883509bd06c7a7121716b1f1f89513fb67ef6dcf4a5d6b9de4f526b496fd15c3"
)
V26_BUDGET_COMMITTED_STORAGE_BYTES = 18_616_402

V26_LAUNCH_FILE_SHA256 = (
    "ad3087fef4f9c75bf56108f6d84506771a6f69ee6942cfef8e17a90845045c5a"
)
V26_LAUNCH_CONTENT_SHA256 = (
    "bb680770c0cd41ff012bc764d1f9e065a715ba8b0ef2dd19a9f518ae2a3a2f54"
)
V26_RELEASE_FILE_SHA256 = (
    "d89a1173d89ed433f93112946c1650550a08b145b2fa17a540278dc84b6c00fe"
)
V26_RELEASE_CONTENT_SHA256 = (
    "2423fa52677dba04fbeaa66d2b873c265ca0590f7e4e832922feb8dd3814be72"
)
V26_PARENT_IMPORT_FILE_SHA256 = (
    "9f641e215a8d41eb58dc48d2a7b0eca45cf065255834c4bdd61eed4d168140aa"
)
V26_PARENT_IMPORT_CONTENT_SHA256 = (
    "cca56ef19258b13c380bfc395d18f902b75b9a7fc4874c0ac35a2cfcb7f83a85"
)
V26_PARENT_STAGE_RECEIPT_FILE_SHA256 = (
    "484d6f514c99026aa94f5facb5eddd8b5225768b1532b1872e23cc1814cd29f2"
)
V26_PARENT_STAGE_RECEIPT_CONTENT_SHA256 = (
    "248b3062b79996c0d554ffe28a7d6c3f376ffd316a734b2058b735caf5b371c2"
)
V26_QREF_FILE_SHA256 = (
    "bf8cba5fd34a30b3b78b681f5be8bd617d3c43f7714f5876166a9a92734ed454"
)
V26_QREF_CONTENT_SHA256 = (
    "cbecfcfaf9a85badb049f0e5024d9ffe896369432b5982b97a71dad6970830f2"
)
V26_QREF_VALUE = 63.50397933257746
V26_QREF_MANIFEST_FILE_SHA256 = (
    "8d299f2cb1646a810eb4311cbf054acd72c643694925a7cc6043dda1f7201d2b"
)
V26_QREF_STAGE_RECEIPT_FILE_SHA256 = (
    "b3de05f77bbc0b533b75d8f302919ae57b6b1c12a3af2f6d525516ab91497107"
)
V26_QREF_STAGE_RECEIPT_CONTENT_SHA256 = (
    "afa5b7be5b31a70e0dc068d7ba6c7c35ebb70ec29977a19e7e41960ca84cc304"
)
V26_STAGE0_CATALOG_FILE_SHA256 = (
    "800b8aed26bbe573d6894696a36f825093aa02a7a6d7db1468e84ba06751f7e7"
)
V26_STAGE0_RECEIPT_FILE_SHA256 = (
    "56392110d442896c72732da999bb67b879d685ef4c7c8dd3add75b62abb92359"
)
V26_STAGE0_RECEIPT_CONTENT_SHA256 = (
    "615394abdd55f1f1cdbd2c9a52df2b6a9f91ef3888ab58d2952bb95777ca23c4"
)
V26_STAGE0_FAILURE_MESSAGE = "run has no pre-shock utility observations"

V26_EVIDENCE_ROOT = PurePosixPath("evidence/current_v2/pilot-v2.6")
V26_EVIDENCE_CHECKSUMS_FILE_SHA256 = (
    "fe8f900dad011cafbf23174c0653d917c16c5f3d2880ed867e042683ca07f45b"
)
V26_EVIDENCE_PACKAGE_FILE_SHA256 = (
    "92506741a82ca24f4038a089e97cf77aac85bfe3d3054062d68f780505f31e72"
)
V26_EVIDENCE_AGGREGATE_FILE_SHA256 = (
    "2d169823113b4e89183b0ce1ba34560c982718c3532f2fb37f8f0dfa5b7f5af7"
)
V26_EVIDENCE_FAILURE_FILE_SHA256 = (
    "6330249ef2d1587cd879939b953c02f8edf72408c6444d48f08cd059233f0e8a"
)

V27_CUMULATIVE_DEBIT = ParentBudgetDebit(
    parent_contract_sha256=V26_CONTRACT_CANONICAL_SHA256,
    parent_run_ledger_sha256=V26_RUN_LEDGER_INTERNAL_SHA256,
    parent_budget_ledger_sha256=V26_BUDGET_LEDGER_INTERNAL_SHA256,
    stage_bucket="parent_v23",
    cost_usd=3.212770875,
    hosted_completions=184,
    storage_bytes=19_181_432,
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_IMPORTABLE_STAGES = frozenset(
    {"parent-import", "q-ref-resolution", "stage0-calibration"}
)
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
V27_ALLOWED_P95_PROFILES = ("gpt52_main", "llama33_local_controlled")
_V26_P95_SOURCES = {
    "gpt52_main": {
        "runtime_model": "openai/gpt-5.2-2025-12-11",
        "served_model": "gpt-5.2-2025-12-11",
        "authority": {
            "path": (
                "experiment_results/pilot-v2.6/raw/parent-import/"
                "observed_p95/gpt52_main/observed_p95_authority_receipt.json"
            ),
            "schema_version":
            "finevo-pilot-v2.6-inherited-observed-p95-authority-v1",
            "file_sha256":
            "8e4354de7f0b6365acf15b023952f6a93570c9e03ad0ccef475dd1477bade3df",
            "content_sha256":
            "e5c0a0ab61826b14fcf75dc22b6713e0d62233411e33747d10301ffcfa95d9c6",
        },
        "projection": {
            "path": (
                "experiment_results/pilot-v2.6/raw/parent-import/"
                "observed_p95/gpt52_main/projection_p95.json"
            ),
            "schema_version": "finevo-pilot-projection-p95-v1",
            "file_sha256":
            "73a3be859d6d2b33290923d5f31aafd5bb30637572fc5c58282b6fb8239abeea",
            "content_sha256":
            "aa1ac22be1010a62bfd0efd3b2794c4c9afdbdca67a03a2ff84d50e2ddc080ce",
        },
    },
    "llama33_local_controlled": {
        "runtime_model": "ollama/llama3.3:70b-instruct-q4_K_M",
        "served_model": "llama3.3:70b-instruct-q4_K_M",
        "authority": {
            "path": (
                "experiment_results/pilot-v2.6/raw/parent-import/observed_p95/"
                "llama33_local_controlled/observed_p95_authority_receipt.json"
            ),
            "schema_version":
            "finevo-pilot-v2.6-inherited-observed-p95-authority-v1",
            "file_sha256":
            "1dc2e0059b11824fcb0b4ea460f13043986903c4499f138c85852764bbe034ef",
            "content_sha256":
            "d64fe792f9fb22dfcbbd4ce7bb44f576a7a7568a22cf027686743f9ac0896715",
        },
        "projection": {
            "path": (
                "experiment_results/pilot-v2.6/raw/parent-import/observed_p95/"
                "llama33_local_controlled/projection_p95.json"
            ),
            "schema_version": "finevo-pilot-projection-p95-v1",
            "file_sha256":
            "b90e3af2cd6de30a7061a16173d2a415e921ad7016368063dc4977bdbf39b8a6",
            "content_sha256":
            "414ff8f4254a978baf66079d1ad95909638e7be4deae5227ee683190faa4b399",
        },
    },
}


class PilotV27Stage0ImportError(RuntimeError):
    """Raised before immutable V2.6 Stage-0 authority can enter V2.7."""


def _translate(exc: Exception) -> PilotV27Stage0ImportError:
    return PilotV27Stage0ImportError(str(exc))


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotV27Stage0ImportError(f"{name} must be an object")
    return value


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
        raise PilotV27Stage0ImportError(f"{name} file hash drifted")
    return path, raw, value


def _sealed_file(
    root: Path,
    relative: PurePosixPath,
    *,
    name: str,
    schema_version: str,
    file_sha256: str,
    content_sha256: str,
    verify_canonical_self_hash: bool = True,
) -> tuple[Path, bytes, dict[str, Any]]:
    path, raw, value = _strict_file(
        root,
        relative,
        name=name,
        expected_sha256=file_sha256,
    )
    if verify_canonical_self_hash:
        try:
            _verify_self_hash(value, schema_version=schema_version, name=name)
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
    integrity = value.get("integrity")
    if (
        value.get("schema_version") != schema_version
        or not isinstance(integrity, Mapping)
        or integrity.get("canonicalization") != CANONICALIZATION
        or integrity.get("content_sha256") != content_sha256
    ):
        raise PilotV27Stage0ImportError(f"{name} content hash drifted")
    return path, raw, value


def _inventory(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise PilotV27Stage0ImportError("raw snapshot root is unavailable")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PilotV27Stage0ImportError(
                "raw snapshot inventory contains a symlink"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise PilotV27Stage0ImportError(
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
            raise PilotV27Stage0ImportError(
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
    summary = {
        "root": V26_RAW_ROOT.as_posix(),
        "inventory_schema_version": "finevo-raw-tree-inventory-v1",
        "inventory_canonicalization": "json-sort-keys-compact-utf8-v1",
        "file_count": len(rows),
        "storage_bytes": sum(row["byte_size"] for row in rows),
        "inventory_sha256": _sha256(canonical),
    }
    return rows, summary


def _verify_exact_v26_inventory(raw_root: Path) -> list[dict[str, Any]]:
    rows, summary = _inventory(raw_root)
    if summary != {
        "root": V26_RAW_ROOT.as_posix(),
        "inventory_schema_version": "finevo-raw-tree-inventory-v1",
        "inventory_canonicalization": "json-sort-keys-compact-utf8-v1",
        "file_count": V26_RAW_FILE_COUNT,
        "storage_bytes": V26_RAW_STORAGE_BYTES,
        "inventory_sha256": V26_RAW_INVENTORY_SHA256,
    }:
        raise PilotV27Stage0ImportError(
            "V2.6 complete raw-tree inventory drifted"
        )
    return rows


def _normalized_spec(spec: PilotRunSpec | Mapping[str, Any]) -> dict[str, Any]:
    value = spec.to_dict() if isinstance(spec, PilotRunSpec) else _json_copy(spec)
    value.pop("contract_id", None)
    value.pop("run_id", None)
    return value


def _spec_pairs(
    source_contract: PilotContract,
    target_contract: PilotContract,
) -> list[tuple[PilotRunSpec, PilotRunSpec]]:
    if (
        source_contract.contract_id != V26_CONTRACT_ID
        or target_contract.contract_id != V27_CONTRACT_ID
    ):
        raise PilotV27Stage0ImportError("source/target contract identity drifted")
    source = {
        json.dumps(_normalized_spec(spec), sort_keys=True, allow_nan=False): spec
        for spec in source_contract.expand()
        if spec.stage_id in _IMPORTABLE_STAGES
    }
    target = {
        json.dumps(_normalized_spec(spec), sort_keys=True, allow_nan=False): spec
        for spec in target_contract.expand()
        if spec.stage_id in _IMPORTABLE_STAGES
    }
    if len(source) != 16 or len(target) != 16 or set(source) != set(target):
        raise PilotV27Stage0ImportError(
            "V2.6/V2.7 imported-cell matrix is not an exact normalized match"
        )
    return [(source[key], target[key]) for key in sorted(source)]


def _parent_helper_manifest() -> dict[str, Any]:
    return {
        "parent": {
            "contract_id": V26_CONTRACT_ID,
            "contract_path": V26_CONTRACT_PATH.as_posix(),
            "contract_file_sha256": V26_CONTRACT_FILE_SHA256,
            "contract_canonical_sha256": V26_CONTRACT_CANONICAL_SHA256,
            "science_tag": V26_SCIENCE_TAG,
            "science_tag_object": V26_SCIENCE_TAG_OBJECT,
            "science_commit": V26_SCIENCE_COMMIT,
            "raw_root": V26_RAW_ROOT.as_posix(),
        },
        "ledgers": {
            "run": {
                "path": f"{V26_RAW_ROOT.as_posix()}/run_ledger.json",
                "file_sha256": V26_RUN_LEDGER_FILE_SHA256,
                "internal_sha256": V26_RUN_LEDGER_INTERNAL_SHA256,
                "event_count": V26_RUN_LEDGER_EVENT_COUNT,
                "event_chain_head": V26_RUN_LEDGER_EVENT_HEAD,
            },
            "budget": {
                "path": f"{V26_RAW_ROOT.as_posix()}/budget_ledger.json",
                "file_sha256": V26_BUDGET_LEDGER_FILE_SHA256,
                "internal_sha256": V26_BUDGET_LEDGER_INTERNAL_SHA256,
                "event_count": V26_BUDGET_LEDGER_EVENT_COUNT,
                "event_chain_head": V26_BUDGET_LEDGER_EVENT_HEAD,
            },
        },
        "terminal_denominator": {
            "registered_cells": 211,
            "status_counts": {
                "complete": 16,
                "integrity-stopped": 195,
            },
        },
        "cumulative_budget_debit": {
            "cost_usd": 3.212770875,
            "hosted_completions": 184,
            "storage_bytes": V26_BUDGET_COMMITTED_STORAGE_BYTES,
        },
    }


def _verify_v26_published_evidence(child_root: Path) -> dict[str, Any]:
    bindings = {
        "checksums.json": V26_EVIDENCE_CHECKSUMS_FILE_SHA256,
        "package_manifest.json": V26_EVIDENCE_PACKAGE_FILE_SHA256,
        "aggregate.json": V26_EVIDENCE_AGGREGATE_FILE_SHA256,
        "failure_ledger.json": V26_EVIDENCE_FAILURE_FILE_SHA256,
    }
    loaded: dict[str, dict[str, Any]] = {}
    for name, expected in bindings.items():
        _, _, loaded[name] = _strict_file(
            child_root,
            V26_EVIDENCE_ROOT / name,
            name=f"published V2.6 {name}",
            expected_sha256=expected,
        )

    package = loaded["package_manifest.json"]
    if (
        package.get("schema_version")
        != "finevo-pilot-v2.6-evidence-package-v1"
        or package.get("contract_id") != V26_CONTRACT_ID
        or package.get("contract_sha256") != V26_CONTRACT_CANONICAL_SHA256
        or package.get("pilot_tag") != V26_SCIENCE_TAG
        or package.get("resolved_git_commit") != V26_SCIENCE_COMMIT
        or package.get("publication_status") != "complete-with-no-go"
        or package.get("scientific_complete") is not False
        or package.get("scientific_matrix_complete") is not False
        or package.get("scientific_claim_gates_supported") is not False
        or package.get("lane_separated") is not True
        or package.get("direction_counts_merged") is not False
        or package.get("narrative_status") != "deferred-unregistered"
    ):
        raise PilotV27Stage0ImportError(
            "published V2.6 evidence claim boundary drifted"
        )

    checksums = loaded["checksums.json"]
    rows = checksums.get("files")
    if (
        checksums.get("schema_version")
        != "finevo-pilot-package-checksums-v1"
        or checksums.get("contract_sha256")
        != V26_CONTRACT_CANONICAL_SHA256
        or not isinstance(rows, list)
        or len(rows) != 13
    ):
        raise PilotV27Stage0ImportError(
            "published V2.6 checksum inventory is malformed"
        )
    observed: set[str] = set()
    for row_value in rows:
        row = _mapping(row_value, name="published V2.6 checksum row")
        relative = _normalized_relative(
            str(row.get("path", "")),
            required_top=None,
            name="published V2.6 checksum path",
        )
        if relative.as_posix() in observed:
            raise PilotV27Stage0ImportError(
                "published V2.6 checksum path is duplicated"
            )
        observed.add(relative.as_posix())
        try:
            _, raw = _guarded_file(
                child_root,
                V26_EVIDENCE_ROOT / relative,
                name=f"published V2.6 file {relative.as_posix()}",
            )
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
        if (
            len(raw) != row.get("byte_size")
            or _sha256(raw) != row.get("sha256")
        ):
            raise PilotV27Stage0ImportError(
                f"published V2.6 checksum drifted for {relative.as_posix()}"
            )
    if observed != set(package.get("published_files", ())) | {
        "package_manifest.json"
    }:
        raise PilotV27Stage0ImportError(
            "published V2.6 package/checksum inventory differs"
        )

    aggregate = loaded["aggregate.json"]
    denominator = _mapping(
        aggregate.get("denominator"), name="V2.6 aggregate denominator"
    )
    budget = _mapping(aggregate.get("budget"), name="V2.6 aggregate budget")
    stage0 = _mapping(
        _mapping(
            aggregate.get("release_controls"),
            name="V2.6 release controls",
        ).get("stage0_selection"),
        name="V2.6 Stage-0 selection control",
    )
    if (
        aggregate.get("publication_status") != "complete-with-no-go"
        or denominator.get("expected_count") != 211
        or denominator.get("observed_ledger_count") != 211
        or denominator.get("all_rows_present") is not True
        or denominator.get("all_rows_terminal") is not True
        or denominator.get("status_counts")
        != {"complete": 16, "integrity-stopped": 195}
        or budget.get("pass") is not True
        or budget.get("raw_root_storage_bytes") != V26_RAW_STORAGE_BYTES
        or budget.get("actual_totals")
        != {
            "cost_usd": 3.212770875,
            "completions": 184,
            "storage_bytes": V26_BUDGET_COMMITTED_STORAGE_BYTES,
        }
        or stage0.get("pass") is not False
    ):
        raise PilotV27Stage0ImportError(
            "published V2.6 denominator, budget, or no-go drifted"
        )
    failure = loaded["failure_ledger.json"]
    if (
        failure.get("schema_version") != "finevo-pilot-failure-ledger-v1"
        or failure.get("contract_sha256")
        != V26_CONTRACT_CANONICAL_SHA256
        or failure.get("denominator") != denominator
        or not isinstance(failure.get("rows"), list)
        or len(failure["rows"]) != 195
        or any(
            row.get("status") != "integrity-stopped"
            for row in failure["rows"]
        )
    ):
        raise PilotV27Stage0ImportError(
            "published V2.6 failure denominator drifted"
        )
    return {
        "root": V26_EVIDENCE_ROOT.as_posix(),
        "schema_version": package["schema_version"],
        "checksums_file_sha256": V26_EVIDENCE_CHECKSUMS_FILE_SHA256,
        "package_manifest_file_sha256": V26_EVIDENCE_PACKAGE_FILE_SHA256,
        "aggregate_file_sha256": V26_EVIDENCE_AGGREGATE_FILE_SHA256,
        "failure_ledger_file_sha256": V26_EVIDENCE_FAILURE_FILE_SHA256,
        "publication_status": "complete-with-no-go",
        "scientific_complete": False,
        "scientific_matrix_complete": False,
        "scientific_claim_gates_supported": False,
    }


def _verify_fixed_v26_receipts(
    parent_root: Path,
    source_contract: PilotContract,
) -> dict[str, Any]:
    raw = parent_root.joinpath(*V26_RAW_ROOT.parts)
    _, _, launch = _strict_file(
        parent_root,
        V26_RAW_ROOT / "scientific_launch_input.json",
        name="V2.6 scientific launch input",
        expected_sha256=V26_LAUNCH_FILE_SHA256,
    )
    _, _, release = _strict_file(
        parent_root,
        V26_RAW_ROOT / "release_attestation.json",
        name="V2.6 release attestation",
        expected_sha256=V26_RELEASE_FILE_SHA256,
    )
    if (
        launch.get("launch_input_sha256") != V26_LAUNCH_CONTENT_SHA256
        or launch.get("contract_sha256") != V26_CONTRACT_CANONICAL_SHA256
        or release.get("attestation_sha256") != V26_RELEASE_CONTENT_SHA256
        or release.get("status") != "pass"
        or release.get("head_commit") != V26_SCIENCE_COMMIT
    ):
        raise PilotV27Stage0ImportError(
            "V2.6 launch/release identity drifted"
        )

    parent_path, _, parent_import = _sealed_file(
        parent_root,
        V26_RAW_ROOT / "parent-import/parent_import_receipt.json",
        name="V2.6 parent-import receipt",
        schema_version="finevo-pilot-v2.6-parent-import-v1",
        file_sha256=V26_PARENT_IMPORT_FILE_SHA256,
        content_sha256=V26_PARENT_IMPORT_CONTENT_SHA256,
    )
    try:
        verified_parent = verify_v26_parent_import_receipt(
            parent_path,
            repo_root=parent_root,
            contract=source_contract,
            expected_git_commit=V26_SCIENCE_COMMIT,
        )
    except PilotV26ParentImportError as exc:
        raise _translate(exc) from exc
    if verified_parent != parent_import:
        raise PilotV27Stage0ImportError(
            "V2.6 parent-import receipt did not reproduce"
        )

    _, _, parent_stage = _sealed_file(
        parent_root,
        V26_RAW_ROOT / "parent-import/stage_receipt.json",
        name="V2.6 parent-import stage receipt",
        schema_version="finevo-pilot-stage-receipt-v2",
        file_sha256=V26_PARENT_STAGE_RECEIPT_FILE_SHA256,
        content_sha256=V26_PARENT_STAGE_RECEIPT_CONTENT_SHA256,
        verify_canonical_self_hash=False,
    )
    _, _, qref_stage = _sealed_file(
        parent_root,
        V26_RAW_ROOT / "q-ref-resolution/stage_receipt.json",
        name="V2.6 q-ref stage receipt",
        schema_version="finevo-pilot-stage-receipt-v2",
        file_sha256=V26_QREF_STAGE_RECEIPT_FILE_SHA256,
        content_sha256=V26_QREF_STAGE_RECEIPT_CONTENT_SHA256,
        verify_canonical_self_hash=False,
    )
    _, _, stage0 = _sealed_file(
        parent_root,
        V26_RAW_ROOT / "stage0-calibration/stage_receipt.json",
        name="V2.6 Stage-0 receipt",
        schema_version="finevo-pilot-stage-receipt-v2",
        file_sha256=V26_STAGE0_RECEIPT_FILE_SHA256,
        content_sha256=V26_STAGE0_RECEIPT_CONTENT_SHA256,
        verify_canonical_self_hash=False,
    )
    failure = _mapping(
        _mapping(stage0.get("artifacts"), name="V2.6 Stage-0 artifacts").get(
            "stage0_selection_failure"
        ),
        name="V2.6 Stage-0 selection failure",
    )
    if (
        parent_stage.get("status") != "complete"
        or parent_stage.get("complete_cell_count") != 1
        or qref_stage.get("status") != "complete"
        or qref_stage.get("complete_cell_count") != 1
        or stage0.get("status") != "complete-with-no-go"
        or stage0.get("complete_cell_count") != 14
        or stage0.get("registered_run_count") != 14
        or stage0.get("go") is not False
        or stage0.get("execution_progression_go") is not False
        or failure
        != {
            "error_type": "ValueError",
            "message": V26_STAGE0_FAILURE_MESSAGE,
        }
    ):
        raise PilotV27Stage0ImportError(
            "V2.6 parent/q-ref/Stage-0 receipt boundary drifted"
        )
    return {
        "launch": {
            "path": f"{V26_RAW_ROOT.as_posix()}/scientific_launch_input.json",
            "file_sha256": V26_LAUNCH_FILE_SHA256,
            "content_sha256": V26_LAUNCH_CONTENT_SHA256,
        },
        "release_attestation": {
            "path": f"{V26_RAW_ROOT.as_posix()}/release_attestation.json",
            "file_sha256": V26_RELEASE_FILE_SHA256,
            "content_sha256": V26_RELEASE_CONTENT_SHA256,
        },
        "parent_import_receipt": {
            "path": (
                f"{V26_RAW_ROOT.as_posix()}/parent-import/"
                "parent_import_receipt.json"
            ),
            "file_sha256": V26_PARENT_IMPORT_FILE_SHA256,
            "content_sha256": V26_PARENT_IMPORT_CONTENT_SHA256,
        },
    }


def _verify_v26_p95_sources(parent_root: Path) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for profile_id in V27_ALLOWED_P95_PROFILES:
        source = _V26_P95_SOURCES[profile_id]
        output = {
            "runtime_model": source["runtime_model"],
            "served_model": source["served_model"],
        }
        for kind in ("authority", "projection"):
            binding = source[kind]
            relative = _normalized_relative(
                binding["path"],
                required_top="experiment_results",
                name=f"V2.6 {profile_id} {kind} path",
            )
            _, _, value = _strict_file(
                parent_root,
                relative,
                name=f"V2.6 {profile_id} {kind}",
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
                raise PilotV27Stage0ImportError(
                    f"V2.6 {profile_id} {kind} identity drifted"
                )
            output[kind] = _json_copy(binding)
        verified[profile_id] = output
    return verified


def _artifact_binding(
    parent_root: Path,
    relative: PurePosixPath,
) -> dict[str, Any]:
    try:
        _, raw = _guarded_file(parent_root, relative, name=relative.as_posix())
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    return {
        "path": relative.as_posix(),
        "file_sha256": _sha256(raw),
        "byte_size": len(raw),
    }


def _verify_imported_cells(
    parent_root: Path,
    *,
    source_contract: PilotContract,
    target_contract: PilotContract,
    run_snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw = parent_root.joinpath(*V26_RAW_ROOT.parts)
    runs = _mapping(run_snapshot.get("runs"), name="V2.6 run ledger rows")
    source_specs = {spec.run_id: spec for spec in source_contract.expand()}
    if set(runs) != set(source_specs):
        raise PilotV27Stage0ImportError(
            "V2.6 run ledger does not contain the exact expanded denominator"
        )
    status_counts = Counter(str(row.get("status")) for row in runs.values())
    completed = {
        run_id for run_id, row in runs.items() if row.get("status") == "complete"
    }
    expected_complete = {
        spec.run_id
        for spec in source_specs.values()
        if spec.stage_id in _IMPORTABLE_STAGES
    }
    if (
        status_counts
        != Counter({"complete": 16, "integrity-stopped": 195})
        or completed != expected_complete
        or any(
            row.get("spec") != source_specs[run_id].to_dict()
            for run_id, row in runs.items()
        )
    ):
        raise PilotV27Stage0ImportError(
            "V2.6 terminal denominator or completed-cell identity drifted"
        )

    rows: list[dict[str, Any]] = []
    for source_spec, target_spec in _spec_pairs(
        source_contract, target_contract
    ):
        row: dict[str, Any] = {
            "stage_id": source_spec.stage_id,
            "source_run_id": source_spec.run_id,
            "target_run_id": target_spec.run_id,
            "source_spec": source_spec.to_dict(),
            "target_spec": target_spec.to_dict(),
        }
        if source_spec.stage_id == "parent-import":
            row["source_artifacts"] = {
                "receipt": _artifact_binding(
                    parent_root,
                    V26_RAW_ROOT / "parent-import/parent_import_receipt.json",
                ),
                "stage_receipt": _artifact_binding(
                    parent_root,
                    V26_RAW_ROOT / "parent-import/stage_receipt.json",
                ),
            }
        elif source_spec.stage_id == "q-ref-resolution":
            run_dir = raw / "q-ref-resolution/runs" / source_spec.run_id
            verification = verify_manifest(run_dir)
            result = load_verified_run_artifacts(run_dir)
            if (
                not verification.valid
                or result.validation_status.get("status") != "pass"
                or result.summary.get("result_complete") is not True
            ):
                raise PilotV27Stage0ImportError(
                    "V2.6 q-ref source run is not a complete verified run"
                )
            _, _, qref = _sealed_file(
                parent_root,
                V26_RAW_ROOT / "q-ref-resolution/q_ref_resolution.json",
                name="V2.6 q-ref resolution",
                schema_version="finevo-q-ref-resolution-v1",
                file_sha256=V26_QREF_FILE_SHA256,
                content_sha256=V26_QREF_CONTENT_SHA256,
            )
            if (
                qref.get("q_ref") != V26_QREF_VALUE
                or qref.get("scientific_evidence") is not False
                or qref.get("checks", {}).get("validation_pass") is not True
            ):
                raise PilotV27Stage0ImportError(
                    "V2.6 q-ref value or boundary drifted"
                )
            manifest_relative = (
                V26_RAW_ROOT
                / "q-ref-resolution/runs"
                / source_spec.run_id
                / "manifest.json"
            )
            manifest_binding = _artifact_binding(
                parent_root, manifest_relative
            )
            if (
                manifest_binding["file_sha256"]
                != V26_QREF_MANIFEST_FILE_SHA256
            ):
                raise PilotV27Stage0ImportError(
                    "V2.6 q-ref source manifest drifted"
                )
            row["source_artifacts"] = {
                "run_root": (
                    V26_RAW_ROOT
                    / "q-ref-resolution/runs"
                    / source_spec.run_id
                ).as_posix(),
                "manifest": manifest_binding,
                "q_ref_resolution": _artifact_binding(
                    parent_root,
                    V26_RAW_ROOT / "q-ref-resolution/q_ref_resolution.json",
                ),
                "stage_receipt": _artifact_binding(
                    parent_root,
                    V26_RAW_ROOT / "q-ref-resolution/stage_receipt.json",
                ),
            }
        else:
            run_relative = (
                V26_RAW_ROOT
                / "stage0-calibration/runs"
                / source_spec.run_id
            )
            run_dir = parent_root.joinpath(*run_relative.parts)
            verification = verify_manifest(run_dir)
            result = load_verified_run_artifacts(
                run_dir, authority_repo_root=parent_root
            )
            if (
                not verification.valid
                or result.validation_status.get("status") != "pass"
                or result.summary.get("result_complete") is not True
                or len(result.records.get("actions", ())) != 48
                or len(result.records.get("utility_ledger", ())) != 48
                or len(result.records.get("api_usage", ())) != 48
                or result.records.get("errors")
            ):
                raise PilotV27Stage0ImportError(
                    f"V2.6 Stage-0 source run is incomplete: {source_spec.run_id}"
                )
            journal_relative = (
                V26_RAW_ROOT
                / "stage0-calibration/provider_call_journals"
                / f"{source_spec.run_id}--actor.json"
            )
            journal_path = parent_root.joinpath(*journal_relative.parts)
            journal = verify_provider_call_journal(
                journal_path,
                expected_run_id=source_spec.run_id,
                expected_contract_hash=V26_CONTRACT_CANONICAL_SHA256,
                require_terminal_dispositions=True,
            )
            events = journal["events"]
            if (
                len(events) != 96
                or Counter(event["event_type"] for event in events)
                != Counter(
                    {"completion_received": 48, "parse_disposition": 48}
                )
            ):
                raise PilotV27Stage0ImportError(
                    f"V2.6 Stage-0 journal is incomplete: {source_spec.run_id}"
                )
            row["source_artifacts"] = {
                "run_root": run_relative.as_posix(),
                "manifest": _artifact_binding(
                    parent_root, run_relative / "manifest.json"
                ),
                "actor_journal": _artifact_binding(
                    parent_root, journal_relative
                ),
            }
        rows.append(row)
    if (
        Counter(row["stage_id"] for row in rows)
        != Counter(
            {
                "parent-import": 1,
                "q-ref-resolution": 1,
                "stage0-calibration": 14,
            }
        )
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
        raise PilotV27Stage0ImportError(
            "V2.6 imported-cell profile/seed grid drifted"
        )
    return sorted(rows, key=lambda row: row["target_run_id"])


def _validate_target_contract(
    contract: PilotContract,
    *,
    require_frozen: bool,
) -> None:
    amendment = contract.stage0_evaluator_retry_amendment
    if (
        contract.contract_id != V27_CONTRACT_ID
        or contract.implementation.get("required_git_tag") != V27_SCIENCE_TAG
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
        != "finevo-pilot-stage0-evaluator-retry-amendment-v1"
    ):
        raise PilotV27Stage0ImportError(
            "V2.7 Stage-0 import requires its exact amended contract"
        )


def _audit_v26_source(
    *,
    parent_repo_root: str | Path,
    child_repo_root: str | Path,
    target_contract: PilotContract,
) -> dict[str, Any]:
    parent_root = _real_root(parent_repo_root, name="V2.6 parent repository")
    child_root = _real_root(child_repo_root, name="V2.7 child repository")
    _validate_target_contract(target_contract, require_frozen=False)
    helper = _parent_helper_manifest()
    try:
        _verify_parent_git(parent_root, helper)
        source_contract = _verify_parent_contract(parent_root, helper)
        run_snapshot, budget_snapshot = _verify_parent_ledgers(
            parent_root,
            parent_contract=source_contract,
            source_manifest=helper,
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if (
        _git(parent_root, "rev-parse", "HEAD") != V26_SCIENCE_COMMIT
        or _git(
            parent_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        )
    ):
        raise PilotV27Stage0ImportError(
            "V2.6 parent HEAD or tracked worktree drifted"
        )
    raw_root = parent_root.joinpath(*V26_RAW_ROOT.parts)
    inventory = _verify_exact_v26_inventory(raw_root)
    if (
        run_snapshot["ledger_sha256"] != V26_RUN_LEDGER_INTERNAL_SHA256
        or len(run_snapshot["events"]) != V26_RUN_LEDGER_EVENT_COUNT
        or run_snapshot["events"][-1]["event_sha256"]
        != V26_RUN_LEDGER_EVENT_HEAD
        or budget_snapshot["ledger_sha256"]
        != V26_BUDGET_LEDGER_INTERNAL_SHA256
        or len(budget_snapshot["events"]) != V26_BUDGET_LEDGER_EVENT_COUNT
        or budget_snapshot["event_chain_head"]
        != V26_BUDGET_LEDGER_EVENT_HEAD
    ):
        raise PilotV27Stage0ImportError("V2.6 verified ledger identity drifted")
    receipts = _verify_fixed_v26_receipts(parent_root, source_contract)
    p95_sources = _verify_v26_p95_sources(parent_root)
    evidence = _verify_v26_published_evidence(child_root)
    imported = _verify_imported_cells(
        parent_root,
        source_contract=source_contract,
        target_contract=target_contract,
        run_snapshot=run_snapshot,
    )
    return {
        "parent_root": parent_root,
        "child_root": child_root,
        "source_contract": source_contract,
        "inventory": inventory,
        "receipts": receipts,
        "p95_sources": p95_sources,
        "evidence": evidence,
        "imported_cells": imported,
    }


def build_v27_source_manifest(
    *,
    parent_repo_root: str | Path,
    child_repo_root: str | Path,
    target_contract: PilotContract | None = None,
) -> dict[str, Any]:
    """Verify frozen V2.6 and build the deterministic V2.7 source manifest."""

    child_root = _real_root(child_repo_root, name="V2.7 child repository")
    contract = target_contract or load_pilot_contract(
        child_root.joinpath(*V27_EXPANDED_CONTRACT_PATH.parts)
    )
    audit = _audit_v26_source(
        parent_repo_root=parent_repo_root,
        child_repo_root=child_root,
        target_contract=contract,
    )
    return _seal(
        {
            "schema_version": V27_SOURCE_MANIFEST_SCHEMA_VERSION,
            "v2_6_terminal_parent": {
                "contract": {
                    "contract_id": V26_CONTRACT_ID,
                    "path": V26_CONTRACT_PATH.as_posix(),
                    "schema_version": "finevo-pilot-contract-v2",
                    "status": "frozen",
                    "file_sha256": V26_CONTRACT_FILE_SHA256,
                    "canonical_sha256": V26_CONTRACT_CANONICAL_SHA256,
                },
                "release": {
                    "science_tag": V26_SCIENCE_TAG,
                    "science_tag_object": V26_SCIENCE_TAG_OBJECT,
                    "science_commit": V26_SCIENCE_COMMIT,
                    "tag_kind": "annotated",
                    "raw_root": V26_RAW_ROOT.as_posix(),
                },
                "raw_snapshot": {
                    "root": V26_RAW_ROOT.as_posix(),
                    "inventory_schema_version":
                    "finevo-raw-tree-inventory-v1",
                    "inventory_canonicalization":
                    "json-sort-keys-compact-utf8-v1",
                    "file_count": V26_RAW_FILE_COUNT,
                    "storage_bytes": V26_RAW_STORAGE_BYTES,
                    "inventory_sha256": V26_RAW_INVENTORY_SHA256,
                },
                "ledgers": _json_copy(_parent_helper_manifest()["ledgers"]),
                "terminal_denominator": {
                    "registered_cells": 211,
                    "scientific_cells": 209,
                    "terminal_cells": 211,
                    "all_rows_present": True,
                    "all_rows_terminal": True,
                    "status_counts": {
                        "complete": 16,
                        "integrity-stopped": 195,
                    },
                    "completed_cell_breakdown": {
                        "parent-import": 1,
                        "q-ref-resolution": 1,
                        "stage0-calibration": 14,
                    },
                    "terminal_status": "complete-with-no-go",
                    "scientific_complete": False,
                    "scientific_matrix_complete": False,
                    "scientific_claim_gates_supported": False,
                },
                "fixed_receipts": audit["receipts"],
                "stage0_failure": {
                    "root_cause_code":
                    "baseline-only-stage0-routed-through-shock-recovery-summary",
                    "error_type": "ValueError",
                    "message": V26_STAGE0_FAILURE_MESSAGE,
                    "selection_observed_before_amendment": True,
                    "a_d_treatment_effect_outcomes_generated": False,
                    "a_d_treatment_effect_outcomes_available": False,
                    "a_d_treatment_effect_outcomes_inspected": False,
                },
            },
            "published_v2_6_evidence": audit["evidence"],
            "v2_6_p95_sources_for_child_reseal": audit["p95_sources"],
            "imported_complete_cells": audit["imported_cells"],
            "cumulative_budget_debit": V27_CUMULATIVE_DEBIT.to_dict(),
            "import_policy": {
                "source_raw_namespace": V26_RAW_ROOT.as_posix(),
                "child_raw_namespace": V27_RAW_ROOT.as_posix(),
                "child_snapshot_namespace": (
                    V27_RAW_ROOT / V27_SNAPSHOT_RELATIVE
                ).as_posix(),
                "exact_full_raw_snapshot_copy": True,
                "source_manifests_rewritten": False,
                "source_journals_rewritten": False,
                "provider_construction_during_import": False,
                "provider_redispatch_for_imported_cells": "forbidden",
                "v2_6_no_go_preserved": True,
                "v2_6_terminal_rows_reclassified": False,
                "scientific_evidence": False,
            },
            "observation_boundary": {
                "stage0_calibration_selection_observed_before_amendment": True,
                "a_d_treatment_effect_outcomes_generated": False,
                "a_d_treatment_effect_outcomes_observed": False,
                "amendment_is_outcome_blind_with_respect_to_a_d_effects": True,
            },
        }
    )


def validate_v27_source_manifest(
    value: Mapping[str, Any],
    *,
    parent_repo_root: str | Path,
    child_repo_root: str | Path,
    target_contract: PilotContract | None = None,
) -> dict[str, Any]:
    """Rebuild and compare a proposed V2.7 source manifest exactly."""

    candidate = _json_copy(value)
    try:
        _verify_self_hash(
            candidate,
            schema_version=V27_SOURCE_MANIFEST_SCHEMA_VERSION,
            name="V2.7 source manifest",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    expected = build_v27_source_manifest(
        parent_repo_root=parent_repo_root,
        child_repo_root=child_repo_root,
        target_contract=target_contract,
    )
    if candidate != expected:
        raise PilotV27Stage0ImportError(
            "V2.7 source manifest differs from verified V2.6 authority"
        )
    return candidate


def write_v27_source_manifest_draft(
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


def _copy_exact_snapshot(
    *,
    source_root: Path,
    destination_root: Path,
    inventory: Sequence[Mapping[str, Any]],
) -> None:
    for row in inventory:
        relative = _normalized_relative(
            str(row.get("path", "")),
            required_top=None,
            name="V2.6 raw inventory path",
        )
        try:
            _, raw = _guarded_file(
                source_root, relative, name=f"V2.6 raw {relative.as_posix()}"
            )
            _atomic_exact_bytes(destination_root.joinpath(*relative.parts), raw)
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
        if (
            len(raw) != row.get("byte_size")
            or _sha256(raw) != row.get("sha256")
        ):
            raise PilotV27Stage0ImportError(
                f"V2.6 raw source changed during copy: {relative.as_posix()}"
            )
    copied_rows, copied = _inventory(destination_root)
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
        or copied["file_count"] != V26_RAW_FILE_COUNT
        or copied["storage_bytes"] != V26_RAW_STORAGE_BYTES
        or copied["inventory_sha256"] != V26_RAW_INVENTORY_SHA256
    ):
        raise PilotV27Stage0ImportError(
            "V2.7 copied V2.6 raw snapshot differs from its source"
        )


def _child_contract_binding(
    child_root: Path,
    contract: PilotContract,
) -> dict[str, Any]:
    path, raw, value = _strict_file(
        child_root,
        V27_EXPANDED_CONTRACT_PATH,
        name="expanded V2.7 contract",
    )
    parsed = PilotContract.from_dict(value)
    if (
        parsed.contract_id != contract.contract_id
        or parsed.canonical_hash != contract.canonical_hash
        or parsed.to_dict() != contract.to_dict()
    ):
        raise PilotV27Stage0ImportError(
            "expanded V2.7 contract differs from selected contract"
        )
    return {
        "path": V27_EXPANDED_CONTRACT_PATH.as_posix(),
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
        "path": V27_SOURCE_MANIFEST_PATH.as_posix(),
        "schema_version": V27_SOURCE_MANIFEST_SCHEMA_VERSION,
        "file_sha256": _sha256(raw),
        "content_sha256": value["integrity"]["content_sha256"],
    }


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
            schema_version=V27_SOURCE_MANIFEST_SCHEMA_VERSION,
            name="V2.7 source manifest",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if set(candidate) != {
        "schema_version",
        "v2_6_terminal_parent",
        "published_v2_6_evidence",
        "v2_6_p95_sources_for_child_reseal",
        "imported_complete_cells",
        "cumulative_budget_debit",
        "import_policy",
        "observation_boundary",
        "integrity",
    }:
        raise PilotV27Stage0ImportError(
            "V2.7 source manifest fields drifted"
        )
    terminal = _mapping(
        candidate.get("v2_6_terminal_parent"),
        name="V2.6 terminal parent",
    )
    raw = _mapping(terminal.get("raw_snapshot"), name="V2.6 raw snapshot")
    denominator = _mapping(
        terminal.get("terminal_denominator"),
        name="V2.6 terminal denominator",
    )
    if (
        terminal.get("contract", {}).get("canonical_sha256")
        != V26_CONTRACT_CANONICAL_SHA256
        or terminal.get("release", {}).get("science_tag")
        != V26_SCIENCE_TAG
        or terminal.get("release", {}).get("science_commit")
        != V26_SCIENCE_COMMIT
        or raw.get("file_count") != V26_RAW_FILE_COUNT
        or raw.get("storage_bytes") != V26_RAW_STORAGE_BYTES
        or raw.get("inventory_sha256") != V26_RAW_INVENTORY_SHA256
        or denominator.get("status_counts")
        != {"complete": 16, "integrity-stopped": 195}
        or denominator.get("terminal_status") != "complete-with-no-go"
        or denominator.get("scientific_complete") is not False
        or denominator.get("scientific_matrix_complete") is not False
        or denominator.get("scientific_claim_gates_supported") is not False
        or candidate.get("cumulative_budget_debit")
        != V27_CUMULATIVE_DEBIT.to_dict()
        or candidate.get("v2_6_p95_sources_for_child_reseal")
        != _V26_P95_SOURCES
    ):
        raise PilotV27Stage0ImportError(
            "V2.7 source manifest parent authority drifted"
        )
    rows = candidate.get("imported_complete_cells")
    if (
        not isinstance(rows, list)
        or len(rows) != 16
        or len({row.get("target_run_id") for row in rows}) != 16
        or Counter(row.get("stage_id") for row in rows)
        != Counter(
            {
                "parent-import": 1,
                "q-ref-resolution": 1,
                "stage0-calibration": 14,
            }
        )
    ):
        raise PilotV27Stage0ImportError(
            "V2.7 source manifest imported-cell inventory drifted"
        )
    policy = _mapping(
        candidate.get("import_policy"), name="V2.7 import policy"
    )
    boundary = _mapping(
        candidate.get("observation_boundary"),
        name="V2.7 observation boundary",
    )
    if (
        policy.get("provider_construction_during_import") is not False
        or policy.get("provider_redispatch_for_imported_cells") != "forbidden"
        or policy.get("v2_6_no_go_preserved") is not True
        or policy.get("scientific_evidence") is not False
        or boundary.get("stage0_calibration_selection_observed_before_amendment")
        is not True
        or boundary.get("a_d_treatment_effect_outcomes_generated") is not False
        or boundary.get("a_d_treatment_effect_outcomes_observed") is not False
    ):
        raise PilotV27Stage0ImportError(
            "V2.7 source manifest observation/import boundary drifted"
        )
    if contract is not None:
        _validate_target_contract(contract, require_frozen=False)
        binding = _mapping(
            contract.stage0_evaluator_retry_amendment.get("source_manifest"),
            name="V2.7 contract source-manifest binding",
        )
        if (
            binding.get("path") != V27_SOURCE_MANIFEST_PATH.as_posix()
            or binding.get("schema_version")
            != V27_SOURCE_MANIFEST_SCHEMA_VERSION
        ):
            raise PilotV27Stage0ImportError(
                "V2.7 contract source-manifest path/schema drifted"
            )
        if contract.status == "frozen" and (
            file_sha256 != binding.get("file_sha256")
            or candidate["integrity"]["content_sha256"]
            != binding.get("content_sha256")
        ):
            raise PilotV27Stage0ImportError(
                "V2.7 frozen contract/source-manifest binding drifted"
            )
    return candidate


def load_v27_source_manifest(
    *,
    repo_root: str | Path,
    contract: PilotContract | None = None,
    parent_repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load the tracked source manifest and optionally reverify V2.6 live."""

    root = _real_root(repo_root, name="V2.7 child repository")
    _, raw, value = _strict_file(
        root,
        V27_SOURCE_MANIFEST_PATH,
        name="tracked V2.7 source manifest",
    )
    selected_contract = contract or load_pilot_contract(
        root.joinpath(*V27_EXPANDED_CONTRACT_PATH.parts)
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
        raise PilotV27Stage0ImportError(
            "tracked V2.7 source manifest is not canonical pretty JSON"
        )
    if parent_repo_root is not None:
        validate_v27_source_manifest(
            candidate,
            parent_repo_root=parent_repo_root,
            child_repo_root=root,
            target_contract=selected_contract,
        )
    return candidate


def parent_budget_debit_for_v27(
    contract: PilotContract,
) -> ParentBudgetDebit | None:
    """Return the exact cumulative V2.6 debit inherited by V2.7."""

    if contract.contract_id != V27_CONTRACT_ID:
        return None
    _validate_target_contract(contract, require_frozen=False)
    amendment = contract.stage0_evaluator_retry_amendment
    carry = _mapping(
        amendment.get("budget_carry_forward"),
        name="V2.7 budget carry-forward",
    )
    expected_prior = V27_CUMULATIVE_DEBIT.to_dict()
    expected_prior.pop("schema_version")
    if (
        carry.get("cumulative_prior") != expected_prior
        or carry.get("budget_reset") is not False
        or carry.get("debit_before_new_dispatch") is not True
    ):
        raise PilotV27Stage0ImportError(
            "V2.7 cumulative parent debit drifted"
        )
    return V27_CUMULATIVE_DEBIT


def _build_v27_parent_import_receipt(
    *,
    child_root: Path,
    contract: PilotContract,
    child_git_commit: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return _seal(
        {
            "schema_version": V27_PARENT_IMPORT_SCHEMA_VERSION,
            "contract": _child_contract_binding(child_root, contract),
            "child_release": {
                "git_tag": V27_SCIENCE_TAG,
                "git_commit": child_git_commit,
            },
            "source_manifest": _source_manifest_binding(manifest),
            "v2_6_terminal_parent": _json_copy(
                manifest["v2_6_terminal_parent"]
            ),
            "published_v2_6_evidence": _json_copy(
                manifest["published_v2_6_evidence"]
            ),
            "v2_6_p95_sources_for_child_reseal": _json_copy(
                manifest["v2_6_p95_sources_for_child_reseal"]
            ),
            "cumulative_budget_debit": _json_copy(
                manifest["cumulative_budget_debit"]
            ),
            "imported_complete_cells": [
                {
                    "stage_id": row["stage_id"],
                    "source_run_id": row["source_run_id"],
                    "target_run_id": row["target_run_id"],
                    "source_artifacts": _json_copy(row["source_artifacts"]),
                }
                for row in manifest["imported_complete_cells"]
            ],
            "copied_raw_snapshot": {
                "source_root": V26_RAW_ROOT.as_posix(),
                "snapshot_root": (
                    V27_RAW_ROOT / V27_SNAPSHOT_RELATIVE
                ).as_posix(),
                "file_count": V26_RAW_FILE_COUNT,
                "storage_bytes": V26_RAW_STORAGE_BYTES,
                "inventory_sha256": V26_RAW_INVENTORY_SHA256,
                "exact_bytes": True,
            },
            "provider_calls": 0,
            "hosted_provider_calls": 0,
            "local_model_calls": 0,
            "stage0_calibration_selection_observed_before_amendment": True,
            "a_d_treatment_effect_outcomes_generated": False,
            "a_d_treatment_effect_outcomes_observed": False,
            "scientific_evidence": False,
            "v2_6_terminal_no_go_preserved": True,
            "source_artifacts_rewritten": False,
        }
    )


def _materialize_v27_resealed_p95(
    *,
    child_root: Path,
    child_raw: Path,
    contract: PilotContract,
    child_git_commit: str,
) -> dict[str, Any]:
    """Build, atomically persist, and reverify both V2.7 p95 authorities."""

    from .observed_p95_authority import (
        ObservedP95AuthorityError,
        build_v27_resealed_observed_p95_authority,
        verify_v27_resealed_observed_p95_authority,
        verify_v27_resealed_observed_p95_projection,
    )

    output: dict[str, Any] = {}
    for profile_id in V27_ALLOWED_P95_PROFILES:
        source = v2_6_p95_source_binding(
            repo_root=child_root,
            child_raw_root=child_raw,
            profile_id=profile_id,
        )
        try:
            built = build_v27_resealed_observed_p95_authority(
                repo_root=child_root,
                contract=contract,
                contract_path=V27_EXPANDED_CONTRACT_PATH.as_posix(),
                raw_root=V27_RAW_ROOT.as_posix(),
                profile_id=profile_id,
                expected_git_commit=child_git_commit,
                verified_v2_6_source_binding=source,
            )
            receipt_path = Path(built["receipt_path"])
            projection_path = Path(built["projection_path"])
            _atomic_exact_json(receipt_path, built["receipt"])
            _atomic_exact_json(projection_path, built["projection"])
            reservations = verify_v27_resealed_observed_p95_authority(
                receipt_path,
                repo_root=child_root,
                expected_git_commit=child_git_commit,
            )
            projection = verify_v27_resealed_observed_p95_projection(
                projection_path,
                receipt_or_path=receipt_path,
                repo_root=child_root,
                expected_git_commit=child_git_commit,
            )
        except (ObservedP95AuthorityError, PilotV24ParentImportError) as exc:
            raise _translate(exc) from exc
        receipt_raw = receipt_path.read_bytes()
        projection_raw = projection_path.read_bytes()
        output[profile_id] = {
            "receipt": {
                "path": _repo_relative(
                    child_root,
                    receipt_path,
                    name=f"{profile_id} V2.7 p95 receipt",
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
                    name=f"{profile_id} V2.7 p95 projection",
                ),
                "file_sha256": _sha256(projection_raw),
                "content_sha256": projection["integrity"]["content_sha256"],
            },
            "runtime_models": sorted(reservations),
            "source_kind": "v2.6-terminal-stage0-import-v2.7",
        }
    return output


def persist_v27_parent_import(
    *,
    contract: PilotContract,
    repo_root: str | Path,
    raw_root: str | Path,
    parent_repo_root: str | Path,
    child_git_tag: str,
    child_git_commit: str,
    source_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify V2.6, copy its exact raw tree, and seal a zero-call receipt."""

    child_root = _real_root(repo_root, name="V2.7 child repository")
    parent_root = _real_root(parent_repo_root, name="V2.6 parent repository")
    child_raw = Path(raw_root).absolute()
    if (
        _repo_relative(child_root, child_raw, name="V2.7 raw root")
        != V27_RAW_ROOT.as_posix()
    ):
        raise PilotV27Stage0ImportError(
            "V2.7 import requires its fresh pilot-v2.7 raw namespace"
        )
    _validate_target_contract(contract, require_frozen=True)
    if (
        child_git_tag != V27_SCIENCE_TAG
        or _COMMIT_RE.fullmatch(child_git_commit) is None
    ):
        raise PilotV27Stage0ImportError(
            "V2.7 child release tag or commit is malformed"
        )
    manifest = (
        validate_v27_source_manifest(
            source_manifest,
            parent_repo_root=parent_root,
            child_repo_root=child_root,
            target_contract=contract,
        )
        if source_manifest is not None
        else load_v27_source_manifest(
            repo_root=child_root,
            contract=contract,
            parent_repo_root=parent_root,
        )
    )
    source_raw = parent_root.joinpath(*V26_RAW_ROOT.parts)
    inventory = _verify_exact_v26_inventory(source_raw)
    snapshot = child_raw.joinpath(*V27_SNAPSHOT_RELATIVE.parts)
    _copy_exact_snapshot(
        source_root=source_raw,
        destination_root=snapshot,
        inventory=inventory,
    )
    receipt = _build_v27_parent_import_receipt(
        child_root=child_root,
        contract=contract,
        child_git_commit=child_git_commit,
        manifest=manifest,
    )
    receipt_path = child_raw / "parent-import/parent_import_receipt.json"
    try:
        _atomic_exact_json(receipt_path, receipt)
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    resealed_p95 = _materialize_v27_resealed_p95(
        child_root=child_root,
        child_raw=child_raw,
        contract=contract,
        child_git_commit=child_git_commit,
    )
    return {
        "receipt": str(receipt_path),
        "receipt_content_sha256": receipt["integrity"]["content_sha256"],
        "snapshot_root": str(snapshot),
        "snapshot_inventory_sha256": V26_RAW_INVENTORY_SHA256,
        "imported_cell_count": 16,
        "imported_profiles": sorted(V27_ALLOWED_P95_PROFILES),
        "resealed_p95_profiles": resealed_p95,
        "provider_calls": 0,
        "scientific_evidence": False,
        "v2_6_terminal_no_go_preserved": True,
    }


def verify_v27_parent_import_receipt(
    receipt_or_path: Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path,
    contract: PilotContract,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Verify a V2.7 import from tracked manifest and local exact snapshot."""

    root = _real_root(repo_root, name="V2.7 child repository")
    _validate_target_contract(contract, require_frozen=True)
    if _COMMIT_RE.fullmatch(expected_git_commit) is None:
        raise PilotV27Stage0ImportError(
            "V2.7 expected commit must be 40 lowercase hex characters"
        )
    if isinstance(receipt_or_path, Mapping):
        value = _json_copy(receipt_or_path)
    else:
        path = Path(receipt_or_path)
        if path.is_absolute():
            try:
                relative = PurePosixPath(*path.absolute().relative_to(root).parts)
            except ValueError as exc:
                raise PilotV27Stage0ImportError(
                    "V2.7 parent-import receipt escaped the repository"
                ) from exc
        else:
            relative = _normalized_relative(
                path,
                required_top="experiment_results",
                name="V2.7 parent-import receipt path",
            )
        _, _, value = _strict_file(
            root, relative, name="V2.7 parent-import receipt"
        )
    try:
        _verify_self_hash(
            value,
            schema_version=V27_PARENT_IMPORT_SCHEMA_VERSION,
            name="V2.7 parent-import receipt",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    manifest = load_v27_source_manifest(repo_root=root, contract=contract)
    snapshot = root.joinpath(*V27_RAW_ROOT.parts, *V27_SNAPSHOT_RELATIVE.parts)
    _verify_exact_v26_inventory(snapshot)
    expected = _build_v27_parent_import_receipt(
        child_root=root,
        contract=contract,
        child_git_commit=expected_git_commit,
        manifest=manifest,
    )
    if value != expected:
        raise PilotV27Stage0ImportError(
            "V2.7 parent-import receipt differs from sealed sources"
        )
    return value


def source_binding_for_target(
    source_manifest: Mapping[str, Any],
    target: PilotRunSpec | Mapping[str, Any] | str,
) -> dict[str, Any]:
    """Return the unique immutable V2.6 source bound to a V2.7 target cell."""

    value = _json_copy(source_manifest)
    try:
        _verify_self_hash(
            value,
            schema_version=V27_SOURCE_MANIFEST_SCHEMA_VERSION,
            name="V2.7 source manifest",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
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
        for row in value.get("imported_complete_cells", ())
        if isinstance(row, Mapping)
        and row.get("target_run_id") == target_run_id
    ]
    if len(rows) != 1:
        raise PilotV27Stage0ImportError(
            "target has no unique imported V2.6 source binding"
        )
    row = _json_copy(rows[0])
    if row["stage_id"] not in _IMPORTABLE_STAGES:
        raise PilotV27Stage0ImportError(
            "only parent/q-ref/Stage-0 targets may use V2.6 sources"
        )
    if not isinstance(target, str):
        target_value = (
            target.to_dict()
            if isinstance(target, PilotRunSpec)
            else _json_copy(target)
        )
        if target_value != row["target_spec"]:
            raise PilotV27Stage0ImportError(
                "target spec differs from its source-manifest binding"
            )
    return row


def imported_v26_raw_root(raw_root: str | Path) -> Path:
    """Return the child-local root of the exact immutable V2.6 snapshot."""

    return Path(raw_root).joinpath(*V27_SNAPSHOT_RELATIVE.parts)


def snapshot_path_for_source_artifact(
    child_raw_root: str | Path,
    source_artifact_path: str,
) -> Path:
    """Map one V2.6 raw artifact path into the exact V2.7 snapshot."""

    relative = _normalized_relative(
        source_artifact_path,
        required_top="experiment_results",
        name="V2.6 source artifact path",
    )
    try:
        inside = PurePosixPath(*relative.parts[len(V26_RAW_ROOT.parts) :])
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise PilotV27Stage0ImportError(
            "V2.6 source artifact path is malformed"
        ) from exc
    if (
        tuple(relative.parts[: len(V26_RAW_ROOT.parts)])
        != V26_RAW_ROOT.parts
        or not inside.parts
    ):
        raise PilotV27Stage0ImportError(
            "source artifact is outside the V2.6 raw namespace"
        )
    return (
        imported_v26_raw_root(child_raw_root)
        .joinpath(*inside.parts)
    )


def imported_v26_run_dir(
    raw_root: str | Path,
    v27_spec: PilotRunSpec | Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> Path:
    """Resolve an imported q-ref/Stage-0 run without changing its identity."""

    binding = source_binding_for_target(source_manifest, v27_spec)
    run_root = binding["source_artifacts"].get("run_root")
    if not isinstance(run_root, str):
        raise PilotV27Stage0ImportError(
            "target is not backed by an imported V2.6 runner directory"
        )
    return snapshot_path_for_source_artifact(raw_root, run_root)


def v2_6_p95_source_binding(
    *,
    repo_root: str | Path,
    child_raw_root: str | Path,
    profile_id: str,
) -> dict[str, Any]:
    """Verify and return one V2.6 p95 source for V2.7 child resealing."""

    if profile_id not in V27_ALLOWED_P95_PROFILES:
        raise PilotV27Stage0ImportError(
            f"{profile_id} has no imported V2.6 p95 source"
        )
    root = _real_root(repo_root, name="V2.7 child repository")
    contract = load_pilot_contract(
        root.joinpath(*V27_EXPANDED_CONTRACT_PATH.parts)
    )
    raw_root = Path(child_raw_root).absolute()
    if (
        _repo_relative(root, raw_root, name="V2.7 raw root")
        != V27_RAW_ROOT.as_posix()
    ):
        raise PilotV27Stage0ImportError(
            "V2.6 p95 source requires the exact V2.7 raw namespace"
        )
    receipt_path = raw_root / "parent-import/parent_import_receipt.json"
    expected_commit = _git(root, "rev-parse", "HEAD")
    verify_v27_parent_import_receipt(
        receipt_path,
        repo_root=root,
        contract=contract,
        expected_git_commit=expected_commit,
    )
    manifest = load_v27_source_manifest(repo_root=root, contract=contract)
    source = manifest["v2_6_p95_sources_for_child_reseal"][profile_id]
    output = {
        "source_contract_sha256": V26_CONTRACT_CANONICAL_SHA256,
        "source_git_tag": V26_SCIENCE_TAG,
        "source_git_commit": V26_SCIENCE_COMMIT,
        "model_id": profile_id,
        "runtime_model": source["runtime_model"],
        "served_model": source["served_model"],
    }
    for kind in ("authority", "projection"):
        expected = source[kind]
        path = snapshot_path_for_source_artifact(
            raw_root, expected["path"]
        )
        if path.is_symlink() or not path.is_file():
            raise PilotV27Stage0ImportError(
                f"imported V2.6 {profile_id} {kind} is unavailable"
            )
        raw = path.read_bytes()
        value = _strict_json(raw, name=f"imported V2.6 {profile_id} {kind}")
        integrity = value.get("integrity")
        if (
            len(raw) <= 0
            or _sha256(raw) != expected["file_sha256"]
            or value.get("schema_version") != expected["schema_version"]
            or not isinstance(integrity, Mapping)
            or integrity.get("content_sha256")
            != expected["content_sha256"]
        ):
            raise PilotV27Stage0ImportError(
                f"imported V2.6 {profile_id} {kind} drifted"
            )
        output[kind] = {
            **_json_copy(expected),
            "snapshot_path": _repo_relative(
                root, path, name=f"imported V2.6 {profile_id} {kind}"
            ),
        }
        if kind == "authority":
            output["reservations"] = _json_copy(value["reservations"])
    return output


__all__ = [
    "PilotV27Stage0ImportError",
    "V26_RAW_FILE_COUNT",
    "V26_RAW_INVENTORY_SHA256",
    "V26_RAW_STORAGE_BYTES",
    "V27_PARENT_IMPORT_SCHEMA_VERSION",
    "V27_ALLOWED_P95_PROFILES",
    "V27_CONTRACT_ID",
    "V27_RAW_ROOT",
    "V27_SNAPSHOT_RELATIVE",
    "V27_SOURCE_MANIFEST_PATH",
    "V27_SOURCE_MANIFEST_SCHEMA_VERSION",
    "build_v27_source_manifest",
    "imported_v26_raw_root",
    "imported_v26_run_dir",
    "load_v27_source_manifest",
    "parent_budget_debit_for_v27",
    "persist_v27_parent_import",
    "snapshot_path_for_source_artifact",
    "source_binding_for_target",
    "validate_v27_source_manifest",
    "v2_6_p95_source_binding",
    "verify_v27_parent_import_receipt",
    "write_v27_source_manifest_draft",
]
