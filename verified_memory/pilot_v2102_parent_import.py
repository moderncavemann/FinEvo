"""Zero-provider V2.9 prerequisite import for the V2.10.2 adapter retry.

V2.10.1 is an immutable ``complete-with-no-go`` release: all sixteen reusable
prerequisites and ten offline candidate-admission cells completed, while 185
actor-facing A-D cells failed before provider construction at the generic
observed-p95 consumer.  V2.10.2 preserves that terminal lineage, imports none
of those A-D cells, and reads the reusable prerequisite and observed-p95
sources from the byte-exact V2.9 snapshot nested inside V2.10.1.

In particular, this module never treats a V2.10.1 current-release p95 wrapper as
V2.10.2 authority.  It re-verifies the V2.8-origin receipts inside the exact
V2.9 snapshot and reseals their unchanged reservation values to the selected
V2.10.2 contract, annotated tag, and commit.  No provider client is constructed
and no provider call is made anywhere in this adapter.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from .pilot_budget import ParentBudgetDebit
from .pilot_contract import (
    PilotContract,
    PilotRunSpec,
    canonical_contract_sha256,
    canonical_sha256,
    load_pilot_contract,
)
from .pilot_v24_parent_import import (
    CANONICALIZATION,
    PilotV24ParentImportError,
    _git,
    _guarded_file,
    _json_copy,
    _normalized_relative,
    _real_root,
    _seal,
    _sha256,
    _strict_json,
    _verify_self_hash,
)
from .pilot_v27_stage0_import import (
    PilotV27Stage0ImportError,
    _atomic_exact_bytes_no_follow,
)
from .pilot_v29_stage0_import import (
    V28_CONTRACT_CANONICAL_SHA256,
    V28_CONTRACT_ID,
    V28_SCIENCE_COMMIT,
    V28_SCIENCE_TAG,
    _inventory,
    verify_v29_imported_v28_observed_p95,
)
from .runner import PreflightP95Reservation
from . import pilot_v210_parent_import as _v210
from . import pilot_v2101_parent_import as _v2101


V2102_CONTRACT_ID = "finevo-pilot-v2.10.2"
V2102_SCIENCE_TAG = "pilot-v2.10.2-science"
V2102_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.10.2/raw")
V2102_EXPANDED_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_10_2.yaml")
V2102_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_10_2_source_manifest.json"
)
V2102_SNAPSHOT_RELATIVE = PurePosixPath("parent-import/v2_9_raw_snapshot")
V2102_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.10.2-source-manifest-v1"
V2102_PARENT_IMPORT_SCHEMA_VERSION = "finevo-pilot-v2.10.2-parent-import-v1"
V2102_RESEALED_P95_AUTHORITY_SCHEMA_VERSION = (
    "finevo-pilot-v2.10.2-resealed-observed-p95-authority-v1"
)
V2102_RESEALED_P95_SOURCE_KIND = "v2.9-exact-raw-via-v2.10.1-terminal-v2.10.2"
V2102_ALLOWED_P95_PROFILES = tuple(_v210.V210_ALLOWED_P95_PROFILES)

# Immutable V2.10.1 scientific release and complete-with-no-go publication.
V2101_PARENT_CONTRACT_ID = "finevo-pilot-v2.10.1"
V2101_PARENT_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_10_1.yaml")
V2101_PARENT_CONTRACT_FILE_SHA256 = (
    "662e17490791675778ba83a31bcaa85bd128233bd20cd0485b23c5f85fde29c8"
)
V2101_PARENT_CONTRACT_CANONICAL_SHA256 = (
    "1f9c642c155d5256815cb14a68335b65a25497523c14210c36f89070b3c8d996"
)
V2101_PARENT_SCIENCE_TAG = "pilot-v2.10.1-science"
V2101_PARENT_SCIENCE_TAG_OBJECT = "2e6137cb5f4c3c8e5dc174efe8813cf04f2490f5"
V2101_PARENT_SCIENCE_COMMIT = "b5bfa9b86d3cdb706cea5be707597bef8ac85aed"
V2101_PARENT_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.10.1/raw")
V2101_PARENT_V29_SNAPSHOT_RELATIVE = PurePosixPath("parent-import/v2_9_raw_snapshot")
V2101_PARENT_RAW_FILE_COUNT = 966
V2101_PARENT_RAW_STORAGE_BYTES = 23_559_957
V2101_PARENT_RAW_INVENTORY_SHA256 = (
    "63385589f81342822f705c47fe09ce10629a1ccc667ec13e47e7de36cec31413"
)
V2101_PARENT_RUN_LEDGER_FILE_SHA256 = (
    "33e0df2243f29a9f3e9c9994376703641588f7f5b718805e345fd53fa1f49f10"
)
V2101_PARENT_RUN_LEDGER_INTERNAL_SHA256 = (
    "75e91445745ec5480577327053a8d7eaefc4352cb6f3f176693460cc712d22b6"
)
V2101_PARENT_RUN_LEDGER_EVENT_COUNT = 213
V2101_PARENT_RUN_LEDGER_EVENT_HEAD = (
    "217dcefc5d1ad7a9a2f222735613332c530ac27623931b3e6faa298160ead995"
)
V2101_PARENT_BUDGET_LEDGER_FILE_SHA256 = (
    "692367a14d1d25cc80766442b04c19add0a9c4ef5153cd663c1752d738b0960c"
)
V2101_PARENT_BUDGET_LEDGER_INTERNAL_SHA256 = (
    "87d313e4f96766f3137c5c0175b0adb6e8a24d4c7697e556e2e0e46f00525161"
)
V2101_PARENT_BUDGET_LEDGER_EVENT_COUNT = 314
V2101_PARENT_BUDGET_LEDGER_EVENT_HEAD = (
    "ec7e88766e84c1c173e094b1569d8d409b3639a67e886ed3d37c2e17026f0a19"
)
V2101_PARENT_QREF_RECEIPT_PATH = PurePosixPath("q-ref-resolution/stage_receipt.json")
V2101_PARENT_QREF_RECEIPT_FILE_SHA256 = (
    "297996e623fbdfccf03c39fb293ea949ce19bb516c27e7a42490f8b022eaaef1"
)
V2101_PARENT_QREF_RECEIPT_CONTENT_SHA256 = (
    "8d22ec395608285dc96da65b0349389255fa6ca997b5de422b35686b496bc7db"
)
V2101_PARENT_PARENT_IMPORT_RECEIPT_FILE_SHA256 = (
    "ad7e65fdb3a78325e92b69dc1af63e778e8dbd57b8de244d95a4b7f1f20508c8"
)
V2101_PARENT_PARENT_IMPORT_RECEIPT_CONTENT_SHA256 = (
    "0a31e7a717141c3caf65be4a406300e05ec30408eb2ce987671784d1cfb1313a"
)
V2101_PARENT_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_10_1_source_manifest.json"
)
V2101_PARENT_SOURCE_MANIFEST_SCHEMA_VERSION = (
    "finevo-pilot-v2.10.1-source-manifest-v1"
)
V2101_PARENT_SOURCE_MANIFEST_FILE_SHA256 = (
    "e9360d9754cd054386ff03264c331091404555379457a59a7b01344f4a8f2d8f"
)
V2101_PARENT_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "11447dd0c231140102411eb231b8716c8f1581d0fa1533e98ccc51c3afb31426"
)
V2101_PARENT_RELEASE_ATTESTATION_PATH = (
    V2101_PARENT_RAW_ROOT / "release_attestation.json"
)
V2101_PARENT_RELEASE_ATTESTATION_SCHEMA_VERSION = (
    "finevo-scientific-release-attestation-v2"
)
V2101_PARENT_RELEASE_ATTESTATION_FILE_SHA256 = (
    "d7ac625c787a0d7287eb6a20ca3172b251966487915dee376d391494b2670443"
)
V2101_PARENT_RELEASE_ATTESTATION_CONTENT_SHA256 = (
    "c45f93e89cb5cb2e49539c3a1ba18ad8a193489fd2ab92a3bf7da0ba8869c3b3"
)

V2101_PARENT_EVIDENCE_COMMIT = "b7001a0174d1a420b592cd68976a3ca8388cb748"
V2101_PARENT_EVIDENCE_MERGE_COMMIT = "a730d0d97118a6d5cf79df66cb97cb1a32c510d9"
V2101_PARENT_EVIDENCE_ROOT = PurePosixPath("evidence/current_v2/pilot-v2.10.1")
V2101_PARENT_EVIDENCE_CHECKSUMS_FILE_SHA256 = (
    "092af0afc03bb88d1c4ccc45e7dad41f5df6d57c9fa4b535b0c69ae3e7b21b54"
)
V2101_PARENT_EVIDENCE_PACKAGE_FILE_SHA256 = (
    "f471a6e3f7a5cd024ac0c34ff9ef1cd42e333e5f141ef01db67f23f088fd3590"
)
V2101_PARENT_EVIDENCE_AGGREGATE_FILE_SHA256 = (
    "d2fe85e42d6f4b7a318a3450b51e38726da7519d6babe674a5dfd78713992232"
)
V2101_PARENT_EVIDENCE_FAILURE_FILE_SHA256 = (
    "93c208129a203196628b0bf47f6662f52fd244902de19d72041c7bcbcc2c40e8"
)
V2101_PARENT_EVIDENCE_REVIEWER_REPORT_FILE_SHA256 = (
    "53e448bff6e958f14b96e27ec09ef7e17a286ad44ae615f5f570ffbce8cca5cb"
)

V29_RAW_ROOT = _v210.V29_RAW_ROOT
V29_RAW_FILE_COUNT = _v210.V29_RAW_FILE_COUNT
V29_RAW_STORAGE_BYTES = _v210.V29_RAW_STORAGE_BYTES
V29_RAW_INVENTORY_SHA256 = _v210.V29_RAW_INVENTORY_SHA256
V29_CONTRACT_ID = _v210.V29_CONTRACT_ID
V29_CONTRACT_CANONICAL_SHA256 = _v210.V29_CONTRACT_CANONICAL_SHA256
V29_SCIENCE_TAG = _v210.V29_SCIENCE_TAG
V29_SCIENCE_COMMIT = _v210.V29_SCIENCE_COMMIT

V2102_CUMULATIVE_DEBIT = ParentBudgetDebit(
    parent_contract_sha256=V2101_PARENT_CONTRACT_CANONICAL_SHA256,
    parent_run_ledger_sha256=V2101_PARENT_RUN_LEDGER_INTERNAL_SHA256,
    parent_budget_ledger_sha256=V2101_PARENT_BUDGET_LEDGER_INTERNAL_SHA256,
    stage_bucket="parent_v23",
    cost_usd=3.212770875,
    hosted_completions=184,
    storage_bytes=92_541_342,
    record_sha256=("4af5a2c29b3dcc417e261f25b7544e9ca3198f3c3b67d43ea6fbdf50e2ccdad9"),
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PREREQUISITE_STAGES = (
    "parent-import",
    "q-ref-resolution",
    "stage0-calibration",
)
_V2101_PARENT_EXPECTED_STATUS_BY_STAGE = {
    "parent-import": {"complete": 1},
    "q-ref-resolution": {"complete": 1},
    "stage0-calibration": {"complete": 14},
    "experiment-a": {"failed": 20},
    "experiment-b": {"failed": 15},
    "experiment-c": {"complete": 5, "failed": 20},
    "experiment-d": {"failed": 30},
    "local-experiment-a": {"failed": 20},
    "local-experiment-b": {"failed": 25},
    "local-experiment-c": {"complete": 5, "failed": 20},
    "local-experiment-d": {"failed": 35},
}


class PilotV2102ParentImportError(RuntimeError):
    """Raised before any V2.10.2 imported authority can be consumed."""


def _translate(exc: Exception) -> PilotV2102ParentImportError:
    return PilotV2102ParentImportError(str(exc))


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotV2102ParentImportError(f"{name} must be an object")
    return value


def _v2101_source_manifest_binding() -> dict[str, Any]:
    return {
        "path": V2101_PARENT_SOURCE_MANIFEST_PATH.as_posix(),
        "schema_version": V2101_PARENT_SOURCE_MANIFEST_SCHEMA_VERSION,
        "file_sha256": V2101_PARENT_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": V2101_PARENT_SOURCE_MANIFEST_CONTENT_SHA256,
    }


def _v2101_release_attestation_binding() -> dict[str, Any]:
    return {
        "path": V2101_PARENT_RELEASE_ATTESTATION_PATH.as_posix(),
        "schema_version": V2101_PARENT_RELEASE_ATTESTATION_SCHEMA_VERSION,
        "file_sha256": V2101_PARENT_RELEASE_ATTESTATION_FILE_SHA256,
        "content_sha256": V2101_PARENT_RELEASE_ATTESTATION_CONTENT_SHA256,
        "status": "pass",
        "science_tag": V2101_PARENT_SCIENCE_TAG,
        "science_tag_object": V2101_PARENT_SCIENCE_TAG_OBJECT,
        "science_commit": V2101_PARENT_SCIENCE_COMMIT,
    }


def _strict_root(value: str | Path, *, name: str) -> Path:
    root = Path(value).absolute()
    for component in (root, *root.parents):
        try:
            if component.is_symlink():
                raise PilotV2102ParentImportError(f"{name} path contains a symlink")
        except OSError as exc:
            raise PilotV2102ParentImportError(f"{name} is unavailable") from exc
    try:
        return _real_root(root, name=name)
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc


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
        raise PilotV2102ParentImportError(f"{name} file hash drifted")
    return path, raw, value


def _repo_relative(
    root: Path,
    path: Path,
    *,
    required_top: str,
    name: str,
) -> str:
    try:
        relative = PurePosixPath(*path.absolute().relative_to(root).parts)
        normalized = _normalized_relative(
            relative,
            required_top=required_top,
            name=name,
        )
    except (ValueError, PilotV24ParentImportError) as exc:
        raise PilotV2102ParentImportError(f"{name} escaped the repository") from exc
    return normalized.as_posix()


def _verify_self_hashed(
    value: Mapping[str, Any],
    *,
    schema_version: str,
    name: str,
) -> None:
    try:
        _verify_self_hash(value, schema_version=schema_version, name=name)
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc


def _verify_stage_receipt_v2(
    value: Mapping[str, Any],
    *,
    name: str,
) -> None:
    candidate = _json_copy(value)
    integrity = candidate.pop("integrity", None)
    if (
        value.get("schema_version") != "finevo-pilot-stage-receipt-v2"
        or not isinstance(integrity, Mapping)
        or set(integrity) != {"canonicalization", "content_sha256"}
        or integrity.get("canonicalization") != CANONICALIZATION
        or integrity.get("content_sha256") != canonical_sha256(candidate)
    ):
        raise PilotV2102ParentImportError(f"{name} schema or content hash mismatch")


def _verify_release_identity(
    root: Path,
    *,
    tag: str,
    expected_commit: str,
    expected_tag_object: str | None,
    name: str,
) -> None:
    if _COMMIT_RE.fullmatch(expected_commit) is None:
        raise PilotV2102ParentImportError(f"{name} commit is malformed")
    try:
        head = _git(root, "rev-parse", "--verify", "HEAD^{commit}")
        tag_object = _git(
            root,
            "rev-parse",
            "--verify",
            f"refs/tags/{tag}^{{tag}}",
        )
        tag_commit = _git(
            root,
            "rev-parse",
            "--verify",
            f"refs/tags/{tag}^{{commit}}",
        )
        tracked_status = _git(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if (
        head != expected_commit
        or tag_commit != expected_commit
        or tag_object == tag_commit
        or (expected_tag_object is not None and tag_object != expected_tag_object)
        or tracked_status
    ):
        raise PilotV2102ParentImportError(f"{name} release/tag identity drifted")


def _validate_target_contract(
    contract: PilotContract,
    *,
    require_frozen: bool,
) -> None:
    implementation = _mapping(
        getattr(contract, "implementation", None),
        name="V2.10.2 implementation policy",
    )
    status = getattr(contract, "status", None)
    if (
        getattr(contract, "contract_id", None) != V2102_CONTRACT_ID
        or implementation.get("required_git_tag") != V2102_SCIENCE_TAG
        or (require_frozen and status != "frozen")
        or (not require_frozen and status not in {"draft", "frozen"})
    ):
        raise PilotV2102ParentImportError(
            "V2.10.2 import requires its exact release-bound contract"
        )


def _validate_event_chain(
    value: Mapping[str, Any],
    *,
    internal_sha256: str,
    event_count: int,
    event_head: str,
    name: str,
) -> None:
    unsigned = _json_copy(value)
    claimed = unsigned.pop("ledger_sha256", None)
    events = value.get("events")
    if (
        claimed != internal_sha256
        or canonical_sha256(unsigned) != internal_sha256
        or not isinstance(events, list)
        or len(events) != event_count
        or events[-1].get("event_sha256") != event_head
    ):
        raise PilotV2102ParentImportError(f"{name} self-hash/head drifted")
    previous = "0" * 64
    for index, source in enumerate(events):
        if not isinstance(source, Mapping):
            raise PilotV2102ParentImportError(f"{name} event is malformed")
        row = _json_copy(source)
        digest = row.pop("event_sha256", None)
        if (
            source.get("event_index") != index
            or source.get("previous_event_sha256") != previous
            or digest != canonical_sha256(row)
        ):
            raise PilotV2102ParentImportError(f"{name} event chain drifted")
        previous = str(digest)


def _load_v2101_parent_contract(parent_root: Path) -> PilotContract:
    _, raw, value = _strict_file(
        parent_root,
        V2101_PARENT_CONTRACT_PATH,
        name="frozen V2.10.1 contract",
        expected_sha256=V2101_PARENT_CONTRACT_FILE_SHA256,
    )
    if (
        value.get("contract_id") != V2101_PARENT_CONTRACT_ID
        or value.get("status") != "frozen"
        or value.get("implementation", {}).get("required_git_tag") != V2101_PARENT_SCIENCE_TAG
        or value.get("integrity", {}).get("declared_sha256")
        != V2101_PARENT_CONTRACT_CANONICAL_SHA256
        or canonical_contract_sha256(value) != V2101_PARENT_CONTRACT_CANONICAL_SHA256
        or _sha256(raw) != V2101_PARENT_CONTRACT_FILE_SHA256
    ):
        raise PilotV2102ParentImportError("frozen V2.10.1 contract identity drifted")
    contract = load_pilot_contract(parent_root.joinpath(*V2101_PARENT_CONTRACT_PATH.parts))
    if contract.canonical_hash != V2101_PARENT_CONTRACT_CANONICAL_SHA256:
        raise PilotV2102ParentImportError("parsed V2.10.1 contract identity drifted")
    return contract


def _verify_v2101_parent_source_manifest(
    parent_root: Path,
    contract: PilotContract,
) -> dict[str, Any]:
    _, raw, value = _strict_file(
        parent_root,
        V2101_PARENT_SOURCE_MANIFEST_PATH,
        name="V2.10.1 source manifest",
        expected_sha256=V2101_PARENT_SOURCE_MANIFEST_FILE_SHA256,
    )
    try:
        manifest = _v2101._validate_source_manifest_structure(value)
    except _v2101.PilotV2101ParentImportError as exc:
        raise _translate(exc) from exc
    binding = _v2101_source_manifest_binding()
    amendment = contract.qref_receipt_verifier_retry_amendment
    if (
        _sha256(raw) != V2101_PARENT_SOURCE_MANIFEST_FILE_SHA256
        or manifest.get("schema_version")
        != V2101_PARENT_SOURCE_MANIFEST_SCHEMA_VERSION
        or manifest.get("integrity", {}).get("content_sha256")
        != V2101_PARENT_SOURCE_MANIFEST_CONTENT_SHA256
        or amendment is None
        or _json_copy(amendment).get("source_manifest") != binding
    ):
        raise PilotV2102ParentImportError(
            "V2.10.1 source manifest/contract binding drifted"
        )
    return binding


def _verify_v2101_parent_release_attestation(
    raw_root: Path,
    contract: PilotContract,
) -> dict[str, Any]:
    relative = V2101_PARENT_RELEASE_ATTESTATION_PATH.relative_to(
        V2101_PARENT_RAW_ROOT
    )
    _, raw, value = _strict_file(
        raw_root,
        relative,
        name="V2.10.1 release attestation",
        expected_sha256=V2101_PARENT_RELEASE_ATTESTATION_FILE_SHA256,
    )
    unsigned = _json_copy(value)
    claimed = unsigned.pop("attestation_sha256", None)
    release_requirements = contract.release_requirements
    if release_requirements is None:
        raise PilotV2102ParentImportError(
            "V2.10.1 contract lacks frozen release requirements"
        )
    if (
        _sha256(raw) != V2101_PARENT_RELEASE_ATTESTATION_FILE_SHA256
        or value.get("schema_version")
        != V2101_PARENT_RELEASE_ATTESTATION_SCHEMA_VERSION
        or claimed != V2101_PARENT_RELEASE_ATTESTATION_CONTENT_SHA256
        or canonical_sha256(unsigned)
        != V2101_PARENT_RELEASE_ATTESTATION_CONTENT_SHA256
        or value.get("status") != "pass"
        or value.get("head_commit") != V2101_PARENT_SCIENCE_COMMIT
        or value.get("contract", {}).get("canonical_sha256")
        != V2101_PARENT_CONTRACT_CANONICAL_SHA256
        or value.get("contract", {}).get("file_sha256")
        != V2101_PARENT_CONTRACT_FILE_SHA256
        or value.get("local_tag")
        != {
            "kind": "annotated",
            "name": V2101_PARENT_SCIENCE_TAG,
            "object_id": V2101_PARENT_SCIENCE_TAG_OBJECT,
            "peeled_commit": V2101_PARENT_SCIENCE_COMMIT,
        }
        or value.get("release_requirements") != release_requirements.to_dict()
        or value.get("remote", {}).get("tag_name") != V2101_PARENT_SCIENCE_TAG
        or value.get("remote", {}).get("tag_object_id")
        != V2101_PARENT_SCIENCE_TAG_OBJECT
        or value.get("remote", {}).get("tag_peeled_commit")
        != V2101_PARENT_SCIENCE_COMMIT
        or value.get("remote", {}).get("branch_commit")
        != V2101_PARENT_SCIENCE_COMMIT
    ):
        raise PilotV2102ParentImportError(
            "V2.10.1 scientific release attestation drifted"
        )
    return _v2101_release_attestation_binding()


def _verify_v2101_parent_raw_inventory(raw_root: Path) -> list[dict[str, Any]]:
    try:
        rows, summary = _inventory(raw_root, declared_root=V2101_PARENT_RAW_ROOT)
    except Exception as exc:
        raise _translate(exc) from exc
    expected = {
        "root": V2101_PARENT_RAW_ROOT.as_posix(),
        "inventory_schema_version": "finevo-raw-tree-inventory-v1",
        "inventory_canonicalization": "json-sort-keys-compact-utf8-v1",
        "file_count": V2101_PARENT_RAW_FILE_COUNT,
        "storage_bytes": V2101_PARENT_RAW_STORAGE_BYTES,
        "inventory_sha256": V2101_PARENT_RAW_INVENTORY_SHA256,
    }
    if summary != expected:
        raise PilotV2102ParentImportError("V2.10.1 raw-tree inventory drifted")
    return rows


def _verify_v2101_parent_ledgers(
    raw_root: Path,
    source_contract: PilotContract,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, _, run = _strict_file(
        raw_root,
        PurePosixPath("run_ledger.json"),
        name="V2.10.1 run ledger",
        expected_sha256=V2101_PARENT_RUN_LEDGER_FILE_SHA256,
    )
    _, _, budget = _strict_file(
        raw_root,
        PurePosixPath("budget_ledger.json"),
        name="V2.10.1 budget ledger",
        expected_sha256=V2101_PARENT_BUDGET_LEDGER_FILE_SHA256,
    )
    _validate_event_chain(
        run,
        internal_sha256=V2101_PARENT_RUN_LEDGER_INTERNAL_SHA256,
        event_count=V2101_PARENT_RUN_LEDGER_EVENT_COUNT,
        event_head=V2101_PARENT_RUN_LEDGER_EVENT_HEAD,
        name="V2.10.1 run ledger",
    )
    _validate_event_chain(
        budget,
        internal_sha256=V2101_PARENT_BUDGET_LEDGER_INTERNAL_SHA256,
        event_count=V2101_PARENT_BUDGET_LEDGER_EVENT_COUNT,
        event_head=V2101_PARENT_BUDGET_LEDGER_EVENT_HEAD,
        name="V2.10.1 budget ledger",
    )
    expected = {spec.run_id: spec.to_dict() for spec in source_contract.expand()}
    runs = _mapping(run.get("runs"), name="V2.10.1 run rows")
    if len(runs) != 211 or set(runs) != set(expected):
        raise PilotV2102ParentImportError(
            "V2.10.1 terminal ledger differs from its 211-cell denominator"
        )
    status_by_stage: Counter[tuple[str, str]] = Counter()
    for run_id, row in runs.items():
        if (
            not isinstance(row, Mapping)
            or row.get("spec") != expected[run_id]
            or row.get("status") not in {"complete", "failed"}
        ):
            raise PilotV2102ParentImportError(
                "V2.10.1 terminal ledger row identity/status drifted"
            )
        status_by_stage[(str(row["spec"]["stage_id"]), str(row["status"]))] += 1
    observed = {
        stage: {
            status: count
            for (candidate, status), count in status_by_stage.items()
            if candidate == stage
        }
        for stage in {candidate for candidate, _ in status_by_stage}
    }
    if observed != _V2101_PARENT_EXPECTED_STATUS_BY_STAGE:
        raise PilotV2102ParentImportError("V2.10.1 terminal status denominator drifted")

    # V2.10.1 created 156 dispatch units because the two Experiment-D lanes
    # share one receipt per seed.  Every fresh unit finalized with zero actual
    # completions/cost; non-zero reservations are retained as failure evidence.
    budget_runs = _mapping(budget.get("runs"), name="V2.10.1 budget rows")
    if (
        len(budget_runs) != 156
        or budget.get("parent_debit") != _v2101.V2101_CUMULATIVE_DEBIT.to_dict()
        or Counter(row.get("status") for row in budget_runs.values())
        != Counter({"complete": 26, "failed": 130})
        or any(
            row.get("actual", {}).get("completions") != 0
            or row.get("actual", {}).get("cost_usd") != 0.0
            for row in budget_runs.values()
        )
    ):
        raise PilotV2102ParentImportError("V2.10.1 zero-provider budget boundary drifted")
    return _json_copy(run), _json_copy(budget)


def _verify_v2101_parent_qref_receipt(raw_root: Path) -> dict[str, Any]:
    _, raw, value = _strict_file(
        raw_root,
        V2101_PARENT_QREF_RECEIPT_PATH,
        name="V2.10.1 q-ref receipt",
        expected_sha256=V2101_PARENT_QREF_RECEIPT_FILE_SHA256,
    )
    _verify_stage_receipt_v2(value, name="V2.10.1 q-ref receipt")
    if (
        _sha256(raw) != V2101_PARENT_QREF_RECEIPT_FILE_SHA256
        or value["integrity"]["content_sha256"]
        != V2101_PARENT_QREF_RECEIPT_CONTENT_SHA256
        or value.get("contract_id") != V2101_PARENT_CONTRACT_ID
        or value.get("contract_sha256") != V2101_PARENT_CONTRACT_CANONICAL_SHA256
        or value.get("stage_id") != "q-ref-resolution"
        or value.get("status") != "complete"
        or value.get("registered_run_count") != 1
        or value.get("complete_cell_count") != 1
        or value.get("status_counts") != {"complete": 1}
        or value.get("go") is not True
        or value.get("execution_progression_go") is not True
        or value.get("failure") is not None
        or value.get("artifacts", {}).get("provider_calls_current_attempt") != 0
    ):
        raise PilotV2102ParentImportError("V2.10.1 q-ref receipt semantics drifted")
    return _json_copy(value)


def _verify_v2101_parent_import_zero_provider(raw_root: Path) -> dict[str, Any]:
    _, raw, value = _strict_file(
        raw_root,
        PurePosixPath("parent-import/parent_import_receipt.json"),
        name="V2.10.1 parent import receipt",
        expected_sha256=V2101_PARENT_PARENT_IMPORT_RECEIPT_FILE_SHA256,
    )
    _verify_self_hashed(
        value,
        schema_version="finevo-pilot-v2.10.1-parent-import-v1",
        name="V2.10.1 parent import receipt",
    )
    boundary = value.get("provider_construction_during_import")
    if (
        _sha256(raw) != V2101_PARENT_PARENT_IMPORT_RECEIPT_FILE_SHA256
        or value["integrity"]["content_sha256"]
        != V2101_PARENT_PARENT_IMPORT_RECEIPT_CONTENT_SHA256
        or boundary is not False
        or value.get("provider_calls_during_import") != 0
        or value.get("hosted_provider_calls_during_import") != 0
        or value.get("hosted_cost_usd_during_import") != 0.0
        or value.get("scientific_evidence") is not False
    ):
        raise PilotV2102ParentImportError(
            "V2.10.1 parent import provider boundary drifted"
        )
    return _json_copy(value)


def _verify_v2101_evidence(evidence_repo_root: Path) -> dict[str, Any]:
    base = evidence_repo_root.joinpath(*V2101_PARENT_EVIDENCE_ROOT.parts)
    if base.is_symlink() or not base.is_dir():
        raise PilotV2102ParentImportError("V2.10.1 evidence package is unavailable")
    _, _, checksums = _strict_file(
        evidence_repo_root,
        V2101_PARENT_EVIDENCE_ROOT / "checksums.json",
        name="V2.10.1 evidence checksums",
        expected_sha256=V2101_PARENT_EVIDENCE_CHECKSUMS_FILE_SHA256,
    )
    _, _, package = _strict_file(
        evidence_repo_root,
        V2101_PARENT_EVIDENCE_ROOT / "package_manifest.json",
        name="V2.10.1 evidence package manifest",
        expected_sha256=V2101_PARENT_EVIDENCE_PACKAGE_FILE_SHA256,
    )
    _, _, aggregate = _strict_file(
        evidence_repo_root,
        V2101_PARENT_EVIDENCE_ROOT / "aggregate.json",
        name="V2.10.1 evidence aggregate",
        expected_sha256=V2101_PARENT_EVIDENCE_AGGREGATE_FILE_SHA256,
    )
    _, _, failures = _strict_file(
        evidence_repo_root,
        V2101_PARENT_EVIDENCE_ROOT / "failure_ledger.json",
        name="V2.10.1 evidence failure ledger",
        expected_sha256=V2101_PARENT_EVIDENCE_FAILURE_FILE_SHA256,
    )
    try:
        _, reviewer_raw = _guarded_file(
            evidence_repo_root,
            V2101_PARENT_EVIDENCE_ROOT / "reviewer_report.md",
            name="V2.10.1 reviewer report",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if _sha256(reviewer_raw) != V2101_PARENT_EVIDENCE_REVIEWER_REPORT_FILE_SHA256:
        raise PilotV2102ParentImportError("V2.10.1 reviewer report hash drifted")

    files = checksums.get("files")
    if not isinstance(files, list) or len(files) != 19:
        raise PilotV2102ParentImportError("V2.10.1 evidence checksum inventory drifted")
    seen: set[str] = set()
    for row in files:
        if not isinstance(row, Mapping):
            raise PilotV2102ParentImportError(
                "V2.10.1 evidence checksum row is malformed"
            )
        try:
            relative = _normalized_relative(
                str(row.get("path", "")),
                required_top=None,
                name="V2.10.1 evidence checksum path",
            )
            _, artifact = _guarded_file(
                base,
                relative,
                name=f"V2.10.1 evidence {relative.as_posix()}",
            )
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
        if (
            relative.as_posix() in seen
            or _sha256(artifact) != row.get("sha256")
            or len(artifact) != row.get("byte_size")
        ):
            raise PilotV2102ParentImportError(
                "V2.10.1 evidence checksum verification failed"
            )
        seen.add(relative.as_posix())

    denominator = _mapping(
        aggregate.get("denominator"),
        name="V2.10.1 evidence denominator",
    )
    budget = _mapping(aggregate.get("budget"), name="V2.10.1 evidence budget")
    actual = _mapping(
        budget.get("actual_totals"),
        name="V2.10.1 evidence budget totals",
    )
    failure_denominator = _mapping(
        failures.get("denominator"),
        name="V2.10.1 evidence failure denominator",
    )
    implementation_failure = _mapping(
        aggregate.get("implementation_failure"),
        name="V2.10.1 implementation failure",
    )
    observed_failure = _mapping(
        implementation_failure.get("observed_failure"),
        name="V2.10.1 observed implementation failure",
    )
    provider_boundary = _mapping(
        implementation_failure.get("provider_boundary"),
        name="V2.10.1 implementation provider boundary",
    )
    outcome_boundary = _mapping(
        implementation_failure.get("outcome_boundary"),
        name="V2.10.1 implementation outcome boundary",
    )
    retry_boundary = _mapping(
        implementation_failure.get("retry_boundary"),
        name="V2.10.1 implementation retry boundary",
    )
    if (
        aggregate.get("contract_id") != V2101_PARENT_CONTRACT_ID
        or aggregate.get("contract_sha256") != V2101_PARENT_CONTRACT_CANONICAL_SHA256
        or aggregate.get("resolved_git_commit") != V2101_PARENT_SCIENCE_COMMIT
        or aggregate.get("publication_status") != "complete-with-no-go"
        or aggregate.get("scientific_complete") is not False
        or aggregate.get("scientific_claim_gates_supported") is not False
        or denominator.get("expected_count") != 211
        or denominator.get("status_counts") != {"complete": 26, "failed": 185}
        or failure_denominator != denominator
        or actual
        != {
            "completions": 184,
            "cost_usd": 3.212770875,
            "storage_bytes": 92_541_342,
        }
        or package.get("contract_id") != V2101_PARENT_CONTRACT_ID
        or package.get("contract_sha256") != V2101_PARENT_CONTRACT_CANONICAL_SHA256
        or package.get("resolved_git_commit") != V2101_PARENT_SCIENCE_COMMIT
        or package.get("publication_status") != "complete-with-no-go"
        or package.get("scientific_complete") is not False
        or package.get("scientific_claim_gates_supported") is not False
        or len(failures.get("rows", [])) != 185
        or implementation_failure.get("root_cause_code")
        != "observed-p95-consumer-schema-dispatch-gap"
        or implementation_failure.get("classification")
        != "implementation-interface-no-go"
        or observed_failure.get("error_type") != "ValueError"
        or observed_failure.get("failed_cell_count") != 185
        or observed_failure.get("message_sha256")
        != "39cb7f19f94e435d9eb4873df49beac2507703522f2ad9ffa7f688a5f6b92ef7"
        or provider_boundary.get("failure_phase")
        != "before-provider-construction-and-dispatch"
        or provider_boundary.get("v2_10_1_fresh_provider_calls") != 0
        or provider_boundary.get("v2_10_1_incremental_hosted_completions") != 0
        or provider_boundary.get("v2_10_1_incremental_hosted_cost_usd") != 0.0
        or outcome_boundary.get("offline_candidate_admission_cells_generated") != 10
        or outcome_boundary.get("offline_candidate_metrics_observed") is not True
        or outcome_boundary.get("offline_candidate_metrics_inspected") is not True
        or outcome_boundary.get("actor_performance_treatment_outcome_blind") is not True
        or outcome_boundary.get("global_a_d_outcome_blind") is not False
        or retry_boundary.get("successor_contract_id") != V2102_CONTRACT_ID
        or retry_boundary.get("fresh_a_d_cells_required") != 195
        or retry_boundary.get("offline_candidate_cells_fresh_rerun_required") != 10
        or retry_boundary.get("offline_candidate_cells_imported") != 0
    ):
        raise PilotV2102ParentImportError(
            "V2.10.1 evidence semantics/claim boundary drifted"
        )
    return {
        "publication_commit": V2101_PARENT_EVIDENCE_COMMIT,
        "merge_commit": V2101_PARENT_EVIDENCE_MERGE_COMMIT,
        "root": V2101_PARENT_EVIDENCE_ROOT.as_posix(),
        "checksums_file_sha256": V2101_PARENT_EVIDENCE_CHECKSUMS_FILE_SHA256,
        "package_manifest_file_sha256": V2101_PARENT_EVIDENCE_PACKAGE_FILE_SHA256,
        "aggregate_file_sha256": V2101_PARENT_EVIDENCE_AGGREGATE_FILE_SHA256,
        "failure_ledger_file_sha256": V2101_PARENT_EVIDENCE_FAILURE_FILE_SHA256,
        "reviewer_report_file_sha256": (V2101_PARENT_EVIDENCE_REVIEWER_REPORT_FILE_SHA256),
        "terminal_status": "complete-with-no-go",
        "status_counts": {"complete": 26, "failed": 185},
        "v2_10_1_incremental_hosted_completions": 0,
        "v2_10_1_incremental_hosted_stage_cost_usd": 0.0,
        "offline_candidate_admission_cells_observed": 10,
        "actor_performance_treatment_outcome_blind": True,
        "scientific_claim_gates_supported": False,
    }


def verify_v2101_terminal_lineage(
    *,
    parent_repo_root: str | Path,
    evidence_repo_root: str | Path,
) -> dict[str, Any]:
    """Verify the exact V2.10.1 no-go release without importing its wrappers."""

    parent = _strict_root(parent_repo_root, name="V2.10.1 source repository")
    evidence_root = _strict_root(
        evidence_repo_root,
        name="V2.10.1 evidence repository",
    )
    _verify_release_identity(
        parent,
        tag=V2101_PARENT_SCIENCE_TAG,
        expected_commit=V2101_PARENT_SCIENCE_COMMIT,
        expected_tag_object=V2101_PARENT_SCIENCE_TAG_OBJECT,
        name="V2.10.1 source",
    )
    contract = _load_v2101_parent_contract(parent)
    source_manifest = _verify_v2101_parent_source_manifest(parent, contract)
    raw_root = parent.joinpath(*V2101_PARENT_RAW_ROOT.parts)
    inventory = _verify_v2101_parent_raw_inventory(raw_root)
    release_attestation = _verify_v2101_parent_release_attestation(
        raw_root,
        contract,
    )
    run_ledger, budget_ledger = _verify_v2101_parent_ledgers(raw_root, contract)
    qref_receipt = _verify_v2101_parent_qref_receipt(raw_root)
    parent_import = _verify_v2101_parent_import_zero_provider(raw_root)
    evidence = _verify_v2101_evidence(evidence_root)
    return {
        "source_contract": contract,
        "source_raw_root": raw_root,
        "source_manifest": source_manifest,
        "release_attestation": release_attestation,
        "raw_inventory": {
            "root": V2101_PARENT_RAW_ROOT.as_posix(),
            "file_count": V2101_PARENT_RAW_FILE_COUNT,
            "storage_bytes": V2101_PARENT_RAW_STORAGE_BYTES,
            "inventory_sha256": V2101_PARENT_RAW_INVENTORY_SHA256,
            "rows": inventory,
        },
        "run_ledger": run_ledger,
        "budget_ledger": budget_ledger,
        "qref_receipt": qref_receipt,
        "parent_import_receipt": parent_import,
        "evidence": evidence,
        "provider_construction_during_import": False,
        "provider_calls_during_import": 0,
        "hosted_provider_calls_during_import": 0,
        "hosted_cost_usd_during_import": 0.0,
    }


def _normalized_spec(
    spec: PilotRunSpec | Mapping[str, Any],
) -> dict[str, Any]:
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
        raise PilotV2102ParentImportError(
            f"{name} source/target matrix is not an exact normalized match"
        )
    return [(source_map[key], target_map[key]) for key in sorted(source_map)]


def build_v2102_prerequisite_bindings(
    *,
    source_contract: PilotContract,
    target_contract: PilotContract,
    source_run_ledger: Mapping[str, Any],
    source_raw_root: str | Path,
) -> list[dict[str, Any]]:
    """Bind exactly sixteen target specs to byte-exact V2.9 artifacts."""

    _validate_target_contract(target_contract, require_frozen=False)
    if (
        source_contract.contract_id != V29_CONTRACT_ID
        or source_contract.canonical_hash != V29_CONTRACT_CANONICAL_SHA256
    ):
        raise PilotV2102ParentImportError("V2.9 source contract identity drifted")
    runs = _mapping(source_run_ledger.get("runs"), name="V2.9 run rows")
    raw_root = Path(source_raw_root).absolute()
    result: list[dict[str, Any]] = []
    for stage_id in _PREREQUISITE_STAGES:
        pairs = _match_specs(
            source_contract.expand(stage=stage_id),
            target_contract.expand(stage=stage_id),
            name=f"V2.10.2 {stage_id} prerequisite",
        )
        for source_spec, target_spec in pairs:
            row = _mapping(
                runs.get(source_spec.run_id),
                name=f"V2.9 prerequisite ledger row {source_spec.run_id}",
            )
            if (
                row.get("status") != "complete"
                or row.get("spec") != source_spec.to_dict()
            ):
                raise PilotV2102ParentImportError(
                    "only complete V2.9 prerequisite rows may be imported"
                )
            result.append(
                {
                    "stage_id": stage_id,
                    "source_authority_contract_id": V29_CONTRACT_ID,
                    "source_run_id": source_spec.run_id,
                    "target_run_id": target_spec.run_id,
                    "source_spec": source_spec.to_dict(),
                    "target_spec": target_spec.to_dict(),
                    "source_artifacts": _v210._prerequisite_source_artifacts(
                        raw_root,
                        source_spec,
                    ),
                    "source_path_kind": (
                        "byte-exact-v2.9-raw-inside-v2.10.1-terminal-snapshot"
                    ),
                    "treatment_effect_evidence": False,
                    "provider_construction_during_import": False,
                    "provider_calls_during_import": 0,
                }
            )
    counts = Counter(row["stage_id"] for row in result)
    if len(result) != 16 or counts != Counter(
        {
            "parent-import": 1,
            "q-ref-resolution": 1,
            "stage0-calibration": 14,
        }
    ):
        raise PilotV2102ParentImportError(
            "V2.10.2 imported prerequisite inventory drifted"
        )
    return sorted(result, key=lambda row: row["target_run_id"])


def _v29_snapshot_inside_v2101(parent_root: Path) -> Path:
    return parent_root.joinpath(
        *V2101_PARENT_RAW_ROOT.parts,
        *V2101_PARENT_V29_SNAPSHOT_RELATIVE.parts,
    )


def verify_v29_exact_source_for_v2102(
    *,
    parent_repo_root: str | Path,
    evidence_repo_root: str | Path,
    target_contract: PilotContract,
    terminal_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read V2.9 raw directly; V2.10.1 p95 wrappers are never accepted."""

    parent = _strict_root(parent_repo_root, name="V2.10.1 source repository")
    _validate_target_contract(target_contract, require_frozen=False)
    terminal = (
        _mapping(terminal_audit, name="V2.10.1 terminal audit")
        if terminal_audit is not None
        else verify_v2101_terminal_lineage(
            parent_repo_root=parent,
            evidence_repo_root=evidence_repo_root,
        )
    )
    raw_root = _v29_snapshot_inside_v2101(parent)
    try:
        inventory = _v210._verify_exact_v29_inventory(raw_root)
        source_contract = _v210._load_verified_v29_contract(parent)
        run_ledger, budget_ledger = _v210._verify_v29_ledgers(
            raw_root,
            source_contract,
        )
        evidence = _v210._verify_v29_evidence(
            _strict_root(
                evidence_repo_root,
                name="V2.9 evidence repository",
            )
        )
    except _v210.PilotV210ParentImportError as exc:
        raise _translate(exc) from exc
    imported = build_v2102_prerequisite_bindings(
        source_contract=source_contract,
        target_contract=target_contract,
        source_run_ledger=run_ledger,
        source_raw_root=raw_root,
    )
    p95: dict[str, Any] = {}
    for profile_id in V2102_ALLOWED_P95_PROFILES:
        try:
            nested = verify_v29_imported_v28_observed_p95(
                raw_root,
                profile_id,
                expected_parent_commit=V28_SCIENCE_COMMIT,
            )
            normalized = _v210.normalize_v29_observed_p95_binding(
                nested,
                profile_id=profile_id,
            )
        except Exception as exc:
            raise _translate(exc) from exc
        p95[profile_id] = {
            "source_path_kind": ("byte-exact-v2.9-raw-inside-v2.10.1-terminal-snapshot"),
            "v2_9_terminal_parent": {
                "contract_id": V29_CONTRACT_ID,
                "contract_sha256": V29_CONTRACT_CANONICAL_SHA256,
                "science_tag": V29_SCIENCE_TAG,
                "science_commit": V29_SCIENCE_COMMIT,
                "raw_file_count": V29_RAW_FILE_COUNT,
                "raw_storage_bytes": V29_RAW_STORAGE_BYTES,
                "raw_inventory_sha256": V29_RAW_INVENTORY_SHA256,
                "terminal_status": "complete-with-no-go",
                "implementation_root_cause": (
                    "imported-p95-runner-binding-shape-mismatch"
                ),
            },
            "v2_8_observed_p95_origin": _json_copy(nested),
            "normalized_v2_9_binding": normalized,
        }
    if (
        terminal.get("provider_calls_during_import") != 0
        or terminal.get("provider_construction_during_import") is not False
    ):
        raise PilotV2102ParentImportError(
            "V2.10.1 terminal audit lost its zero-provider boundary"
        )
    return {
        "source_contract": source_contract,
        "source_raw_root": raw_root,
        "raw_inventory": {
            "root": V29_RAW_ROOT.as_posix(),
            "file_count": V29_RAW_FILE_COUNT,
            "storage_bytes": V29_RAW_STORAGE_BYTES,
            "inventory_sha256": V29_RAW_INVENTORY_SHA256,
            "rows": inventory,
        },
        "run_ledger": run_ledger,
        "budget_ledger": budget_ledger,
        "evidence": evidence,
        "imported_cells": imported,
        "p95_sources": p95,
        "budget_debit": V2102_CUMULATIVE_DEBIT.to_dict(),
        "provider_construction_during_import": False,
        "provider_calls_during_import": 0,
        "hosted_provider_calls_during_import": 0,
        "hosted_cost_usd_during_import": 0.0,
    }


def build_v2102_source_manifest(
    *,
    parent_repo_root: str | Path,
    evidence_repo_root: str | Path,
    target_contract: PilotContract,
) -> dict[str, Any]:
    """Build a deterministic V2.10.2 manifest from frozen terminal sources."""

    terminal = verify_v2101_terminal_lineage(
        parent_repo_root=parent_repo_root,
        evidence_repo_root=evidence_repo_root,
    )
    source = verify_v29_exact_source_for_v2102(
        parent_repo_root=parent_repo_root,
        evidence_repo_root=evidence_repo_root,
        target_contract=target_contract,
        terminal_audit=terminal,
    )
    return _seal(
        {
            "schema_version": V2102_SOURCE_MANIFEST_SCHEMA_VERSION,
            "v2_10_1_terminal_parent": {
                "contract": {
                    "contract_id": V2101_PARENT_CONTRACT_ID,
                    "path": V2101_PARENT_CONTRACT_PATH.as_posix(),
                    "schema_version": "finevo-pilot-contract-v2",
                    "status": "frozen",
                    "file_sha256": V2101_PARENT_CONTRACT_FILE_SHA256,
                    "canonical_sha256": V2101_PARENT_CONTRACT_CANONICAL_SHA256,
                },
                "release": {
                    "science_tag": V2101_PARENT_SCIENCE_TAG,
                    "science_tag_object": V2101_PARENT_SCIENCE_TAG_OBJECT,
                    "science_commit": V2101_PARENT_SCIENCE_COMMIT,
                    "tag_kind": "annotated",
                    "raw_root": V2101_PARENT_RAW_ROOT.as_posix(),
                    "release_attestation": terminal["release_attestation"],
                },
                "source_manifest": terminal["source_manifest"],
                "raw_snapshot": {
                    "root": V2101_PARENT_RAW_ROOT.as_posix(),
                    "inventory_schema_version": ("finevo-raw-tree-inventory-v1"),
                    "inventory_canonicalization": ("json-sort-keys-compact-utf8-v1"),
                    "file_count": V2101_PARENT_RAW_FILE_COUNT,
                    "storage_bytes": V2101_PARENT_RAW_STORAGE_BYTES,
                    "inventory_sha256": V2101_PARENT_RAW_INVENTORY_SHA256,
                },
                "ledgers": {
                    "run": {
                        "path": (V2101_PARENT_RAW_ROOT / "run_ledger.json").as_posix(),
                        "file_sha256": V2101_PARENT_RUN_LEDGER_FILE_SHA256,
                        "internal_sha256": V2101_PARENT_RUN_LEDGER_INTERNAL_SHA256,
                        "event_count": V2101_PARENT_RUN_LEDGER_EVENT_COUNT,
                        "event_chain_head": V2101_PARENT_RUN_LEDGER_EVENT_HEAD,
                    },
                    "budget": {
                        "path": (V2101_PARENT_RAW_ROOT / "budget_ledger.json").as_posix(),
                        "file_sha256": V2101_PARENT_BUDGET_LEDGER_FILE_SHA256,
                        "internal_sha256": (V2101_PARENT_BUDGET_LEDGER_INTERNAL_SHA256),
                        "event_count": V2101_PARENT_BUDGET_LEDGER_EVENT_COUNT,
                        "event_chain_head": V2101_PARENT_BUDGET_LEDGER_EVENT_HEAD,
                    },
                },
                "qref_receipt": {
                    "path": (
                        V2101_PARENT_RAW_ROOT / V2101_PARENT_QREF_RECEIPT_PATH
                    ).as_posix(),
                    "file_sha256": V2101_PARENT_QREF_RECEIPT_FILE_SHA256,
                    "content_sha256": V2101_PARENT_QREF_RECEIPT_CONTENT_SHA256,
                    "status": "complete",
                    "provider_calls_current_attempt": 0,
                },
                "terminal_denominator": {
                    "registered_cells": 211,
                    "terminal_cells": 211,
                    "all_rows_present": True,
                    "all_rows_terminal": True,
                    "status_counts": {
                        "complete": 26,
                        "failed": 185,
                    },
                    "terminal_status": "complete-with-no-go",
                    "scientific_complete": False,
                    "scientific_matrix_complete": False,
                    "scientific_claim_gates_supported": False,
                    "provider_calls": 0,
                    "hosted_stage_cost_usd": 0.0,
                    "offline_candidate_admission_cells_observed": 10,
                    "actor_performance_treatment_outcome_blind": True,
                    "implementation_root_cause": (
                        "observed-p95-consumer-schema-dispatch-gap"
                    ),
                },
            },
            "published_v2_10_1_evidence": terminal["evidence"],
            "v2_9_exact_source": {
                "source_path_kind": (
                    "byte-exact-v2.9-raw-inside-v2.10.1-terminal-snapshot"
                ),
                "source_path": (V2101_PARENT_RAW_ROOT / V2101_PARENT_V29_SNAPSHOT_RELATIVE).as_posix(),
                "declared_root": V29_RAW_ROOT.as_posix(),
                "file_count": V29_RAW_FILE_COUNT,
                "storage_bytes": V29_RAW_STORAGE_BYTES,
                "inventory_sha256": V29_RAW_INVENTORY_SHA256,
                "v2_10_1_wrapper_is_current_authority": False,
            },
            "imported_complete_cells": source["imported_cells"],
            "v2_9_p95_sources_for_current_release_reseal": (source["p95_sources"]),
            "cumulative_budget_debit": V2102_CUMULATIVE_DEBIT.to_dict(),
            "import_policy": {
                "source_raw_namespace": V29_RAW_ROOT.as_posix(),
                "child_raw_namespace": V2102_RAW_ROOT.as_posix(),
                "child_snapshot_namespace": (
                    V2102_RAW_ROOT / V2102_SNAPSHOT_RELATIVE
                ).as_posix(),
                "exact_full_v2_9_raw_snapshot_copy": True,
                "v2_10_1_terminal_lineage_verified": True,
                "v2_10_1_current_wrapper_imported_as_authority": False,
                "imported_cell_count": 16,
                "imported_cell_breakdown": {
                    "parent-import": 1,
                    "q-ref-resolution": 1,
                    "stage0-calibration": 14,
                },
                "q_ref_value": 63.50397933257746,
                "stage0_selected_profile_id": "nu-0.5",
                "provider_construction_during_import": False,
                "provider_calls_during_import": 0,
                "hosted_provider_calls_during_import": 0,
                "hosted_cost_usd_during_import": 0.0,
                "offline_candidate_admission_cells_imported": 0,
                "failed_actor_cells_retried_via_import": 0,
                "prerequisites_are_treatment_effect_evidence": False,
            },
            "p95_reseal_policy": {
                "source_via_exact_v2_9_snapshot": True,
                "source_origin_contract_id": V28_CONTRACT_ID,
                "source_origin_contract_sha256": (V28_CONTRACT_CANONICAL_SHA256),
                "source_reservation_values_unchanged": True,
                "v2_10_1_current_wrapper_direct_reuse": "forbidden",
                "current_release_receipt_and_projection_required": True,
                "current_release_flat_runner_binding_required": True,
                "validation_before_provider_construction": True,
            },
            "observation_boundary": {
                "v2_10_1_no_go_observed_before_retry": True,
                "v2_10_1_actor_performance_outcomes_generated": False,
                "v2_10_1_offline_candidate_admission_cells_observed": 10,
                "v2_10_1_offline_candidate_metrics_descriptive_only": True,
                "v2_10_1_incremental_provider_calls": 0,
                "v2_10_1_incremental_hosted_cost_usd": 0.0,
                "v2_9_prerequisites_preserved_without_relabelling": True,
                "a_d_cells_fresh_in_v2_10_2": 195,
                "offline_candidate_cells_fresh_in_v2_10_2": 10,
                "failed_seed_replacement": "forbidden",
            },
        }
    )


def _validate_source_manifest_structure(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = _json_copy(_mapping(value, name="V2.10.2 source manifest"))
    expected_keys = {
        "schema_version",
        "v2_10_1_terminal_parent",
        "published_v2_10_1_evidence",
        "v2_9_exact_source",
        "imported_complete_cells",
        "v2_9_p95_sources_for_current_release_reseal",
        "cumulative_budget_debit",
        "import_policy",
        "p95_reseal_policy",
        "observation_boundary",
        "integrity",
    }
    if (
        set(candidate) != expected_keys
        or candidate.get("schema_version") != V2102_SOURCE_MANIFEST_SCHEMA_VERSION
    ):
        raise PilotV2102ParentImportError(
            "V2.10.2 source manifest shape/schema drifted"
        )
    _verify_self_hashed(
        candidate,
        schema_version=V2102_SOURCE_MANIFEST_SCHEMA_VERSION,
        name="V2.10.2 source manifest",
    )
    parent = _mapping(
        candidate.get("v2_10_1_terminal_parent"),
        name="V2.10.2 source V2.10.1 parent",
    )
    parent_release = _mapping(
        parent.get("release"),
        name="V2.10.2 source V2.10.1 release",
    )
    terminal = _mapping(
        parent.get("terminal_denominator"),
        name="V2.10.2 source terminal denominator",
    )
    source = _mapping(
        candidate.get("v2_9_exact_source"),
        name="V2.10.2 exact V2.9 source",
    )
    policy = _mapping(
        candidate.get("import_policy"),
        name="V2.10.2 source import policy",
    )
    p95_policy = _mapping(
        candidate.get("p95_reseal_policy"),
        name="V2.10.2 p95 reseal policy",
    )
    rows = candidate.get("imported_complete_cells")
    p95 = candidate.get("v2_9_p95_sources_for_current_release_reseal")
    if (
        parent.get("contract", {}).get("canonical_sha256")
        != V2101_PARENT_CONTRACT_CANONICAL_SHA256
        or parent.get("source_manifest") != _v2101_source_manifest_binding()
        or parent_release.get("science_tag_object")
        != V2101_PARENT_SCIENCE_TAG_OBJECT
        or parent_release.get("science_commit") != V2101_PARENT_SCIENCE_COMMIT
        or parent_release.get("release_attestation")
        != _v2101_release_attestation_binding()
        or parent.get("raw_snapshot", {}).get("inventory_sha256")
        != V2101_PARENT_RAW_INVENTORY_SHA256
        or terminal.get("status_counts") != {"complete": 26, "failed": 185}
        or terminal.get("provider_calls") != 0
        or terminal.get("hosted_stage_cost_usd") != 0.0
        or terminal.get("offline_candidate_admission_cells_observed") != 10
        or terminal.get("actor_performance_treatment_outcome_blind") is not True
        or terminal.get("implementation_root_cause")
        != "observed-p95-consumer-schema-dispatch-gap"
        or source.get("inventory_sha256") != V29_RAW_INVENTORY_SHA256
        or source.get("v2_10_1_wrapper_is_current_authority") is not False
        or not isinstance(rows, list)
        or len(rows) != 16
        or Counter(row.get("stage_id") for row in rows)
        != Counter(
            {
                "parent-import": 1,
                "q-ref-resolution": 1,
                "stage0-calibration": 14,
            }
        )
        or any(
            row.get("source_path_kind")
            != "byte-exact-v2.9-raw-inside-v2.10.1-terminal-snapshot"
            for row in rows
        )
        or not isinstance(p95, Mapping)
        or set(p95) != set(V2102_ALLOWED_P95_PROFILES)
        or candidate.get("cumulative_budget_debit") != V2102_CUMULATIVE_DEBIT.to_dict()
        or policy.get("imported_cell_count") != 16
        or policy.get("provider_construction_during_import") is not False
        or policy.get("provider_calls_during_import") != 0
        or policy.get("hosted_provider_calls_during_import") != 0
        or policy.get("hosted_cost_usd_during_import") != 0.0
        or policy.get("v2_10_1_current_wrapper_imported_as_authority") is not False
        or p95_policy.get("v2_10_1_current_wrapper_direct_reuse") != "forbidden"
    ):
        raise PilotV2102ParentImportError(
            "V2.10.2 source manifest authority/claim boundary drifted"
        )
    for profile_id in V2102_ALLOWED_P95_PROFILES:
        profile_source = _mapping(
            p95[profile_id],
            name=f"V2.10.2 source p95 {profile_id}",
        )
        if set(profile_source) != {
            "source_path_kind",
            "v2_9_terminal_parent",
            "v2_8_observed_p95_origin",
            "normalized_v2_9_binding",
        }:
            raise PilotV2102ParentImportError("V2.10.2 p95 source shape drifted")
        try:
            normalized = _v210.normalize_v29_observed_p95_binding(
                _mapping(
                    profile_source.get("v2_8_observed_p95_origin"),
                    name=f"V2.10.2 p95 origin {profile_id}",
                ),
                profile_id=profile_id,
            )
        except _v210.PilotV210ParentImportError as exc:
            raise _translate(exc) from exc
        if (
            profile_source.get("source_path_kind")
            != "byte-exact-v2.9-raw-inside-v2.10.1-terminal-snapshot"
            or profile_source.get("normalized_v2_9_binding") != normalized
        ):
            raise PilotV2102ParentImportError(
                f"V2.10.2 source p95 normalization drifted: {profile_id}"
            )
    return candidate


def write_v2102_source_manifest_draft(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    value = _validate_source_manifest_structure(manifest)
    target = Path(path)
    if target.exists():
        raise PilotV2102ParentImportError(
            f"refusing to overwrite V2.10.2 source manifest: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(
        (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )
    return target


def load_v2102_source_manifest(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
    expected_content_sha256: str | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise PilotV2102ParentImportError("V2.10.2 source manifest is unavailable")
    raw = source.read_bytes()
    try:
        value = _strict_json(raw, name="V2.10.2 source manifest")
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if expected_file_sha256 is not None and _sha256(raw) != expected_file_sha256:
        raise PilotV2102ParentImportError("V2.10.2 source manifest file hash drifted")
    candidate = _validate_source_manifest_structure(value)
    if (
        expected_content_sha256 is not None
        and candidate["integrity"]["content_sha256"] != expected_content_sha256
    ):
        raise PilotV2102ParentImportError(
            "V2.10.2 source manifest content hash drifted"
        )
    canonical = (
        json.dumps(
            candidate,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise PilotV2102ParentImportError(
            "V2.10.2 source manifest is not canonical pretty JSON"
        )
    return candidate


def validate_v2102_source_manifest(
    source_manifest: Mapping[str, Any],
    *,
    parent_repo_root: str | Path,
    evidence_repo_root: str | Path,
    target_contract: PilotContract,
) -> dict[str, Any]:
    selected = _validate_source_manifest_structure(source_manifest)
    rebuilt = build_v2102_source_manifest(
        parent_repo_root=parent_repo_root,
        evidence_repo_root=evidence_repo_root,
        target_contract=target_contract,
    )
    if selected != rebuilt:
        raise PilotV2102ParentImportError(
            "V2.10.2 source manifest differs from immutable authorities"
        )
    return selected


def imported_v29_raw_root_v2102(child_raw_root: str | Path) -> Path:
    return Path(child_raw_root).joinpath(*V2102_SNAPSHOT_RELATIVE.parts)


def snapshot_path_for_v29_source_artifact_v2102(
    child_raw_root: str | Path,
    source_artifact_path: str,
) -> Path:
    try:
        relative = _normalized_relative(
            source_artifact_path,
            required_top="experiment_results",
            name="V2.9 source artifact path",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if tuple(relative.parts[: len(V29_RAW_ROOT.parts)]) != V29_RAW_ROOT.parts or len(
        relative.parts
    ) <= len(V29_RAW_ROOT.parts):
        raise PilotV2102ParentImportError(
            "source artifact is outside the V2.9 raw namespace"
        )
    inside = PurePosixPath(*relative.parts[len(V29_RAW_ROOT.parts) :])
    return imported_v29_raw_root_v2102(child_raw_root).joinpath(*inside.parts)


def v2102_observed_p95_receipt_path(
    raw_root: str | Path,
    profile_id: str,
) -> Path:
    if profile_id not in V2102_ALLOWED_P95_PROFILES:
        raise PilotV2102ParentImportError(
            f"unsupported V2.10.2 p95 profile: {profile_id}"
        )
    return (
        Path(raw_root)
        / "parent-import"
        / "observed_p95"
        / profile_id
        / "observed_p95_authority_receipt.json"
    )


def v2102_observed_p95_projection_path(
    raw_root: str | Path,
    profile_id: str,
) -> Path:
    return v2102_observed_p95_receipt_path(raw_root, profile_id).with_name(
        "projection_p95.json"
    )


def v2_9_p95_source_binding_v2102(
    *,
    child_raw_root: str | Path,
    profile_id: str,
) -> dict[str, Any]:
    snapshot = imported_v29_raw_root_v2102(child_raw_root)
    try:
        _, summary = _inventory(snapshot, declared_root=V29_RAW_ROOT)
    except Exception as exc:
        raise _translate(exc) from exc
    if summary != {
        "root": V29_RAW_ROOT.as_posix(),
        "inventory_schema_version": "finevo-raw-tree-inventory-v1",
        "inventory_canonicalization": "json-sort-keys-compact-utf8-v1",
        "file_count": V29_RAW_FILE_COUNT,
        "storage_bytes": V29_RAW_STORAGE_BYTES,
        "inventory_sha256": V29_RAW_INVENTORY_SHA256,
    }:
        raise PilotV2102ParentImportError("copied V2.9 raw-tree inventory drifted")
    try:
        nested = verify_v29_imported_v28_observed_p95(
            snapshot,
            profile_id,
            expected_parent_commit=V28_SCIENCE_COMMIT,
        )
        normalized = _v210.normalize_v29_observed_p95_binding(
            nested,
            profile_id=profile_id,
        )
    except Exception as exc:
        raise _translate(exc) from exc
    return {
        "source_path_kind": ("byte-exact-v2.9-raw-inside-v2.10.1-terminal-snapshot"),
        "v2_9_terminal_parent": {
            "contract_id": V29_CONTRACT_ID,
            "contract_sha256": V29_CONTRACT_CANONICAL_SHA256,
            "science_tag": V29_SCIENCE_TAG,
            "science_commit": V29_SCIENCE_COMMIT,
            "raw_file_count": V29_RAW_FILE_COUNT,
            "raw_storage_bytes": V29_RAW_STORAGE_BYTES,
            "raw_inventory_sha256": V29_RAW_INVENTORY_SHA256,
            "terminal_status": "complete-with-no-go",
            "implementation_root_cause": ("imported-p95-runner-binding-shape-mismatch"),
        },
        "v2_8_observed_p95_origin": _json_copy(nested),
        "normalized_v2_9_binding": normalized,
    }


def _contract_binding(
    repo_root: Path,
    contract: PilotContract,
) -> dict[str, Any]:
    _, raw, value = _strict_file(
        repo_root,
        V2102_EXPANDED_CONTRACT_PATH,
        name="expanded V2.10.2 contract",
    )
    if value != contract.to_dict():
        raise PilotV2102ParentImportError(
            "expanded V2.10.2 contract differs from selected contract"
        )
    return {
        "path": V2102_EXPANDED_CONTRACT_PATH.as_posix(),
        "file_sha256": _sha256(raw),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
    }


def build_v2102_resealed_observed_p95_authority(
    *,
    repo_root: str | Path,
    contract: PilotContract,
    raw_root: str | Path,
    profile_id: str,
    expected_git_commit: str,
    verified_v2_9_source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Reseal exact V2.9 values; reject every V2.10.1 wrapper shape."""

    root = _strict_root(repo_root, name="V2.10.2 child repository")
    _validate_target_contract(contract, require_frozen=True)
    if (
        _COMMIT_RE.fullmatch(expected_git_commit) is None
        or profile_id not in V2102_ALLOWED_P95_PROFILES
    ):
        raise PilotV2102ParentImportError(
            "V2.10.2 p95 release commit/profile is invalid"
        )
    raw_path = Path(raw_root)
    if not raw_path.is_absolute():
        raw_path = root.joinpath(*PurePosixPath(str(raw_root)).parts)
    raw_path = raw_path.absolute()
    if (
        _repo_relative(
            root,
            raw_path,
            required_top="experiment_results",
            name="V2.10.2 raw root",
        )
        != V2102_RAW_ROOT.as_posix()
    ):
        raise PilotV2102ParentImportError(
            "V2.10.2 p95 requires the exact raw namespace"
        )
    source = _json_copy(
        _mapping(
            verified_v2_9_source_binding,
            name="verified exact V2.9 p95 source binding",
        )
    )
    if set(source) != {
        "source_path_kind",
        "v2_9_terminal_parent",
        "v2_8_observed_p95_origin",
        "normalized_v2_9_binding",
    }:
        raise PilotV2102ParentImportError("exact V2.9 p95 source lineage shape drifted")
    if (
        source.get("source_path_kind")
        != "byte-exact-v2.9-raw-inside-v2.10.1-terminal-snapshot"
    ):
        raise PilotV2102ParentImportError(
            "V2.10.1 current wrapper cannot be V2.10.2 p95 authority"
        )
    nested = _mapping(
        source["v2_8_observed_p95_origin"],
        name="V2.8 p95 origin",
    )
    try:
        normalized = _v210.normalize_v29_observed_p95_binding(
            nested,
            profile_id=profile_id,
        )
    except _v210.PilotV210ParentImportError as exc:
        raise _translate(exc) from exc
    if source["normalized_v2_9_binding"] != normalized:
        raise PilotV2102ParentImportError("V2.9 nested/flat p95 normalization drifted")
    profile = contract.provider_profiles[profile_id]
    runtime_model = f"{profile.transport}/{profile.requested_model}"
    if (
        nested.get("runtime_model") != runtime_model
        or nested.get("served_model") != profile.served_model
    ):
        raise PilotV2102ParentImportError(
            "V2.10.2 provider profile differs from exact p95 source"
        )
    reservations = _json_copy(normalized["reservations"])
    for call_kind in ("action", "semantic"):
        entry = reservations[runtime_model][call_kind]
        try:
            parsed = PreflightP95Reservation.from_dict(
                model=runtime_model,
                call_kind=call_kind,
                value=entry["reservation"],
            )
        except (TypeError, ValueError) as exc:
            raise PilotV2102ParentImportError(
                "inherited p95 reservation is invalid"
            ) from exc
        if parsed.to_dict() != entry["reservation"]:
            raise PilotV2102ParentImportError("inherited p95 reservation drifted")
        authority = entry["authority"]
        if not isinstance(authority, dict):
            raise PilotV2102ParentImportError(
                "inherited p95 authority is not mutable JSON"
            )
        authority["pilot_contract_hash"] = contract.canonical_hash
        authority["pilot_tag"] = V2102_SCIENCE_TAG

    receipt = _seal(
        {
            "schema_version": (V2102_RESEALED_P95_AUTHORITY_SCHEMA_VERSION),
            "contract": _contract_binding(root, contract),
            "raw_root": V2102_RAW_ROOT.as_posix(),
            "git": {
                "tag": V2102_SCIENCE_TAG,
                "commit": expected_git_commit,
            },
            "model": {
                "model_id": profile_id,
                "runtime_model": runtime_model,
                "served_model": profile.served_model,
            },
            "parent_lineage": source,
            "v2_10_1_terminal_lineage": {
                "contract_sha256": V2101_PARENT_CONTRACT_CANONICAL_SHA256,
                "science_tag": V2101_PARENT_SCIENCE_TAG,
                "science_tag_object": V2101_PARENT_SCIENCE_TAG_OBJECT,
                "science_commit": V2101_PARENT_SCIENCE_COMMIT,
                "source_manifest": _v2101_source_manifest_binding(),
                "release_attestation": _v2101_release_attestation_binding(),
                "raw_inventory_sha256": V2101_PARENT_RAW_INVENTORY_SHA256,
                "run_ledger_sha256": V2101_PARENT_RUN_LEDGER_INTERNAL_SHA256,
                "budget_ledger_sha256": V2101_PARENT_BUDGET_LEDGER_INTERNAL_SHA256,
                "qref_receipt_content_sha256": V2101_PARENT_QREF_RECEIPT_CONTENT_SHA256,
                "status_counts": {
                    "complete": 26,
                    "failed": 185,
                },
                "implementation_root_cause": (
                    "observed-p95-consumer-schema-dispatch-gap"
                ),
                "offline_candidate_admission_cells_observed": 10,
                "actor_performance_treatment_outcome_blind": True,
                "incremental_provider_calls": 0,
                "incremental_hosted_cost_usd": 0.0,
            },
            "reservations": reservations,
            "provider_boundary": {
                "provider_construction_during_reseal": False,
                "provider_calls_during_reseal": 0,
                "hosted_provider_calls_during_reseal": 0,
                "hosted_cost_usd_during_reseal": 0.0,
            },
            "scientific_evidence": False,
            "evidence_use": (
                "V2.10.2 prospective budget authority only; V2.9/V2.8 "
                "parent rows and V2.10.1 no-go rows contribute no V2.10.2 "
                "A-D treatment effect."
            ),
        }
    )
    receipt_path = v2102_observed_p95_receipt_path(raw_path, profile_id)
    receipt_relative = _repo_relative(
        root,
        receipt_path,
        required_top="experiment_results",
        name="V2.10.2 p95 receipt",
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
                "git_tag": V2102_SCIENCE_TAG,
                "git_commit": expected_git_commit,
                "source_kind": V2102_RESEALED_P95_SOURCE_KIND,
                "source_authority_receipt": receipt_relative,
                "source_authority_receipt_content_sha256": (
                    receipt["integrity"]["content_sha256"]
                ),
                "source_v2_10_1_terminal_raw_inventory_sha256": (
                    V2101_PARENT_RAW_INVENTORY_SHA256
                ),
                "source_v2_9_raw_inventory_sha256": (V29_RAW_INVENTORY_SHA256),
                "source_v2_8_authority_content_sha256": (
                    nested["authority"]["content_sha256"]
                ),
                "source_v2_8_projection_content_sha256": (
                    nested["projection"]["content_sha256"]
                ),
            },
        }
    )
    return {
        "receipt_path": receipt_path,
        "projection_path": v2102_observed_p95_projection_path(
            raw_path,
            profile_id,
        ),
        "receipt": receipt,
        "projection": projection_value,
    }


def _atomic_json(
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
        _atomic_exact_bytes_no_follow(
            repo_root=repo_root,
            path=path,
            raw=raw,
        )
    except PilotV27Stage0ImportError as exc:
        raise _translate(exc) from exc


def persist_v2102_resealed_observed_p95_authority(
    *,
    repo_root: str | Path,
    contract: PilotContract,
    raw_root: str | Path,
    profile_id: str,
    expected_git_commit: str,
    verified_v2_9_source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    built = build_v2102_resealed_observed_p95_authority(
        repo_root=repo_root,
        contract=contract,
        raw_root=raw_root,
        profile_id=profile_id,
        expected_git_commit=expected_git_commit,
        verified_v2_9_source_binding=verified_v2_9_source_binding,
    )
    root = _strict_root(repo_root, name="V2.10.2 child repository")
    _atomic_json(
        repo_root=root,
        path=built["receipt_path"],
        value=built["receipt"],
    )
    _atomic_json(
        repo_root=root,
        path=built["projection_path"],
        value=built["projection"],
    )
    return {
        "receipt": str(built["receipt_path"]),
        "projection": str(built["projection_path"]),
        "receipt_content_sha256": (built["receipt"]["integrity"]["content_sha256"]),
        "projection_content_sha256": (
            built["projection"]["integrity"]["content_sha256"]
        ),
        "provider_construction_during_reseal": False,
        "provider_calls_during_reseal": 0,
    }


def _load_current_contract(
    repo_root: Path,
    selected: PilotContract | None,
) -> PilotContract:
    contract = selected or load_pilot_contract(
        repo_root.joinpath(*V2102_EXPANDED_CONTRACT_PATH.parts)
    )
    _validate_target_contract(contract, require_frozen=True)
    _contract_binding(repo_root, contract)
    return contract


def _rebuild_v2102_p95(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
    raw_root: Path,
    contract: PilotContract,
    expected_git_commit: str,
) -> dict[str, Any]:
    value = _json_copy(receipt)
    expected_keys = {
        "schema_version",
        "contract",
        "raw_root",
        "git",
        "model",
        "parent_lineage",
        "v2_10_1_terminal_lineage",
        "reservations",
        "provider_boundary",
        "scientific_evidence",
        "evidence_use",
        "integrity",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != V2102_RESEALED_P95_AUTHORITY_SCHEMA_VERSION
    ):
        raise PilotV2102ParentImportError("V2.10.2 p95 receipt shape/schema drifted")
    _verify_self_hashed(
        value,
        schema_version=V2102_RESEALED_P95_AUTHORITY_SCHEMA_VERSION,
        name="V2.10.2 p95 receipt",
    )
    model = _mapping(value.get("model"), name="V2.10.2 p95 model")
    profile_id = str(model.get("model_id"))
    source = v2_9_p95_source_binding_v2102(
        child_raw_root=raw_root,
        profile_id=profile_id,
    )
    rebuilt = build_v2102_resealed_observed_p95_authority(
        repo_root=repo_root,
        contract=contract,
        raw_root=raw_root,
        profile_id=profile_id,
        expected_git_commit=expected_git_commit,
        verified_v2_9_source_binding=source,
    )
    if value != rebuilt["receipt"]:
        raise PilotV2102ParentImportError(
            "V2.10.2 p95 receipt differs from exact source/release authority"
        )
    return rebuilt


def verify_v2102_resealed_observed_p95_authority(
    receipt: Mapping[str, Any],
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    expected_git_commit: str,
    contract: PilotContract | None = None,
) -> dict[str, Any]:
    root = _strict_root(repo_root, name="V2.10.2 child repository")
    _verify_release_identity(
        root,
        tag=V2102_SCIENCE_TAG,
        expected_commit=expected_git_commit,
        expected_tag_object=None,
        name="V2.10.2 child",
    )
    selected = _load_current_contract(root, contract)
    built = _rebuild_v2102_p95(
        receipt,
        repo_root=root,
        raw_root=Path(raw_root).absolute(),
        contract=selected,
        expected_git_commit=expected_git_commit,
    )
    return _json_copy(built["receipt"]["reservations"])


def verify_v2102_resealed_observed_p95_projection(
    projection: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    repo_root: str | Path,
    raw_root: str | Path,
    expected_git_commit: str,
    contract: PilotContract | None = None,
) -> dict[str, Any]:
    root = _strict_root(repo_root, name="V2.10.2 child repository")
    _verify_release_identity(
        root,
        tag=V2102_SCIENCE_TAG,
        expected_commit=expected_git_commit,
        expected_tag_object=None,
        name="V2.10.2 child",
    )
    selected = _load_current_contract(root, contract)
    built = _rebuild_v2102_p95(
        receipt,
        repo_root=root,
        raw_root=Path(raw_root).absolute(),
        contract=selected,
        expected_git_commit=expected_git_commit,
    )
    candidate = _json_copy(projection)
    _verify_self_hashed(
        candidate,
        schema_version="finevo-pilot-projection-p95-v1",
        name="V2.10.2 p95 projection",
    )
    if candidate != built["projection"]:
        raise PilotV2102ParentImportError(
            "V2.10.2 p95 projection differs from its receipt/source"
        )
    return candidate


def verified_v2102_observed_p95_authority_binding(
    receipt_path: str | Path,
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    expected_git_commit: str,
    contract: PilotContract | None = None,
) -> dict[str, Any]:
    root = _strict_root(repo_root, name="V2.10.2 child repository")
    path = Path(receipt_path)
    if path.is_absolute():
        try:
            relative = PurePosixPath(*path.absolute().relative_to(root).parts)
        except ValueError as exc:
            raise PilotV2102ParentImportError(
                "V2.10.2 p95 receipt escaped the repository"
            ) from exc
    else:
        try:
            relative = _normalized_relative(
                path,
                required_top="experiment_results",
                name="V2.10.2 p95 receipt path",
            )
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
    _, raw, receipt = _strict_file(
        root,
        relative,
        name="V2.10.2 p95 receipt",
    )
    reservations = verify_v2102_resealed_observed_p95_authority(
        receipt,
        repo_root=root,
        raw_root=raw_root,
        expected_git_commit=expected_git_commit,
        contract=contract,
    )
    projection_relative = relative.with_name("projection_p95.json")
    _, _, projection = _strict_file(
        root,
        projection_relative,
        name="V2.10.2 p95 projection",
    )
    verify_v2102_resealed_observed_p95_projection(
        projection,
        receipt=receipt,
        repo_root=root,
        raw_root=raw_root,
        expected_git_commit=expected_git_commit,
        contract=contract,
    )
    return {
        "receipt_path": relative.as_posix(),
        "receipt_file_sha256": _sha256(raw),
        "receipt_content_sha256": receipt["integrity"]["content_sha256"],
        "git_commit": expected_git_commit,
        "reservations": reservations,
    }


def verified_v2102_observed_p95_projection_binding(
    projection_path: str | Path,
    *,
    receipt_path: str | Path,
    repo_root: str | Path,
    raw_root: str | Path,
    expected_git_commit: str,
    contract: PilotContract | None = None,
) -> dict[str, Any]:
    root = _strict_root(repo_root, name="V2.10.2 child repository")
    authority = verified_v2102_observed_p95_authority_binding(
        receipt_path,
        repo_root=root,
        raw_root=raw_root,
        expected_git_commit=expected_git_commit,
        contract=contract,
    )
    path = Path(projection_path)
    if path.is_absolute():
        try:
            relative = PurePosixPath(*path.absolute().relative_to(root).parts)
        except ValueError as exc:
            raise PilotV2102ParentImportError(
                "V2.10.2 p95 projection escaped the repository"
            ) from exc
    else:
        try:
            relative = _normalized_relative(
                path,
                required_top="experiment_results",
                name="V2.10.2 p95 projection path",
            )
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
    _, raw, projection = _strict_file(
        root,
        relative,
        name="V2.10.2 p95 projection",
    )
    receipt_relative = PurePosixPath(authority["receipt_path"])
    _, _, receipt = _strict_file(
        root,
        receipt_relative,
        name="V2.10.2 p95 receipt",
    )
    payload = verify_v2102_resealed_observed_p95_projection(
        projection,
        receipt=receipt,
        repo_root=root,
        raw_root=raw_root,
        expected_git_commit=expected_git_commit,
        contract=contract,
    )
    model = _mapping(receipt.get("model"), name="V2.10.2 p95 model")
    return {
        "projection_path": relative.as_posix(),
        "projection_file_sha256": _sha256(raw),
        "projection_content_sha256": payload["integrity"]["content_sha256"],
        "profile_id": model["model_id"],
        "served_model": model["served_model"],
        "runtime_model": model["runtime_model"],
        "reservations": _json_copy(authority["reservations"]),
        "source_contract_id": V29_CONTRACT_ID,
        "source_contract_sha256": V29_CONTRACT_CANONICAL_SHA256,
        "source_git_tag": V29_SCIENCE_TAG,
        "source_git_commit": V29_SCIENCE_COMMIT,
        "git_commit": expected_git_commit,
        "payload": payload,
    }


def source_binding_for_target_v2102(
    source_manifest: Mapping[str, Any],
    target_spec: PilotRunSpec | Mapping[str, Any] | str,
) -> dict[str, Any]:
    rows = source_manifest.get("imported_complete_cells")
    if not isinstance(rows, list) or len(rows) != 16:
        raise PilotV2102ParentImportError(
            "V2.10.2 source manifest lacks 16 imported prerequisites"
        )
    target_run_id = (
        target_spec
        if isinstance(target_spec, str)
        else (
            target_spec.run_id
            if isinstance(target_spec, PilotRunSpec)
            else target_spec.get("run_id")
        )
    )
    selected = [row for row in rows if row.get("target_run_id") == target_run_id]
    if len(selected) != 1:
        raise PilotV2102ParentImportError(
            "target has no unique imported V2.9 prerequisite binding"
        )
    result = _json_copy(selected[0])
    if not isinstance(target_spec, str):
        expected = (
            target_spec.to_dict()
            if isinstance(target_spec, PilotRunSpec)
            else _json_copy(target_spec)
        )
        if result.get("target_spec") != expected:
            raise PilotV2102ParentImportError(
                "target spec differs from source-manifest binding"
            )
    return result


def imported_prerequisite_path_v2102(
    child_raw_root: str | Path,
    source_manifest: Mapping[str, Any],
    target_spec: PilotRunSpec | Mapping[str, Any] | str,
    artifact_key: str,
) -> Path:
    binding = source_binding_for_target_v2102(source_manifest, target_spec)
    artifacts = _mapping(
        binding.get("source_artifacts"),
        name="V2.10.2 imported prerequisite artifacts",
    )
    artifact = artifacts.get(artifact_key)
    if isinstance(artifact, Mapping):
        source_path = artifact.get("path")
    elif artifact_key == "run_root" and isinstance(artifact, str):
        source_path = artifact
    else:
        raise PilotV2102ParentImportError(
            f"imported prerequisite lacks artifact {artifact_key!r}"
        )
    if not isinstance(source_path, str):
        raise PilotV2102ParentImportError(
            f"imported prerequisite artifact {artifact_key!r} has no path"
        )
    return snapshot_path_for_v29_source_artifact_v2102(
        child_raw_root,
        source_path,
    )


def _verify_bound_source_artifact(
    value: Mapping[str, Any],
    *,
    name: str,
) -> None:
    schema = str(value.get("schema_version", ""))
    if schema == "finevo-pilot-terminal-summary-v1":
        try:
            _v210._verify_terminal_summary_hash(value, name=name)
        except _v210.PilotV210ParentImportError as exc:
            raise _translate(exc) from exc
    elif schema == "finevo-pilot-stage-receipt-v2":
        _verify_stage_receipt_v2(value, name=name)
    else:
        _verify_self_hashed(value, schema_version=schema, name=name)


def verified_v2102_imported_prerequisite_binding(
    child_raw_root: str | Path,
    source_manifest: Mapping[str, Any],
    target_spec: PilotRunSpec | Mapping[str, Any] | str,
) -> dict[str, Any]:
    """Verify one child-local V2.9 prerequisite without relabelling it."""

    binding = source_binding_for_target_v2102(source_manifest, target_spec)
    snapshot = imported_v29_raw_root_v2102(child_raw_root)
    try:
        _, summary = _inventory(snapshot, declared_root=V29_RAW_ROOT)
    except Exception as exc:
        raise _translate(exc) from exc
    if summary != {
        "root": V29_RAW_ROOT.as_posix(),
        "inventory_schema_version": "finevo-raw-tree-inventory-v1",
        "inventory_canonicalization": "json-sort-keys-compact-utf8-v1",
        "file_count": V29_RAW_FILE_COUNT,
        "storage_bytes": V29_RAW_STORAGE_BYTES,
        "inventory_sha256": V29_RAW_INVENTORY_SHA256,
    }:
        raise PilotV2102ParentImportError("copied V2.9 prerequisite snapshot drifted")
    verified: dict[str, Any] = {}
    artifacts = _mapping(
        binding.get("source_artifacts"),
        name="V2.10.2 imported prerequisite artifacts",
    )
    for key, declared in artifacts.items():
        if key == "run_root":
            if not isinstance(declared, str):
                raise PilotV2102ParentImportError(
                    "imported prerequisite run_root is malformed"
                )
            path = snapshot_path_for_v29_source_artifact_v2102(
                child_raw_root,
                declared,
            )
            if path.is_symlink() or not path.is_dir():
                raise PilotV2102ParentImportError(
                    "imported prerequisite run_root is unavailable"
                )
            verified[key] = str(path)
            continue
        declared_map = _mapping(
            declared,
            name=f"imported prerequisite {key}",
        )
        source_path = declared_map.get("path")
        if not isinstance(source_path, str):
            raise PilotV2102ParentImportError(
                f"imported prerequisite {key} lacks a source path"
            )
        path = snapshot_path_for_v29_source_artifact_v2102(
            child_raw_root,
            source_path,
        )
        if path.is_symlink() or not path.is_file():
            raise PilotV2102ParentImportError(
                f"imported prerequisite {key} is unavailable"
            )
        raw = path.read_bytes()
        if _sha256(raw) != declared_map.get("file_sha256") or len(
            raw
        ) != declared_map.get("byte_size", len(raw)):
            raise PilotV2102ParentImportError(
                f"imported prerequisite {key} file binding drifted"
            )
        row = {
            **_json_copy(declared_map),
            "snapshot_path": str(path),
        }
        if "content_sha256" in declared_map:
            try:
                value = _strict_json(
                    raw,
                    name=f"imported prerequisite {key}",
                )
            except PilotV24ParentImportError as exc:
                raise _translate(exc) from exc
            _verify_bound_source_artifact(
                value,
                name=f"imported prerequisite {key}",
            )
            if (
                value.get("integrity", {}).get("content_sha256")
                != declared_map["content_sha256"]
            ):
                raise PilotV2102ParentImportError(
                    f"imported prerequisite {key} content binding drifted"
                )
        verified[key] = row
    terminal = _mapping(
        verified.get("summary"),
        name="imported prerequisite terminal summary",
    )
    result: dict[str, Any] = {
        "source_run_id": binding["source_run_id"],
        "source_stage_id": binding["stage_id"],
        "source_spec": _json_copy(binding["source_spec"]),
        "target_spec": _json_copy(binding["target_spec"]),
        "source_terminal": _json_copy(terminal),
        "source_artifacts": verified,
        "source_release": {
            "contract_id": V29_CONTRACT_ID,
            "contract_sha256": V29_CONTRACT_CANONICAL_SHA256,
            "science_tag": V29_SCIENCE_TAG,
            "science_commit": V29_SCIENCE_COMMIT,
            "snapshot_root": str(snapshot),
            "raw_inventory_sha256": V29_RAW_INVENTORY_SHA256,
            "source_path_kind": ("byte-exact-v2.9-raw-inside-v2.10.1-terminal-snapshot"),
        },
        "provider_construction_during_verification": False,
        "provider_calls_during_verification": 0,
        "treatment_effect_evidence": False,
    }
    if binding["stage_id"] == "q-ref-resolution":
        qref_path = Path(verified["q_ref_resolution"]["snapshot_path"])
        try:
            qref = _strict_json(
                qref_path.read_bytes(),
                name="imported V2.9 q-ref resolution",
            )
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
        if (
            qref.get("schema_version") != "finevo-q-ref-resolution-v1"
            or qref.get("status") != "pass"
            or qref.get("q_ref") != 63.50397933257746
            or qref.get("row_count") != 48
        ):
            raise PilotV2102ParentImportError("imported V2.9 q-ref semantics drifted")
        result["q_ref"] = 63.50397933257746
        result["q_ref_resolution"] = _json_copy(qref)
    if binding["stage_id"] == "stage0-calibration":
        selection_path = Path(verified["stage0_selection"]["snapshot_path"])
        try:
            selection = _strict_json(
                selection_path.read_bytes(),
                name="imported V2.9 Stage-0 selection",
            )
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
        if (
            selection.get("schema_version") != "finevo-stage0-selection-v1"
            or selection.get("contract_sha256") != V29_CONTRACT_CANONICAL_SHA256
            or selection.get("selected_profile_id") != "nu-0.5"
        ):
            raise PilotV2102ParentImportError("imported V2.9 Stage-0 winner drifted")
        terminal_path = Path(terminal["snapshot_path"])
        try:
            terminal_value = _strict_json(
                terminal_path.read_bytes(),
                name="imported V2.9 Stage-0 terminal",
            )
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
        result["metrics"] = _json_copy(
            _mapping(
                terminal_value.get("payload"),
                name="imported Stage-0 payload",
            ).get("metrics")
        )
        result["selected_profile_id"] = "nu-0.5"
    return result


def v2102_imported_v29_run_dir(
    child_raw_root: str | Path,
    source_manifest: Mapping[str, Any],
    target_spec: PilotRunSpec | Mapping[str, Any] | str,
) -> Path:
    return imported_prerequisite_path_v2102(
        child_raw_root,
        source_manifest,
        target_spec,
        "run_root",
    )


def _tracked_source_manifest_binding(
    child_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, raw, value = _strict_file(
        child_root,
        V2102_SOURCE_MANIFEST_PATH,
        name="tracked V2.10.2 source manifest",
    )
    manifest = _validate_source_manifest_structure(value)
    canonical = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise PilotV2102ParentImportError(
            "tracked V2.10.2 source manifest is not canonical JSON"
        )
    return manifest, {
        "path": V2102_SOURCE_MANIFEST_PATH.as_posix(),
        "schema_version": V2102_SOURCE_MANIFEST_SCHEMA_VERSION,
        "file_sha256": _sha256(raw),
        "content_sha256": manifest["integrity"]["content_sha256"],
    }


def _build_v2102_parent_import_receipt(
    *,
    child_root: Path,
    child_raw: Path,
    contract: PilotContract,
    child_git_commit: str,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _validate_source_manifest_structure(source_manifest)
    try:
        _, snapshot = _inventory(
            imported_v29_raw_root_v2102(child_raw),
            declared_root=V29_RAW_ROOT,
        )
    except Exception as exc:
        raise _translate(exc) from exc
    if snapshot != {
        "root": V29_RAW_ROOT.as_posix(),
        "inventory_schema_version": "finevo-raw-tree-inventory-v1",
        "inventory_canonicalization": "json-sort-keys-compact-utf8-v1",
        "file_count": V29_RAW_FILE_COUNT,
        "storage_bytes": V29_RAW_STORAGE_BYTES,
        "inventory_sha256": V29_RAW_INVENTORY_SHA256,
    }:
        raise PilotV2102ParentImportError("V2.10.2 parent snapshot inventory drifted")
    p95: dict[str, Any] = {}
    for profile_id in V2102_ALLOWED_P95_PROFILES:
        receipt_path = v2102_observed_p95_receipt_path(
            child_raw,
            profile_id,
        )
        projection_path = v2102_observed_p95_projection_path(
            child_raw,
            profile_id,
        )
        p95[profile_id] = {
            "authority": verified_v2102_observed_p95_authority_binding(
                receipt_path,
                repo_root=child_root,
                raw_root=child_raw,
                expected_git_commit=child_git_commit,
                contract=contract,
            ),
            "projection": verified_v2102_observed_p95_projection_binding(
                projection_path,
                receipt_path=receipt_path,
                repo_root=child_root,
                raw_root=child_raw,
                expected_git_commit=child_git_commit,
                contract=contract,
            ),
        }
    tracked_manifest, manifest_binding = _tracked_source_manifest_binding(child_root)
    if tracked_manifest != manifest:
        raise PilotV2102ParentImportError(
            "selected V2.10.2 source manifest differs from tracked authority"
        )
    return _seal(
        {
            "schema_version": V2102_PARENT_IMPORT_SCHEMA_VERSION,
            "contract": _contract_binding(child_root, contract),
            "git": {
                "tag": V2102_SCIENCE_TAG,
                "commit": child_git_commit,
            },
            "source_manifest": manifest_binding,
            "terminal_parent": {
                "contract_id": V2101_PARENT_CONTRACT_ID,
                "contract_sha256": V2101_PARENT_CONTRACT_CANONICAL_SHA256,
                "science_tag": V2101_PARENT_SCIENCE_TAG,
                "science_tag_object": V2101_PARENT_SCIENCE_TAG_OBJECT,
                "science_commit": V2101_PARENT_SCIENCE_COMMIT,
                "source_manifest": _json_copy(
                    manifest["v2_10_1_terminal_parent"]["source_manifest"]
                ),
                "release_attestation": _json_copy(
                    manifest["v2_10_1_terminal_parent"]["release"][
                        "release_attestation"
                    ]
                ),
                "raw_inventory_sha256": V2101_PARENT_RAW_INVENTORY_SHA256,
                "run_ledger_sha256": V2101_PARENT_RUN_LEDGER_INTERNAL_SHA256,
                "budget_ledger_sha256": V2101_PARENT_BUDGET_LEDGER_INTERNAL_SHA256,
                "qref_receipt_content_sha256": V2101_PARENT_QREF_RECEIPT_CONTENT_SHA256,
                "terminal_status": "complete-with-no-go",
                "status_counts": {
                    "complete": 26,
                    "failed": 185,
                },
                "implementation_root_cause": (
                    "observed-p95-consumer-schema-dispatch-gap"
                ),
                "offline_candidate_admission_cells_observed": 10,
                "actor_performance_treatment_outcome_blind": True,
                "incremental_provider_calls": 0,
                "incremental_hosted_cost_usd": 0.0,
            },
            "source_parent": {
                "contract_id": V29_CONTRACT_ID,
                "contract_sha256": V29_CONTRACT_CANONICAL_SHA256,
                "science_tag": V29_SCIENCE_TAG,
                "science_commit": V29_SCIENCE_COMMIT,
                "raw_root": V29_RAW_ROOT.as_posix(),
                "source_path_kind": (
                    "byte-exact-v2.9-raw-inside-v2.10.1-terminal-snapshot"
                ),
                "terminal_status": "complete-with-no-go",
                "scientific_complete": False,
            },
            "copied_snapshot": {
                "path": (V2102_RAW_ROOT / V2102_SNAPSHOT_RELATIVE).as_posix(),
                **snapshot,
            },
            "imported_cells": {
                "count": 16,
                "breakdown": {
                    "parent-import": 1,
                    "q-ref-resolution": 1,
                    "stage0-calibration": 14,
                },
                "bindings_content_sha256": canonical_sha256(
                    manifest["imported_complete_cells"]
                ),
                "q_ref": 63.50397933257746,
                "stage0_selected_profile_id": "nu-0.5",
                "offline_candidate_admission_cells_imported": 0,
                "failed_actor_cells_imported": 0,
            },
            "p95_current_release_authorities": p95,
            "cumulative_budget_debit": V2102_CUMULATIVE_DEBIT.to_dict(),
            "provider_construction_during_import": False,
            "provider_calls_during_import": 0,
            "hosted_provider_calls_during_import": 0,
            "hosted_cost_usd_during_import": 0.0,
            "scientific_evidence": False,
            "evidence_use": (
                "immutable V2.9 prerequisite provenance and prospective "
                "V2.10.2 p95 authority only; no imported A-D effect."
            ),
        }
    )


def persist_v2102_parent_import(
    *,
    parent_repo_root: str | Path,
    evidence_repo_root: str | Path,
    child_repo_root: str | Path,
    child_raw_root: str | Path,
    contract: PilotContract,
    child_git_commit: str,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy exact V2.9 bytes and reseal without provider construction."""

    parent = _strict_root(parent_repo_root, name="V2.10.1 source repository")
    child = _strict_root(child_repo_root, name="V2.10.2 child repository")
    child_raw = Path(child_raw_root).absolute()
    if child_raw != child.joinpath(*V2102_RAW_ROOT.parts):
        raise PilotV2102ParentImportError(
            "V2.10.2 child raw root differs from contract namespace"
        )
    _validate_target_contract(contract, require_frozen=True)
    _verify_release_identity(
        child,
        tag=V2102_SCIENCE_TAG,
        expected_commit=child_git_commit,
        expected_tag_object=None,
        name="V2.10.2 child",
    )
    selected = validate_v2102_source_manifest(
        source_manifest,
        parent_repo_root=parent,
        evidence_repo_root=evidence_repo_root,
        target_contract=contract,
    )
    tracked, _ = _tracked_source_manifest_binding(child)
    if tracked != selected:
        raise PilotV2102ParentImportError(
            "tracked V2.10.2 source manifest differs before import"
        )
    terminal = verify_v2101_terminal_lineage(
        parent_repo_root=parent,
        evidence_repo_root=evidence_repo_root,
    )
    audit = verify_v29_exact_source_for_v2102(
        parent_repo_root=parent,
        evidence_repo_root=evidence_repo_root,
        target_contract=contract,
        terminal_audit=terminal,
    )
    snapshot = imported_v29_raw_root_v2102(child_raw)
    try:
        _v210._copy_exact_v29_snapshot(
            source_root=Path(audit["source_raw_root"]),
            destination_root=snapshot,
            child_repo_root=child,
            inventory=audit["raw_inventory"]["rows"],
        )
    except _v210.PilotV210ParentImportError as exc:
        raise _translate(exc) from exc
    for profile_id in V2102_ALLOWED_P95_PROFILES:
        persist_v2102_resealed_observed_p95_authority(
            repo_root=child,
            contract=contract,
            raw_root=child_raw,
            profile_id=profile_id,
            expected_git_commit=child_git_commit,
            verified_v2_9_source_binding=audit["p95_sources"][profile_id],
        )
    receipt = _build_v2102_parent_import_receipt(
        child_root=child,
        child_raw=child_raw,
        contract=contract,
        child_git_commit=child_git_commit,
        source_manifest=selected,
    )
    receipt_path = child_raw / "parent-import" / "parent_import_receipt.json"
    _atomic_json(repo_root=child, path=receipt_path, value=receipt)
    raw = receipt_path.read_bytes()
    return {
        "receipt": str(receipt_path),
        "receipt_file_sha256": _sha256(raw),
        "receipt_content_sha256": receipt["integrity"]["content_sha256"],
        "snapshot_root": str(snapshot),
        "snapshot_inventory_sha256": V29_RAW_INVENTORY_SHA256,
        "imported_cell_count": 16,
        "imported_q_ref_cell_count": 1,
        "imported_stage0_cell_count": 14,
        "stage0_selected_profile_id": "nu-0.5",
        "offline_candidate_admission_cells_imported": 0,
        "provider_construction_during_import": False,
        "provider_calls_during_import": 0,
        "hosted_provider_calls_during_import": 0,
        "hosted_cost_usd_during_import": 0.0,
        "scientific_evidence": False,
        "v2_10_1_terminal_no_go_preserved": True,
        "v2_9_source_bytes_preserved": True,
    }


def verify_v2102_parent_import_receipt(
    *,
    receipt_path: str | Path,
    child_repo_root: str | Path,
    contract: PilotContract,
    expected_git_commit: str,
) -> dict[str, Any]:
    child = _strict_root(child_repo_root, name="V2.10.2 child repository")
    _validate_target_contract(contract, require_frozen=True)
    _verify_release_identity(
        child,
        tag=V2102_SCIENCE_TAG,
        expected_commit=expected_git_commit,
        expected_tag_object=None,
        name="V2.10.2 child",
    )
    path = Path(receipt_path).absolute()
    expected = child.joinpath(
        *V2102_RAW_ROOT.parts,
        "parent-import",
        "parent_import_receipt.json",
    )
    if path != expected:
        raise PilotV2102ParentImportError(
            "V2.10.2 parent receipt path differs from contract namespace"
        )
    relative = V2102_RAW_ROOT / "parent-import/parent_import_receipt.json"
    _, _, value = _strict_file(
        child,
        relative,
        name="V2.10.2 parent import receipt",
    )
    _verify_self_hashed(
        value,
        schema_version=V2102_PARENT_IMPORT_SCHEMA_VERSION,
        name="V2.10.2 parent import receipt",
    )
    manifest, _ = _tracked_source_manifest_binding(child)
    rebuilt = _build_v2102_parent_import_receipt(
        child_root=child,
        child_raw=child.joinpath(*V2102_RAW_ROOT.parts),
        contract=contract,
        child_git_commit=expected_git_commit,
        source_manifest=manifest,
    )
    if value != rebuilt:
        raise PilotV2102ParentImportError(
            "V2.10.2 parent receipt differs from copied/resealed authority"
        )
    return value


def parent_budget_debit_for_v2102(
    contract: PilotContract,
) -> ParentBudgetDebit | None:
    if getattr(contract, "contract_id", None) != V2102_CONTRACT_ID:
        return None
    _validate_target_contract(contract, require_frozen=False)
    return V2102_CUMULATIVE_DEBIT


__all__ = [
    "PilotV2102ParentImportError",
    "V2102_ALLOWED_P95_PROFILES",
    "V2102_CONTRACT_ID",
    "V2102_CUMULATIVE_DEBIT",
    "V2102_EXPANDED_CONTRACT_PATH",
    "V2102_PARENT_IMPORT_SCHEMA_VERSION",
    "V2102_RAW_ROOT",
    "V2102_RESEALED_P95_AUTHORITY_SCHEMA_VERSION",
    "V2102_RESEALED_P95_SOURCE_KIND",
    "V2102_SCIENCE_TAG",
    "V2102_SNAPSHOT_RELATIVE",
    "V2102_SOURCE_MANIFEST_PATH",
    "V2102_SOURCE_MANIFEST_SCHEMA_VERSION",
    "V2101_PARENT_BUDGET_LEDGER_FILE_SHA256",
    "V2101_PARENT_BUDGET_LEDGER_INTERNAL_SHA256",
    "V2101_PARENT_CONTRACT_CANONICAL_SHA256",
    "V2101_PARENT_QREF_RECEIPT_CONTENT_SHA256",
    "V2101_PARENT_QREF_RECEIPT_FILE_SHA256",
    "V2101_PARENT_RAW_FILE_COUNT",
    "V2101_PARENT_RAW_INVENTORY_SHA256",
    "V2101_PARENT_RAW_STORAGE_BYTES",
    "V2101_PARENT_RELEASE_ATTESTATION_CONTENT_SHA256",
    "V2101_PARENT_RELEASE_ATTESTATION_FILE_SHA256",
    "V2101_PARENT_RELEASE_ATTESTATION_PATH",
    "V2101_PARENT_RELEASE_ATTESTATION_SCHEMA_VERSION",
    "V2101_PARENT_RUN_LEDGER_FILE_SHA256",
    "V2101_PARENT_RUN_LEDGER_INTERNAL_SHA256",
    "V2101_PARENT_SCIENCE_COMMIT",
    "V2101_PARENT_SCIENCE_TAG",
    "V2101_PARENT_SCIENCE_TAG_OBJECT",
    "V2101_PARENT_SOURCE_MANIFEST_CONTENT_SHA256",
    "V2101_PARENT_SOURCE_MANIFEST_FILE_SHA256",
    "V2101_PARENT_SOURCE_MANIFEST_PATH",
    "V2101_PARENT_SOURCE_MANIFEST_SCHEMA_VERSION",
    "build_v2102_prerequisite_bindings",
    "build_v2102_resealed_observed_p95_authority",
    "build_v2102_source_manifest",
    "imported_prerequisite_path_v2102",
    "imported_v29_raw_root_v2102",
    "load_v2102_source_manifest",
    "parent_budget_debit_for_v2102",
    "persist_v2102_parent_import",
    "persist_v2102_resealed_observed_p95_authority",
    "snapshot_path_for_v29_source_artifact_v2102",
    "source_binding_for_target_v2102",
    "v2_9_p95_source_binding_v2102",
    "v2102_imported_v29_run_dir",
    "v2102_observed_p95_projection_path",
    "v2102_observed_p95_receipt_path",
    "validate_v2102_source_manifest",
    "verified_v2102_imported_prerequisite_binding",
    "verified_v2102_observed_p95_authority_binding",
    "verified_v2102_observed_p95_projection_binding",
    "verify_v2102_parent_import_receipt",
    "verify_v2102_resealed_observed_p95_authority",
    "verify_v2102_resealed_observed_p95_projection",
    "verify_v2101_terminal_lineage",
    "verify_v29_exact_source_for_v2102",
    "write_v2102_source_manifest_draft",
]
