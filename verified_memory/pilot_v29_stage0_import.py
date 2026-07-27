"""Immutable V2.8 -> V2.9 prerequisite and Stage-0 import primitives.

V2.8 is a terminal ``complete-with-no-go`` release.  V2.9 preserves its
211-cell denominator and copies the complete V2.8 raw namespace byte-for-byte.
Only the completed V2.8 parent prerequisite and the fourteen completed
physical V2.6 Stage-0 calibration cells nested below that namespace may be
bound as imported cells.

The failed V2.8 q-ref run is audit authority only.  Its 48 deterministic
scripted calls, failure receipt, and runner manifest are verified, but neither
its result nor any decoded completion is imported.  No function in this module
constructs a provider.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from .artifacts import ManifestVerificationError, verify_manifest
from .failure_artifacts import verify_failure_receipt
from .pilot_budget import ParentBudgetDebit
from .pilot_contract import (
    PilotContract,
    PilotRunSpec,
)
from .pilot_v24_parent_import import (
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
from .runner import verify_provider_call_journal
from . import pilot_v28_stage0_import as v28


V29_CONTRACT_ID = "finevo-pilot-v2.9"
V29_SCIENCE_TAG = "pilot-v2.9-science"
V29_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.9-source-manifest-v1"
V29_PARENT_IMPORT_SCHEMA_VERSION = "finevo-pilot-v2.9-parent-import-v1"
V29_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_9_source_manifest.json"
)
V29_EXPANDED_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_9.yaml")
V29_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.9/raw")
V29_SNAPSHOT_RELATIVE = PurePosixPath("parent-import/v2_8_raw_snapshot")
V29_ALLOWED_P95_PROFILES = ("gpt52_main", "llama33_local_controlled")

V28_CONTRACT_ID = "finevo-pilot-v2.8"
V28_CONTRACT_CANONICAL_SHA256 = (
    "948eac04516dd2c292d68beb732f97532b13e667a180e8c2db16fbb927f92f19"
)
V28_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_8.yaml")
V28_CONTRACT_FILE_SHA256 = (
    "bffdf90a76923531fdbd6672e3eec5bca3f3206c0117b64feddb5c4bfb7d7116"
)
V28_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_8_source_manifest.json"
)
V28_SOURCE_MANIFEST_FILE_SHA256 = (
    "1e95025be3466faa38936a3c4617ace0c625fa198eb506f1431a0b6401c4e1f8"
)
V28_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "f1cc3e150ae506e933233d42d7e8725c4c91e05a46c48e71c74e098eabeed3b4"
)
V28_SCIENCE_TAG = "pilot-v2.8-science"
V28_SCIENCE_TAG_OBJECT = "e5e77ff7f7f0f17792f77572c4c459db8e25f67e"
V28_SCIENCE_COMMIT = "1988f10b5a06c3b9b3093b969c99593676721a09"
V28_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.8/raw")
V28_RAW_FILE_COUNT = 271
V28_RAW_STORAGE_BYTES = 14_766_598
V28_RAW_INVENTORY_SCHEMA_VERSION = "finevo-raw-tree-inventory-v1"
V28_RAW_INVENTORY_CANONICALIZATION = (
    "json-sort-keys-compact-utf8-v1"
)
V28_RAW_INVENTORY_SHA256 = (
    "3dfdb24e52a1c2291bfc4882c1ff7d1dffed47c6c83a0f9c1f1eae825ec68e61"
)
V28_RUN_LEDGER_FILE_SHA256 = (
    "93cccdea076724b63c6b78553f8a319457f53ba916662036c8fe1e03668572d1"
)
V28_RUN_LEDGER_INTERNAL_SHA256 = (
    "9b5f4bd1acdc5a525fb58b04b02ba29e31b05b594bfc411863e7baf3eb11f0d9"
)
V28_RUN_LEDGER_EVENT_COUNT = 213
V28_RUN_LEDGER_EVENT_HEAD = (
    "d86d5d399ac3d11c36cf433be525ff6dac841b32da6fb2f588aa86a585257016"
)
V28_BUDGET_LEDGER_FILE_SHA256 = (
    "06f8f96f68f4bf922b03c6d8b5ef245c3a4c24a2fc4c316fa0a5fb64256392c1"
)
V28_BUDGET_LEDGER_INTERNAL_SHA256 = (
    "07c936d61a7c38e6a7877ffaeeaf6c8ecb7fd4f495dbe8ed012a9a2861004b8f"
)
V28_BUDGET_LEDGER_EVENT_COUNT = 6
V28_BUDGET_LEDGER_EVENT_HEAD = (
    "dd629e609760087a837fa4b7465aa804340e8d80c3c3ddcb8f0c8a8dc801cad4"
)
V28_RELEASE_ATTESTATION_FILE_SHA256 = (
    "14783912de6760ef843eae1af731556d2b48aef49ecb4d5241b7d548bc859ff1"
)
V28_PARENT_IMPORT_RECEIPT_FILE_SHA256 = (
    "c3fed4bc773018efc03e120d56a2b8ee7f70684c04193a3af731498eb2e06b98"
)
V28_PARENT_IMPORT_RECEIPT_CONTENT_SHA256 = (
    "ac5f36027f51dc6abf8c93dc04aa42629f890286bc3ba37bc33d2130bd8eed9a"
)
V28_PARENT_STAGE_RECEIPT_FILE_SHA256 = (
    "ee23223f17173f7bb90ff6c5e3ee7a1b7a5d421eb73bc6dedaa56f535642be73"
)
V28_PARENT_STAGE_RECEIPT_CONTENT_SHA256 = (
    "0721ed520b954fc84834d7c01710327d699081b6fcfd4e89fcfb2e0e76a8ad04"
)
V28_QREF_STAGE_RECEIPT_FILE_SHA256 = (
    "7f386e0423866f03f8eedf9106200c1c24dd262c9263853625d7a1eaa6b69d72"
)
V28_QREF_STAGE_RECEIPT_CONTENT_SHA256 = (
    "4e8b0a8082dffd05b63a76f2370ea70dd7c7397d07a377f7748bf473f508f0b2"
)
V28_QREF_RUN_ID = (
    "finevo-pilot-v2.8--q-ref-resolution--qref_scripted--qref-scripted--"
    "none--provider-preflight-default--s2010922376"
)
V28_QREF_FAILURE_MESSAGE = (
    "V2.8 fresh q-ref differs from its audit reference: "
    "['run_summary_exact']"
)
V28_QREF_FAILURE_MESSAGE_SHA256 = (
    "713b1d429fd939e74b2007d78d3c3789ce10376ee0ba970e1cfe1359503c246a"
)
V28_QREF_CONFIG_FILE_SHA256 = (
    "f06bf3d62e66669bcd8a705ad62c54e8231516c830c8ce9a71dc8a07df59a8c1"
)
V28_QREF_MANIFEST_FILE_SHA256 = (
    "8a5a2248880810fc95fd6e6d33e462b9ef43eaed74998b9a0be035c943b4f36b"
)
V28_QREF_PROVENANCE_FILE_SHA256 = (
    "640d6cd42f76119757a84a8dbd54a09ada71c4cf17d4a3d23da4ef991d53cdd9"
)
V28_QREF_FAILURE_FILE_SHA256 = (
    "9be1589cea0c7384f5d5a41c3029835a010a79b794444b9af606016507f38a05"
)
V28_QREF_FAILURE_MANIFEST_FILE_SHA256 = (
    "332eb6390c62a417d6ed1dc7c7c335a75ad732bea090418d6a08d8a6bcf2b92e"
)

V28_EVIDENCE_ROOT = PurePosixPath("evidence/current_v2/pilot-v2.8")
V28_EVIDENCE_PUBLICATION_COMMIT = (
    "00cc7142ae7af603f7989804a43c4d509456bad2"
)
V28_EVIDENCE_MERGE_COMMIT = (
    "981e2af20372c0413600f2bbd1b732f2d643593e"
)
V28_EVIDENCE_CHECKSUMS_FILE_SHA256 = (
    "1cc99291ba0bc9582c36414fce2bdc815d3cad0e753bdbb440140ad9f61127a9"
)
V28_EVIDENCE_PACKAGE_FILE_SHA256 = (
    "90580f8471b02ad4d156a6e39ce09676e5cccbeadb6a4d21ad54ff88a3867ef6"
)
V28_EVIDENCE_AGGREGATE_FILE_SHA256 = (
    "ea20a5b2c5534bb9430824683810d8172c0d099b1ee243ad1f169dcc26b367ce"
)
V28_EVIDENCE_FAILURE_FILE_SHA256 = (
    "a4058eebd5db76d9093214299c9f6f5fb187ca78c1f131a8e8c5e46cfaf8353d"
)
V28_EVIDENCE_REVIEWER_REPORT_FILE_SHA256 = (
    "a5c3757afefc1deb19bf33a13c9f8221b4423a6bf81497b3daf29455e5cdfa30"
)

V26_CONTRACT_ID = "finevo-pilot-v2.6"
V26_CONTRACT_CANONICAL_SHA256 = (
    "bb6b12d71227c423e5a67452dc496f26843dec74e359b9b04bf096dc17d0c509"
)
V26_NESTED_IN_V28_RELATIVE = PurePosixPath(
    "parent-import/v2_7_raw_snapshot/parent-import/v2_6_raw_snapshot"
)
V26_RAW_FILE_COUNT = v28.V26_RAW_FILE_COUNT
V26_RAW_STORAGE_BYTES = v28.V26_RAW_STORAGE_BYTES
V26_RAW_INVENTORY_SHA256 = v28.V26_RAW_INVENTORY_SHA256
V26_STAGE0_RECEIPT_FILE_SHA256 = v28.V26_STAGE0_RECEIPT_FILE_SHA256
V26_STAGE0_RECEIPT_CONTENT_SHA256 = v28.V26_STAGE0_RECEIPT_CONTENT_SHA256

V29_CUMULATIVE_DEBIT = ParentBudgetDebit(
    parent_contract_sha256=V28_CONTRACT_CANONICAL_SHA256,
    parent_run_ledger_sha256=V28_RUN_LEDGER_INTERNAL_SHA256,
    parent_budget_ledger_sha256=V28_BUDGET_LEDGER_INTERNAL_SHA256,
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

_V28_P95_SOURCES: dict[str, dict[str, Any]] = {
    "gpt52_main": {
        "runtime_model": "openai/gpt-5.2-2025-12-11",
        "served_model": "gpt-5.2-2025-12-11",
        "authority": {
            "schema_version":
            "finevo-pilot-v2.8-inherited-observed-p95-authority-v1",
            "file_sha256":
            "914e0b72735098677efd2fadf06b98060975eab6390236306464f77cd5984bb3",
            "content_sha256":
            "25034146171fd9a6e21459a906cb7f5d2b76aa042e403513f203776aa8ee56c6",
        },
        "projection": {
            "schema_version": "finevo-pilot-projection-p95-v1",
            "file_sha256":
            "0aa429ee3b732383a0305fd16176733a27c6f7890a7c68cf9253ae29fff4856a",
            "content_sha256":
            "dcd5b321c3dfb669aac8a1880cae3e5378d96176d60633baa124854b13b5d42c",
        },
    },
    "llama33_local_controlled": {
        "runtime_model": "ollama/llama3.3:70b-instruct-q4_K_M",
        "served_model": "llama3.3:70b-instruct-q4_K_M",
        "authority": {
            "schema_version":
            "finevo-pilot-v2.8-inherited-observed-p95-authority-v1",
            "file_sha256":
            "0b3536bdd0e28930d9f3f7598a372c0285ace5d9945904a7d4a10bc7f66e9bf7",
            "content_sha256":
            "3c887a64d255fc0428dbbd34ec8dc2de1ea6c86d21b358c384d2a5316cd44cdd",
        },
        "projection": {
            "schema_version": "finevo-pilot-projection-p95-v1",
            "file_sha256":
            "05e4c02b3307d4928ceb568bd45b1cd5adce5532b83b005f85735579bbd9c6ad",
            "content_sha256":
            "eda5fb22bd59f2fb7593208b5eab6c6dacf2ef15a169809cc2f8e55923a2c3cb",
        },
    },
}


class PilotV29Stage0ImportError(RuntimeError):
    """Raised before immutable V2.8/V2.6 authority can enter V2.9."""


def _translate(exc: Exception) -> PilotV29Stage0ImportError:
    return PilotV29Stage0ImportError(str(exc))


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotV29Stage0ImportError(f"{name} must be an object")
    return value


def _strict_real_root(value: str | Path, *, name: str) -> Path:
    """Reject a repository/directory path with a symlink at any component."""

    root = Path(value).absolute()
    for component in (root, *root.parents):
        try:
            if component.is_symlink():
                raise PilotV29Stage0ImportError(
                    f"{name} path contains a symlink"
                )
        except OSError as exc:
            raise PilotV29Stage0ImportError(
                f"{name} is unavailable"
            ) from exc
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
        raise PilotV29Stage0ImportError(f"{name} file hash drifted")
    return path, raw, value


def _artifact_binding(
    root: Path,
    relative: PurePosixPath,
    *,
    expected_sha256: str | None = None,
    name: str,
) -> dict[str, Any]:
    try:
        _, raw = _guarded_file(root, relative, name=name)
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    digest = _sha256(raw)
    if expected_sha256 is not None and digest != expected_sha256:
        raise PilotV29Stage0ImportError(f"{name} file hash drifted")
    return {
        "path": relative.as_posix(),
        "file_sha256": digest,
        "byte_size": len(raw),
    }


def _verify_self_hashed(
    value: Mapping[str, Any],
    *,
    schema_version: str,
    name: str,
    expected_content_sha256: str | None = None,
) -> None:
    try:
        _verify_self_hash(
            value,
            schema_version=schema_version,
            name=name,
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if (
        expected_content_sha256 is not None
        and value.get("integrity", {}).get("content_sha256")
        != expected_content_sha256
    ):
        raise PilotV29Stage0ImportError(f"{name} content hash drifted")


def _inventory(
    root: Path,
    *,
    declared_root: PurePosixPath,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the canonical no-follow raw-tree inventory.

    Each row is ``{path, byte_size, sha256}``, sorted by path.  The inventory
    digest is SHA-256 over compact, sorted-key, UTF-8 JSON of the full row list.
    """

    try:
        rows, summary = v28._inventory(root, declared_root=declared_root)
    except v28.PilotV28Stage0ImportError as exc:
        raise _translate(exc) from exc
    return rows, summary


def _verify_exact_v28_inventory(raw_root: Path) -> list[dict[str, Any]]:
    rows, summary = _inventory(raw_root, declared_root=V28_RAW_ROOT)
    expected = {
        "root": V28_RAW_ROOT.as_posix(),
        "inventory_schema_version": V28_RAW_INVENTORY_SCHEMA_VERSION,
        "inventory_canonicalization":
        V28_RAW_INVENTORY_CANONICALIZATION,
        "file_count": V28_RAW_FILE_COUNT,
        "storage_bytes": V28_RAW_STORAGE_BYTES,
        "inventory_sha256": V28_RAW_INVENTORY_SHA256,
    }
    if summary != expected:
        raise PilotV29Stage0ImportError(
            "V2.8 raw-tree inventory drifted"
        )
    return rows


def _verify_exact_nested_v26_inventory(raw_root: Path) -> None:
    nested = raw_root.joinpath(*V26_NESTED_IN_V28_RELATIVE.parts)
    try:
        v28._verify_exact_inventory(
            nested,
            declared_root=PurePosixPath(
                "experiment_results/pilot-v2.6/raw"
            ),
            file_count=V26_RAW_FILE_COUNT,
            storage_bytes=V26_RAW_STORAGE_BYTES,
            inventory_sha256=V26_RAW_INVENTORY_SHA256,
            name="nested V2.6",
        )
    except v28.PilotV28Stage0ImportError as exc:
        raise _translate(exc) from exc


def _validate_target_contract(
    contract: PilotContract,
    *,
    require_frozen: bool,
) -> None:
    amendment = contract.qref_summary_equivalence_amendment
    if (
        contract.contract_id != V29_CONTRACT_ID
        or contract.implementation.get("required_git_tag") != V29_SCIENCE_TAG
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
        != "finevo-pilot-qref-summary-equivalence-retry-amendment-v1"
    ):
        raise PilotV29Stage0ImportError(
            "V2.9 import requires its exact q-ref equivalence contract"
        )
    lineage = _mapping(
        amendment.get("source_lineage"),
        name="V2.9 source lineage",
    )
    stage0 = _mapping(
        amendment.get("stage0_import"),
        name="V2.9 Stage-0 import",
    )
    qref = _mapping(
        amendment.get("q_ref_regeneration"),
        name="V2.9 q-ref regeneration",
    )
    retry = _mapping(
        amendment.get("retry_policy"),
        name="V2.9 retry policy",
    )
    if (
        lineage.get("amendment_parent_contract_id") != V28_CONTRACT_ID
        or lineage.get("amendment_parent_contract_sha256")
        != V28_CONTRACT_CANONICAL_SHA256
        or lineage.get("amendment_parent_raw_namespace")
        != V28_RAW_ROOT.as_posix()
        or lineage.get("child_raw_namespace") != V29_RAW_ROOT.as_posix()
        or lineage.get("parent_terminal_no_go_preserved") is not True
        or lineage.get("parent_denominator_reclassified") is not False
        or stage0.get("source_contract_id") != V26_CONTRACT_ID
        or stage0.get("source_contract_sha256")
        != V26_CONTRACT_CANONICAL_SHA256
        or stage0.get("imported_complete_cells") != 14
        or stage0.get("provider_construction_during_import") is not False
        or qref.get("audit_reference_contract_id") != V28_CONTRACT_ID
        or qref.get("source_result_reuse") != "forbidden"
        or qref.get("fresh_zero_hosted_provider_regeneration") is not True
        or qref.get("scripted_diagnostic_calls") != 48
        or qref.get("hosted_provider_calls") != 0
        or qref.get("hosted_cost_usd") != 0.0
        or retry.get("v2_8_raw_resume") != "forbidden"
        or retry.get("v2_8_terminal_cell_reclassification") != "forbidden"
    ):
        raise PilotV29Stage0ImportError(
            "V2.9 immutable-source or fresh-q-ref policy drifted"
        )
    if len(contract.expand()) != 211:
        raise PilotV29Stage0ImportError(
            "V2.9 denominator must contain exactly 211 cells"
        )


def _verify_v28_release_identity(parent_root: Path) -> None:
    try:
        head = _git(parent_root, "rev-parse", "--verify", "HEAD^{commit}")
        tag_object = _git(
            parent_root,
            "rev-parse",
            "--verify",
            f"refs/tags/{V28_SCIENCE_TAG}^{{tag}}",
        )
        tag_commit = _git(
            parent_root,
            "rev-parse",
            "--verify",
            f"refs/tags/{V28_SCIENCE_TAG}^{{commit}}",
        )
        tracked_status = _git(
            parent_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if (
        head != V28_SCIENCE_COMMIT
        or tag_object != V28_SCIENCE_TAG_OBJECT
        or tag_commit != V28_SCIENCE_COMMIT
        or tracked_status
    ):
        raise PilotV29Stage0ImportError(
            "V2.8 science checkout/tag identity drifted"
        )


def _load_verified_v28_contract(parent_root: Path) -> PilotContract:
    _, raw, value = _strict_file(
        parent_root,
        V28_CONTRACT_PATH,
        name="V2.8 contract",
        expected_sha256=V28_CONTRACT_FILE_SHA256,
    )
    contract = PilotContract.from_dict(value)
    if (
        _sha256(raw) != V28_CONTRACT_FILE_SHA256
        or contract.contract_id != V28_CONTRACT_ID
        or contract.canonical_hash != V28_CONTRACT_CANONICAL_SHA256
        or contract.status != "frozen"
        or len(contract.expand()) != 211
    ):
        raise PilotV29Stage0ImportError(
            "V2.8 frozen contract identity drifted"
        )
    return contract


def _load_verified_v28_source_manifest(parent_root: Path) -> dict[str, Any]:
    _, _, value = _strict_file(
        parent_root,
        V28_SOURCE_MANIFEST_PATH,
        name="V2.8 source manifest",
        expected_sha256=V28_SOURCE_MANIFEST_FILE_SHA256,
    )
    _verify_self_hashed(
        value,
        schema_version=v28.V28_SOURCE_MANIFEST_SCHEMA_VERSION,
        name="V2.8 source manifest",
        expected_content_sha256=V28_SOURCE_MANIFEST_CONTENT_SHA256,
    )
    rows = value.get("imported_complete_cells")
    if (
        not isinstance(rows, list)
        or len(rows) != 15
        or Counter(str(row.get("stage_id")) for row in rows)
        != Counter({"parent-import": 1, "stage0-calibration": 14})
        or value.get("import_policy", {}).get(
            "provider_construction_during_import"
        )
        is not False
        or value.get("q_ref_audit_equivalence_reference", {}).get(
            "imported"
        )
        is not False
    ):
        raise PilotV29Stage0ImportError(
            "V2.8 source manifest import boundary drifted"
        )
    return value


def _verify_v28_ledgers(
    raw_root: Path,
    source_contract: PilotContract,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, _, run = _strict_file(
        raw_root,
        PurePosixPath("run_ledger.json"),
        name="V2.8 run ledger",
        expected_sha256=V28_RUN_LEDGER_FILE_SHA256,
    )
    _, _, budget = _strict_file(
        raw_root,
        PurePosixPath("budget_ledger.json"),
        name="V2.8 budget ledger",
        expected_sha256=V28_BUDGET_LEDGER_FILE_SHA256,
    )
    runs = _mapping(run.get("runs"), name="V2.8 run ledger rows")
    events = run.get("events")
    budget_events = budget.get("events")
    expanded = {
        spec.run_id: spec.to_dict()
        for spec in source_contract.expand()
    }
    if (
        run.get("schema_version") != "finevo-pilot-run-ledger-v2"
        or run.get("contract_hash") != V28_CONTRACT_CANONICAL_SHA256
        or run.get("ledger_sha256") != V28_RUN_LEDGER_INTERNAL_SHA256
        or not isinstance(events, list)
        or len(events) != V28_RUN_LEDGER_EVENT_COUNT
        or events[-1].get("event_sha256") != V28_RUN_LEDGER_EVENT_HEAD
        or set(runs) != set(expanded)
        or any(
            row.get("spec") != expanded[run_id]
            for run_id, row in runs.items()
        )
        or Counter(str(row.get("status")) for row in runs.values())
        != Counter({"complete": 1, "failed": 1, "integrity-stopped": 209})
        or budget.get("schema_version") != "finevo-pilot-budget-ledger-v2"
        or budget.get("contract_hash") != V28_CONTRACT_CANONICAL_SHA256
        or budget.get("ledger_sha256") != V28_BUDGET_LEDGER_INTERNAL_SHA256
        or not isinstance(budget_events, list)
        or len(budget_events) != V28_BUDGET_LEDGER_EVENT_COUNT
        or budget_events[-1].get("event_sha256")
        != V28_BUDGET_LEDGER_EVENT_HEAD
    ):
        raise PilotV29Stage0ImportError(
            "V2.8 ledger identity or terminal denominator drifted"
        )
    return run, budget


def _verify_v28_parent_receipts(raw_root: Path) -> dict[str, Any]:
    receipt_relative = PurePosixPath(
        "parent-import/parent_import_receipt.json"
    )
    _, receipt_raw, receipt = _strict_file(
        raw_root,
        receipt_relative,
        name="V2.8 parent import receipt",
        expected_sha256=V28_PARENT_IMPORT_RECEIPT_FILE_SHA256,
    )
    _verify_self_hashed(
        receipt,
        schema_version=v28.V28_PARENT_IMPORT_SCHEMA_VERSION,
        name="V2.8 parent import receipt",
        expected_content_sha256=V28_PARENT_IMPORT_RECEIPT_CONTENT_SHA256,
    )
    stage_relative = PurePosixPath("parent-import/stage_receipt.json")
    _, stage_raw, stage = _strict_file(
        raw_root,
        stage_relative,
        name="V2.8 parent stage receipt",
        expected_sha256=V28_PARENT_STAGE_RECEIPT_FILE_SHA256,
    )
    try:
        v28._verify_v2_stage_receipt_self_hash(
            stage,
            name="V2.8 parent stage receipt",
        )
    except v28.PilotV28Stage0ImportError as exc:
        raise _translate(exc) from exc
    if (
        stage.get("contract_id") != V28_CONTRACT_ID
        or stage.get("contract_sha256") != V28_CONTRACT_CANONICAL_SHA256
        or stage.get("stage_id") != "parent-import"
        or stage.get("status") != "complete"
        or stage.get("complete_cell_count") != 1
        or stage.get("integrity", {}).get("content_sha256")
        != V28_PARENT_STAGE_RECEIPT_CONTENT_SHA256
    ):
        raise PilotV29Stage0ImportError(
            "V2.8 parent prerequisite receipt drifted"
        )
    return {
        "receipt": {
            "path": (V28_RAW_ROOT / receipt_relative).as_posix(),
            "file_sha256": _sha256(receipt_raw),
            "content_sha256": V28_PARENT_IMPORT_RECEIPT_CONTENT_SHA256,
        },
        "stage_receipt": {
            "path": (V28_RAW_ROOT / stage_relative).as_posix(),
            "file_sha256": _sha256(stage_raw),
            "content_sha256": V28_PARENT_STAGE_RECEIPT_CONTENT_SHA256,
        },
    }


def _jsonl_binding(
    parent_root: Path,
    relative: PurePosixPath,
    *,
    name: str,
    expected_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        _, raw = _guarded_file(parent_root, relative, name=name)
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise PilotV29Stage0ImportError(
                f"{name} contains an empty JSONL row"
            )
        try:
            rows.append(_strict_json(line, name=f"{name} row {index}"))
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
    if len(rows) != expected_rows:
        raise PilotV29Stage0ImportError(f"{name} row count drifted")
    return (
        {
            "path": relative.as_posix(),
            "file_sha256": _sha256(raw),
            "byte_size": len(raw),
            "row_count": len(rows),
        },
        rows,
    )


def _verify_v28_qref_failure(
    parent_root: Path,
    raw_root: Path,
    source_contract: PilotContract,
    v28_source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    stage_relative = V28_RAW_ROOT / "q-ref-resolution/stage_receipt.json"
    _, stage_raw, stage = _strict_file(
        parent_root,
        stage_relative,
        name="V2.8 q-ref stage receipt",
        expected_sha256=V28_QREF_STAGE_RECEIPT_FILE_SHA256,
    )
    try:
        v28._verify_v2_stage_receipt_self_hash(
            stage,
            name="V2.8 q-ref stage receipt",
        )
    except v28.PilotV28Stage0ImportError as exc:
        raise _translate(exc) from exc
    if (
        stage.get("contract_id") != V28_CONTRACT_ID
        or stage.get("contract_sha256") != V28_CONTRACT_CANONICAL_SHA256
        or stage.get("stage_id") != "q-ref-resolution"
        or stage.get("status") != "complete-with-no-go"
        or stage.get("registered_run_count") != 1
        or stage.get("complete_cell_count") != 0
        or stage.get("status_counts") != {"failed": 1}
        or stage.get("go") is not False
        or stage.get("execution_progression_go") is not False
        or stage.get("integrity", {}).get("content_sha256")
        != V28_QREF_STAGE_RECEIPT_CONTENT_SHA256
    ):
        raise PilotV29Stage0ImportError(
            "V2.8 q-ref failure stage classification drifted"
        )

    specs = source_contract.expand(stage="q-ref-resolution")
    if len(specs) != 1 or specs[0].run_id != V28_QREF_RUN_ID:
        raise PilotV29Stage0ImportError(
            "V2.8 q-ref denominator identity drifted"
        )
    spec = specs[0]
    run_relative = (
        V28_RAW_ROOT / "q-ref-resolution/runs" / V28_QREF_RUN_ID
    )
    run_dir = parent_root.joinpath(*run_relative.parts)
    try:
        verify_manifest(run_dir)
    except ManifestVerificationError as exc:
        expected_extra = (
            "manifest file set mismatch; missing=[], "
            "extra=['failure_receipt/failure.json', "
            "'failure_receipt/failure_manifest.json']"
        )
        if str(exc) != expected_extra:
            raise PilotV29Stage0ImportError(
                f"V2.8 q-ref runner manifest failed verification: {exc}"
            ) from exc
        # Failure receipts are deliberately written after runner finalization.
        # Re-hash every sealed entry, then permit only those two independently
        # verified receipt files as the post-manifest extension.
        _, _, sealed = _strict_file(
            parent_root,
            run_relative / "manifest.json",
            name="V2.8 q-ref runner manifest",
            expected_sha256=V28_QREF_MANIFEST_FILE_SHA256,
        )
        artifacts = sealed.get("artifacts")
        if not isinstance(artifacts, list):
            raise PilotV29Stage0ImportError(
                "V2.8 q-ref manifest artifact inventory is malformed"
            )
        for entry in artifacts:
            if not isinstance(entry, Mapping):
                raise PilotV29Stage0ImportError(
                    "V2.8 q-ref manifest entry is malformed"
                )
            try:
                relative = _normalized_relative(
                    str(entry.get("path", "")),
                    required_top=None,
                    name="V2.8 q-ref manifest artifact path",
                )
                _, raw = _guarded_file(
                    run_dir,
                    relative,
                    name="V2.8 q-ref manifest artifact",
                )
            except PilotV24ParentImportError as path_exc:
                raise _translate(path_exc) from path_exc
            if (
                _sha256(raw) != entry.get("sha256")
                or len(raw) != entry.get("byte_size")
                or raw.count(b"\n") != entry.get("line_count")
            ):
                raise PilotV29Stage0ImportError(
                    "V2.8 q-ref sealed artifact drifted"
                )
    try:
        failure = verify_failure_receipt(run_dir / "failure_receipt")
    except Exception as exc:
        raise PilotV29Stage0ImportError(
            f"V2.8 q-ref failure receipt failed verification: {exc}"
        ) from exc
    error = _mapping(failure.get("error"), name="V2.8 q-ref error")
    budget = _mapping(
        failure.get("budget_snapshot"),
        name="V2.8 q-ref budget snapshot",
    )
    accounted = _mapping(
        budget.get("accounted_usage"),
        name="V2.8 q-ref accounted usage",
    )
    completions = budget.get("completions")
    if (
        failure.get("status") != "failed"
        or failure.get("scope")
        != "finevo-pilot/q-ref-resolution/q_ref_resolution"
        or error.get("type") != "PilotOrchestrationError"
        or error.get("message") != V28_QREF_FAILURE_MESSAGE
        or error.get("message_sha256") != V28_QREF_FAILURE_MESSAGE_SHA256
        or failure.get("partial_streams_persisted") is not False
        or budget.get("completed_calls") != 48
        or budget.get("active_calls") != 0
        or not isinstance(completions, list)
        or len(completions) != 48
        or {row.get("model") for row in completions}
        != {"diagnostic/scripted-v1"}
        or accounted
        != {
            "prompt_tokens": 14657,
            "completion_tokens": 1248,
            "total_tokens": 15905,
            "cost_usd": 0.0,
        }
    ):
        raise PilotV29Stage0ImportError(
            "V2.8 q-ref failure/provider accounting drifted"
        )

    fixed_artifacts = {
        "config": (
            "config.json",
            V28_QREF_CONFIG_FILE_SHA256,
        ),
        "manifest": (
            "manifest.json",
            V28_QREF_MANIFEST_FILE_SHA256,
        ),
        "provenance": (
            "provenance.json",
            V28_QREF_PROVENANCE_FILE_SHA256,
        ),
        "failure": (
            "failure_receipt/failure.json",
            V28_QREF_FAILURE_FILE_SHA256,
        ),
        "failure_manifest": (
            "failure_receipt/failure_manifest.json",
            V28_QREF_FAILURE_MANIFEST_FILE_SHA256,
        ),
    }
    bindings = {
        key: _artifact_binding(
            parent_root,
            run_relative / suffix,
            expected_sha256=digest,
            name=f"V2.8 q-ref {key}",
        )
        for key, (suffix, digest) in fixed_artifacts.items()
    }
    streams: dict[str, dict[str, Any]] = {}
    stream_rows: dict[str, list[dict[str, Any]]] = {}
    for stream_name, count in (
        ("summary", 1),
        ("actions", 48),
        ("api_usage", 48),
        ("utility_ledger", 48),
        ("shock_events", 12),
    ):
        binding, rows = _jsonl_binding(
            parent_root,
            run_relative / f"streams/{stream_name}.jsonl",
            name=f"V2.8 q-ref {stream_name}",
            expected_rows=count,
        )
        streams[stream_name] = binding
        stream_rows[stream_name] = rows
    api_usage = stream_rows["api_usage"]
    usage_totals = {
        key: sum(
            int(row.get("usage", {}).get(key, 0))
            for row in api_usage
        )
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    usage_cost = sum(
        float(row.get("usage", {}).get("cost_usd", 0.0))
        for row in api_usage
    )
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
        any(
            row.get("provider") != "diagnostic"
            or row.get("model") != "scripted-v1"
            or row.get("response_model") != "scripted-v1"
            or row.get("attempts") != 1
            or row.get("error_type") is not None
            or row.get("output_disposition") != "accepted"
            for row in api_usage
        )
        or usage_totals
        != {
            "prompt_tokens": 14657,
            "completion_tokens": 1248,
            "total_tokens": 15905,
        }
        or usage_cost != 0.0
        or stream_rows["shock_events"] != expected_shocks
    ):
        raise PilotV29Stage0ImportError(
            "V2.8 q-ref source-core identity drifted"
        )
    ancestral = _mapping(
        v28_source_manifest.get("q_ref_audit_equivalence_reference"),
        name="ancestral V2.6 q-ref reference",
    )
    if (
        ancestral.get("source_contract_id") != V26_CONTRACT_ID
        or ancestral.get("source_contract_sha256")
        != V26_CONTRACT_CANONICAL_SHA256
        or ancestral.get("q_ref") != 63.50397933257746
        or ancestral.get("imported") is not False
        or ancestral.get("source_result_reuse") != "forbidden"
    ):
        raise PilotV29Stage0ImportError(
            "ancestral V2.6 q-ref scalar reference drifted"
        )
    return {
        "imported": False,
        "source_result_reuse": "forbidden",
        "source_contract_id": V28_CONTRACT_ID,
        "source_contract_sha256": V28_CONTRACT_CANONICAL_SHA256,
        "source_contract_cell_run_id": spec.run_id,
        "source_run_root": run_relative.as_posix(),
        "source_spec": spec.to_dict(),
        "failed_prerequisite": {
            "status": "failed",
            "error_type": "PilotOrchestrationError",
            "error_message": V28_QREF_FAILURE_MESSAGE,
            "error_message_sha256": V28_QREF_FAILURE_MESSAGE_SHA256,
            "stage_receipt": {
                "path": stage_relative.as_posix(),
                "file_sha256": _sha256(stage_raw),
                "content_sha256":
                V28_QREF_STAGE_RECEIPT_CONTENT_SHA256,
            },
            "failure": bindings["failure"],
            "failure_manifest": bindings["failure_manifest"],
        },
        "verified_runner": {
            "manifest": bindings["manifest"],
            "config": bindings["config"],
            "provenance": bindings["provenance"],
            "streams": streams,
            "identity_grid": {
                "agents": 4,
                "periods": 12,
                "action_rows": 48,
                "api_usage_rows": 48,
                "utility_ledger_rows": 48,
                "shock_rows": 12,
                "summary_rows": 1,
            },
            "provider_accounting": {
                "provider_kind": "scripted-diagnostic",
                "model": "diagnostic/scripted-v1",
                "scripted_diagnostic_calls": 48,
                "hosted_provider_calls": 0,
                "hosted_cost_usd": 0.0,
                "total_tokens": 15905,
            },
        },
        "ancestral_v2_6_scalar_reference": {
            key: _json_copy(ancestral[key])
            for key in (
                "q_ref",
                "source_contract_id",
                "source_contract_sha256",
                "source_run_id",
                "source_run_root",
                "source_config",
                "source_manifest",
                "source_core",
            )
        },
        "fresh_v2_9_policy": {
            "scripted_diagnostic_calls": 48,
            "hosted_provider_calls": 0,
            "hosted_cost_usd": 0.0,
            "provider_construction_during_regeneration": False,
            "source_result_reuse": "forbidden",
        },
    }


def _verify_v28_published_evidence(child_root: Path) -> dict[str, Any]:
    try:
        publication = _git(
            child_root,
            "rev-parse",
            "--verify",
            f"{V28_EVIDENCE_PUBLICATION_COMMIT}^{{commit}}",
        )
        merge = _git(
            child_root,
            "rev-parse",
            "--verify",
            f"{V28_EVIDENCE_MERGE_COMMIT}^{{commit}}",
        )
        parents = _git(
            child_root,
            "rev-list",
            "--parents",
            "-n",
            "1",
            V28_EVIDENCE_MERGE_COMMIT,
        ).split()
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if (
        publication != V28_EVIDENCE_PUBLICATION_COMMIT
        or merge != V28_EVIDENCE_MERGE_COMMIT
        or parents
        != [
            V28_EVIDENCE_MERGE_COMMIT,
            V28_SCIENCE_COMMIT,
            V28_EVIDENCE_PUBLICATION_COMMIT,
        ]
    ):
        raise PilotV29Stage0ImportError(
            "V2.8 evidence publication/merge identity drifted"
        )
    _, checksums_raw, checksums = _strict_file(
        child_root,
        V28_EVIDENCE_ROOT / "checksums.json",
        name="V2.8 evidence checksums",
        expected_sha256=V28_EVIDENCE_CHECKSUMS_FILE_SHA256,
    )
    files = checksums.get("files")
    if (
        checksums.get("contract_sha256")
        != V28_CONTRACT_CANONICAL_SHA256
        or not isinstance(files, list)
        or len(files) != 16
    ):
        raise PilotV29Stage0ImportError(
            "V2.8 evidence checksum inventory drifted"
        )
    evidence_root = child_root.joinpath(*V28_EVIDENCE_ROOT.parts)
    actual_files: set[str] = set()
    for path in evidence_root.rglob("*"):
        if path.is_symlink():
            raise PilotV29Stage0ImportError(
                "V2.8 evidence package contains a symlink"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise PilotV29Stage0ImportError(
                "V2.8 evidence package contains a non-regular entry"
            )
        actual_files.add(path.relative_to(evidence_root).as_posix())
    declared_files = {
        str(row.get("path"))
        for row in files
        if isinstance(row, Mapping)
    }
    if (
        len(declared_files) != 16
        or actual_files != declared_files | {"checksums.json"}
    ):
        raise PilotV29Stage0ImportError(
            "V2.8 evidence package file set drifted"
        )
    try:
        current_checksums_blob = _git(
            child_root,
            "hash-object",
            (V28_EVIDENCE_ROOT / "checksums.json").as_posix(),
        )
        published_checksums_blob = _git(
            child_root,
            "rev-parse",
            f"{V28_EVIDENCE_PUBLICATION_COMMIT}:"
            f"{(V28_EVIDENCE_ROOT / 'checksums.json').as_posix()}",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if current_checksums_blob != published_checksums_blob:
        raise PilotV29Stage0ImportError(
            "V2.8 evidence checksums differ from publication commit"
        )
    bindings: list[dict[str, Any]] = []
    for row in files:
        relative_text = row.get("path")
        try:
            relative = _normalized_relative(
                str(relative_text),
                required_top=None,
                name="V2.8 evidence checksum path",
            )
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
        binding = _artifact_binding(
            child_root,
            V28_EVIDENCE_ROOT / relative,
            expected_sha256=str(row.get("sha256")),
            name=f"V2.8 evidence {relative.as_posix()}",
        )
        if binding["byte_size"] != row.get("byte_size"):
            raise PilotV29Stage0ImportError(
                "V2.8 evidence checksum size drifted"
            )
        bindings.append(binding)
        try:
            current_blob = _git(
                child_root,
                "hash-object",
                (V28_EVIDENCE_ROOT / relative).as_posix(),
            )
            published_blob = _git(
                child_root,
                "rev-parse",
                f"{V28_EVIDENCE_PUBLICATION_COMMIT}:"
                f"{(V28_EVIDENCE_ROOT / relative).as_posix()}",
            )
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
        if current_blob != published_blob:
            raise PilotV29Stage0ImportError(
                "V2.8 evidence differs from publication commit"
            )

    _, _, package = _strict_file(
        child_root,
        V28_EVIDENCE_ROOT / "package_manifest.json",
        name="V2.8 evidence package",
        expected_sha256=V28_EVIDENCE_PACKAGE_FILE_SHA256,
    )
    _, _, aggregate = _strict_file(
        child_root,
        V28_EVIDENCE_ROOT / "aggregate.json",
        name="V2.8 aggregate",
        expected_sha256=V28_EVIDENCE_AGGREGATE_FILE_SHA256,
    )
    _, _, failure = _strict_file(
        child_root,
        V28_EVIDENCE_ROOT / "failure_ledger.json",
        name="V2.8 failure ledger",
        expected_sha256=V28_EVIDENCE_FAILURE_FILE_SHA256,
    )
    _artifact_binding(
        child_root,
        V28_EVIDENCE_ROOT / "reviewer_report.md",
        expected_sha256=V28_EVIDENCE_REVIEWER_REPORT_FILE_SHA256,
        name="V2.8 reviewer report",
    )
    denominator = aggregate.get("denominator")
    budget = aggregate.get("budget")
    if (
        package.get("schema_version")
        != "finevo-pilot-v2.8-evidence-package-v1"
        or package.get("publication_status") != "complete-with-no-go"
        or package.get("resolved_git_commit") != V28_SCIENCE_COMMIT
        or package.get("pilot_tag") != V28_SCIENCE_TAG
        or package.get("scientific_complete") is not False
        or package.get("scientific_matrix_complete") is not False
        or package.get("scientific_claim_gates_supported") is not False
        or not isinstance(denominator, Mapping)
        or denominator.get("expected_count") != 211
        or denominator.get("observed_ledger_count") != 211
        or denominator.get("all_rows_present") is not True
        or denominator.get("all_rows_terminal") is not True
        or denominator.get("status_counts")
        != {"complete": 1, "failed": 1, "integrity-stopped": 209}
        or not isinstance(budget, Mapping)
        or budget.get("pass") is not True
        or budget.get("raw_root_storage_bytes") != V28_RAW_STORAGE_BYTES
        or failure.get("schema_version")
        != "finevo-pilot-failure-ledger-v1"
        or failure.get("contract_sha256")
        != V28_CONTRACT_CANONICAL_SHA256
        or failure.get("denominator") != denominator
        or not isinstance(failure.get("rows"), list)
        or len(failure["rows"]) != 210
        or Counter(str(row.get("status")) for row in failure["rows"])
        != Counter({"failed": 1, "integrity-stopped": 209})
    ):
        raise PilotV29Stage0ImportError(
            "V2.8 evidence denominator or no-go boundary drifted"
        )
    return {
        "root": V28_EVIDENCE_ROOT.as_posix(),
        "schema_version": package["schema_version"],
        "publication_commit": V28_EVIDENCE_PUBLICATION_COMMIT,
        "merge_commit": V28_EVIDENCE_MERGE_COMMIT,
        "checksums": {
            "path": (V28_EVIDENCE_ROOT / "checksums.json").as_posix(),
            "file_sha256": _sha256(checksums_raw),
            "entry_count": len(bindings),
            "files": bindings,
        },
        "package_manifest_file_sha256":
        V28_EVIDENCE_PACKAGE_FILE_SHA256,
        "aggregate_file_sha256": V28_EVIDENCE_AGGREGATE_FILE_SHA256,
        "failure_ledger_file_sha256": V28_EVIDENCE_FAILURE_FILE_SHA256,
        "reviewer_report_file_sha256":
        V28_EVIDENCE_REVIEWER_REPORT_FILE_SHA256,
        "publication_status": "complete-with-no-go",
        "scientific_complete": False,
        "scientific_matrix_complete": False,
        "scientific_claim_gates_supported": False,
    }


def _p95_live_paths(
    parent_root: Path,
    profile_id: str,
) -> tuple[Path, Path]:
    if profile_id not in V29_ALLOWED_P95_PROFILES:
        raise PilotV29Stage0ImportError(
            f"unsupported V2.8 p95 profile: {profile_id}"
        )
    base = (
        parent_root
        / V28_RAW_ROOT
        / "parent-import"
        / "observed_p95"
        / profile_id
    )
    return (
        base / "observed_p95_authority_receipt.json",
        base / "projection_p95.json",
    )


def v2_8_p95_source_binding_v29(
    parent_repo_root: str | Path,
    profile_id: str,
    *,
    expected_parent_commit: str = V28_SCIENCE_COMMIT,
) -> dict[str, Any]:
    """Verify one live V2.8 p95 receipt/projection before snapshot copy."""

    if expected_parent_commit != V28_SCIENCE_COMMIT:
        raise PilotV29Stage0ImportError(
            "V2.8 p95 source commit differs from immutable parent"
        )
    root = _strict_real_root(
        parent_repo_root,
        name="V2.8 p95 source repository",
    )
    receipt_path, projection_path = _p95_live_paths(root, profile_id)
    source = _V28_P95_SOURCES[profile_id]
    try:
        reservations = v28.verify_v28_resealed_observed_p95_authority(
            receipt_path,
            repo_root=root,
            expected_git_commit=expected_parent_commit,
        )
        projection = v28.verify_v28_resealed_observed_p95_projection(
            projection_path,
            receipt_or_path=receipt_path,
            repo_root=root,
            expected_git_commit=expected_parent_commit,
        )
    except v28.PilotV28Stage0ImportError as exc:
        raise _translate(exc) from exc
    authority_relative = (
        V28_RAW_ROOT
        / "parent-import"
        / "observed_p95"
        / profile_id
        / "observed_p95_authority_receipt.json"
    )
    projection_relative = authority_relative.parent / "projection_p95.json"
    _, _, authority = _strict_file(
        root,
        authority_relative,
        name=f"V2.8 {profile_id} p95 authority",
        expected_sha256=source["authority"]["file_sha256"],
    )
    _, _, projection_value = _strict_file(
        root,
        projection_relative,
        name=f"V2.8 {profile_id} p95 projection",
        expected_sha256=source["projection"]["file_sha256"],
    )
    _verify_self_hashed(
        authority,
        schema_version=source["authority"]["schema_version"],
        name=f"V2.8 {profile_id} p95 authority",
        expected_content_sha256=source["authority"]["content_sha256"],
    )
    _verify_self_hashed(
        projection_value,
        schema_version=source["projection"]["schema_version"],
        name=f"V2.8 {profile_id} p95 projection",
        expected_content_sha256=source["projection"]["content_sha256"],
    )
    runtime_model = source["runtime_model"]
    served_model = source["served_model"]
    if (
        set(reservations) != {runtime_model}
        or set(reservations[runtime_model]) != {"action", "semantic"}
        or projection != projection_value
        or any(
            reservations[runtime_model][call_kind]["reservation"]
            != projection["projection"][f"{served_model}::{call_kind}"]
            for call_kind in ("action", "semantic")
        )
    ):
        raise PilotV29Stage0ImportError(
            f"V2.8 {profile_id} p95 receipt/projection differ"
        )
    return {
        "profile_id": profile_id,
        "source_contract_id": V28_CONTRACT_ID,
        "source_contract_sha256": V28_CONTRACT_CANONICAL_SHA256,
        "source_git_commit": V28_SCIENCE_COMMIT,
        "source_git_tag": V28_SCIENCE_TAG,
        "authority": {
            "path": authority_relative.as_posix(),
            **_json_copy(source["authority"]),
        },
        "projection": {
            "path": projection_relative.as_posix(),
            **_json_copy(source["projection"]),
        },
        "runtime_model": runtime_model,
        "served_model": served_model,
        "reservations": _json_copy(reservations),
    }


def imported_v28_raw_root_v29(child_raw_root: str | Path) -> Path:
    """Return the child-local root of the exact immutable V2.8 snapshot."""

    return Path(child_raw_root).joinpath(*V29_SNAPSHOT_RELATIVE.parts)


def snapshot_path_for_v28_source_artifact_v29(
    child_raw_root: str | Path,
    source_artifact_path: str,
) -> Path:
    """Map one V2.8 artifact path into V2.9's exact parent snapshot."""

    try:
        relative = _normalized_relative(
            source_artifact_path,
            required_top="experiment_results",
            name="V2.8 source artifact path",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if (
        tuple(relative.parts[: len(V28_RAW_ROOT.parts)])
        != V28_RAW_ROOT.parts
        or len(relative.parts) <= len(V28_RAW_ROOT.parts)
    ):
        raise PilotV29Stage0ImportError(
            "source artifact is outside the V2.8 raw namespace"
        )
    inside = PurePosixPath(*relative.parts[len(V28_RAW_ROOT.parts) :])
    return imported_v28_raw_root_v29(child_raw_root).joinpath(*inside.parts)


def v29_observed_p95_receipt_path(
    child_raw_root: str | Path,
    profile_id: str,
) -> Path:
    if profile_id not in V29_ALLOWED_P95_PROFILES:
        raise PilotV29Stage0ImportError(
            f"unsupported V2.8 p95 profile: {profile_id}"
        )
    return (
        imported_v28_raw_root_v29(child_raw_root)
        / "parent-import"
        / "observed_p95"
        / profile_id
        / "observed_p95_authority_receipt.json"
    )


def v29_observed_p95_projection_path(
    child_raw_root: str | Path,
    profile_id: str,
) -> Path:
    return v29_observed_p95_receipt_path(
        child_raw_root,
        profile_id,
    ).with_name("projection_p95.json")


def verify_v29_imported_v28_observed_p95(
    child_raw_root: str | Path,
    profile_id: str,
    *,
    expected_parent_commit: str = V28_SCIENCE_COMMIT,
) -> dict[str, Any]:
    """Verify the copied V2.8 p95 pair without relabelling its authority."""

    if (
        expected_parent_commit != V28_SCIENCE_COMMIT
        or profile_id not in V29_ALLOWED_P95_PROFILES
    ):
        raise PilotV29Stage0ImportError(
            "V2.8 copied p95 identity is unsupported"
        )
    snapshot = imported_v28_raw_root_v29(child_raw_root)
    rows, summary = _inventory(snapshot, declared_root=V28_RAW_ROOT)
    if (
        summary.get("file_count") != V28_RAW_FILE_COUNT
        or summary.get("storage_bytes") != V28_RAW_STORAGE_BYTES
        or summary.get("inventory_sha256") != V28_RAW_INVENTORY_SHA256
    ):
        raise PilotV29Stage0ImportError(
            "copied V2.8 raw-tree inventory drifted"
        )
    del rows
    source = _V28_P95_SOURCES[profile_id]
    receipt = v29_observed_p95_receipt_path(child_raw_root, profile_id)
    projection = v29_observed_p95_projection_path(child_raw_root, profile_id)
    try:
        receipt_relative = PurePosixPath(
            receipt.relative_to(snapshot).as_posix()
        )
        projection_relative = PurePosixPath(
            projection.relative_to(snapshot).as_posix()
        )
    except ValueError as exc:
        raise PilotV29Stage0ImportError(
            "copied V2.8 p95 path escaped the snapshot"
        ) from exc
    _, _, receipt_value = _strict_file(
        snapshot,
        receipt_relative,
        name=f"copied V2.8 {profile_id} p95 authority",
        expected_sha256=source["authority"]["file_sha256"],
    )
    _, _, projection_value = _strict_file(
        snapshot,
        projection_relative,
        name=f"copied V2.8 {profile_id} p95 projection",
        expected_sha256=source["projection"]["file_sha256"],
    )
    _verify_self_hashed(
        receipt_value,
        schema_version=source["authority"]["schema_version"],
        name=f"copied V2.8 {profile_id} p95 authority",
        expected_content_sha256=source["authority"]["content_sha256"],
    )
    _verify_self_hashed(
        projection_value,
        schema_version=source["projection"]["schema_version"],
        name=f"copied V2.8 {profile_id} p95 projection",
        expected_content_sha256=source["projection"]["content_sha256"],
    )
    model = receipt_value.get("model")
    git = receipt_value.get("git")
    runtime_model = source["runtime_model"]
    served_model = source["served_model"]
    reservations = receipt_value.get("reservations")
    if (
        model
        != {
            "model_id": profile_id,
            "runtime_model": runtime_model,
            "served_model": served_model,
        }
        or git
        != {"commit": V28_SCIENCE_COMMIT, "tag": V28_SCIENCE_TAG}
        or receipt_value.get("contract", {}).get("contract_sha256")
        != V28_CONTRACT_CANONICAL_SHA256
        or receipt_value.get("scientific_evidence") is not False
        or not isinstance(reservations, Mapping)
        or any(
            reservations[runtime_model][call_kind]["reservation"]
            != projection_value["projection"][
                f"{served_model}::{call_kind}"
            ]
            for call_kind in ("action", "semantic")
        )
    ):
        raise PilotV29Stage0ImportError(
            f"copied V2.8 {profile_id} p95 semantics drifted"
        )
    return {
        "profile_id": profile_id,
        "source_contract_id": V28_CONTRACT_ID,
        "source_contract_sha256": V28_CONTRACT_CANONICAL_SHA256,
        "source_git_commit": V28_SCIENCE_COMMIT,
        "source_git_tag": V28_SCIENCE_TAG,
        "authority": {
            "path": (
                V29_RAW_ROOT
                / V29_SNAPSHOT_RELATIVE
                / receipt_relative
            ).as_posix(),
            **_json_copy(source["authority"]),
        },
        "projection": {
            "path": (
                V29_RAW_ROOT
                / V29_SNAPSHOT_RELATIVE
                / projection_relative
            ).as_posix(),
            **_json_copy(source["projection"]),
        },
        "runtime_model": runtime_model,
        "served_model": served_model,
        "reservations": _json_copy(reservations),
    }


def _normalized_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_copy(value)
    result.pop("contract_id", None)
    result.pop("run_id", None)
    return result


def _v28_path_for_v27_artifact(path: str) -> PurePosixPath:
    try:
        relative = _normalized_relative(
            path,
            required_top="experiment_results",
            name="nested V2.7 artifact path",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    prefix = v28.V27_RAW_ROOT
    if (
        tuple(relative.parts[: len(prefix.parts)]) != prefix.parts
        or len(relative.parts) <= len(prefix.parts)
    ):
        raise PilotV29Stage0ImportError(
            "nested Stage-0 artifact escaped the V2.7 namespace"
        )
    inside = PurePosixPath(*relative.parts[len(prefix.parts) :])
    return V28_RAW_ROOT / v28.V28_SNAPSHOT_RELATIVE / inside


def _verify_nested_v26_stage0_receipt(
    parent_root: Path,
) -> dict[str, Any]:
    relative = (
        V28_RAW_ROOT
        / V26_NESTED_IN_V28_RELATIVE
        / "stage0-calibration/stage_receipt.json"
    )
    _, raw, value = _strict_file(
        parent_root,
        relative,
        name="V2.8-nested V2.6 Stage-0 receipt",
        expected_sha256=V26_STAGE0_RECEIPT_FILE_SHA256,
    )
    try:
        v28._verify_v2_stage_receipt_self_hash(
            value,
            name="V2.8-nested V2.6 Stage-0 receipt",
        )
    except v28.PilotV28Stage0ImportError as exc:
        raise _translate(exc) from exc
    if (
        value.get("integrity", {}).get("content_sha256")
        != V26_STAGE0_RECEIPT_CONTENT_SHA256
        or value.get("status") != "complete-with-no-go"
        or value.get("registered_run_count") != 14
        or value.get("complete_cell_count") != 14
        or value.get("go") is not False
        or value.get("execution_progression_go") is not False
    ):
        raise PilotV29Stage0ImportError(
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


def _verify_imported_cells(
    parent_root: Path,
    *,
    source_contract: PilotContract,
    target_contract: PilotContract,
    v28_source_manifest: Mapping[str, Any],
    v28_run_ledger: Mapping[str, Any],
    parent_receipts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source_parent_specs = source_contract.expand(stage="parent-import")
    target_parent_specs = target_contract.expand(stage="parent-import")
    target_stage0_specs = target_contract.expand(stage="stage0-calibration")
    if (
        len(source_parent_specs) != 1
        or len(target_parent_specs) != 1
        or len(target_stage0_specs) != 14
    ):
        raise PilotV29Stage0ImportError(
            "V2.9 imported cell denominator drifted"
        )
    source_parent = source_parent_specs[0]
    target_parent = target_parent_specs[0]
    if _normalized_spec(source_parent.to_dict()) != _normalized_spec(
        target_parent.to_dict()
    ):
        raise PilotV29Stage0ImportError(
            "V2.8/V2.9 parent prerequisite specs differ"
        )
    ledger_rows = _mapping(
        v28_run_ledger.get("runs"),
        name="V2.8 run ledger rows",
    )
    parent_ledger = _mapping(
        ledger_rows.get(source_parent.run_id),
        name="V2.8 parent prerequisite ledger row",
    )
    if (
        parent_ledger.get("status") != "complete"
        or parent_ledger.get("spec") != source_parent.to_dict()
    ):
        raise PilotV29Stage0ImportError(
            "V2.8 parent prerequisite is not complete"
        )
    result: list[dict[str, Any]] = [
        {
            "stage_id": "parent-import",
            "source_authority_contract_id": V28_CONTRACT_ID,
            "source_run_id": source_parent.run_id,
            "target_run_id": target_parent.run_id,
            "source_spec": source_parent.to_dict(),
            "target_spec": target_parent.to_dict(),
            "source_artifacts": _json_copy(parent_receipts),
        }
    ]

    target_by_key = {
        json.dumps(
            _normalized_spec(spec.to_dict()),
            sort_keys=True,
            allow_nan=False,
        ): spec
        for spec in target_stage0_specs
    }
    source_rows = [
        row
        for row in v28_source_manifest["imported_complete_cells"]
        if row.get("stage_id") == "stage0-calibration"
    ]
    if len(source_rows) != 14 or len(target_by_key) != 14:
        raise PilotV29Stage0ImportError(
            "nested V2.6 Stage-0 source count drifted"
        )
    for source_row in source_rows:
        source_spec = _mapping(
            source_row.get("source_spec"),
            name="nested V2.6 source spec",
        )
        key = json.dumps(
            _normalized_spec(source_spec),
            sort_keys=True,
            allow_nan=False,
        )
        target_spec = target_by_key.pop(key, None)
        if target_spec is None:
            raise PilotV29Stage0ImportError(
                "nested V2.6/V2.9 Stage-0 matrix differs"
            )
        source_artifacts = _mapping(
            source_row.get("source_artifacts"),
            name="nested V2.6 source artifacts",
        )
        physical: dict[str, Any] = {
            "run_root": _v28_path_for_v27_artifact(
                str(source_artifacts["run_root"])
            ).as_posix()
        }
        for artifact_name in ("config", "manifest", "actor_journal"):
            prior = _mapping(
                source_artifacts.get(artifact_name),
                name=f"nested V2.6 {artifact_name}",
            )
            relative = _v28_path_for_v27_artifact(str(prior.get("path")))
            binding = _artifact_binding(
                parent_root,
                relative,
                expected_sha256=str(prior.get("file_sha256")),
                name=f"nested V2.6 Stage-0 {artifact_name}",
            )
            if binding["byte_size"] != prior.get("byte_size"):
                raise PilotV29Stage0ImportError(
                    f"nested V2.6 {artifact_name} size drifted"
                )
            physical[artifact_name] = binding
        run_relative = PurePosixPath(physical["run_root"])
        _, _, config = _strict_file(
            parent_root,
            run_relative / "config.json",
            name="nested V2.6 Stage-0 config",
        )
        source_run_id = str(source_row.get("source_run_id"))
        if (
            run_relative.name != source_run_id
            or config.get("run_id") != source_run_id
        ):
            raise PilotV29Stage0ImportError(
                "nested V2.6 Stage-0 config.run_id drifted"
            )
        if not verify_manifest(
            parent_root.joinpath(*run_relative.parts)
        ).valid:
            raise PilotV29Stage0ImportError(
                "nested V2.6 Stage-0 runner manifest is invalid"
            )
        try:
            journal = verify_provider_call_journal(
                parent_root.joinpath(
                    *PurePosixPath(
                        physical["actor_journal"]["path"]
                    ).parts
                ),
                expected_run_id=source_run_id,
                expected_contract_hash=V26_CONTRACT_CANONICAL_SHA256,
                require_terminal_dispositions=True,
            )
        except Exception as exc:
            raise PilotV29Stage0ImportError(
                f"nested V2.6 Stage-0 journal failed verification: {exc}"
            ) from exc
        events = journal.get("events")
        if (
            not isinstance(events, list)
            or len(events) != 96
            or Counter(event.get("event_type") for event in events)
            != Counter({"completion_received": 48, "parse_disposition": 48})
        ):
            raise PilotV29Stage0ImportError(
                "nested V2.6 Stage-0 journal is incomplete"
            )
        result.append(
            {
                "stage_id": "stage0-calibration",
                "source_authority_contract_id": V28_CONTRACT_ID,
                "physical_source_contract_id": V26_CONTRACT_ID,
                "source_run_id": source_run_id,
                "target_run_id": target_spec.run_id,
                "source_spec": _json_copy(source_spec),
                "target_spec": target_spec.to_dict(),
                "source_artifacts": physical,
            }
        )
    if (
        target_by_key
        or len(result) != 15
        or Counter(row["stage_id"] for row in result)
        != Counter({"parent-import": 1, "stage0-calibration": 14})
        or {
            row["source_spec"]["utility_profile_id"]
            for row in result
            if row["stage_id"] == "stage0-calibration"
        }
        != set(_STAGE0_PROFILES)
        or {
            row["source_spec"]["environment_seed"]
            for row in result
            if row["stage_id"] == "stage0-calibration"
        }
        != set(_STAGE0_SEEDS)
    ):
        raise PilotV29Stage0ImportError(
            "V2.9 imported cell inventory drifted"
        )
    return sorted(result, key=lambda row: row["target_run_id"])


def _audit_v28_source(
    *,
    parent_repo_root: str | Path,
    child_repo_root: str | Path,
    target_contract: PilotContract,
) -> dict[str, Any]:
    parent_root = _strict_real_root(
        parent_repo_root,
        name="V2.8 source repository",
    )
    child_root = _strict_real_root(
        child_repo_root,
        name="V2.9 child repository",
    )
    _validate_target_contract(target_contract, require_frozen=False)
    _verify_v28_release_identity(parent_root)
    source_contract = _load_verified_v28_contract(parent_root)
    source_manifest = _load_verified_v28_source_manifest(parent_root)
    raw_root = parent_root.joinpath(*V28_RAW_ROOT.parts)
    inventory = _verify_exact_v28_inventory(raw_root)
    _verify_exact_nested_v26_inventory(raw_root)
    run_ledger, budget_ledger = _verify_v28_ledgers(
        raw_root,
        source_contract,
    )
    parent_receipts = _verify_v28_parent_receipts(raw_root)
    nested_stage0_receipt = _verify_nested_v26_stage0_receipt(parent_root)
    qref = _verify_v28_qref_failure(
        parent_root,
        raw_root,
        source_contract,
        source_manifest,
    )
    evidence = _verify_v28_published_evidence(child_root)
    p95 = {
        profile_id: v2_8_p95_source_binding_v29(
            parent_root,
            profile_id,
        )
        for profile_id in V29_ALLOWED_P95_PROFILES
    }
    imported = _verify_imported_cells(
        parent_root,
        source_contract=source_contract,
        target_contract=target_contract,
        v28_source_manifest=source_manifest,
        v28_run_ledger=run_ledger,
        parent_receipts=parent_receipts,
    )
    release_attestation = _artifact_binding(
        parent_root,
        V28_RAW_ROOT / "release_attestation.json",
        expected_sha256=V28_RELEASE_ATTESTATION_FILE_SHA256,
        name="V2.8 release attestation",
    )
    return {
        "parent_root": parent_root,
        "child_root": child_root,
        "source_contract": source_contract,
        "source_manifest": source_manifest,
        "inventory": inventory,
        "run_ledger": run_ledger,
        "budget_ledger": budget_ledger,
        "parent_receipts": parent_receipts,
        "nested_stage0_receipt": nested_stage0_receipt,
        "qref": qref,
        "evidence": evidence,
        "p95": p95,
        "imported": imported,
        "release_attestation": release_attestation,
    }


def build_v29_source_manifest(
    *,
    parent_repo_root: str | Path,
    child_repo_root: str | Path,
    target_contract: PilotContract,
) -> dict[str, Any]:
    """Verify frozen V2.8 and build the deterministic V2.9 manifest."""

    audit = _audit_v28_source(
        parent_repo_root=parent_repo_root,
        child_repo_root=child_repo_root,
        target_contract=target_contract,
    )
    return _seal(
        {
            "schema_version": V29_SOURCE_MANIFEST_SCHEMA_VERSION,
            "v2_8_terminal_parent": {
                "contract": {
                    "contract_id": V28_CONTRACT_ID,
                    "path": V28_CONTRACT_PATH.as_posix(),
                    "schema_version": "finevo-pilot-contract-v2",
                    "status": "frozen",
                    "file_sha256": V28_CONTRACT_FILE_SHA256,
                    "canonical_sha256":
                    V28_CONTRACT_CANONICAL_SHA256,
                },
                "source_manifest": {
                    "path": V28_SOURCE_MANIFEST_PATH.as_posix(),
                    "file_sha256": V28_SOURCE_MANIFEST_FILE_SHA256,
                    "content_sha256":
                    V28_SOURCE_MANIFEST_CONTENT_SHA256,
                },
                "release": {
                    "science_tag": V28_SCIENCE_TAG,
                    "science_tag_object": V28_SCIENCE_TAG_OBJECT,
                    "science_commit": V28_SCIENCE_COMMIT,
                    "tag_kind": "annotated",
                    "raw_root": V28_RAW_ROOT.as_posix(),
                    "release_attestation":
                    audit["release_attestation"],
                },
                "raw_snapshot": {
                    "root": V28_RAW_ROOT.as_posix(),
                    "inventory_schema_version":
                    V28_RAW_INVENTORY_SCHEMA_VERSION,
                    "inventory_canonicalization":
                    V28_RAW_INVENTORY_CANONICALIZATION,
                    "file_count": V28_RAW_FILE_COUNT,
                    "storage_bytes": V28_RAW_STORAGE_BYTES,
                    "inventory_sha256": V28_RAW_INVENTORY_SHA256,
                },
                "ledgers": {
                    "run": {
                        "path": (
                            V28_RAW_ROOT / "run_ledger.json"
                        ).as_posix(),
                        "file_sha256": V28_RUN_LEDGER_FILE_SHA256,
                        "internal_sha256":
                        V28_RUN_LEDGER_INTERNAL_SHA256,
                        "event_count": V28_RUN_LEDGER_EVENT_COUNT,
                        "event_chain_head": V28_RUN_LEDGER_EVENT_HEAD,
                    },
                    "budget": {
                        "path": (
                            V28_RAW_ROOT / "budget_ledger.json"
                        ).as_posix(),
                        "file_sha256": V28_BUDGET_LEDGER_FILE_SHA256,
                        "internal_sha256":
                        V28_BUDGET_LEDGER_INTERNAL_SHA256,
                        "event_count": V28_BUDGET_LEDGER_EVENT_COUNT,
                        "event_chain_head":
                        V28_BUDGET_LEDGER_EVENT_HEAD,
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
                        "failed": 1,
                        "integrity-stopped": 209,
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
                "parent_import_receipts":
                audit["parent_receipts"],
                "q_ref_failed_prerequisite":
                audit["qref"]["failed_prerequisite"],
            },
            "published_v2_8_evidence": audit["evidence"],
            "nested_v2_6_stage0_source": {
                "contract": {
                    "contract_id": V26_CONTRACT_ID,
                    "canonical_sha256":
                    V26_CONTRACT_CANONICAL_SHA256,
                },
                "physical_snapshot_root": (
                    V28_RAW_ROOT / V26_NESTED_IN_V28_RELATIVE
                ).as_posix(),
                "inventory": {
                    "file_count": V26_RAW_FILE_COUNT,
                    "storage_bytes": V26_RAW_STORAGE_BYTES,
                    "inventory_sha256": V26_RAW_INVENTORY_SHA256,
                },
                "stage0_receipt": audit["nested_stage0_receipt"],
                "source_via_v2_8_exact_snapshot": True,
            },
            "q_ref_audit_reference": audit["qref"],
            "v2_8_p95_sources_for_child_reseal": audit["p95"],
            "imported_complete_cells": audit["imported"],
            "cumulative_budget_debit": V29_CUMULATIVE_DEBIT.to_dict(),
            "import_policy": {
                "source_raw_namespace": V28_RAW_ROOT.as_posix(),
                "child_raw_namespace": V29_RAW_ROOT.as_posix(),
                "child_snapshot_namespace": (
                    V29_RAW_ROOT / V29_SNAPSHOT_RELATIVE
                ).as_posix(),
                "exact_full_v2_8_raw_snapshot_copy": True,
                "imported_cell_count": 15,
                "imported_cell_breakdown": {
                    "parent-import": 1,
                    "stage0-calibration": 14,
                },
                "q_ref_imported": False,
                "q_ref_fresh_zero_hosted_provider_regeneration_required":
                True,
                "q_ref_scripted_diagnostic_calls": 48,
                "q_ref_hosted_provider_calls": 0,
                "q_ref_hosted_cost_usd": 0.0,
                "source_manifests_rewritten": False,
                "source_journals_rewritten": False,
                "provider_construction_during_import": False,
                "provider_redispatch_for_imported_cells": "forbidden",
                "v2_8_no_go_preserved": True,
                "v2_8_terminal_rows_reclassified": False,
                "scientific_evidence": False,
            },
            "observation_boundary": {
                "v2_8_q_ref_summary_mismatch_observed_before_amendment":
                True,
                "v2_8_q_ref_failure_observed_before_amendment": True,
                "stage0_calibration_selection_observed_before_amendment":
                True,
                "stage0_guardrail_outputs_may_have_been_inspected": True,
                "stage0_candidate_winner_may_have_been_observed": True,
                "a_d_treatment_effect_outcomes_generated": False,
                "a_d_treatment_effect_outcomes_observed": False,
                "amendment_is_outcome_blind_with_respect_to_a_d_effects":
                True,
            },
        }
    )


def _validate_source_manifest_structure(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = _json_copy(value)
    _verify_self_hashed(
        candidate,
        schema_version=V29_SOURCE_MANIFEST_SCHEMA_VERSION,
        name="V2.9 source manifest",
    )
    if set(candidate) != {
        "schema_version",
        "v2_8_terminal_parent",
        "published_v2_8_evidence",
        "nested_v2_6_stage0_source",
        "q_ref_audit_reference",
        "v2_8_p95_sources_for_child_reseal",
        "imported_complete_cells",
        "cumulative_budget_debit",
        "import_policy",
        "observation_boundary",
        "integrity",
    }:
        raise PilotV29Stage0ImportError(
            "V2.9 source manifest fields differ from schema"
        )
    rows = candidate.get("imported_complete_cells")
    policy = _mapping(
        candidate.get("import_policy"),
        name="V2.9 import policy",
    )
    qref = _mapping(
        candidate.get("q_ref_audit_reference"),
        name="V2.9 q-ref audit reference",
    )
    parent = _mapping(
        candidate.get("v2_8_terminal_parent"),
        name="V2.8 terminal parent",
    )
    raw = _mapping(parent.get("raw_snapshot"), name="V2.8 raw snapshot")
    try:
        debit = ParentBudgetDebit.from_dict(
            candidate["cumulative_budget_debit"]
        )
    except (TypeError, ValueError) as exc:
        raise PilotV29Stage0ImportError(
            "V2.9 cumulative parent debit is invalid"
        ) from exc
    if (
        not isinstance(rows, list)
        or len(rows) != 15
        or Counter(str(row.get("stage_id")) for row in rows)
        != Counter({"parent-import": 1, "stage0-calibration": 14})
        or any(row.get("stage_id") == "q-ref-resolution" for row in rows)
        or qref.get("imported") is not False
        or qref.get("source_result_reuse") != "forbidden"
        or qref.get("fresh_v2_9_policy")
        != {
            "scripted_diagnostic_calls": 48,
            "hosted_provider_calls": 0,
            "hosted_cost_usd": 0.0,
            "provider_construction_during_regeneration": False,
            "source_result_reuse": "forbidden",
        }
        or policy.get("provider_construction_during_import") is not False
        or policy.get("q_ref_imported") is not False
        or policy.get("v2_8_no_go_preserved") is not True
        or policy.get("v2_8_terminal_rows_reclassified") is not False
        or raw.get("file_count") != V28_RAW_FILE_COUNT
        or raw.get("storage_bytes") != V28_RAW_STORAGE_BYTES
        or raw.get("inventory_sha256") != V28_RAW_INVENTORY_SHA256
        or debit != V29_CUMULATIVE_DEBIT
        or set(candidate["v2_8_p95_sources_for_child_reseal"])
        != set(V29_ALLOWED_P95_PROFILES)
    ):
        raise PilotV29Stage0ImportError(
            "V2.9 source manifest authority boundary drifted"
        )
    return candidate


def validate_v29_source_manifest(
    manifest: Mapping[str, Any],
    *,
    parent_repo_root: str | Path,
    child_repo_root: str | Path,
    target_contract: PilotContract,
) -> dict[str, Any]:
    """Rebuild and compare a proposed V2.9 source manifest exactly."""

    candidate = _validate_source_manifest_structure(manifest)
    expected = build_v29_source_manifest(
        parent_repo_root=parent_repo_root,
        child_repo_root=child_repo_root,
        target_contract=target_contract,
    )
    if candidate != expected:
        raise PilotV29Stage0ImportError(
            "V2.9 source manifest differs from verified V2.8 authority"
        )
    return candidate


def write_v29_source_manifest_draft(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Write canonical immutable JSON; this does not freeze the contract."""

    candidate = _validate_source_manifest_structure(manifest)
    target = Path(path).absolute()
    raw = (
        json.dumps(
            candidate,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        _atomic_exact_bytes_no_follow(
            repo_root=target.parent,
            path=target,
            raw=raw,
        )
    except (PilotV27Stage0ImportError, PilotV24ParentImportError) as exc:
        raise _translate(exc) from exc
    return target


def load_v29_source_manifest(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
    expected_content_sha256: str | None = None,
) -> dict[str, Any]:
    """Load a canonical V2.9 manifest through a no-symlink guarded read."""

    target = Path(path).absolute()
    try:
        root = _strict_real_root(
            target.parent,
            name="V2.9 manifest directory",
        )
        _, raw = _guarded_file(
            root,
            PurePosixPath(target.name),
            name="V2.9 source manifest",
        )
        value = _strict_json(raw, name="V2.9 source manifest")
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if (
        expected_file_sha256 is not None
        and _sha256(raw) != expected_file_sha256
    ):
        raise PilotV29Stage0ImportError(
            "V2.9 source manifest file hash drifted"
        )
    candidate = _validate_source_manifest_structure(value)
    if (
        expected_content_sha256 is not None
        and candidate["integrity"]["content_sha256"]
        != expected_content_sha256
    ):
        raise PilotV29Stage0ImportError(
            "V2.9 source manifest content hash drifted"
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
        raise PilotV29Stage0ImportError(
            "V2.9 source manifest is not canonical pretty JSON"
        )
    return candidate


def source_binding_for_target_v29(
    source_manifest: Mapping[str, Any],
    target_spec: PilotRunSpec | Mapping[str, Any] | str,
) -> dict[str, Any]:
    """Return the unique imported parent/Stage-0 source for a V2.9 cell."""

    value = _validate_source_manifest_structure(source_manifest)
    target_run_id = (
        target_spec
        if isinstance(target_spec, str)
        else (
            target_spec.run_id
            if isinstance(target_spec, PilotRunSpec)
            else target_spec.get("run_id")
        )
    )
    rows = [
        row
        for row in value["imported_complete_cells"]
        if row.get("target_run_id") == target_run_id
    ]
    if len(rows) != 1:
        raise PilotV29Stage0ImportError(
            "target has no unique imported V2.8/V2.6 source binding"
        )
    row = _json_copy(rows[0])
    if row["stage_id"] not in {"parent-import", "stage0-calibration"}:
        raise PilotV29Stage0ImportError(
            "only parent/Stage-0 targets may use imported sources"
        )
    if not isinstance(target_spec, str):
        selected = (
            target_spec.to_dict()
            if isinstance(target_spec, PilotRunSpec)
            else _json_copy(target_spec)
        )
        if selected != row["target_spec"]:
            raise PilotV29Stage0ImportError(
                "target spec differs from source-manifest binding"
            )
    return row


def q_ref_audit_reference_v29(
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the immutable failed V2.8 q-ref audit reference."""

    value = _validate_source_manifest_structure(source_manifest)
    return _json_copy(value["q_ref_audit_reference"])


def imported_v26_run_dir_v29(
    raw_root: str | Path,
    v29_spec: PilotRunSpec | Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> Path:
    """Resolve an imported V2.6 Stage-0 run inside the V2.8 snapshot."""

    binding = source_binding_for_target_v29(
        source_manifest,
        v29_spec,
    )
    if binding["stage_id"] != "stage0-calibration":
        raise PilotV29Stage0ImportError(
            "target is not a nested V2.6 Stage-0 runner directory"
        )
    run_root = binding["source_artifacts"].get("run_root")
    if not isinstance(run_root, str):
        raise PilotV29Stage0ImportError(
            "Stage-0 target lacks an imported runner directory"
        )
    return snapshot_path_for_v28_source_artifact_v29(raw_root, run_root)


def parent_budget_debit_for_v29(
    contract: PilotContract,
) -> ParentBudgetDebit | None:
    """Return the exact cumulative V2.8 debit inherited by V2.9."""

    if contract.contract_id != V29_CONTRACT_ID:
        return None
    _validate_target_contract(contract, require_frozen=False)
    amendment = _mapping(
        contract.qref_summary_equivalence_amendment,
        name="V2.9 q-ref equivalence amendment",
    )
    carry = _mapping(
        amendment.get("budget_carry_forward"),
        name="V2.9 budget carry-forward",
    )
    expected = V29_CUMULATIVE_DEBIT.to_dict()
    expected.pop("schema_version")
    if (
        carry.get("cumulative_prior") != expected
        or carry.get("budget_reset") is not False
        or carry.get("debit_before_new_dispatch") is not True
    ):
        raise PilotV29Stage0ImportError(
            "V2.9 cumulative parent debit drifted"
        )
    return V29_CUMULATIVE_DEBIT


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
                name="V2.8 raw inventory path",
            )
            _, raw = _guarded_file(
                source_root,
                relative,
                name=f"V2.8 raw {relative.as_posix()}",
            )
            _atomic_exact_bytes_no_follow(
                repo_root=destination_guard_root,
                path=destination_root.joinpath(*relative.parts),
                raw=raw,
            )
        except (PilotV24ParentImportError, PilotV27Stage0ImportError) as exc:
            raise _translate(exc) from exc
        if (
            len(raw) != row.get("byte_size")
            or _sha256(raw) != row.get("sha256")
        ):
            raise PilotV29Stage0ImportError(
                f"V2.8 raw source changed during copy: "
                f"{relative.as_posix()}"
            )
    copied_rows, copied = _inventory(
        destination_root,
        declared_root=V28_RAW_ROOT,
    )
    canonical_source = json.dumps(
        list(inventory),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    canonical_copied = json.dumps(
        copied_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if (
        canonical_source != canonical_copied
        or copied.get("file_count") != V28_RAW_FILE_COUNT
        or copied.get("storage_bytes") != V28_RAW_STORAGE_BYTES
        or copied.get("inventory_sha256") != V28_RAW_INVENTORY_SHA256
    ):
        raise PilotV29Stage0ImportError(
            "V2.9 copied V2.8 raw snapshot differs from source"
        )


def _child_contract_binding(
    child_root: Path,
    contract: PilotContract,
) -> dict[str, Any]:
    _, raw, value = _strict_file(
        child_root,
        V29_EXPANDED_CONTRACT_PATH,
        name="expanded V2.9 contract",
    )
    parsed = PilotContract.from_dict(value)
    if (
        parsed.contract_id != contract.contract_id
        or parsed.canonical_hash != contract.canonical_hash
        or parsed.to_dict() != contract.to_dict()
    ):
        raise PilotV29Stage0ImportError(
            "tracked V2.9 contract differs from selected contract"
        )
    return {
        "path": V29_EXPANDED_CONTRACT_PATH.as_posix(),
        "file_sha256": _sha256(raw),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
    }


def _tracked_source_manifest_binding(
    child_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _, raw, value = _strict_file(
        child_root,
        V29_SOURCE_MANIFEST_PATH,
        name="tracked V2.9 source manifest",
    )
    candidate = _validate_source_manifest_structure(value)
    if candidate != _json_copy(manifest):
        raise PilotV29Stage0ImportError(
            "tracked V2.9 source manifest differs from selected manifest"
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
        raise PilotV29Stage0ImportError(
            "tracked V2.9 source manifest is not canonical JSON"
        )
    return {
        "path": V29_SOURCE_MANIFEST_PATH.as_posix(),
        "file_sha256": _sha256(raw),
        "content_sha256": candidate["integrity"]["content_sha256"],
    }


def _verify_child_release_identity(
    child_root: Path,
    *,
    expected_git_commit: str,
) -> None:
    if _COMMIT_RE.fullmatch(expected_git_commit) is None:
        raise PilotV29Stage0ImportError(
            "V2.9 child git commit is malformed"
        )
    try:
        head = _git(child_root, "rev-parse", "--verify", "HEAD^{commit}")
        tag_object = _git(
            child_root,
            "rev-parse",
            "--verify",
            f"refs/tags/{V29_SCIENCE_TAG}^{{tag}}",
        )
        tag_commit = _git(
            child_root,
            "rev-parse",
            "--verify",
            f"refs/tags/{V29_SCIENCE_TAG}^{{commit}}",
        )
        tracked_status = _git(
            child_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        )
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if (
        head != expected_git_commit
        or tag_commit != expected_git_commit
        or not tag_object
        or tag_object == tag_commit
        or tracked_status
    ):
        raise PilotV29Stage0ImportError(
            "V2.9 child release/tag identity drifted"
        )


def _build_v29_parent_import_receipt(
    *,
    child_root: Path,
    child_raw: Path,
    contract: PilotContract,
    child_git_commit: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = imported_v28_raw_root_v29(child_raw)
    _, snapshot_summary = _inventory(
        snapshot,
        declared_root=V28_RAW_ROOT,
    )
    expected_summary = {
        "root": V28_RAW_ROOT.as_posix(),
        "inventory_schema_version": V28_RAW_INVENTORY_SCHEMA_VERSION,
        "inventory_canonicalization":
        V28_RAW_INVENTORY_CANONICALIZATION,
        "file_count": V28_RAW_FILE_COUNT,
        "storage_bytes": V28_RAW_STORAGE_BYTES,
        "inventory_sha256": V28_RAW_INVENTORY_SHA256,
    }
    if snapshot_summary != expected_summary:
        raise PilotV29Stage0ImportError(
            "V2.9 parent snapshot inventory drifted"
        )
    p95 = {
        profile_id: verify_v29_imported_v28_observed_p95(
            child_raw,
            profile_id,
        )
        for profile_id in V29_ALLOWED_P95_PROFILES
    }
    return _seal(
        {
            "schema_version": V29_PARENT_IMPORT_SCHEMA_VERSION,
            "contract": _child_contract_binding(child_root, contract),
            "git": {
                "tag": V29_SCIENCE_TAG,
                "commit": child_git_commit,
            },
            "source_manifest":
            _tracked_source_manifest_binding(child_root, manifest),
            "source_parent": {
                "contract_id": V28_CONTRACT_ID,
                "contract_sha256": V28_CONTRACT_CANONICAL_SHA256,
                "science_tag": V28_SCIENCE_TAG,
                "science_commit": V28_SCIENCE_COMMIT,
                "raw_root": V28_RAW_ROOT.as_posix(),
                "terminal_status": "complete-with-no-go",
                "status_counts": {
                    "complete": 1,
                    "failed": 1,
                    "integrity-stopped": 209,
                },
                "scientific_complete": False,
            },
            "copied_snapshot": {
                "path": (
                    V29_RAW_ROOT / V29_SNAPSHOT_RELATIVE
                ).as_posix(),
                **snapshot_summary,
            },
            "nested_v2_6_stage0": {
                "physical_source_contract_id": V26_CONTRACT_ID,
                "physical_source_contract_sha256":
                V26_CONTRACT_CANONICAL_SHA256,
                "imported_complete_cells": 14,
                "inventory_sha256": V26_RAW_INVENTORY_SHA256,
            },
            "imported_cells": {
                "count": 15,
                "breakdown": {
                    "parent-import": 1,
                    "stage0-calibration": 14,
                },
            },
            "q_ref": {
                "imported": False,
                "source_result_reuse": "forbidden",
                "source_failed_prerequisite_verified": True,
                "fresh_v2_9_regeneration_required": True,
                "scripted_diagnostic_calls": 48,
                "hosted_provider_calls": 0,
                "hosted_cost_usd": 0.0,
            },
            "p95_sources": p95,
            "cumulative_budget_debit": V29_CUMULATIVE_DEBIT.to_dict(),
            "scripted_diagnostic_calls_during_import": 0,
            "hosted_provider_calls_during_import": 0,
            "provider_calls_during_import": 0,
            "provider_construction_during_import": False,
            "scientific_evidence": False,
            "evidence_use": (
                "immutable prerequisite provenance only; no V2.9 "
                "A-D treatment effect"
            ),
        }
    )


def verify_v29_parent_import_receipt(
    *,
    receipt_path: str | Path,
    child_repo_root: str | Path,
    contract: PilotContract,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Rebuild a V2.9 parent receipt from tracked/copy-bound authority."""

    child_root = _strict_real_root(
        child_repo_root,
        name="V2.9 child repository",
    )
    _validate_target_contract(contract, require_frozen=True)
    _verify_child_release_identity(
        child_root,
        expected_git_commit=expected_git_commit,
    )
    path = Path(receipt_path).absolute()
    expected_path = child_root.joinpath(
        *V29_RAW_ROOT.parts,
        "parent-import",
        "parent_import_receipt.json",
    )
    if path != expected_path:
        raise PilotV29Stage0ImportError(
            "V2.9 parent receipt path differs from contract namespace"
        )
    try:
        _, raw = _guarded_file(
            child_root,
            V29_RAW_ROOT / "parent-import/parent_import_receipt.json",
            name="V2.9 parent import receipt",
        )
        value = _strict_json(raw, name="V2.9 parent import receipt")
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    _verify_self_hashed(
        value,
        schema_version=V29_PARENT_IMPORT_SCHEMA_VERSION,
        name="V2.9 parent import receipt",
    )
    manifest = load_v29_source_manifest(
        child_root.joinpath(*V29_SOURCE_MANIFEST_PATH.parts)
    )
    rebuilt = _build_v29_parent_import_receipt(
        child_root=child_root,
        child_raw=child_root.joinpath(*V29_RAW_ROOT.parts),
        contract=contract,
        child_git_commit=expected_git_commit,
        manifest=manifest,
    )
    if value != rebuilt:
        raise PilotV29Stage0ImportError(
            "V2.9 parent receipt differs from copied authority"
        )
    return value


def persist_v29_parent_import(
    *,
    parent_repo_root: str | Path,
    child_repo_root: str | Path,
    child_raw_root: str | Path,
    contract: PilotContract,
    child_git_commit: str,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy exact V2.8 raw authority and write a zero-provider receipt."""

    parent_root = _strict_real_root(
        parent_repo_root,
        name="V2.8 source repository",
    )
    child_root = _strict_real_root(
        child_repo_root,
        name="V2.9 child repository",
    )
    child_raw = Path(child_raw_root).absolute()
    expected_raw = child_root.joinpath(*V29_RAW_ROOT.parts)
    if child_raw != expected_raw:
        raise PilotV29Stage0ImportError(
            "V2.9 child raw root differs from contract namespace"
        )
    _validate_target_contract(contract, require_frozen=True)
    _verify_child_release_identity(
        child_root,
        expected_git_commit=child_git_commit,
    )
    manifest = validate_v29_source_manifest(
        source_manifest,
        parent_repo_root=parent_root,
        child_repo_root=child_root,
        target_contract=contract,
    )
    _tracked_source_manifest_binding(child_root, manifest)
    source_raw = parent_root.joinpath(*V28_RAW_ROOT.parts)
    inventory = _verify_exact_v28_inventory(source_raw)
    snapshot = imported_v28_raw_root_v29(child_raw)
    _copy_exact_snapshot(
        source_root=source_raw,
        destination_root=snapshot,
        destination_guard_root=child_root,
        inventory=inventory,
    )
    receipt = _build_v29_parent_import_receipt(
        child_root=child_root,
        child_raw=child_raw,
        contract=contract,
        child_git_commit=child_git_commit,
        manifest=manifest,
    )
    receipt_path = (
        child_raw / "parent-import" / "parent_import_receipt.json"
    )
    raw = (
        json.dumps(
            receipt,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        _atomic_exact_bytes_no_follow(
            repo_root=child_root,
            path=receipt_path,
            raw=raw,
        )
    except PilotV27Stage0ImportError as exc:
        raise _translate(exc) from exc
    return {
        "receipt": str(receipt_path),
        "receipt_file_sha256": _sha256(raw),
        "receipt_content_sha256": receipt["integrity"]["content_sha256"],
        "snapshot_root": str(snapshot),
        "snapshot_inventory_sha256": V28_RAW_INVENTORY_SHA256,
        "nested_v2_6_inventory_sha256": V26_RAW_INVENTORY_SHA256,
        "imported_cell_count": 15,
        "imported_stage0_cell_count": 14,
        "q_ref_imported": False,
        "scripted_diagnostic_calls_during_import": 0,
        "hosted_provider_calls_during_import": 0,
        "provider_calls_during_import": 0,
        "scientific_evidence": False,
        "v2_8_terminal_no_go_preserved": True,
    }


__all__ = [
    "PilotV29Stage0ImportError",
    "V26_RAW_FILE_COUNT",
    "V26_RAW_INVENTORY_SHA256",
    "V26_RAW_STORAGE_BYTES",
    "V28_RAW_FILE_COUNT",
    "V28_RAW_INVENTORY_CANONICALIZATION",
    "V28_RAW_INVENTORY_SCHEMA_VERSION",
    "V28_RAW_INVENTORY_SHA256",
    "V28_RAW_STORAGE_BYTES",
    "V28_SCIENCE_COMMIT",
    "V28_SCIENCE_TAG",
    "V29_ALLOWED_P95_PROFILES",
    "V29_CONTRACT_ID",
    "V29_CUMULATIVE_DEBIT",
    "V29_EXPANDED_CONTRACT_PATH",
    "V29_PARENT_IMPORT_SCHEMA_VERSION",
    "V29_RAW_ROOT",
    "V29_SCIENCE_TAG",
    "V29_SNAPSHOT_RELATIVE",
    "V29_SOURCE_MANIFEST_PATH",
    "V29_SOURCE_MANIFEST_SCHEMA_VERSION",
    "build_v29_source_manifest",
    "imported_v26_run_dir_v29",
    "imported_v28_raw_root_v29",
    "load_v29_source_manifest",
    "parent_budget_debit_for_v29",
    "persist_v29_parent_import",
    "q_ref_audit_reference_v29",
    "snapshot_path_for_v28_source_artifact_v29",
    "source_binding_for_target_v29",
    "v2_8_p95_source_binding_v29",
    "v29_observed_p95_projection_path",
    "v29_observed_p95_receipt_path",
    "validate_v29_source_manifest",
    "verify_v29_imported_v28_observed_p95",
    "verify_v29_parent_import_receipt",
    "write_v29_source_manifest_draft",
]
