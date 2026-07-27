"""Immutable V2.9 -> V2.10 prerequisite and observed-p95 import.

V2.9 is a terminal ``complete-with-no-go`` release.  Its 211-cell ledger has
26 completed rows and 185 implementation failures.  V2.10 may import exactly
sixteen non-effect prerequisites (parent import, deterministic q-ref, and the
fourteen Stage-0 calibration cells); the ten completed offline candidate
admission rows and every failed A--D row remain audit-only.

The V2.9 observed-p95 verifier returns its receipt identity below nested
``authority`` and ``projection`` objects.  This module deliberately reseals
that source into a V2.10 release-bound authority and exposes the exact flat
binding consumed by :func:`verified_memory.pilot_orchestrator.
_runner_p95_reservations`.

This module never imports or constructs a provider.  Import, verification,
snapshot copying, and resealing are filesystem-only operations.
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
    canonical_sha256,
    load_pilot_contract,
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
from .pilot_v29_stage0_import import (
    V26_CONTRACT_CANONICAL_SHA256,
    V26_CONTRACT_ID,
    V28_CONTRACT_CANONICAL_SHA256,
    V28_CONTRACT_ID,
    V28_SCIENCE_COMMIT,
    V28_SCIENCE_TAG,
    V29_ALLOWED_P95_PROFILES,
    V29_RAW_ROOT,
    V29_SCIENCE_TAG,
    PilotV29Stage0ImportError,
    _inventory,
    verify_v29_imported_v28_observed_p95,
)
from .runner import PreflightP95Reservation


V210_CONTRACT_ID = "finevo-pilot-v2.10"
V210_SCIENCE_TAG = "pilot-v2.10-science"
V210_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.10/raw")
V210_EXPANDED_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_10.yaml")
V210_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_10_source_manifest.json"
)
V210_SNAPSHOT_RELATIVE = PurePosixPath("parent-import/v2_9_raw_snapshot")
V210_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.10-source-manifest-v1"
V210_PARENT_IMPORT_SCHEMA_VERSION = "finevo-pilot-v2.10-parent-import-v1"
V210_RESEALED_P95_AUTHORITY_SCHEMA_VERSION = (
    "finevo-pilot-v2.10-resealed-observed-p95-authority-v1"
)
V210_RESEALED_P95_SOURCE_KIND = "v2.9-terminal-parent-import-v2.10"
V210_ALLOWED_P95_PROFILES = tuple(V29_ALLOWED_P95_PROFILES)

V29_CONTRACT_ID = "finevo-pilot-v2.9"
V29_CONTRACT_CANONICAL_SHA256 = (
    "0b07881aaceeb020dc5943ede647a665f9e9bf786a1cac109ab720e05d81d361"
)
V29_CONTRACT_FILE_SHA256 = (
    "182d511d6fd64ae41e264eb8a1eec372ff1269854ed09b648e2f0ac262099a69"
)
V29_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_9.yaml")
V29_SCIENCE_TAG_OBJECT = "ca7871231769ab1d7eb811b71dff79f16de363e9"
V29_SCIENCE_COMMIT = "2349ccd41560383965da8880744cf4df366c9ee5"
V29_RAW_FILE_COUNT = 623
V29_RAW_STORAGE_BYTES = 19_288_343
V29_RAW_INVENTORY_SHA256 = (
    "ae478634a83a98bd206bcafa03f87636fcc392f8dd1e8e234f84696f245ef22f"
)
V29_RAW_INVENTORY_SCHEMA_VERSION = "finevo-raw-tree-inventory-v1"
V29_RAW_INVENTORY_CANONICALIZATION = "json-sort-keys-compact-utf8-v1"
V29_RUN_LEDGER_FILE_SHA256 = (
    "8f4ef1a2914f19bd4d7ee887e8c9d9566c45ec70bed5ec02572d6a3e24a7bbc0"
)
V29_RUN_LEDGER_INTERNAL_SHA256 = (
    "9cc948d75c37ffeb59a2d7ed569e140668a997fa314d523906a047375011e409"
)
V29_RUN_LEDGER_EVENT_COUNT = 213
V29_RUN_LEDGER_EVENT_HEAD = (
    "9998f766445a50b95651181361523dea26a6c341842d248d8b34774c573923e1"
)
V29_BUDGET_LEDGER_FILE_SHA256 = (
    "5119b803803bebb83c626159820c42fc22d2d091c76b70816ce40df121335f9e"
)
V29_BUDGET_LEDGER_INTERNAL_SHA256 = (
    "7e75b9c58bccaa746bdc92b926352fc0d3e56adee8426d3962a80ae5ddd59e10"
)
V29_BUDGET_LEDGER_EVENT_COUNT = 314
V29_BUDGET_LEDGER_EVENT_HEAD = (
    "dc9c3bfcc5c96adb90da97d1518448f2260a758d0aa0067a6bd56ca3ef4e5979"
)
V29_RELEASE_ATTESTATION_FILE_SHA256 = (
    "3433d082610919a501cbca23755e6ff3c8819b0924ab070ad96182703d1d80a2"
)

V29_EVIDENCE_PUBLICATION_COMMIT = "51525614e138e5b7ac498d15b409048d5110b753"
V29_EVIDENCE_MERGE_COMMIT = "08fcbc0dd9319fcc86c3f4e812c3db504a0c5a17"
V29_EVIDENCE_ROOT = PurePosixPath("evidence/current_v2/pilot-v2.9")
V29_EVIDENCE_CHECKSUMS_FILE_SHA256 = (
    "b0de7185c710b69736ddfe1d331b7f6308165a9f03bb0c616f14ec1fd7a515db"
)
V29_EVIDENCE_PACKAGE_FILE_SHA256 = (
    "6d006ba59c5af6a1e0dd3931466b90d4599edc0ded47e2de3ea4f8ecd6c4831a"
)
V29_EVIDENCE_AGGREGATE_FILE_SHA256 = (
    "8cddead63df3b9e86703ef54056da87b26fdd3ba63841df29f1a4e5188aa1936"
)
V29_EVIDENCE_FAILURE_FILE_SHA256 = (
    "2c4ef904a4aff0fff9185d63c62736bcbf130853b7371e868f8a3328b6df9bdb"
)
V29_EVIDENCE_REVIEWER_REPORT_FILE_SHA256 = (
    "19991e751ea71c73cd0927ace8ae271e69ea69ab95098c2c030a6266ceb817c0"
)

V210_CUMULATIVE_DEBIT = ParentBudgetDebit(
    parent_contract_sha256=V29_CONTRACT_CANONICAL_SHA256,
    parent_run_ledger_sha256=V29_RUN_LEDGER_INTERNAL_SHA256,
    parent_budget_ledger_sha256=V29_BUDGET_LEDGER_INTERNAL_SHA256,
    stage_bucket="parent_v23",
    cost_usd=3.212770875,
    hosted_completions=184,
    storage_bytes=50_425_235,
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PREREQUISITE_STAGES = (
    "parent-import",
    "q-ref-resolution",
    "stage0-calibration",
)
_EXPECTED_COMPLETE_BY_STAGE = {
    "parent-import": 1,
    "q-ref-resolution": 1,
    "stage0-calibration": 14,
    "experiment-c": 5,
    "local-experiment-c": 5,
}
_EXPECTED_FAILED_BY_STAGE = {
    "experiment-a": 20,
    "experiment-b": 15,
    "experiment-c": 20,
    "experiment-d": 30,
    "local-experiment-a": 20,
    "local-experiment-b": 25,
    "local-experiment-c": 20,
    "local-experiment-d": 35,
}


class PilotV210ParentImportError(RuntimeError):
    """Raised before V2.9 authority can enter V2.10."""


def _translate(exc: Exception) -> PilotV210ParentImportError:
    return PilotV210ParentImportError(str(exc))


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotV210ParentImportError(f"{name} must be an object")
    return value


def _strict_root(value: str | Path, *, name: str) -> Path:
    root = Path(value).absolute()
    for component in (root, *root.parents):
        try:
            if component.is_symlink():
                raise PilotV210ParentImportError(f"{name} path contains a symlink")
        except OSError as exc:
            raise PilotV210ParentImportError(f"{name} is unavailable") from exc
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
        raise PilotV210ParentImportError(f"{name} file hash drifted")
    return path, raw, value


def _artifact_binding(
    root: Path,
    relative: PurePosixPath,
    *,
    name: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        _, raw = _guarded_file(root, relative, name=name)
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    digest = _sha256(raw)
    if expected_sha256 is not None and digest != expected_sha256:
        raise PilotV210ParentImportError(f"{name} file hash drifted")
    result: dict[str, Any] = {
        "path": relative.as_posix(),
        "file_sha256": digest,
        "byte_size": len(raw),
    }
    try:
        value = _strict_json(raw, name=name)
    except PilotV24ParentImportError:
        return result
    integrity = value.get("integrity")
    if isinstance(integrity, Mapping) and _SHA256_RE.fullmatch(
        str(integrity.get("content_sha256", ""))
    ):
        result["content_sha256"] = integrity["content_sha256"]
    return result


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


def _verify_terminal_summary_hash(
    value: Mapping[str, Any],
    *,
    name: str,
) -> None:
    copied = _json_copy(value)
    integrity = copied.pop("integrity", None)
    if (
        value.get("schema_version") != "finevo-pilot-terminal-summary-v1"
        or not isinstance(integrity, Mapping)
        or set(integrity) != {"canonicalization", "content_sha256"}
        or integrity.get("canonicalization") != "json-sort-keys-utf8-v1"
        or integrity.get("content_sha256") != canonical_sha256(copied)
    ):
        raise PilotV210ParentImportError(f"{name} schema or content hash mismatch")


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
        raise PilotV210ParentImportError(f"{name} escaped the repository") from exc
    return normalized.as_posix()


def _normalized_spec(spec: PilotRunSpec | Mapping[str, Any]) -> dict[str, Any]:
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
        raise PilotV210ParentImportError(
            f"{name} source/target matrix is not an exact normalized match"
        )
    return [(source_map[key], target_map[key]) for key in sorted(source_map)]


def _validate_target_contract(
    contract: PilotContract,
    *,
    require_frozen: bool,
) -> None:
    status = getattr(contract, "status", None)
    implementation = _mapping(
        getattr(contract, "implementation", None),
        name="V2.10 implementation policy",
    )
    if (
        getattr(contract, "contract_id", None) != V210_CONTRACT_ID
        or implementation.get("required_git_tag") != V210_SCIENCE_TAG
        or (require_frozen and status != "frozen")
        or (not require_frozen and status not in {"draft", "frozen"})
    ):
        raise PilotV210ParentImportError(
            "V2.10 import requires its exact release-bound contract"
        )


def _verify_release_identity(
    root: Path,
    *,
    tag: str,
    expected_commit: str,
    expected_tag_object: str | None = None,
    name: str,
) -> None:
    if _COMMIT_RE.fullmatch(expected_commit) is None:
        raise PilotV210ParentImportError(f"{name} commit is malformed")
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
        raise PilotV210ParentImportError(f"{name} release/tag identity drifted")


def _load_verified_v29_contract(parent_root: Path) -> PilotContract:
    _, _, value = _strict_file(
        parent_root,
        V29_CONTRACT_PATH,
        name="frozen V2.9 contract",
        expected_sha256=V29_CONTRACT_FILE_SHA256,
    )
    contract = PilotContract.from_dict(value)
    if (
        contract.contract_id != V29_CONTRACT_ID
        or contract.canonical_hash != V29_CONTRACT_CANONICAL_SHA256
        or contract.status != "frozen"
        or contract.implementation.get("required_git_tag") != V29_SCIENCE_TAG
    ):
        raise PilotV210ParentImportError("frozen V2.9 contract identity drifted")
    return contract


def _verify_exact_v29_inventory(raw_root: Path) -> list[dict[str, Any]]:
    try:
        rows, summary = _inventory(raw_root, declared_root=V29_RAW_ROOT)
    except PilotV29Stage0ImportError as exc:
        raise _translate(exc) from exc
    expected = {
        "root": V29_RAW_ROOT.as_posix(),
        "inventory_schema_version": V29_RAW_INVENTORY_SCHEMA_VERSION,
        "inventory_canonicalization": V29_RAW_INVENTORY_CANONICALIZATION,
        "file_count": V29_RAW_FILE_COUNT,
        "storage_bytes": V29_RAW_STORAGE_BYTES,
        "inventory_sha256": V29_RAW_INVENTORY_SHA256,
    }
    if summary != expected:
        raise PilotV210ParentImportError("V2.9 raw-tree inventory drifted")
    return rows


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
        raise PilotV210ParentImportError(f"{name} self-hash/head drifted")
    previous = "0" * 64
    for index, source in enumerate(events):
        if not isinstance(source, Mapping):
            raise PilotV210ParentImportError(f"{name} event is malformed")
        row = _json_copy(source)
        digest = row.pop("event_sha256", None)
        if (
            source.get("event_index") != index
            or source.get("previous_event_sha256") != previous
            or digest != canonical_sha256(row)
        ):
            raise PilotV210ParentImportError(f"{name} event chain drifted")
        previous = str(digest)


def _verify_v29_ledgers(
    raw_root: Path,
    source_contract: PilotContract,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, _, run = _strict_file(
        raw_root,
        PurePosixPath("run_ledger.json"),
        name="V2.9 run ledger",
        expected_sha256=V29_RUN_LEDGER_FILE_SHA256,
    )
    _, _, budget = _strict_file(
        raw_root,
        PurePosixPath("budget_ledger.json"),
        name="V2.9 budget ledger",
        expected_sha256=V29_BUDGET_LEDGER_FILE_SHA256,
    )
    _validate_event_chain(
        run,
        internal_sha256=V29_RUN_LEDGER_INTERNAL_SHA256,
        event_count=V29_RUN_LEDGER_EVENT_COUNT,
        event_head=V29_RUN_LEDGER_EVENT_HEAD,
        name="V2.9 run ledger",
    )
    _validate_event_chain(
        budget,
        internal_sha256=V29_BUDGET_LEDGER_INTERNAL_SHA256,
        event_count=V29_BUDGET_LEDGER_EVENT_COUNT,
        event_head=V29_BUDGET_LEDGER_EVENT_HEAD,
        name="V2.9 budget ledger",
    )
    expected = {spec.run_id: spec.to_dict() for spec in source_contract.expand()}
    runs = _mapping(run.get("runs"), name="V2.9 run rows")
    status_by_stage: Counter[tuple[str, str]] = Counter()
    if set(runs) != set(expected):
        raise PilotV210ParentImportError(
            "V2.9 terminal ledger differs from its 211-cell denominator"
        )
    for run_id, row in runs.items():
        if (
            not isinstance(row, Mapping)
            or row.get("spec") != expected[run_id]
            or row.get("status") not in {"complete", "failed"}
        ):
            raise PilotV210ParentImportError(
                "V2.9 terminal ledger row identity/status drifted"
            )
        status_by_stage[(str(row["spec"]["stage_id"]), str(row["status"]))] += 1
    complete = {
        stage: count
        for (stage, status), count in status_by_stage.items()
        if status == "complete"
    }
    failed = {
        stage: count
        for (stage, status), count in status_by_stage.items()
        if status == "failed"
    }
    if (
        len(runs) != 211
        or complete != _EXPECTED_COMPLETE_BY_STAGE
        or failed != _EXPECTED_FAILED_BY_STAGE
    ):
        raise PilotV210ParentImportError("V2.9 terminal status denominator drifted")
    return _json_copy(run), _json_copy(budget)


def _verify_v29_evidence(evidence_repo_root: Path) -> dict[str, Any]:
    base = evidence_repo_root.joinpath(*V29_EVIDENCE_ROOT.parts)
    if base.is_symlink() or not base.is_dir():
        raise PilotV210ParentImportError("V2.9 evidence package is unavailable")
    relative_base = V29_EVIDENCE_ROOT
    _, _, checksums = _strict_file(
        evidence_repo_root,
        relative_base / "checksums.json",
        name="V2.9 evidence checksums",
        expected_sha256=V29_EVIDENCE_CHECKSUMS_FILE_SHA256,
    )
    _, _, package = _strict_file(
        evidence_repo_root,
        relative_base / "package_manifest.json",
        name="V2.9 evidence package manifest",
        expected_sha256=V29_EVIDENCE_PACKAGE_FILE_SHA256,
    )
    _, _, aggregate = _strict_file(
        evidence_repo_root,
        relative_base / "aggregate.json",
        name="V2.9 evidence aggregate",
        expected_sha256=V29_EVIDENCE_AGGREGATE_FILE_SHA256,
    )
    _artifact_binding(
        evidence_repo_root,
        relative_base / "failure_ledger.json",
        name="V2.9 evidence failure ledger",
        expected_sha256=V29_EVIDENCE_FAILURE_FILE_SHA256,
    )
    _artifact_binding(
        evidence_repo_root,
        relative_base / "reviewer_report.md",
        name="V2.9 reviewer report",
        expected_sha256=V29_EVIDENCE_REVIEWER_REPORT_FILE_SHA256,
    )
    files = checksums.get("files")
    if not isinstance(files, list) or len(files) != 17:
        raise PilotV210ParentImportError("V2.9 evidence checksum inventory drifted")
    seen: set[str] = set()
    for row in files:
        if not isinstance(row, Mapping):
            raise PilotV210ParentImportError("V2.9 evidence checksum row is malformed")
        try:
            relative = _normalized_relative(
                str(row.get("path", "")),
                required_top=None,
                name="V2.9 evidence checksum path",
            )
            _, raw = _guarded_file(
                base,
                relative,
                name=f"V2.9 evidence {relative.as_posix()}",
            )
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
        if (
            relative.as_posix() in seen
            or _sha256(raw) != row.get("sha256")
            or len(raw) != row.get("byte_size")
        ):
            raise PilotV210ParentImportError(
                "V2.9 evidence checksum verification failed"
            )
        seen.add(relative.as_posix())
    implementation = _mapping(
        aggregate.get("implementation_failure"),
        name="V2.9 implementation failure",
    )
    provider_boundary = _mapping(
        implementation.get("provider_boundary"),
        name="V2.9 provider boundary",
    )
    outcome_boundary = _mapping(
        implementation.get("outcome_boundary"),
        name="V2.9 outcome boundary",
    )
    denominator = _mapping(
        aggregate.get("denominator"),
        name="V2.9 evidence denominator",
    )
    if (
        aggregate.get("contract_id") != V29_CONTRACT_ID
        or aggregate.get("contract_sha256") != V29_CONTRACT_CANONICAL_SHA256
        or aggregate.get("publication_status") != "complete-with-no-go"
        or aggregate.get("resolved_git_commit") != V29_SCIENCE_COMMIT
        or aggregate.get("scientific_complete") is not False
        or aggregate.get("scientific_claim_gates_supported") is not False
        or denominator.get("expected_count") != 211
        or denominator.get("status_counts") != {"complete": 26, "failed": 185}
        or implementation.get("root_cause_code")
        != "imported-p95-runner-binding-shape-mismatch"
        or provider_boundary.get("failure_phase")
        != "before-provider-construction-and-dispatch"
        or provider_boundary.get("v2_9_hosted_completions") != 0
        or provider_boundary.get("v2_9_hosted_stage_cost_usd") != 0.0
        or outcome_boundary.get("actor_action_utility_rule_exposure_outcomes_generated")
        is not False
        or outcome_boundary.get("offline_candidate_admission_cells_generated") != 10
        or package.get("resolved_git_commit") != V29_SCIENCE_COMMIT
        or package.get("publication_status") != "complete-with-no-go"
    ):
        raise PilotV210ParentImportError(
            "V2.9 evidence semantics/claim boundary drifted"
        )
    return {
        "publication_commit": V29_EVIDENCE_PUBLICATION_COMMIT,
        "merge_commit": V29_EVIDENCE_MERGE_COMMIT,
        "root": V29_EVIDENCE_ROOT.as_posix(),
        "checksums_file_sha256": V29_EVIDENCE_CHECKSUMS_FILE_SHA256,
        "package_manifest_file_sha256": V29_EVIDENCE_PACKAGE_FILE_SHA256,
        "aggregate_file_sha256": V29_EVIDENCE_AGGREGATE_FILE_SHA256,
        "failure_ledger_file_sha256": V29_EVIDENCE_FAILURE_FILE_SHA256,
        "reviewer_report_file_sha256": V29_EVIDENCE_REVIEWER_REPORT_FILE_SHA256,
        "terminal_status": "complete-with-no-go",
        "root_cause_code": "imported-p95-runner-binding-shape-mismatch",
        "v2_9_hosted_completions": 0,
        "v2_9_hosted_stage_cost_usd": 0.0,
        "actor_treatment_effect_outcomes_generated": False,
        "offline_candidate_admission_cells_generated": 10,
        "scientific_claim_gates_supported": False,
    }


def _summary_binding(
    raw_root: Path,
    spec: PilotRunSpec,
) -> dict[str, Any]:
    relative = PurePosixPath(
        spec.stage_id,
        "summaries",
        f"{spec.run_id}.json",
    )
    _, raw, value = _strict_file(
        raw_root,
        relative,
        name=f"V2.9 prerequisite summary {spec.run_id}",
    )
    _verify_terminal_summary_hash(
        value,
        name=f"V2.9 prerequisite summary {spec.run_id}",
    )
    provenance = _mapping(
        value.get("provenance"),
        name=f"V2.9 prerequisite provenance {spec.run_id}",
    )
    if (
        value.get("contract_id") != V29_CONTRACT_ID
        or value.get("contract_sha256") != V29_CONTRACT_CANONICAL_SHA256
        or value.get("run_spec") != spec.to_dict()
        or provenance.get("git_tag") != V29_SCIENCE_TAG
        or provenance.get("resolved_git_commit") != V29_SCIENCE_COMMIT
    ):
        raise PilotV210ParentImportError(
            f"V2.9 prerequisite summary identity drifted: {spec.run_id}"
        )
    return {
        "path": (V29_RAW_ROOT / relative).as_posix(),
        "file_sha256": _sha256(raw),
        "content_sha256": value["integrity"]["content_sha256"],
        "schema_version": value["schema_version"],
    }


def _prerequisite_source_artifacts(
    raw_root: Path,
    source_spec: PilotRunSpec,
) -> dict[str, Any]:
    result: dict[str, Any] = {"summary": _summary_binding(raw_root, source_spec)}
    if source_spec.stage_id == "parent-import":
        for name in ("parent_import_receipt.json", "stage_receipt.json"):
            relative = PurePosixPath("parent-import", name)
            result[name.removesuffix(".json")] = {
                **_artifact_binding(
                    raw_root,
                    relative,
                    name=f"V2.9 parent prerequisite {name}",
                ),
                "path": (V29_RAW_ROOT / relative).as_posix(),
            }
    elif source_spec.stage_id == "q-ref-resolution":
        run_root = PurePosixPath(
            "q-ref-resolution",
            "runs",
            source_spec.run_id,
        )
        for key, relative in {
            "q_ref_resolution": PurePosixPath("q-ref-resolution/q_ref_resolution.json"),
            "run_manifest": run_root / "manifest.json",
            "stage_receipt": PurePosixPath("q-ref-resolution/stage_receipt.json"),
        }.items():
            result[key] = {
                **_artifact_binding(
                    raw_root,
                    relative,
                    name=f"V2.9 q-ref prerequisite {key}",
                ),
                "path": (V29_RAW_ROOT / relative).as_posix(),
            }
        result["run_root"] = (V29_RAW_ROOT / run_root).as_posix()
    elif source_spec.stage_id == "stage0-calibration":
        envelope = PurePosixPath(
            "stage0-calibration",
            "runs",
            source_spec.run_id,
            "imported_run_envelope.json",
        )
        result["imported_run_envelope"] = {
            **_artifact_binding(
                raw_root,
                envelope,
                name=f"V2.9 Stage-0 envelope {source_spec.run_id}",
            ),
            "path": (V29_RAW_ROOT / envelope).as_posix(),
        }
        for key, relative in {
            "stage0_selection": PurePosixPath(
                "stage0-calibration/stage0_selection.json"
            ),
            "stage_receipt": PurePosixPath("stage0-calibration/stage_receipt.json"),
        }.items():
            result[key] = {
                **_artifact_binding(
                    raw_root,
                    relative,
                    name=f"V2.9 Stage-0 {key}",
                ),
                "path": (V29_RAW_ROOT / relative).as_posix(),
            }
        result["run_root"] = (V29_RAW_ROOT / envelope.parent).as_posix()
    else:  # pragma: no cover - guarded by the caller
        raise PilotV210ParentImportError(
            f"unsupported prerequisite stage {source_spec.stage_id}"
        )
    return result


def build_v210_prerequisite_bindings(
    *,
    source_contract: PilotContract,
    target_contract: PilotContract,
    source_run_ledger: Mapping[str, Any],
    source_raw_root: str | Path,
) -> list[dict[str, Any]]:
    """Map exactly 16 V2.10 prerequisite cells to immutable V2.9 artifacts."""

    _validate_target_contract(target_contract, require_frozen=False)
    if (
        source_contract.contract_id != V29_CONTRACT_ID
        or source_contract.canonical_hash != V29_CONTRACT_CANONICAL_SHA256
    ):
        raise PilotV210ParentImportError("V2.9 source contract identity drifted")
    runs = _mapping(source_run_ledger.get("runs"), name="V2.9 run rows")
    result: list[dict[str, Any]] = []
    raw_root = Path(source_raw_root).absolute()
    for stage_id in _PREREQUISITE_STAGES:
        pairs = _match_specs(
            source_contract.expand(stage=stage_id),
            target_contract.expand(stage=stage_id),
            name=f"V2.10 {stage_id} prerequisite",
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
                raise PilotV210ParentImportError(
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
                    "source_artifacts": _prerequisite_source_artifacts(
                        raw_root,
                        source_spec,
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
        raise PilotV210ParentImportError(
            "V2.10 imported prerequisite inventory drifted"
        )
    return sorted(result, key=lambda row: row["target_run_id"])


def normalize_v29_observed_p95_binding(
    binding: Mapping[str, Any],
    *,
    profile_id: str,
) -> dict[str, Any]:
    """Normalize the nested V2.9 verifier result to the legacy flat shape.

    This is intentionally strict: it accepts the exact V2.9 producer shape
    that caused the terminal V2.9 consumer failure, and no legacy alias.
    """

    value = _json_copy(_mapping(binding, name="V2.9 p95 binding"))
    if set(value) != {
        "profile_id",
        "source_contract_id",
        "source_contract_sha256",
        "source_git_commit",
        "source_git_tag",
        "authority",
        "projection",
        "runtime_model",
        "served_model",
        "reservations",
    }:
        raise PilotV210ParentImportError("V2.9 p95 binding shape drifted")
    authority = _mapping(value.get("authority"), name="V2.9 p95 authority")
    projection = _mapping(value.get("projection"), name="V2.9 p95 projection")
    reservations = _mapping(
        value.get("reservations"),
        name="V2.9 p95 reservations",
    )
    runtime_model = str(value.get("runtime_model", ""))
    if (
        profile_id not in V210_ALLOWED_P95_PROFILES
        or value.get("profile_id") != profile_id
        or value.get("source_contract_id") != V28_CONTRACT_ID
        or value.get("source_contract_sha256") != V28_CONTRACT_CANONICAL_SHA256
        or value.get("source_git_commit") != V28_SCIENCE_COMMIT
        or value.get("source_git_tag") != V28_SCIENCE_TAG
        or set(authority)
        != {
            "path",
            "schema_version",
            "file_sha256",
            "content_sha256",
        }
        or set(projection)
        != {
            "path",
            "schema_version",
            "file_sha256",
            "content_sha256",
        }
        or not all(
            _SHA256_RE.fullmatch(str(authority.get(key, "")))
            for key in ("file_sha256", "content_sha256")
        )
        or not all(
            _SHA256_RE.fullmatch(str(projection.get(key, "")))
            for key in ("file_sha256", "content_sha256")
        )
        or set(reservations) != {runtime_model}
        or not isinstance(reservations.get(runtime_model), Mapping)
    ):
        raise PilotV210ParentImportError("V2.9 p95 binding identity drifted")
    for call_kind in ("action", "semantic"):
        entry = reservations[runtime_model].get(call_kind)
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"authority", "reservation"}
            or not isinstance(entry.get("authority"), Mapping)
        ):
            raise PilotV210ParentImportError(
                f"V2.9 {profile_id}/{call_kind} p95 entry drifted"
            )
        try:
            parsed = PreflightP95Reservation.from_dict(
                model=runtime_model,
                call_kind=call_kind,
                value=entry["reservation"],
            )
        except (TypeError, ValueError) as exc:
            raise PilotV210ParentImportError(
                f"V2.9 {profile_id}/{call_kind} p95 reservation is invalid"
            ) from exc
        if parsed.to_dict() != entry["reservation"]:
            raise PilotV210ParentImportError(
                f"V2.9 {profile_id}/{call_kind} p95 reservation drifted"
            )
    return {
        "receipt_path": authority["path"],
        "receipt_file_sha256": authority["file_sha256"],
        "receipt_content_sha256": authority["content_sha256"],
        "git_commit": value["source_git_commit"],
        "reservations": _json_copy(reservations),
    }


def imported_v29_raw_root_v210(child_raw_root: str | Path) -> Path:
    return Path(child_raw_root).joinpath(*V210_SNAPSHOT_RELATIVE.parts)


def v210_v29_snapshot_root(child_raw_root: str | Path) -> Path:
    """Public spelling used by the V2.10 orchestrator."""

    return imported_v29_raw_root_v210(child_raw_root)


def snapshot_path_for_v29_source_artifact_v210(
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
        raise PilotV210ParentImportError(
            "source artifact is outside the V2.9 raw namespace"
        )
    inside = PurePosixPath(*relative.parts[len(V29_RAW_ROOT.parts) :])
    return imported_v29_raw_root_v210(child_raw_root).joinpath(*inside.parts)


def v210_observed_p95_receipt_path(
    raw_root: str | Path,
    profile_id: str,
) -> Path:
    if profile_id not in V210_ALLOWED_P95_PROFILES:
        raise PilotV210ParentImportError(f"unsupported V2.10 p95 profile: {profile_id}")
    return (
        Path(raw_root)
        / "parent-import"
        / "observed_p95"
        / profile_id
        / "observed_p95_authority_receipt.json"
    )


def v210_observed_p95_projection_path(
    raw_root: str | Path,
    profile_id: str,
) -> Path:
    return v210_observed_p95_receipt_path(raw_root, profile_id).with_name(
        "projection_p95.json"
    )


def v2_9_p95_source_binding_v210(
    *,
    child_raw_root: str | Path,
    profile_id: str,
) -> dict[str, Any]:
    """Verify V2.9's full copied snapshot and its nested V2.8 p95 pair."""

    snapshot = imported_v29_raw_root_v210(child_raw_root)
    rows, summary = _inventory(snapshot, declared_root=V29_RAW_ROOT)
    del rows
    if summary != {
        "root": V29_RAW_ROOT.as_posix(),
        "inventory_schema_version": V29_RAW_INVENTORY_SCHEMA_VERSION,
        "inventory_canonicalization": V29_RAW_INVENTORY_CANONICALIZATION,
        "file_count": V29_RAW_FILE_COUNT,
        "storage_bytes": V29_RAW_STORAGE_BYTES,
        "inventory_sha256": V29_RAW_INVENTORY_SHA256,
    }:
        raise PilotV210ParentImportError("copied V2.9 raw-tree inventory drifted")
    try:
        nested = verify_v29_imported_v28_observed_p95(
            snapshot,
            profile_id,
            expected_parent_commit=V28_SCIENCE_COMMIT,
        )
    except PilotV29Stage0ImportError as exc:
        raise _translate(exc) from exc
    normalized = normalize_v29_observed_p95_binding(
        nested,
        profile_id=profile_id,
    )
    return {
        "v2_9_terminal_parent": {
            "contract_id": V29_CONTRACT_ID,
            "contract_sha256": V29_CONTRACT_CANONICAL_SHA256,
            "science_tag": V29_SCIENCE_TAG,
            "science_commit": V29_SCIENCE_COMMIT,
            "raw_file_count": V29_RAW_FILE_COUNT,
            "raw_storage_bytes": V29_RAW_STORAGE_BYTES,
            "raw_inventory_sha256": V29_RAW_INVENTORY_SHA256,
            "terminal_status": "complete-with-no-go",
            "implementation_root_cause": "imported-p95-runner-binding-shape-mismatch",
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
        V210_EXPANDED_CONTRACT_PATH,
        name="expanded V2.10 contract",
    )
    if value != contract.to_dict():
        raise PilotV210ParentImportError(
            "expanded V2.10 contract differs from selected contract"
        )
    return {
        "path": V210_EXPANDED_CONTRACT_PATH.as_posix(),
        "file_sha256": _sha256(raw),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
    }


def build_v210_resealed_observed_p95_authority(
    *,
    repo_root: str | Path,
    contract: PilotContract,
    raw_root: str | Path,
    profile_id: str,
    expected_git_commit: str,
    verified_v2_9_source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a current-release p95 receipt/projection without persisting it."""

    root = _strict_root(repo_root, name="V2.10 child repository")
    _validate_target_contract(contract, require_frozen=True)
    if (
        _COMMIT_RE.fullmatch(expected_git_commit) is None
        or profile_id not in V210_ALLOWED_P95_PROFILES
    ):
        raise PilotV210ParentImportError("V2.10 p95 release commit/profile is invalid")
    raw_path = Path(raw_root)
    if not raw_path.is_absolute():
        raw_path = root.joinpath(*PurePosixPath(str(raw_root)).parts)
    raw_path = raw_path.absolute()
    if (
        _repo_relative(
            root,
            raw_path,
            required_top="experiment_results",
            name="V2.10 raw root",
        )
        != V210_RAW_ROOT.as_posix()
    ):
        raise PilotV210ParentImportError("V2.10 p95 requires the exact raw namespace")
    source = _json_copy(
        _mapping(
            verified_v2_9_source_binding,
            name="verified V2.9 p95 source binding",
        )
    )
    if set(source) != {
        "v2_9_terminal_parent",
        "v2_8_observed_p95_origin",
        "normalized_v2_9_binding",
    }:
        raise PilotV210ParentImportError("V2.9 p95 source lineage shape drifted")
    nested = _mapping(
        source["v2_8_observed_p95_origin"],
        name="V2.8 p95 origin",
    )
    normalized = normalize_v29_observed_p95_binding(
        nested,
        profile_id=profile_id,
    )
    if source["normalized_v2_9_binding"] != normalized:
        raise PilotV210ParentImportError("V2.9 nested/flat p95 normalization drifted")
    terminal = _mapping(
        source["v2_9_terminal_parent"],
        name="V2.9 terminal p95 parent",
    )
    if terminal != {
        "contract_id": V29_CONTRACT_ID,
        "contract_sha256": V29_CONTRACT_CANONICAL_SHA256,
        "science_tag": V29_SCIENCE_TAG,
        "science_commit": V29_SCIENCE_COMMIT,
        "raw_file_count": V29_RAW_FILE_COUNT,
        "raw_storage_bytes": V29_RAW_STORAGE_BYTES,
        "raw_inventory_sha256": V29_RAW_INVENTORY_SHA256,
        "terminal_status": "complete-with-no-go",
        "implementation_root_cause": "imported-p95-runner-binding-shape-mismatch",
    }:
        raise PilotV210ParentImportError("V2.9 terminal p95 lineage drifted")
    profile = contract.provider_profiles[profile_id]
    runtime_model = f"{profile.transport}/{profile.requested_model}"
    if (
        nested.get("runtime_model") != runtime_model
        or nested.get("served_model") != profile.served_model
    ):
        raise PilotV210ParentImportError(
            "V2.10 provider profile differs from inherited p95 source"
        )
    reservations = _json_copy(normalized["reservations"])
    for call_kind in ("action", "semantic"):
        authority = reservations[runtime_model][call_kind]["authority"]
        if not isinstance(authority, dict):
            raise PilotV210ParentImportError(
                "inherited p95 authority is not mutable JSON"
            )
        authority["pilot_contract_hash"] = contract.canonical_hash
        authority["pilot_tag"] = V210_SCIENCE_TAG

    receipt = _seal(
        {
            "schema_version": V210_RESEALED_P95_AUTHORITY_SCHEMA_VERSION,
            "contract": _contract_binding(root, contract),
            "raw_root": V210_RAW_ROOT.as_posix(),
            "git": {
                "tag": V210_SCIENCE_TAG,
                "commit": expected_git_commit,
            },
            "model": {
                "model_id": profile_id,
                "runtime_model": runtime_model,
                "served_model": profile.served_model,
            },
            "parent_lineage": source,
            "reservations": reservations,
            "provider_boundary": {
                "provider_construction_during_reseal": False,
                "provider_calls_during_reseal": 0,
                "hosted_provider_calls_during_reseal": 0,
                "hosted_cost_usd_during_reseal": 0.0,
            },
            "scientific_evidence": False,
            "evidence_use": (
                "V2.10 prospective budget authority only; V2.9/V2.8 "
                "parent rows contribute no V2.10 A-D treatment effect."
            ),
        }
    )
    receipt_path = v210_observed_p95_receipt_path(raw_path, profile_id)
    receipt_relative = _repo_relative(
        root,
        receipt_path,
        required_top="experiment_results",
        name="V2.10 p95 receipt",
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
                "git_tag": V210_SCIENCE_TAG,
                "git_commit": expected_git_commit,
                "source_kind": V210_RESEALED_P95_SOURCE_KIND,
                "source_authority_receipt": receipt_relative,
                "source_authority_receipt_content_sha256": receipt["integrity"][
                    "content_sha256"
                ],
                "source_v2_9_terminal_raw_inventory_sha256": V29_RAW_INVENTORY_SHA256,
                "source_v2_8_authority_content_sha256": nested["authority"][
                    "content_sha256"
                ],
                "source_v2_8_projection_content_sha256": nested["projection"][
                    "content_sha256"
                ],
            },
        }
    )
    return {
        "receipt_path": receipt_path,
        "projection_path": v210_observed_p95_projection_path(
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


def persist_v210_resealed_observed_p95_authority(
    *,
    repo_root: str | Path,
    contract: PilotContract,
    raw_root: str | Path,
    profile_id: str,
    expected_git_commit: str,
    verified_v2_9_source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    built = build_v210_resealed_observed_p95_authority(
        repo_root=repo_root,
        contract=contract,
        raw_root=raw_root,
        profile_id=profile_id,
        expected_git_commit=expected_git_commit,
        verified_v2_9_source_binding=verified_v2_9_source_binding,
    )
    root = _strict_root(repo_root, name="V2.10 child repository")
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
        "receipt_content_sha256": built["receipt"]["integrity"]["content_sha256"],
        "projection_content_sha256": built["projection"]["integrity"]["content_sha256"],
        "provider_construction_during_reseal": False,
        "provider_calls_during_reseal": 0,
    }


def _load_current_contract(
    repo_root: Path,
    selected: PilotContract | None,
) -> PilotContract:
    contract = selected or load_pilot_contract(
        repo_root.joinpath(*V210_EXPANDED_CONTRACT_PATH.parts)
    )
    _validate_target_contract(contract, require_frozen=True)
    _contract_binding(repo_root, contract)
    return contract


def _rebuild_v210_p95(
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
        "reservations",
        "provider_boundary",
        "scientific_evidence",
        "evidence_use",
        "integrity",
    }
    if (
        set(value) != expected_keys
        or value.get("schema_version") != V210_RESEALED_P95_AUTHORITY_SCHEMA_VERSION
    ):
        raise PilotV210ParentImportError("V2.10 p95 receipt shape/schema drifted")
    _verify_self_hashed(
        value,
        schema_version=V210_RESEALED_P95_AUTHORITY_SCHEMA_VERSION,
        name="V2.10 p95 receipt",
    )
    model = _mapping(value.get("model"), name="V2.10 p95 model")
    profile_id = str(model.get("model_id"))
    source = v2_9_p95_source_binding_v210(
        child_raw_root=raw_root,
        profile_id=profile_id,
    )
    rebuilt = build_v210_resealed_observed_p95_authority(
        repo_root=repo_root,
        contract=contract,
        raw_root=raw_root,
        profile_id=profile_id,
        expected_git_commit=expected_git_commit,
        verified_v2_9_source_binding=source,
    )
    if value != rebuilt["receipt"]:
        raise PilotV210ParentImportError(
            "V2.10 p95 receipt differs from current source/release authority"
        )
    return rebuilt


def verify_v210_resealed_observed_p95_authority(
    receipt: Mapping[str, Any],
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    expected_git_commit: str,
    contract: PilotContract | None = None,
) -> dict[str, Any]:
    root = _strict_root(repo_root, name="V2.10 child repository")
    _verify_release_identity(
        root,
        tag=V210_SCIENCE_TAG,
        expected_commit=expected_git_commit,
        name="V2.10 child",
    )
    selected = _load_current_contract(root, contract)
    built = _rebuild_v210_p95(
        receipt,
        repo_root=root,
        raw_root=Path(raw_root).absolute(),
        contract=selected,
        expected_git_commit=expected_git_commit,
    )
    return _json_copy(built["receipt"]["reservations"])


def verify_v210_resealed_observed_p95_projection(
    projection: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    repo_root: str | Path,
    raw_root: str | Path,
    expected_git_commit: str,
    contract: PilotContract | None = None,
) -> dict[str, Any]:
    root = _strict_root(repo_root, name="V2.10 child repository")
    _verify_release_identity(
        root,
        tag=V210_SCIENCE_TAG,
        expected_commit=expected_git_commit,
        name="V2.10 child",
    )
    selected = _load_current_contract(root, contract)
    built = _rebuild_v210_p95(
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
        name="V2.10 p95 projection",
    )
    if candidate != built["projection"]:
        raise PilotV210ParentImportError(
            "V2.10 p95 projection differs from its receipt/source"
        )
    return candidate


def verified_v210_observed_p95_authority_binding(
    receipt_path: str | Path,
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    expected_git_commit: str,
    contract: PilotContract | None = None,
) -> dict[str, Any]:
    """Return the exact flat runner binding after full receipt/projection replay."""

    root = _strict_root(repo_root, name="V2.10 child repository")
    path = Path(receipt_path)
    if path.is_absolute():
        try:
            relative = PurePosixPath(*path.absolute().relative_to(root).parts)
        except ValueError as exc:
            raise PilotV210ParentImportError(
                "V2.10 p95 receipt escaped the repository"
            ) from exc
    else:
        try:
            relative = _normalized_relative(
                path,
                required_top="experiment_results",
                name="V2.10 p95 receipt path",
            )
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
    _, raw, receipt = _strict_file(
        root,
        relative,
        name="V2.10 p95 receipt",
    )
    reservations = verify_v210_resealed_observed_p95_authority(
        receipt,
        repo_root=root,
        raw_root=raw_root,
        expected_git_commit=expected_git_commit,
        contract=contract,
    )
    projection_path = root.joinpath(*relative.parts).with_name("projection_p95.json")
    projection_relative = PurePosixPath(*projection_path.relative_to(root).parts)
    _, _, projection = _strict_file(
        root,
        projection_relative,
        name="V2.10 p95 projection",
    )
    verify_v210_resealed_observed_p95_projection(
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


def verified_v210_observed_p95_projection_binding(
    projection_path: str | Path,
    *,
    receipt_path: str | Path,
    repo_root: str | Path,
    raw_root: str | Path,
    expected_git_commit: str,
    contract: PilotContract | None = None,
) -> dict[str, Any]:
    """Return a path/hash/identity binding for ``_load_verified_projection``."""

    root = _strict_root(repo_root, name="V2.10 child repository")
    authority = verified_v210_observed_p95_authority_binding(
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
            raise PilotV210ParentImportError(
                "V2.10 p95 projection escaped the repository"
            ) from exc
    else:
        try:
            relative = _normalized_relative(
                path,
                required_top="experiment_results",
                name="V2.10 p95 projection path",
            )
        except PilotV24ParentImportError as exc:
            raise _translate(exc) from exc
    _, raw, projection = _strict_file(
        root,
        relative,
        name="V2.10 p95 projection",
    )
    receipt_relative = PurePosixPath(authority["receipt_path"])
    _, _, receipt = _strict_file(
        root,
        receipt_relative,
        name="V2.10 p95 receipt",
    )
    payload = verify_v210_resealed_observed_p95_projection(
        projection,
        receipt=receipt,
        repo_root=root,
        raw_root=raw_root,
        expected_git_commit=expected_git_commit,
        contract=contract,
    )
    model = _mapping(receipt.get("model"), name="V2.10 p95 model")
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


def verify_v29_terminal_source(
    *,
    parent_repo_root: str | Path,
    evidence_repo_root: str | Path,
    target_contract: PilotContract,
) -> dict[str, Any]:
    """Audit immutable V2.9 raw/evidence and select exactly 16 prerequisites."""

    parent = _strict_root(parent_repo_root, name="V2.9 source repository")
    evidence_root = _strict_root(
        evidence_repo_root,
        name="V2.9 evidence repository",
    )
    _validate_target_contract(target_contract, require_frozen=False)
    _verify_release_identity(
        parent,
        tag=V29_SCIENCE_TAG,
        expected_commit=V29_SCIENCE_COMMIT,
        expected_tag_object=V29_SCIENCE_TAG_OBJECT,
        name="V2.9 source",
    )
    source_contract = _load_verified_v29_contract(parent)
    raw_root = parent.joinpath(*V29_RAW_ROOT.parts)
    inventory = _verify_exact_v29_inventory(raw_root)
    run_ledger, budget_ledger = _verify_v29_ledgers(
        raw_root,
        source_contract,
    )
    evidence = _verify_v29_evidence(evidence_root)
    imported = build_v210_prerequisite_bindings(
        source_contract=source_contract,
        target_contract=target_contract,
        source_run_ledger=run_ledger,
        source_raw_root=raw_root,
    )
    p95 = {}
    for profile_id in V210_ALLOWED_P95_PROFILES:
        try:
            nested = verify_v29_imported_v28_observed_p95(
                raw_root,
                profile_id,
                expected_parent_commit=V28_SCIENCE_COMMIT,
            )
        except PilotV29Stage0ImportError as exc:
            raise _translate(exc) from exc
        p95[profile_id] = {
            "v2_9_terminal_parent": {
                "contract_id": V29_CONTRACT_ID,
                "contract_sha256": V29_CONTRACT_CANONICAL_SHA256,
                "science_tag": V29_SCIENCE_TAG,
                "science_commit": V29_SCIENCE_COMMIT,
                "raw_file_count": V29_RAW_FILE_COUNT,
                "raw_storage_bytes": V29_RAW_STORAGE_BYTES,
                "raw_inventory_sha256": V29_RAW_INVENTORY_SHA256,
                "terminal_status": "complete-with-no-go",
                "implementation_root_cause": "imported-p95-runner-binding-shape-mismatch",
            },
            "v2_8_observed_p95_origin": _json_copy(nested),
            "normalized_v2_9_binding": normalize_v29_observed_p95_binding(
                nested,
                profile_id=profile_id,
            ),
        }
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
        "budget_debit": V210_CUMULATIVE_DEBIT.to_dict(),
        "provider_construction_during_import": False,
        "provider_calls_during_import": 0,
    }


def build_v210_source_manifest(
    *,
    parent_repo_root: str | Path,
    evidence_repo_root: str | Path,
    target_contract: PilotContract,
) -> dict[str, Any]:
    """Build the deterministic V2.10 source manifest from frozen V2.9."""

    audit = verify_v29_terminal_source(
        parent_repo_root=parent_repo_root,
        evidence_repo_root=evidence_repo_root,
        target_contract=target_contract,
    )
    return _seal(
        {
            "schema_version": V210_SOURCE_MANIFEST_SCHEMA_VERSION,
            "v2_9_terminal_parent": {
                "contract": {
                    "contract_id": V29_CONTRACT_ID,
                    "path": V29_CONTRACT_PATH.as_posix(),
                    "schema_version": "finevo-pilot-contract-v2",
                    "status": "frozen",
                    "file_sha256": V29_CONTRACT_FILE_SHA256,
                    "canonical_sha256": V29_CONTRACT_CANONICAL_SHA256,
                },
                "release": {
                    "science_tag": V29_SCIENCE_TAG,
                    "science_tag_object": V29_SCIENCE_TAG_OBJECT,
                    "science_commit": V29_SCIENCE_COMMIT,
                    "tag_kind": "annotated",
                    "raw_root": V29_RAW_ROOT.as_posix(),
                    "release_attestation": {
                        "path": (V29_RAW_ROOT / "release_attestation.json").as_posix(),
                        "file_sha256": V29_RELEASE_ATTESTATION_FILE_SHA256,
                    },
                },
                "raw_snapshot": {
                    "root": V29_RAW_ROOT.as_posix(),
                    "inventory_schema_version": V29_RAW_INVENTORY_SCHEMA_VERSION,
                    "inventory_canonicalization": V29_RAW_INVENTORY_CANONICALIZATION,
                    "file_count": V29_RAW_FILE_COUNT,
                    "storage_bytes": V29_RAW_STORAGE_BYTES,
                    "inventory_sha256": V29_RAW_INVENTORY_SHA256,
                },
                "ledgers": {
                    "run": {
                        "path": (V29_RAW_ROOT / "run_ledger.json").as_posix(),
                        "file_sha256": V29_RUN_LEDGER_FILE_SHA256,
                        "internal_sha256": V29_RUN_LEDGER_INTERNAL_SHA256,
                        "event_count": V29_RUN_LEDGER_EVENT_COUNT,
                        "event_chain_head": V29_RUN_LEDGER_EVENT_HEAD,
                    },
                    "budget": {
                        "path": (V29_RAW_ROOT / "budget_ledger.json").as_posix(),
                        "file_sha256": V29_BUDGET_LEDGER_FILE_SHA256,
                        "internal_sha256": V29_BUDGET_LEDGER_INTERNAL_SHA256,
                        "event_count": V29_BUDGET_LEDGER_EVENT_COUNT,
                        "event_chain_head": V29_BUDGET_LEDGER_EVENT_HEAD,
                    },
                },
                "terminal_denominator": {
                    "registered_cells": 211,
                    "scientific_cells": 209,
                    "terminal_cells": 211,
                    "all_rows_present": True,
                    "all_rows_terminal": True,
                    "status_counts": {
                        "complete": 26,
                        "failed": 185,
                    },
                    "completed_prerequisite_breakdown": {
                        "parent-import": 1,
                        "q-ref-resolution": 1,
                        "stage0-calibration": 14,
                    },
                    "completed_nonimported_offline_candidate_cells": 10,
                    "failed_actor_cells": 185,
                    "terminal_status": "complete-with-no-go",
                    "scientific_complete": False,
                    "scientific_matrix_complete": False,
                    "scientific_claim_gates_supported": False,
                    "implementation_root_cause": "imported-p95-runner-binding-shape-mismatch",
                },
            },
            "published_v2_9_evidence": audit["evidence"],
            "imported_complete_cells": audit["imported_cells"],
            "v2_9_p95_sources_for_current_release_reseal": audit["p95_sources"],
            "cumulative_budget_debit": V210_CUMULATIVE_DEBIT.to_dict(),
            "import_policy": {
                "source_raw_namespace": V29_RAW_ROOT.as_posix(),
                "child_raw_namespace": V210_RAW_ROOT.as_posix(),
                "child_snapshot_namespace": (
                    V210_RAW_ROOT / V210_SNAPSHOT_RELATIVE
                ).as_posix(),
                "exact_full_v2_9_raw_snapshot_copy": True,
                "imported_cell_count": 16,
                "imported_cell_breakdown": {
                    "parent-import": 1,
                    "q-ref-resolution": 1,
                    "stage0-calibration": 14,
                },
                "q_ref_imported": True,
                "q_ref_value": 63.50397933257746,
                "stage0_imported_cells": 14,
                "stage0_selected_profile_id": "nu-0.5",
                "source_manifests_rewritten": False,
                "source_journals_rewritten": False,
                "provider_construction_during_import": False,
                "provider_calls_during_import": 0,
                "provider_redispatch_for_imported_cells": "forbidden",
                "offline_candidate_admission_cells_imported": 0,
                "failed_actor_cells_retried_via_import": 0,
                "v2_9_no_go_preserved": True,
                "v2_9_terminal_rows_reclassified": False,
                "prerequisites_are_treatment_effect_evidence": False,
            },
            "p95_reseal_policy": {
                "source_via_exact_v2_9_snapshot": True,
                "source_origin_contract_id": V28_CONTRACT_ID,
                "source_origin_contract_sha256": V28_CONTRACT_CANONICAL_SHA256,
                "source_reservation_values_unchanged": True,
                "nested_v2_9_binding_direct_reuse": "forbidden",
                "current_release_receipt_and_projection_required": True,
                "current_release_flat_runner_binding_required": True,
                "validation_before_provider_construction": True,
            },
            "observation_boundary": {
                "v2_9_implementation_failure_observed_before_amendment": True,
                "v2_9_actor_treatment_effect_outcomes_generated": False,
                "v2_9_offline_candidate_admission_outcomes_generated": 10,
                "v2_9_offline_candidate_admission_outcomes_imported": 0,
                "q_ref_and_stage0_prerequisites_observed": True,
                "stage0_selected_profile_observed": "nu-0.5",
                "a_d_actor_cells_fresh_in_v2_10": 195,
                "failed_seed_replacement": "forbidden",
            },
        }
    )


def _validate_source_manifest_structure(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = _json_copy(_mapping(value, name="V2.10 source manifest"))
    expected_keys = {
        "schema_version",
        "v2_9_terminal_parent",
        "published_v2_9_evidence",
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
        or candidate.get("schema_version") != V210_SOURCE_MANIFEST_SCHEMA_VERSION
    ):
        raise PilotV210ParentImportError("V2.10 source manifest shape/schema drifted")
    _verify_self_hashed(
        candidate,
        schema_version=V210_SOURCE_MANIFEST_SCHEMA_VERSION,
        name="V2.10 source manifest",
    )
    parent = _mapping(
        candidate.get("v2_9_terminal_parent"),
        name="V2.10 source V2.9 parent",
    )
    terminal = _mapping(
        parent.get("terminal_denominator"),
        name="V2.10 source terminal denominator",
    )
    policy = _mapping(
        candidate.get("import_policy"),
        name="V2.10 source import policy",
    )
    evidence = _mapping(
        candidate.get("published_v2_9_evidence"),
        name="V2.10 source evidence",
    )
    rows = candidate.get("imported_complete_cells")
    p95 = candidate.get("v2_9_p95_sources_for_current_release_reseal")
    if (
        parent.get("contract", {}).get("canonical_sha256")
        != V29_CONTRACT_CANONICAL_SHA256
        or parent.get("release", {}).get("science_commit") != V29_SCIENCE_COMMIT
        or parent.get("raw_snapshot", {}).get("inventory_sha256")
        != V29_RAW_INVENTORY_SHA256
        or terminal.get("status_counts") != {"complete": 26, "failed": 185}
        or terminal.get("implementation_root_cause")
        != "imported-p95-runner-binding-shape-mismatch"
        or evidence.get("checksums_file_sha256") != V29_EVIDENCE_CHECKSUMS_FILE_SHA256
        or evidence.get("package_manifest_file_sha256")
        != V29_EVIDENCE_PACKAGE_FILE_SHA256
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
        or not isinstance(p95, Mapping)
        or set(p95) != set(V210_ALLOWED_P95_PROFILES)
        or candidate.get("cumulative_budget_debit") != V210_CUMULATIVE_DEBIT.to_dict()
        or policy.get("imported_cell_count") != 16
        or policy.get("provider_construction_during_import") is not False
        or policy.get("provider_calls_during_import") != 0
        or policy.get("stage0_selected_profile_id") != "nu-0.5"
        or policy.get("offline_candidate_admission_cells_imported") != 0
    ):
        raise PilotV210ParentImportError(
            "V2.10 source manifest authority/claim boundary drifted"
        )
    for profile_id in V210_ALLOWED_P95_PROFILES:
        source = _mapping(
            p95[profile_id],
            name=f"V2.10 source p95 {profile_id}",
        )
        normalized = normalize_v29_observed_p95_binding(
            _mapping(
                source.get("v2_8_observed_p95_origin"),
                name=f"V2.10 source p95 origin {profile_id}",
            ),
            profile_id=profile_id,
        )
        if source.get("normalized_v2_9_binding") != normalized:
            raise PilotV210ParentImportError(
                f"V2.10 source p95 normalization drifted: {profile_id}"
            )
    return candidate


def write_v210_source_manifest_draft(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    value = _validate_source_manifest_structure(manifest)
    target = Path(path)
    if target.exists():
        raise PilotV210ParentImportError(
            f"refusing to overwrite V2.10 source manifest: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
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
    target.write_bytes(raw)
    return target


def load_v210_source_manifest(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
    expected_content_sha256: str | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise PilotV210ParentImportError("V2.10 source manifest is unavailable")
    raw = source.read_bytes()
    try:
        value = _strict_json(raw, name="V2.10 source manifest")
    except PilotV24ParentImportError as exc:
        raise _translate(exc) from exc
    if expected_file_sha256 is not None and _sha256(raw) != expected_file_sha256:
        raise PilotV210ParentImportError("V2.10 source manifest file hash drifted")
    candidate = _validate_source_manifest_structure(value)
    if (
        expected_content_sha256 is not None
        and candidate["integrity"]["content_sha256"] != expected_content_sha256
    ):
        raise PilotV210ParentImportError("V2.10 source manifest content hash drifted")
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
        raise PilotV210ParentImportError(
            "V2.10 source manifest is not canonical pretty JSON"
        )
    return candidate


def validate_v210_source_manifest(
    source_manifest: Mapping[str, Any],
    *,
    parent_repo_root: str | Path,
    evidence_repo_root: str | Path,
    target_contract: PilotContract,
) -> dict[str, Any]:
    selected = _validate_source_manifest_structure(source_manifest)
    rebuilt = build_v210_source_manifest(
        parent_repo_root=parent_repo_root,
        evidence_repo_root=evidence_repo_root,
        target_contract=target_contract,
    )
    if selected != rebuilt:
        raise PilotV210ParentImportError(
            "V2.10 source manifest differs from immutable V2.9 authority"
        )
    return selected


def _tracked_source_manifest_binding(
    child_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _, raw, value = _strict_file(
        child_root,
        V210_SOURCE_MANIFEST_PATH,
        name="tracked V2.10 source manifest",
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
        raise PilotV210ParentImportError(
            "tracked V2.10 source manifest is not canonical JSON"
        )
    return manifest, {
        "path": V210_SOURCE_MANIFEST_PATH.as_posix(),
        "schema_version": V210_SOURCE_MANIFEST_SCHEMA_VERSION,
        "file_sha256": _sha256(raw),
        "content_sha256": manifest["integrity"]["content_sha256"],
    }


def _copy_exact_v29_snapshot(
    *,
    source_root: Path,
    destination_root: Path,
    child_repo_root: Path,
    inventory: Sequence[Mapping[str, Any]],
) -> None:
    for row in inventory:
        try:
            relative = _normalized_relative(
                str(row.get("path", "")),
                required_top=None,
                name="V2.9 raw inventory path",
            )
            _, raw = _guarded_file(
                source_root,
                relative,
                name=f"V2.9 raw {relative.as_posix()}",
            )
            _atomic_exact_bytes_no_follow(
                repo_root=child_repo_root,
                path=destination_root.joinpath(*relative.parts),
                raw=raw,
            )
        except (PilotV24ParentImportError, PilotV27Stage0ImportError) as exc:
            raise _translate(exc) from exc
        if len(raw) != row.get("byte_size") or _sha256(raw) != row.get("sha256"):
            raise PilotV210ParentImportError(
                f"V2.9 raw changed during copy: {relative.as_posix()}"
            )
    _, copied = _inventory(destination_root, declared_root=V29_RAW_ROOT)
    if copied != {
        "root": V29_RAW_ROOT.as_posix(),
        "inventory_schema_version": V29_RAW_INVENTORY_SCHEMA_VERSION,
        "inventory_canonicalization": V29_RAW_INVENTORY_CANONICALIZATION,
        "file_count": V29_RAW_FILE_COUNT,
        "storage_bytes": V29_RAW_STORAGE_BYTES,
        "inventory_sha256": V29_RAW_INVENTORY_SHA256,
    }:
        raise PilotV210ParentImportError(
            "copied V2.9 raw snapshot differs from its source"
        )


def _build_v210_parent_import_receipt(
    *,
    child_root: Path,
    child_raw: Path,
    contract: PilotContract,
    child_git_commit: str,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _validate_source_manifest_structure(source_manifest)
    _, snapshot = _inventory(
        imported_v29_raw_root_v210(child_raw),
        declared_root=V29_RAW_ROOT,
    )
    expected_snapshot = {
        "root": V29_RAW_ROOT.as_posix(),
        "inventory_schema_version": V29_RAW_INVENTORY_SCHEMA_VERSION,
        "inventory_canonicalization": V29_RAW_INVENTORY_CANONICALIZATION,
        "file_count": V29_RAW_FILE_COUNT,
        "storage_bytes": V29_RAW_STORAGE_BYTES,
        "inventory_sha256": V29_RAW_INVENTORY_SHA256,
    }
    if snapshot != expected_snapshot:
        raise PilotV210ParentImportError("V2.10 parent snapshot inventory drifted")
    p95: dict[str, Any] = {}
    for profile_id in V210_ALLOWED_P95_PROFILES:
        receipt_path = v210_observed_p95_receipt_path(
            child_raw,
            profile_id,
        )
        projection_path = v210_observed_p95_projection_path(
            child_raw,
            profile_id,
        )
        p95[profile_id] = {
            "authority": verified_v210_observed_p95_authority_binding(
                receipt_path,
                repo_root=child_root,
                raw_root=child_raw,
                expected_git_commit=child_git_commit,
                contract=contract,
            ),
            "projection": verified_v210_observed_p95_projection_binding(
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
        raise PilotV210ParentImportError(
            "selected V2.10 source manifest differs from tracked authority"
        )
    return _seal(
        {
            "schema_version": V210_PARENT_IMPORT_SCHEMA_VERSION,
            "contract": _contract_binding(child_root, contract),
            "git": {
                "tag": V210_SCIENCE_TAG,
                "commit": child_git_commit,
            },
            "source_manifest": manifest_binding,
            "source_parent": {
                "contract_id": V29_CONTRACT_ID,
                "contract_sha256": V29_CONTRACT_CANONICAL_SHA256,
                "science_tag": V29_SCIENCE_TAG,
                "science_commit": V29_SCIENCE_COMMIT,
                "raw_root": V29_RAW_ROOT.as_posix(),
                "terminal_status": "complete-with-no-go",
                "status_counts": {"complete": 26, "failed": 185},
                "implementation_root_cause": "imported-p95-runner-binding-shape-mismatch",
                "scientific_complete": False,
            },
            "copied_snapshot": {
                "path": (V210_RAW_ROOT / V210_SNAPSHOT_RELATIVE).as_posix(),
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
            "cumulative_budget_debit": V210_CUMULATIVE_DEBIT.to_dict(),
            "provider_construction_during_import": False,
            "provider_calls_during_import": 0,
            "hosted_provider_calls_during_import": 0,
            "hosted_cost_usd_during_import": 0.0,
            "scientific_evidence": False,
            "evidence_use": (
                "immutable prerequisite provenance and prospective p95 "
                "authority only; no V2.10 A-D treatment effect"
            ),
        }
    )


def persist_v210_parent_import(
    *,
    parent_repo_root: str | Path,
    evidence_repo_root: str | Path,
    child_repo_root: str | Path,
    child_raw_root: str | Path,
    contract: PilotContract,
    child_git_commit: str,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy V2.9, reseal p95, and write a zero-provider parent receipt."""

    parent = _strict_root(parent_repo_root, name="V2.9 source repository")
    child = _strict_root(child_repo_root, name="V2.10 child repository")
    child_raw = Path(child_raw_root).absolute()
    if child_raw != child.joinpath(*V210_RAW_ROOT.parts):
        raise PilotV210ParentImportError(
            "V2.10 child raw root differs from contract namespace"
        )
    _validate_target_contract(contract, require_frozen=True)
    _verify_release_identity(
        child,
        tag=V210_SCIENCE_TAG,
        expected_commit=child_git_commit,
        name="V2.10 child",
    )
    selected = validate_v210_source_manifest(
        source_manifest,
        parent_repo_root=parent,
        evidence_repo_root=evidence_repo_root,
        target_contract=contract,
    )
    tracked, _ = _tracked_source_manifest_binding(child)
    if tracked != selected:
        raise PilotV210ParentImportError(
            "tracked V2.10 source manifest differs before import"
        )
    audit = verify_v29_terminal_source(
        parent_repo_root=parent,
        evidence_repo_root=evidence_repo_root,
        target_contract=contract,
    )
    snapshot = imported_v29_raw_root_v210(child_raw)
    _copy_exact_v29_snapshot(
        source_root=Path(audit["source_raw_root"]),
        destination_root=snapshot,
        child_repo_root=child,
        inventory=audit["raw_inventory"]["rows"],
    )
    for profile_id in V210_ALLOWED_P95_PROFILES:
        persist_v210_resealed_observed_p95_authority(
            repo_root=child,
            contract=contract,
            raw_root=child_raw,
            profile_id=profile_id,
            expected_git_commit=child_git_commit,
            verified_v2_9_source_binding=audit["p95_sources"][profile_id],
        )
    receipt = _build_v210_parent_import_receipt(
        child_root=child,
        child_raw=child_raw,
        contract=contract,
        child_git_commit=child_git_commit,
        source_manifest=selected,
    )
    receipt_path = child_raw / "parent-import" / "parent_import_receipt.json"
    _atomic_json(
        repo_root=child,
        path=receipt_path,
        value=receipt,
    )
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
        "scientific_evidence": False,
        "v2_9_terminal_no_go_preserved": True,
    }


def verify_v210_parent_import_receipt(
    *,
    receipt_path: str | Path,
    child_repo_root: str | Path,
    contract: PilotContract,
    expected_git_commit: str,
) -> dict[str, Any]:
    child = _strict_root(child_repo_root, name="V2.10 child repository")
    _validate_target_contract(contract, require_frozen=True)
    _verify_release_identity(
        child,
        tag=V210_SCIENCE_TAG,
        expected_commit=expected_git_commit,
        name="V2.10 child",
    )
    path = Path(receipt_path).absolute()
    expected = child.joinpath(
        *V210_RAW_ROOT.parts,
        "parent-import",
        "parent_import_receipt.json",
    )
    if path != expected:
        raise PilotV210ParentImportError(
            "V2.10 parent receipt path differs from contract namespace"
        )
    relative = V210_RAW_ROOT / "parent-import/parent_import_receipt.json"
    _, _, value = _strict_file(
        child,
        relative,
        name="V2.10 parent import receipt",
    )
    _verify_self_hashed(
        value,
        schema_version=V210_PARENT_IMPORT_SCHEMA_VERSION,
        name="V2.10 parent import receipt",
    )
    manifest, _ = _tracked_source_manifest_binding(child)
    rebuilt = _build_v210_parent_import_receipt(
        child_root=child,
        child_raw=child.joinpath(*V210_RAW_ROOT.parts),
        contract=contract,
        child_git_commit=expected_git_commit,
        source_manifest=manifest,
    )
    if value != rebuilt:
        raise PilotV210ParentImportError(
            "V2.10 parent receipt differs from copied/resealed authority"
        )
    return value


def source_binding_for_target_v210(
    source_manifest: Mapping[str, Any],
    target_spec: PilotRunSpec | Mapping[str, Any] | str,
) -> dict[str, Any]:
    rows = source_manifest.get("imported_complete_cells")
    if not isinstance(rows, list) or len(rows) != 16:
        raise PilotV210ParentImportError(
            "V2.10 source manifest lacks 16 imported prerequisites"
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
        raise PilotV210ParentImportError(
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
            raise PilotV210ParentImportError(
                "target spec differs from source-manifest binding"
            )
    return result


def imported_prerequisite_path_v210(
    child_raw_root: str | Path,
    source_manifest: Mapping[str, Any],
    target_spec: PilotRunSpec | Mapping[str, Any] | str,
    artifact_key: str,
) -> Path:
    binding = source_binding_for_target_v210(source_manifest, target_spec)
    artifacts = _mapping(
        binding.get("source_artifacts"),
        name="V2.10 imported prerequisite artifacts",
    )
    artifact = artifacts.get(artifact_key)
    if isinstance(artifact, Mapping):
        source_path = artifact.get("path")
    elif artifact_key == "run_root" and isinstance(artifact, str):
        source_path = artifact
    else:
        raise PilotV210ParentImportError(
            f"imported prerequisite lacks artifact {artifact_key!r}"
        )
    if not isinstance(source_path, str):
        raise PilotV210ParentImportError(
            f"imported prerequisite artifact {artifact_key!r} has no path"
        )
    return snapshot_path_for_v29_source_artifact_v210(
        child_raw_root,
        source_path,
    )


def verified_v210_imported_prerequisite_binding(
    child_raw_root: str | Path,
    source_manifest: Mapping[str, Any],
    target_spec: PilotRunSpec | Mapping[str, Any] | str,
) -> dict[str, Any]:
    """Verify and flatten one copied V2.9 prerequisite for V2.10 replay.

    The returned paths are child-local paths inside the exact V2.9 snapshot;
    no V2.9 target ID is relabelled.  The orchestrator is expected to write a
    new V2.10 envelope/terminal summary around this binding.
    """

    binding = source_binding_for_target_v210(source_manifest, target_spec)
    snapshot = imported_v29_raw_root_v210(child_raw_root)
    _, summary = _inventory(snapshot, declared_root=V29_RAW_ROOT)
    if summary != {
        "root": V29_RAW_ROOT.as_posix(),
        "inventory_schema_version": V29_RAW_INVENTORY_SCHEMA_VERSION,
        "inventory_canonicalization": V29_RAW_INVENTORY_CANONICALIZATION,
        "file_count": V29_RAW_FILE_COUNT,
        "storage_bytes": V29_RAW_STORAGE_BYTES,
        "inventory_sha256": V29_RAW_INVENTORY_SHA256,
    }:
        raise PilotV210ParentImportError("copied V2.9 prerequisite snapshot drifted")
    verified: dict[str, Any] = {}
    artifacts = _mapping(
        binding.get("source_artifacts"),
        name="V2.10 imported prerequisite artifacts",
    )
    for key, declared in artifacts.items():
        if key == "run_root":
            if not isinstance(declared, str):
                raise PilotV210ParentImportError(
                    "imported prerequisite run_root is malformed"
                )
            path = snapshot_path_for_v29_source_artifact_v210(
                child_raw_root,
                declared,
            )
            if path.is_symlink() or not path.is_dir():
                raise PilotV210ParentImportError(
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
            raise PilotV210ParentImportError(
                f"imported prerequisite {key} lacks a source path"
            )
        path = snapshot_path_for_v29_source_artifact_v210(
            child_raw_root,
            source_path,
        )
        if path.is_symlink() or not path.is_file():
            raise PilotV210ParentImportError(
                f"imported prerequisite {key} is unavailable"
            )
        raw = path.read_bytes()
        if _sha256(raw) != declared_map.get("file_sha256") or len(
            raw
        ) != declared_map.get("byte_size", len(raw)):
            raise PilotV210ParentImportError(
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
                if value.get("schema_version") == "finevo-pilot-terminal-summary-v1":
                    _verify_terminal_summary_hash(
                        value,
                        name=f"imported prerequisite {key}",
                    )
                else:
                    _verify_self_hash(
                        value,
                        schema_version=str(value.get("schema_version", "")),
                        name=f"imported prerequisite {key}",
                    )
            except PilotV24ParentImportError as exc:
                raise _translate(exc) from exc
            if (
                value.get("integrity", {}).get("content_sha256")
                != declared_map["content_sha256"]
            ):
                raise PilotV210ParentImportError(
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
        },
        "provider_construction_during_verification": False,
        "provider_calls_during_verification": 0,
        "treatment_effect_evidence": False,
    }
    if binding["stage_id"] == "q-ref-resolution":
        qref_path = Path(verified["q_ref_resolution"]["snapshot_path"])
        qref = _strict_json(
            qref_path.read_bytes(),
            name="imported V2.9 q-ref resolution",
        )
        if (
            qref.get("schema_version") != "finevo-q-ref-resolution-v1"
            or qref.get("status") != "pass"
            or qref.get("q_ref") != 63.50397933257746
            or qref.get("row_count") != 48
        ):
            raise PilotV210ParentImportError("imported V2.9 q-ref semantics drifted")
        result["q_ref"] = 63.50397933257746
        result["q_ref_resolution"] = _json_copy(qref)
    if binding["stage_id"] == "stage0-calibration":
        selection_path = Path(verified["stage0_selection"]["snapshot_path"])
        selection = _strict_json(
            selection_path.read_bytes(),
            name="imported V2.9 Stage-0 selection",
        )
        if (
            selection.get("schema_version") != "finevo-stage0-selection-v1"
            or selection.get("contract_sha256") != V29_CONTRACT_CANONICAL_SHA256
            or selection.get("selected_profile_id") != "nu-0.5"
        ):
            raise PilotV210ParentImportError("imported V2.9 Stage-0 winner drifted")
        terminal_path = Path(terminal["snapshot_path"])
        terminal_value = _strict_json(
            terminal_path.read_bytes(),
            name="imported V2.9 Stage-0 terminal",
        )
        result["metrics"] = _json_copy(
            _mapping(
                terminal_value.get("payload"),
                name="imported Stage-0 payload",
            ).get("metrics")
        )
        result["selected_profile_id"] = "nu-0.5"
    return result


def v210_imported_v29_run_dir(
    child_raw_root: str | Path,
    source_manifest: Mapping[str, Any],
    target_spec: PilotRunSpec | Mapping[str, Any] | str,
) -> Path:
    return imported_prerequisite_path_v210(
        child_raw_root,
        source_manifest,
        target_spec,
        "run_root",
    )


def parent_budget_debit_for_v210(
    contract: PilotContract,
) -> ParentBudgetDebit | None:
    if getattr(contract, "contract_id", None) != V210_CONTRACT_ID:
        return None
    _validate_target_contract(contract, require_frozen=False)
    return V210_CUMULATIVE_DEBIT


__all__ = [
    "PilotV210ParentImportError",
    "V210_ALLOWED_P95_PROFILES",
    "V210_CONTRACT_ID",
    "V210_CUMULATIVE_DEBIT",
    "V210_EXPANDED_CONTRACT_PATH",
    "V210_PARENT_IMPORT_SCHEMA_VERSION",
    "V210_RAW_ROOT",
    "V210_RESEALED_P95_AUTHORITY_SCHEMA_VERSION",
    "V210_RESEALED_P95_SOURCE_KIND",
    "V210_SCIENCE_TAG",
    "V210_SNAPSHOT_RELATIVE",
    "V210_SOURCE_MANIFEST_PATH",
    "V210_SOURCE_MANIFEST_SCHEMA_VERSION",
    "V29_RAW_FILE_COUNT",
    "V29_RAW_INVENTORY_SHA256",
    "V29_RAW_STORAGE_BYTES",
    "V29_SCIENCE_COMMIT",
    "V29_SCIENCE_TAG",
    "build_v210_source_manifest",
    "build_v210_prerequisite_bindings",
    "build_v210_resealed_observed_p95_authority",
    "imported_prerequisite_path_v210",
    "imported_v29_raw_root_v210",
    "load_v210_source_manifest",
    "normalize_v29_observed_p95_binding",
    "parent_budget_debit_for_v210",
    "persist_v210_parent_import",
    "persist_v210_resealed_observed_p95_authority",
    "snapshot_path_for_v29_source_artifact_v210",
    "source_binding_for_target_v210",
    "validate_v210_source_manifest",
    "v2_9_p95_source_binding_v210",
    "v210_imported_v29_run_dir",
    "v210_observed_p95_projection_path",
    "v210_observed_p95_receipt_path",
    "v210_v29_snapshot_root",
    "verified_v210_imported_prerequisite_binding",
    "verified_v210_observed_p95_authority_binding",
    "verified_v210_observed_p95_projection_binding",
    "verify_v210_parent_import_receipt",
    "verify_v210_resealed_observed_p95_authority",
    "verify_v210_resealed_observed_p95_projection",
    "verify_v29_terminal_source",
    "write_v210_source_manifest_draft",
]
