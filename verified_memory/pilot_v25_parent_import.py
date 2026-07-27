"""Zero-provider V2.5 retry of the terminal V2.4 parent-import stage.

V2.4 is immutable and remains a 211-cell ``complete-with-no-go`` release.
V2.5 first verifies that tracked terminal package, then revalidates the
original V2.3 observed-p95 source chain with the explicit historical
checkpoint policy.  Only the resulting budget debit and two permitted p95
authorities enter the fresh V2.5 raw namespace.

The V2.4 adapter is deliberately not reused as a V2.5 receipt verifier: every
child receipt, projection, and import receipt has a V2.5-specific schema and
is bound to the V2.5 contract/tag/commit.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from .observed_p95_authority import (
    HistoricalCheckpointVerificationPolicy,
    ObservedP95AuthorityError,
    verified_observed_p95_authority_binding,
)
from .pilot_budget import ParentBudgetDebit
from .pilot_contract import PilotContract
from .pilot_v24_parent_import import (
    PilotV24ParentImportError,
    V24_ALLOWED_P95_PROFILES,
    V24_BOUNDARY_ONLY_P95_PROFILES,
    V24_PARENT_SOURCE_MANIFEST_CONTENT_SHA256,
    V24_PARENT_SOURCE_MANIFEST_PATH,
    _atomic_exact_bytes,
    _atomic_exact_json,
    _bound_content_sha256,
    _guarded_file,
    _json_copy,
    _load_source_manifest as _load_v23_source_manifest_via_v24,
    _normalized_relative,
    _repo_relative,
    _real_root,
    _seal,
    _sha256,
    _strict_json,
    _verify_parent_bound_files,
    _verify_parent_contract,
    _verify_parent_git,
    _verify_parent_ledgers,
)


V25_CONTRACT_ID = "finevo-pilot-v2.5"
V25_SCIENCE_TAG = "pilot-v2.5-science"
V25_PARENT_IMPORT_SCHEMA_VERSION = "finevo-pilot-v2.5-parent-import-v1"
V25_INHERITED_P95_RECEIPT_SCHEMA_VERSION = (
    "finevo-pilot-v2.5-inherited-observed-p95-authority-v1"
)
V25_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.5-source-manifest-v1"
V25_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_5_source_manifest.json"
)
V25_EXPANDED_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_5.yaml")
V25_RAW_ROOT = PurePosixPath("experiment_results/pilot-v2.5/raw")
V25_PARENT_DEBIT_RECORD_SHA256 = (
    "002d0c224af866a2f0f26b5685c22ee850e2a78402012957fe31813d44f00ed9"
)
V25_ALLOWED_P95_PROFILES = V24_ALLOWED_P95_PROFILES
V25_BOUNDARY_ONLY_P95_PROFILES = V24_BOUNDARY_ONLY_P95_PROFILES

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANONICALIZATION = "json-sort-keys-utf8-v1"
_BOUNDARY_REASON = (
    "V2.3 boundary-only model; V2.5 redispatch remains forbidden"
)
_CLAIM_BOUNDARY = (
    "This V2.5 receipt preserves the V2.4 terminal no-go and imports only "
    "V2.3 budget/p95 authority. It contains no V2.4 or V2.5 treatment outcome."
)


class PilotV25ParentImportError(RuntimeError):
    """Raised before V2.3 authority can enter the fresh V2.5 raw tree."""


def _translate_v24(exc: PilotV24ParentImportError) -> PilotV25ParentImportError:
    return PilotV25ParentImportError(str(exc))


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotV25ParentImportError(f"{name} must be an object")
    return value


def _load_v25_source_manifest(
    repo_root: Path,
) -> tuple[dict[str, Any], bytes]:
    try:
        _, raw = _guarded_file(
            repo_root,
            V25_SOURCE_MANIFEST_PATH,
            name="V2.5 source manifest",
        )
        value = _strict_json(raw, name="V2.5 source manifest")
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    if value.get("schema_version") != V25_SOURCE_MANIFEST_SCHEMA_VERSION:
        raise PilotV25ParentImportError("V2.5 source manifest schema drifted")
    integrity = _mapping(
        value.get("integrity"),
        name="V2.5 source manifest integrity",
    )
    if (
        set(integrity) != {"canonicalization", "content_sha256"}
        or integrity.get("canonicalization") != _CANONICALIZATION
        or _SHA256_RE.fullmatch(str(integrity.get("content_sha256", "")))
        is None
        or _bound_content_sha256(value) != integrity["content_sha256"]
    ):
        raise PilotV25ParentImportError(
            "V2.5 source manifest self-hash or canonicalization drifted"
        )
    return value, raw


def _retry_amendment(contract: PilotContract) -> Mapping[str, Any]:
    amendment = getattr(contract, "parent_import_retry_amendment", None)
    if not isinstance(amendment, Mapping):
        raise PilotV25ParentImportError(
            "V2.5 contract lacks its parent-import retry amendment"
        )
    return amendment


def _validate_child_contract(
    contract: PilotContract,
    *,
    source_manifest: Mapping[str, Any],
    source_manifest_raw: bytes,
    require_frozen: bool = True,
) -> None:
    allowed_status = {"frozen"} if require_frozen else {"draft", "frozen"}
    if (
        contract.contract_id != V25_CONTRACT_ID
        or contract.status not in allowed_status
        or contract.implementation.get("required_git_tag") != V25_SCIENCE_TAG
    ):
        raise PilotV25ParentImportError(
            "parent import requires the V2.5 retry science contract"
        )
    amendment = _retry_amendment(contract)
    binding = _mapping(
        amendment.get("source_manifest"),
        name="V2.5 retry source-manifest binding",
    )
    expected_binding = {
        "path": V25_SOURCE_MANIFEST_PATH.as_posix(),
        "schema_version": V25_SOURCE_MANIFEST_SCHEMA_VERSION,
        "file_sha256": _sha256(source_manifest_raw),
        "content_sha256": source_manifest["integrity"]["content_sha256"],
    }
    if _json_copy(binding) != expected_binding:
        raise PilotV25ParentImportError(
            "V2.5 retry amendment does not bind its exact source manifest"
        )
    retry = _mapping(
        amendment.get("retry_policy"),
        name="V2.5 retry policy",
    )
    raw_namespace = _mapping(
        amendment.get("raw_namespace"),
        name="V2.5 raw namespace policy",
    )
    if (
        retry.get("eligible_stage_ids") != ("parent-import",)
        and retry.get("eligible_stage_ids") != ["parent-import"]
    ):
        raise PilotV25ParentImportError(
            "V2.5 retry policy permits an unexpected stage"
        )
    if (
        retry.get("v2_4_raw_resume") != "forbidden"
        or retry.get("provider_redispatch_before_import_success")
        != "forbidden"
        or retry.get("downstream_dispatch_requires_import_success") is not True
        or retry.get("outcome_blind") is not True
        or raw_namespace.get("parent")
        != "experiment_results/pilot-v2.4/raw"
        or raw_namespace.get("child") != V25_RAW_ROOT.as_posix()
        or raw_namespace.get("shared") is not False
    ):
        raise PilotV25ParentImportError(
            "V2.5 retry or raw-namespace policy drifted"
        )


def _load_v23_source_manifest(
    child_root: Path,
    *,
    v25_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    try:
        manifest, raw = _load_v23_source_manifest_via_v24(child_root)
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    authority = _mapping(
        v25_manifest.get("v2_3_authority_parent"),
        name="V2.5 V2.3 authority parent",
    )
    binding = _mapping(
        authority.get("source_manifest"),
        name="V2.5 V2.3 source-manifest binding",
    )
    expected = {
        "path": V24_PARENT_SOURCE_MANIFEST_PATH.as_posix(),
        "schema_version": manifest["schema_version"],
        "file_sha256": _sha256(raw),
        "content_sha256": manifest["integrity"]["content_sha256"],
    }
    if _json_copy(binding) != expected:
        raise PilotV25ParentImportError(
            "V2.5 source manifest does not bind the exact V2.3 authority manifest"
        )
    parent = _mapping(manifest.get("parent"), name="V2.3 parent release")
    compact_debit = _mapping(
        authority.get("cumulative_budget_debit"),
        name="V2.5 V2.3 cumulative debit binding",
    )
    parent_debit = _mapping(
        manifest.get("cumulative_budget_debit"),
        name="V2.3 cumulative debit record",
    )
    expected_compact_debit = {
        field: parent_debit.get(field)
        for field in (
            "cost_usd",
            "hosted_completions",
            "storage_bytes",
            "record_sha256",
        )
    }
    if (
        authority.get("contract_id") != parent.get("contract_id")
        or authority.get("contract_sha256")
        != parent.get("contract_canonical_sha256")
        or authority.get("science_tag") != parent.get("science_tag")
        or authority.get("science_tag_object")
        != parent.get("science_tag_object")
        or authority.get("science_commit") != parent.get("science_commit")
        or set(compact_debit) != set(expected_compact_debit)
        or _json_copy(compact_debit) != expected_compact_debit
    ):
        raise PilotV25ParentImportError(
            "V2.5 V2.3 release or cumulative debit binding drifted"
        )
    expected_sources = _mapping(
        authority.get("observed_p95_sources"),
        name="V2.5 observed-p95 source inventory",
    )
    if set(expected_sources) != set(V25_ALLOWED_P95_PROFILES):
        raise PilotV25ParentImportError(
            "V2.5 dispatch-authority source inventory drifted"
        )
    parent_sources = _mapping(
        manifest.get("observed_p95_sources"),
        name="V2.3 observed-p95 source inventory",
    )
    for profile_id, expected_row in expected_sources.items():
        source_row = _mapping(
            parent_sources.get(profile_id),
            name=f"{profile_id} V2.3 p95 source",
        )
        if _json_copy(expected_row) != {
            "receipt_file_sha256": source_row["file_sha256"],
            "receipt_content_sha256": source_row["content_sha256"],
            "runtime_model": source_row["runtime_model"],
            "served_model": source_row["served_model"],
        }:
            raise PilotV25ParentImportError(
                f"V2.5 V2.3 p95 source binding drifted for {profile_id}"
            )
    return manifest, raw


def _verify_v24_terminal_evidence(
    child_root: Path,
    *,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the tracked V2.4 no-go before touching V2.3 authority."""

    terminal = _mapping(
        source_manifest.get("v2_4_terminal_parent"),
        name="V2.4 terminal parent",
    )
    release = _mapping(terminal.get("release"), name="V2.4 release")
    tag = str(release.get("science_tag", ""))
    tag_ref = f"refs/tags/{tag}"
    try:
        from .pilot_v24_parent_import import _git

        if (
            _git(child_root, "cat-file", "-t", tag_ref) != "tag"
            or _git(child_root, "rev-parse", tag_ref)
            != release.get("science_tag_object")
            or _git(child_root, "rev-parse", f"{tag_ref}^{{commit}}")
            != release.get("science_commit")
        ):
            raise PilotV25ParentImportError(
                "V2.4 annotated release tag or peeled commit drifted"
            )
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc

    contract_row = _mapping(
        terminal.get("contract"),
        name="V2.4 terminal contract binding",
    )
    try:
        contract_path, contract_raw = _guarded_file(
            child_root,
            _normalized_relative(
                contract_row.get("path", ""),
                required_top="experiments",
                name="V2.4 contract path",
            ),
            name="V2.4 terminal contract",
        )
        contract_value = _strict_json(
            contract_raw,
            name="V2.4 terminal contract",
        )
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    v24_contract = PilotContract.from_dict(contract_value)
    if (
        _sha256(contract_raw) != contract_row.get("file_sha256")
        or v24_contract.contract_id != contract_row.get("contract_id")
        or v24_contract.canonical_hash != contract_row.get("canonical_sha256")
        or v24_contract.status != "frozen"
        or v24_contract.implementation.get("required_git_tag") != tag
        or contract_path
        != child_root.joinpath(*PurePosixPath(contract_row["path"]).parts)
    ):
        raise PilotV25ParentImportError(
            "V2.4 frozen contract or release binding drifted"
        )

    published = _mapping(
        source_manifest.get("v2_4_published_evidence"),
        name="V2.4 published evidence",
    )
    if (
        published.get("status") != "complete-with-no-go"
        or published.get("scientific_complete") is not False
        or published.get("scientific_matrix_complete") is not False
        or published.get("registered_cells") != 211
        or published.get("terminal_cells") != 211
    ):
        raise PilotV25ParentImportError(
            "V2.4 published evidence was reinterpreted as scientific evidence"
        )
    evidence_values: dict[str, dict[str, Any]] = {}
    evidence_raw: dict[str, bytes] = {}
    for name in (
        "package_manifest",
        "checksums",
        "aggregate",
        "failure_ledger",
        "copied_parent_source_manifest",
    ):
        row = _mapping(
            published.get(name),
            name=f"V2.4 published {name} binding",
        )
        try:
            _, raw = _guarded_file(
                child_root,
                _normalized_relative(
                    row.get("path", ""),
                    required_top="evidence",
                    name=f"V2.4 published {name} path",
                ),
                name=f"V2.4 published {name}",
            )
        except PilotV24ParentImportError as exc:
            raise _translate_v24(exc) from exc
        if _sha256(raw) != row.get("file_sha256"):
            raise PilotV25ParentImportError(
                f"V2.4 published {name} file hash drifted"
            )
        evidence_raw[name] = raw
        if name != "copied_parent_source_manifest":
            try:
                evidence_values[name] = _strict_json(
                    raw,
                    name=f"V2.4 published {name}",
                )
            except PilotV24ParentImportError as exc:
                raise _translate_v24(exc) from exc

    package = evidence_values["package_manifest"]
    aggregate = evidence_values["aggregate"]
    failure_ledger = evidence_values["failure_ledger"]
    checksums = evidence_values["checksums"]
    denominator = _mapping(
        aggregate.get("denominator"),
        name="V2.4 aggregate denominator",
    )
    if (
        package.get("contract_id") != v24_contract.contract_id
        or package.get("contract_sha256") != v24_contract.canonical_hash
        or package.get("pilot_tag") != tag
        or package.get("publication_status") != "complete-with-no-go"
        or package.get("scientific_complete") is not False
        or package.get("scientific_matrix_complete") is not False
        or package.get("scientific_claim_gates_supported") is not False
        or aggregate.get("contract_id") != v24_contract.contract_id
        or aggregate.get("contract_sha256") != v24_contract.canonical_hash
        or aggregate.get("publication_status") != "complete-with-no-go"
        or aggregate.get("scientific_complete") is not False
        or aggregate.get("scientific_matrix_complete") is not False
        or denominator.get("expected_count") != 211
        or denominator.get("observed_ledger_count") != 211
        or denominator.get("all_rows_present") is not True
        or denominator.get("all_rows_terminal") is not True
        or denominator.get("status_counts") != {"integrity-stopped": 211}
    ):
        raise PilotV25ParentImportError(
            "V2.4 no-go package or terminal denominator drifted"
        )
    rows = failure_ledger.get("rows")
    if (
        failure_ledger.get("contract_sha256") != v24_contract.canonical_hash
        or not isinstance(rows, list)
        or len(rows) != 211
        or any(row.get("status") != "integrity-stopped" for row in rows)
    ):
        raise PilotV25ParentImportError(
            "V2.4 failure ledger does not preserve all 211 terminal cells"
        )
    direct = [row for row in rows if row.get("stage_id") == "parent-import"]
    failure_binding = _mapping(
        terminal.get("parent_import_failure"),
        name="V2.4 parent-import failure binding",
    )
    if (
        len(direct) != 1
        or direct[0].get("failure", {}).get("error_type")
        != failure_binding.get("error_type")
        or direct[0].get("failure", {}).get("message_sha256")
        != failure_binding.get("failure_receipt", {}).get("message_sha256")
        or failure_binding.get("provider_calls") != 0
        or failure_binding.get("scientific_evidence") is not False
        or failure_binding.get("scientific_effect_outcomes_available")
        is not False
        or failure_binding.get("scientific_effect_outcomes_inspected")
        is not False
        or any(
            row.get("stage_id") != "parent-import"
            and row.get("failure", {}).get("source_stage") != "parent-import"
            for row in rows
        )
    ):
        raise PilotV25ParentImportError(
            "V2.4 parent-import failure classification drifted"
        )

    package_root = child_root.joinpath(
        *PurePosixPath(str(published["root"])).parts
    )
    checksum_rows = checksums.get("files")
    if (
        checksums.get("contract_sha256") != v24_contract.canonical_hash
        or not isinstance(checksum_rows, list)
        or not checksum_rows
    ):
        raise PilotV25ParentImportError(
            "V2.4 evidence checksum inventory is malformed"
        )
    for row in checksum_rows:
        if not isinstance(row, Mapping):
            raise PilotV25ParentImportError(
                "V2.4 evidence checksum row is malformed"
            )
        relative = PurePosixPath(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise PilotV25ParentImportError(
                "V2.4 evidence checksum path escaped its package"
            )
        path = package_root.joinpath(*relative.parts)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row.get("byte_size")
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != row.get("sha256")
        ):
            raise PilotV25ParentImportError(
                f"V2.4 evidence checksum drifted for {relative.as_posix()}"
            )
    parent_copy = _strict_json(
        evidence_raw["copied_parent_source_manifest"],
        name="V2.4 copied V2.3 source manifest",
    )
    if (
        parent_copy.get("integrity", {}).get("content_sha256")
        != V24_PARENT_SOURCE_MANIFEST_CONTENT_SHA256
    ):
        raise PilotV25ParentImportError(
            "V2.4 evidence copied a different V2.3 source manifest"
        )

    terminal_denominator = _mapping(
        terminal.get("terminal_denominator"),
        name="V2.4 terminal denominator binding",
    )
    if (
        terminal_denominator.get("registered_cells") != 211
        or terminal_denominator.get("terminal_cells") != 211
        or terminal_denominator.get("status_counts")
        != {"integrity-stopped": 211}
        or terminal_denominator.get("terminal_status")
        != "complete-with-no-go"
        or terminal_denominator.get("scientific_complete") is not False
        or terminal_denominator.get("scientific_matrix_complete") is not False
    ):
        raise PilotV25ParentImportError(
            "V2.4 terminal source binding drifted"
        )
    return {
        "contract_id": v24_contract.contract_id,
        "contract_sha256": v24_contract.canonical_hash,
        "science_tag": tag,
        "science_tag_object": release["science_tag_object"],
        "science_commit": release["science_commit"],
        "publication_status": "complete-with-no-go",
        "terminal_cells": 211,
        "status_counts": {"integrity-stopped": 211},
        "provider_calls": 0,
        "incremental_budget_debit": _json_copy(
            terminal["incremental_budget_debit"]
        ),
        "package_manifest_file_sha256": _sha256(
            evidence_raw["package_manifest"]
        ),
        "failure_ledger_file_sha256": _sha256(
            evidence_raw["failure_ledger"]
        ),
    }


def parent_budget_debit_for_v25(
    contract: PilotContract,
    *,
    repo_root: str | Path | None = None,
) -> ParentBudgetDebit | None:
    """Return the V2.3 debit plus V2.4's zero-call storage debit."""

    if contract.contract_id != V25_CONTRACT_ID:
        return None
    root = _real_root(
        repo_root or Path(__file__).resolve().parents[1],
        name="child repository root",
    )
    manifest, raw = _load_v25_source_manifest(root)
    _validate_child_contract(
        contract,
        source_manifest=manifest,
        source_manifest_raw=raw,
        require_frozen=False,
    )
    v23 = _mapping(
        manifest.get("v2_3_authority_parent"),
        name="V2.3 authority parent",
    )
    v24 = _mapping(
        manifest.get("v2_4_terminal_parent"),
        name="V2.4 terminal parent",
    )
    v23_debit = _mapping(
        v23.get("cumulative_budget_debit"),
        name="V2.3 cumulative debit",
    )
    v24_debit = _mapping(
        v24.get("incremental_budget_debit"),
        name="V2.4 incremental debit",
    )
    run_binding = _mapping(
        v24.get("terminal_denominator"),
        name="V2.4 terminal denominator",
    )
    run_ledger = _mapping(
        run_binding.get("run_ledger"),
        name="V2.4 run-ledger binding",
    )
    v23_manifest, _ = _load_v23_source_manifest(
        root,
        v25_manifest=manifest,
    )
    v23_parent_debit = _mapping(
        v23_manifest.get("cumulative_budget_debit"),
        name="V2.3 parent debit record",
    )
    try:
        debit = ParentBudgetDebit(
            parent_contract_sha256=str(
                v24["contract"]["canonical_sha256"]
            ),
            parent_run_ledger_sha256=str(run_ledger["internal_sha256"]),
            parent_budget_ledger_sha256=str(
                v23_parent_debit["parent_budget_ledger_sha256"]
            ),
            stage_bucket="parent_v23",
            cost_usd=float(v23_debit["cost_usd"])
            + float(v24_debit["cost_usd"]),
            hosted_completions=int(v23_debit["hosted_completions"])
            + int(v24_debit["hosted_completions"]),
            storage_bytes=int(v23_debit["storage_bytes"])
            + int(v24_debit["storage_bytes"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PilotV25ParentImportError(
            "V2.5 cumulative parent debit is malformed"
        ) from exc
    if (
        not math.isclose(debit.cost_usd, 3.212770875, abs_tol=1e-12)
        or debit.hosted_completions != 184
        or debit.storage_bytes != 4_714_322
        or debit.record_sha256 != V25_PARENT_DEBIT_RECORD_SHA256
    ):
        raise PilotV25ParentImportError(
            "V2.5 cumulative parent debit differs from the retry amendment"
        )
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


def _snapshot_path(raw_root: Path, profile_id: str) -> Path:
    return (
        raw_root
        / "parent-import"
        / "parent_snapshots"
        / f"{profile_id}.observed_p95_parent.json"
    )


def _child_contract_binding(
    *,
    repo_root: Path,
    contract: PilotContract,
) -> dict[str, Any]:
    try:
        _, raw = _guarded_file(
            repo_root,
            V25_EXPANDED_CONTRACT_PATH,
            name="expanded V2.5 contract",
        )
        parsed = _strict_json(raw, name="expanded V2.5 contract")
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    parsed_contract = PilotContract.from_dict(parsed)
    if (
        parsed_contract.contract_id != contract.contract_id
        or parsed_contract.canonical_hash != contract.canonical_hash
        or parsed_contract.to_dict() != contract.to_dict()
    ):
        raise PilotV25ParentImportError(
            "expanded V2.5 contract differs from the selected contract"
        )
    return {
        "path": V25_EXPANDED_CONTRACT_PATH.as_posix(),
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
    v25_manifest: Mapping[str, Any],
    v23_manifest: Mapping[str, Any],
    v25_manifest_raw: bytes,
    v23_manifest_raw: bytes,
    profile_id: str,
    parent_receipt: Mapping[str, Any],
    parent_snapshot_path: Path,
    parent_snapshot_raw: bytes,
) -> dict[str, Any]:
    if profile_id not in V25_ALLOWED_P95_PROFILES:
        raise PilotV25ParentImportError(
            "boundary-only parent model cannot create V2.5 dispatch authority"
        )
    source_row = v23_manifest["observed_p95_sources"][profile_id]
    parent_model = parent_receipt.get("model")
    parent_reservations = parent_receipt.get("reservations")
    if (
        not isinstance(parent_model, Mapping)
        or parent_model.get("model_id") != profile_id
        or parent_model.get("runtime_model") != source_row["runtime_model"]
        or parent_model.get("served_model") != source_row["served_model"]
        or not isinstance(parent_reservations, Mapping)
        or set(parent_reservations) != {source_row["runtime_model"]}
    ):
        raise PilotV25ParentImportError(
            f"parent p95 model identity drifted for {profile_id}"
        )
    transformed = _json_copy(parent_reservations)
    for call_kind in ("action", "semantic"):
        entry = transformed[source_row["runtime_model"]].get(call_kind)
        if (
            not isinstance(entry, dict)
            or set(entry) != {"authority", "reservation"}
            or not isinstance(entry.get("authority"), dict)
        ):
            raise PilotV25ParentImportError(
                f"parent p95 reservation is malformed for {profile_id}/{call_kind}"
            )
        entry["authority"]["pilot_contract_hash"] = contract.canonical_hash
        entry["authority"]["pilot_tag"] = child_git_tag
    v23_parent = v23_manifest["parent"]
    v24_terminal = v25_manifest["v2_4_terminal_parent"]
    return _seal(
        {
            "schema_version": V25_INHERITED_P95_RECEIPT_SCHEMA_VERSION,
            "contract": _child_contract_binding(
                repo_root=repo_root,
                contract=contract,
            ),
            "git": {
                "tag": child_git_tag,
                "commit": child_git_commit,
            },
            "model": {
                "model_id": profile_id,
                "runtime_model": source_row["runtime_model"],
                "served_model": source_row["served_model"],
            },
            "retry_source": {
                "manifest_path": V25_SOURCE_MANIFEST_PATH.as_posix(),
                "manifest_file_sha256": _sha256(v25_manifest_raw),
                "manifest_content_sha256": v25_manifest["integrity"][
                    "content_sha256"
                ],
                "v2_4_contract_sha256": v24_terminal["contract"][
                    "canonical_sha256"
                ],
                "v2_4_science_tag": v24_terminal["release"]["science_tag"],
                "v2_4_science_commit": v24_terminal["release"][
                    "science_commit"
                ],
                "v2_4_terminal_cells": 211,
                "v2_4_provider_calls": 0,
            },
            "parent_source": {
                "manifest_path": V24_PARENT_SOURCE_MANIFEST_PATH.as_posix(),
                "manifest_file_sha256": _sha256(v23_manifest_raw),
                "manifest_content_sha256": (
                    V24_PARENT_SOURCE_MANIFEST_CONTENT_SHA256
                ),
                "parent_contract_sha256": v23_parent[
                    "contract_canonical_sha256"
                ],
                "parent_git_tag": v23_parent["science_tag"],
                "parent_git_tag_object": v23_parent["science_tag_object"],
                "parent_git_commit": v23_parent["science_commit"],
                "historical_code_binding_verification": (
                    "annotated-tag-peeled-commit-git-tree"
                ),
                "current_exactness_match_required": True,
                "parent_receipt_source_path": source_row["path"],
                "parent_receipt_snapshot_path": _repo_relative(
                    repo_root,
                    parent_snapshot_path,
                    name="V2.5 parent p95 snapshot",
                ),
                "parent_receipt_file_sha256": _sha256(parent_snapshot_raw),
                "parent_receipt_content_sha256": source_row[
                    "content_sha256"
                ],
            },
            "reservations": transformed,
            "scientific_evidence": False,
            "evidence_use": (
                "V2.5 prospective budget authority only. V2.4 remains a "
                "terminal no-go and V2.3 preflight rows are not treatment "
                "effects."
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
    if profile_id not in V25_ALLOWED_P95_PROFILES:
        raise PilotV25ParentImportError(
            "boundary-only parent model cannot create V2.5 dispatch authority"
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
                "source_kind": "v2.3-historical-parent-import-v2.5",
                "source_parent_manifest_content_sha256": (
                    V24_PARENT_SOURCE_MANIFEST_CONTENT_SHA256
                ),
                "source_retry_manifest_content_sha256": child_receipt[
                    "retry_source"
                ]["manifest_content_sha256"],
                "source_authority_receipt": str(child_receipt_path),
                "source_authority_receipt_content_sha256": child_receipt[
                    "integrity"
                ]["content_sha256"],
            },
        }
    )


def _historical_policy(
    parent_root: Path,
    v23_manifest: Mapping[str, Any],
) -> HistoricalCheckpointVerificationPolicy:
    parent = v23_manifest["parent"]
    return HistoricalCheckpointVerificationPolicy(
        source_repo_root=parent_root,
        source_annotated_tag=parent["science_tag"],
        expected_tag_object=parent["science_tag_object"],
        expected_peeled_commit=parent["science_commit"],
    )


def _parent_ledger_receipt_bindings(
    v23_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild the durable V2.3 ledger summaries from the tracked manifest."""

    ledgers = _mapping(
        v23_manifest.get("ledgers"),
        name="V2.3 parent ledger inventory",
    )

    def one(name: str) -> dict[str, Any]:
        row = _mapping(
            ledgers.get(name),
            name=f"V2.3 parent {name} ledger binding",
        )
        internal_sha256 = str(row.get("internal_sha256", ""))
        event_chain_head = str(row.get("event_chain_head", ""))
        event_count = row.get("event_count")
        if (
            _SHA256_RE.fullmatch(internal_sha256) is None
            or _SHA256_RE.fullmatch(event_chain_head) is None
            or isinstance(event_count, bool)
            or not isinstance(event_count, int)
            or event_count <= 0
        ):
            raise PilotV25ParentImportError(
                f"V2.3 parent {name} ledger summary is malformed"
            )
        return {
            "ledger_sha256": internal_sha256,
            "event_count": event_count,
            "event_chain_head": event_chain_head,
        }

    return one("run"), one("budget")


def _verified_parent_snapshot(
    *,
    repo_root: Path,
    raw_root: Path,
    profile_id: str,
    source_row: Mapping[str, Any],
) -> tuple[Path, bytes, dict[str, Any]]:
    snapshot_path = _snapshot_path(raw_root, profile_id)
    snapshot_relative_text = _repo_relative(
        repo_root,
        snapshot_path,
        name=f"{profile_id} V2.3 p95 snapshot",
    )
    snapshot_relative = _normalized_relative(
        snapshot_relative_text,
        required_top="experiment_results",
        name=f"{profile_id} V2.3 p95 snapshot path",
    )
    try:
        guarded_path, raw = _guarded_file(
            repo_root,
            snapshot_relative,
            name=f"{profile_id} V2.3 p95 snapshot",
        )
        receipt = _strict_json(
            raw,
            name=f"{profile_id} V2.3 p95 snapshot",
        )
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    model = _mapping(
        receipt.get("model"),
        name=f"{profile_id} V2.3 p95 snapshot model",
    )
    integrity = _mapping(
        receipt.get("integrity"),
        name=f"{profile_id} V2.3 p95 snapshot integrity",
    )
    if (
        guarded_path != snapshot_path
        or _sha256(raw) != source_row.get("file_sha256")
        or integrity.get("content_sha256")
        != source_row.get("content_sha256")
        or model.get("model_id") != profile_id
        or model.get("runtime_model") != source_row.get("runtime_model")
        or model.get("served_model") != source_row.get("served_model")
    ):
        raise PilotV25ParentImportError(
            f"{profile_id} V2.3 p95 snapshot binding drifted"
        )
    return snapshot_path, raw, receipt


def _rebuild_v25_parent_import_receipt(
    *,
    repo_root: Path,
    contract: PilotContract,
    expected_git_commit: str,
    v24_terminal: Mapping[str, Any],
    v25_manifest: Mapping[str, Any],
    v25_manifest_raw: bytes,
    v23_manifest: Mapping[str, Any],
    v23_manifest_raw: bytes,
    debit: ParentBudgetDebit,
) -> dict[str, Any]:
    """Rebuild the complete receipt from frozen sources and child artifacts."""

    raw_root = repo_root.joinpath(*V25_RAW_ROOT.parts)
    parent = _mapping(
        v23_manifest.get("parent"),
        name="V2.3 parent release",
    )
    source_inventory = _mapping(
        v23_manifest.get("observed_p95_sources"),
        name="V2.3 observed-p95 source inventory",
    )
    expected_profiles = set(V25_ALLOWED_P95_PROFILES) | set(
        V25_BOUNDARY_ONLY_P95_PROFILES
    )
    if set(source_inventory) != expected_profiles:
        raise PilotV25ParentImportError(
            "V2.3 observed-p95 source inventory drifted"
        )
    policy_binding = {
        "source_annotated_tag": parent["science_tag"],
        "expected_tag_object": parent["science_tag_object"],
        "expected_peeled_commit": parent["science_commit"],
    }
    imported_profiles: dict[str, Any] = {}
    boundary_profiles: dict[str, Any] = {}
    for profile_id, source_value in source_inventory.items():
        source_row = _mapping(
            source_value,
            name=f"{profile_id} V2.3 p95 source",
        )
        snapshot_path, snapshot_raw, parent_receipt = (
            _verified_parent_snapshot(
                repo_root=repo_root,
                raw_root=raw_root,
                profile_id=profile_id,
                source_row=source_row,
            )
        )
        source_binding = {
            "parent_source_path": source_row["path"],
            "snapshot_path": _repo_relative(
                repo_root,
                snapshot_path,
                name=f"{profile_id} V2.3 p95 snapshot",
            ),
            "file_sha256": source_row["file_sha256"],
            "content_sha256": source_row["content_sha256"],
            "runtime_model": source_row["runtime_model"],
            "served_model": source_row["served_model"],
            "historical_checkpoint_policy": _json_copy(policy_binding),
        }
        if profile_id in V25_ALLOWED_P95_PROFILES:
            receipt_path = inherited_p95_receipt_path(raw_root, profile_id)
            projection_path = inherited_projection_path(raw_root, profile_id)
            receipt_relative = _normalized_relative(
                _repo_relative(
                    repo_root,
                    receipt_path,
                    name=f"{profile_id} V2.5 child authority",
                ),
                required_top="experiment_results",
                name=f"{profile_id} V2.5 child authority path",
            )
            projection_relative = _normalized_relative(
                _repo_relative(
                    repo_root,
                    projection_path,
                    name=f"{profile_id} V2.5 projection",
                ),
                required_top="experiment_results",
                name=f"{profile_id} V2.5 projection path",
            )
            try:
                guarded_receipt_path, receipt_raw = _guarded_file(
                    repo_root,
                    receipt_relative,
                    name=f"{profile_id} V2.5 child authority",
                )
                guarded_projection_path, projection_raw = _guarded_file(
                    repo_root,
                    projection_relative,
                    name=f"{profile_id} V2.5 projection",
                )
                child_receipt = _strict_json(
                    receipt_raw,
                    name=f"{profile_id} V2.5 child authority",
                )
                projection = _strict_json(
                    projection_raw,
                    name=f"{profile_id} V2.5 projection",
                )
            except PilotV24ParentImportError as exc:
                raise _translate_v24(exc) from exc
            expected_child_receipt = _build_child_p95_receipt(
                repo_root=repo_root,
                contract=contract,
                child_git_tag=V25_SCIENCE_TAG,
                child_git_commit=expected_git_commit,
                v25_manifest=v25_manifest,
                v23_manifest=v23_manifest,
                v25_manifest_raw=v25_manifest_raw,
                v23_manifest_raw=v23_manifest_raw,
                profile_id=profile_id,
                parent_receipt=parent_receipt,
                parent_snapshot_path=snapshot_path,
                parent_snapshot_raw=snapshot_raw,
            )
            expected_projection = _build_child_projection(
                contract=contract,
                child_git_tag=V25_SCIENCE_TAG,
                child_git_commit=expected_git_commit,
                profile_id=profile_id,
                child_receipt=expected_child_receipt,
                child_receipt_path=receipt_path,
            )
            if (
                guarded_receipt_path != receipt_path
                or guarded_projection_path != projection_path
                or child_receipt != expected_child_receipt
                or projection != expected_projection
            ):
                raise PilotV25ParentImportError(
                    f"V2.5 imported p95 artifacts drifted for {profile_id}"
                )
            imported_profiles[profile_id] = {
                **source_binding,
                "child_authority_receipt": receipt_relative.as_posix(),
                "child_authority_receipt_file_sha256": _sha256(receipt_raw),
                "child_authority_receipt_content_sha256": child_receipt[
                    "integrity"
                ]["content_sha256"],
                "child_projection": projection_relative.as_posix(),
                "child_projection_file_sha256": _sha256(projection_raw),
                "child_projection_content_sha256": projection["integrity"][
                    "content_sha256"
                ],
            }
        else:
            boundary_profiles[profile_id] = {
                **source_binding,
                "dispatch_authority": False,
                "boundary_reason": _BOUNDARY_REASON,
            }

    parent_run_ledger, parent_budget_ledger = (
        _parent_ledger_receipt_bindings(v23_manifest)
    )
    return _seal(
        {
            "schema_version": V25_PARENT_IMPORT_SCHEMA_VERSION,
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "child_release": {
                "git_tag": V25_SCIENCE_TAG,
                "git_commit": expected_git_commit,
            },
            "v2_4_terminal_no_go": _json_copy(v24_terminal),
            "v2_3_parent_release": _json_copy(parent),
            "source_manifest": {
                "path": V25_SOURCE_MANIFEST_PATH.as_posix(),
                "file_sha256": _sha256(v25_manifest_raw),
                "content_sha256": v25_manifest["integrity"][
                    "content_sha256"
                ],
            },
            "v2_3_source_manifest": {
                "path": V24_PARENT_SOURCE_MANIFEST_PATH.as_posix(),
                "file_sha256": _sha256(v23_manifest_raw),
                "content_sha256": (
                    V24_PARENT_SOURCE_MANIFEST_CONTENT_SHA256
                ),
            },
            "cumulative_budget_debit": debit.to_dict(),
            "parent_run_ledger": parent_run_ledger,
            "parent_budget_ledger": parent_budget_ledger,
            "imported_projection_profiles": imported_profiles,
            "boundary_only_profiles": boundary_profiles,
            "provider_calls": 0,
            "scientific_evidence": False,
            "scientific_outcomes_observed_before_amendment": False,
            "claim_boundary": _CLAIM_BOUNDARY,
        }
    )


def _persist_v25_parent_import_impl(
    *,
    contract: PilotContract,
    repo_root: str | Path,
    raw_root: str | Path,
    parent_repo_root: str | Path,
    child_git_tag: str,
    child_git_commit: str,
) -> dict[str, Any]:
    child_root = _real_root(repo_root, name="child repository root")
    parent_root = _real_root(parent_repo_root, name="V2.3 parent repository root")
    child_raw_root = Path(raw_root).absolute()
    if (
        _repo_relative(
            child_root,
            child_raw_root,
            name="V2.5 raw root",
        )
        != V25_RAW_ROOT.as_posix()
    ):
        raise PilotV25ParentImportError(
            "V2.5 parent import requires its fresh pilot-v2.5 raw namespace"
        )
    if (
        child_git_tag != V25_SCIENCE_TAG
        or _COMMIT_RE.fullmatch(child_git_commit) is None
    ):
        raise PilotV25ParentImportError(
            "V2.5 child release tag or commit is malformed"
        )
    v25_manifest, v25_manifest_raw = _load_v25_source_manifest(child_root)
    _validate_child_contract(
        contract,
        source_manifest=v25_manifest,
        source_manifest_raw=v25_manifest_raw,
    )

    # The terminal V2.4 no-go must be verified before the V2.3 source chain.
    v24_terminal = _verify_v24_terminal_evidence(
        child_root,
        source_manifest=v25_manifest,
    )
    v23_manifest, v23_manifest_raw = _load_v23_source_manifest(
        child_root,
        v25_manifest=v25_manifest,
    )
    try:
        _verify_parent_git(parent_root, v23_manifest)
        parent_contract = _verify_parent_contract(parent_root, v23_manifest)
        run_snapshot, budget_snapshot = _verify_parent_ledgers(
            parent_root,
            parent_contract=parent_contract,
            source_manifest=v23_manifest,
        )
        _verify_parent_bound_files(
            parent_root,
            child_root,
            source_manifest=v23_manifest,
        )
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc

    policy = _historical_policy(parent_root, v23_manifest)
    imported_profiles: dict[str, Any] = {}
    boundary_profiles: dict[str, Any] = {}
    for profile_id, source_row in v23_manifest[
        "observed_p95_sources"
    ].items():
        relative = _normalized_relative(
            source_row["path"],
            required_top="experiment_results",
            name=f"{profile_id} V2.3 p95 receipt path",
        )
        try:
            _, parent_raw = _guarded_file(
                parent_root,
                relative,
                name=f"{profile_id} V2.3 p95 receipt",
            )
        except PilotV24ParentImportError as exc:
            raise _translate_v24(exc) from exc
        if _sha256(parent_raw) != source_row["file_sha256"]:
            raise PilotV25ParentImportError(
                f"{profile_id} V2.3 p95 receipt file hash drifted"
            )
        try:
            binding = verified_observed_p95_authority_binding(
                relative.as_posix(),
                repo_root=parent_root,
                expected_git_commit=v23_manifest["parent"][
                    "science_commit"
                ],
                historical_checkpoint_policy=policy,
            )
        except ObservedP95AuthorityError as exc:
            raise PilotV25ParentImportError(
                f"{profile_id} historical p95 source chain failed verification: {exc}"
            ) from exc
        if (
            binding["receipt_file_sha256"] != source_row["file_sha256"]
            or binding["receipt_content_sha256"]
            != source_row["content_sha256"]
        ):
            raise PilotV25ParentImportError(
                f"{profile_id} verified historical p95 binding drifted"
            )
        snapshot_path = _snapshot_path(child_raw_root, profile_id)
        _atomic_exact_bytes(snapshot_path, parent_raw)
        parent_receipt = _strict_json(
            parent_raw,
            name=f"{profile_id} V2.3 p95 snapshot",
        )
        source_binding = {
            "parent_source_path": source_row["path"],
            "snapshot_path": _repo_relative(
                child_root,
                snapshot_path,
                name=f"{profile_id} V2.3 p95 snapshot",
            ),
            "file_sha256": source_row["file_sha256"],
            "content_sha256": source_row["content_sha256"],
            "runtime_model": source_row["runtime_model"],
            "served_model": source_row["served_model"],
            "historical_checkpoint_policy": {
                "source_annotated_tag": policy.source_annotated_tag,
                "expected_tag_object": policy.expected_tag_object,
                "expected_peeled_commit": policy.expected_peeled_commit,
            },
        }
        if profile_id in V25_ALLOWED_P95_PROFILES:
            child_receipt = _build_child_p95_receipt(
                repo_root=child_root,
                contract=contract,
                child_git_tag=child_git_tag,
                child_git_commit=child_git_commit,
                v25_manifest=v25_manifest,
                v23_manifest=v23_manifest,
                v25_manifest_raw=v25_manifest_raw,
                v23_manifest_raw=v23_manifest_raw,
                profile_id=profile_id,
                parent_receipt=parent_receipt,
                parent_snapshot_path=snapshot_path,
                parent_snapshot_raw=parent_raw,
            )
            child_receipt_path = inherited_p95_receipt_path(
                child_raw_root,
                profile_id,
            )
            _atomic_exact_json(child_receipt_path, child_receipt)
            projection = _build_child_projection(
                contract=contract,
                child_git_tag=child_git_tag,
                child_git_commit=child_git_commit,
                profile_id=profile_id,
                child_receipt=child_receipt,
                child_receipt_path=child_receipt_path,
            )
            projection_path = inherited_projection_path(
                child_raw_root,
                profile_id,
            )
            _atomic_exact_json(projection_path, projection)
            imported_profiles[profile_id] = {
                **source_binding,
                "child_authority_receipt": _repo_relative(
                    child_root,
                    child_receipt_path,
                    name=f"{profile_id} V2.5 child authority",
                ),
                "child_authority_receipt_file_sha256": hashlib.sha256(
                    child_receipt_path.read_bytes()
                ).hexdigest(),
                "child_authority_receipt_content_sha256": child_receipt[
                    "integrity"
                ]["content_sha256"],
                "child_projection": _repo_relative(
                    child_root,
                    projection_path,
                    name=f"{profile_id} V2.5 projection",
                ),
                "child_projection_file_sha256": hashlib.sha256(
                    projection_path.read_bytes()
                ).hexdigest(),
                "child_projection_content_sha256": projection["integrity"][
                    "content_sha256"
                ],
            }
        else:
            boundary_profiles[profile_id] = {
                **source_binding,
                "dispatch_authority": False,
                "boundary_reason": _BOUNDARY_REASON,
            }
    if set(imported_profiles) != set(V25_ALLOWED_P95_PROFILES):
        raise PilotV25ParentImportError(
            "V2.5 import did not create the exact two p95 authorities"
        )
    if set(boundary_profiles) != set(V25_BOUNDARY_ONLY_P95_PROFILES):
        raise PilotV25ParentImportError(
            "V2.5 parent boundary p95 inventory drifted"
        )
    debit = parent_budget_debit_for_v25(contract, repo_root=child_root)
    if debit is None:  # pragma: no cover - contract identity guard
        raise PilotV25ParentImportError("V2.5 cumulative debit is unavailable")
    expected_run_ledger, expected_budget_ledger = (
        _parent_ledger_receipt_bindings(v23_manifest)
    )
    actual_run_ledger = {
        "ledger_sha256": run_snapshot["ledger_sha256"],
        "event_count": len(run_snapshot["events"]),
        "event_chain_head": run_snapshot["events"][-1]["event_sha256"],
    }
    actual_budget_ledger = {
        "ledger_sha256": budget_snapshot["ledger_sha256"],
        "event_count": len(budget_snapshot["events"]),
        "event_chain_head": budget_snapshot["event_chain_head"],
    }
    if (
        actual_run_ledger != expected_run_ledger
        or actual_budget_ledger != expected_budget_ledger
    ):
        raise PilotV25ParentImportError(
            "verified V2.3 parent ledgers differ from their tracked bindings"
        )
    receipt = _rebuild_v25_parent_import_receipt(
        repo_root=child_root,
        contract=contract,
        expected_git_commit=child_git_commit,
        v24_terminal=v24_terminal,
        v25_manifest=v25_manifest,
        v25_manifest_raw=v25_manifest_raw,
        v23_manifest=v23_manifest,
        v23_manifest_raw=v23_manifest_raw,
        debit=debit,
    )
    receipt_path = child_raw_root / "parent-import" / "parent_import_receipt.json"
    _atomic_exact_json(receipt_path, receipt)
    verified = verify_v25_parent_import_receipt(
        receipt_path,
        repo_root=child_root,
        contract=contract,
        expected_git_commit=child_git_commit,
    )
    if verified != receipt:
        raise PilotV25ParentImportError(
            "persisted V2.5 parent import differs after verification"
        )
    return {
        "receipt": str(receipt_path),
        "receipt_content_sha256": receipt["integrity"]["content_sha256"],
        "provider_calls": 0,
        "scientific_evidence": False,
        "imported_profiles": sorted(imported_profiles),
        "boundary_only_profiles": sorted(boundary_profiles),
    }


def persist_v25_parent_import(**kwargs: Any) -> dict[str, Any]:
    """Validate both immutable parents and persist one idempotent V2.5 import."""

    try:
        return _persist_v25_parent_import_impl(**kwargs)
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc


def _load_child_contract_from_receipt(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
) -> PilotContract:
    binding = _mapping(
        receipt.get("contract"),
        name="V2.5 inherited child contract binding",
    )
    relative = _normalized_relative(
        binding.get("path", ""),
        required_top="experiments",
        name="V2.5 inherited child contract path",
    )
    try:
        _, raw = _guarded_file(
            repo_root,
            relative,
            name="V2.5 inherited child contract",
        )
        value = _strict_json(raw, name="V2.5 inherited child contract")
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    if _sha256(raw) != binding.get("file_sha256"):
        raise PilotV25ParentImportError(
            "V2.5 inherited child contract file hash drifted"
        )
    contract = PilotContract.from_dict(value)
    if (
        contract.contract_id != binding.get("contract_id")
        or contract.canonical_hash != binding.get("contract_sha256")
    ):
        raise PilotV25ParentImportError(
            "V2.5 inherited child contract identity drifted"
        )
    return contract


def verify_v25_inherited_p95_receipt(
    receipt: Mapping[str, Any],
    *,
    repo_root: str | Path,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Verify one V2.5 child authority without reopening either parent raw tree."""

    root = _real_root(repo_root, name="child repository root")
    value = _json_copy(receipt)
    try:
        from .pilot_v24_parent_import import _verify_self_hash

        _verify_self_hash(
            value,
            schema_version=V25_INHERITED_P95_RECEIPT_SCHEMA_VERSION,
            name="V2.5 inherited p95 receipt",
        )
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    contract = _load_child_contract_from_receipt(value, repo_root=root)
    v25_manifest, v25_raw = _load_v25_source_manifest(root)
    _validate_child_contract(
        contract,
        source_manifest=v25_manifest,
        source_manifest_raw=v25_raw,
    )
    _verify_v24_terminal_evidence(root, source_manifest=v25_manifest)
    v23_manifest, v23_raw = _load_v23_source_manifest(
        root,
        v25_manifest=v25_manifest,
    )
    git = _mapping(value.get("git"), name="V2.5 inherited git binding")
    model = _mapping(value.get("model"), name="V2.5 inherited model binding")
    parent_source = _mapping(
        value.get("parent_source"),
        name="V2.5 inherited parent source",
    )
    if (
        git.get("tag") != V25_SCIENCE_TAG
        or git.get("commit") != expected_git_commit
        or _COMMIT_RE.fullmatch(expected_git_commit) is None
    ):
        raise PilotV25ParentImportError(
            "V2.5 inherited child release binding is malformed"
        )
    profile_id = str(model.get("model_id"))
    if profile_id not in V25_ALLOWED_P95_PROFILES:
        raise PilotV25ParentImportError(
            "boundary-only parent model cannot create V2.5 dispatch authority"
        )
    source_row = v23_manifest["observed_p95_sources"][profile_id]
    snapshot_relative = _normalized_relative(
        parent_source.get("parent_receipt_snapshot_path", ""),
        required_top="experiment_results",
        name="V2.5 parent p95 snapshot path",
    )
    try:
        snapshot_path, snapshot_raw = _guarded_file(
            root,
            snapshot_relative,
            name="V2.5 parent p95 snapshot",
        )
        parent_receipt = _strict_json(
            snapshot_raw,
            name="V2.5 parent p95 snapshot",
        )
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    v23_parent = v23_manifest["parent"]
    if (
        _sha256(snapshot_raw) != source_row["file_sha256"]
        or parent_source.get("parent_receipt_file_sha256")
        != source_row["file_sha256"]
        or parent_source.get("parent_receipt_content_sha256")
        != source_row["content_sha256"]
        or parent_source.get("manifest_file_sha256") != _sha256(v23_raw)
        or parent_source.get("manifest_content_sha256")
        != V24_PARENT_SOURCE_MANIFEST_CONTENT_SHA256
        or parent_source.get("parent_contract_sha256")
        != v23_parent["contract_canonical_sha256"]
        or parent_source.get("parent_git_tag") != v23_parent["science_tag"]
        or parent_source.get("parent_git_tag_object")
        != v23_parent["science_tag_object"]
        or parent_source.get("parent_git_commit")
        != v23_parent["science_commit"]
        or parent_source.get("current_exactness_match_required") is not True
        or parent_receipt.get("integrity", {}).get("content_sha256")
        != source_row["content_sha256"]
    ):
        raise PilotV25ParentImportError(
            "V2.5 inherited parent snapshot binding drifted"
        )
    expected = _build_child_p95_receipt(
        repo_root=root,
        contract=contract,
        child_git_tag=V25_SCIENCE_TAG,
        child_git_commit=expected_git_commit,
        v25_manifest=v25_manifest,
        v23_manifest=v23_manifest,
        v25_manifest_raw=v25_raw,
        v23_manifest_raw=v23_raw,
        profile_id=profile_id,
        parent_receipt=parent_receipt,
        parent_snapshot_path=snapshot_path,
        parent_snapshot_raw=snapshot_raw,
    )
    if value != expected:
        raise PilotV25ParentImportError(
            "V2.5 inherited p95 receipt differs from its tracked parent source"
        )
    reservations = value.get("reservations")
    if not isinstance(reservations, dict):
        raise PilotV25ParentImportError(
            "V2.5 inherited p95 reservations are malformed"
        )
    return _json_copy(reservations)


def verified_v25_inherited_p95_binding(
    receipt_path: str | Path,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Return one guarded V2.5 receipt binding and verified reservations."""

    root = _real_root(repo_root, name="child repository root")
    path = Path(receipt_path)
    if path.is_absolute():
        try:
            relative = PurePosixPath(*path.absolute().relative_to(root).parts)
        except ValueError as exc:
            raise PilotV25ParentImportError(
                "V2.5 inherited p95 receipt escaped the child repository"
            ) from exc
    else:
        relative = _normalized_relative(
            path,
            required_top="experiment_results",
            name="V2.5 inherited p95 receipt path",
        )
    try:
        _, raw = _guarded_file(
            root,
            relative,
            name="V2.5 inherited p95 receipt",
        )
        receipt = _strict_json(raw, name="V2.5 inherited p95 receipt")
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    reservations = verify_v25_inherited_p95_receipt(
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


def verify_v25_parent_import_receipt(
    receipt_or_path: Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path,
    contract: PilotContract,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Verify the V2.5 import receipt and every child authority/projection."""

    root = _real_root(repo_root, name="child repository root")
    if _COMMIT_RE.fullmatch(expected_git_commit) is None:
        raise PilotV25ParentImportError(
            "V2.5 expected child commit must be exactly 40 lowercase hex characters"
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
                raise PilotV25ParentImportError(
                    "V2.5 parent import receipt escaped the child repository"
                ) from exc
        else:
            relative = _normalized_relative(
                path,
                required_top="experiment_results",
                name="V2.5 parent import receipt path",
            )
        try:
            _, raw = _guarded_file(
                root,
                relative,
                name="V2.5 parent import receipt",
            )
            value = _strict_json(raw, name="V2.5 parent import receipt")
        except PilotV24ParentImportError as exc:
            raise _translate_v24(exc) from exc
    try:
        from .pilot_v24_parent_import import _verify_self_hash

        _verify_self_hash(
            value,
            schema_version=V25_PARENT_IMPORT_SCHEMA_VERSION,
            name="V2.5 parent import receipt",
        )
    except PilotV24ParentImportError as exc:
        raise _translate_v24(exc) from exc
    manifest, manifest_raw = _load_v25_source_manifest(root)
    _validate_child_contract(
        contract,
        source_manifest=manifest,
        source_manifest_raw=manifest_raw,
    )
    v24_terminal = _verify_v24_terminal_evidence(
        root,
        source_manifest=manifest,
    )
    v23_manifest, v23_raw = _load_v23_source_manifest(
        root,
        v25_manifest=manifest,
    )
    debit = parent_budget_debit_for_v25(contract, repo_root=root)
    if debit is None:  # pragma: no cover - contract identity guard
        raise PilotV25ParentImportError("V2.5 cumulative debit is unavailable")
    expected = _rebuild_v25_parent_import_receipt(
        repo_root=root,
        contract=contract,
        expected_git_commit=expected_git_commit,
        v24_terminal=v24_terminal,
        v25_manifest=manifest,
        v25_manifest_raw=manifest_raw,
        v23_manifest=v23_manifest,
        v23_manifest_raw=v23_raw,
        debit=debit,
    )
    if value != expected:
        raise PilotV25ParentImportError(
            "V2.5 parent import receipt differs from frozen sources and "
            "verified child artifacts"
        )
    return value


__all__ = [
    "PilotV25ParentImportError",
    "V25_ALLOWED_P95_PROFILES",
    "V25_BOUNDARY_ONLY_P95_PROFILES",
    "V25_CONTRACT_ID",
    "V25_INHERITED_P95_RECEIPT_SCHEMA_VERSION",
    "V25_PARENT_IMPORT_SCHEMA_VERSION",
    "V25_PARENT_DEBIT_RECORD_SHA256",
    "V25_RAW_ROOT",
    "V25_SCIENCE_TAG",
    "V25_SOURCE_MANIFEST_PATH",
    "inherited_p95_receipt_path",
    "inherited_projection_path",
    "parent_budget_debit_for_v25",
    "persist_v25_parent_import",
    "verified_v25_inherited_p95_binding",
    "verify_v25_inherited_p95_receipt",
    "verify_v25_parent_import_receipt",
]
