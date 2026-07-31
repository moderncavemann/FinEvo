"""Outcome-blind, zero-provider import of V2.10.2 prerequisites into V2.11.

Only the deterministic q-ref, the selected Stage-0 utility profile, the
Stage-0 absolute-flow threshold, and the cumulative budget debit may cross
this boundary.  The immutable V2.10.2 A-D outcomes remain external evidence:
no effect cell, effect metric, raw tree, or observed-p95 authority is copied.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
from typing import Any, Mapping

from .pilot_budget import ParentBudgetDebit
from .pilot_contract import canonical_contract_sha256, canonical_sha256


V211_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.11-source-manifest-v1"
V211_PARENT_IMPORT_SCHEMA_VERSION = "finevo-pilot-v2.11-parent-import-v1"
V211_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_11_source_manifest.json"
)
V211_SOURCE_MANIFEST_FILE_SHA256 = (
    "950f115959ace359984d99285aec60ba794162e666d7ea7c36ec56d6f3d76c1d"
)
V211_SOURCE_MANIFEST_CONTENT_SHA256 = (
    "cea1a1b134f89b98ff515b4deead492f738a9df1b9aa8f2c9b891dce5afab48f"
)
V211_DEFAULT_RECEIPT_PATH = PurePosixPath(
    "experiment_results/pilot-v2.11/raw/parent-import/"
    "parent_import_receipt.json"
)

V2102_CONTRACT_ID = "finevo-pilot-v2.10.2"
V2102_CONTRACT_SHA256 = (
    "b8de8cfb2560d894dad65d68df8ae9126527d12d3807bef045fa52f5e9d4159e"
)
V2102_SCIENCE_TAG = "pilot-v2.10.2-science"
V2102_SCIENCE_TAG_OBJECT = "89b91c4646b6698386141fda8651111c0d2d1810"
V2102_SCIENCE_COMMIT = "2dcc20f8dccc7a6a94a60a00d7f3750a9d61396d"
V2102_RUN_LEDGER_SHA256 = (
    "2219a832b9a7dfe235b32db882e126bddc36938f4f201a2ab84ddea6878bb809"
)
V2102_BUDGET_LEDGER_SHA256 = (
    "73b1bac2a424147cbfa88bdb4e351d6c924b6e82847050f7cb1d254fe1ea4068"
)
V2102_Q_REF = 63.50397933257746
V2102_ABSOLUTE_FLOW_THRESHOLD = 0.05617208967516696
V2102_CUMULATIVE_COST_USD = 16.044922812500005
V2102_CUMULATIVE_COMPLETIONS = 816
V2102_CUMULATIVE_STORAGE_BYTES = 217_010_835

_EXPECTED_UTILITY_PROFILE = {
    "profile_id": "nu-0.5",
    "rho": 1.0,
    "labor_weight": 2.0,
    "inverse_frisch": 0.5,
    "consumption_scale": V2102_Q_REF,
    "discount_factor": 0.99,
    "budget_tolerance": 1e-8,
    "max_labor_hours": 168.0,
}
_ZERO_PROVIDER_BOUNDARY = {
    "provider_construction": False,
    "provider_calls": 0,
    "imported_effect_cells": 0,
    "effect_metrics_observed": False,
    "effect_artifact_paths": [],
    "imported_p95_authorities": [],
}


class PilotV211ParentImportError(RuntimeError):
    """Raised before imported V2.10.2 authority may be consumed."""


def _strict_json(raw: bytes, *, name: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PilotV211ParentImportError(
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
    except PilotV211ParentImportError:
        raise
    except Exception as exc:
        raise PilotV211ParentImportError(f"{name} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise PilotV211ParentImportError(f"{name} must be a JSON object")
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(value, sort_keys=True, allow_nan=False)
        )
    except Exception as exc:
        raise PilotV211ParentImportError(
            "value is not canonical-JSON compatible"
        ) from exc


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    candidate = _json_copy(dict(value))
    if "integrity" in candidate:
        raise PilotV211ParentImportError("cannot seal a pre-sealed value")
    candidate["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1"
    }
    candidate["integrity"]["content_sha256"] = canonical_sha256(candidate)
    return candidate


def _verify_seal(
    value: Mapping[str, Any],
    *,
    schema_version: str,
    name: str,
) -> None:
    candidate = _json_copy(dict(value))
    integrity = candidate.get("integrity")
    if isinstance(integrity, dict):
        claimed = integrity.pop("content_sha256", None)
    else:
        claimed = None
    if (
        candidate.get("schema_version") != schema_version
        or not isinstance(integrity, dict)
        or set(value.get("integrity", {}))
        != {"canonicalization", "content_sha256"}
        or integrity.get("canonicalization") != "json-sort-keys-utf8-v1"
        or claimed != canonical_sha256(candidate)
    ):
        raise PilotV211ParentImportError(
            f"{name} schema or content hash mismatch"
        )


def _real_root(value: str | Path, *, name: str) -> Path:
    path = Path(value).expanduser()
    try:
        if path.is_symlink():
            raise PilotV211ParentImportError(f"{name} must not be a symlink")
        resolved = path.resolve(strict=True)
    except PilotV211ParentImportError:
        raise
    except OSError as exc:
        raise PilotV211ParentImportError(f"{name} is unavailable") from exc
    if not resolved.is_dir():
        raise PilotV211ParentImportError(f"{name} must be a directory")
    return resolved


def _normalized_relative(
    value: Any,
    *,
    required_top: str,
    name: str,
) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PilotV211ParentImportError(f"{name} path is malformed")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or not relative.parts
        or relative.parts[0] != required_top
    ):
        raise PilotV211ParentImportError(
            f"{name} path escaped its allowed namespace"
        )
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
                raise PilotV211ParentImportError(
                    f"{name} path contains a symlink"
                )
        if not stat.S_ISREG(path.lstat().st_mode):
            raise PilotV211ParentImportError(
                f"{name} must be a regular file"
            )
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb", closefd=True) as handle:
            raw = handle.read()
    except PilotV211ParentImportError:
        raise
    except (OSError, ValueError) as exc:
        raise PilotV211ParentImportError(f"{name} is unavailable") from exc
    return path, raw


def _bound_json_file(
    root: Path,
    binding: Mapping[str, Any],
    *,
    required_top: str,
    name: str,
) -> dict[str, Any]:
    if not isinstance(binding, Mapping):
        raise PilotV211ParentImportError(f"{name} binding is malformed")
    relative = _normalized_relative(
        binding.get("path"),
        required_top=required_top,
        name=name,
    )
    _, raw = _guarded_file(root, relative, name=name)
    if (
        len(raw) != binding.get("byte_size")
        or _sha256(raw)
        not in {
            binding.get("sha256"),
            binding.get("file_sha256"),
        }
    ):
        raise PilotV211ParentImportError(f"{name} file identity drifted")
    return _strict_json(raw, name=name)


def _run_git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotV211ParentImportError(
            "V2.10.2 git release identity is unavailable"
        ) from exc
    return result.stdout.strip()


def _verify_git_release(science_root: Path) -> dict[str, str]:
    head = _run_git(science_root, "rev-parse", "--verify", "HEAD^{commit}")
    tag_object = _run_git(
        science_root,
        "rev-parse",
        "--verify",
        f"refs/tags/{V2102_SCIENCE_TAG}^{{tag}}",
    )
    tag_commit = _run_git(
        science_root,
        "rev-parse",
        "--verify",
        f"refs/tags/{V2102_SCIENCE_TAG}^{{commit}}",
    )
    tracked = _run_git(
        science_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    )
    if (
        head != V2102_SCIENCE_COMMIT
        or tag_commit != V2102_SCIENCE_COMMIT
        or tag_object != V2102_SCIENCE_TAG_OBJECT
        or tracked
    ):
        raise PilotV211ParentImportError(
            "V2.10.2 annotated tag or worktree identity drifted"
        )
    return {
        "science_tag": V2102_SCIENCE_TAG,
        "science_tag_object": tag_object,
        "resolved_git_commit": tag_commit,
    }


def _verify_event_ledger(
    value: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    schema_version: str,
    name: str,
) -> None:
    candidate = _json_copy(dict(value))
    claimed = candidate.pop("ledger_sha256", None)
    events = value.get("events")
    runs = value.get("runs")
    if (
        value.get("schema_version") != schema_version
        or value.get("contract_hash") != V2102_CONTRACT_SHA256
        or claimed != binding.get("internal_sha256")
        or claimed
        not in {V2102_RUN_LEDGER_SHA256, V2102_BUDGET_LEDGER_SHA256}
        or canonical_sha256(candidate) != claimed
        or not isinstance(events, list)
        or len(events) != binding.get("event_count")
        or not isinstance(runs, Mapping)
        or len(runs) != binding.get("run_count")
        or not events
        or events[-1].get("event_sha256")
        != binding.get("event_head_sha256")
    ):
        raise PilotV211ParentImportError(f"{name} identity drifted")
    previous = "0" * 64
    for index, source in enumerate(events):
        if not isinstance(source, Mapping):
            raise PilotV211ParentImportError(f"{name} event is malformed")
        row = _json_copy(dict(source))
        digest = row.pop("event_sha256", None)
        if (
            source.get("event_index") != index
            or source.get("previous_event_sha256") != previous
            or digest != canonical_sha256(row)
        ):
            raise PilotV211ParentImportError(
                f"{name} event chain drifted"
            )
        previous = str(digest)


def _verify_budget_totals(
    budget: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    parent = budget.get("parent_debit")
    runs = budget.get("runs")
    if not isinstance(parent, Mapping) or not isinstance(runs, Mapping):
        raise PilotV211ParentImportError("V2.10.2 budget ledger is malformed")
    actual_rows = []
    for row in runs.values():
        if not isinstance(row, Mapping) or not isinstance(
            row.get("actual"), Mapping
        ):
            raise PilotV211ParentImportError(
                "V2.10.2 budget actual row is malformed"
            )
        actual_rows.append(row["actual"])
    cost = float(parent.get("cost_usd", math.nan)) + sum(
        float(row.get("cost_usd", math.nan)) for row in actual_rows
    )
    completions = parent.get("hosted_completions")
    storage = parent.get("storage_bytes")
    if isinstance(completions, bool) or not isinstance(completions, int):
        raise PilotV211ParentImportError(
            "V2.10.2 parent completion debit is malformed"
        )
    if isinstance(storage, bool) or not isinstance(storage, int):
        raise PilotV211ParentImportError(
            "V2.10.2 parent storage debit is malformed"
        )
    for row in actual_rows:
        row_completions = row.get("completions")
        row_storage = row.get("storage_bytes")
        if (
            isinstance(row_completions, bool)
            or not isinstance(row_completions, int)
            or isinstance(row_storage, bool)
            or not isinstance(row_storage, int)
        ):
            raise PilotV211ParentImportError(
                "V2.10.2 budget actual count is malformed"
            )
        completions += row_completions
        storage += row_storage
    if (
        not math.isfinite(cost)
        or not math.isclose(
            cost,
            float(expected.get("cost_usd", math.nan)),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or completions != expected.get("hosted_completions")
        or storage != expected.get("storage_bytes")
    ):
        raise PilotV211ParentImportError(
            "V2.10.2 cumulative budget debit drifted"
        )


def _verify_source_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    _verify_seal(
        value,
        schema_version=V211_SOURCE_MANIFEST_SCHEMA_VERSION,
        name="V2.11 source manifest",
    )
    candidate = _json_copy(dict(value))
    if (
        candidate["integrity"]["content_sha256"]
        != V211_SOURCE_MANIFEST_CONTENT_SHA256
        or candidate.get("import_allowlist")
        != {
            **_ZERO_PROVIDER_BOUNDARY,
            "q_ref": V2102_Q_REF,
            "selected_utility_profile": _EXPECTED_UTILITY_PROFILE,
            "stage0_absolute_flow_utility_threshold": candidate.get(
                "import_allowlist", {}
            ).get("stage0_absolute_flow_utility_threshold"),
        }
        or candidate.get("parent_release")
        != {
            "contract_id": V2102_CONTRACT_ID,
            "contract_sha256": V2102_CONTRACT_SHA256,
            "publication_status": "complete-with-no-go",
            "resolved_git_commit": V2102_SCIENCE_COMMIT,
            "science_tag": V2102_SCIENCE_TAG,
            "science_tag_object": V2102_SCIENCE_TAG_OBJECT,
        }
    ):
        raise PilotV211ParentImportError(
            "V2.11 source manifest allowlist drifted"
        )
    threshold = candidate["import_allowlist"].get(
        "stage0_absolute_flow_utility_threshold"
    )
    if (
        not isinstance(threshold, Mapping)
        or threshold.get("value") != V2102_ABSOLUTE_FLOW_THRESHOLD
        or threshold.get("source_profile") != "nu-0.5"
        or threshold.get("treatment_outcomes_inspected") is not False
        or threshold.get("row_count") != 96
        or threshold.get("source_seeds") != [1942013315, 760687867]
        or not isinstance(threshold.get("source_manifests"), list)
        or len(threshold["source_manifests"]) != 2
    ):
        raise PilotV211ParentImportError(
            "V2.11 Stage-0 threshold allowlist drifted"
        )
    debit = candidate.get("parent_cumulative_budget_debit")
    if debit != {
        "parent_contract_sha256": V2102_CONTRACT_SHA256,
        "parent_run_ledger_sha256": V2102_RUN_LEDGER_SHA256,
        "parent_budget_ledger_sha256": V2102_BUDGET_LEDGER_SHA256,
        "stage_bucket": "parent_v2102",
        "cost_usd": V2102_CUMULATIVE_COST_USD,
        "hosted_completions": V2102_CUMULATIVE_COMPLETIONS,
        "storage_bytes": V2102_CUMULATIVE_STORAGE_BYTES,
    }:
        raise PilotV211ParentImportError(
            "V2.11 parent budget allowlist drifted"
        )
    return candidate


def _load_source_manifest(repo_root: Path) -> dict[str, Any]:
    _, raw = _guarded_file(
        repo_root,
        V211_SOURCE_MANIFEST_PATH,
        name="tracked V2.11 source manifest",
    )
    if _sha256(raw) != V211_SOURCE_MANIFEST_FILE_SHA256:
        raise PilotV211ParentImportError(
            "tracked V2.11 source manifest file hash drifted"
        )
    return _verify_source_manifest(
        _strict_json(raw, name="tracked V2.11 source manifest")
    )


def _source_roots(
    repo_root: Path,
    manifest: Mapping[str, Any],
    *,
    science_repo_root: str | Path | None,
    evidence_repo_root: str | Path | None,
) -> tuple[Path, Path]:
    science_hint = manifest.get("science_release", {}).get("root_hint")
    evidence_hint = manifest.get("evidence_release", {}).get("root_hint")
    if not isinstance(science_hint, str) or not isinstance(evidence_hint, str):
        raise PilotV211ParentImportError("source root hints are malformed")
    science_value = (
        repo_root / science_hint
        if science_repo_root is None
        else Path(science_repo_root)
    )
    evidence_value = (
        repo_root / evidence_hint
        if evidence_repo_root is None
        else Path(evidence_repo_root)
    )
    return (
        _real_root(science_value, name="V2.10.2 science source"),
        _real_root(evidence_value, name="V2.10.2 evidence source"),
    )


def _verify_self_hashed_artifact(
    value: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    schema_version: str,
    name: str,
    hash_includes_canonicalization: bool = True,
) -> None:
    candidate = _json_copy(dict(value))
    integrity = candidate.get("integrity")
    if isinstance(integrity, dict):
        claimed = integrity.pop("content_sha256", None)
        if not hash_includes_canonicalization:
            candidate.pop("integrity", None)
    else:
        claimed = None
    if (
        value.get("schema_version") != schema_version
        or not isinstance(integrity, Mapping)
        or set(value.get("integrity", {}))
        != {"canonicalization", "content_sha256"}
        or integrity.get("canonicalization") != "json-sort-keys-utf8-v1"
        or claimed != binding.get("content_sha256")
        or canonical_sha256(candidate) != binding.get("content_sha256")
    ):
        raise PilotV211ParentImportError(f"{name} content hash drifted")


def _verify_prerequisites(
    science_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    release = manifest["science_release"]
    contract_binding = release["contract"]
    contract = _bound_json_file(
        science_root,
        contract_binding,
        required_top="experiments",
        name="V2.10.2 contract",
    )
    if (
        contract.get("contract_id") != V2102_CONTRACT_ID
        or contract.get("status") != "frozen"
        or contract.get("implementation", {}).get("required_git_tag")
        != V2102_SCIENCE_TAG
        or contract.get("integrity", {}).get("declared_sha256")
        != V2102_CONTRACT_SHA256
        or canonical_contract_sha256(contract) != V2102_CONTRACT_SHA256
        or contract_binding.get("canonical_sha256")
        != V2102_CONTRACT_SHA256
    ):
        raise PilotV211ParentImportError(
            "V2.10.2 contract identity drifted"
        )

    run_binding = release["run_ledger"]
    run_ledger = _bound_json_file(
        science_root,
        run_binding,
        required_top="experiment_results",
        name="V2.10.2 run ledger",
    )
    _verify_event_ledger(
        run_ledger,
        run_binding,
        schema_version="finevo-pilot-run-ledger-v2",
        name="V2.10.2 run ledger",
    )

    budget_binding = release["budget_ledger"]
    budget_ledger = _bound_json_file(
        science_root,
        budget_binding,
        required_top="experiment_results",
        name="V2.10.2 budget ledger",
    )
    _verify_event_ledger(
        budget_ledger,
        budget_binding,
        schema_version="finevo-pilot-budget-ledger-v2",
        name="V2.10.2 budget ledger",
    )
    _verify_budget_totals(
        budget_ledger,
        manifest["parent_cumulative_budget_debit"],
    )

    qref_binding = release["q_ref_resolution"]
    qref = _bound_json_file(
        science_root,
        qref_binding,
        required_top="experiment_results",
        name="V2.10.2 q-ref resolution",
    )
    _verify_self_hashed_artifact(
        qref,
        qref_binding,
        schema_version="finevo-pilot-v2.10.2-imported-qref-resolution-v1",
        name="V2.10.2 q-ref resolution",
    )
    qref_bindings = qref.get("bindings")
    if (
        qref.get("q_ref") != V2102_Q_REF
        or qref.get("scientific_evidence") is not False
        or qref.get("claim_boundary")
        != (
            "Exact V2.9 q-ref prerequisite import and V2.10.2 reseal only; "
            "no V2.10.2 A-D treatment-effect or model-performance evidence."
        )
        or qref.get("provider_construction_current_attempt") is not False
        or qref.get("provider_calls_current_attempt") != 0
        or qref.get("hosted_provider_calls_current_attempt") != 0
        or not isinstance(qref_bindings, Mapping)
        or qref_bindings.get("contract_sha256") != V2102_CONTRACT_SHA256
        or qref_bindings.get("git_tag") != V2102_SCIENCE_TAG
        or qref_bindings.get("git_commit") != V2102_SCIENCE_COMMIT
    ):
        raise PilotV211ParentImportError(
            "V2.10.2 q-ref prerequisite semantics drifted"
        )

    selection_binding = release["stage0_selection"]
    selection = _bound_json_file(
        science_root,
        selection_binding,
        required_top="experiment_results",
        name="V2.10.2 Stage-0 selection",
    )
    _verify_self_hashed_artifact(
        selection,
        selection_binding,
        schema_version="finevo-stage0-selection-v1",
        name="V2.10.2 Stage-0 selection",
    )
    threshold = selection.get("absolute_flow_utility_threshold")
    expected_threshold = manifest["import_allowlist"][
        "stage0_absolute_flow_utility_threshold"
    ]
    source_projection = []
    if isinstance(threshold, Mapping):
        for row in threshold.get("source_manifests", []):
            if isinstance(row, Mapping):
                source_projection.append(
                    {
                        "environment_seed": row.get("environment_seed"),
                        "manifest_sha256": row.get("manifest_sha256"),
                        "row_count": row.get("row_count"),
                        "utility_ledger_sha256": row.get(
                            "utility_ledger_sha256"
                        ),
                    }
                )
    threshold_projection = (
        {}
        if not isinstance(threshold, Mapping)
        else {
            "aggregation": threshold.get("aggregation"),
            "field": threshold.get("field"),
            "method": threshold.get("method"),
            "row_count": threshold.get("row_count"),
            "source_manifests": source_projection,
            "source_matrix_sha256": threshold.get("source_matrix_sha256"),
            "source_profile": threshold.get("selected_profile_id"),
            "source_seeds": threshold.get("source_seeds"),
            "treatment_outcomes_inspected": threshold.get(
                "treatment_outcomes_inspected"
            ),
            "value": threshold.get("value"),
        }
    )
    selection_bindings = selection.get("bindings")
    if (
        selection.get("contract_sha256") != V2102_CONTRACT_SHA256
        or selection.get("selected_profile_id") != "nu-0.5"
        or selection.get("selected_utility")
        != {
            key: value
            for key, value in _EXPECTED_UTILITY_PROFILE.items()
            if key != "profile_id"
        }
        or selection.get("outcome_fields_used") != []
        or threshold_projection != expected_threshold
        or not isinstance(selection_bindings, Mapping)
        or selection_bindings.get("contract_sha256")
        != V2102_CONTRACT_SHA256
        or selection_bindings.get("git_tag") != V2102_SCIENCE_TAG
        or selection_bindings.get("git_commit") != V2102_SCIENCE_COMMIT
        or selection_bindings.get("q_ref_file_sha256")
        != qref_binding["file_sha256"]
        or selection_bindings.get("q_ref_content_sha256")
        != qref_binding["content_sha256"]
    ):
        raise PilotV211ParentImportError(
            "V2.10.2 Stage-0 prerequisite semantics drifted"
        )

    receipt_binding = release["stage0_receipt"]
    stage_receipt = _bound_json_file(
        science_root,
        receipt_binding,
        required_top="experiment_results",
        name="V2.10.2 Stage-0 receipt",
    )
    _verify_self_hashed_artifact(
        stage_receipt,
        receipt_binding,
        schema_version="finevo-pilot-stage-receipt-v2",
        name="V2.10.2 Stage-0 receipt",
        hash_includes_canonicalization=False,
    )
    if (
        stage_receipt.get("stage_id") != "stage0-calibration"
        or stage_receipt.get("status") != "complete"
        or stage_receipt.get("go") is not True
        or stage_receipt.get("artifacts", {}).get("selected_profile_id")
        != "nu-0.5"
    ):
        raise PilotV211ParentImportError(
            "V2.10.2 Stage-0 receipt is not a complete go"
        )
    return {
        "q_ref_file_sha256": qref_binding["file_sha256"],
        "q_ref_content_sha256": qref_binding["content_sha256"],
        "stage0_selection_file_sha256": selection_binding["file_sha256"],
        "stage0_selection_content_sha256": selection_binding[
            "content_sha256"
        ],
        "stage0_receipt_file_sha256": receipt_binding["file_sha256"],
        "stage0_receipt_content_sha256": receipt_binding["content_sha256"],
    }


def _verify_evidence(
    evidence_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    release = manifest["evidence_release"]
    aggregate_binding = release["aggregate_json"]
    aggregate = _bound_json_file(
        evidence_root,
        aggregate_binding,
        required_top="evidence",
        name="V2.10.2 evidence aggregate",
    )
    expected_totals = {
        "cost_usd": V2102_CUMULATIVE_COST_USD,
        "completions": V2102_CUMULATIVE_COMPLETIONS,
        "storage_bytes": V2102_CUMULATIVE_STORAGE_BYTES,
    }
    if (
        aggregate.get("schema_version")
        != "finevo-pilot-v2.10.2-evidence-package-v1"
        or aggregate.get("contract_id") != V2102_CONTRACT_ID
        or aggregate.get("contract_sha256") != V2102_CONTRACT_SHA256
        or aggregate.get("pilot_tag") != V2102_SCIENCE_TAG
        or aggregate.get("resolved_git_commit") != V2102_SCIENCE_COMMIT
        or aggregate.get("publication_status") != "complete-with-no-go"
        or aggregate.get("scientific_complete") is not False
        or aggregate.get("scientific_matrix_complete") is not False
        or aggregate.get("budget", {}).get("actual_totals")
        != expected_totals
        or aggregate.get("budget", {}).get("pass") is not True
        or aggregate.get("denominator", {}).get("expected_count") != 211
        or aggregate.get("denominator", {}).get("observed_ledger_count")
        != 211
        or aggregate.get("denominator", {}).get("status_counts")
        != {"complete": 126, "failed": 85}
        or aggregate.get("denominator", {}).get("all_rows_terminal")
        is not True
    ):
        raise PilotV211ParentImportError(
            "V2.10.2 evidence aggregate identity drifted"
        )

    csv_binding = release["aggregate_csv"]
    _bound_json_or_bytes(
        evidence_root,
        csv_binding,
        required_top="evidence",
        name="V2.10.2 evidence aggregate CSV",
    )

    checksums_binding = release["checksums"]
    checksums = _bound_json_file(
        evidence_root,
        checksums_binding,
        required_top="evidence",
        name="V2.10.2 evidence checksums",
    )
    files = checksums.get("files")
    file_index = {
        row.get("path"): row
        for row in files
        if isinstance(row, Mapping)
    } if isinstance(files, list) else {}
    if (
        checksums.get("schema_version")
        != "finevo-pilot-package-checksums-v1"
        or checksums.get("contract_sha256") != V2102_CONTRACT_SHA256
        or not isinstance(files, list)
        or canonical_sha256(files)
        != checksums_binding.get("files_canonical_sha256")
        or file_index.get("aggregate.json", {}).get("sha256")
        != aggregate_binding["sha256"]
        or file_index.get("aggregate.json", {}).get("byte_size")
        != aggregate_binding["byte_size"]
        or file_index.get("aggregate.csv", {}).get("sha256")
        != csv_binding["sha256"]
        or file_index.get("aggregate.csv", {}).get("byte_size")
        != csv_binding["byte_size"]
    ):
        raise PilotV211ParentImportError(
            "V2.10.2 evidence checksums identity drifted"
        )

    package_binding = release["package_manifest"]
    package = _bound_json_file(
        evidence_root,
        package_binding,
        required_top="evidence",
        name="V2.10.2 evidence package manifest",
    )
    if (
        package.get("schema_version")
        != "finevo-pilot-v2.10.2-evidence-package-v1"
        or package.get("contract_id") != V2102_CONTRACT_ID
        or package.get("contract_sha256") != V2102_CONTRACT_SHA256
        or package.get("pilot_tag") != V2102_SCIENCE_TAG
        or package.get("resolved_git_commit") != V2102_SCIENCE_COMMIT
        or package.get("publication_status") != "complete-with-no-go"
        or package.get("scientific_complete") is not False
        or package.get("scientific_matrix_complete") is not False
        or sorted(package.get("published_files", []))
        != sorted(
            path for path in file_index if path != "package_manifest.json"
        )
    ):
        raise PilotV211ParentImportError(
            "V2.10.2 evidence package identity drifted"
        )
    return {
        "aggregate_json_sha256": aggregate_binding["sha256"],
        "aggregate_csv_sha256": csv_binding["sha256"],
        "checksums_sha256": checksums_binding["sha256"],
        "checksums_files_canonical_sha256": checksums_binding[
            "files_canonical_sha256"
        ],
        "package_manifest_sha256": package_binding["sha256"],
        "publication_status": "complete-with-no-go",
        "denominator_status_counts": {"complete": 126, "failed": 85},
    }


def _bound_json_or_bytes(
    root: Path,
    binding: Mapping[str, Any],
    *,
    required_top: str,
    name: str,
) -> bytes:
    relative = _normalized_relative(
        binding.get("path"),
        required_top=required_top,
        name=name,
    )
    _, raw = _guarded_file(root, relative, name=name)
    if (
        len(raw) != binding.get("byte_size")
        or _sha256(raw) != binding.get("sha256")
    ):
        raise PilotV211ParentImportError(f"{name} file identity drifted")
    return raw


def _audit_sources(
    *,
    repo_root: str | Path,
    science_repo_root: str | Path | None,
    evidence_repo_root: str | Path | None,
) -> dict[str, Any]:
    root = _real_root(repo_root, name="V2.11 repository")
    manifest = _load_source_manifest(root)
    science_root, evidence_root = _source_roots(
        root,
        manifest,
        science_repo_root=science_repo_root,
        evidence_repo_root=evidence_repo_root,
    )
    git = _verify_git_release(science_root)
    prerequisites = _verify_prerequisites(science_root, manifest)
    evidence = _verify_evidence(evidence_root, manifest)
    return {
        "repo_root": root,
        "manifest": manifest,
        "science_root": science_root,
        "evidence_root": evidence_root,
        "git": git,
        "prerequisites": prerequisites,
        "evidence": evidence,
    }


def _debit_from_manifest(manifest: Mapping[str, Any]) -> ParentBudgetDebit:
    source = manifest["parent_cumulative_budget_debit"]
    return ParentBudgetDebit(
        parent_contract_sha256=source["parent_contract_sha256"],
        parent_run_ledger_sha256=source["parent_run_ledger_sha256"],
        parent_budget_ledger_sha256=source[
            "parent_budget_ledger_sha256"
        ],
        stage_bucket=source["stage_bucket"],
        cost_usd=source["cost_usd"],
        hosted_completions=source["hosted_completions"],
        storage_bytes=source["storage_bytes"],
    )


def build_v211_parent_import(
    *,
    repo_root: str | Path,
    science_repo_root: str | Path | None = None,
    evidence_repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the immutable parent and build a small, hash-bound receipt."""

    audit = _audit_sources(
        repo_root=repo_root,
        science_repo_root=science_repo_root,
        evidence_repo_root=evidence_repo_root,
    )
    manifest = audit["manifest"]
    return _seal(
        {
            "schema_version": V211_PARENT_IMPORT_SCHEMA_VERSION,
            "source_manifest": {
                "path": V211_SOURCE_MANIFEST_PATH.as_posix(),
                "file_sha256": V211_SOURCE_MANIFEST_FILE_SHA256,
                "content_sha256": V211_SOURCE_MANIFEST_CONTENT_SHA256,
            },
            "parent_release": {
                **audit["git"],
                "contract_id": V2102_CONTRACT_ID,
                "contract_sha256": V2102_CONTRACT_SHA256,
                "run_ledger_sha256": V2102_RUN_LEDGER_SHA256,
                "budget_ledger_sha256": V2102_BUDGET_LEDGER_SHA256,
                "publication_status": "complete-with-no-go",
            },
            "imported_prerequisites": {
                "q_ref": V2102_Q_REF,
                "selected_utility_profile": _json_copy(
                    _EXPECTED_UTILITY_PROFILE
                ),
                "stage0_absolute_flow_utility_threshold": _json_copy(
                    manifest["import_allowlist"][
                        "stage0_absolute_flow_utility_threshold"
                    ]
                ),
                "source_bindings": audit["prerequisites"],
            },
            "evidence_package": audit["evidence"],
            "cumulative_budget_debit": _debit_from_manifest(
                manifest
            ).to_dict(),
            "import_policy": {
                **_ZERO_PROVIDER_BOUNDARY,
                "raw_tree_copied": False,
                "copied_file_count": 0,
                "copied_byte_count": 0,
            },
            "scientific_evidence": False,
            "evidence_use": (
                "V2.10.2 q-ref, Stage-0 utility calibration, absolute-flow "
                "threshold, and cumulative debit only; no imported A-D "
                "effect or observed-p95 authority."
            ),
        }
    )


def _load_receipt(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _json_copy(dict(value))
    path = Path(value).expanduser()
    try:
        root = path.parent.resolve(strict=True)
    except OSError as exc:
        raise PilotV211ParentImportError(
            "V2.11 parent import receipt parent is unavailable"
        ) from exc
    _, raw = _guarded_file(
        root,
        PurePosixPath(path.name),
        name="V2.11 parent import receipt",
    )
    return _strict_json(raw, name="V2.11 parent import receipt")


def verify_v211_parent_import_receipt(
    receipt: Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path,
    science_repo_root: str | Path | None = None,
    evidence_repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild the receipt from source bytes and require exact equality."""

    observed = _load_receipt(receipt)
    _verify_seal(
        observed,
        schema_version=V211_PARENT_IMPORT_SCHEMA_VERSION,
        name="V2.11 parent import receipt",
    )
    expected = build_v211_parent_import(
        repo_root=repo_root,
        science_repo_root=science_repo_root,
        evidence_repo_root=evidence_repo_root,
    )
    if observed != expected:
        raise PilotV211ParentImportError(
            "V2.11 parent import receipt differs from exact allowlist"
        )
    return observed


def _safe_destination(repo_root: Path, destination: str | Path | None) -> Path:
    if destination is None:
        relative = V211_DEFAULT_RECEIPT_PATH
    else:
        candidate = Path(destination)
        if candidate.is_absolute():
            try:
                relative = PurePosixPath(
                    *candidate.relative_to(repo_root).parts
                )
            except ValueError as exc:
                raise PilotV211ParentImportError(
                    "receipt destination escaped V2.11 repository"
                ) from exc
        else:
            relative = PurePosixPath(candidate.as_posix())
    normalized = _normalized_relative(
        relative.as_posix(),
        required_top="experiment_results",
        name="V2.11 receipt destination",
    )
    path = repo_root.joinpath(*normalized.parts)
    current = repo_root
    for part in normalized.parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise PilotV211ParentImportError(
                "receipt destination contains a symlink"
            )
    return path


def _persist_exact_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise PilotV211ParentImportError(
                "receipt destination is a symlink"
            )
        existing = path.read_bytes()
        if existing != raw:
            raise PilotV211ParentImportError(
                "existing receipt differs; refusing to overwrite"
            )
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
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


def persist_v211_parent_import(
    *,
    repo_root: str | Path,
    science_repo_root: str | Path | None = None,
    evidence_repo_root: str | Path | None = None,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Persist only the compact receipt; never copy the parent raw tree."""

    root = _real_root(repo_root, name="V2.11 repository")
    receipt = build_v211_parent_import(
        repo_root=root,
        science_repo_root=science_repo_root,
        evidence_repo_root=evidence_repo_root,
    )
    path = _safe_destination(root, destination)
    _persist_exact_json(path, receipt)
    verified = verify_v211_parent_import_receipt(
        path,
        repo_root=root,
        science_repo_root=science_repo_root,
        evidence_repo_root=evidence_repo_root,
    )
    raw = path.read_bytes()
    return {
        "receipt": str(path),
        "receipt_file_sha256": _sha256(raw),
        "receipt_content_sha256": verified["integrity"]["content_sha256"],
        "imported_effect_cells": 0,
        "effect_metrics_observed": False,
        "effect_artifact_paths": [],
        "imported_p95_authorities": [],
        "provider_construction": False,
        "provider_calls": 0,
        "raw_tree_copied": False,
        "copied_file_count": 0,
        "copied_byte_count": 0,
    }


def parent_budget_debit_for_v211(
    contract: Any = None,
    *,
    repo_root: str | Path,
    science_repo_root: str | Path | None = None,
    evidence_repo_root: str | Path | None = None,
) -> ParentBudgetDebit:
    """Return the exact cumulative parent debit after source verification."""

    contract_id = getattr(contract, "contract_id", None)
    if contract is not None and contract_id not in {
        "finevo-pilot-v2.11",
        "finevo-pilot-v2.11-prospective",
    }:
        raise PilotV211ParentImportError(
            "parent debit requires the V2.11 contract"
        )
    audit = _audit_sources(
        repo_root=repo_root,
        science_repo_root=science_repo_root,
        evidence_repo_root=evidence_repo_root,
    )
    return _debit_from_manifest(audit["manifest"])


# Explicit receipt-oriented aliases make the API legible at integration sites
# while preserving the shorter build/verify names requested by the V2.11 plan.
build_v211_parent_import_receipt = build_v211_parent_import
verify_v211_parent_import = verify_v211_parent_import_receipt


__all__ = [
    "PilotV211ParentImportError",
    "V211_DEFAULT_RECEIPT_PATH",
    "V211_PARENT_IMPORT_SCHEMA_VERSION",
    "V211_SOURCE_MANIFEST_CONTENT_SHA256",
    "V211_SOURCE_MANIFEST_FILE_SHA256",
    "V211_SOURCE_MANIFEST_PATH",
    "V211_SOURCE_MANIFEST_SCHEMA_VERSION",
    "build_v211_parent_import",
    "build_v211_parent_import_receipt",
    "parent_budget_debit_for_v211",
    "persist_v211_parent_import",
    "verify_v211_parent_import",
    "verify_v211_parent_import_receipt",
]
