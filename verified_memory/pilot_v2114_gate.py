"""Zero-provider V2.11.4 reseal and global dispatch gate.

The immutable V2.11.2 long-context preflight is useful only as a prospective
dispatch-budget authority.  This module reseals that authority into the
V2.11.4 raw namespace, one registered model per authority-import cell, and
then builds the two-model global post-gate receipt consumed by scientific
runs.  It never imports a provider adapter and never reclassifies a V2.11.2
scientific outcome.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping

from .pilot_contract import PilotContract, load_pilot_contract
from .pilot_v2114_parent_import import (
    PilotV2114ParentImportError,
    V2112_CONTRACT_ID,
    V2112_CONTRACT_SHA256,
    V2112_POST_GATE_CONTENT_SHA256,
    V2112_POST_GATE_FILE_SHA256,
    V2112_SCIENCE_COMMIT,
    V2112_SCIENCE_TAG,
    V2114_ALLOWED_MODELS,
    V2114_CONTRACT_ID,
    V2114_CONTRACT_PATH,
    V2114_DEFAULT_RECEIPT_PATH,
    V2114_RAW_ROOT,
    V2114_SCIENCE_TAG,
    V2114_SOURCE_MANIFEST_CONTENT_SHA256,
    V2114_SOURCE_MANIFEST_FILE_SHA256,
    V2114_SOURCE_MANIFEST_PATH,
    _atomic_json,
    _json_copy,
    _read_regular,
    _sha256,
    _strict_json,
    _strict_root,
    _validate_child,
    _verify_seal,
    calibration_wrapper_from_v2114_receipt,
    capability_wrappers_from_v2114_receipt,
    preflight_wrappers_from_v2114_receipt,
    validate_v2114_parent_import_receipt,
)
from .runner import (
    OBSERVED_P95_AUTHORITY_ID,
    OBSERVED_P95_PROJECTION_SCHEMA_VERSION,
    OBSERVED_P95_SOURCE_KIND,
    ObservedPreflightP95Reservation,
    PreflightP95Reservation,
)


V2114_RESEALED_P95_AUTHORITY_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.4-resealed-observed-p95-authority-v1"
)
V2114_RESEALED_P95_SOURCE_KIND = "v2.11.2-fresh-preflight-v2.11.4"
V2114_PREFLIGHT_IMPORT_GATE_SCHEMA_VERSION = (
    "finevo-pilot-v2.11.4-preflight-authority-import-gate-v1"
)
V2114_GATE_SCHEMA_VERSION = "finevo-pilot-v2.11.4-post-gate-authority-v1"
V2114_POST_GATE_RELATIVE_PATH = (
    V2114_RAW_ROOT / "long-context-preflight/post_gate_authority.json"
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_MODELS = {
    "gpt52_main": "openai/gpt-5.2-2025-12-11",
    "gpt56_diagnostic": "openai/gpt-5.6-sol",
}
_SAMPLE_COUNTS = {"action": 24, "semantic": 8}
_ZERO_PROVIDER = {
    "provider_construction_during_reseal": False,
    "provider_calls_during_reseal": 0,
    "hosted_provider_calls_during_reseal": 0,
    "hosted_cost_usd_during_reseal": 0.0,
}


class PilotV2114GateError(RuntimeError):
    """Raised before a V2.11.4 dispatch authority can be consumed."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_copy(dict(value))
    result.pop("integrity", None)
    result["integrity"] = {"canonicalization": "json-sort-keys-utf8-v1"}
    result["integrity"]["content_sha256"] = canonical_sha256(result)
    return result


def _receipt_seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_copy(dict(value))
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotV2114GateError(f"{name} must be an object")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PilotV2114GateError(f"{name} must be a lowercase SHA-256")
    return value


def _commit(value: Any, name: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise PilotV2114GateError(f"{name} must be a lowercase git commit")
    return value


def _raw_root(repo_root: Path, raw_root: str | Path) -> Path:
    candidate = Path(raw_root)
    if not candidate.is_absolute():
        candidate = repo_root.joinpath(*PurePosixPath(str(raw_root)).parts)
    candidate = candidate.absolute()
    expected = repo_root.joinpath(*V2114_RAW_ROOT.parts)
    if candidate != expected:
        raise PilotV2114GateError(
            "V2.11.4 observed-p95 authority requires its exact raw namespace"
        )
    return candidate


def _relative_to_repo(repo_root: Path, path: str | Path, *, name: str) -> PurePosixPath:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.absolute().relative_to(repo_root)
        except ValueError as exc:
            raise PilotV2114GateError(f"{name} escaped the repository") from exc
    text = str(candidate)
    relative = PurePosixPath(text)
    if (
        not text
        or "\\" in text
        or "\x00" in text
        or relative.is_absolute()
        or relative.parts[0] != "experiment_results"
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != text
    ):
        raise PilotV2114GateError(f"{name} is not a normalized raw path")
    return relative


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotV2114GateError(
            "V2.11.4 git release identity is unavailable"
        ) from exc
    return result.stdout.strip()


def _verify_release_identity(repo_root: Path, expected_git_commit: str) -> None:
    commit = _commit(expected_git_commit, "V2.11.4 release commit")
    if (
        _git(repo_root, "rev-parse", f"refs/tags/{V2114_SCIENCE_TAG}^{{object}}")
        == commit
    ):
        # An annotated tag object must not equal its peeled commit.
        raise PilotV2114GateError("V2.11.4 science tag is not annotated")
    peeled = _git(repo_root, "rev-parse", f"refs/tags/{V2114_SCIENCE_TAG}^{{commit}}")
    tag_type = _git(repo_root, "cat-file", "-t", f"refs/tags/{V2114_SCIENCE_TAG}")
    if peeled != commit or tag_type != "tag":
        raise PilotV2114GateError("V2.11.4 annotated tag/commit binding drifted")


def _contract(
    repo_root: Path,
    selected: PilotContract | None,
    expected_git_commit: str,
) -> PilotContract:
    contract = selected or load_pilot_contract(
        repo_root.joinpath(*V2114_CONTRACT_PATH.parts)
    )
    try:
        _validate_child(
            contract=contract,
            child_git_commit=expected_git_commit,
            require_frozen=True,
        )
    except PilotV2114ParentImportError as exc:
        raise PilotV2114GateError(str(exc)) from exc
    body = contract.to_dict()
    forward = _mapping(
        body.get("v2114_forward_boundary"),
        "V2.11.4 forward boundary",
    )
    source = _mapping(forward.get("source_manifest"), "V2.11.4 source manifest")
    if source != {
        "path": V2114_SOURCE_MANIFEST_PATH.as_posix(),
        "schema_version": "finevo-pilot-v2.11.4-source-manifest-v1",
        "file_sha256": V2114_SOURCE_MANIFEST_FILE_SHA256,
        "content_sha256": V2114_SOURCE_MANIFEST_CONTENT_SHA256,
    }:
        raise PilotV2114GateError("V2.11.4 contract/source-manifest binding drifted")
    return contract


def _contract_binding(repo_root: Path, contract: PilotContract) -> dict[str, Any]:
    raw = _read_regular(
        repo_root,
        V2114_CONTRACT_PATH,
        name="V2.11.4 contract",
    )
    observed = _strict_json(raw, name="V2.11.4 contract")
    if observed != contract.to_dict():
        raise PilotV2114GateError(
            "V2.11.4 contract bytes differ from selected contract"
        )
    return {
        "path": V2114_CONTRACT_PATH.as_posix(),
        "file_sha256": _sha256(raw),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
    }


def _load_parent_receipt(
    repo_root: Path,
    *,
    contract: PilotContract,
    expected_git_commit: str,
    supplied: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    relative = V2114_DEFAULT_RECEIPT_PATH
    raw = _read_regular(repo_root, relative, name="V2.11.4 parent import receipt")
    value = _strict_json(raw, name="V2.11.4 parent import receipt")
    if supplied is not None and value != _json_copy(dict(supplied)):
        raise PilotV2114GateError(
            "supplied V2.11.4 parent receipt differs from persisted bytes"
        )
    try:
        verified = validate_v2114_parent_import_receipt(
            value,
            contract=contract,
            child_git_commit=expected_git_commit,
            repo_root=repo_root,
        )
    except PilotV2114ParentImportError as exc:
        raise PilotV2114GateError(
            f"V2.11.4 parent import receipt failed: {exc}"
        ) from exc
    return verified, {
        "path": relative.as_posix(),
        "file_sha256": _sha256(raw),
        "content_sha256": str(verified["integrity"]["content_sha256"]),
    }


def v2114_observed_p95_receipt_path(
    raw_root: str | Path,
    profile_id: str,
) -> Path:
    if profile_id not in V2114_ALLOWED_MODELS:
        raise PilotV2114GateError(f"unsupported V2.11.4 p95 profile: {profile_id}")
    return (
        Path(raw_root)
        / "long-context-preflight"
        / "imported_observed_p95"
        / profile_id
        / "observed_p95_authority_receipt.json"
    )


def v2114_observed_p95_projection_path(
    raw_root: str | Path,
    profile_id: str,
) -> Path:
    return v2114_observed_p95_receipt_path(raw_root, profile_id).with_name(
        "projection_p95.json"
    )


def _runtime_profile(contract: PilotContract, profile_id: str) -> tuple[str, str]:
    if profile_id not in contract.provider_profiles:
        raise PilotV2114GateError("V2.11.4 p95 profile is absent from the contract")
    profile = contract.provider_profiles[profile_id]
    runtime = f"{profile.transport}/{profile.requested_model}"
    if runtime != _RUNTIME_MODELS[profile_id]:
        raise PilotV2114GateError("V2.11.4 runtime model differs from parent authority")
    return runtime, profile.served_model


def _inherited_source_reservations(
    *,
    wrapper: Mapping[str, Any],
    contract: PilotContract,
    profile_id: str,
) -> dict[str, dict[str, Any]]:
    runtime, _ = _runtime_profile(contract, profile_id)
    source_gate = _mapping(wrapper.get("source_gate_receipt"), "source gate receipt")
    by_kind = _mapping(wrapper.get("reservations"), "source p95 reservations")
    if set(by_kind) != {"action", "semantic"}:
        raise PilotV2114GateError("V2.11.4 source p95 call-kind denominator drifted")
    result: dict[str, Any] = {}
    for call_kind in ("action", "semantic"):
        source_entry = _mapping(by_kind[call_kind], f"{call_kind} source reservation")
        if set(source_entry) != {"authority", "reservation"}:
            raise PilotV2114GateError("V2.11.4 source p95 entry shape drifted")
        authority = dict(_mapping(source_entry["authority"], "source p95 authority"))
        authority.update(
            {
                "source_authority_receipt_path": str(source_gate["path"]),
                "source_authority_receipt_file_sha256": str(source_gate["file_sha256"]),
                "source_authority_receipt_content_sha256": str(
                    source_gate["content_sha256"]
                ),
                "source_release_commit": str(source_gate["git_commit"]),
            }
        )
        candidate = {
            "authority": authority,
            "reservation": _json_copy(source_entry["reservation"]),
        }
        try:
            parsed = ObservedPreflightP95Reservation.from_dict(
                model=runtime,
                call_kind=call_kind,
                value=candidate,
            ).to_dict()
        except (TypeError, ValueError) as exc:
            raise PilotV2114GateError(
                f"V2.11.4 {profile_id}/{call_kind} p95 authority is invalid"
            ) from exc
        if (
            parsed["authority"]["authority_id"] != OBSERVED_P95_AUTHORITY_ID
            or parsed["authority"]["source_kind"] != OBSERVED_P95_SOURCE_KIND
            or parsed["authority"]["pilot_contract_hash"] != V2112_CONTRACT_SHA256
            or parsed["authority"]["pilot_tag"] != V2112_SCIENCE_TAG
            or parsed["authority"]["source_projection_schema_version"]
            != OBSERVED_P95_PROJECTION_SCHEMA_VERSION
            or parsed["authority"]["source_release_commit"] != V2112_SCIENCE_COMMIT
            or parsed["authority"]["source_authority_receipt_file_sha256"]
            != V2112_POST_GATE_FILE_SHA256
            or parsed["authority"]["source_authority_receipt_content_sha256"]
            != V2112_POST_GATE_CONTENT_SHA256
            or parsed["reservation"]["sample_count"] != _SAMPLE_COUNTS[call_kind]
        ):
            raise PilotV2114GateError("V2.11.4 inherited p95 provenance drifted")
        result[call_kind] = parsed
    return {runtime: result}


def build_v2114_resealed_observed_p95_authority(
    *,
    repo_root: str | Path,
    contract: PilotContract,
    raw_root: str | Path,
    profile_id: str,
    expected_git_commit: str,
    parent_import_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one model's child-bound receipt/projection pair, zero-call."""

    root = _strict_root(repo_root, name="V2.11.4 repository")
    selected = _contract(root, contract, expected_git_commit)
    raw = _raw_root(root, raw_root)
    parent, parent_binding = _load_parent_receipt(
        root,
        contract=selected,
        expected_git_commit=expected_git_commit,
        supplied=parent_import_receipt,
    )
    wrappers = preflight_wrappers_from_v2114_receipt(parent)
    wrapper = wrappers[profile_id]
    runtime, served = _runtime_profile(selected, profile_id)
    reservations = _inherited_source_reservations(
        wrapper=wrapper,
        contract=selected,
        profile_id=profile_id,
    )
    receipt = _seal(
        {
            "schema_version": V2114_RESEALED_P95_AUTHORITY_SCHEMA_VERSION,
            "contract": _contract_binding(root, selected),
            "source_manifest": {
                "path": V2114_SOURCE_MANIFEST_PATH.as_posix(),
                "file_sha256": V2114_SOURCE_MANIFEST_FILE_SHA256,
                "content_sha256": V2114_SOURCE_MANIFEST_CONTENT_SHA256,
            },
            "raw_root": V2114_RAW_ROOT.as_posix(),
            "git": {"tag": V2114_SCIENCE_TAG, "commit": expected_git_commit},
            "model": {
                "model_id": profile_id,
                "runtime_model": runtime,
                "served_model": served,
            },
            "parent_import_receipt": parent_binding,
            "source_preflight_wrapper": {
                "content_sha256": wrapper["integrity"]["content_sha256"],
                "source_contract_id": V2112_CONTRACT_ID,
                "source_contract_sha256": V2112_CONTRACT_SHA256,
                "source_git_tag": V2112_SCIENCE_TAG,
                "source_git_commit": V2112_SCIENCE_COMMIT,
                "historical_sample_counts": dict(_SAMPLE_COUNTS),
                "historical_calls_already_in_parent_debit": True,
            },
            "reservations": reservations,
            "provider_boundary": dict(_ZERO_PROVIDER),
            "scientific_evidence": False,
            "evidence_use": (
                "Prospective V2.11.4 dispatch-budget authority only; no "
                "V2.11.2 scientific outcome or decoded completion is reused."
            ),
        }
    )
    receipt_path = v2114_observed_p95_receipt_path(raw, profile_id)
    receipt_relative = _relative_to_repo(root, receipt_path, name="p95 receipt")
    projection = _seal(
        {
            "schema_version": OBSERVED_P95_PROJECTION_SCHEMA_VERSION,
            "model_id": profile_id,
            "served_model": served,
            "projection": {
                f"{served}::{call_kind}": _json_copy(
                    reservations[runtime][call_kind]["reservation"]
                )
                for call_kind in ("action", "semantic")
            },
            "bindings": {
                "contract_sha256": selected.canonical_hash,
                "git_tag": V2114_SCIENCE_TAG,
                "git_commit": expected_git_commit,
                "source_kind": V2114_RESEALED_P95_SOURCE_KIND,
                "source_authority_receipt": receipt_relative.as_posix(),
                "source_authority_receipt_content_sha256": receipt["integrity"][
                    "content_sha256"
                ],
                "source_parent_import_receipt_content_sha256": parent_binding[
                    "content_sha256"
                ],
                "source_v2_11_2_gate_file_sha256": V2112_POST_GATE_FILE_SHA256,
                "source_v2_11_2_gate_content_sha256": (V2112_POST_GATE_CONTENT_SHA256),
            },
        }
    )
    return {
        "receipt_path": receipt_path,
        "projection_path": v2114_observed_p95_projection_path(raw, profile_id),
        "receipt": receipt,
        "projection": projection,
    }


def _rebuild_pair(
    receipt: Mapping[str, Any],
    *,
    repo_root: Path,
    raw_root: Path,
    contract: PilotContract,
    expected_git_commit: str,
) -> dict[str, Any]:
    value = _json_copy(dict(receipt))
    expected_keys = {
        "schema_version",
        "contract",
        "source_manifest",
        "raw_root",
        "git",
        "model",
        "parent_import_receipt",
        "source_preflight_wrapper",
        "reservations",
        "provider_boundary",
        "scientific_evidence",
        "evidence_use",
        "integrity",
    }
    if set(value) != expected_keys:
        raise PilotV2114GateError("V2.11.4 p95 receipt shape drifted")
    try:
        _verify_seal(
            value,
            schema=V2114_RESEALED_P95_AUTHORITY_SCHEMA_VERSION,
            name="V2.11.4 resealed p95 receipt",
        )
    except PilotV2114ParentImportError as exc:
        raise PilotV2114GateError(str(exc)) from exc
    model = _mapping(value.get("model"), "V2.11.4 p95 model")
    profile_id = str(model.get("model_id"))
    parent, _ = _load_parent_receipt(
        repo_root,
        contract=contract,
        expected_git_commit=expected_git_commit,
    )
    rebuilt = build_v2114_resealed_observed_p95_authority(
        repo_root=repo_root,
        contract=contract,
        raw_root=raw_root,
        profile_id=profile_id,
        expected_git_commit=expected_git_commit,
        parent_import_receipt=parent,
    )
    if value != rebuilt["receipt"]:
        raise PilotV2114GateError(
            "V2.11.4 p95 receipt differs from current parent-import replay"
        )
    return rebuilt


def persist_v2114_resealed_observed_p95_authority(
    *,
    repo_root: str | Path,
    contract: PilotContract,
    raw_root: str | Path,
    profile_id: str | None = None,
    model_id: str | None = None,
    expected_git_commit: str,
    parent_import_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist exactly one preflight-import cell's receipt/projection pair."""

    selected_id = profile_id if profile_id is not None else model_id
    if selected_id is None or (
        model_id is not None and profile_id not in {None, model_id}
    ):
        raise PilotV2114GateError("one exact V2.11.4 profile/model id is required")
    built = build_v2114_resealed_observed_p95_authority(
        repo_root=repo_root,
        contract=contract,
        raw_root=raw_root,
        profile_id=selected_id,
        expected_git_commit=expected_git_commit,
        parent_import_receipt=parent_import_receipt,
    )
    _atomic_json(built["receipt_path"], built["receipt"], repo_root=repo_root)
    _atomic_json(built["projection_path"], built["projection"], repo_root=repo_root)
    gate = _seal(
        {
            "schema_version": V2114_PREFLIGHT_IMPORT_GATE_SCHEMA_VERSION,
            "contract_id": V2114_CONTRACT_ID,
            "contract_sha256": contract.canonical_hash,
            "model_id": selected_id,
            "capability_pass": True,
            "interface_pass": True,
            "go": True,
            "historical_action_samples": 24,
            "historical_semantic_samples": 8,
            "historical_provider_calls": 32,
            "provider_construction_current_attempt": False,
            "provider_calls_current_attempt": 0,
            "scientific_evidence": False,
            "authority_receipt": str(built["receipt_path"]),
            "projection_p95": str(built["projection_path"]),
        }
    )
    return {
        "receipt": str(built["receipt_path"]),
        "projection": str(built["projection_path"]),
        "receipt_content_sha256": built["receipt"]["integrity"]["content_sha256"],
        "projection_content_sha256": built["projection"]["integrity"]["content_sha256"],
        "gate_receipt": gate,
        **_ZERO_PROVIDER,
        "scientific_evidence": False,
    }


def verify_v2114_resealed_observed_p95_authority(
    receipt: Mapping[str, Any],
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    expected_git_commit: str,
    contract: PilotContract | None = None,
) -> dict[str, Any]:
    root = _strict_root(repo_root, name="V2.11.4 repository")
    _verify_release_identity(root, expected_git_commit)
    selected = _contract(root, contract, expected_git_commit)
    raw = _raw_root(root, raw_root)
    built = _rebuild_pair(
        receipt,
        repo_root=root,
        raw_root=raw,
        contract=selected,
        expected_git_commit=expected_git_commit,
    )
    return _json_copy(built["receipt"]["reservations"])


def verify_v2114_resealed_observed_p95_projection(
    projection: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    repo_root: str | Path,
    raw_root: str | Path,
    expected_git_commit: str,
    contract: PilotContract | None = None,
) -> dict[str, Any]:
    root = _strict_root(repo_root, name="V2.11.4 repository")
    _verify_release_identity(root, expected_git_commit)
    selected = _contract(root, contract, expected_git_commit)
    raw = _raw_root(root, raw_root)
    built = _rebuild_pair(
        receipt,
        repo_root=root,
        raw_root=raw,
        contract=selected,
        expected_git_commit=expected_git_commit,
    )
    candidate = _json_copy(dict(projection))
    try:
        _verify_seal(
            candidate,
            schema=OBSERVED_P95_PROJECTION_SCHEMA_VERSION,
            name="V2.11.4 p95 projection",
        )
    except PilotV2114ParentImportError as exc:
        raise PilotV2114GateError(str(exc)) from exc
    if candidate != built["projection"]:
        raise PilotV2114GateError("V2.11.4 p95 projection differs from receipt")
    return candidate


def verified_v2114_observed_p95_authority_binding(
    receipt_path: str | Path,
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    expected_git_commit: str,
    contract: PilotContract | None = None,
) -> dict[str, Any]:
    root = _strict_root(repo_root, name="V2.11.4 repository")
    raw = _raw_root(root, raw_root)
    relative = _relative_to_repo(root, receipt_path, name="V2.11.4 p95 receipt")
    profile_id = relative.parts[-2] if len(relative.parts) >= 2 else ""
    expected = _relative_to_repo(
        root,
        v2114_observed_p95_receipt_path(raw, profile_id),
        name="V2.11.4 p95 receipt",
    )
    if relative != expected:
        raise PilotV2114GateError("V2.11.4 p95 receipt path drifted")
    receipt_raw = _read_regular(root, relative, name="V2.11.4 p95 receipt")
    receipt = _strict_json(receipt_raw, name="V2.11.4 p95 receipt")
    reservations = verify_v2114_resealed_observed_p95_authority(
        receipt,
        repo_root=root,
        raw_root=raw,
        expected_git_commit=expected_git_commit,
        contract=contract,
    )
    projection_relative = relative.with_name("projection_p95.json")
    projection_raw = _read_regular(
        root, projection_relative, name="V2.11.4 p95 projection"
    )
    projection = _strict_json(projection_raw, name="V2.11.4 p95 projection")
    verify_v2114_resealed_observed_p95_projection(
        projection,
        receipt=receipt,
        repo_root=root,
        raw_root=raw,
        expected_git_commit=expected_git_commit,
        contract=contract,
    )
    return {
        "receipt_path": relative.as_posix(),
        "receipt_file_sha256": _sha256(receipt_raw),
        "receipt_content_sha256": receipt["integrity"]["content_sha256"],
        "git_commit": expected_git_commit,
        "reservations": reservations,
    }


def verified_v2114_observed_p95_projection_binding(
    projection_path: str | Path,
    *,
    receipt_path: str | Path,
    repo_root: str | Path,
    raw_root: str | Path,
    expected_git_commit: str,
    contract: PilotContract | None = None,
) -> dict[str, Any]:
    authority = verified_v2114_observed_p95_authority_binding(
        receipt_path,
        repo_root=repo_root,
        raw_root=raw_root,
        expected_git_commit=expected_git_commit,
        contract=contract,
    )
    root = _strict_root(repo_root, name="V2.11.4 repository")
    projection_relative = _relative_to_repo(
        root, projection_path, name="V2.11.4 p95 projection"
    )
    if projection_relative != PurePosixPath(authority["receipt_path"]).with_name(
        "projection_p95.json"
    ):
        raise PilotV2114GateError("V2.11.4 p95 sibling projection path drifted")
    raw = _read_regular(root, projection_relative, name="V2.11.4 p95 projection")
    projection = _strict_json(raw, name="V2.11.4 p95 projection")
    receipt_raw = _read_regular(
        root, PurePosixPath(authority["receipt_path"]), name="V2.11.4 p95 receipt"
    )
    receipt = _strict_json(receipt_raw, name="V2.11.4 p95 receipt")
    payload = verify_v2114_resealed_observed_p95_projection(
        projection,
        receipt=receipt,
        repo_root=root,
        raw_root=raw_root,
        expected_git_commit=expected_git_commit,
        contract=contract,
    )
    model = _mapping(receipt["model"], "V2.11.4 p95 model")
    return {
        "projection_path": projection_relative.as_posix(),
        "projection_file_sha256": _sha256(raw),
        "projection_content_sha256": payload["integrity"]["content_sha256"],
        "profile_id": model["model_id"],
        "served_model": model["served_model"],
        "runtime_model": model["runtime_model"],
        "reservations": _json_copy(authority["reservations"]),
        "source_contract_id": V2112_CONTRACT_ID,
        "source_contract_sha256": V2112_CONTRACT_SHA256,
        "source_git_tag": V2112_SCIENCE_TAG,
        "source_git_commit": V2112_SCIENCE_COMMIT,
        "git_commit": expected_git_commit,
        "payload": payload,
    }


def _normalize_model_bindings(
    bindings: Mapping[str, Any],
    *,
    contract: PilotContract,
    expected_git_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(bindings) != set(V2114_ALLOWED_MODELS):
        raise PilotV2114GateError("V2.11.4 global gate model denominator drifted")
    reservations: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for model_id in V2114_ALLOWED_MODELS:
        binding = _mapping(bindings[model_id], f"{model_id} p95 binding")
        if set(binding) != {
            "receipt_path",
            "receipt_file_sha256",
            "receipt_content_sha256",
            "git_commit",
            "reservations",
        }:
            raise PilotV2114GateError("V2.11.4 per-model p95 binding shape drifted")
        runtime, _ = _runtime_profile(contract, model_id)
        rows = _mapping(binding["reservations"], f"{model_id} reservations")
        if set(rows) != {runtime}:
            raise PilotV2114GateError("V2.11.4 p95 runtime denominator drifted")
        parsed: dict[str, ObservedPreflightP95Reservation] = {}
        for call_kind in ("action", "semantic"):
            try:
                parsed[call_kind] = ObservedPreflightP95Reservation.from_dict(
                    model=runtime,
                    call_kind=call_kind,
                    value=rows[runtime][call_kind],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise PilotV2114GateError("V2.11.4 global p95 row is invalid") from exc
        if binding["git_commit"] != expected_git_commit:
            raise PilotV2114GateError("V2.11.4 p95 binding commit drifted")
        if any(
            getattr(parsed["action"], field) != getattr(parsed["semantic"], field)
            for field in (
                "source_preflight_run_id",
                "source_preflight_run_spec_sha256",
                "source_model_id",
                "source_served_model",
                "source_execution_artifact_sha256",
                "source_provider_call_journal_sha256",
            )
        ):
            raise PilotV2114GateError(
                "V2.11.4 action/semantic preflight provenance disagrees"
            )
        sources[model_id] = {
            "per_model_receipt": {
                "path": str(binding["receipt_path"]),
                "file_sha256": _digest(
                    binding["receipt_file_sha256"],
                    f"{model_id} receipt file hash",
                ),
                "content_sha256": _digest(
                    binding["receipt_content_sha256"],
                    f"{model_id} receipt content hash",
                ),
            },
            "source_preflight": {
                "run_id": parsed["action"].source_preflight_run_id,
                "run_spec_sha256": (parsed["action"].source_preflight_run_spec_sha256),
                "model_id": parsed["action"].source_model_id,
                "served_model": parsed["action"].source_served_model,
                "execution_artifact_sha256": (
                    parsed["action"].source_execution_artifact_sha256
                ),
                "provider_call_journal_sha256": (
                    parsed["action"].source_provider_call_journal_sha256
                ),
            },
            "v2_11_2_gate": {
                "tag": V2112_SCIENCE_TAG,
                "commit": V2112_SCIENCE_COMMIT,
                "file_sha256": V2112_POST_GATE_FILE_SHA256,
                "content_sha256": V2112_POST_GATE_CONTENT_SHA256,
            },
        }
        reservations[runtime] = {
            call_kind: _json_copy(rows[runtime][call_kind]["reservation"])
            for call_kind in ("action", "semantic")
        }
    return reservations, sources


def build_v2114_post_gate_authority(
    *,
    repo_root: str | Path,
    contract: PilotContract,
    expected_git_commit: str,
    parent_import_receipt: Mapping[str, Any],
    per_model_authority_bindings: Mapping[str, Any],
    ledger_event_chain_head: str,
) -> dict[str, Any]:
    """Build the two-model global authority after both import cells complete."""

    root = _strict_root(repo_root, name="V2.11.4 repository")
    selected = _contract(root, contract, expected_git_commit)
    _digest(ledger_event_chain_head, "V2.11.4 run-ledger event-chain head")
    persisted_parent, _ = _load_parent_receipt(
        root,
        contract=selected,
        expected_git_commit=expected_git_commit,
        supplied=parent_import_receipt,
    )
    reservations, sources = _normalize_model_bindings(
        per_model_authority_bindings,
        contract=selected,
        expected_git_commit=expected_git_commit,
    )
    capability = capability_wrappers_from_v2114_receipt(persisted_parent)
    calibration = calibration_wrapper_from_v2114_receipt(persisted_parent)
    parent_content = persisted_parent["integrity"]["content_sha256"]
    return _receipt_seal(
        {
            "schema_version": V2114_GATE_SCHEMA_VERSION,
            "contract_id": V2114_CONTRACT_ID,
            "contract_sha256": selected.canonical_hash,
            "release": {"tag": V2114_SCIENCE_TAG, "commit": expected_git_commit},
            "denominator": {
                "registered_model_ids": list(V2114_ALLOWED_MODELS),
                "eligible_model_ids": list(V2114_ALLOWED_MODELS),
                "registered_authority_import_cells": 2,
                "complete_authority_import_cells": 2,
                "historical_action_samples_per_model": 24,
                "historical_semantic_samples_per_model": 8,
            },
            "capability_wrapper_content_sha256": {
                model_id: capability[model_id]["integrity"]["content_sha256"]
                for model_id in V2114_ALLOWED_MODELS
            },
            "calibration_wrapper_content_sha256": calibration["integrity"][
                "content_sha256"
            ],
            "authority_sources": sources,
            "reservations": reservations,
            "bindings": {
                "parent_import_receipt_content_sha256": parent_content,
                "source_v2_11_2_gate_file_sha256": V2112_POST_GATE_FILE_SHA256,
                "source_v2_11_2_gate_content_sha256": (V2112_POST_GATE_CONTENT_SHA256),
                "ledger_event_chain_head": ledger_event_chain_head,
            },
            "provider_boundary": {
                "provider_construction_during_authority_import": False,
                "provider_calls_during_authority_import": 0,
                "hosted_provider_calls_during_authority_import": 0,
                "hosted_cost_usd_during_authority_import": 0.0,
            },
            "go": True,
            "reasons": [],
            "scientific_evidence": False,
            "claim_boundary": (
                "V2.11.4 prospective dispatch-budget authority only; all 131 "
                "scientific cells remain fresh V2.11.4 denominator cells."
            ),
        }
    )


def verify_v2114_gate_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_git_commit: str,
    expected_contract_sha256: str | None = None,
) -> dict[str, Any]:
    value = _json_copy(dict(receipt))
    expected_keys = {
        "schema_version",
        "contract_id",
        "contract_sha256",
        "release",
        "denominator",
        "capability_wrapper_content_sha256",
        "calibration_wrapper_content_sha256",
        "authority_sources",
        "reservations",
        "bindings",
        "provider_boundary",
        "go",
        "reasons",
        "scientific_evidence",
        "claim_boundary",
        "receipt_sha256",
    }
    observed_hash = value.pop("receipt_sha256", None)
    if (
        set(receipt) != expected_keys
        or receipt.get("schema_version") != V2114_GATE_SCHEMA_VERSION
        or observed_hash != canonical_sha256(value)
        or receipt.get("contract_id") != V2114_CONTRACT_ID
        or receipt.get("release")
        != {"tag": V2114_SCIENCE_TAG, "commit": expected_git_commit}
        or receipt.get("go") is not True
        or receipt.get("reasons") != []
        or receipt.get("scientific_evidence") is not False
    ):
        raise PilotV2114GateError("V2.11.4 global gate shape/seal drifted")
    contract_hash = _digest(receipt.get("contract_sha256"), "contract hash")
    if expected_contract_sha256 is not None and contract_hash != _digest(
        expected_contract_sha256, "expected contract hash"
    ):
        raise PilotV2114GateError("V2.11.4 global gate contract hash drifted")
    denominator = _mapping(receipt.get("denominator"), "global denominator")
    if denominator != {
        "registered_model_ids": list(V2114_ALLOWED_MODELS),
        "eligible_model_ids": list(V2114_ALLOWED_MODELS),
        "registered_authority_import_cells": 2,
        "complete_authority_import_cells": 2,
        "historical_action_samples_per_model": 24,
        "historical_semantic_samples_per_model": 8,
    }:
        raise PilotV2114GateError("V2.11.4 global gate denominator drifted")
    boundary = _mapping(receipt.get("provider_boundary"), "provider boundary")
    if boundary != {
        "provider_construction_during_authority_import": False,
        "provider_calls_during_authority_import": 0,
        "hosted_provider_calls_during_authority_import": 0,
        "hosted_cost_usd_during_authority_import": 0.0,
    }:
        raise PilotV2114GateError("V2.11.4 global gate is not zero-provider")
    sources = _mapping(receipt.get("authority_sources"), "authority sources")
    if set(sources) != set(V2114_ALLOWED_MODELS):
        raise PilotV2114GateError("V2.11.4 global authority sources drifted")
    rows = _mapping(receipt.get("reservations"), "global reservations")
    if set(rows) != set(_RUNTIME_MODELS.values()):
        raise PilotV2114GateError("V2.11.4 global reservation denominator drifted")
    for model_id, runtime in _RUNTIME_MODELS.items():
        by_kind = _mapping(rows[runtime], f"{model_id} reservations")
        if set(by_kind) != {"action", "semantic"}:
            raise PilotV2114GateError("V2.11.4 global call-kind denominator drifted")
        for kind in ("action", "semantic"):
            try:
                parsed = PreflightP95Reservation.from_dict(
                    model=runtime, call_kind=kind, value=by_kind[kind]
                )
            except (TypeError, ValueError) as exc:
                raise PilotV2114GateError(
                    "V2.11.4 global reservation is invalid"
                ) from exc
            if parsed.sample_count != _SAMPLE_COUNTS[kind]:
                raise PilotV2114GateError("V2.11.4 global sample count drifted")
        source = _mapping(sources[model_id], f"{model_id} authority source")
        if set(source) != {
            "per_model_receipt",
            "source_preflight",
            "v2_11_2_gate",
        }:
            raise PilotV2114GateError("V2.11.4 global source shape drifted")
        parent_gate = _mapping(source["v2_11_2_gate"], "V2.11.2 gate source")
        if parent_gate != {
            "tag": V2112_SCIENCE_TAG,
            "commit": V2112_SCIENCE_COMMIT,
            "file_sha256": V2112_POST_GATE_FILE_SHA256,
            "content_sha256": V2112_POST_GATE_CONTENT_SHA256,
        }:
            raise PilotV2114GateError("V2.11.4 parent gate lineage drifted")
    return _json_copy(dict(receipt))


def persist_v2114_post_gate_authority(
    *,
    repo_root: str | Path,
    raw_root: str | Path,
    contract: PilotContract,
    expected_git_commit: str,
    parent_import_receipt: Mapping[str, Any],
    ledger_event_chain_head: str,
    per_model_authority_bindings: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = _strict_root(repo_root, name="V2.11.4 repository")
    raw = _raw_root(root, raw_root)
    _verify_release_identity(root, expected_git_commit)
    selected = _contract(root, contract, expected_git_commit)
    bindings = (
        {
            model_id: verified_v2114_observed_p95_authority_binding(
                v2114_observed_p95_receipt_path(raw, model_id),
                repo_root=root,
                raw_root=raw,
                expected_git_commit=expected_git_commit,
                contract=selected,
            )
            for model_id in V2114_ALLOWED_MODELS
        }
        if per_model_authority_bindings is None
        else _json_copy(dict(per_model_authority_bindings))
    )
    receipt = build_v2114_post_gate_authority(
        repo_root=root,
        contract=selected,
        expected_git_commit=expected_git_commit,
        parent_import_receipt=parent_import_receipt,
        per_model_authority_bindings=bindings,
        ledger_event_chain_head=ledger_event_chain_head,
    )
    verify_v2114_gate_receipt(
        receipt,
        expected_git_commit=expected_git_commit,
        expected_contract_sha256=selected.canonical_hash,
    )
    path = raw / "long-context-preflight" / "post_gate_authority.json"
    _atomic_json(path, receipt, repo_root=root)
    return path, receipt


def _source_reservations_from_gate(
    receipt: Mapping[str, Any],
    *,
    receipt_file_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Build current-release source rows before receipt fields are attached."""

    sources = _mapping(receipt["authority_sources"], "global authority sources")
    raw_rows = _mapping(receipt["reservations"], "global reservations")
    result: dict[str, dict[str, Any]] = {}
    for model_id, runtime in _RUNTIME_MODELS.items():
        source = _mapping(sources[model_id], f"{model_id} authority source")
        preflight = _mapping(source["source_preflight"], "source preflight")
        result[runtime] = {}
        for call_kind in ("action", "semantic"):
            candidate = {
                "authority": {
                    "authority_id": OBSERVED_P95_AUTHORITY_ID,
                    "source_kind": OBSERVED_P95_SOURCE_KIND,
                    "pilot_contract_hash": receipt["contract_sha256"],
                    "pilot_tag": V2114_SCIENCE_TAG,
                    "source_projection_schema_version": (
                        OBSERVED_P95_PROJECTION_SCHEMA_VERSION
                    ),
                    "source_projection_file_sha256": receipt_file_sha256,
                    "source_projection_content_sha256": receipt["receipt_sha256"],
                    "source_preflight_run_id": preflight["run_id"],
                    "source_preflight_run_spec_sha256": preflight["run_spec_sha256"],
                    "source_model_id": preflight["model_id"],
                    "source_served_model": preflight["served_model"],
                    "source_execution_artifact_sha256": preflight[
                        "execution_artifact_sha256"
                    ],
                    "source_provider_call_journal_sha256": preflight[
                        "provider_call_journal_sha256"
                    ],
                },
                "reservation": _json_copy(raw_rows[runtime][call_kind]),
            }
            try:
                parsed = PreflightP95Reservation.from_dict(
                    model=runtime,
                    call_kind=call_kind,
                    value=candidate["reservation"],
                )
            except (TypeError, ValueError) as exc:  # pragma: no cover - verifier owns
                raise PilotV2114GateError(
                    "V2.11.4 current-release source authority is invalid"
                ) from exc
            candidate["reservation"] = parsed.to_dict()
            result[runtime][call_kind] = candidate
    return result


def runner_reservations_from_v2114_gate_binding(
    binding: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Attach the verified current receipt fields required by the runner."""

    value = _mapping(binding, "V2.11.4 gate authority binding")
    if set(value) != {
        "receipt_path",
        "receipt_file_sha256",
        "receipt_content_sha256",
        "git_commit",
        "reservations",
    }:
        raise PilotV2114GateError("V2.11.4 gate binding fields drifted")
    receipt_path = PurePosixPath(str(value["receipt_path"]))
    if receipt_path != V2114_POST_GATE_RELATIVE_PATH:
        raise PilotV2114GateError("V2.11.4 gate binding path drifted")
    file_sha256 = _digest(value["receipt_file_sha256"], "gate receipt file hash")
    content_sha256 = _digest(
        value["receipt_content_sha256"], "gate receipt content hash"
    )
    git_commit = _commit(value["git_commit"], "gate release commit")
    source = _mapping(value["reservations"], "gate source reservations")
    if set(source) != set(_RUNTIME_MODELS.values()):
        raise PilotV2114GateError("V2.11.4 gate runtime denominator drifted")
    result: dict[str, dict[str, Any]] = {}
    for runtime, raw_by_kind in source.items():
        by_kind = _mapping(raw_by_kind, f"{runtime} gate reservations")
        if set(by_kind) != {"action", "semantic"}:
            raise PilotV2114GateError("V2.11.4 gate call-kind denominator drifted")
        result[runtime] = {}
        for call_kind in ("action", "semantic"):
            entry = _mapping(by_kind[call_kind], "gate source reservation")
            if set(entry) != {"authority", "reservation"}:
                raise PilotV2114GateError("V2.11.4 gate source row drifted")
            authority = dict(_mapping(entry["authority"], "gate source authority"))
            authority.update(
                {
                    "source_authority_receipt_path": receipt_path.as_posix(),
                    "source_authority_receipt_file_sha256": file_sha256,
                    "source_authority_receipt_content_sha256": content_sha256,
                    "source_release_commit": git_commit,
                }
            )
            try:
                result[runtime][call_kind] = ObservedPreflightP95Reservation.from_dict(
                    model=runtime,
                    call_kind=call_kind,
                    value={
                        "authority": authority,
                        "reservation": entry["reservation"],
                    },
                ).to_dict()
            except (TypeError, ValueError) as exc:
                raise PilotV2114GateError(
                    "V2.11.4 runner reservation is invalid"
                ) from exc
    return result


def verified_v2114_gate_authority_binding(
    receipt_path: str | Path,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
    expected_contract_sha256: str | None = None,
    contract: PilotContract | None = None,
) -> dict[str, Any]:
    root = _strict_root(repo_root, name="V2.11.4 repository")
    _verify_release_identity(root, expected_git_commit)
    selected = _contract(root, contract, expected_git_commit)
    if (
        expected_contract_sha256 is not None
        and selected.canonical_hash != expected_contract_sha256
    ):
        raise PilotV2114GateError("V2.11.4 selected contract hash drifted")
    relative = _relative_to_repo(root, receipt_path, name="V2.11.4 global gate")
    if relative != V2114_POST_GATE_RELATIVE_PATH:
        raise PilotV2114GateError("V2.11.4 global gate path drifted")
    raw = _read_regular(root, relative, name="V2.11.4 global gate")
    receipt = verify_v2114_gate_receipt(
        _strict_json(raw, name="V2.11.4 global gate"),
        expected_git_commit=expected_git_commit,
        expected_contract_sha256=selected.canonical_hash,
    )
    raw_root = root.joinpath(*V2114_RAW_ROOT.parts)
    bindings = {
        model_id: verified_v2114_observed_p95_authority_binding(
            v2114_observed_p95_receipt_path(raw_root, model_id),
            repo_root=root,
            raw_root=raw_root,
            expected_git_commit=expected_git_commit,
            contract=selected,
        )
        for model_id in V2114_ALLOWED_MODELS
    }
    rebuilt = build_v2114_post_gate_authority(
        repo_root=root,
        contract=selected,
        expected_git_commit=expected_git_commit,
        parent_import_receipt=_load_parent_receipt(
            root,
            contract=selected,
            expected_git_commit=expected_git_commit,
        )[0],
        per_model_authority_bindings=bindings,
        ledger_event_chain_head=receipt["bindings"]["ledger_event_chain_head"],
    )
    if rebuilt != receipt:
        raise PilotV2114GateError("V2.11.4 global gate differs from current sources")
    reservations = _source_reservations_from_gate(
        receipt,
        receipt_file_sha256=_sha256(raw),
    )
    return {
        "receipt_path": relative.as_posix(),
        "receipt_file_sha256": _sha256(raw),
        "receipt_content_sha256": receipt["receipt_sha256"],
        "git_commit": expected_git_commit,
        "reservations": reservations,
    }


def runner_reservations_from_v2114_gate(
    receipt_path: str | Path,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
    expected_contract_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    binding = verified_v2114_gate_authority_binding(
        receipt_path,
        repo_root=repo_root,
        expected_git_commit=expected_git_commit,
        expected_contract_sha256=expected_contract_sha256,
    )
    return runner_reservations_from_v2114_gate_binding(binding)


__all__ = [
    "PilotV2114GateError",
    "V2114_GATE_SCHEMA_VERSION",
    "V2114_POST_GATE_RELATIVE_PATH",
    "V2114_PREFLIGHT_IMPORT_GATE_SCHEMA_VERSION",
    "V2114_RESEALED_P95_AUTHORITY_SCHEMA_VERSION",
    "V2114_RESEALED_P95_SOURCE_KIND",
    "build_v2114_post_gate_authority",
    "build_v2114_resealed_observed_p95_authority",
    "persist_v2114_post_gate_authority",
    "persist_v2114_resealed_observed_p95_authority",
    "runner_reservations_from_v2114_gate",
    "runner_reservations_from_v2114_gate_binding",
    "verified_v2114_gate_authority_binding",
    "verified_v2114_observed_p95_authority_binding",
    "verified_v2114_observed_p95_projection_binding",
    "verify_v2114_gate_receipt",
    "verify_v2114_resealed_observed_p95_authority",
    "verify_v2114_resealed_observed_p95_projection",
    "v2114_observed_p95_projection_path",
    "v2114_observed_p95_receipt_path",
]
