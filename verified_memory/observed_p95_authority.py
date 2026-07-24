"""Source-backed authority receipts for normal scientific p95 reservations.

The runner's :class:`ObservedPreflightP95Reservation` proves that a numeric
reservation is labelled as a closed-loop observation.  This module provides
the lower-level, filesystem-backed proof behind that label.  It rebuilds a
V2.3 projection from the frozen contract and every source artifact used by the
closed-loop preflight:

* the capability result (and evaluator control for an imported result);
* the capability-derived bootstrap projection and amendment control;
* the closed-loop checkpoint and independently recomputed exactness receipt;
* the terminal provider-call journal; and
* the final observed-p95 projection itself.

All paths written into a receipt are repository-relative.  Contract paths are
restricted to ``experiments/`` and every runtime artifact path is restricted
to ``experiment_results/``.  Reads reject missing files, symlinks in any path
component, duplicate JSON keys, non-finite JSON numbers, and lexical path
escape.  The module is deliberately provider-free and never reads environment
variables or API credentials.

Runner and checkpoint imports are local to verification functions.  This
keeps the module safe to import from a lower layer without introducing a
runner/checkpoint import cycle.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence

from .pilot_budget import preflight_p95
from .pilot_contract import (
    PILOT_CONTRACT_ID_V2_3,
    PILOT_CONTRACT_V2_3_CANONICAL_SHA256,
    PilotContract,
    PilotRunSpec,
    canonical_sha256,
)


OBSERVED_P95_AUTHORITY_RECEIPT_SCHEMA_VERSION = (
    "finevo-observed-p95-authority-receipt-v1"
)
OBSERVED_P95_AUTHORITY_RECEIPT_FILENAME = (
    "observed_p95_authority_receipt.json"
)
OBSERVED_P95_PROJECTION_SCHEMA_VERSION = "finevo-pilot-projection-p95-v1"
OBSERVED_P95_AUTHORITY_ID = "finevo-closed-loop-observed-p95-v1"
OBSERVED_P95_SOURCE_KIND = "sealed-closed-loop-observed-p95"
PREFLIGHT_CHECKPOINT_EXACTNESS_SCHEMA_VERSION = (
    "finevo-preflight-checkpoint-exactness-v1"
)
CANONICALIZATION = "json-sort-keys-utf8-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_RUNTIME_PREFIX = {
    "openai": "openai",
    "openrouter": "thirdparty",
    "ollama": "ollama",
}
_SOURCE_NAMES = (
    "projection",
    "capability",
    "evaluator_control",
    "bootstrap_projection",
    "preflight_amendment_control",
    "checkpoint",
    "checkpoint_exactness",
    "provider_call_journal",
)
_V23_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_3.yaml")


class ObservedP95AuthorityError(RuntimeError):
    """Raised before a source-backed reservation can be released."""


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ObservedP95AuthorityError(
            "observed-p95 data is not canonical JSON"
        ) from exc


def _strict_json_from_bytes(value: bytes, *, name: str) -> dict[str, Any]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ObservedP95AuthorityError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            result[key] = item
        return result

    def reject_nonfinite(item: str) -> None:
        raise ObservedP95AuthorityError(
            f"{name} contains non-finite JSON number {item}"
        )

    try:
        parsed = json.loads(
            value.decode("utf-8", "strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObservedP95AuthorityError(
            f"{name} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise ObservedP95AuthorityError(f"{name} must contain a JSON object")
    return parsed


def _normalize_repo_root(value: str | Path) -> Path:
    root = Path(value).absolute()
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise ObservedP95AuthorityError(
            f"repository root is unavailable: {root}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ObservedP95AuthorityError(
            "repository root must be a real, non-symlink directory"
        )
    return root


def _normalize_relative(
    value: str | Path,
    *,
    required_top: str,
    name: str,
) -> PurePosixPath:
    text = str(value)
    if (
        not text
        or "\\" in text
        or "\x00" in text
        or Path(text).is_absolute()
    ):
        raise ObservedP95AuthorityError(
            f"{name} must be a normalized repository-relative POSIX path"
        )
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.parts[0] != required_top
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != text
    ):
        raise ObservedP95AuthorityError(
            f"{name} must stay below {required_top}/ without path escape"
        )
    return path


def _checked_path(
    repo_root: Path,
    relative: PurePosixPath,
    *,
    name: str,
    require_file: bool,
) -> Path:
    current = repo_root
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ObservedP95AuthorityError(
                f"required {name} path is missing: {relative.as_posix()}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ObservedP95AuthorityError(
                f"{name} path contains a symlink: {relative.as_posix()}"
            )
        final = index == len(relative.parts) - 1
        if final and require_file:
            if not stat.S_ISREG(metadata.st_mode):
                raise ObservedP95AuthorityError(
                    f"{name} must be a regular file: {relative.as_posix()}"
                )
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ObservedP95AuthorityError(
                f"{name} parent must be a directory: {relative.as_posix()}"
            )
    return current


def _read_regular_bytes(
    repo_root: Path,
    relative: PurePosixPath,
    *,
    name: str,
) -> bytes:
    """Read once through a no-follow dir-fd walk of every path component."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ObservedP95AuthorityError(
            "source-backed authority requires no-follow directory reads"
        )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(repo_root, directory_flags)
    except OSError as exc:
        raise ObservedP95AuthorityError(
            "repository root cannot be opened for guarded source reads"
        ) from exc
    try:
        for index, part in enumerate(relative.parts):
            final = index == len(relative.parts) - 1
            flags = os.O_RDONLY | os.O_NOFOLLOW
            if not final:
                flags |= os.O_DIRECTORY
            try:
                next_descriptor = os.open(
                    part,
                    flags,
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise ObservedP95AuthorityError(
                    f"{name} path cannot be opened safely: "
                    f"{relative.as_posix()}"
                ) from exc
            os.close(descriptor)
            descriptor = next_descriptor
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ObservedP95AuthorityError(
                f"{name} must be a regular file: {relative.as_posix()}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
        ):
            raise ObservedP95AuthorityError(
                f"{name} changed during its guarded read"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_json_source(
    repo_root: Path,
    relative: PurePosixPath,
    *,
    name: str,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_bytes(
        repo_root,
        relative,
        name=name,
    )
    return _strict_json_from_bytes(raw, name=name), raw


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bound_content_sha256(value: Mapping[str, Any]) -> str:
    copied = _json_copy(value)
    integrity = copied.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("content_sha256", None)
    return canonical_sha256(copied)


def _verify_bound_payload(
    value: Mapping[str, Any],
    *,
    schema_version: str,
    contract_hash: str,
    git_tag: str,
    git_commit: str,
    name: str,
) -> None:
    if value.get("schema_version") != schema_version:
        raise ObservedP95AuthorityError(
            f"{name} has an unsupported schema version"
        )
    bindings = value.get("bindings")
    integrity = value.get("integrity")
    if not isinstance(bindings, Mapping) or not isinstance(
        integrity, Mapping
    ):
        raise ObservedP95AuthorityError(
            f"{name} bindings or integrity are malformed"
        )
    if (
        bindings.get("contract_sha256") != contract_hash
        or bindings.get("git_tag") != git_tag
        or bindings.get("git_commit") != git_commit
    ):
        raise ObservedP95AuthorityError(
            f"{name} contract/tag/commit binding mismatch"
        )
    if (
        set(integrity) != {"canonicalization", "content_sha256"}
        or integrity.get("canonicalization") != CANONICALIZATION
        or integrity.get("content_sha256") != _bound_content_sha256(value)
    ):
        raise ObservedP95AuthorityError(f"{name} self-hash mismatch")


def _source_entry(
    relative: PurePosixPath,
    raw: bytes,
    *,
    content_sha256: str | None = None,
    journal_sha256: str | None = None,
    checkpoint_hash: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": relative.as_posix(),
        "file_sha256": _sha256_bytes(raw),
    }
    if content_sha256 is not None:
        result["content_sha256"] = content_sha256
    if journal_sha256 is not None:
        result["journal_sha256"] = journal_sha256
    if checkpoint_hash is not None:
        result["checkpoint_hash"] = checkpoint_hash
    return result


def _assert_digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ObservedP95AuthorityError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _stage_execution_modes(
    contract: PilotContract,
    stage_id: str,
) -> frozenset[str]:
    return frozenset(
        cell.execution_mode for cell in contract.stage(stage_id).cells
    )


def _preflight_stage_for_model(
    contract: PilotContract,
    model_id: str,
) -> str:
    candidates = [
        stage_id
        for stage_id in contract.stage_ids
        if model_id in contract.models_for_stage(stage_id)
        and "closed_loop_preflight"
        in _stage_execution_modes(contract, stage_id)
    ]
    if len(candidates) != 1:
        raise ObservedP95AuthorityError(
            f"{model_id} lacks one exact closed-loop preflight stage"
        )
    return candidates[0]


def _capability_source_stage(
    contract: PilotContract,
    preflight_stage: str,
    model_id: str,
) -> str:
    candidates = [
        prerequisite
        for prerequisite in contract.stage(preflight_stage).prerequisites
        if "capability_probe"
        in _stage_execution_modes(contract, prerequisite)
        and model_id in contract.models_for_stage(prerequisite)
    ]
    if len(candidates) != 1:
        raise ObservedP95AuthorityError(
            f"{preflight_stage}/{model_id} lacks one exact capability source"
        )
    return candidates[0]


def _single_registered_spec(
    contract: PilotContract,
    *,
    stage_id: str,
    model_id: str,
    execution_mode: str,
) -> PilotRunSpec:
    specs = contract.expand(stage=stage_id, model=model_id)
    if (
        len(specs) != 1
        or specs[0].execution_mode != execution_mode
        or sum(candidate == specs[0] for candidate in contract.expand()) != 1
    ):
        raise ObservedP95AuthorityError(
            f"{stage_id}/{model_id} is not one exact registered "
            f"{execution_mode} cell"
        )
    return specs[0]


def _expected_source_paths(
    *,
    raw_root: PurePosixPath,
    preflight_spec: PilotRunSpec,
    capability_spec: PilotRunSpec,
) -> dict[str, PurePosixPath]:
    run_dir = (
        raw_root
        / preflight_spec.stage_id
        / "runs"
        / preflight_spec.run_id
    )
    capability_dir = (
        raw_root
        / capability_spec.stage_id
        / "runs"
        / capability_spec.run_id
    )
    return {
        "projection": run_dir / "projection_p95.json",
        "capability": capability_dir / "capability.json",
        "evaluator_control": raw_root / "evaluator_amendment_receipt.json",
        "bootstrap_projection": (
            capability_dir / "bootstrap_projection_p95.json"
        ),
        "preflight_amendment_control": (
            raw_root / "preflight_bootstrap_amendment_control.json"
        ),
        "checkpoint": run_dir / "preflight_checkpoint.json",
        "checkpoint_exactness": (
            run_dir / "preflight_checkpoint_exactness.json"
        ),
        "provider_call_journal": (
            raw_root
            / preflight_spec.stage_id
            / "provider_call_journals"
            / f"{preflight_spec.run_id}--preflight.json"
        ),
    }


def _path_binding_matches(
    value: Any,
    *,
    repo_root: Path,
    expected: PurePosixPath,
) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    if path.is_absolute():
        return path == repo_root.joinpath(*expected.parts)
    return value == expected.as_posix()


def _load_frozen_v23_contract(
    repo_root: Path,
    contract_relative: PurePosixPath,
) -> tuple[PilotContract, bytes]:
    document, raw = _read_json_source(
        repo_root,
        contract_relative,
        name="pilot contract",
    )
    try:
        contract = PilotContract.from_dict(document)
    except Exception as exc:
        raise ObservedP95AuthorityError(
            "pilot contract failed typed validation"
        ) from exc
    if (
        contract_relative != _V23_CONTRACT_PATH
        or contract.canonical_hash
        != PILOT_CONTRACT_V2_3_CANONICAL_SHA256
        or contract.contract_id != PILOT_CONTRACT_ID_V2_3
        or contract.status != "frozen"
        or contract.preflight_bootstrap_amendment is None
    ):
        raise ObservedP95AuthorityError(
            "observed-p95 authority requires the frozen full V2.3 contract"
        )
    return contract, raw


def _validate_capability(
    capability: Mapping[str, Any],
    *,
    contract: PilotContract,
    capability_spec: PilotRunSpec,
    evaluator_control: Mapping[str, Any],
) -> None:
    # Lazy imports avoid pulling runner through pilot_evidence at module import.
    from .pilot_evaluation_amendment import (
        CAPABILITY_IMPORT_SCHEMA_VERSION,
        PilotEvaluationAmendmentError,
        model_import_records,
        validate_capability_import,
    )
    from .pilot_evidence import (
        CAPABILITY_V4_SCHEMA_VERSION,
        PilotEvidenceError,
        _validate_capability_v4,
    )

    try:
        # Validate the persisted evaluator control even when this particular
        # model has a fresh capability result.  It is a global V2.3 source
        # artifact and must not become an unchecked, rehashable receipt input.
        model_import_records(contract, evaluator_control)
        if capability.get("schema_version") == CAPABILITY_IMPORT_SCHEMA_VERSION:
            validate_capability_import(
                capability,
                contract,
                capability_spec,
                evaluator_control,
            )
        elif capability.get("schema_version") == CAPABILITY_V4_SCHEMA_VERSION:
            if (
                capability.get("contract_sha256") != contract.canonical_hash
                or capability.get("run_spec") != capability_spec.to_dict()
            ):
                raise ObservedP95AuthorityError(
                    "fresh capability contract/spec binding mismatch"
                )
            _validate_capability_v4(capability)
        else:
            raise ObservedP95AuthorityError(
                "unsupported V2.3 capability source schema"
            )
    except (PilotEvaluationAmendmentError, PilotEvidenceError) as exc:
        raise ObservedP95AuthorityError(
            f"capability source failed validation: {exc}"
        ) from exc
    if (
        capability.get("pass") is not True
        or not isinstance(capability.get("interface_gate"), Mapping)
        or capability["interface_gate"].get("pass") is not True
    ):
        raise ObservedP95AuthorityError(
            "capability source is not a passing interface gate"
        )


def _usage_projection_rows(
    capability: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    contract: PilotContract,
    model_id: str,
) -> list[dict[str, Any]]:
    amendment = contract.preflight_bootstrap_amendment
    if not isinstance(amendment, Mapping) or not isinstance(
        amendment.get("bootstrap_policy"), Mapping
    ):
        raise ObservedP95AuthorityError(
            "V2.3 capability call-kind policy is malformed"
        )
    kind_map = amendment["bootstrap_policy"].get(
        "source_output_contract_map"
    )
    if not isinstance(kind_map, Mapping):
        raise ObservedP95AuthorityError(
            "V2.3 capability output-contract map is malformed"
        )
    served_model = contract.provider_profiles[model_id].served_model

    def normalized_capability_row(
        row: Mapping[str, Any],
        *,
        index: int,
    ) -> dict[str, Any]:
        output_contract = row.get(
            "output_contract_id",
            row.get("call_kind"),
        )
        call_kind = kind_map.get(output_contract)
        if call_kind not in {"action", "semantic"}:
            raise ObservedP95AuthorityError(
                f"capability usage row {index} has an unregistered "
                "output contract"
            )
        observed_model = row.get(
            "served_model",
            row.get("response_model"),
        )
        if observed_model != served_model:
            raise ObservedP95AuthorityError(
                f"capability usage row {index} served-model mismatch"
            )
        return {
            "response_model": served_model,
            "call_kind": call_kind,
            "usage": row.get("usage"),
        }

    rows: list[dict[str, Any]] = []
    capability_rows = capability.get("rows")
    if isinstance(capability_rows, Sequence) and not isinstance(
        capability_rows, (str, bytes)
    ):
        for index, row in enumerate(capability_rows):
            if not isinstance(row, Mapping):
                raise ObservedP95AuthorityError(
                    f"capability row {index} is malformed"
                )
            rows.append(
                normalized_capability_row(
                    row,
                    index=index,
                )
            )
    else:
        imported = capability.get("usage_projection_rows")
        if isinstance(imported, (str, bytes)) or not isinstance(
            imported, Sequence
        ):
            raise ObservedP95AuthorityError(
                "capability source lacks replayable usage rows"
            )
        for index, row in enumerate(imported):
            if not isinstance(row, Mapping):
                raise ObservedP95AuthorityError(
                    f"capability usage row {index} is malformed"
                )
            rows.append(
                normalized_capability_row(
                    row,
                    index=index,
                )
            )
    provider_calls = checkpoint.get("provider_calls")
    if isinstance(provider_calls, (str, bytes)) or not isinstance(
        provider_calls, Sequence
    ):
        raise ObservedP95AuthorityError(
            "preflight checkpoint lacks provider usage rows"
        )
    for index, row in enumerate(provider_calls):
        if not isinstance(row, Mapping):
            raise ObservedP95AuthorityError(
                f"preflight provider row {index} is malformed"
            )
        rows.append(
            {
                "response_model": row.get("response_model"),
                "call_kind": row.get("call_kind"),
                "usage": row.get("usage"),
            }
        )
    return rows


def _runner_config_binding_sha256(
    serialized_config: Mapping[str, Any],
) -> str:
    payload = _json_copy(serialized_config)
    if not isinstance(payload, dict):
        raise ObservedP95AuthorityError(
            "checkpoint runner config is malformed"
        )
    payload.pop("contract_bootstrap_reservations", None)
    payload["preflight_measurement_role"] = "closed_loop_preflight"
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _runtime_model(contract: PilotContract, model_id: str) -> str:
    profile = contract.provider_profiles[model_id]
    prefix = _RUNTIME_PREFIX.get(profile.transport)
    if prefix is None:
        raise ObservedP95AuthorityError(
            f"{model_id} has no supported observed-p95 runtime transport"
        )
    return f"{prefix}/{profile.requested_model}"


def _build_receipt(
    *,
    repo_root: Path,
    contract_relative: PurePosixPath,
    raw_root_relative: PurePosixPath,
    model_id: str,
    expected_git_commit: str,
) -> dict[str, Any]:
    if _COMMIT_RE.fullmatch(expected_git_commit) is None:
        raise ObservedP95AuthorityError(
            "expected_git_commit must be a lowercase 40-hex commit"
        )
    contract, contract_raw = _load_frozen_v23_contract(
        repo_root,
        contract_relative,
    )
    git_tag = str(contract.implementation["required_git_tag"])
    preflight_stage = _preflight_stage_for_model(contract, model_id)
    capability_stage = _capability_source_stage(
        contract,
        preflight_stage,
        model_id,
    )
    preflight_spec = _single_registered_spec(
        contract,
        stage_id=preflight_stage,
        model_id=model_id,
        execution_mode="closed_loop_preflight",
    )
    capability_spec = _single_registered_spec(
        contract,
        stage_id=capability_stage,
        model_id=model_id,
        execution_mode="capability_probe",
    )
    paths = _expected_source_paths(
        raw_root=raw_root_relative,
        preflight_spec=preflight_spec,
        capability_spec=capability_spec,
    )

    source_values: dict[str, dict[str, Any]] = {}
    source_raw: dict[str, bytes] = {}
    for name in _SOURCE_NAMES:
        value, raw = _read_json_source(
            repo_root,
            paths[name],
            name=name.replace("_", " "),
        )
        source_values[name] = value
        source_raw[name] = raw

    projection = source_values["projection"]
    if set(projection) != {
        "schema_version",
        "model_id",
        "served_model",
        "bindings",
        "projection",
        "integrity",
    }:
        raise ObservedP95AuthorityError(
            "preflight projection top-level shape drifted"
        )
    _verify_bound_payload(
        projection,
        schema_version=OBSERVED_P95_PROJECTION_SCHEMA_VERSION,
        contract_hash=contract.canonical_hash,
        git_tag=git_tag,
        git_commit=expected_git_commit,
        name="preflight projection",
    )
    profile = contract.provider_profiles.get(model_id)
    if profile is None or (
        projection.get("model_id") != model_id
        or projection.get("served_model") != profile.served_model
    ):
        raise ObservedP95AuthorityError(
            "preflight projection model identity mismatch"
        )
    bindings = projection["bindings"]
    required_binding_keys = {
        "contract_sha256",
        "git_tag",
        "git_commit",
        "source_capability",
        "source_capability_sha256",
        "source_provider_call_journal",
        "source_provider_call_journal_file_sha256",
        "source_provider_call_journal_sha256",
        "source_bootstrap_projection",
        "source_bootstrap_projection_file_sha256",
        "source_bootstrap_projection_content_sha256",
        "source_preflight_amendment_control",
        "source_preflight_amendment_control_file_sha256",
        "source_preflight_amendment_control_content_sha256",
        "source_checkpoint",
        "source_checkpoint_file_sha256",
        "source_checkpoint_hash",
        "source_checkpoint_exactness",
        "source_checkpoint_exactness_file_sha256",
        "source_checkpoint_exactness_content_sha256",
    }
    if set(bindings) != required_binding_keys:
        raise ObservedP95AuthorityError(
            "preflight projection source bindings drifted"
        )
    for binding_name, source_name in (
        ("source_capability", "capability"),
        ("source_provider_call_journal", "provider_call_journal"),
        ("source_bootstrap_projection", "bootstrap_projection"),
        (
            "source_preflight_amendment_control",
            "preflight_amendment_control",
        ),
        ("source_checkpoint", "checkpoint"),
        ("source_checkpoint_exactness", "checkpoint_exactness"),
    ):
        if not _path_binding_matches(
            bindings.get(binding_name),
            repo_root=repo_root,
            expected=paths[source_name],
        ):
            raise ObservedP95AuthorityError(
                f"projection {binding_name} path mismatch"
            )

    capability = source_values["capability"]
    evaluator_control = source_values["evaluator_control"]
    _validate_capability(
        capability,
        contract=contract,
        capability_spec=capability_spec,
        evaluator_control=evaluator_control,
    )
    capability_file_sha256 = _sha256_bytes(source_raw["capability"])
    if bindings.get("source_capability_sha256") != capability_file_sha256:
        raise ObservedP95AuthorityError(
            "projection capability file hash mismatch"
        )

    # Bootstrap validators import pilot_evidence/runner, so keep them local.
    from .pilot_preflight_amendment import (
        PilotPreflightAmendmentError,
        runner_reservations_from_bootstrap_projection,
        validate_capability_bootstrap_projection,
        validate_preflight_amendment_control,
    )

    bootstrap = source_values["bootstrap_projection"]
    bootstrap_bindings = bootstrap.get("bindings")
    if not isinstance(bootstrap_bindings, Mapping):
        raise ObservedP95AuthorityError(
            "bootstrap projection bindings are malformed"
        )
    authorized_config_sha256 = _assert_digest(
        bootstrap_bindings.get("authorized_config_sha256"),
        name="bootstrap authorized config hash",
    )
    capability_path = repo_root.joinpath(*paths["capability"].parts)
    try:
        validate_capability_bootstrap_projection(
            bootstrap,
            contract,
            capability_spec,
            preflight_spec,
            capability,
            source_capability_path=capability_path,
            source_capability_file_sha256=capability_file_sha256,
            git_tag=git_tag,
            git_commit=expected_git_commit,
            authorized_config_sha256=authorized_config_sha256,
        )
        validate_preflight_amendment_control(
            source_values["preflight_amendment_control"],
            contract,
        )
    except PilotPreflightAmendmentError as exc:
        raise ObservedP95AuthorityError(
            f"bootstrap/control validation failed: {exc}"
        ) from exc
    bootstrap_integrity = bootstrap.get("integrity")
    control_integrity = source_values["preflight_amendment_control"].get(
        "integrity"
    )
    if not isinstance(bootstrap_integrity, Mapping) or not isinstance(
        control_integrity, Mapping
    ):
        raise ObservedP95AuthorityError(
            "bootstrap/control integrity is malformed"
        )
    if (
        bindings.get("source_bootstrap_projection_file_sha256")
        != _sha256_bytes(source_raw["bootstrap_projection"])
        or bindings.get("source_bootstrap_projection_content_sha256")
        != bootstrap_integrity.get("content_sha256")
        or bindings.get(
            "source_preflight_amendment_control_file_sha256"
        )
        != _sha256_bytes(source_raw["preflight_amendment_control"])
        or bindings.get(
            "source_preflight_amendment_control_content_sha256"
        )
        != control_integrity.get("content_sha256")
    ):
        raise ObservedP95AuthorityError(
            "projection bootstrap/control hash binding mismatch"
        )

    # Checkpoint construction and exact replay are intentionally lazy imports.
    from .pilot_checkpoint import (
        PilotCheckpoint,
        PilotCheckpointError,
        verify_closed_loop_preflight_checkpoint,
    )

    try:
        checkpoint = PilotCheckpoint(source_values["checkpoint"])
        recomputed_exactness = verify_closed_loop_preflight_checkpoint(
            checkpoint
        )
    except PilotCheckpointError as exc:
        raise ObservedP95AuthorityError(
            f"closed-loop checkpoint failed exact restore: {exc}"
        ) from exc
    checkpoint_config = checkpoint.payload.get("run_config")
    if not isinstance(checkpoint_config, Mapping):
        raise ObservedP95AuthorityError(
            "preflight checkpoint lacks a runner config"
        )
    expected_bootstrap_reservations = (
        runner_reservations_from_bootstrap_projection(
            bootstrap,
            contract=contract,
            capability_spec=capability_spec,
            target_preflight_spec=preflight_spec,
            capability=capability,
            source_capability_path=capability_path,
            source_capability_file_sha256=capability_file_sha256,
            git_tag=git_tag,
            git_commit=expected_git_commit,
            authorized_config_sha256=authorized_config_sha256,
        )
    )
    if (
        checkpoint_config.get("contract_bootstrap_reservations")
        != expected_bootstrap_reservations
        or checkpoint_config.get("preflight_measurement_role")
        != "closed_loop_preflight"
        or checkpoint_config.get("preflight_p95_reservations") != {}
        or checkpoint_config.get("pilot_contract_hash")
        != contract.canonical_hash
        or checkpoint_config.get("pilot_tag") != git_tag
        or checkpoint_config.get("run_id")
        != f"{preflight_spec.run_id}--actor-preflight"
        or checkpoint_config.get("seed")
        != preflight_spec.environment_seed
        or _runner_config_binding_sha256(checkpoint_config)
        != authorized_config_sha256
    ):
        raise ObservedP95AuthorityError(
            "checkpoint runner/bootstrap authority binding mismatch"
        )

    exactness = source_values["checkpoint_exactness"]
    if set(exactness) != {
        "schema_version",
        "bindings",
        "exactness",
        "integrity",
    }:
        raise ObservedP95AuthorityError(
            "checkpoint exactness top-level shape drifted"
        )
    _verify_bound_payload(
        exactness,
        schema_version=PREFLIGHT_CHECKPOINT_EXACTNESS_SCHEMA_VERSION,
        contract_hash=contract.canonical_hash,
        git_tag=git_tag,
        git_commit=expected_git_commit,
        name="checkpoint exactness receipt",
    )
    exactness_bindings = exactness.get("bindings")
    if not isinstance(exactness_bindings, Mapping) or set(
        exactness_bindings
    ) != {
        "contract_sha256",
        "git_tag",
        "git_commit",
        "checkpoint_path",
        "checkpoint_file_sha256",
        "checkpoint_hash",
    }:
        raise ObservedP95AuthorityError(
            "checkpoint exactness bindings drifted"
        )
    checkpoint_file_sha256 = _sha256_bytes(source_raw["checkpoint"])
    exactness_content_sha256 = exactness["integrity"]["content_sha256"]
    if (
        exactness.get("exactness") != recomputed_exactness
        or not _path_binding_matches(
            exactness_bindings.get("checkpoint_path"),
            repo_root=repo_root,
            expected=paths["checkpoint"],
        )
        or exactness_bindings.get("checkpoint_file_sha256")
        != checkpoint_file_sha256
        or exactness_bindings.get("checkpoint_hash")
        != checkpoint.checkpoint_hash
        or bindings.get("source_checkpoint_file_sha256")
        != checkpoint_file_sha256
        or bindings.get("source_checkpoint_hash")
        != checkpoint.checkpoint_hash
        or bindings.get("source_checkpoint_exactness_file_sha256")
        != _sha256_bytes(source_raw["checkpoint_exactness"])
        or bindings.get("source_checkpoint_exactness_content_sha256")
        != exactness_content_sha256
    ):
        raise ObservedP95AuthorityError(
            "checkpoint/exactness source binding mismatch"
        )

    # Full journal verification is also kept lazy to avoid a top-level runner
    # dependency.  It verifies both the event hash chain and every terminal
    # completion -> parse disposition.
    from .runner import VerifiedRunError, verify_provider_call_journal

    try:
        journal = verify_provider_call_journal(
            source_values["provider_call_journal"],
            expected_run_id=(
                f"{preflight_spec.run_id}--actor-preflight"
            ),
            expected_contract_hash=contract.canonical_hash,
            require_terminal_dispositions=True,
        )
    except VerifiedRunError as exc:
        raise ObservedP95AuthorityError(
            f"provider call journal failed validation: {exc}"
        ) from exc
    journal_file_sha256 = _sha256_bytes(
        source_raw["provider_call_journal"]
    )
    journal_sha256 = _assert_digest(
        journal.get("journal_sha256"),
        name="provider journal hash",
    )
    journal_completion_events = [
        event
        for event in journal["events"]
        if event["event_type"] == "completion_received"
    ]
    journal_disposition_events = [
        event
        for event in journal["events"]
        if event["event_type"] == "parse_disposition"
    ]
    checkpoint_provider_rows = checkpoint.payload.get("provider_calls")
    if (
        not isinstance(checkpoint_provider_rows, Sequence)
        or isinstance(checkpoint_provider_rows, (str, bytes))
        or len(journal_completion_events) != len(checkpoint_provider_rows)
        or len(journal_disposition_events) != len(
            checkpoint_provider_rows
        )
    ):
        raise ObservedP95AuthorityError(
            "checkpoint provider denominator differs from the terminal journal"
        )
    for row, event in zip(
        checkpoint_provider_rows,
        journal_completion_events,
        strict=True,
    ):
        if not isinstance(row, Mapping) or any(
            row.get(key) != value
            for key, value in event["payload"].items()
        ):
            raise ObservedP95AuthorityError(
                "checkpoint provider usage differs from the terminal journal"
            )
    for row, event in zip(
        checkpoint_provider_rows,
        journal_disposition_events,
        strict=True,
    ):
        if not isinstance(row, Mapping) or not isinstance(
            row.get("parse_disposition"), Mapping
        ):
            raise ObservedP95AuthorityError(
                "checkpoint parse disposition is malformed"
            )
        disposition = row["parse_disposition"]
        if any(
            event["payload"].get(key) != disposition.get(key)
            for key in ("parse_status", "parse_mode", "accepted")
        ):
            raise ObservedP95AuthorityError(
                "checkpoint parse disposition differs from the terminal journal"
            )
    checkpoint_journal_binding = checkpoint.payload.get(
        "provider_call_journal_binding"
    )
    if (
        not isinstance(checkpoint_journal_binding, Mapping)
        or checkpoint_journal_binding.get("enabled") is not True
        or checkpoint_journal_binding.get("journal_sha256")
        != journal_sha256
        or checkpoint_journal_binding.get("event_count")
        != len(journal["events"])
        or checkpoint_journal_binding.get("completion_event_count")
        != len(journal_completion_events)
        or checkpoint_journal_binding.get(
            "parse_disposition_event_count"
        )
        != len(journal_disposition_events)
        or checkpoint_journal_binding.get("run_id") != journal["run_id"]
        or checkpoint_journal_binding.get("contract_hash")
        != journal["contract_hash"]
        or checkpoint_journal_binding.get("path_name")
        != paths["provider_call_journal"].name
    ):
        raise ObservedP95AuthorityError(
            "checkpoint journal binding differs from the terminal journal"
        )
    if (
        bindings.get("source_provider_call_journal_file_sha256")
        != journal_file_sha256
        or bindings.get("source_provider_call_journal_sha256")
        != journal_sha256
    ):
        raise ObservedP95AuthorityError(
            "projection provider-journal hash binding mismatch"
        )

    reserve_multiplier = float(
        contract.budgets["pre_dispatch_projection"][
            "reserve_multiplier"
        ]
    )
    try:
        recomputed_projection = preflight_p95(
            _usage_projection_rows(
                capability,
                checkpoint.payload,
                contract=contract,
                model_id=model_id,
            ),
            reserve_multiplier=reserve_multiplier,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ObservedP95AuthorityError(
            f"observed p95 could not be recomputed: {exc}"
        ) from exc
    if projection.get("projection") != recomputed_projection:
        raise ObservedP95AuthorityError(
            "sealed projection differs from recomputed source usage p95"
        )
    served_model = str(projection["served_model"])
    if any(
        str(key).rpartition("::")[0] != served_model
        for key in recomputed_projection
    ):
        raise ObservedP95AuthorityError(
            "projection contains a non-frozen served model"
        )
    expected_normal_keys = {
        f"{served_model}::action",
        f"{served_model}::semantic",
    }
    normal_rows = {
        key: recomputed_projection[key]
        for key in expected_normal_keys
        if key in recomputed_projection
    }
    if set(normal_rows) != expected_normal_keys:
        raise ObservedP95AuthorityError(
            "projection lacks exact action/semantic observed rows"
        )
    amendment = contract.preflight_bootstrap_amendment
    if not isinstance(amendment, Mapping) or not isinstance(
        amendment.get("bootstrap_policy"), Mapping
    ):
        raise ObservedP95AuthorityError(
            "V2.3 bootstrap denominator policy is malformed"
        )
    denominator_policy = amendment["bootstrap_policy"]
    capability_counts = denominator_policy.get("required_sample_counts")
    preflight_counts = denominator_policy.get(
        "target_dispatch_call_counts"
    )
    if not isinstance(capability_counts, Mapping) or not isinstance(
        preflight_counts, Mapping
    ):
        raise ObservedP95AuthorityError(
            "V2.3 bootstrap denominator counts are malformed"
        )
    expected_sample_counts = {
        call_kind: int(capability_counts[call_kind])
        + int(preflight_counts[call_kind])
        for call_kind in ("action", "semantic")
    }
    if any(
        normal_rows[f"{served_model}::{call_kind}"].get("sample_count")
        != expected_sample_counts[call_kind]
        for call_kind in ("action", "semantic")
    ):
        raise ObservedP95AuthorityError(
            "closed-loop action/semantic sample denominator drifted"
        )
    runtime_model = _runtime_model(contract, model_id)
    projection_content_sha256 = _assert_digest(
        projection["integrity"].get("content_sha256"),
        name="projection content hash",
    )
    projection_file_sha256 = _sha256_bytes(source_raw["projection"])
    authority = {
        "authority_id": OBSERVED_P95_AUTHORITY_ID,
        "source_kind": OBSERVED_P95_SOURCE_KIND,
        "pilot_contract_hash": contract.canonical_hash,
        "pilot_tag": git_tag,
        "source_projection_schema_version": (
            OBSERVED_P95_PROJECTION_SCHEMA_VERSION
        ),
        "source_projection_file_sha256": projection_file_sha256,
        "source_projection_content_sha256": projection_content_sha256,
        "source_preflight_run_id": preflight_spec.run_id,
        "source_preflight_run_spec_sha256": canonical_sha256(
            preflight_spec.to_dict()
        ),
        "source_model_id": model_id,
        "source_served_model": served_model,
        "source_execution_artifact_sha256": checkpoint_file_sha256,
        "source_provider_call_journal_sha256": journal_sha256,
    }
    reservations = {
        runtime_model: {
            call_kind: {
                "authority": _json_copy(authority),
                "reservation": _json_copy(
                    normal_rows[f"{served_model}::{call_kind}"]
                ),
            }
            for call_kind in ("action", "semantic")
        }
    }
    # Validate the numeric p95 rows independently of the source-authority
    # wrapper.  Keeping the wrapper out of this builder avoids a future
    # receipt-verifier -> runner-wrapper -> receipt-verifier recursion.
    from .runner import PreflightP95Reservation

    for call_kind in ("action", "semantic"):
        try:
            parsed = PreflightP95Reservation.from_dict(
                model=runtime_model,
                call_kind=call_kind,
                value=reservations[runtime_model][call_kind][
                    "reservation"
                ],
            )
        except (TypeError, ValueError) as exc:
            raise ObservedP95AuthorityError(
                f"{call_kind} reservation failed runner validation: {exc}"
            ) from exc
        reservations[runtime_model][call_kind]["reservation"] = (
            parsed.to_dict()
        )

    evaluator_integrity = evaluator_control.get("integrity")
    if not isinstance(evaluator_integrity, Mapping):
        raise ObservedP95AuthorityError(
            "evaluator control integrity is malformed"
        )
    sources = {
        "projection": _source_entry(
            paths["projection"],
            source_raw["projection"],
            content_sha256=projection_content_sha256,
        ),
        "capability": _source_entry(
            paths["capability"],
            source_raw["capability"],
            content_sha256=(
                capability.get("integrity", {}).get("content_sha256")
                if isinstance(capability.get("integrity"), Mapping)
                else None
            ),
        ),
        "evaluator_control": _source_entry(
            paths["evaluator_control"],
            source_raw["evaluator_control"],
            content_sha256=(
                evaluator_integrity.get("content_sha256")
            ),
        ),
        "bootstrap_projection": _source_entry(
            paths["bootstrap_projection"],
            source_raw["bootstrap_projection"],
            content_sha256=str(
                bootstrap_integrity["content_sha256"]
            ),
        ),
        "preflight_amendment_control": _source_entry(
            paths["preflight_amendment_control"],
            source_raw["preflight_amendment_control"],
            content_sha256=str(
                control_integrity["content_sha256"]
            ),
        ),
        "checkpoint": _source_entry(
            paths["checkpoint"],
            source_raw["checkpoint"],
            checkpoint_hash=checkpoint.checkpoint_hash,
        ),
        "checkpoint_exactness": _source_entry(
            paths["checkpoint_exactness"],
            source_raw["checkpoint_exactness"],
            content_sha256=exactness_content_sha256,
        ),
        "provider_call_journal": _source_entry(
            paths["provider_call_journal"],
            source_raw["provider_call_journal"],
            journal_sha256=journal_sha256,
        ),
    }
    receipt: dict[str, Any] = {
        "schema_version": (
            OBSERVED_P95_AUTHORITY_RECEIPT_SCHEMA_VERSION
        ),
        "contract": {
            "path": contract_relative.as_posix(),
            "file_sha256": _sha256_bytes(contract_raw),
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
        },
        "raw_root": raw_root_relative.as_posix(),
        "git": {
            "tag": git_tag,
            "commit": expected_git_commit,
        },
        "model": {
            "model_id": model_id,
            "served_model": served_model,
            "runtime_model": runtime_model,
        },
        "source_preflight": {
            "stage_id": preflight_stage,
            "run_id": preflight_spec.run_id,
            "run_spec_sha256": canonical_sha256(
                preflight_spec.to_dict()
            ),
            "capability_stage_id": capability_stage,
            "capability_run_id": capability_spec.run_id,
            "capability_run_spec_sha256": canonical_sha256(
                capability_spec.to_dict()
            ),
        },
        "sources": sources,
        "reservations": reservations,
        "scientific_evidence": False,
        "evidence_use": (
            "source-backed dispatch authority; not a scientific result"
        ),
    }
    receipt["integrity"] = {
        "canonicalization": CANONICALIZATION,
        "content_sha256": canonical_sha256(receipt),
    }
    return receipt


def build_observed_p95_authority_receipt(
    *,
    repo_root: str | Path,
    contract_path: str | Path,
    raw_root: str | Path,
    model_id: str,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Rebuild and serialize one model's source-backed p95 authority.

    ``contract_path`` must be below ``experiments/`` and ``raw_root`` must be
    below ``experiment_results/``.  Both are repository-relative.  The
    returned mapping is deterministic, strict-JSON serializable, and
    self-hashed; this function does not write it to disk.
    """

    root = _normalize_repo_root(repo_root)
    contract_relative = _normalize_relative(
        contract_path,
        required_top="experiments",
        name="contract_path",
    )
    raw_root_relative = _normalize_relative(
        raw_root,
        required_top="experiment_results",
        name="raw_root",
    )
    _checked_path(
        root,
        raw_root_relative,
        name="raw_root",
        require_file=False,
    )
    if (
        not isinstance(model_id, str)
        or not model_id
        or model_id != model_id.strip()
    ):
        raise ObservedP95AuthorityError(
            "model_id must be a normalized non-empty string"
        )
    return _build_receipt(
        repo_root=root,
        contract_relative=contract_relative,
        raw_root_relative=raw_root_relative,
        model_id=model_id,
        expected_git_commit=expected_git_commit,
    )


def _read_receipt_input(
    value_or_path: Mapping[str, Any] | str | Path,
    *,
    repo_root: Path,
) -> tuple[
    dict[str, Any],
    PurePosixPath | None,
    bytes | None,
]:
    if isinstance(value_or_path, Mapping):
        value = _json_copy(value_or_path)
        if not isinstance(value, dict):
            raise ObservedP95AuthorityError(
                "observed-p95 receipt must be an object"
            )
        return value, None, None
    relative = _normalize_relative(
        value_or_path,
        required_top="experiment_results",
        name="receipt path",
    )
    value, raw = _read_json_source(
        repo_root,
        relative,
        name="observed-p95 authority receipt",
    )
    return value, relative, raw


def _verify_receipt_value(
    receipt: dict[str, Any],
    *,
    repo_root: Path,
    expected_git_commit: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    expected_keys = {
        "schema_version",
        "contract",
        "raw_root",
        "git",
        "model",
        "source_preflight",
        "sources",
        "reservations",
        "scientific_evidence",
        "evidence_use",
        "integrity",
    }
    if set(receipt) != expected_keys or receipt.get(
        "schema_version"
    ) != OBSERVED_P95_AUTHORITY_RECEIPT_SCHEMA_VERSION:
        raise ObservedP95AuthorityError(
            "observed-p95 receipt top-level shape or schema drifted"
        )
    integrity = receipt.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ObservedP95AuthorityError(
            "observed-p95 receipt integrity is malformed"
        )
    unsigned = _json_copy(receipt)
    unsigned.pop("integrity", None)
    if (
        set(integrity) != {"canonicalization", "content_sha256"}
        or integrity.get("canonicalization") != CANONICALIZATION
        or integrity.get("content_sha256") != canonical_sha256(unsigned)
    ):
        raise ObservedP95AuthorityError(
            "observed-p95 receipt self-hash mismatch"
        )
    contract_binding = receipt.get("contract")
    model_binding = receipt.get("model")
    git_binding = receipt.get("git")
    sources = receipt.get("sources")
    if (
        not isinstance(contract_binding, Mapping)
        or set(contract_binding)
        != {
            "path",
            "file_sha256",
            "contract_id",
            "contract_sha256",
        }
        or not isinstance(model_binding, Mapping)
        or set(model_binding)
        != {"model_id", "served_model", "runtime_model"}
        or not isinstance(git_binding, Mapping)
        or set(git_binding) != {"tag", "commit"}
        or not isinstance(sources, Mapping)
        or set(sources) != set(_SOURCE_NAMES)
    ):
        raise ObservedP95AuthorityError(
            "observed-p95 receipt source bindings are malformed"
        )
    if (
        git_binding.get("commit") != expected_git_commit
        or _COMMIT_RE.fullmatch(expected_git_commit) is None
    ):
        raise ObservedP95AuthorityError(
            "observed-p95 receipt commit differs from release authority"
        )
    contract_relative = _normalize_relative(
        str(contract_binding.get("path")),
        required_top="experiments",
        name="receipt contract path",
    )
    raw_root_relative = _normalize_relative(
        str(receipt.get("raw_root")),
        required_top="experiment_results",
        name="receipt raw_root",
    )
    for source_name, source in sources.items():
        if not isinstance(source, Mapping) or "path" not in source:
            raise ObservedP95AuthorityError(
                f"receipt {source_name} source is malformed"
            )
        _normalize_relative(
            str(source["path"]),
            required_top="experiment_results",
            name=f"receipt {source_name} path",
        )
    expected = _build_receipt(
        repo_root=repo_root,
        contract_relative=contract_relative,
        raw_root_relative=raw_root_relative,
        model_id=str(model_binding.get("model_id")),
        expected_git_commit=expected_git_commit,
    )
    if receipt != expected:
        raise ObservedP95AuthorityError(
            "observed-p95 receipt differs from its reverified source chain"
        )
    reservations = expected["reservations"]
    if not isinstance(reservations, dict):  # pragma: no cover - builder owns it
        raise ObservedP95AuthorityError(
            "observed-p95 reservations are malformed"
        )
    return _json_copy(reservations)


def verify_observed_p95_authority_receipt(
    value_or_path: Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Verify a receipt and return runner-compatible action/semantic rows.

    Verification does not trust hashes stored in the receipt in isolation.
    It reloads every repository-relative source, reruns the existing
    capability/bootstrap/checkpoint/journal validators, recomputes p95, and
    requires byte-for-structure equality with a newly built receipt.
    """

    root = _normalize_repo_root(repo_root)
    receipt, _, _ = _read_receipt_input(
        value_or_path,
        repo_root=root,
    )
    return _verify_receipt_value(
        receipt,
        repo_root=root,
        expected_git_commit=expected_git_commit,
    )


def verified_observed_p95_authority_binding(
    receipt_path: str | Path,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Return a guarded receipt binding plus its verified reservations.

    The receipt file is opened exactly once through the no-follow reader.
    ``file_sha256`` is computed from those exact bytes; ``content_sha256`` is
    taken from the self-hash that was checked on the same parsed bytes.  The
    returned path/hash/content/commit tuple can therefore be copied into a
    runner authority wrapper without a second, unguarded read.
    """

    root = _normalize_repo_root(repo_root)
    receipt, relative, raw = _read_receipt_input(
        receipt_path,
        repo_root=root,
    )
    if relative is None or raw is None:  # pragma: no cover - typed path input
        raise ObservedP95AuthorityError(
            "authority binding requires a repository-relative receipt path"
        )
    reservations = _verify_receipt_value(
        receipt,
        repo_root=root,
        expected_git_commit=expected_git_commit,
    )
    integrity = receipt["integrity"]
    return {
        "receipt_path": relative.as_posix(),
        "receipt_file_sha256": _sha256_bytes(raw),
        "receipt_content_sha256": str(integrity["content_sha256"]),
        "git_commit": expected_git_commit,
        "reservations": reservations,
    }


def validate_observed_p95_authority_receipt(
    value_or_path: Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
) -> None:
    """Validate the complete source chain without returning reservations."""

    verify_observed_p95_authority_receipt(
        value_or_path,
        repo_root=repo_root,
        expected_git_commit=expected_git_commit,
    )


__all__ = [
    "CANONICALIZATION",
    "OBSERVED_P95_AUTHORITY_ID",
    "OBSERVED_P95_AUTHORITY_RECEIPT_FILENAME",
    "OBSERVED_P95_AUTHORITY_RECEIPT_SCHEMA_VERSION",
    "OBSERVED_P95_PROJECTION_SCHEMA_VERSION",
    "OBSERVED_P95_SOURCE_KIND",
    "ObservedP95AuthorityError",
    "build_observed_p95_authority_receipt",
    "validate_observed_p95_authority_receipt",
    "verified_observed_p95_authority_binding",
    "verify_observed_p95_authority_receipt",
]
