"""Provider-free release source authority for the V2.11.11 fresh cohort.

The manifest deliberately has three independent roots: the release being
prepared, the immutable V2.11.10 terminal checkout, and the immutable V2.11.5
science-authority checkout.  It binds the complete Python runtime surface and
both historical lineages without importing any historical effect row.

Only five literal values are normalized to break the source-manifest cycle:
the three V2.11.11 pins in :mod:`pilot_contract` and the two values in the
unique V2.11.11 CI source-manifest anchor.  Historical pins remain part of the
complete normalized AST and therefore cannot drift unnoticed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence

from .pilot_contract import (
    PILOT_CONTRACT_V2_11_11_SCIENCE_DESIGN_SHA256,
    PilotContract,
    canonical_sha256,
    science_design_sha256,
)
from . import pilot_v21111_fresh_cohort as fresh


V21111_CONTRACT_ID = "finevo-pilot-v2.11.11"
V21111_SCIENCE_TAG = "pilot-v2.11.11-science"
V21111_SOURCE_MANIFEST_PATH = PurePosixPath(
    "experiments/pilot_v2_11_11_source_manifest.json"
)
V21111_SOURCE_MANIFEST_SCHEMA_VERSION = "finevo-pilot-v2.11.11-source-manifest-v1"
V21111_SOURCE_REPLAY_SCHEMA_VERSION = "finevo-pilot-v2.11.11-source-manifest-replay-v1"

_CURRENT_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_11_11.yaml")
_V21110_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_11_10.yaml")
_V21110_SOURCE_PATH = PurePosixPath("experiments/pilot_v2_11_10_source_manifest.json")
_V2115_CONTRACT_PATH = PurePosixPath("experiments/pilot_v2_11_5.yaml")
_V2115_SOURCE_PATH = PurePosixPath("experiments/pilot_v2_11_5_source_manifest.json")

_CYCLIC_CONTRACT_PIN_NAMES = frozenset(
    {
        "PILOT_CONTRACT_V2_11_11_CANONICAL_SHA256",
        "PILOT_V2_11_11_SOURCE_MANIFEST_FILE_SHA256",
        "PILOT_V2_11_11_SOURCE_MANIFEST_CONTENT_SHA256",
    }
)
_SOURCE_PIN_NAMES = frozenset(
    {
        "PILOT_V2_11_11_SOURCE_MANIFEST_FILE_SHA256",
        "PILOT_V2_11_11_SOURCE_MANIFEST_CONTENT_SHA256",
    }
)
_NORMALIZED_AST_PATHS = frozenset(
    {
        "verified_memory/pilot_contract.py",
        "verified_memory/ci_release_receipt.py",
    }
)
_RELEASE_ENTRY_PATHS = frozenset(
    {
        "run_pilot.py",
        "llm_providers.py",
        "scripts/render_pilot_v21111_contract.py",
        "scripts/render_pilot_v21111_source_manifest.py",
    }
)
_BOUND_DATA_PATHS = ("config.yaml", "data/profiles.json")


class PilotV21111ReleaseError(ValueError):
    """Raised when the V2.11.11 release source authority is not exact."""


def _json_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_copy(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_copy(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise PilotV21111ReleaseError(
        f"value of type {type(value).__name__} is not canonical JSON"
    )


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PilotV21111ReleaseError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                PilotV21111ReleaseError(f"{name} contains non-finite value {item!r}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotV21111ReleaseError(f"{name} is unavailable or invalid") from exc
    if not isinstance(value, dict):
        raise PilotV21111ReleaseError(f"{name} must be a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PilotV21111ReleaseError(f"cannot hash {path}") from exc
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
    )


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_copy(payload)
    result["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
    }
    result["integrity"]["content_sha256"] = canonical_sha256(result)
    return result


def _verify_seal(payload: Mapping[str, Any], *, name: str) -> None:
    integrity = payload.get("integrity")
    unsigned = _json_copy(payload)
    unsigned_integrity = unsigned.get("integrity")
    if isinstance(unsigned_integrity, dict):
        unsigned_integrity.pop("content_sha256", None)
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("canonicalization") != "json-sort-keys-utf8-v1"
        or integrity.get("content_sha256") != canonical_sha256(unsigned)
    ):
        raise PilotV21111ReleaseError(f"{name} integrity drifted")


def _real_root(value: str | Path, *, name: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(value)))
    try:
        mode = lexical.lstat().st_mode
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise PilotV21111ReleaseError(f"{name} is unavailable") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode) or resolved != lexical:
        raise PilotV21111ReleaseError(
            f"{name} must be a real non-symlink directory without symlink parents"
        )
    return resolved


def _require_distinct_roots(**roots: Path) -> None:
    names = tuple(roots)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            try:
                aliased = os.path.samefile(roots[left_name], roots[right_name])
            except OSError as exc:
                raise PilotV21111ReleaseError(
                    "cannot establish independent release roots"
                ) from exc
            if aliased:
                raise PilotV21111ReleaseError(
                    f"{left_name} and {right_name} roots must be distinct"
                )


def _safe_relative_file(root: Path, relative: str, *, name: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise PilotV21111ReleaseError(f"{name} has an unsafe relative path")
    path = root
    for index, part in enumerate(pure.parts):
        path = path / part
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise PilotV21111ReleaseError(f"{name} is unavailable") from exc
        if stat.S_ISLNK(mode):
            raise PilotV21111ReleaseError(f"{name} contains a symlink component")
        if index < len(pure.parts) - 1 and not stat.S_ISDIR(mode):
            raise PilotV21111ReleaseError(f"{name} parent is not a directory")
    if not stat.S_ISREG(mode) or path.resolve(strict=True) != path:
        raise PilotV21111ReleaseError(f"{name} must be a regular non-symlink file")
    return path


def _source_file_binding(root: Path, relative: str) -> dict[str, Any]:
    path = _safe_relative_file(
        root,
        relative,
        name=f"required V2.11.11 source {relative}",
    )
    return {
        "path": relative,
        "byte_size": path.stat().st_size,
        "file_sha256": _file_sha256(path),
    }


def _regular_python_tree_paths(root: Path, relative_directory: str) -> tuple[str, ...]:
    directory = root
    for part in PurePosixPath(relative_directory).parts:
        directory = directory / part
        try:
            mode = directory.lstat().st_mode
        except OSError as exc:
            raise PilotV21111ReleaseError(
                f"required source directory is unavailable: {relative_directory}"
            ) from exc
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise PilotV21111ReleaseError(
                f"required source directory contains a symlink: {relative_directory}"
            )
    paths: list[str] = []
    for path in directory.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise PilotV21111ReleaseError(
                f"V2.11.11 source inventory contains a symlink: {relative}"
            )
        if path.is_file():
            if path.suffix == ".py":
                paths.append(relative)
        elif not path.is_dir():
            raise PilotV21111ReleaseError(
                f"V2.11.11 source inventory contains a non-regular entry: {relative}"
            )
    return tuple(sorted(paths))


def _release_python_source_paths(root: Path) -> tuple[str, ...]:
    paths = set(_RELEASE_ENTRY_PATHS)
    paths.update(_regular_python_tree_paths(root, "verified_memory"))
    paths.update(_regular_python_tree_paths(root, "ai_economist/foundation"))
    missing = sorted(relative for relative in paths if not (root / relative).is_file())
    if missing:
        raise PilotV21111ReleaseError(
            "V2.11.11 release Python inventory is incomplete: "
            + ", ".join(missing[:10])
        )
    return tuple(sorted(paths))


def _literal_pin_state(node: ast.AST, *, name: str) -> str:
    if isinstance(node, ast.Constant) and node.value is None:
        return "none"
    if isinstance(node, ast.Constant) and _is_sha256(node.value):
        return "sha256"
    raise PilotV21111ReleaseError(
        f"{name} must be one literal lowercase SHA-256 or bootstrap None"
    )


def normalized_contract_module_ast_binding(path: Path) -> dict[str, Any]:
    """Bind the complete module AST, replacing only the three V2.11.11 pins."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise PilotV21111ReleaseError(
            "cannot normalize the V2.11.11 contract module AST"
        ) from exc
    assignments: dict[str, ast.AST] = {}
    states: dict[str, str] = {}
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        elif isinstance(node, ast.Assign):
            names = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
        matched = [name for name in names if name in _CYCLIC_CONTRACT_PIN_NAMES]
        if not matched:
            continue
        if len(matched) != 1 or len(names) != 1 or matched[0] in assignments:
            raise PilotV21111ReleaseError(
                "V2.11.11 cyclic contract pin assignment is ambiguous"
            )
        pin = matched[0]
        states[pin] = _literal_pin_state(node.value, name=pin)
        assignments[pin] = node
        node.value = ast.Constant(value="<v21111-release-cycle-pin>")
    if set(assignments) != set(_CYCLIC_CONTRACT_PIN_NAMES):
        raise PilotV21111ReleaseError("V2.11.11 cyclic contract pin set is incomplete")
    none_names = frozenset(name for name, state in states.items() if state == "none")
    allowed_none_sets = {
        frozenset(_CYCLIC_CONTRACT_PIN_NAMES),
        frozenset({"PILOT_CONTRACT_V2_11_11_CANONICAL_SHA256"}),
        frozenset(),
    }
    if none_names not in allowed_none_sets:
        raise PilotV21111ReleaseError(
            "V2.11.11 source pins must be sealed atomically and before the "
            "canonical pin"
        )
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return {
        "normalization_schema_version": (
            "finevo-pilot-v2.11.11-complete-module-ast-cycle-normalization-v1"
        ),
        "normalized_ast_sha256": canonical_sha256(normalized),
        "top_level_node_count": len(tree.body),
        "replaced_cycle_pins": sorted(assignments),
    }


def normalized_ci_release_module_ast_binding(path: Path) -> dict[str, Any]:
    """Bind the complete CI module, replacing only the unique V2.11.11 row."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise PilotV21111ReleaseError(
            "cannot normalize the V2.11.11 CI release module AST"
        ) from exc
    anchor_assignments: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        elif isinstance(node, ast.Assign):
            names = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
        else:
            names = []
        if "SCIENTIFIC_SOURCE_MANIFEST_ANCHORS" in names:
            anchor_assignments.append(node)
    if len(anchor_assignments) != 1:
        raise PilotV21111ReleaseError(
            "CI source-manifest anchor assignment is ambiguous"
        )
    matching_rows = 0
    states: dict[str, str] = {}
    for node in ast.walk(anchor_assignments[0]):
        if not isinstance(node, ast.Dict):
            continue
        literal_keys = [
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        matching_path_indices = [
            index
            for index, (key, value) in enumerate(zip(node.keys, node.values))
            if isinstance(key, ast.Constant)
            and key.value == "path"
            and isinstance(value, ast.Constant)
            and value.value == V21111_SOURCE_MANIFEST_PATH.as_posix()
        ]
        if not matching_path_indices:
            continue
        if len(matching_path_indices) != 1 or len(literal_keys) != len(
            set(literal_keys)
        ):
            raise PilotV21111ReleaseError(
                "V2.11.11 CI source-manifest anchor contains duplicate keys"
            )
        indices = {
            key.value: index
            for index, key in enumerate(node.keys)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        path_index = indices.get("path")
        assert path_index is not None
        path_value = node.values[path_index]
        if (
            not isinstance(path_value, ast.Constant)
            or path_value.value != V21111_SOURCE_MANIFEST_PATH.as_posix()
        ):
            continue
        matching_rows += 1
        for field in ("file_sha256", "content_sha256"):
            index = indices.get(field)
            if index is None:
                raise PilotV21111ReleaseError(
                    "V2.11.11 CI source-manifest anchor is incomplete"
                )
            states[field] = _literal_pin_state(
                node.values[index], name=f"V2.11.11 CI {field}"
            )
            node.values[index] = ast.Constant(value="<v21111-release-cycle-pin>")
    if matching_rows != 1 or set(states) != {"file_sha256", "content_sha256"}:
        raise PilotV21111ReleaseError(
            "V2.11.11 CI source-manifest anchor must be unique and complete"
        )
    if len(set(states.values())) != 1:
        raise PilotV21111ReleaseError(
            "V2.11.11 CI source-manifest pins must be sealed atomically"
        )
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return {
        "normalization_schema_version": (
            "finevo-pilot-v2.11.11-complete-ci-module-ast-cycle-normalization-v1"
        ),
        "normalized_ast_sha256": canonical_sha256(normalized),
        "top_level_node_count": len(tree.body),
        "replaced_cycle_pins": ["content_sha256", "file_sha256"],
    }


def _current_runtime_source_bindings(root: Path) -> dict[str, Any]:
    paths = _release_python_source_paths(root)
    if not _NORMALIZED_AST_PATHS < set(paths):
        raise PilotV21111ReleaseError(
            "V2.11.11 normalized modules are absent from the complete inventory"
        )
    full_paths = tuple(path for path in paths if path not in _NORMALIZED_AST_PATHS)
    full_bindings = [_source_file_binding(root, path) for path in full_paths]
    data_bindings = [_source_file_binding(root, path) for path in _BOUND_DATA_PATHS]
    verified_paths = tuple(
        path for path in paths if path.startswith("verified_memory/")
    )
    foundation_paths = tuple(
        path for path in paths if path.startswith("ai_economist/foundation/")
    )
    entry_paths = tuple(path for path in paths if path in _RELEASE_ENTRY_PATHS)
    return {
        "release_python_source_paths": list(paths),
        "release_python_source_path_set_sha256": canonical_sha256(paths),
        "full_file_bindings": full_bindings,
        "full_file_binding_set_sha256": canonical_sha256(full_bindings),
        "complete_inventory_partitions": {
            "verified_memory_python_paths": list(verified_paths),
            "verified_memory_python_path_set_sha256": canonical_sha256(verified_paths),
            "foundation_python_paths": list(foundation_paths),
            "foundation_python_path_set_sha256": canonical_sha256(foundation_paths),
            "release_entry_paths": list(entry_paths),
            "release_entry_path_set_sha256": canonical_sha256(entry_paths),
        },
        "pilot_contract_path": "verified_memory/pilot_contract.py",
        "pilot_contract_complete_module_ast_binding": (
            normalized_contract_module_ast_binding(
                root / "verified_memory/pilot_contract.py"
            )
        ),
        "ci_release_receipt_path": "verified_memory/ci_release_receipt.py",
        "ci_release_receipt_complete_module_ast_binding": (
            normalized_ci_release_module_ast_binding(
                root / "verified_memory/ci_release_receipt.py"
            )
        ),
        "bound_data_files": data_bindings,
        "bound_data_file_set_sha256": canonical_sha256(data_bindings),
        "cycle_avoidance": (
            "The complete pilot_contract.py and ci_release_receipt.py ASTs are "
            "bound; only the three V2.11.11 contract pins and two values in "
            "the unique V2.11.11 CI anchor are replaced. Historical pins are "
            "not normalized."
        ),
    }


def _historical_source_binding(
    root: Path,
    *,
    contract_path: PurePosixPath,
    boundary_name: str,
    source_path: PurePosixPath,
    contract_id: str,
    source_declares_contract_id: bool,
) -> dict[str, Any]:
    contract_file = _safe_relative_file(
        root,
        contract_path.as_posix(),
        name=f"{contract_id} contract",
    )
    source_file = _safe_relative_file(
        root,
        source_path.as_posix(),
        name=f"{contract_id} source manifest",
    )
    contract_document = _strict_json(contract_file, name=f"{contract_id} contract")
    boundary = contract_document.get(boundary_name)
    declared = (
        boundary.get("source_manifest") if isinstance(boundary, Mapping) else None
    )
    source = _strict_json(source_file, name=f"{contract_id} source manifest")
    _verify_seal(source, name=f"{contract_id} source manifest")
    observed = {
        "path": source_path.as_posix(),
        "schema_version": source.get("schema_version"),
        "file_sha256": _file_sha256(source_file),
        "content_sha256": source.get("integrity", {}).get("content_sha256"),
    }
    if (
        contract_document.get("contract_id") != contract_id
        or (source_declares_contract_id and source.get("contract_id") != contract_id)
        or (not source_declares_contract_id and "contract_id" in source)
        or not isinstance(declared, Mapping)
        or _json_copy(declared) != observed
    ):
        raise PilotV21111ReleaseError(
            f"{contract_id} historical source-manifest binding drifted"
        )
    return observed


def _contract_design_binding(root: Path, contract: PilotContract) -> dict[str, Any]:
    path = _safe_relative_file(
        root,
        _CURRENT_CONTRACT_PATH.as_posix(),
        name="V2.11.11 contract",
    )
    document = _strict_json(path, name="V2.11.11 contract")
    boundary = document.get("v21111_fresh_cohort_boundary")
    source = boundary.get("source_manifest") if isinstance(boundary, Mapping) else None
    design = science_design_sha256(document)
    if (
        document.get("contract_id") != V21111_CONTRACT_ID
        or contract.contract_id != V21111_CONTRACT_ID
        or contract.to_dict() != document
        or (
            PILOT_CONTRACT_V2_11_11_SCIENCE_DESIGN_SHA256 is not None
            and design != PILOT_CONTRACT_V2_11_11_SCIENCE_DESIGN_SHA256
        )
        or not isinstance(source, Mapping)
        or source.get("path") != V21111_SOURCE_MANIFEST_PATH.as_posix()
        or source.get("schema_version") != V21111_SOURCE_MANIFEST_SCHEMA_VERSION
        or (source.get("file_sha256") is None) != (source.get("content_sha256") is None)
        or str(contract.implementation.get("required_git_tag")) != V21111_SCIENCE_TAG
    ):
        raise PilotV21111ReleaseError("V2.11.11 contract design binding drifted")
    return {
        "path": _CURRENT_CONTRACT_PATH.as_posix(),
        "contract_id": V21111_CONTRACT_ID,
        "science_design_sha256": design,
        "required_git_tag": V21111_SCIENCE_TAG,
        "source_manifest_path": V21111_SOURCE_MANIFEST_PATH.as_posix(),
        "source_manifest_schema_version": V21111_SOURCE_MANIFEST_SCHEMA_VERSION,
        "design_identity_algorithm": "pilot-contract-science-design-sha256-v1",
    }


def _release_lineage_bindings(
    *,
    contract: PilotContract,
    child: Path,
    terminal: Path,
    authority: Path,
) -> dict[str, Any]:
    try:
        parent = fresh.verify_parent_sources(
            contract,
            repo_root=child,
            v21110_repo_root=terminal,
            v2115_repo_root=authority,
        )
    except fresh.PilotV21111FreshCohortError as exc:
        raise PilotV21111ReleaseError(str(exc)) from exc
    boundary = contract.v21111_fresh_cohort_boundary
    if not isinstance(boundary, Mapping):
        raise PilotV21111ReleaseError("V2.11.11 lineage boundary is absent")
    terminal_source = _historical_source_binding(
        terminal,
        contract_path=_V21110_CONTRACT_PATH,
        boundary_name="v21110_recovery_boundary",
        source_path=_V21110_SOURCE_PATH,
        contract_id="finevo-pilot-v2.11.10",
        source_declares_contract_id=True,
    )
    authority_source = _historical_source_binding(
        authority,
        contract_path=_V2115_CONTRACT_PATH,
        boundary_name="v2115_forward_boundary",
        source_path=_V2115_SOURCE_PATH,
        contract_id="finevo-pilot-v2.11.5",
        source_declares_contract_id=False,
    )
    return {
        "v21110_terminal": {
            "verified_release": _json_copy(parent["v21110_terminal"]),
            "frozen_boundary": _json_copy(boundary["v21110_terminal_release"]),
            "contract_path": _V21110_CONTRACT_PATH.as_posix(),
            "source_manifest": terminal_source,
        },
        "v2115_authority": {
            "verified_release": _json_copy(parent["v2115_authority"]),
            "frozen_boundary": _json_copy(boundary["v2115_scientific_authority"]),
            "contract_path": _V2115_CONTRACT_PATH.as_posix(),
            "source_manifest": authority_source,
            "parent_import_receipt": _json_copy(parent["v2115_parent_receipt"]),
        },
        "verification_boundary": {
            "three_release_roots_pairwise_distinct": True,
            "terminal_and_authority_git_tags_verified": True,
            "terminal_and_authority_contracts_verified": True,
            "terminal_and_authority_source_manifests_verified": True,
            "terminal_and_authority_raw_inventories_verified": True,
            "historical_effect_rows_imported": 0,
            "provider_construction": False,
            "provider_calls": 0,
        },
    }


def build_v21111_source_manifest(
    *,
    contract: PilotContract,
    repo_root: str | Path,
    v21110_repo_root: str | Path,
    v2115_repo_root: str | Path,
) -> dict[str, Any]:
    """Build the deterministic V2.11.11 three-root source manifest."""

    if contract.contract_id != V21111_CONTRACT_ID:
        raise PilotV21111ReleaseError("source manifest requires V2.11.11")
    child = _real_root(repo_root, name="V2.11.11 repository")
    terminal = _real_root(v21110_repo_root, name="V2.11.10 terminal repository")
    authority = _real_root(v2115_repo_root, name="V2.11.5 authority repository")
    _require_distinct_roots(child=child, terminal=terminal, authority=authority)
    return _seal(
        {
            "schema_version": V21111_SOURCE_MANIFEST_SCHEMA_VERSION,
            "contract_id": V21111_CONTRACT_ID,
            "release_tag": V21111_SCIENCE_TAG,
            "contract_design": _contract_design_binding(child, contract),
            "release_lineage": _release_lineage_bindings(
                contract=contract,
                child=child,
                terminal=terminal,
                authority=authority,
            ),
            "current_runtime_sources": _current_runtime_source_bindings(child),
            "observation_boundary": {
                "mechanism_fresh_cohort_only": True,
                "v21110_terminal_outcomes_reused": False,
                "v2115_outcomes_remain_external_frozen_evidence": True,
                "decoded_completion_reuse": False,
                "provider_construction": False,
                "provider_calls": 0,
                "scientific_evidence": False,
            },
        }
    )


def _declared_current_binding(contract: PilotContract) -> Mapping[str, Any]:
    boundary = contract.v21111_fresh_cohort_boundary
    declared = (
        boundary.get("source_manifest") if isinstance(boundary, Mapping) else None
    )
    if not isinstance(declared, Mapping):
        raise PilotV21111ReleaseError("V2.11.11 source-manifest declaration is absent")
    return declared


def validate_v21111_source_manifest(
    *,
    contract: PilotContract,
    repo_root: str | Path,
    v21110_repo_root: str | Path,
    v2115_repo_root: str | Path,
) -> dict[str, Any]:
    """Rebuild and compare the tracked manifest and all frozen contract pins."""

    child = _real_root(repo_root, name="V2.11.11 repository")
    path = child.joinpath(*V21111_SOURCE_MANIFEST_PATH.parts)
    path = _safe_relative_file(
        child,
        V21111_SOURCE_MANIFEST_PATH.as_posix(),
        name="V2.11.11 source manifest",
    )
    observed = _strict_json(path, name="V2.11.11 source manifest")
    _verify_seal(observed, name="V2.11.11 source manifest")
    expected = build_v21111_source_manifest(
        contract=contract,
        repo_root=child,
        v21110_repo_root=v21110_repo_root,
        v2115_repo_root=v2115_repo_root,
    )
    if observed != expected:
        raise PilotV21111ReleaseError("V2.11.11 source manifest replay drifted")
    expected_binding = {
        "path": V21111_SOURCE_MANIFEST_PATH.as_posix(),
        "schema_version": V21111_SOURCE_MANIFEST_SCHEMA_VERSION,
        "file_sha256": _file_sha256(path),
        "content_sha256": expected["integrity"]["content_sha256"],
    }
    declared = _declared_current_binding(contract)
    if {key: declared.get(key) for key in expected_binding} != expected_binding:
        raise PilotV21111ReleaseError(
            "V2.11.11 contract source-manifest identity drifted"
        )
    return observed


def replay_v21111_source_manifest(
    *,
    contract: PilotContract,
    repo_root: str | Path,
    v21110_repo_root: str | Path,
    v2115_repo_root: str | Path,
) -> dict[str, Any]:
    """Return a sealed zero-provider receipt for an exact manifest replay."""

    child = _real_root(repo_root, name="V2.11.11 repository")
    manifest = validate_v21111_source_manifest(
        contract=contract,
        repo_root=child,
        v21110_repo_root=v21110_repo_root,
        v2115_repo_root=v2115_repo_root,
    )
    path = child.joinpath(*V21111_SOURCE_MANIFEST_PATH.parts)
    return _seal(
        {
            "schema_version": V21111_SOURCE_REPLAY_SCHEMA_VERSION,
            "contract_id": V21111_CONTRACT_ID,
            "performed": True,
            "recomputed_equal": True,
            "source_root_roles_pairwise_distinct": True,
            "source_manifest": {
                "path": V21111_SOURCE_MANIFEST_PATH.as_posix(),
                "schema_version": V21111_SOURCE_MANIFEST_SCHEMA_VERSION,
                "file_sha256": _file_sha256(path),
                "content_sha256": manifest["integrity"]["content_sha256"],
            },
            "runtime_source_path_set_sha256": manifest["current_runtime_sources"][
                "release_python_source_path_set_sha256"
            ],
            "release_lineage_sha256": canonical_sha256(manifest["release_lineage"]),
            "provider_construction": False,
            "provider_calls": 0,
            "scientific_evidence": False,
        }
    )


__all__ = [
    "PilotV21111ReleaseError",
    "V21111_SOURCE_MANIFEST_PATH",
    "V21111_SOURCE_MANIFEST_SCHEMA_VERSION",
    "V21111_SOURCE_REPLAY_SCHEMA_VERSION",
    "build_v21111_source_manifest",
    "normalized_ci_release_module_ast_binding",
    "normalized_contract_module_ast_binding",
    "replay_v21111_source_manifest",
    "validate_v21111_source_manifest",
]
