"""Fail-closed artifact and provenance boundary for verified-memory runs.

Only standard-library types are accepted at this boundary.  The writer owns a
run directory, appends canonical JSONL records against declared schemas, and
seals every managed file behind a content-addressed manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence, Tuple


MANIFEST_VERSION = 1
CONFIG_PATH = "config.json"
PROVENANCE_PATH = "provenance.json"
SCHEMAS_PATH = "schemas.json"
MANIFEST_PATH = "manifest.json"
_CORE_PATHS = {CONFIG_PATH, PROVENANCE_PATH, SCHEMAS_PATH, MANIFEST_PATH}
_JSON_KINDS = {"any", "array", "boolean", "integer", "null", "number", "object", "string"}


class ArtifactError(RuntimeError):
    """Base class for artifact-boundary failures."""


class ArtifactValidationError(ArtifactError):
    """Raised when data, schema, or completeness validation fails."""


class ArtifactFinalizedError(ArtifactError):
    """Raised when a sealed run is finalized or mutated again."""


class ManifestVerificationError(ArtifactError):
    """Raised when read-only manifest verification finds any mismatch."""


def _reject_constant(value: str) -> None:
    raise ArtifactValidationError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_loads(raw: str) -> Any:
    try:
        return json.loads(raw, parse_constant=_reject_constant)
    except ArtifactValidationError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"invalid JSON: {exc}") from exc


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"value is not finite canonical JSON: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def _json_copy(value: Any) -> Any:
    return _strict_json_loads(_canonical_json_bytes(value).decode("utf-8"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactValidationError("artifact paths must be non-empty strings")
    if "\\" in value:
        raise ArtifactValidationError(f"backslashes are forbidden in artifact path: {value}")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ArtifactValidationError(f"unsafe relative artifact path: {value}")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ArtifactValidationError(f"unsafe relative artifact path: {value}")
    normalized = path.as_posix()
    if normalized in {".", ""}:
        raise ArtifactValidationError(f"unsafe relative artifact path: {value}")
    return normalized


def _safe_join(root: Path, relative_path: str) -> Path:
    normalized = _validate_relative_path(relative_path)
    root_resolved = root.resolve()
    parts = PurePosixPath(normalized).parts
    candidate = root.joinpath(*parts)
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root_resolved):
        raise ArtifactValidationError(f"artifact escapes run directory: {relative_path}")
    component = root
    for part in parts:
        component = component / part
        if component.is_symlink():
            raise ArtifactValidationError(f"artifact symlinks are forbidden: {relative_path}")
    return candidate


def _make_read_only(path: Path) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    path.chmod(mode & ~0o222)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


@dataclass(frozen=True)
class JsonField:
    """One top-level JSON object field in a stream schema."""

    name: str
    kind: str = "any"
    required: bool = True
    nullable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ArtifactValidationError("schema field names must be non-empty strings")
        if self.kind not in _JSON_KINDS:
            raise ArtifactValidationError(f"unsupported JSON field kind: {self.kind}")
        if not isinstance(self.required, bool) or not isinstance(self.nullable, bool):
            raise ArtifactValidationError("required and nullable must be booleans")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "required": self.required,
            "nullable": self.nullable,
        }


@dataclass(frozen=True)
class JsonlStreamSchema:
    """Declared schema and relative path for one canonical JSONL stream."""

    name: str
    relative_path: str
    fields: Tuple[JsonField, ...]
    required: bool = True
    min_records: int = 1
    allow_extra_fields: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or "/" in self.name or "\\" in self.name:
            raise ArtifactValidationError("stream name must be a non-empty path-free string")
        object.__setattr__(self, "relative_path", _validate_relative_path(self.relative_path))
        object.__setattr__(self, "fields", tuple(self.fields))
        if not self.relative_path.endswith(".jsonl"):
            raise ArtifactValidationError("JSONL stream paths must end with .jsonl")
        if self.relative_path in _CORE_PATHS:
            raise ArtifactValidationError(f"stream path collides with core artifact: {self.relative_path}")
        if not self.fields:
            raise ArtifactValidationError("stream schema must declare at least one field")
        if any(not isinstance(field, JsonField) for field in self.fields):
            raise ArtifactValidationError("fields must contain only JsonField values")
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ArtifactValidationError(f"duplicate field in stream schema {self.name}")
        if isinstance(self.min_records, bool) or not isinstance(self.min_records, int):
            raise ArtifactValidationError("min_records must be an integer")
        if self.min_records < 0:
            raise ArtifactValidationError("min_records must be non-negative")
        if not isinstance(self.required, bool) or not isinstance(self.allow_extra_fields, bool):
            raise ArtifactValidationError("schema flags must be booleans")
        if self.required and self.min_records < 1:
            raise ArtifactValidationError("required streams must require at least one record")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "relative_path": self.relative_path,
            "fields": [field.to_dict() for field in self.fields],
            "required": self.required,
            "min_records": self.min_records,
            "allow_extra_fields": self.allow_extra_fields,
        }

    def validate_record(self, record: Mapping[str, Any]) -> None:
        if not isinstance(record, Mapping):
            raise ArtifactValidationError(f"stream {self.name} records must be JSON objects")
        if any(not isinstance(key, str) for key in record):
            raise ArtifactValidationError(f"stream {self.name} record keys must be strings")
        field_map = {field.name: field for field in self.fields}
        missing = [field.name for field in self.fields if field.required and field.name not in record]
        if missing:
            raise ArtifactValidationError(
                f"stream {self.name} is missing required fields: {', '.join(sorted(missing))}"
            )
        extra = sorted(set(record) - set(field_map))
        if extra and not self.allow_extra_fields:
            raise ArtifactValidationError(
                f"stream {self.name} has undeclared fields: {', '.join(extra)}"
            )
        for name, value in record.items():
            field = field_map.get(name)
            if field is None:
                continue
            if value is None:
                if field.nullable or field.kind in {"any", "null"}:
                    continue
                raise ArtifactValidationError(f"stream {self.name} field {name} may not be null")
            if not _matches_kind(value, field.kind):
                raise ArtifactValidationError(
                    f"stream {self.name} field {name} must be {field.kind}"
                )
        # This is also the recursive finite/JSON-type validation.
        _canonical_json_bytes(dict(record))


def _matches_kind(value: Any, kind: str) -> bool:
    if kind == "any":
        return True
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        if isinstance(value, int) and not isinstance(value, bool):
            return True
        return isinstance(value, float) and math.isfinite(value)
    if kind == "string":
        return isinstance(value, str)
    if kind == "object":
        return isinstance(value, Mapping)
    if kind == "array":
        return isinstance(value, list)
    return False


@dataclass
class _StreamState:
    line_count: int
    byte_size: int
    digest: Any


@dataclass(frozen=True)
class ManifestVerification:
    run_dir: str
    artifact_count: int
    manifest_sha256: str
    valid: bool = True


class RunArtifactWriter:
    """Own and eventually seal one verified experiment run directory."""

    def __init__(
        self,
        run_dir: Path,
        schemas: Sequence[JsonlStreamSchema],
        *,
        config: Mapping[str, Any],
        provenance: Mapping[str, Any],
        git_commit: str,
        git_dirty: bool,
        resume: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self._lock = threading.RLock()
        self._finalized = False
        if not isinstance(git_commit, str) or not git_commit:
            raise ArtifactValidationError("git_commit must be a non-empty caller-supplied string")
        if not isinstance(git_dirty, bool):
            raise ArtifactValidationError("git_dirty must be a caller-supplied boolean")
        if not isinstance(config, Mapping) or not isinstance(provenance, Mapping):
            raise ArtifactValidationError("config and provenance must be mappings")

        schemas = tuple(schemas)
        if not schemas or any(not isinstance(schema, JsonlStreamSchema) for schema in schemas):
            raise ArtifactValidationError("at least one JsonlStreamSchema is required")
        names = [schema.name for schema in schemas]
        paths = [schema.relative_path for schema in schemas]
        if len(names) != len(set(names)):
            raise ArtifactValidationError("stream names must be unique")
        if len(paths) != len(set(paths)):
            raise ArtifactValidationError("stream paths must be unique")
        self._schemas = {schema.name: schema for schema in schemas}

        existed = self.run_dir.exists()
        if existed and not self.run_dir.is_dir():
            raise ArtifactValidationError(f"run path is not a directory: {self.run_dir}")
        existing_entries = list(self.run_dir.iterdir()) if existed else []
        if existing_entries and not resume:
            raise FileExistsError(f"run directory is nonempty; explicit resume required: {self.run_dir}")
        if resume and (self.run_dir / MANIFEST_PATH).exists():
            raise ArtifactFinalizedError("finalized runs cannot be resumed")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._config = _json_copy(dict(config))
        self._provenance = _json_copy(
            {
                "git": {"commit": git_commit, "dirty": git_dirty},
                "details": dict(provenance),
            }
        )
        self._schema_document = _json_copy(
            {
                "schema_version": 1,
                "streams": [self._schemas[name].to_dict() for name in sorted(self._schemas)],
            }
        )
        self._metadata_bytes = {
            CONFIG_PATH: _canonical_json_bytes(self._config),
            PROVENANCE_PATH: _canonical_json_bytes(self._provenance),
            SCHEMAS_PATH: _canonical_json_bytes(self._schema_document),
        }

        if resume and existing_entries:
            for relative_path, expected in self._metadata_bytes.items():
                path = _safe_join(self.run_dir, relative_path)
                if not path.is_file() or path.is_symlink() or path.read_bytes() != expected:
                    raise ArtifactValidationError(
                        f"resume metadata does not match immutable {relative_path}"
                    )
        else:
            for relative_path, data in self._metadata_bytes.items():
                path = _safe_join(self.run_dir, relative_path)
                if path.exists():
                    raise ArtifactValidationError(f"refusing to overwrite metadata: {relative_path}")
                _atomic_write(path, data)
                _make_read_only(path)

        self._states: dict[str, _StreamState] = {}
        self._load_existing_streams(resume=resume)

    @classmethod
    def create(
        cls,
        run_dir: Path,
        schemas: Sequence[JsonlStreamSchema],
        *,
        config: Mapping[str, Any],
        provenance: Mapping[str, Any],
        git_commit: str,
        git_dirty: bool,
        resume: bool = False,
    ) -> "RunArtifactWriter":
        return cls(
            run_dir,
            schemas,
            config=config,
            provenance=provenance,
            git_commit=git_commit,
            git_dirty=git_dirty,
            resume=resume,
        )

    def _assert_mutable_locked(self) -> None:
        if self._finalized or (self.run_dir / MANIFEST_PATH).exists():
            raise ArtifactFinalizedError("run artifacts are finalized and immutable")
        for relative_path, expected in self._metadata_bytes.items():
            path = _safe_join(self.run_dir, relative_path)
            if not path.is_file() or path.is_symlink() or path.read_bytes() != expected:
                raise ArtifactValidationError(f"immutable metadata was mutated: {relative_path}")

    def _load_existing_streams(self, *, resume: bool) -> None:
        declared_paths = {schema.relative_path for schema in self._schemas.values()}
        allowed_paths = {CONFIG_PATH, PROVENANCE_PATH, SCHEMAS_PATH} | declared_paths
        for file_path in self.run_dir.rglob("*"):
            if file_path.is_symlink():
                raise ArtifactValidationError(f"symlinks are forbidden in run directory: {file_path}")
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(self.run_dir).as_posix()
            if relative not in allowed_paths:
                raise ArtifactValidationError(f"undeclared artifact in run directory: {relative}")

        for name, schema in self._schemas.items():
            path = _safe_join(self.run_dir, schema.relative_path)
            if not path.exists():
                continue
            if not resume:
                raise ArtifactValidationError(f"unexpected pre-existing stream: {schema.relative_path}")
            data, count = self._validate_stream_file(schema, path)
            digest = hashlib.sha256()
            digest.update(data)
            self._states[name] = _StreamState(count, len(data), digest)

    def _validate_stream_file(self, schema: JsonlStreamSchema, path: Path) -> tuple[bytes, int]:
        if path.is_symlink() or not path.is_file():
            raise ArtifactValidationError(f"stream is not a regular file: {schema.relative_path}")
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            raise ArtifactValidationError(f"stream lacks final newline: {schema.relative_path}")
        count = 0
        for line_number, raw_line in enumerate(data.splitlines(keepends=True), start=1):
            try:
                text = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ArtifactValidationError(
                    f"stream {schema.name} line {line_number} is not UTF-8"
                ) from exc
            record = _strict_json_loads(text)
            schema.validate_record(record)
            if raw_line != _canonical_json_bytes(record):
                raise ArtifactValidationError(
                    f"stream {schema.name} line {line_number} is not canonical JSON"
                )
            count += 1
        return data, count

    def append(self, stream_name: str, record: Mapping[str, Any]) -> int:
        """Append one schema-checked canonical record and return its 1-based line number."""

        with self._lock:
            self._assert_mutable_locked()
            schema = self._schemas.get(stream_name)
            if schema is None:
                raise ArtifactValidationError(f"unknown stream: {stream_name}")
            schema.validate_record(record)
            line = _canonical_json_bytes(dict(record))
            path = _safe_join(self.run_dir, schema.relative_path)
            state = self._states.get(stream_name)
            expected_size = state.byte_size if state else 0
            if path.exists():
                if path.is_symlink() or not path.is_file():
                    raise ArtifactValidationError(f"stream path was replaced: {schema.relative_path}")
                if path.stat().st_size != expected_size:
                    raise ArtifactValidationError(f"stream was externally mutated: {schema.relative_path}")
            elif expected_size:
                raise ArtifactValidationError(f"stream disappeared: {schema.relative_path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            resolved_parent = path.parent.resolve()
            if not resolved_parent.is_relative_to(self.run_dir.resolve()):
                raise ArtifactValidationError(f"stream parent escapes run directory: {schema.relative_path}")
            with path.open("ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            if state is None:
                state = _StreamState(0, 0, hashlib.sha256())
                self._states[stream_name] = state
            state.digest.update(line)
            state.line_count += 1
            state.byte_size += len(line)
            return state.line_count

    def _artifact_entry(self, relative_path: str, kind: str, line_count: int) -> dict[str, object]:
        path = _safe_join(self.run_dir, relative_path)
        if path.is_symlink() or not path.is_file():
            raise ArtifactValidationError(f"managed artifact missing: {relative_path}")
        data = path.read_bytes()
        return {
            "path": relative_path,
            "kind": kind,
            "line_count": line_count,
            "byte_size": len(data),
            "sha256": _sha256(data),
        }

    @staticmethod
    def _normalize_snapshot(value: Any, label: str) -> Any:
        if hasattr(value, "to_dict") and callable(value.to_dict):
            value = value.to_dict()
        if not isinstance(value, Mapping):
            raise ArtifactValidationError(f"{label} must be a mapping or expose to_dict()")
        return _json_copy(dict(value))

    def finalize(
        self,
        *,
        validation_status: Mapping[str, Any],
        budget_snapshot: Any,
        result_complete: bool,
    ) -> Path:
        """Validate completeness, write the manifest atomically, and seal files."""

        with self._lock:
            self._assert_mutable_locked()
            if not isinstance(result_complete, bool):
                raise ArtifactValidationError("result_complete must be a boolean")
            validation = self._normalize_snapshot(validation_status, "validation_status")
            if not isinstance(validation.get("status"), str) or not validation["status"]:
                raise ArtifactValidationError("validation_status must contain a non-empty status")
            budget = self._normalize_snapshot(budget_snapshot, "budget_snapshot")

            missing = []
            stream_counts = {}
            for name, schema in self._schemas.items():
                path = _safe_join(self.run_dir, schema.relative_path)
                state = self._states.get(name)
                count = state.line_count if state else 0
                stream_counts[name] = count
                if schema.required and count < schema.min_records:
                    missing.append(name)
                    continue
                if state is None:
                    continue
                data, scanned_count = self._validate_stream_file(schema, path)
                if (
                    scanned_count != state.line_count
                    or len(data) != state.byte_size
                    or _sha256(data) != state.digest.hexdigest()
                ):
                    raise ArtifactValidationError(
                        f"stream changed outside writer before finalization: {schema.relative_path}"
                    )
            if missing:
                raise ArtifactValidationError(
                    f"required streams are missing or incomplete: {', '.join(sorted(missing))}"
                )

            entries = [
                self._artifact_entry(CONFIG_PATH, "config", 1),
                self._artifact_entry(PROVENANCE_PATH, "provenance", 1),
                self._artifact_entry(SCHEMAS_PATH, "schemas", 1),
            ]
            for name, schema in self._schemas.items():
                state = self._states.get(name)
                if state is not None:
                    entries.append(
                        self._artifact_entry(schema.relative_path, "jsonl_stream", state.line_count)
                    )
            entries.sort(key=lambda item: str(item["path"]))

            declared_files = {str(entry["path"]) for entry in entries}
            actual_files = set()
            for path in self.run_dir.rglob("*"):
                if path.is_symlink():
                    raise ArtifactValidationError(
                        f"symlink found before finalization: {path.relative_to(self.run_dir)}"
                    )
                if path.is_file():
                    actual_files.add(path.relative_to(self.run_dir).as_posix())
            if actual_files != declared_files:
                missing_files = sorted(declared_files - actual_files)
                extra_files = sorted(actual_files - declared_files)
                raise ArtifactValidationError(
                    "run file set does not match declared artifacts; "
                    f"missing={missing_files}, extra={extra_files}"
                )

            manifest = {
                "manifest_version": MANIFEST_VERSION,
                "git": dict(self._provenance["git"]),
                "validation_status": validation,
                "budget_snapshot": budget,
                "result": {
                    "complete": result_complete,
                    "required_streams_present": True,
                    "stream_line_counts": dict(sorted(stream_counts.items())),
                },
                "artifacts": entries,
            }
            manifest_data = _canonical_json_bytes(manifest)
            manifest_path = _safe_join(self.run_dir, MANIFEST_PATH)
            if manifest_path.exists():
                raise ArtifactFinalizedError("manifest already exists")
            _atomic_write(manifest_path, manifest_data)
            for entry in entries:
                _make_read_only(_safe_join(self.run_dir, str(entry["path"])))
            _make_read_only(manifest_path)
            self._finalized = True
            return manifest_path


# Descriptive alias used by runner code and documentation.
VerifiedArtifactWriter = RunArtifactWriter


def verify_manifest(run_dir: Path, manifest_name: str = MANIFEST_PATH) -> ManifestVerification:
    """Read and re-hash a sealed manifest without modifying any file."""

    root = Path(run_dir)
    if not root.is_dir():
        raise ManifestVerificationError(f"run directory does not exist: {root}")
    try:
        manifest_path = _safe_join(root, manifest_name)
    except ArtifactValidationError as exc:
        raise ManifestVerificationError(str(exc)) from exc
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ManifestVerificationError("manifest is missing or is not a regular file")
    manifest_data = manifest_path.read_bytes()
    try:
        manifest = _strict_json_loads(manifest_data.decode("utf-8"))
    except (UnicodeDecodeError, ArtifactValidationError) as exc:
        raise ManifestVerificationError(f"invalid manifest: {exc}") from exc
    if not isinstance(manifest, Mapping) or manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ManifestVerificationError("unsupported or missing manifest version")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ManifestVerificationError("manifest artifacts must be a list")

    declared = set()
    for entry in artifacts:
        if not isinstance(entry, Mapping):
            raise ManifestVerificationError("manifest artifact entry must be an object")
        relative = entry.get("path")
        try:
            normalized = _validate_relative_path(relative)
            path = _safe_join(root, normalized)
        except ArtifactValidationError as exc:
            raise ManifestVerificationError(str(exc)) from exc
        if normalized == manifest_name or normalized in declared:
            raise ManifestVerificationError(f"duplicate or recursive manifest entry: {normalized}")
        declared.add(normalized)
        if path.is_symlink() or not path.is_file():
            raise ManifestVerificationError(f"manifest artifact missing: {normalized}")
        data = path.read_bytes()
        actual = {
            "line_count": data.count(b"\n"),
            "byte_size": len(data),
            "sha256": _sha256(data),
        }
        for key, value in actual.items():
            if entry.get(key) != value:
                raise ManifestVerificationError(
                    f"artifact {normalized} {key} mismatch: expected {entry.get(key)!r}, got {value!r}"
                )

    actual_files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ManifestVerificationError(f"symlink found after finalization: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative != manifest_name:
                actual_files.add(relative)
    if actual_files != declared:
        missing = sorted(declared - actual_files)
        extra = sorted(actual_files - declared)
        raise ManifestVerificationError(
            f"manifest file set mismatch; missing={missing}, extra={extra}"
        )

    return ManifestVerification(
        run_dir=str(root),
        artifact_count=len(artifacts),
        manifest_sha256=_sha256(manifest_data),
    )


__all__ = [
    "ArtifactError",
    "ArtifactFinalizedError",
    "ArtifactValidationError",
    "JsonField",
    "JsonlStreamSchema",
    "ManifestVerification",
    "ManifestVerificationError",
    "RunArtifactWriter",
    "VerifiedArtifactWriter",
    "verify_manifest",
]
