"""Contract-driven, fail-closed scientific release attestation.

This v2 module intentionally lives beside :mod:`pilot_release_attestation`.
The v1 module remains the read-only compatibility path for the frozen
``pilot-v1`` release.  Scientific releases use this module and must supply two
immutable mappings:

* release requirements: exact remote/branch/tag/workflow, run attempt, and
  the two required GitHub Actions job database IDs;
* contract binding: exact contract bytes/canonical digest, policy selectors,
  and the complete sealed-manifest inventory.

Only safe parsed identifiers and content hashes are retained.  Raw command
output and CI logs are represented solely by byte counts and SHA-256 digests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

from .artifacts import verify_manifest
from .pilot_release_attestation import CommandEvidence, CommandResult


SCIENTIFIC_RELEASE_ATTESTATION_SCHEMA_VERSION = (
    "finevo-scientific-release-attestation-v2"
)
SCIENTIFIC_LAUNCH_INPUT_SCHEMA_VERSION = "finevo-scientific-launch-input-v1"
CI_JOB_RECEIPT_SCHEMA_VERSION = "finevo-ci-job-receipt-v1"
CI_JOB_RECEIPT_LOG_PREFIX = "FINEVO_CI_RELEASE_RECEIPT_JSON="

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_REF_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_POLICY_NAMES = ("provider_policy", "price_policy", "budget_policy")
_PROVIDER_POLICY_POINTERS = (
    "/provider_profiles",
    "/parameter_dispatch_policy",
    "/task_output_contracts",
    "/model_roles",
)
_BUDGET_POLICY_POINTERS = (
    "/budgets",
    "/denominator_policy",
    "/stop_go",
)
_CI_JOB_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "repository",
    "head_sha",
    "run_id",
    "run_attempt",
    "job_name",
    "job_key",
    "runner_os",
    "workflow_name",
    "workflow_file",
    "workflow_ref",
    "workflow_source_sha",
    "workflow_file_sha256",
    "workflow_blob_oid",
    "test_count",
    "test_collection_sha256",
    "skipped_test_count",
    "compiled_source_count",
    "compiled_source_inventory_sha256",
    "sealed_manifest_count",
    "sealed_manifest_inventory_sha256",
    "receipt_sha256",
}


class ScientificReleaseAttestationError(RuntimeError):
    """Raised when any release, contract, CI, or inventory binding drifts."""


@dataclass(frozen=True, slots=True)
class RequiredScientificCIJob:
    """One exact required job from one exact workflow run attempt."""

    name: str
    database_id: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "database_id": self.database_id}


@dataclass(frozen=True, slots=True)
class ScientificReleaseRequirements:
    """Normalized immutable release requirements supplied by the contract."""

    remote: str
    branch: str
    tag: str
    workflow_file: str
    workflow_name: str
    required_job_names: tuple[str, str]
    expected_test_count: int
    expected_test_collection_sha256: str
    expected_compiled_source_count: int
    expected_compiled_source_inventory_sha256: str
    expected_sealed_manifest_inventory_sha256: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        compatibility: Mapping[str, Any] | None = None,
    ) -> "ScientificReleaseRequirements":
        normalized = adapt_contract_release_requirements(
            value, compatibility=compatibility
        )
        remote = _ref_text(
            normalized["remote"], "release_requirements.remote"
        )
        branch = _ref_text(
            normalized["branch"], "release_requirements.branch"
        )
        tag = _ref_text(normalized["tag"], "release_requirements.tag")
        workflow_file = _safe_relative_path(
            normalized["workflow_file"],
            "release_requirements.workflow_file",
        )
        if not workflow_file.startswith(".github/workflows/"):
            raise ScientificReleaseAttestationError(
                "release_requirements.workflow_file must be under "
                ".github/workflows"
            )
        job_names_raw = normalized["required_job_names"]
        if (
            not isinstance(job_names_raw, Sequence)
            or isinstance(job_names_raw, (str, bytes, bytearray))
            or len(job_names_raw) != 2
        ):
            raise ScientificReleaseAttestationError(
                "scientific release requires exactly two CI job names"
            )
        job_names = tuple(
            _text(
                item,
                f"release_requirements.required_job_names[{index}]",
            )
            for index, item in enumerate(job_names_raw)
        )
        if len(set(job_names)) != 2:
            raise ScientificReleaseAttestationError(
                "required CI job names must be unique"
            )
        return cls(
            remote=remote,
            branch=branch,
            tag=tag,
            workflow_file=workflow_file,
            workflow_name=_text(
                normalized["workflow_name"],
                "release_requirements.workflow_name",
            ),
            required_job_names=(job_names[0], job_names[1]),
            expected_test_count=_positive_int(
                normalized["expected_ci"]["test_count"],
                "release_requirements.expected_ci.test_count",
            ),
            expected_test_collection_sha256=_sha256(
                normalized["expected_ci"]["test_collection_sha256"],
                (
                    "release_requirements.expected_ci."
                    "test_collection_sha256"
                ),
            ),
            expected_compiled_source_count=_positive_int(
                normalized["expected_ci"]["compiled_source_count"],
                (
                    "release_requirements.expected_ci."
                    "compiled_source_count"
                ),
            ),
            expected_compiled_source_inventory_sha256=_sha256(
                normalized["expected_ci"][
                    "compiled_source_inventory_sha256"
                ],
                (
                    "release_requirements.expected_ci."
                    "compiled_source_inventory_sha256"
                ),
            ),
            expected_sealed_manifest_inventory_sha256=_sha256(
                normalized["expected_ci"][
                    "sealed_manifest_inventory_sha256"
                ],
                (
                    "release_requirements.expected_ci."
                    "sealed_manifest_inventory_sha256"
                ),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "remote": self.remote,
            "branch": self.branch,
            "tag": self.tag,
            "workflow_file": self.workflow_file,
            "workflow_name": self.workflow_name,
            "required_job_names": list(self.required_job_names),
            "expected_ci": {
                "test_count": self.expected_test_count,
                "test_collection_sha256": (
                    self.expected_test_collection_sha256
                ),
                "compiled_source_count": (
                    self.expected_compiled_source_count
                ),
                "compiled_source_inventory_sha256": (
                    self.expected_compiled_source_inventory_sha256
                ),
                "sealed_manifest_inventory_sha256": (
                    self.expected_sealed_manifest_inventory_sha256
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class ScientificCIRunSelection:
    """Post-tag dynamic GitHub run/attempt and exact job database IDs."""

    run_id: int
    run_attempt: int
    jobs: tuple[RequiredScientificCIJob, RequiredScientificCIJob]

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "ScientificCIRunSelection":
        mapping = _exact_mapping(
            value,
            required={"run_id", "run_attempt", "jobs"},
            name="ci_run_selection",
        )
        jobs_raw = mapping["jobs"]
        if (
            not isinstance(jobs_raw, Sequence)
            or isinstance(jobs_raw, (str, bytes, bytearray))
            or len(jobs_raw) != 2
        ):
            raise ScientificReleaseAttestationError(
                "ci_run_selection.jobs must contain exactly two jobs"
            )
        jobs: list[RequiredScientificCIJob] = []
        for index, item in enumerate(jobs_raw):
            job = _exact_mapping(
                item,
                required={"name", "database_id"},
                name=f"ci_run_selection.jobs[{index}]",
            )
            jobs.append(
                RequiredScientificCIJob(
                    name=_text(
                        job["name"],
                        f"ci_run_selection.jobs[{index}].name",
                    ),
                    database_id=_positive_int(
                        job["database_id"],
                        f"ci_run_selection.jobs[{index}].database_id",
                    ),
                )
            )
        if len({job.name for job in jobs}) != 2:
            raise ScientificReleaseAttestationError(
                "selected CI job names must be unique"
            )
        if len({job.database_id for job in jobs}) != 2:
            raise ScientificReleaseAttestationError(
                "selected CI job database IDs must be unique"
            )
        return cls(
            run_id=_positive_int(
                mapping["run_id"], "ci_run_selection.run_id"
            ),
            run_attempt=_positive_int(
                mapping["run_attempt"], "ci_run_selection.run_attempt"
            ),
            jobs=(jobs[0], jobs[1]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_attempt": self.run_attempt,
            "jobs": [job.to_dict() for job in self.jobs],
        }


@dataclass(frozen=True, slots=True)
class PolicyBinding:
    """Canonical hash of one named policy assembled from JSON pointers."""

    pointers: tuple[str, ...]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"pointers": list(self.pointers), "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ScientificContractBinding:
    """Expected contract and evidence inventory hashes."""

    contract_path: str
    contract_file_sha256: str
    contract_canonical_sha256: str
    policies: Mapping[str, PolicyBinding]
    sealed_manifest_paths: tuple[str, ...]
    sealed_manifest_inventory_sha256: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "ScientificContractBinding":
        mapping = _exact_mapping(
            value,
            required={
                "contract_path",
                "contract_file_sha256",
                "contract_canonical_sha256",
                "policies",
                "sealed_manifest_paths",
                "sealed_manifest_inventory_sha256",
            },
            name="contract_binding",
        )
        policies_raw = _exact_mapping(
            mapping["policies"],
            required=set(_POLICY_NAMES),
            name="contract_binding.policies",
        )
        policies: dict[str, PolicyBinding] = {}
        for policy_name in _POLICY_NAMES:
            raw = _exact_mapping(
                policies_raw[policy_name],
                required={"pointers", "sha256"},
                name=f"contract_binding.policies.{policy_name}",
            )
            pointers_raw = raw["pointers"]
            if not isinstance(pointers_raw, Sequence) or isinstance(
                pointers_raw, (str, bytes, bytearray)
            ):
                raise ScientificReleaseAttestationError(
                    f"contract_binding.policies.{policy_name}.pointers "
                    "must be an array"
                )
            pointers = tuple(
                _json_pointer(
                    item,
                    (
                        "contract_binding.policies."
                        f"{policy_name}.pointers[{index}]"
                    ),
                )
                for index, item in enumerate(pointers_raw)
            )
            if not pointers or len(set(pointers)) != len(pointers):
                raise ScientificReleaseAttestationError(
                    f"{policy_name} pointers must be non-empty and unique"
                )
            policies[policy_name] = PolicyBinding(
                pointers=pointers,
                sha256=_sha256(
                    raw["sha256"],
                    f"contract_binding.policies.{policy_name}.sha256",
                ),
            )

        manifests_raw = mapping["sealed_manifest_paths"]
        if not isinstance(manifests_raw, Sequence) or isinstance(
            manifests_raw, (str, bytes, bytearray)
        ):
            raise ScientificReleaseAttestationError(
                "contract_binding.sealed_manifest_paths must be an array"
            )
        manifests = tuple(
            _safe_relative_path(
                item,
                f"contract_binding.sealed_manifest_paths[{index}]",
            )
            for index, item in enumerate(manifests_raw)
        )
        if not manifests or tuple(sorted(manifests)) != manifests:
            raise ScientificReleaseAttestationError(
                "sealed manifest paths must be a non-empty sorted array"
            )
        if len(set(manifests)) != len(manifests):
            raise ScientificReleaseAttestationError(
                "sealed manifest paths must be unique"
            )
        if any(not path.endswith("/manifest.json") for path in manifests):
            raise ScientificReleaseAttestationError(
                "every sealed manifest path must end in /manifest.json"
            )
        return cls(
            contract_path=_safe_relative_path(
                mapping["contract_path"], "contract_binding.contract_path"
            ),
            contract_file_sha256=_sha256(
                mapping["contract_file_sha256"],
                "contract_binding.contract_file_sha256",
            ),
            contract_canonical_sha256=_sha256(
                mapping["contract_canonical_sha256"],
                "contract_binding.contract_canonical_sha256",
            ),
            policies=MappingProxyType(policies),
            sealed_manifest_paths=manifests,
            sealed_manifest_inventory_sha256=_sha256(
                mapping["sealed_manifest_inventory_sha256"],
                "contract_binding.sealed_manifest_inventory_sha256",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_path": self.contract_path,
            "contract_file_sha256": self.contract_file_sha256,
            "contract_canonical_sha256": self.contract_canonical_sha256,
            "policies": {
                name: self.policies[name].to_dict()
                for name in _POLICY_NAMES
            },
            "sealed_manifest_paths": list(self.sealed_manifest_paths),
            "sealed_manifest_inventory_sha256": (
                self.sealed_manifest_inventory_sha256
            ),
        }


@dataclass(frozen=True, slots=True)
class ScientificReleaseAttestation:
    """Canonical JSON-compatible v2 attestation."""

    payload: Mapping[str, Any]
    attestation_sha256: str = ""

    def __post_init__(self) -> None:
        frozen_payload = _freeze_json(self.payload)
        computed = canonical_sha256(frozen_payload)
        if self.attestation_sha256 and self.attestation_sha256 != computed:
            raise ScientificReleaseAttestationError(
                "scientific release attestation self-hash mismatch"
            )
        object.__setattr__(self, "payload", frozen_payload)
        object.__setattr__(self, "attestation_sha256", computed)

    def to_dict(self) -> dict[str, Any]:
        result = _json_copy(self.payload)
        result["attestation_sha256"] = self.attestation_sha256
        return result

    def verify_hash(self) -> None:
        if canonical_sha256(self.payload) != self.attestation_sha256:
            raise ScientificReleaseAttestationError(
                "scientific release attestation self-hash mismatch"
            )


CommandRunner = Callable[[Sequence[str], Path], CommandResult]


def read_only_command_runner(
    argv: Sequence[str], cwd: Path
) -> CommandResult:
    """Execute one attestation command without prompts or raw-output logging."""

    environment = dict(os.environ)
    environment.update(
        {
            "GH_NO_UPDATE_NOTIFIER": "1",
            "GH_PROMPT_DISABLED": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    completed = subprocess.run(
        tuple(argv),
        cwd=cwd,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return CommandResult(
        stdout=bytes(completed.stdout),
        stderr=bytes(completed.stderr),
        returncode=int(completed.returncode),
    )


def canonical_sha256(value: Any) -> str:
    """Return SHA-256 over strict canonical JSON."""

    return hashlib.sha256(_canonical_json_bytes(_json_copy(value))).hexdigest()


def canonical_contract_sha256(value: Mapping[str, Any]) -> str:
    """Hash a pilot contract excluding only its declared self-hash."""

    payload = _json_copy(value)
    integrity = payload.get("integrity")
    if not isinstance(integrity, dict):
        raise ScientificReleaseAttestationError(
            "contract integrity must be a JSON object"
        )
    integrity.pop("declared_sha256", None)
    payload["integrity"] = integrity
    return canonical_sha256(payload)


def adapt_contract_release_requirements(
    value: Mapping[str, Any],
    *,
    compatibility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize current/final ``ReleaseRequirements.to_dict()`` shapes.

    The final contract should provide the canonical static names directly.
    The current transition shape uses ``required_*`` aliases and null
    post-release placeholders.  Callers may supply missing *static* values in
    ``compatibility`` while that shape is migrated; concrete contract values
    can never be overridden.  Dynamic run/attempt/job IDs are deliberately
    rejected here and belong in :class:`ScientificCIRunSelection`.
    """

    source = _mapping(value, "release_requirements")
    fallback = (
        {}
        if compatibility is None
        else _mapping(compatibility, "release_compatibility")
    )
    allowed = {
        # Canonical final static fields.
        "remote",
        "branch",
        "tag",
        "workflow_file",
        "workflow_name",
        "required_job_names",
        "expected_ci",
        "expected_test_count",
        "expected_test_collection_sha256",
        "expected_compiled_source_count",
        "expected_compiled_source_inventory_sha256",
        "expected_sealed_manifest_inventory_sha256",
        # Current ReleaseRequirements.to_dict aliases/placeholders.
        "required_remote",
        "required_branch",
        "required_annotated_tag",
        "required_workflow",
        "required_workflow_name",
        "required_platforms",
        "peeled_commit",
        "workflow_run_id",
        "workflow_run_attempt",
        "ubuntu_job_id",
        "macos_job_id",
        "test_count",
        "test_collection_sha256",
        "compiled_source_count",
        "compiled_source_inventory_sha256",
        "manifest_sha256",
        "contract_sha256",
        "catalog_sha256",
        "price_snapshot_sha256",
        "budget_sha256",
    }
    unknown = set(source) - allowed
    if unknown:
        raise ScientificReleaseAttestationError(
            "release_requirements contains unsupported fields: "
            f"{sorted(unknown)}"
        )
    dynamic_placeholders = (
        "peeled_commit",
        "workflow_run_id",
        "workflow_run_attempt",
        "ubuntu_job_id",
        "macos_job_id",
    )
    if any(source.get(name) is not None for name in dynamic_placeholders):
        raise ScientificReleaseAttestationError(
            "dynamic CI IDs must remain outside static contract requirements"
        )
    platforms = source.get("required_platforms")
    if platforms is not None and list(platforms) != ["ubuntu", "macos"]:
        raise ScientificReleaseAttestationError(
            "current release requirements must retain Ubuntu and macOS order"
        )

    def select(
        canonical_name: str,
        aliases: Sequence[str] = (),
    ) -> Any:
        observed = [
            source[name]
            for name in (canonical_name, *aliases)
            if name in source and source[name] is not None
        ]
        if observed:
            first = observed[0]
            if any(item != first for item in observed[1:]):
                raise ScientificReleaseAttestationError(
                    f"conflicting release requirement aliases for "
                    f"{canonical_name}"
                )
            fallback_value = fallback.get(canonical_name)
            if fallback_value is not None and fallback_value != first:
                raise ScientificReleaseAttestationError(
                    f"release compatibility cannot override {canonical_name}"
                )
            return first
        return fallback.get(canonical_name)

    workflow_file = select("workflow_file", ("required_workflow",))
    if isinstance(workflow_file, str) and "/" not in workflow_file:
        workflow_file = f".github/workflows/{workflow_file}"

    expected_source = source.get("expected_ci")
    if expected_source is not None:
        expected_mapping = _exact_mapping(
            expected_source,
            required={
                "test_count",
                "test_collection_sha256",
                "compiled_source_count",
                "compiled_source_inventory_sha256",
                "sealed_manifest_inventory_sha256",
            },
            name="release_requirements.expected_ci",
        )
    else:
        expected_mapping = {}
    fallback_expected = fallback.get("expected_ci", {})
    if not isinstance(fallback_expected, Mapping):
        raise ScientificReleaseAttestationError(
            "release_compatibility.expected_ci must be a JSON object"
        )

    def expected(
        name: str,
        top_level_aliases: Sequence[str],
    ) -> Any:
        candidates: list[Any] = []
        if name in expected_mapping and expected_mapping[name] is not None:
            candidates.append(expected_mapping[name])
        for alias in top_level_aliases:
            if source.get(alias) is not None:
                candidates.append(source[alias])
        if candidates:
            first = candidates[0]
            if any(item != first for item in candidates[1:]):
                raise ScientificReleaseAttestationError(
                    f"conflicting expected CI aliases for {name}"
                )
            fallback_value = fallback_expected.get(name)
            if fallback_value is not None and fallback_value != first:
                raise ScientificReleaseAttestationError(
                    f"release compatibility cannot override expected {name}"
                )
            return first
        return fallback_expected.get(name)

    normalized = {
        "remote": select("remote", ("required_remote",)),
        "branch": select("branch", ("required_branch",)),
        "tag": select("tag", ("required_annotated_tag",)),
        "workflow_file": workflow_file,
        "workflow_name": select(
            "workflow_name", ("required_workflow_name",)
        ),
        "required_job_names": select("required_job_names"),
        "expected_ci": {
            "test_count": expected(
                "test_count", ("expected_test_count", "test_count")
            ),
            "test_collection_sha256": expected(
                "test_collection_sha256",
                (
                    "expected_test_collection_sha256",
                    "test_collection_sha256",
                ),
            ),
            "compiled_source_count": expected(
                "compiled_source_count",
                (
                    "expected_compiled_source_count",
                    "compiled_source_count",
                ),
            ),
            "compiled_source_inventory_sha256": expected(
                "compiled_source_inventory_sha256",
                (
                    "expected_compiled_source_inventory_sha256",
                    "compiled_source_inventory_sha256",
                ),
            ),
            "sealed_manifest_inventory_sha256": expected(
                "sealed_manifest_inventory_sha256",
                (
                    "expected_sealed_manifest_inventory_sha256",
                    "manifest_sha256",
                ),
            ),
        },
    }
    missing = [
        name
        for name in (
            "remote",
            "branch",
            "tag",
            "workflow_file",
            "workflow_name",
            "required_job_names",
        )
        if normalized[name] is None
    ]
    missing.extend(
        f"expected_ci.{name}"
        for name, item in normalized["expected_ci"].items()
        if item is None
    )
    if missing:
        raise ScientificReleaseAttestationError(
            "static release requirements are not scientifically frozen: "
            f"{missing}"
        )
    return normalized


def sealed_manifest_inventory(
    repo_root: Path | str,
    manifest_paths: Sequence[str],
) -> tuple[tuple[dict[str, Any], ...], str]:
    """Re-hash every sealed run and return its canonical inventory digest."""

    root = Path(repo_root).resolve()
    rows: list[dict[str, Any]] = []
    for relative in manifest_paths:
        safe = _safe_relative_path(relative, "sealed_manifest_path")
        manifest = root / safe
        try:
            manifest.relative_to(root)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ScientificReleaseAttestationError(
                "sealed manifest escaped the repository"
            ) from exc
        if manifest.is_symlink() or not manifest.is_file():
            raise ScientificReleaseAttestationError(
                f"sealed manifest is missing or not regular: {safe}"
            )
        try:
            verification = verify_manifest(manifest.parent)
        except Exception as exc:
            raise ScientificReleaseAttestationError(
                f"sealed manifest verification failed: {safe}"
            ) from exc
        rows.append(
            {
                "path": safe,
                "manifest_sha256": verification.manifest_sha256,
                "artifact_count": verification.artifact_count,
            }
        )
    rows.sort(key=lambda row: row["path"])
    if [row["path"] for row in rows] != list(manifest_paths):
        raise ScientificReleaseAttestationError(
            "sealed manifest inventory input must be sorted and unique"
        )
    frozen_rows = tuple(rows)
    return frozen_rows, canonical_sha256(list(frozen_rows))


def scientific_policy_pointer_sets(
    contract: Mapping[str, Any],
) -> Mapping[str, tuple[str, ...]]:
    """Return the only policy-pointer coverage accepted for a v2 launch.

    The provider and budget sets are fixed.  Price coverage is derived from
    the complete provider-profile registry so adding or removing a profile
    necessarily changes the required pointer set.
    """

    profiles = _mapping(
        contract.get("provider_profiles"), "contract.provider_profiles"
    )
    if not profiles:
        raise ScientificReleaseAttestationError(
            "contract.provider_profiles must not be empty"
        )
    price_pointers: list[str] = []
    for profile_id in sorted(profiles):
        if (
            not isinstance(profile_id, str)
            or not profile_id
            or profile_id != profile_id.strip()
        ):
            raise ScientificReleaseAttestationError(
                "provider profile IDs must be normalized strings"
            )
        profile = _mapping(
            profiles[profile_id],
            f"contract.provider_profiles.{profile_id}",
        )
        if "price_snapshot" not in profile:
            raise ScientificReleaseAttestationError(
                f"provider profile {profile_id!r} lacks price_snapshot"
            )
        token = profile_id.replace("~", "~0").replace("/", "~1")
        price_pointers.append(
            f"/provider_profiles/{token}/price_snapshot"
        )
    return MappingProxyType(
        {
            "provider_policy": _PROVIDER_POLICY_POINTERS,
            "price_policy": tuple(price_pointers),
            "budget_policy": _BUDGET_POLICY_POINTERS,
        }
    )


def _build_policy_bindings(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    pointers_by_policy = scientific_policy_pointer_sets(contract)
    result: dict[str, Any] = {}
    for policy_name in _POLICY_NAMES:
        pointers = pointers_by_policy[policy_name]
        selected = {
            pointer: _json_copy(_resolve_json_pointer(contract, pointer))
            for pointer in pointers
        }
        result[policy_name] = {
            "pointers": list(pointers),
            "sha256": canonical_sha256(selected),
        }
    return result


def _load_contract_document(
    repo_root: Path,
    contract_path: str | Path,
) -> tuple[str, bytes, Mapping[str, Any], ScientificReleaseRequirements]:
    root = repo_root.resolve()
    candidate = Path(contract_path)
    if candidate.is_absolute():
        source = candidate
        try:
            relative = source.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ScientificReleaseAttestationError(
                "scientific contract escapes the repository"
            ) from exc
    else:
        relative = _safe_relative_path(
            candidate.as_posix(), "scientific contract path"
        )
        source = root / relative
    contract_bytes = _regular_file_bytes(
        source, root, "scientific contract"
    )
    contract = _strict_json(contract_bytes, "scientific_contract")
    if not isinstance(contract, Mapping):
        raise ScientificReleaseAttestationError(
            "scientific contract must be a JSON object"
        )
    canonical = canonical_contract_sha256(contract)
    integrity = contract.get("integrity")
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("declared_sha256") != canonical
    ):
        raise ScientificReleaseAttestationError(
            "scientific contract declared SHA-256 does not match its contents"
        )
    release = contract.get("release_requirements")
    if not isinstance(release, Mapping):
        raise ScientificReleaseAttestationError(
            "scientific contract lacks static release_requirements"
        )
    requirements = ScientificReleaseRequirements.from_mapping(release)
    return relative, contract_bytes, contract, requirements


def discover_scientific_manifest_paths(
    repo_root: Path | str,
    *,
    runner: CommandRunner | None = None,
) -> tuple[str, ...]:
    """Discover the exact tracked sealed-manifest inventory."""

    root = Path(repo_root).resolve()
    invoke_runner = runner or read_only_command_runner
    result = invoke_runner(
        (
            "git",
            "ls-files",
            "-z",
            "--",
            "artifacts/verified_replays/*/manifest.json",
            "artifacts/verified_runs/*/manifest.json",
        ),
        root,
    )
    if (
        not isinstance(result, CommandResult)
        or result.returncode != 0
        or result.stderr
    ):
        raise ScientificReleaseAttestationError(
            "tracked sealed-manifest discovery failed"
        )
    try:
        rows = tuple(
            raw.decode("utf-8", "strict")
            for raw in result.stdout.split(b"\0")
            if raw
        )
    except UnicodeDecodeError as exc:
        raise ScientificReleaseAttestationError(
            "tracked sealed-manifest paths are not strict UTF-8"
        ) from exc
    if (
        not rows
        or rows != tuple(sorted(rows))
        or len(set(rows)) != len(rows)
    ):
        raise ScientificReleaseAttestationError(
            "tracked sealed-manifest paths must be non-empty, sorted, and unique"
        )
    for index, row in enumerate(rows):
        safe = _safe_relative_path(
            row, f"tracked sealed manifest path[{index}]"
        )
        if (
            safe != row
            or not row.endswith("/manifest.json")
            or not (
                row.startswith("artifacts/verified_replays/")
                or row.startswith("artifacts/verified_runs/")
            )
        ):
            raise ScientificReleaseAttestationError(
                "tracked sealed-manifest discovery returned an invalid path"
            )
    return rows


def build_scientific_contract_binding(
    repo_root: Path | str,
    *,
    contract_path: str | Path,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Build the complete contract/policy/manifest launch binding."""

    root = Path(repo_root).resolve()
    (
        relative,
        contract_bytes,
        contract,
        requirements,
    ) = _load_contract_document(root, contract_path)
    manifests = discover_scientific_manifest_paths(root, runner=runner)
    _, inventory_sha256 = sealed_manifest_inventory(root, manifests)
    if (
        inventory_sha256
        != requirements.expected_sealed_manifest_inventory_sha256
    ):
        raise ScientificReleaseAttestationError(
            "tracked sealed manifests differ from static contract expectations"
        )
    value = {
        "contract_path": relative,
        "contract_file_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "contract_canonical_sha256": canonical_contract_sha256(contract),
        "policies": _build_policy_bindings(contract),
        "sealed_manifest_paths": list(manifests),
        "sealed_manifest_inventory_sha256": inventory_sha256,
    }
    ScientificContractBinding.from_mapping(value)
    return value


def resolve_scientific_ci_run_selection(
    repo_root: Path | str,
    *,
    release_requirements: Mapping[str, Any],
    run_id: int,
    run_attempt: int,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Resolve exact job IDs for one caller-selected run and attempt."""

    root = Path(repo_root).resolve()
    requirements = ScientificReleaseRequirements.from_mapping(
        release_requirements
    )
    selected_run = _positive_int(run_id, "run_id")
    selected_attempt = _positive_int(run_attempt, "run_attempt")
    invoke_runner = runner or read_only_command_runner
    evidence: list[CommandEvidence] = []
    head = _git_object(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "prepare_local_head",
            ("git", "rev-parse", "--verify", "HEAD^{commit}"),
        ),
        "prepare_local_head",
    )
    repository = _repository_from_origin(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "prepare_origin_url",
            ("git", "remote", "get-url", requirements.remote),
        )
    )
    run_value = _strict_json(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "prepare_github_run_attempt",
            (
                "gh",
                "api",
                "--method",
                "GET",
                (
                    f"repos/{repository}/actions/runs/{selected_run}/"
                    f"attempts/{selected_attempt}"
                ),
            ),
        ),
        "prepare_github_run_attempt",
    )
    jobs_value = _strict_json(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "prepare_github_run_jobs",
            (
                "gh",
                "api",
                "--method",
                "GET",
                (
                    f"repos/{repository}/actions/runs/{selected_run}/"
                    f"attempts/{selected_attempt}/jobs?per_page=100"
                ),
            ),
        ),
        "prepare_github_run_jobs",
    )
    jobs_payload = _mapping(jobs_value, "prepare_github_run_jobs")
    jobs_raw = jobs_payload.get("jobs")
    if (
        not isinstance(jobs_raw, list)
        or jobs_payload.get("total_count") != len(jobs_raw)
    ):
        raise ScientificReleaseAttestationError(
            "selected GitHub jobs response is incomplete or paginated"
        )
    by_name: dict[str, list[Mapping[str, Any]]] = {}
    seen_ids: set[int] = set()
    for index, item in enumerate(jobs_raw):
        row = _mapping(item, f"prepare_github_run_jobs.jobs[{index}]")
        database_id = _positive_int(
            row.get("id"),
            f"prepare_github_run_jobs.jobs[{index}].id",
        )
        if database_id in seen_ids:
            raise ScientificReleaseAttestationError(
                "selected GitHub jobs contain duplicate database IDs"
            )
        seen_ids.add(database_id)
        name = _text(
            row.get("name"),
            f"prepare_github_run_jobs.jobs[{index}].name",
        )
        by_name.setdefault(name, []).append(row)
    selected_jobs: list[RequiredScientificCIJob] = []
    for name in requirements.required_job_names:
        matches = by_name.get(name, [])
        if len(matches) != 1:
            raise ScientificReleaseAttestationError(
                f"selected run must contain exactly one required job {name!r}"
            )
        selected_jobs.append(
            RequiredScientificCIJob(
                name=name,
                database_id=int(matches[0]["id"]),
            )
        )
    selection = ScientificCIRunSelection(
        run_id=selected_run,
        run_attempt=selected_attempt,
        jobs=(selected_jobs[0], selected_jobs[1]),
    )
    _parse_github_run(
        run_value,
        repository=repository,
        requirements=requirements,
        selection=selection,
        head=head,
    )
    _parse_github_jobs(
        jobs_value,
        repository=repository,
        requirements=requirements,
        selection=selection,
    )
    return selection.to_dict()


def build_scientific_launch_input(
    repo_root: Path | str,
    *,
    contract_path: str | Path,
    run_id: int,
    run_attempt: int,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Build a complete self-hashed launch input without writing it."""

    root = Path(repo_root).resolve()
    _, _, contract, requirements = _load_contract_document(
        root, contract_path
    )
    binding = build_scientific_contract_binding(
        root, contract_path=contract_path, runner=runner
    )
    selection = resolve_scientific_ci_run_selection(
        root,
        release_requirements=requirements.to_dict(),
        run_id=run_id,
        run_attempt=run_attempt,
        runner=runner,
    )
    payload = {
        "schema_version": SCIENTIFIC_LAUNCH_INPUT_SCHEMA_VERSION,
        "contract_sha256": canonical_contract_sha256(contract),
        "ci_run_selection": selection,
        "contract_binding": binding,
    }
    return {
        **payload,
        "launch_input_sha256": canonical_sha256(payload),
    }


def _validate_scientific_launch_input(
    value: Any,
    *,
    contract_sha256: str,
    contract_path: str,
) -> dict[str, Any]:
    payload = _exact_mapping(
        value,
        required={
            "schema_version",
            "contract_sha256",
            "ci_run_selection",
            "contract_binding",
            "launch_input_sha256",
        },
        name="scientific_launch_input",
    )
    if (
        payload["schema_version"]
        != SCIENTIFIC_LAUNCH_INPUT_SCHEMA_VERSION
    ):
        raise ScientificReleaseAttestationError(
            "scientific launch input has an unsupported schema"
        )
    if payload["contract_sha256"] != contract_sha256:
        raise ScientificReleaseAttestationError(
            "scientific launch input contract SHA-256 mismatch"
        )
    unsigned = _json_copy(payload)
    claimed = unsigned.pop("launch_input_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ScientificReleaseAttestationError(
            "scientific launch input self-hash mismatch"
        )
    selection = ScientificCIRunSelection.from_mapping(
        _mapping(payload["ci_run_selection"], "ci_run_selection")
    )
    binding = ScientificContractBinding.from_mapping(
        _mapping(payload["contract_binding"], "contract_binding")
    )
    if (
        binding.contract_path != contract_path
        or binding.contract_canonical_sha256 != contract_sha256
    ):
        raise ScientificReleaseAttestationError(
            "scientific launch input binds a different contract"
        )
    return {
        "schema_version": SCIENTIFIC_LAUNCH_INPUT_SCHEMA_VERSION,
        "contract_sha256": contract_sha256,
        "ci_run_selection": selection.to_dict(),
        "contract_binding": binding.to_dict(),
        "launch_input_sha256": claimed,
    }


def verify_scientific_release_attestation(
    repo_root: Path | str,
    *,
    release_requirements: Mapping[str, Any],
    ci_run_selection: Mapping[str, Any],
    contract_binding: Mapping[str, Any],
    release_compatibility: Mapping[str, Any] | None = None,
    runner: CommandRunner | None = None,
) -> ScientificReleaseAttestation:
    """Verify an exact scientific release and return its self-hashed receipt.

    Production uses the non-interactive read-only runner; tests can inject
    deterministic command results without accessing GitHub.
    """

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise ScientificReleaseAttestationError(
            "repository root does not exist or is not a directory"
        )
    requirements = ScientificReleaseRequirements.from_mapping(
        release_requirements, compatibility=release_compatibility
    )
    selection = ScientificCIRunSelection.from_mapping(ci_run_selection)
    if tuple(job.name for job in selection.jobs) != (
        requirements.required_job_names
    ):
        raise ScientificReleaseAttestationError(
            "selected CI job names differ from static contract requirements"
        )
    binding = ScientificContractBinding.from_mapping(contract_binding)
    if (
        binding.sealed_manifest_inventory_sha256
        != requirements.expected_sealed_manifest_inventory_sha256
    ):
        raise ScientificReleaseAttestationError(
            "contract binding and static release requirements disagree on "
            "sealed manifests"
        )
    invoke_runner = runner or read_only_command_runner
    evidence: list[CommandEvidence] = []

    head = _git_object(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "local_head",
            ("git", "rev-parse", "--verify", "HEAD^{commit}"),
        ),
        "local_head",
    )
    if (
        _invoke(
            invoke_runner,
            root,
            evidence,
            "local_status",
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        )
        != b""
    ):
        raise ScientificReleaseAttestationError(
            "local HEAD is not clean, including untracked files"
        )

    tag_object = _git_object(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "local_tag_object",
            (
                "git",
                "rev-parse",
                "--verify",
                f"refs/tags/{requirements.tag}^{{tag}}",
            ),
        ),
        "local_tag_object",
    )
    tag_commit = _git_object(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "local_tag_commit",
            (
                "git",
                "rev-parse",
                "--verify",
                f"refs/tags/{requirements.tag}^{{commit}}",
            ),
        ),
        "local_tag_commit",
    )
    if tag_commit != head or tag_object == head:
        raise ScientificReleaseAttestationError(
            "local annotated release tag does not peel exactly to HEAD"
        )

    workflow_path = root / requirements.workflow_file
    workflow_bytes = _regular_file_bytes(
        workflow_path, root, "workflow file"
    )
    workflow_sha256 = hashlib.sha256(workflow_bytes).hexdigest()
    workflow_blob_oid = _git_object(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "workflow_blob",
            (
                "git",
                "rev-parse",
                "--verify",
                f"HEAD:{requirements.workflow_file}",
            ),
        ),
        "workflow_blob",
    )

    contract_path = root / binding.contract_path
    contract_bytes = _regular_file_bytes(
        contract_path, root, "scientific contract"
    )
    contract_file_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    if contract_file_sha256 != binding.contract_file_sha256:
        raise ScientificReleaseAttestationError(
            "scientific contract file SHA-256 does not match its binding"
        )
    contract_blob_oid = _git_object(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "contract_blob",
            (
                "git",
                "rev-parse",
                "--verify",
                f"HEAD:{binding.contract_path}",
            ),
        ),
        "contract_blob",
    )
    contract = _strict_json(contract_bytes, "scientific_contract")
    if not isinstance(contract, Mapping):
        raise ScientificReleaseAttestationError(
            "scientific contract must be a JSON object"
        )
    contract_canonical = canonical_contract_sha256(contract)
    if contract_canonical != binding.contract_canonical_sha256:
        raise ScientificReleaseAttestationError(
            "scientific contract canonical SHA-256 does not match its binding"
        )
    integrity = contract.get("integrity")
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("declared_sha256") != contract_canonical
    ):
        raise ScientificReleaseAttestationError(
            "scientific contract declared SHA-256 does not match its contents"
        )
    embedded_release = contract.get("release_requirements")
    if not isinstance(embedded_release, Mapping):
        raise ScientificReleaseAttestationError(
            "scientific contract lacks static release_requirements"
        )
    embedded_requirements = ScientificReleaseRequirements.from_mapping(
        embedded_release, compatibility=release_compatibility
    )
    if embedded_requirements != requirements:
        raise ScientificReleaseAttestationError(
            "supplied release requirements differ from the scientific contract"
        )

    required_policy_pointers = scientific_policy_pointer_sets(contract)
    policy_rows: dict[str, Any] = {}
    for policy_name in _POLICY_NAMES:
        policy = binding.policies[policy_name]
        if policy.pointers != required_policy_pointers[policy_name]:
            raise ScientificReleaseAttestationError(
                f"{policy_name} pointers do not match the mandatory "
                "scientific coverage set"
            )
        selected = {
            pointer: _json_copy(_resolve_json_pointer(contract, pointer))
            for pointer in policy.pointers
        }
        observed = canonical_sha256(selected)
        if observed != policy.sha256:
            raise ScientificReleaseAttestationError(
                f"{policy_name} SHA-256 does not match its binding"
            )
        policy_rows[f"{policy_name}_sha256"] = observed
        policy_rows[f"{policy_name}_pointers"] = list(policy.pointers)

    manifest_rows, inventory_sha256 = sealed_manifest_inventory(
        root, binding.sealed_manifest_paths
    )
    if inventory_sha256 != binding.sealed_manifest_inventory_sha256:
        raise ScientificReleaseAttestationError(
            "sealed-manifest inventory SHA-256 does not match its binding"
        )

    origin_repository = _repository_from_origin(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "origin_url",
            ("git", "remote", "get-url", requirements.remote),
        )
    )
    remote_refs = _parse_remote_refs(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "remote_refs",
            (
                "git",
                "ls-remote",
                "--exit-code",
                requirements.remote,
                f"refs/heads/{requirements.branch}",
                f"refs/tags/{requirements.tag}",
                f"refs/tags/{requirements.tag}^{{}}",
            ),
        ),
        requirements,
    )
    if (
        remote_refs["branch_commit"] != head
        or remote_refs["tag_object"] != tag_object
        or remote_refs["tag_commit"] != head
    ):
        raise ScientificReleaseAttestationError(
            "remote branch/tag does not match local HEAD/annotated tag"
        )

    run = _parse_github_run(
        _strict_json(
            _invoke(
                invoke_runner,
                root,
                evidence,
                "github_run_attempt",
                (
                    "gh",
                    "api",
                    "--method",
                    "GET",
                    (
                        f"repos/{origin_repository}/actions/runs/"
                        f"{selection.run_id}/attempts/"
                        f"{selection.run_attempt}"
                    ),
                ),
            ),
            "github_run_attempt",
        ),
        repository=origin_repository,
        requirements=requirements,
        selection=selection,
        head=head,
    )
    jobs = _parse_github_jobs(
        _strict_json(
            _invoke(
                invoke_runner,
                root,
                evidence,
                "github_run_jobs",
                (
                    "gh",
                    "api",
                    "--method",
                    "GET",
                    (
                        f"repos/{origin_repository}/actions/runs/"
                        f"{selection.run_id}/attempts/"
                        f"{selection.run_attempt}/jobs?per_page=100"
                    ),
                ),
            ),
            "github_run_jobs",
        ),
        repository=origin_repository,
        requirements=requirements,
        selection=selection,
    )

    job_receipts: list[dict[str, Any]] = []
    for required_job in selection.jobs:
        receipt = _parse_ci_job_log(
            _invoke(
                invoke_runner,
                root,
                evidence,
                f"github_job_log_{required_job.database_id}",
                (
                    "gh",
                    "run",
                    "view",
                    str(selection.run_id),
                    "--repo",
                    origin_repository,
                    "--attempt",
                    str(selection.run_attempt),
                    "--job",
                    str(required_job.database_id),
                    "--log",
                ),
            ),
            required_job=required_job,
            requirements=requirements,
            selection=selection,
            repository=origin_repository,
            head=head,
            workflow_sha256=workflow_sha256,
            workflow_blob_oid=workflow_blob_oid,
            inventory_sha256=inventory_sha256,
            inventory_count=len(manifest_rows),
        )
        job_receipts.append(receipt)
    _require_equal_ci_measurements(job_receipts)

    payload = {
        "schema_version": SCIENTIFIC_RELEASE_ATTESTATION_SCHEMA_VERSION,
        "status": "pass",
        "release_requirements": requirements.to_dict(),
        "ci_run_selection": selection.to_dict(),
        "head_commit": head,
        "local_tag": {
            "name": requirements.tag,
            "object_id": tag_object,
            "peeled_commit": tag_commit,
            "kind": "annotated",
        },
        "remote": {
            "name": requirements.remote,
            "branch": requirements.branch,
            "branch_commit": remote_refs["branch_commit"],
            "tag_name": requirements.tag,
            "tag_object_id": remote_refs["tag_object"],
            "tag_peeled_commit": remote_refs["tag_commit"],
            "tag_kind": "annotated",
        },
        "workflow": {
            "file": requirements.workflow_file,
            "file_sha256": workflow_sha256,
            "blob_oid": workflow_blob_oid,
            "name": requirements.workflow_name,
        },
        "github_actions": {
            "repository": origin_repository,
            "run": run,
            "jobs": jobs,
            "ci_job_receipts": job_receipts,
            "ci_measurements": {
                key: job_receipts[0][key]
                for key in (
                    "test_count",
                    "test_collection_sha256",
                    "compiled_source_count",
                    "compiled_source_inventory_sha256",
                )
            },
        },
        "contract": {
            "path": binding.contract_path,
            "blob_oid": contract_blob_oid,
            "file_sha256": contract_file_sha256,
            "canonical_sha256": contract_canonical,
            **policy_rows,
        },
        "sealed_manifest_inventory": {
            "inventory_sha256": inventory_sha256,
            "manifest_count": len(manifest_rows),
            "manifests": list(manifest_rows),
        },
        "command_evidence": [row.to_dict() for row in evidence],
    }
    return ScientificReleaseAttestation(payload=payload)


def dry_verify_scientific_launch_input(
    repo_root: Path | str,
    *,
    contract_path: str | Path,
    launch_input: Mapping[str, Any] | str | Path,
    runner: CommandRunner | None = None,
) -> ScientificReleaseAttestation:
    """Verify a launch input and its release without dispatching providers."""

    root = Path(repo_root).resolve()
    relative, _, contract, requirements = _load_contract_document(
        root, contract_path
    )
    if isinstance(launch_input, Mapping):
        source_value: Any = launch_input
    else:
        source = Path(launch_input)
        if source.is_symlink() or not source.is_file():
            raise ScientificReleaseAttestationError(
                "scientific launch input is missing or not regular"
            )
        source_value = _strict_json(
            source.read_bytes(), "scientific_launch_input"
        )
    contract_sha256 = canonical_contract_sha256(contract)
    validated = _validate_scientific_launch_input(
        source_value,
        contract_sha256=contract_sha256,
        contract_path=relative,
    )
    return verify_scientific_release_attestation(
        root,
        release_requirements=requirements.to_dict(),
        ci_run_selection=validated["ci_run_selection"],
        contract_binding=validated["contract_binding"],
        runner=runner,
    )


def _default_launch_output(
    repo_root: Path,
    contract: Mapping[str, Any],
) -> Path:
    contract_id = _text(contract.get("contract_id"), "contract.contract_id")
    namespace = (
        contract_id[len("finevo-") :]
        if contract_id.startswith("finevo-")
        else contract_id
    )
    if (
        not namespace
        or "/" in namespace
        or "\\" in namespace
        or ".." in namespace
    ):
        raise ScientificReleaseAttestationError(
            "contract_id cannot identify an ignored raw namespace"
        )
    return (
        repo_root
        / "experiment_results"
        / namespace
        / "raw"
        / "scientific_launch_input.json"
    )


def _require_ignored_output(
    repo_root: Path,
    output_path: Path,
    *,
    runner: CommandRunner,
) -> str:
    root = repo_root.resolve()
    output = output_path if output_path.is_absolute() else root / output_path
    try:
        relative = output.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ScientificReleaseAttestationError(
            "scientific launch output must remain inside the repository"
        ) from exc
    if output.is_symlink():
        raise ScientificReleaseAttestationError(
            "scientific launch output must not be a symlink"
        )
    result = runner(
        (
            "git",
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            relative,
        ),
        root,
    )
    if (
        not isinstance(result, CommandResult)
        or result.returncode != 0
        or result.stderr
    ):
        raise ScientificReleaseAttestationError(
            "scientific launch output is not covered by a Git ignore rule"
        )
    return relative


def _write_launch_input_no_overwrite(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    encoded = (
        json.dumps(
            _json_copy(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ScientificReleaseAttestationError(
                "scientific launch output exists but is not a regular file"
            )
        if path.read_bytes() != encoded:
            raise ScientificReleaseAttestationError(
                "scientific launch output already exists with different bytes"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ScientificReleaseAttestationError(
            "scientific launch output appeared concurrently"
        ) from exc
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if path.exists() and path.is_file():
            path.unlink()
        raise


def prepare_scientific_launch_input(
    repo_root: Path | str,
    *,
    contract_path: str | Path,
    run_id: int,
    run_attempt: int,
    output_path: str | Path | None = None,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Resolve, dry-verify, and persist one immutable ignored launch input."""

    root = Path(repo_root).resolve()
    _, _, contract, _ = _load_contract_document(root, contract_path)
    invoke_runner = runner or read_only_command_runner
    launch = build_scientific_launch_input(
        root,
        contract_path=contract_path,
        run_id=run_id,
        run_attempt=run_attempt,
        runner=invoke_runner,
    )
    attestation = dry_verify_scientific_launch_input(
        root,
        contract_path=contract_path,
        launch_input=launch,
        runner=invoke_runner,
    )
    output = (
        _default_launch_output(root, contract)
        if output_path is None
        else Path(output_path)
    )
    relative_output = _require_ignored_output(
        root, output, runner=invoke_runner
    )
    destination = root / relative_output
    _write_launch_input_no_overwrite(destination, launch)
    return {
        "schema_version": "finevo-scientific-launch-preparation-v1",
        "status": "pass",
        "provider_calls": 0,
        "output": relative_output,
        "contract_sha256": launch["contract_sha256"],
        "launch_input_sha256": launch["launch_input_sha256"],
        "ci_run_selection": launch["ci_run_selection"],
        "dry_verification_attestation_sha256": (
            attestation.attestation_sha256
        ),
    }


def _parse_ci_job_log(
    raw: bytes,
    *,
    required_job: RequiredScientificCIJob,
    requirements: ScientificReleaseRequirements,
    selection: ScientificCIRunSelection,
    repository: str,
    head: str,
    workflow_sha256: str,
    workflow_blob_oid: str,
    inventory_sha256: str,
    inventory_count: int,
) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise ScientificReleaseAttestationError(
            "CI job log is not strict UTF-8"
        ) from exc
    candidates: list[str] = []
    for line in text.splitlines():
        marker = line.find(CI_JOB_RECEIPT_LOG_PREFIX)
        if marker >= 0:
            candidates.append(
                line[marker + len(CI_JOB_RECEIPT_LOG_PREFIX) :].strip()
            )
    if len(candidates) != 1:
        raise ScientificReleaseAttestationError(
            "required CI job must emit exactly one release receipt log line"
        )
    receipt = _strict_json(candidates[0].encode(), "ci_job_receipt")
    if not isinstance(receipt, Mapping):
        raise ScientificReleaseAttestationError(
            "CI job receipt must be a JSON object"
        )
    if set(receipt) != _CI_JOB_RECEIPT_KEYS:
        raise ScientificReleaseAttestationError(
            "CI job receipt keys do not match the sealed schema"
        )
    receipt_copy = _json_copy(receipt)
    observed_hash = receipt_copy.pop("receipt_sha256", None)
    if (
        not isinstance(observed_hash, str)
        or observed_hash != canonical_sha256(receipt_copy)
    ):
        raise ScientificReleaseAttestationError(
            "CI job receipt self-hash mismatch"
        )
    expected_exact = {
        "schema_version": CI_JOB_RECEIPT_SCHEMA_VERSION,
        "status": "pass",
        "repository": repository,
        "head_sha": head,
        "run_id": selection.run_id,
        "run_attempt": selection.run_attempt,
        "job_name": required_job.name,
        "workflow_name": requirements.workflow_name,
        "workflow_file": requirements.workflow_file,
        "workflow_source_sha": head,
        "workflow_file_sha256": workflow_sha256,
        "workflow_blob_oid": workflow_blob_oid,
        "sealed_manifest_count": inventory_count,
        "sealed_manifest_inventory_sha256": inventory_sha256,
    }
    for key, expected in expected_exact.items():
        if receipt.get(key) != expected:
            raise ScientificReleaseAttestationError(
                f"CI job receipt {key} does not match the release"
            )
    expected_workflow_ref = (
        f"{repository}/{requirements.workflow_file}"
        f"@refs/heads/{requirements.branch}"
    )
    if receipt.get("workflow_ref") != expected_workflow_ref:
        raise ScientificReleaseAttestationError(
            "CI job receipt workflow_ref does not match the release branch"
        )
    for key in (
        "test_count",
        "compiled_source_count",
        "sealed_manifest_count",
    ):
        _positive_int(receipt.get(key), f"ci_job_receipt.{key}")
    skipped = receipt.get("skipped_test_count")
    if isinstance(skipped, bool) or not isinstance(skipped, int) or skipped < 0:
        raise ScientificReleaseAttestationError(
            "ci_job_receipt.skipped_test_count must be non-negative"
        )
    _text(receipt.get("job_key"), "ci_job_receipt.job_key")
    _text(receipt.get("runner_os"), "ci_job_receipt.runner_os")
    for key in (
        "test_collection_sha256",
        "compiled_source_inventory_sha256",
    ):
        _sha256(receipt.get(key), f"ci_job_receipt.{key}")
    expected_measurements = {
        "test_count": requirements.expected_test_count,
        "test_collection_sha256": (
            requirements.expected_test_collection_sha256
        ),
        "compiled_source_count": (
            requirements.expected_compiled_source_count
        ),
        "compiled_source_inventory_sha256": (
            requirements.expected_compiled_source_inventory_sha256
        ),
    }
    for key, expected in expected_measurements.items():
        if receipt.get(key) != expected:
            raise ScientificReleaseAttestationError(
                f"CI job receipt {key} differs from the static contract"
            )
    return receipt_copy | {"receipt_sha256": observed_hash}


def _require_equal_ci_measurements(receipts: Sequence[Mapping[str, Any]]) -> None:
    if len(receipts) != 2:
        raise ScientificReleaseAttestationError(
            "exactly two CI receipts are required"
        )
    fields = (
        "test_count",
        "test_collection_sha256",
        "compiled_source_count",
        "compiled_source_inventory_sha256",
        "skipped_test_count",
        "sealed_manifest_inventory_sha256",
    )
    if any(receipts[0].get(field) != receipts[1].get(field) for field in fields):
        raise ScientificReleaseAttestationError(
            "required CI jobs disagree on tests, sources, or sealed manifests"
        )
    if {receipt.get("runner_os") for receipt in receipts} != {
        "Linux",
        "macOS",
    }:
        raise ScientificReleaseAttestationError(
            "scientific release requires one Linux and one macOS CI receipt"
        )


def _parse_github_run(
    value: Any,
    *,
    repository: str,
    requirements: ScientificReleaseRequirements,
    selection: ScientificCIRunSelection,
    head: str,
) -> dict[str, Any]:
    row = _mapping(value, "github_run_attempt")
    expected = {
        "id": selection.run_id,
        "run_attempt": selection.run_attempt,
        "head_sha": head,
        "head_branch": requirements.branch,
        "status": "completed",
        "conclusion": "success",
        "event": "push",
        "name": requirements.workflow_name,
        "path": requirements.workflow_file,
    }
    for key, wanted in expected.items():
        if row.get(key) != wanted:
            raise ScientificReleaseAttestationError(
                f"GitHub workflow run {key} does not match requirements"
            )
    html_url = _github_url(row.get("html_url"), "github_run_attempt.html_url")
    expected_path = f"/{repository}/actions/runs/{selection.run_id}"
    if urlparse(html_url).path.rstrip("/") != expected_path:
        raise ScientificReleaseAttestationError(
            "GitHub workflow run URL does not match repository/run ID"
        )
    return {
        "database_id": selection.run_id,
        "attempt": selection.run_attempt,
        "head_sha": head,
        "head_branch": requirements.branch,
        "status": "completed",
        "conclusion": "success",
        "event": _text(row.get("event"), "github_run_attempt.event"),
        "workflow_name": requirements.workflow_name,
        "workflow_file": requirements.workflow_file,
        "url": html_url,
    }


def _parse_github_jobs(
    value: Any,
    *,
    repository: str,
    requirements: ScientificReleaseRequirements,
    selection: ScientificCIRunSelection,
) -> list[dict[str, Any]]:
    payload = _mapping(value, "github_run_jobs")
    jobs_raw = payload.get("jobs")
    if not isinstance(jobs_raw, list):
        raise ScientificReleaseAttestationError(
            "github_run_jobs.jobs must be an array"
        )
    if payload.get("total_count") != len(jobs_raw):
        raise ScientificReleaseAttestationError(
            "GitHub jobs response is incomplete or paginated"
        )
    indexed: dict[int, Mapping[str, Any]] = {}
    for index, raw in enumerate(jobs_raw):
        row = _mapping(raw, f"github_run_jobs.jobs[{index}]")
        database_id = _positive_int(
            row.get("id"), f"github_run_jobs.jobs[{index}].id"
        )
        if database_id in indexed:
            raise ScientificReleaseAttestationError(
                "GitHub jobs response contains duplicate IDs"
            )
        indexed[database_id] = row
    result: list[dict[str, Any]] = []
    for required in selection.jobs:
        row = indexed.get(required.database_id)
        if row is None:
            raise ScientificReleaseAttestationError(
                f"required CI job ID {required.database_id} is missing"
            )
        expected = {
            "name": required.name,
            "status": "completed",
            "conclusion": "success",
            "run_attempt": selection.run_attempt,
        }
        for key, wanted in expected.items():
            if row.get(key) != wanted:
                raise ScientificReleaseAttestationError(
                    f"required CI job {required.database_id} {key} mismatches"
                )
        html_url = _github_url(
            row.get("html_url"),
            f"github_run_jobs.jobs[{required.database_id}].html_url",
        )
        expected_path = (
            f"/{repository}/actions/runs/{selection.run_id}/job/"
            f"{required.database_id}"
        )
        if urlparse(html_url).path.rstrip("/") != expected_path:
            raise ScientificReleaseAttestationError(
                "required CI job URL is not bound to the run"
            )
        result.append(
            {
                "name": required.name,
                "database_id": required.database_id,
                "attempt": selection.run_attempt,
                "status": "completed",
                "conclusion": "success",
                "url": html_url,
            }
        )
    return result


def _parse_remote_refs(
    raw: bytes,
    requirements: ScientificReleaseRequirements,
) -> dict[str, str]:
    expected = {
        f"refs/heads/{requirements.branch}": "branch_commit",
        f"refs/tags/{requirements.tag}": "tag_object",
        f"refs/tags/{requirements.tag}^{{}}": "tag_commit",
    }
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise ScientificReleaseAttestationError(
            "remote refs are not strict UTF-8"
        ) from exc
    result: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            raise ScientificReleaseAttestationError(
                "remote refs contain a malformed line"
            )
        object_id, ref = parts
        if (
            not _GIT_OBJECT_RE.fullmatch(object_id)
            or ref not in expected
            or expected[ref] in result
        ):
            raise ScientificReleaseAttestationError(
                "remote refs contain an unexpected or duplicate ref"
            )
        result[expected[ref]] = object_id
    if set(result) != set(expected.values()):
        raise ScientificReleaseAttestationError(
            "remote refs are missing branch, annotated tag, or peeled tag"
        )
    return result


def _resolve_json_pointer(value: Any, pointer: str) -> Any:
    current = value
    if pointer == "":
        return current
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if part not in current:
                raise ScientificReleaseAttestationError(
                    f"contract policy pointer does not exist: {pointer}"
                )
            current = current[part]
        elif isinstance(current, list):
            if not part.isdigit() or int(part) >= len(current):
                raise ScientificReleaseAttestationError(
                    f"contract policy pointer does not exist: {pointer}"
                )
            current = current[int(part)]
        else:
            raise ScientificReleaseAttestationError(
                f"contract policy pointer does not exist: {pointer}"
            )
    return current


def _invoke(
    runner: CommandRunner,
    root: Path,
    evidence: list[CommandEvidence],
    evidence_id: str,
    argv: Sequence[str],
) -> bytes:
    result = runner(tuple(argv), root)
    if not isinstance(result, CommandResult):
        raise ScientificReleaseAttestationError(
            f"{evidence_id} runner returned an invalid result type"
        )
    evidence.append(_command_evidence(evidence_id, argv, result))
    if result.returncode != 0:
        raise ScientificReleaseAttestationError(
            f"{evidence_id} read-only command failed"
        )
    if result.stderr:
        raise ScientificReleaseAttestationError(
            f"{evidence_id} produced unexpected stderr"
        )
    return result.stdout


def _command_evidence(
    evidence_id: str,
    argv: Sequence[str],
    result: CommandResult,
) -> CommandEvidence:
    stdout_hash = hashlib.sha256(result.stdout).hexdigest()
    stderr_hash = hashlib.sha256(result.stderr).hexdigest()
    return CommandEvidence(
        evidence_id=evidence_id,
        argv=tuple(str(item) for item in argv),
        returncode=result.returncode,
        stdout_bytes=len(result.stdout),
        stderr_bytes=len(result.stderr),
        stdout_sha256=stdout_hash,
        stderr_sha256=stderr_hash,
        combined_sha256=hashlib.sha256(
            result.stdout + b"\0" + result.stderr
        ).hexdigest(),
    )


def _strict_json(raw: bytes, name: str) -> Any:
    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ScientificReleaseAttestationError(
                    f"{name} contains duplicate JSON keys"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ScientificReleaseAttestationError(
                    f"{name} contains non-finite JSON value {constant}"
                )
            ),
        )
    except ScientificReleaseAttestationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScientificReleaseAttestationError(
            f"{name} is not strict UTF-8 JSON"
        ) from exc


def _repository_from_origin(raw: bytes) -> str:
    value = _single_line(raw, "origin_url")
    repository: str | None = None
    if value.startswith("git@github.com:"):
        repository = value.removeprefix("git@github.com:")
    elif value.startswith("ssh://git@github.com/"):
        repository = value.removeprefix("ssh://git@github.com/")
    else:
        parsed = urlparse(value)
        if (
            parsed.scheme == "https"
            and parsed.hostname == "github.com"
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        ):
            repository = parsed.path.lstrip("/")
    if repository and repository.endswith(".git"):
        repository = repository[:-4]
    if repository is None or not _REPOSITORY_RE.fullmatch(repository):
        raise ScientificReleaseAttestationError(
            "origin must identify one credential-free GitHub repository"
        )
    return repository


def _regular_file_bytes(path: Path, root: Path, name: str) -> bytes:
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ScientificReleaseAttestationError(
            f"{name} escapes the repository"
        ) from exc
    if path.is_symlink() or not path.is_file():
        raise ScientificReleaseAttestationError(
            f"{name} is missing or not a regular file"
        )
    return path.read_bytes()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScientificReleaseAttestationError(
            "value is not strict canonical JSON"
        ) from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list) or isinstance(value, tuple):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ScientificReleaseAttestationError(
        "attestation contains a non-JSON-compatible value"
    )


def _json_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_copy(item) for item in value]
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScientificReleaseAttestationError(
            f"{name} must be a JSON object"
        )
    if any(not isinstance(key, str) for key in value):
        raise ScientificReleaseAttestationError(
            f"{name} keys must be strings"
        )
    return value


def _exact_mapping(
    value: Any, *, required: set[str], name: str
) -> Mapping[str, Any]:
    mapping = _mapping(value, name)
    if set(mapping) != required:
        raise ScientificReleaseAttestationError(
            f"{name} keys must be exactly {sorted(required)}"
        )
    return mapping


def _text(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise ScientificReleaseAttestationError(
            f"{name} must be normalized non-empty text"
        )
    return value


def _ref_text(value: Any, name: str) -> str:
    text = _text(value, name)
    if (
        not _REF_COMPONENT_RE.fullmatch(text)
        or ".." in text
        or text.endswith("/")
        or text.endswith(".lock")
        or "@{" in text
    ):
        raise ScientificReleaseAttestationError(
            f"{name} is not a safe Git ref component"
        )
    return text


def _safe_relative_path(value: Any, name: str) -> str:
    text = _text(value, name)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or str(path) != text
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ScientificReleaseAttestationError(
            f"{name} must be a normalized repository-relative path"
        )
    return text


def _json_pointer(value: Any, name: str) -> str:
    text = _text(value, name)
    if not text.startswith("/"):
        raise ScientificReleaseAttestationError(
            f"{name} must be a non-root JSON pointer"
        )
    for part in text[1:].split("/"):
        if re.search(r"~(?:[^01]|$)", part):
            raise ScientificReleaseAttestationError(
                f"{name} contains an invalid JSON pointer escape"
            )
    return text


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ScientificReleaseAttestationError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _git_object(raw: bytes, name: str) -> str:
    value = _single_line(raw, name)
    if not _GIT_OBJECT_RE.fullmatch(value):
        raise ScientificReleaseAttestationError(
            f"{name} did not return one lowercase Git object ID"
        )
    return value


def _single_line(raw: bytes, name: str) -> str:
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise ScientificReleaseAttestationError(
            f"{name} is not strict UTF-8"
        ) from exc
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0] or lines[0] != lines[0].strip():
        raise ScientificReleaseAttestationError(
            f"{name} did not return exactly one normalized line"
        )
    return lines[0]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ScientificReleaseAttestationError(
            f"{name} must be a positive integer"
        )
    return int(value)


def _github_url(value: Any, name: str) -> str:
    url = _text(value, name)
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ScientificReleaseAttestationError(
            f"{name} must be a credential-free github.com HTTPS URL"
        )
    return url


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or dry-verify the exact ignored scientific launch input"
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-scientific-launch")
    prepare.add_argument("--contract", type=Path, required=True)
    prepare.add_argument("--run-id", type=int, required=True)
    prepare.add_argument("--run-attempt", type=int, required=True)
    prepare.add_argument("--output", type=Path, default=None)
    verify = subparsers.add_parser("verify-scientific-launch")
    verify.add_argument("--contract", type=Path, required=True)
    verify.add_argument("--input", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "prepare-scientific-launch":
            result = prepare_scientific_launch_input(
                args.repo_root,
                contract_path=args.contract,
                run_id=args.run_id,
                run_attempt=args.run_attempt,
                output_path=args.output,
            )
        else:
            attestation = dry_verify_scientific_launch_input(
                args.repo_root,
                contract_path=args.contract,
                launch_input=args.input,
            )
            result = {
                "schema_version": "finevo-scientific-launch-verification-v1",
                "status": "pass",
                "provider_calls": 0,
                "head_commit": attestation.payload["head_commit"],
                "attestation_sha256": attestation.attestation_sha256,
            }
    except ScientificReleaseAttestationError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


__all__ = [
    "CI_JOB_RECEIPT_LOG_PREFIX",
    "CI_JOB_RECEIPT_SCHEMA_VERSION",
    "SCIENTIFIC_LAUNCH_INPUT_SCHEMA_VERSION",
    "SCIENTIFIC_RELEASE_ATTESTATION_SCHEMA_VERSION",
    "PolicyBinding",
    "RequiredScientificCIJob",
    "ScientificCIRunSelection",
    "ScientificContractBinding",
    "ScientificReleaseAttestation",
    "ScientificReleaseAttestationError",
    "ScientificReleaseRequirements",
    "adapt_contract_release_requirements",
    "build_scientific_contract_binding",
    "build_scientific_launch_input",
    "canonical_contract_sha256",
    "canonical_sha256",
    "discover_scientific_manifest_paths",
    "dry_verify_scientific_launch_input",
    "prepare_scientific_launch_input",
    "resolve_scientific_ci_run_selection",
    "sealed_manifest_inventory",
    "read_only_command_runner",
    "scientific_policy_pointer_sets",
    "verify_scientific_release_attestation",
]


if __name__ == "__main__":  # pragma: no cover - exercised by operator CLI
    raise SystemExit(main())
