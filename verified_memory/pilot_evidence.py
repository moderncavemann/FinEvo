"""Strict current-method evidence packaging for the FinEvo micro-pilot.

The publisher is intentionally downstream of execution.  It does not call a
provider, repair an incomplete run, select seeds, or infer favourable values.
It accepts only:

* a frozen :class:`~verified_memory.pilot_contract.PilotContract`;
* the pilot ITT ledger; and
* either a standard sealed verified-run directory or a
  ``finevo-pilot-terminal-summary-v1`` envelope for each completed cell.

The resulting package is written under ``current_v2/pilot-v1`` in a temporary
directory and atomically installed only after validation.  Missing and stopped
cells remain visible in the denominator.  Corrupt, historical, unsealed, or
diagnostic artifacts presented as scientific evidence fail closed.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
from statistics import mean, median
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import ManifestVerificationError, verify_manifest
from .pilot_analysis import (
    continuation_effect_gate,
    paired_delta_summary,
    retrieval_effect_gate,
    summarize_run,
)
from .pilot_contract import (
    PilotContract,
    PilotRunSpec,
    canonical_sha256,
    load_pilot_contract,
)
from .runner import RUNNER_SCHEMA_VERSION
from .runner_artifacts import (
    LEGACY_RUNNER_SCHEMA_VERSION,
    PREVIOUS_RUNNER_SCHEMA_VERSION,
    load_verified_run_artifacts,
)


PILOT_EVIDENCE_SCHEMA_VERSION = "finevo-pilot-evidence-package-v1"
PILOT_TERMINAL_SUMMARY_SCHEMA_VERSION = "finevo-pilot-terminal-summary-v1"
PILOT_FAILURE_LEDGER_SCHEMA_VERSION = "finevo-pilot-failure-ledger-v1"
PILOT_CHECKSUM_SCHEMA_VERSION = "finevo-pilot-package-checksums-v1"
PILOT_RUN_LEDGER_SCHEMA_VERSION = "finevo-pilot-run-ledger-v1"
PILOT_BUDGET_SCHEMA_VERSION = "finevo-pilot-budget-ledger-v1"
PILOT_RELEASE_ATTESTATION_SCHEMA_VERSION = (
    "finevo-pilot-release-attestation-v1"
)
PILOT_STAGE_RECEIPT_SCHEMA_VERSION = "finevo-pilot-stage-receipt-v1"
PILOT_EXPERIMENT_C_SENSITIVITY_SCHEMA_VERSION = (
    "finevo-experiment-c-sensitivity-v1"
)
LEGACY_CAPABILITY_SCHEMA_VERSION = "finevo-capability-gate-v1"
CURRENT_CAPABILITY_SCHEMA_VERSION = "finevo-capability-gate-v2"
SUPPORTED_CAPABILITY_SCHEMA_VERSIONS = frozenset(
    {
        LEGACY_CAPABILITY_SCHEMA_VERSION,
        CURRENT_CAPABILITY_SCHEMA_VERSION,
    }
)
EVIDENCE_NAMESPACE = "current_v2/pilot-v1"
CURRENT_SCIENTIFIC_SCOPE = "preregistered_mechanism_micro_pilot"
HISTORICAL_SCOPE = "historical_pre_p0_v1"

TERMINAL_STATUSES = frozenset(
    {
        "complete",
        "failed",
        "budget-stopped",
        "integrity-stopped",
        "capability-no-go",
    }
)
KNOWN_NONTERMINAL_STATUSES = frozenset({"scheduled", "running", "reserved"})
NON_SCIENTIFIC_STAGES = frozenset(
    {"capability-preflight", "q-ref-resolution"}
)
CORE_STAGES = frozenset(
    {
        "stage0-calibration",
        "experiment-a",
        "experiment-b",
        "experiment-c",
        "experiment-d",
        "cross-model-sentinels",
    }
)
RUNNER_EXECUTION_MODES = frozenset({"actor_run", "matched_duplicate"})
TERMINAL_EXECUTION_MODES = frozenset(
    {
        "capability_probe",
        "q_ref_resolution",
        "offline_candidate_admission",
        "checkpoint_continuation",
    }
)


class PilotEvidenceError(RuntimeError):
    """Raised when raw evidence cannot enter ``current_v2/pilot-v1``."""


@dataclass(frozen=True, slots=True)
class PilotEvidencePackage:
    """Receipt returned after a package is installed atomically."""

    package_dir: Path
    manifest_path: Path
    checksums_path: Path
    contract_hash: str
    scientific_complete: bool
    claim_gates: Mapping[str, Any]


def _reject_constant(value: str) -> None:
    raise PilotEvidenceError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PilotEvidenceError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _strict_json_load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except FileNotFoundError as exc:
        raise PilotEvidenceError(f"required evidence file is missing: {path}") from exc
    except UnicodeDecodeError as exc:
        raise PilotEvidenceError(f"evidence file is not UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PilotEvidenceError(f"invalid JSON evidence file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PilotEvidenceError(f"evidence root must be an object: {path}")
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PilotEvidenceError(f"evidence is not finite JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _pretty_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PilotEvidenceError(f"evidence is not finite JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_bytes(value))


def _is_finite_scalar(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(
        float(value)
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotEvidenceError(f"{name} must be an object")
    return value


def _contains_text(value: Any, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, Mapping):
        return any(
            _contains_text(key, needle) or _contains_text(item, needle)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_text(item, needle) for item in value)
    return False


def _terminal_summary_hash(value: Mapping[str, Any]) -> str:
    payload = _json_copy(value)
    payload.pop("integrity", None)
    return canonical_sha256(payload)


def _bound_artifact_hash(value: Mapping[str, Any]) -> str:
    payload = _json_copy(value)
    integrity = payload.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("content_sha256", None)
    return canonical_sha256(payload)


def write_terminal_summary(
    path: str | Path,
    *,
    contract: PilotContract,
    run_spec: PilotRunSpec | Mapping[str, Any],
    resolved_git_commit: str,
    git_tag: str,
    payload: Mapping[str, Any],
    scientific_evidence: bool,
    diagnostic_only: bool,
    evidence_scope: str,
) -> Path:
    """Seal one non-runner terminal summary for later evidence publication.

    This is the integration boundary for capability receipts, checkpoint
    continuations, narrative branches, and offline analyses.  The payload must
    already contain observed metrics; this function does not calculate or fill
    missing results.
    """

    target = Path(path)
    if target.exists():
        raise PilotEvidenceError(f"refusing to overwrite terminal summary: {target}")
    spec_value = (
        run_spec.to_dict()
        if isinstance(run_spec, PilotRunSpec)
        else _json_copy(_mapping(run_spec, "run_spec"))
    )
    if not isinstance(scientific_evidence, bool) or not isinstance(
        diagnostic_only, bool
    ):
        raise TypeError("scientific_evidence and diagnostic_only must be booleans")
    if not isinstance(evidence_scope, str) or not evidence_scope:
        raise ValueError("evidence_scope must be non-empty")
    if scientific_evidence and (
        diagnostic_only or evidence_scope != CURRENT_SCIENTIFIC_SCOPE
    ):
        raise PilotEvidenceError(
            "scientific terminal summaries must be non-diagnostic current-method evidence"
        )
    if _contains_text(payload, HISTORICAL_SCOPE):
        raise PilotEvidenceError("historical_pre_p0_v1 cannot be sealed as pilot evidence")
    binding = contract.validate_provenance(resolved_git_commit, git_tag)
    value: dict[str, Any] = {
        "schema_version": PILOT_TERMINAL_SUMMARY_SCHEMA_VERSION,
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "run_spec": spec_value,
        "provenance": {
            **binding,
            "tag_object_type": "tag",
            "worktree_clean": True,
        },
        "evidence_scope": evidence_scope,
        "diagnostic_only": diagnostic_only,
        "scientific_evidence": scientific_evidence,
        "payload": _json_copy(_mapping(payload, "payload")),
    }
    value["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": _terminal_summary_hash(value),
    }
    _atomic_bytes(target, _canonical_bytes(value))
    mode = stat.S_IMODE(target.stat().st_mode)
    target.chmod(mode & ~0o222)
    return target


def _resolve_artifact(raw_root: Path, artifact: Any) -> Path:
    if not isinstance(artifact, str) or not artifact.strip():
        raise PilotEvidenceError("completed run is missing its artifact path")
    candidate = Path(artifact)
    if not candidate.is_absolute():
        candidate = raw_root / candidate
    if candidate.is_symlink():
        raise PilotEvidenceError(f"artifact symlinks are forbidden: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PilotEvidenceError(f"declared artifact is missing: {candidate}") from exc
    root = raw_root.resolve()
    if not resolved.is_relative_to(root):
        raise PilotEvidenceError(f"artifact escapes the pilot raw root: {candidate}")
    return resolved


def _validate_binding(
    contract: PilotContract,
    *,
    git_tag: Any,
    resolved_git_commit: Any,
    binding: Any,
) -> dict[str, Any]:
    if not isinstance(git_tag, str) or not isinstance(resolved_git_commit, str):
        raise PilotEvidenceError("artifact provenance is missing tag/commit identity")
    expected = contract.validate_provenance(resolved_git_commit, git_tag)
    if _json_copy(_mapping(binding, "contract binding")) != expected:
        raise PilotEvidenceError("artifact contract/tag binding does not match frozen contract")
    return expected


def _scope_gate(
    *,
    stage_id: str,
    evidence: Mapping[str, Any],
    source: Path,
) -> bool:
    if _contains_text(evidence, HISTORICAL_SCOPE):
        raise PilotEvidenceError(
            f"historical_pre_p0_v1 artifact cannot enter current_v2: {source}"
        )
    diagnostic = evidence.get("diagnostic_only")
    scientific = evidence.get("scientific_evidence")
    scope = evidence.get("evidence_scope")
    if not isinstance(diagnostic, bool) or not isinstance(scientific, bool):
        raise PilotEvidenceError(f"artifact lacks explicit evidence flags: {source}")
    if diagnostic and scientific:
        raise PilotEvidenceError(
            f"diagnostic artifact falsely claims scientific evidence: {source}"
        )
    if stage_id in CORE_STAGES:
        if diagnostic or not scientific or scope != CURRENT_SCIENTIFIC_SCOPE:
            raise PilotEvidenceError(
                f"core run is not eligible current-method scientific evidence: {source}"
            )
        return True
    if stage_id in NON_SCIENTIFIC_STAGES:
        if scientific:
            raise PilotEvidenceError(
                f"{stage_id} may support gating but cannot claim an effect: {source}"
            )
        return False
    raise PilotEvidenceError(f"unknown pilot stage in evidence scope gate: {stage_id}")


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_finite_fields(
    value: Mapping[str, Any],
    fields: Sequence[str],
    *,
    name: str,
) -> None:
    for field in fields:
        if not _is_finite_scalar(value.get(field)):
            raise PilotEvidenceError(f"{name}.{field} must be finite")


def _validate_rng_binding(value: Any, *, name: str) -> Mapping[str, Any]:
    binding = _mapping(value, name)
    expected_keys = {
        "schema_version",
        "derivation",
        "generated_before_provider_calls",
        "source_hash",
        "schedule_hash",
        "horizon",
    }
    if set(binding) != expected_keys:
        raise PilotEvidenceError(
            f"{name} must contain exactly {sorted(expected_keys)}"
        )
    if (
        binding.get("schema_version") != "finevo-pilot-d-rng-schedule-v1"
        or binding.get("derivation")
        != "checkpoint-bound-domain-separated-sha256"
        or binding.get("generated_before_provider_calls") is not True
        or binding.get("horizon") != 6
        or not _is_sha256(binding.get("source_hash"))
        or not _is_sha256(binding.get("schedule_hash"))
    ):
        raise PilotEvidenceError(f"{name} is not a valid six-step causal RNG binding")
    return binding


def _validate_causal_bindings(
    value: Any,
    *,
    name: str,
    narrative: bool,
    action_grid: Mapping[str, Any],
    narrative_fixture_hash: str | None = None,
) -> Mapping[str, Any]:
    bindings = _mapping(value, name)
    if narrative:
        required = {
            "kind",
            "checkpoint_hash",
            "prefix_hash",
            "pre_generated_rng_hashes",
            "rng_schedule_binding",
            "shared_result_hash",
            "fixture_hash",
            "branch_narrative_id",
            "branch_text_hash",
            "branch_rng_pre_step_hashes",
            "branch_action_completions",
            "proposal_counters_before",
            "proposal_counters_after",
            "proposals_frozen",
            "focal_agent_id",
            "action_grid",
        }
    else:
        required = {
            "kind",
            "checkpoint_hash",
            "prefix_hash",
            "shock_schedule_hash",
            "pre_generated_rng_hashes",
            "rng_schedule_binding",
            "shared_result_hash",
            "matched_replay_equal",
            "branch_treatment",
            "branch_rng_pre_step_hashes",
            "branch_action_completions",
            "proposal_counters_before",
            "proposal_counters_after",
            "proposals_frozen",
            "focal_agent_id",
            "wrong_context_source_agent_id",
            "action_grid",
            "error_common_start_equal",
            "error_common_start_hash",
            "branch_forced_active_start_hash",
        }
    if set(bindings) != required:
        raise PilotEvidenceError(
            f"{name} must contain exactly {sorted(required)}"
        )
    for field in required & {
        "checkpoint_hash",
        "prefix_hash",
        "shock_schedule_hash",
        "shared_result_hash",
        "fixture_hash",
        "branch_text_hash",
        "error_common_start_hash",
    }:
        if field in required and not _is_sha256(bindings.get(field)):
            raise PilotEvidenceError(f"{name}.{field} must be a SHA-256 digest")
    for field in ("pre_generated_rng_hashes", "branch_rng_pre_step_hashes"):
        rng_hashes = bindings.get(field)
        if (
            not isinstance(rng_hashes, Sequence)
            or isinstance(rng_hashes, (str, bytes))
            or len(rng_hashes) != 6
            or any(not _is_sha256(item) for item in rng_hashes)
        ):
            raise PilotEvidenceError(
                f"{name}.{field} must contain six SHA-256 digests"
            )
    if (
        list(bindings["pre_generated_rng_hashes"])
        != list(bindings["branch_rng_pre_step_hashes"])
    ):
        raise PilotEvidenceError(
            f"{name} branch RNG hashes differ from the pre-generated schedule"
        )
    _validate_rng_binding(
        bindings.get("rng_schedule_binding"),
        name=f"{name}.rng_schedule_binding",
    )
    if (
        bindings.get("kind") != ("narrative" if narrative else "continuation")
        or bindings.get("branch_action_completions") != 24
        or bindings.get("proposals_frozen") is not True
        or bindings.get("proposal_counters_before")
        != bindings.get("proposal_counters_after")
        or not isinstance(bindings.get("proposal_counters_before"), Mapping)
        or bindings.get("focal_agent_id") != 0
        or bindings.get("action_grid") != dict(action_grid)
    ):
        raise PilotEvidenceError(
            f"{name} violates the frozen branch/focal/proposal/action-grid contract"
        )
    if narrative:
        if (
            narrative_fixture_hash is None
            or bindings.get("fixture_hash") != narrative_fixture_hash
            or not isinstance(bindings.get("branch_narrative_id"), str)
            or not _is_sha256(bindings.get("branch_text_hash"))
        ):
            raise PilotEvidenceError(
                f"{name} differs from the frozen narrative fixture binding"
            )
    elif (
        bindings.get("matched_replay_equal") is not True
        or bindings.get("wrong_context_source_agent_id") != 1
        or bindings.get("error_common_start_equal") is not True
        or not isinstance(bindings.get("branch_treatment"), str)
        or (
            bindings.get("branch_forced_active_start_hash") is not None
            and not _is_sha256(
                bindings.get("branch_forced_active_start_hash")
            )
        )
    ):
        raise PilotEvidenceError(
            f"{name} violates the matched/error/wrong-context causal contract"
        )
    return bindings


def _validate_capability_v2(capability: Mapping[str, Any]) -> None:
    """Recompute the v2 ITT/interface split before admitting a gate receipt."""

    rows = capability.get("rows")
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or len(rows) != 30
        or not _is_sha256(capability.get("taskset_sha256"))
    ):
        raise PilotEvidenceError(
            "capability v2 must bind one taskset and exactly 30 rows"
        )
    category_contract = {
        "utility-ranking": (12, 10),
        "rule-application": (12, 10),
        "rule-proposal": (6, 5),
    }
    task_ids: set[str] = set()
    expected_totals: dict[str, dict[str, Any]] = {}
    for category, (registered_total, required) in category_contract.items():
        category_rows = [
            row
            for row in rows
            if isinstance(row, Mapping) and row.get("category") == category
        ]
        if len(category_rows) != registered_total:
            raise PilotEvidenceError(
                f"capability v2 category {category!r} has the wrong denominator"
            )
        for row in category_rows:
            interface_status = row.get("interface_status")
            expected_evaluable = interface_status in {"pass", "parse_error"}
            if (
                row.get("schema_version") != CURRENT_CAPABILITY_SCHEMA_VERSION
                or row.get("taskset_sha256") != capability["taskset_sha256"]
                or not isinstance(row.get("task_id"), str)
                or not isinstance(row.get("correct"), bool)
                or not isinstance(row.get("evaluable"), bool)
                or interface_status
                not in {
                    "pass",
                    "parse_error",
                    "provider_error",
                    "incomplete",
                    "invalid_finish",
                }
                or row.get("evaluable") is not expected_evaluable
            ):
                raise PilotEvidenceError(
                    "capability v2 row schema/status is inconsistent"
                )
            if interface_status != "pass" and row.get("correct") is not False:
                raise PilotEvidenceError(
                    "capability v2 non-pass row cannot be correct"
                )
            provider_error = row.get("provider_error")
            provider_error_details = row.get("provider_error_details")
            parse_error = row.get("parse_error")
            parse_error_code = row.get("parse_error_code")
            parse_error_offset = row.get("parse_error_offset")
            if interface_status == "pass" and any(
                value is not None
                for value in (
                    provider_error,
                    provider_error_details,
                    parse_error,
                    parse_error_code,
                    parse_error_offset,
                )
            ):
                raise PilotEvidenceError(
                    "capability v2 pass row contains failure metadata"
                )
            if interface_status == "parse_error" and (
                not isinstance(parse_error, str)
                or not parse_error
                or not isinstance(parse_error_code, str)
                or not parse_error_code
                or (
                    parse_error_offset is not None
                    and (
                        isinstance(parse_error_offset, bool)
                        or not isinstance(parse_error_offset, int)
                        or parse_error_offset < 0
                    )
                )
                or provider_error is not None
                or provider_error_details is not None
            ):
                raise PilotEvidenceError(
                    "capability v2 parse-error row is inconsistent"
                )
            if interface_status in {"provider_error", "incomplete"} and (
                not isinstance(provider_error, str)
                or not provider_error
                or not isinstance(provider_error_details, Mapping)
                or any(
                    value is not None
                    for value in (
                        parse_error,
                        parse_error_code,
                        parse_error_offset,
                    )
                )
            ):
                raise PilotEvidenceError(
                    "capability v2 provider-error row is inconsistent"
                )
            if interface_status == "invalid_finish" and any(
                value is not None
                for value in (
                    provider_error,
                    provider_error_details,
                    parse_error,
                    parse_error_code,
                    parse_error_offset,
                )
            ):
                raise PilotEvidenceError(
                    "capability v2 invalid-finish row contains failure metadata"
                )
            task_ids.add(str(row["task_id"]))
        registered_correct = sum(bool(row["correct"]) for row in category_rows)
        evaluable_count = sum(bool(row["evaluable"]) for row in category_rows)
        conditional_correct = sum(
            bool(row["correct"]) for row in category_rows if row["evaluable"]
        )
        conditional_accuracy = (
            conditional_correct / evaluable_count if evaluable_count else None
        )
        expected_totals[category] = {
            "correct": registered_correct,
            "denominator": registered_total,
            "required": required,
            "registered_correct": registered_correct,
            "registered_total": registered_total,
            "evaluable_count": evaluable_count,
            "conditional_correct": conditional_correct,
            "conditional_accuracy": conditional_accuracy,
            "interface_failure_count": registered_total - evaluable_count,
        }
    if len(task_ids) != 30:
        raise PilotEvidenceError("capability v2 task IDs are not unique")

    totals = _mapping(
        capability.get("category_totals"),
        "capability v2 category_totals",
    )
    if set(totals) != set(category_contract):
        raise PilotEvidenceError("capability v2 category totals are incomplete")
    for category, expected in expected_totals.items():
        observed = _mapping(
            totals.get(category),
            f"capability v2 totals {category}",
        )
        if set(expected) - set(observed):
            raise PilotEvidenceError(
                f"capability v2 totals {category!r} lack required fields"
            )
        for field, value in expected.items():
            actual = observed.get(field)
            if isinstance(value, float):
                if (
                    not _is_finite_scalar(actual)
                    or not math.isclose(
                        float(actual), value, rel_tol=0.0, abs_tol=1e-12
                    )
                ):
                    raise PilotEvidenceError(
                        f"capability v2 totals {category!r} are inconsistent"
                    )
            elif actual != value:
                raise PilotEvidenceError(
                    f"capability v2 totals {category!r} are inconsistent"
                )

    registered_checks = {
        category: expected_totals[category]["correct"]
        >= expected_totals[category]["required"]
        for category in category_contract
    }
    if capability.get("checks") != registered_checks:
        raise PilotEvidenceError("capability v2 registered checks are inconsistent")
    conditional_checks = {
        category: (
            None
            if expected_totals[category]["conditional_accuracy"] is None
            else expected_totals[category]["conditional_accuracy"]
            >= (
                expected_totals[category]["required"]
                / expected_totals[category]["registered_total"]
            )
        )
        for category in category_contract
    }
    interface_failure_count = sum(
        not bool(row["evaluable"]) for row in rows if isinstance(row, Mapping)
    )
    expected_interface = {
        "pass": interface_failure_count == 0,
        "failure_count": interface_failure_count,
    }
    if capability.get("interface_gate") != expected_interface:
        raise PilotEvidenceError("capability v2 interface gate is inconsistent")
    expected_assessment = {
        "status": (
            "not_evaluable"
            if interface_failure_count
            else "pass"
            if all(conditional_checks.values())
            else "fail"
        ),
        "pass": (
            None
            if interface_failure_count
            else all(conditional_checks.values())
        ),
        "checks": conditional_checks,
    }
    if capability.get("capability_assessment") != expected_assessment:
        raise PilotEvidenceError(
            "capability v2 conditional assessment is inconsistent"
        )
    expected_pass = (
        expected_interface["pass"] and expected_assessment["pass"] is True
    )
    if capability.get("pass") is not expected_pass:
        raise PilotEvidenceError("capability v2 overall pass is inconsistent")
    if capability.get("provider_failure_count") != sum(
        row.get("provider_error") is not None
        for row in rows
        if isinstance(row, Mapping)
    ):
        raise PilotEvidenceError(
            "capability v2 provider-failure count is inconsistent"
        )
    if capability.get("parse_failure_count") != sum(
        row.get("parse_error") is not None
        for row in rows
        if isinstance(row, Mapping)
    ):
        raise PilotEvidenceError(
            "capability v2 parse-failure count is inconsistent"
        )


def _validate_terminal_payload_marker(
    contract: PilotContract,
    spec: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    raw_root: Path,
) -> None:
    """Enforce the execution-mode-specific terminal-summary contract."""

    mode = str(spec.get("execution_mode", ""))
    if mode not in TERMINAL_EXECUTION_MODES:
        raise PilotEvidenceError(
            f"execution mode {mode!r} requires a verified-run manifest, "
            "not a terminal summary"
        )
    metrics = _mapping(payload.get("metrics", {}), "terminal metrics")
    gate = _mapping(payload.get("gate_evidence", {}), "terminal gate_evidence")
    if mode == "capability_probe":
        capability = _mapping(payload.get("capability"), "capability payload")
        schema_version = capability.get("schema_version")
        if schema_version not in SUPPORTED_CAPABILITY_SCHEMA_VERSIONS:
            raise PilotEvidenceError(
                f"unsupported capability schema version: {schema_version!r}"
            )
        for field in ("pass", "preflight_go"):
            if not isinstance(capability.get(field), bool):
                raise PilotEvidenceError(f"capability payload lacks boolean {field}")
        if schema_version == CURRENT_CAPABILITY_SCHEMA_VERSION:
            _validate_capability_v2(capability)
        if not isinstance(gate.get("go"), bool):
            raise PilotEvidenceError("capability gate_evidence lacks boolean go")
        if capability["preflight_go"] is not gate["go"]:
            raise PilotEvidenceError(
                "capability preflight_go differs from sealed gate receipt"
            )
        return
    if mode == "q_ref_resolution":
        resolution = _mapping(
            payload.get("q_ref_resolution"),
            "q_ref_resolution payload",
        )
        if (
            not _is_finite_scalar(resolution.get("q_ref"))
            or float(resolution["q_ref"]) <= 0
            or not isinstance(resolution.get("row_count"), int)
            or resolution["row_count"] <= 0
            or not isinstance(resolution.get("source_manifest"), str)
            or not _is_sha256(resolution.get("source_manifest_sha256"))
            or not isinstance(resolution.get("resolution_artifact"), str)
            or not gate
        ):
            raise PilotEvidenceError(
                "q_ref terminal summary lacks its sealed positive resolution marker"
            )
        return
    if mode == "offline_candidate_admission":
        reliability = _mapping(
            metrics.get("rule_reliability"),
            "offline candidate-admission metrics",
        )
        for field in (
            "unsupported_candidate_rejected",
            "false_rule_ever_active",
            "unverified_false_rule_ever_active",
            "same_candidate_content",
        ):
            if not isinstance(reliability.get(field), bool):
                raise PilotEvidenceError(
                    f"offline candidate-admission metrics lack boolean {field}"
                )
        if reliability.get("provider_calls") != 0:
            raise PilotEvidenceError(
                "offline candidate-admission must attest zero provider calls"
            )
        if gate != reliability or not isinstance(payload.get("offline_source"), str):
            raise PilotEvidenceError(
                "offline candidate-admission gate/source marker mismatch"
            )
        return

    completion = _mapping(
        payload.get("completion_receipt"),
        "checkpoint continuation completion_receipt",
    )
    if (
        completion.get("branch_action_completions") != 24
        or completion.get("observed_trajectory_action_rows") != 24
        or not isinstance(completion.get("shared_budget"), Mapping)
    ):
        raise PilotEvidenceError(
            "checkpoint continuation lacks the exact 4x6 completion receipt"
        )
    narrative = str(spec.get("arm_id")) == "narrative-content"
    causal = _validate_causal_bindings(
        gate.get("causal_bindings"),
        name="checkpoint gate_evidence.causal_bindings",
        narrative=narrative,
        action_grid=_mapping(
            contract.stop_go["experiment_d"]["action_grid"],
            "contract experiment_d action_grid",
        ),
        narrative_fixture_hash=str(
            contract.stop_go["experiment_d"]["narrative_fixture_hash"]
        ),
    )
    if narrative:
        narrative_metrics = _mapping(
            metrics.get("narrative"),
            "checkpoint narrative metrics",
        )
        _require_finite_fields(
            narrative_metrics,
            (
                "first_labor_hours",
                "first_consumption_rate",
                "immediate_flow_utility",
                "six_step_discounted_flow_utility",
                "final_wealth",
            ),
            name="checkpoint narrative metrics",
        )
        narrative_payload = _mapping(
            payload.get("narrative"),
            "checkpoint narrative payload",
        )
        if (
            narrative_payload.get("narrative_id") != spec.get("narrative_id")
            or not isinstance(narrative_payload.get("branch"), Mapping)
            or not isinstance(narrative_payload.get("claim_boundary"), str)
            or not isinstance(
                gate.get("semantic_equivalence_within_one_action_bin"),
                Mapping,
            )
            or not isinstance(gate.get("aligned_vs_opposite_delta"), Mapping)
        ):
            raise PilotEvidenceError("narrative terminal marker is incomplete")
    else:
        continuation = _mapping(
            metrics.get("continuation"),
            "checkpoint continuation metrics",
        )
        if (
            not isinstance(continuation.get("focal"), Mapping)
            or not isinstance(continuation.get("population"), Mapping)
            or gate.get("matched_replay_equal") is not True
        ):
            raise PilotEvidenceError(
                "checkpoint continuation metrics/replay marker is incomplete"
            )
    if (
        gate.get("checkpoint_hash") != causal.get("checkpoint_hash")
        or gate.get("prefix_hash") != causal.get("prefix_hash")
    ):
        raise PilotEvidenceError(
            "checkpoint summary top-level hashes differ from causal bindings"
        )
    source_value = payload.get("shared_source")
    source_hash = payload.get("shared_source_sha256")
    if not isinstance(source_value, str) or not _is_sha256(source_hash):
        raise PilotEvidenceError(
            "checkpoint terminal lacks its shared source path/hash binding"
        )
    source_path = Path(source_value)
    if not source_path.is_absolute():
        source_path = raw_root / source_path
    if source_path.is_symlink():
        raise PilotEvidenceError("checkpoint shared-source symlinks are forbidden")
    try:
        resolved_source = source_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PilotEvidenceError(
            "checkpoint shared source is missing"
        ) from exc
    if not resolved_source.is_relative_to(raw_root.resolve()):
        raise PilotEvidenceError("checkpoint shared source escapes the raw root")
    if _sha256_file(resolved_source) != source_hash:
        raise PilotEvidenceError("checkpoint shared source SHA-256 mismatch")
    source = _strict_json_load(resolved_source)
    source_result_hash = source.get("result_hash")
    source_body = _json_copy(source)
    source_body.pop("result_hash", None)
    # The orchestrator appends these evidence-scope fields after the causal
    # engine seals its result.  They remain protected by shared_source_sha256.
    source_body.pop("contract_sha256", None)
    source_body.pop("diagnostic_only", None)
    source_body.pop("scientific_evidence", None)
    if (
        not _is_sha256(source_result_hash)
        or canonical_sha256(source_body) != source_result_hash
        or source_result_hash != causal.get("shared_result_hash")
        or source.get("contract_sha256") != contract.canonical_hash
        or source.get("diagnostic_only") is not False
        or source.get("scientific_evidence") is not True
    ):
        raise PilotEvidenceError(
            "checkpoint shared source result/scope binding is invalid"
        )
    branches = _mapping(source.get("branches"), "checkpoint shared-source branches")
    if narrative:
        if source.get("schema_version") != "finevo-pilot-narrative-v1":
            raise PilotEvidenceError("shared narrative source schema mismatch")
        branch_key = str(causal["branch_narrative_id"])
        source_checks = {
            "checkpoint_hash": causal["checkpoint_hash"],
            "prefix_hash": causal["prefix_hash"],
            "pre_generated_rng_hashes": causal[
                "pre_generated_rng_hashes"
            ],
            "rng_schedule_binding": causal["rng_schedule_binding"],
            "fixture_hash": causal["fixture_hash"],
            "focal_agent_id": causal["focal_agent_id"],
            "action_grid": causal["action_grid"],
        }
    else:
        if source.get("schema_version") != "finevo-pilot-continuation-v1":
            raise PilotEvidenceError("shared continuation source schema mismatch")
        branch_key = str(causal["branch_treatment"])
        common_start = _mapping(
            source.get("erroneous_forced_active_common_start"),
            "shared continuation erroneous common start",
        )
        source_checks = {
            "checkpoint_hash": causal["checkpoint_hash"],
            "prefix_hash": causal["prefix_hash"],
            "shock_schedule_hash": causal["shock_schedule_hash"],
            "pre_generated_rng_hashes": causal[
                "pre_generated_rng_hashes"
            ],
            "rng_schedule_binding": causal["rng_schedule_binding"],
            "matched_replay_equal": causal["matched_replay_equal"],
            "focal_agent_id": causal["focal_agent_id"],
            "wrong_context_source_agent_id": causal[
                "wrong_context_source_agent_id"
            ],
            "action_grid": causal["action_grid"],
        }
        if (
            common_start.get("equal") is not True
            or common_start.get("forced_active_start_hash")
            != causal["error_common_start_hash"]
        ):
            raise PilotEvidenceError(
                "shared continuation erroneous-rule start binding mismatch"
            )
    if any(source.get(key) != expected for key, expected in source_checks.items()):
        raise PilotEvidenceError(
            "checkpoint causal binding differs from its shared source"
        )
    branch = _mapping(
        branches.get(branch_key),
        f"checkpoint shared-source branch {branch_key}",
    )
    api_usage = branch.get("api_usage")
    branch_checks = {
        "rng_pre_step_hashes": causal["branch_rng_pre_step_hashes"],
        "proposal_counters_before": causal["proposal_counters_before"],
        "proposal_counters_after": causal["proposal_counters_after"],
        "freeze_proposals": causal["proposals_frozen"],
    }
    if (
        any(branch.get(key) != expected for key, expected in branch_checks.items())
        or not isinstance(api_usage, Sequence)
        or isinstance(api_usage, (str, bytes))
        or len(api_usage) != causal["branch_action_completions"]
        or branch.get("api_usage_hash") != canonical_sha256(api_usage)
    ):
        raise PilotEvidenceError(
            "checkpoint branch binding differs from its shared source"
        )
    _validate_provider_usage_rows(contract, spec, api_usage)
    if narrative:
        source_narrative = _mapping(
            branch.get("narrative"),
            "shared-source narrative branch fixture",
        )
        if (
            source_narrative.get("narrative_id")
            != causal["branch_narrative_id"]
            or source_narrative.get("text_hash")
            != causal["branch_text_hash"]
        ):
            raise PilotEvidenceError(
                "narrative branch text binding differs from shared source"
            )
    else:
        intervention = _mapping(
            branch.get("intervention"),
            "shared-source continuation intervention",
        )
        if intervention.get("forced_active_start_hash") != causal.get(
            "branch_forced_active_start_hash"
        ):
            raise PilotEvidenceError(
                "continuation forced-start binding differs from shared source"
            )


def _load_terminal_summary(
    contract: PilotContract,
    spec: Mapping[str, Any],
    path: Path,
    *,
    raw_root: Path,
) -> dict[str, Any]:
    value = _strict_json_load(path)
    if value.get("schema_version") != PILOT_TERMINAL_SUMMARY_SCHEMA_VERSION:
        raise PilotEvidenceError(
            f"completed non-runner artifact is not a sealed terminal summary: {path}"
        )
    integrity = _mapping(value.get("integrity"), "terminal summary integrity")
    if integrity.get("canonicalization") != "json-sort-keys-utf8-v1":
        raise PilotEvidenceError("unsupported terminal summary canonicalization")
    if integrity.get("content_sha256") != _terminal_summary_hash(value):
        raise PilotEvidenceError(f"terminal summary checksum mismatch: {path}")
    if value.get("contract_id") != contract.contract_id:
        raise PilotEvidenceError("terminal summary contract ID mismatch")
    if value.get("contract_sha256") != contract.canonical_hash:
        raise PilotEvidenceError("terminal summary contract hash mismatch")
    if _json_copy(_mapping(value.get("run_spec"), "terminal run_spec")) != _json_copy(
        spec
    ):
        raise PilotEvidenceError("terminal summary run spec differs from ITT ledger")
    provenance = _mapping(value.get("provenance"), "terminal provenance")
    binding = _validate_binding(
        contract,
        git_tag=provenance.get("git_tag"),
        resolved_git_commit=provenance.get("resolved_git_commit"),
        binding={
            key: provenance.get(key)
            for key in (
                "git_tag",
                "resolved_git_commit",
                "commit_resolution",
                "p0_base_commit",
                "contract_id",
                "contract_sha256",
            )
        },
    )
    if provenance.get("tag_object_type") != "tag":
        raise PilotEvidenceError("terminal summary does not bind an annotated tag")
    if provenance.get("worktree_clean") is not True:
        raise PilotEvidenceError("terminal summary was not produced from a clean worktree")
    evidence = {
        "diagnostic_only": value.get("diagnostic_only"),
        "scientific_evidence": value.get("scientific_evidence"),
        "evidence_scope": value.get("evidence_scope"),
        "payload": value.get("payload"),
    }
    eligible = _scope_gate(
        stage_id=str(spec["stage_id"]),
        evidence=evidence,
        source=path,
    )
    payload = _mapping(value.get("payload"), "terminal payload")
    _validate_terminal_payload_marker(
        contract,
        spec,
        payload,
        raw_root=raw_root,
    )
    metrics = payload.get("metrics", {})
    gate_evidence = payload.get("gate_evidence", {})
    if not isinstance(metrics, Mapping) or not isinstance(gate_evidence, Mapping):
        raise PilotEvidenceError("terminal metrics and gate_evidence must be objects")
    return {
        "artifact_kind": "terminal-summary",
        "artifact_sha256": _sha256_file(path),
        "binding": binding,
        "scientific_eligible": eligible,
        "metrics": _json_copy(metrics),
        "gate_evidence": _json_copy(gate_evidence),
        "capability": _json_copy(payload.get("capability", {})),
        "narrative": _json_copy(payload.get("narrative", {})),
    }


def _validate_provider_usage_rows(
    contract: PilotContract,
    spec: Mapping[str, Any],
    api_rows: Any,
    *,
    expected_runner_schema: str = RUNNER_SCHEMA_VERSION,
    allow_legacy: bool = False,
) -> None:
    """Validate one scientific call ledger against its frozen provider profile."""

    if (
        not isinstance(api_rows, Sequence)
        or isinstance(api_rows, (str, bytes))
        or not api_rows
    ):
        raise PilotEvidenceError("scientific run has no provider usage denominator")
    profile = contract.provider_profiles[str(spec["model_id"])]
    expected_seed = spec.get("decoding_seed")
    from llm_providers import (  # pylint: disable=import-outside-toplevel
        PINNED_PROVIDER_SDK_VERSIONS,
    )

    expected_provider = {
        "openai": "openai",
        "openrouter": "thirdparty",
        "ollama": "ollama",
    }.get(profile.transport)
    if expected_provider is None:
        raise PilotEvidenceError(
            "scientific run used a non-dispatchable provider transport"
        )
    expected_identity = dict(profile.artifact_identity)
    if profile.transport == "openrouter":
        expected_response_providers = set(profile.provider_pin)
        expected_response_route = expected_identity.get("served_snapshot")
        expected_sdk_name = "openai-python"
        expected_sdk_version = PINNED_PROVIDER_SDK_VERSIONS["openai"]
        expected_route_attestation = "OR_RA_PASS"
        expected_temperature_dispatch = "explicit"
        expected_request_parameters = {
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "top_p",
            *profile.openrouter_request_options().keys(),
        }
        if not expected_response_route:
            raise PilotEvidenceError(
                "OpenRouter profile lacks a frozen served-snapshot route"
            )
    elif profile.transport == "openai":
        expected_response_providers = {"OpenAI-direct"}
        expected_response_route = "direct"
        expected_sdk_name = "openai-python"
        expected_sdk_version = PINNED_PROVIDER_SDK_VERSIONS["openai"]
        expected_route_attestation = None
        reasoning_model = profile.requested_model.startswith(("gpt-5", "o1", "o3"))
        expected_temperature_dispatch = (
            "omitted_unsupported" if reasoning_model else "explicit"
        )
        expected_request_parameters = {
            "model",
            "messages",
            "top_p",
            *profile.openai_request_options().keys(),
            (
                "max_completion_tokens"
                if reasoning_model
                else "max_tokens"
            ),
        }
        if not reasoning_model:
            expected_request_parameters.add("temperature")
    else:
        expected_response_providers = {"local-ollama"}
        expected_response_route = "local"
        expected_sdk_name = "requests"
        expected_sdk_version = PINNED_PROVIDER_SDK_VERSIONS["requests"]
        expected_route_attestation = None
        expected_temperature_dispatch = "explicit"
        expected_request_parameters = {
            "model",
            "messages",
            "stream",
            "options",
        }
        if profile.json_mode == "json_object":
            expected_request_parameters.add("format")
    if expected_seed is not None and profile.transport != "ollama":
        expected_request_parameters.add("seed")

    supported_schema_versions = {
        LEGACY_RUNNER_SCHEMA_VERSION,
        PREVIOUS_RUNNER_SCHEMA_VERSION,
        RUNNER_SCHEMA_VERSION,
    }
    if expected_runner_schema not in supported_schema_versions:
        raise PilotEvidenceError(
            "provider usage validation requested an unsupported runner schema"
        )
    if (
        expected_runner_schema != RUNNER_SCHEMA_VERSION
        and allow_legacy is not True
    ):
        raise PilotEvidenceError(
            "legacy provider usage requires explicit read-only validation"
        )
    for row in api_rows:
        if not isinstance(row, Mapping):
            raise PilotEvidenceError("provider usage row must be an object")
        row_schema = row.get("schema_version")
        if row_schema != expected_runner_schema:
            raise PilotEvidenceError(
                "provider usage row does not match its expected runner schema"
            )
        usage = row.get("usage")
        if not isinstance(usage, Mapping):
            raise PilotEvidenceError(
                "provider usage row lacks an accounted usage object"
            )
        numeric_usage = (
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("cost_usd"),
        )
        if any(
            not _is_finite_scalar(value) or float(value) < 0
            for value in numeric_usage
        ):
            raise PilotEvidenceError(
                "provider usage row contains invalid usage/cost"
            )
        if (
            row.get("provider") != expected_provider
            or row.get("model") != profile.requested_model
            or row.get("response_model") != profile.served_model
            or row.get("attempts") != 1
            or row.get("error_type") is not None
            or row.get("request_seed") != expected_seed
            or row.get("request_profile_id") != profile.profile_id
            or row.get("request_provider_pin") != list(profile.provider_pin)
            or row.get("request_artifact_identity") != expected_identity
            or row.get("request_price_snapshot_source")
            != profile.price_snapshot.source
            or row.get("request_price_snapshot_captured_at")
            != profile.price_snapshot.captured_at
            or row.get("response_provider") not in expected_response_providers
            or row.get("response_route") != expected_response_route
        ):
            raise PilotEvidenceError(
                "provider usage row differs from the frozen request/served-route "
                "profile"
            )
        if expected_runner_schema == RUNNER_SCHEMA_VERSION and (
            row.get("finish_reason") != "stop"
            or row.get("response_completed") is not True
            or "provider_error_details" not in row
            or row.get("provider_error_details") is not None
            or row.get("output_disposition") != "accepted"
            or row.get("provider_sdk_name") != expected_sdk_name
            or row.get("provider_sdk_version") != expected_sdk_version
            or "route_attestation_code" not in row
            or row.get("route_attestation_code")
            != expected_route_attestation
            or row.get("temperature_dispatch")
            != expected_temperature_dispatch
            or row.get("request_parameters")
            != sorted(expected_request_parameters)
        ):
            raise PilotEvidenceError(
                "current provider usage row lacks exact completion/SDK/request "
                "attestation"
            )


def _validate_standard_run_contract(
    contract: PilotContract,
    spec: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    provenance_git: Mapping[str, Any],
    raw_root: Path,
) -> None:
    """Rebuild and compare the exact registered runner/provider configuration.

    The ITT run spec alone does not prove that the sealed runner actually used
    the registered arm, utility, shock, proposal, or provider settings.  Reuse
    the execution-side pure config builder here, but force it to revalidate the
    sealed q-ref, Stage-0 selection, and preflight-p95 inputs first.
    """

    # Delayed imports avoid a module-import cycle: the orchestrator uses this
    # module only for terminal-summary sealing, while evidence publication runs
    # after both modules have been initialized.
    from .pilot_orchestrator import (  # pylint: disable=import-outside-toplevel
        DEFAULT_ENV_CONFIG,
        GitProvenance,
        _runner_p95_reservations,
        config_for_spec,
    )
    from .runner import (  # pylint: disable=import-outside-toplevel
        build_sealed_run_config,
    )

    try:
        paid = GitProvenance(
            git_tag=str(provenance_git["git_tag"]),
            head_commit=str(provenance_git["head_commit"]),
            tag_commit=str(provenance_git["tag_commit"]),
            tag_object_type=str(provenance_git["tag_object_type"]),
            worktree_clean=bool(provenance_git["worktree_clean"]),
            contract_binding=_mapping(
                provenance_git.get("contract_binding"),
                "paid contract binding",
            ),
            release_attestation=(
                _mapping(
                    provenance_git["release_attestation"],
                    "paid release attestation",
                )
                if provenance_git.get("release_attestation") is not None
                else None
            ),
        )
        expected_spec = next(
            item
            for item in contract.expand(stage=str(spec["stage_id"]))
            if item.run_id == spec["run_id"]
        )
        reservations = _runner_p95_reservations(
            contract,
            str(spec["model_id"]),
            raw_root=raw_root,
            paid=paid,
        )
        expected_base_config = config_for_spec(
            contract,
            expected_spec,
            raw_root=raw_root,
            paid_provenance=paid,
            verify_bound_inputs=True,
            preflight_p95_reservations=reservations,
        )
        expected_config = build_sealed_run_config(
            expected_base_config,
            env_config_source=DEFAULT_ENV_CONFIG,
        )
    except (KeyError, StopIteration, TypeError, ValueError) as exc:
        raise PilotEvidenceError(
            "could not reconstruct the registered scientific runner config"
        ) from exc
    except Exception as exc:
        raise PilotEvidenceError(
            f"registered runner inputs failed strict revalidation: {exc}"
        ) from exc

    actual_config = _json_copy(config)
    expected_config = _json_copy(expected_config)
    if actual_config != expected_config:
        differing = sorted(
            key
            for key in set(actual_config) | set(expected_config)
            if actual_config.get(key) != expected_config.get(key)
        )
        raise PilotEvidenceError(
            "sealed runner config differs from the registered arm/spec/shock/"
            f"utility/provider contract; fields={differing}"
        )

    profile = contract.provider_profiles[str(spec["model_id"])]
    provider_prefix = {
        "openai": "openai",
        "openrouter": "thirdparty",
        "ollama": "ollama",
    }.get(profile.transport)
    if provider_prefix is None:
        raise PilotEvidenceError(
            "scientific runner used a non-dispatchable provider transport"
        )
    expected_provider_model = f"{provider_prefix}/{profile.requested_model}"
    if summary.get("provider_model") != expected_provider_model:
        raise PilotEvidenceError(
            "sealed runner provider/model differs from its request profile"
        )

    _validate_provider_usage_rows(
        contract,
        spec,
        records.get("api_usage"),
        expected_runner_schema=str(
            config.get("schema_version", RUNNER_SCHEMA_VERSION)
        ),
    )


def _load_standard_run(
    contract: PilotContract,
    spec: Mapping[str, Any],
    run_dir: Path,
    *,
    raw_root: Path,
) -> dict[str, Any]:
    mode = str(spec.get("execution_mode", ""))
    if mode not in RUNNER_EXECUTION_MODES:
        raise PilotEvidenceError(
            f"execution mode {mode!r} requires a mode-specific terminal summary, "
            "not a verified-run manifest"
        )
    try:
        verification = verify_manifest(run_dir)
        result = load_verified_run_artifacts(run_dir)
    except (ManifestVerificationError, ValueError, TypeError) as exc:
        raise PilotEvidenceError(f"sealed run validation failed for {run_dir}: {exc}") from exc
    manifest = _strict_json_load(run_dir / "manifest.json")
    provenance = _strict_json_load(run_dir / "provenance.json")
    details = _mapping(provenance.get("details"), "run provenance details")
    if details.get("contract_id") != contract.contract_id:
        raise PilotEvidenceError("run provenance contract ID mismatch")
    if details.get("contract_sha256") != contract.canonical_hash:
        raise PilotEvidenceError("run provenance contract hash mismatch")
    if _json_copy(_mapping(details.get("run_spec"), "run provenance spec")) != _json_copy(
        spec
    ):
        raise PilotEvidenceError("sealed run spec differs from ITT ledger")
    paid = _mapping(details.get("git"), "paid git provenance")
    binding = _validate_binding(
        contract,
        git_tag=paid.get("git_tag"),
        resolved_git_commit=paid.get("head_commit"),
        binding=paid.get("contract_binding"),
    )
    if paid.get("tag_object_type") != "tag":
        raise PilotEvidenceError("sealed run does not bind an annotated tag")
    if paid.get("worktree_clean") is not True:
        raise PilotEvidenceError("sealed run was not produced from a clean worktree")
    if paid.get("head_commit") != paid.get("tag_commit"):
        raise PilotEvidenceError("sealed run HEAD does not equal the peeled pilot tag")
    manifest_git = _mapping(manifest.get("git"), "manifest git provenance")
    if (
        manifest_git.get("commit") != paid.get("head_commit")
        or manifest_git.get("dirty") is not False
    ):
        raise PilotEvidenceError("manifest git identity differs from paid provenance")
    summary = _mapping(result.summary, "sealed runner summary")
    validation = _mapping(result.validation_status, "runner validation")
    config = _mapping(result.config, "sealed run config")
    _validate_standard_run_contract(
        contract,
        spec,
        config=config,
        summary=summary,
        records=result.records,
        provenance_git=paid,
        raw_root=raw_root,
    )
    evidence = {
        "diagnostic_only": summary.get("diagnostic_only"),
        "scientific_evidence": summary.get("scientific_evidence"),
        "evidence_scope": summary.get("result_scope"),
        "provenance": details,
        "config": result.config,
    }
    eligible = _scope_gate(
        stage_id=str(spec["stage_id"]),
        evidence=evidence,
        source=run_dir,
    )
    if manifest.get("result", {}).get("complete") is not True:
        raise PilotEvidenceError("ledger marks complete but sealed runner result is incomplete")
    if validation.get("status") != "pass":
        raise PilotEvidenceError("ledger marks complete but runner validation did not pass")
    if eligible:
        if config.get("run_id") != spec["run_id"]:
            raise PilotEvidenceError("scientific runner run_id differs from contract cell")
        if config.get("pilot_contract_hash") != contract.canonical_hash:
            raise PilotEvidenceError("scientific runner config contract hash mismatch")
        if config.get("pilot_tag") != contract.implementation["required_git_tag"]:
            raise PilotEvidenceError("scientific runner config tag mismatch")
    analysis = summarize_run(
        result.records,
        max_labor_hours=float(config["max_labor_hours"]),
        schedule=contract.shocks[str(spec["shock_id"])]["schedule"],
    )
    total_discounted = sum(
        float(row["discounted_flow_utility"])
        for row in result.records.get("utility_ledger", ())
    )
    # Keep the registered analysis fields at the metric root so every standard
    # runner artifact and every terminal-summary artifact has the same query
    # paths (for example ``utility.shock_recovery_discounted``).  Nesting the
    # analysis under an extra key would silently make real runner cells
    # invisible to the A/C/cross-model gates.
    metrics = {
        **analysis,
        "runner": _json_copy(summary),
        "total_discounted_utility": total_discounted,
    }
    return {
        "artifact_kind": "verified-run-manifest",
        "artifact_sha256": verification.manifest_sha256,
        "binding": binding,
        "scientific_eligible": eligible,
        "metrics": metrics,
        "gate_evidence": {},
        "capability": {},
        "narrative": {},
    }


def _load_completed_artifact(
    contract: PilotContract,
    spec: Mapping[str, Any],
    *,
    raw_root: Path,
    artifact: Any,
) -> dict[str, Any]:
    path = _resolve_artifact(raw_root, artifact)
    mode = str(spec.get("execution_mode", ""))
    if path.is_dir():
        if mode not in RUNNER_EXECUTION_MODES:
            raise PilotEvidenceError(
                f"execution mode {mode!r} cannot publish a runner directory"
            )
        if not (path / "manifest.json").is_file():
            raise PilotEvidenceError(
                f"completed artifact directory has no manifest: {path}"
            )
        return _load_standard_run(
            contract,
            spec,
            path,
            raw_root=raw_root,
        )
    if path.name == "manifest.json":
        if mode not in RUNNER_EXECUTION_MODES:
            raise PilotEvidenceError(
                f"execution mode {mode!r} cannot publish a runner manifest"
            )
        return _load_standard_run(
            contract,
            spec,
            path.parent,
            raw_root=raw_root,
        )
    if path.suffix.lower() != ".json":
        raise PilotEvidenceError(f"unsupported completed artifact type: {path}")
    if mode not in TERMINAL_EXECUTION_MODES:
        raise PilotEvidenceError(
            f"execution mode {mode!r} requires a verified-run manifest"
        )
    return _load_terminal_summary(
        contract,
        spec,
        path,
        raw_root=raw_root,
    )


def _normalize_ledger(
    contract: PilotContract,
    ledger: Mapping[str, Any],
    *,
    raw_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    if ledger.get("schema_version") != PILOT_RUN_LEDGER_SCHEMA_VERSION:
        raise PilotEvidenceError("unsupported or missing pilot run ledger schema")
    if ledger.get("contract_hash") != contract.canonical_hash:
        raise PilotEvidenceError("pilot run ledger contract hash mismatch")
    observed = _mapping(ledger.get("runs"), "pilot run ledger runs")
    expected_specs = {spec.run_id: spec.to_dict() for spec in contract.expand()}
    unexpected = sorted(set(observed) - set(expected_specs))
    if unexpected:
        raise PilotEvidenceError(
            f"run ledger contains non-contract cells: {unexpected[:5]}"
        )

    rows: list[dict[str, Any]] = []
    bindings: set[str] = set()
    for run_id, spec in expected_specs.items():
        source = observed.get(run_id)
        if source is None:
            rows.append(
                {
                    **spec,
                    "status": "missing",
                    "failure": {
                        "error_type": "MissingLedgerRow",
                        "message": "preregistered cell has no ledger row",
                    },
                    "artifact_kind": None,
                    "artifact_sha256": None,
                    "scientific_eligible": False,
                    "metrics": {},
                    "gate_evidence": {},
                    "capability": {},
                    "narrative": {},
                }
            )
            continue
        source = _mapping(source, f"ledger run {run_id}")
        if _json_copy(_mapping(source.get("spec"), f"ledger spec {run_id}")) != spec:
            raise PilotEvidenceError(f"ledger spec mismatch for {run_id}")
        status = source.get("status")
        if status not in TERMINAL_STATUSES | KNOWN_NONTERMINAL_STATUSES:
            raise PilotEvidenceError(f"unknown ledger status {status!r} for {run_id}")
        row: dict[str, Any] = {
            **spec,
            "status": status,
            "failure": _json_copy(source.get("failure")),
            "artifact_kind": None,
            "artifact_sha256": None,
            "scientific_eligible": False,
            "metrics": {},
            "gate_evidence": {},
            "capability": {},
            "narrative": {},
        }
        if status == "complete":
            evidence = _load_completed_artifact(
                contract,
                spec,
                raw_root=raw_root,
                artifact=source.get("artifact"),
            )
            row.update(evidence)
            bindings.add(str(evidence["binding"]["resolved_git_commit"]))
        elif source.get("artifact") is not None:
            # Failure receipts may be retained in raw storage, but are not copied
            # into the validated package unless they use the terminal schema.
            artifact_path = _resolve_artifact(raw_root, source.get("artifact"))
            if artifact_path.is_file() and artifact_path.suffix == ".json":
                value = _strict_json_load(artifact_path)
                if value.get("schema_version") == PILOT_TERMINAL_SUMMARY_SCHEMA_VERSION:
                    evidence = _load_terminal_summary(
                        contract,
                        spec,
                        artifact_path,
                        raw_root=raw_root,
                    )
                    row["artifact_kind"] = evidence["artifact_kind"]
                    row["artifact_sha256"] = evidence["artifact_sha256"]
                    # A failed or stopped run never contributes effect metrics or
                    # gate evidence.  A sealed capability no-go may still expose
                    # its fixed-task denominator in the reviewer failure table.
                    if (
                        status == "capability-no-go"
                        and spec["stage_id"] == "capability-preflight"
                    ):
                        row["capability"] = evidence["capability"]
                    bindings.add(str(evidence["binding"]["resolved_git_commit"]))
        rows.append(row)

    if len(bindings) > 1:
        raise PilotEvidenceError(
            "validated artifacts resolve pilot-v1 to multiple commits"
        )
    common_commit = next(iter(bindings), None)
    counts = Counter(str(row["status"]) for row in rows)
    denominator = {
        "expected_count": len(expected_specs),
        "observed_ledger_count": len(observed),
        "all_rows_present": "missing" not in counts,
        "all_rows_terminal": not any(
            status in counts for status in KNOWN_NONTERMINAL_STATUSES
        ),
        "status_counts": dict(sorted(counts.items())),
        "all_completed_artifacts_validated": all(
            row["artifact_kind"] is not None
            for row in rows
            if row["status"] == "complete"
        ),
    }
    denominator["pass"] = all(
        (
            denominator["all_rows_present"],
            denominator["all_rows_terminal"],
            denominator["all_completed_artifacts_validated"],
        )
    )
    return rows, denominator, common_commit


def _validated_experiment_c_sensitivity(
    contract: PilotContract,
    *,
    raw_root: Path,
    rows: Sequence[Mapping[str, Any]],
    common_commit: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Strictly recompute the registered zero-API 3x3 C sensitivity artifact."""

    c_rows = [row for row in rows if row["stage_id"] == "experiment-c"]
    c_complete = bool(c_rows) and all(
        row["status"] == "complete" and row["scientific_eligible"] is True
        for row in c_rows
    )
    path = raw_root / "experiment-c" / "rule_sensitivity.json"
    if not c_complete:
        return None, {
            "pass": False,
            "available": False,
            "path": str(path),
            "reason": "Experiment C ITT cells are not all complete and eligible",
        }
    if path.is_symlink():
        raise PilotEvidenceError(
            "Experiment C sensitivity artifact cannot be a symlink"
        )

    # The execution-side verifier rebuilds every 3x3 cell from the exact five
    # sealed B-full manifests and the sealed Stage-0 threshold.  Importing it
    # lazily keeps the module dependency acyclic at import time.
    from .pilot_orchestrator import (  # pylint: disable=import-outside-toplevel
        _load_verified_experiment_c_sensitivity,
    )

    try:
        value = _load_verified_experiment_c_sensitivity(
            contract,
            raw_root=raw_root,
            paid=None,
        )
    except Exception as exc:
        raise PilotEvidenceError(
            f"Experiment C zero-API sensitivity failed revalidation: {exc}"
        ) from exc

    bindings = _mapping(
        value.get("bindings"),
        "Experiment C sensitivity bindings",
    )
    if common_commit is None or bindings.get("git_commit") != common_commit:
        raise PilotEvidenceError(
            "Experiment C sensitivity commit differs from the validated matrix"
        )
    if (
        value.get("schema_version")
        != PILOT_EXPERIMENT_C_SENSITIVITY_SCHEMA_VERSION
        or value.get("status") != "pass"
        or value.get("terminal") is not True
        or value.get("control_kind")
        != "zero-api-offline-rule-sensitivity"
        or value.get("provider_calls") != 0
        or value.get("descriptive_only") is not True
        or value.get("effectiveness_gate") is not False
    ):
        raise PilotEvidenceError(
            "Experiment C sensitivity scope/control markers are invalid"
        )

    sensitivity_contract = _mapping(
        contract.stop_go["experiment_c"]["zero_api_sensitivity"],
        "Experiment C sensitivity contract",
    )
    expected_weights = list(
        sensitivity_contract["alternative_success_weights"]
    )
    expected_outcomes = list(sensitivity_contract["outcome_definitions"])
    cells = value.get("aggregate_cells")
    if (
        value.get("alternative_success_weights") != expected_weights
        or value.get("outcome_definitions") != expected_outcomes
        or not isinstance(cells, Sequence)
        or isinstance(cells, (str, bytes))
        or len(cells) != len(expected_weights) * len(expected_outcomes)
    ):
        raise PilotEvidenceError(
            "Experiment C sensitivity differs from the frozen 3x3 grid"
        )
    observed_grid = {
        (
            cell.get("alternative_success_weight"),
            cell.get("outcome_definition"),
        )
        for cell in cells
        if isinstance(cell, Mapping)
    }
    expected_grid = {
        (weight, outcome)
        for weight in expected_weights
        for outcome in expected_outcomes
    }
    if observed_grid != expected_grid:
        raise PilotEvidenceError(
            "Experiment C sensitivity has missing or duplicate 3x3 cells"
        )

    expected_sources = {
        row["run_id"]: row["artifact_sha256"]
        for row in rows
        if (
            row["stage_id"] == "experiment-b"
            and row["model_id"] == "gpt52_main"
            and row["arm_id"] == "full"
            and row["status"] == "complete"
        )
    }
    source_rows = bindings.get("source_manifests")
    if (
        len(expected_sources) != 5
        or not isinstance(source_rows, Sequence)
        or isinstance(source_rows, (str, bytes))
        or len(source_rows) != 5
        or {
            str(source.get("run_id")): source.get("manifest_sha256")
            for source in source_rows
            if isinstance(source, Mapping)
        }
        != expected_sources
    ):
        raise PilotEvidenceError(
            "Experiment C sensitivity source manifests differ from aggregate rows"
        )

    integrity = _mapping(
        value.get("integrity"),
        "Experiment C sensitivity integrity",
    )
    return _json_copy(value), {
        "pass": True,
        "available": True,
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "content_sha256": integrity.get("content_sha256"),
        "provider_calls": 0,
        "descriptive_only": True,
        "effectiveness_gate": False,
        "source_run_count": value.get("source_run_count"),
        "grid_cell_count": len(cells),
    }


def _validated_release_controls(
    contract: PilotContract,
    *,
    raw_root: Path,
    rows: Sequence[Mapping[str, Any]],
    common_commit: str | None,
) -> dict[str, Any]:
    """Validate release, Stage-0, and budget controls without repairing them."""

    result: dict[str, Any] = {}

    release_path = raw_root / "release_attestation.json"
    release_reasons: list[str] = []
    try:
        release = _strict_json_load(release_path)
        unsigned = dict(release)
        claimed = unsigned.pop("attestation_sha256", None)
        local_tag = _mapping(release.get("local_tag"), "release local_tag")
        remote = _mapping(release.get("remote"), "release remote")
        actions = _mapping(
            release.get("github_actions"),
            "release github_actions",
        )
        run = _mapping(actions.get("run"), "release github run")
        jobs = actions.get("required_jobs")
        required_jobs = {
            "Python 3.12.7 / ubuntu-24.04",
            "Python 3.12.7 / macos-14",
        }
        observed_jobs = (
            {
                str(job.get("name"))
                for job in jobs
                if isinstance(job, Mapping)
                and job.get("status") == "completed"
                and job.get("conclusion") == "success"
            }
            if isinstance(jobs, Sequence)
            and not isinstance(jobs, (str, bytes))
            else set()
        )
        checks = {
            "schema_and_hash": (
                release.get("schema_version")
                == PILOT_RELEASE_ATTESTATION_SCHEMA_VERSION
                and release.get("status") == "pass"
                and claimed == canonical_sha256(unsigned)
            ),
            "commit_bound": (
                common_commit is not None
                and release.get("head_commit") == common_commit
                and local_tag.get("name")
                == contract.implementation["required_git_tag"]
                and local_tag.get("kind") == "annotated"
                and local_tag.get("peeled_commit") == common_commit
                and remote.get("tag_kind") == "annotated"
                and remote.get("tag_peeled_commit") == common_commit
                and remote.get("main_commit") == common_commit
            ),
            "exact_linux_macos_ci": (
                actions.get("workflow_file") == "verified-memory-ci.yml"
                and run.get("head_sha") == common_commit
                and run.get("head_branch") == "main"
                and run.get("status") == "completed"
                and run.get("conclusion") == "success"
                and observed_jobs == required_jobs
                and isinstance(jobs, Sequence)
                and len(jobs) == 2
            ),
        }
        release_pass = all(checks.values())
        release_reasons.extend(
            name for name, passed in checks.items() if not passed
        )
    except (PilotEvidenceError, TypeError, KeyError) as exc:
        checks = {}
        release_pass = False
        release_reasons.append(str(exc))
    result["release_attestation"] = {
        "pass": release_pass,
        "path": str(release_path),
        "checks": checks,
        "reasons": release_reasons,
    }

    stage0_path = raw_root / "stage0-calibration" / "stage0_selection.json"
    stage0_receipt_path = (
        raw_root / "stage0-calibration" / "stage_receipt.json"
    )
    stage0_reasons: list[str] = []
    try:
        selection = _strict_json_load(stage0_path)
        receipt = _strict_json_load(stage0_receipt_path)
        integrity = _mapping(
            selection.get("integrity"),
            "Stage-0 selection integrity",
        )
        bindings = _mapping(
            selection.get("bindings"),
            "Stage-0 selection bindings",
        )
        source_rows = bindings.get("source_manifests")
        expected_specs = {
            spec.run_id: spec
            for spec in contract.expand(stage="stage0-calibration")
        }
        aggregate_rows = {
            str(row["run_id"]): row
            for row in rows
            if row["stage_id"] == "stage0-calibration"
        }
        source_valid = (
            isinstance(source_rows, Sequence)
            and not isinstance(source_rows, (str, bytes))
            and len(source_rows) == len(expected_specs)
        )
        observed_ids: set[str] = set()
        if source_valid:
            for source in source_rows:
                if not isinstance(source, Mapping):
                    source_valid = False
                    break
                run_id = str(source.get("run_id", ""))
                row = aggregate_rows.get(run_id)
                spec = expected_specs.get(run_id)
                if (
                    run_id in observed_ids
                    or row is None
                    or spec is None
                    or row.get("status") != "complete"
                    or row.get("artifact_kind") != "verified-run-manifest"
                    or row.get("artifact_sha256")
                    != source.get("manifest_sha256")
                    or source.get("utility_profile_id")
                    != spec.utility_profile_id
                    or source.get("environment_seed")
                    != spec.environment_seed
                ):
                    source_valid = False
                    break
                observed_ids.add(run_id)
        stage0_checks = {
            "sealed_selection": (
                selection.get("schema_version")
                == "finevo-stage0-selection-v1"
                and selection.get("contract_sha256")
                == contract.canonical_hash
                and integrity.get("canonicalization")
                == "json-sort-keys-utf8-v1"
                and integrity.get("content_sha256")
                == _bound_artifact_hash(selection)
                and bindings.get("contract_sha256")
                == contract.canonical_hash
                and bindings.get("git_tag")
                == contract.implementation["required_git_tag"]
                and bindings.get("git_commit") == common_commit
            ),
            "complete_source_matrix": (
                source_valid and observed_ids == set(expected_specs)
            ),
            "selection_is_outcome_blind": (
                isinstance(selection.get("selected_profile_id"), str)
                and isinstance(selection.get("selected_utility"), Mapping)
                and selection.get("outcome_fields_used") == []
            ),
            "stage_receipt_go": (
                receipt.get("schema_version")
                == PILOT_STAGE_RECEIPT_SCHEMA_VERSION
                and receipt.get("contract_sha256")
                == contract.canonical_hash
                and receipt.get("stage_id") == "stage0-calibration"
                and receipt.get("status") == "complete"
                and receipt.get("go") is True
                and receipt.get("terminal") is True
                and receipt.get("registered_run_count")
                == len(expected_specs)
                and receipt.get("complete_cell_count")
                == len(expected_specs)
            ),
        }
        stage0_pass = all(stage0_checks.values())
        stage0_reasons.extend(
            name for name, passed in stage0_checks.items() if not passed
        )
    except (PilotEvidenceError, TypeError, KeyError) as exc:
        stage0_checks = {}
        stage0_pass = False
        stage0_reasons.append(str(exc))
    result["stage0_selection"] = {
        "pass": stage0_pass,
        "path": str(stage0_path),
        "stage_receipt_path": str(stage0_receipt_path),
        "checks": stage0_checks,
        "reasons": stage0_reasons,
    }

    budget_path = raw_root / "budget_ledger.json"
    budget_reasons: list[str] = []
    raw_storage_bytes = sum(
        path.stat().st_size
        for path in raw_root.rglob("*")
        if path.is_file()
    )
    try:
        budget = _strict_json_load(budget_path)
        caps = _mapping(budget.get("caps"), "budget caps")
        budget_runs = _mapping(budget.get("runs"), "budget runs")
        expected_caps = {
            "total_usd": float(contract.budgets["total_usd"]),
            "max_completions": int(
                contract.budgets["max_provider_completions"]
            ),
            "completion_scope": str(contract.budgets["completion_scope"]),
            "max_storage_bytes": int(contract.budgets["max_storage_bytes"]),
            "stage_usd_caps": dict(contract.budgets["stage_usd_caps"]),
            "automatic_reserve_usd": float(
                contract.budgets["automatic_reserve_usd"]
            ),
            "dispatchable_usd": float(contract.budgets["total_usd"])
            - float(contract.budgets["automatic_reserve_usd"]),
        }
        expected_standard_ids = {
            spec.run_id
            for spec in contract.expand()
            if spec.execution_mode != "checkpoint_continuation"
        }
        expected_d_ids = {
            f"{contract.contract_id}--experiment-d--gpt52_main--"
            f"checkpoint-group--s{seed}"
            for seed in contract.seeds["sets"]["main"]
        }
        expected_budget_ids = expected_standard_ids | expected_d_ids
        totals = {
            "cost_usd": 0.0,
            "completions": 0,
            "storage_bytes": 0,
        }
        by_stage = {
            str(stage): 0.0 for stage in expected_caps["stage_usd_caps"]
        }
        # A capability, projection, integrity, or budget no-go can terminalize
        # registered ITT cells before a provider reservation exists.  The
        # durable budget ledger must therefore be an exact, finalized account
        # of dispatched units, not a fabricated row for every no-dispatch cell.
        rows_valid = set(budget_runs).issubset(expected_budget_ids)
        if rows_valid:
            for run_id, row in budget_runs.items():
                if not isinstance(row, Mapping):
                    rows_valid = False
                    break
                reservation = row.get("reservation")
                actual = row.get("actual")
                stage = str(row.get("stage_bucket"))
                if (
                    row.get("status")
                    not in {
                        "complete",
                        "failed",
                        "budget-stopped",
                        "integrity-stopped",
                    }
                    or not isinstance(reservation, Mapping)
                    or not isinstance(actual, Mapping)
                    or stage not in by_stage
                    or reservation.get("run_id") != run_id
                    or reservation.get("stage_bucket") != stage
                ):
                    rows_valid = False
                    break
                for field in ("cost_usd", "completions", "storage_bytes"):
                    if not _is_finite_scalar(actual.get(field)):
                        rows_valid = False
                        break
                    if (
                        float(actual[field])
                        > float(reservation[field]) + 1e-12
                        and row.get("status") != "integrity-stopped"
                    ):
                        rows_valid = False
                        break
                if not rows_valid:
                    break
                if (
                    row.get("status") == "integrity-stopped"
                    and not isinstance(row.get("failure"), Mapping)
                ):
                    rows_valid = False
                    break
                totals["cost_usd"] += float(actual["cost_usd"])
                totals["completions"] += int(actual["completions"])
                totals["storage_bytes"] += int(actual["storage_bytes"])
                by_stage[stage] += float(actual["cost_usd"])

        artifact_backed_standard_ids = {
            str(row["run_id"])
            for row in rows
            if (
                row["execution_mode"] != "checkpoint_continuation"
                and row.get("artifact_kind") is not None
            )
        }
        artifact_backed_d_ids = {
            f"{contract.contract_id}--experiment-d--gpt52_main--"
            f"checkpoint-group--s{int(row['environment_seed'])}"
            for row in rows
            if (
                row["execution_mode"] == "checkpoint_continuation"
                and row.get("artifact_kind") is not None
            )
        }
        dispatched_artifacts_accounted = (
            artifact_backed_standard_ids | artifact_backed_d_ids
        ).issubset(set(budget_runs))
        within_caps = bool(
            totals["cost_usd"] <= expected_caps["dispatchable_usd"] + 1e-12
            and totals["completions"] <= expected_caps["max_completions"]
            and totals["storage_bytes"] <= expected_caps["max_storage_bytes"]
            and all(
                by_stage[stage]
                <= float(expected_caps["stage_usd_caps"][stage]) + 1e-12
                for stage in by_stage
            )
            and raw_storage_bytes <= expected_caps["max_storage_bytes"]
        )
        budget_checks = {
            "schema_and_contract": (
                budget.get("schema_version") == PILOT_BUDGET_SCHEMA_VERSION
                and budget.get("contract_hash") == contract.canonical_hash
            ),
            "exact_frozen_caps": dict(caps) == expected_caps,
            "valid_finalized_dispatch_units": rows_valid,
            "all_artifact_backed_dispatches_accounted": (
                dispatched_artifacts_accounted
            ),
            "actual_totals_within_caps": within_caps,
        }
        budget_pass = all(budget_checks.values())
        budget_reasons.extend(
            name for name, passed in budget_checks.items() if not passed
        )
    except (PilotEvidenceError, TypeError, KeyError, ValueError) as exc:
        budget_checks = {}
        budget_pass = False
        budget_reasons.append(str(exc))
        totals = {}
        by_stage = {}
    result["budget_ledger"] = {
        "pass": budget_pass,
        "path": str(budget_path),
        "checks": budget_checks,
        "actual_totals": totals,
        "actual_stage_cost_usd": by_stage,
        "raw_root_storage_bytes": raw_storage_bytes,
        "reasons": budget_reasons,
    }
    result["pass"] = bool(release_pass and stage0_pass and budget_pass)
    return result


def _dig(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _metric(row: Mapping[str, Any], path: str) -> float | None:
    candidates = (
        f"metrics.analysis.{path}",
        f"metrics.{path}",
    )
    for candidate in candidates:
        value = _dig(row, candidate)
        if _is_finite_scalar(value):
            return float(value)
    return None


def _scientific_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    model: str | None = None,
    arm: str | None = None,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row["stage_id"] == stage
        and (model is None or row["model_id"] == model)
        and (arm is None or row["arm_id"] == arm)
        and row["status"] == "complete"
        and row["scientific_eligible"] is True
    ]


def _experiment_a_gate(
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    model = "gpt52_main"
    by_arm = {
        arm: {
            int(row["environment_seed"]): row
            for row in _scientific_rows(
                rows, stage="experiment-a", model=model, arm=arm
            )
        }
        for arm in ("no-context", "prompt-only", "retrieval-only", "full")
    }
    full_rows = by_arm["full"]
    control_rows = by_arm["prompt-only"]
    expected = tuple(int(value) for value in contract.seeds["sets"]["main"])
    usable = [
        seed
        for seed in expected
        if seed in full_rows
        and seed in control_rows
        and _metric(full_rows[seed], "utility.shock_recovery_discounted")
        is not None
        and _metric(control_rows[seed], "utility.shock_recovery_discounted")
        is not None
    ]
    primary: dict[str, Any] | None = None
    gate: dict[str, Any] | None = None
    if usable:
        primary = paired_delta_summary(
            {
                seed: float(
                    _metric(
                        full_rows[seed],
                        "utility.shock_recovery_discounted",
                    )
                )
                for seed in usable
            },
            {
                seed: float(
                    _metric(
                        control_rows[seed],
                        "utility.shock_recovery_discounted",
                    )
                )
                for seed in usable
            },
        )
        gate = retrieval_effect_gate(primary)

    def paired_secondary(path: str) -> dict[str, Any]:
        treatment = {
            seed: _metric(full_rows[seed], path)
            for seed in expected
            if seed in full_rows
        }
        control = {
            seed: _metric(control_rows[seed], path)
            for seed in expected
            if seed in control_rows
        }
        paired = [
            seed
            for seed in expected
            if treatment.get(seed) is not None and control.get(seed) is not None
        ]
        summary = (
            paired_delta_summary(
                {seed: float(treatment[seed]) for seed in paired},
                {seed: float(control[seed]) for seed in paired},
                positive_direction=False,
            )
            if paired
            else None
        )
        return {
            "treatment_full_by_seed": {
                str(seed): treatment.get(seed) for seed in expected
            },
            "control_prompt_only_by_seed": {
                str(seed): control.get(seed) for seed in expected
            },
            "paired_summary_full_minus_prompt_only": summary,
            "paired_seeds": paired,
            "unpaired_or_censored_seeds": [
                seed for seed in expected if seed not in paired
            ],
            "lower_is_better": True,
        }

    secondary_paired_metrics = {
        "utility_deficit_auc": paired_secondary(
            "utility.utility_deficit_auc"
        ),
        "recovery_periods_to_within_10pct_for_two": paired_secondary(
            "utility.recovery_periods_to_within_10pct_for_two"
        ),
    }

    corroborating_seeds = [
        seed
        for seed in expected
        if seed in by_arm["retrieval-only"]
        and seed in by_arm["no-context"]
        and _metric(
            by_arm["retrieval-only"][seed],
            "utility.shock_recovery_discounted",
        )
        is not None
        and _metric(
            by_arm["no-context"][seed],
            "utility.shock_recovery_discounted",
        )
        is not None
    ]
    corroborating = (
        paired_delta_summary(
            {
                seed: float(
                    _metric(
                        by_arm["retrieval-only"][seed],
                        "utility.shock_recovery_discounted",
                    )
                )
                for seed in corroborating_seeds
            },
            {
                seed: float(
                    _metric(
                        by_arm["no-context"][seed],
                        "utility.shock_recovery_discounted",
                    )
                )
                for seed in corroborating_seeds
            },
        )
        if corroborating_seeds
        else None
    )

    def route_flag_ok(arm: str, prompt: bool, retrieval: bool) -> bool:
        arm_rows = by_arm[arm].values()
        expected_rows = contract.stage("experiment-a").num_agents * contract.stage(
            "experiment-a"
        ).episode_length
        return len(by_arm[arm]) >= 4 and all(
            _metric(row, "memory.context_to_prompt_count")
            == (expected_rows if prompt else 0)
            and _metric(row, "memory.context_to_retrieval_count")
            == (expected_rows if retrieval else 0)
            for row in arm_rows
        )

    route_checks = {
        "no_context_routes_neither_channel": route_flag_ok(
            "no-context", False, False
        ),
        "prompt_only_routes_prompt_channel": route_flag_ok(
            "prompt-only", True, False
        ),
        "retrieval_only_routes_retrieval_channel": route_flag_ok(
            "retrieval-only", False, True
        ),
        "full_routes_both_channels": route_flag_ok("full", True, True),
    }

    def trace_index(row: Mapping[str, Any]) -> dict[tuple[int, int], set[str]]:
        trace = _dig(row, "metrics.memory.route_trace_top5")
        if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
            return {}
        indexed: dict[tuple[int, int], set[str]] = {}
        for item in trace:
            if not isinstance(item, Mapping):
                return {}
            key = (int(item["decision_t"]), int(item["agent_id"]))
            if key in indexed:
                return {}
            values = item.get("retrieved_episode_ids", ())
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                return {}
            indexed[key] = {str(value) for value in values}
        return indexed

    expected_trace_rows = (
        contract.stage("experiment-a").num_agents
        * contract.stage("experiment-a").episode_length
    )
    expected_phases = {
        str(item["phase"])
        for item in contract.shocks[
            str(contract.stage("experiment-a").shock_id)
        ]["schedule"]
    }

    def retrieval_trace_complete(row: Mapping[str, Any]) -> bool:
        indexed = trace_index(row)
        return (
            len(indexed) == expected_trace_rows
            and sum(len(values) for values in indexed.values()) > 0
        )

    def phase_relevance_complete(row: Mapping[str, Any]) -> bool:
        value = _dig(row, "metrics.memory.route_relevance_at_5")
        if not isinstance(value, Mapping) or value.get("k") != 5:
            return False
        by_phase = value.get("by_phase")
        if not isinstance(by_phase, Mapping) or set(by_phase) != expected_phases:
            return False
        for phase in expected_phases:
            phase_row = by_phase.get(phase)
            if not isinstance(phase_row, Mapping):
                return False
            retrieved = phase_row.get("retrieved")
            relevant = phase_row.get("relevant")
            relevance = phase_row.get("relevance")
            if (
                not isinstance(retrieved, int)
                or retrieved <= 0
                or not isinstance(relevant, int)
                or relevant < 0
                or relevant > retrieved
                or not _is_finite_scalar(relevance)
                or not 0 <= float(relevance) <= 1
            ):
                return False
        return True

    retrieval_coverage = {
        arm: {
            str(seed): {
                "trace_complete": retrieval_trace_complete(row),
                "phase_relevance_complete": phase_relevance_complete(row),
            }
            for seed, row in sorted(by_arm[arm].items())
        }
        for arm in ("retrieval-only", "full")
    }
    retrieval_coverage_pass = all(
        len(by_arm[arm]) >= 4
        and all(
            item["trace_complete"] and item["phase_relevance_complete"]
            for item in retrieval_coverage[arm].values()
        )
        for arm in ("retrieval-only", "full")
    )

    topk_overlap_by_seed: dict[str, Any] = {}
    overlap_seeds = sorted(set(full_rows) & set(by_arm["retrieval-only"]))
    for seed in overlap_seeds:
        left = trace_index(full_rows[seed])
        right = trace_index(by_arm["retrieval-only"][seed])
        if not left or set(left) != set(right):
            topk_overlap_by_seed[str(seed)] = None
            continue
        scores = []
        for key in sorted(left):
            union = left[key] | right[key]
            scores.append(1.0 if not union else len(left[key] & right[key]) / len(union))
        topk_overlap_by_seed[str(seed)] = mean(scores) if scores else None

    phase_relevance_at_5 = {
        arm: {
            str(seed): _dig(row, "metrics.memory.route_relevance_at_5")
            for seed, row in sorted(seed_rows.items())
        }
        for arm, seed_rows in by_arm.items()
    }
    action_distributions = {
        arm: {
            str(seed): {
                "labor_hours_counts": _dig(
                    row, "metrics.actions.labor_hours_counts"
                ),
                "consumption_rate_counts": _dig(
                    row, "metrics.actions.consumption_rate_counts"
                ),
            }
            for seed, row in sorted(seed_rows.items())
        }
        for arm, seed_rows in by_arm.items()
    }
    complete = (
        len(usable)
        >= int(contract.stop_go["experiment_a"]["complete_pairs_min"])
        and len(corroborating_seeds) >= 4
        and all(route_checks.values())
        and retrieval_coverage_pass
    )
    supported = bool(
        complete
        and gate is not None
        and gate["support_retrieval_effect"]
    )
    reasons = []
    if len(usable) < 4:
        reasons.append("fewer than four validated paired seeds")
    if len(corroborating_seeds) < 4:
        reasons.append("fewer than four validated corroborating paired seeds")
    if not all(route_checks.values()):
        reasons.append("route manipulation checks are incomplete or failed")
    if not retrieval_coverage_pass:
        reasons.append(
            "retrieval traces are empty/incomplete or phase relevance@5 is incomplete"
        )
    if gate is not None and not gate["support_retrieval_effect"]:
        reasons.append("registered direction or median relative-effect gate failed")
    return {
        "status": "supported" if supported else "no-go",
        "scientific_evidence_complete": complete,
        "support_retrieval_effect": supported,
        "expected_seeds": list(expected),
        "usable_paired_seeds": usable,
        "primary_contrast": primary,
        "corroborating_contrast": corroborating,
        "secondary_paired_metrics": secondary_paired_metrics,
        "threshold_gate": gate,
        "route_manipulation_checks": route_checks,
        "retrieval_trace_and_phase_coverage": retrieval_coverage,
        "retrieval_trace_and_phase_coverage_pass": retrieval_coverage_pass,
        "phase_relevance_at_5": phase_relevance_at_5,
        "full_vs_retrieval_only_top5_overlap": topk_overlap_by_seed,
        "action_distributions": action_distributions,
        "claim_action": (
            "retain the narrow retrieval-effect claim"
            if supported
            else "retain route traceability only"
        ),
        "reasons": reasons,
    }


def _delta_descriptives(values: Mapping[int, float]) -> dict[str, Any] | None:
    if not values:
        return None
    normalized = {
        int(seed): float(value) for seed, value in sorted(values.items())
    }
    ordered = list(normalized.values())
    return {
        "raw_paired_deltas": {
            str(seed): value for seed, value in normalized.items()
        },
        "mean": mean(ordered),
        "median": median(ordered),
        "range": [min(ordered), max(ordered)],
        "positive_direction_count": sum(value > 0 for value in ordered),
        "negative_direction_count": sum(value < 0 for value in ordered),
        "zero_count": sum(value == 0 for value in ordered),
        "pair_count": len(ordered),
    }


def _experiment_b_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Preserve the five-arm architecture comparison without ranking arms."""

    output: dict[str, Any] = {}
    full = {
        int(row["environment_seed"]): row
        for row in _scientific_rows(
            rows,
            stage="experiment-b",
            model="gpt52_main",
            arm="full",
        )
    }
    for arm in (
        "no-memory",
        "episodic-only",
        "semantic-only",
        "unverified-dual",
        "full",
    ):
        arm_rows = sorted(
            (
                row
                for row in rows
                if row["stage_id"] == "experiment-b"
                and row["model_id"] == "gpt52_main"
                and row["arm_id"] == arm
            ),
            key=lambda row: int(row["environment_seed"]),
        )
        seed_rows = {}
        for row in arm_rows:
            seed_rows[str(row["environment_seed"])] = {
                "status": row["status"],
                "failure": _json_copy(row["failure"]),
                "scientific_eligible": row["scientific_eligible"],
                "utility": _json_copy(_dig(row, "metrics.utility")),
                "total_discounted_utility": _dig(
                    row, "metrics.total_discounted_utility"
                ),
                "actions": _json_copy(_dig(row, "metrics.actions")),
                "retrieval_and_rules": _json_copy(_dig(row, "metrics.memory")),
                "rule_reliability": _json_copy(
                    _dig(row, "metrics.rule_reliability")
                ),
            }
        output[arm] = {
            "registered_seeds": len(arm_rows),
            "status_counts": dict(
                sorted(Counter(str(row["status"]) for row in arm_rows).items())
            ),
            "seeds": seed_rows,
            "paired_vs_full": {
                metric: _delta_descriptives(
                    {
                        seed: float(left) - float(right)
                        for seed, row in (
                            (
                                int(item["environment_seed"]),
                                item,
                            )
                            for item in arm_rows
                            if item["status"] == "complete"
                            and item["scientific_eligible"] is True
                            and int(item["environment_seed"]) in full
                        )
                        if (
                            left := _metric(row, path)
                        )
                        is not None
                        and (
                            right := _metric(full[seed], path)
                        )
                        is not None
                    }
                )
                for metric, path in {
                    "shock_recovery_discounted_utility": (
                        "utility.shock_recovery_discounted"
                    ),
                    "total_discounted_utility": "total_discounted_utility",
                }.items()
            },
        }
    return {
        "comparison_type": "descriptive_preregistered_architecture_arms",
        "selection_rule": "do not select a winner by wealth or any single outcome",
        "arms": output,
    }


def _unique_gate_evidence(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    stages: frozenset[str],
) -> Mapping[str, Any] | None:
    candidates: list[Mapping[str, Any]] = []
    hashes: set[str] = set()
    for row in rows:
        if (
            row.get("stage_id") not in stages
            or row.get("status") != "complete"
            or row.get("scientific_eligible") is not True
        ):
            continue
        value = _dig(row, f"gate_evidence.{key}")
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise PilotEvidenceError(f"{key} gate evidence must be an object")
        digest = canonical_sha256(value)
        if digest not in hashes:
            candidates.append(value)
            hashes.add(digest)
    if not candidates:
        return None
    if len(candidates) != 1:
        raise PilotEvidenceError(f"conflicting sealed {key} gate evidence")
    return candidates[0]


def _seed_metric_map(value: Any, name: str) -> dict[int, dict[str, float]]:
    source = _mapping(value, name)
    result: dict[int, dict[str, float]] = {}
    for seed, metrics in source.items():
        if not str(seed).isdigit():
            raise PilotEvidenceError(f"{name} seed keys must be integers")
        metrics = _mapping(metrics, f"{name}.{seed}")
        normalized: dict[str, float] = {}
        for metric, item in metrics.items():
            if not _is_finite_scalar(item):
                raise PilotEvidenceError(f"{name}.{seed}.{metric} must be finite")
            normalized[str(metric)] = float(item)
        result[int(seed)] = normalized
    return result


def _experiment_c_gate(
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = set(int(value) for value in contract.seeds["sets"]["main"])

    def indexed(stage: str, arm: str) -> dict[int, Mapping[str, Any]]:
        return {
            int(row["environment_seed"]): row
            for row in _scientific_rows(
                rows,
                stage=stage,
                model="gpt52_main",
                arm=arm,
            )
        }

    admission = indexed("experiment-c", "verified-error-candidate")
    verified_error = indexed("experiment-c", "verified-error-forced")
    unverified_error = indexed("experiment-c", "unverified-error-forced")
    verified_control = indexed("experiment-b", "full")
    unverified_control = indexed("experiment-b", "unverified-dual")
    candidate_seeds = sorted(expected & set(admission))
    forced_seeds = sorted(
        expected
        & set(verified_error)
        & set(unverified_error)
        & set(verified_control)
        & set(unverified_control)
    )
    if not candidate_seeds and not forced_seeds:
        return {
            "status": "no-go",
            "scientific_evidence_complete": False,
            "support_rule_reliability": False,
            "claim_action": "withdraw or narrow the rule-reliability claim",
            "reasons": ["no sealed Experiment C and reused Experiment B evidence"],
        }

    candidate_pairs: dict[int, dict[str, float]] = {}
    for seed in candidate_seeds:
        verified_value = _dig(
            admission[seed],
            "metrics.rule_reliability.false_rule_ever_active",
        )
        unverified_value = _dig(
            admission[seed],
            "metrics.rule_reliability.unverified_false_rule_ever_active",
        )
        verified_active = (
            float(verified_value)
            if isinstance(verified_value, bool) or _is_finite_scalar(verified_value)
            else None
        )
        unverified_active = (
            float(unverified_value)
            if isinstance(unverified_value, bool)
            or _is_finite_scalar(unverified_value)
            else None
        )
        if verified_active is None or unverified_active is None:
            continue
        candidate_pairs[seed] = {
            "verified_false_activation": verified_active,
            "unverified_false_activation": unverified_active,
        }

    forced_pairs: dict[int, dict[str, float]] = {}
    forced_unit_rows: dict[int, dict[str, Any]] = {}
    natural_proposal_audit: list[dict[str, Any]] = []

    def unit_rows(
        row: Mapping[str, Any],
        *,
        seed: int,
        arm: str,
    ) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]] | None:
        value = _dig(row, "metrics.rule_reliability.by_agent_rule_family")
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
        ):
            return None
        injected: list[Mapping[str, Any]] = []
        natural: list[Mapping[str, Any]] = []
        identities: set[tuple[int, str]] = set()
        for unit in value:
            if not isinstance(unit, Mapping):
                return None
            agent_id = unit.get("agent_id")
            family_id = unit.get("rule_family_id")
            identity = (agent_id, str(family_id))
            if (
                not isinstance(agent_id, int)
                or agent_id not in range(contract.stage("experiment-c").num_agents)
                or not isinstance(family_id, str)
                or not family_id
                or identity in identities
                or unit.get("seed") not in {seed, None}
                or not isinstance(unit.get("ever_active"), bool)
                or not isinstance(unit.get("active_exposure_steps"), int)
            ):
                return None
            identities.add(identity)
            if unit.get("injected") is True and unit.get("source") == "injected":
                injected.append(unit)
            elif unit.get("natural") is True and unit.get("source") == "natural":
                natural.append(unit)
            else:
                return None
        if len(injected) != contract.stage("experiment-c").num_agents:
            return None
        for unit in natural:
            natural_proposal_audit.append(
                {
                    "seed": seed,
                    "arm": arm,
                    **_json_copy(unit),
                }
            )
        return injected, natural

    for seed in forced_seeds:
        verified_exposure = _metric(
            verified_error[seed],
            "rule_reliability.active_exposure_steps",
        )
        unverified_exposure = _metric(
            unverified_error[seed],
            "rule_reliability.active_exposure_steps",
        )
        utility_values = {
            "verified_error": _metric(
                verified_error[seed], "total_discounted_utility"
            ),
            "unverified_error": _metric(
                unverified_error[seed], "total_discounted_utility"
            ),
            "verified_control": _metric(
                verified_control[seed], "total_discounted_utility"
            ),
            "unverified_control": _metric(
                unverified_control[seed], "total_discounted_utility"
            ),
        }
        if (
            verified_exposure is None
            or unverified_exposure is None
            or any(value is None for value in utility_values.values())
        ):
            continue
        verified_units = unit_rows(
            verified_error[seed],
            seed=seed,
            arm="verified-error-forced",
        )
        unverified_units = unit_rows(
            unverified_error[seed],
            seed=seed,
            arm="unverified-error-forced",
        )
        if verified_units is None or unverified_units is None:
            continue
        verified_signed_loss = (
            float(utility_values["verified_control"])
            - float(utility_values["verified_error"])
        )
        unverified_signed_loss = (
            float(utility_values["unverified_control"])
            - float(utility_values["unverified_error"])
        )
        forced_pairs[seed] = {
            "verified_harmful_exposure": verified_exposure,
            "unverified_harmful_exposure": unverified_exposure,
            # Retain signed contrasts for audit, but a treatment that improves
            # utility cannot be relabelled as a negative "loss".
            "verified_signed_control_minus_error": verified_signed_loss,
            "unverified_signed_control_minus_error": unverified_signed_loss,
            "verified_cumulative_utility_loss": max(0.0, verified_signed_loss),
            "unverified_cumulative_utility_loss": max(
                0.0, unverified_signed_loss
            ),
            "verified_harmful_to_retirement_delay": (
                _metric(
                    verified_error[seed],
                    "rule_reliability.harmful_to_retirement_delay",
                )
            ),
            "unverified_harmful_to_retirement_delay": (
                _metric(
                    unverified_error[seed],
                    "rule_reliability.harmful_to_retirement_delay",
                )
            ),
        }
        forced_unit_rows[seed] = {
            "verified_injected_units": _json_copy(verified_units[0]),
            "unverified_injected_units": _json_copy(unverified_units[0]),
            "verified_natural_units": _json_copy(verified_units[1]),
            "unverified_natural_units": _json_copy(unverified_units[1]),
        }

    directions = {
        "false_activation": sum(
            value["verified_false_activation"]
            < value["unverified_false_activation"]
            for value in candidate_pairs.values()
        ),
        "harmful_exposure": sum(
            value["verified_harmful_exposure"]
            < value["unverified_harmful_exposure"]
            for value in forced_pairs.values()
        ),
        "cumulative_utility_loss": sum(
            value["unverified_cumulative_utility_loss"] > 0
            and value["verified_cumulative_utility_loss"]
            < value["unverified_cumulative_utility_loss"]
            for value in forced_pairs.values()
        ),
    }
    positive_unverified_loss_count = sum(
        value["unverified_cumulative_utility_loss"] > 0
        for value in forced_pairs.values()
    )
    candidate_rejected = len(candidate_pairs) >= 4 and all(
        value["verified_false_activation"] == 0
        for value in candidate_pairs.values()
    )
    unit_rows_complete = len(forced_unit_rows) >= 4
    complete = (
        len(candidate_pairs) >= 4
        and len(forced_pairs) >= 4
        and unit_rows_complete
    )
    supported = bool(
        complete
        and candidate_rejected
        and all(count >= 4 for count in directions.values())
        and positive_unverified_loss_count >= 4
    )
    reasons = []
    if not candidate_rejected:
        reasons.append("unsupported candidate was not demonstrably rejected")
    if not complete:
        reasons.append("fewer than four complete paired seeds or missing primary metrics")
    if not unit_rows_complete:
        reasons.append(
            "fewer than four forced-error seed pairs retain complete "
            "seed-agent-rule-family units"
        )
    for metric, count in directions.items():
        if count < 4:
            reasons.append(f"verifier did not lower {metric} in four paired seeds")
    if positive_unverified_loss_count < 4:
        reasons.append(
            "unverified error arm did not incur positive cumulative utility "
            "loss in four paired seeds"
        )
    return {
        "status": "supported" if supported else "no-go",
        "scientific_evidence_complete": complete,
        "support_rule_reliability": supported,
        "candidate_admission_rejected": candidate_rejected,
        "candidate_admission_pairs": {
            str(seed): value for seed, value in sorted(candidate_pairs.items())
        },
        "forced_active_pairs": {
            str(seed): value for seed, value in sorted(forced_pairs.items())
        },
        "forced_active_seed_agent_rule_family_units": {
            str(seed): value for seed, value in sorted(forced_unit_rows.items())
        },
        "natural_proposal_descriptive_audit": natural_proposal_audit,
        "usable_candidate_seeds": sorted(candidate_pairs),
        "usable_forced_active_seeds": sorted(forced_pairs),
        "same_direction_counts": directions,
        "positive_unverified_utility_loss_seed_count": (
            positive_unverified_loss_count
        ),
        "claim_action": (
            "retain only the registered rule-reliability claim"
            if supported
            else "withdraw or narrow the rule-reliability claim"
        ),
        "reasons": reasons,
    }


def _direction_and_null_gate(
    treatment_deltas: Mapping[int, float],
    matched_null_deltas: Mapping[int, float],
) -> dict[str, Any]:
    """Qualify a downstream outcome without pretending it has an action bin."""

    if (
        not treatment_deltas
        or set(treatment_deltas) != set(matched_null_deltas)
    ):
        return {
            "passes": False,
            "checks": {
                "at_least_four_complete_pairs": (
                    len(treatment_deltas) >= 4
                    and set(treatment_deltas) == set(matched_null_deltas)
                ),
                "at_least_four_same_direction": False,
                "exceeds_matched_null": False,
            },
            "treatment_deltas": {
                str(seed): float(value)
                for seed, value in sorted(treatment_deltas.items())
            },
            "matched_null_deltas": {
                str(seed): float(value)
                for seed, value in sorted(matched_null_deltas.items())
            },
            "matched_null_max_abs": None,
            "median_abs_treatment_delta": None,
            "incomplete_reason": (
                "missing outcome metrics or treatment/null seed mismatch"
            ),
        }
    values = {
        int(seed): float(value)
        for seed, value in treatment_deltas.items()
    }
    nulls = {
        int(seed): float(value)
        for seed, value in matched_null_deltas.items()
    }
    positive = sum(value > 0 for value in values.values())
    negative = sum(value < 0 for value in values.values())
    null_max = max(abs(value) for value in nulls.values())
    effect_magnitude = median(abs(value) for value in values.values())
    checks = {
        "at_least_four_complete_pairs": len(values) >= 4,
        "at_least_four_same_direction": max(positive, negative) >= 4,
        "exceeds_matched_null": effect_magnitude > null_max,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "positive_direction_count": positive,
        "negative_direction_count": negative,
        "treatment_deltas": {
            str(seed): value for seed, value in sorted(values.items())
        },
        "matched_null_deltas": {
            str(seed): value for seed, value in sorted(nulls.items())
        },
        "matched_null_max_abs": null_max,
        "median_abs_treatment_delta": effect_magnitude,
    }


def _action_change_gate(
    treatment_deltas: Mapping[int, float],
    matched_null_deltas: Mapping[int, float],
    *,
    action_bin_width: float,
) -> dict[str, Any]:
    """Run the registered action gate, preserving incomplete rows as no-go."""

    values = {
        int(seed): float(value)
        for seed, value in treatment_deltas.items()
    }
    nulls = {
        int(seed): float(value)
        for seed, value in matched_null_deltas.items()
    }
    if not values or set(values) != set(nulls):
        return {
            "passes": False,
            "checks": {
                "at_least_four_complete_pairs": len(values) >= 4,
                "at_least_four_same_direction": False,
                "exceeds_matched_null": False,
                "exceeds_one_action_bin": False,
            },
            "treatment_deltas": {
                str(seed): value for seed, value in sorted(values.items())
            },
            "matched_null_deltas": {
                str(seed): value for seed, value in sorted(nulls.items())
            },
            "matched_null_max_abs": None,
            "median_abs_treatment_delta": None,
            "incomplete_reason": (
                "missing action metrics or treatment/null seed mismatch"
            ),
        }
    try:
        gate = continuation_effect_gate(
            values,
            matched_null_deltas=nulls,
            action_bin_width=action_bin_width,
        )
    except (TypeError, ValueError) as exc:
        raise PilotEvidenceError(
            f"invalid sealed action-gate rows: {exc}"
        ) from exc
    return {
        **gate,
        "treatment_deltas": {
            str(seed): value for seed, value in sorted(values.items())
        },
        "matched_null_deltas": {
            str(seed): value for seed, value in sorted(nulls.items())
        },
    }


def _paired_metric_deltas(
    treatment: Mapping[int, Mapping[str, Any]],
    baseline: Mapping[int, Mapping[str, Any]],
    *,
    path: str,
    seeds: Sequence[int],
) -> dict[int, float]:
    output: dict[int, float] = {}
    for seed in seeds:
        left = _metric(treatment[seed], path)
        right = _metric(baseline[seed], path)
        if left is not None and right is not None:
            output[int(seed)] = left - right
    return output


def _experiment_d_gate(
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute D from sealed branches; no run may self-assert its own gate."""

    expected = tuple(int(value) for value in contract.seeds["sets"]["main"])
    arms = (
        "matched-a",
        "matched-b",
        "no-memory",
        "shuffled-episodic",
        "wrong-context",
        "error-verified",
        "error-unverified",
    )
    by_arm = {
        arm: {
            int(row["environment_seed"]): row
            for row in _scientific_rows(
                rows,
                stage="experiment-d",
                model="gpt52_main",
                arm=arm,
            )
        }
        for arm in arms
    }
    baseline = by_arm["matched-a"]
    matched_b = by_arm["matched-b"]
    action_grid = _mapping(
        contract.stop_go["experiment_d"]["action_grid"],
        "contract experiment_d action_grid",
    )
    causal_binding_checks: dict[str, Any] = {}
    shared_fields = (
        "checkpoint_hash",
        "prefix_hash",
        "shock_schedule_hash",
        "pre_generated_rng_hashes",
        "rng_schedule_binding",
        "shared_result_hash",
        "error_common_start_equal",
        "error_common_start_hash",
    )
    expected_treatments = {
        "matched-a": "matched-a",
        "matched-b": "matched-b",
        "no-memory": "no-memory",
        "shuffled-episodic": "shuffled-episodic",
        "wrong-context": "wrong-context",
        "error-verified": "erroneous-verified",
        "error-unverified": "erroneous-unverified",
    }
    for seed in expected:
        seed_rows = {
            arm: by_arm[arm][seed]
            for arm in arms
            if seed in by_arm[arm]
        }
        if not seed_rows:
            continue
        values: dict[str, Mapping[str, Any]] = {}
        errors: list[str] = []
        for arm, row in seed_rows.items():
            gate = _dig(row, "gate_evidence")
            if not isinstance(gate, Mapping):
                errors.append(f"{arm}:missing gate_evidence")
                continue
            if gate.get("matched_replay_equal") is not True:
                errors.append(f"{arm}:matched replay not equal")
                continue
            try:
                causal = _validate_causal_bindings(
                    gate.get("causal_bindings"),
                    name=f"experiment-d seed {seed} {arm} causal bindings",
                    narrative=False,
                    action_grid=action_grid,
                )
            except PilotEvidenceError as exc:
                errors.append(f"{arm}:{exc}")
                continue
            if (
                gate.get("checkpoint_hash") != causal.get("checkpoint_hash")
                or gate.get("prefix_hash") != causal.get("prefix_hash")
            ):
                errors.append(f"{arm}:top-level hash mismatch")
                continue
            forced_hash = causal.get("branch_forced_active_start_hash")
            if (
                causal.get("branch_treatment") != expected_treatments[arm]
                or (
                    arm in {"error-verified", "error-unverified"}
                    and not _is_sha256(forced_hash)
                )
                or (
                    arm not in {"error-verified", "error-unverified"}
                    and forced_hash is not None
                )
            ):
                errors.append(f"{arm}:branch treatment/error-start mismatch")
                continue
            values[arm] = causal
        complete_set = len(values) == len(arms)
        common = bool(
            complete_set
            and all(
                all(
                    value.get(field)
                    == next(iter(values.values())).get(field)
                    for field in shared_fields
                )
                for value in values.values()
            )
        )
        if common and values:
            common = (
                values["error-verified"][
                    "branch_forced_active_start_hash"
                ]
                == values["error-unverified"][
                    "branch_forced_active_start_hash"
                ]
                == values["error-verified"]["error_common_start_hash"]
            )
        passed = bool(complete_set and common and not errors)
        causal_binding_checks[str(seed)] = {
            "pass": passed,
            "complete_arm_count": len(values),
            "registered_arm_count": len(arms),
            "common_checkpoint_prefix_rng_and_error_start": common,
            "errors": errors,
        }
    results: dict[str, Any] = {}
    supported: list[str] = []
    prompt_only: list[str] = []
    reported_paths = {
        "focal_first_labor_hours": (
            "continuation.focal.first_step.labor_hours"
        ),
        "focal_first_consumption_rate": (
            "continuation.focal.first_step.consumption_rate"
        ),
        "focal_immediate_flow_utility": (
            "continuation.focal.first_step.immediate_flow_utility"
        ),
        "focal_next_wealth": (
            "continuation.focal.first_step.next_wealth"
        ),
        "focal_six_step_discounted_utility": (
            "continuation.focal.discounted_flow_utility_sum"
        ),
        "population_next_wealth": (
            "continuation.population.first_step.average_next_wealth"
        ),
        "population_next_gini": (
            "continuation.population.first_step.gini_next_wealth"
        ),
        "population_next_low_labor": (
            "continuation.population.first_step.low_labor_rate"
        ),
        "population_six_step_utility": (
            "continuation.population.flow_utility_sum"
        ),
        "population_final_wealth": (
            "continuation.population.average_final_wealth"
        ),
        "population_final_gini": (
            "continuation.population.gini_final_wealth"
        ),
        "population_mean_low_labor": (
            "continuation.population.mean_low_labor_rate"
        ),
    }
    for treatment_name in arms[2:]:
        treatment = by_arm[treatment_name]
        seeds = [
            seed
            for seed in expected
            if seed in treatment and seed in baseline and seed in matched_b
            and causal_binding_checks.get(str(seed), {}).get("pass") is True
        ]
        reported_deltas = {
            name: _delta_descriptives(
                _paired_metric_deltas(
                    treatment,
                    baseline,
                    path=path,
                    seeds=seeds,
                )
            )
            for name, path in reported_paths.items()
        }
        labor_treatment = _paired_metric_deltas(
            treatment,
            baseline,
            path=reported_paths["focal_first_labor_hours"],
            seeds=seeds,
        )
        labor_null = _paired_metric_deltas(
            matched_b,
            baseline,
            path=reported_paths["focal_first_labor_hours"],
            seeds=seeds,
        )
        consumption_treatment = _paired_metric_deltas(
            treatment,
            baseline,
            path=reported_paths["focal_first_consumption_rate"],
            seeds=seeds,
        )
        consumption_null = _paired_metric_deltas(
            matched_b,
            baseline,
            path=reported_paths["focal_first_consumption_rate"],
            seeds=seeds,
        )
        utility_treatment = _paired_metric_deltas(
            treatment,
            baseline,
            path=reported_paths["focal_six_step_discounted_utility"],
            seeds=seeds,
        )
        utility_null = _paired_metric_deltas(
            matched_b,
            baseline,
            path=reported_paths["focal_six_step_discounted_utility"],
            seeds=seeds,
        )
        labor_gate = _action_change_gate(
            labor_treatment,
            labor_null,
            action_bin_width=float(action_grid["labor_step_hours"]),
        )
        consumption_gate = _action_change_gate(
            consumption_treatment,
            consumption_null,
            action_bin_width=float(action_grid["consumption_step"]),
        )
        utility_gate = _direction_and_null_gate(
            utility_treatment,
            utility_null,
        )
        action_pass = bool(labor_gate["passes"] or consumption_gate["passes"])
        closed_loop_pass = bool(action_pass and utility_gate["passes"])
        if closed_loop_pass:
            supported.append(treatment_name)
        elif action_pass:
            prompt_only.append(treatment_name)
        results[treatment_name] = {
            "usable_seeds": seeds,
            "action_gates": {
                "labor_hours": labor_gate,
                "consumption_rate": consumption_gate,
            },
            "six_step_discounted_utility_gate": utility_gate,
            "qualified_action_change": action_pass,
            "qualified_closed_loop_effect": closed_loop_pass,
            "classification": (
                "closed-loop-continuation-effect"
                if closed_loop_pass
                else "prompt-sensitivity-only"
                if action_pass
                else "no-qualified-effect"
            ),
            "reported_paired_deltas": reported_deltas,
        }
    causal_binding_pass = (
        sum(value["pass"] for value in causal_binding_checks.values()) >= 4
    )
    complete = bool(results) and all(
        len(value["usable_seeds"]) >= 4 for value in results.values()
    ) and causal_binding_pass
    reasons = []
    if not complete:
        reasons.append(
            "one or more registered treatments has fewer than four complete "
            "matched checkpoint seeds"
        )
    if not causal_binding_pass:
        reasons.append(
            "fewer than four seed sets retain common sealed checkpoint, "
            "prefix, RNG, shock, replay, and erroneous-rule start bindings"
        )
    if not supported:
        reasons.append(
            "no treatment passed both the matched-null/action-bin gate and "
            "the six-step utility gate"
        )
    qualified_supported = supported if complete else []
    return {
        "status": "supported" if qualified_supported else "no-go",
        "scientific_evidence_complete": complete,
        "supported_treatments": qualified_supported,
        "prompt_sensitivity_only_treatments": prompt_only,
        "treatment_gates": results,
        "causal_binding_checks": causal_binding_checks,
        "causal_binding_pass": causal_binding_pass,
        "action_bin_widths": {
            "labor_hours": float(action_grid["labor_step_hours"]),
            "consumption_rate": float(action_grid["consumption_step"]),
        },
        "claim_action": (
            "claim only the named closed-loop treatment effects; classify "
            "action-only changes as prompt sensitivity"
            if qualified_supported
            else "do not claim a closed-loop continuation effect"
        ),
        "reasons": reasons,
    }


def _narrative_gate(
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the four content fixtures across checkpoint-bound seeds."""

    expected = tuple(int(value) for value in contract.seeds["sets"]["main"])
    action_grid = _mapping(
        contract.stop_go["experiment_d"]["action_grid"],
        "contract experiment_d action_grid",
    )
    narrative_fixture_hash = str(
        contract.stop_go["experiment_d"]["narrative_fixture_hash"]
    )
    narratives = {
        narrative: {
            int(row["environment_seed"]): row
            for row in _scientific_rows(
                rows,
                stage="experiment-d",
                model="gpt52_main",
                arm="narrative-content",
            )
            if row["narrative_id"] == narrative
        }
        for narrative in ("none", "aligned", "paraphrase", "opposite")
    }
    matched = {
        arm: {
            int(row["environment_seed"]): row
            for row in _scientific_rows(
                rows,
                stage="experiment-d",
                model="gpt52_main",
                arm=arm,
            )
        }
        for arm in ("matched-a", "matched-b")
    }
    complete_seeds = [
        seed
        for seed in expected
        if all(seed in values for values in narratives.values())
        and seed in matched["matched-a"]
        and seed in matched["matched-b"]
    ]
    causal_binding_checks: dict[str, Any] = {}
    for seed in complete_seeds:
        errors: list[str] = []
        narrative_bindings: dict[str, Mapping[str, Any]] = {}
        matched_bindings: dict[str, Mapping[str, Any]] = {}
        for narrative_id, seed_rows in narratives.items():
            gate = _dig(seed_rows[seed], "gate_evidence")
            if not isinstance(gate, Mapping):
                errors.append(f"{narrative_id}:missing gate_evidence")
                continue
            try:
                narrative_bindings[narrative_id] = _validate_causal_bindings(
                    gate.get("causal_bindings"),
                    name=(
                        f"narrative seed {seed} {narrative_id} causal bindings"
                    ),
                    narrative=True,
                    action_grid=action_grid,
                    narrative_fixture_hash=narrative_fixture_hash,
                )
            except PilotEvidenceError as exc:
                errors.append(f"{narrative_id}:{exc}")
        for arm, seed_rows in matched.items():
            gate = _dig(seed_rows[seed], "gate_evidence")
            if not isinstance(gate, Mapping) or gate.get(
                "matched_replay_equal"
            ) is not True:
                errors.append(f"{arm}:missing/equal replay binding")
                continue
            try:
                matched_bindings[arm] = _validate_causal_bindings(
                    gate.get("causal_bindings"),
                    name=f"narrative null seed {seed} {arm} causal bindings",
                    narrative=False,
                    action_grid=action_grid,
                )
            except PilotEvidenceError as exc:
                errors.append(f"{arm}:{exc}")
        narrative_common_fields = (
            "checkpoint_hash",
            "prefix_hash",
            "pre_generated_rng_hashes",
            "rng_schedule_binding",
            "shared_result_hash",
            "fixture_hash",
        )
        matched_common_fields = (
            "checkpoint_hash",
            "prefix_hash",
            "shock_schedule_hash",
            "pre_generated_rng_hashes",
            "rng_schedule_binding",
            "shared_result_hash",
            "error_common_start_equal",
            "error_common_start_hash",
        )
        narrative_common = bool(
            len(narrative_bindings) == len(narratives)
            and all(
                all(
                    value.get(field)
                    == next(iter(narrative_bindings.values())).get(field)
                    for field in narrative_common_fields
                )
                for value in narrative_bindings.values()
            )
        )
        matched_common = bool(
            len(matched_bindings) == len(matched)
            and all(
                all(
                    value.get(field)
                    == next(iter(matched_bindings.values())).get(field)
                    for field in matched_common_fields
                )
                for value in matched_bindings.values()
            )
        )
        cross_source_common = False
        if narrative_common and matched_common:
            left = next(iter(narrative_bindings.values()))
            right = next(iter(matched_bindings.values()))
            cross_source_common = all(
                left.get(field) == right.get(field)
                for field in (
                    "checkpoint_hash",
                    "prefix_hash",
                    "pre_generated_rng_hashes",
                    "rng_schedule_binding",
                )
            )
        if narrative_common:
            narrative_common = all(
                value.get("branch_narrative_id") == narrative_id
                for narrative_id, value in narrative_bindings.items()
            )
        causal_binding_checks[str(seed)] = {
            "pass": bool(
                narrative_common
                and matched_common
                and cross_source_common
                and not errors
            ),
            "narrative_common": narrative_common,
            "matched_null_common": matched_common,
            "narrative_and_null_common_source": cross_source_common,
            "errors": errors,
        }
    complete_seeds = [
        seed
        for seed in complete_seeds
        if causal_binding_checks.get(str(seed), {}).get("pass") is True
    ]
    if not complete_seeds:
        return {
            "status": "no-go",
            "semantic_response": False,
            "scientific_evidence_complete": False,
            "claim_boundary": (
                "no controlled semantic-response claim; never claim real-news "
                "understanding"
            ),
            "reasons": ["no complete sealed narrative checkpoint sets"],
        }

    narrative_paths = {
        "labor_hours": "narrative.first_labor_hours",
        "consumption_rate": "narrative.first_consumption_rate",
        "immediate_flow_utility": "narrative.immediate_flow_utility",
        "six_step_discounted_flow_utility": (
            "narrative.six_step_discounted_flow_utility"
        ),
        "final_wealth": "narrative.final_wealth",
    }
    matched_paths = {
        "labor_hours": "continuation.focal.first_step.labor_hours",
        "consumption_rate": (
            "continuation.focal.first_step.consumption_rate"
        ),
        "immediate_flow_utility": (
            "continuation.focal.first_step.immediate_flow_utility"
        ),
        "six_step_discounted_flow_utility": (
            "continuation.focal.discounted_flow_utility_sum"
        ),
        "final_wealth": "continuation.focal.final_wealth",
    }

    def narrative_delta(left: str, right: str, metric: str) -> dict[int, float]:
        output = {}
        for seed in complete_seeds:
            left_value = _metric(
                narratives[left][seed],
                narrative_paths[metric],
            )
            right_value = _metric(
                narratives[right][seed],
                narrative_paths[metric],
            )
            if left_value is not None and right_value is not None:
                output[seed] = left_value - right_value
        return output

    def matched_null(metric: str) -> dict[int, float]:
        return _paired_metric_deltas(
            matched["matched-b"],
            matched["matched-a"],
            path=matched_paths[metric],
            seeds=complete_seeds,
        )

    labor_delta = narrative_delta("aligned", "opposite", "labor_hours")
    consumption_delta = narrative_delta(
        "aligned", "opposite", "consumption_rate"
    )
    labor_gate = _action_change_gate(
        labor_delta,
        matched_null("labor_hours"),
        action_bin_width=float(action_grid["labor_step_hours"]),
    )
    consumption_gate = _action_change_gate(
        consumption_delta,
        matched_null("consumption_rate"),
        action_bin_width=float(action_grid["consumption_step"]),
    )
    utility_delta = narrative_delta(
        "aligned",
        "opposite",
        "six_step_discounted_flow_utility",
    )
    utility_gate = _direction_and_null_gate(
        utility_delta,
        matched_null("six_step_discounted_flow_utility"),
    )
    equivalence_by_seed = {}
    for seed in complete_seeds:
        aligned_labor = _metric(
            narratives["aligned"][seed],
            narrative_paths["labor_hours"],
        )
        paraphrase_labor = _metric(
            narratives["paraphrase"][seed],
            narrative_paths["labor_hours"],
        )
        aligned_consumption = _metric(
            narratives["aligned"][seed],
            narrative_paths["consumption_rate"],
        )
        paraphrase_consumption = _metric(
            narratives["paraphrase"][seed],
            narrative_paths["consumption_rate"],
        )
        equivalent = bool(
            aligned_labor is not None
            and paraphrase_labor is not None
            and aligned_consumption is not None
            and paraphrase_consumption is not None
            and abs(aligned_labor - paraphrase_labor)
            <= float(action_grid["labor_step_hours"])
            and abs(aligned_consumption - paraphrase_consumption)
            <= float(action_grid["consumption_step"])
        )
        equivalence_by_seed[str(seed)] = equivalent
    equivalence_count = sum(equivalence_by_seed.values())
    action_changed = bool(labor_gate["passes"] or consumption_gate["passes"])
    causal_complete_count = sum(
        value["pass"] for value in causal_binding_checks.values()
    )
    complete = len(complete_seeds) >= 4 and causal_complete_count >= 4
    supported = bool(
        complete
        and action_changed
        and utility_gate["passes"]
        and equivalence_count >= 4
    )
    all_deltas = {
        comparison: {
            metric: _delta_descriptives(
                narrative_delta(left, right, metric)
            )
            for metric in narrative_paths
        }
        for comparison, (left, right) in {
            "aligned_vs_opposite": ("aligned", "opposite"),
            "aligned_vs_none": ("aligned", "none"),
            "paraphrase_vs_aligned": ("paraphrase", "aligned"),
        }.items()
    }
    reasons = []
    if not complete:
        reasons.append("fewer than four complete narrative checkpoint sets")
    if causal_complete_count < 4:
        reasons.append(
            "fewer than four narrative sets share sealed checkpoint, prefix, "
            "shock, and pre-generated RNG bindings with the matched null"
        )
    if not action_changed:
        reasons.append(
            "aligned versus opposite did not clear matched-null and action-bin gates"
        )
    if not utility_gate["passes"]:
        reasons.append(
            "aligned versus opposite six-step utility did not clear the "
            "registered direction and matched-null checks"
        )
    if equivalence_count < 4:
        reasons.append(
            "aligned and paraphrased fixtures were not action-bin equivalent "
            "in four seeds"
        )
    return {
        "status": "supported" if supported else "no-go",
        "semantic_response": supported,
        "scientific_evidence_complete": complete,
        "usable_seeds": complete_seeds,
        "causal_binding_checks": causal_binding_checks,
        "causal_binding_complete_seed_count": causal_complete_count,
        "aligned_vs_opposite_action_gates": {
            "labor_hours": labor_gate,
            "consumption_rate": consumption_gate,
        },
        "aligned_vs_opposite_six_step_utility_gate": utility_gate,
        "paraphrase_equivalence_by_seed": equivalence_by_seed,
        "paraphrase_equivalent_seed_count": equivalence_count,
        "reported_paired_deltas": all_deltas,
        "claim_boundary": (
            "controlled semantic response only; not real-news understanding"
            if supported
            else "prompt sensitivity at most; not real-news understanding"
        ),
        "reasons": reasons,
    }


def _capability_by_model(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["stage_id"] != "capability-preflight":
            continue
        model = str(row["model_id"])
        capability = row.get("capability")
        result[model] = {
            "ledger_status": row["status"],
            "artifact_validated": row["artifact_kind"] is not None,
            "capability": (
                _json_copy(capability) if isinstance(capability, Mapping) else {}
            ),
        }
    return result


def _capability_pass(value: Mapping[str, Any]) -> bool:
    capability = value.get("capability")
    if not isinstance(capability, Mapping):
        return False
    return bool(
        value.get("ledger_status") == "complete"
        and value.get("artifact_validated") is True
        and capability.get("pass") is True
        and capability.get("preflight_go") is True
    )


def _cross_model_summary(
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
    capability: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    models = {"gpt52_main"} | {
        spec.model_id
        for spec in contract.expand(stage="cross-model-sentinels")
    }
    expected_seeds = tuple(
        int(value) for value in contract.seeds["sets"]["cross-model"]
    )
    output: dict[str, Any] = {}
    for model in sorted(models):
        source_stage = (
            "experiment-b"
            if model == "gpt52_main"
            else "cross-model-sentinels"
        )
        full = {
            int(row["environment_seed"]): row
            for row in _scientific_rows(
                rows,
                stage=source_stage,
                model=model,
                arm="full",
            )
            if int(row["environment_seed"]) in expected_seeds
        }
        control = {
            int(row["environment_seed"]): row
            for row in _scientific_rows(
                rows,
                stage=source_stage,
                model=model,
                arm="no-memory",
            )
            if int(row["environment_seed"]) in expected_seeds
        }
        seeds = sorted(
            seed
            for seed in expected_seeds
            if seed in full and seed in control
            if _metric(full[seed], "utility.shock_recovery_discounted") is not None
            and _metric(control[seed], "utility.shock_recovery_discounted")
            is not None
        )
        delta = None
        if seeds:
            delta = paired_delta_summary(
                {
                    seed: float(
                        _metric(full[seed], "utility.shock_recovery_discounted")
                    )
                    for seed in seeds
                },
                {
                    seed: float(
                        _metric(control[seed], "utility.shock_recovery_discounted")
                    )
                    for seed in seeds
                },
            )
        capability_ok = _capability_pass(capability.get(model, {}))
        capability_payload = capability.get(model, {}).get("capability", {})
        if not isinstance(capability_payload, Mapping):
            capability_payload = {}
        category_totals = capability_payload.get("category_totals", {})
        if not isinstance(category_totals, Mapping):
            category_totals = {}
        raw_deltas = (
            {
                int(seed): float(value)
                for seed, value in delta["raw_paired_deltas"].items()
            }
            if delta
            else {}
        )
        positive = bool(
            len(raw_deltas) == 3 and all(value > 0 for value in raw_deltas.values())
        )
        negative = bool(
            len(raw_deltas) == 3 and all(value < 0 for value in raw_deltas.values())
        )
        registered_rows = [
            row
            for row in rows
            if row["stage_id"] == source_stage
            and row["model_id"] == model
            and row["arm_id"] in {"full", "no-memory"}
            and int(row["environment_seed"]) in expected_seeds
        ]
        execution_audit = {
            f"{row['arm_id']}:{row['environment_seed']}": {
                "status": row["status"],
                "failure": _json_copy(row["failure"]),
                "provider_failure_count": _metric(
                    row, "guardrails.provider_failure_count"
                ),
                "proposal_parse_status_counts": _json_copy(
                    _dig(
                        row,
                        "metrics.memory.proposal_parse_status_counts",
                    )
                ),
            }
            for row in sorted(
                registered_rows,
                key=lambda item: (
                    str(item["arm_id"]),
                    int(item["environment_seed"]),
                ),
            )
        }
        matched_null = None
        exceeds_matched_null: bool | None = None
        matched_null_resolution = "not-required"
        if model == "opus48_sentinel":
            matched_a = {
                int(row["environment_seed"]): row
                for row in _scientific_rows(
                    rows,
                    stage="cross-model-sentinels",
                    model=model,
                    arm="matched-a",
                )
            }
            matched_b = {
                int(row["environment_seed"]): row
                for row in _scientific_rows(
                    rows,
                    stage="cross-model-sentinels",
                    model=model,
                    arm="matched-b",
                )
            }
            null_values = _paired_metric_deltas(
                matched_b,
                matched_a,
                path="utility.shock_recovery_discounted",
                seeds=[
                    seed
                    for seed in expected_seeds
                    if seed in matched_a and seed in matched_b
                ],
            )
            matched_null = {
                "reason": (
                    "seed-unsupported provider matched A/A null; not an "
                    "effectiveness contrast"
                ),
                "paired_deltas": {
                    str(seed): value
                    for seed, value in sorted(null_values.items())
                },
                "max_abs": (
                    max(abs(value) for value in null_values.values())
                    if null_values
                    else None
                ),
            }
            null_complete = len(null_values) == 3
            effect_magnitude = (
                min(abs(value) for value in raw_deltas.values())
                if len(raw_deltas) == 3
                else None
            )
            null_max = matched_null["max_abs"]
            exceeds_matched_null = bool(
                null_complete
                and effect_magnitude is not None
                and null_max is not None
                and effect_magnitude > float(null_max)
            )
            matched_null_resolution = (
                "effect-exceeds-matched-a-a-null"
                if exceeds_matched_null
                else "unresolved-within-or-without-complete-matched-a-a-null"
            )
            matched_null["minimum_abs_effect_delta"] = effect_magnitude
        replicated = bool(
            capability_ok
            and delta
            and delta["pair_count"] == 3
            and (positive or negative)
            and (
                model != "opus48_sentinel"
                or exceeds_matched_null is True
            )
        )
        output[model] = {
            "source": (
                "reused experiment-b first-three registered seeds"
                if model == "gpt52_main"
                else "cross-model-sentinels"
            ),
            "capability_and_preflight_pass": capability_ok,
            "utility_ranking_competence": _json_copy(
                category_totals.get("utility-ranking")
            ),
            "rule_application_competence": _json_copy(
                category_totals.get("rule-application")
            ),
            "proposal_competence": _json_copy(
                category_totals.get("rule-proposal")
            ),
            "capability_parse_failure_count": capability_payload.get(
                "parse_failure_count"
            ),
            "capability_provider_failure_count": capability_payload.get(
                "provider_failure_count"
            ),
            "registered_seed_status_and_failures": execution_audit,
            "usable_paired_seeds": seeds,
            "paired_delta": delta,
            "direction": (
                "positive" if positive else "negative" if negative else "mixed-or-incomplete"
            ),
            "directional_micro_pilot_replication": replicated,
            "seed_unsupported_matched_a_a_null": matched_null,
            "matched_null_resolution": matched_null_resolution,
            "effect_exceeds_matched_a_a_null": exceeds_matched_null,
            "claim_boundary": (
                "direction replicated in this model-family micro-pilot only"
                if replicated
                else "no cross-model effectiveness claim"
            ),
        }
    return output


def _claims(
    gates: Mapping[str, Any],
    *,
    denominator: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "claim": "M1 retrieval contributes beyond regime prompting",
            "metric": "full minus prompt-only shock+recovery discounted utility",
            "artifact": "aggregate.json",
            "status": gates["experiment_a"]["status"],
            "boundary": gates["experiment_a"]["claim_action"],
        },
        {
            "claim": "Evidence grounding improves erroneous-rule reliability",
            "metric": (
                "false activation, harmful exposure, and cumulative utility-loss "
                "directions"
            ),
            "artifact": "aggregate.json",
            "status": gates["experiment_c"]["status"],
            "boundary": gates["experiment_c"]["claim_action"],
        },
        {
            "claim": "Memory intervention changes the matched continuation",
            "metric": "matched-null- and action-bin-qualified continuation deltas",
            "artifact": "aggregate.json",
            "status": gates["experiment_d"]["status"],
            "boundary": gates["experiment_d"]["claim_action"],
        },
        {
            "claim": "Narrative channel shows controlled semantic response",
            "metric": "aligned/opposite delta and paraphrase action-bin equivalence",
            "artifact": "aggregate.json",
            "status": gates["narrative"]["status"],
            "boundary": gates["narrative"]["claim_boundary"],
        },
        {
            "claim": "Backbone-independent improvement",
            "metric": "not preregistered or supported by this micro-pilot",
            "artifact": "method_differences_scaffold.json",
            "status": "prohibited",
            "boundary": "never use backbone-independent wording",
        },
        {
            "claim": "Complete preregistered ITT matrix",
            "metric": "one terminal row for every expanded contract cell",
            "artifact": "failure_ledger.json",
            "status": "supported" if denominator["pass"] else "no-go",
            "boundary": "report every failure, stop, nonterminal, and missing cell",
        },
    ]


def _flatten_scalars(
    value: Any,
    *,
    prefix: str = "",
) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_scalars(value[key], prefix=child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            yield from _flatten_scalars(item, prefix=child)
    elif value is None or isinstance(value, (str, bool)) or _is_finite_scalar(value):
        yield prefix, value
    else:
        raise PilotEvidenceError(f"aggregate contains unsupported value at {prefix}")


def _aggregate_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = (
        "run_id",
        "status",
        "stage_id",
        "model_id",
        "arm_id",
        "narrative_id",
        "environment_seed",
        "scientific_eligible",
        "artifact_kind",
        "metric_path",
        "metric_value_json",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        metrics = list(_flatten_scalars(row.get("metrics", {}), prefix="metrics"))
        if not metrics:
            metrics = [("", None)]
        for path, value in metrics:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields[:-2]},
                    "metric_path": path,
                    "metric_value_json": json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    ),
                }
            )
    return stream.getvalue().encode("utf-8")


def _method_scaffold() -> dict[str, Any]:
    """Return a primary-source-backed method comparison, not a result claim.

    The historical filename is retained for read compatibility, but there are
    deliberately no TODO cells.  Comparator statements are bounded to what the
    cited framework sections explicitly describe; in particular, "not
    specified" never means that an implementation could not contain an
    undocumented mechanism.
    """

    sources = {
        "finevo_contract": {
            "title": "FinEvo frozen pilot contract and current_v2 implementation",
            "artifact": "contract/pilot_v1.yaml",
            "sections": [
                "arms",
                "shocks",
                "utility",
                "stop_go",
                "stages",
            ],
        },
        "econagent": {
            "title": (
                "EconAgent: Large Language Model-Empowered Agents for "
                "Simulating Macroeconomic Activities"
            ),
            "url": "https://arxiv.org/pdf/2310.10436",
            "version": "arXiv:2310.10436v4",
            "sections": ["2.1", "3.2", "3.3", "4.3", "4.5"],
        },
        "econai": {
            "title": (
                "EconAI: Dynamic Persona Evolution and Memory-Aware Agents "
                "in Evolving Economic Environments"
            ),
            "url": "https://arxiv.org/pdf/2605.13762",
            "version": "arXiv:2605.13762v1",
            "sections": ["3.1", "3.2", "3.4", "4.4", "4.5"],
        },
    }
    rows = [
        {
            "dimension": "agent types and decisions",
            "FinEvo current_v2": (
                "Household agents choose discretized labor hours and a "
                "consumption fraction inside a monthly closed-loop simulation."
            ),
            "EconAgent": (
                "Household agents produce monthly work and consumption "
                "propensities in a labor/consumption macroeconomic loop."
            ),
            "EconAI": (
                "The framework models households and firms; household work and "
                "consumption decisions coexist with firm-side production, "
                "investment, and employment decisions."
            ),
            "source_refs": {
                "FinEvo current_v2": ["finevo_contract:arms", "finevo_contract:stages"],
                "EconAgent": ["econagent:2.1", "econagent:3.3"],
                "EconAI": ["econai:3.4"],
            },
        },
        {
            "dimension": "temporal memory representation",
            "FinEvo current_v2": (
                "An evidence-linked episodic track is paired with structured "
                "semantic rules; retrieval is separately controllable from "
                "regime prompting."
            ),
            "EconAgent": (
                "A rolling pool stores conversations from the previous L months "
                "and adds an LLM reflection at each quarter end; the reported "
                "experiments use L=1."
            ),
            "EconAI": (
                "Short-term context and embedding-retrieved long-term event "
                "summaries are combined with a persistent dynamic persona bank."
            ),
            "source_refs": {
                "FinEvo current_v2": ["finevo_contract:arms", "finevo_contract:stages"],
                "EconAgent": ["econagent:3.2"],
                "EconAI": ["econai:3.1", "econai:3.4"],
            },
        },
        {
            "dimension": "belief or rule abstraction",
            "FinEvo current_v2": (
                "LLM-proposed rules are parsed into a fixed condition/action "
                "schema and exposed to the actor only under the registered "
                "semantic-memory policy."
            ),
            "EconAgent": (
                "Quarterly reflection is retained as free-form conclusions about "
                "labor, consumption, and financial-market dynamics."
            ),
            "EconAI": (
                "An LLM-derived Economic Sentiment Index is exponentially "
                "smoothed and used with confidence to modulate work and "
                "consumption; the framework also updates personas."
            ),
            "source_refs": {
                "FinEvo current_v2": ["finevo_contract:arms", "finevo_contract:stop_go"],
                "EconAgent": ["econagent:3.2"],
                "EconAI": ["econai:3.2"],
            },
        },
        {
            "dimension": "error control for abstracted memory",
            "FinEvo current_v2": (
                "The evidence-grounded track has explicit candidate admission, "
                "activation, harmful-compliance monitoring, and retirement; the "
                "pilot contrasts it with immediate unverified activation."
            ),
            "EconAgent": (
                "The cited memory section describes quarterly LLM reflection but "
                "does not specify evidence-linked admission or retirement tests "
                "for reflected conclusions."
            ),
            "EconAI": (
                "The cited framework describes event summarization, retrieval, "
                "persona updates, and sentiment weighting, but does not specify "
                "a verifier lifecycle for abstracted decision rules."
            ),
            "source_refs": {
                "FinEvo current_v2": ["finevo_contract:arms", "finevo_contract:stop_go"],
                "EconAgent": ["econagent:3.2"],
                "EconAI": ["econai:3.1", "econai:3.2"],
            },
        },
        {
            "dimension": "macro-state to memory to action trace",
            "FinEvo current_v2": (
                "The registered artifacts separately expose shock events, routed "
                "context, retrieved episode/rule IDs, actions, utility ledger, "
                "and subsequent macro state."
            ),
            "EconAgent": (
                "Economic variables, recent conversations, and quarterly "
                "reflection are inserted into the prompt that emits work and "
                "consumption propensities."
            ),
            "EconAI": (
                "Retrieved events, short-term context, persona, and ESI feed the "
                "response generator, whose decisions then enter the economic "
                "environment."
            ),
            "source_refs": {
                "FinEvo current_v2": ["finevo_contract:stages", "finevo_contract:stop_go"],
                "EconAgent": ["econagent:3.2", "econagent:3.3"],
                "EconAI": ["econai:3.2", "econai:3.4"],
            },
        },
        {
            "dimension": "textual event evaluation",
            "FinEvo current_v2": (
                "One hash-bound checkpoint branches into no text, shock-aligned "
                "text, a semantic paraphrase, and direction-opposite text; the "
                "claim is limited to controlled semantic response."
            ),
            "EconAgent": (
                "A COVID-19 national-emergency sentence is added after March "
                "2020 and compared with a no-injection simulation."
            ),
            "EconAI": (
                "A comparable single COVID-19 textual intervention is contrasted "
                "with a counterfactual without the injection."
            ),
            "source_refs": {
                "FinEvo current_v2": ["finevo_contract:stages", "finevo_contract:non_claims"],
                "EconAgent": ["econagent:4.5"],
                "EconAI": ["econai:4.5"],
            },
        },
        {
            "dimension": "mechanism evidence and denominator",
            "FinEvo current_v2": (
                "Paired seed-level route, architecture, erroneous-rule, and "
                "checkpoint interventions retain provider, parse, budget, and "
                "integrity failures in a preregistered ITT denominator."
            ),
            "EconAgent": (
                "The paper reports perception/reflection ablations and multiple "
                "simulation runs; its cited experimental sections do not define "
                "FinEvo's cell-level provider/parse-failure ITT policy."
            ),
            "EconAI": (
                "The paper reports memory/sentiment/belief ablations and scale "
                "sensitivity; its cited experimental sections do not define "
                "FinEvo's cell-level provider/parse-failure ITT policy."
            ),
            "source_refs": {
                "FinEvo current_v2": ["finevo_contract:stop_go", "finevo_contract:stages"],
                "EconAgent": ["econagent:4.3"],
                "EconAI": ["econai:4.4"],
            },
        },
    ]
    return {
        "schema_version": PILOT_EVIDENCE_SCHEMA_VERSION,
        "status": "primary-source-backed method comparison",
        "comparators": ["FinEvo current_v2", "EconAgent", "EconAI"],
        "sources": sources,
        "rows": rows,
        "evidence_boundary": (
            "This table compares documented mechanisms. It is not an empirical "
            "ranking, and 'not specified' is bounded to the cited paper sections."
        ),
    }


def _json_cell(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _report_markdown(
    *,
    contract: PilotContract,
    denominator: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
    experiment_b: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    capability: Mapping[str, Any],
    cross_model: Mapping[str, Any],
    release_controls: Mapping[str, Any],
    rule_sensitivity: Mapping[str, Any] | None,
) -> str:
    stage_counts: dict[str, Counter[str]] = {}
    for row in rows:
        stage_counts.setdefault(str(row["stage_id"]), Counter())[
            str(row["status"])
        ] += 1
    lines = [
        "# FinEvo preregistered mechanism micro-pilot evidence report",
        "",
        f"- Evidence namespace: `{EVIDENCE_NAMESPACE}`",
        f"- Contract: `{contract.contract_id}` / `{contract.canonical_hash}`",
        f"- Registered cells: {denominator['expected_count']}",
        f"- ITT ledger complete: `{str(denominator['pass']).lower()}`",
        "- Scale boundary: 4 agents × 12 months; this is not the 10×24×5 "
        "confirmatory pilot or the 100×240 simulation.",
        f"- Release/Stage-0/budget controls: "
        f"`{'pass' if release_controls.get('pass') else 'no-go'}`",
        "",
        "## Release, Stage-0, and budget controls",
        "",
        f"- Remote annotated-tag and Linux/macOS CI attestation: "
        f"`{_json_cell(release_controls.get('release_attestation', {}))}`",
        f"- Outcome-blind Stage-0 selection and complete 7×2 source matrix: "
        f"`{_json_cell(release_controls.get('stage0_selection', {}))}`",
        f"- Finalized cross-stage budget ledger and frozen caps: "
        f"`{_json_cell(release_controls.get('budget_ledger', {}))}`",
        "",
        "## Claim → metric → artifact",
        "",
        "| Claim | Metric | Artifact | Status | Required wording |",
        "|---|---|---|---|---|",
    ]
    for claim in claims:
        lines.append(
            "| "
            + " | ".join(
                str(claim[key]).replace("|", "\\|")
                for key in ("claim", "metric", "artifact", "status", "boundary")
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Experiment A — M1 route decomposition",
            "",
            f"- Gate: `{gates['experiment_a']['status']}`",
            f"- Usable paired seeds: "
            f"`{_json_cell(gates['experiment_a'].get('usable_paired_seeds', []))}`",
            f"- Claim action: {gates['experiment_a']['claim_action']}.",
            "- Utility-deficit AUC and recovery-time paired summaries: "
            f"`{_json_cell(gates['experiment_a'].get('secondary_paired_metrics', {}))}`",
            f"- Gate evidence: `{_json_cell(gates['experiment_a'])}`",
            "",
            "## Experiment B — memory architecture",
            "",
        ]
    )
    for arm, value in experiment_b["arms"].items():
        lines.append(
            f"- `{arm}`: status `{_json_cell(value['status_counts'])}`; "
            f"paired utility deltas versus full "
            f"`{_json_cell(value['paired_vs_full'])}`; seed-level "
            "utility/action/retrieval/proposal/lifecycle rows are preserved "
            "in `aggregate.json`."
        )
    lines.extend(
        [
            "",
            "Experiment B remains a descriptive architecture comparison; no winner is "
            "selected by wealth alone.",
            "",
            "## Experiment C — rule reliability",
            "",
            f"- Gate: `{gates['experiment_c']['status']}`",
            f"- Claim action: {gates['experiment_c']['claim_action']}.",
            f"- Gate evidence: `{_json_cell(gates['experiment_c'])}`",
            "- Registered zero-API 3×3 sensitivity control: "
            f"`{_json_cell(release_controls.get('experiment_c_sensitivity', {}))}`",
            "- Sensitivity grid: "
            f"`{_json_cell((rule_sensitivity or {}).get('aggregate_cells', []))}`",
            "- This replay is descriptive only and cannot rescue a failed "
            "Experiment C effectiveness gate.",
            "",
            "## Experiment D — checkpoint continuation",
            "",
            f"- Gate: `{gates['experiment_d']['status']}`",
            f"- Passing named treatments: "
            f"`{_json_cell(gates['experiment_d'].get('supported_treatments', []))}`",
            f"- Claim action: {gates['experiment_d']['claim_action']}.",
            f"- Gate evidence: `{_json_cell(gates['experiment_d'])}`",
            "",
            "## Model capability and cross-model sentinels",
            "",
            "| Model profile | Capability/preflight | Cross-model conclusion |",
            "|---|---|---|",
        ]
    )
    for model in sorted(set(capability) | set(cross_model)):
        capability_status = capability.get(model, {})
        model_result = cross_model.get(model, {})
        lines.append(
            f"| `{model}` | `{_json_cell(capability_status)}` | "
            f"`{_json_cell(model_result)}` |"
        )
    lines.extend(
        [
            "",
            "No result in this table supports the phrase “backbone-independent.”",
            "",
            "## Narrative content intervention",
            "",
            f"- Gate: `{gates['narrative']['status']}`",
            f"- Boundary: {gates['narrative']['claim_boundary']}.",
            f"- Gate evidence: `{_json_cell(gates['narrative'])}`",
            "",
            "## EconAgent / EconAI method differences",
            "",
            "The package contains `method_differences_scaffold.json`, populated from "
            "the frozen FinEvo contract and the primary EconAgent/EconAI papers. "
            "These are documented mechanism differences, not empirical superiority "
            "claims.",
            "",
            "## Denominator and failures",
            "",
            f"- Overall status counts: `{_json_cell(denominator['status_counts'])}`",
        ]
    )
    for stage in contract.stage_ids:
        lines.append(
            f"- `{stage}`: "
            f"`{_json_cell(dict(sorted(stage_counts.get(stage, {}).items())))}`"
        )
    lines.extend(
        [
            "",
            "All missing, failed, capability-no-go, budget-stopped, integrity-stopped, "
            "and nonterminal cells remain in `failure_ledger.json`; none are removed "
            "from the registered denominator.",
            "",
            "## Claim narrowing and non-claims",
            "",
        ]
    )
    for item in contract.non_claims:
        lines.append(f"- {item}")
    for gate_name in ("experiment_a", "experiment_c", "experiment_d", "narrative"):
        gate = gates[gate_name]
        if gate["status"] != "supported":
            lines.append(f"- `{gate_name}` no-go: {gate.get('claim_action') or gate.get('claim_boundary')}.")
    return "\n".join(lines) + "\n"


def _scientific_completion_status(
    *,
    denominator: Mapping[str, Any],
    release_controls: Mapping[str, Any],
    gates: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[bool, bool, bool]:
    matrix_complete = bool(
        denominator.get("pass") is True
        and release_controls.get("pass") is True
        and all(
            row["status"] == "complete"
            and row["scientific_eligible"] is True
            for row in rows
            if row["stage_id"] in CORE_STAGES
        )
    )
    claim_gates_supported = all(
        gate.get("status") == "supported" for gate in gates.values()
    )
    return (
        matrix_complete,
        claim_gates_supported,
        bool(matrix_complete and claim_gates_supported),
    )


def _write_package_files(
    root: Path,
    *,
    contract_path: Path,
    contract: PilotContract,
    rows: Sequence[Mapping[str, Any]],
    denominator: Mapping[str, Any],
    common_commit: str | None,
    gates: Mapping[str, Any],
    capability: Mapping[str, Any],
    cross_model: Mapping[str, Any],
    release_controls: Mapping[str, Any],
    rule_sensitivity: Mapping[str, Any] | None,
) -> tuple[Path, Path, bool]:
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise PilotEvidenceError(f"temporary package directory is not empty: {root}")
    contract_target = root / "contract" / contract_path.name
    contract_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(contract_path, contract_target)
    if load_pilot_contract(contract_target).canonical_hash != contract.canonical_hash:
        raise PilotEvidenceError("copied contract failed hash revalidation")

    claims = _claims(gates, denominator=denominator)
    experiment_b = _experiment_b_summary(rows)
    sanitized_rows = [
        {
            key: _json_copy(row[key])
            for key in (
                "run_id",
                "contract_id",
                "stage_id",
                "model_id",
                "requested_model",
                "arm_id",
                "narrative_id",
                "environment_seed",
                "decoding_seed",
                "utility_profile_id",
                "shock_id",
                "budget_bucket",
                "num_agents",
                "episode_length",
                "execution_mode",
                "status",
                "failure",
                "artifact_kind",
                "artifact_sha256",
                "scientific_eligible",
                "metrics",
                "gate_evidence",
                "capability",
                "narrative",
            )
        }
        for row in rows
    ]
    aggregate = {
        "schema_version": PILOT_EVIDENCE_SCHEMA_VERSION,
        "evidence_namespace": EVIDENCE_NAMESPACE,
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "pilot_tag": contract.implementation["required_git_tag"],
        "resolved_git_commit": common_commit,
        "denominator": _json_copy(denominator),
        "claim_gates": _json_copy(gates),
        "experiment_b": _json_copy(experiment_b),
        "model_capability": _json_copy(capability),
        "cross_model": _json_copy(cross_model),
        "release_controls": _json_copy(release_controls),
        "experiment_c_rule_sensitivity": _json_copy(
            release_controls.get("experiment_c_sensitivity", {})
        ),
        "claims": claims,
        "rows": sanitized_rows,
    }
    _atomic_bytes(root / "aggregate.json", _pretty_bytes(aggregate))
    _atomic_bytes(root / "aggregate.csv", _aggregate_csv(sanitized_rows))

    failures = [
        {
            "run_id": row["run_id"],
            "stage_id": row["stage_id"],
            "model_id": row["model_id"],
            "arm_id": row["arm_id"],
            "narrative_id": row["narrative_id"],
            "environment_seed": row["environment_seed"],
            "status": row["status"],
            "failure": row["failure"],
        }
        for row in rows
        if row["status"] != "complete"
    ]
    failure_ledger = {
        "schema_version": PILOT_FAILURE_LEDGER_SCHEMA_VERSION,
        "contract_sha256": contract.canonical_hash,
        "denominator": _json_copy(denominator),
        "rows": failures,
    }
    _atomic_bytes(root / "failure_ledger.json", _pretty_bytes(failure_ledger))
    _atomic_bytes(
        root / "method_differences_scaffold.json",
        _pretty_bytes(_method_scaffold()),
    )
    if rule_sensitivity is not None:
        _atomic_bytes(
            root / "experiment_c_rule_sensitivity.json",
            _pretty_bytes(rule_sensitivity),
        )
    _atomic_bytes(
        root / "reviewer_report.md",
        _report_markdown(
            contract=contract,
            denominator=denominator,
            claims=claims,
            gates=gates,
            experiment_b=experiment_b,
            rows=rows,
            capability=capability,
            cross_model=cross_model,
            release_controls=release_controls,
            rule_sensitivity=rule_sensitivity,
        ).encode("utf-8"),
    )
    (
        scientific_matrix_complete,
        claim_gates_supported,
        scientific_complete,
    ) = _scientific_completion_status(
        denominator=denominator,
        release_controls=release_controls,
        gates=gates,
        rows=rows,
    )
    published_files = [
        "aggregate.csv",
        "aggregate.json",
        f"contract/{contract_path.name}",
        "failure_ledger.json",
        "method_differences_scaffold.json",
        "reviewer_report.md",
    ]
    if rule_sensitivity is not None:
        published_files.append("experiment_c_rule_sensitivity.json")
    package_manifest = {
        "schema_version": PILOT_EVIDENCE_SCHEMA_VERSION,
        "evidence_namespace": EVIDENCE_NAMESPACE,
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "pilot_tag": contract.implementation["required_git_tag"],
        "resolved_git_commit": common_commit,
        "scientific_matrix_complete": scientific_matrix_complete,
        "scientific_claim_gates_supported": claim_gates_supported,
        "scientific_complete": scientific_complete,
        "release_controls": _json_copy(release_controls),
        "claim_gates": _json_copy(gates),
        "published_files": sorted(published_files),
        "excluded_sources": [
            HISTORICAL_SCOPE,
            "diagnostic artifacts as scientific evidence",
            "raw prompts and raw provider outputs",
        ],
    }
    manifest_path = root / "package_manifest.json"
    _atomic_bytes(manifest_path, _pretty_bytes(package_manifest))

    checksum_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.json"
    )
    checksum_payload = {
        "schema_version": PILOT_CHECKSUM_SCHEMA_VERSION,
        "contract_sha256": contract.canonical_hash,
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
                "byte_size": path.stat().st_size,
            }
            for path in checksum_files
        ],
    }
    checksums_path = root / "checksums.json"
    _atomic_bytes(checksums_path, _pretty_bytes(checksum_payload))
    for row in checksum_payload["files"]:
        path = root / row["path"]
        if (
            _sha256_file(path) != row["sha256"]
            or path.stat().st_size != row["byte_size"]
        ):
            raise PilotEvidenceError("package checksum self-verification failed")
    raw_storage = release_controls.get("budget_ledger", {}).get(
        "raw_root_storage_bytes"
    )
    package_storage = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    if (
        not _is_finite_scalar(raw_storage)
        or float(raw_storage) + package_storage
        > float(contract.budgets["max_storage_bytes"])
    ):
        raise PilotEvidenceError(
            "raw evidence plus reviewer package exceeds the frozen 5 GB cap"
        )
    return manifest_path, checksums_path, scientific_complete


def build_pilot_evidence_package(
    *,
    contract_path: str | Path,
    run_ledger_path: str | Path,
    raw_root: str | Path,
    build_root: str | Path,
) -> PilotEvidencePackage:
    """Validate raw pilot evidence and atomically build the reviewer package.

    ``build_root`` is normally ``evidence``.  The function always creates
    ``build_root/current_v2/pilot-v1`` and refuses to overwrite an existing
    package.  Tests and dry runs should pass a temporary build root.
    """

    contract_source = Path(contract_path).resolve()
    contract = load_pilot_contract(contract_source)
    raw = Path(raw_root).resolve()
    if not raw.is_dir():
        raise PilotEvidenceError(f"pilot raw root does not exist: {raw}")
    ledger = _strict_json_load(Path(run_ledger_path).resolve())
    rows, denominator, common_commit = _normalize_ledger(
        contract,
        ledger,
        raw_root=raw,
    )
    gates = {
        "experiment_a": _experiment_a_gate(contract, rows),
        "experiment_c": _experiment_c_gate(contract, rows),
        "experiment_d": _experiment_d_gate(contract, rows),
        "narrative": _narrative_gate(contract, rows),
    }
    capability = _capability_by_model(rows)
    cross_model = _cross_model_summary(contract, rows, capability)
    rule_sensitivity, sensitivity_control = (
        _validated_experiment_c_sensitivity(
            contract,
            raw_root=raw,
            rows=rows,
            common_commit=common_commit,
        )
    )
    release_controls = _validated_release_controls(
        contract,
        raw_root=raw,
        rows=rows,
        common_commit=common_commit,
    )
    release_controls["experiment_c_sensitivity"] = sensitivity_control
    release_controls["pass"] = bool(
        release_controls.get("pass") and sensitivity_control.get("pass")
    )

    base = Path(build_root).resolve()
    target = base / "current_v2" / "pilot-v1"
    if target.exists():
        raise PilotEvidenceError(f"refusing to overwrite evidence package: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".pilot-v1-build-", dir=target.parent)
    )
    try:
        manifest, checksums, scientific_complete = _write_package_files(
            temporary,
            contract_path=contract_source,
            contract=contract,
            rows=rows,
            denominator=denominator,
            common_commit=common_commit,
            gates=gates,
            capability=capability,
            cross_model=cross_model,
            release_controls=release_controls,
            rule_sensitivity=rule_sensitivity,
        )
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return PilotEvidencePackage(
        package_dir=target,
        manifest_path=target / manifest.name,
        checksums_path=target / checksums.name,
        contract_hash=contract.canonical_hash,
        scientific_complete=scientific_complete,
        claim_gates=_json_copy(gates),
    )


__all__ = [
    "CURRENT_SCIENTIFIC_SCOPE",
    "EVIDENCE_NAMESPACE",
    "HISTORICAL_SCOPE",
    "PILOT_CHECKSUM_SCHEMA_VERSION",
    "PILOT_EVIDENCE_SCHEMA_VERSION",
    "PILOT_FAILURE_LEDGER_SCHEMA_VERSION",
    "PILOT_TERMINAL_SUMMARY_SCHEMA_VERSION",
    "PilotEvidenceError",
    "PilotEvidencePackage",
    "build_pilot_evidence_package",
    "write_terminal_summary",
]
