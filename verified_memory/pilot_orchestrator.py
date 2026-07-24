"""Fail-closed orchestration for the preregistered FinEvo micro-pilot.

This module is the execution boundary behind :mod:`run_pilot`.  It deliberately
keeps three concerns separate:

* contract expansion and an ITT ledger that never drops a scheduled cell;
* release/provenance and cross-stage budget gates before a paid dispatch; and
* stage-specific execution, including an explicitly non-scientific scripted
  A--D integration matrix.

The scripted matrix is a plumbing diagnostic.  Its artifacts always declare
``diagnostic_only=true`` and ``scientific_evidence=false`` and are stored under
their own namespace, so they cannot be mistaken for provider results.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence

from llm_providers import (
    PINNED_PROVIDER_SDK_VERSIONS,
    MultiModelLLM,
    create_llm_provider,
)

from .artifacts import verify_manifest
from .budget import BudgetLimits, RunBudget, UsageRecord
from .failure_artifacts import write_failure_receipt
from .m0_utility import UtilityConfig
from .m2_episodic import EvidenceLinkedEpisodicTrack
from .m3_semantic import VerifiedSemanticRuleTrack
from .pilot_analysis import summarize_run
from .pilot_budget import (
    PilotBudgetCaps,
    PilotBudgetError,
    PilotBudgetLedger,
    RunProjection,
    preflight_p95,
)
from .pilot_calibration import (
    build_q_ref_resolution,
    expand_stage0_ofat,
    q_ref_run_config,
    select_stage0_profile,
)
from .pilot_capability import build_capability_tasks, run_capability_gate
from .pilot_checkpoint import (
    PilotCheckpoint,
    build_closed_loop_preflight_checkpoint,
    build_pilot_checkpoint,
    verify_closed_loop_preflight_checkpoint,
)
from .pilot_continuation import (
    DEFAULT_NARRATIVES,
    DEFAULT_TREATMENTS,
    MEMORY_PULSE_CONTRACT,
    MEMORY_PULSE_TREATMENTS,
    NARRATIVE_PULSE_CONTRACT,
    run_pilot_continuations,
    run_pilot_narratives,
)
from .pilot_provider_catalog import (
    PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION,
    validate_live_provider_catalog,
    verify_provider_catalog_receipt,
)
from .pilot_release_attestation import (
    PILOT_RELEASE_ATTESTATION_SCHEMA_VERSION,
    PilotReleaseAttestationError,
    verify_pilot_release_attestation,
)
from .scientific_release_attestation import (
    SCIENTIFIC_RELEASE_ATTESTATION_SCHEMA_VERSION,
    ScientificReleaseAttestationError,
    verify_scientific_release_attestation,
)
from .pilot_sensitivity import (
    ALTERNATIVE_SUCCESS_WEIGHTS,
    OUTCOME_DEFINITIONS,
    replay_rule_sensitivity,
)
from .pilot_contract import (
    PilotContract,
    PilotContractError,
    PilotRunSpec,
    ProviderRequestProfile,
    canonical_sha256,
    load_pilot_contract,
)
from .pilot_evidence import (
    CURRENT_SCIENTIFIC_SCOPE,
    PILOT_TERMINAL_SUMMARY_SCHEMA_VERSION,
    PilotEvidenceError,
    _validate_capability_v3,
    write_terminal_summary,
)
from .runner import (
    ShockEvent,
    VerifiedRunConfig,
    run_verified_experiment,
    verify_provider_call_journal,
)
from .runner_artifacts import (
    load_verified_run_artifacts,
    write_verified_run_artifacts,
)
from .scripted_provider import ScriptedDiagnosticProvider


PILOT_RUN_LEDGER_SCHEMA_VERSION = "finevo-pilot-run-ledger-v1"
PILOT_RUN_LEDGER_SCHEMA_VERSION_V2 = "finevo-pilot-run-ledger-v2"
PILOT_STAGE_RECEIPT_SCHEMA_VERSION = "finevo-pilot-stage-receipt-v1"
PILOT_STAGE_RECEIPT_SCHEMA_VERSION_V2 = "finevo-pilot-stage-receipt-v2"
PILOT_DEVELOPMENT_MATRIX_SCHEMA_VERSION = "finevo-pilot-development-matrix-v1"
PILOT_OFFLINE_ADMISSION_SCHEMA_VERSION = "finevo-pilot-offline-admission-v1"
PILOT_PROJECTION_SCHEMA_VERSION = "finevo-pilot-projection-p95-v1"
PILOT_EXPERIMENT_C_SENSITIVITY_SCHEMA_VERSION = "finevo-experiment-c-sensitivity-v1"
PILOT_SCIENTIFIC_LAUNCH_INPUT_SCHEMA_VERSION = "finevo-scientific-launch-input-v1"
PILOT_PREFLIGHT_CHECKPOINT_RECEIPT_SCHEMA_VERSION = (
    "finevo-preflight-checkpoint-exactness-v1"
)
PILOT_BOUND_ARTIFACT_CANONICALIZATION = "json-sort-keys-utf8-v1"

CORE_STAGE_IDS = (
    "experiment-a",
    "experiment-b",
    "experiment-c",
    "experiment-d",
)
SCIENTIFIC_STAGE_IDS = (
    "stage0-calibration",
    *CORE_STAGE_IDS,
    "cross-model-sentinels",
)

CAPABILITY_EXECUTION_MODES = frozenset({"capability_probe", "closed_loop_preflight"})

DEFAULT_RAW_ROOT = (
    Path(__file__).resolve().parents[1] / "experiment_results" / "pilot-v1" / "raw"
)
DEFAULT_ENV_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
PILOT_EXECUTION_LOCK_FILENAME = ".real-stage-execution.lock"

TERMINAL_RUN_STATUSES = frozenset(
    {"complete", "failed", "budget-stopped", "integrity-stopped", "capability-no-go"}
)
HARD_STOP_RUN_STATUSES = frozenset({"budget-stopped", "integrity-stopped"})


class PilotOrchestrationError(RuntimeError):
    """Raised before dispatch when pilot execution is not contract-safe."""


@contextmanager
def _exclusive_real_stage_lock(
    raw_root: str | Path,
    *,
    stage_id: str,
) -> Iterable[Path]:
    """Hold one cross-process lock for the complete real-stage transaction.

    The lock is scoped to the raw pilot root and intentionally excludes the
    development fake matrix and evidence publisher.  A second real-stage
    process fails before loading either ledger, so it cannot observe stale
    state and dispatch the same provider cells twice.
    """

    root = Path(raw_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / PILOT_EXECUTION_LOCK_FILENAME
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PilotOrchestrationError(
                "another real pilot stage process holds the raw-root "
                "execution lock; no ledger was loaded and no provider call "
                "was dispatched"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            _canonical_json(
                {
                    "schema_version": "finevo-pilot-execution-lock-v1",
                    "pid": os.getpid(),
                    "stage_id": stage_id,
                    "acquired_at": _utc_now(),
                }
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        yield path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically persist one finite JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_bound_json(path: Path, value: Mapping[str, Any]) -> None:
    """Persist an immutable, self-hashed pilot control artifact."""

    _atomic_json(path, value)
    path.chmod(0o444)


def _atomic_json_no_overwrite(path: Path, value: Mapping[str, Any]) -> None:
    """Persist one append-only control receipt and refuse replacement."""

    if path.exists():
        raise PilotOrchestrationError(
            f"immutable pilot control receipt already exists: {path}"
        )
    _atomic_json(path, value)
    path.chmod(0o444)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PilotOrchestrationError(f"required artifact is missing: {path}") from exc
    if not isinstance(value, dict):
        raise PilotOrchestrationError(f"artifact root must be an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise PilotOrchestrationError(f"required source file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_content_sha256(value: Mapping[str, Any]) -> str:
    """Hash one bound artifact while excluding only its self-hash field."""

    copied = _json_copy(value)
    integrity = copied.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("content_sha256", None)
    return canonical_sha256(copied)


def _seal_bound_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _json_copy(value)
    integrity = payload.setdefault("integrity", {})
    if not isinstance(integrity, dict):
        raise PilotOrchestrationError("bound artifact integrity must be an object")
    integrity["canonicalization"] = PILOT_BOUND_ARTIFACT_CANONICALIZATION
    integrity.pop("content_sha256", None)
    integrity["content_sha256"] = _bound_content_sha256(payload)
    return payload


def _verify_bound_payload(
    value: Mapping[str, Any],
    *,
    contract: PilotContract,
    schema_version: str,
    paid: GitProvenance | None,
    artifact_name: str,
) -> None:
    if value.get("schema_version") != schema_version:
        raise PilotOrchestrationError(
            f"{artifact_name} has an unsupported schema version"
        )
    bindings = value.get("bindings")
    if not isinstance(bindings, Mapping):
        raise PilotOrchestrationError(f"{artifact_name} bindings are malformed")
    if bindings.get("contract_sha256") != contract.canonical_hash:
        raise PilotOrchestrationError(f"{artifact_name} contract hash mismatch")
    expected_tag = str(contract.implementation["required_git_tag"])
    if bindings.get("git_tag") != expected_tag:
        raise PilotOrchestrationError(f"{artifact_name} pilot tag mismatch")
    commit = bindings.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise PilotOrchestrationError(f"{artifact_name} git commit is malformed")
    if paid is not None and (
        bindings.get("git_tag") != paid.git_tag
        or bindings.get("git_commit") != paid.head_commit
    ):
        raise PilotOrchestrationError(
            f"{artifact_name} differs from the active paid tag/commit"
        )
    integrity = value.get("integrity")
    if not isinstance(integrity, Mapping):
        raise PilotOrchestrationError(f"{artifact_name} integrity is malformed")
    if integrity.get(
        "canonicalization"
    ) != PILOT_BOUND_ARTIFACT_CANONICALIZATION or integrity.get(
        "content_sha256"
    ) != _bound_content_sha256(
        value
    ):
        raise PilotOrchestrationError(f"{artifact_name} content hash mismatch")


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _git(repo_root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise PilotOrchestrationError(detail)
    return result.stdout.strip()


@dataclass(frozen=True, slots=True)
class GitProvenance:
    """Resolved annotated-tag binding required for a paid pilot."""

    git_tag: str
    head_commit: str
    tag_commit: str
    tag_object_type: str
    worktree_clean: bool
    contract_binding: Mapping[str, Any]
    release_attestation: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "contract_binding": dict(self.contract_binding),
            "release_attestation": (
                None
                if self.release_attestation is None
                else dict(self.release_attestation)
            ),
        }


def _load_scientific_launch_input(
    contract: PilotContract,
    path: str | Path | None,
) -> dict[str, Any]:
    """Load one explicit, self-hashed V2 CI/contract selection.

    Dynamic GitHub run and job IDs cannot be inferred from "latest" state.
    The operator must place this small JSON document under the ignored raw
    pilot root after the annotated-tag CI run has completed.
    """

    if path is None:
        raise PilotOrchestrationError(
            "V2 paid dispatch requires an explicit scientific launch input "
            "containing immutable ci_run_selection and contract_binding"
        )
    source_path = Path(path)
    if source_path.is_symlink():
        raise PilotOrchestrationError("scientific launch input must not be a symlink")
    launch_path = source_path.resolve()
    if not launch_path.is_file():
        raise PilotOrchestrationError(
            f"scientific launch input is missing or not a regular file: "
            f"{launch_path}"
        )
    payload = _read_json(launch_path)
    required = {
        "schema_version",
        "contract_sha256",
        "ci_run_selection",
        "contract_binding",
        "launch_input_sha256",
    }
    if set(payload) != required:
        raise PilotOrchestrationError(
            "scientific launch input fields must be exactly " f"{sorted(required)}"
        )
    if payload.get("schema_version") != PILOT_SCIENTIFIC_LAUNCH_INPUT_SCHEMA_VERSION:
        raise PilotOrchestrationError(
            "scientific launch input has an unsupported schema"
        )
    if payload.get("contract_sha256") != contract.canonical_hash:
        raise PilotOrchestrationError("scientific launch input contract hash mismatch")
    unsigned = dict(payload)
    claimed_hash = unsigned.pop("launch_input_sha256")
    if claimed_hash != canonical_sha256(unsigned):
        raise PilotOrchestrationError("scientific launch input self-hash mismatch")
    if not isinstance(payload.get("ci_run_selection"), Mapping):
        raise PilotOrchestrationError("scientific launch input lacks ci_run_selection")
    binding = payload.get("contract_binding")
    if not isinstance(binding, Mapping):
        raise PilotOrchestrationError("scientific launch input lacks contract_binding")
    if binding.get("contract_canonical_sha256") != contract.canonical_hash:
        raise PilotOrchestrationError(
            "scientific launch contract_binding canonical hash mismatch"
        )
    return payload


def verify_paid_provenance(
    contract: PilotContract,
    *,
    repo_root: str | Path,
    scientific_launch_input_path: str | Path | None = None,
) -> GitProvenance:
    """Require a clean HEAD at the contract's peeled annotated release tag.

    A lightweight tag is rejected even if it resolves to the same commit.
    Ignored raw results do not dirty the worktree; tracked or untracked source
    files do.  V2 additionally requires a predeclared, self-hashed CI run/job
    selection and scientific contract binding.
    """

    root = Path(repo_root).resolve()
    is_v2 = getattr(contract, "schema_version", "").endswith("-v2")
    launch_input = (
        _load_scientific_launch_input(
            contract,
            scientific_launch_input_path,
        )
        if is_v2
        else None
    )
    required_tag = str(contract.implementation["required_git_tag"])
    tag_ref = f"refs/tags/{required_tag}"
    tag_type = _git(root, "cat-file", "-t", tag_ref)
    if tag_type != "tag":
        raise PilotOrchestrationError(
            f"{required_tag!r} must exist as an annotated tag; observed {tag_type!r}"
        )
    head = _git(root, "rev-parse", "HEAD")
    peeled = _git(root, "rev-parse", f"{tag_ref}^{{commit}}")
    if head != peeled:
        raise PilotOrchestrationError(
            "paid pilot requires HEAD to equal the peeled release-tag commit"
        )
    dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=all"))
    if dirty:
        raise PilotOrchestrationError("paid pilot requires a clean worktree")
    p0_base = str(contract.implementation["p0_base_commit"])
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", p0_base, head],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise PilotOrchestrationError(
            "release commit does not descend from the frozen P0 merge commit"
        )
    binding = contract.validate_provenance(head, required_tag)
    if is_v2:
        if contract.release_requirements is None:  # pragma: no cover - parser
            raise PilotOrchestrationError(
                "V2 contract lacks static release requirements"
            )
        assert launch_input is not None
        try:
            release_attestation = verify_scientific_release_attestation(
                root,
                release_requirements=contract.release_requirements.to_dict(),
                ci_run_selection=launch_input["ci_run_selection"],
                contract_binding=launch_input["contract_binding"],
            ).to_dict()
        except ScientificReleaseAttestationError as exc:
            raise PilotOrchestrationError(
                f"scientific release/CI attestation failed: {exc}"
            ) from exc
    else:
        try:
            release_attestation = verify_pilot_release_attestation(root).to_dict()
        except PilotReleaseAttestationError as exc:
            raise PilotOrchestrationError(
                f"remote release/CI attestation failed: {exc}"
            ) from exc
    return GitProvenance(
        git_tag=required_tag,
        head_commit=head,
        tag_commit=peeled,
        tag_object_type=tag_type,
        worktree_clean=True,
        contract_binding=binding,
        release_attestation=release_attestation,
    )


def _persist_release_attestation(
    raw_root: Path,
    paid: GitProvenance,
) -> Path:
    """Write one immutable release receipt and reject later drift."""

    payload = paid.release_attestation
    if not isinstance(payload, Mapping):
        raise PilotOrchestrationError(
            "paid provenance lacks the remote release/CI attestation"
        )
    if payload.get("schema_version") not in {
        PILOT_RELEASE_ATTESTATION_SCHEMA_VERSION,
        SCIENTIFIC_RELEASE_ATTESTATION_SCHEMA_VERSION,
    } or (
        payload.get("status") != "pass"
        or payload.get("head_commit") != paid.head_commit
    ):
        raise PilotOrchestrationError(
            "remote release/CI attestation does not bind the paid HEAD"
        )
    unsigned = dict(payload)
    claimed = unsigned.pop("attestation_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise PilotOrchestrationError(
            "remote release/CI attestation content hash mismatch"
        )
    path = raw_root / "release_attestation.json"
    if path.exists():
        existing = _read_json(path)
        if existing != dict(payload):
            raise PilotOrchestrationError(
                "release attestation drifted after the pilot ledger was created"
            )
        return path
    _atomic_bound_json(path, dict(payload))
    return path


class PilotRunLedger:
    """Durable ITT ledger: every expanded cell remains present through failure."""

    def __init__(
        self,
        path: str | Path,
        *,
        contract_hash: str,
        tamper_evident: bool = False,
    ) -> None:
        self.path = Path(path)
        self.contract_hash = contract_hash
        if not isinstance(tamper_evident, bool):
            raise TypeError("tamper_evident must be boolean")
        self.tamper_evident = tamper_evident
        self.schema_version = (
            PILOT_RUN_LEDGER_SCHEMA_VERSION_V2
            if tamper_evident
            else PILOT_RUN_LEDGER_SCHEMA_VERSION
        )
        if self.path.exists():
            self._state = self._load()
        else:
            self._state = {
                "schema_version": self.schema_version,
                "contract_hash": contract_hash,
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "runs": {},
            }
            if self.tamper_evident:
                self._state["events"] = []
                self._append_event(
                    "genesis",
                    {
                        "contract_hash": contract_hash,
                        "runs_sha256": canonical_sha256({}),
                    },
                )
            self._write()

    def _load(self) -> dict[str, Any]:
        value = _read_json(self.path)
        if value.get("schema_version") != self.schema_version:
            raise PilotOrchestrationError("unsupported pilot run ledger schema")
        if value.get("contract_hash") != self.contract_hash:
            raise PilotOrchestrationError("pilot run ledger contract mismatch")
        if not isinstance(value.get("runs"), dict):
            raise PilotOrchestrationError("pilot run ledger rows are malformed")
        if self.tamper_evident:
            self._verify_event_chain(value)
            unsigned = dict(value)
            claimed = unsigned.pop("ledger_sha256", None)
            if claimed != canonical_sha256(unsigned):
                raise PilotOrchestrationError("pilot run ledger self-hash mismatch")
        return value

    def _append_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if not self.tamper_evident:
            return
        events = self._state.setdefault("events", [])
        if not isinstance(events, list):
            raise PilotOrchestrationError("pilot run ledger events are malformed")
        previous = events[-1]["event_sha256"] if events else "0" * 64
        event = {
            "event_index": len(events),
            "event_type": str(event_type),
            "created_at": _utc_now(),
            "previous_event_sha256": previous,
            "payload": _json_copy(payload),
        }
        event["event_sha256"] = canonical_sha256(event)
        events.append(event)

    @staticmethod
    def _verify_event_chain(value: Mapping[str, Any]) -> None:
        events = value.get("events")
        runs = value.get("runs")
        if not isinstance(events, list) or not events:
            raise PilotOrchestrationError(
                "pilot run ledger v2 requires a non-empty event chain"
            )
        if not isinstance(runs, Mapping):
            raise PilotOrchestrationError("pilot run ledger rows are malformed")
        previous = "0" * 64
        for index, row in enumerate(events):
            if not isinstance(row, Mapping):
                raise PilotOrchestrationError("pilot run ledger event is malformed")
            unsigned = dict(row)
            digest = unsigned.pop("event_sha256", None)
            if (
                row.get("event_index") != index
                or row.get("previous_event_sha256") != previous
                or digest != canonical_sha256(unsigned)
            ):
                raise PilotOrchestrationError("pilot run ledger event chain mismatch")
            payload = row.get("payload")
            if not isinstance(payload, Mapping):
                raise PilotOrchestrationError(
                    "pilot run ledger event payload is malformed"
                )
            event_type = row.get("event_type")
            if index == 0:
                if (
                    event_type != "genesis"
                    or payload.get("contract_hash") != value.get("contract_hash")
                    or payload.get("runs_sha256") != canonical_sha256({})
                ):
                    raise PilotOrchestrationError(
                        "pilot run ledger genesis event mismatch"
                    )
            elif event_type == "runs_registered":
                registered = payload.get("registered_specs_sha256")
                if not isinstance(registered, Mapping) or not registered:
                    raise PilotOrchestrationError(
                        "pilot run ledger registration event is malformed"
                    )
                for run_id, spec_sha256 in registered.items():
                    current = runs.get(run_id)
                    if (
                        not isinstance(current, Mapping)
                        or spec_sha256
                        != canonical_sha256(current.get("spec"))
                    ):
                        raise PilotOrchestrationError(
                            "pilot run ledger registration event differs from rows"
                        )
            elif event_type == "run_finalized":
                run_id = payload.get("run_id")
                current = runs.get(run_id)
                if not isinstance(current, Mapping):
                    raise PilotOrchestrationError(
                        "pilot run ledger finalization references an unknown run"
                    )
                terminal_state = {
                    "status": current.get("status"),
                    "artifact": current.get("artifact"),
                    "failure": current.get("failure"),
                }
                if payload.get("terminal_state_sha256") != canonical_sha256(
                    terminal_state
                ):
                    raise PilotOrchestrationError(
                        "pilot run ledger finalization differs from its row"
                    )
            else:
                raise PilotOrchestrationError(
                    f"unknown pilot run ledger event type: {event_type!r}"
                )
            previous = str(digest)
        last_payload = events[-1].get("payload")
        if (
            not isinstance(last_payload, Mapping)
            or last_payload.get("runs_sha256") != canonical_sha256(runs)
        ):
            raise PilotOrchestrationError(
                "pilot run ledger event head does not bind the current rows"
            )

    def _write(self) -> None:
        self._state["updated_at"] = _utc_now()
        if self.tamper_evident:
            unsigned = dict(self._state)
            unsigned.pop("ledger_sha256", None)
            self._state["ledger_sha256"] = canonical_sha256(unsigned)
        _atomic_json(self.path, self._state)

    def register(self, specs: Sequence[PilotRunSpec]) -> None:
        changed = False
        for spec in specs:
            row = self._state["runs"].get(spec.run_id)
            spec_value = spec.to_dict()
            if row is None:
                self._state["runs"][spec.run_id] = {
                    "spec": spec_value,
                    "status": "scheduled",
                    "artifact": None,
                    "failure": None,
                    "registered_at": _utc_now(),
                    "terminal_at": None,
                }
                changed = True
            elif row.get("spec") != spec_value:
                raise PilotOrchestrationError(
                    f"run {spec.run_id} was registered with a different spec"
                )
        if changed:
            self._append_event(
                "runs_registered",
                {
                    "registered_specs_sha256": {
                        spec.run_id: canonical_sha256(spec.to_dict())
                        for spec in specs
                        if spec.run_id in self._state["runs"]
                    },
                    "runs_sha256": canonical_sha256(self._state["runs"]),
                },
            )
            self._write()

    def status(self, run_id: str) -> str:
        row = self._state["runs"].get(run_id)
        if not isinstance(row, Mapping):
            raise KeyError(run_id)
        return str(row["status"])

    def is_terminal(self, run_id: str) -> bool:
        return self.status(run_id) in TERMINAL_RUN_STATUSES

    def finalize(
        self,
        run_id: str,
        *,
        status: str,
        artifact: str | None,
        failure: Mapping[str, Any] | None = None,
    ) -> None:
        if status not in TERMINAL_RUN_STATUSES:
            raise ValueError("run status must be terminal")
        row = self._state["runs"].get(run_id)
        if row is None:
            raise PilotOrchestrationError(f"run {run_id} was not registered")
        desired_failure = dict(failure) if failure else None
        if row["status"] in TERMINAL_RUN_STATUSES:
            if (
                row["status"] != status
                or row.get("artifact") != artifact
                or row.get("failure") != desired_failure
            ):
                raise PilotOrchestrationError(
                    f"run {run_id} was already finalized differently"
                )
            return
        row.update(
            {
                "status": status,
                "artifact": artifact,
                "failure": desired_failure,
                "terminal_at": _utc_now(),
            }
        )
        self._append_event(
            "run_finalized",
            {
                "run_id": run_id,
                "terminal_state_sha256": canonical_sha256(
                    {
                        "status": status,
                        "artifact": artifact,
                        "failure": desired_failure,
                    }
                ),
                "runs_sha256": canonical_sha256(self._state["runs"]),
            },
        )
        self._write()

    def stop_pending(
        self,
        specs: Sequence[PilotRunSpec],
        *,
        status: str,
        failure: Mapping[str, Any],
    ) -> None:
        for spec in specs:
            if not self.is_terminal(spec.run_id):
                self.finalize(
                    spec.run_id,
                    status=status,
                    artifact=None,
                    failure=failure,
                )

    def snapshot(self) -> dict[str, Any]:
        return _json_copy(self._state)


def _budget_caps(contract: PilotContract) -> PilotBudgetCaps:
    budgets = contract.budgets
    return PilotBudgetCaps(
        total_usd=float(budgets["total_usd"]),
        max_completions=int(budgets["max_provider_completions"]),
        completion_scope=str(budgets["completion_scope"]),
        max_storage_bytes=int(budgets["max_storage_bytes"]),
        stage_usd_caps={
            str(key): float(value) for key, value in budgets["stage_usd_caps"].items()
        },
        automatic_reserve_usd=float(budgets["automatic_reserve_usd"]),
    )


def _shock_events(
    contract: PilotContract,
    shock_id: str,
    *,
    episode_length: int,
) -> tuple[ShockEvent, ...]:
    shock = contract.shocks[shock_id]
    events: list[ShockEvent] = []
    for interval in shock["schedule"]:
        for decision_t in range(int(interval["start"]), int(interval["end"]) + 1):
            if decision_t >= episode_length:
                continue
            events.append(
                ShockEvent(
                    decision_t=decision_t,
                    phase=str(interval["phase"]),
                    interest_rate=float(interval["interest_rate"]),
                )
            )
    if [event.decision_t for event in events] != list(range(episode_length)):
        raise PilotOrchestrationError(
            f"shock {shock_id!r} does not cover the full declared horizon"
        )
    return tuple(events)


def _utility_from_mapping(value: Mapping[str, Any]) -> UtilityConfig:
    required = (
        "rho",
        "labor_weight",
        "inverse_frisch",
        "consumption_scale",
        "discount_factor",
    )
    if any(value.get(field) is None for field in required):
        raise PilotOrchestrationError("utility profile is unresolved")
    return UtilityConfig(
        rho=float(value["rho"]),
        labor_weight=float(value["labor_weight"]),
        inverse_frisch=float(value["inverse_frisch"]),
        consumption_scale=float(value["consumption_scale"]),
        discount_factor=float(value["discount_factor"]),
        max_labor_hours=168.0,
        budget_tolerance=1e-8,
    )


def _load_verified_q_ref(
    contract: PilotContract,
    *,
    raw_root: Path,
    paid: GitProvenance | None,
) -> dict[str, Any]:
    path = raw_root / "q-ref-resolution" / "q_ref_resolution.json"
    value = _read_json(path)
    # Development fixtures intentionally remain diagnostics and are never
    # accepted by the paid path.
    if paid is None:
        declared_contract = (
            value.get("bindings", {}).get("contract_hash")
            if isinstance(value.get("bindings"), Mapping)
            else None
        )
        if declared_contract not in {None, contract.canonical_hash}:
            raise PilotOrchestrationError("diagnostic q_ref contract hash mismatch")
        return value
    bindings = value.get("bindings")
    if isinstance(bindings, dict) and "contract_hash" in bindings:
        # v1 calibration named this field ``contract_hash``.  The sealed pilot
        # binding uses the common ``contract_sha256`` name.
        bindings.setdefault("contract_sha256", bindings["contract_hash"])
    _verify_bound_payload(
        value,
        contract=contract,
        schema_version="finevo-q-ref-resolution-v1",
        paid=paid,
        artifact_name="q_ref resolution",
    )
    expected_manifest = (
        raw_root
        / "q-ref-resolution"
        / "runs"
        / contract.expand(stage="q-ref-resolution")[0].run_id
        / "manifest.json"
    )
    declared_manifest = value.get("source_manifest")
    if declared_manifest != str(expected_manifest):
        raise PilotOrchestrationError("q_ref source manifest path mismatch")
    verification = verify_manifest(expected_manifest.parent)
    if (
        bindings.get("source_manifest_sha256") != verification.manifest_sha256
        or _file_sha256(expected_manifest) != verification.manifest_sha256
    ):
        raise PilotOrchestrationError("q_ref source manifest hash mismatch")
    source_manifest = _read_json(expected_manifest)
    if (
        source_manifest.get("git", {}).get("commit") != bindings.get("git_commit")
        or source_manifest.get("git", {}).get("dirty") is not False
    ):
        raise PilotOrchestrationError("q_ref source manifest git binding mismatch")
    source_provenance = _read_json(expected_manifest.parent / "provenance.json")
    details = source_provenance.get("details")
    if (
        not isinstance(details, Mapping)
        or details.get("contract_sha256") != contract.canonical_hash
        or details.get("run_spec")
        != contract.expand(stage="q-ref-resolution")[0].to_dict()
    ):
        raise PilotOrchestrationError("q_ref source provenance mismatch")
    if bindings.get("environment_source_hash") != _file_sha256(DEFAULT_ENV_CONFIG):
        raise PilotOrchestrationError("q_ref environment source hash mismatch")
    source_result = load_verified_run_artifacts(expected_manifest.parent)
    recomputed = build_q_ref_resolution(
        source_result,
        contract_hash=contract.canonical_hash,
        environment_source_hash=_file_sha256(DEFAULT_ENV_CONFIG),
    )
    for key in (
        "status",
        "q_ref",
        "aggregation",
        "ledger_field",
        "row_count",
        "run_contract",
        "checks",
        "source",
    ):
        if value.get(key) != recomputed.get(key):
            raise PilotOrchestrationError(
                f"q_ref field {key!r} differs from its sealed runner source"
            )
    for key in (
        "contract_hash",
        "source_config_hash",
        "run_summary_hash",
        "ledger_hash",
        "environment_source_hash",
    ):
        if bindings.get(key) != recomputed["bindings"].get(key):
            raise PilotOrchestrationError(
                f"q_ref binding {key!r} differs from its sealed runner source"
            )
    return value


def _derive_stage0_absolute_flow_threshold(
    contract: PilotContract,
    *,
    selected_profile_id: str,
    sources: Sequence[tuple[PilotRunSpec, Any, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Derive the frozen C sensitivity threshold after Stage-0 selection.

    The selected profile is fixed first by the registered Stage-0 guardrails.
    This function then reads only that profile's two calibration ledgers and
    takes their pooled median ``flow_utility``.  No A--D treatment artifact is
    an input.
    """

    selected = [
        (spec, result, dict(manifest_binding))
        for spec, result, manifest_binding in sources
        if spec.utility_profile_id == selected_profile_id
    ]
    expected_seeds = tuple(contract.seeds["sets"]["calibration"])
    if len(selected) != 2 or {spec.environment_seed for spec, _, _ in selected} != set(
        expected_seeds
    ):
        raise PilotOrchestrationError(
            "absolute flow-utility threshold requires the selected profile's "
            "exact two calibration seeds"
        )

    values: list[float] = []
    source_rows: list[dict[str, Any]] = []
    for spec, result, manifest_binding in sorted(
        selected,
        key=lambda item: item[0].environment_seed,
    ):
        ledger = [dict(row) for row in result.stream("utility_ledger")]
        expected_count = spec.num_agents * spec.episode_length
        identities: set[tuple[int, int]] = set()
        run_values: list[float] = []
        for row in ledger:
            value = row.get("flow_utility")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise PilotOrchestrationError(
                    "Stage-0 flow_utility contains a non-finite value"
                )
            try:
                identity = (int(row["period"]), int(row["agent_id"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise PilotOrchestrationError(
                    "Stage-0 utility ledger identity is malformed"
                ) from exc
            if identity in identities:
                raise PilotOrchestrationError(
                    "Stage-0 utility ledger contains a duplicate period-agent row"
                )
            identities.add(identity)
            run_values.append(float(value))
        if len(ledger) != expected_count or len(identities) != expected_count:
            raise PilotOrchestrationError(
                "Stage-0 selected-profile utility ledger is incomplete"
            )
        values.extend(run_values)
        source_rows.append(
            {
                "run_id": spec.run_id,
                "environment_seed": spec.environment_seed,
                "manifest": manifest_binding.get("manifest"),
                "manifest_sha256": manifest_binding.get("manifest_sha256"),
                "row_count": len(ledger),
                "utility_ledger_sha256": canonical_sha256(ledger),
            }
        )

    sensitivity_contract = contract.stop_go["experiment_c"]["zero_api_sensitivity"][
        "absolute_flow_threshold"
    ]
    threshold = {
        "method": "selected-stage0-pooled-median-flow-utility-v1",
        "field": "flow_utility",
        "aggregation": "median",
        "value": median(values),
        "row_count": len(values),
        "selected_profile_id": selected_profile_id,
        "source_seeds": list(expected_seeds),
        "source_manifests": source_rows,
        "source_matrix_sha256": canonical_sha256(source_rows),
        "derivation_contract": dict(sensitivity_contract),
        "derived_after_profile_selection": True,
        "treatment_outcomes_inspected": False,
    }
    if threshold["row_count"] != 2 * 4 * 12:
        raise PilotOrchestrationError(
            "absolute flow-utility threshold expected exactly 96 Stage-0 rows"
        )
    return threshold


def _load_verified_stage0_selection(
    contract: PilotContract,
    *,
    raw_root: Path,
    paid: GitProvenance | None,
) -> dict[str, Any]:
    path = raw_root / "stage0-calibration" / "stage0_selection.json"
    value = _read_json(path)
    if paid is None and not isinstance(value.get("bindings"), Mapping):
        if value.get("contract_sha256") not in {None, contract.canonical_hash}:
            raise PilotOrchestrationError(
                "diagnostic Stage-0 selection contract mismatch"
            )
        return value
    _verify_bound_payload(
        value,
        contract=contract,
        schema_version="finevo-stage0-selection-v1",
        paid=paid,
        artifact_name="Stage-0 selection",
    )
    bindings = value["bindings"]
    q_ref = _load_verified_q_ref(contract, raw_root=raw_root, paid=paid)
    q_ref_path = raw_root / "q-ref-resolution" / "q_ref_resolution.json"
    if bindings.get("q_ref_content_sha256") != q_ref.get("integrity", {}).get(
        "content_sha256"
    ) or bindings.get("q_ref_file_sha256") != _file_sha256(q_ref_path):
        raise PilotOrchestrationError("Stage-0 q_ref binding mismatch")

    expected_specs = tuple(contract.expand(stage="stage0-calibration"))
    expected_ids = {spec.run_id for spec in expected_specs}
    by_profile: dict[str, list[Mapping[str, Any]]] = {
        profile_id: []
        for profile_id in expand_stage0_ofat(float(q_ref["q_ref"]))["profile_order"]
    }
    source_rows = bindings.get("source_manifests")
    if not isinstance(source_rows, list) or len(source_rows) != len(expected_specs):
        raise PilotOrchestrationError(
            "Stage-0 selection does not bind the exact 7x2 source matrix"
        )
    observed_ids: set[str] = set()
    calibration_sources: list[tuple[PilotRunSpec, Any, Mapping[str, Any]]] = []
    for row in source_rows:
        if not isinstance(row, Mapping):
            raise PilotOrchestrationError("Stage-0 source binding is malformed")
        run_id = str(row.get("run_id", ""))
        if run_id in observed_ids:
            raise PilotOrchestrationError("Stage-0 source binding has duplicates")
        observed_ids.add(run_id)
        expected_spec = next(
            (spec for spec in expected_specs if spec.run_id == run_id),
            None,
        )
        if expected_spec is None:
            raise PilotOrchestrationError("Stage-0 source binding has an extra cell")
        expected_manifest = (
            raw_root / "stage0-calibration" / "runs" / run_id / "manifest.json"
        )
        if row.get("manifest") != str(expected_manifest):
            raise PilotOrchestrationError("Stage-0 source manifest path mismatch")
        if (
            row.get("utility_profile_id") != expected_spec.utility_profile_id
            or row.get("environment_seed") != expected_spec.environment_seed
        ):
            raise PilotOrchestrationError("Stage-0 source cell binding mismatch")
        verification = verify_manifest(expected_manifest.parent)
        if row.get("manifest_sha256") != verification.manifest_sha256:
            raise PilotOrchestrationError("Stage-0 source manifest hash mismatch")
        manifest_value = _read_json(expected_manifest)
        if (
            manifest_value.get("git", {}).get("commit") != bindings.get("git_commit")
            or manifest_value.get("git", {}).get("dirty") is not False
        ):
            raise PilotOrchestrationError(
                "Stage-0 source manifest git binding mismatch"
            )
        result = load_verified_run_artifacts(expected_manifest.parent)
        provenance = _read_json(expected_manifest.parent / "provenance.json")
        details = provenance.get("details")
        if (
            result.config.get("run_id") != run_id
            or result.config.get("seed") != expected_spec.environment_seed
            or not isinstance(details, Mapping)
            or details.get("run_spec") != expected_spec.to_dict()
            or details.get("contract_sha256") != contract.canonical_hash
        ):
            raise PilotOrchestrationError("Stage-0 source run identity mismatch")
        by_profile[expected_spec.utility_profile_id].append(
            summarize_run(
                result.records,
                max_labor_hours=float(result.config["max_labor_hours"]),
                schedule=contract.shocks[expected_spec.shock_id]["schedule"],
            )
        )
        calibration_sources.append((expected_spec, result, row))
    if observed_ids != expected_ids:
        raise PilotOrchestrationError("Stage-0 selection source matrix is incomplete")
    recomputed = select_stage0_profile(
        expand_stage0_ofat(float(q_ref["q_ref"])),
        by_profile,
    )
    recomputed["absolute_flow_utility_threshold"] = (
        _derive_stage0_absolute_flow_threshold(
            contract,
            selected_profile_id=str(recomputed["selected_profile_id"]),
            sources=calibration_sources,
        )
    )
    for key, expected in recomputed.items():
        if value.get(key) != expected:
            raise PilotOrchestrationError(
                f"Stage-0 selection field {key!r} differs from sealed sources"
            )
    return value


def resolve_utility(
    contract: PilotContract,
    spec: PilotRunSpec,
    *,
    raw_root: str | Path,
    paid_provenance: GitProvenance | None = None,
) -> UtilityConfig:
    """Resolve a declared profile without inspecting treatment outcomes."""

    profiles = contract.utility["profiles"]
    profile = profiles[spec.utility_profile_id]
    if spec.utility_profile_id == "stage0-selected":
        selected = _load_verified_stage0_selection(
            contract,
            raw_root=Path(raw_root),
            paid=paid_provenance,
        )
        return _utility_from_mapping(selected["selected_utility"])
    if profile.get("consumption_scale") is not None:
        return _utility_from_mapping(profile)
    q_ref = _load_verified_q_ref(
        contract,
        raw_root=Path(raw_root),
        paid=paid_provenance,
    )
    multiplier = profile.get("consumption_scale_multiplier_of_q_ref")
    if multiplier is None:
        raise PilotOrchestrationError("utility profile lacks a q_ref multiplier")
    resolved = dict(profile)
    resolved["consumption_scale"] = float(q_ref["q_ref"]) * float(multiplier)
    return _utility_from_mapping(resolved)


def config_for_spec(
    contract: PilotContract,
    spec: PilotRunSpec,
    *,
    raw_root: str | Path,
    paid_provenance: GitProvenance | None,
    diagnostic_override: bool = False,
    verify_bound_inputs: bool = False,
    preflight_p95_reservations: Mapping[str, Mapping[str, Any]] | None = None,
    num_agents_override: int | None = None,
    episode_length_override: int | None = None,
) -> VerifiedRunConfig:
    """Map one registered actor arm to the runner's explicit v3 contract."""

    arm = contract.arms[spec.arm_id]
    parameters = arm["parameters"]
    context_mode = str(parameters.get("context_mode", "full"))
    episodic = bool(parameters.get("episodic_actor_exposure", True))
    semantic = bool(parameters.get("semantic_actor_exposure", True))
    semantic_policy = (
        "evidence-grounded"
        if bool(parameters.get("verifier_enabled", True))
        else "unverified-immediate"
    )
    error_mode = str(parameters.get("error_rule_mode", "none"))
    num_agents = spec.num_agents if num_agents_override is None else num_agents_override
    horizon = (
        spec.episode_length
        if episode_length_override is None
        else episode_length_override
    )
    if horizon < 6 and error_mode != "none":
        raise PilotOrchestrationError("error-rule runs require decision_t=5")

    scientific = (
        paid_provenance is not None and not diagnostic_override and verify_bound_inputs
    )
    request_profile = contract.provider_profiles[spec.model_id]
    action_output = contract.task_output_contracts.get("actor-action")
    semantic_output = contract.task_output_contracts.get("semantic-proposal")
    return VerifiedRunConfig(
        run_id=spec.run_id,
        seed=spec.environment_seed,
        num_agents=num_agents,
        episode_length=horizon,
        context_mode=context_mode,
        enable_episodic_retrieval=episodic,
        enable_semantic=semantic,
        retrieval_k=5 if episodic else 0,
        rule_budget=3 if semantic else 0,
        semantic_proposal_after=3,
        semantic_proposal_interval=3,
        max_rule_proposals_per_agent=4,
        semantic_policy=semantic_policy,
        error_rule_mode=error_mode,
        error_rule_injection_t=5,
        semantic_parse_failure_policy="record-and-skip",
        temperature=0.0,
        top_p=1.0,
        action_max_tokens=(
            action_output.max_completion_tokens if action_output is not None else 220
        ),
        rule_max_tokens=(
            semantic_output.max_completion_tokens
            if semantic_output is not None
            else 450
        ),
        action_max_visible_json_bytes=(
            action_output.max_visible_json_bytes
            if action_output is not None
            else 1_000_000
        ),
        rule_max_visible_json_bytes=(
            semantic_output.max_visible_json_bytes
            if semantic_output is not None
            else 1_000_000
        ),
        accepted_action_parse_modes=(
            ("exact_json",)
            if action_output is not None
            and action_output.science_parse_mode == "exact_json_only"
            else (
                "exact_json",
                "fenced_recovery",
                "substring_recovery",
            )
        ),
        accepted_semantic_parse_modes=(
            ("exact_json",)
            if semantic_output is not None
            and semantic_output.science_parse_mode == "exact_json_only"
            else (
                "exact_json",
                "fenced_recovery",
                "substring_recovery",
            )
        ),
        max_retries=1,
        send_decoding_seed=spec.decoding_seed is not None,
        utility=resolve_utility(
            contract,
            spec,
            raw_root=raw_root,
            paid_provenance=(paid_provenance if verify_bound_inputs else None),
        ),
        shock_schedule=_shock_events(
            contract,
            spec.shock_id,
            episode_length=horizon,
        ),
        scientific_scope=(
            "preregistered_mechanism_micro_pilot"
            if scientific
            else "bounded_method_smoke"
        ),
        pilot_contract_hash=contract.canonical_hash if scientific else None,
        pilot_tag=paid_provenance.git_tag if scientific else None,
        allow_scientific_scope=scientific,
        preflight_p95_reservations=preflight_p95_reservations or {},
    )


def _estimate_usage_for_profile(
    profile: ProviderRequestProfile,
    prompt: str,
    max_tokens: int,
) -> UsageRecord:
    rates = profile.price_snapshot.costs_per_1k()
    prompt_tokens = max(1, math.ceil(len(prompt) / 4))
    completion_tokens = int(max_tokens)
    return UsageRecord(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=(
            prompt_tokens * rates["prompt"] + completion_tokens * rates["completion"]
        )
        / 1000.0,
    )


def _provider_for_profile(
    profile: ProviderRequestProfile,
    *,
    per_kind_p95: Mapping[str, UsageRecord] | None = None,
) -> MultiModelLLM:
    if profile.transport == "diagnostic":
        return MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=4)
    provider_type = {
        "openai": "openai",
        "openrouter": "thirdparty",
        "ollama": "ollama",
    }.get(profile.transport)
    if provider_type is None:
        raise PilotOrchestrationError(
            f"unsupported pilot provider transport: {profile.transport}"
        )
    provider = create_llm_provider(
        provider_type,
        model=profile.requested_model,
        max_retries=1,
        request_profile=profile,
    )
    # The runner now owns exact immutable p95 reservations through
    # VerifiedRunConfig.  Keeping replacement estimates in an LLM wrapper
    # would create two competing budget authorities.
    return MultiModelLLM(provider, num_workers=4)


def _runner_p95_reservations(
    contract: PilotContract,
    model_id: str,
    *,
    raw_root: Path,
    paid: GitProvenance,
) -> dict[str, dict[str, Any]]:
    payload, _ = _load_verified_projection(
        contract,
        model_id,
        raw_root=raw_root,
        paid=paid,
    )
    profile = contract.provider_profiles[model_id]
    provider_name = {
        "openai": "openai",
        "openrouter": "thirdparty",
        "ollama": "ollama",
    }.get(profile.transport)
    if provider_name is None:
        raise PilotOrchestrationError(
            f"no scientific p95 model identity for {profile.transport!r}"
        )
    runtime_model = f"{provider_name}/{profile.requested_model}"
    by_kind: dict[str, Any] = {}
    for key, row in payload["projection"].items():
        _, separator, call_kind = str(key).rpartition("::")
        if separator and call_kind in {"action", "semantic"}:
            by_kind[call_kind] = _json_copy(row)
    if "action" not in by_kind:
        raise PilotOrchestrationError("sealed preflight lacks action p95")
    return {runtime_model: by_kind}


def _preflight_usage_caps(
    contract: PilotContract,
    model_id: str,
    *,
    raw_root: Path,
    paid: GitProvenance | None = None,
) -> dict[str, UsageRecord]:
    payload, _ = _load_verified_projection(
        contract,
        model_id,
        raw_root=raw_root,
        paid=paid,
    )
    caps: dict[str, UsageRecord] = {}
    for key, row in payload["projection"].items():
        kind = str(key).rpartition("::")[2]
        reserved = row["reserved_p95"]
        caps[kind] = UsageRecord(
            prompt_tokens=math.ceil(float(reserved["prompt_tokens"])),
            completion_tokens=math.ceil(float(reserved["completion_tokens"])),
            cost_usd=float(reserved["cost_usd"]),
        )
    return caps


def _max_call_projection(
    contract: PilotContract,
    spec: PilotRunSpec,
) -> tuple[int, int, int]:
    """Return conservative calls/prompt/output token ceilings before p95 exists."""

    action_cap = (
        contract.task_output_contracts["actor-action"].max_completion_tokens
        if contract.task_output_contracts
        else 220
    )
    semantic_cap = (
        contract.task_output_contracts["semantic-proposal"].max_completion_tokens
        if contract.task_output_contracts
        else 450
    )
    if spec.execution_mode == "capability_probe":
        if not contract.schema_version.endswith("-v2"):
            return (
                46,
                160_000,
                36 * action_cap + 10 * semantic_cap,
            )
        return 30, 60_000, 24 * action_cap + 6 * semantic_cap
    if spec.execution_mode == "closed_loop_preflight":
        return 16, 100_000, 12 * action_cap + 4 * semantic_cap
    if spec.execution_mode == "q_ref_resolution":
        return spec.num_agents * spec.episode_length, 500_000, 100_000
    if spec.execution_mode == "offline_candidate_admission":
        return 0, 0, 0
    if spec.execution_mode == "checkpoint_continuation":
        # Contract rows share one seed checkpoint; this is intentionally not
        # used for dispatch (the grouped D executor reserves the seed once).
        return spec.num_agents * spec.episode_length, 500_000, 100_000
    action_calls = spec.num_agents * spec.episode_length
    proposal_calls = spec.num_agents * 4
    return (
        action_calls + proposal_calls,
        500_000,
        action_calls * action_cap + proposal_calls * semantic_cap,
    )


def _counts_toward_hosted_completion_cap(
    profile: ProviderRequestProfile,
) -> bool:
    """Return whether calls use a hosted API covered by the 7,500 cap."""

    return profile.transport in {"openai", "openrouter"}


def conservative_projection(
    contract: PilotContract,
    spec: PilotRunSpec,
    *,
    diagnostic: bool = False,
) -> RunProjection:
    """Build a no-understatement preflight reservation for one run."""

    calls, prompt_tokens, completion_tokens = _max_call_projection(
        contract,
        spec,
    )
    profile = contract.provider_profiles[spec.model_id]
    hosted_completion_cap_counted = bool(
        not diagnostic and _counts_toward_hosted_completion_cap(profile)
    )
    if diagnostic or profile.transport == "diagnostic":
        cost = 0.0
    else:
        rates = profile.price_snapshot.costs_per_1k()
        cost = (
            prompt_tokens * rates["prompt"] + completion_tokens * rates["completion"]
        ) / 1000.0
    return RunProjection(
        run_id=spec.run_id,
        stage_bucket=spec.budget_bucket,
        cost_usd=cost,
        completions=calls if hosted_completion_cap_counted else 0,
        storage_bytes=20_000_000,
        basis={
            "method": "preflight-conservative-token-ceiling",
            "diagnostic": diagnostic,
            "run_call_limit": calls,
            "hosted_completion_cap_counted": hosted_completion_cap_counted,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    )


def _run_budget_from_projection(projection: RunProjection) -> RunBudget:
    basis = projection.basis
    run_call_limit = int(basis.get("run_call_limit", projection.completions))
    return RunBudget(
        BudgetLimits(
            max_calls=run_call_limit,
            max_prompt_tokens=int(basis.get("prompt_tokens", 2_000_000)),
            max_completion_tokens=int(basis.get("completion_tokens", 1_000_000)),
            max_total_tokens=int(
                basis.get("prompt_tokens", 2_000_000)
                + basis.get("completion_tokens", 1_000_000)
            ),
            # RunBudget treats reaching a cap as a terminal stop.  Diagnostic
            # providers report exactly zero, so use a non-spendable numerical
            # epsilon rather than a zero cap that is already "reached".
            max_cost_usd=max(projection.cost_usd, 1e-9),
            max_elapsed_seconds=3600.0,
        ),
        budget_id=f"{projection.run_id}-budget",
    )


def _actual_budget_values(
    run_dir: Path,
    budget: RunBudget,
    *,
    additional_paths: Sequence[Path] = (),
    completion_cap_counted: bool = True,
) -> dict[str, Any]:
    snapshot = budget.snapshot()
    usage = snapshot.accounted_usage
    return {
        "cost_usd": float(usage.cost_usd),
        "completions": (int(snapshot.completed_calls) if completion_cap_counted else 0),
        "storage_bytes": _directory_size(run_dir)
        + sum(
            path.stat().st_size
            for path in additional_paths
            if path.is_file() and not path.is_relative_to(run_dir)
        ),
    }


def _provenance_payload(
    contract: PilotContract,
    spec: PilotRunSpec,
    *,
    paid: GitProvenance | None,
    diagnostic: bool,
) -> dict[str, Any]:
    return {
        "purpose": (
            "scripted A-D integration diagnostic"
            if diagnostic
            else "preregistered FinEvo mechanism micro-pilot"
        ),
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "run_spec": spec.to_dict(),
        "git": paid.to_dict() if paid is not None else None,
        "diagnostic_only": diagnostic,
        "scientific_evidence": False if diagnostic else True,
        "python": sys.version.split()[0],
    }


def _exception_failure(
    error: BaseException,
    **context: Any,
) -> dict[str, Any]:
    """Retain the runner's structured failure without exposing raw output."""

    failure = {
        "error_type": type(error).__name__,
        "message": str(error),
        **context,
    }
    structured = getattr(error, "failure", None)
    if structured is not None:
        to_dict = getattr(structured, "to_dict", None)
        if callable(to_dict):
            details = to_dict()
        elif isinstance(structured, Mapping):
            details = dict(structured)
        else:
            details = None
        if details is not None:
            failure["details"] = _json_copy(details)
    return failure


def _verified_provider_call_journal_binding(
    path: Path,
    *,
    expected_run_id: str,
    expected_contract_hash: str | None,
) -> dict[str, Any]:
    """Return a content binding only after full terminal journal validation."""

    payload = verify_provider_call_journal(
        path,
        expected_run_id=expected_run_id,
        expected_contract_hash=expected_contract_hash,
        require_terminal_dispositions=True,
    )
    return {
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "journal_sha256": payload["journal_sha256"],
        "run_id": payload["run_id"],
        "contract_hash": payload["contract_hash"],
        "event_count": len(payload["events"]),
        "terminal_dispositions_verified": True,
    }


def _provider_call_journal_path(
    run_dir: Path,
    *,
    run_id: str,
    kind: str,
) -> Path:
    """Keep journals beside, never inside, immutable artifact directories."""

    if kind not in {"actor", "preflight"}:
        raise ValueError("provider journal kind is invalid")
    return run_dir.parent.parent / "provider_call_journals" / f"{run_id}--{kind}.json"


def _write_execution_failure_receipt(
    receipt_dir: Path,
    *,
    scope: str,
    error: BaseException,
    contract: PilotContract,
    projection: RunProjection,
    budget: RunBudget,
    specs: Sequence[PilotRunSpec],
    paid: GitProvenance | None,
    diagnostic: bool,
) -> Path:
    """Seal one post-reservation failure in an isolated artifact directory."""

    if not specs:
        raise ValueError("failure receipt requires at least one run spec")
    model_ids = sorted({spec.model_id for spec in specs})
    profiles = {
        model_id: contract.provider_profiles[model_id].to_dict()
        for model_id in model_ids
    }
    if paid is not None:
        git_commit = paid.head_commit
        git_dirty = False
    else:
        repository = Path(__file__).resolve().parents[1]
        git_commit = _git(repository, "rev-parse", "HEAD", check=False) or "unknown"
        git_dirty = bool(
            _git(
                repository,
                "status",
                "--porcelain",
                "--untracked-files=all",
                check=False,
            )
        )
    journal_bindings = []
    for spec in specs:
        for kind in ("actor", "preflight"):
            journal = _provider_call_journal_path(
                receipt_dir.parent,
                run_id=spec.run_id,
                kind=kind,
            )
            if not journal.exists():
                continue
            journal_bindings.append(
                _verified_provider_call_journal_binding(
                    journal,
                    expected_run_id=(
                        f"{spec.run_id}--actor-preflight"
                        if kind == "preflight"
                        else spec.run_id
                    ),
                    expected_contract_hash=(
                        None
                        if (
                            diagnostic
                            or paid is None
                            or (
                                kind == "preflight"
                                and not contract.schema_version.endswith("-v2")
                            )
                        )
                        else contract.canonical_hash
                    ),
                )
            )
    return write_failure_receipt(
        receipt_dir,
        scope=scope,
        error=error,
        budget_snapshot=budget.snapshot().to_dict(),
        config={
            "schema_version": "finevo-pilot-failure-config-v1",
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "projection": projection.to_dict(),
            "run_specs": [spec.to_dict() for spec in specs],
            "provider_request_profiles": profiles,
            "provider_call_journals": journal_bindings,
        },
        provenance={
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "paid_provenance": paid.to_dict() if paid is not None else None,
            "diagnostic_only": diagnostic,
            "scientific_evidence": False,
            "evidence_use": "failure denominator and audit only",
        },
        git_commit=git_commit,
        git_dirty=git_dirty,
    )


def _execute_actor_run(
    contract: PilotContract,
    spec: PilotRunSpec,
    *,
    raw_root: Path,
    paid: GitProvenance | None,
    projection: RunProjection,
    budget: RunBudget,
    diagnostic: bool = False,
    num_agents_override: int | None = None,
    episode_length_override: int | None = None,
) -> tuple[Path, RunBudget, Mapping[str, Any]]:
    run_dir = raw_root / spec.stage_id / "runs" / spec.run_id
    llm = (
        MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=4)
        if diagnostic
        else _provider_for_profile(contract.provider_profiles[spec.model_id])
    )
    config = config_for_spec(
        contract,
        spec,
        raw_root=raw_root,
        paid_provenance=paid,
        diagnostic_override=diagnostic,
        verify_bound_inputs=paid is not None and not diagnostic,
        preflight_p95_reservations=(
            {}
            if diagnostic or paid is None
            else _runner_p95_reservations(
                contract,
                spec.model_id,
                raw_root=raw_root,
                paid=paid,
            )
        ),
        num_agents_override=num_agents_override,
        episode_length_override=episode_length_override,
    )
    journal_path = _provider_call_journal_path(
        run_dir,
        run_id=spec.run_id,
        kind="actor",
    )
    result = run_verified_experiment(
        config,
        llm=llm,
        budget=budget,
        env_config_source=DEFAULT_ENV_CONFIG,
        call_journal_path=journal_path,
    )
    journal_binding = _verified_provider_call_journal_binding(
        journal_path,
        expected_run_id=config.run_id,
        expected_contract_hash=config.pilot_contract_hash,
    )
    git_commit = (
        paid.head_commit
        if paid is not None
        else _git(Path(__file__).resolve().parents[1], "rev-parse", "HEAD")
    )
    manifest = write_verified_run_artifacts(
        run_dir,
        result,
        provenance={
            **_provenance_payload(
                contract,
                spec,
                paid=paid,
                diagnostic=diagnostic,
            ),
            "provider_call_journal": journal_binding,
        },
        git_commit=git_commit,
        git_dirty=False if paid is not None else True,
    )
    summary = summarize_run(
        result.records,
        max_labor_hours=config.max_labor_hours,
    )
    if paid is None or diagnostic:
        _atomic_json(
            raw_root / spec.stage_id / "diagnostic_summaries" / f"{spec.run_id}.json",
            {
                "contract_sha256": contract.canonical_hash,
                "run_spec": spec.to_dict(),
                "metrics": {"analysis": summary},
                "provider_call_journal": journal_binding,
                "diagnostic_only": True,
                "scientific_evidence": False,
            },
        )
    return manifest, budget, summary


def _preflight_config(
    contract: PilotContract,
    spec: PilotRunSpec,
    *,
    paid: GitProvenance,
) -> VerifiedRunConfig:
    profile = contract.utility["profiles"]["provider-preflight-default"]
    scientific_authorization = contract.schema_version.endswith("-v2")
    action_output = contract.task_output_contracts.get("actor-action")
    semantic_output = contract.task_output_contracts.get("semantic-proposal")
    return VerifiedRunConfig(
        run_id=f"{spec.run_id}--actor-preflight",
        seed=spec.environment_seed,
        num_agents=2,
        episode_length=6,
        context_mode="full",
        enable_episodic_retrieval=True,
        enable_semantic=True,
        retrieval_k=5,
        rule_budget=3,
        semantic_proposal_after=3,
        semantic_proposal_interval=3,
        max_rule_proposals_per_agent=4,
        semantic_parse_failure_policy="record-and-skip",
        temperature=0.0,
        top_p=1.0,
        action_max_tokens=(
            action_output.max_completion_tokens if action_output is not None else 220
        ),
        rule_max_tokens=(
            semantic_output.max_completion_tokens
            if semantic_output is not None
            else 450
        ),
        action_max_visible_json_bytes=(
            action_output.max_visible_json_bytes
            if action_output is not None
            else 1_000_000
        ),
        rule_max_visible_json_bytes=(
            semantic_output.max_visible_json_bytes
            if semantic_output is not None
            else 1_000_000
        ),
        accepted_action_parse_modes=(
            ("exact_json",)
            if action_output is not None
            and action_output.science_parse_mode == "exact_json_only"
            else (
                "exact_json",
                "fenced_recovery",
                "substring_recovery",
            )
        ),
        accepted_semantic_parse_modes=(
            ("exact_json",)
            if semantic_output is not None
            and semantic_output.science_parse_mode == "exact_json_only"
            else (
                "exact_json",
                "fenced_recovery",
                "substring_recovery",
            )
        ),
        max_retries=1,
        send_decoding_seed=(
            spec.decoding_seed is not None
            if contract.schema_version.endswith("-v2")
            else (
                contract.provider_profiles[spec.model_id].seed_capability
                != "unsupported"
            )
        ),
        utility=_utility_from_mapping(profile),
        shock_schedule=_shock_events(
            contract,
            spec.shock_id,
            episode_length=6,
        ),
        scientific_scope=(
            "preregistered_mechanism_micro_pilot"
            if scientific_authorization
            else "bounded_method_smoke"
        ),
        pilot_contract_hash=(
            contract.canonical_hash if scientific_authorization else None
        ),
        pilot_tag=paid.git_tag if scientific_authorization else None,
        allow_scientific_scope=scientific_authorization,
    )


class _CheckpointPreflightResult:
    """Minimal stream adapter over one hash-validated V2 preflight checkpoint."""

    def __init__(self, checkpoint: PilotCheckpoint) -> None:
        payload = checkpoint.payload
        self.summary = {
            "scientific_evidence": False,
            "result_scope": "preregistered_capability_preflight",
        }
        self.validation_status = {"status": "pass"}
        self._streams = {
            "actions": tuple(
                {"decision": dict(decision)}
                for step in payload["prefix_steps"]
                for decision in step["decisions"].values()
            ),
            "semantic_proposals": tuple(payload["proposal_outcomes"]),
            "api_usage": tuple(payload["provider_calls"]),
        }

    def stream(self, name: str) -> tuple[Mapping[str, Any], ...]:
        try:
            return self._streams[name]
        except KeyError as exc:
            raise KeyError(name) from exc


def _preflight_checks(
    result: Any,
    profile: ProviderRequestProfile,
    *,
    checkpoint_preflight: bool = False,
) -> dict[str, bool]:
    actions = result.stream("actions")
    proposals = result.stream("semantic_proposals")
    api_rows = result.stream("api_usage")
    provider_rows = [row for row in api_rows if row.get("error_type") is not None]
    expected_artifact_identity = dict(profile.artifact_identity)
    declared_dispatch = {
        field: (
            "explicit_supported"
            if disposition.dispatch_mode == "explicit_supported"
            else "omitted_unsupported"
        )
        for field, disposition in profile.decoding_fields
    }
    dispatches = {
        field: status == "explicit_supported"
        for field, status in declared_dispatch.items()
    }
    if profile.transport == "openrouter":
        expected_response_providers = set(profile.provider_pin)
        expected_response_route = expected_artifact_identity.get("served_snapshot")
        expected_route_attestation = "OR_RA_PASS"
        expected_sdk_name = "openai-python"
        expected_sdk_version = PINNED_PROVIDER_SDK_VERSIONS["openai"]
        expected_temperature_dispatch = (
            "explicit"
            if not dispatches or dispatches["temperature"]
            else "omitted_unsupported"
        )
        expected_request_parameters = {
            "model",
            "messages",
            "max_tokens",
            *profile.openrouter_request_options().keys(),
        }
        if not dispatches or dispatches["temperature"]:
            expected_request_parameters.add("temperature")
        if not dispatches or dispatches["top_p"]:
            expected_request_parameters.add("top_p")
        if (not dispatches and profile.seed_capability != "unsupported") or (
            dispatches and dispatches["seed"]
        ):
            expected_request_parameters.add("seed")
        if dispatches and not dispatches["response_format"]:
            expected_request_parameters.discard("response_format")
    elif profile.transport == "openai":
        expected_response_providers = {"OpenAI-direct"}
        expected_response_route = "direct"
        expected_route_attestation = None
        expected_sdk_name = "openai-python"
        expected_sdk_version = PINNED_PROVIDER_SDK_VERSIONS["openai"]
        expected_temperature_dispatch = (
            (
                "omitted_unsupported"
                if profile.requested_model.startswith(("gpt-5", "o1", "o3"))
                else "explicit"
            )
            if not dispatches
            else ("explicit" if dispatches["temperature"] else "omitted_unsupported")
        )
        reasoning_model = profile.requested_model.startswith(("gpt-5", "o1", "o3"))
        expected_request_parameters = {
            "model",
            "messages",
            *profile.openai_request_options().keys(),
            "max_completion_tokens" if reasoning_model else "max_tokens",
        }
        if not dispatches or dispatches["top_p"]:
            expected_request_parameters.add("top_p")
        if (not dispatches and not reasoning_model) or (
            dispatches and dispatches["temperature"]
        ):
            expected_request_parameters.add("temperature")
        if (not dispatches and profile.seed_capability != "unsupported") or (
            dispatches and dispatches["seed"]
        ):
            expected_request_parameters.add("seed")
        if dispatches and not dispatches["response_format"]:
            expected_request_parameters.discard("response_format")
        if dispatches and not dispatches["reasoning"]:
            expected_request_parameters.discard("reasoning_effort")
    elif profile.transport == "ollama":
        expected_response_providers = {"local-ollama"}
        expected_response_route = "local"
        expected_route_attestation = None
        expected_sdk_name = "requests"
        expected_sdk_version = PINNED_PROVIDER_SDK_VERSIONS["requests"]
        expected_temperature_dispatch = "explicit"
        expected_request_parameters = {
            "model",
            "messages",
            "stream",
            "options",
        }
        if profile.json_mode == "json_object":
            expected_request_parameters.add("format")
        if dispatches and not dispatches["response_format"]:
            expected_request_parameters.discard("format")
    else:
        expected_response_providers = set(profile.provider_pin)
        expected_response_route = profile.routing_mode
        expected_route_attestation = None
        expected_sdk_name = None
        expected_sdk_version = None
        expected_temperature_dispatch = None
        expected_request_parameters = set()
    checks = {
        "action_parse_success_12_of_12": len(actions) == 12,
        "proposal_outcomes_accounted": all(
            row.get("candidate_parse_status") in {"success", "failure"}
            for row in proposals
        ),
        "no_clipping": all(not bool(row["decision"]["clipped"]) for row in actions),
        "no_provider_error": not provider_rows,
        "served_model_exact": all(
            row.get("response_model") == profile.served_model for row in api_rows
        ),
        "attempt_count_exact": all(row.get("attempts") == 1 for row in api_rows),
        "request_profile_exact": bool(api_rows)
        and all(
            row.get("request_profile_id") == profile.profile_id for row in api_rows
        ),
        "request_provider_pin_exact": bool(api_rows)
        and all(
            row.get("request_provider_pin") == list(profile.provider_pin)
            for row in api_rows
        ),
        "request_artifact_identity_exact": bool(api_rows)
        and all(
            row.get("request_artifact_identity") == expected_artifact_identity
            for row in api_rows
        ),
        "request_price_snapshot_exact": bool(api_rows)
        and all(
            row.get("request_price_snapshot_source") == profile.price_snapshot.source
            and row.get("request_price_snapshot_captured_at")
            == profile.price_snapshot.captured_at
            for row in api_rows
        ),
        "response_route_metadata_complete": bool(api_rows)
        and all(
            isinstance(row.get("response_provider"), str)
            and bool(row["response_provider"])
            and isinstance(row.get("response_route"), str)
            and bool(row["response_route"])
            for row in api_rows
        ),
        "response_provider_exact": bool(api_rows)
        and all(
            row.get("response_provider") in expected_response_providers
            for row in api_rows
        ),
        "response_snapshot_exact": bool(api_rows)
        and all(
            row.get("response_route") == expected_response_route for row in api_rows
        ),
        "route_attestation_exact": bool(api_rows)
        and all(
            row.get("route_attestation_code") == expected_route_attestation
            for row in api_rows
        ),
        "finish_metadata_complete": bool(api_rows)
        and all(
            row.get("finish_reason") == "stop" and row.get("response_completed") is True
            for row in api_rows
        ),
        "provider_sdk_complete": bool(api_rows)
        and all(
            (
                expected_sdk_name is None
                or row.get("provider_sdk_name") == expected_sdk_name
            )
            and (
                expected_sdk_version is None
                or row.get("provider_sdk_version") == expected_sdk_version
            )
            for row in api_rows
        ),
        "successful_rows_have_no_error_details": all(
            "provider_error_details" in row
            and row.get("provider_error_details") is None
            for row in api_rows
            if row.get("error_type") is None
        ),
        "successful_rows_output_accepted": bool(api_rows)
        and all(
            row.get("output_disposition") == "accepted"
            for row in api_rows
            if row.get("error_type") is None
        ),
        "temperature_dispatch_exact": bool(api_rows)
        and all(
            row.get("temperature_dispatch") == expected_temperature_dispatch
            for row in api_rows
        ),
        "request_parameters_exact": bool(api_rows)
        and all(
            row.get("request_parameters") == sorted(expected_request_parameters)
            for row in api_rows
        ),
        "parameter_dispatch_exact": (
            True
            if not declared_dispatch
            else bool(api_rows)
            and all(
                row.get("parameter_dispatch") == declared_dispatch for row in api_rows
            )
        ),
        "usage_complete": all(
            isinstance(row.get("usage"), Mapping)
            and row["usage"].get("prompt_tokens") is not None
            and row["usage"].get("completion_tokens") is not None
            and row["usage"].get("cost_usd") is not None
            for row in api_rows
        ),
        "runner_validation_pass": result.validation_status.get("status") == "pass",
        "bounded_non_scientific_measurement": (
            result.summary.get("scientific_evidence") is False
            and result.summary.get("result_scope")
            == (
                "preregistered_capability_preflight"
                if checkpoint_preflight
                else "bounded_method_smoke"
            )
        ),
    }
    if checkpoint_preflight:
        checks["proposal_outcomes_accounted_4_of_4"] = len(
            proposals
        ) == 4 and checks.pop("proposal_outcomes_accounted")
        checks["provider_calls_accounted_16_of_16"] = len(api_rows) == 16
    return checks


def _usage_projection_rows(
    capability: Mapping[str, Any],
    preflight_result: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in capability["rows"]:
        rows.append(
            {
                "response_model": row["served_model"],
                "call_kind": row.get(
                    "output_contract_id",
                    (
                        "capability-proposal"
                        if row["category"] == "rule-proposal"
                        else "capability-choice"
                    ),
                ),
                "usage": row["usage"],
            }
        )
    rows.extend(
        {
            "response_model": row["response_model"],
            "call_kind": row["call_kind"],
            "usage": row["usage"],
        }
        for row in preflight_result.stream("api_usage")
    )
    return rows


def _execute_capability_preflight(
    contract: PilotContract,
    spec: PilotRunSpec,
    *,
    raw_root: Path,
    paid: GitProvenance,
    projection: RunProjection,
    budget: RunBudget,
) -> tuple[str, Path, RunBudget, dict[str, Any]]:
    """Run a capability gate and/or its separately registered 2x6 preflight."""

    run_dir = raw_root / spec.stage_id / "runs" / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    profile = contract.provider_profiles[spec.model_id]
    llm = _provider_for_profile(profile)
    estimate = lambda prompt, max_tokens: _estimate_usage_for_profile(  # noqa: E731
        profile, prompt, max_tokens
    )
    split_capability_only = spec.execution_mode == "capability_probe" and (
        contract.schema_version.endswith("-v2")
    )
    split_preflight_only = spec.execution_mode == "closed_loop_preflight"
    if split_preflight_only:
        capability_stage = _capability_source_stage(
            contract,
            spec.stage_id,
            spec.model_id,
        )
        capability_specs = contract.expand(
            stage=capability_stage,
            model=spec.model_id,
        )
        if len(capability_specs) != 1:
            raise PilotOrchestrationError(
                "closed-loop preflight requires one exact capability cell"
            )
        capability_path = (
            raw_root
            / capability_stage
            / "runs"
            / capability_specs[0].run_id
            / "capability.json"
        )
        capability_payload = _read_json(capability_path)
        if (
            capability_payload.get("contract_sha256") != contract.canonical_hash
            or capability_payload.get("run_spec") != capability_specs[0].to_dict()
        ):
            raise PilotOrchestrationError(
                "closed-loop preflight capability binding mismatch"
            )
        capability = capability_payload
    else:
        capability = run_capability_gate(
            llm=llm,
            budget=budget,
            seed=spec.decoding_seed,
            estimate_usage=estimate,
            task_output_contracts=(
                {
                    key: item.to_dict()
                    for key, item in contract.task_output_contracts.items()
                }
                if contract.task_output_contracts
                else None
            ),
        )
        capability_payload = {
            **capability,
            "contract_sha256": contract.canonical_hash,
            "run_spec": spec.to_dict(),
            "scientific_evidence": False,
            "evidence_use": "capability gate denominator only",
            "preflight_go": False,
        }
        capability_path = run_dir / "capability.json"
        _atomic_json(capability_path, capability_payload)
    if not capability["pass"]:
        interface_pass = capability["interface_gate"]["pass"] is True
        capability_status = capability["capability_assessment"]["status"]
        reason = (
            "fixed capability threshold not met"
            if interface_pass
            else "provider/interface failure; capability not evaluable"
        )
        receipt = {
            "capability_pass": False,
            "capability_status": capability_status,
            "interface_pass": interface_pass,
            "preflight_run": None,
            "go": False,
            "reason": reason,
        }
        _atomic_json(run_dir / "gate_receipt.json", receipt)
        terminal = write_terminal_summary(
            raw_root / spec.stage_id / "summaries" / f"{spec.run_id}.json",
            contract=contract,
            run_spec=spec,
            resolved_git_commit=paid.head_commit,
            git_tag=paid.git_tag,
            payload={
                "metrics": {},
                "gate_evidence": receipt,
                "capability": {
                    **capability_payload,
                    "preflight_go": False,
                },
            },
            scientific_evidence=False,
            diagnostic_only=False,
            evidence_scope="preregistered_capability_gate",
        )
        return "capability-no-go", terminal, budget, receipt

    if split_capability_only:
        receipt = {
            "capability_pass": True,
            "capability_status": "pass",
            "interface_pass": True,
            "preflight_run": None,
            "go": True,
            "reason": None,
        }
        _atomic_json(run_dir / "gate_receipt.json", receipt)
        terminal = write_terminal_summary(
            raw_root / spec.stage_id / "summaries" / f"{spec.run_id}.json",
            contract=contract,
            run_spec=spec,
            resolved_git_commit=paid.head_commit,
            git_tag=paid.git_tag,
            payload={
                "metrics": {},
                "gate_evidence": receipt,
                "capability": capability_payload,
            },
            scientific_evidence=False,
            diagnostic_only=False,
            evidence_scope="preregistered_task_capability_gate",
        )
        return "complete", terminal, budget, receipt

    config = _preflight_config(contract, spec, paid=paid)
    preflight_journal_path = _provider_call_journal_path(
        run_dir,
        run_id=spec.run_id,
        kind="preflight",
    )
    checkpoint_path: Path | None = None
    checkpoint_receipt_path: Path | None = None
    checkpoint_receipt: Mapping[str, Any] | None = None
    if contract.schema_version.endswith("-v2"):
        checkpoint = build_closed_loop_preflight_checkpoint(
            config,
            llm=llm,
            budget=budget,
            env_config_source=DEFAULT_ENV_CONFIG,
            call_journal_path=preflight_journal_path,
        )
        checkpoint_path = run_dir / "preflight_checkpoint.json"
        _atomic_bound_json(checkpoint_path, checkpoint.to_dict())
        exactness = verify_closed_loop_preflight_checkpoint(checkpoint)
        checkpoint_receipt = _seal_bound_payload(
            {
                "schema_version": (PILOT_PREFLIGHT_CHECKPOINT_RECEIPT_SCHEMA_VERSION),
                "bindings": {
                    "contract_sha256": contract.canonical_hash,
                    "git_tag": paid.git_tag,
                    "git_commit": paid.head_commit,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_file_sha256": _file_sha256(checkpoint_path),
                    "checkpoint_hash": checkpoint.checkpoint_hash,
                },
                "exactness": exactness,
            }
        )
        checkpoint_receipt_path = run_dir / "preflight_checkpoint_exactness.json"
        _atomic_bound_json(checkpoint_receipt_path, checkpoint_receipt)
        result = _CheckpointPreflightResult(checkpoint)
    else:
        result = run_verified_experiment(
            config,
            llm=llm,
            budget=budget,
            env_config_source=DEFAULT_ENV_CONFIG,
            call_journal_path=preflight_journal_path,
        )
    preflight_journal_binding = _verified_provider_call_journal_binding(
        preflight_journal_path,
        expected_run_id=config.run_id,
        expected_contract_hash=config.pilot_contract_hash,
    )
    checks = _preflight_checks(
        result,
        profile,
        checkpoint_preflight=contract.schema_version.endswith("-v2"),
    )
    catalog_path = (
        raw_root / spec.stage_id / "provider_catalog" / f"{spec.model_id}.json"
    )
    catalog_receipt = _read_json(catalog_path)
    observed_routes = sorted(
        {
            (
                str(row["response_provider"]),
                str(row["response_route"]),
            )
            for row in result.stream("api_usage")
            if row.get("response_provider") is not None
            and row.get("response_route") is not None
        }
    )
    route_binding = {
        "requested_provider_pin": list(profile.provider_pin),
        "allow_fallbacks": profile.allow_fallbacks,
        "require_parameters": profile.require_parameters,
        "catalog_receipt": str(catalog_path),
        "catalog_receipt_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "catalog_status": catalog_receipt.get("status"),
        "served_model_exact": checks["served_model_exact"],
        "response_route_attestation_available": checks[
            "response_route_metadata_complete"
        ],
        "observed_provider_routes": [
            {"provider": provider, "route": route}
            for provider, route in observed_routes
        ],
        "claim_boundary": (
            "The request pin, live catalog endpoint, and provider-returned "
            "route metadata are all retained; any absent or mismatched route "
            "makes this preflight a no-go."
        ),
    }
    manifest: Path | None = None
    if not contract.schema_version.endswith("-v2"):
        manifest = write_verified_run_artifacts(
            run_dir / "preflight",
            result,
            provenance={
                **_provenance_payload(
                    contract,
                    spec,
                    paid=paid,
                    diagnostic=False,
                ),
                "purpose": "bounded capability/interface p95 measurement",
                "scientific_evidence": False,
                "evidence_scope": "preregistered_capability_gate",
                "provider_call_journal": preflight_journal_binding,
            },
            git_commit=paid.head_commit,
            git_dirty=False,
        )
    source_bindings = {
        "contract_sha256": contract.canonical_hash,
        "git_tag": paid.git_tag,
        "git_commit": paid.head_commit,
        "source_capability": str(capability_path),
        "source_capability_sha256": _file_sha256(capability_path),
        "source_provider_call_journal": str(preflight_journal_path),
        "source_provider_call_journal_file_sha256": (
            preflight_journal_binding["file_sha256"]
        ),
        "source_provider_call_journal_sha256": (
            preflight_journal_binding["journal_sha256"]
        ),
    }
    if manifest is not None:
        source_bindings.update(
            {
                "source_manifest": str(manifest),
                "source_manifest_sha256": _file_sha256(manifest),
            }
        )
    else:
        assert checkpoint_path is not None
        assert checkpoint_receipt_path is not None
        assert checkpoint_receipt is not None
        source_bindings.update(
            {
                "source_checkpoint": str(checkpoint_path),
                "source_checkpoint_file_sha256": _file_sha256(checkpoint_path),
                "source_checkpoint_hash": checkpoint.checkpoint_hash,
                "source_checkpoint_exactness": str(checkpoint_receipt_path),
                "source_checkpoint_exactness_file_sha256": _file_sha256(
                    checkpoint_receipt_path
                ),
                "source_checkpoint_exactness_content_sha256": (
                    checkpoint_receipt["integrity"]["content_sha256"]
                ),
            }
        )
    projection_payload = _seal_bound_payload(
        {
            "schema_version": PILOT_PROJECTION_SCHEMA_VERSION,
            "model_id": spec.model_id,
            "served_model": profile.served_model,
            "bindings": source_bindings,
            "projection": preflight_p95(
                _usage_projection_rows(capability, result),
                reserve_multiplier=float(
                    contract.budgets["pre_dispatch_projection"]["reserve_multiplier"]
                ),
            ),
        }
    )
    _atomic_bound_json(run_dir / "projection_p95.json", projection_payload)
    receipt = {
        "capability_pass": True,
        "preflight_checks": checks,
        "preflight_manifest": None if manifest is None else str(manifest),
        "preflight_checkpoint": (
            None if checkpoint_path is None else str(checkpoint_path)
        ),
        "preflight_checkpoint_exactness": (
            None if checkpoint_receipt_path is None else str(checkpoint_receipt_path)
        ),
        "projection": str(run_dir / "projection_p95.json"),
        "provider_call_journal": preflight_journal_binding,
        "request_route_binding": route_binding,
        "go": all(checks.values()),
        "reason": None if all(checks.values()) else "interface preflight failed",
    }
    _atomic_json(run_dir / "gate_receipt.json", receipt)
    status = "complete" if receipt["go"] else "capability-no-go"
    terminal = write_terminal_summary(
        raw_root / spec.stage_id / "summaries" / f"{spec.run_id}.json",
        contract=contract,
        run_spec=spec,
        resolved_git_commit=paid.head_commit,
        git_tag=paid.git_tag,
        payload={
            "metrics": {},
            "gate_evidence": receipt,
            "provider_call_journal": preflight_journal_binding,
            "capability": {
                **capability_payload,
                "preflight_go": receipt["go"],
                "preflight_checks": checks,
            },
        },
        scientific_evidence=False,
        diagnostic_only=False,
        evidence_scope="preregistered_capability_gate",
    )
    return status, terminal, budget, receipt


def _load_verified_projection(
    contract: PilotContract,
    model_id: str,
    *,
    raw_root: Path,
    paid: GitProvenance | None,
) -> tuple[dict[str, Any], Path]:
    preflight_stage = _preflight_stage_for_model(contract, model_id)
    capability_stage = _capability_source_stage(
        contract,
        preflight_stage,
        model_id,
    )
    preflight_specs = contract.expand(
        stage=preflight_stage,
        model=model_id,
    )
    capability_specs = contract.expand(stage=capability_stage, model=model_id)
    if len(preflight_specs) != 1 or len(capability_specs) != 1:
        raise PilotOrchestrationError(
            f"model {model_id} lacks one exact capability/preflight cell"
        )
    run_dir = raw_root / preflight_stage / "runs" / preflight_specs[0].run_id
    path = run_dir / "projection_p95.json"
    payload = _read_json(path)
    _verify_bound_payload(
        payload,
        contract=contract,
        schema_version=PILOT_PROJECTION_SCHEMA_VERSION,
        paid=paid,
        artifact_name=f"{model_id} preflight projection",
    )
    if (
        payload.get("model_id") != model_id
        or payload.get("served_model")
        != contract.provider_profiles[model_id].served_model
    ):
        raise PilotOrchestrationError("preflight projection model identity mismatch")
    bindings = payload["bindings"]
    manifest = run_dir / "preflight" / "manifest.json"
    capability_path = (
        raw_root
        / capability_stage
        / "runs"
        / capability_specs[0].run_id
        / "capability.json"
    )
    journal_path = _provider_call_journal_path(
        run_dir,
        run_id=preflight_specs[0].run_id,
        kind="preflight",
    )
    legacy_journal_path = run_dir / "preflight_provider_call_journal.json"
    if (
        not journal_path.exists()
        and not contract.schema_version.endswith("-v2")
        and legacy_journal_path.exists()
    ):
        journal_path = legacy_journal_path
    journal_binding: dict[str, Any] | None = None
    if contract.schema_version.endswith("-v2") or journal_path.exists():
        journal_binding = _verified_provider_call_journal_binding(
            journal_path,
            expected_run_id=(f"{preflight_specs[0].run_id}--actor-preflight"),
            expected_contract_hash=(
                contract.canonical_hash
                if contract.schema_version.endswith("-v2")
                else None
            ),
        )
    if bindings.get("source_capability") != str(capability_path) or (
        journal_binding is not None
        and (
            bindings.get("source_provider_call_journal") != str(journal_path)
            or bindings.get("source_provider_call_journal_file_sha256")
            != journal_binding["file_sha256"]
            or bindings.get("source_provider_call_journal_sha256")
            != journal_binding["journal_sha256"]
        )
    ):
        raise PilotOrchestrationError("preflight projection source path mismatch")
    if bindings.get("source_capability_sha256") != _file_sha256(capability_path):
        raise PilotOrchestrationError("preflight projection source hash mismatch")
    capability = _read_json(capability_path)
    if (
        capability.get("contract_sha256") != contract.canonical_hash
        or capability.get("run_spec") != capability_specs[0].to_dict()
    ):
        raise PilotOrchestrationError("preflight capability source binding mismatch")
    if contract.schema_version.endswith("-v2"):
        checkpoint_path = run_dir / "preflight_checkpoint.json"
        checkpoint = PilotCheckpoint(_read_json(checkpoint_path))
        exactness_path = run_dir / "preflight_checkpoint_exactness.json"
        exactness = _read_json(exactness_path)
        _verify_bound_payload(
            exactness,
            contract=contract,
            schema_version=PILOT_PREFLIGHT_CHECKPOINT_RECEIPT_SCHEMA_VERSION,
            paid=paid,
            artifact_name=f"{model_id} preflight checkpoint exactness",
        )
        expected_exactness = verify_closed_loop_preflight_checkpoint(checkpoint)
        if (
            exactness.get("exactness") != expected_exactness
            or bindings.get("source_checkpoint") != str(checkpoint_path)
            or bindings.get("source_checkpoint_file_sha256")
            != _file_sha256(checkpoint_path)
            or bindings.get("source_checkpoint_hash") != checkpoint.checkpoint_hash
            or bindings.get("source_checkpoint_exactness") != str(exactness_path)
            or bindings.get("source_checkpoint_exactness_file_sha256")
            != _file_sha256(exactness_path)
            or bindings.get("source_checkpoint_exactness_content_sha256")
            != exactness["integrity"]["content_sha256"]
            or exactness["bindings"].get("checkpoint_hash")
            != checkpoint.checkpoint_hash
            or exactness["bindings"].get("checkpoint_path") != str(checkpoint_path)
            or exactness["bindings"].get("checkpoint_file_sha256")
            != _file_sha256(checkpoint_path)
        ):
            raise PilotOrchestrationError(
                "preflight checkpoint/exactness source binding mismatch"
            )
        result = _CheckpointPreflightResult(checkpoint)
    else:
        if bindings.get("source_manifest") != str(manifest):
            raise PilotOrchestrationError("preflight projection source path mismatch")
        verification = verify_manifest(manifest.parent)
        if bindings.get("source_manifest_sha256") != verification.manifest_sha256:
            raise PilotOrchestrationError("preflight projection source hash mismatch")
        manifest_value = _read_json(manifest)
        if (
            manifest_value.get("git", {}).get("commit") != bindings.get("git_commit")
            or manifest_value.get("git", {}).get("dirty") is not False
        ):
            raise PilotOrchestrationError("preflight projection source git mismatch")
        result = load_verified_run_artifacts(manifest.parent)
        provenance = _read_json(manifest.parent / "provenance.json")
        details = provenance.get("details")
        if (
            not isinstance(details, Mapping)
            or details.get("contract_sha256") != contract.canonical_hash
            or details.get("run_spec") != preflight_specs[0].to_dict()
            or (
                journal_binding is not None
                and details.get("provider_call_journal") != journal_binding
            )
        ):
            raise PilotOrchestrationError(
                "preflight capability source binding mismatch"
            )
    reserve_multiplier = float(
        contract.budgets["pre_dispatch_projection"]["reserve_multiplier"]
    )
    recomputed = preflight_p95(
        _usage_projection_rows(capability, result),
        reserve_multiplier=reserve_multiplier,
    )
    if payload.get("projection") != recomputed:
        raise PilotOrchestrationError(
            "preflight projection differs from sealed api_usage p95"
        )
    if any(
        str(key).rpartition("::")[0] != payload["served_model"] for key in recomputed
    ):
        raise PilotOrchestrationError(
            "preflight projection contains a non-frozen served model"
        )
    return payload, path


def projection_from_preflight(
    contract: PilotContract,
    spec: PilotRunSpec,
    *,
    raw_root: str | Path,
    paid: GitProvenance | None = None,
) -> RunProjection:
    """Project one actor run from its model/call-kind preflight p95."""

    payload, path = _load_verified_projection(
        contract,
        spec.model_id,
        raw_root=Path(raw_root),
        paid=paid,
    )
    projection = payload.get("projection")
    if not isinstance(projection, Mapping):
        raise PilotOrchestrationError("preflight projection is malformed")

    by_kind: dict[str, Mapping[str, Any]] = {}
    for key, value in projection.items():
        _, separator, call_kind = str(key).rpartition("::")
        if not separator or not isinstance(value, Mapping):
            raise PilotOrchestrationError("preflight p95 key is malformed")
        by_kind[call_kind] = value

    calls_by_kind = {"action": spec.num_agents * spec.episode_length}
    arm_parameters = contract.arms[spec.arm_id]["parameters"]
    if bool(arm_parameters.get("semantic_actor_exposure", True)):
        due = sum(
            1
            for current_t in range(1, spec.episode_length + 1)
            if current_t >= 3 and (current_t - 3) % 3 == 0
        )
        calls_by_kind["semantic"] = spec.num_agents * min(due, 4)
    totals = {
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        "cost_usd": 0.0,
    }
    for call_kind, count in calls_by_kind.items():
        row = by_kind.get(call_kind)
        if row is None:
            raise PilotOrchestrationError(
                f"preflight has no {call_kind!r} p95 for {spec.model_id}"
            )
        reserved = row.get("reserved_p95")
        if not isinstance(reserved, Mapping):
            raise PilotOrchestrationError("preflight reserved p95 is malformed")
        for field in totals:
            totals[field] += float(reserved[field]) * count
    run_call_limit = sum(calls_by_kind.values())
    hosted_completion_cap_counted = _counts_toward_hosted_completion_cap(
        contract.provider_profiles[spec.model_id]
    )
    return RunProjection(
        run_id=spec.run_id,
        stage_bucket=spec.budget_bucket,
        cost_usd=totals["cost_usd"],
        completions=(run_call_limit if hosted_completion_cap_counted else 0),
        storage_bytes=20_000_000,
        basis={
            "method": "preflight-model-call-kind-p95-times-1.25",
            "source": str(path),
            "calls_by_kind": calls_by_kind,
            "run_call_limit": run_call_limit,
            "hosted_completion_cap_counted": (hosted_completion_cap_counted),
            "prompt_tokens": math.ceil(totals["prompt_tokens"]),
            "completion_tokens": math.ceil(totals["completion_tokens"]),
            "total_tokens": math.ceil(totals["total_tokens"]),
            "cost_usd": totals["cost_usd"],
        },
    )


def _execute_q_ref(
    contract: PilotContract,
    spec: PilotRunSpec,
    *,
    raw_root: Path,
    paid: GitProvenance,
    projection: RunProjection,
    budget: RunBudget | None = None,
) -> tuple[Path, RunBudget, Mapping[str, Any]]:
    run_dir = raw_root / spec.stage_id / "runs" / spec.run_id
    config = q_ref_run_config()
    llm = MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=4)
    budget = budget or _run_budget_from_projection(projection)
    result = run_verified_experiment(
        config,
        llm=llm,
        budget=budget,
        env_config_source=DEFAULT_ENV_CONFIG,
    )
    manifest = write_verified_run_artifacts(
        run_dir,
        result,
        provenance={
            **_provenance_payload(
                contract,
                spec,
                paid=paid,
                diagnostic=True,
            ),
            "purpose": "deterministic q_ref scale resolution",
        },
        git_commit=paid.head_commit,
        git_dirty=False,
    )
    resolution = build_q_ref_resolution(
        result,
        contract_hash=contract.canonical_hash,
        environment_source_hash=hashlib.sha256(
            DEFAULT_ENV_CONFIG.read_bytes()
        ).hexdigest(),
    )
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    resolution["bindings"]["source_manifest_sha256"] = manifest_hash
    resolution["bindings"]["contract_sha256"] = contract.canonical_hash
    resolution["bindings"]["git_tag"] = paid.git_tag
    resolution["bindings"]["git_commit"] = paid.head_commit
    resolution["source_manifest"] = str(manifest)
    resolution["scientific_evidence"] = False
    resolution = _seal_bound_payload(resolution)
    output = raw_root / spec.stage_id / "q_ref_resolution.json"
    _atomic_bound_json(output, resolution)
    terminal = write_terminal_summary(
        raw_root / spec.stage_id / "summaries" / f"{spec.run_id}.json",
        contract=contract,
        run_spec=spec,
        resolved_git_commit=paid.head_commit,
        git_tag=paid.git_tag,
        payload={
            "metrics": {},
            "gate_evidence": resolution["checks"],
            "q_ref_resolution": {
                "q_ref": resolution["q_ref"],
                "row_count": resolution["row_count"],
                "source_manifest": resolution["source_manifest"],
                "source_manifest_sha256": resolution["bindings"][
                    "source_manifest_sha256"
                ],
                "resolution_artifact": str(output),
            },
        },
        scientific_evidence=False,
        diagnostic_only=True,
        evidence_scope="deterministic_q_ref_resolution",
    )
    return terminal, budget, resolution


def _fixed_error_candidate() -> str:
    return json.dumps(
        {
            "context_scope": {"scope_id": "global", "predicates": []},
            "condition": {
                "field": "interest_rate",
                "operator": ">=",
                "value": 0.0,
                "tolerance": 0.0,
            },
            "action_guidance": {
                "target": "consumption_fraction",
                "direction": "at_most",
                "threshold": 0.0,
                "tolerance": 0.0,
            },
            "rationale": "Fixed unsupported rule for candidate-admission audit.",
            "supporting_episode_ids": ["unsupported-a", "unsupported-b"],
        },
        sort_keys=True,
    )


def _offline_candidate_admission(
    contract: PilotContract,
    spec: PilotRunSpec,
    *,
    raw_root: Path,
    diagnostic: bool,
    paid: GitProvenance | None,
) -> Path:
    """Replay one false candidate through verifier-on and verifier-off paths."""

    verified_episodic = EvidenceLinkedEpisodicTrack(
        run_id=spec.run_id,
        seed=spec.environment_seed,
        agent_id=0,
    )
    verified_semantic = VerifiedSemanticRuleTrack(verified_episodic)
    raw_candidate = _fixed_error_candidate()
    candidate = verified_semantic.parse_candidate(
        raw_candidate,
        generator_id="fixed-preregistered-error",
    )
    verified_rule = verified_semantic.submit_candidate(
        candidate,
        current_t=5,
    )
    unverified_episodic = EvidenceLinkedEpisodicTrack(
        run_id=f"{spec.run_id}--unverified-control",
        seed=spec.environment_seed,
        agent_id=0,
    )
    unverified_semantic = VerifiedSemanticRuleTrack(unverified_episodic)
    unverified_rule = unverified_semantic.propose_unverified_immediate(
        raw_candidate,
        current_t=5,
        generator_id="fixed-preregistered-error",
    )
    payload = {
        "schema_version": PILOT_OFFLINE_ADMISSION_SCHEMA_VERSION,
        "contract_sha256": contract.canonical_hash,
        "run_spec": spec.to_dict(),
        "candidate": candidate.to_dict(),
        "verified_rule": verified_rule.to_dict(),
        "unverified_rule": unverified_rule.to_dict(),
        "check": {
            "unsupported_candidate_rejected": (verified_rule.status == "rejected"),
            "false_rule_ever_active": verified_rule.status == "active",
            "unverified_false_rule_ever_active": (unverified_rule.status == "active"),
            "same_candidate_content": (
                unverified_rule.injection_provenance.get("raw_response_hash")
                == candidate.raw_response_hash
            ),
            "provider_calls": 0,
        },
        "diagnostic_only": diagnostic,
        "scientific_evidence": False if diagnostic else True,
    }
    if verified_rule.status != "rejected":
        raise PilotOrchestrationError(
            "verified candidate-admission fixture unexpectedly became active"
        )
    if (
        unverified_rule.status != "active"
        or payload["check"]["same_candidate_content"] is not True
    ):
        raise PilotOrchestrationError(
            "unverified candidate-admission control did not activate the "
            "same fixed candidate"
        )
    output = (
        raw_root
        / spec.stage_id
        / "runs"
        / spec.run_id
        / "offline_candidate_admission.json"
    )
    _atomic_json(output, payload)
    if paid is None or diagnostic:
        return output
    return write_terminal_summary(
        raw_root / spec.stage_id / "summaries" / f"{spec.run_id}.json",
        contract=contract,
        run_spec=spec,
        resolved_git_commit=paid.head_commit,
        git_tag=paid.git_tag,
        payload={
            "metrics": {
                "rule_reliability": payload["check"],
            },
            "gate_evidence": payload["check"],
            "offline_source": str(output),
        },
        scientific_evidence=True,
        diagnostic_only=False,
        evidence_scope=CURRENT_SCIENTIFIC_SCOPE,
    )


def _stage_receipt_path(raw_root: Path, stage_id: str) -> Path:
    return raw_root / stage_id / "stage_receipt.json"


def _stage_execution_modes(
    contract: PilotContract,
    stage_id: str,
) -> frozenset[str]:
    return frozenset(cell.execution_mode for cell in contract.stage(stage_id).cells)


def _is_capability_stage(
    contract: PilotContract,
    stage_id: str,
) -> bool:
    return bool(_stage_execution_modes(contract, stage_id) & CAPABILITY_EXECUTION_MODES)


def _capability_source_stage(
    contract: PilotContract,
    preflight_stage: str,
    model_id: str,
) -> str:
    """Resolve the capability task source for one preflight model."""

    modes = _stage_execution_modes(contract, preflight_stage)
    if "capability_probe" in modes and "closed_loop_preflight" not in modes:
        # Frozen V1 combines both operations in one cell.
        return preflight_stage
    candidates = [
        prerequisite
        for prerequisite in contract.stage(preflight_stage).prerequisites
        if "capability_probe" in _stage_execution_modes(contract, prerequisite)
        and model_id in contract.models_for_stage(prerequisite)
    ]
    if len(candidates) != 1:
        raise PilotOrchestrationError(
            f"{preflight_stage}/{model_id} does not have one exact "
            "capability source stage"
        )
    return candidates[0]


def _preflight_stage_for_model(
    contract: PilotContract,
    model_id: str,
) -> str:
    """Resolve exactly one projection-producing preflight for a model."""

    split = [
        stage_id
        for stage_id in contract.stage_ids
        if model_id in contract.models_for_stage(stage_id)
        and "closed_loop_preflight" in _stage_execution_modes(contract, stage_id)
    ]
    if len(split) == 1:
        return split[0]
    if split:
        raise PilotOrchestrationError(
            f"model {model_id} has multiple closed-loop preflight stages"
        )
    combined = [
        stage_id
        for stage_id in contract.stage_ids
        if model_id in contract.models_for_stage(stage_id)
        and "capability_probe" in _stage_execution_modes(contract, stage_id)
    ]
    if len(combined) != 1:
        raise PilotOrchestrationError(
            f"model {model_id} lacks one exact projection-producing preflight"
        )
    return combined[0]


def _scientific_stage_ids(contract: PilotContract) -> tuple[str, ...]:
    """Return contract-registered provider science stages, in contract order."""

    if not contract.schema_version.endswith("-v2"):
        return tuple(
            stage_id
            for stage_id in SCIENTIFIC_STAGE_IDS
            if stage_id in contract.stage_ids
        )
    excluded_modes = CAPABILITY_EXECUTION_MODES | {"q_ref_resolution"}
    return tuple(
        stage_id
        for stage_id in contract.stage_ids
        if not (_stage_execution_modes(contract, stage_id) & excluded_modes)
    )


def _cross_model_science_stage_ids(
    contract: PilotContract,
) -> tuple[str, ...]:
    if not contract.model_roles:
        return tuple(
            stage_id
            for stage_id in ("cross-model-sentinels",)
            if stage_id in contract.stage_ids
        )
    cross_roles = {"controlled_second", "secondary_diagnostic"}
    return tuple(
        stage_id
        for stage_id in _scientific_stage_ids(contract)
        if any(
            contract.model_roles[model_id].role in cross_roles
            for model_id in contract.models_for_stage(stage_id)
        )
    )


def _primary_model_ids(contract: PilotContract) -> frozenset[str]:
    if contract.model_roles:
        return frozenset(
            model_id
            for model_id, role in contract.model_roles.items()
            if role.role == "primary"
        )
    return frozenset({"gpt52_main"})


def _v2_stage_rows(
    contract: PilotContract,
    stage_id: str,
    ledger: PilotRunLedger,
) -> tuple[tuple[PilotRunSpec, ...], dict[str, Mapping[str, Any]], dict[str, Any]]:
    if not ledger.tamper_evident:
        raise PilotOrchestrationError("V2 stage gates require the V2 run ledger")
    snapshot = ledger.snapshot()
    specs = tuple(contract.expand(stage=stage_id))
    runs = snapshot.get("runs")
    if not isinstance(runs, Mapping):
        raise PilotOrchestrationError("V2 run ledger rows are malformed")
    rows: dict[str, Mapping[str, Any]] = {}
    for spec in specs:
        row = runs.get(spec.run_id)
        if not isinstance(row, Mapping) or row.get("spec") != spec.to_dict():
            raise PilotOrchestrationError(
                f"V2 run ledger lacks the exact contract cell {spec.run_id}"
            )
        rows[spec.run_id] = row
    return specs, rows, snapshot


def _v2_source_file_binding(raw_root: Path, path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = raw_root / candidate
    if candidate.is_symlink():
        raise PilotOrchestrationError(
            f"V2 gate source must not be a symlink: {candidate}"
        )
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(raw_root.resolve())
    except (FileNotFoundError, ValueError) as exc:
        raise PilotOrchestrationError(
            f"V2 gate source is missing or escapes raw_root: {candidate}"
        ) from exc
    if not resolved.is_file():
        raise PilotOrchestrationError(
            f"V2 gate source must be a regular file: {resolved}"
        )
    binding: dict[str, Any] = {
        "path": str(resolved),
        "file_sha256": _file_sha256(resolved),
    }
    if resolved.suffix.lower() == ".json":
        value = _read_json(resolved)
        integrity = value.get("integrity")
        if isinstance(integrity, Mapping):
            binding["content_sha256"] = integrity.get("content_sha256")
    return binding


def _load_v2_terminal_summary(
    contract: PilotContract,
    spec: PilotRunSpec,
    path: str | Path,
    *,
    raw_root: Path,
    paid: GitProvenance | None,
) -> dict[str, Any]:
    binding = _v2_source_file_binding(raw_root, path)
    value = _read_json(Path(binding["path"]))
    if value.get("schema_version") != PILOT_TERMINAL_SUMMARY_SCHEMA_VERSION:
        raise PilotOrchestrationError(
            f"{spec.run_id} capability artifact is not a terminal summary"
        )
    unsigned = _json_copy(value)
    integrity = unsigned.pop("integrity", None)
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("canonicalization") != PILOT_BOUND_ARTIFACT_CANONICALIZATION
        or integrity.get("content_sha256") != canonical_sha256(unsigned)
    ):
        raise PilotOrchestrationError(
            f"{spec.run_id} terminal summary integrity mismatch"
        )
    if (
        value.get("contract_id") != contract.contract_id
        or value.get("contract_sha256") != contract.canonical_hash
        or value.get("run_spec") != spec.to_dict()
    ):
        raise PilotOrchestrationError(
            f"{spec.run_id} terminal summary contract/spec mismatch"
        )
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise PilotOrchestrationError(
            f"{spec.run_id} terminal summary provenance is malformed"
        )
    if (
        provenance.get("git_tag")
        != contract.implementation["required_git_tag"]
        or provenance.get("tag_object_type") != "tag"
        or provenance.get("worktree_clean") is not True
        or (
            paid is not None
            and provenance.get("resolved_git_commit") != paid.head_commit
        )
    ):
        raise PilotOrchestrationError(
            f"{spec.run_id} terminal summary release binding mismatch"
        )
    return value


def _v2_capability_semantic_go(
    contract: PilotContract,
    spec: PilotRunSpec,
    row: Mapping[str, Any],
    *,
    raw_root: Path,
    paid: GitProvenance | None,
) -> bool:
    status = str(row.get("status"))
    artifact = row.get("artifact")
    if status not in {"complete", "capability-no-go"}:
        return False
    if artifact is None:
        if status == "capability-no-go":
            return False
        raise PilotOrchestrationError(
            f"passing capability cell {spec.run_id} lacks a sealed artifact"
        )
    artifact_value = _read_json(Path(_v2_source_file_binding(raw_root, artifact)["path"]))
    if artifact_value.get("schema_version") != PILOT_TERMINAL_SUMMARY_SCHEMA_VERSION:
        if status != "capability-no-go":
            raise PilotOrchestrationError(
                f"passing capability cell {spec.run_id} lacks a terminal summary"
            )
        try:
            verified = verify_provider_catalog_receipt(
                artifact_value,
                contract_hash=contract.canonical_hash,
                require_pass=False,
            )
        except Exception as exc:
            raise PilotOrchestrationError(
                f"{spec.run_id} catalog no-go receipt is invalid"
            ) from exc
        if (
            verified.get("status") != "no-go"
            or verified.get("model_id") != spec.model_id
        ):
            raise PilotOrchestrationError(
                f"{spec.run_id} catalog no-go receipt has the wrong model/status"
            )
        return False

    summary = _load_v2_terminal_summary(
        contract,
        spec,
        artifact,
        raw_root=raw_root,
        paid=paid,
    )
    payload = summary.get("payload")
    if not isinstance(payload, Mapping):
        raise PilotOrchestrationError(
            f"{spec.run_id} capability terminal payload is malformed"
        )
    capability = payload.get("capability")
    gate = payload.get("gate_evidence")
    if not isinstance(capability, Mapping) or not isinstance(gate, Mapping):
        raise PilotOrchestrationError(
            f"{spec.run_id} capability gate evidence is malformed"
        )
    run_dir = raw_root / spec.stage_id / "runs" / spec.run_id
    gate_path = run_dir / "gate_receipt.json"
    if _read_json(gate_path) != gate:
        raise PilotOrchestrationError(
            f"{spec.run_id} terminal gate differs from its source receipt"
        )
    try:
        _validate_capability_v3(capability)
    except PilotEvidenceError as exc:
        raise PilotOrchestrationError(
            f"{spec.run_id} capability evidence failed semantic recomputation: {exc}"
        ) from exc
    capability_spec = spec
    if spec.execution_mode == "closed_loop_preflight":
        source_stage = _capability_source_stage(
            contract,
            spec.stage_id,
            spec.model_id,
        )
        source_specs = contract.expand(
            stage=source_stage,
            model=spec.model_id,
        )
        if len(source_specs) != 1:
            raise PilotOrchestrationError(
                f"{spec.run_id} lacks one exact capability source cell"
            )
        capability_spec = source_specs[0]
    if (
        capability.get("contract_sha256") != contract.canonical_hash
        or capability.get("run_spec") != capability_spec.to_dict()
    ):
        raise PilotOrchestrationError(
            f"{spec.run_id} capability payload contract/spec mismatch"
        )

    if spec.execution_mode == "capability_probe":
        semantic_go = capability.get("pass") is True
        if _read_json(run_dir / "capability.json") != capability:
            raise PilotOrchestrationError(
                f"{spec.run_id} terminal capability differs from its source"
            )
        if (
            gate.get("capability_pass") is not semantic_go
            or gate.get("interface_pass")
            is not (capability.get("interface_gate", {}).get("pass") is True)
            or gate.get("go") is not semantic_go
            or gate.get("preflight_run") is not None
        ):
            raise PilotOrchestrationError(
                f"{spec.run_id} capability gate differs from recomputed rows"
            )
    elif spec.execution_mode == "closed_loop_preflight":
        projection, projection_path = _load_verified_projection(
            contract,
            spec.model_id,
            raw_root=raw_root,
            paid=paid,
        )
        del projection
        checkpoint_path = run_dir / "preflight_checkpoint.json"
        checkpoint = PilotCheckpoint(_read_json(checkpoint_path))
        checks = _preflight_checks(
            _CheckpointPreflightResult(checkpoint),
            contract.provider_profiles[spec.model_id],
            checkpoint_preflight=True,
        )
        semantic_go = capability.get("pass") is True and all(checks.values())
        capability_stage = _capability_source_stage(
            contract,
            spec.stage_id,
            spec.model_id,
        )
        source_spec = contract.expand(
            stage=capability_stage,
            model=spec.model_id,
        )[0]
        source_capability_path = (
            raw_root
            / capability_stage
            / "runs"
            / source_spec.run_id
            / "capability.json"
        )
        source_capability = _read_json(source_capability_path)
        expected_capability = {
            **source_capability,
            "preflight_go": semantic_go,
            "preflight_checks": checks,
        }
        exactness_path = run_dir / "preflight_checkpoint_exactness.json"
        if (
            capability != expected_capability
            or capability.get("preflight_checks") != checks
            or gate.get("preflight_checks") != checks
            or gate.get("go") is not semantic_go
            or gate.get("projection") != str(projection_path)
            or gate.get("preflight_manifest") is not None
            or gate.get("preflight_checkpoint") != str(checkpoint_path)
            or gate.get("preflight_checkpoint_exactness")
            != str(exactness_path)
        ):
            raise PilotOrchestrationError(
                f"{spec.run_id} preflight gate differs from sealed source recomputation"
            )
    else:  # pragma: no cover - contract validator owns capability modes
        raise PilotOrchestrationError(
            f"unsupported V2 capability mode: {spec.execution_mode}"
        )

    if (status == "complete") is not semantic_go:
        raise PilotOrchestrationError(
            f"{spec.run_id} ledger status differs from recomputed capability gate"
        )
    return semantic_go


def _v2_stage_control_paths(
    contract: PilotContract,
    stage_id: str,
    *,
    raw_root: Path,
) -> tuple[Path, ...]:
    paths: set[Path] = set()
    stage_root = raw_root / stage_id
    for profile_id in contract.models_for_stage(stage_id):
        catalog = stage_root / "provider_catalog" / f"{profile_id}.json"
        if catalog.exists():
            paths.add(catalog)
    if stage_id == "q-ref-resolution":
        path = stage_root / "q_ref_resolution.json"
        if path.exists():
            paths.add(path)
    elif stage_id == "stage0-calibration":
        path = stage_root / "stage0_selection.json"
        if path.exists():
            paths.add(path)
    elif stage_id == "experiment-c":
        path = stage_root / "rule_sensitivity.json"
        if path.exists():
            paths.add(path)
    if _is_capability_stage(contract, stage_id):
        for spec in contract.expand(stage=stage_id):
            run_dir = stage_root / "runs" / spec.run_id
            for name in (
                "capability.json",
                "gate_receipt.json",
                "projection_p95.json",
                "preflight_checkpoint.json",
                "preflight_checkpoint_exactness.json",
            ):
                path = run_dir / name
                if path.exists():
                    paths.add(path)
            paths.update(run_dir.glob("*provider_call_journal*.json"))
    return tuple(sorted(paths, key=lambda item: str(item)))


def _v2_stage_receipt_bindings(
    contract: PilotContract,
    stage_id: str,
    *,
    raw_root: Path,
    ledger: PilotRunLedger,
    paid: GitProvenance | None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    specs, rows, snapshot = _v2_stage_rows(contract, stage_id, ledger)
    source_files: list[dict[str, Any]] = []
    for spec in specs:
        artifact = rows[spec.run_id].get("artifact")
        if artifact is not None:
            source_files.append(_v2_source_file_binding(raw_root, str(artifact)))
    source_files.extend(
        _v2_source_file_binding(raw_root, path)
        for path in _v2_stage_control_paths(
            contract,
            stage_id,
            raw_root=raw_root,
        )
    )
    deduplicated = {
        str(item["path"]): item for item in source_files
    }
    sources = [
        deduplicated[path] for path in sorted(deduplicated)
    ]
    go_models: list[str] = []
    if _is_capability_stage(contract, stage_id):
        for spec in specs:
            if _v2_capability_semantic_go(
                contract,
                spec,
                rows[spec.run_id],
                raw_root=raw_root,
                paid=paid,
            ):
                go_models.append(spec.model_id)
    events = snapshot.get("events")
    if not isinstance(events, list) or not events:
        raise PilotOrchestrationError("V2 run ledger lacks its event chain")
    stage_rows = {spec.run_id: rows[spec.run_id] for spec in specs}
    bindings = {
        "contract_sha256": contract.canonical_hash,
        "run_ledger_schema_version": PILOT_RUN_LEDGER_SCHEMA_VERSION_V2,
        "ledger_event_count": len(events),
        "ledger_event_chain_head": events[-1]["event_sha256"],
        "stage_specs_sha256": canonical_sha256(
            [spec.to_dict() for spec in specs]
        ),
        "stage_rows_sha256": canonical_sha256(stage_rows),
        "source_files": sources,
        "source_files_sha256": canonical_sha256(sources),
    }
    return bindings, tuple(dict.fromkeys(go_models))


def _v2_control_gate_ok(
    contract: PilotContract,
    stage_id: str,
    *,
    raw_root: Path,
    paid: GitProvenance | None,
) -> bool:
    if stage_id == "q-ref-resolution":
        try:
            value = _load_verified_q_ref(
                contract,
                raw_root=raw_root,
                paid=paid,
            )
        except PilotOrchestrationError as exc:
            if "required artifact is missing" in str(exc):
                return False
            raise
        return value.get("status") == "pass"
    if stage_id == "stage0-calibration":
        try:
            _load_verified_stage0_selection(
                contract,
                raw_root=raw_root,
                paid=paid,
            )
        except PilotOrchestrationError as exc:
            if "required artifact is missing" in str(exc):
                return False
            raise
    elif stage_id == "experiment-c":
        path = _experiment_c_sensitivity_path(raw_root)
        if not path.exists():
            return False
        _load_verified_experiment_c_sensitivity(
            contract,
            raw_root=raw_root,
            paid=paid,
        )
    return True


def _v2_recomputed_stage_fields(
    contract: PilotContract,
    stage_id: str,
    *,
    raw_root: Path,
    ledger: PilotRunLedger,
    paid: GitProvenance | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    specs, rows, _ = _v2_stage_rows(contract, stage_id, ledger)
    bindings, go_models = _v2_stage_receipt_bindings(
        contract,
        stage_id,
        raw_root=raw_root,
        ledger=ledger,
        paid=paid,
    )
    counts: dict[str, int] = {}
    for row in rows.values():
        row_status = str(row.get("status"))
        counts[row_status] = counts.get(row_status, 0) + 1
    terminal = len(rows) == len(specs) and all(
        row.get("status") in TERMINAL_RUN_STATUSES for row in rows.values()
    )
    complete_count = counts.get("complete", 0)
    matrix_complete = terminal and complete_count == len(specs) and bool(specs)
    control_go = (
        _v2_control_gate_ok(
            contract,
            stage_id,
            raw_root=raw_root,
            paid=paid,
        )
        if matrix_complete
        else False
    )
    go = matrix_complete and control_go
    hard_stop_count = sum(
        counts.get(run_status, 0) for run_status in HARD_STOP_RUN_STATUSES
    )
    semantic_status = "complete" if go else "complete-with-no-go"
    execution_progression_go = (
        terminal
        and hard_stop_count == 0
        and (
            stage_id in {*CORE_STAGE_IDS, "cross-model-sentinels"}
            or (
                _is_capability_stage(contract, stage_id)
                and bool(go_models)
            )
            or go
        )
    )
    return (
        {
            "terminal": terminal,
            "denominator_terminal": terminal,
            "registered_run_count": len(specs),
            "complete_cell_count": complete_count,
            "status_counts": counts,
            "scientific_matrix_complete": matrix_complete,
            "go": go,
            "execution_progression_go": execution_progression_go,
            "hard_stop_cell_count": hard_stop_count,
            "go_models": list(go_models),
            "semantic_status": semantic_status,
        },
        bindings,
    )


def _verify_v2_stage_receipt(
    contract: PilotContract,
    stage_id: str,
    receipt: Mapping[str, Any],
    *,
    raw_root: Path,
    ledger: PilotRunLedger,
    paid: GitProvenance | None,
) -> dict[str, Any]:
    if (
        receipt.get("schema_version") != PILOT_STAGE_RECEIPT_SCHEMA_VERSION_V2
        or receipt.get("contract_sha256") != contract.canonical_hash
        or receipt.get("stage_id") != stage_id
    ):
        raise PilotOrchestrationError(
            f"V2 stage receipt identity mismatch for {stage_id}"
        )
    unsigned = _json_copy(receipt)
    integrity = unsigned.pop("integrity", None)
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("canonicalization") != PILOT_BOUND_ARTIFACT_CANONICALIZATION
        or integrity.get("content_sha256") != canonical_sha256(unsigned)
    ):
        raise PilotOrchestrationError(
            f"V2 stage receipt content hash mismatch for {stage_id}"
        )
    fields, current_bindings = _v2_recomputed_stage_fields(
        contract,
        stage_id,
        raw_root=raw_root,
        ledger=ledger,
        paid=paid,
    )
    bindings = receipt.get("bindings")
    if not isinstance(bindings, Mapping):
        raise PilotOrchestrationError(
            f"V2 stage receipt bindings are malformed for {stage_id}"
        )
    for key in (
        "contract_sha256",
        "run_ledger_schema_version",
        "stage_specs_sha256",
        "stage_rows_sha256",
        "source_files",
        "source_files_sha256",
    ):
        if bindings.get(key) != current_bindings.get(key):
            raise PilotOrchestrationError(
                f"V2 stage receipt source binding {key!r} drifted for {stage_id}"
            )
    event_count = bindings.get("ledger_event_count")
    snapshot = ledger.snapshot()
    events = snapshot.get("events")
    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count < 1
        or not isinstance(events, list)
        or event_count > len(events)
        or events[event_count - 1].get("event_sha256")
        != bindings.get("ledger_event_chain_head")
    ):
        raise PilotOrchestrationError(
            f"V2 stage receipt ledger event prefix mismatch for {stage_id}"
        )
    for key, expected in fields.items():
        if key == "semantic_status":
            continue
        if receipt.get(key) != expected:
            raise PilotOrchestrationError(
                f"V2 stage receipt field {key!r} differs from source "
                f"recomputation for {stage_id}"
            )
    if fields["hard_stop_cell_count"] == 0 and receipt.get(
        "status"
    ) != fields["semantic_status"]:
        raise PilotOrchestrationError(
            f"V2 stage receipt status differs from source recomputation for {stage_id}"
        )
    return _json_copy(receipt)


def _assert_prerequisites(
    contract: PilotContract,
    stage_id: str,
    *,
    raw_root: Path,
    paid: GitProvenance | None = None,
    ledger: PilotRunLedger | None = None,
) -> dict[str, frozenset[str]]:
    stage = contract.stage(stage_id)
    verified_go_models: dict[str, frozenset[str]] = {}
    v2 = contract.schema_version.endswith("-v2")
    if v2 and ledger is None:
        ledger = PilotRunLedger(
            raw_root / "run_ledger.json",
            contract_hash=contract.canonical_hash,
            tamper_evident=True,
        )
    for prerequisite in stage.prerequisites:
        if v2:
            # Revalidate the complete prerequisite ancestry, not only the
            # immediately preceding stage.  This keeps the original
            # capability/preflight sources on every downstream dispatch path.
            _assert_prerequisites(
                contract,
                prerequisite,
                raw_root=raw_root,
                paid=paid,
                ledger=ledger,
            )
        receipt = _read_json(_stage_receipt_path(raw_root, prerequisite))
        if v2:
            assert ledger is not None
            receipt = _verify_v2_stage_receipt(
                contract,
                prerequisite,
                receipt,
                raw_root=raw_root,
                ledger=ledger,
                paid=paid,
            )
        else:
            if receipt.get("schema_version") != PILOT_STAGE_RECEIPT_SCHEMA_VERSION:
                raise PilotOrchestrationError(
                    f"prerequisite {prerequisite} has an unsupported receipt"
                )
            if receipt.get("contract_sha256") != contract.canonical_hash:
                raise PilotOrchestrationError(
                    f"prerequisite {prerequisite} contract hash mismatch"
                )
        if receipt.get("terminal") is not True:
            raise PilotOrchestrationError(
                f"prerequisite {prerequisite} is not terminal"
            )
        receipt_status = receipt.get("status")
        if receipt_status not in {"complete", "complete-with-no-go"}:
            raise PilotOrchestrationError(
                f"prerequisite {prerequisite} did not complete"
            )
        passed_models = frozenset(
            str(model_id) for model_id in receipt.get("go_models", [])
        )
        verified_go_models[prerequisite] = passed_models
        if v2 and _is_capability_stage(contract, prerequisite):
            prerequisite_models = set(contract.models_for_stage(prerequisite))
            target_models = set(contract.models_for_stage(stage_id))
            relevant_models = prerequisite_models & target_models
            if not relevant_models:
                # A cohort boundary (primary preflight -> secondary
                # capability) is allowed only after the contract-declared
                # primary backbone has passed.
                required_primary = prerequisite_models & _primary_model_ids(contract)
                if required_primary and not (required_primary & passed_models):
                    raise PilotOrchestrationError(
                        f"prerequisite {prerequisite} has no passing "
                        "contract-declared primary model"
                    )
            elif (
                not _is_capability_stage(contract, stage_id)
                and stage_id not in _cross_model_science_stage_ids(contract)
                and not (relevant_models & passed_models)
            ):
                raise PilotOrchestrationError(
                    f"prerequisite {prerequisite} has no passing model "
                    f"eligible for {stage_id}"
                )
            # Gate receipts may be complete-with-no-go by design. Model-level
            # filtering and ITT propagation handle the failed cells.
            continue
        if stage_id != "cross-model-sentinels" and prerequisite == (
            "capability-preflight"
        ):
            if "gpt52_main" not in receipt.get("go_models", []):
                raise PilotOrchestrationError(
                    "GPT-5.2 capability/preflight is no-go; core stages cannot run"
                )
        if prerequisite != "capability-preflight":
            progression_go = receipt.get("execution_progression_go")
            if progression_go is None:
                # Read-only compatibility for pre-field v1 receipts.  Old
                # complete-with-no-go receipts remain fail-closed because they
                # did not distinguish scientific claim support from execution
                # eligibility.
                progression_go = (
                    receipt_status == "complete" and receipt.get("go") is True
                )
            if progression_go is not True:
                raise PilotOrchestrationError(
                    f"prerequisite {prerequisite} has an execution no-go"
                )
        # Experiment C sensitivity is a preregistered descriptive artifact and
        # is validated by the evidence publisher.  Missing or failed
        # sensitivity makes the C claim a no-go, but must not prevent the
        # independently registered Experiment D denominator from running.
    return verified_go_models


def _write_stage_receipt(
    contract: PilotContract,
    stage_id: str,
    *,
    raw_root: Path,
    ledger: PilotRunLedger,
    status: str,
    go_models: Sequence[str] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    failure: Mapping[str, Any] | None = None,
    diagnostic: bool = False,
    paid: GitProvenance | None = None,
) -> Path:
    output = _stage_receipt_path(raw_root, stage_id)
    if contract.schema_version.endswith("-v2"):
        fields, bindings = _v2_recomputed_stage_fields(
            contract,
            stage_id,
            raw_root=raw_root,
            ledger=ledger,
            paid=paid,
        )
        declared_go_models = (
            None
            if go_models is None
            else list(dict.fromkeys(str(item) for item in go_models))
        )
        if (
            declared_go_models is not None
            and declared_go_models != fields["go_models"]
        ):
            raise PilotOrchestrationError(
                f"V2 {stage_id} go_models differ from sealed source recomputation"
            )
        if fields["hard_stop_cell_count"] == 0 and status != fields[
            "semantic_status"
        ]:
            raise PilotOrchestrationError(
                f"V2 {stage_id} status differs from sealed source recomputation"
            )
        payload = {
            "schema_version": PILOT_STAGE_RECEIPT_SCHEMA_VERSION_V2,
            "contract_id": contract.contract_id,
            "contract_sha256": contract.canonical_hash,
            "stage_id": stage_id,
            "status": status,
            **{
                key: value
                for key, value in fields.items()
                if key != "semantic_status"
            },
            "bindings": bindings,
            "artifacts": dict(artifacts or {}),
            "failure": dict(failure) if failure else None,
            "diagnostic_only": diagnostic,
            "scientific_evidence": False if diagnostic else None,
            "created_at": _utc_now(),
        }
        if output.exists():
            existing = _read_json(output)
            _verify_v2_stage_receipt(
                contract,
                stage_id,
                existing,
                raw_root=raw_root,
                ledger=ledger,
                paid=paid,
            )
            payload["created_at"] = existing.get("created_at")
        unsigned = _json_copy(payload)
        payload["integrity"] = {
            "canonicalization": PILOT_BOUND_ARTIFACT_CANONICALIZATION,
            "content_sha256": canonical_sha256(unsigned),
        }
        if output.exists():
            if existing != payload:
                raise PilotOrchestrationError(
                    f"immutable V2 stage receipt drifted for {stage_id}"
                )
            return output
        _atomic_json_no_overwrite(output, payload)
        return output

    specs = contract.expand(stage=stage_id)
    snapshot = ledger.snapshot()
    rows = [
        snapshot["runs"][spec.run_id]
        for spec in specs
        if spec.run_id in snapshot["runs"]
    ]
    counts: dict[str, int] = {}
    for row in rows:
        row_status = str(row["status"])
        counts[row_status] = counts.get(row_status, 0) + 1
    terminal = len(rows) == len(specs) and all(
        row["status"] in TERMINAL_RUN_STATUSES for row in rows
    )
    complete_count = counts.get("complete", 0)
    scientific_matrix_complete = (
        terminal and complete_count == len(specs) and complete_count > 0
    )
    go = scientific_matrix_complete and status == "complete"
    hard_stop_count = sum(
        counts.get(run_status, 0) for run_status in HARD_STOP_RUN_STATUSES
    )
    execution_progression_go = (
        terminal
        and status in {"complete", "complete-with-no-go"}
        and hard_stop_count == 0
        and (
            stage_id in {*CORE_STAGE_IDS, "cross-model-sentinels"}
            or (
                contract.schema_version.endswith("-v2")
                and _is_capability_stage(contract, stage_id)
                and bool(go_models)
            )
            or go
        )
    )
    payload = {
        "schema_version": PILOT_STAGE_RECEIPT_SCHEMA_VERSION,
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "stage_id": stage_id,
        "status": status,
        "terminal": terminal,
        "denominator_terminal": terminal,
        "registered_run_count": len(specs),
        "complete_cell_count": complete_count,
        "status_counts": counts,
        "scientific_matrix_complete": scientific_matrix_complete,
        "go": go,
        "execution_progression_go": execution_progression_go,
        "hard_stop_cell_count": hard_stop_count,
        "go_models": list(dict.fromkeys(go_models or ())),
        "artifacts": dict(artifacts or {}),
        "failure": dict(failure) if failure else None,
        "diagnostic_only": diagnostic,
        "scientific_evidence": False if diagnostic else None,
        "created_at": _utc_now(),
    }
    _atomic_json(output, payload)
    return output


def _descendant_stage_ids(
    contract: PilotContract,
    source_stage: str,
) -> tuple[str, ...]:
    descendants: list[str] = []
    known = {source_stage}
    for stage_id in contract.stage_ids:
        if stage_id == source_stage:
            continue
        if any(
            prerequisite in known
            for prerequisite in contract.stage(stage_id).prerequisites
        ):
            known.add(stage_id)
            descendants.append(stage_id)
    return tuple(descendants)


def _propagate_stage_no_go(
    contract: PilotContract,
    *,
    source_stage: str,
    ledger: PilotRunLedger,
    failure: Mapping[str, Any],
    status: str = "integrity-stopped",
) -> None:
    """Terminalize every still-pending causal descendant of a no-go stage."""

    for descendant in _descendant_stage_ids(contract, source_stage):
        ledger.stop_pending(
            contract.expand(stage=descendant),
            status=status,
            failure={
                **dict(failure),
                "source_stage": source_stage,
                "blocked_stage": descendant,
            },
        )


def _propagate_capability_no_go(
    contract: PilotContract,
    *,
    ledger: PilotRunLedger,
    source_stage: str = "capability-preflight",
) -> None:
    """Propagate each failed gate model through its causal descendants."""

    capability_by_model = {
        spec.model_id: ledger.status(spec.run_id)
        for spec in contract.expand(stage=source_stage)
    }
    descendants = _descendant_stage_ids(contract, source_stage)
    for model_id, status in capability_by_model.items():
        if status == "complete":
            continue
        failure = {
            "error_type": "CapabilityOrInterfaceNoGo",
            "message": (f"{model_id} did not pass its fixed capability/interface gate"),
            "source_stage": source_stage,
            "model_id": model_id,
        }
        for stage_id in descendants:
            for spec in contract.expand(stage=stage_id):
                if spec.model_id == model_id and not ledger.is_terminal(spec.run_id):
                    ledger.finalize(
                        spec.run_id,
                        status="capability-no-go",
                        artifact=None,
                        failure={**failure, "blocked_stage": stage_id},
                    )
        # V1's core main-backbone rows are all gpt52_main and are covered by
        # the same model match. Keep this explicit invariant fail-closed if a
        # future V1-shaped contract names the role but not the profile.
        if not contract.schema_version.endswith("-v2") and model_id == "gpt52_main":
            for stage_id in CORE_STAGE_IDS:
                if stage_id not in contract.stage_ids:
                    continue
                ledger.stop_pending(
                    contract.expand(stage=stage_id),
                    status="capability-no-go",
                    failure={**failure, "blocked_stage": stage_id},
                )


def _select_stage0(
    contract: PilotContract,
    *,
    raw_root: Path,
    specs: Sequence[PilotRunSpec],
    ledger: PilotRunLedger,
    paid: GitProvenance,
) -> Path:
    expected_specs = tuple(contract.expand(stage="stage0-calibration"))
    if {spec.run_id for spec in specs} != {
        spec.run_id for spec in expected_specs
    } or len(specs) != 14:
        raise PilotOrchestrationError(
            "Stage-0 selection requires the exact 7x2 contract matrix"
        )
    if any(ledger.status(spec.run_id) != "complete" for spec in specs):
        raise PilotOrchestrationError(
            "Stage-0 selection requires every preregistered calibration cell"
        )
    q_ref = _load_verified_q_ref(contract, raw_root=raw_root, paid=paid)
    ofat = expand_stage0_ofat(float(q_ref["q_ref"]))
    by_profile: dict[str, list[Mapping[str, Any]]] = {
        profile_id: [] for profile_id in ofat["profile_order"]
    }
    ledger_rows = ledger.snapshot()["runs"]
    source_manifests: list[dict[str, Any]] = []
    calibration_sources: list[tuple[PilotRunSpec, Any, Mapping[str, Any]]] = []
    for spec in specs:
        artifact = ledger_rows[spec.run_id].get("artifact")
        if not isinstance(artifact, str):
            raise PilotOrchestrationError(
                "Stage-0 completed cell lacks its sealed runner artifact"
            )
        manifest = Path(artifact)
        run_dir = manifest.parent if manifest.name == "manifest.json" else manifest
        result = load_verified_run_artifacts(run_dir)
        if result.config.get("run_id") != spec.run_id:
            raise PilotOrchestrationError(
                "Stage-0 sealed runner run_id differs from its contract cell"
            )
        verification = verify_manifest(run_dir)
        manifest_value = _read_json(run_dir / "manifest.json")
        if (
            manifest_value.get("git", {}).get("commit") != paid.head_commit
            or manifest_value.get("git", {}).get("dirty") is not False
        ):
            raise PilotOrchestrationError(
                "Stage-0 source manifest git identity mismatch"
            )
        manifest_binding = {
            "run_id": spec.run_id,
            "utility_profile_id": spec.utility_profile_id,
            "environment_seed": spec.environment_seed,
            "manifest": str(run_dir / "manifest.json"),
            "manifest_sha256": verification.manifest_sha256,
        }
        source_manifests.append(manifest_binding)
        calibration_sources.append((spec, result, manifest_binding))
        summary = summarize_run(
            result.records,
            max_labor_hours=float(result.config["max_labor_hours"]),
            schedule=contract.shocks[spec.shock_id]["schedule"],
        )
        by_profile[spec.utility_profile_id].append(summary)
    selection = select_stage0_profile(ofat, by_profile)
    selection["absolute_flow_utility_threshold"] = (
        _derive_stage0_absolute_flow_threshold(
            contract,
            selected_profile_id=str(selection["selected_profile_id"]),
            sources=calibration_sources,
        )
    )
    q_ref_path = raw_root / "q-ref-resolution" / "q_ref_resolution.json"
    selection.update(
        {
            "contract_sha256": contract.canonical_hash,
            "q_ref_artifact_hash": canonical_sha256(q_ref),
            "bindings": {
                "contract_sha256": contract.canonical_hash,
                "git_tag": paid.git_tag,
                "git_commit": paid.head_commit,
                "q_ref_content_sha256": q_ref["integrity"]["content_sha256"],
                "q_ref_file_sha256": _file_sha256(q_ref_path),
                "source_manifests": sorted(
                    source_manifests,
                    key=lambda row: row["run_id"],
                ),
            },
        }
    )
    selection = _seal_bound_payload(selection)
    output = raw_root / "stage0-calibration" / "stage0_selection.json"
    _atomic_bound_json(output, selection)
    return output


def _experiment_c_sensitivity_path(raw_root: Path) -> Path:
    return raw_root / "experiment-c" / "rule_sensitivity.json"


def _verifier_config_from_runner_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    fields = (
        "min_candidate_support",
        "activation_min_support",
        "activation_min_margin",
        "activation_confidence_threshold",
        "retirement_patience",
        "retirement_confidence_threshold",
    )
    result: dict[str, Any] = {}
    for field in fields:
        if field not in config:
            raise PilotOrchestrationError(
                "sensitivity-source runner config is missing verifier field "
                f"{field!r}"
            )
        result[field] = config[field]
    return result


def _build_experiment_c_sensitivity(
    contract: PilotContract,
    *,
    raw_root: Path,
    git_tag: str,
    git_commit: str,
) -> dict[str, Any]:
    """Recompute descriptive C sensitivity from the frozen full-control arm."""

    sensitivity_contract = contract.stop_go["experiment_c"]["zero_api_sensitivity"]
    if (
        tuple(sensitivity_contract["alternative_success_weights"])
        != ALTERNATIVE_SUCCESS_WEIGHTS
        or tuple(sensitivity_contract["outcome_definitions"]) != OUTCOME_DEFINITIONS
    ):
        raise PilotOrchestrationError(
            "runtime sensitivity grid differs from the frozen contract"
        )
    selection = _load_verified_stage0_selection(
        contract,
        raw_root=raw_root,
        paid=None,
    )
    threshold = selection.get("absolute_flow_utility_threshold")
    if not isinstance(threshold, Mapping):
        raise PilotOrchestrationError(
            "sealed Stage-0 selection lacks the absolute flow-utility threshold"
        )
    threshold_value = threshold.get("value")
    if (
        isinstance(threshold_value, bool)
        or not isinstance(threshold_value, (int, float))
        or not math.isfinite(float(threshold_value))
    ):
        raise PilotOrchestrationError(
            "sealed Stage-0 absolute flow-utility threshold is invalid"
        )

    source_stage = (
        "experiment-c" if contract.schema_version.endswith("-v2") else "experiment-b"
    )
    source_label = f"{source_stage}/full"
    expected_specs = tuple(
        contract.expand(
            stage=source_stage,
            model="gpt52_main",
            arm="full",
        )
    )
    if len(expected_specs) != 5:
        raise PilotOrchestrationError(
            "C sensitivity requires the exact five " f"{source_label} main-seed runs"
        )

    source_manifests: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []
    common_verifier: dict[str, Any] | None = None
    for spec in expected_specs:
        run_dir = raw_root / source_stage / "runs" / spec.run_id
        manifest_path = run_dir / "manifest.json"
        verification = verify_manifest(run_dir)
        manifest = _read_json(manifest_path)
        if (
            manifest.get("git", {}).get("commit") != git_commit
            or manifest.get("git", {}).get("dirty") is not False
        ):
            raise PilotOrchestrationError(
                f"{source_label} sensitivity source has the wrong git identity"
            )
        provenance = _read_json(run_dir / "provenance.json")
        details = provenance.get("details")
        result = load_verified_run_artifacts(run_dir)
        if (
            result.config.get("run_id") != spec.run_id
            or result.config.get("seed") != spec.environment_seed
            or not isinstance(details, Mapping)
            or details.get("contract_sha256") != contract.canonical_hash
            or details.get("run_spec") != spec.to_dict()
        ):
            raise PilotOrchestrationError(
                f"{source_label} sensitivity source identity differs from "
                "its contract cell"
            )
        episodes = [dict(row) for row in result.stream("episodes")]
        proposals = [dict(row) for row in result.stream("semantic_proposals")]
        rules = [dict(row) for row in result.stream("semantic_rules")]
        expected_episode_count = spec.num_agents * spec.episode_length
        expected_proposal_count = spec.num_agents * (
            (spec.episode_length - int(result.config["semantic_proposal_after"]))
            // int(result.config["semantic_proposal_interval"])
            + 1
        )
        if (
            len(episodes) != expected_episode_count
            or len(proposals) != expected_proposal_count
        ):
            raise PilotOrchestrationError(
                f"{source_label} sensitivity source lacks the complete "
                "episode/proposal denominator"
            )
        verifier = _verifier_config_from_runner_config(result.config)
        if common_verifier is None:
            common_verifier = verifier
        elif verifier != common_verifier:
            raise PilotOrchestrationError(
                f"{source_label} sensitivity sources disagree on verifier "
                "configuration"
            )
        replay = replay_rule_sensitivity(
            episode_rows=episodes,
            rule_rows=rules,
            verifier_config=verifier,
            absolute_flow_threshold=float(threshold_value),
        )
        per_run.append(
            {
                "run_id": spec.run_id,
                "environment_seed": spec.environment_seed,
                "episode_count": len(episodes),
                "proposal_count": len(proposals),
                "rule_count": len(rules),
                "episodes_sha256": canonical_sha256(episodes),
                "proposals_sha256": canonical_sha256(proposals),
                "rules_sha256": canonical_sha256(rules),
                "replay": replay,
            }
        )
        source_manifests.append(
            {
                "run_id": spec.run_id,
                "environment_seed": spec.environment_seed,
                "manifest": str(manifest_path),
                "manifest_sha256": verification.manifest_sha256,
            }
        )
    if common_verifier is None:
        raise PilotOrchestrationError(
            f"{source_label} sensitivity source matrix is empty"
        )

    aggregate_cells: list[dict[str, Any]] = []
    for weight in ALTERNATIVE_SUCCESS_WEIGHTS:
        for definition in OUTCOME_DEFINITIONS:
            cells = [
                cell
                for run in per_run
                for cell in run["replay"]["cells"]
                if cell["alternative_success_weight"] == weight
                and cell["outcome_definition"] == definition
            ]
            if len(cells) != len(expected_specs):
                raise PilotOrchestrationError(
                    "per-run sensitivity replay did not produce the frozen 3x3 grid"
                )
            aggregate_cells.append(
                {
                    "alternative_success_weight": weight,
                    "outcome_definition": definition,
                    "source_run_count": len(cells),
                    "natural_rule_count": sum(
                        int(cell["rule_count"]) for cell in cells
                    ),
                    "ever_active_count": sum(
                        int(cell["ever_active_count"]) for cell in cells
                    ),
                    "retired_count": sum(int(cell["retired_count"]) for cell in cells),
                    "active_exposure_steps": sum(
                        int(cell["active_exposure_steps"]) for cell in cells
                    ),
                }
            )

    selection_path = raw_root / "stage0-calibration" / "stage0_selection.json"
    return {
        "schema_version": PILOT_EXPERIMENT_C_SENSITIVITY_SCHEMA_VERSION,
        "status": "pass",
        "terminal": True,
        "control_kind": "zero-api-offline-rule-sensitivity",
        "provider_calls": 0,
        "descriptive_only": True,
        "effectiveness_gate": False,
        "scientific_evidence": True,
        "absolute_flow_utility_threshold": _json_copy(threshold),
        "verifier_config": common_verifier,
        "alternative_success_weights": list(ALTERNATIVE_SUCCESS_WEIGHTS),
        "outcome_definitions": list(OUTCOME_DEFINITIONS),
        "source_run_count": len(per_run),
        "per_run": per_run,
        "aggregate_cells": aggregate_cells,
        "bindings": {
            "contract_sha256": contract.canonical_hash,
            "git_tag": git_tag,
            "git_commit": git_commit,
            "stage0_selection": str(selection_path),
            "stage0_selection_content_sha256": selection["integrity"]["content_sha256"],
            "stage0_selection_file_sha256": _file_sha256(selection_path),
            "source_stage": source_stage,
            "source_arm": "full",
            "source_manifests": sorted(
                source_manifests,
                key=lambda row: row["run_id"],
            ),
            "source_matrix_sha256": canonical_sha256(
                sorted(source_manifests, key=lambda row: row["run_id"])
            ),
        },
        "claim_boundary": (
            "This zero-provider-call replay is descriptive sensitivity over "
            f"natural {source_label} proposals. It is not an effectiveness "
            "gate and "
            "cannot rescue a failed preregistered Experiment C contrast."
        ),
    }


def _load_verified_experiment_c_sensitivity(
    contract: PilotContract,
    *,
    raw_root: Path,
    paid: GitProvenance | None,
) -> dict[str, Any]:
    path = _experiment_c_sensitivity_path(raw_root)
    value = _read_json(path)
    _verify_bound_payload(
        value,
        contract=contract,
        schema_version=PILOT_EXPERIMENT_C_SENSITIVITY_SCHEMA_VERSION,
        paid=paid,
        artifact_name="Experiment C sensitivity",
    )
    bindings = value["bindings"]
    expected = _build_experiment_c_sensitivity(
        contract,
        raw_root=raw_root,
        git_tag=str(bindings["git_tag"]),
        git_commit=str(bindings["git_commit"]),
    )
    actual_without_integrity = _json_copy(value)
    actual_without_integrity.pop("integrity", None)
    if actual_without_integrity != expected:
        raise PilotOrchestrationError(
            "Experiment C sensitivity differs from sealed Stage-0/full-control "
            "sources"
        )
    return value


def _write_experiment_c_sensitivity(
    contract: PilotContract,
    *,
    raw_root: Path,
    paid: GitProvenance,
) -> Path:
    output = _experiment_c_sensitivity_path(raw_root)
    if output.exists():
        _load_verified_experiment_c_sensitivity(
            contract,
            raw_root=raw_root,
            paid=paid,
        )
        return output
    payload = _build_experiment_c_sensitivity(
        contract,
        raw_root=raw_root,
        git_tag=paid.git_tag,
        git_commit=paid.head_commit,
    )
    _atomic_bound_json(output, _seal_bound_payload(payload))
    _load_verified_experiment_c_sensitivity(
        contract,
        raw_root=raw_root,
        paid=paid,
    )
    return output


def _d_continuation_causal_bindings(
    continuation: Mapping[str, Any],
    branch: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact shared-checkpoint binding for one D treatment cell."""

    common_start = continuation.get("erroneous_forced_active_common_start")
    if not isinstance(common_start, Mapping):
        raise PilotOrchestrationError(
            "Experiment D continuation lacks the common erroneous-rule start"
        )
    api_usage = branch.get("api_usage")
    if not isinstance(api_usage, list):
        raise PilotOrchestrationError("Experiment D branch lacks its completion ledger")
    intervention = branch.get("intervention")
    journal = branch.get("provider_call_journal")
    if not isinstance(intervention, Mapping):
        raise PilotOrchestrationError(
            "Experiment D branch lacks its intervention receipt"
        )
    if not isinstance(journal, Mapping):
        raise PilotOrchestrationError(
            "Experiment D branch lacks its provider-call journal binding"
        )
    treatment = str(branch.get("treatment", ""))
    forced_start_hash = (
        intervention.get("forced_active_start_hash")
        if treatment in {"erroneous-verified", "erroneous-unverified"}
        else None
    )
    return {
        "kind": "continuation",
        "checkpoint_hash": continuation["checkpoint_hash"],
        "prefix_hash": continuation["prefix_hash"],
        "shock_schedule_hash": branch["shock_schedule_hash"],
        "pre_generated_rng_hashes": list(continuation["pre_generated_rng_hashes"]),
        "rng_schedule_binding": dict(continuation["rng_schedule_binding"]),
        "shared_result_hash": continuation["result_hash"],
        "matched_replay_equal": continuation["matched_replay_equal"],
        "branch_treatment": treatment,
        "branch_rng_pre_step_hashes": list(branch["rng_pre_step_hashes"]),
        "branch_action_completions": len(api_usage),
        "branch_provider_call_journal": dict(journal),
        "proposal_counters_before": dict(branch["proposal_counters_before"]),
        "proposal_counters_after": dict(branch["proposal_counters_after"]),
        "proposals_frozen": branch["freeze_proposals"],
        "focal_agent_id": continuation["focal_agent_id"],
        "wrong_context_source_agent_id": continuation["wrong_context_source_agent_id"],
        "memory_pulse_contract": dict(continuation["memory_pulse_contract"]),
        "branch_intervention_pulse_only": intervention["pulse_only"],
        "branch_memory_pulse_binding": intervention["memory_pulse_binding"],
        "action_grid": dict(continuation["action_grid"]),
        "error_common_start_equal": common_start["equal"],
        "error_common_start_hash": common_start["forced_active_start_hash"],
        "branch_forced_active_start_hash": forced_start_hash,
    }


def _d_narrative_causal_bindings(
    narratives: Mapping[str, Any],
    branch: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact shared-checkpoint binding for one D narrative cell."""

    api_usage = branch.get("api_usage")
    narrative = branch.get("narrative")
    journal = branch.get("provider_call_journal")
    if (
        not isinstance(api_usage, list)
        or not isinstance(narrative, Mapping)
        or not isinstance(journal, Mapping)
    ):
        raise PilotOrchestrationError(
            "Experiment D narrative branch lacks its execution receipts"
        )
    return {
        "kind": "narrative",
        "checkpoint_hash": narratives["checkpoint_hash"],
        "prefix_hash": narratives["prefix_hash"],
        "shock_schedule_hash": narratives["shock_schedule_hash"],
        "pre_generated_rng_hashes": list(narratives["pre_generated_rng_hashes"]),
        "rng_schedule_binding": dict(narratives["rng_schedule_binding"]),
        "shared_result_hash": narratives["result_hash"],
        "fixture_hash": narratives["fixture_hash"],
        "branch_narrative_id": narrative["narrative_id"],
        "branch_text_hash": narrative["text_hash"],
        "branch_rng_pre_step_hashes": list(branch["rng_pre_step_hashes"]),
        "branch_action_completions": len(api_usage),
        "branch_provider_call_journal": dict(journal),
        "proposal_counters_before": dict(branch["proposal_counters_before"]),
        "proposal_counters_after": dict(branch["proposal_counters_after"]),
        "proposals_frozen": branch["freeze_proposals"],
        "focal_agent_id": narratives["focal_agent_id"],
        "narrative_pulse_contract": dict(
            narratives["narrative_pulse_contract"]
        ),
        "branch_narrative_pulse_only": narrative["pulse_only"],
        "action_grid": dict(narratives["action_grid"]),
    }


def _d_group_projection(
    contract: PilotContract,
    representative: PilotRunSpec,
    *,
    raw_root: Path,
    paid: GitProvenance | None = None,
) -> RunProjection:
    """Reserve one seed's shared prefix, seven D branches, and four narratives."""

    normal = projection_from_preflight(
        contract,
        representative,
        raw_root=raw_root,
        paid=paid,
    )
    calls = {"action": 4 * 6 * (1 + len(DEFAULT_TREATMENTS) + len(DEFAULT_NARRATIVES))}
    calls["semantic"] = 4 * 2  # prefix proposals at outcomes 3 and 6 only
    source = _read_json(Path(normal.basis["source"]))["projection"]
    by_kind: dict[str, Mapping[str, Any]] = {}
    for key, row in source.items():
        by_kind[str(key).rpartition("::")[2]] = row
    totals = {
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        "cost_usd": 0.0,
    }
    for kind, count in calls.items():
        if kind not in by_kind:
            raise PilotOrchestrationError(f"Experiment D preflight lacks {kind!r} p95")
        reserved = by_kind[kind]["reserved_p95"]
        for field in totals:
            totals[field] += float(reserved[field]) * count
    group_id = (
        f"{contract.contract_id}--experiment-d--gpt52_main--"
        f"checkpoint-group--s{representative.environment_seed}"
    )
    return RunProjection(
        run_id=group_id,
        stage_bucket=representative.budget_bucket,
        cost_usd=totals["cost_usd"],
        completions=sum(calls.values()),
        storage_bytes=80_000_000,
        basis={
            "method": "shared-checkpoint-preflight-p95-times-1.25",
            "source": normal.basis["source"],
            "calls_by_kind": calls,
            "run_call_limit": sum(calls.values()),
            "hosted_completion_cap_counted": True,
            "prompt_tokens": math.ceil(totals["prompt_tokens"]),
            "completion_tokens": math.ceil(totals["completion_tokens"]),
            "total_tokens": math.ceil(totals["total_tokens"]),
            "cost_usd": totals["cost_usd"],
        },
    )


def _execute_d_seed(
    contract: PilotContract,
    specs: Sequence[PilotRunSpec],
    *,
    raw_root: Path,
    paid: GitProvenance | None,
    diagnostic: bool,
    budget_ledger: PilotBudgetLedger,
    run_ledger: PilotRunLedger,
    verify_bound_inputs: bool = False,
) -> None:
    """Execute all eleven registered D cells from one seed/checkpoint."""

    if not specs:
        return
    frozen_narratives = {
        narrative_id: row["text"] for narrative_id, row in contract.narratives.items()
    }
    d_contract = contract.stop_go["experiment_d"]
    if frozen_narratives != DEFAULT_NARRATIVES or d_contract[
        "narrative_fixture_hash"
    ] != canonical_sha256(DEFAULT_NARRATIVES):
        raise PilotOrchestrationError(
            "Experiment D runtime narrative fixtures differ from the contract"
        )
    seed = specs[0].environment_seed
    if any(spec.environment_seed != seed for spec in specs):
        raise ValueError("D group must contain exactly one environment seed")
    if all(run_ledger.is_terminal(spec.run_id) for spec in specs):
        return
    representative = next(
        (spec for spec in specs if spec.arm_id == "matched-a"),
        specs[0],
    )
    projection = (
        RunProjection(
            run_id=f"development-fake-d-s{seed}",
            stage_bucket="core",
            cost_usd=0.0,
            completions=0,
            storage_bytes=80_000_000,
            basis={
                "method": "diagnostic-fixed-ceiling",
                "run_call_limit": 400,
                "hosted_completion_cap_counted": False,
                "prompt_tokens": 4_000_000,
                "completion_tokens": 1_000_000,
            },
        )
        if diagnostic
        else _d_group_projection(
            contract,
            representative,
            raw_root=raw_root,
            paid=paid,
        )
    )
    existing = budget_ledger.snapshot()["runs"].get(projection.run_id)
    if existing is not None:
        failure = {
            "error_type": (
                "InterruptedReservation"
                if existing.get("status") == "reserved"
                else "BudgetFinalizedBeforeITT"
            ),
            "message": (
                "D seed already has budget state without a complete ITT group; "
                "no provider redispatch is permitted"
            ),
        }
        if existing.get("status") == "reserved":
            budget_ledger.finalize(
                projection.run_id,
                status="integrity-stopped",
                cost_usd=float(existing["reservation"]["cost_usd"]),
                completions=int(existing["reservation"]["completions"]),
                storage_bytes=int(existing["reservation"]["storage_bytes"]),
                failure={
                    **failure,
                    "accounting_basis": "unreconciled-conservative-reservation",
                },
            )
        for spec in specs:
            if not run_ledger.is_terminal(spec.run_id):
                run_ledger.finalize(
                    spec.run_id,
                    status="integrity-stopped",
                    artifact=None,
                    failure=failure,
                )
        return

    budget = _run_budget_from_projection(projection)
    group_dir = raw_root / "experiment-d" / "checkpoints" / f"s{seed}"
    group_dir.mkdir(parents=True, exist_ok=True)
    d_journal_paths: tuple[Path, ...] = ()
    budget_ledger.reserve(projection)
    try:
        llm = (
            MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=4)
            if diagnostic
            else _provider_for_profile(
                contract.provider_profiles[representative.model_id]
            )
        )
        base = config_for_spec(
            contract,
            representative,
            raw_root=raw_root,
            paid_provenance=paid,
            diagnostic_override=diagnostic,
            verify_bound_inputs=verify_bound_inputs,
            preflight_p95_reservations=(
                {}
                if diagnostic or paid is None or not verify_bound_inputs
                else _runner_p95_reservations(
                    contract,
                    representative.model_id,
                    raw_root=raw_root,
                    paid=paid,
                )
            ),
        )
        action_grid = d_contract["action_grid"]
        if not math.isclose(
            float(base.labor_step),
            float(action_grid["labor_step_hours"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ) or not math.isclose(
            float(base.consumption_step),
            float(action_grid["consumption_step"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise PilotOrchestrationError(
                "Experiment D runner action grid differs from the frozen bins"
            )
        base = replace(
            base,
            run_id=f"{projection.run_id}--prefix",
            freeze_new_proposals_after=6,
            error_rule_mode="none",
            semantic_policy="evidence-grounded",
        )
        branch_map = {
            "matched-a": "matched-a",
            "matched-b": "matched-b",
            "no-memory": "no-memory",
            "shuffled-episodic": "shuffled-episodic",
            "wrong-context": "wrong-context",
            "error-verified": "erroneous-verified",
            "error-unverified": "erroneous-unverified",
        }
        continuation_specs = {
            branch_map[spec.arm_id]: spec
            for spec in specs
            if spec.arm_id in branch_map
        }
        narrative_specs = {
            str(spec.narrative_id): spec
            for spec in specs
            if spec.arm_id == "narrative-content"
        }
        if set(continuation_specs) != set(DEFAULT_TREATMENTS):
            raise PilotOrchestrationError(
                "Experiment D group lacks one or more continuation journal cells"
            )
        if set(narrative_specs) != set(DEFAULT_NARRATIVES):
            raise PilotOrchestrationError(
                "Experiment D group lacks one or more narrative journal cells"
            )

        def journal_target(spec: PilotRunSpec) -> dict[str, Any]:
            return {
                "path": _provider_call_journal_path(
                    group_dir,
                    run_id=spec.run_id,
                    kind="actor",
                ),
                "run_id": spec.run_id,
                "contract_hash": base.pilot_contract_hash,
            }

        d_journal_paths = tuple(
            Path(journal_target(spec)["path"])
            for spec in specs
        )
        checkpoint = build_pilot_checkpoint(
            base,
            llm=llm,
            budget=budget,
            env_config_source=DEFAULT_ENV_CONFIG,
        )
        checkpoint_path = group_dir / "checkpoint.json"
        _atomic_json(checkpoint_path, checkpoint.to_dict())
        continuation = run_pilot_continuations(
            checkpoint,
            llm=llm,
            budget=budget,
            provider_call_journals={
                treatment: journal_target(spec)
                for treatment, spec in continuation_specs.items()
            },
        ).to_dict()
        continuation.update(
            {
                "contract_sha256": contract.canonical_hash,
                "diagnostic_only": diagnostic,
                "scientific_evidence": False if diagnostic else True,
            }
        )
        continuation_path = group_dir / "continuations.json"
        _atomic_json(continuation_path, continuation)
        narratives = run_pilot_narratives(
            checkpoint,
            llm=llm,
            budget=budget,
            provider_call_journals={
                narrative_id: journal_target(spec)
                for narrative_id, spec in narrative_specs.items()
            },
        ).to_dict()
        narratives.update(
            {
                "contract_sha256": contract.canonical_hash,
                "diagnostic_only": diagnostic,
                "scientific_evidence": False if diagnostic else True,
            }
        )
        narrative_path = group_dir / "narratives.json"
        _atomic_json(narrative_path, narratives)
        shared_budget_receipt = budget.snapshot().to_dict()
        completed_artifacts: dict[str, Path] = {}
        for spec in specs:
            if spec.arm_id == "narrative-content":
                narrative_branch = narratives["branches"][spec.narrative_id]
                payload = {
                    "metrics": {
                        "narrative": narratives["metrics"][spec.narrative_id],
                    },
                    "gate_evidence": {
                        "semantic_equivalence_within_one_action_bin": narratives[
                            "semantic_equivalence_within_one_action_bin"
                        ],
                        "aligned_vs_opposite_delta": narratives[
                            "aligned_vs_opposite_delta"
                        ],
                        "causal_bindings": _d_narrative_causal_bindings(
                            narratives,
                            narrative_branch,
                        ),
                    },
                    "narrative": {
                        "narrative_id": spec.narrative_id,
                        "branch": narrative_branch,
                        "claim_boundary": narratives["claim_boundary"],
                    },
                    "shared_source": str(narrative_path),
                    "shared_source_sha256": _file_sha256(narrative_path),
                    "completion_receipt": {
                        "branch_action_completions": 4 * 6,
                        "observed_trajectory_action_rows": sum(
                            len(row["decisions"])
                            for row in narrative_branch["trajectory"]
                        ),
                        "shared_budget": shared_budget_receipt,
                        "usage_attribution": (
                            "provider usage is accounted in the shared "
                            "checkpoint-and-branches budget; no synthetic "
                            "per-branch cost allocation is claimed"
                        ),
                    },
                    "provider_call_journal": narrative_branch[
                        "provider_call_journal"
                    ],
                }
            else:
                branch = continuation["branches"][branch_map[spec.arm_id]]
                payload = {
                    "metrics": {
                        "continuation": branch["metrics"],
                        "delta_vs_matched_a": branch["delta_vs_matched_a"],
                    },
                    "gate_evidence": {
                        "matched_replay_equal": continuation["matched_replay_equal"],
                        "checkpoint_hash": continuation["checkpoint_hash"],
                        "prefix_hash": continuation["prefix_hash"],
                        "causal_bindings": _d_continuation_causal_bindings(
                            continuation,
                            branch,
                        ),
                    },
                    "shared_source": str(continuation_path),
                    "shared_source_sha256": _file_sha256(continuation_path),
                    "completion_receipt": {
                        "branch_action_completions": 4 * 6,
                        "observed_trajectory_action_rows": sum(
                            len(row["decisions"]) for row in branch["trajectory"]
                        ),
                        "shared_budget": shared_budget_receipt,
                        "usage_attribution": (
                            "provider usage is accounted in the shared "
                            "checkpoint-and-branches budget; no synthetic "
                            "per-branch cost allocation is claimed"
                        ),
                    },
                    "provider_call_journal": branch[
                        "provider_call_journal"
                    ],
                }
            if paid is not None and not diagnostic:
                artifact_path = write_terminal_summary(
                    raw_root / "experiment-d" / "summaries" / f"{spec.run_id}.json",
                    contract=contract,
                    run_spec=spec,
                    resolved_git_commit=paid.head_commit,
                    git_tag=paid.git_tag,
                    payload=payload,
                    scientific_evidence=True,
                    diagnostic_only=False,
                    evidence_scope=CURRENT_SCIENTIFIC_SCOPE,
                )
            else:
                artifact_path = (
                    raw_root
                    / "experiment-d"
                    / "diagnostic_summaries"
                    / f"{spec.run_id}.json"
                )
                _atomic_json(
                    artifact_path,
                    {
                        "contract_sha256": contract.canonical_hash,
                        "run_spec": spec.to_dict(),
                        "payload": payload,
                        "diagnostic_only": True,
                        "scientific_evidence": False,
                    },
                )
            completed_artifacts[spec.run_id] = artifact_path
        actual = _actual_budget_values(
            group_dir,
            budget,
            additional_paths=(
                *tuple(completed_artifacts.values()),
                *tuple(path for path in d_journal_paths if path.exists()),
            ),
            completion_cap_counted=bool(
                projection.basis.get(
                    "hosted_completion_cap_counted",
                    True,
                )
            ),
        )
        reconciliation_failure_receipt: Path | None = None
        try:
            budget_ledger.finalize(
                projection.run_id,
                status="complete",
                **actual,
            )
            ledger_status = "complete"
            ledger_failure = None
        except PilotBudgetError as exc:
            ledger_status = "integrity-stopped"
            ledger_failure = {
                "error_type": "BudgetReconciliationOverage",
                "message": str(exc),
                "seed": seed,
                "observed_actual": dict(actual),
                "reservation": projection.to_dict(),
                "run_budget_snapshot": budget.snapshot().to_dict(),
            }
            reconciliation_failure_receipt = _write_execution_failure_receipt(
                group_dir / "failure_receipt",
                scope=(
                    "finevo-pilot/experiment-d/"
                    "shared-checkpoint-group-budget-reconciliation"
                ),
                error=exc,
                contract=contract,
                projection=projection,
                budget=budget,
                specs=specs,
                paid=paid,
                diagnostic=diagnostic,
            )
            actual = _actual_budget_values(
                group_dir,
                budget,
                additional_paths=(
                    *tuple(completed_artifacts.values()),
                    *tuple(path for path in d_journal_paths if path.exists()),
                ),
                completion_cap_counted=bool(
                    projection.basis.get(
                        "hosted_completion_cap_counted",
                        True,
                    )
                ),
            )
            budget_ledger.finalize(
                projection.run_id,
                status=ledger_status,
                failure=ledger_failure,
                **actual,
            )
        for spec in specs:
            run_ledger.finalize(
                spec.run_id,
                status=ledger_status,
                artifact=(
                    str(completed_artifacts[spec.run_id])
                    if ledger_status == "complete"
                    else str(reconciliation_failure_receipt)
                    if reconciliation_failure_receipt is not None
                    else None
                ),
                failure=ledger_failure,
            )
    except Exception as exc:
        failure = _exception_failure(exc, seed=seed)
        failure_receipt = _write_execution_failure_receipt(
            group_dir / "failure_receipt",
            scope="finevo-pilot/experiment-d/shared-checkpoint-group",
            error=exc,
            contract=contract,
            projection=projection,
            budget=budget,
            specs=specs,
            paid=paid,
            diagnostic=diagnostic,
        )
        actual = _actual_budget_values(
            group_dir,
            budget,
            additional_paths=tuple(
                path for path in d_journal_paths if path.exists()
            ),
            completion_cap_counted=bool(
                projection.basis.get(
                    "hosted_completion_cap_counted",
                    True,
                )
            ),
        )
        budget_status = "failed"
        try:
            budget_ledger.finalize(
                projection.run_id,
                status=budget_status,
                failure=failure,
                **actual,
            )
        except PilotBudgetError as budget_exc:
            budget_status = "integrity-stopped"
            failure = {
                **failure,
                "budget_reconciliation": {
                    "error_type": "BudgetReconciliationOverage",
                    "message": str(budget_exc),
                    "observed_actual": dict(actual),
                    "reservation": projection.to_dict(),
                    "run_budget_snapshot": budget.snapshot().to_dict(),
                },
            }
            budget_ledger.finalize(
                projection.run_id,
                status=budget_status,
                **actual,
                failure=failure,
            )
        for spec in specs:
            if not run_ledger.is_terminal(spec.run_id):
                run_ledger.finalize(
                    spec.run_id,
                    status=budget_status,
                    artifact=str(failure_receipt),
                    failure=failure,
                )


def _finalize_budget_safely(
    ledger: PilotBudgetLedger,
    projection: RunProjection,
    *,
    run_dir: Path,
    budget: RunBudget,
    status: str,
    failure: Mapping[str, Any] | None = None,
    additional_paths: Sequence[Path] = (),
) -> tuple[str, Mapping[str, Any] | None, Mapping[str, Any]]:
    actual = _actual_budget_values(
        run_dir,
        budget,
        additional_paths=additional_paths,
        completion_cap_counted=bool(
            projection.basis.get(
                "hosted_completion_cap_counted",
                True,
            )
        ),
    )
    try:
        ledger.finalize(
            projection.run_id,
            status=status,
            failure=failure,
            **actual,
        )
        return status, failure, actual
    except PilotBudgetError as exc:
        # Preserve the exact observed usage, including an overage.  Replacing
        # it with the lower reservation would make the global ledger look safer
        # than the provider and filesystem evidence actually show.
        row = ledger.snapshot()["runs"][projection.run_id]
        reservation = row["reservation"]
        integrity_failure = {
            "error_type": "BudgetReconciliationOverage",
            "message": str(exc),
            "requested_terminal_status": status,
            "observed_actual": dict(actual),
            "reservation": {
                "cost_usd": float(reservation["cost_usd"]),
                "completions": int(reservation["completions"]),
                "storage_bytes": int(reservation["storage_bytes"]),
            },
            "run_budget_snapshot": budget.snapshot().to_dict(),
            "original_failure": dict(failure) if failure else None,
        }
        ledger.finalize(
            projection.run_id,
            status="integrity-stopped",
            cost_usd=float(actual["cost_usd"]),
            completions=int(actual["completions"]),
            storage_bytes=int(actual["storage_bytes"]),
            failure=integrity_failure,
        )
        return "integrity-stopped", integrity_failure, actual


def _recover_or_stop_interrupted_reservation(
    budget_ledger: PilotBudgetLedger,
    run_ledger: PilotRunLedger,
    spec: PilotRunSpec,
) -> bool:
    """Never silently re-dispatch a run whose reservation survived a crash."""

    existing = budget_ledger.snapshot()["runs"].get(spec.run_id)
    if existing is None:
        return False
    failure = {
        "error_type": (
            "InterruptedReservation"
            if existing.get("status") == "reserved"
            else "BudgetFinalizedBeforeITT"
        ),
        "message": (
            "a prior process created budget state without a terminal ITT cell; "
            "the cell is retained and is not redispatched"
        ),
    }
    if existing.get("status") == "reserved":
        reservation = existing["reservation"]
        failure["accounting_basis"] = "unreconciled-conservative-reservation"
        budget_ledger.finalize(
            spec.run_id,
            status="integrity-stopped",
            cost_usd=float(reservation["cost_usd"]),
            completions=int(reservation["completions"]),
            storage_bytes=int(reservation["storage_bytes"]),
            failure=failure,
        )
    run_ledger.finalize(
        spec.run_id,
        status="integrity-stopped",
        artifact=None,
        failure=failure,
    )
    return True


def _assert_projection_matrix_fits(
    ledger: PilotBudgetLedger,
    projections: Sequence[RunProjection],
) -> None:
    """Check the whole remaining matrix before the first stage dispatch."""

    snapshot = ledger.snapshot()
    committed = snapshot["committed_plus_reserved"]
    existing = snapshot["runs"]
    additions = [
        projection for projection in projections if projection.run_id not in existing
    ]
    total_usd = float(committed["cost_usd"]) + sum(
        projection.cost_usd for projection in additions
    )
    total_completions = int(committed["completions"]) + sum(
        projection.completions for projection in additions
    )
    total_storage = int(committed["storage_bytes"]) + sum(
        projection.storage_bytes for projection in additions
    )
    stage_additions: dict[str, float] = {}
    for projection in additions:
        stage_additions[projection.stage_bucket] = (
            stage_additions.get(projection.stage_bucket, 0.0) + projection.cost_usd
        )
    violations: list[str] = []
    caps = ledger.caps
    if total_usd > caps.dispatchable_usd + 1e-12:
        violations.append("dispatchable global USD")
    if total_completions > caps.max_completions:
        violations.append("completion count")
    if total_storage > caps.max_storage_bytes:
        violations.append("storage")
    for stage_bucket, addition in stage_additions.items():
        current = float(committed["stage_cost_usd"][stage_bucket])
        cap = float(caps.stage_usd_caps[stage_bucket])
        if current + addition > cap + 1e-12:
            violations.append(f"{stage_bucket} USD")
    if violations:
        raise PilotBudgetError(
            "complete preregistered stage projection exceeds " + ", ".join(violations)
        )


def _remaining_core_projections(
    contract: PilotContract,
    *,
    raw_root: Path,
    paid: GitProvenance,
    run_ledger: PilotRunLedger,
) -> tuple[RunProjection, ...]:
    """Project every still-dispatchable A--D unit under the shared core cap.

    Experiment C's admission rows consume storage but zero provider calls.
    Experiment D is represented exactly once per seed because its eleven ITT
    cells share one prefix/continuation provider budget.
    """

    projections: list[RunProjection] = []
    for stage_id in ("experiment-a", "experiment-b", "experiment-c"):
        for spec in contract.expand(stage=stage_id):
            if run_ledger.is_terminal(spec.run_id):
                continue
            if spec.execution_mode == "offline_candidate_admission":
                projections.append(
                    RunProjection(
                        run_id=spec.run_id,
                        stage_bucket="core",
                        cost_usd=0.0,
                        completions=0,
                        storage_bytes=2_000_000,
                        basis={"method": "offline-zero-provider-call"},
                    )
                )
            else:
                projections.append(
                    projection_from_preflight(
                        contract,
                        spec,
                        raw_root=raw_root,
                        paid=paid,
                    )
                )

    d_specs = tuple(contract.expand(stage="experiment-d"))
    for seed in sorted({spec.environment_seed for spec in d_specs}):
        seed_specs = tuple(spec for spec in d_specs if spec.environment_seed == seed)
        if all(run_ledger.is_terminal(spec.run_id) for spec in seed_specs):
            continue
        representative = next(spec for spec in seed_specs if spec.arm_id == "matched-a")
        projections.append(
            _d_group_projection(
                contract,
                representative,
                raw_root=raw_root,
                paid=paid,
            )
        )
    return tuple(projections)


def _remaining_scientific_projections(
    contract: PilotContract,
    *,
    raw_root: Path,
    paid: GitProvenance,
    run_ledger: PilotRunLedger,
) -> tuple[RunProjection, ...]:
    """Project every still-dispatchable scientific unit before Stage 0.

    This is the post-capability, pre-scientific-spend gate.  It includes the
    complete remaining Stage-0 matrix, A--D (with one shared D projection per
    seed and zero-call offline C rows), and only cross-model cells whose model
    has a completed capability/preflight cell.  A nonterminal cross-model cell
    without a capability pass is an integrity error rather than something that
    may be silently omitted from the hard global caps.
    """

    projections: list[RunProjection] = []
    for spec in contract.expand(stage="stage0-calibration"):
        if run_ledger.is_terminal(spec.run_id):
            continue
        projections.append(
            projection_from_preflight(
                contract,
                spec,
                raw_root=raw_root,
                paid=paid,
            )
        )

    projections.extend(
        _remaining_core_projections(
            contract,
            raw_root=raw_root,
            paid=paid,
            run_ledger=run_ledger,
        )
    )

    for cross_stage in _cross_model_science_stage_ids(contract):
        for spec in contract.expand(stage=cross_stage):
            if run_ledger.is_terminal(spec.run_id):
                continue
            preflight_stage = _preflight_stage_for_model(
                contract,
                spec.model_id,
            )
            preflight_specs = contract.expand(
                stage=preflight_stage,
                model=spec.model_id,
            )
            if (
                len(preflight_specs) != 1
                or run_ledger.status(preflight_specs[0].run_id) != "complete"
            ):
                raise PilotOrchestrationError(
                    "nonterminal cross-model cell lacks a completed "
                    f"capability/preflight gate: {spec.model_id}"
                )
            projections.append(
                projection_from_preflight(
                    contract,
                    spec,
                    raw_root=raw_root,
                    paid=paid,
                )
            )
    return tuple(projections)


def _execute_stage_locked(
    *,
    contract_path: str | Path,
    stage_id: str,
    resume: bool,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute one frozen real stage after all release and prerequisite gates.

    This function is intentionally side-effectful only beneath ``raw_root``.
    It does not modify the contract, source tree, tag, or evidence release.
    """

    contract = load_pilot_contract(contract_path)
    if stage_id not in contract.stage_ids:
        raise PilotOrchestrationError(f"unknown frozen stage: {stage_id}")
    stage = contract.stage(stage_id)
    if not stage.enabled:
        raise PilotOrchestrationError(f"stage {stage_id} is disabled")
    root = Path(raw_root).resolve()
    repository = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    paid = verify_paid_provenance(
        contract,
        repo_root=repository,
        scientific_launch_input_path=(
            root / "scientific_launch_input.json"
            if contract.schema_version.endswith("-v2")
            else None
        ),
    )
    _persist_release_attestation(root, paid)
    specs = contract.expand(stage=stage_id)
    run_ledger = PilotRunLedger(
        root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=contract.schema_version.endswith("-v2"),
    )
    # Register the complete preregistered denominator before the first stage.
    # Later no-go propagation therefore terminalizes future cells instead of
    # making them disappear.
    run_ledger.register(contract.expand())
    if not resume and any(
        run_ledger.status(spec.run_id) != "scheduled" for spec in specs
    ):
        raise PilotOrchestrationError(
            "stage already has terminal cells; use --resume for idempotent recovery"
        )
    budget_ledger = PilotBudgetLedger(
        root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=_budget_caps(contract),
        tamper_evident=contract.schema_version.endswith("-v2"),
    )
    try:
        verified_prerequisite_go = _assert_prerequisites(
            contract,
            stage_id,
            raw_root=root,
            paid=paid,
            ledger=run_ledger,
        )
    except Exception as exc:
        failure = {
            "error_type": "PrerequisiteNoGo",
            "cause_type": type(exc).__name__,
            "message": str(exc),
            "source_stage": stage_id,
        }
        run_ledger.stop_pending(
            specs,
            status="integrity-stopped",
            failure=failure,
        )
        _propagate_stage_no_go(
            contract,
            source_stage=stage_id,
            ledger=run_ledger,
            failure=failure,
        )
        receipt = _write_stage_receipt(
            contract,
            stage_id,
            raw_root=root,
            ledger=run_ledger,
            status="prerequisite-no-go",
            failure=failure,
            paid=paid,
        )
        raise PilotOrchestrationError(
            f"stage prerequisites failed; receipt={receipt}"
        ) from exc

    catalog_go: set[str] = set()
    for model_id in contract.models_for_stage(stage_id):
        profile = contract.provider_profiles[model_id]
        if profile.transport == "diagnostic":
            catalog_go.add(model_id)
            continue
        model_specs = tuple(spec for spec in specs if spec.model_id == model_id)
        if all(run_ledger.is_terminal(spec.run_id) for spec in model_specs):
            continue
        catalog_path = root / stage_id / "provider_catalog" / f"{model_id}.json"
        try:
            if catalog_path.exists():
                if not resume:
                    raise PilotOrchestrationError(
                        "catalog receipt already exists; use --resume to "
                        f"reuse its exact verified bytes for {model_id}"
                    )
                catalog_receipt = verify_provider_catalog_receipt(
                    _read_json(catalog_path),
                    contract_hash=contract.canonical_hash,
                )
                row_ids = [
                    (
                        row.get("profile_id")
                        if contract.schema_version.endswith("-v2")
                        else row.get("profile_id", row.get("model_id"))
                    )
                    for row in catalog_receipt["rows"]
                    if isinstance(row, Mapping)
                ]
                if row_ids != [model_id]:
                    raise PilotOrchestrationError(
                        "resumed catalog receipt does not bind exactly one "
                        f"requested profile: {model_id}"
                    )
            else:
                catalog_receipt = validate_live_provider_catalog(
                    contract,
                    model_ids=(model_id,),
                )
                if (
                    not contract.schema_version.endswith("-v2")
                    and "receipt_sha256" not in catalog_receipt
                ):
                    # Read-only V1 compatibility for callers/tests that
                    # implement the pre-self-hash catalog interface. This is
                    # allowed only for a fresh in-memory observation; resumed
                    # receipts are always verified exactly as persisted.
                    catalog_receipt = dict(catalog_receipt)
                    catalog_receipt["receipt_sha256"] = canonical_sha256(
                        catalog_receipt
                    )
                verify_provider_catalog_receipt(
                    catalog_receipt,
                    contract_hash=contract.canonical_hash,
                )
                _atomic_json_no_overwrite(catalog_path, catalog_receipt)
            catalog_go.add(model_id)
        except Exception as exc:
            failure = {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "model_id": model_id,
                "paid_completions": 0,
            }
            no_go_receipt = {
                "schema_version": PROVIDER_CATALOG_RECEIPT_SCHEMA_VERSION,
                "captured_at": _utc_now(),
                "contract_sha256": contract.canonical_hash,
                "status": "no-go",
                "paid_completions": 0,
                "model_id": model_id,
                "rows": [],
                "failure": failure,
            }
            no_go_receipt["receipt_sha256"] = canonical_sha256(no_go_receipt)
            if not catalog_path.exists():
                _atomic_json_no_overwrite(catalog_path, no_go_receipt)
            terminal_status = (
                "capability-no-go"
                if (
                    _is_capability_stage(contract, stage_id)
                    or stage_id in _cross_model_science_stage_ids(contract)
                )
                else "integrity-stopped"
            )
            for spec in model_specs:
                if not run_ledger.is_terminal(spec.run_id):
                    run_ledger.finalize(
                        spec.run_id,
                        status=terminal_status,
                        artifact=str(catalog_path),
                        failure=failure,
                    )

    # Recompute the complete prerequisite semantics after the zero-completion
    # catalog checks and immediately before any reservation/provider dispatch.
    # The exclusive stage lock prevents a concurrent writer, while this second
    # pass detects local artifact drift during the catalog interval.
    verified_prerequisite_go = _assert_prerequisites(
        contract,
        stage_id,
        raw_root=root,
        paid=paid,
        ledger=run_ledger,
    )
    gate_model_go: set[str] | None = None
    if contract.schema_version.endswith("-v2"):
        target_models = set(contract.models_for_stage(stage_id))
        for prerequisite in stage.prerequisites:
            if not _is_capability_stage(contract, prerequisite):
                continue
            relevant = target_models & set(contract.models_for_stage(prerequisite))
            if not relevant:
                continue
            passed = set(verified_prerequisite_go.get(prerequisite, frozenset()))
            eligible = relevant & passed
            gate_model_go = (
                eligible if gate_model_go is None else gate_model_go & eligible
            )
    elif stage_id == "cross-model-sentinels":
        capability_receipt = _read_json(
            _stage_receipt_path(root, "capability-preflight")
        )
        gate_model_go = {
            str(model_id) for model_id in capability_receipt.get("go_models", [])
        }
    if gate_model_go is not None:
        gate_model_go &= catalog_go
    full_scientific_projection = stage_id == "stage0-calibration"
    try:
        if full_scientific_projection:
            projections = list(
                _remaining_scientific_projections(
                    contract,
                    raw_root=root,
                    paid=paid,
                    run_ledger=run_ledger,
                )
            )
        elif stage_id in CORE_STAGE_IDS:
            projections = list(
                _remaining_core_projections(
                    contract,
                    raw_root=root,
                    paid=paid,
                    run_ledger=run_ledger,
                )
            )
        elif stage_id == "experiment-d":
            projections = [
                _d_group_projection(
                    contract,
                    next(
                        spec
                        for spec in specs
                        if spec.environment_seed == seed and spec.arm_id == "matched-a"
                    ),
                    raw_root=root,
                    paid=paid,
                )
                for seed in sorted({spec.environment_seed for spec in specs})
                if not all(
                    run_ledger.is_terminal(spec.run_id)
                    for spec in specs
                    if spec.environment_seed == seed
                )
            ]
        else:
            projections = []
            for spec in specs:
                if run_ledger.is_terminal(spec.run_id):
                    continue
                if gate_model_go is not None and spec.model_id not in gate_model_go:
                    continue
                if (
                    spec.execution_mode in CAPABILITY_EXECUTION_MODES
                    or spec.execution_mode == "q_ref_resolution"
                ):
                    projection = conservative_projection(contract, spec)
                elif spec.execution_mode == "offline_candidate_admission":
                    projection = RunProjection(
                        run_id=spec.run_id,
                        stage_bucket=spec.budget_bucket,
                        cost_usd=0.0,
                        completions=0,
                        storage_bytes=2_000_000,
                        basis={"method": "offline-zero-provider-call"},
                    )
                else:
                    projection = projection_from_preflight(
                        contract,
                        spec,
                        raw_root=root,
                        paid=paid,
                    )
                projections.append(projection)
        _assert_projection_matrix_fits(budget_ledger, projections)
    except Exception as exc:
        failure = {"error_type": type(exc).__name__, "message": str(exc)}
        stopped_specs = specs
        if full_scientific_projection:
            stopped_specs = tuple(
                spec
                for scientific_stage in _scientific_stage_ids(contract)
                for spec in contract.expand(stage=scientific_stage)
            )
        elif stage_id in CORE_STAGE_IDS:
            stopped_specs = tuple(
                spec
                for core_stage in CORE_STAGE_IDS
                for spec in contract.expand(stage=core_stage)
            )
        terminal_status = (
            "budget-stopped"
            if isinstance(exc, PilotBudgetError)
            else "integrity-stopped"
        )
        projection_scope = (
            "all-remaining-stage0-a-b-c-d-and-capability-eligible-cross"
            if full_scientific_projection
            else (
                "all-remaining-a-b-c-d"
                if stage_id in CORE_STAGE_IDS
                else "current-stage"
            )
        )
        run_ledger.stop_pending(
            stopped_specs,
            status=terminal_status,
            failure={
                **failure,
                "projection_scope": projection_scope,
            },
        )
        if not full_scientific_projection:
            _propagate_stage_no_go(
                contract,
                source_stage=stage_id,
                ledger=run_ledger,
                failure={
                    **failure,
                    "error_type": "ProjectionNoGo",
                },
                status=terminal_status,
            )
        receipt = _write_stage_receipt(
            contract,
            stage_id,
            raw_root=root,
            ledger=run_ledger,
            status=terminal_status,
            failure={**failure, "projection_scope": projection_scope},
            paid=paid,
        )
        raise PilotOrchestrationError(
            f"full stage projection failed before dispatch; receipt={receipt}"
        ) from exc

    if stage_id == "experiment-d":
        for seed in sorted({spec.environment_seed for spec in specs}):
            group = tuple(spec for spec in specs if spec.environment_seed == seed)
            if contract.schema_version.endswith("-v2"):
                _assert_prerequisites(
                    contract,
                    stage_id,
                    raw_root=root,
                    paid=paid,
                    ledger=run_ledger,
                )
            _execute_d_seed(
                contract,
                group,
                raw_root=root,
                paid=paid,
                diagnostic=False,
                budget_ledger=budget_ledger,
                run_ledger=run_ledger,
                verify_bound_inputs=True,
            )
        failed = any(run_ledger.status(spec.run_id) != "complete" for spec in specs)
        receipt = _write_stage_receipt(
            contract,
            stage_id,
            raw_root=root,
            ledger=run_ledger,
            status="complete-with-no-go" if failed else "complete",
            artifacts={"checkpoint_root": str(root / stage_id / "checkpoints")},
            paid=paid,
        )
        receipt_value = _read_json(receipt)
        if receipt_value["execution_progression_go"] is not True:
            _propagate_stage_no_go(
                contract,
                source_stage=stage_id,
                ledger=run_ledger,
                failure={
                    "error_type": "StageExecutionNoGo",
                    "message": (
                        "Experiment D contains a budget or integrity hard stop"
                    ),
                },
            )
        return receipt_value

    go_models: list[str] = []
    stage_artifacts: dict[str, Any] = {}
    stop_stage = False
    for spec in specs:
        if run_ledger.is_terminal(spec.run_id):
            if _is_capability_stage(contract, stage_id):
                gate_path = (
                    root / spec.stage_id / "runs" / spec.run_id / "gate_receipt.json"
                )
                if gate_path.exists() and _read_json(gate_path).get("go") is True:
                    go_models.append(spec.model_id)
            continue
        if gate_model_go is not None and spec.model_id not in gate_model_go:
            run_ledger.finalize(
                spec.run_id,
                status="capability-no-go",
                artifact=None,
                failure={
                    "error_type": "CapabilityOrInterfaceNoGo",
                    "message": (
                        "model did not pass its fixed capability/interface gate; "
                        "cell remains in the ITT denominator without dispatch"
                    ),
                },
            )
            continue
        if stop_stage:
            run_ledger.finalize(
                spec.run_id,
                status="budget-stopped",
                artifact=None,
                failure={
                    "error_type": "StageStopped",
                    "message": "an earlier budget/integrity failure stopped this stage",
                },
            )
            continue
        if _recover_or_stop_interrupted_reservation(
            budget_ledger,
            run_ledger,
            spec,
        ):
            stop_stage = True
            continue

        projection = (
            conservative_projection(contract, spec)
            if (
                spec.execution_mode in CAPABILITY_EXECUTION_MODES
                or spec.execution_mode == "q_ref_resolution"
            )
            else (
                RunProjection(
                    run_id=spec.run_id,
                    stage_bucket=spec.budget_bucket,
                    cost_usd=0.0,
                    completions=0,
                    storage_bytes=2_000_000,
                    basis={"method": "offline-zero-provider-call"},
                )
                if spec.execution_mode == "offline_candidate_admission"
                else projection_from_preflight(
                    contract,
                    spec,
                    raw_root=root,
                    paid=paid,
                )
            )
        )
        run_dir = root / spec.stage_id / "runs" / spec.run_id
        budget: RunBudget | None = None
        try:
            if contract.schema_version.endswith("-v2"):
                _assert_prerequisites(
                    contract,
                    stage_id,
                    raw_root=root,
                    paid=paid,
                    ledger=run_ledger,
                )
            budget_ledger.reserve(projection)
            # Construct the budget at the caller boundary so provider
            # construction, dispatch, parsing, and artifact exceptions all
            # preserve the same partially-accounted object for reconciliation.
            budget = _run_budget_from_projection(projection)
            if spec.execution_mode in {
                "capability_probe",
                "closed_loop_preflight",
            }:
                status, artifact, _, gate = _execute_capability_preflight(
                    contract,
                    spec,
                    raw_root=root,
                    paid=paid,
                    projection=projection,
                    budget=budget,
                )
                if gate["go"]:
                    go_models.append(spec.model_id)
                gate_paths = [Path(artifact)]
                gate_journal = _provider_call_journal_path(
                    run_dir,
                    run_id=spec.run_id,
                    kind="preflight",
                )
                if gate_journal.exists():
                    gate_paths.append(gate_journal)
                budget_status, budget_failure, _ = _finalize_budget_safely(
                    budget_ledger,
                    projection,
                    run_dir=run_dir,
                    budget=budget,
                    status="complete",
                    additional_paths=tuple(gate_paths),
                )
                terminal_status = (
                    status if budget_status == "complete" else budget_status
                )
                run_ledger.finalize(
                    spec.run_id,
                    status=terminal_status,
                    artifact=(
                        str(artifact)
                        if terminal_status in {"complete", "capability-no-go"}
                        else None
                    ),
                    failure=(
                        budget_failure
                        if budget_status != "complete"
                        else (
                            None
                            if terminal_status == "complete"
                            else {
                                "error_type": "CapabilityOrInterfaceNoGo",
                                "message": str(gate["reason"]),
                            }
                        )
                    ),
                )
            elif spec.execution_mode == "q_ref_resolution":
                artifact, _, resolution = _execute_q_ref(
                    contract,
                    spec,
                    raw_root=root,
                    paid=paid,
                    projection=projection,
                    budget=budget,
                )
                budget_status, budget_failure, _ = _finalize_budget_safely(
                    budget_ledger,
                    projection,
                    run_dir=run_dir,
                    budget=budget,
                    status="complete",
                    additional_paths=(
                        Path(artifact),
                        root / spec.stage_id / "q_ref_resolution.json",
                    ),
                )
                run_ledger.finalize(
                    spec.run_id,
                    status=budget_status,
                    artifact=str(artifact) if budget_status == "complete" else None,
                    failure=budget_failure,
                )
                if budget_status == "complete":
                    stage_artifacts["q_ref_resolution"] = str(
                        root / spec.stage_id / "q_ref_resolution.json"
                    )
                    stage_artifacts["q_ref_source_manifest_sha256"] = resolution[
                        "bindings"
                    ]["source_manifest_sha256"]
            elif spec.execution_mode == "offline_candidate_admission":
                artifact = _offline_candidate_admission(
                    contract,
                    spec,
                    raw_root=root,
                    diagnostic=False,
                    paid=paid,
                )
                budget_status, budget_failure, _ = _finalize_budget_safely(
                    budget_ledger,
                    projection,
                    run_dir=run_dir,
                    budget=budget,
                    status="complete",
                    additional_paths=(Path(artifact),),
                )
                run_ledger.finalize(
                    spec.run_id,
                    status=budget_status,
                    artifact=str(artifact) if budget_status == "complete" else None,
                    failure=budget_failure,
                )
            elif spec.execution_mode in {"actor_run", "matched_duplicate"}:
                manifest, _, _ = _execute_actor_run(
                    contract,
                    spec,
                    raw_root=root,
                    paid=paid,
                    projection=projection,
                    budget=budget,
                )
                budget_status, budget_failure, _ = _finalize_budget_safely(
                    budget_ledger,
                    projection,
                    run_dir=run_dir,
                    budget=budget,
                    status="complete",
                    additional_paths=(
                        Path(manifest),
                        _provider_call_journal_path(
                            run_dir,
                            run_id=spec.run_id,
                            kind="actor",
                        ),
                    ),
                )
                run_ledger.finalize(
                    spec.run_id,
                    status=budget_status,
                    artifact=str(manifest) if budget_status == "complete" else None,
                    failure=budget_failure,
                )
            else:
                raise PilotOrchestrationError(
                    f"stage executor is not implemented for {spec.execution_mode!r}"
                )
        except PilotBudgetError as exc:
            failure = _exception_failure(exc)
            terminal_status = "budget-stopped"
            terminal_failure: Mapping[str, Any] = failure
            failure_receipt: Path | None = None
            if budget is not None:
                failure_receipt = _write_execution_failure_receipt(
                    run_dir / "failure_receipt",
                    scope=(f"finevo-pilot/{spec.stage_id}/" f"{spec.execution_mode}"),
                    error=exc,
                    contract=contract,
                    projection=projection,
                    budget=budget,
                    specs=(spec,),
                    paid=paid,
                    diagnostic=False,
                )
                terminal_status, reconciled, _ = _finalize_budget_safely(
                    budget_ledger,
                    projection,
                    run_dir=run_dir,
                    budget=budget,
                    status="budget-stopped",
                    failure=failure,
                )
                terminal_failure = reconciled or failure
            run_ledger.finalize(
                spec.run_id,
                status=terminal_status,
                artifact=(
                    str(failure_receipt) if failure_receipt is not None else None
                ),
                failure=terminal_failure,
            )
            stop_stage = True
        except Exception as exc:
            failure = _exception_failure(exc)
            failure_receipt = None
            if budget is not None:
                failure_receipt = _write_execution_failure_receipt(
                    run_dir / "failure_receipt",
                    scope=(f"finevo-pilot/{spec.stage_id}/" f"{spec.execution_mode}"),
                    error=exc,
                    contract=contract,
                    projection=projection,
                    budget=budget,
                    specs=(spec,),
                    paid=paid,
                    diagnostic=False,
                )
                budget_status, budget_failure, _ = _finalize_budget_safely(
                    budget_ledger,
                    projection,
                    run_dir=run_dir,
                    budget=budget,
                    status="failed",
                    failure=failure,
                )
            else:
                budget_status, budget_failure = "failed", failure
            run_ledger.finalize(
                spec.run_id,
                status=budget_status,
                artifact=(
                    str(failure_receipt) if failure_receipt is not None else None
                ),
                failure=budget_failure or failure,
            )

    if stage_id == "stage0-calibration":
        try:
            selection = _select_stage0(
                contract,
                raw_root=root,
                specs=specs,
                ledger=run_ledger,
                paid=paid,
            )
            stage_artifacts["stage0_selection"] = str(selection)
        except Exception as exc:
            stage_artifacts["stage0_selection_failure"] = {
                "error_type": type(exc).__name__,
                "message": str(exc),
            }

    if stage_id == "experiment-c":
        try:
            sensitivity = _write_experiment_c_sensitivity(
                contract,
                raw_root=root,
                paid=paid,
            )
            sensitivity_value = _read_json(sensitivity)
            stage_artifacts["zero_api_rule_sensitivity"] = {
                "path": str(sensitivity),
                "content_sha256": sensitivity_value["integrity"]["content_sha256"],
                "file_sha256": _file_sha256(sensitivity),
                "provider_calls": sensitivity_value["provider_calls"],
                "descriptive_only": sensitivity_value["descriptive_only"],
                "effectiveness_gate": sensitivity_value["effectiveness_gate"],
            }
        except Exception as exc:
            stage_artifacts["zero_api_rule_sensitivity_failure"] = {
                "error_type": type(exc).__name__,
                "message": str(exc),
            }

    statuses = [run_ledger.status(spec.run_id) for spec in specs]
    complete = all(status == "complete" for status in statuses)
    if _is_capability_stage(contract, stage_id):
        status = "complete" if complete else "complete-with-no-go"
    elif complete and (
        (stage_id != "stage0-calibration" or "stage0_selection" in stage_artifacts)
        and (
            stage_id != "experiment-c" or "zero_api_rule_sensitivity" in stage_artifacts
        )
    ):
        status = "complete"
    else:
        status = "complete-with-no-go"
    if _is_capability_stage(contract, stage_id):
        _propagate_capability_no_go(
            contract,
            ledger=run_ledger,
            source_stage=stage_id,
        )
    receipt = _write_stage_receipt(
        contract,
        stage_id,
        raw_root=root,
        ledger=run_ledger,
        status=status,
        go_models=go_models,
        artifacts=stage_artifacts,
        paid=paid,
    )
    receipt_value = _read_json(receipt)
    if (
        not _is_capability_stage(contract, stage_id)
        and receipt_value["execution_progression_go"] is not True
    ):
        _propagate_stage_no_go(
            contract,
            source_stage=stage_id,
            ledger=run_ledger,
            failure={
                "error_type": "StageExecutionNoGo",
                "message": (
                    f"{stage_id} contains a budget/integrity hard stop or "
                    "lacks its mandatory pre-science selection"
                ),
            },
        )
    return receipt_value


def execute_stage(
    *,
    contract_path: str | Path,
    stage_id: str,
    resume: bool,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute one real stage while holding the raw-root process lock.

    Acquiring the lock outside :func:`_execute_stage_locked` guarantees that
    run-ledger and budget-ledger construction, all provider dispatch, and both
    finalizations are one cross-process critical section.
    """

    with _exclusive_real_stage_lock(raw_root, stage_id=stage_id):
        return _execute_stage_locked(
            contract_path=contract_path,
            stage_id=stage_id,
            resume=resume,
            raw_root=raw_root,
            repo_root=repo_root,
        )


def _bootstrap_development_utility(
    contract: PilotContract,
    *,
    raw_root: Path,
) -> dict[str, Any]:
    """Resolve q_ref and a fixed center profile for plumbing diagnostics."""

    qref_dir = raw_root / "q-ref-resolution" / "runs" / "development-qref"
    config = q_ref_run_config()
    budget = RunBudget(
        BudgetLimits(
            max_calls=48,
            max_prompt_tokens=500_000,
            max_completion_tokens=100_000,
            max_cost_usd=1e-9,
        ),
        budget_id="development-qref-budget",
    )
    result = run_verified_experiment(
        config,
        llm=MultiModelLLM(ScriptedDiagnosticProvider(), num_workers=4),
        budget=budget,
        env_config_source=DEFAULT_ENV_CONFIG,
    )
    manifest = write_verified_run_artifacts(
        qref_dir,
        result,
        provenance={
            "purpose": "development-only q_ref integration fixture",
            "contract_sha256": contract.canonical_hash,
            "diagnostic_only": True,
            "scientific_evidence": False,
        },
        git_commit=_git(Path(__file__).resolve().parents[1], "rev-parse", "HEAD"),
        git_dirty=True,
    )
    resolution = build_q_ref_resolution(
        result,
        contract_hash=contract.canonical_hash,
        environment_source_hash=hashlib.sha256(
            DEFAULT_ENV_CONFIG.read_bytes()
        ).hexdigest(),
    )
    resolution["bindings"]["source_manifest_sha256"] = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    resolution["source_manifest"] = str(manifest)
    resolution["diagnostic_only"] = True
    resolution["scientific_evidence"] = False
    qref_path = raw_root / "q-ref-resolution" / "q_ref_resolution.json"
    _atomic_json(qref_path, resolution)

    ofat = expand_stage0_ofat(float(resolution["q_ref"]))
    center = next(
        profile for profile in ofat["profiles"] if profile["profile_id"] == "center"
    )
    selection = {
        "schema_version": "finevo-stage0-selection-v1",
        "selected_profile_id": "center",
        "selected_utility": center["utility"],
        "selection_basis": [
            "development diagnostic fixed center profile",
            "no model-performance field was inspected",
        ],
        "outcome_fields_used": [],
        "contract_sha256": contract.canonical_hash,
        "q_ref_artifact_hash": canonical_sha256(resolution),
        "diagnostic_only": True,
        "scientific_evidence": False,
        "evidence_boundary": (
            "This fixed-center development fixture only exercises A-D plumbing; "
            "it is not Stage-0 scientific calibration."
        ),
    }
    selection_path = raw_root / "stage0-calibration" / "stage0_selection.json"
    _atomic_json(selection_path, selection)
    return {
        "q_ref_resolution": str(qref_path),
        "stage0_selection": str(selection_path),
    }


def run_development_fake_matrix(
    *,
    contract_path: str | Path,
    resume: bool,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
) -> dict[str, Any]:
    """Run one-seed A--D integration coverage with the no-network provider."""

    contract = load_pilot_contract(contract_path)
    root = Path(raw_root).resolve() / "development-fake"
    root.mkdir(parents=True, exist_ok=True)
    bootstrap = _bootstrap_development_utility(contract, raw_root=root)
    selected: list[PilotRunSpec] = []
    for stage_id in ("experiment-a", "experiment-b", "experiment-c", "experiment-d"):
        stage_specs = contract.expand(stage=stage_id)
        first_seed = int(contract.seeds["sets"]["main"][0])
        selected.extend(
            spec for spec in stage_specs if spec.environment_seed == first_seed
        )
    run_ledger = PilotRunLedger(
        root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=contract.schema_version.endswith("-v2"),
    )
    run_ledger.register(selected)
    if not resume and any(
        run_ledger.status(spec.run_id) != "scheduled" for spec in selected
    ):
        raise PilotOrchestrationError("development matrix already exists; use --resume")
    budget_ledger = PilotBudgetLedger(
        root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=_budget_caps(contract),
        tamper_evident=contract.schema_version.endswith("-v2"),
    )

    d_specs = tuple(spec for spec in selected if spec.stage_id == "experiment-d")
    actor_specs = tuple(spec for spec in selected if spec.stage_id != "experiment-d")
    for spec in actor_specs:
        if run_ledger.is_terminal(spec.run_id):
            continue
        if _recover_or_stop_interrupted_reservation(budget_ledger, run_ledger, spec):
            continue
        projection = RunProjection(
            run_id=spec.run_id,
            stage_bucket=spec.budget_bucket,
            cost_usd=0.0,
            completions=(
                0 if spec.execution_mode == "offline_candidate_admission" else 24
            ),
            storage_bytes=20_000_000,
            basis={
                "method": "development-diagnostic-fixed-ceiling",
                "prompt_tokens": 500_000,
                "completion_tokens": 100_000,
            },
        )
        budget_ledger.reserve(projection)
        run_dir = root / spec.stage_id / "runs" / spec.run_id
        budget: RunBudget | None = _run_budget_from_projection(projection)
        try:
            if spec.execution_mode == "offline_candidate_admission":
                artifact = _offline_candidate_admission(
                    contract,
                    spec,
                    raw_root=root,
                    diagnostic=True,
                    paid=None,
                )
            else:
                artifact, _, _ = _execute_actor_run(
                    contract,
                    spec,
                    raw_root=root,
                    paid=None,
                    projection=projection,
                    budget=budget,
                    diagnostic=True,
                    num_agents_override=2,
                    episode_length_override=6,
                )
            budget_status, budget_failure, _ = _finalize_budget_safely(
                budget_ledger,
                projection,
                run_dir=run_dir,
                budget=budget,
                status="complete",
            )
            run_ledger.finalize(
                spec.run_id,
                status=budget_status,
                artifact=str(artifact) if budget_status == "complete" else None,
                failure=budget_failure,
            )
        except Exception as exc:
            failure = {"error_type": type(exc).__name__, "message": str(exc)}
            if budget is not None:
                budget_status, budget_failure, _ = _finalize_budget_safely(
                    budget_ledger,
                    projection,
                    run_dir=run_dir,
                    budget=budget,
                    status="failed",
                    failure=failure,
                )
            else:
                budget_status, budget_failure = "failed", failure
            run_ledger.finalize(
                spec.run_id,
                status=budget_status,
                artifact=None,
                failure=budget_failure or failure,
            )

    _execute_d_seed(
        contract,
        d_specs,
        raw_root=root,
        paid=None,
        diagnostic=True,
        budget_ledger=budget_ledger,
        run_ledger=run_ledger,
    )
    statuses = [run_ledger.status(spec.run_id) for spec in selected]
    payload = {
        "schema_version": PILOT_DEVELOPMENT_MATRIX_SCHEMA_VERSION,
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
        "status": (
            "pass" if all(status == "complete" for status in statuses) else "fail"
        ),
        "registered_cells": len(selected),
        "status_counts": {
            status: statuses.count(status) for status in sorted(set(statuses))
        },
        "stages": ["experiment-a", "experiment-b", "experiment-c", "experiment-d"],
        "one_environment_seed": int(contract.seeds["sets"]["main"][0]),
        "actor_fixture_shape": {"num_agents": 2, "episode_length": 6},
        "experiment_d_shape": {"num_agents": 4, "episode_length": 12},
        "bootstrap_artifacts": bootstrap,
        "run_ledger": str(root / "run_ledger.json"),
        "budget_ledger": str(root / "budget_ledger.json"),
        "diagnostic_only": True,
        "scientific_evidence": False,
        "claim_boundary": (
            "Synthetic scripted integration coverage only. No row is model "
            "performance, treatment-effect evidence, or Stage-0 calibration."
        ),
    }
    output = root / "development_matrix_receipt.json"
    _atomic_json(output, payload)
    return {**payload, "receipt": str(output)}


__all__ = [
    "DEFAULT_RAW_ROOT",
    "GitProvenance",
    "PILOT_DEVELOPMENT_MATRIX_SCHEMA_VERSION",
    "PILOT_RUN_LEDGER_SCHEMA_VERSION",
    "PILOT_RUN_LEDGER_SCHEMA_VERSION_V2",
    "PILOT_SCIENTIFIC_LAUNCH_INPUT_SCHEMA_VERSION",
    "PILOT_STAGE_RECEIPT_SCHEMA_VERSION",
    "PILOT_STAGE_RECEIPT_SCHEMA_VERSION_V2",
    "PilotOrchestrationError",
    "PilotRunLedger",
    "config_for_spec",
    "conservative_projection",
    "execute_stage",
    "projection_from_preflight",
    "resolve_utility",
    "run_development_fake_matrix",
    "verify_paid_provenance",
]
