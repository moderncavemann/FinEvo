"""Fail-closed release provenance gate for paid FinEvo pilot execution.

This module performs only read-only Git, remote-ref, and GitHub Actions
queries.  It binds a clean local ``HEAD`` to the merged ``origin/main``
commit, the local and remote annotated ``pilot-v1`` tag object, and the two
required successful CI jobs.  The returned attestation is a frozen dataclass
whose JSON representation is deterministically SHA-256 sealed.

Raw command output is never copied into the attestation or exception text.
Only safe parsed identifiers/URLs, byte counts, and output hashes are retained.
This prevents an accidentally credential-bearing remote URL or CLI diagnostic
from leaking into pilot evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse


PILOT_RELEASE_ATTESTATION_SCHEMA_VERSION = (
    "finevo-pilot-release-attestation-v1"
)
PILOT_RELEASE_TAG = "pilot-v1"
PILOT_RELEASE_REMOTE = "origin"
PILOT_RELEASE_BRANCH = "main"
PILOT_RELEASE_WORKFLOW = "verified-memory-ci.yml"
PILOT_RELEASE_REQUIRED_JOBS = (
    "Python 3.12.7 / ubuntu-24.04",
    "Python 3.12.7 / macos-14",
)

_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_RE = re.compile(
    r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$"
)


class PilotReleaseAttestationError(RuntimeError):
    """Raised before dispatch when release provenance cannot be proved."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Raw result returned by an injectable command runner."""

    stdout: bytes
    stderr: bytes = b""
    returncode: int = 0


@dataclass(frozen=True, slots=True)
class CommandEvidence:
    """Non-sensitive receipt for one read-only command."""

    evidence_id: str
    argv: tuple[str, ...]
    returncode: int
    stdout_bytes: int
    stderr_bytes: int
    stdout_sha256: str
    stderr_sha256: str
    combined_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "argv": list(self.argv),
            "returncode": self.returncode,
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
            "combined_sha256": self.combined_sha256,
        }


@dataclass(frozen=True, slots=True)
class RequiredCIJob:
    """Identity and success state for one required matrix job."""

    name: str
    database_id: int
    status: str
    conclusion: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "database_id": self.database_id,
            "status": self.status,
            "conclusion": self.conclusion,
            "url": self.url,
        }


@dataclass(frozen=True, slots=True)
class PilotReleaseAttestation:
    """Immutable, canonical-JSON-compatible release attestation."""

    head_commit: str
    tag_object_id: str
    tag_peeled_commit: str
    remote_main_commit: str
    remote_tag_object_id: str
    remote_tag_peeled_commit: str
    github_repository: str
    github_repository_url: str
    github_run_database_id: int
    github_run_url: str
    github_run_event: str
    github_run_workflow_name: str
    required_jobs: tuple[RequiredCIJob, ...]
    command_evidence: tuple[CommandEvidence, ...]
    attestation_sha256: str = ""

    def __post_init__(self) -> None:
        computed = _sha256_json(self._unsigned_dict())
        if self.attestation_sha256 and self.attestation_sha256 != computed:
            raise PilotReleaseAttestationError(
                "release attestation SHA-256 does not match its contents"
            )
        object.__setattr__(self, "attestation_sha256", computed)

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PILOT_RELEASE_ATTESTATION_SCHEMA_VERSION,
            "status": "pass",
            "head_commit": self.head_commit,
            "local_tag": {
                "name": PILOT_RELEASE_TAG,
                "object_id": self.tag_object_id,
                "peeled_commit": self.tag_peeled_commit,
                "kind": "annotated",
            },
            "remote": {
                "name": PILOT_RELEASE_REMOTE,
                "branch": PILOT_RELEASE_BRANCH,
                "main_commit": self.remote_main_commit,
                "tag_name": PILOT_RELEASE_TAG,
                "tag_object_id": self.remote_tag_object_id,
                "tag_peeled_commit": self.remote_tag_peeled_commit,
                "tag_kind": "annotated",
            },
            "github_actions": {
                "repository": self.github_repository,
                "repository_url": self.github_repository_url,
                "workflow_file": PILOT_RELEASE_WORKFLOW,
                "run": {
                    "database_id": self.github_run_database_id,
                    "head_sha": self.head_commit,
                    "head_branch": PILOT_RELEASE_BRANCH,
                    "status": "completed",
                    "conclusion": "success",
                    "event": self.github_run_event,
                    "workflow_name": self.github_run_workflow_name,
                    "url": self.github_run_url,
                },
                "required_jobs": [
                    job.to_dict() for job in self.required_jobs
                ],
            },
            "command_evidence": [
                row.to_dict() for row in self.command_evidence
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._unsigned_dict()
        payload["attestation_sha256"] = self.attestation_sha256
        return payload

    def verify_hash(self) -> None:
        """Raise if the immutable object's content hash is not valid."""

        if self.attestation_sha256 != _sha256_json(self._unsigned_dict()):
            raise PilotReleaseAttestationError(
                "release attestation SHA-256 does not match its contents"
            )


CommandRunner = Callable[[Sequence[str], Path], CommandResult]


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _default_runner(argv: Sequence[str], cwd: Path) -> CommandResult:
    env = dict(os.environ)
    env.update(
        {
            "GH_NO_UPDATE_NOTIFIER": "1",
            "GH_PROMPT_DISABLED": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return CommandResult(
        stdout=bytes(completed.stdout),
        stderr=bytes(completed.stderr),
        returncode=int(completed.returncode),
    )

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


def _invoke(
    runner: CommandRunner,
    repo_root: Path,
    evidence: list[CommandEvidence],
    evidence_id: str,
    argv: Sequence[str],
) -> bytes:
    result = runner(tuple(argv), repo_root)
    if not isinstance(result, CommandResult):
        raise PilotReleaseAttestationError(
            f"{evidence_id} runner returned an invalid result type"
        )
    evidence.append(_command_evidence(evidence_id, argv, result))
    if result.returncode != 0:
        raise PilotReleaseAttestationError(
            f"{evidence_id} read-only command failed"
        )
    if result.stderr:
        raise PilotReleaseAttestationError(
            f"{evidence_id} produced unexpected stderr"
        )
    return result.stdout


def _strict_json(raw: bytes, evidence_id: str) -> Any:
    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PilotReleaseAttestationError(
                    f"{evidence_id} contains duplicate JSON keys"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotReleaseAttestationError(
            f"{evidence_id} did not return strict UTF-8 JSON"
        ) from exc


def _single_line(raw: bytes, evidence_id: str) -> str:
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise PilotReleaseAttestationError(
            f"{evidence_id} did not return UTF-8 text"
        ) from exc
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0] or lines[0] != lines[0].strip():
        raise PilotReleaseAttestationError(
            f"{evidence_id} did not return exactly one normalized line"
        )
    return lines[0]


def _git_object(raw: bytes, evidence_id: str) -> str:
    value = _single_line(raw, evidence_id)
    if not _GIT_OBJECT_RE.fullmatch(value):
        raise PilotReleaseAttestationError(
            f"{evidence_id} did not return one lowercase Git object ID"
        )
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PilotReleaseAttestationError(
            f"{name} must be a positive integer"
        )
    return int(value)


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PilotReleaseAttestationError(
            f"{name} must be normalized non-empty text"
        )
    return value


def _github_url(value: Any, name: str) -> str:
    url = _nonempty_text(value, name)
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise PilotReleaseAttestationError(
            f"{name} must be a credential-free github.com HTTPS URL"
        )
    return url


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
    if repository is None:
        raise PilotReleaseAttestationError(
            "origin must be a credential-free github.com remote"
        )
    if repository.endswith(".git"):
        repository = repository[:-4]
    if not _REPOSITORY_RE.fullmatch(repository):
        raise PilotReleaseAttestationError(
            "origin does not identify exactly one GitHub repository"
        )
    return repository


def _remote_refs(raw: bytes) -> dict[str, str]:
    expected = {
        f"refs/heads/{PILOT_RELEASE_BRANCH}",
        f"refs/tags/{PILOT_RELEASE_TAG}",
        f"refs/tags/{PILOT_RELEASE_TAG}^{{}}",
    }
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise PilotReleaseAttestationError(
            "remote_refs did not return UTF-8 text"
        ) from exc
    observed: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            raise PilotReleaseAttestationError(
                "remote_refs contains a malformed line"
            )
        object_id, ref = parts
        if (
            not _GIT_OBJECT_RE.fullmatch(object_id)
            or ref not in expected
            or ref in observed
        ):
            raise PilotReleaseAttestationError(
                "remote_refs contains an unexpected or ambiguous ref"
            )
        observed[ref] = object_id
    if set(observed) != expected:
        raise PilotReleaseAttestationError(
            "remote_refs is missing main, annotated tag, or peeled tag"
        )
    return observed


def _github_repository(raw: bytes, expected_repository: str) -> tuple[str, str]:
    payload = _strict_json(raw, "github_repository")
    if not isinstance(payload, Mapping):
        raise PilotReleaseAttestationError(
            "github_repository must be a JSON object"
        )
    repository = _nonempty_text(
        payload.get("nameWithOwner"),
        "github_repository.nameWithOwner",
    )
    if repository != expected_repository:
        raise PilotReleaseAttestationError(
            "GitHub repository identity does not match origin"
        )
    url = _github_url(payload.get("url"), "github_repository.url")
    parsed = urlparse(url)
    if parsed.path.rstrip("/") != f"/{repository}":
        raise PilotReleaseAttestationError(
            "GitHub repository URL does not match its repository identity"
        )
    return repository, url


def _parse_run_row(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotReleaseAttestationError(f"{name} must be a JSON object")
    return {
        "database_id": _positive_integer(
            value.get("databaseId"), f"{name}.databaseId"
        ),
        "head_sha": _nonempty_text(value.get("headSha"), f"{name}.headSha"),
        "head_branch": _nonempty_text(
            value.get("headBranch"), f"{name}.headBranch"
        ),
        "status": _nonempty_text(value.get("status"), f"{name}.status"),
        "conclusion": _nonempty_text(
            value.get("conclusion"), f"{name}.conclusion"
        ),
        "url": _github_url(value.get("url"), f"{name}.url"),
        "event": _nonempty_text(value.get("event"), f"{name}.event"),
        "workflow_name": _nonempty_text(
            value.get("workflowName"), f"{name}.workflowName"
        ),
    }


def _select_github_run(
    raw: bytes,
    *,
    head_commit: str,
) -> dict[str, Any]:
    payload = _strict_json(raw, "github_runs")
    if not isinstance(payload, list):
        raise PilotReleaseAttestationError(
            "github_runs must be a JSON array"
        )
    rows = [
        _parse_run_row(value, name=f"github_runs[{index}]")
        for index, value in enumerate(payload)
    ]
    for row in rows:
        if (
            row["head_sha"] != head_commit
            or row["head_branch"] != PILOT_RELEASE_BRANCH
        ):
            raise PilotReleaseAttestationError(
                "github_runs returned a run outside the exact commit/main filter"
            )
    candidates = [
        row
        for row in rows
        if row["status"] == "completed"
        and row["conclusion"] == "success"
    ]
    if len(candidates) != 1:
        raise PilotReleaseAttestationError(
            "expected exactly one successful completed workflow run "
            "for the exact main commit"
        )
    return candidates[0]


def _parse_required_jobs(
    value: Any,
    *,
    repository: str,
    run_database_id: int,
) -> tuple[RequiredCIJob, ...]:
    if not isinstance(value, list):
        raise PilotReleaseAttestationError(
            "github_run.jobs must be a JSON array"
        )
    jobs_by_name: dict[str, list[RequiredCIJob]] = {
        name: [] for name in PILOT_RELEASE_REQUIRED_JOBS
    }
    for index, raw_job in enumerate(value):
        if not isinstance(raw_job, Mapping):
            raise PilotReleaseAttestationError(
                f"github_run.jobs[{index}] must be a JSON object"
            )
        name = _nonempty_text(
            raw_job.get("name"), f"github_run.jobs[{index}].name"
        )
        database_id = _positive_integer(
            raw_job.get("databaseId"),
            f"github_run.jobs[{index}].databaseId",
        )
        status = _nonempty_text(
            raw_job.get("status"), f"github_run.jobs[{index}].status"
        )
        conclusion = _nonempty_text(
            raw_job.get("conclusion"),
            f"github_run.jobs[{index}].conclusion",
        )
        url = _github_url(
            raw_job.get("url"), f"github_run.jobs[{index}].url"
        )
        if name not in jobs_by_name:
            continue
        parsed = urlparse(url)
        expected_prefix = (
            f"/{repository}/actions/runs/{run_database_id}/job/"
        )
        if not parsed.path.startswith(expected_prefix):
            raise PilotReleaseAttestationError(
                f"required CI job {name!r} URL is not bound to the selected run"
            )
        jobs_by_name[name].append(
            RequiredCIJob(
                name=name,
                database_id=database_id,
                status=status,
                conclusion=conclusion,
                url=url,
            )
        )
    selected: list[RequiredCIJob] = []
    for name in PILOT_RELEASE_REQUIRED_JOBS:
        matches = jobs_by_name[name]
        if len(matches) != 1:
            raise PilotReleaseAttestationError(
                f"expected exactly one required CI job named {name!r}"
            )
        job = matches[0]
        if job.status != "completed" or job.conclusion != "success":
            raise PilotReleaseAttestationError(
                f"required CI job {name!r} did not complete successfully"
            )
        selected.append(job)
    return tuple(selected)


def _validate_run_view(
    raw: bytes,
    *,
    repository: str,
    selected_run: Mapping[str, Any],
    head_commit: str,
) -> tuple[RequiredCIJob, ...]:
    payload = _strict_json(raw, "github_run")
    if not isinstance(payload, Mapping):
        raise PilotReleaseAttestationError(
            "github_run must be a JSON object"
        )
    run = _parse_run_row(payload, name="github_run")
    exact_fields = (
        "database_id",
        "head_sha",
        "head_branch",
        "status",
        "conclusion",
        "url",
        "event",
        "workflow_name",
    )
    if any(run[field] != selected_run[field] for field in exact_fields):
        raise PilotReleaseAttestationError(
            "github_run detail does not match the selected run"
        )
    if (
        run["head_sha"] != head_commit
        or run["head_branch"] != PILOT_RELEASE_BRANCH
        or run["status"] != "completed"
        or run["conclusion"] != "success"
    ):
        raise PilotReleaseAttestationError(
            "github_run is not a successful run for the exact main commit"
        )
    parsed_url = urlparse(run["url"])
    expected_path = (
        f"/{repository}/actions/runs/{run['database_id']}"
    )
    if parsed_url.path.rstrip("/") != expected_path:
        raise PilotReleaseAttestationError(
            "github_run URL is not bound to its repository and database ID"
        )
    return _parse_required_jobs(
        payload.get("jobs"),
        repository=repository,
        run_database_id=run["database_id"],
    )


def verify_pilot_release_attestation(
    repo_root: Path | str,
    *,
    runner: CommandRunner | None = None,
) -> PilotReleaseAttestation:
    """Verify and seal the exact release/CI state required for paid calls.

    The function does not fetch Git objects or mutate local/remote state.
    ``git ls-remote`` and ``gh`` are used only for read-only live attestation.
    Every required identity must be present exactly once.
    """

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise PilotReleaseAttestationError(
            "repository root does not exist or is not a directory"
        )
    invoke_runner = runner or _default_runner
    evidence: list[CommandEvidence] = []

    head_commit = _git_object(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "local_head",
            ("git", "rev-parse", "--verify", "HEAD^{commit}"),
        ),
        "local_head",
    )

    status = _invoke(
        invoke_runner,
        root,
        evidence,
        "local_status",
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
    )
    if status != b"":
        raise PilotReleaseAttestationError(
            "local HEAD is not clean, including untracked files"
        )

    tag_object_id = _git_object(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "local_tag_object",
            (
                "git",
                "rev-parse",
                "--verify",
                f"refs/tags/{PILOT_RELEASE_TAG}^{{tag}}",
            ),
        ),
        "local_tag_object",
    )
    tag_peeled_commit = _git_object(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "local_tag_commit",
            (
                "git",
                "rev-parse",
                "--verify",
                f"refs/tags/{PILOT_RELEASE_TAG}^{{commit}}",
            ),
        ),
        "local_tag_commit",
    )
    if tag_peeled_commit != head_commit or tag_object_id == head_commit:
        raise PilotReleaseAttestationError(
            "local annotated pilot-v1 tag does not peel exactly to HEAD"
        )

    origin_repository = _repository_from_origin(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "origin_url",
            ("git", "remote", "get-url", PILOT_RELEASE_REMOTE),
        )
    )

    remote = _remote_refs(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "remote_refs",
            (
                "git",
                "ls-remote",
                "--exit-code",
                PILOT_RELEASE_REMOTE,
                f"refs/heads/{PILOT_RELEASE_BRANCH}",
                f"refs/tags/{PILOT_RELEASE_TAG}",
                f"refs/tags/{PILOT_RELEASE_TAG}^{{}}",
            ),
        )
    )
    remote_main_commit = remote[f"refs/heads/{PILOT_RELEASE_BRANCH}"]
    remote_tag_object_id = remote[f"refs/tags/{PILOT_RELEASE_TAG}"]
    remote_tag_peeled_commit = remote[
        f"refs/tags/{PILOT_RELEASE_TAG}^{{}}"
    ]
    if (
        remote_main_commit != head_commit
        or remote_tag_object_id != tag_object_id
        or remote_tag_peeled_commit != head_commit
    ):
        raise PilotReleaseAttestationError(
            "origin/main or the remote annotated pilot-v1 tag "
            "does not match local HEAD/tag"
        )

    repository, repository_url = _github_repository(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "github_repository",
            (
                "gh",
                "repo",
                "view",
                origin_repository,
                "--json",
                "nameWithOwner,url",
            ),
        ),
        origin_repository,
    )

    runs_fields = (
        "databaseId,headSha,headBranch,status,conclusion,url,"
        "event,workflowName"
    )
    selected_run = _select_github_run(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "github_runs",
            (
                "gh",
                "run",
                "list",
                "--repo",
                repository,
                "--workflow",
                PILOT_RELEASE_WORKFLOW,
                "--branch",
                PILOT_RELEASE_BRANCH,
                "--commit",
                head_commit,
                "--limit",
                "20",
                "--json",
                runs_fields,
            ),
        ),
        head_commit=head_commit,
    )

    jobs = _validate_run_view(
        _invoke(
            invoke_runner,
            root,
            evidence,
            "github_run",
            (
                "gh",
                "run",
                "view",
                str(selected_run["database_id"]),
                "--repo",
                repository,
                "--json",
                f"{runs_fields},jobs",
            ),
        ),
        repository=repository,
        selected_run=selected_run,
        head_commit=head_commit,
    )

    return PilotReleaseAttestation(
        head_commit=head_commit,
        tag_object_id=tag_object_id,
        tag_peeled_commit=tag_peeled_commit,
        remote_main_commit=remote_main_commit,
        remote_tag_object_id=remote_tag_object_id,
        remote_tag_peeled_commit=remote_tag_peeled_commit,
        github_repository=repository,
        github_repository_url=repository_url,
        github_run_database_id=selected_run["database_id"],
        github_run_url=selected_run["url"],
        github_run_event=selected_run["event"],
        github_run_workflow_name=selected_run["workflow_name"],
        required_jobs=jobs,
        command_evidence=tuple(evidence),
    )
