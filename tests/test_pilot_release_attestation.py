import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from verified_memory.pilot_release_attestation import (
    PILOT_RELEASE_ATTESTATION_SCHEMA_VERSION,
    CommandResult,
    PilotReleaseAttestationError,
    verify_pilot_release_attestation,
)


HEAD = "1" * 40
TAG_OBJECT = "2" * 40
REPOSITORY = "owner/finevo"
RUN_ID = 12345
RUN_URL = f"https://github.com/{REPOSITORY}/actions/runs/{RUN_ID}"
WORKFLOW_NAME = "Verified memory CI"


def _json_bytes(value):
    return json.dumps(value, sort_keys=True).encode()


def _run_row(**updates):
    row = {
        "databaseId": RUN_ID,
        "headSha": HEAD,
        "headBranch": "main",
        "status": "completed",
        "conclusion": "success",
        "url": RUN_URL,
        "event": "push",
        "workflowName": WORKFLOW_NAME,
    }
    row.update(updates)
    return row


def _job(name, database_id):
    return {
        "name": name,
        "databaseId": database_id,
        "status": "completed",
        "conclusion": "success",
        "url": f"{RUN_URL}/job/{database_id}",
    }


def _responses():
    run = _run_row()
    run_view = {
        **run,
        "jobs": [
            _job("Python 3.12.7 / ubuntu-24.04", 2001),
            _job("Python 3.12.7 / macos-14", 2002),
        ],
    }
    remote_refs = (
        f"{HEAD}\trefs/heads/main\n"
        f"{TAG_OBJECT}\trefs/tags/pilot-v1\n"
        f"{HEAD}\trefs/tags/pilot-v1^{{}}\n"
    ).encode()
    return {
        "local_head": CommandResult(f"{HEAD}\n".encode()),
        "local_status": CommandResult(b""),
        "local_tag_object": CommandResult(f"{TAG_OBJECT}\n".encode()),
        "local_tag_commit": CommandResult(f"{HEAD}\n".encode()),
        "origin_url": CommandResult(
            f"https://github.com/{REPOSITORY}.git\n".encode()
        ),
        "remote_refs": CommandResult(remote_refs),
        "github_repository": CommandResult(
            _json_bytes(
                {
                    "nameWithOwner": REPOSITORY,
                    "url": f"https://github.com/{REPOSITORY}",
                }
            )
        ),
        "github_runs": CommandResult(_json_bytes([run])),
        "github_run": CommandResult(_json_bytes(run_view)),
    }


class FakeRunner:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, argv, cwd):
        argv = tuple(argv)
        self.calls.append((argv, cwd))
        if argv[:3] == ("git", "rev-parse", "--verify"):
            expression = argv[3]
            if expression == "HEAD^{commit}":
                key = "local_head"
            elif expression == "refs/tags/pilot-v1^{tag}":
                key = "local_tag_object"
            elif expression == "refs/tags/pilot-v1^{commit}":
                key = "local_tag_commit"
            else:  # pragma: no cover - guards fixture drift
                raise AssertionError(expression)
        elif argv[:2] == ("git", "status"):
            key = "local_status"
        elif argv[:3] == ("git", "remote", "get-url"):
            key = "origin_url"
        elif argv[:2] == ("git", "ls-remote"):
            key = "remote_refs"
        elif argv[:3] == ("gh", "repo", "view"):
            key = "github_repository"
        elif argv[:3] == ("gh", "run", "list"):
            key = "github_runs"
        elif argv[:3] == ("gh", "run", "view"):
            key = "github_run"
        else:  # pragma: no cover - guards fixture drift
            raise AssertionError(argv)
        return self.responses[key]


def _attest(tmp_path: Path, responses=None):
    runner = FakeRunner(responses or _responses())
    return verify_pilot_release_attestation(tmp_path, runner=runner), runner


def test_release_attestation_binds_clean_head_remote_tag_and_exact_ci_jobs(
    tmp_path: Path,
):
    attestation, runner = _attest(tmp_path)
    payload = attestation.to_dict()

    assert payload["schema_version"] == PILOT_RELEASE_ATTESTATION_SCHEMA_VERSION
    assert payload["status"] == "pass"
    assert payload["head_commit"] == HEAD
    assert payload["local_tag"] == {
        "name": "pilot-v1",
        "object_id": TAG_OBJECT,
        "peeled_commit": HEAD,
        "kind": "annotated",
    }
    assert payload["remote"]["main_commit"] == HEAD
    assert payload["remote"]["tag_object_id"] == TAG_OBJECT
    assert payload["remote"]["tag_peeled_commit"] == HEAD
    assert payload["github_actions"]["repository"] == REPOSITORY
    assert payload["github_actions"]["workflow_file"] == (
        "verified-memory-ci.yml"
    )
    assert payload["github_actions"]["run"]["database_id"] == RUN_ID
    assert [
        row["name"]
        for row in payload["github_actions"]["required_jobs"]
    ] == [
        "Python 3.12.7 / ubuntu-24.04",
        "Python 3.12.7 / macos-14",
    ]
    assert all(
        row["conclusion"] == "success"
        for row in payload["github_actions"]["required_jobs"]
    )
    assert len(payload["command_evidence"]) == 9
    assert [row["evidence_id"] for row in payload["command_evidence"]] == [
        "local_head",
        "local_status",
        "local_tag_object",
        "local_tag_commit",
        "origin_url",
        "remote_refs",
        "github_repository",
        "github_runs",
        "github_run",
    ]
    assert json.loads(json.dumps(payload)) == payload
    unsigned = dict(payload)
    observed_hash = unsigned.pop("attestation_sha256")
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    assert observed_hash == hashlib.sha256(canonical).hexdigest()
    attestation.verify_hash()
    with pytest.raises(dataclasses.FrozenInstanceError):
        attestation.head_commit = "3" * 40

    assert all(cwd == tmp_path.resolve() for _, cwd in runner.calls)
    run_list_argv = next(
        argv
        for argv, _ in runner.calls
        if argv[:3] == ("gh", "run", "list")
    )
    assert "--commit" in run_list_argv
    assert run_list_argv[run_list_argv.index("--commit") + 1] == HEAD
    assert "--branch" in run_list_argv
    assert run_list_argv[run_list_argv.index("--branch") + 1] == "main"
    assert "verified-memory-ci.yml" in run_list_argv


def test_attestation_does_not_copy_raw_command_output_or_credentials(
    tmp_path: Path,
):
    responses = _responses()
    # A credential-bearing origin is rejected, and the error never echoes it.
    secret = "ghp_EXAMPLE_SHOULD_NEVER_APPEAR"
    responses["origin_url"] = CommandResult(
        f"https://{secret}@github.com/{REPOSITORY}.git\n".encode()
    )
    with pytest.raises(PilotReleaseAttestationError) as caught:
        _attest(tmp_path, responses)
    assert secret not in str(caught.value)

    attestation, _ = _attest(tmp_path)
    serialized = json.dumps(attestation.to_dict())
    assert f"{HEAD}\\t" not in serialized


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda rows: rows.__setitem__(
                "local_status", CommandResult(b"?? untracked.txt\n")
            ),
            "not clean",
        ),
        (
            lambda rows: rows.__setitem__(
                "local_tag_commit", CommandResult(f"{'3' * 40}\n".encode())
            ),
            "does not peel",
        ),
        (
            lambda rows: rows.__setitem__(
                "local_tag_object",
                CommandResult(b"", b"not an annotated tag", 128),
            ),
            "read-only command failed",
        ),
        (
            lambda rows: rows.__setitem__(
                "local_head", CommandResult(f"{HEAD}\n".encode(), b"warning")
            ),
            "unexpected stderr",
        ),
    ],
)
def test_local_release_state_fails_closed(tmp_path: Path, mutate, message):
    responses = _responses()
    mutate(responses)
    with pytest.raises(PilotReleaseAttestationError, match=message):
        _attest(tmp_path, responses)


def test_remote_main_and_annotated_tag_must_be_complete_and_exact(
    tmp_path: Path,
):
    for remote_refs in (
        (
            f"{'3' * 40}\trefs/heads/main\n"
            f"{TAG_OBJECT}\trefs/tags/pilot-v1\n"
            f"{HEAD}\trefs/tags/pilot-v1^{{}}\n"
        ),
        (
            f"{HEAD}\trefs/heads/main\n"
            f"{TAG_OBJECT}\trefs/tags/pilot-v1\n"
        ),
        (
            f"{HEAD}\trefs/heads/main\n"
            f"{TAG_OBJECT}\trefs/tags/pilot-v1\n"
            f"{HEAD}\trefs/tags/pilot-v1^{{}}\n"
            f"{HEAD}\trefs/tags/pilot-v1^{{}}\n"
        ),
    ):
        responses = _responses()
        responses["remote_refs"] = CommandResult(remote_refs.encode())
        with pytest.raises(PilotReleaseAttestationError):
            _attest(tmp_path, responses)


def test_github_run_selection_rejects_missing_mismatch_and_ambiguity(
    tmp_path: Path,
):
    cases = (
        [],
        [_run_row(headSha="4" * 40)],
        [_run_row(), _run_row(databaseId=67890, url=(
            f"https://github.com/{REPOSITORY}/actions/runs/67890"
        ))],
    )
    for runs in cases:
        responses = _responses()
        responses["github_runs"] = CommandResult(_json_bytes(runs))
        with pytest.raises(PilotReleaseAttestationError):
            _attest(tmp_path, responses)


@pytest.mark.parametrize(
    "jobs",
    [
        [_job("Python 3.12.7 / ubuntu-24.04", 2001)],
        [
            _job("Python 3.12.7 / ubuntu-24.04", 2001),
            {
                **_job("Python 3.12.7 / macos-14", 2002),
                "conclusion": "failure",
            },
        ],
        [
            _job("Python 3.12.7 / ubuntu-24.04", 2001),
            _job("Python 3.12.7 / ubuntu-24.04", 2003),
            _job("Python 3.12.7 / macos-14", 2002),
        ],
        [
            _job("Python 3.12.7 / ubuntu-24.04", 2001),
            {
                **_job("Python 3.12.7 / macos-14", 2002),
                "url": (
                    f"https://github.com/{REPOSITORY}/actions/runs/999/job/2002"
                ),
            },
        ],
    ],
)
def test_both_exact_ci_matrix_jobs_are_required_and_successful(
    tmp_path: Path,
    jobs,
):
    responses = _responses()
    responses["github_run"] = CommandResult(
        _json_bytes({**_run_row(), "jobs": jobs})
    )
    with pytest.raises(PilotReleaseAttestationError):
        _attest(tmp_path, responses)


def test_run_detail_must_match_selected_run_exactly(tmp_path: Path):
    responses = _responses()
    responses["github_run"] = CommandResult(
        _json_bytes(
            {
                **_run_row(event="workflow_dispatch"),
                "jobs": [
                    _job("Python 3.12.7 / ubuntu-24.04", 2001),
                    _job("Python 3.12.7 / macos-14", 2002),
                ],
            }
        )
    )
    with pytest.raises(PilotReleaseAttestationError, match="does not match"):
        _attest(tmp_path, responses)


def test_duplicate_json_keys_are_rejected(tmp_path: Path):
    responses = _responses()
    responses["github_repository"] = CommandResult(
        (
            '{"nameWithOwner":"owner/finevo",'
            '"nameWithOwner":"other/repo",'
            '"url":"https://github.com/owner/finevo"}'
        ).encode()
    )
    with pytest.raises(PilotReleaseAttestationError, match="duplicate JSON"):
        _attest(tmp_path, responses)
