import json
import stat
from pathlib import Path

import pytest

from verified_memory.artifacts import (
    ArtifactFinalizedError,
    ArtifactValidationError,
    JsonField,
    JsonlStreamSchema,
    ManifestVerificationError,
    RunArtifactWriter,
    verify_manifest,
)
from verified_memory.budget import BudgetLimits, RunBudget


def schemas():
    return (
        JsonlStreamSchema(
            name="actions",
            relative_path="streams/actions.jsonl",
            fields=(
                JsonField("month", "integer"),
                JsonField("agent_id", "integer"),
                JsonField("work", "number"),
                JsonField("note", "string", required=False, nullable=True),
            ),
        ),
        JsonlStreamSchema(
            name="errors",
            relative_path="streams/errors.jsonl",
            fields=(JsonField("error_type", "string"),),
            required=False,
            min_records=0,
        ),
    )


def make_writer(tmp_path: Path, *, resume=False, config=None, provenance=None):
    return RunArtifactWriter.create(
        tmp_path / "run",
        schemas(),
        config=config or {"model": "stub", "temperature": 0.0},
        provenance=provenance or {"experiment": "smoke"},
        git_commit="abc123",
        git_dirty=True,
        resume=resume,
    )


def finalize(writer: RunArtifactWriter, *, complete=True):
    budget = RunBudget(BudgetLimits(max_calls=2), budget_id="artifact-test")
    return writer.finalize(
        validation_status={"status": "pass", "checks": 4},
        budget_snapshot=budget.snapshot(),
        result_complete=complete,
    )


def test_canonical_schema_checked_jsonl_and_complete_manifest(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    assert writer.append("actions", {"work": 0.8, "agent_id": 2, "month": 1}) == 1
    assert writer.append(
        "actions", {"month": 2, "agent_id": 2, "work": 1, "note": None}
    ) == 2

    stream_path = tmp_path / "run/streams/actions.jsonl"
    assert stream_path.read_bytes().splitlines()[0] == (
        b'{"agent_id":2,"month":1,"work":0.8}'
    )
    manifest_path = finalize(writer)
    manifest = json.loads(manifest_path.read_text())

    assert manifest["git"] == {"commit": "abc123", "dirty": True}
    assert manifest["validation_status"] == {"checks": 4, "status": "pass"}
    assert manifest["result"]["complete"] is True
    assert manifest["result"]["stream_line_counts"] == {"actions": 2, "errors": 0}
    paths = [entry["path"] for entry in manifest["artifacts"]]
    assert paths == sorted(paths)
    assert all(not Path(path).is_absolute() for path in paths)
    action_entry = next(entry for entry in manifest["artifacts"] if entry["path"].endswith("actions.jsonl"))
    assert action_entry["line_count"] == 2
    assert action_entry["byte_size"] == stream_path.stat().st_size
    assert len(action_entry["sha256"]) == 64

    verification = verify_manifest(tmp_path / "run")
    assert verification.valid is True
    assert verification.artifact_count == 4
    assert len(verification.manifest_sha256) == 64


def test_nonempty_directory_requires_explicit_exact_resume(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    writer.append("actions", {"month": 1, "agent_id": 0, "work": 1.0})

    with pytest.raises(FileExistsError):
        make_writer(tmp_path)

    resumed = make_writer(tmp_path, resume=True)
    assert resumed.append("actions", {"month": 2, "agent_id": 0, "work": 0.5}) == 2
    finalize(resumed)
    assert verify_manifest(tmp_path / "run").valid


def test_resume_rejects_config_or_provenance_change(tmp_path: Path) -> None:
    make_writer(tmp_path)
    with pytest.raises(ArtifactValidationError, match="config"):
        make_writer(tmp_path, resume=True, config={"model": "different"})
    with pytest.raises(ArtifactValidationError, match="provenance"):
        make_writer(tmp_path, resume=True, provenance={"experiment": "different"})


def test_missing_required_stream_and_duplicate_finalization_fail_closed(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    with pytest.raises(ArtifactValidationError, match="required streams"):
        finalize(writer, complete=False)

    writer.append("actions", {"month": 1, "agent_id": 0, "work": 1.0})
    finalize(writer)
    with pytest.raises(ArtifactFinalizedError):
        finalize(writer)
    with pytest.raises(ArtifactFinalizedError):
        writer.append("actions", {"month": 2, "agent_id": 0, "work": 1.0})
    with pytest.raises(ArtifactFinalizedError):
        make_writer(tmp_path, resume=True)


def test_nan_in_config_record_validation_or_finalization_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ArtifactValidationError):
        make_writer(tmp_path / "config-nan", config={"value": float("nan")})

    writer = make_writer(tmp_path / "record-nan")
    with pytest.raises(ArtifactValidationError):
        writer.append("actions", {"month": 1, "agent_id": 0, "work": float("nan")})
    writer.append("actions", {"month": 1, "agent_id": 0, "work": 1.0})
    with pytest.raises(ArtifactValidationError):
        writer.finalize(
            validation_status={"status": "pass", "metric": float("inf")},
            budget_snapshot={"calls": 0},
            result_complete=True,
        )


@pytest.mark.parametrize(
    "unsafe",
    [
        "../escape.jsonl",
        "/tmp/escape.jsonl",
        "a/../../b.jsonl",
        "a/./b.jsonl",
        "a//b.jsonl",
        "a\\b.jsonl",
    ],
)
def test_stream_schema_rejects_path_traversal(unsafe: str) -> None:
    with pytest.raises(ArtifactValidationError):
        JsonlStreamSchema(
            name="unsafe",
            relative_path=unsafe,
            fields=(JsonField("x", "integer"),),
        )


def test_schema_rejects_missing_extra_and_wrong_typed_fields(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    with pytest.raises(ArtifactValidationError, match="missing required"):
        writer.append("actions", {"month": 1, "work": 1.0})
    with pytest.raises(ArtifactValidationError, match="undeclared"):
        writer.append("actions", {"month": 1, "agent_id": 0, "work": 1.0, "extra": 1})
    with pytest.raises(ArtifactValidationError, match="must be integer"):
        writer.append("actions", {"month": True, "agent_id": 0, "work": 1.0})


def test_metadata_or_stream_mutation_before_finalize_is_detected(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    writer.append("actions", {"month": 1, "agent_id": 0, "work": 1.0})

    config_path = tmp_path / "run/config.json"
    config_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    config_path.write_text('{"mutated":true}\n')
    with pytest.raises(ArtifactValidationError, match="metadata was mutated"):
        writer.append("actions", {"month": 2, "agent_id": 0, "work": 1.0})

    # A separate run exercises same-size/canonical stream mutation detection at finalize.
    other = make_writer(tmp_path / "stream")
    other.append("actions", {"month": 1, "agent_id": 0, "work": 1.0})
    stream = tmp_path / "stream/run/streams/actions.jsonl"
    original = stream.read_bytes()
    mutated = original.replace(b'"month":1', b'"month":2')
    assert len(mutated) == len(original)
    stream.write_bytes(mutated)
    with pytest.raises(ArtifactValidationError, match="outside writer"):
        finalize(other)


def test_undeclared_file_or_symlink_fails_before_sealing(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    writer.append("actions", {"month": 1, "agent_id": 0, "work": 1.0})
    unexpected = tmp_path / "run/untracked.txt"
    unexpected.write_text("unexpected")
    with pytest.raises(ArtifactValidationError, match="file set"):
        finalize(writer)
    assert not (tmp_path / "run/manifest.json").exists()

    unexpected.unlink()
    finalize(writer)

    symlink_run = make_writer(tmp_path / "symlink")
    redirect = tmp_path / "symlink/run/redirect"
    redirect.mkdir()
    (tmp_path / "symlink/run/streams").symlink_to(redirect, target_is_directory=True)
    with pytest.raises(ArtifactValidationError, match="symlinks are forbidden"):
        symlink_run.append("actions", {"month": 1, "agent_id": 0, "work": 1.0})


def test_resume_rejects_undeclared_non_stream_artifact(tmp_path: Path) -> None:
    make_writer(tmp_path)
    (tmp_path / "run/notes.txt").write_text("not declared")
    with pytest.raises(ArtifactValidationError, match="undeclared artifact"):
        make_writer(tmp_path, resume=True)


def test_verify_manifest_detects_hash_mutation_and_extra_files_read_only(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    writer.append("actions", {"month": 1, "agent_id": 0, "work": 1.0})
    finalize(writer)

    stream = tmp_path / "run/streams/actions.jsonl"
    stream.chmod(stat.S_IRUSR | stat.S_IWUSR)
    stream.write_bytes(stream.read_bytes().replace(b'"month":1', b'"month":9'))
    with pytest.raises(ManifestVerificationError, match="sha256 mismatch"):
        verify_manifest(tmp_path / "run")

    # Restore by creating a fresh sealed run, then add an undeclared artifact.
    fresh = make_writer(tmp_path / "extra")
    fresh.append("actions", {"month": 1, "agent_id": 0, "work": 1.0})
    finalize(fresh)
    (tmp_path / "extra/run/untracked.txt").write_text("unexpected")
    with pytest.raises(ManifestVerificationError, match="file set mismatch"):
        verify_manifest(tmp_path / "extra/run")


def test_manifest_with_traversal_entry_is_rejected_without_reading_outside(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    writer.append("actions", {"month": 1, "agent_id": 0, "work": 1.0})
    manifest_path = finalize(writer)
    manifest_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][0]["path"] = "../outside"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n")

    with pytest.raises(ManifestVerificationError, match="unsafe relative"):
        verify_manifest(tmp_path / "run")


def test_finalized_files_are_marked_read_only(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    writer.append("actions", {"month": 1, "agent_id": 0, "work": 1.0})
    finalize(writer)

    for relative in ("config.json", "provenance.json", "schemas.json", "manifest.json", "streams/actions.jsonl"):
        mode = stat.S_IMODE((tmp_path / "run" / relative).stat().st_mode)
        assert mode & 0o222 == 0
