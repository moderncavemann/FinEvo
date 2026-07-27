from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
import subprocess
from typing import Any

import pytest

import verified_memory.pilot_checkpoint as checkpoint_module
from verified_memory.pilot_checkpoint import (
    PilotCheckpoint,
    PilotCheckpointError,
    canonical_hash,
    verify_checkpoint_code_binding_from_annotated_tag,
    verify_closed_loop_preflight_checkpoint,
    verify_historical_closed_loop_preflight_checkpoint,
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8").strip()


def _tagged_source_repo(
    root: Path,
) -> tuple[dict[str, Any], str, str, str]:
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "Historical Checkpoint Test")
    _git(root, "config", "user.email", "checkpoint@example.invalid")

    source_hashes: dict[str, str] = {}
    for index, relative in enumerate(checkpoint_module._CODE_FILES):
        raw = f"historical source {index}: {relative}\n".encode("utf-8")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        source_hashes[relative] = hashlib.sha256(raw).hexdigest()
    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "-m", "historical source")
    tag = "historical-checkpoint-v1"
    _git(root, "tag", "-a", tag, "-m", "historical checkpoint source")

    binding: dict[str, Any] = {"source_hashes": source_hashes}
    binding["binding_hash"] = canonical_hash(binding)
    tag_object = _git(root, "rev-parse", f"refs/tags/{tag}")
    peeled_commit = _git(
        root,
        "rev-parse",
        f"refs/tags/{tag}^{{commit}}",
    )
    return binding, tag, tag_object, peeled_commit


def _synthetic_checkpoint(binding: dict[str, Any]) -> PilotCheckpoint:
    """Construct a focused unit-test object without a Foundation execution."""

    checkpoint = object.__new__(PilotCheckpoint)
    object.__setattr__(
        checkpoint,
        "payload",
        {
            "checkpoint_hash": "c" * 64,
            "code_binding": deepcopy(binding),
        },
    )
    return checkpoint


def test_historical_gate_rejects_tagged_blob_and_binding_tampering(
    tmp_path: Path,
) -> None:
    binding, tag, tag_object, peeled_commit = _tagged_source_repo(
        tmp_path / "parent"
    )

    blob_drift = deepcopy(binding)
    first_source = checkpoint_module._CODE_FILES[0]
    blob_drift["source_hashes"][first_source] = "a" * 64
    blob_drift["binding_hash"] = canonical_hash(
        {"source_hashes": blob_drift["source_hashes"]}
    )
    with pytest.raises(
        PilotCheckpointError,
        match="tagged code binding differs",
    ):
        verify_checkpoint_code_binding_from_annotated_tag(
            _synthetic_checkpoint(blob_drift),
            source_repo_root=tmp_path / "parent",
            source_annotated_tag=tag,
            expected_tag_object=tag_object,
            expected_peeled_commit=peeled_commit,
        )

    hash_tamper = deepcopy(binding)
    hash_tamper["binding_hash"] = "b" * 64
    with pytest.raises(
        PilotCheckpointError,
        match="code binding hash is invalid",
    ):
        verify_checkpoint_code_binding_from_annotated_tag(
            _synthetic_checkpoint(hash_tamper),
            source_repo_root=tmp_path / "parent",
            source_annotated_tag=tag,
            expected_tag_object=tag_object,
            expected_peeled_commit=peeled_commit,
        )

    with pytest.raises(
        PilotCheckpointError,
        match="annotated tag object differs",
    ):
        verify_checkpoint_code_binding_from_annotated_tag(
            _synthetic_checkpoint(binding),
            source_repo_root=tmp_path / "parent",
            source_annotated_tag=tag,
            expected_tag_object="d" * 40,
            expected_peeled_commit=peeled_commit,
        )


def test_historical_gate_allows_one_expected_child_code_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, tag, tag_object, peeled_commit = _tagged_source_repo(
        tmp_path / "parent"
    )
    checkpoint = _synthetic_checkpoint(binding)
    child_binding = deepcopy(binding)
    changed_source = "verified_memory/pilot_continuation.py"
    child_binding["source_hashes"][changed_source] = "e" * 64
    child_binding["binding_hash"] = canonical_hash(
        {"source_hashes": child_binding["source_hashes"]}
    )
    parent_root = tmp_path / "parent"
    post_tag_path = parent_root / checkpoint_module._CODE_FILES[0]
    post_tag_path.write_text(
        "a later HEAD must not replace the tagged source tree\n",
        encoding="utf-8",
    )
    _git(parent_root, "add", "--all")
    _git(parent_root, "commit", "--quiet", "-m", "later parent HEAD")
    assert _git(parent_root, "rev-parse", "HEAD") != peeled_commit
    post_tag_path.write_text(
        "an uncommitted working-tree edit must also be ignored\n",
        encoding="utf-8",
    )
    frozen_exactness = {
        "schema_version": "finevo-checkpoint-exactness-receipt-v1",
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "provider_calls_during_verification": 0,
        "component_hashes": {"environment_hash": "f" * 64},
    }
    monkeypatch.setattr(
        checkpoint_module,
        "current_code_binding",
        lambda: deepcopy(child_binding),
    )

    def replay(
        candidate: PilotCheckpoint,
        *,
        rng_preview_draws: int,
        strict_code_binding: bool,
    ) -> dict[str, Any]:
        assert candidate is checkpoint
        assert rng_preview_draws == 16
        assert strict_code_binding is False
        assert checkpoint_module.current_code_binding() != binding
        return deepcopy(frozen_exactness)

    monkeypatch.setattr(
        checkpoint_module,
        "verify_closed_loop_preflight_checkpoint",
        replay,
    )
    receipt = verify_historical_closed_loop_preflight_checkpoint(
        checkpoint,
        source_repo_root=parent_root,
        source_annotated_tag=tag,
        expected_tag_object=tag_object,
        expected_peeled_commit=peeled_commit,
        frozen_exactness_receipt=frozen_exactness,
    )

    assert receipt["historical_code_binding"]["head_consulted"] is False
    assert (
        receipt["historical_code_binding"]["working_tree_consulted"]
        is False
    )
    assert receipt["replay"] == {
        "strict_code_binding": False,
        "frozen_exactness_hash": canonical_hash(frozen_exactness),
        "replayed_exactness_hash": canonical_hash(frozen_exactness),
        "exact_match": True,
        "provider_calls_during_verification": 0,
    }
    assert receipt["exactness"] == frozen_exactness


def test_historical_gate_rejects_current_replay_exactness_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, tag, tag_object, peeled_commit = _tagged_source_repo(
        tmp_path / "parent"
    )
    checkpoint = _synthetic_checkpoint(binding)
    frozen_exactness = {
        "schema_version": "finevo-checkpoint-exactness-receipt-v1",
        "checkpoint_hash": checkpoint.checkpoint_hash,
        "provider_calls_during_verification": 0,
        "component_hashes": {"environment_hash": "1" * 64},
    }
    replayed_exactness = deepcopy(frozen_exactness)
    replayed_exactness["component_hashes"]["environment_hash"] = "2" * 64

    def replay(
        candidate: PilotCheckpoint,
        *,
        rng_preview_draws: int,
        strict_code_binding: bool,
    ) -> dict[str, Any]:
        assert candidate is checkpoint
        assert rng_preview_draws == 16
        assert strict_code_binding is False
        return deepcopy(replayed_exactness)

    monkeypatch.setattr(
        checkpoint_module,
        "verify_closed_loop_preflight_checkpoint",
        replay,
    )
    with pytest.raises(
        PilotCheckpointError,
        match="replay exactness differs",
    ):
        verify_historical_closed_loop_preflight_checkpoint(
            checkpoint,
            source_repo_root=tmp_path / "parent",
            source_annotated_tag=tag,
            expected_tag_object=tag_object,
            expected_peeled_commit=peeled_commit,
            frozen_exactness_receipt=frozen_exactness,
        )


def test_same_release_exactness_default_remains_strict() -> None:
    default = inspect.signature(
        verify_closed_loop_preflight_checkpoint
    ).parameters["strict_code_binding"].default
    assert default is True
