from pathlib import Path

import pytest

from verified_memory.failure_artifacts import (
    verify_failure_receipt,
    write_failure_receipt,
)


def test_failure_receipt_is_content_addressed_and_tamper_evident(tmp_path: Path) -> None:
    write_failure_receipt(
        tmp_path,
        scope="unit",
        error=RuntimeError("bounded failure"),
        budget_snapshot={"completed_calls": 1},
        config={"seed": 7},
        provenance={"scientific_evidence": False},
        git_commit="abc123",
        git_dirty=False,
    )

    receipt = verify_failure_receipt(tmp_path)
    assert receipt["error"]["type"] == "RuntimeError"
    assert receipt["budget_snapshot"]["completed_calls"] == 1

    failure_path = tmp_path / "failure.json"
    failure_path.write_text(failure_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_failure_receipt(tmp_path)
