from pathlib import Path

import simulate_verified
from verified_memory.artifacts import verify_manifest
from verified_memory.failure_artifacts import verify_failure_receipt


def test_cli_diagnostic_creates_verified_manifest(tmp_path: Path) -> None:
    output = tmp_path / "diagnostic"
    code = simulate_verified.main(
        [
            "--provider",
            "diagnostic",
            "--run-id",
            "cli-diagnostic",
            "--episode-length",
            "5",
            "--output-dir",
            str(output),
        ]
    )
    assert code == 0
    assert verify_manifest(output).valid is True


def test_cli_rejects_accidental_large_or_underbudgeted_run(tmp_path: Path) -> None:
    assert simulate_verified.main(["--num-agents", "5"]) == 1
    assert simulate_verified.main(["--max-calls", "1"]) == 1


def test_cli_seals_budget_receipt_when_bounded_run_fails(tmp_path: Path) -> None:
    output = tmp_path / "failed-diagnostic"
    code = simulate_verified.main(
        [
            "--provider",
            "diagnostic",
            "--run-id",
            "cli-failure",
            "--max-prompt-tokens",
            "1",
            "--output-dir",
            str(output),
        ]
    )

    assert code == 1
    receipt = verify_failure_receipt(output)
    assert receipt["status"] == "failed"
    assert receipt["scope"] == "verified_simulation"
    assert receipt["budget_snapshot"]["completed_calls"] == 0
    assert receipt["partial_streams_persisted"] is False
