from __future__ import annotations

from pathlib import Path

import pytest

import run_pilot
from verified_memory import pilot_orchestrator as orchestrator


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_1.yaml"


def test_v2111_parent_import_requires_parent_repo_root() -> None:
    args = run_pilot.build_parser().parse_args(
        [
            "--contract",
            str(CONTRACT_PATH),
            "--stage",
            "parent-import",
        ]
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=r"V2\.11\.1 parent-import requires --parent-repo-root",
    ):
        run_pilot.execute(args)


def test_v2111_parent_repo_root_is_rejected_for_non_parent_stage(
    tmp_path: Path,
) -> None:
    args = run_pilot.build_parser().parse_args(
        [
            "--contract",
            str(CONTRACT_PATH),
            "--stage",
            "long-context-preflight",
            "--parent-repo-root",
            str(tmp_path / "pilot-v2.11-science"),
        ]
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="--parent-repo-root is accepted only for a parent-import stage",
    ):
        run_pilot.execute(args)


def test_v2111_development_fake_cli_is_provider_free_and_25_of_25(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("V2.11.1 development fake attempted a provider")

    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden)
    monkeypatch.setattr(run_pilot, "execute_stage", forbidden)
    args = run_pilot.build_parser().parse_args(
        [
            "--contract",
            str(CONTRACT_PATH),
            "--stage",
            "development-a-d",
            "--development-fake",
            "--raw-root",
            str(tmp_path),
        ]
    )

    result = run_pilot.execute(args)

    assert result["status"] == "pass"
    assert result["registered_cells"] == 25
    assert result["status_counts"] == {"complete": 25}
    assert result["diagnostic_only"] is True
    assert result["scientific_evidence"] is False
    assert Path(result["receipt"]).is_file()
