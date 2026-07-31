from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import run_pilot
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.pilot_contract import PILOT_CONTRACT_ID_V2_11_3


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_3.yaml"
REGISTERED_STAGES = (
    "parent-import",
    "capability-gate",
    "long-context-preflight",
    "experiment-c",
    "experiment-a",
    "experiment-d",
    "experiment-b",
    "cross-model",
)


def _forbidden(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("V2.11.3 CLI crossed the requested boundary")


def test_v2113_raw_namespace_is_fresh_and_exact() -> None:
    assert run_pilot._raw_root_for_contract(CONTRACT_PATH) == (
        ROOT / "experiment_results" / "pilot-v2.11.3" / "raw"
    )


def test_v2113_parent_import_requires_immutable_parent_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)
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
        match=r"V2\.11\.3 parent-import requires --parent-repo-root",
    ):
        run_pilot.execute(args)


@pytest.mark.parametrize("stage_id", REGISTERED_STAGES)
def test_v2113_draft_real_stages_stop_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_id: str,
) -> None:
    monkeypatch.setattr(
        run_pilot,
        "load_pilot_contract",
        lambda _path: SimpleNamespace(
            contract_id=PILOT_CONTRACT_ID_V2_11_3,
            status="draft",
        ),
    )
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)
    argv = [
        "--contract",
        str(CONTRACT_PATH),
        "--stage",
        stage_id,
        "--raw-root",
        str(tmp_path / "raw"),
        "--resume",
    ]
    if stage_id == "parent-import":
        argv.extend(["--parent-repo-root", str(tmp_path / "pilot-v2.11.2-science")])

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=r"V2\.11\.3 real stages require a frozen contract",
    ):
        run_pilot.execute(run_pilot.build_parser().parse_args(argv))


@pytest.mark.parametrize("stage_id", REGISTERED_STAGES)
def test_v2113_frozen_stage_and_resume_forward_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_id: str,
) -> None:
    raw_root = tmp_path / "experiment_results" / "pilot-v2.11.3" / "raw"
    parent_root = tmp_path / "pilot-v2.11.2-science"
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        run_pilot,
        "load_pilot_contract",
        lambda _path: SimpleNamespace(
            contract_id=PILOT_CONTRACT_ID_V2_11_3,
            status="frozen",
        ),
    )

    def fake_execute_stage(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "complete", "provider_calls": 0}

    monkeypatch.setattr(run_pilot, "execute_stage", fake_execute_stage)
    argv = [
        "--contract",
        str(CONTRACT_PATH),
        "--stage",
        stage_id,
        "--raw-root",
        str(raw_root),
        "--resume",
    ]
    if stage_id == "parent-import":
        argv.extend(["--parent-repo-root", str(parent_root)])

    result = run_pilot.execute(run_pilot.build_parser().parse_args(argv))

    assert result == {"status": "complete", "provider_calls": 0}
    assert calls == [
        {
            "contract_path": CONTRACT_PATH,
            "stage_id": stage_id,
            "resume": True,
            "raw_root": raw_root,
            "repo_root": ROOT,
            "parent_repo_root": (parent_root if stage_id == "parent-import" else None),
        }
    ]
