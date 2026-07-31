from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import run_pilot
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.pilot_contract import PILOT_CONTRACT_ID_V2_11_5


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_5.yaml"
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
    raise AssertionError("V2.11.5 CLI crossed the requested boundary")


def _contract(status: str = "frozen") -> SimpleNamespace:
    return SimpleNamespace(contract_id=PILOT_CONTRACT_ID_V2_11_5, status=status)


def test_v2115_raw_namespace_is_fresh_and_exact() -> None:
    assert run_pilot._raw_root_for_contract(CONTRACT_PATH) == (
        ROOT / "experiment_results" / "pilot-v2.11.5" / "raw"
    )


@pytest.mark.parametrize(
    ("provided", "message"),
    (
        ((), r"V2\.11\.5 parent-import requires --parent-repo-root"),
        (
            ("parent",),
            r"V2\.11\.5 parent-import requires --authority-repo-root",
        ),
    ),
)
def test_v2115_parent_import_requires_both_immutable_roots_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provided: tuple[str, ...],
    message: str,
) -> None:
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)
    argv = ["--contract", str(CONTRACT_PATH), "--stage", "parent-import"]
    if "parent" in provided:
        argv.extend(["--parent-repo-root", str(tmp_path / "v2114")])

    with pytest.raises(orchestrator.PilotOrchestrationError, match=message):
        run_pilot.execute(run_pilot.build_parser().parse_args(argv))


@pytest.mark.parametrize("stage_id", REGISTERED_STAGES)
def test_v2115_draft_real_stages_stop_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_id: str,
) -> None:
    monkeypatch.setattr(
        run_pilot, "load_pilot_contract", lambda _path: _contract("draft")
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
        argv.extend(
            [
                "--parent-repo-root",
                str(tmp_path / "pilot-v2.11.4-science"),
                "--authority-repo-root",
                str(tmp_path / "pilot-v2.11.2-science"),
            ]
        )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=r"V2\.11\.5 real stages require a frozen contract",
    ):
        run_pilot.execute(run_pilot.build_parser().parse_args(argv))


@pytest.mark.parametrize("stage_id", REGISTERED_STAGES)
def test_v2115_frozen_stage_forwards_dual_roots_only_to_parent_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_id: str,
) -> None:
    raw_root = tmp_path / "experiment_results" / "pilot-v2.11.5" / "raw"
    parent_root = tmp_path / "pilot-v2.11.4-science"
    authority_root = tmp_path / "pilot-v2.11.2-science"
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())

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
        argv.extend(
            [
                "--parent-repo-root",
                str(parent_root),
                "--authority-repo-root",
                str(authority_root),
            ]
        )

    result = run_pilot.execute(run_pilot.build_parser().parse_args(argv))

    assert result == {"status": "complete", "provider_calls": 0}
    assert calls == [
        {
            "contract_path": CONTRACT_PATH,
            "stage_id": stage_id,
            "resume": True,
            "raw_root": raw_root,
            "repo_root": ROOT,
            "parent_repo_root": (
                parent_root if stage_id == "parent-import" else None
            ),
            "authority_repo_root": (
                authority_root if stage_id == "parent-import" else None
            ),
        }
    ]


def test_v2115_authority_root_is_rejected_outside_parent_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)
    args = run_pilot.build_parser().parse_args(
        [
            "--contract",
            str(CONTRACT_PATH),
            "--stage",
            "capability-gate",
            "--authority-repo-root",
            str(tmp_path / "pilot-v2.11.2-science"),
        ]
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="--authority-repo-root is accepted only for a parent-import stage",
    ):
        run_pilot.execute(args)


def test_v2115_draft_acceptance_stops_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_pilot,
        "load_pilot_contract",
        lambda _path: _contract("draft"),
    )
    monkeypatch.setattr(run_pilot, "accept_v2115_scientific_dispatch", _forbidden)

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=r"V2\.11\.5 real stages require a frozen contract",
    ):
        run_pilot.execute(
            run_pilot.build_parser().parse_args(
                [
                    "--contract",
                    str(CONTRACT_PATH),
                    "--accept-scientific-dispatch",
                    "--raw-root",
                    str(tmp_path / "raw"),
                ]
            )
        )


def test_v2115_frozen_acceptance_dispatches_exact_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "experiment_results" / "pilot-v2.11.5" / "raw"
    launch = raw_root / "scientific_launch_input.json"
    output = raw_root / "scientific_dispatch_acceptance.json"
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())
    monkeypatch.setattr(run_pilot, "accept_v2113_scientific_dispatch", _forbidden)
    monkeypatch.setattr(run_pilot, "accept_v2114_scientific_dispatch", _forbidden)

    def fake_acceptance(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "go", "go": True, "provider_calls": 0}

    monkeypatch.setattr(
        run_pilot,
        "accept_v2115_scientific_dispatch",
        fake_acceptance,
    )

    result = run_pilot.execute(
        run_pilot.build_parser().parse_args(
            [
                "--contract",
                str(CONTRACT_PATH),
                "--accept-scientific-dispatch",
                "--raw-root",
                str(raw_root),
                "--scientific-launch-input",
                str(launch),
                "--acceptance-output",
                str(output),
            ]
        )
    )

    assert result == {"status": "go", "go": True, "provider_calls": 0}
    assert calls == [
        {
            "contract_path": CONTRACT_PATH,
            "repo_root": ROOT,
            "raw_root": raw_root,
            "scientific_launch_input_path": launch,
            "receipt_path": output,
        }
    ]
