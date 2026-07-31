from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import run_pilot
from scripts.render_pilot_v2112_contract import build_contract
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2_11_2,
    load_pilot_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_2.yaml"
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
    raise AssertionError("V2.11.2 CLI crossed a provider or execution boundary")


def _draft_contract_path(tmp_path: Path) -> Path:
    path = tmp_path / "pilot_v2_11_2.draft.json"
    manifest = ROOT / "experiments" / "pilot_v2_11_2_source_manifest.json"
    (tmp_path / manifest.name).write_bytes(manifest.read_bytes())
    path.write_text(
        json.dumps(build_contract(ROOT, status="draft"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_v2112_contract_id_stage_inventory_and_raw_namespace_are_exact() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)

    assert contract.contract_id == PILOT_CONTRACT_ID_V2_11_2
    assert contract.status == "frozen"
    assert tuple(stage.stage_id for stage in contract.stages) == REGISTERED_STAGES
    assert run_pilot._raw_root_for_contract(CONTRACT_PATH) == (
        ROOT / "experiment_results" / "pilot-v2.11.2" / "raw"
    )


def test_v2112_parent_import_requires_parent_repo_root_before_dispatch(
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
        match=r"V2\.11\.2 parent-import requires --parent-repo-root",
    ):
        run_pilot.execute(args)


def test_v2112_parent_repo_root_is_rejected_outside_parent_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)
    args = run_pilot.build_parser().parse_args(
        [
            "--contract",
            str(CONTRACT_PATH),
            "--stage",
            "long-context-preflight",
            "--parent-repo-root",
            str(tmp_path / "pilot-v2.11.1-science"),
        ]
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="--parent-repo-root is accepted only for a parent-import stage",
    ):
        run_pilot.execute(args)


@pytest.mark.parametrize("stage_id", REGISTERED_STAGES)
def test_v2112_draft_real_stages_fail_closed_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_id: str,
) -> None:
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)
    draft_contract = _draft_contract_path(tmp_path)
    argv = [
        "--contract",
        str(draft_contract),
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
                str(tmp_path / "pilot-v2.11.1-science"),
            ]
        )
    args = run_pilot.build_parser().parse_args(argv)

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=r"V2\.11\.2 real stages require a frozen contract",
    ):
        run_pilot.execute(args)


def test_v2112_draft_publish_is_also_blocked_before_artifact_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)
    monkeypatch.setattr(run_pilot, "build_pilot_evidence_package", _forbidden)
    monkeypatch.setattr(
        run_pilot,
        "build_pilot_v24_evidence_package",
        _forbidden,
    )
    draft_contract = _draft_contract_path(tmp_path)
    args = run_pilot.build_parser().parse_args(
        [
            "--contract",
            str(draft_contract),
            "--stage",
            "publish-evidence",
            "--raw-root",
            str(tmp_path / "raw"),
            "--evidence-root",
            str(tmp_path / "evidence"),
            "--resume",
        ]
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=r"V2\.11\.2 real stages require a frozen contract",
    ):
        run_pilot.execute(args)


def test_v2112_frozen_publish_routes_to_dedicated_evidence_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    evidence_root = tmp_path / "evidence"
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        run_pilot,
        "load_pilot_contract",
        lambda _path: SimpleNamespace(
            contract_id=PILOT_CONTRACT_ID_V2_11_2,
            status="frozen",
        ),
    )

    def dedicated_builder(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            scientific_complete=False,
            package_dir=evidence_root / "current_v2" / "pilot-v2.11.2",
            manifest_path=evidence_root / "package_manifest.json",
            checksums_path=evidence_root / "checksums.json",
            contract_hash="a" * 64,
            claim_gates={"experiment_a": {"status": "no-go"}},
        )

    monkeypatch.setattr(
        run_pilot,
        "build_pilot_v2112_evidence_package",
        dedicated_builder,
    )
    monkeypatch.setattr(run_pilot, "build_pilot_evidence_package", _forbidden)
    monkeypatch.setattr(run_pilot, "build_pilot_v24_evidence_package", _forbidden)
    args = run_pilot.build_parser().parse_args(
        [
            "--contract",
            str(CONTRACT_PATH),
            "--stage",
            "publish-evidence",
            "--raw-root",
            str(raw_root),
            "--evidence-root",
            str(evidence_root),
            "--resume",
        ]
    )

    result = run_pilot.execute(args)

    assert result["status"] == "complete-with-no-go"
    assert result["provider_calls"] == 0
    assert calls == [
        {
            "contract_path": CONTRACT_PATH,
            "run_ledger_path": raw_root / "run_ledger.json",
            "raw_root": raw_root,
            "build_root": evidence_root,
        }
    ]


@pytest.mark.parametrize("stage_id", REGISTERED_STAGES)
def test_v2112_frozen_stage_and_resume_are_forwarded_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_id: str,
) -> None:
    raw_root = tmp_path / "experiment_results" / "pilot-v2.11.2" / "raw"
    parent_root = tmp_path / "pilot-v2.11.1-science"
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        run_pilot,
        "load_pilot_contract",
        lambda _path: SimpleNamespace(
            contract_id=PILOT_CONTRACT_ID_V2_11_2,
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
    args = run_pilot.build_parser().parse_args(argv)

    assert run_pilot.execute(args) == {
        "status": "complete",
        "provider_calls": 0,
    }
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


def test_v2112_draft_development_fake_is_provider_free_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orchestrator, "_provider_for_profile", _forbidden)
    monkeypatch.setattr(orchestrator, "create_llm_provider", _forbidden)
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)
    draft_contract = _draft_contract_path(tmp_path)
    args = run_pilot.build_parser().parse_args(
        [
            "--contract",
            str(draft_contract),
            "--stage",
            "development-a-d",
            "--development-fake",
            "--raw-root",
            str(tmp_path),
            "--resume",
        ]
    )

    result = run_pilot.execute(args)

    assert result["status"] == "pass"
    assert result["registered_cells"] == 25
    assert result["status_counts"] == {"complete": 25}
    assert result["stages"] == [
        "experiment-a",
        "experiment-b",
        "experiment-c",
        "experiment-d",
    ]
    assert result["diagnostic_only"] is True
    assert result["scientific_evidence"] is False
    receipt = Path(result["receipt"])
    assert receipt.is_file()
    before = receipt.read_bytes()

    resumed = run_pilot.execute(args)

    assert resumed == result
    assert receipt.read_bytes() == before
