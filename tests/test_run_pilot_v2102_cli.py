from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import run_pilot
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_evidence import PilotEvidencePackage


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_10_2.yaml"


def test_v2102_contract_maps_to_its_fresh_raw_namespace() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)

    assert contract.contract_id == run_pilot.V2102_CONTRACT_ID
    assert run_pilot._raw_root_for_contract(CONTRACT_PATH) == (
        ROOT / "experiment_results" / "pilot-v2.10.2" / "raw"
    )


@pytest.mark.parametrize(
    "stage_id",
    [
        "parent-import",
        "q-ref-resolution",
        "stage0-calibration",
        "local-experiment-c",
        "local-experiment-a",
        "local-experiment-d",
        "local-experiment-b",
        "experiment-c",
        "experiment-a",
        "experiment-d",
        "experiment-b",
    ],
)
def test_v2102_registered_stage_and_resume_are_forwarded_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_id: str,
) -> None:
    raw_root = tmp_path / "experiment_results" / "pilot-v2.10.2" / "raw"
    parent_root = tmp_path / "pilot-v2.10.1"
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
    calls: list[dict[str, Any]] = []

    def fake_execute_stage(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "complete", "provider_calls": 0}

    monkeypatch.setattr(run_pilot, "execute_stage", fake_execute_stage)

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
            "parent_repo_root": (
                parent_root if stage_id == "parent-import" else None
            ),
        }
    ]


def test_v2102_publish_evidence_routes_to_lane_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_dir = tmp_path / "evidence" / "current_v2" / "pilot-v2.10.2"
    calls: list[dict[str, Any]] = []

    def fake_lane_builder(**kwargs: Any) -> PilotEvidencePackage:
        calls.append(kwargs)
        return PilotEvidencePackage(
            package_dir=package_dir,
            manifest_path=package_dir / "package_manifest.json",
            checksums_path=package_dir / "checksums.json",
            contract_hash="e" * 64,
            scientific_complete=False,
            claim_gates={"lanes": {"local": {}, "gpt52": {}}},
        )

    def forbidden(**_kwargs: Any) -> None:
        raise AssertionError("V2.10.2 publish must remain zero-provider")

    monkeypatch.setattr(
        run_pilot,
        "load_pilot_contract",
        lambda _path: SimpleNamespace(contract_id=run_pilot.V2102_CONTRACT_ID),
    )
    monkeypatch.setattr(
        run_pilot,
        "build_pilot_v24_evidence_package",
        fake_lane_builder,
    )
    monkeypatch.setattr(run_pilot, "build_pilot_evidence_package", forbidden)
    monkeypatch.setattr(run_pilot, "execute_stage", forbidden)

    raw_root = tmp_path / "raw"
    evidence_root = tmp_path / "evidence"
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

    assert calls == [
        {
            "contract_path": CONTRACT_PATH,
            "run_ledger_path": raw_root / "run_ledger.json",
            "raw_root": raw_root,
            "build_root": evidence_root,
        }
    ]
    assert result["status"] == "complete-with-no-go"
    assert result["provider_calls"] == 0
    assert result["contract_sha256"] == "e" * 64


def test_v2102_development_fake_preserves_resume_and_contract_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "diagnostic"
    calls: list[dict[str, Any]] = []

    def fake_development(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "complete", "provider_calls": 0}

    monkeypatch.setattr(
        run_pilot,
        "run_development_fake_matrix",
        fake_development,
    )
    args = run_pilot.build_parser().parse_args(
        [
            "--contract",
            str(CONTRACT_PATH),
            "--stage",
            "development-a-d",
            "--development-fake",
            "--raw-root",
            str(raw_root),
            "--resume",
        ]
    )

    assert run_pilot.execute(args) == {
        "status": "complete",
        "provider_calls": 0,
    }
    assert calls == [
        {
            "contract_path": CONTRACT_PATH,
            "resume": True,
            "raw_root": raw_root,
        }
    ]
