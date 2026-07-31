from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import run_pilot
from verified_memory.pilot_contract import PILOT_CONTRACT_ID_V2_11_5


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_5.yaml"


def _forbidden(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("V2.11.5 publish-evidence crossed its consumer boundary")


def test_v2115_publish_routes_only_to_exact_zero_provider_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "science" / "experiment_results/pilot-v2.11.5/raw"
    evidence_root = tmp_path / "publication" / "evidence"
    source_root = tmp_path / "science"
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        run_pilot,
        "load_pilot_contract",
        lambda _path: SimpleNamespace(
            contract_id=PILOT_CONTRACT_ID_V2_11_5,
            status="frozen",
        ),
    )

    def dedicated_builder(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            scientific_complete=False,
            package_dir=evidence_root / "current_v2/pilot-v2.11.5",
            manifest_path=evidence_root / "package_manifest.json",
            checksums_path=evidence_root / "checksums.json",
            contract_hash="a" * 64,
            claim_gates={"experiment_a": {"status": "no-go"}},
        )

    monkeypatch.setattr(
        run_pilot,
        "build_pilot_v2115_evidence_package",
        dedicated_builder,
    )
    monkeypatch.setattr(run_pilot, "build_pilot_v2112_evidence_package", _forbidden)
    monkeypatch.setattr(run_pilot, "build_pilot_v24_evidence_package", _forbidden)
    monkeypatch.setattr(run_pilot, "build_pilot_evidence_package", _forbidden)
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)
    args = run_pilot.build_parser().parse_args(
        [
            "--contract",
            str(CONTRACT_PATH),
            "--stage",
            "publish-evidence",
            "--raw-root",
            str(raw_root),
            "--source-repo-root",
            str(source_root),
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
            "source_repo_root": source_root,
        }
    ]
