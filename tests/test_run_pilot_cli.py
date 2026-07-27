from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_pilot
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_V2_3_CANONICAL_SHA256,
    load_pilot_contract,
)
from verified_memory.pilot_evidence import PilotEvidencePackage
from verified_memory.pilot_orchestrator import PilotOrchestrationError


def _args(tmp_path: Path, **updates) -> Namespace:
    values = {
        "contract": tmp_path / "pilot.yaml",
        "stage": "publish-evidence",
        "resume": True,
        "development_fake": False,
        "raw_root": tmp_path / "raw",
        "evidence_root": tmp_path / "evidence",
    }
    values.update(updates)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("scientific_complete", "expected_status"),
    [(True, "complete"), (False, "complete-with-no-go")],
)
def test_publish_evidence_uses_strict_builder_without_provider_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_complete: bool,
    expected_status: str,
) -> None:
    calls = []
    package_dir = tmp_path / "evidence" / "current_v2" / "pilot-v1"

    def fake_build(**kwargs):
        calls.append(kwargs)
        return PilotEvidencePackage(
            package_dir=package_dir,
            manifest_path=package_dir / "manifest.json",
            checksums_path=package_dir / "checksums.json",
            contract_hash="a" * 64,
            scientific_complete=scientific_complete,
            claim_gates={"experiment_a": {"supported": scientific_complete}},
        )

    def forbidden_dispatch(**_kwargs):
        raise AssertionError("publish-evidence must not dispatch a pilot stage")

    monkeypatch.setattr(run_pilot, "build_pilot_evidence_package", fake_build)
    monkeypatch.setattr(
        run_pilot,
        "load_pilot_contract",
        lambda _path: SimpleNamespace(contract_id="finevo-pilot-v2.3"),
    )
    monkeypatch.setattr(run_pilot, "execute_stage", forbidden_dispatch)

    args = _args(tmp_path)
    result = run_pilot.execute(args)

    assert calls == [
        {
            "contract_path": args.contract,
            "run_ledger_path": args.raw_root / "run_ledger.json",
            "raw_root": args.raw_root,
            "build_root": args.evidence_root,
        }
    ]
    assert result["status"] == expected_status
    assert result["provider_calls"] == 0
    assert result["scientific_complete"] is scientific_complete
    assert result["contract_sha256"] == "a" * 64


def test_v24_publish_evidence_selects_lane_separated_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    package_dir = tmp_path / "evidence" / "current_v2" / "pilot-v2.4"

    def fake_v24_build(**kwargs):
        calls.append(kwargs)
        return PilotEvidencePackage(
            package_dir=package_dir,
            manifest_path=package_dir / "package_manifest.json",
            checksums_path=package_dir / "checksums.json",
            contract_hash="b" * 64,
            scientific_complete=False,
            claim_gates={
                "lanes": {
                    "local": {"status": "supported"},
                    "gpt52": {"status": "no-go"},
                }
            },
        )

    monkeypatch.setattr(
        run_pilot,
        "load_pilot_contract",
        lambda _path: SimpleNamespace(
            contract_id=run_pilot.PILOT_V24_CONTRACT_ID
        ),
    )
    monkeypatch.setattr(
        run_pilot,
        "build_pilot_v24_evidence_package",
        fake_v24_build,
    )
    monkeypatch.setattr(
        run_pilot,
        "build_pilot_evidence_package",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("V2.4 must not use the V1-V2.3 evidence builder")
        ),
    )
    monkeypatch.setattr(
        run_pilot,
        "execute_stage",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("publish-evidence must not dispatch a pilot stage")
        ),
    )

    args = _args(tmp_path)
    result = run_pilot.execute(args)

    assert calls == [
        {
            "contract_path": args.contract,
            "run_ledger_path": args.raw_root / "run_ledger.json",
            "raw_root": args.raw_root,
            "build_root": args.evidence_root,
        }
    ]
    assert result["status"] == "complete-with-no-go"
    assert result["provider_calls"] == 0
    assert set(result["claim_gates"]["lanes"]) == {"local", "gpt52"}


def test_v25_publish_evidence_selects_lane_separated_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    package_dir = tmp_path / "evidence" / "current_v2" / "pilot-v2.5"

    def fake_lane_build(**kwargs):
        calls.append(kwargs)
        return PilotEvidencePackage(
            package_dir=package_dir,
            manifest_path=package_dir / "package_manifest.json",
            checksums_path=package_dir / "checksums.json",
            contract_hash="c" * 64,
            scientific_complete=False,
            claim_gates={"lanes": {"local": {}, "gpt52": {}}},
        )

    monkeypatch.setattr(
        run_pilot,
        "load_pilot_contract",
        lambda _path: SimpleNamespace(contract_id=run_pilot.V25_CONTRACT_ID),
    )
    monkeypatch.setattr(
        run_pilot,
        "build_pilot_v24_evidence_package",
        fake_lane_build,
    )
    monkeypatch.setattr(
        run_pilot,
        "build_pilot_evidence_package",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("V2.5 must use the lane-separated evidence builder")
        ),
    )
    monkeypatch.setattr(
        run_pilot,
        "execute_stage",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("publish-evidence must not dispatch a pilot stage")
        ),
    )

    args = _args(tmp_path)
    result = run_pilot.execute(args)

    assert calls == [
        {
            "contract_path": args.contract,
            "run_ledger_path": args.raw_root / "run_ledger.json",
            "raw_root": args.raw_root,
            "build_root": args.evidence_root,
        }
    ]
    assert result["status"] == "complete-with-no-go"
    assert result["provider_calls"] == 0


def test_development_matrix_still_requires_explicit_fake_flag(
    tmp_path: Path,
) -> None:
    with pytest.raises(PilotOrchestrationError, match="explicit"):
        run_pilot.execute(
            _args(
                tmp_path,
                stage="development-a-d",
                development_fake=False,
            )
        )


def test_parser_exposes_a_separate_evidence_root(tmp_path: Path) -> None:
    parsed = run_pilot.build_parser().parse_args(
        [
            "--contract",
            str(tmp_path / "pilot.yaml"),
            "--stage",
            "publish-evidence",
            "--raw-root",
            str(tmp_path / "raw"),
            "--evidence-root",
            str(tmp_path / "reviewer-evidence"),
            "--resume",
        ]
    )

    assert parsed.stage == "publish-evidence"
    assert parsed.raw_root == tmp_path / "raw"
    assert parsed.evidence_root == tmp_path / "reviewer-evidence"
    assert parsed.resume is True


def test_parser_defaults_to_the_frozen_v2_3_preflight_amendment() -> None:
    parsed = run_pilot.build_parser().parse_args(
        ["--stage", "capability-gate"]
    )

    expected_path = (
        Path(run_pilot.__file__).resolve().parent
        / "experiments"
        / "pilot_v2_3.yaml"
    )
    overlay_path = expected_path.with_name("pilot_v2_3_overlay.yaml")
    assert parsed.contract == expected_path

    full = load_pilot_contract(parsed.contract)
    overlay = load_pilot_contract(overlay_path)
    assert full.status == overlay.status == "frozen"
    assert full.canonical_hash == overlay.canonical_hash == (
        PILOT_CONTRACT_V2_3_CANONICAL_SHA256
    )
    assert full.to_dict() == overlay.to_dict()
