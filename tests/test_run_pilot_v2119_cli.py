from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import run_pilot
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.pilot_contract import PILOT_CONTRACT_ID_V2_11_9


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_9.yaml"


def _contract(status: str = "frozen") -> SimpleNamespace:
    return SimpleNamespace(contract_id=PILOT_CONTRACT_ID_V2_11_9, status=status)


def _args(*values: str) -> Any:
    return run_pilot.build_parser().parse_args(
        ["--contract", str(CONTRACT_PATH), *values]
    )


def _forbidden(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("V2.11.9 CLI crossed its fail-closed boundary")


@pytest.mark.parametrize(
    ("extra", "message"),
    (
        ((), r"V2\.11\.9 parent-import requires --failed-repo-root"),
        (
            ("--failed-repo-root", "failed-v2118"),
            r"V2\.11\.9 parent-import requires --authority-repo-root",
        ),
    ),
)
def test_v2119_parent_import_requires_both_source_roots_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    extra: tuple[str, ...],
    message: str,
) -> None:
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)

    with pytest.raises(orchestrator.PilotOrchestrationError, match=message):
        run_pilot.execute(_args("--stage", "parent-import", *extra))


def test_v2119_parent_import_forwards_failed_and_authority_roots_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "experiment_results/pilot-v2.11.9/raw"
    failed_root = tmp_path / "pilot-v2.11.8-science"
    authority_root = tmp_path / "pilot-v2.11.5-science"
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())

    def execute_stage(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "complete", "provider_calls": 0}

    monkeypatch.setattr(run_pilot, "execute_stage", execute_stage)
    result = run_pilot.execute(
        _args(
            "--stage",
            "parent-import",
            "--raw-root",
            str(raw_root),
            "--failed-repo-root",
            str(failed_root),
            "--authority-repo-root",
            str(authority_root),
        )
    )

    assert result == {"status": "complete", "provider_calls": 0}
    assert calls == [
        {
            "contract_path": CONTRACT_PATH,
            "stage_id": "parent-import",
            "resume": False,
            "raw_root": raw_root,
            "repo_root": ROOT,
            "parent_repo_root": failed_root,
            "authority_repo_root": authority_root,
        }
    ]


def test_v2119_rejects_legacy_parent_root_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=r"V2\.11\.9 parent-import uses --failed-repo-root",
    ):
        run_pilot.execute(
            _args(
                "--stage",
                "parent-import",
                "--parent-repo-root",
                str(tmp_path / "legacy-alias"),
                "--failed-repo-root",
                str(tmp_path / "failed-v2118"),
                "--authority-repo-root",
                str(tmp_path / "authority-v2115"),
            )
        )


@pytest.mark.parametrize(
    ("flag", "message"),
    (
        ("--failed-repo-root", "accepted only for a parent-import stage"),
        ("--authority-repo-root", "accepted only for a parent-import stage"),
    ),
)
def test_v2119_source_roots_are_rejected_outside_parent_import(
    flag: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)

    with pytest.raises(orchestrator.PilotOrchestrationError, match=message):
        run_pilot.execute(
            _args(
                "--stage",
                "experiment-d",
                flag,
                str(tmp_path / "forbidden-source-root"),
            )
        )


def test_v2119_draft_real_stage_fails_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_pilot, "load_pilot_contract", lambda _path: _contract("draft")
    )
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=r"V2\.11\.9 real stages require a frozen contract",
    ):
        run_pilot.execute(
            _args("--stage", "experiment-d", "--raw-root", str(tmp_path / "raw"))
        )


@pytest.mark.parametrize("stage", ("experiment-d", "experiment-b", "cross-model"))
def test_v2119_scientific_stages_dispatch_to_fresh_raw_root(
    stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "experiment_results/pilot-v2.11.9/raw"
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())

    def execute_stage(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(run_pilot, "execute_stage", execute_stage)
    assert run_pilot.execute(_args("--stage", stage, "--raw-root", str(raw_root))) == {
        "status": "complete"
    }
    assert calls == [
        {
            "contract_path": CONTRACT_PATH,
            "stage_id": stage,
            "resume": False,
            "raw_root": raw_root,
            "repo_root": ROOT,
            "parent_repo_root": None,
            "authority_repo_root": None,
        }
    ]


def test_v2119_acceptance_routes_only_to_zero_provider_acceptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "experiment_results/pilot-v2.11.9/raw"
    launch = raw_root / "scientific_launch_input.json"
    output = raw_root / "scientific_dispatch_acceptance.json"
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)

    def accept(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "go", "go": True, "provider_calls": 0}

    monkeypatch.setattr(run_pilot, "accept_v2119_scientific_dispatch", accept)
    result = run_pilot.execute(
        _args(
            "--accept-scientific-dispatch",
            "--raw-root",
            str(raw_root),
            "--scientific-launch-input",
            str(launch),
            "--acceptance-output",
            str(output),
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


def test_v2119_acceptance_rejects_development_fake_without_assertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())
    monkeypatch.setattr(run_pilot, "accept_v2119_scientific_dispatch", _forbidden)
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=r"incompatible with --development-fake",
    ):
        run_pilot.execute(
            _args("--accept-scientific-dispatch", "--development-fake")
        )


@pytest.mark.parametrize(
    ("provided", "message"),
    (
        (
            ("--authority-repo-root", "authority-v2115"),
            r"requires --source-repo-root",
        ),
        (
            ("--source-repo-root", "science-v2119"),
            r"requires --authority-repo-root",
        ),
    ),
)
def test_v2119_publish_evidence_requires_both_release_roots(
    provided: tuple[str, str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())
    monkeypatch.setattr(run_pilot, "build_pilot_v2119_evidence_package", _forbidden)

    with pytest.raises(orchestrator.PilotOrchestrationError, match=message):
        run_pilot.execute(_args("--stage", "publish-evidence", *provided))


def test_v2119_publish_evidence_routes_only_to_dedicated_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "science-v2119/experiment_results/pilot-v2.11.9/raw"
    source_root = tmp_path / "science-v2119"
    authority_root = tmp_path / "authority-v2115"
    evidence_root = tmp_path / "evidence"
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())
    for name in (
        "build_pilot_evidence_package",
        "build_pilot_v24_evidence_package",
        "build_pilot_v2112_evidence_package",
        "build_pilot_v2115_evidence_package",
    ):
        monkeypatch.setattr(run_pilot, name, _forbidden)

    package_dir = evidence_root / "current_v2/pilot-v2.11.9"

    def build(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            package_dir=package_dir,
            manifest_path=package_dir / "package_manifest.json",
            checksums_path=package_dir / "checksums.json",
            contract_hash="a" * 64,
            scientific_complete=False,
            claim_gates={"experiment_a": {"status": "no-go"}},
        )

    monkeypatch.setattr(run_pilot, "build_pilot_v2119_evidence_package", build)
    result = run_pilot.execute(
        _args(
            "--stage",
            "publish-evidence",
            "--raw-root",
            str(raw_root),
            "--source-repo-root",
            str(source_root),
            "--authority-repo-root",
            str(authority_root),
            "--evidence-root",
            str(evidence_root),
        )
    )

    assert calls == [
        {
            "contract_path": CONTRACT_PATH,
            "run_ledger_path": raw_root / "run_ledger.json",
            "raw_root": raw_root,
            "build_root": evidence_root,
            "source_repo_root": source_root,
            "authority_repo_root": authority_root,
        }
    ]
    assert result == {
        "status": "complete-with-no-go",
        "provider_calls": 0,
        "package_dir": str(package_dir),
        "manifest_path": str(package_dir / "package_manifest.json"),
        "checksums_path": str(package_dir / "checksums.json"),
        "contract_sha256": "a" * 64,
        "scientific_complete": False,
        "claim_gates": {"experiment_a": {"status": "no-go"}},
    }


def test_non_v2119_publish_evidence_rejects_authority_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_pilot,
        "load_pilot_contract",
        lambda _path: SimpleNamespace(
            contract_id="finevo-pilot-v2.11.5",
            status="frozen",
        ),
    )
    monkeypatch.setattr(run_pilot, "build_pilot_v2115_evidence_package", _forbidden)

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=r"accepted by publish-evidence only for V2\.11\.9",
    ):
        run_pilot.execute(
            _args(
                "--stage",
                "publish-evidence",
                "--authority-repo-root",
                str(tmp_path / "forbidden-authority"),
            )
        )


def test_v2119_provider_key_guard_precedes_release_or_provider_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GuardReached(RuntimeError):
        pass

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _path: _contract())
    monkeypatch.setattr(
        orchestrator,
        "require_v2119_provider_keys_absent",
        lambda: (_ for _ in ()).throw(GuardReached("guard reached")),
    )
    monkeypatch.setattr(orchestrator, "verify_paid_provenance", _forbidden)

    with pytest.raises(GuardReached, match="guard reached"):
        orchestrator._execute_stage_locked(
            contract_path=CONTRACT_PATH,
            stage_id="parent-import",
            resume=False,
            raw_root=tmp_path / "raw",
            repo_root=ROOT,
            parent_repo_root=tmp_path / "failed-v2118",
            authority_repo_root=tmp_path / "authority-v2115",
        )
