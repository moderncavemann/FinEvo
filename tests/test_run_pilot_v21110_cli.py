from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import run_pilot
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.pilot_contract import PILOT_CONTRACT_ID_V2_11_10


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_10.yaml"


@pytest.fixture(autouse=True)
def _provider_credentials_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def _contract(status: str = "frozen") -> SimpleNamespace:
    return SimpleNamespace(contract_id=PILOT_CONTRACT_ID_V2_11_10, status=status)


def _args(*values: str) -> Any:
    return run_pilot.build_parser().parse_args(
        ["--contract", str(CONTRACT_PATH), *values]
    )


def _forbidden(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("V2.11.10 CLI crossed its provider-free boundary")


@pytest.mark.parametrize(
    ("extra", "message"),
    (
        ((), r"V2\.11\.10 parent-import requires --failed-repo-root"),
        (
            ("--failed-repo-root", "failed-v2119"),
            r"V2\.11\.10 parent-import requires --authority-repo-root",
        ),
    ),
)
def test_v21110_parent_import_requires_failed_and_authority_roots(
    monkeypatch: pytest.MonkeyPatch,
    extra: tuple[str, ...],
    message: str,
) -> None:
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)

    with pytest.raises(orchestrator.PilotOrchestrationError, match=message):
        run_pilot.execute(_args("--stage", "parent-import", *extra))


def test_v21110_parent_import_forwards_v2119_and_v2115_roots_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "experiment_results/pilot-v2.11.10/raw"
    failed_root = tmp_path / "pilot-v2.11.9-science"
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


def test_v21110_parent_import_rejects_legacy_parent_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=r"V2\.11\.10 parent-import uses --failed-repo-root",
    ):
        run_pilot.execute(
            _args(
                "--stage",
                "parent-import",
                "--parent-repo-root",
                str(tmp_path / "legacy-parent"),
                "--failed-repo-root",
                str(tmp_path / "failed-v2119"),
                "--authority-repo-root",
                str(tmp_path / "authority-v2115"),
            )
        )


@pytest.mark.parametrize(
    "flag",
    ("--failed-repo-root", "--authority-repo-root"),
)
def test_v21110_source_roots_are_rejected_for_paid_stages(
    flag: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="accepted only for a parent-import stage",
    ):
        run_pilot.execute(
            _args(
                "--stage",
                "experiment-d",
                flag,
                str(tmp_path / "forbidden-source-root"),
            )
        )


@pytest.mark.parametrize("stage", ("experiment-d", "experiment-b", "cross-model"))
def test_v21110_draft_paid_stages_fail_before_dispatch(
    stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_pilot, "load_pilot_contract", lambda _path: _contract("draft")
    )
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=r"V2\.11\.10 real stages require a frozen contract",
    ):
        run_pilot.execute(
            _args("--stage", stage, "--raw-root", str(tmp_path / "raw"))
        )


@pytest.mark.parametrize("stage", ("experiment-d", "experiment-b", "cross-model"))
def test_v21110_frozen_scientific_stage_uses_fresh_namespace_without_parent_roots(
    stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "experiment_results/pilot-v2.11.10/raw"
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


def test_v21110_acceptance_routes_only_to_v21110_acceptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "experiment_results/pilot-v2.11.10/raw"
    launch = raw_root / "scientific_launch_input.json"
    output = raw_root / "scientific_dispatch_acceptance.json"
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())
    monkeypatch.setattr(run_pilot, "accept_v2119_scientific_dispatch", _forbidden)
    monkeypatch.setattr(run_pilot, "execute_stage", _forbidden)

    def accept(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "go", "go": True, "provider_calls": 0}

    monkeypatch.setattr(run_pilot, "accept_v21110_scientific_dispatch", accept)
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


def test_v21110_draft_acceptance_fails_before_acceptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_pilot, "load_pilot_contract", lambda _path: _contract("draft")
    )
    monkeypatch.setattr(run_pilot, "accept_v21110_scientific_dispatch", _forbidden)

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=r"V2\.11\.10 real stages require a frozen contract",
    ):
        run_pilot.execute(_args("--accept-scientific-dispatch"))


@pytest.mark.parametrize(
    ("provided", "message"),
    (
        ((), r"requires --source-repo-root"),
        (
            ("--source-repo-root", "science-v21110"),
            r"requires --failed-repo-root",
        ),
        (
            (
                "--source-repo-root",
                "science-v21110",
                "--failed-repo-root",
                "failed-v2119",
            ),
            r"requires --authority-repo-root",
        ),
    ),
)
def test_v21110_publish_evidence_requires_all_three_release_roots(
    provided: tuple[str, ...],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())
    monkeypatch.setattr(run_pilot, "build_pilot_v21110_evidence_package", _forbidden)

    with pytest.raises(orchestrator.PilotOrchestrationError, match=message):
        run_pilot.execute(_args("--stage", "publish-evidence", *provided))
    if "--failed-repo-root" in provided:
        with pytest.raises(
            orchestrator.PilotOrchestrationError,
            match=r"requires exactly two --publication-ci-receipt",
        ):
            run_pilot.execute(
                _args(
                    "--stage",
                    "publish-evidence",
                    *provided,
                    "--authority-repo-root",
                    "authority-v2115",
                )
            )


def test_v21110_publish_evidence_routes_only_to_dedicated_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "science-v21110"
    raw_root = source_root / "experiment_results/pilot-v2.11.10/raw"
    failed_root = tmp_path / "failed-v2119"
    authority_root = tmp_path / "authority-v2115"
    evidence_root = tmp_path / "evidence"
    ci_receipts = (tmp_path / "linux.json", tmp_path / "macos.json")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(run_pilot, "load_pilot_contract", lambda _path: _contract())
    for name in (
        "build_pilot_evidence_package",
        "build_pilot_v24_evidence_package",
        "build_pilot_v2112_evidence_package",
        "build_pilot_v2115_evidence_package",
        "build_pilot_v2119_evidence_package",
    ):
        monkeypatch.setattr(run_pilot, name, _forbidden)

    package_dir = evidence_root / "current_v2/pilot-v2.11.10"

    def build(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            package_dir=package_dir,
            manifest_path=package_dir / "package_manifest.json",
            checksums_path=package_dir / "checksums.json",
            contract_hash="b" * 64,
            scientific_complete=True,
            claim_gates={"experiment_d": {"status": "go"}},
        )

    monkeypatch.setattr(run_pilot, "build_pilot_v21110_evidence_package", build)
    result = run_pilot.execute(
        _args(
            "--stage",
            "publish-evidence",
            "--raw-root",
            str(raw_root),
            "--source-repo-root",
            str(source_root),
            "--failed-repo-root",
            str(failed_root),
            "--authority-repo-root",
            str(authority_root),
            "--evidence-root",
            str(evidence_root),
            "--publication-ci-receipt",
            str(ci_receipts[0]),
            "--publication-ci-receipt",
            str(ci_receipts[1]),
        )
    )

    assert calls == [
        {
            "contract_path": CONTRACT_PATH,
            "run_ledger_path": raw_root / "run_ledger.json",
            "raw_root": raw_root,
            "build_root": evidence_root,
            "source_repo_root": source_root,
            "failed_repo_root": failed_root,
            "authority_repo_root": authority_root,
            "publication_ci_receipt_paths": ci_receipts,
        }
    ]
    assert result == {
        "status": "complete",
        "provider_calls": 0,
        "package_dir": str(package_dir),
        "manifest_path": str(package_dir / "package_manifest.json"),
        "checksums_path": str(package_dir / "checksums.json"),
        "contract_sha256": "b" * 64,
        "scientific_complete": True,
        "claim_gates": {"experiment_d": {"status": "go"}},
    }


def test_v21110_publish_evidence_help_names_three_roots() -> None:
    help_text = run_pilot.build_parser().format_help()
    assert "--source-repo-root" in help_text
    assert "--failed-repo-root" in help_text
    assert "--authority-repo-root" in help_text
    assert "--publication-ci-receipt" in help_text
    assert "V2.11.10 publish-evidence" in help_text
