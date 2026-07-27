from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import run_pilot
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.pilot_contract import load_pilot_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_4.yaml"


class _OpenRunLedger:
    def is_terminal(self, _run_id: str) -> bool:
        return False


def _projection_stage(contract: Any, run_id: str) -> str:
    matches = [
        stage_id
        for stage_id in contract.stage_ids
        if f"--{stage_id}--" in run_id
    ]
    assert len(matches) == 1
    return matches[0]


def _install_projection_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservations = {
        "gpt52_main": {
            "action": {
                "prompt_tokens": 1138,
                "completion_tokens": 2499,
                "total_tokens": 3637,
                "cost_usd": 0.0363540625,
            },
            "semantic": {
                "prompt_tokens": 3018,
                "completion_tokens": 1379,
                "total_tokens": 4397,
                "cost_usd": 0.0224896875,
            },
        },
        "llama33_local_controlled": {
            "action": {
                "prompt_tokens": 1134,
                "completion_tokens": 42,
                "total_tokens": 1176,
                "cost_usd": 0.0,
            },
            "semantic": {
                "prompt_tokens": 3124,
                "completion_tokens": 432,
                "total_tokens": 3556,
                "cost_usd": 0.0,
            },
        },
    }
    payloads: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for model_id, by_kind in reservations.items():
        payload = {
            "projection": {
                f"served::{call_kind}": {"reserved_p95": row}
                for call_kind, row in by_kind.items()
            }
        }
        path = tmp_path / f"{model_id}.projection.json"
        path.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
        payloads[model_id] = payload
        paths[model_id] = path

    def fake_load_verified_projection(
        _contract: Any,
        model_id: str,
        *,
        raw_root: Path,
        paid: Any = None,
    ) -> tuple[dict[str, Any], Path]:
        del raw_root, paid
        return payloads[model_id], paths[model_id]

    monkeypatch.setattr(
        orchestrator,
        "_load_verified_projection",
        fake_load_verified_projection,
    )


def test_v24_stage_families_keep_local_and_hosted_lanes_separate() -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    expected_counts = {
        "local-experiment-c": 25,
        "local-experiment-a": 20,
        "local-experiment-d": 35,
        "local-experiment-b": 25,
        "experiment-c": 25,
        "experiment-a": 20,
        "experiment-d": 30,
        "experiment-b": 15,
    }
    expected_models = {
        "local-experiment-c": "llama33_local_controlled",
        "local-experiment-a": "llama33_local_controlled",
        "local-experiment-d": "llama33_local_controlled",
        "local-experiment-b": "llama33_local_controlled",
        "experiment-c": "gpt52_main",
        "experiment-a": "gpt52_main",
        "experiment-d": "gpt52_main",
        "experiment-b": "gpt52_main",
    }

    assert orchestrator._contract_core_stage_ids(contract) == tuple(
        expected_counts
    )
    for stage_id, expected_count in expected_counts.items():
        specs = contract.expand(stage=stage_id)
        assert len(specs) == expected_count
        assert {spec.model_id for spec in specs} == {
            expected_models[stage_id]
        }
        assert orchestrator._core_stage_family(stage_id) == (
            stage_id.removeprefix("local-")
        )

    assert orchestrator._core_stage_family("stage0-calibration") is None
    assert orchestrator._core_stage_family("local-experiment-e") is None


def test_v24_runtime_projection_matches_frozen_call_and_cost_totals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    _install_projection_fixture(tmp_path, monkeypatch)

    core = orchestrator._remaining_core_projections(
        contract,
        raw_root=tmp_path,
        paid=object(),
        run_ledger=_OpenRunLedger(),
    )
    stage0 = tuple(
        orchestrator.projection_from_preflight(
            contract,
            spec,
            raw_root=tmp_path,
            paid=object(),
        )
        for spec in contract.expand(stage="stage0-calibration")
    )
    projections = (*stage0, *core)
    by_stage: dict[str, list[Any]] = {}
    for projection in projections:
        stage_id = _projection_stage(contract, projection.run_id)
        by_stage.setdefault(stage_id, []).append(projection)

    expected = {
        "stage0-calibration": (672, 0, 0.0),
        "local-experiment-c": (1280, 0, 0.0),
        "local-experiment-a": (1280, 0, 0.0),
        "local-experiment-d": (1000, 0, 0.0),
        "local-experiment-b": (1440, 0, 0.0),
        "experiment-c": (1280, 1280, 42.0966),
        "experiment-a": (1280, 1280, 42.0966),
        "experiment-d": (880, 880, 31.437),
        "experiment-b": (800, 800, 27.9741),
    }
    for stage_id, (logical, hosted, cost) in expected.items():
        rows = by_stage[stage_id]
        assert sum(
            int(row.basis.get("run_call_limit", 0)) for row in rows
        ) == logical
        assert sum(row.completions for row in rows) == hosted
        assert sum(row.cost_usd for row in rows) == pytest.approx(cost)

    local_stage_ids = {
        "stage0-calibration",
        "local-experiment-c",
        "local-experiment-a",
        "local-experiment-d",
        "local-experiment-b",
    }
    local_rows = [
        row
        for stage_id, rows in by_stage.items()
        if stage_id in local_stage_ids
        for row in rows
    ]
    hosted_rows = [
        row
        for stage_id, rows in by_stage.items()
        if stage_id.startswith("experiment-")
        for row in rows
    ]
    assert sum(
        int(row.basis.get("run_call_limit", 0)) for row in local_rows
    ) == 5672
    assert sum(row.completions for row in local_rows) == 0
    assert sum(row.completions for row in hosted_rows) == 4240
    assert sum(row.cost_usd for row in hosted_rows) == pytest.approx(
        143.6043
    )

    for stage_id in ("local-experiment-c", "experiment-c"):
        rows = by_stage[stage_id]
        assert len(rows) == 25
        assert sum(
            row.basis["method"] == "offline-zero-provider-call"
            for row in rows
        ) == 5
    assert {
        row.basis["run_call_limit"]
        for row in by_stage["local-experiment-d"]
    } == {200}
    assert {
        row.completions for row in by_stage["local-experiment-d"]
    } == {0}
    assert {
        row.basis["run_call_limit"]
        for row in by_stage["experiment-d"]
    } == {176}
    assert {
        row.completions for row in by_stage["experiment-d"]
    } == {176}


def test_v24_stage0_full_matrix_projection_has_no_legacy_cross_model_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    _install_projection_fixture(tmp_path, monkeypatch)

    assert orchestrator._cross_model_science_stage_ids(contract) == ()
    projections = orchestrator._remaining_scientific_projections(
        contract,
        raw_root=tmp_path,
        paid=object(),
        run_ledger=_OpenRunLedger(),
    )

    # One projection per non-D provider row, plus one shared projection per
    # D model/seed.  No Stage-0 or local A--D row may be duplicated through
    # the legacy cross-model-sentinel branch.
    expected_projection_count = (
        14  # Stage 0
        + 25
        + 20
        + 5  # local C, A, shared-D groups
        + 25
        + 25
        + 20
        + 5  # hosted C, A, shared-D groups
        + 15
    )
    assert len(projections) == expected_projection_count == 154
    assert sum(row.completions for row in projections) == 4240
    assert sum(row.cost_usd for row in projections) == pytest.approx(
        143.6043
    )


def test_v24_experiment_c_sensitivity_artifacts_are_lane_isolated(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    local_path = orchestrator._experiment_c_sensitivity_path(
        tmp_path,
        "local-experiment-c",
    )
    hosted_path = orchestrator._experiment_c_sensitivity_path(
        tmp_path,
        "experiment-c",
    )
    local_path.parent.mkdir(parents=True)
    hosted_path.parent.mkdir(parents=True)
    local_path.write_text('{"lane":"local"}\n', encoding="utf-8")
    hosted_path.write_text('{"lane":"hosted"}\n', encoding="utf-8")

    local_controls = orchestrator._v2_stage_control_paths(
        contract,
        "local-experiment-c",
        raw_root=tmp_path,
    )
    hosted_controls = orchestrator._v2_stage_control_paths(
        contract,
        "experiment-c",
        raw_root=tmp_path,
    )
    assert local_path in local_controls
    assert hosted_path not in local_controls
    assert hosted_path in hosted_controls
    assert local_path not in hosted_controls
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="not an Experiment C stage",
    ):
        orchestrator._experiment_c_sensitivity_path(
            tmp_path,
            "experiment-a",
        )


def test_parent_import_control_inventory_never_self_binds_stage_receipt(
    tmp_path: Path,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    stage_root = tmp_path / "parent-import"
    nested = stage_root / "observed_p95" / "gpt52_main" / "authority.json"
    nested.parent.mkdir(parents=True)
    nested.write_text('{"authority":true}\n', encoding="utf-8")
    stage_receipt = stage_root / "stage_receipt.json"
    stage_receipt.write_text('{"receipt":"already-written"}\n', encoding="utf-8")

    controls = orchestrator._v2_stage_control_paths(
        contract,
        "parent-import",
        raw_root=tmp_path,
    )

    assert nested in controls
    assert stage_receipt not in controls


def test_parent_import_routes_before_any_provider_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    paid = SimpleNamespace(
        git_tag="pilot-v2.4-science",
        head_commit="a" * 40,
    )
    ledger = SimpleNamespace(register=lambda _specs: None)
    parent_root = tmp_path / "parent"
    parent_root.mkdir()
    provider_calls: list[str] = []
    route_calls: list[dict[str, Any]] = []

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> None:
        provider_calls.append("forbidden")
        raise AssertionError("parent-import must not construct a provider")

    def fake_parent_import(
        _contract: Any,
        _specs: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        route_calls.append(kwargs)
        return {"status": "complete", "provider_calls": 0}

    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: paid,
    )
    monkeypatch.setattr(
        orchestrator,
        "_persist_release_attestation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator,
        "PilotRunLedger",
        lambda *_args, **_kwargs: ledger,
    )
    monkeypatch.setattr(
        orchestrator,
        "_provider_for_profile",
        forbidden_provider,
    )
    monkeypatch.setattr(
        orchestrator,
        "create_llm_provider",
        forbidden_provider,
    )
    monkeypatch.setattr(
        orchestrator,
        "build_preflight_amendment_control",
        forbidden_provider,
    )
    monkeypatch.setattr(
        orchestrator,
        "persist_evaluator_correction_receipt",
        forbidden_provider,
    )
    monkeypatch.setattr(
        orchestrator,
        "_parent_budget_debit",
        forbidden_provider,
    )
    monkeypatch.setattr(
        orchestrator,
        "PilotBudgetLedger",
        forbidden_provider,
    )
    monkeypatch.setattr(
        orchestrator,
        "_execute_v24_parent_import_stage",
        fake_parent_import,
    )
    monkeypatch.setattr(
        orchestrator,
        "load_pilot_contract",
        lambda _path: contract,
    )

    result = orchestrator._execute_stage_locked(
        contract_path=CONTRACT_PATH,
        stage_id="parent-import",
        resume=True,
        raw_root=tmp_path / "raw",
        repo_root=tmp_path,
        parent_repo_root=parent_root,
    )

    assert result == {"status": "complete", "provider_calls": 0}
    assert provider_calls == []
    assert len(route_calls) == 1
    assert route_calls[0]["parent_repo_root"] == parent_root


def test_v24_later_stages_skip_inherited_legacy_raw_amendments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    assert contract.evaluator_amendment is not None
    assert contract.preflight_bootstrap_amendment is not None
    assert (
        orchestrator._materializes_legacy_amendment_controls(contract)
        is False
    )

    paid = SimpleNamespace(
        git_tag="pilot-v2.4-science",
        head_commit="a" * 40,
    )
    ledger = SimpleNamespace(register=lambda _specs: None)

    def forbidden_legacy(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError(
            "V2.4 must not rematerialize a V2.1-V2.3 raw amendment"
        )

    class ReachedBudgetLedger(RuntimeError):
        pass

    def stop_at_budget(*_args: Any, **_kwargs: Any) -> None:
        raise ReachedBudgetLedger

    monkeypatch.setattr(
        orchestrator,
        "load_pilot_contract",
        lambda _path: contract,
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: paid,
    )
    monkeypatch.setattr(
        orchestrator,
        "_persist_release_attestation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator,
        "PilotRunLedger",
        lambda *_args, **_kwargs: ledger,
    )
    monkeypatch.setattr(
        orchestrator,
        "build_preflight_amendment_control",
        forbidden_legacy,
    )
    monkeypatch.setattr(
        orchestrator,
        "persist_evaluator_correction_receipt",
        forbidden_legacy,
    )
    monkeypatch.setattr(
        orchestrator,
        "_parent_budget_debit",
        lambda _contract: object(),
    )
    monkeypatch.setattr(
        orchestrator,
        "PilotBudgetLedger",
        stop_at_budget,
    )

    with pytest.raises(ReachedBudgetLedger):
        orchestrator._execute_stage_locked(
            contract_path=CONTRACT_PATH,
            stage_id="q-ref-resolution",
            resume=True,
            raw_root=tmp_path / "raw",
            repo_root=tmp_path,
        )

    evaluator = orchestrator.evaluator_amendment_control_path(
        raw_root=tmp_path,
    )
    preflight = orchestrator.preflight_amendment_control_path(
        raw_root=tmp_path,
    )
    evaluator.write_text('{"legacy":true}\n', encoding="utf-8")
    preflight.write_text('{"legacy":true}\n', encoding="utf-8")
    controls = orchestrator._v2_stage_control_paths(
        contract,
        "q-ref-resolution",
        raw_root=tmp_path,
    )
    assert evaluator not in controls
    assert preflight not in controls


@pytest.mark.parametrize(
    ("stage_id", "development_fake"),
    [
        ("experiment-a", False),
        ("publish-evidence", False),
        ("development-a-d", True),
    ],
)
def test_cli_parent_repo_root_is_rejected_outside_parent_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_id: str,
    development_fake: bool,
) -> None:
    parent_root = tmp_path / "parent"
    args = run_pilot.build_parser().parse_args(
        [
            "--contract",
            str(CONTRACT_PATH),
            "--stage",
            stage_id,
            "--raw-root",
            str(tmp_path / "raw"),
            "--parent-repo-root",
            str(parent_root),
            *(["--development-fake"] if development_fake else []),
        ]
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("invalid CLI scope must stop before execution")

    monkeypatch.setattr(run_pilot, "execute_stage", forbidden)
    monkeypatch.setattr(run_pilot, "run_development_fake_matrix", forbidden)
    monkeypatch.setattr(run_pilot, "build_pilot_evidence_package", forbidden)
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="accepted only",
    ):
        run_pilot.execute(args)


def test_cli_parent_repo_scope_rejects_before_contract_or_raw_root_loading(
    tmp_path: Path,
) -> None:
    args = run_pilot.build_parser().parse_args(
        [
            "--contract",
            str(tmp_path / "missing-contract.yaml"),
            "--stage",
            "experiment-a",
            "--parent-repo-root",
            str(tmp_path / "parent"),
        ]
    )

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="accepted only",
    ):
        run_pilot.execute(args)


def test_cli_parent_repo_root_is_forwarded_only_for_parent_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_root = tmp_path / "parent"
    args = run_pilot.build_parser().parse_args(
        [
            "--contract",
            str(CONTRACT_PATH),
            "--stage",
            "parent-import",
            "--raw-root",
            str(tmp_path / "raw"),
            "--parent-repo-root",
            str(parent_root),
            "--resume",
        ]
    )
    calls: list[dict[str, Any]] = []

    def fake_execute_stage(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "complete", "provider_calls": 0}

    monkeypatch.setattr(run_pilot, "execute_stage", fake_execute_stage)
    result = run_pilot.execute(args)

    assert result == {"status": "complete", "provider_calls": 0}
    assert len(calls) == 1
    assert calls[0]["parent_repo_root"] == parent_root


def test_v24_scripted_development_matrix_covers_both_registered_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_provider(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("V2.4 development matrix attempted a live provider")

    monkeypatch.setattr(
        orchestrator,
        "_provider_for_profile",
        forbidden_provider,
    )
    result = orchestrator.run_development_fake_matrix(
        contract_path=CONTRACT_PATH,
        resume=False,
        raw_root=tmp_path,
    )

    assert result["status"] == "pass"
    assert result["registered_cells"] == 39
    assert result["status_counts"] == {"complete": 39}
    assert result["stages"] == [
        "local-experiment-c",
        "local-experiment-a",
        "local-experiment-d",
        "local-experiment-b",
        "experiment-c",
        "experiment-a",
        "experiment-d",
        "experiment-b",
    ]
    assert result["diagnostic_only"] is True
    assert result["scientific_evidence"] is False

    root = tmp_path / "development-fake"
    assert len(
        list(
            (root / "local-experiment-d" / "diagnostic_summaries").glob(
                "*.json"
            )
        )
    ) == 7
    assert len(
        list(
            (root / "experiment-d" / "diagnostic_summaries").glob("*.json")
        )
    ) == 6
