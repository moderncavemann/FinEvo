from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.pilot_budget import PilotBudgetLedger, RunProjection
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_V2_10_CANONICAL_SHA256,
    load_pilot_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_10.yaml"


def _contract():
    return load_pilot_contract(CONTRACT_PATH)


def _paid() -> orchestrator.GitProvenance:
    return orchestrator.GitProvenance(
        git_tag="pilot-v2.10-science",
        head_commit="a" * 40,
        tag_commit="a" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )


def _projection_and_flat_binding(contract, model_id: str):
    profile = contract.provider_profiles[model_id]
    runtime_model = orchestrator._runtime_model_for_profile(profile)
    reservation = {
        "sample_count": 1,
        "raw_p95": {
            "prompt_tokens": 10.0,
            "completion_tokens": 5.0,
            "total_tokens": 15.0,
            "cost_usd": 0.0,
        },
        "reserved_p95": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20,
            "cost_usd": 0.0,
        },
        "reserve_multiplier": 1.25,
    }
    projection = {
        f"{profile.served_model}::{kind}": deepcopy(reservation)
        for kind in ("action", "semantic")
    }
    reservations = {
        runtime_model: {
            kind: {
                "authority": {
                    "pilot_contract_hash": contract.canonical_hash,
                    "pilot_tag": "pilot-v2.10-science",
                },
                "reservation": deepcopy(reservation),
            }
            for kind in ("action", "semantic")
        }
    }
    return (
        {
            "served_model": profile.served_model,
            "projection": projection,
        },
        {
            "receipt_path": (
                "experiment_results/pilot-v2.10/raw/parent-import/"
                f"observed_p95/{model_id}/observed_p95_authority_receipt.json"
            ),
            "receipt_file_sha256": "b" * 64,
            "receipt_content_sha256": "c" * 64,
            "git_commit": "a" * 40,
            "reservations": reservations,
        },
    )


def test_v210_denominator_is_fresh_211_with_exact_sixteen_prerequisites(
    tmp_path: Path,
) -> None:
    contract = _contract()
    specs = tuple(contract.expand())
    counts = Counter(spec.stage_id for spec in specs)

    assert contract.contract_id == orchestrator.V210_CONTRACT_ID
    assert len(specs) == 211
    assert counts["parent-import"] == 1
    assert counts["q-ref-resolution"] == 1
    assert counts["stage0-calibration"] == 14
    assert sum(
        count
        for stage_id, count in counts.items()
        if stage_id.startswith("experiment-")
        or stage_id.startswith("local-experiment-")
    ) == 195
    amendment = contract.p95_runner_binding_retry_amendment
    assert amendment["prerequisite_import"]["imported_complete_cells"] == 16
    assert amendment["fresh_science_dispatch"]["a_d_cells"] == 195
    assert (
        amendment["fresh_science_dispatch"]["a_d_provider_dispatch"]
        == "fresh-only"
    )

    ledger = orchestrator.PilotRunLedger(
        tmp_path / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(specs)
    rows = ledger.snapshot()["runs"]
    assert len(rows) == 211
    assert Counter(row["status"] for row in rows.values()) == {
        "scheduled": 211
    }
    a_d_specs = tuple(
        spec
        for spec in specs
        if spec.stage_id.startswith("experiment-")
        or spec.stage_id.startswith("local-experiment-")
    )
    assert len(a_d_specs) == 195
    assert all(
        rows[spec.run_id]["status"] == "scheduled"
        and rows[spec.run_id]["artifact"] is None
        and rows[spec.run_id]["failure"] is None
        for spec in a_d_specs
    )


@pytest.mark.parametrize(
    "model_id",
    ("gpt52_main", "llama33_local_controlled"),
)
def test_v210_runner_consumes_exact_flat_p95_binding_for_both_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
) -> None:
    contract = _contract()
    projection, binding = _projection_and_flat_binding(contract, model_id)
    projection_path = tmp_path / f"{model_id}-projection.json"
    monkeypatch.setattr(
        orchestrator,
        "_load_verified_projection",
        lambda *_args, **_kwargs: (deepcopy(projection), projection_path),
    )
    monkeypatch.setattr(
        orchestrator,
        "_verified_observed_p95_binding",
        lambda *_args, **_kwargs: deepcopy(binding),
    )

    reservations = orchestrator._runner_p95_reservations(
        contract,
        model_id,
        raw_root=tmp_path,
        paid=_paid(),
    )

    runtime_model = orchestrator._runtime_model_for_profile(
        contract.provider_profiles[model_id]
    )
    assert set(reservations) == {runtime_model}
    for call_kind in ("action", "semantic"):
        authority = reservations[runtime_model][call_kind]["authority"]
        assert authority["source_authority_receipt_path"] == binding["receipt_path"]
        assert (
            authority["source_authority_receipt_file_sha256"]
            == binding["receipt_file_sha256"]
        )
        assert (
            authority["source_authority_receipt_content_sha256"]
            == binding["receipt_content_sha256"]
        )
        assert authority["source_release_commit"] == binding["git_commit"]


@pytest.mark.parametrize(
    "model_id,stage_id",
    (
        ("gpt52_main", "experiment-a"),
        ("llama33_local_controlled", "local-experiment-a"),
    ),
)
def test_v210_malformed_flat_p95_stops_actor_before_provider_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    stage_id: str,
) -> None:
    contract = _contract()
    projection, binding = _projection_and_flat_binding(contract, model_id)
    binding.pop("receipt_path")
    monkeypatch.setattr(
        orchestrator,
        "_load_verified_projection",
        lambda *_args, **_kwargs: (deepcopy(projection), tmp_path / "p95.json"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_verified_observed_p95_binding",
        lambda *_args, **_kwargs: deepcopy(binding),
    )
    monkeypatch.setattr(
        orchestrator,
        "config_for_spec",
        lambda *_args, **_kwargs: pytest.fail(
            "config construction ran after malformed p95 binding"
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_provider_for_profile",
        lambda *_args, **_kwargs: pytest.fail(
            "provider constructed after malformed p95 binding"
        ),
    )
    spec = next(
        spec
        for spec in contract.expand(stage=stage_id, model=model_id)
        if spec.execution_mode == "actor_run"
    )
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="exact current-release flat contract",
    ):
        orchestrator._execute_actor_run(
            contract,
            spec,
            raw_root=tmp_path,
            paid=_paid(),
            projection=RunProjection(
                run_id=spec.run_id,
                stage_bucket=spec.budget_bucket,
                cost_usd=0.0,
                completions=0,
                storage_bytes=1,
                basis={"hosted_completion_cap_counted": False},
            ),
            budget=object(),
        )


@pytest.mark.parametrize(
    "model_id,stage_id",
    (
        ("gpt52_main", "experiment-d"),
        ("llama33_local_controlled", "local-experiment-d"),
    ),
)
def test_v210_d_path_stops_on_malformed_p95_before_provider_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    stage_id: str,
) -> None:
    contract = _contract()
    projection, binding = _projection_and_flat_binding(contract, model_id)
    binding.pop("receipt_content_sha256")
    monkeypatch.setattr(
        orchestrator,
        "_load_verified_projection",
        lambda *_args, **_kwargs: (deepcopy(projection), tmp_path / "p95.json"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_verified_observed_p95_binding",
        lambda *_args, **_kwargs: deepcopy(binding),
    )
    monkeypatch.setattr(
        orchestrator,
        "_provider_for_profile",
        lambda *_args, **_kwargs: pytest.fail(
            "D provider constructed after malformed p95 binding"
        ),
    )
    group = tuple(
        spec
        for spec in contract.expand(stage=stage_id, model=model_id)
        if spec.environment_seed
        == contract.expand(stage=stage_id, model=model_id)[0].environment_seed
    )
    group_projection = RunProjection(
        run_id=f"{contract.contract_id}--{stage_id}--{model_id}--test-group",
        stage_bucket=group[0].budget_bucket,
        cost_usd=0.0,
        completions=0,
        storage_bytes=2_000_000,
        basis={
            "hosted_completion_cap_counted": False,
            "run_call_limit": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "_d_group_projection",
        lambda *_args, **_kwargs: group_projection,
    )
    run_ledger = orchestrator.PilotRunLedger(
        tmp_path / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    run_ledger.register(contract.expand())
    budget_ledger = PilotBudgetLedger(
        tmp_path / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=orchestrator._parent_budget_debit(contract),
    )

    orchestrator._execute_d_seed(
        contract,
        group,
        raw_root=tmp_path,
        paid=_paid(),
        diagnostic=False,
        budget_ledger=budget_ledger,
        run_ledger=run_ledger,
        verify_bound_inputs=True,
    )

    assert {
        run_ledger.status(spec.run_id) for spec in group
    } == {"failed"}
    assert budget_ledger.snapshot()["runs"][group_projection.run_id][
        "status"
    ] == "failed"


def test_v210_prerequisite_stages_route_to_zero_provider_importers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    routed: list[str] = []

    class Ledger:
        def register(self, _specs: Any) -> None:
            return None

        def status(self, _run_id: str) -> str:
            return "scheduled"

    class Budget:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("V2.10 prerequisite route constructed a provider")

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _p: contract)
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: _paid(),
    )
    monkeypatch.setattr(
        orchestrator,
        "_persist_release_attestation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(orchestrator, "PilotRunLedger", lambda *_a, **_k: Ledger())
    monkeypatch.setattr(orchestrator, "PilotBudgetLedger", Budget)
    monkeypatch.setattr(orchestrator, "_provider_for_profile", forbidden)
    monkeypatch.setattr(orchestrator, "create_llm_provider", forbidden)

    def parent(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["budget_ledger"] is not None
        routed.append("parent-import")
        return {"status": "complete"}

    def qref(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["budget_ledger"] is not None
        routed.append("q-ref-resolution")
        return {"status": "complete"}

    def stage0(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["budget_ledger"] is not None
        routed.append("stage0-calibration")
        return {"status": "complete"}

    monkeypatch.setattr(orchestrator, "_execute_v24_parent_import_stage", parent)
    monkeypatch.setattr(orchestrator, "_execute_v210_q_ref_import_stage", qref)
    monkeypatch.setattr(orchestrator, "_execute_imported_stage0_stage", stage0)
    raw_root = tmp_path / "experiment_results" / "pilot-v2.10" / "raw"
    for stage_id in (
        "parent-import",
        "q-ref-resolution",
        "stage0-calibration",
    ):
        result = orchestrator._execute_stage_locked(
            contract_path=CONTRACT_PATH,
            stage_id=stage_id,
            resume=True,
            raw_root=raw_root,
            repo_root=tmp_path,
            parent_repo_root=tmp_path,
        )
        assert result["status"] == "complete"
    assert routed == [
        "parent-import",
        "q-ref-resolution",
        "stage0-calibration",
    ]


def test_v210_stage0_import_rejects_partial_matrix(tmp_path: Path) -> None:
    contract = _contract()
    specs = tuple(contract.expand(stage="stage0-calibration"))
    assert len(specs) == 14
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="exact 7x2 actor matrix",
    ):
        orchestrator._execute_imported_stage0_stage(
            contract,
            specs[:-1],
            raw_root=tmp_path,
            paid=_paid(),
            run_ledger=object(),
            budget_ledger=object(),
        )


def test_v210_frozen_development_fake_matrix_is_exactly_39_diagnostic_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    assert contract.status == "frozen"
    assert PILOT_CONTRACT_V2_10_CANONICAL_SHA256 is not None
    assert contract.canonical_hash == PILOT_CONTRACT_V2_10_CANONICAL_SHA256

    def forbidden_provider(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("V2.10 development matrix attempted a live provider")

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
    result = orchestrator.run_development_fake_matrix(
        contract_path=CONTRACT_PATH,
        resume=False,
        raw_root=tmp_path,
    )

    expected_stages = [
        "local-experiment-c",
        "local-experiment-a",
        "local-experiment-d",
        "local-experiment-b",
        "experiment-c",
        "experiment-a",
        "experiment-d",
        "experiment-b",
    ]
    assert result["schema_version"] == (
        orchestrator.PILOT_DEVELOPMENT_MATRIX_SCHEMA_VERSION
    )
    assert result["contract_id"] == contract.contract_id
    assert result["contract_sha256"] == contract.canonical_hash
    assert result["status"] == "pass"
    assert result["registered_cells"] == 39
    assert result["status_counts"] == {"complete": 39}
    assert result["stages"] == expected_stages
    assert result["one_environment_seed"] == 1099057501
    assert result["actor_fixture_shape"] == {
        "num_agents": 2,
        "episode_length": 6,
    }
    assert result["experiment_d_shape"] == {
        "num_agents": 4,
        "episode_length": 12,
    }
    assert result["diagnostic_only"] is True
    assert result["scientific_evidence"] is False
    assert orchestrator._read_json(Path(result["receipt"])) == {
        key: value for key, value in result.items() if key != "receipt"
    }

    run_ledger = orchestrator._read_json(Path(result["run_ledger"]))
    run_rows = tuple(run_ledger["runs"].values())
    assert len(run_rows) == 39
    assert Counter(row["status"] for row in run_rows) == {"complete": 39}
    assert Counter(row["spec"]["stage_id"] for row in run_rows) == {
        "local-experiment-c": 5,
        "local-experiment-a": 4,
        "local-experiment-d": 7,
        "local-experiment-b": 5,
        "experiment-c": 5,
        "experiment-a": 4,
        "experiment-d": 6,
        "experiment-b": 3,
    }
    assert Counter(row["spec"]["execution_mode"] for row in run_rows) == {
        "actor_run": 24,
        "checkpoint_continuation": 13,
        "offline_candidate_admission": 2,
    }
    assert all(row["failure"] is None for row in run_rows)
    assert all(row["artifact"] is not None for row in run_rows)

    # Every registered cell must remain explicitly diagnostic at its terminal
    # artifact, rather than relying only on the matrix-level claim boundary.
    for row in run_rows:
        artifact = Path(row["artifact"])
        if row["spec"]["execution_mode"] == "actor_run":
            terminal = orchestrator._read_json(artifact.parent / "provenance.json")[
                "details"
            ]
        else:
            terminal = orchestrator._read_json(artifact)
        assert terminal["diagnostic_only"] is True
        assert terminal["scientific_evidence"] is False

    budget_ledger = orchestrator._read_json(Path(result["budget_ledger"]))
    budget_rows = tuple(budget_ledger["runs"].values())
    # The 26 non-D cells reserve individually; all 13 D branches share one
    # reservation per local/hosted seed group.
    assert len(budget_rows) == 28
    assert Counter(row["status"] for row in budget_rows) == {"complete": 28}
    assert all(row["failure"] is None for row in budget_rows)
    assert all(row["actual"]["cost_usd"] == 0.0 for row in budget_rows)
    assert sum(row["actual"]["cost_usd"] for row in budget_rows) == 0.0

    for artifact in result["bootstrap_artifacts"].values():
        bootstrap = orchestrator._read_json(Path(artifact))
        assert bootstrap["diagnostic_only"] is True
        assert bootstrap["scientific_evidence"] is False
