from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.render_pilot_v21110_contract import (
    _parse_with_bootstrap_design_pin,
    build_contract,
)
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.pilot_budget import RunProjection


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_10.yaml"
COMMIT = "a" * 40
TAG_OBJECT = "b" * 40


@pytest.fixture(autouse=True)
def _provider_credentials_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def contract() -> Any:
    return _parse_with_bootstrap_design_pin(build_contract(ROOT, status="draft"))


def _paid(contract: Any) -> orchestrator.GitProvenance:
    tag = str(contract.implementation["required_git_tag"])
    return orchestrator.GitProvenance(
        git_tag=tag,
        head_commit=COMMIT,
        tag_commit=COMMIT,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={"fixture": True},
        release_attestation={
            "local_tag": {
                "name": tag,
                "object_id": TAG_OBJECT,
                "peeled_commit": COMMIT,
                "kind": "annotated",
            }
        },
    )


def _forbidden(name: str):
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(f"{name} must remain provider-free and unreachable")

    return fail


def test_v21110_parent_debit_routes_to_exact_cumulative_v2119_record(
    contract: Any,
) -> None:
    debit = orchestrator._parent_budget_debit(contract)

    assert debit.to_dict() == contract.v21110_recovery_boundary[
        "parent_budget_debit"
    ]
    assert debit.to_dict() == {
        "schema_version": "finevo-parent-budget-debit-v1",
        "parent_contract_sha256": (
            "ec16563bf906b8f6c1492d2a30f291d2c849cd639c2f314e7a1c8ac619e3fa3f"
        ),
        "parent_run_ledger_sha256": (
            "b2891fb152825cac846955b9c2fe4a041e80eab8cbebef9bc4d861d2313fc923"
        ),
        "parent_budget_ledger_sha256": (
            "02adeb470b823664c67d09cd34df8787a68760e6270f46b59cca204701e3465d"
        ),
        "stage_bucket": "parent_v2119",
        "cost_usd": 63.1196450625,
        "hosted_completions": 3440,
        "storage_bytes": 270_993_662,
        "record_sha256": (
            "5e0c39817c32c845c2f771a02320c55e85e9a6bfb5f3e705046b822593b4c592"
        ),
    }


def test_v21110_current_authority_path_and_binding_use_current_release(
    contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "experiment_results/pilot-v2.11.10/raw"
    paid = _paid(contract)
    expected_path = (
        raw_root / "parent-import/current_authority/post_gate_authority.json"
    )
    assert (
        orchestrator._observed_p95_authority_receipt_path(
            contract,
            "gpt52_main",
            raw_root=raw_root,
        )
        == expected_path
    )
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="has no V2.11.10 continuation dispatch authority",
    ):
        orchestrator._observed_p95_authority_receipt_path(
            contract,
            "unsupported-model",
            raw_root=raw_root,
        )

    runtime = orchestrator._runtime_model_for_profile(
        contract.provider_profiles["gpt52_main"]
    )
    expected = {
        "receipt_path": expected_path.relative_to(tmp_path).as_posix(),
        "receipt_file_sha256": "1" * 64,
        "receipt_content_sha256": "2" * 64,
        "git_commit": COMMIT,
        "reservations": {runtime: {"action": {}, "semantic": {}}},
    }
    calls: list[dict[str, Any]] = []

    def verify(path: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({"path": path, **kwargs})
        return expected

    monkeypatch.setattr(
        orchestrator,
        "verified_v21110_observed_p95_authority_binding",
        verify,
    )
    observed = orchestrator._verified_observed_p95_binding(
        contract,
        "gpt52_main",
        raw_root=raw_root,
        paid=paid,
        authority_repo_root=tmp_path,
    )

    assert observed is expected
    assert calls == [
        {
            "path": expected_path.relative_to(tmp_path).as_posix(),
            "repo_root": tmp_path.resolve(),
            "expected_git_commit": COMMIT,
            "expected_contract_sha256": contract.canonical_hash,
        }
    ]


def test_v21110_runner_preserves_exact_13_to_17_authority_layering(
    contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid(contract)
    raw_root = tmp_path / "experiment_results/pilot-v2.11.10/raw"
    runtime = orchestrator._runtime_model_for_profile(
        contract.provider_profiles["gpt52_main"]
    )
    envelope = {
        "source_authority_receipt_path": (
            "experiment_results/pilot-v2.11.10/raw/parent-import/"
            "current_authority/post_gate_authority.json"
        ),
        "source_authority_receipt_file_sha256": "3" * 64,
        "source_authority_receipt_content_sha256": "4" * 64,
        "source_release_commit": COMMIT,
    }
    source_by_kind: dict[str, dict[str, Any]] = {}
    runner_by_kind: dict[str, dict[str, Any]] = {}
    for kind in ("action", "semantic"):
        authority = {f"source_field_{index:02d}": f"{kind}-{index}" for index in range(13)}
        reservation = {"reserved_p95": {"cost_usd": 0.01}, "kind": kind}
        source_by_kind[kind] = {
            "reservation": reservation,
            "authority": authority,
        }
        runner_by_kind[kind] = {
            "reservation": reservation,
            "authority": {**authority, **envelope},
        }
    binding = {
        "receipt_path": envelope["source_authority_receipt_path"],
        "receipt_file_sha256": envelope["source_authority_receipt_file_sha256"],
        "receipt_content_sha256": envelope[
            "source_authority_receipt_content_sha256"
        ],
        "git_commit": envelope["source_release_commit"],
        "reservations": {runtime: source_by_kind},
    }
    runner = {runtime: runner_by_kind}
    monkeypatch.setattr(
        orchestrator,
        "_load_verified_projection",
        lambda *_args, **_kwargs: ({"fixture": "projection"}, tmp_path / "p95.json"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_verified_observed_p95_binding",
        lambda *_args, **_kwargs: binding,
    )
    monkeypatch.setattr(
        orchestrator,
        "runner_reservations_for_v21110",
        lambda *_args, **_kwargs: runner,
    )

    observed = orchestrator._runner_p95_reservations(
        contract,
        "gpt52_main",
        raw_root=raw_root,
        paid=paid,
        authority_repo_root=tmp_path,
    )

    assert observed == runner
    for kind in ("action", "semantic"):
        producer = binding["reservations"][runtime][kind]["authority"]
        serialized = observed[runtime][kind]["authority"]
        assert len(producer) == 13
        assert len(serialized) == 17
        assert {key: serialized[key] for key in envelope} == envelope
        assert {key: value for key, value in serialized.items() if key not in envelope} == producer

    legacy_binding = {
        **binding,
        "reservations": {
            runtime: {
                kind: {
                    **source_by_kind[kind],
                    "authority": {**source_by_kind[kind]["authority"], **envelope},
                }
                for kind in ("action", "semantic")
            }
        },
    }
    monkeypatch.setattr(
        orchestrator,
        "_verified_observed_p95_binding",
        lambda *_args, **_kwargs: legacy_binding,
    )
    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match="runner authority layering differs from the flat source binding",
    ):
        orchestrator._runner_p95_reservations(
            contract,
            "gpt52_main",
            raw_root=raw_root,
            paid=paid,
            authority_repo_root=tmp_path,
        )


def test_v21110_projection_loader_delegates_exact_release_inputs(
    contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    paid = _paid(contract)
    expected = ({"fixture": "projection"}, tmp_path / "projection.json")
    calls: list[dict[str, Any]] = []

    def verify(contract_arg: Any, model_id: str, **kwargs: Any) -> Any:
        calls.append({"contract": contract_arg, "model_id": model_id, **kwargs})
        return expected

    monkeypatch.setattr(orchestrator, "verified_v21110_projection", verify)
    observed = orchestrator._load_verified_projection(
        contract,
        "gpt56_diagnostic",
        raw_root=raw_root,
        paid=paid,
        authority_repo_root=tmp_path,
    )

    assert observed == expected
    assert calls == [
        {
            "contract": contract,
            "model_id": "gpt56_diagnostic",
            "repo_root": tmp_path.resolve(),
            "raw_root": raw_root,
            "paid": paid,
        }
    ]


def test_v21110_parent_import_projection_is_zero_provider_and_bound_to_v2119(
    contract: Any,
) -> None:
    spec = contract.expand(stage="parent-import")[0]
    projection = orchestrator._v21110_parent_import_projection(spec)

    assert projection.run_id == spec.run_id
    assert projection.stage_bucket == "parent_v2119"
    assert projection.cost_usd == 0.0
    assert projection.completions == 0
    assert projection.storage_bytes == 5_000_000
    assert projection.basis["provider_construction"] is False
    assert projection.basis["provider_calls"] == 0
    assert projection.basis["failed_v2119_terminal_rows_bound"] == 87
    assert projection.basis["mapped_v2115_scheduled_cells"] == 86
    assert projection.basis["imported_effect_cells"] == 0


def test_v21110_d_group_plan_keeps_exact_eleven_cell_mechanism_group(
    contract: Any,
) -> None:
    first = contract.expand(stage="experiment-d")[0]
    group = tuple(
        spec
        for spec in contract.expand(stage="experiment-d")
        if spec.model_id == first.model_id
        and spec.environment_seed == first.environment_seed
    )
    plan = orchestrator.build_v21110_experiment_d_group_plan(contract, group)
    receipt = plan.to_receipt()

    assert receipt["stage_id"] == "experiment-d"
    assert receipt["model_id"] == "gpt52_main"
    assert receipt["environment_seed"] == first.environment_seed
    assert receipt["cell_count"] == 11
    assert len(plan.continuation_specs) == 7
    assert len(plan.narrative_specs) == 4
    assert plan.prefix_config is None
    assert all(
        spec.run_id.startswith("finevo-pilot-v2.11.10--")
        for spec in (*plan.continuation_specs.values(), *plan.narrative_specs.values())
    )


def test_v21110_d_projection_and_executor_both_use_exact_group_plan() -> None:
    projection_source = inspect.getsource(orchestrator._d_group_projection)
    executor_source = inspect.getsource(orchestrator._execute_d_seed)

    assert "build_v21110_experiment_d_group_plan" in projection_source
    assert "build_v21110_experiment_d_group_plan" in executor_source


def test_v21110_parent_control_gate_verifies_receipt_authority_then_both_projections(
    contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    receipt = raw_root / "parent-import/parent_import_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    paid = _paid(contract)
    order: list[str] = []

    monkeypatch.setattr(
        orchestrator,
        "verify_v21110_parent_import_receipt",
        lambda *_args, **_kwargs: order.append("receipt"),
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v21110_current_authority",
        lambda *_args, **_kwargs: order.append("authority"),
    )

    def projection(_contract: Any, model_id: str, **_kwargs: Any) -> None:
        order.append(f"projection:{model_id}")

    monkeypatch.setattr(orchestrator, "verified_v21110_projection", projection)

    assert orchestrator._v2_control_gate_ok(
        contract,
        "parent-import",
        raw_root=raw_root,
        paid=paid,
        release_repo_root=tmp_path,
    )
    assert order == [
        "receipt",
        "authority",
        "projection:gpt52_main",
        "projection:gpt56_diagnostic",
    ]


def test_v21110_local_release_guard_rechecks_annotated_tag_and_profile_binding(
    contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tag = str(contract.implementation["required_git_tag"])
    observations = {
        ("rev-parse", "HEAD"): COMMIT,
        ("cat-file", "-t", f"refs/tags/{tag}"): "tag",
        ("rev-parse", f"refs/tags/{tag}"): TAG_OBJECT,
        ("rev-parse", f"refs/tags/{tag}^{{commit}}"): COMMIT,
        ("status", "--porcelain", "--untracked-files=all"): "",
    }
    commands: list[tuple[str, ...]] = []
    profiles: list[Path] = []

    def git(repo_root: Path, *args: str, **_kwargs: Any) -> str:
        assert repo_root == tmp_path.resolve()
        commands.append(args)
        return observations[args]

    monkeypatch.setattr(orchestrator, "_git", git)
    monkeypatch.setattr(
        orchestrator,
        "_verify_v21110_bound_working_directory",
        lambda root: profiles.append(Path(root)),
    )
    orchestrator._assert_v21110_local_release_guard(
        contract,
        repo_root=tmp_path,
        paid=_paid(contract),
    )

    assert commands == [
        ("rev-parse", "HEAD"),
        ("cat-file", "-t", f"refs/tags/{tag}"),
        ("rev-parse", f"refs/tags/{tag}"),
        ("rev-parse", f"refs/tags/{tag}^{{commit}}"),
        ("status", "--porcelain", "--untracked-files=all"),
    ]
    assert profiles == [tmp_path.resolve()]


def test_v21110_parent_stage_validates_sources_before_budget_or_dispatch(
    contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    parent_debit = object()
    expected = {"status": "complete", "provider_calls": 0}

    class FakeRunLedger:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            order.append(
                f"run-ledger:bind-terminal={kwargs['bind_terminal_artifacts']}"
            )
            self.rows: dict[str, str] = {}

        def register(self, specs: Any) -> None:
            self.rows.update({spec.run_id: "scheduled" for spec in specs})

        def status(self, run_id: str) -> str:
            return self.rows[run_id]

    class FakeBudgetLedger:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            order.append("budget-ledger")
            assert kwargs["parent_debit"] is parent_debit

    def debit(_contract: Any) -> object:
        order.append("parent-debit")
        return parent_debit

    def execute_parent(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        order.append("parent-executor")
        assert kwargs["failed_repo_root"] == tmp_path / "failed-v2119"
        assert kwargs["authority_repo_root"] == tmp_path / "authority-v2115"
        return expected

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        orchestrator,
        "require_v21110_provider_keys_absent",
        lambda: order.append("keys-absent"),
    )
    monkeypatch.setattr(
        orchestrator,
        "validate_v21110_source_manifest",
        lambda **_kwargs: order.append("source-manifest"),
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: order.append("paid-provenance") or _paid(contract),
    )
    monkeypatch.setattr(
        orchestrator,
        "_persist_release_attestation",
        lambda *_args, **_kwargs: order.append("release-attestation"),
    )
    monkeypatch.setattr(orchestrator, "PilotRunLedger", FakeRunLedger)
    monkeypatch.setattr(orchestrator, "PilotBudgetLedger", FakeBudgetLedger)
    monkeypatch.setattr(orchestrator, "_budget_caps", lambda _contract: object())
    monkeypatch.setattr(orchestrator, "_parent_budget_debit", debit)
    monkeypatch.setattr(
        orchestrator,
        "_execute_v21110_parent_import_stage",
        execute_parent,
        raising=False,
    )
    monkeypatch.setattr(
        orchestrator,
        "_execute_v24_parent_import_stage",
        _forbidden("legacy parent executor"),
    )
    monkeypatch.setattr(orchestrator, "_provider_for_profile", _forbidden("provider"))

    result = orchestrator._execute_stage_locked(
        contract_path=CONTRACT_PATH,
        stage_id="parent-import",
        resume=False,
        raw_root=tmp_path / "raw",
        repo_root=tmp_path,
        parent_repo_root=tmp_path / "failed-v2119",
        authority_repo_root=tmp_path / "authority-v2115",
    )

    assert result is expected
    assert "run-ledger:bind-terminal=True" in order
    assert order == [
        "keys-absent",
        "source-manifest",
        "paid-provenance",
        "release-attestation",
        "run-ledger:bind-terminal=True",
        "parent-debit",
        "budget-ledger",
        "parent-executor",
    ]


def test_v21110_source_manifest_failure_precedes_provenance_ledgers_and_provider(
    contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    def invalid_source(**_kwargs: Any) -> None:
        order.append("source-manifest")
        raise orchestrator.PilotV21110ContinuationError("fixture source drift")

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        orchestrator,
        "require_v21110_provider_keys_absent",
        lambda: order.append("keys-absent"),
    )
    monkeypatch.setattr(orchestrator, "validate_v21110_source_manifest", invalid_source)
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        _forbidden("paid provenance"),
    )
    monkeypatch.setattr(orchestrator, "PilotRunLedger", _forbidden("run ledger"))
    monkeypatch.setattr(orchestrator, "PilotBudgetLedger", _forbidden("budget ledger"))
    monkeypatch.setattr(orchestrator, "_provider_for_profile", _forbidden("provider"))

    with pytest.raises(
        orchestrator.PilotOrchestrationError,
        match=(
            "V2.11.10 source-manifest replay failed before release-attestation, "
            "ledger, budget, or receipt writes: fixture source drift"
        ),
    ):
        orchestrator._execute_stage_locked(
            contract_path=CONTRACT_PATH,
            stage_id="parent-import",
            resume=False,
            raw_root=tmp_path / "raw",
            repo_root=tmp_path,
            parent_repo_root=tmp_path / "failed-v2119",
            authority_repo_root=tmp_path / "authority-v2115",
        )

    assert order == ["keys-absent", "source-manifest"]


def test_v21110_scientific_validation_precedes_reserve_and_provider_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    spec = SimpleNamespace(
        run_id="finevo-pilot-v2.11.10--experiment-b--fixture",
        contract_id=orchestrator.V21110_CONTRACT_ID,
        stage_id="experiment-b",
        model_id="gpt52_main",
        environment_seed=1099057501,
        budget_bucket="hosted_v21110",
        execution_mode="actor_run",
        arm_id="full-evidence-grounded",
    )
    stage = SimpleNamespace(
        stage_id="experiment-b",
        enabled=True,
        prerequisites=(),
        cells=(spec,),
    )
    contract = SimpleNamespace(
        contract_id=orchestrator.V21110_CONTRACT_ID,
        schema_version="finevo-pilot-contract-v2",
        canonical_hash="c" * 64,
        stage_ids=("experiment-b",),
        stages=(stage,),
        model_roles={},
        provider_profiles={
            "gpt52_main": SimpleNamespace(transport="diagnostic")
        },
        evaluator_amendment=None,
        stage=lambda stage_id: stage,
        expand=lambda stage=None, model=None: (spec,),
        models_for_stage=lambda stage_id: ("gpt52_main",),
    )
    projection = RunProjection(
        run_id=spec.run_id,
        stage_bucket=spec.budget_bucket,
        cost_usd=1.0,
        completions=1,
        storage_bytes=100,
        basis={"fixture": True},
    )
    receipt_path = tmp_path / "stage_receipt.json"

    class FakeRunLedger:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            order.append(
                f"run-ledger:bind-terminal={kwargs['bind_terminal_artifacts']}"
            )
            self.rows: dict[str, dict[str, Any]] = {}

        def register(self, specs: Any) -> None:
            for item in specs:
                self.rows.setdefault(
                    item.run_id,
                    {"status": "scheduled", "artifact": None, "failure": None},
                )

        def status(self, run_id: str) -> str:
            return str(self.rows[run_id]["status"])

        def is_terminal(self, run_id: str) -> bool:
            return self.status(run_id) != "scheduled"

        def finalize(
            self,
            run_id: str,
            *,
            status: str,
            artifact: str | None,
            failure: Any,
        ) -> None:
            self.rows[run_id] = {
                "status": status,
                "artifact": artifact,
                "failure": failure,
            }

        def stop_pending(self, specs: Any, *, status: str, failure: Any) -> None:
            for item in specs:
                if not self.is_terminal(item.run_id):
                    self.finalize(
                        item.run_id,
                        status=status,
                        artifact=None,
                        failure=failure,
                    )

    class FakeBudgetLedger:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def reserve(self, observed: RunProjection) -> None:
            assert observed == projection
            order.append("reserve")

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(orchestrator, "_persist_release_attestation", lambda *_a: None)
    monkeypatch.setattr(orchestrator, "PilotRunLedger", FakeRunLedger)
    monkeypatch.setattr(orchestrator, "PilotBudgetLedger", FakeBudgetLedger)
    monkeypatch.setattr(orchestrator, "_budget_caps", lambda _contract: object())
    monkeypatch.setattr(orchestrator, "_parent_budget_debit", lambda _contract: None)
    monkeypatch.setattr(
        orchestrator,
        "verify_v21110_scientific_dispatch_acceptance",
        lambda *_args, **_kwargs: order.append("acceptance"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_recover_v21110_interrupted_reservations_before_dispatch",
        lambda *_args, **_kwargs: order.append("recovery") or (),
    )
    monkeypatch.setattr(
        orchestrator,
        "_assert_v21110_local_release_guard",
        lambda *_args, **_kwargs: order.append("release-guard"),
    )
    monkeypatch.setattr(
        orchestrator,
        "audit_v21110_scientific_stage_namespace",
        lambda *_args, **_kwargs: order.append("namespace"),
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v21110_terminal_scientific_artifacts",
        lambda *_args, **_kwargs: order.append("terminal-replay"),
    )
    monkeypatch.setattr(
        orchestrator,
        "assert_v21110_dispatch_target_fresh",
        lambda *_args, **_kwargs: order.append("fresh-target"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_assert_prerequisites",
        lambda *_args, **_kwargs: order.append("prerequisites") or {},
    )
    monkeypatch.setattr(
        orchestrator,
        "_remaining_core_projections",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        orchestrator,
        "_assert_projection_matrix_fits",
        lambda *_args, **_kwargs: order.append("matrix-fit"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_recover_or_stop_interrupted_reservation",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        orchestrator,
        "projection_from_preflight",
        lambda *_args, **_kwargs: projection,
    )
    monkeypatch.setattr(
        orchestrator,
        "_run_budget_from_projection",
        lambda _projection: order.append("run-budget") or object(),
    )
    monkeypatch.setattr(orchestrator, "_provider_for_profile", _forbidden("provider"))

    def actor(*_args: Any, **_kwargs: Any) -> tuple[Path, object, dict[str, Any]]:
        order.append("provider-construction")
        return tmp_path / "manifest.json", object(), {}

    monkeypatch.setattr(orchestrator, "_execute_actor_run", actor)
    monkeypatch.setattr(
        orchestrator,
        "_finalize_budget_safely",
        lambda *_args, **_kwargs: ("complete", None, {}),
    )
    monkeypatch.setattr(orchestrator, "_propagate_stage_no_go", lambda *_a, **_k: None)
    monkeypatch.setattr(
        orchestrator,
        "_write_stage_receipt",
        lambda *_args, **_kwargs: receipt_path,
    )
    monkeypatch.setattr(
        orchestrator,
        "_read_json",
        lambda _path: {"execution_progression_go": True},
    )

    result = orchestrator._execute_stage_locked(
        contract_path=CONTRACT_PATH,
        stage_id="experiment-b",
        resume=False,
        raw_root=tmp_path / "raw",
        repo_root=tmp_path,
    )

    assert result == {"execution_progression_go": True}
    required_validations = {
        "acceptance",
        "recovery",
        "release-guard",
        "namespace",
        "terminal-replay",
        "fresh-target",
    }
    assert required_validations <= set(order), (
        f"missing V2.11.10 pre-dispatch validations: "
        f"{sorted(required_validations - set(order))}; order={order}"
    )
    assert "run-ledger:bind-terminal=True" in order
    assert order.count("namespace") >= 2
    assert order.count("terminal-replay") >= 2
    reserve_index = order.index("reserve")
    provider_index = order.index("provider-construction")
    for validation in (
        "acceptance",
        "recovery",
        "release-guard",
        "namespace",
        "terminal-replay",
        "fresh-target",
    ):
        assert max(index for index, value in enumerate(order) if value == validation) < (
            reserve_index
        )
    assert reserve_index < provider_index


def test_v21110_development_fake_is_scripted_and_never_scientific(
    contract: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(orchestrator, "_provider_for_profile", _forbidden("provider"))
    monkeypatch.setattr(orchestrator, "create_llm_provider", _forbidden("provider"))

    result = orchestrator.run_development_fake_matrix(
        contract_path=CONTRACT_PATH,
        resume=False,
        raw_root=tmp_path,
    )

    assert result["status"] == "pass"
    assert result["registered_cells"] == 18
    assert result["status_counts"] == {"complete": 18}
    assert result["stages"] == ["experiment-d", "experiment-b", "cross-model"]
    assert result["diagnostic_only"] is True
    assert result["scientific_evidence"] is False
    assert "No row is model performance" in result["claim_boundary"]
