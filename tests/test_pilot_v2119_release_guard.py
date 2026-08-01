from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.pilot_budget import RunProjection
from verified_memory.pilot_contract import load_pilot_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments/pilot_v2_11_9.yaml"
COMMIT = "a" * 40
TAG_OBJECT = "b" * 40
TAG = "pilot-v2.11.9-science"


def _guard_contract() -> SimpleNamespace:
    return SimpleNamespace(
        contract_id=orchestrator.V2119_CONTRACT_ID,
        implementation={"required_git_tag": TAG},
    )


def _paid() -> orchestrator.GitProvenance:
    return orchestrator.GitProvenance(
        git_tag=TAG,
        head_commit=COMMIT,
        tag_commit=COMMIT,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={"fixture": True},
        release_attestation={
            "local_tag": {
                "name": TAG,
                "object_id": TAG_OBJECT,
                "peeled_commit": COMMIT,
                "kind": "annotated",
            }
        },
    )


def _git_observations(*, drift: str | None = None) -> dict[tuple[str, ...], str]:
    return {
        ("rev-parse", "HEAD"): "c" * 40 if drift == "head" else COMMIT,
        ("cat-file", "-t", f"refs/tags/{TAG}"): (
            "commit" if drift == "tag-type" else "tag"
        ),
        ("rev-parse", f"refs/tags/{TAG}"): (
            "d" * 40 if drift == "tag-object" else TAG_OBJECT
        ),
        ("rev-parse", f"refs/tags/{TAG}^{{commit}}"): (
            "e" * 40 if drift == "peeled" else COMMIT
        ),
        ("status", "--porcelain", "--untracked-files=all"): (
            "?? drift.txt" if drift == "worktree" else ""
        ),
    }


def test_v2119_local_release_guard_rechecks_exact_local_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = _git_observations()
    commands: list[tuple[str, ...]] = []
    profile_roots: list[Path] = []

    def fake_git(repo_root: Path, *args: str, **_kwargs: Any) -> str:
        assert repo_root == tmp_path.resolve()
        commands.append(args)
        return observations[args]

    monkeypatch.setattr(orchestrator, "_git", fake_git)
    monkeypatch.setattr(
        orchestrator,
        "_verify_v2119_bound_working_directory",
        lambda root: profile_roots.append(Path(root)),
    )

    orchestrator._assert_v2119_local_release_guard(
        _guard_contract(),
        repo_root=tmp_path,
        paid=_paid(),
    )

    assert commands == [
        ("rev-parse", "HEAD"),
        ("cat-file", "-t", f"refs/tags/{TAG}"),
        ("rev-parse", f"refs/tags/{TAG}"),
        ("rev-parse", f"refs/tags/{TAG}^{{commit}}"),
        ("status", "--porcelain", "--untracked-files=all"),
    ]
    assert profile_roots == [tmp_path.resolve()]


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("head", "HEAD drifted"),
        ("tag-type", "no longer an annotated tag"),
        ("tag-object", "tag object drifted"),
        ("peeled", "tag target drifted"),
        ("worktree", "worktree drifted"),
    ),
)
def test_v2119_local_release_guard_fails_closed_on_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    message: str,
) -> None:
    observations = _git_observations(drift=drift)
    profile_checks = 0

    def fake_git(_repo_root: Path, *args: str, **_kwargs: Any) -> str:
        return observations[args]

    def profile_check(_root: Path) -> None:
        nonlocal profile_checks
        profile_checks += 1

    monkeypatch.setattr(orchestrator, "_git", fake_git)
    monkeypatch.setattr(
        orchestrator,
        "_verify_v2119_bound_working_directory",
        profile_check,
    )

    with pytest.raises(orchestrator.PilotOrchestrationError, match=message):
        orchestrator._assert_v2119_local_release_guard(
            _guard_contract(),
            repo_root=tmp_path,
            paid=_paid(),
        )
    assert profile_checks == 0


def test_v2119_second_b_unit_source_drift_stops_remaining_itt_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_pilot_contract(CONTRACT_PATH)
    stage_specs = contract.expand(stage="experiment-b")
    observed: dict[str, Any] = {"provider_run_ids": [], "guard_run_ids": []}

    class FakeRunLedger:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.rows: dict[str, dict[str, Any]] = {}
            observed["run_ledger"] = self

        def register(self, specs: Any) -> None:
            for spec in specs:
                self.rows.setdefault(
                    spec.run_id,
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
            assert self.rows[run_id]["status"] == "scheduled"
            self.rows[run_id] = {
                "status": status,
                "artifact": artifact,
                "failure": failure,
            }

        def stop_pending(
            self,
            specs: Any,
            *,
            status: str,
            failure: Any,
        ) -> None:
            for spec in specs:
                if not self.is_terminal(spec.run_id):
                    self.finalize(
                        spec.run_id,
                        status=status,
                        artifact=None,
                        failure=failure,
                    )

    class FakeBudgetLedger:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.reserved: list[str] = []

        def reserve(self, projection: RunProjection) -> None:
            self.reserved.append(projection.run_id)

    paid = _paid()
    receipt_path = tmp_path / "stage-receipt.json"

    monkeypatch.setattr(orchestrator, "load_pilot_contract", lambda _path: contract)
    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: paid,
    )
    monkeypatch.setattr(orchestrator, "_persist_release_attestation", lambda *_a: None)
    monkeypatch.setattr(orchestrator, "PilotRunLedger", FakeRunLedger)
    monkeypatch.setattr(orchestrator, "PilotBudgetLedger", FakeBudgetLedger)
    monkeypatch.setattr(orchestrator, "_budget_caps", lambda _contract: object())
    monkeypatch.setattr(orchestrator, "_parent_budget_debit", lambda _contract: None)
    monkeypatch.setattr(
        orchestrator,
        "verify_v2119_scientific_dispatch_acceptance",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        orchestrator,
        "_recover_v2119_interrupted_reservations_before_dispatch",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        orchestrator,
        "audit_v2119_scientific_stage_namespace",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v2119_terminal_scientific_artifacts",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        orchestrator,
        "_assert_prerequisites",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        orchestrator,
        "validate_live_provider_catalog",
        lambda *_args, **_kwargs: {"fixture": "catalog"},
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_provider_catalog_receipt",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        orchestrator,
        "_atomic_json_no_overwrite",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator,
        "_remaining_core_projections",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        orchestrator,
        "_assert_projection_matrix_fits",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator,
        "_recover_or_stop_interrupted_reservation",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        orchestrator,
        "projection_from_preflight",
        lambda _contract, spec, **_kwargs: RunProjection(
            run_id=spec.run_id,
            stage_bucket=spec.budget_bucket,
            cost_usd=0.0,
            completions=1,
            storage_bytes=0,
            basis={"fixture": True},
        ),
    )
    monkeypatch.setattr(
        orchestrator, "_run_budget_from_projection", lambda _p: object()
    )
    monkeypatch.setattr(
        orchestrator,
        "assert_v2119_dispatch_target_fresh",
        lambda *_args, **_kwargs: None,
    )

    def execute_actor(
        _contract: Any, spec: Any, **_kwargs: Any
    ) -> tuple[Path, Any, dict]:
        observed["provider_run_ids"].append(spec.run_id)
        return tmp_path / "manifest.json", object(), {}

    monkeypatch.setattr(orchestrator, "_execute_actor_run", execute_actor)
    monkeypatch.setattr(
        orchestrator,
        "_finalize_budget_safely",
        lambda *_args, **_kwargs: ("complete", None, {}),
    )

    def release_guard(_contract: Any, **_kwargs: Any) -> None:
        next_index = len(observed["guard_run_ids"])
        observed["guard_run_ids"].append(stage_specs[next_index].run_id)
        if next_index == 1:
            raise orchestrator.PilotOrchestrationError("simulated source drift")

    monkeypatch.setattr(
        orchestrator,
        "_assert_v2119_local_release_guard",
        release_guard,
    )
    monkeypatch.setattr(
        orchestrator,
        "_propagate_stage_no_go",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator,
        "_write_stage_receipt",
        lambda *_args, **_kwargs: receipt_path,
    )
    monkeypatch.setattr(
        orchestrator,
        "_read_json",
        lambda _path: {"execution_progression_go": False},
    )

    orchestrator._execute_stage_locked(
        contract_path=CONTRACT_PATH,
        stage_id="experiment-b",
        resume=True,
        raw_root=tmp_path / "raw",
        repo_root=ROOT,
    )

    ledger = observed["run_ledger"]
    assert observed["provider_run_ids"] == [stage_specs[0].run_id]
    assert observed["guard_run_ids"] == [
        stage_specs[0].run_id,
        stage_specs[1].run_id,
    ]
    assert ledger.status(stage_specs[0].run_id) == "complete"
    assert {ledger.status(spec.run_id) for spec in stage_specs[1:]} == {
        "integrity-stopped"
    }
    assert all(
        ledger.rows[spec.run_id]["failure"]
        == {
            "error_type": "PreDispatchIntegrityStop",
            "cause_type": "PilotOrchestrationError",
            "message": "simulated source drift",
            "run_id": stage_specs[1].run_id,
            "provider_dispatch_started": False,
            "stop_origin": "actor-pre-provider-revalidation",
        }
        for spec in stage_specs[1:]
    )
