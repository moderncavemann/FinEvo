from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from llm_providers import MultiModelLLM
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory.budget import UsageRecord
from verified_memory.pilot_budget import ParentBudgetDebit
from verified_memory.pilot_contract import PilotContract, load_pilot_contract
from verified_memory.pilot_v2112_gate import (
    V2112_PREFLIGHT_EXACTNESS_SCHEMA_VERSION,
    verify_v2112_gate_receipt,
)
from verified_memory.pilot_v2112_bootstrap import (
    V2112_BOOTSTRAP_PROJECTION_FILENAME,
    V2112_BOOTSTRAP_SCHEMA_VERSION,
)
from verified_memory.pilot_v2112_parent_import import (
    V2112_CAPABILITY_WRAPPER_SCHEMA_VERSION,
    V2112_PARENT_IMPORT_SCHEMA_VERSION,
    verify_v2112_parent_import_receipt,
)
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_2.yaml"
PARENT_ROOT = ROOT.parent / "finevo-pilot-v2-11-1-science"
GRANDPARENT_ROOT = ROOT.parent / "finevo-pilot-v2-11-science"
MODELS = ("gpt52_main", "gpt56_diagnostic")
SCIENCE_STAGES = (
    "experiment-c",
    "experiment-a",
    "experiment-d",
    "experiment-b",
    "cross-model",
)


def _contract():
    return load_pilot_contract(CONTRACT_PATH)


def _paid() -> orchestrator.GitProvenance:
    return orchestrator.GitProvenance(
        git_tag="pilot-v2.11.2-science",
        head_commit="c" * 40,
        tag_commit="c" * 40,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
    )


def _require_immutable_sources() -> None:
    missing = [path for path in (PARENT_ROOT, GRANDPARENT_ROOT) if not path.is_dir()]
    if missing:
        pytest.skip(
            "immutable V2.11.1/V2.11 science worktrees are unavailable: "
            + ", ".join(str(path) for path in missing)
        )


def _forbidden_provider(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("V2.11.2 zero-provider import constructed a provider")


class _ProfileScriptedProvider(ScriptedDiagnosticProvider):
    """No-network provider that emits the exact frozen profile metadata."""

    def __init__(self, profile: Any) -> None:
        self.profile = profile
        self.calls = 0

    def get_model_name(self) -> str:
        return f"openai/{self.profile.requested_model}"

    def get_structured_completion(self, messages: Any, **kwargs: Any) -> Any:
        result = super().get_structured_completion(messages, **kwargs)
        self.calls += 1
        rates = self.profile.price_snapshot.costs_per_1k()
        cost_usd = (
            result.usage.prompt_tokens * rates["prompt"]
            + result.usage.completion_tokens * rates["completion"]
        ) / 1000.0
        dispatch = tuple(
            (
                field,
                (
                    "explicit_supported"
                    if disposition.dispatch_mode == "explicit_supported"
                    else "omitted_unsupported"
                ),
            )
            for field, disposition in self.profile.decoding_fields
        )
        request_parameters = tuple(
            sorted(
                {
                    "max_completion_tokens",
                    "messages",
                    "model",
                    *self.profile.openai_request_options().keys(),
                }
            )
        )
        return replace(
            result,
            usage=UsageRecord(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                cost_usd=cost_usd,
            ),
            model=self.profile.requested_model,
            provider="openai",
            request_id=f"req_v2112_{self.profile.profile_id}_{self.calls:02d}",
            response_model=self.profile.served_model,
            response_provider="OpenAI-direct",
            response_route="direct",
            request_profile_id=self.profile.profile_id,
            request_provider_pin=tuple(self.profile.provider_pin),
            request_artifact_identity=tuple(self.profile.artifact_identity),
            request_price_snapshot_source=self.profile.price_snapshot.source,
            request_price_snapshot_captured_at=(
                self.profile.price_snapshot.captured_at
            ),
            finish_reason="stop",
            native_finish_reason="stop",
            response_completed=True,
            provider_sdk_name="openai-python",
            provider_sdk_version="2.46.0",
            request_parameters=request_parameters,
            temperature_dispatch="omitted_unsupported",
            parameter_dispatch=dispatch,
        )


def _surrogate_provenance(
    contract: PilotContract,
    git_commit: str,
    git_tag: str,
) -> dict[str, Any]:
    """Bind test artifacts without pretending the draft release is frozen."""

    return {
        "git_tag": git_tag,
        "resolved_git_commit": git_commit,
        "commit_resolution": contract.implementation["commit_resolution"],
        "p0_base_commit": contract.implementation["p0_base_commit"],
        "contract_id": contract.contract_id,
        "contract_sha256": contract.canonical_hash,
    }


def _live_import_case() -> Iterator[dict[str, Any]]:
    """Run the exact parent/capability import against immutable local sources."""

    _require_immutable_sources()
    contract = _contract()
    paid = _paid()
    experiment_results = ROOT / "experiment_results"
    experiment_results.mkdir(exist_ok=True)
    temporary = tempfile.TemporaryDirectory(
        prefix=".pytest-v2112-orchestrator-",
        dir=experiment_results,
    )
    raw_root = Path(temporary.name) / "raw"
    raw_root.mkdir(parents=True)
    run_ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    run_ledger.register(contract.expand())
    boundary = contract.v2112_forward_boundary
    assert boundary is not None
    budget_ledger = orchestrator.PilotBudgetLedger(
        raw_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=ParentBudgetDebit.from_dict(boundary["parent_budget_debit"]),
    )
    patch = pytest.MonkeyPatch()
    patch.setattr(orchestrator, "_provider_for_profile", _forbidden_provider)
    patch.setattr(orchestrator, "create_llm_provider", _forbidden_provider)
    patch.setattr(PilotContract, "validate_provenance", _surrogate_provenance)
    try:
        parent_stage = orchestrator._execute_v2112_parent_import_stage(
            contract,
            contract.expand(stage="parent-import"),
            raw_root=raw_root,
            repo_root=ROOT,
            parent_repo_root=PARENT_ROOT,
            paid=paid,
            run_ledger=run_ledger,
            budget_ledger=budget_ledger,
        )
        capability_stage = orchestrator._execute_v2112_capability_import_stage(
            contract,
            contract.expand(stage="capability-gate"),
            raw_root=raw_root,
            repo_root=ROOT,
            paid=paid,
            run_ledger=run_ledger,
            budget_ledger=budget_ledger,
        )
        yield {
            "contract": contract,
            "paid": paid,
            "raw_root": raw_root,
            "run_ledger": run_ledger,
            "budget_ledger": budget_ledger,
            "parent_stage": parent_stage,
            "capability_stage": capability_stage,
        }
    finally:
        patch.undo()
        temporary.cleanup()


@pytest.fixture(scope="module")
def live_import() -> Iterator[dict[str, Any]]:
    yield from _live_import_case()


@pytest.fixture
def live_preflight_case() -> Iterator[dict[str, Any]]:
    """Isolate the stateful 64-call offline preflight from import tests."""

    yield from _live_import_case()


def test_v2112_real_parent_and_capability_import_are_zero_provider(
    live_import: dict[str, Any],
) -> None:
    contract = live_import["contract"]
    paid = live_import["paid"]
    raw_root = live_import["raw_root"]
    run_ledger = live_import["run_ledger"]
    parent_stage = live_import["parent_stage"]
    capability_stage = live_import["capability_stage"]
    receipt_path = raw_root / "parent-import" / "parent_import_receipt.json"

    receipt = verify_v2112_parent_import_receipt(
        receipt_path,
        repo_root=ROOT,
        parent_science_root=PARENT_ROOT,
        grandparent_science_root=GRANDPARENT_ROOT,
        child_contract_sha256=contract.canonical_hash,
        child_git_tag=paid.git_tag,
        child_git_commit=paid.head_commit,
    )

    assert receipt["schema_version"] == V2112_PARENT_IMPORT_SCHEMA_VERSION
    assert receipt["import_policy"]["provider_calls_during_import"] == 0
    assert receipt["import_policy"]["imported_preflight_samples"] == 0
    assert receipt["import_policy"]["imported_checkpoint_artifacts"] == []
    assert receipt["import_policy"]["imported_p95_authorities"] == []
    assert receipt["import_policy"]["imported_effect_cells"] == 0
    assert receipt["terminal_parent_denominator"]["status_counts"] == {
        "complete": 3,
        "failed": 2,
        "integrity-stopped": 131,
    }
    assert parent_stage["status"] == "complete"
    assert parent_stage["registered_run_count"] == 1
    assert capability_stage["status"] == "complete"
    assert capability_stage["go_models"] == list(MODELS)

    for model_id in MODELS:
        spec = contract.expand(stage="capability-gate", model=model_id)[0]
        run_dir = raw_root / spec.stage_id / "runs" / spec.run_id
        wrapper = orchestrator._read_json(run_dir / "capability.json")
        gate = orchestrator._read_json(run_dir / "gate_receipt.json")
        bootstrap_path = run_dir / V2112_BOOTSTRAP_PROJECTION_FILENAME
        bootstrap = orchestrator._read_json(bootstrap_path)

        assert wrapper["schema_version"] == (V2112_CAPABILITY_WRAPPER_SCHEMA_VERSION)
        assert wrapper["provider_calls_current_attempt"] == 0
        assert wrapper["imported_preflight_samples"] == 0
        assert wrapper["imported_checkpoint_artifacts"] == []
        assert wrapper["imported_p95_authorities"] == []
        assert gate["go"] is True
        assert gate["provider_calls_current_attempt"] == 0
        assert gate["bootstrap_projection"] == str(bootstrap_path)
        assert bootstrap["schema_version"] == V2112_BOOTSTRAP_SCHEMA_VERSION
        assert bootstrap["target"]["contract_id"] == contract.contract_id
        assert bootstrap["source"]["contract_id"] == ("finevo-pilot-v2.11.1")
        assert "capability-gate" in bootstrap["source"]["capability_path"]
        assert "long-context-preflight" not in bootstrap["source"]["capability_path"]
        assert "journal" not in bootstrap["source"]["capability_path"]
        assert bootstrap["capability_projection"]["action"]["sample_count"] == 24
        assert bootstrap["capability_projection"]["semantic"]["sample_count"] == 6

    snapshot = run_ledger.snapshot()
    statuses = [row["status"] for row in snapshot["runs"].values()]
    assert statuses.count("complete") == 3
    assert statuses.count("scheduled") == 133


def test_v2112_orchestrator_bootstrap_roundtrip_uses_only_v2111_capability(
    live_import: dict[str, Any],
) -> None:
    contract = live_import["contract"]
    paid = live_import["paid"]
    raw_root = live_import["raw_root"]

    for model_id in MODELS:
        preflight_spec = contract.expand(
            stage="long-context-preflight",
            model=model_id,
        )[0]
        projection, path, reservations = orchestrator._load_v2112_bootstrap_projection(
            contract,
            preflight_spec=preflight_spec,
            raw_root=raw_root,
            repo_root=ROOT,
            paid=paid,
        )

        assert path.name == V2112_BOOTSTRAP_PROJECTION_FILENAME
        assert projection["source"]["contract_id"] == ("finevo-pilot-v2.11.1")
        runtime_model = projection["model"]["runtime_model"]
        assert set(reservations) == {runtime_model}
        by_kind = reservations[runtime_model]
        assert set(by_kind) == {"action", "semantic"}
        assert all(
            row["authority"]["target_contract_id"] == contract.contract_id
            for row in by_kind.values()
        )
        assert all(
            row["authority"]["source_contract_id"] == "finevo-pilot-v2.11.1"
            for row in by_kind.values()
        )
        assert all(
            row["authority"]["authorized_seed"] == 2010922376
            for row in by_kind.values()
        )


def test_v2112_stage_controls_bind_the_new_bootstrap_schema(
    live_import: dict[str, Any],
) -> None:
    contract = live_import["contract"]
    raw_root = live_import["raw_root"]
    expected = {
        raw_root
        / "capability-gate"
        / "runs"
        / contract.expand(stage="capability-gate", model=model_id)[0].run_id
        / V2112_BOOTSTRAP_PROJECTION_FILENAME
        for model_id in MODELS
    }

    controls = set(
        orchestrator._v2_stage_control_paths(
            contract,
            "capability-gate",
            raw_root=raw_root,
        )
    )

    assert expected <= controls


def test_v2112_loader_rejects_v2111_bootstrap_as_current_authority(
    live_import: dict[str, Any],
) -> None:
    contract = live_import["contract"]
    paid = live_import["paid"]
    raw_root = live_import["raw_root"]
    model_id = "gpt52_main"
    capability_spec = contract.expand(
        stage="capability-gate",
        model=model_id,
    )[0]
    preflight_spec = contract.expand(
        stage="long-context-preflight",
        model=model_id,
    )[0]
    target = (
        raw_root
        / capability_spec.stage_id
        / "runs"
        / capability_spec.run_id
        / V2112_BOOTSTRAP_PROJECTION_FILENAME
    )
    original = target.read_bytes()
    source_spec = load_pilot_contract(
        PARENT_ROOT / "experiments" / "pilot_v2_11_1.yaml"
    ).expand(stage="capability-gate", model=model_id)[0]
    stale = (
        PARENT_ROOT
        / "experiment_results"
        / "pilot-v2.11.1"
        / "raw"
        / source_spec.stage_id
        / "runs"
        / source_spec.run_id
        / orchestrator.V2111_BOOTSTRAP_PROJECTION_FILENAME
    )
    assert stale.is_file()
    original_mode = stat.S_IMODE(target.stat().st_mode)

    try:
        target.chmod(original_mode | stat.S_IWUSR)
        target.write_bytes(stale.read_bytes())
        with pytest.raises(
            orchestrator.PilotOrchestrationError,
            match=r"V2\.11\.2 bootstrap load failed validation",
        ):
            orchestrator._load_v2112_bootstrap_projection(
                contract,
                preflight_spec=preflight_spec,
                raw_root=raw_root,
                repo_root=ROOT,
                paid=paid,
            )
    finally:
        target.chmod(original_mode | stat.S_IWUSR)
        target.write_bytes(original)
        target.chmod(original_mode)


def test_v2112_family_verifier_is_selected_over_older_verifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    observed: list[str] = []
    expected = {"schema_version": "fixture", "go": False}

    monkeypatch.setattr(
        orchestrator,
        "verify_v2112_gate_receipt",
        lambda value, **_kwargs: observed.append("v2112") or dict(value),
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v2111_gate_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("V2.11.1 verifier selected for V2.11.2")
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_v211_gate_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("V2.11 verifier selected for V2.11.2")
        ),
    )

    assert (
        orchestrator._verify_v211_family_post_gate_receipt(
            contract,
            expected,
            paid=_paid(),
        )
        == expected
    )
    assert observed == ["v2112"]


@pytest.mark.parametrize(
    ("stage_id", "selected_name", "forbidden_name"),
    [
        (
            "parent-import",
            "_execute_v2112_parent_import_stage",
            "_execute_v211_parent_import_stage",
        ),
        (
            "capability-gate",
            "_execute_v2112_capability_import_stage",
            "_execute_v2111_capability_import_stage",
        ),
    ],
)
def test_v2112_stage_dispatch_selects_only_v2112_executors(
    stage_id: str,
    selected_name: str,
    forbidden_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"status": "provider-free-v2112-dispatch", "stage": stage_id}
    calls: list[str] = []

    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: _paid(),
    )
    monkeypatch.setattr(
        orchestrator,
        "_persist_release_attestation",
        lambda root, _paid_value: root / "release_attestation.json",
    )
    monkeypatch.setattr(
        orchestrator,
        "_parent_budget_debit",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator,
        selected_name,
        lambda *_args, **_kwargs: calls.append(selected_name) or expected,
    )
    monkeypatch.setattr(
        orchestrator,
        forbidden_name,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(f"{forbidden_name} selected for V2.11.2")
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "_provider_for_profile",
        _forbidden_provider,
    )
    monkeypatch.setattr(
        orchestrator,
        "create_llm_provider",
        _forbidden_provider,
    )

    result = orchestrator._execute_stage_locked(
        contract_path=CONTRACT_PATH,
        stage_id=stage_id,
        resume=False,
        raw_root=tmp_path / stage_id,
        repo_root=tmp_path,
        parent_repo_root=(
            tmp_path / "immutable-v2111" if stage_id == "parent-import" else None
        ),
    )

    assert result == expected
    assert calls == [selected_name]


def test_v2112_wrong_parent_release_stops_the_full_registered_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_immutable_sources()
    contract = _contract()
    paid = _paid()
    experiment_results = ROOT / "experiment_results"
    experiment_results.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".pytest-v2112-wrong-parent-",
        dir=experiment_results,
    ) as temporary:
        raw_root = Path(temporary) / "raw"
        raw_root.mkdir(parents=True)
        run_ledger = orchestrator.PilotRunLedger(
            raw_root / "run_ledger.json",
            contract_hash=contract.canonical_hash,
            tamper_evident=True,
        )
        run_ledger.register(contract.expand())
        boundary = contract.v2112_forward_boundary
        assert boundary is not None
        budget_ledger = orchestrator.PilotBudgetLedger(
            raw_root / "budget_ledger.json",
            contract_hash=contract.canonical_hash,
            caps=orchestrator._budget_caps(contract),
            tamper_evident=True,
            parent_debit=ParentBudgetDebit.from_dict(boundary["parent_budget_debit"]),
        )
        monkeypatch.setattr(
            orchestrator,
            "_provider_for_profile",
            _forbidden_provider,
        )
        monkeypatch.setattr(
            orchestrator,
            "create_llm_provider",
            _forbidden_provider,
        )
        monkeypatch.setattr(
            PilotContract,
            "validate_provenance",
            _surrogate_provenance,
        )

        with pytest.raises(
            orchestrator.PilotOrchestrationError,
            match=r"V2\.11\.2 parent import failed",
        ):
            orchestrator._execute_v2112_parent_import_stage(
                contract,
                contract.expand(stage="parent-import"),
                raw_root=raw_root,
                repo_root=ROOT,
                parent_repo_root=GRANDPARENT_ROOT,
                paid=paid,
                run_ledger=run_ledger,
                budget_ledger=budget_ledger,
            )

        rows = run_ledger.snapshot()["runs"]
        assert len(rows) == 136
        assert {row["status"] for row in rows.values()} == {"integrity-stopped"}
        assert not any(
            row.get("actual", {}).get("completions", 0)
            for row in budget_ledger.snapshot()["runs"].values()
        )


def test_v2112_mixed_preflight_failure_is_global_no_go_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    raw_root = tmp_path / "raw"
    ledger = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    ledger.register(contract.expand())
    for stage_id in ("parent-import", "capability-gate"):
        for spec in contract.expand(stage=stage_id):
            ledger.finalize(spec.run_id, status="complete", artifact=None)

    monkeypatch.setattr(
        orchestrator,
        "verify_paid_provenance",
        lambda *_args, **_kwargs: _paid(),
    )
    monkeypatch.setattr(
        PilotContract,
        "validate_provenance",
        _surrogate_provenance,
    )
    monkeypatch.setattr(
        orchestrator,
        "_persist_release_attestation",
        lambda root, _paid_value: root / "release_attestation.json",
    )
    monkeypatch.setattr(
        orchestrator,
        "_parent_budget_debit",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator,
        "_assert_prerequisites",
        lambda *_args, **_kwargs: {"capability-gate": frozenset(MODELS)},
    )

    def fake_catalog(_contract: Any, *, model_ids: Any) -> dict[str, Any]:
        model_id = tuple(model_ids)[0]
        receipt = {
            "schema_version": "provider-free-catalog-fixture-v1",
            "contract_sha256": contract.canonical_hash,
            "rows": [{"profile_id": model_id}],
        }
        receipt["receipt_sha256"] = orchestrator.canonical_sha256(receipt)
        return receipt

    monkeypatch.setattr(
        orchestrator,
        "validate_live_provider_catalog",
        fake_catalog,
    )
    monkeypatch.setattr(
        orchestrator,
        "verify_provider_catalog_receipt",
        lambda value, **_kwargs: dict(value),
    )

    def fake_preflight(
        _contract: Any,
        spec: Any,
        *,
        raw_root: Path,
        **_kwargs: Any,
    ) -> tuple[str, Path, SimpleNamespace, dict[str, Any]]:
        if spec.model_id == "gpt52_main":
            raise ValueError("provider-free mixed V2.11.2 preflight failure")
        gate = {"go": True, "reason": "provider-free local pass"}
        gate_path = (
            raw_root / spec.stage_id / "runs" / spec.run_id / "gate_receipt.json"
        )
        orchestrator._atomic_json(gate_path, gate)
        return "complete", gate_path, SimpleNamespace(), gate

    monkeypatch.setattr(
        orchestrator,
        "_execute_capability_preflight",
        fake_preflight,
    )
    monkeypatch.setattr(
        orchestrator,
        "_persist_v2112_post_gate_authority",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            orchestrator.PilotOrchestrationError(
                "provider-free mixed V2.11.2 post-gate failure"
            )
        ),
    )

    first = orchestrator._execute_stage_locked(
        contract_path=CONTRACT_PATH,
        stage_id="long-context-preflight",
        resume=False,
        raw_root=raw_root,
        repo_root=tmp_path,
    )
    receipt_path = raw_root / "long-context-preflight" / "stage_receipt.json"
    first_bytes = receipt_path.read_bytes()

    assert first["status"] == "complete-with-no-go"
    assert first["go"] is False
    assert first["execution_progression_go"] is False
    assert first["go_models"] == []
    assert not (
        raw_root
        / "long-context-preflight"
        / orchestrator.PILOT_V211_POST_GATE_AUTHORITY_FILENAME
    ).exists()

    terminal = orchestrator.PilotRunLedger(
        raw_root / "run_ledger.json",
        contract_hash=contract.canonical_hash,
        tamper_evident=True,
    )
    assert {
        spec.model_id: terminal.status(spec.run_id)
        for spec in contract.expand(stage="long-context-preflight")
    } == {
        "gpt52_main": "failed",
        "gpt56_diagnostic": "complete",
    }
    science_specs = tuple(
        spec for stage_id in SCIENCE_STAGES for spec in contract.expand(stage=stage_id)
    )
    assert len(science_specs) == 131
    assert {terminal.status(spec.run_id) for spec in science_specs} == {
        "integrity-stopped"
    }

    budget = orchestrator.PilotBudgetLedger(
        raw_root / "budget_ledger.json",
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=None,
    ).snapshot()
    assert not any(spec.run_id in budget["runs"] for spec in science_specs)

    resumed = orchestrator._execute_stage_locked(
        contract_path=CONTRACT_PATH,
        stage_id="long-context-preflight",
        resume=True,
        raw_root=raw_root,
        repo_root=tmp_path,
    )

    assert resumed == first
    assert receipt_path.read_bytes() == first_bytes


def test_v2112_post_gate_status_inference_keeps_model_scoped_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "gpt52_main": "eligible",
        "gpt56_diagnostic": "interface-no-go",
    }

    def fake_build(**inputs: Any) -> dict[str, Any]:
        if inputs["model_terminal_statuses"] != expected:
            raise orchestrator.PilotV2112GateError("status mismatch")
        return {
            "go": True,
            "denominator": {"eligible_model_ids": ["gpt52_main"]},
        }

    monkeypatch.setattr(
        orchestrator,
        "build_v2112_post_gate_authority",
        fake_build,
    )

    receipt, statuses = orchestrator._build_v2112_post_gate_with_inferred_statuses({})

    assert receipt["go"] is True
    assert receipt["denominator"]["eligible_model_ids"] == ["gpt52_main"]
    assert statuses == expected


def test_v2112_actual_scripted_preflight_reseals_and_resumes_postgate(
    live_preflight_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real checkpoint/preflight/postgate path without network."""

    contract = live_preflight_case["contract"]
    paid = live_preflight_case["paid"]
    raw_root = live_preflight_case["raw_root"]
    run_ledger = live_preflight_case["run_ledger"]
    budget_ledger = live_preflight_case["budget_ledger"]
    providers: dict[str, _ProfileScriptedProvider] = {}

    def scripted_provider(profile: Any) -> MultiModelLLM:
        provider = providers.setdefault(
            profile.profile_id,
            _ProfileScriptedProvider(profile),
        )
        return MultiModelLLM(provider, num_workers=1)

    monkeypatch.setattr(
        orchestrator,
        "_provider_for_profile",
        scripted_provider,
    )

    for spec in contract.expand(stage="long-context-preflight"):
        catalog_path = (
            raw_root / spec.stage_id / "provider_catalog" / f"{spec.model_id}.json"
        )
        orchestrator._atomic_json(
            catalog_path,
            {
                "schema_version": "provider-free-catalog-fixture-v1",
                "status": "available",
                "model_id": spec.model_id,
            },
        )
        projection = orchestrator.conservative_projection(contract, spec)
        budget_ledger.reserve(projection)
        run_budget = orchestrator._run_budget_from_projection(projection)
        status, terminal, observed_budget, gate = (
            orchestrator._execute_capability_preflight(
                contract,
                spec,
                raw_root=raw_root,
                paid=paid,
                projection=projection,
                budget=run_budget,
                resume=False,
            )
        )

        failed_checks = {
            key: value
            for key, value in gate["preflight_checks"].items()
            if value is not True
        }
        if "action_parse_success_24_of_24" in failed_checks:
            checkpoint_debug = orchestrator._read_json(
                raw_root
                / spec.stage_id
                / "runs"
                / spec.run_id
                / "preflight_checkpoint.json"
            )
            failed_checks["observed_action_parse_modes"] = sorted(
                {
                    decision.get("parse_mode")
                    for step in checkpoint_debug["prefix_steps"]
                    for decision in step["decisions"].values()
                }
            )
        assert gate["go"] is True, failed_checks
        assert status == "complete", gate["reason"]
        assert observed_budget is run_budget
        assert providers[spec.model_id].calls == 32

        run_dir = raw_root / spec.stage_id / "runs" / spec.run_id
        checkpoint_path = run_dir / "preflight_checkpoint.json"
        exactness_path = run_dir / "preflight_checkpoint_exactness.json"
        exactness_bytes = exactness_path.read_bytes()
        exactness = orchestrator._read_json(exactness_path)
        claimed_hash = exactness["receipt_hash"]
        unhashed = dict(exactness)
        unhashed.pop("receipt_hash")
        assert exactness["schema_version"] == V2112_PREFLIGHT_EXACTNESS_SCHEMA_VERSION
        assert claimed_hash == orchestrator.canonical_sha256(unhashed)

        journal_path = orchestrator._provider_call_journal_path(
            run_dir,
            run_id=spec.run_id,
            kind="preflight",
        )
        _, _, bootstrap_reservations = orchestrator._load_v2112_bootstrap_projection(
            contract,
            preflight_spec=spec,
            raw_root=raw_root,
            repo_root=ROOT,
            paid=paid,
        )
        resume_config = orchestrator._preflight_config(
            contract,
            spec,
            paid=paid,
            raw_root=raw_root,
            contract_bootstrap_reservations=bootstrap_reservations,
        )
        resumed_checkpoint = orchestrator.build_v211_long_context_preflight_checkpoint(
            resume_config,
            llm=MultiModelLLM(providers[spec.model_id], num_workers=1),
            budget=orchestrator._run_budget_from_projection(projection),
            env_config_source=orchestrator.DEFAULT_ENV_CONFIG,
            checkpoint_path=checkpoint_path,
            call_journal_path=journal_path,
            resume=True,
        )
        resumed_legacy_exactness = (
            orchestrator.verify_v211_long_context_preflight_checkpoint(
                resumed_checkpoint,
                call_journal_path=journal_path,
            )
        )
        assert providers[spec.model_id].calls == 32
        assert (
            orchestrator._v2112_reseal_long_context_exactness(resumed_legacy_exactness)
            == exactness
        )
        assert exactness_path.read_bytes() == exactness_bytes

        budget_status, budget_failure, _ = orchestrator._finalize_budget_safely(
            budget_ledger,
            projection,
            run_dir=run_dir,
            budget=run_budget,
            status="complete",
            additional_paths=(Path(terminal), journal_path),
        )
        assert budget_status == "complete"
        assert budget_failure is None
        run_ledger.finalize(
            spec.run_id,
            status=status,
            artifact=str(terminal),
        )

    postgate_path, postgate = orchestrator._persist_v2112_post_gate_authority(
        contract,
        raw_root=raw_root,
        paid=paid,
        budget_ledger=budget_ledger,
        run_ledger=run_ledger,
    )
    first_bytes = postgate_path.read_bytes()
    verified = verify_v2112_gate_receipt(
        postgate,
        expected_contract_sha256=contract.canonical_hash,
        expected_git_commit=paid.head_commit,
    )

    assert verified == postgate
    assert postgate["go"] is True
    assert postgate["denominator"]["fresh_preflight_calls"] == 64
    assert postgate["denominator"]["cumulative_full_matrix_calls"] == 6_820
    assert postgate["denominator"]["registered_call_headroom"] == 680
    assert all(
        decision["sample_counts"] == {"action": 24, "semantic": 8}
        for decision in postgate["model_decisions"].values()
    )
    for spec in contract.expand(stage="long-context-preflight"):
        exactness = orchestrator._read_json(
            raw_root
            / spec.stage_id
            / "runs"
            / spec.run_id
            / "preflight_checkpoint_exactness.json"
        )
        fresh_binding = postgate["bindings"]["gate_artifacts"][spec.model_id][
            "fresh_preflight"
        ]
        assert (
            fresh_binding["exactness_schema_version"]
            == V2112_PREFLIGHT_EXACTNESS_SCHEMA_VERSION
        )
        assert fresh_binding["exactness_content_sha256"] == exactness["receipt_hash"]

    for spec in contract.expand(stage="long-context-preflight"):
        projection_payload, projection_path = orchestrator._load_verified_projection(
            contract,
            spec.model_id,
            raw_root=raw_root,
            paid=paid,
        )
        assert projection_path == (
            raw_root / spec.stage_id / "runs" / spec.run_id / "projection_p95.json"
        )
        assert set(projection_payload["projection"]) == {
            f"{contract.provider_profiles[spec.model_id].served_model}::action",
            f"{contract.provider_profiles[spec.model_id].served_model}::semantic",
        }

    resumed_path, resumed = orchestrator._persist_v2112_post_gate_authority(
        contract,
        raw_root=raw_root,
        paid=paid,
        budget_ledger=budget_ledger,
        run_ledger=run_ledger,
    )
    assert resumed_path == postgate_path
    assert resumed == postgate
    assert resumed_path.read_bytes() == first_bytes
    assert sum(provider.calls for provider in providers.values()) == 64
