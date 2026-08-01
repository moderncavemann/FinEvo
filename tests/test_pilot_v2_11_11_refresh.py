from __future__ import annotations

from collections import Counter
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_providers import StructuredCompletion
from verified_memory import pilot_orchestrator as orchestrator
from verified_memory import pilot_v21111_dispatch_refresh as refresh_module
from verified_memory.budget import UsageRecord
from verified_memory.pilot_budget import PilotBudgetLedger
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_v21111_dispatch_refresh import (
    DispatchRefreshLedger,
    PilotV21111DispatchRefreshError,
    V21111_REFRESH_LEDGER_FILENAME,
    V21111_REFRESH_RECEIPT_FILENAME,
    execute_dispatch_refresh,
    refresh_projections,
    verify_dispatch_refresh_go,
    verify_dispatch_refresh_terminal,
)
from verified_memory.pilot_v21111_fresh_cohort import (
    parent_budget_debit_for_v21111,
)
from verified_memory.pilot_contract import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_11_11.yaml"


def _frozen_contract():
    return replace(load_pilot_contract(CONTRACT_PATH), status="frozen")


def _catalog_evidence(contract) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for profile_id in ("gpt52_main", "gpt56_diagnostic"):
        profile = contract.provider_profiles[profile_id]
        assert profile.price_snapshot.model_reference is not None
        result[profile_id] = {
            "profile_id": profile_id,
            "captured_at": profile.price_snapshot.captured_at,
            "price_source_url": profile.price_snapshot.source,
            "model_reference_url": profile.price_snapshot.model_reference,
            "price_source_sha256": "a" * 64,
            "model_reference_sha256": "b" * 64,
        }
    return result


def _budget_ledger(contract, path: Path) -> PilotBudgetLedger:
    return PilotBudgetLedger(
        path,
        contract_hash=contract.canonical_hash,
        caps=orchestrator._budget_caps(contract),
        tamper_evident=True,
        parent_debit=parent_budget_debit_for_v21111(contract),
    )


def _refresh_ready_budget_ledger(contract, path: Path) -> PilotBudgetLedger:
    """Represent the real prerequisite: a completed provider-free parent row."""

    ledger = _budget_ledger(contract, path)
    (parent_spec,) = contract.expand(stage="parent-import")
    projection = orchestrator._v21111_parent_import_projection(parent_spec)
    ledger.reserve(projection)
    ledger.finalize(
        parent_spec.run_id,
        status="complete",
        cost_usd=0.0,
        completions=0,
        storage_bytes=1,
    )
    return ledger


class _FakeRefreshProvider:
    def __init__(self, profile) -> None:
        self.profile = profile
        self.calls = 0

    def get_structured_completion(
        self,
        messages,
        *,
        temperature,
        max_tokens,
        top_p,
        max_retries,
        seed,
    ) -> StructuredCompletion:
        assert temperature == 0.0
        assert top_p == 1.0
        assert max_retries == 1
        assert seed is None
        assert max_tokens in {4_096, 8_192}
        expected = json.loads(messages[-1]["content"].split("object: ", 1)[1])
        self.calls += 1
        prompt_tokens = 20
        completion_tokens = 5
        price = self.profile.price_snapshot
        assert price.dispatch_input is not None
        assert price.dispatch_output is not None
        cost = (
            prompt_tokens * float(price.dispatch_input)
            + completion_tokens * float(price.dispatch_output)
        ) / 1_000_000.0
        return StructuredCompletion(
            text=json.dumps(expected, sort_keys=True, separators=(",", ":")),
            usage=UsageRecord(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
            ),
            model=self.profile.requested_model,
            provider="openai",
            attempts=1,
            latency_seconds=0.0,
            response_model=self.profile.served_model,
            request_id=f"refresh_{self.profile.profile_id}_{self.calls}",
            response_provider="OpenAI-direct",
            response_route="direct",
            request_profile_id=self.profile.profile_id,
            finish_reason="stop",
            native_finish_reason="stop",
            response_completed=True,
            provider_sdk_name="offline-fixture",
            provider_sdk_version="0.test",
            request_parameters=(
                "model",
                "messages",
                "response_format",
                "reasoning_effort",
                "service_tier",
                "max_completion_tokens",
            ),
            temperature_dispatch="omitted_unsupported",
        )


def _complete_refresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    contract = _frozen_contract()
    raw = tmp_path / "raw"
    raw.mkdir()
    acceptance = {"integrity": {"content_sha256": "c" * 64}}
    monkeypatch.setattr(
        "verified_memory.pilot_v21111_dispatch_refresh."
        "verify_scientific_dispatch_acceptance",
        lambda **_kwargs: acceptance,
    )
    budget = _refresh_ready_budget_ledger(contract, raw / "budget_ledger.json")
    paid = SimpleNamespace(
        git_tag="pilot-v2.11.11-science",
        head_commit="a" * 40,
        worktree_clean=True,
    )
    providers: dict[str, _FakeRefreshProvider] = {}
    factory_calls: list[str] = []

    def provider_factory(profile_id, profile):
        factory_calls.append(profile_id)
        provider = _FakeRefreshProvider(profile)
        providers[profile_id] = provider
        return provider

    receipt = execute_dispatch_refresh(
        contract=contract,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        provider_factory=provider_factory,
        catalog_evidence=_catalog_evidence(contract),
        resume=False,
    )
    return contract, raw, paid, budget, providers, factory_calls, receipt


def _durable_success_for_first_refresh_row(contract):
    plan = refresh_module._refresh_plan(contract)
    plan_row = plan["rows"][0]
    profile = contract.provider_profiles[plan_row["profile_id"]]
    definition = refresh_module._definition_by_id(plan)[plan_row["probe_id"]]
    completion = _FakeRefreshProvider(profile).get_structured_completion(
        [
            {
                "role": "user",
                "content": "object: "
                + json.dumps(definition["expected_json"], sort_keys=True),
            }
        ],
        temperature=0.0,
        max_tokens=int(plan_row["max_completion_tokens"]),
        top_p=1.0,
        max_retries=1,
        seed=None,
    )
    response, passed, failure = refresh_module._completion_record(
        profile=profile,
        definition=definition,
        completion=completion,
    )
    assert passed is True
    assert failure is None
    return (
        plan_row,
        refresh_module._request_record(
            profile=profile,
            row=plan_row,
            definition=definition,
        ),
        response,
        float(response["usage"]["frozen_price_cost_usd"]),
    )


def test_v21111_refresh_projections_are_exactly_twenty_full_cap_rows() -> None:
    contract = _frozen_contract()
    projections = refresh_projections(contract)

    assert len(projections) == 20
    assert len({projection.run_id for projection in projections}) == 20
    assert {projection.completions for projection in projections} == {1}
    assert {projection.stage_bucket for projection in projections} == {
        "dispatch_refresh"
    }
    assert all(
        projection.basis["method"] == "fixed-full-cap-price-times-1.25"
        and projection.basis["cached_input_discount_assumed"] is False
        and projection.basis["scientific_evidence"] is False
        for projection in projections
    )
    assert sum(
        (Decimal(str(projection.cost_usd)) for projection in projections),
        start=Decimal("0"),
    ) == Decimal("3.4019812500")


def test_v21111_refresh_acceptance_failure_precedes_ledger_and_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _frozen_contract()
    raw = tmp_path / "raw"
    calls: list[str] = []

    def reject_acceptance(**_kwargs):
        calls.append("acceptance")
        raise RuntimeError("fixture acceptance no-go")

    monkeypatch.setattr(
        "verified_memory.pilot_v21111_dispatch_refresh."
        "verify_scientific_dispatch_acceptance",
        reject_acceptance,
    )

    def forbidden_provider(*_args, **_kwargs):
        calls.append("provider")
        raise AssertionError("provider factory ran before acceptance")

    with pytest.raises(RuntimeError, match="acceptance no-go"):
        execute_dispatch_refresh(
            contract=contract,
            raw_root=raw,
            paid=object(),
            budget_ledger=object(),  # type: ignore[arg-type]
            provider_factory=forbidden_provider,
            catalog_evidence={},
            resume=False,
        )
    assert calls == ["acceptance"]
    assert not raw.exists()


def test_v21111_refresh_budget_ledger_inherits_parent_before_new_rows(
    tmp_path: Path,
) -> None:
    contract = _frozen_contract()
    ledger = _budget_ledger(contract, tmp_path / "budget_ledger.json")
    snapshot = ledger.snapshot()
    parent = snapshot["parent_debit"]

    assert parent["cost_usd"] == 78.3237413125
    assert parent["hosted_completions"] == 4_192
    assert parent["storage_bytes"] == 280_945_417
    assert snapshot["runs"] == {}


def test_v21111_refresh_requires_completed_parent_and_whole_plan_before_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _frozen_contract()
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(
        "verified_memory.pilot_v21111_dispatch_refresh."
        "verify_scientific_dispatch_acceptance",
        lambda **_kwargs: {"integrity": {"content_sha256": "c" * 64}},
    )
    budget = _budget_ledger(contract, raw / "budget_ledger.json")
    provider_calls = 0

    def forbidden_provider(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("provider constructed before whole-plan budget gate")

    paid = SimpleNamespace(
        git_tag="pilot-v2.11.11-science",
        head_commit="a" * 40,
        worktree_clean=True,
    )
    with pytest.raises(
        PilotV21111DispatchRefreshError,
        match="parent-import budget row drifted before refresh",
    ):
        execute_dispatch_refresh(
            contract=contract,
            raw_root=raw,
            paid=paid,
            budget_ledger=budget,
            provider_factory=forbidden_provider,
            catalog_evidence=_catalog_evidence(contract),
            resume=False,
        )
    assert provider_calls == 0
    assert budget.snapshot()["runs"] == {}


def test_v21111_refresh_fake_provider_completes_exact_twenty_then_resume_is_zero_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        contract,
        raw,
        paid,
        budget,
        providers,
        factory_calls,
        receipt,
    ) = _complete_refresh(
        tmp_path,
        monkeypatch,
    )
    assert receipt["status"] == "go"
    assert receipt["go"] is True
    assert receipt["provider_calls_attempted"] == 20
    assert receipt["row_status_counts"] == {"complete": 20}
    assert receipt["science_capacity"]["complete_matrix_fits"] is True
    assert receipt["scientific_evidence"] is False
    assert factory_calls == ["gpt52_main", "gpt56_diagnostic"]
    assert sum(provider.calls for provider in providers.values()) == 20
    assert Counter(
        row["status"]
        for row in budget.snapshot()["runs"].values()
        if row["stage_bucket"] == "dispatch_refresh"
    ) == {"complete": 20}
    assert (
        verify_dispatch_refresh_go(
            contract=contract,
            raw_root=raw,
            paid=paid,
        )["integrity"]
        == receipt["integrity"]
    )

    resumed_factory_calls = 0

    def forbidden_resume_factory(*_args, **_kwargs):
        nonlocal resumed_factory_calls
        resumed_factory_calls += 1
        raise AssertionError("terminal GO refresh was redispatched")

    resumed = execute_dispatch_refresh(
        contract=contract,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        provider_factory=forbidden_resume_factory,
        catalog_evidence=_catalog_evidence(contract),
        resume=True,
    )
    assert resumed["integrity"] == receipt["integrity"]
    assert resumed_factory_calls == 0


@pytest.mark.parametrize(
    "window",
    (
        "returned-with-reserved-budget",
        "returned-with-terminal-budget",
        "complete-with-terminal-budget",
    ),
)
def test_v21111_refresh_resume_reuses_durable_success_without_redispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window: str,
) -> None:
    contract = _frozen_contract()
    raw = tmp_path / window / "raw"
    raw.mkdir(parents=True)
    monkeypatch.setattr(
        "verified_memory.pilot_v21111_dispatch_refresh."
        "verify_scientific_dispatch_acceptance",
        lambda **_kwargs: {"integrity": {"content_sha256": "c" * 64}},
    )
    budget = _refresh_ready_budget_ledger(contract, raw / "budget_ledger.json")
    authority = DispatchRefreshLedger(
        raw / "dispatch-refresh" / V21111_REFRESH_LEDGER_FILENAME,
        contract=contract,
    )
    plan_row, request, response, actual_cost = _durable_success_for_first_refresh_row(
        contract
    )
    run_id = plan_row["run_id"]
    projection = {row.run_id: row for row in refresh_projections(contract)}[run_id]
    budget.reserve(projection)
    authority.begin(run_id, request)
    authority.returned(run_id, response)
    if window != "returned-with-reserved-budget":
        budget.finalize(
            run_id,
            status="complete",
            cost_usd=actual_cost,
            completions=1,
            storage_bytes=0,
        )
    if window == "complete-with-terminal-budget":
        authority.finalize(run_id, status="complete", failure=None)
    budget_events_before = [
        event
        for event in budget.snapshot()["events"]
        if event.get("payload", {}).get("run_id") == run_id
    ]
    authority_events_before = [
        event
        for event in authority.snapshot()["events"]
        if event.get("payload", {}).get("run_id") == run_id
    ]

    paid = SimpleNamespace(
        git_tag="pilot-v2.11.11-science",
        head_commit="a" * 40,
        worktree_clean=True,
    )
    providers: list[_FakeRefreshProvider] = []

    def provider_factory(_profile_id, profile):
        provider = _FakeRefreshProvider(profile)
        providers.append(provider)
        return provider

    receipt = execute_dispatch_refresh(
        contract=contract,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        provider_factory=provider_factory,
        catalog_evidence=_catalog_evidence(contract),
        resume=True,
    )

    assert receipt["status"] == "go"
    assert receipt["row_status_counts"] == {"complete": 20}
    # The first call already has a durable return. Only the untouched 19 rows
    # may reach the provider adapter during recovery.
    assert sum(provider.calls for provider in providers) == 19
    recovered_authority = DispatchRefreshLedger(
        authority.path,
        contract=contract,
    ).snapshot()["rows"][run_id]
    assert recovered_authority["status"] == "complete"
    assert recovered_authority["response"] == response
    recovered_budget = budget.snapshot()["runs"][run_id]
    assert recovered_budget["status"] == "complete"
    assert recovered_budget["actual"] == {
        "cost_usd": actual_cost,
        "completions": 1,
        "storage_bytes": 0,
    }
    budget_events_after = [
        event
        for event in budget.snapshot()["events"]
        if event.get("payload", {}).get("run_id") == run_id
    ]
    authority_events_after = [
        event
        for event in DispatchRefreshLedger(
            authority.path,
            contract=contract,
        ).snapshot()["events"]
        if event.get("payload", {}).get("run_id") == run_id
    ]
    if window == "returned-with-reserved-budget":
        assert budget_events_after[: len(budget_events_before)] == budget_events_before
        assert len(budget_events_after) == len(budget_events_before) + 1
    else:
        assert budget_events_after == budget_events_before
    if window == "complete-with-terminal-budget":
        assert authority_events_after == authority_events_before
    else:
        assert authority_events_after[: len(authority_events_before)] == (
            authority_events_before
        )
        assert len(authority_events_after) == len(authority_events_before) + 1


@pytest.mark.parametrize(
    "tamper",
    ("scientific-evidence", "budget-prefix", "profile-service-tier"),
)
def test_v21111_refresh_go_rejects_resealed_deep_receipt_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    contract, raw, paid, _, _, _, _ = _complete_refresh(tmp_path, monkeypatch)
    path = raw / "dispatch-refresh" / V21111_REFRESH_RECEIPT_FILENAME
    value = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "scientific-evidence":
        value["scientific_evidence"] = True
    elif tamper == "budget-prefix":
        value["budget_ledger_prefix"]["refresh_reserved_cost_usd"] = 0.0
    else:
        value["profile_bindings"]["gpt52_main"]["profile"]["service_tier"] = "flex"
    unsigned = dict(value)
    unsigned.pop("integrity")
    value["integrity"]["content_sha256"] = canonical_sha256(unsigned)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PilotV21111DispatchRefreshError, match="receipt drifted"):
        verify_dispatch_refresh_go(
            contract=contract,
            raw_root=raw,
            paid=paid,
        )


@pytest.mark.parametrize("window", ("budget-reserved", "dispatch-running"))
def test_v21111_refresh_crash_window_is_charged_once_and_never_redispatched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    window: str,
) -> None:
    contract = _frozen_contract()
    raw = tmp_path / window / "raw"
    raw.mkdir(parents=True)
    acceptance = {"integrity": {"content_sha256": "c" * 64}}
    monkeypatch.setattr(
        "verified_memory.pilot_v21111_dispatch_refresh."
        "verify_scientific_dispatch_acceptance",
        lambda **_kwargs: acceptance,
    )
    budget = _refresh_ready_budget_ledger(contract, raw / "budget_ledger.json")
    projections = refresh_projections(contract)
    first = projections[0]
    authority_path = raw / "dispatch-refresh" / V21111_REFRESH_LEDGER_FILENAME
    if window == "budget-reserved":
        budget.reserve(first)
    else:
        authority = DispatchRefreshLedger(authority_path, contract=contract)
        authority.begin(first.run_id, {"fixture": "pre-provider-running-marker"})

    provider_calls = 0

    def forbidden_provider(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("interrupted refresh row was redispatched")

    paid = SimpleNamespace(
        git_tag="pilot-v2.11.11-science",
        head_commit="a" * 40,
        worktree_clean=True,
    )
    receipt = execute_dispatch_refresh(
        contract=contract,
        raw_root=raw,
        paid=paid,
        budget_ledger=budget,
        provider_factory=forbidden_provider,
        catalog_evidence=_catalog_evidence(contract),
        resume=True,
    )

    assert receipt["status"] == "no-go"
    assert receipt["go"] is False
    assert receipt["provider_calls_attempted"] == (
        1 if window == "dispatch-running" else 0
    )
    assert receipt["row_status_counts"] == {"failed": 1, "skipped": 19}
    assert receipt["scientific_evidence"] is False
    assert provider_calls == 0
    budget_row = budget.snapshot()["runs"][first.run_id]
    assert budget_row["status"] == "integrity-stopped"
    assert budget_row["actual"] == {
        "cost_usd": first.cost_usd,
        "completions": 1,
        "storage_bytes": 0,
    }
    authority = DispatchRefreshLedger(authority_path, contract=contract).snapshot()
    assert authority["rows"][first.run_id]["status"] == "failed"
    assert Counter(row["status"] for row in authority["rows"].values()) == {
        "failed": 1,
        "skipped": 19,
    }
    terminal = verify_dispatch_refresh_terminal(
        contract=contract,
        raw_root=raw,
        paid=paid,
    )
    assert terminal["status"] == "no-go"
    assert terminal["go"] is False
    with pytest.raises(
        PilotV21111DispatchRefreshError,
        match="terminal receipt is no-go",
    ):
        verify_dispatch_refresh_go(
            contract=contract,
            raw_root=raw,
            paid=paid,
        )

    with pytest.raises(PilotV21111DispatchRefreshError, match="receipt drifted"):
        execute_dispatch_refresh(
            contract=contract,
            raw_root=raw,
            paid=paid,
            budget_ledger=budget,
            provider_factory=forbidden_provider,
            catalog_evidence=_catalog_evidence(contract),
            resume=True,
        )
    assert provider_calls == 0


def test_v21111_refresh_ledger_rejects_resealed_nested_plan_tamper(
    tmp_path: Path,
) -> None:
    contract = _frozen_contract()
    path = tmp_path / "authority_ledger.json"
    ledger = DispatchRefreshLedger(path, contract=contract)
    value = ledger.snapshot()
    first_run_id = next(iter(value["rows"]))
    value["rows"][first_run_id]["plan"]["reserved_cost_usd"] += 0.01
    unsigned = dict(value)
    unsigned.pop("ledger_sha256")
    value["ledger_sha256"] = canonical_sha256(unsigned)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        PilotV21111DispatchRefreshError,
        match="refresh authority ledger drifted",
    ):
        DispatchRefreshLedger(path, contract=contract)
