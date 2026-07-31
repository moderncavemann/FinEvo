from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

import verified_memory.pilot_checkpoint as checkpoint_module
from llm_providers import MultiModelLLM
from verified_memory.budget import BudgetLimits, RunBudget, UsageRecord
from verified_memory.pilot_checkpoint import (
    PILOT_CHECKPOINT_SCHEMA_VERSION_V4,
    V211_LONG_CONTEXT_PREFLIGHT_CHECKPOINT_PURPOSE,
    PilotCheckpoint,
    PilotCheckpointError,
    build_v211_long_context_preflight_checkpoint,
    canonical_hash,
    verify_v211_long_context_preflight_checkpoint,
)
from verified_memory.runner import (
    PROMPT_TIER_UPPER_BOUND_METHOD,
    ShockEvent,
    VerifiedRunConfig,
    VerifiedRunError,
    verify_provider_call_journal,
)
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


ROOT = Path(__file__).resolve().parents[1]


class _CountingScriptedProvider(ScriptedDiagnosticProvider):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def get_structured_completion(self, messages, **kwargs):
        self.prompts.append(self._prompt(messages))
        result = super().get_structured_completion(messages, **kwargs)
        call_number = len(self.prompts)
        return replace(
            result,
            usage=UsageRecord(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                cost_usd=0.0001,
            ),
            model="gpt-v211-checkpoint-fixture",
            provider="openai",
            request_id=f"req_v211_checkpoint_{call_number:02d}",
            response_model="gpt-v211-checkpoint-fixture-2026-07-31",
            response_provider="OpenAI-direct",
            response_route="direct",
            request_profile_id="v211-checkpoint-fixture-profile",
            request_provider_pin=("OpenAI-direct",),
            request_artifact_identity=(
                (
                    "served_snapshot",
                    "gpt-v211-checkpoint-fixture-2026-07-31",
                ),
            ),
            request_price_snapshot_source="fixture-price-snapshot",
            request_price_snapshot_captured_at="2026-07-31T00:00:00Z",
            finish_reason="stop",
            native_finish_reason="stop",
            response_completed=True,
            provider_sdk_name="fixture-openai-python",
            provider_sdk_version="0.0.test",
            request_parameters=(
                "max_tokens",
                "messages",
                "model",
                "reasoning_effort",
                "response_format",
                "seed",
                "temperature",
                "top_p",
            ),
            temperature_dispatch="explicit",
            parameter_dispatch=(
                ("reasoning", "explicit_supported"),
                ("response_format", "explicit_supported"),
                ("seed", "explicit_supported"),
                ("temperature", "explicit_supported"),
                ("top_p", "explicit_supported"),
            ),
        )


class _MalformedSemanticProvider(_CountingScriptedProvider):
    def get_structured_completion(self, messages, **kwargs):
        result = super().get_structured_completion(messages, **kwargs)
        if "Propose one semantic decision rule" in self._prompt(messages):
            return replace(result, text="not-json")
        return result


def _shock_schedule() -> tuple[ShockEvent, ...]:
    return tuple(
        ShockEvent(
            decision_t=decision_t,
            phase=(
                "pre-shock"
                if decision_t <= 4
                else "shock" if decision_t <= 7 else "recovery"
            ),
            interest_rate=0.08 if 5 <= decision_t <= 7 else 0.03,
        )
        for decision_t in range(12)
    )


def _config(run_id: str) -> VerifiedRunConfig:
    return VerifiedRunConfig(
        run_id=run_id,
        seed=31,
        num_agents=2,
        episode_length=12,
        max_rule_proposals_per_agent=4,
        freeze_new_proposals_after=12,
        shock_schedule=_shock_schedule(),
        action_max_tokens=4096,
        rule_max_tokens=4096,
        action_max_visible_json_bytes=1024,
        rule_max_visible_json_bytes=4096,
        accepted_action_parse_modes=("exact_json",),
        accepted_semantic_parse_modes=("exact_json",),
        semantic_parse_failure_policy="record-and-skip",
        prompt_tier_ceiling_tokens=200_000,
    )


def _budget(run_id: str) -> RunBudget:
    return RunBudget(
        BudgetLimits(max_calls=40, max_cost_usd=0.02),
        budget_id=f"{run_id}-budget",
    )


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    return (
        tmp_path / "checkpoint.json",
        tmp_path / "provider-calls.json",
    )


def _build(
    tmp_path: Path,
    run_id: str,
    *,
    provider: _CountingScriptedProvider | None = None,
    resume: bool = False,
) -> tuple[PilotCheckpoint, _CountingScriptedProvider, RunBudget]:
    provider = provider or _CountingScriptedProvider()
    checkpoint_path, journal_path = _paths(tmp_path)
    budget = _budget(run_id)
    checkpoint = build_v211_long_context_preflight_checkpoint(
        _config(run_id),
        llm=MultiModelLLM(provider, num_workers=2),
        budget=budget,
        env_config_source=ROOT / "config.yaml",
        checkpoint_path=checkpoint_path,
        call_journal_path=journal_path,
        resume=resume,
    )
    return checkpoint, provider, budget


def _rehash(payload: dict) -> None:
    body = deepcopy(payload)
    body.pop("checkpoint_hash", None)
    payload["checkpoint_hash"] = canonical_hash(body)


def test_v211_roundtrip_exact_denominator_restore_and_zero_dispatch_resume(
    tmp_path: Path,
) -> None:
    run_id = "v211-long-context-roundtrip"
    checkpoint, provider, budget = _build(tmp_path, run_id)
    calls_after_build = len(provider.prompts)

    assert checkpoint.payload["schema_version"] == (PILOT_CHECKPOINT_SCHEMA_VERSION_V4)
    assert checkpoint.payload["checkpoint_purpose"] == (
        V211_LONG_CONTEXT_PREFLIGHT_CHECKPOINT_PURPOSE
    )
    assert checkpoint.next_decision_t == 12
    assert len(checkpoint.payload["prefix_steps"]) == 12
    assert len(checkpoint.payload["ledger_records"]) == 24
    assert checkpoint.payload["proposals_made"] == {"0": 4, "1": 4}
    assert calls_after_build == 32
    assert budget.snapshot().completed_calls == 32
    rows = checkpoint.payload["provider_calls"]
    assert len(rows) == 32
    assert sum(row["call_kind"] == "action" for row in rows) == 24
    assert sum(row["call_kind"] == "semantic" for row in rows) == 8
    assert all(
        row["parse_disposition"]["parse_mode"] == "exact_json"
        and row["parse_disposition"]["accepted"] is True
        and row["prompt_token_upper_bound_method"] == PROMPT_TIER_UPPER_BOUND_METHOD
        and 0
        < row["prompt_token_upper_bound"]
        < row["prompt_tier_ceiling_tokens"]
        == 200_000
        for row in rows
    )
    assert checkpoint.payload["provider_denominator"] == {
        "planned_calls": 32,
        "observed_calls": 32,
        "successful_terminal_calls": 32,
        "failed_calls": 0,
        "action_calls": 24,
        "semantic_calls": 8,
        "semantic_candidate_parse_failures": 0,
    }

    checkpoint_path, journal_path = _paths(tmp_path)
    loaded = PilotCheckpoint.read_json(checkpoint_path)
    assert loaded.to_dict() == checkpoint.to_dict()
    assert loaded.checkpoint_hash == checkpoint.checkpoint_hash
    roundtrip_path = tmp_path / "roundtrip-checkpoint.json"
    loaded.write_json(roundtrip_path)
    assert roundtrip_path.read_bytes() == checkpoint_path.read_bytes()
    journal = verify_provider_call_journal(
        journal_path,
        expected_run_id=run_id,
        expected_contract_hash=None,
        require_terminal_dispositions=True,
    )
    assert len(journal["events"]) == 64

    receipt = verify_v211_long_context_preflight_checkpoint(
        loaded,
        call_journal_path=journal_path,
        rng_preview_draws=8,
    )
    assert len(provider.prompts) == calls_after_build
    assert receipt["provider_calls_during_verification"] == 0
    assert receipt["num_agents"] == 2
    assert receipt["completed_months"] == 12
    assert receipt["provider_journal_event_count"] == 64
    assert all(receipt["verified_components"].values())

    resumed, _, resumed_budget = _build(
        tmp_path,
        run_id,
        provider=provider,
        resume=True,
    )
    assert resumed.checkpoint_hash == checkpoint.checkpoint_hash
    assert len(provider.prompts) == calls_after_build
    assert resumed_budget.snapshot().completed_calls == 0


def test_v211_interruption_before_first_journal_append_never_redispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "v211-interrupt-before-journal"
    provider = _CountingScriptedProvider()
    checkpoint_path, journal_path = _paths(tmp_path)

    def interrupt(*args, **kwargs):
        raise OSError("simulated interruption before journal append")

    monkeypatch.setattr(
        checkpoint_module,
        "_append_provider_call_journal",
        interrupt,
    )
    with pytest.raises(OSError, match="before journal append"):
        build_v211_long_context_preflight_checkpoint(
            _config(run_id),
            llm=MultiModelLLM(provider, num_workers=2),
            budget=_budget(run_id),
            env_config_source=ROOT / "config.yaml",
            checkpoint_path=checkpoint_path,
            call_journal_path=journal_path,
        )
    calls_at_interruption = len(provider.prompts)
    assert calls_at_interruption == 2
    assert not checkpoint_path.exists()
    assert not journal_path.exists()
    intent_path = checkpoint_module._v211_long_context_preflight_run_intent_path(
        checkpoint_path
    )
    assert intent_path.exists()

    monkeypatch.undo()
    with pytest.raises(
        PilotCheckpointError,
        match="intent exists without a sealed checkpoint",
    ):
        build_v211_long_context_preflight_checkpoint(
            _config(run_id),
            llm=MultiModelLLM(provider, num_workers=2),
            budget=_budget(run_id),
            env_config_source=ROOT / "config.yaml",
            checkpoint_path=checkpoint_path,
            call_journal_path=journal_path,
            resume=True,
        )
    assert len(provider.prompts) == calls_at_interruption


def test_v211_resume_recovers_checkpoint_sealed_before_intent_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "v211-interrupt-after-seal"
    provider = _CountingScriptedProvider()
    checkpoint_path, journal_path = _paths(tmp_path)
    original_remove = checkpoint_module._remove_v211_long_context_preflight_run_intent

    def interrupt(path):
        raise OSError("simulated interruption after checkpoint seal")

    monkeypatch.setattr(
        checkpoint_module,
        "_remove_v211_long_context_preflight_run_intent",
        interrupt,
    )
    with pytest.raises(OSError, match="after checkpoint seal"):
        build_v211_long_context_preflight_checkpoint(
            _config(run_id),
            llm=MultiModelLLM(provider, num_workers=2),
            budget=_budget(run_id),
            env_config_source=ROOT / "config.yaml",
            checkpoint_path=checkpoint_path,
            call_journal_path=journal_path,
        )
    assert len(provider.prompts) == 32
    assert checkpoint_path.exists()
    assert journal_path.exists()
    intent_path = checkpoint_module._v211_long_context_preflight_run_intent_path(
        checkpoint_path
    )
    assert intent_path.exists()

    monkeypatch.setattr(
        checkpoint_module,
        "_remove_v211_long_context_preflight_run_intent",
        original_remove,
    )
    resumed = build_v211_long_context_preflight_checkpoint(
        _config(run_id),
        llm=MultiModelLLM(provider, num_workers=2),
        budget=_budget(run_id),
        env_config_source=ROOT / "config.yaml",
        checkpoint_path=checkpoint_path,
        call_journal_path=journal_path,
        resume=True,
    )
    assert resumed.next_decision_t == 12
    assert len(provider.prompts) == 32
    assert not intent_path.exists()


def test_v211_rejects_rehashed_prompt_tier_tampering(
    tmp_path: Path,
) -> None:
    checkpoint, _, _ = _build(tmp_path, "v211-prompt-tier-tamper")
    payload = checkpoint.to_dict()
    payload["provider_calls"][0]["prompt_token_upper_bound"] = 200_000
    payload["provider_calls_hash"] = canonical_hash(payload["provider_calls"])
    _rehash(payload)

    with pytest.raises(
        PilotCheckpointError,
        match="short-tier prompt bound",
    ):
        PilotCheckpoint.from_dict(payload)


def test_v211_rejects_rehashed_route_usage_and_cost_tampering(
    tmp_path: Path,
) -> None:
    checkpoint, _, _ = _build(tmp_path, "v211-provider-tamper")
    cases = (
        (
            lambda row: row.__setitem__("served_route", ""),
            "metadata is incomplete",
        ),
        (
            lambda row: row["usage"].__setitem__("prompt_tokens", 0),
            "successful terminal call",
        ),
        (
            lambda row: row["usage"].__setitem__("cost_usd", 0.0),
            "lacks positive cost",
        ),
    )

    for mutate, message in cases:
        payload = checkpoint.to_dict()
        mutate(payload["provider_calls"][0])
        payload["provider_calls_hash"] = canonical_hash(payload["provider_calls"])
        _rehash(payload)
        with pytest.raises(PilotCheckpointError, match=message):
            PilotCheckpoint.from_dict(payload)


def test_v211_external_journal_tampering_fails_exact_verification(
    tmp_path: Path,
) -> None:
    checkpoint, _, _ = _build(tmp_path, "v211-journal-tamper")
    _, journal_path = _paths(tmp_path)
    journal_payload = json.loads(journal_path.read_text(encoding="utf-8"))
    journal_payload["events"][0]["payload"]["response_route"] = "tampered-route"
    journal_path.write_text(
        json.dumps(journal_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(VerifiedRunError, match="hash mismatch"):
        verify_v211_long_context_preflight_checkpoint(
            checkpoint,
            call_journal_path=journal_path,
        )


def test_v211_rejects_rehashed_proposal_and_prefix_tampering(
    tmp_path: Path,
) -> None:
    checkpoint, _, _ = _build(tmp_path, "v211-state-tamper")

    proposal_payload = checkpoint.to_dict()
    proposal_payload["proposals_made"]["0"] = 3
    proposal_payload["proposal_counters_hash"] = canonical_hash(
        proposal_payload["proposals_made"]
    )
    _rehash(proposal_payload)
    with pytest.raises(
        PilotCheckpointError,
        match="four proposal attempts",
    ):
        PilotCheckpoint.from_dict(proposal_payload)

    prefix_payload = checkpoint.to_dict()
    prefix_payload["prefix_steps"][0]["foundation_actions"]["0"][0] += 1
    prefix_payload["prefix_hash"] = canonical_hash(prefix_payload["prefix_steps"])
    prefix_payload["prefix_actions_hash"] = canonical_hash(
        [step["foundation_actions"] for step in prefix_payload["prefix_steps"]]
    )
    _rehash(prefix_payload)
    rebound = PilotCheckpoint.from_dict(prefix_payload)
    with pytest.raises(
        PilotCheckpointError,
        match="Foundation action mapping mismatch",
    ):
        verify_v211_long_context_preflight_checkpoint(
            rebound,
            call_journal_path=_paths(tmp_path)[1],
        )


def test_v211_semantic_parse_failures_are_fully_ledgered_record_and_skip(
    tmp_path: Path,
) -> None:
    run_id = "v211-malformed-semantic"
    provider = _MalformedSemanticProvider()
    checkpoint_path, journal_path = _paths(tmp_path)

    checkpoint = build_v211_long_context_preflight_checkpoint(
        _config(run_id),
        llm=MultiModelLLM(provider, num_workers=2),
        budget=_budget(run_id),
        env_config_source=ROOT / "config.yaml",
        checkpoint_path=checkpoint_path,
        call_journal_path=journal_path,
    )
    assert len(provider.prompts) == 32
    assert checkpoint_path.exists()
    assert journal_path.exists()
    assert checkpoint.payload["provider_denominator"] == {
        "planned_calls": 32,
        "observed_calls": 32,
        "successful_terminal_calls": 32,
        "failed_calls": 0,
        "action_calls": 24,
        "semantic_calls": 8,
        "semantic_candidate_parse_failures": 8,
    }
    semantic_rows = [
        row
        for row in checkpoint.payload["provider_calls"]
        if row["call_kind"] == "semantic"
    ]
    assert len(semantic_rows) == 8
    assert all(
        row["parse_disposition"]
        == {
            "parse_status": "failure",
            "parse_mode": "parse_failure",
            "accepted": False,
            "rejection": "non_exact_json",
        }
        for row in semantic_rows
    )
    outcomes = checkpoint.payload["proposal_outcomes"]
    assert len(outcomes) == 8
    assert all(
        row["candidate_parse_status"] == "failure"
        and row["failure_reason"] == "non_exact_json"
        and row["semantic_events"] == []
        for row in outcomes
    )
    assert not (
        checkpoint_module._v211_long_context_preflight_run_intent_path(checkpoint_path)
    ).exists()
