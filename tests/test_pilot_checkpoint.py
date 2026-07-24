from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from llm_providers import MultiModelLLM
from verified_memory.budget import (
    BudgetExceeded,
    BudgetLimits,
    RunBudget,
    UsageRecord,
)
from verified_memory.pilot_checkpoint import (
    CLOSED_LOOP_PREFLIGHT_CHECKPOINT_PURPOSE,
    PILOT_CHECKPOINT_SCHEMA_VERSION,
    PILOT_CHECKPOINT_SCHEMA_VERSION_V2,
    PilotCheckpoint,
    PilotCheckpointError,
    build_closed_loop_preflight_checkpoint,
    build_pilot_checkpoint,
    canonical_hash,
    capture_environment_state,
    restore_pilot_checkpoint,
    verify_closed_loop_preflight_checkpoint,
)
from verified_memory.runner import (
    ShockEvent,
    VerifiedRunConfig,
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
            model="gpt-checkpoint-fixture",
            provider="openai",
            request_id=f"req_checkpoint_{call_number:02d}",
            response_model="gpt-checkpoint-fixture-2026-07-24",
            response_provider="OpenAI-direct",
            response_route="direct",
            request_profile_id="checkpoint-fixture-profile",
            request_provider_pin=("OpenAI-direct",),
            request_artifact_identity=(
                ("served_snapshot", "gpt-checkpoint-fixture-2026-07-24"),
            ),
            request_price_snapshot_source="fixture-price-snapshot",
            request_price_snapshot_captured_at="2026-07-24T00:00:00Z",
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


class _FencedActionProvider(_CountingScriptedProvider):
    def get_structured_completion(self, messages, **kwargs):
        result = super().get_structured_completion(messages, **kwargs)
        if "monthly decision t=0" in self._prompt(messages):
            return replace(result, text=f"```json\n{result.text}\n```")
        return result


class _TruncatedProvider(_CountingScriptedProvider):
    def get_structured_completion(self, messages, **kwargs):
        result = super().get_structured_completion(messages, **kwargs)
        if "monthly decision t=0" in self._prompt(messages):
            return replace(
                result,
                finish_reason="length",
                native_finish_reason="length",
                response_completed=False,
                output_disposition="discarded_incomplete",
            )
        return result


class _ZeroCostHostedProvider(_CountingScriptedProvider):
    def get_structured_completion(self, messages, **kwargs):
        result = super().get_structured_completion(messages, **kwargs)
        return replace(
            result,
            usage=replace(result.usage, cost_usd=0.0),
        )


class _OversizedActionProvider(_CountingScriptedProvider):
    def get_structured_completion(self, messages, **kwargs):
        result = super().get_structured_completion(messages, **kwargs)
        if "monthly decision t=0" in self._prompt(messages):
            return replace(result, text=result.text + (" " * 2048))
        return result


class _ClippedActionProvider(_CountingScriptedProvider):
    def get_structured_completion(self, messages, **kwargs):
        result = super().get_structured_completion(messages, **kwargs)
        if "monthly decision t=0" in self._prompt(messages):
            value = json.loads(result.text)
            value["work"] = 2.0
            return replace(result, text=json.dumps(value, sort_keys=True))
        return result


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
                else "shock"
                if decision_t <= 7
                else "recovery"
            ),
            interest_rate=0.08 if 5 <= decision_t <= 7 else 0.03,
        )
        for decision_t in range(12)
    )


def _build_checkpoint(run_id: str = "pilot-checkpoint-test") -> PilotCheckpoint:
    return build_pilot_checkpoint(
        VerifiedRunConfig(
            run_id=run_id,
            seed=17,
            num_agents=4,
            episode_length=12,
            max_rule_proposals_per_agent=4,
            freeze_new_proposals_after=6,
            shock_schedule=_shock_schedule(),
        ),
        llm=MultiModelLLM(
            ScriptedDiagnosticProvider(), num_workers=4
        ),
        budget=RunBudget(
            BudgetLimits(max_calls=40, max_cost_usd=0.01),
            budget_id=f"{run_id}-budget",
        ),
        env_config_source=ROOT / "config.yaml",
    )


def _build_preflight_checkpoint(
    run_id: str = "closed-loop-preflight-checkpoint-test",
    *,
    journal_path: Path | None = None,
    provider: _CountingScriptedProvider | None = None,
    budget_limits: BudgetLimits | None = None,
) -> tuple[PilotCheckpoint, _CountingScriptedProvider]:
    provider = provider or _CountingScriptedProvider()
    checkpoint = build_closed_loop_preflight_checkpoint(
        VerifiedRunConfig(
            run_id=run_id,
            seed=23,
            num_agents=2,
            episode_length=6,
            max_rule_proposals_per_agent=2,
            freeze_new_proposals_after=6,
            shock_schedule=_shock_schedule()[:6],
            action_max_tokens=2048,
            rule_max_tokens=4096,
            action_max_visible_json_bytes=1024,
            rule_max_visible_json_bytes=4096,
            accepted_action_parse_modes=("exact_json",),
            accepted_semantic_parse_modes=("exact_json",),
            semantic_parse_failure_policy="record-and-skip",
        ),
        llm=MultiModelLLM(provider, num_workers=2),
        budget=RunBudget(
            budget_limits
            or BudgetLimits(max_calls=20, max_cost_usd=0.01),
            budget_id=f"{run_id}-budget",
        ),
        env_config_source=ROOT / "config.yaml",
        call_journal_path=journal_path,
    )
    return checkpoint, provider


def _rehash_v2_rng_binding(payload: dict) -> None:
    payload["rng_binding_hash"] = canonical_hash(
        {
            "numpy_rng_before_env_construction": payload[
                "numpy_rng_before_env_construction"
            ],
            "foundation_reset_seed_state": payload[
                "foundation_reset_seed_state"
            ],
            "python_rng_at_start": payload["python_rng_at_start"],
            "step_seed_states": [
                step["step_seed_state"] for step in payload["prefix_steps"]
            ],
            "python_step_seed_states": [
                step["python_step_seed_state"]
                for step in payload["prefix_steps"]
            ],
            "numpy_rng_after_prefix": payload["numpy_rng_after_prefix"],
            "python_rng_after_prefix": payload["python_rng_after_prefix"],
        }
    )


def _rehash_checkpoint(payload: dict) -> None:
    body = deepcopy(payload)
    body.pop("checkpoint_hash", None)
    payload["checkpoint_hash"] = canonical_hash(body)


def test_checkpoint_round_trip_replays_exact_rng_environment_memory_and_ledger() -> None:
    checkpoint = _build_checkpoint()

    assert (
        checkpoint.payload["schema_version"]
        == PILOT_CHECKPOINT_SCHEMA_VERSION
    )
    assert checkpoint.next_decision_t == 6
    assert [step["decision_t"] for step in checkpoint.payload["prefix_steps"]] == list(
        range(6)
    )
    assert len(checkpoint.payload["ledger_records"]) == 24
    assert checkpoint.payload["proposals_made"] == {
        "0": 2,
        "1": 2,
        "2": 2,
        "3": 2,
    }
    assert all(
        [
            event["timestamp"]
            for event in memory["semantic"]["events"]
            if event["event_type"]
            in {
                "candidate_verified",
                "candidate_rejected",
                "duplicate_semantic_candidate_ignored",
            }
        ]
        == [3, 6]
        for memory in checkpoint.payload["memories"].values()
    )
    assert all(
        len(step["step_seed_state"]["keys"]) == 624
        for step in checkpoint.payload["prefix_steps"]
    )
    assert all(
        len(step["python_step_seed_state"]["internal"]) == 625
        for step in checkpoint.payload["prefix_steps"]
    )
    assert [
        step["shock_event"]["interest_rate"]
        for step in checkpoint.payload["prefix_steps"]
    ] == [0.03, 0.03, 0.03, 0.03, 0.03, 0.08]
    assert all(
        step["shock_event_hash"] == canonical_hash(step["shock_event"])
        for step in checkpoint.payload["prefix_steps"]
    )

    loaded = PilotCheckpoint.from_dict(checkpoint.to_dict())
    restored_a = restore_pilot_checkpoint(loaded)
    restored_b = restore_pilot_checkpoint(loaded)

    assert capture_environment_state(restored_a.env) == checkpoint.payload[
        "previous_state"
    ]
    assert capture_environment_state(restored_b.env) == checkpoint.payload[
        "previous_state"
    ]
    assert restored_a.ledger.records() == checkpoint.payload["ledger_records"]
    assert restored_b.ledger.records() == checkpoint.payload["ledger_records"]
    assert {
        str(agent_id): memory.to_dict()
        for agent_id, memory in restored_a.memories.items()
    } == checkpoint.payload["memories"]
    assert restored_a.prefix_hash == restored_b.prefix_hash
    assert restored_a.last_decisions.keys() == {"0", "1", "2", "3"}
    assert restored_a.last_transitions.keys() == {"0", "1", "2", "3"}


def test_checkpoint_tampering_and_code_drift_fail_closed() -> None:
    checkpoint = _build_checkpoint("pilot-checkpoint-tamper")

    tampered_prefix = checkpoint.to_dict()
    tampered_prefix["prefix_steps"][0]["foundation_actions"]["0"][0] += 1
    with pytest.raises(PilotCheckpointError, match="checkpoint hash mismatch"):
        PilotCheckpoint.from_dict(tampered_prefix)

    tampered_proposals = checkpoint.to_dict()
    tampered_proposals["proposals_made"]["0"] = 1
    proposal_body = deepcopy(tampered_proposals)
    proposal_body.pop("checkpoint_hash")
    tampered_proposals["checkpoint_hash"] = canonical_hash(proposal_body)
    with pytest.raises(PilotCheckpointError, match="two proposal attempts"):
        PilotCheckpoint.from_dict(tampered_proposals)

    tampered_code = checkpoint.to_dict()
    source_hashes = tampered_code["code_binding"]["source_hashes"]
    source_hashes["verified_memory/actions.py"] = "0" * 64
    tampered_code["code_binding"]["binding_hash"] = canonical_hash(
        {"source_hashes": source_hashes}
    )
    body = deepcopy(tampered_code)
    body.pop("checkpoint_hash")
    tampered_code["checkpoint_hash"] = canonical_hash(body)
    rebound = PilotCheckpoint.from_dict(tampered_code)
    with pytest.raises(PilotCheckpointError, match="code binding"):
        restore_pilot_checkpoint(rebound)


def test_v2_closed_loop_preflight_checkpoint_is_exact_without_restore_calls() -> None:
    checkpoint, provider = _build_preflight_checkpoint()
    calls_after_execution = len(provider.prompts)

    assert checkpoint.payload["schema_version"] == (
        PILOT_CHECKPOINT_SCHEMA_VERSION_V2
    )
    assert checkpoint.payload["checkpoint_purpose"] == (
        CLOSED_LOOP_PREFLIGHT_CHECKPOINT_PURPOSE
    )
    assert calls_after_execution == 16
    assert len(checkpoint.payload["ledger_records"]) == 12
    assert checkpoint.payload["proposals_made"] == {"0": 2, "1": 2}
    assert len(checkpoint.payload["memories"]) == 2
    provider_rows = checkpoint.payload["provider_calls"]
    assert len(provider_rows) == 16
    assert sum(row["call_kind"] == "action" for row in provider_rows) == 12
    assert sum(row["call_kind"] == "semantic" for row in provider_rows) == 4
    assert [row["call_index"] for row in provider_rows] == list(range(16))
    assert all(
        row["finish_reason"] == "stop"
        and row["response_completed"] is True
        and row["output_disposition"] == "accepted"
        and row["parse_disposition"]["parse_mode"] == "exact_json"
        and row["parse_disposition"]["accepted"] is True
        and row["served_model"] == "gpt-checkpoint-fixture-2026-07-24"
        and row["served_route"] == "direct"
        and set(row["parameter_dispatch"])
        == {"reasoning", "response_format", "seed", "temperature", "top_p"}
        for row in provider_rows
    )
    assert checkpoint.payload["provider_denominator"] == {
        "planned_calls": 16,
        "observed_calls": 16,
        "successful_terminal_calls": 16,
        "failed_calls": 0,
        "action_calls": 12,
        "semantic_calls": 4,
        "semantic_candidate_parse_failures": 0,
    }
    assert checkpoint.payload["provider_totals"]["hosted"] is True
    assert checkpoint.payload["provider_totals"]["cost_usd"] == pytest.approx(
        0.0016
    )
    assert checkpoint.payload["budget_snapshot_at_checkpoint"][
        "completed_calls"
    ] == 16
    assert len(checkpoint.payload["proposal_outcomes"]) == 4
    assert all(
        row["candidate_parse_status"] == "success"
        and row["candidate_parse_mode"] == "exact_json"
        and row["semantic_events"]
        for row in checkpoint.payload["proposal_outcomes"]
    )

    receipt = verify_closed_loop_preflight_checkpoint(
        checkpoint,
        rng_preview_draws=8,
    )

    assert len(provider.prompts) == calls_after_execution
    assert receipt["provider_calls_during_verification"] == 0
    assert receipt["num_agents"] == 2
    assert receipt["completed_months"] == 6
    assert all(receipt["verified_components"].values())
    assert receipt["component_hashes"]["environment_hash"] == (
        checkpoint.payload["previous_state_hash"]
    )
    assert receipt["component_hashes"]["prefix_hash"] == (
        checkpoint.payload["prefix_hash"]
    )
    assert receipt["rng_binding_hash"] == checkpoint.payload[
        "rng_binding_hash"
    ]
    assert receipt["provider_denominator"]["observed_calls"] == 16
    assert receipt["provider_calls_hash"] == checkpoint.payload[
        "provider_calls_hash"
    ]


def test_v2_checkpoint_binds_complete_terminal_provider_journal(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "provider-calls.json"
    checkpoint, provider = _build_preflight_checkpoint(
        "closed-loop-preflight-journal",
        journal_path=journal_path,
    )

    journal = verify_provider_call_journal(
        journal_path,
        expected_run_id="closed-loop-preflight-journal",
        expected_contract_hash=None,
        require_terminal_dispositions=True,
    )
    assert len(provider.prompts) == 16
    assert len(journal["events"]) == 32
    assert sum(
        event["event_type"] == "completion_received"
        for event in journal["events"]
    ) == 16
    assert sum(
        event["event_type"] == "parse_disposition"
        for event in journal["events"]
    ) == 16
    binding = checkpoint.payload["provider_call_journal_binding"]
    assert binding["enabled"] is True
    assert binding["journal_sha256"] == journal["journal_sha256"]
    assert binding["event_count"] == 32
    assert checkpoint.payload["provider_call_journal_binding_hash"] == (
        canonical_hash(binding)
    )


def test_v2_semantic_parse_failures_are_recorded_and_skipped(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "semantic-parse-failures.json"
    checkpoint, provider = _build_preflight_checkpoint(
        "closed-loop-preflight-semantic-record-skip",
        journal_path=journal_path,
        provider=_MalformedSemanticProvider(),
    )

    assert len(provider.prompts) == 16
    assert checkpoint.payload["provider_denominator"][
        "semantic_candidate_parse_failures"
    ] == 4
    assert all(
        row["candidate_parse_status"] == "failure"
        and row["candidate_parse_mode"] == "parse_failure"
        and row["failure_reason"] == "non_exact_json"
        and row["rule_id"] is None
        and row["rule_status"] is None
        and row["semantic_events"] == []
        for row in checkpoint.payload["proposal_outcomes"]
    )
    semantic_rows = [
        row
        for row in checkpoint.payload["provider_calls"]
        if row["call_kind"] == "semantic"
    ]
    assert len(semantic_rows) == 4
    assert all(
        row["parse_disposition"]["parse_status"] == "failure"
        and row["parse_disposition"]["accepted"] is False
        for row in semantic_rows
    )
    journal = verify_provider_call_journal(
        journal_path,
        expected_run_id="closed-loop-preflight-semantic-record-skip",
        expected_contract_hash=None,
        require_terminal_dispositions=True,
    )
    assert len(journal["events"]) == 32
    receipt = verify_closed_loop_preflight_checkpoint(checkpoint)
    assert receipt["provider_calls_during_verification"] == 0


def test_v2_budget_overage_terminalizes_every_dispatched_completion(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "budget-overage.json"
    provider = _CountingScriptedProvider()
    with pytest.raises(BudgetExceeded) as caught:
        _build_preflight_checkpoint(
            "closed-loop-preflight-budget-overage",
            journal_path=journal_path,
            provider=provider,
            budget_limits=BudgetLimits(
                max_calls=20,
                max_cost_usd=0.00015,
            ),
        )

    assert len(provider.prompts) == 2
    assert len(caught.value.structured_completions) == 2
    journal = verify_provider_call_journal(
        journal_path,
        expected_run_id="closed-loop-preflight-budget-overage",
        expected_contract_hash=None,
        require_terminal_dispositions=True,
    )
    completions = [
        event
        for event in journal["events"]
        if event["event_type"] == "completion_received"
    ]
    dispositions = [
        event["payload"]
        for event in journal["events"]
        if event["event_type"] == "parse_disposition"
    ]
    assert len(completions) == len(dispositions) == 2
    assert all(
        row["parse_status"] == "not_evaluated"
        and row["parse_mode"] == "budget_failure"
        and row["accepted"] is False
        and row["rejection"] == "run_budget_exceeded"
        for row in dispositions
    )


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (_FencedActionProvider(), "not exact JSON"),
        (_TruncatedProvider(), "truncated or non-terminal"),
        (_ZeroCostHostedProvider(), "positive cost"),
        (_OversizedActionProvider(), "visible-JSON byte cap"),
        (_ClippedActionProvider(), "clipped prefix action"),
    ],
)
def test_v2_provider_output_policy_fails_closed_and_closes_journal(
    tmp_path: Path,
    provider: _CountingScriptedProvider,
    message: str,
) -> None:
    journal_path = tmp_path / "failed-provider-calls.json"
    with pytest.raises(PilotCheckpointError, match=message):
        _build_preflight_checkpoint(
            f"closed-loop-preflight-failure-{type(provider).__name__}",
            journal_path=journal_path,
            provider=provider,
        )

    journal = verify_provider_call_journal(
        journal_path,
        expected_run_id=(
            f"closed-loop-preflight-failure-{type(provider).__name__}"
        ),
        expected_contract_hash=None,
        require_terminal_dispositions=True,
    )
    assert len(journal["events"]) == 4
    dispositions = [
        event["payload"]
        for event in journal["events"]
        if event["event_type"] == "parse_disposition"
    ]
    assert len(dispositions) == 2
    assert all(row["accepted"] is False for row in dispositions)


def test_v2_recomputed_tampering_still_fails_closed() -> None:
    checkpoint, _ = _build_preflight_checkpoint(
        "closed-loop-preflight-tamper"
    )

    tampered_rng = checkpoint.to_dict()
    state = tampered_rng["numpy_rng_after_prefix"]
    state["position"] = (int(state["position"]) + 1) % len(state["keys"])
    _rehash_v2_rng_binding(tampered_rng)
    _rehash_checkpoint(tampered_rng)
    rebound_rng = PilotCheckpoint.from_dict(tampered_rng)
    with pytest.raises(
        PilotCheckpointError,
        match="continuation RNG is not exact",
    ):
        restore_pilot_checkpoint(rebound_rng)

    tampered_state = checkpoint.to_dict()
    tampered_state["previous_state"]["timestep"] -= 1
    tampered_state["previous_state_hash"] = canonical_hash(
        tampered_state["previous_state"]
    )
    _rehash_checkpoint(tampered_state)
    rebound_state = PilotCheckpoint.from_dict(tampered_state)
    with pytest.raises(
        PilotCheckpointError,
        match="restored Foundation state is not exact",
    ):
        restore_pilot_checkpoint(rebound_state)

    tampered_proposals = checkpoint.to_dict()
    tampered_proposals["proposals_made"]["0"] = 1
    tampered_proposals["proposal_counters_hash"] = canonical_hash(
        tampered_proposals["proposals_made"]
    )
    _rehash_checkpoint(tampered_proposals)
    with pytest.raises(
        PilotCheckpointError,
        match="two proposal attempts",
    ):
        PilotCheckpoint.from_dict(tampered_proposals)

    tampered_provider = checkpoint.to_dict()
    tampered_provider["provider_calls"][0]["parse_disposition"][
        "parse_mode"
    ] = "fenced_recovery"
    tampered_provider["provider_calls_hash"] = canonical_hash(
        tampered_provider["provider_calls"]
    )
    _rehash_checkpoint(tampered_provider)
    with pytest.raises(
        PilotCheckpointError,
        match="exact-action/record-and-skip",
    ):
        PilotCheckpoint.from_dict(tampered_provider)
