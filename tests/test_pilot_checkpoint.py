from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

import verified_memory.pilot_checkpoint as pilot_checkpoint_module
from llm_providers import MultiModelLLM, ProviderErrorDetails
from verified_memory.budget import (
    BudgetExceeded,
    BudgetLimits,
    RunBudget,
    UsageRecord,
)
from verified_memory.pilot_checkpoint import (
    CLOSED_LOOP_PREFLIGHT_CHECKPOINT_PURPOSE,
    EXPERIMENT_D_SHARED_PREFIX_CHECKPOINT_PURPOSE,
    PILOT_CHECKPOINT_SCHEMA_VERSION,
    PILOT_CHECKPOINT_SCHEMA_VERSION_V2,
    PILOT_CHECKPOINT_SCHEMA_VERSION_V3,
    PilotCheckpoint,
    PilotCheckpointError,
    PilotCheckpointProviderFailure,
    build_closed_loop_preflight_checkpoint,
    build_experiment_d_shared_prefix_checkpoint,
    build_pilot_checkpoint,
    canonical_hash,
    capture_environment_state,
    restore_pilot_checkpoint,
    verify_closed_loop_preflight_checkpoint,
)
from verified_memory.runner import (
    ShockEvent,
    VerifiedRunConfig,
    VerifiedRunError,
    verify_provider_call_journal,
)
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


ROOT = Path(__file__).resolve().parents[1]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


class _FailedFourthPrefixActionProvider(_CountingScriptedProvider):
    def get_structured_completion(self, messages, **kwargs):
        result = super().get_structured_completion(messages, **kwargs)
        prompt = self._prompt(messages)
        if "monthly decision t=0" in prompt and "agent 3," in prompt:
            return replace(
                result,
                text="Error",
                usage=UsageRecord(
                    prompt_tokens=result.usage.prompt_tokens,
                    completion_tokens=2048,
                    cost_usd=0.001,
                ),
                error_type="IncompleteCompletionError",
                reasoning_tokens=2048,
                finish_reason="length",
                native_finish_reason="length",
                response_completed=False,
                output_disposition="discarded_incomplete",
                provider_error_details=ProviderErrorDetails(
                    error_type="IncompleteCompletionError",
                    stage="response_completion",
                    sdk_name="fixture-openai-python",
                    sdk_version="0.0.test",
                    http_status=200,
                    code="max_output_tokens",
                    param="max_completion_tokens",
                    request_id="req_checkpoint_failed_a3",
                ),
                request_id="req_checkpoint_failed_a3",
            )
        return result


class _HostileFailureMetadataProvider(_FailedFourthPrefixActionProvider):
    sentinel = "SENTINEL-provider-controlled-text"

    def get_structured_completion(self, messages, **kwargs):
        result = super().get_structured_completion(messages, **kwargs)
        if result.error_type == "IncompleteCompletionError":
            return replace(
                result,
                provider=self.sentinel,
                model=self.sentinel,
                finish_reason=self.sentinel,
                native_finish_reason=self.sentinel,
                output_disposition=self.sentinel,
                provider_error_details=ProviderErrorDetails(
                    error_type="IncompleteCompletionError",
                    stage=self.sentinel,
                    sdk_name=self.sentinel,
                    sdk_version=self.sentinel,
                    http_status=200,
                    code="max_output_tokens",
                    param="max_completion_tokens",
                    request_id="req_checkpoint_hostile_a3",
                ),
                request_id="req_checkpoint_hostile_a3",
            )
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


def _experiment_d_config(run_id: str) -> VerifiedRunConfig:
    return VerifiedRunConfig(
        run_id=run_id,
        seed=29,
        num_agents=4,
        episode_length=12,
        max_rule_proposals_per_agent=4,
        freeze_new_proposals_after=6,
        shock_schedule=_shock_schedule(),
        action_max_tokens=4096,
        rule_max_tokens=4096,
        action_max_visible_json_bytes=1024,
        rule_max_visible_json_bytes=4096,
        accepted_action_parse_modes=("exact_json",),
        accepted_semantic_parse_modes=("exact_json",),
        semantic_parse_failure_policy="record-and-skip",
    )


def _build_experiment_d_checkpoint(
    tmp_path: Path,
    run_id: str,
    *,
    provider: _CountingScriptedProvider | None = None,
    resume: bool = False,
) -> tuple[PilotCheckpoint, _CountingScriptedProvider, RunBudget]:
    provider = provider or _CountingScriptedProvider()
    budget = RunBudget(
        BudgetLimits(max_calls=40, max_cost_usd=0.02),
        budget_id=f"{run_id}-budget",
    )
    checkpoint = build_experiment_d_shared_prefix_checkpoint(
        _experiment_d_config(run_id),
        llm=MultiModelLLM(provider, num_workers=4),
        budget=budget,
        env_config_source=ROOT / "config.yaml",
        checkpoint_path=tmp_path / "checkpoint.json",
        call_journal_path=tmp_path / "prefix-provider-calls.json",
        resume=resume,
    )
    return checkpoint, provider, budget


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


def test_v3_prompt_tier_gate_precedes_first_prefix_dispatch(
    tmp_path: Path,
) -> None:
    provider = _CountingScriptedProvider()
    budget = RunBudget(
        BudgetLimits(max_calls=40, max_cost_usd=0.02),
        budget_id="prompt-tier-prefix-budget",
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    journal_path = tmp_path / "prefix-provider-calls.json"
    config = replace(
        _experiment_d_config("prompt-tier-prefix"),
        prompt_tier_ceiling_tokens=1,
    )

    with pytest.raises(
        VerifiedRunError,
        match="pricing-tier ceiling before provider dispatch",
    ):
        build_experiment_d_shared_prefix_checkpoint(
            config,
            llm=MultiModelLLM(provider, num_workers=4),
            budget=budget,
            env_config_source=ROOT / "config.yaml",
            checkpoint_path=checkpoint_path,
            call_journal_path=journal_path,
        )

    assert provider.prompts == []
    assert budget.snapshot().completed_calls == 0
    assert not checkpoint_path.exists()
    assert not journal_path.exists()
    assert pilot_checkpoint_module._experiment_d_run_intent_path(
        checkpoint_path
    ).exists()


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


def test_v3_experiment_d_prefix_binds_32_calls_and_resume_is_zero_dispatch(
    tmp_path: Path,
) -> None:
    run_id = "experiment-d-shared-prefix-journal"
    checkpoint, provider, budget = _build_experiment_d_checkpoint(
        tmp_path,
        run_id,
    )

    assert (
        checkpoint.payload["schema_version"]
        == PILOT_CHECKPOINT_SCHEMA_VERSION_V3
    )
    assert checkpoint.payload["checkpoint_purpose"] == (
        EXPERIMENT_D_SHARED_PREFIX_CHECKPOINT_PURPOSE
    )
    assert len(provider.prompts) == 32
    assert len(checkpoint.payload["provider_calls"]) == 32
    assert checkpoint.payload["provider_denominator"] == {
        "planned_calls": 32,
        "observed_calls": 32,
        "successful_terminal_calls": 32,
        "failed_calls": 0,
        "action_calls": 24,
        "semantic_calls": 8,
        "semantic_candidate_parse_failures": 0,
    }
    assert budget.snapshot().completed_calls == 32
    assert not (
        tmp_path / "checkpoint.json.run-intent.json"
    ).exists()

    journal_path = tmp_path / "prefix-provider-calls.json"
    journal = verify_provider_call_journal(
        journal_path,
        expected_run_id=run_id,
        expected_contract_hash=None,
        require_terminal_dispositions=True,
    )
    assert len(journal["events"]) == 64
    binding = checkpoint.payload["provider_call_journal_binding"]
    assert binding == {
        "enabled": True,
        "journal_sha256": journal["journal_sha256"],
        "event_count": 64,
        "completion_event_count": 32,
        "parse_disposition_event_count": 32,
        "run_id": run_id,
        "contract_hash": None,
        "path_name": journal_path.name,
    }

    resumed, resumed_provider, resumed_budget = (
        _build_experiment_d_checkpoint(
            tmp_path,
            run_id,
            provider=provider,
            resume=True,
        )
    )
    assert resumed.checkpoint_hash == checkpoint.checkpoint_hash
    assert resumed_provider is provider
    assert len(provider.prompts) == 32
    assert resumed_budget.snapshot().completed_calls == 0
    restored = restore_pilot_checkpoint(resumed)
    assert restored.next_decision_t == 6
    assert len(provider.prompts) == 32


def test_v3_experiment_d_failure_retains_safe_provider_cause_and_no_redispatch(
    tmp_path: Path,
) -> None:
    run_id = "experiment-d-shared-prefix-failure"
    provider = _FailedFourthPrefixActionProvider()
    budget = RunBudget(
        BudgetLimits(max_calls=40, max_cost_usd=0.02),
        budget_id=f"{run_id}-budget",
    )
    kwargs = {
        "config": _experiment_d_config(run_id),
        "llm": MultiModelLLM(provider, num_workers=4),
        "budget": budget,
        "env_config_source": ROOT / "config.yaml",
        "checkpoint_path": tmp_path / "checkpoint.json",
        "call_journal_path": tmp_path / "prefix-provider-calls.json",
    }
    with pytest.raises(
        PilotCheckpointError,
        match="IncompleteCompletionError",
    ) as caught:
        build_experiment_d_shared_prefix_checkpoint(**kwargs)

    assert len(provider.prompts) == 4
    assert not (tmp_path / "checkpoint.json").exists()
    failure = caught.value.failure
    assert failure is not None
    assert failure.to_dict() == {
        "schema_version": "finevo-pilot-checkpoint-provider-failure-v1",
        "error_stage": "shared-prefix-provider",
        "call_kind": "action",
        "decision_t": 0,
        "agent_id": 3,
        "error_type": "IncompleteCompletionError",
        "finish_reason": "length",
        "native_finish_reason": "length",
        "reasoning_tokens": 2048,
        "response_completed": False,
        "output_disposition": "discarded_incomplete",
        "provider_identity_sha256": _sha256_text("openai"),
        "model_identity_sha256": _sha256_text(
            "gpt-checkpoint-fixture"
        ),
        "attempts": 1,
        "provider_error_summary": {
            "schema_version": (
                "finevo-checkpoint-provider-error-summary-v1"
            ),
            "error_type": "IncompleteCompletionError",
            "http_status": 200,
            "code_sha256": _sha256_text("max_output_tokens"),
            "param_sha256": _sha256_text("max_completion_tokens"),
            "request_id_sha256": _sha256_text(
                "req_checkpoint_failed_a3"
            ),
            "stage_sha256": _sha256_text("response_completion"),
            "sdk_name_sha256": _sha256_text(
                "fixture-openai-python"
            ),
            "sdk_version_sha256": _sha256_text("0.0.test"),
            "redaction_policy": "allowlist-and-digest-v1",
        },
    }

    journal = verify_provider_call_journal(
        tmp_path / "prefix-provider-calls.json",
        expected_run_id=run_id,
        expected_contract_hash=None,
        require_terminal_dispositions=True,
    )
    completions = [
        event["payload"]
        for event in journal["events"]
        if event["event_type"] == "completion_received"
    ]
    dispositions = [
        event["payload"]
        for event in journal["events"]
        if event["event_type"] == "parse_disposition"
    ]
    assert len(completions) == len(dispositions) == 4
    failed = next(row for row in completions if row["agent_id"] == 3)
    assert failed["error_type"] == "IncompleteCompletionError"
    assert failed["finish_reason"] == "length"
    assert failed["reasoning_tokens"] == 2048
    assert failed["response_completed"] is False
    disposition_by_agent = {row["agent_id"]: row for row in dispositions}
    assert set(disposition_by_agent) == {0, 1, 2, 3}
    assert all(
        disposition_by_agent[agent_id]["accepted"] is True
        for agent_id in (0, 1, 2)
    )
    assert disposition_by_agent[3]["accepted"] is False
    assert disposition_by_agent[3]["parse_status"] == "unavailable"

    calls_before_resume = len(provider.prompts)
    with pytest.raises(
        PilotCheckpointError,
        match="run intent exists without a sealed checkpoint",
    ):
        build_experiment_d_shared_prefix_checkpoint(
            **kwargs,
            resume=True,
        )
    assert len(provider.prompts) == calls_before_resume


def test_v3_run_intent_closes_pre_journal_interrupt_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "experiment-d-pre-journal-interrupt"
    provider = _CountingScriptedProvider()
    budget = RunBudget(
        BudgetLimits(max_calls=40, max_cost_usd=0.02),
        budget_id=f"{run_id}-budget",
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    journal_path = tmp_path / "prefix-provider-calls.json"
    original_append = pilot_checkpoint_module._append_provider_call_journal

    def interrupt_before_first_append(*args, **kwargs):
        raise RuntimeError("fixture interrupt before first journal append")

    monkeypatch.setattr(
        pilot_checkpoint_module,
        "_append_provider_call_journal",
        interrupt_before_first_append,
    )
    kwargs = {
        "config": _experiment_d_config(run_id),
        "llm": MultiModelLLM(provider, num_workers=4),
        "budget": budget,
        "env_config_source": ROOT / "config.yaml",
        "checkpoint_path": checkpoint_path,
        "call_journal_path": journal_path,
    }
    with pytest.raises(
        RuntimeError,
        match="before first journal append",
    ):
        build_experiment_d_shared_prefix_checkpoint(**kwargs)

    assert len(provider.prompts) == 4
    assert not checkpoint_path.exists()
    assert not journal_path.exists()
    intent_path = tmp_path / "checkpoint.json.run-intent.json"
    assert intent_path.exists()
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    unsigned = dict(intent)
    claimed = unsigned.pop("intent_sha256")
    assert claimed == canonical_hash(unsigned)
    assert intent["run_id_sha256"] == _sha256_text(run_id)
    assert intent["run_config_sha256"] == canonical_hash(
        _experiment_d_config(run_id).to_dict()
    )
    assert intent["checkpoint_path_sha256"] == _sha256_text(
        str(checkpoint_path.resolve())
    )
    assert intent["journal_path_sha256"] == _sha256_text(
        str(journal_path.resolve())
    )
    serialized_intent = json.dumps(intent, sort_keys=True)
    assert run_id not in serialized_intent
    assert "gpt-checkpoint-fixture" not in serialized_intent
    assert str(tmp_path) not in serialized_intent

    monkeypatch.setattr(
        pilot_checkpoint_module,
        "_append_provider_call_journal",
        original_append,
    )
    calls_before_resume = len(provider.prompts)
    with pytest.raises(
        PilotCheckpointError,
        match="run intent exists without a sealed checkpoint",
    ):
        build_experiment_d_shared_prefix_checkpoint(
            **kwargs,
            resume=True,
        )
    assert len(provider.prompts) == calls_before_resume


def test_v3_low_level_builder_cannot_bypass_run_intent(
    tmp_path: Path,
) -> None:
    provider = _CountingScriptedProvider()
    with pytest.raises(ValueError, match="requires a sealed run intent"):
        build_pilot_checkpoint(
            _experiment_d_config("experiment-d-intent-bypass"),
            llm=MultiModelLLM(provider, num_workers=4),
            budget=RunBudget(
                BudgetLimits(max_calls=40, max_cost_usd=0.02),
                budget_id="experiment-d-intent-bypass-budget",
            ),
            env_config_source=ROOT / "config.yaml",
            _schema_version=PILOT_CHECKPOINT_SCHEMA_VERSION_V3,
            _checkpoint_purpose=(
                EXPERIMENT_D_SHARED_PREFIX_CHECKPOINT_PURPOSE
            ),
            _call_journal_path=tmp_path / "provider-calls.json",
        )
    assert provider.prompts == []
    assert not (tmp_path / "provider-calls.json").exists()


def test_v3_resume_recovers_checkpoint_sealed_before_intent_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "experiment-d-sealed-before-intent-cleanup"
    provider = _CountingScriptedProvider()
    config = _experiment_d_config(run_id)
    checkpoint_path = tmp_path / "checkpoint.json"
    journal_path = tmp_path / "prefix-provider-calls.json"
    intent_path = tmp_path / "checkpoint.json.run-intent.json"
    original_remove = (
        pilot_checkpoint_module._remove_experiment_d_run_intent
    )

    def interrupt_cleanup(path):
        raise OSError("fixture interrupt after checkpoint seal")

    monkeypatch.setattr(
        pilot_checkpoint_module,
        "_remove_experiment_d_run_intent",
        interrupt_cleanup,
    )
    with pytest.raises(OSError, match="after checkpoint seal"):
        build_experiment_d_shared_prefix_checkpoint(
            config,
            llm=MultiModelLLM(provider, num_workers=4),
            budget=RunBudget(
                BudgetLimits(max_calls=40, max_cost_usd=0.02),
                budget_id=f"{run_id}-initial-budget",
            ),
            env_config_source=ROOT / "config.yaml",
            checkpoint_path=checkpoint_path,
            call_journal_path=journal_path,
        )

    assert len(provider.prompts) == 32
    assert checkpoint_path.exists()
    assert journal_path.exists()
    assert intent_path.exists()
    monkeypatch.setattr(
        pilot_checkpoint_module,
        "_remove_experiment_d_run_intent",
        original_remove,
    )
    resume_budget = RunBudget(
        BudgetLimits(max_calls=40, max_cost_usd=0.02),
        budget_id=f"{run_id}-resume-budget",
    )
    resumed = build_experiment_d_shared_prefix_checkpoint(
        config,
        llm=MultiModelLLM(provider, num_workers=4),
        budget=resume_budget,
        env_config_source=ROOT / "config.yaml",
        checkpoint_path=checkpoint_path,
        call_journal_path=journal_path,
        resume=True,
    )
    assert resumed.payload["schema_version"] == (
        PILOT_CHECKPOINT_SCHEMA_VERSION_V3
    )
    assert len(provider.prompts) == 32
    assert resume_budget.snapshot().completed_calls == 0
    assert not intent_path.exists()


def test_v3_failure_sanitizes_hostile_metadata_and_validates_schema(
    tmp_path: Path,
) -> None:
    run_id = "experiment-d-hostile-provider-metadata"
    provider = _HostileFailureMetadataProvider()
    with pytest.raises(PilotCheckpointError) as caught:
        build_experiment_d_shared_prefix_checkpoint(
            _experiment_d_config(run_id),
            llm=MultiModelLLM(provider, num_workers=4),
            budget=RunBudget(
                BudgetLimits(max_calls=40, max_cost_usd=0.02),
                budget_id=f"{run_id}-budget",
            ),
            env_config_source=ROOT / "config.yaml",
            checkpoint_path=tmp_path / "checkpoint.json",
            call_journal_path=tmp_path / "prefix-provider-calls.json",
        )

    failure = caught.value.failure
    assert isinstance(failure, PilotCheckpointProviderFailure)
    row = failure.to_dict()
    serialized = json.dumps(row, sort_keys=True)
    assert provider.sentinel not in serialized
    assert row["finish_reason"] == "unknown"
    assert row["native_finish_reason"] == "unknown"
    assert row["output_disposition"] == "unknown"
    assert row["provider_identity_sha256"] == _sha256_text(
        provider.sentinel
    )
    assert row["model_identity_sha256"] == _sha256_text(
        provider.sentinel
    )
    summary = row["provider_error_summary"]
    assert summary["stage_sha256"] == _sha256_text(provider.sentinel)
    assert summary["sdk_name_sha256"] == _sha256_text(provider.sentinel)
    assert summary["sdk_version_sha256"] == _sha256_text(
        provider.sentinel
    )

    for field, value in (
        ("schema_version", "sentinel"),
        ("error_stage", "sentinel"),
        ("call_kind", "sentinel"),
        ("finish_reason", "sentinel"),
        ("native_finish_reason", "sentinel"),
        ("output_disposition", "sentinel"),
    ):
        with pytest.raises(ValueError):
            replace(failure, **{field: value})


def test_frozen_v1_v2_json_remain_read_only_compatible() -> None:
    v1 = _build_checkpoint("frozen-v1-read-only-compatibility")
    v2, provider = _build_preflight_checkpoint(
        "frozen-v2-read-only-compatibility"
    )
    provider_calls_before_restore = len(provider.prompts)

    for checkpoint, schema_version in (
        (v1, PILOT_CHECKPOINT_SCHEMA_VERSION),
        (v2, PILOT_CHECKPOINT_SCHEMA_VERSION_V2),
    ):
        frozen_json = json.dumps(
            checkpoint.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        loaded = PilotCheckpoint.from_dict(json.loads(frozen_json))
        assert loaded.payload["schema_version"] == schema_version
        assert json.dumps(
            loaded.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) == frozen_json
        restored = restore_pilot_checkpoint(loaded)
        assert restored.next_decision_t == 6

    assert len(provider.prompts) == provider_calls_before_restore
    assert "checkpoint_purpose" not in v1.payload
    assert v2.payload["checkpoint_purpose"] == (
        CLOSED_LOOP_PREFLIGHT_CHECKPOINT_PURPOSE
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
