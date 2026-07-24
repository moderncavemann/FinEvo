from __future__ import annotations

from pathlib import Path

import pytest

from llm_providers import (
    PINNED_PROVIDER_SDK_VERSIONS,
    MultiModelLLM,
    StructuredCompletion,
)
from verified_memory.budget import BudgetLimits, RunBudget
from verified_memory.pilot_contract import load_pilot_contract
from verified_memory.pilot_evidence import (
    PilotEvidenceError,
    _validate_standard_run_contract,
)
from verified_memory.runner import VerifiedRunConfig, run_verified_experiment
from verified_memory.runner_artifacts import (
    load_verified_run_artifacts,
    write_verified_run_artifacts,
)
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v1.yaml"
EFFECTIVE_CONFIG_FIELDS = {
    "context_features",
    "prompt_schema_version",
    "effective_semantic_verifier",
    "foundation_env",
    "foundation_env_hash",
}


class _ProfiledScriptedProvider(ScriptedDiagnosticProvider):
    """Use scripted text while preserving one frozen request-profile envelope."""

    def __init__(self, profile) -> None:
        self.profile = profile

    def get_model_name(self) -> str:
        return f"openai/{self.profile.requested_model}"

    def get_structured_completion(self, messages, **kwargs) -> StructuredCompletion:
        scripted = super().get_structured_completion(messages, **kwargs)
        request_parameters = {
            "model",
            "messages",
            "top_p",
            *self.profile.openai_request_options().keys(),
            "max_completion_tokens",
        }
        if scripted.request_seed is not None:
            request_parameters.add("seed")
        return StructuredCompletion(
            text=scripted.text,
            usage=scripted.usage,
            model=self.profile.requested_model,
            provider="openai",
            attempts=1,
            latency_seconds=scripted.latency_seconds,
            request_seed=scripted.request_seed,
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
            provider_sdk_version=PINNED_PROVIDER_SDK_VERSIONS["openai"],
            route_attestation_code=None,
            request_parameters=tuple(sorted(request_parameters)),
            temperature_dispatch="omitted_unsupported",
            output_disposition="accepted",
        )


def test_real_sealed_config_round_trip_matches_evidence_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: execution and evidence rebuild the same enriched config."""

    contract = load_pilot_contract(CONTRACT_PATH)
    spec = contract.expand(
        stage="experiment-a",
        model="gpt52_main",
        arm="full",
    )[0]
    profile = contract.provider_profiles[spec.model_id]
    base_config = VerifiedRunConfig(
        run_id=spec.run_id,
        seed=spec.environment_seed,
        num_agents=2,
        episode_length=4,
        context_mode="full",
        enable_episodic_retrieval=True,
        enable_semantic=True,
        retrieval_k=5,
        rule_budget=3,
        semantic_proposal_after=3,
        semantic_proposal_interval=3,
        max_rule_proposals_per_agent=4,
        send_decoding_seed=True,
    )
    result = run_verified_experiment(
        base_config,
        llm=MultiModelLLM(
            _ProfiledScriptedProvider(profile),
            num_workers=1,
        ),
        budget=RunBudget(
            BudgetLimits(
                max_calls=12,
                max_prompt_tokens=500_000,
                max_completion_tokens=100_000,
                max_total_tokens=600_000,
                max_cost_usd=100.0,
            ),
            budget_id="sealed-config-evidence-regression",
        ),
        env_config_source=ROOT / "config.yaml",
    )
    run_dir = tmp_path / "sealed-run"
    write_verified_run_artifacts(
        run_dir,
        result,
        provenance={
            "purpose": "sealed config evidence integration regression",
            "scientific_evidence": False,
        },
        git_commit="e" * 40,
        git_dirty=False,
    )
    loaded = load_verified_run_artifacts(run_dir)

    assert EFFECTIVE_CONFIG_FIELDS <= set(loaded.config)
    assert loaded.config["effective_semantic_verifier"] is not None
    assert loaded.config == result.config

    import verified_memory.pilot_orchestrator as orchestrator

    monkeypatch.setattr(
        orchestrator,
        "_runner_p95_reservations",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        orchestrator,
        "config_for_spec",
        lambda *args, **kwargs: base_config,
    )
    provenance_git = {
        "git_tag": contract.implementation["required_git_tag"],
        "head_commit": "e" * 40,
        "tag_commit": "e" * 40,
        "tag_object_type": "tag",
        "worktree_clean": True,
        "contract_binding": contract.validate_provenance(
            "e" * 40,
            contract.implementation["required_git_tag"],
        ),
        "release_attestation": None,
    }
    _validate_standard_run_contract(
        contract,
        spec.to_dict(),
        config=loaded.config,
        summary=loaded.summary,
        records=loaded.records,
        provenance_git=provenance_git,
        raw_root=tmp_path,
    )

    tampered = dict(loaded.config)
    tampered["foundation_env_hash"] = "0" * 64
    with pytest.raises(
        PilotEvidenceError,
        match=r"fields=\['foundation_env_hash'\]",
    ):
        _validate_standard_run_contract(
            contract,
            spec.to_dict(),
            config=tampered,
            summary=loaded.summary,
            records=loaded.records,
            provenance_git=provenance_git,
            raw_root=tmp_path,
        )
