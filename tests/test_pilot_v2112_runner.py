from __future__ import annotations

from dataclasses import replace

import pytest

from verified_memory.budget import UsageRecord
from verified_memory.runner import (
    LONG_CONTEXT_PREFLIGHT_AUTHORITY_BY_ID,
    ContractEnvelopeBootstrapReservation,
    PreflightP95Reservation,
    ShockEvent,
    V2111_CONTRACT_ENVELOPE_AUTHORITY_ID,
    V2111_CONTRACT_ID,
    V2111_RELEASE_TAG,
    V2111_SOURCE_CONTRACT_ID,
    V2111_SOURCE_RELEASE_TAG,
    V2112_CONTRACT_ENVELOPE_AUTHORITY_ID,
    V2112_CONTRACT_ID,
    V2112_PREFLIGHT_COMPLETION_ENVELOPE_TOKENS,
    V2112_PREFLIGHT_ENVELOPE_COST_USD,
    V2112_PREFLIGHT_PROMPT_ENVELOPE_TOKENS,
    V2112_PREFLIGHT_SEED,
    V2112_RELEASE_TAG,
    V2112_SOURCE_CONTRACT_ID,
    V2112_SOURCE_RELEASE_TAG,
    VerifiedRunConfig,
    bootstrap_config_binding_sha256,
    preflight_p95_reservation_for_call,
)


MODEL_ID = "gpt52_main"
RUNTIME_MODEL = "openai/gpt-5.2-2025-12-11"
TARGET_CONTRACT_HASH = "1" * 64
SOURCE_CONTRACT_HASH = "2" * 64


def _provisional_config(authority_id: str) -> VerifiedRunConfig:
    authority = LONG_CONTEXT_PREFLIGHT_AUTHORITY_BY_ID[authority_id]
    return VerifiedRunConfig(
        run_id=authority.target_run_id(MODEL_ID),
        seed=authority.preflight_seed,
        num_agents=2,
        episode_length=12,
        context_mode="full",
        enable_episodic_retrieval=True,
        enable_semantic=True,
        retrieval_k=5,
        rule_budget=3,
        semantic_proposal_after=3,
        semantic_proposal_interval=3,
        max_rule_proposals_per_agent=4,
        freeze_new_proposals_after=12,
        send_decoding_seed=False,
        temperature=0.0,
        top_p=1.0,
        action_max_tokens=authority.completion_envelope_tokens,
        rule_max_tokens=authority.completion_envelope_tokens,
        action_max_visible_json_bytes=1_024,
        rule_max_visible_json_bytes=4_096,
        accepted_action_parse_modes=("exact_json",),
        accepted_semantic_parse_modes=("exact_json",),
        max_retries=1,
        fail_on_clipped_action=True,
        semantic_parse_failure_policy="record-and-skip",
        shock_schedule=tuple(
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
        ),
        scientific_scope="preregistered_mechanism_micro_pilot",
        pilot_contract_hash=TARGET_CONTRACT_HASH,
        pilot_tag=authority.target_release_tag,
        allow_scientific_scope=True,
        prompt_tier_ceiling_tokens=authority.prompt_envelope_tokens,
    )


def _reservation_kwargs(
    authority_id: str,
    *,
    call_kind: str,
    authorized_config_sha256: str,
) -> dict:
    authority = LONG_CONTEXT_PREFLIGHT_AUTHORITY_BY_ID[authority_id]
    sample_count = authority.capability_sample_counts[call_kind]
    return {
        "capability_projection": PreflightP95Reservation(
            model=RUNTIME_MODEL,
            call_kind=call_kind,
            sample_count=sample_count,
            raw_prompt_tokens=80.0,
            raw_completion_tokens=80.0,
            raw_cost_usd=0.008,
            reserved_usage=UsageRecord(
                prompt_tokens=100,
                completion_tokens=100,
                cost_usd=0.01,
            ),
        ),
        "envelope_usage": UsageRecord(
            prompt_tokens=authority.prompt_envelope_tokens,
            completion_tokens=authority.completion_envelope_tokens,
            cost_usd=authority.envelope_cost_usd[RUNTIME_MODEL],
        ),
        "authority_id": authority.authority_id,
        "target_contract_id": authority.target_contract_id,
        "pilot_contract_hash": TARGET_CONTRACT_HASH,
        "pilot_tag": authority.target_release_tag,
        "pilot_commit": "3" * 40,
        "model_id": MODEL_ID,
        "authorized_run_id": authority.target_run_id(MODEL_ID),
        "authorized_seed": authority.preflight_seed,
        "authorized_config_sha256": authorized_config_sha256,
        "target_run_spec_sha256": "4" * 64,
        "source_contract_id": authority.source_contract_id,
        "source_contract_hash": SOURCE_CONTRACT_HASH,
        "source_tag": authority.source_release_tag,
        "source_commit": "5" * 40,
        "source_run_id": authority.source_run_id(MODEL_ID),
        "source_run_spec_sha256": "6" * 64,
        "source_capability_file_sha256": "7" * 64,
        "source_capability_payload_sha256": "8" * 64,
        "source_group_sha256": "9" * 64,
        "capability_projection_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "provider_profile_sha256": "c" * 64,
        "price_snapshot_sha256": "d" * 64,
        "source_projection_sha256": "e" * 64,
    }


def _reservations(
    authority_id: str,
) -> tuple[
    VerifiedRunConfig,
    tuple[ContractEnvelopeBootstrapReservation, ...],
]:
    provisional = _provisional_config(authority_id)
    config_sha256 = bootstrap_config_binding_sha256(
        provisional,
        measurement_role="closed_loop_preflight",
    )
    reservations = tuple(
        ContractEnvelopeBootstrapReservation(
            **_reservation_kwargs(
                authority_id,
                call_kind=call_kind,
                authorized_config_sha256=config_sha256,
            )
        )
        for call_kind in ("action", "semantic")
    )
    return provisional, reservations


def _final_config(authority_id: str) -> VerifiedRunConfig:
    provisional, reservations = _reservations(authority_id)
    return replace(
        provisional,
        contract_bootstrap_reservations=reservations,
        preflight_measurement_role="closed_loop_preflight",
    )


def test_v2112_authority_maps_direct_v2111_source_to_v2112_target() -> None:
    authority = LONG_CONTEXT_PREFLIGHT_AUTHORITY_BY_ID[
        V2112_CONTRACT_ENVELOPE_AUTHORITY_ID
    ]

    assert authority.target_contract_id == V2112_CONTRACT_ID
    assert authority.target_release_tag == V2112_RELEASE_TAG
    assert authority.source_contract_id == V2112_SOURCE_CONTRACT_ID
    assert authority.source_release_tag == V2112_SOURCE_RELEASE_TAG
    assert V2112_SOURCE_CONTRACT_ID == V2111_CONTRACT_ID
    assert V2112_SOURCE_RELEASE_TAG == V2111_RELEASE_TAG
    assert authority.source_contract_id != V2111_SOURCE_CONTRACT_ID
    assert authority.source_release_tag != V2111_SOURCE_RELEASE_TAG
    assert dict(authority.capability_sample_counts) == {
        "action": 24,
        "semantic": 6,
    }
    assert dict(authority.runtime_model_by_model_id) == {
        "gpt52_main": "openai/gpt-5.2-2025-12-11",
        "gpt56_diagnostic": "openai/gpt-5.6-sol",
    }
    assert dict(authority.envelope_cost_usd) == {
        "openai/gpt-5.2-2025-12-11": 0.407344,
        "openai/gpt-5.6-sol": 1.12288,
    }
    assert authority.target_run_id(MODEL_ID) == (
        "finevo-pilot-v2.11.2--long-context-preflight--gpt52_main--"
        "closed-loop-preflight--none--stage0-selected--s2010922376--"
        "actor-preflight"
    )
    assert authority.source_run_id(MODEL_ID) == (
        "finevo-pilot-v2.11.1--capability-gate--gpt52_main--"
        "capability-probe--none--provider-preflight-default--s2010922376"
    )


def test_v2112_runner_accepts_exact_authority_and_serialized_roundtrip() -> None:
    final = _final_config(V2112_CONTRACT_ENVELOPE_AUTHORITY_ID)

    assert final.seed == V2112_PREFLIGHT_SEED
    assert final.pilot_tag == V2112_RELEASE_TAG
    for call_kind in ("action", "semantic"):
        usage = preflight_p95_reservation_for_call(
            final,
            provider_model_name=RUNTIME_MODEL,
            call_kind=call_kind,
        )
        assert usage.prompt_tokens == V2112_PREFLIGHT_PROMPT_ENVELOPE_TOKENS
        assert usage.completion_tokens == V2112_PREFLIGHT_COMPLETION_ENVELOPE_TOKENS
        assert usage.cost_usd == pytest.approx(
            V2112_PREFLIGHT_ENVELOPE_COST_USD[RUNTIME_MODEL]
        )

    serialized = final.to_dict()["contract_bootstrap_reservations"]
    provisional = _provisional_config(V2112_CONTRACT_ENVELOPE_AUTHORITY_ID)
    roundtrip = replace(
        provisional,
        contract_bootstrap_reservations=serialized,
        preflight_measurement_role="closed_loop_preflight",
    )
    assert roundtrip.to_dict()["contract_bootstrap_reservations"] == serialized


@pytest.mark.parametrize(
    ("field_name", "wrong_value", "message"),
    [
        (
            "authority_id",
            V2111_CONTRACT_ENVELOPE_AUTHORITY_ID,
            "contract/tag lineage drifted",
        ),
        (
            "target_contract_id",
            V2111_CONTRACT_ID,
            "contract/tag lineage drifted",
        ),
        (
            "pilot_tag",
            V2111_RELEASE_TAG,
            "contract/tag lineage drifted",
        ),
        (
            "source_contract_id",
            V2111_SOURCE_CONTRACT_ID,
            "contract/tag lineage drifted",
        ),
        (
            "source_tag",
            V2111_SOURCE_RELEASE_TAG,
            "contract/tag lineage drifted",
        ),
        (
            "authorized_run_id",
            "finevo-pilot-v2.11.2--long-context-preflight--lookalike--actor-preflight",
            "run/seed scope drifted",
        ),
        (
            "source_run_id",
            "finevo-pilot-v2.11--capability-gate--gpt52_main--"
            "capability-probe--none--provider-preflight-default--s2010922376",
            "run/seed scope drifted",
        ),
        (
            "authorized_seed",
            V2112_PREFLIGHT_SEED + 1,
            "run/seed scope drifted",
        ),
    ],
)
def test_v2112_reservation_rejects_cross_release_lineage_and_scope(
    field_name: str,
    wrong_value: object,
    message: str,
) -> None:
    provisional = _provisional_config(V2112_CONTRACT_ENVELOPE_AUTHORITY_ID)
    config_sha256 = bootstrap_config_binding_sha256(
        provisional,
        measurement_role="closed_loop_preflight",
    )
    kwargs = _reservation_kwargs(
        V2112_CONTRACT_ENVELOPE_AUTHORITY_ID,
        call_kind="action",
        authorized_config_sha256=config_sha256,
    )
    kwargs[field_name] = wrong_value

    with pytest.raises(ValueError, match=message):
        ContractEnvelopeBootstrapReservation(**kwargs)


@pytest.mark.parametrize("mutation", ["prompt", "completion", "price"])
def test_v2112_reservation_rejects_envelope_or_price_drift(
    mutation: str,
) -> None:
    provisional = _provisional_config(V2112_CONTRACT_ENVELOPE_AUTHORITY_ID)
    config_sha256 = bootstrap_config_binding_sha256(
        provisional,
        measurement_role="closed_loop_preflight",
    )
    kwargs = _reservation_kwargs(
        V2112_CONTRACT_ENVELOPE_AUTHORITY_ID,
        call_kind="action",
        authorized_config_sha256=config_sha256,
    )
    usage = kwargs["envelope_usage"]
    assert isinstance(usage, UsageRecord)
    if mutation == "prompt":
        usage = replace(usage, prompt_tokens=usage.prompt_tokens - 1)
        message = "exact 200000-token prompt"
    elif mutation == "completion":
        usage = replace(
            usage,
            completion_tokens=usage.completion_tokens - 1,
        )
        message = "exact 200000-token prompt"
    else:
        usage = replace(usage, cost_usd=usage.cost_usd + 0.000001)
        message = "exact frozen provider-profile price"
    kwargs["envelope_usage"] = usage

    with pytest.raises(ValueError, match=message):
        ContractEnvelopeBootstrapReservation(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    [
        ("pilot_tag", V2111_RELEASE_TAG),
        (
            "run_id",
            "finevo-pilot-v2.11.2--long-context-preflight--gpt52_main--"
            "closed-loop-preflight--none--stage0-selected--s2010922376--"
            "actor-preflight--lookalike",
        ),
        ("seed", V2112_PREFLIGHT_SEED + 1),
    ],
)
def test_v2112_runner_config_rejects_tag_run_or_seed_drift(
    field_name: str,
    wrong_value: object,
) -> None:
    final = _final_config(V2112_CONTRACT_ENVELOPE_AUTHORITY_ID)

    with pytest.raises(ValueError, match="exact V2.11.2"):
        replace(final, **{field_name: wrong_value})


def test_v2111_bootstrap_cannot_authorize_v2112_runner_config() -> None:
    _, v2111_reservations = _reservations(V2111_CONTRACT_ENVELOPE_AUTHORITY_ID)
    v2112_provisional = _provisional_config(V2112_CONTRACT_ENVELOPE_AUTHORITY_ID)

    with pytest.raises(ValueError, match="exact V2.11.1"):
        replace(
            v2112_provisional,
            contract_bootstrap_reservations=v2111_reservations,
            preflight_measurement_role="closed_loop_preflight",
        )


def test_runner_rejects_mixed_v2111_and_v2112_authorities() -> None:
    _, v2111_reservations = _reservations(V2111_CONTRACT_ENVELOPE_AUTHORITY_ID)
    v2112_provisional, v2112_reservations = _reservations(
        V2112_CONTRACT_ENVELOPE_AUTHORITY_ID
    )
    mixed = (v2111_reservations[0], v2112_reservations[1])

    with pytest.raises(ValueError, match="cannot be mixed across releases"):
        replace(
            v2112_provisional,
            contract_bootstrap_reservations=mixed,
            preflight_measurement_role="closed_loop_preflight",
        )
