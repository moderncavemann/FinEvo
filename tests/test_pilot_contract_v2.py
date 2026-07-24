import json
from pathlib import Path

import pytest

from verified_memory.pilot_contract import (
    PILOT_CONTRACT_SCHEMA_VERSION_V1,
    PILOT_CONTRACT_SCHEMA_VERSION_V2,
    PilotContract,
    PilotContractError,
    canonical_contract_sha256,
    load_pilot_contract,
)


ROOT = Path(__file__).resolve().parents[1]
V1_PATH = ROOT / "experiments" / "pilot_v1.yaml"
V2_PATH = ROOT / "experiments" / "pilot_v2.yaml"


@pytest.fixture(scope="module")
def contract_v2() -> PilotContract:
    return load_pilot_contract(V2_PATH)


def _source() -> dict:
    return json.loads(V2_PATH.read_text(encoding="utf-8"))


def _rehash(payload: dict) -> dict:
    payload["integrity"]["declared_sha256"] = canonical_contract_sha256(payload)
    return payload


def test_v2_round_trip_and_v1_read_compatibility(
    contract_v2: PilotContract,
) -> None:
    assert contract_v2.schema_version == PILOT_CONTRACT_SCHEMA_VERSION_V2
    assert contract_v2.contract_id == "finevo-pilot-v2"
    assert contract_v2.declared_sha256 == canonical_contract_sha256(_source())
    assert PilotContract.from_dict(contract_v2.to_dict()).to_dict() == (
        contract_v2.to_dict()
    )

    v1 = load_pilot_contract(V1_PATH)
    assert v1.schema_version == PILOT_CONTRACT_SCHEMA_VERSION_V1
    assert v1.parameter_dispatch_policy is None
    assert v1.task_output_contracts == {}
    assert v1.model_roles == {}
    assert v1.denominator_policy is None
    assert v1.release_requirements is None


@pytest.mark.parametrize(
    ("profile_id", "price_field"),
    [
        ("gpt52_main", "endpoint_input"),
        ("gpt52_main", "endpoint_output"),
        ("llama4_maverick_diagnostic", "endpoint_input"),
        ("llama4_maverick_diagnostic", "endpoint_output"),
    ],
)
def test_hosted_frozen_dispatch_prices_must_be_positive(
    profile_id: str,
    price_field: str,
) -> None:
    payload = _source()
    payload["provider_profiles"][profile_id]["price_snapshot"][price_field] = 0
    _rehash(payload)

    with pytest.raises(PilotContractError, match="finite and positive"):
        PilotContract.from_dict(payload)


def test_uniform_dispatch_policy_and_profile_dispositions_are_typed(
    contract_v2: PilotContract,
) -> None:
    policy = contract_v2.parameter_dispatch_policy
    assert policy is not None
    assert set(policy.fields) == {
        "temperature",
        "top_p",
        "seed",
        "reasoning",
        "response_format",
    }
    assert set(policy.allowed_modes) == {
        "explicit_supported",
        "documented_unsupported_omitted",
    }
    assert policy.omission_receipt_status == "omitted_unsupported"

    for profile in contract_v2.provider_profiles.values():
        fields = dict(profile.decoding_fields)
        assert set(fields) == set(policy.fields)
        assert {
            field.dispatch_mode for field in fields.values()
        } <= set(policy.allowed_modes)
        if profile.transport in {"openai", "openrouter"}:
            assert all(
                field.catalog_evidence_required for field in fields.values()
            )

    gpt = contract_v2.provider_profiles["gpt52_main"]
    assert (
        dict(gpt.decoding_fields)["temperature"].dispatch_mode
        == "documented_unsupported_omitted"
    )
    gpt.validate_dispatch(
        transport="openai",
        model=gpt.requested_model,
        seed=None,
        max_attempts=1,
    )
    with pytest.raises(PilotContractError, match="omit decoding seed"):
        gpt.validate_dispatch(
            transport="openai",
            model=gpt.requested_model,
            seed=2010922376,
            max_attempts=1,
        )
    opus = contract_v2.provider_profiles["opus48_no_go"]
    assert opus.dispatch_eligible is False
    assert opus.ineligibility_reason == (
        "cross_model_budget_no_go_under_nonshrink_policy"
    )
    with pytest.raises(PilotContractError, match="not dispatch eligible"):
        opus.validate_dispatch(
            transport="openrouter",
            model=opus.requested_model,
            seed=None,
            max_attempts=1,
        )


def test_task_caps_are_common_by_call_role_not_by_model(
    contract_v2: PilotContract,
) -> None:
    expected = {
        "capability-choice": (2048, 512),
        "capability-proposal": (4096, 4096),
        "actor-action": (2048, 1024),
        "semantic-proposal": (4096, 4096),
    }
    assert {
        task_id: (task.max_completion_tokens, task.max_visible_json_bytes)
        for task_id, task in contract_v2.task_output_contracts.items()
    } == expected
    assert all(
        task.science_parse_mode == "exact_json_only"
        and task.report_recovery_modes == ("fenced_json", "substring_json")
        and task.recovered_output_scientific_success is False
        and task.required_finish_reason == "stop"
        and task.visible_token_count_required
        and task.reasoning_token_count_required
        for task in contract_v2.task_output_contracts.values()
    )


def test_model_roles_stage_order_and_dispatch_matrix_are_exact(
    contract_v2: PilotContract,
) -> None:
    assert contract_v2.stage_ids == (
        "capability-gate",
        "closed-loop-preflight",
        "secondary-capability-gate",
        "secondary-closed-loop-preflight",
        "q-ref-resolution",
        "stage0-calibration",
        "experiment-a",
        "experiment-c",
        "experiment-d",
        "experiment-b",
        "controlled-second",
        "cross-model-diagnostics",
    )
    assert contract_v2.stage("experiment-c").prerequisites == (
        "stage0-calibration",
    )
    assert contract_v2.stage("experiment-d").prerequisites == (
        "experiment-a",
        "experiment-c",
    )
    # D executes one independently budgeted prefix per seed, then branches all
    # treatments from that shared hash-bound checkpoint.  It must not claim to
    # reuse an Experiment-A checkpoint that the actor runner never emits.
    assert contract_v2.stage("experiment-d").reuse == ()
    assert contract_v2.stage("experiment-b").prerequisites == ("experiment-d",)

    assert contract_v2.model_roles["gpt52_main"].role == "primary"
    assert (
        contract_v2.model_roles["llama33_local_controlled"].role
        == "controlled_second"
    )
    assert (
        contract_v2.model_roles["opus48_no_go"].role == "capability_no_go"
    )
    assert all(
        item.model_id != "opus48_no_go"
        for item in contract_v2.expand(include_disabled=True)
    )
    assert {
        stage_id: len(contract_v2.expand(stage=stage_id))
        for stage_id in contract_v2.stage_ids
    } == {
        "capability-gate": 2,
        "closed-loop-preflight": 2,
        "secondary-capability-gate": 3,
        "secondary-closed-loop-preflight": 3,
        "q-ref-resolution": 1,
        "stage0-calibration": 14,
        "experiment-a": 20,
        "experiment-c": 25,
        "experiment-d": 55,
        "experiment-b": 25,
        "controlled-second": 6,
        "cross-model-diagnostics": 18,
    }
    preflight = contract_v2.expand(stage="closed-loop-preflight")
    assert {item.execution_mode for item in preflight} == {
        "closed_loop_preflight"
    }
    assert {
        item.decoding_seed
        for item in preflight
        if item.model_id == "gpt52_main"
    } == {None}
    assert all(
        item.decoding_seed == item.environment_seed
        for item in preflight
        if item.model_id
        == "llama33_local_controlled"
    )
    secondary = contract_v2.expand(stage="secondary-closed-loop-preflight")
    assert {
        item.decoding_seed
        for item in secondary
        if item.model_id == "gpt56_diagnostic"
    } == {None}
    assert all(
        item.decoding_seed == item.environment_seed
        for item in secondary
        if item.model_id
        in {"gemini35_flash_diagnostic", "llama4_maverick_diagnostic"}
    )


def test_controlled_local_profile_freezes_real_runtime_identity(
    contract_v2: PilotContract,
) -> None:
    local = contract_v2.provider_profiles["llama33_local_controlled"]
    assert local.json_mode == "json_object"
    assert dict(local.artifact_identity) == {
        "adapter": "ollama-python",
        "base_url": "http://127.0.0.1:11434",
        "manifest_sha256": (
            "a6eb4748fd2990ad2952b2335a95a7f952d1a06119a0aa6a2df6cd052a93a3fa"
        ),
        "model_layer_digest": (
            "sha256:4824460d29f2058aaf6e1118a63a7a197a09bed509f0e7d4e2efb1ee273b447d"
        ),
        "model_layer_size_bytes": "42520398528",
        "ollama_version": "0.15.4",
    }
    assert (
        dict(local.decoding_fields)["response_format"].requested_value
        == {"format": "json"}
    )


def test_v2_scientific_freeze_and_release_placeholders(
    contract_v2: PilotContract,
) -> None:
    assert contract_v2.seeds["failed_seed_replacement"] == "forbidden"
    assert contract_v2.shocks["registered-rate-shock"]["hook_semantics"] == {
        "prompt_effective_before_decision": True,
        "environment_effective_before_step": True,
        "write_independent_event_stream": True,
        "future_values_hidden": True,
    }
    assert contract_v2.budgets["total_usd"] == 25.0
    assert contract_v2.budgets["stage_usd_caps"] == {
        "capability": 2.0,
        "calibration": 3.0,
        "core": 13.0,
        "cross_model": 6.0,
        "manual_reserve": 1.0,
    }
    assert contract_v2.stop_go["calibration"]["no_candidate_action"] == "stop"

    denominator = contract_v2.denominator_policy
    assert denominator is not None
    assert denominator.registered_cells_are_itt
    assert denominator.failed_seed_replacement == "forbidden"
    assert denominator.seed_inference_unit == "seed"

    release = contract_v2.release_requirements
    assert release is not None
    assert release.remote == "origin"
    assert release.branch == "main"
    assert release.tag == "pilot-v2-science"
    assert release.workflow_file == ".github/workflows/verified-memory-ci.yml"
    assert release.workflow_name == "Verified memory CI"
    assert release.required_job_names == (
        "Python 3.12.7 / ubuntu-24.04",
        "Python 3.12.7 / macos-14",
    )
    assert release.expected_ci == {
        "test_count": 622,
        "test_collection_sha256": (
            "6d6540b2743b697081d606155c8612be4e4530f72c22b8858bd32a9ff30d5631"
        ),
        "compiled_source_count": 134,
        "compiled_source_inventory_sha256": (
            "3b70e5f1064ed4f7c09917201eca474392def5514c4ba9ff882872926e29256c"
        ),
        "sealed_manifest_inventory_sha256": (
            "b5c5a817d09d10752c1f5f00ba556b417d16e06c64b5fcbb15671e49a1d81952"
        ),
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["task_output_contracts"]["actor-action"].update(
                {"max_completion_tokens": 1024}
            ),
            "output limits differ",
        ),
        (
            lambda value: value["provider_profiles"]["llama33_local_controlled"].update(
                {"json_mode": "prompt_only"}
            ),
            "controlled local Llama",
        ),
        (
            lambda value: value["release_requirements"].update(
                {"workflow_name": "wrong"}
            ),
            "workflow name drifted",
        ),
        (
            lambda value: next(
                stage
                for stage in value["stages"]
                if stage["stage_id"] == "experiment-c"
            ).update({"prerequisites": ["experiment-b"]}),
            "Experiment C cannot depend",
        ),
    ],
)
def test_semantic_tampering_fails_even_when_attacker_rehashes(
    mutation,
    message: str,
) -> None:
    payload = _source()
    mutation(payload)
    _rehash(payload)
    with pytest.raises(PilotContractError, match=message):
        PilotContract.from_dict(payload)


def test_hash_tampering_fails_without_rehash() -> None:
    payload = _source()
    payload["non_claims"].append("unregistered drift")
    with pytest.raises(PilotContractError, match="hash mismatch"):
        PilotContract.from_dict(payload)


def test_opus_cannot_be_inserted_into_any_dispatch_stage() -> None:
    payload = _source()
    payload["stages"][0]["cells"][0]["models"].append("opus48_no_go")
    _rehash(payload)
    with pytest.raises(
        PilotContractError, match="capability tiers|dispatch-ineligible"
    ):
        PilotContract.from_dict(payload)
