import json
from pathlib import Path

import pytest

from verified_memory.pilot_contract import (
    PILOT_V1_ACTION_GRID,
    PILOT_V1_NARRATIVE_FIXTURES,
    PilotContract,
    PilotContractError,
    ProviderRequestProfile,
    canonical_contract_sha256,
    canonical_sha256,
    load_pilot_contract,
)
from verified_memory.pilot_continuation import DEFAULT_NARRATIVES


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v1.yaml"


@pytest.fixture(scope="module")
def contract() -> PilotContract:
    return load_pilot_contract(CONTRACT_PATH)


def test_frozen_contract_round_trips_with_canonical_hash(
    contract: PilotContract,
) -> None:
    assert contract.schema_version == "finevo-pilot-contract-v1"
    assert contract.contract_id == "finevo-pilot-v1"
    assert contract.status == "frozen"
    source = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    expected_hash = canonical_contract_sha256(source)
    assert contract.canonical_hash == expected_hash
    assert contract.declared_sha256 == expected_hash

    round_tripped = PilotContract.from_dict(contract.to_dict())
    assert round_tripped.to_dict() == contract.to_dict()
    assert round_tripped.canonical_hash == expected_hash


def test_loader_rejects_hash_tampering_and_duplicate_keys(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    payload["budgets"]["total_usd"] = 26.0
    tampered = tmp_path / "tampered.yaml"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PilotContractError, match="hash mismatch"):
        load_pilot_contract(tampered)

    duplicate = tmp_path / "duplicate.yaml"
    source = CONTRACT_PATH.read_text(encoding="utf-8")
    duplicate.write_text(
        source.replace(
            "{",
            '{"schema_version":"finevo-pilot-contract-v1",',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(PilotContractError, match="duplicate JSON key"):
        load_pilot_contract(duplicate)


def test_seed_registry_is_preregistered_and_immutable(
    contract: PilotContract,
) -> None:
    generation = contract.seeds["generation"]
    expected = (
        1099057501,
        1421875452,
        1769977770,
        959809858,
        617806385,
    )
    assert generation["method"] == "user-preregistered-v1"
    assert generation["generated_before_results"] is True
    assert generation["values"] == expected
    assert contract.seeds["sets"]["calibration"] == (1942013315, 760687867)
    assert contract.seeds["sets"]["main"] == expected
    assert contract.seeds["sets"]["cross-model"] == expected[:3]
    assert contract.seeds["preflight_seed"] not in expected
    with pytest.raises(TypeError):
        contract.seeds["preflight_seed"] = 1


def test_registered_matrix_expands_to_exact_run_counts(
    contract: PilotContract,
) -> None:
    expected = {
        "capability-preflight": 6,
        "q-ref-resolution": 1,
        "stage0-calibration": 14,
        "experiment-a": 20,
        "experiment-b": 25,
        "experiment-c": 15,
        "experiment-d": 55,
        "cross-model-sentinels": 36,
    }
    assert {
        stage_id: len(contract.expand(stage=stage_id))
        for stage_id in contract.stage_ids
    } == expected
    assert len(contract.expand()) == 172
    assert len(contract.expand(stage="experiment-a", arm="full")) == 5
    assert len(contract.expand(stage="stage0-calibration")) == 2 * 7
    assert len(
        contract.expand(stage="experiment-d", arm="narrative-content")
    ) == 20

    opus = contract.expand(
        stage="cross-model-sentinels",
        model="opus48_sentinel",
    )
    assert len(opus) == 12
    assert {item.arm_id for item in opus} == {
        "full",
        "no-memory",
        "matched-a",
        "matched-b",
    }
    assert {item.decoding_seed for item in opus} == {None}
    assert len({item.environment_seed for item in opus}) == 3

    llama = contract.expand(
        stage="cross-model-sentinels",
        model="llama4_maverick_sentinel",
    )
    assert len(llama) == 6
    assert all(item.decoding_seed == item.environment_seed for item in llama)


def test_interventions_budget_and_stop_go_are_frozen(
    contract: PilotContract,
) -> None:
    assert contract.shocks["registered-rate-shock"]["schedule"] == (
        {
            "start": 0,
            "end": 4,
            "interest_rate": 0.03,
            "phase": "pre-shock",
        },
        {
            "start": 5,
            "end": 7,
            "interest_rate": 0.08,
            "phase": "shock",
        },
        {
            "start": 8,
            "end": 11,
            "interest_rate": 0.03,
            "phase": "recovery",
        },
    )
    assert contract.budgets["total_usd"] == 25.0
    assert contract.budgets["max_provider_completions"] == 7500
    assert contract.budgets["completion_scope"] == "hosted-api-only"
    assert contract.budgets["max_storage_bytes"] == 5_000_000_000
    assert contract.budgets["stage_usd_caps"] == {
        "capability_preflight": 2.0,
        "calibration": 3.0,
        "core": 13.0,
        "cross_model": 6.0,
        "manual_reserve": 1.0,
    }
    assert contract.stop_go["experiment_a"] == {
        "complete_pairs_min": 4,
        "same_direction_min": 4,
        "total_registered_pairs": 5,
        "median_relative_effect_min": 0.05,
        "route_manipulation_checks_required": True,
    }
    assert contract.stop_go["experiment_d"]["effect_exceeds_one_action_bin"] is True
    assert any("proposer-by-actor model cross" in item for item in contract.non_claims)


def test_d_action_bins_and_exact_narrative_fixtures_are_fail_closed(
    contract: PilotContract,
) -> None:
    texts = {
        narrative_id: row["text"]
        for narrative_id, row in contract.narratives.items()
    }
    assert texts == PILOT_V1_NARRATIVE_FIXTURES == DEFAULT_NARRATIVES
    assert contract.stop_go["experiment_d"]["action_grid"] == PILOT_V1_ACTION_GRID
    assert (
        contract.stop_go["experiment_d"]["narrative_fixture_hash"]
        == canonical_sha256(DEFAULT_NARRATIVES)
    )

    changed_text = contract.to_dict()
    changed_text["narratives"]["aligned"]["text"] += " drift"
    changed_text["integrity"]["declared_sha256"] = canonical_contract_sha256(
        changed_text
    )
    with pytest.raises(PilotContractError, match="narrative fixture text drifted"):
        PilotContract.from_dict(changed_text)

    changed_grid = contract.to_dict()
    changed_grid["stop_go"]["experiment_d"]["action_grid"][
        "labor_step_hours"
    ] = 7.0
    changed_grid["integrity"]["declared_sha256"] = canonical_contract_sha256(
        changed_grid
    )
    with pytest.raises(PilotContractError, match="action grid drifted"):
        PilotContract.from_dict(changed_grid)


def test_c_zero_api_sensitivity_grid_and_threshold_derivation_are_frozen(
    contract: PilotContract,
) -> None:
    sensitivity = contract.stop_go["experiment_c"]["zero_api_sensitivity"]
    assert sensitivity["alternative_success_weights"] == (0.25, 0.5, 0.75)
    assert sensitivity["outcome_definitions"] == (
        "utility_advantage_positive",
        "absolute_flow_utility",
        "three_period_cumulative_advantage_positive",
    )
    assert sensitivity["absolute_flow_threshold"] == {
        "source_stage": "stage0-calibration",
        "source_profile": "selected-profile-only",
        "source_seeds": "all-two-calibration-seeds",
        "field": "flow_utility",
        "aggregation": "median",
        "derived_after_profile_selection": True,
        "treatment_outcomes_inspected": False,
    }
    assert sensitivity["effectiveness_gate"] is False
    assert sensitivity["descriptive_only"] is True

    changed = contract.to_dict()
    changed["stop_go"]["experiment_c"]["zero_api_sensitivity"][
        "alternative_success_weights"
    ][0] = 0.2
    changed["integrity"]["declared_sha256"] = canonical_contract_sha256(changed)
    with pytest.raises(PilotContractError, match="sensitivity weights"):
        PilotContract.from_dict(changed)


def test_q_ref_resolution_and_local_artifact_identity_are_exact(
    contract: PilotContract,
) -> None:
    q_ref = contract.utility["q_ref_resolution"]
    assert q_ref["seed"] == 2010922376
    assert q_ref["num_agents"] == 4
    assert q_ref["episode_length"] == 12
    assert q_ref["work_fraction_cycle"] == (0.25, 0.5, 0.75, 0.5)
    assert q_ref["consumption_fraction_cycle"] == (0.3, 0.35, 0.3, 0.25)
    assert q_ref["ledger_field"] == "realized_consumption_quantity"
    assert q_ref["expected_rows"] == 48
    assert q_ref["aggregation"] == "median"
    assert contract.utility["selection_rule"] == {
        "method": "guardrail-then-registered-tiebreak-v1",
        "tiebreak_order": (
            "maximize mean interior action coverage",
            "minimize component-balance log distance from one",
            "minimize normalized center distance",
            "declaration order only for an exact remaining tie",
        ),
        "outcome_blind": True,
        "result_artifact": "stage0_selection.json",
        "required_before": (
            "experiment-a",
            "experiment-b",
            "experiment-c",
            "experiment-d",
            "cross-model-sentinels",
        ),
    }

    local = contract.provider_profiles["llama33_local_sentinel"]
    assert dict(local.artifact_identity) == {
        "manifest_sha256": (
            "a6eb4748fd2990ad2952b2335a95a7f952d1a06119a0aa6a2df6cd052a93a3fa"
        ),
        "model_layer_digest": (
            "sha256:4824460d29f2058aaf6e1118a63a7a197a09bed509f0e7d4e2efb1ee273b447d"
        ),
    }


def test_endpoint_price_is_dispatch_basis_and_unknown_price_fails_closed(
    contract: PilotContract,
) -> None:
    gpt56 = contract.provider_profiles["gpt56_upper"]
    assert (
        gpt56.price_snapshot.catalog_input,
        gpt56.price_snapshot.catalog_cached_input,
        gpt56.price_snapshot.catalog_output,
    ) == (5.0, 0.5, 30.0)
    assert (
        gpt56.price_snapshot.endpoint_input,
        gpt56.price_snapshot.endpoint_cached_input,
        gpt56.price_snapshot.endpoint_output,
    ) == (5.0, 0.5, 30.0)
    assert gpt56.price_snapshot.costs_per_1k() == {
        "prompt": 0.005,
        "cached_prompt": 0.0005,
        "completion": 0.03,
    }

    llama = contract.provider_profiles["llama4_maverick_sentinel"]
    price = llama.price_snapshot
    assert (price.catalog_input, price.catalog_output) == (0.2, 0.8)
    assert (price.endpoint_input, price.endpoint_output) == (0.2, 0.8)
    assert price.costs_per_1k() == {
        "prompt": 0.0002,
        "cached_prompt": 0.0002,
        "completion": 0.0008,
    }

    unknown_payload = llama.to_dict()
    unknown_payload["price_snapshot"]["endpoint_input"] = None
    unknown = ProviderRequestProfile.from_dict(unknown_payload)
    with pytest.raises(PilotContractError, match="price is unknown"):
        unknown.validate_dispatch(
            transport="openrouter",
            model=unknown.requested_model,
            seed=1099057501,
            max_attempts=1,
        )


def test_annotated_tag_peel_provenance_binding(contract: PilotContract) -> None:
    resolved = "1" * 40
    manifest = contract.validate_provenance(resolved, "pilot-v1")
    assert manifest == {
        "git_tag": "pilot-v1",
        "resolved_git_commit": resolved,
            "commit_resolution": "annotated_tag_peel",
            "p0_base_commit": "01684c5fd465d43eeb748d11f5937383df9ce602",
            "contract_id": "finevo-pilot-v1",
            "contract_sha256": contract.canonical_hash,
        }
    with pytest.raises(PilotContractError, match="annotated tag"):
        contract.validate_provenance(resolved, "wrong-tag")
    with pytest.raises(PilotContractError, match="40-character commit"):
        contract.validate_provenance("not-a-commit", "pilot-v1")
