from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from verified_memory.pilot_contract import (
    PILOT_CONTRACT_ID_V2,
    PILOT_CONTRACT_ID_V2_1,
    PILOT_CONTRACT_ID_V2_2,
    PILOT_CONTRACT_ID_V2_3,
    PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_3,
    PILOT_CONTRACT_SCHEMA_VERSION_V2,
    PILOT_CONTRACT_TAG_V2_3,
    PILOT_CONTRACT_V2_CANONICAL_SHA256,
    PILOT_CONTRACT_V2_1_CANONICAL_SHA256,
    PILOT_CONTRACT_V2_2_CANONICAL_SHA256,
    PILOT_CONTRACT_V2_3_CANONICAL_SHA256,
    PILOT_CONTRACT_V2_SCIENCE_DESIGN_SHA256,
    PilotContract,
    PilotContractError,
    canonical_contract_sha256,
    load_pilot_contract,
    science_design_sha256,
)
from verified_memory.pilot_preflight_amendment import (
    parent_budget_debit_for_preflight_amendment,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "experiments" / "pilot_v2_2.yaml"
OVERLAY_PATH = ROOT / "experiments" / "pilot_v2_3_overlay.yaml"
FULL_PATH = ROOT / "experiments" / "pilot_v2_3.yaml"

SCIENCE_DESIGN_FIELDS = (
    "seeds",
    "provider_profiles",
    "arms",
    "narratives",
    "shocks",
    "utility",
    "stop_go",
    "stages",
    "parameter_dispatch_policy",
    "task_output_contracts",
    "model_roles",
    "non_claims",
)

LEGACY_CONTRACTS = (
    (
        ROOT / "experiments" / "pilot_v2.yaml",
        PILOT_CONTRACT_ID_V2,
        PILOT_CONTRACT_V2_CANONICAL_SHA256,
    ),
    (
        ROOT / "experiments" / "pilot_v2_1.yaml",
        PILOT_CONTRACT_ID_V2_1,
        PILOT_CONTRACT_V2_1_CANONICAL_SHA256,
    ),
    (
        ROOT / "experiments" / "pilot_v2_2.yaml",
        PILOT_CONTRACT_ID_V2_2,
        PILOT_CONTRACT_V2_2_CANONICAL_SHA256,
    ),
)


def _overlay_document() -> dict[str, Any]:
    return json.loads(OVERLAY_PATH.read_text(encoding="utf-8"))


def _write_resealed_overlay(
    tmp_path: Path,
    value: dict[str, Any],
) -> Path:
    value["integrity"]["declared_sha256"] = canonical_contract_sha256(value)
    (tmp_path / "pilot_v2_2.yaml").write_text(
        BASE_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    path = tmp_path / "pilot_v2_3_overlay.yaml"
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _without_run_identity(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("run_id")
    result.pop("contract_id")
    return result


def test_v2_3_overlay_loads_in_draft_and_freeze_states() -> None:
    """Keep draft checks useful now and turn on release checks at freeze."""

    source = _overlay_document()
    contract = load_pilot_contract(OVERLAY_PATH)

    assert source["schema_version"] == (
        PILOT_CONTRACT_OVERLAY_SCHEMA_VERSION_V2_3
    )
    assert source["integrity"]["declared_sha256"] == (
        canonical_contract_sha256(source)
    )
    assert contract.schema_version == PILOT_CONTRACT_SCHEMA_VERSION_V2
    assert contract.contract_id == PILOT_CONTRACT_ID_V2_3
    assert contract.status == source["status"]
    assert contract.declared_sha256 == contract.canonical_hash
    assert contract.implementation["required_git_tag"] == (
        PILOT_CONTRACT_TAG_V2_3
    )
    assert contract.release_requirements is not None
    assert contract.release_requirements.tag == PILOT_CONTRACT_TAG_V2_3

    expected_ci = dict(contract.release_requirements.expected_ci)
    if source["status"] == "draft":
        assert all(value is None for value in expected_ci.values())
        with pytest.raises(PilotContractError, match="draft contract"):
            contract.validate_provenance(
                "1" * 40,
                PILOT_CONTRACT_TAG_V2_3,
            )
    else:
        assert source["status"] == "frozen"
        assert all(value is not None for value in expected_ci.values())
        assert contract.canonical_hash == (
            PILOT_CONTRACT_V2_3_CANONICAL_SHA256
        )
        assert FULL_PATH.exists()
        assert load_pilot_contract(FULL_PATH).to_dict() == contract.to_dict()
        assert contract.validate_provenance(
            "1" * 40,
            PILOT_CONTRACT_TAG_V2_3,
        )["contract_sha256"] == contract.canonical_hash


def test_v2_3_preserves_science_design_and_exact_174_cell_matrix() -> None:
    base = load_pilot_contract(BASE_PATH)
    amended = load_pilot_contract(OVERLAY_PATH)
    base_value = base.to_dict()
    amended_value = amended.to_dict()

    for field in SCIENCE_DESIGN_FIELDS:
        assert amended_value[field] == base_value[field]
    assert amended_value["budgets"] == base_value["budgets"]
    assert amended_value["operational_amendment"] == (
        base_value["operational_amendment"]
    )
    assert amended_value["evaluator_amendment"] == (
        base_value["evaluator_amendment"]
    )
    assert science_design_sha256(amended_value) == (
        PILOT_CONTRACT_V2_SCIENCE_DESIGN_SHA256
    )
    assert science_design_sha256(base_value) == (
        PILOT_CONTRACT_V2_SCIENCE_DESIGN_SHA256
    )

    base_denominator = dict(base_value["denominator_policy"])
    amended_denominator = dict(amended_value["denominator_policy"])
    assert base_denominator.pop("policy_id") == "finevo-pilot-v2.2-itt"
    assert amended_denominator.pop("policy_id") == "finevo-pilot-v2.3-itt"
    assert amended_denominator == base_denominator

    base_specs = base.expand()
    amended_specs = amended.expand()
    assert len(base_specs) == len(amended_specs) == 174
    assert [
        _without_run_identity(spec.to_dict()) for spec in amended_specs
    ] == [
        _without_run_identity(spec.to_dict()) for spec in base_specs
    ]
    assert all(
        spec.contract_id == PILOT_CONTRACT_ID_V2_3
        and spec.run_id.startswith(f"{PILOT_CONTRACT_ID_V2_3}--")
        for spec in amended_specs
    )


def test_v2_3_amendment_has_exact_retry_and_bootstrap_scope() -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    amendment = contract.preflight_bootstrap_amendment
    assert amendment is not None

    retry = amendment["retry_policy"]
    assert tuple(retry["eligible_stage_ids"]) == ("closed-loop-preflight",)
    assert tuple(retry["eligible_model_ids"]) == (
        "gpt52_main",
        "llama33_local_controlled",
    )
    assert retry["same_environment_seed_required"] == 2010922376
    assert retry["provider_redispatch"] == (
        "allowed-once-after-zero-dispatch-parent"
    )
    assert retry["provider_calls_in_parent_attempt"] == 0
    assert retry["parent_actual_cost_usd"] == 0.0
    assert retry["capability_import_reused"] is True
    assert retry["failed_seed_replacement"] == "forbidden"
    assert retry["model_outputs_inspected"] is False
    assert retry["scientific_effect_outcomes_inspected"] is False

    policy = amendment["bootstrap_policy"]
    assert policy["allowed_execution_mode"] == "closed_loop_preflight"
    assert policy["applies_to_all_registered_closed_loop_preflights"] is True
    assert policy["same_model_validated_capability_source_required"] is True
    assert policy["source_output_contract_map"] == {
        "actor-action": "action",
        "semantic-proposal": "semantic",
    }
    assert policy["required_sample_counts"] == {
        "action": 24,
        "semantic": 6,
    }
    assert policy["target_dispatch_call_counts"] == {
        "action": 12,
        "semantic": 4,
    }
    assert policy["p95_method"] == (
        "nearest-rank-with-observed-maximum-floor"
    )
    assert policy["reserve_multiplier"] == 1.25
    assert policy["unknown_price_policy"] == "stop-before-dispatch"
    assert policy["missing_or_malformed_source_policy"] == (
        "stop-before-dispatch"
    )
    assert policy["normal_scientific_dispatch_policy_unchanged"] is True
    assert policy["normal_scientific_dispatch_reservation_source"] == (
        "sealed-closed-loop-preflight-projection-only"
    )

    registered = tuple(
        spec
        for spec in contract.expand()
        if spec.execution_mode == "closed_loop_preflight"
    )
    assert {
        (spec.stage_id, spec.model_id) for spec in registered
    } == {
        ("closed-loop-preflight", "gpt52_main"),
        ("closed-loop-preflight", "llama33_local_controlled"),
        ("secondary-closed-loop-preflight", "gpt56_diagnostic"),
        (
            "secondary-closed-loop-preflight",
            "gemini35_flash_diagnostic",
        ),
        ("secondary-closed-loop-preflight", "llama4_maverick_diagnostic"),
    }
    assert all(
        spec.environment_seed == 2010922376
        and spec.num_agents == 2
        and spec.episode_length == 6
        for spec in registered
    )

    failure_audits = {
        row["model_id"]: row for row in amendment["failure_audits"]
    }
    assert set(failure_audits) == set(retry["eligible_model_ids"])
    assert all(
        row["completed_provider_calls"] == 0
        and row["prompt_tokens"] == 0
        and row["completion_tokens"] == 0
        and row["cost_usd"] == 0.0
        and row["partial_streams_persisted"] is False
        for row in failure_audits.values()
    )
    assert amendment["defect"]["failure_before_first_provider_dispatch"] is True
    assert amendment["defect"]["provider_key_failure"] is False
    assert amendment["defect"]["model_output_failure"] is False
    assert amendment["defect"]["scientific_effect_outcomes_available"] is False


def test_v2_3_parent_budget_debit_is_exact_and_not_reimported_by_legacy() -> None:
    contract = load_pilot_contract(OVERLAY_PATH)
    debit = parent_budget_debit_for_preflight_amendment(contract)
    assert debit is not None

    assert debit.parent_contract_sha256 == (
        PILOT_CONTRACT_V2_2_CANONICAL_SHA256
    )
    assert debit.parent_run_ledger_sha256 == (
        "19c89a56e6b2317bf97eccd631a472fd2772fd37deede0117d2723b827ed9d42"
    )
    assert debit.parent_budget_ledger_sha256 == (
        "021d451e5b06a893d466848fd6313555dc24c13269376de06e398ff47b3bd998"
    )
    assert debit.stage_bucket == "capability"
    assert debit.cost_usd == 1.53775475
    assert debit.hosted_completions == 60
    assert debit.storage_bytes == 751_437
    assert debit.record_sha256 == (
        "71a1349168861a0ff9dc1546a394e7e0da6ea783e7029e4b3b1056c23906ff59"
    )

    for path, _, _ in LEGACY_CONTRACTS:
        assert (
            parent_budget_debit_for_preflight_amendment(
                load_pilot_contract(path)
            )
            is None
        )


def test_v2_3_unresealed_overlay_tamper_is_rejected(tmp_path: Path) -> None:
    value = _overlay_document()
    value["preflight_bootstrap_amendment"]["defect"][
        "provider_key_failure"
    ] = True
    (tmp_path / "pilot_v2_2.yaml").write_text(
        BASE_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    path = tmp_path / "pilot_v2_3_overlay.yaml"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(PilotContractError, match="overlay hash mismatch"):
        load_pilot_contract(path)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["base_contract"].__setitem__(
                "canonical_sha256",
                "1" * 64,
            ),
            "base contract binding drifted",
        ),
        (
            lambda value: value["changes"].__setitem__("seeds", {}),
            "invalid V2.3 changes keys",
        ),
        (
            lambda value: value["preflight_bootstrap_amendment"][
                "retry_policy"
            ].__setitem__("same_environment_seed_required", 1),
            "preflight bootstrap amendment drifted",
        ),
        (
            lambda value: value["preflight_bootstrap_amendment"][
                "bootstrap_policy"
            ]["target_dispatch_call_counts"].__setitem__("semantic", 3),
            "preflight bootstrap amendment drifted",
        ),
        (
            lambda value: value["preflight_bootstrap_amendment"][
                "budget_carry_forward"
            ].__setitem__("storage_bytes", 751_436),
            "preflight bootstrap amendment drifted",
        ),
    ],
)
def test_v2_3_resealed_forbidden_change_is_rejected(
    tmp_path: Path,
    mutator: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    value = _overlay_document()
    mutator(value)
    path = _write_resealed_overlay(tmp_path, value)

    with pytest.raises(PilotContractError, match=message):
        load_pilot_contract(path)


def test_v2_v2_1_and_v2_2_remain_read_only_compatible() -> None:
    for path, contract_id, canonical_sha256 in LEGACY_CONTRACTS:
        contract = load_pilot_contract(path)
        assert contract.contract_id == contract_id
        assert contract.canonical_hash == canonical_sha256
        assert contract.preflight_bootstrap_amendment is None
        assert len(contract.expand()) == 174
        assert (
            PilotContract.from_dict(contract.to_dict()).to_dict()
            == contract.to_dict()
        )
