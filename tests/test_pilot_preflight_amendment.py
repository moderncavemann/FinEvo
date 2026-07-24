from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from verified_memory.pilot_contract import canonical_sha256, load_pilot_contract
from verified_memory.pilot_evaluation_amendment import (
    build_capability_import,
    load_evaluator_amendment_receipt,
)
from verified_memory.pilot_orchestrator import (
    GitProvenance,
    _preflight_config,
)
from verified_memory.pilot_preflight_amendment import (
    PilotPreflightAmendmentError,
    build_capability_bootstrap_projection,
    runner_reservations_from_bootstrap_projection,
    validate_capability_bootstrap_projection,
)
from verified_memory.runner import (
    VerifiedRunError,
    bootstrap_config_binding_sha256,
    preflight_p95_reservation_for_call,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "experiments" / "pilot_v2_3_overlay.yaml"
MODEL_EXPECTATIONS = {
    "gpt52_main": {
        "runtime_model": "openai/gpt-5.2-2025-12-11",
        "hosted": True,
        "action": {
            "sample_count": 24,
            "raw_prompt_tokens": 910.0,
            "raw_completion_tokens": 1907.0,
            "raw_cost_usd": 0.02726675,
            "reserved_prompt_tokens": 1138,
            "reserved_completion_tokens": 2384,
            "reserved_cost_usd": 0.0340834375,
        },
        "semantic": {
            "sample_count": 6,
            "raw_prompt_tokens": 1920.0,
            "raw_completion_tokens": 758.0,
            "raw_cost_usd": 0.0139195,
            "reserved_prompt_tokens": 2400,
            "reserved_completion_tokens": 948,
            "reserved_cost_usd": 0.017399375,
        },
    },
    "llama33_local_controlled": {
        "runtime_model": "ollama/llama3.3:70b-instruct-q4_K_M",
        "hosted": False,
        "action": {
            "sample_count": 24,
            "raw_prompt_tokens": 907.0,
            "raw_completion_tokens": 31.0,
            "raw_cost_usd": 0.0,
            "reserved_prompt_tokens": 1134,
            "reserved_completion_tokens": 39,
            "reserved_cost_usd": 0.0,
        },
        "semantic": {
            "sample_count": 6,
            "raw_prompt_tokens": 1913.0,
            "raw_completion_tokens": 170.0,
            "raw_cost_usd": 0.0,
            "reserved_prompt_tokens": 2392,
            "reserved_completion_tokens": 213,
            "reserved_cost_usd": 0.0,
        },
    },
}


@pytest.fixture(scope="module")
def contract():
    return load_pilot_contract(CONTRACT_PATH)


@pytest.fixture(scope="module")
def evaluator_receipt(contract):
    receipt, _ = load_evaluator_amendment_receipt(
        repo_root=ROOT,
        contract=contract,
    )
    return receipt


def _paid(contract) -> GitProvenance:
    commit = "1" * 40
    return GitProvenance(
        git_tag=contract.implementation["required_git_tag"],
        head_commit=commit,
        tag_commit=commit,
        tag_object_type="tag",
        worktree_clean=True,
        contract_binding={},
        release_attestation=None,
    )


def _reseal_import(value: dict[str, Any]) -> None:
    unsigned = deepcopy(value)
    unsigned.pop("integrity")
    value["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(unsigned),
    }


def _reseal_projection(value: dict[str, Any]) -> None:
    unsigned = deepcopy(value)
    unsigned.pop("integrity")
    value["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": canonical_sha256(unsigned),
    }


def _source_kwargs_for_payload(
    case: "BootstrapCase",
    payload: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    kwargs = case.kwargs()
    kwargs["source_capability_path"] = path
    kwargs["source_capability_file_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    return kwargs


@dataclass(frozen=True)
class BootstrapCase:
    contract: Any
    capability_spec: Any
    preflight_spec: Any
    capability: dict[str, Any]
    capability_path: Path
    capability_file_sha256: str
    paid: GitProvenance
    authorized_config_sha256: str
    projection: dict[str, Any]
    reservations: dict[str, dict[str, Any]]

    @property
    def runtime_model(self) -> str:
        return str(self.projection["runtime_model"])

    def kwargs(self) -> dict[str, Any]:
        return {
            "source_capability_path": self.capability_path,
            "source_capability_file_sha256": self.capability_file_sha256,
            "git_tag": self.paid.git_tag,
            "git_commit": self.paid.head_commit,
            "authorized_config_sha256": self.authorized_config_sha256,
        }


def _bootstrap_case(
    tmp_path: Path,
    *,
    contract,
    evaluator_receipt,
    model_id: str,
) -> BootstrapCase:
    capability_spec = contract.expand(
        stage="capability-gate",
        model=model_id,
    )[0]
    preflight_spec = contract.expand(
        stage="closed-loop-preflight",
        model=model_id,
    )[0]
    capability = build_capability_import(
        contract,
        capability_spec,
        evaluator_receipt,
    )
    capability_path = tmp_path / model_id / "capability.json"
    capability_path.parent.mkdir(parents=True, exist_ok=True)
    capability_path.write_text(
        json.dumps(capability, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    capability_file_sha256 = hashlib.sha256(
        capability_path.read_bytes()
    ).hexdigest()
    paid = _paid(contract)
    provisional = _preflight_config(
        contract,
        preflight_spec,
        paid=paid,
    )
    authorized_config_sha256 = bootstrap_config_binding_sha256(
        provisional,
        measurement_role="closed_loop_preflight",
    )
    build_kwargs = {
        "source_capability_path": capability_path,
        "source_capability_file_sha256": capability_file_sha256,
        "git_tag": paid.git_tag,
        "git_commit": paid.head_commit,
        "authorized_config_sha256": authorized_config_sha256,
    }
    projection = build_capability_bootstrap_projection(
        contract,
        capability_spec,
        preflight_spec,
        capability,
        **build_kwargs,
    )
    reservations = runner_reservations_from_bootstrap_projection(
        projection,
        contract=contract,
        capability_spec=capability_spec,
        target_preflight_spec=preflight_spec,
        capability=capability,
        **build_kwargs,
    )
    return BootstrapCase(
        contract=contract,
        capability_spec=capability_spec,
        preflight_spec=preflight_spec,
        capability=capability,
        capability_path=capability_path,
        capability_file_sha256=capability_file_sha256,
        paid=paid,
        authorized_config_sha256=authorized_config_sha256,
        projection=projection,
        reservations=reservations,
    )


@pytest.mark.parametrize("model_id", tuple(MODEL_EXPECTATIONS))
def test_imported_capability_builds_exact_24_6_bootstrap_projection(
    tmp_path: Path,
    contract,
    evaluator_receipt,
    model_id: str,
) -> None:
    case = _bootstrap_case(
        tmp_path,
        contract=contract,
        evaluator_receipt=evaluator_receipt,
        model_id=model_id,
    )
    expected = MODEL_EXPECTATIONS[model_id]

    assert case.projection["runtime_model"] == expected["runtime_model"]
    assert case.projection["scientific_evidence"] is False
    assert set(case.projection["projection"]) == {
        f"{case.runtime_model}::action",
        f"{case.runtime_model}::semantic",
    }
    for call_kind in ("action", "semantic"):
        observed = case.projection["projection"][
            f"{case.runtime_model}::{call_kind}"
        ]
        target = expected[call_kind]
        assert observed["sample_count"] == target["sample_count"]
        assert observed["reserve_multiplier"] == 1.25
        assert observed["raw_p95"]["prompt_tokens"] == pytest.approx(
            target["raw_prompt_tokens"]
        )
        assert observed["raw_p95"]["completion_tokens"] == pytest.approx(
            target["raw_completion_tokens"]
        )
        assert observed["raw_p95"]["cost_usd"] == pytest.approx(
            target["raw_cost_usd"]
        )
        assert observed["reserved_p95"]["prompt_tokens"] == (
            target["reserved_prompt_tokens"]
        )
        assert observed["reserved_p95"]["completion_tokens"] == (
            target["reserved_completion_tokens"]
        )
        assert observed["reserved_p95"]["cost_usd"] == pytest.approx(
            target["reserved_cost_usd"]
        )
        if expected["hosted"]:
            assert observed["reserved_p95"]["cost_usd"] > 0
        else:
            assert observed["reserved_p95"]["cost_usd"] == 0


def test_imported_source_rejects_duplicate_rows_and_extra_usage_fields(
    tmp_path: Path,
    contract,
    evaluator_receipt,
) -> None:
    case = _bootstrap_case(
        tmp_path,
        contract=contract,
        evaluator_receipt=evaluator_receipt,
        model_id="gpt52_main",
    )

    duplicate = deepcopy(case.capability)
    duplicate["usage_projection_rows"][1] = deepcopy(
        duplicate["usage_projection_rows"][0]
    )
    _reseal_import(duplicate)
    duplicate_kwargs = _source_kwargs_for_payload(
        case,
        duplicate,
        tmp_path / "duplicate-capability.json",
    )
    with pytest.raises(
        PilotPreflightAmendmentError,
        match="task denominator/order drifted",
    ):
        build_capability_bootstrap_projection(
            case.contract,
            case.capability_spec,
            case.preflight_spec,
            duplicate,
            **duplicate_kwargs,
        )

    extra_field = deepcopy(case.capability)
    extra_field["usage_projection_rows"][0]["usage"]["cached_tokens"] = 1
    _reseal_import(extra_field)
    extra_field_kwargs = _source_kwargs_for_payload(
        case,
        extra_field,
        tmp_path / "extra-field-capability.json",
    )
    with pytest.raises(
        PilotPreflightAmendmentError,
        match="usage fields drifted",
    ):
        build_capability_bootstrap_projection(
            case.contract,
            case.capability_spec,
            case.preflight_spec,
            extra_field,
            **extra_field_kwargs,
        )


def test_projection_rejects_stale_hash_and_rehashed_extra_field(
    tmp_path: Path,
    contract,
    evaluator_receipt,
) -> None:
    case = _bootstrap_case(
        tmp_path,
        contract=contract,
        evaluator_receipt=evaluator_receipt,
        model_id="gpt52_main",
    )

    stale = deepcopy(case.projection)
    stale["integrity"]["content_sha256"] = "0" * 64
    with pytest.raises(
        PilotPreflightAmendmentError,
        match="differs from its capability source",
    ):
        validate_capability_bootstrap_projection(
            stale,
            case.contract,
            case.capability_spec,
            case.preflight_spec,
            case.capability,
            **case.kwargs(),
        )

    extra_field = deepcopy(case.projection)
    extra_field["unregistered_authority"] = True
    _reseal_projection(extra_field)
    with pytest.raises(
        PilotPreflightAmendmentError,
        match="differs from its capability source",
    ):
        validate_capability_bootstrap_projection(
            extra_field,
            case.contract,
            case.capability_spec,
            case.preflight_spec,
            case.capability,
            **case.kwargs(),
        )


def test_converter_emits_only_the_separate_exact_bootstrap_authority(
    tmp_path: Path,
    contract,
    evaluator_receipt,
) -> None:
    case = _bootstrap_case(
        tmp_path,
        contract=contract,
        evaluator_receipt=evaluator_receipt,
        model_id="gpt52_main",
    )

    assert set(case.reservations) == {case.runtime_model}
    assert set(case.reservations[case.runtime_model]) == {
        "action",
        "semantic",
    }
    expected_authority_fields = {
        "authority_id",
        "pilot_contract_hash",
        "pilot_tag",
        "authorized_run_id",
        "authorized_seed",
        "authorized_config_sha256",
        "target_run_spec_sha256",
        "source_run_id",
        "source_capability_file_sha256",
        "source_group_sha256",
        "policy_sha256",
        "source_projection_sha256",
    }
    for call_kind, item in case.reservations[case.runtime_model].items():
        assert set(item) == {"authority", "reservation"}
        assert set(item["authority"]) == expected_authority_fields
        assert item["authority"]["authority_id"] == (
            "finevo-capability-observed-bootstrap-v1"
        )
        assert item["authority"]["authorized_run_id"] == (
            f"{case.preflight_spec.run_id}--actor-preflight"
        )
        assert item["authority"]["authorized_config_sha256"] == (
            case.authorized_config_sha256
        )
        assert item["reservation"] == case.projection["projection"][
            f"{case.runtime_model}::{call_kind}"
        ]

    tampered = deepcopy(case.projection)
    tampered["projection"][f"{case.runtime_model}::action"]["sample_count"] = 23
    _reseal_projection(tampered)
    with pytest.raises(
        PilotPreflightAmendmentError,
        match="differs from its capability source",
    ):
        runner_reservations_from_bootstrap_projection(
            tampered,
            contract=case.contract,
            capability_spec=case.capability_spec,
            target_preflight_spec=case.preflight_spec,
            capability=case.capability,
            **case.kwargs(),
        )


def test_runner_accepts_only_the_exact_contract_tag_run_seed_and_config(
    tmp_path: Path,
    contract,
    evaluator_receipt,
) -> None:
    case = _bootstrap_case(
        tmp_path,
        contract=contract,
        evaluator_receipt=evaluator_receipt,
        model_id="gpt52_main",
    )
    config = _preflight_config(
        case.contract,
        case.preflight_spec,
        paid=case.paid,
        contract_bootstrap_reservations=case.reservations,
    )

    assert config.preflight_measurement_role == "closed_loop_preflight"
    assert bootstrap_config_binding_sha256(config) == (
        case.authorized_config_sha256
    )
    assert len(config.contract_bootstrap_reservations) == 2
    usage = preflight_p95_reservation_for_call(
        config,
        provider_model_name=case.runtime_model,
        call_kind="action",
    )
    assert usage.prompt_tokens == 1138
    assert usage.completion_tokens == 2384
    assert usage.cost_usd == pytest.approx(0.0340834375)

    for escaped in (
        {"seed": config.seed + 1},
        {"pilot_tag": "not-the-authorized-tag"},
        {"retrieval_k": config.retrieval_k - 1},
    ):
        with pytest.raises(
            ValueError,
            match="not bound to this exact contract/tag/run/seed/config",
        ):
            replace(config, **escaped)


def test_builder_rejects_preflight_seed_or_tag_escape(
    tmp_path: Path,
    contract,
    evaluator_receipt,
) -> None:
    case = _bootstrap_case(
        tmp_path,
        contract=contract,
        evaluator_receipt=evaluator_receipt,
        model_id="gpt52_main",
    )

    with pytest.raises(
        PilotPreflightAmendmentError,
        match="exact registered contract cells",
    ):
        build_capability_bootstrap_projection(
            case.contract,
            case.capability_spec,
            replace(
                case.preflight_spec,
                environment_seed=case.preflight_spec.environment_seed + 1,
            ),
            case.capability,
            **case.kwargs(),
        )

    for source_spec, target_spec in (
        (
            replace(
                case.capability_spec,
                stage_id="unregistered-capability-stage",
            ),
            case.preflight_spec,
        ),
        (
            case.capability_spec,
            replace(
                case.preflight_spec,
                stage_id="unregistered-preflight-stage",
            ),
        ),
        (
            case.capability_spec,
            replace(
                case.preflight_spec,
                arm_id="unregistered-preflight-arm",
            ),
        ),
    ):
        with pytest.raises(
            PilotPreflightAmendmentError,
            match="exact registered contract cells",
        ):
            build_capability_bootstrap_projection(
                case.contract,
                source_spec,
                target_spec,
                case.capability,
                **case.kwargs(),
            )

    wrong_tag = case.kwargs()
    wrong_tag["git_tag"] = "not-the-contract-tag"
    with pytest.raises(
        PilotPreflightAmendmentError,
        match="cell/model/seed/tag drifted",
    ):
        build_capability_bootstrap_projection(
            case.contract,
            case.capability_spec,
            case.preflight_spec,
            case.capability,
            **wrong_tag,
        )

    wrong_commit = case.kwargs()
    wrong_commit["git_commit"] = "not-a-commit"
    with pytest.raises(
        PilotPreflightAmendmentError,
        match="lowercase 40-hex commit",
    ):
        build_capability_bootstrap_projection(
            case.contract,
            case.capability_spec,
            case.preflight_spec,
            case.capability,
            **wrong_commit,
        )

    stale_source = case.kwargs()
    stale_source["source_capability_file_sha256"] = "0" * 64
    with pytest.raises(
        PilotPreflightAmendmentError,
        match="source capability file hash mismatch",
    ):
        build_capability_bootstrap_projection(
            case.contract,
            case.capability_spec,
            case.preflight_spec,
            case.capability,
            **stale_source,
        )

    missing_source = case.kwargs()
    missing_source["source_capability_path"] = tmp_path / "missing.json"
    with pytest.raises(
        PilotPreflightAmendmentError,
        match="regular non-symlink file",
    ):
        build_capability_bootstrap_projection(
            case.contract,
            case.capability_spec,
            case.preflight_spec,
            case.capability,
            **missing_source,
        )

    mismatched_capability = deepcopy(case.capability)
    usage = mismatched_capability["usage_projection_rows"][0]["usage"]
    usage["prompt_tokens"] += 1
    usage["total_tokens"] += 1
    _reseal_import(mismatched_capability)
    with pytest.raises(
        PilotPreflightAmendmentError,
        match="source capability file/payload mismatch",
    ):
        build_capability_bootstrap_projection(
            case.contract,
            case.capability_spec,
            case.preflight_spec,
            mismatched_capability,
            **case.kwargs(),
        )

    invalid_json_path = tmp_path / "invalid-capability.json"
    invalid_json_path.write_bytes(b'{"incomplete":')
    invalid_json_source = case.kwargs()
    invalid_json_source["source_capability_path"] = invalid_json_path
    invalid_json_source["source_capability_file_sha256"] = hashlib.sha256(
        invalid_json_path.read_bytes()
    ).hexdigest()
    with pytest.raises(
        PilotPreflightAmendmentError,
        match="not strict UTF-8 JSON",
    ):
        build_capability_bootstrap_projection(
            case.contract,
            case.capability_spec,
            case.preflight_spec,
            case.capability,
            **invalid_json_source,
        )

    symlink_path = tmp_path / "capability-link.json"
    symlink_path.symlink_to(case.capability_path)
    symlink_source = case.kwargs()
    symlink_source["source_capability_path"] = symlink_path
    with pytest.raises(
        PilotPreflightAmendmentError,
        match="regular non-symlink file",
    ):
        build_capability_bootstrap_projection(
            case.contract,
            case.capability_spec,
            case.preflight_spec,
            case.capability,
            **symlink_source,
        )


def test_bootstrap_authority_cannot_masquerade_as_normal_scientific_p95(
    tmp_path: Path,
    contract,
    evaluator_receipt,
) -> None:
    case = _bootstrap_case(
        tmp_path,
        contract=contract,
        evaluator_receipt=evaluator_receipt,
        model_id="gpt52_main",
    )
    measurement = _preflight_config(
        case.contract,
        case.preflight_spec,
        paid=case.paid,
        contract_bootstrap_reservations=case.reservations,
    )

    with pytest.raises(
        ValueError,
        match="restricted to the exact 2-agent x 6-month closed-loop preflight",
    ):
        replace(measurement, episode_length=12)

    with pytest.raises(
        ValueError,
        match="sealed observed p95 authority fields drifted",
    ):
        replace(
            measurement,
            contract_bootstrap_reservations=(),
            preflight_measurement_role=None,
            preflight_p95_reservations=case.reservations,
        )

    laundered = {
        case.runtime_model: {
            call_kind: item["reservation"]
            for call_kind, item in case.reservations[
                case.runtime_model
            ].items()
        }
    }
    with pytest.raises(
        ValueError,
        match="requires sealed closed-loop observed authority",
    ):
        replace(
            measurement,
            contract_bootstrap_reservations=(),
            preflight_measurement_role=None,
            preflight_p95_reservations=laundered,
        )

    normal_science = replace(
        measurement,
        contract_bootstrap_reservations=(),
        preflight_measurement_role=None,
    )
    with pytest.raises(
        VerifiedRunError,
        match="lacks an exact observed\\+25% preflight p95 reservation",
    ):
        preflight_p95_reservation_for_call(
            normal_science,
            provider_model_name=case.runtime_model,
            call_kind="action",
        )
