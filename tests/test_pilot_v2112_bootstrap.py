from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import verified_memory.pilot_v2112_bootstrap as v2112_bootstrap
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_V2_11_1_CANONICAL_SHA256,
    PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256,
    canonical_sha256,
)
from verified_memory.pilot_v2111_parent_import import (
    V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION,
    V2111_PARENT_IMPORT_SCHEMA_VERSION,
)
from verified_memory.pilot_v2112_bootstrap import (
    PilotV2112BootstrapError,
    V2112_BOOTSTRAP_POLICY_ID,
    V2112_BOOTSTRAP_PROJECTION_FILENAME,
    V2112_BOOTSTRAP_SCHEMA_VERSION,
    V2112_SOURCE_GIT_COMMIT,
    build_v2112_contract_envelope_bootstrap_projection,
    runner_reservations_from_v2112_bootstrap_projection,
    validate_v2112_contract_envelope_bootstrap_projection,
)
from verified_memory.runner import (
    ContractEnvelopeBootstrapReservation,
    V2112_CONTRACT_ENVELOPE_AUTHORITY_ID,
    V2112_CONTRACT_ID,
    V2112_RELEASE_TAG,
    V2112_RUNTIME_MODEL_BY_MODEL_ID,
)


SOURCE_HASH = PILOT_CONTRACT_V2_11_1_CANONICAL_SHA256
TARGET_HASH = PILOT_CONTRACT_V2_11_2_CANONICAL_SHA256 or ("b" * 64)
TARGET_COMMIT = "c" * 40
CONFIG_HASH = "d" * 64
SOURCE_RELEASE = {
    "contract_id": "finevo-pilot-v2.11.1",
    "contract_sha256": SOURCE_HASH,
    "git_tag": "pilot-v2.11.1-science",
    "resolved_git_commit": V2112_SOURCE_GIT_COMMIT,
}
MODEL_DATA = {
    "gpt52_main": {
        "requested_model": "gpt-5.2-2025-12-11",
        "runtime_model": "openai/gpt-5.2-2025-12-11",
        "expected_cost": 0.407344,
        "price": {
            "captured_at": "2026-07-22",
            "catalog_cached_input": 0.175,
            "catalog_input": 1.75,
            "catalog_output": 14.0,
            "currency": "USD",
            "dispatch_basis": "endpoint",
            "endpoint_cached_input": 0.175,
            "endpoint_input": 1.75,
            "endpoint_output": 14.0,
            "source": "https://developers.openai.com/api/docs/models/gpt-5.2",
            "unit": "per_million_tokens",
        },
    },
    "gpt56_diagnostic": {
        "requested_model": "gpt-5.6-sol",
        "runtime_model": "openai/gpt-5.6-sol",
        "expected_cost": 1.12288,
        "price": {
            "captured_at": "2026-07-31",
            "catalog_cached_input": 0.5,
            "catalog_input": 5.0,
            "catalog_output": 30.0,
            "currency": "USD",
            "dispatch_basis": "endpoint",
            "endpoint_cached_input": 0.5,
            "endpoint_input": 5.0,
            "endpoint_output": 30.0,
            "source": "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
            "unit": "per_million_tokens",
        },
    },
}


def _seal(value: dict) -> dict:
    result = deepcopy(value)
    result["integrity"] = {"canonicalization": "json-sort-keys-utf8-v1"}
    result["integrity"]["content_sha256"] = canonical_sha256(result)
    return result


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _historical_run_id(model_id: str) -> str:
    return (
        f"finevo-pilot-v2.11--capability-gate--{model_id}"
        "--capability-probe--none--provider-preflight-default"
        "--s2010922376"
    )


def _source_run_id(model_id: str) -> str:
    return (
        f"finevo-pilot-v2.11.1--capability-gate--{model_id}"
        "--capability-probe--none--provider-preflight-default"
        "--s2010922376"
    )


def _target_run_id(model_id: str) -> str:
    return (
        f"finevo-pilot-v2.11.2--long-context-preflight--{model_id}"
        "--closed-loop-preflight--none--stage0-selected--s2010922376"
    )


def _capability_wrapper(model_id: str) -> dict:
    model = MODEL_DATA[model_id]
    usage_rows = []
    samples = {"action": [], "semantic": []}
    for call_kind, count, prompt_base, completion_base in (
        ("action", 24, 100, 40),
        ("semantic", 6, 500, 100),
    ):
        for index in range(count):
            prompt_tokens = prompt_base + index
            completion_tokens = completion_base + index
            cost_usd = 0.01 + (index / 100_000.0)
            usage_rows.append(
                {
                    "response_model": model["requested_model"],
                    "call_kind": call_kind,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                        "cost_usd": cost_usd,
                    },
                }
            )
            samples[call_kind].append(
                {
                    "parse_success": True,
                    "clipped": False,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )
    actual_usage = {
        "prompt_tokens": sum(row["usage"]["prompt_tokens"] for row in usage_rows),
        "completion_tokens": sum(
            row["usage"]["completion_tokens"] for row in usage_rows
        ),
        "total_tokens": sum(row["usage"]["total_tokens"] for row in usage_rows),
        "cost_usd": sum(row["usage"]["cost_usd"] for row in usage_rows),
    }
    historical_run_id = _historical_run_id(model_id)
    capability = {
        "model_id": model_id,
        "run_id": historical_run_id,
        "runtime_model": model["runtime_model"],
        "requested_model": model["requested_model"],
        "served_model": model["requested_model"],
        "taskset_sha256": "a" * 64,
        "historical_source_calls": 30,
        "action_sample_count": 24,
        "semantic_sample_count": 6,
        "category_totals": {},
        "checks": {},
        "interface_gate": {"pass": True, "failure_count": 0},
        "capability_assessment": {"pass": True},
        "prompt_tier_gate": {
            "passed": True,
            "ceiling_tokens": 200_000,
        },
        "actual_usage": actual_usage,
        "samples": samples,
        "usage_rows": usage_rows,
        "provider_failure_count": 0,
        "parse_failure_count": 0,
        "recovered_parse_count": 0,
        "strict_parse_count": 30,
        "truncation_count": 0,
        "capability_pass": True,
        "interface_pass": True,
        "stage_receipt_content_sha256": "e" * 64,
    }
    return _seal(
        {
            "schema_version": V2111_CAPABILITY_WRAPPER_SCHEMA_VERSION,
            "child_release": deepcopy(SOURCE_RELEASE),
            "parent_release": {
                "contract_id": "finevo-pilot-v2.11",
                "contract_sha256": "f" * 64,
                "git_tag": "pilot-v2.11-science",
                "resolved_git_commit": "1" * 40,
            },
            "source_manifest": {
                "path": "experiments/pilot_v2_11_1_source_manifest.json",
                "file_sha256": "2" * 64,
                "content_sha256": "3" * 64,
            },
            "source_artifacts": {
                "run_id": historical_run_id,
                "runtime_model": model["runtime_model"],
                "historical_source_calls": 30,
                "action_sample_count": 24,
                "semantic_sample_count": 6,
                "capability_pass": True,
                "interface_pass": True,
                "scientific_evidence": False,
                "actual_usage": actual_usage,
            },
            "capability": capability,
            "provider_construction_current_attempt": False,
            "provider_calls_current_attempt": 0,
            "hosted_provider_calls_current_attempt": 0,
            "current_attempt_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            },
            "imported_effect_cells": 0,
            "imported_p95_authorities": [],
            "scientific_evidence": False,
            "evidence_scope": "preregistered_task_capability_gate",
            "evidence_use": "capability audit only",
        }
    )


def _parent_receipt(wrappers: dict[str, dict]) -> dict:
    return _seal(
        {
            "schema_version": V2111_PARENT_IMPORT_SCHEMA_VERSION,
            "child_release": deepcopy(SOURCE_RELEASE),
            "capability_wrappers": deepcopy(wrappers),
            "provider_construction_current_attempt": False,
            "provider_calls_current_attempt": 0,
            "hosted_provider_calls_current_attempt": 0,
            "import_policy": {
                "provider_construction_during_import": False,
                "provider_calls_during_import": 0,
                "hosted_provider_calls_during_import": 0,
                "imported_capability_wrappers": 2,
                "historical_capability_calls": 60,
                "imported_effect_cells": 0,
                "imported_p95_authorities": [],
            },
            "scientific_evidence": False,
        }
    )


def _source_spec(model_id: str) -> dict:
    return {
        "model_id": model_id,
        "requested_model": MODEL_DATA[model_id]["requested_model"],
        "environment_seed": 2010922376,
        "decoding_seed": None,
        "narrative_id": "none",
        "num_agents": 2,
        "contract_id": "finevo-pilot-v2.11.1",
        "stage_id": "capability-gate",
        "execution_mode": "capability_authority_import",
        "run_id": _source_run_id(model_id),
        "arm_id": "capability-probe",
        "budget_bucket": "parent_v211",
        "episode_length": 1,
        "shock_id": "baseline-3pct",
        "utility_profile_id": "provider-preflight-default",
    }


def _target_spec(model_id: str) -> dict:
    return {
        "model_id": model_id,
        "requested_model": MODEL_DATA[model_id]["requested_model"],
        "environment_seed": 2010922376,
        "decoding_seed": None,
        "narrative_id": "none",
        "num_agents": 2,
        "contract_id": V2112_CONTRACT_ID,
        "stage_id": "long-context-preflight",
        "execution_mode": "closed_loop_preflight",
        "run_id": _target_run_id(model_id),
        "arm_id": "closed-loop-preflight",
        "budget_bucket": "hosted_v2112",
        "episode_length": 12,
        "shock_id": "registered-rate-shock",
        "utility_profile_id": "stage0-selected",
    }


def _provider_profile(model_id: str) -> dict:
    model = MODEL_DATA[model_id]
    return {
        "profile_id": model_id,
        "transport": "openai",
        "requested_model": model["requested_model"],
        "served_model": model["requested_model"],
        "price_snapshot": deepcopy(model["price"]),
    }


def _case(
    tmp_path: Path,
    model_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, dict, dict]:
    wrappers = {item: _capability_wrapper(item) for item in sorted(MODEL_DATA)}
    receipt = _parent_receipt(wrappers)
    wrapper = wrappers[model_id]
    receipt_path = tmp_path / "parent-import" / "parent_import_receipt.json"
    receipt_hash = _write_json(receipt_path, receipt)
    wrapper_path = tmp_path / "capability-gate" / model_id / "capability.json"
    wrapper_hash = _write_json(wrapper_path, wrapper)
    monkeypatch.setattr(
        v2112_bootstrap,
        "V2112_SOURCE_PARENT_RECEIPT_FILE_SHA256",
        receipt_hash,
    )
    monkeypatch.setattr(
        v2112_bootstrap,
        "V2112_SOURCE_PARENT_RECEIPT_CONTENT_SHA256",
        receipt["integrity"]["content_sha256"],
    )
    synthetic_bindings = deepcopy(
        v2112_bootstrap.V2112_SOURCE_CAPABILITY_WRAPPER_BINDINGS
    )
    synthetic_bindings[model_id] = {
        "file_sha256": wrapper_hash,
        "content_sha256": wrapper["integrity"]["content_sha256"],
    }
    monkeypatch.setattr(
        v2112_bootstrap,
        "V2112_SOURCE_CAPABILITY_WRAPPER_BINDINGS",
        synthetic_bindings,
    )
    kwargs = {
        "source_parent_receipt": receipt,
        "model_id": model_id,
        "source_contract_sha256": SOURCE_HASH,
        "source_capability_spec": _source_spec(model_id),
        "target_contract_sha256": TARGET_HASH,
        "target_preflight_spec": _target_spec(model_id),
        "provider_profile": _provider_profile(model_id),
        "source_parent_receipt_path": receipt_path,
        "source_parent_receipt_file_sha256": receipt_hash,
        "source_capability_path": wrapper_path,
        "source_capability_file_sha256": wrapper_hash,
        "source_git_tag": "pilot-v2.11.1-science",
        "source_git_commit": V2112_SOURCE_GIT_COMMIT,
        "target_git_tag": V2112_RELEASE_TAG,
        "target_git_commit": TARGET_COMMIT,
        "authorized_config_sha256": CONFIG_HASH,
    }
    projection = build_v2112_contract_envelope_bootstrap_projection(
        wrapper,
        **kwargs,
    )
    return wrapper, projection, kwargs


@pytest.mark.parametrize("model_id", sorted(MODEL_DATA))
def test_v2112_bootstrap_is_exact_v2111_to_v2112_envelope_authority(
    tmp_path: Path,
    model_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, projection, kwargs = _case(tmp_path, model_id, monkeypatch)
    validate_v2112_contract_envelope_bootstrap_projection(
        projection,
        wrapper,
        **kwargs,
    )
    reservations = runner_reservations_from_v2112_bootstrap_projection(
        projection,
        wrapper,
        **kwargs,
    )

    runtime_model = MODEL_DATA[model_id]["runtime_model"]
    assert V2112_BOOTSTRAP_PROJECTION_FILENAME == (
        "v2112_contract_envelope_bootstrap.json"
    )
    assert projection["schema_version"] == V2112_BOOTSTRAP_SCHEMA_VERSION
    assert projection["policy"]["policy_id"] == V2112_BOOTSTRAP_POLICY_ID
    assert projection["scientific_evidence"] is False
    assert projection["source"]["contract_id"] == "finevo-pilot-v2.11.1"
    assert projection["source"]["git_tag"] == "pilot-v2.11.1-science"
    assert projection["source"]["run_spec"]["run_id"] == _source_run_id(model_id)
    assert projection["target"]["contract_id"] == V2112_CONTRACT_ID
    assert projection["target"]["git_tag"] == V2112_RELEASE_TAG
    assert projection["target"]["authorized_seed"] == 2010922376
    assert projection["target"]["authorized_runner_run_id"] == (
        f"{_target_run_id(model_id)}--actor-preflight"
    )
    assert projection["policy"]["target_shape"] == {
        "num_agents": 2,
        "episode_length": 12,
        "action_calls": 24,
        "semantic_calls": 8,
    }
    assert projection["capability_projection"]["action"]["sample_count"] == 24
    assert projection["capability_projection"]["semantic"]["sample_count"] == 6
    for call_kind in ("action", "semantic"):
        assert projection["contract_envelope"][call_kind] == {
            "prompt_tokens": 200_000,
            "completion_tokens": 4_096,
            "total_tokens": 204_096,
            "cost_usd": pytest.approx(MODEL_DATA[model_id]["expected_cost"]),
        }
        entry = reservations[runtime_model][call_kind]
        parsed = ContractEnvelopeBootstrapReservation.from_dict(
            model=runtime_model,
            call_kind=call_kind,
            value=entry,
        )
        assert parsed.authority_id == V2112_CONTRACT_ENVELOPE_AUTHORITY_ID
        assert parsed.target_contract_id == V2112_CONTRACT_ID
        assert parsed.source_contract_id == "finevo-pilot-v2.11.1"
        assert parsed.source_run_id == _source_run_id(model_id)
        assert parsed.authorized_run_id == (
            f"{_target_run_id(model_id)}--actor-preflight"
        )
        assert parsed.reserved_usage.prompt_tokens == 200_000
        assert parsed.reserved_usage.completion_tokens == 4_096

    assert set(reservations) == {V2112_RUNTIME_MODEL_BY_MODEL_ID[model_id]}
    assert projection["source"]["parent_receipt_payload_sha256"] == (
        canonical_sha256(kwargs["source_parent_receipt"])
    )
    assert projection["source"]["capability_wrapper_content_sha256"] == (
        wrapper["integrity"]["content_sha256"]
    )


def test_v2112_bootstrap_rejects_v2111_projection_and_cross_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, projection, kwargs = _case(
        tmp_path,
        "gpt52_main",
        monkeypatch,
    )

    old_projection = deepcopy(projection)
    old_projection["schema_version"] = (
        "finevo-pilot-v2.11.1-contract-envelope-bootstrap-v1"
    )
    unsigned = deepcopy(old_projection)
    unsigned.pop("integrity")
    old_projection["integrity"]["content_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(
        PilotV2112BootstrapError,
        match="not the V2.11.2 schema",
    ):
        validate_v2112_contract_envelope_bootstrap_projection(
            old_projection,
            wrapper,
            **kwargs,
        )

    old_source_projection = deepcopy(projection)
    old_source_projection["schema_version"] = (
        "finevo-pilot-v2.11.1-contract-envelope-bootstrap-v1"
    )
    old_source_projection["integrity"] = {
        "canonicalization": "json-sort-keys-utf8-v1",
        "content_sha256": "0" * 64,
    }
    with pytest.raises(
        PilotV2112BootstrapError,
        match="capability wrapper schema drifted",
    ):
        build_v2112_contract_envelope_bootstrap_projection(
            old_source_projection,
            **kwargs,
        )

    wrong_source_spec = deepcopy(kwargs["source_capability_spec"])
    wrong_source_spec["contract_id"] = V2112_CONTRACT_ID
    wrong_source_spec["run_id"] = wrong_source_spec["run_id"].replace(
        "v2.11.1", "v2.11.2"
    )
    with pytest.raises(
        PilotV2112BootstrapError,
        match="exact V2.11.1 cell",
    ):
        build_v2112_contract_envelope_bootstrap_projection(
            wrapper,
            **{**kwargs, "source_capability_spec": wrong_source_spec},
        )

    with pytest.raises(
        PilotV2112BootstrapError,
        match="release lineage drifted",
    ):
        build_v2112_contract_envelope_bootstrap_projection(
            wrapper,
            **{**kwargs, "target_git_tag": "pilot-v2.11.1-science"},
        )


def test_v2112_bootstrap_rejects_receipt_wrapper_and_price_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, _, kwargs = _case(tmp_path, "gpt52_main", monkeypatch)

    bad_receipt = deepcopy(kwargs["source_parent_receipt"])
    bad_receipt["integrity"]["content_sha256"] = "0" * 64
    with pytest.raises(
        PilotV2112BootstrapError,
        match="parent import receipt self-hash mismatch",
    ):
        build_v2112_contract_envelope_bootstrap_projection(
            wrapper,
            **{**kwargs, "source_parent_receipt": bad_receipt},
        )

    bad_wrapper = deepcopy(wrapper)
    bad_wrapper["capability"]["usage_rows"][0]["usage"]["prompt_tokens"] += 1
    with pytest.raises(
        PilotV2112BootstrapError,
        match="capability wrapper self-hash mismatch",
    ):
        build_v2112_contract_envelope_bootstrap_projection(
            bad_wrapper,
            **kwargs,
        )

    repriced = deepcopy(kwargs["provider_profile"])
    repriced["price_snapshot"]["endpoint_input"] = 0.0
    with pytest.raises(
        PilotV2112BootstrapError,
        match="frozen V2.11.2 model/price",
    ):
        build_v2112_contract_envelope_bootstrap_projection(
            wrapper,
            **{**kwargs, "provider_profile": repriced},
        )

    swapped = _provider_profile("gpt56_diagnostic")
    with pytest.raises(
        PilotV2112BootstrapError,
        match="frozen V2.11.2 model/price",
    ):
        build_v2112_contract_envelope_bootstrap_projection(
            wrapper,
            **{**kwargs, "provider_profile": swapped},
        )


def test_v2112_bootstrap_rejects_shape_seed_file_and_projection_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, projection, kwargs = _case(
        tmp_path,
        "gpt56_diagnostic",
        monkeypatch,
    )

    wrong_target = deepcopy(kwargs["target_preflight_spec"])
    wrong_target["num_agents"] = 4
    with pytest.raises(
        PilotV2112BootstrapError,
        match="exact V2.11.2 cell",
    ):
        build_v2112_contract_envelope_bootstrap_projection(
            wrapper,
            **{**kwargs, "target_preflight_spec": wrong_target},
        )

    wrong_source = deepcopy(kwargs["source_capability_spec"])
    wrong_source["environment_seed"] += 1
    with pytest.raises(
        PilotV2112BootstrapError,
        match="exact V2.11.1 cell",
    ):
        build_v2112_contract_envelope_bootstrap_projection(
            wrapper,
            **{**kwargs, "source_capability_spec": wrong_source},
        )

    with pytest.raises(
        PilotV2112BootstrapError,
        match="capability wrapper file hash mismatch",
    ):
        build_v2112_contract_envelope_bootstrap_projection(
            wrapper,
            **{**kwargs, "source_capability_file_sha256": "9" * 64},
        )

    tampered = deepcopy(projection)
    tampered["contract_envelope"]["action"]["prompt_tokens"] = 199_999
    tampered["contract_envelope"]["action"]["total_tokens"] = 204_095
    unsigned = deepcopy(tampered)
    unsigned.pop("integrity")
    tampered["integrity"]["content_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(
        PilotV2112BootstrapError,
        match="differs from its exact reconstructed source",
    ):
        validate_v2112_contract_envelope_bootstrap_projection(
            tampered,
            wrapper,
            **kwargs,
        )
