from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

import pytest

from verified_memory.observed_p95_authority import (
    ObservedP95AuthorityError,
    _read_regular_bytes,
    verify_observed_p95_authority_receipt,
)
from verified_memory.pilot_checkpoint import (
    verify_closed_loop_preflight_checkpoint,
)
from verified_memory.pilot_contract import (
    PILOT_CONTRACT_V2_3_CANONICAL_SHA256,
)
from verified_memory.pilot_orchestrator import (
    _CheckpointPreflightResult,
    _load_verified_projection,
    _persist_observed_p95_authority_receipt,
    _runner_p95_reservations,
)
from verified_memory.runner import bootstrap_config_binding_sha256


RUNTIME_MODEL = "openai/gpt-5.2-2025-12-11"
SERVED_MODEL = "gpt-5.2-2025-12-11"


def test_v2_3_bootstrap_authority_runs_exact_offline_preflight_checkpoint(
    observed_p95_source_chain: Any,
) -> None:
    case = observed_p95_source_chain
    contract = case.contract
    provider = case.provider
    checkpoint = case.checkpoint
    config = case.config

    assert contract.status == "frozen"
    assert contract.canonical_hash == (
        PILOT_CONTRACT_V2_3_CANONICAL_SHA256
    )
    assert case.raw_root.parent.name == "experiment_results"
    assert all(
        path.is_relative_to(case.repo_root / "experiment_results")
        for path in case.source_paths.values()
    )

    action_prompts = [
        prompt for prompt in provider.prompts if "monthly decision t=" in prompt
    ]
    semantic_prompts = [
        prompt
        for prompt in provider.prompts
        if "Propose one semantic decision rule" in prompt
    ]
    assert len(provider.prompts) == 16
    assert len(action_prompts) == 12
    assert len(semantic_prompts) == 4
    assert checkpoint.payload["provider_denominator"] == {
        "planned_calls": 16,
        "observed_calls": 16,
        "successful_terminal_calls": 16,
        "failed_calls": 0,
        "action_calls": 12,
        "semantic_calls": 4,
        "semantic_candidate_parse_failures": 0,
    }

    checkpoint_config = checkpoint.payload["run_config"]
    assert checkpoint_config == config.to_dict()
    assert (
        checkpoint_config["preflight_measurement_role"]
        == "closed_loop_preflight"
    )
    assert checkpoint_config["preflight_p95_reservations"] == {}
    assert checkpoint_config["contract_bootstrap_reservations"]
    assert (
        bootstrap_config_binding_sha256(config)
        == next(
            iter(
                checkpoint_config["contract_bootstrap_reservations"][
                    RUNTIME_MODEL
                ].values()
            )
        )["authority"]["authorized_config_sha256"]
    )
    for call_kind in ("action", "semantic"):
        authority = checkpoint_config["contract_bootstrap_reservations"][
            RUNTIME_MODEL
        ][call_kind]["authority"]
        assert authority["authorized_run_id"] == config.run_id
        assert authority["authorized_seed"] == config.seed
        assert authority["pilot_contract_hash"] == contract.canonical_hash
        assert authority["pilot_tag"] == case.paid.git_tag

    result = _CheckpointPreflightResult(checkpoint)
    assert result.summary == {
        "scientific_evidence": False,
        "result_scope": "preregistered_capability_preflight",
    }
    prompts_after_execution = tuple(provider.prompts)
    exactness = verify_closed_loop_preflight_checkpoint(checkpoint)
    assert exactness == case.exactness["exactness"]
    assert tuple(provider.prompts) == prompts_after_execution
    assert exactness["provider_calls_during_verification"] == 0
    assert all(
        row["provider"] == "openai"
        and row["served_model"] == SERVED_MODEL
        and row["usage"]["cost_usd"] == pytest.approx(0.0001)
        for row in checkpoint.payload["provider_calls"]
    )

    receipt_path = case.source_paths["receipt"].relative_to(
        case.repo_root
    ).as_posix()
    verified = verify_observed_p95_authority_receipt(
        receipt_path,
        repo_root=case.repo_root,
        expected_git_commit=case.paid.head_commit,
    )
    assert verified == case.receipt_binding["reservations"]
    assert case.receipt["scientific_evidence"] is False
    assert case.receipt["evidence_use"] == (
        "source-backed dispatch authority; not a scientific result"
    )
    assert case.receipt["reservations"][RUNTIME_MODEL][
        "action"
    ]["reservation"]["sample_count"] == 36
    assert case.receipt["reservations"][RUNTIME_MODEL][
        "semantic"
    ]["reservation"]["sample_count"] == 10
    for call_kind in ("action", "semantic"):
        authority = case.reservations[RUNTIME_MODEL][call_kind]["authority"]
        assert authority["source_authority_receipt_path"] == receipt_path
        assert authority["source_authority_receipt_file_sha256"] == (
            case.receipt_binding["receipt_file_sha256"]
        )
        assert authority["source_authority_receipt_content_sha256"] == (
            case.receipt_binding["receipt_content_sha256"]
        )
        assert authority["source_release_commit"] == case.paid.head_commit

    projection, projection_path = _load_verified_projection(
        contract,
        "gpt52_main",
        raw_root=case.raw_root,
        paid=case.paid,
    )
    assert projection_path == case.source_paths["projection"]
    assert projection["projection"] == {
        f"{SERVED_MODEL}::{call_kind}": verified[RUNTIME_MODEL][
            call_kind
        ]["reservation"]
        for call_kind in ("action", "semantic")
    }
    assert _runner_p95_reservations(
        contract,
        "gpt52_main",
        raw_root=case.raw_root,
        paid=case.paid,
    ) == case.reservations
    persisted_path, persisted_binding = (
        _persist_observed_p95_authority_receipt(
            contract,
            "gpt52_main",
            raw_root=case.raw_root,
            paid=case.paid,
        )
    )
    assert persisted_path == case.source_paths["receipt"]
    assert persisted_binding == case.receipt_binding

    real_parent = case.raw_root / "guarded-real-parent"
    real_parent.mkdir()
    (real_parent / "source.json").write_text("{}", encoding="utf-8")
    linked_parent = case.raw_root / "guarded-linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    linked_source = (linked_parent / "source.json").relative_to(
        case.repo_root
    ).as_posix()
    with pytest.raises(
        ObservedP95AuthorityError,
        match="cannot be opened safely",
    ):
        _read_regular_bytes(
            case.repo_root,
            PurePosixPath(linked_source),
            name="parent-symlink fixture",
        )
