from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from llm_providers import MultiModelLLM
from verified_memory.artifacts import (
    ArtifactValidationError,
    RunArtifactWriter,
    verify_manifest,
)
from verified_memory.budget import BudgetLimits, RunBudget
from verified_memory.m3_semantic import VerifiedRule
from verified_memory.prompts import compose_decision_prompt
from verified_memory.runner import (
    ShockEvent,
    VerifiedRunConfig,
    run_verified_experiment,
)
from verified_memory.runner_artifacts import (
    PREVIOUS_RUNNER_SCHEMA_VERSION,
    load_verified_run_artifacts,
    verified_run_schemas,
    write_verified_run_artifacts,
)
from verified_memory.scripted_provider import ScriptedDiagnosticProvider


ROOT = Path(__file__).resolve().parents[1]


def _diagnostic_result(
    *,
    run_id: str = "artifact-diagnostic",
    episode_length: int = 5,
    enable_semantic: bool = True,
    provider=None,
    **config_overrides,
):
    return run_verified_experiment(
        VerifiedRunConfig(
            run_id=run_id,
            episode_length=episode_length,
            enable_semantic=enable_semantic,
            **config_overrides,
        ),
        llm=MultiModelLLM(
            provider or ScriptedDiagnosticProvider(),
            num_workers=2,
        ),
        budget=RunBudget(BudgetLimits(max_calls=20, max_cost_usd=0.01)),
        env_config_source=ROOT / "config.yaml",
    )


class _VersionedScriptedProvider(ScriptedDiagnosticProvider):
    """Reject v1 with one support, then create v2 from later support."""

    @staticmethod
    def _proposal(prompt: str) -> str:
        payload = json.loads(ScriptedDiagnosticProvider._proposal(prompt))
        evidence = json.loads(prompt.split("Evidence:\n", 1)[1])
        latest_decision_t = max(
            int(str(item["episode_id"]).rsplit(":t", 1)[1])
            for item in evidence
        )
        if latest_decision_t <= 2:
            payload["supporting_episode_ids"] = [evidence[0]["episode_id"]]
        else:
            payload["supporting_episode_ids"] = [
                item["episode_id"] for item in evidence[-2:]
            ]
        return json.dumps(payload, sort_keys=True)


class _ScopedScriptedProvider(ScriptedDiagnosticProvider):
    @staticmethod
    def _proposal(prompt: str) -> str:
        payload = json.loads(ScriptedDiagnosticProvider._proposal(prompt))
        payload["context_scope"] = {
            "scope_id": "nonnegative-wealth",
            "predicates": [
                {
                    "field": "wealth",
                    "operator": ">=",
                    "value": 0.0,
                    "tolerance": 0.0,
                }
            ],
        }
        return json.dumps(payload, sort_keys=True)


class _IrrelevantScriptedProvider(ScriptedDiagnosticProvider):
    @staticmethod
    def _proposal(prompt: str) -> str:
        payload = json.loads(ScriptedDiagnosticProvider._proposal(prompt))
        payload["condition"] = {
            "field": "inflation",
            "operator": ">",
            "value": 999.0,
            "tolerance": 0.0,
        }
        return json.dumps(payload, sort_keys=True)


class _StateBoundScriptedProvider(ScriptedDiagnosticProvider):
    @staticmethod
    def _proposal(prompt: str) -> str:
        payload = json.loads(ScriptedDiagnosticProvider._proposal(prompt))
        payload["condition"] = {
            "field": "wealth",
            "operator": "<=",
            "value": 12000.0,
            "tolerance": 0.0,
        }
        return json.dumps(payload, sort_keys=True)


def _versioned_diagnostic_result():
    return _diagnostic_result(
        run_id="versioned-artifact-diagnostic",
        episode_length=6,
        provider=_VersionedScriptedProvider(),
        max_rule_proposals_per_agent=2,
    )


def _recompute_rule_key(rule: dict) -> str:
    payload = {
        "context_scope": rule["context_scope"],
        "condition": rule["condition"],
        "action_guidance": rule["action_guidance"],
        "outcome_criterion": rule["outcome_criterion"],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"rule-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:20]}"


def _rehash_episode(episode: dict) -> dict:
    result = json.loads(json.dumps(episode))
    payload = {key: value for key, value in result.items() if key != "record_hash"}
    result["record_hash"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return result


def _rehash_semantic_event(event: dict) -> dict:
    result = json.loads(json.dumps(event))
    event_index = int(str(result["event_id"]).split("-", 2)[1])
    payload = {
        key: value
        for key, value in result.items()
        if key not in {"agent_id", "event_id", "schema_version"}
    }
    payload["index"] = event_index
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    result["event_id"] = f"rle-{event_index:06d}-{digest[:12]}"
    return result


def test_diagnostic_result_is_sealed_and_reverified(tmp_path: Path) -> None:
    result = _diagnostic_result()
    run_dir = tmp_path / "run"
    manifest_path = write_verified_run_artifacts(
        run_dir,
        result,
        provenance={"purpose": "unit diagnostic", "scientific_evidence": False},
        git_commit="test-commit",
        git_dirty=True,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["result"]["complete"] is True
    assert manifest["validation_status"]["status"] == "pass"
    assert verify_manifest(run_dir).valid is True
    assert (run_dir / "streams/actions.jsonl").read_text().count("\n") == 10
    sealed_config = json.loads((run_dir / "config.json").read_text())
    assert sealed_config["foundation_env"]["n_agents"] == 2
    assert len(sealed_config["foundation_env_hash"]) == 64


@pytest.mark.parametrize(
    ("target", "replacement", "rehash", "error_match"),
    (
        ("hash", 99, False, "foundation_env_hash"),
        ("n_agents", 99, True, r"foundation_env\.n_agents"),
        ("episode_length", 99, True, r"foundation_env\.episode_length"),
        ("labor_step", 16.0, True, r"SimpleLabor\.labor_step"),
        ("num_labor_hours", 160.0, True, r"SimpleLabor\.num_labor_hours"),
        (
            "consumption_rate_step",
            0.05,
            True,
            r"SimpleConsumption\.consumption_rate_step",
        ),
    ),
)
def test_writer_rejects_forged_foundation_environment_contract(
    tmp_path: Path,
    target: str,
    replacement: float,
    rehash: bool,
    error_match: str,
) -> None:
    result = _diagnostic_result(
        run_id=f"forged-foundation-{target}",
        episode_length=2,
        enable_semantic=False,
    )
    config = json.loads(json.dumps(result.config))
    foundation_env = config["foundation_env"]
    if target == "hash":
        foundation_env["dense_log_frequency"] = replacement
    elif target in {"n_agents", "episode_length"}:
        foundation_env[target] = replacement
    else:
        component_name = (
            "SimpleConsumption"
            if target == "consumption_rate_step"
            else "SimpleLabor"
        )
        component = next(
            item[component_name]
            for item in foundation_env["components"]
            if component_name in item
        )
        component[target] = replacement
    if rehash:
        config["foundation_env_hash"] = hashlib.sha256(
            json.dumps(
                foundation_env,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    with pytest.raises(ArtifactValidationError, match=error_match):
        write_verified_run_artifacts(
            tmp_path / f"forged-foundation-{target}",
            replace(result, config=config),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


@pytest.mark.parametrize(("row_index", "forged_done"), ((0, True), (-1, False)))
def test_writer_rejects_forged_complete_run_done_sequence(
    tmp_path: Path,
    row_index: int,
    forged_done: bool,
) -> None:
    result = _diagnostic_result(
        run_id=f"forged-done-{row_index}",
        episode_length=3,
        enable_semantic=False,
    )
    macro_steps = list(result.stream("macro_steps"))
    macro_steps[row_index] = {**macro_steps[row_index], "done": forged_done}
    records = {**result.records, "macro_steps": tuple(macro_steps)}

    with pytest.raises(
        ArtifactValidationError,
        match="complete fixed-duration macro_steps",
    ):
        write_verified_run_artifacts(
            tmp_path / f"forged-done-{row_index}",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_fully_rehashed_ineligible_rule_selection(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(
        run_id="forged-ineligible-selection",
        episode_length=6,
        provider=_StateBoundScriptedProvider(),
    )
    rules = list(result.stream("semantic_rules"))
    rule_row = next(row for row in rules if row["agent_id"] == 0)
    rule = VerifiedRule.from_dict(
        {key: value for key, value in rule_row.items() if key != "agent_id"}
    )
    episodes = list(result.stream("episodes"))
    episode_index = next(
        index
        for index, row in enumerate(episodes)
        if row["agent_id"] == 0
        and not row["selected_rule_ids"]
        and row["decision_t"] >= rule.created_at
        and not rule.condition.matches(row["pre_state"])
    )
    target_episode = json.loads(json.dumps(episodes[episode_index]))
    decision_t = target_episode["decision_t"]

    events = list(result.stream("semantic_rule_events"))
    marker_index = next(
        index
        for index, row in enumerate(events)
        if row["agent_id"] == 0
        and row["timestamp"] == decision_t
        and row["event_type"] == "active_rule_retrieval_empty"
    )
    historical = next(
        row
        for row in reversed(events[:marker_index])
        if row["agent_id"] == 0
        and row["rule_id"] == rule.rule_id
        and row["metrics"]
    )
    forged_retrieval = {
        **events[marker_index],
        "event_type": "active_rule_retrieved",
        "rule_id": rule.rule_id,
        "candidate_id": None,
        "from_status": "active",
        "to_status": "active",
        "episode_ids": [],
        "reason": "active rule condition matched current state",
        "metrics": json.loads(json.dumps(historical["metrics"])),
        "provenance": {},
    }
    events[marker_index] = _rehash_semantic_event(forged_retrieval)

    actions = list(result.stream("actions"))
    action_index = next(
        index
        for index, row in enumerate(actions)
        if row["agent_id"] == 0 and row["decision_t"] == decision_t
    )
    actions[action_index] = {
        **actions[action_index],
        "selected_rule_ids": [rule.rule_id],
    }
    traces = list(result.stream("context_trace"))
    trace_index = next(
        index
        for index, row in enumerate(traces)
        if row["agent_id"] == 0 and row["decision_t"] == decision_t
    )
    trace = json.loads(json.dumps(traces[trace_index]))
    trace["selected_rule_ids"] = [rule.rule_id]

    target_episode["selected_rule_ids"] = [rule.rule_id]
    episodes[episode_index] = _rehash_episode(target_episode)
    snapshots = list(result.stream("decision_snapshots"))
    snapshot_index = next(
        index
        for index, row in enumerate(snapshots)
        if row["agent_id"] == 0 and row["decision_t"] == decision_t
    )
    snapshot = json.loads(json.dumps(snapshots[snapshot_index]))
    prompt_rule = replace(
        rule,
        confidence=float(forged_retrieval["metrics"]["confidence"]),
    )
    rule_text = f"Verified active rules: - {prompt_rule.to_prompt_text()}"
    memory_text = " ".join(
        part for part in (snapshot["memory_text"], rule_text) if part
    )
    prompt = compose_decision_prompt(snapshot["base_prompt"], memory_text)
    snapshot.update(
        {
            "memory_text": memory_text,
            "memory_hash": prompt.memory_hash,
            "full_prompt_hash": prompt.full_prompt_hash,
        }
    )
    snapshots[snapshot_index] = snapshot
    actions[action_index] = {
        **actions[action_index],
        "prompt_hash": prompt.full_prompt_hash,
    }

    api_usage = list(result.stream("api_usage"))
    api_index = next(
        index
        for index, row in enumerate(api_usage)
        if row["call_kind"] == "action"
        and row["agent_id"] == 0
        and row["decision_t"] == decision_t
    )
    api_usage[api_index] = {
        **api_usage[api_index],
        "prompt_hash": prompt.full_prompt_hash,
    }
    protected_context = snapshot["protected_context_text"]
    combined_prompt = " ".join(
        part
        for part in (
            (
                f"Causal context summary: {protected_context}"
                if protected_context
                else ""
            ),
            memory_text,
        )
        if part
    )
    trace.update(
        {
            "memory_prompt_hash": hashlib.sha256(
                memory_text.encode("utf-8")
            ).hexdigest(),
            "combined_prompt_hash": hashlib.sha256(
                combined_prompt.encode("utf-8")
            ).hexdigest(),
        }
    )
    bundle_payload = {
        "decision_t": decision_t,
        "context_id": trace["context_id"],
        "context_mode": trace["context_mode"],
        "episode_ids": trace["retrieved_episode_ids"],
        "episode_scores": trace["retrieval_scores"],
        "rule_ids": trace["selected_rule_ids"],
        "protected_context_prompt": protected_context,
        "memory_prompt": memory_text,
        "prompt": combined_prompt,
    }
    trace["memory_bundle_hash"] = hashlib.sha256(
        json.dumps(
            bundle_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    traces[trace_index] = trace

    summary = json.loads(json.dumps(result.summary))
    summary["memory_diagnostics"]["active_rule_retrieval_count"] += 1
    records = {
        **result.records,
        "actions": tuple(actions),
        "api_usage": tuple(api_usage),
        "context_trace": tuple(traces),
        "decision_snapshots": tuple(snapshots),
        "episodes": tuple(episodes),
        "semantic_rule_events": tuple(events),
    }
    with pytest.raises(
        ArtifactValidationError,
        match="exact M3 eligibility/ranking query",
    ):
        write_verified_run_artifacts(
            tmp_path / "forged-ineligible-selection",
            replace(result, records=records, summary=summary),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_empty_retrieval_marker_with_forged_state_hash(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(
        run_id="forged-empty-marker-state",
        episode_length=2,
    )
    events = list(result.stream("semantic_rule_events"))
    marker_index = next(
        index
        for index, row in enumerate(events)
        if row["agent_id"] == 0
        and row["event_type"] == "active_rule_retrieval_empty"
    )
    marker = json.loads(json.dumps(events[marker_index]))
    marker["provenance"]["state_hash"] = "0" * 64
    events[marker_index] = _rehash_semantic_event(marker)
    records = {**result.records, "semantic_rule_events": tuple(events)}

    with pytest.raises(
        ArtifactValidationError,
        match="empty M3 retrieval marker.*decision state/budget",
    ):
        write_verified_run_artifacts(
            tmp_path / "forged-empty-marker-state",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_semantic_summary_stream_counter_mismatch(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(run_id="forged-semantic-summary")
    summary = json.loads(json.dumps(result.summary))
    summary["memory_diagnostics"]["semantic_rule_status_counts"]["active"] += 7
    forged = replace(result, summary=summary)

    run_dir = tmp_path / "forged-semantic-summary"
    with pytest.raises(
        ArtifactValidationError,
        match=r"semantic_rule_status_counts\.active mismatch",
    ):
        write_verified_run_artifacts(
            run_dir,
            forged,
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )
    assert not run_dir.exists()


def test_writer_rejects_forged_legacy_result_schema(tmp_path: Path) -> None:
    result = _diagnostic_result(run_id="forged-legacy-write")
    config = {**result.config, "schema_version": "verified-simulation-runner-v1"}
    summary = {**result.summary, "schema_version": "verified-simulation-runner-v1"}
    forged = replace(result, config=config, summary=summary)

    run_dir = tmp_path / "forged-legacy-write"
    with pytest.raises(
        ArtifactValidationError,
        match="new writes require exact current config schema",
    ):
        write_verified_run_artifacts(
            run_dir,
            forged,
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )
    assert not run_dir.exists()


def test_writer_rejects_legacy_m3_rows_inside_current_result(tmp_path: Path) -> None:
    result = _diagnostic_result(run_id="forged-legacy-m3-row")
    assert result.stream("semantic_rules")
    records = dict(result.records)
    records["semantic_rules"] = tuple(
        {**row, "schema_version": "m3-verified-rule-v1"}
        for row in result.stream("semantic_rules")
    )
    forged = replace(result, records=records)

    run_dir = tmp_path / "forged-legacy-m3-row"
    with pytest.raises(
        ArtifactValidationError,
        match=r"semantic_rules\[0\]\.schema_version must be 'm3-verified-rule-v2'",
    ):
        write_verified_run_artifacts(
            run_dir,
            forged,
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )
    assert not run_dir.exists()


def test_writer_rejects_duplicate_decision_agent_identity(tmp_path: Path) -> None:
    result = _diagnostic_result(run_id="duplicate-action-identity")
    actions = list(result.stream("actions"))
    actions[1] = actions[0]
    records = {**result.records, "actions": tuple(actions)}

    with pytest.raises(ArtifactValidationError, match="actions identity grid mismatch"):
        write_verified_run_artifacts(
            tmp_path / "duplicate-action-identity",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_action_prompt_hash_not_bound_to_snapshot(tmp_path: Path) -> None:
    result = _diagnostic_result(run_id="forged-action-prompt-hash")
    actions = list(result.stream("actions"))
    actions[0] = {**actions[0], "prompt_hash": "0" * 64}
    records = {**result.records, "actions": tuple(actions)}

    with pytest.raises(ArtifactValidationError, match="prompt hash is not bound"):
        write_verified_run_artifacts(
            tmp_path / "forged-action-prompt-hash",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_context_marker_in_retrieval_only_base_prompt(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(run_id="forged-prompt-route")
    snapshots = list(result.stream("decision_snapshots"))
    actions = list(result.stream("actions"))
    api_usage = list(result.stream("api_usage"))
    snapshot = snapshots[0]
    forged_base = snapshot["base_prompt"] + " Causal context summary: forged."
    forged_prompt = compose_decision_prompt(forged_base, snapshot["memory_text"])
    snapshots[0] = {
        **snapshot,
        "base_prompt": forged_prompt.base_prompt,
        "base_prompt_hash": forged_prompt.base_prompt_hash,
        "memory_hash": forged_prompt.memory_hash,
        "full_prompt_hash": forged_prompt.full_prompt_hash,
    }
    action = actions[0]
    actions[0] = {**action, "prompt_hash": forged_prompt.full_prompt_hash}
    api_index = next(
        index
        for index, row in enumerate(api_usage)
        if row["call_kind"] == "action"
        and row["decision_t"] == action["decision_t"]
        and row["agent_id"] == action["agent_id"]
    )
    api_usage[api_index] = {
        **api_usage[api_index],
        "prompt_hash": forged_prompt.full_prompt_hash,
    }
    records = {
        **result.records,
        "actions": tuple(actions),
        "api_usage": tuple(api_usage),
        "decision_snapshots": tuple(snapshots),
    }

    with pytest.raises(
        ArtifactValidationError,
        match="non-prompt context route contains a causal context summary marker",
    ):
        write_verified_run_artifacts(
            tmp_path / "forged-prompt-route",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_episode_mutation_with_stale_record_hash(tmp_path: Path) -> None:
    result = _diagnostic_result(run_id="stale-episode-record-hash")
    episodes = list(result.stream("episodes"))
    episodes[1] = {**episodes[1], "context_id": "ctx-forged"}
    records = {**result.records, "episodes": tuple(episodes)}

    with pytest.raises(ArtifactValidationError, match="record_hash/integrity"):
        write_verified_run_artifacts(
            tmp_path / "stale-episode-record-hash",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_context_retrieval_ids_not_bound_to_episode(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(run_id="forged-context-retrieval")
    traces = list(result.stream("context_trace"))
    target = next(
        index for index, row in enumerate(traces) if row["retrieved_episode_ids"]
    )
    forged_ids = list(traces[target]["retrieved_episode_ids"])
    forged_ids[0] = "forged-episode-id"
    traces[target] = {**traces[target], "retrieved_episode_ids": forged_ids}
    records = {**result.records, "context_trace": tuple(traces)}

    with pytest.raises(ArtifactValidationError, match="retrieved/selected IDs are not bound"):
        write_verified_run_artifacts(
            tmp_path / "forged-context-retrieval",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_snapshot_full_prompt_hash_not_reconstructible(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(run_id="forged-full-prompt-hash")
    snapshots = list(result.stream("decision_snapshots"))
    snapshots[0] = {**snapshots[0], "full_prompt_hash": "f" * 64}
    records = {**result.records, "decision_snapshots": tuple(snapshots)}

    with pytest.raises(ArtifactValidationError, match="prompt/hash fields do not reconstruct"):
        write_verified_run_artifacts(
            tmp_path / "forged-full-prompt-hash",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_rehashed_episode_action_not_bound_to_action_row(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(run_id="forged-episode-action")
    episodes = list(result.stream("episodes"))
    episode = json.loads(json.dumps(episodes[0]))
    episode["proposed_action"]["work_propensity"] = 1.0
    integrity_payload = {key: value for key, value in episode.items() if key != "record_hash"}
    episode["record_hash"] = hashlib.sha256(
        json.dumps(
            integrity_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    episodes[0] = episode
    records = {**result.records, "episodes": tuple(episodes)}

    with pytest.raises(
        ArtifactValidationError,
        match="episode proposed/executed action or reflection does not match action row",
    ):
        write_verified_run_artifacts(
            tmp_path / "forged-episode-action",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_budget_summary_stable_field_mismatch(tmp_path: Path) -> None:
    result = _diagnostic_result(run_id="forged-budget-summary")
    summary = json.loads(json.dumps(result.summary))
    summary["api"]["active_calls"] += 123

    with pytest.raises(
        ArtifactValidationError,
        match="summary.api and budget_snapshot stable fields differ",
    ):
        write_verified_run_artifacts(
            tmp_path / "forged-budget-summary",
            replace(result, summary=summary),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_final_rule_status_without_lifecycle_event(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(run_id="forged-final-rule-status")
    rules = tuple({**row, "status": "rejected"} for row in result.stream("semantic_rules"))
    summary = json.loads(json.dumps(result.summary))
    counts = summary["memory_diagnostics"]["semantic_rule_status_counts"]
    counts["rejected"] = counts["active"]
    counts["active"] = 0
    records = {**result.records, "semantic_rules": rules}

    with pytest.raises(
        ArtifactValidationError,
        match="final semantic rule status does not match lifecycle events",
    ):
        write_verified_run_artifacts(
            tmp_path / "forged-final-rule-status",
            replace(result, records=records, summary=summary),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_successful_proposal_without_same_agent_rule(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(run_id="orphaned-successful-proposal")
    proposals = list(result.stream("semantic_proposals"))
    proposals[0] = {**proposals[0], "rule_id": "missing-rule"}
    records = {**result.records, "semantic_proposals": tuple(proposals)}

    with pytest.raises(
        ArtifactValidationError,
        match="successful semantic proposal must link to a same-agent rule",
    ):
        write_verified_run_artifacts(
            tmp_path / "orphaned-successful-proposal",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_semantic_event_with_stale_content_id(tmp_path: Path) -> None:
    result = _diagnostic_result(run_id="stale-semantic-event-id")
    events = list(result.stream("semantic_rule_events"))
    events[0] = {**events[0], "reason": "forged reason with stale event ID"}
    records = {**result.records, "semantic_rule_events": tuple(events)}

    with pytest.raises(ArtifactValidationError, match="event ledger hash mismatch"):
        write_verified_run_artifacts(
            tmp_path / "stale-semantic-event-id",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_semantic_rule_with_stale_identity_hash(tmp_path: Path) -> None:
    result = _diagnostic_result(run_id="stale-semantic-rule-key")
    rules = list(result.stream("semantic_rules"))
    condition = {**rules[0]["condition"], "value": 123.0}
    rules[0] = {**rules[0], "condition": condition}
    records = {**result.records, "semantic_rules": tuple(rules)}

    with pytest.raises(ArtifactValidationError, match="inconsistent rule_key"):
        write_verified_run_artifacts(
            tmp_path / "stale-semantic-rule-key",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_reparses_semantic_raw_even_when_receipt_hashes_are_reforged(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(run_id="reforged-semantic-raw")
    proposals = list(result.stream("semantic_proposals"))
    api_usage = list(result.stream("api_usage"))
    proposal = proposals[0]
    malformed = "malformed semantic candidate"
    malformed_hash = hashlib.sha256(malformed.encode("utf-8")).hexdigest()
    proposals[0] = {
        **proposal,
        "raw_output": malformed,
        "raw_output_hash": malformed_hash,
    }
    api_index = next(
        index
        for index, row in enumerate(api_usage)
        if row["call_kind"] == "semantic"
        and row["decision_t"] == proposal["current_t"]
        and row["agent_id"] == proposal["agent_id"]
    )
    api_usage[api_index] = {**api_usage[api_index], "raw_output_hash": malformed_hash}
    records = {
        **result.records,
        "semantic_proposals": tuple(proposals),
        "api_usage": tuple(api_usage),
    }

    with pytest.raises(
        ArtifactValidationError,
        match="semantic proposal parse status/error does not reproduce",
    ):
        write_verified_run_artifacts(
            tmp_path / "reforged-semantic-raw",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_non_injected_rule_without_creation_candidate_id(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(run_id="missing-creation-candidate-id")
    rules = list(result.stream("semantic_rules"))
    rules[0] = {**rules[0], "candidate_ids": []}
    records = {**result.records, "semantic_rules": tuple(rules)}

    with pytest.raises(
        ArtifactValidationError,
        match="exactly one creation candidate ID",
    ):
        write_verified_run_artifacts(
            tmp_path / "missing-creation-candidate-id",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_rule_not_bound_to_reparsed_creation_candidate(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(run_id="unbound-creation-candidate")
    rules = list(result.stream("semantic_rules"))
    rules[0] = {
        **rules[0],
        "rationale": "forged rationale absent from the raw creation proposal",
    }
    records = {**result.records, "semantic_rules": tuple(rules)}

    with pytest.raises(
        ArtifactValidationError,
        match="not exactly bound to its reparsed creation candidate",
    ):
        write_verified_run_artifacts(
            tmp_path / "unbound-creation-candidate",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_versioned_rule_artifact_satisfies_terminal_parent_contract(
    tmp_path: Path,
) -> None:
    result = _versioned_diagnostic_result()
    rules = [
        row for row in result.stream("semantic_rules") if row["agent_id"] == 0
    ]
    assert [(row["rule_version"], row["status"]) for row in rules] == [
        (1, "rejected"),
        (2, "provisional"),
    ]

    run_dir = tmp_path / "valid-versioned-artifact"
    write_verified_run_artifacts(
        run_dir,
        result,
        provenance={"purpose": "versioned artifact contract"},
        git_commit="test-commit",
        git_dirty=True,
    )
    assert load_verified_run_artifacts(run_dir).stream("semantic_rules")


def test_writer_rejects_child_created_at_terminal_parent_update(
    tmp_path: Path,
) -> None:
    result = _versioned_diagnostic_result()
    rules = list(result.stream("semantic_rules"))
    parent_index = next(
        index
        for index, row in enumerate(rules)
        if row["agent_id"] == 0 and row["rule_version"] == 1
    )
    child = next(
        row
        for row in rules
        if row["agent_id"] == 0 and row["rule_version"] == 2
    )
    rules[parent_index] = {**rules[parent_index], "updated_at": child["created_at"]}
    records = {**result.records, "semantic_rules": tuple(rules)}

    with pytest.raises(
        ArtifactValidationError,
        match=(
            "updated_at is not bound to its lifecycle|must be created strictly "
            "after its terminal parent update"
        ),
    ):
        write_verified_run_artifacts(
            tmp_path / "noncausal-version-time",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_child_without_new_post_parent_support(
    tmp_path: Path,
) -> None:
    result = _versioned_diagnostic_result()
    rules = list(result.stream("semantic_rules"))
    child_index = next(
        index
        for index, row in enumerate(rules)
        if row["agent_id"] == 0 and row["rule_version"] == 2
    )
    parent = next(
        row
        for row in rules
        if row["agent_id"] == 0 and row["rule_version"] == 1
    )
    child = rules[child_index]
    old_support = [
        row["episode_id"]
        for row in result.stream("episodes")
        if row["agent_id"] == 0 and row["outcome_t"] <= parent["updated_at"]
    ][: len(child["supporting_episode_ids"])]
    assert len(old_support) == len(child["supporting_episode_ids"])
    rules[child_index] = {**child, "supporting_episode_ids": old_support}
    records = {**result.records, "semantic_rules": tuple(rules)}

    with pytest.raises(
        ArtifactValidationError,
        match="revision lacks genuinely new support",
    ):
        write_verified_run_artifacts(
            tmp_path / "reused-version-support",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


@pytest.mark.parametrize(
    ("component", "replacement"),
    (("condition", "<="), ("action_guidance", "at_least")),
)
def test_writer_rejects_family_collision_across_policy_operator_shape(
    tmp_path: Path,
    component: str,
    replacement: str,
) -> None:
    result = _diagnostic_result(run_id=f"family-shape-{component}")
    rules = list(result.stream("semantic_rules"))
    rule = json.loads(json.dumps(rules[0]))
    if component == "condition":
        rule["condition"]["operator"] = replacement
    else:
        rule["action_guidance"]["direction"] = replacement
    rule["rule_key"] = _recompute_rule_key(rule)
    rules[0] = rule
    records = {**result.records, "semantic_rules": tuple(rules)}

    with pytest.raises(ArtifactValidationError, match="inconsistent rule_family_id"):
        write_verified_run_artifacts(
            tmp_path / f"family-shape-{component}",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_family_collision_across_context_predicate_shape(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(
        run_id="family-context-predicate-shape",
        provider=_ScopedScriptedProvider(),
    )
    rules = list(result.stream("semantic_rules"))
    rule = json.loads(json.dumps(rules[0]))
    rule["context_scope"]["predicates"][0]["operator"] = "<="
    rule["rule_key"] = _recompute_rule_key(rule)
    rules[0] = rule
    records = {**result.records, "semantic_rules": tuple(rules)}

    with pytest.raises(ArtifactValidationError, match="inconsistent rule_family_id"):
        write_verified_run_artifacts(
            tmp_path / "family-context-predicate-shape",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_out_of_order_per_agent_episode_ledger(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(run_id="out-of-order-episode-ledger")
    episodes = list(result.stream("episodes"))
    first = next(
        index
        for index, row in enumerate(episodes)
        if row["agent_id"] == 0 and row["decision_t"] == 0
    )
    second = next(
        index
        for index, row in enumerate(episodes)
        if row["agent_id"] == 0 and row["decision_t"] == 1
    )
    episodes[first], episodes[second] = episodes[second], episodes[first]
    records = {**result.records, "episodes": tuple(episodes)}

    with pytest.raises(
        ArtifactValidationError,
        match="strictly increasing per-agent causal order",
    ):
        write_verified_run_artifacts(
            tmp_path / "out-of-order-episode-ledger",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


@pytest.mark.parametrize("field", ("utility_advantage", "importance"))
def test_writer_recomputes_causal_episode_utility_statistics(
    tmp_path: Path,
    field: str,
) -> None:
    result = _diagnostic_result(run_id=f"forged-episode-{field}")
    episodes = list(result.stream("episodes"))
    target = next(
        index
        for index, row in enumerate(episodes)
        if row["agent_id"] == 0 and row["decision_t"] == 1
    )
    forged = {**episodes[target], field: float(episodes[target][field]) + 0.25}
    episodes[target] = _rehash_episode(forged)
    records = {**result.records, "episodes": tuple(episodes)}

    with pytest.raises(ArtifactValidationError, match=field):
        write_verified_run_artifacts(
            tmp_path / f"forged-episode-{field}",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_rule_retrieval_before_activation_timestamp(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(run_id="preactivation-artifact-retrieval")
    events = list(result.stream("semantic_rule_events"))
    activation_index = next(
        index
        for index, row in enumerate(events)
        if row["agent_id"] == 0 and row["event_type"] == "rule_activated"
    )
    retrieval = next(
        row
        for row in events
        if row["agent_id"] == 0
        and row["event_type"] == "active_rule_retrieved"
    )
    activation = events[activation_index]
    assert activation["timestamp"] == retrieval["timestamp"]
    events[activation_index] = _rehash_semantic_event(
        {**activation, "timestamp": retrieval["timestamp"] + 1}
    )
    records = {**result.records, "semantic_rule_events": tuple(events)}

    with pytest.raises(
        ArtifactValidationError,
        match="retrieved before its unique activation|timestamps move backward",
    ):
        write_verified_run_artifacts(
            tmp_path / "preactivation-artifact-retrieval",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_recomputes_historical_rule_event_metrics(tmp_path: Path) -> None:
    result = _diagnostic_result(run_id="forged-historical-rule-metrics")
    events = list(result.stream("semantic_rule_events"))
    event_index = next(
        index
        for index, row in enumerate(events)
        if row["agent_id"] == 0 and row["event_type"] == "active_rule_retrieved"
    )
    forged = json.loads(json.dumps(events[event_index]))
    forged["metrics"]["confidence"] = 0.123
    events[event_index] = _rehash_semantic_event(forged)
    records = {**result.records, "semantic_rule_events": tuple(events)}

    with pytest.raises(
        ArtifactValidationError,
        match="evidence lifecycle does not reproduce.*event metrics",
    ):
        write_verified_run_artifacts(
            tmp_path / "forged-historical-rule-metrics",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_replays_exhaustive_creation_irrelevant_search(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(
        run_id="forged-creation-search",
        provider=_IrrelevantScriptedProvider(),
    )
    rules = list(result.stream("semantic_rules"))
    rule_index = next(
        index
        for index, row in enumerate(rules)
        if row["agent_id"] == 0 and row["irrelevant_episode_ids"]
    )
    target_rule = json.loads(json.dumps(rules[rule_index]))
    target_rule["irrelevant_episode_ids"] = target_rule[
        "irrelevant_episode_ids"
    ][1:]
    rules[rule_index] = target_rule

    events = list(result.stream("semantic_rule_events"))
    event_index = next(
        index
        for index, row in enumerate(events)
        if row["agent_id"] == 0
        and row["rule_id"] == target_rule["rule_id"]
        and row["event_type"] in {"candidate_verified", "candidate_rejected"}
    )
    for index, row in enumerate(events):
        if row["agent_id"] != 0 or row["rule_id"] != target_rule["rule_id"]:
            continue
        forged_event = json.loads(json.dumps(row))
        forged_event["metrics"]["evidence_type_counts"]["irrelevant"] -= 1
        events[index] = _rehash_semantic_event(forged_event)
    records = {
        **result.records,
        "semantic_rules": tuple(rules),
        "semantic_rule_events": tuple(events),
    }

    with pytest.raises(
        ArtifactValidationError,
        match="creation evidence search does not reproduce",
    ):
        write_verified_run_artifacts(
            tmp_path / "forged-creation-search",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_recomputes_semantic_harmful_failure_streak(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(run_id="forged-semantic-failure-streak")
    rules = list(result.stream("semantic_rules"))
    rules[0] = {**rules[0], "consecutive_failures": 1}
    records = {**result.records, "semantic_rules": tuple(rules)}

    with pytest.raises(
        ArtifactValidationError,
        match="evidence lifecycle does not reproduce",
    ):
        write_verified_run_artifacts(
            tmp_path / "forged-semantic-failure-streak",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_rejects_observed_rate_forged_into_t0_missing_state(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(run_id="forged-t0-rate")
    episodes = list(result.stream("episodes"))
    episode_index = next(
        index
        for index, row in enumerate(episodes)
        if row["agent_id"] == 0 and row["decision_t"] == 0
    )
    episode = json.loads(json.dumps(episodes[episode_index]))
    episode["pre_state"].update(
        {"low_labor_rate": 1.0, "unemployment_rate": 1.0}
    )
    episodes[episode_index] = _rehash_episode(episode)
    snapshots = list(result.stream("decision_snapshots"))
    snapshot_index = next(
        index
        for index, row in enumerate(snapshots)
        if row["agent_id"] == 0 and row["decision_t"] == 0
    )
    snapshots[snapshot_index] = {
        **snapshots[snapshot_index],
        "environment_state_hash": hashlib.sha256(
            json.dumps(
                episode["pre_state"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    records = {
        **result.records,
        "episodes": tuple(episodes),
        "decision_snapshots": tuple(snapshots),
    }

    with pytest.raises(
        ArtifactValidationError,
        match="must be absent when its availability mask is 0",
    ):
        write_verified_run_artifacts(
            tmp_path / "forged-t0-rate",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


@pytest.mark.parametrize("field", ("flow_utility", "budget_residual"))
def test_writer_reconstructs_utility_ledger_rows(
    tmp_path: Path,
    field: str,
) -> None:
    result = _diagnostic_result(run_id=f"forged-ledger-{field}")
    rows = list(result.stream("utility_ledger"))
    rows[0] = {**rows[0], field: 999999.0}
    records = {**result.records, "utility_ledger": tuple(rows)}

    with pytest.raises(
        ArtifactValidationError,
        match="authoritative accounting",
    ):
        write_verified_run_artifacts(
            tmp_path / f"forged-ledger-{field}",
            replace(result, records=records),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_recomputes_headline_final_metrics(tmp_path: Path) -> None:
    result = _diagnostic_result(run_id="forged-final-metrics")
    summary = json.loads(json.dumps(result.summary))
    summary["final_metrics"] = {
        metric: 999999.0 for metric in summary["final_metrics"]
    }

    with pytest.raises(
        ArtifactValidationError,
        match=r"summary\.final_metrics\.",
    ):
        write_verified_run_artifacts(
            tmp_path / "forged-final-metrics",
            replace(result, summary=summary),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_recomputes_validation_status_from_streams(tmp_path: Path) -> None:
    result = _diagnostic_result(run_id="forged-validation-status")
    validation = json.loads(json.dumps(result.validation_status))
    validation["checks"]["budget_identity"] = False
    validation["status"] = "fail"
    summary = json.loads(json.dumps(result.summary))
    summary["validation"] = validation

    with pytest.raises(
        ArtifactValidationError,
        match="validation_status does not reproduce",
    ):
        write_verified_run_artifacts(
            tmp_path / "forged-validation-status",
            replace(
                result,
                summary=summary,
                validation_status=validation,
            ),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_writer_recomputes_semantic_candidate_parse_mode(tmp_path: Path) -> None:
    result = _diagnostic_result(run_id="forged-candidate-parse-mode")
    proposals = list(result.stream("semantic_proposals"))
    proposals[0] = {
        **proposals[0],
        "candidate_parse_mode": "substring_recovery",
    }
    records = {**result.records, "semantic_proposals": tuple(proposals)}
    summary = json.loads(json.dumps(result.summary))
    mode_counts = summary["memory_diagnostics"]["semantic_candidate_parse"][
        "mode_counts"
    ]
    mode_counts["exact_json"] -= 1
    mode_counts["substring_recovery"] += 1

    with pytest.raises(
        ArtifactValidationError,
        match="candidate_parse_mode does not reproduce",
    ):
        write_verified_run_artifacts(
            tmp_path / "forged-candidate-parse-mode",
            replace(result, records=records, summary=summary),
            provenance={"purpose": "adversarial test"},
            git_commit="test-commit",
            git_dirty=True,
        )


def test_loader_rejects_resealed_legacy_semantic_counter_mismatch(
    tmp_path: Path,
) -> None:
    """A hash-consistent v1 clone still cannot lie about semantic streams."""

    source = ROOT / "artifacts" / "verified_runs" / "g4b-api-gpt52-s11"
    forged = tmp_path / "forged-legacy-load"
    shutil.copytree(source, forged)

    summary_path = forged / "streams" / "summary.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["memory_diagnostics"]["semantic_rule_status_counts"]["active"] += 1
    summary_bytes = (
        json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    summary_path.chmod(summary_path.stat().st_mode | 0o200)
    summary_path.write_bytes(summary_bytes)

    manifest_path = forged / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_entry = next(
        row
        for row in manifest["artifacts"]
        if row["path"] == "streams/summary.jsonl"
    )
    summary_entry.update(
        {
            "line_count": 1,
            "byte_size": len(summary_bytes),
            "sha256": hashlib.sha256(summary_bytes).hexdigest(),
        }
    )
    manifest_bytes = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    manifest_path.chmod(manifest_path.stat().st_mode | 0o200)
    manifest_path.write_bytes(manifest_bytes)

    assert verify_manifest(forged).valid is True
    with pytest.raises(
        ArtifactValidationError,
        match=r"semantic_rule_status_counts\.active mismatch",
    ):
        load_verified_run_artifacts(forged)


def test_real_legacy_g4b_fixture_survives_reconciliation() -> None:
    loaded = load_verified_run_artifacts(
        ROOT / "artifacts" / "verified_runs" / "g4b-api-gpt52-s11"
    )
    assert loaded.config["schema_version"] == "verified-simulation-runner-v1"
    assert loaded.summary["memory_diagnostics"]["semantic_rule_status_counts"] == {
        "active": 2,
        "provisional": 0,
        "rejected": 0,
        "retired": 0,
    }


def test_hash_consistent_v2_clone_loads_without_v3_pilot_streams(
    tmp_path: Path,
) -> None:
    source = _diagnostic_result(
        run_id="v2-compatibility-clone", episode_length=4
    )
    config = json.loads(json.dumps(source.config))
    config["schema_version"] = PREVIOUS_RUNNER_SCHEMA_VERSION
    summary = json.loads(json.dumps(source.summary))
    summary["schema_version"] = PREVIOUS_RUNNER_SCHEMA_VERSION
    validation = json.loads(json.dumps(source.validation_status))
    for name in (
        "shock_schedule_applied_exactly",
        "no_future_shock_in_prompt",
        "proposal_freeze_respected",
        "error_rule_injection_accounted",
        "unverified_policy_has_no_evidence_or_retirement",
    ):
        validation["checks"].pop(name)
    validation["status"] = (
        "pass" if all(validation["checks"].values()) else "fail"
    )
    summary["validation"] = json.loads(json.dumps(validation))

    runner_streams = {
        "actions",
        "api_usage",
        "decision_snapshots",
        "macro_steps",
        "semantic_proposals",
        "errors",
    }
    records = {}
    for stream_name, rows in source.records.items():
        if stream_name in {"shock_events", "error_rule_injections"}:
            continue
        converted = []
        for row in rows:
            item = json.loads(json.dumps(row))
            if stream_name in runner_streams:
                item["schema_version"] = PREVIOUS_RUNNER_SCHEMA_VERSION
            if stream_name == "decision_snapshots":
                item["prompt_schema_version"] = "verified-decision-prompt-v2"
                item.pop("shock_event", None)
            converted.append(item)
        records[stream_name] = tuple(converted)

    run_dir = tmp_path / "v2-compatibility-clone"
    schemas = verified_run_schemas(
        semantic_required=True,
        run_schema_version=PREVIOUS_RUNNER_SCHEMA_VERSION,
    )
    writer = RunArtifactWriter.create(
        run_dir,
        schemas,
        config=config,
        provenance={"purpose": "v2 compatibility regression"},
        git_commit="test-v2",
        git_dirty=False,
    )
    for stream_name, rows in records.items():
        for row in rows:
            writer.append(stream_name, row)
    writer.append("summary", summary)
    writer.finalize(
        validation_status=validation,
        budget_snapshot=source.budget_snapshot,
        result_complete=True,
    )

    assert verify_manifest(run_dir).valid is True
    loaded = load_verified_run_artifacts(run_dir)
    assert loaded.config["schema_version"] == PREVIOUS_RUNNER_SCHEMA_VERSION
    assert loaded.records == records
    assert "shock_events" not in loaded.records
    assert "error_rule_injections" not in loaded.records


def test_semantic_disabled_zero_rule_run_seals_and_loads(tmp_path: Path) -> None:
    result = _diagnostic_result(
        run_id="semantic-disabled",
        episode_length=2,
        enable_semantic=False,
    )
    assert result.summary["memory_diagnostics"]["semantic_activation_observed"] is False

    run_dir = tmp_path / "semantic-disabled"
    write_verified_run_artifacts(
        run_dir,
        result,
        provenance={"purpose": "zero semantic rule contract"},
        git_commit="test-commit",
        git_dirty=True,
    )
    loaded = load_verified_run_artifacts(run_dir)
    assert loaded.stream("semantic_proposals") == ()
    assert loaded.stream("semantic_rule_events") == ()
    assert loaded.stream("semantic_rules") == ()


@pytest.mark.parametrize(
    ("case", "config"),
    [
        (
            "shock",
            {
                "episode_length": 2,
                "enable_semantic": False,
                "shock_schedule": (
                    ShockEvent(0, "shock", 0.08),
                    ShockEvent(1, "recovery", 0.03),
                ),
            },
        ),
        (
            "unverified",
            {
                "episode_length": 4,
                "semantic_policy": "unverified-immediate",
            },
        ),
        (
            "forced-error",
            {
                "episode_length": 4,
                "error_rule_mode": "forced-active",
                "error_rule_injection_t": 3,
                "freeze_new_proposals_after": 2,
            },
        ),
    ],
)
def test_v3_pilot_streams_seal_and_load(
    tmp_path: Path, case: str, config: dict
) -> None:
    result = _diagnostic_result(run_id=f"sealed-{case}", **config)
    run_dir = tmp_path / case
    write_verified_run_artifacts(
        run_dir,
        result,
        provenance={"purpose": f"{case} v3 seal regression"},
        git_commit="test-commit",
        git_dirty=True,
    )

    loaded = load_verified_run_artifacts(run_dir)
    assert loaded.records == result.records
    assert loaded.config["schema_version"] == "verified-simulation-runner-v3"
    if case == "shock":
        assert len(loaded.stream("shock_events")) == 2
    elif case == "unverified":
        assert all(
            row["semantic_policy"] == "unverified-immediate"
            and row["rule_status"] == "active"
            for row in loaded.stream("semantic_proposals")
        )
    else:
        assert all(
            row["mode"] == "forced-active"
            and row["verifier_bypassed"] is True
            for row in loaded.stream("error_rule_injections")
        )


def test_writer_rejects_rehashed_unverified_source_candidate_forgery(
    tmp_path: Path,
) -> None:
    result = _diagnostic_result(
        run_id="forged-unverified-source",
        episode_length=4,
        semantic_policy="unverified-immediate",
    )
    events = list(result.stream("semantic_rule_events"))
    index = next(
        position
        for position, row in enumerate(events)
        if row["event_type"] == "experimental_rule_injected_active"
    )
    forged_event = json.loads(json.dumps(events[index]))
    forged_event["provenance"]["source_candidate_id"] = (
        "cand-" + "0" * 20
    )
    events[index] = _rehash_semantic_event(forged_event)
    forged = replace(
        result,
        records={
            **result.records,
            "semantic_rule_events": tuple(events),
        },
    )

    with pytest.raises(
        ArtifactValidationError,
        match=(
            "unverified|content-addressed|injection provenance|"
            "causal activation event"
        ),
    ):
        write_verified_run_artifacts(
            tmp_path / "forged-unverified-source",
            forged,
            provenance={"purpose": "unverified provenance forgery"},
            git_commit="test-commit",
            git_dirty=True,
        )
