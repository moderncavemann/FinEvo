"""Build and execute paired memory interventions from a sealed verified run."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

from llm_providers import MultiModelLLM

from .artifacts import JsonField, JsonlStreamSchema, RunArtifactWriter, verify_manifest
from .budget import RunBudget
from .replay import (
    DecisionSnapshot,
    PairedReplayResult,
    PairedReplayRunner,
    ProviderCompletion,
)
from .runner import VerifiedRunResult, _estimate_usage


REPLAY_EXPERIMENT_SCHEMA_VERSION = "verified-replay-experiment-v1"


def _select_row(
    rows: tuple[Mapping[str, Any], ...], *, decision_t: int, agent_id: int
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("decision_t") == decision_t and row.get("agent_id") == agent_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one row for decision_t={decision_t}, agent_id={agent_id}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _shuffle_episode_order(memory_text: str) -> str:
    chunks = memory_text.split(" - ")
    if len(chunks) < 3:
        raise ValueError("matched memory has too few entries for shuffled treatment")
    header = chunks[0]
    episodes = [chunk for chunk in chunks[1:] if chunk.lstrip().startswith("[")]
    other = [chunk for chunk in chunks[1:] if not chunk.lstrip().startswith("[")]
    if len(episodes) < 2:
        raise ValueError("matched memory needs at least two episodic entries")
    shuffled = [header, *reversed(episodes), *other]
    result = " - ".join(shuffled)
    if result == memory_text:
        raise ValueError("shuffled treatment did not change memory bytes")
    return result


def build_paired_snapshot(
    source: VerifiedRunResult,
    *,
    decision_t: int,
    agent_id: int,
    provider: str,
    model: str,
    max_tokens: int = 220,
) -> DecisionSnapshot:
    """Construct five signed treatments without changing any protected field."""

    if not isinstance(source, VerifiedRunResult):
        raise TypeError("source must be VerifiedRunResult")
    action = _select_row(
        source.stream("actions"), decision_t=decision_t, agent_id=agent_id
    )
    if not action.get("selected_rule_ids"):
        raise ValueError("paired replay requires a decision that retrieved an active rule")
    snapshot = _select_row(
        source.stream("decision_snapshots"),
        decision_t=decision_t,
        agent_id=agent_id,
    )
    matched = str(snapshot["memory_text"])
    if not matched:
        raise ValueError("matched treatment must contain memory")
    wrong_candidates = [
        row
        for row in source.stream("decision_snapshots")
        if row.get("decision_t") == decision_t
        and row.get("agent_id") != agent_id
        and row.get("memory_text")
        and row.get("memory_hash") != snapshot.get("memory_hash")
    ]
    if not wrong_candidates:
        raise ValueError("no different-agent memory is available for wrong-context")
    wrong_context = str(wrong_candidates[0]["memory_text"])
    injected = (
        matched
        + " Experimental unverified rule: when interest_rate >= 0, decrease "
        "labor_hours toward 0. This sentence is an explicitly injected false-rule "
        "treatment and is not verified evidence."
    )
    return DecisionSnapshot.create(
        environment_state_hash=str(snapshot["environment_state_hash"]),
        base_prompt=str(snapshot["base_prompt"]),
        context_packet_id=str(snapshot["context_packet_id"]),
        context_packet_hash=str(snapshot["context_packet_hash"]),
        provider=provider,
        model=model,
        memory_bundles={
            "matched": matched,
            "no-memory": "",
            "shuffled": _shuffle_episode_order(matched),
            "wrong-context": wrong_context,
            "injected-rule": injected,
        },
        max_labor_hours=float(source.config["max_labor_hours"]),
        labor_step=float(source.config["labor_step"]),
        consumption_step=float(source.config["consumption_step"]),
        temperature=float(source.config["temperature"]),
        top_p=float(source.config["top_p"]),
        max_tokens=max_tokens,
        decoding_seed=int(source.config["seed"]),
    )


def run_paired_replay(
    snapshot: DecisionSnapshot,
    *,
    llm: MultiModelLLM,
    budget: RunBudget,
    max_retries: int = 2,
) -> PairedReplayResult:
    """Execute all signed treatments through one structured provider boundary."""

    full_name = llm.get_model_name()
    provider, _, model = full_name.partition("/")
    if (provider, model) != (snapshot.provider, snapshot.model):
        raise ValueError("snapshot provider/model does not match replay provider")

    def complete(request: Any) -> ProviderCompletion:
        result = llm.get_structured_completion(
            [{"role": "user", "content": request.prompt}],
            temperature=snapshot.temperature,
            max_tokens=snapshot.max_tokens,
            top_p=snapshot.top_p,
            budget=budget,
            estimated_usage=_estimate_usage(
                request.prompt,
                max_tokens=snapshot.max_tokens,
                provider_model_name=full_name,
            ),
            label=f"replay:{snapshot.snapshot_id}:{request.treatment}",
            tags={
                "call_kind": "paired_replay",
                "snapshot_id": snapshot.snapshot_id,
                "treatment": request.treatment,
            },
            max_retries=max_retries,
        )
        if not result.ok or result.text == "Error":
            raise RuntimeError(f"provider replay failure: {result.error_type}")
        return ProviderCompletion.create(
            result.text,
            provider=result.provider,
            model=result.model,
            metadata={
                "usage": result.usage.to_dict(),
                "attempts": result.attempts,
                "latency_seconds": result.latency_seconds,
                "error_type": result.error_type,
            },
        )

    return PairedReplayRunner(complete).run(snapshot)


def summarize_paired_replay(result: PairedReplayResult) -> dict[str, Any]:
    records = result.records
    changed = [
        row.treatment
        for row in records
        if row.labor_action_index_delta_vs_matched != 0
        or row.consumption_action_index_delta_vs_matched != 0
    ]
    return {
        "schema_version": REPLAY_EXPERIMENT_SCHEMA_VERSION,
        "snapshot_id": result.snapshot.snapshot_id,
        "snapshot_hash": result.snapshot.snapshot_hash,
        "integrity_verified": all(row.integrity_verified for row in records),
        "treatment_count": len(records),
        "memory_sensitive": bool(changed),
        "changed_treatments": changed,
        "max_abs_labor_hours_delta": max(
            abs(row.executed_labor_hours_delta_vs_matched) for row in records
        ),
        "max_abs_consumption_rate_delta": max(
            abs(row.executed_consumption_rate_delta_vs_matched) for row in records
        ),
        "interpretation": (
            "Action sensitivity only; no treatment has a simulated downstream outcome "
            "in this prompt-level replay."
        ),
        "scientific_evidence": False,
    }


def write_paired_replay_artifacts(
    run_dir: str | Path,
    result: PairedReplayResult,
    *,
    budget_snapshot: Mapping[str, Any],
    provenance: Mapping[str, Any],
    git_commit: str,
    git_dirty: bool,
) -> Path:
    summary = summarize_paired_replay(result)
    schemas = (
        JsonlStreamSchema(
            name="records",
            relative_path="streams/replay_records.jsonl",
            fields=(
                JsonField("schema_version", "string"),
                JsonField("snapshot_id", "string"),
                JsonField("treatment", "string"),
                JsonField("integrity_verified", "boolean"),
            ),
            allow_extra_fields=True,
        ),
        JsonlStreamSchema(
            name="summary",
            relative_path="streams/summary.jsonl",
            fields=(
                JsonField("schema_version", "string"),
                JsonField("snapshot_id", "string"),
                JsonField("integrity_verified", "boolean"),
                JsonField("memory_sensitive", "boolean"),
            ),
            allow_extra_fields=True,
        ),
    )
    writer = RunArtifactWriter.create(
        Path(run_dir),
        schemas,
        config={"snapshot": result.snapshot.manifest_dict()},
        provenance=provenance,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    for record in result.records:
        writer.append("records", record.to_dict())
    writer.append("summary", summary)
    path = writer.finalize(
        validation_status={
            "status": "pass" if summary["integrity_verified"] else "fail",
            "memory_sensitive": summary["memory_sensitive"],
            "scientific_evidence": False,
        },
        budget_snapshot=budget_snapshot,
        result_complete=True,
    )
    verify_manifest(Path(run_dir))
    return path


__all__ = [
    "REPLAY_EXPERIMENT_SCHEMA_VERSION",
    "build_paired_snapshot",
    "run_paired_replay",
    "summarize_paired_replay",
    "write_paired_replay_artifacts",
]
