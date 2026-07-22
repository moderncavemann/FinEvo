"""Artifact schemas and persistence for :mod:`verified_memory.runner`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import json

from .artifacts import JsonField, JsonlStreamSchema, RunArtifactWriter, verify_manifest
from .runner import VerifiedRunResult


def _schema(
    name: str,
    fields: tuple[JsonField, ...],
    *,
    required: bool = True,
) -> JsonlStreamSchema:
    return JsonlStreamSchema(
        name=name,
        relative_path=f"streams/{name}.jsonl",
        fields=fields,
        required=required,
        min_records=1 if required else 0,
        allow_extra_fields=True,
    )


def verified_run_schemas(*, semantic_required: bool) -> tuple[JsonlStreamSchema, ...]:
    """Return the complete declared stream contract for a verified run."""

    core = (
        _schema(
            "actions",
            (
                JsonField("schema_version", "string"),
                JsonField("decision_t", "integer"),
                JsonField("agent_id", "integer"),
                JsonField("decision", "object"),
            ),
        ),
        _schema(
            "api_usage",
            (
                JsonField("schema_version", "string"),
                JsonField("call_kind", "string"),
                JsonField("decision_t", "integer"),
                JsonField("agent_id", "integer"),
                JsonField("usage", "object"),
            ),
        ),
        _schema(
            "context_trace",
            (
                JsonField("schema_version", "string"),
                JsonField("decision_t", "integer"),
                JsonField("agent_id", "integer"),
                JsonField("context_hash", "string"),
            ),
        ),
        _schema(
            "decision_snapshots",
            (
                JsonField("schema_version", "string"),
                JsonField("decision_t", "integer"),
                JsonField("agent_id", "integer"),
                JsonField("environment_state_hash", "string"),
                JsonField("base_prompt_hash", "string"),
                JsonField("memory_hash", "string"),
            ),
        ),
        _schema(
            "episodes",
            (
                JsonField("schema_version", "string"),
                JsonField("episode_id", "string"),
                JsonField("decision_t", "integer"),
                JsonField("outcome_t", "integer"),
                JsonField("record_hash", "string"),
            ),
        ),
        _schema(
            "utility_ledger",
            (
                JsonField("schema_version", "string"),
                JsonField("period", "integer"),
                JsonField("agent_id", "string"),
                JsonField("flow_utility", "number"),
                JsonField("budget_residual", "number"),
            ),
        ),
        _schema(
            "macro_steps",
            (
                JsonField("schema_version", "string"),
                JsonField("decision_t", "integer"),
                JsonField("outcome_t", "integer"),
                JsonField("low_labor_rate", "number"),
            ),
        ),
        _schema(
            "summary",
            (
                JsonField("schema_version", "string"),
                JsonField("run_id", "string"),
                JsonField("result_complete", "boolean"),
                JsonField("validation", "object"),
            ),
        ),
    )
    semantic = (
        _schema(
            "semantic_proposals",
            (
                JsonField("schema_version", "string"),
                JsonField("current_t", "integer"),
                JsonField("agent_id", "integer"),
            ),
            required=semantic_required,
        ),
        _schema(
            "semantic_rule_events",
            (
                JsonField("schema_version", "string"),
                JsonField("event_id", "string"),
                JsonField("agent_id", "integer"),
            ),
            required=semantic_required,
        ),
        _schema(
            "semantic_rules",
            (
                JsonField("schema_version", "string"),
                JsonField("rule_id", "string"),
                JsonField("agent_id", "integer"),
                JsonField("status", "string"),
            ),
            required=semantic_required,
        ),
        _schema(
            "errors",
            (JsonField("error_type", "string", required=False, nullable=True),),
            required=False,
        ),
    )
    return core + semantic


def write_verified_run_artifacts(
    run_dir: str | Path,
    result: VerifiedRunResult,
    *,
    provenance: Mapping[str, Any],
    git_commit: str,
    git_dirty: bool,
) -> Path:
    """Persist, seal, and independently re-hash one completed run."""

    if not isinstance(result, VerifiedRunResult):
        raise TypeError("result must be VerifiedRunResult")
    semantic_required = bool(result.config.get("enable_semantic"))
    writer = RunArtifactWriter.create(
        Path(run_dir),
        verified_run_schemas(semantic_required=semantic_required),
        config=result.config,
        provenance=dict(provenance),
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    for stream_name, rows in result.records.items():
        if stream_name not in {schema.name for schema in verified_run_schemas(
            semantic_required=semantic_required
        )}:
            raise ValueError(f"runner produced undeclared stream: {stream_name}")
        for row in rows:
            writer.append(stream_name, row)
    writer.append("summary", result.summary)
    manifest_path = writer.finalize(
        validation_status=result.validation_status,
        budget_snapshot=result.budget_snapshot,
        result_complete=bool(result.summary["result_complete"]),
    )
    verification = verify_manifest(Path(run_dir))
    if not verification.valid:
        raise RuntimeError("artifact manifest verification unexpectedly failed")
    return manifest_path


def load_verified_run_artifacts(run_dir: str | Path) -> VerifiedRunResult:
    """Load a sealed run only after its manifest and every file hash verify."""

    root = Path(run_dir)
    verify_manifest(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records: dict[str, tuple[Mapping[str, Any], ...]] = {}
    summary: Mapping[str, Any] | None = None
    for schema in verified_run_schemas(
        semantic_required=bool(config.get("enable_semantic"))
    ):
        path = root / schema.relative_path
        rows: tuple[Mapping[str, Any], ...] = ()
        if path.exists():
            rows = tuple(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            )
        if schema.name == "summary":
            if len(rows) != 1:
                raise ValueError("sealed verified run must contain exactly one summary")
            summary = rows[0]
        else:
            records[schema.name] = rows
    if summary is None:
        raise ValueError("sealed verified run is missing summary")
    return VerifiedRunResult(
        config=config,
        summary=summary,
        validation_status=manifest["validation_status"],
        budget_snapshot=manifest["budget_snapshot"],
        records=records,
    )


__all__ = [
    "load_verified_run_artifacts",
    "verified_run_schemas",
    "write_verified_run_artifacts",
]
