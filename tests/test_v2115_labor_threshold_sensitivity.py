from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest

from verified_memory import v2115_labor_threshold_sensitivity as labor


ROOT = Path(__file__).resolve().parents[1]
SOURCE_AGGREGATE = ROOT / labor.SOURCE_AGGREGATE_RELATIVE
PUBLISHED_PACKAGE = ROOT / labor.OUTPUT_RELATIVE


def _aggregate():
    return labor._mapping(labor._strict_json_load(SOURCE_AGGREGATE), "aggregate")


def _fake_publisher() -> dict:
    return {
        "git_commit": "a" * 40,
        "tracked_worktree_clean": True,
        "required_tracked_files": {},
        "provider_calls": 0,
        "hosted_cost_usd": 0.0,
        "credential_reads": 0,
    }


def _fake_provenance() -> dict:
    publisher = _fake_publisher()
    value = {
        "schema_version": labor.SOURCE_PROVENANCE_SCHEMA_VERSION,
        "diagnostic_id": labor.DIAGNOSTIC_ID,
        "source_package": {
            "logical_path": labor.SOURCE_PACKAGE_RELATIVE.as_posix(),
            "aggregate_sha256": labor.SOURCE_AGGREGATE_SHA256,
        },
        "publisher": publisher,
        "publication_execution": {
            "provider_calls": 0,
            "hosted_cost_usd": 0.0,
            "credential_reads": 0,
            "source_mutations": 0,
        },
    }
    value["content_sha256"] = labor._content_sha256(value)
    return value


def _payload() -> dict:
    publisher = _fake_publisher()
    return labor._build_diagnostic_payload(
        _aggregate(),
        source_provenance=_fake_provenance(),
        publisher_provenance=publisher,
    )


def _write_files(root: Path, files: dict[str, bytes]) -> None:
    root.mkdir()
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def test_tagged_source_authority_and_full_checksum_inventory_are_valid() -> None:
    tag = labor._validate_source_git_anchor(ROOT)
    assert tag == {
        "git_tag": "pilot-v2.11.5-diagnostic-evidence-v1",
        "git_tag_object": "7b83cf3953c4f59e3c79051f40c1e40456b92ef2",
        "git_commit": "34134f2624833e45f0e1f559332b0d11ea1942d6",
    }
    source = labor._validate_source_checksums(ROOT)
    assert source["aggregate_sha256"] == labor.SOURCE_AGGREGATE_SHA256
    assert source["checksums_sha256"] == labor.SOURCE_CHECKSUMS_SHA256
    assert source["package_manifest_sha256"] == labor.SOURCE_MANIFEST_SHA256
    assert source["file_count_including_checksums"] == 23
    manifest = labor._validate_source_manifest(ROOT)
    assert manifest["scientific_complete"] is False
    assert manifest["scientific_claim_gates_supported"] is False


def test_exact_a_c_cell_and_action_denominators_are_preserved() -> None:
    payload = _payload()
    labor._validate_diagnostic_internal(payload)
    assert payload["denominator"] == {
        "registered_a_c_cell_count": 45,
        "registered_actor_run_count": 40,
        "structural_not_applicable_cell_count": 5,
        "complete_actor_run_count": 37,
        "failed_actor_run_count": 3,
        "per_actor_run_registered_action_denominator": 48,
        "registered_actor_action_opportunities": 1920,
        "observed_actor_action_count": 1776,
        "missing_actor_action_count": 144,
        "itt_failure_policy": (
            "all registered actor runs retained; failed runs have null threshold "
            "values and contribute 48 missing actions; no imputation"
        ),
        "structural_na_policy": (
            "candidate-admission cells are retained in the 45-cell inventory but "
            "excluded from the actor action-opportunity denominator"
        ),
    }
    assert len(payload["run_records"]) == 45
    assert sum(row["cell_class"] == "actor-run" for row in payload["run_records"]) == 40
    assert sum(
        row["cell_class"] == "structural-not-applicable"
        for row in payload["run_records"]
    ) == 5


def test_frozen_thresholds_use_exclusive_executed_hour_bins() -> None:
    payload = _payload()
    assert payload["metric"]["name"] == "below-threshold executed labor action rate"
    assert payload["metric"]["not_unemployment"] is True
    assert payload["thresholds"] == [
        {
            "threshold_id": "h_lt_1",
            "upper_bound_exclusive_hours": 1,
            "included_frozen_grid_hours": [0],
        },
        {
            "threshold_id": "h_lt_20",
            "upper_bound_exclusive_hours": 20,
            "included_frozen_grid_hours": [0, 8, 16],
        },
        {
            "threshold_id": "h_lt_40",
            "upper_bound_exclusive_hours": 40,
            "included_frozen_grid_hours": [0, 8, 16, 24, 32],
        },
    ]
    row = next(
        item
        for item in payload["run_records"]
        if item["stage_id"] == "experiment-a"
        and item["arm_id"] == "no-context"
        and item["seed"] == 1099057501
    )
    assert row["labor_hours_counts"] == {
        "8": 11,
        "16": 2,
        "24": 1,
        "32": 12,
        "40": 7,
        "48": 8,
        "56": 4,
        "64": 1,
        "72": 1,
        "88": 1,
    }
    assert row["threshold_results"] == {
        "h_lt_1": {
            "below_threshold_count": 0,
            "registered_action_denominator": 48,
            "rate": 0.0,
        },
        "h_lt_20": {
            "below_threshold_count": 13,
            "registered_action_denominator": 48,
            "rate": 13 / 48,
        },
        "h_lt_40": {
            "below_threshold_count": 26,
            "registered_action_denominator": 48,
            "rate": 26 / 48,
        },
    }


def test_failed_runs_are_null_and_structural_cells_are_not_quantified() -> None:
    records = _payload()["run_records"]
    failed = [row for row in records if row["source_status"] == "failed"]
    assert len(failed) == 3
    assert {
        (row["arm_id"], row["seed"])
        for row in failed
    } == {
        ("no-context", 617806385),
        ("retrieval-only", 1769977770),
        ("retrieval-only", 959809858),
    }
    for row in failed:
        assert row["threshold_results"] is None
        assert row["labor_hours_counts"] is None
        assert row["observed_action_count"] == 0
        assert row["missing_action_count"] == 48
        assert row["failure"] is not None
    structural = [
        row for row in records if row["cell_class"] == "structural-not-applicable"
    ]
    assert len(structural) == 5
    for row in structural:
        assert row["arm_id"] == "verified-error-candidate"
        assert row["threshold_results"] is None
        assert row["registered_action_denominator"] is None
        assert row["observed_action_count"] is None
        assert row["missing_action_count"] is None


def test_arm_completeness_and_pair_intersections_are_explicit() -> None:
    payload = _payload()
    arm_counts = {
        (row["stage_id"], row["arm_id"]): (
            row["complete_seed_count"],
            row["registered_seed_count"],
        )
        for row in payload["arm_summaries"]
    }
    assert arm_counts == {
        ("experiment-a", "no-context"): (4, 5),
        ("experiment-a", "prompt-only"): (5, 5),
        ("experiment-a", "retrieval-only"): (3, 5),
        ("experiment-a", "full"): (5, 5),
        ("experiment-c", "full"): (5, 5),
        ("experiment-c", "unverified-dual"): (5, 5),
        ("experiment-c", "verified-error-forced"): (5, 5),
        ("experiment-c", "unverified-error-forced"): (5, 5),
    }
    pairs = {
        row["contrast_id"]: (row["complete_pair_count"], row["complete_pair_seeds"])
        for row in payload["paired_contrasts"]
    }
    assert pairs == {
        "a_full_minus_prompt_only": (5, list(labor.SEEDS)),
        "a_retrieval_only_minus_no_context": (2, [1099057501, 1421875452]),
        "c_full_minus_unverified_dual": (5, list(labor.SEEDS)),
        "c_verified_error_forced_minus_unverified_error_forced": (
            5,
            list(labor.SEEDS),
        ),
    }


def test_raw_seed_values_and_summary_statistics_are_reproducible() -> None:
    payload = _payload()
    full_a = next(
        row
        for row in payload["arm_summaries"]
        if row["stage_id"] == "experiment-a" and row["arm_id"] == "full"
    )
    h20 = full_a["thresholds"]["h_lt_20"]
    assert h20["rates_by_seed"] == {
        "1099057501": 9 / 48,
        "1421875452": 6 / 48,
        "1769977770": 4 / 48,
        "959809858": 0.0,
        "617806385": 1 / 48,
    }
    summary = h20["summary_over_complete_seed_runs"]
    assert summary["n"] == 5
    assert summary["mean"] == pytest.approx((9 + 6 + 4 + 0 + 1) / (5 * 48))
    assert summary["median"] == 4 / 48
    assert summary["range"] == [0.0, 9 / 48]


def test_source_claim_no_go_boundaries_cannot_be_relabelled() -> None:
    payload = _payload()
    assert payload["classification"]["scientific_evidence"] is False
    assert payload["classification"]["provider_calls"] == 0
    assert payload["source_claim_boundaries"]["experiment_a"] == {
        "status": "no-go",
        "support_retrieval_effect": False,
        "scientific_evidence_complete": False,
        "primary_pair_count": 5,
        "primary_direction_count": 3,
        "primary_median_relative_effect": 0.030620326693089294,
        "threshold_checks": {
            "at_least_four_complete_pairs": True,
            "at_least_four_same_direction": False,
            "median_relative_effect_at_least_5pct": False,
        },
    }
    assert payload["source_claim_boundaries"]["experiment_c"]["status"] == "no-go"
    assert (
        payload["source_claim_boundaries"]["experiment_c"][
            "support_rule_reliability"
        ]
        is False
    )
    assert "cannot restore or reverse" in payload["claim_boundary"]


def test_file_rendering_is_byte_deterministic() -> None:
    aggregate = _aggregate()
    provenance = _fake_provenance()
    publisher = _fake_publisher()
    first = labor._build_file_bytes(
        ROOT,
        aggregate=aggregate,
        source_provenance=provenance,
        publisher_provenance=publisher,
    )
    second = labor._build_file_bytes(
        ROOT,
        aggregate=aggregate,
        source_provenance=deepcopy(provenance),
        publisher_provenance=deepcopy(publisher),
    )
    assert first == second
    assert set(first) == set(labor._OUTPUT_FILES)
    assert b"unemployment rate" in first["report.md"]
    assert b"provider calls: `0`" in first["report.md"]


def test_payload_rejects_threshold_tamper_even_with_resealed_content_hash() -> None:
    payload = _payload()
    row = next(
        item
        for item in payload["run_records"]
        if item["source_status"] == "complete" and item["cell_class"] == "actor-run"
    )
    row["threshold_results"]["h_lt_20"]["below_threshold_count"] += 1
    without_hash = dict(payload)
    without_hash.pop("content_sha256")
    payload["content_sha256"] = labor._content_sha256(without_hash)
    with pytest.raises(labor.LaborThresholdDiagnosticError, match="threshold computation"):
        labor._validate_diagnostic_internal(payload)


def test_payload_rejects_failed_run_imputation_even_with_resealed_hash() -> None:
    payload = _payload()
    row = next(item for item in payload["run_records"] if item["source_status"] == "failed")
    row["threshold_results"] = {
        "h_lt_1": {
            "below_threshold_count": 0,
            "registered_action_denominator": 48,
            "rate": 0.0,
        }
    }
    without_hash = dict(payload)
    without_hash.pop("content_sha256")
    payload["content_sha256"] = labor._content_sha256(without_hash)
    with pytest.raises(labor.LaborThresholdDiagnosticError, match="imputed"):
        labor._validate_diagnostic_internal(payload)


def test_source_validator_rejects_actor_histogram_denominator_drift() -> None:
    aggregate = deepcopy(_aggregate())
    row = next(
        item
        for item in aggregate["rows"]
        if item["stage_id"] == "experiment-a" and item["status"] == "complete"
    )
    first_key = next(iter(row["metrics"]["actions"]["labor_hours_counts"]))
    row["metrics"]["actions"]["labor_hours_counts"][first_key] += 1
    with pytest.raises(labor.LaborThresholdDiagnosticError, match="sum to 48"):
        labor._validate_aggregate(aggregate)


def test_publication_scan_rejects_absolute_paths_and_secret_shapes() -> None:
    with pytest.raises(labor.LaborThresholdDiagnosticError, match="path leaked"):
        labor._scan_publication_bytes({"bad.json": b'{"path":"/Users/example/raw"}'})
    with pytest.raises(labor.LaborThresholdDiagnosticError, match="secret leaked"):
        labor._scan_publication_bytes(
            {"bad.json": b'{"api_key":"sk-examplecredential123456789"}'}
        )


def test_strict_json_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(labor.LaborThresholdDiagnosticError, match="duplicate JSON key"):
        labor._strict_json_load(path)


def test_package_writer_is_no_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "package"
    target.mkdir()
    sentinel = target / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(labor.LaborThresholdDiagnosticError, match="refusing overwrite"):
        labor._write_new_package(target, {"value.json": b"{}\n"})
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_package_validator_rejects_checksum_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate = _aggregate()
    provenance = _fake_provenance()
    publisher = _fake_publisher()
    files = labor._build_file_bytes(
        ROOT,
        aggregate=aggregate,
        source_provenance=provenance,
        publisher_provenance=publisher,
    )
    package = tmp_path / "package"
    _write_files(package, files)
    monkeypatch.setattr(labor, "_verify_publisher_binding", lambda *_args: None)
    monkeypatch.setattr(labor, "_source_provenance", lambda *_args, **_kwargs: provenance)
    result = labor.validate_v2115_labor_threshold_package(
        package_dir=package,
        repo_root=ROOT,
    )
    assert result["registered_cells"] == 45
    (package / "per_run.csv").write_bytes(
        (package / "per_run.csv").read_bytes() + b"tamper\n"
    )
    with pytest.raises(labor.LaborThresholdDiagnosticError, match="checksum mismatch"):
        labor.validate_v2115_labor_threshold_package(
            package_dir=package,
            repo_root=ROOT,
        )


def test_builder_source_has_no_provider_or_network_imports() -> None:
    forbidden_roots = {
        "anthropic",
        "google",
        "httpx",
        "openai",
        "requests",
        "urllib",
    }
    for relative in (
        "verified_memory/v2115_labor_threshold_sensitivity.py",
        "scripts/build_v2115_labor_threshold_sensitivity.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported.isdisjoint(forbidden_roots)


def test_tracked_schema_freezes_denominator_and_metric_name() -> None:
    schema = labor._strict_json_load(ROOT / labor.SCHEMA_RELATIVE)
    assert schema["properties"]["diagnostic_id"]["const"] == labor.DIAGNOSTIC_ID
    assert (
        schema["properties"]["metric"]["properties"]["name"]["const"]
        == "below-threshold executed labor action rate"
    )
    denominator = schema["properties"]["denominator"]["properties"]
    assert denominator["registered_a_c_cell_count"]["const"] == 45
    assert denominator["registered_actor_action_opportunities"]["const"] == 1920
    assert denominator["observed_actor_action_count"]["const"] == 1776
    assert denominator["missing_actor_action_count"]["const"] == 144


def test_tracked_schema_is_valid_and_accepts_the_recomputed_payload() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = labor._strict_json_load(ROOT / labor.SCHEMA_RELATIVE)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(_payload())


@pytest.mark.skipif(
    not PUBLISHED_PACKAGE.exists(),
    reason="evidence package is added only by the second, deterministic evidence commit",
)
def test_committed_evidence_package_revalidates() -> None:
    result = labor.validate_v2115_labor_threshold_package(
        package_dir=PUBLISHED_PACKAGE,
        repo_root=ROOT,
    )
    assert result == {
        "schema_version": labor.DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_id": labor.DIAGNOSTIC_ID,
        "status": "valid",
        "content_sha256": result["content_sha256"],
        "package_checksums_sha256": result["package_checksums_sha256"],
        "registered_cells": 45,
        "registered_actor_actions": 1920,
        "observed_actor_actions": 1776,
        "missing_actor_actions": 144,
        "provider_calls": 0,
        "scientific_evidence": False,
    }
    assert len(result["content_sha256"]) == 64
    assert len(result["package_checksums_sha256"]) == 64
