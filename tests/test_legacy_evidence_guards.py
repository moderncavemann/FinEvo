import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOW_MACRO = r"\FinEvoAllowHistoricalPrePZeroEvidence"
LEGACY_GUARD = (
    r"\errmessage{HISTORICAL PRE-P0 V1 evidence requires explicit legacy opt-in}"
)
HISTORICAL_BANNER = "HISTORICAL PRE-P0 V1 EVIDENCE ONLY"


def test_legacy_manuscript_banner_is_visible_and_generated_inputs_fail_closed() -> None:
    main = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    assert rf"\def{ALLOW_MACRO}{{}}" in main
    assert r"\textbf{HISTORICAL PRE-P0 V1 MANUSCRIPT.}" in main

    guarded = [ROOT / "paper" / "experiments.tex"]
    guarded.extend(sorted((ROOT / "paper" / "generated_tex").glob("*.tex")))
    assert len(guarded) == 18
    for path in guarded:
        contents = path.read_text(encoding="utf-8")
        assert rf"\ifdefined{ALLOW_MACRO}\else" in contents, path
        assert LEGACY_GUARD in contents, path


def test_historical_rebuttal_path_is_a_redirect_not_a_second_live_draft() -> None:
    redirect = (ROOT / "paper" / "reviewer_HoMs_rebuttal_draft.md").read_text(
        encoding="utf-8"
    )
    archived = ROOT / "paper" / "legacy" / "reviewer_HoMs_rebuttal_legacy.md"
    assert archived.is_file()
    assert "HISTORICAL / LEGACY REDIRECT" in redirect
    assert "legacy/reviewer_HoMs_rebuttal_legacy.md" in redirect
    assert "## Response to Reviewer HoMs" not in redirect


def test_repository_level_legacy_boundaries_remain_explicit() -> None:
    expected_markers = {
        ROOT / "README.md": "No current-method performance result is claimed here.",
        ROOT
        / "artifacts"
        / "verified_memory_smoke_report.md": HISTORICAL_BANNER,
        ROOT
        / "docs"
        / "reviewer_evidence_and_claim_audit.md": "HISTORICAL / LEGACY EVIDENCE ONLY",
        ROOT / "paper" / "legacy" / "README.md": "HISTORICAL / LEGACY EVIDENCE INDEX",
        ROOT
        / "paper"
        / "legacy"
        / "reviewer_HoMs_rebuttal_legacy.md": "HISTORICAL / LEGACY EVIDENCE ONLY",
    }
    for path, marker in expected_markers.items():
        assert marker in path.read_text(encoding="utf-8"), path


def test_legacy_analysis_artifacts_are_self_labeling_and_regeneration_safe() -> None:
    human_entrypoints = [
        ROOT / "artifacts" / "rule_audit" / "README.md",
        ROOT / "artifacts" / "rule_audit" / "results" / "qa_validation.md",
        ROOT / "artifacts" / "rule_audit" / "results" / "source_notes.md",
        ROOT / "artifacts" / "case_study" / "case_study_extraction_report.md",
        ROOT / "artifacts" / "labor_threshold_sensitivity" / "report.md",
        ROOT / "artifacts" / "no_visible_cue" / "README.md",
        ROOT / "artifacts" / "no_visible_cue" / "paired_report.md",
        ROOT / "artifacts" / "no_visible_cue" / "smoke_report.md",
        ROOT / "docs" / "missing_result_requirements.md",
    ]
    for path in human_entrypoints:
        assert HISTORICAL_BANNER in path.read_text(encoding="utf-8"), path

    html_entrypoints = [
        ROOT / "artifacts" / "rule_audit" / "results" / "audit_report.html",
        ROOT / "artifacts" / "rule_audit" / "results" / "audit_report_shell.html",
    ]
    for path in html_entrypoints:
        contents = path.read_text(encoding="utf-8")
        assert HISTORICAL_BANNER in contents, path
        assert 'data-evidence-scope="historical-pre-p0-v1"' in contents, path

    generators = [
        ROOT / "artifacts" / "rule_audit" / "audit_semantic_rules.py",
        ROOT / "artifacts" / "case_study" / "extract_case_study.py",
        ROOT
        / "artifacts"
        / "labor_threshold_sensitivity"
        / "analyze_labor_thresholds.py",
        ROOT / "artifacts" / "no_visible_cue" / "analyze_paired_results.py",
    ]
    for path in generators:
        assert HISTORICAL_BANNER in path.read_text(encoding="utf-8"), path

    legacy_tex_generators = [
        ROOT / "paper" / "generate_missing_result_outputs.py",
        ROOT / "paper" / "generate_additional_emnlp_outputs.py",
    ]
    for path in legacy_tex_generators:
        contents = path.read_text(encoding="utf-8")
        assert ALLOW_MACRO in contents, path
        assert LEGACY_GUARD in contents, path

    machine_records = [
        ROOT / "artifacts" / "rule_audit" / "results" / "audit_manifest.json",
        ROOT / "artifacts" / "rule_audit" / "results" / "audit_report_payload.json",
        ROOT / "artifacts" / "case_study" / "case_study_trace.json",
        ROOT / "artifacts" / "labor_threshold_sensitivity" / "source_manifest.json",
        ROOT / "artifacts" / "no_visible_cue" / "paired_analysis_status.json",
        ROOT / "artifacts" / "no_visible_cue" / "code_provenance.json",
    ]
    for path in machine_records:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["evidence_scope"] == "historical_pre_p0_v1", path
        assert record["current_method_scientific_evidence"] is False, path

    case_trace = json.loads(
        (
            ROOT / "artifacts" / "case_study" / "case_study_trace.json"
        ).read_text(encoding="utf-8")
    )
    assert case_trace["trace_scope"]["supports_single_agent_causal_attribution"] is False
    assert "old rebuttal runs" in case_trace["run_metadata"]["legacy_export_note"].lower()

    case_schema = json.loads(
        (
            ROOT / "artifacts" / "case_study" / "case_study_trace_schema.json"
        ).read_text(encoding="utf-8")
    )
    for field in (
        "evidence_scope",
        "current_method_scientific_evidence",
        "method_implementation",
    ):
        assert field in case_schema["required"]

    for figure_name in (
        "case_study_trace_figure.pdf",
        "case_study_trace_figure.png",
    ):
        figure = ROOT / "artifacts" / "case_study" / figure_name
        assert HISTORICAL_BANNER.encode() in figure.read_bytes(), figure

    manuscript_figures = sorted((ROOT / "figs" / "emnlp").glob("*.pdf")) + sorted(
        (ROOT / "figs" / "emnlp").glob("*.png")
    )
    expected_figure_names = {
        f"{stem}.{suffix}"
        for stem in (
            "fig3_sentiment_robustness",
            "fig_case_study",
            "fig_component_ablation",
            "fig_e2_trajectory_diagnostics",
            "fig_prompt_robustness",
            "fig_text_event_controls",
        )
        for suffix in ("pdf", "png")
    }
    assert expected_figure_names <= {path.name for path in manuscript_figures}
    for figure in manuscript_figures:
        assert HISTORICAL_BANNER.encode() in figure.read_bytes(), figure

    generated_tables = sorted((ROOT / "paper" / "generated_tables").glob("*.csv"))
    expected_table_names = {
        "E1_diagnostic_pending.csv",
        "E1_diagnostic_summary.csv",
        "E1_summary.csv",
        "E2_summary.csv",
        "E3_summary.csv",
        "E4_event_fixture_examples.csv",
        "E4_summary.csv",
        "E5_summary.csv",
        "appendix_cost_error_summary.csv",
        "case_study_trace.csv",
        "main_significance_and_ablation.csv",
    }
    assert expected_table_names <= {path.name for path in generated_tables}
    for table in generated_tables:
        with table.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, table
        for row in rows:
            assert row["evidence_scope"] == "historical_pre_p0_v1", table
            assert row["current_method_scientific_evidence"] == "False", table
            assert (
                row["method_implementation"]
                == "legacy_simulate_py_deterministic_template_memory"
            ), table

    validator = (
        ROOT / "artifacts" / "no_visible_cue" / "validate_no_visible_cue.py"
    ).read_text(encoding="utf-8")
    assert '"evidence_scope": "historical_pre_p0_v1"' in validator
    assert "current_method_scientific_evidence" in validator

    figure_generators = [
        ROOT / "paper" / "generate_emnlp_figures.py",
        ROOT / "paper" / "generate_additional_emnlp_outputs.py",
        ROOT / "paper" / "generate_missing_result_outputs.py",
        ROOT / "paper" / "generate_figures.py",
    ]
    for path in figure_generators:
        contents = path.read_text(encoding="utf-8")
        assert "HISTORICAL_FIGURE_LABEL" in contents, path
        assert HISTORICAL_BANNER in contents, path

    for path in legacy_tex_generators:
        contents = path.read_text(encoding="utf-8")
        assert contents.count(".write_text(") == 1, path
        assert contents.count("write_legacy_tex(") >= 3, path

    legacy_index = (ROOT / "paper" / "legacy" / "README.md").read_text(
        encoding="utf-8"
    )
    for artifact_dir in (
        "artifacts/rule_audit/",
        "artifacts/case_study/",
        "artifacts/labor_threshold_sensitivity/",
        "artifacts/no_visible_cue/",
    ):
        assert artifact_dir in legacy_index
    assert "not shipped in a fresh clone" in legacy_index
    assert "paper/generated_tables/" in legacy_index
