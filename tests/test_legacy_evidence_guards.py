from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOW_MACRO = r"\FinEvoAllowHistoricalPrePZeroEvidence"
LEGACY_GUARD = (
    r"\errmessage{HISTORICAL PRE-P0 V1 evidence requires explicit legacy opt-in}"
)


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
