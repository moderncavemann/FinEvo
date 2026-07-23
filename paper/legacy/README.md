> [!WARNING]
> **HISTORICAL / LEGACY EVIDENCE INDEX**
>
> The files indexed here comprise two historical strata: the
> deterministic-template implementation before commit `3a3f30c`, and the later
> verified-dual-track v1 smoke/replay fixtures that still predate the current P0
> method. Neither stratum may support conclusions about Evidence-Grounded Rule
> Memory without a fresh, method-matched run and provenance audit.

# Legacy evidence index

This index prevents the pre-redesign paper package from being mistaken for the
active method contract or current experimental evidence.

## Indexed legacy material

| File or family | Legacy scope and reuse risk |
| --- | --- |
| [Reviewer HoMs rebuttal draft](reviewer_HoMs_rebuttal_legacy.md) | Reports deterministic rule templates, zero activation in short runs, a fixed `regime_match` component, and binary executed labor. Those findings belong only to the evaluated legacy implementation. |
| [`paper/main.tex`](../main.tex) | Full pre-redesign manuscript. Its method names and aggregate claims can be confused with the current architecture if copied in isolation. |
| [`paper/experiments.tex`](../experiments.tex) | Legacy experiment narrative and result tables. It is not a run contract or result record for Evidence-Grounded Rule Memory. |
| [`docs/reviewer_evidence_and_claim_audit.md`](../../docs/reviewer_evidence_and_claim_audit.md) | Calls the deterministic-template system "current code" in its original context. Treat that phrase as historical, not present-tense repository state. |
| [`docs/missing_result_requirements.md`](../../docs/missing_result_requirements.md) | Historical integration checklist whose completed/pending labels apply only to the legacy run matrix, not to current-method Experiments A--D. |
| [`artifacts/rule_audit/`](../../artifacts/rule_audit/) | Audits deterministic semantic templates in legacy E1/E2 logs. It cannot measure hallucinated or rejected LLM-authored rules and is not evidence for the current M3 lifecycle. |
| [`artifacts/case_study/`](../../artifacts/case_study/) | Contains one provenance-checked legacy GPT-4o E1 observational trace. It is neither a causal estimate nor a current-method checkpoint continuation. |
| [`artifacts/labor_threshold_sensitivity/`](../../artifacts/labor_threshold_sensitivity/) | Shows that the legacy low-labor result is not a 1-versus-20/40-hour cutoff artifact, while also exposing the old binary 0/168 execution and ceiling saturation. It is not current utility calibration. |
| [`artifacts/no_visible_cue/`](../../artifacts/no_visible_cue/) | Contains the five-seed E9 cue-visibility pair run with legacy `simulate.py` and deterministic semantic memory. Its `complete` status refers only to that historical pair, not to current M1 Experiment A. |
| [Historical pre-P0 v1 smoke report](../../artifacts/verified_memory_smoke_report.md) | Later historical stratum: covers the post-`3a3f30c` verified-dual-track v1 implementation and sealed run/replay fixtures, but predates the current P0 completion. Its integrity checks and quantitative observations do not validate Evidence-Grounded Rule Memory v2. |
| [`related_work_econagent_econai.tex`](../generated_tex/related_work_econagent_econai.tex) and [`table_semantic_rule_audit.tex`](../generated_tex/table_semantic_rule_audit.tex) | Explicitly characterize deterministic semantic-rule state and its legacy audit. They cannot validate LLM-generated rule evidence in the redesigned method. |
| [`table_no_visible_cue.tex`](../generated_tex/table_no_visible_cue.tex), [`limitations_snippet.tex`](../generated_tex/limitations_snippet.tex), and [`reviewer_artifacts_compile_check.tex`](../generated_tex/reviewer_artifacts_compile_check.tex) | Reviewer-response artifacts produced alongside the legacy package; preserve their original statistical and scope qualifiers. |
| [`figs/emnlp/`](../../figs/emnlp/) and the legacy figure/table generators | Historical manuscript visuals derived from pre-P0 exports. They are contextualized by the manuscript warning and are not current-method scientific artifacts. Generators that write LaTeX retain the fail-closed legacy opt-in. |
| [`paper/generated_tables/`](../generated_tables/) | Standalone historical CSV derivatives. Every row carries `evidence_scope=historical_pre_p0_v1` and `current_method_scientific_evidence=False`; these fields must survive any regeneration. |
| Other files under [`paper/generated_tex/`](../generated_tex/) | Generated tables and draft snippets share the old result package. Every TeX input now fails closed unless the historical manuscript explicitly opts in; that opt-in preserves compilation only and does not make the evidence current. |

## Current method boundary

Use the [Evidence-Grounded Rule Memory architecture specification](../../docs/verified_dual_track_architecture.md)
and [P0 methodology completion audit](../../docs/p0_methodology_completion_audit.md)
as the current design and claim-boundary contracts. Older sealed runs, replays,
and the historical smoke report predate the current P0 completion changes.
They are not current-method evidence and do not retroactively upgrade legacy
long-horizon results.

Any reviewer-facing claim about the current method must be backed by a rerun
whose configuration, logs, checkpoints, and summary artifacts identify the
Evidence-Grounded Rule Memory implementation. Legacy deterministic-template
results may be discussed only when labeled as such.

The raw `data/` and `runs/` inputs used by several historical analyses are
ignored and are not shipped in a fresh clone. Their checked-in derived outputs
and source hashes remain useful for auditing on the source machine, but they
are not a portable, independently replayable evidence package.
