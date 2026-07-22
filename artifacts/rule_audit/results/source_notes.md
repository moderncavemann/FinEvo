> [!WARNING]
> **HISTORICAL PRE-P0 V1 EVIDENCE ONLY**
>
> These notes cover the legacy deterministic-template audit, not current
> Evidence-Grounded Rule Memory v2.

# FinEvo rule-audit source notes

## Report contract

- Audience: technical.
- Delivery mode: self-contained HTML.
- Primary question: do existing logs support a rule-level hallucination/error-reinforcement claim?
- Cohort: two E1 FinEvo seed-13 runs with semantic activity plus five GPT-5.2 E2 full seeds with observed zero activity.
- High-confidence threshold: `0.75`.
- Outcome window: next `3` simulation months.
- Runtime embedding: `embedded_runtime:embed_html_report_runtime.py`.

## Chart map

- Segment: key findings.
- Question: which audited runs contain accepted semantic-rule updates?
- Family/type: comparison / horizontal bar.
- Fields: run label, accepted create-or-merge event count; snapshot and selected-decision counts retained for tooltips.
- Takeaway: semantic activity exists only in the two long E1 runs; five GPT-5.2 E2 full seeds are observed zeroes.
- Palette: single blue root plus neutral axes and direct labels.
- Scale: absolute counts starting at zero.
- Static fallback: inline SVG generated from the same rows as `audit_report_payload.json`.

## Omitted charts

Confidence and outcome distributions are left as CSV detail because only one active seed exists per model and a polished comparative visual could imply unsupported cross-model inference. Exact audit lookup is the primary task for those fields.

## Required-structure mapping

The HTML preserves the technical report roles in order: title, technical summary, key findings, scope/data/metric definitions, methodology, limitations/uncertainty/robustness checks, recommended next steps, and further questions.

## Interpretation constraints

- `verified_supported` is a code-level replay result, not an economic-validity or human semantic-quality label.
- Proposal and rejection metrics are not recoverable and remain blank.
- `supported_by_episode` from the source logger is not accepted as audit evidence.
- Rule-use outcomes are descriptive associations without a no-rule counterfactual.
- Pickles are trusted local artifacts; their hashes are saved in `source_manifest.csv`.
