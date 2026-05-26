# FinEvo Missing-Result and Integration Requirements

This file is for internal integration only. Do not copy internal experiment IDs
into the paper body. Reader-facing text should use names such as
"large-scale diagnostic", "component ablation", "regime-cue robustness", and
"prompt robustness".

## Current Status Snapshot

Completed:

- GPT-5.2 controlled component ablation: 25 runs.
- GPT-5.2 regime/sentiment parameterization robustness: 50 runs.
- GPT-5.2 textual-event controls: 20 runs.
- GPT-5.2 prompt/decoding robustness: 35 runs.
- GPT-4o large-scale seed-13 paired baseline + FinEvo.
- Gemini-3-Flash large-scale seed-13 paired baseline + FinEvo.
- Qwen3-235B large-scale seed-13 baseline.

Still missing or pending:

- GPT-5.2 large-scale text-only baseline five-seed export for the main significance table.
- GPT-5.2 large-scale FinEvo five-seed export for the main significance table.
- Qwen3-235B large-scale FinEvo seed-13 export.
- Optional matched five-seed large-scale rows for GPT-4o, Gemini, Qwen, and any clean Llama route.
- Log-grounded crash-state case-study figure/table.
- Clean Llama route, if Llama is to be reported quantitatively.
- Optional heterogeneous-cognitive-depth and rule-survival controls.

Current GPT-5.2 large-scale provenance note:

- `ACL24-EconAgent/data/openai-gpt-5.2-gap_fixed-100agents-240months/summary.json` exists and reports `avg_wealth = 2,614,208.149`, `avg_unemployment = 0.1625%`, `avg_inflation = 9.591%`, with no recorded random seed.
- `ACL24-EconAgent/data/openai-gpt-5.2-baseline-100agents-240months/summary.json` exists and reports `avg_wealth = 428,417.062`, `avg_unemployment = 15.6083%`, `avg_inflation = 1.7338%`, with no recorded random seed.
- These legacy folders are single-run provenance for the headline direction only. They are not the five-seed source for standard deviations or paired p-values.
- No GPT-5.2 large-scale five-seed FinEvo exports are currently visible under `runs/E1/GPT-5.2/finevo/default/seed_*`.
- `ENMLP26.zip` did not contain visible `gpt-5.2`/`100agents-240months` entries during the current local search.

## R1. GPT-5.2 Large-Scale Text-Only Baseline Five-Seed Row

Purpose: complete the main significance table. The matched text-only GPT-5.2
baseline row must be exported as mean +/- standard deviation before final
submission.

Inputs:

- model: `openai/gpt-5.2`
- setting: text-only baseline
- agents: 100
- months: 240
- seeds: `13,21,42,87,2026`
- temperature: `0.2`
- top_p: `1.0`
- decision max tokens: `800`
- reflection max tokens: `200` if invoked; baseline should not invoke FinEvo reflection.

Expected output per seed:

```text
runs/E1/GPT-5.2/text-only/baseline/seed_<SEED>/metrics_summary.csv
runs/E1/GPT-5.2/text-only/baseline/seed_<SEED>/trajectory.csv
runs/E1/GPT-5.2/text-only/baseline/seed_<SEED>/agent_state.csv
runs/E1/GPT-5.2/text-only/baseline/seed_<SEED>/actions.jsonl
runs/E1/GPT-5.2/text-only/baseline/seed_<SEED>/api_errors.jsonl
runs/E1/GPT-5.2/text-only/baseline/seed_<SEED>/config.yaml
```

Completion criteria:

- Five `metrics_summary.csv` files exist.
- `api_error_rate` and `invalid_action_rate` are near zero.
- Main row can be summarized for wealth, Gini, unemployment, and inflation deviation.
- Paired p-values can be recomputed against matched GPT-5.2 FinEvo seeds.

## R1b. GPT-5.2 Large-Scale FinEvo Five-Seed Row

Purpose: provide the matched FinEvo row for the main significance table. If the
five-seed source exists outside this repository, export it into the normalized
schema; otherwise rerun the five seeds.

Expected output per seed:

```text
runs/E1/GPT-5.2/finevo/default/seed_<SEED>/metrics_summary.csv
runs/E1/GPT-5.2/finevo/default/seed_<SEED>/trajectory.csv
runs/E1/GPT-5.2/finevo/default/seed_<SEED>/agent_state.csv
runs/E1/GPT-5.2/finevo/default/seed_<SEED>/actions.jsonl
runs/E1/GPT-5.2/finevo/default/seed_<SEED>/memory_retrieval.jsonl
runs/E1/GPT-5.2/finevo/default/seed_<SEED>/semantic_rules.jsonl
runs/E1/GPT-5.2/finevo/default/seed_<SEED>/api_errors.jsonl
runs/E1/GPT-5.2/finevo/default/seed_<SEED>/config.yaml
```

Completion criteria: five matched seeds exist and can be paired against the
text-only baseline row for paired t-tests. Do not use the legacy single-run
`gap_fixed` folder as a five-seed result.

## R2. Qwen3-235B Large-Scale FinEvo Seed-13 Row

Purpose: complete the Qwen baseline vs FinEvo diagnostic pair.

Inputs:

- model: `qwen/qwen3-235b-a22b-2507` through OpenRouter
- setting: FinEvo/default
- agents: 100
- months: 240
- seed: `13`
- workers: `2`

Expected output:

```text
runs/E1/Qwen3-235B/finevo/default/seed_13/metrics_summary.csv
runs/E1/Qwen3-235B/finevo/default/seed_13/trajectory.csv
runs/E1/Qwen3-235B/finevo/default/seed_13/agent_state.csv
runs/E1/Qwen3-235B/finevo/default/seed_13/actions.jsonl
runs/E1/Qwen3-235B/finevo/default/seed_13/memory_retrieval.jsonl
runs/E1/Qwen3-235B/finevo/default/seed_13/semantic_rules.jsonl
runs/E1/Qwen3-235B/finevo/default/seed_13/config.yaml
```

Completion criteria: source `summary.json` exists; exported run folder has the
required files and the metrics row includes wealth, Gini, unemployment,
inflation deviation, API error rate, invalid action rate, and wall time.

## R3. Cross-Model Diagnostic Summary Generation

Purpose: regenerate the cross-model diagnostic table after GPT-4o, Gemini, and
Qwen rows are exported.

Expected output:

```text
paper/generated_tables/E1_diagnostic_summary.csv
paper/generated_tex/table_cross_model_diagnostic.tex
```

Completion criteria: GPT-4o baseline + FinEvo present, Gemini baseline + FinEvo
present, and Qwen baseline + FinEvo present if Qwen completed. Otherwise, Qwen
FinEvo stays out of final main text.

## R4. Main Significance Table Regeneration

Purpose: replace pending values in the main significance and ablation table.

Inputs:

- GPT-5.2 text-only baseline five seeds.
- GPT-5.2 FinEvo five seeds.
- Controlled ablation five seeds.

Expected output:

```text
paper/generated_tables/main_significance_and_ablation.csv
paper/generated_tex/table_main_significance_and_ablation.tex
```

Statistical tests: paired t-test between GPT-5.2 baseline and GPT-5.2 FinEvo
using matched seeds. Report p-values for wealth, Gini, and unemployment at
minimum.

## R5. Crash-State Case Study

Purpose: produce a log-grounded qualitative figure/table analogous to an
input-output trace.

Default deterministic selection rule: choose the lowest global sentiment month
for each completed FinEvo large-scale diagnostic run.

Required input files per row:

```text
actions.jsonl
memory_retrieval.jsonl
semantic_rules.jsonl
trajectory.csv
agent_state.csv
config.yaml
```

Expected output:

```text
paper/generated_tables/case_study_trace.csv
paper/generated_tex/table_case_study_trace.tex
figs/emnlp/fig_case_study.pdf
figs/emnlp/fig_case_study.png
```

Completion criteria: every excerpt must be copied from logs, not invented. Do
not include Llama unless a clean JSON-compliant route is available.

## R6. EconAgent-Style Baseline Regeneration

Purpose: make the prior-method reference row reproducible. If only the legacy
reference row is used, mark it as reference rather than a matched five-seed
comparison.

## R7. Llama Clean-Route Probe and Optional Full Run

Purpose: determine whether Llama can be included without output-compliance
artifacts.

Completion criteria:

- invalid_action_rate below 5% in the probe.
- raw outputs conform to JSON without long reasoning text.
- if the probe fails, do not include Llama in the quantitative table; mention route-level compliance in limitations or appendix.

## R8. Optional Heterogeneous Cognitive-Depth Control

Purpose: address cognitive-depth concern if time permits. Include only with at
least five seeds for a controlled setting.

## R9. Optional Rule Survival Control

Purpose: test whether tail-risk rules survive calm periods. Include only if
rule categories and confidence trajectories are extracted from logs.
