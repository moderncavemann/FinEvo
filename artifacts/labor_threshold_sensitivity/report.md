# Labor-hours threshold sensitivity audit

## Scope and provenance

This audit uses the matched GPT-5.2 legacy runs `GPT_Baseline` and `GPT_Full`: seeds 0, 1, and 2; 10 agents; 24 months. Each setting therefore contributes 720 agent-month observations. Variability is reported across the three seeds; the agent-month rows are not treated as independent experimental replicates.

## Executed labor (environment action)

| Strict threshold | Text-only low-labor rate | FinEvo low-labor rate | FinEvo - text-only |
|---:|---:|---:|---:|
| h < 1 h | 2.500% +/- 0.722 | 1.389% +/- 0.481 | -1.111 pp |
| h < 20 h | 2.500% +/- 0.722 | 1.389% +/- 0.481 | -1.111 pp |
| h < 40 h | 2.500% +/- 0.722 | 1.389% +/- 0.481 | -1.111 pp |

The three executed-action thresholds are identical by construction, not three independent robustness checks. The runner samples the LLM work propensity to a binary `SimpleLabor` action, and this configuration maps the two possible actions to 0 or 168 monthly hours. The executed `h < 1` rate exactly reproduces each run's stored `avg_unemployment` value.
With only three matched seeds, these values are descriptive; no inferential p-value is reported.

## Proposed labor (pre-sampling LLM propensity)

| Strict threshold | Text-only low-labor rate | FinEvo low-labor rate |
|---:|---:|---:|
| h < 1 h | 0.000% +/- 0.000 | 0.000% +/- 0.000 |
| h < 20 h | 0.000% +/- 0.000 | 0.000% +/- 0.000 |
| h < 40 h | 0.000% +/- 0.000 | 0.000% +/- 0.000 |

| Setting | Mean proposed hours | Median | IQR | Range |
|---|---:|---:|---:|---:|
| Text-only | 162.960 | 164.640 | 164.640-164.640 | 120.960-168.000 |
| FinEvo | 166.012 | 168.000 | 164.640-168.000 | 144.480-168.000 |

No proposed action is near the 1/20/40-hour cutoffs. This rules out the specific artifact in which FinEvo merely moves agents from about 0 hours to 2-5 hours. It does **not** establish realistic labor-market calibration: both systems operate close to the 168-hour ceiling, and only three matched seeds are available. The defensible paper claim is therefore narrow: the reported difference is not caused by agents clustering just above the original `h < 1` cutoff, while the binary action space and ceiling saturation remain limitations.

## Reproduction

Run from the `eccv26_EconAgent` repository root:

```bash
python artifacts/labor_threshold_sensitivity/analyze_labor_thresholds.py
```

Machine-readable outputs are `labor_agent_month.csv`, `labor_threshold_by_seed.csv`, `labor_threshold_summary.csv`, `labor_threshold_paired_differences.csv`, and `source_manifest.json`.
