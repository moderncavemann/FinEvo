> [!WARNING]
> **HISTORICAL PRE-P0 V1 EVIDENCE ONLY**
>
> This E9 pair used legacy `simulate.py` and deterministic-template memory. Its
> completion does not constitute current M1 route-decomposition evidence.

# E9 paired cue-visibility analysis

Status: **complete**

Paired seeds: [13, 21, 42, 87, 2026]

This analysis compares only the same-day `visible-cue-control` and `no-visible-cue` E9 rows. It does not fall back to older E2 runs.

| Metric | Visible mean +/- SD | No-visible mean +/- SD | No-visible - visible [95% t CI] | Paired p |
|---|---:|---:|---:|---:|
| avg_wealth_final | 183147 +/- 13295.7 | 178011 +/- 6598.87 | -5135.98 [-19310.3, 9038.32] | 0.3713 |
| gini_final | 0.31785 +/- 0.0438843 | 0.363155 +/- 0.0143885 | 0.0453054 [-0.0182649, 0.108876] | 0.119 |
| unemployment_final_pct | 0.166667 +/- 0.372678 | 0.333333 +/- 0.745356 | 0.166667 [-0.296074, 0.629408] | 0.3739 |
| low_labor_h_lt_1_full_horizon_pct | 0.75 +/- 0.456435 | 1.41667 +/- 1.33723 | 0.666667 [-0.823833, 2.15717] | 0.2821 |
| inflation_dev_abs_pct | 41.6225 +/- 3.60848 | 40.7835 +/- 5.23358 | -0.839001 [-4.32079, 2.64279] | 0.5401 |
| gdp_volatility_pct | 0 +/- 0 | 0 +/- 0 | 0 [0, 0] | 1 |
| invalid_action_rate | 0 +/- 0 | 0 +/- 0 | 0 [0, 0] | 1 |
| api_error_rate | 0 +/- 0 | 0 +/- 0 | 0 [0, 0] | 1 |
| total_cost_usd | 1.57281 +/- 0.00601423 | 1.54432 +/- 0.00629817 | -0.0284892 [-0.0332607, -0.0237177] | 7.756e-05 |
| wall_time_min | 1.68067 +/- 0.0979634 | 1.58861 +/- 0.0998737 | -0.0920559 [-0.269289, 0.0851766] | 0.2227 |

## Interpretation

Hiding the prompt-level cue reduced mean final wealth by 2.80% (-5136.0; lower in 4/5 seeds), increased Gini by 0.0453 (worse in 4/5), and increased the full-horizon low-labor rate by 0.667 percentage points (worse in 4/5). Absolute inflation deviation improved by 0.839 points.

The point estimates suggest that the prompt cue helps, especially for distributional and labor outcomes. The no-visible condition retains 97.20% of visible-control mean wealth and has a 1.42% full-horizon low-labor rate, but these point estimates alone do not establish robustness.

For wealth, Gini, and full-horizon low labor, the 95% paired Student-t CIs all include zero; their two-sided paired-test p-values are 0.371, 0.119, and 0.282, respectively. With n=5 and no pre-specified non-inferiority margin, the result supports neither a statistically detectable cue effect nor an equivalence/non-inferiority claim. The defensible conclusion is that hiding the cue does not produce an obvious point-estimate collapse, while uncertainty remains large.

`low_labor_h_lt_1_full_horizon_pct` is reconstructed from all executed agent-month actions in `agent_state.csv`; it is distinct from the exported final-period `unemployment_final_pct` field.

`unemployment_final_pct` is the final logged annual-window metric, not a full-horizon average. In these 24-month runs there is only one logged annual GDP-growth observation, so `gdp_volatility_pct = 0` is mechanical, not interpretable, and is not used in the conclusion.

The experiment isolates explicit cue text only. Endogenous sentiment still enters the numeric state-similarity feature vector; the separate `regime_match` retrieval component is currently fixed at 0.0.
