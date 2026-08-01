# V2.11.5 executed-labor threshold sensitivity diagnostic

## Verdict

This is a retrospective, descriptive, zero-provider diagnostic. It does **not** restore or reverse the authoritative Experiment A or Experiment C no-go.

The metric is **below-threshold executed labor action rate**. It is not an unemployment rate, employment-state estimate, causal effect, or effectiveness result.

## Frozen metric

For each complete 4-agent × 12-month seed/run, the denominator is 48 executed actions. The frozen thresholds are `h < 1` (0 hours), `h < 20` (0/8/16), and `h < 40` (0/8/16/24/32). Rates are computed within each seed/run; summaries use the seed/run as the unit.

## Denominator

| Item | Count |
|---|---:|
| Registered A+C cells | 45 |
| Registered actor runs | 40 |
| Structural candidate-admission N/A cells | 5 |
| Complete actor runs | 37 |
| Failed actor runs retained | 3 |
| Registered actor action opportunities | 1920 |
| Observed actor actions | 1776 |
| Missing actor actions | 144 |

All three failed A cells remain null and contribute 48 missing actions each. No failed cell was retried, replaced, removed, or imputed. The five candidate-admission cells are retained in the 45-cell inventory but are structurally N/A for an actor-action metric.

## Arm summaries

Values below are mean / median / range over complete seed/runs only; the complete/registered column exposes missingness.

| Stage | Arm | Complete / registered | h<1 | h<20 | h<40 |
|---|---|---:|---:|---:|---:|
| experiment-a | no-context | 4/5 | 0.52% / 0.00% / [0.00%, 2.08%] | 27.08% / 27.08% / [25.00%, 29.17%] | 57.81% / 55.21% / [54.17%, 66.67%] |
| experiment-a | prompt-only | 5/5 | 0.00% / 0.00% / [0.00%, 0.00%] | 17.08% / 18.75% / [4.17%, 22.92%] | 43.33% / 39.58% / [35.42%, 64.58%] |
| experiment-a | retrieval-only | 3/5 | 0.69% / 0.00% / [0.00%, 2.08%] | 31.25% / 33.33% / [20.83%, 39.58%] | 54.86% / 56.25% / [50.00%, 58.33%] |
| experiment-a | full | 5/5 | 0.00% / 0.00% / [0.00%, 0.00%] | 8.33% / 8.33% / [0.00%, 18.75%] | 31.25% / 37.50% / [12.50%, 43.75%] |
| experiment-c | full | 5/5 | 0.00% / 0.00% / [0.00%, 0.00%] | 12.08% / 8.33% / [2.08%, 22.92%] | 40.42% / 37.50% / [27.08%, 56.25%] |
| experiment-c | unverified-dual | 5/5 | 0.83% / 0.00% / [0.00%, 2.08%] | 21.25% / 20.83% / [18.75%, 25.00%] | 45.42% / 47.92% / [39.58%, 50.00%] |
| experiment-c | verified-error-forced | 5/5 | 11.67% / 8.33% / [6.25%, 20.83%] | 33.75% / 31.25% / [29.17%, 45.83%] | 55.83% / 52.08% / [47.92%, 68.75%] |
| experiment-c | unverified-error-forced | 5/5 | 25.00% / 25.00% / [18.75%, 31.25%] | 36.67% / 39.58% / [20.83%, 43.75%] | 53.75% / 52.08% / [43.75%, 60.42%] |

## Paired descriptive contrasts

Each delta is left-arm rate minus right-arm rate. No contrast is interpreted as causal or used as a replacement pass/fail gate.

| Contrast | Complete pairs | h<1 median delta | h<20 median delta | h<40 median delta |
|---|---:|---:|---:|---:|
| a_full_minus_prompt_only | 5/5 | 0.00% | -10.42% | -14.58% |
| a_retrieval_only_minus_no_context | 2/5 | 0.00% | 1.04% | -1.04% |
| c_full_minus_unverified_dual | 5/5 | 0.00% | -14.58% | -10.42% |
| c_verified_error_forced_minus_unverified_error_forced | 5/5 | -10.42% | -10.42% | 6.25% |

The A `full − prompt-only` contrast has 5/5 complete pairs. The A `retrieval-only − no-context` contrast has only 2/5 complete pairs; the other three registered pairs remain explicitly excluded-null because at least one source run failed.

## Claim boundary

This zero-provider retrospective diagnostic cannot restore or reverse the authoritative Experiment A retrieval-effect no-go or Experiment C rule-reliability no-go.

Experiment A remains a no-go (3/5 primary directions; median relative effect 3.062%, below the frozen 5% threshold, with a failed retrieval-only route manipulation check). Experiment C remains a no-go because its preregistered zero-API sensitivity artifact was not sealed by the authoritative stage. This publication-time labor diagnostic changes neither decision.

## Provenance

Source aggregate SHA-256: `5b50767e7e6f6f53aee8cc64f7f99a7c83a61cf8d57f28c73b0a205e30ac0c97`.

Selected A/C row projection SHA-256: `91d3ad1a2c60cbf32fa87d926dc034cb78845058590c90a81b1d90e5dbb52002`.

Diagnostic content SHA-256: `32eae1e166353f07e1fadd2e7596544f5459efb22aa43ceda435cd5de1e9dc26`.

New provider calls: `0`; hosted cost: `$0`; credential reads: `0`.
