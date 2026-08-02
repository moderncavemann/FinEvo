# EGRM metric cards

Status date: 2026-08-02

These cards define the estimands for the proposed EGRM study. They do not
report results. The current scientific contract is a non-dispatchable design
draft with 105 registered cells: E0 has 48, E1 has 25, E2 has 20, and E3 has
12. A metric is scientific evidence only after the corresponding runner,
ledger, full denominator, and terminal receipt exist at a frozen release.

## Global denominator and failure rules

- E0 uses the oracle scenario family as the paired unit. E1 uses the seed as
  the primary unit and `seed-agent-rule-family` for rule audits. E2 and E3 use
  the seed-checkpoint.
- Every registered cell remains in the intention-to-treat ledger. Parse,
  provider, budget, integrity, and capability failures are terminal statuses;
  they are not dropped, retried with a replacement seed, or imputed as a
  numerical zero.
- A paired effect is computed only when both required cells are complete. The
  report must show complete-pair count separately from total registered pairs
  and list every non-complete status.
- E1 and E2 effectiveness language requires at least four of five complete
  pairs. E3 direction language requires all three registered pairs. E0 metric
  comparisons require all 12 scenario families per compared policy. Below
  those thresholds the effect is unavailable, not zero.
- Report every raw paired delta plus mean, median, range, and direction count.
  Five seeds do not support an asymptotic significance claim.

## Rule admission precision

**Question.** Among candidates admitted by the lifecycle, how many are
oracle-supported?

For candidate set $C$, oracle label $y_c\in\{0,1\}$, and admission indicator
$A_c$ for status `provisional`, `active`, or `retired`,

```text
admission precision = sum_c A_c y_c / sum_c A_c
```

Rejected candidates remain in $C$. A zero admitted-candidate denominator is
reported as `NA (zero denominator)`. This metric requires an oracle label or a
predeclared blinded adjudication protocol; a model's own confidence is not a
truth label.

**Unit and status.** Candidate-level, aggregated by scenario or
seed-agent-rule-family. Implemented for the deterministic fixture in
`compute_lifecycle_metrics`; scientific E0/E1 collection is planned.

## Unsupported activation

**Question.** How often does an oracle-unsupported candidate ever obtain
active authority?

```text
unsupported activation rate =
  unsupported candidates ever active / registered unsupported candidates
```

The denominator is all registered oracle-unsupported candidates, including
rejected and failed proposals. `unsupported share among activated rules` uses
all ever-active rules as a different diagnostic and must not replace the
primary denominator. `Ever active` is binary per candidate family even if a
rule activates more than once.

**Unit and status.** Candidate family within scenario or seed-agent-family.
Implemented for schema-valid fixture candidates; natural LLM proposals still
need the separate oracle/adjudication audit.

## Harmful exposure

**Question.** Once a false rule can influence behavior, for how long and how
often is the agent exposed to its guidance?

```text
active exposure steps = count of decision steps where the false rule is active
selected exposure steps = count where it is retrieved and selected
harmful exposure steps = count where it is selected, followed, and the
                         registered outcome criterion fails
harmful-compliance rate = harmful exposure steps / selected exposure steps
```

The primary forced-active comparison uses `harmful exposure steps`. A lifecycle
event count is not a substitute for action-ledger exposure, especially for the
unverified policy, which may ignore evidence events. Zero selected exposure is
reported with a zero step count and an `NA` rate.

**Unit and status.** Seed-agent-rule-family. Event counting exists in the
fixture; action-ledger exposure and the scientific paired comparison are
planned.

## Retirement latency with right censoring

**Question.** After the first registered harmful-compliance observation, how
long until the rule retires?

For first harmful time $t_h$, retirement time $t_r$, and observation end $T$:

```text
observed latency = t_r - t_h                     if retirement occurs
censored duration = T - t_h, right_censored=true otherwise
```

Retirement before $t_h$ is invalid. Every activated rule with harmful evidence
must produce a latency record. Censored durations cannot be averaged as if
retirement occurred at $T$; report records and a censor-aware summary. The
`no-retirement` arm is censored by construction, so its latency is a
manipulation check. Effectiveness rests on exposure and utility, not the
tautological fact that this arm cannot retire.

**Unit and status.** Seed-agent-rule-family. Right-censored records and temporal
validation are implemented; scientific survival summaries are planned.

## Alternative success

```text
alternative-success rate =
  successful in-scope noncompliance observations /
  all in-scope noncompliance observations
```

The denominator contains `alternative_success` plus `alternative_failure` and
excludes `irrelevant` observations. It measures whether good outcomes are
available without following the rule; it is evidence about necessity, not a
standalone correctness label. Report the weighted contribution to the rule
margin separately from this unweighted rate.

**Unit and status.** Seed-agent-rule-family. Alternative-success events are
classified by the extracted lifecycle; the rate and scientific aggregation are
planned.

## Paired discounted utility loss

**Question.** How much downstream utility is lost because the matched false
rule is present, and does EGRM reduce that loss?

Freeze discount $\gamma$ before dispatch. For policy $p$, seed-checkpoint $s$,
and six continuation steps,

```text
U_p,x(s) = sum_{h=0}^{5} gamma^h * sum_i u_{i,t+h}^{p,x}
L_p(s)   = U_p,no-error(s) - U_p,error(s)
benefit(s) = L_unverified(s) - L_EGRM(s)
```

Positive `benefit` means lower false-rule damage under EGRM. Focal-agent and
population utility must be separate columns. For E2, the unchanged matched A/A
branches define the no-error reference; do not combine E1 and E2 as if they
were independent observations.

**Unit and status.** Seed-checkpoint paired estimand. Planned; the extracted
prompt replay cannot produce this metric because it does not continue the
environment.

## Matched action and state effects

For action coordinate $j$, define the registered action-bin width $b_j$ and
the maximum matched null

```text
N_j = max_s |a_AA-A,j(s) - a_AA-B,j(s)|
delta_j(s) = a_treatment,j(s) - a_reference,j(s)
```

A treatment clears the manipulation threshold only if its absolute effect
exceeds both $N_j$ and one full bin $b_j$, with at least four of five paired
seeds in the registered direction. Report focal action, immediate utility,
next state, six-step focal utility, and population aggregates separately. An
action change without a downstream utility or state change supports prompt
sensitivity only.

**Unit and status.** Seed-checkpoint. The hash-bound intervention manifest is
implemented; exact restore, continuation, A/A aggregation, and action-bin
freezing are planned.

## Calibration, regret, and version diagnostics

- **Rule calibration error:** mean Brier score
  `mean_c (p_c - y_c)^2` over all oracle-labeled registered candidates. The
  score time and confidence field must be frozen. Missing predictions caused
  by a failed cell remain failures rather than invented probabilities.
- **Adaptation regret:** within each E0 scenario, the cumulative difference
  between the frozen oracle-policy utility and the evaluated-policy utility
  over a fixed post-switch horizon. Report the oracle schedule and horizon.
- **Version replacement accuracy:** proportion of switch-required rule
  families whose terminal active version matches the oracle-current version.
  No active version is incorrect for this metric; a failed cell remains a
  failure status.
- **Regime-switch delay:** completed steps from the known switch to first use of
  the oracle-current rule, right-censored at the scenario horizon.

**Unit and status.** E0 scenario family. All four are planned; none is produced
by the current deterministic lifecycle fixture.

## Provenance coverage

```text
provenance coverage = resolved finalized-episode links / all evidence links
```

Zero links yield `NA`, not perfect coverage. Coverage measures referential
integrity, not rule truth. Source-code provenance is a separate release gate
that verifies the pinned Git commit, annotated tag, blobs, file hashes, and
exact extracted inventory.

**Unit and status.** Rule ledger and release. Episode-link coverage and Git
source verification are implemented.

## Model competence and operational diagnostics

E3 reports proposal legality, known-rule application accuracy, action parse
failures, served-model identity, provider failures, token usage, and cost next
to the utility delta. Proposal failure must not be relabeled as actor-reasoning
failure. These diagnostics do not make a backbone-independent claim.

**Unit and status.** Registered model-call kind and seed. Planned; hosted calls
are disabled and no EGRM provider budget is authorized by the current release.

## Current implementation map

| Metric or gate | Current state |
|---|---|
| Admission precision and unsupported activation | Implemented for provider-free fixture |
| Evidence-event counts and episode-link coverage | Implemented for provider-free fixture |
| Right-censored retirement records | Implemented |
| Git source provenance and exact extraction inventory | Implemented |
| Harmful action-ledger exposure | Planned |
| Calibration, regret, version accuracy, and switch delay | Planned |
| Exact checkpoint continuation and matched A/A null | Planned |
| Six-step focal/population utility effects | Planned |
| Provider competence, usage, cost, and failure ledger | Planned; hosted calls disabled |
