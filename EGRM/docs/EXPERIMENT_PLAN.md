# Fresh EGRM experiment plan

The current non-dispatchable scientific design draft is
[`configs/scientific_contract_v1.json`](../configs/scientific_contract_v1.json).
It expands to an exact 105-cell inventory, but some arms are still planned or
manifest-only, its release binding is intentionally empty, and hosted calls
remain disabled. It must be implemented, reviewed, source-bound, and frozen as
a new contract before any scientific dispatch.

## E0: Known-truth RuleSwitch benchmark

Construct 12 scenario families with oracle-valid rules, regime switches, and
counterevidence. Replay one candidate stream through four policies:

1. full EGRM;
2. no post-proposal delay;
3. no retirement;
4. unverified-immediate.

Report admission precision, unsupported activation, Brier-style calibration,
adaptation regret, version replacement accuracy, provenance coverage, and
right-censored retirement latency. Candidate-generation failures stay in the
denominator. This benchmark establishes rule reliability separately from the
complex macro simulator.

The current extraction implements only `full-egrm` and
`unverified-immediate`. The `no-postproposal-delay` and `no-retirement` policy
arms are registered as planned work and must receive manipulation tests before
this design can be frozen or dispatched.

## E1: Fresh closed-loop verifier test

Use GPT-5.2, 4 agents, 12 months, five paired seeds, one frozen shock path, one
prompt/action schema, and five arms:

1. full EGRM without injected error;
2. unverified-immediate without injected error;
3. unsupported false candidate with evidence admission;
4. the false rule forced active under EGRM;
5. the same false rule forced active under the unverified policy.

The fixed false rule and injection time must be frozen before dispatch. Primary
outcomes are unsupported ever-active, selected harmful exposure, retirement
latency, and paired loss `U_no-error - U_error`. Wealth, Gini, and labor are
diagnostics.

Natural proposals are audited against the frozen scenario oracle or a
predeclared blinded adjudication protocol. The fixed erroneous candidate tests
policy behavior only and cannot establish that spontaneous LLM proposal errors
are mitigated.

The effectiveness go condition requires at least four of five complete pairs
and at least four of five seeds in which EGRM reduces both harmful exposure and
cumulative loss. Otherwise the effectiveness claim is withdrawn.

## E2: Checkpoint-matched continuation

Freeze one checkpoint per seed and branch only:

- matched A;
- matched B;
- erroneous forced-active rule with EGRM;
- the identical erroneous rule with unverified memory.

All agents continue for six steps from identical environment, model, decoding,
prompt, and RNG state. The effect must exceed the maximum matched A/A null and
one action bin, with at least four of five seeds in the same direction. An
action-only result is reported as prompt sensitivity.

## E3: Second-model diagnostic

Repeat the central four-branch E2 subset for three seeds with GPT-5.6 after a
model-specific capability gate. Report proposal competence, rule-application
competence, parse/provider failure, and utility delta separately. A 3/3
direction supports only a small-pilot, model-specific statement.

## Denominator and stopping

- Register all cells before any scientific outcome is read.
- Preserve parse, provider, budget, and integrity failures as ITT outcomes.
- Do not replace a failed seed, reduce the matrix, change reasoning, or select a
  cheaper model after seeing outcomes.
- Run E0 before E1, E1 before E2, and E2 before E3.
- Freeze a terminal no-go receipt when a gate fails.
- Do not use old FinEvo A-D, V2.11.x, macro tables, or deterministic legacy
  rules as EGRM scientific evidence.

## Cost gate

No EGRM hosted budget is authorized by this code-generation task. Before a
provider call, run a bounded capability/preflight sample, project
model-by-call-kind token p95 with reserve, freeze per-stage and total caps, and
obtain explicit EGRM spending authorization. A prior FinEvo budget is not
silently inherited.
