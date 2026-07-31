# FinEvo V2.10.2 local-first mechanism pilot evidence report

- Contract: `finevo-pilot-v2.10.2` / `b8de8cfb2560d894dad65d68df8ae9126527d12d3807bef045fa52f5e9d4159e`
- Publication status: `complete-with-no-go`
- Registered denominator: `211` cells
- Matrix order: `C -> A -> D -> B` in each lane.
- Local and GPT-5.2 directions are never pooled.
- Narrative intervention: `deferred-unregistered`.

## Claim -> metric -> artifact

| Lane | Claim | Metric | Artifact | Status | Boundary |
|---|---|---|---|---|---|
| local | Evidence grounding improves erroneous-rule reliability | false activation, harmful exposure, and cumulative utility-loss directions | aggregate.json#/lanes/local/gates/experiment-c | no-go | withdraw or narrow the rule-reliability claim |
| local | M1 retrieval contributes beyond regime prompting | full minus prompt-only shock+recovery discounted utility | aggregate.json#/lanes/local/gates/experiment-a | supported | retain the narrow environment-paired, decoding-unmatched retrieval-effect claim |
| local | A focal memory/error pulse changes the matched six-step continuation | matched-null- and action-bin-qualified continuation deltas | aggregate.json#/lanes/local/gates/experiment-d | supported | claim only named one-decision focal memory-pulse or error-rule intervention effects on the matched six-step downstream continuation; classify action-only changes as prompt sensitivity |
| local | Registered memory architectures can be compared descriptively | seed-level utility, action, retrieval, proposal, and lifecycle summaries | aggregate.json#/lanes/local/gates/experiment-b | descriptive-complete | report the registered architecture comparison descriptively; do not select a winner by wealth |
| gpt52 | Evidence grounding improves erroneous-rule reliability | false activation, harmful exposure, and cumulative utility-loss directions | aggregate.json#/lanes/gpt52/gates/experiment-c | no-go | withdraw or narrow the rule-reliability claim |
| gpt52 | M1 retrieval contributes beyond regime prompting | full minus prompt-only shock+recovery discounted utility | aggregate.json#/lanes/gpt52/gates/experiment-a | no-go | retain route traceability only |
| gpt52 | A focal memory/error pulse changes the matched six-step continuation | matched-null- and action-bin-qualified continuation deltas | aggregate.json#/lanes/gpt52/gates/experiment-d | no-go | do not claim a closed-loop continuation effect |
| gpt52 | Registered memory architectures can be compared descriptively | seed-level utility, action, retrieval, proposal, and lifecycle summaries | aggregate.json#/lanes/gpt52/gates/experiment-b | no-go | report the incomplete architecture denominator without a winner |
| cross-lane | experiment-c direction appears in two backbone micro-pilots | separate lane-level 4/5 gate, mechanism status, and primary-direction agreement | aggregate.json#/cross_lane_mechanism_comparison/experiment-c | inconclusive | cross-backbone mechanism direction is inconclusive; do not pool seed directions |
| cross-lane | experiment-a direction appears in two backbone micro-pilots | separate lane-level 4/5 gate, mechanism status, and primary-direction agreement | aggregate.json#/cross_lane_mechanism_comparison/experiment-a | inconclusive | cross-backbone mechanism direction is inconclusive; do not pool seed directions |
| cross-lane | experiment-d direction appears in two backbone micro-pilots | separate lane-level 4/5 gate, mechanism status, and primary-direction agreement | aggregate.json#/cross_lane_mechanism_comparison/experiment-d | inconclusive | cross-backbone mechanism direction is inconclusive; do not pool seed directions |
| cross-lane | experiment-b direction appears in two backbone micro-pilots | separate lane-level 4/5 gate, mechanism status, and primary-direction agreement | aggregate.json#/cross_lane_mechanism_comparison/experiment-b | inconclusive | cross-backbone mechanism direction is inconclusive; do not pool seed directions |
| not-applicable | Narrative channel shows controlled semantic response | not registered in the V2.10.2 core matrix | aggregate.json#/narrative | deferred-unregistered | make no V2.10.2 narrative or real-news-understanding claim |
| cross-lane | Backbone-independent improvement | prohibited pooled inference | aggregate.json#/cross_lane_policy | prohibited | report local and GPT-5.2 directions separately; never pool direction counts or use backbone-independent wording |
| all | Complete V2.10.2 preregistered ITT denominator | one terminal ledger row for every expanded contract cell | failure_ledger.json | supported | retain every failed, stopped, nonterminal, and missing cell |
| local | availability of the registered zero-API Experiment C rule sensitivity for this lane | available=false, provider_calls=0, and recorded reason | aggregate.json#/experiment_c_rule_sensitivities/local | no-go | descriptive sensitivity over natural proposals only; it cannot rescue a failed Experiment C effectiveness contrast |
| gpt52 | availability of the registered zero-API Experiment C rule sensitivity for this lane | available=false, provider_calls=0, and recorded reason | aggregate.json#/experiment_c_rule_sensitivities/gpt52 | no-go | descriptive sensitivity over natural proposals only; it cannot rescue a failed Experiment C effectiveness contrast |
| prerequisite-non-effect | V2.10.2 imports the exact parent, q-ref, and Stage-0 prerequisites without provider dispatch | 16 registered imported prerequisite cells; 0 provider calls during import; 195 fresh A-D cells required | aggregate.json#/prerequisites | complete | the 16 V2.9-derived parent, q-ref, and Stage-0 cells are reverified through the immutable V2.10.1 no-go lineage and excluded from all V2.10.2 A-D gates; all 195 V2.10.2 A-D cells, including 10 offline candidate-admission cells, must be fresh |
| parent-lineage | V2.10.1 remains an immutable complete-with-no-go package | exact parent manifest hash, terminal denominator, evidence commit, and merge commit | parent_evidence_reference.json | preserved | reference only; do not rewrite, reclassify, or import V2.10.1 rows into V2.10.2 effects |
| historical-model-boundary | Historical GPT-5.6 diagnostic boundary | V2.3 capability/preflight plus six registered directional cells | aggregate.json#/historical_model_boundaries/gpt56_diagnostic | not-evaluated | uncalibrated historical diagnostic only; no directional replication, cross-model effectiveness, model-choice superiority, or backbone-independent claim; budget-stopped is not a negative effect result |

## V2.10.2 amendment lineage and prerequisite boundary

- V2.10.1 remains an immutable `complete-with-no-go` package; namespace `evidence/current_v2/pilot-v2.10.1`, evidence commit `b7001a0174d1a420b592cd68976a3ca8388cb748`, merge commit `a730d0d97118a6d5cf79df66cb97cb1a32c510d9`.
- V2.10.1 root cause: `observed-p95-consumer-schema-dispatch-gap` — source-backed observed p95 receipt verification failed: observed-p95 receipt top-level shape or schema drifted.
- Parent denominator preserved: `211` cells / `{"complete": 26, "failed": 185}`.
- Cumulative hosted budget before new V2.10.2 dispatch: `$3.212770875` / `184` hosted completions, under the `$500.0` hard cap.
- Parent outcome boundary: V2.10.1 generated no actor performance treatment-effect outcome. Its 10 offline candidate-admission metrics were observed and inspected but remain immutable descriptive parent evidence, not V2.10.2 effects.
- Prerequisite classification: The 16 schema- and hash-verified V2.9 parent/q-ref/Stage-0 prerequisites are reverified through the immutable V2.10.1 no-go with 0 provider calls during V2.10.2 import. They are excluded from all A-D gates; all 195 V2.10.2 A-D cells, including 10 offline candidate-admission cells, are fresh.
- All prerequisites complete: `true`.

## Frozen model choice and historical GPT-5.6 boundary

- Classification: `frozen historical diagnostic only`; this is not a V2.10.2 treatment lane.
- GPT-5.2 remains the V2.10.2 primary because the `gpt52_main` profile was frozen before dispatch with requested model `gpt-5.2-2025-12-11`; replacing it inside this retry would be a post-registration model substitution.
- GPT-5.6 was not ignored: its frozen V2.3 diagnostic passed `30/30` capability tasks and accounted for `16/16` closed-loop preflight calls.
- Effect boundary: `6/6 budget-stopped` registered directional cells, no paired delta, no matched A/A null, no usable paired seed, and no directional replication.
- V2.10.2 status: GPT-5.6 was not redispatched; it contributes `0` current registered cells and `0` current effect rows.
- Interpretation: capability/preflight pass is not effectiveness evidence, and the budget stop is not a negative effect result. The admissible next step is a separate prospective registered GPT-5.6 replication lane.
- Claim boundary: no cross-model effectiveness, model-choice superiority, or backbone-independent claim.

## gpt52 lane

- Model profile: `gpt52_main`
- 4/5 paired matrix complete: `false`
- `experiment-c`: 0/5 complete paired seeds; gate `no-go`; claim action: withdraw or narrow the rule-reliability claim.
- `experiment-a`: 0/5 complete paired seeds; gate `no-go`; claim action: retain route traceability only.
- `experiment-d`: 0/5 complete paired seeds; gate `no-go`; claim action: do not claim a closed-loop continuation effect.
- `experiment-b`: 0/5 complete paired seeds; gate `no-go`; claim action: report the incomplete architecture denominator without a winner.

## local lane

- Model profile: `llama33_local_controlled`
- 4/5 paired matrix complete: `true`
- `experiment-c`: 4/5 complete paired seeds; gate `no-go`; claim action: withdraw or narrow the rule-reliability claim.
- `experiment-a`: 4/5 complete paired seeds; gate `supported`; claim action: retain the narrow environment-paired, decoding-unmatched retrieval-effect claim.
- `experiment-d`: 5/5 complete paired seeds; gate `supported`; claim action: claim only named one-decision focal memory-pulse or error-rule intervention effects on the matched six-step downstream continuation; classify action-only changes as prompt sensitivity.
- `experiment-b`: 5/5 complete paired seeds; gate `descriptive-complete`; claim action: report the registered architecture comparison descriptively; do not select a winner by wealth.

## Cross-lane mechanism comparison

| Stage | Local status / 4-of-5 | GPT-5.2 status / 4-of-5 | Direction agreement | Classification | Boundary |
|---|---|---|---|---|---|
| `experiment-c` | `no-go` / `true` | `no-go` / `false` | `false` | `inconclusive` | cross-backbone mechanism direction is inconclusive; do not pool seed directions |
| `experiment-a` | `supported` / `true` | `no-go` / `false` | `false` | `inconclusive` | cross-backbone mechanism direction is inconclusive; do not pool seed directions |
| `experiment-d` | `supported` / `true` | `no-go` / `false` | `false` | `inconclusive` | cross-backbone mechanism direction is inconclusive; do not pool seed directions |
| `experiment-b` | `descriptive-complete` / `true` | `no-go` / `false` | `false` | `inconclusive` | cross-backbone mechanism direction is inconclusive; do not pool seed directions |

## Denominator, failures, and budget

- ITT denominator pass: `true`
- Status counts: `{"complete": 126, "failed": 85}`
- Budget control: `true`
- Every failed, stopped, nonterminal, and missing cell remains in `failure_ledger.json` and the aggregate rows.

## Explicit claim narrowing

- `local/experiment-c`: the preregistered mechanism gate was not supported; withdraw or narrow the rule-reliability claim.
- `gpt52/experiment-c`: 0/5 complete paired seeds; the registered minimum is 4/5; denominator/failure report only; no effectiveness claim.
- `gpt52/experiment-a`: 0/5 complete paired seeds; the registered minimum is 4/5; denominator/failure report only; no effectiveness claim.
- `gpt52/experiment-d`: 0/5 complete paired seeds; the registered minimum is 4/5; denominator/failure report only; no effectiveness claim.
- `gpt52/experiment-b`: 0/5 complete paired seeds; the registered minimum is 4/5; denominator/failure report only; no effectiveness claim.
- `cross-lane/experiment-c`: the two independently gated lanes do not establish the same registered primary direction; cross-backbone mechanism direction is inconclusive; do not pool seed directions.
- `cross-lane/experiment-a`: the two independently gated lanes do not establish the same registered primary direction; cross-backbone mechanism direction is inconclusive; do not pool seed directions.
- `cross-lane/experiment-d`: the two independently gated lanes do not establish the same registered primary direction; cross-backbone mechanism direction is inconclusive; do not pool seed directions.
- `cross-lane/experiment-b`: the two independently gated lanes do not establish the same registered primary direction; cross-backbone mechanism direction is inconclusive; do not pool seed directions.
- `narrative`: narrative intervention is deferred and unregistered; no narrative or real-news-understanding claim.
- `cross-lane`: the local and GPT lanes are separate replications; report each lane's seed directions separately; never pool them.
- `local/experiment-c-sensitivity`: local-experiment-c ITT cells are not all complete and scientifically eligible; registered descriptive sensitivity unavailable; do not cite or reconstruct an absent sensitivity artifact.
- `gpt52/experiment-c-sensitivity`: experiment-c ITT cells are not all complete and scientifically eligible; registered descriptive sensitivity unavailable; do not cite or reconstruct an absent sensitivity artifact.
- `historical-model/gpt56_diagnostic`: all 6/6 V2.3 directional cells were budget-stopped; V2.10.2 did not redispatch GPT-5.6; capability/preflight pass is not effectiveness evidence or a negative effect result; use a prospective registered replication.
