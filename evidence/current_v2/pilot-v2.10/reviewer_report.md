# FinEvo V2.10 local-first mechanism pilot evidence report

- Contract: `finevo-pilot-v2.10` / `d1b54c14d016c2b157db9e334d054ab9c7e86371d3fb9662a95fb94e50ce964b`
- Publication status: `complete-with-no-go`
- Registered denominator: `211` cells
- Matrix order: `C -> A -> D -> B` in each lane.
- Local and GPT-5.2 directions are never pooled.
- Narrative intervention: `deferred-unregistered`.

## Claim -> metric -> artifact

| Lane | Claim | Metric | Artifact | Status | Boundary |
|---|---|---|---|---|---|
| local | Evidence grounding improves erroneous-rule reliability | false activation, harmful exposure, and cumulative utility-loss directions | aggregate.json#/lanes/local/gates/experiment-c | no-go | withdraw or narrow the rule-reliability claim |
| local | M1 retrieval contributes beyond regime prompting | full minus prompt-only shock+recovery discounted utility | aggregate.json#/lanes/local/gates/experiment-a | no-go | retain route traceability only |
| local | A focal memory/error pulse changes the matched six-step continuation | matched-null- and action-bin-qualified continuation deltas | aggregate.json#/lanes/local/gates/experiment-d | no-go | do not claim a closed-loop continuation effect |
| local | Registered memory architectures can be compared descriptively | seed-level utility, action, retrieval, proposal, and lifecycle summaries | aggregate.json#/lanes/local/gates/experiment-b | no-go | report the incomplete architecture denominator without a winner |
| gpt52 | Evidence grounding improves erroneous-rule reliability | false activation, harmful exposure, and cumulative utility-loss directions | aggregate.json#/lanes/gpt52/gates/experiment-c | no-go | withdraw or narrow the rule-reliability claim |
| gpt52 | M1 retrieval contributes beyond regime prompting | full minus prompt-only shock+recovery discounted utility | aggregate.json#/lanes/gpt52/gates/experiment-a | no-go | retain route traceability only |
| gpt52 | A focal memory/error pulse changes the matched six-step continuation | matched-null- and action-bin-qualified continuation deltas | aggregate.json#/lanes/gpt52/gates/experiment-d | no-go | do not claim a closed-loop continuation effect |
| gpt52 | Registered memory architectures can be compared descriptively | seed-level utility, action, retrieval, proposal, and lifecycle summaries | aggregate.json#/lanes/gpt52/gates/experiment-b | no-go | report the incomplete architecture denominator without a winner |
| cross-lane | experiment-c direction appears in two backbone micro-pilots | separate lane-level 4/5 gate, mechanism status, and primary-direction agreement | aggregate.json#/cross_lane_mechanism_comparison/experiment-c | inconclusive | cross-backbone mechanism direction is inconclusive; do not pool seed directions |
| cross-lane | experiment-a direction appears in two backbone micro-pilots | separate lane-level 4/5 gate, mechanism status, and primary-direction agreement | aggregate.json#/cross_lane_mechanism_comparison/experiment-a | inconclusive | cross-backbone mechanism direction is inconclusive; do not pool seed directions |
| cross-lane | experiment-d direction appears in two backbone micro-pilots | separate lane-level 4/5 gate, mechanism status, and primary-direction agreement | aggregate.json#/cross_lane_mechanism_comparison/experiment-d | inconclusive | cross-backbone mechanism direction is inconclusive; do not pool seed directions |
| cross-lane | experiment-b direction appears in two backbone micro-pilots | separate lane-level 4/5 gate, mechanism status, and primary-direction agreement | aggregate.json#/cross_lane_mechanism_comparison/experiment-b | inconclusive | cross-backbone mechanism direction is inconclusive; do not pool seed directions |
| not-applicable | Narrative channel shows controlled semantic response | not registered in the V2.10 core matrix | aggregate.json#/narrative | deferred-unregistered | make no V2.10 narrative or real-news-understanding claim |
| cross-lane | Backbone-independent improvement | prohibited pooled inference | aggregate.json#/cross_lane_policy | prohibited | report local and GPT-5.2 directions separately; never pool direction counts or use backbone-independent wording |
| all | Complete V2.10 preregistered ITT denominator | one terminal ledger row for every expanded contract cell | failure_ledger.json | supported | retain every failed, stopped, nonterminal, and missing cell |
| local | registered zero-API Experiment C rule sensitivity is available for this lane | 3 alternative-success weights x 3 outcome definitions replayed from five full-control seeds | local_experiment_c_rule_sensitivity.json | no-go | descriptive sensitivity over natural proposals only; it cannot rescue a failed Experiment C effectiveness contrast |
| gpt52 | registered zero-API Experiment C rule sensitivity is available for this lane | 3 alternative-success weights x 3 outcome definitions replayed from five full-control seeds | experiment_c_rule_sensitivity.json | no-go | descriptive sensitivity over natural proposals only; it cannot rescue a failed Experiment C effectiveness contrast |
| prerequisite-non-effect | V2.10 imports the exact parent, q-ref, and Stage-0 prerequisites without provider dispatch | 16 registered imported prerequisite cells; 0 provider calls during import; 195 fresh A-D cells required | aggregate.json#/prerequisites | no-go | the 16 V2.9-derived parent, q-ref, and Stage-0 cells are hash-verified V2.10 prerequisites only; all 195 V2.10 A-D cells, including candidate-admission cells, must be fresh |
| parent-lineage | V2.9 remains an immutable complete-with-no-go package | exact parent manifest hash, terminal denominator, evidence commit, and merge commit | parent_evidence_reference.json | preserved | reference only; do not rewrite, reclassify, or import V2.9 rows into V2.10 effects |

## V2.10 amendment lineage and prerequisite boundary

- V2.9 remains an immutable `complete-with-no-go` package; namespace `evidence/current_v2/pilot-v2.9`, evidence commit `51525614e138e5b7ac498d15b409048d5110b753`, merge commit `08fcbc0dd9319fcc86c3f4e812c3db504a0c5a17`.
- V2.9 root cause: `imported-p95-runner-binding-shape-mismatch` — The imported authority producer returned nested receipt identity fields while the runner consumer dereferenced the legacy flat names..
- Parent denominator preserved: `211` cells / `{"complete": 26, "failed": 185}`.
- Cumulative hosted budget before new V2.10 dispatch: `$3.212770875` / `184` hosted completions, under the `$500.0` hard cap.
- Parent outcome boundary: V2.9 generated no actor treatment-effect outcome; its 10 offline candidate-admission outcomes are disclosed but are not V2.10 effect evidence.
- Prerequisite classification: The 16 hash-verified V2.9 parent/q-ref/Stage-0 prerequisites used 0 provider calls during V2.10 import and are excluded from all A-D gates; every one of the 195 V2.10 A-D cells is fresh.
- All prerequisites complete: `false`.

## gpt52 lane

- Model profile: `gpt52_main`
- 4/5 paired matrix complete: `false`
- `experiment-c`: 0/5 complete paired seeds; gate `no-go`; claim action: withdraw or narrow the rule-reliability claim.
- `experiment-a`: 0/5 complete paired seeds; gate `no-go`; claim action: retain route traceability only.
- `experiment-d`: 0/5 complete paired seeds; gate `no-go`; claim action: do not claim a closed-loop continuation effect.
- `experiment-b`: 0/5 complete paired seeds; gate `no-go`; claim action: report the incomplete architecture denominator without a winner.

## local lane

- Model profile: `llama33_local_controlled`
- 4/5 paired matrix complete: `false`
- `experiment-c`: 0/5 complete paired seeds; gate `no-go`; claim action: withdraw or narrow the rule-reliability claim.
- `experiment-a`: 0/5 complete paired seeds; gate `no-go`; claim action: retain route traceability only.
- `experiment-d`: 0/5 complete paired seeds; gate `no-go`; claim action: do not claim a closed-loop continuation effect.
- `experiment-b`: 0/5 complete paired seeds; gate `no-go`; claim action: report the incomplete architecture denominator without a winner.

## Cross-lane mechanism comparison

| Stage | Local status / 4-of-5 | GPT-5.2 status / 4-of-5 | Direction agreement | Classification | Boundary |
|---|---|---|---|---|---|
| `experiment-c` | `no-go` / `false` | `no-go` / `false` | `false` | `inconclusive` | cross-backbone mechanism direction is inconclusive; do not pool seed directions |
| `experiment-a` | `no-go` / `false` | `no-go` / `false` | `false` | `inconclusive` | cross-backbone mechanism direction is inconclusive; do not pool seed directions |
| `experiment-d` | `no-go` / `false` | `no-go` / `false` | `false` | `inconclusive` | cross-backbone mechanism direction is inconclusive; do not pool seed directions |
| `experiment-b` | `no-go` / `false` | `no-go` / `false` | `false` | `inconclusive` | cross-backbone mechanism direction is inconclusive; do not pool seed directions |

## Denominator, failures, and budget

- ITT denominator pass: `true`
- Status counts: `{"complete": 1, "integrity-stopped": 210}`
- Budget control: `true`
- Every failed, stopped, nonterminal, and missing cell remains in `failure_ledger.json` and the aggregate rows.

## Explicit claim narrowing

- `local/experiment-c`: 0/5 complete paired seeds; the registered minimum is 4/5; denominator/failure report only; no effectiveness claim.
- `local/experiment-a`: 0/5 complete paired seeds; the registered minimum is 4/5; denominator/failure report only; no effectiveness claim.
- `local/experiment-d`: 0/5 complete paired seeds; the registered minimum is 4/5; denominator/failure report only; no effectiveness claim.
- `local/experiment-b`: 0/5 complete paired seeds; the registered minimum is 4/5; denominator/failure report only; no effectiveness claim.
- `gpt52/experiment-c`: 0/5 complete paired seeds; the registered minimum is 4/5; denominator/failure report only; no effectiveness claim.
- `gpt52/experiment-a`: 0/5 complete paired seeds; the registered minimum is 4/5; denominator/failure report only; no effectiveness claim.
- `gpt52/experiment-d`: 0/5 complete paired seeds; the registered minimum is 4/5; denominator/failure report only; no effectiveness claim.
- `gpt52/experiment-b`: 0/5 complete paired seeds; the registered minimum is 4/5; denominator/failure report only; no effectiveness claim.
- `release-stage0-budget`: release, Stage-0, or budget controls did not all pass; complete-with-no-go; do not report scientific effectiveness.
- `cross-lane/experiment-c`: the two independently gated lanes do not establish the same registered primary direction; cross-backbone mechanism direction is inconclusive; do not pool seed directions.
- `cross-lane/experiment-a`: the two independently gated lanes do not establish the same registered primary direction; cross-backbone mechanism direction is inconclusive; do not pool seed directions.
- `cross-lane/experiment-d`: the two independently gated lanes do not establish the same registered primary direction; cross-backbone mechanism direction is inconclusive; do not pool seed directions.
- `cross-lane/experiment-b`: the two independently gated lanes do not establish the same registered primary direction; cross-backbone mechanism direction is inconclusive; do not pool seed directions.
- `narrative`: narrative intervention is deferred and unregistered; no narrative or real-news-understanding claim.
- `cross-lane`: the local and GPT lanes are separate replications; report each lane's seed directions separately; never pool them.
