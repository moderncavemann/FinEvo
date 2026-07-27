# FinEvo V2.8 local-first mechanism pilot evidence report

- Contract: `finevo-pilot-v2.8` / `948eac04516dd2c292d68beb732f97532b13e667a180e8c2db16fbb927f92f19`
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
| not-applicable | Narrative channel shows controlled semantic response | not registered in the V2.8 core matrix | aggregate.json#/narrative | deferred-unregistered | make no V2.8 narrative or real-news-understanding claim |
| cross-lane | Backbone-independent improvement | prohibited pooled inference | aggregate.json#/cross_lane_policy | prohibited | report local and GPT-5.2 directions separately; never pool direction counts or use backbone-independent wording |
| all | Complete V2.8 preregistered ITT denominator | one terminal ledger row for every expanded contract cell | failure_ledger.json | supported | retain every failed, stopped, nonterminal, and missing cell |
| prerequisite-non-effect | V2.8 parent authority, fresh scripted q-ref, and imported Stage-0 inputs are complete prerequisites | 16 registered prerequisite cells; q-ref 0 hosted / 48 scripted diagnostic calls; 14 imported Stage-0 cells | aggregate.json#/prerequisites | no-go | parent authority, the fresh scripted q-ref, and the 14 imported Stage-0 cells are prerequisites/non-effect evidence; only fresh V2.8 A-D cells may support treatment-effect claims |
| parent-lineage | V2.7 remains an immutable complete-with-no-go package | exact parent manifest hash, terminal denominator, evidence commit, and merge commit | parent_evidence_reference.json | preserved | reference only; do not rewrite, reclassify, or import V2.7 rows into V2.8 effects |

## V2.8 amendment lineage and prerequisite boundary

- V2.7 remains an immutable `complete-with-no-go` package; namespace `evidence/current_v2/pilot-v2.7`, evidence commit `f15a26418264b5de31f53dbe7c46c1949761fcb6`, merge commit `e951aa865186a7c2e841316fc6bb08a716aeaf80`.
- V2.7 root cause: `qref-contract-cell-id-conflated-with-runner-execution-id` — imported source run identity is malformed.
- Parent denominator preserved: `211` cells / `{"complete": 1, "integrity-stopped": 210}`.
- Cumulative hosted budget before new V2.8 dispatch: `$3.212770875` / `184` hosted completions, under the `$500.0` hard cap.
- Fresh q-ref accounting: `0` hosted provider calls, `$0` hosted cost, and `48` scripted diagnostic calls.
- Prerequisite classification: parent authority, fresh scripted q-ref, and 14 imported Stage-0 cells are excluded from every A-D treatment-effect gate.
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
- Status counts: `{"complete": 1, "failed": 1, "integrity-stopped": 209}`
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
