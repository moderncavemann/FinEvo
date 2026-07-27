# FinEvo Pilot V2.7: Stage-0 evaluator/import amendment

## Status and scope

V2.7 is a **frozen evaluator/import contract and implementation candidate**,
not a completed release and not a scientific result. The expanded contract is
[`experiments/pilot_v2_7.yaml`](../experiments/pilot_v2_7.yaml), and its compact
source overlay is
[`experiments/pilot_v2_7_overlay.yaml`](../experiments/pilot_v2_7_overlay.yaml).
The frozen expanded-contract hash is
`938627d42ec8ec78e8424793797593736b79936b00813b81259af54e6df6779f`;
the compact overlay self-hash is
`bec6e6816a1d68aa932592796ff5aac8a693e9659dd64ba4eea6c740e25e799e`.
The release inventory is fixed at 954 collected tests, 174 tracked Python
sources, and six sealed manifests. Independent Linux/macOS CI on the eventual
merged release commit, the annotated tag, and the clean-tag launch attestation
remain separate gates.

V2.7 creates a fresh 211-cell ITT denominator, including 209 scientific cells,
under `experiment_results/pilot-v2.7/raw/`. It does not reopen, resume,
reclassify, or rewrite a V2.6 cell.

## Immutable V2.6 terminal no-go

V2.6 is terminal `complete-with-no-go`. All 211 registered cells remain
accounted for:

- 1 complete parent-import cell;
- 1 complete deterministic q-ref cell;
- 14 complete Stage-0 calibration cells; and
- 195 downstream `integrity-stopped` cells.

The 14 Stage-0 cells completed 672 local Llama calls. They used zero hosted
provider calls, zero hosted completions, and `$0` incremental API cost. No
Experiment A, B, C, or D treatment run was generated. Consequently, no A–D
treatment-effect outcome existed or was inspected.

The V2.6 denominator and its terminal status counts are immutable. The 16
complete cells cannot be relabeled as newly executed V2.7 results, and the 195
stopped cells cannot be reclassified as missing-at-random observations.

## Root cause and interpretation

Stage 0 deliberately used the constant `baseline-3pct` schedule: all 12
periods had a 3% rate and the phase label `baseline`. The Stage-0 selector
nevertheless routed those completed records through the general
shock/recovery analysis reader. That reader requires at least one
`pre-shock` period and later a `recovery` period, so it stopped with:

> `run has no pre-shock utility observations`

This is a reader-to-stage interface mismatch. It is not evidence of a model
capability failure, an invalid utility candidate, or a positive or negative
FinEvo treatment effect. Relabeling baseline periods as `pre-shock` or
injecting a synthetic recovery phase is forbidden because either operation
would alter the frozen Stage-0 data semantics.

## Observation disclosure

The Stage-0 artifacts were inspected after V2.6 became terminal. A diagnostic
audit using the phase-agnostic guardrail projection exposed the Stage-0
selection winner before the V2.7 contract could be frozen. V2.7 therefore
records
`stage0_calibration_selection_observed_before_amendment=true`; it does not
describe the Stage-0 retry as outcome-blind with respect to calibration
selection.

This disclosure does not expose any A–D treatment result: those outcomes were
never generated. The V2.7 amendment is outcome-blind only with respect to the
registered A–D treatment effects. Because the calibration winner is already
known, the pre-existing selector thresholds, tie-break order, candidate grid,
calibration seeds, model, actions, and downstream matrix must remain exactly
unchanged.

## Exact import and child wrapper

The zero-provider V2.7 import may admit exactly the 16 completed V2.6 cells:
the parent-import artifact, q-ref artifact, and 14 Stage-0 run artifacts. Before
admission, the importer must verify the frozen V2.6 contract and annotated tag,
peeled release commit, release attestation, run and budget ledgers, exact raw
file inventory, q-ref receipt, parent-authority chain, every Stage-0 manifest,
and all declared file/content hashes. The frozen V2.7 source manifest binds that
complete package with file hash
`ee0ef62f5dcde9fc820aef6d23d1ce5a8c5bca7b9f20486bf42233f18763a1c8`
and content hash
`f195661d01d0aa6742d9e2f2658b6b1acb38715ddbd43e4e5fd375309d78dbe4`.

Verified V2.6 bytes remain immutable parent sources. A child wrapper may
reseal verified identities to the eventual V2.7 contract, annotated tag, and
release HEAD, but it may not rewrite a V2.6 manifest, completion, receipt, or
ledger. Provider construction during import is forbidden. Provider redispatch
for all 16 imported cells is forbidden. Reusing their decoded completions
outside the imported Stage-0 calibration scope is also forbidden.

Any absent file, hash mismatch, schema mismatch, route mismatch, or failed
parent-authority check stops the fresh V2.7 denominator before downstream
dispatch. It cannot be bypassed by a permissive compatibility mode.

## Phase-agnostic Stage-0 reader

The correction is a dedicated `finevo-pilot-stage0-analysis-v1` reader with
scope `stage0-baseline-calibration`. It reads only:

- `actions`;
- `utility_ledger`; and
- `errors`.

It supplies only the preregistered selector inputs:

- maximum absolute budget residual;
- clipping count;
- ceiling-labor rate;
- zero-labor rate;
- interior-labor rate;
- interior-consumption rate; and
- median labor-disutility to consumption-utility ratio.

The reader is phase-agnostic. It requires no pre-shock, shock, or recovery
phase and computes no baseline, recovery time, utility-deficit AUC, route
effect, or shock/recovery treatment metric. The general A–D analysis reader
remains unchanged and must still enforce the registered shock schedule.

The imported Stage-0 records must be summarized, gated, and selected again
through this dedicated reader inside the hash-bound V2.7 workflow. A
diagnostic result observed before freeze is not itself a release-authorized
selection artifact.

## Frozen selector and matrix invariants

The following values may not be changed in response to the observed
calibration audit:

- calibration seeds: `1942013315` and `760687867`;
- candidates, in declaration order: `center`, `psi-1`, `psi-4`, `nu-0.5`,
  `nu-2`, `q0-0.5x`, and `q0-2x`;
- budget residual at most `1e-8`;
- clipping count exactly `0`;
- ceiling-labor rate at most `0.50`;
- zero-labor rate at most `0.25`;
- interior-labor rate at least `0.50`;
- interior-consumption rate at least `0.75`; and
- median component ratio within `[0.5, 2.0]`.

The selector remains
`guardrail-then-registered-tiebreak-v1`, with this exact order:

1. maximize mean interior action coverage;
2. minimize component-balance log distance from one;
3. minimize normalized center distance; and
4. use declaration order only for an exact remaining tie.

V2.7 also preserves the V2.6 models, provider profiles, five main seeds, arms,
shock, utility grid, stop/go rules, and fixed local-then-hosted `C-A-D-B`
order. Failed seeds cannot be replaced. The matrix cannot be shrunk, expanded,
or reordered; no extra candidate, arm, model, narrative, retry, or cheaper
reasoning configuration can be introduced after observing Stage 0.

## Budget boundary

The hosted-API hard cap remains `$500`, with at most 7,500 hosted completions
and 5 GB of storage. The cumulative inherited debit remains `$3.212770875` and
184 hosted completions. V2.6 adds 672 local calls but `$0` hosted cost and zero
hosted completions. The `$1` manual reserve remains unavailable for automatic
dispatch.

The correction does not create a new budget or erase earlier debits. Before
any new provider call, the complete remaining matrix must pass the existing
model-by-call-role p95 projection plus 25% reserve. An over-budget projection
produces a no-go receipt; it does not authorize fewer seeds, fewer arms, lower
reasoning, a substitute model, or silent use of the manual reserve.

## Execution order

V2.7 may progress only in this order:

1. Bind the exact immutable V2.6 terminal source package, implementation
   inventory, and release inventory.
2. Pass the full local tests, compile checks, contract expansion checks,
   manifest rehash, and secret scan; replace every draft placeholder with its
   concrete binding and freeze the contract without provider dispatch.
3. Pass independent Linux/macOS CI on the frozen tree, merge that exact tree,
   verify main-branch CI, create a new annotated `pilot-v2.7-science` tag, and
   produce a clean-tag launch attestation.
4. Run the zero-provider exact import for the 16 V2.6 cells.
5. Run the dedicated offline Stage-0 reader, guardrails, and frozen selector.
6. Recompute the full remaining budget projection without changing the matrix.
7. If and only if every prior gate passes, execute local C→A→D→B, followed by
   GPT-5.2 C→A→D→B.
8. Publish the complete denominator, failures, budget ledger, raw paired
   deltas, aggregate evidence, and negative results without seed replacement
   or post-hoc reruns.

Steps 1–2 are satisfied by the frozen contract candidate. Steps 3–8 remain
pending; freezing alone does not authorize provider dispatch.

Failure at any import, integrity, selection, or budget gate terminalizes the
fresh denominator and yields another auditable no-go. It does not permit
falling through to hosted dispatch.

## Claim boundaries

- V2.6 remains operational/calibration provenance, not A–D effectiveness
  evidence.
- Import success establishes traceability and exact reuse only; it does not
  establish a memory, retrieval, verifier, causal, or robustness effect.
- Stage-0 selection establishes only that one preregistered utility profile
  passes the fixed calibration procedure. It cannot support a FinEvo
  performance claim.
- The current V2.7 artifacts freeze the preregistration and implementation
  inventory only. They do not support wording such as “V2.7 passed,” “the
  pilot confirms the paper,” or any A–D effectiveness conclusion.
- A–D claims require the corresponding fresh, complete paired V2.7 evidence
  and their preregistered direction/effect gates. A failed gate must narrow the
  associated paper claim.
- The 4-agent × 12-month micro-pilot cannot establish backbone independence,
  full-scale validity, real-news understanding, or equivalence to the original
  100-agent × 240-month experiments.

Until merged-commit Linux/macOS CI, the annotated tag, and the clean-tag launch
attestation exist, no V2.7 scientific dispatch is authorized.
