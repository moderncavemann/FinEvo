# FinEvo Pilot V2.8: q-ref contract-cell identity amendment

## Status and scope

V2.8 is a **frozen prospective operational amendment**, not a scientific result. Its
expanded contract is
[`experiments/pilot_v2_8.yaml`](../experiments/pilot_v2_8.yaml), its compact
source overlay is
[`experiments/pilot_v2_8_overlay.yaml`](../experiments/pilot_v2_8_overlay.yaml),
and its source-lineage manifest is
[`experiments/pilot_v2_8_source_manifest.json`](../experiments/pilot_v2_8_source_manifest.json).
Contract freeze alone does not authorize dispatch: the release gates must pass
and the exact clean tag must receive a launch attestation.

V2.8 creates a fresh 211-cell ITT denominator under
`experiment_results/pilot-v2.8/raw/`: one parent-import cell, one freshly
regenerated q-ref cell, 14 exactly imported Stage-0 cells, and 195 fresh A–D
cells. The latter 209 cells retain the registered scientific matrix. V2.8 does
not resume, relabel, delete, or rewrite a V2.7 cell.

## Immutable V2.7 complete-with-no-go

V2.7 is terminal `complete-with-no-go`. All 211 registered cells remain
accounted for:

- 1 complete parent-import cell; and
- 210 `integrity-stopped` cells.

The failed workflow produced no V2.7 q-ref resolution artifact, no completed
V2.7 Stage-0 calibration cell, and no Experiment A, B, C, or D treatment
output. No A–D treatment-effect outcome was generated or inspected. V2.7 added
zero hosted provider calls, zero hosted completions, and `$0` incremental API
cost.

Those counts and artifacts are immutable. The one completed parent cell cannot
be presented as a scientific result, and the 210 stopped cells cannot be
reclassified as missing observations or silently retried inside the V2.7
denominator.

## Root cause and interpretation

The q-ref source configuration carried the short runner execution identity
`q-ref-resolution-s2010922376`. The V2.7 contract instead required the exact
cell identity
`finevo-pilot-v2.7--q-ref-resolution--qref_scripted--qref-scripted--none--provider-preflight-default--s2010922376`.
The importer correctly failed closed when the source configuration identity
did not equal the registered contract-cell identity, reporting:

> `imported source run identity is malformed`

This is a q-ref identity/interface failure. It is not evidence of model
capability failure, an invalid utility calibration, a budget failure, or a
positive or negative FinEvo treatment effect. Weakening the identity check or
editing the historical source configuration would destroy the provenance
invariant and is forbidden.

## Fresh q-ref regeneration

V2.8 regenerates q-ref under the exact current contract-cell `run_id`. The
directory name, serialized run configuration, run-spec provenance, resolution
source, and receipts must all bind to that same identity.

The regeneration uses the unchanged deterministic action schedule and
`ScriptedDiagnosticProvider` for exactly 48 scripted diagnostic calls. It
constructs no hosted provider, makes zero hosted calls, and incurs `$0` hosted
cost. The historical q-ref result is not imported or reused.

The regenerated action, utility-ledger, shock, summary, environment-source,
and q-ref scalar records must be exactly equivalent to the verified nested
V2.6 source core, except for the intentionally fresh contract/run identity and
its dependent hashes. A hash-bound equivalence receipt records that comparison.
Any missing record, semantic difference, call-count difference, identity
mismatch, or failed source verification stops the new denominator before
downstream dispatch.

## Exact Stage-0 import

V2.8 may import exactly the 14 completed V2.6 Stage-0 calibration cells exposed
through the immutable V2.7 nested snapshot at
`parent-import/v2_6_raw_snapshot`. It may not import a V2.7 Stage-0 result,
because none exists, and it may not redispatch the 14 source cells.

Before admission, the importer must verify the V2.7 terminal evidence and
release lineage, the nested V2.6 contract and tag, the complete nested
inventory, all Stage-0 manifests and receipts, and every declared file and
content hash. Verified V2.6 bytes remain immutable. A V2.8 child wrapper may
reseal their verified identities to the current contract, tag, and release
HEAD, but it may not rewrite a source artifact or reuse its decoded completion
outside Stage-0 calibration. Provider construction during import is forbidden.

The dedicated phase-agnostic Stage-0 reader, fixed guardrails, candidate grid,
seeds, and registered tie-break order remain unchanged. Import and selection
success establish only prerequisite/calibration traceability; they are not
FinEvo effectiveness evidence.

## Frozen matrix and observation boundary

V2.8 preserves the registered models, provider profiles, seeds, arms, shock,
utility grid, metrics, stop/go rules, p95 authority, and full local-first
execution order:

1. local Llama C → A → D → B; then
2. GPT-5.2 C → A → D → B.

All 195 A–D cells require fresh V2.8 execution. No A–D decoded completion may
be imported. Failed seeds cannot be replaced, and the matrix cannot be
shrunk, expanded, reordered, or moved to a cheaper model or reasoning mode in
response to a result or cost projection.

The q-ref identity failure was observed before this amendment. Stage-0
guardrail outputs and the selected calibration candidate may also have been
observed in earlier terminal workflows. V2.8 therefore does not claim global
outcome blindness. It is outcome-blind only with respect to A–D treatment
effects, because no such V2.7 output existed. Calibration thresholds, candidate
profiles, seeds, model/actions, and tie-break order remain fixed despite the
prior observation.

Only fresh V2.8 A–D cells may enter V2.8 effect aggregation. Q-ref and Stage 0
are prerequisite evidence only. Every registered cell, parse failure, provider
failure, budget stop, and integrity stop remains in the ITT denominator.

## Budget boundary

The cumulative hosted-API hard cap is `$500`, with the unchanged limits of
7,500 hosted completions and 5,000,000,000 storage bytes. Before any new hosted
dispatch, V2.8 inherits and debits:

- `$3.212770875` prior hosted cost;
- 184 prior hosted completions; and
- 32,158,175 prior storage bytes.

V2.7 contributes `$0`, zero hosted completions, and zero hosted provider calls
to that debit. The `$1` manual reserve remains unavailable for automatic use.
Unknown pricing stops before dispatch.

After prerequisite import, q-ref, and Stage-0 gates, the whole remaining
registered matrix must fit the model-by-call-role p95 projection plus the
frozen 25% reserve. A failed projection produces a complete no-go receipt. It
does not authorize fewer cells, seed replacement, reduced reasoning, a
substitute model, or silent use of the manual reserve.

## Release and launch gates

V2.8 may progress only in this order:

1. Bind the exact V2.7 terminal package and nested V2.6 Stage-0/q-ref source
   inventory without provider dispatch.
2. Pass the full local test suite, source compilation, contract expansion,
   manifest rehash, secret scan, and source-inventory checks; replace every
   draft placeholder and freeze the contract.
3. Pass independent Linux and macOS CI on the exact release tree, merge that
   tree, verify CI on the merged commit, create the annotated
   `pilot-v2.8-science` tag, and attest a clean worktree checked out at that
   exact tag.
4. From the attested clean tag, execute the zero-provider parent import, fresh
   48-call scripted q-ref, and exact 14-cell Stage-0 import.
5. Verify the q-ref equivalence receipt, run the fixed Stage-0 reader and
   selector, and recompute the whole-matrix budget projection.
6. If and only if every prior gate passes, execute local Llama C → A → D → B.
7. If the local-first gates remain valid, execute GPT-5.2 C → A → D → B using
   the configured hosted-provider credentials.
8. Publish all 211 denominator rows, failures, budget/storage ledgers, raw
   paired deltas, aggregates, checksums, and negative results without
   post-hoc seed replacement or result-selective reruns.

Failure at any source, identity, release, launch, calibration, integrity, or
budget gate terminalizes the V2.8 denominator as another auditable no-go. It
does not permit falling through to a later or hosted stage.

## Claim boundaries

- V2.7 remains immutable operational failure provenance: one completed parent
  cell and 210 integrity stops, with no A–D effectiveness evidence.
- A successful V2.8 q-ref regeneration demonstrates exact identity binding and
  deterministic reference reconstruction only.
- A successful 14-cell Stage-0 import and selection demonstrates exact
  provenance reuse and fixed calibration only.
- Contract freeze, tests, CI, tagging, and clean launch attestation validate
  implementation/release readiness; they do not demonstrate a memory,
  retrieval, verifier, narrative, causal, robustness, or utility effect.
- Any A–D claim requires the corresponding fresh V2.8 paired evidence and its
  preregistered direction/effect gate. A failed C, A, D, or B gate must narrow
  the associated paper claim and must remain visible as a negative result.
- The 4-agent × 12-month mechanism micro-pilot cannot establish backbone
  independence, full-scale validity, real-news understanding, policy
  prediction, or equivalence to the original 100-agent × 240-month study.

Until the exact merged commit passes Linux/macOS CI, receives the annotated
`pilot-v2.8-science` tag, and obtains a clean-tag launch attestation, no V2.8
scientific or hosted dispatch is authorized.
