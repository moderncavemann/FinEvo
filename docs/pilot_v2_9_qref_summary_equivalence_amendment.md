# FinEvo Pilot V2.9: q-ref Summary-Equivalence Amendment

## Status and purpose

Pilot V2.8 is terminal and immutable. Its 211-cell denominator ended as:

- 1 complete `parent-import` cell;
- 1 failed `q-ref-resolution` cell; and
- 209 `integrity-stopped` Stage-0/A-D cells.

The V2.8 q-ref run completed all 48 deterministic
`ScriptedDiagnosticProvider` action calls, produced 48 actions, 48 utility
rows, and 12 shock rows, and made zero hosted-provider calls at zero hosted
cost. It then failed before Stage 0 because the audit-equivalence check
compared the complete run summary byte-for-byte. The compared summaries
contain run-specific identities and monotonic wall-clock measurements, so an
exact raw equality check is not a valid deterministic-equivalence criterion.

V2.9 is a prospective implementation/audit amendment. It does not resume,
rewrite, or reclassify V2.8. No Stage-0 or A-D treatment-effect outcome was
generated under V2.8, so the amendment remains outcome-blind with respect to
the registered scientific effects.

## Frozen parent boundary

V2.9 must bind all of the following before any new scientific dispatch:

- V2.8 contract `finevo-pilot-v2.8` and science tag
  `pilot-v2.8-science`;
- the annotated tag peel and merged release commit;
- the exact V2.8 raw inventory, run ledger, budget ledger, stage receipts,
  q-ref failure receipt, and verified q-ref run artifacts;
- the published V2.8 `complete-with-no-go` evidence package;
- the nested V2.6 Stage-0 source and its 14 complete calibration cells; and
- the inherited cumulative debit of USD 3.212770875 and 184 hosted
  completions.

The V2.8 raw namespace remains read-only. V2.9 uses a fresh raw namespace and
a new 211-cell ITT denominator.

## Versioned deterministic run-summary projection

The V2.9 q-ref equivalence receipt uses
`finevo-pilot-v2.9-qref-run-summary-projection-v1`.

The comparison is allowlist-first:

1. Both raw summaries must have the same JSON structure.
2. Run and budget identities are checked against each summary's bound source
   config before they are replaced by canonical identity sentinels.
3. The only non-deterministic scalar paths are:
   - `$.run_id`;
   - `$.api.budget_id`;
   - `$.api.elapsed_seconds`;
   - `$.api.completions[*].budget_id`;
   - `$.api.completions[*].started_elapsed_seconds`;
   - `$.api.completions[*].finished_elapsed_seconds`; and
   - `$.api.completions[*].elapsed_seconds`.
4. Wall-clock values must be finite and non-negative. Each completion must
   satisfy `finished >= started`, and its elapsed value must be consistent
   with the corresponding interval within a fixed numerical tolerance.
5. After those paths are normalized, the entire remaining summary is
   compared exactly. Tokens, costs, models, labels, tags, reservation IDs,
   call counts, stop reasons, validation checks, diagnostics, and final
   metrics are never omitted.

The receipt stores both raw-summary hashes and both projected-summary hashes.
It also stores the exact allowed-path policy, the observed 1,002-leaf
inventory and 195 normalized leaves, provider boundary, and hashes of actions,
utility rows, and shock rows.

Any unexpected difference, missing allowed path, extra completion, malformed
identity, non-finite timer, token/cost/model/label/tag drift, or projected
summary mismatch stops the q-ref cell and terminalizes the denominator.

## Fresh and imported work

V2.9 permits:

- one fresh, zero-hosted q-ref run with exactly 48 scripted diagnostic calls;
- hash-verified import and resealing of the same 14 V2.6 Stage-0 calibration
  cells; and
- fresh-only execution of all 195 registered A-D cells after q-ref and
  Stage-0 gates pass.

V2.9 forbids:

- reusing the V2.8 decoded q-ref result as the V2.9 result;
- provider construction during parent or Stage-0 import;
- provider redispatch for imported Stage-0 cells;
- resuming or editing V2.8 ledgers;
- replacing failed seeds;
- shrinking the registered matrix after freeze; and
- changing utility thresholds, selection order, arms, seeds, models, shocks,
  or stop/go rules.

## Budget and release boundary

The cumulative hosted hard cap remains USD 500, with USD 1 unavailable for
automatic dispatch. Hosted-completion and storage limits remain 7,500 and
5 GB. V2.8 added zero hosted cost and zero hosted completions; its 48 scripted
calls are recorded separately and are not relabeled as hosted completions.

No provider call may occur until the V2.9 source manifest, compact overlay,
expanded contract, implementation inventory, tests, secret scan, Linux/macOS
CI, merged-main CI, annotated tag, and launch attestation all pass. Paid work
may start only from the clean `pilot-v2.9-science` tag.

## Claim boundary

V2.9 remains a preregistered 4-agent by 12-month mechanism micro-pilot. Its
prerequisite checks are not treatment-effect evidence. Only fresh V2.9 A-D
cells may support A-D claims, subject to the existing paired-seed, manipulation,
matched-null, and negative-result gates. It is not the 10x24x5 confirmatory
pilot and not the 100x240 large-scale experiment.
