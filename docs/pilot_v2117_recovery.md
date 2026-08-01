# Pilot V2.11.7 accounting-scope recovery

V2.11.7 is a prospective, outcome-blind operational amendment. It does not
resume or rewrite V2.11.6 and it does not create a new scientific denominator.

## Immutable lineage

- V2.11.5 remains the scientific authority: 136 registered cells, comprising
  5 operational cells and 131 scientific cells. Its 47 complete, 3 failed, and
  86 never-dispatched scheduled rows remain unchanged.
- V2.11.6 remains a terminal pre-dispatch no-go: all 87 rows are
  `integrity-stopped`, provider construction never occurred, provider calls are
  zero, and no acceptance receipt or science reservation exists.
- V2.11.7 has 87 execution rows: one zero-provider dual-lineage import plus the
  same 86 logical D/B/cross-model cells that were never dispatched in V2.11.5.
  Cross-release logical-cell deduplication therefore preserves 136 registered
  and 131 scientific cells; V2.11.6 rows are audit-only.

## Accounting correction

The V2.11.6 verifier selected only 47 rows in budget bucket `hosted_v2115` but
compared them with a frozen total covering all 50 current-release rows. The
three omitted operational rows contributed zero cost, zero hosted completions,
and 164,153 bytes. The immutable V2.11.5 totals are:

- hosted rows: `$43.1214245`, 2,436 completions, 47,975,380 bytes;
- operational rows: `$0`, 0 completions, 164,153 bytes;
- all current rows: `$43.1214245`, 2,436 completions, 48,139,533 bytes;
- cumulative through V2.11.5: `$63.1196450625`, 3,440 completions,
  270,188,235 bytes.

V2.11.7 additionally carries the 1,696-byte V2.11.6 zero-call audit record, so
its parent debit is `$63.1196450625`, 3,440 completions, and 270,189,931 bytes.
The fresh hosted cap is `$436.8803549375`; the global hosted cap remains `$500`.

## Release and dispatch gates

Paid dispatch is forbidden while the contract is draft. Before a science tag
may be created, the release must pass all of the following:

1. Validate the exact V2.11.5 authority raw and the exact V2.11.6 no-go raw from
   separate clean detached worktrees, before provider construction.
2. Build a canonical source manifest binding both roots and prove a one-to-one
   mapping for all 86 never-dispatched logical cells.
3. Run the complete continuation fake matrix (D/B/cross; A/C remain inherited
   terminal V2.11.5 evidence), checkpoint restore/equality tests, the full local
   suite, compile inventory, manifest rehash, and secret scan.
4. Merge to `main`, require Linux and macOS CI to pass, create an annotated
   `pilot-v2.11.7-science` tag, and prepare a clean detached launch worktree.
5. Produce a zero-provider acceptance receipt showing one completed import,
   86 scheduled science rows, no science reservation, and no provider call.

Only after those gates pass may execution proceed in the preregistered order
Experiment D, Experiment B, then cross-model diagnostics. Failures remain in
the ITT denominator; seeds, models, reasoning settings, and caps are not
silently changed.

## Claim boundary

This remains a 4-agent by 12-month mechanism micro-pilot. It is neither the
10x24x5 confirmatory pilot nor a 100x240 result. Existing negative/no-go results
for V2.11.5 Experiments A and C remain unchanged.
