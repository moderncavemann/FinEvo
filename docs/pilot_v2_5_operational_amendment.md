# FinEvo Pilot V2.5 Operational Amendment

## Status and scope

V2.5 is a **frozen, outcome-blind operational amendment**. It does not
reinterpret or resume V2.4, and it does not change the registered scientific
design. Its 211-cell denominator and `$500` hard cap are fixed before any
V2.5 provider call or treatment outcome.

The V2.4 launch terminated in `parent-import` before provider construction.
Its immutable denominator contains 211 terminal `integrity-stopped` cells,
including all 209 registered scientific cells. The failure receipt records
`provider_calls=0`, incremental cost `$0`, and
`scientific_evidence=false`. The compact V2.4 package is published as
`complete-with-no-go`; it is operational provenance, not treatment evidence.

## Permitted correction

The failure was caused by validating a V2.3 checkpoint's source-code binding
against the importing V2.4 code tree. V2.5 permits one correction with two
mandatory gates:

1. The historical checkpoint `source_hashes` and `binding_hash` must match the
   Git tree of the peeled annotated tag `pilot-v2.3-science`
   (`ab32e3c9dcf581a40f3093652e144b56f853c782`; tag object
   `e985abd6749471363db6b27bda66485c0b578bb3`).
2. Only after the historical gate passes may the current child code perform a
   compatibility replay without treating the child code binding as the
   historical authority. Its recomputed exactness must equal the frozen V2.3
   exactness receipt.

An unconditional strict-binding bypass is forbidden. The parent checkpoint
cannot be rewritten, and the parent observed-p95 authority cannot be
re-signed. Parent import must still construct no provider.

## Frozen invariants

The V2.5 overlay inherits the V2.4 science-design hash
`ac11b024435d6d6b03a68b59e5f59f28d92a822ddd3712b1b4c612b668a20586`.
It preserves exactly:

- 211 registered cells and 209 scientific cells;
- the five main seeds, all arms, models, narratives, shocks, utility grid,
  stop/go thresholds, and `C-A-D-B` stage order;
- the GPT-5.2 and local Llama provider profiles and request contracts;
- the `$500` total cap, 7,500 hosted-completion cap, and 5 GB storage cap;
- the V2.3 cumulative debit of `$3.212770875`, 184 hosted completions, and
  4,196,087 bytes;
- the V2.4 incremental debit of `$0` and zero hosted completions.

The V2.4 raw tree remains
`experiment_results/pilot-v2.4/raw`. V2.5 must use the disjoint namespace
`experiment_results/pilot-v2.5/raw`; deleting, reopening, relabeling, or
resuming a V2.4 cell is forbidden.

## Provenance and release gate

The exact V2.4 terminal receipts, V2.4 evidence package, and original V2.3
authority sources are bound by
`experiments/pilot_v2_5_source_manifest.json`.

The frozen contract is
`experiments/pilot_v2_5.yaml`, with canonical SHA-256
`1f9809062684a1a2afb96b7342b88a06810e0e87ac883aa63a858a65a81d188d`.
Its release inventory contains 853 test node IDs, 163 tracked Python source
paths, and the unchanged six-manifest sealed inventory.

The V2.4 GitHub Actions run may be cited only as parent provenance. V2.5
requires a new PR, a new Linux/macOS CI run on the merged V2.5 tree, a new
annotated `pilot-v2.5-science` tag, and new launch-input and release-attestation
receipts. The existing OpenAI credential may be reused because the V2.4
failure occurred before provider construction and was not a credential
failure; the secret must remain outside contracts, logs, and evidence files.
