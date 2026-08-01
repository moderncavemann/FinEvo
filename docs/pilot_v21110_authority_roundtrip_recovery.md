# FinEvo V2.11.10 authority-roundtrip recovery

V2.11.10 is a fresh preregistered successor to the immutable V2.11.9
implementation no-go. It is not a retry, resume, or reclassification of any
V2.11.9 cell. V2.11.9 remains terminal with its full ITT denominator, zero new
provider completions, and zero new hosted cost.

## Failure being repaired

The V2.11.9 current-authority producer serialized the four receipt-envelope
fields inside each source-authority row. The runner also treated those fields
as its outer receipt envelope. During scientific config validation the runner
correctly stripped the envelope before comparing against the verified source,
leaving a 13-field row to compare with the producer's 17-field row. Every
fresh science cell therefore stopped before its first provider completion.

V2.11.10 freezes one authority-layer contract:

1. The verified current-authority producer emits exactly 13 source fields.
2. The runner attaches exactly four receipt fields, producing 17 fields.
3. Source-backed validation strips exactly those four fields and requires the
   remaining 13 fields to equal the independently rebuilt producer row.
4. Numeric action and semantic reservations must equal the sealed projection.

The four runner-owned fields are `source_authority_receipt_path`,
`source_authority_receipt_file_sha256`,
`source_authority_receipt_content_sha256`, and `source_release_commit`.

## Immutable lineage and denominator

The V2.11.10 parent import is provider-free and binds two distinct roots:

- the annotated `pilot-v2.11.9-science` release and its immutable terminal
  no-go raw tree;
- the independently verified V2.11.5 authority release used only for
  calibration and observed-P95 dispatch authority.

The successor registers fresh V2.11.10 run identities. It preserves the same
mechanism design, seeds, arms, stage ordering, ITT treatment of failures, and
the `$500` cumulative hosted cap. It does not import V2.11.9 failures as
effects and does not replace failed seeds.

## Release gate

No hosted provider credential may be loaded until all of the following are
true:

1. The V2.11.10 contract is frozen and its canonical hash is pinned.
2. The source manifest is rendered from the prospective checkout plus the
   immutable V2.11.5 authority checkout, then its file and content hashes are
   pinned in both the contract module and CI source-manifest anchor.
3. The tracked contract, source manifest, two renderers, continuation module,
   complete Python-source inventory, and complete pytest node inventory are
   covered by Linux and macOS CI receipts.
4. CI verifies the immutable V2.11.9 annotated tag object
   `f0af244b64a69b3ee4571452df6d3611fd8c6220` and peeled commit
   `d850902af6218c72a6b0e71275c62c81c9143fb9`.
5. A clean annotated `pilot-v2.11.10-science` tag points to the exact merge
   commit covered by both CI jobs.
6. Provider-free parent import and scientific-dispatch acceptance rebuild the
   real runner reservations, serialize and restore every scientific config,
   and run source-backed reservation validation for both GPT-5.2 and GPT-5.6.

During bootstrap the V2.11.10 source-manifest file/content pins may both be
`None`. This is an explicit draft no-go state: unit tests must pass while the
source-manifest release check fails closed. The two pins must be populated
atomically before the contract can be frozen or a release receipt emitted.

## Provider-free preparation

From the prospective V2.11.10 checkout, use the tracked renderers only after
the contract and source implementation are complete:

```bash
python scripts/render_pilot_v21110_contract.py --help
python scripts/render_pilot_v21110_source_manifest.py --help
python -m verified_memory.ci_release_receipt verify-source-manifests \
  --output /tmp/finevo-v21110-source-manifests.json
```

These commands do not authorize scientific dispatch. Paid D, then B, then
cross-model execution can begin only from the clean frozen tag after the
zero-provider acceptance receipt is sealed. A failure at any authority,
budget, namespace, provenance, or config-roundtrip check remains in the ITT
ledger and stops before provider construction.
