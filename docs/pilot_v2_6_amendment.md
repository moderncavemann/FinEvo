# FinEvo Pilot V2.6: inherited-p95 reader amendment

## Status and scope

V2.6 is an outcome-blind operational retry. It does not resume, delete, or
reclassify any V2.5 cell. It creates a fresh 211-cell denominator under
`experiment_results/pilot-v2.6/raw/` and preserves the 209-cell scientific
matrix, seeds, arms, provider profiles, shock, utility grid, stage order, and
stop/go rules from V2.5.

The hosted-API hard cap remains `$500`; the 7,500 hosted-completion and 5 GB
caps also remain unchanged. V2.6 carries forward `$3.212770875`, 184 hosted
completions, and 6,303,635 bytes of cumulative parent storage. The `$1`
reserve remains unavailable for automatic dispatch.

## Frozen V2.5 no-go

V2.5 is terminal with all 211 rows accounted for:

- 2 complete operational cells: parent import and deterministic q-ref;
- 14 failed Stage-0 calibration cells; and
- 195 downstream integrity-stopped cells.

Every Stage-0 failure occurred before provider construction with zero new
provider calls and zero new API cost. The V2.5-specific importer accepted its
inherited observed-p95 receipts, but the central runner reader lacked an
explicit dispatch for
`finevo-pilot-v2.5-inherited-observed-p95-authority-v1`.
Consequently, those receipts fell through to the legacy V2.3 shape check.
This is an interface-integrity no-go, not model-capability, calibration, or
treatment-effect evidence.

The complete terminal package is retained under
`evidence/current_v2/pilot-v2.5/`. The V2.6 source manifest binds its contract,
annotated tag, ledgers, failure signature, complete raw-tree inventory,
published evidence, parent-import receipt, and both p95 receipts/projections.

## Correction

The central reader now dispatches only explicitly registered inherited
receipt schemas. Unknown schemas continue to fail closed. During V2.6 parent
import:

1. the immutable V2.5 release, raw tree, evidence package, parent-import
   receipt, and two p95 authorities are independently reverified;
2. the verified V2.5 receipts are retained only as parent sources;
3. the same numeric reservations are resealed into V2.6 child receipts bound
   to the V2.6 contract, `pilot-v2.6-science`, and current release HEAD; and
4. both the V2.6-specific verifier and the central runner verifier must accept
   the child receipts and serialized runner configuration before downstream
   dispatch is allowed.

No V2.5 or V2.3 receipt is rewritten or resigned. The current-HEAD/annotated-tag
identity requirement is not relaxed.

## Release and execution gates

V2.6 remains non-dispatchable while its contract is draft. Before any provider
call, the exact merged tree must pass the full local suite, source compilation,
sealed-manifest rehash, secret scan, and Linux/macOS CI; then it must receive
an annotated `pilot-v2.6-science` tag and a clean-tag launch attestation.

The first scientific-workflow action is the zero-provider `parent-import`
stage. Failure at that gate terminalizes the new denominator and produces
another `complete-with-no-go`. Only a successful import and central-reader
round trip permits q-ref and Stage 0.
