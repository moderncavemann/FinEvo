# FinEvo V2.11.9 release-binding recovery

## Status and scope

V2.11.9 is a successor contract, not a resume, retry, or reclassification of
V2.11.8. The tracked contract is frozen as a release candidate: its science
design, dual-root source manifest, exact CI inventory, and canonical identity
are sealed. Paid dispatch remains forbidden until the candidate is merged to
`main`, both release CI jobs pass, the annotated science tag is created, and
the clean-tag zero-provider parent import and dispatch acceptance both pass.

The scientific matrix remains unchanged. The primary implementation defect in
scope is historical release-provenance reconstruction during V2.11.5
acceptance replay. V2.11.8 reconstructed the
historical `GitProvenance` with the correct V2.11.5 tag, commit, and release
attestation, but supplied an empty `contract_binding`. The frozen acceptance
stores the hash of
`contract.validate_provenance(V2115_COMMIT, V2115_TAG)`, so the empty mapping
recomputed to a different release hash and failed closed before provider
construction. The launch-input contract binding is a distinct CI-policy object
and is not a valid substitute.

V2.11.9 prospectively reconstructs `contract_binding` with
`contract.validate_provenance(V2115_COMMIT, V2115_TAG)`. A read-only replay
with that binding reproduced all seven frozen acceptance material fields; this
is an implementation gate only, not a scientific result.

The release audit also closes two provenance gaps before the contract is
frozen. The source manifest covers the complete `verified_memory` and
Foundation Python trees, the provider module, unique CLI, and release
renderers; only five literal SHA-256 values that form unavoidable release hash
cycles are normalized. The legacy environment reads `data/profiles.json`
relative to the process working directory, so acceptance and every scientific
stage now require the process cwd to be the clean release root and verify the
profile as a regular, non-symlink file with frozen SHA-256
`1bc90a92ef8e32f3da6e474f787207b79b1c82cc0b7b13c5ea3bd6cd1439b223`
before provider construction.

## Immutable inputs

- Failed lineage: clean detached annotated tag `pilot-v2.11.8-science`, commit
  `67aa0fcce68fa5ac43b48dd3b81b849112137093`. Its 87 rows remain
  `integrity-stopped`; provider construction and provider calls were both zero.
- Scientific authority: clean detached annotated tag
  `pilot-v2.11.5-science`, commit
  `2351ac2283f9fedb9dce70067174020be56ed9cc`. Its terminal A/C results remain
  external evidence and its 86 never-dispatched D/B/cross-model rows are the
  direct logical source for V2.11.9.
- V2.11.8 raw data is read-only and audit-only. No V2.11.8 row is copied into
  the V2.11.9 ledger or changed to another status.

## Contract and denominator

V2.11.9 registers exactly 87 fresh run identities:

| Stage | Rows | Budget bucket |
|---|---:|---|
| parent-import | 1 | `parent_v2118` |
| experiment-d | 55 | `hosted_v2119` |
| experiment-b | 25 | `hosted_v2119` |
| cross-model | 6 | `hosted_v2119` |

The 86 scientific rows are normalized-equal to the untouched V2.11.5 rows.
Cross-release logical deduplication therefore preserves the preregistered
136-row registered denominator and 131-row scientific denominator. Failed
seeds are not replaced; the matrix, reasoning settings, prices, prompts,
utility, shocks, actions, and metrics are unchanged.

## Budget boundary

The inherited debit is `$63.1196450625`, 3,440 hosted completions, and
270,193,500 bytes. The storage total adds only the V2.11.8 parent-import run's
1,772-byte actual to the prior cumulative debit; it does not add the full raw
tree or its 5 MB reservation. The hosted V2.11.9 cap remains
`$436.8803549375`, within the user-authorized `$500` total cap. The projected
complete continuation is `$212.4498325625`, 6,696 hosted completions, and
1,290,193,500 bytes cumulatively.

## Frozen candidate identity

- Contract SHA-256: `ec16563bf906b8f6c1492d2a30f291d2c849cd639c2f314e7a1c8ac619e3fa3f`
- Science-design SHA-256: `ad2609dbc1b2d736560bcfc874d2af5899f7a048a0b6aeadbe2e350f91244e01`
- Source-manifest file/content SHA-256:
  `609adf9d12543b4caa7adb0cbddb8c8a9073a10f689adf52a8670608d16e9cb1` /
  `36a790fe5edd6269218d6010046ec9293c3c418d8bc58a4dd5d89a6a70a547d6`
- Frozen CI inventory: 2,115 tests, 316 compiled Python sources, and six
  sealed run/replay manifests. These are release requirements, not evidence
  that CI or the scientific matrix has completed.

## Freeze and launch gates

1. Render the draft contract and verify its 87-row expansion and direct
   V2.11.5-to-V2.11.9 one-to-one mapping.
2. Render the V2.11.9 source manifest from the current checkout, the immutable
   V2.11.8 failed checkout, and the immutable V2.11.5 authority checkout. The
   three roots must be pairwise distinct even through filesystem aliases. The
   manifest binds the complete current runtime/release source surface, the
   Foundation environment and profile input, and the exact historical
   release-binding recovery; it rejects the prior empty binding.
3. Seal science-design and source-manifest hashes, run the complete fake matrix
   and full local/remote release gates, then seal the canonical contract and
   annotated `pilot-v2.11.9-science` tag.
4. With provider keys absent, complete zero-provider parent import and
   scientific-dispatch acceptance from the clean tagged release root. The
   acceptance raw namespace must contain only its exact preregistered files;
   stale stage/provider/development files and all symlinks fail before ledger
   markers or provider sentinels. Only after acceptance may the existing key be
   loaded for paid D, then B, then cross-model execution.

Passing implementation or release gates is not scientific evidence. A/C keep
their existing negative/no-go interpretations, and D/B/cross claims remain
unavailable until terminal V2.11.9 artifacts cover the preregistered ITT
denominator.

## Source-manifest runbook

Use three separate clean roots: the prospective V2.11.9 checkout, a detached
V2.11.8 failed-release checkout, and a detached V2.11.5 authority checkout.
Before rendering, confirm that the two source roots are clean and resolve to
their annotated-tag commits. Do not load provider keys for this gate.

```bash
git -C "$FAILED_V2118_ROOT" status --porcelain
git -C "$FAILED_V2118_ROOT" rev-parse HEAD
git -C "$AUTHORITY_V2115_ROOT" status --porcelain
git -C "$AUTHORITY_V2115_ROOT" rev-parse HEAD

python scripts/render_pilot_v2119_source_manifest.py \
  --failed-repo-root "$FAILED_V2118_ROOT" \
  --authority-repo-root "$AUTHORITY_V2115_ROOT"

python scripts/render_pilot_v2119_source_manifest.py \
  --failed-repo-root "$FAILED_V2118_ROOT" \
  --authority-repo-root "$AUTHORITY_V2115_ROOT" \
  --check
```

The first command writes the canonical tracked artifact at
`experiments/pilot_v2_11_9_source_manifest.json`; use it only during the
explicit source-freeze pass. The second command is read-only and fails on any
byte drift. A passing replay binds implementation and lineage provenance only;
it does not authorize paid dispatch and is not scientific evidence.

## Terminal evidence publication

Publication is a separate provider-free step after all 87 V2.11.9 ledger rows
are terminal. Run it from the clean tagged V2.11.9 checkout with provider keys
absent and the independent clean V2.11.5 authority checkout supplied
explicitly:

```bash
python run_pilot.py \
  --contract "$V2119_ROOT/experiments/pilot_v2_11_9.yaml" \
  --stage publish-evidence \
  --raw-root "$V2119_ROOT/experiment_results/pilot-v2.11.9/raw" \
  --source-repo-root "$V2119_ROOT" \
  --authority-repo-root "$AUTHORITY_V2115_ROOT" \
  --evidence-root "$V2119_ROOT/evidence"
```

The dedicated consumer reconstructs the 136 registered / 131 scientific
logical V2.11.5 denominator from 50 immutable parent terminals and 86 fresh
V2.11.9 continuations. It retains every ITT failure, keeps A/C as external
V2.11.5 no-go evidence, and writes the reviewer package only after replaying
both release roots, budget ownership, terminal artifacts, stage namespaces,
package checksums, and the provider-construction sentinels. Publication does
not retry a cell or change a scientific conclusion.
