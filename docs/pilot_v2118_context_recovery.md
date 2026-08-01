# FinEvo V2.11.8 observed-p95 context recovery

## Status and scope

V2.11.8 is a prospective successor contract. It is not a resume, retry, or
reclassification of V2.11.7. The tracked contract remains `draft`; its science
design, source-manifest, and canonical release pins remain unset until the
dual-root source replay and final CI inventory are sealed.

The only implementation defect in scope is the observed-p95 authority
repository context used during historical V2.11.5 acceptance replay. V2.11.7
constructed runner configs while the V2.11.5 checkout was active, but the
subsequent Experiment D plan reconstruction revalidated the same source-backed
rows after that context had ended. It therefore compared the V2.11.5 source
commit against the V2.11.7 checkout and failed closed before provider
construction.

## Immutable inputs

- Failed lineage: clean detached annotated tag `pilot-v2.11.7-science`, commit
  `57c53588440dc2647f6b6ffae519049db4cd4844`. Its 87 rows remain
  `integrity-stopped`; provider construction and provider calls were both zero.
- Scientific authority: clean detached annotated tag
  `pilot-v2.11.5-science`, commit
  `2351ac2283f9fedb9dce70067174020be56ed9cc`. Its terminal A/C results remain
  external evidence and its 86 never-dispatched D/B/cross-model rows are the
  direct logical source for V2.11.8.
- V2.11.7 raw data is read-only and audit-only. No V2.11.7 row is copied into
  the V2.11.8 ledger or changed to another status.

## Contract and denominator

V2.11.8 registers exactly 87 fresh run identities:

| Stage | Rows | Budget bucket |
|---|---:|---|
| parent-import | 1 | `parent_v2117` |
| experiment-d | 55 | `hosted_v2118` |
| experiment-b | 25 | `hosted_v2118` |
| cross-model | 6 | `hosted_v2118` |

The 86 scientific rows are normalized-equal to the untouched V2.11.5 rows.
Cross-release logical deduplication therefore preserves the preregistered
136-row registered denominator and 131-row scientific denominator. Failed
seeds are not replaced; the matrix, reasoning settings, prices, prompts,
utility, shocks, actions, and metrics are unchanged.

## Budget boundary

The inherited debit is `$63.1196450625`, 3,440 hosted completions, and
270,191,728 bytes. The storage total adds only the V2.11.7 parent-import run's
1,797-byte actual to the prior cumulative debit; it does not add the full raw
tree or its 5 MB reservation. The hosted V2.11.8 cap remains
`$436.8803549375`, within the user-authorized `$500` total cap. The projected
complete continuation is `$212.4498325625`, 6,696 hosted completions, and
1,290,191,728 bytes cumulatively.

## Freeze and launch gates

1. Render the draft contract and verify its 87-row expansion and direct
   V2.11.5-to-V2.11.8 one-to-one mapping.
2. Render the V2.11.8 source manifest from the current checkout, the immutable
   V2.11.7 failed checkout, and the immutable V2.11.5 authority checkout.
3. Seal science-design and source-manifest hashes, run the complete fake matrix
   and full local/remote release gates, then seal the canonical contract and
   annotated `pilot-v2.11.8-science` tag.
4. With provider keys absent, complete zero-provider parent import and
   scientific-dispatch acceptance. Only after acceptance may the existing key
   be loaded for paid D, then B, then cross-model execution.

Passing implementation or release gates is not scientific evidence. A/C keep
their existing negative/no-go interpretations, and D/B/cross claims remain
unavailable until terminal V2.11.8 artifacts cover the preregistered ITT
denominator.

## Source-manifest runbook

Use three separate clean roots: the prospective V2.11.8 checkout, a detached
V2.11.7 failed-release checkout, and a detached V2.11.5 authority checkout.
Before rendering, confirm that the two source roots are clean and resolve to
their annotated-tag commits. Do not load provider keys for this gate.

```bash
git -C "$FAILED_V2117_ROOT" status --porcelain
git -C "$FAILED_V2117_ROOT" rev-parse HEAD
git -C "$AUTHORITY_V2115_ROOT" status --porcelain
git -C "$AUTHORITY_V2115_ROOT" rev-parse HEAD

python scripts/render_pilot_v2118_source_manifest.py \
  --failed-repo-root "$FAILED_V2117_ROOT" \
  --authority-repo-root "$AUTHORITY_V2115_ROOT"

python scripts/render_pilot_v2118_source_manifest.py \
  --failed-repo-root "$FAILED_V2117_ROOT" \
  --authority-repo-root "$AUTHORITY_V2115_ROOT" \
  --check
```

The first command writes the canonical tracked artifact at
`experiments/pilot_v2_11_8_source_manifest.json`; use it only during the
explicit source-freeze pass. The second command is read-only and fails on any
byte drift. A passing replay binds implementation and lineage provenance only;
it does not authorize paid dispatch and is not scientific evidence.
