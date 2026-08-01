# FinEvo V2.11.6 continuation contract

## Purpose

V2.11.6 is a provenance-only continuation of the preregistered V2.11.5
mechanism micro-pilot. It exists because the provider-free publication
consumer merge advanced `origin/main` after Experiments C and A had reached
terminal states. The V2.11.5 paid-release gate correctly refused a later D
launch before provider construction because `origin/main` no longer equalled
the peeled V2.11.5 science tag.

V2.11.6 must not rerun, replace, or reclassify any V2.11.5 A/C cell. It may
dispatch only the 86 V2.11.5 cells that were still `scheduled` at the bound
parent-ledger prefix.

## Immutable parent state

- Contract: `finevo-pilot-v2.11.5`
- Canonical contract SHA-256:
  `e1ecdec43e3f7a7b9a3d0977e2522d95861e826fc68781377d7eaceeb5e6e2ef`
- Annotated tag: `pilot-v2.11.5-science`
- Tag object: `bccfb13cee7d592470d1873cfacc3b12bed38be4`
- Peeled commit: `2351ac2283f9fedb9dce70067174020be56ed9cc`
- Parent run ledger: 136 rows, 53 events, 47 complete, 3 failed, 86
  scheduled; internal SHA-256
  `8a86231f0906ea117626190cc7a2699933c968ce555612cb1bc6378473601fa7`.
- Parent run-ledger event head:
  `61489ef64e71400e603e2fb1110e5e8af3ba772ac083361338a4ccff9641022f`.
- Parent budget ledger: 103 events; internal SHA-256
  `53e70f6c0b9053674408de385e1a5b5bf42ace7e82dc8e0c6f227ea124b7a38f`.
- Immutable parent raw inventory: 691 regular files, 48,820,556 bytes,
  canonical SHA-256
  `f2fdb1ccedcb70e6793d3b8f3c87425f0d602552f0a3e0e7f35db9c5777c6746`.
  The later `.real-stage-execution.lock` is excluded because it records only
  the rejected, zero-provider D pre-dispatch attempt and is not a scientific
  result artifact.
- V2.11.5 current-release actuals: USD 43.1214245, 2,436 hosted
  completions, and 48,139,533 storage bytes.
- Cumulative debit including V2.11.4: USD 63.1196450625, 3,440 hosted
  completions, and 270,188,235 storage bytes.
- Experiment C receipt: 25/25 complete, formal status
  `complete-with-no-go`, progression allowed, content SHA-256
  `39a9d35f4961fee4b0bc59ac67f7a9a2da0c3f95fddf77a418b92e518b6e2eba`.
- Experiment A receipt: 20/20 terminal, 17 complete and 3 failed,
  `complete-with-no-go`, progression allowed, content SHA-256
  `177dc8ce4d1957eac0734bb1716279676f77931e30b3a1d10dd2c138a43a5457`.

Every value above is re-read from the immutable parent checkout during the
zero-provider import. A mismatch is a pre-dispatch integrity no-go.

## Continuation matrix

The combined scientific denominator remains the original 131 V2.11.5
scientific cells. V2.11.6 contains one zero-provider import cell plus a
one-to-one mapping of the 86 parent cells that were still scheduled:

| Stage | Cells | Hosted calls | Status at freeze |
| --- | ---: | ---: | --- |
| parent import | 1 | 0 | new operational cell |
| Experiment D | 55 | 1,480 | eligible after import |
| Experiment B | 25 | 1,440 | eligible after terminal D |
| cross-model | 6 | 336 | eligible after terminal B |
| total continuation science | 86 | 3,256 | not yet run |

The mapping must be exact after removing only `contract_id`, `run_id`, and
the renamed hosted budget bucket. Seeds, arms, narratives, model IDs,
requested/served models, environment, shock, utility, prompts, output caps,
reasoning settings, metrics, and stop/go rules are unchanged.

## Budget and denominator rules

- Hosted USD hard cap remains USD 500.
- Hosted completion hard cap remains 7,500.
- Storage hard cap remains 5 GB.
- V2.11.6 begins with the full cumulative parent debit above.
- The remaining hosted USD bucket is USD 436.8803549375.
- The preregistered remaining matrix projects 3,256 calls, USD
  149.3301875 at observed p95 plus reserve, and 1,020,000,000 bytes of
  continuation storage.
- If every remaining cell runs to its registered cap, the cumulative ledger
  projects USD 212.4498325625, 6,696 hosted completions, and 1,290,188,235
  bytes, all below the unchanged hard caps.
- The registered completion ceiling after all remaining calls is 6,696.
- A failed continuation cell stays in the original ITT mapping; no seed is
  replaced and no failed cell is retried.
- Import, acceptance, CI, evidence publication, and analysis are provider-free.
- Credentials may be loaded only after parent import, release attestation,
  source equivalence, acceptance, prerequisite, full-matrix projection, and
  budget checks pass.

## Claim boundary

V2.11.6 does not turn the negative/incomplete A or C gates into positive
results. It does not claim a fresh A/C replication, backbone independence,
real-news understanding, the 10x24x5 confirmatory pilot, or the 100x240
experiment. Reviewer claims are determined only from the combined immutable
parent results and terminal continuation receipts.
