# FinEvo V2.4 prospective local-first amendment

## Status

This document specifies an outcome-blind amendment. It is implementation and
review material, not a scientific result.

- Parent V2.3 remains an immutable 174-cell `complete-with-no-go` receipt.
- V2.3's 151 `budget-stopped` cells are not reopened, relabeled, or resumed.
- No Stage 0 or A–D outcome existed when this amendment was drafted.
- The V2.4 contract remains `draft`; no paid or scientific dispatch is
  authorized.
- The proposed `$150` hard cap is a feasibility value, not user authorization.

The expanded draft contract is
[`experiments/pilot_v2_4.yaml`](../experiments/pilot_v2_4.yaml), its source
overlay is
[`experiments/pilot_v2_4_overlay.yaml`](../experiments/pilot_v2_4_overlay.yaml),
and the exact call-role projection is
[`experiments/pilot_v2_4_cost_projection.json`](../experiments/pilot_v2_4_cost_projection.json).

## Immutable parent boundary

V2.4 imports only two kinds of authority from V2.3:

1. the cumulative debit of `$3.212770875`, 184 hosted completions, and
   4,196,087 bytes; and
2. source-backed, model-by-call-role observed p95 reservations for the exact
   GPT-5.2 and local Llama-3.3 profiles.

The import revalidates the annotated V2.3 science tag, frozen contract,
tamper-evident run and budget ledgers, terminal denominator, stage receipts,
published no-go package, and p95 source chain without constructing a provider.
Imported capability or preflight rows remain non-scientific and cannot be
counted as V2.4 effects. GPT-5.6 remains a boundary diagnostic only; Gemini and
Llama-4 retain their recorded preflight/capability failures.

## Prospective denominator and order

V2.4 registers 211 ITT cells before dispatch:

| Stage | Model | Registered cells | Executed call groups | Logical/provider calls | Reserved USD |
|---|---|---:|---:|---:|---:|
| parent import | scripted | 1 | 1 | 0 | 0 |
| q-ref resolution | scripted | 1 | 1 | 0 | 0 |
| Stage 0 calibration | local Llama-3.3 | 14 | 14 | 672 local | 0 |
| C | local Llama-3.3 | 25 | 25 | 1,280 local | 0 |
| A | local Llama-3.3 | 20 | 20 | 1,280 local | 0 |
| D | local Llama-3.3 | 35 | 5 shared checkpoints | 1,000 local | 0 |
| B | local Llama-3.3 | 25 | 25 | 1,440 local | 0 |
| C | GPT-5.2 | 25 | 25 | 1,280 hosted | 42.0966 |
| A | GPT-5.2 | 20 | 20 | 1,280 hosted | 42.0966 |
| D | GPT-5.2 | 30 | 5 shared checkpoints | 880 hosted | 31.4370 |
| B | GPT-5.2 | 15 | 15 | 800 hosted | 27.9741 |

The two administrative cells are non-scientific, leaving 209 scientific ITT
cells. Stage execution is fixed as parent import, q-ref, Stage 0, then local
C→A→D→B and GPT-5.2 C→A→D→B. Failed seeds are never replaced. Narrative and
additional cross-model cells are deferred and unregistered.

Stage 0 must select utility parameters using only the preregistered interior
coverage, component-balance, clipping, and residual rules. It may not inspect
which utility candidate favors FinEvo.

## Budget feasibility

The V2.3 observed p95 plus 25% reservation yields:

- 5,672 new local logical calls, which do not consume the hosted-completion
  cap;
- 4,240 new GPT-5.2 hosted completions;
- `$143.6043` for all hosted confirmatory cells;
- 3.52 GB of new conservative storage reservations;
- `$147.817070875` for the parent debit, hosted matrix, and `$1` manual
  reserve.

The draft therefore proposes a `$150` total hard cap with
`$2.182929125` hosted headroom. The full preregistered matrix must fit before
Stage 0. A projection failure causes a no-go receipt; it cannot silently reduce
seeds, arms, reasoning, or models.

## Claim-to-evidence contract

Local and GPT-5.2 lanes are always evaluated separately. A direction count may
never combine the two backbones. Each effect requires at least four complete
paired seeds out of five; missing, failed, parse, provider, budget, and
integrity rows remain in the ITT denominator.

| Claim | Primary metric and gate | Required artifact | Failure wording |
|---|---|---|---|
| M3 verifier reliability | false-rule ever-active, harmful exposure, retirement latency, and cumulative utility loss; verified improves at least one reliability endpoint in ≥4/5 paired seeds without material utility worsening | lane-specific C paired deltas, lifecycle ledger, candidate-admission receipt, zero-API sensitivity | verifier reliability not supported in this micro-pilot |
| M1 retrieval contribution | `full − prompt-only` shock/recovery utility plus `retrieval-only − no-context`; manipulation checks and ≥4/5 direction with ≥5% median primary effect | lane-specific A paired deltas, relevance@5, overlap, action and recovery traces | route traceability only |
| Downstream memory causality | treatment exceeds matched A/A null and one action bin, with ≥4/5 direction and six-step downstream utility or next-state effect | lane-specific D checkpoint binding, RNG/prefix hashes, branch actions and outcomes | prompt sensitivity only, or no supported downstream effect |
| Memory architecture attribution | descriptive no-memory/episodic/semantic/unverified/full comparisons with complete failures and proposals | lane-specific B aggregates and raw paired deltas | architecture comparison only; no winning-arm selection |
| Cross-backbone consistency | the same preregistered contrast has a supported, matching direction independently in both local and GPT-5.2 lanes | two lane gate receipts and a direction-consistency table | backbone interaction or inconclusive; never backbone-independent |

Only C, A, and D gates jointly passing can support escalation to the separate
10-agent × 24-month × 5-seed confirmatory design. This 4×12 run is never
described as the original 100×240 large-scale experiment.

## Release and execution boundary

Before any scientific dispatch:

1. all tests, compile checks, contract expansion checks, manifest rehash,
   secret scan, and Linux/macOS CI must pass;
2. an explicit hard-cap authorization must be recorded;
3. the contract must change from `draft` to `frozen`, bind the exact clean
   release commit, and be merged;
4. a new annotated `pilot-v2.4-science` tag must peel to that commit; and
5. the zero-provider parent import must pass.

Until all five conditions hold, the valid outcome is an implementation-ready
draft, not a science run.
