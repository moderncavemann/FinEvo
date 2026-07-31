# FinEvo V2.11 hosted-model boundary: full-scope prospective contract design

Status: implementation design only; no provider request was made while preparing
this document.

## 1. Decision

V2.11 must be a fresh, full-scope prospective denominator:

- GPT-5.2 reruns the complete registered A--D mechanism matrix.
- GPT-5.6 receives a fresh capability gate, a fresh 2-agent by 12-month
  long-context preflight, and a 3-seed `full` versus `no-memory` boundary lane.
- V2.10.2 contributes only the selected calibration inputs and cumulative
  budget debit. No V2.10.2 treatment effect, run summary, A--D P95 authority, or
  outcome direction is imported.
- The hosted cumulative hard caps remain USD 500, 7,500 provider completions,
  and 5 GB.
- The outcome-blind wire-cap candidate is 4,096 completion tokens for
  capability choices and actor actions. A fresh preflight must validate it.
  V2.11 must not adaptively switch to 8,192 after observing a failure.

Recommended identities:

| Item | Frozen value |
|---|---|
| Contract ID | `finevo-pilot-v2.11` |
| Prospective boundary schema | `finevo-pilot-v2.11-hosted-model-boundary-v1` |
| Expanded contract | `experiments/pilot_v2_11.yaml` |
| Parent source manifest | `experiments/pilot_v2_11_source_manifest.json` |
| Required annotated tag | `pilot-v2.11-science` |
| Raw namespace | `experiment_results/pilot-v2.11/raw/` |
| Reviewer package | `evidence/current_v2/pilot-v2.11/` |

The expanded V2.11 contract should be independently canonicalized rather than
silently inheriting V2.10.2's 211-cell stage validator. V2.10.2 is lineage, not
an executable base matrix.

## 2. Exact parent-import boundary

The zero-provider `parent-import` stage must verify the immutable V2.10.2
science tag, contract, terminal ledgers, q-ref artifact, Stage-0 selection, and
published evidence identities. Its allowlist is:

1. `q_ref = 63.50397933257746`;
2. selected utility profile `nu-0.5`:
   `rho=1`, `labor_weight=2`, `inverse_frisch=0.5`,
   `consumption_scale=1*q_ref`, `discount_factor=0.99`;
3. the Stage-0 absolute-flow threshold and its source binding, for the already
   preregistered offline sensitivity only;
4. the exact cumulative parent debit.

Observed V2.10.2 source identities at design time:

| Artifact | Identity |
|---|---|
| Contract canonical SHA-256 | `b8de8cfb2560d894dad65d68df8ae9126527d12d3807bef045fa52f5e9d4159e` |
| Science commit | `2dcc20f8dccc7a6a94a60a00d7f3750a9d61396d` |
| Science tag | `pilot-v2.10.2-science` |
| Run-ledger internal SHA-256 | `2219a832b9a7dfe235b32db882e126bddc36938f4f201a2ab84ddea6878bb809` |
| Budget-ledger internal SHA-256 | `73b1bac2a424147cbfa88bdb4e351d6c924b6e82847050f7cb1d254fe1ea4068` |
| q-ref content SHA-256 | `50d75c846c5e9d2b58fb92faf674da8a06ebb3b0ba7f21a6b1b2ad689034c40c` |
| Stage-0 selection content SHA-256 | `68c810055fc38683d3a8a7d597c54ffed4fb2c6332c2c02e1964b3ebfb61743c` |

The V2.11 parent debit is:

```json
{
  "schema_version": "finevo-parent-budget-debit-v1",
  "parent_contract_sha256": "b8de8cfb2560d894dad65d68df8ae9126527d12d3807bef045fa52f5e9d4159e",
  "parent_run_ledger_sha256": "2219a832b9a7dfe235b32db882e126bddc36938f4f201a2ab84ddea6878bb809",
  "parent_budget_ledger_sha256": "73b1bac2a424147cbfa88bdb4e351d6c924b6e82847050f7cb1d254fe1ea4068",
  "stage_bucket": "parent_v2102",
  "cost_usd": 16.044922812500005,
  "hosted_completions": 816,
  "storage_bytes": 217010835,
  "record_sha256": "c841dc4cbdfdb548c6917fbb2670c31ba3759f3d4f52ffb0fbb5b9d8bcbbc74d"
}
```

The receipt must also assert:

```json
{
  "imported_effect_cells": 0,
  "effect_metrics_observed": false,
  "effect_artifact_paths": [],
  "imported_p95_authorities": [],
  "provider_construction_during_import": false,
  "provider_calls_during_import": 0
}
```

Do not copy the V2.10.2 raw A--D tree into V2.11. Reseal only the four
allowlisted calibration/budget objects and their source hashes.

## 3. Models and provider profiles

### GPT-5.2

Retain the current direct profile:

- profile ID `gpt52_main`;
- requested and served model `gpt-5.2-2025-12-11`;
- OpenAI direct provider pin;
- JSON-object response format;
- medium reasoning;
- unsupported temperature, top-p, and seed fields omitted exactly as declared;
- one provider attempt and no fallback;
- USD 1.75/M input, 0.175/M cached input, and 14/M output, subject to
  launch-time catalog equality.

### GPT-5.6

The repository already has a usable direct profile shape in V2.3 and generic
wire support in `OpenAIProvider`. Register it prospectively as
`gpt56_diagnostic`:

- requested and served model `gpt-5.6-sol`;
- OpenAI direct provider pin;
- JSON-object response format;
- medium reasoning;
- unsupported temperature, top-p, and seed fields omitted;
- one provider attempt and no fallback;
- USD 5/M input, 0.5/M cached input, and 30/M output, subject to launch-time
  catalog equality.

`OpenAIProvider` already sends `max_completion_tokens`, `reasoning_effort`, and
`response_format` for GPT-5-family Chat Completions calls. The provider catalog
reader is model-generic and recognizes the current GPT-5.6 page. What remains
unverified until the fresh capability call is account permission and actual
served-model behavior.

`gpt-5.6-sol` is not a dated immutable weight snapshot. Evidence may claim an
exact route/model-ID boundary, not immutable model weights.

For both models, reject a request before dispatch if its conservatively encoded
prompt could cross 200,000 tokens. This keeps GPT-5.6 below the 272K
long-context price tier covered by a different rate and leaves ample context
headroom. Unknown or tier-mismatched pricing is an interface no-go, never a
zero-cost estimate.

## 4. Output-cap decision and interface gates

Freeze these V2.11 output contracts:

| Call role | Completion cap | Visible JSON byte cap |
|---|---:|---:|
| capability-choice | 4,096 | 512 |
| capability-proposal | 4,096 | 4,096 |
| actor-action | 4,096 | 1,024 |
| semantic-proposal | 4,096 | 4,096 |

Why 4,096 is the minimum outcome-blind candidate:

- the sealed historical GPT-5.2 action profile had raw P95 1,999 and reserved
  P95 2,499, so the old 2,048 wire cap was below its own reservation;
- 4,096 covers 2,499 and leaves 2,097 tokens, or 51.2%, above raw P95;
- the historical GPT-5.6 action/semantic reserved P95 values were 1,539/577;
- 8,192 doubles the maximum interface exposure without evidence that 4,096 is
  insufficient.

Historical values are design inputs only. They are not V2.11 budget or
scientific authorities.

The fresh capability plus long-context preflight must produce 48 action-kind
samples and 14 semantic-kind samples per model. For each kind, require:

- exact requested profile, provider pin, response route, and served model;
- one attempt, accepted JSON, `finish_reason=stop`, and response completion;
- provider-reported prompt, completion, cached, and reasoning usage;
- no provider error, clipping, parser recovery, or missing usage;
- `ceil(raw_p95 * 1.25) <= 4096`;
- raw-P95 completion headroom of at least 25% of the 4,096 wire cap;
- exact minimum sample counts of 48 actions and 14 proposals;
- no single prompt above the frozen 200K short-tier ceiling.

If any check fails, V2.11 terminalizes the affected model's descendants and
does not switch caps. An 8,192 retry requires a new immutable amendment and a
fresh denominator.

Cost comparison, using hard completion ceilings only:

| Candidate | Full new matrix completion-only ceiling |
|---|---:|
| 4,096 action / 4,096 semantic | USD 366.706688 |
| 8,192 action / 4,096 semantic | USD 672.235520 |

The 8,192 completion-only ceiling already exceeds the remaining cumulative USD
cap before input tokens. This reinforces using 4,096 as the current
outcome-blind candidate. Normal science dispatch is still governed by fresh
P95 reservations, not by assuming every completion reaches the wire cap.

## 5. Exact full-scope matrix

Frozen seeds:

- preflight: `2010922376`;
- main: `1099057501, 1421875452, 1769977770, 959809858, 617806385`;
- GPT-5.6 boundary: the first three main seeds.

All scientific actor runs use 4 agents, 12 months, the registered rate shock,
and imported `nu-0.5`. The long-context preflight uses 2 agents, 12 months,
full memory, the same utility profile, and the registered shock.

Suggested execution order remains C--A--D--B so existing checkpoint and
sensitivity dependencies remain valid:

| Stage | Model and cells | Ledger cells | Hosted calls |
|---|---|---:|---:|
| parent-import | one zero-provider authority import | 1 | 0 |
| capability-gate | 2 models x 1 seed | 2 | 60 |
| long-context-preflight | 2 models x 1 seed, 2x12 | 2 | 64 |
| experiment-c | GPT-5.2, 5 arms x 5 seeds; candidate admission is offline | 25 | 1,280 |
| experiment-a | GPT-5.2, 4 arms x 5 seeds | 20 | 1,280 |
| experiment-d | GPT-5.2, (7 continuation + 4 narrative) x 5 seeds | 55 | 1,480 |
| experiment-b | GPT-5.2, 5 arms x 5 seeds | 25 | 1,440 |
| cross-model | GPT-5.6, `full`/`no-memory` x 3 seeds | 6 | 336 |
| **Total** | 131 scientific cells plus 5 operational cells | **136** | **5,940** |

Call-role derivation:

| Lane | Action calls | Semantic calls | Total |
|---|---:|---:|---:|
| GPT-5.2 capability + preflight | 48 | 14 | 62 |
| GPT-5.2 A | 960 | 320 | 1,280 |
| GPT-5.2 B | 1,200 | 240 | 1,440 |
| GPT-5.2 C | 960 | 320 | 1,280 |
| GPT-5.2 D | 1,440 | 40 | 1,480 |
| GPT-5.6 capability + preflight | 48 | 14 | 62 |
| GPT-5.6 full/no-memory | 288 | 48 | 336 |
| **Total** | **4,944** | **996** | **5,940** |

The B call count is arm-aware: `no-memory` and `episodic-only` do not invoke
the semantic proposer. C's candidate-admission arm is a zero-provider offline
cell. D reserves one shared prefix per seed, then seven continuation and four
narrative branches; proposals are frozen after the prefix.

The GPT-5.6 boundary lane does not add a second matched A/A null. GPT-5.2
matched-A/matched-B remain registered inside D. Consequently, the GPT-5.6 lane
can support only a capability-qualified, three-seed directional replication
statement; it cannot claim a GPT-5.6-specific repeatability-null-qualified
effect.

## 6. Cumulative budget and mandatory projection

After importing V2.10.2, the exact remaining capacities are:

| Resource | Global cap | Parent debit | Remaining |
|---|---:|---:|---:|
| Hosted USD | 500 | 16.044922812500005 | 483.9550771875 |
| Hosted completions | 7,500 | 816 | 6,684 |
| Storage bytes | 5,000,000,000 | 217,010,835 | 4,782,989,165 |

The newly registered 5,940 calls yield cumulative completion count 6,756,
leaving 744. One provider attempt is mandatory; no retry, replacement probe, or
unregistered paid diagnostic may consume this headroom.

A conservative storage projection using current runner reservations is:

- parent import: 5 MB;
- capability: 40 MB;
- long-context preflight: 40 MB;
- A: 400 MB;
- B plus GPT-5.6 boundary: 620 MB;
- C: 410 MB;
- D: 400 MB;
- raw run/receipt subtotal: 1.915 GB;
- projection and publication-staging reservation: 105 MB;
- new reservation total: 2.020 GB;
- parent plus new reservation total: 2,237,010,835 bytes.

This leaves 2,762,989,165 bytes for reviewer-package output and operational
headroom.

The old P95 values give only an outcome-blind planning check:

- GPT-5.2: USD 188.524888125;
- GPT-5.6: USD 18.223600000;
- prospective total: USD 206.748488125;
- prospective plus parent: USD 222.7934109375.

Runtime authority must instead be the fresh model-by-call-kind P95 generated
from V2.11 capability and 2x12 preflight. After multiplying by 1.25, a
zero-provider full-matrix projection receipt must atomically project every
remaining A--D and GPT-5.6 cell before the first scientific dispatch. The
projection receipt is an operational gate bound to the preflight receipt, not
an extra ITT cell. It passes only if all three remaining caps above are
satisfied. On failure it marks every undispatched scientific cell
`budget-stopped`; it may not shrink arms, seeds, or reasoning settings.

At this post-preflight gate, use ledger actuals for the 124 already-dispatched
capability/preflight calls and fresh reserved-P95 values only for the remaining
5,816 calls. The remaining call-kind counts are GPT-5.2 action/semantic
`4560/920` and GPT-5.6 action/semantic `288/48`. Do not replace already-paid
actuals with P95 estimates, and do not add both versions.

Use these stage caps with zero automatic reserve so the user-authorized USD 500
is the exact hard ceiling:

```json
{
  "total_usd": 500.0,
  "automatic_reserve_usd": 0.0,
  "stage_usd_caps": {
    "parent_v2102": 16.044922812500005,
    "hosted_v211": 483.9550771875,
    "manual_reserve": 0.0
  }
}
```

## 7. Minimal implementation map

### Contract and parent import

1. `experiments/pilot_v2_11.yaml`
   - add the exact profiles, 4,096 caps, seeds, stages, cells, budgets, and
     claim boundaries above;
   - keep V2.10.2 effects and P95 reuse explicitly forbidden.
2. `experiments/pilot_v2_11_source_manifest.json`
   - bind only V2.10.2 terminal identity, q-ref, selected Stage-0 profile,
     absolute threshold, and cumulative debit.
3. `verified_memory/pilot_v211_parent_import.py`
   - verify the parent tag/commit, contract, evidence, terminal ledgers, and
     allowlisted artifacts;
   - persist small resealed wrappers rather than a full A--D snapshot;
   - expose `parent_budget_debit_for_v211`.

### Contract parser

4. `verified_memory/pilot_contract.py`
   - register V2.11 ID/tag/canonical constants and one V2.11 boundary policy;
   - add V2.11 to supported IDs and release-tag validation;
   - move task-cap equality out of `TaskOutputContract.__post_init__` into a
     contract-ID-specific validation so V2.10.2 remains exactly 2,048 while
     V2.11 uses 4,096;
   - add a dedicated V2.11 stage/model-role/cell-count validator instead of
     routing it through `_v2_4_expected_stages()`;
   - accept the exact USD 500 parent/new stage caps and zero reserve;
   - retain 4/5 core and 2/3 cross-model denominator minima, while the
     cross-model directional wording still requires 3/3 same direction;
   - preserve byte-identical loading and hashes for V1 through V2.10.2.

### Orchestration and interface authority

5. `verified_memory/pilot_orchestrator.py`
   - add V2.11 parent-debit and parent-import dispatch;
   - separate “has a parent import” from the old local-first
     `_cross_model_science_stage_ids()` special case;
   - make `_preflight_config()` contract-driven and use 2 agents x 12 months,
     imported utility, and a 12-month shock;
   - change `_max_call_projection()` to 32 preflight calls and arm-aware
     semantic-call counts;
   - change `_preflight_checks()` from hard-coded 12/12 actions to exact 24/24
     actions, 8 proposal outcomes, and 32 provider rows;
   - generate fresh P95 from 48 action and 14 semantic observations per model;
   - integrate `completion_capacity_gate()` for the 4,096 wire-cap check;
   - emit a zero-provider full-matrix projection receipt after preflight and
     bind it to all 131 scientific cells without creating another ITT cell;
   - make observed-P95 readers profile-driven for `gpt52_main` and
     `gpt56_diagnostic`, without accepting a V2.10.2 authority;
   - include `cross-model` in the all-remaining projection;
   - preserve grouped D accounting and exact 7+4 branch publication.
6. `verified_memory/pilot_interface_gate.py`
   - use the existing outcome-blind cap/headroom validator;
   - extend it with exact finish/parse/sample and per-request prompt-tier
     assertions when integrating it into the preflight receipt.
7. `verified_memory/pilot_budget.py`
   - no schema change is required: zero automatic/manual reserve is already
     representable;
   - verify parent debit plus every remaining projection against all three
     cumulative caps.
8. `llm_providers.py`
   - the GPT-5.6 Chat Completions wire path already exists;
   - add only the pre-dispatch prompt-tier ceiling and ensure provider errors
     retain conservative accounting without turning reserved tokens into
     reported model output.

### CLI and evidence

9. `run_pilot.py`
   - document V2.11 parent import and stage sequence;
   - register V2.11 for `--parent-repo-root` and the V2.11 evidence builder;
   - retain the contract-derived raw namespace.
10. `verified_memory/pilot_v211_evidence.py`
    - build a dedicated package rather than adding another branch to the
      V2.4--V2.10.2 lane adapter;
    - validate all 136 ITT rows and terminal receipts;
    - publish parent/calibration lineage without any V2.10.2 effect table;
    - recompute A, C, D, narrative, B, and GPT-5.6 full/no-memory summaries;
    - keep GPT-5.2 and GPT-5.6 capability, parse/provider failures, and deltas
      separate;
    - prohibit `backbone-independent` and GPT-5.6 null-qualified wording.
11. `verified_memory/pilot_evidence.py`
    - add the V2.11 parent-receipt verifier only if shared normalizers are
      reused; do not route V2.11 through the old generic A/C/D package.
12. `verified_memory/scientific_release_attestation.py`
    - the raw namespace logic is already generic;
    - add the V2.11 annotated tag through `ReleaseRequirements`.

## 8. Required tests

Add these focused suites:

- `tests/test_pilot_contract_v2_11.py`
  - canonical round trip and mutation rejection;
  - exact 136 ledger cells, 131 scientific cells, and stage breakdown;
  - exact 4,944 action + 996 semantic = 5,940 hosted calls;
  - exact 4,096 task caps and cumulative resource caps;
  - unchanged V2.10.2 canonical round trip.
- `tests/test_pilot_v211_parent_import.py`
  - exact parent tag/commit/ledger identities and debit;
  - q-ref/`nu-0.5`/threshold allowlist;
  - zero provider construction/calls;
  - effect path, metric, P95, or A--D artifact injection fails closed.
- `tests/test_pilot_orchestrator_v2_11.py`
  - 2x12 preflight emits 24 actions, 8 proposals, and 32 calls per model;
  - fresh P95 has 48/14 samples and 1.25 reserve;
  - 4,096 capacity/headroom and 200K prompt-tier gates;
  - full-matrix projection includes A--D plus GPT-5.6 and stops all pending
    science rows on any cap breach;
  - arm-aware B/C/GPT-5.6 call counts and exact cumulative completion count;
  - no interrupted reservation is redispatched.
- `tests/test_pilot_interface_gate.py`
  - keep the historical 2,048 failure and 4,096 pass;
  - add fresh-sample-count, finish-reason, and prompt-tier failures.
- `tests/test_pilot_provider_catalog.py`
  - GPT-5.6 direct profile, price, snapshot ID, JSON mode, and reasoning
    parameter fixtures.
- `tests/test_llm_provider_budget.py`
  - GPT-5.2/GPT-5.6 send `max_completion_tokens=4096`;
  - reasoning/JSON fields and omitted unsupported fields are exact;
  - provider failure accounting does not masquerade as reported usage.
- `tests/test_pilot_checkpoint.py`
  - exact 2x12 preflight checkpoint restore and call journal;
  - 7+4 D branches retain equal RNG start and frozen proposals.
- `tests/test_pilot_v211_evidence.py`
  - all ITT failures stay in denominators;
  - main 4/5 and cross 2/3 completeness are recomputed;
  - GPT-5.6 directional wording requires 3/3 and capability pass;
  - no imported V2.10.2 effect and no backbone-independent claim.
- `tests/test_run_pilot_v211_cli.py`
  - parent-root scope, raw namespace, stage dispatch, and evidence builder.
- `tests/test_pilot_release_attestation.py`
  - V2.11 annotated tag, clean peeled HEAD, Linux/macOS CI identities.

Before tagging, run the complete existing suite, new tests, compile inventory,
manifest rehash, secret scan, and both required CI jobs. Paid work starts only
from the clean annotated tag.

## 9. Current blockers and claim boundary

1. GPT-5.6 catalog and code wiring are available, but account permission and
   live response behavior remain intentionally untested until the registered
   capability cell.
2. The current worktree contains concurrent uncommitted implementation changes.
   They are not part of this design and must be reconciled before a clean tag.
3. Fresh 2x12 P95 may still fail the 4,096 capacity or USD projection gate. The
   correct result is a terminal V2.11 no-go, not a cap/seed/arm change.
4. Only 744 hosted completions remain after the full registered matrix. No
   unregistered paid probe or retry is safe.
5. The GPT-5.6 model ID is not a dated immutable weight snapshot. Limit claims
   to the exact observed route/model ID.
6. The micro-pilot still does not support `backbone-independent`, full
   confirmatory, or 100x240 claims.
