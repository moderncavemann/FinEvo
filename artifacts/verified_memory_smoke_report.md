> [!WARNING]
> **HISTORICAL PRE-P0 V1 EVIDENCE ONLY**
>
> This report and its sealed run/replay artifacts predate the current P0
> completion changes. They must not be cited as validation or scientific
> evidence for Evidence-Grounded Rule Memory v2. The current status and claim
> boundary are in the
> [P0 methodology completion audit](../docs/p0_methodology_completion_audit.md).

# Verified dual-track memory: bounded validation report

Date: 2026-07-22
Branch: `codex/verified-dual-memory`
Scope: method/integration validation only; **not scientific performance evidence**.

## Outcome

The redesigned execution path passes its local closed loop, limited GPT-5.2
simulation smoke, and corrected five-treatment replay. It is ready for a small
matched pilot, but not for the full experiment matrix. The replay now forwards
and records the requested seed client-side, preserves memory-section boundaries,
records the served model, and runs from a clean tracked commit. OpenAI returned no
system fingerprint and documents seed determinism as best-effort, so the replay
supports controlled prompt-level action sensitivity rather than strict causal
identification. Four-way API context ablations and downstream checkpoint replay
remain pending; multi-seed/full-scale runs stay disabled.

## Research contract

- **Setting:** a closed-loop Foundation economy in which each decision can
  affect the next agent and macro state.
- **Input:** only information causally available at decision time: the current
  state, an M1 context route, finalized M2 episodes, and verifier-admitted M3
  rules belonging to the acting agent.
- **Output:** a directly executed labor allocation on an 8-hour grid and a
  bounded consumption fraction, followed by an immutable transition and
  realized flow-utility record.
- **Assumptions:** Foundation's transition and accounting code define the
  economic testbed; the verified path has no debt/default state and therefore
  does not invent a bankruptcy penalty.
- **Constraints:** retrieval is causal, identity-scoped, hash-checked, and
  budgeted; a semantic proposal cannot become active using the same outcomes
  that created it.
- **Failure criterion:** the run fails closed on malformed provider output,
  temporal leakage, broken hashes/references, non-finite values, budget
  exhaustion, or cash-flow/accounting mismatch.
- **One-sentence claim:** in the bounded implementation tests reported here,
  the redesigned dual-track memory produces an auditable
  state-to-memory-to-action-to-next-state chain and shows controlled action
  sensitivity to memory treatments in one seeded GPT-5.2 replay.
- **Non-claims:** these checks do not establish higher utility, better macro
  outcomes, general resistance to hallucinated rules, model independence, or
  effective use of rich narrative events.

## Central claim-to-evidence map

| Reviewer gap | Required mechanism | Observable intermediate | Metric/test | Current evidence | Status |
|---|---|---|---|---|---|
| Reflection may reinforce errors | M3 proposal, independent verifier, provisional state, later-only activation, counterevidence and retirement | Rule event ledger with cited M2 IDs and lifecycle transitions | Reference/hash validity; temporal separation; support/contradiction counts; injected-error replay | Unit/adversarial tests plus one corrected sealed injected-rule snapshot | Partial |
| Closed-loop causal chain is opaque | M1 causal context + M2 pre/post transition + action/prompt/retrieval provenance + M0 utility | One decision record links state, memory IDs, prompt hash, action, next state, and utility | No future retrieval; exact transition alignment; cash-flow identity; content-addressed manifest | G0 and G4b sealed runs | Supported as implementation traceability |
| Text context may be a switch rather than information | Four matched M1 routes: no context, prompt-only, retrieval-only, full | Route-specific prompt and retrieval payloads | Paired action, utility, and downstream-state deltas | Route unit tests only | Missing behavioral evidence |
| Gains may depend on the base LLM | Provider-neutral interface and identical gate definitions | Same treatment ledger under another model | Matched second-model effect estimates and failure rates | GPT-5.2 smoke only | Missing |
| Dual-track contribution is not separately attributable | Episodic-only and semantic-only treatments under the same checkpoint | Treatment-specific hash-bound memory payload | Paired action/utility/macro deltas | Infrastructure exists; experiment not run | Missing |
| Direct narrative strategy adjustment is not demonstrated | Structured event features tied to decisions and later outcomes | Event-to-retrieval-to-action trace | Matched event content, counterfactual ordering, downstream effects | Schema support only | Missing |

The map deliberately separates software validity from empirical effectiveness:
passing the former is a prerequisite for, not evidence of, the latter.

## What changed

| Reviewer risk | Implemented response |
|---|---|
| Reflection hallucination/error reinforcement | M3 candidates cite finalized M2 IDs; verifier checks condition, executed action, outcome, counterevidence, later support, and retirement. |
| Unclear macro -> memory -> action chain | Every decision stores causal context, retrieved episode/rule IDs, prompt hashes, action, aligned next state, utility, and a hash-bound snapshot. |
| Hidden Bernoulli labor artifact | Verified path uses deterministic 8-hour action steps over 0--168 hours. |
| Wealth-only objective ambiguity | Prompt and M0 ledger use explicit realized consumption/labor flow utility; wealth/Gini remain diagnostics. |
| Narrative channel overclaim | M1 supports structured causal event features, but the present smoke makes no rich-narrative effectiveness claim. |
| LLM capability dependence | Provider interfaces and gates are model-agnostic; cross-model effectiveness is still untested and remains an explicit paper limitation. |

## Gate evidence

### G0: deterministic no-network loop

- Run: `artifacts/verified_runs/g0-local-s11/`
- Configuration: 2 agents, 6 months, seed 11, retrieval-only context.
- Result: pass; 12 actions, 12 aligned episodes, 12 utility rows, 2 active
  rules, 6 active-rule retrievals, and cash-flow residuals within the `1e-8`
  gate (maximum absolute residual `3.64e-12`).
- Boundary: provider is a scripted synthetic fixture and `scientific_evidence`
  is false.
- Manifest SHA-256:
  `d4aadb1d212dc197aa2001d82023627185784bc3b4ff477715cd96a88232f44d`.

### G4a: first limited GPT-5.2 loop (useful failure)

- Run: `artifacts/verified_runs/g4-api-gpt52-s11/`.
- 14 API calls; 6,432 prompt + 960 completion tokens. The artifact's old
  estimator records `$0.030816`; at the current standard uncached GPT-5.2 list
  rate the same totals estimate to `$0.024696`.
- Environment/action/M0/M2 gates passed, but both M3 candidates were rejected.
- Root cause: proposal occurred after only two heterogeneous episodes and the
  LLM described a cross-episode comparison; the strict verifier correctly found
  that the claimed supports did not all independently satisfy the predicate/
  action/outcome rule.
- Remediation: propose after three completed episodes and state the verifier's
  absolute-threshold semantics plus per-support checklist in the prompt.
- Manifest SHA-256:
  `c49330b0daeb6bfefdd0005fe7ae70bef478e20b99858eea42f9a2da5bb40814`.

### G4b: remediated limited GPT-5.2 loop

- Run: `artifacts/verified_runs/g4b-api-gpt52-s11/`.
- Same bounded shape: 2 agents, 6 months, seed 11, retrieval-only context.
- 14 API calls; 7,315 prompt + 923 completion tokens. The artifact's old
  estimator records `$0.033021`; at the current standard uncached GPT-5.2 list
  rate the same totals estimate to `$0.02572325`.
- Result: pass; no provider/parser errors, no clipping, 12/12 aligned
  transitions, 12/12 utility rows, and cash-flow identities within the `1e-8`
  gate (maximum absolute residual `1.46e-11`).
- Both candidates entered `provisional` at `t=3`, activated only after distinct
  outcomes at `t=4`, and were retrieved four times at `t=4..5`.
- Final rules had support/contradiction counts `5/0` and `6/0`, with confidence
  `0.8571` and `0.8750`.
- Direct actions used 152 and 168 hours, demonstrating that the verified path is
  no longer restricted to the legacy 0/168 Bernoulli outcome.
- In G4a/G4b, seed 11 controlled simulator RNG; those runs predate forwarding a
  decoding seed to the provider.
- Manifest SHA-256:
  `aa45a372e89fa901ac1abbc119f1123f6f009e17583b993bbb1ce07437a2ed94`.

### Paired memory replay

Current corrected artifact:
`artifacts/verified_replays/g4b-agent0-t5-v3/`.

- Source: sealed G4b run, agent 0 at `t=5`; execution code commit
  `d328e0b56dd0dbab58a69eea334abf0cbded5738`, with tracked worktree
  `dirty=false`.
- Fixed at the request layer: environment-state hash, base prompt, context
  ID/hash, requested model alias, temperature/top-p, parser, discretization,
  and decoding seed 11. Only the hash-bound memory treatment varies.
- The shuffler reverses only finalized episode entries and preserves the active
  rule section byte-for-byte. The wrong-context arm requires both a different
  agent and a different context hash.
- 5 one-attempt API calls; 2,606 prompt + 188 completion tokens; zero cached
  prompt tokens; current list-rate estimate `$0.0071925`.
- All five requests record client-side forwarding of seed 11; all responses
  report the same served model snapshot, `gpt-5.2-2025-12-11`. OpenAI returned
  no response-side seed field and no system fingerprint. The artifact field
  `decoding_seed_verified=true` denotes the local adapter invariant, not a
  provider attestation.
- Matched: 152 labor hours, consumption `0.60`.
- No-memory: 152 hours, consumption `0.56` (change `-0.04`).
- Shuffled: 152 hours, consumption `0.60` (unchanged).
- Wrong-context: 168 hours, consumption `0.84` (changes `+16` hours and
  `+0.24` consumption rate).
- Injected high-confidence erroneous rule: 152 hours, consumption `0.60`
  (unchanged in this snapshot).
- Independent manifest rehash: pass. Manifest SHA-256:
  `5b4772d46f1fd124fca78f4b5572de034ae2629e09ff01f031939256ddbee651`.

This is controlled prompt-level action sensitivity, not strict causal
identification: the
[Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
documents `seed` as best-effort and does not guarantee determinism, while the
optional system fingerprint was absent. It also provides no utility or
downstream macro effect because the replay is prompt-level.

The v1 and v2 directories remain superseded iteration receipts. V1 used a
visibly unverified injected rule; v2 recorded but did not send the decoding
seed and allowed the shuffle to cross the active-rule header.

## API usage and cost-estimate ledger

Across the two 14-call API loops and all three five-call replay iterations:

- 43 calls;
- 21,574 prompt tokens;
- 2,457 completion tokens;
- the first four artifacts' historical estimator total is `$0.084132`, while
  v3 uses the corrected current-rate estimator and records `$0.0071925`;
- repricing all recorded tokens at the current standard uncached rate gives
  `$0.0721525`.

The old artifacts do not retain cached-input token counts, so the aggregate is
not presented as the actual billed amount. V3 records zero cached tokens.
The pricing basis is the official
[GPT-5.2 model page](https://developers.openai.com/api/docs/models/gpt-5.2):
`$1.75`/M input, `$0.175`/M cached input, and `$14`/M output.

No full experiment was launched.

## Final implementation checks

- `138` repository tests pass.
- Python compilation and `git diff --check` pass.
- Budgeted calls enforce one provider attempt per reservation.
- Effective Foundation configuration and its canonical hash are sealed in new
  run configs; labor and consumption grids are validated and synchronized into
  that effective environment configuration.
- Post-setup execution failures write a content-addressed
  error/config/budget receipt; preflight failures remain stderr-only, and
  partial in-memory streams are explicitly not checkpointed.
- Credential-value and common token/private-key pattern scans over the intended
  release scope found no match.

## Remaining work, ordered by evidence value

### P0: required before a scientific result claim

1. Freeze seeds, shock schedule, primary endpoints, denominators, stopping
   rules, provider settings, and cost ceiling in a run contract.
2. Run the four M1 context routes under a matched small API design; unit routing
   tests alone do not establish behavioral value.
3. Add checkpoint-backed paired replay so action deltas can be translated into
   realized utility and next-state deltas without rerunning unrelated history.
4. Run explicit episodic-only versus semantic-only paired controls to attribute
   effects within the dual track.

### P1: required for the reviewer's broader concerns

1. Test a small second-model pilot before making any model-independence claim.
2. Sample naturally proposed, accepted, rejected, and retired rules; manually
   label factual support and failure modes before quantifying reflection safety.
3. Run matched narrative-content and narrative-order interventions with event,
   retrieval, action, and downstream outcome all retained in the trace.
4. Add an exact manuscript comparison with EconAgent and EconAI after reading
   and citing their primary papers; no related-work distinction is inferred
   from this code audit.

### P2: only after P0/P1 gates pass

Run multi-seed/full-scale performance comparisons and report uncertainty,
failure rates, token/cost totals, and all prespecified endpoints. A positive
smoke result is not a reason to skip the matched controls.

Until these are complete, the defensible claim is: the redesigned method is
causally traceable and fail-closed in bounded simulation tests, with controlled
prompt-level action sensitivity to memory treatments in one corrected seeded
replay. Residual provider nondeterminism, utility effects, macro effects, and
baseline outperformance remain unestablished.
