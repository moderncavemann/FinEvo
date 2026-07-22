# Verified dual-track memory: bounded validation report

Date: 2026-07-22
Branch: `codex/verified-dual-memory`
Scope: method/integration validation only; **not scientific performance evidence**.

## Outcome

The redesigned execution path passes its local closed loop and limited GPT-5.2
simulation smoke, but it is not yet ready for the full experiment matrix. Final
audit found that the stored replay did not forward its recorded decoding seed
and its shuffle crossed a section boundary. Both code defects are fixed and
covered by tests, but a fresh sealed replay is required before treating action
differences as an authoritative intervention result. Four-way API context
ablations and downstream checkpoint replay also remain pending, so
multi-seed/full-scale runs stay disabled.

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
  state-to-memory-to-action-to-next-state chain; the strengthened intervention
  runner is implemented but awaits a replacement sealed API replay.
- **Non-claims:** these checks do not establish higher utility, better macro
  outcomes, general resistance to hallucinated rules, model independence, or
  effective use of rich narrative events.

## Central claim-to-evidence map

| Reviewer gap | Required mechanism | Observable intermediate | Metric/test | Current evidence | Status |
|---|---|---|---|---|---|
| Reflection may reinforce errors | M3 proposal, independent verifier, provisional state, later-only activation, counterevidence and retirement | Rule event ledger with cited M2 IDs and lifecycle transitions | Reference/hash validity; temporal separation; support/contradiction counts; injected-error replay | Unit/adversarial tests plus one sealed injected-rule snapshot | Partial |
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
- Manifest SHA-256:
  `aa45a372e89fa901ac1abbc119f1123f6f009e17583b993bbb1ce07437a2ed94`.

### Paired memory replay

Superseded strengthened iteration:
`artifacts/verified_replays/g4b-agent0-t5-v2/`.

- Intended fixed fields: environment-state hash, base prompt, context ID/hash,
  requested GPT-5.2 alias, temperature/top-p, parser, and discretization.
- 5 API calls; 2,606 prompt + 193 completion tokens. The artifact's old
  estimator records `$0.010134`; current standard uncached list-rate estimate
  is `$0.0072625`.
- All five checks implemented by the old harness passed, but those checks did
  not attest provider seed forwarding or section-preserving shuffling.
- No-memory changed executed consumption from `0.60` to `0.72` and left labor
  at 152 hours.
- Wrong-agent memory changed labor from 152 to 168 hours and consumption from
  `0.60` to `0.82`.
- These action differences are retained only as an iteration receipt. The
  recorded seed was not sent to the provider, and the old shuffler moved some
  episodes below the active-rule header. The artifact therefore cannot support
  an only-memory-varied causal claim.
- Manifest SHA-256:
  `39d9352b9ec39439240c930ea62a59670f721331e9982d01718f510ed42ae1f6`.

The earlier replay at `artifacts/verified_replays/g4b-agent0-t5/` is also
retained as an iteration receipt. Its injected rule was visibly marked
unverified and was therefore too weak. Neither stored replay is authoritative.

## API usage and cost-estimate ledger

Across the two 14-call API loops and both five-call replay iterations:

- 38 calls;
- 18,968 prompt tokens;
- 2,269 completion tokens;
- total historical estimator cost recorded in the artifacts: `$0.084132`;
- current standard uncached list-rate estimate for those same token totals:
  `$0.064960`.

The old artifacts do not retain cached-input token counts, so neither figure is
presented as the actual billed amount. New provider records retain cache usage
when exposed by the API.

No full experiment was launched.

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
causally traceable and fail-closed in bounded simulation tests. A corrected
seeded replay must be sealed before claiming responsiveness to memory
interventions, and no result yet shows baseline outperformance.
