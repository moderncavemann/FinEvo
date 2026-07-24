# FinEvo Pilot-v2 Provider Interface Debug Results

Date: 2026-07-24

## Claim boundary

This report records bounded provider-interface diagnostics only. Every receipt
sets `scientific_evidence=false`, `diagnostic_only=true`, and
`denominator_inclusion=false`. These results do not establish model capability,
memory effectiveness, or any Experiment A-D effect.

The immutable `pilot-v1` evidence remains unchanged. Its failed or no-go cells
must not be rerun, replaced, or relabeled.

## Provenance and gates

- Interface implementation commit:
  `17ecd4dbcb4e68f2adb7bd81248fe6ca31e3a7e8`.
- Annotated diagnostic tags:
  `pilot-v2-debug-interface-v2`, `pilot-v2-debug-interface-v3`, and
  `pilot-v2-debug-interface-v4`.
- The branch and all peeled tag commits were verified against the remote.
- Local hermetic validation completed with 516 tests, 126 tracked Python files
  compiled, six sealed manifests rehashed, and no high-confidence secret
  findings across all tracked files.
- GitHub Actions run
  [30061897216](https://github.com/moderncavemann/FinEvo/actions/runs/30061897216)
  passed on Ubuntu 24.04 and macOS 14.
- Existing keys were read only from the original worktree's ignored `.env`;
  only `OPENAI_API_KEY` and `OPENROUTER_API_KEY` were exported to hosted
  probes. Endpoint and proxy overrides were removed, and no key was printed or
  copied into the debug worktree.

## Verified receipts

Raw diagnostic receipts remain ignored under
`experiment_results/pilot-v2-debug/raw/interface/`. The table reports only
verified, redacted fields.

| Model / tag | Status | Token cap | Prompt | Completion | Reasoning | Cost (USD) | Receipt SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| GPT-5.2 / v2 | pass | 80 | 21 | 23 | 6 | 0.00035875 | `84347bc13c5a47e7a35cd24daaae659e544de5c5b1faaf0e4ca79cd613166e72` |
| GPT-5.6 Sol / v2 | pass | 80 | 21 | 11 | 0 | 0.00043500 | `e9cd5900c2fc6989175ef004f00e64794dfc3466960eedd452d2e5662d6139f0` |
| Llama-4-Maverick / v2 | pass | 80 | 25 | 6 | 0 | 0.00000980 | `8332867825a7dc8539349fad6708e564d6800d8f77073122d4dd2a18aa43c948` |
| Gemini-3.5-Flash / v2 | no-go: length | 80 | 15 | 80 | 74 | 0.00074250 | `f3826e44875c862a155d3d45a3e1c6c7a3a47086f5ad9eef4c822f75f5fe96ee` |
| Gemini-3.5-Flash / v3 | no-go: length | 128 | 15 | 128 | 118 | 0.00117450 | `20ed681f1fd560e885ef46023d4c5283dde3e1ef3dc014792268ccabed689055` |
| Gemini-3.5-Flash / v4 | pass | 2,048 | 15 | 201 | 196 | 0.00183150 | `fc274ba9a0d9871906efb2d9b56d09e92a752afa15557c6915b3a54db87c5cfb` |
| Local Llama-3.3-70B / v4 | pass | 128 | 25 | 6 | 0 | 0.00000000 | `c0d12c05035b1a8223ecfc704c5606d18ed9abb9b48c8fae7452aa2abe3e6d9a` |

All passing hosted receipts have exact served-model, SDK, request-parameter,
provider-pin, route-snapshot, single-attempt, strict-JSON, and canonical
`finish_reason=stop` checks. Llama-4 and Gemini additionally have
`OR_RA_PASS` route attestation.

The local receipt used a recorded diagnostic-only JSON-mode override. Its
source contract remains `prompt_only`. Before dispatch, the frozen local
manifest and 42.5 GB model layer were revalidated:

- manifest:
  `a6eb4748fd2990ad2952b2335a95a7f952d1a06119a0aa6a2df6cd052a93a3fa`;
- model layer:
  `sha256:4824460d29f2058aaf6e1118a63a7a197a09bed509f0e7d4e2efb1ee273b447d`.

The local server was stopped after the one-call probe. No model was downloaded
or replaced.

## Failure diagnosis

The original failures were not credential failures:

1. GPT-5.2 rejected an explicit zero `temperature`; the corrected adapter
   omits that unsupported parameter and records `omitted_unsupported` rather
   than falsely claiming that temperature zero was applied.
2. Llama-4 returned valid inline OpenRouter route metadata, but the old
   validator treated an optional route field as mandatory.
3. Ollama's old adapter discarded native completion metadata, so truncation
   could not be distinguished from a valid stop.
4. Gemini's two debug no-go receipts were genuine truncations under caps of 80
   and 128. Its provider route and request contract were otherwise valid.

OpenRouter documents that Gemini 3 reasoning effort maps to Google's
`thinkingLevel`, while Google determines actual reasoning-token consumption
without public fixed breakpoints. Reasoning tokens count against the output
limit even when excluded from returned text. The bounded 2,048-token probe
completed after 201 actual completion tokens, including 196 reasoning tokens:

https://openrouter.ai/docs/guides/best-practices/reasoning-tokens

Claude Opus-4.8 was not called. The frozen endpoint catalog lacks the required
`temperature` and `top_p` parameters, so the existing strict
`require_parameters=true` contract correctly retains this profile as a
capability/interface no-go without spending API budget.

## Budget ledger

- Actual cost of the seven recorded live debug calls: `$0.00455205`.
- Conservative historical reserve: `$0.01`.
- Preserved stale test reservation: `$0.05`.
- Accounted cost plus active/reserved amount: `$0.06455205`.
- Diagnostic hard cap: `$0.30`.
- Final ledger SHA-256:
  `0aaa9983f729ec4ae6bfde32a58555ea614220b76888d6c4aff74323e71a913d`.

The stale reservation was deliberately retained. No failed receipt or ledger
entry was deleted or rewritten.

## Scientific release decision

The provider implementation is ready for a new preregistered scientific
contract, but the frozen `pilot-v1` scientific matrix remains unusable.
Stage 0 and Experiments A-D must not run from any debug tag.

A new scientific contract and clean tag must, at minimum:

1. register a common output limit and budget projection that can accommodate
   Gemini medium reasoning; 80 and 128 are empirically insufficient;
2. explicitly register JSON mode for the local Llama profile instead of using
   the diagnostic override;
3. retain Opus-4.8 as an interface no-go unless a new endpoint profile can
   satisfy every required parameter without weakening the contract;
4. rerun capability tasks and the 2-agent by 6-month preflight before any
   Stage-0 calibration;
5. keep every provider, parse, budget, and integrity failure in the
   preregistered denominator.

No result here supports an effectiveness, mechanism, or
backbone-independence claim.
