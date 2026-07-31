# FinEvo V2.11.1 bootstrap repair runbook

Status: release and launch procedure. V2.11.1 is a preregistered bootstrap
repair, not a reinterpretation or continuation of the terminal V2.11 run.

## Frozen boundary

V2.11 remains immutable at annotated tag `pilot-v2.11-science`, commit
`5d6c7920bd4a872b02931fdee8a47b9ac4e7b352`. Its 136-cell ledger ended with
3 complete, 2 failed, and 131 integrity-stopped cells. The two long-context
preflight failures occurred before provider dispatch, so their provider-call
count is zero. They remain failures in the V2.11 ITT denominator.

V2.11.1 uses a new contract, tag, denominator, and raw namespace:

| Item | V2.11.1 value |
|---|---|
| Contract | `experiments/pilot_v2_11_1.yaml` |
| Contract ID | `finevo-pilot-v2.11.1` |
| Required annotated tag | `pilot-v2.11.1-science` |
| Raw namespace | `experiment_results/pilot-v2.11.1/raw/` |
| Parent checkout | immutable `pilot-v2.11-science` checkout |
| Ledger denominator | 136 cells: 5 operational and 131 scientific |

Only hash-bound calibration, the two passed V2.11 capability authorities, the
terminal failure boundary, and cumulative budget debit may cross the parent
boundary. No V2.11 treatment-effect cell, outcome direction, failed preflight
result, or observed-P95 authority is imported or reclassified as V2.11.1
evidence.

The two import stages are provider-free:

1. `parent-import` validates and seals the immutable V2.11 source: 0 fresh
   calls.
2. `capability-gate` imports the two passed capability cells: 0 fresh calls.
   Their 60 historical calls remain accounted in the cumulative parent debit;
   they are not dispatched again.

The only bootstrap retry is `long-context-preflight`: two models, each with 24
actor-action and 8 semantic-proposal calls, for exactly 64 fresh hosted calls.
Its temporary reservation uses the frozen 200,000-prompt-token and
4,096-completion-token contract envelope. Normal science can start only after
the preflight seals a fresh observed P95 plus 25% authority and the global
post-gate receipt accepts the model. The capability audit itself is not a
normal science dispatch reservation.

## Budget and denominator

The cumulative hosted limits are fixed at USD 500, 7,500 completions, and
5,000,000,000 storage bytes. V2.11.1 begins with the exact parent debit:

| Resource | Parent debit | Remaining before fresh calls |
|---|---:|---:|
| Hosted USD | 17.166524062500006 | 482.8334759375 |
| Hosted completions | 876 | 6,624 |
| Storage bytes | 217,581,135 | 4,782,418,865 |

The registered fresh matrix contains 5,880 hosted calls: 64 preflight calls
and 5,816 scientific calls. If all registered calls run, cumulative hosted
completions are 6,756, leaving 744. This call headroom is not an adaptive retry
pool. After preflight, the sealed model-by-call-kind P95 projection plus 25%
must fit the cumulative USD, completion, storage, and stage caps. If it does
not fit, the result is a budget no-go; do not drop arms, models, or seeds.

All 136 model-arm-seed cells remain in the ITT denominator. Provider, parse,
budget, and integrity failures are terminal outcomes, not deletion criteria.
Use `--resume` to preserve completed cells and recover or terminalize existing
reservations. Never redispatch a terminal cell, replace a failed seed, increase
the 4,096 output cap after observing a failure, or run an unregistered paid
probe. One provider attempt and no fallback remain mandatory.

## Release gate

No paid command may run from the repair branch, a dirty checkout, a lightweight
tag, or a commit other than the peeled annotated-tag commit. Before merging,
run the same local gates as CI:

```bash
cd '/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-1'
python scripts/render_pilot_v2111_contract.py \
  --status frozen \
  --test-count 1446 \
  --test-collection-sha256 e41477ac9cfe33942ac40eae2bcc08f8b077c96ce609872765e4b8a12beea1ab \
  --compiled-source-count 232 \
  --compiled-source-inventory-sha256 4655c91198fd3c151706481183e7970e7311157c35b49974106e43bf2e21adda \
  --sealed-manifest-inventory-sha256 b5c5a817d09d10752c1f5f00ba556b417d16e06c64b5fcbb15671e49a1d81952 \
  --output /tmp/pilot_v2_11_1.rendered.json
cmp /tmp/pilot_v2_11_1.rendered.json experiments/pilot_v2_11_1.yaml
python -m pytest -q -p no:cacheprovider
python -m verified_memory.ci_release_receipt collect-tests \
  --output /tmp/finevo-v2111-test-collection.json
python -m verified_memory.ci_release_receipt compile-sources \
  --output /tmp/finevo-v2111-python-sources.json
git diff --check
git status --short
```

The frozen render command, complete pytest run, compilation/source inventory, six
tracked manifest anchors, high-confidence secret scan, and both GitHub CI jobs
(`ubuntu-24.04` and `macos-14`) must pass. `git status --short` is expected to
show the intended patch before commit; it must be empty in the eventual science
checkout. Do not print or commit `.env`.

The draft renderer remains independently testable, but its output is
intentionally not byte-equal to the tracked frozen contract. No expected-CI
field in the tracked contract may be null. If any test node ID or tracked
Python path changes, recollect the inventories, rerender the contract, update
its canonical constant, and rerun all gates.

Push the repair branch, merge it through the normal reviewed PR, wait for the
merged `main` CI to be green, and only then create and push the annotated tag.
The primary worktree may contain deliberately preserved unrelated files, so
the tag can be created directly at the verified remote-main commit without
switching or cleaning that worktree:

```bash
cd '/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-1'
git fetch origin main --tags
release_commit="$(git rev-parse origin/main)"
git merge-base --is-ancestor HEAD "${release_commit}"
git tag -a pilot-v2.11.1-science \
  -m 'FinEvo preregistered V2.11.1 bootstrap repair science release' \
  "${release_commit}"
git cat-file -t pilot-v2.11.1-science
git rev-parse origin/main
git rev-parse 'pilot-v2.11.1-science^{commit}'
git push origin refs/tags/pilot-v2.11.1-science
git worktree add \
  '/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-1-science' \
  pilot-v2.11.1-science
```

`git cat-file -t` must print `tag`; the two commit hashes must be identical.
Do not move or recreate the tag after any paid call.

## Paid launch from the clean release

Reuse the existing OpenAI key without displaying it. V2.11.1 has only direct
OpenAI hosted profiles, so remove unrelated route keys from the launch shell:

```bash
cd '/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-1-science'
test -z "$(git status --porcelain)"
test "$(git cat-file -t pilot-v2.11.1-science)" = tag
test "$(git rev-parse HEAD)" = "$(git rev-parse 'pilot-v2.11.1-science^{commit}')"

python -m verified_memory.scientific_release_attestation \
  prepare-scientific-launch \
  --contract experiments/pilot_v2_11_1.yaml \
  --run-id '<verified-main-ci-run-id>' \
  --run-attempt '<verified-main-ci-run-attempt>' \
  --output experiment_results/pilot-v2.11.1/raw/scientific_launch_input.json
python -m verified_memory.scientific_release_attestation \
  verify-scientific-launch \
  --contract experiments/pilot_v2_11_1.yaml \
  --input experiment_results/pilot-v2.11.1/raw/scientific_launch_input.json

set -a
source '/Users/guanghaowu/Develop/financial world/baselines/eccv26_EconAgent/.env'
set +a
unset OPENROUTER_API_KEY GEMINI_API_KEY
test -n "${OPENAI_API_KEY:-}"
```

Run the two zero-call imports first. The parent root is accepted only by the
parent-import command:

```bash
python run_pilot.py \
  --contract experiments/pilot_v2_11_1.yaml \
  --stage parent-import \
  --parent-repo-root '/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-science' \
  --resume

python run_pilot.py \
  --contract experiments/pilot_v2_11_1.yaml \
  --stage capability-gate \
  --resume
```

At this checkpoint, the parent and two capability cells should be terminal and
there must have been zero fresh provider calls. Verify the terminal stage
receipts, `run_ledger.json`, `budget_ledger.json`, imported wrapper hashes, and
`v2111_contract_envelope_bootstrap.json` before continuing.

Launch exactly the fresh 64-call preflight:

```bash
python run_pilot.py \
  --contract experiments/pilot_v2_11_1.yaml \
  --stage long-context-preflight \
  --resume
```

Do not launch science unless both preflight cells are terminal, every call is
accounted, and these artifacts verify together:

- `long-context-preflight/stage_receipt.json`;
- `long-context-preflight/post_gate_authority.json`;
- each run's `observed_p95_authority_receipt.json`;
- top-level `run_ledger.json` and `budget_ledger.json`.

If the post-gate is a go, preserve the contract dependency order C, A, D, B,
then the GPT-5.6 cross-model sentinel:

```bash
for stage in experiment-c experiment-a experiment-d experiment-b cross-model; do
  python run_pilot.py \
    --contract experiments/pilot_v2_11_1.yaml \
    --stage "${stage}" \
    --resume || break
done
```

The loop stops on the first CLI failure; stage receipts and ledgers determine
whether resumption is valid. Shell exit success is not scientific evidence.
Paper claims remain unopened until all registered denominators are terminal
and the reviewer package, checksums, failure ledger, and claim-to-artifact table
have been generated and independently verified.
