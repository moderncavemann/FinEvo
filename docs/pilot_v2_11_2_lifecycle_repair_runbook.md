# FinEvo V2.11.2 lifecycle repair runbook

Status: release and launch procedure. V2.11.2 is a preregistered lifecycle
validator repair with a fresh denominator and raw namespace. It is not a
continuation, retry in place, or reinterpretation of the terminal V2.11.1 run.

## Frozen boundary

V2.11.1 remains immutable at annotated tag `pilot-v2.11.1-science`, tag object
`c12f6bd5b74cb676109b83fcbfdb4376adf7abdf`, and peeled commit
`e9871353ad307fdd134f3c74764d201efbc81081`. Its 136-cell ledger ended with
3 complete, 2 failed, and 131 integrity-stopped cells. The two failed
long-context preflight cells made exactly 64 paid provider calls. Those calls,
their terminal failures, and their journals remain in the V2.11.1 ITT and
budget records; they are audit-only inputs to V2.11.2.

V2.11.2 uses a new contract, annotated tag, denominator, and raw namespace:

| Item | V2.11.2 value |
|---|---|
| Contract | `experiments/pilot_v2_11_2.yaml` |
| Contract ID | `finevo-pilot-v2.11.2` |
| Required annotated tag | `pilot-v2.11.2-science` |
| Raw namespace | `experiment_results/pilot-v2.11.2/raw/` |
| Immutable parent checkout | `/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-1-science` |
| Repair worktree | `/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-2` |
| Clean science worktree | `/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-2-science` |
| Ledger denominator | 136 cells: 5 operational and 131 scientific |

The repair changes one validator condition only: an already active semantic
rule may remain active while its current score is below the admission threshold
but still above the unchanged retirement threshold. Historical
activation-threshold crossing, unique activation-event replay, post-proposal
activation evidence, retirement policy, prompts, seeds, arms, models,
environment, and metrics remain unchanged.

Only the source-manifest-bound calibration authority, the two passed V2.11.1
capability authorities, the terminal parent boundary, and the cumulative parent
budget debit may cross into V2.11.2. In particular, V2.11.1's 64-call failed
preflight journals are not observed-P95 samples, checkpoint evidence, exactness
evidence, reusable preflight cells, or scientific samples. No old treatment
effect or outcome direction is imported or reclassified.

The two import stages are provider-free:

1. `parent-import` replays and seals the immutable V2.11.1 source manifest and
   parent receipt: zero provider construction and zero fresh calls.
2. `capability-gate` imports the two passed capability authorities: zero
   provider construction and zero fresh calls. Their historical calls stay in
   the cumulative parent debit and are never redispatched.

V2.11.2 then runs a new `long-context-preflight`: two models, each with 24
actor-action and 8 semantic-proposal calls, for exactly 64 fresh hosted calls.
It must create new V2.11.2 checkpoint, exactness, journal, observed-P95, and
post-gate artifacts in the new raw namespace. Normal science has no dispatch
authority until the new global post-gate receipt accepts the model and its
fresh model-by-call-role observed P95 plus 25% projection fits every frozen
cap.

## Budget and denominator

The cumulative hosted limits are USD 500, 7,500 provider completions, and
5,000,000,000 storage bytes. V2.11.2 starts with this immutable parent debit:

| Resource | Parent debit | Remaining before V2.11.2 calls |
|---|---:|---:|
| Hosted USD | 18.586399812500005 | 481.4136001875 |
| Hosted completions | 940 | 6,560 |
| Storage bytes | 217,838,625 | 4,782,161,375 |

The registered fresh matrix contains exactly 5,880 hosted calls: 64 fresh
preflight calls and 5,816 scientific calls. If every registered call runs, the
cumulative total is 6,820, leaving 680 completions of headroom. This headroom
is not an adaptive retry pool. Unknown prices stop before dispatch. After the
fresh preflight, the sealed P95 projection plus 25% must fit the cumulative
USD, completion, storage, and stage caps. A projection failure is a no-go; do
not shrink the matrix, drop models or arms, replace seeds, reduce reasoning, or
silently alter output limits.

All 136 V2.11.2 model-arm-seed cells stay in its ITT denominator. Provider,
parse, budget, and integrity failures are terminal outcomes, not deletion
criteria. `--resume` may recover a valid in-progress reservation, but it must
never redispatch a terminal cell. One provider attempt, no fallback, and the
frozen provider route remain mandatory.

## Freeze and local release gate

The tracked V2.11.2 contract is a draft until the complete source and test patch
is final. No paid command may use that draft. Before collecting the final
inventories, ensure every intended new source and test path has been added to
the Git index so `git ls-files` sees it. Do not stage `.env`, raw results, or
unrelated files.

From the repair worktree, run the provider-free local gate and collect the
candidate inventories:

```bash
cd '/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-2'
python scripts/render_pilot_v2112_contract.py \
  --status draft \
  --output /tmp/pilot_v2_11_2.draft.json
cmp /tmp/pilot_v2_11_2.draft.json experiments/pilot_v2_11_2.yaml
python -m verified_memory.ci_release_receipt collect-tests \
  --output /tmp/finevo-v2112-test-collection.json
python -m pytest -q -p no:cacheprovider
PYTHONPYCACHEPREFIX=/tmp/finevo-v2112-compile-cache \
  python -m verified_memory.ci_release_receipt compile-sources \
  --output /tmp/finevo-v2112-python-sources.json
git diff --check
git status --short
```

The following five expected-CI values were collected from the tracked draft
implementation and the verified six-manifest inventory. If any test node ID or
tracked Python path changes, recollect them before rerendering the contract.

```bash
python scripts/render_pilot_v2112_contract.py \
  --status frozen \
  --test-count 1553 \
  --test-collection-sha256 e599e883988ad04dd1eb40f6810902cc60fb91495c042486c335fbe08ecc33fe \
  --compiled-source-count 245 \
  --compiled-source-inventory-sha256 \
  4ca58622096459153ea503b26d2d7cfe41dd09b1e1ebc1e16901fcbdea9c55a1 \
  --sealed-manifest-inventory-sha256 \
  b5c5a817d09d10752c1f5f00ba556b417d16e06c64b5fcbb15671e49a1d81952 \
  --output experiments/pilot_v2_11_2.yaml
python scripts/render_pilot_v2112_contract.py \
  --status frozen \
  --test-count 1553 \
  --test-collection-sha256 e599e883988ad04dd1eb40f6810902cc60fb91495c042486c335fbe08ecc33fe \
  --compiled-source-count 245 \
  --compiled-source-inventory-sha256 \
  4ca58622096459153ea503b26d2d7cfe41dd09b1e1ebc1e16901fcbdea9c55a1 \
  --sealed-manifest-inventory-sha256 \
  b5c5a817d09d10752c1f5f00ba556b417d16e06c64b5fcbb15671e49a1d81952 \
  --output /tmp/pilot_v2_11_2.frozen.json
cmp /tmp/pilot_v2_11_2.frozen.json experiments/pilot_v2_11_2.yaml
python -m pytest -q -p no:cacheprovider
python -m verified_memory.ci_release_receipt collect-tests \
  --output /tmp/finevo-v2112-test-collection.final.json
PYTHONPYCACHEPREFIX=/tmp/finevo-v2112-compile-cache-final \
  python -m verified_memory.ci_release_receipt compile-sources \
  --output /tmp/finevo-v2112-python-sources.final.json
git diff --check
```

The first frozen renderer invocation is the deliberate draft-to-frozen
contract update; the second must reproduce it byte for byte. The final
collection and source receipts must reproduce the five frozen values. The full
pytest run, compile inventory, six tracked sealed-manifest anchors,
high-confidence secret scan, deterministic local G0 checks, and both GitHub CI
jobs (`Python 3.12.7 / ubuntu-24.04` and `Python 3.12.7 / macos-14`) must pass.
Do not print, copy, or commit `.env`.

Commit the complete frozen patch, push the repair branch, merge it through the
normal reviewed PR, and wait for the merged `main` CI run to be green. Confirm
that both required jobs emitted scientific release receipts for the exact
merged commit. Only then create the annotated release tag and clean science
worktree:

```bash
cd '/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-2'
git fetch origin main --tags
release_commit="$(git rev-parse origin/main)"
git merge-base --is-ancestor HEAD "${release_commit}"
git tag -a pilot-v2.11.2-science \
  -m 'FinEvo preregistered V2.11.2 lifecycle repair science release' \
  "${release_commit}"
test "$(git cat-file -t pilot-v2.11.2-science)" = tag
test "$(git rev-parse origin/main)" = \
  "$(git rev-parse 'pilot-v2.11.2-science^{commit}')"
git push origin refs/tags/pilot-v2.11.2-science
git worktree add \
  '/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-2-science' \
  pilot-v2.11.2-science
```

The tag must peel to the verified remote-main commit and must never be moved or
recreated after a paid call. The clean science worktree must remain detached at
that annotated tag, clean, and separate from the repair and primary worktrees.

## Paid launch from the clean release

Reuse the existing OpenAI key without displaying it. V2.11.2 has direct OpenAI
hosted profiles, so remove unrelated route keys from the launch shell. Perform
the attestation before loading credentials:

```bash
cd '/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-2-science'
test -z "$(git status --porcelain)"
test "$(git cat-file -t pilot-v2.11.2-science)" = tag
test "$(git rev-parse HEAD)" = \
  "$(git rev-parse 'pilot-v2.11.2-science^{commit}')"

python -m verified_memory.scientific_release_attestation \
  prepare-scientific-launch \
  --contract experiments/pilot_v2_11_2.yaml \
  --run-id '<verified-main-ci-run-id>' \
  --run-attempt '<verified-main-ci-run-attempt>' \
  --output experiment_results/pilot-v2.11.2/raw/scientific_launch_input.json
python -m verified_memory.scientific_release_attestation \
  verify-scientific-launch \
  --contract experiments/pilot_v2_11_2.yaml \
  --input experiment_results/pilot-v2.11.2/raw/scientific_launch_input.json

set -a
source '/Users/guanghaowu/Develop/financial world/baselines/eccv26_EconAgent/.env'
set +a
unset OPENROUTER_API_KEY GEMINI_API_KEY
test -n "${OPENAI_API_KEY:-}"
```

Run the two zero-provider import stages first. Only `parent-import` may receive
the immutable parent checkout path:

```bash
python run_pilot.py \
  --contract experiments/pilot_v2_11_2.yaml \
  --stage parent-import \
  --parent-repo-root '/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-1-science' \
  --resume

python run_pilot.py \
  --contract experiments/pilot_v2_11_2.yaml \
  --stage capability-gate \
  --resume
```

Before preflight, verify that the parent and capability cells are terminal,
that fresh provider-call count is zero, and that these artifacts agree:

- `parent-import/stage_receipt.json`;
- `capability-gate/stage_receipt.json`;
- top-level `run_ledger.json` and `budget_ledger.json`;
- the source manifest and imported wrapper file/content hashes;
- `v2112_contract_envelope_bootstrap.json`.

Then launch exactly the fresh 64-call V2.11.2 preflight:

```bash
python run_pilot.py \
  --contract experiments/pilot_v2_11_2.yaml \
  --stage long-context-preflight \
  --resume
```

Do not launch science unless both fresh preflight cells are terminal, all 64
fresh calls are accounted, and the following V2.11.2 artifacts verify
together:

- `long-context-preflight/stage_receipt.json`;
- `long-context-preflight/post_gate_authority.json`;
- each fresh run's checkpoint and exactness receipts;
- each fresh run's `observed_p95_authority_receipt.json`;
- top-level `run_ledger.json` and `budget_ledger.json`.

No file under the V2.11.1 raw namespace can satisfy this gate. If the new
post-gate is a global no-go, the 131 registered scientific cells must remain
terminally stopped in the V2.11.2 denominator. If it is a go, preserve the
contract dependency order C, A, D, B, then the GPT-5.6 cross-model sentinel:

```bash
for stage in experiment-c experiment-a experiment-d experiment-b cross-model; do
  python run_pilot.py \
    --contract experiments/pilot_v2_11_2.yaml \
    --stage "${stage}" \
    --resume || break
done
```

The loop stops at the first CLI failure. Shell success, release gates, imported
capability competence, or a healthy running process are not scientific
evidence. Only terminal V2.11.2 receipts, ledgers, checksums, failure records,
and verified aggregate artifacts can support the A-D or cross-model claims.
The older 64-call failure audit can explain provenance and the repair, but it
cannot support P95, checkpoint restoration, exactness, mechanism, performance,
or narrative-response claims.
