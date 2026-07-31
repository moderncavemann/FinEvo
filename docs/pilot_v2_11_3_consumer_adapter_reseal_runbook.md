# FinEvo V2.11.3 consumer-adapter reseal runbook

Status: freeze, release, zero-call authority reseal, and scientific launch
procedure. V2.11.3 is a fresh preregistered denominator. It is not an in-place
resume, retry, or reinterpretation of V2.11.2.

## Frozen boundary

V2.11.2 remains immutable at annotated tag `pilot-v2.11.2-science`, tag object
`1b9d9f163934e946255ec19aeebe2f121fba4cc3`, and peeled commit
`78870956b528946d415a9be5f5769b0893d16d74`. Its 136 cells are terminal: 10
complete and 126 failed. The two fresh long-context preflights made 64 hosted
calls and produced a valid two-model observed-P95 authority. The 131 scientific
cells made zero provider calls; five offline candidate-admission cells completed
before the remaining 126 cells failed closed at the consumer schema boundary.

The V2.11.2 failure signature is a dispatch-interface failure, not evidence
about model reasoning or FinEvo effectiveness: the generic consumer did not
recognize the verified `finevo-pilot-v2.11.2-post-gate-authority-v1` schema.
V2.11.3 adds its dedicated consumer adapter and changes no prompt, seed, arm,
model, environment, metric, or scientific stop/go rule.

| Item | V2.11.3 value |
|---|---|
| Contract | `experiments/pilot_v2_11_3.yaml` |
| Contract ID | `finevo-pilot-v2.11.3` |
| Source manifest | `experiments/pilot_v2_11_3_source_manifest.json` |
| Required annotated tag | `pilot-v2.11.3-science` |
| Raw namespace | `experiment_results/pilot-v2.11.3/raw/` |
| Immutable parent checkout | `/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-2-science` |
| Repair worktree | `/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-3` |
| Clean science worktree | `/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-3-science` |
| Ledger denominator | 136 cells: 5 operational and 131 scientific |

The V2.11.3 source manifest is independently anchored by file SHA-256
`f05dbac4951e99476c06883e3c1b792e7ccb459c16eb4d78ac15ddf7905598de`
and canonical content SHA-256
`5c8e554d1a00803b81deb4f31b4a87ddf54a272861a7c750985cd72b18a95f00`.
CI verifies this source authority separately from the unchanged six sealed-run
manifests. The two categories use different schemas and must not be merged.

Only the source-manifest-bound calibration wrapper, two capability wrappers,
two preflight dispatch reservations, immutable parent release identity, and
cumulative parent debit cross the boundary. No decoded completion, scientific
outcome, checkpoint, A-D cell, cross-model cell, or V2.11.2 treatment-effect
direction is imported.

## Budget, denominator, and no-retry rules

The cumulative hard limits are USD 500, 7,500 hosted completions, and
5,000,000,000 storage bytes. The immutable parent debit is USD
19.998220562500006, 1,004 hosted completions, and 221,668,707 bytes. Thus the
fresh hosted USD bucket is at most 480.0017794375. The registered science matrix
contains 5,816 fresh provider calls; if all run, the cumulative completion count
is 6,820, leaving 680 calls of non-adaptive headroom.

That headroom is not a retry pool. Every one of the 131 scientific cells remains
in the V2.11.3 ITT denominator. A provider, parse, budget, or integrity failure
terminates the registered cell and stays in the ledger. Do not replace a seed,
drop a failed cell, shrink the matrix, change a model or route, reduce reasoning,
or lower an output cap. `--resume` only continues untouched cells or verifies a
valid in-progress reservation; it never redispatches a terminal cell. Unknown
prices or a 1.25x observed-P95 projection that exceeds any frozen cap stop before
provider construction.

## Draft freeze and local release gate

This subsection is the one-time transition from the staged draft candidate to
the frozen release. Once the tracked contract is frozen, do not rerun the draft
render/`cmp`; use the ordinary pinned frozen render and post-freeze checks below.

No provider key is needed or permitted during freeze. Before collecting final
inventories, add every intended source, test, contract, manifest, workflow, and
runbook path to the Git index so `git ls-files` sees the release candidate. Never
stage `.env`, ignored raw results, or unrelated workspace files.

```bash
cd '/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-3'
unset OPENAI_API_KEY OPENROUTER_API_KEY ANTHROPIC_API_KEY GOOGLE_API_KEY GEMINI_API_KEY

python scripts/render_pilot_v2113_contract.py \
  --status draft \
  --output /tmp/pilot_v2_11_3.draft.json
cmp /tmp/pilot_v2_11_3.draft.json experiments/pilot_v2_11_3.yaml

python -m verified_memory.ci_release_receipt verify-source-manifests \
  --output /tmp/finevo-v2113-source-manifests.json
python -m verified_memory.ci_release_receipt collect-tests \
  --output /tmp/finevo-v2113-test-collection.json
python -m pytest -q -p no:cacheprovider
PYTHONPYCACHEPREFIX=/tmp/finevo-v2113-compile-cache \
  python -m verified_memory.ci_release_receipt compile-sources \
  --output /tmp/finevo-v2113-python-sources.json
git diff --check
git status --short
```

Collect the unchanged sealed-run inventory digest separately:

```bash
python - <<'PY'
import json
from pathlib import Path
from verified_memory.scientific_release_attestation import (
    discover_scientific_manifest_paths,
    sealed_manifest_inventory,
)

root = Path.cwd()
paths = discover_scientific_manifest_paths(root)
rows, digest = sealed_manifest_inventory(root, paths)
assert len(rows) == 6
Path('/tmp/finevo-v2113-sealed-manifests.json').write_text(
    json.dumps({'count': len(rows), 'sha256': digest}, sort_keys=True) + '\n',
    encoding='utf-8',
)
PY
```

Read the five final CI values from those three JSON inventories. Freezing is a
two-stage bootstrap because the production parser must reject a frozen contract
until its canonical hash is pinned. First render an explicitly unpinned
candidate outside the repository contract path. Substitute only the exact
locally collected values:

```bash
python scripts/render_pilot_v2113_contract.py \
  --status frozen \
  --frozen-candidate \
  --test-count FINAL_TEST_COUNT \
  --test-collection-sha256 FINAL_TEST_COLLECTION_SHA256 \
  --compiled-source-count FINAL_COMPILED_SOURCE_COUNT \
  --compiled-source-inventory-sha256 FINAL_COMPILED_SOURCE_INVENTORY_SHA256 \
  --sealed-manifest-inventory-sha256 FINAL_SEALED_MANIFEST_INVENTORY_SHA256 \
  --output /tmp/pilot_v2_11_3.frozen-candidate.json
```

Candidate mode temporarily checks the candidate against its independently
computed canonical hash while retaining every other `PilotContract` invariant;
it restores the unpinned process state before exit. It refuses
`experiments/pilot_v2_11_3.yaml` as an output and cannot authorize dispatch.

Copy only the candidate's
`integrity.declared_sha256` into
`PILOT_CONTRACT_V2_11_3_CANONICAL_SHA256` in
`verified_memory/pilot_contract.py`. Do not change any science-design field or
CI value. Then use the ordinary pinned path to render first to a second temporary
file and require exact equality:

```bash
python scripts/render_pilot_v2113_contract.py \
  --status frozen \
  --test-count FINAL_TEST_COUNT \
  --test-collection-sha256 FINAL_TEST_COLLECTION_SHA256 \
  --compiled-source-count FINAL_COMPILED_SOURCE_COUNT \
  --compiled-source-inventory-sha256 FINAL_COMPILED_SOURCE_INVENTORY_SHA256 \
  --sealed-manifest-inventory-sha256 FINAL_SEALED_MANIFEST_INVENTORY_SHA256 \
  --output /tmp/pilot_v2_11_3.frozen-pinned.json
cmp /tmp/pilot_v2_11_3.frozen-candidate.json \
  /tmp/pilot_v2_11_3.frozen-pinned.json

python scripts/render_pilot_v2113_contract.py \
  --status frozen \
  --test-count FINAL_TEST_COUNT \
  --test-collection-sha256 FINAL_TEST_COLLECTION_SHA256 \
  --compiled-source-count FINAL_COMPILED_SOURCE_COUNT \
  --compiled-source-inventory-sha256 FINAL_COMPILED_SOURCE_INVENTORY_SHA256 \
  --sealed-manifest-inventory-sha256 FINAL_SEALED_MANIFEST_INVENTORY_SHA256 \
  --output experiments/pilot_v2_11_3.yaml
cmp /tmp/pilot_v2_11_3.frozen-pinned.json experiments/pilot_v2_11_3.yaml
```

The ordinary frozen render must still fail while the canonical constant is
unset or wrong; never use candidate mode after pinning. Repeat source-manifest
verification, full pytest, test collection, tracked-source compilation, secret
scan, and `git diff --check`. The final test and Python path inventories must
reproduce the contract values exactly. The source manifest must reproduce its
one-entry CI inventory, while the sealed-run inventory must remain exactly six
entries.

The two freeze edits occur after the initial release-candidate staging, so they
must be staged again explicitly. Before committing, require the index to contain
the same frozen contract and canonical pin as the working tree:

```bash
git add -- experiments/pilot_v2_11_3.yaml verified_memory/pilot_contract.py
git diff --quiet
git diff --cached --check
python - <<'PY'
import json
import subprocess
from pathlib import Path

contract_path = Path("experiments/pilot_v2_11_3.yaml")
pin_path = Path("verified_memory/pilot_contract.py")
staged_contract_bytes = subprocess.check_output(
    ["git", "show", f":{contract_path.as_posix()}"]
)
staged_pin_bytes = subprocess.check_output(
    ["git", "show", f":{pin_path.as_posix()}"]
)
assert staged_contract_bytes == contract_path.read_bytes()
assert staged_pin_bytes == pin_path.read_bytes()
contract = json.loads(staged_contract_bytes)
declared = contract["integrity"]["declared_sha256"]
assert contract["status"] == "frozen"
assert all(contract["release_requirements"]["expected_ci"].values())
assert declared.encode("ascii") in staged_pin_bytes
PY
git status --short
```

The final status must contain only the intended staged release paths and no
unstaged or untracked file. A staged draft contract, null CI inventory, or
unpinned canonical hash is a release no-go even if the working-tree tests pass.

Commit and push the complete frozen patch, merge through the reviewed PR, and
wait for both merged-main CI jobs to pass:

- `Python 3.12.7 / ubuntu-24.04`;
- `Python 3.12.7 / macos-14`.

Each job must emit a scientific release receipt for the exact merged commit.
Receipt emission also reloads the frozen V2.11.3 contract and fails the job
unless the collected-test, tracked-Python, and six-manifest inventories exactly
match all five `release_requirements.expected_ci` values. This comparison must
pass before the annotated tag is created; the clean-tag launch attestation
repeats it later as an independent gate.

A generic GitHub checkout contains the annotated V2.11.2 tag but intentionally
does not contain its ignored 205-file raw tree. CI therefore verifies the exact
parent tag anchor and tracked source-manifest seals, while the 12 tests that
replay those ignored bytes report an explicit `requires the ignored hash-bound
parent raw tree` skip. They must pass, not skip, in the local release worktree
and in the clean-tag zero-call `parent-import` gate. A green CI job alone cannot
replace that parent replay or authorize a provider call.

Only then create and push the annotated tag and create the detached clean
science worktree:

```bash
set -euo pipefail
cd '/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-3'
feature_tip="$(git rev-parse HEAD)"
pr_number=PR_NUMBER
git fetch origin main --tags
release_commit="$(
  gh pr view "${pr_number}" \
    --repo moderncavemann/FinEvo \
    --json mergeCommit \
    --jq '.mergeCommit.oid'
)"
test -n "${release_commit}"
git merge-base --is-ancestor "${feature_tip}" "${release_commit}"
git merge-base --is-ancestor "${release_commit}" origin/main
run_id="$(
  gh run list \
    --repo moderncavemann/FinEvo \
    --workflow verified-memory-ci.yml \
    --branch main \
    --event push \
    --commit "${release_commit}" \
    --status success \
    --limit 1 \
    --json databaseId \
    --jq '.[0].databaseId'
)"
test -n "${run_id}"
test "$(gh run view "${run_id}" --repo moderncavemann/FinEvo \
  --json headSha --jq '.headSha')" = "${release_commit}"
for job in \
  'Python 3.12.7 / ubuntu-24.04' \
  'Python 3.12.7 / macos-14'; do
  test "$(gh run view "${run_id}" --repo moderncavemann/FinEvo \
    --json jobs --jq ".jobs[] | select(.name == \"${job}\") | .conclusion")" = success
done
git tag -a pilot-v2.11.3-science \
  -m 'FinEvo V2.11.3 consumer-adapter reseal science release' \
  "${release_commit}"
test "$(git cat-file -t pilot-v2.11.3-science)" = tag
test "$(git rev-parse 'pilot-v2.11.3-science^{commit}')" = "${release_commit}"
git push origin refs/tags/pilot-v2.11.3-science
git worktree add \
  '/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-3-science' \
  pilot-v2.11.3-science
```

Do not require the feature-branch `HEAD` itself to equal the release commit:
GitHub normally creates a distinct merge commit. The ancestry checks above bind
the reviewed feature tip to that exact PR merge commit, and the CI lookup binds
the tag candidate to the successful merged-main workflow rather than to a later
unrelated `main` commit.

The tag must never move after any provider call.

## Clean-tag attestation and zero-call reseal

Only the detached, clean tagged worktree may ever load the existing OpenAI key.
First attest the release and execute all three operational stages with provider
keys absent:

```bash
cd '/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-3-science'
test -z "$(git status --porcelain)"
test "$(git cat-file -t pilot-v2.11.3-science)" = tag
test "$(git rev-parse HEAD)" = \
  "$(git rev-parse 'pilot-v2.11.3-science^{commit}')"

python -m verified_memory.scientific_release_attestation \
  prepare-scientific-launch \
  --contract experiments/pilot_v2_11_3.yaml \
  --run-id VERIFIED_MAIN_CI_RUN_ID \
  --run-attempt VERIFIED_MAIN_CI_RUN_ATTEMPT \
  --output experiment_results/pilot-v2.11.3/raw/scientific_launch_input.json
python -m verified_memory.scientific_release_attestation \
  verify-scientific-launch \
  --contract experiments/pilot_v2_11_3.yaml \
  --input experiment_results/pilot-v2.11.3/raw/scientific_launch_input.json

unset OPENAI_API_KEY OPENROUTER_API_KEY ANTHROPIC_API_KEY GOOGLE_API_KEY GEMINI_API_KEY
python run_pilot.py \
  --contract experiments/pilot_v2_11_3.yaml \
  --stage parent-import \
  --parent-repo-root \
  '/Users/guanghaowu/Develop/financial world/worktrees/finevo-pilot-v2-11-2-science' \
  --resume
python run_pilot.py \
  --contract experiments/pilot_v2_11_3.yaml \
  --stage capability-gate \
  --resume
python run_pilot.py \
  --contract experiments/pilot_v2_11_3.yaml \
  --stage long-context-preflight \
  --resume
```

In V2.11.3, `long-context-preflight` means authority import and current-release
reseal. It must construct no provider and make zero fresh calls. Keep all
provider credentials unset and run the single executable acceptance gate:

```bash
python run_pilot.py \
  --contract experiments/pilot_v2_11_3.yaml \
  --accept-scientific-dispatch \
  --raw-root experiment_results/pilot-v2.11.3/raw \
  --scientific-launch-input \
  experiment_results/pilot-v2.11.3/raw/scientific_launch_input.json \
  --acceptance-output \
  experiment_results/pilot-v2.11.3/raw/scientific_dispatch_acceptance.json
```

The command atomically creates the deterministic, self-hashed
`scientific_dispatch_acceptance.json` receipt only after it verifies together:

- all five operational ledger cells are terminal complete;
- all operational reservations finalize with cost USD 0 and completions 0;
- `parent-import/stage_receipt.json`, `capability-gate/stage_receipt.json`, and
  `long-context-preflight/stage_receipt.json` agree with both top-level ledgers;
- `parent-import/parent_import_receipt.json` binds the frozen source manifest;
- both `imported_observed_p95/<model>/observed_p95_authority_receipt.json` and
  `projection_p95.json` pairs verify;
- `long-context-preflight/post_gate_authority.json` has `go=true`, both frozen
  model IDs eligible, and an all-zero current-attempt provider boundary;
- all 126 provider-backed runner configs round-trip with their sealed authority,
  all five 11-cell Experiment-D groups match the registered branch/narrative
  denominator, and all 81 budget units reproduce exactly;
- projected calls are exactly 4,848 action plus 968 semantic calls (5,816
  total), with every hard USD/completion/storage cap checked as one matrix;
- exactly 131 fresh scientific cells remain registered; the pre-science raw
  namespace contains no symlink, no A/B/C/D/cross-model science-stage path, and
  no path carrying a V2.11.2 scientific run identifier. The source-import
  receipt separately records zero imported scientific cells, summaries,
  outcomes, and decoded completions.

Any mismatch is a zero-call no-go. Do not load credentials or repair raw files
in place. The receipt has `scientific_evidence=false`; it authorizes dispatch
plumbing only and is not a scientific result. Every V2.11.3 science stage
(`experiment-c`, `experiment-a`, `experiment-d`, `experiment-b`, and
`cross-model`) re-verifies this exact contract/commit/ledger-bound receipt
by recomputing the configs, D groups, projections, release binding, authority
sources, and accepted ledger prefixes before provider catalog lookup, budget
reservation, or provider construction.
If the receipt is missing or drifts, the stage fails closed and preserves the
ITT denominator without making a provider call.

## Load the existing key and launch science

Only after the command exits zero and the immutable acceptance receipt reports
`status=go`, `go=true`, 131 scientific cells, and zero provider construction
or calls, load the existing key in the same clean tagged worktree without
printing it. Remove unrelated route keys:

```bash
set -a
source '/Users/guanghaowu/Develop/financial world/baselines/eccv26_EconAgent/.env'
set +a
unset OPENROUTER_API_KEY ANTHROPIC_API_KEY GOOGLE_API_KEY GEMINI_API_KEY
test -n "${OPENAI_API_KEY:-}"
test -z "$(git status --porcelain)"
```

Run science in the frozen dependency order C, A, D, B, then cross-model:

```bash
set -euo pipefail
for stage in experiment-c experiment-a experiment-d experiment-b cross-model; do
  python run_pilot.py \
    --contract experiments/pilot_v2_11_3.yaml \
    --stage "${stage}" \
    --resume
done
```

Stop at the first CLI failure. Do not automatically rerun the failed stage or
cell. A live process, release gate, imported authority, shell exit code, or
provider-call count is operational evidence only. Reviewer-facing A-D or
cross-model claims require terminal stage receipts, both ledgers, failure
records, checksums, all registered ITT cells, and verified aggregate artifacts.
If a preregistered mechanism gate fails, narrow the corresponding paper claim;
do not select a replacement seed or modify the matrix after observing outcomes.
