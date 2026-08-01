# V2.11.5 reviewer closed-loop trace

This publication consumer builds one deterministic, descriptive mechanism
trace from the sealed `pilot-v2.11.5-science` raw tree. It performs no model or
provider dispatch and does not alter any scientific run, denominator, receipt,
or gate.

## Fixed selection policy

The coordinate is derived from the frozen contract before any outcome stream
is read:

- stage: Experiment A;
- model: its sole primary model, `gpt52_main`;
- arm: `full`;
- narrative / utility: `none` / `stage0-selected`;
- seed: first entry of `seeds.sets.main`;
- agent: minimum agent ID;
- decision: start of the first registered recovery interval;
- continuation: the immediately following decision.

The selection occurred at publication time after the science seal, was not
preregistered, and was made with prior human awareness of the case. Outcome,
action, retrieval-score, rule-status, provider-status, and wealth fields are
explicitly excluded from the selector. If the fixed coordinate is unavailable,
the builder emits an `unavailable` artifact and never searches for a substitute.

## Evidence boundary

The artifact binds every copied observation to an exact source file SHA-256,
one-based JSONL line number, and raw-line SHA-256. Cross-stream checks connect
the registered shock, decision-time context, retrieved episodes and rule,
prompt hashes, frozen provider record, parsed/executed action, utility ledger,
next state, verifier events, retirement, and the next retrieval.

`publication_provider_calls=0` means the publication build made no new provider
calls. Fields named `frozen_source_provider_call` are historical records copied
from the sealed science run; they do not represent publication-time requests.

The result is a single observational trace. It supports traceability only. It
does not establish that retrieval or a rule caused the action, that the focal
action caused the macro transition, that the verifier is effective, or that the
case is representative. Experiment A remains `complete-with-no-go` with 17
complete and 3 failed ITT cells; this artifact cannot reverse that receipt.

## Build

Run from a tracked-clean publication-consumer checkout whose commit descends
from `34134f2624833e45f0e1f559332b0d11ea1942d6`:

```bash
python scripts/build_v2115_reviewer_trace.py \
  --source-repo-root /path/to/detached/pilot-v2.11.5-science \
  --output-dir evidence/current_v2/pilot-v2.11.5-reviewer-trace-v1
```

The source checkout must be detached at the annotated science tag and
tracked-clean. The builder publishes logical source/publisher IDs, commits,
tag objects, tracked blob IDs, relative source paths, and content hashes. It
does not publish host-specific absolute worktree paths or branch names, so the
same commits can be reproduced from a branch or detached checkout.

The output directory is no-overwrite and contains the trace JSON, a copy of its
JSON Schema, and file checksums.
