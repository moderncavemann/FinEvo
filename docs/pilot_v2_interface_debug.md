# Pilot-v2 interface debug boundary

The frozen `pilot-v1` run and its terminal no-go artifacts remain immutable.
They must not be rerun, relabelled, or interpreted as evidence about FinEvo
effectiveness.

## Pilot-v1 failures and current interface diagnosis

The frozen run proves that the registered capability/interface gate ended in
no-go. The provider-specific causes below were reproduced in bounded,
session-local probes. They become repository-verifiable debug findings only
after the clean-tag runner writes the redacted receipts described below.

- Direct GPT-5.2 and GPT-5.6 requests failed because Chat Completions rejected
  the explicit `temperature=0` field. Both models accepted the otherwise
  identical request after only that field was omitted.
- OpenRouter returned inline route metadata, but its documented optional
  `attempts` array was absent. The old validator required the optional field
  and collapsed the result into an undifferentiated `PilotContractError`.
- The local Ollama profile used prompt-only JSON guidance. Thirteen malformed
  capability outputs reached the 80-token cap, while the adapter discarded
  `done` and `done_reason`; the historical artifacts therefore cannot
  distinguish truncation from another parse failure.

These are interface and instrumentation findings. They do not establish model
capability, memory effectiveness, or any A-D mechanism effect.

## Debug implementation

- Strict GPT-5 requests omit only the unsupported `temperature` field and
  record `temperature_dispatch=omitted_unsupported`. `top_p`, seed, JSON mode,
  reasoning effort, and the completion cap remain explicit.
- Strict provider profiles require `openai==2.46.0` or `requests==2.34.2`
  before dispatch. The actual SDK name and version are retained per response.
- Provider exceptions retain only allowlisted status/code/parameter/request-ID
  fields. Exception messages, bodies, requests, headers, and tracebacks are not
  serialized.
- OpenRouter still fails closed on requested model, direct first-attempt route,
  provider pin, and served snapshot. The optional `attempts` array is validated
  when present; a successful `attempt=1` remains required.
- Ollama retains native finish metadata. Only an exact `stop` with `done=true`
  is accepted; `length` is an `IncompleteCompletionError` even if the partial
  text happens to parse.
- Capability gate v2 keeps every registered task in the ITT denominator while
  reporting interface validity, evaluable count, and conditional accuracy
  separately. Any interface failure makes the capability assessment
  `not_evaluable`.

## Safe live interface probe

`diagnose_provider_interface.py` is a diagnostic-only, one-call entry. It:

- requires a clean worktree at an annotated debug tag;
- permits at most one provider attempt and a `$0.10` caller cap;
- permits output only under
  `experiment_results/pilot-v2-debug/raw/interface/`, with no overwrite;
- accounts all probes in a `$0.30` cumulative diagnostic ledger, including a
  conservative `$0.01` reserve for the earlier session-local probes;
- writes only an ignored, redacted receipt with output hash/byte count;
- sets `scientific_evidence=false`, `diagnostic_only=true`, and
  `denominator_inclusion=false`.

Example:

```bash
python diagnose_provider_interface.py \
  --contract experiments/pilot_v1.yaml \
  --model-id gpt52_main \
  --required-tag pilot-v2-debug-interface-v2 \
  --output experiment_results/pilot-v2-debug/raw/interface/gpt52.json
```

For the historical prompt-only Ollama profile, a probe must explicitly add
`--force-json-object`. The override is recorded in the receipt and never
changes the frozen source contract.

Passing this probe is only an interface gate. Stage 0, A-D, narrative
interventions, and cross-model scientific cells remain prohibited until a new
contract, clean scientific tag, budget projection, and all registered gates
are separately complete.
