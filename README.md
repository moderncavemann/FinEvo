# FinEvo EMNLP Experiment Toolkit

Experiment runner and output normalizer for the FinEvo EMNLP revision.
Tracked aggregate results and sealed run/replay artifacts are historical
pre-P0 evidence. No current-method performance result is claimed here.

## Evidence-Grounded Rule Memory under Endogenous Multi-Agent Feedback

This is the current research-method name. Code retains `verified_memory` and
`VerifiedDualTrackMemory` identifiers for compatibility, but `verified` means
evidence-consistency and provenance checks only. It does not mean truth,
correctness, hallucination immunity, economic optimality, or causal
identification.

The method is implemented as a separate, auditable path. The legacy
`simulate.py` and `memory_module.py` remain available for reproduction; current
method work uses:

- `verified_memory/m1_context.py`: causal context encoding with independent
  retrieval/prompt routes;
- `verified_memory/m2_episodic.py`: staged `action_t` and finalized
  `outcome_t+1` evidence;
- `verified_memory/m3_semantic.py`: LLM candidates with support,
  counterevidence, provisional activation, and retirement;
- `verified_memory/m0_utility.py`: evaluation-only cash-flow and flow-utility
  ledger;
- `simulate_verified.py`: direct-hours, hard-budget, bounded runner;
- `replay_verified.py`: hash-bound five-treatment paired replay.

The durable completion ledger and current claim boundary are in
[`docs/p0_methodology_completion_audit.md`](docs/p0_methodology_completion_audit.md).
The implementation specification is in
[`docs/verified_dual_track_architecture.md`](docs/verified_dual_track_architecture.md).
Current status is **locally validated and code-ready for a small matched
pilot**; the status of any pushed commit must be read from its remote CI checks.
`Code-ready` does not mean scientifically validated. Experiments A-D, utility
calibration/sensitivity, a second model, a method-matched 5-seed result, and a
full run have not been completed for this method. If that pilot uses a hosted
provider, its first bounded stage is the preregistered API preflight smoke; no
current-method API smoke is claimed by this repository state.

### Preregistered mechanism micro-pilot

The only scientific-pilot entry point is:

```bash
python run_pilot.py \
  --contract experiments/pilot_v2_2.yaml \
  --stage <stage> \
  --resume
```

The V2.2 frozen order is `capability-gate`, `closed-loop-preflight`,
`secondary-capability-gate`, `secondary-closed-loop-preflight`,
`q-ref-resolution`, `stage0-calibration`, `experiment-a`, `experiment-c`,
`experiment-d`, `experiment-b`, `controlled-second`, and
`cross-model-diagnostics`. The runner registers the complete 174-cell ITT
denominator before dispatch and stops rather than dropping seeds, changing
models, reducing reasoning, or silently substituting routes when a capability,
provenance, integrity, or budget gate fails. V1 remains read-only compatibility
evidence. The original V2 launch is also immutable: its GPT-5.2 capability
probe was not evaluable after a malformed credential header, while the valid
local-Llama capability result was a no-go (12/12 utility ranking, 10/12 rule
application, 0/6 under the historical hidden-exact-match evaluator). V2.1
preserves both outcomes and contains the one preregistered GPT-5.2 operational
retry. That retry produced 30/30 strict structured responses, but the same
hidden-exact-match diagnostic again scored proposals 0/6.

V2.2 does not rewrite either release. Its tracked evaluator-amendment receipt
shows that all twelve source proposal rows (six per model) were strict,
schema-valid, verifier-admitted provisional candidates with three registered
support IDs. It therefore applies the preregistered
`semantic_candidate_acceptance_required` gate uniformly, retains the old 0/6
scores as diagnostics, and imports corrected 6/6 capability denominators with
zero provider calls. This correction inspected capability outcomes but no
scientific-effect outcome. Passing it is permission to attempt the separately
registered closed-loop preflight, not evidence that FinEvo improves utility,
wealth, robustness, or rule reliability.

Real V2.2 stages require a clean checkout at the peeled annotated
`pilot-v2.2-science` tag,
the same commit on `origin/main`, the remote annotated tag, and successful
Python 3.12.7 CI jobs on both `ubuntu-24.04` and `macos-14`. Provider runs use
the sealed model-by-call-kind preflight p95 plus 25% reservation; unknown hosted
prices and missing route metadata fail before scientific dispatch. The frozen
7,500-completion ceiling counts hosted OpenAI/OpenRouter calls only; local
Ollama and deterministic scripted calls remain in each run's operational call
limit and ITT ledger but cannot consume or obscure that hosted cap.

Exercise all A–D paths without network access or scientific claims:

```bash
python run_pilot.py \
  --contract experiments/pilot_v2_2.yaml \
  --stage development-a-d \
  --development-fake \
  --resume
```

The failed parent attempt remains under
`experiment_results/pilot-v2/raw/` and is never rewritten. New raw state is
ignored under `experiment_results/pilot-v2.2/raw/`. Only
validated contracts, aggregates, checksums, failure ledgers, and reviewer
reports may enter `evidence/current_v2/pilot-v2.2/`. Historical artifacts remain
separate and cannot satisfy this pilot contract. This is a 4-agent × 12-month
mechanism micro-pilot, not the 10×24×5 confirmatory design and not a 100×240
simulation.

After every preregistered cell has a terminal ITT ledger row, build the
zero-provider reviewer package through the same entry point:

```bash
python run_pilot.py \
  --contract experiments/pilot_v2_2.yaml \
  --stage publish-evidence \
  --resume
```

The publisher refuses to overwrite an existing package and reports
`complete-with-no-go` when the denominator is complete but a preregistered
scientific claim gate is not supported.

Budgeted calls use one HTTP attempt per reserved call; post-setup execution
failures write a content-addressed error/config/budget receipt. Preflight
failures remain stderr-only. Partial in-memory simulation streams are not yet
checkpointed on failure. Effective Foundation parameters,
request seed, served model, system fingerprint, cache usage, and provider
request ID are retained when available.

Run the diagnostic-provider integration check (its scripted path makes no
provider/network call; synthetic fixture, not a result). CI additionally
installs a fail-closed outbound-socket guard for this check:

```bash
python simulate_verified.py \
  --provider diagnostic \
  --run-id local-g0 \
  --num-agents 2 \
  --episode-length 6
```

Run a limited API smoke only after local tests pass:

```bash
python simulate_verified.py \
  --provider openai \
  --model gpt-5.2-2025-12-11 \
  --run-id api-g4 \
  --num-agents 2 \
  --episode-length 6 \
  --max-calls 24 \
  --max-cost-usd 0.25
```

The runner blocks more than four agents or twelve months unless
`--allow-larger-run` is explicit. Passing a smoke establishes implementation
validity, not superiority. Older sealed run/replay artifacts and the historical
smoke report predate the current P0 completion changes and are not evidence for
the current method.

The current replay implementation separates prompt-routed context into a
protected base-prompt field and hash-binds the reconstructed matched prompt.
That implementation contract has passed local validation. Hosted replay still
supports controlled prompt-level action sensitivity only until a compatible
checkpoint continuation establishes downstream utility and next-state effects.

## Historical FinEvo/EMNLP toolkit

The commands and features below describe the retained legacy experiment path,
not results or validation for Evidence-Grounded Rule Memory. Use them only for
explicitly labeled historical reproduction.

- **Multi-model support**: OpenAI, OpenRouter-compatible APIs, Google Gemini, and local OpenAI-compatible servers
- **FinEvo components**: market sentiment, episodic memory, semantic rule memory, and reflection
- **Revision experiments**: E1-E5 matrices for main results, component ablations, sentiment robustness, text-event controls, and prompt/decoding robustness
- **Structured artifacts**: action logs, memory retrieval logs, semantic rule logs, event logs, summaries, trajectories, and seed-level configs

## Installation

```bash
pip install -r requirements.txt
```

Set API keys:
```bash
export OPENAI_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```

## Usage

### OpenAI Models

```bash
# GPT-5.2 FinEvo run
python simulate.py --model=gpt-5.2 --provider=openai --num_agents=100 --episode_length=240 --gap_fixes=True

# GPT-4o text-only baseline
python simulate.py --model=gpt-4o --provider=openai --gap_fixes=False
```

### OpenRouter-Compatible Models

```bash
python simulate.py \
  --provider=thirdparty \
  --api_base="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}" \
  --model=google/gemini-3-flash-preview \
  --workers=3
```

### Local Models (MLX)

```bash
# Start MLX server
mlx_lm.server --model mlx-community/Llama-3.3-70B-Instruct-4bit --port 8002

# Run simulation
python simulate.py --model=mlx-community/Llama-3.3-70B-Instruct-4bit --provider=local --local_url=http://localhost:8002/v1
```

### EMNLP Revision Runs

Print the supported E2-E5 experiment matrix without spending API budget:

```bash
python run_emnlp_experiments.py --exps E2,E3,E4,E5 --seeds 13,21,42,87,2026
```

Execute a focused subset and export it to the required `runs/<exp_id>/<model>/<setting>/<variant>/seed_<seed>/` layout:

```bash
python run_emnlp_experiments.py --run --exps E2 --variants default --seeds 13
```

Existing legacy `data/<run>/summary.json` folders can be normalized directly:

```bash
python export_emnlp_outputs.py "data/<run-folder>" --runs-root runs --exp-id E2 --model GPT-5.2 --setting finevo --variant default --seed 13
```

Generate paper figures locally when result folders are present:

```bash
python paper/generate_emnlp_figures.py
```

Generated `data/`, `runs/`, `figs/`, logs, pickles, and zip artifacts are intentionally ignored by Git.

## Project Structure

```
eccv26_EconAgent/
├── simulate.py          # Main simulation script
├── simulate_verified.py # Bounded verified-memory runner
├── replay_verified.py   # Paired memory-intervention runner
├── llm_providers.py     # Multi-model LLM interface
├── memory_module.py     # Episodic and semantic memory
├── verified_memory/     # M0/M1/M2/M3, replay, budgets, provenance
├── market_sentiment.py  # Market sentiment dynamics
├── run_emnlp_experiments.py # EMNLP E1-E8 run matrix
├── export_emnlp_outputs.py  # Normalizes outputs to required schema
├── utils.py             # Utility functions
├── config.yaml          # Environment configuration
├── requirements.txt     # Dependencies
└── README.md
```

## License

MIT
