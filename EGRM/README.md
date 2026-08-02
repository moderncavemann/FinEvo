# Evidence-Grounded Rule Memory (EGRM)

EGRM is a new paper and code boundary for one question: **when should a
long-horizon economic agent trust a rule distilled from its own experience?**
It separates LLM rule generation from evidence admission, requires distinct
post-proposal support before activation, tracks five kinds of counterevidence,
and makes revision and retirement auditable.

The intended paper is:

> *When Should an Economic Agent Trust Its Memory? Evidence-Grounded Rule
> Evolution under Endogenous Feedback*

This directory is not a relabeling of historical FinEvo results. It contains an
independently installable, source-pinned extraction of the rule-memory
implementation, a zero-provider lifecycle fixture, a non-dispatchable
scientific design draft, matched-intervention utilities, tests, and a
double-blind manuscript skeleton. No
scientific EGRM effectiveness result has been produced by this release.

## What is new

The paper studies the lifecycle

```text
finalized experience
  -> evidence attribution
  -> provisional rule
  -> post-proposal validation
  -> activation
  -> counterevidence
  -> revision or retirement
  -> downstream endogenous feedback
```

The three primary research questions are:

1. Does evidence admission reject unsupported rule candidates grounded in
   finalized action--outcome records?
2. When the same false rule is forced active, does counterevidence shorten
   harmful exposure and retirement latency?
3. Do lifecycle differences cause checkpoint-matched action and six-step
   utility effects larger than the matched A/A null?

Aggregate wealth, Gini, and labor outcomes are exploratory diagnostics, not the
primary mechanism claim.

## Quick start

The extracted package has no runtime dependency outside the Python standard
library. Tests use `pytest`.

```bash
cd EGRM
python -m pip install -e ".[test]"
pytest
egrm-controlled \
  --contract configs/controlled_benchmark_v1.json \
  --output /tmp/egrm-controlled-result.json
```

Without installing:

```bash
PYTHONPATH=EGRM/src python EGRM/scripts/run_controlled_benchmark.py \
  --contract EGRM/configs/controlled_benchmark_v1.json

PYTHONPATH=EGRM/src python EGRM/scripts/expand_scientific_design.py \
  --contract EGRM/configs/scientific_contract_v1.json \
  --output /tmp/egrm-design-inventory.json

PYTHONPATH=EGRM/src python EGRM/scripts/validate_provenance.py \
  --manifest EGRM/provenance/SOURCE_PROVENANCE.json \
  --source-repo-root .
```

The controlled output must always say
`scope=deterministic_implementation_fixture` and
`scientific_evidence=false`. It validates lifecycle wiring; it does not estimate
an effect.

## Directory map

```text
EGRM/
├── src/egrm/        extracted M0-M3, replay, artifacts, contracts, metrics
├── configs/         implementation fixture and non-dispatchable design draft
├── tests/           inherited invariant tests plus EGRM-specific claim guards
├── docs/            method, experiment, overlap, and claim-evidence contracts
├── paper/           anonymous ICAIF-format manuscript skeleton
└── provenance/      source commit, blobs, hashes, and explicit exclusions
```

Start with [STATUS.md](STATUS.md),
[docs/METHOD_CONTRACT.md](docs/METHOD_CONTRACT.md), and
[docs/CLAIM_EVIDENCE_MATRIX.md](docs/CLAIM_EVIDENCE_MATRIX.md). Exact
estimands, denominators, censoring, and failure handling are frozen as design
requirements in [docs/METRIC_CARDS.md](docs/METRIC_CARDS.md).

## Evidence and reuse boundary

- The reusable implementation is source-pinned to FinEvo commit
  `ffc063acf19c778f109ae6a8552bab6d192175fc`.
- Historical paper tables, figures, reviewer responses, raw pilots, provider
  calls, and macroeconomic effects are not copied and are not EGRM evidence.
- The controlled fixture makes no provider call.
- Hosted scientific runs remain disabled until the separate EGRM cost gate and
  submission-eligibility gate pass.
- Prompt-only replay supports action sensitivity. Downstream causal language
  requires exact checkpoint continuation and a matched A/A null.

See [provenance/SOURCE_PROVENANCE.json](provenance/SOURCE_PROVENANCE.json) for
the machine-readable extraction record.

## Submission warning

ICAIF 2026 prohibits a full paper that is simultaneously under review at
another archival venue. A distinct question and fresh experiments do not waive
that rule. If FinEvo is still under archival review, EGRM must not be submitted
to ICAIF unless eligibility is resolved with the program chairs. See
[docs/OVERLAP_AND_SUBMISSION_POLICY.md](docs/OVERLAP_AND_SUBMISSION_POLICY.md).

## License status

The source repository describes itself as MIT in its README but does not contain
an actual license file at the pinned commit. This extraction does not invent a
new legal grant. Before a standalone EGRM release, the authors must add the
intended license. See [NOTICE.md](NOTICE.md).
