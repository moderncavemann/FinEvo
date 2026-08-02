# FinEvo overlap and ICAIF submission policy

## Distinctness statement

FinEvo studies whether sentiment-aware episodic-semantic memory supports
long-horizon adaptation, with aggregate wealth, unemployment, inequality, and
output as endpoints. EGRM shares a stylized simulator and some provider/action
infrastructure but asks when a self-generated action-outcome rule should be
admitted, activated, contradicted, versioned, or retired.

### Inherited implementation

The core lifecycle behavior---finalized action--outcome provenance, a
verifier-fixed outcome criterion, post-proposal activation evidence, the
five-way evidence taxonomy, delayed activation, version lineage, and
retirement---is inherited from the source-pinned FinEvo implementation. The
simulator, provider, action, and checkpoint infrastructure are also shared.
These components are evaluation subjects and infrastructure; they are not
claimed as newly implemented contributions of EGRM. The exact FinEvo source pin
is retained in the private provenance manifest and must be represented in any
permitted artifact without exposing author identity.

### Candidate contribution and new evidence package

Before experiments, the defensible candidate contribution is limited to the
rule-reliability formulation under endogenous feedback, the oracle-labeled
known-truth benchmark, and the predeclared evaluation design: verifier on/off
$\times$ error on/off contrasts, candidate-admission and forced-active
false-rule manipulations, and matched A/A checkpoint continuations. The planned
primary metrics are unsupported activation, harmful exposure, retirement
latency, calibration, regret, and paired downstream utility loss. These are a
protocol, not findings, until every registered cell and failure is accounted
for.

A fixed erroneous-rule injection evaluates response to a controlled known
error. It cannot by itself support a claim that EGRM prevents, detects, or
mitigates hallucinations among naturally generated LLM proposals. That claim
requires a separate natural-proposal audit; otherwise the manuscript must use
the narrower controlled-error wording.

No FinEvo result table, legacy deterministic-rule trace, macroeconomic effect,
or reviewer artifact is reused as EGRM evidence. Shared code and testbed
components are disclosed and source-pinned.

The overlap is nevertheless substantive because the lifecycle method code,
testbed, and reviewer-motivated experiments are shared. Fresh metrics or fresh
result tables do not automatically make a second archival paper distinct. A
confidential side-by-side matrix of research question, method, text, figures,
experimental cells, and results must be prepared for an eligibility inquiry
and retained with the submission record.

## Venue gate

The official ICAIF 2026 call for papers was checked on 2026-08-02. It states:

- deadline: 2026-08-09 Anywhere on Earth;
- maximum: eight pages total in ACM `sigconf`, including figures and references;
- no supplementary materials or appendices;
- double-blind review;
- a full paper cannot be under review at, accepted by, or already published in
  another archival venue at submission time; authors also may not submit it to
  another archival venue during ICAIF review.

Source: [official ICAIF 2026 Call for Papers](https://icaif2026.org/call-for-papers.html).

Therefore, scientific distinctness is necessary but not sufficient. Before
submitting EGRM, record one of these auditable states:

1. FinEvo is no longer under archival review; or
2. the ICAIF program chairs have confirmed eligibility in writing.

If neither is true, do not submit EGRM to ICAIF. This repository does not make a
legal or venue-eligibility determination.

## Double-blind handling

The manuscript must cite shared prior work in the third person and must not link
to an identity-revealing repository during review unless the venue explicitly
permits an anonymized artifact. The author list must be final at initial CMT
submission. The internal FinEvo commit pin remains in confidential provenance;
the anonymous manuscript must not expose a repository URL, commit, account, or
other linkage that reveals author identity.
