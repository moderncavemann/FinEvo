# Related-work scope

Primary sources checked on 2026-08-02 are grouped below by the claim they
constrain.

## Economic-agent context

- [EconAgent](https://arxiv.org/abs/2310.10436) introduces LLM agents for
  macroeconomic simulation with heterogeneous decision mechanisms and memory
  over individual experience and market dynamics.
- [EconAI](https://arxiv.org/abs/2605.13762) emphasizes sentiment indexing,
  memory weighting, and dynamic persona/decision adaptation in evolving
  economic environments.

These works constrain claims about economic-agent simulation, adaptation, and
memory. EGRM cannot claim that it is the first economic simulation with an LLM
agent or persistent memory.

## Experience-based reflection and memory regulation: closest threats

- [Reflexion](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html)
  uses verbal feedback and stores reflective text in episodic memory to guide
  later trials.
- [ExpeL](https://ojs.aaai.org/index.php/AAAI/article/view/29936) autonomously
  gathers experiences, extracts natural-language insights, and recalls both
  insights and past experiences for subsequent tasks.
- [MetaReflection](https://aclanthology.org/2024.emnlp-main.477/) learns reusable
  instructions from past reflections for language agents, including an offline
  learning formulation.
- [How Memory Management Impacts LLM Agents](https://aclanthology.org/2026.acl-long.27/)
  directly studies experience-following behavior, including error propagation
  and misaligned replay, and uses subsequent task evaluations as signals for
  experience quality.

These are direct novelty threats. EGRM must not claim to be the first system to
use experience-based reflection, natural-language rule or insight memory, or
feedback-based memory regulation. The defensible candidate distinction is more
specific: a reliability formulation for self-generated action--outcome rules
under endogenous feedback, paired with a known-truth benchmark and a
predeclared factorial plus checkpoint evaluation.

## General memory organization, evaluation, and safety

- [A-MEM](https://arxiv.org/abs/2502.12110) organizes memories into a dynamically
  linked agentic network.
- [MemoryAgentBench](https://arxiv.org/abs/2507.05257) evaluates retrieval,
  test-time learning, long-range understanding, and selective forgetting in
  incremental interactions.
- [A-MemGuard](https://arxiv.org/abs/2510.02373) studies adversarial memory
  injection and a consensus/lessons defense.
- [MemEvoBench](https://arxiv.org/abs/2604.15774) benchmarks long-horizon memory
  misevolution under adversarial injection, noisy tools, and biased feedback.

The intended distinction from these safety benchmarks is also narrow. They
largely study contaminated or unreliable memory inputs, whereas the proposed
evaluation centers a rule whose use changes the endogenous environment that
produces its later evidence. This remains a literature-audit inference, not a
universal first claim.

## Provenance and claim boundary

The core lifecycle implementation---finalized episodes, the evidence taxonomy,
delayed activation, and rule lineage/retirement---is inherited from a
source-pinned FinEvo implementation. It is not a new EGRM implementation
contribution. The exact source pin belongs in the private provenance manifest;
it must not be exposed in a double-blind artifact if doing so would identify the
authors.

The candidate paper contributions, pending execution and results, are:

1. the rule-reliability formulation under endogenous feedback;
2. the oracle-labeled, known-truth benchmark; and
3. the predeclared verifier factorial, forced-active manipulation, and
   hash-matched checkpoint evaluation.

A fixed false-rule injection tests the policy response to a controlled known
error. It does **not** establish that the method detects or mitigates naturally
occurring hallucinations in LLM-generated proposals. Such a claim requires a
separately defined natural-proposal audit and corresponding evidence.
