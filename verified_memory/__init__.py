"""Verified dual-track memory package.

Public modules are intentionally explicit:

``m0_utility`` evaluation ledger; ``m1_context`` causal router;
``m2_episodic`` finalized evidence; ``m3_semantic`` verified rules;
``system`` dual-track facade; ``runner`` bounded integration path; and
``replay``/``replay_experiment`` paired interventions.

The package initializer stays import-light because ``llm_providers`` itself uses
``verified_memory.budget``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
