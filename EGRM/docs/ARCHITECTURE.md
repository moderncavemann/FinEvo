# EGRM architecture

```text
state_t + causal event_t
        |
        v
M1 causal context router ---------> protected prompt context
        |
        +--------------------------> retrieval query
                                      |
finalized M2 episodes <--------------+
        |
        +--> LLM candidate proposal (untrusted)
                 |
                 v
        M3 evidence attribution
          |       |        |
       reject  provisional  counterevidence search
                    |
             post-proposal support
                    |
                  active
                    |
        harmful/alternative evidence
                    |
            retire or supersede
```

## Extracted modules

- `m0_utility.py`: realized flow utility and budget-balanced evaluation ledger.
- `m1_context.py`: causal, observed-through-time context packet and route.
- `m2_episodic.py`: pending decision then finalized `state_t, action_t,
  outcome_t+1` record; pending records cannot be retrieved.
- `m3_semantic.py`: strict candidate schema, five-way evidence taxonomy,
  provisional activation, counterevidence, retirement, and lineage.
- `system.py`: dual-track composition and serialized restore.
- `replay.py`: hash-bound prompt-level matched interventions.
- `artifacts.py`, `failure_artifacts.py`, `budget.py`: sealed outputs, terminal
  failure receipts, and call accounting.

## Important boundary

`replay.py` alone changes a memory block inside a protected prompt and measures
action sensitivity. It does not restore and continue the economy. Claims about
utility or next state require the checkpoint continuation protocol in the
scientific experiment plan.

The M1 projection is frozen/inference-only in this extraction. No end-to-end
learning claim is made for it.
