# Rule-audit validation record

Validated on 2026-07-10 against the local source artifacts.

## Automated checks

- `python -m unittest artifacts.rule_audit.test_audit_semantic_rules -v`:
  4 tests passed.
- `python -m py_compile artifacts/rule_audit/audit_semantic_rules.py
  artifacts/rule_audit/test_audit_semantic_rules.py`: passed.
- Full default audit rerun: 7 runs, 12,400 semantic snapshots, 9,400
  accepted create/merge events, 27,000 selected-rule decisions, and 0
  code-level contradictions.
- Active-run reconciliation: inferred update events equal the final
  `strategy_evolution` event counts (4,100 GPT-4o; 5,300 Gemini).
- All five GPT-5.2 E2 full seeds reconcile to zero snapshot rows, zero final
  semantic rules, zero evolution events, and zero threshold-trigger episodes.

## Rendered HTML checks

- Desktop viewport: report reading order, exact tables, and chart labels are
  readable; no chart or table is used to imply a causal or cross-seed claim.
- Narrow viewport (500 CSS px): headings and narrative wrap; metric cards stack;
  chart remains horizontally scrollable; source tooltips use the mobile tray.
- Runtime DOM: `data-recharts-ready="true"` and a live SVG are present.
- Source tooltip interaction: trigger opens, is keyboard/click reachable, and
  shows the real source artifact names.
- Static same-data SVG remains in the HTML as the no-script fallback.

The validation screenshots were temporary QA files outside the repository and
are not part of the delivered artifact.
