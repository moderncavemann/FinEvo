# Claim-to-evidence matrix

| Claim | Required metric | Required artifact | Current status |
|---|---|---|---|
| Extracted lifecycle is wired as declared | expected status sequence, integrity validation | controlled fixture JSON plus test log | Implementation fixture available; must pass at release commit |
| EGRM rejects unsupported candidates | admission precision, unsupported ever-active | E0 oracle labels and full candidate ledger | Not run |
| Delayed activation prevents premature rule use | false activation and exposure before independent support | E0/E1 event ledger | Not run |
| Counterevidence reduces harmful exposure | selected harmful exposure steps/rate | method-matched E1 forced-active pairs | Not run |
| Counterevidence reduces damage | paired cumulative utility loss | E1 no-error/error pairs | Not run |
| Retirement causes downstream change | action, immediate utility, next state, six-step utility beyond A/A null | E2 hash-bound checkpoint continuations | Not run |
| Direction appears in a second model | competence gate, failures, 3/3 raw paired direction | E3 model-specific receipt | Not run |
| EGRM improves macroeconomic welfare | population wealth/Gini/labor diagnostics | fresh, adequately powered closed-loop study | Outside primary paper claim |
| EGRM guarantees correct memory | impossible under this design | none | Forbidden |

## Result admission rule

A row changes from `Not run` only when its complete registered denominator,
failure ledger, raw paired deltas, aggregate table, hashes, and terminal receipt
are present. Passing unit tests or the deterministic fixture cannot change a
scientific row.
