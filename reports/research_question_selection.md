# Research question selection

The project was selected before implementation. Scores are relative (1 = weak,
5 = strong); leakage/bias risk is scored in the favorable direction, so 5 means the
risk is easiest to control.

| Candidate question | Schonfeld relevance | Statistical rigor | Data reliability | Implementation tractability | Interview value | Leakage/bias control | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Does 12-2 cross-sectional momentum across 49 U.S. industries persist after 2000, costs, and standard Momentum attribution? | 5 | 5 | 5 | 4 | 5 | 5 | **29** |
| Does inverse-volatility scaling improve the post-2000 risk-adjusted return of the U.S. market factor after leverage caps and costs? | 4 | 5 | 5 | 5 | 4 | 5 | **28** |
| Does the Treasury-curve regime predict the next 12-month equity premium in a genuinely real-time design? | 4 | 4 | 4 | 3 | 5 | 2 | **22** |

## Selection

The first question was chosen. It has a clean economic prior, a natural publication
boundary after Moskowitz and Grinblatt (1999), a long official monthly panel, and a
useful negative-result possibility: a positive raw strategy can still fail once costs,
uncertainty, crash risk, and exposure to the standard Momentum factor are measured.

The inverse-volatility question was a close second but offers less cross-sectional
research depth. The yield-curve question was deferred because revised macro series,
release timestamps, overlapping 12-month outcomes, and multiple plausible regime
definitions create materially greater vintage and specification risk.

No candidate was ranked using realized strategy returns.
