# Research note: does U.S. industry momentum persist out of sample?

## 1. Question and prior

Does a fixed 12-2 cross-sectional momentum rule across 49 value-weighted U.S.
industry portfolios retain positive performance after January 2000, after charging
turnover-based transaction costs, and does it add information beyond the standard
Fama-French Momentum factor?

Moskowitz and Grinblatt (1999) documented strong industry momentum. The prior is
therefore directional, but the post-2000 test is deliberately treated as the locked
sample rather than as another parameter-search window.

## 2. Data and timestamp assumptions

- Source: Kenneth R. French Data Library, 49 Industry Portfolios, Fama-French three
  factors, and Momentum factor.
- Data vintage: CRSP 202606; latest monthly observation: 2026-06.
- Source ZIP SHA-256 values are recorded in `reports/run_metadata.json`.
- Monthly portfolio returns are treated as known only after the relevant month ends.
- The stable 49-industry panel begins 1969-07; the first investable
  month is 1970-07 after the full formation window.
- The Data Library warns that historical series can be revised. Re-running against a
  later vintage may therefore change the output; exact reproduction requires the
  recorded checksums.

## 3. Method

For each month t, the signal compounds returns from t-12 through t-2. The most recent
month t-1 and the current return t are excluded. Ten winner industries receive +10%
each and ten loser industries receive -10% each, producing 100% long, 100% short, and
zero net exposure. Month-t weights are applied to month-t returns.

Traded notional is `sum(abs(w_t - w_(t-1)))`. The primary net series charges 10 basis
points per dollar traded, including initial entry. No borrow fee, financing spread,
market impact, tax, or ETF-tracking error is modeled.

The fixed temporal split is:

| Segment | Dates | Purpose |
|---|---:|---|
| Development | 1970-07 to 1984-12 | implementation and exploratory diagnostics |
| Validation | 1985-01 to 1999-12 | pre-publication confirmation; no final-test tuning |
| Locked test | 2000-01 to 2026-06 | final out-of-sample evaluation |

## 4. Main results

| Result | Validation | Locked test |
|---|---:|---:|
| Net annualized arithmetic return | 10.5% | 3.0% |
| Net annualized volatility | 12.4% | 17.7% |
| Net annualized Sharpe | 0.85 | 0.17 |
| Net maximum drawdown | -15.7% | -55.5% |
| HAC t-stat of mean | 3.23 | 0.90 |

Gross post-2000 return is 4.1%
per year, versus 3.0% after the
primary cost assumption. The post-2000 Fama-French Momentum factor Sharpe is
0.17. The moving-block 95% interval for the net
industry-momentum Sharpe is [-0.19,
0.57], which spans zero.

## 5. Factor attribution

In the post-2000 four-factor regression, the strategy's loading on the standard
Momentum factor is 0.88; the monthly return correlation is
0.84. Annualized intercept is
-0.25% with a Newey-West t-statistic of -0.13, and model R-squared
is 0.72. The data therefore do not support a
claim that this implementation delivers distinct alpha beyond generic momentum.

## 6. Robustness and falsification

- Cost sensitivity is monotone: post-2000 annualized arithmetic return is
  4.1% at 0 bps,
  1.4% at 25 bps, and
  -1.3% at 50 bps.
- The full lookback, selection-fraction, and cost grid is reported rather than only
  the best row.
- Leaving out one industry at a time produces post-2000 net Sharpe ratios from
  0.09 to 0.21; this diagnoses concentration without selecting the
  most favorable omission.
- A deterministic perturbation test confirms that changing month-t returns cannot
  change signals or weights at t or earlier.
- The 2000-2009, 2010-2019, and 2020-2026H1 subperiods are all shown separately.

## 7. Interpretation

The hypothesis receives weak, not decisive, out-of-sample support. The gross rule is
positive after 2000, but turnover consumes a meaningful fraction of the return, the
confidence interval includes zero, the drawdown is severe, and factor attribution
shows that most variation is standard Momentum exposure. A strong pre-2000 backtest
would have overstated the evidence available to a post-2000 investor.

## 8. Limitations

The 49 portfolios are academic constructs, not directly tradable instruments.
Transaction costs are stylized. Short availability, borrow fees, financing, impact,
fund capacity, taxes, constituent-level implementation, and publication-date data
vintages are not modeled. The post-2000 boundary is economically motivated by prior
publication but is not a randomized experiment. A single U.S. industry taxonomy does
not establish international generality.

## 9. Conclusion

This study does **not** claim a new alpha. It finds a weaker post-publication industry
momentum premium that is cost-sensitive and largely subsumed by the standard Momentum
factor. The useful result is methodological: careful lags, a locked temporal test,
explicit costs, full robustness tables, and honest attribution materially change the
story told by the in-sample equity curve.

## 10. Next tests

1. Repeat on international industry portfolios with a predeclared specification.
2. Replace academic portfolios with investable sector instruments and realistic
   histories, borrow assumptions, and spreads.
3. Test whether exposure control reduces momentum-crash risk without tuning on the
   final sample.

## References

- Kenneth R. French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- 49 Industry Portfolios construction: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_49_ind_port.html
- Momentum factor construction: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/det_mom_factor.html
- Moskowitz, T. J., and Grinblatt, M. (1999), *Do Industries Explain Momentum?*, DOI: https://doi.org/10.1111/0022-1082.00146
- Newey, W. K., and West, K. D. (1987), DOI: https://doi.org/10.2307/1913610
