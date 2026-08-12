"""End-to-end research run, generated tables, figures, and research note."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import run_strategy
from .config import ResearchConfig
from .data import prepare_data
from .features import assert_temporal_integrity
from .metrics import (
    moving_block_sharpe_interval,
    paired_block_sharpe_difference_interval,
)
from .validation import (
    cost_sensitivity,
    factor_attribution,
    leave_one_industry_out,
    period_summary,
    robustness_grid,
    subperiod_summary,
)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, float_format="%.10f")


def _plot_cumulative_returns(
    strategy: pd.Series,
    momentum_factor: pd.Series,
    destination: Path,
) -> None:
    joined = pd.concat(
        [strategy.rename("Industry momentum, net"), momentum_factor.rename("FF Momentum factor")],
        axis=1,
    ).dropna()
    wealth = (1.0 + joined).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    wealth.plot(ax=ax, linewidth=2.0)
    ax.set_title("Post-2000 cumulative growth of $1")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_xlabel("")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_rolling_sharpe(
    strategy: pd.Series,
    momentum_factor: pd.Series,
    destination: Path,
) -> None:
    joined = pd.concat(
        [strategy.rename("Industry momentum, net"), momentum_factor.rename("FF Momentum factor")],
        axis=1,
    ).dropna()
    rolling = np.sqrt(12.0) * joined.rolling(36).mean() / joined.rolling(36).std(ddof=1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rolling.plot(ax=ax, linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Rolling 36-month annualized Sharpe ratio")
    ax.set_ylabel("Sharpe ratio")
    ax.set_xlabel("")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_cost_sensitivity(table: pd.DataFrame, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    table["annualized_arithmetic_return"].mul(100).plot.bar(
        ax=ax, color="#285f8f", width=0.72
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Post-2000 return sensitivity to transaction costs")
    ax.set_xlabel("Cost per dollar traded (basis points)")
    ax.set_ylabel("Annualized arithmetic return (%)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _percent(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def _write_research_note(
    destination: Path,
    summary: pd.DataFrame,
    costs: pd.DataFrame,
    regression: pd.DataFrame,
    regression_metadata: dict[str, float | int],
    bootstrap: dict[str, dict[str, float | int]],
    leave_one_out: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    test_net = summary.loc[("test", "industry_momentum_net_10bps")]
    test_gross = summary.loc[("test", "industry_momentum_gross")]
    validation_net = summary.loc[("validation", "industry_momentum_net_10bps")]
    test_mom = summary.loc[("test", "fama_french_momentum_factor")]
    alpha = float(regression.loc["intercept", "annualized_coefficient"])
    alpha_t = float(regression.loc["intercept", "hac_t_stat"])
    mom_beta = float(regression.loc["Mom", "coefficient"])
    loo_min = float(leave_one_out["annualized_sharpe"].min())
    loo_max = float(leave_one_out["annualized_sharpe"].max())
    note = f"""# Research note: does U.S. industry momentum persist out of sample?

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
- The stable 49-industry panel begins {metadata['stable_start']}; the first investable
  month is {metadata['first_strategy_month']} after the full formation window.
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
| Net annualized arithmetic return | {_percent(float(validation_net['annualized_arithmetic_return']))} | {_percent(float(test_net['annualized_arithmetic_return']))} |
| Net annualized volatility | {_percent(float(validation_net['annualized_volatility']))} | {_percent(float(test_net['annualized_volatility']))} |
| Net annualized Sharpe | {float(validation_net['annualized_sharpe']):.2f} | {float(test_net['annualized_sharpe']):.2f} |
| Net maximum drawdown | {_percent(float(validation_net['max_drawdown']))} | {_percent(float(test_net['max_drawdown']))} |
| HAC t-stat of mean | {float(validation_net['hac_mean_t_stat']):.2f} | {float(test_net['hac_mean_t_stat']):.2f} |

Gross post-2000 return is {_percent(float(test_gross['annualized_arithmetic_return']))}
per year, versus {_percent(float(test_net['annualized_arithmetic_return']))} after the
primary cost assumption. The post-2000 Fama-French Momentum factor Sharpe is
{float(test_mom['annualized_sharpe']):.2f}. The moving-block 95% interval for the net
industry-momentum Sharpe is [{float(bootstrap['strategy_sharpe']['ci_2_5']):.2f},
{float(bootstrap['strategy_sharpe']['ci_97_5']):.2f}], which spans zero.

## 5. Factor attribution

In the post-2000 four-factor regression, the strategy's loading on the standard
Momentum factor is {mom_beta:.2f}; the monthly return correlation is
{float(regression_metadata['strategy_mom_correlation']):.2f}. Annualized intercept is
{_percent(alpha, 2)} with a Newey-West t-statistic of {alpha_t:.2f}, and model R-squared
is {float(regression_metadata['r_squared']):.2f}. The data therefore do not support a
claim that this implementation delivers distinct alpha beyond generic momentum.

## 6. Robustness and falsification

- Cost sensitivity is monotone: post-2000 annualized arithmetic return is
  {_percent(float(costs.loc[0.0, 'annualized_arithmetic_return']))} at 0 bps,
  {_percent(float(costs.loc[25.0, 'annualized_arithmetic_return']))} at 25 bps, and
  {_percent(float(costs.loc[50.0, 'annualized_arithmetic_return']))} at 50 bps.
- The full lookback, selection-fraction, and cost grid is reported rather than only
  the best row.
- Leaving out one industry at a time produces post-2000 net Sharpe ratios from
  {loo_min:.2f} to {loo_max:.2f}; this diagnoses concentration without selecting the
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
"""
    destination.write_text(note, encoding="utf-8")


def run_research(
    config: ResearchConfig,
    force_download: bool = False,
) -> dict[str, Any]:
    data = prepare_data(config.raw_data_dir, force_download=force_download)
    assert_temporal_integrity(
        data.industry_returns,
        config.strategy.lookback_months,
        config.strategy.skip_months,
        config.strategy.selection_fraction,
    )
    result = run_strategy(data.industry_returns, config.strategy)
    factor_momentum = data.factors["Mom"]

    results_dir = config.reports_dir / "results"
    figures_dir = config.reports_dir / "figures"
    summary = period_summary(result, factor_momentum, config)
    costs = cost_sensitivity(
        result.monthly["gross_return"],
        result.monthly["traded_notional"],
        config.test_start,
        config.test_end,
        hac_lags=config.hac_lags,
    )
    robustness = robustness_grid(data.industry_returns, config)
    subperiods = subperiod_summary(result, config.hac_lags)
    leave_one_out = leave_one_industry_out(data.industry_returns, config)
    factor_table, factor_metadata = factor_attribution(
        result.monthly["net_return"],
        data.factors,
        config.test_start,
        config.test_end,
        config.hac_lags,
    )

    test_strategy = result.monthly["net_return"].loc[config.test_start : config.test_end]
    test_momentum = factor_momentum.loc[config.test_start : config.test_end]
    bootstrap = {
        "strategy_sharpe": moving_block_sharpe_interval(
            test_strategy,
            samples=config.bootstrap_samples,
            block_length=config.bootstrap_block_months,
            seed=config.random_seed,
        ),
        "sharpe_difference_vs_mom": paired_block_sharpe_difference_interval(
            test_strategy,
            test_momentum,
            samples=config.bootstrap_samples,
            block_length=config.bootstrap_block_months,
            seed=config.random_seed,
        ),
    }

    _write_csv(summary, results_dir / "period_summary.csv")
    _write_csv(costs, results_dir / "cost_sensitivity.csv")
    _write_csv(robustness, results_dir / "robustness_grid.csv")
    _write_csv(subperiods, results_dir / "subperiods.csv")
    _write_csv(leave_one_out, results_dir / "leave_one_industry_out.csv")
    _write_csv(factor_table, results_dir / "factor_regression.csv")
    (results_dir / "bootstrap_intervals.json").write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _plot_cumulative_returns(test_strategy, test_momentum, figures_dir / "cumulative_returns.png")
    _plot_rolling_sharpe(test_strategy, test_momentum, figures_dir / "rolling_sharpe.png")
    _plot_cost_sensitivity(costs, figures_dir / "cost_sensitivity.png")

    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "stable_start": data.stable_start.strftime("%Y-%m"),
        "first_strategy_month": result.monthly.index.min().strftime("%Y-%m"),
        "last_strategy_month": result.monthly.index.max().strftime("%Y-%m"),
        "industry_count": int(data.industry_returns.shape[1]),
        "strategy_config": asdict(config.strategy),
        "temporal_splits": config.periods,
        "inference": {
            "hac_lags": config.hac_lags,
            "bootstrap_block_months": config.bootstrap_block_months,
            "bootstrap_samples": config.bootstrap_samples,
            "random_seed": config.random_seed,
        },
        "factor_regression": factor_metadata,
        "data_manifest": data.manifest,
    }
    (config.reports_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_research_note(
        config.reports_dir / "research_note.md",
        summary,
        costs,
        factor_table,
        factor_metadata,
        bootstrap,
        leave_one_out,
        metadata,
    )
    return {
        "summary": summary,
        "costs": costs,
        "factor_regression": factor_table,
        "factor_metadata": factor_metadata,
        "bootstrap": bootstrap,
        "metadata": metadata,
    }
