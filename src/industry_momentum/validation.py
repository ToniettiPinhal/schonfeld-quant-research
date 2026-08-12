"""Temporal splits, robustness grids, concentration checks, and factor attribution."""

from __future__ import annotations

import pandas as pd

from .backtest import BacktestResult, run_strategy
from .config import ResearchConfig, StrategyConfig
from .metrics import hac_mean_t_stat, newey_west_ols, performance_metrics


def slice_period(series: pd.Series | pd.DataFrame, start: str, end: str):
    return series.loc[start:end]


def period_summary(
    result: BacktestResult,
    factor_momentum: pd.Series,
    config: ResearchConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strategies = {
        "industry_momentum_gross": result.monthly["gross_return"],
        f"industry_momentum_net_{config.strategy.cost_bps:g}bps": result.monthly["net_return"],
        "fama_french_momentum_factor": factor_momentum,
    }
    for period, (start, end) in config.periods.items():
        for name, returns in strategies.items():
            sample = slice_period(returns, start, end).dropna()
            metrics = performance_metrics(
                sample,
                turnover=(
                    slice_period(result.monthly["traded_notional"], start, end)
                    if name.startswith("industry_momentum")
                    else None
                ),
            )
            metrics.update(
                {
                    "period": period,
                    "strategy": name,
                    "hac_mean_t_stat": hac_mean_t_stat(sample, config.hac_lags),
                }
            )
            rows.append(metrics)
    return pd.DataFrame(rows).set_index(["period", "strategy"]).sort_index()


def cost_sensitivity(
    gross_returns: pd.Series,
    traded_notional: pd.Series,
    start: str,
    end: str,
    costs_bps: tuple[float, ...] = (0.0, 10.0, 25.0, 50.0),
    hac_lags: int = 6,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    gross = slice_period(gross_returns, start, end)
    turnover = slice_period(traded_notional, start, end)
    for cost in costs_bps:
        net = gross - cost / 10_000.0 * turnover
        metrics = performance_metrics(net, turnover=turnover)
        metrics.update({"cost_bps": cost, "hac_mean_t_stat": hac_mean_t_stat(net, hac_lags)})
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("cost_bps")


def robustness_grid(
    returns: pd.DataFrame,
    research_config: ResearchConfig,
    lookbacks: tuple[int, ...] = (6, 9, 12, 18),
    fractions: tuple[float, ...] = (0.20, 0.30),
    costs_bps: tuple[float, ...] = (0.0, 10.0, 25.0, 50.0),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lookback in lookbacks:
        for fraction in fractions:
            gross_config = StrategyConfig(
                lookback_months=lookback,
                skip_months=1,
                selection_fraction=fraction,
                cost_bps=0.0,
            )
            result = run_strategy(returns, gross_config)
            test_gross = slice_period(
                result.monthly["gross_return"],
                research_config.test_start,
                research_config.test_end,
            )
            test_turnover = slice_period(
                result.monthly["traded_notional"],
                research_config.test_start,
                research_config.test_end,
            )
            for cost in costs_bps:
                net = test_gross - cost / 10_000.0 * test_turnover
                metrics = performance_metrics(net, test_turnover)
                rows.append(
                    {
                        "lookback_months": lookback,
                        "skip_months": 1,
                        "selection_fraction": fraction,
                        "cost_bps": cost,
                        "annualized_arithmetic_return": metrics[
                            "annualized_arithmetic_return"
                        ],
                        "annualized_volatility": metrics["annualized_volatility"],
                        "annualized_sharpe": metrics["annualized_sharpe"],
                        "max_drawdown": metrics["max_drawdown"],
                        "annualized_traded_notional": metrics[
                            "annualized_traded_notional"
                        ],
                        "hac_mean_t_stat": hac_mean_t_stat(net, research_config.hac_lags),
                    }
                )
    return pd.DataFrame(rows).set_index(
        ["lookback_months", "skip_months", "selection_fraction", "cost_bps"]
    )


def subperiod_summary(result: BacktestResult, hac_lags: int = 6) -> pd.DataFrame:
    subperiods = {
        "2000-2009": ("2000-01", "2009-12"),
        "2010-2019": ("2010-01", "2019-12"),
        "2020-2026H1": ("2020-01", "2026-06"),
    }
    rows: list[dict[str, object]] = []
    for name, (start, end) in subperiods.items():
        sample = slice_period(result.monthly["net_return"], start, end)
        turnover = slice_period(result.monthly["traded_notional"], start, end)
        metrics = performance_metrics(sample, turnover)
        metrics.update({"subperiod": name, "hac_mean_t_stat": hac_mean_t_stat(sample, hac_lags)})
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("subperiod")


def leave_one_industry_out(
    returns: pd.DataFrame,
    config: ResearchConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for omitted in returns.columns:
        result = run_strategy(returns.drop(columns=omitted), config.strategy)
        sample = slice_period(result.monthly["net_return"], config.test_start, config.test_end)
        turnover = slice_period(
            result.monthly["traded_notional"], config.test_start, config.test_end
        )
        metrics = performance_metrics(sample, turnover)
        rows.append(
            {
                "omitted_industry": omitted,
                "annualized_arithmetic_return": metrics["annualized_arithmetic_return"],
                "annualized_sharpe": metrics["annualized_sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "hac_mean_t_stat": hac_mean_t_stat(sample, config.hac_lags),
            }
        )
    return pd.DataFrame(rows).set_index("omitted_industry").sort_index()


def factor_attribution(
    strategy_returns: pd.Series,
    factors: pd.DataFrame,
    start: str,
    end: str,
    hac_lags: int = 6,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    sample = pd.concat(
        [strategy_returns.rename("strategy"), factors[["Mkt-RF", "SMB", "HML", "Mom"]]],
        axis=1,
    ).dropna()
    sample = slice_period(sample, start, end)
    regression = newey_west_ols(
        sample["strategy"], sample[["Mkt-RF", "SMB", "HML", "Mom"]], lags=hac_lags
    )
    table = regression.to_frame()
    table.loc["intercept", "annualized_coefficient"] = 12.0 * table.loc[
        "intercept", "coefficient"
    ]
    table.loc[table.index != "intercept", "annualized_coefficient"] = table.loc[
        table.index != "intercept", "coefficient"
    ]
    metadata: dict[str, float | int] = {
        "r_squared": regression.r_squared,
        "n_observations": regression.n_observations,
        "strategy_mom_correlation": float(sample["strategy"].corr(sample["Mom"])),
    }
    return table, metadata
