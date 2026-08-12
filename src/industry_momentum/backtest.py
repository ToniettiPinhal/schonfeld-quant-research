"""Portfolio accounting with explicit alignment and traded-notional costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .features import cross_sectional_weights, momentum_signal


@dataclass(frozen=True)
class BacktestResult:
    signals: pd.DataFrame
    weights: pd.DataFrame
    monthly: pd.DataFrame


def traded_notional(weights: pd.DataFrame) -> pd.Series:
    """Total absolute weight change, including entry from cash at inception."""

    previous = weights.shift(1).fillna(0.0)
    return (weights - previous).abs().sum(axis=1).rename("traded_notional")


def run_backtest(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float,
) -> BacktestResult:
    """Apply month-t weights to month-t returns and subtract trading costs.

    A cost of ``cost_bps`` is charged on every dollar of absolute weight change:
    ``cost = cost_bps / 10_000 * sum(abs(w_t - w_{t-1}))``. Borrow fees,
    financing spreads, market impact, and implementation shortfall are not modeled.
    """

    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    if not returns.index.equals(weights.index) or not returns.columns.equals(weights.columns):
        raise ValueError("Returns and weights must have identical labels")
    if not signals.index.equals(weights.index) or not signals.columns.equals(weights.columns):
        raise ValueError("Signals and weights must have identical labels")

    active = weights.abs().sum(axis=1) > 0
    active_weights = weights.loc[active]
    active_returns = returns.loc[active]
    if (active_returns.isna() & active_weights.ne(0)).any().any():
        raise ValueError("An active position has a missing contemporaneous return")

    long_exposure = active_weights.clip(lower=0).sum(axis=1)
    short_exposure = active_weights.clip(upper=0).sum(axis=1)
    if not np.allclose(long_exposure, 1.0) or not np.allclose(short_exposure, -1.0):
        raise ValueError("Active portfolio must be 100% long and 100% short")

    gross = (active_weights * active_returns).sum(axis=1).rename("gross_return")
    long_leg = (active_weights.clip(lower=0) * active_returns).sum(axis=1).rename("long_leg")
    short_leg = (active_weights.clip(upper=0) * active_returns).sum(axis=1).rename("short_leg")
    turnover = traded_notional(active_weights)
    cost = (cost_bps / 10_000.0 * turnover).rename("transaction_cost")
    net = (gross - cost).rename("net_return")
    monthly = pd.concat([gross, net, long_leg, short_leg, turnover, cost], axis=1)
    return BacktestResult(signals=signals.loc[active], weights=active_weights, monthly=monthly)


def run_strategy(returns: pd.DataFrame, config: StrategyConfig) -> BacktestResult:
    signals = momentum_signal(
        returns,
        lookback_months=config.lookback_months,
        skip_months=config.skip_months,
    )
    weights = cross_sectional_weights(signals, config.selection_fraction)
    return run_backtest(returns, signals, weights, config.cost_bps)


def equal_weight_universe_return(returns: pd.DataFrame) -> pd.Series:
    """Equal weight across all 49 value-weighted industry portfolios."""

    if returns.isna().any().any():
        raise ValueError("Equal-weight benchmark requires a complete return panel")
    return returns.mean(axis=1).rename("equal_weight_industries")
