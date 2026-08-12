"""Leakage-aware signal construction and cross-sectional portfolio weights."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def momentum_signal(
    returns: pd.DataFrame,
    lookback_months: int = 12,
    skip_months: int = 1,
) -> pd.DataFrame:
    """Compound returns from t-lookback through t-skip-1 for month t.

    The default is the conventional 12-2 signal: month-t weights use returns
    from t-12 through t-2. The shift is deliberately ``skip_months + 1`` so
    neither the current return nor the immediately preceding month enters the
    signal.
    """

    if lookback_months <= skip_months:
        raise ValueError("lookback_months must exceed skip_months")
    if skip_months < 0:
        raise ValueError("skip_months must be non-negative")
    formation_length = lookback_months - skip_months
    lagged = returns.shift(skip_months + 1)
    return (1.0 + lagged).rolling(
        window=formation_length,
        min_periods=formation_length,
    ).apply(np.prod, raw=True) - 1.0


def cross_sectional_weights(
    signals: pd.DataFrame,
    selection_fraction: float = 0.20,
) -> pd.DataFrame:
    """Build equal-weighted, dollar-neutral winner-minus-loser weights."""

    if not 0 < selection_fraction < 0.5:
        raise ValueError("selection_fraction must lie strictly between 0 and 0.5")
    weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    for date, row in signals.iterrows():
        valid = row.dropna().sort_values(kind="mergesort")
        if valid.empty:
            continue
        count = math.ceil(len(valid) * selection_fraction)
        if 2 * count > len(valid):
            raise ValueError("Long and short selections would overlap")
        short_names = valid.index[:count]
        long_names = valid.index[-count:]
        weights.loc[date, short_names] = -1.0 / count
        weights.loc[date, long_names] = 1.0 / count
    return weights


def assert_temporal_integrity(
    returns: pd.DataFrame,
    lookback_months: int = 12,
    skip_months: int = 1,
    selection_fraction: float = 0.20,
) -> None:
    """Perturb one month and verify current/past signals and weights are unchanged."""

    if len(returns) < lookback_months + 6:
        raise ValueError("Need a longer sample for the temporal integrity check")
    pivot = len(returns) // 2
    original = momentum_signal(returns, lookback_months, skip_months)
    shocked = returns.copy()
    shocked.iloc[pivot] = shocked.iloc[pivot] + 7.0
    changed = momentum_signal(shocked, lookback_months, skip_months)
    pd.testing.assert_frame_equal(original.iloc[: pivot + 1], changed.iloc[: pivot + 1])
    original_weights = cross_sectional_weights(original, selection_fraction)
    changed_weights = cross_sectional_weights(changed, selection_fraction)
    pd.testing.assert_frame_equal(
        original_weights.iloc[: pivot + 1], changed_weights.iloc[: pivot + 1]
    )
