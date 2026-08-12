from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from industry_momentum.features import (
    assert_temporal_integrity,
    cross_sectional_weights,
    momentum_signal,
)


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.date_range("2020-01-31", periods=20, freq="ME")

    def test_signal_uses_only_t_minus_4_through_t_minus_2(self) -> None:
        returns = pd.DataFrame({"x": np.arange(20) / 100.0}, index=self.index)
        signal = momentum_signal(returns, lookback_months=4, skip_months=1)
        date_position = 8
        expected = np.prod(1.0 + returns.iloc[4:7, 0]) - 1.0
        self.assertAlmostEqual(signal.iloc[date_position, 0], expected)

    def test_current_return_shock_cannot_change_current_signal(self) -> None:
        returns = pd.DataFrame(
            np.arange(100, dtype=float).reshape(20, 5) / 1_000.0,
            index=self.index,
            columns=list("abcde"),
        )
        assert_temporal_integrity(returns, lookback_months=4, skip_months=1)

    def test_cross_sectional_weights_are_dollar_neutral(self) -> None:
        signals = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0, 5.0]],
            index=[self.index[0]],
            columns=list("abcde"),
        )
        weights = cross_sectional_weights(signals, selection_fraction=0.20)
        self.assertEqual(weights.loc[self.index[0], "a"], -1.0)
        self.assertEqual(weights.loc[self.index[0], "e"], 1.0)
        self.assertAlmostEqual(weights.sum(axis=1).iloc[0], 0.0)
        self.assertAlmostEqual(weights.abs().sum(axis=1).iloc[0], 2.0)


if __name__ == "__main__":
    unittest.main()
