from __future__ import annotations

import unittest

import pandas as pd

from industry_momentum.backtest import run_backtest, traded_notional


class BacktestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.date_range("2020-01-31", periods=2, freq="ME")
        self.columns = list("abcd")
        self.weights = pd.DataFrame(
            [
                [0.5, 0.5, -0.5, -0.5],
                [-0.5, 0.5, 0.5, -0.5],
            ],
            index=self.index,
            columns=self.columns,
        )

    def test_traded_notional_includes_initial_entry(self) -> None:
        turnover = traded_notional(self.weights)
        self.assertAlmostEqual(turnover.iloc[0], 2.0)
        self.assertAlmostEqual(turnover.iloc[1], 2.0)

    def test_cost_is_charged_on_absolute_weight_change(self) -> None:
        returns = pd.DataFrame(
            [[0.10, 0.00, -0.10, 0.00], [0.10, 0.00, -0.10, 0.00]],
            index=self.index,
            columns=self.columns,
        )
        signals = pd.DataFrame(0.0, index=self.index, columns=self.columns)
        result = run_backtest(returns, signals, self.weights, cost_bps=10.0)
        self.assertAlmostEqual(result.monthly["gross_return"].iloc[0], 0.10)
        self.assertAlmostEqual(result.monthly["transaction_cost"].iloc[0], 0.002)
        self.assertAlmostEqual(result.monthly["net_return"].iloc[0], 0.098)
        self.assertAlmostEqual(result.monthly["gross_return"].iloc[1], -0.10)

    def test_missing_return_in_active_position_fails(self) -> None:
        returns = pd.DataFrame(0.0, index=self.index, columns=self.columns)
        returns.iloc[0, 0] = float("nan")
        signals = pd.DataFrame(0.0, index=self.index, columns=self.columns)
        with self.assertRaises(ValueError):
            run_backtest(returns, signals, self.weights, cost_bps=0.0)


if __name__ == "__main__":
    unittest.main()
