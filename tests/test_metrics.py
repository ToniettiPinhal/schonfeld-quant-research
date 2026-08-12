from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from industry_momentum.metrics import (
    max_drawdown,
    moving_block_sharpe_interval,
    newey_west_ols,
    performance_metrics,
)


class MetricTests(unittest.TestCase):
    def test_max_drawdown_uses_compounded_wealth(self) -> None:
        returns = pd.Series([0.10, -0.20, 0.05])
        self.assertAlmostEqual(max_drawdown(returns), -0.20)

    def test_performance_metrics_have_expected_annualized_mean(self) -> None:
        index = pd.date_range("2020-01-31", periods=12, freq="ME")
        returns = pd.Series([0.01] * 12, index=index)
        metrics = performance_metrics(returns)
        self.assertAlmostEqual(metrics["annualized_arithmetic_return"], 0.12)
        self.assertEqual(metrics["months"], 12)

    def test_ols_coefficients_match_exact_linear_relation(self) -> None:
        index = pd.RangeIndex(20)
        x = pd.DataFrame({"x": np.arange(20, dtype=float)}, index=index)
        y = pd.Series(1.5 + 2.0 * x["x"], index=index)
        result = newey_west_ols(y, x, lags=0)
        self.assertAlmostEqual(result.params["intercept"], 1.5, places=10)
        self.assertAlmostEqual(result.params["x"], 2.0, places=10)
        self.assertAlmostEqual(result.r_squared, 1.0, places=12)

    def test_block_bootstrap_is_deterministic(self) -> None:
        index = pd.date_range("2000-01-31", periods=120, freq="ME")
        returns = pd.Series(np.sin(np.arange(120)) / 100 + 0.002, index=index)
        left = moving_block_sharpe_interval(
            returns, samples=100, block_length=12, seed=7
        )
        right = moving_block_sharpe_interval(
            returns, samples=100, block_length=12, seed=7
        )
        self.assertEqual(left, right)
        self.assertLessEqual(left["ci_2_5"], left["ci_97_5"])


if __name__ == "__main__":
    unittest.main()
