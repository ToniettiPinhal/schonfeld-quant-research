"""Performance statistics, block bootstrap, and Newey-West inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegressionResult:
    params: pd.Series
    standard_errors: pd.Series
    t_statistics: pd.Series
    r_squared: float
    n_observations: int

    def to_frame(self) -> pd.DataFrame:
        return pd.concat(
            [
                self.params.rename("coefficient"),
                self.standard_errors.rename("hac_standard_error"),
                self.t_statistics.rename("hac_t_stat"),
            ],
            axis=1,
        )


def max_drawdown(returns: pd.Series) -> float:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return float("nan")
    wealth = (1.0 + clean).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 12) -> float:
    clean = returns.dropna().astype(float)
    volatility = clean.std(ddof=1)
    if clean.empty or volatility == 0 or np.isnan(volatility):
        return float("nan")
    return float(np.sqrt(periods_per_year) * clean.mean() / volatility)


def performance_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    periods_per_year: int = 12,
) -> dict[str, float | int | str]:
    clean = returns.dropna().astype(float)
    if clean.empty:
        raise ValueError("Cannot summarize an empty return series")
    wealth = (1.0 + clean).cumprod()
    ending_wealth = float(wealth.iloc[-1])
    cagr = (
        ending_wealth ** (periods_per_year / len(clean)) - 1.0
        if ending_wealth > 0
        else float("nan")
    )
    result: dict[str, float | int | str] = {
        "start": clean.index.min().strftime("%Y-%m"),
        "end": clean.index.max().strftime("%Y-%m"),
        "months": int(len(clean)),
        "annualized_arithmetic_return": float(periods_per_year * clean.mean()),
        "annualized_volatility": float(np.sqrt(periods_per_year) * clean.std(ddof=1)),
        "annualized_sharpe": annualized_sharpe(clean, periods_per_year),
        "cagr": float(cagr),
        "max_drawdown": max_drawdown(clean),
        "best_month": float(clean.max()),
        "worst_month": float(clean.min()),
        "positive_month_fraction": float((clean > 0).mean()),
        "skewness": float(clean.skew()),
        "excess_kurtosis": float(clean.kurt()),
    }
    if turnover is not None:
        aligned_turnover = turnover.reindex(clean.index)
        if aligned_turnover.isna().any():
            raise ValueError("Turnover does not cover every return observation")
        result["annualized_traded_notional"] = float(
            periods_per_year * aligned_turnover.mean()
        )
    return result


def newey_west_ols(
    y: pd.Series,
    x: pd.DataFrame | None = None,
    lags: int = 6,
) -> RegressionResult:
    """OLS with Bartlett-kernel Newey-West standard errors.

    The implementation uses a fixed lag count and a finite-sample ``n/(n-k)``
    correction. It is intentionally compact and covered by unit tests so the
    inference does not depend on a black-box econometrics package.
    """

    if lags < 0:
        raise ValueError("lags must be non-negative")
    response_name = y.name or "y"
    regressors = pd.DataFrame(index=y.index) if x is None else x.copy()
    joined = pd.concat([y.rename(response_name), regressors], axis=1).dropna()
    response = joined.iloc[:, 0].to_numpy(dtype=float)
    regressor_names = ["intercept", *joined.columns[1:].astype(str).tolist()]
    design = np.column_stack(
        [np.ones(len(joined)), joined.iloc[:, 1:].to_numpy(dtype=float)]
    )
    n_obs, n_parameters = design.shape
    if n_obs <= n_parameters:
        raise ValueError("Not enough observations for the requested regression")

    gram_inverse = np.linalg.pinv(design.T @ design)
    coefficients = gram_inverse @ design.T @ response
    residuals = response - design @ coefficients
    score = design * residuals[:, None]
    meat = score.T @ score
    effective_lags = min(lags, n_obs - 1)
    for lag in range(1, effective_lags + 1):
        weight = 1.0 - lag / (effective_lags + 1.0)
        gamma = score[lag:].T @ score[:-lag]
        meat += weight * (gamma + gamma.T)
    covariance = gram_inverse @ meat @ gram_inverse
    covariance *= n_obs / (n_obs - n_parameters)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    t_statistics = np.divide(
        coefficients,
        standard_errors,
        out=np.full_like(coefficients, np.nan),
        where=standard_errors > 0,
    )
    total_sum_squares = float(((response - response.mean()) ** 2).sum())
    residual_sum_squares = float((residuals**2).sum())
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else float("nan")
    )
    index = pd.Index(regressor_names, name="term")
    return RegressionResult(
        params=pd.Series(coefficients, index=index, name="coefficient"),
        standard_errors=pd.Series(standard_errors, index=index, name="hac_standard_error"),
        t_statistics=pd.Series(t_statistics, index=index, name="hac_t_stat"),
        r_squared=r_squared,
        n_observations=n_obs,
    )


def hac_mean_t_stat(returns: pd.Series, lags: int = 6) -> float:
    return float(newey_west_ols(returns, x=None, lags=lags).t_statistics["intercept"])


def _circular_block_indices(
    sample_size: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    blocks = int(np.ceil(sample_size / block_length))
    starts = rng.integers(0, sample_size, size=blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % sample_size).ravel()[:sample_size]


def moving_block_sharpe_interval(
    returns: pd.Series,
    samples: int = 5_000,
    block_length: int = 12,
    seed: int = 20260812,
    periods_per_year: int = 12,
) -> dict[str, float | int]:
    clean = returns.dropna().to_numpy(dtype=float)
    if len(clean) < 2 * block_length:
        raise ValueError("Return history is too short for the requested block length")
    if samples <= 0 or block_length <= 0:
        raise ValueError("samples and block_length must be positive")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        draw = clean[_circular_block_indices(len(clean), block_length, rng)]
        estimates[index] = np.sqrt(periods_per_year) * draw.mean() / draw.std(ddof=1)
    lower, median, upper = np.quantile(estimates, [0.025, 0.5, 0.975])
    return {
        "estimate": annualized_sharpe(returns, periods_per_year),
        "bootstrap_median": float(median),
        "ci_2_5": float(lower),
        "ci_97_5": float(upper),
        "block_length_months": int(block_length),
        "bootstrap_samples": int(samples),
    }


def paired_block_sharpe_difference_interval(
    left: pd.Series,
    right: pd.Series,
    samples: int = 5_000,
    block_length: int = 12,
    seed: int = 20260812,
    periods_per_year: int = 12,
) -> dict[str, float | int]:
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    values = joined.to_numpy(dtype=float)
    if len(values) < 2 * block_length:
        raise ValueError("Paired history is too short for the requested block length")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        draw = values[_circular_block_indices(len(values), block_length, rng)]
        left_sharpe = np.sqrt(periods_per_year) * draw[:, 0].mean() / draw[:, 0].std(ddof=1)
        right_sharpe = np.sqrt(periods_per_year) * draw[:, 1].mean() / draw[:, 1].std(ddof=1)
        estimates[index] = left_sharpe - right_sharpe
    lower, median, upper = np.quantile(estimates, [0.025, 0.5, 0.975])
    estimate = annualized_sharpe(joined["left"], periods_per_year) - annualized_sharpe(
        joined["right"], periods_per_year
    )
    return {
        "estimate": float(estimate),
        "bootstrap_median": float(median),
        "ci_2_5": float(lower),
        "ci_97_5": float(upper),
        "block_length_months": int(block_length),
        "bootstrap_samples": int(samples),
    }
