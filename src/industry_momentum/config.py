"""Configuration objects and locked temporal splits for the study."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters fixed before evaluating the post-2000 test sample.

    ``lookback_months=12`` and ``skip_months=1`` implement the conventional
    12-2 signal: compound returns from t-12 through t-2, inclusive. The return
    in t-1 is deliberately skipped and month-t returns are never used to form
    month-t weights.
    """

    lookback_months: int = 12
    skip_months: int = 1
    selection_fraction: float = 0.20
    cost_bps: float = 10.0

    def __post_init__(self) -> None:
        if self.lookback_months <= self.skip_months:
            raise ValueError("lookback_months must exceed skip_months")
        if self.skip_months < 0:
            raise ValueError("skip_months must be non-negative")
        if not 0 < self.selection_fraction < 0.5:
            raise ValueError("selection_fraction must lie strictly between 0 and 0.5")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")


@dataclass(frozen=True)
class ResearchConfig:
    """Paths, inference settings, and immutable date splits."""

    raw_data_dir: Path = Path("data/raw")
    reports_dir: Path = Path("reports")
    development_start: str = "1970-07"
    development_end: str = "1984-12"
    validation_start: str = "1985-01"
    validation_end: str = "1999-12"
    test_start: str = "2000-01"
    test_end: str = "2026-06"
    hac_lags: int = 6
    bootstrap_block_months: int = 12
    bootstrap_samples: int = 5_000
    random_seed: int = 20260812
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    @property
    def periods(self) -> dict[str, tuple[str, str]]:
        return {
            "development": (self.development_start, self.development_end),
            "validation": (self.validation_start, self.validation_end),
            "test": (self.test_start, self.test_end),
            "full": (self.development_start, self.test_end),
        }
