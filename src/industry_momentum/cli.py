"""Command-line entry point for the reproducible research run."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import run_research
from .config import ResearchConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the post-2000 U.S. industry momentum study."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ResearchConfig(
        raw_data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        bootstrap_samples=args.bootstrap_samples,
    )
    output = run_research(config, force_download=args.force_download)
    row = output["summary"].loc[("test", "industry_momentum_net_10bps")]
    print(
        "Completed locked test through "
        f"{output['metadata']['last_strategy_month']}: "
        f"net annualized return={100 * row['annualized_arithmetic_return']:.2f}%, "
        f"Sharpe={row['annualized_sharpe']:.3f}."
    )
    print(f"Reports written to {args.reports_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
